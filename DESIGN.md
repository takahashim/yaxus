# yaxus 設計書

PLaMo 3 Translate を Rust で直接推論する翻訳サーバ/CLI。
llama.cpp や Python に依存せず、単一プロセスで動作し、HTTPS 経由で別マシンの CLI・ブラウザから利用する。

このファイルには **現在の設計** を書く。どう決めたか、何を試して採らなかったか、計測の記録は [HISTORY.md](HISTORY.md) (以下「経緯」。節番号はそのファイルのもの) にある。設計を変えたら、このファイルを書き換え、経緯には新しい節を足す。

---

## 1. 前提条件と調査結果

### 1.1 実行マシン

| 項目 | 値 |
|---|---|
| CPU / GPU | Ryzen 7 8745H / Radeon 780M (RDNA3, gfx1103, 内蔵 GPU) |
| メモリ | 60 GiB (GPU と共有。GTT 上限 約 31 GiB、専用 VRAM 2 GiB) |
| OS | Ubuntu 24.04.5, kernel 7.0 |
| GPU ドライバ | Mesa 25.2.8 RADV (Vulkan 1.4)。ROCm/HIP は未インストール |
| Rust | 1.97.1 |
| wgpu/Vulkan 確認済み | `shader_f16`、32 レーンの subgroup、cooperative matrix (16×16×16 f16)、整数ドット積、1 バッファ上限 2047 MiB |

GPU 経路は **wgpu + 手書きの WGSL コンピュートシェーダ**。Linux/Windows では Vulkan、macOS では Metal で動く (Metal は Apple M2 で確認、経緯 11.39)。Apple GPU では、行列積と小さなカーネルを MSL で書いたものに置き換える (4.2)。Vulkan を選んだ理由:

- ROCm は未導入で、gfx1103 は ROCm の公式サポート外 (動かすには `HSA_OVERRIDE_GFX_VERSION` などの回避が必要)。
- Mesa RADV は標準パッケージで動作確認済み。f16 演算と subgroup 命令が使えるので、推論カーネルを書くには十分。
- candle は公式には ROCm 非対応 (フォーク/PR 段階)。Burn は wgpu/ROCm 両対応だが、本プロジェクトは「Rust で直接書く」方針なので、カーネルを自前で持つ方が制御しやすい。

内蔵 GPU で、メモリは DDR5-5600 の 2 チャネル (理論値 89.6 GB/s、GPU の読み出しの実測上限は 82〜84 GB/s、経緯 11.27)。**デコードは帯域律速**で、速度はほぼ「重み総バイト数 ÷ 帯域」で決まる。GPU の価値はプリフィル (入力文の処理) の高速化と、CPU より高い帯域利用率にある。

注意: Linux ではユーザーが `render` グループに入っていて、そのセッションで有効になっている必要がある (入れた直後は再ログインするか `sg render -c '...'` で起動する)。そうでないと Vulkan から GPU が見えず、llvmpipe (CPU) しか見つからない。

### 1.2 モデル (pfnet/plamo-3-translate-{0.8b,2b,8b,31b})

ライセンス: PLaMo Community License (個人/非商用は無償、商用は売上規模に応じて登録または別途契約)。

アーキテクチャ (`model_type: plamo3`, `modeling_plamo.py` より):

| 項目 | 2B | 8B |
|---|---|---|
| hidden_size | 2560 | 4096 |
| layers | 24 | 32 |
| attention heads / KV heads | 24 / 4 | 32 / 4 |
| head_dim | 128 | 128 |
| intermediate_size | 10240 | 16384 |
| vocab_size | 107520 | 107520 |
| bf16 重みサイズ | 約 5.8 GB | 約 17 GB |

共通の構造:

- 密な Transformer (Mamba 等の SSM は無し)。
- 層ごとに attention 種別が変わる: `(layer_idx + 1) % 8 == 0` の層は **full attention**、それ以外は **sliding window attention (window=2048)**。
- サンドイッチ型 RMSNorm: `pre_mixer_norm`(offset 1.0) → attention → `post_mixer_norm`(offset 1/5) → 残差加算 → `pre_mlp_norm`(offset 1.0) → MLP → `post_mlp_norm`(offset 1/5^1.5) → 残差加算。RMSNorm は `(offset + weight) * normalize(x)` を f32 で計算。eps=1e-6。
- attention: 融合 `qkv_proj` (出力を q, k, v の順に split)、ヘッドごとの `q_norm`/`k_norm` (RMSNorm, head_dim, offset 1.0) を RoPE の前に適用、GQA (8 query heads が 1 KV head を共有)、`o_proj`。
- RoPE は half-split 形式 (interleave ではない)。sliding 層は theta=10000 の通常 RoPE。full 層は **YaRN** (theta=1e6, factor=256, original=4096, beta_fast=32, beta_slow=1) で、cos/sin に concentration `0.1*ln(256)+1 ≈ 1.5545` を乗じる。
- MLP: 融合 `gate_up_proj` を前半/後半に split し `silu(h0) * h1` → `down_proj`。
- 最終 `model.norm` (offset 1.0) → `lm_head` (embedding とは非共有)。
- 学習時の最大長は入出力合計 32,768 トークン。
- 参照実装 (`modeling_plamo.py`) は KV キャッシュに RoPE 前の K を積み、毎ステップ「キャッシュ内の相対位置 0..kv_len」で q/k に RoPE をかけ直す (NICT inference path)。RoPE のスコアは位置差だけに依存し、YaRN の concentration も一様な倍率なので、**RoPE 済みの K を絶対位置でキャッシュする実装と数学的に等価**。yaxus は後者を採用 (HF と logits 誤差 3e-5、greedy 完全一致を確認)。
- sliding window の可視範囲はクエリ位置 `p` に対してキー `(p-2048, p]` (`swa_mask`)。
- `scale_embedding: false` のため埋め込みのスケーリングは無い。

state dict のテンソル名:

```
model.embed_tokens.weight
model.layers.layers.{N}.pre_mixer_norm.weight
model.layers.layers.{N}.mixer.qkv_proj.weight
model.layers.layers.{N}.mixer.q_norm.weight
model.layers.layers.{N}.mixer.k_norm.weight
model.layers.layers.{N}.mixer.o_proj.weight
model.layers.layers.{N}.post_mixer_norm.weight
model.layers.layers.{N}.pre_mlp_norm.weight
model.layers.layers.{N}.mlp.gate_up_proj.weight
model.layers.layers.{N}.mlp.down_proj.weight
model.layers.layers.{N}.post_mlp_norm.weight
model.norm.weight
lm_head.weight
```

### 1.3 トークナイザ

HF `tokenizers` 互換ではなく、PFN 独自実装 (`tokenization_plamo.py`, 語彙は `tokenizer.jsonl`)。Rust で再実装が必要。

- `tokenizer.jsonl`: 1 行 1 エントリ `[token, score, type, ...]`。type に `BYTE` (`<0xNN>`) と `CONTROL` を含む。
- 分割アルゴリズム: Aho-Corasick トライ + 末尾から前方への Viterbi DP (スコア最大、同点なら最長一致)。マッチが無い文字は UTF-8 バイト列へフォールバック。
- 特殊トークン `<|plamo:...|>` は正規表現 `<\|plamo:[^|\s]{,64}\|>` で切り出し、境界文字 `` で囲んで分割境界を強制する。連続スペースや同一文字の繰り返しも境界になる。
- NFKC 等の正規化や prefix space は無い。
- デコードはバイトトークンを結合して UTF-8 復元、境界文字は除去。
- bos/eos はともに id 1 (`<|plamo:bos|>`)。pad は 3。

### 1.4 プロンプト形式 (PFML3)

```
<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation
<|plamo:tag|>context<|plamo:msg|>{参考情報 (任意)}
<|plamo:tag|>glossary<|plamo:index|>{原語}<|plamo:item|>{訳語}...   (任意、複数可)
<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>{source_lang}<|plamo:msg|>{原文}
<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>{target_lang}<|plamo:key|>form<|plamo:val|>{polite|plain}<|plamo:msg|>
```

- 改行は実際には無く、タグは連続して並ぶ (上は見やすさのため改行)。
- 過去の input/output ペアを前に並べると翻訳履歴として一貫性に使われる。
- `source_lang` は省略可 (自動判定)。`form` は target_lang=ja のときのみ。
- 言語コードは小文字 (`ja`, `en`, `zh-hans`, `zh-hant`, `ja-easy` など 53 言語)。
- 生成は **greedy (temperature 0)**、停止条件は `<|plamo:tag|>` または `<|plamo:bos|>` (EOS=1) の出現。
- 入力に `<|plamo:` を含めてはいけない (サーバ側で拒否する)。

---

## 2. 全体構成

```
                 HTTPS (rustls、ローカル CA で署名した証明書)
別マシン ──┬── ブラウザ ── GET /  (組み込み Web UI)
          └── yaxus client ── POST /v1/translate, /v1/translate/document (JSON / SSE)
                                   │
                ┌──────────────────▼──────────────────┐
                │ yaxus serve (1 プロセス)               │
                │  axum ルータ + 認証 + キュー (tokio)    │
                │  推論ワーカースレッド ×1                │
                │   ├ Session / 文書のチェーン / 翻訳メモリ │
                │   └ Engine (スケジューラ)               │
                │       ├ 本体モデル (2B など)            │
                │       ├ 下書きモデル (0.8B、任意)        │
                │       └ backend: wgpu (Vulkan/Metal)    │
                │                  または CPU             │
                └─────────────────────────────────────┘
```

単一プロセス。GPU を触るのは推論ワーカースレッド 1 本だけで、HTTP ハンドラはチャネル経由で依頼とイベントの受け取りだけを行う。複数のリクエストは、エンジンが同じパスにまとめて同時に進める (§3.4)。`yaxus translate` と `yaxus bench` は、サーバなしで同じエンジンを直接使う。

### 2.1 ワークスペース構成

```
yaxus/
  Cargo.toml                 # workspace
  DESIGN.md                  # 現在の設計 (このファイル)
  HISTORY.md                 # 開発の経緯
  crates/
    yaxus-api/             # サーバとクライアントが共有する JSON の型、HistoryWindow
    yaxus-tokenizer/       # PLaMo トークナイザ (pure Rust、依存最小)
    yaxus-pfml/            # PFML3 プロンプト (chat_template.jinja の移植) と翻訳リクエストの検証
    yaxus-model/           # 設定・safetensors / GGUF のロード・plamo3 の計算グラフ・Backend トレイト・KV アリーナ・エンジン
    yaxus-backend-cpu/     # 参照実装 (f32、rayon)。正しさの基準とフォールバック
    yaxus-backend-gpu/     # GPU バックエンド (wgpu で Vulkan / Metal、または Metal 直接。手書きの WGSL と MSL のカーネル)
    yaxus-runtime/         # モデル名の解決・読み込み (バックエンド、精度、下書き)・プロンプト・実行中の要求 (Session)・文書のチェーン・翻訳メモリ
    yaxus-server/          # axum + rustls、認証、推論ワーカー、Web UI の埋め込み
    yaxus-cli/             # バイナリ `yaxus` (serve / translate / bench / client)
    yaxus-web/             # ブラウザ用 (wasm32 + WebGPU)。native では空のライブラリ
  web/                       # Web UI (素の HTML/JS 1 ファイル。ビルド不要、include_str! で埋め込み)
  web/local/                 # ブラウザでモデルを動かす実験版 (wasm-bindgen の pkg と HTML)
  tests/golden/              # Python 側で生成した tokenizer / logits のゴールデンデータ
  tests/bench/               # ベンチの文セットと参照出力、Ollama との比較スクリプト
```

`yaxus-cli` の feature: `local` (`translate`、`bench`: `yaxus-runtime` だけを使う) と `server` (`serve` も。既定)。`--no-default-features` ならクライアントだけで、GPU 依存も入らない。

サーバとローカル実行で共通の部分 (読み込み、文書、翻訳メモリ) は `yaxus-runtime` に置く。`yaxus-server` は HTTP と推論スレッドへの受け渡しだけを持つ。

1 件の要求がエンジン上でたどる処理 (停止トークン、翻訳メモリのヒント、トークンから文字列への復号、終わったら出力をメモリへ) は `yaxus_runtime::Session<T>` の 1 か所にある。`T` はその要求の持ち主で、サーバのワーカーでは HTTP の接続 (単文) か文書の段落、`run_document` では段落の番号、`run_request` では無し。呼び出し側は `Session::step` が返す更新 (`Delta`、`Done`、`Failed`) を持ち主へ届けるだけ。

### 2.2 ブラウザ (wasm32 + WebGPU)

`crates/yaxus-web` は `yaxus-model` と `yaxus-backend-gpu` を wasm32 に載せ、ユーザーが選んだ GGUF をタブの中だけで動かす。サーバも CLI も使わない。Cloudflare などに置くのは静的ファイル (wasm 約 0.8 MB と HTML) だけで、モデルと計算はユーザーのマシンに残る。

- **範囲:** 英→日と日→英、greedy、f32、KV 4096 トークン。Engine、draft、文書チェーン、翻訳メモリ、同時実行は持たない。入口は `load_model(file, on_progress)` と `Translator::translate(text, target, form, on_text, on_progress)` の 2 つ。原文の言語は渡さない (CLI で `--from` を省いたときと同じプロンプト)。
- **Worker:** wasm はページのスレッドではなく専用 Worker (`web/local/worker.js`) で動く。読み込みも翻訳もページを止めない。ファイルは Worker が `File.stream()` で読み、`ModelBuffer` (wasm 側の `Vec<u8>`) へチャンクごとに足すので、メモリ上のコピーは wasm の 1 つだけ。読み込みのたびに新しい Worker を作り、前のモデルのメモリを返す。
- **モデル:** `Gguf::parse(Vec<u8>)` でメモリ上の GGUF を読み、`TensorRef` の実体を mmap か `Arc<Vec<u8>>` の enum にする。トークナイザは GGUF の語彙から `Tokenizer::from_vocab`。埋め込み表は Q4_K のまま GPU に置き (`embed_q4k`)、bf16 に展開しない (0.8B で 330 MB → 93 MB)。
- **デバイス:** `WgpuDev::open_async` が WebGPU の adapter/device を await で開く。`Tuning` は f32 固定。必要な `maxBufferSize` と `maxStorageBufferBindingSize` を要求し、受け入れられなければ WebGPU の基本の上限 (buffer 256 MB、binding 128 MB) で開き直す。要求する大きさは `GpuBackend::largest_weight_bytes` が、GPU に置く形式で見積もる。行列はファイルの大きさではなく、読み替えたあとの最大のバッファ (`ggml_largest_buffer`) で数える。binding はバッファごとなので、0.8B Q4_K_M の Q6_K 出力行列 (ファイルでは 135 MB) も最大のバッファは 124 MB で、128 MB に収まる。開いたデバイスの binding 上限を超えるバッファを持つ行列があれば、バックエンドの `load_weight` がその行列だけを Q4 で読む (形式を `Auto` / `Auto4` に任されたときだけ)。ページはその有無を `GpuBackend::reduced_to_q4` で知る。
- **push constants の代わり:** WebGPU に push constants はない。カーネルの `var<immediate> p` を `@group(0) @binding(15) var<uniform> p` に置き換え、1 MB の uniform バッファに 256 バイトずつ書いて dynamic offset で渡す。送信ごとにリングを先頭へ戻す。
- **読み戻し:** WebGPU の `Device::poll` は no-op なので、`map_async` の完了を `read_async` で await する。同期の `read` は wasm では呼ばない。argmax の 2 ディスパッチだけを記録する `argmax_prob_pending` を挟み、`GpuBackend::argmax_async` が読み戻す。prefill の時間は、プロンプトのパスから最初のトークンの読み戻しまで (記録しただけでは GPU はまだ計算していない)。
- **エラー:** WebGPU の error scope は pop が async で、読み込み中の同期の `Gpu::checked` からは待てない。そこで wasm では scope を使わず、デバイスの uncaptured-error と device-lost のコールバックが最初のエラーを `Fault` に残し、`read_async` が読み戻しの前後でそれを返す (失敗した map もエラーで、panic しない)。読み込みは `GpuBackend::watch_errors` の error scope で囲み、その pop を await する。pop はそれまでの操作のエラーがすべて分かってから解決するので、確保の失敗は必ず読み込みのエラーになる (`Fault` だけでは、uncaptured-error のイベントが読み戻しの完了より後に届くことがある)。翻訳も 1 要求全体を同じ error scope で囲み、結果を返す前に pop を await するので、推論中のエラーが遅れて届いても、次の翻訳ではなくその翻訳のエラーになる。入力の誤り (空の文、GGUF でないファイル) はモデルをそのまま使えるが、GPU のエラーと panic のあとは `Translator::broken` が真になり、ページは Worker を作り直して「読み込む」を促す。
- **長文:** `yaxus_pfml::segments::split_for_context` が、空行で段落に分け、1 段落が 768 入力トークンを超えるときは文末 (`.`、`!`、`?`、`。`、改行) で文に分け、さらに語・文字で切って 768 トークン以下に詰める。各セグメントは 1 要求として順に訳し、それまでの組を CLI の文書モードと同じ `HistoryWindow` (上限 1600 文字) で選んで履歴にする。プロンプトと出力の予算を 4096 に収めるのも CLI と同じ `yaxus_pfml::fit_prompt` で、予算が足りなければ履歴を捨てる。段落の切れ目は `\n\n`、同じ段落の文の間は、英語の出力ならスペース、日本語なら区切りなしで連結する (入力側でも、`。` などで終わる文には空白を挟まずに詰める)。`on_progress(done, total)` で進み具合を返す。
- **スレッド:** wasm では rayon は逐次に落ち、カーネルの並列コンパイル (`compile_all`) も逐次。時間は `web-time` で測る (std の `Instant` は wasm32-unknown-unknown で使えない)。
- **ビルド:** `web/local/build.sh` が `cargo build --target wasm32-unknown-unknown --release` と `wasm-bindgen --target web` を実行する。`web/local/serve.sh` で配る。`web/local/pkg` は生成物なので git に入れない。

---

## 3. 推論エンジン (`yaxus-model`)

### 3.1 行とパス

- **行** = (系列, 位置, トークン)。**パス** = 行の集合に対する 1 回の forward で、重みを 1 回だけ読む。
- 同じ系列の行はパス内で位置が連続する。各行は自分の系列の、自分の位置以下のキーだけを見る (sliding 層はさらに直近 2048)。別の系列の行は互いに独立。
- プリフィル、単発のデコード、投機の検証、複数リクエストの同時処理は、すべて「どの行を 1 パスに載せるか」の違いで、同じ経路を通る。
- 1 パスの時間は、行数が少ないうちはほとんど増えない (デコードは重みの読み出しで律速される)。2B の 1 行と 16 行で 35 ms と 47 ms (780M、int16)。速さの大半は、1 回の読み出しで進む行を増やすことから来る。

### 3.2 Backend トレイト

演算単位のトレイトで、`yaxus-model` は plamo3 の forward をこのトレイトだけで書く。CPU と GPU が同じ経路を通るので、CPU 実装を基準に GPU 実装を検証できる。

- 型: `Tensor` (行優先 `[rows, cols]` の f32 活性値)、`Weight` (行列またはベクトル)、`Kv` (全層のページ式 KV)。
- 読み込み: `load_weight(TensorRef, Quant)`、`load_embedding`、`reserve_activations` (重みより先に活性値のバッファを確保させる。780M ではこれで GEMM が速い、経緯 11.14)。
- KV: `new_kv(pages, max_pos)`、`copy_page`。
- パス: `begin_pass(PassMeta)` で行のメタデータ (トークン、位置、KV スロット、ページ表) を渡し、`embed` と `attention` がそれを読む。
- 演算: `rmsnorm`、`matmul`、`attention` (qkv の分割、q/k の RMSNorm と RoPE、KV への書き込み、因果 + sliding window の GQA)、`swiglu`、`add_assign`、`add_norm` (残差加算 + post-norm + 次の pre-norm。融合してよい)、`matmul_glu` (gate/up の行列積 + SwiGLU。融合してよい)、`select_rows`。
- 同期するのは `argmax`、`argmax_rows`、`to_host` だけで、ほかの演算は非同期に記録してよい。

### 3.3 KV アリーナとプレフィックスの再利用

- KV は層ごとに K と V のバッファを 1 つずつ持ち、128 トークンのページに分ける。ページは参照カウント付きのプールで管理し、系列 (`SeqId`) はページ表と長さを持つ。位置は絶対位置で、RoPE 済みの K を置く (リングバッファではない)。
- 既定のアリーナは 16384 トークン分 (`--kv-tokens`)、1 リクエストのプロンプトと生成の合計は 8192 トークンまで (`--max-context`)。KV は f32 (f16 化は試して取り消した、経緯 11.20)。
- 終わった系列は、消費したトークン列ごと LRU で保持する (最大 32 系列、ページが足りなければ古い順に解放)。新しいプロンプトとの最長共通接頭辞が 16 トークン以上ならフォークして (完全なページは共有、末尾の半端なページはコピー)、残りだけをプリフィルする。投機の巻き戻しは `truncate`、共有ページへの書き込みはコピーオンライト。
- API は状態を持たない。クライアントは履歴を毎回全部送り、同じ文書を訳し進めると前の段落までの KV がそのまま再利用される。これを活かすため、クライアントの履歴 (`HistoryWindow`) は開始位置を固定し、上限 (既定 4000 文字) を超えたときだけ最新の組に切り詰める (経緯 11.15)。
- `--no-kv-cache` で再利用をやめる (応答キャッシュも止まる)。

### 3.4 スケジューラ (`Engine`)

`Engine` が複数のリクエストを `step()` ごとに 1 パスずつ進める。生成は greedy だけで、どのリクエストの出力も、ほかに何が動いていても本体モデル単独の greedy 出力と同じになる。

- **プリフィル**: 待っているプロンプトを古い順にまとめて 1 パスにする。空いていれば 512 行まで、デコード中のリクエストがあれば 64 行までに抑えて、デコードと交互に進める。
- **デコード**: デコード中のリクエストを、1 パス最大 `--decode-rows` 本 (整数の GEMV なら既定 16、f32 なら 8) ずつ総当たりで 1 トークン進める。同時に受け付けるのは `--max-streams` (既定 16) まで。
- **投機的デコード**: デコード中が 1 本だけで下書きモデルがあれば、投機のラウンドで進める。下書きが最大 `--draft-k` (既定 7) トークンを出し、本体が「直前の確定トークン + 下書き」を 1 パスで検証して、一致する最長の接頭辞と最初の不一致位置の本体の予測を採る。下書きは本体の消費済みトークン列から追いつくので、投機と通常のデコードはいつでも切り替えられる。
- **下書きの長さの自動調整**: ラウンドごとに k (0 なら通常の 1 ステップ) を選ぶ。本体・下書き・行数ごとの検証の時間を指数移動平均で測り、当たり率 (1 番目と、2 番目以降の条件付き) と合わせて、1 秒あたりの期待トークン数が最大の k を選ぶ (k=5 より 5% 以上よいときだけ切り替える、経緯 11.30)。
- **ヒント**: リクエストは「出力に現れそうなトークン列」(前回の訳) を持てる。直近の出力の末尾がヒントと一致する位置から、続く最大 7 トークンを下書きと同じように 1 パスで検証する (下書きモデルは動かさない)。出力がヒントから外れたら、何回か 1 行のデコードで再合流を待ち、それでも外れたままなら下書きモデルに切り替える。
- **応答キャッシュ**: 終わったリクエストの出力を、プロンプトと停止トークンをキーに LRU で 256 件保持し、同じリクエストにはパスを回さずに答える (`--no-response-cache`)。

### 3.5 文書と翻訳メモリ (`yaxus-runtime`)

- **文書** (`DocJob`): 連続する段落の **チェーン** に分ける (既定はチェーン数 = 段落数 / 4、最大 8)。チェーン内は前の段落を履歴にして順に訳し、チェーン同士はエンジン上で並列に進める。空の段落は空の訳で終え、履歴にも入れない。チェーン数を段落数以上、`history_chars: 0` にすると、履歴なしの完全な並列になる (速いが訳の一貫性が落ちる、経緯 11.18)。
- サーバのワーカーと `run_document` (ローカル) は同じ `DocJob` と `Session` を使い、段落が終わるたびに同じチェーンの次の段落を投入する。
- **翻訳メモリ**: 原文・対象言語・原文の言語・文体をキーに、前回の出力トークンを 4096 件保持し (履歴、context、用語集はキーに含めない)、`Session` が各リクエストのヒントとして渡す。文書の途中の段落を直して訳し直すと、後続の段落はプロンプト (履歴) が変わって応答キャッシュには当たらないが、ヒントがほぼそのまま当たる。

### 3.6 重みの形式

| 形式 | 中身 | 由来 |
|---|---|---|
| `Q8` | int8、32 個ごとに f16 スケール | safetensors の既定 (ロード時に量子化)、GGUF の Q8_0、そのまま読めない GGUF の型 |
| `Q4` | 4 ビット、32 個ごとに f16 スケール | `--quant q4`、GGUF の Q4_0、下書きの `--draft-quant q4` |
| `Q4K` | ggml の Q4_K と同じ 144 バイトのブロック | GGUF の Q4_K |
| `Q6` | Q6_K の値、16 個ごとに int8 と 256 個ごとに f16 のスケール。GEMM 用に 16 個ごとの f16 スケールも持つ | GGUF の Q6_K |
| `Q3`、`Q2` | Q3_K / Q2_K のまま | GGUF の Q3_K / Q2_K (下書きの既定は 0.8B の Q2_K) |
| `Bf16` | bf16 を u32 に 2 個詰めたまま | `--quant bf16` (safetensors で無損失) |

- `--quant auto` (既定) は、GGUF なら上の対応でファイルの形式に近いまま持ち、safetensors なら Q8。CPU バックエンドは常に Q8 か bf16。
- GGUF は設定 (`plamo3.*`) とトークナイザ (`tokenizer.ggml.*`) も読むので、HF のディレクトリは要らない。llama.cpp の変換器が RMSNorm の重みに足したオフセットは、読み込み時に引き戻す。
- 埋め込み表は、GGUF の Q4_K の表なら 4 ビットのまま (`embed_q4k`) 読み、ほかは bf16 で GPU に置く。0.8B で 93 MB (bf16 では 330 MB で、WebGPU のバッファ上限にも触れる)。native もブラウザも同じなので、両者の訳は一致する。行の値は Q4_K を厳密に戻したもので、bf16 に丸めない。
- 1 行のスケールの並びが 2 のべきの間隔になると GEMM が遅くなるので、該当する行には詰め物を入れる (`pad_scales`、経緯 11.22)。

---

## 4. GPU バックエンド (`yaxus-backend-gpu`)

### 4.1 構成

- `Gpu` がデバイス、カーネル、バッファプールを持つ。本体と下書きは `GpuBackend` ハンドル (`Arc<Mutex<Gpu>>`) で同じ `Gpu` を共有し、2 つのモデルの命令が 1 回の投入に混ざる。
- `dev/`: デバイス層。バッファ、書き込み、コピー、ディスパッチ、投入、読み戻し、カーネルのコンパイルを、2 つの実装で同じ形にする。`dev::wgpu` は wgpu (Vulkan、または wgpu を通した Metal)、`dev::metal` は objc2-metal で Metal を直接使う (macOS、`YAXUS_DEVICE=metal`)。直接の場合、バッファは共有ストレージで、WGSL のカーネルは naga で MSL に変換し、fast math を切ってコンパイルする。既定はまだ wgpu (速さは同じ)。wasm32 では `dev::wgpu` の WebGPU だけを使い、オープンと読み戻しが async になる (§2.2)。
- `msl/`: Apple GPU 用に MSL で書いたカーネル。`access` (形式ごとの重みの展開)、`gemv2` (1〜16 行の GEMV)、`gemm` (prefill の GEMM)、`small` (`rmsnorm`・`add_norm`・`qk_rope`)。カーネルは値 (`msl::Gemv2 { t, glu, tiles }`、`msl::Gemm { f16, rb }`) で表し、名前は `kernels::name`、ソースは値から作る。wgpu ではパススルーで、Metal 直接ではそのままコンパイルする。ソースの先頭行 `// msl workgroup_size=N` で WGSL と見分ける。
- `dev/metal/translate.rs`: Metal 直接のときに、WGSL のカーネルを naga で MSL に変換する。immediates を `buffer(0)`、binding `i` を `buffer(i + 1)` に置き、MSL のカーネルと同じ形でディスパッチできるようにする (単体テストで固定)。
- `device.rs`:
  - `DeviceCaps`: アダプタから調べる機能 (バックエンド、ベンダー、f16、32 レーン以上の subgroup、16×16×16 の cooperative matrix、fast math、タイムスタンプ、workgroup メモリの大きさ)。
  - `CompileSettings`: naga と wgpu が足す安全装置のうち外すもの。ループのカウンタはすべてのバックエンドで外し、workgroup メモリのゼロ初期化は Metal だけで外す (経緯 11.39、11.40)。外す前提 (どのループも必ず終わる) は `every_kernel_loop_ends` が全カーネルのソースで検査する。範囲チェックは残す。
  - `DeviceCaps` は wgpu の型を持たない (`Api`、`DeviceKind`)。wgpu のアダプタからの検出は `dev::wgpu` にある。
  - `Tuning`: 速さだけで決まる選択 (投入の FLOP 閾値、早期投入、GEMV と GEMM を切り替える行数、attention の分割など)。780M で測った値が基準で、ほかのデバイスは `for_device` で上書きする (Apple GPU は f32 のカーネル)。MSL のカーネルのスイッチと大きさは `Tuning::apple` (`AppleTuning`) にまとめる。環境変数での上書きは持たない。テストは `GpuBackend` の setter (`set_gemm`、`set_gemv`、`set_tiled_attention`、`set_msl_small`) で切り替える。
- `plan.rs`: 形状、`DeviceCaps`、`Tuning` だけから、行列積 (`MatmulPlan`、`GluPlan`) と attention (`AttentionPlan`) に使うカーネルを決める。MSL のカーネルも同じで、パスを行で分ける形 (`gemv2_pieces`、`gemm_msl_pieces`) も plan の関数。`ops/` は plan を実行するだけ。事前コンパイルも同じ plan の関数とカーネルの名前を使うので、実行するカーネルと食い違わない。
- `kernels.rs`: カーネルの名前 (`name::*`)、ソースの一覧 (`base_sources`、`int_jobs`)、コンパイル。基本のカーネルはデバイスを開いたときに、整数カーネルはモデルを読み込んだあと全コアで事前コンパイルする (`precompile`)。それ以外の行数の整数カーネルは初回の使用時にコンパイルする。
- `weights.rs` は重みの形式とバッファ、`commands.rs` は記録・投入・読み戻しとバインドグループのキャッシュ、`profile.rs` は `YAXUS_PROFILE` の集計。
- カーネル固有の引数は push constant (wgpu の `IMMEDIATES`) で渡す。RoPE の cos/sin は CPU で表を作って渡す。

### 4.2 カーネル

**行列積** (`MatmulPlan`):

| 行数 | 整数 (AMD の既定) | f32 (`--prefill-precision f32 --decode-precision f32`) | Apple GPU の既定 |
|---|---|---|---|
| 1〜16 | 整数 GEMV `gemv_int` (活性値を `xquantr` で int8 の 2 プレーンに丸め、`dot4I8Packed`。行数ごとに生成。SwiGLU 融合版あり) | 1 行は `gemv1` (SwiGLU 融合の `glu`)、2〜16 行は複数行 GEMV `gemvm` (subgroup なしは `gemv8`) | `msl::gemv2` (1〜2 行はスカラー、3〜16 行は 8×8 の simdgroup 行列。算術は同じ) |
| 17〜48 | 整数 GEMV を 16 行ずつ | 17〜20 行は `gemvm` を 2 回、それより上は f32 GEMM | `msl::gemm` (下記) |
| 49〜 | 整数 GEMM `gemm_int` (活性値を `xquant` で丸め、128 トークンのタイル、端は 64 か 32) | f32 GEMM (64×128 タイル) | `msl::gemm` |

- prefill と decode の精度は別々に選べる。16 行以下は decode の、それより上は prefill の設定に従う (decode だけ f32 なら 17 行以上は整数 GEMM)。
- 整数の経路は、活性値を 32 個ごとに `max|x|` で正規化した 15 bit + 符号 (int16、`128·hi + lo` の 2 プレーン) で計算する。`int8` (1 プレーン、llama.cpp と同じ 7 bit) も選べる。
- 整数の GEMV では、行の結果は同じパスの他の行に依存せず、ビット単位で同じになる (`integer_rows_do_not_depend_on_the_pass`)。fast math でコンパイルされる Metal では、1 行も `xquant` を通して丸めることでこれを保つ (`plan::gemv_int_inline`)。
- f16 の行列演算ユニット (cooperative matrix) の GEMM `gemm_coop` も残している (`--prefill-precision f16`)。
- Apple GPU の `gemv2` は Q8、Q4_K、Q6_K、Q3_K、Q2_K を読む。K を 32 値の区間に切り、区間 `c` をスライス `c % 8` に割り当てる。スライスごとに 4 本のアキュムレータを `fma` で進め、最後に決まった木の形で足す。M2 では 8×8 の行列演算が k の順の `fma` の鎖とビット単位で一致するので、スカラーと行列の 2 つのカーネルで同じ値になる (経緯 12.8)。起動時に 1 行・8 行・16 行のパスを比べ (`Gpu::gemv2_rows_agree`)、違えば `gemv_f32` (同じ性質を持つ WGSL の f32 GEMV) に戻す。ほかの形式 (Q4_0、bf16) も `gemv_f32` を使う。
- Apple GPU の prefill は `msl::gemm` (llama.cpp の `kernel_mul_mm` と同じ形)。重み 64 行 × パスの 64 行のタイルで、threadgroup メモリに 8×8 の塊を置く。既定は f32 のタイル (何も丸めない) で、`--prefill-precision f16` で重みと活性値を half に丸める (`GemmKind::CoopF16`)。最後の 64 行タイルの後の残りは、16 行以下なら `gemv2`、32 行以下なら 32 行のタイルで計算する。MSL 版が読まない形式は WGSL の `gemm_coop8` (8×8 の f32 行列)。
- bf16 の重みは f32 のカーネルだけで読む。
- **WebGPU**: subgroup を要求しないので、行列積は `gemv1` / `gemv8` (subgroup なしの workgroup リダクション) と `gemm_f32`、attention は行ごとの `attention` になる。整数ドット積と cooperative matrix は WebGPU にない。lavapipe の CI と同じ経路。

**attention** (`AttentionPlan`):

- 16 行以下のパス: 分割版 (flash-decoding)。(行, KV ヘッド, 16 または 32 キーのチャンク) ごとに 1 ワークグループで、KV ヘッドを共有するクエリヘッドをまとめて処理し、`attn_combine` で合成する。
- それより大きいパス: タイル版 `attn_tile`。同じ系列で位置が連続する行を 32 行のタイルにまとめ、K/V を LDS に載せてタイルの全行で共有する。ワークグループが少ないときはキーをチャンクに分けて合成する。workgroup メモリが 38 KB に満たないデバイス (Apple GPU の 32 KiB) では、クエリを LDS に置かない版を使う。
- subgroup がない、またはタイル版が使えないとき: 行ごとの `attention`。
- 計算はすべて f32。

**その他**: `embed`、`qk_rope`、`rmsnorm`、`add_norm` / `add_norm_sg`、`swiglu`、`add`、2 段の `argmax1` / `argmax2`。Apple GPU では `rmsnorm`、`add_norm`、`qk_rope` を MSL 版 (`*_msl`) で動かす。

### 4.3 投入

- APU では計算ジョブもグラフィックスのリングで動き、1 ジョブが約 2 秒を超えるとカーネルが GPU をリセットする。行列積の FLOP を積算し、32 GFLOP (780M で約 40 ms) ごとに投入を分ける。デスクトップのカクつき防止も兼ねる。
- 8 行以下の小さいパスは、最初の約 1 層 (12 ディスパッチ) を記録した時点で一度投入し、残りの記録と GPU の実行を重ねる (経緯 11.29)。
- 読み込み中は 512 MB ごとに投入して待つ (一時バッファが溜まってメモリが倍にならないように)。

### 4.4 デバイスごとの違い

- **Radeon 780M (RADV)**: 基準のデバイス。整数ドット積、32 レーンの subgroup、cooperative matrix がそろう。
- **Apple GPU (Metal)**:
  - Metal は workgroup メモリの vec4 の成分を別々のレーンから書くと値が消えるので、カーネルは workgroup メモリに vec4 をまるごと書く。
  - wgpu は WGSL を fast math でコンパイルし、止められない (`DeviceCaps::fast_math`)。整数 GEMV の積算は明示的な `fma`、スケールは `keep` で包んで、行の結果がパスの大きさによらないようにしている。MSL のカーネルは `math_mode(safe)` で、Metal 直接のときは全カーネルを fast math なしでコンパイルする。
  - 整数ドット積の命令がないので、既定は prefill も decode も f32。行列積は MSL のカーネル (4.2) で、cooperative matrix は 8×8 だけなので `gemm_coop` (16×16) は使えない。
  - llama.cpp b11972 との比較 (M2、19 件の翻訳): 2B は下書きありで 1 件あたり 4〜20% 速く、下書きなしでは 5〜12% 遅い。0.8B はほぼ同じ (経緯 12.11)。
- **ソフトウェア (lavapipe)**: 32 レーンの subgroup がないので、f32 の単純なカーネルだけで動く。CI のカーネルテストで使う (`YAXUS_SOFTWARE_GPU=1`)。

---

## 5. 精度と再現性

- 既定 (`auto`) はデバイスが選ぶ。AMD では prefill・decode とも int16、Apple GPU では f32。Apple GPU で `--prefill-precision f16` を選ぶと、prefill の GEMM が重みと活性値を half に丸める (llama.cpp と同じ。256 行で約 9% 速い)。
- `--prefill-precision int16|int8|f16|f32` と `--decode-precision int16|f32` で明示できる。int16 の logits は f32 と 5e-3 程度違う。int8 はその約 100 倍動く。
- 行の結果は、同じパスに他の行があってもなくても同じ (整数 GEMV と、Apple GPU の `gemv2`・`gemv_f32` はビット単位、ほかの f32 は 1e-5 程度)。そのため、同時実行や投機の有無では出力が変わらない。
- KV の再利用や同時実行では、同じリクエストでも KV が「デコードで作られた値」か「プリフィルで作られた値」かが変わり得る。差は上の範囲で、greedy の結果が変わるのは上位 2 候補がほぼ同点のときだけ。
- 完全に再現したいとき (テスト、CPU の f32 参照との比較) は `--prefill-precision f32 --decode-precision f32 --no-kv-cache --max-streams 1`。
- 投機の下書きの精度は出力に影響しない (採用率が変わるだけ)。

---

## 6. サーバ (`yaxus-server`)

### 6.1 HTTPS と認証

- axum + `axum-server` (rustls)。推論スレッド 1 本と tokio。
- 証明書は `--cert/--key` で指定するか、未指定なら初回起動時に `~/.config/yaxus/ca.pem` (ローカル CA) を作り、それで署名したサーバ証明書を発行する。SAN は localhost / 127.0.0.1 / ::1 / ホスト名 / `ホスト名.local` / 主 IP と `--san` で足したもの。SAN が変わればサーバ証明書だけ再発行するので、クライアントが信頼する `ca.pem` は変わらない。秘密鍵とトークンは 0600。
- LAN 公開を前提に、翻訳の API は **Bearer トークン認証が必須**。`--token` / `YAXUS_TOKEN`、未指定なら初回に 192 bit の乱数を作って `~/.config/yaxus/token` に保存する。比較は定数時間。
- `/`、`/healthz`、`/v1/languages` は認証なし。

### 6.2 API

```
GET  /                         Web UI (単一 HTML)
GET  /healthz                  モデル名、バックエンド、キュー長 (推論スレッドが止まっていれば 503)
GET  /v1/languages             対応言語コード一覧
POST /v1/translate             1 段落
POST /v1/translate/document    文書 (段落の列)
```

`POST /v1/translate` リクエスト:

```json
{
  "text": "会議は来週の月曜日に変更されました。",
  "target_lang": "en",
  "source_lang": "ja",            // 任意
  "form": "polite",               // 任意、target_lang=ja のみ
  "context": "社内連絡メール",     // 任意
  "glossary": [{"source": "owner", "target": "オーナー"}],   // 任意
  "history": [{"source": "...", "target": "..."}],          // 任意、件数の制限なし
  "max_new_tokens": 512,
  "stream": true
}
```

- `form` は `polite` / `plain` のどちらか (それ以外は解析の時点で 400)。省略は「指定なし」で、空文字列は受け付けない。
- `stream: false` → `{"text": "...", "usage": {...}}`。`usage` は `prompt_tokens`、`reused_tokens`、`completion_tokens`、`queue_ms`、`prefill_ms`、`elapsed_ms`。
- `stream: true` → `text/event-stream`。各 `data:` は `type` で種類を表す: `{"type":"delta","text":"..."}` を逐次送り、最後に `{"type":"done","usage":{...}}` か `{"type":"error","error":{...}}`。
- 文書 (`POST /v1/translate/document`) は `segments`、共通のオプション、`chains`、`history_chars`、`neighbor_context`、`stream` を受け取る。各段落は受付時にテンプレートで検証し、不正なら 400。イベントは `{"type":"delta","segment":i,"text":".."}`、`{"type":"segment_done","segment":i}`、最後に `done` (`usage` は上の項目に `segments` と `chains` を足したもの) か `error`。
- エラーは本文もストリームも `{"error": {"code": "...", "message": "...", "segment": i}}` の形 (`segment` は文書の段落を指すときだけ)。`code` と HTTP ステータスは 1 対 1: `invalid_request` / `prompt_too_long` (400)、`unauthorized` (401)、`queue_full` (429)、`unavailable` (503、推論スレッドの停止)、`internal` (500、生成の失敗)。
- 入力に `<|plamo:` を含む場合は 400。言語コードは chat_template と同じ正規表現で検証。未知の JSON フィールドも 400。
- 推論スレッドへの依頼はキュー (`--queue`、既定 16。エンジン側でも同じ数まで待てる) で待ち、あふれたら 429。SSE はクライアントの切断を検知して生成を取り消す。

### 6.3 Web UI

- 依存なしの単一 HTML (textarea 2 枚、言語セレクタ、form/glossary の折り畳み)。ビルド工程を持たず `include_str!` で埋め込むので、バイナリ 1 つで配布できる。
- 文書 API を使い、複数の段落が並行して埋まっていく。トークンは localStorage に保持する。

---

## 7. CLI

```
yaxus serve      [--model 2b] [--backend auto|gpu|cpu] [--listen 0.0.0.0:8443] [--token ...] [--san ...]
yaxus translate  --to en [--from ja] [--form polite] "テキスト"     # サーバを立てずローカルで実行
yaxus client     --server https://host:8443 --to en < input.txt   # 標準入力を段落に分けて文書 API へ
yaxus bench      rows | decode | set | streams | doc               # 計測 (§8.2)
```

- モデルは `--model` に HF のディレクトリ、GGUF ファイル、ローカルの HF キャッシュにあるリポジトリ、短縮名 (`0.8b`、`2b`、`8b`、`31b`)、`REPO:QUANT` / `SIZE:QUANT` (例 `2b:Q4_K_M`) で指定する。ダウンロードはしない (HF キャッシュを読む)。
- 下書きは `--draft auto` (既定。HF キャッシュの 0.8B から Q2_K の GGUF、safetensors、Q4_K_M の順に探す)、`--draft none` で無効。CPU バックエンドでは使わない。
- `translate` と `client` は段落数によらず文書の経路を使い (`--chains`、`--history-chars`、`--neighbor-context`)、並列に届く段落を順番どおりに表示する。
- `--backend auto` は GPU (Vulkan、macOS では Metal) があれば使い、無ければ CPU。llvmpipe などソフトウェアのアダプタは選ばない。

---

## 8. 検証と計測

### 8.1 テスト

```
YAXUS_REQUIRE_GOLDEN=1 cargo test --workspace                                  # 単体テスト、トークナイザのゴールデン
cargo test --release -p yaxus-backend-cpu -- --ignored                         # HF の logits と比較 (2B)
cargo test --release -p yaxus-backend-gpu --test kernels -- --ignored         # カーネルをホスト参照と比較
cargo test --release -p yaxus-backend-gpu --test vs_cpu -- --ignored --test-threads=1   # GPU と CPU の比較 (2B)
```

- **トークナイザ**: Python の `Plamo3Tokenizer` で作ったゴールデン (パブリックドメイン作品の抜粋を含む約 1,700 件) の encode/decode、プロンプトテンプレート、ストリーミングデコードが完全一致。GGUF のトークナイザも同じ結果になる。
- **モデル (CPU)**: transformers (f32) の logits と比べ、最大誤差 3e-5、greedy 一致。0.8B と 8B のゴールデンも持つ (`YAXUS_MODEL_DIR`、`YAXUS_GOLDEN`)。
- **カーネル**: 全形式 × 全経路 (1〜16 行の GEMV、GEMM、GLU) をホストの参照と比べる。整数 GEMV と、Apple GPU の f32 GEMV (`gemv2`、`gemv_f32`) の行が、パスによらずビット単位で同じことも確かめる (`f32_rows_do_not_depend_on_the_pass`)。`YAXUS_DEVICE=metal` で、Metal 直接のデバイスでも同じテストを回せる。
- **GPU と CPU**: 同じ入力の logits、分割・タイルの attention、同時実行と投機とヒントが単独実行と同じ出力になること。
- **ソースの検査**: 全カーネルのループが必ず終わる形であること (`every_kernel_loop_ends`)。

ゴールデンデータは `tests/golden/gen_*.py` で作り直す (transformers + numba + torch(CPU) が要る)。

### 8.2 ベンチ

- `yaxus bench rows`: 1 パスの時間と行数の関係。
- `yaxus bench set --compare tests/bench/ref_default.txt`: 19 件の文セットを訳し、参照の出力と比べる (再現性の設定では `ref_f32.txt`)。カーネルを変えたら、出力が 19/19 一致することを確かめる。
- `yaxus bench doc`: モデル同梱の `README_ja.md` から散文 14 段落を取り出して訳す文書シナリオ (`--edit N` で再翻訳)。
- `yaxus bench streams N`: 同時リクエスト。
- `tests/bench/compare/compare.py`: Ollama との比較。
- `YAXUS_PROFILE=1|2` で記録と待ちの時間、カーネルごとの GPU 時間を出す (Metal 直接では、2 のとき dispatch ごとにコマンドバッファを分けて測る)。
- llama.cpp との比較は、同じ GGUF で `llama-bench` (pp、tg) と `llama-server` (19 件、同じプロンプト書式) を使い、交互に測った (経緯 12.10、12.11)。

### 8.3 CI (`.github/workflows/ci.yml`)

- fmt、clippy (`-D warnings`、クライアントだけの構成も)、テスト。
- lavapipe でのカーネルテスト (`YAXUS_SOFTWARE_GPU=1`)。
- macOS (Metal) でのカーネルテスト。デバイスの機能を出力し、32 レーンの subgroup がなければ失敗する。
- 0.8B のモデルを使う HF との比較 (`model.yml`)。

---

## 9. 主要クレート

| 用途 | クレート |
|---|---|
| GPU | `wgpu` 30 (Linux/Windows は Vulkan、macOS は Metal。実験的機能 cooperative matrix を使用), `pollster`, `bytemuck`, `half`。macOS では `objc2`、`objc2-metal`、`objc2-foundation` と `naga` (Metal 直接のデバイス) |
| CPU 並列 | `rayon` |
| 重み | `safetensors`, `memmap2` (GGUF は自前で読む) |
| HTTP | `axum` 0.8, `axum-server` (rustls), `tokio` |
| TLS | `rustls`, `rcgen` |
| CLI | `clap`, `reqwest` (rustls-tls, stream) |
| 共通 | `serde`, `serde_json`, `anyhow`, `thiserror`, `tracing` |

---

## 10. 決定事項と残課題

決定:

- GPU 経路は wgpu で自前カーネル (ROCm も MLX も使わない)。Metal は wgpu の上で対応し、Apple GPU 向けのカーネルは MSL で書く。Metal を直接使うデバイス (`YAXUS_DEVICE=metal`) も同じ ops とカーネルで動く。
- Apple GPU の prefill の既定は f32 のタイル。f16 (llama.cpp と同じ丸め) は `--prefill-precision f16` で選ぶ (経緯 12.9)。
- 既定モデルは 2B。safetensors の既定の重み形式は Q8、GGUF はファイルの形式のまま。
- 生成は greedy 固定 (サンプリングの API は持たない)。訳文は、精度の設定が同じなら同時実行・投機・KV 再利用の有無によらず同じ。
- 認証はトークン必須、TLS はローカル CA を自動生成。
- 長文は段落単位で分け、文書のチェーンで並列に訳す (既定は自動のチェーン数 + チェーン内の履歴)。
- ブラウザ版は wasm32 + WebGPU で、GPU バックエンドの f32 経路を再利用する (§2.2)。Engine は使わず、1 要求・greedy・KV 4096。push constants は uniform バッファで置き換え、読み戻しだけ async にする。Worker で動かし、GPU のエラーのあとは Worker ごと作り直す。

試して採らなかったもの (詳細は経緯):

- 投機のラウンドの 1 投入化と GPU 上のトークン受け渡し (11.16、11.29)。
- KV の f16 化 (11.20)。
- 下書きの語彙の絞り込み、`lm_head` の低ビット化、木構造の下書き (11.19、11.35〜11.37)。
- プリフィルとデコードを 1 パスに混ぜること (11.18、11.33)。

残課題:

- Apple GPU の 1 行のパス (M2 の 2B で 21〜24 ms。llama.cpp は約 21 ms)。行列積はほぼ帯域の上限なので、残りは小さなカーネルの融合 (経緯 12.10)。
- Metal 直接のデバイスを macOS の既定にするか (いまは wgpu と同じ速さ)。GPU に閉じたデコードループは、ホストの時間が 2% ほどなので後回し (経緯 12.10)。
- 8×8 の行列演算が `fma` の鎖と一致する性質は M2 でしか測っていない。M1、M3 以降で `gemv2` が起動時の検査に通るか。
- 16 並列のデコードの合計速度 (経緯 11.17 で 220 tok/s に届かなかった)。
- 2B 本体の 4 ビット化の品質評価。
- ブラウザ版で WebGPU の subgroup を使えるか (32 レーンを保証できるか)。使えれば `GEMV_T1` とタイル attention が効く。
- ブラウザ版で、モデルの再選択を省くキャッシュ (OPFS)。
