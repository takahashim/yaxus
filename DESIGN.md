# yaxus 設計書

PLaMo 3 Translate を Rust で直接推論する翻訳サーバ/CLI。
llama.cpp や Python に依存せず、単一プロセスで動作し、HTTPS 経由で別マシンの CLI・ブラウザから利用する。

作成日: 2026-11-10

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
| wgpu/Vulkan 確認済み | `shader_f16=true`, `subgroup=true`, 1 バッファ上限 2047 MiB |

GPU 経路の結論は **Vulkan (wgpu + WGSL コンピュートシェーダ)**。理由:

- ROCm は未導入で、gfx1103 は ROCm の公式サポート外 (動かすには `HSA_OVERRIDE_GFX_VERSION` などの回避が必要)。
- Mesa RADV は標準パッケージで動作確認済み。f16 演算と subgroup 命令が使えるので、推論カーネルを書くには十分。
- candle は公式には ROCm 非対応 (フォーク/PR 段階)。Burn は wgpu/ROCm 両対応だが、本プロジェクトは「Rust で直接書く」方針なので、カーネルを自前で持つ方が制御しやすい。

内蔵 GPU なので実効メモリ帯域は概ね 60〜90 GB/s。**デコードは帯域律速**で、速度はほぼ「重み総バイト数 ÷ 帯域」で決まる。GPU の価値はプリフィル (入力文の処理) の高速化と、CPU より高い帯域利用率にある。

注意: 現在のログインセッションでは `render` グループが有効になっていない (`/etc/group` には登録済み)。再ログインするか `sg render -c '...'` で起動すること。これをしないと Vulkan から GPU が見えず llvmpipe (CPU) に落ちる。

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
- 参照実装 (`modeling_plamo.py`) は KV キャッシュに RoPE 前の K を積み、毎ステップ「キャッシュ内の相対位置 0..kv_len」で q/k に RoPE をかけ直す (NICT inference path)。RoPE のスコアは位置差だけに依存し、YaRN の concentration も一様な倍率なので、**RoPE 済みの K を絶対位置でキャッシュする実装と数学的に等価**。yaxus は後者を採用 (M1 で HF と logits 誤差 3e-5、greedy 完全一致を確認)。
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
                 HTTPS (rustls, 自己署名 or 指定証明書)
別マシン ──┬── ブラウザ ── GET /  (組み込み Web UI)
          └── yaxus CLI ── POST /v1/translate (JSON / SSE ストリーミング)
                                   │
                        ┌──────────▼──────────┐
                        │ yaxus serve (1 プロセス) │
                        │  axum ルータ + 認証      │
                        │  リクエストキュー (mpsc)  │
                        │  推論ワーカースレッド ×1   │
                        │   ├ tokenizer (Rust 再実装) │
                        │   ├ model (plamo3)         │
                        │   └ backend: wgpu/Vulkan   │
                        │        (fallback: CPU)     │
                        └────────────────────────┘
```

単一プロセス・単一モデルインスタンス。GPU を触るのは推論ワーカースレッド 1 本のみとし、HTTP ハンドラはチャネル経由で依頼とストリーム受信だけを行う。

### 2.1 ワークスペース構成

```
yaxus/
  Cargo.toml                 # workspace
  DESIGN.md
  crates/
    yaxus-api/             # サーバとクライアントが共有する JSON の型
    yaxus-tokenizer/       # PLaMo トークナイザ (pure Rust, 依存最小)
    yaxus-pfml/            # PFML3 プロンプト (chat_template.jinja の移植) と翻訳リクエストの検証
    yaxus-model/           # 設定・safetensors / GGUF のロード・plamo3 の計算グラフ・Backend トレイト・エンジン
    yaxus-backend-cpu/     # 参照実装 (f32, rayon)。正しさの基準とフォールバック
    yaxus-backend-wgpu/    # Vulkan 用 WGSL カーネル群
    yaxus-runtime/         # モデル名の解決・読み込み (バックエンド、精度、下書き)・プロンプト・実行中の要求 (Session)・文書のチェーン・翻訳メモリ
    yaxus-server/          # axum + rustls、認証、推論ワーカー、Web UI の埋め込み
    yaxus-cli/             # バイナリ `yaxus` (serve / translate / bench / client)
  web/                       # Web UI (素の HTML/JS 1 ファイル。ビルド不要、include_str! で埋め込み)
  tests/golden/              # Python 側で生成した tokenizer / logits のゴールデンデータ
```

`yaxus-cli` の feature: `local` (`translate`、`bench`: `yaxus-runtime` だけを使う) と `server` (`serve` も。既定)。`--no-default-features` ならクライアントだけで、GPU 依存も入らない。

サーバとローカル実行で共通の部分 (読み込み、文書、翻訳メモリ) は `yaxus-runtime` に置く。`yaxus-server` は HTTP と推論スレッドへの受け渡しだけを持つ (2026-11-11 に `yaxus-server` と CLI から切り出した)。

1 件の要求がエンジン上でたどる処理 (停止トークン、翻訳メモリのヒント、トークンから文字列への復号、終わったら出力をメモリへ) は `yaxus_runtime::Session<T>` の 1 か所にある。`T` はその要求の持ち主で、サーバのワーカーでは HTTP の接続 (単文) か文書の段落、`run_document` では段落の番号、`run_request` では無し。呼び出し側は `Session::step` が返す更新 (`Delta`、`Done`、`Failed`) を持ち主へ届けるだけ。

---

## 3. 推論コア

### 3.1 Backend トレイト

```rust
pub trait Backend {
    type Buf;                                           // デバイス上のテンソル
    fn upload_weights(&mut self, w: &WeightSource) -> Result<Self::Buf>;
    fn embed(&mut self, ids: &[u32]) -> Self::Buf;      // [T, H]
    fn rmsnorm(&mut self, x: &Self::Buf, w: &Self::Buf, offset: f32) -> Self::Buf;
    fn matmul(&mut self, x: &Self::Buf, w: &Self::Buf) -> Self::Buf; // [T,K] x [N,K]^T
    fn qk_norm_rope(&mut self, qkv: &Self::Buf, layer: &LayerMeta, pos0: usize) -> (Self::Buf, Self::Buf, Self::Buf);
    fn attention(&mut self, q: &Self::Buf, kv: &mut KvCache<Self::Buf>, layer: &LayerMeta) -> Self::Buf;
    fn swiglu(&mut self, gate_up: &Self::Buf) -> Self::Buf;
    fn add(&mut self, a: &Self::Buf, b: &Self::Buf) -> Self::Buf;
    fn argmax_last(&mut self, logits: &Self::Buf) -> u32; // greedy 専用
}
```

`yaxus-model` はこのトレイトだけに依存して plamo3 の forward を書く。CPU と wgpu で同じコードパスを通すことで、CPU 実装をゴールデンにして GPU 実装を検証できる。

### 3.2 重みの持ち方

- safetensors (bf16) を mmap で読み、テンソル単位で GPU バッファにアップロードする。1 バッファ上限 2047 MiB に対し、2B/8B の最大テンソルは `lm_head`/`embed_tokens` の 880 MB (8B) なので分割不要。31B は要分割だが当面対象外。
- WGSL に bf16 型は無いので、**bf16 を u32 に 2 個パックしたままバッファに置き、カーネル内で `bitcast<f32>(u << 16)` で f32 に展開**する。変換は無損失で、ロード時の型変換も不要。
- 活性値は f32 (必要なら f16 に落とす)。KV キャッシュは f16。
- 量子化 (後段フェーズ): ブロック 32 ごとに f16 スケールを持つ Q8 / Q4 を自前定義し、`yaxus quantize` で事前変換した独自フォーマット (`.ysq`、safetensors のメタデータ拡張) として保存。デコードはカーネル内で逆量子化する。8B を実用速度にするにはこれが必須。

### 3.3 必要なカーネル (WGSL)

| カーネル | 用途 | 備考 |
|---|---|---|
| `embed` | トークン ID → 埋め込み行の gather | |
| `rmsnorm` | offset 付き RMSNorm | f32 で計算、subgroup reduce |
| `gemv_bf16` | デコード用 (T=1) の行列ベクトル積 | 帯域律速。1 ワークグループで複数行、subgroup で内積を reduce |
| `gemm_bf16` | プリフィル用 (T>1) のタイル行列積 | 共有メモリ 64 KiB を使ったタイル化、f16 積算も検討 |
| `rope_qknorm` | q/k の per-head RMSNorm と RoPE (通常/YaRN) | cos/sin テーブルは層種別ごとに事前計算 |
| `attn_prefill` | 因果 + sliding window マスク付き attention | GQA 対応、オンライン softmax |
| `attn_decode` | 1 クエリ vs KV キャッシュ | KV head ごとに 8 query head をまとめて処理 |
| `swiglu` | `silu(h0)*h1` | |
| `add` | 残差加算 | rmsnorm に融合してもよい |
| `argmax` | 最終トークンの logits の argmax | 107520 要素。GPU 上で完結させ転送を避ける |
| `gemv_q8` / `gemv_q4` | 量子化重み版 | フェーズ 4 |

### 3.4 KV キャッシュ

- sliding 層 (8 層中 7 層) は **window=2048 のリングバッファ**で十分。位置 `p` の K/V をスロット `p % 2048` に書く。
- full 層は最大コンテキスト分を確保 (既定 8192 トークン、設定で変更可)。
- 8B でのトークン当たりの KV サイズ: 1 層あたり 2 × 4 heads × 128 × 2 bytes = 2 KiB。full 4 層 × 8192 トークン ≈ 64 MiB、sliding 28 層 × 2048 ≈ 112 MiB。十分小さい。
- バッチ 1 で開始。後からリクエスト間のバッチ化 (continuous batching) を入れられるよう、キャッシュはシーケンス ID で引けるテーブル構造にしておく。

### 3.5 生成ループ

1. PFML3 プロンプトを組み立て → トークナイズ (bos 付与)。
2. プリフィル: 入力全体を 1 回の forward (長い場合は 512 トークン単位でチャンク)。
3. デコード: `argmax` → ストリームへ送出 → 次トークンを forward。
4. 停止: `<|plamo:tag|>` / `<|plamo:bos|>` / `max_new_tokens` / クライアント切断。
5. 長文入力は、改行や文境界で分割して複数リクエストにするのをクライアント側 (CLI/Web UI) の責務にする。サーバは 1 リクエスト = 1 プロンプトを守る。

---

## 4. サーバ

### 4.1 HTTPS

- axum + `axum-server` (rustls)。
- 証明書は `--cert/--key` 指定か、未指定なら初回起動時に `rcgen` で自己署名証明書を `~/.config/yaxus/` に生成して再利用する。CLI クライアントは `--ca` でその証明書をピン留めできる (`--insecure` も用意)。
- LAN 公開前提なので **Bearer トークン認証は必須** (`--token` または設定ファイル。未設定なら起動時に生成して表示)。Web UI はトークンを localStorage に保持。

### 4.2 API

```
GET  /                      Web UI (単一 HTML)
GET  /healthz               {"status":"ok","model":"plamo-3-translate-2b","backend":"vulkan"}
POST /v1/translate          本体
GET  /v1/languages          対応言語コード一覧
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
  "history": [{"source": "...", "target": "..."}],          // 任意、直前の訳の一貫性用
  "max_new_tokens": 512,
  "stream": true
}
```

- `form` は `polite` / `plain` のどちらか (それ以外は解析の時点で 400)。省略は「指定なし」で、空文字列は受け付けない。
- `stream: false` → `{"text": "...", "usage": {...}}`。`usage` は `prompt_tokens`、`reused_tokens`、`completion_tokens`、`queue_ms`、`prefill_ms`、`elapsed_ms`。
- `stream: true` → `text/event-stream`。各 `data:` は `type` で種類を表す: `{"type":"delta","text":"..."}` を逐次送り、最後に `{"type":"done","usage":{...}}` か `{"type":"error","error":{...}}`。
- 文書 (`POST /v1/translate/document`) のイベントは `{"type":"delta","segment":i,"text":".."}`、`{"type":"segment_done","segment":i}`、最後に `done` (`usage` は上の項目に `segments` と `chains` を足したもの) か `error`。
- エラーは本文もストリームも `{"error": {"code": "...", "message": "...", "segment": i}}` の形 (`segment` は文書の段落を指すときだけ)。`code` と HTTP ステータスは 1 対 1: `invalid_request` / `prompt_too_long` (400)、`unauthorized` (401)、`queue_full` (429)、`unavailable` (503、推論スレッドの停止)、`internal` (500、生成の失敗)。
- 入力に `<|plamo:` を含む場合は 400。言語コードは chat_template と同じ正規表現で検証。
- 同時接続は受け付けるが、推論は直列実行。キュー長上限 (既定 16) を超えたら 429。

### 4.3 Web UI

- 依存なしの単一 HTML (textarea 2 枚、言語セレクタ、form/glossary の折り畳み、SSE で逐次表示)。
- ビルド工程を持たないため `include_str!` で埋め込み、バイナリ 1 つで配布できる。

---

## 5. CLI

```
yaxus serve   --model ~/.cache/huggingface/hub/... [--backend auto|gpu|cpu] [--listen 0.0.0.0:8443] [--token ...]
yaxus translate --to en [--from ja] [--form polite] "テキスト"   # サーバを立てずローカルで実行
yaxus client  --server https://host:8443 --to en  < input.txt    # 標準入力を段落単位で送る
yaxus quantize --model ... --scheme q8|q4 --out ...
yaxus download --model pfnet/plamo-3-translate-2b              # hf-hub crate で取得
yaxus bench   --model ...                                         # プリフィル/デコード速度計測
```

---

## 6. 検証方針

1. **トークナイザ**: Python の `Plamo3Tokenizer` で多言語・特殊トークン・絵文字・連続空白などの入力を encode/decode したゴールデンデータ (`tests/golden/tokenizer.jsonl`) を作り、Rust 実装と完全一致させる。
2. **モデル (CPU)**: transformers (CPU, bf16→f32) で短いプロンプトの logits を保存し、CPU バックエンドの logits と最大絶対誤差で比較。greedy 出力列の一致も確認。
3. **モデル (Vulkan)**: CPU バックエンドと同一入力で logits を比較。カーネル単体も CPU 実装と突き合わせる単体テストを持つ。
4. **E2E**: README のサンプル文で期待訳が得られること。
5. **性能**: `yaxus bench` でプリフィル tok/s とデコード tok/s を継続的に記録。

---

## 7. 性能見積もり (このマシン、帯域 60〜90 GB/s と仮定)

| モデル | 重み形式 | サイズ | デコード理論上限 | 現実的な目安 |
|---|---|---|---|---|
| 2B | bf16 | 5.8 GB | 10〜15 tok/s | 7〜12 tok/s |
| 2B | Q8 | 3.1 GB | 20〜29 tok/s | 15〜22 tok/s |
| 8B | bf16 | 17 GB | 3.5〜5 tok/s | 3〜4 tok/s |
| 8B | Q4 | 約 5 GB | 12〜18 tok/s | 9〜14 tok/s |
| 31B | Q4 | 約 17 GB | 3.5〜5 tok/s | (GTT 31 GiB に収まるが非実用的) |

- 日本語 1 文は 20〜40 トークン程度なので、2B bf16 でも対話的な利用には十分。
- 品質差は 2B→8B で COMET 平均 0.606→0.612 と小さく、8B→31B はさらに小さい。**既定は 2B、品質優先時に 8B (Q4/Q8)** とする。
- プリフィルは計算律速。780M の f16 ピーク (約 16 TFLOPS) に対し実効 20〜30% とすると、8B で 1000 トークンの入力が 3〜5 秒。

---

## 8. 実装フェーズ

| フェーズ | 内容 | 完了条件 |
|---|---|---|
| M0 | ワークスペース作成、2B ダウンロード、トークナイザ実装 | ゴールデンテスト全一致 |
| M1 | config/safetensors ロード、plamo3 forward、CPU バックエンド | HF と greedy 出力が一致 |
| M2 | wgpu/Vulkan バックエンド (bf16 カーネル、KV リングバッファ) | CPU と logits 一致、2B で 7 tok/s 以上 |
| M3 | HTTPS サーバ、認証、SSE、Web UI、CLI client | 別マシンのブラウザと CLI から翻訳できる |
| M4 | Q8/Q4 量子化、8B 対応、bench | 8B Q4 で 9 tok/s 以上 |
| M5 | continuous batching、チャンクプリフィル、長文分割の UI 支援 | 複数クライアント同時利用 |

### 8.1 進捗 (2026-11-10)

| フェーズ | 状態 | メモ |
|---|---|---|
| M0 | 完了 | `yaxus-tokenizer`: ゴールデン 1,814 件 (2026-11-10 にモデルカード由来の行を除き、パブリックドメイン作品の抜粋 `tests/golden/corpus/` に置き換えて 1,696 件) の encode/decode、プロンプトテンプレート 14 件 (エラー系含む)、ストリーミングデコードが Python と完全一致。`(.)\2{3,}` の後方参照と Python の `\s` 定義は手書きで再現 |
| M1 | 完了 | `yaxus-model` (Backend トレイト、forward、greedy 生成)、`yaxus-backend-cpu`。HF (f32) と最終 logits の最大誤差 3e-5、greedy 48 トークンが 3 ケースとも一致 |
| 高速化 1 | 完了 | rayon を物理コア数 (8) で起動 (`RAYON_NUM_THREADS` で上書き可)、`.cargo/config.toml` で `target-cpu=native`、T<4 では bf16/Q8 を一時バッファなしで直接内積 |
| 高速化 2 (Q8) | 完了 | ロード時量子化 (32 要素ブロック、f16 スケール = max\|w\|/127)。行列 + `lm_head` が対象、埋め込みと norm は bf16 のまま。重み 5.76→3.32 GB、ロード時の変換 約 0.65 s。CLI の既定を q8 に変更 (`--quant bf16` で無損失) |
| M3 | 完了 | `yaxus-api` (共有の型)、`yaxus-server` (axum + axum-server/rustls(ring))、`web/index.html`、`yaxus client`。下記「M3 の実装メモ」参照。M2 より先に実施 |
| M2 | 完了 | `yaxus-backend-wgpu`。下記「M2 の実装メモ」参照 |
| v2 エンジン | 計画済み | 行ベースのエンジンへの作り直し (KV 再利用、投機の 1 投入化、同時処理、document API)。第 11 章 |
| 投機的デコード | 完了 | 0.8B を下書きに使う greedy 投機的デコード。下記 8.4 参照。GPU で 22.5 → 30.4 tok/s、出力は完全一致 |
| プリフィルの f16 化 | 完了 | wgpu 30 の cooperative matrix (RDNA3 の WMMA) で f16 入力・f32 累積の GEMM。下記 8.6 参照。512 トークンで 209 → **392 tok/s** |
| 下書き側の高速化 | 完了 | 融合カーネル・単一パス・バインドグループキャッシュ・下書き Q4。下記 8.5 参照。投機ありで 30.4 → **46.4 tok/s**、投機なしでも 22.5 → 26.7 tok/s |

CPU 実装の 2B 実測 (32 トークンの英文 → 日本語、18 トークン生成):

| 構成 | プリフィル | デコード |
|---|---|---|
| bf16, 16 スレッド, 汎用 x86-64 (M1 時点) | 37.5 tok/s | 9.0 tok/s |
| bf16, 8 スレッド, native | 45 tok/s | 9.9 tok/s |
| **Q8**, 8 スレッド, native | 41 tok/s | **17.6 tok/s** |

デコードは帯域律速で、どの構成も実効 47〜49 GB/s (1 トークンあたりの重み読み出し量 × tok/s)。Q8 の品質: HF ゴールデン 3 件で greedy 完全一致 (上位 20 logits の誤差 ≤ 0.09)、さらに 7 言語 18 文で bf16 と出力が完全一致。

`target-cpu=native` はこのマシン向け。別マシンに配る client バイナリは `RUSTFLAGS= cargo build` などで上書きしてビルドすること。

ゴールデンデータの再生成 (Python 環境は transformers + numba + torch(CPU) が必要):

```
python tests/golden/gen_tokenizer.py $MODEL_DIR > tests/golden/tokenizer.jsonl
python tests/golden/gen_logits.py   $MODEL_DIR > tests/golden/logits_2b.jsonl
python tests/golden/gen_logits.py   $MODEL_0_8B_DIR > tests/golden/logits_0_8b.jsonl   # CI (.github/workflows/model.yml)
YAXUS_REQUIRE_GOLDEN=1 cargo test --release   # トークナイザ (モデルが無ければ失敗させる)
cargo test --release -p yaxus-backend-cpu -- --ignored   # モデル (2B ロードあり)
YAXUS_MODEL_DIR=$MODEL_0_8B_DIR YAXUS_GOLDEN=$PWD/tests/golden/logits_0_8b.jsonl \
  cargo test --release -p yaxus-backend-cpu --test golden -- --ignored   # CI と同じ 0.8B
```

### 8.2 M3 の実装メモ

- **バックエンドは当面 CPU (Q8)**。`yaxus serve` は推論スレッド 1 本 + tokio (2 スレッド)。キューは `--queue` (既定 16) 件まで待たせ、超えたら 429。
- **TLS**: §4.1 の「自己署名証明書」は「ローカル CA + サーバ証明書」に変更した。初回起動時に `~/.config/yaxus/ca.pem` (CA) を作り、それで署名したサーバ証明書を発行する。SAN は localhost / 127.0.0.1 / ::1 / ホスト名 / `ホスト名.local` / 主 IP + `--san` で追加したもの。SAN が変わればサーバ証明書だけ再発行するので、クライアントが信頼する `ca.pem` は変わらない。秘密鍵とトークンは 0600。
- **トークン**: `--token` / `YAXUS_TOKEN`、未指定なら初回に 192 bit の乱数を生成し `~/.config/yaxus/token` に保存 (以降の起動ではパスだけ表示)。比較は定数時間。
- **認証が必要なのは `POST /v1/translate` のみ**。`/`、`/healthz` (モデル名・キュー長)、`/v1/languages` は認証なし。
- `Usage` に `prefill_ms` (最初のトークンまで) と `queue_ms` (待ち時間) を追加。SSE はクライアント切断を検知して生成を打ち切る。未知の JSON フィールドは 400。
- 長文分割 (§10 の保留事項) は **段落 (空行区切り) 単位** に決定。CLI client と Web UI の両方で、直前 2 段落の原文/訳文を `history` として渡す。
- `yaxus-cli` の `server` feature (既定で有効) を外す (`--no-default-features`) と、client だけのバイナリになる (GPU/モデル依存なし)。

使い方:

```
# サーバ (初回はトークンと CA を生成して表示)
yaxus serve                              # 0.0.0.0:8443, 2B, Q8
yaxus serve --san myhost.example.lan     # 証明書に名前を追加

# 別マシン: サーバの ~/.config/yaxus/ca.pem をコピーしておく
export YAXUS_SERVER=https://myhost.local:8443 YAXUS_TOKEN=... YAXUS_CA=~/yaxus-ca.pem
yaxus client --to en < doc.txt
yaxus client --to ja --form polite -g "owner=オーナー" "The owner approved it."

# ブラウザ: https://<host>:8443/ を開き、トークンを入力 (ca.pem を取り込めば警告なし)
```

### 8.3 M2 の実装メモ

- wgpu 28 (Vulkan のみ)、手書き WGSL 12 本。重みは Q8 (int8 ×4 を u32 に詰め、f16 スケールを 2 個ずつ u32 に詰める) または bf16 (u32 に 2 個) のまま GPU に置き、カーネル内で展開。活性値と KV キャッシュは f32。埋め込み表は GPU に載せず、CPU 側で該当行だけ読んで転送する。
- RoPE の cos/sin はセッション作成時に CPU で表を作って渡す (GPU の `cos` は大きな角度で精度が保証されないため)。カーネル固有の引数は push constant (wgpu の `IMMEDIATES`) で渡す。
- 行列積: T=1 は `gemv1` (8 行/WG、1 行 32 レーン、16 バイト単位の読み込み)、T≤8 は `gemv8`、それより大きいプリフィルは 64×64 タイルの `gemm`。
- **APU では計算ジョブも gfx リングで動き、1 ジョブが約 2 秒を超えるとカーネルが GPU をリセットする** (448 トークンのプリフィルを 1 回で投入して発生)。行列積の FLOP を積算し、32 GFLOP (780M で約 40 ms) ごとに投入を分ける。デスクトップのカクつき防止も兼ねる。
- `--backend auto` (既定) は Vulkan の GPU があれば使い、無ければ CPU。llvmpipe などソフトウェア実装のアダプタは選ばない。
- CPU との比較 (`cargo test --release -p yaxus-backend-wgpu -- --ignored`): Q8/bf16 とも全語彙の logits 差 ≤ 5e-5、greedy 完全一致、600 トークン (2 チャンク) でも 8e-5。7 言語 18 文の訳が CPU と完全一致。
- `YAXUS_PROFILE=1` で命令記録時間と GPU 待ち時間を表示。デコード 1 トークンは記録 約 1 ms、GPU 約 42 ms (GPU 律速)。

`yaxus bench` (2B、Q8、32 トークン生成、ベスト値):

| プロンプト長 | CPU プリフィル | Vulkan プリフィル | CPU デコード | Vulkan デコード |
|---|---|---|---|---|
| 32 | 42 tok/s | 148 tok/s | 18.7 tok/s | 22.5 tok/s |
| 160 | 40 tok/s | 187 tok/s | 18.4 tok/s | 20.9 tok/s |
| 512 | 30 tok/s | 167 tok/s | 17.6 tok/s | 19.8 tok/s |
| 1024 | — | 126 tok/s | — | 17.6 tok/s |

Vulkan のデコードは Q8 で実効 約 66 GB/s、bf16 で 約 69 GB/s (13.4 tok/s) とほぼ帯域の上限。ここから速くするには読むバイト数を減らす (Q4) か、1 回の読み込みで複数トークンを確定させる (投機的デコード) しかない。

### 8.4 投機的デコード

- `yaxus-model::speculative`。0.8B (トークナイザ・テンプレートは 2B と完全一致を確認済み) が k トークンを greedy に下書きし、2B が「直前の確定トークン + 下書き k 個」を 1 回の forward で検証して、各位置の argmax と一致する最長の接頭辞を採用し、最初の不一致位置では 2B の予測を採用する。greedy なので **出力は 2B 単体と完全に同一**。
- KV キャッシュは `Session::rewind` で位置を戻すだけ (attention はクエリ位置より後ろのキーを読まないので、捨てたエントリは後で上書きされる)。
- 検証用に T=2..8 行専用の `gemvm{T}` カーネルを追加: K を 512 ずつ区切って全行の活性値をワークグループメモリに置き、8 レーングループ × 2 行で共有、レーン間の和は subgroup shuffle。これで T=5 の forward が 66 → 48 ms (T=1 は 42 ms)。最初の版は活性値をレジスタに抱えて遅く (40〜51 GB/s)、K=10240 の `down_proj` は活性値が L1 に収まらず 35 GB/s だった。`YAXUS_PROFILE=2` (タイムスタンプクエリ) で行列の形ごとの時間と帯域を出して特定した。
- CLI/サーバの既定は `--draft auto` (0.8B が HF キャッシュにあれば使う)、`--draft-k 5`、`--draft none` で無効。CPU バックエンドでは自動的に無効 (CPU の行列積は複数行をまとめても得をせず、18.5 → 14.6 tok/s と遅くなるため)。
- 7 言語 18 文 + 長文 1 (計 19 件) の実測 (2B Q8、780M):

| 設定 | デコード | 1 回の検証で確定するトークン数 (平均) |
|---|---|---|
| 投機なし | 22.5 tok/s | 1 |
| k=3 | 28.9 tok/s | 3.6 |
| k=4 | 29.5 tok/s | 4.5 |
| **k=5 (既定)** | **30.4 tok/s** | 5.3 |
| k=6 | 27.8 tok/s | 5.6 |

- 1 ラウンド (k=4) の内訳: 2B の検証 約 50 ms + 0.8B の下書き 約 15 ms × 4 + 同期など 約 12 ms。以後の伸びしろは主に下書き側 (0.8B の lm_head 107520×1536 が重い、ディスパッチ数 217/トークン)。
- テスト: `speculative_matches_plain` (k=1/3/5/7、上限がラウンド途中で切れる場合を含め完全一致)。

### 8.5 下書き側 (と全体) の高速化

`YAXUS_PROFILE=2` で 0.8B の 1 トークン (約 16 ms) を分解すると、カーネル合計 11.2 ms に対して **ディスパッチ間の空き 約 4.4 ms + CPU での記録 0.8 ms** が無駄だった (213 ディスパッチ、各 約 20 µs)。行列積自体は 55〜83 GB/s で帯域上限に近い。

| 変更 | 0.8B デコード | 2B デコード |
|---|---|---|
| 開始時点 | 61 tok/s | 23.5 tok/s |
| 融合 (`add_norm`: post-norm + 残差加算 + 次の pre-norm、`glu`: gate_up GEMV に SwiGLU を融合) で 1 層 13 → 8 ディスパッチ。バインドグループのキャッシュ、attention の vec4 化、argmax の 2 段化 | 73 tok/s | 24.9 tok/s |
| 1 回の投入を 1 つのコンピュートパスにまとめる (ディスパッチごとのパス開始/終了をやめる) | 86 tok/s | 26.9 tok/s |
| 下書きを Q4 (32 要素ブロック、f16 スケール、`max\|w\|/7`、0.5625 B/重み) | 131 tok/s | — |

- `add_norm` / `matmul_glu` は Backend トレイトの既定実装 (従来の演算の組み合わせ) を持つので、CPU は変更なし・結果も不変。GPU の `glu` は T=1 のみ融合し、T>1 は従来どおり。
- 埋め込みもプールしたバッファに `write_buffer` する (書き込みは次の投入の先頭に入るので、先に記録済みの命令を投入してから書く)。
- argmax の 2 段目で softmax の最大確率も計算する (`argmax_prob`)。下書きを確率で打ち切る `--draft-min-prob` を用意したが、0.3〜0.7 で効果は誤差範囲 (0.8B は外すときも自信を持って外す) なので既定は 0 (無効)。
- Q4 は出力が変わる程度に非可逆なので下書き専用 (`--draft-quant q4`、既定)。下書きの精度は出力に影響せず、採用率 (1 ラウンドの確定トークン 5.7 → 5.1) だけが下がる。CPU バックエンドは Q4 非対応。
- 投機的デコードの実測 (19 件、出力はすべて投機なしと完全一致):

| 設定 | デコード |
|---|---|
| 投機なし | 26.7 tok/s |
| 下書き Q8、k=5 | 39.0 tok/s |
| **下書き Q4、k=5 (既定)** | **46.4 tok/s** |
| 下書き Q4、k=6 / 7 | 45.3 / 45.7 tok/s |

- テスト追加: `tests/kernels.rs` (bf16/Q8/Q4 × gemv1/gemvm/gemv8/gemm/glu をホスト参照と比較)、`argmax_prob` の確率をホスト計算と比較。

### 8.6 プリフィルの f16 化

512 トークンのプリフィル (2B Q8) の内訳は GEMM が約 9 割、attention が残り。最初の GEMM (64×64 タイル、1 スレッド 4×4) は約 2 TFLOPS で、ワークグループメモリの読み出しが律速だった。

| GEMM | 160 トークン | 512 トークン | 1024 トークン | CPU との logits 差 |
|---|---|---|---|---|
| 64×64 タイル f32 (従来) | 198 tok/s | 209 tok/s | 181 tok/s | 5e-5 |
| 64×128 タイル f32、1 スレッド 4×8 (`--prefill-precision f32`) | 261 tok/s | 285 tok/s | 235 tok/s | 5e-5 |
| 同じタイルで f16 のタイルと f16 パック演算 | 229 tok/s | 255 tok/s | 215 tok/s | 精度テスト不合格 |
| 転置しない配置で vec4 の dot (f32 / f16) | 200 / 178 | 244 / 208 | 206 / 180 | — |
| **cooperative matrix (WMMA)、f16 入力・f32 累積 (既定)** | **376 tok/s** | **392 tok/s** | **304 tok/s** | 5e-3〜1e-2 |

- 普通のシェーダで f16 にしても速くならなかった (naga → SPIR-V → RADV ACO でパック演算が効かず、f16 への変換と 2 バイト単位の共有メモリ書き込みが増えるだけ)。kohagi で f16 が約 2 倍だったのは、CubeCL が行列演算ユニットを使っていたためと思われる。
- RADV 25.2.8 は `VK_KHR_cooperative_matrix` で 16×16×16 の f16×f16→f32 を報告する。wgpu 28 には無いので **wgpu 30 に移行**し (変更は 4 箇所)、実験的機能 `EXPERIMENTAL_COOPERATIVE_MATRIX` を有効にした (`ExperimentalFeatures::enabled()` は unsafe。使うのはこの機能だけで、カーネルはテストで検証)。
- カーネル `GEMM_COOP`: 128 スレッドで 64 トークン × 64 出力。活性値 64×32 と重み 64×32 (Q8/Q4/bf16 から展開) を f16 でワークグループメモリに置き、各 subgroup が 16 列のストライプを 16×16 の累積行列 4 枚 (64 トークン) で受け持つ。subgroup が 4 個 (wave32) ならストライプ 1 本、2 個 (wave64) なら 2 本。N % 64 == 0 が必要 (それ以外は f32 GEMM)、出力は 64 行単位で確保する。
- naga 30 の制約と回避: cooperative 演算は一様な制御フローが必要で、`subgroup_id` などのビルトインは一様と見なされない → `num_subgroups` をワークグループメモリ経由 (`workgroupUniformLoad`) で一様化し、早期 return をやめた。`coopStore`/`coopLoad` の引数に `var` の読み出しや添字の式を直接書くと SPIR-V 出力で "Expression is not cached" と落ちる → すべて `let` に束縛してから渡す。
- 精度: logits の CPU との差は 5e-5 → 5e-3〜1e-2 に増えるが、HF ゴールデン 3 件と 19 件の greedy 出力は f32 プリフィルと完全一致。`--prefill-precision f32` で従来の精度に戻せる。
- 試して負けた f16 シェーダ版・dot 版・64×64 版は削除し、f32 の 64×128 版だけをフォールバックとして残した。
- 残り: 512 トークンで attention が約 15%。GEMM の段取り (f16 への変換と書き込み) がまだ重く、WMMA の理論値には遠い。

---

## 9. 主要クレート

| 用途 | クレート |
|---|---|
| GPU | `wgpu` 30 (Vulkan backend のみ有効化、実験的機能 cooperative matrix を使用), `bytemuck`, `half` |
| CPU 並列 | `rayon` |
| 重み | `safetensors`, `memmap2`, `hf-hub` |
| HTTP | `axum` 0.8, `axum-server` (rustls), `tokio`, `tower-http` |
| TLS | `rustls`, `rcgen` |
| CLI | `clap`, `reqwest` (rustls-tls, stream) |
| 共通 | `serde`, `serde_json`, `anyhow`, `thiserror`, `tracing` |

---

## 10. 決定事項と保留事項

決定:

- GPU 経路は Vulkan/wgpu で自前カーネル。ROCm は採用しない。
- 既定モデルは 2B。既定の重み形式は Q8 (ロード時量子化)。bf16 は `--quant bf16` で選べる無損失モード。
- 生成は greedy 固定 (サンプリング API は持たない)。
- 認証はトークン必須、TLS は自己署名を自動生成。

保留 (実装中に判断):

- プリフィル GEMM で f16 積算を使うか (精度と速度のトレードオフ。CPU との logits 比較で判断)。
- KV キャッシュを f16 にするか f32 にするか (まず f16、誤差が出れば f32)。

---

## 11. v2 エンジン: 仕様と計画 (2026-11-10)

対話的な 1 段落の翻訳と、長い文書の一括翻訳の両方を速くするための作り直し。後方互換性 (API、内部トレイト、CLI オプション) は考慮しない。2B 本体の Q4 化は対象外 (後回し)。

### 11.1 出発点 (実測)

- デコード 1 トークン 37 ms は「2B Q8 の重み 2.77 GB ÷ 約 75 GB/s」そのもの。DDR5 デュアルチャネルの上限は約 90 GB/s で、カーネル改良の余地はほぼ無い。
- 1 回の forward (パス) で処理する行数を増やしても時間はほとんど増えない:

| 行数/パス | 1 | 2 | 4 | 8 | 9〜32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| 時間 (ms) | 36.9 | 37.9 | 39.8 | 46.7 | 140〜144 | 155 | 306 | 600 |
| 行/秒 | 27 | 53 | 100 | 171 | 64〜223 | 413 | 419 | 427 |

  GPU は毎秒約 400 行を処理できるが、単一ストリームは 1 パスに 1〜6 行しか渡せていない。9〜63 行は専用カーネルが無く 64 行タイルの GEMM に落ちる「穴」。
- 文書シナリオ (モデルカード `README_ja.md` の散文 14 段落を ja→en、直前 2 段落の履歴つき、現行の既定設定): プロンプト平均 282 トークン、生成平均 57 トークン。**プリフィル 10.7 s (37%)**、デコード 18.1 s、通しで 27.5 tok/s (デコードのみなら 43.8 tok/s)、訳が出始めるまで平均 0.76 s。
- 投機的デコード 1 ラウンド (k=5): 検証 約 49 ms + 下書き 5 × 7.7 ms + CPU 往復など 約 10 ms ≈ 97 ms で平均 5.0 トークン。下書きの 1 トークン当たり採用率は約 0.93。
- 2B 本体の Q4 (参考、今回は対象外): 単体 44.8 tok/s、投機込み 54.6 tok/s、19 件中 5 件で訳が言い換えレベルで変わる。

### 11.2 目標

| 指標 | 現状 | 目標 | 手段 |
|---|---|---|---|
| 単一ストリームのデコード (19 件セット) | 48 tok/s | 60 tok/s 以上 (努力目標 70) | 11.6 |
| 続きの段落で訳が出始めるまで | 平均 0.76 s | 0.3 s 以下 | 11.5 |
| 文書シナリオ (履歴つき逐次) の通し速度 | 27.5 tok/s | 36 tok/s 以上 | 11.5 |
| 8 並列の合計デコード | (逐次のみ) | 150 tok/s 以上 | 11.7 |
| 16 並列の合計デコード | — | 220 tok/s 以上 | 11.7 + 中間カーネル |
| 長い文書の一括翻訳の通し速度 | 27.5 tok/s | 4 倍以上 | 11.8 |

訳文は、同じ精度設定 (11.9) の単独実行と同一であることを各段階の合格条件にする。

### 11.3 中心となる考え方: 行とパス

- **行** = (系列スロット, 位置, トークン)。**パス** = 行の集合に対する 1 回の forward で、重みを 1 回だけ読む。
- 同じスロットの行はパス内で位置が連続する (ラン)。各行は自分のスロットの、自分の位置以下のキーだけを見る (sliding 層はさらに直近 2048)。別スロットの行は互いに独立。
- プリフィル、単発デコード、投機的デコードの検証、複数リクエストの同時処理は、すべて「どの行を 1 パスに載せるか」の違いとして同じ経路で実行する。
- 行ごとに独立な演算 (norm、行列積、SwiGLU、argmax) は現状のまま使える。変わるのは埋め込み、q/k の RoPE と KV 書き込み、attention の 3 つで、行メタデータ (スロット、位置) のバッファを読む。

### 11.4 エンジンの構成

- **1 デバイス**: `GpuContext` (device、queue、パイプライン、バッファプール、コマンドエンコーダ) を本体モデルと下書きモデルで共有する。重みと KV だけがモデルごと。2 モデルの命令を 1 回の投入に混ぜられる。
- **トークン ID の所在**: `Ids::Host(Vec<u32>)` と `Ids::Device(バッファ)`。argmax の結果を GPU 上のバッファに残し、次のステップの埋め込みがそれを読む (CPU 往復なし)。埋め込み表は GPU に置く (bf16 のまま、2B 550 MB + 0.8B 330 MB)。
- **KV アリーナ (ページ方式)**: 層ごとに K と V のバッファを 1 つずつ持ち、128 トークン単位のページに分ける。スロットはページ表 (最大 64 エントリ = 8192 トークン) を持ち、キーのアドレスは `ページ表[位置 / 128] * 128 + 位置 % 128`。リングバッファは廃止し、位置は絶対位置のみ。
  - 既定のアリーナは 16384 トークン分 (2B: 12.6 MB/ページ × 128 = 1.6 GB、0.8B: 2.1 MB/ページ × 128 = 0.27 GB)。`--kv-tokens` で変更。KV は f32 のまま (f16 化は後の選択肢)。
  - attention はページ単位で回す (128 キー/ページが現行カーネルの 1 ステップと一致)。
- **Backend トレイト**: 演算単位の構成は維持しつつ、`attention(qkv, rows, arena)`、`embed(ids)`、`argmax_rows -> Ids::Device`、`read_ids` (同期点) に改める。ページの割り当て・参照カウント・ページ表は `yaxus-model` 側 (CPU) で管理し、バックエンドは保存領域とページコピーだけを提供する。CPU バックエンドも同じインタフェースを実装して参照実装であり続ける (速度は問わない)。

### 11.5 プレフィックスの KV 再利用

- 完了した系列のスロットは解放せず、消費済みトークン列と一緒に **キャッシュ済み** として残す (LRU)。
- 新しいリクエストはプロンプトをトークン化し、キャッシュ済み・実行中のスロットの中から **最長共通接頭辞 (LCP)** が最も長いものを探す。16 トークン以上一致すれば **フォーク**: 一致範囲の完全なページは共有 (参照カウント)、末尾の半端なページだけ層ごとにコピーし、残りのトークンだけをプリフィルする。
- API は状態を持たないまま (クライアントは今までどおり履歴を全部送る)。同じ文書を訳し進めると、前の段落までの KV がそのまま再利用される。
- 履歴は 2 段落に限らず全件送れるようにする。プロンプトが上限 (既定 4096 トークン) を超えたら、古い履歴を半分まとめて落とす (落とした回だけ全体を処理し直す)。
- `form` 指定時は履歴側の出力ヘッダと最後の出力ヘッダが異なるため、一致は直前の原文までになる (訳文 1 つ分だけ処理し直す)。
- 下書きモデルも同じスロット対応でキャッシュする。下書き側が消費していない末尾は再利用時に追いつかせる。
- ページが足りなければキャッシュ済みスロットを LRU で捨てる。実行中の系列の分が確保できなければ 503。

### 11.6 投機的デコードの作り直し

- 1 ラウンド = 1 回の投入 + 1 回の読み戻し。下書き k ステップ (2 ステップ目以降は GPU 上のトークンを直接埋め込み)、本体の検証 (先頭の確定トークン + 下書き k 個)、全行の argmax を続けて記録し、下書きトークン列と検証結果をまとめて読む。
- 途中打ち切り (停止トークン、確率しきい値) はラウンド内ではできなくなる。確率しきい値は効果が無かったので削除する。停止トークン後の下書きは無駄になるが最終ラウンドだけ。
- k は 7 を既定候補にして測り直す (検証 8 行までは 46.7 ms)。
- 見込み: 往復 約 10 → 約 3 ms、k=7 で 1 ラウンド 約 105 ms・6.3 トークン → 約 60 tok/s。下書き側の改善 (11.10) が乗れば 70 tok/s 台。
- 訳文は投機なしと同一 (現行と同じ性質)。

### 11.7 スケジューラと同時処理

- 推論スレッドは 1 本のまま、ループごとに 1 パスを組み立てる: 新規リクエストの取り込み → パスの種類と行の選択 → 実行 → 結果の配信。
- **パスは 2 種類で、混ぜない** (精度の方針 11.9 のため):
  - デコードパス: 実行中の各ストリームの行。f32 カーネルのみ。
  - プリフィルパス: 待っている系列のプロンプト。行列演算ユニット (f16 入力) を使う。実行中のストリームがあるときは 1 パス 64 行まで (約 155 ms) に抑えてデコードと交互に実行し、ストリームの途切れを抑える。
- **投機の使い分け**: デコード行の合計が 8 行に収まる範囲でだけ投機を使う (ストリーム 1 本: k=7、2 本: k=3)。3 本以上は投機なしで各ストリーム 1 行ずつ (下書きの外れに行を使うより、実ストリームに使う方が得)。しきい値はベンチで調整する。
- **中間カーネル (9〜32 行、f32)**: いまの穴を埋める。複数行 GEMV を 32 行まで拡張するか、f32 の小タイル GEMM を作り、行数ごとに速い方を選ぶ表を持つ。目標は 16 行 60 ms 以下、32 行 90 ms 以下。
- 同時実行の上限は `--max-streams` (既定 16)。超えた分は従来どおりキューで待ち、あふれたら 429。

### 11.8 API、CLI、Web UI

- `POST /v1/translate`: 1 セグメント。項目は現行と同じ。`history` の件数制限なし。同時に複数投げれば自動的に同じパスで処理される。
- `POST /v1/translate/document` (新設): `segments: [..]` と共通オプション、`chains` (既定は自動) を受け取り、SSE で `{"i": 段落番号, "delta": ".."}` を流す。サーバ側で文書を **チェーン** に分ける: チェーン内は履歴つきの逐次 (11.5 の再利用が効く)、チェーン同士は並列。チェーン数は `min(8, 段落数 / 4)` を初期値にする。
- 履歴の扱いは評価してから決める (11.11 の検討事項)。チェーン方式なら履歴を捨てるのはチェーンの境目だけで済む。
- CLI: `yaxus client` と `translate` は複数段落の入力で document 経路を使う。`--chains 1` で従来どおりの完全な逐次。`yaxus bench` に `rows` (行数スケーリング)、`doc` (文書シナリオ)、`streams N` (並列) を追加。
- Web UI: 複数段落は document 経路で、段落ごとに並行して埋まっていく表示にする。

### 11.9 精度と再現性の方針

- デコード行は常に f32 カーネル (行数によるカーネルの違いで logits は 1e-5 程度しか動かない)。プリフィルは既定で f16 入力の行列演算ユニット (5e-3 程度)、`--prefill-precision f32` で f32。 11.33 から既定は int16 (整数ドット積、活性化 15 bit)。
  - **11.34 で変更**: デコード行 (16 行以下のパス) も既定で int16 の整数ドット積にした (`--decode-precision f32` で従来どおり)。f32 との logits の差は 5e-3 程度。行の結果は同じパスの他の行に依存せず、ビット単位で同じなので、同時実行や投機の有無で出力が変わらない性質は f32 のとき (1e-5) より強くなった。
- KV の再利用や同時実行により、同じリクエストでも KV が「デコードで作られた値」か「プリフィルで作られた値」かが変わり得る。差は上記の範囲で、greedy の結果が変わるのは上位 2 候補がほぼ同点のときだけ。
- 完全な再現性が必要なとき (テスト、比較) は `--prefill-precision f32 --no-kv-cache --max-streams 1` (`--no-kv-cache` は P6 の応答キャッシュも止める)。11.34 からは `--decode-precision f32` も付ける (CPU の f32 参照と比べる場合)。各段階の合格判定の「訳文が同一」はこの設定と既定設定の両方で確認し、既定設定で差が出た場合は件数と内容を記録する。

### 11.10 下書き側の追加改善 (11.6 の後、実験ベース)

効果を測って採否を決める。いずれも出力は変わらない。

- 小さい行列の帯域 (0.8B の `o_proj` 34 GB/s、`down_proj` 51 GB/s): 行数の少ない行列向けに 1 ワークグループの行数を調整。
- デコード時の attention (0.8B は KV ヘッド 1 個で 16 ワークグループしかなく 1 回 38 µs): キー方向の並列化。
- 下書きの `lm_head` (下書き時間の 16%) の候補語彙への絞り込み: プロンプト中のトークンと、対象言語でこれまで本体が出力したトークンから候補集合を作る。外れた分は採用率の低下として現れるので、効果は実測で判断。
- 再翻訳時に前回の訳を下書きに使う (`DraftSource` を「モデル」と「トークン列」の 2 種にする)。同一リクエストの応答キャッシュ。

### 11.11 段階と合格条件

| 段階 | 内容 | 合格条件 |
|---|---|---|
| P0 | 計測基盤: `bench rows / doc / streams`、文書シナリオ (モデル同梱の README から抽出、リポジトリには本文を置かない)、基準値の記録 | 11.1 の数字を再現 |
| P1 | エンジン中核: `GpuContext` 共有、ページ式 KV アリーナ、行メタデータ、埋め込み・RoPE・attention カーネルの行対応、CPU バックエンドの追従、既存の生成・投機を新経路へ移植 | 既存テスト全通過 (HF 比較、CPU/GPU 比較、カーネル単体、投機 = 通常)。単一ストリームの速度が ±3% 以内 |
| P2 | プレフィックス KV 再利用 (LCP、フォーク、LRU、履歴の上限処理) | 文書シナリオ (逐次) で通し 36 tok/s 以上、2 段落目以降の出始め 0.3 s 以下。再現性設定で訳文が再利用なしと同一 |
| P3 | 投機ラウンドの 1 投入化 (GPU 上のトークン受け渡し、k の再調整) | 19 件セットで 60 tok/s 以上、訳文が投機なしと同一 |
| P4 | スケジューラ、同時処理、中間カーネル、投機の使い分け、サーバのエンジンループ | 8 並列 150 tok/s 以上、16 並列 220 tok/s 以上。各ストリームの訳文が単独実行と同一 (再現性設定) |
| P5 | document API、チェーン分割、client、Web UI。履歴の扱いの評価 | 60 段落程度の文書で通し 4 倍以上 (現行比)。評価結果を記録して既定を決定 |
| P6 | 11.10 の実験 | 項目ごとに実測して採否を記録 |
| 後回し | 2B 本体の Q4 (品質評価つき)、KV の f16 化、8B (旧 M4) | — |

**P5 の検討事項 (履歴)**: 同じ文書を (a) 完全な逐次・全履歴、(b) チェーン分割 (4、8 本)、(c) 履歴なし完全並列、(d) 履歴なし + 前後の原文を `context` に渡す、の 4 通りで訳し、訳語の一貫性 (同じ原語に対する訳語の揺れの件数) と (a) との差分を比べて既定を決める。

### 11.13 P0 の結果 (基準値、2026-11-10)

計測は `yaxus bench rows|decode|set|doc` で行う (`set` は `tests/bench/sentences.tsv` の 19 件、`doc` はモデル同梱の `README_ja.md` から散文 14 段落を実行時に抽出)。

- `bench rows --draft none`: 1 行 37.0 ms、8 行 46.1 ms、9 行 139.4 ms、64 行 153.6 ms、256 行 598.5 ms (11.1 の表を再現)。
- `bench set`: 再現性設定 (`--draft none --prefill-precision f32`) 25.5 tok/s、既定 48.3 tok/s・平均プリフィル 174 ms。両者の訳文は 19 件とも同一。参照として `tests/bench/ref_f32.txt` と `ref_default.txt` を保存 (以降の段階は `bench set --compare` で確認する)。
- `bench doc` (ja→en、履歴 2): 既定で通し 27.9 tok/s、プリフィル 10.5 s (37%)、2 段落目以降の出始め平均 777 ms。再現性設定では通し 17.2 tok/s、出始め 1082 ms。
- **既定と再現性設定で文書の訳が 1 段落だけ異なった** (14 段落中 8 段落目、"across model sizes" と "for each model size" の言い換え、以降の段落への波及なし)。f16 プリフィル (行列演算ユニット) は、500 トークン級のプロンプトでは同点に近い候補の選択を変えうる。11.9 の方針どおり、合否は両設定で確認し、既定設定の差は記録する。

### 11.14 P1 の結果 (エンジン中核)

- `yaxus-model`: `Session` を廃止し、モデルが KV アリーナ (128 トークンのページ、参照カウント付きページプール) と系列 (`SeqId`: ページ表と長さ) を管理する。`forward(&[(SeqId, &[u32])], all_rows)` が複数系列の行を 1 パスで処理する (行メタデータ `PassMeta`: トークン、位置、KV スロット、ページ表オフセット)。`prefill`、`truncate` (投機の巻き戻し)、`fork` (P2 用、共有ページ + 末尾ページのコピー)、コピーオンライト (`reserve`) を用意。既定のアリーナは 16384 トークン (`YAXUS_KV_TOKENS` で変更)、最大位置 8192。
- Backend トレイト: `Cache`/`new_cache`/`last_row` を廃止し、`Kv`/`new_kv`/`copy_page`/`begin_pass`/`select_rows`/`reserve_activations` を追加。`embed` と `attention` はパスのメタデータを読む。CPU バックエンドも同じ形に移行 (HF との誤差は従来と完全に同じ)。
- wgpu: コンテキスト `Gpu` を `Arc<Mutex<..>>` で共有するハンドル `WgpuBackend` (`share()`) に分け、本体と下書きが 1 つのデバイス・1 つのエンコーダを使う。埋め込み表は GPU (bf16)。行メタデータは一時バッファからのエンコーダ内コピーで転送し、記録済みの命令との順序を保つ (宛先が固定なのでバインドグループのキャッシュも効く)。`qk_rope` は行ごとの位置と KV スロット、`attention` はページ表を引く (キーのスロットは 1 ステップに 1 回だけ計算して共有メモリに置く)。
- **落とし穴**: 移行直後、512 トークンのプリフィルが 7% 遅くなった。カーネル時間を比べると `down_proj` の行列演算ユニット GEMM だけが +18% (K=10240 で活性値を 40 KB 間隔の 64 行から読むので、バッファがどのメモリに置かれたかに敏感)。KV アリーナや埋め込み表の大きさ・確保順は無関係で、**活性値のバッファを重みより先に確保すると解消** した (専用領域に載るためと見られる)。`reserve_activations` でモデル読み込みの最初に 512 行分のバッファをプールに作っておく。down_proj は 7.6 ms (旧 7.7 ms)。
- 合格判定: テスト全通過 (HF 比較、CPU/GPU 比較 f32・行列演算ユニット両方、カーネル単体、投機 = 通常)。行数スケーリング 1 行 36.3 ms・64 行 153 ms・512 行 1301 ms (P0: 37.0 / 153.6 / 約 1297)。`bench set` 48.4 tok/s (P0 48.3)・再現性設定 25.4 tok/s (25.5)、訳文は両設定とも参照と 19/19 一致。`bench doc` 27.8 tok/s、2 段落目以降の出始め 772 ms (P0 27.9 / 777)、訳文は P0 の既定設定と同一。

### 11.15 P2 の結果 (プレフィックス KV 再利用)

- `Model::start(prompt)` / `finish(seq)`: 終わった系列を消費トークン列ごと LRU で保持し (最大 32 系列、ページ不足時は古い順に解放)、新しいプロンプトとの最長共通接頭辞が 16 トークン以上ならフォークして残りだけを処理する。投機的デコードは本体・下書きとも同じ仕組みを使い、下書きの `start` は最初のトークンを出した後に回す (出始めを遅らせない)。`--no-kv-cache` で無効、`--kv-tokens` でアリーナの大きさ。
- WebGPU は同一バッファ内のコピーを禁止しているので、ページのコピーは 1 ページ分の作業バッファを経由する。
- **履歴の方式を変えた**: 直前 N 組の窓をずらすと毎回プロンプトの先頭が変わって再利用できない (直前 2 組で再利用率 17%)。クライアント側の `HistoryWindow` (`yaxus-api`、CLI、Web UI で同じ規則) は開始位置を固定し、上限 (既定 4000 文字、`--history-chars`) を超えたときだけ、上限の 1/4 に収まる最新の組に切り詰める。切り詰めのたびに残した分を再計算するが、平均すると 1 段落あたり約 1/3 組分で済む (「古い半分を落とす」だと上限によらず 1 組分になり、上限 2000 文字では 14 段落中 4 回の全再計算が起きた)。
- 文書シナリオ (14 段落、ja→en):

| 構成 | 通し | 2 段落目以降の出始め | 再利用率 |
|---|---|---|---|
| P0 (直前 2 組、再利用なし) | 27.9 tok/s | 777 ms | — |
| 直前 2 組、再利用あり | 30.0 tok/s | 654 ms | 17% |
| 新方式 4000 文字、再利用なし | 15.8 tok/s | 1987 ms | 0% |
| **新方式 4000 文字、再利用あり (既定)** | **34.2 tok/s** | **320 ms** | 86% |
| 同 再現性設定 (f32、投機なし) | 19.3 tok/s | 447 ms | 86% |

- 訳文は再利用あり/なしで、既定設定・再現性設定とも 14 段落すべて同一 (この文書では既定設定と再現性設定の訳文も一致した)。`bench set` は参照と 19/19 一致。テスト全通過。
- 目標 (36 tok/s、0.3 s) にはわずかに届かない。残りは、窓が動かない段落でも新しい 40〜80 トークンの処理に 1 パス 140〜155 ms かかること (9〜63 行の区間。行列演算ユニットの GEMM は 64 行タイルで、行数が少ないと N/64 個のワークグループしかなく GPU が埋まらない)。P4 の中間カーネルで扱う。ほかに、前の訳を再トークン化すると生成時と別のトークン列になる場合があり、そこから先は再計算になる (14 段落中 1 回)。
- 長い文書では出力トークンと履歴で文脈が伸びるので、デコード時の attention が少し重くなる (デコードのみの速度 44.6 → 42.8 tok/s)。

### 11.16 P3 の結果 (計画変更: 1 投入化は不採用、カーネル改善に置き換え)

**1 投入化 (11.6) は効果が無く、取り消した。** デバイス上のトークンスロット (argmax の結果を GPU 上に残し、次の下書きステップや検証パスの埋め込みが直接読む) で 1 ラウンドを 1 回の投入・1 回の読み戻しにまとめたが、19 件セットで 48.4 → 47.2 tok/s とかえって遅くなった。GPU 待ちは 1 ラウンド 83.8 ms で、下書き 5 × 7.7 ms + 検証 約 45 ms の GPU 時間とほぼ一致し、11.1 で「CPU 往復など約 10 ms」と見積もった分は実際には GPU の実行と重なっていた (見積もりの誤り)。途中で読み戻さないため停止トークンの後も下書き・検証を続けてしまう分だけ損になった。パッチは保存したがコードには入れていない。

代わりに、プロファイルで見つかった次の 2 点を直した。

- **下書き (0.8B、Q4) の小さい行列**: 1 レーンが 32 重みのブロック (16 バイト) を読む Q4 専用の T=1 GEMV を追加し、ワークグループあたりの行数を N に応じて 8/16 にした (o_proj 1536×2048: 53 → 42 µs、down 1536×6144: 105 → 98 µs、lm_head: 1219 → 1169 µs)。gate/up の SwiGLU 融合版は同じ方式だと遅くなったので従来のまま。下書き単体 131 → 137 tok/s。
- **長い文脈でのデコードの attention**: 従来のカーネルはクエリヘッドごとに 1 ワークグループ (2B で 24 個) が全キーを順に処理するので、文脈 1024 で 1 呼び出し約 0.8 ms、2B の 1 トークンあたり約 20 ms かかっていた。8 行以下のパスでは分割版 (flash-decoding) を使う: (行, KV ヘッド, 64 キーのチャンク) ごとに 1 ワークグループで、KV ヘッドを共有するクエリヘッドをまとめて処理し (K/V を 1 回だけ読む)、2 つ目のカーネルでチャンクを合成する。ヘッド数 G をコンパイル時定数にしないと累積用の配列がスクラッチに落ちる (97 → 52 µs)。KV ヘッドが 1 つで文脈が短いと分割版のほうが並列度が低いので、`2 × チャンク数 × KV ヘッド数 ≥ クエリヘッド数` のときだけ使う。文脈 1024 のデコード 20.4 → 25.4 tok/s。

| 指標 | P2 | P3 | 目標 |
|---|---|---|---|
| 19 件セット (投機あり) | 48.4 tok/s | 49.4 tok/s | 60 (未達) |
| 文書シナリオ 通し | 34.2 tok/s | **37.6 tok/s** | 36 (P2 の目標、達成) |
| 文書シナリオ デコードのみ | 42.8 tok/s | 48.3 tok/s | — |
| 2 段落目以降の出始め | 320 ms | 319 ms | 300 (未達) |

- 訳文: 19 件セット (両設定) と文書シナリオとも P2 と同一。分割 attention は文脈 601 のデコードで CPU と 6.6e-5 で一致 (f32 プリフィル時)。
- 単一ストリームの投機的デコードは、1 ラウンドが「検証 約 45 ms + 下書き 5 × 約 7.3 ms」の GPU 時間で律速されている。60 tok/s に必要な削減 (約 25%) は、残りの改善候補 (下書き lm_head の語彙絞り込み 約 7%、検証の複数行 GEMV の 1 行比 +23% の解消 約 8%) を全部合わせても届かない見込み。

### 11.17 P4 の結果 (スケジューラと同時処理)

- `yaxus-model::engine`: `Engine` が複数のリクエストを `step()` ごとに 1 パスずつ進める。
  - プリフィル: 待っているリクエストのプロンプトを古い順にまとめて 1 パスで処理する (上限は、空いていれば 512 行、デコード中なら 64 行で、デコードと交互)。
  - デコード: デコード中のリクエストを 1 パス最大 8 本 (総当たり) で 1 トークンずつ進める。
  - 投機的デコード: デコード中が 1 本以下なら、その 1 本を投機ラウンドで進める。下書きは本体の消費済みトークン列から追いつくので、途中で投機と通常のデコードを行き来できる。
  - `Speculative` は廃止し、`Generator` トレイトは `submit`/`step` が中心 (`generate` はその上の既定実装)。
- サーバのワーカーはジョブをエンジンに投入し、イベントをリクエストごとのストリームデコーダ経由で配信する。クライアントの切断はトークン配信時に検知して取り消す。`--max-streams` (既定 16)、`--decode-rows` (既定 8)。
- **行数ごとのカーネル**: 複数行 GEMV を 16 行まで生成し、17〜20 行は 2 回に分けて実行する (それ以上は 64 行タイルの GEMM)。9 行 59 ms、16 行 114 ms、17 行 101 ms (GEMM は 140 ms 前後)。重みをレジスタに抱えて活性値を 4 個ずつ流す版も試したが、16 行で 281 ms と悪化したので不採用。
- **1 パスの行数**: 行あたりの時間は 8 行 (46 ms、5.8 ms/行) が最小で、16 行 (114 ms、7.1 ms/行) より 8 行 × 2 パスのほうが速い。そのためデコードは 1 パス 8 本まで。
- `bench streams` (文セットの先頭 N 件を同時投入、`--gen N` で全員が同じトークン数を生成):

| 並列数 | デコードのみ (`--gen 64`) | 通し (`--gen 64`、プリフィル込み) | 文セット (投機あり、通し) |
|---|---|---|---|
| 1 | 26.5 tok/s | 26.5 tok/s | 48.0 tok/s |
| 8 | **161.6 tok/s** | 139.7 tok/s | 68.3 tok/s |
| 16 | 161.6〜176.4 tok/s | 150.7 tok/s | 80.2 tok/s |

  文セットは訳の長さがばらばらで、最後に長い 1 件が残る (その間は投機ありの 1 本) ので、通しの値は低めに出る。
- 合格判定: 8 並列 150 tok/s 以上は **デコードのみで達成** (161.6)。16 並列 220 tok/s は **未達** (8 行パス × 2 で頭打ち、11.12 のリスクどおり)。訳文は、`concurrent_matches_solo` テスト (9 件同時、下書き切り替え・プリフィル割り込みを含む) と `bench streams --compare` (既定・再現性設定とも 16/16)、サーバ経由の 8 件同時で、すべて単独実行と同一。単独実行の性能は変わらない (文セット 50.3 tok/s、文書 37.6 tok/s)。

### 11.18 P5 の結果 (文書 API と履歴の扱いの評価)

- `yaxus-runtime::document` (当初は `yaxus-server::document`): `DocJob` が文書を連続する段落の **チェーン** に分ける (既定はチェーン数 = 段落数 / 4、最大 8)。チェーン内は前の段落を履歴にして順に訳し (`HistoryWindow`、4000 文字)、チェーン同士はエンジン上で並列に流す。空の段落は空の訳で完了し、履歴にも入れない。`run_document` (ローカル用ドライバ) とサーバのワーカー (段落リクエストが終わるたびに同じチェーンの次の段落を投入) が同じ `DocJob` と `Session` を使う。
- API: `POST /v1/translate/document` (`segments`、共通オプション、`chains`、`history_chars`、`neighbor_context`、`stream`)。SSE は `{"segment": i, "delta": ..}`、`{"segment": i, "end": true}`、最後に `{"done": true, "usage": {...}}`。各段落は受付時にテンプレートで検証し、不正なら 400。
- CLI: `translate` と `client` は入力の段落数によらず文書経路を使い (`--chains`、`--history-chars`、`--neighbor-context`)、並列に届く段落を順番どおりに表示する (先頭の未完了段落だけ逐次表示し、後続はバッファ)。クライアントの出力はローカルで同じチェーン数を指定した場合と一致。Web UI も文書 API に切り替え、複数の段落が同時に埋まっていく。
- **履歴の扱いの評価** (逐次 (a) との比較、技術文 = モデルカード 14 段落 / 『羅生門』29 段落):

| 方式 | 逐次と同一の段落 | 平均類似度 | 速度 (逐次比) |
|---|---|---|---|
| (b) 4 チェーン | 8/14, 12/29 | 0.96, 0.95 | 1.5〜1.7 倍 |
| (c) 8 チェーン | 6/14, 11/29 | 0.91, 0.94 | 2.1〜2.6 倍 |
| (d) 履歴なし・完全並列 | 5/14, 3/29 | 0.91, 0.88 | 2.5〜3.3 倍 |
| (e) 履歴なし + 前後の原文を context | 4/14, 2/29 | 0.93, 0.90 | 1.7〜2.1 倍 |

  主要な訳語の揺れはどの方式でも小さい (「下人」はほぼ servant、「老婆」は old woman / woman)。履歴なしでは文脈に依存する台詞を取り違えることがある (「この髪を抜いてな…鬘にしようと思うたのじゃ」を逐次は "I intended to pluck this hair … and turn it into a wig"、履歴なしは命令文 "Pull out this hair …")。前後の原文を渡すと部分的に改善するが、速度の利点はほぼ消える。
  **既定は自動チェーン + チェーン内の履歴** とし、履歴なしの完全並列 (`chains` を段落数以上、`history_chars: 0`) は速度優先のオプションとする。
- 長い文書 (『羅生門』+『吾輩は猫である』の抜粋 40 段落、ja→en) の通し速度:

| 構成 | 通し | 時間 | P0 比 |
|---|---|---|---|
| P0 (逐次、直前 2 組、P0 のコミットをビルド) | 17.4 tok/s | 277 s | 1.0 |
| 1 チェーン (履歴 4000 文字) | 25.9 tok/s | 188 s | 1.5 |
| **自動 = 8 チェーン (既定)** | **53.5 tok/s** | **91 s** | **3.1** |
| 履歴なし・完全並列 | 87.6 tok/s | 56 s | 5.0 |

- 合格判定: 4 倍以上は **履歴なしでのみ達成** (5.0 倍)、既定では 3.1 倍で未達。8 チェーンが伸びきらないのは、各チェーンが段落ごとにプリフィル (この文書で再利用できない新規分は合計約 8200 行、約 20 秒) を挟み、その間はデコードのパスに 8 本そろわないため。デコードとプリフィルの行を 1 パスに混ぜれば重みの読み出しを共有できるが、混ぜたパスのデコード行は行列演算ユニット (f16) を通ることになり、11.9 の再現性の方針に反する。見込める改善も約 2 割なので見送った。

### 11.19 P6 の結果 (下書き側の実験)

11.10 のうち小さい行列の帯域と decode の attention は P3 で対応済み。残りの 2 項目を実測した。

**再翻訳 (採用)**。Web UI で段落を直して訳し直すと、文書全体が再び送られてくる。この場合を速くする仕組みを 2 つ入れた。どちらも訳文は変えない。

- **応答キャッシュ** (`Engine`): 終わったリクエストの出力を、プロンプトと停止トークンをキーに LRU で 256 件保持する。同じリクエストはパスを一切回さずに即答する。`max_new_tokens` で途中まで切った出力は、より大きい上限の要求には使わない。`--no-kv-cache` または `--no-response-cache` で無効になる。
- **ヒント** (`Request.hint`): 「出力に現れそうなトークン列」を受け取る。直近の出力の末尾 3, 2, 1 トークンが一致するヒント上の位置 (1 トークン一致は想定位置の近くに限る) から、続く最大 7 トークンを下書きと同じように 1 パスで検証する。下書きモデルは動かさない。ヒントの末尾には、直前に止まったリクエストの停止トークンを足す。こうすると出力の終わりもヒントで検証でき、下書きモデルの追いつき (プロンプト全体のプリフィル、約 200 ms) が要らない。出力がヒントから外れたら、4 回までは 1 行のデコードでヒントへの再合流を待ち、それでも合流しなければ下書きモデルに切り替える (この切り替えで 8.50 → 8.01 s)。下書きモデルがなくてもヒントは使える。
- **翻訳メモリ** (`yaxus-runtime::memory`、当初は `yaxus-server::memory`): 原文・対象言語・原文の言語・文体をキーに、前回の出力トークンを 4096 件保持する (履歴、context、用語集はキーに含めない)。`Session` が各リクエストのヒントとして渡す (サーバのワーカーと `run_document`)。前の段落を直すと後続段落のプロンプト (履歴) が変わって応答キャッシュには当たらないが、翻訳メモリのヒントがほぼそのまま当たる。
- `bench doc --edit N`: 同じエンジンで文書を訳したあと、段落 N の最後の文を削って再び訳す。

| 再翻訳の例 | 1 回目 | 2 回目 | 内訳 |
|---|---|---|---|
| 技術文 14 段落、1 チェーン、段落 5 を編集 | 20.9 s | **8.0 s** | 段落 1–4 はキャッシュから即答、6–14 はヒントで検証 (8 トークン/約 60 ms) |
| 40 段落、自動 8 チェーン、段落 20 (チェーン末尾) を編集 | 90.5 s | **0.77 s** | 39 段落をキャッシュから即答 |
| 40 段落、自動 8 チェーン、段落 16 (チェーン先頭) を編集 | 90.5 s | **4.45 s** | 35 段落をキャッシュから、17–20 はヒント |

- 段落 16 を編集した 2 回目の訳文は、編集後の文書を新しいプロセスで最初から訳したものと完全に一致した。
- GPU テスト `hints_and_response_cache_keep_output` でも、ヒントが正しい・一部違う・ずれている・無関係のいずれの場合も、出力はヒントなしと同一だった (下書きの有無、上限がラウンドの途中で切れる場合を含む)。同じテストで、キャッシュからの応答がパスを回さないことも確認した。
- 19 件セットは 50.1 tok/s・19/19 同一で、変化なし。
- サーバでも、同じ単文の 2 回目は 17 µs で返り、同じ文書の 2 回目は 0 ms で終わった。

**下書きの `lm_head` の語彙絞り込み (不採用)**。下書きの argmax を候補集合に制限して、採用率を CPU 側で模擬した。候補集合は、トークナイザのスコア上位 K 個に、プロンプトと本体の出力に出たトークンを加えたもの。

| K | 下書きの argmax が集合外 | 採用率 (19 件セット) |
|---|---|---|
| 制限なし | — | 81% |
| 2048 | 53% | 16% |
| 8192 | 37% | 29% |
| 32768 | 22% | 47% |

スコア上位は短いトークンに偏り、実際の出力に出る長い語のトークンを取りこぼす。削減できるのは最大でもラウンドの約 7% (`lm_head` 1.17 ms × 5) なので、実装しなかった。

### 11.20 KV キャッシュの f16 化 (試行して取り消し)

- KV の読み書きをする 3 種類のカーネルに、2 値を 1 つの u32 に詰める f16 版を作って測った (コミット 1c75acc、ae25286 で取り消し)。
- 速度: デコードはほぼ変わらない (文脈 2802 で 24.2 → 24.5 tok/s)。プリフィルはプロンプト 2048 で 166 → 250 tok/s、2802 で 129 → 214 tok/s。文書シナリオは +2〜2.5%。
- メモリ: アリーナ 16384 トークンで、2B が 1.61 → 0.81 GB。
- 精度: CPU との差が、プロンプト後 8.8e-5 → 1.0e-2、デコード 8.4e-5 → 5.4e-2 に増えた (デコード行にも乗る)。訳文はこれまでの比較で変わらなかった。
- プリフィルが速くなった理由は、プリフィルの attention が (行, ヘッド) ごとに全キーを読み直していたこと (2048 行で GPU 時間の 64.5%)。精度を落とさないタイル化 (11.21) で対処するほうがよいので、f16 化は取り消した。
- 文書シナリオのデコードでは、8 本同時の attention (`attn_split6`、行ごとに別の系列の KV を読むので帯域律速) が GPU 時間の約 10% を占める。f16 KV が効くのは主にここなので、将来メモリや多数ストリームの長文脈を優先するときの選択肢として記録しておく。

### 11.21 プリフィル attention のタイル化

- `attn_tile`: 9 行以上のパスで使う。`begin_pass` が、同じ系列で位置が連続する行を 32 行ずつの **タイル** にまとめる。1 ワークグループ (128 スレッド) が (タイル, キーのチャンク, クエリヘッド) を担当し、K/V を 32 キーずつ LDS に載せてタイルの全行で共有する。K/V の読み出しは、行ごとからタイルごと (最大 1/32) になる。
  - スコア: 1 スレッドが 2 行 × 4 キー。softmax: オンライン softmax、1 行を 4 レーンで分担し subgroup shuffle で集計。出力: 1 スレッドが 8 行 × 4 次元。LDS の行は 33 vec4 に詰め物してバンク衝突を避ける。全体で約 38 KB。
  - 計算はすべて f32。8 行以下は従来どおり分割 attention (P3)。
  - ワークグループが少ないとき (長い文脈に短いパスを足す場合) は、キーをチャンクに分け (目安は合計 384 ワークグループ)、`attn_combine` で合成する。スライディングウィンドウの層は、タイルのキー範囲が「ウィンドウ + 31」まで広がるのでそれを含めて分割する。効果は数 % (64 行、文脈 1000 で 208 → 204 ms)。
- 速度:

| 測定 | 従来 | タイル化 |
|---|---|---|
| プリフィル、プロンプト 1024 / 2048 / 2802 (tok/s) | 308 / 166 / 129 | **403 / 352 / 326** |
| 2048 行のプリフィル中の attention | 8.15 s (64.5%) | 1.48 s (24.8%) |
| 64 行のパス、文脈 1000 / 2000 | 271 / 555 ms | 204 / 251 ms |
| 技術文 14 段落、1 チェーン | 37.6 tok/s | 38.3 tok/s |
| 40 段落、自動 8 チェーン | 53.8 tok/s (90.3 s) | 56.6 tok/s (85.8 s) |

  2048 行で attention は約 420 GFLOP/s (f32 の SIMT)。さらに速くするには、行列演算ユニットを使うか、レジスタのブロック化を増やす必要がある。
- 精度: f32 GEMM では、従来カーネルと同じく CPU と一致する (`tiled_attention_matches_cpu`: 2600 トークンの後に 40 行足した場合で、従来 4.25e-4、タイル 4.27e-4、チャンク分割を強制した場合 4.27e-4)。f16 GEMM のプリフィルでは、CPU との差は f16 のばらつき (600 トークン、6 通りのプロンプトの平均で、従来 1.8e-2、タイル 2.2e-2。どちらが大きいかはプロンプトで入れ替わる) で、`gpu_matches_cpu_long_prompt` のしきい値を f16 時だけ緩めた。
- 訳文: 19 件セット (既定・再現性設定とも 19/19)、文書 14 段落・40 段落とも従来と同一。

### 11.22 8B 対応 (旧 M4)

- 8B (`pfnet/plamo-3-translate-8b`: hidden 4096、32 層、クエリヘッド 32 / KV ヘッド 4、intermediate 16384) は、アーキテクチャ・トークナイザ・参照実装とも 2B と同じで、コードの変更なしで動いた。`--model 8b` (`0.8b`/`2b`/`8b`/`31b` の短縮名) で指定できる。
- 正しさ: HF の f32 参照で `tests/golden/logits_8b.jsonl` を作り (`YAXUS_MODEL_DIR` と `YAXUS_GOLDEN` で既存テストを 8B に向けられる)、CPU の bf16 が上位 20 logits で 3e-5 以内、greedy 完全一致。GPU の bf16 も greedy が一致した。Q8 の CPU は 3 件中 2 件が途中から言い回しの違う正しい訳になった (「お願いします」と「お願いいたします」、「会」と「将」)。19 件セットでは Q8 + 下書きが 19/19、bf16 と同一。
- 直した問題:
  - **読み込み中のメモリ**: `create_buffer_init` の一時バッファは次の送信まで解放されず、読み込み中に重みの 2 倍のメモリを使っていた。8B の bf16 (17 GB) が TTM の上限 (30.3 GiB) を超えて「Not enough memory for command submission」で落ちていた。512 MB ごとに送信して待つようにして解消し、読み込みも 11 → 7 秒に縮んだ。
  - **量子化スケールの配置**: 8B の `down_proj` (K = 16384) は 1 行のスケールが 512 個 (1024 バイト、2 の累乗) になる。行列演算ユニットの GEMM のタイルの 64 行が衝突するアドレスからスケールを読み、Q8/Q4 だけが bf16 の 1.7〜2.8 倍遅かった。重みの行に詰め物を入れても改善せず、スケールの行に 8 個 (16 バイト) 詰め物を入れたら直った (ブロック数が 256 の倍数の行だけ、`pad_scales` / `sstride`)。4096×16384 で、40 行は 12.3 → 5.5 ms、512 行は 61.7 → 37.5 ms。8B の 19 件セットの平均プリフィルは 827 → 594 ms。
- 速度 (19 件セット、`bench set`):

| 構成 | デコード | bf16 と同一 |
|---|---|---|
| bf16、下書きなし | 4.7 tok/s | (基準) |
| Q8、下書きなし | 8.3 tok/s | 19/19 |
| Q8 + 0.8B 下書き | **23.7 tok/s** | 19/19 |
| Q4、下書きなし | 13.8 tok/s | 13/19 |
| Q4 + 0.8B 下書き | 28.4 tok/s | 13/19 |
| Q8 + 2B 下書き | 16.8 tok/s (採用率 87%) | 19/19 |

  旧 M4 の合格条件「8B Q4 で 9 tok/s 以上」は満たした (下書きなしで 13.8 tok/s)。下書きは 0.8B が最適 (2B は採用率が高いが下書き自体が重い)。文書 14 段落 (技術文、1 チェーン) は 17.4 tok/s。

### 11.23 Ollama との比較

- Ollama 0.30.8 (インストール済み) は PLaMo 3 のトークナイザに未対応だった。そこで v0.40.16 を別ポートで動かし (`OLLAMA_VULKAN=1 OLLAMA_IGPU_ENABLE=1 OLLAMA_FLASH_ATTENTION=1 OLLAMA_NUM_PARALLEL=8`、KV は f16)、公式 GGUF の Q4_K_M を使った。GPU (Vulkan) で動いていることを `ollama ps` で確認した。
- 計測は `tests/bench/compare/compare.py`。両方に同じ方法で送る (Ollama には HF のチャットテンプレートで作った PFML3 プロンプトを `raw` で、yaxus には `/v1/translate`)。文書はクライアント側で、yaxus と同じチェーン分割と `HistoryWindow` を使う。yaxus は `--no-response-cache`。greedy、最大 256 トークン。
- 時間 (秒、少ないほど速い):

| 測定 | 2B: yaxus Q8 | 2B: yaxus Q4 | 2B: Ollama Q4_K_M | 8B: yaxus Q8 | 8B: yaxus Q4 | 8B: Ollama Q4_K_M |
|---|---|---|---|---|---|---|
| 文セット 19 件 | 9.74 | **8.82** | 9.91 | 25.5 | **25.2** | 28.5 |
| 文書 14 段落 (逐次) | 49.1 | 45.2 | **39.9** | 110.7 | **100.8** | 113.2 |
| 文書 40 段落 (8 チェーン) | 85.0 | 70.9 | **70.5** | 241.6 | 210.7 | **176.4** |
| 同時 8 件 | 2.24 | **1.95** | 2.82 | **7.04** | 7.14 | 7.91 |
| 同時 16 件 | 2.70 | **2.30** | 4.01 | 8.36 | **7.56** | 10.79 |

- 訳の品質 (同じ yaxus の bf16・下書きなしの出力を基準に、完全一致の数 / 平均類似度):

| 測定 | yaxus Q8 | yaxus Q4 | Ollama Q4_K_M |
|---|---|---|---|
| 2B 文セット | 18/19, 1.000 | 14/19, 0.966 | 15/19, 0.972 |
| 2B 文書 14 段落 | 6/14, 0.975 | 1/14, 0.890 | 1/14, 0.929 |
| 8B 文セット | 19/19, 1.000 | 13/19, 0.957 | 18/19, 1.000 |
| 8B 文書 14 段落 | 10/14, 0.996 | 0/14, 0.893 | 1/14, 0.949 |

  文書は前の段落の訳を履歴に使うので、1 か所の違いが後の段落に広がる。
- 分かったこと:
  - 1 本のデコード速度は同等。2B の Q4 で、yaxus は下書きなし 40〜44 tok/s、Ollama は 44〜47 tok/s。
  - yaxus が速いのは、同時リクエスト (16 件で 1.4〜1.7 倍) と、下書きがよく当たる短文。
  - Ollama が速いのは、逐次の文学的な文書 (2B) と、プリフィルの多い 8B の 8 チェーン文書。
    - 前者: 下書きの採用率が低く (約 47%)、Q4 本体では投機がかえって損になっていた (下書きありで 38.9 tok/s、下書きなしで 40.3)。
    - 後者: 平均プリフィルが yaxus 3.7 s、Ollama 2.3 s で、llama.cpp の行列演算 (8B、Q4_K_M) が速い。
  - 品質は Q8 が bf16 に最も近い。4 ビット同士では、Q4_K_M (6 ビットの部分を含む k-quant) のほうが yaxus の Q4 (Q4_0 相当) より bf16 に近い。
- 改善の候補:
  1. 投機の自動調整 (採用率が低いときは下書きを止める、k を変える)。
  2. 8B のプリフィル GEMM の高速化。
  3. 4 ビットの形式を k-quant 相当にする (または GGUF の Q4_K_M を直接読む)。

### 11.24 GGUF の読み込み

- `--model` に GGUF ファイルのパス、`REPO:QUANT` (例 `pfnet/plamo-3-translate-2b-GGUF:Q4_K_M`)、`SIZE:QUANT` (例 `2b:Q4_K_M`) を指定できる。GGUF リポジトリを量子化を付けずに指定すると、選べるファイルの一覧をエラーで示す。設定 (`plamo3.*` メタデータ) とトークナイザ (`tokenizer.ggml.*`) は GGUF から作るので、HF のディレクトリは要らない。
- 実装:
  - `yaxus-model::gguf` (ヘッダとメタデータの読み取り、llama.cpp の名前から HF の名前への対応) と `yaxus-model::ggml` (CPU での展開、GPU 向けの並べ替え)。
  - `TensorRef` が型 (`GgmlType`) を持ち、どの型でも bf16/f32 に展開できる。
- llama.cpp の変換器は、RMSNorm の重みにオフセット (1.0、0.2、0.0894) を足して保存している。`blk.0.attn_norm` の平均が 0.5 を超えるときは足されているとみなし、読み込み時に引き戻す。gate/up は `ffn_up` に結合された順番のまま、qkv も並べ替えなしだった (いずれも safetensors との比較で確認)。
- `--quant auto` (新しい既定):
  - GGUF の行列は GPU でファイルの形式に近いまま持つ。
    - Q8_0 → Q8、Q4_0 → Q4 (どちらも値は同一)。
    - Q4_K → **Q4M** (32 個ごとに f16 のスケールと最小値、4 ビット)。
    - Q6_K → **Q8B16** (16 個ごとに f16 のスケール、int8)。
    - Q4_K/Q6_K は「6 ビットの整数 × f16」のスケールを f16 に丸めるだけで、行列演算の誤差は 1e-3 未満 (`ggml_matmul_kernels_match_host`)。
  - Q2_K / Q3_K / Q5_K / Q4_1 / Q5_x / F16 / BF16 は、CPU で展開して Q8 にする (メモリはファイルより大きくなる)。 Q2_K と Q3_K は 11.32 から GPU でそのまま読む。
  - safetensors はこれまでどおり Q8。CPU バックエンドは常に Q8。
  - `--quant q8/q4/bf16` を明示すると、展開してから指定の形式にする。
  - 埋め込み表は bf16 に展開する。
  - 下書きに GGUF を使うときは、`--draft-quant q4` でもファイルの 4 ビットをそのまま使う。`--draft auto` は、safetensors の 0.8B がなければ `0.8b:Q4_K_M` を探す。 11.32 から既定は `auto4` で、`--draft auto` は Q2_K ファイルを最初に探す。
- 検証:
  - 0.8B の Q2_K / Q3_K_M / Q4_0 / Q4_K_M の全テンソルを safetensors と比べた (`tests/gguf.rs`)。相対誤差は Q2_K 30%、Q3_K 15〜18%、Q4_x 8〜10%、Q5_K 3.8%、Q6_K 1.9%、norm 0 と、ビット数どおりだった。
  - GGUF のトークナイザは、golden の全テキストで `tokenizer.jsonl` と同じ結果になる (`gguf_tokenizer_matches_jsonl`)。
- 品質と速度 (19 件セット、bf16 と同一の件数):

| 構成 | 2B | 8B |
|---|---|---|
| yaxus、GGUF Q4_K_M | **16/19** | 17/19 |
| Ollama、同じ GGUF | 15/19 | 18/19 |
| yaxus、独自の Q4 | 14/19 | 13/19 |
| デコード (yaxus GGUF、下書きなし / 0.8B 下書き) | 33.5 / 54.4 tok/s | 11.3 / 26.5 tok/s |

  同じ重みでも Ollama とは 2B で 4 件、8B で 2 件の訳が違う (llama.cpp は活性値を量子化して内積を取る経路を使うため、と考えられる)。
- 最初の版は Q4_K を Q4M (0.625 バイト/重み)、Q6_K を Q8B16 (1.125 バイト/重み) に広げて持っていたので、読み出し量がファイルより約 2 割多かった。11.25 でファイルと同じビット数の形式に置き換えた。

### 11.25 GGUF 用カーネルの高速化

- **Q4K** (Q4_K): 4 ビットの値と、256 個ごとの Q4_K のヘッダ (d, dmin と 12 バイトの 6 ビットスケール) をそのまま持ち、カーネルの中で ggml と同じ式で展開する (0.5625 バイト/重み、値は CPU の展開と完全に同じ)。
- **Q6** (Q6_K): 32 個ごとに 24 バイト (下位 4 ビット 16 バイト + 上位 2 ビット 8 バイト) を隣接させ、16 個ごとに f16 スケール (0.875 バイト/重み)。下位と上位を行の前半・後半に分けた最初の版は、行列演算ユニットの GEMM が 1.4 倍遅かった (19.1 ms、隣接させて 12.2 ms)。
- Q4K のヘッダの行も、ブロック数が 32 の倍数 (512 バイトの倍数の間隔) なら 16 バイト詰め物する。8B の `down_proj` (K = 16384) の GEMM が 47.2 → 34.8 ms になった (11.22 のスケールと同じ問題)。
- T=1 の Q4K は、ブロック単位の `gemv1b` より通常の `gemv1` のほうがわずかに速かった (2560×10240 で 239 → 222 µs)。
- Q4M と Q8B16 は削除した。

| 2B、GGUF Q4_K_M | 旧形式 (Q4M/Q8B16) | 新形式 (Q4K/Q6) |
|---|---|---|
| デコード、文脈 32 / 1024 (下書きなし) | 35.2 / 32.5 tok/s | **40.6 / 37.4 tok/s** |
| 19 件セット (下書きなし / 0.8B 下書き) | 33.5 / 54.4 tok/s | **38.5 / 56.5 tok/s** |
| プリフィル、プロンプト 1024 | 417 tok/s | 420 tok/s |
| `lm_head` (Q6_K、1 トークン) | 3.75 ms | 2.98 ms |

| 8B、GGUF Q4_K_M | 旧形式 | 新形式 |
|---|---|---|
| 19 件セット (下書きなし / 0.8B 下書き) | 11.3 / 26.5 tok/s | **12.9 / 27.5 tok/s** |
| 平均プリフィル (19 件セット) | 603 ms | 654 ms |
| プリフィル 40 / 512 行 | — | 640 / 3943 ms (独自 Q4 は 657 / 3864 ms) |

- 訳文は旧形式と 19/19 同一 (2B・8B とも)。
- デコードの各行列積は 75〜81 GB/s で、帯域の上限に近い。2B の下書きなしは Ollama (約 46 tok/s) にまだ 1 割ほど届かない。残りは行列積以外 (attention、norm、ディスパッチの合間) で、1 トークンあたり約 6 ms ある。
- 短いプロンプトの 8B のプリフィルは、旧形式より 8% 遅い。Q4K のスケールの展開と Q6 の 2 か所の読み出しの計算が、64 行未満のタイルでは目立つため。

### 11.26 行列積以外の時間の削減

- 計り直し: 11.25 で「行列積以外が 1 トークンあたり約 6 ms」と書いたのは誤りだった。タイムスタンプ付きのプロファイル (`YAXUS_PROFILE=2`) はそれ自体で 2 割近く遅くなり、その実行時間から引き算していた。プロファイルなしの実行時間とカーネル時間の合計を比べると、2B GGUF の文脈 32 で行列積以外は約 2 ms (1 トークン 24.6 ms のうち)。ディスパッチ間の空きはほぼ無かった。ホストが次のパスを記録する約 0.35 ms (と読み戻しの待ち) の間は GPU が止まる。
- 文脈が長いと、デコードの attention が増える (文脈 1024 で 1 層 129 µs、1 トークン 3.1 ms)。次の 3 点を直した。
  1. 分割 attention のチャンクの大きさを可変 (`{CH}`、1 キーを `128 / CH` レーンで処理) にした。文脈 64 以下は 16 キー、それより長いと 32 キー。subgroup があれば、8 行以下では常に分割版を使う (行ごとの版は文脈 32 で 1 層 35.6 µs かかっていたが、分割版は 6 µs)。
  2. 値の足し込みのループを 4 キーずつ展開し、読み出しを並べて出す (32 キーのチャンクで 87 → 67 µs)。
  3. 合成 (`attn_combine`) を、最大値を逐次更新する 1 回の走査に変え、4 チャンクずつ展開した (15 → 7.6 µs)。空のチャンクは `part` を書かないので、重みだけでなく値も `select` で落とす (0 × NaN を避ける)。
  - 文脈 1024 の attention は 1 層 129 → 75 µs。
- `add_norm` を subgroup による集計と、値をレジスタに持つ版にしたが、10.1 → 9.1 µs にしかならず (1 トークン 0.05 ms)、採らなかった。ホストの記録とパスを重ねるには、トークンを GPU 上で受け渡す仕組み (P3 で見送った方式) が要る。効果は約 2% なので見送った。

| デコード (tok/s、下書きなし) | 文脈 32 | 1024 | 2048 |
|---|---|---|---|
| 2B Q8 | 26.8 → 27.2 | 25.4 → 26.1 | 24.2 → 25.2 |
| 2B GGUF Q4_K_M | 40.6 → 41.5 | 37.4 → 39.0 | 34.7 → 36.6 |
| 0.8B Q4 (下書き) | 143 → 142 | 125 → 135 (文脈 256: 123 → 140) | — |

- 文書: 技術文 14 段落 (1 チェーン) は 38.3 → 40.5 tok/s、40 段落 (8 チェーン) は 56.6 → 58.2 tok/s (85.8 → 83.5 s)。19 件セット (投機あり) は 50.3 → 50.8 tok/s。
- 訳文は 19 件セット (既定・再現性設定)、40 段落とも同一。`decode_attention_matches_cpu` (f32 GEMM、文脈 20 / 600 / 2100) で、CPU との差は 4.7e-5〜2.6e-4。
- 2B GGUF のデコードの残りの差 (Ollama 約 46 tok/s) は、ほぼ行列積 (1 トークン約 22.4 ms、平均 73 GB/s 前後) にある。

### 11.27 デコードの GEMV と、Ollama との差の分析

- GPU の読み出し帯域の上限は、単純な読み出しカーネルで 82〜84 GB/s。
- T=1 の GEMV (subgroup あり) を新しくした (`GEMV_T1`)。32 レーンの集計を LDS とバリアから shuffle に替え、活性値の読み出しを重みより先に出す。Q4 専用のブロック GEMV (`gemv1b`) も、これに置き換えた。
  - 2B GGUF のデコードは 41.4 → 42.6 tok/s (文脈 1024 で 38.9 → 40.3)。
  - 2B Q8 は 27.2 → 27.9、0.8B Q4 (下書き) は 142 → 148、2B Q4 は 45.4 → 48.3 tok/s。
- 効果がなかったもの:
  - K 方向の読み出しを 2〜4 個先に出す展開。
  - 1 組で 2〜4 行を持って活性値を共有する版 (かえって遅い)。
  - SwiGLU 融合版への同じ変更 (もとから帯域の上限)。
  - Q4K のヘッダを 20 バイトに展開して 6 ビットの取り出しをなくす版 (わずかに遅い)。
  - 複数行 GEMV の 1 組あたりの行数 (R=2 が最適で、1 や 4 は遅い)。
  - 参考: Q4K のスケールの計算を定数にすると 13% 速くなるので、ヘッダの読み出しと展開に伸びしろはある。
- Ollama (v0.40.16、llama.cpp の Vulkan) との比較。同じ GGUF (2B Q4_K_M)、同じプロンプトで greedy、Ollama の `eval_duration` で測った。
  - Ollama は 45.3 tok/s (22.1 ms/トークン)、yaxus は 42.9 tok/s (23.3 ms)。
  - 以前の「約 46 tok/s」は比較スクリプトの近似 (応答時間 − プリフィル時間) による値で、同じ条件の差はもっと小さかった。
- `GGML_VK_PERF_LOGGER=1` による llama.cpp の演算ごとの時間 (1 トークン、GPU 時間の合計 22.4 ms) と、yaxus の比較:

| 行列 | llama.cpp | yaxus |
|---|---|---|
| gate/up Q4_K 20480×2560 | 364 µs (+ GLU 5 µs) | 374 µs (SwiGLU 込み) |
| down Q4_K 2560×10240 | **184 µs** | 200 µs |
| o_proj Q4_K 2560×3072 | 58 µs | 60 µs |
| qkv Q4_K 4096×2560 | 77 µs | 約 78 µs |
| down Q6_K 2560×10240 | **272 µs** | 293 µs |
| qkv Q6_K 4096×2560 | **110 µs** | 119 µs |
| lm_head Q6_K | 3054 µs | **2975 µs** |
| attention (1 層) | 13.7 µs | 約 9 µs (分割 + 合成) |
| norm・RoPE などの合計 | 約 1.0 ms | 約 0.6 ms |

- 差 1.2 ms/トークンの内訳 (推定):
  1. **Q6_K のデータ量**: llama.cpp はファイルのまま 0.82 バイト/重み (16 個ごとの int8 スケールと 256 個ごとの f16)。yaxus の Q6 は 16 個ごとに f16 スケールを持つので 0.875 (+7%)。約 0.35 ms。
  2. **Q4_K の計算**:
     - llama.cpp (`mul_mat_vec_q4_k.comp`) は最小値の項を `dmin·m·Σx` として 16 要素ごとにまとめる。yaxus は要素ごとに `q·scale − min` を計算する。
     - llama.cpp は 6 ビットのスケールを、u32 を `unpack8` して 4 つずつ取り出す。
     - down で 8% の差、約 0.25 ms。
  3. **ホスト側の待ち**: yaxus は毎トークン、パスの記録 (約 0.33 ms) と読み戻しの間 GPU が止まる。llama.cpp は、GPU 時間の合計 (22.4 ms) と実時間 (22.1 ms、計時なし) がほぼ同じ。約 0.4 ms。
- プリフィルは yaxus のほうが速い (2B で 68 行: llama.cpp の行列積の合計 約 190 ms、yaxus の 64 行パス 約 155 ms)。

### 11.28 Q6 と Q4K の改善 (llama.cpp との差を詰める)

- **Q4K を ggml と同じ 144 バイトのブロックに**: ヘッダ 16 バイトと 4 ビット値 128 バイトを隣接させる (4 ビット値の並びは Q4 と同じ)。同じバッファを 2 つのバインディング (`wq` と `wsc`) で読む。ヘッダを別のバッファに分けていたときは、読み出しが 2 本の流れになっていた。
  - デコード: down 200 → 186 µs (llama.cpp 184)、o_proj 60 → 58 µs (同 58)。
  - プリフィル: 8B の Q4K の GEMM も 3〜13% 速くなった。
  - ヘッダ行の詰め物は不要になった。
- **Q6 のスケールを Q6_K と同じに**: 16 個ごとに int8 と、256 個ごとに f16 の `d`。行ごとに `meta` に並べる (`q6_meta_stride`)。読み出し量は 0.875 → 約 0.83 バイト/重み。
  - デコード: down 293 → 280 µs、qkv 119 → 112 µs、`lm_head` 2975 → 2792 µs。
  - スケールは `d * sc` を f32 で計算するので、ggml と値が同じになった。
- このスケールは取り出しに読み出し 2 回が要る。4 値ごとにスケールを引く行列演算ユニットの GEMM では、Q6 が 1.5 倍遅くなった (8B の 4096×16384 で 40.7 → 62 ms。スケールを定数にすると 31 ms)。
  - そこで、プリフィル用 (GEMM) には別に「16 個ごとに f16」のスケールも持たせる (`Weight::Q6::s16`、Q6 の行列で +0.125 バイト/重みのメモリ。読むのはプリフィル時だけ)。
- 効果がなかったもの:
  - **最小値の項をまとめる計算** (llama.cpp の `dmin·m·Σx`): Q4K でも Q6 でも速さは同じだった。差はデータの配置で決まっていた。
  - **GEMM で 1 スレッドが 1 行 16 値を読む形**: スケールを引く回数は 4 分の 1 になるが、全体では 8〜11% 遅かった。

| 測定 | 11.27 の後 | 今回 |
|---|---|---|
| 2B GGUF デコード、文脈 32 / 80 / 1024 | 42.6 / 42.9 / 40.3 tok/s | **44.0 / 44.1 / 41.4 tok/s** |
| 2B GGUF 19 件セット (下書きなし / あり) | 39.1 / 56.5 tok/s | **41.8 / 58.4 tok/s** |
| 2B GGUF プリフィル 64 / 512 行 | 149 / 1163 ms | 145 / 1124 ms |
| 8B GGUF 19 件セット (下書きなし / あり) | 12.9 / 27.5 tok/s | **14.2 / 28.9 tok/s** |
| 8B GGUF プリフィル 40 / 512 行 | 640 / 3943 ms | **530 / 3823 ms** |

- 同じ条件の Ollama (2B、文脈 約 80) は 45.3 tok/s。差は 5.6% → 2.7% になった。残りは、主にホスト側で GPU が止まる時間 (11.27 の 3)。
- 訳文は 2B・8B の GGUF とも 11.27 と 19/19 同一。Q8 の 19 件セットも基準と同一。

### 11.29 ホスト側の待ちの解消 (小さいパスの早期投入)

- 11.27 の 3 (毎トークン、ホストがパスを記録する約 0.33 ms と読み戻しの間 GPU が止まる) への対処。
- **GPU 上のトークン受け渡し (見送り)**: 前のトークンの argmax を GPU 上に残し、次のパスの埋め込みに直接読ませる。こうすれば、結果を待たずに次のパスを記録・投入できる。ただ、停止トークンが出たときは先に投入したパス (2B で約 23 ms) が無駄になり、1 リクエストに 1 回起きる。翻訳の出力は 20〜50 トークンが多く、節約できる量 (0.4 ms × トークン数) と同じ程度を失う。投機のラウンドでは P3 で試して遅かった (11.16)。
- **採ったのは早期投入**: 8 行以下のパス (デコード、投機の検証、下書きの各ステップ) は、最初の約 1 層 (12 ディスパッチ) を記録した時点で一度投入する (`EARLY_SUBMIT_DISPATCHES`)。残りを記録している間に、GPU は先頭の層を進める。無駄なパスは出ない。投入を何度にも分けると、投入の負担のほうが大きかった (1 層ごとや 2 層ごとでは、1 回だけのときより遅い)。

| 測定 | 11.28 | 早期投入 |
|---|---|---|
| 2B GGUF デコード、文脈 32 / 80 / 1024 | 44.0 / 44.1 / 41.4 tok/s | **45.0 / 44.8 / 42.1 tok/s** |
| 2B Q8 デコード、文脈 32 / 1024 | 27.9 / 26.8 tok/s | 28.3 / 27.1 tok/s |
| 19 件セット (Q8 + 下書き / GGUF + 下書き) | 51.0 / 58.4 tok/s | **52.4 / 60.1 tok/s** |
| 8 並列 (デコードのみ) | 141.9 tok/s | 144.1 tok/s |
| 技術文 14 段落 (1 チェーン) | 40.5 tok/s | 41.5 tok/s |

- 同じ条件の Ollama (2B GGUF Q4_K_M、文脈 約 80) は 45.3 tok/s で、差は約 1%。1 トークンの実時間 (22.3 ms) は GPU 時間の合計とほぼ同じになった。
- 訳文は 19 件セット (既定・再現性設定) とも同一。

### 11.30 投機の自動調整

- 投機のラウンドごとに、下書きの長さ k (0 は通常の 1 ステップ) を `Engine::draft_len` が選ぶ。上限は `--draft-k` (既定 7。9 行の検証パスは 8 行より大きく遅いので 7 まで)。
- **時間**: 本体の 1 ステップ、下書き 1 ステップ、行数ごとの検証パス、ラウンドのその他の手間を、指数移動平均で実測する。
  - 測っていない行数は、近い行数から 1 行あたり 4% 増しで推定する。
  - 下書きが遅れを取り戻すプリフィルを含むラウンドは測らない。含めると手間が 48 ms に見え、投機を止める → 取り戻しが増える、という悪循環になった。
- **当たり率**: 1 番目の下書きトークンの当たり率と、2 番目以降の「前がすべて当たったという条件で当たる」率を、別々に数える (`Accept`)。
  - 当たり方は連続しやすく、2 番目以降のほうが当たりやすい。深さごとに数えると、選ばれなくなった深さの数が古いまま残り (深さ 4 だけ 0.75 など)、短い k に偏った。
  - エンジン全体の数 (減衰 0.995/ラウンド) を事前分布にし、リクエストの数 (減衰 0.9) を重み 20 ラウンド分で混ぜる。
- **選び方**:
  - 期待トークン数 `1 + Σ_j P(先頭 j 個が当たる)` を、所要時間 `k × 下書き + 検証(k+1 行) + 手間` で割った値を、k ごとに比べる。
  - 既定の k=5 より 5% 以上速いと見込めるときだけ切り替える。見積もりの小さな誤差による切り替えは、得より損が大きかった。
  - 全体で 30 ラウンド分の数がたまるまでは、k=5 のまま。最初の数リクエストは、事前分布 (0.75) の影響で短い k を選びがちだった。
  - k=0 が選ばれたまま 16 ステップ進むと、k=2 で 1 回試して当たり率を測り直す。
- 試した段階と、19 件セット (2B Q8) の速さ: 1 つの当たり率 49.7 → 深さごと 47.3 → 2 種類 50.5 → 手間込み 51.1 → 切り替えの閾値と準備期間あり 52.5 tok/s (固定 k=5 は 52.4)。

| 測定 | 固定 k=5 | 自動調整 |
|---|---|---|
| 19 件セット、2B Q8 / 2B GGUF / 8B GGUF | 52.4 / 59.5 / 29.2 tok/s | 52.5 / 59.3 / 29.3 tok/s |
| 技術文 14 段落、2B Q8 | 41.6 tok/s | 41.8 tok/s |
| 『羅生門』14 段落、2B Q8 | 32.4 tok/s | 32.8 tok/s |
| 『羅生門』14 段落、2B GGUF | 36.4 tok/s (下書きなし 35.4) | **40.2 tok/s** |
| 『羅生門』14 段落、8B GGUF | 15.3 tok/s | **16.5 tok/s** |

- 当たりにくい文学的な文章 (採用率 約 50%) で 8〜10% 速くなり、当たりやすい文では固定と同じ。2B GGUF の『羅生門』では主に k=2 を選んだ (投機を止めることはなかった)。
- 訳文は変わらない (投機は greedy の検証なので、長さによらず本体の出力と同一)。19 件セット (既定・再現性設定)、『羅生門』の下書きあり・なしとも同一。

### 11.31 Ollama を超えるための検討 (2026-11-11)

**現状** (同じ GGUF Q4_K_M、`tests/bench/compare/compare.py`、秒。Ollama は v0.40.16 で 11.23 と同じ値):

| 測定 | 2B: yaxus | 2B: Ollama | 8B: yaxus | 8B: Ollama |
|---|---|---|---|---|
| 文セット 19 件 | **8.67** | 9.91 | **21.9** | 28.5 |
| 文書 14 段落 (逐次) | **37.4** | 39.9 | **89.4** | 113.2 |
| 文書 40 段落 (8 チェーン) | **66.3** | 70.5 | 197.4 | **176.4** |
| 同時 8 件 | **1.90** | 2.82 | **6.35** | 7.91 |
| 同時 16 件 | **2.23** | 4.01 | **7.07** | 10.79 |
| 下書きなしのデコード (2B) | 45.0 tok/s | **45.3 tok/s** | — | — |

Ollama が勝つのは、8B の 8 チェーン文書 (プリフィルが多い) と、下書きなしの単独デコード (1% 差) の 2 つ。

**物理的な上限**:

- メモリは DDR5-5600 の 2 チャネルで、理論値は 89.6 GB/s。GPU の読み出しは実測 82〜84 GB/s で、すでに壁に当たっている。CPU と GPU で分担しても、同じメモリバスを分け合うだけで合計は増えない。
- 2B GGUF は 1 トークンあたり 1654 MB を読む。83 GB/s なら 19.9 ms (50 tok/s) が、下書きなし・同じ重みでの上限。現状は 22.2 ms (行列積 21.1 + attention/norm 0.75 + ホスト 0.3)。伸びしろは約 8% で、Ollama も同じ壁の手前にいる。
- したがって大きく超えるには、「1 回の重み読み出しで進むトークン数を増やす」か「読む量を減らす」しかない。前者は投機と同時実行で、Ollama にはない yaxus の領分。

**候補と見込み** (効果の大きい順):

1. **複数行 GEMV の作り直し** (投機の検証、同時デコード) → f32 のままでは改善なし (11.32)。整数ドット積にして 6 行 24.9 ms・8 行 25.8 ms を達成 (11.34)
   - 2B GGUF の検証パスは 1 行 22.0 ms に対して、6 行 31.9 ms、8 行 38.0 ms。帯域だけなら数 % 増で済むはずで、6 行の演算量 (約 35 GFLOP) は GPU の能力なら 4 ms 程度。
   - 6 行 26 ms・8 行 28 ms になれば、投機が約 10%、8 並列のデコードが約 35% 速くなる。
2. **下書きを安くする** → (a) を実装、1 ステップ 6.5 → 5.1 ms (11.32)
   - 投機 1 ラウンドの読み出しは、下書き 5 × 408 MB と検証 1654 MB で、下書きのほうが多い。
   - (a) 低ビットの下書き: 当たり率は Q4 82% / Q3_K_M 84% / Q2_K 81% でほぼ同じ (19 件セット、実測)。Q2_K/Q3_K を GPU でそのまま読むカーネルを書けば、下書きの読み出しが約 25% 減る。
   - (b) 下書きの `lm_head` (1 ステップの 17%): 候補語彙に絞る。 → 出力頻度の固定集合は日本語で破綻、英訳だけなら +3〜5% (11.36)。2〜3 bit にする案は、採用率の低下で相殺されて効果なし (11.37)。11.19 ではトークナイザのスコア順の固定集合で失敗したので、今度は本体の実際の出力頻度から学習した集合 + プロンプト中のトークンを使う。採否は実測で決める。
   - (c) 小さい行列の効率: o_proj は 46 GB/s、down は 63 GB/s、attention は 24 µs。
   - 合わせて、1 ステップ 6.5 ms → 3.5〜4 ms を見込む。
3. **木構造の下書き** → 見込みは 0〜5% (多くの文では負) と分かり、見送りを提案 (11.35)
   - 下書きの各ステップで上位 2〜3 候補を別の行として同じパスで進め (行を増やしても読み出し量は同じ)、本体が枝をまとめて 1 パスで検証する (KV のフォークを使う)。
   - 文献 (SpecInfer、Sequoia) では、1 パスあたりの採用トークンが 3〜5 割増える。外れやすい文章ほど効く。
4. **プリフィルとデコードを 1 パスに融合** (8B の文書) → 効果は 1 割弱と分かり、5 を先にした (11.33)
   - チェーンの切り替えのたびに 64 行のプリフィルが重みを丸ごと読み直している (8B で 1 パス約 0.53 秒が下限)。1 回の重み読み出しで、デコード行は f32、プリフィル行は行列演算ユニット、と同時に処理するカーネルにする。
   - 精度の方針 (11.9) は保てる。11.18 の見積もりで約 2 割。
5. **プリフィル GEMM の効率** → 整数ドット積で 1.3〜1.9 倍 (11.33)
   - 現状は約 2.4〜2.7 TFLOPS。重みを 64 トークンのタイルごとに展開し直している。1 回の展開で 128〜256 トークン分を処理する形にすれば、1.5〜2 倍が狙える。
6. **下書きなしの残り 8%**: 行列積以外の 1 ms と、行列積の 5%。Ollama との差を逆転できるが、絶対値は小さい。 → ディスパッチの融合は逆効果。`add_norm` と 1 行の attention の待ち時間を縮めて、デコード +1.6〜1.8%、投機 +1〜2% (11.38)
7. **下書きモデルの蒸留** (学習が必要): 0.8B を 2B/8B の出力に合わせて追加学習し、採用率を上げる。このマシンでは学習できないので別枠。

**見込み** (2B GGUF、当たりやすい文): 現状 60 tok/s → 1 と 2 で 85〜95 tok/s → 3 を加えて 100 tok/s 超 (Ollama は 45 tok/s)。見送るもの: CPU との分担 (メモリの壁)、重みのエントロピー符号化 (効果 1 割・複雑)、本体の語彙絞り込み (厳密な greedy でなくなる)。

### 11.32 項目 1 と 2 の結果: 複数行 GEMV は見送り、下書きは Q2_K で 1.27 倍

**項目 1 (複数行 GEMV の作り直し) は改善なし。** 11.31 の「6 行 26 ms・8 行 28 ms」は達成できないと分かった。

- 見積もりの誤り: 6 行分の演算を 4 ms としたのは公称の 8.6 TFLOPS を使ったためで、この GPU の f32 の実測ピークは約 4 TFLOPS。
- 実測 (2B GGUF): 行を 1 つ増やすごとに 1.3〜1.4 ms かかる。これは演算のピーク速度そのもので、メモリ待ちとは重ならない。活性化の受け渡し (LDS) は 6 行で約 3 ms、8 行で約 5.6 ms。
- 試した形と結果:

  | 形 | 結果 |
  |---|---|
  | 活性化をレジスタに持つ | 遅くなる (レジスタがあふれる) |
  | 重みを先読みする | 遅くなる |
  | LDS を使わず活性化を直接読む | 2B GGUF の 6 行だけ 31.7 → 29.1 ms。Q8 と 8B では大幅に遅い |
  | LDS への展開を複数スライスに分ける | 遅くなる |
  | 1 レーンあたり 1・3・4 行 | 2 行 (現状) より良くならない |

- 結論: 現在の `GEMV_MULTI` (2 行/レーン、512 列ごとに LDS) を維持する。検証パスを安くする余地は演算量そのもの (行数) にしかないので、項目 3 (木構造) は「行を増やす」前提を見直す必要がある。

**項目 2 (下書きを安くする)**: GGUF の Q2_K と Q3_K をそのまま読むカーネルを追加し、0.8B の Q2_K ファイルを既定の下書きにした。

- レイアウト (`yaxus-model/src/ggml.rs`、値は GGUF と同一):
  - Q2: 2 bit を要素順に並べ (16 個で u32 1 つ)、行ごとのメタに `d`・`dmin` (256 個ごとの f16) と scale/min (16 個ごとの 4 bit ずつ) を置く。デコードのカーネルは `s * Σ(q·x) - m * Σx` の形で計算する。
  - Q3: 32 個を 12 バイト (下位 2 bit × 32 + 上位 1 bit × 32) にまとめ、スケールは Q6 と同じ形。
  - プリフィルの GEMM 用に、16 個ごとの f16 (Q2 は scale と min の組) を別に持つ (Q6 と同じ理由)。
- `Quant::Auto4` (下書きの既定、`--draft-quant auto4`): 4 bit 以下のテンソルはファイルのまま、それ以外は Q4 にする。Q2_K ファイルの出力行列は Q6_K (135 MB) で 1 ステップの 30% を占めていたので、これが効く。
- 下書き単体 (0.8B、プロンプト 32 トークンのデコード):

  | 下書き | 1 ステップの読み出し | tok/s |
  |---|---|---|
  | チェックポイントを Q4 (従来の既定) | 408 MB | 154.6 |
  | Q4_K_M ファイル | — | 143.6 |
  | Q3_K_M ファイル (Q5_K は Q8 に展開) | — | 154.2 |
  | Q2_K ファイル、出力行列は Q6_K のまま | 353 MB | 179.4 |
  | Q2_K ファイル + `auto4` (新しい既定) | 約 306 MB | 197.0 |

- 全体 (`bench set` は `--no-kv-cache`、`bench doc` は 14 段落・既定設定):

  | 本体 | 測定 | 従来の下書き | Q2_K + `auto4` |
  |---|---|---|---|
  | 2B Q8 | 文セット | 52.4 tok/s (採用 81%) | 55.9 tok/s (80%) |
  | 2B GGUF Q4_K_M | 文セット | 59.0 tok/s (82%) | 65.4 tok/s (81%) |
  | 8B GGUF Q4_K_M | 文セット | 29.6 tok/s (78%) | 29.7 tok/s (73%) |
  | 2B Q8 | 文書 | 18.79 s | 17.78 s |
  | 2B GGUF Q4_K_M | 文書 | 17.05 s | 16.21 s |
  | 8B GGUF Q4_K_M | 文書 | 39.49 s | 39.42 s |

  - 2B では 5〜11% 速い。8B では採用率が 5 ポイント下がり、下書きが速くなった分と相殺して同じ。
  - `--draft auto` は、ローカルキャッシュにあれば Q2_K ファイル、なければチェックポイント、Q4_K_M ファイルの順で選ぶ。
  - 既定設定の 19 件は参照訳と 19/19 一致 (下書きは出力を変えない)。
- 下書きのカーネルにこれ以上の余地は小さい:
  - Q2 のカーネルから演算を抜いても 196 → 199 tok/s で、演算は律速でない。
  - 1 レーンに複数行を持たせて活性化のロードを共有する形 (T=1 の GEMV と GLU、2・4 行) も ±1% で、採用しなかった。
  - 行列積全体で平均 68 GB/s。上限 (83 GB/s) まで届いても 1 ステップ 5.0 → 4.2 ms。
- 残り: (b) 下書きの `lm_head` を候補語彙に絞る。出力行列は Q4 でも 1 ステップの約 2 割 (1.0 ms、93 MB)。文字種で絞る場合、英語などラテン文字の訳先なら約半分 (ASCII のみのトークンは 107520 中 53960) になるが、日本語が訳先だと 96% が残る。効果は訳先によって全体の 0〜4% で、仕組み (要求ごとの集合、行の集め読み) が大きいので後回しにする。

### 11.33 項目 5: プリフィルの GEMM を整数ドット積にする (f16 版の 1.3〜1.9 倍)

**順序を 5 → 4 に入れ替えた理由。** 項目 4 (プリフィルとデコードの融合) の前にパス時間を測ったところ、8B の差の主因は融合で省ける重みの読み直しではなく、GEMM の演算効率だった。

- 8B GGUF のパス: 1 行 64.5 ms、7 行 112 ms、64 行 548 ms、512 行 3786 ms。64 行のうち重みの読み出しは約 65 ms で、残りは演算 (約 2.1 TFLOPS)。融合で省けるのは 1 交互あたり約 60 ms (9%)。
- Ollama (llama.cpp) のプリフィルを直接測った (`prompt_eval_duration`): 8B は 408 トークンで 1760 ms (232 tok/s、傾き 3.9 ms/トークン)、2B は 578 ms (705 tok/s、1.29 ms/トークン)。yaxus は 8B が 7.4 ms/行 (135 行/s)、2B が 2.23 ms/行 (448 行/s) で、しかも 64 行のタイル単位 (32 トークンでも 1 タイル分)。
- 融合カーネルは GEMM の上に作るものなので、GEMM を先に作り直した。

**llama.cpp が速い理由。** 活性化を int8 (32 個ごとにスケールを持つ Q8_1) に量子化し、重みの整数値と int8 × int8 → int32 で掛けている (`mul_mmq_cm1.comp`、RDNA3 では整数の行列演算ユニットを使う経路)。Ollama v0.40.16 の `libggml-vulkan.so` に `matmul_q4_k_q8_1` と `cm1_accumulate` が入っていることを確認した。f32/f16 の FMA は 1 命令で積和 1 つ、整数ドット積は 1 命令で 4 つなので、演算律速のプリフィルではそのまま差になる。

**yaxus での実現** (`yaxus-backend-wgpu/src/gemm_int.rs`):

- naga の cooperative matrix は f16/f32 だけで、int8 の行列は WGSL で書けない。一方 `dot4I8Packed` は WGSL の標準関数で、wgpu 30 は Vulkan の `OpSDot` (`VK_KHR_shader_integer_dot_product`) にそのまま落とす。行列演算ユニットなしで同じ密度が得られる。
- 前処理 `xquant`: 活性化を行ごと・32 個ごとに `max|x|` で正規化して int8 にする。
  - 1 プレーン (`int8`): 7 bit + 符号。llama.cpp と同じ精度。
  - 2 プレーン (`int16`): 15 bit + 符号を `128·hi + lo` に分け、ドット積を 2 回取る。
  - 並びは [ブロック][4 トークンの組][プレーン][8 u32] の `vec4<u32>`。GEMM は 1 回のロードで 4 トークン分を読む。スケールと、min 項に使うブロックの和 (丸める前の値の和) も同じ並びで持つ。
- GEMM: 1 ワークグループが 64 出力 × 128 トークン (端数用に 64・32 トークン版)、1 レーンが 4 出力 × 8 トークン。
  - 重みは 32 列ごとに整数値のまま LDS に展開する (Q4_K はニブルを符号付きバイトにするだけで、1 u32 あたり 3 命令)。活性化はバッファから直接読む。
  - スケールのブロックごとに整数の和 (厳密) を f32 に直して足す: `y += f32(和) × 重みのスケール × 活性化の単位 − 重みの min × 活性化の和`。
  - 重みの値は変えない (GGUF の整数値とスケールをそのまま使う)。丸めるのは活性化だけ。
  - `q × scale − min` の形式 (Q4_K、Q2_K、自前の Q4) は、q を中心に寄せて (Q4_K なら q − 8) min を合わせる。活性化の丸め誤差に掛かる値が 0〜15 から −8〜7 になり、誤差が約半分になる。
  - 対応: Q8、Q4、Q4_K、Q6_K、Q3_K、Q2_K。bf16 は整数値を持たないので従来の f16/f32 の GEMM。
  - WGSL は Rust で生成し、初回の使用時にコンパイルする (1 モデルが使うのは 2〜3 形式)。
- 16 行以下のパスは従来どおり f32 の GEMV (デコード行は常に f32、11.9 の方針のまま)。17 行以上が整数 GEMM に入る (17〜20 行を GEMV 2 回に分ける経路は、整数 GEMM のほうが速いので使わない)。
  - **訂正 (11.34 で判明)**: このときのコードは境界が 8 行になっていて、9〜16 行のパスも整数 GEMM に入っていた (2B で 12 行 78 ms。f32 の GEMV なら 71 ms)。同時 9 件以上のデコードがこれに当たる。11.34 で経路ごと置き換えた。

**作る過程で分かったこと:**

- ループで書くと f16 版と同じ速度にしかならない (RADV は 3 重ループと配列を展開しない)。手で展開した形にして 1.7 倍になった。生成器にしたのはこのため。
- LDS の `vec4` 読み出し 1 回は、ドット積約 6 回分の重さ。活性化を LDS に展開せずバッファから直接読む形にして、20480×2560・256 トークンが 6.55 → 4.96 ms になった。
- `xquant` の出力を K 方向優先に飛び飛びで書くと 7 倍遅い (63 → 441 µs)。書き込みと GEMM の読み出しが両方連続する並びにして 94 µs。
- 内訳 (同じ行列、4.9 ms): ドット積そのものは 2.2 ms (約 12 TFLOPS 相当)、重みの展開 0.4 ms、バリア 0.1 ms、残りはロードと整数 → f32 のスケール掛け。GGUF の形式は 32 個 (Q6_K などは 16 個) ごとにスケールが変わるので、スケール掛けは減らせない。
- レーンの形は 4 × 8 が最良 (4 × 4 は 6.4 ms、8 × 8 は 4 × 8 より遅い)。

**速度** (2B GGUF Q4_K_M、1 パスの ms):

| 行数 | f16 (従来の既定) | int16 | int8 |
|---|---|---|---|
| 17 | 84.8 | 77.9 | 58.7 |
| 32 | 137 | 80 | 61 |
| 64 | 144 | 119 | 83 |
| 128 | 283 | 210 | 138 |
| 256 | 549 | 413 | 273 |
| 512 | 1115 (459 行/s) | 848 (604 行/s) | 572 (896 行/s) |

8B GGUF Q4_K_M (ms):

| 行数 | f16 | int16 | int8 |
|---|---|---|---|
| 17 | 281 | 289 | 260 |
| 32 | 514 | 295 | 263 |
| 64 | 558 | 401 | 292 |
| 128 | 1032 | 716 | 461 |
| 256 | 2104 | 1415 | 920 |
| 512 | 3794 (135 行/s) | 2848 (180 行/s) | 1860 (275 行/s) |

- Ollama のプリフィルは 2B が約 710 tok/s、8B が約 232 tok/s。int8 は同じ精度で 2B 1.26 倍・8B 1.19 倍、int16 は 2B 0.85 倍・8B 0.78 倍。
- 8B では 32 トークンのタイルでも 260〜290 ms かかる。重みの展開と LDS からの読み出しはトークン数によらず、ブロック数 (重みの量) に比例するため。

**精度** (f32 のプリフィルとの差):

- 最初に出るトークンの logits (ゴールデン 3 件 + 600 トークン、`prefill_kinds_against_f32`):

  | | 2B Q8: 最大差 | KL | 2B Q4_K_M: 最大差 | KL |
  |---|---|---|---|---|
  | f16 | 2.8e-2 | 1.2e-6 | 1.8e-2 | 1.3e-6 |
  | int16 | 5.8e-3 | 1.2e-6 | 6.2e-3 | 1.1e-6 |
  | int8 | 5.9e-1 | 6.4e-3 | 4.1e-1 | 1.4e-3 |

- 訳文が f32 のプリフィルと同一の段落数 (14 段落、履歴つき。技術文は生成 790 トークン、文学は 1496 トークン・3 チェーン):

  | 本体 | 文書 | f16 | int16 | int8 |
  |---|---|---|---|---|
  | 2B Q8 | 技術文 | 14/14 | 14/14 | 14/14 |
  | 2B Q4_K_M | 技術文 | 14/14 | 14/14 | 14/14 |
  | 2B Q4_K_M | 文学 | 14/14 | 14/14 | 4/14 (類似度 0.937) |
  | 8B Q4_K_M | 技術文 | 12/14 | 14/14 | 9/14 (0.993) |
  | 8B Q4_K_M | 文学 | 12/14 | 14/14 | 7/14 (0.957) |

  - int16 は 5 条件すべてで f32 と同一。従来の f16 より速く、かつ正確なので、**既定を int16 にした**。
  - int8 は llama.cpp と同じ精度で、さらに 1.5 倍速いが、難しい文章では 3〜7 割の段落で訳が変わる (1 か所の違いが履歴を通じて後ろに広がる)。`--prefill-precision int8` で選べる。
- 同じ文書の通し時間 (秒、`bench doc`、f32 / f16 / int16 / int8): 2B Q8 技術文 19.7 / 17.7 / 17.0 / 15.8、2B GGUF 文学 21.9 / 19.9 / 18.7 / 17.2、8B 技術文 46.4 / 39.5 / 35.8 / 32.4、8B 文学 65.3 / 57.3 / 52.9 / 48.7。

**CLI**: `--prefill-precision int16|int8|f16|f32` (既定 int16)。再現性設定は従来どおり `--prefill-precision f32 --draft none --no-kv-cache`。

**Ollama との比較** (11.31 と同じハーネス、秒。下書きは 11.32 の Q2_K):

| 測定 | 2B: 11.31 | 2B: int16 (既定) | 2B: int8 | 2B: Ollama | 8B: 11.31 | 8B: int16 (既定) | 8B: int8 | 8B: Ollama |
|---|---|---|---|---|---|---|---|---|
| 文セット 19 件 | 8.67 | 7.15 | **6.72** | 9.91 | 21.9 | 17.89 | **16.80** | 28.5 |
| 文書 14 段落 (逐次) | 37.4 | 35.67 | **33.64** | 39.9 | 89.4 | 84.03 | **77.86** | 113.2 |
| 文書 40 段落 (8 チェーン) | 66.3 | 62.71 | **58.12** | 70.5 | 197.4 | 180.08 | **164.95** | 176.4 |
| 同時 8 件 | 1.90 | 1.75 | **1.58** | 2.82 | 6.35 | 5.56 | **4.88** | 7.91 |
| 同時 16 件 | 2.23 | 2.22 | **2.09** | 4.01 | 7.07 | 6.72 | **6.32** | 10.79 |

- 既定 (int16) で全項目が 11.31 より速くなった。Ollama に届いていないのは 8B の 8 チェーン文書だけで、差は 2% (180.1 対 176.4)。Ollama と同じ精度の int8 なら 165.0 s で、全項目で Ollama より速い。
- 8B の 8 チェーン文書の内訳: プロンプト 21488 トークンのうち 61% は KV の再利用で、計算するのは約 8400 行 (約 50 秒)。残りの約 125 秒は 4862 トークンのデコードで、7 行のパスが 112 ms (1 行は 64.5 ms)。つまりこの測定の 7 割はデコードの複数行パスで、プリフィルではない。
- デコード中のプリフィルの刻み (`busy_prefill_rows`) を 64 → 128 行にすると 181.2 → 177.3 s (−2%)。他のストリームのトークンの間隔が 8B で 0.4 → 0.7 秒に伸びるので、採用しなかった。

**項目 4 (融合) の見直し。** 省けるのは「プリフィルのパスと対になるデコードのパスの重み読み出し」で、8B のこの文書では約 130 回 × 65 ms = 8.5 秒。一方、融合したパスではデコード行の f32 の積和が整数 GEMM のレーンに上乗せされ (1 交互あたり +20 ms 程度の見込み)、正味は約 3%。カーネルは複雑になり、デコード行の加算順も GEMV と変わる。**見送りを提案する。**

**代わりに効く候補: デコードの複数行パスを整数ドット積にする。**

- 項目 1 で分かったとおり、複数行の GEMV は演算律速 (2B で 1 行あたり +1.3 ms、8B で +8 ms)。整数ドット積なら 1 命令で積和 4 つなので、2 プレーン (15 bit) でも追加行のコストが約半分になる見込み (未実測)。
- 見込み: 8B の 7 行パス 112 → 約 88 ms (8 チェーン文書が約 14% 短縮)、2B の検証パス 6 行 31.9 → 約 27 ms (投機が約 8%)。項目 1 の当初の目標に近い。
- ただし 11.9 の方針 (デコード行は常に f32) を変えることになる。行ごと・ブロックごとの量子化なので、同じ行の結果は同時に走る行に左右されない。整数の和は厳密で、スケール掛けの順序もカーネルによらず同じにできるので、「1 行のデコードも同じ整数経路にする」なら、同時実行や投機の有無で出力が変わらない性質は保てる。変わるのは CPU の f32 参照との差で、logits が 1e-5 → 6e-3 程度になる (int16 のプリフィルと同じ水準)。
- 方針の変更なので、実装前に確認する。

### 11.34 デコード行も整数ドット積にする (複数行パスが 2B で 1.3〜1.5 倍、8B で 1.5〜1.6 倍)

11.33 の最後に挙げた候補を実装した。**11.9 の方針を変える**: 16 行以下のパス (デコード、投機の検証、短いプロンプト) も、既定で活性化を 15 bit + 符号に丸めた整数ドット積 (int16) で計算する。f32 のままにするには `--decode-precision f32`。

**カーネル** (`yaxus-backend-wgpu/src/gemv_int.rs`、WGSL を生成):

- 前処理 `xquantr`: プリフィルの `xquant` と同じ丸め (行ごと・32 個ごとに `max|x|/16383`、`128·hi + lo` の 2 プレーン)。並びだけが違う (下記)。
- GEMV: 1 ワークグループが重み 8 組 × 32 レーン。1 レーンが 1 ステップで重み 32 個 (スケールの 1 ブロック) を読み、整数値のままレジスタに置いて、すべての活性化行と `dot4I8Packed` を取る。重みは 1 回しか読まない (LDS も使わない)。
  - ブロックごとに `y += f32(整数の和) × 重みのスケール × 活性化の単位 − 重みの min × 活性化の和`。32 レーンの和は subgroup の shuffle。
  - 行数 T ごとに専用のカーネルを生成し、初回の使用時にコンパイルする。
  - SwiGLU 融合版 (gate と up の行を同じワークグループで計算) も全ての T で使う。f32 版は T=1 だけが融合で、T≥2 は行列積 + `swiglu` の 2 ディスパッチだった。
  - Q8、Q4、Q4_K、Q6_K、Q3_K、Q2_K に対応。Q6_K などのスケールはファイルのままの形 (16 個ごとの int8 × 256 個ごとの f16) を読む。bf16 は従来の f32 GEMV。
- **行の結果は、同じパスの他の行に依存しない。** 量子化は行ごと・ブロックごとで、整数の和は厳密、f32 の演算列は T によらず同じ。1 行だけで実行した結果と 7 行・3 行のパスの中の結果が、6 形式すべてでビット単位に一致することをテストにした (`integer_rows_do_not_depend_on_the_pass`)。したがって、同時実行や投機の有無で本体の出力は変わらない (`concurrent_matches_solo`、`speculative_matches_plain` も通過)。

**作る過程で分かったこと:**

- **活性化の並び。** 最初は [ブロック][行][プレーン] の順に置いた。T × プレーン数が 4・8・16 のとき (レーン間の間隔が 256・512 バイト) だけ 1.3〜1.5 倍遅くなった (2B int16: 3 行 23.4 ms に対して 4 行 31.0 ms、8 行 60.8 ms)。11.25 のスケールの件と同じ、2 のべきの間隔でのキャッシュの衝突とみられる。[行][プレーン][ブロック] にして、1 回のロードで 32 レーンが連続する 512 バイトを読む形にすると消えた (4 行 24.0 ms、8 行 33.3 ms)。
- **1 レーンが持つ重みの行数。** 1 行だと、活性化のロード 1 回 (`vec4<u32>`) がドット積 4 回にしか使われず、ロードが律速になる (2B 8 行 33.3 ms)。同じレーンが重み 4 行を持つと、同じロードが 16 回のドット積に使われる。

  | 1 パスの ms (int16) | 重み 1 行 | 2 行 | 4 行 | 8 行 |
  |---|---|---|---|---|
  | 2B GGUF、8 行 | 33.3 | 26.0 | 26.0 | 28.9 |
  | 2B GGUF、16 行 | 59.6 | 38.9 | 39.7 | 56.1 |
  | 8B GGUF、8 行 | 107.9 | 76.6 | 74.4 | 81.3 |
  | 8B GGUF、16 行 | 185.0 | 115.6 | 109.4 | 179.6 |

  すべての T で 4 行にした。T=1 (下記のカーネル内量子化) でも 4 行が最良だった (下書きなしのデコード、tok/s、1 / 4 / 8 行: 2B Q4_K_M 44.1 / 44.6 / 43.6、0.8B Q2_K 166 / 188 / 184)。重みの行数を変えても行ごとの演算列は同じなので、結果は変わらない。
- **1 行のデコードは、量子化をカーネルの中で行う。** 最初は 1 行でも `xquantr` を別のディスパッチにしていた。
  - カーネルの GPU 時間は f32 版より約 3% 短い (2B の gate/up 376 → 365 µs、Q6_K の down 277 → 273 µs)。しかし行列積ごとにディスパッチが 1 つ増える (1 トークンあたり 224 → 321 個)。ディスパッチは中身が 1.3 µs でも 1 個あたり約 6 µs かかり (前のディスパッチの完了待ち)、差し引きで 1.4% 遅くなった (2B GGUF、下書きなし 44.1 → 43.5 tok/s)。
  - そこで T=1 専用に、ワークグループが 1 ステップ分 (32 ブロック) の活性化を自分で丸めて LDS に置く版を作った。8 レーンが 1 ブロックを受け持ち (1 レーンが 4 値)、ブロックの最大値と和は subgroup の shuffle で集める。ディスパッチは増えない。
  - 丸めの式と和の順序を `xquantr` と揃えたので、結果は複数行パスとビット単位で一致する (上のテストがこの組み合わせを比べている)。
  - 下書きなしのデコードは f32 版と同等以上になった (tok/s、f32 → int16): 2B Q4_K_M 44.0 → 44.3、2B Q8 28.2 → 27.9、8B Q4_K_M 15.3 → 15.1、0.8B Q2_K 189 → 184。
- **下書きモデルの 1 行パスは f32 のまま。** 下書きは提案するだけで出力の厳密さが要らず、Q2_K では f32 の GEMV のほうがわずかに速い (19 件セット: 整数 74.0、f32 74.4 tok/s。カーネル内量子化の前は 67.7 対 70.9)。下書きの複数行パス (外れたあとの追いつき) は整数版。本体を f32 にしない理由: 1 行のときと複数行のときで logits が 5e-3 ずれ、投機の長さや同時実行で出力が変わり得る。
- **17〜48 行のパスも、整数 GEMV を 2〜3 回に分けて実行する。** 重みを 2〜3 回読むほうが、GEMM のタイル 1 枚より安い (GEMM は重みの展開と LDS の読み出しがトークン数によらず掛かる)。

  | 1 パスの ms | 2B: GEMV 分割 | 2B: int16 GEMM | 2B: int8 GEMM | 8B: GEMV 分割 | 8B: int16 GEMM | 8B: int8 GEMM |
  |---|---|---|---|---|---|---|
  | 17 行 | 50 | 79 | 59 | 147 | 290 | 260 |
  | 32 行 | 73 | 82 | 61 | 212 | 295 | 263 |
  | 48 行 | 108 | 119 | 83 | 318 | 396 | 292 |
  | 64 行 | 142 | 122 | 83 | 423 | 402 | 292 |

  プリフィルの設定が int16 か f16 なら 48 行まで、int8 なら 24 行まで GEMV を使う。f32 (参照) なら 16 行まで。
- **attention。** 9 行以上のパスはプリフィル用のタイル版 attention を使っていたが、デコードのパスは別々のリクエストの行なので合わない (16 リクエストを 1 パスにすると 74 ms、2B)。行ごとの分割版を 16 行まで使うようにした (同 38 ms)。
- **カーネルの事前コンパイル。** 行数ごとのカーネルは展開した長いソースで、コンパイルに 1 個あたり最大 40 ms かかる。初回の使用時にコンパイルすると、19 件セットの中で 57 個・計 1 秒の引っかかりになった。モデルの読み込み時に、そのモデルの重みの形式について全行数分 (GEMM のタイルも) を全コアで並列にコンパイルする (2B GGUF の本体 72 個で 0.12 秒、下書き 99 個で 0.20 秒)。

**1 パスの時間** (ms、下書きなし、`bench rows`。f32 → int16):

| 行数 | 2B Q4_K_M | 2B Q8 | 8B Q4_K_M | 0.8B Q2_K |
|---|---|---|---|---|
| 1 | 22.3 → 22.4 | 35.5 → 35.9 | 65.5 → 64.8 | 5.6 → 5.5 |
| 2 | 23.5 → 23.2 | 37.6 → 36.4 | 70.8 → 66.0 | 6.4 → 6.5 |
| 4 | 25.9 → 23.7 | 39.1 → 36.9 | 81.4 → 67.9 | 6.7 → 6.3 |
| 6 | 32.0 → 23.9 | 41.9 → 37.3 | 102.0 → 70.4 | 9.1 → 6.8 |
| 7 | 35.3 → 25.1 | 44.1 → 37.9 | 112.9 → 73.5 | 10.1 → 7.5 |
| 8 | 38.0 → 25.5 | 45.6 → 38.7 | 120.9 → 73.3 | 10.7 → 7.9 |
| 12 | 70.1 → 32.3 | 74.9 → 41.3 | 225.9 → 90.7 | 18.8 → 9.6 |
| 16 | 107.3 → 38.6 | 111.6 → 46.6 | 351.3 → 108.2 | 27.0 → 11.9 |
| 24 | 80.1 → 61.0 | 96.6 → 78.6 | 291.8 → 177.7 | 23.7 → 18.3 |
| 32 | 81.9 → 73.3 | 98.6 → 89.4 | 295.3 → 211.5 | 23.8 → 22.2 |

- 11.31 の項目 1 の目標 (2B GGUF で 6 行 26 ms・8 行 28 ms) を、6 行 23.9 ms・8 行 25.5 ms で上回った。8B の 7 行は 112.9 → 73.5 ms (11.33 の見込み 88 ms より速い)。
- 0.8B の 1 行は表では int16 が 5.5 ms だが、通しのデコードでは f32 のほうが少し速い (上記)。
- 9〜16 行の f32 の値は、11.33 の境界の誤り (下記の訂正) で整数 GEMM に入っていた分ではなく、f32 の複数行 GEMV のもの。

**精度** (f32 のデコードとの差):

- logits (`decode_kinds_against_f32`: f32 のプリフィルのあと、教師強制で 8 ステップ × 4 プロンプト): int16 は最大差 5.4e-3 (2B Q8)・5.5e-3 (2B Q4_K_M)、KL 4.5e-7、argmax は 32/32 一致。参考に int8 (1 プレーン) は最大差 0.6〜0.7、argmax 30/32 (Q8) で、デコードには使わない (選択肢にも出していない)。
- 訳文 (11.33 と同じ 5 条件、14 段落ずつ。プリフィルは int16): **int16 のデコードは、5 条件すべてで全 f32 の参照と 14/14 段落一致**。文学の 2 条件は複数チェーンなので、同時デコード (1 パス最大 16 行) も含んでいる。
- 19 件セット: 既定設定は `ref_default.txt` と 19/19、再現性設定 (`--prefill-precision f32 --decode-precision f32 --draft none --no-kv-cache`) は `ref_f32.txt` と 19/19。
- テスト: `concurrent_matches_solo`、`speculative_matches_plain`、`hints_and_response_cache_keep_output` は整数デコードのまま通過。CPU との f32 比較 (`decode_attention_matches_cpu`) は `Gemv::F32` を指定して従来の許容差のまま。

**文書の通し時間** (秒、`bench doc`、プリフィルは int16。デコード f32 → int16):

| 本体 | 文書 | f32 | int16 | 差 |
|---|---|---|---|---|
| 2B Q8 | 技術文 14 段落 | 17.44 | 16.65 | −4.5% |
| 2B Q4_K_M | 技術文 14 段落 | 15.60 | 14.41 | −7.6% |
| 2B Q4_K_M | 文学 14 段落 (3 チェーン) | 19.00 | 17.79 | −6.4% |
| 8B Q4_K_M | 技術文 14 段落 | 35.95 | 29.93 | −16.7% |
| 8B Q4_K_M | 文学 14 段落 (3 チェーン) | 53.07 | 46.48 | −12.4% |

**エンジンの既定値の見直し:**

- `--decode-rows` (1 パスで進めるリクエスト数) の既定を、int16 のとき 8 → 16 にした。f32 では 8 行が 1 行あたり最安 (2B: 8 行 38 ms、16 行 108 ms) だったが、整数版では 16 行が最安 (26 ms と 38 ms)。同時 16 件のスループット (`bench streams`、tok/s): 2B GGUF 284 → 346、8B GGUF 98.9 → 123.6。
- `--draft-k` (下書きの最大長、既定 7) は 10・14 に上げても変わらなかった (19 件セット 65.7 → 65.7 tok/s、技術文 15.25 秒で同じ)。検証の行は安くなったが、下書きの 1 ステップ (5.1 ms) が長さに比例して掛かるため。

**CLI**: `--decode-precision int16|f32` (既定 int16)。`--prefill-precision` は長いパスだけに効く (int16・f16 では 49 行以上、int8 では 25 行以上、f32 では 17 行以上)。

**Ollama との比較** (11.31 と同じハーネス、秒。既定設定 = プリフィル int16、デコード int16、下書き Q2_K。Ollama は v0.40.16 で以前の値):

| 測定 | 2B: 11.33 | 2B: 11.34 | 2B: Ollama | 8B: 11.33 | 8B: 11.34 | 8B: Ollama |
|---|---|---|---|---|---|---|
| 文セット 19 件 | 7.15 | **6.10** | 9.91 | 17.89 | **12.99** | 28.5 |
| 文書 14 段落 (逐次) | 35.67 | **34.18** | 39.9 | 84.03 | **76.04** | 113.2 |
| 文書 40 段落 (8 チェーン) | 62.71 | **56.68** | 70.5 | 180.08 | **152.32** | 176.4 |
| 同時 8 件 | 1.75 | **1.46** | 2.82 | 5.56 | **4.10** | 7.91 |
| 同時 16 件 | 2.22 | **1.54** | 4.01 | 6.72 | **4.37** | 10.79 |

- 既定設定で、全項目で Ollama より速くなった。11.33 で唯一負けていた 8B の 8 チェーン文書は 180.1 → 152.3 秒 (Ollama 176.4 秒の 0.86 倍)。8B の文セットは 0.46 倍、同時 16 件は 2B・8B とも約 0.4 倍の時間。
- 下書きなしの単独デコード (2B GGUF) は 44.3 tok/s で、Ollama (45.3) にまだ 2% 届かない (11.31 の項目 6)。

### 11.35 項目 3 (木構造の下書き) の見積もり: 見送りを提案する

作る前に、効果を実データで見積もった。下書きの各ステップの上位 4 候補と確率を記録し (一時的な計測で、コミットしていない)、本体の選択と比べた。対象は 2B GGUF と 8B GGUF の本体 (下書きは 0.8B Q2_K) で、文セット 19 件、技術文 14 段落、『羅生門』14 段落 (投機を使うよう 1 チェーン)。

**食い違った位置で、本体のトークンが下書きの何番目の候補だったか:**

| 本体 | 文書 | ラウンド | 1 ラウンドのトークン | 食い違ったラウンド | 2 番目 | 2〜3 番目 |
|---|---|---|---|---|---|---|
| 2B | 文セット | 81 | 4.67 | 33% | 59% | 74% |
| 2B | 技術文 | 176 | 4.53 | 34% | 58% | 78% |
| 2B | 文学 | 627 | 2.40 | 48% | 44% | 59% |
| 8B | 文セット | 85 | 4.40 | 38% | 75% | 91% |
| 8B | 技術文 | 172 | 4.67 | 38% | 67% | 83% |
| 8B | 文学 | 513 | 2.92 | 74% | 38% | 52% |

- 2 番目の候補はよく当たる。しかし**食い違いの半分以上は、下書きが上位 1 候補に 0.8 以上の確率を付けた位置で起きる**。下書きの確信度から「どこで枝分かれすべきか」が分からない。
- 逆に各ステップで見ると、2 番目の候補の確率が 0.1 未満のステップ (全体の大半) では、2 番目が本体のトークンになるのは 3〜7% だけ。

**速さの見積もり** (11.34 の実測のパス時間、下書き 1 ステップ 5.6 ms・2 行 6.5 ms、1 ラウンドの手間 1 ms。ラウンドごとの下書き長は記録どおり):

- 葉だけの代替 (各深さで 2 番目の候補を 1 行足す。当たれば +1 トークン。次のラウンドで 1 行増える):

  | 本体 / 文書 | 全深さ | 2 番目の確率 ≥ 0.1 の深さ | 最初の 2 深さ |
  |---|---|---|---|
  | 2B 文セット / 技術文 / 文学 | −7.4% / −7.3% / +5.5% | +0.2% / +1.0% / +4.0% | −1.8% / −0.8% / +6.2% |
  | 8B 文セット / 技術文 / 文学 | −8.4% / −10.9% / −5.0% | +1.6% / +0.8% / +1.8% | +0.5% / −1.4% / +2.4% |

- 枝を下書きで伸ばす 2 本の木 (楽観的な上限。枝の先は本体の「前が当たったときの採用率」0.74〜0.93 で当たると仮定し、下書きの 2 行目はほぼ無料): 2B −0.5〜+4.5%、8B −0.8〜+4.5%。根元で枝分かれすると −6〜−16%。
- 効かない理由:
  - 1 ラウンドの時間の約半分が下書きのステップ (2B の文セット: 下書き 5 × 5.6 = 28 ms、検証 6 行 24 ms)。木にしても下書きのステップは減らない。
  - 検証の行は安くなったが (11.34)、無料ではない (2B: 6 行 23.9 ms、11 行 約 31 ms)。外れやすい文学的な文章では下書き長 k=2 がよく選ばれ、そもそも 1 ラウンドが短い。

**結論**: 実装には、木の注意マスク (attention で行ごとに自分のスロットを使う)、行ごとの位置を指定するパス、下書きの上位 2 候補を出す argmax、採用後の KV の扱いが要る。見込みは 0〜5% で、文によっては遅くなる。**見送りを提案する。**

**代わりに効きそうなこと**: 下書きのステップが 1 ラウンドの約半分になったので、下書きを安くする余地が大きくなった。0.8B の 1 ステップ (1 行) のうち、`lm_head` (107520 × 1536、Q4) が 1.14 ms、18.5% を占める。項目 2(b) (下書きの `lm_head` を候補語彙に絞る) は 11.32 の時点で「訳先の言語次第で 0〜4%」と見積もって後回しにしたが、いまは 1 ラウンドの約 9% に当たる。出力は変わらない (本体が検証する) ので、採用率がどれだけ落ちるかで採否が決まる。

### 11.36 項目 2(b) (下書きの語彙の絞り込み) の見積もり

11.35 で、下書きのステップが 1 ラウンドの約半分を占め、そのうち `lm_head` (107520 × 1536、Q4) が 1.14 ms (18.5%) だと分かった。行を候補語彙だけに絞れば、その分だけ下書きが速くなる。出力は変わらない (本体が検証する)。効くかどうかは、下書きが出せなくなるトークンを本体がどれだけ使うかで決まる。

**本体の出力頻度から作った固定集合は、日本語で破綻する。**

- 候補を作るため、テストに使っていない文章 (`/usr/share/doc` の英文 480 段落と、日本語の man ページ 435 段落) を 2B に翻訳させ、出力 57,000 トークン (日本語 35,000、英語 22,000) の頻度を数えた。自然文 (英語 39 万トークン、日本語 2.5 万トークン) の頻度も弱く混ぜた。
- 頻度の上位 N 個に、制御・バイトトークン 315 個とその文書の原文のトークンを加え、11.35 の記録 (本体が実際に出したトークンと下書きの提案) で、提案が集合の外に出た所で採用の連鎖が切れるとして見積もった。

  | 集合の大きさ | 2B 文セット: 被覆率 / 速さ | 2B 技術文 (和→英) | 2B 文学 (和→英) |
  |---|---|---|---|
  | 約 4,600 | 55% / −55% | 77% / −20% | 44% / −36% |
  | 約 16,800 | 71% / −36% | 93% / −5% | 66% / −23% |
  | 約 27,200 (頻度が 0 でないもの全部) | 75% / −30% | 98% / +4% | 78% / −15% |

- 日本語の語彙は 49,000 トークンあり、「飛躍的に」「が、その」「原動力」のような長いトークンが多い。そのため頻度の裾が長い。文セットで本体が出したトークンの 35% は、57,000 トークンの出力コーパスに一度も現れなかった。11.19 のスコア順の集合 (採用率 81% → 47%) が失敗したのと同じ理由。
- 8B も同じ傾向 (文セット −33% 以上)。

**訳先が英語なら、日本語・韓国語のトークンを外すだけで足りる。**

- 漢字・かな・全角・ハングルを含むトークンを外すと、残りは 57,236 (語彙の 53%)。和→英の 4 条件 (技術文・文学 × 2B・8B) で、本体が出したトークンの被覆率は、原文のトークンを足せば 100%、足さなければ 98.7〜100%。技術文では「ライセンス」「対応言語」などの日本語を英訳の中にそのまま残すことがあり、足さないと採用が 1.5〜1.8% 減る。
- 下書き 1 ステップの `lm_head` が 1.14 → 約 0.61 ms になる。見込みは英訳で 2B が +5% 前後 (1 ラウンド 53 ms のうち下書き 5 ステップで 2.6 ms 減)、8B が +2〜3%。和訳では効かない。
- 作るとすれば、下書きの `lm_head` の行を「非 CJK、CJK」の順に並べ替えて読み込み、英訳のリクエストでは前半の行だけで GEMV と argmax を行う (索引を元のトークン ID に戻す)。原文の CJK トークンは、その行だけを別に計算するか、諦めて 1.5% の採用減を受け入れる。リクエストに訳先言語を渡す配線も要る。

**別案: 下書きの `lm_head` を 2〜3 bit にする。** 今は Q4 で 0.56 バイト/重み。Q2_K 相当 (約 0.33) にすれば、すべての言語で `lm_head` が約 4 割減る (1 ステップ約 0.45 ms)。下書きの確率分布が粗くなるので、採用率への影響は実測が必要。 → 11.37 で実測し、効果なし。

### 11.37 下書きの `lm_head` の低ビット化 (不採用)

11.36 の別案を試した。0.8B Q2_K の下書きの `lm_head` (GGUF では Q6_K、いまは読み込み時に Q4) を、読み込み時に Q3_K または Q2_K へ量子化し直す。GPU は GGUF の Q3_K/Q2_K をそのまま読めるので (11.32)、カーネルの追加は要らない。

- **量子化器:** llama.cpp の参照実装 (`quantize_row_q3_K_ref`, `quantize_row_q2_K_ref`) と、重要度ベクトル付きの版 (`*_impl`) を Rust で書いた。重要度は、`lm_head` への入力 (最終ノルムの出力) の次元ごとの二乗平均。テストに使っていない文章 (11.36 の英文と日本語の各 80 段落) を 0.8B で訳して 22,000 行から取った。中央値 1.1 に対して最大 107 と、外れ値の次元がある。
- **下書き 1 ステップ** (1 行、f32): Q4 4.9 ms、Q3_K 4.6 ms、Q2_K 4.35 ms。

| `lm_head` | 19 件セット: 採用率 / tok/s | 羅生門 和→英 14 段落 | アリス 英→和 14 段落 |
|---|---|---|---|
| Q4 (今) | 80% / 60.6 | 49% / 42.6 s | 46% / 26.3 s |
| Q8 | 79% / 56.5 | | |
| Q3_K | 78% / 59.8 | | |
| Q2_K | 72% / 57.2 | | |
| Q3_K + 重要度 | 79% / 60.2 | 49% / 42.4 s | 44% / 26.5 s |
| Q2_K + 重要度 | 76% / 59.1 | 48% / 42.6 s | 45% / 26.3 s |

(本体は 2B safetensors q8。文書の時間は 2 回測ったベースラインの差 (±0.1 s) と同程度の差しかない。訳文はすべて同一。)

- 重要度で採用率はかなり戻る (Q2_K 72% → 76%) が、残った低下でステップの短縮 (0.3〜0.55 ms) が相殺される。下書きのステップは文書では 1 ラウンド (約 85 ms) の数 % にすぎず、もともと上限は 2〜3% だった。8B の本体ではラウンドが長いので、さらに小さい。
- Q8 の `lm_head` でも採用率は Q4 と同じなので、今の Q4 (単純な max/7 の丸め) で失っているものはない。
- 採用率を落とさずに読み出しだけ減らすには、Q2_K で上位数個の候補を出し、その行だけ Q4 で計算し直す方法がある。ただし上位 k 個を返す argmax カーネルが要り、上限が 2〜3% なので見送る。
- 実験のコードはブランチ `draft-head-exp` (82a1bcb) に残した。main には入れない。

### 11.38 項目 6: 1 行のパスの行列積以外

11.34 では「ディスパッチ 1 個が約 6 µs」と見て、融合で減らす計画だった。先に測った。

**測定** (1 行のパス。0.8B Q2_K の下書き設定 (f32) は 1 パス 4.75 ms・152 ディスパッチ、2B GGUF は 21.9 ms・224 ディスパッチ):

- 空のディスパッチを各ディスパッチの後に 1 個足すと、0.8B が +0.47 ms、2B が +0.57 ms。1 個あたり 2.5〜3 µs で、6 µs ではなかった。
- `add_norm` を丸ごと省くと (結果は壊れる)、0.8B が −0.25 ms (5%)、2B が −0.54 ms (2.5%)。`qk_rope` と `attn_combine` を省いても 0.05〜0.15 ms しか変わらない。
- したがって、融合で取れるのは最大でも `add_norm` の分。

**`add_norm` の GEMV への融合は逆効果 (不採用)。** qkv と gate_up の GEMV の入口で、各ワークグループがノルムを計算し直す形を試した。`add_norm` と同じ要素順と縮約の木にしたので、訳文は融合なしと同一。

- そのままでは下書き 1 ステップが 4.75 → 8.4 ms。gate_up はワークグループが 1536 個あり、全部がノルムを計算するため 111 → 394 µs になった。
- ワークグループを GPU が同時に動かせる数 (96 個) に絞って行をループさせると 5.98 ms。共有メモリの読み方をバンク衝突のない並びに変えると 5.16 ms。
- 原因: `add_norm` の時間はディスパッチの費用ではなく、1 行分の計算の待ち時間 (メモリの往復と縮約が直列に並ぶ) だった。入口に移しても同じ待ち時間 (10〜15 µs) が GEMV の前に残る。そのうえ、x を共有メモリから読む GEMV 自体が遅い (gate_up はノルムなしでも 113 → 133 µs)。
- 試作は `fused-prototype.patch` としてセッションのスクラッチパッドにだけ残した。

**採ったもの 1: `add_norm` の待ち時間を縮める** (`ADD_NORM_SG`、subgroup がある場合)。

- 読み出し (d、x、両方のノルムの重み) を最初にまとめて発行し、行はレジスタに置く。縮約は 32 レーン内のシャッフルと 8 個の部分和で、バリアは 9 回から 1 回になる。
- 1 回あたり 8.7 → 4.3 µs。バリアを減らしただけでは変わらず、効いたのは読み出しをまとめたこと。
- `subgroupAdd` を使った最初の版は訳文を壊した (19 件中 7 件だけ一致)。このカーネルは 64 レーンの subgroup で動くことがあり、32 レーンごとに部分和を書くと重複して数えるため。GEMV と同じく 32 レーン内の `subgroupShuffleXor` にした。
- 縮約の順序が変わるので、厳密には f32 の丸めが変わる。それでも 19 件セット (2B Q8 で 19/19、2B GGUF で 16/19) と文書 2 本 (羅生門 2B、技術文 8B) で、訳文は変更前と同一だった。

**採ったもの 2: 文脈が短いときの attention をヘッドで分割する。**

- 分割版 attention は、KV ヘッド 1 個とキーのチャンク 1 個ごとに 1 ワークグループ。0.8B は KV ヘッドが 1 個なので、文脈 64 ではワークグループが 2〜4 個しかなく、1 個で 16 ヘッドを処理していた (1 層 26 µs、16 層で 0.4 ms、下書き 1 ステップの約 8%)。
- 1 ワークグループの担当ヘッド数を可変にした (`attn_split{G}_{CH}_{HG}`)。ワークグループの総数 (行数 × KV ヘッド数 × 分割数 × チャンク数) が 64 以下になる範囲で最も細かく分ける。それぞれのヘッドの計算は分け方によらず同じなので、結果はビット単位で変わらない。
- 0.8B: 文脈 64 で 26 → 8.4 µs、256 で 26 → 14.8 µs。1024 では分割しない (ワークグループが既に多く、分けるとキーの読み直しが増えて遅い: 2B の 1024 キーで 67 → 78 µs)。2B の文脈 64 では 15 → 11.3 µs。
- カーネルが 165 → 201 個に増え、起動時のコンパイルは 82 → 104 ms。

**結果** (同じ日に、変更前と交互に測定):

| 測定 | 変更前 | 変更後 |
|---|---|---|
| 2B GGUF 下書きなし、文脈 32 / 80 / 1024 (tok/s) | 44.1 / 44.0 / 41.4 | **44.9 / 44.7 / 41.8** |
| 19 件セット、2B GGUF + 下書き (tok/s) | 74.4 | **75.9** |
| 19 件セット、2B Q8 + 下書き (tok/s) | 60.6 | **61.4** |
| 羅生門 14 段落、2B GGUF (s) | 17.80 | **17.68** |
| 技術文 14 段落、8B GGUF (s) | 30.05 | **29.82** |

- 訳文はすべて変更前と同一。
- Ollama との比較は日によって揺れる。11.29 で 45.0 tok/s だった同じコミット (早期投入) は、この日は 44.0 tok/s だった。この日の Ollama v0.40.16 (内蔵 GPU、`OLLAMA_IGPU_ENABLE=1` が要る) は、同じ 2B GGUF で 46.0 tok/s (`eval_duration`、プロンプト 49 トークン、生成 41 トークン)。変更後の yaxus (文脈 32 で 44.9) はまだ約 2.5% 遅い。
- 残る差は行列積。

### 11.39 Metal (Apple M2) で動かす

PR #23 で macOS では wgpu の `metal` を有効にした。その上で M2 (10 コア GPU、ユニファイドメモリ 16 GB、macOS 26.7、wgpu 30.0.1) で動かした。

**正しさ: workgroup メモリの vec4 の成分書き込みが消える。**

- 当初、f32 GEMM (`GEMM2`)、整数 GEMM、1 行の整数 GEMV (行の丸めをカーネル内で行う版) の 3 つが誤った値を出した。ほかのカーネルは CPU と一致した。
- 原因は最小のシェーダで確かめた。`var<workgroup>` の vec4 配列に、レーンごとに異なる成分を動的な添字で書く (`xs[i][j % 4u] = v`) と、4 レーンのうち 1 レーンの値しか残らない。vec4 をまるごと書けば正しく、プライベート変数なら成分書き込みでも正しい。3 つのカーネルは、まさにこの書き方で workgroup メモリにステージングしていた。
- 対処: GEMM ではレーンが 4x4 のブロックを読み、レジスタで転置して vec4 ごと書く (f32 GEMM はレーン 0〜191、整数 GEMM は 128〜191)。内側ループの読み方は変えていない。1 行の整数 GEMV では、4 レーン分の値をシャッフルで先頭のレーンに集めて書く。
- **Vulkan でもこのステージングに変わったので、780M での再計測が要る。**

**行がパスの大きさによらないこと: Metal は fast math でコンパイルする。**

- wgpu は `MTLCompileOptions` を既定のまま使うので、fast math (結合則の組み替えや FMA 縮約) が有効になる。naga にも wgpu にもこれを止める設定はない。MSL のパススルーで `#pragma METAL fp math_mode(safe)` を足す回避策はあるが、wgpu 30 のパススルーは workgroup メモリの確保とバッファ長の受け渡しをしないので、naga の MSL をそのまま渡せない。
- その結果、`integer_rows_do_not_depend_on_the_pass` が落ちた (相対 1e-5 程度の差)。
- 整数 GEMV の積算を明示的な `fma()` にすると、複数行のパスどうし (3 行と 7 行) は全形式で一致した。
- K-quant のスケール `d * sc` は f32 で厳密だが、`(d * sc) * unit` が 1 行のカーネルだけ別の順に組み替えられる。スケールを `select` で包む (`keep`、条件はカーネルが他に読まない `p.t`) と一致した。
- カーネル内での行の丸めは、障壁をいろいろ試しても `xquant` と同じビットにならなかった。Metal では 1 行も `xquant` を通す (`Tuning::gemvi_inline = false`)。
- これはコンパイラの振る舞いに依存した経験的な一致で、テストが保証の代わりになっている。

**速度: naga の安全装置と整数ドット積。**

2B Q4_K_M、1 パスの時間 (ms):

| 行数 | 1 | 4 | 8 | 16 | 256 |
|---|---|---|---|---|---|
| 整数、wgpu の既定 | 52 | 363 | 1040 | 3560 | 43200 |
| f32、wgpu の既定 | 32 | 96 | 1800 | 5069 | 16200 |
| f32、下の設定 | 30 | 56 | 171 | 305 | 1681 |

- naga は Metal で workgroup メモリをカーネルの入口でゼロにする。それも 1 レーンが配列全体を書く (`gemvm16` は 32 KiB)。全カーネルが読む前に書いているので、Metal ではパイプラインの `zero_initialize_workgroup_memory` を切った。
- naga はすべてのループに無限ループ防止のカウンタを入れる。これで Metal のコンパイラがループを展開できず、ループの添字で引くプライベート配列 (f32 GEMM のアキュムレータ) がメモリに置かれる。ループはどれも形状で終わるので、Metal では `create_shader_module_trusted` でカウンタを外した。範囲チェックは残した (外すと 1 行が 1 割ほど速い)。
- Apple GPU には整数ドット積の命令がない。`dot4I8Packed` は 4 回の乗算とバイトの取り出しになり、整数カーネルは f32 に負ける。Apple GPU の既定は prefill、decode とも f32 にした (`Tuning::for_device`)。
- 範囲チェックと無限ループ防止は SPIR-V でも入る。Vulkan で外すとどうなるかは未計測。

**結果** (M2、2B Q4_K_M、下書きなし): デコード約 31 tok/s (CPU バックエンドは 4.4 tok/s)。下書き (0.8B Q2_K) ありでも訳文は同一。kernels テスト 7 本はすべて通る。

- 初回の実行だけ、Metal がシェーダをコンパイルしてキャッシュするので、最初の出力まで数秒余計にかかる。
- `vs_cpu` は CPU と GPU でモデルを 2 つ持ち、bf16 版はさらに展開する。16 GB の M2 ではメモリが足りず、スワップでマシンが固まった。通ったのは GGUF での abort_all、concurrent、decode_attention、decode_kinds の 4 本。
- 32 KiB の workgroup メモリではタイル attention がコンパイルされず、prefill の attention は行ごとのカーネルになる。

**残り**: Apple 向けの `Tuning` の値 (投入の FLOP 閾値、GEMV と GEMM の切り替え行数など) は 780M の値のまま。prefill はピークの 2 割弱 (256 行で 152 行/s) で、simdgroup 行列 (8x8) を使う GEMM の余地がある。

### 11.40 Metal 対応の Vulkan での再計測と整理

11.39 で「780M での再計測が要る」「Vulkan で外すとどうなるかは未計測」とした 2 点を、780M (RADV) で測った。

**ステージングの変更 (PR #24) は Vulkan で中立。** main と PR #24 を、2B (Q8) の `bench rows` で比べた (ms/pass)。

| | 1 行 | 16 行 | 256 行 |
|---|---|---|---|
| int16、main | 35.3 | 46.9 | 475.7 |
| int16、PR #24 | 35.6 | 47.0 | 476.9 |
| f32、main | 35.2 | 111.8 | 816.6 |
| f32、PR #24 | 35.3 | 111.9 | 823.5 |

文セットは両方の参照 (`ref_default`、`ref_f32`) と 19/19 一致した。`fma` と `keep` を入れても、Vulkan の出力は 1 ビットも変わっていない。

**naga の安全装置は、Vulkan ではループのカウンタだけを外す。** `YAXUS_LOOP_BOUNDING` と `YAXUS_ZERO_WGMEM` (`CompileSettings`) で切り替えて測った。

- ループのカウンタを外すと速くなった。int16 は 256 行で 477 → 465、128 行で 246 → 241。f32 は 9 行で 57.4 → 46.7、16 行で 111.7 → 101.4。文書ベンチは 47.9 → 49.2 tok/s。文セットは 19/19 で変わらない。RADV でも、カウンタがループの展開を妨げていたとみられる。
- workgroup メモリのゼロ初期化は、外しても差が出なかった。Vulkan では残す。
- 既定は、全バックエンドでループのカウンタなし、ゼロ初期化は Metal だけ切る。

**整理。**

- Metal の fast math は `DeviceCaps::fast_math` にした。1 行の整数 GEMV で丸めをカーネル内に入れるかは、`plan::gemv_int_inline` が決める。`Tuning::gemvi_inline` は廃止した。これは速度のつまみではなく、正しさの制約だったため。
- `MatmulPlan::GemvInt` が `inline` を持つ。実行するカーネルと事前コンパイルするカーネルは、同じ関数から決まる。
- CI の Metal ジョブは、デバイスの `DeviceCaps` を出力する。32 レーンのサブグループがなければ失敗する (`YAXUS_EXPECT_SUBGROUPS=1`)。

### 11.12 リスク

- ページ表の間接参照で attention が遅くなる可能性 → P1 の合格条件 (±3%) で確認。デコードの attention は全体の 2% 程度なので影響は小さい見込み。
- wgpu 30 の実験的機能と naga の制約 (8.6) の上に行対応カーネルを書くことになる。行列演算ユニットはプリフィルパスだけで使い、デコードパスは通常の機能だけに保つ。
- 中間カーネルが目標に届かない場合、16 並列の目標は未達になる (8 並列までは既存カーネルで到達できる)。
- 埋め込み表と KV アリーナで GPU メモリが約 2.8 GB 増える (合計約 6 GB、GTT 上限 31 GiB に対して余裕あり)。
