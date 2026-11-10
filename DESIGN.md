# yakusel 設計書

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
- 参照実装 (`modeling_plamo.py`) は KV キャッシュに RoPE 前の K を積み、毎ステップ「キャッシュ内の相対位置 0..kv_len」で q/k に RoPE をかけ直す (NICT inference path)。RoPE のスコアは位置差だけに依存し、YaRN の concentration も一様な倍率なので、**RoPE 済みの K を絶対位置でキャッシュする実装と数学的に等価**。yakusel は後者を採用 (M1 で HF と logits 誤差 3e-5、greedy 完全一致を確認)。
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
          └── yakusel CLI ── POST /v1/translate (JSON / SSE ストリーミング)
                                   │
                        ┌──────────▼──────────┐
                        │ yakusel serve (1 プロセス) │
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
yakusel/
  Cargo.toml                 # workspace
  DESIGN.md
  crates/
    yakusel-tokenizer/       # PLaMo トークナイザ (pure Rust, 依存最小)
    yakusel-model/           # 設定・safetensors ロード・plamo3 の計算グラフ・Backend トレイト
    yakusel-backend-cpu/     # 参照実装 (f32, rayon)。正しさの基準とフォールバック
    yakusel-backend-wgpu/    # Vulkan 用 WGSL カーネル群
    yakusel-server/          # axum + rustls、API、Web UI の埋め込み
    yakusel-cli/             # バイナリ `yakusel` (serve / translate / client / quantize / download)
  web/                       # Web UI (素の HTML/JS 1 ファイル。ビルド不要、include_str! で埋め込み)
  tests/golden/              # Python 側で生成した tokenizer / logits のゴールデンデータ
```

クライアント用途だけなら GPU 依存を外せるよう、`yakusel-cli` は feature で `server` と `client` を分ける。

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

`yakusel-model` はこのトレイトだけに依存して plamo3 の forward を書く。CPU と wgpu で同じコードパスを通すことで、CPU 実装をゴールデンにして GPU 実装を検証できる。

### 3.2 重みの持ち方

- safetensors (bf16) を mmap で読み、テンソル単位で GPU バッファにアップロードする。1 バッファ上限 2047 MiB に対し、2B/8B の最大テンソルは `lm_head`/`embed_tokens` の 880 MB (8B) なので分割不要。31B は要分割だが当面対象外。
- WGSL に bf16 型は無いので、**bf16 を u32 に 2 個パックしたままバッファに置き、カーネル内で `bitcast<f32>(u << 16)` で f32 に展開**する。変換は無損失で、ロード時の型変換も不要。
- 活性値は f32 (必要なら f16 に落とす)。KV キャッシュは f16。
- 量子化 (後段フェーズ): ブロック 32 ごとに f16 スケールを持つ Q8 / Q4 を自前定義し、`yakusel quantize` で事前変換した独自フォーマット (`.ysq`、safetensors のメタデータ拡張) として保存。デコードはカーネル内で逆量子化する。8B を実用速度にするにはこれが必須。

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
- 証明書は `--cert/--key` 指定か、未指定なら初回起動時に `rcgen` で自己署名証明書を `~/.config/yakusel/` に生成して再利用する。CLI クライアントは `--ca` でその証明書をピン留めできる (`--insecure` も用意)。
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

- `stream: false` → `{"text": "...", "usage": {"prompt_tokens": n, "completion_tokens": m, "elapsed_ms": t}}`。
- `stream: true` → `text/event-stream`。`data: {"delta":"..."}` を逐次送り、最後に `data: {"done":true,"usage":{...}}`。
- 入力に `<|plamo:` を含む場合は 400。言語コードは chat_template と同じ正規表現で検証。
- 同時接続は受け付けるが、推論は直列実行。キュー長上限 (既定 16) を超えたら 429。

### 4.3 Web UI

- 依存なしの単一 HTML (textarea 2 枚、言語セレクタ、form/glossary の折り畳み、SSE で逐次表示)。
- ビルド工程を持たないため `include_str!` で埋め込み、バイナリ 1 つで配布できる。

---

## 5. CLI

```
yakusel serve   --model ~/.cache/huggingface/hub/... [--backend vulkan|cpu] [--listen 0.0.0.0:8443] [--token ...]
yakusel translate --to en [--from ja] [--form polite] "テキスト"   # サーバを立てずローカルで実行
yakusel client  --server https://host:8443 --to en  < input.txt    # 標準入力を段落単位で送る
yakusel quantize --model ... --scheme q8|q4 --out ...
yakusel download --model pfnet/plamo-3-translate-2b              # hf-hub crate で取得
yakusel bench   --model ...                                         # プリフィル/デコード速度計測
```

---

## 6. 検証方針

1. **トークナイザ**: Python の `Plamo3Tokenizer` で多言語・特殊トークン・絵文字・連続空白などの入力を encode/decode したゴールデンデータ (`tests/golden/tokenizer.jsonl`) を作り、Rust 実装と完全一致させる。
2. **モデル (CPU)**: transformers (CPU, bf16→f32) で短いプロンプトの logits を保存し、CPU バックエンドの logits と最大絶対誤差で比較。greedy 出力列の一致も確認。
3. **モデル (Vulkan)**: CPU バックエンドと同一入力で logits を比較。カーネル単体も CPU 実装と突き合わせる単体テストを持つ。
4. **E2E**: README のサンプル文で期待訳が得られること。
5. **性能**: `yakusel bench` でプリフィル tok/s とデコード tok/s を継続的に記録。

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
| M0 | 完了 | `yakusel-tokenizer`: ゴールデン 1,814 件の encode/decode、プロンプトテンプレート 14 件 (エラー系含む)、ストリーミングデコードが Python と完全一致。`(.)\2{3,}` の後方参照と Python の `\s` 定義は手書きで再現 |
| M1 | 完了 | `yakusel-model` (Backend トレイト、forward、greedy 生成)、`yakusel-backend-cpu`。HF (f32) と最終 logits の最大誤差 3e-5、greedy 48 トークンが 3 ケースとも一致 |
| 高速化 1 | 完了 | rayon を物理コア数 (8) で起動 (`RAYON_NUM_THREADS` で上書き可)、`.cargo/config.toml` で `target-cpu=native`、T<4 では bf16/Q8 を一時バッファなしで直接内積 |
| 高速化 2 (Q8) | 完了 | ロード時量子化 (32 要素ブロック、f16 スケール = max\|w\|/127)。行列 + `lm_head` が対象、埋め込みと norm は bf16 のまま。重み 5.76→3.32 GB、ロード時の変換 約 0.65 s。CLI の既定を q8 に変更 (`--quant bf16` で無損失) |
| M3 | 完了 | `yakusel-api` (共有の型)、`yakusel-server` (axum + axum-server/rustls(ring))、`web/index.html`、`yakusel client`。下記「M3 の実装メモ」参照。M2 より先に実施 |
| M2 | 完了 | `yakusel-backend-wgpu`。下記「M2 の実装メモ」参照 |
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
cargo test --release                         # トークナイザ
cargo test --release -p yakusel-backend-cpu -- --ignored   # モデル (2B ロードあり)
```

### 8.2 M3 の実装メモ

- **バックエンドは当面 CPU (Q8)**。`yakusel serve` は推論スレッド 1 本 + tokio (2 スレッド)。キューは `--queue` (既定 16) 件まで待たせ、超えたら 429。
- **TLS**: §4.1 の「自己署名証明書」は「ローカル CA + サーバ証明書」に変更した。初回起動時に `~/.config/yakusel/ca.pem` (CA) を作り、それで署名したサーバ証明書を発行する。SAN は localhost / 127.0.0.1 / ::1 / ホスト名 / `ホスト名.local` / 主 IP + `--san` で追加したもの。SAN が変わればサーバ証明書だけ再発行するので、クライアントが信頼する `ca.pem` は変わらない。秘密鍵とトークンは 0600。
- **トークン**: `--token` / `YAKUSEL_TOKEN`、未指定なら初回に 192 bit の乱数を生成し `~/.config/yakusel/token` に保存 (以降の起動ではパスだけ表示)。比較は定数時間。
- **認証が必要なのは `POST /v1/translate` のみ**。`/`、`/healthz` (モデル名・キュー長)、`/v1/languages` は認証なし。
- `Usage` に `prefill_ms` (最初のトークンまで) と `queue_ms` (待ち時間) を追加。SSE はクライアント切断を検知して生成を打ち切る。未知の JSON フィールドは 400。
- 長文分割 (§10 の保留事項) は **段落 (空行区切り) 単位** に決定。CLI client と Web UI の両方で、直前 2 段落の原文/訳文を `history` として渡す。
- `yakusel-cli` の `server` feature (既定で有効) を外す (`--no-default-features`) と、client だけのバイナリになる (GPU/モデル依存なし)。

使い方:

```
# サーバ (初回はトークンと CA を生成して表示)
yakusel serve                              # 0.0.0.0:8443, 2B, Q8
yakusel serve --san myhost.example.lan     # 証明書に名前を追加

# 別マシン: サーバの ~/.config/yakusel/ca.pem をコピーしておく
export YAKUSEL_SERVER=https://myhost.local:8443 YAKUSEL_TOKEN=... YAKUSEL_CA=~/yakusel-ca.pem
yakusel client --to en < doc.txt
yakusel client --to ja --form polite -g "owner=オーナー" "The owner approved it."

# ブラウザ: https://<host>:8443/ を開き、トークンを入力 (ca.pem を取り込めば警告なし)
```

### 8.3 M2 の実装メモ

- wgpu 28 (Vulkan のみ)、手書き WGSL 12 本。重みは Q8 (int8 ×4 を u32 に詰め、f16 スケールを 2 個ずつ u32 に詰める) または bf16 (u32 に 2 個) のまま GPU に置き、カーネル内で展開。活性値と KV キャッシュは f32。埋め込み表は GPU に載せず、CPU 側で該当行だけ読んで転送する。
- RoPE の cos/sin はセッション作成時に CPU で表を作って渡す (GPU の `cos` は大きな角度で精度が保証されないため)。カーネル固有の引数は push constant (wgpu の `IMMEDIATES`) で渡す。
- 行列積: T=1 は `gemv1` (8 行/WG、1 行 32 レーン、16 バイト単位の読み込み)、T≤8 は `gemv8`、それより大きいプリフィルは 64×64 タイルの `gemm`。
- **APU では計算ジョブも gfx リングで動き、1 ジョブが約 2 秒を超えるとカーネルが GPU をリセットする** (448 トークンのプリフィルを 1 回で投入して発生)。行列積の FLOP を積算し、32 GFLOP (780M で約 40 ms) ごとに投入を分ける。デスクトップのカクつき防止も兼ねる。
- `--backend auto` (既定) は Vulkan の GPU があれば使い、無ければ CPU。llvmpipe などソフトウェア実装のアダプタは選ばない。
- CPU との比較 (`cargo test --release -p yakusel-backend-wgpu -- --ignored`): Q8/bf16 とも全語彙の logits 差 ≤ 5e-5、greedy 完全一致、600 トークン (2 チャンク) でも 8e-5。7 言語 18 文の訳が CPU と完全一致。
- `YAKUSEL_PROFILE=1` で命令記録時間と GPU 待ち時間を表示。デコード 1 トークンは記録 約 1 ms、GPU 約 42 ms (GPU 律速)。

`yakusel bench` (2B、Q8、32 トークン生成、ベスト値):

| プロンプト長 | CPU プリフィル | Vulkan プリフィル | CPU デコード | Vulkan デコード |
|---|---|---|---|---|
| 32 | 42 tok/s | 148 tok/s | 18.7 tok/s | 22.5 tok/s |
| 160 | 40 tok/s | 187 tok/s | 18.4 tok/s | 20.9 tok/s |
| 512 | 30 tok/s | 167 tok/s | 17.6 tok/s | 19.8 tok/s |
| 1024 | — | 126 tok/s | — | 17.6 tok/s |

Vulkan のデコードは Q8 で実効 約 66 GB/s、bf16 で 約 69 GB/s (13.4 tok/s) とほぼ帯域の上限。ここから速くするには読むバイト数を減らす (Q4) か、1 回の読み込みで複数トークンを確定させる (投機的デコード) しかない。

### 8.4 投機的デコード

- `yakusel-model::speculative`。0.8B (トークナイザ・テンプレートは 2B と完全一致を確認済み) が k トークンを greedy に下書きし、2B が「直前の確定トークン + 下書き k 個」を 1 回の forward で検証して、各位置の argmax と一致する最長の接頭辞を採用し、最初の不一致位置では 2B の予測を採用する。greedy なので **出力は 2B 単体と完全に同一**。
- KV キャッシュは `Session::rewind` で位置を戻すだけ (attention はクエリ位置より後ろのキーを読まないので、捨てたエントリは後で上書きされる)。
- 検証用に T=2..8 行専用の `gemvm{T}` カーネルを追加: K を 512 ずつ区切って全行の活性値をワークグループメモリに置き、8 レーングループ × 2 行で共有、レーン間の和は subgroup shuffle。これで T=5 の forward が 66 → 48 ms (T=1 は 42 ms)。最初の版は活性値をレジスタに抱えて遅く (40〜51 GB/s)、K=10240 の `down_proj` は活性値が L1 に収まらず 35 GB/s だった。`YAKUSEL_PROFILE=2` (タイムスタンプクエリ) で行列の形ごとの時間と帯域を出して特定した。
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

`YAKUSEL_PROFILE=2` で 0.8B の 1 トークン (約 16 ms) を分解すると、カーネル合計 11.2 ms に対して **ディスパッチ間の空き 約 4.4 ms + CPU での記録 0.8 ms** が無駄だった (213 ディスパッチ、各 約 20 µs)。行列積自体は 55〜83 GB/s で帯域上限に近い。

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
- **Backend トレイト**: 演算単位の構成は維持しつつ、`attention(qkv, rows, arena)`、`embed(ids)`、`argmax_rows -> Ids::Device`、`read_ids` (同期点) に改める。ページの割り当て・参照カウント・ページ表は `yakusel-model` 側 (CPU) で管理し、バックエンドは保存領域とページコピーだけを提供する。CPU バックエンドも同じインタフェースを実装して参照実装であり続ける (速度は問わない)。

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
- CLI: `yakusel client` と `translate` は複数段落の入力で document 経路を使う。`--chains 1` で従来どおりの完全な逐次。`yakusel bench` に `rows` (行数スケーリング)、`doc` (文書シナリオ)、`streams N` (並列) を追加。
- Web UI: 複数段落は document 経路で、段落ごとに並行して埋まっていく表示にする。

### 11.9 精度と再現性の方針

- デコード行は常に f32 カーネル (行数によるカーネルの違いで logits は 1e-5 程度しか動かない)。プリフィルは既定で f16 入力の行列演算ユニット (5e-3 程度)、`--prefill-precision f32` で f32。
- KV の再利用や同時実行により、同じリクエストでも KV が「デコードで作られた値」か「プリフィルで作られた値」かが変わり得る。差は上記の範囲で、greedy の結果が変わるのは上位 2 候補がほぼ同点のときだけ。
- 完全な再現性が必要なとき (テスト、比較) は `--prefill-precision f32 --no-kv-cache --max-streams 1`。各段階の合格判定の「訳文が同一」はこの設定と既定設定の両方で確認し、既定設定で差が出た場合は件数と内容を記録する。

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

計測は `yakusel bench rows|decode|set|doc` で行う (`set` は `tests/bench/sentences.tsv` の 19 件、`doc` はモデル同梱の `README_ja.md` から散文 14 段落を実行時に抽出)。

- `bench rows --draft none`: 1 行 37.0 ms、8 行 46.1 ms、9 行 139.4 ms、64 行 153.6 ms、256 行 598.5 ms (11.1 の表を再現)。
- `bench set`: 再現性設定 (`--draft none --prefill-precision f32`) 25.5 tok/s、既定 48.3 tok/s・平均プリフィル 174 ms。両者の訳文は 19 件とも同一。参照として `tests/bench/ref_f32.txt` と `ref_default.txt` を保存 (以降の段階は `bench set --compare` で確認する)。
- `bench doc` (ja→en、履歴 2): 既定で通し 27.9 tok/s、プリフィル 10.5 s (37%)、2 段落目以降の出始め平均 777 ms。再現性設定では通し 17.2 tok/s、出始め 1082 ms。
- **既定と再現性設定で文書の訳が 1 段落だけ異なった** (14 段落中 8 段落目、"across model sizes" と "for each model size" の言い換え、以降の段落への波及なし)。f16 プリフィル (行列演算ユニット) は、500 トークン級のプロンプトでは同点に近い候補の選択を変えうる。11.9 の方針どおり、合否は両設定で確認し、既定設定の差は記録する。

### 11.14 P1 の結果 (エンジン中核)

- `yakusel-model`: `Session` を廃止し、モデルが KV アリーナ (128 トークンのページ、参照カウント付きページプール) と系列 (`SeqId`: ページ表と長さ) を管理する。`forward(&[(SeqId, &[u32])], all_rows)` が複数系列の行を 1 パスで処理する (行メタデータ `PassMeta`: トークン、位置、KV スロット、ページ表オフセット)。`prefill`、`truncate` (投機の巻き戻し)、`fork` (P2 用、共有ページ + 末尾ページのコピー)、コピーオンライト (`reserve`) を用意。既定のアリーナは 16384 トークン (`YAKUSEL_KV_TOKENS` で変更)、最大位置 8192。
- Backend トレイト: `Cache`/`new_cache`/`last_row` を廃止し、`Kv`/`new_kv`/`copy_page`/`begin_pass`/`select_rows`/`reserve_activations` を追加。`embed` と `attention` はパスのメタデータを読む。CPU バックエンドも同じ形に移行 (HF との誤差は従来と完全に同じ)。
- wgpu: コンテキスト `Gpu` を `Arc<Mutex<..>>` で共有するハンドル `WgpuBackend` (`share()`) に分け、本体と下書きが 1 つのデバイス・1 つのエンコーダを使う。埋め込み表は GPU (bf16)。行メタデータは一時バッファからのエンコーダ内コピーで転送し、記録済みの命令との順序を保つ (宛先が固定なのでバインドグループのキャッシュも効く)。`qk_rope` は行ごとの位置と KV スロット、`attention` はページ表を引く (キーのスロットは 1 ステップに 1 回だけ計算して共有メモリに置く)。
- **落とし穴**: 移行直後、512 トークンのプリフィルが 7% 遅くなった。カーネル時間を比べると `down_proj` の行列演算ユニット GEMM だけが +18% (K=10240 で活性値を 40 KB 間隔の 64 行から読むので、バッファがどのメモリに置かれたかに敏感)。KV アリーナや埋め込み表の大きさ・確保順は無関係で、**活性値のバッファを重みより先に確保すると解消** した (専用領域に載るためと見られる)。`reserve_activations` でモデル読み込みの最初に 512 行分のバッファをプールに作っておく。down_proj は 7.6 ms (旧 7.7 ms)。
- 合格判定: テスト全通過 (HF 比較、CPU/GPU 比較 f32・行列演算ユニット両方、カーネル単体、投機 = 通常)。行数スケーリング 1 行 36.3 ms・64 行 153 ms・512 行 1301 ms (P0: 37.0 / 153.6 / 約 1297)。`bench set` 48.4 tok/s (P0 48.3)・再現性設定 25.4 tok/s (25.5)、訳文は両設定とも参照と 19/19 一致。`bench doc` 27.8 tok/s、2 段落目以降の出始め 772 ms (P0 27.9 / 777)、訳文は P0 の既定設定と同一。

### 11.15 P2 の結果 (プレフィックス KV 再利用)

- `Model::start(prompt)` / `finish(seq)`: 終わった系列を消費トークン列ごと LRU で保持し (最大 32 系列、ページ不足時は古い順に解放)、新しいプロンプトとの最長共通接頭辞が 16 トークン以上ならフォークして残りだけを処理する。投機的デコードは本体・下書きとも同じ仕組みを使い、下書きの `start` は最初のトークンを出した後に回す (出始めを遅らせない)。`--no-kv-cache` で無効、`--kv-tokens` でアリーナの大きさ。
- WebGPU は同一バッファ内のコピーを禁止しているので、ページのコピーは 1 ページ分の作業バッファを経由する。
- **履歴の方式を変えた**: 直前 N 組の窓をずらすと毎回プロンプトの先頭が変わって再利用できない (直前 2 組で再利用率 17%)。クライアント側の `HistoryWindow` (`yakusel-api`、CLI、Web UI で同じ規則) は開始位置を固定し、上限 (既定 4000 文字、`--history-chars`) を超えたときだけ、上限の 1/4 に収まる最新の組に切り詰める。切り詰めのたびに残した分を再計算するが、平均すると 1 段落あたり約 1/3 組分で済む (「古い半分を落とす」だと上限によらず 1 組分になり、上限 2000 文字では 14 段落中 4 回の全再計算が起きた)。
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

### 11.12 リスク

- ページ表の間接参照で attention が遅くなる可能性 → P1 の合格条件 (±3%) で確認。デコードの attention は全体の 2% 程度なので影響は小さい見込み。
- wgpu 30 の実験的機能と naga の制約 (8.6) の上に行対応カーネルを書くことになる。行列演算ユニットはプリフィルパスだけで使い、デコードパスは通常の機能だけに保つ。
- 中間カーネルが目標に届かない場合、16 並列の目標は未達になる (8 並列までは既存カーネルで到達できる)。
- 埋め込み表と KV アリーナで GPU メモリが約 2.8 GB 増える (合計約 6 GB、GTT 上限 31 GiB に対して余裕あり)。
