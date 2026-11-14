# Engine の非同期化 (Web でも Engine を使う)

状態: 検討 (未着手)。2026-11-14。

## 背景

`crates/yaxus-web` (WASM + WebGPU) は、`Engine` を使わず `Model<GpuBackend>` を直接 1 ステップずつ
await して回している (DESIGN.md §11.1)。WebGPU の読み戻しは await でしか取れず、`Backend` の
同期点 (`argmax_rows`、`to_host`) が wasm では panic するため。

その結果、Web では次のものを使っていない。

- ページ化 KV の共有と共通 prefix の再利用 (`prefix_cache: false`)
- 複数リクエストのバッチ
- 翻訳メモリのヒント
- 文書のチェーン (`yaxus-runtime` の `DocJob`、`Session` は Engine 越し)

plamo-translate-desktop のブラウザ拡張 (wllama) はスロット 4 本でバッチを回し、80 段落で 66 秒が
46 秒になっている (RX 9070 XT、2B)。Web でバッチと文書翻訳を使うには、Engine を非同期で回せる必要がある。

## 現状の同期点

`Backend` トレイトは「各 op は非同期でよく、同期するのは `argmax*` と `to_host` だけ」としている。
Engine でそれに当たるのは 4 か所だけ。

| 場所 | 内容 |
|---|---|
| `prefill_step` (`engine/mod.rs:395`) | プロンプトが完了した行の argmax |
| `decode_step` (`engine/mod.rs:427`) | バッチ全体の argmax |
| `verify` (`engine/mod.rs:507`) | 投機の検証 |
| `draft_tokens` (`engine/mod.rs:546`) | 下書きモデルの argmax。k 回の連続した待ち |

ほかに `Model::generate` と `Model::score` (`lib.rs`) にも同期点がある。`Backend` の実装は
`GpuBackend`、`CpuBackend`、テスト用の `NullBackend` の 3 つ。

## 方針: async 版を本体にして、同期版はその上に載せる

別実装を 2 つ持つと挙動がずれるので、`step_async` を唯一の実装にする。

### 1. Backend の同期点を async にする

- `argmax_rows` と `to_host` を `async fn` に置き換える (トレイトの `async fn`、edition 2024)。
- ネイティブの実装は即座に完了する Future を返す。
- `#[cfg(wasm32)]` の panic スタブ、`to_host_box`、`argmax_rows_async` / `to_host_async` の個別メソッドが不要になる。
- `Engine<B>` は `B` に対してジェネリックで dyn を使わないので、Future の boxing は要らない。

### 2. Engine は `step_async` に一本化する

- `prefill_step`、`decode_step`、`verify`、`spec_round`、`propose`、`draft_tokens` を `async fn` にする。
- `emit: &mut dyn FnMut(Event) -> bool` は同期のままでよい。
- `prefill_step` の `.map(|logits| ...argmax_rows...)` は await できないので書き直す。
- 同期の `step` は `pollster::block_on(self.step_async(emit))` にする (`pollster` は `Model::load` が既に使っている)。
- `Generator` トレイトは同期のまま残し、server、CLI、runtime の呼び出し側は変えない。

### 3. 時計を `web-time` にする

Engine と tuner の `std::time::Instant` は wasm で panic する。GPU クレートが使っている `web-time`
に差し替える (ネイティブでは std と同じ)。

## 第 2 段階: runtime を Web に通す

`Session` と文書の実行ループ (`run.rs`、`document/run.rs`) が `dyn Generator::step` を同期で呼んでいる。
Web で文書翻訳をするには、ここも async にする。

- `Generator` に `step_async` (`LocalBoxFuture` を返す) を足す。既定実装は `step` の即完了版にし、Engine が上書きする。
- `Session::step` の本体はイベント処理の関数にまとめ、同期版と async 版の入口を薄く 2 つ持つ。ループの殻は重複を許す。
- `yaxus-runtime` は `yaxus-backend-cpu` とファイル I/O (`resolve.rs`、`load.rs`) に依存している。wasm では `native` フィーチャーで外す。`session`、`document`、`prompt`、`memory` は純粋なのでそのまま使える。

## 性能と検証

- ネイティブの回帰: 1 ステップに増えるのは即完了の poll 1 回。投機デコードは `draft_tokens` が 1 トークンずつ待つので、`spec_round` が一番影響を受ける。`yaxus bench` を 2B と 0.8B、下書きあり・なしで測り、既存の数字 (0.8B の CLI は 72.9 tok/s など) が変わらないことを受け入れ条件にする。
- 挙動の同一性: 既存のテスト (`NullBackend`、CPU のゴールデン、`vs_cpu`) がそのまま通ること。Web の出力が今の CLI と完全一致すること。
- Web での確認: headless Chrome で、1 リクエストの翻訳が `Model` を直接回す今の経路と同じトークンになること。そのうえで同時 2 リクエストと共通 prefix の再利用 (`StartInfo.reused`) を確認する。

## 注意点

1. **待ちのたびに `setTimeout(0)` を使っている。** `event_loop_turn` (`dev/wgpu.rs:672`) は readback のたびに `setTimeout(0)` で 1 回待つ。ブラウザは入れ子のタイマーを 4 ms 程度に引き上げる場合があるので、1 トークン 14.7 ms のうち数 ms がこの待ちかもしれない (未測定)。`map_async` の Promise を直接 await する形で足りるかを先に試す価値がある。非同期化の前後を問わず、Web の速度に効く。
2. **下書きモデルは Web では最初は外す。** `draft_tokens` は k 回の await になる。ブラウザの 1 往復あたりの待ちを考えると得になるかは測るまで分からない。まず下書きなしで始め、`tuner` が自動で 0 を選ぶかどうかは後で確かめる。
3. **Worker 化とは独立。** Worker に移すときも async 版はそのまま使えるので、順序の制約はない。
4. **`Generator::score` と `Model::generate`。** Web でも使うなら async 版が要るが、翻訳の経路には要らないので後回しでよい。

## 進め方

1. Backend の同期点を async 化し、Engine を `step_async` に一本化する。ネイティブの bench で回帰がないことを確認する (`yaxus-model` と各バックエンドだけ。server、CLI は無変更)。
2. `yaxus-web` の `Translator` を Engine に切り替え、`prefix_cache` と 2 リクエスト同時実行を確認する。
3. `Generator::step_async` と `Session` の async 化、runtime の `native` フィーチャー分離を行い、Web に文書翻訳を載せる。

先に注意点 1 (`setTimeout` の待ち) を計測するかどうかは未決。

## 関連: plamo-translate-desktop との比較から出た他の改善候補

この検討のきっかけになった比較 (2026-11-13、14) で挙がった項目。優先度は下がる。

- 出力ループの検出と回復 (周期 64 トークン以上が 3 回以上。小さい単位、文ごと、DRY サンプラーの順に回復)。Web 版は `max_new` で止まるだけなので同じ弱点がある。
- 長い段落の分割 (1 チャンク約 1,024 トークン)。履歴の上限は文字数でなくトークン数で持つ。
- sliding window 層の KV で、窓を外れたページを解放しているかの確認。
- OPFS へのモデルのキャッシュ (今は毎回 GGUF を選ぶ)、2B 対応、Worker 化。
- デバイスの自動選択と、メモリ見積もり (ファイルサイズ × 1.2 + 384 MiB)。
- 量子化ごとの品質表示。

トークナイザは llama.cpp の `plamo3` と同じアルゴリズム (末尾からの Viterbi、`round(score*1e4)`、バイトフォールバック) で、改善点はない。
