# Engine の非同期化 (Web でも Engine を使う)

状態: 検討 (未着手)。2026-11-14 作成、同日にレビューを受けて改訂。

## 背景

`crates/yaxus-web` (WASM + WebGPU) は、`Engine` を使わず `Model<GpuBackend>` を直接 1 ステップずつ
await して回している。WebGPU はブロックして待てないので、`Engine` はコンパイルできない
(`Engine<B: SyncReadback>`、`SyncReadback` は CPU とネイティブ GPU だけが実装する)。

その結果、Web では次のものを使っていない。

- ページ化 KV の共有と共通 prefix の再利用 (`prefix_cache: false`)
- 複数リクエストのバッチ
- 翻訳メモリのヒント
- 文書のチェーン (`yaxus-runtime` の `DocJob`、`Session` は Engine 越し)

plamo-translate-desktop のブラウザ拡張 (wllama) はスロット 4 本でバッチを回し、80 段落で 66 秒が
46 秒になっている (RX 9070 XT、2B)。Web でバッチと文書翻訳を使うには、Engine を非同期で回せる必要がある。

## 現状 (改訂の前提)

- `Backend` は読み戻しを持たない。読み戻しは `SyncReadback` (`argmax`、`argmax_rows`、`to_host`) で、
  ネイティブ専用 (`yaxus-model/src/lib.rs:171`)。`Model::generate`、`Model::score`、`Engine` はこれに依存する。
- WebGPU の読み戻しは `read_async` (`yaxus-backend-gpu/src/dev/wgpu/web.rs:250`)。`map_async` の
  コールバックを待つ。`setTimeout(0)` は使っていない。性能比較の基準はこの現行の経路。
- Engine の同期点は 4 か所: `prefill_step`、`decode_step`、`verify`、`draft_tokens` の argmax。
- Web の `Translator` は、読み戻しとエラースコープの失敗を GPU 障害として `broken` にし、Worker を
  作り直させる。Worker は `translate` を 1 件ずつ直列に処理する (`web/local/worker.js:12`)。
- Web の KV アリーナは合計 4,096 トークン (`MAX_CONTEXT`)、1 ページ 128 トークンなので 32 ページ。

## 方針: スケジューラは `Engine::step_async` に一本化する

別実装を 2 つ持つと挙動がずれるので、async 版を唯一の実装にし、同期版はその上に載せる。
ただし、最初に変えるのは Engine が使う argmax だけに絞る。

### 1. 読み戻しは argmax だけを共通化する

- Engine 向けに、`Result<Vec<u32>>` を返す非同期の argmax を持つトレイトを用意する
  (名前は未定。`SyncReadback` を置き換えず、並べて置く)。
- CPU とネイティブ GPU は、既存の同期実装をその場で呼んで即完了の Future を返す。
- WebGPU は `read_async` を使う。読み戻しの失敗が `Err` で返る。
- `Engine::step_async` を本体にする。`prefill_step`、`decode_step`、`verify`、`spec_round`、
  `propose`、`draft_tokens` を `async fn` にする。`emit: &mut dyn FnMut(Event) -> bool` は同期のままでよい。
- `Engine::step` は、`SyncReadback` を持つ型に限って `block_on(step_async)` で包む
  (`pollster` は `Model::load` が既に使っている)。これで `Model::generate` と `score` の同期 API、
  `Generator` を使う server、CLI、runtime の呼び出し側は変わらない。
- `to_host` の非同期化は、Engine の経路では要らないので扱わない。
- `Engine` と tuner の `std::time::Instant` は wasm で panic するので、`web-time` に差し替える
  (ネイティブでは std と同じ)。

### 2. GPU 障害はバッチ全体に伝える

現行の Web は、GPU 障害で `Translator` を `broken` にして Worker を作り直す。この挙動を保つ。

- GPU 障害 (読み戻しの失敗、エラースコープの失敗、デバイスロスト) は、個々の `Event::Error` に変換せず、
  `step_async` 自体の `Err` として返す。今の Engine は、forward などの失敗を各ストリームの `error` にして
  `Event::Error` を出すので、そこを区別する必要がある。
- KV 不足のようなリクエスト単位のエラーは、これまでどおり `Event::Error` にする。
- `forward` は読み戻しを待つ前に KV を更新する。そのため、進行中の Future を破棄して同じ Engine を
  再開する経路は作らない。キャンセルは、ステップを完了させてから `emit` が `false` を返す今の経路で、
  対象のリクエストだけを終了させる。`step_async` が `Err` を返した Engine は、作り直すまで使わない。

### 3. Worker 内で 1 つの駆動ループにリクエストを集める

- 今の Worker は `translate(&mut self)` を完了まで直列化している。これを、1 つの Engine に
  リクエストを投入し続け、`step_async` を回し続ける駆動ループに変える。
- KV アリーナが合計 4,096 トークンなので、`max_streams` を増やす前に、同時に受け付ける要求の
  合計ページ数を見るか、容量が足りないときに待たせる方針を決める。今の `admit` は、ページが足りないと
  `Event::Error` にする。
- 文書チェーンをつなぐ段階では `Generator: Send` を見直す。`LocalBoxFuture` を足すだけでは、
  Send でない WebGPU の Future を載せられない。`Send` は、ネイティブの推論スレッドへ渡す境界でだけ
  要求し、`Session` の `std::time::Instant` も `web-time` に移す。

## 進め方

1. Engine 向けの非同期 argmax と `Engine::step_async` を入れ、`Engine::step` を `block_on` で包む。
   ネイティブの回帰がないことを確認する。続けて、Web の単一リクエストの翻訳を Engine に載せ替える。
2. バッチを有効にする。同じ Engine に 2 件を先に投入し、同じ decode パスで両方が進むことを確認する。
   KV の容量の方針をここで決める。
3. `Generator` と `Session` の async 化、runtime の `native` フィーチャー分離を行い、文書チェーンを載せる。

## 受け入れ条件

- 同じブラウザ、GGUF、プロンプトで、現行の Web と単一リクエストのトークン列が一致する。
  (既存の Chrome E2E: `web/local/e2e.cjs`)
- 2 件のバッチが、同じ decode パスで両方進む。
- 異なるプロンプトの間で prefix が再利用される (`StartInfo.reused > 0` かつ `cached == false`)。
- GPU 障害の後、モデルの再ロードで復帰できる。
- ネイティブの性能は、同じ環境で変更の前後を複数回測り、ばらつきを踏まえて判定する。
  基準は 0.8B と 2B の `yaxus bench` (下書きあり・なし)。投機デコードは `draft_tokens` が 1 トークンずつ
  待つので、一番影響を受けやすい。

## 注意点

- **下書きモデルは Web では最初は外す。** `draft_tokens` は k 回の await になる。ブラウザで得になるかは
  測るまで分からない。
- **Worker 化は済んでいる。** Worker 内で async 版がそのまま使える。
- **`Generator::score` と `Model::generate`。** 同期のまま残す。Web の翻訳の経路には要らない。

## 関連: plamo-translate-desktop との比較から出た他の改善候補

この検討のきっかけになった比較 (2026-11-13、14) で挙がった項目。優先度は下がる。

- 出力ループの検出と回復 (周期 64 トークン以上が 3 回以上。小さい単位、文ごと、DRY サンプラーの順に回復)。
- 長い段落の分割 (1 チャンク約 1,024 トークン)。履歴の上限は文字数でなくトークン数で持つ。
- sliding window 層の KV で、窓を外れたページを解放しているかの確認。
- OPFS へのモデルのキャッシュ (今は毎回 GGUF を選ぶ)、2B 対応。
- デバイスの自動選択と、メモリ見積もり (ファイルサイズ × 1.2 + 384 MiB)。
- 量子化ごとの品質表示。

トークナイザは llama.cpp の `plamo3` と同じアルゴリズム (末尾からの Viterbi、`round(score*1e4)`、
バイトフォールバック) で、改善点はない。
