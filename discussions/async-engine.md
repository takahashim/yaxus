# Engine の非同期化と、Web を含めた全体の設計

状態: 設計 (2026-11-14 に全体を見直し、レビューを受けて改訂)。後方互換性は考えない。
下の「出発点」の部分は main に入っている。この文書が提案するのは、その上の設計である。

## 要約

WebGPU は GPU の完了をブロックして待てない。これを理由に、Web は Engine を使わず、
段落の分割、履歴、出力の組み立て、統計を、runtime とは別に自前で持っている。
問題は Engine ではなく、**同期か非同期かで層が割れていること**で、Engine より上の層
(`Generator`、`Session`、文書、server の worker) がすべて同期を前提にしている。

方針は 1 つ。**推論から出力までのスケジューラを 1 つにし、それを非同期で書く。**
同期の環境 (server、CLI) はそれを `block_on` で動かし、Web は `spawn_local` で動かす。
Web 固有のコードは、GGUF の読み込みと JS との接続だけにする。

## 出発点 (実施済み、main に入っている)

- `AsyncReadback` と `Engine::step_async`。`Engine::step` は `SyncReadback` の型に限り、1 回 poll する。
- Web の `Translator` は Engine で翻訳し、prefix 再利用が効いている
  (12 段落の文書の prefill 2212 ms → 約 390 ms、出力は再利用なしとバイト単位で同じ)。
- ネイティブの速度と出力は変わらない (2B、19 文: 下書きあり 61.8 tok/s、なし 26.65 tok/s、19/19 一致)。

これは設計の土台として正しかった。ただ、Web の `Translator` はまだ 1 リクエストずつ、
段落を自前で処理している。ここから先を、層ごとに作り直す。

## いまの構造と、割れている所

```
Backend ─ Model ─ Engine ─ Generator (dyn、同期、Send) ─ Session<T> ─┬─ DocRun / run_document (CLI)
                                                                     ├─ server の worker (Docs、jobs)
                                                                     └─ run_request
yaxus-web: Translator が Model→Engine を直接使い、分割・履歴・出力・統計を自前で持つ (約 300 行の重複)
```

- Web の `translate_inner`、`fit_segment`、`decode_segment`、`Output`、`Stats` は、runtime の
  `DocJob`、`Session`、`DocRun` の働きを別の形で書き直したもの。
- 長い段落を文に分ける処理 (`split_for_context`、トークン数で見る) は Web にしかない。
  native は段落の分割しか持たず、長い段落は 1 つのまま訳す。
- 同時に走れるのは、server の worker だけ (`Docs` で複数の文書をつなぐ)。Web は 1 件ずつ。

## 設計

### 1. Engine: 容量を見て受け付ける

`Engine::admit` は、`target.begin` が KV のページ不足で失敗すると `Event::Error` にしている。
同時に複数のリクエストを走らせる前提では、これは待たせるべき状況であって、エラーではない。

上限は 2 つあり、別々に判定する。

- **系列ごとの長さ** (`max_pos`): 1 つの系列が取れる位置の上限。`prompt + max_new > max_pos` の要求は、
  単独でも収まらないので、`Event::Error` にする (待っても改善しない)。
- **アリーナの物理ページ数** (全系列で共有): 単独で収まる要求は、空きが出るまで `waiting` に残す。
  ストリームが終わるたびに再判定する。

**予約の規則 (初回の実装)。** 各要求の最大ページ数をそのまま予約する。prefix の再利用分は差し引かない。

```
reserve(req) = ceil(min(prompt + max_new, max_pos) / PAGE) + 1
```

- 末尾の `+1` は、投機の検証 (`verify`) が一時的に `len` の先まで書く分 (`k + 1` 行。`draft_k` は 16 以下、`hint_k` も `PAGE` より十分小さい)。
  下書きモデルがあるときは、下書き側のアリーナにも同じ式で予約する。
- 受け付けた要求の予約の合計が、アリーナの総ページ数を超えるなら受け付けない。
  終了済みで cache に残っている系列は、追い出せるので数えない。
- prefix 分を差し引かないのは、検証しやすさのため。再利用で共有されるページは、使用中として一度だけ
  数える必要があり、末尾の部分ページは `fork` がコピーするので、そのページも別に数える必要がある
  (`kv.rs` の `fork`)。この会計は、実際の再利用は既存の `KvCache` に任せたままで、受付の条件だけを
  単純にしておく。実測で詰まりが目立つなら、あとで差し引く式に替える。
- 最悪の出力長 (`max_new`) で予約すると無駄が大きい。同じく、実測で目立つなら、出力が予約を超えた
  ストリームを止めて再計算に回す (preemption) を足す。
- `max_streams` は上限として残すが、実際の同時数は KV の容量が決める。

`step_async` の戻り値の意味は、先の改訂のまま: `Err` はバックエンドの失敗で、エンジンは作り直すまで
使わない。要求ごとの失敗は `Event::Error`。

**キャンセル。** `Engine::cancel(id)` を足す。呼べるのはステップの合間だけ (ステップの Future が
`&mut self` を持っているので、型で保証される)。

- `waiting` にいる要求は、取り除いて `Done { Cancelled }` を出す。
- 実行中のストリームは、印を付け、現在のステップを完了させてから、`sweep` が `Done { Cancelled }` を出す。
- 今の `emit` が `false` を返す経路は、トークンが届いて初めて効くので、容量待ちの要求には効かない。
  `cancel` はそれを補う。両方を残す。

### 2. `Generator`: 非同期で、`Send` を要求しない

```rust
pub trait Generator {
    fn submit(&mut self, req: Request);
    fn step<'a>(&'a mut self, emit: &'a mut dyn FnMut(Event) -> bool) -> LocalBoxFuture<'a, Result<bool>>;
    // fresh_id、abort_all、score、... は今のまま (score も非同期にする)
}
```

- `LocalBoxFuture` (Send でない) にする。WebGPU の Future は Send でないので、`Generator: Send` は外す。
- ネイティブの推論スレッドへ渡す境界でだけ、`Box<dyn Generator + Send>` を要求する。
- ステップごとの boxing は、ステップが ms 単位なので無視できる。必要なら `enum AnyEngine { Cpu, Gpu }`
  にして dyn を避けることもできる (要測定、いまは dyn のまま)。
- `Engine::step` (同期) は、テストと CLI の小さな用途のために `block_on` の薄い包みとして残す。

### 3. `yaxus-runtime` を、ターゲットに依存しない非同期のコアにする

runtime を 2 つに分ける。

- **コア (wasm でも動く)**: `Prompt`、`Memory`、`Session<T>`、`DocJob` / `DocRun`、`Scheduler`。
  `tokio` もファイルも使わない。`futures` の型だけに頼る。
- **`native` フィーチャー**: モデルの解決 (`resolve.rs`、HF キャッシュ)、バックエンド (CPU、GPU) の読み込み (`load.rs`)。

`Session::step`、`DocRun` の実行ループは `async fn` にする。同期版は作らない (呼び出し側が `block_on` する)。

### 4. `Scheduler`: server の worker を一般化する

server の `worker::run` が、すでに正しい形をしている: 入口の queue から job を受け、Engine を回し、
文書のチェーンをつなぎ、イベントを返す。これを runtime のコアに移し、server と Web で共有する。

```rust
pub struct Scheduler { engine: Box<dyn Generator>, tok: Rc<Tokenizer>, memory: Memory, session, docs, ... }

impl Scheduler {
    /// jobs が尽きるまで回る。待つ物が無いときは jobs.next().await で眠る。
    pub async fn run(&mut self, jobs: impl JobSource) -> Result<(), Poisoned>;
}
pub enum Job {
    Single { prompt, max_new_tokens, events },
    Document { job: DocJob, events },
    Cancel { id },
}
```

- イベントの出口は、`yaxus-api` の `StreamEvent` / `DocumentEvent` をそのまま使う
  (`Delta { segment, text }`、`SegmentDone`、`Done { usage }`、`Error`)。server は SSE に、Web は JS の関数に流す。
- `JobSource` は `async fn next()` と `try_next()` を持つ小さなトレイト。server は `tokio::sync::mpsc`
  を包んだもの (今の `Jobs` と同じ上限と計数)、Web は `futures_channel::mpsc` を包んだもの。
- **server の推論スレッド**は、`pollster::block_on(scheduler.run(jobs))` を 1 回呼ぶだけになる。
  ネイティブの future はすべて即座に完了するので、待ち (`jobs.next().await`) 以外で眠らない。
- **キャンセル。** `Job::Cancel` は、新規ジョブの受付の上限 (`jobs.limit()`) に関係なく処理する。
  今の worker は、待機中の要求が上限に達すると job を取り込まないので、キャンセルが後ろに詰まる。
  `Cancel` は別の経路 (専用の小さな channel、または job より先に読む) で受け、`Engine::cancel` を呼ぶ。
  進行中のステップは捨てずに、完了してから対象を終える。
  **文書のキャンセル**は、その文書に属する待機中と実行中の両方のセグメントを取り消し、
  以後のセグメントは投入しない (`DocRun::stop` に `Engine::cancel` を足す)。
- **失敗の方針を 1 つにする。** 今は「panic ならエンジンを捨てて停止」と「`Err` なら全部を失敗にして続行」
  の 2 つがある。`step` の `Err` はバックエンドの失敗でありエンジンは使えないので、両者を
  `Poisoned` に統一する: 全リクエストを失敗にし、`run` は返る。server は 503 と `/healthz` で再起動を促し
  (今の panic の扱い)、Web は `broken` にして Worker を作り直す (今のまま)。
  どの時点で完了を通知するかは、次の節で決める。

### 4a. GPU 障害と完了通知 (WebGPU)

今の Web は、翻訳ごとに GPU のエラースコープを開き、翻訳の最後に `watch.finish().await` で
遅れて届くエラーを確かめてから、成功にしている。複数の翻訳を同じパスで進めると、翻訳ごとに
スコープを開く方式では、障害をどの翻訳のものか決められない。WebGPU のエラースコープは
デバイス単位のスタックで、パスは複数の要求に共通だからである。

- **スコープの単位を、翻訳からパス (step) に替える。** バックエンドが、パスの区間を 1 つのスコープで
  覆う: `begin_pass` で開き、パスの終わりの待ちで閉じる。パスは 1 つずつ直列に await するので、
  スコープは入れ子にならない。
- **スコープの結果を step の成否に含める。** パスの終わりの待ち (argmax の読み戻し) は、
  読み戻しが終わったあとにスコープの結果を取り、エラーがあれば `Err` を返す。
  これにより、読み戻しの前に出したトークンも、後に出す `Done` も、エラーのないパスのものだけになる。
- **トークンと `Done` は、パスが検証されてから出す。** `Engine` は argmax の結果を受け取ってから
  `emit` するので、この順序が自然に守られる。守るべき規則は 2 つ:
  1. `emit` は、そのパスの読み戻しとスコープの結果が `Ok` のあとにだけ呼ぶ。
  2. 読み戻しを持たないパス (完了しない prompt のチャンク) にも、スコープを閉じる待ちを置く。
     `AsyncReadback` に、argmax と別の `end_pass` (既定は何もしない) を足し、Engine が各パスの終わりに必ず呼ぶ。
     チャンクごとに CPU と GPU の往復が 1 回増える。チャンクの計算に比べて小さいはずだが、未測定 (下の決めきれていない点)。
- **帰属は試みない。** エラーがあれば、そのパスに乗っていた要求のどれのせいかは分からないので、
  デバイス全体を信用しない (`Poisoned`)。全要求 (実行中と待機中) を `Error` にし、`Done` は出さない。
  デバイスロスト (`device.lost`) も同じ経路にする。
- **ネイティブ**は、スコープを持たず、`end_pass` も何もしない。
- これで、現行の `watch_errors()` / `finish()` と、E2E の GPU 障害のケース (遅れて届くエラー、
  割り当て失敗、デバイスロスト) を、スコープの位置を替えただけで引き継げる。

### 5. 文書の構築を共通にする

共通化には、挙動を変えない部分と、訳を変える部分がある。**別の段階にする。**

**挙動を変えない共通化 (段階 3)。** Web の `Translator` を共有の `Scheduler` に載せる。
分割、履歴、チェーンの方針は、今のそれぞれのものをそのまま使う。

- Web: セグメントは今の `split_for_context` で作り (`Segment { text, paragraph }` を `DocJob` に渡せるようにする)、
  チェーンは 1 本、履歴は文字数 (`HISTORY_CHARS`) で、**出力の上限に達した訳は次の履歴に入れない**。
- native: `DocRun` は `Finish::Length` の訳も次の履歴に渡している。この差は、後続のプロンプトを変える。
  共通化の段階では、`DocOptions` に「上限に達した訳を履歴に入れるか」の設定を足して、
  それぞれの今の挙動のままにする (native は入れる、Web は入れない)。
- 確かめること: 同じ入力で、各セグメントのプロンプトのトークン列が、今の Web と一致する。

**訳を変える変更 (段階 4)。** 1 つずつ入れて、文書の所要時間と訳の一貫性で評価する。

- 長い段落の文への分割を、native の `DocJob` にも入れる。
- 履歴の上限を、文字数からトークン数にする。
- Web のチェーンの数を増やす (並列)。
- 「上限に達した訳を履歴に入れるか」を 1 つに決める。

評価は、19 文のセットと `bench doc` の文書で、訳が変わった件数と、所要時間を見る
(同じ入力で変わる訳は、履歴の違いから後続に広がるので、件数を見る)。

### 6. Web: 薄いグルー

`yaxus-web` に残すのは次だけ。

- GGUF のストリーミング読み込みと WebGPU の初期化 (今のまま)。
- `Rc<RefCell<..>>` を持たず、**投入と駆動を分ける**。`load_model` が `Runtime` を返し、
  そのコンストラクタが `Scheduler::run` を `spawn_local` で 1 回だけ起動する。
  JS からの `translate(request, on_event) -> Promise` は、job を `JobSource` に積み、完了で解決する
  Promise を返すだけ。wasm-bindgen の `&mut self` を await 中に再入する問題は、これで起きない
  (`&self` で積むだけだから)。
- `cancel(id)`。ページの中止ボタンと、将来のタブの切り替えで使う。
- `Poisoned` で `broken` にし、Worker を作り直す (今のまま)。E2E の GPU 障害のケースがそのまま使える。
- ページは、サーバーの Web UI と同じく、段落が並行して埋まる表示にする
  (イベントの `segment` で場所が決まる)。サーバーの UI と共通の部分は、トランスポート
  (SSE か、Worker のメッセージか) だけを差し替える形にできる。まとめるのは後でよい。
- Web の設定: `max_streams` は KV の容量から決める。KV のアリーナは、チェーンの数 × 1 セグメントの
  コンテキストに合わせて、`MAX_CONTEXT` の固定 (4,096) をやめる。0.8B は 1 ページ 2.1 MB なので、
  128 ページ (16k トークン) で約 270 MB。デバイスの `max_buffer_size` と空きを見て決める。
  応答キャッシュは、E2E の GPU 障害のケースが GPU を通る必要があるので、いまは切る。

### 7. 後回しにするもの

- **下書きモデルの Web 対応**: Engine は対応済みだが、2 つのモデルを 1 つのデバイスに載せるメモリと、
  k 回の連続した読み戻しの往復が割に合うかを測ってから。
- **デコードの往復を隠す**: 次のトークンを GPU 上で次のパスへ渡し、読み戻しを待たずに次のパスを投入する
  (ネイティブの投機でやっている GPU 上のトークン受け渡しの応用)。ストリーム 1 本のときの速度に効くが、
  複数のチェーンが走るときは並行で隠れる。まず 1 トークンあたりの時間のうち、往復の待ちが
  どれだけかを `YAXUS_PROFILE` で測る。
- **ループ検出と回復**、**sliding window の KV の解放**、**OPFS へのモデルのキャッシュ**、
  **2B/8B の Web 対応**、**デバイスの自動選択**、**量子化ごとの品質表示**。
  plamo-translate-desktop との比較で挙がった項目で、この設計の上に独立に足せる。
  ループ検出は `Session` に置く (全フロントエンドに効く)。

## 実施順

各段階で、ネイティブの出力が変わらないこと (19 文のセットが 19/19 一致)、速度が
ばらつきの範囲に収まること (交互に複数回)、既存のテストと Chrome E2E が通ることを確かめる。
共通化の段階では訳を変えない。訳を変える変更は、段階 4 で 1 つずつ入れる。

1. **runtime を非同期のコアにする。** `Generator` を非同期にし (`Send` は外す)、`Session`、
   `DocRun` / `run_document` を async にする。`Scheduler` を server の worker から取り出し、
   server と CLI をそれに載せ替える。`native` フィーチャーの分離もここで行う。
   (ネイティブだけで検証できる。Web はまだ触らない)
2. **Engine の容量を見た受付と、キャンセル。** 予約の規則 (prefix を差し引かない) で `waiting` に残す。
   `Engine::cancel` と、`Cancel` が受付の上限に関係なく処理されること。`NullBackend` の単体テストで、
   ページ不足でも待つこと、収まらない要求だけがエラーになること、待機中の要求がキャンセルできることを確かめる。
3. **Web を `Scheduler` に載せる (訳は変えない)。** パス単位のエラースコープ (4a) を入れ、
   `Translator` を `Runtime` に置き換える。確かめること:
   1. 同じ入力で、各セグメントのプロンプトと出力のトークン列が、現行の Web と一致する。
   2. 2 件の別々の文書を同時に投入すると、同じ decode パスで両方が進み、それぞれの出力は単独のときと一致する。
   3. 異なるプロンプトの間で prefix が再利用される (`reused > 0` かつ `cached == false`)。
   4. GPU 障害 (デバイスロスト、検証エラー、遅れて届くエラー、割り当て失敗) の後に、再読み込みで復帰でき、
      障害のあったパスの `Done` が出ない。
   5. 容量待ちの要求と、実行中の要求のキャンセル。
4. **訳を変える変更。** 5 の節のとおり、1 つずつ入れて評価する。
5. 後回しの項目を、測ってから個別に。

## 決めきれていない点

- `Generator` を dyn のままにするか、`enum AnyEngine` にするか。後者はステップごとの boxing を避けるが、
  バックエンドを増やすたびに分岐が増える。ステップが ms 単位なので、いまは dyn のままでよいと考える。
- KV の予約 (prefix を差し引かず、最悪の出力長で予約する) が、実際にどれだけ待たせるか。
  Web の 0.8B で、チェーン 4 本が同時に走るときを、段階 3 の測定で見て、差し引く式や preemption が要るか決める。
- 完了しないチャンクの `end_pass` が、prefill の速度にどれだけ影響するか (見積もりでは小さい)。段階 3 で測る。
- server の UI と Web のページを 1 つの UI にまとめるか (トランスポートだけの差し替え)。
  この設計の範囲外だが、イベントを `yaxus-api` に揃えたことで可能になる。
