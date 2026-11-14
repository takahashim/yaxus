//! The scheduler: runs many requests over one model (and an optional draft model for
//! speculative decoding), one forward pass per `step`.
//!
//! - Prompts are processed in chunks; while other requests are decoding, a prompt chunk
//!   is at most `busy_prefill_rows` rows and alternates with decode passes.
//! - Decoding requests advance together: one pass takes the next token of up to
//!   `decode_rows` of them (round robin), reading the weights once for all.
//! - With at most `spec_max_streams` decoding requests and a draft model, a request
//!   advances by a speculative round instead (the draft proposes up to `draft_k` tokens,
//!   as many as the measured acceptance and pass times make worthwhile, or none; the
//!   target checks them in one pass). The draft catches up from the target's tokens,
//!   so a request can switch between speculative and batched decoding at any time.
//!
//! - A request may carry a `hint`, tokens its output is likely to repeat (an earlier
//!   translation of the same text). Where the output so far matches the hint, the
//!   following hint tokens are verified like draft tokens, without running the draft.
//! - Finished outputs are kept in a small response cache keyed by the prompt; a repeated
//!   request is answered from it without any pass.
//!
//! Decoding is greedy, so every request produces the target model's greedy output
//! whatever else is running (up to the summation-order differences of the kernels).

mod cache;
mod hint;
mod tuning;

use std::collections::VecDeque;
use std::future::Future;
use std::task::{Context, Poll, Waker};

use anyhow::Result;
use web_time::Instant;

use self::cache::ResponseCache;
use self::hint::HintCursor;
use self::tuning::{DraftTuner, StreamTuning};
use crate::{AsyncReadback, Backend, Model, SeqId, StartInfo, SyncReadback};

pub type ReqId = u64;

#[derive(Debug, Clone)]
pub struct Request {
    pub id: ReqId,
    pub prompt: Vec<u32>,
    pub max_new_tokens: usize,
    /// Tokens that end the generation (not emitted).
    pub stop: Vec<u32>,
    /// Tokens the output is likely to repeat, proposed as draft tokens where the output
    /// matches them. It never changes the output, only the speed.
    pub hint: Vec<u32>,
}

impl Request {
    pub fn new(id: ReqId, prompt: Vec<u32>, max_new_tokens: usize, stop: Vec<u32>) -> Self {
        Request { id, prompt, max_new_tokens, stop, hint: Vec::new() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// A stop token was generated.
    Stop,
    /// `max_new_tokens` reached.
    Length,
    /// The event sink returned `false` for one of its tokens.
    Cancelled,
}

#[derive(Debug, Clone)]
pub enum Event {
    Token { id: ReqId, token: u32 },
    Done { id: ReqId, finish: Finish, generated: usize, start: StartInfo },
    Error { id: ReqId, message: String },
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Requests decoding or prefilling at once (the rest wait).
    pub max_streams: usize,
    /// Most requests advanced by one decode pass. With f32 kernels 8 rows are the
    /// cheapest per row on a Radeon 780M (2B: 38 ms for 8 rows, 108 ms for 16); with the
    /// integer kernels 16 are (26 and 38 ms). The default here is 8; callers that load
    /// through `yaxus-runtime` get the precision's own (`default_decode_rows`).
    pub decode_rows: usize,
    /// Prompt rows per pass while other requests are decoding.
    pub busy_prefill_rows: usize,
    /// Most draft tokens per speculative round (the engine picks the length per round
    /// from the measured acceptance and pass times, see `DraftTuner::choose`).
    pub draft_k: usize,
    /// Speculative decoding is used while at most this many requests are decoding.
    pub spec_max_streams: usize,
    /// Hint tokens verified per pass (verifying 8 rows costs little more than 1).
    pub hint_k: usize,
    /// Finished requests kept in the response cache (0: no cache).
    pub response_cache: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            max_streams: 16,
            decode_rows: 8,
            busy_prefill_rows: 64,
            draft_k: 7,
            spec_max_streams: 1,
            hint_k: 7,
            response_cache: 256,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct SpecStats {
    /// Target verification passes.
    pub rounds: usize,
    /// Draft tokens proposed / accepted by the target.
    pub proposed: usize,
    pub accepted: usize,
    /// Rounds that verified hint tokens instead of draft tokens (included above).
    pub hint_rounds: usize,
    /// Single decode steps taken instead of a round (the draft was not worth it).
    pub plain_steps: usize,
    /// Requests answered from the response cache.
    pub cache_hits: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// `done` prompt tokens are in the target's KV.
    Prefill { done: usize },
    /// `next` was emitted and is not yet consumed by the target.
    Decode { next: u32 },
}

struct Stream {
    req: Request,
    ts: SeqId,
    ds: Option<SeqId>,
    phase: Phase,
    generated: usize,
    /// Emitted tokens.
    out: Vec<u32>,
    hint: HintCursor,
    tuning: StreamTuning,
    start: StartInfo,
    finish: Option<Finish>,
    error: Option<String>,
    /// KV pages kept for the request from its start to its end (see [`Engine::admit`]).
    reserved: usize,
}

/// Where the tokens of a speculative round come from.
enum Source {
    /// The hint, from index `at`.
    Hint { at: usize },
    /// The draft model, whose round started at `started`.
    Draft { started: Instant },
}

pub struct Engine<B: Backend> {
    target: Model<B>,
    draft: Option<Model<B>>,
    cfg: EngineConfig,
    spec: SpecStats,
    waiting: VecDeque<Request>,
    streams: Vec<Stream>,
    /// Round-robin start for decode passes.
    rr: usize,
    /// Alternation between prompt chunks and decode passes.
    prefill_turn: bool,
    next_id: ReqId,
    last_start: StartInfo,
    cache: ResponseCache,
    tuner: DraftTuner,
    /// The stop token that ended the last stopped request: hints (outputs without their
    /// stop token) are extended by it, so the end of an output is verified with the hint.
    end_token: Option<u32>,
    /// Events of requests cancelled while they waited, delivered at the next step.
    pending: Vec<Event>,
}

impl<B: AsyncReadback> Engine<B> {
    /// `draft` must share `target`'s backend context when given.
    pub fn new(target: Model<B>, draft: Option<Model<B>>, cfg: EngineConfig) -> Result<Self> {
        if let Some(d) = &draft {
            anyhow::ensure!(
                d.cfg.vocab_size == target.cfg.vocab_size,
                "draft vocabulary ({}) differs from the target's ({})",
                d.cfg.vocab_size,
                target.cfg.vocab_size
            );
            anyhow::ensure!((1..=16).contains(&cfg.draft_k), "draft length must be 1..=16");
        }
        // A zero row budget would make every prompt pass empty and the engine spin.
        anyhow::ensure!(
            cfg.decode_rows >= 1 && cfg.max_streams >= 1 && cfg.busy_prefill_rows >= 1,
            "decode_rows, max_streams and busy_prefill_rows must be at least 1"
        );
        Ok(Engine {
            target,
            draft,
            cfg,
            spec: SpecStats::default(),
            waiting: VecDeque::new(),
            streams: Vec::new(),
            rr: 0,
            prefill_turn: false,
            next_id: 1 << 48,
            last_start: StartInfo::default(),
            cache: ResponseCache::default(),
            tuner: DraftTuner::default(),
            end_token: None,
            pending: Vec::new(),
        })
    }

    /// The backend, e.g. for the device's error handling in the browser.
    pub fn backend(&self) -> &B {
        &self.target.backend
    }

    /// Speculative decoding and response cache counts so far.
    pub fn spec_stats(&self) -> &SpecStats {
        &self.spec
    }

    /// Free KV pages of the target and the draft model.
    pub fn free_pages(&self) -> (usize, Option<usize>) {
        (self.target.free_pages(), self.draft.as_ref().map(Model::free_pages))
    }

    /// Changes the longest draft run (1..=16), as [`EngineConfig::draft_k`].
    pub fn set_draft_k(&mut self, k: usize) -> Result<()> {
        anyhow::ensure!((1..=16).contains(&k), "draft length must be 1..=16");
        self.cfg.draft_k = k;
        Ok(())
    }

    /// Changes the response cache size, as [`EngineConfig::response_cache`] (0: off).
    pub fn set_response_cache(&mut self, entries: usize) {
        self.cfg.response_cache = entries;
    }

    /// Turns prefix reuse on or off in both models (see [`Model::set_prefix_cache`]).
    pub fn set_prefix_cache(&mut self, on: bool) {
        self.target.set_prefix_cache(on);
        if let Some(d) = self.draft.as_mut() {
            d.set_prefix_cache(on);
        }
    }

    /// A request id not used by callers that number their requests from 0.
    pub fn fresh_id(&mut self) -> ReqId {
        self.next_id += 1;
        self.next_id
    }

    pub fn submit(&mut self, req: Request) {
        self.waiting.push_back(req);
    }

    /// Requests waiting for a stream slot.
    pub fn waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Requests being processed.
    pub fn active(&self) -> usize {
        self.streams.len()
    }

    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.streams.is_empty() && self.pending.is_empty()
    }

    /// Cancels request `id`; `false` if it is not waiting or running (it has ended, or
    /// never existed). It ends with `Event::Done { finish: Cancelled }` at the next
    /// [`step_async`](Engine::step_async), whichever state it is in: a waiting request
    /// never starts, and a running one is not given another pass (its KV is kept for
    /// prefix reuse, as for one that ended on its own). Callable only between steps,
    /// which the borrow of a step in progress guarantees.
    pub fn cancel(&mut self, id: ReqId) -> bool {
        if let Some(at) = self.waiting.iter().position(|r| r.id == id) {
            self.waiting.remove(at);
            self.pending.push(Event::Done { id, finish: Finish::Cancelled, generated: 0, start: StartInfo::default() });
            return true;
        }
        match self.streams.iter_mut().find(|s| s.req.id == id && s.finish.is_none() && s.error.is_none()) {
            Some(s) => {
                s.finish = Some(Finish::Cancelled);
                true
            }
            None => false,
        }
    }

    /// Drops every request, waiting or running, without events, and frees their KV (for
    /// recovery after a failed [`step`](Engine::step): the caller has given up on them).
    pub fn abort_all(&mut self) {
        self.waiting.clear();
        for s in self.streams.drain(..) {
            self.target.free_seq(s.ts);
            if let (Some(d), Some(dm)) = (s.ds, self.draft.as_mut()) {
                dm.free_seq(d);
            }
        }
        self.rr = 0;
        self.prefill_turn = false;
        self.tuner.lose_sync();
    }

    /// Runs one pass. `emit` receives tokens and completions; returning `false` for a
    /// token cancels that request. Returns `false` when there was nothing to do.
    ///
    /// An `Err` is a failure of the backend (a read-back failed): it belongs to the whole
    /// batch, not to one request, and the engine is not to be used again. A failure of
    /// one request is an [`Event::Error`] for it.
    ///
    /// The step is never dropped half-way: its pass has already changed the KV by the
    /// time the read-back is awaited. Cancel a request through `emit`.
    pub async fn step_async(&mut self, emit: &mut dyn FnMut(Event) -> bool) -> Result<bool> {
        // The ends of requests cancelled since the last step come first, so that a
        // cancelled stream is not given another pass and its pages are free for admission.
        let cancelled = !self.pending.is_empty() || self.streams.iter().any(|s| s.finish.is_some());
        for event in std::mem::take(&mut self.pending) {
            emit(event);
        }
        self.sweep(emit);
        self.admit(emit);
        if self.streams.is_empty() {
            return Ok(cancelled);
        }
        let decoding: Vec<usize> =
            (0..self.streams.len()).filter(|&i| matches!(self.streams[i].phase, Phase::Decode { .. })).collect();
        let prefilling: Vec<usize> =
            (0..self.streams.len()).filter(|&i| matches!(self.streams[i].phase, Phase::Prefill { .. })).collect();
        match prefilling.first() {
            Some(_) if decoding.is_empty() || self.prefill_turn => {
                self.prefill_turn = false;
                let busy = !decoding.is_empty();
                self.prefill_step(&prefilling, busy, emit).await?;
            }
            _ => {
                self.prefill_turn = true;
                let speculate = |s: &Stream| self.draft.is_some() || !s.req.hint.is_empty();
                if decoding.len() <= self.cfg.spec_max_streams && decoding.iter().all(|&i| speculate(&self.streams[i]))
                {
                    for &i in &decoding {
                        if let Err(e) = self.spec_round(i, emit).await {
                            self.streams[i].error = Some(request_failure(e)?);
                        }
                    }
                } else {
                    let n = decoding.len().min(self.cfg.decode_rows);
                    let start = self.rr % decoding.len();
                    self.rr = self.rr.wrapping_add(n);
                    let picked: Vec<usize> = (0..n).map(|j| decoding[(start + j) % decoding.len()]).collect();
                    if let Err(e) = self.decode_step(&picked, emit).await {
                        let message = request_failure(e)?;
                        for &i in &picked {
                            self.streams[i].error = Some(message.clone());
                        }
                    }
                }
            }
        }
        self.sweep(emit);
        Ok(true)
    }

    /// Starts waiting requests while there is room for them.
    ///
    /// A request is started only when the KV pages it can need are free of the other
    /// requests: `ceil(min(prompt + max_new, max_pos) / PAGE)`, kept for it until it ends.
    /// That bound holds for speculation too, whose proposals never go past `max_new`. The
    /// pages of a prefix it reuses are not subtracted (a shared page is counted once for
    /// each request that holds it), which keeps the rule simple and a little cautious.
    /// Sequences kept for prefix reuse are not counted: they are evicted when needed.
    ///
    /// A request that waits stays at the front of the queue. One that cannot fit even in
    /// an empty arena would wait for ever, and fails at once instead.
    fn admit(&mut self, emit: &mut dyn FnMut(Event) -> bool) {
        let (mut total, max_pos) = self.target.arena();
        if let Some(draft) = &self.draft {
            total = total.min(draft.arena().0);
        }
        while self.streams.len() < self.cfg.max_streams
            && let Some(req) = self.waiting.front()
        {
            if req.prompt.is_empty() || req.max_new_tokens == 0 {
                let req = self.waiting.pop_front().unwrap();
                emit(Event::Done { id: req.id, finish: Finish::Length, generated: 0, start: StartInfo::default() });
                continue;
            }
            if self.cfg.response_cache > 0
                && let Some((out, mut finish)) = self.cache.get(req)
            {
                let req = self.waiting.pop_front().unwrap();
                self.spec.cache_hits += 1;
                let mut generated = 0;
                for &token in &out {
                    generated += 1;
                    if !emit(Event::Token { id: req.id, token }) {
                        finish = Finish::Cancelled;
                        break;
                    }
                }
                let start = StartInfo { reused: req.prompt.len(), computed: 0, cached: true };
                self.last_start = start;
                emit(Event::Done { id: req.id, finish, generated, start });
                continue;
            }
            let tokens = req.prompt.len() + req.max_new_tokens;
            let need = tokens.min(max_pos).div_ceil(crate::PAGE);
            if tokens > max_pos || need > total {
                let req = self.waiting.pop_front().unwrap();
                let message = if tokens > max_pos {
                    format!("the request needs {tokens} positions, more than the {max_pos} the model's KV covers")
                } else {
                    format!("the request needs {need} KV pages, more than the {total} the arena has")
                };
                emit(Event::Error { id: req.id, message });
                continue;
            }
            let reserved: usize = self.streams.iter().map(|s| s.reserved).sum();
            if reserved + need > total {
                break;
            }
            let req = self.waiting.pop_front().unwrap();
            match self.target.begin(&req.prompt) {
                Ok((ts, reused)) => {
                    let start = StartInfo { reused, computed: req.prompt.len() - reused, cached: false };
                    self.streams.push(Stream {
                        req,
                        ts,
                        ds: None,
                        phase: Phase::Prefill { done: reused },
                        generated: 0,
                        out: Vec::new(),
                        hint: HintCursor::default(),
                        tuning: StreamTuning::default(),
                        start,
                        finish: None,
                        error: None,
                        reserved: need,
                    });
                }
                Err(e) => {
                    emit(Event::Error { id: req.id, message: format!("{e:#}") });
                }
            }
        }
    }

    /// Processes prompt chunks of waiting streams (oldest first) in one pass, up to the
    /// row budget; streams whose prompt completes get their first token.
    async fn prefill_step(
        &mut self,
        prefilling: &[usize],
        busy: bool,
        emit: &mut dyn FnMut(Event) -> bool,
    ) -> Result<()> {
        let mut budget = if busy { self.cfg.busy_prefill_rows } else { self.target.chunk };
        let mut plan: Vec<(usize, usize, usize)> = Vec::new(); // (stream, from, to)
        for &i in prefilling {
            if budget == 0 {
                break;
            }
            let Phase::Prefill { done } = self.streams[i].phase else { unreachable!() };
            let n = (self.streams[i].req.prompt.len() - done).min(budget);
            budget -= n;
            plan.push((i, done, done + n));
        }
        let items: Vec<(SeqId, &[u32])> =
            plan.iter().map(|&(i, a, b)| (self.streams[i].ts, &self.streams[i].req.prompt[a..b])).collect();
        let forward = self.target.forward(&items, false);
        let result: Result<Vec<(usize, u32)>> = match forward {
            Ok(logits) => {
                // Only the streams whose prompt is complete need their last row's argmax.
                let complete: Vec<usize> =
                    (0..plan.len()).filter(|&j| plan[j].2 == self.streams[plan[j].0].req.prompt.len()).collect();
                if complete.is_empty() {
                    Ok(Vec::new())
                } else {
                    let toks = self.target.backend.argmax_rows_async(&logits).await.map_err(backend_failure)?;
                    Ok(complete.into_iter().map(|j| (j, toks[j])).collect())
                }
            }
            Err(e) => Err(e),
        };
        match result {
            Ok(firsts) => {
                for &(i, _, to) in &plan {
                    self.streams[i].phase = Phase::Prefill { done: to };
                }
                for (j, tok) in firsts {
                    self.on_token(plan[j].0, tok, emit);
                }
            }
            Err(e) => {
                let message = request_failure(e)?;
                for &(i, _, _) in &plan {
                    self.streams[i].error = Some(message.clone());
                }
            }
        }
        Ok(())
    }

    async fn decode_step(&mut self, picked: &[usize], emit: &mut dyn FnMut(Event) -> bool) -> Result<()> {
        let nexts: Vec<[u32; 1]> = picked
            .iter()
            .map(|&i| match self.streams[i].phase {
                Phase::Decode { next } => [next],
                _ => unreachable!(),
            })
            .collect();
        let items: Vec<(SeqId, &[u32])> =
            picked.iter().zip(&nexts).map(|(&i, n)| (self.streams[i].ts, n.as_slice())).collect();
        let t0 = Instant::now();
        let logits = self.target.forward(&items, false)?;
        let toks = self.target.backend.argmax_rows_async(&logits).await.map_err(backend_failure)?;
        drop(logits);
        if picked.len() == 1 {
            self.tuner.plain_step(t0.elapsed().as_secs_f64());
        }
        for (&i, &t) in picked.iter().zip(&toks) {
            self.on_token(i, t, emit);
        }
        Ok(())
    }

    /// One speculative round for decoding stream `i`: verifies hint tokens when the
    /// output matches the hint, else draft tokens; without either, decodes one token.
    async fn spec_round(&mut self, i: usize, emit: &mut dyn FnMut(Event) -> bool) -> Result<()> {
        let Phase::Decode { next } = self.streams[i].phase else { unreachable!() };
        let Some((props, source)) = self.propose(i, next).await? else { return self.decode_step(&[i], emit).await };
        let ts = self.streams[i].ts;
        let base = self.target.seq_len(ts);
        let g = self.verify(ts, next, &props).await?;
        let m = props.iter().zip(&g).take_while(|(p, t)| p == t).count();
        self.spec.rounds += 1;
        self.spec.proposed += props.len();
        self.spec.accepted += m;
        match source {
            Source::Hint { at } => {
                self.spec.hint_rounds += 1;
                self.streams[i].hint.advance(at, m);
            }
            Source::Draft { started } => {
                let secs = started.elapsed().as_secs_f64();
                self.tuner.draft_round(&mut self.streams[i].tuning, props.len(), m, secs);
            }
        }
        self.rewind(i, base + 1 + m);
        for &t in props[..m].iter().chain(std::iter::once(&g[m])) {
            self.on_token(i, t, emit);
            if self.streams[i].finish.is_some() {
                break;
            }
        }
        Ok(())
    }

    /// The tokens decoding stream `i` (whose next input is `next`) should verify, and
    /// where they come from; `None` when a single decode step is the better next move.
    async fn propose(&mut self, i: usize, next: u32) -> Result<Option<(Vec<u32>, Source)>> {
        let s = &mut self.streams[i];
        let left = s.req.max_new_tokens - s.generated;
        let end = self.end_token.filter(|e| s.req.stop.contains(e));
        if let Some((at, props)) = s.hint.propose(&s.req.hint, &s.out, self.cfg.hint_k.min(left), end) {
            return Ok(Some((props, Source::Hint { at })));
        }
        let draft_lag =
            s.ds.map_or(usize::MAX, |d| self.target.seq_len(s.ts) - self.draft.as_ref().map_or(0, |m| m.seq_len(d)));
        if !s.req.hint.is_empty() && draft_lag > 16 && s.hint.wait() {
            return Ok(None);
        }
        if self.draft.is_none() {
            return Ok(None);
        }
        match self.tuner.choose(&mut s.tuning, self.cfg.draft_k.min(left)) {
            0 => {
                self.spec.plain_steps += 1;
                Ok(None)
            }
            k => {
                let started = Instant::now();
                Ok(Some((self.draft_tokens(i, next, k).await?, Source::Draft { started })))
            }
        }
    }

    /// Runs the target over `next` and `props` on sequence `ts`: `g[j]` is its choice
    /// after consuming `next` and `props[..j]`.
    async fn verify(&mut self, ts: SeqId, next: u32, props: &[u32]) -> Result<Vec<u32>> {
        let mut ver = Vec::with_capacity(props.len() + 1);
        ver.push(next);
        ver.extend_from_slice(props);
        let t0 = Instant::now();
        let logits = self.target.forward(&[(ts, &ver)], true)?;
        let g = self.target.backend.argmax_rows_async(&logits).await.map_err(backend_failure)?;
        drop(logits);
        self.tuner.verify_pass(ver.len(), t0.elapsed().as_secs_f64());
        Ok(g)
    }

    /// Cuts stream `i`'s sequences back to `len` tokens (the draft's only if it is longer).
    fn rewind(&mut self, i: usize, len: usize) {
        self.target.truncate(self.streams[i].ts, len);
        if let (Some(ds), Some(draft)) = (self.streams[i].ds, self.draft.as_mut())
            && draft.seq_len(ds) > len
        {
            draft.truncate(ds, len);
        }
    }

    /// Runs the draft model for stream `i` (whose next input is `next`) and returns up
    /// to `k` proposals, ending early at a stop token.
    async fn draft_tokens(&mut self, i: usize, next: u32, k: usize) -> Result<Vec<u32>> {
        let ts = self.streams[i].ts;
        let draft = self.draft.as_mut().unwrap();
        let ds = match self.streams[i].ds {
            Some(d) => d,
            None => {
                let (d, _) = draft.begin(&self.streams[i].req.prompt)?;
                self.streams[i].ds = Some(d);
                d
            }
        };
        // The draft's tokens are always a prefix of the target's; feed it the rest plus
        // `next`, which gives its first proposal.
        let mut feed = self.target.seq_tokens(ts)[draft.seq_len(ds)..].to_vec();
        feed.push(next);
        let stop = &self.streams[i].req.stop;
        let mut props = Vec::with_capacity(k);
        let caught_up = feed.len() == 1;
        let t0 = Instant::now();
        let mut dl = draft.prefill(ds, &feed)?;
        loop {
            let d = *draft.backend.argmax_rows_async(&dl).await.map_err(backend_failure)?.last().unwrap();
            props.push(d);
            if props.len() == k || stop.contains(&d) {
                break;
            }
            dl = draft.forward(&[(ds, &[d])], false)?;
        }
        // Steps after a catch-up include its prefill; time only the plain ones.
        self.tuner.draft_run(caught_up, t0.elapsed().as_secs_f64() / props.len() as f64);
        Ok(props)
    }

    /// A token computed for stream `i`: emit it, or end the stream.
    fn on_token(&mut self, i: usize, tok: u32, emit: &mut dyn FnMut(Event) -> bool) {
        let s = &mut self.streams[i];
        if s.req.stop.contains(&tok) {
            s.finish = Some(Finish::Stop);
            self.end_token = Some(tok);
            return;
        }
        s.generated += 1;
        s.out.push(tok);
        if !emit(Event::Token { id: s.req.id, token: tok }) {
            s.finish = Some(Finish::Cancelled);
        } else if s.generated >= s.req.max_new_tokens {
            s.finish = Some(Finish::Length);
        } else {
            s.phase = Phase::Decode { next: tok };
        }
    }

    /// Removes finished and failed streams, keeping their KV for prefix reuse.
    fn sweep(&mut self, emit: &mut dyn FnMut(Event) -> bool) {
        let mut i = 0;
        while i < self.streams.len() {
            let s = &self.streams[i];
            if s.finish.is_none() && s.error.is_none() {
                i += 1;
                continue;
            }
            let s = self.streams.swap_remove(i);
            match &s.error {
                Some(msg) => {
                    self.target.free_seq(s.ts);
                    if let (Some(d), Some(dm)) = (s.ds, self.draft.as_mut()) {
                        dm.free_seq(d);
                    }
                    emit(Event::Error { id: s.req.id, message: msg.clone() });
                }
                None => {
                    self.target.finish(s.ts);
                    if let (Some(d), Some(dm)) = (s.ds, self.draft.as_mut()) {
                        dm.finish(d);
                    }
                    self.last_start = s.start;
                    let finish = s.finish.unwrap();
                    emit(Event::Done { id: s.req.id, finish, generated: s.generated, start: s.start });
                    if self.cfg.response_cache > 0 && finish != Finish::Cancelled {
                        self.cache.put(self.cfg.response_cache, s.req.prompt, s.req.stop, s.out, finish);
                    }
                }
            }
        }
    }
}

/// A boxed future that need not be `Send` (WebGPU's are not).
pub type LocalBoxFuture<'a, T> = std::pin::Pin<Box<dyn Future<Output = T> + 'a>>;

/// Object-safe view of an engine, so callers can pick the backend at run time.
///
/// It is not `Send`: a WebGPU engine cannot move between threads, and its steps are
/// awaited. An engine that is handed to another thread (the server's inference thread)
/// is a `Box<dyn Generator + Send>`, which the native backends allow.
pub trait Generator {
    fn submit(&mut self, req: Request);
    /// See [`Engine::step_async`].
    fn step_async<'a>(&'a mut self, emit: &'a mut dyn FnMut(Event) -> bool) -> LocalBoxFuture<'a, Result<bool>>;
    fn fresh_id(&mut self) -> ReqId;
    /// See [`Engine::cancel`]. An engine that cannot cancel says `false`: the request
    /// then ends when its next token is refused, as before.
    fn cancel(&mut self, _id: ReqId) -> bool {
        false
    }
    fn waiting(&self) -> usize;
    fn active(&self) -> usize;
    /// See [`Engine::abort_all`].
    fn abort_all(&mut self);
    fn backend_name(&self) -> String;
    fn weight_bytes(&self) -> usize;
    /// Speculative decoding statistics, if any rounds ran.
    fn stats(&self) -> Option<String>;
    /// Prefix reuse of the last finished request.
    fn last_start(&self) -> StartInfo;
    /// How likely the target model finds `forced` as the output for `prompt` (see
    /// [`Model::score`](crate::Model::score)). Runs at once, between steps.
    fn score(&mut self, _prompt: &[u32], _forced: &[u32]) -> Result<Vec<crate::TokenScore>> {
        anyhow::bail!("this engine does not score text")
    }
}

/// A [`Generator`] whose steps never wait, which can therefore also be stepped without
/// an async context: the engines of the native backends (`SyncReadback`). A WebGPU engine
/// is not one: its step waits for the device, so it can only be awaited.
pub trait SyncGenerator: Generator {
    /// [`step_async`](Generator::step_async), polled once.
    fn step(&mut self, emit: &mut dyn FnMut(Event) -> bool) -> Result<bool> {
        poll_once(self.step_async(emit))
    }

    /// Runs one request to completion on its own. `on_token` returns `false` to stop.
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        stop: &[u32],
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let id = self.fresh_id();
        self.submit(Request::new(id, prompt.to_vec(), max_new_tokens, stop.to_vec()));
        let mut out = Vec::new();
        let mut done = false;
        let mut error = None;
        while !done {
            let worked = self.step(&mut |ev| match ev {
                Event::Token { id: r, token } if r == id => {
                    out.push(token);
                    on_token(token)
                }
                Event::Done { id: r, .. } if r == id => {
                    done = true;
                    true
                }
                Event::Error { id: r, message } if r == id => {
                    error = Some(message);
                    done = true;
                    true
                }
                _ => true,
            })?;
            anyhow::ensure!(worked || done, "engine went idle before the request finished");
        }
        match error {
            Some(e) => Err(anyhow::anyhow!(e)),
            None => Ok(out),
        }
    }
}

impl<B: SyncReadback> SyncGenerator for Engine<B> {}

impl<B: SyncReadback> Engine<B> {
    /// [`Engine::step_async`] for a backend that reads back without waiting, where the
    /// step never has to wait for anything.
    pub fn step(&mut self, emit: &mut dyn FnMut(Event) -> bool) -> Result<bool> {
        poll_once(self.step_async(emit))
    }
}

/// The output of a future that is ready at its first poll, as the steps of an engine on
/// a native backend are. Panics if it is not: that one waits for something (WebGPU) and
/// has to be awaited.
pub fn poll_once<F: Future>(f: F) -> F::Output {
    let mut f = std::pin::pin!(f);
    match f.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(r) => r,
        Poll::Pending => unreachable!("the future waits: drive it from an async context"),
    }
}

/// A failure of the backend itself (a read-back failed), as opposed to one of a request:
/// it travels up through `anyhow::Error` to [`Engine::step_async`], which returns it.
#[derive(Debug)]
struct BackendFailure(anyhow::Error);

impl std::fmt::Display for BackendFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#}", self.0)
    }
}

impl std::error::Error for BackendFailure {}

fn backend_failure(e: anyhow::Error) -> anyhow::Error {
    anyhow::Error::new(BackendFailure(e))
}

/// The message of a failure that belongs to the requests of a pass; a failure of the
/// backend is returned as `Err`, the error of the step.
fn request_failure(e: anyhow::Error) -> Result<String> {
    match e.downcast::<BackendFailure>() {
        Ok(BackendFailure(e)) => Err(e),
        Err(e) => Ok(format!("{e:#}")),
    }
}

impl<B: AsyncReadback> Generator for Engine<B> {
    fn submit(&mut self, req: Request) {
        Engine::submit(self, req)
    }
    fn step_async<'a>(&'a mut self, emit: &'a mut dyn FnMut(Event) -> bool) -> LocalBoxFuture<'a, Result<bool>> {
        Box::pin(Engine::step_async(self, emit))
    }
    fn fresh_id(&mut self) -> ReqId {
        Engine::fresh_id(self)
    }
    fn cancel(&mut self, id: ReqId) -> bool {
        Engine::cancel(self, id)
    }
    fn waiting(&self) -> usize {
        Engine::waiting(self)
    }
    fn active(&self) -> usize {
        Engine::active(self)
    }
    fn abort_all(&mut self) {
        Engine::abort_all(self)
    }
    fn backend_name(&self) -> String {
        match &self.draft {
            Some(_) => format!("{} + draft (k={})", self.target.backend.name(), self.cfg.draft_k),
            None => self.target.backend.name(),
        }
    }
    fn weight_bytes(&self) -> usize {
        self.target.backend.weight_bytes() + self.draft.as_ref().map_or(0, |d| d.backend.weight_bytes())
    }
    fn stats(&self) -> Option<String> {
        let s = &self.spec;
        let mut parts = Vec::new();
        if s.rounds > 0 {
            parts.push(format!(
                "{} rounds ({} from hints, {} plain steps), {:.0}% of drafts accepted, {:.2} tokens per target pass",
                s.rounds,
                s.hint_rounds,
                s.plain_steps,
                100.0 * s.accepted as f64 / s.proposed.max(1) as f64,
                (s.accepted + s.rounds) as f64 / s.rounds as f64
            ));
        }
        if s.cache_hits > 0 {
            parts.push(format!("{} answered from the response cache", s.cache_hits));
        }
        (!parts.is_empty()).then(|| parts.join("; "))
    }
    fn last_start(&self) -> StartInfo {
        self.last_start
    }
    fn score(&mut self, prompt: &[u32], forced: &[u32]) -> Result<Vec<crate::TokenScore>> {
        match B::scorer() {
            Some(score) => score(&mut self.target, prompt, forced),
            None => anyhow::bail!("this backend cannot score text"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs the engine until it is idle; returns each request's `(id, finish or error)`.
    fn run_all(eng: &mut Engine<crate::testing::NullBackend>) -> Vec<(ReqId, Result<Finish, String>)> {
        let mut ended = Vec::new();
        while eng
            .step(&mut |ev| {
                match ev {
                    Event::Done { id, finish, .. } => ended.push((id, Ok(finish))),
                    Event::Error { id, message } => ended.push((id, Err(message))),
                    Event::Token { .. } => {}
                }
                true
            })
            .unwrap()
        {}
        ended
    }

    fn request(id: ReqId, prompt_len: usize, max_new: usize) -> Request {
        // Distinct prompts, so that no prefix is shared and the KV needs are plain.
        Request::new(id, (0..prompt_len as u32).map(|t| t + id as u32 * 1000).collect(), max_new, vec![])
    }

    #[test]
    fn a_waiting_request_is_cancelled_before_it_starts() {
        let mut eng = Engine::new(crate::testing::model(2), None, EngineConfig::default()).unwrap();
        eng.submit(request(1, 130, 3));
        eng.submit(request(2, 130, 3));
        eng.step(&mut |_| true).unwrap();
        assert_eq!((eng.active(), eng.waiting()), (1, 1));
        assert!(eng.cancel(2), "it waits");
        assert!(!eng.cancel(2), "and only once");
        assert!(!eng.cancel(99), "an unknown request");
        let ended = run_all(&mut eng);
        assert!(ended.contains(&(2, Ok(Finish::Cancelled))), "{ended:?}");
        assert!(ended.contains(&(1, Ok(Finish::Length))), "the other runs on: {ended:?}");
        assert!(eng.is_idle());
    }

    #[test]
    fn a_running_request_is_cancelled_and_its_pages_serve_the_next() {
        let mut eng = Engine::new(crate::testing::model(2), None, EngineConfig::default()).unwrap();
        eng.submit(request(1, 130, 120)); // would run for 120 more steps
        eng.submit(request(2, 130, 3)); // waits for 1's pages
        eng.step(&mut |_| true).unwrap();
        assert!(eng.cancel(1));
        let mut tokens_of_1_after = 0;
        let mut ended = Vec::new();
        while eng
            .step(&mut |ev| {
                match ev {
                    Event::Token { id: 1, .. } => tokens_of_1_after += 1,
                    Event::Done { id, finish, .. } => ended.push((id, finish)),
                    _ => {}
                }
                true
            })
            .unwrap()
        {}
        assert_eq!(tokens_of_1_after, 0, "a cancelled request is not given another pass");
        assert_eq!(ended, [(1, Finish::Cancelled), (2, Finish::Length)]);
        assert!(eng.is_idle());
    }

    #[test]
    fn a_request_waits_for_the_pages_of_the_others() {
        // 2 pages of 128 tokens; each request below needs both.
        let mut eng = Engine::new(crate::testing::model(2), None, EngineConfig::default()).unwrap();
        eng.submit(request(1, 130, 3));
        eng.submit(request(2, 130, 3));
        eng.step(&mut |_| true).unwrap();
        assert_eq!((eng.active(), eng.waiting()), (1, 1), "the second one waits instead of failing");
        let ended = run_all(&mut eng);
        assert_eq!(ended, [(1, Ok(Finish::Length)), (2, Ok(Finish::Length))]);
        assert_eq!(eng.active() + eng.waiting(), 0);
    }

    #[test]
    fn requests_that_fit_together_run_together() {
        let mut eng = Engine::new(crate::testing::model(2), None, EngineConfig::default()).unwrap();
        eng.submit(request(1, 100, 20));
        eng.submit(request(2, 100, 20));
        eng.step(&mut |_| true).unwrap();
        assert_eq!((eng.active(), eng.waiting()), (2, 0), "one page each");
    }

    #[test]
    fn a_request_that_cannot_fit_alone_fails_at_once_and_blocks_nobody() {
        let mut eng = Engine::new(crate::testing::model(2), None, EngineConfig::default()).unwrap();
        eng.submit(request(1, 250, 10)); // 260 tokens: 3 pages of the 2
        eng.submit(request(2, 10, 3));
        let ended = run_all(&mut eng);
        assert_eq!(ended.len(), 2);
        let (id, r) = &ended[0];
        assert_eq!(*id, 1);
        assert!(r.as_ref().unwrap_err().contains("KV pages"), "{r:?}");
        assert_eq!(ended[1], (2, Ok(Finish::Length)));
    }

    #[test]
    fn a_request_past_the_positions_of_the_model_fails_at_once() {
        let mut eng = Engine::new(crate::testing::model_with_max_pos(4, 200), None, EngineConfig::default()).unwrap();
        eng.submit(request(1, 150, 100));
        let ended = run_all(&mut eng);
        assert!(ended[0].1.as_ref().unwrap_err().contains("positions"), "{ended:?}");
    }

    #[test]
    fn a_request_may_take_the_whole_arena() {
        // The page rule needs no spare page: 256 tokens fit in 2 pages of 128.
        let mut eng = Engine::new(crate::testing::model(2), None, EngineConfig::default()).unwrap();
        eng.submit(request(1, 250, 6));
        assert_eq!(run_all(&mut eng), [(1, Ok(Finish::Length))]);
    }

    #[test]
    fn zero_row_budgets_are_rejected() {
        let cfg = EngineConfig { busy_prefill_rows: 0, ..Default::default() };
        assert!(Engine::new(crate::testing::model(2), None, cfg).is_err());
        assert!(Engine::new(crate::testing::model(2), None, EngineConfig::default()).is_ok());
    }
}
