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
use std::time::Instant;

use anyhow::Result;

use self::cache::ResponseCache;
use self::hint::HintCursor;
use self::tuning::{DraftTuner, StreamTuning};
use crate::{Backend, Model, SeqId, StartInfo};

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
}

impl<B: Backend> Engine<B> {
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
        })
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
        self.waiting.is_empty() && self.streams.is_empty()
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
    pub fn step(&mut self, emit: &mut dyn FnMut(Event) -> bool) -> Result<bool> {
        self.admit(emit);
        if self.streams.is_empty() {
            return Ok(false);
        }
        let decoding: Vec<usize> =
            (0..self.streams.len()).filter(|&i| matches!(self.streams[i].phase, Phase::Decode { .. })).collect();
        let prefilling: Vec<usize> =
            (0..self.streams.len()).filter(|&i| matches!(self.streams[i].phase, Phase::Prefill { .. })).collect();
        match prefilling.first() {
            Some(_) if decoding.is_empty() || self.prefill_turn => {
                self.prefill_turn = false;
                let busy = !decoding.is_empty();
                self.prefill_step(&prefilling, busy, emit);
            }
            _ => {
                self.prefill_turn = true;
                let speculate = |s: &Stream| self.draft.is_some() || !s.req.hint.is_empty();
                if decoding.len() <= self.cfg.spec_max_streams && decoding.iter().all(|&i| speculate(&self.streams[i]))
                {
                    for &i in &decoding {
                        if let Err(e) = self.spec_round(i, emit) {
                            self.streams[i].error = Some(format!("{e:#}"));
                        }
                    }
                } else {
                    let n = decoding.len().min(self.cfg.decode_rows);
                    let start = self.rr % decoding.len();
                    self.rr = self.rr.wrapping_add(n);
                    let picked: Vec<usize> = (0..n).map(|j| decoding[(start + j) % decoding.len()]).collect();
                    if let Err(e) = self.decode_step(&picked, emit) {
                        for &i in &picked {
                            self.streams[i].error = Some(format!("{e:#}"));
                        }
                    }
                }
            }
        }
        self.sweep(emit);
        Ok(true)
    }

    fn admit(&mut self, emit: &mut dyn FnMut(Event) -> bool) {
        while self.streams.len() < self.cfg.max_streams
            && let Some(req) = self.waiting.pop_front()
        {
            if req.prompt.is_empty() || req.max_new_tokens == 0 {
                emit(Event::Done { id: req.id, finish: Finish::Length, generated: 0, start: StartInfo::default() });
                continue;
            }
            if self.cfg.response_cache > 0
                && let Some((out, mut finish)) = self.cache.get(&req)
            {
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
    fn prefill_step(&mut self, prefilling: &[usize], busy: bool, emit: &mut dyn FnMut(Event) -> bool) {
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
        let result = self.target.forward(&items, false).map(|logits| {
            // Only the streams whose prompt is complete need their last row's argmax.
            let complete: Vec<usize> =
                (0..plan.len()).filter(|&j| plan[j].2 == self.streams[plan[j].0].req.prompt.len()).collect();
            if complete.is_empty() {
                return Vec::new();
            }
            let toks = self.target.backend.argmax_rows(&logits);
            complete.into_iter().map(|j| (j, toks[j])).collect()
        });
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
                for &(i, _, _) in &plan {
                    self.streams[i].error = Some(format!("{e:#}"));
                }
            }
        }
    }

    fn decode_step(&mut self, picked: &[usize], emit: &mut dyn FnMut(Event) -> bool) -> Result<()> {
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
        let toks = self.target.backend.argmax_rows(&logits);
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
    fn spec_round(&mut self, i: usize, emit: &mut dyn FnMut(Event) -> bool) -> Result<()> {
        let Phase::Decode { next } = self.streams[i].phase else { unreachable!() };
        let Some((props, source)) = self.propose(i, next)? else { return self.decode_step(&[i], emit) };
        let ts = self.streams[i].ts;
        let base = self.target.seq_len(ts);
        let g = self.verify(ts, next, &props)?;
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
    fn propose(&mut self, i: usize, next: u32) -> Result<Option<(Vec<u32>, Source)>> {
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
                Ok(Some((self.draft_tokens(i, next, k)?, Source::Draft { started })))
            }
        }
    }

    /// Runs the target over `next` and `props` on sequence `ts`: `g[j]` is its choice
    /// after consuming `next` and `props[..j]`.
    fn verify(&mut self, ts: SeqId, next: u32, props: &[u32]) -> Result<Vec<u32>> {
        let mut ver = Vec::with_capacity(props.len() + 1);
        ver.push(next);
        ver.extend_from_slice(props);
        let t0 = Instant::now();
        let logits = self.target.forward(&[(ts, &ver)], true)?;
        let g = self.target.backend.argmax_rows(&logits);
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
    fn draft_tokens(&mut self, i: usize, next: u32, k: usize) -> Result<Vec<u32>> {
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
            let d = draft.backend.argmax(&dl);
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

/// Object-safe view of an engine, so callers can pick the backend at run time.
pub trait Generator: Send {
    fn submit(&mut self, req: Request);
    /// See [`Engine::step`].
    fn step(&mut self, emit: &mut dyn FnMut(Event) -> bool) -> Result<bool>;
    fn fresh_id(&mut self) -> ReqId;
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

impl<B: Backend + Send> Generator for Engine<B>
where
    B::Weight: Send,
    B::Kv: Send,
{
    fn submit(&mut self, req: Request) {
        Engine::submit(self, req)
    }
    fn step(&mut self, emit: &mut dyn FnMut(Event) -> bool) -> Result<bool> {
        Engine::step(self, emit)
    }
    fn fresh_id(&mut self) -> ReqId {
        Engine::fresh_id(self)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_row_budgets_are_rejected() {
        let cfg = EngineConfig { busy_prefill_rows: 0, ..Default::default() };
        assert!(Engine::new(crate::testing::model(2), None, cfg).is_err());
        assert!(Engine::new(crate::testing::model(2), None, EngineConfig::default()).is_ok());
    }
}
