//! The scheduler: runs many requests over one model (and an optional draft model for
//! speculative decoding), one forward pass per `step`.
//!
//! - Prompts are processed in chunks; while other requests are decoding, a prompt chunk
//!   is at most `busy_prefill_rows` rows and alternates with decode passes.
//! - Decoding requests advance together: one pass takes the next token of up to
//!   `decode_rows` of them (round robin), reading the weights once for all.
//! - With at most `spec_max_streams` decoding requests and a draft model, a request
//!   advances by a speculative round instead (the draft proposes `draft_k` tokens, the
//!   target checks them in one pass). The draft catches up from the target's tokens,
//!   so a request can switch between speculative and batched decoding at any time.
//!
//! Decoding is greedy, so every request produces the target model's greedy output
//! whatever else is running (up to the summation-order differences of the kernels).

use std::collections::VecDeque;

use anyhow::Result;

use crate::{Backend, Model, SeqId, StartInfo};

pub type ReqId = u64;

#[derive(Debug, Clone)]
pub struct Request {
    pub id: ReqId,
    pub prompt: Vec<u32>,
    pub max_new_tokens: usize,
    /// Tokens that end the generation (not emitted).
    pub stop: Vec<u32>,
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
    /// Most requests advanced by one decode pass (8 rows is the cheapest per row on a
    /// Radeon 780M: 46 ms for 8 rows, 114 ms for 16).
    pub decode_rows: usize,
    /// Prompt rows per pass while other requests are decoding.
    pub busy_prefill_rows: usize,
    /// Draft tokens per speculative round.
    pub draft_k: usize,
    /// Speculative decoding is used while at most this many requests are decoding.
    pub spec_max_streams: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig { max_streams: 16, decode_rows: 8, busy_prefill_rows: 64, draft_k: 5, spec_max_streams: 1 }
    }
}

#[derive(Debug, Default, Clone)]
pub struct SpecStats {
    /// Target verification passes.
    pub rounds: usize,
    /// Draft tokens proposed / accepted by the target.
    pub proposed: usize,
    pub accepted: usize,
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
    start: StartInfo,
    finish: Option<Finish>,
    error: Option<String>,
}

pub struct Engine<B: Backend> {
    pub target: Model<B>,
    pub draft: Option<Model<B>>,
    pub cfg: EngineConfig,
    pub spec: SpecStats,
    waiting: VecDeque<Request>,
    streams: Vec<Stream>,
    /// Round-robin start for decode passes.
    rr: usize,
    /// Alternation between prompt chunks and decode passes.
    prefill_turn: bool,
    next_id: ReqId,
    last_start: StartInfo,
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
        anyhow::ensure!(cfg.decode_rows >= 1 && cfg.max_streams >= 1);
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
        })
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
                if decoding.len() <= self.cfg.spec_max_streams && self.draft.is_some() {
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
            match self.target.begin(&req.prompt) {
                Ok((ts, reused)) => {
                    let start = StartInfo { reused, computed: req.prompt.len() - reused };
                    self.streams.push(Stream {
                        req,
                        ts,
                        ds: None,
                        phase: Phase::Prefill { done: reused },
                        generated: 0,
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
        let result = self.target.forward(&items, false).and_then(|logits| {
            // Only the streams whose prompt is complete need their last row's argmax.
            let complete: Vec<usize> =
                (0..plan.len()).filter(|&j| plan[j].2 == self.streams[plan[j].0].req.prompt.len()).collect();
            if complete.is_empty() {
                return Ok(Vec::new());
            }
            let toks = self.target.backend.argmax_rows(&logits);
            Ok(complete.into_iter().map(|j| (j, toks[j])).collect())
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
        let logits = self.target.forward(&items, false)?;
        let toks = self.target.backend.argmax_rows(&logits);
        drop(logits);
        for (&i, &t) in picked.iter().zip(&toks) {
            self.on_token(i, t, emit);
        }
        Ok(())
    }

    /// One speculative round for decoding stream `i`.
    fn spec_round(&mut self, i: usize, emit: &mut dyn FnMut(Event) -> bool) -> Result<()> {
        let Phase::Decode { next } = self.streams[i].phase else { unreachable!() };
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
        let base = self.target.seq_len(ts);
        let mut feed = self.target.seq_tokens(ts)[draft.seq_len(ds)..].to_vec();
        feed.push(next);
        let left = self.streams[i].req.max_new_tokens - self.streams[i].generated;
        let k = self.cfg.draft_k.min(left);
        let stop = &self.streams[i].req.stop;
        let mut props = Vec::with_capacity(k);
        let mut dl = draft.prefill(ds, &feed)?;
        loop {
            let d = draft.backend.argmax(&dl);
            props.push(d);
            if props.len() == k || stop.contains(&d) {
                break;
            }
            dl = draft.forward(&[(ds, &[d])], false)?;
        }
        drop(dl);

        // Verify: g[j] is the target's choice after consuming ver[..=j].
        let mut ver = Vec::with_capacity(props.len() + 1);
        ver.push(next);
        ver.extend_from_slice(&props);
        let logits = self.target.forward(&[(ts, &ver)], true)?;
        let g = self.target.backend.argmax_rows(&logits);
        drop(logits);
        let m = props.iter().zip(&g).take_while(|(p, t)| p == t).count();
        self.spec.rounds += 1;
        self.spec.proposed += props.len();
        self.spec.accepted += m;

        self.target.truncate(ts, base + 1 + m);
        let draft = self.draft.as_mut().unwrap();
        if draft.seq_len(ds) > base + 1 + m {
            draft.truncate(ds, base + 1 + m);
        }
        for &t in props[..m].iter().chain(std::iter::once(&g[m])) {
            self.on_token(i, t, emit);
            if self.streams[i].finish.is_some() {
                break;
            }
        }
        Ok(())
    }

    /// A token computed for stream `i`: emit it, or end the stream.
    fn on_token(&mut self, i: usize, tok: u32, emit: &mut dyn FnMut(Event) -> bool) {
        let s = &mut self.streams[i];
        if s.req.stop.contains(&tok) {
            s.finish = Some(Finish::Stop);
            return;
        }
        s.generated += 1;
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
                    emit(Event::Done { id: s.req.id, finish: s.finish.unwrap(), generated: s.generated, start: s.start });
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
        self.submit(Request { id, prompt: prompt.to_vec(), max_new_tokens, stop: stop.to_vec() });
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
        (s.rounds > 0).then(|| {
            format!(
                "{} rounds, {:.0}% of drafts accepted, {:.2} tokens per target pass",
                s.rounds,
                100.0 * s.accepted as f64 / s.proposed.max(1) as f64,
                (s.accepted + s.rounds) as f64 / s.rounds as f64
            )
        })
    }
    fn last_start(&self) -> StartInfo {
        self.last_start
    }
}
