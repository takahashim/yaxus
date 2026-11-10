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

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};

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
    /// Most requests advanced by one decode pass (8 rows is the cheapest per row on a
    /// Radeon 780M: 46 ms for 8 rows, 114 ms for 16).
    pub decode_rows: usize,
    /// Prompt rows per pass while other requests are decoding.
    pub busy_prefill_rows: usize,
    /// Most draft tokens per speculative round (the engine picks the length per round
    /// from the measured acceptance and pass times, see `Engine::draft_len`).
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
    /// Where the hint is expected to continue the output.
    hint_pos: usize,
    /// Rounds in a row in which the output did not match the hint.
    hint_misses: usize,
    /// Draft acceptance by depth (this request's rounds; see `Accept`).
    accept: Accept,
    /// Single decode steps since the last draft round.
    plain_run: usize,
    start: StartInfo,
    finish: Option<Finish>,
    error: Option<String>,
}

struct Cached {
    prompt: Vec<u32>,
    stop: Vec<u32>,
    out: Vec<u32>,
    finish: Finish,
}

/// Outputs of finished requests by prompt (least recently used first in `order`).
#[derive(Default)]
struct ResponseCache {
    map: HashMap<u64, Cached>,
    order: VecDeque<u64>,
}

fn cache_key(prompt: &[u32], stop: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    prompt.hash(&mut h);
    stop.hash(&mut h);
    h.finish()
}

impl ResponseCache {
    /// The cached output for `req`, if it is the full answer to it.
    fn get(&mut self, req: &Request) -> Option<(Vec<u32>, Finish)> {
        let key = cache_key(&req.prompt, &req.stop);
        let c = self.map.get(&key).filter(|c| c.prompt == req.prompt && c.stop == req.stop)?;
        let hit = if c.out.len() >= req.max_new_tokens {
            Some((c.out[..req.max_new_tokens].to_vec(), Finish::Length))
        } else if c.finish == Finish::Stop {
            Some((c.out.clone(), Finish::Stop))
        } else {
            None // cut by a smaller limit
        };
        if hit.is_some() {
            self.touch(key);
        }
        hit
    }

    fn put(&mut self, cap: usize, prompt: Vec<u32>, stop: Vec<u32>, out: Vec<u32>, finish: Finish) {
        let key = cache_key(&prompt, &stop);
        self.map.insert(key, Cached { prompt, stop, out, finish });
        self.touch(key);
        while self.map.len() > cap
            && let Some(old) = self.order.pop_front()
        {
            self.map.remove(&old);
        }
    }

    fn touch(&mut self, key: u64) {
        self.order.retain(|&k| k != key);
        self.order.push_back(key);
    }
}

/// Exponential moving average of a duration in seconds.
#[derive(Debug, Clone, Copy, Default)]
struct Ema {
    v: f64,
    n: u32,
}

impl Ema {
    fn add(&mut self, x: f64) {
        self.v = if self.n == 0 { x } else { 0.8 * self.v + 0.2 * x };
        self.n += 1;
    }
    fn get(&self) -> Option<f64> {
        (self.n > 0).then_some(self.v)
    }
}

/// Measured pass times that choose the draft length (see `Engine::draft_len`).
#[derive(Debug, Default)]
struct Timing {
    /// One target decode step (one row).
    plain: Ema,
    /// One draft step.
    draft: Ema,
    /// A target verification pass, by rows.
    verify: [Ema; 18],
    /// What a whole draft round takes beyond its draft steps and verification (host
    /// work, the draft's first step and truncation).
    round_extra: Ema,
}

impl Timing {
    /// Time of a target pass of `rows` rows: measured, else extrapolated from the
    /// nearest measurement (a few % per extra row on a bandwidth-bound GPU).
    fn target(&self, rows: usize) -> Option<f64> {
        let measured = |r: usize| if r == 1 { self.plain.get() } else { self.verify.get(r).and_then(Ema::get) };
        if let Some(t) = measured(rows) {
            return Some(t);
        }
        (1..self.verify.len())
            .filter_map(|r| measured(r).map(|t| (r, t)))
            .min_by_key(|&(r, _)| r.abs_diff(rows))
            .map(|(r, t)| t * (1.0 + 0.04 * rows as f64) / (1.0 + 0.04 * r as f64))
    }
}

/// Draft length until the pass times are known, and unless another one is expected to
/// be `SWITCH_GAIN` times faster.
const DEFAULT_DRAFT_K: usize = 5;
const SWITCH_GAIN: f64 = 1.05;
/// Rounds (decayed) observed before the length adapts.
const WARMUP_ROUNDS: f64 = 30.0;

/// Decayed counts of draft rounds: `reach1`/`hit1` for the first draft token (drafted /
/// accepted), `reach`/`hit` for the later ones (drafted after all earlier ones of the
/// round were accepted / accepted too). Acceptance is bursty, so the first token gets its
/// own rate and the later ones share one (pooled over depths: depth-wise counts went
/// stale at depths the chosen length stopped reaching).
#[derive(Debug, Clone, Copy, Default)]
struct Accept {
    reach1: f64,
    hit1: f64,
    reach: f64,
    hit: f64,
}

impl Accept {
    /// Records a round of `k` drafts of which the first `m` were accepted.
    fn add(&mut self, k: usize, m: usize, decay: f64) {
        for v in [&mut self.reach1, &mut self.hit1, &mut self.reach, &mut self.hit] {
            *v *= decay;
        }
        if k == 0 {
            return;
        }
        self.reach1 += 1.0;
        self.hit1 += (m >= 1) as u8 as f64;
        // Depths 2..=k reached while the run went on: min(k, m + 1) - 1 of them.
        let later = k.min(m + 1).saturating_sub(1);
        self.reach += later as f64;
        self.hit += m.saturating_sub(1).min(later) as f64;
    }

    /// (first, later) acceptance with `weight` pseudo-rounds of `prior`.
    fn rates(&self, prior: (f64, f64), weight: f64) -> (f64, f64) {
        let r = |hit: f64, reach: f64, p: f64| ((hit + weight * p) / (reach + weight)).min(0.995);
        (r(self.hit1, self.reach1, prior.0), r(self.hit, self.reach, prior.1))
    }
}

/// Single steps after which a stream that stopped drafting tries a short round again.
const PROBE_AFTER: usize = 16;

/// Single decode steps tried after the output left its hint before the draft model
/// takes over.
const HINT_PATIENCE: usize = 4;

/// Hint tokens to verify after `out`: the ones following the latest `n` output tokens
/// (n = 3, 2, or 1 near `pos`) in the hint, at the occurrence closest to `pos`.
/// Returns the hint index of the first proposed token.
fn hint_match(hint: &[u32], out: &[u32], pos: usize) -> Option<usize> {
    for n in (1..=out.len().min(3)).rev() {
        let tail = &out[out.len() - n..];
        let best = (n..hint.len())
            .filter(|&p| &hint[p - n..p] == tail)
            .min_by_key(|&p| p.abs_diff(pos));
        match best {
            // A single token matches too easily far from where the hint is aligned.
            Some(p) if n > 1 || p.abs_diff(pos) <= 4 => return Some(p),
            _ => {}
        }
    }
    None
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
    cache: ResponseCache,
    timing: Timing,
    /// Draft acceptance by depth over all requests (the prior of each request's own).
    accept: Accept,
    /// The last draft run started in sync with the target (no catch-up prefill).
    draft_in_sync: bool,
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
            cache: ResponseCache::default(),
            timing: Timing::default(),
            accept: Accept::default(),
            draft_in_sync: false,
            end_token: None,
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
                let speculate = |s: &Stream| self.draft.is_some() || !s.req.hint.is_empty();
                if decoding.len() <= self.cfg.spec_max_streams && decoding.iter().all(|&i| speculate(&self.streams[i])) {
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
                        hint_pos: 0,
                        hint_misses: 0,
                        accept: Accept::default(),
                        plain_run: 0,
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
        let t0 = std::time::Instant::now();
        let logits = self.target.forward(&items, false)?;
        let toks = self.target.backend.argmax_rows(&logits);
        drop(logits);
        if picked.len() == 1 {
            self.timing.plain.add(t0.elapsed().as_secs_f64());
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
        let left = self.streams[i].req.max_new_tokens - self.streams[i].generated;
        let s = &self.streams[i];
        let from_hint = hint_match(&s.req.hint, &s.out, s.hint_pos).map(|p| {
            let hint = &s.req.hint;
            let k = self.cfg.hint_k.min(left);
            let mut props = hint[p..(p + k).min(hint.len())].to_vec();
            if props.len() < k
                && let Some(end) = self.end_token.filter(|e| s.req.stop.contains(e))
            {
                props.push(end);
            }
            (p, props)
        });
        let hinted = !s.req.hint.is_empty();
        let mut round_t0 = None;
        // After the output leaves the hint it often rejoins it a few tokens later; until
        // then single decode steps are cheaper than catching the draft model up.
        let draft_lag = s.ds.map_or(usize::MAX, |d| {
            self.target.seq_len(s.ts) - self.draft.as_ref().map_or(0, |m| m.seq_len(d))
        });
        let (props, hint_at) = match from_hint {
            Some((p, props)) if !props.is_empty() => {
                self.streams[i].hint_misses = 0;
                (props, Some(p))
            }
            _ if hinted && self.streams[i].hint_misses < HINT_PATIENCE && draft_lag > 16 => {
                self.streams[i].hint_misses += 1;
                return self.decode_step(&[i], emit);
            }
            _ if self.draft.is_some() => match self.draft_len(i, left) {
                0 => {
                    self.streams[i].plain_run += 1;
                    self.spec.plain_steps += 1;
                    return self.decode_step(&[i], emit);
                }
                k => {
                    self.streams[i].plain_run = 0;
                    round_t0 = Some(std::time::Instant::now());
                    (self.draft_tokens(i, next, k)?, None)
                }
            },
            _ => return self.decode_step(&[i], emit),
        };
        let ts = self.streams[i].ts;
        let base = self.target.seq_len(ts);

        // Verify: g[j] is the target's choice after consuming ver[..=j].
        let mut ver = Vec::with_capacity(props.len() + 1);
        ver.push(next);
        ver.extend_from_slice(&props);
        let t0 = std::time::Instant::now();
        let logits = self.target.forward(&[(ts, &ver)], true)?;
        let g = self.target.backend.argmax_rows(&logits);
        drop(logits);
        if let Some(e) = self.timing.verify.get_mut(ver.len()) {
            e.add(t0.elapsed().as_secs_f64());
        }
        let m = props.iter().zip(&g).take_while(|(p, t)| p == t).count();
        self.spec.rounds += 1;
        self.spec.proposed += props.len();
        self.spec.accepted += m;
        // Rounds that first caught the draft up are not typical: leave them out.
        if let (Some(r0), true, Some(d), Some(v)) =
            (round_t0, self.draft_in_sync, self.timing.draft.get(), self.timing.target(ver.len()))
        {
            let model = props.len() as f64 * d + v;
            self.timing.round_extra.add((r0.elapsed().as_secs_f64() - model).max(0.0));
        }
        if let Some(p) = hint_at {
            self.spec.hint_rounds += 1;
            self.streams[i].hint_pos = p + m + 1;
        } else {
            self.streams[i].accept.add(props.len(), m, 0.9);
            self.accept.add(props.len(), m, 0.995);
        }

        self.target.truncate(ts, base + 1 + m);
        if let (Some(ds), Some(draft)) = (self.streams[i].ds, self.draft.as_mut())
            && draft.seq_len(ds) > base + 1 + m
        {
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

    /// The draft length for stream `i`'s next round (0: a single decode step is faster).
    /// It maximizes the expected tokens per second: a round of `k` drafts yields
    /// `1 + sum_j P(first j accepted)` tokens and costs `k` draft steps plus a target pass
    /// of `k + 1` rows, against one token per plain step.
    fn draft_len(&self, i: usize, left: usize) -> usize {
        let s = &self.streams[i];
        let kmax = self.cfg.draft_k.min(left);
        let (Some(plain), Some(draft)) = (self.timing.target(1), self.timing.draft.get()) else {
            return kmax.min(DEFAULT_DRAFT_K); // measure first
        };
        if self.accept.reach1 < WARMUP_ROUNDS {
            return kmax.min(DEFAULT_DRAFT_K); // too little acceptance data yet
        }
        // This request's acceptance, starting from that of all requests so far.
        let global = self.accept.rates((0.75, 0.75), 1.0);
        let (c1, c) = s.accept.rates(global, 20.0);
        let extra = self.timing.round_extra.get().unwrap_or(0.0);
        // Expected tokens per second of each length (0: a plain step).
        let mut rates = vec![1.0 / plain];
        let (mut tokens, mut surv) = (1.0, 1.0);
        for k in 1..=kmax {
            surv *= if k == 1 { c1 } else { c };
            tokens += surv;
            let rate = self.timing.target(k + 1).map_or(0.0, |verify| tokens / (k as f64 * draft + verify + extra));
            rates.push(rate);
        }
        // Keep the default length unless another one is clearly better: the estimates are
        // rough, and small mispredictions cost more than they gain.
        let default = kmax.min(DEFAULT_DRAFT_K);
        let (k_best, r_best) = rates.iter().copied().enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        let best = (r_best, if r_best > SWITCH_GAIN * rates[default] { k_best } else { default });
        if best.1 == 0 && s.plain_run >= PROBE_AFTER {
            return kmax.min(2); // re-measure the acceptance now and then
        }
        best.1
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
        let t0 = std::time::Instant::now();
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
        self.draft_in_sync = caught_up;
        if caught_up {
            self.timing.draft.add(t0.elapsed().as_secs_f64() / props.len() as f64);
        }
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
    fn hint_follows_the_output() {
        let hint = [10, 11, 12, 13, 11, 12, 14];
        // Start of the output: the first token matches near the start of the hint.
        assert_eq!(hint_match(&hint, &[10], 0), Some(1));
        // A longer match wins; between equal ones, the one closest to `pos`.
        assert_eq!(hint_match(&hint, &[10, 11, 12], 3), Some(3));
        assert_eq!(hint_match(&hint, &[13, 11, 12], 3), Some(6));
        assert_eq!(hint_match(&hint, &[9, 11, 12], 6), Some(6));
        // One token far from the alignment is not trusted.
        assert_eq!(hint_match(&hint, &[9, 14], 0), None);
        assert_eq!(hint_match(&hint, &[99], 0), None);
    }

    #[test]
    fn acceptance_counts_by_position() {
        let mut a = Accept::default();
        // 5 drafts, 2 accepted: first token accepted, later ones reached twice (depths 2
        // and 3), accepted once.
        a.add(5, 2, 1.0);
        assert_eq!((a.reach1, a.hit1, a.reach, a.hit), (1.0, 1.0, 2.0, 1.0));
        // All of 3 accepted: depths 2 and 3 reached and accepted.
        a.add(3, 3, 1.0);
        assert_eq!((a.reach1, a.hit1, a.reach, a.hit), (2.0, 2.0, 4.0, 3.0));
        // First rejected: only depth 1 reached.
        a.add(4, 0, 1.0);
        assert_eq!((a.reach1, a.hit1, a.reach, a.hit), (3.0, 2.0, 4.0, 3.0));
        let (c1, c) = a.rates((0.5, 0.5), 1.0);
        assert!((c1 - 2.5 / 4.0).abs() < 1e-9 && (c - 3.5 / 5.0).abs() < 1e-9);
    }

    #[test]
    fn response_cache_answers_only_complete_outputs() {
        let mut c = ResponseCache::default();
        let req = |max| Request::new(0, vec![1, 2, 3], max, vec![7]);
        c.put(2, vec![1, 2, 3], vec![7], vec![4, 5], Finish::Stop);
        assert_eq!(c.get(&req(10)), Some((vec![4, 5], Finish::Stop)));
        assert_eq!(c.get(&req(1)), Some((vec![4], Finish::Length)));
        // Another stop list is another request.
        assert_eq!(c.get(&Request::new(0, vec![1, 2, 3], 10, vec![8])), None);
        // An output cut at the limit does not answer a request with a larger limit.
        c.put(2, vec![9], vec![7], vec![4, 5], Finish::Length);
        assert_eq!(c.get(&Request::new(0, vec![9], 3, vec![7])), None);
        assert_eq!(c.get(&Request::new(0, vec![9], 2, vec![7])), Some((vec![4, 5], Finish::Length)));
        // Least recently used entries go first.
        c.put(2, vec![8], vec![7], vec![1], Finish::Stop);
        assert_eq!(c.get(&req(10)), None);
    }
}
