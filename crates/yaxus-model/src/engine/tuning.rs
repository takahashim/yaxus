//! How long a speculative round should be: measured pass times and draft acceptance.

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

/// Measured pass times.
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

/// What the tuner knows of one request.
#[derive(Debug, Default)]
pub(super) struct StreamTuning {
    /// Draft acceptance in this request's rounds.
    accept: Accept,
    /// Single decode steps since the last draft round.
    plain_run: usize,
}

/// Chooses the draft length of each round from the pass times and acceptance it is told
/// about.
#[derive(Debug, Default)]
pub(super) struct DraftTuner {
    timing: Timing,
    /// Draft acceptance over all requests (the prior of each request's own).
    accept: Accept,
    /// The last draft run started in sync with the target (no catch-up prefill), so its
    /// round is typical.
    in_sync: bool,
}

impl DraftTuner {
    /// The draft length for a request's next round, at most `kmax` (0: a single decode
    /// step is faster); counts the request's single steps. It maximizes the expected
    /// tokens per second: a round of `k` drafts yields `1 + sum_j P(first j accepted)`
    /// tokens and costs `k` draft steps plus a target pass of `k + 1` rows, against one
    /// token per plain step.
    pub(super) fn choose(&self, st: &mut StreamTuning, kmax: usize) -> usize {
        let k = self.best_len(st, kmax);
        st.plain_run = if k == 0 { st.plain_run + 1 } else { 0 };
        k
    }

    fn best_len(&self, st: &StreamTuning, kmax: usize) -> usize {
        let default = kmax.min(DEFAULT_DRAFT_K);
        let (Some(plain), Some(draft)) = (self.timing.target(1), self.timing.draft.get()) else {
            return default; // measure first
        };
        if self.accept.reach1 < WARMUP_ROUNDS {
            return default; // too little acceptance data yet
        }
        // This request's acceptance, starting from that of all requests so far.
        let global = self.accept.rates((0.75, 0.75), 1.0);
        let (c1, c) = st.accept.rates(global, 20.0);
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
        let (k_best, r_best) = rates.iter().copied().enumerate().max_by(|a, b| a.1.total_cmp(&b.1)).unwrap();
        let best = if r_best > SWITCH_GAIN * rates[default] { k_best } else { default };
        if best == 0 && st.plain_run >= PROBE_AFTER {
            return kmax.min(2); // re-measure the acceptance now and then
        }
        best
    }

    /// A single target decode step took `secs`.
    pub(super) fn plain_step(&mut self, secs: f64) {
        self.timing.plain.add(secs);
    }

    /// A target verification pass of `rows` rows took `secs`.
    pub(super) fn verify_pass(&mut self, rows: usize, secs: f64) {
        if let Some(e) = self.timing.verify.get_mut(rows) {
            e.add(secs);
        }
    }

    /// A draft run took `secs_per_step`; `caught_up` when it started in sync with the
    /// target (otherwise its time includes a catch-up prefill and is not recorded).
    pub(super) fn draft_run(&mut self, caught_up: bool, secs_per_step: f64) {
        self.in_sync = caught_up;
        if caught_up {
            self.timing.draft.add(secs_per_step);
        }
    }

    /// The draft run before the next one does not continue the last (the engine dropped
    /// its requests).
    pub(super) fn lose_sync(&mut self) {
        self.in_sync = false;
    }

    /// A draft round of `k` drafts of which `m` were accepted took `secs` in all
    /// (verifying `k + 1` rows).
    pub(super) fn draft_round(&mut self, st: &mut StreamTuning, k: usize, m: usize, secs: f64) {
        // Rounds that first caught the draft up are not typical: leave them out.
        if self.in_sync
            && let (Some(d), Some(v)) = (self.timing.draft.get(), self.timing.target(k + 1))
        {
            self.timing.round_extra.add((secs - (k as f64 * d + v)).max(0.0));
        }
        st.accept.add(k, m, 0.9);
        self.accept.add(k, m, 0.995);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn the_length_adapts_once_measured() {
        let mut t = DraftTuner::default();
        let mut st = StreamTuning::default();
        // Nothing measured: the default length (or less when fewer tokens are left).
        assert_eq!((t.choose(&mut st, 7), t.choose(&mut st, 3)), (DEFAULT_DRAFT_K, 3));
        // Drafting costs as much as decoding and nothing is accepted: plain steps.
        t.plain_step(0.01);
        t.draft_run(true, 0.01);
        for k in 2..=8 {
            t.verify_pass(k, 0.01);
        }
        for _ in 0..100 {
            t.draft_round(&mut st, 5, 0, 0.06);
        }
        assert_eq!(t.choose(&mut st, 7), 0);
        // After PROBE_AFTER of them, a short round re-measures the acceptance.
        for _ in 1..PROBE_AFTER {
            t.choose(&mut st, 7);
        }
        assert_eq!(t.choose(&mut st, 7), 2);
        // A cheap draft that is always accepted: the longest round.
        let mut t = DraftTuner::default();
        let mut st = StreamTuning::default();
        t.plain_step(0.02);
        t.draft_run(true, 0.001);
        for k in 2..=8 {
            t.verify_pass(k, 0.02);
        }
        for _ in 0..100 {
            t.draft_round(&mut st, 7, 7, 0.03);
        }
        assert_eq!(t.choose(&mut st, 7), 7);
    }
}
