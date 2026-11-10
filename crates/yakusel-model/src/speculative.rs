//! Greedy speculative decoding: a small draft model proposes `k` tokens, the target model
//! checks them all in one forward pass, and the longest agreeing prefix is kept.
//!
//! With greedy decoding the output is exactly the target model's own greedy output;
//! the draft only decides how many tokens each target pass can confirm.

use anyhow::{Result, ensure};

use crate::{Backend, Generator, Model, SeqId};

#[derive(Debug, Default, Clone)]
pub struct SpecStats {
    /// Target verification passes.
    pub rounds: usize,
    /// Draft tokens proposed / accepted by the target.
    pub proposed: usize,
    pub accepted: usize,
}

pub struct Speculative<A: Backend, D: Backend> {
    pub target: Model<A>,
    pub draft: Model<D>,
    /// Maximum tokens drafted per round (the verification pass handles `k + 1` rows).
    pub k: usize,
    pub stats: SpecStats,
}

impl<A: Backend, D: Backend> Speculative<A, D> {
    pub fn new(target: Model<A>, draft: Model<D>, k: usize) -> Result<Self> {
        ensure!(
            target.cfg.vocab_size == draft.cfg.vocab_size,
            "draft vocabulary ({}) differs from the target's ({})",
            draft.cfg.vocab_size,
            target.cfg.vocab_size
        );
        ensure!((1..=16).contains(&k), "draft length must be 1..=16");
        Ok(Speculative { target, draft, k, stats: SpecStats::default() })
    }

    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        stop: &[u32],
        mut on_token: impl FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        let ts = self.target.new_seq()?;
        let ds = match self.draft.new_seq() {
            Ok(d) => d,
            Err(e) => {
                self.target.free_seq(ts);
                return Err(e);
            }
        };
        let r = self.run(ts, ds, prompt, max_new_tokens, stop, &mut on_token);
        self.target.free_seq(ts);
        self.draft.free_seq(ds);
        r
    }

    fn run(
        &mut self,
        ts: SeqId,
        ds: SeqId,
        prompt: &[u32],
        max_new_tokens: usize,
        stop: &[u32],
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let logits = self.target.prefill(ts, prompt)?;
        let mut next = self.target.backend.argmax(&logits);
        drop(logits);
        // The draft only needs the prompt in its cache.
        drop(self.draft.prefill(ds, prompt)?);
        // Committed tokens the draft has not consumed yet.
        let mut pending: Vec<u32> = Vec::new();

        loop {
            // `next` is the target's greedy token; the target has not consumed it yet.
            if stop.contains(&next) {
                return Ok(out);
            }
            out.push(next);
            if !on_token(next) || out.len() >= max_new_tokens {
                return Ok(out);
            }
            pending.push(next);
            let base = self.target.seq_len(ts);
            debug_assert_eq!(self.draft.seq_len(ds) + pending.len(), base + 1);

            // Draft up to k tokens, no more than can still be emitted.
            let k = self.k.min(max_new_tokens - out.len());
            let mut props = Vec::with_capacity(k);
            let mut dl = self.draft.forward(&[(ds, &pending)], false)?;
            pending.clear();
            loop {
                let d = self.draft.backend.argmax(&dl);
                props.push(d);
                if props.len() == k || stop.contains(&d) {
                    break;
                }
                dl = self.draft.forward(&[(ds, &[d])], false)?;
            }
            drop(dl);

            // Verify: g[i] is the target's choice after consuming ver[..=i].
            let mut ver = Vec::with_capacity(props.len() + 1);
            ver.push(next);
            ver.extend_from_slice(&props);
            let logits = self.target.forward(&[(ts, &ver)], true)?;
            let g = self.target.backend.argmax_rows(&logits);
            drop(logits);
            let m = props.iter().zip(&g).take_while(|(p, t)| p == t).count();
            self.stats.rounds += 1;
            self.stats.proposed += props.len();
            self.stats.accepted += m;

            self.target.truncate(ts, base + 1 + m);
            if m < props.len() {
                // The draft consumed props[..len-1]; keep only the accepted ones.
                self.draft.truncate(ds, base + 1 + m);
            } else {
                // All accepted: the draft never consumed the last proposal.
                pending.push(props[m - 1]);
            }
            for &t in &props[..m] {
                if stop.contains(&t) {
                    return Ok(out);
                }
                out.push(t);
                if !on_token(t) || out.len() >= max_new_tokens {
                    return Ok(out);
                }
            }
            next = g[m];
        }
    }
}

impl<A: Backend + Send, D: Backend + Send> Generator for Speculative<A, D>
where
    A::Weight: Send,
    D::Weight: Send,
    A::Kv: Send,
    D::Kv: Send,
{
    fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        stop: &[u32],
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        Speculative::generate(self, prompt, max_new_tokens, stop, on_token)
    }
    fn backend_name(&self) -> String {
        format!("{} + draft on {} (k={})", self.target.backend.name(), self.draft.backend.name(), self.k)
    }
    fn weight_bytes(&self) -> usize {
        self.target.backend.weight_bytes() + self.draft.backend.weight_bytes()
    }
    fn stats(&self) -> Option<String> {
        let s = &self.stats;
        (s.rounds > 0).then(|| {
            format!(
                "{} rounds, {:.0}% of drafts accepted, {:.2} tokens per target pass",
                s.rounds,
                100.0 * s.accepted as f64 / s.proposed.max(1) as f64,
                (s.accepted + s.rounds) as f64 / s.rounds as f64
            )
        })
    }
}
