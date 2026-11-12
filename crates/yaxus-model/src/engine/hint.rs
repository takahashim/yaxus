//! Hints: tokens an output is likely to repeat, verified like draft tokens where the
//! output matches them.

/// Single decode steps tried after the output left its hint before the draft model
/// takes over.
const HINT_PATIENCE: usize = 4;

/// Where a request's output stands against its hint.
#[derive(Debug, Default)]
pub(super) struct HintCursor {
    /// Where the hint is expected to continue the output.
    pos: usize,
    /// Rounds in a row in which the output did not match the hint.
    misses: usize,
}

impl HintCursor {
    /// The hint tokens to verify after `out`, at most `k` of them, and the hint index of
    /// the first; `end` (a stop token) is appended when the hint runs out, so that the
    /// end of an output is verified with it.
    pub(super) fn propose(
        &mut self,
        hint: &[u32],
        out: &[u32],
        k: usize,
        end: Option<u32>,
    ) -> Option<(usize, Vec<u32>)> {
        let found = hint_match(hint, out, self.pos).map(|p| {
            let mut props = hint[p..(p + k).min(hint.len())].to_vec();
            if props.len() < k
                && let Some(end) = end
            {
                props.push(end);
            }
            (p, props)
        });
        let found = found.filter(|(_, props)| !props.is_empty());
        if found.is_some() {
            self.misses = 0;
        }
        found
    }

    /// Whether to decode a single step in the hope that the output rejoins the hint (it
    /// often does a few tokens later) rather than catch the draft model up: yes for
    /// `HINT_PATIENCE` rounds in a row.
    pub(super) fn wait(&mut self) -> bool {
        let wait = self.misses < HINT_PATIENCE;
        if wait {
            self.misses += 1;
        }
        wait
    }

    /// The first `accepted` of the tokens proposed from hint index `at` were accepted.
    pub(super) fn advance(&mut self, at: usize, accepted: usize) {
        self.pos = at + accepted + 1;
    }
}

/// Hint tokens to verify after `out`: the ones following the latest `n` output tokens
/// (n = 3, 2, or 1 near `pos`) in the hint, at the occurrence closest to `pos`.
/// Returns the hint index of the first proposed token.
fn hint_match(hint: &[u32], out: &[u32], pos: usize) -> Option<usize> {
    for n in (1..=out.len().min(3)).rev() {
        let tail = &out[out.len() - n..];
        let best = (n..hint.len()).filter(|&p| &hint[p - n..p] == tail).min_by_key(|&p| p.abs_diff(pos));
        match best {
            // A single token matches too easily far from where the hint is aligned.
            Some(p) if n > 1 || p.abs_diff(pos) <= 4 => return Some(p),
            _ => {}
        }
    }
    None
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
    fn the_cursor_follows_the_hint_and_waits_a_while_after_leaving_it() {
        let hint = [10, 11, 12, 13];
        let mut c = HintCursor::default();
        assert_eq!(c.propose(&hint, &[10], 2, None), Some((1, vec![11, 12])));
        c.advance(1, 2);
        // The rest of the hint, then the end token.
        assert_eq!(c.propose(&hint, &[10, 11, 12], 3, Some(1)), Some((3, vec![13, 1])));
        // Off the hint: HINT_PATIENCE single steps, then the draft model.
        assert_eq!(c.propose(&hint, &[10, 99], 3, None), None);
        assert!((0..HINT_PATIENCE).all(|_| c.wait()));
        assert!(!c.wait());
        // Back on it: patience again.
        assert!(c.propose(&hint, &[10, 11], 3, None).is_some());
        assert!(c.wait());
    }
}
