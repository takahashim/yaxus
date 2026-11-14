//! The history each chain keeps for its next segment: the pairs of source and
//! translation of the segments it has finished, and the window that says how many of the
//! newest it sends.

use yaxus_api::{HistoryWindow, Pair};
use yaxus_pfml::validate_history_pair;

/// Which earlier translations of a chain go into the history of the next segment, and how.
/// The default is the CLI's and the server's: every one, as the model wrote it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HistoryRules {
    /// Leave out a translation that was cut at the output budget: it would be a
    /// misleading example for the next segment.
    pub skip_truncated: bool,
    /// Put a translation in without the whitespace around it.
    pub trim: bool,
}

/// The history of every chain of a document.
pub struct ChainHistories {
    rules: HistoryRules,
    chains: Vec<Chain>,
}

struct Chain {
    window: HistoryWindow,
    /// The pairs a prompt may hold, oldest first, checked once when they were added: no
    /// blank segment or translation, and no translation with a reserved token (a control
    /// token the model emitted), nor one the rules leave out.
    pairs: Vec<Pair>,
}

impl ChainHistories {
    /// `chains` empty histories, each sending at most `budget_chars` characters (0: none).
    pub fn new(chains: usize, budget_chars: usize, rules: HistoryRules) -> Self {
        let chain = || Chain { window: HistoryWindow::new(budget_chars), pairs: Vec::new() };
        ChainHistories { rules, chains: (0..chains).map(|_| chain()).collect() }
    }

    /// The segment `source` of `chain` ended with `translation`; `truncated` is whether it
    /// was cut at the output budget.
    pub fn record(&mut self, chain: usize, source: &str, translation: &str, truncated: bool) {
        if self.rules.skip_truncated && truncated {
            return;
        }
        let target = if self.rules.trim { translation.trim() } else { translation };
        let pair = Pair { source: source.to_string(), target: target.to_string() };
        if validate_history_pair(&pair).is_ok() {
            self.chains[chain].pairs.push(pair);
        }
    }

    /// The history for the next segment of `chain`: the pairs its window keeps (only
    /// those are copied).
    pub fn select(&mut self, chain: usize) -> Vec<Pair> {
        let Chain { window, pairs } = &mut self.chains[chain];
        window.select(pairs).to_vec()
    }

    /// The characters a chain's history may hold.
    pub fn budget_chars(&self) -> usize {
        self.chains.first().map_or(0, |c| c.window.budget_chars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn targets(h: &mut ChainHistories, chain: usize) -> Vec<String> {
        h.select(chain).into_iter().map(|p| p.target).collect()
    }

    #[test]
    fn what_a_prompt_cannot_hold_is_not_recorded() {
        let mut h = ChainHistories::new(1, 1000, HistoryRules::default());
        h.record(0, "a", "A", false);
        h.record(0, "b", "  ", false); // a blank translation
        h.record(0, "c", "C<|plamo:msg|>", false); // a control token the model emitted
        h.record(0, "  ", "D", false); // a blank source
        h.record(0, "e", "E", true); // truncated: kept, without the rule
        assert_eq!(targets(&mut h, 0), ["A", "E"]);
    }

    #[test]
    fn the_rules_trim_and_leave_out_truncated_translations() {
        let mut h = ChainHistories::new(1, 1000, HistoryRules { skip_truncated: true, trim: true });
        h.record(0, "a", " A\n", false);
        h.record(0, "b", "B", true);
        assert_eq!(targets(&mut h, 0), ["A"]);
    }

    #[test]
    fn chains_keep_their_own_history() {
        let mut h = ChainHistories::new(2, 1000, HistoryRules::default());
        h.record(0, "a", "A", false);
        h.record(1, "b", "B", false);
        assert_eq!((targets(&mut h, 0), targets(&mut h, 1)), (vec!["A".to_string()], vec!["B".to_string()]));
        assert_eq!(h.budget_chars(), 1000);
    }

    #[test]
    fn the_window_sends_only_the_newest_pairs_when_over_the_budget() {
        // 10 characters per pair: over the budget of 25, the newest pairs within a quarter
        // of it (at least one) are kept.
        let mut h = ChainHistories::new(1, 25, HistoryRules::default());
        for i in 0..4 {
            h.record(0, &format!("src-{i}"), &format!("trg-{i}"), false);
        }
        let kept = h.select(0);
        assert!(kept.len() < 4 && !kept.is_empty(), "{kept:?}");
        assert_eq!(kept.last().unwrap().source, "src-3", "the newest is kept");
    }
}
