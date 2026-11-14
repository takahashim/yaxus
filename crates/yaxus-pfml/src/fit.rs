//! Fitting a prompt and its generation budget into a context window.

use yaxus_tokenizer::Tokenizer;

use crate::{PromptError, ValidRequest};

/// Generation budget a request keeps its history for in [`fit_prompt`]: `max_new` is a
/// cap that outputs rarely approach, so a prompt with history that leaves this much is
/// kept.
pub const MIN_BUDGET: usize = 512;

/// The generation budget `max_new` cut to what a prompt of `prompt_len` tokens leaves of
/// `max_context` (at least 1), or an error when the prompt alone does not fit.
pub fn budget_within(prompt_len: usize, max_context: usize, max_new: usize) -> Result<usize, PromptError> {
    if prompt_len >= max_context {
        return Err(PromptError::too_long(format!("prompt is {prompt_len} tokens; the limit is {max_context}")));
    }
    Ok(max_new.min(max_context - prompt_len).max(1))
}

/// The prompt of `req` such that it and the generation budget fit in `max_context`
/// tokens: the budget is cut to what the prompt leaves, and the history is dropped only
/// when that would leave less than [`MIN_BUDGET`] (or `max_new`, if smaller). Returns the
/// request the prompt is of (without its history when it was dropped), the prompt's
/// token ids and the budget, or an error when even the prompt without history does not
/// fit.
pub fn fit_prompt(
    tok: &Tokenizer,
    req: ValidRequest,
    max_context: usize,
    max_new: usize,
) -> Result<(ValidRequest, Vec<u32>, usize), PromptError> {
    let mut req = req;
    let mut ids = tok.encode(&req.render(), false);
    if max_context.saturating_sub(ids.len()) < max_new.min(MIN_BUDGET) && !req.history().is_empty() {
        req = req.without_history();
        ids = tok.encode(&req.render(), false);
    }
    let max_new = budget_within(ids.len(), max_context, max_new)?;
    Ok((req, ids, max_new))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Pair, TranslationRequest};

    #[test]
    fn history_is_dropped_only_when_it_leaves_too_little_room() {
        let tok = yaxus_tokenizer::testing::plamo_like(&[]);
        let req = |history: Vec<Pair>| {
            TranslationRequest { text: "x".into(), target_lang: "en".into(), history, ..Default::default() }
                .validate()
                .unwrap()
        };
        let pair = Pair { source: "a".repeat(40), target: "b".repeat(40) };
        let bare = tok.encode(&req(vec![]).render(), false).len();
        let full = tok.encode(&req(vec![pair.clone()]).render(), false).len();
        assert!(full > bare);
        // Room for the history and the whole budget: kept, budget as asked.
        let (r, ids, max_new) = fit_prompt(&tok, req(vec![pair.clone()]), full + 100, 100).unwrap();
        assert_eq!((r.history().len(), ids.len(), max_new), (1, full, 100));
        // With the history, less than `min(max_new, MIN_BUDGET)` left: dropped.
        let (r, ids, max_new) = fit_prompt(&tok, req(vec![pair.clone()]), full + 10, 100).unwrap();
        assert_eq!((r.history().len(), ids.len(), max_new), (0, bare, 100.min(full + 10 - bare)));
        // Not even the bare prompt fits.
        let e = fit_prompt(&tok, req(vec![pair]), bare, 100).unwrap_err();
        assert_eq!(e.kind, crate::PromptErrorKind::TooLong);
    }
}
