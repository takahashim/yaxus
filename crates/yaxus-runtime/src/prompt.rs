//! Requests of the HTTP API as prompts.

use yaxus_api::{DEFAULT_MAX_NEW_TOKENS, TranslateRequest};
use yaxus_pfml::{PromptError, TranslationRequest, ValidRequest};
use yaxus_tokenizer::Tokenizer;

use crate::memory::{self, MemoryKey};

/// The prompt fields of an API request.
pub fn prompt_request(r: &TranslateRequest) -> TranslationRequest {
    TranslationRequest {
        text: r.text.clone(),
        target_lang: r.target_lang.clone(),
        source_lang: r.source_lang.clone(),
        form: r.form,
        context: r.context.clone(),
        glossary: r.glossary.clone(),
        history: r.history.clone(),
    }
}

/// A translation request as the engine takes it: the prompt's token ids, and the key
/// under which the translation memory keeps its output.
#[derive(Debug, Clone)]
pub struct Prompt {
    ids: Vec<u32>,
    memo: MemoryKey,
}

impl Prompt {
    pub fn new(tok: &Tokenizer, req: &ValidRequest) -> Self {
        Prompt { ids: tok.encode(&req.render(), false), memo: memory::key(req) }
    }

    /// The prompt of an API request and its generation budget (`max_new_tokens`, cut to
    /// what the prompt leaves of `max_context`). Its history is kept: a request that does
    /// not fit is an error.
    pub fn for_request(
        tok: &Tokenizer,
        r: &TranslateRequest,
        max_context: usize,
    ) -> Result<(Self, usize), PromptError> {
        let req = prompt_request(r).validate()?;
        Prompt::new(tok, &req).within(max_context, r.max_new_tokens.unwrap_or(DEFAULT_MAX_NEW_TOKENS))
    }

    /// The prompt of `req` and its budget within `max_context` (see
    /// [`yaxus_pfml::fit_prompt`]: a history that leaves too little room is dropped).
    pub fn fit(
        tok: &Tokenizer,
        req: ValidRequest,
        max_context: usize,
        max_new: usize,
    ) -> Result<(Self, usize), PromptError> {
        let (req, ids, max_new) = yaxus_pfml::fit_prompt(tok, req, max_context, max_new)?;
        Ok((Prompt { ids, memo: memory::key(&req) }, max_new))
    }

    /// This prompt with the budget `max_new` cut to what it leaves of `max_context` (at
    /// least 1), or an error when the prompt alone does not fit.
    fn within(self, max_context: usize, max_new: usize) -> Result<(Self, usize), PromptError> {
        let max_new = yaxus_pfml::budget_within(self.len(), max_context, max_new)?;
        Ok((self, max_new))
    }

    /// The token ids, for callers that build engine requests themselves (benchmarks).
    pub fn into_ids(self) -> Vec<u32> {
        self.ids
    }

    /// Prompt length in tokens.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn from_parts(ids: Vec<u32>, memo: MemoryKey) -> Self {
        Prompt { ids, memo }
    }

    pub(crate) fn into_parts(self) -> (Vec<u32>, MemoryKey) {
        (self.ids, self.memo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_request_keeps_its_history_and_gets_the_budget_left() {
        let tok = yaxus_tokenizer::testing::plamo_like(&[("x", 0.0)]);
        let mut r = TranslateRequest {
            text: "x".into(),
            target_lang: "en".into(),
            history: vec![yaxus_api::Pair { source: "x".into(), target: "x".into() }],
            max_new_tokens: Some(1000),
            ..Default::default()
        };
        let full = Prompt::new(&tok, &prompt_request(&r).validate().unwrap()).len();
        let (p, max_new) = Prompt::for_request(&tok, &r, full + 10).unwrap();
        assert_eq!((p.len(), max_new), (full, 10));
        // No room left: an error, not a prompt without the history.
        let e = Prompt::for_request(&tok, &r, full).unwrap_err();
        assert!(e.message.contains("the limit is") && e.kind == yaxus_pfml::PromptErrorKind::TooLong, "{e:?}");
        r.target_lang.clear();
        assert!(Prompt::for_request(&tok, &r, full + 10).is_err());
    }
}
