//! Requests of the HTTP API as prompts.

use yaxus_api::{Pair, TranslateRequest};
use yaxus_tokenizer::{GlossaryEntry, PromptError, Tokenizer, TranslationPair, TranslationRequest};

use crate::memory;

/// The prompt fields of an API request.
pub fn prompt_request(r: &TranslateRequest) -> TranslationRequest {
    TranslationRequest {
        text: r.text.clone(),
        target_lang: r.target_lang.clone(),
        source_lang: r.requested_source_lang().map(String::from),
        form: r.requested_form().map(String::from),
        context: r.context.clone(),
        glossary: glossary(&r.glossary),
        history: history(&r.history),
    }
}

/// The API's glossary pairs as prompt entries.
pub(crate) fn glossary(pairs: &[Pair]) -> Vec<GlossaryEntry> {
    pairs.iter().map(|p| GlossaryEntry { source: p.source.clone(), target: p.target.clone() }).collect()
}

/// The API's history pairs (source, translation) as prompt entries.
pub(crate) fn history(pairs: &[Pair]) -> Vec<TranslationPair> {
    pairs.iter().map(|p| TranslationPair { source: p.source.clone(), target: p.target.clone() }).collect()
}

/// A translation request as the engine takes it: the prompt's token ids, and the key
/// under which the translation memory keeps its output.
#[derive(Debug, Clone)]
pub struct Prompt {
    ids: Vec<u32>,
    memo: u64,
}

impl Prompt {
    pub fn new(tok: &Tokenizer, req: &TranslationRequest) -> Result<Self, PromptError> {
        Ok(Prompt { ids: tok.encode(&req.to_prompt()?, false), memo: memory::key(req) })
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
    pub(crate) fn from_parts(ids: Vec<u32>, memo: u64) -> Self {
        Prompt { ids, memo }
    }

    pub(crate) fn into_parts(self) -> (Vec<u32>, u64) {
        (self.ids, self.memo)
    }
}
