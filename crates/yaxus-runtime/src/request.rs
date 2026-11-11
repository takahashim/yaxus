//! Requests of the HTTP API as prompts, and a single translation run to completion.

use std::time::{Duration, Instant};

use anyhow::Result;
use yaxus_api::TranslateRequest;
use yaxus_model::Generator;
use yaxus_tokenizer::{EOS_ID, GlossaryEntry, StreamDecoder, TAG_ID, Tokenizer, TranslationPair, TranslationRequest};

/// An optional request field (`source_lang`, `form`) as given: an empty string means
/// unset, as if the field were left out.
pub fn non_empty(field: &Option<String>) -> Option<&str> {
    field.as_deref().filter(|s| !s.is_empty())
}

/// The prompt fields of an API request.
pub fn prompt_request(r: &TranslateRequest) -> TranslationRequest {
    TranslationRequest {
        text: r.text.clone(),
        target_lang: r.target_lang.clone(),
        source_lang: non_empty(&r.source_lang).map(String::from),
        form: non_empty(&r.form).map(String::from),
        context: r.context.clone(),
        glossary: r
            .glossary
            .iter()
            .map(|p| GlossaryEntry { source: p.source.clone(), target: p.target.clone() })
            .collect(),
        history: r
            .history
            .iter()
            .map(|p| TranslationPair { source: p.source.clone(), target: p.target.clone() })
            .collect(),
    }
}

/// Timing of one translation request.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunStats {
    pub generated: usize,
    /// Request start to the first generated token (prompt processing).
    pub prefill: Duration,
    /// First token to the end.
    pub decode: Duration,
}

/// Translates one request on `model` (with nothing else in flight), streaming text
/// pieces to `emit`.
pub fn run_request(
    model: &mut dyn Generator,
    tok: &Tokenizer,
    req: &TranslateRequest,
    emit: &mut dyn FnMut(&str),
) -> Result<(String, RunStats)> {
    let prompt = prompt_request(req).to_prompt()?;
    let ids = tok.encode(&prompt, false);
    let t0 = Instant::now();
    let mut first: Option<Instant> = None;
    let mut dec = StreamDecoder::new();
    let mut text = String::new();
    let max_new = req.max_new_tokens.unwrap_or(yaxus_api::DEFAULT_MAX_NEW_TOKENS);
    let out = model.generate(&ids, max_new, &[EOS_ID, TAG_ID], &mut |id| {
        first.get_or_insert_with(Instant::now);
        let start = text.len();
        dec.push_into(tok, id, &mut text);
        emit(&text[start..]);
        true
    })?;
    let tail = dec.finish();
    emit(&tail);
    text += &tail;
    let end = Instant::now();
    let first = first.unwrap_or(end);
    Ok((text, RunStats { generated: out.len(), prefill: first - t0, decode: end - first }))
}
