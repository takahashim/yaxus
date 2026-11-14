//! One translation run to completion, as the local commands do it.

use std::time::Duration;

use anyhow::Result;
use yaxus_api::TranslateRequest;
use yaxus_model::{Generator, poll_once};
use yaxus_tokenizer::Tokenizer;

use crate::memory::Memory;
use crate::prompt::{Prompt, prompt_request};
use crate::session::{Session, Update};

/// Timing of one translation request.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunStats {
    pub generated: usize,
    /// Request start to the first generated token (prompt processing).
    pub prefill: Duration,
    /// First token to the end.
    pub decode: Duration,
}

/// Translates one request on `model` (with nothing else in flight, and without the
/// translation memory), streaming text pieces to `emit`. For an engine whose steps never
/// wait (see [`Generator::step`]); [`run_request_async`] is the form for any.
pub fn run_request(
    model: &mut dyn Generator,
    tok: &Tokenizer,
    req: &TranslateRequest,
    emit: &mut dyn FnMut(&str),
) -> Result<(String, RunStats)> {
    poll_once(run_request_async(model, tok, req, emit))
}

/// [`run_request`], awaiting the engine.
pub async fn run_request_async(
    model: &mut dyn Generator,
    tok: &Tokenizer,
    req: &TranslateRequest,
    emit: &mut dyn FnMut(&str),
) -> Result<(String, RunStats)> {
    let mut memory = Memory::new(0);
    let mut session = Session::new(model, tok, &mut memory);
    let prompt = Prompt::new(tok, &prompt_request(req).validate()?);
    session.submit(prompt, req.max_new_tokens.unwrap_or(yaxus_api::DEFAULT_MAX_NEW_TOKENS), ());
    let (mut done, mut failed) = (None, None);
    while done.is_none() && failed.is_none() {
        let worked = session
            .step_async(&mut |u| {
                match u {
                    Update::Delta { text, .. } => emit(text),
                    Update::Done { done: d, .. } => done = Some(d),
                    Update::Failed { message, .. } => failed = Some(message),
                }
                true
            })
            .await?;
        anyhow::ensure!(worked || session.is_idle(), "engine idle with the request unfinished");
    }
    if let Some(message) = failed {
        anyhow::bail!(message);
    }
    let done = done.expect("loop ends with an outcome");
    let stats = RunStats { generated: done.generated, prefill: done.prefill(), decode: done.decode() };
    Ok((done.text, stats))
}
