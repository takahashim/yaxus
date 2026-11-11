//! Driving a document through a [`Session`]: which segments are submitted, what their
//! ends and failures do to the document, and when it is over. The server's worker and
//! [`run_document`] differ only in where the text and the reports go.

use anyhow::Result;
use yaxus_api::DocumentUsage;
use yaxus_model::{Finish, Generator};
use yaxus_pfml::PromptError;
use yaxus_tokenizer::Tokenizer;

use super::DocJob;
use crate::memory::Memory;
use crate::session::{Finished, Session, Update};

/// What a segment that ended brought, for its report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentEnd {
    pub segment: usize,
    pub prompt_tokens: usize,
    /// Prompt tokens whose KV was reused.
    pub reused: usize,
    pub generated: usize,
}

/// A document in progress on a session.
///
/// Once stopped (by [`stop`](Self::stop), a failed segment or a prompt that cannot be
/// built), no segment is submitted, the segments in flight are cancelled at their next
/// text, and their ends are not recorded; the document is over when none is left.
pub struct DocRun {
    job: DocJob,
    /// Segments submitted and not ended.
    live: usize,
    stopped: bool,
    usage: DocumentUsage,
}

impl DocRun {
    pub fn new(job: DocJob) -> Self {
        let usage = DocumentUsage { segments: job.len(), chains: job.chains(), ..Default::default() };
        DocRun { job, live: 0, stopped: false, usage }
    }

    pub fn job(&self) -> &DocJob {
        &self.job
    }

    pub fn into_job(self) -> DocJob {
        self.job
    }

    /// Token counts of the segments that ended so far (the times are the caller's).
    pub fn usage(&self) -> &DocumentUsage {
        &self.usage
    }

    /// Submits the segments that can start, each tagged with `tag(segment)`. A prompt that
    /// cannot be built stops the document and is returned (as `segment N: ...`).
    pub fn submit_ready<T>(
        &mut self,
        session: &mut Session<'_, T>,
        tag: impl Fn(usize) -> T,
    ) -> Result<(), PromptError> {
        if self.stopped {
            return Ok(());
        }
        for (segment, req) in self.job.ready() {
            match self.job.prompt(session.tokenizer(), req) {
                Ok((prompt, max_new)) => {
                    self.live += 1;
                    session.submit(prompt, max_new, tag(segment));
                }
                Err(e) => {
                    self.stopped = true;
                    return Err(PromptError(format!("segment {segment}: {}", e.0)));
                }
            }
        }
        Ok(())
    }

    /// Whether the segments' text is still wanted; `false` cancels a segment's request.
    pub fn accepts(&self) -> bool {
        !self.stopped
    }

    /// Stops the document (e.g. its reader has gone).
    pub fn stop(&mut self) {
        self.stopped = true;
    }

    /// Records the end of `segment`'s request; returns its report, unless the document
    /// was stopped or the request cancelled.
    pub fn on_done(&mut self, segment: usize, done: Finished) -> Option<SegmentEnd> {
        self.live -= 1;
        if self.stopped || done.finish == Finish::Cancelled {
            return None;
        }
        let end = SegmentEnd {
            segment,
            prompt_tokens: done.prompt_tokens,
            reused: done.start.reused,
            generated: done.generated,
        };
        self.usage.prompt_tokens += end.prompt_tokens;
        self.usage.reused_tokens += end.reused;
        self.usage.completion_tokens += end.generated;
        self.job.finished(segment, done.text);
        Some(end)
    }

    /// Records the failure of `segment`'s request, which stops the document; returns the
    /// error to report (`segment N: ...`), unless the document was stopped already.
    pub fn on_failed(&mut self, segment: usize, message: &str) -> Option<String> {
        self.live -= 1;
        if std::mem::replace(&mut self.stopped, true) {
            return None;
        }
        Some(format!("segment {segment}: {message}"))
    }

    /// No segment is in flight and none will be submitted.
    pub fn is_over(&self) -> bool {
        self.live == 0 && (self.stopped || self.job.is_done())
    }

    /// Every segment is translated.
    pub fn is_complete(&self) -> bool {
        self.job.is_done()
    }
}

/// Events of [`run_document`].
pub enum DocEvent<'a> {
    Delta { segment: usize, text: &'a str },
    SegmentDone(SegmentEnd),
}

/// Translates a document on `engine` (which must not have other requests in flight) and
/// returns it with the translations. Earlier translations in `memory` serve as hints; the
/// new ones are added to it.
///
/// `on` returns `false` to stop (e.g. when the output it writes to has gone away): no
/// further segments are started, the ones in flight are cancelled and drained from the
/// engine, no more events are sent, and the document is returned unfinished. A failed
/// segment also stops it; its error is returned once the engine is drained.
pub fn run_document(
    engine: &mut dyn Generator,
    tok: &Tokenizer,
    job: DocJob,
    memory: &mut Memory,
    on: &mut dyn FnMut(DocEvent) -> bool,
) -> Result<DocJob> {
    let mut session = Session::new(engine, tok, memory);
    let mut run = DocRun::new(job);
    let mut failure: Option<String> = None;
    if let Err(e) = run.submit_ready(&mut session, |s| s) {
        failure = Some(e.0);
    }
    while !run.is_over() {
        let worked = session.step(&mut |u| {
            match u {
                Update::Delta { tag, text } => {
                    if !run.accepts() {
                        return false;
                    }
                    if !text.is_empty() && !on(DocEvent::Delta { segment: *tag, text }) {
                        run.stop();
                        return false;
                    }
                }
                Update::Done { tag, done } => {
                    if let Some(end) = run.on_done(tag, done)
                        && !on(DocEvent::SegmentDone(end))
                    {
                        run.stop();
                    }
                }
                Update::Failed { tag, message } => {
                    if let Some(e) = run.on_failed(tag, &message) {
                        failure.get_or_insert(e);
                    }
                }
            }
            true
        })?;
        if let Err(e) = run.submit_ready(&mut session, |s| s) {
            failure.get_or_insert(e.0);
        }
        anyhow::ensure!(worked || run.is_over() || !session.is_idle(), "engine idle with segments left");
    }
    if let Some(e) = failure {
        anyhow::bail!(e);
    }
    Ok(run.into_job())
}
