//! Driving a document through a [`Session`]: which segments are submitted, what their
//! ends and failures do to the document, and when it is over. The server's worker and
//! [`run_document`] differ only in where the text and the reports go.

use std::time::Duration;

use anyhow::Result;
use yaxus_api::DocumentUsage;
use yaxus_model::{Finish, Generator, SyncGenerator, poll_once};
use yaxus_tokenizer::Tokenizer;

use super::{DocJob, SegmentError};
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
    /// How it ended: `Stop` (complete) or `Length` (cut at the output budget).
    pub finish: Finish,
    /// Submission to the first token (prompt processing), as the session measured it.
    pub prefill: Duration,
    /// First token to the end.
    pub decode: Duration,
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
        let usage = DocumentUsage { segments: job.len(), chains: job.chains(), total: Default::default() };
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
    /// cannot be built stops the document and is returned.
    pub fn submit_ready<T>(
        &mut self,
        session: &mut Session<'_, T>,
        tag: impl Fn(usize) -> T,
    ) -> Result<(), SegmentError> {
        if self.stopped {
            return Ok(());
        }
        for (segment, req) in self.job.take_ready() {
            match self.job.prompt(session.tokenizer(), req) {
                Ok((prompt, max_new)) => {
                    self.live += 1;
                    session.submit(prompt, max_new, tag(segment));
                }
                Err(error) => {
                    self.stopped = true;
                    return Err(SegmentError { segment, error });
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
            finish: done.finish,
            prefill: done.prefill(),
            decode: done.decode(),
        };
        self.usage.total.prompt_tokens += end.prompt_tokens;
        self.usage.total.reused_tokens += end.reused;
        self.usage.total.completion_tokens += end.generated;
        self.job.finished(segment, done.text, done.finish == Finish::Length);
        Some(end)
    }

    /// Records the failure of a segment's request, which stops the document; returns
    /// whether to report it (not when the document was stopped already).
    pub fn on_failed(&mut self) -> bool {
        self.live -= 1;
        !std::mem::replace(&mut self.stopped, true)
    }

    /// What a [`Session`] update means for the document, which the caller then shows or
    /// reports in its own way: this is where a document reacts to its segments, once for
    /// every caller. A caller that cannot deliver a [`Reaction::Show`] stops the document
    /// ([`stop`](Self::stop)) and refuses the token.
    pub fn on_update<'u>(&mut self, update: Update<'u, usize>) -> Reaction<'u> {
        match update {
            Update::Delta { .. } if !self.accepts() => Reaction::Refuse,
            // A token that completes no character yet has nothing to show.
            Update::Delta { text: "", .. } => Reaction::Quiet,
            Update::Delta { tag, text } => Reaction::Show { segment: *tag, text },
            Update::Done { tag, done } => self.on_done(tag, done).map_or(Reaction::Quiet, Reaction::Ended),
            Update::Failed { tag, message } if self.on_failed() => Reaction::Failed { segment: tag, message },
            Update::Failed { .. } => Reaction::Quiet,
        }
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

/// What a document makes of one update of its segments' requests ([`DocRun::on_update`]).
#[derive(Debug)]
pub enum Reaction<'u> {
    /// New text of `segment` to deliver.
    Show { segment: usize, text: &'u str },
    /// The document has stopped: refuse the token, which cancels the segment's request.
    Refuse,
    /// A segment ended and counts: its report.
    Ended(SegmentEnd),
    /// A segment failed, which stops the document: to be reported once.
    Failed { segment: usize, message: String },
    /// Nothing to do.
    Quiet,
}

/// Why a document run failed.
#[derive(Debug, thiserror::Error)]
pub enum DocRunError {
    /// The engine failed (a read-back of its backend): it is not to be used again.
    #[error("{0:#}")]
    Engine(anyhow::Error),
    /// A segment's prompt could not be built. The engine is as before.
    #[error(transparent)]
    Prompt(SegmentError),
    /// A segment's request failed. The engine is as before.
    #[error("segment {segment}: {message}")]
    Segment { segment: usize, message: String },
}

/// Events of a document run.
pub enum DocEvent<'a> {
    Delta { segment: usize, text: &'a str },
    SegmentDone(SegmentEnd),
}

/// Translates a document on `engine` (which must not have other requests in flight):
/// the segments of every chain run side by side in its passes. The finished document is
/// returned with the translations. Earlier translations in `memory` serve as hints; the
/// new ones are added to it.
///
/// `on` returns `false` to stop (e.g. when the output it writes to has gone away). So
/// does the `cancelled` callback, which is asked between the engine's steps. Stopping
/// starts no further segment, cancels in the engine the ones waiting or running, sends no
/// more events once it has happened, and returns the document unfinished. A failed
/// segment stops it too; its error is returned once the engine is drained.
pub struct DocRunner<'a> {
    engine: &'a mut dyn Generator,
    tok: &'a Tokenizer,
    memory: &'a mut Memory,
    cancelled: Option<&'a mut dyn FnMut() -> bool>,
}

impl<'a> DocRunner<'a> {
    pub fn new(engine: &'a mut dyn Generator, tok: &'a Tokenizer, memory: &'a mut Memory) -> Self {
        DocRunner { engine, tok, memory, cancelled: None }
    }

    /// Asks `cancelled` between the engine's steps, and stops the document when it says
    /// `true`.
    pub fn cancel_when(mut self, cancelled: &'a mut dyn FnMut() -> bool) -> Self {
        self.cancelled = Some(cancelled);
        self
    }

    pub async fn run(self, job: DocJob, on: &mut dyn FnMut(DocEvent) -> bool) -> Result<DocJob, DocRunError> {
        let DocRunner { engine, tok, memory, mut cancelled } = self;
        let mut session = Session::new(engine, tok, memory);
        let mut run = DocRun::new(job);
        let mut failure: Option<DocRunError> = None;
        let mut cancelled_all = false;
        if let Err(e) = run.submit_ready(&mut session, |s| s) {
            failure = Some(DocRunError::Prompt(e));
        }
        while !run.is_over() {
            let worked = session
                .step_async(&mut |update| {
                    match run.on_update(update) {
                        Reaction::Refuse => return false,
                        Reaction::Show { segment, text } => {
                            if !on(DocEvent::Delta { segment, text }) {
                                run.stop();
                                return false;
                            }
                        }
                        Reaction::Ended(end) => {
                            if !on(DocEvent::SegmentDone(end)) {
                                run.stop();
                            }
                        }
                        Reaction::Failed { segment, message } => {
                            failure.get_or_insert(DocRunError::Segment { segment, message });
                        }
                        Reaction::Quiet => {}
                    }
                    true
                })
                .await
                .map_err(DocRunError::Engine)?;
            // (asked while there is something to cancel: after the last step it is too late,
            // and saying so would call a finished document cancelled)
            if run.accepts() && !run.is_over() && cancelled.as_mut().is_some_and(|c| c()) {
                run.stop();
            }
            if !run.accepts() && !cancelled_all {
                cancelled_all = true;
                session.cancel_where(|_| true);
            }
            if let Err(e) = run.submit_ready(&mut session, |s| s) {
                failure.get_or_insert(DocRunError::Prompt(e));
            }
            if !(worked || run.is_over() || !session.is_idle()) {
                return Err(DocRunError::Engine(anyhow::anyhow!("engine idle with segments left")));
            }
        }
        match failure {
            Some(e) => Err(e),
            None => Ok(run.into_job()),
        }
    }
}

/// [`DocRunner`] without a cancel callback, for an engine whose steps never wait (see
/// [`SyncGenerator`]): the CLI's way to translate a document.
pub fn run_document(
    engine: &mut dyn SyncGenerator,
    tok: &Tokenizer,
    job: DocJob,
    memory: &mut Memory,
    on: &mut dyn FnMut(DocEvent) -> bool,
) -> Result<DocJob> {
    Ok(poll_once(DocRunner::new(engine, tok, memory).run(job, on))?)
}

#[cfg(test)]
mod tests {
    use web_time::Instant;
    use yaxus_model::StartInfo;

    use super::*;
    use crate::document::{Chains, DocOptions};

    fn run_with_one_segment_in_flight() -> DocRun {
        let opts = DocOptions { chains: Chains::from(1), ..DocOptions::new("en") };
        let mut run = DocRun::new(DocJob::new(vec!["a".into(), "b".into()], opts));
        run.live = 1;
        run
    }

    fn finished(finish: Finish) -> Finished {
        let now = Instant::now();
        Finished {
            text: "A".into(),
            finish,
            prompt_tokens: 3,
            generated: 2,
            start: StartInfo::default(),
            submitted: now,
            first_token: Some(now),
            ended: now,
        }
    }

    #[test]
    fn text_is_shown_and_an_empty_token_is_not() {
        let mut run = run_with_one_segment_in_flight();
        assert!(matches!(
            run.on_update(Update::Delta { tag: &0, text: "x" }),
            Reaction::Show { segment: 0, text: "x" }
        ));
        assert!(matches!(run.on_update(Update::Delta { tag: &0, text: "" }), Reaction::Quiet));
    }

    #[test]
    fn a_stopped_document_refuses_text_and_does_not_count_ends() {
        let mut run = run_with_one_segment_in_flight();
        run.stop();
        assert!(matches!(run.on_update(Update::Delta { tag: &0, text: "x" }), Reaction::Refuse));
        assert!(matches!(run.on_update(Update::Done { tag: 0, done: finished(Finish::Stop) }), Reaction::Quiet));
        assert!(run.is_over(), "nothing is left in flight");
    }

    #[test]
    fn an_end_is_reported_unless_it_was_cancelled() {
        let mut run = run_with_one_segment_in_flight();
        let Reaction::Ended(end) = run.on_update(Update::Done { tag: 0, done: finished(Finish::Length) }) else {
            panic!("a finished segment ends")
        };
        assert_eq!((end.segment, end.prompt_tokens, end.generated, end.finish), (0, 3, 2, Finish::Length));
        let mut run = run_with_one_segment_in_flight();
        assert!(matches!(run.on_update(Update::Done { tag: 0, done: finished(Finish::Cancelled) }), Reaction::Quiet));
    }

    #[test]
    fn only_the_first_failure_is_reported_and_it_stops_the_document() {
        let mut run = run_with_one_segment_in_flight();
        run.live = 2;
        let fail = |run: &mut DocRun| run.on_update(Update::Failed { tag: 1, message: "boom".into() });
        assert!(matches!(fail(&mut run), Reaction::Failed { segment: 1, .. }));
        assert!(!run.accepts());
        assert!(matches!(fail(&mut run), Reaction::Quiet), "a second failure has nothing new to say");
    }
}
