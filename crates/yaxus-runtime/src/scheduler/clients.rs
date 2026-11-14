//! The clients of the scheduler's requests: whose each engine request is, and what the
//! updates of the [`Session`] mean for them. A single request streams to its client; a
//! document is a [`DocRun`] whose segments are requests, which reports to its client as
//! the segments go on.

use std::collections::{HashMap, HashSet};

use web_time::Instant;
use yaxus_api::{ApiError, DocumentEvent, DocumentUsage, ErrorCode, StreamEvent};

use super::report::{log_document, log_single, segment_error, usage};
use super::{Job, Sink, Work};
use crate::document::{DocRun, Reaction};
use crate::{Session, Update};

/// Whose an engine request is.
pub(super) enum Owner {
    Single { events: Box<dyn Sink<StreamEvent>>, label: String, enqueued: Instant },
    Doc { key: u64, segment: usize },
}

/// How far a document has come, after the segments that could start were submitted.
enum Progress {
    Running,
    /// Over without being complete (it failed, or its client left), with nothing in flight.
    Abandoned,
    Complete(DocumentUsage),
}

struct Doc {
    run: DocRun,
    events: Box<dyn Sink<DocumentEvent>>,
    label: String,
    enqueued: Instant,
    start: Instant,
    /// When the document's first token came.
    first_token: Option<Instant>,
}

impl Doc {
    /// Submits the segments that can start. A prompt that cannot be built is returned, to
    /// be reported to the client, and the document goes on draining what is in flight.
    fn advance(&mut self, key: u64, session: &mut Session<Owner>) -> (Option<ApiError>, Progress) {
        // A history that leaves too little room is dropped.
        let error = self.run.submit_ready(session, |segment| Owner::Doc { key, segment }).err().map(segment_error);
        let progress = if !self.run.is_over() {
            Progress::Running
        } else if self.run.is_complete() {
            // (segments without text complete without the engine)
            Progress::Complete(self.usage())
        } else {
            Progress::Abandoned
        };
        (error, progress)
    }

    /// The tokens counted as the segments ended, and the times of the whole document.
    fn usage(&self) -> DocumentUsage {
        let mut usage = self.run.usage().clone();
        let ms = |t: std::time::Duration| t.as_millis() as u64;
        usage.total.queue_ms = ms(self.start - self.enqueued);
        usage.total.prefill_ms = self.first_token.map_or(0, |t| ms(t - self.start));
        usage.total.elapsed_ms = ms(self.start.elapsed());
        usage
    }

    /// Delivers what the document made of one update of its segment; `false` refuses a
    /// token (the request is cancelled).
    fn on_update(&mut self, update: Update<'_, usize>) -> bool {
        match self.run.on_update(update) {
            Reaction::Refuse => false,
            Reaction::Show { segment, text } => {
                self.first_token.get_or_insert_with(Instant::now);
                let sent = self.events.send(DocumentEvent::Delta { segment, text: text.to_string() });
                if !sent {
                    self.run.stop();
                }
                sent
            }
            Reaction::Ended(end) => {
                if !self.events.send(DocumentEvent::SegmentDone { segment: end.segment }) {
                    self.run.stop();
                }
                true
            }
            Reaction::Failed { segment, message } => {
                tracing::warn!("{}: segment {segment} failed: {message}", self.label);
                let error = ApiError::new(ErrorCode::Internal, message).in_segment(segment);
                let _ = self.events.send(DocumentEvent::Error { error });
                true
            }
            Reaction::Quiet => true,
        }
    }
}

/// The documents in progress, and the single requests are routed by their owner.
#[derive(Default)]
pub(super) struct Clients {
    docs: HashMap<u64, Doc>,
    next_key: u64,
    /// Documents whose chains may continue after this step.
    resubmit: Vec<u64>,
}

impl Clients {
    /// Takes a job: a single request goes to the engine, a document starts its segments.
    pub(super) fn add(&mut self, job: Job, session: &mut Session<Owner>) {
        if job.is_abandoned() {
            return;
        }
        match job.work {
            Work::Single { prompt, max_new_tokens, events } => {
                let owner = Owner::Single { events, label: job.label, enqueued: job.enqueued };
                session.submit(prompt, max_new_tokens, owner);
            }
            Work::Document { job: doc, events } => {
                self.next_key += 1;
                let key = self.next_key;
                let d = Doc {
                    run: DocRun::new(*doc),
                    events,
                    label: job.label,
                    enqueued: job.enqueued,
                    start: Instant::now(),
                    first_token: None,
                };
                self.docs.insert(key, d);
                self.advance(key, session);
            }
        }
    }

    /// Delivers one update to its client; returns `false` to cancel a request whose client
    /// is gone.
    pub(super) fn handle(&mut self, update: Update<'_, Owner>) -> bool {
        match update {
            Update::Delta { tag: Owner::Single { events, .. }, text } => {
                // (a token that completes no character yet has nothing to send)
                text.is_empty() || events.send(StreamEvent::Delta { text: text.to_string() })
            }
            Update::Done { tag: Owner::Single { events, label, enqueued }, done } => {
                let usage = usage(&done, enqueued);
                log_single(&label, &usage, &done);
                let _ = events.send(StreamEvent::Done { usage });
                true
            }
            Update::Failed { tag: Owner::Single { events, label, .. }, message } => {
                tracing::warn!("{label}: generation failed: {message}");
                let _ = events.send(StreamEvent::Error { error: ApiError::new(ErrorCode::Internal, message) });
                true
            }
            Update::Delta { tag: Owner::Doc { key, segment }, text } => {
                let Some(d) = self.docs.get_mut(key) else { return false };
                d.on_update(Update::Delta { tag: segment, text })
            }
            Update::Done { tag: Owner::Doc { key, segment }, done } => {
                if let Some(d) = self.docs.get_mut(&key) {
                    d.on_update(Update::Done { tag: segment, done });
                    self.resubmit.push(key);
                }
                true
            }
            Update::Failed { tag: Owner::Doc { key, segment }, message } => {
                if let Some(d) = self.docs.get_mut(&key) {
                    d.on_update(Update::Failed { tag: segment, message });
                    self.resubmit.push(key);
                }
                true
            }
        }
    }

    /// Submits the next segments of the documents whose segments ended in the last step.
    pub(super) fn continue_documents(&mut self, session: &mut Session<Owner>) {
        for key in std::mem::take(&mut self.resubmit) {
            self.advance(key, session);
        }
    }

    /// Cancels, in the engine, the requests whose client has gone or whose document has
    /// stopped: the ones still waiting too, which would otherwise start and be cancelled
    /// at their first text.
    pub(super) fn cancel_abandoned(&mut self, session: &mut Session<Owner>) {
        let mut stopped = HashSet::new();
        for (key, d) in self.docs.iter_mut() {
            if d.events.is_closed() {
                d.run.stop();
            }
            if !d.run.accepts() {
                stopped.insert(*key);
            }
        }
        session.cancel_where(|owner| match owner {
            Owner::Single { events, .. } => events.is_closed(),
            Owner::Doc { key, .. } => stopped.contains(key),
        });
    }

    /// Tells the clients of the requests in flight, and of every document, that they failed.
    pub(super) fn fail_all(&mut self, owners: Vec<Owner>, message: &str) {
        let error = ApiError::new(ErrorCode::Internal, message);
        for owner in owners {
            if let Owner::Single { events, .. } = owner {
                let _ = events.send(StreamEvent::Error { error: error.clone() });
            }
        }
        for (_, d) in self.docs.drain() {
            let _ = d.events.send(DocumentEvent::Error { error: error.clone() });
        }
    }

    /// Submits the segments of document `key` that can start, tells its client of an error
    /// and of the end, and forgets the document once it is over.
    fn advance(&mut self, key: u64, session: &mut Session<Owner>) {
        let Some(d) = self.docs.get_mut(&key) else { return };
        let (error, progress) = d.advance(key, session);
        if let Some(error) = error {
            let _ = d.events.send(DocumentEvent::Error { error });
        }
        match progress {
            Progress::Running => {}
            Progress::Abandoned => {
                self.docs.remove(&key);
            }
            Progress::Complete(usage) => {
                log_document(&d.label, &usage);
                let _ = d.events.send(DocumentEvent::Done { usage });
                self.docs.remove(&key);
            }
        }
    }
}
