//! The inference loop of a process: jobs come in from a [`JobSource`], an engine runs
//! many of them in the same passes, and the events of the API go out through [`Sink`]s.
//!
//! A document job becomes one engine request per segment; when a segment ends, the next
//! segment of its chain is submitted. What each request goes through (hints from the
//! translation memory, decoding, recording its output) is the [`Session`], and how a
//! document advances is its [`DocRun`]; this module routes their updates to the clients.
//!
//! The loop is asynchronous, so that WebGPU, which cannot be waited on, can run it; the
//! server runs it on its inference thread with `block_on`, since the steps of an engine
//! on a native backend are ready at once and only an empty queue makes it wait.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;

use futures_util::FutureExt;
use web_time::Instant;
use yaxus_api::{ApiError, DocumentEvent, DocumentUsage, ErrorCode, StreamEvent, Usage};
use yaxus_model::{Finish, Generator};
use yaxus_pfml::{PromptError, PromptErrorKind};
use yaxus_tokenizer::Tokenizer;

use crate::document::{DocJob, DocRun, SegmentError};
use crate::memory::Memory;
use crate::{Finished, Prompt, Session, Update};

/// `Send` where there are threads (not on wasm, where a page's values often are not).
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSend: Send {}
#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + ?Sized> MaybeSend for T {}
#[cfg(target_arch = "wasm32")]
pub trait MaybeSend {}
#[cfg(target_arch = "wasm32")]
impl<T: ?Sized> MaybeSend for T {}

/// Where the events of one job go.
pub trait Sink<E>: MaybeSend {
    /// Delivers `event`; `false` when nobody is listening any more.
    fn send(&self, event: E) -> bool;
    /// Nobody is listening any more (the client left).
    fn is_closed(&self) -> bool;
}

/// What a job is, and where its events (those of the API) go.
pub enum Work {
    Single { prompt: Prompt, max_new_tokens: usize, events: Box<dyn Sink<StreamEvent>> },
    Document { job: Box<DocJob>, events: Box<dyn Sink<DocumentEvent>> },
}

pub struct Job {
    pub work: Work,
    pub enqueued: Instant,
    /// Short description for the log line.
    pub label: String,
}

impl Job {
    /// The client left while the job waited.
    fn is_abandoned(&self) -> bool {
        match &self.work {
            Work::Single { events, .. } => events.is_closed(),
            Work::Document { events, .. } => events.is_closed(),
        }
    }
}

/// What a non-waiting look at a [`JobSource`] found.
pub enum Next {
    Job(Job),
    Empty,
    /// Every handle of the queue is gone and it is empty.
    Closed,
}

/// Where the jobs come from.
pub trait JobSource {
    /// The next job, waiting for one; `None` when the source is closed.
    fn next(&mut self) -> impl std::future::Future<Output = Option<Job>>;
    fn try_next(&mut self) -> Next;
    /// How many requests the loop keeps waiting in the engine before it takes no more.
    fn limit(&self) -> usize;
}

/// How [`run`] ended.
#[derive(Debug, PartialEq, Eq)]
pub enum Exit {
    /// The source is closed and nothing is left to run.
    Closed,
    /// The engine failed (a panic, or an error of its backend): it is not to be used
    /// again. Every request in flight was failed, and the source was dropped.
    Poisoned { message: String },
}

/// Whose an engine request is.
enum Owner {
    Single { events: Box<dyn Sink<StreamEvent>>, label: String, enqueued: Instant },
    Doc { key: u64, segment: usize },
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

/// The documents in progress.
#[derive(Default)]
struct Docs {
    docs: HashMap<u64, Doc>,
    next_key: u64,
    /// Documents whose chains may continue after this step.
    resubmit: Vec<u64>,
}

/// Runs jobs on `engine` until `jobs` is closed and everything has ended, or the engine
/// fails. `engine` must have no requests of its own in flight.
pub async fn run(engine: &mut dyn Generator, tok: &Tokenizer, memory: &mut Memory, mut jobs: impl JobSource) -> Exit {
    let mut session: Session<Owner> = Session::new(engine, tok, memory);
    let mut docs = Docs::default();
    loop {
        // Wait only when there is nothing to run.
        if session.is_idle() {
            match jobs.next().await {
                Some(job) => add(job, &mut session, &mut docs),
                None => return Exit::Closed,
            }
        }
        while session.waiting() < jobs.limit() {
            match jobs.try_next() {
                Next::Job(job) => add(job, &mut session, &mut docs),
                Next::Empty => break,
                Next::Closed if session.is_idle() => return Exit::Closed,
                Next::Closed => break,
            }
        }
        if session.is_idle() {
            continue;
        }
        cancel_abandoned(&mut session, &mut docs);
        let step = AssertUnwindSafe(session.step_async(&mut |u| handle(u, &mut docs))).catch_unwind().await;
        let failure = match step {
            Ok(Ok(_)) => None,
            // An `Err` of a step is a failure of the backend, not of one request.
            Ok(Err(e)) => Some(format!("{e:#}")),
            Err(_) => Some("internal error during inference".to_string()),
        };
        if let Some(message) = failure {
            tracing::error!("inference step failed: {message}");
            // The engine may be half-way through a pass (KV pages, stream state): do not
            // call it again. The caller shuts down so that it can be restarted.
            fail_all(session.abandon(), &mut docs, &message);
            tracing::error!("stopping the inference loop");
            return Exit::Poisoned { message };
        }
        for key in std::mem::take(&mut docs.resubmit) {
            submit_ready(key, &mut session, &mut docs);
        }
    }
}

/// Cancels, in the engine, the requests whose client has gone or whose document has
/// stopped: the ones still waiting too, which would otherwise start and be cancelled at
/// their first text.
fn cancel_abandoned(session: &mut Session<Owner>, docs: &mut Docs) {
    let mut gone = Vec::new();
    for (key, d) in docs.docs.iter_mut() {
        if d.events.is_closed() {
            d.run.stop();
        }
        if !d.run.accepts() {
            gone.push(*key);
        }
    }
    session.cancel_where(|owner| match owner {
        Owner::Single { events, .. } => events.is_closed(),
        Owner::Doc { key, .. } => gone.contains(key),
    });
}

/// Tells the clients of the requests in flight and of every document that they failed.
fn fail_all(owners: Vec<Owner>, docs: &mut Docs, msg: &str) {
    let error = ApiError::new(ErrorCode::Internal, msg);
    for owner in owners {
        if let Owner::Single { events, .. } = owner {
            let _ = events.send(StreamEvent::Error { error: error.clone() });
        }
    }
    for (_, d) in docs.docs.drain() {
        let _ = d.events.send(DocumentEvent::Error { error: error.clone() });
    }
}

fn add(job: Job, session: &mut Session<Owner>, docs: &mut Docs) {
    if job.is_abandoned() {
        return;
    }
    match job.work {
        Work::Single { prompt, max_new_tokens, events } => {
            let owner = Owner::Single { events, label: job.label, enqueued: job.enqueued };
            session.submit(prompt, max_new_tokens, owner);
        }
        Work::Document { job: doc, events } => {
            docs.next_key += 1;
            let key = docs.next_key;
            let d = Doc {
                run: DocRun::new(*doc),
                events,
                label: job.label,
                enqueued: job.enqueued,
                start: Instant::now(),
                first_token: None,
            };
            docs.docs.insert(key, d);
            submit_ready(key, session, docs);
        }
    }
}

/// Submits the segments of document `key` that can start, and completes the document
/// when nothing is left.
fn submit_ready(key: u64, session: &mut Session<Owner>, docs: &mut Docs) {
    let Some(d) = docs.docs.get_mut(&key) else { return };
    // A history that leaves too little room is dropped.
    if let Err(e) = d.run.submit_ready(session, |segment| Owner::Doc { key, segment }) {
        let _ = d.events.send(DocumentEvent::Error { error: segment_error(e) });
    }
    if !d.run.is_over() {
        return;
    }
    // Segments without text complete without the engine.
    if d.run.is_complete() {
        let mut usage: DocumentUsage = d.run.usage().clone();
        let ms = |t: std::time::Duration| t.as_millis() as u64;
        usage.total.queue_ms = ms(d.start - d.enqueued);
        usage.total.prefill_ms = d.first_token.map_or(0, |t| ms(t - d.start));
        usage.total.elapsed_ms = ms(d.start.elapsed());
        let t = &usage.total;
        tracing::info!(
            "{}: document of {} segments in {} chains, {} -> {} tokens ({} reused) in {} ms",
            d.label,
            usage.segments,
            usage.chains,
            t.prompt_tokens,
            t.completion_tokens,
            t.reused_tokens,
            t.elapsed_ms
        );
        let _ = d.events.send(DocumentEvent::Done { usage });
    }
    docs.docs.remove(&key);
}

/// Delivers one update to its client; returns `false` to cancel a request whose client
/// is gone.
fn handle(u: Update<'_, Owner>, docs: &mut Docs) -> bool {
    match u {
        Update::Delta { tag: Owner::Single { events, .. }, text } => {
            if text.is_empty() {
                !events.is_closed()
            } else {
                events.send(StreamEvent::Delta { text: text.to_string() })
            }
        }
        Update::Delta { tag: Owner::Doc { key, segment }, text } => {
            let Some(d) = docs.docs.get_mut(key) else { return false };
            let ok = d.run.accepts()
                && if text.is_empty() {
                    !d.events.is_closed()
                } else {
                    d.first_token.get_or_insert_with(Instant::now);
                    d.events.send(DocumentEvent::Delta { segment: *segment, text: text.to_string() })
                };
            if !ok {
                d.run.stop();
            }
            ok
        }
        Update::Done { tag: Owner::Single { events, label, enqueued }, done } => {
            let usage = usage(&done, enqueued);
            log_single(&label, &usage, &done);
            let _ = events.send(StreamEvent::Done { usage });
            true
        }
        Update::Done { tag: Owner::Doc { key, segment }, done } => {
            if let Some(d) = docs.docs.get_mut(&key) {
                if d.run.on_done(segment, done).is_some() && !d.events.send(DocumentEvent::SegmentDone { segment }) {
                    d.run.stop();
                }
                docs.resubmit.push(key);
            }
            true
        }
        Update::Failed { tag: Owner::Single { events, label, .. }, message } => {
            tracing::warn!("{label}: generation failed: {message}");
            let _ = events.send(StreamEvent::Error { error: ApiError::new(ErrorCode::Internal, message) });
            true
        }
        Update::Failed { tag: Owner::Doc { key, segment }, message } => {
            if let Some(d) = docs.docs.get_mut(&key) {
                tracing::warn!("{}: segment {segment} failed: {message}", d.label);
                if d.run.on_failed() {
                    let error = ApiError::new(ErrorCode::Internal, message).in_segment(segment);
                    let _ = d.events.send(DocumentEvent::Error { error });
                }
                docs.resubmit.push(key);
            }
            true
        }
    }
}

/// The usage report of a single request that waited in the queue since `enqueued`.
fn usage(done: &Finished, enqueued: Instant) -> Usage {
    Usage {
        prompt_tokens: done.prompt_tokens,
        reused_tokens: done.start.reused,
        completion_tokens: done.generated,
        queue_ms: (done.submitted - enqueued).as_millis() as u64,
        prefill_ms: done.prefill().as_millis() as u64,
        elapsed_ms: done.elapsed().as_millis() as u64,
    }
}

fn log_single(label: &str, usage: &Usage, done: &Finished) {
    tracing::info!(
        "{label}: {} -> {} tokens in {:.2?} ({}, queued {} ms){}",
        usage.prompt_tokens,
        usage.completion_tokens,
        done.elapsed(),
        if done.start.cached {
            "from the response cache".to_string()
        } else {
            format!("reused {} prompt tokens", done.start.reused)
        },
        usage.queue_ms,
        if done.finish == Finish::Cancelled { ", cancelled by client" } else { "" }
    );
}

/// The API error for a prompt that cannot be built.
pub fn prompt_error(e: PromptError) -> ApiError {
    let code = match e.kind {
        PromptErrorKind::Invalid => ErrorCode::InvalidRequest,
        PromptErrorKind::TooLong => ErrorCode::PromptTooLong,
    };
    ApiError::new(code, e.message)
}

/// The API error for a segment whose prompt cannot be built.
pub fn segment_error(e: SegmentError) -> ApiError {
    prompt_error(e.error).in_segment(e.segment)
}
