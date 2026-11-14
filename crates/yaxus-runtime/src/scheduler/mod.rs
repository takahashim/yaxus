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

use std::panic::AssertUnwindSafe;

use futures_util::FutureExt;
use web_time::Instant;
use yaxus_api::{DocumentEvent, StreamEvent};
use yaxus_model::Generator;
use yaxus_tokenizer::Tokenizer;

use crate::Prompt;
use crate::Session;
use crate::document::DocJob;
use crate::memory::Memory;

mod clients;
mod report;

use clients::{Clients, Owner};
pub use report::prompt_error;

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
    /// `work`, enqueued now.
    pub fn new(work: Work, label: impl Into<String>) -> Self {
        Job { work, enqueued: Instant::now(), label: label.into() }
    }

    /// The client left while the job waited.
    pub(super) fn is_abandoned(&self) -> bool {
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

/// Runs jobs on `engine` until `jobs` is closed and everything has ended, or the engine
/// fails. `engine` must have no requests of its own in flight.
pub async fn run(engine: &mut dyn Generator, tok: &Tokenizer, memory: &mut Memory, mut jobs: impl JobSource) -> Exit {
    let mut session: Session<Owner> = Session::new(engine, tok, memory);
    let mut clients = Clients::default();
    loop {
        // Wait only when there is nothing to run.
        if session.is_idle() {
            match jobs.next().await {
                Some(job) => clients.add(job, &mut session),
                None => return Exit::Closed,
            }
        }
        while session.waiting() < jobs.limit() {
            match jobs.try_next() {
                Next::Job(job) => clients.add(job, &mut session),
                Next::Empty => break,
                Next::Closed if session.is_idle() => return Exit::Closed,
                Next::Closed => break,
            }
        }
        if session.is_idle() {
            continue;
        }
        clients.cancel_abandoned(&mut session);
        let step = AssertUnwindSafe(session.step_async(&mut |u| clients.handle(u))).catch_unwind().await;
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
            clients.fail_all(session.abandon(), &message);
            tracing::error!("stopping the inference loop");
            return Exit::Poisoned { message };
        }
        clients.continue_documents(&mut session);
    }
}
