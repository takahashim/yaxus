//! The single inference thread. HTTP handlers only enqueue jobs and read events; this
//! thread feeds them to the engine, which runs many of them in the same passes.
//! A document job becomes one engine request per segment; when a segment finishes, the
//! next segment of its chain is submitted. What each request goes through (hints from
//! the translation memory, decoding, recording its output) is the runtime's [`Session`];
//! this module only routes its updates to the clients.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TryRecvError;
use yaxus_api::{DocumentUsage, Usage};
use yaxus_model::{Finish, Generator};
use yaxus_runtime::document::DocJob;
use yaxus_runtime::memory::{self, Memory};
use yaxus_runtime::{Finished, Prompt, Session, Update};
use yaxus_tokenizer::Tokenizer;

pub enum Work {
    Single { prompt: Prompt, max_new_tokens: usize },
    Document(Box<DocJob>),
}

pub struct Job {
    pub work: Work,
    pub enqueued: Instant,
    pub events: mpsc::UnboundedSender<Event>,
    /// Short description for the log line.
    pub label: String,
}

#[derive(Debug)]
pub enum Event {
    Delta(String),
    Done(Usage),
    Error(String),
    /// Document jobs: text of one segment.
    SegmentDelta(usize, String),
    SegmentEnd(usize),
    DocumentDone(DocumentUsage),
}

/// Whose an engine request is.
enum Owner {
    Single { events: mpsc::UnboundedSender<Event>, label: String, enqueued: Instant },
    Doc { key: u64, segment: usize },
}

struct Doc {
    doc: DocJob,
    events: mpsc::UnboundedSender<Event>,
    label: String,
    enqueued: Instant,
    start: Instant,
    usage: DocumentUsage,
    /// Segments submitted and not finished.
    live: usize,
    /// The client left or a segment failed: no more segments are submitted.
    stopped: bool,
}

/// The documents in progress.
#[derive(Default)]
struct Docs {
    docs: HashMap<u64, Doc>,
    next_key: u64,
    /// Documents whose chains may continue after this step.
    resubmit: Vec<u64>,
}

pub fn run(
    mut engine: Box<dyn Generator>,
    tok: Arc<Tokenizer>,
    mut rx: mpsc::Receiver<Job>,
    queued: Arc<AtomicUsize>,
    queue_limit: usize,
) {
    let mut memory = Memory::new(memory::DEFAULT_ENTRIES);
    let mut session: Session<Owner> = Session::new(engine.as_mut(), &tok, &mut memory);
    let mut docs = Docs::default();
    loop {
        // Block only when there is nothing to run.
        if session.is_idle() {
            match rx.blocking_recv() {
                Some(job) => add(job, &mut session, &mut docs, &queued),
                None => return,
            }
        }
        while session.waiting() < queue_limit {
            match rx.try_recv() {
                Ok(job) => add(job, &mut session, &mut docs, &queued),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) if session.is_idle() => return,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        if session.is_idle() {
            continue;
        }
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| session.step(&mut |u| handle(u, &mut docs))));
        let failure = match result {
            Ok(Ok(_)) => None,
            Ok(Err(e)) => Some(format!("{e:#}")),
            Err(_) => Some("internal error during inference".to_string()),
        };
        if let Some(msg) = failure {
            tracing::error!("inference step failed: {msg}");
            for owner in session.abort_all() {
                if let Owner::Single { events, .. } = owner {
                    let _ = events.send(Event::Error(msg.clone()));
                }
            }
            for (_, d) in docs.docs.drain() {
                let _ = d.events.send(Event::Error(msg.clone()));
            }
            continue;
        }
        for key in std::mem::take(&mut docs.resubmit) {
            submit_ready(key, &mut session, &mut docs);
        }
    }
}

fn add(job: Job, session: &mut Session<Owner>, docs: &mut Docs, queued: &AtomicUsize) {
    queued.fetch_sub(1, Ordering::SeqCst);
    if job.events.is_closed() {
        return; // the client left while waiting
    }
    match job.work {
        Work::Single { prompt, max_new_tokens } => {
            let owner = Owner::Single { events: job.events, label: job.label, enqueued: job.enqueued };
            session.submit(prompt, max_new_tokens, owner);
        }
        Work::Document(doc) => {
            docs.next_key += 1;
            let key = docs.next_key;
            let usage = DocumentUsage { segments: doc.len(), chains: doc.chains(), ..Default::default() };
            let d = Doc {
                doc: *doc,
                events: job.events,
                label: job.label,
                enqueued: job.enqueued,
                start: Instant::now(),
                usage,
                live: 0,
                stopped: false,
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
    if !d.stopped {
        for (segment, req) in d.doc.ready() {
            let prompt = match Prompt::new(session.tokenizer(), &req) {
                Ok(p) => p,
                Err(e) => {
                    let _ = d.events.send(Event::Error(format!("segment {segment}: {e}")));
                    d.stopped = true;
                    break;
                }
            };
            d.usage.prompt_tokens += prompt.len();
            d.live += 1;
            session.submit(prompt, d.doc.max_new_tokens(), Owner::Doc { key, segment });
        }
        // Segments without text complete without the engine.
        if d.doc.is_done() {
            d.usage.elapsed_ms = d.start.elapsed().as_millis() as u64;
            d.usage.queue_ms = (d.start - d.enqueued).as_millis() as u64;
            tracing::info!(
                "{}: document of {} segments in {} chains, {} -> {} tokens ({} reused) in {} ms",
                d.label,
                d.usage.segments,
                d.usage.chains,
                d.usage.prompt_tokens,
                d.usage.completion_tokens,
                d.usage.reused_tokens,
                d.usage.elapsed_ms
            );
            let _ = d.events.send(Event::DocumentDone(d.usage.clone()));
            docs.docs.remove(&key);
            return;
        }
    }
    if d.stopped && d.live == 0 {
        docs.docs.remove(&key);
    }
}

/// Delivers one update to its client; returns `false` to cancel a request whose client
/// is gone.
fn handle(u: Update<'_, Owner>, docs: &mut Docs) -> bool {
    match u {
        Update::Delta { tag: Owner::Single { events, .. }, text } => {
            if text.is_empty() {
                !events.is_closed()
            } else {
                events.send(Event::Delta(text.to_string())).is_ok()
            }
        }
        Update::Delta { tag: Owner::Doc { key, segment }, text } => {
            let Some(d) = docs.docs.get_mut(key) else { return false };
            let ok = !d.stopped
                && if text.is_empty() {
                    !d.events.is_closed()
                } else {
                    d.events.send(Event::SegmentDelta(*segment, text.to_string())).is_ok()
                };
            if !ok {
                d.stopped = true;
            }
            ok
        }
        Update::Done { tag: Owner::Single { events, label, enqueued }, done } => {
            let usage = usage(&done, enqueued);
            log_single(&label, &usage, &done);
            let _ = events.send(Event::Done(usage));
            true
        }
        Update::Done { tag: Owner::Doc { key, segment }, done } => {
            if let Some(d) = docs.docs.get_mut(&key) {
                d.live -= 1;
                if !d.stopped {
                    d.usage.completion_tokens += done.generated;
                    d.usage.reused_tokens += done.start.reused;
                    d.doc.finished(segment, done.text);
                    let _ = d.events.send(Event::SegmentEnd(segment));
                }
                docs.resubmit.push(key);
            }
            true
        }
        Update::Failed { tag: Owner::Single { events, label, .. }, message } => {
            tracing::warn!("{label}: generation failed: {message}");
            let _ = events.send(Event::Error(message));
            true
        }
        Update::Failed { tag: Owner::Doc { key, segment }, message } => {
            if let Some(d) = docs.docs.get_mut(&key) {
                tracing::warn!("{}: segment {segment} failed: {message}", d.label);
                d.live -= 1;
                if !d.stopped {
                    let _ = d.events.send(Event::Error(format!("segment {segment}: {message}")));
                }
                d.stopped = true;
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
        completion_tokens: done.generated,
        elapsed_ms: done.elapsed().as_millis() as u64,
        prefill_ms: done.prefill().as_millis() as u64,
        queue_ms: (done.submitted - enqueued).as_millis() as u64,
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
