//! The single inference thread. HTTP handlers only enqueue jobs and read events; this
//! thread feeds them to the engine, which runs many of them in the same passes.
//! A document job becomes one engine request per segment; when a segment finishes, the
//! next segment of its chain is submitted. What each request goes through (hints from
//! the translation memory, decoding, recording its output) is the runtime's [`Session`],
//! and how a document advances is its [`DocRun`]; this module only routes their updates
//! to the clients.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TryRecvError;
use yaxus_api::{DocumentUsage, Usage};
use yaxus_model::{Finish, Generator};
use yaxus_runtime::document::{DocJob, DocRun};
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
    run: DocRun,
    events: mpsc::UnboundedSender<Event>,
    label: String,
    enqueued: Instant,
    start: Instant,
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
    failed: tokio::sync::oneshot::Sender<()>,
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
        let panicked = result.is_err();
        let failure = match result {
            Ok(Ok(_)) => None,
            Ok(Err(e)) => Some(format!("{e:#}")),
            Err(_) => Some("internal error during inference".to_string()),
        };
        if let Some(msg) = failure {
            tracing::error!("inference step failed: {msg}");
            if panicked {
                // The engine may be half-way through a pass (KV pages, stream state): do
                // not call it again. The server shuts down so that it can be restarted;
                // until then new requests get 503 and /healthz says so.
                fail_all(session.abandon(), &mut docs, &msg);
                tracing::error!("stopping the inference thread after a panic");
                let _ = failed.send(());
                return;
            }
            fail_all(session.abort_all(), &mut docs, &msg);
            continue;
        }
        for key in std::mem::take(&mut docs.resubmit) {
            submit_ready(key, &mut session, &mut docs);
        }
    }
}

/// Tells the clients of the requests in flight and of every document that they failed.
fn fail_all(owners: Vec<Owner>, docs: &mut Docs, msg: &str) {
    for owner in owners {
        if let Owner::Single { events, .. } = owner {
            let _ = events.send(Event::Error(msg.to_string()));
        }
    }
    for (_, d) in docs.docs.drain() {
        let _ = d.events.send(Event::Error(msg.to_string()));
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
            let d = Doc {
                run: DocRun::new(*doc),
                events: job.events,
                label: job.label,
                enqueued: job.enqueued,
                start: Instant::now(),
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
        let _ = d.events.send(Event::Error(e.0));
    }
    if !d.run.is_over() {
        return;
    }
    // Segments without text complete without the engine.
    if d.run.is_complete() {
        let usage = DocumentUsage {
            elapsed_ms: d.start.elapsed().as_millis() as u64,
            queue_ms: (d.start - d.enqueued).as_millis() as u64,
            ..d.run.usage().clone()
        };
        tracing::info!(
            "{}: document of {} segments in {} chains, {} -> {} tokens ({} reused) in {} ms",
            d.label,
            usage.segments,
            usage.chains,
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.reused_tokens,
            usage.elapsed_ms
        );
        let _ = d.events.send(Event::DocumentDone(usage));
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
                events.send(Event::Delta(text.to_string())).is_ok()
            }
        }
        Update::Delta { tag: Owner::Doc { key, segment }, text } => {
            let Some(d) = docs.docs.get_mut(key) else { return false };
            let ok = d.run.accepts()
                && if text.is_empty() {
                    !d.events.is_closed()
                } else {
                    d.events.send(Event::SegmentDelta(*segment, text.to_string())).is_ok()
                };
            if !ok {
                d.run.stop();
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
                if d.run.on_done(segment, done).is_some() && d.events.send(Event::SegmentEnd(segment)).is_err() {
                    d.run.stop();
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
                if let Some(e) = d.run.on_failed(segment, &message) {
                    let _ = d.events.send(Event::Error(e));
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

#[cfg(test)]
mod tests {
    use super::*;
    use yaxus_model::{Event as EngineEvent, ReqId, Request, StartInfo};
    use yaxus_pfml::TranslationRequest;

    /// An engine whose every pass panics.
    #[derive(Default)]
    struct Panicking {
        aborted: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Generator for Panicking {
        fn submit(&mut self, _: Request) {}
        fn step(&mut self, _: &mut dyn FnMut(EngineEvent) -> bool) -> anyhow::Result<bool> {
            panic!("boom");
        }
        fn fresh_id(&mut self) -> ReqId {
            0
        }
        fn waiting(&self) -> usize {
            0
        }
        fn active(&self) -> usize {
            1
        }
        fn abort_all(&mut self) {
            self.aborted.store(true, Ordering::SeqCst);
        }
        fn backend_name(&self) -> String {
            "panicking".into()
        }
        fn weight_bytes(&self) -> usize {
            0
        }
        fn stats(&self) -> Option<String> {
            None
        }
        fn last_start(&self) -> StartInfo {
            StartInfo::default()
        }
    }

    fn tokenizer() -> Tokenizer {
        yaxus_tokenizer::testing::plamo_like(&[])
    }

    #[test]
    fn a_panic_fails_the_requests_and_stops_the_worker() {
        let tok = Arc::new(tokenizer());
        let (jobs, rx) = mpsc::channel(4);
        let (events, mut out) = mpsc::unbounded_channel();
        let req = TranslationRequest { text: "x".into(), target_lang: "en".into(), ..Default::default() };
        let job = Job {
            work: Work::Single { prompt: Prompt::new(&tok, &req.validate().unwrap()), max_new_tokens: 4 },
            enqueued: Instant::now(),
            events,
            label: "test".into(),
        };
        let queued = Arc::new(AtomicUsize::new(1));
        jobs.try_send(job).unwrap();
        let engine = Panicking::default();
        let aborted = engine.aborted.clone();
        let (failed_tx, mut failed_rx) = tokio::sync::oneshot::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            run(Box::new(engine), tok, rx, queued, 4, failed_tx);
            let _ = done_tx.send(());
        });
        // The worker returns instead of serving from the broken engine (a timeout rather
        // than join(), so that a regression fails instead of hanging the test run).
        assert!(done_rx.recv_timeout(std::time::Duration::from_secs(30)).is_ok(), "the worker kept running");
        assert!(matches!(out.try_recv(), Ok(Event::Error(m)) if m.contains("internal error")));
        assert!(jobs.is_closed());
        assert!(failed_rx.try_recv().is_ok(), "the server was not told to shut down");
        assert!(!aborted.load(Ordering::SeqCst), "the broken engine was called again");
    }
}
