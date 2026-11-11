//! The single inference thread. HTTP handlers only enqueue jobs and read events; this
//! thread feeds them to the engine, which runs many of them in the same passes.
//! A document job becomes one engine request per segment; when a segment finishes, the
//! next segment of its chain is submitted.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TryRecvError;
use yaxus_api::{DocumentUsage, Usage};
use yaxus_model::{Event as EngineEvent, Finish, Generator, ReqId, Request};
use yaxus_tokenizer::{EOS_ID, StreamDecoder, TAG_ID, Tokenizer};

use crate::document::{DocJob, encode_within};
use crate::memory::{self, Memory};

pub enum Work {
    /// `memo`: the request's translation memory key.
    Single {
        prompt_ids: Vec<u32>,
        max_new_tokens: usize,
        memo: u64,
    },
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

enum Owner {
    Single { events: mpsc::UnboundedSender<Event>, label: String, enqueued: Instant, prompt_len: usize },
    Doc { key: u64, segment: usize },
}

struct Active {
    owner: Owner,
    dec: StreamDecoder,
    text: String,
    tokens: Vec<u32>,
    memo: u64,
    /// When the engine accepted the request.
    start: Instant,
    first: Option<Instant>,
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

struct State {
    active: HashMap<ReqId, Active>,
    docs: HashMap<u64, Doc>,
    next_id: ReqId,
    /// Documents whose chains may continue after this step.
    resubmit: Vec<u64>,
    memory: Memory,
    /// Most prompt + generated tokens per request.
    max_context: usize,
}

pub fn run(
    mut engine: Box<dyn Generator>,
    tok: Arc<Tokenizer>,
    mut rx: mpsc::Receiver<Job>,
    queued: Arc<AtomicUsize>,
    queue_limit: usize,
    max_context: usize,
    failed: tokio::sync::oneshot::Sender<()>,
) {
    let mut st = State {
        active: HashMap::new(),
        docs: HashMap::new(),
        next_id: 0,
        resubmit: Vec::new(),
        memory: Memory::new(memory::DEFAULT_ENTRIES),
        max_context,
    };
    loop {
        // Block only when there is nothing to run.
        if st.active.is_empty() {
            match rx.blocking_recv() {
                Some(job) => add(job, &mut st, engine.as_mut(), &tok, &queued),
                None => return,
            }
        }
        while engine.waiting() < queue_limit {
            match rx.try_recv() {
                Ok(job) => add(job, &mut st, engine.as_mut(), &tok, &queued),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) if st.active.is_empty() => return,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        if st.active.is_empty() {
            continue;
        }
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| engine.step(&mut |ev| handle(ev, &tok, &mut st))));
        let panicked = result.is_err();
        let failure = match result {
            Ok(Ok(_)) => None,
            Ok(Err(e)) => Some(format!("{e:#}")),
            Err(_) => Some("internal error during inference".to_string()),
        };
        if let Some(msg) = failure {
            tracing::error!("inference step failed: {msg}");
            for (_, a) in st.active.drain() {
                if let Owner::Single { events, .. } = a.owner {
                    let _ = events.send(Event::Error(msg.clone()));
                }
            }
            for (_, d) in st.docs.drain() {
                let _ = d.events.send(Event::Error(msg.clone()));
            }
            if panicked {
                // The engine may be half-way through a pass (KV pages, stream state): do
                // not serve from it again. The server shuts down so that it can be
                // restarted; until then new requests get 503 and /healthz says so.
                tracing::error!("stopping the inference thread after a panic");
                let _ = failed.send(());
                return;
            }
            continue;
        }
        for key in std::mem::take(&mut st.resubmit) {
            submit_ready(key, &mut st, engine.as_mut(), &tok);
        }
    }
}

fn add(job: Job, st: &mut State, engine: &mut dyn Generator, tok: &Tokenizer, queued: &AtomicUsize) {
    queued.fetch_sub(1, Ordering::SeqCst);
    if job.events.is_closed() {
        return; // the client left while waiting
    }
    match job.work {
        Work::Single { prompt_ids, max_new_tokens, memo } => {
            st.next_id += 1;
            let prompt_len = prompt_ids.len();
            let mut req = Request::new(st.next_id, prompt_ids, max_new_tokens, vec![EOS_ID, TAG_ID]);
            req.hint = st.memory.get(memo);
            engine.submit(req);
            st.active.insert(
                st.next_id,
                Active {
                    owner: Owner::Single { events: job.events, label: job.label, enqueued: job.enqueued, prompt_len },
                    dec: StreamDecoder::new(),
                    text: String::new(),
                    tokens: Vec::new(),
                    memo,
                    start: Instant::now(),
                    first: None,
                },
            );
        }
        Work::Document(doc) => {
            st.next_id += 1;
            let key = st.next_id;
            let usage = DocumentUsage { segments: doc.segments.len(), chains: doc.chains(), ..Default::default() };
            st.docs.insert(
                key,
                Doc {
                    doc: *doc,
                    events: job.events,
                    label: job.label,
                    enqueued: job.enqueued,
                    start: Instant::now(),
                    usage,
                    live: 0,
                    stopped: false,
                },
            );
            submit_ready(key, st, engine, tok);
        }
    }
}

/// Submits the segments of document `key` that can start, and completes the document
/// when nothing is left.
fn submit_ready(key: u64, st: &mut State, engine: &mut dyn Generator, tok: &Tokenizer) {
    let Some(d) = st.docs.get_mut(&key) else { return };
    if !d.stopped {
        for (segment, mut req) in d.doc.ready() {
            // A long history may not fit with the segment: it is dropped then.
            let (ids, max_new) = match encode_within(tok, &mut req, st.max_context, d.doc.opts.max_new_tokens) {
                Ok(x) => x,
                Err(e) => {
                    let _ = d.events.send(Event::Error(format!("segment {segment}: {e}")));
                    d.stopped = true;
                    break;
                }
            };
            st.next_id += 1;
            d.usage.prompt_tokens += ids.len();
            d.live += 1;
            let memo = memory::key(&req);
            let mut r = Request::new(st.next_id, ids, max_new, vec![EOS_ID, TAG_ID]);
            r.hint = st.memory.get(memo);
            engine.submit(r);
            st.active.insert(
                st.next_id,
                Active {
                    owner: Owner::Doc { key, segment },
                    dec: StreamDecoder::new(),
                    text: String::new(),
                    tokens: Vec::new(),
                    memo,
                    start: Instant::now(),
                    first: None,
                },
            );
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
            st.docs.remove(&key);
            return;
        }
    }
    if d.stopped && d.live == 0 {
        st.docs.remove(&key);
    }
}

/// Delivers one engine event; returns `false` to cancel a request whose client is gone.
fn handle(ev: EngineEvent, tok: &Tokenizer, st: &mut State) -> bool {
    match ev {
        EngineEvent::Token { id, token } => {
            let Some(a) = st.active.get_mut(&id) else { return false };
            a.first.get_or_insert_with(Instant::now);
            a.tokens.push(token);
            let s = a.dec.push(tok, token);
            a.text += &s;
            match &a.owner {
                Owner::Single { events, .. } => {
                    if s.is_empty() {
                        !events.is_closed()
                    } else {
                        events.send(Event::Delta(s)).is_ok()
                    }
                }
                Owner::Doc { key, segment } => {
                    let Some(d) = st.docs.get_mut(key) else { return false };
                    let ok = !d.stopped
                        && if s.is_empty() {
                            !d.events.is_closed()
                        } else {
                            d.events.send(Event::SegmentDelta(*segment, s)).is_ok()
                        };
                    if !ok {
                        d.stopped = true;
                    }
                    ok
                }
            }
        }
        EngineEvent::Done { id, finish, generated, start } => {
            let Some(mut a) = st.active.remove(&id) else { return true };
            if finish == Finish::Stop {
                st.memory.put(a.memo, std::mem::take(&mut a.tokens));
            }
            let tail = a.dec.finish();
            a.text += &tail;
            let elapsed = a.start.elapsed();
            match a.owner {
                Owner::Single { events, label, enqueued, prompt_len } => {
                    if !tail.is_empty() {
                        let _ = events.send(Event::Delta(tail));
                    }
                    let usage = Usage {
                        prompt_tokens: prompt_len,
                        completion_tokens: generated,
                        elapsed_ms: elapsed.as_millis() as u64,
                        // Without any token (immediate stop), prompt processing was the whole time.
                        prefill_ms: a.first.map_or(elapsed, |t| t - a.start).as_millis() as u64,
                        queue_ms: (a.start - enqueued).as_millis() as u64,
                    };
                    tracing::info!(
                        "{label}: {} -> {} tokens in {elapsed:.2?} ({}, queued {} ms){}",
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        if start.cached {
                            "from the response cache".to_string()
                        } else {
                            format!("reused {} prompt tokens", start.reused)
                        },
                        usage.queue_ms,
                        if finish == Finish::Cancelled { ", cancelled by client" } else { "" }
                    );
                    let _ = events.send(Event::Done(usage));
                }
                Owner::Doc { key, segment } => {
                    if let Some(d) = st.docs.get_mut(&key) {
                        d.live -= 1;
                        if !d.stopped {
                            if !tail.is_empty() {
                                let _ = d.events.send(Event::SegmentDelta(segment, tail));
                            }
                            d.usage.completion_tokens += generated;
                            d.usage.reused_tokens += start.reused;
                            d.doc.finished(segment, a.text);
                            let _ = d.events.send(Event::SegmentEnd(segment));
                        }
                        st.resubmit.push(key);
                    }
                }
            }
            true
        }
        EngineEvent::Error { id, message } => {
            if let Some(a) = st.active.remove(&id) {
                match a.owner {
                    Owner::Single { events, label, .. } => {
                        tracing::warn!("{label}: generation failed: {message}");
                        let _ = events.send(Event::Error(message));
                    }
                    Owner::Doc { key, segment } => {
                        if let Some(d) = st.docs.get_mut(&key) {
                            tracing::warn!("{}: segment {segment} failed: {message}", d.label);
                            d.live -= 1;
                            if !d.stopped {
                                let _ = d.events.send(Event::Error(format!("segment {segment}: {message}")));
                            }
                            d.stopped = true;
                            st.resubmit.push(key);
                        }
                    }
                }
            }
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yaxus_model::StartInfo;

    /// An engine whose every pass panics.
    struct Panicking;

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
        use yaxus_tokenizer::{BOS_ID, TokenKind};
        let mut tokens: Vec<String> = (0..=TAG_ID)
            .map(|id| match id {
                BOS_ID => "<|plamo:bos|>".into(),
                TAG_ID => "<|plamo:tag|>".into(),
                _ => format!("<|plamo:c{id}|>"),
            })
            .collect();
        let mut kinds = vec![TokenKind::Control; tokens.len()];
        tokens.extend((0..=255u8).map(|b| format!("<0x{b:02X}>")));
        kinds.extend([TokenKind::Byte; 256]);
        let scores = vec![0.0; tokens.len()];
        Tokenizer::from_vocab(tokens, kinds, scores).unwrap()
    }

    #[test]
    fn a_panic_fails_the_requests_and_stops_the_worker() {
        let (jobs, rx) = mpsc::channel(4);
        let (events, mut out) = mpsc::unbounded_channel();
        let job = Job {
            work: Work::Single { prompt_ids: vec![1, 2, 3], max_new_tokens: 4, memo: 0 },
            enqueued: Instant::now(),
            events,
            label: "test".into(),
        };
        let queued = Arc::new(AtomicUsize::new(1));
        jobs.try_send(job).unwrap();
        let tok = Arc::new(tokenizer());
        let (failed_tx, mut failed_rx) = tokio::sync::oneshot::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            run(Box::new(Panicking), tok, rx, queued, 4, 8192, failed_tx);
            let _ = done_tx.send(());
        });
        // The worker returns instead of serving from the broken engine (a timeout rather
        // than join(), so that a regression fails instead of hanging the test run).
        assert!(done_rx.recv_timeout(std::time::Duration::from_secs(30)).is_ok(), "the worker kept running");
        assert!(matches!(out.try_recv(), Ok(Event::Error(m)) if m.contains("internal error")));
        assert!(jobs.is_closed());
        assert!(failed_rx.try_recv().is_ok(), "the server was not told to shut down");
    }
}
