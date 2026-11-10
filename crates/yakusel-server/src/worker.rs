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
use yakusel_api::{DocumentUsage, Usage};
use yakusel_model::{Event as EngineEvent, Finish, Generator, ReqId, Request};
use yakusel_tokenizer::{EOS_ID, StreamDecoder, TAG_ID, Tokenizer};

use crate::document::DocJob;

pub enum Work {
    Single { prompt_ids: Vec<u32>, max_new_tokens: usize },
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
}

pub fn run(
    mut engine: Box<dyn Generator>,
    tok: Arc<Tokenizer>,
    mut rx: mpsc::Receiver<Job>,
    queued: Arc<AtomicUsize>,
    queue_limit: usize,
) {
    let mut st = State { active: HashMap::new(), docs: HashMap::new(), next_id: 0, resubmit: Vec::new() };
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
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            engine.step(&mut |ev| handle(ev, &tok, &mut st))
        }));
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
        Work::Single { prompt_ids, max_new_tokens } => {
            st.next_id += 1;
            let prompt_len = prompt_ids.len();
            engine.submit(Request { id: st.next_id, prompt: prompt_ids, max_new_tokens, stop: vec![EOS_ID, TAG_ID] });
            st.active.insert(
                st.next_id,
                Active {
                    owner: Owner::Single { events: job.events, label: job.label, enqueued: job.enqueued, prompt_len },
                    dec: StreamDecoder::new(),
                    text: String::new(),
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
        for (segment, req) in d.doc.ready() {
            let prompt = match req.to_prompt() {
                Ok(p) => p,
                Err(e) => {
                    let _ = d.events.send(Event::Error(format!("segment {segment}: {}", e.0)));
                    d.stopped = true;
                    break;
                }
            };
            let ids = tok.encode(&prompt, false);
            st.next_id += 1;
            d.usage.prompt_tokens += ids.len();
            d.live += 1;
            let max_new = d.doc.opts.max_new_tokens;
            engine.submit(Request { id: st.next_id, prompt: ids, max_new_tokens: max_new, stop: vec![EOS_ID, TAG_ID] });
            st.active.insert(
                st.next_id,
                Active {
                    owner: Owner::Doc { key, segment },
                    dec: StreamDecoder::new(),
                    text: String::new(),
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
            let s = a.dec.push(tok, token);
            a.text += &s;
            match &a.owner {
                Owner::Single { events, .. } => {
                    if s.is_empty() { !events.is_closed() } else { events.send(Event::Delta(s)).is_ok() }
                }
                Owner::Doc { key, segment } => {
                    let Some(d) = st.docs.get_mut(key) else { return false };
                    let ok = !d.stopped
                        && if s.is_empty() { !d.events.is_closed() } else { d.events.send(Event::SegmentDelta(*segment, s)).is_ok() };
                    if !ok {
                        d.stopped = true;
                    }
                    ok
                }
            }
        }
        EngineEvent::Done { id, finish, generated, start } => {
            let Some(mut a) = st.active.remove(&id) else { return true };
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
                        "{label}: {} -> {} tokens in {elapsed:.2?} (reused {} prompt tokens, queued {} ms){}",
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        start.reused,
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
