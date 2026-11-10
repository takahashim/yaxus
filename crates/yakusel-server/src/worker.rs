//! The single inference thread. HTTP handlers only enqueue jobs and read events; this
//! thread feeds them to the engine, which runs many of them in the same passes.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tokio::sync::mpsc;
use tokio::sync::mpsc::error::TryRecvError;
use yakusel_api::Usage;
use yakusel_model::{Event as EngineEvent, Finish, Generator, ReqId, Request};
use yakusel_tokenizer::{EOS_ID, StreamDecoder, TAG_ID, Tokenizer};

pub struct Job {
    pub prompt_ids: Vec<u32>,
    pub max_new_tokens: usize,
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
}

struct Active {
    job: Job,
    dec: StreamDecoder,
    /// When the engine accepted the job.
    start: Instant,
    first: Option<Instant>,
}

pub fn run(
    mut engine: Box<dyn Generator>,
    tok: Arc<Tokenizer>,
    mut rx: mpsc::Receiver<Job>,
    queued: Arc<AtomicUsize>,
    queue_limit: usize,
) {
    let mut active: HashMap<ReqId, Active> = HashMap::new();
    let mut next_id: ReqId = 0;
    let mut add = |job: Job, engine: &mut Box<dyn Generator>, active: &mut HashMap<ReqId, Active>| {
        queued.fetch_sub(1, Ordering::SeqCst);
        if job.events.is_closed() {
            return; // the client left while waiting
        }
        next_id += 1;
        engine.submit(Request {
            id: next_id,
            prompt: job.prompt_ids.clone(),
            max_new_tokens: job.max_new_tokens,
            stop: vec![EOS_ID, TAG_ID],
        });
        active.insert(next_id, Active { job, dec: StreamDecoder::new(), start: Instant::now(), first: None });
    };
    loop {
        // Block only when there is nothing to run.
        if active.is_empty() {
            match rx.blocking_recv() {
                Some(job) => add(job, &mut engine, &mut active),
                None => return,
            }
        }
        while engine.waiting() < queue_limit {
            match rx.try_recv() {
                Ok(job) => add(job, &mut engine, &mut active),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) if active.is_empty() => return,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        if active.is_empty() {
            continue;
        }
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            engine.step(&mut |ev| handle(ev, &tok, &mut active))
        }));
        let failure = match result {
            Ok(Ok(_)) => None,
            Ok(Err(e)) => Some(format!("{e:#}")),
            Err(_) => Some("internal error during inference".to_string()),
        };
        if let Some(msg) = failure {
            tracing::error!("inference step failed: {msg}");
            for (_, a) in active.drain() {
                let _ = a.job.events.send(Event::Error(msg.clone()));
            }
        }
    }
}

/// Delivers one engine event; returns `false` to cancel a request whose client is gone.
fn handle(ev: EngineEvent, tok: &Tokenizer, active: &mut HashMap<ReqId, Active>) -> bool {
    match ev {
        EngineEvent::Token { id, token } => {
            let Some(a) = active.get_mut(&id) else { return false };
            a.first.get_or_insert_with(Instant::now);
            let s = a.dec.push(tok, token);
            if s.is_empty() {
                !a.job.events.is_closed()
            } else {
                a.job.events.send(Event::Delta(s)).is_ok()
            }
        }
        EngineEvent::Done { id, finish, generated, start } => {
            let Some(mut a) = active.remove(&id) else { return true };
            let tail = a.dec.finish();
            if !tail.is_empty() {
                let _ = a.job.events.send(Event::Delta(tail));
            }
            let elapsed = a.start.elapsed();
            let usage = Usage {
                prompt_tokens: a.job.prompt_ids.len(),
                completion_tokens: generated,
                elapsed_ms: elapsed.as_millis() as u64,
                // Without any token (immediate stop), prompt processing was the whole time.
                prefill_ms: a.first.map_or(elapsed, |t| t - a.start).as_millis() as u64,
                queue_ms: (a.start - a.job.enqueued).as_millis() as u64,
            };
            tracing::info!(
                "{}: {} -> {} tokens in {:.2?} (reused {} prompt tokens, queued {} ms){}",
                a.job.label,
                usage.prompt_tokens,
                usage.completion_tokens,
                elapsed,
                start.reused,
                usage.queue_ms,
                if finish == Finish::Cancelled { ", cancelled by client" } else { "" }
            );
            let _ = a.job.events.send(Event::Done(usage));
            true
        }
        EngineEvent::Error { id, message } => {
            if let Some(a) = active.remove(&id) {
                tracing::warn!("{}: generation failed: {message}", a.job.label);
                let _ = a.job.events.send(Event::Error(message));
            }
            true
        }
    }
}
