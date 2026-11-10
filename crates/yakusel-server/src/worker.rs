//! The single inference thread. HTTP handlers only enqueue jobs and read events.

use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use tokio::sync::mpsc;
use yakusel_api::Usage;
use yakusel_model::Generator;
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

pub fn run(
    mut model: Box<dyn Generator>,
    tok: Arc<Tokenizer>,
    mut rx: mpsc::Receiver<Job>,
    queued: Arc<AtomicUsize>,
) {
    while let Some(job) = rx.blocking_recv() {
        queued.fetch_sub(1, Ordering::SeqCst);
        if job.events.is_closed() {
            continue; // the client left while waiting
        }
        let start = Instant::now();
        let queue_ms = (start - job.enqueued).as_millis() as u64;
        let mut dec = StreamDecoder::new();
        let mut cancelled = false;
        let mut first_token: Option<Instant> = None;
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            model.generate(&job.prompt_ids, job.max_new_tokens, &[EOS_ID, TAG_ID], &mut |id| {
                first_token.get_or_insert_with(Instant::now);
                let s = dec.push(&tok, id);
                if !s.is_empty() && job.events.send(Event::Delta(s)).is_err() {
                    cancelled = true;
                    return false;
                }
                true
            })
        }));
        let elapsed = start.elapsed();
        match result {
            Ok(Ok(out)) => {
                let tail = dec.finish();
                if !tail.is_empty() {
                    let _ = job.events.send(Event::Delta(tail));
                }
                let usage = Usage {
                    prompt_tokens: job.prompt_ids.len(),
                    completion_tokens: out.len(),
                    elapsed_ms: elapsed.as_millis() as u64,
                    // Without any token (immediate stop), prompt processing was the whole time.
                    prefill_ms: first_token.map_or(elapsed, |t| t - start).as_millis() as u64,
                    queue_ms,
                };
                tracing::info!(
                    "{}: {} -> {} tokens in {:.2?} (queued {queue_ms} ms){}",
                    job.label,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    elapsed,
                    if cancelled { ", cancelled by client" } else { "" }
                );
                let _ = job.events.send(Event::Done(usage));
            }
            Ok(Err(e)) => {
                tracing::warn!("{}: generation failed: {e:#}", job.label);
                let _ = job.events.send(Event::Error(format!("{e:#}")));
            }
            Err(_) => {
                tracing::error!("{}: inference panicked", job.label);
                let _ = job.events.send(Event::Error("internal error during inference".into()));
            }
        }
    }
}
