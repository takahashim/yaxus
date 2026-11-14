//! The single inference thread. HTTP handlers only enqueue jobs and read events; this
//! thread feeds them to the engine, which runs many of them in the same passes. The loop
//! itself is the runtime's [`scheduler`]; this module runs it on a thread and gives it
//! the events' channels.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::{mpsc, oneshot};

use crate::queue::{JobQueue, Jobs, job_queue};
use yaxus_model::Generator;
use yaxus_runtime::memory::{self, Memory};
use yaxus_runtime::scheduler::{self, Exit, Sink};
use yaxus_tokenizer::Tokenizer;

/// The sending end of a client's event channel.
pub struct Events<E>(pub mpsc::UnboundedSender<E>);

impl<E: Send> Sink<E> for Events<E> {
    fn send(&self, event: E) -> bool {
        self.0.send(event).is_ok()
    }
    fn is_closed(&self) -> bool {
        self.0.is_closed()
    }
}

/// The inference thread's report that its engine failed, and the server has to shut
/// down to be restarted.
pub struct Failure {
    rx: oneshot::Receiver<()>,
    reported: Arc<AtomicBool>,
}

impl Failure {
    fn new() -> (oneshot::Sender<()>, Self) {
        let (tx, rx) = oneshot::channel();
        (tx, Failure { rx, reported: Arc::default() })
    }

    /// Whether the failure has been reported, readable after [`wait`](Self::wait) has
    /// taken the report.
    pub fn reported(&self) -> Arc<AtomicBool> {
        self.reported.clone()
    }

    /// Waits for the report: `false` when the thread ended normally instead.
    pub async fn wait(self) -> bool {
        // (an Err means the thread ended normally and dropped the sender)
        let failed = self.rx.await.is_ok();
        self.reported.store(failed, Ordering::SeqCst);
        failed
    }
}

/// Starts the inference thread on `engine`, with a queue of `queue_limit` jobs.
pub fn spawn(
    engine: Box<dyn Generator + Send>,
    tok: Arc<Tokenizer>,
    queue_limit: usize,
) -> std::io::Result<(JobQueue, Failure)> {
    let (queue, jobs) = job_queue(queue_limit);
    let (failed, failure) = Failure::new();
    std::thread::Builder::new().name("inference".into()).spawn(move || run(engine, tok, jobs, failed))?;
    Ok((queue, failure))
}

/// Runs the inference loop until the queue is closed or the engine fails; then tells
/// `failed` if the server has to shut down to be restarted.
fn run(mut engine: Box<dyn Generator + Send>, tok: Arc<Tokenizer>, jobs: Jobs, failed: oneshot::Sender<()>) {
    let mut memory = Memory::new(memory::DEFAULT_ENTRIES);
    if let Exit::Poisoned { .. } = pollster::block_on(scheduler::run(engine.as_mut(), &tok, &mut memory, jobs)) {
        let _ = failed.send(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::Ordering;
    use yaxus_api::{ApiError, ErrorCode, StreamEvent};
    use yaxus_runtime::Prompt;
    use yaxus_runtime::scheduler::{Job, Work};

    use yaxus_model::{Event as EngineEvent, ReqId, Request, StartInfo};
    use yaxus_pfml::TranslationRequest;

    /// An engine whose every pass panics, counting the passes.
    #[derive(Default)]
    struct Panicking {
        steps: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Generator for Panicking {
        fn submit(&mut self, _: Request) {}
        fn step_async<'a>(
            &'a mut self,
            _: &'a mut dyn FnMut(EngineEvent) -> bool,
        ) -> yaxus_model::LocalBoxFuture<'a, anyhow::Result<bool>> {
            self.steps.fetch_add(1, Ordering::SeqCst);
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
        yaxus_tokenizer::testing::plamo_like(&[])
    }

    #[test]
    fn a_panic_fails_the_requests_and_stops_the_worker() {
        let tok = Arc::new(tokenizer());
        let (queue, jobs) = job_queue(4);
        let (events, mut out) = mpsc::unbounded_channel();
        let req = TranslationRequest { text: "x".into(), target_lang: "en".into(), ..Default::default() };
        let work = Work::Single {
            prompt: Prompt::new(&tok, &req.validate().unwrap()),
            max_new_tokens: 4,
            events: Box::new(Events(events)),
        };
        let job = Job::new(work, "test");
        queue.push(job).unwrap();
        let engine = Panicking::default();
        let steps = engine.steps.clone();
        let (failed_tx, mut failed_rx) = oneshot::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            run(Box::new(engine), tok, jobs, failed_tx);
            let _ = done_tx.send(());
        });
        // The worker returns instead of serving from the broken engine (a timeout rather
        // than join(), so that a regression fails instead of hanging the test run).
        assert!(done_rx.recv_timeout(std::time::Duration::from_secs(30)).is_ok(), "the worker kept running");
        assert!(
            matches!(out.try_recv(), Ok(StreamEvent::Error { error: ApiError { code: ErrorCode::Internal, message: m, .. } }) if m.contains("internal error"))
        );
        assert!(queue.is_closed());
        assert!(failed_rx.try_recv().is_ok(), "the server was not told to shut down");
        assert_eq!(steps.load(Ordering::SeqCst), 1, "the broken engine was called again");
    }
}
