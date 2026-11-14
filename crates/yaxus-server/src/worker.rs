//! The single inference thread. HTTP handlers only enqueue jobs and read events; this
//! thread feeds them to the engine, which runs many of them in the same passes. The loop
//! itself is the runtime's [`scheduler`]; this module runs it on a thread and gives it
//! the events' channels.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::queue::Jobs;
use yaxus_model::Generator;
use yaxus_runtime::memory::{self, Memory};
use yaxus_runtime::scheduler::{self, Exit, Sink};
use yaxus_tokenizer::Tokenizer;

pub use yaxus_runtime::scheduler::{Job, Work};

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

/// Runs the inference loop until the queue is closed or the engine fails; then tells
/// `failed` if the server has to shut down to be restarted.
pub fn run(
    mut engine: Box<dyn Generator + Send>,
    tok: Arc<Tokenizer>,
    jobs: Jobs,
    failed: tokio::sync::oneshot::Sender<()>,
) {
    let mut memory = Memory::new(memory::DEFAULT_ENTRIES);
    if let Exit::Poisoned { .. } = pollster::block_on(scheduler::run(engine.as_mut(), &tok, &mut memory, jobs)) {
        let _ = failed.send(());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    use std::sync::atomic::Ordering;
    use yaxus_api::{ApiError, ErrorCode, StreamEvent};
    use yaxus_runtime::Prompt;

    use yaxus_model::{Event as EngineEvent, ReqId, Request, StartInfo};
    use yaxus_pfml::TranslationRequest;

    /// An engine whose every pass panics.
    #[derive(Default)]
    struct Panicking {
        aborted: Arc<std::sync::atomic::AtomicBool>,
    }

    impl Generator for Panicking {
        fn submit(&mut self, _: Request) {}
        fn step_async<'a>(
            &'a mut self,
            _: &'a mut dyn FnMut(EngineEvent) -> bool,
        ) -> yaxus_model::LocalBoxFuture<'a, anyhow::Result<bool>> {
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
        let (queue, jobs) = crate::queue::job_queue(4);
        let (events, mut out) = mpsc::unbounded_channel();
        let req = TranslationRequest { text: "x".into(), target_lang: "en".into(), ..Default::default() };
        let job = Job {
            work: Work::Single {
                prompt: Prompt::new(&tok, &req.validate().unwrap()),
                max_new_tokens: 4,
                events: Box::new(Events(events)),
            },
            enqueued: Instant::now(),
            label: "test".into(),
        };
        queue.push(job).unwrap();
        let engine = Panicking::default();
        let aborted = engine.aborted.clone();
        let (failed_tx, mut failed_rx) = tokio::sync::oneshot::channel();
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
        assert!(!aborted.load(Ordering::SeqCst), "the broken engine was called again");
    }
}
