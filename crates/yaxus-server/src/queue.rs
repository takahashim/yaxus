//! The bounded queue of jobs from the HTTP handlers to the inference thread, with the
//! count of the jobs waiting in it (for `/healthz`).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tokio::sync::mpsc;
use tokio::sync::mpsc::error::{TryRecvError, TrySendError};

use yaxus_runtime::scheduler::{Job, JobSource, Next};

/// Why a job was not queued.
#[derive(Debug, PartialEq, Eq)]
pub enum Rejected {
    Full,
    /// The inference thread has stopped.
    Stopped,
}

/// The handlers' end.
pub struct JobQueue {
    tx: mpsc::Sender<Job>,
    queued: Arc<AtomicUsize>,
}

/// The inference thread's end.
pub struct Jobs {
    rx: mpsc::Receiver<Job>,
    queued: Arc<AtomicUsize>,
    limit: usize,
}

/// A queue of at most `limit` jobs (at least 1).
pub fn job_queue(limit: usize) -> (JobQueue, Jobs) {
    let limit = limit.max(1);
    let (tx, rx) = mpsc::channel(limit);
    let queued = Arc::new(AtomicUsize::new(0));
    (JobQueue { tx, queued: queued.clone() }, Jobs { rx, queued, limit })
}

impl JobQueue {
    pub fn push(&self, job: Job) -> Result<(), Rejected> {
        // Counted before it can be received, so that the count never goes below zero.
        self.queued.fetch_add(1, Ordering::SeqCst);
        self.tx.try_send(job).map_err(|e| {
            self.queued.fetch_sub(1, Ordering::SeqCst);
            match e {
                TrySendError::Full(_) => Rejected::Full,
                TrySendError::Closed(_) => Rejected::Stopped,
            }
        })
    }

    /// Jobs waiting for the inference thread.
    pub fn len(&self) -> usize {
        self.queued.load(Ordering::SeqCst)
    }

    /// The inference thread has stopped.
    pub fn is_closed(&self) -> bool {
        self.tx.is_closed()
    }
}

impl Jobs {
    pub fn try_recv(&mut self) -> Result<Job, TryRecvError> {
        self.rx.try_recv().inspect(|_| self.taken())
    }

    fn taken(&self) {
        self.queued.fetch_sub(1, Ordering::SeqCst);
    }
}

impl JobSource for Jobs {
    /// The next job, waiting for one; `None` when every handle of the queue is gone.
    async fn next(&mut self) -> Option<Job> {
        self.rx.recv().await.inspect(|_| self.taken())
    }

    fn try_next(&mut self) -> Next {
        match self.try_recv() {
            Ok(job) => Next::Job(job),
            Err(TryRecvError::Empty) => Next::Empty,
            Err(TryRecvError::Disconnected) => Next::Closed,
        }
    }

    /// How many jobs the queue holds, which is also how many requests the loop keeps
    /// waiting in the engine.
    fn limit(&self) -> usize {
        self.limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::Events;
    use yaxus_runtime::scheduler::Work;

    fn job() -> Job {
        let tok = yaxus_tokenizer::testing::plamo_like(&[]);
        let req = yaxus_pfml::TranslationRequest { text: "x".into(), target_lang: "en".into(), ..Default::default() };
        let prompt = yaxus_runtime::Prompt::new(&tok, &req.validate().unwrap());
        let (events, _) = mpsc::unbounded_channel();
        Job::new(Work::Single { prompt, max_new_tokens: 1, events: Box::new(Events(events)) }, "t")
    }

    #[test]
    fn the_count_follows_the_jobs_in_the_queue() {
        let (q, mut jobs) = job_queue(1);
        q.push(job()).unwrap();
        assert_eq!(q.push(job()), Err(Rejected::Full));
        assert_eq!(q.len(), 1);
        assert!(jobs.try_recv().is_ok());
        assert_eq!(q.len(), 0);
        drop(jobs);
        assert!(q.is_closed());
        assert_eq!(q.push(job()), Err(Rejected::Stopped));
        assert_eq!(q.len(), 0);
    }
}
