//! What the scheduler says about finished work: the usage figures of the API and the log
//! lines. Kept apart from the loop and the routing, which only call it.

use web_time::Instant;
use yaxus_api::{ApiError, DocumentUsage, ErrorCode, Usage};
use yaxus_model::Finish;
use yaxus_pfml::{PromptError, PromptErrorKind};

use crate::Finished;
use crate::document::SegmentError;

/// The usage report of a single request that waited in the queue since `enqueued`.
pub(super) fn usage(done: &Finished, enqueued: Instant) -> Usage {
    Usage {
        prompt_tokens: done.prompt_tokens,
        reused_tokens: done.start.reused,
        completion_tokens: done.generated,
        queue_ms: (done.submitted - enqueued).as_millis() as u64,
        prefill_ms: done.prefill().as_millis() as u64,
        elapsed_ms: done.elapsed().as_millis() as u64,
    }
}

pub(super) fn log_single(label: &str, usage: &Usage, done: &Finished) {
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

pub(super) fn log_document(label: &str, usage: &DocumentUsage) {
    let t = &usage.total;
    tracing::info!(
        "{label}: document of {} segments in {} chains, {} -> {} tokens ({} reused) in {} ms",
        usage.segments,
        usage.chains,
        t.prompt_tokens,
        t.completion_tokens,
        t.reused_tokens,
        t.elapsed_ms
    );
}

/// The API error for a prompt that cannot be built.
pub fn prompt_error(e: PromptError) -> ApiError {
    let code = match e.kind {
        PromptErrorKind::Invalid => ErrorCode::InvalidRequest,
        PromptErrorKind::TooLong => ErrorCode::PromptTooLong,
    };
    ApiError::new(code, e.message)
}

/// The API error for a segment whose prompt cannot be built.
pub fn segment_error(e: SegmentError) -> ApiError {
    prompt_error(e.error).in_segment(e.segment)
}
