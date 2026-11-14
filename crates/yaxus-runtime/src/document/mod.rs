//! Document translation: segments are split into chains of consecutive segments
//! (`ChainSchedule`). Within a chain they are translated in
//! order, each with the earlier ones of the chain as history (see [`HistoryWindow`]); the
//! chains run in parallel on the engine ([`DocRun`], [`run_document`]).

use yaxus_api::DocumentRequest;
use yaxus_model::DEFAULT_MAX_POS;
use yaxus_pfml::{PromptError, TranslationRequest, is_blank};
use yaxus_tokenizer::Tokenizer;

use crate::prompt::Prompt;

mod history;
mod run;
mod schedule;

pub use history::{ChainHistories, HistoryRules};
pub use run::{DocEvent, DocRun, DocRunError, DocRunner, Reaction, SegmentEnd, run_document};
use schedule::ChainSchedule;

/// Why a document cannot be translated: an option every segment shares, or one
/// segment's prompt.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DocError {
    #[error(transparent)]
    Options(PromptError),
    #[error(transparent)]
    Segment(SegmentError),
}

/// A segment's prompt that cannot be built.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("segment {segment}: {error}")]
pub struct SegmentError {
    pub segment: usize,
    pub error: PromptError,
}

/// Chains used when the caller does not choose: one per 4 segments, at most 8.
pub fn auto_chains(segments: usize) -> usize {
    segments.div_ceil(4).clamp(1, 8)
}

/// How many chains a document is split into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Chains {
    /// [`auto_chains`]: one per 4 segments.
    #[default]
    Auto,
    Fixed(std::num::NonZeroUsize),
}

impl From<usize> for Chains {
    /// For a count that says 0 for the automatic number (a command-line option).
    fn from(n: usize) -> Self {
        std::num::NonZeroUsize::new(n).map_or(Chains::Auto, Chains::Fixed)
    }
}

impl Chains {
    /// The number of chains for a document of `segments` segments.
    pub fn count(self, segments: usize) -> usize {
        match self {
            Chains::Auto => auto_chains(segments),
            Chains::Fixed(n) => n.get(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct DocOptions {
    /// The options every segment's prompt shares: languages, form, context and glossary
    /// (its `text` and `history` are not used).
    pub request: TranslationRequest,
    /// History budget per chain in characters (0: no history).
    pub history_chars: usize,
    pub chains: Chains,
    /// Pass the neighbouring source segments as context.
    pub neighbor_context: bool,
    pub max_new_tokens: usize,
    pub history_rules: HistoryRules,
}

impl DocOptions {
    /// Options for a translation into `target_lang`, with the API defaults otherwise.
    pub fn new(target_lang: impl Into<String>) -> Self {
        DocOptions {
            request: TranslationRequest { target_lang: target_lang.into(), ..Default::default() },
            history_chars: yaxus_api::DEFAULT_HISTORY_CHARS,
            chains: Chains::Auto,
            neighbor_context: false,
            max_new_tokens: yaxus_api::DEFAULT_MAX_NEW_TOKENS,
            history_rules: HistoryRules::default(),
        }
    }
}

/// The prompt of each segment: the shared options, the segment's text, the neighbouring
/// segments as context when asked for, and the history its chain's window keeps.
struct SegmentPrompter {
    /// The shared options, with empty text and history.
    base: TranslationRequest,
    neighbor_context: bool,
    /// Most prompt + generated tokens per segment.
    max_context: usize,
    max_new_tokens: usize,
}

impl SegmentPrompter {
    /// The request for segment `i` of `segments` without history.
    fn bare(&self, segments: &[String], i: usize) -> TranslationRequest {
        let mut context = self.base.context.clone();
        if self.neighbor_context {
            let mut parts = Vec::new();
            if i > 0 {
                parts.push(segments[i - 1].as_str());
            }
            if i + 1 < segments.len() {
                parts.push(segments[i + 1].as_str());
            }
            let neighbors = parts.join("\n\n");
            context = Some(match context {
                Some(c) if !c.is_empty() => format!("{c}\n\n{neighbors}"),
                _ => neighbors,
            });
        }
        TranslationRequest { text: segments[i].clone(), context, ..self.base.clone() }
    }

    /// The request for segment `i` with `history`, the pairs its chain keeps.
    fn with_history(&self, segments: &[String], i: usize, history: Vec<yaxus_api::Pair>) -> TranslationRequest {
        TranslationRequest { history, ..self.bare(segments, i) }
    }

    /// The prompt of `req` and its generation budget, both within the context: a history
    /// that leaves too little room is dropped.
    fn fit(&self, tok: &Tokenizer, req: TranslationRequest, max_new: usize) -> Result<(Prompt, usize), PromptError> {
        Prompt::fit(tok, req.validate()?, self.max_context, max_new)
    }
}

/// The state of one document: its segments, their translations so far, the schedule
/// of the chains and the segments' prompts.
pub struct DocJob {
    segments: Vec<String>,
    outputs: Vec<Option<String>>,
    history: ChainHistories,
    schedule: ChainSchedule,
    prompter: SegmentPrompter,
}

impl DocJob {
    /// The job of a document request, with the API defaults for what it leaves out, for
    /// at most `max_context` prompt + generated tokens per segment. A segment generates at
    /// most half of that (and at least 1 token).
    pub fn from_request(r: DocumentRequest, max_context: usize) -> Self {
        let request = TranslationRequest {
            source_lang: r.source_lang,
            form: r.form,
            target_lang: r.target_lang,
            context: r.context,
            glossary: r.glossary,
            ..Default::default()
        };
        let opts = DocOptions {
            request,
            history_chars: r.history_chars.unwrap_or(yaxus_api::DEFAULT_HISTORY_CHARS),
            chains: r.chains.map_or(Chains::Auto, Chains::Fixed),
            neighbor_context: r.neighbor_context,
            max_new_tokens: r.max_new_tokens.unwrap_or(yaxus_api::DEFAULT_MAX_NEW_TOKENS).min(max_context / 2).max(1),
            history_rules: HistoryRules::default(),
        };
        DocJob::new(r.segments, opts).with_max_context(max_context)
    }

    /// At most `max_context` prompt + generated tokens per segment.
    pub fn with_max_context(mut self, max_context: usize) -> Self {
        self.prompter.max_context = max_context;
        self
    }

    /// A job with at most [`DEFAULT_MAX_POS`] prompt + generated tokens per segment, the
    /// positions the model's KV covers.
    pub fn new(segments: Vec<String>, opts: DocOptions) -> Self {
        let n = segments.len();
        let schedule = ChainSchedule::new(n, opts.chains.count(n));
        let base = TranslationRequest { text: String::new(), history: Vec::new(), ..opts.request };
        let prompter = SegmentPrompter {
            base,
            neighbor_context: opts.neighbor_context,
            max_context: DEFAULT_MAX_POS,
            max_new_tokens: opts.max_new_tokens,
        };
        DocJob {
            outputs: vec![None; n],
            history: ChainHistories::new(schedule.chains(), opts.history_chars, opts.history_rules),
            segments,
            schedule,
            prompter,
        }
    }

    /// Checks the segments' prompts before any is sent: the options once, then each
    /// segment's prompt without history (its text, and the neighbouring segments when
    /// they are the context), which must also fit the context (blank segments are allowed;
    /// they come back empty). History is earlier output; what does not fit is dropped
    /// when the segment is submitted.
    pub fn validate(&self, tok: &Tokenizer) -> Result<(), DocError> {
        self.prompter.base.validate_options().map_err(DocError::Options)?;
        for i in (0..self.segments.len()).filter(|&i| !is_blank(&self.segments[i])) {
            self.prompter
                .fit(tok, self.prompter.bare(&self.segments, i), 0)
                .map_err(|error| DocError::Segment(SegmentError { segment: i, error }))?;
        }
        Ok(())
    }

    /// The prompt of a request from [`ready`](Self::ready) and its generation budget, both
    /// within the context: a history that leaves too little room is dropped.
    pub(crate) fn prompt(&self, tok: &Tokenizer, req: TranslationRequest) -> Result<(Prompt, usize), PromptError> {
        self.prompter.fit(tok, req, self.prompter.max_new_tokens)
    }

    pub fn chains(&self) -> usize {
        self.schedule.chains()
    }

    /// Number of segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// New tokens allowed per segment.
    pub fn max_new_tokens(&self) -> usize {
        self.prompter.max_new_tokens
    }

    /// The translation of each segment, once it is done.
    pub fn outputs(&self) -> &[Option<String>] {
        &self.outputs
    }

    pub fn is_done(&self) -> bool {
        self.schedule.is_done()
    }

    /// Takes the segments that can start now (the next one of every idle chain, which is
    /// then busy), with their requests. Taking also completes the segments without text,
    /// at once with an empty translation, and moves on in their chains.
    pub(crate) fn take_ready(&mut self) -> Vec<(usize, TranslationRequest)> {
        let DocJob { segments, outputs, history, schedule, prompter } = self;
        let ready = schedule.take_ready(|i| {
            let blank = is_blank(&segments[i]);
            if blank {
                outputs[i] = Some(String::new());
            }
            blank
        });
        ready
            .into_iter()
            .map(|i| (i, prompter.with_history(segments, i, history.select(schedule.chain_of(i)))))
            .collect()
    }

    /// Records the translation of segment `i`; its chain may continue (see `ready`).
    pub(crate) fn finished(&mut self, i: usize, text: String, truncated: bool) {
        self.history.record(self.schedule.chain_of(i), &self.segments[i], &text, truncated);
        self.outputs[i] = Some(text);
        self.schedule.finish(i);
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use yaxus_model::{Event, Finish, Generator, ReqId, Request};
    use yaxus_pfml::ValidRequest;

    use yaxus_api::Pair;

    use super::*;
    use crate::memory::Memory;

    fn opts(chains: usize, history_chars: usize) -> DocOptions {
        DocOptions { history_chars, chains: Chains::from(chains), ..DocOptions::new("en") }
    }

    #[test]
    fn requests_take_api_defaults_and_the_limit() {
        let r = DocumentRequest {
            segments: vec!["a".into()],
            target_lang: "en".into(),
            max_new_tokens: Some(10_000),
            ..Default::default()
        };
        let job = DocJob::from_request(r.clone(), 512);
        assert_eq!(job.prompter.base.source_lang, None);
        assert_eq!((job.max_new_tokens(), job.prompter.max_context), (256, 512));
        assert_eq!(job.history.budget_chars(), yaxus_api::DEFAULT_HISTORY_CHARS);
        // A context of 1 still allows one token, without panicking.
        assert_eq!(DocJob::from_request(r, 1).max_new_tokens(), 1);
        assert_eq!(DocJob::new(vec![], opts(0, 0)).prompter.max_context, DEFAULT_MAX_POS);
    }

    #[test]
    fn validation_checks_options_once_and_every_segment() {
        let req = |segments: &[&str], form: Option<yaxus_api::Form>| DocumentRequest {
            segments: segments.iter().map(|s| s.to_string()).collect(),
            target_lang: "en".into(),
            form,
            ..Default::default()
        };
        let (tok, _) = tokenizer();
        let n = DEFAULT_MAX_POS;
        assert!(DocJob::from_request(req(&["a", "  ", "b"], None), n).validate(&tok).is_ok());
        let e = DocJob::from_request(req(&["a", "x<|plamo:msg|>"], None), n).validate(&tok).unwrap_err();
        assert!(matches!(e, DocError::Segment(SegmentError { segment: 1, .. })), "{e}");
        // `form` needs target_lang = "ja".
        let e = DocJob::from_request(req(&["a"], Some(yaxus_api::Form::Polite)), n).validate(&tok).unwrap_err();
        assert!(matches!(e, DocError::Options(_)), "{e}");
        // Each segment's prompt must fit the context: blank ones need not.
        let b = TranslationRequest { text: "b".into(), target_lang: "en".into(), ..Default::default() };
        let fits = Prompt::new(&tok, &b.validate().unwrap()).len() + 1;
        assert!(DocJob::from_request(req(&["b", " "], None), fits).validate(&tok).is_ok());
        let e = DocJob::from_request(req(&["b", "bb"], None), fits).validate(&tok).unwrap_err();
        let DocError::Segment(SegmentError { segment: 1, error }) = &e else { panic!("{e}") };
        assert_eq!(error.kind, yaxus_pfml::PromptErrorKind::TooLong, "{e}");
        assert!(e.to_string().starts_with("segment 1: prompt is"), "{e}");
    }

    /// Emits one "x" token per step for every request it holds, and stops each request
    /// after `len` tokens (or when the sink returns `false`).
    #[derive(Default)]
    struct FakeEngine {
        next_id: ReqId,
        len: usize,
        token: u32,
        running: Vec<(ReqId, usize)>,
        submitted: usize,
        cancelled: usize,
        /// The request with this id fails at its first step.
        fail: Option<ReqId>,
        /// Whether it can cancel a request at once (as the real engine); else a request is
        /// cancelled only when its next token is refused.
        cancels: bool,
        /// Requests cancelled by `cancel`, which end at the next step.
        to_cancel: Vec<ReqId>,
        /// How many requests ended cancelled through `cancel`.
        cancelled_at_once: usize,
    }

    impl Generator for FakeEngine {
        fn submit(&mut self, req: Request) {
            self.submitted += 1;
            self.running.push((req.id, 0));
        }

        fn step_async<'a>(
            &'a mut self,
            emit: &'a mut dyn FnMut(Event) -> bool,
        ) -> yaxus_model::LocalBoxFuture<'a, Result<bool>> {
            Box::pin(std::future::ready(yaxus_model::SyncGenerator::step(self, emit)))
        }

        fn fresh_id(&mut self) -> ReqId {
            self.next_id += 1;
            self.next_id
        }
        fn cancel(&mut self, id: ReqId) -> bool {
            let Some(at) = self.running.iter().position(|&(i, _)| i == id).filter(|_| self.cancels) else {
                return false;
            };
            self.running.remove(at);
            self.to_cancel.push(id);
            true
        }
        fn waiting(&self) -> usize {
            0
        }
        fn active(&self) -> usize {
            self.running.len()
        }
        fn backend_name(&self) -> String {
            "fake".into()
        }
        fn weight_bytes(&self) -> usize {
            0
        }
        fn stats(&self) -> Option<String> {
            None
        }
        fn last_start(&self) -> yaxus_model::StartInfo {
            Default::default()
        }
    }

    impl yaxus_model::SyncGenerator for FakeEngine {
        fn score(&mut self, _: &[u32], _: &[u32]) -> Result<Vec<yaxus_model::TokenScore>> {
            anyhow::bail!("the fake engine does not score")
        }

        fn step(&mut self, emit: &mut dyn FnMut(Event) -> bool) -> Result<bool> {
            for id in std::mem::take(&mut self.to_cancel) {
                self.cancelled_at_once += 1;
                emit(Event::Done { id, finish: Finish::Cancelled, generated: 0, start: Default::default() });
            }
            if self.running.is_empty() {
                return Ok(self.cancelled_at_once > 0);
            }
            let mut keep = Vec::new();
            for (id, n) in std::mem::take(&mut self.running) {
                if self.fail == Some(id) {
                    emit(Event::Error { id, message: "boom".into() });
                    continue;
                }
                let finish = if !emit(Event::Token { id, token: self.token }) {
                    self.cancelled += 1;
                    Some(Finish::Cancelled)
                } else if n + 1 == self.len {
                    Some(Finish::Stop)
                } else {
                    None
                };
                match finish {
                    Some(finish) => {
                        emit(Event::Done { id, finish, generated: n + 1, start: Default::default() });
                    }
                    None => keep.push((id, n + 1)),
                }
            }
            self.running = keep;
            Ok(true)
        }
    }

    /// A tokenizer with the PLaMo special-token layout and a single "x" token.
    fn tokenizer() -> (Tokenizer, u32) {
        let tok = yaxus_tokenizer::testing::plamo_like(&[("x", 0.0)]);
        let x = tok.vocab_size() as u32 - 1;
        (tok, x)
    }

    fn fake(len: usize) -> (Tokenizer, FakeEngine) {
        let (tok, x) = tokenizer();
        (tok, FakeEngine { len, token: x, ..Default::default() })
    }

    #[test]
    fn fit_drops_history_then_cuts_the_budget() {
        let (tok, _) = tokenizer();
        let base = TranslationRequest { text: "hello".into(), target_lang: "en".into(), ..Default::default() };
        let with_history = TranslationRequest {
            history: vec![Pair { source: "a".repeat(300), target: "b".repeat(300) }],
            ..base.clone()
        };
        let (base, with_history) = (base.validate().unwrap(), with_history.validate().unwrap());
        let bare = Prompt::new(&tok, &base).len();
        let full = Prompt::new(&tok, &with_history).len();
        assert!(full > bare + 500);
        let fit =
            |r: &ValidRequest, ctx, max_new| Prompt::fit(&tok, r.clone(), ctx, max_new).map(|(p, n)| (p.len(), n));

        // Everything fits: unchanged.
        assert_eq!(fit(&with_history, full + 10, 10).unwrap(), (full, 10));
        // A large cap is cut while the history leaves a useful budget.
        assert_eq!(fit(&with_history, full + 600, 4096).unwrap(), (full, 600));
        // History plus budget too long: the history goes, the budget stays.
        assert_eq!(fit(&with_history, full, 10).unwrap(), (bare, 10));
        // Without history the budget is cut to what is left (at least 1).
        assert_eq!(fit(&base, bare + 3, 10).unwrap(), (bare, 3));
        // The prompt alone does not fit.
        assert!(fit(&base, bare, 0).unwrap_err().message.contains("the limit is"));
        // The memory key is that of the prompt sent.
        let (p, _) = Prompt::fit(&tok, with_history.clone(), full, 10).unwrap();
        assert_eq!(p.into_parts().1, crate::memory::key(&base));
    }

    #[test]
    fn run_document_translates_every_segment() {
        let (tok, mut engine) = fake(3);
        let segs: Vec<String> = (0..5).map(|i| format!("p{i}")).collect();
        let job = DocJob::new(segs, opts(2, 1000));
        let mut memory = Memory::new(8);
        let mut done = Vec::new();
        let job = run_document(&mut engine, &tok, job, &mut memory, &mut |ev| {
            if let DocEvent::SegmentDone(end) = ev {
                done.push(end.segment);
            }
            true
        })
        .unwrap();
        assert!(job.is_done());
        assert!(job.outputs.iter().all(|o| o.as_deref() == Some("xxx")));
        done.sort();
        assert_eq!(done, [0, 1, 2, 3, 4]);
        assert_eq!((engine.submitted, engine.cancelled), (5, 0));
    }

    #[test]
    fn a_stopped_document_leaves_nothing_in_flight() {
        let (tok, mut engine) = fake(50);
        engine.cancels = true;
        let segs: Vec<String> = (0..8).map(|i| format!("p{i}")).collect();
        let mut memory = Memory::new(8);
        let mut texts = 0;
        // Four chains: four segments run side by side. The sink stops at the first text.
        let job = run_document(&mut engine, &tok, DocJob::new(segs, opts(4, 1000)), &mut memory, &mut |ev| {
            texts += usize::from(matches!(ev, DocEvent::Delta { .. }));
            false
        })
        .unwrap();
        assert!(!job.is_done());
        assert_eq!(engine.submitted, 4, "nothing new is started");
        assert_eq!(engine.cancelled + engine.cancelled_at_once, 4, "each of the four ended cancelled");
        assert!(engine.running.is_empty());
        assert_eq!(texts, 1, "no text after the sink said no");
    }

    #[test]
    fn an_outside_cancel_stops_the_document_between_steps() {
        let (tok, mut engine) = fake(50);
        engine.cancels = true;
        let segs: Vec<String> = (0..8).map(|i| format!("p{i}")).collect();
        let mut memory = Memory::new(8);
        let mut steps = 0;
        let mut texts = 0;
        let mut cancel = || {
            steps += 1;
            steps >= 3 // asked after each step: the third step is the last
        };
        let job = yaxus_model::poll_once(DocRunner::new(&mut engine, &tok, &mut memory).cancel_when(&mut cancel).run(
            DocJob::new(segs, opts(4, 1000)),
            &mut |ev| {
                texts += usize::from(matches!(ev, DocEvent::Delta { .. }));
                true
            },
        ))
        .unwrap();
        assert!(!job.is_done());
        assert_eq!(texts, 4 * 3, "four segments for three steps");
        assert_eq!(engine.submitted, 4);
        assert_eq!(
            engine.cancelled_at_once, 4,
            "the four in flight were cancelled in the engine, not at their next text"
        );
        assert!(engine.running.is_empty(), "everything in flight ended");
    }

    #[test]
    fn a_cancel_asked_after_the_last_step_is_not_a_cancel() {
        let (tok, mut engine) = fake(1);
        engine.cancels = true;
        let segs: Vec<String> = (0..2).map(|i| format!("p{i}")).collect();
        let mut memory = Memory::new(8);
        let mut asked = 0;
        let mut cancel = || {
            asked += 1;
            true
        };
        // Two chains, one token each: the whole document ends in the first step.
        let job = yaxus_model::poll_once(
            DocRunner::new(&mut engine, &tok, &mut memory)
                .cancel_when(&mut cancel)
                .run(DocJob::new(segs, opts(2, 1000)), &mut |_| true),
        )
        .unwrap();
        assert!(job.is_done(), "it was finished, not cancelled");
        assert_eq!(asked, 0, "nothing was left to cancel");
    }

    #[test]
    fn run_document_stops_when_the_sink_says_so() {
        // Stop at the first delta: the two chains' first segments are in flight.
        let (tok, mut engine) = fake(3);
        let segs: Vec<String> = (0..6).map(|i| format!("p{i}")).collect();
        let job = DocJob::new(segs, opts(2, 1000));
        let mut memory = Memory::new(8);
        let mut events = 0;
        let job = run_document(&mut engine, &tok, job, &mut memory, &mut |_| {
            events += 1;
            false
        })
        .unwrap();
        assert_eq!(events, 1, "no events after the sink said stop");
        assert!(!job.is_done());
        assert!(engine.running.is_empty(), "requests in flight are drained from the engine");
        assert_eq!((engine.submitted, engine.cancelled), (2, 2), "nothing new is started");

        // Stop at the end of the first segment: it is kept, nothing else is started.
        let (tok, mut engine) = fake(2);
        let segs: Vec<String> = (0..4).map(|i| format!("p{i}")).collect();
        let job = DocJob::new(segs, opts(1, 1000));
        let job = run_document(&mut engine, &tok, job, &mut memory, &mut |ev| !matches!(ev, DocEvent::SegmentDone(_)))
            .unwrap();
        assert_eq!(job.outputs[0].as_deref(), Some("xx"));
        assert!(job.outputs[1..].iter().all(Option::is_none));
        assert_eq!((engine.submitted, engine.cancelled), (1, 0));
    }

    #[test]
    fn a_failed_segment_stops_the_document_after_draining() {
        let (tok, mut engine) = fake(3);
        engine.fail = Some(1);
        let segs: Vec<String> = (0..6).map(|i| format!("p{i}")).collect();
        let mut memory = Memory::new(8);
        let mut ends = 0;
        let r = run_document(&mut engine, &tok, DocJob::new(segs, opts(2, 1000)), &mut memory, &mut |ev| {
            ends += usize::from(matches!(ev, DocEvent::SegmentDone(_)));
            true
        });
        let Err(e) = r else { panic!("the failure is returned") };
        assert_eq!(e.to_string(), "segment 0: boom");
        assert!(engine.running.is_empty(), "the other chain's segment is drained");
        assert_eq!((engine.submitted, engine.cancelled, ends), (2, 1, 0));
    }

    #[test]
    fn chains_split_contiguously_and_advance_in_order() {
        let segs: Vec<String> = (0..10).map(|i| format!("p{i}")).collect();
        let mut job = DocJob::new(segs, opts(3, 1000));
        assert_eq!(job.chains(), 3);
        let first: Vec<usize> = job.take_ready().into_iter().map(|(i, _)| i).collect();
        assert_eq!(first, vec![0, 4, 7]); // chains of 4, 3, 3
        assert!(job.take_ready().is_empty(), "in-flight segments are not offered again");
        job.finished(4, "t4".into(), false);
        let next = job.take_ready();
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].0, 5);
        assert_eq!(next[0].1.history.len(), 1);
        assert_eq!(next[0].1.history[0].source, "p4");
    }

    #[test]
    fn blank_segments_complete_empty_and_stay_out_of_history() {
        let segs = vec!["a".to_string(), "  ".to_string(), "b".to_string()];
        let mut job = DocJob::new(segs, opts(1, 1000));
        assert_eq!(job.take_ready()[0].0, 0);
        job.finished(0, "A".into(), false);
        let next = job.take_ready();
        assert_eq!(next[0].0, 2);
        assert_eq!(next[0].1.history.len(), 1);
        assert_eq!(job.outputs()[1].as_deref(), Some(""));
        assert!(next[0].1.clone().validate().is_ok());
    }

    #[test]
    fn blank_means_what_the_prompt_template_means() {
        // U+001F is whitespace to Python's strip (and so to the template), not to Rust's.
        let segs = vec!["a".to_string(), "\u{1f}".to_string()];
        let mut job = DocJob::new(segs, opts(1, 1000));
        assert!(job.validate(&tokenizer().0).is_ok());
        assert_eq!(job.take_ready()[0].0, 0);
        job.finished(0, "A".into(), false);
        assert!(job.take_ready().is_empty() && job.is_done(), "the blank segment completes empty");
    }

    #[test]
    fn translations_with_reserved_tokens_stay_out_of_history() {
        let segs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut job = DocJob::new(segs, opts(1, 1000));
        job.take_ready();
        job.finished(0, "A<|plamo:msg|>".into(), false); // a control token the model emitted
        let next = job.take_ready();
        assert!(next[0].1.history.is_empty());
        assert!(next[0].1.clone().validate().is_ok());
        job.finished(1, "B".into(), false);
        assert_eq!(job.take_ready()[0].1.history.len(), 1, "later pairs are kept");
    }

    #[test]
    fn history_rules_leave_out_truncated_translations_and_trim_the_rest() {
        let segs: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let rules = HistoryRules { skip_truncated: true, trim: true };
        let mut job = DocJob::new(segs.clone(), DocOptions { history_rules: rules, ..opts(1, 1000) });
        job.take_ready();
        job.finished(0, " A\n".into(), false);
        let next = job.take_ready();
        assert_eq!(next[0].1.history[0].target, "A", "trimmed");
        job.finished(1, "B".into(), true);
        let next = job.take_ready();
        let sources: Vec<&str> = next[0].1.history.iter().map(|p| p.source.as_str()).collect();
        assert_eq!(sources, ["a"], "the one cut at the budget is left out");

        // The default keeps every translation as the model wrote it.
        let mut job = DocJob::new(segs, opts(1, 1000));
        job.take_ready();
        job.finished(0, " A\n".into(), false);
        assert_eq!(job.take_ready()[0].1.history[0].target, " A\n");
        job.finished(1, "B".into(), true);
        assert_eq!(job.take_ready()[0].1.history.len(), 2);
    }

    #[test]
    fn auto_chain_count() {
        assert_eq!(auto_chains(1), 1);
        assert_eq!(auto_chains(14), 4);
        assert_eq!(auto_chains(100), 8);
    }
}
