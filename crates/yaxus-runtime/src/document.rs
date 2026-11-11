//! Document translation: segments are split into chains of consecutive segments. Within a
//! chain they are translated in order, each with the earlier ones of the chain as history
//! (see [`HistoryWindow`]); the chains run in parallel on the engine.

use anyhow::Result;
use yaxus_api::{DocumentRequest, HistoryWindow, Pair};
use yaxus_model::{DEFAULT_MAX_POS, Finish, Generator};
use yaxus_tokenizer::{PromptError, Tokenizer, TranslationRequest, is_blank, validate_message};

use crate::memory::Memory;
use crate::prompt::{self, Prompt};
use crate::session::{Session, Update};

/// Chains used when the caller does not choose: one per 4 segments, at most 8.
pub fn auto_chains(segments: usize) -> usize {
    segments.div_ceil(4).clamp(1, 8)
}

#[derive(Debug, Clone)]
pub struct DocOptions {
    pub target_lang: String,
    pub source_lang: Option<String>,
    pub form: Option<String>,
    pub context: Option<String>,
    pub glossary: Vec<Pair>,
    /// History budget per chain in characters (0: no history).
    pub history_chars: usize,
    /// Number of chains (0: `auto_chains`).
    pub chains: usize,
    /// Pass the neighbouring source segments as context.
    pub neighbor_context: bool,
    pub max_new_tokens: usize,
}

impl DocOptions {
    /// Options for a translation into `target_lang`, with the API defaults otherwise.
    pub fn new(target_lang: impl Into<String>) -> Self {
        DocOptions {
            target_lang: target_lang.into(),
            source_lang: None,
            form: None,
            context: None,
            glossary: Vec::new(),
            history_chars: yaxus_api::DEFAULT_HISTORY_CHARS,
            chains: 0,
            neighbor_context: false,
            max_new_tokens: yaxus_api::DEFAULT_MAX_NEW_TOKENS,
        }
    }
}

struct Chain {
    /// Segment indices of the chain, in order.
    segs: Vec<usize>,
    next: usize,
    window: HistoryWindow,
}

/// The state of one document.
pub struct DocJob {
    segments: Vec<String>,
    opts: DocOptions,
    outputs: Vec<Option<String>>,
    /// Segments submitted and not finished yet.
    in_flight: Vec<bool>,
    chains: Vec<Chain>,
    /// Chain of each segment.
    chain_of: Vec<usize>,
    /// Most prompt + generated tokens per segment.
    max_context: usize,
}

impl DocJob {
    /// The job of a document request, with the API defaults for what it leaves out, for
    /// at most `max_context` prompt + generated tokens per segment. A segment generates at
    /// most half of that (and at least 1 token).
    pub fn from_request(r: DocumentRequest, max_context: usize) -> Self {
        let opts = DocOptions {
            source_lang: r.requested_source_lang().map(String::from),
            form: r.requested_form().map(String::from),
            history_chars: r.history_chars.unwrap_or(yaxus_api::DEFAULT_HISTORY_CHARS),
            chains: r.chains.unwrap_or(0),
            neighbor_context: r.neighbor_context,
            max_new_tokens: r.max_new_tokens.unwrap_or(yaxus_api::DEFAULT_MAX_NEW_TOKENS).min(max_context / 2).max(1),
            target_lang: r.target_lang,
            context: r.context,
            glossary: r.glossary,
        };
        DocJob { max_context, ..DocJob::new(r.segments, opts) }
    }

    /// A job with at most [`DEFAULT_MAX_POS`] prompt + generated tokens per segment, the
    /// positions the model's KV covers.
    pub fn new(segments: Vec<String>, opts: DocOptions) -> Self {
        let n = segments.len();
        let k = if opts.chains == 0 { auto_chains(n) } else { opts.chains }.clamp(1, n.max(1));
        // Contiguous, nearly equal chains, so each chain's history is local context.
        let mut chains = Vec::with_capacity(k);
        let mut chain_of = vec![0; n];
        let mut at = 0;
        for c in 0..k {
            let len = n / k + usize::from(c < n % k);
            let segs: Vec<usize> = (at..at + len).collect();
            for &s in &segs {
                chain_of[s] = c;
            }
            at += len;
            chains.push(Chain { segs, next: 0, window: HistoryWindow::new(opts.history_chars) });
        }
        DocJob {
            outputs: vec![None; n],
            in_flight: vec![false; n],
            segments,
            opts,
            chains,
            chain_of,
            max_context: DEFAULT_MAX_POS,
        }
    }

    /// Checks the segments' prompts before any is sent: the options once, then each
    /// segment's prompt without history (its text, and the neighbouring segments when
    /// they are the context), which must also fit the context (blank segments are allowed;
    /// they come back empty). History is earlier output; what does not fit is dropped
    /// when the segment is submitted (see [`prompt`](Self::prompt)).
    pub fn validate(&self, tok: &Tokenizer) -> Result<(), PromptError> {
        self.options_request().validate_options()?;
        for i in (0..self.segments.len()).filter(|&i| !is_blank(&self.segments[i])) {
            Prompt::fit(tok, self.bare_request(i), self.max_context, 0)
                .map_err(|e| PromptError(format!("segment {i}: {}", e.0)))?;
        }
        Ok(())
    }

    /// The prompt of a request from [`ready`](Self::ready) and its generation budget, both
    /// within the context: a history that leaves too little room is dropped.
    pub fn prompt(&self, tok: &Tokenizer, req: TranslationRequest) -> Result<(Prompt, usize), PromptError> {
        Prompt::fit(tok, req, self.max_context, self.opts.max_new_tokens)
    }

    /// The options of every segment's prompt (empty text and history).
    fn options_request(&self) -> TranslationRequest {
        TranslationRequest {
            text: String::new(),
            target_lang: self.opts.target_lang.clone(),
            source_lang: self.opts.source_lang.clone(),
            form: self.opts.form.clone(),
            context: self.opts.context.clone(),
            glossary: prompt::glossary(&self.opts.glossary),
            history: Vec::new(),
        }
    }

    pub fn chains(&self) -> usize {
        self.chains.len()
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
        self.opts.max_new_tokens
    }

    /// The translation of each segment, once it is done.
    pub fn outputs(&self) -> &[Option<String>] {
        &self.outputs
    }

    pub fn is_done(&self) -> bool {
        self.outputs.iter().all(Option::is_some)
    }

    /// The prompt request of segment `i`; its chain's earlier segments must be done.
    fn request(&mut self, i: usize) -> TranslationRequest {
        let c = self.chain_of[i];
        let chain = &mut self.chains[c];
        let done: Vec<Pair> = chain
            .segs
            .iter()
            .take_while(|&&s| s != i)
            .map(|&s| Pair { source: self.segments[s].clone(), target: self.outputs[s].clone().unwrap_or_default() })
            // Only pairs a prompt may hold: not blank segments or translations, and no
            // translation with reserved tokens (a control token the model emitted).
            .filter(|p| validate_message(&p.source).is_ok() && validate_message(&p.target).is_ok())
            .collect();
        let history = prompt::history(chain.window.select(&done));
        TranslationRequest { history, ..self.bare_request(i) }
    }

    /// The request for segment `i` without history: its text, the options and the
    /// context (with the neighbouring segments when asked for).
    fn bare_request(&self, i: usize) -> TranslationRequest {
        let mut context = self.opts.context.clone();
        if self.opts.neighbor_context {
            let mut parts = Vec::new();
            if i > 0 {
                parts.push(self.segments[i - 1].as_str());
            }
            if i + 1 < self.segments.len() {
                parts.push(self.segments[i + 1].as_str());
            }
            let neighbors = parts.join("\n\n");
            context = Some(match context {
                Some(c) if !c.is_empty() => format!("{c}\n\n{neighbors}"),
                _ => neighbors,
            });
        }
        TranslationRequest { text: self.segments[i].clone(), context, ..self.options_request() }
    }

    /// Segments that can start now (the next one of every idle chain). Segments without
    /// text are completed immediately with an empty translation.
    pub fn ready(&mut self) -> Vec<(usize, TranslationRequest)> {
        let mut out = Vec::new();
        for c in 0..self.chains.len() {
            loop {
                let chain = &self.chains[c];
                let Some(&i) = chain.segs.get(chain.next) else { break };
                if self.in_flight[i] {
                    break;
                }
                if self.outputs[i].is_some() {
                    self.chains[c].next += 1;
                    continue;
                }
                if is_blank(&self.segments[i]) {
                    self.outputs[i] = Some(String::new());
                    self.chains[c].next += 1;
                    continue;
                }
                self.in_flight[i] = true;
                out.push((i, self.request(i)));
                break;
            }
        }
        out
    }

    /// Records the translation of segment `i`; its chain may continue (see `ready`).
    pub fn finished(&mut self, i: usize, text: String) {
        self.in_flight[i] = false;
        self.outputs[i] = Some(text);
        let c = self.chain_of[i];
        if self.chains[c].segs.get(self.chains[c].next) == Some(&i) {
            self.chains[c].next += 1;
        }
    }
}

/// Events of [`run_document`].
pub enum DocEvent<'a> {
    Delta { segment: usize, text: &'a str },
    SegmentDone { segment: usize, prompt_tokens: usize, reused: usize, generated: usize },
}

/// Translates a document on `engine` (which must not have other requests in flight).
/// Earlier translations in `memory` serve as hints; the new ones are added to it.
///
/// `on` returns `false` to stop (e.g. when the output it writes to has gone away): no
/// further segments are started, the ones in flight are cancelled and drained from the
/// engine, no more events are sent, and `job` is left unfinished.
pub fn run_document(
    engine: &mut dyn Generator,
    tok: &Tokenizer,
    job: &mut DocJob,
    memory: &mut Memory,
    on: &mut dyn FnMut(DocEvent) -> bool,
) -> Result<()> {
    let mut session = Session::new(engine, tok, memory);
    let submit = |session: &mut Session<usize>, job: &mut DocJob| -> Result<()> {
        for (segment, req) in job.ready() {
            let (prompt, max_new) = job.prompt(tok, req).map_err(|e| anyhow::anyhow!("segment {segment}: {}", e.0))?;
            session.submit(prompt, max_new, segment);
        }
        Ok(())
    };
    submit(&mut session, job)?;
    let mut stopped = false;
    // Once stopped, run only until the requests in flight have left the engine.
    while !job.is_done() && !(stopped && session.is_idle()) {
        let (mut finished, mut failed) = (Vec::new(), None);
        // The request whose own text was refused: its end still counts (as it ended).
        let mut refused = None;
        let worked = session.step(&mut |u| {
            match u {
                // Returning `false` cancels the request.
                Update::Delta { .. } if stopped => return false,
                Update::Delta { tag, text } => {
                    if !text.is_empty() && !on(DocEvent::Delta { segment: *tag, text }) {
                        (stopped, refused) = (true, Some(*tag));
                        return false;
                    }
                }
                Update::Done { tag, done } => finished.push((tag, done)),
                Update::Failed { tag, message } => failed = Some(format!("segment {tag}: {message}")),
            }
            true
        })?;
        if let Some(e) = failed {
            anyhow::bail!(e);
        }
        for (segment, done) in finished {
            // After a stop: cancelled requests and the others' ends are dropped.
            if done.finish == Finish::Cancelled || (stopped && refused != Some(segment)) {
                continue;
            }
            if !stopped {
                stopped = !on(DocEvent::SegmentDone {
                    segment,
                    prompt_tokens: done.prompt_tokens,
                    reused: done.start.reused,
                    generated: done.generated,
                });
            }
            job.finished(segment, done.text);
        }
        if !stopped {
            submit(&mut session, job)?;
        }
        anyhow::ensure!(worked || job.is_done() || !session.is_idle(), "engine idle with segments left");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use yaxus_model::{Event, ReqId, Request};
    use yaxus_tokenizer::{TAG_ID, TranslationPair};

    use super::*;

    fn opts(chains: usize, history_chars: usize) -> DocOptions {
        DocOptions { history_chars, chains, ..DocOptions::new("en") }
    }

    #[test]
    fn requests_take_api_defaults_and_the_limit() {
        let r = DocumentRequest {
            segments: vec!["a".into()],
            target_lang: "en".into(),
            source_lang: Some(String::new()),
            max_new_tokens: Some(10_000),
            ..Default::default()
        };
        let job = DocJob::from_request(r.clone(), 512);
        assert_eq!(job.opts.source_lang, None);
        assert_eq!((job.max_new_tokens(), job.max_context), (256, 512));
        assert_eq!(job.opts.history_chars, yaxus_api::DEFAULT_HISTORY_CHARS);
        // A context of 1 still allows one token, without panicking.
        assert_eq!(DocJob::from_request(r, 1).max_new_tokens(), 1);
        assert_eq!(DocJob::new(vec![], opts(0, 0)).max_context, DEFAULT_MAX_POS);
    }

    #[test]
    fn validation_checks_options_once_and_every_segment() {
        let req = |segments: &[&str], form: Option<&str>| DocumentRequest {
            segments: segments.iter().map(|s| s.to_string()).collect(),
            target_lang: "en".into(),
            form: form.map(String::from),
            ..Default::default()
        };
        let (tok, _) = tokenizer();
        let n = DEFAULT_MAX_POS;
        assert!(DocJob::from_request(req(&["a", "  ", "b"], None), n).validate(&tok).is_ok());
        let e = DocJob::from_request(req(&["a", "x<|plamo:msg|>"], None), n).validate(&tok).unwrap_err();
        assert!(e.0.starts_with("segment 1:"), "{}", e.0);
        // `form` needs target_lang = "ja".
        assert!(DocJob::from_request(req(&["a"], Some("polite")), n).validate(&tok).is_err());
        // Each segment's prompt must fit the context: blank ones need not.
        let fits =
            Prompt::new(&tok, &TranslationRequest { text: "b".into(), target_lang: "en".into(), ..Default::default() })
                .unwrap()
                .len()
                + 1;
        assert!(DocJob::from_request(req(&["b", " "], None), fits).validate(&tok).is_ok());
        let e = DocJob::from_request(req(&["b", "bb"], None), fits).validate(&tok).unwrap_err();
        assert!(e.0.starts_with("segment 1:") && e.0.contains("the limit is"), "{}", e.0);
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
    }

    impl Generator for FakeEngine {
        fn submit(&mut self, req: Request) {
            self.submitted += 1;
            self.running.push((req.id, 0));
        }

        fn step(&mut self, emit: &mut dyn FnMut(Event) -> bool) -> Result<bool> {
            if self.running.is_empty() {
                return Ok(false);
            }
            let mut keep = Vec::new();
            for (id, n) in std::mem::take(&mut self.running) {
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

        fn fresh_id(&mut self) -> ReqId {
            self.next_id += 1;
            self.next_id
        }
        fn waiting(&self) -> usize {
            0
        }
        fn active(&self) -> usize {
            self.running.len()
        }
        fn abort_all(&mut self) {
            self.running.clear();
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

    /// A tokenizer with the PLaMo special-token layout and a single "x" token.
    fn tokenizer() -> (Tokenizer, u32) {
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
        tokens.push("x".into());
        kinds.push(TokenKind::Normal);
        let x = tokens.len() as u32 - 1;
        let scores = vec![0.0; tokens.len()];
        (Tokenizer::from_vocab(tokens, kinds, scores).unwrap(), x)
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
            history: vec![TranslationPair { source: "a".repeat(300), target: "b".repeat(300) }],
            ..base.clone()
        };
        let bare = Prompt::new(&tok, &base).unwrap().len();
        let full = Prompt::new(&tok, &with_history).unwrap().len();
        assert!(full > bare + 500);
        let fit = |r: &TranslationRequest, ctx, max_new| {
            Prompt::fit(&tok, r.clone(), ctx, max_new).map(|(p, n)| (p.len(), n))
        };

        // Everything fits: unchanged.
        assert_eq!(fit(&with_history, full + 10, 10).unwrap(), (full, 10));
        // A large cap is cut while the history leaves a useful budget.
        assert_eq!(fit(&with_history, full + 600, 4096).unwrap(), (full, 600));
        // History plus budget too long: the history goes, the budget stays.
        assert_eq!(fit(&with_history, full, 10).unwrap(), (bare, 10));
        // Without history the budget is cut to what is left (at least 1).
        assert_eq!(fit(&base, bare + 3, 10).unwrap(), (bare, 3));
        // The prompt alone does not fit.
        assert!(fit(&base, bare, 0).unwrap_err().0.contains("the limit is"));
        // The memory key is that of the prompt sent.
        let (p, _) = Prompt::fit(&tok, with_history.clone(), full, 10).unwrap();
        assert_eq!(p.into_parts().1, crate::memory::key(&base));
    }

    #[test]
    fn run_document_translates_every_segment() {
        let (tok, mut engine) = fake(3);
        let segs: Vec<String> = (0..5).map(|i| format!("p{i}")).collect();
        let mut job = DocJob::new(segs, opts(2, 1000));
        let mut memory = Memory::new(8);
        let mut done = Vec::new();
        run_document(&mut engine, &tok, &mut job, &mut memory, &mut |ev| {
            if let DocEvent::SegmentDone { segment, .. } = ev {
                done.push(segment);
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
    fn run_document_stops_when_the_sink_says_so() {
        // Stop at the first delta: the two chains' first segments are in flight.
        let (tok, mut engine) = fake(3);
        let segs: Vec<String> = (0..6).map(|i| format!("p{i}")).collect();
        let mut job = DocJob::new(segs, opts(2, 1000));
        let mut memory = Memory::new(8);
        let mut events = 0;
        run_document(&mut engine, &tok, &mut job, &mut memory, &mut |_| {
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
        let mut job = DocJob::new(segs, opts(1, 1000));
        run_document(&mut engine, &tok, &mut job, &mut memory, &mut |ev| !matches!(ev, DocEvent::SegmentDone { .. }))
            .unwrap();
        assert_eq!(job.outputs[0].as_deref(), Some("xx"));
        assert!(job.outputs[1..].iter().all(Option::is_none));
        assert_eq!((engine.submitted, engine.cancelled), (1, 0));
    }

    #[test]
    fn chains_split_contiguously_and_advance_in_order() {
        let segs: Vec<String> = (0..10).map(|i| format!("p{i}")).collect();
        let mut job = DocJob::new(segs, opts(3, 1000));
        assert_eq!(job.chains(), 3);
        let first: Vec<usize> = job.ready().into_iter().map(|(i, _)| i).collect();
        assert_eq!(first, vec![0, 4, 7]); // chains of 4, 3, 3
        assert!(job.ready().is_empty(), "in-flight segments are not offered again");
        job.finished(4, "t4".into());
        let next = job.ready();
        assert_eq!(next.len(), 1);
        assert_eq!(next[0].0, 5);
        assert_eq!(next[0].1.history.len(), 1);
        assert_eq!(next[0].1.history[0].source, "p4");
    }

    #[test]
    fn blank_segments_complete_empty_and_stay_out_of_history() {
        let segs = vec!["a".to_string(), "  ".to_string(), "b".to_string()];
        let mut job = DocJob::new(segs, opts(1, 1000));
        assert_eq!(job.ready()[0].0, 0);
        job.finished(0, "A".into());
        let next = job.ready();
        assert_eq!(next[0].0, 2);
        assert_eq!(next[0].1.history.len(), 1);
        assert_eq!(job.outputs()[1].as_deref(), Some(""));
        assert!(next[0].1.to_prompt().is_ok());
    }

    #[test]
    fn blank_means_what_the_prompt_template_means() {
        // U+001F is whitespace to Python's strip (and so to the template), not to Rust's.
        let segs = vec!["a".to_string(), "\u{1f}".to_string()];
        let mut job = DocJob::new(segs, opts(1, 1000));
        assert!(job.validate(&tokenizer().0).is_ok());
        assert_eq!(job.ready()[0].0, 0);
        job.finished(0, "A".into());
        assert!(job.ready().is_empty() && job.is_done(), "the blank segment completes empty");
    }

    #[test]
    fn translations_with_reserved_tokens_stay_out_of_history() {
        let segs = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut job = DocJob::new(segs, opts(1, 1000));
        job.ready();
        job.finished(0, "A<|plamo:msg|>".into()); // a control token the model emitted
        let next = job.ready();
        assert!(next[0].1.history.is_empty());
        assert!(next[0].1.to_prompt().is_ok());
        job.finished(1, "B".into());
        assert_eq!(job.ready()[0].1.history.len(), 1, "later pairs are kept");
    }

    #[test]
    fn auto_chain_count() {
        assert_eq!(auto_chains(1), 1);
        assert_eq!(auto_chains(14), 4);
        assert_eq!(auto_chains(100), 8);
    }
}
