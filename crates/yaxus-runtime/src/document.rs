//! Document translation: segments are split into chains of consecutive segments. Within a
//! chain they are translated in order, each with the earlier ones of the chain as history
//! (see [`HistoryWindow`]); the chains run in parallel on the engine.

use anyhow::Result;
use yaxus_api::{DocumentRequest, HistoryWindow, Pair};
use yaxus_model::Generator;
use yaxus_tokenizer::{GlossaryEntry, PromptError, Tokenizer, TranslationPair, TranslationRequest, validate_message};

use crate::memory::Memory;
use crate::prompt::Prompt;
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
}

impl DocJob {
    /// The job of a document request, with the API defaults for what it leaves out and
    /// at most `max_new_tokens_limit` new tokens per segment (at least 1).
    pub fn from_request(r: DocumentRequest, max_new_tokens_limit: usize) -> Self {
        let opts = DocOptions {
            source_lang: r.requested_source_lang().map(String::from),
            form: r.requested_form().map(String::from),
            history_chars: r.history_chars.unwrap_or(yaxus_api::DEFAULT_HISTORY_CHARS),
            chains: r.chains.unwrap_or(0),
            neighbor_context: r.neighbor_context,
            max_new_tokens: r
                .max_new_tokens
                .unwrap_or(yaxus_api::DEFAULT_MAX_NEW_TOKENS)
                .min(max_new_tokens_limit)
                .max(1),
            target_lang: r.target_lang,
            context: r.context,
            glossary: r.glossary,
        };
        DocJob::new(r.segments, opts)
    }

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
        DocJob { outputs: vec![None; n], in_flight: vec![false; n], segments, opts, chains, chain_of }
    }

    /// Checks what the segments' prompts are made of before any is sent: the options once
    /// and each segment's text (blank segments are allowed; they come back empty). The
    /// history added later is earlier output, and the neighbour context earlier text.
    pub fn validate(&self) -> Result<(), PromptError> {
        self.options_request().validate_options()?;
        for (i, seg) in self.segments.iter().enumerate().filter(|(_, s)| !s.trim().is_empty()) {
            validate_message(seg).map_err(|e| PromptError(format!("segment {i}: {}", e.0)))?;
        }
        Ok(())
    }

    /// The options of every segment's prompt (empty text and history).
    fn options_request(&self) -> TranslationRequest {
        TranslationRequest {
            text: String::new(),
            target_lang: self.opts.target_lang.clone(),
            source_lang: self.opts.source_lang.clone(),
            form: self.opts.form.clone(),
            context: self.opts.context.clone(),
            glossary: self
                .opts
                .glossary
                .iter()
                .map(|p| GlossaryEntry { source: p.source.clone(), target: p.target.clone() })
                .collect(),
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
            // Blank segments (and empty translations) cannot be history entries.
            .filter(|p| !p.source.trim().is_empty() && !p.target.trim().is_empty())
            .collect();
        let history = chain.window.select(&done);
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
        TranslationRequest {
            text: self.segments[i].clone(),
            context,
            history: history
                .iter()
                .map(|p| TranslationPair { source: p.source.clone(), target: p.target.clone() })
                .collect(),
            ..self.options_request()
        }
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
                if self.segments[i].trim().is_empty() {
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
pub fn run_document(
    engine: &mut dyn Generator,
    tok: &Tokenizer,
    job: &mut DocJob,
    memory: &mut Memory,
    on: &mut dyn FnMut(DocEvent),
) -> Result<()> {
    let mut session = Session::new(engine, tok, memory);
    let submit = |session: &mut Session<usize>, job: &mut DocJob| -> Result<()> {
        for (segment, req) in job.ready() {
            session.submit(Prompt::new(tok, &req)?, job.max_new_tokens(), segment);
        }
        Ok(())
    };
    submit(&mut session, job)?;
    while !job.is_done() {
        let (mut finished, mut failed) = (Vec::new(), None);
        let worked = session.step(&mut |u| {
            match u {
                Update::Delta { tag, text } if !text.is_empty() => on(DocEvent::Delta { segment: *tag, text }),
                Update::Delta { .. } => {}
                Update::Done { tag, done } => finished.push((tag, done)),
                Update::Failed { tag, message } => failed = Some(format!("segment {tag}: {message}")),
            }
            true
        })?;
        if let Some(e) = failed {
            anyhow::bail!(e);
        }
        for (segment, done) in finished {
            on(DocEvent::SegmentDone {
                segment,
                prompt_tokens: done.prompt_tokens,
                reused: done.start.reused,
                generated: done.generated,
            });
            job.finished(segment, done.text);
        }
        submit(&mut session, job)?;
        anyhow::ensure!(worked || job.is_done() || !session.is_idle(), "engine idle with segments left");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
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
        let job = DocJob::from_request(r.clone(), 256);
        assert_eq!(job.opts.source_lang, None);
        assert_eq!(job.max_new_tokens(), 256);
        assert_eq!(job.opts.history_chars, yaxus_api::DEFAULT_HISTORY_CHARS);
        // A limit of 0 (from a tiny --max-context) still allows one token, without panicking.
        assert_eq!(DocJob::from_request(r, 0).max_new_tokens(), 1);
    }

    #[test]
    fn validation_checks_options_once_and_every_segment() {
        let req = |segments: &[&str], form: Option<&str>| DocumentRequest {
            segments: segments.iter().map(|s| s.to_string()).collect(),
            target_lang: "en".into(),
            form: form.map(String::from),
            ..Default::default()
        };
        assert!(DocJob::from_request(req(&["a", "  ", "b"], None), 64).validate().is_ok());
        let e = DocJob::from_request(req(&["a", "x<|plamo:msg|>"], None), 64).validate().unwrap_err();
        assert!(e.0.starts_with("segment 1:"), "{}", e.0);
        // `form` needs target_lang = "ja".
        assert!(DocJob::from_request(req(&["a"], Some("polite")), 64).validate().is_err());
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
    fn auto_chain_count() {
        assert_eq!(auto_chains(1), 1);
        assert_eq!(auto_chains(14), 4);
        assert_eq!(auto_chains(100), 8);
    }
}
