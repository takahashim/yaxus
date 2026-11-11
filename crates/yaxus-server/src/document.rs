//! Document translation: segments are split into chains of consecutive segments. Within a
//! chain they are translated in order, each with the earlier ones of the chain as history
//! (see [`HistoryWindow`]); the chains run in parallel on the engine.

use std::collections::HashMap;

use anyhow::Result;
use yaxus_api::{HistoryWindow, Pair};
use yaxus_model::{Event, Finish, Generator, ReqId, Request};
use yaxus_tokenizer::{EOS_ID, GlossaryEntry, StreamDecoder, TAG_ID, Tokenizer, TranslationPair, TranslationRequest};

use crate::memory::{self, Memory};

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

struct Chain {
    /// Segment indices of the chain, in order.
    segs: Vec<usize>,
    next: usize,
    window: HistoryWindow,
}

/// The state of one document.
pub struct DocJob {
    pub segments: Vec<String>,
    pub opts: DocOptions,
    pub outputs: Vec<Option<String>>,
    /// Segments submitted and not finished yet.
    in_flight: Vec<bool>,
    chains: Vec<Chain>,
    /// Chain of each segment.
    chain_of: Vec<usize>,
}

impl DocJob {
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

    pub fn chains(&self) -> usize {
        self.chains.len()
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
            target_lang: self.opts.target_lang.clone(),
            source_lang: self.opts.source_lang.clone(),
            form: self.opts.form.clone(),
            context,
            glossary: self
                .opts
                .glossary
                .iter()
                .map(|p| GlossaryEntry { source: p.source.clone(), target: p.target.clone() })
                .collect(),
            history: history
                .iter()
                .map(|p| TranslationPair { source: p.source.clone(), target: p.target.clone() })
                .collect(),
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
    struct Live {
        seg: usize,
        dec: StreamDecoder,
        text: String,
        tokens: Vec<u32>,
        prompt_tokens: usize,
        memo: u64,
    }
    let mut live: HashMap<ReqId, Live> = HashMap::new();
    let submit = |engine: &mut dyn Generator,
                  job: &mut DocJob,
                  live: &mut HashMap<ReqId, Live>,
                  memory: &Memory|
     -> Result<()> {
        for (seg, req) in job.ready() {
            let ids = tok.encode(&req.to_prompt()?, false);
            let id = engine.fresh_id();
            let memo = memory::key(&req);
            let prompt_tokens = ids.len();
            live.insert(
                id,
                Live { seg, dec: StreamDecoder::new(), text: String::new(), tokens: Vec::new(), prompt_tokens, memo },
            );
            let mut r = Request::new(id, ids, job.opts.max_new_tokens, vec![EOS_ID, TAG_ID]);
            r.hint = memory.get(memo);
            engine.submit(r);
        }
        Ok(())
    };
    submit(engine, job, &mut live, memory)?;
    let mut stopped = false;
    // Once stopped, run only until the requests in flight have left the engine.
    while !job.is_done() && !(stopped && live.is_empty()) {
        let mut finished = Vec::new();
        let mut error = None;
        let worked = engine.step(&mut |ev| {
            match ev {
                // Returning `false` cancels the request.
                Event::Token { .. } if stopped => return false,
                Event::Token { id, token } => {
                    if let Some(l) = live.get_mut(&id) {
                        l.tokens.push(token);
                        let start = l.text.len();
                        l.dec.push_into(tok, token, &mut l.text);
                        if l.text.len() > start && !on(DocEvent::Delta { segment: l.seg, text: &l.text[start..] }) {
                            stopped = true;
                            return false;
                        }
                    }
                }
                Event::Done { id, generated, start, finish } => finished.push((id, generated, start.reused, finish)),
                Event::Error { message, .. } => error = Some(message),
            }
            true
        })?;
        if let Some(e) = error {
            anyhow::bail!(e);
        }
        for (id, generated, reused, finish) in finished {
            if let Some(mut l) = live.remove(&id) {
                if finish == Finish::Stop {
                    memory.put(l.memo, std::mem::take(&mut l.tokens));
                }
                if stopped {
                    continue;
                }
                let tail = l.dec.finish();
                if !tail.is_empty() {
                    stopped = !on(DocEvent::Delta { segment: l.seg, text: &tail });
                    l.text += &tail;
                }
                if !stopped {
                    stopped = !on(DocEvent::SegmentDone {
                        segment: l.seg,
                        prompt_tokens: l.prompt_tokens,
                        reused,
                        generated,
                    });
                }
                job.finished(l.seg, l.text);
            }
        }
        if !stopped {
            submit(engine, job, &mut live, memory)?;
        }
        anyhow::ensure!(worked || job.is_done() || !live.is_empty(), "engine idle with segments left");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(chains: usize, history_chars: usize) -> DocOptions {
        DocOptions {
            target_lang: "en".into(),
            source_lang: None,
            form: None,
            context: None,
            glossary: vec![],
            history_chars,
            chains,
            neighbor_context: false,
            max_new_tokens: 64,
        }
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
        assert_eq!(job.outputs[1].as_deref(), Some(""));
        assert!(next[0].1.to_prompt().is_ok());
    }

    #[test]
    fn auto_chain_count() {
        assert_eq!(auto_chains(1), 1);
        assert_eq!(auto_chains(14), 4);
        assert_eq!(auto_chains(100), 8);
    }
}
