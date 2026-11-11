//! Translations in flight on one engine: what every caller of the engine does for each
//! request, in one place. A request is submitted with the hint the translation memory
//! has for it; its tokens are decoded into text as they come; when it ends at a stop
//! token its output goes into the memory. The server's worker,
//! [`run_document`](crate::document::run_document) and [`run_request`](crate::run_request)
//! differ only in where the text goes, which is what the tag `T` of each request tells.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use yaxus_model::{Event, Finish, Generator, ReqId, Request, StartInfo};
use yaxus_tokenizer::{EOS_ID, StreamDecoder, TAG_ID, Tokenizer};

use crate::memory::Memory;
use crate::prompt::Prompt;

/// Tokens that end a translation (not part of it).
const STOP: [u32; 2] = [EOS_ID, TAG_ID];

/// A request that has ended.
#[derive(Debug, Clone)]
pub struct Finished {
    pub text: String,
    pub finish: Finish,
    pub prompt_tokens: usize,
    pub generated: usize,
    /// How the prompt was started (KV reuse, response cache).
    pub start: StartInfo,
    /// When the request was handed to the engine.
    pub submitted: Instant,
    pub first_token: Option<Instant>,
    pub ended: Instant,
}

impl Finished {
    /// Submission to the end.
    pub fn elapsed(&self) -> Duration {
        self.ended - self.submitted
    }

    /// Submission to the first token (prompt processing); all of it if no token came.
    pub fn prefill(&self) -> Duration {
        self.first_token.unwrap_or(self.ended) - self.submitted
    }

    /// First token to the end.
    pub fn decode(&self) -> Duration {
        self.ended - self.first_token.unwrap_or(self.ended)
    }
}

/// What happened to a request during [`Session::step`].
pub enum Update<'u, T> {
    /// New text of the request. It may be empty (a token that completes no character
    /// yet); returning `false` for it cancels the request, which then ends with
    /// [`Finish::Cancelled`].
    Delta { tag: &'u T, text: &'u str },
    /// The request ended; its last text came as a `Delta` just before.
    Done { tag: T, done: Finished },
    /// The engine reported an error for the request.
    Failed { tag: T, message: String },
}

struct Live<T> {
    tag: T,
    dec: StreamDecoder,
    text: String,
    tokens: Vec<u32>,
    memo: u64,
    prompt_tokens: usize,
    submitted: Instant,
    first_token: Option<Instant>,
}

/// The requests of one engine, each with a tag `T` saying whose it is.
pub struct Session<'a, T> {
    engine: &'a mut dyn Generator,
    tok: &'a Tokenizer,
    memory: &'a mut Memory,
    live: HashMap<ReqId, Live<T>>,
}

impl<'a, T> Session<'a, T> {
    /// `engine` must have no requests of its own in flight.
    pub fn new(engine: &'a mut dyn Generator, tok: &'a Tokenizer, memory: &'a mut Memory) -> Self {
        Session { engine, tok, memory, live: HashMap::new() }
    }

    pub fn tokenizer(&self) -> &'a Tokenizer {
        self.tok
    }

    /// Hands `prompt` to the engine, with the translation memory's hint for it.
    pub fn submit(&mut self, prompt: Prompt, max_new_tokens: usize, tag: T) -> ReqId {
        let id = self.engine.fresh_id();
        let prompt_tokens = prompt.len();
        let (ids, memo) = prompt.into_parts();
        let mut req = Request::new(id, ids, max_new_tokens, STOP.to_vec());
        req.hint = self.memory.get(memo);
        self.engine.submit(req);
        let live = Live {
            tag,
            dec: StreamDecoder::new(),
            text: String::new(),
            tokens: Vec::new(),
            memo,
            prompt_tokens,
            submitted: Instant::now(),
            first_token: None,
        };
        self.live.insert(id, live);
        id
    }

    /// Runs one engine step and reports what it did to the requests. Returns whether
    /// there was work.
    pub fn step(&mut self, on: &mut dyn FnMut(Update<'_, T>) -> bool) -> Result<bool> {
        let Session { engine, tok, memory, live } = self;
        engine.step(&mut |ev| match ev {
            Event::Token { id, token } => {
                let Some(l) = live.get_mut(&id) else { return false };
                l.first_token.get_or_insert_with(Instant::now);
                l.tokens.push(token);
                let start = l.text.len();
                l.dec.push_into(tok, token, &mut l.text);
                on(Update::Delta { tag: &l.tag, text: &l.text[start..] })
            }
            Event::Done { id, finish, generated, start } => {
                let Some(mut l) = live.remove(&id) else { return true };
                if finish == Finish::Stop {
                    memory.put(l.memo, std::mem::take(&mut l.tokens));
                }
                let tail = l.dec.finish();
                if !tail.is_empty() {
                    on(Update::Delta { tag: &l.tag, text: &tail });
                    l.text += &tail;
                }
                let done = Finished {
                    text: l.text,
                    finish,
                    prompt_tokens: l.prompt_tokens,
                    generated,
                    start,
                    submitted: l.submitted,
                    first_token: l.first_token,
                    ended: Instant::now(),
                };
                on(Update::Done { tag: l.tag, done });
                true
            }
            Event::Error { id, message } => {
                if let Some(l) = live.remove(&id) {
                    on(Update::Failed { tag: l.tag, message });
                }
                true
            }
        })
    }

    /// No request is in flight.
    pub fn is_idle(&self) -> bool {
        self.live.is_empty()
    }

    /// Requests the engine has not started yet.
    pub fn waiting(&self) -> usize {
        self.engine.waiting()
    }

    /// Forgets every request (after a failed [`step`](Session::step)) and returns their
    /// tags, so that their owners can be told.
    pub fn abort_all(&mut self) -> Vec<T> {
        self.live.drain().map(|(_, l)| l.tag).collect()
    }
}
