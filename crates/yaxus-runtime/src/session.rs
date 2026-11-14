//! Translations in flight on one engine: what every caller of the engine does for each
//! request, in one place. A request is submitted with the hint the translation memory
//! has for it; its tokens are decoded into text as they come; when it ends at a stop
//! token its output goes into the memory. The server's worker,
//! [`run_document`](crate::document::run_document) and [`run_request`](crate::run_request)
//! differ only in where the text goes, which is what the tag `T` of each request tells.

use std::collections::HashMap;
use std::time::Duration;

use web_time::Instant;

use anyhow::Result;
use yaxus_model::{Event, Finish, Generator, ReqId, Request, StartInfo};
use yaxus_tokenizer::{EOS_ID, StreamDecoder, TAG_ID, Tokenizer};

use crate::memory::Memory;
use crate::prompt::Prompt;

/// Tokens that end a translation (not part of it).
pub const STOP_TOKENS: [u32; 2] = [EOS_ID, TAG_ID];

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
        let mut req = Request::new(id, ids, max_new_tokens, STOP_TOKENS.to_vec());
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
    pub async fn step_async(&mut self, on: &mut dyn FnMut(Update<'_, T>) -> bool) -> Result<bool> {
        let Session { engine, tok, memory, live } = self;
        engine.step_async(&mut |ev| on_event(live, tok, memory, ev, on)).await
    }

    /// Cancels the requests whose tag `pred` accepts, waiting or running, in the engine
    /// too: each ends at the next step as `Update::Done` with [`Finish::Cancelled`].
    /// Returns how many were cancelled.
    pub fn cancel_where(&mut self, mut pred: impl FnMut(&T) -> bool) -> usize {
        let ids: Vec<ReqId> = self.live.iter().filter(|(_, l)| pred(&l.tag)).map(|(id, _)| *id).collect();
        ids.into_iter().filter(|&id| self.engine.cancel(id)).count()
    }

    /// No request is in flight.
    pub fn is_idle(&self) -> bool {
        self.live.is_empty()
    }

    /// Requests the engine has not started yet.
    pub fn waiting(&self) -> usize {
        self.engine.waiting()
    }

    /// Drops every request, in the engine too (after a failed [`step_async`](Session::step_async)),
    /// and returns their tags, so that their owners can be told.
    pub fn abort_all(&mut self) -> Vec<T> {
        self.engine.abort_all();
        self.live.drain().map(|(_, l)| l.tag).collect()
    }

    /// Gives up on the engine (after a panic in [`step_async`](Session::step_async), when its state
    /// cannot be trusted and it must not be called again): returns the tags of the
    /// requests in flight without telling the engine.
    pub fn abandon(self) -> Vec<T> {
        self.live.into_values().map(|l| l.tag).collect()
    }
}

/// What the engine's event `ev` does to the requests in `live`; returns the answer to
/// give the engine (`false` cancels the request of a token).
fn on_event<T>(
    live: &mut HashMap<ReqId, Live<T>>,
    tok: &Tokenizer,
    memory: &mut Memory,
    ev: Event,
    on: &mut dyn FnMut(Update<'_, T>) -> bool,
) -> bool {
    match ev {
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
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    /// Plays back scripted events, one list per step, and records what it was given.
    #[derive(Default)]
    struct Script {
        steps: VecDeque<Vec<Event>>,
        submitted: Vec<Request>,
        /// What the session answered to each event.
        answers: Vec<bool>,
        next_id: ReqId,
        aborted: bool,
    }

    impl Generator for Script {
        fn submit(&mut self, req: Request) {
            self.submitted.push(req);
        }
        fn step_async<'a>(
            &'a mut self,
            emit: &'a mut dyn FnMut(Event) -> bool,
        ) -> yaxus_model::LocalBoxFuture<'a, Result<bool>> {
            let Some(events) = self.steps.pop_front() else { return Box::pin(std::future::ready(Ok(false))) };
            for ev in events {
                self.answers.push(emit(ev));
            }
            Box::pin(std::future::ready(Ok(true)))
        }
        fn fresh_id(&mut self) -> ReqId {
            self.next_id += 1;
            self.next_id
        }
        fn waiting(&self) -> usize {
            0
        }
        fn active(&self) -> usize {
            0
        }
        fn abort_all(&mut self) {
            self.aborted = true;
        }
        fn backend_name(&self) -> String {
            "script".into()
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

    const HEL: u32 = 17 + 256;
    const LO: u32 = HEL + 1;
    /// The byte token of `b`.
    fn byte(b: u8) -> u32 {
        17 + b as u32
    }

    /// The PLaMo special-token layout with the tokens `Hel` and `lo`.
    fn tokenizer() -> Tokenizer {
        yaxus_tokenizer::testing::plamo_like(&[("Hel", 0.0), ("lo", 0.0)])
    }

    fn token(id: ReqId, token: u32) -> Event {
        Event::Token { id, token }
    }

    fn done(id: ReqId, finish: Finish, generated: usize) -> Event {
        Event::Done { id, finish, generated, start: StartInfo::default() }
    }

    /// The updates of one step, as text: `tag:delta`, `tag:done:text` or `tag:failed`.
    fn run_step(session: &mut Session<'_, &'static str>, cancel_on: Option<&str>) -> Vec<String> {
        let mut seen = Vec::new();
        yaxus_model::poll_once(session.step_async(&mut |u| match u {
            Update::Delta { tag, text } => {
                seen.push(format!("{tag}:{text}"));
                cancel_on != Some(text)
            }
            Update::Done { tag, done } => {
                seen.push(format!("{tag}:done:{}", done.text));
                true
            }
            Update::Failed { tag, .. } => {
                seen.push(format!("{tag}:failed"));
                true
            }
        }))
        .unwrap();
        seen
    }

    #[test]
    fn tokens_become_text_and_a_stop_goes_into_memory() {
        let tok = tokenizer();
        let mut memory = Memory::new(8);
        memory.put(42, vec![HEL]);
        let mut engine = Script::default();
        // "é" is two byte tokens: the first completes no character yet.
        engine.steps.push_back(vec![token(1, HEL), token(1, LO), token(1, byte(0xC3))]);
        engine.steps.push_back(vec![token(1, byte(0xA9)), done(1, Finish::Stop, 4)]);
        let mut session = Session::new(&mut engine, &tok, &mut memory);
        session.submit(Prompt::from_parts(vec![5, 6], 42), 32, "a");
        assert_eq!(run_step(&mut session, None), ["a:Hel", "a:lo", "a:"]);
        assert_eq!(run_step(&mut session, None), ["a:é", "a:done:Helloé"]);
        assert!(session.is_idle());
        drop(session);
        let req = &engine.submitted[0];
        assert_eq!((req.prompt.as_slice(), req.max_new_tokens), (&[5, 6][..], 32));
        assert_eq!(req.stop, STOP_TOKENS);
        assert_eq!(req.hint, [HEL], "the memory's tokens are the hint");
        assert_eq!(memory.get(42), [HEL, LO, byte(0xC3), byte(0xA9)]);
    }

    #[test]
    fn the_tail_comes_before_done_and_only_stops_are_remembered() {
        let tok = tokenizer();
        let mut memory = Memory::new(8);
        let mut engine = Script::default();
        // Ends inside a character: the decoder's tail comes out as a last Delta.
        engine.steps.push_back(vec![token(1, HEL), token(1, byte(0xC3)), done(1, Finish::Length, 2)]);
        let mut session = Session::new(&mut engine, &tok, &mut memory);
        session.submit(Prompt::from_parts(vec![5], 7), 2, "a");
        let seen = run_step(&mut session, None);
        assert_eq!(seen[..2], ["a:Hel", "a:"]);
        assert!(seen[2].starts_with("a:") && seen[2].len() > 2, "tail delta: {seen:?}");
        assert!(seen[3].starts_with("a:done:Hel"), "{seen:?}");
        drop(session);
        assert!(memory.get(7).is_empty(), "a length cut is not remembered");
    }

    #[test]
    fn a_false_answer_cancels_and_unknown_requests_are_refused() {
        let tok = tokenizer();
        let mut memory = Memory::new(8);
        let mut engine = Script::default();
        engine.steps.push_back(vec![
            token(1, HEL),
            token(2, LO),
            token(9, LO),
            Event::Error { id: 9, message: "x".into() },
        ]);
        engine.steps.push_back(vec![done(1, Finish::Cancelled, 1), Event::Error { id: 2, message: "boom".into() }]);
        let mut session = Session::new(&mut engine, &tok, &mut memory);
        session.submit(Prompt::from_parts(vec![5], 1), 8, "a");
        session.submit(Prompt::from_parts(vec![5], 2), 8, "b");
        assert_eq!(run_step(&mut session, Some("Hel")), ["a:Hel", "b:lo"]);
        assert_eq!(run_step(&mut session, None), ["a:done:Hel", "b:failed"]);
        assert!(session.is_idle());
        drop(session);
        // Hel cancelled (false); lo kept; unknown request 9 refused; its error ignored.
        assert_eq!(engine.answers, [false, true, false, true, true, true]);
        assert!(memory.get(1).is_empty(), "a cancelled request is not remembered");
    }

    #[test]
    fn abort_returns_the_tags_in_flight() {
        let tok = tokenizer();
        let mut memory = Memory::new(8);
        let mut engine = Script::default();
        let mut session = Session::new(&mut engine, &tok, &mut memory);
        session.submit(Prompt::from_parts(vec![5], 1), 8, "a");
        session.submit(Prompt::from_parts(vec![5], 2), 8, "b");
        let mut tags = session.abort_all();
        tags.sort();
        assert_eq!(tags, ["a", "b"]);
        assert!(session.is_idle());
        drop(session);
        assert!(engine.aborted, "the engine drops its requests too");
    }

    #[test]
    fn abandon_leaves_the_engine_alone() {
        let tok = tokenizer();
        let mut memory = Memory::new(8);
        let mut engine = Script::default();
        let mut session = Session::new(&mut engine, &tok, &mut memory);
        session.submit(Prompt::from_parts(vec![5], 1), 8, "a");
        assert_eq!(session.abandon(), ["a"]);
        assert!(!engine.aborted);
    }
}
