//! How likely the model finds a text as the translation of a request: each piece of the
//! text with its log-probability and what the model would have written there instead.

use anyhow::Result;
use yaxus_model::SyncGenerator;
use yaxus_pfml::ValidRequest;
use yaxus_tokenizer::{SPECIAL_PREFIX, Tokenizer};

use crate::STOP_TOKENS;
use crate::prompt::Prompt;

/// What the model would have written at a place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Choice {
    /// A token of text.
    Text(String),
    /// The end of the output (a stop token): the model would have stopped there.
    End,
    /// Another special token, or a byte that is part of a character.
    Other,
}

impl Choice {
    fn of(tok: &Tokenizer, id: u32) -> Self {
        if STOP_TOKENS.contains(&id) {
            return Choice::End;
        }
        match std::str::from_utf8(tok.token_bytes(id)) {
            Ok(t) if !t.starts_with(SPECIAL_PREFIX) => Choice::Text(t.to_string()),
            _ => Choice::Other,
        }
    }

    /// The text, if it is one.
    pub fn text(&self) -> Option<&str> {
        match self {
            Choice::Text(t) => Some(t),
            _ => None,
        }
    }
}

impl std::fmt::Display for Choice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Choice::Text(t) => f.write_str(t.trim_end()),
            Choice::End => f.write_str("(end)"),
            Choice::Other => f.write_str("?"),
        }
    }
}

/// A piece of the scored text: one token, or several when a character spans tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    /// Byte range in the text, on character boundaries.
    pub start: usize,
    pub end: usize,
    /// The log-probability of the piece (the sum of its tokens').
    pub logprob: f32,
    /// The model's most likely token at the piece's start, and its log-probability.
    pub best: Choice,
    pub best_logprob: f32,
}

/// Scores `output` as the model's translation for `req` (see
/// [`SyncGenerator::score`]): the pieces cover `output` in order.
pub fn score_text(
    engine: &mut dyn SyncGenerator,
    tok: &Tokenizer,
    req: &ValidRequest,
    output: &str,
) -> Result<Vec<Piece>> {
    let prompt = Prompt::new(tok, req).into_ids();
    let forced = tok.encode(output, false);
    let scores = engine.score(&prompt, &forced)?;
    anyhow::ensure!(scores.len() == forced.len(), "{} scores for {} tokens", scores.len(), forced.len());
    let mut pieces: Vec<Piece> = Vec::new();
    let mut pos = 0;
    // A piece that does not end on a character boundary takes the next token.
    let mut open = false;
    for (&t, s) in forced.iter().zip(&scores) {
        let end = pos + tok.token_bytes(t).len();
        match pieces.last_mut() {
            Some(p) if open => {
                p.end = end;
                p.logprob += s.logprob;
            }
            _ => pieces.push(Piece {
                start: pos,
                end,
                logprob: s.logprob,
                best: Choice::of(tok, s.best),
                best_logprob: s.best_logprob,
            }),
        }
        open = !output.is_char_boundary(end.min(output.len()));
        pos = end;
    }
    anyhow::ensure!(pos == output.len(), "the tokens of the text are {pos} bytes, the text {}", output.len());
    Ok(pieces)
}
