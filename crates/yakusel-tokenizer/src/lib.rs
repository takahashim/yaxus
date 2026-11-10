//! Pure Rust port of the PLaMo 3 tokenizer (`tokenization_plamo.py`).
//!
//! The reference implementation builds an Aho-Corasick automaton over reversed
//! suffixes and runs a Viterbi DP from the end of the string. This port keeps the
//! exact same scoring and tie-breaking rules, but finds the candidate tokens at each
//! position with a plain forward trie, which yields the identical candidate set:
//!
//! * every vocabulary token (except byte tokens) that is a prefix of `text[i..]`,
//!   with cost `-round_half_even(score * 1e4)`;
//! * among equal costs the longest token wins;
//! * a final fallback of length 1 with cost `1e7` that is emitted as UTF-8 byte tokens.
//!
//! Before the DP, the input is pre-processed exactly like the Python code: special
//! tokens `<|plamo:...|>` and runs of repeated characters / spaces are surrounded by
//! the private-use boundary character U+EE00, which forces token boundaries and is
//! dropped from the output.

use std::collections::HashMap;
use std::path::Path;

mod pfml;
mod stream;

pub use pfml::{GlossaryEntry, PromptError, TranslationPair, TranslationRequest, validate_lang};
pub use stream::StreamDecoder;

const INVALID_SCORE: i64 = -20_000_000;
const UNKNOWN_SCORE: i64 = -10_000_000;
const BOUNDARY_CHAR: char = '\u{EE00}';
const BOUNDARY_TOKEN_ID: u32 = 10_000_000;

/// `break_around_repeated_chars_threshold` from tokenizer_config.json.
const REPEATED_CHARS_THRESHOLD: usize = 4;
/// `break_around_consecutive_spaces_threshold` from tokenizer_config.json.
const CONSECUTIVE_SPACES_THRESHOLD: usize = 2;

pub const UNK_ID: u32 = 0;
pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 1; // tokenizer_config: eos_token = <|plamo:bos|>
pub const PAD_ID: u32 = 3;
pub const TAG_ID: u32 = 16;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("tokenizer.jsonl line {line}: {msg}")]
    Format { line: usize, msg: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Normal,
    Byte,
    Control,
    Unknown,
    Other,
}

#[derive(Debug, Clone, Copy)]
struct Node {
    /// (token id, integer score) when the path from the root spells a token.
    token: Option<(u32, i64)>,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    tokens: Vec<String>,
    kinds: Vec<TokenKind>,
    /// For byte tokens, the byte value they represent.
    byte_of: Vec<Option<u8>>,
    bytes: [u32; 256],
    nodes: Vec<Node>,
    /// Trie edges keyed by `(node << 32) | codepoint`.
    edges: HashMap<u64, u32>,
}

/// Python's `round()` (round half to even) applied to `score * 1e4`.
fn int_score(score: f64) -> i64 {
    (score * 1e4).round_ties_even() as i64
}

impl Tokenizer {
    /// Loads `tokenizer.jsonl` from a file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_jsonl(&text)
    }

    /// Loads the vocabulary from the contents of `tokenizer.jsonl`.
    pub fn from_jsonl(text: &str) -> Result<Self, Error> {
        let mut tokens = Vec::new();
        let mut kinds = Vec::new();
        let mut scores = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let fmt = |msg: &str| Error::Format {
                line: lineno + 1,
                msg: msg.to_string(),
            };
            let row: serde_json::Value =
                serde_json::from_str(line).map_err(|e| fmt(&e.to_string()))?;
            let row = row.as_array().ok_or_else(|| fmt("row is not an array"))?;
            let token = row
                .first()
                .and_then(|v| v.as_str())
                .ok_or_else(|| fmt("token is not a string"))?;
            let score = row
                .get(1)
                .and_then(|v| v.as_f64())
                .ok_or_else(|| fmt("score is not a number"))?;
            let kind = match row.get(2).and_then(|v| v.as_str()) {
                Some("NORMAL") => TokenKind::Normal,
                Some("BYTE") => TokenKind::Byte,
                Some("CONTROL") => TokenKind::Control,
                Some("UNKNOWN") => TokenKind::Unknown,
                _ => TokenKind::Other,
            };
            tokens.push(token.to_string());
            kinds.push(kind);
            scores.push(score);
        }
        Self::build(tokens, kinds, scores)
    }

    fn build(tokens: Vec<String>, kinds: Vec<TokenKind>, scores: Vec<f64>) -> Result<Self, Error> {
        let mut t = Tokenizer {
            byte_of: vec![None; tokens.len()],
            tokens,
            kinds,
            bytes: [0; 256],
            nodes: vec![Node { token: None }],
            edges: HashMap::new(),
        };
        let mut byte_set = [false; 256];
        let n = t.tokens.len();
        for id in 0..n {
            if t.kinds[id] == TokenKind::Byte {
                let tok = &t.tokens[id];
                let b = (tok.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>'))
                    .then(|| u8::from_str_radix(&tok[3..5], 16).ok())
                    .flatten()
                    .ok_or_else(|| Error::Format {
                        line: id + 1,
                        msg: format!("malformed byte token {tok:?}"),
                    })?;
                t.bytes[b as usize] = id as u32;
                t.byte_of[id] = Some(b);
                byte_set[b as usize] = true;
                continue;
            }
            let tok = t.tokens[id].clone();
            t.insert(&tok, id as u32, int_score(scores[id]));
        }
        // The boundary character is added last, as in the Python implementation.
        t.insert(&BOUNDARY_CHAR.to_string(), BOUNDARY_TOKEN_ID, 0);
        if let Some(b) = byte_set.iter().position(|x| !x) {
            return Err(Error::Format {
                line: 0,
                msg: format!("byte token <0x{b:02X}> is not set"),
            });
        }
        Ok(t)
    }

    fn insert(&mut self, token: &str, id: u32, score: i64) {
        let mut node = 0u32;
        for c in token.chars() {
            let key = ((node as u64) << 32) | c as u64;
            node = match self.edges.get(&key) {
                Some(&next) => next,
                None => {
                    let next = self.nodes.len() as u32;
                    self.nodes.push(Node { token: None });
                    self.edges.insert(key, next);
                    next
                }
            };
        }
        if node != 0 && score > INVALID_SCORE {
            // Later rows overwrite earlier ones, like the Python dicts.
            self.nodes[node as usize].token = Some((id, score));
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    pub fn token_kind(&self, id: u32) -> Option<TokenKind> {
        self.kinds.get(id as usize).copied()
    }

    /// Encodes text, optionally prefixing `<|plamo:bos|>` (the model expects it).
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let chars: Vec<char> = preprocess(text).chars().collect();
        let mut out = Vec::with_capacity(chars.len() + 1);
        if add_bos {
            out.push(BOS_ID);
        }
        self.viterbi(&chars, &mut out);
        out
    }

    fn viterbi(&self, data: &[char], out: &mut Vec<u32>) {
        let n = data.len();
        // costs[i]: minimum cost to encode data[i..]; path[i]: (length, token id or -1).
        let mut costs = vec![i64::MAX; n + 1];
        let mut path: Vec<(u32, i64)> = vec![(0, -1); n + 1];
        costs[n] = 0;
        for i in (0..n).rev() {
            let mut best = i64::MAX;
            let mut best_path = (0u32, -1i64);
            // Walk the trie forward. Lengths are visited in increasing order, so `<=`
            // makes the longest token win among ties (Python: decreasing order, `<`).
            let mut node = 0u32;
            for (k, &c) in data[i..].iter().enumerate() {
                match self.edges.get(&(((node as u64) << 32) | c as u64)) {
                    Some(&next) => node = next,
                    None => break,
                }
                if let Some((id, score)) = self.nodes[node as usize].token {
                    let len = k + 1;
                    let s = costs[i + len].saturating_sub(score);
                    if s <= best {
                        best = s;
                        best_path = (len as u32, id as i64);
                    }
                }
            }
            // Unknown-character fallback (sentinel row), only if strictly better.
            let s = costs[i + 1].saturating_sub(UNKNOWN_SCORE);
            if s < best {
                best = s;
                best_path = (1, -1);
            }
            costs[i] = best;
            path[i] = best_path;
        }

        let mut pos = 0;
        while pos < n {
            let (len, id) = path[pos];
            if id >= 0 {
                if id as u32 != BOUNDARY_TOKEN_ID {
                    out.push(id as u32);
                }
            } else {
                let mut buf = [0u8; 4];
                for &b in data[pos].encode_utf8(&mut buf).as_bytes() {
                    out.push(self.bytes[b as usize]);
                }
            }
            pos += len as usize;
        }
    }

    /// Returns the raw bytes a token stands for (special tokens yield their text).
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        let i = id as usize;
        match self.byte_of.get(i) {
            Some(Some(b)) => std::slice::from_ref(b),
            Some(None) => self.tokens[i].as_bytes(),
            None => &[],
        }
    }

    /// Decodes ids to text, replacing invalid UTF-8 with U+FFFD.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(self.token_bytes(id));
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Python `str.isspace()`, which is what `\s` matches in a `str` regex.
fn py_isspace(c: char) -> bool {
    matches!(c,
        '\t'..='\r' | '\x1c'..='\x1f' | ' ' | '\u{85}' | '\u{a0}' | '\u{1680}'
        | '\u{2000}'..='\u{200a}' | '\u{2028}' | '\u{2029}' | '\u{202f}' | '\u{205f}' | '\u{3000}')
}

/// Applies the two regex substitutions of `AhoCorasick.encode`:
/// 1. `(<\|plamo:[^|\s]{,64}\|>)` -> `\1`
/// 2. `((.)\2{3,}| {2,})` -> `\1`
fn preprocess(text: &str) -> String {
    let step1 = mark_special_tokens(text);
    mark_runs(&step1)
}

fn mark_special_tokens(text: &str) -> String {
    const PREFIX: &str = "<|plamo:";
    if !text.contains(PREFIX) {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len() + 16);
    let mut rest = text;
    while let Some(start) = rest.find(PREFIX) {
        let body = &rest[start + PREFIX.len()..];
        let mut count = 0usize;
        let mut end = 0usize; // byte offset within body after the class run
        for (off, c) in body.char_indices() {
            if c == '|' || py_isspace(c) {
                end = off;
                break;
            }
            count += 1;
            end = off + c.len_utf8();
        }
        if count <= 64 && body[end..].starts_with("|>") {
            let tok_end = start + PREFIX.len() + end + 2;
            out.push_str(&rest[..start]);
            out.push(BOUNDARY_CHAR);
            out.push_str(&rest[start..tok_end]);
            out.push(BOUNDARY_CHAR);
            rest = &rest[tok_end..];
        } else {
            // No match here; the regex engine retries from the next character.
            out.push_str(&rest[..start + 1]);
            rest = &rest[start + 1..];
        }
    }
    out.push_str(rest);
    out
}

fn mark_runs(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() + 8);
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let mut j = i + 1;
        while j < chars.len() && chars[j] == c {
            j += 1;
        }
        let run = j - i;
        // `.` does not match '\n'; a run that fails at its start also fails at every
        // later start, so matching whole runs is equivalent to the regex scan.
        let matched = (c != '\n' && run >= REPEATED_CHARS_THRESHOLD)
            || (c == ' ' && run >= CONSECUTIVE_SPACES_THRESHOLD);
        if matched {
            out.push(BOUNDARY_CHAR);
        }
        out.extend(&chars[i..j]);
        if matched {
            out.push(BOUNDARY_CHAR);
        }
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_special_tokens() {
        let b = BOUNDARY_CHAR;
        assert_eq!(
            mark_special_tokens("a<|plamo:msg|>b"),
            format!("a{b}<|plamo:msg|>{b}b")
        );
        assert_eq!(mark_special_tokens("<|plamo:a b|>"), "<|plamo:a b|>");
        assert_eq!(
            mark_special_tokens("<|plamo:<|plamo:x|>"),
            format!("<|plamo:{b}<|plamo:x|>{b}")
        );
        let long = format!("<|plamo:{}|>", "x".repeat(65));
        assert_eq!(mark_special_tokens(&long), long);
    }

    #[test]
    fn preprocess_runs() {
        let b = BOUNDARY_CHAR;
        assert_eq!(mark_runs("aaab"), "aaab");
        assert_eq!(mark_runs("aaaab"), format!("{b}aaaa{b}b"));
        assert_eq!(mark_runs("a  b"), format!("a{b}  {b}b"));
        assert_eq!(mark_runs("\n\n\n\n"), "\n\n\n\n");
    }

    #[test]
    fn rounding_is_half_even() {
        assert_eq!(int_score(-99.993), -999930);
        assert_eq!(int_score(0.0), 0);
        assert_eq!(int_score(-20.0), -200000);
    }
}
