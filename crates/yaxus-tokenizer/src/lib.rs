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

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::path::Path;

mod pfml;
mod stream;

pub use pfml::{GlossaryEntry, PromptError, TranslationPair, TranslationRequest, validate_lang};
pub use stream::StreamDecoder;

const INVALID_SCORE: i64 = -20_000_000;
const UNKNOWN_SCORE: i64 = -10_000_000;
const BOUNDARY_CHAR: char = '\u{EE00}';
const BOUNDARY_TOKEN_ID: u32 = 10_000_000;
/// `path` entry for the unknown-character fallback (emitted as UTF-8 byte tokens).
const NO_TOKEN: u32 = u32::MAX;

/// `break_around_repeated_chars_threshold` from tokenizer_config.json.
const REPEATED_CHARS_THRESHOLD: usize = 4;
/// `break_around_consecutive_spaces_threshold` from tokenizer_config.json.
const CONSECUTIVE_SPACES_THRESHOLD: usize = 2;

pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 1; // tokenizer_config: eos_token = <|plamo:bos|>
pub const TAG_ID: u32 = 16;

/// Token ids the rest of the crate (and its users) hard-code, checked at load time
/// so that a vocabulary with a different layout fails loudly instead of breaking
/// prompt construction or stop conditions.
const FIXED_TOKENS: [(u32, &str); 2] = [(BOS_ID, "<|plamo:bos|>"), (TAG_ID, "<|plamo:tag|>")];

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to read {path}: {source}")]
    Io { path: String, source: std::io::Error },
    /// A line of `tokenizer.jsonl` could not be parsed.
    #[error("tokenizer.jsonl line {line}: {msg}")]
    Format { line: usize, msg: String },
    /// The vocabulary parsed, but is not a usable PLaMo vocabulary.
    #[error("invalid vocabulary: {0}")]
    Vocab(String),
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

/// Hasher for the trie's `(node << 32) | codepoint` keys. The default SipHash dominates
/// the Viterbi inner loop; one 128-bit multiply mixes both halves into every bit.
#[derive(Default)]
struct EdgeHasher(u64);

impl Hasher for EdgeHasher {
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u64 keys are hashed");
    }

    fn write_u64(&mut self, x: u64) {
        let p = (x as u128) * 0x9e37_79b9_7f4a_7c15;
        self.0 = (p as u64) ^ ((p >> 64) as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type EdgeMap = HashMap<u64, u32, BuildHasherDefault<EdgeHasher>>;

#[derive(Debug, Clone)]
pub struct Tokenizer {
    tokens: Vec<String>,
    kinds: Vec<TokenKind>,
    /// For byte tokens, the byte value they represent.
    byte_of: Vec<Option<u8>>,
    bytes: [u32; 256],
    nodes: Vec<Node>,
    /// Trie edges keyed by `(node << 32) | codepoint`.
    edges: EdgeMap,
}

/// Python's `round()` (round half to even) applied to `score * 1e4`.
fn int_score(score: f64) -> i64 {
    (score * 1e4).round_ties_even() as i64
}

impl Tokenizer {
    /// Loads `tokenizer.jsonl` from a file.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).map_err(|source| Error::Io { path: path.display().to_string(), source })?;
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
            let fmt = |msg: &str| Error::Format { line: lineno + 1, msg: msg.to_string() };
            let row: serde_json::Value = serde_json::from_str(line).map_err(|e| fmt(&e.to_string()))?;
            let row = row.as_array().ok_or_else(|| fmt("row is not an array"))?;
            let token = row.first().and_then(|v| v.as_str()).ok_or_else(|| fmt("token is not a string"))?;
            let score = row.get(1).and_then(|v| v.as_f64()).ok_or_else(|| fmt("score is not a number"))?;
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

    /// Builds the tokenizer from a vocabulary given as parallel arrays (e.g. the
    /// `tokenizer.ggml.*` metadata of a GGUF file).
    pub fn from_vocab(tokens: Vec<String>, kinds: Vec<TokenKind>, scores: Vec<f64>) -> Result<Self, Error> {
        if tokens.len() != kinds.len() || tokens.len() != scores.len() {
            return Err(Error::Vocab("vocabulary arrays differ in length".into()));
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
            edges: EdgeMap::default(),
        };
        for (id, want) in FIXED_TOKENS {
            let got = t.tokens.get(id as usize);
            if got.map(String::as_str) != Some(want) {
                return Err(Error::Vocab(format!("token id {id} must be {want:?}, found {got:?}")));
            }
        }
        let mut byte_set = [false; 256];
        for (id, &score) in scores.iter().enumerate() {
            if t.kinds[id] == TokenKind::Byte {
                let b = parse_byte_token(&t.tokens[id])
                    .ok_or_else(|| Error::Vocab(format!("token id {id}: malformed byte token {:?}", t.tokens[id])))?;
                if byte_set[b as usize] {
                    return Err(Error::Vocab(format!("token id {id}: byte token <0x{b:02X}> is defined twice")));
                }
                t.bytes[b as usize] = id as u32;
                t.byte_of[id] = Some(b);
                byte_set[b as usize] = true;
                continue;
            }
            let tok = t.tokens[id].clone();
            t.insert(&tok, id as u32, int_score(score));
        }
        // The boundary character is added last, as in the Python implementation.
        t.insert(&BOUNDARY_CHAR.to_string(), BOUNDARY_TOKEN_ID, 0);
        if let Some(b) = byte_set.iter().position(|x| !x) {
            return Err(Error::Vocab(format!("byte token <0x{b:02X}> is not set")));
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
    ///
    /// As in the Python tokenizer, U+EE00 in `text` is taken as a token boundary and
    /// produces no token, so `decode(encode(text))` drops it.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let chars = preprocess(text);
        let mut out = Vec::with_capacity(chars.len() + 1);
        if add_bos {
            out.push(BOS_ID);
        }
        self.viterbi(&chars, &mut out);
        out
    }

    fn viterbi(&self, data: &[char], out: &mut Vec<u32>) {
        let n = data.len();
        // costs[i]: minimum cost to encode data[i..]; path[i]: (length, token id or NO_TOKEN).
        let mut costs = vec![i64::MAX; n + 1];
        let mut path: Vec<(u32, u32)> = vec![(0, NO_TOKEN); n + 1];
        costs[n] = 0;
        for i in (0..n).rev() {
            let mut best = i64::MAX;
            let mut best_path = (0u32, NO_TOKEN);
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
                        best_path = (len as u32, id);
                    }
                }
            }
            // Unknown-character fallback (sentinel row), only if strictly better.
            let s = costs[i + 1].saturating_sub(UNKNOWN_SCORE);
            if s < best {
                best = s;
                best_path = (1, NO_TOKEN);
            }
            costs[i] = best;
            path[i] = best_path;
        }

        let mut pos = 0;
        while pos < n {
            let (len, id) = path[pos];
            if id != NO_TOKEN {
                if id != BOUNDARY_TOKEN_ID {
                    out.push(id);
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

/// Parses a byte token `<0xNN>` (exactly two hex digits).
fn parse_byte_token(tok: &str) -> Option<u8> {
    let hex = tok.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 || !hex.bytes().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
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
fn preprocess(text: &str) -> Vec<char> {
    mark_runs(&mark_special_tokens(text))
}

fn mark_special_tokens(text: &str) -> Cow<'_, str> {
    const PREFIX: &str = "<|plamo:";
    if !text.contains(PREFIX) {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len() + 16);
    let mut rest = text;
    while let Some(start) = rest.find(PREFIX) {
        let body = &rest[start + PREFIX.len()..];
        // Byte offset in `body` of the first character outside `[^|\s]`, if it comes
        // within 64 characters; a 65th class character means no match.
        let mut end = None;
        for (n, (off, c)) in body.char_indices().enumerate() {
            if c == '|' || py_isspace(c) {
                end = Some(off);
                break;
            }
            if n == 64 {
                break;
            }
        }
        if let Some(end) = end.filter(|&e| body[e..].starts_with("|>")) {
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
    Cow::Owned(out)
}

/// Marks the runs and returns the characters the Viterbi DP works on.
fn mark_runs(text: &str) -> Vec<char> {
    let mut out = Vec::with_capacity(text.len() + 8);
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        let mut run = 1;
        while chars.next_if_eq(&c).is_some() {
            run += 1;
        }
        // `.` does not match '\n'; a run that fails at its start also fails at every
        // later start, so matching whole runs is equivalent to the regex scan.
        let matched =
            (c != '\n' && run >= REPEATED_CHARS_THRESHOLD) || (c == ' ' && run >= CONSECUTIVE_SPACES_THRESHOLD);
        if matched {
            out.push(BOUNDARY_CHAR);
        }
        out.extend(std::iter::repeat_n(c, run));
        if matched {
            out.push(BOUNDARY_CHAR);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_special_tokens() {
        let b = BOUNDARY_CHAR;
        assert_eq!(mark_special_tokens("a<|plamo:msg|>b"), format!("a{b}<|plamo:msg|>{b}b"));
        assert_eq!(mark_special_tokens("<|plamo:a b|>"), "<|plamo:a b|>");
        assert_eq!(mark_special_tokens("<|plamo:<|plamo:x|>"), format!("<|plamo:{b}<|plamo:x|>{b}"));
        let long = format!("<|plamo:{}|>", "x".repeat(65));
        assert_eq!(mark_special_tokens(&long), long);
    }

    #[test]
    fn preprocess_special_tokens_length_limit() {
        let b = BOUNDARY_CHAR;
        let max = format!("<|plamo:{}|>", "x".repeat(64));
        assert_eq!(mark_special_tokens(&max), format!("{b}{max}{b}"));
        // Multi-byte characters count as one each.
        let max = format!("<|plamo:{}|>", "あ".repeat(64));
        assert_eq!(mark_special_tokens(&max), format!("{b}{max}{b}"));
        let over = format!("<|plamo:{}|>", "あ".repeat(65));
        assert_eq!(mark_special_tokens(&over), over);
        // Unterminated, or ended by whitespace or a lone '|'.
        for s in ["<|plamo:", "<|plamo:abc", "<|plamo:ab c|>", "<|plamo:ab|", "<|plamo:ab|x"] {
            assert_eq!(mark_special_tokens(s), s);
        }
        assert!(matches!(mark_special_tokens("no special tokens"), Cow::Borrowed(_)));
    }

    #[test]
    fn preprocess_runs() {
        let b = BOUNDARY_CHAR;
        let runs = |s: &str| mark_runs(s).into_iter().collect::<String>();
        assert_eq!(runs(""), "");
        assert_eq!(runs("aaab"), "aaab");
        assert_eq!(runs("aaaab"), format!("{b}aaaa{b}b"));
        assert_eq!(runs("a  b"), format!("a{b}  {b}b"));
        assert_eq!(runs("\n\n\n\n"), "\n\n\n\n");
        assert_eq!(runs("ああああ  x    "), format!("{b}ああああ{b}{b}  {b}x{b}    {b}"));
    }

    /// A minimal PLaMo-shaped vocabulary: control tokens at ids 0..=16 (bos at 1,
    /// tag at 16), the 256 byte tokens, then `normal` with their scores.
    fn vocab(normal: &[(&str, f64)]) -> (Vec<String>, Vec<TokenKind>, Vec<f64>) {
        let mut tokens = Vec::new();
        let mut kinds = Vec::new();
        let mut scores = Vec::new();
        for id in 0..=TAG_ID {
            tokens.push(match id {
                BOS_ID => "<|plamo:bos|>".to_string(),
                TAG_ID => "<|plamo:tag|>".to_string(),
                _ => format!("<|plamo:c{id}|>"),
            });
            kinds.push(TokenKind::Control);
            scores.push(0.0);
        }
        for b in 0..=255u8 {
            tokens.push(format!("<0x{b:02X}>"));
            kinds.push(TokenKind::Byte);
            scores.push(0.0);
        }
        for &(t, score) in normal {
            tokens.push(t.to_string());
            kinds.push(TokenKind::Normal);
            scores.push(score);
        }
        (tokens, kinds, scores)
    }

    fn tokenizer(normal: &[(&str, f64)]) -> Tokenizer {
        let (t, k, s) = vocab(normal);
        Tokenizer::from_vocab(t, k, s).unwrap()
    }

    /// Encodes and maps the ids back to token strings, for readable assertions.
    fn pieces(tok: &Tokenizer, text: &str) -> Vec<String> {
        let ids = tok.encode(text, false);
        assert_eq!(tok.decode(&ids), text.replace(BOUNDARY_CHAR, ""));
        ids.iter().map(|&id| tok.id_to_token(id).unwrap().to_string()).collect()
    }

    #[test]
    fn viterbi_picks_lowest_cost() {
        let tok = tokenizer(&[("a", -1.0), ("b", -1.0), ("ab", -3.0), ("abc", -10.0), ("c", -1.0)]);
        // a+b (2.0) beats ab (3.0); abc (10.0) loses to a+b+c (3.0).
        assert_eq!(pieces(&tok, "ab"), ["a", "b"]);
        assert_eq!(pieces(&tok, "abc"), ["a", "b", "c"]);
    }

    #[test]
    fn viterbi_ties_prefer_longest_token() {
        let tok = tokenizer(&[("a", -1.0), ("b", -1.0), ("ab", -2.0)]);
        assert_eq!(pieces(&tok, "ab"), ["ab"]);
        assert_eq!(pieces(&tok, "abab"), ["ab", "ab"]);
    }

    #[test]
    fn unknown_characters_fall_back_to_utf8_bytes() {
        let tok = tokenizer(&[("a", -1.0)]);
        assert_eq!(pieces(&tok, "aé"), ["a", "<0xC3>", "<0xA9>"]);
        assert_eq!(pieces(&tok, "😀"), ["<0xF0>", "<0x9F>", "<0x98>", "<0x80>"]);
        assert!(tok.encode("", false).is_empty());
        assert_eq!(tok.encode("", true), [BOS_ID]);
    }

    #[test]
    fn fallback_wins_only_when_strictly_better() {
        // Score -1000 gives exactly UNKNOWN_SCORE: the token wins the tie.
        let tok = tokenizer(&[("x", -1000.0)]);
        assert_eq!(pieces(&tok, "x"), ["x"]);
        // Scores at or below INVALID_SCORE are never used.
        let tok = tokenizer(&[("x", -2000.0)]);
        assert_eq!(pieces(&tok, "x"), ["<0x78>"]);
    }

    #[test]
    fn later_duplicate_rows_win() {
        let tok = tokenizer(&[("ab", -9.0), ("a", -1.0), ("b", -1.0), ("ab", -1.0)]);
        let ids = tok.encode("ab", false);
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0] as usize, tok.vocab_size() - 1);
    }

    #[test]
    fn boundaries_split_tokens_and_are_dropped() {
        let b = BOUNDARY_CHAR;
        let tok = tokenizer(&[("a", -1.0), ("b", -1.0), ("ab", -1.0), ("  ", -1.0), (" ", -1.0)]);
        assert_eq!(pieces(&tok, &format!("a{b}b")), ["a", "b"]);
        assert_eq!(pieces(&tok, &format!("{b}{b}")), Vec::<String>::new());
        // Consecutive spaces are isolated from their neighbours.
        let tok = tokenizer(&[("a", -1.0), ("a ", -1.0), (" ", -1.0), ("  ", -1.0)]);
        assert_eq!(pieces(&tok, "a  a"), ["a", "  ", "a"]);
        assert_eq!(pieces(&tok, "a a"), ["a ", "a"]);
    }

    #[test]
    fn special_tokens_are_isolated() {
        // Without the boundaries, "x<" + "|plamo:tag|>" (cost 0.1) would beat
        // "x" + "<|plamo:tag|>" (cost 1.0).
        let tok = tokenizer(&[("x", -1.0), ("x<", -0.1), ("|plamo:tag|>", 0.0)]);
        assert_eq!(tok.encode("x<|plamo:tag|>x", false)[1], TAG_ID);
        assert_eq!(pieces(&tok, "x<|plamo:tag|>x"), ["x", "<|plamo:tag|>", "x"]);
    }

    #[test]
    fn rejects_bad_vocabularies() {
        let err = |(t, k, s): (Vec<String>, Vec<TokenKind>, Vec<f64>)| {
            Tokenizer::from_vocab(t, k, s).unwrap_err().to_string()
        };
        let base = vocab(&[]);
        let byte = TAG_ID as usize + 1 + 0x41;

        let mut v = base.clone();
        v.0[TAG_ID as usize] = "<|plamo:msg|>".into();
        assert!(err(v).contains("token id 16 must be \"<|plamo:tag|>\""));

        let mut v = base.clone();
        v.0.truncate(BOS_ID as usize);
        v.1.truncate(BOS_ID as usize);
        v.2.truncate(BOS_ID as usize);
        assert!(err(v).contains("token id 1 must be"));

        for bad in ["<0x+1>", "<0x4G>", "<0x041>", "<0xé>", "0x41"] {
            let mut v = base.clone();
            v.0[byte] = bad.into();
            assert!(err(v).contains("malformed byte token"), "{bad}");
        }

        let mut v = base.clone();
        v.0[byte] = "<0x42>".into();
        assert!(err(v).contains("<0x42> is defined twice"));

        let mut v = base.clone();
        v.1[byte] = TokenKind::Normal;
        assert!(err(v).contains("<0x41> is not set"));

        let mut v = base.clone();
        v.2.pop();
        assert!(err(v).contains("differ in length"));
    }

    #[test]
    fn stream_decoder_matches_decode() {
        let tok = tokenizer(&[("あ", -1.0), ("ab", -1.0)]);
        let byte = |b: u8| tok.bytes[b as usize];
        let normal = [tok.encode("あ", false)[0], tok.encode("ab", false)[0], BOS_ID];
        // Byte tokens biased towards UTF-8 lead/continuation bytes, so that split,
        // truncated and invalid sequences all occur, mixed with whole tokens.
        let pool: Vec<u32> = [0x41, 0x80, 0x81, 0xBF, 0xC3, 0xA9, 0xE3, 0x81, 0x82, 0xF0, 0x9F, 0x98, 0xFF]
            .iter()
            .map(|&b| byte(b))
            .chain(normal)
            .collect();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for len in 0..200 {
            let ids: Vec<u32> = (0..len % 24)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    pool[(state % pool.len() as u64) as usize]
                })
                .collect();
            let (mut a, mut b) = (StreamDecoder::new(), StreamDecoder::new());
            let (mut via_push, mut via_into) = (String::new(), String::new());
            for &id in &ids {
                let s = a.push(&tok, id);
                let before = via_into.len();
                b.push_into(&tok, id, &mut via_into);
                // push_into appends exactly what push returns.
                assert_eq!(&via_into[before..], s, "{ids:?}");
                via_push += &s;
            }
            via_push += &a.finish();
            via_into += &b.finish();
            assert_eq!(via_push, tok.decode(&ids), "{ids:?}");
            assert_eq!(via_into, via_push, "{ids:?}");
        }
    }

    #[test]
    fn jsonl_errors_report_the_line() {
        let e = Tokenizer::from_jsonl("[\"a\", 0.0]\n\n[\"b\"]\n").unwrap_err();
        assert_eq!(e.to_string(), "tokenizer.jsonl line 3: score is not a number");
    }

    #[test]
    fn rounding_is_half_even() {
        assert_eq!(int_score(-99.993), -999930);
        assert_eq!(int_score(0.0), 0);
        assert_eq!(int_score(-20.0), -200000);
    }
}
