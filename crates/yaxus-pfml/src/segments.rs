//! Splitting long text into prompts that fit a context budget.
//!
//! Paragraphs (blank lines) are kept whole when they fit. A paragraph that does not fit
//! is split at sentence endings, then at words, and the pieces are packed greedily up to
//! the token budget.

use yaxus_api::split_paragraphs;
use yaxus_tokenizer::Tokenizer;

/// One piece of the input to translate. `paragraph` is true for the first segment of each
/// paragraph (the caller joins translations with a blank line there, and with nothing
/// between sentences of one paragraph).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub text: String,
    pub paragraph: bool,
}

/// Splits `text` into segments of at most `budget` tokens each (measured with `tok`
/// without the prompt tags). The returned segments are never blank.
pub fn split_for_context(text: &str, tok: &Tokenizer, budget: usize) -> Vec<Segment> {
    let budget = budget.max(1);
    let mut out = Vec::new();
    for paragraph in split_paragraphs(text) {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }
        if count(tok, paragraph) <= budget {
            out.push(Segment { text: paragraph.to_string(), paragraph: true });
            continue;
        }

        let mut first = true;
        let mut current = String::new();
        for sentence in split_sentences(paragraph) {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            if count(tok, sentence) > budget {
                flush(&mut current, &mut first, &mut out);
                for piece in split_long(sentence, tok, budget) {
                    if piece.is_empty() {
                        continue;
                    }
                    out.push(Segment { text: piece, paragraph: first });
                    first = false;
                }
                continue;
            }
            let candidate = if current.is_empty() { sentence.to_string() } else { format!("{current} {sentence}") };
            if count(tok, &candidate) <= budget {
                current = candidate;
            } else {
                flush(&mut current, &mut first, &mut out);
                current = sentence.to_string();
            }
        }
        flush(&mut current, &mut first, &mut out);
    }
    out
}

fn count(tok: &Tokenizer, text: &str) -> usize {
    tok.encode(text, false).len()
}

fn flush(current: &mut String, first: &mut bool, out: &mut Vec<Segment>) {
    let text = current.trim();
    if !text.is_empty() {
        out.push(Segment { text: text.to_string(), paragraph: *first });
        *first = false;
        current.clear();
    }
}

/// Sentence endings keep their punctuation; a closing quote or bracket stays with the
/// sentence. A line break ends a sentence too.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if !matches!(c, '.' | '!' | '?' | '。' | '！' | '？' | '\n') {
            continue;
        }
        let mut end = i + c.len_utf8();
        while let Some(&(j, q)) = chars.peek() {
            if matches!(q, '"' | '\'' | '”' | '’' | ')' | ']' | '）' | '】' | '」' | '』') {
                chars.next();
                end = j + q.len_utf8();
            } else {
                break;
            }
        }
        // Japanese punctuation ends a sentence without a following space.
        let cjk_end = matches!(c, '。' | '！' | '？' | '\n');
        if cjk_end || chars.peek().is_none_or(|&(_, n)| n.is_whitespace()) {
            let sentence = text[start..end].trim();
            if !sentence.is_empty() {
                out.push(sentence.to_string());
            }
            start = end;
        }
    }
    let tail = text[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

/// Splits a sentence longer than the budget at spaces, then at characters.
fn split_long(sentence: &str, tok: &Tokenizer, budget: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for word in sentence.split_whitespace() {
        let candidate = if current.is_empty() { word.to_string() } else { format!("{current} {word}") };
        if count(tok, &candidate) <= budget {
            current = candidate;
            continue;
        }
        if !current.is_empty() {
            out.push(std::mem::take(&mut current));
        }
        if count(tok, word) <= budget {
            current = word.to_string();
            continue;
        }
        // A single "word" longer than the budget (a URL, a long identifier): split it by
        // characters. One character may itself exceed a tiny budget; it still has to go
        // somewhere.
        let mut piece = String::new();
        for ch in word.chars() {
            piece.push(ch);
            if count(tok, &piece) > budget && piece.chars().count() > 1 {
                piece.pop();
                if !piece.is_empty() {
                    out.push(std::mem::take(&mut piece));
                }
                piece.push(ch);
            }
        }
        if !piece.is_empty() {
            out.push(piece);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use yaxus_tokenizer::testing::plamo_like;

    #[test]
    fn sentences_keep_their_punctuation_and_closing_quotes() {
        assert_eq!(split_sentences("One. Two! Three?"), ["One.", "Two!", "Three?"]);
        assert_eq!(split_sentences("He said \"Go.\" Then left."), ["He said \"Go.\"", "Then left."]);
        assert_eq!(split_sentences("これは一つ。これは二つ。"), ["これは一つ。", "これは二つ。"]);
    }

    #[test]
    fn paragraphs_that_fit_stay_whole() {
        let tok = plamo_like(&[]);
        let segments = split_for_context("One.\n\nTwo.", &tok, 1000);
        assert_eq!(
            segments.iter().map(|s| (s.text.as_str(), s.paragraph)).collect::<Vec<_>>(),
            [("One.", true), ("Two.", true)]
        );
    }

    #[test]
    fn a_long_paragraph_is_split_at_sentences() {
        let tok = plamo_like(&[]);
        let text = "One two three. Four five six. Seven eight nine.";
        let segments = split_for_context(text, &tok, 8);
        assert!(segments.len() > 1, "{segments:?}");
        assert!(segments[0].paragraph);
        assert!(segments[1..].iter().all(|s| !s.paragraph), "{segments:?}");
        let joined = segments.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");
        for word in ["One", "three.", "Four", "nine."] {
            assert!(joined.contains(word), "{joined}");
        }
    }

    #[test]
    fn a_word_longer_than_the_budget_is_split() {
        let tok = plamo_like(&[]);
        let segments = split_for_context("abcdefghijklmnop", &tok, 4);
        assert!(segments.len() > 1, "{segments:?}");
        assert_eq!(segments.iter().map(|s| s.text.as_str()).collect::<String>(), "abcdefghijklmnop");
    }
}
