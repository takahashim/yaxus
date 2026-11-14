//! Splitting long text into prompts that fit a context budget.
//!
//! Paragraphs (blank lines) are kept whole when they fit. A paragraph that does not fit
//! is split at sentence endings, then at words, and the pieces are packed greedily up to
//! the token budget.
//!
//! Each sentence and word is tokenized once: a packed segment counts as the sum of its
//! pieces plus one token per join (the budget is a target, and the caller measures the
//! final prompt exactly), so packing stays linear in the text.

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
        let mut current_tokens = 0;
        for sentence in split_sentences(paragraph) {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }
            let n = count(tok, sentence);
            if n > budget {
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
            let joined = if current.is_empty() { n } else { current_tokens + n + 1 };
            if joined <= budget {
                current = join_sentences(&current, sentence);
                current_tokens = joined;
            } else {
                flush(&mut current, &mut first, &mut out);
                current = sentence.to_string();
                current_tokens = n;
            }
        }
        flush(&mut current, &mut first, &mut out);
    }
    out
}

/// `current` and the next sentence of its paragraph: Japanese sentences follow one
/// another directly, other sentences with a space.
fn join_sentences(current: &str, sentence: &str) -> String {
    if current.is_empty() {
        sentence.to_string()
    } else if current.ends_with(['。', '！', '？', '」', '』', '）']) {
        format!("{current}{sentence}")
    } else {
        format!("{current} {sentence}")
    }
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
/// sentence. A line break ends a sentence too (a quote after it opens the next line).
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = text.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if !matches!(c, '.' | '!' | '?' | '。' | '！' | '？' | '\n') {
            continue;
        }
        let mut end = i + c.len_utf8();
        while c != '\n'
            && let Some(&(j, q)) = chars.peek()
        {
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
    let mut current_tokens = 0;
    for word in sentence.split_whitespace() {
        let n = count(tok, word);
        if n > budget {
            // A single "word" longer than the budget (a URL, a long identifier, Japanese
            // without punctuation).
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            out.extend(split_chars(word, tok, budget));
            continue;
        }
        let joined = if current.is_empty() { n } else { current_tokens + n + 1 };
        if joined <= budget {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
            current_tokens = joined;
        } else {
            out.push(std::mem::replace(&mut current, word.to_string()));
            current_tokens = n;
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// A word longer than the budget, cut at characters: each piece is the longest prefix
/// of the rest within the budget. The search looks only near the next cut, so a piece
/// costs a few encodes of about its own length, whatever the length of the rest. It
/// starts from the length of the piece before, and alternates a guess from the tokens
/// per character just measured (text of one kind is nearly uniform, so this lands close)
/// with a step that halves the bracket (or doubles the length while no prefix has been
/// too long), which bounds it at twice a bisection. A piece has at least one character,
/// which may itself exceed a tiny budget.
fn split_chars(word: &str, tok: &Tokenizer, budget: usize) -> Vec<String> {
    // The byte offset after the first `n` characters of `s` (all of `s` if shorter).
    let end_of = |s: &str, n: usize| s.char_indices().nth(n).map_or(s.len(), |(i, _)| i);
    let mut out = Vec::new();
    let mut rest = word;
    let mut guess = budget.max(1);
    while !rest.is_empty() {
        // `fits` characters are known to fit (0: none yet), `too_long` known not to.
        let (mut fits, mut too_long) = (0usize, None::<usize>);
        let mut probe = guess;
        for step in 1.. {
            let end = end_of(rest, probe);
            let tokens = count(tok, &rest[..end]);
            if tokens <= budget {
                if end == rest.len() {
                    out.push(rest.to_string());
                    return out;
                }
                fits = probe;
            } else {
                too_long = Some(probe);
            }
            let hi = too_long.unwrap_or(usize::MAX);
            if hi - fits <= 1 {
                break;
            }
            probe = if step % 2 == 1 {
                (probe * budget / tokens.max(1)).clamp(fits + 1, hi - 1)
            } else {
                match too_long {
                    Some(hi) => fits + (hi - fits) / 2,
                    None => fits * 2,
                }
            };
        }
        let n = fits.max(1);
        let end = end_of(rest, n);
        out.push(rest[..end].to_string());
        rest = &rest[end..];
        guess = n;
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
    fn a_quote_after_a_line_break_opens_the_next_sentence() {
        assert_eq!(split_sentences("He said.\n\"Hello,\" she replied."), ["He said.", "\"Hello,\" she replied."]);
        assert_eq!(split_sentences("「行く。」\n（次）"), ["「行く。」", "（次）"]);
    }

    #[test]
    fn a_long_word_is_cut_into_pieces_within_the_budget() {
        let tok = plamo_like(&[]);
        let word: String = "あいうえおかきくけこ".chars().cycle().take(5000).collect();
        let pieces = split_for_context(&word, &tok, 300);
        assert!(pieces.len() > 1);
        assert!(
            pieces.iter().all(|p| count(&tok, &p.text) <= 300),
            "{:?}",
            pieces.iter().map(|p| count(&tok, &p.text)).collect::<Vec<_>>()
        );
        assert_eq!(pieces.iter().map(|p| p.text.as_str()).collect::<String>(), word);
        // Longest prefixes: every piece but the last is full up to one character.
        for p in &pieces[..pieces.len() - 1] {
            assert!(count(&tok, &p.text) > 300 - 4, "{}", count(&tok, &p.text));
        }
    }

    #[test]
    fn japanese_sentences_are_packed_without_spaces() {
        let tok = plamo_like(&[]);
        // Two sentences and the join's token, not three.
        let budget = 2 * count(&tok, "これは一つ。") + 1;
        let segments = split_for_context("これは一つ。これは二つ。これは三つ。", &tok, budget);
        assert_eq!(
            segments.iter().map(|s| (s.text.as_str(), s.paragraph)).collect::<Vec<_>>(),
            [("これは一つ。これは二つ。", true), ("これは三つ。", false)]
        );
    }

    #[test]
    fn a_word_longer_than_the_budget_is_split() {
        let tok = plamo_like(&[]);
        let segments = split_for_context("abcdefghijklmnop", &tok, 4);
        assert!(segments.len() > 1, "{segments:?}");
        assert_eq!(segments.iter().map(|s| s.text.as_str()).collect::<String>(), "abcdefghijklmnop");
    }
}
