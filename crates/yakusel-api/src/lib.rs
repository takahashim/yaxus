//! JSON types of the yakusel HTTP API (`POST /v1/translate` etc.).

use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_NEW_TOKENS: usize = 512;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranslateRequest {
    pub text: String,
    pub target_lang: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_lang: Option<String>,
    /// `polite` or `plain`; only with `target_lang = "ja"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub form: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub glossary: Vec<Pair>,
    /// Earlier source/translation pairs, oldest first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub history: Vec<Pair>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pair {
    pub source: String,
    pub target: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Time from dequeue to the end of generation.
    pub elapsed_ms: u64,
    /// Time from dequeue to the first generated token (prompt processing).
    #[serde(default)]
    pub prefill_ms: u64,
    /// Time spent waiting in the queue.
    pub queue_ms: u64,
}

/// Response for `stream: false`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateResponse {
    pub text: String,
    pub usage: Usage,
}

/// One `data:` payload of the SSE stream for `stream: true`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StreamEvent {
    Delta { delta: String },
    Done { done: bool, usage: Usage },
    Error { error: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Health {
    pub status: String,
    pub model: String,
    pub backend: String,
    pub quant: String,
    pub queue: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Language {
    pub code: String,
    pub name: String,
}

/// Languages listed in the PLaMo 3 Translate model card (+ `ja-easy`).
pub const LANGUAGES: &[(&str, &str)] = &[
    ("ja", "Japanese"),
    ("ja-easy", "Easy Japanese"),
    ("en", "English"),
    ("zh-hans", "Chinese (Simplified)"),
    ("zh-hant", "Chinese (Traditional)"),
    ("ko", "Korean"),
    ("ar", "Arabic"),
    ("az", "Azerbaijani"),
    ("be", "Belarusian"),
    ("bg", "Bulgarian"),
    ("bn", "Bengali"),
    ("ca", "Catalan"),
    ("ceb", "Cebuano"),
    ("de", "German"),
    ("es", "Spanish"),
    ("et", "Estonian"),
    ("eu", "Basque"),
    ("fa", "Persian"),
    ("fr", "French"),
    ("gu", "Gujarati"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hr", "Croatian"),
    ("hu", "Hungarian"),
    ("hy", "Armenian"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("jv", "Javanese"),
    ("ka", "Georgian"),
    ("kk", "Kazakh"),
    ("km", "Khmer"),
    ("lt", "Lithuanian"),
    ("lv", "Latvian"),
    ("ml", "Malayalam"),
    ("mr", "Marathi"),
    ("ms", "Malay"),
    ("my", "Burmese"),
    ("ne", "Nepali"),
    ("nl", "Dutch"),
    ("pa", "Punjabi"),
    ("pl", "Polish"),
    ("ru", "Russian"),
    ("si", "Sinhala"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("sr", "Serbian"),
    ("sw", "Swahili"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("th", "Thai"),
    ("tl", "Tagalog"),
    ("ur", "Urdu"),
    ("uz", "Uzbek"),
    ("vi", "Vietnamese"),
];

/// Default history budget in characters (sources and translations of the kept pairs).
pub const DEFAULT_HISTORY_CHARS: usize = 4000;

/// Chooses the history sent with each segment of a document.
///
/// The kept history starts at a fixed pair and only moves forward when it exceeds the
/// budget; it then keeps just the newest pairs fitting in a quarter of the budget.
/// Consecutive prompts therefore share a long prefix, which the server reuses instead of
/// recomputing (a sliding window of the last N pairs would change the prefix on every
/// segment). Each move costs one recomputation of the kept quarter, so moving rarely and
/// keeping little is cheapest: about a third of a pair per segment on average.
#[derive(Debug, Clone)]
pub struct HistoryWindow {
    start: usize,
    /// Budget in characters; 0 disables the history.
    pub budget_chars: usize,
}

impl HistoryWindow {
    pub fn new(budget_chars: usize) -> Self {
        HistoryWindow { start: 0, budget_chars }
    }

    /// The pairs of `all` (every earlier segment, oldest first) to send now.
    pub fn select<'a>(&mut self, all: &'a [Pair]) -> &'a [Pair] {
        if self.budget_chars == 0 {
            return &all[all.len()..];
        }
        let chars = |p: &Pair| p.source.chars().count() + p.target.chars().count();
        self.start = self.start.min(all.len());
        let total: usize = all[self.start..].iter().map(chars).sum();
        if total > self.budget_chars {
            // Keep the newest pairs within a quarter of the budget (at least one).
            let mut start = all.len() - 1;
            let mut kept = chars(&all[start]);
            while start > self.start && kept + chars(&all[start - 1]) <= self.budget_chars / 4 {
                start -= 1;
                kept += chars(&all[start]);
            }
            self.start = start;
        }
        &all[self.start..]
    }
}

/// Splits text into paragraphs (separated by blank lines) for segment-wise translation.
/// Lines inside a paragraph keep their line breaks; translations are joined with `\n\n`.
pub fn split_paragraphs(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !cur.is_empty() {
                out.push(cur.join("\n"));
                cur.clear();
            }
        } else {
            cur.push(line);
        }
    }
    if !cur.is_empty() {
        out.push(cur.join("\n"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn languages_match_model_card() {
        // 53 languages + the ja-easy variant.
        assert_eq!(LANGUAGES.len(), 54);
    }

    #[test]
    fn history_window_moves_in_halves() {
        let pair = |n: usize| Pair { source: "x".repeat(n), target: String::new() };
        let all: Vec<Pair> = (0..10).map(|_| pair(100)).collect();
        let mut w = HistoryWindow::new(800);
        assert_eq!(w.select(&all[..8]).len(), 8);
        // 9 pairs (900 chars) exceed the budget: keep the newest within 200 chars.
        assert_eq!(w.select(&all[..9]).len(), 2);
        // The start stays put while the window fits.
        assert_eq!(w.select(&all[..10]).len(), 3);
        assert_eq!(HistoryWindow::new(0).select(&all).len(), 0);
    }

    #[test]
    fn paragraphs() {
        assert_eq!(split_paragraphs("a\nb\n\n \n\nc\r\n"), vec!["a\nb", "c"]);
        assert!(split_paragraphs("\n\n").is_empty());
    }
}
