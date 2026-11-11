//! JSON types of the yaxus HTTP API (`POST /v1/translate` etc.).

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

/// Missing fields read as 0, so clients and servers of different versions still agree.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    /// Time from dequeue to the end of generation.
    pub elapsed_ms: u64,
    /// Time from dequeue to the first generated token (prompt processing).
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

/// `POST /v1/translate/document`: several segments (paragraphs) of one document.
///
/// The segments are split into chains of consecutive segments; a chain is translated in
/// order with its earlier segments as history (up to `history_chars`), and the chains run
/// in parallel. One chain gives the most consistent translation; more chains are faster.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentRequest {
    pub segments: Vec<String>,
    pub target_lang: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_lang: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub form: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub glossary: Vec<Pair>,
    /// Chains translated in parallel; 0 or absent: one per 4 segments, at most 8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chains: Option<usize>,
    /// History budget per chain in characters (default `DEFAULT_HISTORY_CHARS`, 0: none).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub history_chars: Option<usize>,
    /// Pass the neighbouring source segments as context (useful without history).
    #[serde(default)]
    pub neighbor_context: bool,
    /// Per segment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_new_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
}

/// Missing fields read as 0, as for [`Usage`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DocumentUsage {
    pub segments: usize,
    pub chains: usize,
    pub prompt_tokens: usize,
    /// Prompt tokens whose KV was reused from earlier requests.
    pub reused_tokens: usize,
    pub completion_tokens: usize,
    pub elapsed_ms: u64,
    pub queue_ms: u64,
}

/// Response for `stream: false`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentResponse {
    pub segments: Vec<String>,
    pub usage: DocumentUsage,
}

/// One `data:` payload of the SSE stream for `stream: true`. Segments progress in
/// parallel, so deltas of different segments interleave.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DocumentEvent {
    Delta { segment: usize, delta: String },
    SegmentEnd { segment: usize, end: bool },
    Done { done: bool, usage: DocumentUsage },
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
/// budget; it then keeps just the newest pairs fitting in a quarter of the budget (at
/// least the newest one, unless that alone exceeds the whole budget).
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

    /// The pairs of `all` (every earlier segment, oldest first) to send now. `all` is
    /// expected to grow between calls; a shorter one starts a new sequence.
    pub fn select<'a>(&mut self, all: &'a [Pair]) -> &'a [Pair] {
        if self.budget_chars == 0 {
            return &[];
        }
        let chars = |p: &Pair| p.source.chars().count() + p.target.chars().count();
        if self.start > all.len() {
            self.start = 0;
        }
        let total: usize = all[self.start..].iter().map(chars).sum();
        if total > self.budget_chars {
            let mut start = all.len() - 1;
            let mut kept = chars(&all[start]);
            if kept > self.budget_chars {
                // Not even the newest pair fits: send none rather than exceed the budget.
                self.start = all.len();
                return &[];
            }
            // Keep the newest pairs within a quarter of the budget (at least one).
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
    fn history_window_moves_to_a_quarter() {
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
    fn history_window_never_exceeds_the_budget_with_one_pair() {
        let pair = |n: usize| Pair { source: "x".repeat(n), target: String::new() };
        let all = [pair(100), pair(100), pair(900), pair(100)];
        let mut w = HistoryWindow::new(800);
        assert_eq!(w.select(&all[..2]).len(), 2);
        // The 900-character pair alone is over the budget: no history.
        assert!(w.select(&all[..3]).is_empty());
        // Later pairs start a new window after it.
        assert_eq!(w.select(&all[..4]), &all[3..]);
    }

    #[test]
    fn history_window_restarts_on_a_shorter_list() {
        let pair = |n: usize| Pair { source: "x".repeat(n), target: String::new() };
        let long: Vec<Pair> = (0..10).map(|_| pair(100)).collect();
        let mut w = HistoryWindow::new(800);
        w.select(&long[..9]);
        w.select(&long);
        // A new, shorter sequence gets its whole history again.
        assert_eq!(w.select(&long[..3]).len(), 3);
    }

    #[test]
    fn stream_events_round_trip() {
        fn round<T: Serialize + for<'de> Deserialize<'de>>(v: &T) -> T {
            serde_json::from_str(&serde_json::to_string(v).unwrap()).unwrap()
        }
        let usage = Usage { prompt_tokens: 1, completion_tokens: 2, elapsed_ms: 3, prefill_ms: 4, queue_ms: 5 };
        assert!(
            matches!(round(&StreamEvent::Delta { delta: "a".into() }), StreamEvent::Delta { delta } if delta == "a")
        );
        assert!(matches!(
            round(&StreamEvent::Done { done: true, usage: usage.clone() }),
            StreamEvent::Done { done: true, usage: Usage { prefill_ms: 4, .. } }
        ));
        assert!(
            matches!(round(&StreamEvent::Error { error: "e".into() }), StreamEvent::Error { error } if error == "e")
        );

        let du = DocumentUsage { segments: 2, reused_tokens: 7, ..Default::default() };
        assert!(matches!(
            round(&DocumentEvent::Delta { segment: 1, delta: "b".into() }),
            DocumentEvent::Delta { segment: 1, delta } if delta == "b"
        ));
        assert!(matches!(
            round(&DocumentEvent::SegmentEnd { segment: 1, end: true }),
            DocumentEvent::SegmentEnd { segment: 1, end: true }
        ));
        assert!(matches!(
            round(&DocumentEvent::Done { done: true, usage: du }),
            DocumentEvent::Done { done: true, usage: DocumentUsage { segments: 2, reused_tokens: 7, .. } }
        ));
        assert!(matches!(round(&DocumentEvent::Error { error: "e".into() }), DocumentEvent::Error { .. }));
    }

    #[test]
    fn usage_tolerates_missing_fields() {
        // An older or newer peer may not send every usage field.
        let ev: DocumentEvent = serde_json::from_str(r#"{"done":true,"usage":{"segments":3}}"#).unwrap();
        assert!(matches!(ev, DocumentEvent::Done { usage: DocumentUsage { segments: 3, chains: 0, .. }, .. }));
        let ev: StreamEvent = serde_json::from_str(r#"{"done":true,"usage":{"prompt_tokens":5}}"#).unwrap();
        assert!(matches!(ev, StreamEvent::Done { usage: Usage { prompt_tokens: 5, elapsed_ms: 0, .. }, .. }));
    }

    #[test]
    fn paragraphs() {
        assert_eq!(split_paragraphs("a\nb\n\n \n\nc\r\n"), vec!["a\nb", "c"]);
        assert!(split_paragraphs("\n\n").is_empty());
    }
}
