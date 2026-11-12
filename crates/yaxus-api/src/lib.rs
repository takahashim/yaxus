//! JSON types of the yaxus HTTP API (`POST /v1/translate` etc.), and the document
//! helpers its client and server share ([`HistoryWindow`], [`split_paragraphs`]).

use std::num::NonZeroUsize;

use serde::{Deserialize, Serialize};

pub const DEFAULT_MAX_NEW_TOKENS: usize = 512;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TranslateRequest {
    pub text: String,
    pub target_lang: String,
    /// Detected from the text when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_lang: Option<String>,
    /// Only with `target_lang = "ja"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub form: Option<Form>,
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

/// The Japanese politeness a translation into `ja` may ask for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Form {
    Polite,
    Plain,
}

impl Form {
    pub fn as_str(self) -> &'static str {
        match self {
            Form::Polite => "polite",
            Form::Plain => "plain",
        }
    }
}

impl std::fmt::Display for Form {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Form {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "polite" => Ok(Form::Polite),
            "plain" => Ok(Form::Plain),
            _ => Err(format!("unknown form {s:?} (expected polite or plain)")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Pair {
    pub source: String,
    pub target: String,
}

/// Token counts and times of a request (of all segments, for a document). Missing fields
/// read as 0.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Usage {
    pub prompt_tokens: usize,
    /// Prompt tokens whose KV was reused from earlier requests.
    pub reused_tokens: usize,
    pub completion_tokens: usize,
    /// Time spent waiting in the queue.
    pub queue_ms: u64,
    /// Time from dequeue to the first generated token (prompt processing).
    pub prefill_ms: u64,
    /// Time from dequeue to the end of generation.
    pub elapsed_ms: u64,
}

/// Response for `stream: false`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslateResponse {
    pub text: String,
    pub usage: Usage,
}

/// One `data:` payload of the SSE stream for `stream: true`, tagged by `type`:
/// any number of `delta`, then one `done` or `error`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    Delta { text: String },
    Done { usage: Usage },
    Error { error: ApiError },
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
    pub form: Option<Form>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub glossary: Vec<Pair>,
    /// Chains translated in parallel; absent: one per 4 segments, at most 8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chains: Option<NonZeroUsize>,
    /// History budget per chain in characters; absent: `DEFAULT_HISTORY_CHARS`, 0: none.
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

/// [`Usage`] of all segments, and how the document was split.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DocumentUsage {
    pub segments: usize,
    pub chains: usize,
    /// (`prefill_ms` is the time to the document's first generated token.)
    #[serde(flatten)]
    pub total: Usage,
}

/// Response for `stream: false`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentResponse {
    pub segments: Vec<String>,
    pub usage: DocumentUsage,
}

/// One `data:` payload of the SSE stream for `stream: true`, tagged by `type`. Segments
/// progress in parallel, so deltas of different segments interleave; each segment ends
/// with `segment_done`, and the document with one `done` or `error`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentEvent {
    Delta { segment: usize, text: String },
    SegmentDone { segment: usize },
    Done { usage: DocumentUsage },
    Error { error: ApiError },
}

/// What went wrong, for programs (`message` is for people).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// A malformed or invalid request (400).
    InvalidRequest,
    /// The prompt does not fit the server's context (400).
    PromptTooLong,
    /// A missing or wrong bearer token (401).
    Unauthorized,
    /// The queue is full; retry later (429).
    QueueFull,
    /// The inference thread has stopped; the server must be restarted (503).
    Unavailable,
    /// Generation failed (500).
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApiError {
    pub code: ErrorCode,
    pub message: String,
    /// The segment of a document request it is about.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segment: Option<usize>,
}

impl ApiError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        ApiError { code, message: message.into(), segment: None }
    }

    pub fn in_segment(self, segment: usize) -> Self {
        ApiError { segment: Some(segment), ..self }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.segment {
            Some(s) => write!(f, "segment {s}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

/// The body of every error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ApiError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Ok,
    /// The inference thread has stopped (`/healthz` answers 503).
    Stopped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Health {
    pub status: HealthStatus,
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
            return &[];
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
    fn forms_are_a_closed_set() {
        let r: TranslateRequest = serde_json::from_str(r#"{"text":"x","target_lang":"ja","form":"polite"}"#).unwrap();
        assert_eq!(r.form, Some(Form::Polite));
        for bad in [r#""casual""#, r#""""#, r#""Polite""#] {
            let json = format!(r#"{{"text":"x","target_lang":"ja","form":{bad}}}"#);
            assert!(serde_json::from_str::<TranslateRequest>(&json).is_err(), "{bad}");
        }
        assert_eq!("plain".parse::<Form>(), Ok(Form::Plain));
        assert!("".parse::<Form>().is_err());
    }

    #[test]
    fn zero_chains_are_rejected() {
        let doc = |chains: &str| format!(r#"{{"segments":["a"],"target_lang":"en","chains":{chains}}}"#);
        assert_eq!(serde_json::from_str::<DocumentRequest>(&doc("2")).unwrap().chains, NonZeroUsize::new(2));
        assert!(serde_json::from_str::<DocumentRequest>(&doc("0")).is_err());
    }

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
    fn events_are_tagged_by_type() {
        fn json<T: Serialize>(v: &T) -> serde_json::Value {
            serde_json::to_value(v).unwrap()
        }
        fn round<T: Serialize + for<'de> Deserialize<'de>>(v: &T) -> T {
            serde_json::from_value(json(v)).unwrap()
        }
        let usage = Usage {
            prompt_tokens: 1,
            reused_tokens: 6,
            completion_tokens: 2,
            queue_ms: 5,
            prefill_ms: 4,
            elapsed_ms: 3,
        };
        let error = ApiError::new(ErrorCode::PromptTooLong, "too long").in_segment(3);
        let single = [
            StreamEvent::Delta { text: "a".into() },
            StreamEvent::Done { usage: usage.clone() },
            StreamEvent::Error { error: error.clone() },
        ];
        for ev in &single {
            assert_eq!(&round(ev), ev);
        }
        assert_eq!(json(&single[0]), serde_json::json!({"type": "delta", "text": "a"}));
        assert_eq!(
            json(&single[2]),
            serde_json::json!({"type": "error", "error": {"code": "prompt_too_long", "message": "too long", "segment": 3}})
        );

        let du = DocumentUsage { segments: 2, chains: 1, total: usage };
        let doc = [
            DocumentEvent::Delta { segment: 1, text: "b".into() },
            DocumentEvent::SegmentDone { segment: 1 },
            DocumentEvent::Done { usage: du },
            DocumentEvent::Error { error: ApiError::new(ErrorCode::Internal, "e") },
        ];
        for ev in &doc {
            assert_eq!(&round(ev), ev);
        }
        assert_eq!(json(&doc[1]), serde_json::json!({"type": "segment_done", "segment": 1}));
        // The document's usage is flat, as a single request's with two more fields.
        assert_eq!(json(&doc[2])["usage"]["reused_tokens"], 6);
        assert_eq!(json(&doc[2])["usage"]["segments"], 2);
        // Untagged or unknown events are errors, not a guess.
        assert!(serde_json::from_str::<StreamEvent>(r#"{"done":true,"usage":{}}"#).is_err());
        assert!(serde_json::from_str::<StreamEvent>(r#"{"type":"finished"}"#).is_err());
    }

    #[test]
    fn errors_and_health_status() {
        let body = ErrorResponse { error: ApiError::new(ErrorCode::QueueFull, "busy") };
        assert_eq!(
            serde_json::to_value(&body).unwrap(),
            serde_json::json!({"error": {"code": "queue_full", "message": "busy"}})
        );
        assert_eq!(ApiError::new(ErrorCode::Internal, "x").in_segment(2).to_string(), "segment 2: x");
        assert_eq!(serde_json::to_value(HealthStatus::Stopped).unwrap(), "stopped");
    }

    #[test]
    fn usage_tolerates_missing_fields() {
        let ev: DocumentEvent = serde_json::from_str(r#"{"type":"done","usage":{"segments":3}}"#).unwrap();
        assert!(matches!(ev, DocumentEvent::Done { usage: DocumentUsage { segments: 3, chains: 0, .. } }));
        let ev: StreamEvent = serde_json::from_str(r#"{"type":"done","usage":{"prompt_tokens":5}}"#).unwrap();
        assert!(matches!(ev, StreamEvent::Done { usage: Usage { prompt_tokens: 5, elapsed_ms: 0, .. } }));
    }

    #[test]
    fn paragraphs() {
        assert_eq!(split_paragraphs("a\nb\n\n \n\nc\r\n"), vec!["a\nb", "c"]);
        assert!(split_paragraphs("\n\n").is_empty());
    }
}
