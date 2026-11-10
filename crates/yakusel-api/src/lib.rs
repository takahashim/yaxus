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
    fn paragraphs() {
        assert_eq!(split_paragraphs("a\nb\n\n \n\nc\r\n"), vec!["a\nb", "c"]);
        assert!(split_paragraphs("\n\n").is_empty());
    }
}
