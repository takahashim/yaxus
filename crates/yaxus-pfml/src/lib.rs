//! PFML3 prompts of PLaMo 3 Translate, a direct port of `chat_template.jinja`.
//!
//! A [`TranslationRequest`] is the unchecked input. [`TranslationRequest::validate`]
//! turns it into a [`ValidRequest`], whose [`render`](ValidRequest::render) cannot fail.

pub mod segments;

use yaxus_tokenizer::{SPECIAL_PREFIX, py_isspace};

/// Glossary entries and history pairs (source, translation) are the API's pairs, and
/// the form is the API's.
pub use yaxus_api::{Form, Pair};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TranslationRequest {
    pub text: String,
    pub target_lang: String,
    pub source_lang: Option<String>,
    /// Only valid when `target_lang == "ja"`.
    pub form: Option<Form>,
    pub context: Option<String>,
    pub glossary: Vec<Pair>,
    /// Earlier source/translation pairs, oldest first.
    pub history: Vec<Pair>,
}

/// Why a request has no prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptErrorKind {
    /// A field the template does not accept.
    Invalid,
    /// The prompt does not fit the context it is for.
    TooLong,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct PromptError {
    pub kind: PromptErrorKind,
    pub message: String,
}

impl PromptError {
    pub fn invalid(message: impl Into<String>) -> Self {
        PromptError { kind: PromptErrorKind::Invalid, message: message.into() }
    }

    pub fn too_long(message: impl Into<String>) -> Self {
        PromptError { kind: PromptErrorKind::TooLong, message: message.into() }
    }
}

/// Whether `s` is empty or only whitespace, as the prompt template sees it (Jinja's
/// `trim` is Python's `str.strip()`, which also strips U+001C..U+001F).
pub fn is_blank(s: &str) -> bool {
    s.chars().all(py_isspace)
}

fn check_text(value: &str, name: &str, allow_empty: bool) -> Result<(), PromptError> {
    if !allow_empty && is_blank(value) {
        return Err(PromptError::invalid(format!("{name} must not be empty.")));
    }
    if value.contains(SPECIAL_PREFIX) {
        return Err(PromptError::invalid(format!(
            "{name} must not contain reserved PFML tokens ({SPECIAL_PREFIX}...)."
        )));
    }
    Ok(())
}

/// Checks a PFML3 language code (`en`, `ja`, `zh-hans`, `ja-easy`, ...).
fn check_lang(value: &str, name: &str) -> Result<(), PromptError> {
    check_text(value, name, false)?;
    let parts: Vec<&str> = value.split('-').collect();
    let bad_shape = parts.len() > 2
        || !(2..=3).contains(&parts[0].chars().count())
        || (parts.len() == 2 && parts[1].chars().count() != 4);
    if bad_shape {
        return Err(PromptError::invalid(format!(
            "{name} must be a lowercase language code, e.g. en, ja, zh-hans, or ja-easy."
        )));
    }
    if !value.chars().all(|c| c.is_ascii_lowercase() || c == '-') {
        return Err(PromptError::invalid(format!("{name} must use lowercase language codes without region subtags.")));
    }
    Ok(())
}

/// The model card's spelling `ja(easy)` of `ja-easy`.
fn alias(lang: &str) -> &str {
    if lang == "ja(easy)" { "ja-easy" } else { lang }
}

/// Checks a text to translate (or a side of a history pair): not blank, no reserved
/// tokens. The template calls them all messages.
fn check_message(text: &str) -> Result<(), PromptError> {
    check_text(text, "message content", false)
}

/// Checks a pair that a request may have in its history: both sides are messages.
pub fn validate_history_pair(pair: &Pair) -> Result<(), PromptError> {
    check_message(&pair.source)?;
    check_message(&pair.target)
}

/// What [`TranslationRequest::validate_options`] checks, in normalized form.
struct Options {
    target_lang: String,
    source_lang: Option<String>,
}

impl TranslationRequest {
    /// Checks everything [`validate`](Self::validate) checks except the text and the
    /// history: the languages, the form, the context and the glossary.
    pub fn validate_options(&self) -> Result<(), PromptError> {
        self.options().map(|_| ())
    }

    fn options(&self) -> Result<Options, PromptError> {
        let target_lang = alias(&self.target_lang);
        check_lang(target_lang, "target_lang")?;
        let source_lang = self.source_lang.as_deref().map(alias);
        if let Some(s) = source_lang {
            check_lang(s, "source_lang")?;
        }
        if let Some(c) = &self.context {
            check_text(c, "context", true)?;
        }
        if self.form.is_some() && target_lang != "ja" {
            return Err(PromptError::invalid("form requires target_lang=\"ja\"."));
        }
        for term in &self.glossary {
            check_text(&term.source, "glossary source", false)?;
            check_text(&term.target, "glossary target", false)?;
        }
        Ok(Options { target_lang: target_lang.into(), source_lang: source_lang.map(String::from) })
    }

    /// Checks the whole request, normalizing the language aliases.
    pub fn validate(self) -> Result<ValidRequest, PromptError> {
        self.history.iter().try_for_each(validate_history_pair)?;
        check_message(&self.text)?;
        let Options { target_lang, source_lang } = self.options()?;
        Ok(ValidRequest { req: TranslationRequest { target_lang, source_lang, ..self } })
    }
}

/// A request that passed [`TranslationRequest::validate`]: its language aliases are
/// normalized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidRequest {
    req: TranslationRequest,
}

impl ValidRequest {
    pub fn text(&self) -> &str {
        &self.req.text
    }

    /// The target language, `ja(easy)` spelled `ja-easy`.
    pub fn target_lang(&self) -> &str {
        &self.req.target_lang
    }

    /// The source language, `ja(easy)` spelled `ja-easy`.
    pub fn source_lang(&self) -> Option<&str> {
        self.req.source_lang.as_deref()
    }

    pub fn form(&self) -> Option<Form> {
        self.req.form
    }

    /// Earlier source/translation pairs, oldest first.
    pub fn history(&self) -> &[Pair] {
        &self.req.history
    }

    /// The same request without history (still valid).
    pub fn without_history(mut self) -> Self {
        self.req.history.clear();
        self
    }

    /// Renders the PFML3 prompt, including the leading `<|plamo:bos|>`.
    /// Tokenize the result with `add_bos = false`.
    pub fn render(&self) -> String {
        let r = &self.req;
        let mut p = String::from("<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation");
        if let Some(c) = r.context.as_deref().filter(|c| !c.is_empty()) {
            p += "<|plamo:tag|>context<|plamo:msg|>";
            p += c;
        }
        let input_header = |p: &mut String| {
            *p += "<|plamo:tag|>input";
            if let Some(s) = &r.source_lang {
                *p += "<|plamo:key|>lang<|plamo:val|>";
                *p += s;
            }
        };
        for pair in &r.history {
            input_header(&mut p);
            p += "<|plamo:msg|>";
            p += &pair.source;
            p += "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>";
            p += &r.target_lang;
            p += "<|plamo:msg|>";
            p += &pair.target;
        }
        if !r.glossary.is_empty() {
            p += "<|plamo:tag|>glossary";
            for term in &r.glossary {
                p += "<|plamo:index|>";
                p += &term.source;
                p += "<|plamo:item|>";
                p += &term.target;
            }
        }
        input_header(&mut p);
        p += "<|plamo:msg|>";
        p += &r.text;
        p += "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>";
        p += &r.target_lang;
        if let Some(form) = r.form {
            p += "<|plamo:key|>form<|plamo:val|>";
            p += form.as_str();
        }
        p += "<|plamo:msg|>";
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(text: &str, target: &str) -> TranslationRequest {
        TranslationRequest { text: text.into(), target_lang: target.into(), ..Default::default() }
    }

    fn err(r: &TranslationRequest) -> String {
        r.clone().validate().unwrap_err().message
    }

    fn render(r: &TranslationRequest) -> String {
        r.clone().validate().unwrap().render()
    }

    #[test]
    fn language_codes() {
        for ok in ["en", "ja", "ceb", "zh-hans", "ja-easy"] {
            assert!(check_lang(ok, "lang").is_ok(), "{ok}");
        }
        for (bad, why) in [
            ("", "must not be empty"),
            ("  ", "must not be empty"),
            ("e", "lowercase language code"),
            ("engl", "lowercase language code"),
            ("en-us", "lowercase language code"),
            ("zh-hans-cn", "lowercase language code"),
            ("EN", "without region subtags"),
            ("zh-Hans", "without region subtags"),
            ("e1", "without region subtags"),
        ] {
            assert!(check_lang(bad, "lang").unwrap_err().message.contains(why), "{bad}");
        }
        // The model card's alias.
        assert!(render(&req("x", "ja(easy)")).contains("<|plamo:val|>ja-easy<|plamo:msg|>"));
    }

    #[test]
    fn form_needs_japanese() {
        let mut r = req("x", "ja");
        r.form = Some(Form::Polite);
        assert!(render(&r).ends_with("<|plamo:key|>form<|plamo:val|>polite<|plamo:msg|>"));
        for (target, form) in [("ja-easy", Form::Polite), ("en", Form::Plain)] {
            let mut r = req("x", target);
            r.form = Some(form);
            assert!(err(&r).starts_with("form requires"), "{target} {form}");
        }
    }

    #[test]
    fn reserved_tokens_are_rejected_in_every_field() {
        let reserved = "a<|plamo:tag|>b";
        let mut cases = Vec::new();
        cases.push(req(reserved, "en"));
        let mut r = req("x", "en");
        r.context = Some(reserved.into());
        cases.push(r);
        for (s, t) in [(reserved, "y"), ("y", reserved)] {
            let mut r = req("x", "en");
            r.history = vec![Pair { source: s.into(), target: t.into() }];
            cases.push(r);
            let mut r = req("x", "en");
            r.glossary = vec![Pair { source: s.into(), target: t.into() }];
            cases.push(r);
        }
        for r in &cases {
            assert!(err(r).contains("reserved PFML tokens"), "{r:?}");
        }
        // Only the full prefix is reserved.
        assert!(req("<|plamo and plamo:|>", "en").validate().is_ok());
    }

    #[test]
    fn blank_fields() {
        assert!(err(&req(" \n\u{3000}", "en")).contains("message content must not be empty"));
        let mut r = req("x", "en");
        r.glossary = vec![Pair { source: "a".into(), target: " ".into() }];
        assert!(err(&r).contains("glossary target must not be empty"));
        // A history pair needs both sides.
        assert!(validate_history_pair(&Pair { source: "a".into(), target: "b".into() }).is_ok());
        assert!(validate_history_pair(&Pair { source: "a".into(), target: "\n".into() }).is_err());
        // An empty context is left out of the prompt.
        let mut r = req("x", "en");
        r.context = Some(String::new());
        assert!(!render(&r).contains("context"));
    }

    #[test]
    fn validation_normalizes_and_parses() {
        let mut r = req("x", "ja");
        r.source_lang = Some("ja(easy)".into());
        r.form = Some(Form::Plain);
        r.history = vec![Pair { source: "a".into(), target: "b".into() }];
        let v = r.validate().unwrap();
        assert_eq!((v.source_lang(), v.target_lang()), (Some("ja-easy"), "ja"));
        assert_eq!(v.form(), Some(Form::Plain));
        let bare = v.clone().without_history();
        assert_eq!(v.history().len(), 1);
        assert!(bare.history().is_empty());
        assert!(v.render().len() > bare.render().len());
        assert!(!bare.render().contains("<|plamo:msg|>a<|plamo:tag|>"));
    }
}
