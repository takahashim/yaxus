//! PFML3 prompt construction, a direct port of `chat_template.jinja`.

/// A previous source/translation pair, used for consistency across segments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslationPair {
    pub source: String,
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlossaryEntry {
    pub source: String,
    pub target: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TranslationRequest {
    pub text: String,
    pub target_lang: String,
    pub source_lang: Option<String>,
    /// `polite` or `plain`; only valid when `target_lang == "ja"`.
    pub form: Option<String>,
    pub context: Option<String>,
    pub glossary: Vec<GlossaryEntry>,
    pub history: Vec<TranslationPair>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct PromptError(pub String);

const RESERVED: &str = "<|plamo:";

/// Whether `s` is empty or only whitespace, as the prompt template sees it (Jinja's
/// `trim` is Python's `str.strip()`, which also strips U+001C..U+001F).
pub fn is_blank(s: &str) -> bool {
    // Jinja `trim` uses Python `str.strip()`.
    s.chars().all(crate::py_isspace)
}

fn check_text(value: &str, name: &str, allow_empty: bool) -> Result<(), PromptError> {
    if !allow_empty && is_blank(value) {
        return Err(PromptError(format!("{name} must not be empty.")));
    }
    if value.contains(RESERVED) {
        return Err(PromptError(format!("{name} must not contain reserved PFML tokens (<|plamo:...).")));
    }
    Ok(())
}

/// Validates a PFML3 language code (`en`, `ja`, `zh-hans`, `ja-easy`, ...).
pub fn validate_lang(value: &str, name: &str) -> Result<(), PromptError> {
    check_text(value, name, false)?;
    let parts: Vec<&str> = value.split('-').collect();
    let bad_shape = parts.len() > 2
        || !(2..=3).contains(&parts[0].chars().count())
        || (parts.len() == 2 && parts[1].chars().count() != 4);
    if bad_shape {
        return Err(PromptError(format!(
            "{name} must be a lowercase language code, e.g. en, ja, zh-hans, or ja-easy."
        )));
    }
    if !value.chars().all(|c| c.is_ascii_lowercase() || c == '-') {
        return Err(PromptError(format!("{name} must use lowercase language codes without region subtags.")));
    }
    Ok(())
}

fn alias(lang: &str) -> &str {
    if lang == "ja(easy)" { "ja-easy" } else { lang }
}

/// Checks a text to translate (or a history entry): not blank, no reserved tokens.
pub fn validate_message(text: &str) -> Result<(), PromptError> {
    check_text(text, "message content", false)
}

impl TranslationRequest {
    /// Checks everything [`to_prompt`](TranslationRequest::to_prompt) checks except the
    /// text and the history: the languages, the form, the context and the glossary.
    pub fn validate_options(&self) -> Result<(), PromptError> {
        let target_lang = alias(&self.target_lang);
        validate_lang(target_lang, "target_lang")?;
        if let Some(s) = self.source_lang.as_deref().map(alias) {
            validate_lang(s, "source_lang")?;
        }
        if let Some(c) = &self.context {
            check_text(c, "context", true)?;
        }
        if let Some(form) = &self.form
            && (!matches!(form.as_str(), "polite" | "plain") || target_lang != "ja")
        {
            return Err(PromptError("form must be polite or plain and requires target_lang=\"ja\".".into()));
        }
        for term in &self.glossary {
            check_text(&term.source, "glossary source", false)?;
            check_text(&term.target, "glossary target", false)?;
        }
        Ok(())
    }

    /// Renders the PFML3 prompt, including the leading `<|plamo:bos|>`.
    /// Tokenize the result with `add_bos = false`.
    pub fn to_prompt(&self) -> Result<String, PromptError> {
        for pair in &self.history {
            validate_message(&pair.source)?;
            validate_message(&pair.target)?;
        }
        validate_message(&self.text)?;
        self.validate_options()?;
        let target_lang = alias(&self.target_lang);
        let source_lang = self.source_lang.as_deref().map(alias);

        let mut p = String::from("<|plamo:bos|><|plamo:tag|>dataset<|plamo:msg|>translation");
        if let Some(c) = self.context.as_deref().filter(|c| !c.is_empty()) {
            p += "<|plamo:tag|>context<|plamo:msg|>";
            p += c;
        }
        let input_header = |p: &mut String| {
            *p += "<|plamo:tag|>input";
            if let Some(s) = source_lang {
                *p += "<|plamo:key|>lang<|plamo:val|>";
                *p += s;
            }
        };
        for pair in &self.history {
            input_header(&mut p);
            p += "<|plamo:msg|>";
            p += &pair.source;
            p += "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>";
            p += target_lang;
            p += "<|plamo:msg|>";
            p += &pair.target;
        }
        if !self.glossary.is_empty() {
            p += "<|plamo:tag|>glossary";
            for term in &self.glossary {
                p += "<|plamo:index|>";
                p += &term.source;
                p += "<|plamo:item|>";
                p += &term.target;
            }
        }
        input_header(&mut p);
        p += "<|plamo:msg|>";
        p += &self.text;
        p += "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>";
        p += target_lang;
        if let Some(form) = &self.form {
            p += "<|plamo:key|>form<|plamo:val|>";
            p += form;
        }
        p += "<|plamo:msg|>";
        Ok(p)
    }
}
