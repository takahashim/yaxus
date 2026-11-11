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

fn is_blank(s: &str) -> bool {
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

impl TranslationRequest {
    /// Renders the PFML3 prompt, including the leading `<|plamo:bos|>`.
    /// Tokenize the result with `add_bos = false`.
    pub fn to_prompt(&self) -> Result<String, PromptError> {
        for pair in &self.history {
            check_text(&pair.source, "message content", false)?;
            check_text(&pair.target, "message content", false)?;
        }
        check_text(&self.text, "message content", false)?;

        let target_lang = alias(&self.target_lang);
        let source_lang = self.source_lang.as_deref().map(alias);
        validate_lang(target_lang, "target_lang")?;
        if let Some(s) = source_lang {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn req(text: &str, target: &str) -> TranslationRequest {
        TranslationRequest { text: text.into(), target_lang: target.into(), ..Default::default() }
    }

    fn err(r: &TranslationRequest) -> String {
        r.to_prompt().unwrap_err().0
    }

    #[test]
    fn language_codes() {
        for ok in ["en", "ja", "ceb", "zh-hans", "ja-easy"] {
            assert!(validate_lang(ok, "lang").is_ok(), "{ok}");
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
            assert!(validate_lang(bad, "lang").unwrap_err().0.contains(why), "{bad}");
        }
        // The model card's alias.
        assert!(req("x", "ja(easy)").to_prompt().unwrap().contains("<|plamo:val|>ja-easy<|plamo:msg|>"));
    }

    #[test]
    fn form_needs_japanese() {
        let mut r = req("x", "ja");
        r.form = Some("polite".into());
        assert!(r.to_prompt().unwrap().ends_with("<|plamo:key|>form<|plamo:val|>polite<|plamo:msg|>"));
        for (target, form) in [("ja-easy", "polite"), ("en", "plain"), ("ja", "casual")] {
            let mut r = req("x", target);
            r.form = Some(form.into());
            assert!(err(&r).starts_with("form must be"), "{target} {form}");
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
            r.history = vec![TranslationPair { source: s.into(), target: t.into() }];
            cases.push(r);
            let mut r = req("x", "en");
            r.glossary = vec![GlossaryEntry { source: s.into(), target: t.into() }];
            cases.push(r);
        }
        for r in &cases {
            assert!(err(r).contains("reserved PFML tokens"), "{r:?}");
        }
        // Only the full prefix is reserved.
        assert!(req("<|plamo and plamo:|>", "en").to_prompt().is_ok());
    }

    #[test]
    fn blank_fields() {
        assert!(err(&req(" \n\u{3000}", "en")).contains("message content must not be empty"));
        let mut r = req("x", "en");
        r.glossary = vec![GlossaryEntry { source: "a".into(), target: " ".into() }];
        assert!(err(&r).contains("glossary target must not be empty"));
        // An empty context is left out of the prompt.
        let mut r = req("x", "en");
        r.context = Some(String::new());
        assert!(!r.to_prompt().unwrap().contains("context"));
    }
}
