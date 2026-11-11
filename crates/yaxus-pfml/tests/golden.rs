//! Compares the prompts with `tests/golden/tokenizer.jsonl`, produced by the Python
//! chat template (the `prompt` and `prompt_error` cases). Their token ids are checked
//! by the tokenizer's golden tests.

use std::path::Path;

use serde_json::Value;
use yaxus_pfml::{Pair, TranslationRequest};

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(|x| x.as_str()).map(str::to_string)
}

fn pairs(v: &Value, k: &str) -> Vec<Pair> {
    v.get(k)
        .and_then(|x| x.as_array())
        .map(|a| a.iter().map(|e| Pair { source: s(e, "source").unwrap(), target: s(e, "target").unwrap() }).collect())
        .unwrap_or_default()
}

fn request(v: &Value) -> TranslationRequest {
    TranslationRequest {
        text: s(v, "text").unwrap(),
        target_lang: s(v, "target_lang").unwrap(),
        source_lang: s(v, "source_lang"),
        form: s(v, "form"),
        context: s(v, "context"),
        glossary: pairs(v, "glossary"),
        history: pairs(v, "history"),
    }
}

#[test]
fn golden_prompts() {
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/tokenizer.jsonl");
    let mut n = 0;
    for line in std::fs::read_to_string(golden).unwrap().lines() {
        let c: Value = serde_json::from_str(line).unwrap();
        match c["kind"].as_str().unwrap() {
            "prompt" => {
                let prompt = request(&c["request"]).validate().unwrap().render();
                assert_eq!(prompt, c["prompt"].as_str().unwrap());
            }
            "prompt_error" => {
                let req = request(&c["request"]);
                assert!(req.clone().validate().is_err(), "expected error for {req:?}");
            }
            _ => continue,
        }
        n += 1;
    }
    assert!(n > 5);
}
