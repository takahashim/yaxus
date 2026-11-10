//! Compares against `tests/golden/tokenizer.jsonl` produced by the Python tokenizer.
//!
//! Needs the model's `tokenizer.jsonl`; set `YAKUSEL_MODEL_DIR` or have
//! pfnet/plamo-3-translate-2b in the Hugging Face cache. Skips otherwise.

use std::path::{Path, PathBuf};

use serde_json::Value;
use yakusel_tokenizer::{GlossaryEntry, Tokenizer, TranslationPair, TranslationRequest};

fn model_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("YAKUSEL_MODEL_DIR") {
        return Some(PathBuf::from(d));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = Path::new(&home)
        .join(".cache/huggingface/hub/models--pfnet--plamo-3-translate-2b/snapshots");
    std::fs::read_dir(snaps).ok()?.flatten().map(|e| e.path()).next()
}

fn load() -> Option<(Tokenizer, Vec<Value>)> {
    let Some(dir) = model_dir() else {
        eprintln!("model directory not found; skipping golden tests");
        return None;
    };
    let tok = Tokenizer::from_file(dir.join("tokenizer.jsonl")).unwrap();
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/tokenizer.jsonl");
    let cases = std::fs::read_to_string(golden)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    Some((tok, cases))
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn s(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(|x| x.as_str()).map(str::to_string)
}

fn pairs(v: &Value, k: &str) -> Vec<(String, String)> {
    v.get(k)
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .map(|e| (s(e, "source").unwrap(), s(e, "target").unwrap()))
                .collect()
        })
        .unwrap_or_default()
}

fn request(v: &Value) -> TranslationRequest {
    TranslationRequest {
        text: s(v, "text").unwrap(),
        target_lang: s(v, "target_lang").unwrap(),
        source_lang: s(v, "source_lang"),
        form: s(v, "form"),
        context: s(v, "context"),
        glossary: pairs(v, "glossary")
            .into_iter()
            .map(|(source, target)| GlossaryEntry { source, target })
            .collect(),
        history: pairs(v, "history")
            .into_iter()
            .map(|(source, target)| TranslationPair { source, target })
            .collect(),
    }
}

#[test]
fn golden_encode_decode() {
    let Some((tok, cases)) = load() else { return };
    assert_eq!(tok.vocab_size(), 107520);
    let mut n = 0;
    let mut failures = Vec::new();
    for c in cases.iter().filter(|c| c["kind"] == "encode") {
        let text = c["text"].as_str().unwrap();
        let want = ids(&c["ids"]);
        let got = tok.encode(text, false);
        if got != want {
            failures.push(format!("encode {text:?}\n  want {want:?}\n  got  {got:?}"));
        }
        let decoded = tok.decode(&want);
        if decoded != c["decoded"].as_str().unwrap() {
            failures.push(format!("decode {text:?}: got {decoded:?}"));
        }
        n += 1;
    }
    assert!(failures.is_empty(), "{} / {n} failed:\n{}", failures.len(), failures.join("\n"));
    assert!(n > 100);
}

#[test]
fn golden_prompts() {
    let Some((tok, cases)) = load() else { return };
    let mut n = 0;
    for c in &cases {
        match c["kind"].as_str().unwrap() {
            "prompt" => {
                let req = request(&c["request"]);
                let prompt = req.to_prompt().unwrap();
                assert_eq!(prompt, c["prompt"].as_str().unwrap());
                assert_eq!(tok.encode(&prompt, false), ids(&c["ids"]), "{prompt}");
            }
            "prompt_error" => {
                let req = request(&c["request"]);
                assert!(req.to_prompt().is_err(), "expected error for {req:?}");
            }
            _ => continue,
        }
        n += 1;
    }
    assert!(n > 5);
}

#[test]
fn stream_decoder_matches_decode() {
    let Some((tok, cases)) = load() else { return };
    for c in cases.iter().filter(|c| c["kind"] == "encode") {
        let want = ids(&c["ids"]);
        let mut d = yakusel_tokenizer::StreamDecoder::new();
        let mut out = String::new();
        for &id in &want {
            out += &d.push(&tok, id);
        }
        out += &d.finish();
        assert_eq!(out, tok.decode(&want));
    }
}
