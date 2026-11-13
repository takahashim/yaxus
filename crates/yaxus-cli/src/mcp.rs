//! `yaxus mcp`: `polish` as a Model Context Protocol server on stdio, for Claude Code, Codex
//! and other clients. The model is loaded once and stays loaded for the session. Messages
//! are JSON-RPC, one per line; stdout carries nothing else (logs go to stderr).
//!
//! The tools: `polish` and `polish_mark` for Japanese text, and `write_japanese`, where the
//! calling LLM decides what is said (in English, say) and the model writes the Japanese.

use std::io::{BufRead, Write};

use anyhow::{Result, bail};
use clap::ValueEnum;
use serde_json::{Value, json};

use crate::polish::{Draft, FormChoice, Opts, Polisher, Via, mark_tool, polish_tool, write_tool};

/// Protocol versions this server speaks (the newest first). The client's is answered when
/// it is one of them, else the newest.
const VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

/// Most paragraphs one call takes: the model runs one request at a time, and a client
/// gives up on a call that takes too long.
const MAX_PARAGRAPHS: usize = 40;

const INSTRUCTIONS: &str = "Japanese proofreading with PLaMo 3 Translate, a translation model that \
writes natural Japanese. polish rewrites Japanese text (it goes through English and back by \
default) and shows what changed; polish_mark rewrites nothing and marks the wording the model \
finds unlikely. write_japanese turns paragraphs you wrote in English into Japanese: to remove \
emphasis, stock phrases or summaries from machine-written Japanese, write each paragraph's content in \
plain English yourself, then let write_japanese choose the Japanese wording. The rewrites are \
suggestions: read what they changed, and the words they lost.";

const FORM: &str = "Politeness of the Japanese: auto (each paragraph keeps its own, from its sentence endings; \
default), polite (です・ます), plain (だ・である) or none (the model chooses).";

fn tools() -> Value {
    json!([
        {
            "name": "polish",
            "description": "Rewrite Japanese text so that it reads naturally, and show what changed. Each \
    paragraph (separated by blank lines) is translated into another language and back (via en, \
    default), so the model writes the Japanese anew in its own words, or translated Japanese to \
    Japanese (via direct), which fixes errors but keeps the wording. Returns, for each paragraph and \
    route: output, diff ([-removed-]{+added+}), lost (numbers, Latin words, URLs or quotations the \
    rewrite dropped: check them) and the English pivot. A rewrite can drift in meaning: review it.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Japanese text; paragraphs separated by blank lines."},
                    "via": {"type": "string", "description": "Routes, comma-separated, one candidate each: a language code (en, zh-hans, ...) or direct. Default en."},
                    "form": {"type": "string", "enum": ["auto", "polite", "plain", "none"], "description": FORM}
                },
                "required": ["text"]
            }
        },
        {
            "name": "polish_mark",
            "description": "Find the wording in Japanese text that the model finds unlikely, without rewriting \
    it. Returns, for each paragraph, marks: the piece (text, byte range start/end in source), its \
    probability and what the model would have written there (best; best_ends: it would have ended). \
    Good at errors (particles, ら抜き, wrong words, redundancy). A high probability means typical \
    wording, not necessarily human wording.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Japanese text; paragraphs separated by blank lines."},
                    "threshold": {"type": "number", "description": "Mark pieces below this probability (0 to 1; default 0.05)."},
                    "via": {"type": "string", "description": "direct (default: finds errors) or a language code (en: the wording alone; adds the log-probability per character next to the model's own rewrite)."},
                    "form": {"type": "string", "enum": ["auto", "polite", "plain", "none"], "description": FORM}
                },
                "required": ["text"]
            }
        },
        {
            "name": "write_japanese",
            "description": "Write Japanese from paragraphs you wrote in another language (English by default): \
    you decide what is said, the model chooses how it is said, in natural Japanese wording. To clean \
    up Japanese text (emphasis such as **bold**, emoji, headings and lists, stock phrases, needless \
    summaries), write each paragraph's content in plain English and give the Japanese paragraph as \
    its source: the politeness is taken from the source, and the output is diffed against it and \
    checked for lost numbers, Latin words, URLs and quotations (lost). Keep every fact, number and name. \
    Give one English paragraph for each source paragraph.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paragraphs": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": MAX_PARAGRAPHS,
                        "description": "The paragraphs, in order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "The paragraph in English (or lang)."},
                                "source": {"type": "string", "description": "The Japanese paragraph it was written from (all or none)."}
                            },
                            "required": ["text"]
                        }
                    },
                    "lang": {"type": "string", "description": "Language of the paragraphs (default en)."},
                    "form": {"type": "string", "enum": ["auto", "polite", "plain", "none"], "description": "As above; without sources, auto means none."}
                },
                "required": ["paragraphs"]
            }
        }
    ])
}

/// Serves requests from `input` until it ends, answering on `out`.
pub fn serve(p: &mut dyn Polisher, input: impl BufRead, mut out: impl Write) -> Result<()> {
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(msg) => handle(p, &msg),
            Err(e) => Some(error(Value::Null, -32700, &format!("parse error: {e}"))),
        };
        if let Some(r) = reply {
            writeln!(out, "{r}")?;
            out.flush()?;
        }
    }
    Ok(())
}

fn error(id: Value, code: i32, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

/// The reply to one message; none for a notification.
fn handle(p: &mut dyn Polisher, msg: &Value) -> Option<Value> {
    // (no id: a notification, which gets no answer, whatever it is)
    let id = msg.get("id")?.clone();
    let Some(method) = msg.get("method").and_then(Value::as_str) else {
        return Some(error(id, -32600, "invalid request"));
    };
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    let result = match method {
        "initialize" => {
            let asked = params.get("protocolVersion").and_then(Value::as_str);
            let version = VERSIONS.iter().find(|v| Some(**v) == asked).unwrap_or(&VERSIONS[0]);
            json!({
                "protocolVersion": version,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "yaxus", "version": env!("CARGO_PKG_VERSION")},
                "instructions": INSTRUCTIONS,
            })
        }
        "ping" => json!({}),
        "tools/list" => json!({ "tools": tools() }),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or_default();
            if !tools().as_array().is_some_and(|t| t.iter().any(|t| t["name"] == name)) {
                return Some(error(id, -32602, &format!("unknown tool {name:?}")));
            }
            let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));
            // A failing tool is a result the model reads (isError), not a protocol error.
            match call(p, name, &args) {
                Ok(v) => json!({"content": [{"type": "text", "text": v.to_string()}], "isError": false}),
                Err(e) => json!({"content": [{"type": "text", "text": format!("{e:#}")}], "isError": true}),
            }
        }
        _ => return Some(error(id, -32601, &format!("method not found: {method}"))),
    };
    Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

fn string<'a>(args: &'a Value, key: &str) -> Result<Option<&'a str>> {
    match args.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => bail!("{key} must be a string"),
    }
}

fn form(args: &Value) -> Result<FormChoice> {
    match string(args, "form")? {
        None => Ok(FormChoice::Auto),
        Some(f) => {
            FormChoice::from_str(f, true).map_err(|_| anyhow::anyhow!("form must be auto, polite, plain or none"))
        }
    }
}

fn via(s: &str) -> Result<Via> {
    s.parse::<Via>().map_err(anyhow::Error::msg)
}

/// The paragraphs of the `text` argument.
fn paragraphs(args: &Value) -> Result<Vec<String>> {
    let Some(text) = string(args, "text")? else { bail!("text is required") };
    let segments = yaxus_api::split_paragraphs(text);
    if segments.is_empty() {
        bail!("no text");
    }
    if segments.len() > MAX_PARAGRAPHS {
        bail!("{} paragraphs: at most {MAX_PARAGRAPHS} in a call", segments.len());
    }
    Ok(segments)
}

fn call(p: &mut dyn Polisher, name: &str, args: &Value) -> Result<Value> {
    let o = Opts::default();
    match name {
        "polish" => {
            let vias = string(args, "via")?.unwrap_or("en").split(',').map(via).collect::<Result<Vec<_>>>()?;
            polish_tool(p, &o, &paragraphs(args)?, &vias, form(args)?)
        }
        "polish_mark" => {
            let via = via(string(args, "via")?.unwrap_or("direct"))?;
            let threshold = match args.get("threshold") {
                None | Some(Value::Null) => 0.05,
                Some(v) => v.as_f64().ok_or_else(|| anyhow::anyhow!("threshold must be a number"))? as f32,
            };
            mark_tool(p, &o, &paragraphs(args)?, &via, form(args)?, threshold)
        }
        "write_japanese" => {
            let Some(items) = args.get("paragraphs").and_then(Value::as_array) else {
                bail!("paragraphs is required: an array of {{text, source}}");
            };
            if items.len() > MAX_PARAGRAPHS {
                bail!("{} paragraphs: at most {MAX_PARAGRAPHS} in a call", items.len());
            }
            let drafts = items
                .iter()
                .map(|i| {
                    let text = string(i, "text")?.ok_or_else(|| anyhow::anyhow!("every paragraph needs text"))?;
                    Ok(Draft { text: text.to_string(), source: string(i, "source")?.map(String::from) })
                })
                .collect::<Result<Vec<_>>>()?;
            let lang = match via(string(args, "lang")?.unwrap_or("en"))? {
                Via::Lang(l) => l,
                Via::Direct => bail!("lang is a language code (en, zh-hans, ...)"),
            };
            write_tool(p, &o, &lang, &drafts, form(args)?)
        }
        _ => bail!("unknown tool {name:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polish::{ScoreRequest, Scored};

    /// A model that writes `text@lang`, and finds every text unlikely.
    struct Fake;

    impl Polisher for Fake {
        fn translate(&mut self, _stage: &str, req: yaxus_api::DocumentRequest) -> Result<Vec<String>> {
            Ok(req.segments.iter().map(|s| format!("{s}@{}", req.target_lang)).collect())
        }

        fn score(&mut self, r: &ScoreRequest) -> Result<Vec<Scored>> {
            let n = r.output.len();
            Ok(vec![Scored { start: 0, end: n, logprob: -9.0, best: crate::polish::Choice::End, best_logprob: 0.0 }])
        }
    }

    /// The replies to these lines, parsed.
    fn talk(lines: &[Value]) -> Vec<Value> {
        let input: String = lines.iter().map(|l| format!("{l}\n")).collect();
        let mut out = Vec::new();
        serve(&mut Fake, input.as_bytes(), &mut out).unwrap();
        String::from_utf8(out).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
    }

    fn call_tool(name: &str, args: Value) -> Value {
        let r = talk(&[
            json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}}),
        ]);
        r.into_iter().next().unwrap()
    }

    /// The JSON a tool returned in its text.
    fn tool_result(reply: &Value) -> Value {
        assert_eq!(reply["result"]["isError"], false, "{reply}");
        serde_json::from_str(reply["result"]["content"][0]["text"].as_str().unwrap()).unwrap()
    }

    #[test]
    fn the_handshake_and_the_tools() {
        let r = talk(&[
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}}),
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
            json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            json!({"jsonrpc": "2.0", "id": 3, "method": "ping"}),
            json!({"jsonrpc": "2.0", "id": 4, "method": "nope"}),
        ]);
        // (the notification has no reply)
        assert_eq!(r.len(), 4);
        assert_eq!(r[0]["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(r[0]["result"]["serverInfo"]["name"], "yaxus");
        let names: Vec<_> = r[1]["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].clone()).collect();
        assert_eq!(names, ["polish", "polish_mark", "write_japanese"]);
        assert_eq!(r[2]["result"], json!({}));
        assert_eq!(r[3]["error"]["code"], -32601);
    }

    #[test]
    fn an_unknown_version_is_answered_with_the_newest() {
        let r = talk(&[
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}}),
        ]);
        assert_eq!(r[0]["result"]["protocolVersion"], VERSIONS[0]);
    }

    #[test]
    fn bad_json_is_a_parse_error() {
        let mut out = Vec::new();
        serve(&mut Fake, "{oops\n\n".as_bytes(), &mut out).unwrap();
        let r: Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(r["error"]["code"], -32700);
        assert_eq!(r["id"], Value::Null);
    }

    #[test]
    fn polish_goes_through_english() {
        let v = tool_result(&call_tool("polish", json!({"text": "一つ目です。\n\n二つ目。"})));
        let ps = v["paragraphs"].as_array().unwrap();
        assert_eq!(ps.len(), 2);
        let c = &ps[0]["candidates"][0];
        assert_eq!(c["via"], "en");
        assert_eq!(c["pivot"], "一つ目です。@en");
        assert_eq!(c["output"], "一つ目です。@en@ja");
        assert_eq!(c["changed"], true);
        assert_eq!(c["diff"], "一つ目です。{+@en@ja+}");
        assert_eq!(ps[0]["form"], "polite");
    }

    #[test]
    fn polish_takes_several_routes() {
        let v = tool_result(&call_tool("polish", json!({"text": "猫です。", "via": "en,direct", "form": "plain"})));
        let c = v["paragraphs"][0]["candidates"].as_array().unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c[1]["via"], "direct");
        assert_eq!(c[1]["pivot"], Value::Null);
        assert_eq!(v["paragraphs"][0]["form"], "plain");
    }

    #[test]
    fn mark_finds_the_unlikely_wording() {
        let v = tool_result(&call_tool("polish_mark", json!({"text": "猫です。"})));
        let m = &v["paragraphs"][0]["marks"][0];
        assert_eq!(m["text"], "猫です。");
        assert_eq!(m["best_ends"], true);
        // (the tool leaves the pieces out)
        assert!(v["paragraphs"][0].get("pieces").is_none());
    }

    #[test]
    fn write_japanese_checks_against_the_source() {
        let v = tool_result(&call_tool(
            "write_japanese",
            json!({"paragraphs": [{"text": "The cat sleeps.", "source": "猫が**寝ています**。"}]}),
        ));
        let p = &v["paragraphs"][0];
        assert_eq!(p["output"], "The cat sleeps.@ja");
        assert_eq!(p["form"], "polite");
        assert!(p["diff"].as_str().unwrap().contains("[-"));
        // Without sources there is nothing to diff, and `auto` leaves the form to the model.
        let v = tool_result(&call_tool("write_japanese", json!({"paragraphs": [{"text": "Hello."}]})));
        assert_eq!(v["paragraphs"][0]["output"], "Hello.@ja");
        assert_eq!(v["paragraphs"][0]["form"], Value::Null);
    }

    #[test]
    fn a_bad_call_is_a_result_the_model_reads() {
        for (name, args, said) in [
            ("polish", json!({}), "text is required"),
            ("polish", json!({"text": "  "}), "no text"),
            ("polish", json!({"text": "猫", "via": "ja"}), "use direct"),
            ("polish", json!({"text": "猫", "form": "rude"}), "form must be"),
            ("polish_mark", json!({"text": "猫", "threshold": 2}), "between 0 and 1"),
            ("polish_mark", json!({"text": "猫", "threshold": "x"}), "must be a number"),
            (
                "write_japanese",
                json!({"paragraphs": [{"text": "a", "source": "あ"}, {"text": "b"}]}),
                "every paragraph",
            ),
            ("write_japanese", json!({"paragraphs": [{"source": "あ"}]}), "needs text"),
            ("write_japanese", json!({"paragraphs": [{"text": "a"}], "lang": "direct"}), "language code"),
        ] {
            let r = call_tool(name, args);
            assert_eq!(r["result"]["isError"], true, "{name}: {r}");
            let text = r["result"]["content"][0]["text"].as_str().unwrap();
            assert!(text.contains(said), "{name}: {text}");
        }
        // An unknown tool is a protocol error.
        assert_eq!(call_tool("nope", json!({}))["error"]["code"], -32602);
    }

    #[test]
    fn too_many_paragraphs_are_refused() {
        let text = vec!["猫。"; MAX_PARAGRAPHS + 1].join("\n\n");
        let r = call_tool("polish", json!({ "text": text }));
        assert_eq!(r["result"]["isError"], true);
    }
}
