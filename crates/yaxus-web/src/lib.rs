//! yaxus in the browser: a chosen GGUF file is parsed in wasm memory, the weights go to
//! WebGPU, and translation streams back into the page.
//!
//! The browser path deliberately keeps one request and one model: no engine, draft
//! model, document chains, batching or KV-prefix cache. Everything above the GPU is the
//! same code as the CLI (`yaxus-model` and the kernels of `yaxus-backend-gpu`).
#![cfg(target_arch = "wasm32")]

use std::fmt::Display;

use js_sys::Function;
use wasm_bindgen::prelude::*;
use web_time::Instant;
use yaxus_backend_gpu::{GpuBackend, Tensor};
use yaxus_model::{Model, ModelOptions, Quant, SeqId, gguf::Gguf};
use yaxus_pfml::{
    Form, Pair, TranslationRequest,
    segments::{Segment, split_for_context},
    validate_history_pair,
};
use yaxus_tokenizer::{EOS_ID, StreamDecoder, TAG_ID, TokenKind, Tokenizer};

/// The longest output one segment asks for.
const MAX_NEW_TOKENS: usize = 1024;
/// Prompt plus output of one segment; the KV arena and RoPE tables are this long.
const MAX_CONTEXT: usize = 4096;
/// Input tokens one segment aims for. A paragraph that fits stays whole; a longer one is
/// split at sentences. Paragraphs/history/output all live inside `MAX_CONTEXT`.
const SEGMENT_INPUT_TOKENS: usize = 768;
/// Characters of earlier source/translation pairs sent as history (at most two pairs).
const HISTORY_CHARS: usize = 1600;
/// Room a prompt must leave for the answer before history is dropped.
const MIN_NEW: usize = 256;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

fn jserr(e: impl Display) -> JsError {
    JsError::new(&format!("{e:#}"))
}

fn report(on_progress: &Function, message: &str) {
    let _ = on_progress.call1(&JsValue::NULL, &JsValue::from_str(message));
}

/// Lets the browser paint the progress message before a long synchronous step.
async fn tick() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        match web_sys::window() {
            Some(window) => {
                let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0);
            }
            // A worker (or a test runtime) has no `window`; resolve at once.
            None => {
                let _ = resolve.call0(&JsValue::NULL);
            }
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// Loads a GGUF file (already in wasm memory) and opens WebGPU.
///
/// `on_progress` is called with one short message per phase. The errors are the same
/// ones the CLI reports (not a PLaMo GGUF, a tensor missing, a device too small).
#[wasm_bindgen]
pub async fn load_model(bytes: Vec<u8>, on_progress: Function) -> Result<Translator, JsError> {
    let total = Instant::now();
    report(&on_progress, "GGUF を解析しています…");
    tick().await;
    let g = Gguf::parse(bytes).map_err(jserr)?;
    let vocab = g.vocab().map_err(jserr)?;
    g.config().map_err(jserr)?;

    report(&on_progress, "トークナイザを構築しています…");
    tick().await;
    let kinds = vocab.types.iter().copied().map(TokenKind::from_ggml_type).collect();
    let tokenizer = Tokenizer::from_vocab(vocab.tokens, kinds, vocab.scores).map_err(jserr)?;

    // The largest buffer a single binding has to cover. For the 0.8B Q4_K_M this is the
    // Q6_K output matrix (135 MB); the embedding stays Q4_K (93 MB) instead of bf16.
    let max_tensor = g.tensors.iter().map(|(_, t)| t.bytes().len() as u64).max().unwrap_or(0);
    // A table of another format grows to bf16 in `load_embedding`, so ask for that room.
    let embed =
        g.tensors
            .iter()
            .find(|(name, _)| name == "token_embd" || name == "model.embed_tokens.weight")
            .map(|(_, t)| {
                if t.dtype == yaxus_model::ggml::GgmlType::Q4K {
                    t.bytes().len() as u64
                } else {
                    (t.numel() * 2) as u64
                }
            })
            .unwrap_or(0);
    report(&on_progress, "WebGPU デバイスを開いています…");
    tick().await;
    let need_binding = max_tensor.max(embed);
    let backend = GpuBackend::new_async(need_binding, need_binding.max(64 << 20)).await.map_err(jserr)?;
    let name = backend.caps().name.clone();
    let max_binding = backend.max_binding_bytes();
    // A device whose storage binding is 128 MB cannot bind the 135 MB Q6_K output; Q4
    // there is 82 MB. This only happens on the conservative limit tier.
    let lm_head_quant = if max_tensor > max_binding { Some(Quant::Q4) } else { None };

    report(&on_progress, "重みを GPU に転送しています…");
    tick().await;
    let opts = ModelOptions {
        quant: Quant::Auto,
        kv_tokens: MAX_CONTEXT,
        max_pos: MAX_CONTEXT,
        prefix_cache: false,
        lm_head_quant,
    };
    let model = Model::from_gguf(g, backend, &opts).map_err(jserr)?;
    report(&on_progress, "準備できました");
    let load_ms = total.elapsed().as_secs_f64() * 1e3;
    Ok(Translator { model, tokenizer, name, load_ms, max_binding, lm_head_q4: lm_head_quant.is_some() })
}

/// One loaded model. Only one translation runs at a time (the page's button is disabled
/// while it does).
#[wasm_bindgen]
pub struct Translator {
    model: Model<GpuBackend>,
    tokenizer: Tokenizer,
    name: String,
    load_ms: f64,
    max_binding: u64,
    lm_head_q4: bool,
}

#[wasm_bindgen]
impl Translator {
    /// The WebGPU adapter's name.
    #[wasm_bindgen(getter)]
    pub fn device(&self) -> String {
        self.name.clone()
    }

    /// Loading time (parse, tokenizer, upload) in milliseconds.
    #[wasm_bindgen(getter)]
    pub fn load_ms(&self) -> f64 {
        self.load_ms
    }

    /// The largest storage binding the model got.
    #[wasm_bindgen(getter)]
    pub fn max_binding(&self) -> f64 {
        self.max_binding as f64
    }

    /// True when the output matrix had to be reduced to Q4 to fit the binding limit.
    #[wasm_bindgen(getter)]
    pub fn lm_head_q4(&self) -> bool {
        self.lm_head_q4
    }

    /// Translates English into Japanese. Long text is split into paragraphs and
    /// sentences; each segment is translated in order, with the previous segments as
    /// history, and `on_text` receives text as it is produced. `on_progress` gets
    /// `(segments_done, segments_total)`.
    pub async fn translate(
        &mut self,
        text: String,
        form: Option<String>,
        on_text: Function,
        on_progress: Option<Function>,
    ) -> Result<Stats, JsError> {
        let form = match form.as_deref() {
            None | Some("") | Some("none") => None,
            Some("polite") => Some(Form::Polite),
            Some("plain") => Some(Form::Plain),
            Some(other) => return Err(JsError::new(&format!("unknown form {other:?}"))),
        };
        let segments = split_for_context(&text, &self.tokenizer, SEGMENT_INPUT_TOKENS);
        if segments.is_empty() {
            return Err(JsError::new("the text is empty or only whitespace"));
        }
        let total = segments.len();
        let mut history: Vec<Pair> = Vec::new();
        let mut prompt_tokens = 0u32;
        let mut output_tokens = 0u32;
        let mut chars = 0usize;
        let mut prefill_ms = 0.0;
        let mut decode_ms = 0.0;
        let mut any_limit = false;
        let mut last_stop = "tag";
        let mut last_max_new = MAX_NEW_TOKENS as u32;

        for (i, segment) in segments.iter().enumerate() {
            if let Some(progress) = &on_progress {
                let _ = progress.call2(&JsValue::NULL, &JsValue::from_f64(i as f64), &JsValue::from_f64(total as f64));
            }
            let (ids, max_new) = self.fit_segment(segment, form, &history)?;
            if chars > 0 {
                let separator = if segment.paragraph { "\n\n" } else { "" };
                if !separator.is_empty() {
                    chars += separator.chars().count();
                    emit(&on_text, separator)?;
                }
            }
            let prompt_len = ids.len() as u32;
            let prefill_start = Instant::now();
            let (id, first) = self.model.start(&ids).map_err(jserr)?;
            prefill_ms += prefill_start.elapsed().as_secs_f64() * 1e3;

            let decode_start = Instant::now();
            let (translated, stop, generated) = self.decode_segment(id, first, max_new, &on_text, &mut chars).await?;
            decode_ms += decode_start.elapsed().as_secs_f64() * 1e3;
            prompt_tokens += prompt_len;
            output_tokens += generated;
            last_max_new = max_new as u32;
            if stop == "limit" {
                any_limit = true;
            }
            last_stop = stop;

            // Use complete translations only: a segment cut at the budget is a misleading
            // history for the next one.
            if stop != "limit" && !translated.trim().is_empty() {
                let pair = Pair { source: segment.text.clone(), target: translated.trim().to_string() };
                if validate_history_pair(&pair).is_ok() {
                    history.push(pair);
                    if history.len() > 4 {
                        history.remove(0);
                    }
                }
            }
        }
        if let Some(progress) = &on_progress {
            let _ = progress.call2(&JsValue::NULL, &JsValue::from_f64(total as f64), &JsValue::from_f64(total as f64));
        }
        Ok(Stats {
            prompt_tokens,
            output_tokens,
            max_new: last_max_new,
            segments: total as u32,
            stop: (if any_limit { "limit" } else { last_stop }).into(),
            chars: chars as u32,
            prefill_ms,
            decode_ms,
        })
    }

    /// The prompt for `segment` with as much history as fits, and the output budget.
    fn fit_segment(
        &self,
        segment: &Segment,
        form: Option<Form>,
        earlier: &[Pair],
    ) -> Result<(Vec<u32>, usize), JsError> {
        let mut history = select_history(earlier);
        loop {
            let req = TranslationRequest {
                text: segment.text.clone(),
                target_lang: "ja".into(),
                source_lang: Some("en".into()),
                form,
                context: None,
                glossary: Vec::new(),
                history: history.clone(),
            };
            let req = req.validate().map_err(jserr)?;
            let ids = self.tokenizer.encode(&req.render(), false);
            let len = ids.len();
            if len < MAX_CONTEXT && (history.is_empty() || len + MIN_NEW <= MAX_CONTEXT) {
                return Ok((ids, MAX_NEW_TOKENS.min(MAX_CONTEXT - len)));
            }
            if history.is_empty() {
                return Err(JsError::new(&format!("a segment needs {len} prompt tokens; the limit is {MAX_CONTEXT}")));
            }
            history.remove(0);
        }
    }

    /// Greedy generation of one segment. The text goes to `on_text` as it is produced.
    async fn decode_segment(
        &mut self,
        id: SeqId,
        first: Tensor,
        max_new: usize,
        on_text: &Function,
        chars: &mut usize,
    ) -> Result<(String, &'static str, u32), JsError> {
        let mut logits = first;
        let mut decoder = StreamDecoder::new();
        let mut translated = String::new();
        let mut output_tokens = 0u32;
        let mut stop = "limit";
        let result = async {
            while (output_tokens as usize) < max_new {
                let next = self.model.backend.argmax_async(&logits).await;
                if next == EOS_ID {
                    stop = "eos";
                    break;
                }
                if next == TAG_ID {
                    stop = "tag";
                    break;
                }
                let delta = decoder.push(&self.tokenizer, next);
                if !delta.is_empty() {
                    *chars += delta.chars().count();
                    translated.push_str(&delta);
                    emit(on_text, &delta)?;
                }
                output_tokens += 1;
                if output_tokens as usize >= max_new {
                    break;
                }
                logits = self.model.forward(&[(id, &[next])], false).map_err(jserr)?;
            }
            // A character split across byte tokens is held until the next token; there is
            // no next token at the end, so emit its replacement here.
            let tail = decoder.finish();
            if !tail.is_empty() {
                *chars += tail.chars().count();
                translated.push_str(&tail);
                emit(on_text, &tail)?;
            }
            Ok::<(), JsError>(())
        }
        .await;
        match result {
            Ok(()) => {
                self.model.finish(id);
                Ok((translated, stop, output_tokens))
            }
            Err(e) => {
                self.model.free_seq(id);
                Err(e)
            }
        }
    }
}

/// Emits text to the page.
fn emit(on_text: &Function, text: &str) -> Result<(), JsError> {
    on_text
        .call1(&JsValue::NULL, &JsValue::from_str(text))
        .map(|_| ())
        .map_err(|e| JsError::new(&format!("on_text callback failed: {e:?}")))
}

/// The newest pairs within the history budget (at most two).
fn select_history(all: &[Pair]) -> Vec<Pair> {
    let mut selected = Vec::new();
    let mut chars = 0;
    for pair in all.iter().rev().take(2) {
        if validate_history_pair(pair).is_err() {
            continue;
        }
        let n = pair.source.chars().count() + pair.target.chars().count();
        if !selected.is_empty() && chars + n > HISTORY_CHARS {
            break;
        }
        selected.push(pair.clone());
        chars += n;
    }
    selected.reverse();
    selected
}

/// What one translation cost (shown under the output).
#[wasm_bindgen]
pub struct Stats {
    prompt_tokens: u32,
    output_tokens: u32,
    max_new: u32,
    segments: u32,
    stop: String,
    chars: u32,
    prefill_ms: f64,
    decode_ms: f64,
}

#[wasm_bindgen]
impl Stats {
    #[wasm_bindgen(getter)]
    pub fn prompt_tokens(&self) -> u32 {
        self.prompt_tokens
    }

    #[wasm_bindgen(getter)]
    pub fn output_tokens(&self) -> u32 {
        self.output_tokens
    }

    /// How many tokens the last segment's answer was allowed (prompt + answer <= 4096).
    #[wasm_bindgen(getter)]
    pub fn max_new(&self) -> u32 {
        self.max_new
    }

    /// How many segments the input was split into.
    #[wasm_bindgen(getter)]
    pub fn segments(&self) -> u32 {
        self.segments
    }

    /// `tag` or `eos` at the model's stop token, `limit` at the page's output budget.
    #[wasm_bindgen(getter)]
    pub fn stop(&self) -> String {
        self.stop.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn chars(&self) -> u32 {
        self.chars
    }

    #[wasm_bindgen(getter)]
    pub fn prefill_ms(&self) -> f64 {
        self.prefill_ms
    }

    #[wasm_bindgen(getter)]
    pub fn decode_ms(&self) -> f64 {
        self.decode_ms
    }

    /// Output tokens per second over the decode loop.
    #[wasm_bindgen(getter)]
    pub fn tokens_per_second(&self) -> f64 {
        if self.decode_ms <= 0.0 { 0.0 } else { f64::from(self.output_tokens) * 1e3 / self.decode_ms }
    }
}
