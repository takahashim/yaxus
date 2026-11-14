//! yaxus in the browser: a chosen GGUF file is read into wasm memory, the weights go to
//! WebGPU, and translation streams back into the page. It runs in a dedicated worker
//! (`web/local/worker.js`), so a long load or translation never blocks the page.
//!
//! The browser path deliberately keeps one request and one model: no engine, draft
//! model, document chains, batching or KV-prefix cache. Everything above the GPU is the
//! same code as the CLI (`yaxus-model` and the kernels of `yaxus-backend-gpu`).
//!
//! Errors come in two kinds. A bad input (no text, a file that is not a PLaMo GGUF) is an
//! ordinary error and the model stays usable. An error of the GPU (a failed allocation,
//! a lost device) or a panic leaves the device or the wasm instance in an unknown state:
//! [`Translator::broken`] turns true and the page starts a new worker.
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

/// The bytes of the GGUF file, filled chunk by chunk from the worker's file stream, so
/// the file is in memory once (in wasm), not also as one JavaScript array.
#[wasm_bindgen]
pub struct ModelBuffer {
    bytes: Vec<u8>,
    len: usize,
}

#[wasm_bindgen]
impl ModelBuffer {
    /// Reserves room for a file of `len` bytes; fails rather than aborting when wasm
    /// memory cannot grow that far.
    #[wasm_bindgen(constructor)]
    pub fn new(len: f64) -> Result<ModelBuffer, JsError> {
        let len = len as usize;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(len)
            .map_err(|_| JsError::new(&format!("not enough memory for a {} MB file", len >> 20)))?;
        Ok(ModelBuffer { bytes, len })
    }

    /// Appends one chunk of the file.
    pub fn push(&mut self, chunk: &[u8]) -> Result<(), JsError> {
        if self.bytes.len() + chunk.len() > self.len {
            return Err(JsError::new("the file grew while it was read"));
        }
        self.bytes.extend_from_slice(chunk);
        Ok(())
    }
}

/// Loads a GGUF file (read into `file`) and opens WebGPU.
///
/// `on_progress` is called with one short message per phase. The errors are the same
/// ones the CLI reports (not a PLaMo GGUF, a tensor missing, a device too small), and a
/// device that failed while the weights were uploaded.
#[wasm_bindgen]
pub async fn load_model(file: ModelBuffer, on_progress: Function) -> Result<Translator, JsError> {
    let total = Instant::now();
    if file.bytes.len() != file.len {
        return Err(JsError::new("the file was not read to the end"));
    }
    report(&on_progress, "GGUF を解析しています…");
    let g = Gguf::parse(file.bytes).map_err(jserr)?;
    let vocab = g.vocab().map_err(jserr)?;
    g.config().map_err(jserr)?;

    report(&on_progress, "トークナイザを構築しています…");
    let kinds = vocab.types.iter().copied().map(TokenKind::from_ggml_type).collect();
    let tokenizer = Tokenizer::from_vocab(vocab.tokens, kinds, vocab.scores).map_err(jserr)?;

    // The largest buffer a single binding has to cover (for the 0.8B Q4_K_M, the Q6_K
    // output matrix's 124 MB of 6-bit values). A device that cannot bind a matrix's
    // buffer gets that matrix at Q4.
    report(&on_progress, "WebGPU デバイスを開いています…");
    let need_binding = GpuBackend::largest_weight_bytes(&g.weights().map_err(jserr)?);
    let backend = GpuBackend::new_async(need_binding, need_binding.max(64 << 20)).await.map_err(jserr)?;
    let name = backend.caps().name.clone();
    let max_binding = backend.max_binding_bytes();

    report(&on_progress, "重みを GPU に転送しています…");
    let opts = ModelOptions {
        kv_tokens: MAX_CONTEXT,
        max_pos: MAX_CONTEXT,
        prefix_cache: false,
        ..ModelOptions::new(Quant::Auto)
    };
    // WebGPU reports a failed allocation after the call that made it: the error scope
    // around the uploads makes such a failure a load error, not a broken first
    // translation.
    let watch = backend.watch_errors();
    let mut model = Model::from_gguf(g, backend, &opts).map_err(jserr)?;
    watch.finish().await.map_err(jserr)?;
    model.backend.sync_async().await.map_err(jserr)?;
    report(&on_progress, "準備できました");
    let load_ms = total.elapsed().as_secs_f64() * 1e3;
    let reduced_to_q4 = model.backend.reduced_to_q4() > 0;
    Ok(Translator { model, tokenizer, name, load_ms, max_binding, reduced_to_q4, broken: false })
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
    reduced_to_q4: bool,
    /// Set by an error of the GPU (see the crate documentation).
    broken: bool,
}

/// An error of `translate`: whether the device (and so the model) is still usable.
enum Failure {
    Input(JsError),
    Gpu(JsError),
}

fn gpu(e: impl Display) -> Failure {
    Failure::Gpu(jserr(e))
}

fn input(e: impl Display) -> Failure {
    Failure::Input(jserr(e))
}

#[wasm_bindgen]
impl Translator {
    /// The WebGPU adapter's name (often empty: browsers hide it).
    #[wasm_bindgen(getter)]
    pub fn device(&self) -> String {
        self.name.clone()
    }

    /// True after an error of the GPU: the model cannot be used again.
    #[wasm_bindgen(getter)]
    pub fn broken(&self) -> bool {
        self.broken
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

    /// True when a matrix (for the 0.8B, the output matrix under a 124 MB binding) was loaded
    /// as Q4 because the storage binding limit is smaller than it.
    #[wasm_bindgen(getter)]
    pub fn reduced_to_q4(&self) -> bool {
        self.reduced_to_q4
    }

    /// Translates `text` into `target` (`ja` or `en`; the source language is left to
    /// the model, as the CLI does without `--from`). `form` (`polite`, `plain`, or
    /// none) applies to Japanese output. Long text is split into paragraphs and
    /// sentences; each segment is translated in order, with the previous segments as
    /// history, and `on_text` receives text as it is produced. `on_progress` gets
    /// `(segments_done, segments_total)`.
    pub async fn translate(
        &mut self,
        text: String,
        target: String,
        form: Option<String>,
        on_text: Function,
        on_progress: Option<Function>,
    ) -> Result<Stats, JsError> {
        if self.broken {
            return Err(JsError::new("the GPU failed earlier; load the model again"));
        }
        match self.translate_inner(&text, &target, form.as_deref(), &on_text, on_progress.as_ref()).await {
            Ok(stats) => Ok(stats),
            Err(Failure::Input(e)) => Err(e),
            Err(Failure::Gpu(e)) => {
                self.broken = true;
                Err(e)
            }
        }
    }
}

impl Translator {
    async fn translate_inner(
        &mut self,
        text: &str,
        target: &str,
        form: Option<&str>,
        on_text: &Function,
        on_progress: Option<&Function>,
    ) -> Result<Stats, Failure> {
        if !matches!(target, "ja" | "en") {
            return Err(input(format!("unsupported target language {target:?}")));
        }
        let form = match form {
            _ if target != "ja" => None,
            None | Some("") | Some("none") => None,
            Some("polite") => Some(Form::Polite),
            Some("plain") => Some(Form::Plain),
            Some(other) => return Err(input(format!("unknown form {other:?}"))),
        };
        let segments = split_for_context(text, &self.tokenizer, SEGMENT_INPUT_TOKENS);
        if segments.is_empty() {
            return Err(input("the text is empty or only whitespace"));
        }
        // Sentences of one paragraph are translated apart and joined: with a space in
        // English, directly in Japanese.
        let sentence_gap = if target == "en" { " " } else { "" };
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
            if let Some(progress) = on_progress {
                let _ = progress.call2(&JsValue::NULL, &JsValue::from_f64(i as f64), &JsValue::from_f64(total as f64));
            }
            let (ids, max_new) = self.fit_segment(segment, target, form, &history)?;
            if chars > 0 {
                let separator = if segment.paragraph { "\n\n" } else { sentence_gap };
                if !separator.is_empty() {
                    chars += separator.chars().count();
                    emit(on_text, separator)?;
                }
            }
            let prompt_len = ids.len() as u32;
            let started = Instant::now();
            let (id, first) = self.model.start(&ids).map_err(gpu)?;
            let decoded = self.decode_segment(id, first, max_new, on_text, &mut chars).await?;
            let (translated, stop, generated) = (decoded.text, decoded.stop, decoded.tokens);
            // The prompt's pass ends with the first token's read-back: until then the GPU
            // has only been given the work, so that is where prefill ends.
            let first_ms = (decoded.first_token - started).as_secs_f64() * 1e3;
            prefill_ms += first_ms;
            decode_ms += started.elapsed().as_secs_f64() * 1e3 - first_ms;
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
        if let Some(progress) = on_progress {
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
        target: &str,
        form: Option<Form>,
        earlier: &[Pair],
    ) -> Result<(Vec<u32>, usize), Failure> {
        let mut history = select_history(earlier);
        loop {
            let req = TranslationRequest {
                text: segment.text.clone(),
                target_lang: target.into(),
                source_lang: None,
                form,
                context: None,
                glossary: Vec::new(),
                history: history.clone(),
            };
            let req = req.validate().map_err(input)?;
            let ids = self.tokenizer.encode(&req.render(), false);
            let len = ids.len();
            if len < MAX_CONTEXT && (history.is_empty() || len + MIN_NEW <= MAX_CONTEXT) {
                return Ok((ids, MAX_NEW_TOKENS.min(MAX_CONTEXT - len)));
            }
            if history.is_empty() {
                return Err(input(format!("a segment needs {len} prompt tokens; the limit is {MAX_CONTEXT}")));
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
    ) -> Result<Decoded, Failure> {
        let mut logits = first;
        let mut decoder = StreamDecoder::new();
        let mut translated = String::new();
        let mut output_tokens = 0u32;
        let mut stop = "limit";
        let mut first_token = None;
        let result = async {
            while (output_tokens as usize) < max_new {
                let next = self.model.backend.argmax_async(&logits).await.map_err(gpu)?;
                first_token.get_or_insert_with(Instant::now);
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
                logits = self.model.forward(&[(id, &[next])], false).map_err(gpu)?;
            }
            // A character split across byte tokens is held until the next token; there is
            // no next token at the end, so emit its replacement here.
            let tail = decoder.finish();
            if !tail.is_empty() {
                *chars += tail.chars().count();
                translated.push_str(&tail);
                emit(on_text, &tail)?;
            }
            Ok::<(), Failure>(())
        }
        .await;
        match result {
            Ok(()) => {
                self.model.finish(id);
                let first_token = first_token.unwrap_or_else(Instant::now);
                Ok(Decoded { text: translated, stop, tokens: output_tokens, first_token })
            }
            Err(e) => {
                self.model.free_seq(id);
                Err(e)
            }
        }
    }
}

/// One segment's output.
struct Decoded {
    text: String,
    /// `tag`, `eos` or `limit` (see [`Stats::stop`]).
    stop: &'static str,
    tokens: u32,
    /// When the first token was read back.
    first_token: Instant,
}

/// Emits text to the page.
fn emit(on_text: &Function, text: &str) -> Result<(), Failure> {
    on_text
        .call1(&JsValue::NULL, &JsValue::from_str(text))
        .map(|_| ())
        .map_err(|e| input(format!("on_text callback failed: {e:?}")))
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

    /// Output tokens per second over the decode loop (the first token of each segment
    /// comes with the prefill, so it is not counted).
    #[wasm_bindgen(getter)]
    pub fn tokens_per_second(&self) -> f64 {
        let tokens = self.output_tokens.saturating_sub(self.segments);
        if self.decode_ms <= 0.0 { 0.0 } else { f64::from(tokens) * 1e3 / self.decode_ms }
    }
}
