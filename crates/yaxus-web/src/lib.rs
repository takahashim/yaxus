//! yaxus in the browser: a chosen GGUF file is read into wasm memory, the weights go to
//! WebGPU, and translation streams back into the page. It runs in a dedicated worker
//! (`web/local/worker.js`), so a long load or translation never blocks the page.
//!
//! The browser path runs the CLI's engine (`yaxus_model::Engine`, awaited step by step,
//! since WebGPU cannot be waited on) with one request at a time, without a draft model.
//! Everything above the GPU is the same code as the CLI (`yaxus-model` and the kernels
//! of `yaxus-backend-gpu`).
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
use yaxus_backend_gpu::GpuBackend;
use yaxus_model::{Engine, EngineConfig, Finish, Model, ModelOptions, Quant, gguf::Gguf};
use yaxus_pfml::{Form, TranslationRequest, segments::split_for_context};
use yaxus_runtime::document::{DocEvent, DocJob, DocOptions, DocRunError, HistoryRules, run_document_async};
use yaxus_runtime::memory::Memory;
use yaxus_tokenizer::{TokenKind, Tokenizer};

/// The longest output one segment asks for.
const MAX_NEW_TOKENS: usize = 1024;
/// Prompt plus output of one segment; the KV arena and RoPE tables are this long.
const MAX_CONTEXT: usize = 4096;
/// Input tokens one segment aims for. A paragraph that fits stays whole; a longer one is
/// split at sentences. Paragraphs/history/output all live inside `MAX_CONTEXT`.
const SEGMENT_INPUT_TOKENS: usize = 768;
/// Characters of earlier source/translation pairs sent as history (`HistoryWindow`, as
/// the CLI's document mode; smaller than its default to leave room in `MAX_CONTEXT`).
const HISTORY_CHARS: usize = 1600;
/// One request at a time, no response cache (a repeated translation runs again, so a
/// fault of the device shows), no draft model.
const ENGINE: EngineConfig = EngineConfig {
    max_streams: 1,
    decode_rows: 1,
    busy_prefill_rows: 64,
    draft_k: 7,
    spec_max_streams: 1,
    hint_k: 7,
    response_cache: 0,
};

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
    let cfg = g.config().map_err(jserr)?;
    let weights = g.weights().map_err(jserr)?;

    report(&on_progress, "トークナイザを構築しています…");
    let kinds = vocab.types.iter().copied().map(TokenKind::from_ggml_type).collect();
    let tokenizer = Tokenizer::from_vocab(vocab.tokens, kinds, vocab.scores).map_err(jserr)?;

    // The largest buffer a single binding has to cover (for the 0.8B Q4_K_M, the Q6_K
    // output matrix's 124 MB of 6-bit values). A device that cannot bind a matrix's
    // buffer gets that matrix at Q4.
    report(&on_progress, "WebGPU デバイスを開いています…");
    let need_binding = GpuBackend::largest_weight_bytes(&weights);
    let backend = GpuBackend::new_async(need_binding, need_binding.max(64 << 20)).await.map_err(jserr)?;
    let name = backend.caps().name.clone();
    let max_binding = backend.max_binding_bytes();

    report(&on_progress, "重みを GPU に転送しています…");
    let opts = ModelOptions {
        kv_tokens: MAX_CONTEXT,
        max_pos: MAX_CONTEXT,
        prefix_cache: true,
        ..ModelOptions::new(Quant::Auto)
    };
    // WebGPU reports a failed allocation after the call that made it: the error scope
    // around the uploads makes such a failure a load error, not a broken first
    // translation.
    let watch = backend.watch_errors();
    let mut model = Model::from_parts(cfg, weights, backend, &opts).map_err(jserr)?;
    watch.finish().await.map_err(jserr)?;
    model.backend.sync_async().await.map_err(jserr)?;
    report(&on_progress, "準備できました");
    let load_ms = total.elapsed().as_secs_f64() * 1e3;
    let reduced_to_q4 = model.backend.reduced_to_q4() > 0;
    let engine = Engine::new(model, None, ENGINE).map_err(jserr)?;
    Ok(Translator { engine, tokenizer, name, load_ms, max_binding, reduced_to_q4, broken: false })
}

/// One loaded model. Only one translation runs at a time (the page's button is disabled
/// while it does).
#[wasm_bindgen]
pub struct Translator {
    engine: Engine<GpuBackend>,
    tokenizer: Tokenizer,
    name: String,
    load_ms: f64,
    max_binding: u64,
    reduced_to_q4: bool,
    /// Set by an error of the GPU (see the crate documentation).
    broken: bool,
}

/// An error of `translate`, by whether the model is still usable.
enum Failure {
    /// An error of this request (its input, a limit of the KV or the context): the model
    /// and its KV bookkeeping are as before.
    Request(JsError),
    /// An error of the device, which only a read-back reports (the recorded passes do
    /// not fail on their own): the model cannot be trusted any more.
    Device(JsError),
}

fn device_err(e: impl Display) -> Failure {
    Failure::Device(jserr(e))
}

fn request_err(e: impl Display) -> Failure {
    Failure::Request(jserr(e))
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
        // The whole request runs inside an error scope, settled before it returns: an
        // error WebGPU reports late fails this translation, not the next one (`Fault`
        // and the read-backs only see what has arrived by then).
        let watch = self.engine.backend().watch_errors();
        let result = self.translate_inner(&text, &target, form.as_deref(), &on_text, on_progress.as_ref()).await;
        let scope = watch.finish().await;
        match (result, scope) {
            (Err(Failure::Device(e)), _) => {
                self.broken = true;
                Err(e)
            }
            (_, Err(e)) => {
                self.broken = true;
                Err(jserr(e))
            }
            (Ok(stats), Ok(())) => Ok(stats),
            (Err(Failure::Request(e)), Ok(())) => Err(e),
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
            return Err(request_err(format!("unsupported target language {target:?}")));
        }
        let form = match form {
            _ if target != "ja" => None,
            None | Some("") | Some("none") => None,
            Some("polite") => Some(Form::Polite),
            Some("plain") => Some(Form::Plain),
            Some(other) => return Err(request_err(format!("unknown form {other:?}"))),
        };
        let segments = split_for_context(text, &self.tokenizer, SEGMENT_INPUT_TOKENS);
        if segments.is_empty() {
            return Err(request_err("the text is empty or only whitespace"));
        }
        // Sentences of one paragraph are translated apart and joined: with a space in
        // English, directly in Japanese.
        let sentence_gap = if target == "en" { " " } else { "" };
        let total = segments.len();

        // One chain: each segment is translated with the earlier ones as history, as the
        // CLI does for a document, with the page's history rules.
        let opts = DocOptions {
            request: TranslationRequest { target_lang: target.into(), form, ..Default::default() },
            history_chars: HISTORY_CHARS,
            chains: 1,
            neighbor_context: false,
            max_new_tokens: MAX_NEW_TOKENS,
            history_rules: HistoryRules { skip_truncated: true, trim: true },
        };
        let job = DocJob::new(segments.iter().map(|s| s.text.clone()).collect(), opts).with_max_context(MAX_CONTEXT);
        job.validate(&self.tokenizer).map_err(request_err)?;

        let progress = |done: usize| {
            if let Some(progress) = on_progress {
                let _ =
                    progress.call2(&JsValue::NULL, &JsValue::from_f64(done as f64), &JsValue::from_f64(total as f64));
            }
        };
        progress(0);

        let mut out = Output { on_text, chars: 0, owed: "" };
        // The separator a segment owes is set before its text, in order, also for a
        // segment that has none.
        let mut owed_upto = 0;
        let mut write_failure = None;
        let (mut prompt_tokens, mut output_tokens, mut decode_steps, mut reused_tokens) = (0u32, 0u32, 0u32, 0u32);
        let (mut prefill_ms, mut decode_ms) = (0.0, 0.0);
        let (mut any_limit, mut last_max_new) = (false, MAX_NEW_TOKENS as u32);

        // The page keeps no translation memory: a repeated translation is computed again.
        let mut memory = Memory::new(0);
        let result = run_document_async(&mut self.engine, &self.tokenizer, job, &mut memory, &mut |event| {
            let segment = match &event {
                DocEvent::Delta { segment, .. } => *segment,
                DocEvent::SegmentDone(end) => end.segment,
            };
            while owed_upto <= segment {
                out.owe(if segments[owed_upto].paragraph { "\n\n" } else { sentence_gap });
                owed_upto += 1;
            }
            match event {
                DocEvent::Delta { text, .. } => {
                    if let Err(e) = out.text(text) {
                        write_failure = Some(e);
                        return false;
                    }
                }
                DocEvent::SegmentDone(end) => {
                    // As the session measured them: the prompt's pass ends with the first
                    // token, until then the GPU has only been given the work.
                    prefill_ms += end.prefill.as_secs_f64() * 1e3;
                    decode_ms += end.decode.as_secs_f64() * 1e3;
                    prompt_tokens += end.prompt_tokens as u32;
                    reused_tokens += end.reused as u32;
                    output_tokens += end.generated as u32;
                    decode_steps += end.generated.saturating_sub(1) as u32;
                    last_max_new = MAX_NEW_TOKENS.min(MAX_CONTEXT.saturating_sub(end.prompt_tokens)).max(1) as u32;
                    any_limit |= end.finish == Finish::Length;
                    progress(end.segment + 1);
                }
            }
            true
        })
        .await;
        if let Some(e) = write_failure {
            return Err(e);
        }
        match result {
            Ok(_) => {}
            // The engine's step failed: only a read-back reports an error of the device.
            Err(DocRunError::Engine(e)) => return Err(device_err(format!("{e:#}"))),
            Err(DocRunError::Segment(message)) => return Err(request_err(message)),
        }
        Ok(Stats {
            prompt_tokens,
            output_tokens,
            max_new: last_max_new,
            segments: total as u32,
            stop: (if any_limit { "limit" } else { "tag" }).into(),
            chars: out.chars as u32,
            decode_steps,
            reused_tokens,
            prefill_ms,
            decode_ms,
        })
    }
}

/// The page's output. The separator a segment owes the text before it is written just
/// before its own first text, so a segment that produces nothing adds no separator.
struct Output<'a> {
    on_text: &'a Function,
    /// Characters sent so far.
    chars: usize,
    owed: &'static str,
}

impl Output<'_> {
    /// The next segment follows the text so far after `separator` (a paragraph break
    /// owed by a segment with no text is kept).
    fn owe(&mut self, separator: &'static str) {
        if self.chars > 0 && (self.owed.is_empty() || separator == "\n\n") {
            self.owed = separator;
        }
    }

    fn text(&mut self, text: &str) -> Result<(), Failure> {
        if text.is_empty() {
            return Ok(());
        }
        let owed = std::mem::take(&mut self.owed);
        for part in [owed, text] {
            if part.is_empty() {
                continue;
            }
            self.chars += part.chars().count();
            self.on_text
                .call1(&JsValue::NULL, &JsValue::from_str(part))
                .map_err(|e| request_err(format!("on_text callback failed: {e:?}")))?;
        }
        Ok(())
    }
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
    /// Decode passes (see [`Stats::tokens_per_second`]).
    decode_steps: u32,
    /// Prompt tokens served from the KV of earlier segments.
    reused_tokens: u32,
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

    /// Prompt tokens that were not computed again because an earlier segment's prompt
    /// began the same way.
    #[wasm_bindgen(getter)]
    pub fn reused_tokens(&self) -> u32 {
        self.reused_tokens
    }

    #[wasm_bindgen(getter)]
    pub fn prefill_ms(&self) -> f64 {
        self.prefill_ms
    }

    #[wasm_bindgen(getter)]
    pub fn decode_ms(&self) -> f64 {
        self.decode_ms
    }

    /// Tokens per second over the decode loop: the passes after each segment's first
    /// token (which comes with the prefill), over their time.
    #[wasm_bindgen(getter)]
    pub fn tokens_per_second(&self) -> f64 {
        if self.decode_ms <= 0.0 { 0.0 } else { f64::from(self.decode_steps) * 1e3 / self.decode_ms }
    }
}
