//! yaxus in the browser: a chosen GGUF file is read into wasm memory, the weights go to
//! WebGPU, and translation streams back into the page. It runs in a dedicated worker
//! (`www/worker.js`), so a long load or translation never blocks the page.
//!
//! The browser path runs the CLI's engine (`yaxus_model::Engine`, awaited step by step,
//! since WebGPU cannot be waited on) without a draft model. One translation runs at a
//! time; with several chains, its segments advance side by side in the same passes.
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
use yaxus_runtime::document::{Chains, DocEvent, DocJob, DocOptions, DocRunError, DocRunner, HistoryRules};
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
/// Segments of one document translated side by side (the page's chains), at most. The
/// KV arena holds this many segments' contexts.
const MAX_CHAINS: usize = 4;
/// Up to `MAX_CHAINS` requests side by side, no response cache (a repeated translation
/// runs again, so a fault of the device shows), no draft model.
const ENGINE: EngineConfig = EngineConfig {
    max_streams: MAX_CHAINS,
    decode_rows: MAX_CHAINS,
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
        kv_tokens: MAX_CONTEXT * MAX_CHAINS,
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
    Ok(Translator {
        engine,
        tokenizer,
        name,
        load_ms,
        max_binding,
        reduced_to_q4,
        broken: false,
        chains: Chains::Auto,
        neighbor_context: false,
    })
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
    /// Chains of the next translation: the segments translated side by side.
    chains: Chains,
    /// Give each segment the source of its neighbours as context.
    neighbor_context: bool,
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

    /// How many segments of a document are translated side by side: 1 to 4, or 0 for the
    /// CLI's and the server's rule (a chain per 4 segments), at most 4. The rule by default.
    /// With more, a segment's history is only that of its own chain, so the translation
    /// differs from the one in a single chain.
    pub fn set_chains(&mut self, chains: u32) {
        // (at most as many as the engine and its KV arena take)
        self.chains = match Chains::from(chains as usize) {
            Chains::Auto => Chains::Auto,
            Chains::Fixed(n) => Chains::from(n.get().min(MAX_CHAINS)),
        };
    }

    /// Whether each segment of a document gets the source of the segments before and after
    /// it as context (off by default). The prompts differ from those without it, so the
    /// translation does too.
    pub fn set_neighbor_context(&mut self, on: bool) {
        self.neighbor_context = on;
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
    /// sentences; each segment is translated with the earlier ones as history.
    ///
    /// `on_plan` is called once, before any text, with what the page needs to lay the
    /// output out: for each segment, the separator that goes before its text (a blank line
    /// after a paragraph, a space or nothing between sentences). `on_text` receives
    /// `(segment, text)` as the text is produced: the text of different segments may
    /// interleave, so the page keeps each segment's text apart. `cancelled` is asked
    /// between the engine's steps: once it says true, the translation stops (what it has
    /// produced stays shown) and the result says so. `progress` (optional) gets
    /// `(segments_done, segments_total)`.
    pub async fn translate(
        &mut self,
        text: String,
        target: String,
        form: Option<String>,
        callbacks: &PageCallbacks,
    ) -> Result<Stats, JsError> {
        if self.broken {
            return Err(JsError::new("the GPU failed earlier; load the model again"));
        }
        // The whole request runs inside an error scope, settled before it returns: an
        // error WebGPU reports late fails this translation, not the next one (`Fault`
        // and the read-backs only see what has arrived by then).
        let watch = self.engine.backend().watch_errors();
        let (plan, text_fn, cancelled, progress) =
            (callbacks.plan(), callbacks.text(), callbacks.cancelled(), callbacks.progress());
        let hooks = Callbacks { plan: &plan, text: &text_fn, cancelled: &cancelled, progress: progress.as_ref() };
        let result = self.translate_inner(&text, &target, form.as_deref(), &hooks).await;
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
        page: &Callbacks<'_>,
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
            chains: Chains::from(self.chains.count(total).min(MAX_CHAINS)),
            neighbor_context: self.neighbor_context,
            max_new_tokens: MAX_NEW_TOKENS,
            history_rules: HistoryRules { skip_truncated: true, trim: true },
        };
        let job = DocJob::new(segments.iter().map(|s| s.text.clone()).collect(), opts).with_max_context(MAX_CONTEXT);
        job.validate(&self.tokenizer).map_err(request_err)?;

        let breaks: js_sys::Array =
            segments.iter().map(|s| JsValue::from_str(if s.paragraph { "\n\n" } else { sentence_gap })).collect();
        page.plan.call1(&JsValue::NULL, &breaks).map_err(|e| request_err(format!("on_plan callback failed: {e:?}")))?;
        let progress = |done: usize| {
            if let Some(progress) = page.progress {
                let _ =
                    progress.call2(&JsValue::NULL, &JsValue::from_f64(done as f64), &JsValue::from_f64(total as f64));
            }
        };
        progress(0);

        let mut write_failure = None;
        let (mut prompt_tokens, mut output_tokens, mut reused_tokens) = (0u32, 0u32, 0u32);
        let started = Instant::now();
        let (mut prefill_ms, mut decode_ms) = (0.0, 0.0);
        let (mut any_limit, mut last_max_new) = (false, MAX_NEW_TOKENS as u32);
        // Segments finish out of order when chains run side by side: count them.
        let mut segments_done = 0usize;

        // The page keeps no translation memory: a repeated translation is computed again.
        let mut memory = Memory::new(0);
        let mut was_cancelled = false;
        let mut cancelled = || {
            let yes = page.cancelled.call0(&JsValue::NULL).map(|v| v.is_truthy()).unwrap_or(false);
            was_cancelled |= yes;
            yes
        };
        let result = DocRunner::new(&mut self.engine, &self.tokenizer, &mut memory)
            .cancel_when(&mut cancelled)
            .run(job, &mut |event| {
                match event {
                    DocEvent::Delta { segment, text } => {
                        let sent = page.text.call2(
                            &JsValue::NULL,
                            &JsValue::from_f64(segment as f64),
                            &JsValue::from_str(text),
                        );
                        if let Err(e) = sent {
                            write_failure = Some(request_err(format!("on_text callback failed: {e:?}")));
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
                        last_max_new = MAX_NEW_TOKENS.min(MAX_CONTEXT.saturating_sub(end.prompt_tokens)).max(1) as u32;
                        any_limit |= end.finish == Finish::Length;
                        segments_done += 1;
                        progress(segments_done);
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
            Err(e @ (DocRunError::Prompt(_) | DocRunError::Segment { .. })) => return Err(request_err(e)),
        }
        Ok(Stats {
            prompt_tokens,
            output_tokens,
            max_new: last_max_new,
            segments: total as u32,
            stop: (if any_limit { "limit" } else { "tag" }).into(),
            elapsed_ms: started.elapsed().as_secs_f64() * 1e3,
            cancelled: was_cancelled,
            reused_tokens,
            prefill_ms,
            decode_ms,
        })
    }
}

/// The functions the page gives `translate`, as one object: `{plan, text, cancelled, progress}`.
#[wasm_bindgen]
extern "C" {
    pub type PageCallbacks;
    #[wasm_bindgen(method, getter)]
    fn plan(this: &PageCallbacks) -> Function;
    #[wasm_bindgen(method, getter)]
    fn text(this: &PageCallbacks) -> Function;
    #[wasm_bindgen(method, getter)]
    fn cancelled(this: &PageCallbacks) -> Function;
    #[wasm_bindgen(method, getter)]
    fn progress(this: &PageCallbacks) -> Option<Function>;
}

/// What the page gave `translate`, unpacked.
struct Callbacks<'a> {
    plan: &'a Function,
    text: &'a Function,
    cancelled: &'a Function,
    progress: Option<&'a Function>,
}

/// What one translation cost (shown under the output).
#[wasm_bindgen]
pub struct Stats {
    prompt_tokens: u32,
    output_tokens: u32,
    max_new: u32,
    segments: u32,
    stop: String,
    /// The translation from start to end (segments translated side by side overlap).
    elapsed_ms: f64,
    /// Stopped by the page before it was done.
    cancelled: bool,
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

    /// Prompt tokens that were not computed again because an earlier segment's prompt
    /// began the same way.
    #[wasm_bindgen(getter)]
    pub fn reused_tokens(&self) -> u32 {
        self.reused_tokens
    }

    /// The translation was stopped by the page: what was produced is partial.
    #[wasm_bindgen(getter)]
    pub fn cancelled(&self) -> bool {
        self.cancelled
    }

    /// Time from the start to the end of the translation.
    #[wasm_bindgen(getter)]
    pub fn elapsed_ms(&self) -> f64 {
        self.elapsed_ms
    }

    /// The prompt processing of all segments, added up: with segments translated side by
    /// side it is more than the time that passed.
    #[wasm_bindgen(getter)]
    pub fn prefill_ms(&self) -> f64 {
        self.prefill_ms
    }

    /// The decoding of all segments, added up (see [`Stats::prefill_ms`]).
    #[wasm_bindgen(getter)]
    pub fn decode_ms(&self) -> f64 {
        self.decode_ms
    }

    /// Output tokens per second over the whole translation, prompts included.
    #[wasm_bindgen(getter)]
    pub fn tokens_per_second(&self) -> f64 {
        if self.elapsed_ms <= 0.0 { 0.0 } else { f64::from(self.output_tokens) * 1e3 / self.elapsed_ms }
    }
}
