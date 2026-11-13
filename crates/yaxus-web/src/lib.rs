//! The web frontend of yaxus: PLaMo 3 translation in the browser, on WebGPU.
//!
//! The page hands the model's `File` over here; the GGUF header is parsed from a
//! prefix of it (`GgufHeader`), and the tensors are then read one at a time,
//! uploaded to the GPU, and dropped — the model never sits in wasm memory whole.
//! Generation runs on `Model<GpuBackend>` directly, one forward pass per await:
//! the ops are recorded synchronously, and `argmax_rows_async` is the awaited
//! sync point of every step.
#![cfg(target_arch = "wasm32")]

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use wasm_bindgen::prelude::*;

use yaxus_backend_gpu::GpuBackend;
use yaxus_model::gguf::{GgufHeader, TensorInfo, Value};
use yaxus_model::{Model, ModelOptions, Quant, TensorRef};
use yaxus_pfml::TranslationRequest;
use yaxus_tokenizer::{EOS_ID, TAG_ID, TokenKind, Tokenizer};

/// Prefixes tried until the header parses (the vocabulary metadata can make it
/// several megabytes).
const PREFIX_SIZES: [usize; 5] = [1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20];

/// Reads `[start, start + len)` of `file` into wasm memory.
async fn read_range(file: &web_sys::File, start: usize, len: usize) -> anyhow::Result<Vec<u8>> {
    let slice = file
        .slice_with_f64_and_f64(start as f64, (start + len) as f64)
        .map_err(|e| anyhow::anyhow!("slicing the file at {start}: {e:?}"))?;
    let buf = wasm_bindgen_futures::JsFuture::from(slice.array_buffer())
        .await
        .map_err(|e| anyhow::anyhow!("reading the file at {start}: {e:?}"))?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

/// Reads the first `n` bytes of `file`.
async fn read_prefix(file: &web_sys::File, n: usize) -> anyhow::Result<Vec<u8>> {
    read_range(file, 0, n).await
}

/// Reads and parses the GGUF header: the prefix grows until it holds it.
async fn read_header(file: &web_sys::File) -> anyhow::Result<GgufHeader> {
    let size = file.size();
    let mut last = None;
    for &n in &PREFIX_SIZES {
        if n as f64 > size + 1.0 && last.is_some() {
            break;
        }
        let buf = read_prefix(file, n).await?;
        match GgufHeader::parse(&buf, Some(size as usize)) {
            Ok(h) => return Ok(h),
            Err(e) if format!("{e:#}").contains("truncated") => last = Some(e),
            Err(e) => return Err(e),
        }
    }
    Err(last.unwrap_or_else(|| anyhow::anyhow!("the header does not fit in the file")))
}

/// What the page shows about a picked GGUF file: its identity, quantization,
/// configuration, and where the tensors are.
#[wasm_bindgen]
pub struct GgufSummary {
    name: String,
    arch: String,
    quant: String,
    tensors: usize,
    file_bytes: u64,
    tensor_bytes: u64,
    vocab_size: usize,
    config: String,
    tensors_preview: String,
}

#[wasm_bindgen]
impl GgufSummary {
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn arch(&self) -> String {
        self.arch.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn quant(&self) -> String {
        self.quant.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn tensors(&self) -> usize {
        self.tensors
    }
    #[wasm_bindgen(getter)]
    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }
    #[wasm_bindgen(getter)]
    pub fn tensor_bytes(&self) -> u64 {
        self.tensor_bytes
    }
    #[wasm_bindgen(getter)]
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    #[wasm_bindgen(getter)]
    pub fn config(&self) -> String {
        self.config.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn tensors_preview(&self) -> String {
        self.tensors_preview.clone()
    }
}

/// Parses the header of `file` (a GGUF of a PLaMo 3 model). Errors carry the
/// reason ("not a GGUF file", a malformed header, an unsupported version, ...).
#[wasm_bindgen]
pub async fn summarize(file: web_sys::File) -> Result<GgufSummary, JsValue> {
    summarize_inner(&file).await.map_err(|e| JsValue::from_str(&format!("{e:#}")))
}

async fn summarize_inner(file: &web_sys::File) -> anyhow::Result<GgufSummary> {
    let header = read_header(file).await?;
    let name = header.meta.get("general.name").and_then(Value::as_str).unwrap_or("?").to_string();
    let arch = header.meta.get("general.architecture").and_then(Value::as_str).unwrap_or("?").to_string();
    let quant = header.main_type().map(|t| format!("{t:?}")).unwrap_or_else(|| "?".into());
    let tensor_bytes = header.tensors.iter().map(|t| t.len as u64).sum();
    let cfg = header.config()?;
    let config = format!(
        concat!(
            "{{\"hidden_size\": {}, \"num_hidden_layers\": {}, \"num_attention_heads\": {}, ",
            "\"num_key_value_heads\": {}, \"head_dim\": {}, \"intermediate_size\": {}, \"vocab_size\": {}, ",
            "\"rms_norm_eps\": {:?}, \"window_size\": {}, \"sliding_window_pattern\": {}}}"
        ),
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim,
        cfg.intermediate_size,
        cfg.vocab_size,
        cfg.rms_norm_eps,
        cfg.window_size,
        cfg.sliding_window_pattern,
    );
    let vocab_size = match header.meta.get("tokenizer.ggml.tokens") {
        Some(Value::Array(t)) => t.len(),
        _ => cfg.vocab_size,
    };
    // The largest tensors, the ones that will each take one streaming read.
    let mut big: Vec<&TensorInfo> = header.tensors.iter().collect();
    big.sort_by_key(|t| std::cmp::Reverse(t.len));
    big.truncate(8);
    let tensors_preview = big
        .iter()
        .map(|t| format!("{}  {:?}  {:?}  {:.1} MB", t.name, t.shape, t.dtype, t.len as f64 / 1e6))
        .collect::<Vec<_>>()
        .join("\n");
    Ok(GgufSummary {
        name,
        arch,
        quant,
        tensors: header.tensors.len(),
        file_bytes: file.size() as u64,
        tensor_bytes,
        vocab_size,
        config,
        tensors_preview,
    })
}

/// What a streamed translation took: the prefill (call → first token), the decode
/// rate, and the full text.
#[wasm_bindgen]
pub struct TranslateStats {
    text: String,
    tokens: u32,
    ttft_ms: f64,
    ms_per_token: f64,
}

#[wasm_bindgen]
impl TranslateStats {
    #[wasm_bindgen(getter)]
    pub fn text(&self) -> String {
        self.text.clone()
    }
    #[wasm_bindgen(getter)]
    pub fn tokens(&self) -> u32 {
        self.tokens
    }
    #[wasm_bindgen(getter)]
    pub fn ttft_ms(&self) -> f64 {
        self.ttft_ms
    }
    #[wasm_bindgen(getter)]
    pub fn ms_per_token(&self) -> f64 {
        self.ms_per_token
    }
}

/// A loaded model with its tokenizer, ready to translate. Loading reads the GGUF
/// file's tensors one at a time (`Model::load_with_async`).
#[wasm_bindgen]
pub struct Translator {
    model: Model<GpuBackend>,
    tok: Tokenizer,
}

/// Opens the WebGPU device, reads the GGUF header, and streams the tensors into
/// the GPU. `kv_tokens` sizes the KV arena (short texts: a few thousand).
#[wasm_bindgen]
pub async fn load_model(file: web_sys::File, kv_tokens: u32, precise: bool) -> Result<Translator, JsValue> {
    load_model_inner(file, kv_tokens as usize, precise).await.map_err(|e| JsValue::from_str(&format!("{e:#}")))
}

async fn load_model_inner(file: web_sys::File, kv_tokens: usize, precise: bool) -> anyhow::Result<Translator> {
    let header = read_header(&file).await?;
    let cfg = header.config()?;
    let vocab = header.vocab()?;
    let tok = Tokenizer::from_vocab(
        vocab.tokens,
        vocab.types.iter().map(|&t| TokenKind::from_ggml_type(t)).collect(),
        vocab.scores,
    )?;
    let mut backend = GpuBackend::new_web().await?;
    let quant = if precise { Quant::Bf16 } else { Quant::Auto };
    let opts = ModelOptions { quant, kv_tokens, max_pos: kv_tokens, prefix_cache: false };
    // Where each safetensors name lives in the file, and the offset its norm weights
    // carry when the converter baked them in.
    let idx_of: HashMap<String, (usize, Option<f32>)> =
        header.weight_list()?.into_iter().map(|(hf, i, off)| (hf, (i, off))).collect();
    let tensors: Arc<Vec<TensorInfo>> = Arc::new(header.tensors.clone());
    let file = Arc::new(file);
    // Whether the norm weights sit around their offsets (not around 0): the loader
    // reads the layer-0 `attn_norm` first, exactly like the file loader does.
    let an_idx = tensors
        .iter()
        .position(|t| t.name == "blk.0.attn_norm" || t.name == "blk.0.attn_norm.weight")
        .with_context(|| "blk.0.attn_norm missing")?;
    let an = &tensors[an_idx];
    let baked = header.baked_norms(&read_range(&file, an.start, an.len).await?);
    let tensors2 = tensors.clone();
    let file2 = file.clone();
    let n_tensors = tensors.len();
    let done = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let mut model = Model::load_with_async(cfg, backend, &opts, move |name| {
        let (file, tensors, idx_of, done) = (file2.clone(), tensors2.clone(), idx_of.clone(), done.clone());
        async move {
            let (idx, off) = idx_of.get(&name).with_context(|| format!("GGUF has no tensor for {name}"))?.clone();
            let t = &tensors[idx];
            let bytes = read_range(&file, t.start, t.len).await?;
            let minus = if baked && let Some(off) = off { off } else { 0.0 };
            done.set(done.get() + 1);
            if done.get() % 16 == 0 || done.get() == n_tensors {
                web_sys::console::log_1(&format!("loaded {}/{} tensors", done.get(), n_tensors).into());
            }
            Ok(TensorRef::from_owned(bytes, t.shape.clone(), t.dtype, minus))
        }
    })
    .await?;
    if precise {
        use yaxus_backend_gpu::{GemmKind, Gemv};
        model.backend.set_gemm(GemmKind::F32);
        model.backend.set_gemv(Gemv::F32);
    }
    Ok(Translator { model, tok })
}

#[wasm_bindgen]
impl Translator {
    /// Greedy generation with an awaited read-back per token and the incremental
    /// text pushed to the `on_token` JS callback as it completes
    /// (`function (delta) {}`). The timing: from the call to the first token is the
    /// prefill (TTFT), the rest is the decode (the per-token rate comes from its length).
    #[wasm_bindgen]
    pub async fn translate_stream(
        &mut self,
        text: String,
        target_lang: String,
        source_lang: Option<String>,
        max_new_tokens: u32,
        on_token: js_sys::Function,
    ) -> Result<TranslateStats, JsValue> {
        let mut on = move |delta: String| {
            let _ = on_token.call1(&JsValue::NULL, &delta.into());
        };
        self.translate_stream_inner(text, target_lang, source_lang, max_new_tokens as usize, &mut on)
            .await
            .map_err(|e| JsValue::from_str(&format!("{e:#}")))
    }

    async fn translate_stream_inner(
        &mut self,
        text: String,
        target_lang: String,
        source_lang: Option<String>,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(String),
    ) -> anyhow::Result<TranslateStats> {
        use web_time::Instant;
        let req = TranslationRequest { text, target_lang, source_lang, form: None, context: None, glossary: vec![], history: vec![] }
            .validate()?;
        // The prompt carries its own BOS (`add_bos = false`), as everywhere else.
        let prompt = self.tok.encode(&req.render(), false);
        let t0 = Instant::now();
        let (id, mut logits) = self.model.start(&prompt)?;
        let mut first = None;
        let mut out: Vec<u32> = Vec::new();
        let mut dec = yaxus_tokenizer::StreamDecoder::new();
        let r = async {
            while out.len() < max_new_tokens {
                let next = self.model.backend.argmax_rows_async(&logits).await;
                let next = *next.last().context("no logits row")?;
                if next == EOS_ID || next == TAG_ID {
                    break;
                }
                out.push(next);
                if first.is_none() {
                    first = Some(Instant::now());
                }
                let delta = dec.push(&self.tok, next);
                if !delta.is_empty() {
                    on_token(delta);
                }
                if out.len() == max_new_tokens {
                    break;
                }
                logits = self.model.forward(&[(id, &[next])], false)?;
            }
            Ok::<_, anyhow::Error>(out)
        }
        .await;
        match &r {
            Ok(_) => self.model.finish(id),
            Err(_) => self.model.free_seq(id),
        }
        let ids = r?;
        let tail = dec.finish();
        if !tail.is_empty() {
            on_token(tail);
        }
        let decode = first.map_or(std::time::Duration::ZERO, |f| Instant::now().saturating_duration_since(f));
        let n = ids.len();
        Ok(TranslateStats {
            text: self.tok.decode(&ids),
            tokens: n as u32,
            ttft_ms: first.map_or(0.0, |f| f.saturating_duration_since(t0).as_secs_f64() * 1e3),
            ms_per_token: if n > 0 { decode.as_secs_f64() * 1e3 / n as f64 } else { 0.0 },
        })
    }

    /// Translates one paragraph, greedy, and returns the whole text (the streaming
    /// API comes later). `source_lang` as the API's language code, or none for
    /// automatic detection.
    pub async fn translate(
        &mut self,
        text: String,
        target_lang: String,
        source_lang: Option<String>,
        max_new_tokens: u32,
    ) -> Result<String, JsValue> {
        self.translate_inner(text, target_lang, source_lang, max_new_tokens as usize)
            .await
            .map_err(|e| JsValue::from_str(&format!("{e:#}")))
    }

    async fn translate_inner(
        &mut self,
        text: String,
        target_lang: String,
        source_lang: Option<String>,
        max_new_tokens: usize,
    ) -> anyhow::Result<String> {
        let req = TranslationRequest {
            text,
            target_lang,
            source_lang,
            form: None,
            context: None,
            glossary: vec![],
            history: vec![],
        }
        .validate()?;
        // The prompt carries its own BOS (`add_bos = false`), as everywhere else.
        let prompt = self.tok.encode(&req.render(), false);
        let (id, mut logits) = self.model.start(&prompt)?;
        let r = async {
            let mut out: Vec<u32> = Vec::new();
            while out.len() < max_new_tokens {
                let next = self.model.backend.argmax_rows_async(&logits).await;
                let next = *next.last().context("no logits row")?;
                if next == EOS_ID || next == TAG_ID {
                    break;
                }
                out.push(next);
                logits = self.model.forward(&[(id, &[next])], false)?;
            }
            Ok::<_, anyhow::Error>(out)
        }
        .await;
        match &r {
            Ok(_) => self.model.finish(id),
            Err(_) => self.model.free_seq(id),
        }
        let ids = r?;
        let text = self.tok.decode(&ids);
        Ok(text)
    }

    /// Diagnostic: the same generation as `translate`, but reporting the prompt ids,
    /// the logits after the prompt, and the first tokens — to compare against the
    /// native path (greedy, so the numbers must match).
    pub async fn probe(&mut self, text: String, target_lang: String) -> Result<String, JsValue> {
        self.probe_inner(text, target_lang).await.map_err(|e| JsValue::from_str(&format!("{e:#}")))
    }

    /// The first forward pass op by op (read-backs after every op), for comparing
    /// against the native trace: the first divergent op is the broken one.
    pub async fn trace(&mut self, text: String, target_lang: String) -> Result<String, JsValue> {
        self.trace_inner(text, target_lang).await.map_err(|e| JsValue::from_str(&format!("{e:#}")))
    }

    async fn trace_inner(&mut self, text: String, target_lang: String) -> anyhow::Result<String> {
        let req = TranslationRequest { text, target_lang, source_lang: None, form: None, context: None, glossary: vec![], history: vec![] }
            .validate()?;
        let prompt = self.tok.encode(&req.render(), false);
        let (id, reused) = self.model.begin(&prompt)?;
        let mut reader = self.model.backend.share();
        let mut lines = Vec::new();
        {
            let items = [(id, &prompt[reused..])];
            let mut trace = |op: &str, first: &[f32], abs: f32| {
                lines.push(format!(
                    "{op}: [{:.6}, {:.6}, {:.6}, {:.6}] |x| = {abs:.6}",
                    first.first().copied().unwrap_or(0.0),
                    first.get(1).copied().unwrap_or(0.0),
                    first.get(2).copied().unwrap_or(0.0),
                    first.get(3).copied().unwrap_or(0.0),
                ));
            };
            self.model.forward_trace_async(&items, &mut reader, &mut trace).await?;
        }
        self.model.finish(id);
        Ok(lines.join("\n"))
    }

    /// Two traced forward passes in a row: the second pass reads a KV that the first
    /// already used, which is where a second-translation failure would show.
    pub async fn trace2(&mut self, text: String, target_lang: String) -> Result<String, JsValue> {
        self.trace2_inner(text, target_lang).await.map_err(|e| JsValue::from_str(&format!("{e:#}")))
    }

    async fn trace2_inner(&mut self, text: String, target_lang: String) -> anyhow::Result<String> {
        let req = TranslationRequest {
            text: text.clone(),
            target_lang: target_lang.clone(),
            source_lang: None,
            form: None,
            context: None,
            glossary: vec![],
            history: vec![],
        }
        .validate()?;
        let prompt = self.tok.encode(&req.render(), false);
        let mut lines = Vec::new();
        // Three generations with different prompts (the second translation of the page
        // returned nothing): then a traced pass to find where it breaks.
        for (i, (text2, to2)) in [
            ("The weather is very nice today, isn't it?".to_string(), "ja".to_string()),
            ("これはペンです。昨日、東京駅で友人と待ち合わせて、一緒に美術館へ行きました。".to_string(), "en".to_string()),
            ("I would like to book a table for two at seven thirty this evening, by the window if possible.".to_string(), "ja".to_string()),
        ]
        .into_iter()
        .enumerate()
        {
            let out = self.translate_inner(text2, to2, None, 256).await?;
            lines.push(format!("translate #{i} = {out:?}"));
        }
        let out = self.translate_inner(text, target_lang, None, 256).await?;
        lines.push(format!("translate #3 = {out:?}"));
        for run in 4..6 {
            let (id, reused) = self.model.begin(&prompt)?;
            let mut reader = self.model.backend.share();
            {
                let items = [(id, &prompt[reused..])];
                let mut trace = |op: &str, first: &[f32], abs: f32| {
                    lines.push(format!(
                        "run {run} {op}: [{:.4}, {:.4}, {:.4}, {:.4}] |x| = {abs:.4}",
                        first.first().copied().unwrap_or(0.0),
                        first.get(1).copied().unwrap_or(0.0),
                        first.get(2).copied().unwrap_or(0.0),
                        first.get(3).copied().unwrap_or(0.0),
                    ));
                };
                self.model.forward_trace_async(&items, &mut reader, &mut trace).await?;
            }
            self.model.finish(id);
            lines.push(format!("run {run} done"));
        }
        Ok(lines.join("\n"))
    }

    async fn probe_inner(&mut self, text: String, target_lang: String) -> anyhow::Result<String> {        let req = TranslationRequest { text, target_lang, source_lang: None, form: None, context: None, glossary: vec![], history: vec![] }
            .validate()?;
        let prompt = self.tok.encode(&req.render(), false);
        let mut lines = vec![format!(
            "prompt {} tok: {:?}",
            prompt.len(),
            &prompt[..prompt.len().min(12)]
        )];
        let (id, mut logits) = self.model.start(&prompt)?;
        let host = self.model.backend.to_host_async(&logits).await;
        lines.push(format!("logits[0..8] = {:?}", &host[..8]));
        let mut out: Vec<u32> = Vec::new();
        for _ in 0..12 {
            let next = *self
                .model
                .backend
                .argmax_rows_async(&logits)
                .await
                .last()
                .context("no logits row")?;
            if next == EOS_ID || next == TAG_ID {
                lines.push(format!("stop at {next}"));
                break;
            }
            out.push(next);
            lines.push(format!("tok {next} = {:?}", self.tok.id_to_token(next)));
            logits = self.model.forward(&[(id, &[next])], false)?;
            if out.len() <= 3 {
                let host = self.model.backend.to_host_async(&logits).await;
                lines.push(format!("  logits[0..6] = {:?}", &host[..6]));
            }
        }
        self.model.finish(id);
        lines.push(format!("out ids: {out:?}"));
        Ok(lines.join("\n"))
    }
}

/// A panic in a wasm call must not abort the page: report it like an error.
/// The page calls this once after `init`.
#[wasm_bindgen]
pub fn hook_panics() {
    std::panic::set_hook(Box::new(|info| web_sys::console::error_1(&format!("yaxus-web panic: {info}").into())));
}

/// Single-op diagnostics on synthetic data, for bisecting a wrong output in the
/// browser (the native test suite covers the same kernels on Vulkan/Metal).
/// Returns one "name: PASS/FAIL ..." line per test.
#[wasm_bindgen]
pub async fn run_diag() -> Result<String, JsValue> {
    run_diag_inner().await.map_err(|e| JsValue::from_str(&format!("{e:#}")))
}

async fn run_diag_inner() -> anyhow::Result<String> {
    use yaxus_backend_gpu::{GemmKind, Gemv};
    use yaxus_model::ggml::GgmlType;
    use yaxus_model::{Backend, PassMeta};

    let mut b = GpuBackend::new_web().await?;
    let mut lines: Vec<String> = Vec::new();
    let bf16_bytes = |v: &[f32]| -> Vec<u8> {
        v.iter().flat_map(|&x| half::bf16::from_f32(x).to_le_bytes()).collect()
    };
    let ok = |name: &str, got: &[f32], want: &[f32], tol: f32| -> String {
        let diff = got.iter().zip(want).map(|(&g, &w)| (g - w).abs()).fold(0f32, f32::max);
        let scale = want.iter().fold(0f32, |a, &w| a.max(w.abs())).max(1e-9);
        let pass = diff / scale < tol;
        let mut line =
            format!("{name}: {} (max diff {diff:.3e}, rel {:.1e})", if pass { "PASS" } else { "FAIL" }, diff / scale);
        if !pass {
            let (at, _) = got.iter().zip(want).enumerate().max_by_key(|(_, (g, w))| ((*g - **w).abs() * 1e9) as u64).unwrap();
            let sample = |v: &[f32]| v.iter().take(4).map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(", ");
            line += &format!(
                "  at[{at}] got={:.4} want={:.4}  got[0..4]=({}, …) want[0..4]=({}, …)",
                got[at], want[at],
                sample(got),
                sample(want)
            );
        }
        line
    };

    // rmsnorm (weights 1, offset 0): y = x / sqrt(mean(x²) + eps).
    {
        let (rows, cols) = (4usize, 64usize);
        let x: Vec<f32> = (0..rows * cols).map(|i| (i % 19) as f32 * 0.7 - 5.0).collect();
        let xt = b.debug_upload(&x, rows, cols);
        let w = b.load_weight(&TensorRef::from_owned(bf16_bytes(&vec![1.0; cols]), vec![cols], GgmlType::Bf16, 0.0), Quant::Bf16)?;
        let y = b.rmsnorm(&xt, &w, 0.0, 1e-6);
        let got = b.to_host_async(&y).await;
        let want: Vec<f32> = (0..rows)
            .map(|r| {
                let row = &x[r * cols..(r + 1) * cols];
                let ms = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
                let rms = (ms + 1e-6).sqrt();
                row.iter().map(|v| v / rms).collect::<Vec<f32>>()
            })
            .flatten()
            .collect();
        lines.push(ok("rmsnorm", &got, &want, 2e-4));
    }

    // rmsnorm with a nonzero offset and varying weights (the f32 scalars and the
    // bf16 weight rounding matter here).
    {
        let (rows, cols) = (4usize, 64usize);
        let x: Vec<f32> = (0..rows * cols).map(|i| (i % 19) as f32 * 0.7 - 5.0).collect();
        let w: Vec<f32> = (0..cols).map(|i| 0.9 + (i % 5) as f32 * 0.02).collect();
        let xt = b.debug_upload(&x, rows, cols);
        let wt = b.load_weight(&TensorRef::from_owned(bf16_bytes(&w), vec![cols], GgmlType::Bf16, 0.0), Quant::Bf16)?;
        let y = b.rmsnorm(&xt, &wt, 0.1, 1e-6);
        let got = b.to_host_async(&y).await;
        // The GPU sees the bf16-rounded weights: the reference uses them too.
        let wb: Vec<f32> = w.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect();
        let want: Vec<f32> = (0..rows)
            .map(|r| {
                let row = &x[r * cols..(r + 1) * cols];
                let ms = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
                let rms = (ms + 1e-6).sqrt();
                row.iter().zip(&wb).map(|(&v, &w)| (0.1 + w) * v / rms).collect::<Vec<f32>>()
            })
            .flatten()
            .collect();
        lines.push(ok("rmsnorm_off", &got, &want, 2e-4));
    }

    // argmax (the two-stage reduction and the read-back).
    {
        let mut logits = vec![0f32; 256];
        logits[35] = 3.0;
        logits[200] = 7.0;
        let got = b.debug_argmax_async(&logits, 1).await;
        lines.push(format!("argmax: {} ({got:?}, want [200])", if got == vec![200] { "PASS" } else { "FAIL" }));
    }

    // A Q8_0 matrix [8, 256]: block b's scale is (b % 7 + 1) / 8, values (b*13 + i*7) % 15 - 7.
    let (n, k) = (8usize, 256usize);
    let blocks = n * k / 32;
    let mut bytes = Vec::with_capacity(blocks * 34);
    let mut vals = vec![0f32; n * k];
    for blk in 0..blocks {
        let d = 0.125f32 * ((blk % 7) as f32 + 1.0);
        bytes.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        for i in 0..32 {
            let q = ((blk * 13 + i * 7) % 15) as i32 - 7;
            bytes.push(q as i8 as u8);
            vals[blk * 32 + i] = d * q as f32;
        }
    }
    let wt = b.load_weight(&TensorRef::from_owned(bytes.clone(), vec![n, k], GgmlType::Q8_0, 0.0), Quant::Auto)?;
    let ref_row = |x: &[f32], r: usize| -> Vec<f32> {
        (0..n).map(|m| (0..k).map(|j| x[j] * vals[m * k + j]).sum::<f32>()).collect()
    };
    // One input row for every GEMV test below.
    let x: Vec<f32> = (0..k).map(|j| ((j * 31) % 23) as f32 * 0.1 - 1.0).collect();

    // The f32 GEMV (a control: the same path the probe verified).
    {
        b.set_gemv(Gemv::F32);
        let want = ref_row(&x, 0);
        let got = b.debug_matmul_ggml_async(&x, 1, GgmlType::Q8_0, &bytes, n, k, false).await;
        lines.push(ok("gemv1_q8_f32", &got, &want, 2e-3));
    }

    // The int16 decode GEMV (the model's decode path).
    {
        b.set_gemv(Gemv::INT16);
        let want = ref_row(&x, 0);
        let got = b.debug_matmul_ggml_async(&x, 1, GgmlType::Q8_0, &bytes, n, k, false).await;
        lines.push(ok("gemv1_q8_int16", &got, &want, 5e-2));
    }

    // The int16 prefill GEMM.
    {
        b.set_gemm(GemmKind::Int16);
        b.set_gemv(Gemv::INT16);
        let t = 32usize;
        let x: Vec<f32> = (0..t * k).map(|i| ((i * 17 + i / k * 5) % 29) as f32 * 0.2 - 3.0).collect();
        let want: Vec<f32> = (0..t).flat_map(|r| ref_row(&x[r * k..(r + 1) * k], r)).collect();
        let got = b.debug_matmul_ggml_async(&x, t, GgmlType::Q8_0, &bytes, n, k, false).await;
        lines.push(ok("gemm_int16_32rows", &got, &want, 5e-2));
    }

    // embed: a table whose row r is all `0.25 r`, gathered for ids [1, 5, 3, 15].
    {
        let (v, h) = (16usize, 64usize);
        let table: Vec<f32> = (0..v * h).map(|i| (i / h) as f32 * 0.25).collect();
        let emb = b.load_embedding(&TensorRef::from_owned(bf16_bytes(&table), vec![v, h], GgmlType::Bf16, 0.0))?;
        let pass = PassMeta {
            ids: vec![1, 5, 3, 15],
            pos: vec![0, 1, 2, 3],
            kv_slot: vec![0, 1, 2, 3],
            pt_off: vec![0; 4],
            pages: vec![0],
        };
        b.begin_pass(&pass);
        let e = b.embed(&emb);
        let got = b.to_host_async(&e).await;
        let want: Vec<f32> = pass.ids.iter().flat_map(|&id| vec![0.25 * id as f32; h]).collect();
        lines.push(ok("embed", &got, &want, 1e-4));
    }

    // add_norm: x += post_norm(delta), h = pre_norm(x) (the sandwich residual step).
    // The wide variant (cols > ADD_NORM_SG_MAX_COLS) takes the plain kernel.
    for (name, cols) in [("add_norm_sg", 64usize), ("add_norm_plain", 4208usize)] {
        let rows = 4usize;
        let x: Vec<f32> = (0..rows * cols).map(|i| (i % 13) as f32 * 0.6 - 3.0).collect();
        let delta: Vec<f32> = (0..rows * cols).map(|i| (i % 11) as f32 * 0.4 - 2.0).collect();
        let w1: Vec<f32> = (0..cols).map(|i| 0.9 + (i % 5) as f32 * 0.02).collect();
        let w2: Vec<f32> = (0..cols).map(|i| 0.8 + (i % 7) as f32 * 0.03).collect();
        let mut xt = b.debug_upload(&x, rows, cols);
        let dt = b.debug_upload(&delta, rows, cols);
        let w1t = b.load_weight(&TensorRef::from_owned(bf16_bytes(&w1), vec![cols], GgmlType::Bf16, 0.0), Quant::Bf16)?;
        let w2t = b.load_weight(&TensorRef::from_owned(bf16_bytes(&w2), vec![cols], GgmlType::Bf16, 0.0), Quant::Bf16)?;
        let h = b.add_norm(&mut xt, &dt, &w1t, 0.1, &w2t, 0.2, 1e-6);
        let norm = |w: &[f32], o: f32, v: &[f32]| -> Vec<f32> {
            // (the GPU sees the bf16-rounded weights: the reference uses them too)
            v.chunks_exact(cols)
                .map(|row| {
                    let ms = row.iter().map(|x| x * x).sum::<f32>() / cols as f32;
                    let rms = (ms + 1e-6).sqrt();
                    row.iter().zip(w).map(|(&x, &w)| (o + w) * x / rms).collect::<Vec<f32>>()
                })
                .flatten()
                .collect()
        };
        let w1b: Vec<f32> = w1.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect();
        let w2b: Vec<f32> = w2.iter().map(|&v| half::bf16::from_f32(v).to_f32()).collect();
        let part = norm(&w1b, 0.1, &delta);
        let want_x: Vec<f32> = x.iter().zip(&part).map(|(a, b)| a + b).collect();
        let want_h = norm(&w2b, 0.2, &want_x);
        let got_h = b.to_host_async(&h).await;
        let got_x = b.to_host_async(&xt).await;
        lines.push(ok(&format!("{name}(x)"), &got_x, &want_x, 2e-4));
        lines.push(ok(&format!("{name}(h)"), &got_h, &want_h, 2e-4));
    }

    // The K-quant decode GEMVs the model actually uses (Q4_K matrices everywhere,
    // Q6_K for lm_head): structured pseudo-random blocks (no inf/nan scales),
    // referenced by the host's dequantize (the bit-exact decoder the CLI tests check).
    {
        b.set_gemv(Gemv::F32);
        let mut seed = 0x1234_5678u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for (name, t) in [("gemv1_q4k", GgmlType::Q4K), ("gemv1_q6k", GgmlType::Q6K)] {
            let (elems, block) = t.block();
            let nblocks = n * k / elems;
            let mut bytes: Vec<u8> = (0..nblocks * block).map(|i| (next() >> ((i % 8) * 8)) as u8).collect();
            for blk in 0..nblocks {
                let d = half::f16::from_f32(0.5).to_le_bytes();
                if t == GgmlType::Q4K {
                    bytes[blk * block..blk * block + 2].copy_from_slice(&d);
                    bytes[blk * block + 2..blk * block + 4].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());
                } else {
                    // Q6_K keeps `d` in the last 2 bytes of the block.
                    bytes[blk * block + block - 2..blk * block + block].copy_from_slice(&d);
                }
            }
            let wvals = yaxus_model::ggml::dequantize(t, &bytes, n * k);
            let want: Vec<f32> = (0..n).map(|m| (0..k).map(|j| x[j] * wvals[m * k + j]).sum::<f32>()).collect();
            let got = b.debug_matmul_ggml_async(&x, 1, t, &bytes, n, k, false).await;
            lines.push(ok(name, &got, &want, 2e-3));
        }
    }

    // attention: qk_rope (per-head norm + half-split RoPE) writes k/v to the paged KV
    // and the split kernel attends over the row's own keys. Variants: `full` (the real
    // math), `q0`/`k0` (scores identically 0: a uniform softmax of the row's keys,
    // whatever the norms and rope do), and `2seq` (two sequences, one key each).
    {
        use yaxus_model::{Config, LayerMeta, Rope, PAGE};
        let (hq, hkv, d) = (16usize, 1usize, 128usize);
        let theta = 10000.0f64;
        let rope = Rope {
            inv_freq: (0..d / 2).map(|j| (theta.powf(-(j as f64) / (d as f64 / 2.0))) as f32).collect(),
            concentration: 1.0,
        };
        let rope_freqs = rope.inv_freq.clone();
        let meta0 = LayerMeta { index: 0, full_attn: true, window: None, rope: rope.clone() };
        let cfg = Config {
            hidden_size: hq * d,
            num_hidden_layers: 1,
            num_attention_heads: hq,
            num_key_value_heads: hkv,
            head_dim: d,
            intermediate_size: 64,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            window_size: 4,
            sliding_window_pattern: 1,
            rope_theta: theta,
            rope_local_theta: theta,
            scale_embedding: false,
            tie_word_embeddings: false,
            rope_scaling_factor: 1.0,
            initial_context_length: 8,
            rope_parameters: None,
        };
        let (qd, kvd) = (hq * d, hkv * d);
        let ones = bf16_bytes(&vec![1.0; d]);
        let qn = b.load_weight(&TensorRef::from_owned(ones.clone(), vec![d], GgmlType::Bf16, 0.0), Quant::Bf16)?;
        let kn = b.load_weight(&TensorRef::from_owned(ones, vec![d], GgmlType::Bf16, 0.0), Quant::Bf16)?;
        let ones32 = vec![1.0f32; d];
        let norm1 = |val: &[f32], w: &[f32]| -> Vec<f32> {
            let ms = val.iter().map(|x| x * x).sum::<f32>() / d as f32;
            let r = 1.0 / (ms + 1e-6).sqrt();
            val.iter().zip(w).map(|(&v, &w)| (1.0 + w) * v * r).collect()
        };
        for (vname, kind) in [("full", 0u8), ("full2", 5), ("q0", 1), ("k0", 2), ("2seq", 3), ("full_id", 4)] {
            let two_seq = kind == 3;
            let full_math = kind == 0 || kind == 5;
            let rope = if kind == 4 {
                Rope { inv_freq: vec![0.0; d / 2], concentration: 1.0 }
            } else {
                Rope { inv_freq: rope_freqs.clone(), concentration: 1.0 }
            };
            let meta = LayerMeta { index: 0, full_attn: true, window: None, rope: rope.clone() };
            let cs = |pos: usize| -> ([f32; 64], [f32; 64]) {
                let mut c = [0f32; 64];
                let mut s = [0f32; 64];
                for (j, f) in rope.inv_freq.iter().enumerate() {
                    // (the host computes pos·f in f32 and the trig in f64, as config.rs does)
                    let a = pos as f32 * *f;
                    c[j] = (a as f64).cos() as f32;
                    s[j] = (a as f64).sin() as f32;
                }
                (c, s)
            };
            let half = d / 2;
            let ropev = |val: &[f32], pos: usize| -> Vec<f32> {
                let (c, s) = cs(pos);
                (0..d)
                    .map(|li| {
                        let j = li % half;
                        if li < half {
                            val[li] * c[j] - val[li + half] * s[j]
                        } else {
                            val[li] * c[j] + val[li - half] * s[j]
                        }
                    })
                    .collect()
            };
            let mut seed = 0x9e37_79b9u64;
            let mut next = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
            };
            let mut qkv = vec![0f32; 2 * (qd + 2 * kvd)];
            for v in qkv.iter_mut() {
                *v = next();
            }
            if kind == 1 {
                // q of both rows (q first, then row 1's q after k/v).
                for (i, v) in qkv.iter_mut().enumerate() {
                    if i < qd || (qd + 2 * kvd..qd + 2 * kvd + qd).contains(&i) {
                        *v = 0.0;
                    }
                }
            } else if kind == 2 {
                // k of both rows.
                for (i, v) in qkv.iter_mut().enumerate() {
                    if (qd..qd + kvd).contains(&i) || (qd + 2 * kvd + qd..qd + 2 * kvd + qd + kvd).contains(&i) {
                        *v = 0.0;
                    }
                }
            }
            let head = |r: usize, h: usize, off: usize| -> Vec<f32> {
                let base = r * (qd + 2 * kvd) + off;
                qkv[base..base + d].to_vec()
            };
            let pass = if two_seq {
                PassMeta {
                    ids: vec![0, 1],
                    pos: vec![0, 0],
                    kv_slot: vec![0, PAGE as u32],
                    pt_off: vec![0, 1],
                    pages: vec![0, 1],
                }
            } else {
                PassMeta { ids: vec![0, 1], pos: vec![0, 1], kv_slot: vec![0, 1], pt_off: vec![0, 0], pages: vec![0] }
            };
            let mut kv = b.new_kv(&cfg, &[meta.clone()], if two_seq { 2 } else { 1 }, 2)?;
            let qkvt = b.debug_upload(&qkv, 2, qd + 2 * kvd);
            b.begin_pass(&pass);
            let o = b.attention(&qkvt, &qn, &kn, &mut kv, &cfg, &meta);
            let got = b.to_host_async(&o).await;
            let scale = 1.0 / (d as f32).sqrt();
            let mut want = vec![0f32; 2 * qd];
            for r in 0..2 {
                for h in 0..hq {
                    let mut out = vec![0f32; d];
                    if full_math {
                        let q = ropev(&norm1(&head(r, h, 0), &ones32), r);
                        let nk = r + 1;
                        let mut scores = Vec::with_capacity(nk);
                        for j in 0..nk {
                            let k = ropev(&norm1(&head(j, 0, qd), &ones32), j);
                            scores.push(q.iter().zip(&k).map(|(a, b2)| a * b2).sum::<f32>() * scale);
                        }
                        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let exps: Vec<f32> = scores.iter().map(|&s2| (s2 - max).exp()).collect();
                        let sum: f32 = exps.iter().sum();
                        for j in 0..nk {
                            let v = head(j, 0, qd + kvd);
                            for li in 0..d {
                                out[li] += exps[j] / sum * v[li];
                            }
                        }
                    } else if kind == 3 {
                        out.copy_from_slice(&head(r, 0, qd + kvd));
                    } else {
                        for j in 0..=r {
                            let v = head(j, 0, qd + kvd);
                            for li in 0..d {
                                out[li] += v[li] / (r + 1) as f32;
                            }
                        }
                    }
                    want[r * qd + h * d..r * qd + (h + 1) * d].copy_from_slice(&out);
                }
            }
            lines.push(ok(&format!("attn_{vname}"), &got, &want, 2e-4));
            if vname == "full" {
                let at = |r: usize, h: usize| r * qd + h * d;
                lines.push(format!(
                    "  got row1 h0..3[0] = ({:.4}, {:.4}, {:.4}, {:.4}) want = ({:.4}, {:.4}, {:.4}, {:.4})",
                    got[at(1, 0)],
                    got[at(1, 1)],
                    got[at(1, 2)],
                    got[at(1, 3)],
                    want[at(1, 0)],
                    want[at(1, 1)],
                    want[at(1, 2)],
                    want[at(1, 3)],
                ));
            }
        }
    }

    // qk_rope alone: read back the k and v it writes to the paged KV (the attention
    // output never checks them directly — a single-key softmax hides k entirely).
    {
        use yaxus_model::{Config, LayerMeta, Rope};
        let (hq, hkv, d) = (16usize, 1usize, 128usize);
        let theta = 10000.0f64;
        let rope = Rope {
            inv_freq: (0..d / 2).map(|j| (theta.powf(-(j as f64) / (d as f64 / 2.0))) as f32).collect(),
            concentration: 1.0,
        };
        let meta = LayerMeta { index: 0, full_attn: true, window: None, rope: rope.clone() };
        let cfg = Config {
            hidden_size: hq * d,
            num_hidden_layers: 1,
            num_attention_heads: hq,
            num_key_value_heads: hkv,
            head_dim: d,
            intermediate_size: 64,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            window_size: 4,
            sliding_window_pattern: 1,
            rope_theta: theta,
            rope_local_theta: theta,
            scale_embedding: false,
            tie_word_embeddings: false,
            rope_scaling_factor: 1.0,
            initial_context_length: 8,
            rope_parameters: None,
        };
        let mut kv = b.new_kv(&cfg, &[meta.clone()], 1, 2)?;
        let mut seed = 0x9e37_79b9u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let (qd, kvd) = (hq * d, hkv * d);
        let mut qkv = vec![0f32; 2 * (qd + 2 * kvd)];
        for v in qkv.iter_mut() {
            *v = next();
        }
        let ones = bf16_bytes(&vec![1.0; d]);
        let qn = b.load_weight(&TensorRef::from_owned(ones.clone(), vec![d], GgmlType::Bf16, 0.0), Quant::Bf16)?;
        let kn = b.load_weight(&TensorRef::from_owned(ones, vec![d], GgmlType::Bf16, 0.0), Quant::Bf16)?;
        let pass = PassMeta { ids: vec![0, 1], pos: vec![0, 1], kv_slot: vec![0, 1], pt_off: vec![0, 0], pages: vec![0] };
        b.begin_pass(&pass);
        let got = b.debug_qk_rope_async(&qkv, &qn, &kn, &mut kv, &cfg, &meta).await;
        let cs = |pos: usize| -> ([f32; 64], [f32; 64]) {
            let mut c = [0f32; 64];
            let mut s = [0f32; 64];
            for (j, f) in rope.inv_freq.iter().enumerate() {
                let a = pos as f32 * *f;
                c[j] = (a as f64).cos() as f32;
                s[j] = (a as f64).sin() as f32;
            }
            (c, s)
        };
        let half = d / 2;
        let ropev = |val: &[f32], pos: usize| -> Vec<f32> {
            let (c, s) = cs(pos);
            (0..d)
                .map(|li| {
                    let j = li % half;
                    if li < half {
                        val[li] * c[j] - val[li + half] * s[j]
                    } else {
                        val[li] * c[j] + val[li - half] * s[j]
                    }
                })
                .collect()
        };
        let norm1 = |val: &[f32], w: &[f32]| -> Vec<f32> {
            let ms = val.iter().map(|x| x * x).sum::<f32>() / d as f32;
            let r = 1.0 / (ms + 1e-6).sqrt();
            val.iter().zip(w).map(|(&v, &w)| (1.0 + w) * v * r).collect()
        };
        let ones32 = vec![1.0f32; d];
        let mut want = Vec::with_capacity(2 * (qd + 2 * kvd));
        for r in 0..2 {
            for h in 0..hq {
                want.extend(ropev(&norm1(&qkv[r * (qd + 2 * kvd) + h * d..r * (qd + 2 * kvd) + (h + 1) * d], &ones32), r));
            }
        }
        for r in 0..2 {
            want.extend(ropev(&norm1(&qkv[r * (qd + 2 * kvd) + qd..r * (qd + 2 * kvd) + qd + kvd], &ones32), r));
        }
        for r in 0..2 {
            want.extend(&qkv[r * (qd + 2 * kvd) + qd + kvd..r * (qd + 2 * kvd) + qd + 2 * kvd]);
        }
        lines.push(ok("qk_rope", &got, &want, 2e-4));
    }

    // The split kernel's per-head (max, sum) partials, with the roped q from the host
    // and the KV the qk_rope test filled: where the scores themselves go wrong.
    {
        use yaxus_model::{Config, LayerMeta, Rope};
        let (hq, hkv, d) = (16usize, 1usize, 128usize);
        let theta = 10000.0f64;
        let rope = Rope {
            inv_freq: (0..d / 2).map(|j| (theta.powf(-(j as f64) / (d as f64 / 2.0))) as f32).collect(),
            concentration: 1.0,
        };
        let meta = LayerMeta { index: 0, full_attn: true, window: None, rope: rope.clone() };
        let cfg = Config {
            hidden_size: hq * d,
            num_hidden_layers: 1,
            num_attention_heads: hq,
            num_key_value_heads: hkv,
            head_dim: d,
            intermediate_size: 64,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            window_size: 4,
            sliding_window_pattern: 1,
            rope_theta: theta,
            rope_local_theta: theta,
            scale_embedding: false,
            tie_word_embeddings: false,
            rope_scaling_factor: 1.0,
            initial_context_length: 8,
            rope_parameters: None,
        };
        let mut kv = b.new_kv(&cfg, &[meta.clone()], 1, 2)?;
        let mut seed = 0x9e37_79b9u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let (qd, kvd) = (hq * d, hkv * d);
        let mut qkv = vec![0f32; 2 * (qd + 2 * kvd)];
        for v in qkv.iter_mut() {
            *v = next();
        }
        let ones = bf16_bytes(&vec![1.0; d]);
        let qn = b.load_weight(&TensorRef::from_owned(ones.clone(), vec![d], GgmlType::Bf16, 0.0), Quant::Bf16)?;
        let kn = b.load_weight(&TensorRef::from_owned(ones, vec![d], GgmlType::Bf16, 0.0), Quant::Bf16)?;
        let pass = PassMeta { ids: vec![0, 1], pos: vec![0, 1], kv_slot: vec![0, 1], pt_off: vec![0, 0], pages: vec![0] };
        b.begin_pass(&pass);
        b.debug_qk_rope_async(&qkv, &qn, &kn, &mut kv, &cfg, &meta).await;
        // The roped q, as the host computes it (verified by the qk_rope test).
        let cs = |pos: usize| -> ([f32; 64], [f32; 64]) {
            let mut c = [0f32; 64];
            let mut s = [0f32; 64];
            for (j, f) in rope.inv_freq.iter().enumerate() {
                let a = pos as f32 * *f;
                c[j] = (a as f64).cos() as f32;
                s[j] = (a as f64).sin() as f32;
            }
            (c, s)
        };
        let half = d / 2;
        let ropev = |val: &[f32], pos: usize| -> Vec<f32> {
            let (c, s) = cs(pos);
            (0..d)
                .map(|li| {
                    let j = li % half;
                    if li < half {
                        val[li] * c[j] - val[li + half] * s[j]
                    } else {
                        val[li] * c[j] + val[li - half] * s[j]
                    }
                })
                .collect()
        };
        let ones32 = vec![1.0f32; d];
        let norm1 = |val: &[f32], w: &[f32]| -> Vec<f32> {
            let ms = val.iter().map(|x| x * x).sum::<f32>() / d as f32;
            let r = 1.0 / (ms + 1e-6).sqrt();
            val.iter().zip(w).map(|(&v, &w)| (1.0 + w) * v * r).collect()
        };
        let mut q = Vec::with_capacity(2 * qd);
        for r in 0..2 {
            for h in 0..hq {
                q.extend(ropev(&norm1(&qkv[r * (qd + 2 * kvd) + h * d..r * (qd + 2 * kvd) + (h + 1) * d], &ones32), r));
            }
        }
        let ml = b.debug_attn_split_async(&q, 2, hq, hkv, &kv, 0).await;
        // The reference (max, sum) per row and head (one chunk holds both keys).
        let scale = 1.0 / (d as f32).sqrt();
        let mut want = Vec::with_capacity(ml.len());
        for r in 0..2 {
            for h in 0..hq {
                let qr = &q[r * qd + h * d..r * qd + (h + 1) * d];
                let mut scores = Vec::new();
                for j in 0..=r {
                    let k = ropev(&norm1(&qkv[j * (qd + 2 * kvd) + qd..j * (qd + 2 * kvd) + qd + kvd], &ones32), j);
                    scores.push(qr.iter().zip(&k).map(|(a, b2)| a * b2).sum::<f32>() * scale);
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let l: f32 = scores.iter().map(|&s2| (s2 - max).exp()).sum();
                want.push(max);
                want.push(l);
            }
        }
        let nch = ml.len() / (2 * hq * 2) / hkv; // (row, head, chunk) pairs of 2
        if nch == 1 {
            lines.push(ok("attn_split_ml", &ml, &want, 2e-4));
        } else {
            lines.push(format!("attn_split_ml: nch = {nch}, skipped (the reference expects one chunk)"));
        }

        // The same split, but with the q the qk_rope kernel wrote in the same
        // submission (no host round trip in between): a difference here means the
        // two dispatches do not see the same q.
        let (ml2, part2, o2, q2) = b.debug_attn_chain_async(&qkv, &qn, &kn, &mut kv, &cfg, &meta).await;
        if nch == 1 {
            lines.push(ok("attn_chain_ml", &ml2, &want, 2e-4));
        }
        // The roped q read back after the split (the split only reads it).
        let mut qwant = Vec::with_capacity(2 * qd);
        for r in 0..2 {
            for h in 0..hq {
                qwant.extend(ropev(&norm1(&qkv[r * (qd + 2 * kvd) + h * d..r * (qd + 2 * kvd) + (h + 1) * d], &ones32), r));
            }
        }
        lines.push(ok("attn_chain_q", &q2, &qwant, 2e-4));
        // The combine's output in the same chain (the `attn_full` path's o).
        let mut owant = vec![0f32; 2 * qd];
        let scale = 1.0 / (d as f32).sqrt();
        for r in 0..2 {
            for h in 0..hq {
                let qh = ropev(&norm1(&qkv[r * (qd + 2 * kvd) + h * d..r * (qd + 2 * kvd) + (h + 1) * d], &ones32), r);
                let mut out = vec![0f32; d];
                let nk = r + 1;
                let mut scores = Vec::with_capacity(nk);
                for j in 0..nk {
                    let k = ropev(&norm1(&qkv[j * (qd + 2 * kvd) + qd..j * (qd + 2 * kvd) + qd + kvd], &ones32), j);
                    scores.push(qh.iter().zip(&k).map(|(a, b2)| a * b2).sum::<f32>() * scale);
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|&s2| (s2 - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for j in 0..nk {
                    let v = &qkv[j * (qd + 2 * kvd) + qd + kvd..j * (qd + 2 * kvd) + qd + 2 * kvd];
                    for li in 0..d {
                        out[li] += exps[j] / sum * v[li];
                    }
                }
                owant[r * qd + h * d..r * qd + (h + 1) * d].copy_from_slice(&out);
            }
        }
        lines.push(ok("attn_chain_o", &o2, &owant, 2e-4));
    }

    // attn_tile: the prefill kernel (passes of more than 16 rows): 20 rows like the
    // model's prompt pass, each row attending to its own keys 0..=pos.
    {
        use yaxus_model::{Config, LayerMeta, Rope};
        let (hq, hkv, d) = (16usize, 1usize, 128usize);
        let t = 20usize;
        let theta = 10000.0f64;
        let rope = Rope {
            inv_freq: (0..d / 2).map(|j| (theta.powf(-(j as f64) / (d as f64 / 2.0))) as f32).collect(),
            concentration: 1.0,
        };
        let meta = LayerMeta { index: 0, full_attn: true, window: None, rope: rope.clone() };
        let cfg = Config {
            hidden_size: hq * d,
            num_hidden_layers: 1,
            num_attention_heads: hq,
            num_key_value_heads: hkv,
            head_dim: d,
            intermediate_size: 64,
            vocab_size: 16,
            rms_norm_eps: 1e-6,
            window_size: 4,
            sliding_window_pattern: 1,
            rope_theta: theta,
            rope_local_theta: theta,
            scale_embedding: false,
            tie_word_embeddings: false,
            rope_scaling_factor: 1.0,
            initial_context_length: 8,
            rope_parameters: None,
        };
        let mut kv = b.new_kv(&cfg, &[meta.clone()], 1, t)?;
        let mut seed = 0x9e37_79b9u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        };
        let (qd, kvd) = (hq * d, hkv * d);
        let mut qkv = vec![0f32; t * (qd + 2 * kvd)];
        for v in qkv.iter_mut() {
            *v = next();
        }
        let ones = bf16_bytes(&vec![1.0; d]);
        let qn = b.load_weight(&TensorRef::from_owned(ones.clone(), vec![d], GgmlType::Bf16, 0.0), Quant::Bf16)?;
        let kn = b.load_weight(&TensorRef::from_owned(ones, vec![d], GgmlType::Bf16, 0.0), Quant::Bf16)?;
        let pass = PassMeta {
            ids: (0..t as u32).collect(),
            pos: (0..t as u32).collect(),
            kv_slot: (0..t as u32).collect(),
            pt_off: vec![0; t],
            pages: vec![0],
        };
        b.begin_pass(&pass);
        let qkvt = b.debug_upload(&qkv, t, qd + 2 * kvd);
        let o = b.attention(&qkvt, &qn, &kn, &mut kv, &cfg, &meta);
        let got = b.to_host_async(&o).await;
        let cs = |pos: usize| -> ([f32; 64], [f32; 64]) {
            let mut c = [0f32; 64];
            let mut s = [0f32; 64];
            for (j, f) in rope.inv_freq.iter().enumerate() {
                let a = pos as f32 * *f;
                c[j] = (a as f64).cos() as f32;
                s[j] = (a as f64).sin() as f32;
            }
            (c, s)
        };
        let half = d / 2;
        let ropev = |val: &[f32], pos: usize| -> Vec<f32> {
            let (c, s) = cs(pos);
            (0..d)
                .map(|li| {
                    let j = li % half;
                    if li < half {
                        val[li] * c[j] - val[li + half] * s[j]
                    } else {
                        val[li] * c[j] + val[li - half] * s[j]
                    }
                })
                .collect()
        };
        let ones32 = vec![1.0f32; d];
        let norm1 = |val: &[f32], w: &[f32]| -> Vec<f32> {
            let ms = val.iter().map(|x| x * x).sum::<f32>() / d as f32;
            let r = 1.0 / (ms + 1e-6).sqrt();
            val.iter().zip(w).map(|(&v, &w)| (1.0 + w) * v * r).collect()
        };
        let mut want = vec![0f32; t * qd];
        let scale = 1.0 / (d as f32).sqrt();
        for r in 0..t {
            for h in 0..hq {
                let q = ropev(&norm1(&qkv[r * (qd + 2 * kvd) + h * d..r * (qd + 2 * kvd) + (h + 1) * d], &ones32), r);
                let mut out = vec![0f32; d];
                let mut scores = Vec::with_capacity(r + 1);
                for j in 0..=r {
                    let k = ropev(&norm1(&qkv[j * (qd + 2 * kvd) + qd..j * (qd + 2 * kvd) + qd + kvd], &ones32), j);
                    scores.push(q.iter().zip(&k).map(|(a, b2)| a * b2).sum::<f32>() * scale);
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|&s2| (s2 - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for j in 0..=r {
                    let v = &qkv[j * (qd + 2 * kvd) + qd + kvd..j * (qd + 2 * kvd) + qd + 2 * kvd];
                    for li in 0..d {
                        out[li] += exps[j] / sum * v[li];
                    }
                }
                want[r * qd + h * d..r * qd + (h + 1) * d].copy_from_slice(&out);
            }
        }
        lines.push(ok("attn_tile_20rows", &got, &want, 2e-4));
    }

    Ok(lines.join("\n"))
}
