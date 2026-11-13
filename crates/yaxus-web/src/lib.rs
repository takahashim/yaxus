//! The web frontend of yaxus: PLaMo 3 translation in the browser, on WebGPU.
//!
//! The page hands the model's `File` over here; the GGUF header is parsed from a
//! prefix of it (`GgufHeader`), and from the next step on, the tensors are read one
//! at a time, uploaded to the GPU, and dropped — the model never sits in wasm
//! memory whole.

use wasm_bindgen::prelude::*;

use yaxus_model::gguf::GgufHeader;

/// Prefixes tried until the header parses (the vocabulary metadata can make it
/// several megabytes).
const PREFIX_SIZES: [usize; 5] = [1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20];

/// Reads the first `n` bytes of `file` into wasm memory.
async fn read_prefix(file: &web_sys::File, n: usize) -> anyhow::Result<Vec<u8>> {
    let slice = file
        .slice_with_f64_and_f64(0.0, n as f64)
        .map_err(|e| anyhow::anyhow!("slicing the file: {e:?}"))?;
    let buf = wasm_bindgen_futures::JsFuture::from(slice.array_buffer())
        .await
        .map_err(|e| anyhow::anyhow!("reading the file: {e:?}"))?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

/// Reads and parses the GGUF header: the prefix grows until it holds it.
async fn read_header(file: &web_sys::File) -> anyhow::Result<GgufHeader> {
    let size = file.size();
    let mut last = None;
    for &n in &PREFIX_SIZES {
        if n as f64 > size + 1.0 && last.is_some() {
            break;
        }
        web_sys::console::log_1(&format!("read_header: trying prefix of {n}").into());
        let buf = read_prefix(file, n).await?;
        web_sys::console::log_1(&format!("read_header: read {} bytes", buf.len()).into());
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
    web_sys::console::log_1(&"summarize: start".into());
    let header = read_header(file).await?;
    web_sys::console::log_1(&format!("summarize: header ok, {} tensors", header.tensors.len()).into());
    let name = header
        .meta
        .get("general.name")
        .and_then(yaxus_model::gguf::Value::as_str)
        .unwrap_or("?")
        .to_string();
    let arch =
        header.meta.get("general.architecture").and_then(yaxus_model::gguf::Value::as_str).unwrap_or("?").to_string();
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
        Some(yaxus_model::gguf::Value::Array(t)) => t.len(),
        _ => cfg.vocab_size,
    };
    // The largest tensors, the ones that will each take one streaming read.
    let mut big: Vec<&yaxus_model::gguf::TensorInfo> = header.tensors.iter().collect();
    big.sort_by_key(|t| std::cmp::Reverse(t.len));
    big.truncate(8);
    let tensors_preview = big
        .iter()
        .map(|t| format!("{}  {:?}  {:?}  {:.1} MB", t.name, t.shape, t.dtype, t.len as f64 / 1e6))
        .collect::<Vec<_>>()
        .join("\n");
    let s = GgufSummary {
        name,
        arch,
        quant,
        tensors: header.tensors.len(),
        file_bytes: file.size() as u64,
        tensor_bytes,
        vocab_size,
        config,
        tensors_preview,
    };
    web_sys::console::log_1(&"summarize: done".into());
    Ok(s)
}

/// A panic in a wasm call must not abort the page: report it like an error.
pub fn hook_panics() {
    std::panic::set_hook(Box::new(|info| web_sys::console::error_1(&format!("yaxus-web panic: {info}").into())));
}
