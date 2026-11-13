#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use yaxus_backend_gpu::GpuBackend;
use yaxus_model::{Model, ModelOptions, Quant, gguf::Gguf};
use yaxus_pfml::TranslationRequest;
use yaxus_tokenizer::{EOS_ID, StreamDecoder, TAG_ID, TokenKind, Tokenizer};

fn js_error(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn js_error_chain(e: anyhow::Error) -> JsValue {
    JsValue::from_str(&format!("{e:#}"))
}

#[wasm_bindgen]
pub struct Translator {
    model: Model<GpuBackend>,
    tokenizer: Tokenizer,
}

#[wasm_bindgen]
pub async fn load_model(file: web_sys::File) -> Result<Translator, JsValue> {
    console_error_panic_hook::set_once();
    let bytes = read_model_file(&file).await?;
    let gguf = Gguf::from_bytes(bytes).map_err(js_error_chain)?;
    let vocab = gguf.vocab().map_err(js_error_chain)?;
    let kinds = vocab.types.iter().map(|&t| TokenKind::from_ggml_type(t)).collect();
    let tokenizer = Tokenizer::from_vocab(vocab.tokens, kinds, vocab.scores).map_err(js_error)?;
    let backend = GpuBackend::new_web().await.map_err(js_error_chain)?;
    let opts = ModelOptions { kv_tokens: 1024, max_pos: 1024, prefix_cache: false, ..ModelOptions::new(Quant::Auto) };
    let scope = backend.web_error_scope();
    let model = Model::from_gguf(&gguf, backend, &opts);
    scope.finish().await.map_err(js_error_chain)?;
    let model = model.map_err(js_error_chain)?;
    Ok(Translator { model, tokenizer })
}

/// Read bounded browser buffers into the WASM allocation, avoiding a second whole-file copy.
async fn read_model_file(file: &web_sys::File) -> Result<Vec<u8>, JsValue> {
    const CHUNK: usize = 8 * 1024 * 1024;
    let size = file.size();
    if size > isize::MAX as f64 {
        return Err(JsValue::from_str("モデルファイルが大きすぎます"));
    }
    let size = size as usize;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(size).map_err(js_error)?;
    bytes.resize(size, 0);
    for start in (0..size).step_by(CHUNK) {
        let end = (start + CHUNK).min(size);
        let blob = file.slice_with_f64_and_f64(start as f64, end as f64)?;
        let buffer = JsFuture::from(blob.array_buffer()).await?;
        let chunk = js_sys::Uint8Array::new(&buffer);
        if chunk.length() as usize != end - start {
            return Err(JsValue::from_str("モデルファイルの読み込みが途中で終了しました"));
        }
        chunk.copy_to(&mut bytes[start..end]);
    }
    Ok(bytes)
}

#[wasm_bindgen]
impl Translator {
    /// Returns true when the output reached the token limit without a stop token.
    pub async fn translate(
        &mut self,
        text: String,
        target: String,
        on_text: js_sys::Function,
    ) -> Result<bool, JsValue> {
        let scope = self.model.backend.web_error_scope();
        let result = self.translate_inner(text, target, on_text).await;
        scope.finish().await.map_err(js_error_chain)?;
        result
    }
}

impl Translator {
    async fn translate_inner(
        &mut self,
        text: String,
        target: String,
        on_text: js_sys::Function,
    ) -> Result<bool, JsValue> {
        let request = TranslationRequest { text, target_lang: target, ..Default::default() };
        let prompt = request.validate().map_err(js_error)?.render();
        let ids = self.tokenizer.encode(&prompt, false);
        if ids.len() >= 768 {
            return Err(JsValue::from_str("入力が長すぎます"));
        }
        let (seq, mut logits) = self.model.start(&ids).map_err(js_error_chain)?;
        let result = async {
            let mut decoder = StreamDecoder::new();
            let mut truncated = true;
            for i in 0..256 {
                let token = self.model.backend.argmax_web(&logits).await.map_err(js_error_chain)?;
                if token == EOS_ID || token == TAG_ID {
                    truncated = false;
                    break;
                }
                let text = decoder.push(&self.tokenizer, token);
                if !text.is_empty() {
                    on_text.call1(&JsValue::NULL, &JsValue::from_str(&text))?;
                }
                if i < 255 {
                    logits = self.model.forward(&[(seq, &[token])], false).map_err(js_error_chain)?;
                }
            }
            let tail = decoder.finish();
            if !tail.is_empty() {
                on_text.call1(&JsValue::NULL, &JsValue::from_str(&tail))?;
            }
            Ok(truncated)
        }
        .await;
        self.model.free_seq(seq);
        result
    }
}
