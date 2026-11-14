//! Loading a model for inference: the engine on the chosen backend with the optional
//! draft model ([`LoadOptions`]), and the tokenizer (from `tokenizer.jsonl` or a GGUF file).

use std::path::Path;

use anyhow::Result;
use yaxus_backend_cpu::CpuBackend;
use yaxus_backend_gpu::{Gemv, GpuBackend};
use yaxus_model::{Engine, Generator, Model, ModelOptions, Quant, SyncReadback};
use yaxus_tokenizer::Tokenizer;

use crate::options::{BackendKind, DraftOptions, LoadOptions};

/// The tokenizer of a model: `tokenizer.jsonl` of a Hugging Face directory, or the
/// vocabulary embedded in a GGUF file.
pub fn load_tokenizer(model: &Path) -> Result<Tokenizer> {
    use yaxus_tokenizer::TokenKind;
    if !yaxus_model::gguf::is_gguf(model) {
        return Ok(Tokenizer::from_file(model.join("tokenizer.jsonl"))?);
    }
    let v = yaxus_model::gguf::Gguf::open(model)?.vocab()?;
    let kinds = v.types.iter().map(|&t| TokenKind::from_ggml_type(t)).collect();
    Ok(Tokenizer::from_vocab(v.tokens, kinds, v.scores)?)
}

/// Loads the model (and optional draft model) on the requested backend.
pub fn load_generator(dir: &Path, o: &LoadOptions) -> Result<Box<dyn Generator + Send>> {
    match gpu(o.backend)? {
        Some(mut gpu) => {
            // One device for the target and the draft model, with their own kernels.
            // (the device's kernels unless the options choose)
            let tuning = gpu.tuning();
            gpu.set_gemm(o.prefill.gemm_kind().unwrap_or(tuning.gemm));
            let gemv = o.decode.gemv(tuning.gemv);
            gpu.set_gemv(gemv);
            let draft_gpu = gpu.share_for_draft();
            let draft = o.draft.as_ref().map(|d| (d, draft_gpu));
            load_engine(dir, o, gpu, draft, gemv, |b: &GpuBackend| b.precompile())
        }
        None => {
            if o.draft.is_some() {
                // The CPU matmul gains nothing from verifying several rows at once, so
                // drafting only adds work (measured 18.5 -> 14.6 tok/s on the 2B).
                tracing::warn!("speculative decoding is disabled on the CPU backend (it is slower there)");
            }
            yaxus_backend_cpu::init_threads();
            // (`gemv` only sets the default decode rows: 16 as with the integer kernels)
            load_engine(dir, o, CpuBackend::new(), None, Gemv::INT16, |_: &CpuBackend| {})
        }
    }
}

/// The GPU backend `kind` asks for; `None` for the CPU.
fn gpu(kind: BackendKind) -> Result<Option<GpuBackend>> {
    Ok(match kind {
        BackendKind::Cpu => None,
        BackendKind::Gpu => Some(GpuBackend::new()?),
        BackendKind::Auto => match GpuBackend::new() {
            Ok(g) => Some(g),
            Err(e) => {
                tracing::warn!("no GPU available, using the CPU: {e:#}");
                None
            }
        },
    })
}

/// The target model at `dir` on `target`, and the draft model on its own backend, in
/// one engine. `warm` runs on each backend once its model is loaded.
fn load_engine<B>(
    dir: &Path,
    o: &LoadOptions,
    target: B,
    draft: Option<(&DraftOptions, B)>,
    gemv: Gemv,
    warm: impl Fn(&B),
) -> Result<Box<dyn Generator + Send>>
where
    B: SyncReadback + Send + 'static,
    B::Weight: Send,
    B::Kv: Send,
{
    let load = |dir: &Path, backend: B, quant: Quant| -> Result<Model<B>> {
        let opts = ModelOptions { kv_tokens: o.kv_tokens, prefix_cache: o.prefix_cache, ..ModelOptions::new(quant) };
        let m = Model::load(dir, backend, &opts)?;
        warm(&m.backend);
        Ok(m)
    };
    let cfg = o.engine_config(draft.as_ref().map(|(d, _)| *d), gemv);
    let target = load(dir, target, o.quant)?;
    let draft = match draft {
        Some((d, backend)) => Some(load(&d.dir, backend, d.quant)?),
        None => None,
    };
    Ok(Box::new(Engine::new(target, draft, cfg)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The vocabulary of a GGUF file gives the same tokenizer as `tokenizer.jsonl`.
    /// Skipped without the models in the Hugging Face cache, unless `YAXUS_REQUIRE_GOLDEN`
    /// is set (as in CI), where that is a failure.
    #[test]
    #[ignore]
    fn gguf_tokenizer_matches_jsonl() {
        let snapshot = |repo: &str| yaxus_model::hub::snapshot(repo).ok();
        let (Some(hf), Some(gg)) =
            (snapshot("pfnet/plamo-3-translate-0.8b"), snapshot("pfnet/plamo-3-translate-0.8b-GGUF"))
        else {
            assert!(
                !yaxus_model::hub::models_required(),
                "YAXUS_REQUIRE_GOLDEN is set but the 0.8B model or GGUF files are not in the cache"
            );
            eprintln!("0.8B model or GGUF files not found; skipping");
            return;
        };
        let file = std::fs::read_dir(&gg)
            .unwrap()
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "gguf"))
            .unwrap();
        let a = load_tokenizer(&hf).unwrap();
        let b = load_tokenizer(&file).unwrap();
        let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/tokenizer.jsonl");
        let mut n = 0;
        for line in std::fs::read_to_string(golden).unwrap().lines() {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            if let Some(text) = v.get("text").and_then(|t| t.as_str()) {
                assert_eq!(a.encode(text, false), b.encode(text, false), "{text:?}");
                n += 1;
            }
        }
        assert!(n > 1000);
    }
}
