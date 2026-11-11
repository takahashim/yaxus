//! Loading a model for inference: the backend, the arithmetic of the GPU passes, the
//! optional draft model, and the tokenizer (from `tokenizer.jsonl` or a GGUF file).

use std::path::{Path, PathBuf};

use anyhow::Result;
use yaxus_backend_cpu::CpuBackend;
use yaxus_backend_wgpu::WgpuBackend;
use yaxus_model::{Engine, Generator, Model, Quant};
use yaxus_tokenizer::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Vulkan when a GPU is available, otherwise CPU.
    Auto,
    Vulkan,
    Cpu,
}

impl std::str::FromStr for BackendKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(BackendKind::Auto),
            "vulkan" | "gpu" => Ok(BackendKind::Vulkan),
            "cpu" => Ok(BackendKind::Cpu),
            _ => anyhow::bail!("unknown backend {s:?} (expected auto, vulkan or cpu)"),
        }
    }
}

/// A small model of the same family used for speculative decoding.
#[derive(Debug, Clone)]
pub struct DraftOptions {
    pub dir: PathBuf,
    /// Maximum tokens drafted per verification pass.
    pub k: usize,
    /// Weight format of the draft model (it only affects speed, never the output).
    pub quant: Quant,
}

/// Arithmetic of the GPU prompt processing: passes of more than 16 rows, or with
/// [`Decode::Int16`] more than 48 (`Int16`, `F16`) or 24 (`Int8`), below which the decode
/// kernels are faster. The figures are for a 780M against `F32`: speed of the pass and how far
/// the logits of the first generated token move.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Prefill {
    /// Integer dot products, activations rounded to 15 bits per block of 32: 2.6x, 6e-3.
    #[default]
    Int16,
    /// The same with 7 bits, as llama.cpp does: 3.9x, 4e-1 to 6e-1.
    Int8,
    /// Matrix cores with f16 operands: 2.0x, 2e-2 to 3e-2.
    F16,
    /// f32 throughout: the reference.
    F32,
}

impl std::str::FromStr for Prefill {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "int16" => Ok(Prefill::Int16),
            "int8" => Ok(Prefill::Int8),
            "f16" => Ok(Prefill::F16),
            "f32" => Ok(Prefill::F32),
            _ => anyhow::bail!("unknown prefill precision {s:?} (expected int16, int8, f16 or f32)"),
        }
    }
}

/// Arithmetic of GPU passes of up to 16 rows (decode steps, speculative verification,
/// short prompts; see [`Prefill`] for when longer passes take the same kernels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Decode {
    /// Integer dot products, activations rounded to 15 bits per block of 32: logits move
    /// by about 6e-3 against `F32`; a 7-row pass is 1.4x (2B) to 1.6x (8B) faster on a
    /// 780M, single rows cost about 1% more. A row's logits do not depend on the other
    /// rows of its pass.
    #[default]
    Int16,
    /// f32 throughout: the reference.
    F32,
}

impl Decode {
    /// Requests one decode pass advances by default: 16 with `Int16`, where extra rows are
    /// cheap, 8 with `F32`, where 8 rows is the cheapest per row.
    pub fn default_rows(self) -> usize {
        match self {
            Decode::Int16 => 16,
            Decode::F32 => 8,
        }
    }
}

impl std::str::FromStr for Decode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "int16" => Ok(Decode::Int16),
            "f32" => Ok(Decode::F32),
            _ => anyhow::bail!("unknown decode precision {s:?} (expected int16 or f32)"),
        }
    }
}

/// How to load and run the model.
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub backend: BackendKind,
    pub quant: Quant,
    pub draft: Option<DraftOptions>,
    /// Arithmetic of the GPU prompt processing.
    pub prefill: Prefill,
    pub decode: Decode,
    /// Most requests one decode pass advances; `None`: [`Decode::default_rows`].
    pub decode_rows: Option<usize>,
    /// Reuse the KV of finished requests for prompts sharing a prefix.
    pub prefix_cache: bool,
    /// KV arena size in tokens (per model).
    pub kv_tokens: usize,
    /// Scheduling. Its `draft_k` is taken from `draft` and its `decode_rows` from
    /// `decode_rows`.
    pub engine: yaxus_model::EngineConfig,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            backend: BackendKind::Auto,
            quant: Quant::Auto,
            draft: None,
            prefill: Prefill::default(),
            decode: Decode::default(),
            decode_rows: None,
            prefix_cache: true,
            kv_tokens: yaxus_model::DEFAULT_KV_TOKENS,
            engine: yaxus_model::EngineConfig::default(),
        }
    }
}

fn prepare<B: yaxus_model::Backend>(mut m: Model<B>, o: &LoadOptions) -> Result<Model<B>> {
    m.prefix_cache = o.prefix_cache;
    m.init_kv(o.kv_tokens, yaxus_model::DEFAULT_MAX_POS)?;
    Ok(m)
}

/// How the weights of `model` are stored, for display: the quantization, or for a GGUF
/// file loaded with `auto` its main tensor type (`gguf Q4_K`).
pub fn quant_label(model: &Path, quant: Quant) -> String {
    if matches!(quant, Quant::Auto | Quant::Auto4) {
        if yaxus_model::gguf::is_gguf(model) {
            let main = yaxus_model::gguf::Gguf::open(model).ok().and_then(|g| g.main_type());
            return format!("gguf {}", main.map_or("?", |t| t.name()));
        }
        return if quant == Quant::Auto { "q8" } else { "q4" }.into();
    }
    format!("{quant:?}").to_lowercase()
}

/// The tokenizer of a model: `tokenizer.jsonl` of a Hugging Face directory, or the
/// vocabulary embedded in a GGUF file.
pub fn load_tokenizer(model: &Path) -> Result<Tokenizer> {
    use yaxus_tokenizer::TokenKind;
    if !yaxus_model::gguf::is_gguf(model) {
        return Ok(Tokenizer::from_file(model.join("tokenizer.jsonl"))?);
    }
    let v = yaxus_model::gguf::Gguf::open(model)?.vocab()?;
    let kinds = v
        .types
        .iter()
        .map(|t| match t {
            1 => TokenKind::Normal,
            2 => TokenKind::Unknown,
            3 => TokenKind::Control,
            6 => TokenKind::Byte,
            _ => TokenKind::Other,
        })
        .collect();
    Ok(Tokenizer::from_vocab(v.tokens, kinds, v.scores)?)
}

/// Loads the model (and optional draft model) on the requested backend.
pub fn load_generator(dir: &Path, o: &LoadOptions) -> Result<Box<dyn Generator>> {
    let gpu = match o.backend {
        BackendKind::Cpu => None,
        BackendKind::Vulkan => Some(WgpuBackend::new()?),
        BackendKind::Auto => match WgpuBackend::new() {
            Ok(g) => Some(g),
            Err(e) => {
                tracing::warn!("Vulkan unavailable, using the CPU: {e:#}");
                None
            }
        },
    };
    let draft = if gpu.is_none() && o.draft.is_some() {
        // The CPU matmul gains nothing from verifying several rows at once, so drafting
        // only adds work (measured 18.5 -> 14.6 tok/s on the 2B).
        tracing::info!("speculative decoding is disabled on the CPU backend (it is slower there)");
        None
    } else {
        o.draft.as_ref()
    };
    let mut cfg = o.engine.clone();
    cfg.decode_rows = o.decode_rows.unwrap_or(o.decode.default_rows());
    if let Some(d) = draft {
        cfg.draft_k = d.k;
    }
    let Some(mut gpu) = gpu else {
        yaxus_backend_cpu::init_threads();
        let target = prepare(Model::load(dir, CpuBackend::new(), o.quant)?, o)?;
        let draft = match draft {
            Some(d) => Some(prepare(Model::load(&d.dir, CpuBackend::new(), d.quant)?, o)?),
            None => None,
        };
        return Ok(Box::new(Engine::new(target, draft, cfg)?));
    };
    // One device for the target and the draft model.
    use yaxus_backend_wgpu::GemmKind;
    gpu.set_gemm(match o.prefill {
        Prefill::Int16 => GemmKind::Int16,
        Prefill::Int8 => GemmKind::Int8,
        Prefill::F16 => GemmKind::Coop,
        Prefill::F32 => GemmKind::F32,
    });
    use yaxus_backend_wgpu::Gemv;
    let mut draft_gpu = gpu.share();
    match o.decode {
        Decode::Int16 => {
            gpu.set_gemv(Gemv::INT16);
            // The draft only proposes tokens: its single rows take the f32 kernel, a
            // little faster for the Q2_K draft (19-sentence set: 74.4 vs 74.0 tok/s).
            draft_gpu.set_gemv(Gemv { planes: Some(2), single_f32: true });
        }
        Decode::F32 => {
            gpu.set_gemv(Gemv::F32);
            draft_gpu.set_gemv(Gemv::F32);
        }
    }
    let target = prepare(Model::load(dir, gpu, o.quant)?, o)?;
    target.backend.precompile();
    let draft = match draft {
        Some(d) => {
            let m = prepare(Model::load(&d.dir, draft_gpu, d.quant)?, o)?;
            m.backend.precompile();
            Some(m)
        }
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
        let snapshot = |repo: &str| crate::resolve::hf_snapshot(repo).ok();
        let (Some(hf), Some(gg)) =
            (snapshot("pfnet/plamo-3-translate-0.8b"), snapshot("pfnet/plamo-3-translate-0.8b-GGUF"))
        else {
            let required = std::env::var_os("YAXUS_REQUIRE_GOLDEN").is_some_and(|v| !v.is_empty() && v != "0");
            assert!(!required, "YAXUS_REQUIRE_GOLDEN is set but the 0.8B model or GGUF files are not in the cache");
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
