//! How to load and run a model: the backend, the arithmetic of the GPU passes, the
//! optional draft model and the engine's scheduling, with their command-line spellings.

use std::path::PathBuf;

use anyhow::Result;
use yaxus_backend_gpu::{GemmKind, Gemv};
use yaxus_model::{EngineConfig, Quant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// The GPU when one is available, otherwise the CPU.
    Auto,
    /// The platform's GPU: Vulkan, or Metal on macOS.
    Gpu,
    Cpu,
}

impl std::str::FromStr for BackendKind {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(BackendKind::Auto),
            "gpu" => Ok(BackendKind::Gpu),
            "cpu" => Ok(BackendKind::Cpu),
            _ => anyhow::bail!("unknown backend {s:?} (expected auto, gpu or cpu)"),
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
    /// The device's choice ([`yaxus_backend_gpu::Tuning::gemm`]; `Int16` on the AMD GPUs
    /// measured so far; on Apple GPUs their 8x8 matrices in f32, nothing rounded).
    #[default]
    Auto,
    /// Integer dot products, activations rounded to 15 bits per block of 32: 2.6x, 6e-3.
    Int16,
    /// The same with 7 bits, as llama.cpp does: 3.9x, 4e-1 to 6e-1.
    Int8,
    /// Matrix cores with f16 operands: 2.0x, 2e-2 to 3e-2. On Metal the weights and the
    /// activations are rounded to half as llama.cpp does: on an M2 a 256-row pass takes
    /// 492 ms against 539 with the device's choice (f32).
    F16,
    /// f32 throughout: the reference.
    F32,
}

impl Prefill {
    /// The GPU kernels for it; `None`: the device's choice.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn gemm_kind(self) -> Option<GemmKind> {
        match self {
            Prefill::Auto => None,
            Prefill::Int16 => Some(GemmKind::Int16),
            Prefill::Int8 => Some(GemmKind::Int8),
            Prefill::F16 => Some(GemmKind::CoopF16),
            Prefill::F32 => Some(GemmKind::F32),
        }
    }
}

impl std::str::FromStr for Prefill {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(Prefill::Auto),
            "int16" => Ok(Prefill::Int16),
            "int8" => Ok(Prefill::Int8),
            "f16" => Ok(Prefill::F16),
            "f32" => Ok(Prefill::F32),
            _ => anyhow::bail!("unknown prefill precision {s:?} (expected auto, int16, int8, f16 or f32)"),
        }
    }
}

/// Arithmetic of GPU passes of up to 16 rows (decode steps, speculative verification,
/// short prompts; see [`Prefill`] for when longer passes take the same kernels).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Decode {
    /// The device's choice ([`yaxus_backend_gpu::Tuning::gemv`]; `Int16` on the AMD GPUs
    /// measured so far, `F32` on Apple GPUs).
    #[default]
    Auto,
    /// Integer dot products, activations rounded to 15 bits per block of 32: logits move
    /// by about 6e-3 against `F32`; a 7-row pass is 1.4x (2B) to 1.6x (8B) faster on a
    /// 780M, single rows cost about 1% more. A row's logits do not depend on the other
    /// rows of its pass.
    Int16,
    /// f32 throughout: the reference.
    F32,
}

impl Decode {
    /// The GPU kernels for it (of the target model; see [`Gemv::for_draft`]), with
    /// `device` the device's choice.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn gemv(self, device: Gemv) -> Gemv {
        match self {
            Decode::Auto => device,
            Decode::Int16 => Gemv::INT16,
            Decode::F32 => Gemv::F32,
        }
    }
}

/// Requests one decode pass advances by default with `gemv`: 16 with the integer kernels,
/// where extra rows are cheap, 8 with f32, where 8 rows is the cheapest per row.
pub fn default_decode_rows(gemv: Gemv) -> usize {
    if gemv.planes.is_some() { 16 } else { 8 }
}

impl std::str::FromStr for Decode {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(Decode::Auto),
            "int16" => Ok(Decode::Int16),
            "f32" => Ok(Decode::F32),
            _ => anyhow::bail!("unknown decode precision {s:?} (expected auto, int16 or f32)"),
        }
    }
}

/// How the engine takes requests. The rest of its configuration follows from the
/// other [`LoadOptions`] or keeps the engine's defaults.
#[derive(Debug, Clone)]
pub struct Scheduling {
    /// Requests decoding or prefilling at once (the rest wait).
    pub max_streams: usize,
    /// Most requests one decode pass advances; `None`: [`default_decode_rows`].
    pub decode_rows: Option<usize>,
    /// Finished requests kept in the response cache (0: none).
    pub response_cache: usize,
}

impl Default for Scheduling {
    fn default() -> Self {
        let e = EngineConfig::default();
        Scheduling { max_streams: e.max_streams, decode_rows: None, response_cache: e.response_cache }
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
    /// Reuse the KV of finished requests for prompts sharing a prefix.
    pub prefix_cache: bool,
    /// KV arena size in tokens (per model).
    pub kv_tokens: usize,
    pub scheduling: Scheduling,
}

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            backend: BackendKind::Auto,
            quant: Quant::Auto,
            draft: None,
            prefill: Prefill::default(),
            decode: Decode::default(),
            prefix_cache: true,
            kv_tokens: yaxus_model::DEFAULT_KV_TOKENS,
            scheduling: Scheduling::default(),
        }
    }
}

impl LoadOptions {
    /// The engine's configuration, with or without the draft model, for the target's
    /// decode kernels `gemv`.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub(crate) fn engine_config(&self, draft: Option<&DraftOptions>, gemv: Gemv) -> EngineConfig {
        let s = &self.scheduling;
        let mut cfg = EngineConfig {
            max_streams: s.max_streams,
            decode_rows: s.decode_rows.unwrap_or(default_decode_rows(gemv)),
            response_cache: s.response_cache,
            ..EngineConfig::default()
        };
        if let Some(d) = draft {
            cfg.draft_k = d.k;
        }
        cfg
    }
}
