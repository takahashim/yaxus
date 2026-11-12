//! What the GPU can do ([`DeviceCaps`], detected from the adapter) and the measured
//! choices the kernels are run with on it ([`Tuning`]).

use crate::{GemmKind, Gemv};

/// The graphics API the backend runs on: Vulkan on Linux and Windows, Metal on macOS.
pub(crate) fn platform_backends() -> wgpu::Backends {
    if cfg!(target_os = "macos") { wgpu::Backends::METAL } else { wgpu::Backends::VULKAN }
}

/// GPU vendors the tuning distinguishes (PCI vendor ids; Apple's on Metal).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor {
    Amd,
    Apple,
    Intel,
    Nvidia,
    Other,
}

impl Vendor {
    fn of(info: &wgpu::AdapterInfo) -> Self {
        match info.vendor {
            0x1002 => Vendor::Amd,
            0x106b => Vendor::Apple,
            0x8086 => Vendor::Intel,
            0x10de => Vendor::Nvidia,
            _ if info.backend == wgpu::Backend::Metal => Vendor::Apple,
            _ => Vendor::Other,
        }
    }
}

/// Features of the device the kernels depend on.
#[derive(Debug, Clone)]
pub struct DeviceCaps {
    /// The adapter's name, e.g. `AMD Radeon 780M Graphics (RADV PHOENIX)`.
    pub name: String,
    pub backend: wgpu::Backend,
    pub vendor: Vendor,
    /// Integrated (sharing the host's memory) or discrete.
    pub device_type: wgpu::DeviceType,
    /// `shader-f16`.
    pub f16: bool,
    /// Subgroups of at least 32 lanes: the multi-row and integer GEMVs, the split
    /// attention and `add_norm_sg` reduce within 32 lanes with shuffles.
    pub subgroups: bool,
    /// Cooperative matrices of 16x16x16 with f16 operands and f32 accumulators (the
    /// `Coop` GEMM).
    pub coop: bool,
    /// Timestamp queries (`YAXUS_PROFILE=2`).
    pub timestamps: bool,
    pub max_workgroup_storage: u32,
}

impl DeviceCaps {
    pub(crate) fn detect(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        let features = adapter.features();
        let vendor = Vendor::of(&info);
        let f16 = features.contains(wgpu::Features::SHADER_F16);
        // Metal reports subgroup sizes of 4..=64 whatever the GPU, but Apple GPUs run
        // SIMD-groups of 32 threads, as the kernels need.
        let lanes32 = info.subgroup_min_size >= 32 || (info.backend == wgpu::Backend::Metal && vendor == Vendor::Apple);
        let subgroups = features.contains(wgpu::Features::SUBGROUP) && lanes32;
        // (Metal's simdgroup matrices are 8x8, so the `Coop` GEMM is not available there.)
        let coop = f16
            && subgroups
            && features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            && adapter.cooperative_matrix_properties().iter().any(|c| {
                (c.m_size, c.n_size, c.k_size) == (16, 16, 16)
                    && c.ab_type == wgpu::CooperativeScalarType::F16
                    && c.cr_type == wgpu::CooperativeScalarType::F32
            });
        DeviceCaps {
            name: info.name.clone(),
            backend: info.backend,
            vendor,
            device_type: info.device_type,
            f16,
            subgroups,
            coop,
            timestamps: features.contains(wgpu::Features::TIMESTAMP_QUERY),
            max_workgroup_storage: adapter.limits().max_compute_workgroup_storage_size.min(65536),
        }
    }
}

/// Choices that depend on the device's speeds rather than its features: when to submit,
/// which kernel a pass size takes, how attention is split. The values are measured on a
/// Radeon 780M (the comments at their uses give the figures); [`Tuning::for_device`] is
/// where other devices get their own once measured.
#[derive(Debug, Clone)]
pub struct Tuning {
    /// Matmul FLOPs recorded before the commands are submitted. On APUs compute jobs run
    /// on the graphics ring, where the kernel resets the GPU if one job runs for ~2 s
    /// (observed with a 448-token prefill in one submission) and long jobs also stall the
    /// desktop. 32 GFLOP is roughly 40 ms on a 780M; a 2B decode step is ~5 GFLOP.
    pub submit_flops: f64,
    /// Dispatches of a small pass recorded before it is first submitted (about one layer):
    /// the GPU then runs while the host records the rest. 2B GGUF decode 44.0 -> 44.8 tok/s.
    pub early_submit_dispatches: usize,
    /// Rows up to which a pass runs as integer GEMVs (when the handle's `Gemv` has them)
    /// rather than the prefill GEMM, with the int16 (or f16) and the int8 GEMM: reading the
    /// weights two or three times costs less than a GEMM tile. Measured on a 780M, ms per
    /// pass, GEMVs against the int16 / int8 GEMM: 2B Q4_K_M 17 rows 50 / 79 / 59, 32 rows
    /// 73 / 82 / 61, 48 rows 108 / 119 / 83, 64 rows 142 / 122 / 83; 8B 17 rows
    /// 147 / 290 / 260, 32 rows 212 / 295 / 263, 48 rows 318 / 396 / 292, 64 rows
    /// 423 / 402 / 292. (With the f32 GEMM, asked for as the reference, only up to 16.)
    pub gemv_int_max_rows: usize,
    pub gemv_int8_max_rows: usize,
    /// Passes of 17 rows up to this many run as two multi-row f32 GEMVs rather than the
    /// 64-row-tile GEMM: measured on the 2B, two GEMVs win up to about 20 rows (17 rows:
    /// 101 vs 140 ms; 24 rows: 149 vs 142 ms).
    pub multi_split_max_rows: usize,
    /// Weight rows per 32-lane group in the multi-row f32 GEMV.
    pub gemvm_rows: usize,
    /// Weight rows per 32-lane group in the integer GEMV: the loads of the activations are
    /// shared by more dot products, and a single row's rounding (done per workgroup) by
    /// more weights. Measured on a 780M with 1 / 2 / 4 / 8 weight rows, ms per pass: 2B
    /// Q4_K_M 8 rows 33.3 / 26.0 / 26.0 / 28.9, 8B 8 rows 107.9 / 76.6 / 74.4 / 81.3,
    /// 16 rows 185 / 116 / 109 / 180; single-row decode in tok/s with 1 / 4 / 8: 2B Q4_K_M
    /// 44.1 / 44.6 / 43.6, 0.8B Q2_K 166 / 188 / 184.
    pub gemvi_rows: usize,
    /// Single rows through the integer GEMV round their activations inside the kernel
    /// (no `xquant` dispatch of their own, which costs more than its work when a pass is
    /// bound by reading the weights). Off on Metal: it compiles with fast math, and the
    /// rounding inside the kernel then need not give the bits of `xquant`'s, so a row's
    /// result would depend on whether it ran alone.
    pub gemvi_inline: bool,
    /// Passes with at most this many rows use the split-key attention (a workgroup per
    /// row and chunk of keys). Decode passes are rows of different requests, which the
    /// tiled attention of prompt passes handles badly: 16 requests in one pass took 74 ms
    /// with it against 38 ms (2B), so this covers every pass the decode kernels take in
    /// one dispatch. Passes this small are also the ones submitted early.
    pub split_attn_max_rows: usize,
    /// Workgroups the split attention aims for at most when it splits heads: few
    /// workgroups (short contexts, one KV head) leave the GPU idle, more reread the keys
    /// for little gain (780M: 0.8B at 64 keys 26 -> 8.6 us per layer; 2B and 8B at 1024
    /// keys slower when split).
    pub attn_split_wgs: usize,
    /// Contexts up to this many keys take 16-key chunks in the split attention, longer
    /// ones 32: short chunks give short contexts more workgroups, longer ones keep the
    /// merge (`attn_combine`, linear in the chunk count) cheap. Measured on a 780M:
    /// decode attention at 1024 keys 129 -> 102 us per layer (2B), and the 0.8B (one KV
    /// head) gains 11% at 256 keys over the per-head kernel.
    pub short_context: usize,
    /// Tiled attention for passes of more than `split_attn_max_rows` rows: keys are split
    /// into chunks until about this many workgroups run (0: the per-row kernel instead).
    pub tile_wgs: usize,
    /// The prefill GEMM and the kernels of passes of up to 16 rows when the caller does
    /// not choose.
    pub gemm: GemmKind,
    pub gemv: Gemv,
}

impl Tuning {
    /// The values measured on a Radeon 780M.
    fn radeon_780m() -> Self {
        Tuning {
            submit_flops: 32e9,
            early_submit_dispatches: 12,
            gemv_int_max_rows: 48,
            gemv_int8_max_rows: 24,
            multi_split_max_rows: 20,
            gemvm_rows: 2,
            gemvi_rows: 4,
            gemvi_inline: true,
            split_attn_max_rows: 16,
            attn_split_wgs: 64,
            short_context: 64,
            tile_wgs: 384,
            gemm: GemmKind::Int16,
            gemv: Gemv::INT16,
        }
    }

    /// The tuning for `caps`. Only the 780M is measured: every device starts from its
    /// values, which other devices (Apple GPUs on Metal, discrete AMD GPUs) replace here
    /// as they are measured.
    pub fn for_device(caps: &DeviceCaps) -> Self {
        let mut t = Tuning::radeon_780m();
        if caps.backend == wgpu::Backend::Metal {
            t.gemvi_inline = false;
        }
        if caps.vendor == Vendor::Apple {
            // Apple GPUs have no integer dot product: `dot4I8Packed` is four multiplies
            // and the byte extractions, and the integer kernels lose to the f32 ones
            // (M2, 2B Q4_K_M, ms per pass of 1 / 4 / 16 / 256 rows: integer 48 / 382 /
            // 3450 / 39900, f32 30 / 56 / 305 / 1681).
            t.gemm = GemmKind::F32;
            t.gemv = Gemv::F32;
        }
        t
    }

    /// The tuning with the overrides of the environment, for experiments and tests:
    /// `YAXUS_GEMM=f32|coop|int16|int8`, `YAXUS_GEMV=f32|int16|int8|mixed`,
    /// `YAXUS_TILE_WGS` and `YAXUS_GEMVM_R`.
    pub fn with_env(mut self) -> Self {
        let var = |name: &str| std::env::var(name).ok();
        match var("YAXUS_GEMM").as_deref() {
            Some("f32") => self.gemm = GemmKind::F32,
            Some("coop") => self.gemm = GemmKind::Coop,
            Some("int8") => self.gemm = GemmKind::Int8,
            Some("int16") => self.gemm = GemmKind::Int16,
            _ => {}
        }
        match var("YAXUS_GEMV").as_deref() {
            Some("f32") => self.gemv = Gemv::F32,
            Some("int8") => self.gemv = Gemv { planes: Some(1), single_f32: false },
            Some("mixed") => self.gemv = Gemv { planes: Some(2), single_f32: true },
            Some("int16") => self.gemv = Gemv::INT16,
            _ => {}
        }
        if let Some(v) = var("YAXUS_TILE_WGS").and_then(|v| v.parse().ok()) {
            self.tile_wgs = v;
        }
        if let Some(v) = var("YAXUS_GEMVM_R").and_then(|v| v.parse().ok()) {
            self.gemvm_rows = v;
        }
        self
    }

    /// Rows up to which a pass runs as GEMVs of up to 16 rows each with `gemm` as the
    /// prefill GEMM.
    pub(crate) fn gemv_max_rows(&self, gemm: GemmKind) -> usize {
        match gemm {
            GemmKind::F32 => crate::GEMVM_MAX_T,
            GemmKind::Int8 => self.gemv_int8_max_rows,
            GemmKind::Coop | GemmKind::Int16 => self.gemv_int_max_rows,
        }
    }
}

/// Diagnostics asked for in the environment, read once.
#[derive(Debug, Clone, Default)]
pub(crate) struct Diagnostics {
    /// `YAXUS_PROFILE=1`: time recording vs. waiting; `=2`: per-kernel GPU time too.
    pub profile: Option<u8>,
    /// `YAXUS_COMPILE_LOG`: kernel compilations and their times.
    pub compile_log: bool,
    /// `YAXUS_SOFTWARE_GPU=1`: accept a software adapter (lavapipe in CI).
    pub software: bool,
}

impl Diagnostics {
    pub(crate) fn from_env() -> Self {
        Diagnostics {
            profile: std::env::var("YAXUS_PROFILE").ok().map(|v| if v == "2" { 2 } else { 1 }),
            compile_log: std::env::var_os("YAXUS_COMPILE_LOG").is_some(),
            software: std::env::var_os("YAXUS_SOFTWARE_GPU").is_some_and(|v| v == "1"),
        }
    }
}
