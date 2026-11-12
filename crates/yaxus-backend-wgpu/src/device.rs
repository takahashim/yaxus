//! What the GPU can do ([`DeviceCaps`], detected from the adapter), how the kernels are
//! compiled for it ([`CompileSettings`]) and the measured choices they are run with on
//! it ([`Tuning`]).

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
    /// Cooperative matrices for the `Coop` GEMM (see `coop_size`).
    pub coop: bool,
    /// The shader compiler may reassociate float arithmetic and contract it into FMAs
    /// (Metal: wgpu leaves `MTLCompileOptions` at their defaults and offers no way to turn
    /// it off). Kernels that must agree to the bit then may not.
    pub fast_math: bool,
    /// Kernels in Metal Shading Language through wgpu's passthrough (`msl`).
    pub msl: bool,
    /// Their size: 16 (16x16x16, f16 operands and f32 accumulators: matrix cores on
    /// Vulkan) or 8 (8x8x8 in f32: Metal's simdgroup matrices); 0 without.
    pub coop_size: u32,
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
        let props = adapter.cooperative_matrix_properties();
        let has = |size: u32, ab: wgpu::CooperativeScalarType| {
            props.iter().any(|c| {
                (c.m_size, c.n_size, c.k_size) == (size, size, size)
                    && c.ab_type == ab
                    && c.cr_type == wgpu::CooperativeScalarType::F32
            })
        };
        let matrices = subgroups && features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
        let coop_size = if matrices && f16 && has(16, wgpu::CooperativeScalarType::F16) {
            16
        } else if matrices && info.backend == wgpu::Backend::Metal && has(8, wgpu::CooperativeScalarType::F32) {
            8
        } else {
            0
        };
        let coop = coop_size != 0;
        DeviceCaps {
            name: info.name.clone(),
            backend: info.backend,
            vendor,
            device_type: info.device_type,
            f16,
            subgroups,
            coop,
            fast_math: info.backend == wgpu::Backend::Metal,
            msl: info.backend == wgpu::Backend::Metal && features.contains(wgpu::Features::PASSTHROUGH_SHADERS),
            coop_size,
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
    /// f32 passes of up to 16 rows take `gemv_f32` (one kernel family for every row
    /// count, a row's result the same whatever the pass) rather than `gemv1`, `gemvm`
    /// and the single-row GLU kernel.
    pub gemv_f32: bool,
    /// Weight rows per 32-lane group in `gemv_f32`.
    pub gemv_f32_rows: usize,
    /// The MSL kernels of Apple GPUs and their sizes (used where `DeviceCaps::msl`).
    pub apple: AppleTuning,
    /// Weight rows per 32-lane group in the integer GEMV: the loads of the activations are
    /// shared by more dot products, and a single row's rounding (done per workgroup) by
    /// more weights. Measured on a 780M with 1 / 2 / 4 / 8 weight rows, ms per pass: 2B
    /// Q4_K_M 8 rows 33.3 / 26.0 / 26.0 / 28.9, 8B 8 rows 107.9 / 76.6 / 74.4 / 81.3,
    /// 16 rows 185 / 116 / 109 / 180; single-row decode in tok/s with 1 / 4 / 8: 2B Q4_K_M
    /// 44.1 / 44.6 / 43.6, 0.8B Q2_K 166 / 188 / 184.
    pub gemvi_rows: usize,
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

/// The MSL kernels of Apple GPUs (`msl`), and their sizes. They run where the device
/// takes MSL (`DeviceCaps::msl`) and the switch is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AppleTuning {
    /// `rmsnorm`, `add_norm` and `qk_rope` in MSL (`msl::small_kernels`).
    pub msl_small: bool,
    /// Passes of up to 16 rows on `msl::gemv2` (for the formats it reads), in place of
    /// `gemv_f32`.
    pub gemv2: bool,
    /// The most rows gemv2 takes in its scalar kernel; more go to simdgroup matrices (M2,
    /// 2B Q4_K_M, 2 rows: scalar 26.2 ms, matrices 34.7).
    pub gemv2_scalar_max: usize,
    /// Weight tiles of 8 rows a simdgroup of gemv2's matrix kernel takes, for passes of
    /// up to 8 and up to 16 rows (16 rows with 4: 1470 ms, registers spill).
    pub gemv2_tiles: [usize; 2],
    /// Rows a gemv2 dispatch takes at most; more go to further dispatches.
    pub gemv2_max_rows: usize,
    /// On the WGSL matrix GEMM (`gemm_coop8`), a rest of at most this many rows past the
    /// last whole 64-row tile goes to gemv2 (M2, 2B Q4_K_M, ms: a tile 166, gemv2 16 rows
    /// 56, 8 rows 35).
    pub gemv2_tail_rows: usize,
    /// The matrix GEMM (`GemmKind::Coop`, `CoopF16`) on `msl::gemm` in place of
    /// `gemm_coop8`.
    pub gemm_msl: bool,
}

impl Default for AppleTuning {
    fn default() -> Self {
        AppleTuning {
            msl_small: false,
            gemv2: false,
            gemv2_scalar_max: 2,
            gemv2_tiles: [4, 2],
            gemv2_max_rows: 16,
            gemv2_tail_rows: 40,
            gemm_msl: false,
        }
    }
}

impl AppleTuning {
    /// With the environment's overrides: `YAXUS_MSL_SMALL=0|1`, `YAXUS_GEMV2=0|1`,
    /// `YAXUS_GEMV2_SCALAR`, `YAXUS_GEMV2_NT`, `YAXUS_GEMV2_NT2`, `YAXUS_GEMV2_MAX_ROWS`,
    /// `YAXUS_GEMV2_TAIL` and `YAXUS_GEMM_MSL=wgsl|f32|f16` (`f16` also selects
    /// `GemmKind::CoopF16`, in `Tuning::with_env`).
    fn with_env(mut self) -> Self {
        let var = |name: &str| std::env::var(name).ok();
        let num = |name: &str| var(name).and_then(|v| v.parse::<usize>().ok());
        let switch = |name: &str, v: &mut bool| match var(name).as_deref() {
            Some("0") => *v = false,
            Some("1") => *v = true,
            _ => {}
        };
        switch("YAXUS_MSL_SMALL", &mut self.msl_small);
        switch("YAXUS_GEMV2", &mut self.gemv2);
        if let Some(v) = num("YAXUS_GEMV2_SCALAR") {
            self.gemv2_scalar_max = v;
        }
        if let Some(v) = num("YAXUS_GEMV2_NT") {
            self.gemv2_tiles[0] = v;
        }
        if let Some(v) = num("YAXUS_GEMV2_NT2") {
            self.gemv2_tiles[1] = v;
        }
        if let Some(v) = num("YAXUS_GEMV2_MAX_ROWS") {
            self.gemv2_max_rows = v.clamp(1, crate::GEMVM_MAX_T);
        }
        if let Some(v) = num("YAXUS_GEMV2_TAIL") {
            self.gemv2_tail_rows = v;
        }
        match var("YAXUS_GEMM_MSL").as_deref() {
            Some("wgsl") => self.gemm_msl = false,
            Some("f32" | "f16") => self.gemm_msl = true,
            _ => {}
        }
        self
    }
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
            gemv_f32: false,
            gemv_f32_rows: 2,
            apple: AppleTuning::default(),
            gemvi_rows: 4,
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
        // (by vendor, not backend: an AMD GPU in an Intel Mac would keep the integer
        // kernels on Metal, unmeasured)
        if caps.vendor == Vendor::Apple {
            // Apple GPUs have no integer dot product: `dot4I8Packed` is four multiplies
            // and the byte extractions, and the integer kernels lose to the f32 ones
            // (M2, 2B Q4_K_M, ms per pass of 1 / 4 / 16 / 256 rows: integer 48 / 382 /
            // 3450 / 39900, f32 30 / 56 / 305 / 1681).
            t.gemm = GemmKind::F32;
            t.gemv = Gemv::F32;
            t.gemv_f32 = true;
            // (M2, 2B Q4_K_M, ms per pass of 1 / 2 / 4 / 8 / 16 rows: gemv_f32 22.5 / 28.2 /
            // 40.9 / 59.4 / 140, gemv2 21.1 / 26.4 / 34.0 / 34.9 / 55.8; gemv2 is checked
            // at start, `Gpu::gemv2_rows_agree`. 32 / 64 / 256 rows: `gemm_coop8` 161 / 166
            // / 648, MSL f32 tiles about 78 / 136 / 536; llama.cpp, f16 tiles, 71 / 140 / 515)
            t.apple = AppleTuning { msl_small: true, gemv2: true, gemm_msl: true, ..t.apple };
            // (the tiled attention without staged queries: M2, 2B, ms per pass of 256 /
            // 512 / 1024 rows with 64 / 128 / 192 / 384 workgroups 677 / 686 / 678 / 716,
            // 1403 / 1399 / 1396 / 1427, 2931 / 2933 / 2933 / 2996; per-row 732 / 1628 / 3886)
            t.tile_wgs = 128;
            if caps.coop_size == 8 {
                // The 8x8 matrix GEMM in f32: 2.1x the f32 GEMM (256 rows 1744 -> 809 ms),
                // and a whole 64-row tile (32 rows: 197 ms) costs less than two 16-row
                // f32 GEMVs (17 rows: 246 ms), so passes of more than 16 rows take it.
                t.gemm = GemmKind::Coop;
                t.multi_split_max_rows = crate::GEMVM_MAX_T;
            }
        }
        t
    }

    /// The tuning with the overrides of the environment, for experiments and tests:
    /// `YAXUS_GEMM=f32|coop|coop16|int16|int8`, `YAXUS_GEMV=f32|int16|int8|mixed`,
    /// `YAXUS_TILE_WGS`, `YAXUS_GEMVM_R`, `YAXUS_EARLY_SUBMIT`, `YAXUS_ATTN_SPLIT_WGS`,
    /// `YAXUS_SHORT_CONTEXT`, `YAXUS_GEMV_F32=0|1`, `YAXUS_GEMV_F32_R`.
    pub fn with_env(mut self) -> Self {
        let var = |name: &str| std::env::var(name).ok();
        match var("YAXUS_GEMM").as_deref() {
            Some("f32") => self.gemm = GemmKind::F32,
            Some("coop") => self.gemm = GemmKind::Coop,
            Some("coop16") => self.gemm = GemmKind::CoopF16,
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
        if let Some(v) = var("YAXUS_EARLY_SUBMIT").and_then(|v| v.parse().ok()) {
            self.early_submit_dispatches = v;
        }
        if let Some(v) = var("YAXUS_ATTN_SPLIT_WGS").and_then(|v| v.parse().ok()) {
            self.attn_split_wgs = v;
        }
        if let Some(v) = var("YAXUS_SHORT_CONTEXT").and_then(|v| v.parse().ok()) {
            self.short_context = v;
        }
        if let Some(v) = var("YAXUS_TILE_WGS").and_then(|v| v.parse().ok()) {
            self.tile_wgs = v;
        }
        if let Some(v) = var("YAXUS_GEMVM_R").and_then(|v| v.parse().ok()) {
            self.gemvm_rows = v;
        }
        match var("YAXUS_GEMV_F32").as_deref() {
            Some("0") => self.gemv_f32 = false,
            Some("1") => self.gemv_f32 = true,
            _ => {}
        }
        if var("YAXUS_GEMM_MSL").as_deref() == Some("f16") {
            self.gemm = GemmKind::CoopF16;
        }
        self.apple = self.apple.with_env();
        if let Some(v) = var("YAXUS_GEMV_F32_R").and_then(|v| v.parse().ok()) {
            self.gemv_f32_rows = v;
        }
        self
    }

    /// Rows up to which a pass runs as GEMVs of up to 16 rows each with `gemm` as the
    /// prefill GEMM.
    pub(crate) fn gemv_max_rows(&self, gemm: GemmKind) -> usize {
        match gemm {
            GemmKind::F32 => crate::GEMVM_MAX_T,
            GemmKind::Int8 => self.gemv_int8_max_rows,
            GemmKind::Coop | GemmKind::CoopF16 | GemmKind::Int16 => self.gemv_int_max_rows,
        }
    }
}

/// The safety code naga and wgpu add to every kernel, where the kernels do without it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CompileSettings {
    /// naga's counter on every loop against infinite loops. It keeps the compilers from
    /// unrolling loops: on Metal private arrays indexed by the loop (the f32 GEMM's
    /// accumulators) then live in memory, and the 256-row f32 pass of the 2B took 15.4 s
    /// instead of 1.7 s on an M2. On a 780M without it, ms per pass of the 2B: int16 256
    /// rows 477 -> 465, f32 9 / 16 rows 57.4 / 111.7 -> 46.7 / 101.4; the document
    /// bench 47.9 -> 49.2 tok/s. The kernels' loops all end (their bounds are shapes).
    pub loop_bounding: bool,
    /// Zeroing workgroup memory on entry. Every kernel writes its workgroup memory before
    /// reading it; on Metal naga zeroes it from one lane per workgroup (32 KiB in
    /// `gemvm16`, an 8-row f32 pass of the 2B 1.8 s instead of 0.17 s on an M2). (No
    /// difference measured on a 780M, so Vulkan keeps it.)
    pub zero_workgroup_memory: bool,
}

impl CompileSettings {
    /// No loop counters anywhere; workgroup memory zeroed except on Metal.
    pub(crate) fn for_device(caps: &DeviceCaps) -> Self {
        CompileSettings { loop_bounding: false, zero_workgroup_memory: caps.backend != wgpu::Backend::Metal }
    }

    /// With the overrides of the environment, for experiments: `YAXUS_LOOP_BOUNDING=0|1`
    /// and `YAXUS_ZERO_WGMEM=0|1`.
    pub(crate) fn with_env(mut self) -> Self {
        let flag = |name: &str| match std::env::var(name).as_deref() {
            Ok("0") => Some(false),
            Ok("1") => Some(true),
            _ => None,
        };
        if let Some(v) = flag("YAXUS_LOOP_BOUNDING") {
            self.loop_bounding = v;
        }
        if let Some(v) = flag("YAXUS_ZERO_WGMEM") {
            self.zero_workgroup_memory = v;
        }
        self
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
