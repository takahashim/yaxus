//! Which kernels run a matrix product or an attention, from the shapes, the device and
//! its tuning only. The ops in `lib.rs` carry the plans out.

use crate::device::{AppleTuning, DeviceCaps, Tuning};
use crate::msl;
use crate::{ATTN_TILE_SHARED_BYTES, ATTN_TILE_SHARED_BYTES_NO_QS, Format, GEMV_MAX_T, GEMVM_MAX_T, GemmKind, Gemv};

/// The kernels of `x [t, k] · w [n, k]^T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatmulPlan {
    /// Integer GEMVs (`gemv_int`) of up to 16 rows each, with activations in `planes`
    /// int8 planes; a single row rounds them inside the kernel with `inline` (see
    /// [`gemv_int_inline`]).
    GemvInt { planes: usize, inline: bool },
    /// The integer GEMM (`gemm_int`), for more than 16 rows.
    GemmInt { planes: usize },
    /// Multi-row f32 GEMVs (`gemvm`): the first `first` rows in one dispatch, the rest
    /// (if any) in a second; each reads the weights once.
    GemvMulti { first: usize },
    /// The f32 GEMV of `gemv_f32` for up to 16 rows, in one dispatch.
    GemvF32,
    /// `msl::gemv2` for up to 16 rows ([`gemv2_pieces`]).
    Gemv2,
    /// `msl::gemm` for more than 16 rows ([`gemm_msl_pieces`]; `gemv2`: whether gemv2
    /// may take a short rest). Reads the decode layout.
    GemmMsl { f16: bool, gemv2: bool },
    /// The single-row f32 GEMV.
    Gemv1,
    /// The f32 GEMV for up to `GEMV_MAX_T` rows, without subgroups.
    Gemv8,
    /// The matrix-core GEMM (f16 operands; f32 on Metal's 8x8 matrices), after the last
    /// `gemv2_rest` rows, past the last whole 64-row tile, go to gemv2.
    GemmCoop { gemv2_rest: usize },
    /// The f32 GEMM.
    GemmF32,
}

impl MatmulPlan {
    /// Whether it runs a GEMM, which reads the prefill layout of the weights (the
    /// per-16 scales of Q6, Q3 and Q2); the GEMVs read the decode layout.
    pub(crate) fn is_gemm(self) -> bool {
        matches!(self, MatmulPlan::GemmInt { .. } | MatmulPlan::GemmCoop { .. } | MatmulPlan::GemmF32)
    }
}

/// What a matrix product's plan depends on besides the device.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MatmulCase {
    pub t: usize,
    pub n: usize,
    pub k: usize,
    pub fmt: Format,
    /// The prefill GEMM chosen for the device.
    pub gemm: GemmKind,
    /// The decode kernels of the model's handle.
    pub gemv: Gemv,
}

impl MatmulCase {
    pub(crate) fn plan(&self, caps: &DeviceCaps, tuning: &Tuning) -> MatmulPlan {
        let &MatmulCase { t, n, k, fmt, gemm, gemv } = self;
        if t <= tuning.gemv_max_rows(gemm)
            && caps.subgroups
            && let Some(planes) = gemv.planes(t)
            && fmt.gemv_int().is_some()
        {
            return MatmulPlan::GemvInt { planes, inline: gemv_int_inline(caps, t) };
        }
        if t > GEMVM_MAX_T
            && let Some(planes) = gemm.planes()
            && fmt.gemm_int().is_some()
        {
            return MatmulPlan::GemmInt { planes };
        }
        if t <= GEMVM_MAX_T && tuning.gemv_f32 && caps.subgroups && k.is_multiple_of(16) {
            return if gemv2_takes(caps, tuning, fmt, k) { MatmulPlan::Gemv2 } else { MatmulPlan::GemvF32 };
        }
        // The multi-row GEMVs reduce with subgroup shuffles over blocks of 512 values.
        let multi = caps.subgroups && k.is_multiple_of(512);
        if multi && t > 1 && t <= 2 * GEMVM_MAX_T && (t <= GEMVM_MAX_T || t <= tuning.multi_split_max_rows) {
            let first = if t <= GEMVM_MAX_T { t } else { t.div_ceil(2) };
            return MatmulPlan::GemvMulti { first };
        }
        if t == 1 {
            MatmulPlan::Gemv1
        } else if t <= GEMV_MAX_T {
            MatmulPlan::Gemv8
        } else if gemm != GemmKind::F32 && n.is_multiple_of(64) && caps.coop {
            let gemv2 = gemv2_takes(caps, tuning, fmt, k);
            if gemm_msl_takes(caps, tuning, fmt) {
                return MatmulPlan::GemmMsl { f16: gemm == GemmKind::CoopF16, gemv2 };
            }
            // A tile costs its 64 rows whatever it holds: a short rest goes to gemv2.
            let rest = t % 64;
            let gemv2_rest = if gemv2 && rest <= tuning.apple.gemv2_tail_rows { rest } else { 0 };
            MatmulPlan::GemmCoop { gemv2_rest }
        } else {
            MatmulPlan::GemmF32
        }
    }

    /// The kernels of `swiglu(x · w^T)` for a fused gate/up matrix `w`.
    pub(crate) fn plan_glu(&self, caps: &DeviceCaps, tuning: &Tuning) -> GluPlan {
        match self.plan(caps, tuning) {
            MatmulPlan::GemvInt { planes, inline } => GluPlan::GemvInt { planes, inline },
            MatmulPlan::GemvF32 => GluPlan::GemvF32,
            MatmulPlan::Gemv2 => GluPlan::Gemv2,
            _ if self.t == 1 => GluPlan::Glu1,
            _ => GluPlan::MatmulThenSwiglu,
        }
    }
}

/// Whether an integer GEMV of `t` rows rounds its activations inside the kernel rather
/// than in an `xquant` dispatch of their own (which costs more than its work when a pass
/// is bound by reading the weights): single rows, except where the compiler uses fast
/// math. The rounding inside the kernel then need not give the bits of `xquant`'s, and a
/// row's result would depend on whether it ran alone.
pub(crate) fn gemv_int_inline(caps: &DeviceCaps, t: usize) -> bool {
    t == 1 && !caps.fast_math
}

/// Whether the device runs `msl::gemv2` on matrices of `fmt` (with K a multiple of 256).
pub(crate) fn gemv2_for(caps: &DeviceCaps, tuning: &Tuning, fmt: Format) -> bool {
    tuning.apple.gemv2 && caps.msl && msl::gemv2_reads(fmt)
}

/// Whether passes of up to 16 rows over a matrix of `fmt` with `k` columns run
/// `msl::gemv2` (every row count alike, so that a row's bits do not depend on the pass).
pub(crate) fn gemv2_takes(caps: &DeviceCaps, tuning: &Tuning, fmt: Format, k: usize) -> bool {
    gemv2_for(caps, tuning, fmt) && k.is_multiple_of(256)
}

/// Whether the matrix GEMM over a matrix of `fmt` runs `msl::gemm`.
pub(crate) fn gemm_msl_takes(caps: &DeviceCaps, tuning: &Tuning, fmt: Format) -> bool {
    tuning.apple.gemm_msl && caps.msl && msl::gemm_reads(fmt)
}

/// The tile sizes of `msl::gemm`, in 32 rows of the pass.
pub(crate) const GEMM_MSL_RB: [usize; 2] = [1, 2];

/// One dispatch of a matrix product split over the pass's rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Piece {
    /// Rows `r0 .. r0 + kernel.t` on gemv2.
    Gemv2 { r0: usize, kernel: msl::Gemv2 },
    /// Rows `r0 .. r0 + rows` on `msl::gemm`.
    Gemm { r0: usize, rows: usize, kernel: msl::Gemm },
}

/// The gemv2 dispatches of rows `r0 .. r0 + rows`: up to `gemv2_max_rows` each.
pub(crate) fn gemv2_pieces(r0: usize, rows: usize, glu: bool, apple: &AppleTuning) -> Vec<Piece> {
    let max = apple.gemv2_max_rows.max(1);
    (r0..r0 + rows)
        .step_by(max)
        .map(|start| Piece::Gemv2 { r0: start, kernel: msl::Gemv2::for_rows(max.min(r0 + rows - start), glu, apple) })
        .collect()
}

/// The `msl::gemm` dispatches of a `t`-row pass: 64-row tiles, then the rest past the last
/// whole one by what costs least (M2, 2B Q4_K_M, ms: a 64-row tile 136, a 32-row one
/// about 78, gemv2 16 rows 56): up to 16 rows by gemv2 (where `gemv2`), up to 32 by a
/// 32-row tile, more by another 64-row one.
pub(crate) fn gemm_msl_pieces(t: usize, f16: bool, gemv2: bool, apple: &AppleTuning) -> Vec<Piece> {
    let whole = t / 64 * 64;
    let rest = t - whole;
    let tile = |r0, rows, rb| Piece::Gemm { r0, rows, kernel: msl::Gemm { f16, rb } };
    let mut pieces: Vec<Piece> = Vec::new();
    let wide = if rest > 32 { t } else { whole };
    if wide > 0 {
        pieces.push(tile(0, wide, 2));
    }
    match rest {
        1..=16 if gemv2 => pieces.extend(gemv2_pieces(whole, rest, false, apple)),
        1..=32 => pieces.push(tile(whole, rest, 1)),
        _ => {}
    }
    pieces
}

/// The kernels of `swiglu(x · w^T)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GluPlan {
    /// Integer GEMVs that store `silu(gate) * up` themselves.
    GemvInt { planes: usize, inline: bool },
    /// `gemv_f32` storing `silu(gate) * up` itself.
    GemvF32,
    /// `msl::gemv2` storing `silu(gate) * up` itself.
    Gemv2,
    /// The single-row f32 GLU kernel.
    Glu1,
    /// The matrix product's plan, then the `swiglu` kernel.
    MatmulThenSwiglu,
}

/// Query heads per KV head with a compiled split-attention kernel (`attn_split{g}_..`).
pub(crate) const SPLIT_GROUPS: [usize; 7] = [1, 2, 4, 6, 8, 12, 16];
/// Query heads of one KV head a split-attention workgroup may take (`.._{hg}`).
pub(crate) const SPLIT_HEAD_GROUPS: [usize; 8] = [1, 2, 3, 4, 6, 8, 12, 16];
/// Keys per chunk of the split attention kernels (`.._{ch}_..`).
pub(crate) const SPLIT_CHUNKS: [usize; 2] = [16, 32];
/// Rows per attention tile (`ATTN_TILE`'s `BM`).
pub(crate) const ATTN_TILE_ROWS: usize = 32;

/// The kernels of one layer's attention over a pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttentionPlan {
    /// A workgroup per row, KV head (`hg` of its query heads) and chunk of `ch` keys
    /// (`nch` chunks), then `attn_combine`.
    Split { g: usize, hg: usize, ch: usize, nch: usize },
    /// The tiled kernel, with the keys in `nch` chunks of `ch` (merged by `attn_combine`
    /// when more than one).
    Tile { nch: usize, ch: usize },
    /// A workgroup per row and head.
    PerRow,
}

/// What an attention's plan depends on besides the device.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttentionCase {
    pub rows: usize,
    pub hq: usize,
    pub hkv: usize,
    /// The layer's sliding window, if any.
    pub window: Option<usize>,
    /// The largest position in the pass.
    pub max_pos: usize,
    /// Attention tiles of the pass (runs of consecutive positions of one sequence).
    pub tiles: usize,
}

impl AttentionCase {
    pub(crate) fn plan(&self, caps: &DeviceCaps, tuning: &Tuning) -> AttentionPlan {
        let &AttentionCase { rows: t, hq, hkv, window, max_pos, tiles } = self;
        let g = hq / hkv;
        let ctx = match window {
            Some(w) => (max_pos + 1).min(w),
            None => max_pos + 1,
        };
        let ch = if ctx <= tuning.short_context { 16 } else { 32 };
        let nch = ctx.div_ceil(ch);
        // Query heads per workgroup: few workgroups (short contexts, one KV head) leave
        // the GPU idle, so the heads of a KV head are split over up to `attn_split_wgs`.
        let hg = SPLIT_HEAD_GROUPS
            .into_iter()
            .find(|&h| g.is_multiple_of(h) && t * hkv * (g / h) * nch <= tuning.attn_split_wgs)
            .unwrap_or(g);
        // (the split attention reduces with subgroup shuffles)
        let split_compiled = caps.subgroups && SPLIT_GROUPS.contains(&g) && SPLIT_HEAD_GROUPS.contains(&hg);
        if t <= tuning.split_attn_max_rows && split_compiled {
            return AttentionPlan::Split { g, hg, ch, nch };
        }
        if t > tuning.split_attn_max_rows && tuning.tile_wgs > 0 && tile_compiled(caps) {
            // Split the keys into chunks while there are few workgroups (short passes over
            // long contexts). Keys a tile can span: its first row's window up to its last.
            let span = match window {
                Some(w) => (max_pos + 1).min(w + ATTN_TILE_ROWS - 1),
                None => max_pos + 1,
            };
            let nch = tuning.tile_wgs.div_ceil(tiles * hq).clamp(1, span.div_ceil(128));
            let ch = span.div_ceil(nch).next_multiple_of(32);
            return AttentionPlan::Tile { nch: span.div_ceil(ch), ch };
        }
        AttentionPlan::PerRow
    }
}

/// Whether the tiled attention kernel is compiled: it needs subgroups, and the workgroup
/// memory of one of its variants (see [`tile_stages_queries`]).
pub(crate) fn tile_compiled(caps: &DeviceCaps) -> bool {
    caps.subgroups && caps.max_workgroup_storage >= ATTN_TILE_SHARED_BYTES_NO_QS
}

/// Whether the tiled attention stages the tile's queries in workgroup memory (37.6 KiB in
/// all), or reads them from the buffer at each block of keys (21.1 KiB: devices with 32
/// KiB, such as Apple GPUs).
pub(crate) fn tile_stages_queries(caps: &DeviceCaps) -> bool {
    caps.max_workgroup_storage >= ATTN_TILE_SHARED_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Vendor;

    #[test]
    fn msl_pieces_cover_the_pass() {
        let apple = AppleTuning { gemv2_max_rows: 16, ..AppleTuning::default() };
        let gemv = |r0, t| Piece::Gemv2 { r0, kernel: msl::Gemv2::for_rows(t, false, &apple) };
        let gemm = |r0, rows, rb| Piece::Gemm { r0, rows, kernel: msl::Gemm { f16: false, rb } };
        assert_eq!(gemv2_pieces(0, 16, false, &apple), vec![gemv(0, 16)]);
        assert_eq!(gemv2_pieces(64, 22, false, &apple), vec![gemv(64, 16), gemv(80, 6)]);
        // A rest of up to 16 rows by gemv2, up to 32 by a 32-row tile, more by a 64-row one.
        assert_eq!(gemm_msl_pieces(140, false, true, &apple), vec![gemm(0, 128, 2), gemv(128, 12)]);
        assert_eq!(gemm_msl_pieces(150, false, true, &apple), vec![gemm(0, 128, 2), gemm(128, 22, 1)]);
        assert_eq!(gemm_msl_pieces(70, false, false, &apple), vec![gemm(0, 64, 2), gemm(64, 6, 1)]);
        assert_eq!(gemm_msl_pieces(20, false, true, &apple), vec![gemm(0, 20, 1)]);
        assert_eq!(gemm_msl_pieces(40, false, true, &apple), vec![gemm(0, 40, 2)]);
        assert_eq!(gemm_msl_pieces(256, false, true, &apple), vec![gemm(0, 256, 2)]);
        // The scalar kernel for up to `gemv2_scalar_max` rows, then 4 tiles to 8 rows, 2 to 16.
        let tiles = |t| msl::Gemv2::for_rows(t, false, &apple).tiles;
        assert_eq!((tiles(2), tiles(3), tiles(9)), (None, Some(4), Some(2)));
    }

    fn caps(subgroups: bool, coop: bool) -> DeviceCaps {
        DeviceCaps {
            name: "test".into(),
            backend: wgpu::Backend::Vulkan,
            vendor: Vendor::Amd,
            device_type: wgpu::DeviceType::IntegratedGpu,
            f16: true,
            subgroups,
            coop,
            fast_math: false,
            msl: false,
            coop_size: if coop { 16 } else { 0 },
            timestamps: false,
            max_workgroup_storage: 65536,
        }
    }

    fn tuning() -> Tuning {
        Tuning::for_device(&caps(true, true))
    }

    fn mm(t: usize, fmt: Format, gemm: GemmKind, gemv: Gemv) -> MatmulCase {
        MatmulCase { t, n: 2048, k: 2048, fmt, gemm, gemv }
    }

    #[test]
    fn matmul_plans_follow_the_pass_size() {
        let (c, tu) = (caps(true, true), tuning());
        let int = |t, fmt| mm(t, fmt, GemmKind::Int16, Gemv::INT16).plan(&c, &tu);
        // Integer GEMVs up to 48 rows, then the integer GEMM.
        assert_eq!(int(1, Format::Q8), MatmulPlan::GemvInt { planes: 2, inline: true });
        assert_eq!(int(48, Format::Q4K), MatmulPlan::GemvInt { planes: 2, inline: false });
        assert_eq!(int(49, Format::Q4K), MatmulPlan::GemmInt { planes: 2 });
        // With fast math (Metal) single rows are rounded by `xquant`, as longer passes are.
        let fast = DeviceCaps { fast_math: true, ..c.clone() };
        let one = mm(1, Format::Q8, GemmKind::Int16, Gemv::INT16).plan(&fast, &tu);
        assert_eq!(one, MatmulPlan::GemvInt { planes: 2, inline: false });
        // bf16 has no integer kernels: multi-row GEMVs, two for 17..=20 rows, then the GEMM.
        assert_eq!(int(1, Format::Bf16), MatmulPlan::Gemv1);
        assert_eq!(int(16, Format::Bf16), MatmulPlan::GemvMulti { first: 16 });
        assert_eq!(int(19, Format::Bf16), MatmulPlan::GemvMulti { first: 10 });
        assert_eq!(int(21, Format::Bf16), MatmulPlan::GemmCoop { gemv2_rest: 0 });
        // The f32 reference: GEMVs up to 16 rows, the f32 GEMM beyond.
        let f32 = |t| mm(t, Format::Q8, GemmKind::F32, Gemv::F32).plan(&c, &tu);
        assert_eq!(f32(8), MatmulPlan::GemvMulti { first: 8 });
        assert_eq!(f32(64), MatmulPlan::GemmF32);
        // Without subgroups (software renderers): the plain kernels.
        let plain = caps(false, false);
        let p = |t| mm(t, Format::Q8, GemmKind::Int16, Gemv::INT16).plan(&plain, &tu);
        assert_eq!(
            (p(1), p(8), p(9), p(100)),
            (MatmulPlan::Gemv1, MatmulPlan::Gemv8, MatmulPlan::GemmF32, MatmulPlan::GemmInt { planes: 2 })
        );
        // 9..=16 rows there are a GEMM, which binds the prefill layout of the weights.
        assert!(p(16).is_gemm() && !p(8).is_gemm());
    }

    #[test]
    fn a_draft_handle_decodes_single_rows_in_f32() {
        let (c, tu) = (caps(true, true), tuning());
        let case = |t| mm(t, Format::Q2, GemmKind::Int16, Gemv::INT16.for_draft());
        assert_eq!(case(1).plan(&c, &tu), MatmulPlan::Gemv1);
        assert_eq!(case(1).plan_glu(&c, &tu), GluPlan::Glu1);
        assert_eq!(case(4).plan_glu(&c, &tu), GluPlan::GemvInt { planes: 2, inline: false });
        let bf16 = MatmulCase { fmt: Format::Bf16, ..case(4) };
        assert_eq!(bf16.plan_glu(&c, &tu), GluPlan::MatmulThenSwiglu);
    }

    #[test]
    fn attention_plans_split_decode_and_tile_prompts() {
        let (c, tu) = (caps(true, true), tuning());
        let a =
            |rows, max_pos, window| AttentionCase { rows, hq: 16, hkv: 8, window, max_pos, tiles: rows.div_ceil(32) };
        // A decode row over a short context: 16-key chunks, heads split for workgroups.
        assert_eq!(a(1, 40, None).plan(&c, &tu), AttentionPlan::Split { g: 2, hg: 1, ch: 16, nch: 3 });
        assert!(matches!(a(16, 1000, None).plan(&c, &tu), AttentionPlan::Split { ch: 32, .. }));
        // A prompt pass takes the tiled kernel; a sliding window bounds its span.
        assert!(matches!(a(512, 511, None).plan(&c, &tu), AttentionPlan::Tile { .. }));
        // (64 rows over a 1024-key window span 1055 keys: 9 chunks of 128)
        assert_eq!(a(64, 4000, Some(1024)).plan(&c, &tu), AttentionPlan::Tile { nch: 9, ch: 128 });
        // Without subgroups, or with the tiles off: the per-row kernel.
        assert_eq!(a(1, 40, None).plan(&caps(false, false), &tu), AttentionPlan::PerRow);
        let off = Tuning { tile_wgs: 0, ..tuning() };
        assert_eq!(a(512, 511, None).plan(&c, &off), AttentionPlan::PerRow);
    }
}
