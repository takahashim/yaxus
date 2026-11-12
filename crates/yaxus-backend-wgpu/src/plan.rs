//! Which kernels run a matrix product or an attention, from the shapes, the device and
//! its tuning only. The ops in `lib.rs` carry the plans out.

use crate::device::{DeviceCaps, Tuning};
use crate::{ATTN_TILE_SHARED_BYTES, Format, GEMV_MAX_T, GEMVM_MAX_T, GemmKind, Gemv};

/// The kernels of `x [t, k] · w [n, k]^T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatmulPlan {
    /// Integer GEMVs (`gemv_int`) of up to 16 rows each, with activations in `planes`
    /// int8 planes.
    GemvInt { planes: usize },
    /// The integer GEMM (`gemm_int`), for more than 16 rows.
    GemmInt { planes: usize },
    /// Multi-row f32 GEMVs (`gemvm`): the first `first` rows in one dispatch, the rest
    /// (if any) in a second; each reads the weights once.
    GemvMulti { first: usize },
    /// The single-row f32 GEMV.
    Gemv1,
    /// The f32 GEMV for up to `GEMV_MAX_T` rows, without subgroups.
    Gemv8,
    /// The matrix-core GEMM (f16 operands).
    GemmCoop,
    /// The f32 GEMM.
    GemmF32,
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
            return MatmulPlan::GemvInt { planes };
        }
        if t > GEMVM_MAX_T
            && let Some(planes) = gemm.planes()
            && fmt.gemm_int().is_some()
        {
            return MatmulPlan::GemmInt { planes };
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
            MatmulPlan::GemmCoop
        } else {
            MatmulPlan::GemmF32
        }
    }

    /// The kernels of `swiglu(x · w^T)` for a fused gate/up matrix `w`.
    pub(crate) fn plan_glu(&self, caps: &DeviceCaps, tuning: &Tuning) -> GluPlan {
        match self.plan(caps, tuning) {
            MatmulPlan::GemvInt { planes } => GluPlan::GemvInt { planes },
            _ if self.t == 1 => GluPlan::Glu1,
            _ => GluPlan::MatmulThenSwiglu,
        }
    }
}

/// The kernels of `swiglu(x · w^T)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GluPlan {
    /// Integer GEMVs that store `silu(gate) * up` themselves.
    GemvInt { planes: usize },
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

/// Whether the tiled attention kernel is compiled: it needs subgroups and more workgroup
/// memory than some devices (32 KiB) have.
pub(crate) fn tile_compiled(caps: &DeviceCaps) -> bool {
    caps.subgroups && caps.max_workgroup_storage >= ATTN_TILE_SHARED_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Vendor;

    fn caps(subgroups: bool, coop: bool) -> DeviceCaps {
        DeviceCaps {
            name: "test".into(),
            backend: wgpu::Backend::Vulkan,
            vendor: Vendor::Amd,
            device_type: wgpu::DeviceType::IntegratedGpu,
            f16: true,
            subgroups,
            coop,
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
        assert_eq!(int(1, Format::Q8), MatmulPlan::GemvInt { planes: 2 });
        assert_eq!(int(48, Format::Q4K), MatmulPlan::GemvInt { planes: 2 });
        assert_eq!(int(49, Format::Q4K), MatmulPlan::GemmInt { planes: 2 });
        // bf16 has no integer kernels: multi-row GEMVs, two for 17..=20 rows, then the GEMM.
        assert_eq!(int(1, Format::Bf16), MatmulPlan::Gemv1);
        assert_eq!(int(16, Format::Bf16), MatmulPlan::GemvMulti { first: 16 });
        assert_eq!(int(19, Format::Bf16), MatmulPlan::GemvMulti { first: 10 });
        assert_eq!(int(21, Format::Bf16), MatmulPlan::GemmCoop);
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
    }

    #[test]
    fn a_draft_handle_decodes_single_rows_in_f32() {
        let (c, tu) = (caps(true, true), tuning());
        let case = |t| mm(t, Format::Q2, GemmKind::Int16, Gemv::INT16.for_draft());
        assert_eq!(case(1).plan(&c, &tu), MatmulPlan::Gemv1);
        assert_eq!(case(1).plan_glu(&c, &tu), GluPlan::Glu1);
        assert_eq!(case(4).plan_glu(&c, &tu), GluPlan::GemvInt { planes: 2 });
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
