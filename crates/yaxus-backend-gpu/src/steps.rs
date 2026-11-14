//! The dispatches of a matrix product: kernels, immediates, grids and which buffers each
//! binds, from its plan and shapes alone. `ops::matmul` records them, and
//! `Gpu::precompile` compiles their kernels ahead of use, so the two take the same
//! kernels by construction.

use crate::commands::grid;
use crate::device::{DeviceCaps, Tuning};
use crate::kernels::KernelId;
use crate::plan::{self, GluPlan, MatmulCase, MatmulPlan, Piece};
use crate::{Format, GEMVM_MAX_T};

/// Which of a matrix's buffers a dispatch binds (see `Matrix::bufs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    /// The values and the scales the GEMVs read.
    Decode,
    /// The values and the per-16 scales of Q6, Q3 and Q2 the GEMMs read.
    Prefill,
}

/// What a dispatch binds, after the output `y`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Binds {
    /// The activation rows `x`, then the weights.
    Rows(Layout),
    /// The int8 planes of the rows, the weights, then the rows' scales and sums (see
    /// [`Quantize`]).
    Planes(Layout),
}

/// One dispatch of a matrix product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Dispatch {
    pub kernel: KernelId,
    pub binds: Binds,
    pub imm: [u32; 4],
    pub groups: (u32, u32),
}

/// The rounding of the activation rows to the int8 planes the integer kernels read, into
/// buffers of `rows` rows (bindings: the planes, `x`, the scales and sums).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Quantize {
    pub kernel: KernelId,
    pub planes: usize,
    pub rows: usize,
    pub imm: [u32; 4],
    pub groups: (u32, u32),
}

/// The dispatches of one matrix product, into an output of `rows` rows (room for
/// `out_rows`) and `cols` columns.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MatmulSteps {
    pub rows: usize,
    pub out_rows: usize,
    pub cols: usize,
    pub quantize: Option<Quantize>,
    pub dispatches: Vec<Dispatch>,
    /// The matrix work, for `Recorder::account`.
    pub flops: f64,
    fmt: Format,
    k: usize,
}

/// `x · w^T` over `case` (`n` the matrix's rows).
pub(crate) fn matmul(case: &MatmulCase, caps: &DeviceCaps, tuning: &Tuning) -> MatmulSteps {
    let &MatmulCase { t, n, k, fmt, .. } = case;
    let plan = case.plan(caps, tuning);
    let mut s = MatmulSteps::new(case, plan.output_rows(t), n);
    let apple = &tuning.apple;
    match plan {
        MatmulPlan::GemvInt { planes, inline } => s.gemv_int(planes, inline, false, tuning),
        MatmulPlan::GemmInt { planes } => s.gemm_int(planes),
        MatmulPlan::GemvF32 => s.gemv_f32(false, tuning),
        MatmulPlan::Gemv2 => s.pieces(&plan::gemv2_pieces(0, t, false, apple)),
        MatmulPlan::GemmMsl { f16, gemv2 } => s.pieces(&plan::gemm_msl_pieces(t, f16, gemv2, apple)),
        MatmulPlan::GemvMulti { first } => {
            let groups = grid(n.div_ceil(8 * tuning.gemvm_rows) as u32);
            for (r0, rows) in [(0, first), (first, t - first)] {
                if rows > 0 {
                    s.push(KernelId::Gemvm { t: rows, fmt }, Binds::Rows(Layout::Decode), [n, k, rows, r0], groups);
                }
            }
        }
        MatmulPlan::Gemv1 => {
            s.push(KernelId::Gemv1(fmt), Binds::Rows(Layout::Decode), [n, k, 1, 0], grid(n.div_ceil(8) as u32))
        }
        MatmulPlan::Gemv8 => {
            s.push(KernelId::Gemv8(fmt), Binds::Rows(Layout::Decode), [n, k, t, 0], grid(n.div_ceil(8) as u32))
        }
        MatmulPlan::GemmCoop { gemv2_rest } => {
            // The rows past the last whole tile go to gemv2.
            let tiled = t - gemv2_rest;
            s.pieces(&plan::gemv2_pieces(tiled, gemv2_rest, false, apple));
            if tiled > 0 {
                // (`gemm_coop8` reads the decode layout: it decodes through `load16`)
                let layout = if caps.coop_size == 8 { Layout::Decode } else { Layout::Prefill };
                let groups = ((n / 64) as u32, tiled.div_ceil(64) as u32);
                s.push(KernelId::GemmCoop(fmt), Binds::Rows(layout), [n, k, tiled, 0], groups);
            }
        }
        MatmulPlan::GemmF32 => {
            let groups = (n.div_ceil(128) as u32, t.div_ceil(64) as u32);
            s.push(KernelId::GemmF32(fmt), Binds::Rows(Layout::Prefill), [n, k, t, 0], groups);
        }
    }
    s
}

/// `swiglu(x · w^T)` over `case` for a fused gate/up matrix, or `None` when it is the
/// product followed by the `swiglu` kernel.
pub(crate) fn glu(case: &MatmulCase, caps: &DeviceCaps, tuning: &Tuning) -> Option<MatmulSteps> {
    let &MatmulCase { t, n, k, fmt, .. } = case;
    let inter = n / 2;
    let mut s = MatmulSteps::new(case, t, inter);
    match case.plan_glu(caps, tuning) {
        GluPlan::GemvInt { planes, inline } => s.gemv_int(planes, inline, true, tuning),
        GluPlan::GemvF32 => s.gemv_f32(true, tuning),
        GluPlan::Gemv2 => s.pieces(&plan::gemv2_pieces(0, t, true, &tuning.apple)),
        GluPlan::Glu1 => {
            // (for Q4 the block variant `GLU_32` measured slower: 174 vs 138 us on 12288x1536)
            let groups = grid(inter.div_ceil(4) as u32);
            s.push(KernelId::Glu1(fmt), Binds::Rows(Layout::Decode), [inter, k, 1, 0], groups);
        }
        GluPlan::MatmulThenSwiglu => return None,
    }
    Some(s)
}

/// The dispatches of `pieces` alone, over `case` (gemv2's start check runs gemv2 itself,
/// whatever the plan would take).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn of_pieces(case: &MatmulCase, pieces: &[Piece]) -> MatmulSteps {
    let mut s = MatmulSteps::new(case, case.t, case.n);
    s.pieces(pieces);
    s
}

impl MatmulSteps {
    fn new(case: &MatmulCase, out_rows: usize, cols: usize) -> Self {
        MatmulSteps {
            rows: case.t,
            out_rows,
            cols,
            quantize: None,
            dispatches: Vec::new(),
            // (every plan covers the pass's rows once, against the whole matrix)
            flops: 2.0 * (case.t * case.n * case.k) as f64,
            fmt: case.fmt,
            k: case.k,
        }
    }

    /// Every kernel the steps take.
    pub(crate) fn kernels(&self) -> impl Iterator<Item = KernelId> + '_ {
        self.quantize.iter().map(|q| q.kernel).chain(self.dispatches.iter().map(|d| d.kernel))
    }

    fn push(&mut self, kernel: KernelId, binds: Binds, imm: [usize; 4], groups: (u32, u32)) {
        self.dispatches.push(Dispatch { kernel, binds, imm: imm.map(|v| v as u32), groups });
    }

    /// Workgroups of a GEMV whose 32-lane groups take `rw` output rows each (8 groups a
    /// workgroup; with `glu` 4 take gate rows and 4 up rows).
    fn lane_groups(&self, rw: usize, glu: bool) -> (u32, u32) {
        grid(self.cols.div_ceil(rw * if glu { 4 } else { 8 }) as u32)
    }

    /// Integer GEMVs of up to 16 rows each (see [`plan::gemv_int_parts`]); a single row
    /// rounds itself with `inline`.
    fn gemv_int(&mut self, planes: usize, inline: bool, glu: bool, tuning: &Tuning) {
        let (fmt, t, k, out) = (self.fmt, self.rows, self.k, self.cols);
        let rw = tuning.gemvi_rows;
        let groups = self.lane_groups(rw, glu);
        let kernel = |t, inline| KernelId::GemvInt { fmt, planes, t, rw, glu, inline };
        if inline {
            debug_assert_eq!(t, 1);
            self.push(kernel(1, true), Binds::Rows(Layout::Decode), [out, k, 1, 0], groups);
            return;
        }
        let xmode = fmt.gemv_int().expect("the plan takes the integer GEMV only for its formats").xmode;
        self.quantize = Some(Quantize {
            kernel: KernelId::XquantRows { planes },
            planes,
            rows: t,
            imm: [k as u32, t as u32, xmode, 0],
            groups: grid((t * (k / 32)).div_ceil(64) as u32),
        });
        for (r0, rows) in plan::gemv_int_parts(t) {
            self.push(kernel(rows, false), Binds::Planes(Layout::Decode), [out, k, rows, r0], groups);
        }
    }

    /// The integer GEMM: the rows rounded into planes padded to whole 128-row tiles, then
    /// the tiles of [`plan::gemm_int_tiles`].
    fn gemm_int(&mut self, planes: usize) {
        let (fmt, t, k, n) = (self.fmt, self.rows, self.k, self.cols);
        let xmode = fmt.gemm_int().expect("the plan takes the integer GEMM only for its formats").xmode;
        let padded = t.div_ceil(128) * 128;
        self.quantize = Some(Quantize {
            kernel: KernelId::Xquant { planes },
            planes,
            rows: padded,
            imm: [k as u32, t as u32, xmode, 0],
            groups: grid(((padded / 4) * (k / 32)).div_ceil(64) as u32),
        });
        for (tn, r0, tiles) in plan::gemm_int_tiles(t) {
            let groups = (n.div_ceil(64) as u32, tiles as u32);
            self.push(KernelId::GemmInt { fmt, planes, tn }, Binds::Planes(Layout::Prefill), [n, k, t, r0], groups);
        }
    }

    /// `gemv_f32` for up to 16 rows, in one dispatch.
    fn gemv_f32(&mut self, glu: bool, tuning: &Tuning) {
        let (fmt, t, k, out) = (self.fmt, self.rows, self.k, self.cols);
        debug_assert!(t <= GEMVM_MAX_T);
        let rw = tuning.gemv_f32_rows;
        let groups = self.lane_groups(rw, glu);
        self.push(KernelId::GemvF32 { fmt, t, rw, glu }, Binds::Rows(Layout::Decode), [out, k, t, 0], groups);
    }

    /// The MSL kernels of `pieces`, each on its rows of `x` and `y`.
    fn pieces(&mut self, pieces: &[Piece]) {
        let (fmt, k, out) = (self.fmt, self.k, self.cols);
        for &piece in pieces {
            match piece {
                Piece::Gemv2 { r0, kernel } => {
                    let groups = grid(out.div_ceil(kernel.outputs()) as u32);
                    let imm = [out, k, kernel.t, r0];
                    self.push(KernelId::Gemv2 { fmt, k: kernel }, Binds::Rows(Layout::Decode), imm, groups);
                }
                Piece::Gemm { r0, rows, kernel } => {
                    let groups = (rows.div_ceil(kernel.rows()) as u32, (out / 64) as u32);
                    let imm = [out, k, r0 + rows, r0];
                    self.push(KernelId::GemmMsl { fmt, k: kernel }, Binds::Rows(Layout::Decode), imm, groups);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{Api, DeviceKind, Vendor};
    use crate::{GemmKind, Gemv};

    fn caps() -> DeviceCaps {
        DeviceCaps {
            name: "test".into(),
            api: Api::Vulkan,
            vendor: Vendor::Amd,
            kind: DeviceKind::Integrated,
            f16: true,
            subgroups: true,
            coop: true,
            fast_math: false,
            msl: false,
            coop_size: 16,
            timestamps: false,
            max_workgroup_storage: 65536,
        }
    }

    #[test]
    fn integer_passes_round_once_and_split_by_16_rows() {
        let (c, tu) = (caps(), Tuning::for_device(&caps()));
        let case = MatmulCase { t: 40, n: 2048, k: 2048, fmt: Format::Q4K, gemm: GemmKind::Int16, gemv: Gemv::INT16 };
        let s = matmul(&case, &c, &tu);
        assert_eq!(s.quantize.map(|q| (q.kernel, q.rows)), Some((KernelId::XquantRows { planes: 2 }, 40)));
        let parts: Vec<_> = s.dispatches.iter().map(|d| (d.imm[2], d.imm[3])).collect();
        assert_eq!(parts, [(14, 0), (13, 14), (13, 27)]);
        assert!(s.dispatches.iter().all(|d| d.binds == Binds::Planes(Layout::Decode)));
        // A GEMM pass's output has room for whole 64-row tiles; a GEMV pass's does not.
        let gemm = matmul(&MatmulCase { t: 150, ..case }, &c, &tu);
        assert_eq!((gemm.out_rows, s.out_rows), (192, 40));
        assert_eq!(gemm.quantize.map(|q| q.rows), Some(256));
        // A single row rounds itself and binds the f32 row.
        let one = matmul(&MatmulCase { t: 1, ..case }, &c, &tu);
        assert!(one.quantize.is_none() && one.dispatches[0].binds == Binds::Rows(Layout::Decode));
    }

    #[test]
    fn the_glu_of_a_single_f32_row_is_one_dispatch_of_half_the_rows() {
        let (c, tu) = (caps(), Tuning::for_device(&caps()));
        let case = MatmulCase { t: 1, n: 4096, k: 1024, fmt: Format::Bf16, gemm: GemmKind::F32, gemv: Gemv::F32 };
        let s = glu(&case, &c, &tu).unwrap();
        assert_eq!((s.cols, s.dispatches.len()), (2048, 1));
        assert_eq!(s.dispatches[0].kernel, KernelId::Glu1(Format::Bf16));
        assert!(glu(&MatmulCase { t: 64, ..case }, &c, &tu).is_none());
    }
}
