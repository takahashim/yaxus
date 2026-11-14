//! Matrix products: the f32 kernels, the integer GEMVs and GEMM, and the fused GLU.

use super::grid;
use crate::dev::Buf;
use crate::kernels::name;
use crate::plan::{self, GluPlan, MatmulCase, MatmulPlan, Piece};
use crate::{GEMV_MAX_T, GEMVM_MAX_T, GemmKind, Gemv, Gpu, Tensor, Weight, gemm_int, gemv_int};

/// How an integer GEMV runs: activations in `planes` int8 planes, rounded inside the
/// kernel with `inline` (a single row, see `plan::gemv_int_inline`), and `swiglu` of the
/// product with `glu`.
#[derive(Debug, Clone, Copy)]
struct IntGemv {
    planes: usize,
    inline: bool,
    glu: bool,
}

impl Gpu {
    /// Whether `msl::gemv2` gives a row the same bits alone (the scalar kernel) and in
    /// passes of 8 and 16 rows (simdgroup matrices), on a Q8_0 matrix of varied values.
    /// The matrix kernel relies on the device multiplying 8x8 matrices as an `fma` chain
    /// in order, measured on the M2 only; a device that does not is left without gemv2.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn gemv2_rows_agree(&mut self) -> bool {
        let (n, k, t) = (40, 512, 16);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut bytes = Vec::with_capacity(n * k / 32 * 34);
        for _ in 0..n * k / 32 {
            let d = half::f16::from_f32((1 + next() % 1000) as f32 * 1e-4);
            bytes.extend_from_slice(&d.to_le_bytes());
            bytes.extend((0..32).map(|_| next() as u8));
        }
        let x: Vec<f32> = (0..t * k)
            .map(|_| {
                let r = next();
                let v = (r >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                v * (1u64 << (r % 16)) as f32
            })
            .collect();
        let Some(m) = self.load_ggml(yaxus_model::ggml::GgmlType::Q8_0, &bytes, n, k) else { return false };
        // (gemv2 itself, whatever the plan would take)
        let run = |g: &mut Gpu, r0: usize, rows: usize| {
            let xt = g.tensor(rows, k);
            g.dev.write(xt.buf(), 0, bytemuck::cast_slice(&x[r0 * k..(r0 + rows) * k]));
            let y = g.tensor(rows, n);
            let pieces = plan::gemv2_pieces(0, rows, false, &g.tuning.apple);
            g.run_pieces(&xt, &y, &m, n, false, &pieces);
            g.to_host(&y)
        };
        let all = run(self, 0, t);
        let eight = run(self, 8, 8);
        let ones: Vec<f32> = [0, 9, 15].iter().flat_map(|&r| run(self, r, 1)).collect();
        let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(p, q)| p.to_bits() == q.to_bits());
        same(&eight, &all[8 * n..])
            && [0, 9, 15].iter().enumerate().all(|(i, &r)| same(&ones[i * n..(i + 1) * n], &all[r * n..(r + 1) * n]))
    }

    /// `x · w^T`; passes of up to 16 rows (and more where the integer GEMVs win) take
    /// the kernels `gemv` says.
    pub(crate) fn matmul(&mut self, x: &Tensor, w: &Weight, gemv: Gemv) -> Tensor {
        let m = w.matrix();
        let (n, k) = (m.n, m.k);
        assert_eq!(x.cols, k, "matmul inner dim");
        let t = x.rows;
        // The matrix-core GEMM stores whole 64-token tiles.
        let y = if t > GEMV_MAX_T && self.gemm != GemmKind::F32 {
            // (more than needed when a multi-row GEMV takes it; harmless)
            self.tensor_padded(t, n, t.div_ceil(64) * 64)
        } else {
            self.tensor(t, n)
        };
        let case = MatmulCase { t, n, k, fmt: m.fmt, gemm: self.gemm, gemv };
        let apple = self.tuning.apple;
        let (n32, k32) = (n as u32, k as u32);
        let plan = case.plan(&self.caps, &self.tuning);
        match plan {
            MatmulPlan::GemvInt { planes, inline } => self.gemv_int(x, &y, m, IntGemv { planes, inline, glu: false }),
            MatmulPlan::GemmInt { planes } => self.gemm_int(x, &y, m, planes),
            MatmulPlan::GemvF32 => self.gemv_f32(x, &y, m, n, false),
            MatmulPlan::Gemv2 => self.run_pieces(x, &y, m, n, false, &plan::gemv2_pieces(0, t, false, &apple)),
            MatmulPlan::GemmMsl { f16, gemv2 } => {
                self.run_pieces(x, &y, m, n, false, &plan::gemm_msl_pieces(t, f16, gemv2, &apple));
            }
            MatmulPlan::GemvMulti { first } => {
                let groups = grid(n.div_ceil(8 * self.tuning.gemvm_rows) as u32);
                let dispatches: Vec<_> = [(0, first), (first, t - first)]
                    .into_iter()
                    .filter(|&(_, rows)| rows > 0)
                    .map(|(r0, rows)| (name::gemvm(rows, m.fmt), [n32, k32, rows as u32, r0 as u32], groups))
                    .collect();
                self.f32_kernels(plan, x, &y, m, &dispatches, t);
            }
            MatmulPlan::Gemv1 => {
                let d = (name::gemv1(m.fmt), [n32, k32, 1, 0], grid(n.div_ceil(8) as u32));
                self.f32_kernels(plan, x, &y, m, &[d], t);
            }
            MatmulPlan::Gemv8 => {
                let d = (name::gemv8(m.fmt), [n32, k32, t as u32, 0], grid(n.div_ceil(8) as u32));
                self.f32_kernels(plan, x, &y, m, &[d], t);
            }
            MatmulPlan::GemmCoop { gemv2_rest } => {
                // The rows past the last whole tile go to gemv2 (they account for themselves).
                let tiled = t - gemv2_rest;
                self.run_pieces(x, &y, m, n, false, &plan::gemv2_pieces(tiled, gemv2_rest, false, &apple));
                if tiled > 0 {
                    let groups = ((n / 64) as u32, tiled.div_ceil(64) as u32);
                    let d = (name::gemm_coop(m.fmt), [n32, k32, tiled as u32, 0], groups);
                    self.f32_kernels(plan, x, &y, m, &[d], tiled);
                }
            }
            MatmulPlan::GemmF32 => {
                let groups = (n.div_ceil(128) as u32, t.div_ceil(64) as u32);
                let d = (name::gemm_f32(m.fmt), [n32, k32, t as u32, 0], groups);
                self.f32_kernels(plan, x, &y, m, &[d], t);
            }
        }
        y
    }

    /// The f32 kernels' `dispatches` (name, immediates, grid) of `plan`, `y = x W^T` over
    /// `rows` rows. The GEMMs bind the weights' prefill layout (the per-16 scales, see
    /// `W_Q6`), the GEMVs their decode layout.
    fn f32_kernels(
        &mut self,
        plan: MatmulPlan,
        x: &Tensor,
        y: &Tensor,
        m: &crate::Matrix,
        dispatches: &[(String, [u32; 4], (u32, u32))],
        rows: usize,
    ) {
        // (`gemm_coop8` reads the decode layout too: it decodes through `load16`)
        let coop8 = matches!(plan, MatmulPlan::GemmCoop { .. }) && self.caps.coop_size == 8;
        let mut all = vec![y.buf().clone(), x.buf().clone()];
        all.extend(if plan.is_gemm() && !coop8 { m.prefill_bufs() } else { m.decode_bufs() });
        let refs: Vec<&Buf> = all.iter().collect();
        for (name, params, groups) in dispatches {
            self.dispatch_matmul(name, &refs, params, *groups, (m.n, m.k));
        }
        self.account(2.0 * (rows * m.n * m.k) as f64);
    }

    pub(crate) fn matmul_glu(&mut self, x: &Tensor, w: &Weight, gemv: Gemv) -> Tensor {
        let m = w.matrix();
        let (n2, k) = (m.n, m.k);
        assert_eq!(x.cols, k);
        let case = MatmulCase { t: x.rows, n: n2, k, fmt: m.fmt, gemm: self.gemm, gemv };
        match case.plan_glu(&self.caps, &self.tuning) {
            GluPlan::GemvInt { planes, inline } => {
                let y = self.tensor(x.rows, n2 / 2);
                self.gemv_int(x, &y, m, IntGemv { planes, inline, glu: true });
                return y;
            }
            GluPlan::GemvF32 => {
                let y = self.tensor(x.rows, n2 / 2);
                self.gemv_f32(x, &y, m, n2 / 2, true);
                return y;
            }
            GluPlan::Gemv2 => {
                let y = self.tensor(x.rows, n2 / 2);
                self.run_pieces(x, &y, m, n2 / 2, true, &plan::gemv2_pieces(0, x.rows, true, &self.tuning.apple));
                return y;
            }
            GluPlan::MatmulThenSwiglu => {
                let gu = self.matmul(x, w, gemv);
                return self.swiglu(&gu);
            }
            GluPlan::Glu1 => {}
        }
        // (for Q4 the block variant `GLU_32` measured slower: 174 vs 138 us on 12288x1536)
        let kernel = name::glu(m.fmt);
        let bufs = m.decode_bufs();
        let inter = n2 / 2;
        let y = self.tensor(1, inter);
        let mut all = vec![y.buf().clone(), x.buf().clone()];
        all.extend(bufs);
        let refs: Vec<&Buf> = all.iter().collect();
        self.dispatch_matmul(&kernel, &refs, &[inter as u32, k as u32, 1], grid(inter.div_ceil(4) as u32), (n2, k));
        self.account(2.0 * (n2 * k) as f64);
        y
    }

    /// The MSL dispatches `pieces` of `y = x W^T` (or `swiglu` of it with `glu`, `n` then
    /// the intermediate size), each on its rows of `x` and `y` (see `plan::Piece`).
    fn run_pieces(&mut self, x: &Tensor, y: &Tensor, m: &crate::Matrix, n: usize, glu: bool, pieces: &[Piece]) {
        if pieces.is_empty() {
            return;
        }
        let k = x.cols;
        let bufs = [y.buf().clone(), x.buf().clone()].into_iter().chain(m.decode_bufs()).collect::<Vec<_>>();
        let refs: Vec<&Buf> = bufs.iter().collect();
        let layout = crate::kernels::decode_layout(m.fmt);
        let mut rows = 0;
        // (the weight's shape: `2 * n` rows with `glu`)
        let shape = (if glu { 2 * n } else { n }, k);
        for &piece in pieces {
            match piece {
                Piece::Gemv2 { r0, kernel } => {
                    let name = self.int_kernel(name::gemv2(kernel, m.fmt), layout, || kernel.source(m.fmt).unwrap());
                    let imm = [n as u32, k as u32, kernel.t as u32, r0 as u32];
                    self.dispatch_matmul(&name, &refs, &imm, grid(n.div_ceil(kernel.outputs()) as u32), shape);
                    rows += kernel.t;
                }
                Piece::Gemm { r0, rows: r, kernel } => {
                    let name = self.int_kernel(name::gemm_msl(kernel, m.fmt), layout, || kernel.source(m.fmt).unwrap());
                    let imm = [n as u32, k as u32, (r0 + r) as u32, r0 as u32];
                    let groups = (r.div_ceil(kernel.rows()) as u32, (n / 64) as u32);
                    self.dispatch_matmul(&name, &refs, &imm, groups, shape);
                    rows += r;
                }
            }
        }
        self.account(2.0 * (rows * n * k) as f64 * if glu { 2.0 } else { 1.0 });
    }

    /// `y = x W^T` for more than 16 rows with integer dot products (`gemm_int`): the rows
    /// are quantized to `planes` int8 planes, then tiles of 128 tokens and one of 64 or 32
    /// for the rest (a tile costs its full size whatever it holds).
    fn gemm_int(&mut self, x: &Tensor, y: &Tensor, m: &crate::Matrix, planes: usize) {
        let (t, k, n) = (x.rows, x.cols, m.n);
        let f = m.fmt.gemm_int().expect("the plan takes the integer GEMM only for its formats");
        // The integer GEMM reads the per-16 scale arrays where a format has them.
        let [wq, scales] = <[Buf; 2]>::try_from(m.prefill_bufs()).unwrap();
        let tp = t.div_ceil(128) * 128;
        let xq = self.tensor(tp, planes * k / 4);
        let xs = self.tensor(tp, k / 8);
        let xquant = self.int_kernel(name::xquant(planes), &[false, true, false], || gemm_int::xquant_source(planes));
        let lanes = (tp / 4) * (k / 32);
        self.dispatch(
            &xquant,
            &[xq.buf(), x.buf(), xs.buf()],
            &[k as u32, t as u32, f.xmode, 0],
            grid(lanes.div_ceil(64) as u32),
        );
        let bufs = [y.buf().clone(), xq.buf().clone(), wq.clone(), scales.clone(), xs.buf().clone()];
        let refs: Vec<&Buf> = bufs.iter().collect();
        let rest = t % 128;
        let wide = t / 128 + usize::from(rest > 64);
        let tail = match rest {
            0 | 65.. => None,
            33.. => Some(4),
            _ => Some(2),
        };
        for (tn, r0, tiles) in [(8, 0, wide), (tail.unwrap_or(2), wide * 128, usize::from(tail.is_some()))] {
            if tiles == 0 {
                continue;
            }
            let kernel = self.int_kernel(name::gemm_int(planes, tn, f.name), &[false, true, true, true, true], || {
                gemm_int::source(f, tn, planes)
            });
            self.dispatch_matmul(
                &kernel,
                &refs,
                &[n as u32, k as u32, t as u32, r0 as u32],
                (n.div_ceil(64) as u32, tiles as u32),
                (n, k),
            );
        }
        self.account(2.0 * (t * n * k) as f64);
    }

    /// `y = x W^T` for up to 16 rows with integer dot products (`gemv_int`), or `swiglu`
    /// of it (see [`IntGemv`]).
    fn gemv_int(&mut self, x: &Tensor, y: &Tensor, m: &crate::Matrix, IntGemv { planes, inline, glu }: IntGemv) {
        let (t, k) = (x.rows, x.cols);
        // (with `glu` the matrix is [gate; up], and the output the intermediate size)
        let n = if glu { m.n / 2 } else { m.n };
        let f = m.fmt.gemv_int().expect("the plan takes the integer GEMV only for its formats");
        let [wq, wsc] = <[Buf; 2]>::try_from(m.decode_bufs()).unwrap();
        let (wq, wsc) = (&wq, &wsc);
        if inline {
            debug_assert_eq!(t, 1);
            let rw = self.tuning.gemvi_rows;
            let kernel =
                self.int_kernel(name::gemv_int(planes, 1, rw, glu, true, f.name), &[false, true, true, true], || {
                    gemv_int::source(f, 1, planes, rw, glu, true)
                });
            let groups = grid(n.div_ceil(rw * if glu { 4 } else { 8 }) as u32);
            self.dispatch_matmul(
                &kernel,
                &[y.buf(), x.buf(), wq, wsc],
                &[n as u32, k as u32, 1, 0],
                groups,
                (if glu { 2 * n } else { n }, k),
            );
            self.account(2.0 * (n * k) as f64 * if glu { 2.0 } else { 1.0 });
            return;
        }
        let xq = self.tensor(t, planes * k / 4);
        let xs = self.tensor(t, k / 8);
        let xquant =
            self.int_kernel(name::xquant_rows(planes), &[false, true, false], || gemv_int::xquant_source(planes));
        let lanes = t * (k / 32);
        self.dispatch(
            &xquant,
            &[xq.buf(), x.buf(), xs.buf()],
            &[k as u32, t as u32, f.xmode, 0],
            grid(lanes.div_ceil(64) as u32),
        );
        // Up to 16 rows per dispatch (each reads the weights once).
        let parts = t.div_ceil(GEMVM_MAX_T);
        let mut r0 = 0;
        for i in 0..parts {
            let rows = (t - r0).div_ceil(parts - i);
            let rw = self.tuning.gemvi_rows;
            let kernel = self.int_kernel(
                name::gemv_int(planes, rows, rw, glu, false, f.name),
                &[false, true, true, true, true],
                || gemv_int::source(f, rows, planes, rw, glu, false),
            );
            let groups = grid(n.div_ceil(rw * if glu { 4 } else { 8 }) as u32);
            let params = [n as u32, k as u32, rows as u32, r0 as u32];
            self.dispatch_matmul(
                &kernel,
                &[y.buf(), xq.buf(), wq, wsc, xs.buf()],
                &params,
                groups,
                (if glu { 2 * n } else { n }, k),
            );
            r0 += rows;
        }
        self.account(2.0 * (t * n * k) as f64 * if glu { 2.0 } else { 1.0 });
    }

    /// `y = x W^T` for up to 16 rows in f32 (`gemv_f32`), or `swiglu` of it with `glu`
    /// (`n` is then the intermediate size).
    fn gemv_f32(&mut self, x: &Tensor, y: &Tensor, m: &crate::Matrix, n: usize, glu: bool) {
        let (t, k) = (x.rows, x.cols);
        let (rw, wgmem) = (self.tuning.gemv_f32_rows, self.caps.max_workgroup_storage);
        let kernel = self.int_kernel(name::gemv_f32(t, rw, glu, m.fmt), crate::kernels::decode_layout(m.fmt), || {
            crate::kernels::gemv_f32_source(m.fmt, t, rw, glu, wgmem)
        });
        let mut bufs = vec![y.buf().clone(), x.buf().clone()];
        bufs.extend(m.decode_bufs());
        let refs: Vec<&Buf> = bufs.iter().collect();
        let groups = grid(n.div_ceil(rw * if glu { 4 } else { 8 }) as u32);
        self.dispatch_matmul(
            &kernel,
            &refs,
            &[n as u32, k as u32, t as u32, 0],
            groups,
            (if glu { 2 * n } else { n }, k),
        );
        self.account(2.0 * (t * n * k) as f64 * if glu { 2.0 } else { 1.0 });
    }

    pub(crate) fn swiglu(&mut self, x: &Tensor) -> Tensor {
        let inter = x.cols / 2;
        let y = self.tensor(x.rows, inter);
        let total = x.rows * inter;
        self.dispatch("swiglu", &[y.buf(), x.buf()], &[inter as u32, total as u32], grid(total.div_ceil(256) as u32));
        y
    }
}
