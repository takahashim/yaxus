//! Matrix products: the f32 kernels, the integer GEMVs and GEMM, and the fused GLU.

use super::grid;
use crate::kernels::gemvi_name;
use crate::plan::{GluPlan, MatmulCase, MatmulPlan};
use crate::{GEMV_MAX_T, GEMVM_MAX_T, GemmKind, Gemv, Gpu, Tensor, Weight, gemm_int, gemv_int};

impl Gpu {
    /// `x · w^T`; passes of up to 16 rows (and more where the integer GEMVs win) take
    /// the kernels `gemv` says.
    pub(crate) fn matmul(&mut self, x: &Tensor, w: &Weight, gemv: Gemv) -> Tensor {
        let m = w.matrix();
        let (n, k, suffix) = (m.n, m.k, m.fmt.suffix());
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
        let plan = case.plan(&self.caps, &self.tuning);
        if let MatmulPlan::GemvInt { planes } = plan {
            let [wq, wsc] = <[wgpu::Buffer; 2]>::try_from(m.decode_bufs()).unwrap();
            self.gemv_int(x, &y, &wq, &wsc, n, m.fmt.gemv_int().unwrap(), planes, false);
            return y;
        }
        if let MatmulPlan::GemmInt { planes } = plan {
            // The integer GEMM reads the per-16 scale arrays where a format has them.
            let [wq, scales] = <[wgpu::Buffer; 2]>::try_from(m.prefill_bufs()).unwrap();
            self.gemm_int(x, &y, &wq, &scales, n, m.fmt.gemm_int().unwrap(), planes);
            return y;
        }
        // The f32 kernels: the GEMVs read the decode layout, the GEMMs the per-16 scales
        // (see `W_Q6`).
        let mut all = vec![y.buf().clone(), x.buf().clone()];
        all.extend(if t <= GEMVM_MAX_T || matches!(plan, MatmulPlan::GemvMulti { .. }) {
            m.decode_bufs()
        } else {
            m.prefill_bufs()
        });
        let refs: Vec<&wgpu::Buffer> = all.iter().collect();
        let (n32, k32) = (n as u32, k as u32);
        let dispatches: Vec<(String, [u32; 4], (u32, u32))> = match plan {
            MatmulPlan::GemvMulti { first } => [(0, first), (first, t - first)]
                .into_iter()
                .filter(|&(_, rows)| rows > 0)
                .map(|(r0, rows)| {
                    let groups = grid(n.div_ceil(8 * self.tuning.gemvm_rows) as u32);
                    (format!("gemvm{rows}_{suffix}"), [n32, k32, rows as u32, r0 as u32], groups)
                })
                .collect(),
            MatmulPlan::Gemv1 => vec![(format!("gemv1_{suffix}"), [n32, k32, 1, 0], grid(n.div_ceil(8) as u32))],
            MatmulPlan::Gemv8 => {
                vec![(format!("gemv8_{suffix}"), [n32, k32, t as u32, 0], grid(n.div_ceil(8) as u32))]
            }
            MatmulPlan::GemmCoop => {
                vec![(format!("gemmcoop_{suffix}"), [n32, k32, t as u32, 0], ((n / 64) as u32, t.div_ceil(64) as u32))]
            }
            MatmulPlan::GemmF32 => vec![(
                format!("gemm2f32_{suffix}"),
                [n32, k32, t as u32, 0],
                (n.div_ceil(128) as u32, t.div_ceil(64) as u32),
            )],
            MatmulPlan::GemvInt { .. } | MatmulPlan::GemmInt { .. } => unreachable!(),
        };
        for (name, params, groups) in dispatches {
            if self.kernels_profile.is_some() {
                self.prof_detail = Some(format!("{n}x{k}"));
            }
            self.dispatch(&name, &refs, &params, groups);
        }
        self.account(2.0 * (t * n * k) as f64);
        y
    }

    pub(crate) fn matmul_glu(&mut self, x: &Tensor, w: &Weight, gemv: Gemv) -> Tensor {
        let m = w.matrix();
        let (n2, k) = (m.n, m.k);
        assert_eq!(x.cols, k);
        let case = MatmulCase { t: x.rows, n: n2, k, fmt: m.fmt, gemm: self.gemm, gemv };
        match case.plan_glu(&self.caps, &self.tuning) {
            GluPlan::GemvInt { planes } => {
                let y = self.tensor(x.rows, n2 / 2);
                let [q, s] = <[wgpu::Buffer; 2]>::try_from(m.decode_bufs()).unwrap();
                self.gemv_int(x, &y, &q, &s, n2 / 2, m.fmt.gemv_int().unwrap(), planes, true);
                return y;
            }
            GluPlan::MatmulThenSwiglu => {
                let gu = self.matmul(x, w, gemv);
                return self.swiglu(&gu);
            }
            GluPlan::Glu1 => {}
        }
        // (for Q4 the block variant `GLU_32` measured slower: 174 vs 138 us on 12288x1536)
        let name = format!("glu_{}", m.fmt.suffix());
        let bufs = m.decode_bufs();
        let inter = n2 / 2;
        let y = self.tensor(1, inter);
        let mut all = vec![y.buf().clone(), x.buf().clone()];
        all.extend(bufs);
        let refs: Vec<&wgpu::Buffer> = all.iter().collect();
        if self.kernels_profile.is_some() {
            self.prof_detail = Some(format!("{n2}x{k}"));
        }
        self.dispatch(&name, &refs, &[inter as u32, k as u32, 1], grid(inter.div_ceil(4) as u32));
        self.account(2.0 * (n2 * k) as f64);
        y
    }

    /// `y = x W^T` for more than 16 rows with integer dot products (`gemm_int`): the rows
    /// are quantized to int8 planes, then tiles of 128 tokens and one of 64 or 32 for the
    /// rest (a tile costs its full size whatever it holds).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm_int(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        wq: &wgpu::Buffer,
        scales: &wgpu::Buffer,
        n: usize,
        f: &'static gemm_int::IntFmt,
        planes: usize,
    ) {
        let (t, k) = (x.rows, x.cols);
        let tp = t.div_ceil(128) * 128;
        let xq = self.tensor(tp, planes * k / 4);
        let xs = self.tensor(tp, k / 8);
        let xquant =
            self.int_kernel(format!("xquant{planes}"), &[false, true, false], || gemm_int::xquant_source(planes));
        let lanes = (tp / 4) * (k / 32);
        self.dispatch(
            xquant,
            &[xq.buf(), x.buf(), xs.buf()],
            &[k as u32, t as u32, f.xmode, 0],
            grid(lanes.div_ceil(64) as u32),
        );
        let bufs = [y.buf().clone(), xq.buf().clone(), wq.clone(), scales.clone(), xs.buf().clone()];
        let refs: Vec<&wgpu::Buffer> = bufs.iter().collect();
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
            let name =
                self.int_kernel(format!("gemmint{planes}_{tn}_{}", f.name), &[false, true, true, true, true], || {
                    gemm_int::source(f, tn, planes)
                });
            if self.kernels_profile.is_some() {
                self.prof_detail = Some(format!("{n}x{k}"));
            }
            self.dispatch(
                name,
                &refs,
                &[n as u32, k as u32, t as u32, r0 as u32],
                (n.div_ceil(64) as u32, tiles as u32),
            );
        }
        self.account(2.0 * (t * n * k) as f64);
    }

    /// `y = x W^T` for up to 16 rows with integer dot products (`gemv_int`), or
    /// `swiglu` of it with `glu` (`n` is then the intermediate size).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemv_int(
        &mut self,
        x: &Tensor,
        y: &Tensor,
        wq: &wgpu::Buffer,
        wsc: &wgpu::Buffer,
        n: usize,
        f: &'static gemv_int::IntFmt,
        planes: usize,
        glu: bool,
    ) {
        let (t, k) = (x.rows, x.cols);
        if t == 1 {
            // A single row is rounded inside the kernel (no dispatch of its own).
            let rw = self.tuning.gemvi_rows;
            let name =
                self.int_kernel(gemvi_name(planes, 1, rw, glu, true, f.name), &[false, true, true, true], || {
                    gemv_int::source(f, 1, planes, rw, glu, true)
                });
            if self.kernels_profile.is_some() {
                self.prof_detail = Some(format!("{}x{k}", if glu { 2 * n } else { n }));
            }
            let groups = grid(n.div_ceil(rw * if glu { 4 } else { 8 }) as u32);
            self.dispatch(name, &[y.buf(), x.buf(), wq, wsc], &[n as u32, k as u32, 1, 0], groups);
            self.account(2.0 * (n * k) as f64 * if glu { 2.0 } else { 1.0 });
            return;
        }
        let xq = self.tensor(t, planes * k / 4);
        let xs = self.tensor(t, k / 8);
        let xquant =
            self.int_kernel(format!("xquantr{planes}"), &[false, true, false], || gemv_int::xquant_source(planes));
        let lanes = t * (k / 32);
        self.dispatch(
            xquant,
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
            let name = self.int_kernel(
                gemvi_name(planes, rows, rw, glu, false, f.name),
                &[false, true, true, true, true],
                || gemv_int::source(f, rows, planes, rw, glu, false),
            );
            if self.kernels_profile.is_some() {
                self.prof_detail = Some(format!("{}x{k}", if glu { 2 * n } else { n }));
            }
            let groups = grid(n.div_ceil(rw * if glu { 4 } else { 8 }) as u32);
            let params = [n as u32, k as u32, rows as u32, r0 as u32];
            self.dispatch(name, &[y.buf(), xq.buf(), wq, wsc, xs.buf()], &params, groups);
            r0 += rows;
        }
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
