//! Matrix products: [`steps`] says which dispatches one takes, and `record` records
//! them.

use crate::commands::grid;
use crate::dev::Buf;
use crate::kernels::KernelId;
use crate::plan::MatmulCase;
use crate::steps::{self, Binds, MatmulSteps};
use crate::{Gemv, Gpu, Matrix, Tensor, Weight};

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
            g.rec.dev.write(xt.buf(), 0, bytemuck::cast_slice(&x[r0 * k..(r0 + rows) * k]));
            let case = MatmulCase { t: rows, n, k, fmt: m.fmt, gemm: g.gemm, gemv: Gemv::F32 };
            let pieces = crate::plan::gemv2_pieces(0, rows, false, &g.tuning.apple);
            let y = g.record(&xt, &m, &steps::of_pieces(&case, &pieces));
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
        assert_eq!(x.cols, m.k, "matmul inner dim");
        let steps = steps::matmul(&self.case(x, m, gemv), &self.caps, &self.tuning);
        self.record(x, m, &steps)
    }

    /// `swiglu(x · w^T)` of a fused gate/up matrix.
    pub(crate) fn matmul_glu(&mut self, x: &Tensor, w: &Weight, gemv: Gemv) -> Tensor {
        let m = w.matrix();
        assert_eq!(x.cols, m.k, "matmul inner dim");
        match steps::glu(&self.case(x, m, gemv), &self.caps, &self.tuning) {
            Some(steps) => self.record(x, m, &steps),
            None => {
                let gu = self.matmul(x, w, gemv);
                self.swiglu(&gu)
            }
        }
    }

    fn case(&self, x: &Tensor, m: &Matrix, gemv: Gemv) -> MatmulCase {
        MatmulCase { t: x.rows, n: m.n, k: m.k, fmt: m.fmt, gemm: self.gemm, gemv }
    }

    /// Records `steps` on `x` and `m`, into a new output.
    fn record(&mut self, x: &Tensor, m: &Matrix, steps: &MatmulSteps) -> Tensor {
        let y = self.tensor_padded(steps.rows, steps.cols, steps.out_rows);
        let planes = steps.quantize.map(|q| {
            let xq = self.tensor(q.rows, q.planes * m.k / 4);
            let xs = self.tensor(q.rows, m.k / 8);
            self.dispatch(q.kernel, &[xq.buf(), x.buf(), xs.buf()], &q.imm, q.groups);
            (xq, xs)
        });
        for d in &steps.dispatches {
            let bufs: Vec<&Buf> = match (d.binds, &planes) {
                (Binds::Rows(layout), _) => [y.buf(), x.buf()].into_iter().chain(m.bufs(layout)).collect(),
                (Binds::Planes(layout), Some((xq, xs))) => {
                    [y.buf(), xq.buf()].into_iter().chain(m.bufs(layout)).chain([xs.buf()]).collect()
                }
                (Binds::Planes(_), None) => unreachable!("the steps of integer planes quantize first"),
            };
            self.dispatch_matmul(d.kernel, &bufs, &d.imm, d.groups, (m.n, m.k));
        }
        self.rec.account(steps.flops);
        y
    }

    pub(crate) fn swiglu(&mut self, x: &Tensor) -> Tensor {
        let inter = x.cols / 2;
        let y = self.tensor(x.rows, inter);
        let total = x.rows * inter;
        self.dispatch(
            KernelId::Swiglu,
            &[y.buf(), x.buf()],
            &[inter as u32, total as u32],
            grid(total.div_ceil(256) as u32),
        );
        y
    }
}
