//! Hooks for the integration tests (feature `test-hooks`): single ops on given data,
//! through the kernels the model uses.

use anyhow::Result;
use yaxus_model::Quant;

use crate::{GemmKind, Gemv, Gpu, Weight, WgpuBackend};

impl Gpu {
    /// Whether the prefill GEMM rounds its operands (to f16 or to integers).
    #[doc(hidden)]
    pub fn gemm_uses_f16(&self) -> bool {
        self.gemm != GemmKind::F32
    }

    /// Test hook: `add_norm` of `x` and `delta` (`[rows, cols]` each) with the norm weights
    /// `w1` (post, offset `o1`) and `w2` (pre, offset `o2`). Returns `(x, h)` afterwards.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn debug_add_norm(
        &mut self,
        x: &[f32],
        delta: &[f32],
        rows: usize,
        cols: usize,
        (w1, o1): (&[f32], f32),
        (w2, o2): (&[f32], f32),
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut xt = self.tensor(rows, cols);
        self.dev.write(xt.buf(), 0, bytemuck::cast_slice(x));
        let dt = self.tensor(rows, cols);
        self.dev.write(dt.buf(), 0, bytemuck::cast_slice(delta));
        let w1 = Weight::Vector(self.upload("vector", bytemuck::cast_slice(w1)));
        let w2 = Weight::Vector(self.upload("vector", bytemuck::cast_slice(w2)));
        let h = self.add_norm(&mut xt, &dt, &w1, o1, &w2, o2, eps);
        (self.to_host(&xt), self.to_host(&h))
    }

    /// Test hook: like `debug_matmul` for a matrix given as raw GGML blocks, kept in its
    /// GPU layout (Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q6_K).
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn debug_matmul_ggml(
        &mut self,
        x: &[f32],
        t: usize,
        dtype: yaxus_model::ggml::GgmlType,
        bytes: &[u8],
        n: usize,
        k: usize,
        glu: bool,
        gemv: Gemv,
    ) -> Vec<f32> {
        let w = Weight::Matrix(self.load_ggml(dtype, bytes, n, k).expect("a type with a GPU layout"));
        let xt = self.tensor(t, k);
        self.dev.write(xt.buf(), 0, bytemuck::cast_slice(x));
        let y = if glu { self.matmul_glu(&xt, &w, gemv) } else { self.matmul(&xt, &w, gemv) };
        self.to_host(&y)
    }

    /// Test hook: `x [t, k] · W^T` (or `swiglu` of it with `glu`) for a bf16 matrix `W [n, k]`
    /// stored as `quant`, through the same kernels the model uses.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn debug_matmul(
        &mut self,
        x: &[f32],
        t: usize,
        w: &[u16],
        n: usize,
        k: usize,
        quant: Quant,
        glu: bool,
        gemv: Gemv,
    ) -> Vec<f32> {
        let w = Weight::Matrix(self.load_matrix(w, n, k, quant).unwrap());
        let xt = self.tensor(t, k);
        self.dev.write(xt.buf(), 0, bytemuck::cast_slice(x));
        let y = if glu { self.matmul_glu(&xt, &w, gemv) } else { self.matmul(&xt, &w, gemv) };
        self.to_host(&y)
    }

    #[doc(hidden)]
    pub fn debug_alloc(&mut self, size: u64) -> Result<()> {
        self.checked("allocating a test buffer", |g| {
            let _buf = g.dev.storage("test", size);
            Ok(())
        })
    }

    #[doc(hidden)]
    pub fn debug_argmax(&mut self, logits: &[f32], rows: usize) -> Vec<u32> {
        let x = self.tensor(rows, logits.len() / rows);
        self.dev.write(x.buf(), 0, bytemuck::cast_slice(logits));
        self.argmax_rows(&x)
    }
}

impl WgpuBackend {
    /// For tests: the attention of passes with more than 16 rows uses the tiled kernel,
    /// splitting keys until about `target_wgs` workgroups run (default 384), or the
    /// per-row kernel when `target_wgs` is 0.
    pub fn set_tiled_attention(&self, target_wgs: usize) {
        self.g().tuning.tile_wgs = target_wgs;
    }

    /// Whether the small per-row kernels run in MSL (`Tuning::msl_small`).
    pub fn set_msl_small(&self, on: bool) {
        self.g().tuning.msl_small = on;
    }

    /// Whether the prefill GEMM rounds its operands (to f16 or to integers).
    #[doc(hidden)]
    pub fn gemm_uses_f16(&self) -> bool {
        self.g().gemm_uses_f16()
    }

    /// Test hook, see [`Gpu::debug_add_norm`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn debug_add_norm(
        &mut self,
        x: &[f32],
        delta: &[f32],
        rows: usize,
        cols: usize,
        w1: (&[f32], f32),
        w2: (&[f32], f32),
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        self.g().debug_add_norm(x, delta, rows, cols, w1, w2, eps)
    }

    /// Test hook, see [`Gpu::debug_matmul_ggml`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn debug_matmul_ggml(
        &mut self,
        x: &[f32],
        t: usize,
        dtype: yaxus_model::ggml::GgmlType,
        bytes: &[u8],
        n: usize,
        k: usize,
        glu: bool,
    ) -> Vec<f32> {
        let gemv = self.gemv;
        self.g().debug_matmul_ggml(x, t, dtype, bytes, n, k, glu, gemv)
    }

    /// Test hook, see [`Gpu::debug_matmul`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn debug_matmul(
        &mut self,
        x: &[f32],
        t: usize,
        w: &[u16],
        n: usize,
        k: usize,
        quant: Quant,
        glu: bool,
    ) -> Vec<f32> {
        let gemv = self.gemv;
        self.g().debug_matmul(x, t, w, n, k, quant, glu, gemv)
    }

    #[doc(hidden)]
    pub fn debug_argmax(&mut self, logits: &[f32], rows: usize) -> Vec<u32> {
        self.g().debug_argmax(logits, rows)
    }

    #[doc(hidden)]
    pub fn debug_alloc(&mut self, size: u64) -> Result<()> {
        self.g().debug_alloc(size)
    }

    #[doc(hidden)]
    pub fn max_buffer_size(&self) -> u64 {
        self.g().dev.max_buffer()
    }
}
