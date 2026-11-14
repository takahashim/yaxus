//! Hooks for the integration tests (feature `test-hooks`): single ops on given data,
//! through the kernels the model uses.

use anyhow::Result;
use yaxus_model::ggml::GgmlType;
use yaxus_model::{PassMeta, Quant};

use crate::{Format, GemmKind, Gpu, GpuBackend, Tensor, Weight};

impl Gpu {
    /// A tensor of `rows` rows holding `data`.
    fn tensor_of(&mut self, rows: usize, data: &[f32]) -> Tensor {
        let t = self.tensor(rows, data.len() / rows);
        self.rec.dev.write(t.buf(), 0, bytemuck::cast_slice(data));
        t
    }
}

impl GpuBackend {
    /// For tests: the attention of passes with more than 16 rows uses the tiled kernel,
    /// splitting keys until about `target_wgs` workgroups run (default 384), or the
    /// per-row kernel when `target_wgs` is 0.
    pub fn set_tiled_attention(&self, target_wgs: usize) {
        self.g().tuning.tile_wgs = target_wgs;
    }

    /// Whether the small per-row kernels run in MSL (`AppleTuning::msl_small`).
    pub fn set_msl_small(&self, on: bool) {
        self.g().tuning.apple.msl_small = on;
    }

    /// Whether the prefill GEMM rounds its operands (to f16 or to integers).
    #[doc(hidden)]
    pub fn gemm_uses_f16(&self) -> bool {
        self.g().gemm != GemmKind::F32
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
        let mut g = self.g();
        assert_eq!((x.len(), delta.len()), (rows * cols, rows * cols));
        let mut xt = g.tensor_of(rows, x);
        let dt = g.tensor_of(rows, delta);
        let w1 = Weight::Vector(g.rec.dev.upload("vector", bytemuck::cast_slice(w1)));
        let w2 = Weight::Vector(g.rec.dev.upload("vector", bytemuck::cast_slice(w2)));
        let h = g.add_norm(&mut xt, &dt, &w1, o1, &w2, o2, eps);
        (g.to_host(&xt), g.to_host(&h))
    }

    /// Test hook: like `debug_matmul` for a matrix given as raw GGML blocks, kept in its
    /// GPU layout (Q8_0, Q4_0, Q2_K, Q3_K, Q4_K, Q6_K).
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn debug_matmul_ggml(
        &mut self,
        x: &[f32],
        t: usize,
        dtype: GgmlType,
        bytes: &[u8],
        n: usize,
        k: usize,
        glu: bool,
    ) -> Vec<f32> {
        let gemv = self.gemv;
        let mut g = self.g();
        let w = Weight::Matrix(g.load_ggml(dtype, bytes, n, k).expect("a type with a GPU layout"));
        let xt = g.tensor_of(t, x);
        let y = if glu { g.matmul_glu(&xt, &w, gemv) } else { g.matmul(&xt, &w, gemv) };
        g.to_host(&y)
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
    ) -> Vec<f32> {
        let gemv = self.gemv;
        let mut g = self.g();
        let w = Weight::Matrix(g.load_matrix(w, n, k, quant).unwrap());
        let xt = g.tensor_of(t, x);
        let y = if glu { g.matmul_glu(&xt, &w, gemv) } else { g.matmul(&xt, &w, gemv) };
        g.to_host(&y)
    }

    #[doc(hidden)]
    pub fn debug_argmax(&mut self, logits: &[f32], rows: usize) -> Vec<u32> {
        let mut g = self.g();
        let x = g.tensor_of(rows, logits);
        g.argmax_rows(&x)
    }

    #[doc(hidden)]
    pub fn debug_alloc(&mut self, size: u64) -> Result<()> {
        self.g().checked("allocating a test buffer", |g| {
            let _buf = g.rec.dev.storage("test", size);
            Ok(())
        })
    }

    /// Test hook: embedding rows `ids` from a `[V, h]` table stored as Q4_K (raw GGUF
    /// blocks) or bf16.
    #[doc(hidden)]
    pub fn debug_embed(&mut self, bytes: &[u8], dtype: GgmlType, h: usize, ids: &[u32]) -> Vec<f32> {
        let mut g = self.g();
        let table = match dtype {
            GgmlType::Q4K => {
                let q = yaxus_model::ggml::q4k_from_q4_k(bytes);
                Weight::Embedding { table: g.rec.dev.upload("embedding-q4k", &q.data), h, fmt: Format::Q4K }
            }
            GgmlType::Bf16 => Weight::Embedding { table: g.rec.dev.upload("embedding", bytes), h, fmt: Format::Bf16 },
            _ => panic!("debug_embed: Q4K or bf16 tables only"),
        };
        let pass = PassMeta {
            ids: ids.to_vec(),
            pos: vec![0; ids.len()],
            kv_slot: vec![0; ids.len()],
            pt_off: vec![0; ids.len()],
            pages: Vec::new(),
        };
        g.begin_pass(&pass);
        let y = g.embed(&table);
        g.to_host(&y)
    }

    #[doc(hidden)]
    pub fn max_buffer_size(&self) -> u64 {
        self.g().rec.dev.max_buffer()
    }
}
