//! Hooks for the integration tests (feature `test-hooks`): single ops on given data,
//! through the kernels the model uses.

use anyhow::Result;
#[cfg(not(target_arch = "wasm32"))]
use yaxus_model::Quant;
use yaxus_model::{Config, LayerMeta};

#[cfg(target_arch = "wasm32")]
use crate::{Kv, Tensor};
use crate::{GemmKind, Gemv, Gpu, GpuBackend, Weight};

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
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(not(target_arch = "wasm32"))]
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
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(not(target_arch = "wasm32"))]
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
    #[cfg(not(target_arch = "wasm32"))]
    #[cfg(not(target_arch = "wasm32"))]
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

    // The WebGPU variants of the hooks: the read-back is awaited (see `GpuBackend`).
    #[cfg(target_arch = "wasm32")]
    #[doc(hidden)]
    pub fn debug_upload(&mut self, x: &[f32], rows: usize, cols: usize) -> Tensor {
        assert_eq!(x.len(), rows * cols);
        let t = self.tensor(rows, cols);
        let buf = t.buf().clone();
        self.dev.write(&buf, 0, bytemuck::cast_slice(x));
        t
    }

    #[cfg(target_arch = "wasm32")]
    #[doc(hidden)]
    pub async fn debug_matmul_ggml_async(
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
        let xt = self.debug_upload(x, t, k);
        let y = if glu { self.matmul_glu(&xt, &w, gemv) } else { self.matmul(&xt, &w, gemv) };
        self.to_host_async(&y).await
    }

    #[cfg(target_arch = "wasm32")]
    #[doc(hidden)]
    pub async fn debug_argmax_async(&mut self, logits: &[f32], rows: usize) -> Vec<u32> {
        let x = self.debug_upload(logits, rows, logits.len() / rows);
        self.argmax_rows_async(&x).await
    }

    /// The split attention with the roped q from the host: `[row][head] = (max, sum)`
    /// per chunk; the KV must already hold the keys.
    #[cfg(target_arch = "wasm32")]
    #[doc(hidden)]
    pub async fn debug_attn_split_async(
        &mut self,
        q: &[f32],
        rows: usize,
        hq: usize,
        hkv: usize,
        kv: &Kv,
        layer_index: usize,
    ) -> Vec<f32> {
        let (kc, vc) = (kv.k[layer_index].clone(), kv.v[layer_index].clone());
        let qt = self.debug_upload(q, rows, hq * 128);
        let case = crate::plan::AttentionCase {
            rows,
            hq,
            hkv,
            window: None,
            max_pos: self.pass_max_pos,
            tiles: self.pass_tiles,
        };
        let crate::plan::AttentionPlan::Split { g: gp, hg, ch, nch } = case.plan(&self.caps, &self.tuning) else {
            panic!("the plan does not take the split kernel");
        };
        let t = rows;
        let part = self.tensor(t * hkv * nch * gp, crate::HEAD_DIM);
        let ml = self.tensor(t * hkv * nch * gp, 2);
        let (rows_buf, pages_buf) = (self.meta_buf(crate::MetaBuf::Rows), self.meta_buf(crate::MetaBuf::Pages));
        self.dispatch(
            &crate::kernels::name::attn_split(gp, ch, hg),
            &[part.buf(), ml.buf(), qt.buf(), &kc, &vc, &rows_buf, &pages_buf],
            &[hq as u32, hkv as u32, 0, (1.0 / 128.0f32.sqrt()).to_bits(), nch as u32],
            ((hkv * (gp / hg) * nch) as u32, t as u32),
        );
        self.to_host_async(&ml).await
    }

    /// Just the `qk_rope` kernel of `attention`, then the q, k and v read back:
    /// `[q of row 0][q of row 1]…[k of row 0]…[v of row 0]…`.
    #[cfg(target_arch = "wasm32")]
    #[doc(hidden)]
    pub async fn debug_qk_rope_async(
        &mut self,
        qkv: &[f32],
        q_norm: &Weight,
        k_norm: &Weight,
        kv: &mut Kv,
        cfg: &Config,
        layer: &LayerMeta,
    ) -> Vec<f32> {
        let t = self.pass_rows;
        let (hq, kvd) = (cfg.num_attention_heads, kv.kv_dim);
        let qkvt = self.debug_upload(qkv, t, hq * 128 + 2 * kvd);
        let q = self.qk_rope(&qkvt, q_norm, k_norm, kv, cfg, layer);
        let (kc, vc) = (kv.k[layer.index].clone(), kv.v[layer.index].clone());
        let mut out = self.read_async(q.buf(), 0, (t * hq * 128 * 4) as u64).await;
        let bytes = t * kvd * 4;
        out.extend(self.read_async(&kc, 0, bytes as u64).await);
        out.extend(self.read_async(&vc, 0, bytes as u64).await);
        bytemuck::cast_slice(&out).to_vec()
    }

    #[doc(hidden)]
    #[cfg(not(target_arch = "wasm32"))]
    pub fn debug_argmax(&mut self, logits: &[f32], rows: usize) -> Vec<u32> {
        let x = self.tensor(rows, logits.len() / rows);
        self.dev.write(x.buf(), 0, bytemuck::cast_slice(logits));
        self.argmax_rows(&x)
    }

    /// qk_rope and the split kernel in one submission (as `attention` runs them):
    /// reads back the ml partials and the roped q afterwards, to compare with the
    /// host-roped-q version of `debug_attn_split_async`.
    #[cfg(target_arch = "wasm32")]
    #[doc(hidden)]
    pub async fn debug_attn_chain_async(
        &mut self,
        qkv: &[f32],
        q_norm: &Weight,
        k_norm: &Weight,
        kv: &mut Kv,
        cfg: &Config,
        layer: &LayerMeta,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let t = self.pass_rows;
        let (hq, hkv, kvd) = (cfg.num_attention_heads, cfg.num_key_value_heads, kv.kv_dim);
        let qkvt = self.debug_upload(qkv, t, hq * 128 + 2 * kvd);
        let q = self.qk_rope(&qkvt, q_norm, k_norm, kv, cfg, layer);
        let (kc, vc) = (kv.k[layer.index].clone(), kv.v[layer.index].clone());
        let case = crate::plan::AttentionCase {
            rows: t,
            hq,
            hkv,
            window: None,
            max_pos: self.pass_max_pos,
            tiles: self.pass_tiles,
        };
        let crate::plan::AttentionPlan::Split { g: gp, hg, ch, nch } = case.plan(&self.caps, &self.tuning) else {
            panic!("the plan does not take the split kernel");
        };
        let part = self.tensor(t * hkv * nch * gp, crate::HEAD_DIM);
        let ml = self.tensor(t * hkv * nch * gp, 2);
        self.attn_split(&q, &kc, &vc, &part, &ml, hq, hkv, gp, hg, ch, nch, 0, 1.0 / 128.0f32.sqrt(), t);
        let o = self.tensor(t, hq * 128);
        self.attn_combine(&o, &part, &ml, hq, hkv, nch, t);
        let mut out = self.read_async(ml.buf(), 0, (ml.rows * 2 * 4) as u64).await;
        out.extend(self.read_async(part.buf(), 0, (part.rows * 128 * 4) as u64).await);
        out.extend(self.read_async(o.buf(), 0, (t * hq * 128 * 4) as u64).await);
        out.extend(self.read_async(q.buf(), 0, (t * hq * 128 * 4) as u64).await);
        let vals: Vec<f32> = bytemuck::cast_slice(&out).to_vec();
        let mut at = ml.rows * 2;
        let ml_v = vals[..at].to_vec();
        let part_n = part.rows * 128;
        let part_v = vals[at..at + part_n].to_vec();
        at += part_n;
        let o_v = vals[at..at + t * hq * 128].to_vec();
        at += t * hq * 128;
        let q_v = vals[at..].to_vec();
        (ml_v, part_v, o_v, q_v)
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
        self.g().gemm_uses_f16()
    }

    /// Test hook, see [`Gpu::debug_add_norm`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    #[cfg(not(target_arch = "wasm32"))]
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
    #[cfg(not(target_arch = "wasm32"))]
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
    #[cfg(not(target_arch = "wasm32"))]
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
    #[cfg(not(target_arch = "wasm32"))]
    pub fn debug_argmax(&mut self, logits: &[f32], rows: usize) -> Vec<u32> {
        self.g().debug_argmax(logits, rows)
    }

    #[doc(hidden)]
    pub fn debug_alloc(&mut self, size: u64) -> Result<()> {
        self.g().debug_alloc(size)
    }

    // The WebGPU variants: the read-back is awaited, one per call.
    #[cfg(target_arch = "wasm32")]
    #[doc(hidden)]
    pub fn debug_upload(&mut self, x: &[f32], rows: usize, cols: usize) -> Tensor {
        self.g().debug_upload(x, rows, cols)
    }

    #[cfg(target_arch = "wasm32")]
    #[doc(hidden)]
    pub async fn debug_matmul_ggml_async(
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
        self.gpu.lock().unwrap().debug_matmul_ggml_async(x, t, dtype, bytes, n, k, glu, gemv).await
    }

    #[cfg(target_arch = "wasm32")]
    #[doc(hidden)]
    pub async fn debug_argmax_async(&mut self, logits: &[f32], rows: usize) -> Vec<u32> {
        self.gpu.lock().unwrap().debug_argmax_async(logits, rows).await
    }

    #[cfg(target_arch = "wasm32")]
    #[doc(hidden)]
    pub async fn debug_qk_rope_async(
        &mut self,
        qkv: &[f32],
        q_norm: &Weight,
        k_norm: &Weight,
        kv: &mut Kv,
        cfg: &Config,
        layer: &LayerMeta,
    ) -> Vec<f32> {
        self.gpu.lock().unwrap().debug_qk_rope_async(qkv, q_norm, k_norm, kv, cfg, layer).await
    }

    /// qk_rope and the split kernel in one submission (as `attention` runs them):
    /// reads back the ml partials and the roped q afterwards, to compare with the
    /// host-roped-q version of `debug_attn_split_async`.
    #[cfg(target_arch = "wasm32")]
    #[doc(hidden)]
    pub async fn debug_attn_split_async(
        &mut self,
        q: &[f32],
        rows: usize,
        hq: usize,
        hkv: usize,
        kv: &Kv,
        layer_index: usize,
    ) -> Vec<f32> {
        self.gpu.lock().unwrap().debug_attn_split_async(q, rows, hq, hkv, kv, layer_index).await
    }

    #[cfg(target_arch = "wasm32")]
    #[doc(hidden)]
    pub async fn debug_attn_chain_async(
        &mut self,
        qkv: &[f32],
        q_norm: &Weight,
        k_norm: &Weight,
        kv: &mut Kv,
        cfg: &Config,
        layer: &LayerMeta,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        self.gpu.lock().unwrap().debug_attn_chain_async(qkv, q_norm, k_norm, kv, cfg, layer).await
    }

    #[doc(hidden)]
    pub fn max_buffer_size(&self) -> u64 {
        self.g().dev.max_buffer()
    }
}
