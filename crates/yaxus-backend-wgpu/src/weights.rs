//! Weights on the GPU: their formats, and how checkpoint and GGUF tensors are packed
//! into them.

use anyhow::Result;
use yaxus_model::{Quant, TensorRef};

use crate::{Gpu, gemm_int, gemv_int};

/// How a weight matrix is stored on the GPU (and which kernels read it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    /// Blocks of 32 int8 with an f16 scale.
    Q8,
    /// Blocks of 32 nibbles with an f16 scale.
    Q4,
    /// GGUF Q4_K values: nibbles and the per-256 headers, decoded in the kernels.
    Q4K,
    /// 6-bit values (GGUF Q6_K).
    Q6,
    /// 3-bit values (GGUF Q3_K).
    Q3,
    /// 2-bit values (GGUF Q2_K), with mins.
    Q2,
    Bf16,
}

impl Format {
    /// The suffix of the kernels that read it (`gemv1_q8`, ...).
    pub fn suffix(self) -> &'static str {
        match self {
            Format::Q8 => "q8",
            Format::Q4 => "q4",
            Format::Q4K => "q4k",
            Format::Q6 => "q6",
            Format::Q3 => "q3",
            Format::Q2 => "q2",
            Format::Bf16 => "bf16",
        }
    }

    /// The integer GEMV's description of it, if that kernel reads it.
    pub(crate) fn gemv_int(self) -> Option<&'static gemv_int::IntFmt> {
        gemv_int::format(self.suffix())
    }

    /// The integer GEMM's description of it, if that kernel reads it.
    pub(crate) fn gemm_int(self) -> Option<&'static gemm_int::IntFmt> {
        gemm_int::format(self.suffix())
    }
}

/// An `[n, k]` weight matrix on the GPU.
pub struct Matrix {
    pub(crate) fmt: Format,
    pub(crate) n: usize,
    pub(crate) k: usize,
    /// The values (for `Bf16`, the whole matrix).
    pub(crate) q: wgpu::Buffer,
    /// The scales as the decode kernels read them (none for `Bf16`; for `Q4K` the same
    /// buffer as `q`, read through a second binding).
    pub(crate) s: Option<wgpu::Buffer>,
    /// For `Q6`, `Q3`: an f16 scale per 16 values; for `Q2`, a pair of f16 per 16 (the
    /// prefill GEMMs read these instead of `s`).
    pub(crate) s16: Option<wgpu::Buffer>,
}

impl Matrix {
    /// The weight bindings of the decode kernels (GEMVs).
    pub(crate) fn decode_bufs(&self) -> Vec<wgpu::Buffer> {
        std::iter::once(&self.q).chain(&self.s).cloned().collect()
    }

    /// The weight bindings of the prefill GEMMs: the per-16 scales where there are some.
    pub(crate) fn prefill_bufs(&self) -> Vec<wgpu::Buffer> {
        std::iter::once(&self.q).chain(self.s16.as_ref().or(self.s.as_ref())).cloned().collect()
    }
}

pub enum Weight {
    Matrix(Matrix),
    /// 1-D weights (norms), converted to f32.
    Vector(wgpu::Buffer),
    /// Embedding table `[V, H]` in bf16 (two values per u32).
    Embedding {
        table: wgpu::Buffer,
        h: usize,
    },
}

impl Weight {
    pub(crate) fn matrix(&self) -> &Matrix {
        match self {
            Weight::Matrix(m) => m,
            _ => panic!("a matrix product needs a matrix"),
        }
    }
}

/// f16 values added to the scales of each row of a quantized matrix when the row has a
/// multiple of 256 blocks (`sstride` in the shaders). At such a power-of-two stride the
/// 64 rows of a matrix-core GEMM tile read their scales from conflicting addresses,
/// which made the 8B `down_proj` (K = 16384) 1.6-2.2x slower.
pub(crate) const SCALE_PAD: usize = 8;

/// Per-row scales at the padded stride (`sstride` in the shaders), as an even number of
/// u16. A row has `blocks` entries of `unit` u16 each (1 for an f16 scale, 2 for a pair); rows of a multiple of 256 entries get `SCALE_PAD` more.
pub(crate) fn pad_scales(sc: Vec<u16>, n: usize, blocks: usize, unit: usize) -> Vec<u16> {
    let row = blocks * unit;
    let mut out = if !blocks.is_multiple_of(256) {
        sc
    } else {
        let stride = row + SCALE_PAD * unit;
        let mut out = vec![0u16; n * stride];
        for r in 0..n {
            out[r * stride..r * stride + row].copy_from_slice(&sc[r * row..(r + 1) * row]);
        }
        out
    };
    if out.len() % 2 == 1 {
        out.push(0);
    }
    out
}

impl Gpu {
    pub(crate) fn load_embedding(&mut self, t: &TensorRef) -> Result<Weight> {
        let raw;
        let bf16: &[u16] = match t.as_bf16() {
            Some(b) => b,
            None => {
                raw = t.to_bf16_vec();
                &raw
            }
        };
        Ok(Weight::Embedding { table: self.upload("embedding", bytemuck::cast_slice(bf16)), h: t.shape[1] })
    }

    pub(crate) fn load_weight(&mut self, t: &TensorRef, quant: Quant) -> Result<Weight> {
        if t.shape.len() == 1 {
            let v = t.to_f32_vec();
            return Ok(Weight::Vector(self.upload("vector", bytemuck::cast_slice(&v))));
        }
        let (n, k) = (t.shape[0], t.shape[1]);
        use yaxus_model::ggml::GgmlType;
        let low_bit = matches!(t.dtype, GgmlType::Q4_0 | GgmlType::Q4K | GgmlType::Q3K | GgmlType::Q2K);
        if (quant == Quant::Auto || (quant == Quant::Auto4 && low_bit))
            && let Some(m) = self.load_gguf_matrix(t, n, k)?
        {
            return Ok(Weight::Matrix(m));
        }
        let raw;
        let bf16: &[u16] = match t.as_bf16() {
            Some(b) => b,
            None => {
                raw = t.to_bf16_vec();
                &raw
            }
        };
        self.load_matrix(bf16, n, k, quant).map(Weight::Matrix)
    }

    pub(crate) fn load_matrix(&mut self, bf16: &[u16], n: usize, k: usize, quant: Quant) -> Result<Matrix> {
        anyhow::ensure!(k.is_multiple_of(32), "K={k} must be a multiple of 32");
        Ok(match quant {
            Quant::Q8 | Quant::Auto => self.upload_q8(yaxus_model::quant::quantize_q8(bf16, k), n, k),
            Quant::Q4 | Quant::Auto4 => self.upload_q4(yaxus_model::quant::quantize_q4(bf16, k), n, k),
            Quant::Bf16 => {
                let q = self.upload("bf16", bytemuck::cast_slice(bf16));
                Matrix { fmt: Format::Bf16, n, k, q, s: None, s16: None }
            }
        })
    }

    /// A GGUF matrix in a layout the kernels read directly, without changing its values
    /// (up to f16 rounding of Q6_K's combined scales); `None` for other types.
    pub(crate) fn load_gguf_matrix(&mut self, t: &TensorRef, n: usize, k: usize) -> Result<Option<Matrix>> {
        use yaxus_model::ggml::GgmlType;
        // The repacked layouts and the kernels take whole blocks per row: a file with
        // rows that straddle blocks would load without complaint and compute garbage.
        let row_block = match t.dtype {
            GgmlType::Q8_0 | GgmlType::Q4_0 => 32,
            GgmlType::Q2K | GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q6K => 256,
            _ => return Ok(None),
        };
        anyhow::ensure!(
            k.is_multiple_of(row_block),
            "{:?} matrix [{n}, {k}]: rows must be a multiple of {row_block} values",
            t.dtype
        );
        Ok(self.load_ggml(t.dtype, t.bytes(), n, k))
    }

    pub(crate) fn load_ggml(
        &mut self,
        dtype: yaxus_model::ggml::GgmlType,
        b: &[u8],
        n: usize,
        k: usize,
    ) -> Option<Matrix> {
        use yaxus_model::ggml::{self, GgmlType};
        Some(match dtype {
            GgmlType::Q8_0 => self.upload_q8(ggml::q8_from_q8_0(b), n, k),
            GgmlType::Q4_0 => self.upload_q4(ggml::q4_from_q4_0(b), n, k),
            GgmlType::Q4K => {
                // One buffer, read through two bindings (nibbles and headers).
                let qb = self.upload("q4k", &ggml::q4k_from_q4_k(b).data);
                Matrix { fmt: Format::Q4K, n, k, q: qb.clone(), s: Some(qb), s16: None }
            }
            GgmlType::Q6K => {
                let q = ggml::q6_from_q6_k(b, n, k);
                let qb = self.upload("q6", &q.qs);
                let sb = self.upload("q6-meta", &q.meta);
                let s16 = pad_scales(q.scales16.iter().map(|s| s.to_bits()).collect(), n, k / 16, 1);
                let s16 = self.upload("q6-scales", bytemuck::cast_slice(&s16));
                Matrix { fmt: Format::Q6, n, k, q: qb, s: Some(sb), s16: Some(s16) }
            }
            GgmlType::Q3K => {
                let q = ggml::q3_from_q3_k(b, n, k);
                let qb = self.upload("q3", &q.qs);
                let sb = self.upload("q3-meta", &q.meta);
                let s16 = pad_scales(q.scales16.iter().map(|s| s.to_bits()).collect(), n, k / 16, 1);
                let s16 = self.upload("q3-scales", bytemuck::cast_slice(&s16));
                Matrix { fmt: Format::Q3, n, k, q: qb, s: Some(sb), s16: Some(s16) }
            }
            GgmlType::Q2K => {
                let q = ggml::q2_from_q2_k(b, n, k);
                let qb = self.upload("q2", &q.qs);
                let sb = self.upload("q2-meta", &q.meta);
                let s16 = pad_scales(q.pairs16.iter().map(|s| s.to_bits()).collect(), n, k / 16, 2);
                let s16 = self.upload("q2-scales", bytemuck::cast_slice(&s16));
                Matrix { fmt: Format::Q2, n, k, q: qb, s: Some(sb), s16: Some(s16) }
            }
            _ => return None,
        })
    }

    pub(crate) fn upload_q8(&mut self, q: yaxus_model::quant::Q8Data, n: usize, k: usize) -> Matrix {
        let qb = self.upload("q8", bytemuck::cast_slice(&q.qs));
        let sc = pad_scales(q.scales.iter().map(|s| s.to_bits()).collect(), n, k / 32, 1);
        let sb = self.upload("q8-scales", bytemuck::cast_slice(&sc));
        Matrix { fmt: Format::Q8, n, k, q: qb, s: Some(sb), s16: None }
    }

    pub(crate) fn upload_q4(&mut self, q: yaxus_model::quant::Q4Data, n: usize, k: usize) -> Matrix {
        let qb = self.upload("q4", &q.qs);
        let sc = pad_scales(q.scales.iter().map(|s| s.to_bits()).collect(), n, k / 32, 1);
        let sb = self.upload("q4-scales", bytemuck::cast_slice(&sc));
        Matrix { fmt: Format::Q4, n, k, q: qb, s: Some(sb), s16: None }
    }
}
