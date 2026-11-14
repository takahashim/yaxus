//! Weights on the GPU: their formats, how checkpoint and GGUF tensors are packed into
//! them, and what loading them left behind ([`LoadReport`]).

use anyhow::Result;
use yaxus_model::{Quant, TensorRef, Weights};

use crate::dev::Buf;
use crate::steps::Layout;
use crate::{Gpu, GpuBackend, gemm_int, gemv_int};

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
    /// Every format.
    pub(crate) const ALL: [Format; 7] =
        [Format::Q8, Format::Q4, Format::Q4K, Format::Q6, Format::Q3, Format::Q2, Format::Bf16];

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
        gemv_int::format(self)
    }

    /// The integer GEMM's description of it, if that kernel reads it.
    pub(crate) fn gemm_int(self) -> Option<&'static gemm_int::IntFmt> {
        gemm_int::format(self)
    }
}

/// An `[n, k]` weight matrix on the GPU.
pub struct Matrix {
    pub(crate) fmt: Format,
    pub(crate) n: usize,
    pub(crate) k: usize,
    /// The values (for `Bf16`, the whole matrix).
    pub(crate) q: Buf,
    /// The scales as the decode kernels read them (none for `Bf16`; for `Q4K` the same
    /// buffer as `q`, read through a second binding).
    pub(crate) s: Option<Buf>,
    /// For `Q6`, `Q3`: an f16 scale per 16 values; for `Q2`, a pair of f16 per 16 (the
    /// prefill GEMMs read these instead of `s`).
    pub(crate) s16: Option<Buf>,
}

impl Matrix {
    /// The weight bindings of a kernel that reads `layout`: the values, then the scales
    /// (for the prefill GEMMs the per-16 scales where there are some).
    pub(crate) fn bufs(&self, layout: Layout) -> impl Iterator<Item = &Buf> {
        let scales = match layout {
            Layout::Decode => self.s.as_ref(),
            Layout::Prefill => self.s16.as_ref().or(self.s.as_ref()),
        };
        std::iter::once(&self.q).chain(scales)
    }
}

pub enum Weight {
    Matrix(Matrix),
    /// 1-D weights (norms), converted to f32.
    Vector(Buf),
    /// Embedding table `[V, H]`, in bf16 (two values per u32) or as the GGUF Q4_K
    /// blocks; a quantized table is read directly, which saves the 330 MB a bf16 copy
    /// of the 0.8B's 107520 x 1536 table would take.
    Embedding {
        table: Buf,
        h: usize,
        fmt: Format,
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

/// What loading the weights left behind.
#[derive(Default)]
pub(crate) struct LoadReport {
    /// Bytes of weights uploaded.
    pub(crate) weight_bytes: usize,
    /// An uploaded buffer larger than a storage binding may be (see `Gpu::load_checked`).
    oversized: Option<(String, usize)>,
    /// Matrices loaded as Q4 because their own format exceeds the binding limit.
    pub(crate) reduced_to_q4: usize,
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

/// The largest buffer [`Gpu::load_ggml`] uploads for an `[n, k]` matrix of `dtype`, or
/// `None` when the type has no GPU layout. Each buffer is bound on its own, so this is
/// what has to fit the storage binding limit (pinned to the layouts by a test).
pub(crate) fn ggml_largest_buffer(dtype: yaxus_model::ggml::GgmlType, n: usize, k: usize) -> Option<usize> {
    use yaxus_model::ggml::GgmlType;
    use yaxus_model::quant::q6_meta_stride;
    // `pad_scales`: `blocks` entries of `unit` u16 per row, padded, as an even count.
    let scales = |blocks: usize, unit: usize| {
        let row = blocks * unit;
        let stride = if blocks.is_multiple_of(256) { row + SCALE_PAD * unit } else { row };
        (n * stride).next_multiple_of(2) * 2
    };
    Some(match dtype {
        GgmlType::Q8_0 => (n * k).max(scales(k / 32, 1)),
        GgmlType::Q4_0 => (n * k / 2).max(scales(k / 32, 1)),
        GgmlType::Q4K => n * (k / 256) * 144,
        GgmlType::Q6K => (n * 3 * k / 4).max(n * q6_meta_stride(k)).max(scales(k / 16, 1)),
        GgmlType::Q3K => (n * 3 * k / 8).max(n * q6_meta_stride(k)).max(scales(k / 16, 1)),
        GgmlType::Q2K => (n * k / 4).max(n * 20 * (k / 256)).max(scales(k / 16, 2)),
        _ => return None,
    })
}

/// How a matrix is stored, decided from the tensor and the requested `quant` alone, so
/// that loading it ([`Gpu::load_weight`]) and sizing it before the device is open
/// ([`GpuBackend::largest_weight_bytes`]) cannot disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatrixPlan {
    /// The file's GGUF blocks in their GPU layout (`load_ggml`).
    FileLayout,
    /// Converted from bf16 to `Q8`, `Q4` or kept as `Bf16` (`load_matrix`).
    Convert(Quant),
}

impl MatrixPlan {
    /// The plan for matrix `t` with `quant`: `Auto` keeps any GGUF type with a GPU layout,
    /// `Auto4` only those of at most 4 bits; what is not kept becomes Q8 (`Auto`) or Q4
    /// (`Auto4`).
    fn of(t: &TensorRef, quant: Quant) -> Self {
        use yaxus_model::ggml::GgmlType;
        let has_layout = ggml_largest_buffer(t.dtype, 1, 256).is_some();
        let low_bit = matches!(t.dtype, GgmlType::Q4_0 | GgmlType::Q4K | GgmlType::Q3K | GgmlType::Q2K);
        match quant {
            Quant::Auto if has_layout => MatrixPlan::FileLayout,
            Quant::Auto4 if has_layout && low_bit => MatrixPlan::FileLayout,
            Quant::Auto => MatrixPlan::Convert(Quant::Q8),
            Quant::Auto4 => MatrixPlan::Convert(Quant::Q4),
            q => MatrixPlan::Convert(q),
        }
    }

    /// The largest buffer an `[n, k]` matrix of `dtype` takes with this plan.
    fn largest_buffer(self, dtype: yaxus_model::ggml::GgmlType, n: usize, k: usize) -> usize {
        use yaxus_model::ggml::GgmlType;
        let largest = |t: GgmlType| ggml_largest_buffer(t, n, k).expect("a type with a GPU layout");
        match self {
            MatrixPlan::FileLayout => largest(dtype),
            MatrixPlan::Convert(Quant::Q4) => largest(GgmlType::Q4_0),
            MatrixPlan::Convert(Quant::Bf16) => n * k * 2,
            MatrixPlan::Convert(_) => largest(GgmlType::Q8_0),
        }
    }
}

/// The values of `t` as bf16: borrowed from the file when they are stored so, else
/// converted.
fn bf16_of(t: &TensorRef) -> std::borrow::Cow<'_, [u16]> {
    match t.as_bf16() {
        Some(b) => std::borrow::Cow::Borrowed(b),
        None => std::borrow::Cow::Owned(t.to_bf16_vec()),
    }
}

/// Whether the embedding table is kept as the file's Q4_K blocks (93 MB for the 0.8B
/// instead of 330 MB in bf16), on every device: the browser needs the room and WebGPU's
/// buffer limits, and native keeps the same numbers (the rows are the exact dequantized
/// values, not bf16-rounded), so the CLI and the browser translate alike.
fn embedding_stays_q4k(t: &TensorRef) -> bool {
    t.dtype == yaxus_model::ggml::GgmlType::Q4K && t.shape.len() == 2 && t.shape[1].is_multiple_of(256)
}

impl GpuBackend {
    /// The largest buffer one of the weights of `w` takes on the device with
    /// `Quant::Auto`, before any device is open (WebGPU asks for its limits up front):
    /// each matrix's largest buffer, vectors in f32, and the embedding table as
    /// [`Gpu::load_embedding`] stores it.
    pub fn largest_weight_bytes(w: &Weights) -> u64 {
        w.names()
            .filter_map(|name| w.get(name).ok().map(|t| (name, t)))
            .map(|(name, t)| {
                let bytes = if name == yaxus_model::EMBED_TOKENS {
                    if embedding_stays_q4k(t) { t.bytes().len() } else { t.numel() * 2 }
                } else if let &[n, k] = &t.shape[..] {
                    MatrixPlan::of(t, Quant::Auto).largest_buffer(t.dtype, n, k)
                } else {
                    t.numel() * 4
                };
                bytes as u64
            })
            .max()
            .unwrap_or(0)
    }
}

impl Gpu {
    /// Uploads a buffer of a weight, counting its bytes. One larger than a storage binding
    /// may be is noted for [`Gpu::load_checked`]: binding it would fail at the first
    /// dispatch, long after loading.
    fn upload_weight(&mut self, label: &str, bytes: &[u8]) -> Buf {
        self.loaded.weight_bytes += bytes.len();
        if bytes.len() as u64 > self.rec.dev.max_binding() {
            self.loaded.oversized.get_or_insert((label.to_string(), bytes.len()));
        }
        self.rec.dev.upload(label, bytes)
    }

    /// Runs a load with the device's errors captured (see [`Gpu::captured`]), and fails it
    /// if it uploaded a buffer that exceeds the storage binding limit.
    pub(crate) fn load_checked<T>(&mut self, what: &str, f: impl FnOnce(&mut Self) -> Result<T>) -> Result<T> {
        // Only report a buffer from this load.
        self.loaded.oversized = None;
        let (r, error) = self.captured(f);
        if let Some(e) = error {
            anyhow::bail!("{what}: {e}");
        }
        if let Some((label, bytes)) = self.loaded.oversized.take() {
            anyhow::bail!(
                "{what}: a {label} buffer of {bytes} bytes exceeds the device's storage binding limit of {}",
                self.rec.dev.max_binding()
            );
        }
        r
    }

    pub(crate) fn load_embedding(&mut self, t: &TensorRef) -> Result<Weight> {
        let h = t.shape[1];
        if embedding_stays_q4k(t) {
            let q = yaxus_model::ggml::q4k_from_q4_k(t.bytes());
            let table = self.upload_weight("embedding-q4k", &q.data);
            return Ok(Weight::Embedding { table, h, fmt: Format::Q4K });
        }
        let table = self.upload_weight("embedding", bytemuck::cast_slice(&bf16_of(t)));
        Ok(Weight::Embedding { table, h, fmt: Format::Bf16 })
    }

    pub(crate) fn load_weight(&mut self, t: &TensorRef, quant: Quant) -> Result<Weight> {
        if t.shape.len() == 1 {
            let v = t.to_f32_vec();
            return Ok(Weight::Vector(self.upload_weight("vector", bytemuck::cast_slice(&v))));
        }
        let (n, k) = (t.shape[0], t.shape[1]);
        let plan = MatrixPlan::of(t, quant);
        // A matrix whose buffers one storage binding cannot cover in the format chosen for
        // it is read at 4 bits instead (only when the format was left to us). Each buffer
        // is bound on its own: the 0.8B's Q6_K output is 135 MB in the file but its
        // largest buffer is 124 MB, which a 128 MB binding covers. Native devices bind
        // far more.
        let format_left_to_us = matches!(quant, Quant::Auto | Quant::Auto4);
        if format_left_to_us
            && plan != MatrixPlan::Convert(Quant::Q4)
            && plan.largest_buffer(t.dtype, n, k) as u64 > self.rec.dev.max_binding()
        {
            tracing::warn!(
                "a {n}x{k} {} matrix exceeds the storage binding limit ({} bytes); loading it as Q4",
                t.dtype.name(),
                self.rec.dev.max_binding()
            );
            self.loaded.reduced_to_q4 += 1;
            return self.load_matrix(&bf16_of(t), n, k, Quant::Q4).map(Weight::Matrix);
        }
        match plan {
            MatrixPlan::FileLayout => {
                let m = self.load_gguf_matrix(t, n, k)?;
                Ok(Weight::Matrix(m.expect("`MatrixPlan::of` keeps only types with a GPU layout")))
            }
            MatrixPlan::Convert(q) => self.load_matrix(&bf16_of(t), n, k, q).map(Weight::Matrix),
        }
    }

    pub(crate) fn load_matrix(&mut self, bf16: &[u16], n: usize, k: usize, quant: Quant) -> Result<Matrix> {
        anyhow::ensure!(k.is_multiple_of(32), "K={k} must be a multiple of 32");
        Ok(match quant {
            Quant::Q8 | Quant::Auto => self.upload_q8(yaxus_model::quant::quantize_q8(bf16, k), n, k),
            Quant::Q4 | Quant::Auto4 => self.upload_q4(yaxus_model::quant::quantize_q4(bf16, k), n, k),
            Quant::Bf16 => {
                let q = self.upload_weight("bf16", bytemuck::cast_slice(bf16));
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
                let qb = self.upload_weight("q4k", &ggml::q4k_from_q4_k(b).data);
                Matrix { fmt: Format::Q4K, n, k, q: qb.clone(), s: Some(qb), s16: None }
            }
            GgmlType::Q6K => {
                let q = ggml::q6_from_q6_k(b, n, k);
                let qb = self.upload_weight("q6", &q.qs);
                let sb = self.upload_weight("q6-meta", &q.meta);
                let s16 = pad_scales(q.scales16.iter().map(|s| s.to_bits()).collect(), n, k / 16, 1);
                let s16 = self.upload_weight("q6-scales", bytemuck::cast_slice(&s16));
                Matrix { fmt: Format::Q6, n, k, q: qb, s: Some(sb), s16: Some(s16) }
            }
            GgmlType::Q3K => {
                let q = ggml::q3_from_q3_k(b, n, k);
                let qb = self.upload_weight("q3", &q.qs);
                let sb = self.upload_weight("q3-meta", &q.meta);
                let s16 = pad_scales(q.scales16.iter().map(|s| s.to_bits()).collect(), n, k / 16, 1);
                let s16 = self.upload_weight("q3-scales", bytemuck::cast_slice(&s16));
                Matrix { fmt: Format::Q3, n, k, q: qb, s: Some(sb), s16: Some(s16) }
            }
            GgmlType::Q2K => {
                let q = ggml::q2_from_q2_k(b, n, k);
                let qb = self.upload_weight("q2", &q.qs);
                let sb = self.upload_weight("q2-meta", &q.meta);
                let s16 = pad_scales(q.pairs16.iter().map(|s| s.to_bits()).collect(), n, k / 16, 2);
                let s16 = self.upload_weight("q2-scales", bytemuck::cast_slice(&s16));
                Matrix { fmt: Format::Q2, n, k, q: qb, s: Some(sb), s16: Some(s16) }
            }
            _ => return None,
        })
    }

    pub(crate) fn upload_q8(&mut self, q: yaxus_model::quant::Q8Data, n: usize, k: usize) -> Matrix {
        let qb = self.upload_weight("q8", bytemuck::cast_slice(&q.qs));
        let sc = pad_scales(q.scales.iter().map(|s| s.to_bits()).collect(), n, k / 32, 1);
        let sb = self.upload_weight("q8-scales", bytemuck::cast_slice(&sc));
        Matrix { fmt: Format::Q8, n, k, q: qb, s: Some(sb), s16: None }
    }

    pub(crate) fn upload_q4(&mut self, q: yaxus_model::quant::Q4Data, n: usize, k: usize) -> Matrix {
        let qb = self.upload_weight("q4", &q.qs);
        let sc = pad_scales(q.scales.iter().map(|s| s.to_bits()).collect(), n, k / 32, 1);
        let sb = self.upload_weight("q4-scales", bytemuck::cast_slice(&sc));
        Matrix { fmt: Format::Q4, n, k, q: qb, s: Some(sb), s16: None }
    }
}

#[cfg(test)]
mod tests {
    use yaxus_model::ggml::{self, GgmlType};

    use super::*;

    /// `ggml_largest_buffer` follows the layouts `load_ggml` uploads, padded scales
    /// included (K = 4096 gives Q6_K, Q3_K and Q2_K rows of 256 scale entries).
    #[test]
    fn largest_buffer_matches_the_packed_layouts() {
        let u16s = |v: Vec<u16>| v.len() * 2;
        for (n, k) in [(3, 512), (2, 4096), (1, 8192)] {
            let bytes = |t: GgmlType| vec![0u8; t.bytes_for(n * k)];
            let sc = |v: &[half::f16], blocks: usize, unit: usize| {
                u16s(pad_scales(v.iter().map(|s| s.to_bits()).collect(), n, blocks, unit))
            };
            let q8 = ggml::q8_from_q8_0(&bytes(GgmlType::Q8_0));
            let q4 = ggml::q4_from_q4_0(&bytes(GgmlType::Q4_0));
            let q6 = ggml::q6_from_q6_k(&bytes(GgmlType::Q6K), n, k);
            let q3 = ggml::q3_from_q3_k(&bytes(GgmlType::Q3K), n, k);
            let q2 = ggml::q2_from_q2_k(&bytes(GgmlType::Q2K), n, k);
            let want = [
                (GgmlType::Q8_0, q8.qs.len().max(sc(&q8.scales, k / 32, 1))),
                (GgmlType::Q4_0, q4.qs.len().max(sc(&q4.scales, k / 32, 1))),
                (GgmlType::Q4K, ggml::q4k_from_q4_k(&bytes(GgmlType::Q4K)).data.len()),
                (GgmlType::Q6K, q6.qs.len().max(q6.meta.len()).max(sc(&q6.scales16, k / 16, 1))),
                (GgmlType::Q3K, q3.qs.len().max(q3.meta.len()).max(sc(&q3.scales16, k / 16, 1))),
                (GgmlType::Q2K, q2.qs.len().max(q2.meta.len()).max(sc(&q2.pairs16, k / 16, 2))),
            ];
            for (t, w) in want {
                assert_eq!(ggml_largest_buffer(t, n, k), Some(w), "{t:?} [{n}, {k}]");
            }
        }
        // The 0.8B's Q6_K output fits a 128 MiB binding; only its file size does not.
        let (n, k) = (107520, 1536);
        assert!(GgmlType::Q6K.bytes_for(n * k) > 128 << 20);
        assert!(ggml_largest_buffer(GgmlType::Q6K, n, k).unwrap() < 128 << 20);
    }

    /// The plan of each quantization and type, and that the size estimate follows it.
    #[test]
    fn matrix_plans_keep_the_file_layout_only_where_asked() {
        use MatrixPlan::*;
        let t = |dtype: GgmlType| {
            let n = 4;
            let k = 256;
            let mut b = b"GGUF".to_vec();
            b.extend(3u32.to_le_bytes());
            b.extend(1u64.to_le_bytes());
            b.extend(0u64.to_le_bytes());
            b.extend(1u64.to_le_bytes());
            b.push(b't');
            b.extend(2u32.to_le_bytes());
            b.extend((k as u64).to_le_bytes());
            b.extend((n as u64).to_le_bytes());
            let id = match dtype {
                GgmlType::Bf16 => 30u32,
                GgmlType::Q6K => 14,
                GgmlType::Q4K => 12,
                _ => unreachable!(),
            };
            b.extend(id.to_le_bytes());
            b.extend(0u64.to_le_bytes());
            b.resize(b.len().div_ceil(32) * 32 + dtype.bytes_for(n * k), 0);
            yaxus_model::gguf::Gguf::parse(b).unwrap().tensors.remove(0).1
        };
        let (q6, q4k, bf16) = (t(GgmlType::Q6K), t(GgmlType::Q4K), t(GgmlType::Bf16));
        assert_eq!(MatrixPlan::of(&q6, Quant::Auto), FileLayout);
        assert_eq!(MatrixPlan::of(&q6, Quant::Auto4), Convert(Quant::Q4));
        assert_eq!(MatrixPlan::of(&q4k, Quant::Auto4), FileLayout);
        assert_eq!(MatrixPlan::of(&bf16, Quant::Auto), Convert(Quant::Q8));
        assert_eq!(MatrixPlan::of(&q6, Quant::Bf16), Convert(Quant::Bf16));
        // Sizes: the layout of the plan, not of the file.
        assert_eq!(
            FileLayout.largest_buffer(GgmlType::Q6K, 4, 256),
            ggml_largest_buffer(GgmlType::Q6K, 4, 256).unwrap()
        );
        assert_eq!(Convert(Quant::Q8).largest_buffer(GgmlType::Bf16, 4, 256), 4 * 256);
        assert_eq!(Convert(Quant::Q4).largest_buffer(GgmlType::Q6K, 4, 256), 4 * 256 / 2);
    }
}
