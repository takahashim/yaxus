//! Reference CPU backend: f32 activations, bf16 weights decoded on the fly, rayon.
//!
//! Not optimized for speed; it is the ground truth the GPU kernels are checked against.

use anyhow::Result;
use half::f16;
use rayon::prelude::*;
use yaxus_model::{Backend, Config, LayerMeta, PAGE, PassMeta, Quant, TensorRef, bf16_to_f32};

use yaxus_model::quant::Q8_BLOCK;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor {
    fn zeros(rows: usize, cols: usize) -> Self {
        Tensor { data: vec![0.0; rows * cols], rows, cols }
    }
    fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.cols..(r + 1) * self.cols]
    }
}

/// A weight as the CPU keeps it: bf16 (vectors, and matrices loaded as bf16) or Q8.
pub enum Weight {
    Bf16(Bf16Matrix),
    Q8(Q8Matrix),
}

/// The bf16 values of a weight: borrowed from the file's mapping, or decoded from
/// another format.
enum Bf16Data {
    Mapped(TensorRef),
    Owned(Vec<u16>),
}

/// A bf16 weight of any shape (`[N, K]` for matrices).
pub struct Bf16Matrix {
    data: Bf16Data,
    shape: Vec<usize>,
}

impl Bf16Matrix {
    /// The tensor's values as bf16, without a copy when they are stored so.
    fn from_tensor(t: &TensorRef) -> Self {
        let data = match t.as_bf16() {
            Some(_) => Bf16Data::Mapped(t.clone()),
            None => Bf16Data::Owned(t.to_bf16_vec()),
        };
        Bf16Matrix { data, shape: t.shape.clone() }
    }

    #[cfg(test)]
    fn owned(values: Vec<u16>, shape: &[usize]) -> Self {
        Bf16Matrix { data: Bf16Data::Owned(values), shape: shape.to_vec() }
    }

    fn values(&self) -> &[u16] {
        match &self.data {
            Bf16Data::Mapped(t) => t.as_bf16().expect("checked at load"),
            Bf16Data::Owned(v) => v,
        }
    }
}

/// `[N, K]` matrix; row `n` block `b` is `qs[n*K + b*32 ..][..32] * scales[n*K/32 + b]`.
pub struct Q8Matrix {
    qs: Vec<i8>,
    scales: Vec<f16>,
    shape: Vec<usize>,
}

impl Q8Matrix {
    pub fn quantize(w: &[u16], shape: &[usize]) -> Self {
        let q = yaxus_model::quant::quantize_q8(w, shape[1]);
        Q8Matrix { qs: q.qs, scales: q.scales, shape: shape.to_vec() }
    }

    fn from_tensor(t: &TensorRef) -> Self {
        match t.as_bf16() {
            Some(b) => Q8Matrix::quantize(b, &t.shape),
            None => Q8Matrix::quantize(&t.to_bf16_vec(), &t.shape),
        }
    }

    fn row(&self, n: usize) -> (&[i8], &[f16]) {
        let k = self.shape[1];
        (&self.qs[n * k..(n + 1) * k], &self.scales[n * k / Q8_BLOCK..(n + 1) * k / Q8_BLOCK])
    }
}

/// The rows of an `[N, K]` weight matrix, as `matmul` reads them.
trait MatrixRows: Sync {
    fn shape(&self) -> (usize, usize);
    /// `w[j] · x`, reading the stored row directly (few activation rows).
    fn dot_row(&self, j: usize, x: &[f32]) -> f32;
    /// Row `j` as f32, to be reused for many activation rows.
    fn decode_row(&self, j: usize, out: &mut [f32]);
}

/// The rows of a bf16 matrix, its values looked up once.
struct Bf16Rows<'a> {
    values: &'a [u16],
    n: usize,
    k: usize,
}

impl Bf16Matrix {
    fn rows(&self) -> Bf16Rows<'_> {
        Bf16Rows { values: self.values(), n: self.shape[0], k: self.shape[1] }
    }
}

impl MatrixRows for Bf16Rows<'_> {
    fn shape(&self) -> (usize, usize) {
        (self.n, self.k)
    }
    fn dot_row(&self, j: usize, x: &[f32]) -> f32 {
        dot_bf16(&self.values[j * self.k..(j + 1) * self.k], x)
    }
    fn decode_row(&self, j: usize, out: &mut [f32]) {
        for (r, &b) in out.iter_mut().zip(&self.values[j * self.k..(j + 1) * self.k]) {
            *r = bf16_to_f32(b);
        }
    }
}

impl MatrixRows for Q8Matrix {
    fn shape(&self) -> (usize, usize) {
        (self.shape[0], self.shape[1])
    }
    fn dot_row(&self, j: usize, x: &[f32]) -> f32 {
        let (qs, sc) = self.row(j);
        dot_q8(qs, sc, x)
    }
    fn decode_row(&self, j: usize, out: &mut [f32]) {
        let (qs, sc) = self.row(j);
        dequant_q8_row(qs, sc, out);
    }
}

/// How the CPU keeps a weight loaded with `quant`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CpuFormat {
    Bf16,
    Q8,
}

impl CpuFormat {
    /// Bf16 as asked; Q8 for Q8 and the file-dependent formats (the CPU keeps no GGUF
    /// format of its own), for matrices whose rows are whole Q8 blocks; Q4 is Vulkan only.
    fn of(quant: Quant, shape: &[usize]) -> Result<Self> {
        match quant {
            Quant::Bf16 => Ok(CpuFormat::Bf16),
            Quant::Q4 => anyhow::bail!("q4 weights are only supported on the Vulkan backend"),
            Quant::Q8 | Quant::Auto | Quant::Auto4 => {
                anyhow::ensure!(
                    matches!(shape, [_, k] if k.is_multiple_of(Q8_BLOCK)),
                    "q8 needs a matrix whose rows are a multiple of {Q8_BLOCK} (shape {shape:?})"
                );
                Ok(CpuFormat::Q8)
            }
        }
    }
}

impl Weight {
    /// The values of a vector (or of the embedding table), which are always bf16.
    fn as_vector(&self) -> &[u16] {
        match self {
            Weight::Bf16(m) => m.values(),
            Weight::Q8(_) => panic!("vectors and the embedding table are loaded as bf16"),
        }
    }

    fn shape(&self) -> &[usize] {
        match self {
            Weight::Bf16(m) => &m.shape,
            Weight::Q8(q) => &q.shape,
        }
    }

    /// Bytes this weight occupies.
    pub fn nbytes(&self) -> usize {
        match self {
            Weight::Bf16(m) => m.values().len() * 2,
            Weight::Q8(q) => q.qs.len() + q.scales.len() * 2,
        }
    }

    fn vector_f32(&self) -> Vec<f32> {
        self.as_vector().iter().map(|&x| bf16_to_f32(x)).collect()
    }
}

struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
}

/// Paged KV storage: slot `s` of layer `l` holds `kv_dim` floats at `s * kv_dim`.
pub struct Kv {
    layers: Vec<LayerKv>,
    kv_dim: usize,
}

#[derive(Default)]
pub struct CpuBackend {
    /// Total bytes of loaded weights.
    pub weight_bytes: usize,
    pass: PassMeta,
}

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend::default()
    }
}

/// Number of physical cores (SMT siblings counted once), if it can be determined.
pub fn physical_cores() -> Option<usize> {
    let mut seen = std::collections::HashSet::new();
    for e in std::fs::read_dir("/sys/devices/system/cpu").ok()?.flatten() {
        let name = e.file_name();
        let name = name.to_string_lossy();
        if name.starts_with("cpu")
            && name[3..].chars().all(|c| c.is_ascii_digit())
            && let Ok(s) = std::fs::read_to_string(e.path().join("topology/thread_siblings_list"))
        {
            seen.insert(s.trim().to_string());
        }
    }
    (!seen.is_empty()).then_some(seen.len())
}

/// Sets up the global rayon pool. Decoding is memory-bound, and one thread per physical
/// core is slightly faster than using SMT siblings. `RAYON_NUM_THREADS` overrides this.
pub fn init_threads() {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }
    if let Some(n) = physical_cores() {
        // The host's cores are not all ours under a cgroup CPU quota (containers) or an
        // affinity mask (taskset); available_parallelism accounts for both.
        let n = std::thread::available_parallelism().map_or(n, |a| n.min(a.get()));
        let _ = rayon::ThreadPoolBuilder::new().num_threads(n).build_global();
    }
}

/// Dot product of bf16 weights with f32 activations, without a temporary buffer.
fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (cw, rw) = w.as_chunks::<16>();
    let (cx, rx) = x.as_chunks::<16>();
    for (a, b) in cw.iter().zip(cx) {
        for i in 0..16 {
            acc[i] += bf16_to_f32(a[i]) * b[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (&a, &b) in rw.iter().zip(rx) {
        s += bf16_to_f32(a) * b;
    }
    s
}

/// Dot product of one Q8 row with f32 activations.
fn dot_q8(qs: &[i8], scales: &[f16], x: &[f32]) -> f32 {
    let mut total = [0f32; 16];
    let (cq, _) = qs.as_chunks::<Q8_BLOCK>();
    let (cx, _) = x.as_chunks::<Q8_BLOCK>();
    for ((q, xs), sc) in cq.iter().zip(cx).zip(scales) {
        let mut acc = [0f32; 16];
        for i in 0..16 {
            acc[i] = q[i] as f32 * xs[i] + q[i + 16] as f32 * xs[i + 16];
        }
        let d = sc.to_f32();
        for i in 0..16 {
            total[i] += acc[i] * d;
        }
    }
    total.iter().sum()
}

fn dequant_q8_row(qs: &[i8], scales: &[f16], out: &mut [f32]) {
    for ((o, q), sc) in out.as_chunks_mut::<Q8_BLOCK>().0.iter_mut().zip(qs.as_chunks::<Q8_BLOCK>().0).zip(scales) {
        let d = sc.to_f32();
        for (a, &b) in o.iter_mut().zip(q) {
            *a = b as f32 * d;
        }
    }
}

/// With this many activation rows or more, a weight row is decoded to f32 once and reused.
const DECODE_ROW_MIN_T: usize = 4;

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (ca, ra) = a.as_chunks::<16>();
    let (cb, rb) = b.as_chunks::<16>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..16 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

fn rmsnorm_row(x: &[f32], w: &[f32], offset: f32, eps: f32, out: &mut [f32]) {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    for ((o, &v), &g) in out.iter_mut().zip(x).zip(w) {
        *o = (offset + g) * (v * r);
    }
}

fn rope_inplace(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = x.len() / 2;
    for j in 0..half {
        let (a, b) = (x[j], x[j + half]);
        x[j] = a * cos[j] - b * sin[j];
        x[j + half] = b * cos[j] + a * sin[j];
    }
}

/// `x [T, K] · w[N, K]^T -> [T, N]`.
fn matmul<M: MatrixRows>(x: &Tensor, w: &M) -> Tensor {
    let (n, k) = w.shape();
    assert_eq!(x.cols, k, "matmul inner dim");
    let t = x.rows;
    // Compute transposed [N, T] so each task owns one weight row.
    let mut tr = vec![0f32; n * t];
    tr.par_chunks_mut(t).enumerate().with_min_len(8).for_each_init(
        || vec![0f32; k],
        |row, (j, out)| {
            if t < DECODE_ROW_MIN_T {
                for (i, o) in out.iter_mut().enumerate() {
                    *o = w.dot_row(j, x.row(i));
                }
            } else {
                w.decode_row(j, row);
                for (i, o) in out.iter_mut().enumerate() {
                    *o = dot(x.row(i), row);
                }
            }
        },
    );
    if t == 1 {
        return Tensor { data: tr, rows: 1, cols: n };
    }
    // Transpose back, one output row per task (about 2% of a 512-row matmul; a serial
    // loop took a few times longer).
    let mut out = Tensor::zeros(t, n);
    out.data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for (j, o) in row.iter_mut().enumerate() {
            *o = tr[j * t + i];
        }
    });
    out
}

impl Backend for CpuBackend {
    type Tensor = Tensor;
    type Weight = Weight;
    type Kv = Kv;

    fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    fn name(&self) -> String {
        format!("cpu ({} threads)", rayon::current_num_threads())
    }

    fn load_weight(&mut self, t: &TensorRef, quant: Quant) -> Result<Weight> {
        let w = match CpuFormat::of(quant, &t.shape)? {
            CpuFormat::Bf16 => Weight::Bf16(Bf16Matrix::from_tensor(t)),
            CpuFormat::Q8 => Weight::Q8(Q8Matrix::from_tensor(t)),
        };
        self.weight_bytes += w.nbytes();
        Ok(w)
    }

    fn new_kv(&mut self, cfg: &Config, metas: &[LayerMeta], pages: usize, _max_pos: usize) -> Result<Kv> {
        let kv_dim = cfg.kv_dim();
        let n = pages * PAGE * kv_dim;
        // Zeroed allocations are lazily backed by the OS, so a large arena is cheap.
        let layers = metas.iter().map(|_| LayerKv { k: vec![0.0; n], v: vec![0.0; n] }).collect();
        Ok(Kv { layers, kv_dim })
    }

    fn copy_page(&mut self, kv: &mut Kv, src: u32, dst: u32) {
        let n = PAGE * kv.kv_dim;
        let (s, d) = (src as usize * n, dst as usize * n);
        for l in &mut kv.layers {
            l.k.copy_within(s..s + n, d);
            l.v.copy_within(s..s + n, d);
        }
    }

    fn begin_pass(&mut self, pass: &PassMeta) {
        self.pass = pass.clone();
    }

    fn embed(&mut self, table: &Weight) -> Tensor {
        let ids = &self.pass.ids;
        let h = table.shape()[1];
        let w = table.as_vector();
        let mut out = Tensor::zeros(ids.len(), h);
        for (r, &id) in ids.iter().enumerate() {
            let src = &w[id as usize * h..(id as usize + 1) * h];
            for (o, &s) in out.data[r * h..(r + 1) * h].iter_mut().zip(src) {
                *o = bf16_to_f32(s);
            }
        }
        out
    }

    fn rmsnorm(&mut self, x: &Tensor, w: &Weight, offset: f32, eps: f32) -> Tensor {
        let w = w.vector_f32();
        let mut out = Tensor::zeros(x.rows, x.cols);
        out.data
            .par_chunks_mut(x.cols)
            .zip(x.data.par_chunks(x.cols))
            .for_each(|(o, xr)| rmsnorm_row(xr, &w, offset, eps, o));
        out
    }

    fn matmul(&mut self, x: &Tensor, w: &Weight) -> Tensor {
        match w {
            Weight::Bf16(m) => matmul(x, &m.rows()),
            Weight::Q8(q) => matmul(x, q),
        }
    }

    fn attention(
        &mut self,
        qkv: &Tensor,
        q_norm: &Weight,
        k_norm: &Weight,
        kv: &mut Kv,
        cfg: &Config,
        layer: &LayerMeta,
    ) -> Tensor {
        let pass = &self.pass;
        let (hq, hkv, d) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let (qd, kvd) = (hq * d, hkv * d);
        let group = hq / hkv;
        let t = qkv.rows;
        let qn = q_norm.vector_f32();
        let kn = k_norm.vector_f32();
        let eps = cfg.rms_norm_eps;
        let lc = &mut kv.layers[layer.index];
        assert_eq!(t, pass.pos.len(), "attention rows differ from the pass");

        let mut cos = vec![0f32; d / 2];
        let mut sin = vec![0f32; d / 2];
        let mut q = vec![0f32; t * qd];
        for i in 0..t {
            let row = qkv.row(i);
            let pos = pass.pos[i] as usize;
            layer.rope.cos_sin(pos, &mut cos, &mut sin);
            for h in 0..hq {
                let dst = &mut q[i * qd + h * d..i * qd + (h + 1) * d];
                rmsnorm_row(&row[h * d..(h + 1) * d], &qn, 1.0, eps, dst);
                rope_inplace(dst, &cos, &sin);
            }
            let slot = pass.kv_slot[i] as usize;
            for h in 0..hkv {
                let src = &row[qd + h * d..qd + (h + 1) * d];
                let dst = &mut lc.k[slot * kvd + h * d..slot * kvd + (h + 1) * d];
                rmsnorm_row(src, &kn, 1.0, eps, dst);
                rope_inplace(dst, &cos, &sin);
                let vsrc = &row[qd + kvd + h * d..qd + kvd + (h + 1) * d];
                lc.v[slot * kvd + h * d..slot * kvd + (h + 1) * d].copy_from_slice(vsrc);
            }
        }

        let scale = 1.0 / (d as f32).sqrt();
        let (kc, vc) = (&lc.k, &lc.v);
        let mut out = Tensor::zeros(t, qd);
        out.data.par_chunks_mut(d).enumerate().for_each(|(idx, o)| {
            let (i, h) = (idx / hq, idx % hq);
            let kh = h / group;
            let p = pass.pos[i] as usize;
            let pages = &pass.pages[pass.pt_off[i] as usize..];
            let slot = |s: usize| pages[s / PAGE] as usize * PAGE + s % PAGE;
            let lo = match layer.window {
                Some(w) => (p + 1).saturating_sub(w),
                None => 0,
            };
            let qv = &q[i * qd + h * d..i * qd + (h + 1) * d];
            let scores: Vec<f32> = (lo..=p)
                .map(|s| {
                    let off = slot(s) * kvd + kh * d;
                    dot(qv, &kc[off..off + d]) * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f32;
            for (s, &sc) in (lo..=p).zip(&scores) {
                let e = (sc - m).exp();
                denom += e;
                let off = slot(s) * kvd + kh * d;
                for (ov, &vv) in o.iter_mut().zip(&vc[off..off + d]) {
                    *ov += e * vv;
                }
            }
            for ov in o.iter_mut() {
                *ov /= denom;
            }
        });
        out
    }

    fn swiglu(&mut self, x: &Tensor) -> Tensor {
        let i = x.cols / 2;
        let mut out = Tensor::zeros(x.rows, i);
        out.data.par_chunks_mut(i).zip(x.data.par_chunks(x.cols)).for_each(|(o, r)| {
            for j in 0..i {
                let g = r[j];
                o[j] = g / (1.0 + (-g).exp()) * r[i + j];
            }
        });
        out
    }

    fn add_assign(&mut self, a: &mut Tensor, b: &Tensor) {
        assert_eq!(a.data.len(), b.data.len());
        a.data.iter_mut().zip(&b.data).for_each(|(x, y)| *x += y);
    }

    fn select_rows(&mut self, x: &Tensor, rows: &[usize]) -> Tensor {
        let data = rows.iter().flat_map(|&r| x.row(r).iter().copied()).collect();
        Tensor { data, rows: rows.len(), cols: x.cols }
    }

    fn argmax_rows(&mut self, logits: &Tensor) -> Vec<u32> {
        (0..logits.rows)
            .map(|r| {
                let row = logits.row(r);
                let mut best = 0;
                for (i, &v) in row.iter().enumerate() {
                    if v > row[best] {
                        best = i;
                    }
                }
                best as u32
            })
            .collect()
    }

    fn to_host(&mut self, x: &Tensor) -> Vec<f32> {
        x.data.clone()
    }
}

#[cfg(test)]
mod tests {
    use yaxus_model::Rope;

    use super::*;

    /// Small deterministic values in [-1, 1).
    fn values(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn config(hq: usize, hkv: usize, d: usize) -> Config {
        serde_json::from_str(&format!(
            r#"{{"hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": {hq},
                "num_key_value_heads": {hkv}, "head_dim": {d}, "intermediate_size": 8,
                "vocab_size": 16, "rms_norm_eps": 1e-6, "window_size": 16,
                "sliding_window_pattern": 1, "rope_theta": 10000.0, "rope_local_theta": 10000.0}}"#
        ))
        .unwrap()
    }

    /// Dense attention of one sequence's rows (positions 0..n), as the paged kernel
    /// should compute it.
    fn naive(qkv: &[Vec<f32>], cfg: &Config, layer: &LayerMeta) -> Vec<Vec<f32>> {
        let (hq, hkv, d) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let (qd, kvd) = (hq * d, hkv * d);
        let zero_w = vec![0f32; d];
        let (mut cos, mut sin) = (vec![0f32; d / 2], vec![0f32; d / 2]);
        let (mut qs, mut ks, mut vs) = (Vec::new(), Vec::new(), Vec::new());
        for (p, row) in qkv.iter().enumerate() {
            layer.rope.cos_sin(p, &mut cos, &mut sin);
            let mut q = vec![0f32; qd];
            for h in 0..hq {
                rmsnorm_row(&row[h * d..(h + 1) * d], &zero_w, 1.0, cfg.rms_norm_eps, &mut q[h * d..(h + 1) * d]);
                rope_inplace(&mut q[h * d..(h + 1) * d], &cos, &sin);
            }
            let mut k = vec![0f32; kvd];
            for h in 0..hkv {
                let src = &row[qd + h * d..qd + (h + 1) * d];
                rmsnorm_row(src, &zero_w, 1.0, cfg.rms_norm_eps, &mut k[h * d..(h + 1) * d]);
                rope_inplace(&mut k[h * d..(h + 1) * d], &cos, &sin);
            }
            qs.push(q);
            ks.push(k);
            vs.push(row[qd + kvd..].to_vec());
        }
        let scale = 1.0 / (d as f32).sqrt();
        (0..qkv.len())
            .map(|p| {
                let lo = layer.window.map_or(0, |w| (p + 1).saturating_sub(w));
                let mut out = vec![0f32; qd];
                for h in 0..hq {
                    let kh = h / (hq / hkv);
                    let sc: Vec<f32> = (lo..=p)
                        .map(|s| dot(&qs[p][h * d..(h + 1) * d], &ks[s][kh * d..(kh + 1) * d]) * scale)
                        .collect();
                    let m = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let w: Vec<f32> = sc.iter().map(|x| (x - m).exp()).collect();
                    let z: f32 = w.iter().sum();
                    for (s, wi) in (lo..=p).zip(&w) {
                        for j in 0..d {
                            out[h * d + j] += wi / z * vs[s][kh * d + j];
                        }
                    }
                }
                out
            })
            .collect()
    }

    /// Runs the paged attention for some rows of several sequences in one pass.
    /// `rows[i] = (sequence, position)`; `pages[sequence]` is its page table.
    fn pass(
        b: &mut CpuBackend,
        kv: &mut Kv,
        cfg: &Config,
        layer: &LayerMeta,
        qkv: &[Vec<Vec<f32>>],
        pages: &[Vec<u32>],
        rows: &[(usize, usize)],
    ) -> Vec<Vec<f32>> {
        let mut meta = PassMeta::default();
        let mut off = vec![None; pages.len()];
        for &(sq, p) in rows {
            let o = *off[sq].get_or_insert_with(|| {
                let o = meta.pages.len() as u32;
                meta.pages.extend(&pages[sq]);
                o
            });
            meta.ids.push(0);
            meta.pos.push(p as u32);
            meta.kv_slot.push(pages[sq][p / PAGE] * PAGE as u32 + (p % PAGE) as u32);
            meta.pt_off.push(o);
        }
        b.begin_pass(&meta);
        let cols = qkv[0][0].len();
        let data = rows.iter().flat_map(|&(sq, p)| qkv[sq][p].iter().copied()).collect();
        let x = Tensor { data, rows: rows.len(), cols };
        let norm = Weight::Bf16(Bf16Matrix::owned(vec![0; cfg.head_dim], &[cfg.head_dim]));
        let out = b.attention(&x, &norm, &norm, kv, cfg, layer);
        (0..out.rows).map(|r| out.row(r).to_vec()).collect()
    }

    #[test]
    fn paged_attention_matches_dense_attention() {
        let (hq, hkv, d) = (4, 2, 8);
        let cfg = config(hq, hkv, d);
        let cols = (hq + 2 * hkv) * d;
        let rope = Rope { inv_freq: vec![1.0, 0.3, 0.05, 0.01], concentration: 1.1 };
        for window in [None, Some(16)] {
            let layer = LayerMeta { index: 0, full_attn: window.is_none(), window, rope: rope.clone() };
            let mut b = CpuBackend::new();
            let mut kv = b.new_kv(&cfg, std::slice::from_ref(&layer), 4, 1024).unwrap();
            // Sequence 0 spans two non-adjacent pages; sequence 1 has one.
            let pages = vec![vec![3, 1], vec![0]];
            let lens = [140, 9];
            let qkv: Vec<Vec<Vec<f32>>> = lens
                .iter()
                .enumerate()
                .map(|(s, &n)| (0..n).map(|p| values(cols, (s * 1000 + p) as u64)).collect())
                .collect();
            let want: Vec<Vec<Vec<f32>>> = qkv.iter().map(|q| naive(q, &cfg, &layer)).collect();

            // First pass: 100 rows of sequence 0 and 4 of sequence 1; second pass: the
            // rest, so the second reads keys written by the first, across the page boundary.
            let mut got: Vec<Vec<Vec<f32>>> = lens.iter().map(|&n| vec![Vec::new(); n]).collect();
            let first: Vec<(usize, usize)> = (0..100).map(|p| (0, p)).chain((0..4).map(|p| (1, p))).collect();
            let second: Vec<(usize, usize)> = (100..140).map(|p| (0, p)).chain((4..9).map(|p| (1, p))).collect();
            for rows in [first, second] {
                for (row, &(s, p)) in pass(&mut b, &mut kv, &cfg, &layer, &qkv, &pages, &rows).into_iter().zip(&rows) {
                    got[s][p] = row;
                }
            }
            for s in 0..2 {
                for p in 0..lens[s] {
                    for (a, w) in got[s][p].iter().zip(&want[s][p]) {
                        assert!((a - w).abs() < 1e-5, "window {window:?}, seq {s}, pos {p}: {a} vs {w}");
                    }
                }
            }
        }
    }

    #[test]
    fn matmul_matches_a_plain_loop() {
        let mut b = CpuBackend::new();
        let (n, k) = (40, 64);
        let w: Vec<u16> = values(n * k, 7).iter().map(|&x| yaxus_model::weights::f32_to_bf16(x)).collect();
        // bf16 is exact up to f32 rounding; Q8 is off by the 8-bit quantization of the
        // weights (and, depending on the path, of the input), so it is compared with a
        // tolerance relative to the size of the terms.
        let weights = [
            ("bf16", Weight::Bf16(Bf16Matrix::owned(w.clone(), &[n, k])), 1e-4),
            ("q8", Weight::Q8(Q8Matrix::quantize(&w, &[n, k])), 2e-2),
        ];
        for (name, weight, tol) in &weights {
            for t in [1, 3, 9] {
                let x = Tensor { data: values(t * k, t as u64), rows: t, cols: k };
                let y = b.matmul(&x, weight);
                assert_eq!((y.rows, y.cols), (t, n));
                for i in 0..t {
                    for j in 0..n {
                        let terms = (0..k).map(|c| bf16_to_f32(w[j * k + c]) * x.data[i * k + c]);
                        let (want, scale) = terms.fold((0f32, 0f32), |(s, a), v| (s + v, a + v.abs()));
                        let got = y.data[i * n + j];
                        assert!((got - want).abs() <= tol * scale.max(1.0), "{name} t {t} ({i}, {j}): {got} vs {want}");
                    }
                }
            }
        }
    }
}
