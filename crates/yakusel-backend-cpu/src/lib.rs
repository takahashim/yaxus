//! Reference CPU backend: f32 activations, bf16 weights decoded on the fly, rayon.
//!
//! Not optimized for speed; it is the ground truth the GPU kernels are checked against.

use anyhow::Result;
use rayon::prelude::*;
use half::f16;
use yakusel_model::{Backend, Config, LayerMeta, Quant, TensorRef, bf16_to_f32};

use yakusel_model::quant::Q8_BLOCK;

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

pub enum Weight {
    Mapped(TensorRef),
    Owned(Vec<u16>, Vec<usize>),
    Q8(Q8Matrix),
}

/// `[N, K]` matrix; row `n` block `b` is `qs[n*K + b*32 ..][..32] * scales[n*K/32 + b]`.
pub struct Q8Matrix {
    qs: Vec<i8>,
    scales: Vec<f16>,
    shape: Vec<usize>,
}

impl Q8Matrix {
    pub fn quantize(w: &[u16], shape: &[usize]) -> Self {
        let q = yakusel_model::quant::quantize_q8(w, shape[1]);
        Q8Matrix { qs: q.qs, scales: q.scales, shape: shape.to_vec() }
    }

    fn row(&self, n: usize) -> (&[i8], &[f16]) {
        let k = self.shape[1];
        (&self.qs[n * k..(n + 1) * k], &self.scales[n * k / Q8_BLOCK..(n + 1) * k / Q8_BLOCK])
    }
}

impl Weight {
    fn bf16(&self) -> &[u16] {
        match self {
            Weight::Mapped(t) => t.as_bf16().expect("checked at load"),
            Weight::Owned(v, _) => v,
            Weight::Q8(_) => panic!("bf16 view of a quantized weight"),
        }
    }
    fn shape(&self) -> &[usize] {
        match self {
            Weight::Mapped(t) => &t.shape,
            Weight::Owned(_, s) => s,
            Weight::Q8(q) => &q.shape,
        }
    }

    /// Bytes this weight occupies.
    pub fn nbytes(&self) -> usize {
        match self {
            Weight::Q8(q) => q.qs.len() + q.scales.len() * 2,
            _ => self.bf16().len() * 2,
        }
    }
    fn to_f32(&self) -> Vec<f32> {
        self.bf16().iter().map(|&x| bf16_to_f32(x)).collect()
    }
}

struct LayerCache {
    k: Vec<f32>,
    v: Vec<f32>,
    /// Ring capacity in tokens.
    cap: usize,
}

pub struct Cache {
    layers: Vec<LayerCache>,
}

#[derive(Default)]
pub struct CpuBackend {
    /// Total bytes of loaded weights.
    pub weight_bytes: usize,
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
        if name.starts_with("cpu") && name[3..].chars().all(|c| c.is_ascii_digit()) {
            if let Ok(s) = std::fs::read_to_string(e.path().join("topology/thread_siblings_list")) {
                seen.insert(s.trim().to_string());
            }
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
    for ((o, q), sc) in out.chunks_exact_mut(Q8_BLOCK).zip(qs.chunks_exact(Q8_BLOCK)).zip(scales) {
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

impl Backend for CpuBackend {
    type Tensor = Tensor;
    type Weight = Weight;
    type Cache = Cache;

    fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    fn name(&self) -> String {
        format!("cpu ({} threads)", rayon::current_num_threads())
    }

    fn load_weight(&mut self, t: &TensorRef, quant: Quant) -> Result<Weight> {
        let w = match (quant, t.as_bf16()) {
            (Quant::Q4, _) => anyhow::bail!("q4 weights are only supported on the Vulkan backend"),
            (Quant::Q8, Some(b)) => Weight::Q8(Q8Matrix::quantize(b, &t.shape)),
            (Quant::Q8, None) => Weight::Q8(Q8Matrix::quantize(&t.to_bf16_vec(), &t.shape)),
            (Quant::Bf16, Some(_)) => Weight::Mapped(t.clone()),
            (Quant::Bf16, None) => Weight::Owned(t.to_bf16_vec(), t.shape.clone()),
        };
        self.weight_bytes += w.nbytes();
        Ok(w)
    }

    fn new_cache(&mut self, cfg: &Config, metas: &[LayerMeta], max_len: usize, chunk: usize) -> Result<Cache> {
        let kv = cfg.kv_dim();
        let layers = metas
            .iter()
            .map(|m| {
                // A sliding layer needs the window plus one chunk of in-flight queries.
                let cap = match m.window {
                    Some(w) => max_len.min(w + chunk),
                    None => max_len,
                };
                LayerCache { k: vec![0.0; cap * kv], v: vec![0.0; cap * kv], cap }
            })
            .collect();
        Ok(Cache { layers })
    }

    fn embed(&mut self, table: &Weight, ids: &[u32]) -> Tensor {
        let h = table.shape()[1];
        let w = table.bf16();
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
        let w = w.to_f32();
        let mut out = Tensor::zeros(x.rows, x.cols);
        out.data
            .par_chunks_mut(x.cols)
            .zip(x.data.par_chunks(x.cols))
            .for_each(|(o, xr)| rmsnorm_row(xr, &w, offset, eps, o));
        out
    }

    fn matmul(&mut self, x: &Tensor, w: &Weight) -> Tensor {
        let (n, k) = (w.shape()[0], w.shape()[1]);
        assert_eq!(x.cols, k, "matmul inner dim");
        let t = x.rows;
        // Compute transposed [N, T] so each task owns one weight row.
        let mut tr = vec![0f32; n * t];
        let tasks = tr.par_chunks_mut(t).enumerate().with_min_len(8);
        match w {
            Weight::Q8(q) => tasks.for_each_init(
                || vec![0f32; k],
                |row, (j, out)| {
                    let (qs, sc) = q.row(j);
                    if t < DECODE_ROW_MIN_T {
                        for (i, o) in out.iter_mut().enumerate() {
                            *o = dot_q8(qs, sc, x.row(i));
                        }
                    } else {
                        dequant_q8_row(qs, sc, row);
                        for (i, o) in out.iter_mut().enumerate() {
                            *o = dot(x.row(i), row);
                        }
                    }
                },
            ),
            _ => {
                let wb = w.bf16();
                tasks.for_each_init(
                    || vec![0f32; k],
                    |row, (j, out)| {
                        let src = &wb[j * k..(j + 1) * k];
                        if t < DECODE_ROW_MIN_T {
                            for (i, o) in out.iter_mut().enumerate() {
                                *o = dot_bf16(src, x.row(i));
                            }
                        } else {
                            for (r, &b) in row.iter_mut().zip(src) {
                                *r = bf16_to_f32(b);
                            }
                            for (i, o) in out.iter_mut().enumerate() {
                                *o = dot(x.row(i), row);
                            }
                        }
                    },
                )
            }
        }
        if t == 1 {
            return Tensor { data: tr, rows: 1, cols: n };
        }
        let mut out = Tensor::zeros(t, n);
        for j in 0..n {
            for i in 0..t {
                out.data[i * n + j] = tr[j * t + i];
            }
        }
        out
    }

    fn attention(
        &mut self,
        qkv: &Tensor,
        q_norm: &Weight,
        k_norm: &Weight,
        cache: &mut Cache,
        cfg: &Config,
        layer: &LayerMeta,
        pos0: usize,
    ) -> Tensor {
        let (hq, hkv, d) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let (qd, kvd) = (hq * d, hkv * d);
        let group = hq / hkv;
        let t = qkv.rows;
        let qn = q_norm.to_f32();
        let kn = k_norm.to_f32();
        let eps = cfg.rms_norm_eps;
        let lc = &mut cache.layers[layer.index];
        assert!(t <= lc.cap, "chunk larger than cache capacity");

        let mut cos = vec![0f32; d / 2];
        let mut sin = vec![0f32; d / 2];
        let mut q = vec![0f32; t * qd];
        for i in 0..t {
            let row = qkv.row(i);
            let pos = pos0 + i;
            layer.rope.cos_sin(pos, &mut cos, &mut sin);
            for h in 0..hq {
                let dst = &mut q[i * qd + h * d..i * qd + (h + 1) * d];
                rmsnorm_row(&row[h * d..(h + 1) * d], &qn, 1.0, eps, dst);
                rope_inplace(dst, &cos, &sin);
            }
            let slot = pos % lc.cap;
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
        let (kc, vc, cap) = (&lc.k, &lc.v, lc.cap);
        let mut out = Tensor::zeros(t, qd);
        out.data.par_chunks_mut(d).enumerate().for_each(|(idx, o)| {
            let (i, h) = (idx / hq, idx % hq);
            let kh = h / group;
            let p = pos0 + i;
            let lo = match layer.window {
                Some(w) => (p + 1).saturating_sub(w),
                None => 0,
            };
            let qv = &q[i * qd + h * d..i * qd + (h + 1) * d];
            let scores: Vec<f32> = (lo..=p)
                .map(|s| {
                    let off = (s % cap) * kvd + kh * d;
                    dot(qv, &kc[off..off + d]) * scale
                })
                .collect();
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0f32;
            for (s, &sc) in (lo..=p).zip(&scores) {
                let e = (sc - m).exp();
                denom += e;
                let off = (s % cap) * kvd + kh * d;
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
        out.data
            .par_chunks_mut(i)
            .zip(x.data.par_chunks(x.cols))
            .for_each(|(o, r)| {
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

    fn last_row(&mut self, x: &Tensor) -> Tensor {
        Tensor { data: x.row(x.rows - 1).to_vec(), rows: 1, cols: x.cols }
    }

    fn argmax(&mut self, logits: &Tensor) -> u32 {
        let r = logits.row(logits.rows - 1);
        let mut best = 0;
        for (i, &v) in r.iter().enumerate() {
            if v > r[best] {
                best = i;
            }
        }
        best as u32
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
