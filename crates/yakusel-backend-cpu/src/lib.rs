//! Reference CPU backend: f32 activations, bf16 weights decoded on the fly, rayon.
//!
//! Not optimized for speed; it is the ground truth the GPU kernels are checked against.

use anyhow::Result;
use rayon::prelude::*;
use half::f16;
use yakusel_model::{Backend, Config, LayerMeta, PAGE, PassMeta, Quant, TensorRef, bf16_to_f32};

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
    type Kv = Kv;

    fn weight_bytes(&self) -> usize {
        self.weight_bytes
    }

    fn name(&self) -> String {
        format!("cpu ({} threads)", rayon::current_num_threads())
    }

    fn load_weight(&mut self, t: &TensorRef, quant: Quant) -> Result<Weight> {
        let w = match (quant, t.as_bf16()) {
            (Quant::Q4, _) => anyhow::bail!("q4 weights are only supported on the Vulkan backend"),
            (Quant::Q8 | Quant::Auto, Some(b)) => Weight::Q8(Q8Matrix::quantize(b, &t.shape)),
            (Quant::Q8 | Quant::Auto, None) => Weight::Q8(Q8Matrix::quantize(&t.to_bf16_vec(), &t.shape)),
            (Quant::Bf16, Some(_)) => Weight::Mapped(t.clone()),
            (Quant::Bf16, None) => Weight::Owned(t.to_bf16_vec(), t.shape.clone()),
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
        kv: &mut Kv,
        cfg: &Config,
        layer: &LayerMeta,
    ) -> Tensor {
        let pass = &self.pass;
        let (hq, hkv, d) = (cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim);
        let (qd, kvd) = (hq * d, hkv * d);
        let group = hq / hkv;
        let t = qkv.rows;
        let qn = q_norm.to_f32();
        let kn = k_norm.to_f32();
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
