//! Q8 quantization shared by all backends (so they see bit-identical weights).

use half::f16;
use rayon::prelude::*;

use crate::bf16_to_f32;

/// Weights per scale.
pub const Q8_BLOCK: usize = 32;

/// `[N, K]` matrix; row `n`, block `b` is `qs[n*K + b*32 ..][..32] * scales[n*K/32 + b]`.
pub struct Q8Data {
    pub qs: Vec<i8>,
    pub scales: Vec<f16>,
}

/// Symmetric per-block int8: `scale = f16(max|w| / 127)`, `q = round(w / scale)`.
pub fn quantize_q8(w: &[u16], k: usize) -> Q8Data {
    assert_eq!(k % Q8_BLOCK, 0, "Q8 needs K divisible by {Q8_BLOCK}");
    let nb = w.len() / Q8_BLOCK;
    let mut qs = vec![0i8; w.len()];
    let mut scales = vec![f16::ZERO; nb];
    qs.par_chunks_mut(Q8_BLOCK)
        .zip(scales.par_iter_mut())
        .zip(w.par_chunks(Q8_BLOCK))
        .with_min_len(1024)
        .for_each(|((q, sc), src)| {
            let amax = src.iter().map(|&x| bf16_to_f32(x).abs()).fold(0f32, f32::max);
            let d = f16::from_f32(amax / 127.0);
            *sc = d;
            let d = d.to_f32();
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            for (o, &x) in q.iter_mut().zip(src) {
                *o = (bf16_to_f32(x) * id).round().clamp(-127.0, 127.0) as i8;
            }
        });
    Q8Data { qs, scales }
}

/// `[N, K]` matrix of 4-bit values: weight `e` of a 32-block lives in byte `b*16 + e/2`
/// (low nibble for even `e`), stored as `q + 8` for `q` in -8..=7.
pub struct Q4Data {
    pub qs: Vec<u8>,
    pub scales: Vec<f16>,
}

/// Symmetric per-block 4-bit: `scale = f16(max|w| / 7)`, `q = clamp(round(w / scale), -8, 7)`.
pub fn quantize_q4(w: &[u16], k: usize) -> Q4Data {
    assert_eq!(k % Q8_BLOCK, 0, "Q4 needs K divisible by {Q8_BLOCK}");
    let nb = w.len() / Q8_BLOCK;
    let mut qs = vec![0u8; w.len() / 2];
    let mut scales = vec![f16::ZERO; nb];
    qs.par_chunks_mut(Q8_BLOCK / 2)
        .zip(scales.par_iter_mut())
        .zip(w.par_chunks(Q8_BLOCK))
        .with_min_len(1024)
        .for_each(|((q, sc), src)| {
            let amax = src.iter().map(|&x| bf16_to_f32(x).abs()).fold(0f32, f32::max);
            let d = f16::from_f32(amax / 7.0);
            *sc = d;
            let d = d.to_f32();
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let nib = |x: u16| ((bf16_to_f32(x) * id).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
            for (e, o) in q.iter_mut().enumerate() {
                *o = nib(src[2 * e]) | (nib(src[2 * e + 1]) << 4);
            }
        });
    Q4Data { qs, scales }
}

/// `[N, K]` matrix of 4-bit values with a scale and a min per 32-block (from GGUF Q4_K):
/// weight = `q * sm[2b] - sm[2b + 1]`, `q` in 0..=15, nibbles laid out as in [`Q4Data`].
pub struct Q4MData {
    pub qs: Vec<u8>,
    pub sm: Vec<f16>,
}

/// `[N, K]` matrix of int8 values with one f16 scale per 16 (from GGUF Q6_K).
pub struct Q8B16Data {
    pub qs: Vec<i8>,
    pub scales: Vec<f16>,
}
