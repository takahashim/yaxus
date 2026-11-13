//! Q8 quantization shared by all backends (so they see bit-identical weights).

use half::f16;

use crate::par;
#[cfg(not(target_arch = "wasm32"))]
use crate::par::*;
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
    let blocks =
        par::chunks_mut(&mut qs, Q8_BLOCK).zip(par::iter_mut(&mut scales)).zip(par::chunks(w, Q8_BLOCK));
    #[cfg(not(target_arch = "wasm32"))]
    let blocks = blocks.with_min_len(1024);
    blocks.for_each(
        |((q, sc), src)| {
            let amax = src.iter().map(|&x| bf16_to_f32(x).abs()).fold(0f32, f32::max);
            let d = f16::from_f32(amax / 127.0);
            *sc = d;
            let d = d.to_f32();
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            for (o, &x) in q.iter_mut().zip(src) {
                *o = (bf16_to_f32(x) * id).round().clamp(-127.0, 127.0) as i8;
            }
        },
    );
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
    let blocks =
        par::chunks_mut(&mut qs, Q8_BLOCK / 2).zip(par::iter_mut(&mut scales)).zip(par::chunks(w, Q8_BLOCK));
    #[cfg(not(target_arch = "wasm32"))]
    let blocks = blocks.with_min_len(1024);
    blocks.for_each(
        |((q, sc), src)| {
            let amax = src.iter().map(|&x| bf16_to_f32(x).abs()).fold(0f32, f32::max);
            let d = f16::from_f32(amax / 7.0);
            *sc = d;
            let d = d.to_f32();
            let id = if d > 0.0 { 1.0 / d } else { 0.0 };
            let nib = |x: u16| ((bf16_to_f32(x) * id).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
            for (e, o) in q.iter_mut().enumerate() {
                *o = nib(src[2 * e]) | (nib(src[2 * e + 1]) << 4);
            }
        },
    );
    Q4Data { qs, scales }
}

/// `[N, K]` matrix of 6-bit values (from GGUF Q6_K) in 24-byte groups of 32: 16 bytes
/// of low nibbles (element `e` of the group in byte `e / 2`, low nibble for even `e`),
/// then 8 bytes of high 2-bit fields (element `e` at bits `2 * (e % 4)` of byte
/// `16 + e / 4`). Weight = `(q - 32) * d * sc` as in Q6_K: per row, `meta` holds the
/// `K / 256` f16 `d` values, then the `K / 16` int8 `sc` values, padded as
/// [`q6_meta_stride`] says. `scales16` holds `d * sc` as one f16 per 16 values, for
/// kernels that look a scale up per 4 values (the prefill GEMMs).
pub struct Q6Data {
    pub qs: Vec<u8>,
    pub meta: Vec<u8>,
    pub scales16: Vec<f16>,
}

/// Bytes per row of [`Q6Data::meta`]: `18 * K / 256` rounded up to 4, plus 16 when that
/// is a multiple of 512 (power-of-two strides slow the GPU GEMMs; `q6row` in the shaders).
pub fn q6_meta_stride(k: usize) -> usize {
    let s = (18 * (k / 256)).div_ceil(4) * 4;
    if s.is_multiple_of(512) { s + 16 } else { s }
}

/// `[N, K]` matrix as GGUF Q4_K blocks of 144 bytes per 256 elements: the 16 header
/// bytes (`d`, `dmin` as f16, then the 12 bytes of packed 6-bit scales and mins), then
/// the nibbles laid out as in [`Q4Data`] (ggml interleaves them by 32-byte chunk).
pub struct Q4KData {
    pub data: Vec<u8>,
}

/// `[N, K]` matrix with GGUF Q2_K's values: 2-bit values in element order (element `e`
/// of a row at bits `2 * (e % 4)` of byte `e / 4`). Per row, `meta` holds `K / 256`
/// pairs of f16 (`d`, `dmin`), then `K / 16` bytes with a 4-bit scale (low) and a 4-bit
/// min (high): weight = `q * d * sc - dmin * mn`. `pairs16` holds (`d * sc`,
/// `dmin * mn`) as f16 per 16 values for the prefill GEMMs.
pub struct Q2Data {
    pub qs: Vec<u8>,
    pub meta: Vec<u8>,
    pub pairs16: Vec<f16>,
}

/// `[N, K]` matrix with GGUF Q3_K's values in 12-byte groups of 32: 8 bytes of low
/// 2-bit fields in element order, then 4 bytes with the high bit of each element.
/// Weight = `(q - 4) * d * sc` with `meta` as in [`Q6Data`] (`sc` already offset by
/// -32) and `scales16` its f16 products per 16 values.
pub struct Q3Data {
    pub qs: Vec<u8>,
    pub meta: Vec<u8>,
    pub scales16: Vec<f16>,
}
