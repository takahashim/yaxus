//! GGML tensor types (as stored in GGUF files): dequantization to f32 on the CPU and
//! lossless repacking of the common types into the layouts of [`crate::quant`].
//!
//! Block layouts follow ggml's `ggml-common.h` / `dequantize_row_*` (QK_K = 256).

use anyhow::{Result, bail};
use half::f16;
use rayon::prelude::*;

use crate::quant::{Q4Data, Q4MData, Q8B16Data, Q8Data};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlType {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            30 => GgmlType::Bf16,
            other => bail!("GGML tensor type {other} is not supported"),
        })
    }

    /// The ggml name (`Q4_K`, `BF16`, ...).
    pub fn name(self) -> &'static str {
        match self {
            GgmlType::F32 => "F32",
            GgmlType::F16 => "F16",
            GgmlType::Bf16 => "BF16",
            GgmlType::Q4_0 => "Q4_0",
            GgmlType::Q4_1 => "Q4_1",
            GgmlType::Q5_0 => "Q5_0",
            GgmlType::Q5_1 => "Q5_1",
            GgmlType::Q8_0 => "Q8_0",
            GgmlType::Q2K => "Q2_K",
            GgmlType::Q3K => "Q3_K",
            GgmlType::Q4K => "Q4_K",
            GgmlType::Q5K => "Q5_K",
            GgmlType::Q6K => "Q6_K",
        }
    }

    /// (elements, bytes) per block.
    pub fn block(self) -> (usize, usize) {
        match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 | GgmlType::Bf16 => (1, 2),
            GgmlType::Q4_0 => (32, 18),
            GgmlType::Q4_1 => (32, 20),
            GgmlType::Q5_0 => (32, 22),
            GgmlType::Q5_1 => (32, 24),
            GgmlType::Q8_0 => (32, 34),
            GgmlType::Q2K => (256, 84),
            GgmlType::Q3K => (256, 110),
            GgmlType::Q4K => (256, 144),
            GgmlType::Q5K => (256, 176),
            GgmlType::Q6K => (256, 210),
        }
    }

    pub fn bytes_for(self, n: usize) -> usize {
        let (e, b) = self.block();
        n / e * b
    }
}

fn h(b: &[u8], at: usize) -> f32 {
    f16::from_le_bytes([b[at], b[at + 1]]).to_f32()
}

/// Scale and min of sub-block `j` (0..8) from the 12 packed bytes of Q4_K/Q5_K.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

fn dequant_block(t: GgmlType, b: &[u8], y: &mut [f32]) {
    match t {
        GgmlType::F32 => y[0] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        GgmlType::F16 => y[0] = h(b, 0),
        GgmlType::Bf16 => y[0] = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16),
        GgmlType::Q4_0 => {
            let d = h(b, 0);
            for j in 0..16 {
                y[j] = ((b[2 + j] & 0xF) as f32 - 8.0) * d;
                y[j + 16] = ((b[2 + j] >> 4) as f32 - 8.0) * d;
            }
        }
        GgmlType::Q4_1 => {
            let (d, m) = (h(b, 0), h(b, 2));
            for j in 0..16 {
                y[j] = (b[4 + j] & 0xF) as f32 * d + m;
                y[j + 16] = (b[4 + j] >> 4) as f32 * d + m;
            }
        }
        GgmlType::Q5_0 | GgmlType::Q5_1 => {
            let one = t == GgmlType::Q5_1;
            let d = h(b, 0);
            let m = if one { h(b, 2) } else { 0.0 };
            let o = if one { 4 } else { 2 };
            let qh = u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
            let qs = &b[o + 4..];
            let off = if one { 0 } else { 16 };
            for j in 0..16 {
                let h0 = ((qh >> j) << 4) & 0x10;
                let h1 = (qh >> (j + 12)) & 0x10;
                let x0 = ((qs[j] & 0x0F) as u32 | h0) as i32 - off;
                let x1 = ((qs[j] >> 4) as u32 | h1) as i32 - off;
                y[j] = x0 as f32 * d + m;
                y[j + 16] = x1 as f32 * d + m;
            }
        }
        GgmlType::Q8_0 => {
            let d = h(b, 0);
            for j in 0..32 {
                y[j] = b[2 + j] as i8 as f32 * d;
            }
        }
        GgmlType::Q2K => {
            let (sc, qs) = (&b[0..16], &b[16..80]);
            let (d, dmin) = (h(b, 80), h(b, 82));
            let (mut is, mut o) = (0, 0);
            for n in 0..2 {
                let q = &qs[32 * n..];
                for shift in [0, 2, 4, 6] {
                    for half in 0..2 {
                        let s = sc[is];
                        is += 1;
                        let (dl, ml) = (d * (s & 0xF) as f32, dmin * (s >> 4) as f32);
                        for l in 0..16 {
                            y[o] = dl * ((q[l + 16 * half] >> shift) & 3) as f32 - ml;
                            o += 1;
                        }
                    }
                }
            }
        }
        GgmlType::Q3K => {
            let (hm, qs, raw) = (&b[0..32], &b[32..96], &b[96..108]);
            let d = h(b, 108);
            let mut aux = [0u32; 4];
            for (i, a) in aux.iter_mut().enumerate().take(3) {
                *a = u32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]]);
            }
            let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
            aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
            aux[0] = (aux[0] & k2) | ((tmp & k1) << 4);
            aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
            let scales: Vec<i8> = aux.iter().flat_map(|a| a.to_le_bytes()).map(|x| x as i8).collect();
            let (mut is, mut o, mut m) = (0, 0, 1u8);
            for n in 0..2 {
                let q = &qs[32 * n..];
                for shift in [0, 2, 4, 6] {
                    for half in 0..2 {
                        let dl = d * (scales[is] as i32 - 32) as f32;
                        is += 1;
                        for l in 0..16 {
                            let i = l + 16 * half;
                            let v = ((q[i] >> shift) & 3) as i32 - if hm[i] & m != 0 { 0 } else { 4 };
                            y[o] = dl * v as f32;
                            o += 1;
                        }
                    }
                    m <<= 1;
                }
            }
        }
        GgmlType::Q4K | GgmlType::Q5K => {
            let five = t == GgmlType::Q5K;
            let (d, dmin) = (h(b, 0), h(b, 2));
            let sc = &b[4..16];
            let (qh, ql) = if five { (&b[16..48], &b[48..176]) } else { (&b[0..0], &b[16..144]) };
            for c in 0..4 {
                let (s1, m1) = scale_min_k4(2 * c, sc);
                let (s2, m2) = scale_min_k4(2 * c + 1, sc);
                let (d1, d2) = (d * s1 as f32, d * s2 as f32);
                let (n1, n2) = (dmin * m1 as f32, dmin * m2 as f32);
                for l in 0..32 {
                    let q = ql[32 * c + l];
                    let (mut lo, mut hi) = ((q & 0xF) as u32, (q >> 4) as u32);
                    if five {
                        lo |= (((qh[l] >> (2 * c)) & 1) as u32) << 4;
                        hi |= (((qh[l] >> (2 * c + 1)) & 1) as u32) << 4;
                    }
                    y[64 * c + l] = d1 * lo as f32 - n1;
                    y[64 * c + 32 + l] = d2 * hi as f32 - n2;
                }
            }
        }
        GgmlType::Q6K => {
            for (i, v) in q6k_values(b).into_iter().enumerate() {
                let (q, s) = v;
                y[i] = s * q as f32;
            }
        }
    }
}

/// The 256 `(q - 32, d * scale)` pairs of a Q6_K block, in element order. Element `i`
/// uses scale `i / 16`.
fn q6k_values(b: &[u8]) -> Vec<(i32, f32)> {
    let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
    let d = h(b, 208);
    let mut out = vec![(0, 0.0); 256];
    for half in 0..2 {
        let (l_, hh) = (&ql[64 * half..], &qh[32 * half..]);
        for l in 0..32 {
            let q = [
                (l_[l] & 0xF) | ((hh[l] & 3) << 4),
                (l_[l + 32] & 0xF) | (((hh[l] >> 2) & 3) << 4),
                (l_[l] >> 4) | (((hh[l] >> 4) & 3) << 4),
                (l_[l + 32] >> 4) | (((hh[l] >> 6) & 3) << 4),
            ];
            for (k, &qk) in q.iter().enumerate() {
                let i = 128 * half + 32 * k + l;
                out[i] = (qk as i32 - 32, d * sc[i / 16] as i8 as f32);
            }
        }
    }
    out
}

/// Dequantizes `n` elements of type `t`.
pub fn dequantize(t: GgmlType, bytes: &[u8], n: usize) -> Vec<f32> {
    let (e, bsz) = t.block();
    let mut y = vec![0f32; n];
    y.par_chunks_mut(e)
        .zip(bytes.par_chunks(bsz))
        .with_min_len(256)
        .for_each(|(out, blk)| dequant_block(t, blk, out));
    y
}

/// Q8_0 as [`Q8Data`] (identical values: 32 int8 and one f16 scale per block).
pub fn q8_from_q8_0(bytes: &[u8]) -> Q8Data {
    let nb = bytes.len() / 34;
    let mut qs = vec![0i8; nb * 32];
    let mut scales = vec![f16::ZERO; nb];
    qs.par_chunks_mut(32).zip(scales.par_iter_mut()).zip(bytes.par_chunks(34)).for_each(|((q, s), b)| {
        *s = f16::from_le_bytes([b[0], b[1]]);
        for (o, &x) in q.iter_mut().zip(&b[2..34]) {
            *o = x as i8;
        }
    });
    Q8Data { qs, scales }
}

/// Q4_0 as [`Q4Data`] (identical values; ggml keeps elements `j` and `j + 16` in one
/// byte, `Q4Data` keeps `2i` and `2i + 1`).
pub fn q4_from_q4_0(bytes: &[u8]) -> Q4Data {
    let nb = bytes.len() / 18;
    let mut qs = vec![0u8; nb * 16];
    let mut scales = vec![f16::ZERO; nb];
    qs.par_chunks_mut(16).zip(scales.par_iter_mut()).zip(bytes.par_chunks(18)).for_each(|((q, s), b)| {
        *s = f16::from_le_bytes([b[0], b[1]]);
        let nib = |e: usize| if e < 16 { b[2 + e] & 0xF } else { b[2 + e - 16] >> 4 };
        for (i, o) in q.iter_mut().enumerate() {
            *o = nib(2 * i) | (nib(2 * i + 1) << 4);
        }
    });
    Q4Data { qs, scales }
}

/// Q4_K as [`Q4MData`]: each 32-element sub-block becomes 16 bytes of nibbles with its
/// scale `d * sc` and min `dmin * m` (rounded to f16; Q4_K keeps them as 6-bit integers
/// times f16 super-block factors).
pub fn q4m_from_q4_k(bytes: &[u8]) -> Q4MData {
    let nb = bytes.len() / 144;
    let mut qs = vec![0u8; nb * 128];
    let mut sm = vec![f16::ZERO; nb * 16];
    qs.par_chunks_mut(128).zip(sm.par_chunks_mut(16)).zip(bytes.par_chunks(144)).for_each(|((q, s), b)| {
        let (d, dmin) = (h(b, 0), h(b, 2));
        let (sc, ql) = (&b[4..16], &b[16..144]);
        for sub in 0..8 {
            let (sv, mv) = scale_min_k4(sub, sc);
            s[2 * sub] = f16::from_f32(d * sv as f32);
            s[2 * sub + 1] = f16::from_f32(dmin * mv as f32);
            let (c, hi) = (sub / 2, sub % 2 == 1);
            let nib = |l: usize| {
                let x = ql[32 * c + l];
                if hi { x >> 4 } else { x & 0xF }
            };
            for i in 0..16 {
                q[16 * sub + i] = nib(2 * i) | (nib(2 * i + 1) << 4);
            }
        }
    });
    Q4MData { qs, sm }
}

/// Q6_K as [`Q8B16Data`]: the 6-bit values `q - 32` as int8 with one f16 scale
/// (`d * sc`) per 16 elements.
pub fn q8b16_from_q6_k(bytes: &[u8]) -> Q8B16Data {
    let nb = bytes.len() / 210;
    let mut qs = vec![0i8; nb * 256];
    let mut scales = vec![f16::ZERO; nb * 16];
    qs.par_chunks_mut(256).zip(scales.par_chunks_mut(16)).zip(bytes.par_chunks(210)).for_each(|((q, s), b)| {
        for (i, (v, sc)) in q6k_values(b).into_iter().enumerate() {
            q[i] = v as i8;
            s[i / 16] = f16::from_f32(sc);
        }
    });
    Q8B16Data { qs, scales }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q4k_block(seed: u32) -> Vec<u8> {
        let mut b: Vec<u8> = (0..144u32).map(|i| (i.wrapping_mul(2654435761).wrapping_add(seed) >> 13) as u8).collect();
        b[0..2].copy_from_slice(&f16::from_f32(0.01).to_le_bytes());
        b[2..4].copy_from_slice(&f16::from_f32(0.003).to_le_bytes());
        b
    }

    #[test]
    fn repacked_q4_k_matches_dequant() {
        let b: Vec<u8> = (0..3).flat_map(q4k_block).collect();
        let want = dequantize(GgmlType::Q4K, &b, 768);
        let p = q4m_from_q4_k(&b);
        for (e, &w) in want.iter().enumerate() {
            let byte = p.qs[e / 2];
            let q = if e % 2 == 0 { byte & 0xF } else { byte >> 4 } as f32;
            let blk = e / 32;
            let got = q * p.sm[2 * blk].to_f32() - p.sm[2 * blk + 1].to_f32();
            assert!((got - w).abs() <= 1e-3 * w.abs().max(1e-2), "{e}: {got} vs {w}");
        }
    }

    #[test]
    fn repacked_q6_k_matches_dequant() {
        let mut b: Vec<u8> = (0..420u32).map(|i| (i.wrapping_mul(2246822519) >> 11) as u8).collect();
        for blk in 0..2 {
            b[210 * blk + 208..210 * blk + 210].copy_from_slice(&f16::from_f32(0.002).to_le_bytes());
        }
        let want = dequantize(GgmlType::Q6K, &b, 512);
        let p = q8b16_from_q6_k(&b);
        for (e, &w) in want.iter().enumerate() {
            let got = p.qs[e] as f32 * p.scales[e / 16].to_f32();
            assert!((got - w).abs() <= 1e-3 * w.abs().max(1e-2), "{e}: {got} vs {w}");
        }
    }

    #[test]
    fn q8_0_and_q4_0_repack_exactly() {
        let mut b: Vec<u8> = (0..68u32).map(|i| (i * 37 + 5) as u8).collect();
        b[0..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        b[34..36].copy_from_slice(&f16::from_f32(-0.25).to_le_bytes());
        let want = dequantize(GgmlType::Q8_0, &b, 64);
        let p = q8_from_q8_0(&b);
        for (e, &w) in want.iter().enumerate() {
            assert_eq!(p.qs[e] as f32 * p.scales[e / 32].to_f32(), w);
        }
        let mut b: Vec<u8> = (0..36u32).map(|i| (i * 53 + 7) as u8).collect();
        b[0..2].copy_from_slice(&f16::from_f32(0.125).to_le_bytes());
        b[18..20].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        let want = dequantize(GgmlType::Q4_0, &b, 64);
        let p = q4_from_q4_0(&b);
        for (e, &w) in want.iter().enumerate() {
            let byte = p.qs[e / 2];
            let q = if e % 2 == 0 { byte & 0xF } else { byte >> 4 } as f32 - 8.0;
            assert_eq!(q * p.scales[e / 32].to_f32(), w);
        }
    }
}
