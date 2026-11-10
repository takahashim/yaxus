//! Matmul kernels (every path: gemv1, gemvm2..8, gemm, glu) against a host reference
//! computed from the same quantized weights. Needs a Vulkan GPU:
//! `cargo test --release -p yakusel-backend-wgpu --test kernels -- --ignored`

use yakusel_backend_wgpu::WgpuBackend;
use yakusel_model::Quant;
use yakusel_model::ggml::GgmlType;
use yakusel_model::quant::{Q8_BLOCK, quantize_q4, quantize_q8};

fn f32_to_bf16(x: f32) -> u16 {
    (x.to_bits() >> 16) as u16
}

/// Deterministic pseudo-random values in [-1, 1).
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// The weights each format actually stores, dequantized.
fn dequant(w: &[u16], k: usize, quant: Quant) -> Vec<f32> {
    let bf = |x: u16| f32::from_bits((x as u32) << 16);
    match quant {
        Quant::Bf16 => w.iter().map(|&x| bf(x)).collect(),
        Quant::Q8 | Quant::Auto => {
            let q = quantize_q8(w, k);
            q.qs.iter().enumerate().map(|(i, &v)| v as f32 * q.scales[i / Q8_BLOCK].to_f32()).collect()
        }
        Quant::Q4 | Quant::Auto4 => {
            let q = quantize_q4(w, k);
            (0..w.len())
                .map(|i| {
                    let byte = q.qs[i / 2];
                    let nib = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
                    (nib as f32 - 8.0) * q.scales[i / Q8_BLOCK].to_f32()
                })
                .collect()
        }
    }
}

fn reference(x: &[f32], t: usize, w: &[f32], n: usize, k: usize, glu: bool) -> Vec<f32> {
    let mut y = vec![0f32; t * n];
    for i in 0..t {
        for j in 0..n {
            y[i * n + j] = (0..k).map(|c| x[i * k + c] * w[j * k + c]).sum();
        }
    }
    if !glu {
        return y;
    }
    let h = n / 2;
    (0..t)
        .flat_map(|i| {
            let y = &y;
            (0..h).map(move |j| {
                let g = y[i * n + j];
                g / (1.0 + (-g).exp()) * y[i * n + h + j]
            })
        })
        .collect()
}

#[test]
#[ignore]
fn matmul_kernels_match_host() {
    let mut gpu = WgpuBackend::new().unwrap();
    // K multiple of 512 (gemvm) and not (fallback); N a multiple of 64 (matrix-core GEMM)
    // and not (tile tails).
    let f16_gemm = gpu.gemm_uses_f16();
    for (n, k) in [(256, 1536), (200, 1536), (130, 320)] {
        let w: Vec<u16> = values(n * k, 7).iter().map(|&v| f32_to_bf16(v * 0.05)).collect();
        for quant in [Quant::Bf16, Quant::Q8, Quant::Q4] {
            let wd = dequant(&w, k, quant);
            for t in [1, 2, 3, 5, 8, 12, 16, 20, 24, 70] {
                for glu in [false, true] {
                    if glu && t > 1 {
                        continue; // glu is fused only for decode; other T use matmul + swiglu
                    }
                    let x = values(t * k, 11 + t as u64);
                    let want = reference(&x, t, &wd, n, k, glu);
                    let got = gpu.debug_matmul(&x, t, &w, n, k, quant, glu);
                    assert_eq!(got.len(), want.len());
                    let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    // f16 operands (matrix cores) round each input to 11 bits.
                    let tol = if f16_gemm && t > 8 && n % 64 == 0 { 3e-3 } else { 1e-4 };
                    assert!(err <= tol * scale.max(1.0), "{quant:?} n={n} k={k} t={t} glu={glu}: err {err}");
                }
            }
        }
    }
}

/// Random but valid GGML blocks for an `[n, k]` matrix of type `t` (scales kept small so
/// the products stay in a sane range).
fn ggml_blocks(t: GgmlType, n: usize, k: usize, seed: u64) -> Vec<u8> {
    let (e, b) = t.block();
    let blocks = n * k / e;
    let mut out: Vec<u8> = values(blocks * b, seed).iter().map(|v| ((v + 1.0) * 127.5) as u8).collect();
    let f16b = |v: f32| half::f16::from_f32(v).to_le_bytes();
    for i in 0..blocks {
        let blk = &mut out[i * b..(i + 1) * b];
        match t {
            GgmlType::Q4K => {
                blk[0..2].copy_from_slice(&f16b(0.002));
                blk[2..4].copy_from_slice(&f16b(0.001));
            }
            GgmlType::Q6K => blk[208..210].copy_from_slice(&f16b(0.0005)),
            GgmlType::Q3K => blk[108..110].copy_from_slice(&f16b(0.0015)),
            GgmlType::Q2K => {
                blk[80..82].copy_from_slice(&f16b(0.003));
                blk[82..84].copy_from_slice(&f16b(0.002));
            }
            GgmlType::Q8_0 => blk[0..2].copy_from_slice(&f16b(0.0004)),
            GgmlType::Q4_0 => blk[0..2].copy_from_slice(&f16b(0.006)),
            _ => unreachable!(),
        }
    }
    out
}

#[test]
#[ignore]
fn ggml_matmul_kernels_match_host() {
    let mut gpu = WgpuBackend::new().unwrap();
    let f16_gemm = gpu.gemm_uses_f16();
    // K a multiple of 256 (K-quant super-blocks); K = 4096 and 8192 pad the per-16 scale
    // rows, K = 8192 and 16384 the Q4_K header rows.
    for (n, k) in [(256, 1536), (200, 1536), (128, 8192), (64, 4096), (64, 16384)] {
        for dtype in [GgmlType::Q2K, GgmlType::Q3K, GgmlType::Q4K, GgmlType::Q6K, GgmlType::Q8_0, GgmlType::Q4_0] {
            let bytes = ggml_blocks(dtype, n, k, 3);
            let wd = yakusel_model::ggml::dequantize(dtype, &bytes, n * k);
            for t in [1, 2, 5, 8, 16, 20, 70] {
                for glu in [false, true] {
                    if glu && t > 1 {
                        continue;
                    }
                    let x = values(t * k, 21 + t as u64);
                    let want = reference(&x, t, &wd, n, k, glu);
                    let got = gpu.debug_matmul_ggml(&x, t, dtype, &bytes, n, k, glu);
                    let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    // Q4_K/Q6_K scales are rounded to f16 on the GPU (~5e-4 relative).
                    let tol = if f16_gemm && t > 8 && n % 64 == 0 { 3e-3 } else { 1e-3 };
                    assert!(err <= tol * scale.max(1.0), "{dtype:?} n={n} k={k} t={t} glu={glu}: err {err} (scale {scale})");
                }
            }
        }
    }
}
