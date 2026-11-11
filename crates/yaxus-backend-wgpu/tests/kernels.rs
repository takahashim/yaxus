//! Matmul kernels (every path: gemv1, gemvm2..8, gemm, glu) against a host reference
//! computed from the same quantized weights. Needs a Vulkan GPU:
//! `cargo test --release -p yaxus-backend-wgpu --test kernels -- --ignored`

use yaxus_backend_wgpu::{GemmKind, Gemv, WgpuBackend};
use yaxus_model::Quant;
use yaxus_model::ggml::GgmlType;
use yaxus_model::quant::{Q8_BLOCK, quantize_q4, quantize_q8};

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

/// The prefill GEMM kinds this device has, and the error each may add to a product
/// relative to the largest output (operands rounded to f16, to 15 bits, to 7 bits).
fn gemm_kinds(gpu: &WgpuBackend) -> Vec<(GemmKind, f32)> {
    let mut kinds = vec![(GemmKind::F32, 0.0), (GemmKind::Int16, 3e-3), (GemmKind::Int8, 3e-2)];
    if gpu.has_matrix_cores() {
        kinds.push((GemmKind::Coop, 3e-3)); // (the default where matrix cores exist)
    }
    kinds
}

/// The kernels for up to 16 rows (`set_gemv`) and the error each may add.
const GEMV_KINDS: [(Option<usize>, f32); 3] = [(None, 0.0), (Some(2), 3e-3), (Some(1), 3e-2)];

#[test]
#[ignore]
fn matmul_kernels_match_host() {
    let mut gpu = WgpuBackend::new().unwrap();
    // K multiple of 512 (gemvm) and not (fallback); N a multiple of 64 (matrix-core GEMM)
    // and not (tile tails).
    for (n, k) in [(256, 1536), (200, 1536), (130, 320)] {
        let w: Vec<u16> = values(n * k, 7).iter().map(|&v| f32_to_bf16(v * 0.05)).collect();
        for quant in [Quant::Bf16, Quant::Q8, Quant::Q4] {
            let wd = dequant(&w, k, quant);
            // Up to 16 rows: the f32 GEMVs whatever the GEMM kind. Above: the GEMM, in
            // tiles of 32, 64 and 128 tokens.
            for t in [1, 2, 3, 5, 8, 12, 16, 20, 24, 40, 70, 150, 200] {
                for glu in [false, true] {
                    let x = values(t * k, 11 + t as u64);
                    let want = reference(&x, t, &wd, n, k, glu);
                    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    // (GEMM kind, GEMV planes, tolerance): the pass uses one of the two.
                    let mut cases: Vec<(GemmKind, Option<usize>, f32)> = if t <= 16 {
                        GEMV_KINDS.iter().map(|&(v, tol)| (GemmKind::F32, v, tol)).collect()
                    } else {
                        gemm_kinds(&gpu).into_iter().map(|(g, tol)| (g, None, tol)).collect()
                    };
                    if t > 16 && t <= 48 {
                        // Several integer GEMVs in place of the GEMM.
                        cases.push((GemmKind::Int16, Some(2), 3e-3));
                    }
                    for (kind, gemv, tol) in cases {
                        gpu.set_gemm(kind);
                        gpu.set_gemv(Gemv { planes: gemv, single_f32: false });
                        let got = gpu.debug_matmul(&x, t, &w, n, k, quant, glu);
                        assert_eq!(got.len(), want.len());
                        let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                        // bf16 weights have no integer form: the f16 or f32 kernels take them.
                        let tol = if t > 16 && quant == Quant::Bf16 && kind != GemmKind::F32 { 3e-3 } else { tol };
                        let tol = if quant == Quant::Bf16 && t <= 16 { 0.0 } else { tol };
                        let tol = tol.max(1e-4);
                        assert!(
                            err <= tol * scale.max(1.0),
                            "{quant:?} {kind:?} {gemv:?} n={n} k={k} t={t} glu={glu}: err {err} (scale {scale})"
                        );
                    }
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
    // K a multiple of 256 (K-quant super-blocks); K = 4096 and 8192 pad the per-16 scale
    // rows, K = 8192 and 16384 the Q4_K header rows.
    for (n, k) in [(256, 1536), (200, 1536), (128, 8192), (64, 4096), (64, 16384)] {
        for dtype in [GgmlType::Q2K, GgmlType::Q3K, GgmlType::Q4K, GgmlType::Q6K, GgmlType::Q8_0, GgmlType::Q4_0] {
            let bytes = ggml_blocks(dtype, n, k, 3);
            let wd = yaxus_model::ggml::dequantize(dtype, &bytes, n * k);
            for t in [1, 2, 3, 5, 8, 16, 20, 40, 70, 150] {
                for glu in [false, true] {
                    let x = values(t * k, 21 + t as u64);
                    let want = reference(&x, t, &wd, n, k, glu);
                    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
                    let mut cases: Vec<(GemmKind, Option<usize>, f32)> = if t <= 16 {
                        GEMV_KINDS.iter().map(|&(v, tol)| (GemmKind::F32, v, tol)).collect()
                    } else {
                        gemm_kinds(&gpu).into_iter().map(|(g, tol)| (g, None, tol)).collect()
                    };
                    if t > 16 && t <= 48 {
                        // Several integer GEMVs in place of the GEMM.
                        cases.push((GemmKind::Int16, Some(2), 3e-3));
                    }
                    for (kind, gemv, tol) in cases {
                        gpu.set_gemm(kind);
                        gpu.set_gemv(Gemv { planes: gemv, single_f32: false });
                        let got = gpu.debug_matmul_ggml(&x, t, dtype, &bytes, n, k, glu);
                        let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                        // Q4_K/Q6_K scales are rounded to f16 on the GPU (~5e-4 relative).
                        let tol = tol.max(1e-3);
                        assert!(
                            err <= tol * scale.max(1.0),
                            "{dtype:?} {kind:?} {gemv:?} n={n} k={k} t={t} glu={glu}: err {err} (scale {scale})"
                        );
                    }
                }
            }
        }
    }
}

/// With the integer kernels, a row's result is the same bits whether it runs alone or
/// among the rows of a larger pass (so concurrency and speculation cannot change it).
#[test]
#[ignore]
fn integer_rows_do_not_depend_on_the_pass() {
    let mut gpu = WgpuBackend::new().unwrap();
    let (n, k, t) = (200, 1536, 7);
    for dtype in [GgmlType::Q2K, GgmlType::Q3K, GgmlType::Q4K, GgmlType::Q6K, GgmlType::Q8_0, GgmlType::Q4_0] {
        let bytes = ggml_blocks(dtype, n, k, 5);
        let x = values(t * k, 31);
        for planes in [2, 1] {
            gpu.set_gemv(Gemv { planes: Some(planes), single_f32: false });
            for glu in [false, true] {
                let cols = if glu { n / 2 } else { n };
                let all = gpu.debug_matmul_ggml(&x, t, dtype, &bytes, n, k, glu);
                // Alone, and as the second of three rows.
                for j in 0..t {
                    let one = gpu.debug_matmul_ggml(&x[j * k..(j + 1) * k], 1, dtype, &bytes, n, k, glu);
                    assert!(
                        one.iter().zip(&all[j * cols..(j + 1) * cols]).all(|(a, b)| a.to_bits() == b.to_bits()),
                        "{dtype:?} planes={planes} glu={glu}: row {j} alone differs from the 7-row pass"
                    );
                }
                let three = gpu.debug_matmul_ggml(&x[2 * k..5 * k], 3, dtype, &bytes, n, k, glu);
                assert!(
                    three.iter().zip(&all[2 * cols..5 * cols]).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{dtype:?} planes={planes} glu={glu}: a 3-row pass differs from the 7-row pass"
                );
            }
        }
    }
}
