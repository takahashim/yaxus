//! Matmul kernels (every path: gemv1, gemvm2..8, gemm, glu) against a host reference
//! computed from the same quantized weights. Needs a GPU (Vulkan, or Metal on macOS):
//! `cargo test --release -p yaxus-backend-gpu --test kernels -- --ignored`
//! or a software one (lavapipe, as in CI; without 32-lane subgroups only the f32 decode
//! kernels and the prefill GEMMs run): `YAXUS_SOFTWARE_GPU=1
//! VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json` in front.

use yaxus_backend_gpu::{GemmKind, Gemv, GpuBackend};
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
fn gemm_kinds(gpu: &GpuBackend) -> Vec<(GemmKind, f32)> {
    let mut kinds = vec![(GemmKind::F32, 0.0), (GemmKind::Int16, 3e-3), (GemmKind::Int8, 3e-2)];
    if gpu.caps().coop {
        kinds.push((GemmKind::Coop, 3e-3)); // (the default where matrix cores exist)
        // (on Metal f16 tiles, rounding weights and activations; on Vulkan as `Coop`)
        kinds.push((GemmKind::CoopF16, 3e-3));
    }
    kinds
}

/// The kernels for up to 16 rows (`set_gemv`) and the error each may add.
const GEMV_KINDS: [(Option<usize>, f32); 3] = [(None, 0.0), (Some(2), 3e-3), (Some(1), 3e-2)];

/// The device the tests run on and what it supports (CI prints it with `--nocapture`).
/// The other tests skip what the device lacks; `YAXUS_EXPECT_SUBGROUPS=1` makes their
/// absence an error instead, so a runner whose device lost them does not pass silently.
#[test]
#[ignore]
fn reports_the_device() {
    let gpu = GpuBackend::new().unwrap();
    let caps = gpu.caps();
    eprintln!("{caps:#?}\n{:#?}", gpu.tuning());
    if std::env::var_os("YAXUS_EXPECT_SUBGROUPS").is_some_and(|v| v == "1") {
        assert!(caps.subgroups, "{}: no 32-lane subgroups, so the integer decode kernels went untested", caps.name);
    }
}

#[test]
#[ignore]
fn matmul_kernels_match_host() {
    let mut gpu = GpuBackend::new().unwrap();
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

/// `add_norm` against the host: `x += norm(delta) * (o1 + w1)`, then
/// `h = norm(x) * (o2 + w2)`. Widths up to 4096 take the subgroup kernel where there is
/// one, wider rows (and every width without subgroups) the plain one.
#[test]
#[ignore]
fn add_norm_matches_host() {
    let mut gpu = GpuBackend::new().unwrap();
    let (o1, o2, eps) = (1.0f32, 1.0f32, 1e-6f32);
    let norm = |v: &[f32]| 1.0 / (v.iter().map(|a| a * a).sum::<f32>() / v.len() as f32 + eps).sqrt();
    for cols in [64, 1000, 2560, 4096, 4352, 6000] {
        for rows in [1, 3] {
            let x = values(rows * cols, cols as u64);
            let d = values(rows * cols, cols as u64 + 1);
            let (w1, w2) = (values(cols, 3), values(cols, 4));
            let (gx, gh) = gpu.debug_add_norm(&x, &d, rows, cols, (&w1, o1), (&w2, o2), eps);
            for r in 0..rows {
                let row = r * cols..(r + 1) * cols;
                let r1 = norm(&d[row.clone()]);
                let wx: Vec<f32> = (0..cols).map(|i| x[r * cols + i] + (o1 + w1[i]) * (d[r * cols + i] * r1)).collect();
                let r2 = norm(&wx);
                for i in 0..cols {
                    let wh = (o2 + w2[i]) * (wx[i] * r2);
                    let (ex, eh) = ((gx[r * cols + i] - wx[i]).abs(), (gh[r * cols + i] - wh).abs());
                    assert!(ex <= 1e-4 * wx[i].abs().max(1.0), "x cols={cols} row {r} [{i}]");
                    assert!(eh <= 1e-4 * wh.abs().max(1.0), "h cols={cols} row {r} [{i}]");
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
    let mut gpu = GpuBackend::new().unwrap();
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
    let mut gpu = GpuBackend::new().unwrap();
    if !gpu.caps().subgroups {
        eprintln!("no 32-lane subgroups (the integer decode kernels need them); skipping");
        return;
    }
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

/// With the f32 kernels of `gemv_f32` (Apple GPUs), a row's result is the same bits
/// whether it runs alone or among the rows of a larger pass, as with the integer kernels.
/// Elsewhere the f32 GEMVs make no such promise: the test only reports how many differ.
#[test]
#[ignore]
fn f32_rows_do_not_depend_on_the_pass() {
    let mut gpu = GpuBackend::new().unwrap();
    let strict = gpu.tuning().gemv_f32;
    gpu.set_gemm(GemmKind::F32);
    gpu.set_gemv(Gemv::F32);
    let (n, k) = (200, 1536);
    for dtype in [GgmlType::Q2K, GgmlType::Q3K, GgmlType::Q4K, GgmlType::Q6K, GgmlType::Q8_0, GgmlType::Q4_0] {
        let bytes = ggml_blocks(dtype, n, k, 5);
        let x = values(16 * k, 31);
        for glu in [false, true] {
            let cols = if glu { n / 2 } else { n };
            let all = gpu.debug_matmul_ggml(&x, 16, dtype, &bytes, n, k, glu);
            let mut differ = 0;
            // Alone, and as rows of passes of 3 and 7.
            for (start, t) in [(0, 1), (5, 1), (15, 1), (2, 3), (8, 7)] {
                let part = gpu.debug_matmul_ggml(&x[start * k..(start + t) * k], t, dtype, &bytes, n, k, glu);
                let want = &all[start * cols..(start + t) * cols];
                differ += part.iter().zip(want).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            }
            if strict {
                assert_eq!(differ, 0, "{dtype:?} glu={glu}: rows differ from the 16-row pass");
            } else if differ > 0 {
                eprintln!("{dtype:?} glu={glu}: {differ} outputs differ from the 16-row pass (no gemv_f32 here)");
            }
        }
    }
}

/// Argmax over a whole pass of rows (more than one workgroup's worth of partials), with
/// the CPU's choice on degenerate rows.
#[test]
#[ignore]
fn argmax_handles_a_whole_pass() {
    let mut gpu = GpuBackend::new().unwrap();
    let cols = 1000;
    // 1100: more than one dispatch's 512 rows, done in batches.
    for rows in [1, 64, 65, 300, 512, 1100] {
        let mut x = values(rows * cols, rows as u64);
        let want: Vec<u32> = (0..rows).map(|r| ((r * 37) % cols) as u32).collect();
        for (r, &w) in want.iter().enumerate() {
            x[r * cols + w as usize] = 5.0;
        }
        assert_eq!(gpu.debug_argmax(&x, rows), want, "{rows} rows");
    }
    // A row of only -inf (or NaN) gives token 0, like the CPU, not an id beyond it.
    let mut x = vec![f32::NEG_INFINITY; 2 * cols];
    x[cols..].fill(f32::NAN);
    assert_eq!(gpu.debug_argmax(&x, 2), [0, 0]);
}

/// A model whose layers are all full attention (or all sliding) gets a KV cache too.
#[test]
#[ignore]
fn kv_cache_for_a_single_kind_of_layer() {
    use yaxus_model::{Backend, Config, LayerMeta, Rope};
    let mut gpu = GpuBackend::new().unwrap();
    let cfg: Config = serde_json::from_str(
        r#"{"hidden_size": 256, "num_hidden_layers": 2, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 128, "intermediate_size": 256,
            "vocab_size": 300, "rms_norm_eps": 1e-6, "window_size": 64,
            "sliding_window_pattern": 1, "rope_theta": 10000.0, "rope_local_theta": 10000.0}"#,
    )
    .unwrap();
    let rope = Rope { inv_freq: vec![0.5; 64], concentration: 1.0 };
    for full in [true, false] {
        let metas: Vec<LayerMeta> = (0..2)
            .map(|index| LayerMeta { index, full_attn: full, window: (!full).then_some(64), rope: rope.clone() })
            .collect();
        assert!(gpu.new_kv(&cfg, &metas, 2, 512).is_ok(), "full_attn = {full}");
    }
}

/// A buffer the device cannot create is an error, not a panic in wgpu's handler.
#[test]
#[ignore]
fn failed_allocations_are_errors() {
    let mut gpu = GpuBackend::new().unwrap();
    let too_big = gpu.max_buffer_size() + 4;
    assert!(gpu.debug_alloc(too_big).is_err());
    // The device is still usable afterwards.
    assert!(gpu.debug_alloc(1 << 20).is_ok());
    assert_eq!(gpu.debug_argmax(&[0.0, 1.0, 0.5], 1), [1]);
}

/// The embedding kernels read a Q4_K table (the browser's 0.8B path) and a bf16 table
/// row for row.
#[test]
#[ignore]
fn embedding_rows_match_the_table() {
    let mut gpu = GpuBackend::new().unwrap();
    let (rows, h) = (3, 512);
    let ids = [2u32, 0, 1];

    let bytes = ggml_blocks(GgmlType::Q4K, rows, h, 7);
    let all = yaxus_model::ggml::dequantize(GgmlType::Q4K, &bytes, rows * h);
    let want: Vec<f32> = ids.iter().flat_map(|&i| all[i as usize * h..][..h].to_vec()).collect();
    let got = gpu.debug_embed(&bytes, GgmlType::Q4K, h, &ids);
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1.0);
    let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(err <= 1e-4 * scale, "Q4_K embedding: err {err} (scale {scale})");

    let bf: Vec<u16> = values(rows * h, 11).iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
    let bytes: Vec<u8> = bf.iter().flat_map(|u| u.to_le_bytes()).collect();
    let want: Vec<f32> =
        ids.iter().flat_map(|&i| bf[i as usize * h..][..h].iter().map(|&u| f32::from_bits((u as u32) << 16))).collect();
    let got = gpu.debug_embed(&bytes, GgmlType::Bf16, h, &ids);
    assert_eq!(got, want);
}
