//! Compares the Vulkan backend with the CPU reference on the golden prompts.
//! Needs a GPU and the 2B model: `cargo test --release -p yaxus-backend-wgpu -- --ignored --nocapture`
//! (in a session without the `render` group, run it under `sg render -c '...'`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::Value;
use yaxus_backend_cpu::CpuBackend;
use yaxus_backend_wgpu::WgpuBackend;
use yaxus_model::{Backend, Engine, EngineConfig, Event, Generator, Model, Quant, Request};

fn model_dir() -> PathBuf {
    if let Ok(d) = std::env::var("YAXUS_MODEL_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = Path::new(&home).join(".cache/huggingface/hub/models--pfnet--plamo-3-translate-2b/snapshots");
    std::fs::read_dir(snaps).unwrap().flatten().map(|e| e.path()).next().unwrap()
}

fn cases() -> Vec<Value> {
    // YAXUS_GOLDEN selects another model's file (e.g. logits_8b.jsonl with YAXUS_MODEL_DIR).
    let golden = std::env::var("YAXUS_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/logits_2b.jsonl"));
    std::fs::read_to_string(golden).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn compare(quant: Quant) {
    let dir = model_dir();
    let mut cpu = Model::load(&dir, CpuBackend::new(), quant).unwrap();
    let t0 = Instant::now();
    let mut gpu = Model::load(&dir, WgpuBackend::new().unwrap(), quant).unwrap();
    eprintln!("[{quant:?}] {} loaded in {:.2?}", gpu.backend.name(), t0.elapsed());
    for c in &cases() {
        let name = c["name"].as_str().unwrap();
        let ids = u32s(&c["ids"]);
        let sc = cpu.new_seq().unwrap();
        let lc = cpu.prefill(sc, &ids).unwrap();
        let lc = cpu.backend.to_host(&lc);
        cpu.free_seq(sc);
        let sg = gpu.new_seq().unwrap();
        let lg = gpu.prefill(sg, &ids).unwrap();
        let lg = gpu.backend.to_host(&lg);
        gpu.free_seq(sg);
        assert_eq!(lc.len(), lg.len());
        let max_err = lc.iter().zip(&lg).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b]).then(b.cmp(&a))).unwrap();
        eprintln!("[{quant:?}] {name}: max |cpu - gpu| over all logits = {max_err:.2e}");
        assert_eq!(argmax(&lc), argmax(&lg), "{name}: different next token");
        assert!(max_err < 2e-2, "{name}: logits differ by {max_err}");

        let want = u32s(&c["greedy"]);
        let a = cpu.generate(&ids, want.len(), &[], |_| true).unwrap();
        let t = Instant::now();
        let b = gpu.generate(&ids, want.len(), &[], |_| true).unwrap();
        eprintln!("[{quant:?}] {name}: gpu generated {} tokens in {:.2?}", b.len(), t.elapsed());
        assert_eq!(a, b, "{name}: greedy output differs between CPU and GPU");
    }
}

/// A prompt longer than one prefill chunk (512), so chunking, the multi-submission path
/// and attention over a long context are covered.
#[test]
#[ignore]
fn gpu_matches_cpu_long_prompt() {
    let dir = model_dir();
    let mut cpu = Model::load(&dir, CpuBackend::new(), Quant::Q8).unwrap();
    let mut gpu = Model::load(&dir, WgpuBackend::new().unwrap(), Quant::Q8).unwrap();
    let ids: Vec<u32> = (0..600u32).map(|i| 1000 + (i * 7919) % 50000).collect();
    let sc = cpu.new_seq().unwrap();
    let lc = cpu.prefill(sc, &ids).unwrap();
    let lc = cpu.backend.to_host(&lc);
    let sg = gpu.new_seq().unwrap();
    let lt = gpu.prefill(sg, &ids).unwrap();
    let idx = gpu.backend.argmax(&lt);
    let lg = gpu.backend.to_host(&lt);
    let max_err = lc.iter().zip(&lg).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("600-token prompt: max |cpu - gpu| = {max_err:.2e}");
    // With the f16 prefill GEMM the error is noise of 1e-2..5e-2 that depends on the
    // summation order (attention kernel); with f32 it is ~1e-4.
    assert!(max_err < if gpu.backend.gemm_uses_f16() { 5e-2 } else { 2e-2 });
    // The GPU's softmax probability of its argmax against a host-side computation.
    let m = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let z: f64 = lg.iter().map(|&v| ((v - m) as f64).exp()).sum();
    let want = (1.0 / z) as f32;
    eprintln!("argmax {idx} (host p={want:.5})");
    assert_eq!(lg[idx as usize], m);
    // One decode step at a 601-token context: the split-key attention path.
    let lc = cpu.forward(&[(sc, &[idx])], false).unwrap();
    let lc = cpu.backend.to_host(&lc);
    let lt = gpu.forward(&[(sg, &[idx])], false).unwrap();
    let lg = gpu.backend.to_host(&lt);
    let max_err = lc.iter().zip(&lg).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("decode at 601 tokens: max |cpu - gpu| = {max_err:.2e}");
    // An f16 prefill leaves a slightly different KV cache, which the decode step inherits
    // (noise of up to ~1e-1 whose draw depends on the summation order of the kernels).
    let tol = if gpu.backend.gemm_uses_f16() { 2e-1 } else { 2e-2 };
    assert!(max_err < tol);
}

#[test]
#[ignore]
fn gpu_matches_cpu_q8() {
    compare(Quant::Q8);
}

#[test]
#[ignore]
fn gpu_matches_cpu_bf16() {
    compare(Quant::Bf16);
}

fn draft_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").unwrap();
    let snaps = Path::new(&home).join(".cache/huggingface/hub/models--pfnet--plamo-3-translate-0.8b/snapshots");
    std::fs::read_dir(snaps).ok()?.next()?.ok().map(|e| e.path())
}

/// Speculative decoding must reproduce the target model's plain greedy output exactly.
#[test]
#[ignore]
fn speculative_matches_plain() {
    let dir = model_dir();
    let Some(draft_dir) = draft_dir() else {
        eprintln!("0.8B model not found; skipping");
        return;
    };
    let gpu = WgpuBackend::new().unwrap();
    let mut plain =
        Engine::new(Model::load(&dir, gpu.share(), Quant::Q8).unwrap(), None, EngineConfig::default()).unwrap();
    let mut spec = Engine::new(
        Model::load(&dir, gpu.share(), Quant::Q8).unwrap(),
        Some(Model::load(&draft_dir, gpu.share(), Quant::Q4).unwrap()),
        EngineConfig::default(),
    )
    .unwrap();
    for c in &cases() {
        let ids = u32s(&c["ids"]);
        for k in [1, 3, 5, 7] {
            spec.cfg.draft_k = k;
            let a = plain.generate(&ids, 40, &[1, 16], &mut |_| true).unwrap();
            let b = spec.generate(&ids, 40, &[1, 16], &mut |_| true).unwrap();
            assert_eq!(a, b, "{} k={k}", c["name"]);
            // A budget that ends inside a round.
            let a = plain.generate(&ids, 7, &[], &mut |_| true).unwrap();
            let b = spec.generate(&ids, 7, &[], &mut |_| true).unwrap();
            assert_eq!(a, b, "{} k={k} max_new=7", c["name"]);
        }
    }
    eprintln!("speculative: {:?}", spec.spec);
}

/// Requests run together (batched decode passes, interleaved prefill) must produce what
/// each produces alone, with and without a draft model.
#[test]
#[ignore]
fn concurrent_matches_solo() {
    let dir = model_dir();
    let gpu = WgpuBackend::new().unwrap();
    // f32 prefill: the f16 prefill of a prompt run together with others could take a
    // different kernel than alone (64-row tiles vs. multi-row GEMV).
    gpu.set_gemm(yaxus_backend_wgpu::GemmKind::F32);
    let draft = draft_dir().map(|d| Model::load(&d, gpu.share(), Quant::Q4).unwrap());
    // No response cache: the requests run again after running alone.
    let cfg = EngineConfig { decode_rows: 4, busy_prefill_rows: 16, response_cache: 0, ..EngineConfig::default() };
    let mut eng = Engine::new(Model::load(&dir, gpu.share(), Quant::Q8).unwrap(), draft, cfg).unwrap();
    eng.target.prefix_cache = false;
    if let Some(d) = eng.draft.as_mut() {
        d.prefix_cache = false;
    }
    // The golden prompts plus truncated variants: 9 requests with different lengths.
    let mut prompts: Vec<Vec<u32>> = Vec::new();
    for c in &cases() {
        let ids = u32s(&c["ids"]);
        prompts.push(ids.clone());
        prompts.push(ids[..ids.len() - 3].to_vec());
        prompts.push(ids[..ids.len() / 2].to_vec());
    }
    let solo: Vec<Vec<u32>> = prompts.iter().map(|p| eng.generate(p, 24, &[1, 16], &mut |_| true).unwrap()).collect();
    for (i, p) in prompts.iter().enumerate() {
        eng.submit(Request::new(i as u64, p.clone(), 24, vec![1, 16]));
    }
    let mut outs: Vec<Vec<u32>> = vec![Vec::new(); prompts.len()];
    let mut done = 0;
    let mut steps = 0;
    while done < prompts.len() {
        eng.step(&mut |ev| {
            match ev {
                Event::Token { id, token } => outs[id as usize].push(token),
                Event::Done { .. } => done += 1,
                Event::Error { message, .. } => panic!("{message}"),
            }
            true
        })
        .unwrap();
        steps += 1;
    }
    for i in 0..prompts.len() {
        assert_eq!(outs[i], solo[i], "request {i} differs when run together");
    }
    eprintln!("{} requests together in {steps} steps (speculative: {:?})", prompts.len(), eng.spec);
}

/// Hint tokens, right, partly wrong or unrelated, never change the output; with an exact
/// hint the response cache answers a repeated request without running it.
#[test]
#[ignore]
fn hints_and_response_cache_keep_output() {
    let dir = model_dir();
    let gpu = WgpuBackend::new().unwrap();
    gpu.set_gemm(yaxus_backend_wgpu::GemmKind::F32);
    for with_draft in [false, true] {
        let draft =
            if with_draft { draft_dir().map(|d| Model::load(&d, gpu.share(), Quant::Q4).unwrap()) } else { None };
        if with_draft && draft.is_none() {
            continue;
        }
        let cfg = EngineConfig { response_cache: 0, ..EngineConfig::default() };
        let mut eng = Engine::new(Model::load(&dir, gpu.share(), Quant::Q8).unwrap(), draft, cfg).unwrap();
        eng.target.prefix_cache = false;
        if let Some(d) = eng.draft.as_mut() {
            d.prefix_cache = false;
        }
        let run = |eng: &mut Engine<WgpuBackend>, p: &[u32], max: usize, hint: Vec<u32>| {
            let mut r = Request::new(7, p.to_vec(), max, vec![1, 16]);
            r.hint = hint;
            eng.submit(r);
            let mut out = Vec::new();
            let mut done = false;
            while !done {
                eng.step(&mut |ev| {
                    match ev {
                        Event::Token { token, .. } => out.push(token),
                        Event::Done { .. } => done = true,
                        Event::Error { message, .. } => panic!("{message}"),
                    }
                    true
                })
                .unwrap();
            }
            out
        };
        for c in &cases() {
            let ids = u32s(&c["ids"]);
            let solo = run(&mut eng, &ids, 40, vec![]);
            let mut wrong = solo.clone();
            for j in (2..wrong.len()).step_by(5) {
                wrong[j] = wrong[j].wrapping_add(1);
            }
            let mut shifted = solo[3.min(solo.len())..].to_vec();
            shifted.extend_from_slice(&solo);
            for (name, hint) in
                [("exact", solo.clone()), ("wrong", wrong), ("shifted", shifted), ("other", vec![5, 6, 7, 8])]
            {
                assert_eq!(run(&mut eng, &ids, 40, hint.clone()), solo, "{} hint {name} draft {with_draft}", c["name"]);
                // A budget that ends inside a round.
                assert_eq!(run(&mut eng, &ids, 6, hint), solo[..6.min(solo.len())], "{} hint {name} max 6", c["name"]);
            }
        }
        eprintln!("draft {with_draft}: {:?}", eng.spec);
        // The response cache: a repeated request gives the same tokens without a pass.
        eng.cfg.response_cache = 8;
        let ids = u32s(&cases()[0]["ids"]);
        let first = run(&mut eng, &ids, 40, vec![]);
        let rounds = eng.spec.rounds;
        assert_eq!(run(&mut eng, &ids, 40, vec![]), first);
        assert_eq!(run(&mut eng, &ids, 5, vec![]), first[..5.min(first.len())]);
        assert_eq!(eng.spec.cache_hits, 2);
        assert_eq!(eng.spec.rounds, rounds);
    }
}

/// The tiled prefill attention (`attn_tile`, with and without key chunks) against the
/// CPU and the per-row kernel: 40 rows after a 2600-token prompt, past the 2048-token
/// sliding window. f32 GEMM so only the attention kernels differ.
#[test]
#[ignore]
fn tiled_attention_matches_cpu() {
    let dir = model_dir();
    let mut cpu = Model::load(&dir, CpuBackend::new(), Quant::Q8).unwrap();
    let gpu = WgpuBackend::new().unwrap();
    gpu.set_gemm(yaxus_backend_wgpu::GemmKind::F32);
    let mut gpu = Model::load(&dir, gpu, Quant::Q8).unwrap();
    gpu.prefix_cache = false;
    cpu.prefix_cache = false;
    let ids: Vec<u32> = (0..2640u32).map(|i| 1000 + (i * 7919) % 50000).collect();
    let (a, b) = ids.split_at(2600);
    let sc = cpu.new_seq().unwrap();
    cpu.prefill(sc, a).unwrap();
    let l = cpu.forward(&[(sc, b)], true).unwrap();
    let lc = cpu.backend.to_host(&l);
    for (name, wgs) in [("per-row", 0), ("tiled", 384), ("tiled, split keys", 100_000)] {
        gpu.backend.set_tiled_attention(wgs);
        let sg = gpu.new_seq().unwrap();
        gpu.prefill(sg, a).unwrap();
        let l = gpu.forward(&[(sg, b)], true).unwrap();
        let lg = gpu.backend.to_host(&l);
        gpu.free_seq(sg);
        let e = lc.iter().zip(&lg).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        eprintln!("{name}: 40 rows after 2600: max |cpu - gpu| = {e:.2e}");
        assert!(e < 2e-3, "{name}: {e}");
    }
}

/// Decode steps (the split attention, 16-key chunks for short and 32-key chunks for long
/// contexts, and its merge) against the CPU with the f32 prefill GEMM, teacher-forced.
#[test]
#[ignore]
fn decode_attention_matches_cpu() {
    let dir = model_dir();
    let mut cpu = Model::load(&dir, CpuBackend::new(), Quant::Q8).unwrap();
    let mut gpu = WgpuBackend::new().unwrap();
    gpu.set_gemm(yaxus_backend_wgpu::GemmKind::F32);
    gpu.set_gemv(yaxus_backend_wgpu::Gemv::F32);
    let mut gpu = Model::load(&dir, gpu, Quant::Q8).unwrap();
    for n in [20u32, 600, 2100] {
        let ids: Vec<u32> = (0..n).map(|i| 1000 + (i * 7919) % 50000).collect();
        let sc = cpu.new_seq().unwrap();
        let sg = gpu.new_seq().unwrap();
        cpu.prefill(sc, &ids).unwrap();
        gpu.prefill(sg, &ids).unwrap();
        let mut worst = 0f32;
        for step in 0..4u32 {
            let tok = 2000 + step * 37;
            let lc = cpu.forward(&[(sc, &[tok])], false).unwrap();
            let lc = cpu.backend.to_host(&lc);
            let lg = gpu.forward(&[(sg, &[tok])], false).unwrap();
            let lg = gpu.backend.to_host(&lg);
            worst = worst.max(lc.iter().zip(&lg).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max));
        }
        cpu.free_seq(sc);
        gpu.free_seq(sg);
        eprintln!("{n}-token context: worst |cpu - gpu| over 4 decode steps = {worst:.2e}");
        assert!(worst < 2e-3, "{n}: {worst}");
    }
}

/// How far each prefill GEMM kind moves the logits of the first generated token from the
/// f32 GEMM's, on the golden prompts and on a 600-token one: the largest difference and
/// the KL divergence of the distributions. `YAXUS_MODEL_DIR` may name a GGUF file.
#[test]
#[ignore]
fn prefill_kinds_against_f32() {
    use yaxus_backend_wgpu::GemmKind;
    let dir = model_dir();
    let quant = if yaxus_model::gguf::is_gguf(&dir) { Quant::Auto } else { Quant::Q8 };
    let mut gpu = Model::load(&dir, WgpuBackend::new().unwrap(), quant).unwrap();
    gpu.prefix_cache = false;
    // (prompts of up to 16 rows would take the decode kernels)
    gpu.backend.set_gemv(yaxus_backend_wgpu::Gemv::F32);
    let mut prompts: Vec<Vec<u32>> = cases().iter().map(|c| u32s(&c["ids"])).collect();
    prompts.push((0..600u32).map(|i| 1000 + (i * 7919) % 50000).collect());
    let logits = |gpu: &mut Model<WgpuBackend>, kind| -> Vec<Vec<f32>> {
        gpu.backend.set_gemm(kind);
        prompts
            .iter()
            .map(|ids| {
                let s = gpu.new_seq().unwrap();
                let l = gpu.prefill(s, ids).unwrap();
                let l = gpu.backend.to_host(&l);
                gpu.free_seq(s);
                l
            })
            .collect()
    };
    let softmax = |l: &[f32]| -> Vec<f64> {
        let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f64> = l.iter().map(|&v| ((v - m) as f64).exp()).collect();
        let z: f64 = e.iter().sum();
        e.into_iter().map(|v| v / z).collect()
    };
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let base = logits(&mut gpu, GemmKind::F32);
    let mut kinds = vec![(GemmKind::Int16, 5e-2), (GemmKind::Int8, 3.0)];
    if gpu.backend.has_matrix_cores() {
        kinds.insert(0, (GemmKind::Coop, 3e-1));
    }
    for (kind, bound) in kinds {
        let got = logits(&mut gpu, kind);
        let (mut worst, mut kl_max, mut same) = (0f32, 0f64, 0);
        for (a, b) in base.iter().zip(&got) {
            worst = worst.max(a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max));
            let (p, q) = (softmax(a), softmax(b));
            let kl: f64 = p.iter().zip(&q).filter(|(p, _)| **p > 0.0).map(|(p, q)| p * (p / q.max(1e-300)).ln()).sum();
            kl_max = kl_max.max(kl);
            same += usize::from(argmax(a) == argmax(b));
        }
        eprintln!(
            "{kind:?}: max |logit - f32| = {worst:.2e}, max KL = {kl_max:.2e}, same first token {same}/{}",
            base.len()
        );
        assert!(worst < bound, "{kind:?}: {worst}");
    }
}

/// How far the integer decode kernels move the logits from the f32 GEMV's: 8 teacher-forced
/// decode steps after an f32 prefill of the golden prompts and of a 600-token one.
/// `YAXUS_MODEL_DIR` may name a GGUF file.
#[test]
#[ignore]
fn decode_kinds_against_f32() {
    use yaxus_backend_wgpu::{GemmKind, Gemv};
    let dir = model_dir();
    let quant = if yaxus_model::gguf::is_gguf(&dir) { Quant::Auto } else { Quant::Q8 };
    let mut gpu = Model::load(&dir, WgpuBackend::new().unwrap(), quant).unwrap();
    gpu.prefix_cache = false;
    gpu.backend.set_gemm(GemmKind::F32);
    let mut prompts: Vec<Vec<u32>> = cases().iter().map(|c| u32s(&c["ids"])).collect();
    prompts.push((0..600u32).map(|i| 1000 + (i * 7919) % 50000).collect());
    let logits = |gpu: &mut Model<WgpuBackend>, gemv: Gemv| -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        for ids in &prompts {
            gpu.backend.set_gemv(Gemv::F32);
            let s = gpu.new_seq().unwrap();
            gpu.prefill(s, ids).unwrap();
            gpu.backend.set_gemv(gemv);
            for step in 0..8u32 {
                let l = gpu.forward(&[(s, &[2000 + step * 37])], false).unwrap();
                out.push(gpu.backend.to_host(&l));
            }
            gpu.free_seq(s);
        }
        out
    };
    let softmax = |l: &[f32]| -> Vec<f64> {
        let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f64> = l.iter().map(|&v| ((v - m) as f64).exp()).collect();
        let z: f64 = e.iter().sum();
        e.into_iter().map(|v| v / z).collect()
    };
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let base = logits(&mut gpu, Gemv::F32);
    for (gemv, bound) in [(Gemv::INT16, 5e-2), (Gemv { planes: Some(1), single_f32: false }, 3.0)] {
        let got = logits(&mut gpu, gemv);
        let (mut worst, mut kl_max, mut same) = (0f32, 0f64, 0);
        for (a, b) in base.iter().zip(&got) {
            worst = worst.max(a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max));
            let (p, q) = (softmax(a), softmax(b));
            let kl: f64 = p.iter().zip(&q).filter(|(p, _)| **p > 0.0).map(|(p, q)| p * (p / q.max(1e-300)).ln()).sum();
            kl_max = kl_max.max(kl);
            same += usize::from(argmax(a) == argmax(b));
        }
        eprintln!(
            "{:?} planes: max |logit - f32| = {worst:.2e}, max KL = {kl_max:.2e}, same token {same}/{}",
            gemv.planes,
            base.len()
        );
        assert!(worst < bound, "{gemv:?}: {worst}");
    }
}
