//! Compares the Vulkan backend with the CPU reference on the golden prompts.
//! Needs a GPU and the 2B model: `cargo test --release -p yakusel-backend-wgpu -- --ignored --nocapture`
//! (in a session without the `render` group, run it under `sg render -c '...'`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::Value;
use yakusel_backend_cpu::CpuBackend;
use yakusel_backend_wgpu::WgpuBackend;
use yakusel_model::{Backend, Engine, EngineConfig, Event, Generator, Model, Quant, Request};

fn model_dir() -> PathBuf {
    if let Ok(d) = std::env::var("YAKUSEL_MODEL_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap();
    let snaps = Path::new(&home).join(".cache/huggingface/hub/models--pfnet--plamo-3-translate-2b/snapshots");
    std::fs::read_dir(snaps).unwrap().flatten().map(|e| e.path()).next().unwrap()
}

fn cases() -> Vec<Value> {
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/logits_2b.jsonl");
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
    assert!(max_err < 2e-2);
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
    // An f16 prefill leaves a slightly different KV cache, which the decode step inherits.
    let tol = if gpu.backend.gemm_uses_f16() { 1e-1 } else { 2e-2 };
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
    let mut plain = Engine::new(Model::load(&dir, gpu.share(), Quant::Q8).unwrap(), None, EngineConfig::default()).unwrap();
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
    gpu.set_gemm(yakusel_backend_wgpu::GemmKind::F32);
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
    gpu.set_gemm(yakusel_backend_wgpu::GemmKind::F32);
    for with_draft in [false, true] {
        let draft = if with_draft { draft_dir().map(|d| Model::load(&d, gpu.share(), Quant::Q4).unwrap()) } else { None };
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
            for (name, hint) in [("exact", solo.clone()), ("wrong", wrong), ("shifted", shifted), ("other", vec![5, 6, 7, 8])] {
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

/// An f16 KV cache against the CPU (f32), next to the f32 KV cache: logits after a
/// prompt and after decode steps, and greedy output on the golden prompts. The prefill
/// GEMM is f32 so only the KV storage differs.
#[test]
#[ignore]
fn f16_kv_matches_cpu() {
    let dir = model_dir();
    let mut cpu = Model::load(&dir, CpuBackend::new(), Quant::Q8).unwrap();
    let gpu = WgpuBackend::new().unwrap();
    gpu.set_gemm(yakusel_backend_wgpu::GemmKind::F32);
    let mut gpu = Model::load(&dir, gpu, Quant::Q8).unwrap();
    gpu.prefix_cache = false;
    let err = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let ids: Vec<u32> = (0..600u32).map(|i| 1000 + (i * 7919) % 50000).collect();
    // CPU reference: the prompt's logits and 8 greedy decode steps.
    let sc = cpu.new_seq().unwrap();
    let l = cpu.prefill(sc, &ids).unwrap();
    let mut want = vec![cpu.backend.to_host(&l)];
    let argmax = |v: &[f32]| (0..v.len()).max_by(|&a, &b| v[a].total_cmp(&v[b]).then(b.cmp(&a))).unwrap() as u32;
    let mut toks = Vec::new();
    for _ in 0..8 {
        let t = argmax(want.last().unwrap());
        toks.push(t);
        let l = cpu.forward(&[(sc, &[t])], false).unwrap();
        want.push(cpu.backend.to_host(&l));
    }
    let mut worst = [0f32; 2];
    for f16 in [false, true] {
        gpu.backend.set_kv_f16(f16);
        gpu.init_kv(4096, 4096).unwrap();
        let sg = gpu.new_seq().unwrap();
        let l = gpu.prefill(sg, &ids).unwrap();
        let mut errs = vec![err(&want[0], &gpu.backend.to_host(&l))];
        // Teacher-forced with the CPU's tokens, so every step compares the same context.
        for (j, &t) in toks.iter().enumerate() {
            let l = gpu.forward(&[(sg, &[t])], false).unwrap();
            let lg = gpu.backend.to_host(&l);
            errs.push(err(&want[j + 1], &lg));
        }
        gpu.free_seq(sg);
        eprintln!("KV {}: max |cpu - gpu| prompt {:.2e}, worst of 8 decode steps {:.2e}", if f16 { "f16" } else { "f32" }, errs[0], errs[1..].iter().cloned().fold(0f32, f32::max));
        worst[f16 as usize] = errs.iter().cloned().fold(0f32, f32::max);
        for c in &cases() {
            let ids = u32s(&c["ids"]);
            let want = u32s(&c["greedy"]);
            let got = gpu.generate(&ids, want.len(), &[], |_| true).unwrap();
            assert_eq!(got, want, "{}: greedy output differs (f16 KV: {f16})", c["name"]);
        }
    }
    // Measured: f32 KV ~8e-5, f16 KV ~5e-2 (the size of the f16 prefill GEMM's error).
    assert!(worst[0] < 1e-3, "f32 KV error {:.2e}", worst[0]);
    assert!(worst[1] < 1e-1, "f16 KV error {:.2e}", worst[1]);
}
