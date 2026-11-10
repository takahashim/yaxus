//! Compares the Vulkan backend with the CPU reference on the golden prompts.
//! Needs a GPU and the 2B model: `cargo test --release -p yakusel-backend-wgpu -- --ignored --nocapture`
//! (in a session without the `render` group, run it under `sg render -c '...'`).

use std::path::{Path, PathBuf};
use std::time::Instant;

use serde_json::Value;
use yakusel_backend_cpu::CpuBackend;
use yakusel_backend_wgpu::WgpuBackend;
use yakusel_model::speculative::Speculative;
use yakusel_model::{Backend, Model, Quant};

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

/// Speculative decoding must reproduce the target model's plain greedy output exactly.
#[test]
#[ignore]
fn speculative_matches_plain() {
    let dir = model_dir();
    let draft_dir = {
        let home = std::env::var("HOME").unwrap();
        let snaps = Path::new(&home).join(".cache/huggingface/hub/models--pfnet--plamo-3-translate-0.8b/snapshots");
        match std::fs::read_dir(snaps) {
            Ok(mut d) => d.next().unwrap().unwrap().path(),
            Err(_) => {
                eprintln!("0.8B model not found; skipping");
                return;
            }
        }
    };
    let gpu = WgpuBackend::new().unwrap();
    let mut plain = Model::load(&dir, gpu.share(), Quant::Q8).unwrap();
    let mut spec = Speculative::new(
        Model::load(&dir, gpu.share(), Quant::Q8).unwrap(),
        Model::load(&draft_dir, gpu.share(), Quant::Q4).unwrap(),
        5,
    )
    .unwrap();
    for c in &cases() {
        let ids = u32s(&c["ids"]);
        for k in [1, 3, 5, 7] {
            spec.k = k;
            let a = plain.generate(&ids, 40, &[1, 16], |_| true).unwrap();
            let b = spec.generate(&ids, 40, &[1, 16], |_| true).unwrap();
            assert_eq!(a, b, "{} k={k}", c["name"]);
            // A budget that ends inside a round.
            let a = plain.generate(&ids, 7, &[], |_| true).unwrap();
            let b = spec.generate(&ids, 7, &[], |_| true).unwrap();
            assert_eq!(a, b, "{} k={k} max_new=7", c["name"]);
        }
    }
    eprintln!("speculative: {:?}", spec.stats);
}
