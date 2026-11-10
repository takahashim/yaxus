//! Compares the CPU backend against `tests/golden/logits_2b.jsonl` (HF transformers, f32).
//! Slow (loads the 2B model); run with `cargo test --release -p yakusel-backend-cpu -- --ignored`.

use std::path::{Path, PathBuf};

use serde_json::Value;
use yakusel_backend_cpu::CpuBackend;
use yakusel_model::{Backend, Model, Quant};

fn model_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("YAKUSEL_MODEL_DIR") {
        return Some(PathBuf::from(d));
    }
    let home = std::env::var("HOME").ok()?;
    let snaps = Path::new(&home)
        .join(".cache/huggingface/hub/models--pfnet--plamo-3-translate-2b/snapshots");
    std::fs::read_dir(snaps).ok()?.flatten().map(|e| e.path()).next()
}

fn u32s(v: &Value) -> Vec<u32> {
    v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn cases() -> Vec<Value> {
    let golden = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/golden/logits_2b.jsonl");
    std::fs::read_to_string(golden)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// Q8 is lossy: require the same top-1 next token, report how far greedy output agrees.
#[test]
#[ignore]
fn q8_quality() {
    let dir = model_dir().expect("model dir");
    let mut model = Model::load(&dir, CpuBackend::new(), Quant::Q8).unwrap();
    for c in &cases() {
        let name = c["name"].as_str().unwrap();
        let ids = u32s(&c["ids"]);
        let s = model.new_seq().unwrap();
        let logits = model.prefill(s, &ids).unwrap();
        let logits = model.backend.to_host(&logits);
        model.free_seq(s);
        let top_ids = u32s(&c["top_ids"]);
        let top_vals: Vec<f64> = c["top_logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let max_err = top_ids
            .iter()
            .zip(&top_vals)
            .map(|(&id, &v)| (logits[id as usize] as f64 - v).abs())
            .fold(0.0, f64::max);
        let want = u32s(&c["greedy"]);
        let got = model.generate(&ids, want.len(), &[], |_| true).unwrap();
        let agree = got.iter().zip(&want).take_while(|(a, b)| a == b).count();
        eprintln!(
            "{name}: top20 max abs err {max_err:.3}, greedy agrees for {agree}/{} tokens (exact: {})",
            want.len(),
            got[..got.len().min(want.len())] == want[..got.len().min(want.len())]
        );
        let argmax = (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap();
        assert_eq!(argmax as u32, top_ids[0], "{name}: Q8 changes the first token");
    }
}

#[test]
#[ignore]
fn golden_logits_and_greedy() {
    let dir = model_dir().expect("model dir");
    let cases = cases();
    let mut model = Model::load(&dir, CpuBackend::new(), Quant::Bf16).unwrap();
    for c in &cases {
        let name = c["name"].as_str().unwrap();
        let ids = u32s(&c["ids"]);
        let s = model.new_seq().unwrap();
        let logits = model.prefill(s, &ids).unwrap();
        let logits = model.backend.to_host(&logits);
        model.free_seq(s);

        let top_ids = u32s(&c["top_ids"]);
        let top_vals: Vec<f64> = c["top_logits"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
        let mut max_err = 0f64;
        for (&id, &want) in top_ids.iter().zip(&top_vals) {
            max_err = max_err.max((logits[id as usize] as f64 - want).abs());
        }
        let sum: f64 = logits.iter().map(|&x| x as f64).sum();
        let sq: f64 = logits.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let want_sq = c["logits_sq_sum"].as_f64().unwrap();
        eprintln!(
            "{name}: top20 max abs err {max_err:.2e}, sum {sum:.3} (want {:.3}), sq-sum rel err {:.2e}",
            c["logits_sum"].as_f64().unwrap(),
            (sq - want_sq).abs() / want_sq
        );
        assert!(max_err < 1e-2, "{name}: top-20 logits differ by {max_err}");

        let want = u32s(&c["greedy"]);
        let got = model.generate(&ids, want.len(), &[], |_| true).unwrap();
        // HF stops on eos (1/16) and includes it; compare up to and including the stop token.
        let n = want.len().min(got.len());
        assert_eq!(&got[..n], &want[..n], "{name}: greedy output differs");
    }
}
