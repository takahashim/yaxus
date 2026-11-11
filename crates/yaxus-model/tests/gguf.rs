//! GGUF files against the HF checkpoint they were converted from: every tensor,
//! dequantized on the CPU, must be close to the bf16 original (within its type's
//! quantization error). Needs the 0.8B model and its GGUF files in the HF cache.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use yaxus_model::Weights;
use yaxus_model::ggml::GgmlType;
use yaxus_model::gguf::Gguf;

fn snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let snaps = Path::new(&home).join(format!(".cache/huggingface/hub/models--{}/snapshots", repo.replace('/', "--")));
    std::fs::read_dir(snaps).ok()?.flatten().map(|e| e.path()).next()
}

/// Relative L2 error allowed per type (measured errors are well below these).
fn bound(t: GgmlType) -> f64 {
    match t {
        GgmlType::F32 | GgmlType::F16 | GgmlType::Bf16 => 1e-3,
        GgmlType::Q8_0 => 0.01,
        GgmlType::Q6K => 0.03,
        GgmlType::Q5K | GgmlType::Q5_0 | GgmlType::Q5_1 => 0.06,
        GgmlType::Q4K | GgmlType::Q4_0 | GgmlType::Q4_1 => 0.12,
        GgmlType::Q3K => 0.25,
        GgmlType::Q2K => 0.5,
    }
}

#[test]
#[ignore]
fn gguf_tensors_match_checkpoint() {
    let (Some(hf), Some(gg)) = (snapshot("pfnet/plamo-3-translate-0.8b"), snapshot("pfnet/plamo-3-translate-0.8b-GGUF")) else {
        eprintln!("0.8B checkpoint or GGUF files not found; skipping");
        return;
    };
    let reference = Weights::open(&hf).unwrap();
    let hf_cfg = yaxus_model::Config::from_file(hf.join("config.json")).unwrap();
    let mut files: Vec<PathBuf> =
        std::fs::read_dir(&gg).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "gguf")).collect();
    files.sort();
    assert!(!files.is_empty());
    for f in files {
        let g = Gguf::open(&f).unwrap();
        let cfg = g.config().unwrap();
        assert_eq!((cfg.hidden_size, cfg.num_hidden_layers, cfg.intermediate_size), (hf_cfg.hidden_size, hf_cfg.num_hidden_layers, hf_cfg.intermediate_size));
        let w = g.weights().unwrap();
        let mut worst: HashMap<GgmlType, (f64, String)> = HashMap::new();
        for name in w.names() {
            let t = w.get(name).unwrap();
            let r = reference.get(name).unwrap();
            assert_eq!(t.shape, r.shape, "{name}");
            let (a, b) = (t.to_f32_vec(), r.to_f32_vec());
            let num: f64 = a.iter().zip(&b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
            let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
            let rel = (num / den.max(1e-30)).sqrt();
            assert!(rel < bound(t.dtype), "{}: {name} ({:?}) relative error {rel:.4}", f.display(), t.dtype);
            let e = worst.entry(t.dtype).or_insert((0.0, String::new()));
            if rel > e.0 {
                *e = (rel, name.to_string());
            }
        }
        let mut v: Vec<_> = worst.into_iter().collect();
        v.sort_by_key(|(t, _)| format!("{t:?}"));
        eprintln!("{}: {:?}", f.file_name().unwrap().to_string_lossy(), v.iter().map(|(t, (e, _))| format!("{t:?} {e:.4}")).collect::<Vec<_>>());
    }
}
