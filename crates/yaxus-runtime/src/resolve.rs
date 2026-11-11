//! Finding models on disk: directories, GGUF files, and names in the local Hugging Face
//! cache (`2b`, `pfnet/plamo-3-translate-2b`, `2b:Q4_K_M`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Resolves a model directory or a Hugging Face repo id in the local cache.
/// Returns (directory, display name).
pub fn resolve_model(model: &str) -> Result<(PathBuf, String)> {
    const SIZES: [&str; 4] = ["0.8b", "2b", "8b", "31b"];
    let p = Path::new(model);
    if p.is_file() {
        anyhow::ensure!(yaxus_model::gguf::is_gguf(p), "{model} is not a GGUF file");
        let name = p.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| model.to_string());
        return Ok((p.to_path_buf(), name));
    }
    let size = model.to_ascii_lowercase();
    if !p.is_dir() && SIZES.contains(&size.as_str()) {
        return resolve_model(&format!("pfnet/plamo-3-translate-{size}"));
    }
    // REPO:QUANT or SIZE:QUANT: one file of a GGUF repository.
    if !p.exists()
        && let Some((repo, quant)) = model.rsplit_once(':')
    {
        let repo = if SIZES.contains(&repo.to_ascii_lowercase().as_str()) {
            format!("pfnet/plamo-3-translate-{}-GGUF", repo.to_ascii_lowercase())
        } else {
            repo.to_string()
        };
        let dir = hf_snapshot(&repo)
            .with_context(|| format!("download it with `hf download {repo} --include '*{quant}.gguf'`"))?;
        let want = format!("{}.gguf", quant.to_ascii_lowercase());
        let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)?
            .flatten()
            .map(|e| e.path())
            .filter(|f| f.file_name().is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().ends_with(&want)))
            .collect();
        found.sort();
        let file = found.into_iter().next().with_context(|| {
            format!(
                "no *{quant}.gguf in {}; download it with `hf download {repo} --include '*{quant}.gguf'`",
                dir.display()
            )
        })?;
        return resolve_model(file.to_str().context("non-UTF-8 path")?);
    }
    if p.is_dir() {
        let name = p
            .ancestors()
            .filter_map(|a| a.file_name()?.to_str())
            .find_map(|n| n.strip_prefix("models--").map(|r| r.rsplit("--").next().unwrap_or(r).to_string()))
            .or_else(|| p.file_name().map(|n| n.to_string_lossy().into_owned()))
            .unwrap_or_else(|| model.to_string());
        return Ok((p.to_path_buf(), name));
    }
    let snapshot = hf_snapshot(model)?;
    if !snapshot.join("config.json").exists() {
        let mut quants: Vec<String> = std::fs::read_dir(&snapshot)?
            .flatten()
            .filter_map(|e| {
                let n = e.file_name().to_string_lossy().into_owned();
                let stem = n.strip_suffix(".gguf")?;
                Some(format!("{model}:{}", stem.rsplit('-').next().unwrap_or(stem)))
            })
            .collect();
        quants.sort();
        if !quants.is_empty() {
            anyhow::bail!("{model} holds GGUF files; choose one: {}", quants.join(", "));
        }
    }
    let name = model.rsplit('/').next().unwrap_or(model).to_string();
    Ok((snapshot, name))
}

/// The snapshot directory of repo `model` in the local Hugging Face cache.
pub fn hf_snapshot(model: &str) -> Result<PathBuf> {
    let cache = std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface/hub"));
    let repo = cache.join(format!("models--{}", model.replace('/', "--")));
    let snapshot = std::fs::read_to_string(repo.join("refs/main"))
        .ok()
        .map(|r| repo.join("snapshots").join(r.trim()))
        .filter(|p| p.is_dir())
        .or_else(|| std::fs::read_dir(repo.join("snapshots")).ok()?.flatten().map(|e| e.path()).next())
        .with_context(|| {
            format!(
                "{} is neither a directory nor a model in {}; download it with `hf download {}`",
                model,
                cache.display(),
                model
            )
        })?;
    Ok(snapshot)
}

/// The draft model for `spec`: `none`, `auto` (from the local cache: the Q2_K file of
/// pfnet/plamo-3-translate-0.8b-GGUF, else pfnet/plamo-3-translate-0.8b, else its Q4_K_M
/// file; nothing if none is there), or a model as for [`resolve_model`]. `None` also when
/// it is the target model itself.
pub fn resolve_draft(spec: &str, target: &Path) -> Result<Option<PathBuf>> {
    let dir = match spec {
        "none" => return Ok(None),
        // The 2-bit GGUF first: its steps are 1.27x faster than the checkpoint's at 4 bits
        // and about as many of its tokens are accepted (DESIGN.md 11.32).
        "auto" => match resolve_model("0.8b:Q2_K")
            .or_else(|_| resolve_model("pfnet/plamo-3-translate-0.8b"))
            .or_else(|_| resolve_model("0.8b:Q4_K_M"))
        {
            Ok((d, _)) => d,
            Err(_) => return Ok(None),
        },
        other => resolve_model(other)?.0,
    };
    let same = |a: &Path, b: &Path| a.canonicalize().ok() == b.canonicalize().ok();
    Ok(if same(&dir, target) { None } else { Some(dir) })
}
