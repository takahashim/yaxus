//! The local Hugging Face cache (`HF_HUB_CACHE`, else `HF_HOME/hub`, else
//! `~/.cache/huggingface/hub`).

use std::path::PathBuf;

use anyhow::{Context, Result};

/// The cache directory.
pub fn cache_dir() -> PathBuf {
    std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
        .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface/hub"))
}

/// The snapshot directory of repository `repo` (`pfnet/plamo-3-translate-2b`): the one
/// `refs/main` names, else any.
pub fn snapshot(repo: &str) -> Result<PathBuf> {
    let cache = cache_dir();
    let dir = cache.join(format!("models--{}", repo.replace('/', "--")));
    std::fs::read_to_string(dir.join("refs/main"))
        .ok()
        .map(|r| dir.join("snapshots").join(r.trim()))
        .filter(|p| p.is_dir())
        .or_else(|| std::fs::read_dir(dir.join("snapshots")).ok()?.flatten().map(|e| e.path()).next())
        .with_context(|| {
            format!(
                "{repo} is neither a directory nor a model in {}; download it with `hf download {repo}`",
                cache.display()
            )
        })
}
