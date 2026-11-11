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
/// `refs/main` names, else the most recently modified one (downloads of a pinned
/// revision have no `refs/main`), so the choice does not depend on directory order.
pub fn snapshot(repo: &str) -> Result<PathBuf> {
    let cache = cache_dir();
    let dir = cache.join(format!("models--{}", repo.replace('/', "--")));
    std::fs::read_to_string(dir.join("refs/main"))
        .ok()
        .map(|r| dir.join("snapshots").join(r.trim()))
        .filter(|p| p.is_dir())
        .or_else(|| newest(&dir.join("snapshots")))
        .with_context(|| {
            format!(
                "{repo} is neither a directory nor a model in {}; download it with `hf download {repo}`",
                cache.display()
            )
        })
}

/// The most recently modified entry of `dir` (ties: the last by name).
fn newest(dir: &std::path::Path) -> Option<PathBuf> {
    let mtime = |p: &PathBuf| p.metadata().and_then(|m| m.modified()).ok();
    std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).max_by(|a, b| (mtime(a), a).cmp(&(mtime(b), b)))
}

/// Whether tests that need models in the cache must fail instead of skipping without
/// them: `YAXUS_REQUIRE_GOLDEN` set to anything but empty or `0` (as in CI).
pub fn models_required() -> bool {
    std::env::var_os("YAXUS_REQUIRE_GOLDEN").is_some_and(|v| !v.is_empty() && v != "0")
}
