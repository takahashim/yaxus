//! Finding models on disk: directories, GGUF files, and names in the local Hugging Face
//! cache (`2b`, `pfnet/plamo-3-translate-2b`, `2b:Q4_K_M`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// A model found on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedModel {
    /// A Hugging Face directory or a GGUF file.
    pub path: PathBuf,
    /// Short name for display (`plamo-3-translate-2b`, the GGUF file's stem).
    pub name: String,
}

/// Model sizes that stand for pfnet/plamo-3-translate-<size>.
const SIZES: [&str; 4] = ["0.8b", "2b", "8b", "31b"];

/// What a model argument names, before looking anything up.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ModelSpec {
    /// An existing file (it must be GGUF).
    File(PathBuf),
    /// An existing directory.
    Dir(PathBuf),
    /// A repository in the Hugging Face cache.
    Repo(String),
    /// The file of a GGUF repository whose name ends in `<quant>.gguf`.
    GgufFile { repo: String, quant: String },
}

impl ModelSpec {
    /// Existing paths first; then `SIZE` or `REPO`, with an optional `:QUANT`.
    fn parse(model: &str) -> ModelSpec {
        let p = Path::new(model);
        if p.is_file() {
            return ModelSpec::File(p.to_path_buf());
        }
        if p.is_dir() {
            return ModelSpec::Dir(p.to_path_buf());
        }
        let repo = |name: &str, suffix: &str| match SIZES.iter().find(|s| s.eq_ignore_ascii_case(name)) {
            Some(size) => format!("pfnet/plamo-3-translate-{size}{suffix}"),
            None => name.to_string(),
        };
        match model.rsplit_once(':') {
            Some((name, quant)) => ModelSpec::GgufFile { repo: repo(name, "-GGUF"), quant: quant.to_string() },
            None => ModelSpec::Repo(repo(model, "")),
        }
    }

    /// Finds it on disk. `arg` is what the user wrote, the display name when the path
    /// gives none (`.`).
    fn locate(self, arg: &str) -> Result<ResolvedModel> {
        match self {
            ModelSpec::File(p) => {
                anyhow::ensure!(yaxus_model::gguf::is_gguf(&p), "{} is not a GGUF file", p.display());
                let name = p.file_stem().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| arg.to_string());
                Ok(ResolvedModel { path: p, name })
            }
            ModelSpec::Dir(p) => {
                // A snapshot in the Hugging Face cache is named after its repository.
                let name = p
                    .ancestors()
                    .filter_map(|a| a.file_name()?.to_str())
                    .find_map(|n| n.strip_prefix("models--").map(|r| r.rsplit("--").next().unwrap_or(r).to_string()))
                    .or_else(|| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                    .unwrap_or_else(|| arg.to_string());
                Ok(ResolvedModel { path: p, name })
            }
            ModelSpec::GgufFile { repo, quant } => {
                let hint = format!("download it with `hf download {repo} --include '*{quant}.gguf'`");
                let dir = yaxus_model::hub::snapshot(&repo).context(hint.clone())?;
                let want = format!("{}.gguf", quant.to_ascii_lowercase());
                let mut found: Vec<PathBuf> = std::fs::read_dir(&dir)?
                    .flatten()
                    .map(|e| e.path())
                    .filter(|f| {
                        f.file_name().is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().ends_with(&want))
                    })
                    .collect();
                found.sort();
                let file = found
                    .into_iter()
                    .next()
                    .with_context(|| format!("no *{quant}.gguf in {}; {hint}", dir.display()))?;
                ModelSpec::File(file).locate(arg)
            }
            ModelSpec::Repo(repo) => {
                let snapshot = yaxus_model::hub::snapshot(&repo)?;
                if !snapshot.join("config.json").exists() {
                    let mut quants: Vec<String> = std::fs::read_dir(&snapshot)?
                        .flatten()
                        .filter_map(|e| {
                            let n = e.file_name().to_string_lossy().into_owned();
                            let stem = n.strip_suffix(".gguf")?;
                            Some(format!("{repo}:{}", stem.rsplit('-').next().unwrap_or(stem)))
                        })
                        .collect();
                    quants.sort();
                    if !quants.is_empty() {
                        anyhow::bail!("{repo} holds GGUF files; choose one: {}", quants.join(", "));
                    }
                }
                let name = repo.rsplit('/').next().unwrap_or(&repo).to_string();
                Ok(ResolvedModel { path: snapshot, name })
            }
        }
    }
}

/// Finds a model: a directory, a GGUF file, a repository in the local Hugging Face cache,
/// a size (`2b` for pfnet/plamo-3-translate-2b), or `REPO:QUANT` / `SIZE:QUANT` for one
/// file of a GGUF repository (`2b:Q4_K_M`).
pub fn resolve_model(model: &str) -> Result<ResolvedModel> {
    ModelSpec::parse(model).locate(model)
}

/// The draft model for target `target` and `spec`: `none`, `auto` (from the local cache:
/// the Q2_K file of pfnet/plamo-3-translate-0.8b-GGUF, else pfnet/plamo-3-translate-0.8b,
/// else its Q4_K_M file; nothing if none is there), or a model as for [`resolve_model`].
/// `None` also when it is the target's own file or directory. A draft of the target's
/// size is kept: at fewer bits it still pays (0.8B target with the 0.8B Q2_K draft:
/// 87.2 -> 95.2 tok/s on the 19-sentence set).
pub fn resolve_draft(spec: &str, target: &Path) -> Result<Option<PathBuf>> {
    let dir = match spec {
        "none" => return Ok(None),
        // The 2-bit GGUF first: its steps are 1.27x faster than the checkpoint's at 4 bits
        // and about as many of its tokens are accepted (DESIGN.md 11.32).
        "auto" => match resolve_model("0.8b:Q2_K")
            .or_else(|_| resolve_model("pfnet/plamo-3-translate-0.8b"))
            .or_else(|_| resolve_model("0.8b:Q4_K_M"))
        {
            Ok(m) => m.path,
            Err(_) => return Ok(None),
        },
        other => resolve_model(other)?.path,
    };
    // Paths that cannot be canonicalized (missing, broken links) compare as given.
    let same = match (dir.canonicalize(), target.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => dir == target,
    };
    Ok(if same { None } else { Some(dir) })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_names_parse() {
        let repo = |s: &str| ModelSpec::Repo(s.into());
        let gguf = |r: &str, q: &str| ModelSpec::GgufFile { repo: r.into(), quant: q.into() };
        assert_eq!(ModelSpec::parse("2b"), repo("pfnet/plamo-3-translate-2b"));
        assert_eq!(ModelSpec::parse("0.8B"), repo("pfnet/plamo-3-translate-0.8b"));
        assert_eq!(ModelSpec::parse("someone/model"), repo("someone/model"));
        assert_eq!(ModelSpec::parse("8b:Q4_K_M"), gguf("pfnet/plamo-3-translate-8b-GGUF", "Q4_K_M"));
        assert_eq!(ModelSpec::parse("someone/model-GGUF:Q8_0"), gguf("someone/model-GGUF", "Q8_0"));
        let dir = std::env::temp_dir();
        assert_eq!(ModelSpec::parse(dir.to_str().unwrap()), ModelSpec::Dir(dir));
    }

    #[test]
    fn a_path_without_a_name_shows_as_written() {
        let m = ModelSpec::Dir(".".into()).locate(".").unwrap();
        assert_eq!(m.name, ".");
    }
}
