//! Finding models on disk: directories, GGUF files, and names in the local Hugging Face
//! cache (`2b`, `pfnet/plamo-3-translate-2b`, `2b:Q4_K_M`).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use yaxus_model::Quant;

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
                let file = find_gguf(&dir, &quant, &hint)?;
                ModelSpec::File(file).locate(arg)
            }
            ModelSpec::Repo(repo) => {
                let snapshot = yaxus_model::hub::snapshot(&repo)?;
                if !snapshot.join("config.json").exists() {
                    let mut quants: Vec<String> = std::fs::read_dir(&snapshot)?
                        .flatten()
                        .filter_map(|e| Some(format!("{repo}:{}", gguf_quant(&e.path())?)))
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

/// The GGUF file of quantization `quant` in `dir`: the one whose name ends in
/// `-<quant>.gguf`, so `Q2_K` does not pick `IQ2_K` or `Q2_K_L`. `hint` says how to get one.
fn find_gguf(dir: &Path, quant: &str, hint: &str) -> Result<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|f| gguf_quant(f).is_some_and(|q| q.eq_ignore_ascii_case(quant)))
        .collect();
    found.sort();
    match found.len() {
        0 => anyhow::bail!("no *-{quant}.gguf in {}; {hint}", dir.display()),
        1 => Ok(found.remove(0)),
        _ => anyhow::bail!(
            "several files in {} are {quant}: {}; give the path of one",
            dir.display(),
            found.iter().filter_map(|f| f.file_name()?.to_str()).collect::<Vec<_>>().join(", ")
        ),
    }
}

/// The quantization of a GGUF file name: its last `-` part (`Q4_K_M` of
/// `plamo-3-translate-2b-Q4_K_M.gguf`).
fn gguf_quant(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf").or_else(|| name.strip_suffix(".GGUF"))?;
    Some(stem.rsplit('-').next().unwrap_or(stem).to_string())
}

/// Finds a model: a directory, a GGUF file, a repository in the local Hugging Face cache,
/// a size (`2b` for pfnet/plamo-3-translate-2b), or `REPO:QUANT` / `SIZE:QUANT` for one
/// file of a GGUF repository (`2b:Q4_K_M`).
pub fn resolve_model(model: &str) -> Result<ResolvedModel> {
    ModelSpec::parse(model).locate(model)
}

/// Which draft model to use for speculative decoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DraftSpec {
    /// No draft model.
    None,
    /// The 0.8B model from the local cache, if it is there and pays (see [`resolve_draft`]).
    Auto,
    /// A model as for [`resolve_model`].
    Model(String),
}

impl std::str::FromStr for DraftSpec {
    type Err = anyhow::Error;
    /// `none`, `auto`, or a model.
    fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "none" => DraftSpec::None,
            "auto" => DraftSpec::Auto,
            "" => anyhow::bail!("empty draft model (expected auto, none or a model)"),
            model => DraftSpec::Model(model.to_string()),
        })
    }
}

/// The draft model for target `target`: for [`DraftSpec::Auto`] from the local cache,
/// the Q2_K file of pfnet/plamo-3-translate-0.8b-GGUF, else pfnet/plamo-3-translate-0.8b,
/// else its Q4_K_M file (nothing if none is there). `None` also when it is the target's own
/// file or directory, and for `Auto` when the target has the draft's size and is a GGUF
/// file: a draft pays only when it reads much less than the target. 19-sentence set, 0.8B
/// target with the 0.8B Q2_K draft: the checkpoint (loaded at 8 bits) 87.0 -> 96.9 tok/s,
/// but Q4_K_M 126.5 -> 105.6 and Q3_K_M 134.7 -> 105.9.
pub fn resolve_draft(spec: &DraftSpec, target: &Path) -> Result<Option<PathBuf>> {
    let dir = match spec {
        DraftSpec::None => return Ok(None),
        // The 2-bit GGUF first: its steps are 1.27x faster than the checkpoint's at 4 bits
        // and about as many of its tokens are accepted (docs/HISTORY.md 11.32).
        DraftSpec::Auto => match resolve_model("0.8b:Q2_K")
            .or_else(|_| resolve_model("pfnet/plamo-3-translate-0.8b"))
            .or_else(|_| resolve_model("0.8b:Q4_K_M"))
        {
            Ok(m) => m.path,
            Err(_) => return Ok(None),
        },
        DraftSpec::Model(model) => resolve_model(model)?.path,
    };
    // Paths that cannot be canonicalized (missing, broken links) compare as given.
    let same = match (dir.canonicalize(), target.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => dir == target,
    };
    if same {
        return Ok(None);
    }
    let auto = matches!(spec, DraftSpec::Auto);
    if auto && yaxus_model::gguf::is_gguf(target) && shape(&dir).is_some_and(|s| Some(s) == shape(target)) {
        return Ok(None);
    }
    Ok(Some(dir))
}

/// (hidden size, layers) of a model directory or GGUF file.
fn shape(path: &Path) -> Option<(usize, usize)> {
    let cfg = if yaxus_model::gguf::is_gguf(path) {
        yaxus_model::gguf::Gguf::open(path).ok()?.config().ok()?
    } else {
        yaxus_model::Config::from_file(path.join("config.json")).ok()?
    };
    Some((cfg.hidden_size, cfg.num_hidden_layers))
}

/// How the weights of `model` are stored, for display: the quantization, or for a GGUF
/// file loaded with `auto` its main tensor type (`gguf Q4_K`).
pub fn quant_label(model: &Path, quant: Quant) -> String {
    if matches!(quant, Quant::Auto | Quant::Auto4) {
        if yaxus_model::gguf::is_gguf(model) {
            let main = yaxus_model::gguf::Gguf::open(model).ok().and_then(|g| g.main_type());
            return format!("gguf {}", main.map_or("?", |t| t.name()));
        }
        return if quant == Quant::Auto { "q8" } else { "q4" }.into();
    }
    format!("{quant:?}").to_lowercase()
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
    fn quantization_is_the_last_part_of_the_name() {
        let q = |n: &str| gguf_quant(Path::new(n));
        assert_eq!(q("plamo-3-translate-0.8b-Q2_K.gguf").as_deref(), Some("Q2_K"));
        assert_eq!(q("x-IQ2_K.gguf").as_deref(), Some("IQ2_K"));
        assert_eq!(q("x-UD-Q2_K_XL.gguf").as_deref(), Some("Q2_K_XL"));
        assert_eq!(q("README.md"), None);
    }

    #[test]
    fn draft_specs_parse() {
        assert_eq!("none".parse::<DraftSpec>().unwrap(), DraftSpec::None);
        assert_eq!("auto".parse::<DraftSpec>().unwrap(), DraftSpec::Auto);
        assert_eq!("0.8b:Q2_K".parse::<DraftSpec>().unwrap(), DraftSpec::Model("0.8b:Q2_K".into()));
        assert!("".parse::<DraftSpec>().is_err());
        assert_eq!(resolve_draft(&DraftSpec::None, Path::new(".")).unwrap(), None);
    }

    #[test]
    fn a_path_without_a_name_shows_as_written() {
        let m = ModelSpec::Dir(".".into()).locate(".").unwrap();
        assert_eq!(m.name, ".");
    }
}
