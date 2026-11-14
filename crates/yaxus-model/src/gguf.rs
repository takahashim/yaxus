//! GGUF files (llama.cpp): metadata, the model configuration, the vocabulary and the
//! tensors under the safetensors names [`crate::Model::load`] uses.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;

use crate::config::{Config, RopeParam, RopeParameters, Yarn};
use crate::ggml::GgmlType;
use crate::weights::{Backing, TensorRef, Weights};
use crate::{POST_MIXER_NORM_OFFSET, POST_MLP_NORM_OFFSET, PRE_NORM_OFFSET};

#[derive(Debug, Clone)]
pub enum Value {
    /// Any GGUF integer type, exactly (`i128` holds both `u64` and `i64`).
    Int(i128),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Int(i) => u64::try_from(*i).ok(),
            _ => None,
        }
    }
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(i) => Some(*i as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// The vocabulary stored in a GGUF file.
pub struct Vocab {
    pub tokens: Vec<String>,
    pub scores: Vec<f64>,
    /// llama.cpp token types: 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte.
    pub types: Vec<i32>,
}

pub struct Gguf {
    pub meta: HashMap<String, Value>,
    /// Tensors under their GGUF names, in file order.
    pub tensors: Vec<(String, TensorRef)>,
}

/// True when `path` is a GGUF file (by its magic).
pub fn is_gguf(path: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 4];
    path.is_file() && std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok() && &magic == b"GGUF"
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

/// A count, size or offset from the file as a `usize`. On wasm32 a `u64 as usize` would
/// drop the upper 32 bits (2^32 tensors would read as none), so values that do not fit
/// are errors.
fn to_usize(v: u64, what: impl FnOnce() -> String) -> Result<usize> {
    usize::try_from(v).map_err(|_| anyhow::anyhow!("{}: {v} does not fit in memory", what()))
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        // Lengths come from the file: checked so that a corrupt one cannot overflow.
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len()).context("GGUF header is truncated")?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = to_usize(self.u64()?, || "GGUF string length".into())?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, t: u32) -> Result<Value> {
        Ok(match t {
            0 => Value::Int(self.take(1)?[0] as i128),
            1 => Value::Int(self.take(1)?[0] as i8 as i128),
            2 => Value::Int(u16::from_le_bytes(self.take(2)?.try_into()?) as i128),
            3 => Value::Int(i16::from_le_bytes(self.take(2)?.try_into()?) as i128),
            4 => Value::Int(self.u32()? as i128),
            5 => Value::Int(self.u32()? as i32 as i128),
            6 => Value::Float(f32::from_le_bytes(self.take(4)?.try_into()?) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let it = self.u32()?;
                let n = to_usize(self.u64()?, || "GGUF array length".into())?;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(it)?);
                }
                Value::Array(v)
            }
            10 => Value::Int(self.u64()? as i128),
            11 => Value::Int(self.u64()? as i64 as i128),
            12 => Value::Float(f64::from_le_bytes(self.take(8)?.try_into()?)),
            other => bail!("unknown GGUF value type {other}"),
        })
    }
}

impl Gguf {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // SAFETY: read-only mapping; modifying the file while mapped is UB like with any
        // mmap-based loader.
        let mmap = Arc::new(unsafe { Mmap::map(&file)? });
        Self::from_backing(Backing::Mmap(mmap)).with_context(|| format!("reading {}", path.display()))
    }

    /// Parses a GGUF file already in memory (the browser reads the file the user chose).
    pub fn parse(bytes: Vec<u8>) -> Result<Self> {
        Self::from_backing(Backing::Bytes(Arc::new(bytes)))
    }

    fn from_backing(backing: Backing) -> Result<Self> {
        let bytes = backing.as_slice();
        let mut r = Reader { b: bytes, at: 0 };
        ensure!(r.take(4)? == b"GGUF", "not a GGUF file");
        let version = r.u32()?;
        ensure!(version == 2 || version == 3, "GGUF version {version} is not supported");
        let n_tensors = to_usize(r.u64()?, || "GGUF tensor count".into())?;
        let n_kv = to_usize(r.u64()?, || "GGUF metadata count".into())?;
        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let k = r.string()?;
            let t = r.u32()?;
            meta.insert(k, r.value(t)?);
        }
        // Counts come from the file: reserve little up front and let a truncated file
        // fail on its first missing byte.
        let mut infos = Vec::with_capacity(n_tensors.min(1 << 16));
        for _ in 0..n_tensors {
            let name = r.string()?;
            let nd = r.u32()? as usize;
            let mut dims = Vec::with_capacity(nd.min(8));
            for _ in 0..nd {
                dims.push(to_usize(r.u64()?, || format!("{name}: dimension"))?);
            }
            let t = GgmlType::from_id(r.u32()?).with_context(|| format!("tensor {name}"))?;
            let off = to_usize(r.u64()?, || format!("{name}: offset"))?;
            infos.push((name, dims, t, off));
        }
        // An alignment that is there but not an unsigned integer is an error, not 32.
        let align = match meta.get("general.alignment") {
            None => 32,
            Some(v) => {
                let a = v.as_u64().context("GGUF alignment is not an unsigned integer")?;
                to_usize(a, || "GGUF alignment".into())?
            }
        };
        ensure!(align.is_power_of_two(), "GGUF alignment {align} is not a power of two");
        let data = r.at.checked_next_multiple_of(align).context("GGUF alignment overflows")?;
        let mut tensors = Vec::with_capacity(n_tensors.min(1 << 16));
        for (name, dims, t, off) in infos {
            // GGUF lists dimensions innermost first.
            let shape: Vec<usize> = dims.into_iter().rev().collect();
            let n = shape
                .iter()
                .try_fold(1usize, |a, &d| a.checked_mul(d))
                .with_context(|| format!("{name}: too many elements"))?;
            ensure!(n.is_multiple_of(t.block().0), "{name}: {n} elements do not fill {t:?} blocks");
            let len =
                (n / t.block().0).checked_mul(t.block().1).with_context(|| format!("{name}: too many elements"))?;
            ensure!(off.is_multiple_of(align), "{name}: offset {off} is not a multiple of the alignment {align}");
            let end = data.checked_add(off).and_then(|s| s.checked_add(len));
            ensure!(end.is_some_and(|e| e <= bytes.len()), "{name}: data beyond the end of the file");
            let start = data + off;
            tensors.push((name, TensorRef { backing: backing.clone(), start, len, shape, dtype: t, minus: 0.0 }));
        }
        Ok(Gguf { meta, tensors })
    }

    fn arch(&self) -> Result<&str> {
        self.meta.get("general.architecture").and_then(Value::as_str).context("general.architecture missing")
    }

    fn num(&self, key: &str) -> Result<f64> {
        let k = format!("{}.{key}", self.arch()?);
        self.meta.get(&k).and_then(Value::as_f64).with_context(|| format!("GGUF metadata {k} missing"))
    }

    /// The model configuration from the `plamo3.*` metadata.
    pub fn config(&self) -> Result<Config> {
        let arch = self.arch()?;
        ensure!(arch == "plamo3", "GGUF architecture {arch:?} is not supported (expected plamo3)");
        let u = |k: &str| self.num(k).map(|v| v as usize);
        let truncate = matches!(self.meta.get("plamo3.rope.scaling.yarn_truncate"), Some(Value::Bool(true)));
        let vocab = match self.meta.get("tokenizer.ggml.tokens") {
            Some(Value::Array(t)) => t.len(),
            _ => u("vocab_size")?,
        };
        let yarn = Yarn {
            factor: self.num("rope.scaling.factor")?,
            original_max_position_embeddings: u("rope.scaling.original_context_length")?,
            beta_fast: self.num("rope.scaling.yarn_beta_fast")?,
            beta_slow: self.num("rope.scaling.yarn_beta_slow")?,
            truncate,
        };
        let (factor, original) = (yarn.factor, yarn.original_max_position_embeddings);
        let full = RopeParam::yarn(self.num("rope.freq_base")?, yarn);
        let sliding = RopeParam::plain(self.num("rope.freq_base_swa")?);
        let cfg = Config {
            hidden_size: u("embedding_length")?,
            num_hidden_layers: u("block_count")?,
            num_attention_heads: u("attention.head_count")?,
            num_key_value_heads: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            intermediate_size: u("feed_forward_length")?,
            vocab_size: vocab,
            rms_norm_eps: self.num("attention.layer_norm_rms_epsilon")? as f32,
            window_size: u("attention.sliding_window")?,
            sliding_window_pattern: u("attention.sliding_window_pattern")?,
            rope_theta: full.rope_theta,
            rope_local_theta: sliding.rope_theta,
            scale_embedding: false,
            tie_word_embeddings: false,
            rope_scaling_factor: factor,
            initial_context_length: original,
            rope_parameters: Some(RopeParameters { full_attention: full, sliding_attention: sliding }),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// The vocabulary (`tokenizer.ggml.*`); the model must use the PLaMo 3 tokenizer.
    pub fn vocab(&self) -> Result<Vocab> {
        let model = self.meta.get("tokenizer.ggml.model").and_then(Value::as_str).unwrap_or("");
        ensure!(model == "plamo3", "GGUF tokenizer {model:?} is not supported (expected plamo3)");
        let arr = |k: &str| match self.meta.get(k) {
            Some(Value::Array(v)) => Ok(v),
            _ => bail!("GGUF metadata {k} missing"),
        };
        let tokens: Vec<String> =
            arr("tokenizer.ggml.tokens")?.iter().map(|v| v.as_str().unwrap_or_default().to_string()).collect();
        let scores: Vec<f64> = arr("tokenizer.ggml.scores")?.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();
        let types: Vec<i32> =
            arr("tokenizer.ggml.token_type")?.iter().map(|v| v.as_f64().unwrap_or(1.0) as i32).collect();
        ensure!(tokens.len() == scores.len() && tokens.len() == types.len(), "GGUF vocabulary arrays differ in length");
        Ok(Vocab { tokens, scores, types })
    }

    /// The tensors under the safetensors names of the HF checkpoint. RMSNorm weights are
    /// stored with their offset added by llama.cpp's converter; the offset is removed so
    /// the model adds it back as usual.
    pub fn weights(&self) -> Result<Weights> {
        let by_name: HashMap<&str, &TensorRef> = self.tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let get = |n: &str| {
            by_name
                .get(n)
                .or_else(|| by_name.get(format!("{n}.weight").as_str()))
                .copied()
                .with_context(|| format!("GGUF tensor {n} missing"))
        };
        // HF norm weights are small corrections around 0; converted ones sit around the offset.
        let first = get("blk.0.attn_norm")?.to_f32_vec();
        let baked = first.iter().sum::<f32>() / first.len() as f32 > 0.5;
        let norm = |n: &str, offset: f32| -> Result<TensorRef> {
            let mut t = get(n)?.clone();
            if baked {
                t.minus = offset;
            }
            Ok(t)
        };
        let mut out = HashMap::new();
        out.insert(crate::EMBED_TOKENS.to_string(), get("token_embd")?.clone());
        out.insert("lm_head.weight".to_string(), get("output")?.clone());
        out.insert("model.norm.weight".to_string(), norm("output_norm", PRE_NORM_OFFSET)?);
        let layers = self.num("block_count")? as usize;
        for i in 0..layers {
            let (b, p) = (format!("blk.{i}"), format!("model.layers.layers.{i}"));
            let plain = [
                ("attn_qkv", "mixer.qkv_proj"),
                ("attn_output", "mixer.o_proj"),
                ("ffn_up", "mlp.gate_up_proj"),
                ("ffn_down", "mlp.down_proj"),
            ];
            for (g, h) in plain {
                out.insert(format!("{p}.{h}.weight"), get(&format!("{b}.{g}"))?.clone());
            }
            let norms = [
                ("attn_norm", "pre_mixer_norm", PRE_NORM_OFFSET),
                ("attn_q_norm", "mixer.q_norm", PRE_NORM_OFFSET),
                ("attn_k_norm", "mixer.k_norm", PRE_NORM_OFFSET),
                ("post_attention_norm", "post_mixer_norm", POST_MIXER_NORM_OFFSET),
                ("ffn_norm", "pre_mlp_norm", PRE_NORM_OFFSET),
                ("post_ffw_norm", "post_mlp_norm", POST_MLP_NORM_OFFSET),
            ];
            for (g, h, off) in norms {
                out.insert(format!("{p}.{h}.weight"), norm(&format!("{b}.{g}"), off)?);
            }
        }
        Ok(Weights { tensors: out })
    }

    /// The most common matrix type, for display (e.g. `Q4_K`).
    pub fn main_type(&self) -> Option<GgmlType> {
        let mut count: HashMap<GgmlType, usize> = HashMap::new();
        for (_, t) in &self.tensors {
            if t.shape.len() == 2 {
                *count.entry(t.dtype).or_default() += t.numel();
            }
        }
        count.into_iter().max_by_key(|&(_, n)| n).map(|(t, _)| t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct File(std::path::PathBuf);
    impl Drop for File {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn write(name: &str, bytes: &[u8]) -> File {
        let p = std::env::temp_dir().join(format!("yaxus-gguf-{}-{name}.gguf", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        File(p)
    }

    /// Header: magic, version 3, tensor and key/value counts.
    fn head(n_tensors: u64, n_kv: u64) -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(n_tensors.to_le_bytes());
        b.extend(n_kv.to_le_bytes());
        b
    }

    fn string(b: &mut Vec<u8>, s: &str) {
        b.extend((s.len() as u64).to_le_bytes());
        b.extend(s.as_bytes());
    }

    /// One tensor entry: name, dims (innermost first), F32, offset.
    fn tensor(b: &mut Vec<u8>, dims: &[u64], off: u64) {
        string(b, "t");
        b.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend(d.to_le_bytes());
        }
        b.extend(0u32.to_le_bytes());
        b.extend(off.to_le_bytes());
    }

    fn open_err(name: &str, bytes: &[u8]) -> String {
        let f = write(name, bytes);
        match Gguf::open(&f.0) {
            Ok(_) => panic!("{name}: opened"),
            Err(e) => format!("{e:#}"),
        }
    }

    #[test]
    fn corrupt_files_are_errors_not_panics() {
        // A string length near u64::MAX.
        let mut b = head(0, 1);
        b.extend(u64::MAX.to_le_bytes());
        assert!(open_err("strlen", &b).contains("truncated"));

        // Alignment 0.
        let mut b = head(0, 1);
        string(&mut b, "general.alignment");
        b.extend(4u32.to_le_bytes());
        b.extend(0u32.to_le_bytes());
        assert!(open_err("align", &b).contains("power of two"));

        // A tensor offset that overflows (aligned, so that the overflow is what fails).
        let mut b = head(1, 0);
        tensor(&mut b, &[4], u64::MAX - 31);
        assert!(open_err("offset", &b).contains("beyond the end"));

        // Dimensions whose product overflows.
        let mut b = head(1, 0);
        tensor(&mut b, &[u64::MAX / 2, 4], 0);
        assert!(open_err("dims", &b).contains("too many elements"));

        // A huge tensor count in a short file. On wasm32 the count itself does not fit (2^32
        // would read as 0 with `as usize`).
        assert!(open_err("count", &head(u64::MAX, 0)).contains("truncated"));
        let e = open_err("count32", &head(1 << 32, 0));
        assert!(e.contains("truncated") || e.contains("does not fit"), "{e}");

        // An alignment of 2^63 puts the data beyond the file (on wasm32 it does not even
        // fit a `usize`).
        let mut b = head(1, 1);
        string(&mut b, "general.alignment");
        b.extend(10u32.to_le_bytes());
        b.extend((1u64 << 63).to_le_bytes());
        tensor(&mut b, &[4], 0);
        let e = open_err("align63", &b);
        assert!(e.contains("beyond the end") || e.contains("does not fit"), "{e}");

        // A negative (INT64) alignment is an error, not the default of 32.
        let mut b = head(0, 1);
        string(&mut b, "general.alignment");
        b.extend(11u32.to_le_bytes());
        b.extend((-64i64).to_le_bytes());
        assert!(open_err("negalign", &b).contains("not an unsigned integer"));

        // A tensor offset that is not a multiple of the alignment.
        let mut b = head(1, 0);
        tensor(&mut b, &[1], 4);
        b.resize(b.len().div_ceil(32) * 32 + 8, 0);
        assert!(open_err("misaligned", &b).contains("not a multiple of the alignment"));
    }

    #[test]
    fn integers_keep_their_value() {
        // UINT64 at and above 2^63, and a negative INT64.
        let mut b = head(0, 3);
        for (k, t, v) in [("a", 10u32, u64::MAX), ("b", 10, 1 << 63), ("c", 11, (-5i64) as u64)] {
            string(&mut b, k);
            b.extend(t.to_le_bytes());
            b.extend(v.to_le_bytes());
        }
        let g = Gguf::parse(b).unwrap();
        assert_eq!(g.meta["a"].as_u64(), Some(u64::MAX));
        assert_eq!(g.meta["b"].as_u64(), Some(1 << 63));
        assert_eq!((g.meta["c"].as_u64(), g.meta["c"].as_f64()), (None, Some(-5.0)));
    }

    #[test]
    fn first_rows_drops_the_trailing_rows() {
        // A [3, 2] F32 matrix (dims innermost first).
        let mut b = head(1, 0);
        tensor(&mut b, &[2, 3], 0);
        b.resize(b.len().div_ceil(32) * 32, 0);
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            b.extend(v.to_le_bytes());
        }
        let f = write("rows", &b);
        let g = Gguf::open(&f.0).unwrap();
        let t = &g.tensors[0].1;
        let top = t.first_rows(2).unwrap();
        assert_eq!((top.shape.clone(), top.to_f32_vec()), (vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]));
        assert!(t.first_rows(4).is_none());
    }

    #[test]
    fn first_rows_of_a_quantized_matrix() {
        // Two rows of one Q8_0 block each (f16 scale, then 32 int8 values).
        let mut b = head(1, 0);
        string(&mut b, "t");
        b.extend(2u32.to_le_bytes());
        b.extend(32u64.to_le_bytes());
        b.extend(2u64.to_le_bytes());
        b.extend(8u32.to_le_bytes()); // Q8_0
        b.extend(0u64.to_le_bytes());
        b.resize(b.len().div_ceil(32) * 32, 0);
        for (scale, base) in [(0x3c00u16, 1i8), (0x4000u16, -1i8)] {
            b.extend(scale.to_le_bytes()); // 1.0, then 2.0
            b.extend((0..32).map(|i| (base * (i as i8 % 5)) as u8));
        }
        let f = write("q8rows", &b);
        let g = Gguf::open(&f.0).unwrap();
        let t = &g.tensors[0].1;
        let all = t.to_f32_vec();
        let top = t.first_rows(1).unwrap();
        assert_eq!(top.shape, [1, 32]);
        assert_eq!(top.to_f32_vec(), all[..32]);
    }

    #[test]
    fn an_in_memory_file_parses_like_a_mapped_one() {
        let mut b = head(1, 0);
        tensor(&mut b, &[2], 0);
        b.resize(b.len().div_ceil(32) * 32, 0);
        b.extend(1.5f32.to_le_bytes());
        b.extend((-2.0f32).to_le_bytes());
        let f = write("mem", &b);
        let mapped = Gguf::open(&f.0).unwrap();
        let parsed = Gguf::parse(b).unwrap();
        assert_eq!(parsed.tensors.len(), mapped.tensors.len());
        assert_eq!(parsed.tensors[0].1.shape, mapped.tensors[0].1.shape);
        assert_eq!(parsed.tensors[0].1.dtype, mapped.tensors[0].1.dtype);
        assert_eq!(parsed.tensors[0].1.bytes(), mapped.tensors[0].1.bytes());
        assert_eq!(parsed.tensors[0].1.to_f32_vec(), [1.5, -2.0]);
    }

    #[test]
    fn a_small_valid_file_opens() {
        let mut b = head(1, 0);
        tensor(&mut b, &[2], 0);
        b.resize(b.len().div_ceil(32) * 32, 0);
        b.extend(1.5f32.to_le_bytes());
        b.extend((-2.0f32).to_le_bytes());
        let f = write("valid", &b);
        let g = Gguf::open(&f.0).unwrap();
        assert_eq!(g.tensors.len(), 1);
        assert_eq!(g.tensors[0].1.to_f32_vec(), [1.5, -2.0]);
    }
}
