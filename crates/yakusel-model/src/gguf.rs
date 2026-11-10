//! GGUF files (llama.cpp): metadata, the model configuration, the vocabulary and the
//! tensors under the safetensors names [`crate::Model::load`] uses.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;

use crate::config::{Config, RopeParam, RopeParameters};
use crate::ggml::GgmlType;
use crate::weights::{TensorRef, Weights};
use crate::{POST_MIXER_NORM_OFFSET, POST_MLP_NORM_OFFSET, PRE_NORM_OFFSET};

#[derive(Debug, Clone)]
pub enum Value {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Int(i) if *i >= 0 => Some(*i as u64),
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

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        ensure!(self.at + n <= self.b.len(), "GGUF header is truncated");
        let s = &self.b[self.at..self.at + n];
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()? as usize;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, t: u32) -> Result<Value> {
        Ok(match t {
            0 => Value::Int(self.take(1)?[0] as i64),
            1 => Value::Int(self.take(1)?[0] as i8 as i64),
            2 => Value::Int(u16::from_le_bytes(self.take(2)?.try_into()?) as i64),
            3 => Value::Int(i16::from_le_bytes(self.take(2)?.try_into()?) as i64),
            4 => Value::Int(self.u32()? as i64),
            5 => Value::Int(self.u32()? as i32 as i64),
            6 => Value::Float(f32::from_le_bytes(self.take(4)?.try_into()?) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let it = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(it)?);
                }
                Value::Array(v)
            }
            10 => Value::Int(self.u64()? as i64),
            11 => Value::Int(self.u64()? as i64),
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
        let mut r = Reader { b: &mmap[..], at: 0 };
        ensure!(r.take(4)? == b"GGUF", "{} is not a GGUF file", path.display());
        let version = r.u32()?;
        ensure!(version == 2 || version == 3, "GGUF version {version} is not supported");
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;
        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let k = r.string()?;
            let t = r.u32()?;
            meta.insert(k, r.value(t)?);
        }
        let mut infos = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let nd = r.u32()? as usize;
            let mut dims = Vec::with_capacity(nd);
            for _ in 0..nd {
                dims.push(r.u64()? as usize);
            }
            let t = GgmlType::from_id(r.u32()?).with_context(|| format!("tensor {name}"))?;
            let off = r.u64()? as usize;
            infos.push((name, dims, t, off));
        }
        let align = meta.get("general.alignment").and_then(Value::as_u64).unwrap_or(32) as usize;
        let data = r.at.div_ceil(align) * align;
        let mut tensors = Vec::with_capacity(n_tensors);
        for (name, dims, t, off) in infos {
            // GGUF lists dimensions innermost first.
            let shape: Vec<usize> = dims.into_iter().rev().collect();
            let n: usize = shape.iter().product();
            ensure!(n.is_multiple_of(t.block().0), "{name}: {n} elements do not fill {t:?} blocks");
            let (start, len) = (data + off, t.bytes_for(n));
            ensure!(start + len <= mmap.len(), "{name}: data beyond the end of the file");
            tensors.push((name, TensorRef { mmap: mmap.clone(), start, len, shape, dtype: t, minus: 0.0 }));
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
        let full = RopeParam {
            rope_type: "yarn".into(),
            rope_theta: self.num("rope.freq_base")?,
            factor: Some(self.num("rope.scaling.factor")?),
            original_max_position_embeddings: Some(u("rope.scaling.original_context_length")?),
            beta_fast: Some(self.num("rope.scaling.yarn_beta_fast")?),
            beta_slow: Some(self.num("rope.scaling.yarn_beta_slow")?),
            truncate,
        };
        let sliding = RopeParam {
            rope_type: "default".into(),
            rope_theta: self.num("rope.freq_base_swa")?,
            factor: None,
            original_max_position_embeddings: None,
            beta_fast: None,
            beta_slow: None,
            truncate: false,
        };
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
            rope_scaling_factor: full.factor.unwrap(),
            initial_context_length: full.original_max_position_embeddings.unwrap(),
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
        out.insert("model.embed_tokens.weight".to_string(), get("token_embd")?.clone());
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
