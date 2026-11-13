//! Memory-mapped safetensors (possibly sharded) access.

use std::collections::HashMap;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
#[cfg(not(target_arch = "wasm32"))]
use memmap2::Mmap;
#[cfg(not(target_arch = "wasm32"))]
use safetensors::{Dtype, SafeTensors};

use crate::ggml::GgmlType;

/// The bytes a tensor lives in: a memory-mapped file (native), or an owned
/// buffer (a web page: a tensor's bytes read out of the File, and freed once
/// the backend holds the weight).
pub enum Backing {
    /// A read-only memory mapping.
    #[cfg(not(target_arch = "wasm32"))]
    Mmap(Mmap),
    /// Owned bytes (the web loader reads a tensor, uploads it, drops it).
    Owned(Vec<u8>),
}

impl Backing {
    fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(not(target_arch = "wasm32"))]
            Backing::Mmap(m) => &m[..],
            Backing::Owned(v) => v,
        }
    }
}

/// A tensor living in a [`Backing`]: bf16 from safetensors, or any
/// [`GgmlType`] from a GGUF file. `shape` is row-major (`[N, K]` for matrices).
#[derive(Clone)]
pub struct TensorRef {
    pub(crate) backing: Arc<Backing>,
    pub(crate) start: usize,
    pub(crate) len: usize,
    pub shape: Vec<usize>,
    pub dtype: GgmlType,
    /// Subtracted from every value read through `to_f32_vec`/`to_bf16_vec` (GGUF files
    /// store RMSNorm weights with their offset added).
    pub(crate) minus: f32,
}

impl TensorRef {
    /// A tensor over owned bytes `b` (`shape` row-major, `len` = `b.len()`).
    pub fn from_owned(b: Vec<u8>, shape: Vec<usize>, dtype: GgmlType, minus: f32) -> Self {
        let len = b.len();
        TensorRef { backing: Arc::new(Backing::Owned(b)), start: 0, len, shape, dtype, minus }
    }

    /// Raw bytes in the file's encoding.
    pub fn bytes(&self) -> &[u8] {
        &self.backing.as_slice()[self.start..self.start + self.len]
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// The first `rows` rows of a matrix, without copying (rows are contiguous). `None`
    /// when this is not a matrix of at least `rows` rows whose rows are whole blocks.
    pub fn first_rows(&self, rows: usize) -> Option<TensorRef> {
        let &[n, k] = &self.shape[..] else { return None };
        if rows > n || !k.is_multiple_of(self.dtype.block().0) {
            return None;
        }
        Some(TensorRef { len: rows * self.dtype.bytes_for(k), shape: vec![rows, k], ..self.clone() })
    }

    /// Borrows the data as raw bf16 `u16`s when it is bf16 and suitably aligned.
    pub fn as_bf16(&self) -> Option<&[u16]> {
        let b = self.bytes();
        if self.dtype == GgmlType::Bf16
            && self.minus == 0.0
            && cfg!(target_endian = "little")
            && (b.as_ptr() as usize).is_multiple_of(2)
        {
            // SAFETY: aligned, length is even (bf16), and every bit pattern is a valid u16.
            Some(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) })
        } else {
            None
        }
    }

    /// The values as bf16 (`u16`), dequantized and rounded when stored otherwise.
    pub fn to_bf16_vec(&self) -> Vec<u16> {
        if self.dtype == GgmlType::Bf16 && self.minus == 0.0 {
            return self.bytes().as_chunks::<2>().0.iter().map(|&b| u16::from_le_bytes(b)).collect();
        }
        self.to_f32_vec().into_iter().map(f32_to_bf16).collect()
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        let mut v = crate::ggml::dequantize(self.dtype, self.bytes(), self.numel());
        if self.minus != 0.0 {
            v.iter_mut().for_each(|x| *x -= self.minus);
        }
        v
    }
}

/// Round to nearest even (NaN stays NaN).
#[inline]
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

#[inline]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

pub struct Weights {
    pub(crate) tensors: HashMap<String, TensorRef>,
}

impl Weights {
    /// Opens `model.safetensors.index.json` shards or a single `model.safetensors`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(dir: &Path) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let files: Vec<PathBuf> = if index.exists() {
            let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index)?)?;
            let map = v["weight_map"].as_object().context("weight_map")?;
            let mut f: Vec<String> = map.values().filter_map(|x| x.as_str().map(String::from)).collect();
            f.sort();
            f.dedup();
            f.into_iter().map(|n| dir.join(n)).collect()
        } else {
            vec![dir.join("model.safetensors")]
        };
        let mut tensors = HashMap::new();
        for path in files {
            let file = std::fs::File::open(&path).with_context(|| format!("opening {}", path.display()))?;
            // SAFETY: the file is treated as read-only; modifying it while mapped is UB
            // like with any mmap-based loader.
            let mmap = Arc::new(Backing::Mmap(unsafe { Mmap::map(&file)? }));
            let (header_len, meta) = SafeTensors::read_metadata(mmap.as_slice())?;
            let data_start = 8 + header_len;
            for (name, info) in meta.tensors() {
                if info.dtype != Dtype::BF16 {
                    bail!("{name}: dtype {:?} is not supported (expected BF16)", info.dtype);
                }
                let (s, e) = info.data_offsets;
                tensors.insert(
                    name.clone(),
                    TensorRef {
                        backing: mmap.clone(),
                        start: data_start + s,
                        len: e - s,
                        shape: info.shape.clone(),
                        dtype: GgmlType::Bf16,
                        minus: 0.0,
                    },
                );
            }
        }
        Ok(Weights { tensors })
    }

    /// The web loader reads GGUF tensors one at a time instead of opening
    /// safetensors files.
    #[cfg(target_arch = "wasm32")]
    pub fn open(_dir: &Path) -> Result<Self> {
        bail!("safetensors loading is not supported on wasm")
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn get(&self, name: &str) -> Result<&TensorRef> {
        self.tensors.get(name).with_context(|| format!("missing tensor {name}"))
    }

    pub fn get_shaped(&self, name: &str, shape: &[usize]) -> Result<&TensorRef> {
        let t = self.get(name)?;
        ensure!(t.shape == shape, "{name}: shape {:?}, expected {:?}", t.shape, shape);
        Ok(t)
    }
}
