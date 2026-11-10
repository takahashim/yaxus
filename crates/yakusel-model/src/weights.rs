//! Memory-mapped safetensors (possibly sharded) access.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

/// A raw bf16 tensor living inside a memory-mapped file.
#[derive(Clone)]
pub struct TensorRef {
    mmap: Arc<Mmap>,
    start: usize,
    len: usize,
    pub shape: Vec<usize>,
}

impl TensorRef {
    /// Little-endian bf16 bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.mmap[self.start..self.start + self.len]
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Borrows the data as raw bf16 `u16`s when the mapping is suitably aligned.
    pub fn as_bf16(&self) -> Option<&[u16]> {
        let b = self.bytes();
        if cfg!(target_endian = "little") && (b.as_ptr() as usize) % 2 == 0 {
            // SAFETY: aligned, length is even (bf16), and every bit pattern is a valid u16.
            Some(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) })
        } else {
            None
        }
    }

    /// bf16 values as raw `u16`, copied out (handles unaligned offsets).
    pub fn to_bf16_vec(&self) -> Vec<u16> {
        self.bytes()
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect()
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        self.bytes()
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect()
    }
}

#[inline]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

pub struct Weights {
    tensors: HashMap<String, TensorRef>,
}

impl Weights {
    /// Opens `model.safetensors.index.json` shards or a single `model.safetensors`.
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
            let mmap = Arc::new(unsafe { Mmap::map(&file)? });
            let (header_len, meta) = SafeTensors::read_metadata(&mmap[..])?;
            let data_start = 8 + header_len;
            for (name, info) in meta.tensors() {
                if info.dtype != Dtype::BF16 {
                    bail!("{name}: dtype {:?} is not supported (expected BF16)", info.dtype);
                }
                let (s, e) = info.data_offsets;
                tensors.insert(
                    name.clone(),
                    TensorRef {
                        mmap: mmap.clone(),
                        start: data_start + s,
                        len: e - s,
                        shape: info.shape.clone(),
                    },
                );
            }
        }
        Ok(Weights { tensors })
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
