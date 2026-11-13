//! The model's operations on the GPU (the `Backend` trait forwards here).

mod attention;
mod matmul;

use anyhow::{Context, Result};
use yaxus_model::{Config, LayerMeta, PAGE, PassMeta};

use crate::{Buf, Format, Gpu, HEAD_DIM, Kv, MetaBuf, Tensor, Weight, plan};

/// Workgroups per row in the first argmax stage (stage 2 is one 64-wide workgroup).
pub(crate) const ARGMAX_SLICES: usize = 64;

/// Rows of logits one argmax dispatch handles (one per stream of a pass, or the rows of
/// a speculative verification); more are done in batches.
pub(crate) const ARGMAX_ROWS: usize = 512;

/// Columns `add_norm_sg` keeps in registers (16 per lane of its 256); wider rows take the
/// plain `add_norm`.
pub(crate) const ADD_NORM_SG_MAX_COLS: usize = 16 * 256;

/// Splits a 1-D workgroup count into a 2-D grid within the 65535 per-dimension limit.
pub(crate) fn grid(groups: u32) -> (u32, u32) {
    if groups <= 65535 { (groups.max(1), 1) } else { (65535, groups.div_ceil(65535)) }
}

impl Gpu {
    pub(crate) fn name(&self) -> String {
        format!("{}: {}", self.caps.api, self.caps.name)
    }

    /// Pre-creates pooled buffers for the tensors of a `rows`-row pass (see the trait).
    pub(crate) fn reserve_activations(&mut self, cfg: &Config, rows: usize) {
        let (h, inter) = (cfg.hidden_size, cfg.intermediate_size);
        let qkv = cfg.q_dim() + 2 * cfg.kv_dim();
        let padded = rows.div_ceil(64) * 64;
        // (elements, simultaneously live tensors) of one forward pass.
        let shapes =
            [(rows * h, 4), (rows * qkv, 1), (rows * cfg.q_dim(), 2), (padded * 2 * inter, 1), (rows * inter, 1)];
        let tensors: Vec<Tensor> =
            shapes.iter().flat_map(|&(n, count)| (0..count).map(move |_| n)).map(|n| self.tensor(1, n)).collect();
        drop(tensors); // back to the pool
    }

    pub(crate) fn new_kv(&mut self, cfg: &Config, metas: &[LayerMeta], pages: usize, max_pos: usize) -> Result<Kv> {
        anyhow::ensure!(
            cfg.head_dim == HEAD_DIM,
            "head_dim is {}; the attention kernels are written for {HEAD_DIM}",
            cfg.head_dim
        );
        let kv_dim = cfg.kv_dim();
        let size = (pages * PAGE * kv_dim * 4) as u64;
        anyhow::ensure!(size <= self.dev.max_binding(), "KV arena too large per layer");
        let k = metas.iter().map(|_| self.dev.storage("k", size)).collect();
        let v = metas.iter().map(|_| self.dev.storage("v", size)).collect();
        // A model may have only one kind of layer: the other table is then unused.
        let any = metas.first().context("a model without layers")?;
        let sliding = metas.iter().find(|m| !m.full_attn).unwrap_or(any);
        let full = metas.iter().find(|m| m.full_attn).unwrap_or(any);
        let scratch = self.dev.storage("kv-page-scratch", (PAGE * kv_dim * 4) as u64);
        Ok(Kv {
            k,
            v,
            kv_dim,
            scratch,
            rope_sliding: self.rope_table(sliding, max_pos, cfg.head_dim),
            rope_full: self.rope_table(full, max_pos, cfg.head_dim),
        })
    }

    pub(crate) fn copy_page(&mut self, kv: &mut Kv, src: u32, dst: u32) {
        let bytes = (PAGE * kv.kv_dim * 4) as u64;
        for b in kv.k.iter().chain(&kv.v) {
            self.dev.copy(b, src as u64 * bytes, &kv.scratch, 0, bytes);
            self.dev.copy(&kv.scratch, 0, b, dst as u64 * bytes, bytes);
        }
    }

    pub(crate) fn begin_pass(&mut self, pass: &PassMeta) {
        let rows = pass.ids.len();
        let meta: Vec<[u32; 4]> = (0..rows).map(|i| [pass.pos[i], pass.kv_slot[i], pass.pt_off[i], 0]).collect();
        self.upload_ordered(MetaBuf::Ids, bytemuck::cast_slice(&pass.ids));
        self.upload_ordered(MetaBuf::Rows, bytemuck::cast_slice(&meta));
        self.upload_ordered(MetaBuf::Pages, bytemuck::cast_slice(&pass.pages));
        // Tiles: runs of consecutive positions of one sequence, at most 32 rows each.
        let mut tiles: Vec<[u32; 4]> = Vec::new();
        for i in 0..rows {
            let joins = tiles.last().is_some_and(|t: &[u32; 4]| {
                let (r0, n) = (t[0] as usize, t[1] as usize);
                n < plan::ATTN_TILE_ROWS && pass.pt_off[r0] == pass.pt_off[i] && pass.pos[r0] + n as u32 == pass.pos[i]
            });
            match tiles.last_mut() {
                Some(t) if joins => t[1] += 1,
                _ => tiles.push([i as u32, 1, 0, 0]),
            }
        }
        self.upload_ordered(MetaBuf::Tiles, bytemuck::cast_slice(&tiles));
        self.pass_tiles = tiles.len();
        self.pass_rows = rows;
        self.pass_dispatches = 0;
        self.pass_max_pos = pass.pos.iter().copied().max().unwrap_or(0) as usize;
    }

    pub(crate) fn embed(&mut self, table: &Weight) -> Tensor {
        let Weight::Embedding { table, h, fmt } = table else { panic!("embed needs the embedding table") };
        let y = self.tensor(self.pass_rows, *h);
        let ids = self.meta_buf(MetaBuf::Ids);
        match fmt {
            // The Q4_K table is bound twice (nibbles and headers), as the matrix kernels do.
            Format::Q4K => {
                self.dispatch(
                    "embed_q4k",
                    &[y.buf(), &ids, table, table],
                    &[*h as u32, *h as u32],
                    (self.pass_rows as u32, 1),
                );
            }
            _ => {
                self.dispatch("embed", &[y.buf(), table, &ids], &[*h as u32], (self.pass_rows as u32, 1));
            }
        }
        y
    }

    pub(crate) fn rmsnorm(&mut self, x: &Tensor, w: &Weight, offset: f32, eps: f32) -> Tensor {
        let Weight::Vector(wb) = w else { panic!("rmsnorm weight must be a vector") };
        let y = self.tensor(x.rows, x.cols);
        let msl = self.tuning.apple.msl_small && self.caps.msl && x.cols.is_multiple_of(4);
        self.dispatch(
            if msl { "rmsnorm_msl" } else { "rmsnorm" },
            &[y.buf(), x.buf(), wb],
            &[x.cols as u32, offset.to_bits(), eps.to_bits()],
            (x.rows as u32, 1),
        );
        y
    }

    #[allow(clippy::too_many_arguments)] // (the `Backend` trait's signature)
    pub(crate) fn add_norm(
        &mut self,
        x: &mut Tensor,
        delta: &Tensor,
        post_w: &Weight,
        post_offset: f32,
        pre_w: &Weight,
        pre_offset: f32,
        eps: f32,
    ) -> Tensor {
        let (Weight::Vector(w1), Weight::Vector(w2)) = (post_w, pre_w) else { panic!("norm weights") };
        assert_eq!((x.rows, x.cols), (delta.rows, delta.cols));
        let h = self.tensor(x.rows, x.cols);
        let xb = x.buf().clone();
        // (the subgroup kernel holds a row in registers, which bounds its width)
        let name = if self.tuning.apple.msl_small && self.caps.msl && x.cols.is_multiple_of(4) {
            "add_norm_msl"
        } else if self.caps.subgroups && x.cols <= ADD_NORM_SG_MAX_COLS {
            "add_norm_sg"
        } else {
            "add_norm"
        };
        self.dispatch(
            name,
            &[h.buf(), &xb, delta.buf(), w1, w2],
            &[x.cols as u32, post_offset.to_bits(), pre_offset.to_bits(), eps.to_bits()],
            (x.rows as u32, 1),
        );
        h
    }

    pub(crate) fn add_assign(&mut self, a: &mut Tensor, b: &Tensor) {
        let total = a.rows * a.cols;
        assert_eq!(total, b.rows * b.cols);
        let (ab, bb) = (a.buf().clone(), b.buf().clone());
        self.dispatch("add", &[&ab, &bb], &[total as u32], grid(total.div_ceil(256) as u32));
    }

    pub(crate) fn select_rows(&mut self, x: &Tensor, rows: &[usize]) -> Tensor {
        let y = self.tensor(rows.len(), x.cols);
        let bytes = (x.cols * 4) as u64;
        let (src, dst) = (x.buf().clone(), y.buf().clone());
        // Copy runs of consecutive rows at once.
        let mut i = 0;
        while i < rows.len() {
            let mut j = i + 1;
            while j < rows.len() && rows[j] == rows[j - 1] + 1 {
                j += 1;
            }
            self.dev.copy(&src, rows[i] as u64 * bytes, &dst, i as u64 * bytes, (j - i) as u64 * bytes);
            i = j;
        }
        y
    }

    pub(crate) fn argmax_rows(&mut self, logits: &Tensor) -> Vec<u32> {
        self.argmax_prob_rows(logits).into_iter().map(|(i, _)| i).collect()
    }

    // (`&mut`: reading back submits the recorded work)
    #[allow(clippy::wrong_self_convention)]
    pub(crate) fn to_host(&mut self, x: &Tensor) -> Vec<f32> {
        let size = (x.rows * x.cols * 4) as u64;
        let b = self.read(&x.buf().clone(), 0, size);
        bytemuck::cast_slice(&b).to_vec()
    }

    /// Greedy token and its softmax probability for every row (two-stage reduction).
    pub(crate) fn argmax_prob_rows(&mut self, logits: &Tensor) -> Vec<(u32, f32)> {
        if logits.rows > ARGMAX_ROWS {
            // More streams than one dispatch covers (--max-streams has no upper bound).
            let mut out = Vec::with_capacity(logits.rows);
            for start in (0..logits.rows).step_by(ARGMAX_ROWS) {
                let rows: Vec<usize> = (start..(start + ARGMAX_ROWS).min(logits.rows)).collect();
                let part = self.select_rows(logits, &rows);
                out.extend(self.argmax_prob_rows(&part));
            }
            return out;
        }
        let (out, size) = self.argmax_prob_pending(logits);
        let b = self.read(&out, 0, size);
        parse_argmax(&b, logits.cols as u32)
    }

    /// Records the two argmax dispatches and returns the buffer to read (for the async
    /// read of the browser).
    pub(crate) fn argmax_prob_pending(&mut self, logits: &Tensor) -> (Buf, u64) {
        let (out, part) = (self.argmax_out.clone(), self.argmax_part.clone());
        let slice = logits.cols.div_ceil(ARGMAX_SLICES);
        self.dispatch(
            "argmax1",
            &[&part, logits.buf()],
            &[logits.cols as u32, slice as u32],
            (ARGMAX_SLICES as u32, logits.rows as u32),
        );
        self.dispatch("argmax2", &[&out, &part], &[ARGMAX_SLICES as u32], (logits.rows as u32, 1));
        (out, (logits.rows * 8) as u64)
    }
}

/// The `(index, probability)` pairs of an argmax read-back.
pub(crate) fn parse_argmax(bytes: &[u8], cols: u32) -> Vec<(u32, f32)> {
    let v: &[u32] = bytemuck::cast_slice(bytes);
    // A row of only -inf or NaN keeps the kernel's "none" index; the CPU picks the
    // first element then, and an id beyond the vocabulary must not escape.
    v.as_chunks::<2>().0.iter().map(|&[i, x]| (if i < cols { i } else { 0 }, f32::from_bits(x))).collect()
}
