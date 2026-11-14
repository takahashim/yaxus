//! Attention over the paged KV cache, and the RoPE tables it reads.

use yaxus_model::{Config, LayerMeta};

use crate::dev::Buf;
use crate::kernels::KernelId;
use crate::plan::{self, AttentionCase, AttentionPlan};
use crate::{Gpu, HEAD_DIM, Kv, MetaBuf, Tensor, Weight};

/// The rows and heads of an attention.
#[derive(Clone, Copy)]
struct Heads {
    t: usize,
    hq: usize,
    hkv: usize,
}

impl Gpu {
    pub(crate) fn attention(
        &mut self,
        qkv: &Tensor,
        q_norm: &Weight,
        k_norm: &Weight,
        kv: &mut Kv,
        cfg: &Config,
        layer: &LayerMeta,
    ) -> Tensor {
        let (Weight::Vector(qn), Weight::Vector(kn)) = (q_norm, k_norm) else { panic!("q/k norm") };
        let (hq, hkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        let t = qkv.rows;
        assert_eq!(t, self.pass.rows, "attention rows differ from the pass");
        let (kc, vc) = (kv.k[layer.index].clone(), kv.v[layer.index].clone());
        let rope = if layer.full_attn { kv.rope_full.clone() } else { kv.rope_sliding.clone() };
        // (cos and sin, HEAD_DIM / 2 f32 each per position)
        let rope_bytes = (self.pass.max_pos as u64 + 1) * (HEAD_DIM as u64 / 2) * 4 * 2;
        assert!(rope_bytes <= rope.size(), "position beyond the rope table");
        let (rows, pages) = (self.meta_buf(MetaBuf::Rows), self.meta_buf(MetaBuf::Pages));
        let q = self.tensor(t, hq * HEAD_DIM);
        // (the fourth immediate: the rope table's half length, which only the MSL kernel reads)
        self.dispatch(
            KernelId::QkRope { msl: plan::msl_small(&self.caps, &self.tuning) },
            &[q.buf(), &kc, &vc, qkv.buf(), qn, kn, &rope, &rows],
            &[hq as u32, hkv as u32, cfg.rms_norm_eps.to_bits(), (rope.size() / 8) as u32],
            ((hq + hkv) as u32, t as u32),
        );
        let o = self.tensor(t, hq * HEAD_DIM);
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let window = layer.window.unwrap_or(0) as u32;
        let case = AttentionCase {
            rows: t,
            hq,
            hkv,
            window: layer.window,
            max_pos: self.pass.max_pos,
            tiles: self.pass.tiles,
        };
        let heads = Heads { t, hq, hkv };
        match case.plan(&self.caps, &self.tuning) {
            AttentionPlan::Split { g, hg, ch, nch } => {
                let (part, ml) = self.partials(heads, nch);
                self.dispatch(
                    KernelId::AttnSplit { g, ch, hg },
                    &[part.buf(), ml.buf(), q.buf(), &kc, &vc, &rows, &pages],
                    &[hq as u32, hkv as u32, window, scale.to_bits(), nch as u32],
                    ((hkv * (g / hg) * nch) as u32, t as u32),
                );
                self.combine(heads, &o, &part, &ml, nch);
            }
            AttentionPlan::Tile { nch, ch } => {
                let tiles = self.meta_buf(MetaBuf::Tiles);
                // (one chunk writes the output directly: the partials are unused then)
                let (part, ml) =
                    if nch > 1 { self.partials(heads, nch) } else { (self.tensor(1, HEAD_DIM), self.tensor(1, 2)) };
                self.dispatch(
                    KernelId::AttnTile,
                    &[o.buf(), part.buf(), ml.buf(), q.buf(), &kc, &vc, &rows, &pages, &tiles],
                    &[hq as u32, hkv as u32, window, scale.to_bits(), nch as u32, ch as u32],
                    ((self.pass.tiles * nch) as u32, hq as u32),
                );
                if nch > 1 {
                    self.combine(heads, &o, &part, &ml, nch);
                }
            }
            AttentionPlan::PerRow => {
                self.dispatch(
                    KernelId::Attention,
                    &[o.buf(), q.buf(), &kc, &vc, &rows, &pages],
                    &[hq as u32, hkv as u32, window, scale.to_bits()],
                    (hq as u32, t as u32),
                );
            }
        }
        o
    }

    /// The partial outputs of `nch` chunks of keys, and their (max, sum) pairs.
    fn partials(&mut self, h: Heads, nch: usize) -> (Tensor, Tensor) {
        let n = h.t * h.hq * nch;
        (self.tensor(n, HEAD_DIM), self.tensor(n, 2))
    }

    /// Merges the partials of `nch` chunks into `o`.
    fn combine(&mut self, h: Heads, o: &Tensor, part: &Tensor, ml: &Tensor, nch: usize) {
        self.dispatch(
            KernelId::AttnCombine,
            &[o.buf(), part.buf(), ml.buf()],
            &[h.hq as u32, h.hkv as u32, nch as u32],
            (h.hq as u32, h.t as u32),
        );
    }

    pub(crate) fn rope_table(&mut self, meta: &LayerMeta, max_len: usize, d: usize) -> Buf {
        let half = d / 2;
        let mut tab = vec![0f32; 2 * max_len * half];
        let (cos, sin) = tab.split_at_mut(max_len * half);
        for pos in 0..max_len {
            meta.rope.cos_sin(pos, &mut cos[pos * half..(pos + 1) * half], &mut sin[pos * half..(pos + 1) * half]);
        }
        self.rec.dev.upload("rope", bytemuck::cast_slice(&tab))
    }
}
