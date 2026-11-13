//! Attention over the paged KV cache, and the RoPE tables it reads.

use yaxus_model::{Config, LayerMeta};

use crate::dev::Buf;
use crate::plan::{AttentionCase, AttentionPlan};
use crate::{Gpu, HEAD_DIM, Kv, MetaBuf, Tensor, Weight};

impl Gpu {
    /// Per-head RMSNorm and half-split RoPE of q and k, and the raw k/v write into the
    /// row's KV slot (the `qk_rope` kernel; the split/tile/row kernels follow it).
    pub(crate) fn qk_rope(
        &mut self,
        qkv: &Tensor,
        q_norm: &Weight,
        k_norm: &Weight,
        kv: &mut Kv,
        cfg: &Config,
        layer: &LayerMeta,
    ) -> Tensor {
        let Weight::Vector(qn) = q_norm else { panic!("q/k norm") };
        let Weight::Vector(kn) = k_norm else { panic!("q/k norm") };
        let (hq, hkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        let t = qkv.rows;
        let (kc, vc) = (kv.k[layer.index].clone(), kv.v[layer.index].clone());
        let rope = if layer.full_attn { kv.rope_full.clone() } else { kv.rope_sliding.clone() };
        // (cos and sin, HEAD_DIM / 2 f32 each per position)
        let rope_bytes = (self.pass_max_pos as u64 + 1) * (HEAD_DIM as u64 / 2) * 4 * 2;
        assert!(rope_bytes <= rope.size(), "position beyond the rope table");
        let rows = self.meta_buf(MetaBuf::Rows);
        let q = self.tensor(t, hq * HEAD_DIM);
        // (the fourth immediate: the rope table's half length, which only the MSL kernel reads)
        let msl = self.tuning.apple.msl_small && self.caps.msl;
        self.dispatch(
            if msl { "qk_rope_msl" } else { "qk_rope" },
            &[q.buf(), &kc, &vc, qkv.buf(), qn, kn, &rope, &rows],
            &[hq as u32, hkv as u32, cfg.rms_norm_eps.to_bits(), (rope.size() / 8) as u32],
            ((hq + hkv) as u32, t as u32),
        );
        q
    }

    pub(crate) fn attention(
        &mut self,
        qkv: &Tensor,
        q_norm: &Weight,
        k_norm: &Weight,
        kv: &mut Kv,
        cfg: &Config,
        layer: &LayerMeta,
    ) -> Tensor {
        let (hq, hkv) = (cfg.num_attention_heads, cfg.num_key_value_heads);
        let t = qkv.rows;
        assert_eq!(t, self.pass_rows, "attention rows differ from the pass");
        let (kc, vc) = (kv.k[layer.index].clone(), kv.v[layer.index].clone());
        let (rows, pages) = (self.meta_buf(MetaBuf::Rows), self.meta_buf(MetaBuf::Pages));
        let q = self.qk_rope(qkv, q_norm, k_norm, kv, cfg, layer);
        let o = self.tensor(t, hq * HEAD_DIM);
        let scale = 1.0 / (cfg.head_dim as f32).sqrt();
        let window = layer.window.unwrap_or(0) as u32;
        let case = AttentionCase {
            rows: t,
            hq,
            hkv,
            window: layer.window,
            max_pos: self.pass_max_pos,
            tiles: self.pass_tiles,
        };
        let g = hq / hkv;
        let plan = case.plan(&self.caps, &self.tuning);
        if let AttentionPlan::Split { g, hg, ch, nch } = plan {
            let part = self.tensor(t * hkv * nch * g, HEAD_DIM);
            let ml = self.tensor(t * hkv * nch * g, 2);
            self.attn_split(&q, &kc, &vc, &part, &ml, hq, hkv, g, hg, ch, nch, window, scale, t);
            self.attn_combine(&o, &part, &ml, hq, hkv, nch, t);
        } else if let AttentionPlan::Tile { nch, ch } = plan {
            let tiles = self.meta_buf(MetaBuf::Tiles);
            let (part, ml) = if nch > 1 {
                (self.tensor(t * hkv * nch * g, HEAD_DIM), self.tensor(t * hkv * nch * g, 2))
            } else {
                (self.tensor(1, HEAD_DIM), self.tensor(1, 2))
            };
            self.dispatch(
                "attn_tile",
                &[o.buf(), part.buf(), ml.buf(), q.buf(), &kc, &vc, &rows, &pages, &tiles],
                &[hq as u32, hkv as u32, window, scale.to_bits(), nch as u32, ch as u32],
                ((self.pass_tiles * nch) as u32, hq as u32),
            );
            if nch > 1 {
                self.dispatch(
                    "attn_combine",
                    &[o.buf(), part.buf(), ml.buf()],
                    &[hq as u32, hkv as u32, nch as u32],
                    (hq as u32, t as u32),
                );
            }
        } else {
            self.dispatch(
                "attention",
                &[o.buf(), q.buf(), &kc, &vc, &rows, &pages],
                &[hq as u32, hkv as u32, window, scale.to_bits()],
                (hq as u32, t as u32),
            );
        }
        o
    }

    /// The split attention (part 1) and its combine (part 2), as `attention` runs them
    /// (also used by the test hooks to read the partials back).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_split(
        &mut self,
        q: &Tensor,
        kc: &Buf,
        vc: &Buf,
        part: &Tensor,
        ml: &Tensor,
        hq: usize,
        hkv: usize,
        g: usize,
        hg: usize,
        ch: usize,
        nch: usize,
        window: u32,
        scale: f32,
        t: usize,
    ) {
        let (rows, pages) = (self.meta_buf(MetaBuf::Rows), self.meta_buf(MetaBuf::Pages));
        self.dispatch(
            &crate::kernels::name::attn_split(g, ch, hg),
            &[part.buf(), ml.buf(), q.buf(), kc, vc, &rows, &pages],
            &[hq as u32, hkv as u32, window, scale.to_bits(), nch as u32],
            ((hkv * (g / hg) * nch) as u32, t as u32),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_combine(&mut self, o: &Tensor, part: &Tensor, ml: &Tensor, hq: usize, hkv: usize, nch: usize, t: usize) {
        self.dispatch(
            "attn_combine",
            &[o.buf(), part.buf(), ml.buf()],
            &[hq as u32, hkv as u32, nch as u32],
            (hq as u32, t as u32),
        );
    }

    pub(crate) fn rope_table(&mut self, meta: &LayerMeta, max_len: usize, d: usize) -> Buf {
        let half = d / 2;
        let mut tab = vec![0f32; 2 * max_len * half];
        let (cos, sin) = tab.split_at_mut(max_len * half);
        for pos in 0..max_len {
            meta.rope.cos_sin(pos, &mut cos[pos * half..(pos + 1) * half], &mut sin[pos * half..(pos + 1) * half]);
        }
        self.dev.upload("rope", bytemuck::cast_slice(&tab))
    }
}
