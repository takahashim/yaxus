//! Attention over the paged KV cache, and the RoPE tables it reads.

use wgpu::util::DeviceExt;
use yaxus_model::{Config, LayerMeta};

use crate::plan::{AttentionCase, AttentionPlan};
use crate::{Gpu, HEAD_DIM, Kv, MetaBuf, Tensor, Weight};

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
        assert_eq!(t, self.pass_rows, "attention rows differ from the pass");
        let (kc, vc) = (kv.k[layer.index].clone(), kv.v[layer.index].clone());
        let rope = if layer.full_attn { kv.rope_full.clone() } else { kv.rope_sliding.clone() };
        // (cos and sin, HEAD_DIM / 2 f32 each per position)
        let rope_bytes = (self.pass_max_pos as u64 + 1) * (HEAD_DIM as u64 / 2) * 4 * 2;
        assert!(rope_bytes <= rope.size(), "position beyond the rope table");
        let (rows, pages) = (self.meta_buf(MetaBuf::Rows), self.meta_buf(MetaBuf::Pages));
        let q = self.tensor(t, hq * HEAD_DIM);
        // (the fourth immediate: the rope table's half length, which only the MSL kernel reads)
        let msl = self.tuning.msl_small && self.caps.msl;
        self.dispatch(
            if msl { "qk_rope_msl" } else { "qk_rope" },
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
            max_pos: self.pass_max_pos,
            tiles: self.pass_tiles,
        };
        let g = hq / hkv;
        let plan = case.plan(&self.caps, &self.tuning);
        if let AttentionPlan::Split { g, hg, ch, nch } = plan {
            let part = self.tensor(t * hkv * nch * g, HEAD_DIM);
            let ml = self.tensor(t * hkv * nch * g, 2);
            self.dispatch(
                &crate::kernels::name::attn_split(g, ch, hg),
                &[part.buf(), ml.buf(), q.buf(), &kc, &vc, &rows, &pages],
                &[hq as u32, hkv as u32, window, scale.to_bits(), nch as u32],
                ((hkv * (g / hg) * nch) as u32, t as u32),
            );
            self.dispatch(
                "attn_combine",
                &[o.buf(), part.buf(), ml.buf()],
                &[hq as u32, hkv as u32, nch as u32],
                (hq as u32, t as u32),
            );
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

    pub(crate) fn rope_table(&self, meta: &LayerMeta, max_len: usize, d: usize) -> wgpu::Buffer {
        let half = d / 2;
        let mut tab = vec![0f32; 2 * max_len * half];
        let (cos, sin) = tab.split_at_mut(max_len * half);
        for pos in 0..max_len {
            meta.rope.cos_sin(pos, &mut cos[pos * half..(pos + 1) * half], &mut sin[pos * half..(pos + 1) * half]);
        }
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rope"),
            contents: bytemuck::cast_slice(&tab),
            usage: wgpu::BufferUsages::STORAGE,
        })
    }
}
