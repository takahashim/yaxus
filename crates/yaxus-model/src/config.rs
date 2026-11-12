use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

/// The subset of `config.json` (model_type `plamo3`) that inference needs.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub window_size: usize,
    pub sliding_window_pattern: usize,
    pub rope_theta: f64,
    pub rope_local_theta: f64,
    #[serde(default)]
    pub scale_embedding: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    pub rope_parameters: Option<RopeParameters>,
    #[serde(default = "default_factor")]
    pub rope_scaling_factor: f64,
    #[serde(default = "default_original")]
    pub initial_context_length: usize,
}

fn default_factor() -> f64 {
    256.0
}
fn default_original() -> usize {
    4096
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub full_attention: RopeParam,
    pub sliding_attention: RopeParam,
}

/// RoPE of one kind of layer (`rope_parameters.*` of `config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct RopeParam {
    pub rope_theta: f64,
    #[serde(flatten)]
    pub scaling: RopeScaling,
}

/// `rope_type` and its parameters.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "rope_type", rename_all = "lowercase")]
pub enum RopeScaling {
    Default,
    Yarn(Yarn),
}

/// YaRN context extension.
#[derive(Debug, Clone, Deserialize)]
pub struct Yarn {
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
    #[serde(default)]
    pub truncate: bool,
}

impl RopeParam {
    pub fn plain(rope_theta: f64) -> Self {
        RopeParam { rope_theta, scaling: RopeScaling::Default }
    }

    pub fn yarn(rope_theta: f64, yarn: Yarn) -> Self {
        RopeParam { rope_theta, scaling: RopeScaling::Yarn(yarn) }
    }
}

impl Config {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let cfg: Config = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub(crate) fn validate(&self) -> Result<()> {
        ensure!(!self.scale_embedding, "scale_embedding=true is not supported");
        ensure!(!self.tie_word_embeddings, "tied embeddings are not supported");
        let sizes = [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("intermediate_size", self.intermediate_size),
            ("vocab_size", self.vocab_size),
            ("sliding_window_pattern", self.sliding_window_pattern),
        ];
        for (name, v) in sizes {
            ensure!(v > 0, "{name} must be positive");
        }
        // With sliding layers, an empty window leaves them nothing to attend to (NaN).
        ensure!(
            self.sliding_window_pattern == 1 || self.window_size > 0,
            "window_size must be positive when there are sliding-window layers"
        );
        ensure!(
            self.num_attention_heads.is_multiple_of(self.num_key_value_heads),
            "num_attention_heads ({}) is not a multiple of num_key_value_heads ({})",
            self.num_attention_heads,
            self.num_key_value_heads
        );
        ensure!(self.head_dim.is_multiple_of(2), "head_dim ({}) must be even", self.head_dim);
        Ok(())
    }

    pub fn is_full_attn(&self, layer: usize) -> bool {
        (layer + 1).is_multiple_of(self.sliding_window_pattern)
    }

    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    fn rope_param(&self, full: bool) -> RopeParam {
        if let Some(p) = &self.rope_parameters {
            return if full { p.full_attention.clone() } else { p.sliding_attention.clone() };
        }
        // Configurations without `rope_parameters`: YaRN on the full-attention layers.
        if full {
            RopeParam::yarn(
                self.rope_theta,
                Yarn {
                    factor: self.rope_scaling_factor,
                    original_max_position_embeddings: self.initial_context_length,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                    truncate: false,
                },
            )
        } else {
            RopeParam::plain(self.rope_local_theta)
        }
    }

    /// Per-layer static metadata.
    pub fn layer_meta(&self, layer: usize) -> Result<LayerMeta> {
        let full = self.is_full_attn(layer);
        let rope = Rope::new(self.head_dim, &self.rope_param(full))?;
        Ok(LayerMeta { index: layer, full_attn: full, window: if full { None } else { Some(self.window_size) }, rope })
    }
}

#[derive(Debug, Clone)]
pub struct LayerMeta {
    pub index: usize,
    pub full_attn: bool,
    /// Sliding window: query at `p` sees keys in `(p - window, p]`.
    pub window: Option<usize>,
    pub rope: Rope,
}

/// Half-split RoPE parameters: `cos/sin(pos * inv_freq[j]) * concentration`
/// applied to dims `j` and `j + dim/2`.
#[derive(Debug, Clone)]
pub struct Rope {
    pub inv_freq: Vec<f32>,
    pub concentration: f32,
}

impl Rope {
    /// Mirrors `RotaryEmbedding._compute_buffers` (computed in f32 like torch).
    pub fn new(dim: usize, p: &RopeParam) -> Result<Self> {
        let base = p.rope_theta;
        let freq: Vec<f32> = (0..dim).step_by(2).map(|i| (base as f32).powf(i as f32 / dim as f32)).collect();
        match &p.scaling {
            RopeScaling::Default => Ok(Rope { inv_freq: freq.iter().map(|f| 1.0 / f).collect(), concentration: 1.0 }),
            RopeScaling::Yarn(y) => {
                ensure!(!y.truncate, "yarn truncate=true is not supported");
                let (factor, beta_fast, beta_slow) = (y.factor, y.beta_fast, y.beta_slow);
                let orig = y.original_max_position_embeddings as f64;
                let d_h = (dim / 2) as f64;
                let pi = std::f64::consts::PI;
                let low = d_h * (orig / (beta_fast * 2.0 * pi)).ln() / base.ln();
                let high = d_h * (orig / (beta_slow * 2.0 * pi)).ln() / base.ln();
                ensure!(0.0 < low && low < high && high < d_h - 1.0);
                let inv_freq = freq
                    .iter()
                    .enumerate()
                    .map(|(j, &f)| {
                        let interp = 1.0 / (factor as f32 * f);
                        let extrap = 1.0 / f;
                        let ramp = (j as f32 - low as f32) / (high - low) as f32;
                        let mask = 1.0 - ramp.clamp(0.0, 1.0);
                        interp * (1.0 - mask) + extrap * mask
                    })
                    .collect();
                Ok(Rope { inv_freq, concentration: (0.1 * factor.ln() + 1.0) as f32 })
            }
        }
    }

    /// Returns `(cos, sin)` for one position, each `dim/2` long, already scaled.
    pub fn cos_sin(&self, pos: usize, cos: &mut [f32], sin: &mut [f32]) {
        for (j, &f) in self.inv_freq.iter().enumerate() {
            let a = pos as f32 * f; // torch: f32 einsum
            cos[j] = (a as f64).cos() as f32 * self.concentration;
            sin[j] = (a as f64).sin() as f32 * self.concentration;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_parameters_as_in_the_checkpoints() {
        let p: RopeParameters = serde_json::from_str(
            r#"{"full_attention": {"rope_type": "yarn", "rope_theta": 1000000, "factor": 256.0,
                    "original_max_position_embeddings": 4096, "beta_fast": 32.0, "beta_slow": 1.0,
                    "truncate": false},
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000}}"#,
        )
        .unwrap();
        assert!(
            matches!(&p.full_attention.scaling, RopeScaling::Yarn(y) if y.original_max_position_embeddings == 4096)
        );
        assert!(matches!(p.sliding_attention.scaling, RopeScaling::Default));
        assert_eq!(p.sliding_attention.rope_theta, 10000.0);
        assert!(Rope::new(128, &p.full_attention).unwrap().concentration > 1.0);
        let other = serde_json::from_str::<RopeParam>(r#"{"rope_type": "linear", "rope_theta": 1.0}"#);
        assert!(other.unwrap_err().to_string().contains("linear"));
    }
}
