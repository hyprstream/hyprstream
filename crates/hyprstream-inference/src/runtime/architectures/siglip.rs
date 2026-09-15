//! SigLIP (Sigmoid Loss for Language-Image Pre-training) Vision Transformer
//!
//! Implementation based on the timm (PyTorch Image Models) SigLIP architecture.
//! This is a standalone implementation that can be used by multimodal models like Janus.
//!
//! Architecture:
//! - Patch embedding (Conv2d)
//! - Position embeddings (learned)
//! - Transformer blocks (self-attention + MLP with pre-norm)
//! - Final layer norm
//! - Optional attention pooling head

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tch::{Device, Kind as DType, Tensor};
use tracing::debug;

/// Configuration for SigLIP vision encoder
#[derive(Debug, Clone)]
pub struct SigLIPConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub use_attention_pool: bool,
}

impl Default for SigLIPConfig {
    fn default() -> Self {
        // SigLIP Large/16 @ 384px defaults
        Self {
            image_size: 384,
            patch_size: 16,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            use_attention_pool: true,
        }
    }
}

/// Layer normalization
#[derive(Debug)]
struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let weight_key = format!("{prefix}.weight");
        let bias_key = format!("{prefix}.bias");

        let weight = weights
            .get(&weight_key)
            .ok_or_else(|| anyhow!("Missing LayerNorm weight: {}", weight_key))?
            .to_device(device)
            .to_kind(dtype);

        let bias = weights
            .get(&bias_key)
            .ok_or_else(|| anyhow!("Missing LayerNorm bias: {}", bias_key))?
            .to_device(device)
            .to_kind(dtype);

        Ok(Self {
            weight,
            bias,
            eps: 1e-6,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.layer_norm([self.weight.size()[0]], Some(&self.weight), Some(&self.bias), self.eps, true))
    }
}

/// Self-attention layer with fused QKV projection
#[derive(Debug)]
struct SigLIPAttention {
    qkv_weight: Tensor,
    qkv_bias: Tensor,
    proj_weight: Tensor,
    proj_bias: Tensor,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl SigLIPAttention {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        hidden_size: usize,
        num_heads: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let qkv_weight = weights
            .get(&format!("{prefix}.qkv.weight"))
            .ok_or_else(|| anyhow!("Missing QKV weight: {}.qkv.weight", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let qkv_bias = weights
            .get(&format!("{prefix}.qkv.bias"))
            .ok_or_else(|| anyhow!("Missing QKV bias: {}.qkv.bias", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let proj_weight = weights
            .get(&format!("{prefix}.proj.weight"))
            .ok_or_else(|| anyhow!("Missing proj weight: {}.proj.weight", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let proj_bias = weights
            .get(&format!("{prefix}.proj.bias"))
            .ok_or_else(|| anyhow!("Missing proj bias: {}.proj.bias", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let head_dim = hidden_size / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        Ok(Self {
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            num_heads,
            head_dim,
            scale,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden_size) = x.size3()?;

        // QKV projection: [B, N, hidden_size] -> [B, N, 3 * hidden_size]
        let qkv = x.matmul(&self.qkv_weight.tr()) + &self.qkv_bias;

        // Reshape to [B, N, 3, num_heads, head_dim]
        let qkv = qkv.reshape([batch_size, seq_len, 3, self.num_heads as i64, self.head_dim as i64]);

        // Permute to [3, B, num_heads, N, head_dim]
        let qkv = qkv.permute([2, 0, 3, 1, 4]);

        // Split into Q, K, V
        let q = qkv.get(0) * self.scale;
        let k = qkv.get(1);
        let v = qkv.get(2);

        // Attention scores: [B, num_heads, N, N]
        let attn = q.matmul(&k.transpose(-2, -1));
        let attn = attn.softmax(-1, attn.kind());

        // Apply attention to values: [B, num_heads, N, head_dim]
        let out = attn.matmul(&v);

        // Reshape back: [B, N, hidden_size]
        let out = out
            .transpose(1, 2)
            .contiguous()
            .reshape([batch_size, seq_len, (self.num_heads * self.head_dim) as i64]);

        // Output projection
        let out = out.matmul(&self.proj_weight.tr()) + &self.proj_bias;

        Ok(out)
    }
}

/// MLP block with GELU activation
#[derive(Debug)]
struct SigLIPMlp {
    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,
}

impl SigLIPMlp {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let fc1_weight = weights
            .get(&format!("{prefix}.fc1.weight"))
            .ok_or_else(|| anyhow!("Missing MLP fc1 weight: {}.fc1.weight", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let fc1_bias = weights
            .get(&format!("{prefix}.fc1.bias"))
            .ok_or_else(|| anyhow!("Missing MLP fc1 bias: {}.fc1.bias", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let fc2_weight = weights
            .get(&format!("{prefix}.fc2.weight"))
            .ok_or_else(|| anyhow!("Missing MLP fc2 weight: {}.fc2.weight", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let fc2_bias = weights
            .get(&format!("{prefix}.fc2.bias"))
            .ok_or_else(|| anyhow!("Missing MLP fc2 bias: {}.fc2.bias", prefix))?
            .to_device(device)
            .to_kind(dtype);

        Ok(Self {
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // fc1 + GELU
        let mut x = x.matmul(&self.fc1_weight.tr()) + &self.fc1_bias;
        x = x.gelu("none");

        // fc2
        x = x.matmul(&self.fc2_weight.tr()) + &self.fc2_bias;

        Ok(x)
    }
}

/// Transformer block (attention + MLP with pre-norm and residual)
#[derive(Debug)]
struct SigLIPBlock {
    norm1: LayerNorm,
    attn: SigLIPAttention,
    norm2: LayerNorm,
    mlp: SigLIPMlp,
}

impl SigLIPBlock {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        hidden_size: usize,
        num_heads: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let norm1 = LayerNorm::from_weights(
            weights,
            &format!("{prefix}.norm1"),
            device,
            dtype,
        )?;

        let attn = SigLIPAttention::from_weights(
            weights,
            &format!("{prefix}.attn"),
            hidden_size,
            num_heads,
            device,
            dtype,
        )?;

        let norm2 = LayerNorm::from_weights(
            weights,
            &format!("{prefix}.norm2"),
            device,
            dtype,
        )?;

        let mlp = SigLIPMlp::from_weights(
            weights,
            &format!("{prefix}.mlp"),
            device,
            dtype,
        )?;

        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-norm + attention + residual
        let x = x + self.attn.forward(&self.norm1.forward(x)?)?;

        // Pre-norm + MLP + residual
        let x = &x + self.mlp.forward(&self.norm2.forward(&x)?)?;

        Ok(x)
    }
}

/// Attention pooling head (MAPool - Multihead Attention Pooling)
#[derive(Debug)]
struct AttentionPool {
    latent: Tensor,
    q_weight: Tensor,
    q_bias: Tensor,
    kv_weight: Tensor,
    kv_bias: Tensor,
    proj_weight: Tensor,
    proj_bias: Tensor,
    norm: LayerNorm,
    mlp: SigLIPMlp,
    num_heads: usize,
    head_dim: usize,
}

impl AttentionPool {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        hidden_size: usize,
        num_heads: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let latent = weights
            .get(&format!("{prefix}.latent"))
            .ok_or_else(|| anyhow!("Missing attention pool latent: {}.latent", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let q_weight = weights
            .get(&format!("{prefix}.q.weight"))
            .ok_or_else(|| anyhow!("Missing attention pool Q weight: {}.q.weight", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let q_bias = weights
            .get(&format!("{prefix}.q.bias"))
            .ok_or_else(|| anyhow!("Missing attention pool Q bias: {}.q.bias", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let kv_weight = weights
            .get(&format!("{prefix}.kv.weight"))
            .ok_or_else(|| anyhow!("Missing attention pool KV weight: {}.kv.weight", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let kv_bias = weights
            .get(&format!("{prefix}.kv.bias"))
            .ok_or_else(|| anyhow!("Missing attention pool KV bias: {}.kv.bias", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let proj_weight = weights
            .get(&format!("{prefix}.proj.weight"))
            .ok_or_else(|| anyhow!("Missing attention pool proj weight: {}.proj.weight", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let proj_bias = weights
            .get(&format!("{prefix}.proj.bias"))
            .ok_or_else(|| anyhow!("Missing attention pool proj bias: {}.proj.bias", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let norm = LayerNorm::from_weights(
            weights,
            &format!("{prefix}.norm"),
            device,
            dtype,
        )?;

        let mlp = SigLIPMlp::from_weights(
            weights,
            &format!("{prefix}.mlp"),
            device,
            dtype,
        )?;

        let head_dim = hidden_size / num_heads;

        Ok(Self {
            latent,
            q_weight,
            q_bias,
            kv_weight,
            kv_bias,
            proj_weight,
            proj_bias,
            norm,
            mlp,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = x.size3()?;

        // Expand latent queries: [num_queries, hidden_size] -> [B, num_queries, hidden_size]
        // The latent tensor may have extra dimensions, so squeeze first
        let latent = self.latent.squeeze();  // Remove any singleton dimensions
        let latent = if latent.dim() == 1 {
            // If it's 1D [hidden_size], reshape to [1, hidden_size]
            latent.unsqueeze(0)
        } else {
            // Should be 2D [num_queries, hidden_size]
            latent
        };
        let latent = latent.unsqueeze(0).expand([batch_size, -1, -1], false);

        // Q from latent
        let q = latent.matmul(&self.q_weight.tr()) + &self.q_bias;
        let num_queries = q.size()[1];

        // K, V from input
        let kv = x.matmul(&self.kv_weight.tr()) + &self.kv_bias;
        let kv = kv.reshape([batch_size, seq_len, 2, self.num_heads as i64, self.head_dim as i64]);
        let kv = kv.permute([2, 0, 3, 1, 4]);
        let k = kv.get(0);
        let v = kv.get(1);

        // Reshape Q: [B, num_queries, num_heads, head_dim]
        let q = q.reshape([batch_size, num_queries, self.num_heads as i64, self.head_dim as i64]);
        let q = q.transpose(1, 2);  // [B, num_heads, num_queries, head_dim]

        // Attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(-2, -1)) * scale;
        let attn = attn.softmax(-1, attn.kind());

        // Apply attention: [B, num_heads, num_queries, head_dim]
        let out = attn.matmul(&v);

        // Reshape: [B, num_queries, hidden_size]
        let out = out
            .transpose(1, 2)
            .contiguous()
            .reshape([batch_size, num_queries, hidden_size]);

        // Output projection
        let out = out.matmul(&self.proj_weight.tr()) + &self.proj_bias;

        // MLP (with residual and norm)
        let out = &out + self.mlp.forward(&self.norm.forward(&out)?)?;

        Ok(out)
    }
}

/// SigLIP Vision Transformer
pub struct SigLIPVisionTransformer {
    patch_embed_weight: Tensor,
    patch_embed_bias: Option<Tensor>,
    pos_embed: Tensor,
    blocks: Vec<SigLIPBlock>,
    norm: LayerNorm,
    attn_pool: Option<AttentionPool>,
    config: SigLIPConfig,
}

impl SigLIPVisionTransformer {
    /// Load SigLIP model from weights
    pub fn from_weights(
        config: &SigLIPConfig,
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        debug!("Loading SigLIP vision transformer with prefix: {}", prefix);

        // Patch embedding (Conv2d projection)
        let patch_embed_weight = weights
            .get(&format!("{prefix}.patch_embed.proj.weight"))
            .ok_or_else(|| anyhow!("Missing patch embed weight: {}.patch_embed.proj.weight", prefix))?
            .to_device(device)
            .to_kind(dtype);

        let patch_embed_bias = weights
            .get(&format!("{prefix}.patch_embed.proj.bias"))
            .map(|t| t.to_device(device).to_kind(dtype));

        // Position embeddings
        let pos_embed = weights
            .get(&format!("{prefix}.pos_embed"))
            .ok_or_else(|| anyhow!("Missing position embeddings: {}.pos_embed", prefix))?
            .to_device(device)
            .to_kind(dtype);

        debug!("Position embeddings shape: {:?}", pos_embed.size());

        // Transformer blocks
        let mut blocks = Vec::new();
        for i in 0..config.num_hidden_layers {
            let block_prefix = format!("{prefix}.blocks.{i}");
            let block = SigLIPBlock::from_weights(
                weights,
                &block_prefix,
                config.hidden_size,
                config.num_attention_heads,
                device,
                dtype,
            )?;
            blocks.push(block);
        }
        debug!("Loaded {} transformer blocks", blocks.len());

        // Final layer norm
        let norm = LayerNorm::from_weights(
            weights,
            &format!("{prefix}.norm"),
            device,
            dtype,
        )?;

        // Optional attention pooling
        let attn_pool = if config.use_attention_pool {
            Some(AttentionPool::from_weights(
                weights,
                &format!("{prefix}.attn_pool"),
                config.hidden_size,
                config.num_attention_heads,
                device,
                dtype,
            )?)
        } else {
            None
        };

        Ok(Self {
            patch_embed_weight,
            patch_embed_bias,
            pos_embed,
            blocks,
            norm,
            attn_pool,
            config: config.clone(),
        })
    }

    /// Forward pass
    pub fn forward(&self, images: &Tensor) -> Result<Tensor> {
        let batch_size = images.size()[0];
        debug!("SigLIP forward: input shape {:?}", images.size());

        // Convert input to match model dtype (weights are typically F16/BF16)
        let images = images.to_kind(self.patch_embed_weight.kind());

        // Patch embedding: [B, 3, H, W] -> [B, hidden_size, num_patches_h, num_patches_w]
        let stride = self.config.patch_size as i64;
        let mut x = images.conv2d(
            &self.patch_embed_weight,
            self.patch_embed_bias.as_ref(),
            [stride, stride],  // stride
            [0, 0],            // padding
            [1, 1],            // dilation
            1,                  // groups
        );

        // Reshape: [B, hidden_size, H', W'] -> [B, num_patches, hidden_size]
        let (_, hidden_size, h, w) = x.size4()?;
        let num_patches = h * w;
        x = x.permute([0, 2, 3, 1])  // [B, H', W', hidden_size]
            .reshape([batch_size, num_patches, hidden_size]);

        debug!("After patch embed: {:?}", x.size());

        // Add position embeddings
        // pos_embed is typically [1, num_patches, hidden_size]
        x += &self.pos_embed;

        // Transformer blocks
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;
            if i == 0 || i == self.blocks.len() - 1 {
                debug!("After block {}: {:?}", i, x.size());
            }
        }

        // Final layer norm
        x = self.norm.forward(&x)?;

        // Optional attention pooling
        if let Some(ref attn_pool) = self.attn_pool {
            x = attn_pool.forward(&x)?;
            debug!("After attention pooling: {:?}", x.size());
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_siglip_config_default() {
        let config = SigLIPConfig::default();
        assert_eq!(config.image_size, 384);
        assert_eq!(config.patch_size, 16);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 24);
    }
}
