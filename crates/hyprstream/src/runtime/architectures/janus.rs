//! Janus multimodal vision-language model implementation
//!
//! Janus combines vision and language understanding with optional image generation.
//! It uses a composition pattern to reuse existing language models (typically Llama)
//! while adding vision encoders and cross-modal alignment.
//!
//! NOTE: This module contains vision transformer components that are planned for
//! future multimodal support. They are currently unused but retained for the
//! upcoming Janus VLM integration.

#![allow(dead_code, clippy::type_complexity)]

use super::{ArchitectureConfig, ModelArchitecture, ModelOperations, VisionEncoderType};
use super::config::AttentionType;
use crate::runtime::kv_cache::LayerKVCache;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use tch::{Device, Kind as DType, Tensor};

/// Janus model configuration
pub struct JanusConfig {
    /// Configuration for the base language model
    pub language_config: Box<dyn ArchitectureConfig>,
    /// Vision encoder configuration
    pub vision_config: VisionEncoderConfig,
    /// Aligner configuration (vision to language)
    pub aligner_config: ProjectorConfig,
    /// Generation configuration (optional)
    pub generation_config: Option<GenerationConfig>,
    /// Device for computation
    pub device: Device,
    /// Data type for computation
    pub dtype: DType,
}

impl ArchitectureConfig for JanusConfig {
    fn hidden_size(&self) -> usize {
        self.language_config.hidden_size()
    }

    fn intermediate_size(&self) -> usize {
        self.language_config.intermediate_size()
    }

    fn num_attention_heads(&self) -> usize {
        self.language_config.num_attention_heads()
    }

    fn num_key_value_heads(&self) -> usize {
        self.language_config.num_key_value_heads()
    }

    fn head_dim(&self) -> usize {
        self.language_config.head_dim()
    }

    fn vocab_size(&self) -> usize {
        self.language_config.vocab_size()
    }

    fn max_position_embeddings(&self) -> usize {
        self.language_config.max_position_embeddings()
    }

    fn rope_theta(&self) -> Option<f32> {
        self.language_config.rope_theta()
    }

    fn rope_dim(&self) -> Option<usize> {
        self.language_config.rope_dim()
    }

    fn layer_norm_eps(&self) -> f32 {
        self.language_config.layer_norm_eps()
    }

    fn use_rms_norm(&self) -> bool {
        self.language_config.use_rms_norm()
    }

    fn attention_type(&self) -> AttentionType {
        self.language_config.attention_type()
    }
}

/// Vision encoder configuration
#[derive(Debug, Clone)]
pub struct VisionEncoderConfig {
    pub encoder_type: VisionEncoderType,
    pub hidden_size: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_layers: usize,
    pub num_patches: usize,  // (image_size / patch_size)^2
    pub num_attention_heads: Option<usize>,
    pub intermediate_size: Option<usize>,
}

/// MLP projector configuration (for aligners)
#[derive(Debug, Clone)]
pub struct ProjectorConfig {
    pub input_dim: usize,
    pub output_dim: usize,
    pub hidden_dim: Option<usize>,  // For 2-layer MLP
}

/// Generation pipeline configuration
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub vq_config: VQModelConfig,
    pub gen_aligner_config: ProjectorConfig,
    pub gen_head_config: VisionHeadConfig,
    pub image_token_size: usize,  // Vocabulary size for image tokens
}

/// VQ-VAE model configuration
#[derive(Debug, Clone)]
pub struct VQModelConfig {
    pub codebook_size: usize,
    pub latent_dim: usize,
}

/// Vision head configuration for image generation
#[derive(Debug, Clone)]
pub struct VisionHeadConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,  // image_token_size
}

/// Main Janus multimodal model
pub struct JanusModel {
    /// Core components (required)
    language_model: Box<dyn ModelOperations>,  // Reuse existing models!
    vision_encoder: VisionEncoder,
    vision_aligner: MlpProjector,

    /// Generation components (optional)
    generation: Option<GenerationPipeline>,

    /// Model metadata
    device: Device,
    dtype: DType,
    architecture: ModelArchitecture,
    config: JanusConfig,

    /// Optimization: Cache last vision encoding
    /// Uses Mutex instead of RefCell for thread safety (Send requirement)
    vision_cache: parking_lot::Mutex<Option<(Vec<u8>, Tensor)>>,  // (input_hash, features)
}

/// Vision encoder implementations
#[allow(clippy::large_enum_variant)] // Created once, used frequently - boxing adds unwanted indirection
pub enum VisionEncoder {
    SigLIP(SigLIPVisionTower),
    CLIP(CLIPVisionTower),
}

impl VisionEncoder {
    fn forward(&self, images: &Tensor) -> Result<Tensor> {
        match self {
            Self::SigLIP(model) => model.forward(images),
            Self::CLIP(model) => model.forward(images),
        }
    }

    fn from_weights(
        config: &VisionEncoderConfig,
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        match &config.encoder_type {
            VisionEncoderType::SigLIP { .. } => {
                Ok(Self::SigLIP(SigLIPVisionTower::from_weights(
                    config, weights, prefix, device, dtype,
                )?))
            }
            VisionEncoderType::CLIP { .. } => {
                Ok(Self::CLIP(CLIPVisionTower::from_weights(
                    config, weights, prefix, device, dtype,
                )?))
            }
            _ => Err(anyhow!("Unsupported vision encoder type")),
        }
    }
}

/// SigLIP vision encoder (wrapper around siglip module)
pub struct SigLIPVisionTower {
    model: super::siglip::SigLIPVisionTransformer,
    config: VisionEncoderConfig,
}

impl SigLIPVisionTower {
    fn from_weights(
        config: &VisionEncoderConfig,
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        // Convert VisionEncoderConfig to SigLIPConfig
        let siglip_config = super::siglip::SigLIPConfig {
            image_size: config.image_size,
            patch_size: config.patch_size,
            hidden_size: config.hidden_size,
            num_hidden_layers: config.num_layers,
            num_attention_heads: config.num_attention_heads.unwrap_or(16),
            intermediate_size: config.intermediate_size.unwrap_or(config.hidden_size * 4),
            use_attention_pool: false,  // Janus does NOT use attention pooling - needs all 576 patches
        };

        let model = super::siglip::SigLIPVisionTransformer::from_weights(
            &siglip_config,
            weights,
            prefix,
            device,
            dtype,
        )?;

        Ok(Self {
            model,
            config: config.clone(),
        })
    }

    fn forward(&self, images: &Tensor) -> Result<Tensor> {
        self.model.forward(images)
    }
}

/// CLIP vision encoder (placeholder - not yet implemented with new architecture)
pub struct CLIPVisionTower {
    config: VisionEncoderConfig,
}

impl CLIPVisionTower {
    fn from_weights(
        _config: &VisionEncoderConfig,
        _weights: &HashMap<String, Tensor>,
        _prefix: &str,
        _device: Device,
        _dtype: DType,
    ) -> Result<Self> {
        // TODO: Implement CLIP vision encoder similar to SigLIP
        // CLIP and SigLIP have very similar architectures
        Err(anyhow!("CLIP vision encoder not yet implemented - use SigLIP instead"))
    }

    fn forward(&self, _images: &Tensor) -> Result<Tensor> {
        Err(anyhow!("CLIP vision encoder not yet implemented - use SigLIP instead"))
    }
}

/// Vision transformer layer
struct VisionTransformerLayer {
    self_attn: MultiHeadAttention,
    mlp: Mlp,
    ln_1: LayerNorm,
    ln_2: LayerNorm,
}

impl VisionTransformerLayer {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        hidden_size: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let self_attn = MultiHeadAttention::from_weights(
            weights,
            &format!("{prefix}.self_attn"),
            hidden_size,
            device,
            dtype,
        )?;

        let mlp = Mlp::from_weights(
            weights,
            &format!("{prefix}.mlp"),
            hidden_size,
            hidden_size * 4,  // Standard 4x expansion
            device,
            dtype,
        )?;

        let ln_1 = LayerNorm::from_weights(
            weights,
            &format!("{prefix}.ln_1"),
            hidden_size,
            device,
            dtype,
        )?;

        let ln_2 = LayerNorm::from_weights(
            weights,
            &format!("{prefix}.ln_2"),
            hidden_size,
            device,
            dtype,
        )?;

        Ok(Self {
            self_attn,
            mlp,
            ln_1,
            ln_2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-norm architecture
        let mut output = x.shallow_clone();

        // Self-attention with residual
        let attn_out = self.self_attn.forward(&self.ln_1.forward(&output)?)?;
        output = &output + &attn_out;

        // MLP with residual
        let mlp_out = self.mlp.forward(&self.ln_2.forward(&output)?)?;
        output = &output + &mlp_out;

        Ok(output)
    }
}

/// Multi-head attention for vision transformer
struct MultiHeadAttention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    out_proj: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        hidden_size: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut q_proj = weights
            .get(&format!("{prefix}.q_proj.weight"))
            .ok_or_else(|| anyhow!("Missing q_proj weight"))?
            .shallow_clone();
        q_proj = q_proj.to_device(device);
        q_proj = q_proj.to_kind(dtype);

        let mut k_proj = weights
            .get(&format!("{prefix}.k_proj.weight"))
            .ok_or_else(|| anyhow!("Missing k_proj weight"))?
            .shallow_clone();
        k_proj = k_proj.to_device(device);
        k_proj = k_proj.to_kind(dtype);

        let mut v_proj = weights
            .get(&format!("{prefix}.v_proj.weight"))
            .ok_or_else(|| anyhow!("Missing v_proj weight"))?
            .shallow_clone();
        v_proj = v_proj.to_device(device);
        v_proj = v_proj.to_kind(dtype);

        let mut out_proj = weights
            .get(&format!("{prefix}.out_proj.weight"))
            .ok_or_else(|| anyhow!("Missing out_proj weight"))?
            .shallow_clone();
        out_proj = out_proj.to_device(device);
        out_proj = out_proj.to_kind(dtype);

        // Infer num_heads from weight shapes
        let num_heads = 16;  // Standard for many vision models, could be made configurable
        let head_dim = hidden_size / num_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.size3()?;

        // Project to Q, K, V
        let q = x.matmul(&self.q_proj.transpose(0, 1));
        let k = x.matmul(&self.k_proj.transpose(0, 1));
        let v = x.matmul(&self.v_proj.transpose(0, 1));

        // Reshape for multi-head attention
        let q = q.reshape([batch_size, seq_len, self.num_heads as i64, self.head_dim as i64])
            .transpose(1, 2);  // [B, H, N, D]
        let k = k.reshape([batch_size, seq_len, self.num_heads as i64, self.head_dim as i64])
            .transpose(1, 2);
        let v = v.reshape([batch_size, seq_len, self.num_heads as i64, self.head_dim as i64])
            .transpose(1, 2);

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(-2, -1))
            .divide_scalar(scale);
        let attn_weights = scores.softmax(-1, scores.kind());
        let attn_output = attn_weights.matmul(&v);

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)
            .reshape([batch_size, seq_len, -1]);

        // Output projection
        let output = attn_output.matmul(&self.out_proj.transpose(0, 1));

        Ok(output)
    }
}

/// MLP for vision transformer
struct Mlp {
    fc1: Tensor,
    fc2: Tensor,
}

impl Mlp {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        _hidden_size: usize,
        _intermediate_size: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut fc1 = weights
            .get(&format!("{prefix}.fc1.weight"))
            .ok_or_else(|| anyhow!("Missing fc1 weight"))?
            .shallow_clone();
        fc1 = fc1.to_device(device);
        fc1 = fc1.to_kind(dtype);

        let mut fc2 = weights
            .get(&format!("{prefix}.fc2.weight"))
            .ok_or_else(|| anyhow!("Missing fc2 weight"))?
            .shallow_clone();
        fc2 = fc2.to_device(device);
        fc2 = fc2.to_kind(dtype);

        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut output = x.matmul(&self.fc1.transpose(0, 1));
        output = output.gelu("none");  // GELU activation
        output = output.matmul(&self.fc2.transpose(0, 1));
        Ok(output)
    }
}

/// Layer normalization
struct LayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f64,
}

impl LayerNorm {
    fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        _hidden_size: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut weight = weights
            .get(&format!("{prefix}.weight"))
            .ok_or_else(|| anyhow!("Missing layer norm weight"))?
            .shallow_clone();
        weight = weight.to_device(device);
        weight = weight.to_kind(dtype);

        let bias = weights
            .get(&format!("{prefix}.bias"))
            .map(|b| {
                let mut b = b.shallow_clone();
                b = b.to_device(device);
                b.to_kind(dtype)
            });

        Ok(Self {
            weight,
            bias,
            eps: 1e-5,  // Standard epsilon
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Simplified layer norm implementation
        // TODO: Use proper layer_norm once we figure out the signature
        let mean = x.mean_dim(&[-1i64][..], true, x.kind());
        let var = x.var_dim(&[-1i64][..], false, true);
        let normalized = (x - &mean) / (&var + self.eps).sqrt();

        let output = &normalized * &self.weight;
        let output = if let Some(ref bias) = self.bias {
            output + bias
        } else {
            output
        };
        Ok(output)
    }
}

/// MLP projector for vision-language alignment
pub struct MlpProjector {
    layers: Vec<Tensor>,
    config: ProjectorConfig,
}

impl MlpProjector {
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
        config: &ProjectorConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        use tracing::debug;

        // Debug: log available keys
        debug!("Loading MLP projector with prefix: {}", prefix);
        let aligner_keys: Vec<_> = weights.keys()
            .filter(|k| k.starts_with(prefix))
            .collect();
        debug!("Available aligner keys: {:?}", aligner_keys);

        let mut layers = Vec::new();

        if let Some(_hidden_dim) = config.hidden_dim {
            // 2-layer MLP with GELU
            // Structure: Linear(0) -> GELU(1) -> Linear(2)
            // Only Linear layers have weights

            // First linear layer
            let mut layer0_weight = weights
                .get(&format!("{prefix}.layers.0.weight"))
                .ok_or_else(|| anyhow!("Missing projector layer 0 weight"))?
                .shallow_clone();
            layer0_weight = layer0_weight.to_device(device);
            layer0_weight = layer0_weight.to_kind(dtype);

            let mut layer0_bias = weights
                .get(&format!("{prefix}.layers.0.bias"))
                .ok_or_else(|| anyhow!("Missing projector layer 0 bias"))?
                .shallow_clone();
            layer0_bias = layer0_bias.to_device(device);
            layer0_bias = layer0_bias.to_kind(dtype);

            // Second linear layer (index 2 in Sequential, after GELU at index 1)
            let mut layer2_weight = weights
                .get(&format!("{prefix}.layers.2.weight"))
                .ok_or_else(|| anyhow!("Missing projector layer 2 weight"))?
                .shallow_clone();
            layer2_weight = layer2_weight.to_device(device);
            layer2_weight = layer2_weight.to_kind(dtype);

            let mut layer2_bias = weights
                .get(&format!("{prefix}.layers.2.bias"))
                .ok_or_else(|| anyhow!("Missing projector layer 2 bias"))?
                .shallow_clone();
            layer2_bias = layer2_bias.to_device(device);
            layer2_bias = layer2_bias.to_kind(dtype);

            // Store weights and biases as tuples
            layers.push(layer0_weight);
            layers.push(layer0_bias);
            layers.push(layer2_weight);
            layers.push(layer2_bias);
        } else {
            // Single linear layer
            let mut weight = weights
                .get(&format!("{prefix}.layers.0.weight"))
                .ok_or_else(|| anyhow!("Missing projector weight"))?
                .shallow_clone();
            weight = weight.to_device(device);
            weight = weight.to_kind(dtype);

            let mut bias = weights
                .get(&format!("{prefix}.layers.0.bias"))
                .ok_or_else(|| anyhow!("Missing projector bias"))?
                .shallow_clone();
            bias = bias.to_device(device);
            bias = bias.to_kind(dtype);

            layers.push(weight);
            layers.push(bias);
        }

        Ok(Self {
            layers,
            config: config.clone(),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.config.hidden_dim.is_some() {
            // 2-layer MLP: Linear -> GELU -> Linear
            // layers = [weight0, bias0, weight2, bias2]

            // First linear layer
            let mut output = x.matmul(&self.layers[0].transpose(0, 1)) + &self.layers[1];

            // GELU activation
            output = output.gelu("none");

            // Second linear layer
            output = output.matmul(&self.layers[2].transpose(0, 1)) + &self.layers[3];

            Ok(output)
        } else {
            // Single linear layer
            // layers = [weight, bias]
            let output = x.matmul(&self.layers[0].transpose(0, 1)) + &self.layers[1];
            Ok(output)
        }
    }
}

/// Generation pipeline for image generation
pub struct GenerationPipeline {
    vq_model: VQModel,
    gen_aligner: MlpProjector,
    gen_head: VisionHead,
    gen_embed: Tensor,  // [image_token_size, hidden_size]
}

/// VQ-VAE model for image generation
pub struct VQModel {
    // Simplified VQ model - would need full encoder/decoder in practice
    codebook: Tensor,
    config: VQModelConfig,
}

/// Vision head for generating image tokens
pub struct VisionHead {
    output_mlp_projector: Tensor,
    vision_head: Tensor,
}

impl JanusModel {
    /// Create Janus model from weights
    pub fn from_weights(
        weights: HashMap<String, Tensor>,
        config: JanusConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        // Extract weights by prefix
        let (language_weights, vision_weights, aligner_weights, _gen_weights) =
            Self::extract_weights_by_prefix(&weights)?;

        // Load language model
        let language_model = Self::load_language_model(
            language_weights,
            &config,
            device,
            dtype,
        )?;

        // Load vision encoder
        // Janus stores vision weights under vision_model.vision_tower prefix
        let vision_encoder = VisionEncoder::from_weights(
            &config.vision_config,
            &vision_weights,
            "vision_model.vision_tower",
            device,
            dtype,
        )?;

        // Load vision aligner
        let vision_aligner = MlpProjector::from_weights(
            &aligner_weights,
            "aligner",
            &config.aligner_config,
            device,
            dtype,
        )?;

        // Load generation components if configured
        let generation = if let Some(_gen_config) = &config.generation_config {
            // TODO: Implement generation pipeline loading
            None
        } else {
            None
        };

        // Create architecture enum
        let architecture = ModelArchitecture::Janus {
            base_architecture: Box::new(ModelArchitecture::Llama { version: 3 }), // TODO: Detect from config
            vision_encoder: config.vision_config.encoder_type.clone(),
            has_generation: generation.is_some(),
        };

        Ok(Self {
            language_model,
            vision_encoder,
            vision_aligner,
            generation,
            device,
            dtype,
            architecture,
            config,
            vision_cache: parking_lot::Mutex::new(None),
        })
    }

    /// Extract weights by component prefix
    fn extract_weights_by_prefix(
        weights: &HashMap<String, Tensor>,
    ) -> Result<(
        HashMap<String, Tensor>,  // language_model.*
        HashMap<String, Tensor>,  // vision_model.*
        HashMap<String, Tensor>,  // aligner.*
        HashMap<String, Tensor>,  // gen_* components - currently unused
    )> {
        let mut language_weights = HashMap::new();
        let mut vision_weights = HashMap::new();
        let mut aligner_weights = HashMap::new();
        let mut _gen_weights = HashMap::new();

        for (key, tensor) in weights {
            if let Some(suffix) = key.strip_prefix("language_model.") {
                // Remove prefix for language model
                language_weights.insert(suffix.to_owned(), tensor.shallow_clone());
            } else if key.starts_with("vision_model.") {
                vision_weights.insert(key.clone(), tensor.shallow_clone());
            } else if key.starts_with("aligner.") {
                aligner_weights.insert(key.clone(), tensor.shallow_clone());
            } else if key.starts_with("gen_") {
                _gen_weights.insert(key.clone(), tensor.shallow_clone());
            }
        }

        Ok((language_weights, vision_weights, aligner_weights, _gen_weights))
    }

    /// Load the language model component
    fn load_language_model(
        mut weights: HashMap<String, Tensor>,
        _config: &JanusConfig,
        device: Device,
        dtype: DType,
    ) -> Result<Box<dyn ModelOperations>> {
        // For now, assume Llama - could be made more flexible
        use super::llama::LlamaModel;

        // Create Llama config from Janus language config
        // This would need proper conversion logic
        let llama_model = LlamaModel::from_weights(&mut weights, &device, dtype)?;

        Ok(Box::new(llama_model))
    }

    /// Prepare inputs by combining vision and text embeddings
    /// Prepare inputs by combining vision and text embeddings
    pub fn prepare_inputs_embeds(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        images_seq_mask: Option<&Tensor>,
        images_emb_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        use tracing::debug;

        // Get text embeddings from language model
        let mut inputs_embeds = self.language_model.embed_tokens(input_ids)?;

        // If we have images, process them and insert into sequence
        if let (Some(images), Some(seq_mask), Some(emb_mask)) =
            (pixel_values, images_seq_mask, images_emb_mask) {

            let images_shape = images.size();

            // Handle both [batch, 3, h, w] and [batch, n_images, 3, h, w] formats
            let (batch_size, n_images, channels, height, width) = if images_shape.len() == 4 {
                // Single image: [batch, 3, h, w] → treat as [batch, 1, 3, h, w]
                (images_shape[0], 1, images_shape[1], images_shape[2], images_shape[3])
            } else if images_shape.len() == 5 {
                // Multiple images: [batch, n_images, 3, h, w]
                (images_shape[0], images_shape[1], images_shape[2], images_shape[3], images_shape[4])
            } else {
                return Err(anyhow!(
                    "Unexpected pixel_values shape: {:?}. Expected [batch, 3, h, w] or [batch, n_images, 3, h, w]",
                    images_shape
                ));
            };

            debug!(
                "Processing {} images: batch={}, n_images={}, shape=[{}, {}, {}]",
                n_images * batch_size, batch_size, n_images, height, width, channels
            );

            // Reshape from [batch, n_images, 3, h, w] → [batch*n_images, 3, h, w]
            // For 4D input, this is just a simple reshape
            let images_flat = if images_shape.len() == 4 {
                // [batch, 3, h, w] → [batch, 3, h, w] (no change needed, batch=batch*1)
                images.shallow_clone()
            } else {
                // [batch, n_images, 3, h, w] → [batch*n_images, 3, h, w]
                images.reshape([batch_size * n_images, channels, height, width])
            };

            // Encode images through vision encoder
            let image_features = self.vision_encoder.forward(&images_flat)?;

            // image_features shape: [batch*n_images, num_patches, encoder_hidden_size]
            let feat_shape = image_features.size();
            let num_patches = feat_shape[1];
            let encoder_hidden = feat_shape[2];

            debug!(
                "Vision encoder output: num_patches={}, hidden_size={}",
                num_patches, encoder_hidden
            );

            // Project to language space
            let image_embeds = self.vision_aligner.forward(&image_features)?;

            // image_embeds shape: [batch*n_images, num_patches, language_hidden_size]
            let lang_shape = image_embeds.size();
            let language_hidden = lang_shape[2];

            // Reshape from [batch*n_images, num_patches, hidden]
            // to [batch, n_images*num_patches, hidden]
            let image_embeds_reshaped = image_embeds
                .reshape([batch_size, n_images * num_patches, language_hidden]);

            debug!(
                "Image embeddings reshaped: {:?}",
                image_embeds_reshaped.size()
            );

            // Verify dimension compatibility
            let input_shape = inputs_embeds.size();
            let input_hidden = input_shape[2];

            if language_hidden != input_hidden {
                return Err(anyhow!(
                    "Language model hidden size {} != vision aligner output {}",
                    input_hidden,
                    language_hidden
                ));
            }

            // Reshape emb_mask to [batch, n_images*num_patches]
            // Input can be either [batch, num_patches] or [batch, n_images, num_patches]
            // IMPORTANT: We use emb_mask's dimensions, not vision encoder output,
            // because emb_mask contains the ground truth number of image tokens (576)
            let emb_mask_shape = emb_mask.size();
            let emb_mask_patches = if emb_mask_shape.len() == 3 {
                emb_mask_shape[2]  // [batch, n_images, num_patches] → extract num_patches
            } else {
                emb_mask_shape[1]  // [batch, num_patches] → extract num_patches
            };

            let emb_mask_flat = if emb_mask_shape.len() == 2 {
                // Already [batch, num_patches] for single image
                emb_mask.shallow_clone()
            } else {
                // [batch, n_images, num_patches] → [batch, n_images*num_patches]
                emb_mask.reshape([batch_size, n_images * emb_mask_patches])
            };

            // Replace image placeholders with actual image embeddings
            inputs_embeds = self.merge_embeddings(
                &inputs_embeds,
                &image_embeds_reshaped,
                seq_mask,
                &emb_mask_flat,
            )?;

            debug!("Merged embeddings: {:?}", inputs_embeds.size());
        }

        Ok(inputs_embeds)
    }

    /// Merge text and image embeddings based on masks
    ///
    /// This function takes text embeddings and vision embeddings and combines them
    /// by replacing positions marked in seq_mask with corresponding vision embeddings.
    ///
    /// # Arguments
    /// * `text_embeds` - Text token embeddings [batch, text_seq_len, hidden_dim]
    /// * `image_embeds` - Vision embeddings [batch, num_images * num_patches, hidden_dim]
    /// * `seq_mask` - Boolean mask [batch, text_seq_len] indicating positions to replace
    /// * `emb_mask` - Boolean mask [batch, num_images * num_patches] indicating which vision embeddings to use
    ///
    /// # Returns
    /// Combined embedding sequence where image positions are replaced with vision embeddings
    /// Merge text and image embeddings using in-place updates
    ///
    /// # Implementation Note
    /// This function uses shallow_clone() which creates a view sharing the same underlying data.
    /// Mutations through copy_() will modify this view, which is the intended behavior.
    /// The caller should ensure text_embeds is not aliased elsewhere if this is undesired.
    ///
    /// This approach avoids tch-rs index operation issues by processing batch-by-batch
    /// and using copy_ operations carefully.
    fn merge_embeddings(
        &self,
        text_embeds: &Tensor,
        image_embeds: &Tensor,
        seq_mask: &Tensor,
        emb_mask: &Tensor,
    ) -> Result<Tensor> {
        use tracing::debug;

        let text_shape = text_embeds.size();
        let image_shape = image_embeds.size();

        let batch_size = text_shape[0];
        let _seq_len = text_shape[1];
        let hidden_dim = text_shape[2];

        debug!(
            "merge_embeddings: text={:?}, image={:?}",
            text_shape, image_shape
        );

        // Verify dimensions match
        if image_shape[2] != hidden_dim {
            return Err(anyhow!(
                "Hidden dimension mismatch: text={}, image={}",
                hidden_dim,
                image_shape[2]
            ));
        }

        // Create a view we can modify in-place
        // NOTE: shallow_clone() shares underlying data, mutations will affect the view
        let result = text_embeds.shallow_clone();

        // Process batch by batch for safety
        for b in 0..batch_size {
            let batch_result = result.get(b);
            let batch_seq_mask = seq_mask.get(b);
            let batch_image_embeds = image_embeds.get(b);
            let batch_emb_mask = emb_mask.get(b);

            // Find all positions where seq_mask is True
            let seq_positions = batch_seq_mask.nonzero();  // [N, 1]
            let num_positions = seq_positions.size()[0];

            if num_positions == 0 {
                continue;
            }

            // Find all valid image embeddings
            let emb_mask_flat = batch_emb_mask.reshape([-1]);
            let image_flat = batch_image_embeds.reshape([-1, hidden_dim]);
            let valid_image_embeds = image_flat.masked_select(&emb_mask_flat.unsqueeze(-1))
                .reshape([num_positions, hidden_dim]);

            debug!(
                "Batch {}: Merging {} image embeddings at {} positions",
                b, valid_image_embeds.size()[0], num_positions
            );

            // Do the replacement: for each True position, use corresponding image embedding
            for i in 0..num_positions {
                let pos = seq_positions.int64_value(&[i, 0]);
                let mut row = batch_result.get(pos);
                let img_row = valid_image_embeds.get(i);

                // Update in-place
                row.copy_(&img_row);
            }
        }

        Ok(result)
    }
}

impl ModelOperations for JanusModel {
    fn architecture(&self) -> ModelArchitecture {
        self.architecture.clone()
    }

    fn config(&self) -> &dyn ArchitectureConfig {
        &self.config
    }

    fn forward(&self, input: &Tensor, past_kv: Option<&Tensor>) -> Result<Tensor> {
        // For text-only forward, delegate to language model
        self.language_model.forward(input, past_kv)
    }

    fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.language_model.embed_tokens(input_ids)
    }

    fn forward_from_embeddings(&self, embeddings: &Tensor, start_pos: usize) -> Result<Tensor> {
        // Delegate to language model's forward_from_embeddings
        self.language_model.forward_from_embeddings(embeddings, start_pos)
    }

    fn forward_with_cache(&self, input: &Tensor, start_pos: usize) -> Result<Tensor> {
        // CRITICAL: Delegate to language model's forward_with_cache
        // The default trait implementation calls forward(input, None) which loses start_pos!
        self.language_model.forward_with_cache(input, start_pos)
    }

    fn prepare_multimodal_inputs(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        images_seq_mask: Option<&Tensor>,
        images_emb_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Call the existing prepare_inputs_embeds method
        self.prepare_inputs_embeds(input_ids, pixel_values, images_seq_mask, images_emb_mask)
    }

    fn decode_layer(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_kv: Option<&LayerKVCache>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        self.language_model.decode_layer(
            layer_idx,
            hidden_states,
            attention_mask,
            position_ids,
            past_kv,
        )
    }

    fn apply_final_norm(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.language_model.apply_final_norm(hidden_states)
    }

    fn lm_head(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.language_model.lm_head(hidden_states)
    }

    fn num_layers(&self) -> usize {
        self.language_model.num_layers()
    }

    fn reshape_for_attention(&self, tensor: &Tensor, is_key_value: bool) -> Result<Tensor> {
        self.language_model.reshape_for_attention(tensor, is_key_value)
    }

    fn apply_rope(&self, tensor: &Tensor, position_ids: &Tensor) -> Result<Tensor> {
        self.language_model.apply_rope(tensor, position_ids)
    }

    fn normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        self.language_model.normalize(tensor)
    }

    fn get_attention_mask(&self, seq_len: usize, past_kv_len: usize) -> Result<Tensor> {
        self.language_model.get_attention_mask(seq_len, past_kv_len)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// SAFETY: JanusModel is Send because:
// - All Tensor fields are Send (guaranteed by tch-rs)
// - language_model (Box<dyn ModelOperations>) is Send (required by trait bound)
// - vision_cache uses Mutex (which is Send) instead of RefCell
// - No thread-local state or non-Send fields
unsafe impl Send for JanusModel {}

// SAFETY: VisionEncoder is Send because both enum variants are Send
unsafe impl Send for VisionEncoder {}

// SAFETY: SigLIPVisionTower is Send because all fields are Send:
// - Tensors are Send (tch-rs guarantee)
// - SigLIPConfig contains only primitives
unsafe impl Send for SigLIPVisionTower {}

// SAFETY: CLIPVisionTower is Send because VisionEncoderConfig is Send
unsafe impl Send for CLIPVisionTower {}

// SAFETY: Vision transformer components are Send because all Tensor fields are Send
unsafe impl Send for VisionTransformerLayer {}
unsafe impl Send for MultiHeadAttention {}
unsafe impl Send for Mlp {}
unsafe impl Send for LayerNorm {}
unsafe impl Send for MlpProjector {}
unsafe impl Send for GenerationPipeline {}
unsafe impl Send for VQModel {}

// ============================================================================
// Janus-specific placeholder token utilities
// ============================================================================

/// Configuration for Janus image placeholder replacement
///
/// Janus uses special tokens to mark where images should be injected:
/// - `<image_placeholder>` - User writes this in prompts
/// - `<begin_of_image>` - Marks start of image token sequence
/// - `<end_of_image>` - Marks end of image token sequence
/// - Each placeholder is replaced with: <boi> + 576 image tokens + <eoi>
#[derive(Debug, Clone)]
pub struct JanusPlaceholderConfig {
    /// Token ID for `<image_placeholder>`
    pub image_placeholder_id: u32,
    /// Token ID for `<begin_of_image>`
    pub image_start_id: u32,
    /// Token ID for `<end_of_image>`
    pub image_end_id: u32,
    /// Token ID used for actual image embeddings (typically same as placeholder)
    pub image_token_id: u32,
    /// Number of image tokens per image (typically 576 for 24x24 patches)
    pub num_image_tokens: usize,
}

impl JanusPlaceholderConfig {
    /// Create configuration from a tokenizer
    pub fn from_tokenizer(tokenizer: &tokenizers::Tokenizer) -> Result<Self> {
        let vocab = tokenizer.get_vocab(true);

        let image_placeholder_id = vocab.get("<image_placeholder>")
            .copied()
            .ok_or_else(|| anyhow!("Tokenizer missing <image_placeholder> token"))?;

        let image_start_id = vocab.get("<begin_of_image>")
            .copied()
            .ok_or_else(|| anyhow!("Tokenizer missing <begin_of_image> token"))?;

        let image_end_id = vocab.get("<end_of_image>")
            .copied()
            .ok_or_else(|| anyhow!("Tokenizer missing <end_of_image> token"))?;

        Ok(Self {
            image_placeholder_id,
            image_start_id,
            image_end_id,
            image_token_id: image_placeholder_id, // Use same ID for actual image tokens
            num_image_tokens: 576, // Standard for 384x384 images with 16x16 patches = 24x24 = 576
        })
    }
}

/// Result of Janus placeholder token replacement
#[derive(Debug)]
pub struct JanusPlaceholderReplacement {
    /// Modified input IDs with placeholders replaced
    /// Shape: [batch, expanded_seq_len]
    pub input_ids: Tensor,

    /// Mask indicating where image tokens are in the sequence
    /// Shape: [batch, expanded_seq_len]
    pub images_seq_mask: Tensor,

    /// Mask indicating which vision embeddings to use
    /// Shape: [batch, n_images, num_image_tokens]
    pub images_emb_mask: Tensor,

    /// Number of images detected
    pub num_images: usize,
}

/// Replace Janus image placeholder tokens with proper token sequence
///
/// This follows the official Janus design:
/// 1. Find all `<image_placeholder>` tokens in input_ids
/// 2. Replace each with: <begin_of_image> + 576 x <image_token> + <end_of_image>
/// 3. Create masks for embedding replacement
///
/// # Arguments
/// * `input_ids` - Original token IDs [seq_len]
/// * `config` - Placeholder configuration with token IDs
/// * `device` - Target device for tensors
///
/// # Returns
/// JanusPlaceholderReplacement with expanded tokens and masks
pub fn replace_janus_placeholders(
    input_ids: &[i64],
    config: &JanusPlaceholderConfig,
    device: Device,
) -> Result<JanusPlaceholderReplacement> {
    use tracing::{debug, warn};

    // Find indices of all placeholder tokens
    let placeholder_indices: Vec<usize> = input_ids
        .iter()
        .enumerate()
        .filter(|(_, &id)| id == config.image_placeholder_id as i64)
        .map(|(idx, _)| idx)
        .collect();

    let num_images = placeholder_indices.len();

    if num_images == 0 {
        warn!("No image placeholder tokens found in input");
        // Return original input with empty masks
        let input_ids_tensor = Tensor::from_slice(input_ids)
            .to_kind(DType::Int64)
            .to_device(device)
            .unsqueeze(0); // [1, seq_len]

        let seq_len = input_ids.len();
        let images_seq_mask = Tensor::zeros([1, seq_len as i64], (DType::Bool, device));
        let images_emb_mask = Tensor::zeros([1, 1, config.num_image_tokens as i64], (DType::Bool, device));

        return Ok(JanusPlaceholderReplacement {
            input_ids: input_ids_tensor,
            images_seq_mask,
            images_emb_mask,
            num_images: 0,
        });
    }

    debug!("Found {} image placeholder(s) at positions: {:?}", num_images, placeholder_indices);

    // Build new token sequence with replacements
    let mut new_tokens = Vec::new();
    let mut last_pos = 0;

    for &placeholder_idx in &placeholder_indices {
        // Add tokens before placeholder
        new_tokens.extend_from_slice(&input_ids[last_pos..placeholder_idx]);

        // Replace placeholder with: <boi> + 576 x <image> + <eoi>
        new_tokens.push(config.image_start_id as i64);
        new_tokens.extend(std::iter::repeat_n(config.image_token_id as i64, config.num_image_tokens));
        new_tokens.push(config.image_end_id as i64);

        last_pos = placeholder_idx + 1;
    }

    // Add remaining tokens after last placeholder
    new_tokens.extend_from_slice(&input_ids[last_pos..]);

    let expanded_seq_len = new_tokens.len();
    debug!(
        "Expanded sequence length: {} -> {} (added {} tokens for {} images)",
        input_ids.len(),
        expanded_seq_len,
        expanded_seq_len - input_ids.len(),
        num_images
    );

    // Create input_ids tensor [1, expanded_seq_len]
    let input_ids_tensor = Tensor::from_slice(&new_tokens)
        .to_kind(DType::Int64)
        .to_device(device)
        .unsqueeze(0);

    // Create images_seq_mask: mark positions where image tokens are
    // This includes the 576 image tokens, NOT the <boi> and <eoi> tokens
    let mut seq_mask = vec![false; expanded_seq_len];
    let mut current_pos = 0;

    for (img_idx, &placeholder_idx) in placeholder_indices.iter().enumerate() {
        // Calculate how many tokens were added before this placeholder
        let tokens_before = if img_idx == 0 {
            placeholder_idx
        } else {
            placeholder_idx - placeholder_indices[img_idx - 1] - 1
        };

        current_pos += tokens_before;

        // Skip <boi> token
        current_pos += 1;

        // Mark the 576 image tokens with bounds checking
        for _ in 0..config.num_image_tokens {
            if current_pos >= seq_mask.len() {
                return Err(anyhow!(
                    "Placeholder replacement exceeded sequence length: pos={}, len={}, image={}/{}",
                    current_pos, seq_mask.len(), img_idx + 1, num_images
                ));
            }
            seq_mask[current_pos] = true;
            current_pos += 1;
        }

        // Skip <eoi> token
        current_pos += 1;
    }

    let images_seq_mask = Tensor::from_slice(&seq_mask.iter().map(|&b| if b { 1u8 } else { 0u8 }).collect::<Vec<_>>())
        .to_kind(DType::Bool)
        .to_device(device)
        .unsqueeze(0); // [1, expanded_seq_len]

    // Create images_emb_mask: all vision embeddings are valid
    // Shape: [1, num_images, num_image_tokens] - all True
    let images_emb_mask = Tensor::ones(
        [1, num_images as i64, config.num_image_tokens as i64],
        (DType::Bool, device)
    );

    Ok(JanusPlaceholderReplacement {
        input_ids: input_ids_tensor,
        images_seq_mask,
        images_emb_mask,
        num_images,
    })
}

// SAFETY: VisionHead is Send because all Tensor fields are Send
unsafe impl Send for VisionHead {}