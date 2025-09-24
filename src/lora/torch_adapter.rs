//! PyTorch-native LoRA adapter implementation with full autograd support

use anyhow::{Result, anyhow};
use tch::{nn, Device, Kind, Tensor};
use std::collections::HashMap;
use std::path::Path;
use async_trait::async_trait;
use super::{LoRAConfig, LoRAAdapter};

/// A single LoRA layer with automatic differentiation support
pub struct TorchLoRALayer {
    /// LoRA A matrix [in_features, rank] - trainable
    pub lora_a: Tensor,

    /// LoRA B matrix [rank, out_features] - trainable
    pub lora_b: Tensor,

    /// Scaling factor (alpha / rank)
    pub scaling: f64,

    /// Configuration
    pub config: LoRALayerConfig,
}

// SAFETY: TorchLoRALayer can be safely sent between threads because:
// 1. PyTorch tensors are internally thread-safe for their operations
// 2. VarStore operations are protected by internal synchronization
// 3. All mutations happen under mutex protection in the containing engine
unsafe impl Send for TorchLoRALayer {}
unsafe impl Sync for TorchLoRALayer {}

#[derive(Debug, Clone)]
pub struct LoRALayerConfig {
    pub in_features: i64,
    pub out_features: i64,
    pub rank: i64,
    pub alpha: f64,
    pub dropout: f64,
    pub device: Device,
}

impl TorchLoRALayer {
    /// Create a new LoRA layer with proper initialization
    pub fn new(vs: &nn::VarStore, config: LoRALayerConfig) -> Result<Self> {
        let root = vs.root();
        
        // Initialize LoRA A with Kaiming uniform (good for ReLU-like activations)
        // Scale by sqrt(5) as per Kaiming initialization
        let fan_in = config.in_features;
        let gain = (5.0_f64).sqrt();
        let std = gain / (fan_in as f64).sqrt();
        let bound = (3.0_f64).sqrt() * std;
        
        let lora_a = root.var(
            "lora_a",
            &[config.in_features, config.rank],
            nn::Init::Uniform {
                lo: -bound,
                up: bound,
            },
        );
        
        // Initialize LoRA B with zeros (standard LoRA practice)
        // This ensures the adapter starts as identity
        let lora_b = root.var(
            "lora_b",
            &[config.rank, config.out_features],
            nn::Init::Const(0.0),
        );
        
        let scaling = config.alpha / config.rank as f64;

        Ok(Self {
            lora_a,
            lora_b,
            scaling,
            config,
        })
    }
    
    /// Forward pass through the LoRA layer
    pub fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        // x shape: [batch_size, seq_len, in_features] or [batch_size * seq_len, in_features]
        
        // Apply LoRA: (x @ A) @ B * scaling
        let mut hidden = x.matmul(&self.lora_a);
        
        // Apply dropout during training
        if training && self.config.dropout > 0.0 {
            hidden = hidden.dropout(self.config.dropout, training);
        }
        
        let output = hidden.matmul(&self.lora_b);
        Ok(&output * self.scaling)
    }
    
    /// Get the effective weight matrix W = A @ B * scaling
    pub fn get_weight_matrix(&self) -> Result<Tensor> {
        let weight = self.lora_a.matmul(&self.lora_b);
        Ok(&weight * self.scaling)
    }
    
    /// Merge LoRA weights into base weights for inference
    pub fn merge_weight(&self, base_weight: &Tensor) -> Result<Tensor> {
        let lora_weight = &self.lora_a.matmul(&self.lora_b) * self.scaling;
        Ok(base_weight + lora_weight.transpose(0, 1))
    }
    
    /// Get number of trainable parameters
    pub fn num_parameters(&self) -> i64 {
        let a_params = self.config.in_features * self.config.rank;
        let b_params = self.config.rank * self.config.out_features;
        a_params + b_params
    }
    
}

/// Collection of LoRA layers for a model
pub struct LoRAModel {
    /// Map of module name to LoRA layer
    pub layers: HashMap<String, TorchLoRALayer>,

    /// Combined variable store for all layers
    pub vs: nn::VarStore,

    /// Configuration
    pub config: LoRAConfig,

    /// Device
    pub device: Device,
}

// SAFETY: LoRAModel can be safely sent between threads because:
// 1. It contains TorchLoRALayer which is Send + Sync
// 2. HashMap<String, TorchLoRALayer> is Send + Sync when TorchLoRALayer is
// 3. VarStore is thread-safe for concurrent access
// 4. Device and config are plain data types that are Send + Sync
unsafe impl Send for LoRAModel {}
unsafe impl Sync for LoRAModel {}

impl LoRAModel {
    /// Create LoRA adapters for specified modules
    pub fn new(
        config: LoRAConfig,
        module_configs: HashMap<String, (usize, usize)>, // module_name -> (in_features, out_features)
        device: Device,
    ) -> Result<Self> {
        let mut vs = nn::VarStore::new(device);
        let mut layers = HashMap::new();
        
        for (module_name, (in_features, out_features)) in module_configs {
            // Only create LoRA for target modules
            if !config.target_modules.contains(&module_name) {
                continue;
            }
            
            let layer_config = LoRALayerConfig {
                in_features: in_features as i64,
                out_features: out_features as i64,
                rank: config.rank as i64,
                alpha: config.alpha as f64,
                dropout: config.dropout as f64,
                device,
            };
            
            let layer = TorchLoRALayer::new(&vs, layer_config)?;
            layers.insert(module_name, layer);
        }
        
        let total_params: i64 = layers.values().map(|layer| layer.num_parameters()).sum();
        tracing::info!(
            "Created LoRA model with {} layers, {} total parameters",
            layers.len(),
            total_params
        );
        
        Ok(Self {
            layers,
            vs,
            config,
            device,
        })
    }
    
    /// Apply LoRA to a specific module
    pub fn forward(&self, module_name: &str, x: &Tensor, training: bool) -> Result<Option<Tensor>> {
        if let Some(layer) = self.layers.get(module_name) {
            Ok(Some(layer.forward(x, training)?))
        } else {
            Ok(None)
        }
    }
    
    /// Save all LoRA weights
    pub fn save(&self, path: &str) -> Result<()> {
        self.vs.save(path)?;
        Ok(())
    }
    
    /// Load all LoRA weights
    pub fn load(&mut self, path: &str) -> Result<()> {
        self.vs.load(path)?;
        Ok(())
    }
    
    /// Get total number of trainable parameters
    pub fn num_parameters(&self) -> i64 {
        self.layers.values().map(|layer| layer.num_parameters()).sum()
    }
    
    
    /// Get configuration for layer
    pub fn get_layer_config(&self, module_name: &str) -> Option<LoRALayerConfig> {
        self.layers.get(module_name).map(|layer| layer.get_config())
    }
}

impl TorchLoRALayer {
    /// Get layer configuration
    pub fn get_config(&self) -> LoRALayerConfig {
        self.config.clone()
    }
}

#[async_trait]
impl LoRAAdapter for LoRAModel {
    fn config(&self) -> &LoRAConfig {
        &self.config
    }
    
    async fn save(&self, path: &Path) -> Result<()> {
        self.vs.save(path)?;
        Ok(())
    }
    
    async fn load(&mut self, path: &Path) -> Result<()> {
        self.vs.load(path)?;
        Ok(())
    }

    fn forward(&self, module_name: &str, input: &Tensor) -> Result<Option<Tensor>> {
        // Look up the layer by module name
        if let Some(layer) = self.layers.get(module_name) {
            Ok(Some(layer.forward(input, false)?)) // Use inference mode by default
        } else {
            Ok(None)
        }
    }

    fn num_parameters(&self) -> i64 {
        self.layers.values()
            .map(|layer| layer.lora_a.size().iter().product::<i64>() +
                        layer.lora_b.size().iter().product::<i64>())
            .sum()
    }
}