//! PyTorch-native LoRA adapter implementation with full autograd support

use super::{LoRAAdapter, LoRAConfig};
use anyhow::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use tch::{nn, Device, Tensor};

/// Extract LoRA components from hierarchical tensor names
/// e.g., "model.layers.0.q_proj.lora_a.weight" -> Some(("q_proj", "lora_a"))
pub fn extract_lora_components(tensor_name: &str) -> Option<(String, String)> {
    // Handle various patterns:
    // "model.layers.X.module_name.lora_matrix.weight"
    // "module_name.lora_matrix"
    // "layers.X.module_name.lora_matrix.weight"

    let parts: Vec<&str> = tensor_name.split('.').collect();

    // Look for lora_a or lora_b in the parts
    for (i, &part) in parts.iter().enumerate() {
        if part.starts_with("lora_") && (part == "lora_a" || part == "lora_b") {
            // The module name should be the part before lora_a/lora_b
            if i > 0 {
                let module = parts[i - 1];
                // Filter to only the modules we care about
                if matches!(
                    module,
                    "q_proj"
                        | "k_proj"
                        | "v_proj"
                        | "o_proj"
                        | "gate_proj"
                        | "up_proj"
                        | "down_proj"
                ) {
                    return Some((module.to_string(), part.to_string()));
                }
            }
        }
    }

    None
}

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
    pub module_name: String,
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
            &format!("{}_lora_a", config.module_name),
            &[config.in_features, config.rank],
            nn::Init::Uniform {
                lo: -bound,
                up: bound,
            },
        );

        // Initialize LoRA B with zeros (standard LoRA practice)
        // This ensures the adapter starts as identity
        let lora_b = root.var(
            &format!("{}_lora_b", config.module_name),
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
        let vs = nn::VarStore::new(device);
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
                module_name: module_name.clone(),
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

    /// Load all LoRA weights with dynamic tensor discovery
    pub fn load(&mut self, path: &str) -> Result<()> {
        // First try the standard VarStore load approach
        match self.vs.load(path) {
            Ok(()) => {
                tracing::debug!("Successfully loaded LoRA weights using VarStore::load");
                return Ok(());
            }
            Err(e) => {
                tracing::debug!(
                    "VarStore::load failed: {}, trying dynamic tensor discovery",
                    e
                );
            }
        }

        // Fallback: Dynamic tensor discovery and loading
        self.load_with_dynamic_discovery(path)
    }

    /// Load LoRA weights with dynamic tensor name discovery
    fn load_with_dynamic_discovery(&mut self, path: &str) -> Result<()> {
        tracing::info!(
            "Attempting dynamic tensor discovery from SafeTensors file: {}",
            path
        );

        // Load the SafeTensors file to discover available tensors
        let safetensors_data = std::fs::read(path)?;
        let safetensors = safetensors::SafeTensors::deserialize(&safetensors_data)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize SafeTensors: {}", e))?;

        // Get all tensor names in the file
        let available_tensors: Vec<String> = safetensors
            .names()
            .into_iter()
            .map(|s| s.to_string())
            .collect();
        tracing::debug!(
            "Available tensors in SafeTensors file: {:?}",
            available_tensors
        );

        // Use the lower-level approach to load tensors
        // Instead of directly loading into VarStore, we'll use the load mechanism with partial loading
        let mut tensors_to_load = std::collections::HashMap::new();

        // First, collect matching tensor names and their SafeTensors data
        for available_name in &available_tensors {
            // Try to find a VarStore tensor that matches this SafeTensors tensor
            let vs_tensors = self.vs.variables();

            let matching_vs_name = vs_tensors.keys().find(|vs_name| {
                // Exact match
                available_name == *vs_name ||
                    // Match without module prefix (e.g., "lora_a" matches "q_proj_lora_a")
                    available_name.ends_with(&format!("_{}", vs_name)) ||
                    // Match with different naming conventions
                    self.tensor_names_match(vs_name, available_name)
            });

            if let Some(vs_name) = matching_vs_name {
                tracing::debug!("Found tensor match: '{}' -> '{}'", available_name, vs_name);
                match safetensors.tensor(available_name) {
                    Ok(tensor_data) => {
                        // Convert SafeTensors tensor to raw data for tch loading
                        let shape: Vec<i64> =
                            tensor_data.shape().iter().map(|&s| s as i64).collect();
                        let tch_tensor = match tensor_data.dtype() {
                            safetensors::Dtype::F32 => {
                                let f32_data: Vec<f32> = tensor_data
                                    .data()
                                    .chunks_exact(4)
                                    .map(|chunk| {
                                        f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                                    })
                                    .collect();
                                Tensor::from_slice(&f32_data)
                                    .reshape(&shape)
                                    .to_device(self.device)
                            }
                            safetensors::Dtype::F16 => {
                                // Convert f16 to f32
                                let f16_data: &[u8] = tensor_data.data();
                                let f32_data: Vec<f32> = f16_data
                                    .chunks_exact(2)
                                    .map(|chunk| {
                                        let f16_bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                                        half::f16::from_bits(f16_bits).to_f32()
                                    })
                                    .collect();
                                Tensor::from_slice(&f32_data)
                                    .reshape(&shape)
                                    .to_device(self.device)
                            }
                            _ => {
                                tracing::warn!(
                                    "Unsupported tensor dtype for {}: {:?}",
                                    available_name,
                                    tensor_data.dtype()
                                );
                                continue;
                            }
                        };

                        tensors_to_load.insert(vs_name.clone(), tch_tensor);
                        tracing::debug!(
                            "Prepared tensor '{}' -> '{}' with shape {:?}",
                            available_name,
                            vs_name,
                            shape
                        );
                    }
                    Err(e) => {
                        tracing::warn!("Failed to extract tensor '{}': {}", available_name, e);
                    }
                }
            } else {
                tracing::debug!(
                    "No VarStore match found for SafeTensors tensor '{}' (expected patterns: {:?})",
                    available_name,
                    vs_tensors.keys().collect::<Vec<_>>()
                );
            }
        }

        // Write matched tensors to a temporary SafeTensors file and load via VarStore
        if tensors_to_load.is_empty() {
            return Err(anyhow::anyhow!(
                "No matching tensors found between SafeTensors file and LoRA model"
            ));
        }

        let temp_file = format!("{}.temp", path);
        let mut tensor_vec = Vec::new();

        for (vs_name, tensor) in tensors_to_load {
            // Convert tensor back to CPU for saving
            let cpu_tensor = tensor.to(tch::Device::Cpu);
            tensor_vec.push((vs_name.clone(), cpu_tensor));
        }

        // Save to temporary file
        tch::Tensor::save_multi(&tensor_vec, &temp_file)?;
        let loaded_count = tensor_vec.len();

        // Load via VarStore
        match self.vs.load(&temp_file) {
            Ok(()) => {
                tracing::debug!(
                    "Successfully loaded {} tensors via temporary file",
                    loaded_count
                );
                // Clean up temporary file
                let _ = std::fs::remove_file(&temp_file);
            }
            Err(e) => {
                // Clean up temporary file on error
                let _ = std::fs::remove_file(&temp_file);
                return Err(anyhow::anyhow!(
                    "Failed to load tensors via VarStore: {}",
                    e
                ));
            }
        }

        if loaded_count == 0 {
            return Err(anyhow::anyhow!(
                "No tensors were successfully loaded from SafeTensors file"
            ));
        }

        tracing::info!(
            "Successfully loaded {} tensors using dynamic discovery",
            loaded_count
        );
        Ok(())
    }

    /// Check if two tensor names should be considered a match
    fn tensor_names_match(&self, vs_name: &str, safetensors_name: &str) -> bool {
        // Handle common naming variations
        let vs_normalized = vs_name.replace("__", "_");
        let st_normalized = safetensors_name.replace("__", "_");

        // Check for various common patterns
        if vs_normalized == st_normalized {
            return true;
        }

        // Handle hierarchical naming like "model.layers.0.q_proj.lora_a.weight" -> "q_proj_lora_a"
        if let Some(captures) = extract_lora_components(safetensors_name) {
            let (module, matrix) = captures;
            let expected_vs_name = format!("{}_{}", module, matrix);
            if vs_name == expected_vs_name {
                return true;
            }
        }

        // Fallback patterns
        safetensors_name.contains(&vs_normalized) ||
        vs_name.contains(safetensors_name) ||
        // Handle patterns like "lora_a" matching "lora_a__2"
        (safetensors_name.starts_with(&format!("{}_", vs_name)) || safetensors_name.starts_with(&format!("{}__", vs_name)))
    }

    /// Get total number of trainable parameters
    pub fn num_parameters(&self) -> i64 {
        self.layers
            .values()
            .map(|layer| layer.num_parameters())
            .sum()
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
        self.layers
            .values()
            .map(|layer| {
                layer.lora_a.size().iter().product::<i64>()
                    + layer.lora_b.size().iter().product::<i64>()
            })
            .sum()
    }
}
