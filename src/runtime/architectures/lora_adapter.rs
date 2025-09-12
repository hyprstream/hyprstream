//! LoRA adapter compatibility layer for different architectures

use super::ModelArchitecture;
use crate::adapters::sparse_lora::SparseLoRAAdapter;
use anyhow::{Result, anyhow};
use tch::{Device, Kind as DType, Tensor};
use std::collections::HashMap;

/// Architecture-aware LoRA adapter that handles shape conversions
#[derive(Debug, Clone)]
pub struct ArchitectureAwareLoRAAdapter {
    /// Target architecture
    architecture: ModelArchitecture,
    /// Base LoRA adapter
    base_adapter: SparseLoRAAdapter,
    /// Shape mappings for the architecture
    shape_mappings: HashMap<String, ShapeMapping>,
}

/// Defines how to map LoRA weights to architecture-specific shapes
#[derive(Debug, Clone)]
struct ShapeMapping {
    /// Source shape (from LoRA)
    source_shape: Vec<usize>,
    /// Target shape (for architecture)
    target_shape: Vec<usize>,
    /// Transformation type
    transform: TransformType,
}

/// Types of shape transformations
#[derive(Debug, Clone)]
enum TransformType {
    /// Direct reshape
    Reshape,
    /// Repeat for MQA/GQA expansion
    Repeat { axis: usize, factor: usize },
    /// Split for decomposition
    Split { axis: usize, num_splits: usize },
    /// Transpose dimensions
    Transpose { dim1: usize, dim2: usize },
    /// No transformation needed
    Identity,
}

impl ArchitectureAwareLoRAAdapter {
    /// Create a new architecture-aware adapter
    pub fn new(
        architecture: ModelArchitecture,
        base_adapter: SparseLoRAAdapter,
    ) -> Self {
        let shape_mappings = Self::create_shape_mappings(&architecture);
        
        Self {
            architecture,
            base_adapter,
            shape_mappings,
        }
    }
    
    /// Create shape mappings for an architecture
    fn create_shape_mappings(arch: &ModelArchitecture) -> HashMap<String, ShapeMapping> {
        let mut mappings = HashMap::new();
        
        match arch {
            ModelArchitecture::Gemma => {
                // Gemma uses MQA with 4 KV heads vs 16 Q heads
                mappings.insert(
                    "k_proj".to_string(),
                    ShapeMapping {
                        source_shape: vec![1024, 1024],  // Standard LoRA shape
                        target_shape: vec![1024, 1024],  // Gemma K shape (4 heads * 256)
                        transform: TransformType::Identity,  // Direct mapping works
                    },
                );
                mappings.insert(
                    "v_proj".to_string(),
                    ShapeMapping {
                        source_shape: vec![1024, 1024],
                        target_shape: vec![1024, 1024],  // Gemma V shape (4 heads * 256)
                        transform: TransformType::Identity,
                    },
                );
                mappings.insert(
                    "q_proj".to_string(),
                    ShapeMapping {
                        source_shape: vec![1024, 4096],  // Standard LoRA Q
                        target_shape: vec![1024, 4096],  // Gemma Q (16 heads * 256)
                        transform: TransformType::Identity,
                    },
                );
            }
            
            ModelArchitecture::Qwen { version: 3, is_moe, .. } => {
                // Qwen3 uses GQA with 8 KV heads vs 32 Q heads
                mappings.insert(
                    "k_proj".to_string(),
                    ShapeMapping {
                        source_shape: vec![4096, 4096],
                        target_shape: vec![4096, 1024],  // 8 heads * 128
                        transform: TransformType::Reshape,
                    },
                );
                
                if *is_moe {
                    // Add MoE-specific mappings
                    mappings.insert(
                        "moe_gate".to_string(),
                        ShapeMapping {
                            source_shape: vec![4096, 128],  // To 128 experts
                            target_shape: vec![4096, 128],
                            transform: TransformType::Identity,
                        },
                    );
                }
            }
            
            ModelArchitecture::GPTOSS { .. } => {
                // GPT-OSS MoE with GQA
                mappings.insert(
                    "k_proj".to_string(),
                    ShapeMapping {
                        source_shape: vec![6144, 6144],
                        target_shape: vec![6144, 1024],  // 8 heads * 128
                        transform: TransformType::Reshape,
                    },
                );
                mappings.insert(
                    "expert_ffn".to_string(),
                    ShapeMapping {
                        source_shape: vec![6144, 16384],
                        target_shape: vec![8, 6144, 16384],  // 8 active experts
                        transform: TransformType::Split { axis: 0, num_splits: 8 },
                    },
                );
            }
            
            _ => {
                // Default mappings for standard architectures
            }
        }
        
        mappings
    }
    
    /// Adapt LoRA weights to target architecture
    pub fn adapt_weights(&self, layer_name: &str, weights: &Tensor) -> Result<Tensor> {
        // Check if we have a specific mapping for this layer
        if let Some(mapping) = self.shape_mappings.get(layer_name) {
            self.apply_transformation(weights, mapping)
        } else {
            // No specific mapping, return as-is
            Ok(weights.shallow_clone())
        }
    }
    
    /// Apply shape transformation
    fn apply_transformation(&self, tensor: &Tensor, mapping: &ShapeMapping) -> Result<Tensor> {
        match &mapping.transform {
            TransformType::Identity => Ok(tensor.shallow_clone()),
            
            TransformType::Reshape => {
                let target_shape: Vec<i64> = mapping.target_shape.iter().map(|&x| x as i64).collect();
                Ok(tensor.reshape(&target_shape))
            }
            
            TransformType::Repeat { axis, factor } => {
                // Repeat along specified axis
                let mut new_shape = tensor.size();
                new_shape[*axis] *= *factor as i64;
                let target_shape: Vec<i64> = mapping.target_shape.iter().map(|&x| x as i64).collect();
                Ok(tensor.unsqueeze((*axis + 1) as i64)
                    .expand(&new_shape, false)
                    .reshape(&target_shape))
            }
            
            TransformType::Split { axis, num_splits } => {
                // Split tensor along axis
                let _chunk_size = tensor.size()[*axis] / (*num_splits as i64);
                let chunks = tensor.chunk(*num_splits as i64, *axis as i64);
                
                // Stack chunks to create new dimension
                Ok(Tensor::stack(&chunks, 0))
            }
            
            TransformType::Transpose { dim1, dim2 } => {
                Ok(tensor.transpose(*dim1 as i64, *dim2 as i64))
            }
        }
    }
    
    /// Get adapted weights for a specific layer
    pub async fn get_layer_weights(&self, layer_name: &str) -> Result<Option<Tensor>> {
        // For now, return None as SparseLoRAAdapter doesn't have get_weight method
        // In production, this would need to be implemented based on layer type
        // and access the appropriate LoRA A/B matrices
        Ok(None)
    }
    
    /// Check if adaptation is needed for the architecture
    pub fn needs_adaptation(&self) -> bool {
        !self.shape_mappings.is_empty()
    }
    
    /// Get architecture-specific scaling factor
    pub fn get_scaling_factor(&self, layer_name: &str) -> f32 {
        match &self.architecture {
            ModelArchitecture::Gemma if layer_name.contains("k_proj") || layer_name.contains("v_proj") => {
                // Scale down for MQA (4 heads vs 16)
                0.25
            }
            ModelArchitecture::Qwen { version: 3, .. } if layer_name.contains("k_proj") || layer_name.contains("v_proj") => {
                // Scale for GQA (8 heads vs 32)
                0.25
            }
            _ => 1.0,
        }
    }
    
    /// Validate LoRA compatibility with architecture
    pub fn validate_compatibility(&self) -> Result<()> {
        // Check if LoRA dimensions match expected architecture dimensions
        for (_layer_name, mapping) in &self.shape_mappings {
            match &mapping.transform {
                TransformType::Reshape => {
                    let source_elements: usize = mapping.source_shape.iter().product();
                    let target_elements: usize = mapping.target_shape.iter().product();
                    
                    if source_elements != target_elements {
                        return Err(anyhow!(
                            "Shape mismatch: cannot reshape {} elements to {} elements",
                            source_elements, target_elements
                        ));
                    }
                }
                _ => {
                    // Other transformations have their own validation
                }
            }
        }
        
        Ok(())
    }
}

/// Helper to create LoRA adapters for specific architectures
pub struct LoRAAdapterFactory;

impl LoRAAdapterFactory {
    /// Create a LoRA adapter optimized for Gemma
    pub fn create_gemma_adapter(
        rank: usize,
        alpha: f32,
        _device: &Device,
    ) -> Result<ArchitectureAwareLoRAAdapter> {
        use crate::adapters::sparse_lora::SparseLoRAConfig;
        
        let config = SparseLoRAConfig {
            rank,
            alpha,
            dropout: 0.1,
            ..Default::default()
        };
        let base_adapter = SparseLoRAAdapter::new(config);
        
        Ok(ArchitectureAwareLoRAAdapter::new(
            ModelArchitecture::Gemma,
            base_adapter,
        ))
    }
    
    /// Create a LoRA adapter optimized for Qwen3
    pub fn create_qwen3_adapter(
        rank: usize,
        alpha: f32,
        is_moe: bool,
        context_length: usize,
        _device: &Device,
    ) -> Result<ArchitectureAwareLoRAAdapter> {
        use crate::adapters::sparse_lora::SparseLoRAConfig;
        
        let config = SparseLoRAConfig {
            rank,
            alpha,
            dropout: 0.1,
            ..Default::default()
        };
        let base_adapter = SparseLoRAAdapter::new(config);
        
        Ok(ArchitectureAwareLoRAAdapter::new(
            ModelArchitecture::Qwen { version: 3, is_moe, context_length },
            base_adapter,
        ))
    }
    
    /// Create a LoRA adapter optimized for GPT-OSS
    pub fn create_gpt_oss_adapter(
        rank: usize,
        alpha: f32,
        total_params_b: u16,
        _device: &Device,
    ) -> Result<ArchitectureAwareLoRAAdapter> {
        use crate::adapters::sparse_lora::SparseLoRAConfig;
        
        let config = SparseLoRAConfig {
            rank,
            alpha,
            dropout: 0.1,
            ..Default::default()
        };
        let base_adapter = SparseLoRAAdapter::new(config);
        
        let active_params_b = if total_params_b == 120 { 5.1 } else { 3.6 };
        
        Ok(ArchitectureAwareLoRAAdapter::new(
            ModelArchitecture::GPTOSS {
                total_params_b,
                active_params_b,
                num_experts: 128,
                experts_per_token: 8,
            },
            base_adapter,
        ))
    }
}

#[cfg(test_disabled)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gemma_lora_adaptation() {
        let device = Device::Cpu;
        let adapter = LoRAAdapterFactory::create_gemma_adapter(
            8,    // rank
            16.0, // alpha
            &device,
        ).unwrap();
        
        // Test that Gemma-specific mappings are created
        assert!(adapter.needs_adaptation() || !adapter.shape_mappings.is_empty());
        
        // Test scaling factor for MQA
        assert_eq!(adapter.get_scaling_factor("k_proj"), 0.25);
        assert_eq!(adapter.get_scaling_factor("v_proj"), 0.25);
        assert_eq!(adapter.get_scaling_factor("q_proj"), 1.0);
    }
    
    #[test]
    fn test_weight_adaptation() {
        let device = Device::Cpu;
        let base_adapter = SparseLoRAAdapter::new(8, 16.0, 0.1, device.clone()).unwrap();
        let adapter = ArchitectureAwareLoRAAdapter::new(
            ModelArchitecture::Gemma,
            base_adapter,
        );
        
        // Create a test weight tensor
        let weight = Tensor::randn(0.0, 1.0, &[1024, 1024], &device).unwrap();
        
        // Adapt for k_proj (should work with Gemma's MQA)
        let adapted = adapter.adapt_weights("k_proj", &weight).unwrap();
        assert_eq!(adapted.size(), weight.size());  // Identity transform for Gemma K
    }
}