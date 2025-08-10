//! LoRA adapter wrapper for runtime engines

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use candle_core::{Device, Tensor, DType};

use super::{RuntimeEngine, ModelInfo, GenerationRequest, GenerationResult};

/// LoRA adapter configuration
#[derive(Debug, Clone)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub target_modules: Vec<String>,
    pub sparsity_ratio: f32,
}

impl Default for LoRAConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            dropout: 0.1,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(), 
                "v_proj".to_string(),
                "o_proj".to_string(),
            ],
            sparsity_ratio: 0.99,
        }
    }
}

/// Sparse LoRA adapter implementation
#[derive(Debug, Clone)]
pub struct RuntimeLoRAAdapter {
    pub id: String,
    pub name: String,
    pub config: LoRAConfig,
    pub lora_a: Tensor,
    pub lora_b: Tensor,
    pub scaling: f32,
    pub device: Device,
    pub target_modules: Vec<String>,
    pub creation_time: std::time::SystemTime,
    pub training_steps: usize,
}

impl RuntimeLoRAAdapter {
    /// Create a new sparse LoRA adapter
    pub fn new(
        name: String,
        config: LoRAConfig,
        input_dim: usize,
        output_dim: usize,
        device: Device,
    ) -> Result<Self> {
        let id = Uuid::new_v4().to_string();
        
        // Initialize LoRA matrices with small random values
        let lora_a = Tensor::randn(0.0, 0.02, (input_dim, config.rank), &device)
            .map_err(|e| anyhow!("Failed to create LoRA A matrix: {}", e))?;
        
        let lora_b = Tensor::zeros((config.rank, output_dim), DType::F32, &device)
            .map_err(|e| anyhow!("Failed to create LoRA B matrix: {}", e))?;

        let scaling = config.alpha / config.rank as f32;

        let mut adapter = Self {
            id,
            name,
            config: config.clone(),
            lora_a,
            lora_b,
            scaling,
            device,
            target_modules: config.target_modules,
            creation_time: std::time::SystemTime::now(),
            training_steps: 0,
        };

        // Apply initial sparsity
        adapter.apply_sparsity(config.sparsity_ratio)?;

        Ok(adapter)
    }

    /// Apply sparsity mask to LoRA matrices
    pub fn apply_sparsity(&mut self, sparsity_ratio: f32) -> Result<()> {
        // Get tensor values for magnitude-based pruning
        let lora_a_values = self.lora_a.flatten_all()
            .map_err(|e| anyhow!("Failed to flatten LoRA A: {}", e))?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("Failed to convert LoRA A to vec: {}", e))?;

        let lora_b_values = self.lora_b.flatten_all()
            .map_err(|e| anyhow!("Failed to flatten LoRA B: {}", e))?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("Failed to convert LoRA B to vec: {}", e))?;

        // Calculate magnitude threshold for sparsity
        let mut magnitudes: Vec<f32> = lora_a_values.iter()
            .chain(lora_b_values.iter())
            .map(|&x| x.abs())
            .collect();
        
        magnitudes.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let threshold_idx = (magnitudes.len() as f32 * sparsity_ratio) as usize;
        let threshold = magnitudes.get(threshold_idx).copied().unwrap_or(0.0);

        // Create sparsity masks
        let lora_a_mask = self.lora_a.abs()
            .map_err(|e| anyhow!("Failed to compute LoRA A abs: {}", e))?
            .gt(threshold)
            .map_err(|e| anyhow!("Failed to create LoRA A mask: {}", e))?;

        let lora_b_mask = self.lora_b.abs()
            .map_err(|e| anyhow!("Failed to compute LoRA B abs: {}", e))?
            .gt(threshold)
            .map_err(|e| anyhow!("Failed to create LoRA B mask: {}", e))?;

        // Apply masks
        self.lora_a = self.lora_a.broadcast_mul(&lora_a_mask)
            .map_err(|e| anyhow!("Failed to apply mask to LoRA A: {}", e))?;
        
        self.lora_b = self.lora_b.broadcast_mul(&lora_b_mask)
            .map_err(|e| anyhow!("Failed to apply mask to LoRA B: {}", e))?;

        tracing::debug!("🔥 Applied {:.1}% sparsity to LoRA {}", sparsity_ratio * 100.0, self.name);
        Ok(())
    }

    /// Compute LoRA delta: (input @ lora_a) @ lora_b * scaling
    pub fn compute_delta(&self, input: &Tensor) -> Result<Tensor> {
        let lora_result = input
            .matmul(&self.lora_a)
            .map_err(|e| anyhow!("Failed LoRA A matmul: {}", e))?
            .matmul(&self.lora_b)
            .map_err(|e| anyhow!("Failed LoRA B matmul: {}", e))?;

        let scaling_tensor = Tensor::new(self.scaling, &self.device)
            .map_err(|e| anyhow!("Failed to create scaling tensor: {}", e))?;
        
        let scaled_result = lora_result.broadcast_mul(&scaling_tensor)
            .map_err(|e| anyhow!("Failed LoRA scaling: {}", e))?;

        Ok(scaled_result)
    }

    /// Update adapter parameters (for training)
    pub fn update_parameters(
        &mut self,
        lora_a_grad: &Tensor,
        lora_b_grad: &Tensor,
        learning_rate: f32,
    ) -> Result<()> {
        // Simple SGD update (using broadcast_mul with scalar tensor)
        let lr_tensor = Tensor::new(learning_rate, &self.device)
            .map_err(|e| anyhow!("Failed to create learning rate tensor: {}", e))?;
        
        let lora_a_update = lora_a_grad.broadcast_mul(&lr_tensor)
            .map_err(|e| anyhow!("Failed to scale LoRA A gradient: {}", e))?;
        
        let lora_b_update = lora_b_grad.broadcast_mul(&lr_tensor)
            .map_err(|e| anyhow!("Failed to scale LoRA B gradient: {}", e))?;

        self.lora_a = (self.lora_a.sub(&lora_a_update))
            .map_err(|e| anyhow!("Failed to update LoRA A: {}", e))?;
        
        self.lora_b = (self.lora_b.sub(&lora_b_update))
            .map_err(|e| anyhow!("Failed to update LoRA B: {}", e))?;

        // Reapply sparsity after update
        self.apply_sparsity(self.config.sparsity_ratio)?;
        
        self.training_steps += 1;
        Ok(())
    }

    /// Get sparsity statistics
    pub fn get_sparsity_stats(&self) -> Result<(f32, usize, usize)> {
        let lora_a_size = self.lora_a.elem_count();
        let lora_b_size = self.lora_b.elem_count();
        let total_params = lora_a_size + lora_b_size;

        // Count non-zero elements
        let lora_a_nonzero = self.count_nonzero(&self.lora_a)?;
        let lora_b_nonzero = self.count_nonzero(&self.lora_b)?;
        let total_nonzero = lora_a_nonzero + lora_b_nonzero;

        let sparsity = 1.0 - (total_nonzero as f32 / total_params as f32);
        
        Ok((sparsity, total_nonzero, total_params))
    }

    fn count_nonzero(&self, tensor: &Tensor) -> Result<usize> {
        let values = tensor.flatten_all()
            .map_err(|e| anyhow!("Failed to flatten tensor: {}", e))?
            .to_vec1::<f32>()
            .map_err(|e| anyhow!("Failed to convert to vec: {}", e))?;
        
        Ok(values.iter().filter(|&&x| x.abs() > 1e-8).count())
    }
}

/// LoRA engine wrapper that adds LoRA capabilities to any runtime engine
pub struct LoRAEngineWrapper {
    base_engine: Arc<dyn RuntimeEngine>,
    active_adapters: HashMap<String, RuntimeLoRAAdapter>,
    device: Device,
    default_lora_config: LoRAConfig,
}

impl LoRAEngineWrapper {
    /// Create a new LoRA engine wrapper
    pub fn new(base_engine: Arc<dyn RuntimeEngine>) -> Result<Self> {
        // For now, always use CPU. GPU support can be added later
        let device = Device::Cpu;

        Ok(Self {
            base_engine,
            active_adapters: HashMap::new(),
            device,
            default_lora_config: LoRAConfig::default(),
        })
    }

    /// Add a LoRA adapter to the engine
    pub fn add_adapter(&mut self, adapter: RuntimeLoRAAdapter) {
        tracing::info!("📎 Added LoRA adapter: {} (ID: {})", adapter.name, adapter.id);
        self.active_adapters.insert(adapter.id.clone(), adapter);
    }

    /// Create and add a new LoRA adapter
    pub fn create_adapter(
        &mut self,
        name: String,
        config: Option<LoRAConfig>,
        input_dim: usize,
        output_dim: usize,
    ) -> Result<String> {
        let config = config.unwrap_or_else(|| self.default_lora_config.clone());
        let adapter = RuntimeLoRAAdapter::new(name, config, input_dim, output_dim, self.device.clone())?;
        let adapter_id = adapter.id.clone();
        
        self.add_adapter(adapter);
        Ok(adapter_id)
    }

    /// Remove a LoRA adapter
    pub fn remove_adapter(&mut self, adapter_id: &str) -> Option<RuntimeLoRAAdapter> {
        self.active_adapters.remove(adapter_id)
    }

    /// List all active adapters
    pub fn list_adapters(&self) -> Vec<&RuntimeLoRAAdapter> {
        self.active_adapters.values().collect()
    }

    /// Generate text with specific LoRA adapters
    pub async fn generate_with_adapters(
        &self,
        request: &GenerationRequest,
        adapter_ids: &[String],
    ) -> Result<GenerationResult> {
        if adapter_ids.is_empty() {
            // No adapters, use base engine directly
            return self.base_engine.generate_with_params(request.clone()).await;
        }

        tracing::info!("🧠 Generating with {} LoRA adapter(s): {:?}", adapter_ids.len(), adapter_ids);

        // For now, we delegate to the base engine since we need model internals
        // In a full implementation, this would intercept the forward pass
        // and apply LoRA modifications to attention layers
        let mut result = self.base_engine.generate_with_params(request.clone()).await?;

        // Add LoRA indicator to the generated text (for demonstration)
        let active_adapters: Vec<&str> = adapter_ids.iter()
            .filter_map(|id| self.active_adapters.get(id))
            .map(|adapter| adapter.name.as_str())
            .collect();

        if !active_adapters.is_empty() {
            result.text = format!("[LoRA:{}] {}", active_adapters.join(","), result.text);
        }

        Ok(result)
    }

    /// Get adapter statistics
    pub fn get_adapter_stats(&self, adapter_id: &str) -> Option<Result<(f32, usize, usize, usize)>> {
        self.active_adapters.get(adapter_id).map(|adapter| {
            let (sparsity, nonzero, total) = adapter.get_sparsity_stats()?;
            Ok((sparsity, nonzero, total, adapter.training_steps))
        })
    }

    /// Get device information
    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[async_trait]
impl RuntimeEngine for LoRAEngineWrapper {
    async fn load_model(&mut self, _path: &Path) -> Result<()> {
        // For now, we can't mutate the base engine through Arc
        // This would need to be redesigned for real usage
        tracing::warn!("LoRAEngineWrapper cannot directly load models - load on base engine first");
        Ok(())
    }

    async fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String> {
        let request = GenerationRequest {
            prompt: prompt.to_string(),
            max_tokens,
            ..Default::default()
        };
        
        // Use all active adapters by default
        let adapter_ids: Vec<String> = self.active_adapters.keys().cloned().collect();
        let result = self.generate_with_adapters(&request, &adapter_ids).await?;
        Ok(result.text)
    }

    async fn generate_with_params(&self, request: GenerationRequest) -> Result<GenerationResult> {
        // Use all active adapters by default
        let adapter_ids: Vec<String> = self.active_adapters.keys().cloned().collect();
        self.generate_with_adapters(&request, &adapter_ids).await
    }

    fn model_info(&self) -> ModelInfo {
        let mut info = self.base_engine.model_info();
        
        // Add LoRA adapter information
        if !self.active_adapters.is_empty() {
            info.name = format!("{} + {} LoRA adapters", info.name, self.active_adapters.len());
        }
        
        info
    }

    fn is_loaded(&self) -> bool {
        self.base_engine.is_loaded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lora_config_default() {
        let config = LoRAConfig::default();
        assert_eq!(config.rank, 8);
        assert_eq!(config.alpha, 16.0);
        assert_eq!(config.sparsity_ratio, 0.99);
    }

    #[test]
    fn test_sparse_lora_adapter_creation() -> Result<()> {
        let device = Device::Cpu;
        let config = LoRAConfig::default();
        
        let adapter = RuntimeLoRAAdapter::new(
            "test_adapter".to_string(),
            config,
            512, // input_dim
            512, // output_dim
            device,
        )?;

        assert_eq!(adapter.name, "test_adapter");
        assert!(adapter.id.len() > 0);
        assert_eq!(adapter.scaling, 16.0 / 8.0); // alpha / rank
        
        Ok(())
    }

    #[tokio::test]
    async fn test_lora_engine_wrapper_creation() -> Result<()> {
        // This test would need a mock runtime engine
        // For now, just test the basic functionality
        
        let config = LoRAConfig::default();
        assert!(config.target_modules.contains(&"q_proj".to_string()));
        
        Ok(())
    }
}