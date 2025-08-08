//! Registry for managing LoRA layers

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use anyhow::Result;

/// LoRA layer registration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRALayer {
    pub id: String,
    pub name: String,
    pub base_model: String,
    pub config: crate::api::LoRAConfig,
    pub created_at: i64,
    pub updated_at: i64,
    pub training_enabled: bool,
    pub total_tokens_trained: u64,
    pub sparsity_ratio: f32,
}

/// Registry for all LoRA layers
pub struct LoRARegistry {
    layers: Arc<RwLock<HashMap<String, LoRALayer>>>,
    metrics: Arc<RwLock<HashMap<String, LayerMetrics>>>,
}

/// Metrics for a LoRA layer
#[derive(Debug, Clone, Default)]
pub struct LayerMetrics {
    pub total_requests: u64,
    pub total_tokens: u64,
    pub avg_latency_ms: f64,
    pub last_accessed: i64,
}

impl LoRARegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            layers: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Register a new LoRA layer
    pub async fn register(&self, layer: LoRALayer) -> Result<()> {
        let mut layers = self.layers.write().await;
        if layers.contains_key(&layer.id) {
            return Err(anyhow::anyhow!("LoRA layer {} already exists", layer.id));
        }
        
        let layer_id = layer.id.clone();
        layers.insert(layer.id.clone(), layer);
        
        // Initialize metrics
        let mut metrics = self.metrics.write().await;
        metrics.insert(layer_id, LayerMetrics::default());
        
        Ok(())
    }
    
    /// Unregister a LoRA layer
    pub async fn unregister(&self, lora_id: &str) -> Result<()> {
        let mut layers = self.layers.write().await;
        layers.remove(lora_id)
            .ok_or_else(|| anyhow::anyhow!("LoRA layer {} not found", lora_id))?;
        
        let mut metrics = self.metrics.write().await;
        metrics.remove(lora_id);
        
        Ok(())
    }
    
    /// Get a LoRA layer by ID
    pub async fn get(&self, lora_id: &str) -> Result<LoRALayer> {
        let layers = self.layers.read().await;
        layers.get(lora_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("LoRA layer {} not found", lora_id))
    }
    
    /// List all LoRA layers
    pub async fn list_all(&self) -> Result<Vec<LoRALayer>> {
        let layers = self.layers.read().await;
        Ok(layers.values().cloned().collect())
    }
    
    /// Update layer metrics
    pub async fn update_metrics(
        &self,
        lora_id: &str,
        requests: u64,
        tokens: u64,
        latency_ms: f64,
    ) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        let layer_metrics = metrics.entry(lora_id.to_string())
            .or_insert_with(LayerMetrics::default);
        
        layer_metrics.total_requests += requests;
        layer_metrics.total_tokens += tokens;
        layer_metrics.avg_latency_ms = 
            (layer_metrics.avg_latency_ms * (layer_metrics.total_requests - requests) as f64 
             + latency_ms * requests as f64) / layer_metrics.total_requests as f64;
        layer_metrics.last_accessed = chrono::Utc::now().timestamp();
        
        Ok(())
    }
    
    /// Get statistics for a LoRA layer
    pub async fn get_stats(&self, lora_id: &str) -> Result<crate::api::LoRAStats> {
        let metrics = self.metrics.read().await;
        let layer_metrics = metrics.get(lora_id)
            .ok_or_else(|| anyhow::anyhow!("Metrics not found for {}", lora_id))?;
        
        let layers = self.layers.read().await;
        let layer = layers.get(lora_id)
            .ok_or_else(|| anyhow::anyhow!("Layer not found for {}", lora_id))?;
        
        Ok(crate::api::LoRAStats {
            total_requests: layer_metrics.total_requests,
            total_tokens_generated: layer_metrics.total_tokens,
            avg_latency_ms: layer_metrics.avg_latency_ms,
            sparsity_ratio: layer.sparsity_ratio,
            memory_usage_mb: estimate_memory_usage(layer),
            compression_ratio: estimate_compression_ratio(layer),
        })
    }
    
    /// Update training progress
    pub async fn update_training_progress(
        &self,
        lora_id: &str,
        tokens_trained: u64,
    ) -> Result<()> {
        let mut layers = self.layers.write().await;
        let layer = layers.get_mut(lora_id)
            .ok_or_else(|| anyhow::anyhow!("LoRA layer {} not found", lora_id))?;
        
        layer.total_tokens_trained += tokens_trained;
        layer.updated_at = chrono::Utc::now().timestamp();
        
        Ok(())
    }
}

fn estimate_memory_usage(layer: &LoRALayer) -> u64 {
    // Estimate based on rank and sparsity
    let dense_params = layer.config.rank * 1536 * 2; // rank * hidden_dim * 2 (for A and B matrices)
    let sparse_params = (dense_params as f32 * (1.0 - layer.sparsity_ratio)) as u64;
    sparse_params * 4 / (1024 * 1024) // Convert to MB (4 bytes per float32)
}

fn estimate_compression_ratio(layer: &LoRALayer) -> f32 {
    // Compression ratio from sparsity and neural compression
    let base_ratio = 1.0 / (1.0 - layer.sparsity_ratio);
    if layer.config.use_neural_compression {
        base_ratio * 10.0 // Neural compression adds 10x
    } else {
        base_ratio
    }
}

use chrono;