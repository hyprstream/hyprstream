//! Registry for managing LoRA layers

use std::collections::HashMap;
use std::sync::Arc;
use std::path::PathBuf;
use tokio::sync::RwLock;
use tokio::fs;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use uuid::Uuid;

/// UUID-based LoRA identifier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LoRAId(pub Uuid);

impl LoRAId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for LoRAId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::str::FromStr for LoRAId {
    type Err = uuid::Error;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

/// LoRA layer registration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRALayer {
    pub id: LoRAId,
    pub name: String,
    pub base_model: String, // Can be UUID or URI
    pub config: crate::api::LoRAConfig,
    pub created_at: i64,
    pub updated_at: i64,
    pub training_enabled: bool,
    pub total_tokens_trained: u64,
    pub sparsity_ratio: f32,
}

impl LoRALayer {
    /// Create a new LoRA layer with generated UUID
    pub fn new(
        name: String,
        base_model: String,
        config: crate::api::LoRAConfig,
        sparsity_ratio: f32,
    ) -> Self {
        let now = chrono::Utc::now().timestamp();
        Self {
            id: LoRAId::new(),
            name,
            base_model,
            config,
            created_at: now,
            updated_at: now,
            training_enabled: false,
            total_tokens_trained: 0,
            sparsity_ratio,
        }
    }
}

/// Registry for all LoRA layers
pub struct LoRARegistry {
    layers: Arc<RwLock<HashMap<LoRAId, LoRALayer>>>,
    metrics: Arc<RwLock<HashMap<LoRAId, LayerMetrics>>>,
    // Legacy mapping for backward compatibility
    name_to_id: Arc<RwLock<HashMap<String, LoRAId>>>,
    // Persistence
    base_dir: PathBuf,
    registry_file: PathBuf,
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
    /// Create new registry with persistence
    pub async fn new(base_dir: PathBuf) -> Result<Self> {
        // Ensure base directory exists
        fs::create_dir_all(&base_dir).await?;
        
        let registry_file = base_dir.join("lora_registry.json");
        let registry = Self {
            layers: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            name_to_id: Arc::new(RwLock::new(HashMap::new())),
            base_dir,
            registry_file,
        };
        
        // Load existing data
        registry.load_registry().await?;
        
        Ok(registry)
    }
    
    /// Create new in-memory registry (for testing)
    pub fn new_in_memory() -> Self {
        Self {
            layers: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            name_to_id: Arc::new(RwLock::new(HashMap::new())),
            base_dir: PathBuf::new(),
            registry_file: PathBuf::new(),
        }
    }
    
    /// Load registry from disk
    async fn load_registry(&self) -> Result<()> {
        if !self.registry_file.exists() {
            return Ok(());
        }
        
        let content = fs::read_to_string(&self.registry_file).await?;
        if content.trim().is_empty() {
            return Ok(());
        }
        
        let registry_data: HashMap<LoRAId, LoRALayer> = serde_json::from_str(&content)?;
        
        let mut layers = self.layers.write().await;
        let mut name_to_id = self.name_to_id.write().await;
        
        // Load layers and build name mapping
        for (lora_id, layer) in registry_data {
            name_to_id.insert(layer.name.clone(), lora_id.clone());
            layers.insert(lora_id, layer);
        }
        
        Ok(())
    }
    
    /// Save registry to disk
    async fn save_registry(&self) -> Result<()> {
        if self.registry_file.as_os_str().is_empty() {
            // In-memory only, don't save
            return Ok(());
        }
        
        let layers = self.layers.read().await;
        let content = serde_json::to_string_pretty(&*layers)?;
        fs::write(&self.registry_file, content).await?;
        Ok(())
    }
    
    /// Register a new LoRA layer
    pub async fn register(&self, layer: LoRALayer) -> Result<()> {
        let mut layers = self.layers.write().await;
        if layers.contains_key(&layer.id) {
            return Err(anyhow::anyhow!("LoRA layer {} already exists", layer.id));
        }
        
        let layer_id = layer.id.clone();
        let layer_name = layer.name.clone();
        layers.insert(layer.id.clone(), layer);
        
        // Initialize metrics
        let mut metrics = self.metrics.write().await;
        metrics.insert(layer_id.clone(), LayerMetrics::default());
        
        // Maintain name mapping for backward compatibility
        let mut name_to_id = self.name_to_id.write().await;
        name_to_id.insert(layer_name, layer_id);
        
        // Release locks before saving
        drop(layers);
        drop(metrics);
        drop(name_to_id);
        
        // Save to disk
        self.save_registry().await?;
        
        Ok(())
    }
    
    /// Unregister a LoRA layer by ID
    pub async fn unregister(&self, lora_id: &LoRAId) -> Result<()> {
        let mut layers = self.layers.write().await;
        let layer = layers.remove(lora_id)
            .ok_or_else(|| anyhow::anyhow!("LoRA layer {} not found", lora_id))?;
        
        let mut metrics = self.metrics.write().await;
        metrics.remove(lora_id);
        
        // Remove from name mapping
        let mut name_to_id = self.name_to_id.write().await;
        name_to_id.retain(|_, id| id != lora_id);
        
        // Release locks before saving
        drop(layers);
        drop(metrics);
        drop(name_to_id);
        
        // Save to disk
        self.save_registry().await?;
        
        Ok(())
    }
    
    /// Unregister a LoRA layer by name (legacy support)
    pub async fn unregister_by_name(&self, name: &str) -> Result<()> {
        let name_to_id = self.name_to_id.read().await;
        if let Some(lora_id) = name_to_id.get(name) {
            let lora_id = lora_id.clone();
            drop(name_to_id);
            self.unregister(&lora_id).await
        } else {
            Err(anyhow::anyhow!("LoRA layer with name {} not found", name))
        }
    }
    
    /// Get a LoRA layer by ID
    pub async fn get(&self, lora_id: &LoRAId) -> Result<LoRALayer> {
        let layers = self.layers.read().await;
        layers.get(lora_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("LoRA layer {} not found", lora_id))
    }
    
    /// Get a LoRA layer by name (legacy support)
    pub async fn get_by_name(&self, name: &str) -> Result<LoRALayer> {
        let name_to_id = self.name_to_id.read().await;
        if let Some(lora_id) = name_to_id.get(name) {
            let lora_id = lora_id.clone();
            drop(name_to_id);
            self.get(&lora_id).await
        } else {
            Err(anyhow::anyhow!("LoRA layer with name {} not found", name))
        }
    }
    
    /// Get a LoRA layer by ID or name (flexible lookup)
    pub async fn get_by_id_or_name(&self, identifier: &str) -> Result<LoRALayer> {
        // Try to parse as UUID first
        if let Ok(lora_id) = identifier.parse::<LoRAId>() {
            self.get(&lora_id).await
        } else {
            // Fall back to name lookup
            self.get_by_name(identifier).await
        }
    }
    
    /// List all LoRA layers
    pub async fn list_all(&self) -> Result<Vec<LoRALayer>> {
        let layers = self.layers.read().await;
        Ok(layers.values().cloned().collect())
    }
    
    /// Update layer metrics
    pub async fn update_metrics(
        &self,
        lora_id: &LoRAId,
        requests: u64,
        tokens: u64,
        latency_ms: f64,
    ) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        let layer_metrics = metrics.entry(lora_id.clone())
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
    pub async fn get_stats(&self, lora_id: &LoRAId) -> Result<crate::api::LoRAStats> {
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
        lora_id: &LoRAId,
        tokens_trained: u64,
    ) -> Result<()> {
        let mut layers = self.layers.write().await;
        let layer = layers.get_mut(lora_id)
            .ok_or_else(|| anyhow::anyhow!("LoRA layer {} not found", lora_id))?;
        
        layer.total_tokens_trained += tokens_trained;
        layer.updated_at = chrono::Utc::now().timestamp();
        
        // Release lock before saving
        drop(layers);
        
        // Save to disk
        self.save_registry().await?;
        
        Ok(())
    }
    
    /// List all registered adapter IDs
    pub async fn list_adapters(&self) -> Result<Vec<LoRAId>> {
        let layers = self.layers.read().await;
        Ok(layers.keys().cloned().collect())
    }
    
    /// List all registered adapter names (legacy support)
    pub async fn list_adapter_names(&self) -> Result<Vec<String>> {
        let name_to_id = self.name_to_id.read().await;
        Ok(name_to_id.keys().cloned().collect())
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