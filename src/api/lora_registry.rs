//! Registry for managing LoRA layers

use std::collections::HashMap;
use std::sync::Arc;
use std::path::PathBuf;
use tokio::sync::RwLock;
use tokio::fs;
use serde::{Deserialize, Serialize};
use anyhow::Result;
use uuid::Uuid;
use crate::api::model_storage::{ComposedModelId, ComposedModelMetadata, ModelId};

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
        let _layer = layers.remove(lora_id)
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
    
    /// Batch update training progress for multiple adapters (optimization)
    pub async fn batch_update_training_progress(
        &self,
        updates: &[(LoRAId, u64)]
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        
        let mut layers = self.layers.write().await;
        let now = chrono::Utc::now().timestamp();
        
        for (lora_id, tokens_trained) in updates {
            if let Some(layer) = layers.get_mut(lora_id) {
                layer.total_tokens_trained += tokens_trained;
                layer.updated_at = now;
            }
        }
        
        // Release lock before saving
        drop(layers);
        
        // Save to disk once for all updates
        self.save_registry().await?;
        
        Ok(())
    }
    
    /// Get storage path for a specific LoRA adapter's weights
    pub fn get_lora_storage_path(&self, lora_id: &LoRAId) -> PathBuf {
        self.base_dir.join("weights").join(format!("{}.vdb", lora_id))
    }
    
    /// Validate LoRA configuration
    pub fn validate_config(config: &crate::api::LoRAConfig) -> Result<()> {
        if config.rank == 0 {
            return Err(anyhow::anyhow!("LoRA rank must be greater than 0"));
        }
        
        if config.alpha <= 0.0 {
            return Err(anyhow::anyhow!("LoRA alpha must be positive"));
        }
        
        if config.sparsity_ratio < 0.0 || config.sparsity_ratio >= 1.0 {
            return Err(anyhow::anyhow!("Sparsity ratio must be in range [0.0, 1.0)"));
        }
        
        if config.dropout < 0.0 || config.dropout >= 1.0 {
            return Err(anyhow::anyhow!("Dropout must be in range [0.0, 1.0)"));
        }
        
        if config.target_modules.is_empty() {
            return Err(anyhow::anyhow!("At least one target module must be specified"));
        }
        
        Ok(())
    }
    
    /// Register a new LoRA layer with validation
    pub async fn register_validated(&self, mut layer: LoRALayer) -> Result<LoRAId> {
        // Validate configuration
        Self::validate_config(&layer.config)?;
        
        // Ensure unique name
        let name_to_id = self.name_to_id.read().await;
        if name_to_id.contains_key(&layer.name) {
            return Err(anyhow::anyhow!("LoRA layer with name '{}' already exists", layer.name));
        }
        drop(name_to_id);
        
        // Generate new UUID if not set
        if layer.id.0.is_nil() {
            layer.id = LoRAId::new();
        }
        
        let layer_id = layer.id.clone();
        
        // Use the existing register method
        self.register(layer).await?;
        
        Ok(layer_id)
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

impl LoRARegistry {
    /// Create a composed model (base + LoRA stack)
    pub async fn create_composed_model(
        &self,
        name: String,
        base_model_id: ModelId,
        lora_ids: Vec<String>,
    ) -> Result<ComposedModelId> {
        let composed_id = ComposedModelId::new();
        
        // Verify all LoRAs exist
        for lora_id_str in &lora_ids {
            let lora_id: LoRAId = lora_id_str.parse()
                .map_err(|_| anyhow::anyhow!("Invalid LoRA ID: {}", lora_id_str))?;
            self.get(&lora_id).await
                .map_err(|_| anyhow::anyhow!("LoRA not found: {}", lora_id_str))?;
        }
        
        let composed_metadata = ComposedModelMetadata {
            composed_id: composed_id.clone(),
            name,
            base_model_id,
            lora_stack: lora_ids,
            created_at: chrono::Utc::now().timestamp(),
            last_used: chrono::Utc::now().timestamp(),
        };
        
        // Store composed model metadata (simplified for now)
        // In a full implementation, this would use proper storage
        tracing::info!("Created composed model: {} with {} LoRA layers", 
                      composed_id, composed_metadata.lora_stack.len());
        
        Ok(composed_id)
    }
}

use chrono;