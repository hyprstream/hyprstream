//! LoRA Storage Manager - Bridges LoRA Registry and VDB Storage
//! 
//! This module integrates the UUID-based LoRA Registry with the VDB-backed
//! AdapterStore to provide unified weight persistence, loading, and management.

use std::collections::HashMap;
use std::sync::Arc;
use std::path::PathBuf;
use tokio::sync::RwLock;
use anyhow::Result;

use crate::api::lora_registry::{LoRARegistry, LoRAId, LoRALayer};
use crate::storage::vdb::adapter_store::{AdapterStore, AdapterMetadata};
use crate::storage::vdb::grid::{SparseWeights, Coordinate3D};
use crate::adapters::sparse_lora::{SparseLoRAAdapter, SparseLoRAConfig};

/// Unified LoRA storage manager that bridges registry and VDB storage
pub struct LoRAStorageManager {
    /// LoRA registry for metadata and UUID management
    registry: Arc<LoRARegistry>,
    
    /// VDB-backed adapter store for weight persistence
    adapter_store: Arc<AdapterStore>,
    
    /// UUID to domain mapping for VDB integration
    uuid_to_domain: Arc<RwLock<HashMap<LoRAId, String>>>,
    
    /// Domain to UUID reverse mapping
    domain_to_uuid: Arc<RwLock<HashMap<String, LoRAId>>>,
    
    /// Configuration
    config: LoRAStorageConfig,
}

/// Configuration for LoRA storage management
#[derive(Debug, Clone)]
pub struct LoRAStorageConfig {
    /// Enable automatic weight persistence after updates
    pub auto_persist: bool,
    
    /// Batch size for streaming updates before persistence
    pub batch_update_threshold: usize,
    
    /// Enable neural compression for long-term storage
    pub enable_neural_compression: bool,
    
    /// Maximum number of adapters to keep in memory
    pub max_memory_adapters: usize,
}

impl Default for LoRAStorageConfig {
    fn default() -> Self {
        Self {
            auto_persist: true,
            batch_update_threshold: 100,
            enable_neural_compression: true,
            max_memory_adapters: 50,
        }
    }
}

impl LoRAStorageManager {
    /// Create new LoRA storage manager
    pub async fn new(
        registry_dir: PathBuf,
        vdb_storage_dir: PathBuf,
        config: Option<LoRAStorageConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        
        // Initialize registry
        let registry = Arc::new(LoRARegistry::new(registry_dir).await?);
        
        // Initialize adapter store with VDB backend
        let adapter_store = Arc::new(AdapterStore::new(vdb_storage_dir, None).await?);
        
        let manager = Self {
            registry,
            adapter_store,
            uuid_to_domain: Arc::new(RwLock::new(HashMap::new())),
            domain_to_uuid: Arc::new(RwLock::new(HashMap::new())),
            config,
        };
        
        // Build initial mapping from existing data
        manager.rebuild_uuid_domain_mapping().await?;
        
        Ok(manager)
    }
    
    /// Register a new LoRA adapter with automatic VDB storage setup
    pub async fn register_adapter(
        &self,
        name: String,
        base_model: String,
        lora_config: crate::api::LoRAConfig,
        initial_weights: Option<SparseLoRAAdapter>,
    ) -> Result<LoRAId> {
        // Create LoRA layer in registry
        let layer = LoRALayer::new(
            name.clone(),
            base_model,
            lora_config.clone(),
            lora_config.sparsity_ratio,
        );
        
        let lora_id = layer.id.clone();
        self.registry.register_validated(layer).await?;
        
        // Create corresponding VDB adapter
        let domain = self.generate_domain_name(&name, &lora_id);
        
        // Convert LoRA config to adapter metadata
        let adapter_metadata = AdapterMetadata {
            domain: domain.clone(),
            sparsity: lora_config.sparsity_ratio,
            active_parameters: ((lora_config.rank * 1536 * 2) as f32 * (1.0 - lora_config.sparsity_ratio)) as usize,
            total_parameters: lora_config.rank * 1536 * 2, // rank * hidden_dim * 2 (A and B matrices)
            learning_rate: 1e-4, // Default learning rate
            adapter_type: "sparse_lora".to_string(),
            ..Default::default()
        };
        
        // Initialize weights in VDB
        let weights = if let Some(initial_adapter) = initial_weights {
            initial_adapter.to_vdb_weights().await
        } else {
            // Create default sparse weights with proper dimensions
            let mut weights = SparseWeights::new(vec![
                lora_config.rank * 1536,  // Combined A matrix dimensions  
                1536                       // Output dimension
            ]);
            
            // Initialize with small sparse random values (1% active)
            self.initialize_sparse_weights(&mut weights, lora_config.sparsity_ratio);
            weights
        };
        
        let _adapter_id = self.adapter_store.store_adapter(&domain, &weights, adapter_metadata).await?;
        
        // Update UUID-domain mappings
        {
            let mut uuid_to_domain = self.uuid_to_domain.write().await;
            let mut domain_to_uuid = self.domain_to_uuid.write().await;
            
            uuid_to_domain.insert(lora_id.clone(), domain.clone());
            domain_to_uuid.insert(domain, lora_id.clone());
        }
        
        println!("📚 Registered LoRA adapter {} with VDB storage", lora_id);
        
        Ok(lora_id)
    }
    
    /// Load LoRA adapter weights from VDB storage
    pub async fn load_adapter_weights(&self, lora_id: &LoRAId) -> Result<SparseLoRAAdapter> {
        // Get domain from UUID
        let domain = {
            let uuid_to_domain = self.uuid_to_domain.read().await;
            uuid_to_domain.get(lora_id)
                .ok_or_else(|| anyhow::anyhow!("Domain not found for LoRA ID: {}", lora_id))?
                .clone()
        };
        
        // Load from adapter store
        let (weights, _metadata) = self.adapter_store.load_adapter(&domain).await?
            .ok_or_else(|| anyhow::anyhow!("Weights not found for domain: {}", domain))?;
        
        // Get LoRA layer configuration
        let layer = self.registry.get(lora_id).await?;
        
        // Convert to sparse LoRA config
        let sparse_config = SparseLoRAConfig {
            in_features: 1536,  // Model hidden dimension
            out_features: 1536,
            rank: layer.config.rank,
            sparsity: layer.config.sparsity_ratio,
            learning_rate: 1e-4,
            alpha: layer.config.alpha,
            target_modules: layer.config.target_modules.clone(),
            ..Default::default()
        };
        
        // Create sparse LoRA adapter and load weights
        let adapter = SparseLoRAAdapter::new(sparse_config);
        adapter.from_vdb_weights(&weights).await;
        
        println!("📚 Loaded LoRA adapter {} from VDB storage", lora_id);
        
        Ok(adapter)
    }
    
    /// Save LoRA adapter weights to VDB storage
    pub async fn save_adapter_weights(
        &self,
        lora_id: &LoRAId,
        adapter: &SparseLoRAAdapter,
        update_training_stats: bool,
    ) -> Result<()> {
        // Get domain from UUID
        let domain = {
            let uuid_to_domain = self.uuid_to_domain.read().await;
            uuid_to_domain.get(lora_id)
                .ok_or_else(|| anyhow::anyhow!("Domain not found for LoRA ID: {}", lora_id))?
                .clone()
        };
        
        // Convert adapter to VDB weights
        let weights = adapter.to_vdb_weights().await;
        
        // Update adapter metadata if requested
        if update_training_stats {
            let stats = adapter.get_stats().await;
            
            // Update training progress in registry
            self.registry.update_training_progress(lora_id, stats.forward_passes).await?;
            
            // Create metadata update for VDB
            let metadata = AdapterMetadata {
                domain: domain.clone(),
                last_updated: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                sparsity: weights.sparsity(),
                active_parameters: weights.active_count(),
                total_parameters: weights.dense_size(),
                training_steps: stats.updates_applied,
                ..Default::default()
            };
            
            // Store updated weights and metadata
            self.adapter_store.store_adapter(&domain, &weights, metadata).await?;
        } else {
            // Simple weight update without metadata changes
            let sparse_updates = self.weights_to_sparse_updates(&weights);
            self.adapter_store.update_adapter_weights(&domain, &sparse_updates).await?;
        }
        
        println!("📚 Saved LoRA adapter {} to VDB storage", lora_id);
        
        Ok(())
    }
    
    /// Apply streaming weight updates (for real-time training)
    pub async fn stream_weight_updates(
        &self,
        lora_id: &LoRAId,
        updates: &HashMap<Coordinate3D, f32>,
    ) -> Result<()> {
        let domain = {
            let uuid_to_domain = self.uuid_to_domain.read().await;
            uuid_to_domain.get(lora_id)
                .ok_or_else(|| anyhow::anyhow!("Domain not found for LoRA ID: {}", lora_id))?
                .clone()
        };
        
        // Apply updates directly to VDB storage
        self.adapter_store.update_adapter_weights(&domain, updates).await?;
        
        // Update registry metrics if auto-persist is enabled
        if self.config.auto_persist && updates.len() > 0 {
            self.registry.update_metrics(
                lora_id,
                1, // requests
                updates.len() as u64, // tokens (approximate)
                1.0, // latency (placeholder)
            ).await?;
        }
        
        Ok(())
    }
    
    /// List all available LoRA adapters with their storage info
    pub async fn list_all_adapters(&self) -> Result<Vec<LoRAAdapterInfo>> {
        let registry_layers = self.registry.list_all().await?;
        let mut adapter_infos = Vec::new();
        
        for layer in registry_layers {
            let domain_opt = {
                let uuid_to_domain = self.uuid_to_domain.read().await;
                uuid_to_domain.get(&layer.id).cloned()
            };
            
            let storage_info = if let Some(domain) = domain_opt {
                // Get VDB storage info
                match self.adapter_store.load_adapter(&domain).await {
                    Ok(Some((weights, metadata))) => Some(LoRAStorageInfo {
                        vdb_domain: domain,
                        weights_size_bytes: weights.memory_usage(),
                        active_parameters: weights.active_count(),
                        sparsity_ratio: weights.sparsity(),
                        last_updated: metadata.last_updated,
                        training_steps: metadata.training_steps,
                    }),
                    _ => None,
                }
            } else {
                None
            };
            
            adapter_infos.push(LoRAAdapterInfo {
                layer,
                storage_info,
            });
        }
        
        Ok(adapter_infos)
    }
    
    /// Delete LoRA adapter and its VDB storage
    pub async fn delete_adapter(&self, lora_id: &LoRAId) -> Result<()> {
        // Get domain for VDB cleanup
        let domain_opt = {
            let uuid_to_domain = self.uuid_to_domain.read().await;
            uuid_to_domain.get(lora_id).cloned()
        };
        
        // Delete from registry first
        self.registry.unregister(lora_id).await?;
        
        // Delete from VDB storage if domain exists
        if let Some(domain) = domain_opt {
            self.adapter_store.delete_adapter(&domain).await?;
            
            // Remove from mappings
            let mut uuid_to_domain = self.uuid_to_domain.write().await;
            let mut domain_to_uuid = self.domain_to_uuid.write().await;
            
            uuid_to_domain.remove(lora_id);
            domain_to_uuid.remove(&domain);
        }
        
        println!("📚 Deleted LoRA adapter {} and its VDB storage", lora_id);
        
        Ok(())
    }
    
    /// Get comprehensive statistics for a LoRA adapter
    pub async fn get_adapter_statistics(&self, lora_id: &LoRAId) -> Result<LoRAFullStats> {
        // Get registry stats
        let registry_stats = self.registry.get_stats(lora_id).await?;
        
        // Get VDB storage info
        let domain = {
            let uuid_to_domain = self.uuid_to_domain.read().await;
            uuid_to_domain.get(lora_id)
                .ok_or_else(|| anyhow::anyhow!("Domain not found for LoRA ID: {}", lora_id))?
                .clone()
        };
        
        let (weights, metadata) = self.adapter_store.load_adapter(&domain).await?
            .ok_or_else(|| anyhow::anyhow!("Storage data not found for domain: {}", domain))?;
        
        Ok(LoRAFullStats {
            registry_stats,
            storage_size_bytes: weights.memory_usage(),
            active_voxels: weights.active_count(),
            total_voxels: weights.dense_size(),
            actual_sparsity: weights.sparsity(),
            vdb_training_steps: metadata.training_steps,
            last_vdb_update: metadata.last_updated,
        })
    }
    
    /// Rebuild the UUID-domain mapping from existing data and migrate legacy adapters
    async fn rebuild_uuid_domain_mapping(&self) -> Result<()> {
        // Get all registry adapters
        let registry_adapters = self.registry.list_all().await?;
        
        // Get all VDB adapters
        let vdb_adapters = self.adapter_store.list_adapters().await;
        
        let mut uuid_to_domain = self.uuid_to_domain.write().await;
        let mut domain_to_uuid = self.domain_to_uuid.write().await;
        
        // First try to match existing VDB adapters
        for layer in &registry_adapters {
            let mut found_match = false;
            
            // Try to find matching VDB adapter by name pattern
            for vdb_adapter in &vdb_adapters {
                if vdb_adapter.domain.contains(&layer.name.replace(' ', "_")) {
                    uuid_to_domain.insert(layer.id.clone(), vdb_adapter.domain.clone());
                    domain_to_uuid.insert(vdb_adapter.domain.clone(), layer.id.clone());
                    found_match = true;
                    break;
                }
            }
            
            // If no VDB adapter found, migrate this legacy adapter to VDB storage
            if !found_match {
                println!("🔄 Migrating legacy adapter to VDB: {} ({})", layer.name, layer.id);
                
                // Generate domain name for VDB storage
                let domain = self.generate_domain_name(&layer.name, &layer.id);
                
                // Create default sparse weights for this adapter
                let mut weights = SparseWeights::new(vec![
                    layer.config.rank * 1536,  // Combined A matrix dimensions  
                    1536                        // Output dimension
                ]);
                
                // Initialize with small sparse random values matching the configuration
                self.initialize_sparse_weights(&mut weights, layer.config.sparsity_ratio);
                
                // Convert LoRA config to adapter metadata
                let adapter_metadata = AdapterMetadata {
                    domain: domain.clone(),
                    sparsity: layer.config.sparsity_ratio,
                    active_parameters: ((layer.config.rank * 1536 * 2) as f32 * (1.0 - layer.config.sparsity_ratio)) as usize,
                    total_parameters: layer.config.rank * 1536 * 2,
                    learning_rate: 1e-4,
                    adapter_type: "sparse_lora".to_string(),
                    created_at: layer.created_at as u64,
                    last_updated: layer.updated_at as u64,
                    training_steps: layer.total_tokens_trained,
                    ..Default::default()
                };
                
                // Store in VDB
                match self.adapter_store.store_adapter(&domain, &weights, adapter_metadata).await {
                    Ok(_adapter_id) => {
                        // Add to mappings
                        uuid_to_domain.insert(layer.id.clone(), domain.clone());
                        domain_to_uuid.insert(domain, layer.id.clone());
                        println!("✅ Successfully migrated adapter: {}", layer.name);
                    }
                    Err(e) => {
                        eprintln!("❌ Failed to migrate adapter {}: {}", layer.name, e);
                    }
                }
            }
        }
        
        println!("📚 Rebuilt UUID-domain mapping: {} associations", uuid_to_domain.len());
        
        // If we found legacy adapters, this was likely the first run with the new system
        let migrated_count = registry_adapters.len() - vdb_adapters.len();
        if migrated_count > 0 {
            println!("🎉 Migration complete! {} legacy adapters migrated to VDB storage", migrated_count);
        }
        
        Ok(())
    }
    
    /// Generate domain name for VDB storage from LoRA name and UUID
    fn generate_domain_name(&self, name: &str, lora_id: &LoRAId) -> String {
        format!("lora_{}_{}", 
            name.replace(' ', "_").to_lowercase(),
            lora_id.to_string().split('-').next().unwrap_or("unknown"))
    }
    
    /// Initialize sparse weights with random values
    fn initialize_sparse_weights(&self, weights: &mut SparseWeights, sparsity_ratio: f32) {
        use rand::Rng;
        
        let total_elements = weights.dense_size();
        let active_count = ((total_elements as f32) * (1.0 - sparsity_ratio)) as usize;
        
        let mut rng = rand::thread_rng();
        
        for i in 0..active_count {
            let index = (i * total_elements / active_count.max(1)) % total_elements;
            let value = (rng.gen::<f32>() - 0.5) * 0.02; // Small random initialization
            weights.set(index, value);
        }
    }
    
    /// Convert SparseWeights to coordinate-based updates
    fn weights_to_sparse_updates(&self, weights: &SparseWeights) -> HashMap<Coordinate3D, f32> {
        let mut updates = HashMap::new();
        let shape = weights.shape();
        
        for (linear_idx, value) in weights.active_iter() {
            // Convert linear index to 3D coordinate for VDB
            let coord = if shape.len() >= 2 {
                let x = linear_idx % shape[1];
                let y = linear_idx / shape[1];
                Coordinate3D::new(x as i32, y as i32, 0)
            } else {
                Coordinate3D::new(linear_idx as i32, 0, 0)
            };
            
            updates.insert(coord, value);
        }
        
        updates
    }
}

/// Combined information about a LoRA adapter and its storage
#[derive(Debug, Clone)]
pub struct LoRAAdapterInfo {
    pub layer: LoRALayer,
    pub storage_info: Option<LoRAStorageInfo>,
}

/// Storage information for a LoRA adapter
#[derive(Debug, Clone)]
pub struct LoRAStorageInfo {
    pub vdb_domain: String,
    pub weights_size_bytes: usize,
    pub active_parameters: usize,
    pub sparsity_ratio: f32,
    pub last_updated: u64,
    pub training_steps: u64,
}

/// Comprehensive statistics combining registry and storage data
#[derive(Debug, Clone)]
pub struct LoRAFullStats {
    pub registry_stats: crate::api::LoRAStats,
    pub storage_size_bytes: usize,
    pub active_voxels: usize,
    pub total_voxels: usize,
    pub actual_sparsity: f32,
    pub vdb_training_steps: u64,
    pub last_vdb_update: u64,
}


#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_lora_storage_manager_lifecycle() {
        let temp_dir = tempdir().unwrap();
        let registry_dir = temp_dir.path().join("registry");
        let vdb_dir = temp_dir.path().join("vdb");
        
        let manager = LoRAStorageManager::new(
            registry_dir,
            vdb_dir,
            None,
        ).await.unwrap();
        
        // Register new adapter
        let lora_config = crate::api::LoRAConfig {
            rank: 16,
            alpha: 16.0,
            sparsity_ratio: 0.99,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            ..Default::default()
        };
        
        let lora_id = manager.register_adapter(
            "test_adapter".to_string(),
            "qwen/qwen3-1.7b".to_string(),
            lora_config,
            None,
        ).await.unwrap();
        
        // Load the adapter
        let loaded_adapter = manager.load_adapter_weights(&lora_id).await.unwrap();
        
        // Verify adapter properties
        assert_eq!(loaded_adapter.get_config().rank, 16);
        assert!(loaded_adapter.get_config().sparsity > 0.98);
        
        // Save modified adapter
        manager.save_adapter_weights(&lora_id, &loaded_adapter, true).await.unwrap();
        
        // Get statistics
        let stats = manager.get_adapter_statistics(&lora_id).await.unwrap();
        assert!(stats.actual_sparsity > 0.98);
        
        // List all adapters
        let all_adapters = manager.list_all_adapters().await.unwrap();
        assert_eq!(all_adapters.len(), 1);
        assert_eq!(all_adapters[0].layer.name, "test_adapter");
        
        // Clean up
        manager.delete_adapter(&lora_id).await.unwrap();
    }
    
    #[tokio::test] 
    async fn test_streaming_updates() {
        let temp_dir = tempdir().unwrap();
        let registry_dir = temp_dir.path().join("registry");
        let vdb_dir = temp_dir.path().join("vdb");
        
        let manager = LoRAStorageManager::new(
            registry_dir,
            vdb_dir,
            None,
        ).await.unwrap();
        
        // Register adapter
        let lora_config = crate::api::LoRAConfig::default();
        let lora_id = manager.register_adapter(
            "streaming_test".to_string(),
            "test_model".to_string(),
            lora_config,
            None,
        ).await.unwrap();
        
        // Apply streaming updates
        let mut updates = HashMap::new();
        updates.insert(Coordinate3D::new(10, 20, 0), 0.1);
        updates.insert(Coordinate3D::new(30, 40, 0), -0.05);
        
        manager.stream_weight_updates(&lora_id, &updates).await.unwrap();
        
        // Verify updates were applied
        let stats = manager.get_adapter_statistics(&lora_id).await.unwrap();
        assert!(stats.vdb_training_steps > 0);
    }
}