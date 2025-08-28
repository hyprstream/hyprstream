//! High-level adapter storage interface using VDB backend

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::path::Path;
use std::io::Result;
use serde::{Serialize, Deserialize};

use crate::storage::vdb::{VDBStorage, VDBConfig};
use crate::storage::vdb::grid::{SparseWeights, Coordinate3D};

/// Metadata for stored adapters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterMetadata {
    pub domain: String,
    pub created_at: u64,
    pub last_updated: u64,
    pub version: u32,
    pub sparsity: f32,
    pub active_parameters: usize,
    pub total_parameters: usize,
    pub training_steps: u64,
    pub learning_rate: f32,
    pub adapter_type: String, // "lora", "ia3", etc.
}

impl Default for AdapterMetadata {
    fn default() -> Self {
        Self {
            domain: "default".to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            last_updated: 0,
            version: 1,
            sparsity: 0.99,
            active_parameters: 0,
            total_parameters: 0,
            training_steps: 0,
            learning_rate: 1e-4,
            adapter_type: "lora".to_string(),
        }
    }
}

/// High-level adapter storage system
pub struct AdapterStore {
    /// VDB backend
    vdb_storage: Arc<VDBStorage>,
    
    /// Adapter metadata cache
    metadata_cache: Arc<RwLock<HashMap<String, AdapterMetadata>>>,
    
    /// Domain to adapter ID mapping
    domain_mapping: Arc<RwLock<HashMap<String, String>>>,
    
    /// Configuration
    config: AdapterStoreConfig,
}

#[derive(Debug, Clone)]
pub struct AdapterStoreConfig {
    pub max_cached_adapters: usize,
    pub auto_checkpoint_steps: u64,
    pub compression_enabled: bool,
    pub enable_versioning: bool,
}

impl Default for AdapterStoreConfig {
    fn default() -> Self {
        Self {
            max_cached_adapters: 100,
            auto_checkpoint_steps: 1000,
            compression_enabled: true,
            enable_versioning: true,
        }
    }
}

impl AdapterStore {
    /// Create new adapter store
    pub async fn new<P: AsRef<Path>>(
        storage_path: P,
        config: Option<AdapterStoreConfig>,
    ) -> Result<Self> {
        let config = config.unwrap_or_default();
        
        // Initialize VDB storage
        let vdb_config = VDBConfig {
            storage_path: storage_path.as_ref().to_path_buf(),
            compression: if config.compression_enabled { "lz4".to_string() } else { "none".to_string() },
            cache_size_mb: 1000,
            background_value: 0.0,
        };
        
        let vdb_storage = Arc::new(VDBStorage::new(vdb_config).await?);
        
        Ok(Self {
            vdb_storage,
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            domain_mapping: Arc::new(RwLock::new(HashMap::new())),
            config,
        })
    }
    
    /// Store adapter weights for a domain
    pub async fn store_adapter(
        &self,
        domain: &str,
        weights: &SparseWeights,
        metadata: AdapterMetadata,
    ) -> Result<String> {
        let adapter_id = self.generate_adapter_id(domain, metadata.version);
        
        // Store weights in VDB
        self.vdb_storage.store_adapter(&adapter_id, weights).await?;
        
        // Update metadata
        {
            let mut cache = self.metadata_cache.write().await;
            cache.insert(adapter_id.clone(), metadata.clone());
        }
        
        // Update domain mapping
        {
            let mut mapping = self.domain_mapping.write().await;
            mapping.insert(domain.to_string(), adapter_id.clone());
        }
        
        println!("Stored adapter for domain '{}' with ID: {}", domain, adapter_id);
        Ok(adapter_id)
    }
    
    /// Load adapter weights by domain
    pub async fn load_adapter(&self, domain: &str) -> Result<Option<(SparseWeights, AdapterMetadata)>> {
        // Get adapter ID from domain
        let adapter_id = {
            let mapping = self.domain_mapping.read().await;
            mapping.get(domain).cloned()
        };
        
        let adapter_id = match adapter_id {
            Some(id) => id,
            None => return Ok(None),
        };
        
        // Load weights from VDB
        let weights = self.vdb_storage.load_adapter(&adapter_id).await?;
        
        // Get metadata
        let metadata = {
            let cache = self.metadata_cache.read().await;
            cache.get(&adapter_id).cloned().unwrap_or_default()
        };
        
        Ok(Some((weights, metadata)))
    }
    
    /// Update adapter weights (streaming updates)
    pub async fn update_adapter_weights(
        &self,
        domain: &str,
        sparse_updates: &HashMap<Coordinate3D, f32>,
    ) -> Result<()> {
        // Get adapter ID
        let adapter_id = {
            let mapping = self.domain_mapping.read().await;
            mapping.get(domain).cloned()
        };
        
        let adapter_id = match adapter_id {
            Some(id) => id,
            None => {
                // Create new adapter if doesn't exist
                let weights = SparseWeights::new(vec![1536, 1536]); // Default Qwen3 shape
                let metadata = AdapterMetadata {
                    domain: domain.to_string(),
                    ..Default::default()
                };
                return Ok(self.store_adapter(domain, &weights, metadata).await.map(|_| ())?);
            }
        };
        
        // Apply updates to VDB
        self.vdb_storage.update_weights(&adapter_id, sparse_updates).await?;
        
        // Update metadata
        {
            let mut cache = self.metadata_cache.write().await;
            if let Some(metadata) = cache.get_mut(&adapter_id) {
                metadata.last_updated = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                metadata.training_steps += 1;
                
                // Auto-checkpoint if needed
                if metadata.training_steps % self.config.auto_checkpoint_steps == 0 {
                    self.create_checkpoint(&adapter_id).await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Create new adapter for domain
    pub async fn create_adapter(
        &self,
        domain: &str,
        adapter_type: &str,
        config: &AdapterCreateConfig,
    ) -> Result<String> {
        let shape = match adapter_type {
            "lora" => vec![config.hidden_size, config.rank],
            "ia3" => vec![config.hidden_size],
            _ => return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Unsupported adapter type: {}", adapter_type)
            )),
        };
        
        // Initialize sparse weights
        let mut weights = SparseWeights::new(shape);
        
        // Initialize with small random values (sparse)
        let active_count = (weights.dense_size() as f64 * (1.0 - config.target_sparsity)) as usize;
        let mut rng = rand::thread_rng();
        for i in 0..active_count {
            let index = (i * weights.dense_size() / active_count) % weights.dense_size();
            weights.set(index, (rng.gen::<f32>() - 0.5) * 0.02);
        }
        
        let metadata = AdapterMetadata {
            domain: domain.to_string(),
            sparsity: config.target_sparsity as f32,
            active_parameters: weights.active_count(),
            total_parameters: weights.dense_size(),
            learning_rate: config.learning_rate as f32,
            adapter_type: adapter_type.to_string(),
            ..Default::default()
        };
        
        self.store_adapter(domain, &weights, metadata).await
    }
    
    /// List all available adapters
    pub async fn list_adapters(&self) -> Vec<AdapterInfo> {
        let cache = self.metadata_cache.read().await;
        let mapping = self.domain_mapping.read().await;
        
        let mut adapters = Vec::new();
        for (domain, adapter_id) in mapping.iter() {
            if let Some(metadata) = cache.get(adapter_id) {
                adapters.push(AdapterInfo {
                    domain: domain.clone(),
                    adapter_id: adapter_id.clone(),
                    metadata: metadata.clone(),
                });
            }
        }
        
        adapters.sort_by(|a, b| a.domain.cmp(&b.domain));
        adapters
    }
    
    /// Delete adapter
    pub async fn delete_adapter(&self, domain: &str) -> Result<bool> {
        // Get adapter ID
        let adapter_id = {
            let mut mapping = self.domain_mapping.write().await;
            mapping.remove(domain)
        };
        
        if let Some(adapter_id) = adapter_id {
            // Remove from metadata cache
            {
                let mut cache = self.metadata_cache.write().await;
                cache.remove(&adapter_id);
            }
            
            // TODO: Remove from VDB storage (implement cleanup)
            // For now, just mark as deleted in metadata
            
            println!("Deleted adapter for domain: {}", domain);
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Create checkpoint of current adapter state
    async fn create_checkpoint(&self, adapter_id: &str) -> Result<()> {
        if !self.config.enable_versioning {
            return Ok(());
        }
        
        // Load current state
        let weights = self.vdb_storage.load_adapter(adapter_id).await?;
        
        // Create versioned backup
        let checkpoint_id = format!("{}_checkpoint_{}", adapter_id, 
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs());
        
        self.vdb_storage.store_adapter(&checkpoint_id, &weights).await?;
        
        println!("Created checkpoint: {}", checkpoint_id);
        Ok(())
    }
    
    fn generate_adapter_id(&self, domain: &str, version: u32) -> String {
        format!("{}_{}_v{}", domain.replace(' ', "_"), 
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(), 
            version)
    }
}

/// Configuration for creating new adapters
#[derive(Debug, Clone)]
pub struct AdapterCreateConfig {
    pub hidden_size: usize,
    pub rank: usize, // For LoRA
    pub target_sparsity: f64,
    pub learning_rate: f32,
}

impl Default for AdapterCreateConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1536, // Qwen3-1.7B
            rank: 16,
            target_sparsity: 0.99,
            learning_rate: 1e-4,
        }
    }
}

/// Information about stored adapters
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    pub domain: String,
    pub adapter_id: String,
    pub metadata: AdapterMetadata,
}

// Use external rand crate instead of custom implementation
use rand::Rng;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_adapter_store_basic() {
        let temp_dir = tempdir().unwrap();
        let store = AdapterStore::new(temp_dir.path(), None).await.unwrap();
        
        // Create test adapter
        let adapter_id = store.create_adapter(
            "test_domain",
            "lora",
            &AdapterCreateConfig::default()
        ).await.unwrap();
        
        assert!(!adapter_id.is_empty());
        
        // Load adapter
        let loaded = store.load_adapter("test_domain").await.unwrap();
        assert!(loaded.is_some());
        
        let (weights, metadata) = loaded.unwrap();
        assert_eq!(metadata.domain, "test_domain");
        assert!(weights.sparsity() > 0.98); // Should be very sparse
    }
    
    #[tokio::test]
    async fn test_streaming_updates() {
        let temp_dir = tempdir().unwrap();
        let store = AdapterStore::new(temp_dir.path(), None).await.unwrap();
        
        // Create adapter
        store.create_adapter("streaming_test", "lora", &AdapterCreateConfig::default())
            .await.unwrap();
        
        // Simulate streaming updates
        let mut updates = HashMap::new();
        updates.insert(Coordinate3D::new(100, 5, 0), 0.1);
        updates.insert(Coordinate3D::new(200, 10, 0), -0.05);
        
        store.update_adapter_weights("streaming_test", &updates).await.unwrap();
        
        // Verify updates
        let (updated_weights, metadata) = store.load_adapter("streaming_test")
            .await.unwrap().unwrap();
        
        assert_eq!(metadata.training_steps, 1);
        assert!(updated_weights.active_count() >= 2);
    }
    
    #[tokio::test]
    async fn test_multi_domain() {
        let temp_dir = tempdir().unwrap();
        let store = AdapterStore::new(temp_dir.path(), None).await.unwrap();
        
        // Create multiple domain adapters
        let domains = ["medical", "legal", "finance"];
        for domain in &domains {
            store.create_adapter(domain, "lora", &AdapterCreateConfig::default())
                .await.unwrap();
        }
        
        // List all adapters
        let adapters = store.list_adapters().await;
        assert_eq!(adapters.len(), 3);
        
        // Verify each domain
        for domain in &domains {
            let loaded = store.load_adapter(domain).await.unwrap();
            assert!(loaded.is_some());
        }
    }
}