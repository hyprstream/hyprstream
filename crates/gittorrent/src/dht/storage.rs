//! Custom storage backend for Git objects in Kademlia DHT

use libp2p::kad::store::{RecordStore, Result as StoreResult};
use libp2p::kad::{ProviderRecord, Record, RecordKey};
use libp2p::PeerId;
use std::collections::HashMap;
use std::borrow::Cow;

/// Custom record store for Git objects
pub struct GitObjectStore {
    /// Store for Git object records
    records: HashMap<RecordKey, Record>,
    /// Store for provider records
    providers: HashMap<RecordKey, HashMap<PeerId, ProviderRecord>>,
    /// Repository to objects mapping
    repository_objects: HashMap<String, Vec<crate::types::Sha256Hash>>,
    /// Object to repositories mapping
    object_repositories: HashMap<crate::types::Sha256Hash, Vec<String>>,
    /// Maximum number of records to store
    max_records: usize,
    /// Maximum number of providers per key
    max_providers_per_key: usize,
}

impl GitObjectStore {
    /// Create a new Git object store
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            providers: HashMap::new(),
            repository_objects: HashMap::new(),
            object_repositories: HashMap::new(),
            max_records: 10_000,
            max_providers_per_key: 100,
        }
    }

    /// Create with custom limits
    pub fn with_limits(max_records: usize, max_providers_per_key: usize) -> Self {
        Self {
            records: HashMap::new(),
            providers: HashMap::new(),
            repository_objects: HashMap::new(),
            object_repositories: HashMap::new(),
            max_records,
            max_providers_per_key,
        }
    }

    /// Add a repository mapping
    pub fn add_repository_mapping(&mut self,
        repo_name: &str,
        object_hashes: &[crate::types::Sha256Hash]
    ) {
        // Store repository -> objects mapping
        self.repository_objects.insert(repo_name.to_owned(), object_hashes.to_vec());

        // Store object -> repositories mapping
        for hash in object_hashes {
            self.object_repositories
                .entry(hash.clone())
                .or_default()
                .push(repo_name.to_owned());
        }

        tracing::debug!("Added repository mapping for {} with {} objects", repo_name, object_hashes.len());
    }

    /// Get all objects for a repository
    pub fn get_repository_objects(&self, repo_name: &str) -> Vec<crate::types::Sha256Hash> {
        self.repository_objects
            .get(repo_name)
            .cloned()
            .unwrap_or_default()
    }

    /// Get repositories that contain a specific object
    pub fn get_object_repositories(&self, hash: &crate::types::Sha256Hash) -> Vec<String> {
        self.object_repositories
            .get(hash)
            .cloned()
            .unwrap_or_default()
    }

    /// Remove repository mapping
    pub fn remove_repository_mapping(&mut self, repo_name: &str) {
        if let Some(object_hashes) = self.repository_objects.remove(repo_name) {
            // Remove from object -> repositories mapping
            for hash in object_hashes {
                if let Some(repos) = self.object_repositories.get_mut(&hash) {
                    repos.retain(|r| r != repo_name);
                    if repos.is_empty() {
                        self.object_repositories.remove(&hash);
                    }
                }
            }
        }
    }

    /// Get statistics about the store
    pub fn stats(&self) -> GitObjectStoreStats {
        GitObjectStoreStats {
            record_count: self.records.len(),
            provider_count: self.providers.values().map(std::collections::HashMap::len).sum(),
            total_keys: self.records.len() + self.providers.len(),
            repository_count: self.repository_objects.len(),
            unique_objects: self.object_repositories.len(),
        }
    }
}

impl Default for GitObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the Git object store
#[derive(Debug, Clone)]
pub struct GitObjectStoreStats {
    pub record_count: usize,
    pub provider_count: usize,
    pub total_keys: usize,
    pub repository_count: usize,
    pub unique_objects: usize,
}

impl RecordStore for GitObjectStore {
    type RecordsIter<'a> = std::vec::IntoIter<Cow<'a, Record>>;

    type ProvidedIter<'a> = std::vec::IntoIter<Cow<'a, ProviderRecord>>;

    fn get(&self, key: &RecordKey) -> Option<Cow<'_, Record>> {
        self.records.get(key).map(Cow::Borrowed)
    }

    fn put(&mut self, record: Record) -> StoreResult<()> {
        // Check if we're at capacity and need to remove old records
        if self.records.len() >= self.max_records && !self.records.contains_key(&record.key) {
            // Simple eviction: remove the first record (in practice, use LRU)
            if let Some(key_to_remove) = self.records.keys().next().cloned() {
                self.records.remove(&key_to_remove);
                tracing::debug!("Evicted record to make space: {:?}", key_to_remove);
            }
        }

        tracing::debug!("Storing Git object record: {:?}", record.key);
        self.records.insert(record.key.clone(), record);
        Ok(())
    }

    fn remove(&mut self, key: &RecordKey) {
        self.records.remove(key);
        tracing::debug!("Removed record: {:?}", key);
    }

    fn records(&self) -> Self::RecordsIter<'_> {
        self.records.values().map(Cow::Borrowed).collect::<Vec<_>>().into_iter()
    }

    fn add_provider(&mut self, record: ProviderRecord) -> StoreResult<()> {
        let providers = self.providers.entry(record.key.clone()).or_default();

        // Check provider limit per key
        if providers.len() >= self.max_providers_per_key && !providers.contains_key(&record.provider) {
            // Remove oldest provider (in practice, use LRU)
            if let Some(provider_to_remove) = providers.keys().next().cloned() {
                providers.remove(&provider_to_remove);
                tracing::debug!("Evicted provider to make space: {:?} for key {:?}",
                    provider_to_remove, record.key);
            }
        }

        tracing::debug!("Adding provider: {:?} for key: {:?}", record.provider, record.key);
        providers.insert(record.provider, record);
        Ok(())
    }

    fn providers(&self, key: &RecordKey) -> Vec<ProviderRecord> {
        self.providers
            .get(key)
            .map(|providers| providers.values().cloned().collect())
            .unwrap_or_default()
    }

    fn provided(&self) -> Self::ProvidedIter<'_> {
        self.providers
            .values()
            .flat_map(HashMap::values)
            .map(Cow::Borrowed)
            .collect::<Vec<_>>()
            .into_iter()
    }

    fn remove_provider(&mut self, key: &RecordKey, provider: &PeerId) {
        if let Some(providers) = self.providers.get_mut(key) {
            providers.remove(provider);
            if providers.is_empty() {
                self.providers.remove(key);
            }
            tracing::debug!("Removed provider: {:?} for key: {:?}", provider, key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libp2p::kad::RecordKey;

    #[test]
    fn test_record_store_basic_operations() -> Result<(), libp2p::kad::store::Error> {
        let mut store = GitObjectStore::new();

        // Test put and get
        let key = RecordKey::new(b"test-key");
        let record = Record {
            key: key.clone(),
            value: b"test-value".to_vec(),
            publisher: None,
            expires: None,
        };

        store.put(record.clone())?;
        if let Some(stored) = store.get(&key) {
            assert_eq!(stored.value, b"test-value");
        } else {
            panic!("Expected record not found");
        }

        // Test remove
        store.remove(&key);
        assert!(store.get(&key).is_none());
        Ok(())
    }

    #[test]
    fn test_provider_operations() -> Result<(), libp2p::kad::store::Error> {
        let mut store = GitObjectStore::new();
        let key = RecordKey::new(b"test-key");
        let peer_id = PeerId::random();

        let provider_record = ProviderRecord {
            key: key.clone(),
            provider: peer_id,
            expires: None,
            addresses: vec![],
        };

        // Add provider
        store.add_provider(provider_record.clone())?;
        let providers = store.providers(&key);
        assert_eq!(providers.len(), 1);
        assert_eq!(providers[0].provider, peer_id);

        // Remove provider
        store.remove_provider(&key, &peer_id);
        assert!(store.providers(&key).is_empty());
        Ok(())
    }

    #[test]
    fn test_capacity_limits() -> Result<(), libp2p::kad::store::Error> {
        let mut store = GitObjectStore::with_limits(2, 2);

        // Add records up to capacity
        for i in 0..3 {
            let key = RecordKey::new(&format!("key-{i}").as_bytes());
            let record = Record {
                key: key.clone(),
                value: format!("value-{i}").as_bytes().to_vec(),
                publisher: None,
                expires: None,
            };
            store.put(record)?;
        }

        // Should only have 2 records due to capacity limit
        assert_eq!(store.records.len(), 2);
        Ok(())
    }
}