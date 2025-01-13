use std::collections::HashMap;
use arrow::datatypes::Schema;
use hyprstream::storage::vector::VectorEmbedding;
use hyprstream::models::Model;
use hyprstream::storage::duckdb::DuckDbBackend;
use tempfile::tempdir;

// Database fixtures
pub struct TestDatabase {
    pub backend: DuckDbBackend,
    pub temp_dir: tempfile::TempDir,
}

impl TestDatabase {
    pub fn new() -> Self {
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let backend = DuckDbBackend::new(
            db_path.to_str().unwrap().to_string(),
            HashMap::new(),
            None,
        ).unwrap();
        
        Self {
            backend,
            temp_dir,
        }
    }
}

// Vector embedding fixtures
pub fn create_test_embeddings(count: usize, dim: usize) -> Vec<VectorEmbedding> {
    (0..count)
        .map(|i| VectorEmbedding {
            id: format!("test_{}", i),
            embedding: vec![1.0; dim],
            timestamp: i as i64 * 1000,
            metadata: HashMap::new(),
        })
        .collect()
}

// Model fixtures
pub fn create_test_model() -> Model {
    Model {
        metadata: Default::default(),
        layers: vec![],
        id: "test_model".to_string(),
        version: "1.0.0".to_string(),
    }
}

// Service fixtures
pub async fn create_test_service() -> (TestDatabase, String) {
    let db = TestDatabase::new();
    let addr = "127.0.0.1:0".to_string();
    
    // Initialize database
    db.backend.init().await.unwrap();
    
    (db, addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_embeddings() {
        let embeddings = create_test_embeddings(5, 4);
        assert_eq!(embeddings.len(), 5);
        assert_eq!(embeddings[0].embedding.len(), 4);
    }

    #[test]
    fn test_create_model() {
        let model = create_test_model();
        assert_eq!(model.id, "test_model");
        assert_eq!(model.version, "1.0.0");
    }

    #[tokio::test]
    async fn test_database_fixture() {
        let (db, addr) = create_test_service().await;
        assert!(!addr.is_empty());
        assert!(db.temp_dir.path().exists());
    }
}
