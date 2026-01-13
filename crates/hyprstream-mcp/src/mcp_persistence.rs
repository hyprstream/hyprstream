/// MCP Server Index Persistence
///
/// Manages persistent storage of MCP server configurations
/// across restarts using platform-specific data directories
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::fs;

use crate::mcp_client::McpServerConfig;
use crate::platform_dirs;

/// MCP Server Index containing all registered servers
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct McpServerIndex {
    /// Map of server ID to configuration
    pub servers: HashMap<String, McpServerEntry>,
    /// Index version for future migrations
    pub version: u32,
    /// Last time the index was updated
    pub last_updated: DateTime<Utc>,
}

/// Entry for a single MCP server in the index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerEntry {
    /// Server configuration
    pub config: McpServerConfig,
    /// When this server was added
    pub added_at: DateTime<Utc>,
    /// When this server was last modified
    pub updated_at: DateTime<Utc>,
    /// Whether to auto-start this server
    pub auto_start: bool,
    /// Custom metadata (for future extensions)
    pub metadata: HashMap<String, serde_json::Value>,
}

/// MCP Server persistence manager
pub struct McpPersistence {
    index_path: std::path::PathBuf,
    index: McpServerIndex,
}

impl McpPersistence {
    /// Create a new MCP persistence manager
    pub async fn new() -> Result<Self> {
        // Ensure platform directories exist
        platform_dirs::init_directories()?;

        let index_path = platform_dirs::mcp_index_path()?;
        let index = Self::load_or_create_index(&index_path).await?;

        Ok(Self { index_path, index })
    }

    /// Load index from disk or create a new one
    async fn load_or_create_index(path: &std::path::Path) -> Result<McpServerIndex> {
        if path.exists() {
            match fs::read_to_string(path).await {
                Ok(content) => match serde_json::from_str::<McpServerIndex>(&content) {
                    Ok(index) => {
                        tracing::info!(
                            "Loaded MCP server index with {} servers",
                            index.servers.len()
                        );
                        Ok(index)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse MCP index, creating new: {}", e);
                        Ok(McpServerIndex {
                            version: 1,
                            last_updated: Utc::now(),
                            ..Default::default()
                        })
                    }
                },
                Err(e) => {
                    tracing::warn!("Failed to read MCP index, creating new: {}", e);
                    Ok(McpServerIndex {
                        version: 1,
                        last_updated: Utc::now(),
                        ..Default::default()
                    })
                }
            }
        } else {
            tracing::info!("No existing MCP index found, creating new");
            Ok(McpServerIndex {
                version: 1,
                last_updated: Utc::now(),
                ..Default::default()
            })
        }
    }

    /// Save the current index to disk
    async fn save_index(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.index)?;
        fs::write(&self.index_path, json).await?;
        tracing::debug!("Saved MCP index to {}", self.index_path.display());
        Ok(())
    }

    /// Add or update a server configuration
    pub async fn save_server(
        &mut self,
        id: String,
        config: McpServerConfig,
        auto_start: bool,
    ) -> Result<()> {
        let now = Utc::now();

        // Create a config with the provided auto_start setting
        let mut updated_config = config;
        updated_config.auto_start = auto_start;

        let entry = if let Some(existing) = self.index.servers.get(&id) {
            // Update existing entry
            McpServerEntry {
                config: updated_config,
                added_at: existing.added_at,
                updated_at: now,
                auto_start,
                metadata: existing.metadata.clone(),
            }
        } else {
            // Create new entry
            McpServerEntry {
                config: updated_config,
                added_at: now,
                updated_at: now,
                auto_start,
                metadata: HashMap::new(),
            }
        };

        self.index.servers.insert(id.clone(), entry);
        self.index.last_updated = now;
        self.save_index().await?;

        tracing::info!("Saved MCP server configuration: {}", id);
        Ok(())
    }

    /// Remove a server configuration
    pub async fn remove_server(&mut self, id: &str) -> Result<bool> {
        if self.index.servers.remove(id).is_some() {
            self.index.last_updated = Utc::now();
            self.save_index().await?;
            tracing::info!("Removed MCP server configuration: {}", id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Get a server configuration by ID
    pub fn get_server(&self, id: &str) -> Option<&McpServerEntry> {
        self.index.servers.get(id)
    }

    /// List all server configurations
    pub fn list_servers(&self) -> Vec<(String, McpServerEntry)> {
        self.index
            .servers
            .iter()
            .map(|(id, entry)| (id.clone(), entry.clone()))
            .collect()
    }

    /// Get servers that should auto-start
    pub fn get_auto_start_servers(&self) -> Vec<(String, McpServerConfig)> {
        self.index
            .servers
            .iter()
            .filter(|(_, entry)| entry.auto_start)
            .map(|(id, entry)| (id.clone(), entry.config.clone()))
            .collect()
    }

    /// Update server metadata
    pub async fn update_metadata(
        &mut self,
        id: &str,
        key: String,
        value: serde_json::Value,
    ) -> Result<bool> {
        if let Some(entry) = self.index.servers.get_mut(id) {
            entry.metadata.insert(key, value);
            entry.updated_at = Utc::now();
            self.index.last_updated = Utc::now();
            self.save_index().await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    async fn create_test_persistence() -> Result<(McpPersistence, TempDir)> {
        let temp_dir = TempDir::new()?;
        
        // Use a unique test-specific path to avoid race conditions between concurrent tests
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let test_id = format!("test_{}_{}", std::process::id(), timestamp);
        let unique_home = temp_dir.path().join(&test_id);
        std::fs::create_dir_all(&unique_home)?;
        std::env::set_var("HOME", &unique_home);

        let persistence = McpPersistence::new().await?;
        Ok((persistence, temp_dir))
    }

    fn create_test_config() -> McpServerConfig {
        McpServerConfig {
            id: "test-server".to_string(),
            name: "Test Server".to_string(),
            description: Some("A test MCP server".to_string()),
            command: "npx".to_string(),
            args: vec![
                "-y".to_string(),
                "@modelcontextprotocol/server-everything".to_string(),
            ],
            env: std::collections::HashMap::new(),
            auto_start: true,
            timeout_ms: 30000,
            max_retries: 3,
            created_at: chrono::Utc::now(),
        }
    }

    #[tokio::test]
    async fn test_save_and_load_server() -> Result<()> {
        let (mut persistence, _temp) = create_test_persistence().await?;
        let config = create_test_config();

        // Save server
        persistence
            .save_server("test-server".to_string(), config.clone(), true)
            .await?;

        // Get server
        let entry = persistence.get_server("test-server").unwrap();
        assert_eq!(entry.config.command, config.command);
        assert!(entry.auto_start);

        Ok(())
    }

    #[tokio::test]
    async fn test_auto_start_servers() -> Result<()> {
        let (mut persistence, _temp) = create_test_persistence().await?;

        // Save servers with different auto_start settings
        persistence
            .save_server("auto1".to_string(), create_test_config(), true)
            .await?;
        persistence
            .save_server("manual1".to_string(), create_test_config(), false)
            .await?;
        persistence
            .save_server("auto2".to_string(), create_test_config(), true)
            .await?;

        // Get auto-start servers
        let auto_start = persistence.get_auto_start_servers();
        assert_eq!(auto_start.len(), 2);

        let ids: Vec<String> = auto_start.iter().map(|(id, _)| id.clone()).collect();
        assert!(ids.contains(&"auto1".to_string()));
        assert!(ids.contains(&"auto2".to_string()));

        Ok(())
    }
}
