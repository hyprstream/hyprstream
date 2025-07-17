use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use uuid::Uuid;

use crate::namespace::{Namespace, ToolPath};
use crate::platform_dirs;
use crate::tcl_tools::ToolDefinition;

/// Metadata associated with a persisted tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolMetadata {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub checksum: String,
    pub file_version: u32,
}

/// A tool with its metadata for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedTool {
    pub metadata: ToolMetadata,
    pub tool: ToolDefinition,
}

/// Index for fast tool lookups
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolIndex {
    pub tools: HashMap<String, ToolIndexEntry>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolIndexEntry {
    pub path: ToolPath,
    pub file_path: PathBuf,
    pub checksum: String,
    pub updated_at: DateTime<Utc>,
}

/// File-based tool persistence manager
pub struct FilePersistence {
    storage_dir: PathBuf,
    index_path: PathBuf,
    index: ToolIndex,
}

impl FilePersistence {
    /// Create a new file persistence manager
    pub async fn new() -> Result<Self> {
        let storage_dir = get_storage_directory()?;
        let index_path = storage_dir.join("index.json");

        // Create storage directory if it doesn't exist
        fs::create_dir_all(&storage_dir).await?;

        // Load or create index
        let index = Self::load_or_create_index(&index_path).await?;

        Ok(Self {
            storage_dir,
            index_path,
            index,
        })
    }

    /// Create with custom storage directory (for testing)
    #[cfg(test)]
    pub async fn with_directory(storage_dir: PathBuf) -> Result<Self> {
        let index_path = storage_dir.join("index.json");

        fs::create_dir_all(&storage_dir).await?;
        let index = Self::load_or_create_index(&index_path).await?;

        Ok(Self {
            storage_dir,
            index_path,
            index,
        })
    }

    async fn load_or_create_index(index_path: &Path) -> Result<ToolIndex> {
        if index_path.exists() {
            let content = fs::read_to_string(index_path).await?;
            match serde_json::from_str(&content) {
                Ok(index) => Ok(index),
                Err(e) => {
                    tracing::warn!("Failed to parse index file, creating new one: {}", e);
                    Ok(ToolIndex::default())
                }
            }
        } else {
            Ok(ToolIndex::default())
        }
    }

    /// Save a tool to persistent storage
    pub async fn save_tool(&mut self, tool: &ToolDefinition) -> Result<()> {
        let file_path = self.get_tool_file_path(&tool.path);

        // Create directory structure if needed
        if let Some(parent) = file_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Calculate checksum
        let checksum = calculate_checksum(&tool.script);

        // Create persisted tool
        let now = Utc::now();
        let persisted = PersistedTool {
            metadata: ToolMetadata {
                id: Uuid::new_v4().to_string(),
                created_at: now,
                updated_at: now,
                checksum: checksum.clone(),
                file_version: 1,
            },
            tool: tool.clone(),
        };

        // Write tool file
        let json = serde_json::to_string_pretty(&persisted)?;
        fs::write(&file_path, json).await?;

        // Update index
        let path_key = tool.path.to_string();
        self.index.tools.insert(
            path_key,
            ToolIndexEntry {
                path: tool.path.clone(),
                file_path: file_path.clone(),
                checksum,
                updated_at: now,
            },
        );
        self.index.last_updated = now;

        // Save index
        self.save_index().await?;

        tracing::info!("Saved tool to {}", file_path.display());
        Ok(())
    }

    /// Load a tool from persistent storage
    pub async fn load_tool(&self, path: &ToolPath) -> Result<Option<ToolDefinition>> {
        let path_key = path.to_string();

        // Check index first
        if let Some(entry) = self.index.tools.get(&path_key) {
            if entry.file_path.exists() {
                let content = fs::read_to_string(&entry.file_path).await?;
                let persisted: PersistedTool = serde_json::from_str(&content)?;

                // Verify checksum if desired
                if persisted.metadata.checksum == entry.checksum {
                    return Ok(Some(persisted.tool));
                } else {
                    tracing::warn!("Checksum mismatch for tool {}, file may be corrupted", path);
                }
            }
        }

        // Fallback: try to load directly from expected path
        let file_path = self.get_tool_file_path(path);
        if file_path.exists() {
            let content = fs::read_to_string(&file_path).await?;
            let persisted: PersistedTool = serde_json::from_str(&content)?;
            return Ok(Some(persisted.tool));
        }

        Ok(None)
    }

    /// List all persisted tools
    pub async fn list_tools(&self, namespace_filter: Option<&str>) -> Result<Vec<ToolDefinition>> {
        let mut tools = Vec::new();

        for entry in self.index.tools.values() {
            // Apply namespace filter if specified
            if let Some(filter) = namespace_filter {
                let matches = match &entry.path.namespace {
                    Namespace::User(user) => user == filter,
                    Namespace::Bin => filter == "bin",
                    Namespace::Sbin => filter == "sbin",
                    Namespace::Docs => filter == "docs",
                    Namespace::Mcp(server) => server == filter || filter == "mcp",
                };

                if !matches {
                    continue;
                }
            }

            // Load tool
            if let Ok(Some(tool)) = self.load_tool(&entry.path).await {
                tools.push(tool);
            }
        }

        Ok(tools)
    }

    /// Delete a tool from persistent storage
    pub async fn delete_tool(&mut self, path: &ToolPath) -> Result<bool> {
        let path_key = path.to_string();

        // Remove from index
        if let Some(entry) = self.index.tools.remove(&path_key) {
            // Delete file
            if entry.file_path.exists() {
                fs::remove_file(&entry.file_path).await?;
                tracing::info!("Deleted tool file {}", entry.file_path.display());
            }

            // Clean up empty directories
            self.cleanup_empty_dirs(&entry.file_path).await?;

            // Update index
            self.index.last_updated = Utc::now();
            self.save_index().await?;

            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn save_index(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.index)?;
        fs::write(&self.index_path, json).await?;
        Ok(())
    }

    fn get_tool_file_path(&self, path: &ToolPath) -> PathBuf {
        let mut file_path = self.storage_dir.clone();

        match &path.namespace {
            Namespace::User(user) => {
                file_path = file_path.join("users").join(user);
                if let Some(package) = &path.package {
                    file_path = file_path.join(package);
                }
            }
            Namespace::Bin => file_path = file_path.join("system").join("bin"),
            Namespace::Sbin => file_path = file_path.join("system").join("sbin"),
            Namespace::Docs => file_path = file_path.join("system").join("docs"),
            Namespace::Mcp(server) => file_path = file_path.join("mcp").join(server),
        }

        let filename = if path.version == "latest" {
            format!("{}.json", path.name)
        } else {
            format!("{}_{}.json", path.name, path.version)
        };

        file_path.join(filename)
    }

    fn cleanup_empty_dirs<'a>(
        &'a self,
        file_path: &'a Path,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            if let Some(parent) = file_path.parent() {
                // Only remove directories within our storage area
                if parent.starts_with(&self.storage_dir) && parent != self.storage_dir {
                    // Check if directory is empty
                    if let Ok(mut entries) = fs::read_dir(parent).await {
                        if entries.next_entry().await?.is_none() {
                            // Directory is empty, remove it
                            fs::remove_dir(parent).await?;
                            tracing::debug!("Removed empty directory {}", parent.display());

                            // Recursively clean up parent directories
                            self.cleanup_empty_dirs(parent).await?;
                        }
                    }
                }
            }
            Ok(())
        })
    }
}

/// Get the appropriate storage directory for the current platform
fn get_storage_directory() -> Result<PathBuf> {
    platform_dirs::tools_dir()
}

/// Calculate a simple checksum for tool script content
fn calculate_checksum(content: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    format!("{:x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcl_tools::ParameterDefinition;
    use tempfile::TempDir;

    async fn create_test_persistence() -> Result<(FilePersistence, TempDir)> {
        let temp_dir = TempDir::new()?;
        let persistence = FilePersistence::with_directory(temp_dir.path().to_path_buf()).await?;
        Ok((persistence, temp_dir))
    }

    fn create_test_tool() -> ToolDefinition {
        ToolDefinition {
            path: ToolPath::user("alice", "utils", "test_tool", "1.0"),
            description: "A test tool".to_string(),
            script: "puts \"Hello from test tool\"".to_string(),
            parameters: vec![ParameterDefinition {
                name: "message".to_string(),
                description: "Message to display".to_string(),
                required: true,
                type_name: "string".to_string(),
            }],
        }
    }

    #[tokio::test]
    async fn test_save_and_load_tool() -> Result<()> {
        let (mut persistence, _temp) = create_test_persistence().await?;
        let tool = create_test_tool();

        // Save tool
        persistence.save_tool(&tool).await?;

        // Load tool
        let loaded = persistence.load_tool(&tool.path).await?;
        assert!(loaded.is_some());

        let loaded_tool = loaded.unwrap();
        assert_eq!(loaded_tool.path, tool.path);
        assert_eq!(loaded_tool.description, tool.description);
        assert_eq!(loaded_tool.script, tool.script);

        Ok(())
    }

    #[tokio::test]
    async fn test_list_tools() -> Result<()> {
        let (mut persistence, _temp) = create_test_persistence().await?;

        // Save multiple tools
        let tool1 = create_test_tool();
        let tool2 = ToolDefinition {
            path: ToolPath::user("bob", "math", "calculator", "2.0"),
            description: "Calculator tool".to_string(),
            script: "expr $a + $b".to_string(),
            parameters: vec![],
        };

        persistence.save_tool(&tool1).await?;
        persistence.save_tool(&tool2).await?;

        // List all tools
        let all_tools = persistence.list_tools(None).await?;
        assert_eq!(all_tools.len(), 2);

        // List tools by namespace
        let alice_tools = persistence.list_tools(Some("alice")).await?;
        assert_eq!(alice_tools.len(), 1);
        assert_eq!(
            alice_tools[0].path.namespace,
            Namespace::User("alice".to_string())
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_delete_tool() -> Result<()> {
        let (mut persistence, _temp) = create_test_persistence().await?;
        let tool = create_test_tool();

        // Save tool
        persistence.save_tool(&tool).await?;

        // Verify it exists
        assert!(persistence.load_tool(&tool.path).await?.is_some());

        // Delete tool
        let deleted = persistence.delete_tool(&tool.path).await?;
        assert!(deleted);

        // Verify it's gone
        assert!(persistence.load_tool(&tool.path).await?.is_none());

        // Try to delete again
        let deleted_again = persistence.delete_tool(&tool.path).await?;
        assert!(!deleted_again);

        Ok(())
    }
}
