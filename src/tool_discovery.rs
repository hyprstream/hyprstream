use crate::namespace::{Namespace, ToolPath};
use crate::tcl_tools::ParameterDefinition;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;

/// Tool discovery system for finding and indexing tools from the filesystem
#[derive(Debug, Clone)]
pub struct ToolDiscovery {
    /// Base directory for tool discovery
    tools_dir: PathBuf,
    /// Cache of discovered tools
    discovered_tools: HashMap<ToolPath, DiscoveredTool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredTool {
    pub path: ToolPath,
    pub description: String,
    pub file_path: PathBuf,
    pub parameters: Vec<ParameterDefinition>,
}

impl ToolDiscovery {
    /// Create a new tool discovery instance
    pub fn new() -> Self {
        // Default tools directory - can be configured later
        let tools_dir = PathBuf::from("tools");
        Self {
            tools_dir,
            discovered_tools: HashMap::new(),
        }
    }

    /// Set the base directory for tool discovery (for testing)
    #[cfg(test)]
    pub fn with_tools_dir(mut self, dir: PathBuf) -> Self {
        self.tools_dir = dir;
        self
    }

    /// Discover all tools in the filesystem
    pub async fn discover_tools(&mut self) -> Result<Vec<DiscoveredTool>> {
        self.discovered_tools.clear();

        // Scan system directories
        self.scan_directory(&self.tools_dir.join("bin"), Namespace::Bin)
            .await?;
        self.scan_directory(&self.tools_dir.join("sbin"), Namespace::Sbin)
            .await?;
        self.scan_directory(&self.tools_dir.join("docs"), Namespace::Docs)
            .await?;

        // Scan user directories
        let user_dir = self.tools_dir.join("users");
        if user_dir.exists() {
            self.scan_user_directories(&user_dir).await?;
        }

        Ok(self.discovered_tools.values().cloned().collect())
    }

    /// Scan a specific directory for tools
    async fn scan_directory(&mut self, dir: &Path, namespace: Namespace) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

        let mut entries = fs::read_dir(dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            // Only process .tcl files
            if path.extension().and_then(|s| s.to_str()) == Some("tcl") {
                if let Some(tool_name) = path.file_stem().and_then(|s| s.to_str()) {
                    // Read tool metadata from file header
                    let metadata = self.read_tool_metadata(&path).await?;

                    let tool_path = match &namespace {
                        Namespace::Bin => ToolPath::bin(tool_name),
                        Namespace::Sbin => ToolPath::sbin(tool_name),
                        Namespace::Docs => ToolPath::docs(tool_name),
                        Namespace::User(_) => continue, // Handled separately
                        Namespace::Mcp(_) => continue,  // MCP tools handled separately
                    };

                    let discovered = DiscoveredTool {
                        path: tool_path.clone(),
                        description: metadata.description,
                        file_path: path,
                        parameters: metadata.parameters,
                    };

                    self.discovered_tools.insert(tool_path, discovered);
                }
            }
        }

        Ok(())
    }

    /// Scan user directories for tools
    async fn scan_user_directories(&mut self, users_dir: &Path) -> Result<()> {
        let mut user_entries = fs::read_dir(users_dir).await?;

        while let Some(user_entry) = user_entries.next_entry().await? {
            let user_path = user_entry.path();
            if !user_path.is_dir() {
                continue;
            }

            let user_name = user_entry.file_name().to_string_lossy().to_string();

            // Scan packages within user directory
            let mut package_entries = fs::read_dir(&user_path).await?;
            while let Some(package_entry) = package_entries.next_entry().await? {
                let package_path = package_entry.path();
                if !package_path.is_dir() {
                    continue;
                }

                let package_name = package_entry.file_name().to_string_lossy().to_string();

                // Scan tools within package
                let mut tool_entries = fs::read_dir(&package_path).await?;
                while let Some(tool_entry) = tool_entries.next_entry().await? {
                    let tool_file = tool_entry.path();

                    if tool_file.extension().and_then(|s| s.to_str()) == Some("tcl") {
                        if let Some(tool_name) = tool_file.file_stem().and_then(|s| s.to_str()) {
                            let metadata = self.read_tool_metadata(&tool_file).await?;

                            let tool_path = ToolPath::user(
                                &user_name,
                                &package_name,
                                tool_name,
                                metadata.version.unwrap_or_else(|| "latest".to_string()),
                            );

                            let discovered = DiscoveredTool {
                                path: tool_path.clone(),
                                description: metadata.description,
                                file_path: tool_file,
                                parameters: metadata.parameters,
                            };

                            self.discovered_tools.insert(tool_path, discovered);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Read tool metadata from file header comments
    async fn read_tool_metadata(&self, file_path: &Path) -> Result<ToolMetadata> {
        let content = fs::read_to_string(file_path).await?;
        let mut metadata = ToolMetadata::default();

        // Parse header comments for metadata
        for line in content.lines() {
            if !line.trim_start().starts_with('#') {
                break; // Stop at first non-comment line
            }

            let comment = line.trim_start_matches('#').trim();

            if let Some(desc) = comment.strip_prefix("@description ") {
                metadata.description = desc.to_string();
            } else if let Some(version) = comment.strip_prefix("@version ") {
                metadata.version = Some(version.to_string());
            } else if let Some(param_line) = comment.strip_prefix("@param ") {
                // Parse parameter definition: @param name:type:required description
                if let Some((def, desc)) = param_line.split_once(' ') {
                    let parts: Vec<&str> = def.split(':').collect();
                    if parts.len() >= 2 {
                        let param = ParameterDefinition {
                            name: parts[0].to_string(),
                            type_name: parts[1].to_string(),
                            required: parts.get(2).map(|&r| r == "required").unwrap_or(false),
                            description: desc.to_string(),
                        };
                        metadata.parameters.push(param);
                    }
                }
            }
        }

        if metadata.description.is_empty() {
            metadata.description = format!("Tool from {}", file_path.display());
        }

        Ok(metadata)
    }
}

#[derive(Debug, Default)]
struct ToolMetadata {
    description: String,
    version: Option<String>,
    parameters: Vec<ParameterDefinition>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tokio::fs;

    #[tokio::test]
    async fn test_tool_discovery() {
        // Create temporary directory structure
        let temp_dir = tempfile::tempdir().unwrap();
        let tools_dir = temp_dir.path().join("tools");

        // Create bin directory with a tool
        let bin_dir = tools_dir.join("bin");
        fs::create_dir_all(&bin_dir).await.unwrap();

        let tool_content = r#"#!/usr/bin/env tclsh
# @description List directory contents
# @param path:string:required Directory path to list

puts [glob -directory $path *]
"#;

        let tool_path = bin_dir.join("list_dir.tcl");
        let mut file = std::fs::File::create(&tool_path).unwrap();
        file.write_all(tool_content.as_bytes()).unwrap();

        // Test discovery
        let mut discovery = ToolDiscovery::new().with_tools_dir(tools_dir);
        let tools = discovery.discover_tools().await.unwrap();

        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].path.name, "list_dir");
        assert_eq!(tools[0].description, "List directory contents");
        assert_eq!(tools[0].parameters.len(), 1);
        assert_eq!(tools[0].parameters[0].name, "path");
        assert_eq!(tools[0].parameters[0].type_name, "string");
        assert!(tools[0].parameters[0].required);
    }
}
