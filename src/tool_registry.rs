use crate::mcp_client::McpClient;
use crate::namespace::{Namespace, ToolPath};
use crate::tcl_tools::{ParameterDefinition, ToolDefinition};
use crate::tool_discovery::{DiscoveredTool, ToolDiscovery};
/// Tiered Tool Registry with Virtual File System
///
/// Provides a unified interface over multiple tool registries:
/// - Native TCL tools (bin, sbin, docs, user)
/// - External MCP servers (xmcp namespace)
/// - Discovered filesystem tools
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unified tool information combining all registry types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedTool {
    pub path: ToolPath,
    pub description: String,
    pub parameters: Vec<ParameterDefinition>,
    pub source: ToolSource,
    pub schema: Option<serde_json::Value>,
}

/// Source of a tool in the registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolSource {
    /// Built-in system tool
    System,
    /// User-created TCL tool
    UserTcl { script: String },
    /// External MCP server tool
    ExternalMcp {
        server_id: String,
        tool_name: String,
    },
    /// Discovered filesystem tool
    Filesystem { file_path: std::path::PathBuf },
}

/// Virtual File System path patterns
pub struct VfsPath {
    pub components: Vec<String>,
}

impl VfsPath {
    pub fn parse(path: &str) -> Self {
        let components = path
            .split('/')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        Self { components }
    }

    pub fn matches_namespace(&self, namespace: &str) -> bool {
        self.components
            .first()
            .map_or(false, |first| first == namespace)
    }

    pub fn matches_filter(&self, filter: &str) -> bool {
        self.components.iter().any(|comp| comp.contains(filter))
    }

    pub fn to_string(&self) -> String {
        format!("/{}", self.components.join("/"))
    }
}

/// Query parameters for tool listing
#[derive(Debug, Clone, Default)]
pub struct ToolQuery {
    /// Filter by namespace (bin, sbin, docs, user, xmcp)
    pub namespace: Option<String>,
    /// Filter by server (for xmcp namespace)
    pub server: Option<String>,
    /// Text search filter
    pub search: Option<String>,
    /// Include detailed schemas
    pub include_schemas: bool,
    /// Limit number of results
    pub limit: Option<usize>,
}

/// Unified tool registry combining all sources
pub struct ToolRegistry {
    /// Native TCL tools
    tcl_tools: HashMap<ToolPath, ToolDefinition>,
    /// System tools (hardcoded)
    system_tools: Vec<ToolPath>,
    /// Discovered filesystem tools
    discovered_tools: HashMap<ToolPath, DiscoveredTool>,
    /// MCP client for external servers
    mcp_client: McpClient,
    /// Tool discovery engine
    tool_discovery: ToolDiscovery,
}

impl ToolRegistry {
    pub fn new(mcp_client: McpClient) -> Self {
        let system_tools = vec![
            ToolPath::bin("tcl_execute"),
            ToolPath::bin("exec_tool"),
            ToolPath::bin("discover_tools"),
            ToolPath::bin("list_tools"),
            ToolPath::bin("inspect_tool"),
            ToolPath::bin("list_namespaces"),
            ToolPath::bin("search_tools"),
            ToolPath::bin("list_xmcp_tools"),
            ToolPath::sbin("tcl_tool_add"),
            ToolPath::sbin("tcl_tool_remove"),
            ToolPath::sbin("mcp_add"),
            ToolPath::sbin("mcp_remove"),
            ToolPath::sbin("mcp_list"),
            ToolPath::sbin("mcp_connect"),
            ToolPath::sbin("mcp_info"),
            ToolPath::sbin("mcp_ping"),
            ToolPath::docs("molt_book"),
        ];

        Self {
            tcl_tools: HashMap::new(),
            system_tools,
            discovered_tools: HashMap::new(),
            mcp_client,
            tool_discovery: ToolDiscovery::new(),
        }
    }

    /// Query tools with advanced filtering
    pub async fn query_tools(&self, query: ToolQuery) -> Result<Vec<UnifiedTool>> {
        let mut tools = Vec::new();

        // Add system tools
        if query
            .namespace
            .as_deref()
            .map_or(true, |ns| ns == "bin" || ns == "sbin" || ns == "docs")
        {
            for tool_path in &self.system_tools {
                if self.matches_query(tool_path, &query) {
                    let unified = UnifiedTool {
                        path: tool_path.clone(),
                        description: self.get_system_tool_description(tool_path),
                        parameters: self.get_system_tool_parameters(tool_path),
                        source: ToolSource::System,
                        schema: if query.include_schemas {
                            Some(self.get_system_tool_schema(tool_path))
                        } else {
                            None
                        },
                    };
                    tools.push(unified);
                }
            }
        }

        // Add user TCL tools
        if query.namespace.as_deref().map_or(true, |ns| ns == "user") {
            for (path, tool_def) in &self.tcl_tools {
                if self.matches_query(path, &query) {
                    let unified = UnifiedTool {
                        path: path.clone(),
                        description: tool_def.description.clone(),
                        parameters: tool_def.parameters.clone(),
                        source: ToolSource::UserTcl {
                            script: tool_def.script.clone(),
                        },
                        schema: if query.include_schemas {
                            Some(self.generate_schema_from_parameters(&tool_def.parameters))
                        } else {
                            None
                        },
                    };
                    tools.push(unified);
                }
            }
        }

        // Add discovered filesystem tools
        if query
            .namespace
            .as_deref()
            .map_or(true, |ns| ["bin", "sbin", "docs", "user"].contains(&ns))
        {
            for (path, discovered) in &self.discovered_tools {
                if self.matches_query(path, &query) {
                    let unified = UnifiedTool {
                        path: path.clone(),
                        description: discovered.description.clone(),
                        parameters: discovered.parameters.clone(),
                        source: ToolSource::Filesystem {
                            file_path: discovered.file_path.clone(),
                        },
                        schema: if query.include_schemas {
                            Some(self.generate_schema_from_parameters(&discovered.parameters))
                        } else {
                            None
                        },
                    };
                    tools.push(unified);
                }
            }
        }

        // Add external MCP tools (xmcp namespace)
        if query.namespace.as_deref().map_or(true, |ns| ns == "xmcp") {
            let mcp_tools = self.get_mcp_tools(&query).await?;
            tools.extend(mcp_tools);
        }

        // Apply search filter
        if let Some(search) = &query.search {
            tools.retain(|tool| {
                tool.path.to_string().contains(search) || tool.description.contains(search)
            });
        }

        // Apply limit
        if let Some(limit) = query.limit {
            tools.truncate(limit);
        }

        // Sort by path
        tools.sort_by(|a, b| a.path.to_string().cmp(&b.path.to_string()));

        Ok(tools)
    }

    /// Get MCP tools with xmcp namespace mapping
    async fn get_mcp_tools(&self, query: &ToolQuery) -> Result<Vec<UnifiedTool>> {
        let mut tools = Vec::new();
        let servers = self.mcp_client.list_servers().await;

        for (server_id, status) in servers {
            // Filter by server if specified
            if let Some(ref filter_server) = query.server {
                if server_id != *filter_server {
                    continue;
                }
            }

            // Only get tools from connected servers
            if matches!(status, crate::mcp_client::ConnectionStatus::Connected) {
                match self.mcp_client.get_server_tools(&server_id).await {
                    Ok(server_tools) => {
                        for mcp_tool in server_tools {
                            // Map to xmcp namespace: xmcp__<server>__<tool>
                            let tool_path = ToolPath {
                                namespace: Namespace::User("xmcp".to_string()),
                                package: Some(server_id.clone()),
                                name: mcp_tool.name.clone(),
                                version: "latest".to_string(),
                            };

                            if self.matches_query(&tool_path, query) {
                                let unified = UnifiedTool {
                                    path: tool_path,
                                    description: mcp_tool
                                        .description
                                        .unwrap_or_else(|| "No description".to_string()),
                                    parameters: self
                                        .convert_mcp_schema_to_parameters(&mcp_tool.input_schema),
                                    source: ToolSource::ExternalMcp {
                                        server_id: server_id.clone(),
                                        tool_name: mcp_tool.name.clone(),
                                    },
                                    schema: if query.include_schemas {
                                        Some(mcp_tool.input_schema)
                                    } else {
                                        None
                                    },
                                };
                                tools.push(unified);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to get tools from MCP server '{}': {}",
                            server_id,
                            e
                        );
                    }
                }
            }
        }

        Ok(tools)
    }

    /// Check if a tool path matches the query criteria
    fn matches_query(&self, path: &ToolPath, query: &ToolQuery) -> bool {
        // Namespace filter
        if let Some(ref ns) = query.namespace {
            let matches = match (&path.namespace, ns.as_str()) {
                (Namespace::Bin, "bin") => true,
                (Namespace::Sbin, "sbin") => true,
                (Namespace::Docs, "docs") => true,
                (Namespace::User(_user), "user") => true,
                (Namespace::User(user), filter) if user == filter => true,
                (Namespace::User(user), "xmcp") if user == "xmcp" => true,
                _ => false,
            };
            if !matches {
                return false;
            }
        }

        // Server filter (for xmcp namespace)
        if let Some(ref server_filter) = query.server {
            if let Namespace::User(user) = &path.namespace {
                if user == "xmcp" {
                    if let Some(ref package) = path.package {
                        if package != server_filter {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }

    /// Generate JSON schema from parameter definitions
    fn generate_schema_from_parameters(
        &self,
        parameters: &[ParameterDefinition],
    ) -> serde_json::Value {
        let mut properties = serde_json::Map::new();
        let mut required = Vec::new();

        for param in parameters {
            let json_type = match param.type_name.to_lowercase().as_str() {
                "string" | "str" | "text" => "string",
                "number" | "float" | "double" => "number",
                "integer" | "int" => "integer",
                "boolean" | "bool" => "boolean",
                "array" | "list" => "array",
                "object" | "dict" => "object",
                _ => "string",
            };

            properties.insert(
                param.name.clone(),
                serde_json::json!({
                    "type": json_type,
                    "description": param.description
                }),
            );

            if param.required {
                required.push(param.name.clone());
            }
        }

        serde_json::json!({
            "type": "object",
            "properties": properties,
            "required": required
        })
    }

    /// Convert MCP JSON schema to parameter definitions
    fn convert_mcp_schema_to_parameters(
        &self,
        schema: &serde_json::Value,
    ) -> Vec<ParameterDefinition> {
        let mut parameters = Vec::new();

        if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
            let required_fields: Vec<String> = schema
                .get("required")
                .and_then(|r| r.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(String::from)
                        .collect()
                })
                .unwrap_or_default();

            for (name, prop) in properties {
                let type_name = prop
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("string")
                    .to_string();

                let description = prop
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("No description")
                    .to_string();

                parameters.push(ParameterDefinition {
                    name: name.clone(),
                    description,
                    required: required_fields.contains(name),
                    type_name,
                });
            }
        }

        parameters
    }

    /// Get description for system tools
    fn get_system_tool_description(&self, path: &ToolPath) -> String {
        match path.to_string().as_str() {
            "/bin/tcl_execute" => "Execute a TCL script and return the result".to_string(),
            "/bin/exec_tool" => "Execute a tool by its path with parameters".to_string(),
            "/bin/discover_tools" => "Discover and index tools from the filesystem".to_string(),
            "/bin/list_tools" => {
                "List tools with flexible output formats and advanced filtering".to_string()
            }
            "/bin/inspect_tool" => {
                "Get detailed information about a specific tool including schema".to_string()
            }
            "/bin/list_namespaces" => "List available namespaces and their tool counts".to_string(),
            "/bin/search_tools" => "Search tools by description or name".to_string(),
            "/bin/list_xmcp_tools" => {
                "List tools in xmcp namespace with server grouping".to_string()
            }
            "/sbin/tcl_tool_add" => {
                "Add a new TCL tool to the available tools (PRIVILEGED)".to_string()
            }
            "/sbin/tcl_tool_remove" => {
                "Remove a TCL tool from the available tools (PRIVILEGED)".to_string()
            }
            "/sbin/mcp_add" => "Add an MCP server configuration (PRIVILEGED)".to_string(),
            "/sbin/mcp_remove" => "Remove an MCP server configuration (PRIVILEGED)".to_string(),
            "/sbin/mcp_list" => "List all registered MCP servers (PRIVILEGED)".to_string(),
            "/sbin/mcp_connect" => "Manually connect to an MCP server (PRIVILEGED)".to_string(),
            "/sbin/mcp_info" => {
                "Get detailed information about an MCP server (PRIVILEGED)".to_string()
            }
            "/sbin/mcp_ping" => "Test connectivity to an MCP server (PRIVILEGED)".to_string(),
            "/docs/molt_book" => {
                "Access Molt TCL interpreter documentation and examples".to_string()
            }
            _ => "System tool".to_string(),
        }
    }

    /// Get parameters for system tools
    fn get_system_tool_parameters(&self, path: &ToolPath) -> Vec<ParameterDefinition> {
        match path.to_string().as_str() {
            "/bin/tcl_execute" => vec![ParameterDefinition {
                name: "script".to_string(),
                description: "TCL script to execute".to_string(),
                required: true,
                type_name: "string".to_string(),
            }],
            "/bin/tcl_tool_list" => vec![
                ParameterDefinition {
                    name: "namespace".to_string(),
                    description: "Filter tools by namespace (optional)".to_string(),
                    required: false,
                    type_name: "string".to_string(),
                },
                ParameterDefinition {
                    name: "filter".to_string(),
                    description: "Filter tools by name pattern (optional)".to_string(),
                    required: false,
                    type_name: "string".to_string(),
                },
            ],
            "/bin/exec_tool" => vec![
                ParameterDefinition {
                    name: "tool_path".to_string(),
                    description: "Full path to the tool (e.g., '/bin/list_dir')".to_string(),
                    required: true,
                    type_name: "string".to_string(),
                },
                ParameterDefinition {
                    name: "params".to_string(),
                    description: "Parameters to pass to the tool".to_string(),
                    required: false,
                    type_name: "object".to_string(),
                },
            ],
            "/bin/list_tools" => vec![
                ParameterDefinition {
                    name: "namespace".to_string(),
                    description: "Filter by namespace (bin, sbin, docs, user, xmcp)".to_string(),
                    required: false,
                    type_name: "string".to_string(),
                },
                ParameterDefinition {
                    name: "server".to_string(),
                    description: "Filter by server (for xmcp namespace)".to_string(),
                    required: false,
                    type_name: "string".to_string(),
                },
                ParameterDefinition {
                    name: "search".to_string(),
                    description: "Text search filter".to_string(),
                    required: false,
                    type_name: "string".to_string(),
                },
                ParameterDefinition {
                    name: "include_schemas".to_string(),
                    description: "Include detailed schemas".to_string(),
                    required: false,
                    type_name: "boolean".to_string(),
                },
                ParameterDefinition {
                    name: "limit".to_string(),
                    description: "Limit number of results".to_string(),
                    required: false,
                    type_name: "integer".to_string(),
                },
                ParameterDefinition {
                    name: "format".to_string(),
                    description:
                        "Output format: 'simple' (paths only) or 'detailed' (rich metadata)"
                            .to_string(),
                    required: false,
                    type_name: "string".to_string(),
                },
            ],
            "/bin/inspect_tool" => vec![ParameterDefinition {
                name: "tool_path".to_string(),
                description: "Tool path to inspect (e.g., '/bin/tcl_execute')".to_string(),
                required: true,
                type_name: "string".to_string(),
            }],
            "/bin/list_namespaces" => vec![],
            "/bin/search_tools" => vec![
                ParameterDefinition {
                    name: "query".to_string(),
                    description: "Search query string".to_string(),
                    required: true,
                    type_name: "string".to_string(),
                },
                ParameterDefinition {
                    name: "limit".to_string(),
                    description: "Limit number of results".to_string(),
                    required: false,
                    type_name: "integer".to_string(),
                },
            ],
            "/bin/list_xmcp_tools" => vec![ParameterDefinition {
                name: "server".to_string(),
                description: "Filter by server ID".to_string(),
                required: false,
                type_name: "string".to_string(),
            }],
            _ => Vec::new(),
        }
    }

    /// Get JSON schema for system tools
    fn get_system_tool_schema(&self, path: &ToolPath) -> serde_json::Value {
        let parameters = self.get_system_tool_parameters(path);
        self.generate_schema_from_parameters(&parameters)
    }

    /// Add a TCL tool to the registry
    pub fn add_tcl_tool(&mut self, tool: ToolDefinition) {
        self.tcl_tools.insert(tool.path.clone(), tool);
    }

    /// Remove a TCL tool from the registry
    pub fn remove_tcl_tool(&mut self, path: &ToolPath) -> bool {
        self.tcl_tools.remove(path).is_some()
    }

    /// Add discovered tools to the registry
    pub fn add_discovered_tools(&mut self, tools: Vec<DiscoveredTool>) {
        for tool in tools {
            self.discovered_tools.insert(tool.path.clone(), tool);
        }
    }

    /// Get a specific tool by path
    pub async fn get_tool(&self, path: &ToolPath) -> Option<UnifiedTool> {
        // Check system tools
        if self.system_tools.contains(path) {
            return Some(UnifiedTool {
                path: path.clone(),
                description: self.get_system_tool_description(path),
                parameters: self.get_system_tool_parameters(path),
                source: ToolSource::System,
                schema: Some(self.get_system_tool_schema(path)),
            });
        }

        // Check TCL tools
        if let Some(tool_def) = self.tcl_tools.get(path) {
            return Some(UnifiedTool {
                path: path.clone(),
                description: tool_def.description.clone(),
                parameters: tool_def.parameters.clone(),
                source: ToolSource::UserTcl {
                    script: tool_def.script.clone(),
                },
                schema: Some(self.generate_schema_from_parameters(&tool_def.parameters)),
            });
        }

        // Check discovered tools
        if let Some(discovered) = self.discovered_tools.get(path) {
            return Some(UnifiedTool {
                path: path.clone(),
                description: discovered.description.clone(),
                parameters: discovered.parameters.clone(),
                source: ToolSource::Filesystem {
                    file_path: discovered.file_path.clone(),
                },
                schema: Some(self.generate_schema_from_parameters(&discovered.parameters)),
            });
        }

        // Check MCP tools (xmcp namespace)
        if let Namespace::User(user) = &path.namespace {
            if user == "xmcp" {
                if let Some(ref server_id) = path.package {
                    if let Ok(server_tools) = self.mcp_client.get_server_tools(server_id).await {
                        for mcp_tool in server_tools {
                            if mcp_tool.name == path.name {
                                return Some(UnifiedTool {
                                    path: path.clone(),
                                    description: mcp_tool
                                        .description
                                        .unwrap_or_else(|| "No description".to_string()),
                                    parameters: self
                                        .convert_mcp_schema_to_parameters(&mcp_tool.input_schema),
                                    source: ToolSource::ExternalMcp {
                                        server_id: server_id.clone(),
                                        tool_name: mcp_tool.name.clone(),
                                    },
                                    schema: Some(mcp_tool.input_schema),
                                });
                            }
                        }
                    }
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vfs_path_parsing() {
        let path = VfsPath::parse("/xmcp/context7/get_library_docs");
        assert_eq!(
            path.components,
            vec!["xmcp", "context7", "get_library_docs"]
        );
        assert!(path.matches_namespace("xmcp"));
        assert!(!path.matches_namespace("bin"));
        assert!(path.matches_filter("context7"));
    }

    #[test]
    fn test_tool_query_defaults() {
        let query = ToolQuery::default();
        assert!(query.namespace.is_none());
        assert!(query.server.is_none());
        assert!(!query.include_schemas);
    }
}
