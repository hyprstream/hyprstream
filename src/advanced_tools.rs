use crate::namespace::ToolPath;
use crate::tool_registry::{ToolQuery, ToolRegistry, ToolSource};
/// Advanced TCL tools using the unified tool registry
/// Provides enhanced listing, searching, and inspection capabilities
use anyhow::Result;
use serde_json::json;

/// Unified tool listing with VFS support and flexible output formats
pub async fn list_tools(
    registry: &ToolRegistry,
    namespace: Option<String>,
    server: Option<String>,
    search: Option<String>,
    include_schemas: bool,
    limit: Option<usize>,
    format: Option<String>,
) -> Result<String> {
    let query = ToolQuery {
        namespace,
        server,
        search,
        include_schemas,
        limit,
    };

    let tools = registry.query_tools(query).await?;

    if tools.is_empty() {
        return Ok("No tools found matching criteria".to_string());
    }

    let mut output = Vec::new();

    for tool in &tools {
        let mut tool_info = json!({
            "path": tool.path.to_string(),
            "mcp_name": tool.path.to_mcp_name(),
            "description": tool.description,
            "source": format!("{:?}", tool.source),
            "parameters": tool.parameters.len()
        });

        if include_schemas && tool.schema.is_some() {
            tool_info["schema"] = tool.schema.clone().unwrap();
        }

        output.push(tool_info);
    }

    // Handle output format
    match format.as_deref().unwrap_or("detailed") {
        "simple" => {
            // Simple format: just array of paths for backward compatibility
            let paths: Vec<String> = tools.iter().map(|t| t.path.to_string()).collect();
            Ok(serde_json::to_string_pretty(&paths)?)
        }
        _ => {
            // Detailed format (default): rich metadata
            Ok(serde_json::to_string_pretty(&json!({
                "tools": output,
                "count": output.len()
            }))?)
        }
    }
}

/// Get detailed information about a specific tool
pub async fn inspect_tool(registry: &ToolRegistry, tool_path: &str) -> Result<String> {
    let path = ToolPath::parse(tool_path)?;

    if let Some(tool) = registry.get_tool(&path).await {
        let mut info = json!({
            "path": tool.path.to_string(),
            "mcp_name": tool.path.to_mcp_name(),
            "description": tool.description,
            "source": describe_source(&tool.source),
            "namespace": format!("{:?}", tool.path.namespace),
            "parameters": tool.parameters,
            "schema": tool.schema
        });

        // Add source-specific information
        match &tool.source {
            ToolSource::System => {
                info["system_info"] = json!({
                    "type": "built-in",
                    "privileged": tool.path.to_string().starts_with("/sbin/")
                });
            }
            ToolSource::UserTcl { script } => {
                info["tcl_info"] = json!({
                    "script_length": script.len(),
                    "script_preview": if script.len() > 100 {
                        format!("{}...", &script[..100])
                    } else {
                        script.clone()
                    }
                });
            }
            ToolSource::ExternalMcp {
                server_id,
                tool_name,
            } => {
                info["mcp_info"] = json!({
                    "server_id": server_id,
                    "original_name": tool_name,
                    "mapped_path": format!("/xmcp/{}/{}", server_id, tool_name)
                });
            }
            ToolSource::Filesystem { file_path } => {
                info["filesystem_info"] = json!({
                    "file_path": file_path,
                    "exists": file_path.exists()
                });
            }
        }

        Ok(serde_json::to_string_pretty(&info)?)
    } else {
        Ok(format!("Tool not found: {}", tool_path))
    }
}

/// List available namespaces and their tool counts
pub async fn list_namespaces(registry: &ToolRegistry) -> Result<String> {
    let mut namespace_counts = std::collections::HashMap::new();

    // Query all tools
    let all_tools = registry.query_tools(ToolQuery::default()).await?;

    for tool in all_tools {
        let ns = match &tool.path.namespace {
            crate::namespace::Namespace::Bin => "bin".to_string(),
            crate::namespace::Namespace::Sbin => "sbin".to_string(),
            crate::namespace::Namespace::Docs => "docs".to_string(),
            crate::namespace::Namespace::User(user) => {
                if user == "xmcp" {
                    "xmcp".to_string()
                } else {
                    format!("user:{}", user)
                }
            }
            crate::namespace::Namespace::Mcp(server) => format!("mcp:{}", server),
        };

        *namespace_counts.entry(ns).or_insert(0) += 1;
    }

    let mut namespaces: Vec<_> = namespace_counts.into_iter().collect();
    namespaces.sort_by(|a, b| a.0.cmp(&b.0));

    let namespace_info: Vec<_> = namespaces
        .into_iter()
        .map(|(ns, count)| {
            json!({
                "namespace": ns,
                "tool_count": count,
                "description": describe_namespace(&ns)
            })
        })
        .collect();

    Ok(serde_json::to_string_pretty(&json!({
        "namespaces": namespace_info,
        "total_namespaces": namespace_info.len()
    }))?)
}

/// Search tools by description or name
pub async fn search_tools(
    registry: &ToolRegistry,
    query: &str,
    limit: Option<usize>,
) -> Result<String> {
    let search_query = ToolQuery {
        search: Some(query.to_string()),
        include_schemas: false,
        limit,
        ..Default::default()
    };

    let tools = registry.query_tools(search_query).await?;

    if tools.is_empty() {
        return Ok(format!("No tools found matching search: '{}'", query));
    }

    let results: Vec<_> = tools
        .into_iter()
        .map(|tool| {
            json!({
                "path": tool.path.to_string(),
                "mcp_name": tool.path.to_mcp_name(),
                "description": tool.description,
                "source_type": match tool.source {
                    ToolSource::System => "system",
                    ToolSource::UserTcl { .. } => "user_tcl",
                    ToolSource::ExternalMcp { .. } => "external_mcp",
                    ToolSource::Filesystem { .. } => "filesystem",
                }
            })
        })
        .collect();

    Ok(serde_json::to_string_pretty(&json!({
        "query": query,
        "results": results,
        "count": results.len()
    }))?)
}

/// List tools in xmcp namespace with server grouping
pub async fn list_xmcp_tools(
    registry: &ToolRegistry,
    server_filter: Option<String>,
) -> Result<String> {
    let query = ToolQuery {
        namespace: Some("xmcp".to_string()),
        server: server_filter.clone(),
        include_schemas: true,
        ..Default::default()
    };

    let tools = registry.query_tools(query).await?;

    if tools.is_empty() {
        return Ok(match server_filter {
            Some(server) => format!("No external MCP tools found for server: {}", server),
            None => "No external MCP tools found".to_string(),
        });
    }

    // Group by server
    let mut servers = std::collections::HashMap::new();
    for tool in tools {
        if let Some(server_id) = tool.path.package.as_ref() {
            let server_tools = servers.entry(server_id.clone()).or_insert_with(Vec::new);
            server_tools.push(json!({
                "name": tool.path.name,
                "mcp_name": tool.path.to_mcp_name(),
                "description": tool.description,
                "parameters": tool.parameters,
                "schema": tool.schema
            }));
        }
    }

    let server_info: Vec<_> = servers
        .into_iter()
        .map(|(server_id, tools)| {
            json!({
                "server_id": server_id,
                "tool_count": tools.len(),
                "tools": tools
            })
        })
        .collect();

    Ok(serde_json::to_string_pretty(&json!({
        "external_mcp_servers": server_info,
        "total_servers": server_info.len(),
        "total_tools": server_info.iter().map(|s| s["tool_count"].as_u64().unwrap_or(0)).sum::<u64>()
    }))?)
}

fn describe_source(source: &ToolSource) -> String {
    match source {
        ToolSource::System => "Built-in system tool".to_string(),
        ToolSource::UserTcl { .. } => "User-created TCL script".to_string(),
        ToolSource::ExternalMcp { server_id, .. } => format!("External MCP server: {}", server_id),
        ToolSource::Filesystem { file_path } => format!("Filesystem tool: {}", file_path.display()),
    }
}

fn describe_namespace(ns: &str) -> String {
    match ns {
        "bin" => "System binaries and utilities".to_string(),
        "sbin" => "System administration tools (privileged)".to_string(),
        "docs" => "Documentation and help tools".to_string(),
        "xmcp" => "External MCP server tools".to_string(),
        s if s.starts_with("user:") => format!("User namespace: {}", &s[5..]),
        s if s.starts_with("mcp:") => format!("Legacy MCP server: {}", &s[4..]),
        _ => "Unknown namespace".to_string(),
    }
}
