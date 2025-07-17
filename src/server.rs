use anyhow::{anyhow, Result};
use jsonrpc_core::{IoHandler, Params, Value};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, info};

use crate::namespace::ToolPath;
use crate::tcl_executor::TclExecutor;
use crate::tcl_runtime::RuntimeConfig;
use crate::tcl_tools::{
    TclExecToolRequest, TclExecuteRequest, TclToolAddRequest, TclToolBox, TclToolRemoveRequest,
};

#[derive(Clone)]
pub struct TclMcpServer {
    tool_box: TclToolBox,
    handler: IoHandler,
}

#[derive(Debug, Serialize, Deserialize)]
struct McpToolInfo {
    name: String,
    description: Option<String>,
    #[serde(rename = "inputSchema")]
    input_schema: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct McpListToolsResult {
    tools: Vec<McpToolInfo>,
}

#[derive(Debug, Serialize, Deserialize)]
struct McpCallToolParams {
    name: String,
    arguments: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct McpCallToolResult {
    content: Vec<McpContent>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum McpContent {
    #[serde(rename = "text")]
    Text { text: String },
}

impl TclMcpServer {
    pub fn new(privileged: bool) -> Self {
        // Spawn the TCL executor with privilege settings
        let executor = TclExecutor::spawn(privileged);
        let tool_box = TclToolBox::new(executor);
        let handler = IoHandler::new();

        Self::setup_handler(tool_box, handler, privileged)
    }

    pub fn new_with_runtime(
        privileged: bool,
        runtime_config: RuntimeConfig,
    ) -> Result<Self, String> {
        // Spawn the TCL executor with privilege and runtime settings
        let executor = TclExecutor::spawn_with_runtime(privileged, runtime_config)?;
        let tool_box = TclToolBox::new(executor);
        let handler = IoHandler::new();

        Ok(Self::setup_handler(tool_box, handler, privileged))
    }

    fn setup_handler(tool_box: TclToolBox, mut handler: IoHandler, privileged: bool) -> Self {
        // Register MCP methods
        handler.add_sync_method("initialize", move |_params: Params| {
            info!("MCP initialize called");
            Ok(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "tcl-mcp-server",
                    "version": "1.0.0"
                }
            }))
        });

        let tb = tool_box.clone();
        let tb2 = tool_box.clone();
        let is_privileged = privileged;
        handler.add_sync_method("tools/list", move |_params: Params| {
            debug!("MCP tools/list called (privileged: {})", is_privileged);
            let tb = tb.clone();
            
            // Don't use async block here since we're in a sync context
            let mut tools = vec![];
                
                // Add system tools with MCP-compatible names
                let mut system_tools = vec![
                    (ToolPath::bin("tcl_execute"), "Execute a TCL script and return the result", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "script": {
                                "type": "string",
                                "description": "TCL script to execute"
                            }
                        },
                        "required": ["script"]
                    })),
                    (ToolPath::docs("molt_book"), "Access Molt TCL interpreter documentation and examples", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Documentation topic: 'overview', 'commands', 'examples', 'links', or 'basic_syntax'",
                                "enum": ["overview", "commands", "examples", "links", "basic_syntax"]
                            }
                        },
                        "required": ["topic"]
                    })),
                    (ToolPath::bin("exec_tool"), "Execute a tool by its path with parameters", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "tool_path": {
                                "type": "string",
                                "description": "Full path to the tool (e.g., '/bin/list_dir')"
                            },
                            "params": {
                                "type": "object",
                                "description": "Parameters to pass to the tool",
                                "default": {}
                            }
                        },
                        "required": ["tool_path"]
                    })),
                    (ToolPath::bin("discover_tools"), "Discover and index tools from the filesystem", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {}
                    })),
                    (ToolPath::bin("execute_mcp"), "Execute a tool on a registered MCP server", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "server_id": {
                                "type": "string",
                                "description": "ID of the MCP server to execute tool on"
                            },
                            "tool_name": {
                                "type": "string",
                                "description": "Name of the tool to execute"
                            },
                            "params": {
                                "type": "object",
                                "description": "Parameters to pass to the tool",
                                "default": {}
                            },
                            "response_format": {
                                "type": "string",
                                "description": "Response format: json, text, or auto",
                                "enum": ["json", "text", "auto"],
                                "default": "auto"
                            },
                            "timeout_ms": {
                                "type": "integer",
                                "description": "Timeout in milliseconds",
                                "default": 30000
                            }
                        },
                        "required": ["server_id", "tool_name"]
                    })),
                    (ToolPath::bin("list_tools"), "List tools with flexible output formats and advanced filtering", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "namespace": {
                                "type": "string",
                                "description": "Filter by namespace (bin, sbin, docs, user, xmcp)"
                            },
                            "server": {
                                "type": "string", 
                                "description": "Filter by server (for xmcp namespace)"
                            },
                            "search": {
                                "type": "string",
                                "description": "Text search filter"
                            },
                            "include_schemas": {
                                "type": "boolean",
                                "description": "Include detailed schemas",
                                "default": false
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Limit number of results"
                            },
                            "format": {
                                "type": "string",
                                "description": "Output format: 'simple' (paths only) or 'detailed' (rich metadata)",
                                "default": "detailed"
                            }
                        }
                    })),
                    (ToolPath::bin("inspect_tool"), "Get detailed information about a specific tool including schema", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "tool_path": {
                                "type": "string",
                                "description": "Tool path to inspect (e.g., '/bin/tcl_execute')"
                            }
                        },
                        "required": ["tool_path"]
                    })),
                    (ToolPath::bin("list_namespaces"), "List available namespaces and their tool counts", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {}
                    })),
                    (ToolPath::bin("search_tools"), "Search tools by description or name", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query string"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Limit number of results"
                            }
                        },
                        "required": ["query"]
                    })),
                    (ToolPath::bin("list_xmcp_tools"), "List tools in xmcp namespace with server grouping", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "server": {
                                "type": "string",
                                "description": "Filter by server ID"
                            }
                        }
                    })),
                    (ToolPath::bin("mcp_list"), "List registered MCP servers", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {}
                    })),
                ];
                
                // Add privileged tools only if in privileged mode
                if is_privileged {
                    system_tools.push((ToolPath::sbin("tcl_tool_add"), "Add a new TCL tool to the available tools (PRIVILEGED)", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "user": {
                                "type": "string",
                                "description": "User namespace"
                            },
                            "package": {
                                "type": "string",
                                "description": "Package name"
                            },
                            "name": {
                                "type": "string",
                                "description": "Name of the new tool"
                            },
                            "version": {
                                "type": "string",
                                "description": "Version of the tool (defaults to 'latest')",
                                "default": "latest"
                            },
                            "description": {
                                "type": "string",
                                "description": "Description of what the tool does"
                            },
                            "script": {
                                "type": "string",
                                "description": "TCL script that implements the tool"
                            },
                            "parameters": {
                                "type": "array",
                                "description": "Parameters that the tool accepts",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": { "type": "string" },
                                        "description": { "type": "string" },
                                        "required": { "type": "boolean" },
                                        "type_name": { "type": "string" }
                                    },
                                    "required": ["name", "description", "required", "type_name"]
                                }
                            }
                        },
                        "required": ["user", "package", "name", "description", "script"]
                    })));
                    system_tools.push((ToolPath::sbin("tcl_tool_remove"), "Remove a TCL tool from the available tools (PRIVILEGED)", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Full tool path (e.g., '/alice/utils/reverse_string:1.0')"
                            }
                        },
                        "required": ["path"]
                    })));
                    
                    system_tools.push((ToolPath::sbin("tcl_tool_reload"), "Reload tools from persistent storage (PRIVILEGED)", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {}
                    })));
                    
                    system_tools.push((ToolPath::sbin("mcp_add"), "Register a new MCP server (PRIVILEGED)", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique identifier for the server"
                            },
                            "name": {
                                "type": "string",
                                "description": "Human-readable name for the server"
                            },
                            "description": {
                                "type": "string",
                                "description": "Optional description of the server"
                            },
                            "command": {
                                "type": "string",
                                "description": "Command to start the server"
                            },
                            "args": {
                                "type": "array",
                                "description": "Command line arguments",
                                "items": { "type": "string" },
                                "default": []
                            },
                            "env": {
                                "type": "object",
                                "description": "Environment variables",
                                "additionalProperties": { "type": "string" },
                                "default": {}
                            },
                            "auto_start": {
                                "type": "boolean",
                                "description": "Whether to auto-start the server",
                                "default": true
                            },
                            "timeout_ms": {
                                "type": "integer",
                                "description": "Connection timeout in milliseconds",
                                "default": 30000
                            },
                            "max_retries": {
                                "type": "integer",
                                "description": "Maximum retry attempts",
                                "default": 3
                            }
                        },
                        "required": ["id", "name", "command"]
                    })));
                    system_tools.push((ToolPath::sbin("mcp_remove"), "Remove an MCP server (PRIVILEGED)", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "server_id": {
                                "type": "string",
                                "description": "ID of the server to remove"
                            },
                            "force": {
                                "type": "boolean",
                                "description": "Whether to force removal (kill process)",
                                "default": false
                            }
                        },
                        "required": ["server_id"]
                    })));
                    
                    // Add MCP debugging tools
                    system_tools.push((ToolPath::sbin("mcp_connect"), "Manually connect to an MCP server (PRIVILEGED)", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "server_id": {
                                "type": "string",
                                "description": "ID of the server to connect to"
                            }
                        },
                        "required": ["server_id"]
                    })));
                    
                    system_tools.push((ToolPath::sbin("mcp_disconnect"), "Manually disconnect from an MCP server (PRIVILEGED)", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "server_id": {
                                "type": "string",
                                "description": "ID of the server to disconnect from"
                            }
                        },
                        "required": ["server_id"]
                    })));
                    
                    system_tools.push((ToolPath::sbin("mcp_info"), "Get detailed information about an MCP server (PRIVILEGED)", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "server_id": {
                                "type": "string",
                                "description": "ID of the server to get info for"
                            }
                        },
                        "required": ["server_id"]
                    })));
                    
                    system_tools.push((ToolPath::sbin("mcp_ping"), "Test connectivity to an MCP server (PRIVILEGED)", json!({
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {
                            "server_id": {
                                "type": "string",
                                "description": "ID of the server to ping"
                            }
                        },
                        "required": ["server_id"]
                    })));
                }
                
                for (path, description, schema) in system_tools {
                    tools.push(McpToolInfo {
                        name: path.to_mcp_name(),
                        description: Some(description.to_string()),
                        input_schema: schema,
                    });
                }
                
            // Get custom tools synchronously - this should be fast
            let tb_for_tools = tb.clone();
            let custom_tools = match std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(tb_for_tools.get_tool_definitions())
            }).join() {
                Ok(result) => result,
                Err(_) => {
                    return Err(jsonrpc_core::Error::internal_error());
                }
            };
            
            // Add custom tools to the list
            if let Ok(tool_defs) = custom_tools {
                for tool_def in tool_defs {
                    // Build input schema for custom tool
                    let mut properties = serde_json::Map::new();
                    let mut required = Vec::new();
                    
                    for param in &tool_def.parameters {
                        // Validate and normalize JSON Schema type
                        let json_type = match param.type_name.to_lowercase().as_str() {
                            "string" | "str" | "text" => "string",
                            "number" | "float" | "double" | "real" => "number",
                            "integer" | "int" | "long" => "integer", 
                            "boolean" | "bool" => "boolean",
                            "array" | "list" => "array",
                            "object" | "dict" | "map" => "object",
                            "null" | "nil" | "none" => "null",
                            // Default to string for unknown types to maintain compatibility
                            _ => "string"
                        };
                        
                        properties.insert(
                            param.name.clone(),
                            json!({
                                "type": json_type,
                                "description": param.description,
                            }),
                        );
                        
                        if param.required {
                            required.push(param.name.clone());
                        }
                    }
                    
                    // Build the schema object, only including "required" if it's not empty
                    let mut schema_obj = serde_json::Map::new();
                    schema_obj.insert("$schema".to_string(), json!("https://json-schema.org/draft/2020-12/schema"));
                    schema_obj.insert("type".to_string(), json!("object"));
                    schema_obj.insert("properties".to_string(), json!(properties));
                    
                    // Only add "required" array if there are required parameters
                    if !required.is_empty() {
                        schema_obj.insert("required".to_string(), json!(required));
                    }
                    
                    let input_schema = serde_json::Value::Object(schema_obj);
                    
                    tools.push(McpToolInfo {
                        name: tool_def.path.to_mcp_name(),
                        description: Some(format!("{} [{}]", tool_def.description, tool_def.path)),
                        input_schema,
                    });
                }
            }
            
            // Add MCP server tools
            let tb_for_mcp = tb.clone();
            match std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async move {
                    // Get list of MCP servers and their tools
                    tb_for_mcp.get_mcp_server_tools().await
                })
            }).join() {
                Ok(Ok(mcp_tools)) => {
                    // Add each MCP server's tools to the list
                    for (server_id, server_tools) in mcp_tools {
                        for tool in server_tools {
                            // Create MCP tool path under the MCP namespace
                            let tool_path = crate::namespace::ToolPath {
                                namespace: crate::namespace::Namespace::Mcp(server_id.clone()),
                                name: tool.name.clone(),
                                package: None,
                                version: "latest".to_string(),
                            };
                            
                            tools.push(McpToolInfo {
                                name: tool_path.to_mcp_name(),
                                description: Some(format!("{} [MCP:{}]", 
                                    tool.description.as_ref().unwrap_or(&"No description".to_string()), 
                                    server_id
                                )),
                                input_schema: tool.input_schema,
                            });
                        }
                    }
                }
                Ok(Err(e)) => {
                    debug!("Failed to get MCP server tools: {}", e);
                }
                Err(_) => {
                    debug!("Thread panic while getting MCP server tools");
                }
            }
            
            Ok(json!(McpListToolsResult { tools }))
        });

        let is_privileged_call = privileged;
        handler.add_sync_method("tools/call", move |params: Params| {
            debug!("MCP tools/call called with params: {:?}", params);
            let tb = tb2.clone();
            
            let params: McpCallToolParams = params.parse()?;
            info!("Calling tool: {} (privileged: {})", params.name, is_privileged_call);
            
            let result = std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().unwrap();
                rt.block_on(async move {
                // Check if it's a system tool by MCP name
                match params.name.as_str() {
                    "bin__tcl_execute" => {
                        let request: TclExecuteRequest = serde_json::from_value(params.arguments)?;
                        tb.tcl_execute(request).await
                    }
                    "sbin__tcl_tool_add" => {
                        if !is_privileged_call {
                            return Err(anyhow::anyhow!("Tool management requires --privileged mode"));
                        }
                        let request: TclToolAddRequest = serde_json::from_value(params.arguments)?;
                        tb.tcl_tool_add(request).await
                    }
                    "sbin__tcl_tool_remove" => {
                        if !is_privileged_call {
                            return Err(anyhow::anyhow!("Tool management requires --privileged mode"));
                        }
                        let request: TclToolRemoveRequest = serde_json::from_value(params.arguments)?;
                        tb.tcl_tool_remove(request).await
                    }
                    "sbin__tcl_tool_reload" => {
                        if !is_privileged_call {
                            return Err(anyhow::anyhow!("Tool management requires --privileged mode"));
                        }
                        tb.reload_tools().await
                    }
                    "bin__exec_tool" => {
                        let request: TclExecToolRequest = serde_json::from_value(params.arguments)?;
                        tb.exec_tool(request).await
                    }
                    "bin__discover_tools" => {
                        tb.discover_tools().await
                    }
                    "bin__execute_mcp" => {
                        let request: crate::tcl_tools::McpExecuteRequest = serde_json::from_value(params.arguments)?;
                        tb.mcp_execute(request).await
                    }
                    "sbin__mcp_add" => {
                        if !is_privileged_call {
                            return Err(anyhow::anyhow!("MCP server management requires --privileged mode"));
                        }
                        let request: crate::tcl_tools::McpServerAddRequest = serde_json::from_value(params.arguments)?;
                        tb.mcp_add_server(request).await
                    }
                    "sbin__mcp_remove" => {
                        if !is_privileged_call {
                            return Err(anyhow::anyhow!("MCP server management requires --privileged mode"));
                        }
                        let request: crate::tcl_tools::McpServerRemoveRequest = serde_json::from_value(params.arguments)?;
                        tb.mcp_remove_server(request).await
                    }
                    "bin__mcp_list" => {
                        tb.mcp_list_servers().await
                    }
                    "sbin__mcp_connect" => {
                        if !is_privileged_call {
                            return Err(anyhow::anyhow!("MCP debugging requires --privileged mode"));
                        }
                        let request: crate::tcl_tools::McpDebugRequest = serde_json::from_value(params.arguments)?;
                        tb.debug_connect_mcp(request).await
                    }
                    "sbin__mcp_disconnect" => {
                        if !is_privileged_call {
                            return Err(anyhow::anyhow!("MCP debugging requires --privileged mode"));
                        }
                        let request: crate::tcl_tools::McpDebugRequest = serde_json::from_value(params.arguments)?;
                        tb.debug_disconnect_mcp(request).await
                    }
                    "sbin__mcp_info" => {
                        if !is_privileged_call {
                            return Err(anyhow::anyhow!("MCP debugging requires --privileged mode"));
                        }
                        let request: crate::tcl_tools::McpDebugRequest = serde_json::from_value(params.arguments)?;
                        tb.debug_mcp_info(request).await
                    }
                    "sbin__mcp_ping" => {
                        if !is_privileged_call {
                            return Err(anyhow::anyhow!("MCP debugging requires --privileged mode"));
                        }
                        let request: crate::tcl_tools::McpDebugRequest = serde_json::from_value(params.arguments)?;
                        tb.debug_ping_mcp(request).await
                    }
                    "docs__molt_book" => {
                        // Handle documentation request
                        let topic = params.arguments.get("topic")
                            .and_then(|v| v.as_str())
                            .unwrap_or("overview");
                        
                        match topic {
                            "overview" => Ok(format!(r#"# Molt TCL Interpreter Overview

## What is Molt?
Molt is a TCL (Tool Command Language) interpreter implemented in Rust. It provides a memory-safe, 
embeddable scripting language with familiar TCL syntax.

## Key Features
- Memory-safe implementation in Rust
- Compatible with core TCL commands
- Embeddable in Rust applications
- Thread-safe design
- Standard TCL control structures and data types

## Documentation Links
- Molt Book: https://wduquette.github.io/molt/
- GitHub Repository: https://github.com/wduquette/molt
- Source Documentation: https://github.com/wduquette/molt/tree/master/molt-book/src

Use 'basic_syntax', 'commands', 'examples', or 'links' for more specific information."#)),
                            "basic_syntax" => Ok(format!(r#"# TCL Basic Syntax

## Variables
```tcl
set name "Alice"
set age 30
puts "Hello, $name! You are $age years old."
```

## Lists
```tcl
set fruits [list apple banana cherry]
set first [lindex $fruits 0]  ;# apple
set length [llength $fruits]  ;# 3
```

## Control Structures
```tcl
# If statement
if {{$age >= 18}} {{
    puts "Adult"
}} else {{
    puts "Minor"
}}

# For loop
for {{set i 0}} {{$i < 5}} {{incr i}} {{
    puts "Count: $i"
}}

# Foreach loop
foreach fruit $fruits {{
    puts "Fruit: $fruit"
}}
```

## Procedures
```tcl
proc greet {{name}} {{
    return "Hello, $name!"
}}

set message [greet "World"]
puts $message
```"#)),
                            "commands" => Ok(format!(r#"# Common TCL Commands in Molt

## String Operations
- `string length $str` - Get string length
- `string index $str $idx` - Get character at index
- `string range $str $start $end` - Extract substring
- `string toupper $str` - Convert to uppercase
- `string tolower $str` - Convert to lowercase

## List Operations
- `list $item1 $item2 ...` - Create list
- `lindex $list $index` - Get list element
- `llength $list` - Get list length
- `lappend listVar $item` - Append to list
- `lrange $list $start $end` - Extract sublist

## Math and Logic
- `expr $expression` - Evaluate mathematical expression
- `incr varName ?increment?` - Increment variable
- `+ - * / %` - Arithmetic operators
- `== != < > <= >=` - Comparison operators
- `&& || !` - Logical operators

## Control Flow
- `if {{condition}} {{...}} else {{...}}` - Conditional
- `for {{init}} {{condition}} {{update}} {{...}}` - For loop
- `foreach var $list {{...}}` - Iterate over list
- `while {{condition}} {{...}}` - While loop
- `break` / `continue` - Loop control

## I/O and Variables
- `puts $string` - Print to stdout
- `set varName $value` - Set variable
- `unset varName` - Delete variable
- `global varName` - Access global variable"#)),
                            "examples" => Ok(format!(r#"# TCL Examples

## Example 1: Calculator
```tcl
proc calculate {{op a b}} {{
    switch $op {{
        "+" {{ return [expr {{$a + $b}}] }}
        "-" {{ return [expr {{$a - $b}}] }}
        "*" {{ return [expr {{$a * $b}}] }}
        "/" {{ 
            if {{$b == 0}} {{
                error "Division by zero"
            }}
            return [expr {{$a / $b}}] 
        }}
        default {{ error "Unknown operation: $op" }}
    }}
}}

puts [calculate + 5 3]    ;# 8
puts [calculate * 4 7]    ;# 28
```

## Example 2: List Processing
```tcl
set numbers [list 1 2 3 4 5]
set sum 0

foreach num $numbers {{
    set sum [expr {{$sum + $num}}]
}}

puts "Sum: $sum"  ;# Sum: 15

# Find maximum
set max [lindex $numbers 0]
foreach num $numbers {{
    if {{$num > $max}} {{
        set max $num
    }}
}}
puts "Max: $max"  ;# Max: 5
```

## Example 3: String Processing
```tcl
proc word_count {{text}} {{
    set words [split $text]
    return [llength $words]
}}

proc reverse_string {{str}} {{
    set result ""
    set len [string length $str]
    for {{set i [expr {{$len - 1}}]}} {{$i >= 0}} {{incr i -1}} {{
        append result [string index $str $i]
    }}
    return $result
}}

puts [word_count "Hello world from TCL"]  ;# 4
puts [reverse_string "hello"]              ;# olleh
```"#)),
                            "links" => Ok(format!(r#"# Molt TCL Documentation Links

## Official Documentation
- **Molt Book**: https://wduquette.github.io/molt/
  Complete guide to the Molt TCL interpreter
  
- **GitHub Repository**: https://github.com/wduquette/molt
  Source code, examples, and issue tracking
  
- **Book Source**: https://github.com/wduquette/molt/tree/master/molt-book/src
  Markdown source files for the Molt Book

## Specific Sections
- **Getting Started**: https://wduquette.github.io/molt/user/getting_started.html
- **Language Reference**: https://wduquette.github.io/molt/ref/
- **Embedding Guide**: https://wduquette.github.io/molt/embed/
- **API Documentation**: https://docs.rs/molt/

## TCL Language Resources
- **TCL/Tk Official**: https://www.tcl.tk/
- **TCL Tutorial**: https://www.tcl.tk/man/tcl8.6/tutorial/
- **TCL Commands**: https://www.tcl.tk/man/tcl8.6/TclCmd/

## Example Code
- **Molt Examples**: https://github.com/wduquette/molt/tree/master/examples
- **Test Suite**: https://github.com/wduquette/molt/tree/master/tests

Note: Molt implements a subset of full TCL but covers the core language features.
For Molt-specific capabilities and limitations, refer to the Molt Book."#)),
                            _ => Err(anyhow::anyhow!("Unknown documentation topic: {}. Available topics: overview, basic_syntax, commands, examples, links", topic))
                        }
                    }
                    "bin__list_tools" => {
                        let request: crate::tcl_tools::TclExecToolRequest = serde_json::from_value(json!({
                            "tool_path": "bin__list_tools",
                            "params": params.arguments
                        }))?;
                        tb.exec_tool(request).await
                    }
                    "bin__inspect_tool" => {
                        let request: crate::tcl_tools::TclExecToolRequest = serde_json::from_value(json!({
                            "tool_path": "bin__inspect_tool", 
                            "params": params.arguments
                        }))?;
                        tb.exec_tool(request).await
                    }
                    "bin__list_namespaces" => {
                        let request: crate::tcl_tools::TclExecToolRequest = serde_json::from_value(json!({
                            "tool_path": "bin__list_namespaces",
                            "params": params.arguments
                        }))?;
                        tb.exec_tool(request).await
                    }
                    "bin__search_tools" => {
                        let request: crate::tcl_tools::TclExecToolRequest = serde_json::from_value(json!({
                            "tool_path": "bin__search_tools",
                            "params": params.arguments
                        }))?;
                        tb.exec_tool(request).await
                    }
                    "bin__list_xmcp_tools" => {
                        let request: crate::tcl_tools::TclExecToolRequest = serde_json::from_value(json!({
                            "tool_path": "bin__list_xmcp_tools",
                            "params": params.arguments
                        }))?;
                        tb.exec_tool(request).await
                    }
                    mcp_name => {
                        // Try to execute as a custom tool
                        tb.execute_custom_tool(mcp_name, params.arguments).await
                    }
                }
                })
            }).join();
            
            match result {
                Ok(Ok(text)) => Ok(json!(McpCallToolResult {
                    content: vec![McpContent::Text { text }],
                })),
                Ok(Err(e)) => Err(jsonrpc_core::Error {
                    code: jsonrpc_core::ErrorCode::InternalError,
                    message: e.to_string(),
                    data: None,
                }),
                Err(_) => Err(jsonrpc_core::Error {
                    code: jsonrpc_core::ErrorCode::InternalError,
                    message: "Thread panic".to_string(),
                    data: None,
                }),
            }
        });

        Self { tool_box, handler }
    }

    pub async fn initialize_persistence(&self) -> Result<()> {
        match self.tool_box.initialize_persistence().await {
            Ok(message) => {
                info!("{}", message);
                Ok(())
            }
            Err(e) => {
                tracing::warn!("Failed to initialize persistence: {}", e);
                Err(e.into())
            }
        }
    }

    pub async fn handle_request(&self, request: serde_json::Value) -> Result<serde_json::Value> {
        let request_str = serde_json::to_string(&request)?;
        let response_str = self
            .handler
            .handle_request(&request_str)
            .await
            .ok_or_else(|| anyhow!("No response from handler"))?;

        serde_json::from_str(&response_str).map_err(|e| anyhow!("Failed to parse response: {}", e))
    }

    pub async fn run_stdio(self) -> Result<()> {
        info!("Starting TCL MCP server on stdio");

        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();
            let n = reader.read_line(&mut line).await?;

            if n == 0 {
                break; // EOF
            }

            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Parse JSON-RPC request
            let request: serde_json::Value = match serde_json::from_str(line) {
                Ok(req) => req,
                Err(e) => {
                    debug!("Failed to parse request: {}", e);
                    continue;
                }
            };

            // Get the request ID before moving the request
            let request_id = request.get("id").cloned();

            // Handle request
            match self.handle_request(request).await {
                Ok(response) => {
                    let response_str = serde_json::to_string(&response)?;
                    stdout.write_all(response_str.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                }
                Err(e) => {
                    debug!("Error handling request: {}", e);
                    // Send error response
                    let error_response = json!({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": e.to_string()
                        },
                        "id": request_id
                    });
                    let response_str = serde_json::to_string(&error_response)?;
                    stdout.write_all(response_str.as_bytes()).await?;
                    stdout.write_all(b"\n").await?;
                    stdout.flush().await?;
                }
            }
        }

        Ok(())
    }
}
