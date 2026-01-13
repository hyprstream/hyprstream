use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::info;
use chrono::Utc;
use std::collections::HashMap;

mod advanced_tools;
mod mcp_client;
mod mcp_persistence;
mod namespace;
mod path_format;
mod persistence;
mod platform_dirs;
mod server;
mod tcl_executor;
mod tcl_runtime;
mod tcl_tools;
mod tool_discovery;
mod tool_registry;

use server::TclMcpServer;
use mcp_client::{McpClient, McpServerConfig};
use mcp_persistence::McpPersistence;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
#[command(about = "TCL MCP Server - Execute TCL scripts via Model Context Protocol")]
struct Args {
    /// Enable privileged mode (full TCL language access and tool management)
    #[arg(
        long,
        help = "Enable privileged mode with full TCL access and tool management capabilities"
    )]
    privileged: bool,

    /// Select TCL runtime implementation
    #[arg(
        long,
        value_name = "RUNTIME",
        help = "TCL runtime to use (molt|tcl). Can also be set via TCL_MCP_RUNTIME environment variable"
    )]
    runtime: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a specific tool by name
    Run {
        /// Tool name (e.g., "tcl_execute", "simple_calc", or full name like "bin___tcl_execute")
        tool: String,
        /// Arguments to pass to the tool (JSON or key=value format, e.g., 'script=puts "Hello"' or '{"script": "puts \"Hello\""}')
        #[arg(value_parser = parse_json_args, help = "Arguments to pass to the tool. Supports JSON format or comma-separated key=value pairs.\n\nExamples:\n  - Key=value: 'script=puts \"Hello\"'\n  - Multiple args: 'operation=add,a=5,b=3'\n  - JSON: '{\"script\": \"puts \\\"Hello\\\"\"}'")]
        args: Option<serde_json::Value>,
    },
    /// List all available tools
    List {
        /// Filter tools by namespace (e.g., "bin", "sbin", "user")
        #[arg(short, long)]
        namespace: Option<String>,
        /// Filter tools by pattern
        #[arg(short, long)]
        filter: Option<String>,
    },
    /// Get information about a specific tool
    Info {
        /// Tool name to get information about
        tool: String,
    },
    /// Start the MCP server (default behavior)
    Server,
    /// Manage downstream MCP clients
    Mcp {
        #[command(subcommand)]
        command: McpCommands,
    },
}

#[derive(Subcommand)]
enum McpCommands {
    /// Add a new MCP server
    Add {
        /// Unique identifier for the server
        id: String,
        /// Human-readable name for the server
        name: String,
        /// Optional description of the server
        #[arg(long)]
        description: Option<String>,
        /// Environment variables in KEY=VALUE format
        #[arg(long, help = "Environment variables in KEY=VALUE format (can be specified multiple times)")]
        env: Vec<String>,
        /// Whether to auto-start the server
        #[arg(long, default_value = "true")]
        auto_start: bool,
        /// Connection timeout in milliseconds
        #[arg(long, default_value = "30000")]
        timeout_ms: u64,
        /// Maximum retry attempts
        #[arg(long, default_value = "3")]
        max_retries: u32,
        /// Command and arguments (everything after options)
        #[arg(last = true, help = "Command and arguments to start the server (use -- to separate if needed)")]
        command_args: Vec<String>,
    },
    /// Remove an MCP server
    Remove {
        /// Server ID to remove
        id: String,
        /// Whether to force removal (kill process)
        #[arg(long)]
        force: bool,
    },
    /// List all registered MCP servers
    List {
        /// Show detailed information
        #[arg(long)]
        detailed: bool,
    },
    /// Get information about a specific MCP server
    Info {
        /// Server ID to get information about
        id: String,
    },
    /// Connect to an MCP server
    Connect {
        /// Server ID to connect to
        id: String,
    },
    /// Disconnect from an MCP server
    Disconnect {
        /// Server ID to disconnect from
        id: String,
    },
    /// Test connectivity to an MCP server
    Ping {
        /// Server ID to ping
        id: String,
    },
}

fn parse_json_args(s: &str) -> Result<serde_json::Value, String> {
    // Try to parse as JSON first
    if let Ok(json) = serde_json::from_str(s) {
        return Ok(json);
    }

    // If not valid JSON, try to interpret as key=value pairs
    if s.contains('=') {
        let mut map = serde_json::Map::new();
        for pair in s.split(',') {
            let parts: Vec<&str> = pair.splitn(2, '=').collect();
            if parts.len() == 2 {
                let key = parts[0].trim();
                let value = parts[1].trim();

                // Try to parse value as number or boolean, otherwise use as string
                let json_value = if let Ok(n) = value.parse::<i64>() {
                    serde_json::json!(n)
                } else if let Ok(f) = value.parse::<f64>() {
                    serde_json::json!(f)
                } else if let Ok(b) = value.parse::<bool>() {
                    serde_json::json!(b)
                } else {
                    serde_json::json!(value)
                };

                map.insert(key.to_string(), json_value);
            }
        }
        return Ok(serde_json::Value::Object(map));
    }

    // Otherwise, treat as a single string argument
    Ok(serde_json::json!({ "value": s }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    // Determine runtime configuration
    let env_runtime = std::env::var("TCL_MCP_RUNTIME").ok();
    let runtime_config = match tcl_runtime::RuntimeConfig::from_args_and_env(
        args.runtime.as_deref(),
        env_runtime.as_deref(),
    ) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    };

    // Show available runtimes if requested runtime is not available
    let requested_available = runtime_config
        .runtime_type
        .as_ref()
        .map(|rt| rt.is_available())
        .unwrap_or(true);
    if !requested_available {
        let available = tcl_runtime::RuntimeConfig::available_runtimes();
        let available_names: Vec<&str> = available.iter().map(|r| r.as_str()).collect();
        eprintln!(
            "Warning: {} runtime not available. Available runtimes: {}",
            runtime_config
                .runtime_type
                .as_ref()
                .map(|rt| rt.as_str())
                .unwrap_or("unknown"),
            available_names.join(", ")
        );
    }

    // Handle different commands
    match args.command {
        Some(Commands::Run {
            tool,
            args: tool_args,
        }) => run_tool(&tool, tool_args, args.privileged, runtime_config).await,
        Some(Commands::List { namespace, filter }) => {
            list_tools(namespace, filter, args.privileged, runtime_config).await
        }
        Some(Commands::Info { tool }) => tool_info(&tool, args.privileged, runtime_config).await,
        Some(Commands::Mcp { command }) => handle_mcp_command(command).await,
        Some(Commands::Server) | None => {
            // Default behavior - run as MCP server
            if args.privileged {
                info!("Starting TCL MCP Server in PRIVILEGED mode - full TCL access enabled");
            } else {
                info!("Starting TCL MCP Server in RESTRICTED mode - limited TCL access");
            }

            // Create and run the MCP server with privilege and runtime settings
            let server = match TclMcpServer::new_with_runtime(args.privileged, runtime_config) {
                Ok(server) => server,
                Err(e) => {
                    eprintln!("Failed to create server: {}", e);
                    std::process::exit(1);
                }
            };

            // Initialize persistence (load existing tools)
            if let Err(e) = server.initialize_persistence().await {
                tracing::warn!("Failed to initialize persistence: {}", e);
                // Continue without persistence rather than failing
            }

            // Handle stdio communication
            server.run_stdio().await?;
            Ok(())
        }
    }
}

async fn run_tool(
    tool_name: &str,
    args: Option<serde_json::Value>,
    privileged: bool,
    runtime_config: tcl_runtime::RuntimeConfig,
) -> Result<()> {
    // Create a simple server instance just to run the tool
    let server = TclMcpServer::new_with_runtime(privileged, runtime_config.clone())
        .map_err(|e| anyhow::anyhow!("Failed to create server: {}", e))?;

    // Initialize persistence
    if let Err(e) = server.initialize_persistence().await {
        tracing::warn!("Failed to initialize persistence: {}", e);
    }

    // Convert tool name to MCP format if needed
    let mcp_tool_name = if tool_name.contains("__") {
        // Already in MCP format
        tool_name.to_string()
    } else {
        // Try to find the tool by searching for it
        // First, try with common namespace prefixes
        let candidates = vec![
            format!("bin__{}", tool_name),
            format!("sbin__{}", tool_name),
            format!("docs__{}", tool_name),
            format!("user__{}", tool_name),
            format!("debug__{}", tool_name),
            format!("test__{}", tool_name),
        ];

        // Get the list of tools to find the exact match
        let tools_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        });

        if let Ok(response) = server.handle_request(tools_request).await {
            if let Some(result) = response.get("result") {
                if let Some(tools) = result.get("tools").and_then(|t| t.as_array()) {
                    // First try exact matches with common namespaces
                    for candidate in &candidates {
                        if tools
                            .iter()
                            .any(|t| t.get("name").and_then(|n| n.as_str()) == Some(candidate))
                        {
                            break;
                        }
                    }

                    // If no exact match, look for any tool ending with the name
                    for tool in tools {
                        if let Some(name) = tool.get("name").and_then(|n| n.as_str()) {
                            if name.ends_with(&format!("__{}", tool_name)) {
                                // Found the tool - recursively call with full name
                                return Box::pin(run_tool(&name, args, privileged, runtime_config))
                                    .await;
                            }
                        }
                    }
                }
            }
        }

        // Default to bin namespace if not found
        format!("bin__{}", tool_name)
    };

    // Build the JSON-RPC request
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": mcp_tool_name,
            "arguments": args.unwrap_or(serde_json::json!({}))
        }
    });

    // Process the request
    match server.handle_request(request).await {
        Ok(response) => {
            // Extract and print the result
            if let Some(result) = response.get("result") {
                if let Some(content) = result.get("content") {
                    if let Some(array) = content.as_array() {
                        for item in array {
                            if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                                println!("{}", text);
                            }
                        }
                    } else if let Some(text) = content.as_str() {
                        println!("{}", text);
                    }
                } else {
                    // Print raw result
                    println!("{}", serde_json::to_string_pretty(result)?);
                }
            } else if let Some(error) = response.get("error") {
                eprintln!("Error: {}", serde_json::to_string_pretty(error)?);
                std::process::exit(1);
            }
            Ok(())
        }
        Err(e) => {
            eprintln!("Error executing tool '{}': {}", tool_name, e);
            std::process::exit(1);
        }
    }
}

async fn list_tools(
    namespace_filter: Option<String>,
    pattern_filter: Option<String>,
    privileged: bool,
    runtime_config: tcl_runtime::RuntimeConfig,
) -> Result<()> {
    // Use the tcl_tool_list tool
    let mut args = serde_json::json!({});

    if let Some(ns) = namespace_filter {
        args["namespace"] = serde_json::json!(ns);
    }

    if let Some(filter) = pattern_filter {
        args["filter"] = serde_json::json!(filter);
    }

    run_tool("bin__list_tools", Some(args), privileged, runtime_config).await
}

async fn tool_info(
    tool_name: &str,
    privileged: bool,
    runtime_config: tcl_runtime::RuntimeConfig,
) -> Result<()> {
    // Create a server to get tool info
    let server = TclMcpServer::new_with_runtime(privileged, runtime_config)
        .map_err(|e| anyhow::anyhow!("Failed to create server: {}", e))?;

    // Initialize persistence
    if let Err(e) = server.initialize_persistence().await {
        tracing::warn!("Failed to initialize persistence: {}", e);
    }

    // Get the list of tools
    let tools_request = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list"
    });

    match server.handle_request(tools_request).await {
        Ok(response) => {
            if let Some(result) = response.get("result") {
                if let Some(tools) = result.get("tools").and_then(|t| t.as_array()) {
                    // Find the tool
                    for tool in tools {
                        if let Some(name) = tool.get("name").and_then(|n| n.as_str()) {
                            if name == tool_name || name.ends_with(&format!("__{}", tool_name)) {
                                // Found it - print info
                                println!("Tool: {}", name);

                                if let Some(desc) = tool.get("description").and_then(|d| d.as_str())
                                {
                                    println!("Description: {}", desc);
                                }

                                if let Some(schema) = tool.get("inputSchema") {
                                    println!("\nParameters:");
                                    if let Some(props) =
                                        schema.get("properties").and_then(|p| p.as_object())
                                    {
                                        for (param_name, prop_schema) in props {
                                            let desc = prop_schema
                                                .get("description")
                                                .and_then(|d| d.as_str())
                                                .unwrap_or("");
                                            let prop_type = prop_schema
                                                .get("type")
                                                .and_then(|t| t.as_str())
                                                .unwrap_or("unknown");

                                            let required = schema
                                                .get("required")
                                                .and_then(|r| r.as_array())
                                                .map(|arr| {
                                                    arr.iter()
                                                        .any(|v| v.as_str() == Some(param_name))
                                                })
                                                .unwrap_or(false);

                                            println!(
                                                "  {} ({}) {} - {}",
                                                param_name,
                                                prop_type,
                                                if required { "[required]" } else { "[optional]" },
                                                desc
                                            );
                                        }
                                    } else {
                                        println!("  No parameters required");
                                    }
                                }

                                return Ok(());
                            }
                        }
                    }

                    eprintln!("Tool '{}' not found", tool_name);
                    std::process::exit(1);
                } else {
                    eprintln!("No tools found");
                    std::process::exit(1);
                }
            } else if let Some(error) = response.get("error") {
                eprintln!("Error: {}", serde_json::to_string_pretty(error)?);
                std::process::exit(1);
            }

            Ok(())
        }
        Err(e) => {
            eprintln!("Error getting tool info: {}", e);
            std::process::exit(1);
        }
    }
}

async fn handle_mcp_command(command: McpCommands) -> Result<()> {
    match command {
        McpCommands::Add {
            id,
            name,
            description,
            env,
            auto_start,
            timeout_ms,
            max_retries,
            command_args,
        } => add_mcp_server(id, name, command_args, description, env, auto_start, timeout_ms, max_retries).await,
        McpCommands::Remove { id, force } => remove_mcp_server(id, force).await,
        McpCommands::List { detailed } => list_mcp_servers(detailed).await,
        McpCommands::Info { id } => mcp_server_info(id).await,
        McpCommands::Connect { id } => connect_mcp_server(id).await,
        McpCommands::Disconnect { id } => disconnect_mcp_server(id).await,
        McpCommands::Ping { id } => ping_mcp_server(id).await,
    }
}

async fn add_mcp_server(
    id: String,
    name: String,
    command_args: Vec<String>,
    description: Option<String>,
    env: Vec<String>,
    auto_start: bool,
    timeout_ms: u64,
    max_retries: u32,
) -> Result<()> {
    // Parse command and arguments
    if command_args.is_empty() {
        eprintln!("Error: No command provided");
        std::process::exit(1);
    }
    
    let command = command_args[0].clone();
    let args = command_args[1..].to_vec();

    // Parse environment variables
    let mut env_map = HashMap::new();
    for env_var in env {
        if let Some((key, value)) = env_var.split_once('=') {
            env_map.insert(key.to_string(), value.to_string());
        } else {
            eprintln!("Warning: Invalid environment variable format: {}", env_var);
        }
    }

    // Create server configuration
    let config = McpServerConfig {
        id: id.clone(),
        name,
        description,
        command,
        args,
        env: env_map,
        auto_start,
        timeout_ms,
        max_retries,
        created_at: Utc::now(),
    };

    // Save to persistence
    let mut persistence = McpPersistence::new().await?;
    persistence.save_server(id.clone(), config.clone(), auto_start).await?;

    // Create MCP client and register server
    let client = McpClient::new();
    if let Err(e) = client.register_server(config).await {
        eprintln!("Warning: Failed to register server with MCP client: {}", e);
        eprintln!("The server configuration has been saved, but auto-start may not work until the server is available.");
    }

    println!("✓ MCP server '{}' added successfully", id);
    if auto_start {
        println!("  Auto-start is enabled - the server will start automatically");
    }
    Ok(())
}

async fn remove_mcp_server(id: String, force: bool) -> Result<()> {
    // Remove from persistence
    let mut persistence = McpPersistence::new().await?;
    let removed = persistence.remove_server(&id).await?;

    if !removed {
        eprintln!("Server '{}' not found in configuration", id);
        std::process::exit(1);
    }

    // Remove from client (if running)
    let client = McpClient::new();
    if let Err(e) = client.remove_server(&id, force).await {
        eprintln!("Warning: Error removing server from client: {}", e);
    }

    println!("✓ MCP server '{}' removed successfully", id);
    Ok(())
}

async fn list_mcp_servers(detailed: bool) -> Result<()> {
    let persistence = McpPersistence::new().await?;
    let servers = persistence.list_servers();

    if servers.is_empty() {
        println!("No MCP servers configured");
        return Ok(());
    }

    if detailed {
        println!("Registered MCP servers:");
        println!();
        for (id, entry) in servers {
            println!("Server: {}", id);
            println!("  Name: {}", entry.config.name);
            println!("  Command: {}", entry.config.command);
            if !entry.config.args.is_empty() {
                println!("  Args: {:?}", entry.config.args);
            }
            if let Some(desc) = &entry.config.description {
                println!("  Description: {}", desc);
            }
            println!("  Auto-start: {}", entry.auto_start);
            println!("  Timeout: {}ms", entry.config.timeout_ms);
            println!("  Max retries: {}", entry.config.max_retries);
            println!("  Added: {}", entry.added_at.format("%Y-%m-%d %H:%M:%S UTC"));
            println!("  Updated: {}", entry.updated_at.format("%Y-%m-%d %H:%M:%S UTC"));
            if !entry.config.env.is_empty() {
                println!("  Environment:");
                for (key, value) in &entry.config.env {
                    println!("    {}={}", key, value);
                }
            }
            println!();
        }
    } else {
        println!("Registered MCP servers:");
        for (id, entry) in servers {
            let status_indicator = if entry.auto_start { "🟢" } else { "⚪" };
            println!("  {} {} - {}", status_indicator, id, entry.config.name);
        }
        println!();
        println!("Legend: 🟢 = auto-start enabled, ⚪ = manual start");
    }

    Ok(())
}

async fn mcp_server_info(id: String) -> Result<()> {
    let persistence = McpPersistence::new().await?;
    
    if let Some(entry) = persistence.get_server(&id) {
        println!("MCP Server Information");
        println!("=====================");
        println!("ID: {}", id);
        println!("Name: {}", entry.config.name);
        println!("Command: {}", entry.config.command);
        
        if !entry.config.args.is_empty() {
            println!("Arguments: {:?}", entry.config.args);
        }
        
        if let Some(desc) = &entry.config.description {
            println!("Description: {}", desc);
        }
        
        println!("Auto-start: {}", entry.auto_start);
        println!("Timeout: {}ms", entry.config.timeout_ms);
        println!("Max retries: {}", entry.config.max_retries);
        println!("Added: {}", entry.added_at.format("%Y-%m-%d %H:%M:%S UTC"));
        println!("Updated: {}", entry.updated_at.format("%Y-%m-%d %H:%M:%S UTC"));
        
        if !entry.config.env.is_empty() {
            println!("Environment variables:");
            for (key, value) in &entry.config.env {
                println!("  {}={}", key, value);
            }
        }
        
        // Try to get runtime status
        let client = McpClient::new();
        if let Ok(status) = client.get_server_status(&id).await {
            println!("Runtime status: {:?}", status);
        }
    } else {
        eprintln!("Server '{}' not found", id);
        std::process::exit(1);
    }

    Ok(())
}

async fn connect_mcp_server(id: String) -> Result<()> {
    let persistence = McpPersistence::new().await?;
    
    if let Some(entry) = persistence.get_server(&id) {
        let client = McpClient::new();
        client.register_server(entry.config.clone()).await?;
        
        match client.debug_connect_server(&id).await {
            Ok(result) => {
                println!("{}", result);
            }
            Err(e) => {
                eprintln!("Failed to connect to server '{}': {}", id, e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("Server '{}' not found", id);
        std::process::exit(1);
    }

    Ok(())
}

async fn disconnect_mcp_server(id: String) -> Result<()> {
    let client = McpClient::new();
    
    match client.debug_disconnect_server(&id).await {
        Ok(result) => {
            println!("{}", result);
        }
        Err(e) => {
            eprintln!("Failed to disconnect from server '{}': {}", id, e);
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn ping_mcp_server(id: String) -> Result<()> {
    let persistence = McpPersistence::new().await?;
    
    if let Some(entry) = persistence.get_server(&id) {
        let client = McpClient::new();
        client.register_server(entry.config.clone()).await?;
        
        match client.debug_ping_server(&id).await {
            Ok(result) => {
                println!("{}", result);
            }
            Err(e) => {
                eprintln!("Failed to ping server '{}': {}", id, e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("Server '{}' not found", id);
        std::process::exit(1);
    }

    Ok(())
}
