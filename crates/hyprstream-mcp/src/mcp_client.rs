use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::process::Stdio;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command as AsyncCommand};
use tokio::sync::{Mutex, RwLock};
use tokio::time::{timeout, Duration};
use tracing::{debug, error, info, warn};

/// MCP server configuration for registration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub command: String,
    pub args: Vec<String>,
    pub env: HashMap<String, String>,
    pub auto_start: bool,
    pub timeout_ms: u64,
    pub max_retries: u32,
    pub created_at: DateTime<Utc>,
}

/// MCP tool definition from server introspection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    pub name: String,
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// MCP server connection status
#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionStatus {
    Disconnected,
    Connecting,
    Connected,
    Error(String),
}

/// MCP server connection information
#[derive(Debug)]
pub struct McpServerConnection {
    pub config: McpServerConfig,
    pub status: ConnectionStatus,
    pub tools: HashMap<String, McpToolDefinition>,
    pub process: Option<Child>,
    pub stdin: Option<ChildStdin>,
    pub stdout: Option<BufReader<ChildStdout>>,
    pub last_heartbeat: Option<DateTime<Utc>>,
    pub retry_count: u32,
}

/// JSON-RPC request
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: u64,
    method: String,
    params: Option<Value>,
}

/// JSON-RPC response
#[derive(Debug, Deserialize)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    #[allow(dead_code)]
    id: u64,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

/// JSON-RPC error
#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i32,
    message: String,
    #[allow(dead_code)]
    data: Option<Value>,
}

/// MCP client for managing connections to external MCP servers
#[derive(Debug, Clone)]
pub struct McpClient {
    servers: Arc<RwLock<HashMap<String, Arc<Mutex<McpServerConnection>>>>>,
    request_id: Arc<Mutex<u64>>,
}

impl McpClient {
    /// Create a new MCP client
    pub fn new() -> Self {
        Self {
            servers: Arc::new(RwLock::new(HashMap::new())),
            request_id: Arc::new(Mutex::new(1)),
        }
    }

    /// Register a new MCP server
    pub async fn register_server(&self, config: McpServerConfig) -> Result<()> {
        let server_id = config.id.clone();

        // Validate configuration
        if server_id.is_empty() {
            return Err(anyhow!("Server ID cannot be empty"));
        }

        if config.command.is_empty() {
            return Err(anyhow!("Server command cannot be empty"));
        }

        // Create connection
        let connection = McpServerConnection {
            config: config.clone(),
            status: ConnectionStatus::Disconnected,
            tools: HashMap::new(),
            process: None,
            stdin: None,
            stdout: None,
            last_heartbeat: None,
            retry_count: 0,
        };

        // Store connection
        let mut servers = self.servers.write().await;
        servers.insert(server_id.clone(), Arc::new(Mutex::new(connection)));
        drop(servers);

        info!("Registered MCP server: {}", server_id);

        // Auto-start if configured
        if config.auto_start {
            self.connect_server(&server_id).await?;
        }

        Ok(())
    }

    /// Remove an MCP server
    pub async fn remove_server(&self, server_id: &str, force: bool) -> Result<()> {
        let mut servers = self.servers.write().await;

        if let Some(connection_arc) = servers.get(server_id) {
            let mut connection = connection_arc.lock().await;

            // Disconnect if connected
            if connection.status == ConnectionStatus::Connected {
                if !force {
                    // Graceful shutdown
                    if let Err(e) = self.disconnect_server_internal(&mut connection).await {
                        warn!("Error during graceful shutdown of {}: {}", server_id, e);
                    }
                } else {
                    // Force termination
                    if let Some(mut process) = connection.process.take() {
                        if let Err(e) = process.kill().await {
                            warn!("Error force-killing server {}: {}", server_id, e);
                        }
                    }
                }
            }
        }

        servers.remove(server_id);
        info!("Removed MCP server: {}", server_id);
        Ok(())
    }

    /// Connect to an MCP server
    pub async fn connect_server(&self, server_id: &str) -> Result<()> {
        let servers = self.servers.read().await;
        let connection_arc = servers
            .get(server_id)
            .ok_or_else(|| anyhow!("Server not found: {}", server_id))?
            .clone();
        drop(servers);

        let mut connection = connection_arc.lock().await;

        if connection.status == ConnectionStatus::Connected {
            return Ok(());
        }

        connection.status = ConnectionStatus::Connecting;

        // Start server process
        let mut cmd = AsyncCommand::new(&connection.config.command);

        // Setup stderr logging using file redirection for robust handling
        let stderr_config = if std::env::var("TCL_MCP_DEBUG_STDERR").is_ok() {
            // Create logs directory if it doesn't exist
            if let Err(e) = fs::create_dir_all("/tmp/tcl-mcp-logs") {
                warn!("Failed to create MCP logs directory: {}", e);
                Stdio::null()
            } else {
                // Create a log file for this MCP server's stderr
                let log_path = format!("/tmp/tcl-mcp-logs/mcp-{}.stderr.log", server_id);
                match std::fs::File::create(&log_path) {
                    Ok(file) => {
                        debug!(
                            "Created stderr log file for MCP server '{}': {}",
                            server_id, log_path
                        );
                        Stdio::from(file)
                    }
                    Err(e) => {
                        warn!(
                            "Failed to create stderr log file for '{}': {}",
                            server_id, e
                        );
                        Stdio::null()
                    }
                }
            }
        } else {
            // Default: redirect stderr to null to prevent JSON-RPC interference
            Stdio::null()
        };

        cmd.args(&connection.config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(stderr_config);

        // Set environment variables
        for (key, value) in &connection.config.env {
            cmd.env(key, value);
        }

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(e) => {
                connection.status =
                    ConnectionStatus::Error(format!("Failed to start process: {}", e));
                return Err(anyhow!("Failed to start MCP server {}: {}", server_id, e));
            }
        };

        // Take stdin and stdout for communication
        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("Failed to get stdin from process"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("Failed to get stdout from process"))?;

        connection.stdin = Some(stdin);
        connection.stdout = Some(BufReader::new(stdout));
        connection.process = Some(child);

        // Initialize MCP connection
        if let Err(e) = self.initialize_connection(&mut connection).await {
            // Kill the process if initialization failed
            if let Some(mut process) = connection.process.take() {
                let _ = process.kill().await;
            }
            connection.status = ConnectionStatus::Error(e.to_string());
            connection.stdin = None;
            connection.stdout = None;
            return Err(e);
        }

        connection.status = ConnectionStatus::Connected;
        connection.last_heartbeat = Some(Utc::now());
        connection.retry_count = 0;

        info!("Connected to MCP server: {}", server_id);
        Ok(())
    }

    /// Execute a tool on an MCP server
    pub async fn execute_tool(
        &self,
        server_id: &str,
        tool_name: &str,
        params: Value,
    ) -> Result<Value> {
        let servers = self.servers.read().await;
        let connection_arc = servers
            .get(server_id)
            .ok_or_else(|| anyhow!("Server not found: {}", server_id))?
            .clone();
        drop(servers);

        let connection = connection_arc.lock().await;

        if connection.status != ConnectionStatus::Connected {
            return Err(anyhow!("Server {} is not connected", server_id));
        }

        // Check if tool exists
        if !connection.tools.contains_key(tool_name) {
            return Err(anyhow!(
                "Tool '{}' not found on server {}",
                tool_name,
                server_id
            ));
        }

        drop(connection);

        // Execute the tool
        self.call_tool_on_server(server_id, tool_name, params).await
    }

    /// Get list of available tools for a server
    pub async fn get_server_tools(&self, server_id: &str) -> Result<Vec<McpToolDefinition>> {
        let servers = self.servers.read().await;
        let connection_arc = servers
            .get(server_id)
            .ok_or_else(|| anyhow!("Server not found: {}", server_id))?;

        let connection = connection_arc.lock().await;
        Ok(connection.tools.values().cloned().collect())
    }

    /// Get list of all registered servers
    pub async fn list_servers(&self) -> Vec<(String, ConnectionStatus)> {
        let servers = self.servers.read().await;
        let mut result = Vec::new();

        for (id, connection_arc) in servers.iter() {
            if let Ok(connection) = connection_arc.try_lock() {
                result.push((id.clone(), connection.status.clone()));
            } else {
                result.push((id.clone(), ConnectionStatus::Error("Locked".to_string())));
            }
        }

        result
    }

    /// Initialize MCP connection and discover tools
    async fn initialize_connection(&self, connection: &mut McpServerConnection) -> Result<()> {
        let stdin = connection
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("No stdin available"))?;
        let stdout = connection
            .stdout
            .as_mut()
            .ok_or_else(|| anyhow!("No stdout available"))?;

        // Send initialize request
        let init_request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: self.next_request_id().await,
            method: "initialize".to_string(),
            params: Some(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "clientInfo": {
                    "name": "tcl-mcp-server",
                    "version": env!("CARGO_PKG_VERSION")
                }
            })),
        };

        let request_data = serde_json::to_string(&init_request)?;
        debug!("Sending initialize request: {}", request_data);
        stdin
            .write_all(format!("{}\n", request_data).as_bytes())
            .await?;
        stdin.flush().await?;
        debug!("Initialize request sent, waiting for response...");

        // Read initialization response
        let mut response_line = String::new();
        let timeout_duration = Duration::from_millis(connection.config.timeout_ms);

        timeout(timeout_duration, stdout.read_line(&mut response_line))
            .await
            .map_err(|_| anyhow!("Timeout waiting for initialization response"))?
            .map_err(|e| anyhow!("Error reading initialization response: {}", e))?;

        debug!("Received initialize response: {}", response_line.trim());
        let _init_response: JsonRpcResponse = serde_json::from_str(&response_line)?;
        debug!("Initialize response parsed successfully");

        // Send initialized notification
        let init_notification = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });

        let notification_data = serde_json::to_string(&init_notification)?;
        debug!("Sending initialized notification: {}", notification_data);
        stdin
            .write_all(format!("{}\n", notification_data).as_bytes())
            .await?;
        stdin.flush().await?;
        debug!("Initialized notification sent");

        // Add a small delay after initialized notification
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Discover tools - create request without params field for tools/list
        let tools_request = serde_json::json!({
            "jsonrpc": "2.0",
            "id": self.next_request_id().await,
            "method": "tools/list"
        });

        let request_data = serde_json::to_string(&tools_request)?;
        debug!("Sending tools/list request: {}", request_data);
        stdin
            .write_all(format!("{}\n", request_data).as_bytes())
            .await?;
        stdin.flush().await?;
        debug!("Tools/list request sent, waiting for response...");

        // Read tools response
        response_line.clear();
        timeout(timeout_duration, stdout.read_line(&mut response_line))
            .await
            .map_err(|_| anyhow!("Timeout waiting for tools list response"))?
            .map_err(|e| anyhow!("Error reading tools list response: {}", e))?;

        debug!("Received tools/list response: {}", response_line.trim());
        let tools_response: JsonRpcResponse = serde_json::from_str(&response_line)?;

        if let Some(error) = tools_response.error {
            return Err(anyhow!("Tools list error: {}", error.message));
        }

        if let Some(result) = tools_response.result {
            if let Some(tools_array) = result.get("tools").and_then(|t| t.as_array()) {
                debug!("Found {} tools in response", tools_array.len());
                for tool in tools_array {
                    match serde_json::from_value::<McpToolDefinition>(tool.clone()) {
                        Ok(tool_def) => {
                            debug!("Successfully parsed tool: {}", tool_def.name);
                            connection.tools.insert(tool_def.name.clone(), tool_def);
                        }
                        Err(e) => {
                            warn!("Failed to parse tool definition: {}", e);
                            debug!(
                                "Tool JSON: {}",
                                serde_json::to_string_pretty(tool).unwrap_or_default()
                            );
                        }
                    }
                }
            } else {
                warn!("No tools array found in response");
            }
        } else {
            warn!("No result in tools response");
        }

        info!(
            "Discovered {} tools for server {}",
            connection.tools.len(),
            connection.config.id
        );
        Ok(())
    }

    /// Call a tool on a server
    async fn call_tool_on_server(
        &self,
        server_id: &str,
        tool_name: &str,
        params: Value,
    ) -> Result<Value> {
        let servers = self.servers.read().await;
        let connection_arc = servers
            .get(server_id)
            .ok_or_else(|| anyhow!("Server not found: {}", server_id))?
            .clone();
        drop(servers);

        let mut connection = connection_arc.lock().await;

        // Ensure we're connected
        if connection.status != ConnectionStatus::Connected {
            return Err(anyhow!("Server {} is not connected", server_id));
        }

        // Get timeout duration before any borrows
        let timeout_duration = Duration::from_millis(connection.config.timeout_ms);

        // Create tool call request
        let tool_request = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: self.next_request_id().await,
            method: "tools/call".to_string(),
            params: Some(serde_json::json!({
                "name": tool_name,
                "arguments": params
            })),
        };

        // Send request
        let request_data = serde_json::to_string(&tool_request)?;
        debug!(
            "Sending tool call request to {}: {}",
            server_id, request_data
        );

        // Send the request
        if let Some(stdin) = connection.stdin.as_mut() {
            stdin
                .write_all(format!("{}\n", request_data).as_bytes())
                .await?;
            stdin.flush().await?;
        } else {
            return Err(anyhow!("No stdin available for server {}", server_id));
        }

        // Read response with timeout
        let mut response_line = String::new();

        // Read the response
        if let Some(stdout) = connection.stdout.as_mut() {
            match timeout(timeout_duration, stdout.read_line(&mut response_line)).await {
                Ok(Ok(_)) => {
                    debug!(
                        "Received response from {}: {}",
                        server_id,
                        response_line.trim()
                    );

                    let response: JsonRpcResponse =
                        serde_json::from_str(&response_line).map_err(|e| {
                            anyhow!("Failed to parse response from {}: {}", server_id, e)
                        })?;

                    // Check for errors
                    if let Some(error) = response.error {
                        return Err(anyhow!(
                            "Tool execution error on {}: {} (code: {})",
                            server_id,
                            error.message,
                            error.code
                        ));
                    }

                    // Return the result
                    response
                        .result
                        .ok_or_else(|| anyhow!("No result in response from {}", server_id))
                }
                Ok(Err(e)) => {
                    error!("Error reading response from {}: {}", server_id, e);
                    connection.status = ConnectionStatus::Error(format!("Read error: {}", e));
                    Err(anyhow!("Error reading response from {}: {}", server_id, e))
                }
                Err(_) => {
                    error!(
                        "Timeout waiting for response from {} ({}ms)",
                        server_id,
                        timeout_duration.as_millis()
                    );
                    connection.status = ConnectionStatus::Error("Timeout".to_string());
                    Err(anyhow!(
                        "Timeout waiting for response from {} after {}ms",
                        server_id,
                        timeout_duration.as_millis()
                    ))
                }
            }
        } else {
            Err(anyhow!("No stdout available for server {}", server_id))
        }
    }

    /// Disconnect from a server
    async fn disconnect_server_internal(&self, connection: &mut McpServerConnection) -> Result<()> {
        // Clean up stdin/stdout
        connection.stdin = None;
        connection.stdout = None;

        // Kill the process
        if let Some(mut process) = connection.process.take() {
            // Try graceful shutdown first
            if let Some(mut stdin) = connection.stdin.take() {
                let shutdown_notification = serde_json::json!({
                    "jsonrpc": "2.0",
                    "method": "shutdown"
                });

                if let Ok(data) = serde_json::to_string(&shutdown_notification) {
                    let _ = stdin.write_all(format!("{}\n", data).as_bytes()).await;
                    let _ = stdin.flush().await;
                }
            }

            // Give it a moment to shut down gracefully
            tokio::time::sleep(Duration::from_millis(100)).await;

            // Force kill if still running
            let _ = process.kill().await;
        }

        connection.status = ConnectionStatus::Disconnected;
        connection.tools.clear();
        connection.last_heartbeat = None;

        Ok(())
    }

    /// Get next request ID
    async fn next_request_id(&self) -> u64 {
        let mut id = self.request_id.lock().await;
        let current = *id;
        *id += 1;
        current
    }

    /// Reconnect to a server if disconnected
    pub async fn reconnect_server(&self, server_id: &str) -> Result<()> {
        let servers = self.servers.read().await;
        let connection_arc = servers
            .get(server_id)
            .ok_or_else(|| anyhow!("Server not found: {}", server_id))?
            .clone();
        drop(servers);

        let mut connection = connection_arc.lock().await;

        // Only reconnect if disconnected or in error state
        match connection.status {
            ConnectionStatus::Connected => {
                info!("Server {} is already connected", server_id);
                Ok(())
            }
            _ => {
                info!("Reconnecting to server {}", server_id);
                connection.retry_count += 1;

                // Clean up any existing resources
                connection.stdin = None;
                connection.stdout = None;
                if let Some(mut process) = connection.process.take() {
                    let _ = process.kill().await;
                }

                drop(connection);
                self.connect_server(server_id).await
            }
        }
    }

    /// Check server health
    pub async fn check_server_health(&self, server_id: &str) -> Result<bool> {
        let servers = self.servers.read().await;
        let connection_arc = servers
            .get(server_id)
            .ok_or_else(|| anyhow!("Server not found: {}", server_id))?
            .clone();
        drop(servers);

        let connection = connection_arc.lock().await;
        Ok(connection.status == ConnectionStatus::Connected)
    }

    /// Get server status
    pub async fn get_server_status(&self, server_id: &str) -> Result<ConnectionStatus> {
        let servers = self.servers.read().await;
        let connection_arc = servers
            .get(server_id)
            .ok_or_else(|| anyhow!("Server not found: {}", server_id))?
            .clone();
        drop(servers);

        let connection = connection_arc.lock().await;
        Ok(connection.status.clone())
    }

    /// Manually connect to a server (for debugging)
    pub async fn debug_connect_server(&self, server_id: &str) -> Result<String> {
        let start_time = std::time::Instant::now();

        // Get server info before connection attempt
        let servers = self.servers.read().await;
        let connection_arc = servers
            .get(server_id)
            .ok_or_else(|| anyhow!("Server not found: {}", server_id))?
            .clone();
        drop(servers);

        let connection = connection_arc.lock().await;
        let command = format!(
            "{} {}",
            connection.config.command,
            connection.config.args.join(" ")
        );
        let timeout_ms = connection.config.timeout_ms;
        drop(connection);

        info!(
            "Debug: Attempting to connect to server '{}' with command: {}",
            server_id, command
        );
        info!("Debug: Timeout set to: {}ms", timeout_ms);

        match self.connect_server(server_id).await {
            Ok(_) => {
                let duration = start_time.elapsed();
                Ok(format!(
                    "Successfully connected to server '{}' in {}ms\nCommand: {}\nTimeout: {}ms",
                    server_id,
                    duration.as_millis(),
                    command,
                    timeout_ms
                ))
            }
            Err(e) => {
                let duration = start_time.elapsed();
                Ok(format!(
                    "Failed to connect to server '{}' after {}ms: {}\nCommand: {}\nTimeout: {}ms",
                    server_id,
                    duration.as_millis(),
                    e,
                    command,
                    timeout_ms
                ))
            }
        }
    }

    /// Disconnect from a server (for debugging)
    pub async fn debug_disconnect_server(&self, server_id: &str) -> Result<String> {
        let servers = self.servers.read().await;
        let connection_arc = servers
            .get(server_id)
            .ok_or_else(|| anyhow!("Server not found: {}", server_id))?
            .clone();
        drop(servers);

        let mut connection = connection_arc.lock().await;

        if connection.status == ConnectionStatus::Connected {
            if let Err(e) = self.disconnect_server_internal(&mut connection).await {
                return Ok(format!("Error during disconnect: {}", e));
            }
            connection.status = ConnectionStatus::Disconnected;
            Ok(format!(
                "Successfully disconnected from server '{}'",
                server_id
            ))
        } else {
            Ok(format!(
                "Server '{}' is not connected (status: {:?})",
                server_id, connection.status
            ))
        }
    }

    /// Get detailed server information (for debugging)
    pub async fn debug_server_info(&self, server_id: &str) -> Result<String> {
        let servers = self.servers.read().await;
        let connection_arc = servers
            .get(server_id)
            .ok_or_else(|| anyhow!("Server not found: {}", server_id))?
            .clone();
        drop(servers);

        let connection = connection_arc.lock().await;

        let mut info = vec![
            format!("Server ID: {}", connection.config.id),
            format!("Name: {}", connection.config.name),
            format!("Command: {}", connection.config.command),
            format!("Args: {:?}", connection.config.args),
            format!("Status: {:?}", connection.status),
            format!("Auto-start: {}", connection.config.auto_start),
            format!("Timeout: {}ms", connection.config.timeout_ms),
            format!("Max retries: {}", connection.config.max_retries),
            format!("Retry count: {}", connection.retry_count),
            format!("Tools discovered: {}", connection.tools.len()),
        ];

        if let Some(last_heartbeat) = connection.last_heartbeat {
            info.push(format!(
                "Last heartbeat: {}",
                last_heartbeat.format("%Y-%m-%d %H:%M:%S UTC")
            ));
        }

        if !connection.tools.is_empty() {
            info.push("Available tools:".to_string());
            for tool_name in connection.tools.keys() {
                info.push(format!("  - {}", tool_name));
            }
        }

        Ok(info.join("\n"))
    }

    /// Test basic communication with a server (for debugging)
    pub async fn debug_ping_server(&self, server_id: &str) -> Result<String> {
        let servers = self.servers.read().await;
        let connection_arc = servers
            .get(server_id)
            .ok_or_else(|| anyhow!("Server not found: {}", server_id))?
            .clone();
        drop(servers);

        let connection = connection_arc.lock().await;

        if connection.status != ConnectionStatus::Connected {
            return Ok(format!(
                "Cannot ping server '{}': not connected (status: {:?})",
                server_id, connection.status
            ));
        }

        // Try to get tools list as a basic ping
        drop(connection);
        match self.get_server_tools(server_id).await {
            Ok(tools) => Ok(format!(
                "Server '{}' is responsive. {} tools available.",
                server_id,
                tools.len()
            )),
            Err(e) => Ok(format!("Server '{}' ping failed: {}", server_id, e)),
        }
    }
}

impl Default for McpClient {
    fn default() -> Self {
        Self::new()
    }
}
