use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};
use tracing::info;

use crate::tcl_executor::TclCommand;

use crate::mcp_client::McpServerConfig;
use crate::namespace::ToolPath;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub path: ToolPath,
    pub description: String,
    pub script: String,
    pub parameters: Vec<ParameterDefinition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDefinition {
    pub name: String,
    pub description: String,
    pub required: bool,
    pub type_name: String,
}

#[derive(Clone)]
pub struct TclToolBox {
    executor: mpsc::Sender<TclCommand>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TclExecuteRequest {
    /// TCL script to execute
    pub script: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TclToolAddRequest {
    /// User namespace (required for user tools)
    pub user: String,
    /// Package name (required for user tools)
    pub package: String,
    /// Name of the new tool
    pub name: String,
    /// Version of the tool (defaults to "latest")
    #[serde(default = "default_version")]
    pub version: String,
    /// Description of what the tool does
    pub description: String,
    /// TCL script that implements the tool
    pub script: String,
    /// Parameters that the tool accepts
    #[serde(default)]
    pub parameters: Vec<ParameterDefinition>,
}

fn default_version() -> String {
    "latest".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TclToolRemoveRequest {
    /// Full tool path (e.g., "user__alice__utils__reverse_string__v1_0")
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TclToolListRequest {
    /// Filter tools by namespace (optional)
    #[serde(default)]
    pub namespace: Option<String>,
    /// Filter tools by name pattern (optional)
    #[serde(default)]
    pub filter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TclExecToolRequest {
    /// Tool path to execute (e.g., "bin__list_dir")
    pub tool_path: String,
    /// Parameters to pass to the tool
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpExecuteRequest {
    /// MCP server ID to execute tool on
    pub server_id: String,
    /// Tool name to execute
    pub tool_name: String,
    /// Parameters to pass to the tool
    #[serde(default)]
    pub params: serde_json::Value,
    /// Response format (json, text, auto)
    #[serde(default = "default_response_format")]
    pub response_format: String,
    /// Timeout in milliseconds
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerAddRequest {
    /// Unique identifier for the server
    pub id: String,
    /// Human-readable name for the server
    pub name: String,
    /// Optional description of the server
    pub description: Option<String>,
    /// Command to start the server
    pub command: String,
    /// Command line arguments
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment variables
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
    /// Whether to auto-start the server
    #[serde(default = "default_auto_start")]
    pub auto_start: bool,
    /// Connection timeout in milliseconds
    #[serde(default = "default_timeout")]
    pub timeout_ms: u64,
    /// Maximum retry attempts
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerRemoveRequest {
    /// Server ID to remove
    pub server_id: String,
    /// Whether to force removal (kill process)
    #[serde(default)]
    pub force: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpDebugRequest {
    /// Server ID to debug
    pub server_id: String,
}

fn default_response_format() -> String {
    "auto".to_string()
}

fn default_timeout() -> u64 {
    30000
}

fn default_auto_start() -> bool {
    true
}

fn default_max_retries() -> u32 {
    3
}

impl TclToolBox {
    pub fn new(executor: mpsc::Sender<TclCommand>) -> Self {
        Self { executor }
    }

    pub async fn tcl_execute(&self, request: TclExecuteRequest) -> Result<String> {
        info!("Executing TCL script: {}", request.script);

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::Execute {
                script: request.script,
                response: tx,
            })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn tcl_tool_add(&self, request: TclToolAddRequest) -> Result<String> {
        let path = ToolPath::user(
            &request.user,
            &request.package,
            &request.name,
            &request.version,
        );
        info!("Adding new TCL tool: {}", path);

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::AddTool {
                path,
                description: request.description,
                script: request.script,
                parameters: request.parameters,
                response: tx,
            })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn tcl_tool_remove(&self, request: TclToolRemoveRequest) -> Result<String> {
        let path = ToolPath::parse(&request.path)?;
        info!("Removing TCL tool: {}", path);

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::RemoveTool { path, response: tx })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn execute_custom_tool(
        &self,
        mcp_name: &str,
        params: serde_json::Value,
    ) -> Result<String> {
        let path = ToolPath::from_mcp_name(mcp_name)?;

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::ExecuteCustomTool {
                path,
                params,
                response: tx,
            })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn get_tool_definitions(&self) -> Result<Vec<ToolDefinition>> {
        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::GetToolDefinitions { response: tx })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        Ok(rx
            .await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?)
    }

    pub async fn initialize_persistence(&self) -> Result<String> {
        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::InitializePersistence { response: tx })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn exec_tool(&self, request: TclExecToolRequest) -> Result<String> {
        info!(
            "Executing tool: {} with params: {:?}",
            request.tool_path, request.params
        );

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::ExecTool {
                tool_path: request.tool_path,
                params: request.params,
                response: tx,
            })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn discover_tools(&self) -> Result<String> {
        info!("Discovering tools from filesystem");

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::DiscoverTools { response: tx })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn reload_tools(&self) -> Result<String> {
        info!("Reloading tools from persistent storage");

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::ReloadTools { response: tx })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    // MCP server management methods
    pub async fn mcp_execute(&self, request: McpExecuteRequest) -> Result<String> {
        info!(
            "Executing MCP tool: {}/{}",
            request.server_id, request.tool_name
        );

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::ExecuteMcp {
                server_id: request.server_id,
                tool_name: request.tool_name,
                params: request.params,
                response_format: request.response_format,
                timeout_ms: request.timeout_ms,
                response: tx,
            })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn mcp_add_server(&self, request: McpServerAddRequest) -> Result<String> {
        info!("Adding MCP server: {} ({})", request.id, request.name);

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::AddMcpServer {
                config: McpServerConfig {
                    id: request.id,
                    name: request.name,
                    description: request.description,
                    command: request.command,
                    args: request.args,
                    env: request.env,
                    auto_start: request.auto_start,
                    timeout_ms: request.timeout_ms,
                    max_retries: request.max_retries,
                    created_at: chrono::Utc::now(),
                },
                response: tx,
            })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn mcp_remove_server(&self, request: McpServerRemoveRequest) -> Result<String> {
        info!(
            "Removing MCP server: {} (force: {})",
            request.server_id, request.force
        );

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::RemoveMcpServer {
                server_id: request.server_id,
                force: request.force,
                response: tx,
            })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn mcp_list_servers(&self) -> Result<String> {
        info!("Listing MCP servers");

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::ListMcpServers { response: tx })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn get_mcp_server_tools(
        &self,
    ) -> Result<Vec<(String, Vec<crate::mcp_client::McpToolDefinition>)>> {
        info!("Getting MCP server tools");

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::GetMcpServerTools { response: tx })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn debug_connect_mcp(&self, request: McpDebugRequest) -> Result<String> {
        info!("Debug connecting to MCP server: {}", request.server_id);

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::DebugConnectMcp {
                server_id: request.server_id,
                response: tx,
            })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn debug_disconnect_mcp(&self, request: McpDebugRequest) -> Result<String> {
        info!("Debug disconnecting from MCP server: {}", request.server_id);

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::DebugDisconnectMcp {
                server_id: request.server_id,
                response: tx,
            })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn debug_mcp_info(&self, request: McpDebugRequest) -> Result<String> {
        info!("Getting debug info for MCP server: {}", request.server_id);

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::DebugMcpServerInfo {
                server_id: request.server_id,
                response: tx,
            })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }

    pub async fn debug_ping_mcp(&self, request: McpDebugRequest) -> Result<String> {
        info!("Pinging MCP server: {}", request.server_id);

        let (tx, rx) = oneshot::channel();
        self.executor
            .send(TclCommand::DebugPingMcp {
                server_id: request.server_id,
                response: tx,
            })
            .await
            .map_err(|_| anyhow!("Failed to send command to executor"))?;

        rx.await
            .map_err(|_| anyhow!("Failed to receive response from executor"))?
    }
}
