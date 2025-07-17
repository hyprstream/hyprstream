use anyhow::{anyhow, Result};
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::thread;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};

use crate::advanced_tools;
use crate::mcp_client::{McpClient, McpServerConfig};
use crate::mcp_persistence::McpPersistence;
use crate::namespace::{Namespace, ToolPath};
use crate::persistence::FilePersistence;
use crate::tcl_runtime::{create_runtime, RuntimeConfig, TclRuntime};
use crate::tcl_tools::{ParameterDefinition, ToolDefinition};
use crate::tool_discovery::{DiscoveredTool, ToolDiscovery};
use crate::tool_registry::ToolRegistry;

pub enum TclCommand {
    Execute {
        script: String,
        response: oneshot::Sender<Result<String>>,
    },
    AddTool {
        path: ToolPath,
        description: String,
        script: String,
        parameters: Vec<ParameterDefinition>,
        response: oneshot::Sender<Result<String>>,
    },
    RemoveTool {
        path: ToolPath,
        response: oneshot::Sender<Result<String>>,
    },
    ListTools {
        namespace: Option<String>,
        filter: Option<String>,
        response: oneshot::Sender<Result<Vec<String>>>,
    },
    ExecuteCustomTool {
        path: ToolPath,
        params: serde_json::Value,
        response: oneshot::Sender<Result<String>>,
    },
    GetToolDefinitions {
        response: oneshot::Sender<Vec<ToolDefinition>>,
    },
    InitializePersistence {
        response: oneshot::Sender<Result<String>>,
    },
    ExecTool {
        tool_path: String,
        params: serde_json::Value,
        response: oneshot::Sender<Result<String>>,
    },
    DiscoverTools {
        response: oneshot::Sender<Result<String>>,
    },
    ExecuteMcp {
        server_id: String,
        tool_name: String,
        params: serde_json::Value,
        response_format: String,
        timeout_ms: u64,
        response: oneshot::Sender<Result<String>>,
    },
    AddMcpServer {
        config: McpServerConfig,
        response: oneshot::Sender<Result<String>>,
    },
    RemoveMcpServer {
        server_id: String,
        force: bool,
        response: oneshot::Sender<Result<String>>,
    },
    ListMcpServers {
        response: oneshot::Sender<Result<String>>,
    },
    GetMcpServerTools {
        response: oneshot::Sender<Result<Vec<(String, Vec<crate::mcp_client::McpToolDefinition>)>>>,
    },
    DebugConnectMcp {
        server_id: String,
        response: oneshot::Sender<Result<String>>,
    },
    DebugDisconnectMcp {
        server_id: String,
        response: oneshot::Sender<Result<String>>,
    },
    DebugMcpServerInfo {
        server_id: String,
        response: oneshot::Sender<Result<String>>,
    },
    DebugPingMcp {
        server_id: String,
        response: oneshot::Sender<Result<String>>,
    },
    ReloadTools {
        response: oneshot::Sender<Result<String>>,
    },
}

pub struct TclExecutor {
    runtime: Box<dyn TclRuntime>,
    custom_tools: HashMap<ToolPath, ToolDefinition>,
    discovered_tools: HashMap<ToolPath, DiscoveredTool>,
    tool_discovery: ToolDiscovery,
    persistence: Option<FilePersistence>,
    mcp_persistence: Option<McpPersistence>,
    mcp_client: McpClient,
    tool_registry: ToolRegistry,
}

impl TclExecutor {
    pub fn new(privileged: bool) -> Self {
        let runtime = create_runtime();

        // In non-privileged mode, we could disable certain commands here
        // For now, we'll just store the flag and use it during execution
        if !privileged {
            // TODO: Consider filtering dangerous commands like 'exec', 'file', etc.
            // For now, we rely on the runtime's default safety features
        }

        tracing::info!("Initialized TCL runtime: {}", runtime.name());

        let mcp_client = McpClient::new();
        let tool_registry = ToolRegistry::new(mcp_client.clone());

        Self {
            runtime,
            custom_tools: HashMap::new(),
            discovered_tools: HashMap::new(),
            tool_discovery: ToolDiscovery::new(),
            persistence: None,
            mcp_persistence: None,
            mcp_client,
            tool_registry,
        }
    }

    pub fn new_with_runtime(
        privileged: bool,
        runtime_config: RuntimeConfig,
    ) -> Result<Self, String> {
        let runtime = crate::tcl_runtime::create_runtime_with_config(runtime_config)
            .map_err(|e| format!("Failed to create TCL runtime: {}", e))?;

        // In non-privileged mode, we could disable certain commands here
        // For now, we'll just store the flag and use it during execution
        if !privileged {
            // TODO: Consider filtering dangerous commands like 'exec', 'file', etc.
            // For now, we rely on the runtime's default safety features
        }

        tracing::info!("Initialized TCL runtime: {}", runtime.name());

        let mcp_client = McpClient::new();
        let tool_registry = ToolRegistry::new(mcp_client.clone());

        Ok(Self {
            runtime,
            custom_tools: HashMap::new(),
            discovered_tools: HashMap::new(),
            tool_discovery: ToolDiscovery::new(),
            persistence: None,
            mcp_persistence: None,
            mcp_client,
            tool_registry,
        })
    }

    pub fn spawn(privileged: bool) -> mpsc::Sender<TclCommand> {
        Self::spawn_with_runtime(privileged, RuntimeConfig::default())
            .expect("Failed to create executor with default runtime")
    }

    pub fn spawn_with_runtime(
        privileged: bool,
        runtime_config: RuntimeConfig,
    ) -> Result<mpsc::Sender<TclCommand>, String> {
        let (tx, mut rx) = mpsc::channel::<TclCommand>(100);
        let tx_clone = tx.clone();

        // Spawn a dedicated thread for the TCL interpreter
        thread::spawn(move || {
            let mut executor = match TclExecutor::new_with_runtime(privileged, runtime_config) {
                Ok(exec) => exec,
                Err(e) => {
                    eprintln!("Failed to create TCL executor: {}", e);
                    return;
                }
            };

            // Create a single-threaded runtime for this thread
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("Failed to create Tokio runtime");

            runtime.block_on(async move {
                // Setup filesystem watcher for tools directory
                let _watcher = setup_filesystem_watcher(tx_clone).await;

                while let Some(cmd) = rx.recv().await {
                    match cmd {
                        TclCommand::Execute { script, response } => {
                            let result = executor.execute_script(&script);
                            let _ = response.send(result);
                        }
                        TclCommand::AddTool {
                            path,
                            description,
                            script,
                            parameters,
                            response,
                        } => {
                            let result = executor
                                .add_tool(path, description, script, parameters)
                                .await;
                            let _ = response.send(result);
                        }
                        TclCommand::RemoveTool { path, response } => {
                            let result = executor.remove_tool(&path).await;
                            let _ = response.send(result);
                        }
                        TclCommand::ListTools {
                            namespace,
                            filter,
                            response,
                        } => {
                            let tools = executor.list_tools(namespace, filter);
                            let _ = response.send(Ok(tools));
                        }
                        TclCommand::ExecuteCustomTool {
                            path,
                            params,
                            response,
                        } => {
                            let result = executor.execute_custom_tool(&path, params);
                            let _ = response.send(result);
                        }
                        TclCommand::GetToolDefinitions { response } => {
                            let tools = executor.get_tool_definitions();
                            let _ = response.send(tools);
                        }
                        TclCommand::InitializePersistence { response } => {
                            let result = executor.initialize_persistence().await;
                            let _ = response.send(result);
                        }
                        TclCommand::ExecTool {
                            tool_path,
                            params,
                            response,
                        } => {
                            let result = executor.exec_tool(&tool_path, params).await;
                            let _ = response.send(result);
                        }
                        TclCommand::DiscoverTools { response } => {
                            let result = executor.discover_tools().await;
                            let _ = response.send(result);
                        }
                        TclCommand::ExecuteMcp {
                            server_id,
                            tool_name,
                            params,
                            response_format,
                            timeout_ms,
                            response,
                        } => {
                            let result = executor
                                .execute_mcp_tool(
                                    &server_id,
                                    &tool_name,
                                    params,
                                    &response_format,
                                    timeout_ms,
                                )
                                .await;
                            let _ = response.send(result);
                        }
                        TclCommand::AddMcpServer { config, response } => {
                            let result = executor.add_mcp_server(config).await;
                            let _ = response.send(result);
                        }
                        TclCommand::RemoveMcpServer {
                            server_id,
                            force,
                            response,
                        } => {
                            let result = executor.remove_mcp_server(&server_id, force).await;
                            let _ = response.send(result);
                        }
                        TclCommand::ListMcpServers { response } => {
                            let result = executor.list_mcp_servers().await;
                            let _ = response.send(result);
                        }
                        TclCommand::GetMcpServerTools { response } => {
                            let result = executor.get_mcp_server_tools().await;
                            let _ = response.send(result);
                        }
                        TclCommand::DebugConnectMcp {
                            server_id,
                            response,
                        } => {
                            let result = executor.debug_connect_mcp_server(&server_id).await;
                            let _ = response.send(result);
                        }
                        TclCommand::DebugDisconnectMcp {
                            server_id,
                            response,
                        } => {
                            let result = executor.debug_disconnect_mcp_server(&server_id).await;
                            let _ = response.send(result);
                        }
                        TclCommand::DebugMcpServerInfo {
                            server_id,
                            response,
                        } => {
                            let result = executor.debug_mcp_server_info(&server_id).await;
                            let _ = response.send(result);
                        }
                        TclCommand::DebugPingMcp {
                            server_id,
                            response,
                        } => {
                            let result = executor.debug_ping_mcp_server(&server_id).await;
                            let _ = response.send(result);
                        }
                        TclCommand::ReloadTools { response } => {
                            let result = executor.reload_tools().await;
                            let _ = response.send(result);
                        }
                    }
                }
            });
        });

        Ok(tx)
    }

    fn execute_script(&mut self, script: &str) -> Result<String> {
        self.runtime.eval(script)
    }

    async fn add_tool(
        &mut self,
        path: ToolPath,
        description: String,
        script: String,
        parameters: Vec<ParameterDefinition>,
    ) -> Result<String> {
        // Only allow adding tools to user namespace
        if !matches!(path.namespace, Namespace::User(_)) {
            return Err(anyhow!(
                "Can only add tools to user namespace, not {}",
                path
            ));
        }

        if self.custom_tools.contains_key(&path) {
            return Err(anyhow!("Tool '{}' already exists", path));
        }

        // Initialize persistence if not already initialized
        if self.persistence.is_none() {
            match FilePersistence::new().await {
                Ok(persistence) => {
                    // Load existing tools from storage
                    match persistence.list_tools(None).await {
                        Ok(stored_tools) => {
                            for tool in stored_tools {
                                if matches!(tool.path.namespace, Namespace::User(_)) {
                                    self.custom_tools.insert(tool.path.clone(), tool.clone());
                                    // Also add to tool registry for immediate availability
                                    self.tool_registry.add_tcl_tool(tool);
                                }
                            }
                            tracing::info!(
                                "Initialized persistence and loaded {} existing tools",
                                self.custom_tools.len()
                            );
                        }
                        Err(e) => {
                            tracing::warn!("Failed to load existing tools: {}", e);
                        }
                    }
                    self.persistence = Some(persistence);
                }
                Err(e) => {
                    tracing::warn!("Failed to initialize persistence: {}", e);
                }
            }
        }

        let tool_def = ToolDefinition {
            path: path.clone(),
            description,
            script,
            parameters,
        };

        // Save to persistence if available
        let persisted = if let Some(ref mut persistence) = self.persistence {
            match persistence.save_tool(&tool_def).await {
                Ok(_) => true,
                Err(e) => {
                    tracing::warn!("Failed to persist tool: {}", e);
                    false
                }
            }
        } else {
            false
        };

        // Add to in-memory cache
        self.custom_tools.insert(path.clone(), tool_def.clone());

        // Also add to tool registry for immediate availability in list_tools_advanced
        self.tool_registry.add_tcl_tool(tool_def);

        if persisted {
            Ok(format!("Tool '{}' added successfully and persisted", path))
        } else {
            Ok(format!(
                "Tool '{}' added to memory (persistence unavailable)",
                path
            ))
        }
    }

    async fn remove_tool(&mut self, path: &ToolPath) -> Result<String> {
        // Cannot remove system tools
        if path.is_system() {
            return Err(anyhow!("Cannot remove system tool '{}'", path));
        }

        // Remove from in-memory cache first
        let removed_from_memory = self.custom_tools.remove(path).is_some();

        // Also remove from tool registry
        let removed_from_registry = self.tool_registry.remove_tcl_tool(path);

        // Remove from persistent storage
        let removed_from_storage = self.remove_tool_from_storage(path).await?;

        if removed_from_memory || removed_from_registry || removed_from_storage {
            Ok(format!("Tool '{}' removed successfully", path))
        } else {
            Err(anyhow!("Tool '{}' not found", path))
        }
    }

    fn list_tools(&self, namespace: Option<String>, filter: Option<String>) -> Vec<String> {
        let mut tools = Vec::new();

        // Add system tools
        let system_tools = vec![
            ToolPath::bin("tcl_execute"),
            ToolPath::sbin("tcl_tool_add"),
            ToolPath::sbin("tcl_tool_remove"),
            ToolPath::bin("tcl_tool_list"),
            ToolPath::bin("list_tools"),
            ToolPath::bin("inspect_tool"),
            ToolPath::bin("list_namespaces"),
            ToolPath::bin("search_tools"),
            ToolPath::bin("list_xmcp_tools"),
            ToolPath::bin("exec_tool"),
            ToolPath::bin("discover_tools"),
            ToolPath::docs("molt_book"),
        ];

        for tool in system_tools {
            if let Some(ref ns) = namespace {
                let matches = match (&tool.namespace, ns.as_str()) {
                    (Namespace::Bin, "bin") => true,
                    (Namespace::Sbin, "sbin") => true,
                    (Namespace::User(user_ns), filter_ns) if user_ns == filter_ns => true,
                    _ => false,
                };
                if !matches {
                    continue;
                }
            }

            let path_str = tool.to_string();
            if filter
                .as_ref()
                .map(|f| path_str.contains(f))
                .unwrap_or(true)
            {
                tools.push(path_str);
            }
        }

        // Add custom tools
        for path in self.custom_tools.keys() {
            if let Some(ref ns) = namespace {
                let matches = match (&path.namespace, ns.as_str()) {
                    (Namespace::User(user_ns), filter_ns) if user_ns == filter_ns => true,
                    _ => false,
                };
                if !matches {
                    continue;
                }
            }

            let path_str = path.to_string();
            if filter
                .as_ref()
                .map(|f| path_str.contains(f))
                .unwrap_or(true)
            {
                tools.push(path_str);
            }
        }

        // Add discovered tools
        for path in self.discovered_tools.keys() {
            if let Some(ref ns) = namespace {
                let matches = match (&path.namespace, ns.as_str()) {
                    (Namespace::Bin, "bin") => true,
                    (Namespace::Sbin, "sbin") => true,
                    (Namespace::Docs, "docs") => true,
                    (Namespace::User(user_ns), filter_ns) if user_ns == filter_ns => true,
                    _ => false,
                };
                if !matches {
                    continue;
                }
            }

            let path_str = path.to_string();
            if filter
                .as_ref()
                .map(|f| path_str.contains(f))
                .unwrap_or(true)
            {
                tools.push(path_str);
            }
        }

        tools.sort();
        tools
    }

    fn execute_custom_tool(
        &mut self,
        path: &ToolPath,
        params: serde_json::Value,
    ) -> Result<String> {
        let tool = self
            .custom_tools
            .get(path)
            .ok_or_else(|| anyhow!("Tool '{}' not found", path))?
            .clone();

        // Set parameters as TCL variables
        if let Some(params_obj) = params.as_object() {
            for param_def in &tool.parameters {
                if let Some(value) = params_obj.get(&param_def.name) {
                    let tcl_value = match value {
                        serde_json::Value::String(s) => s.to_string(),
                        _ => value.to_string(),
                    };
                    self.runtime.set_var(&param_def.name, &tcl_value)?;
                } else if param_def.required {
                    return Err(anyhow!("Missing required parameter: {}", param_def.name));
                }
            }
        }

        // Execute the tool script
        self.execute_script(&tool.script)
    }

    fn get_tool_definitions(&self) -> Vec<ToolDefinition> {
        let mut tools = Vec::new();

        // Add custom tools
        tools.extend(self.custom_tools.values().cloned());

        // Convert discovered tools to ToolDefinition format
        for discovered in self.discovered_tools.values() {
            let tool_def = ToolDefinition {
                path: discovered.path.clone(),
                description: discovered.description.clone(),
                script: format!("# Tool loaded from: {}", discovered.file_path.display()),
                parameters: discovered.parameters.clone(),
            };
            tools.push(tool_def);
        }

        tools
    }

    /// Initialize persistence and load existing tools
    async fn initialize_persistence(&mut self) -> Result<String> {
        let mut message = String::new();

        // Initialize tool persistence if not already done
        if self.persistence.is_none() {
            let persistence = FilePersistence::new().await?;

            // Load existing tools from storage
            let stored_tools = persistence.list_tools(None).await?;
            let loaded_count = stored_tools.len();

            // Add stored tools to in-memory cache
            for tool in stored_tools {
                // Only load user tools, system tools are hardcoded
                if matches!(tool.path.namespace, Namespace::User(_)) {
                    self.custom_tools.insert(tool.path.clone(), tool.clone());
                    // Also add to the unified registry
                    self.tool_registry.add_tcl_tool(tool);
                }
            }

            self.persistence = Some(persistence);
            message.push_str(&format!(
                "Tool persistence initialized. Loaded {} tools from storage. ",
                loaded_count
            ));
        }

        // Initialize MCP persistence if not already done
        if self.mcp_persistence.is_none() {
            let mcp_persistence = McpPersistence::new().await?;

            // Get auto-start servers
            let auto_start_servers = mcp_persistence.get_auto_start_servers();
            let auto_start_count = auto_start_servers.len();

            // Start auto-start servers
            for (server_id, config) in auto_start_servers {
                if let Err(e) = self.mcp_client.register_server(config).await {
                    tracing::warn!("Failed to auto-start MCP server '{}': {}", server_id, e);
                } else {
                    tracing::info!("Auto-started MCP server: {}", server_id);
                }
            }

            self.mcp_persistence = Some(mcp_persistence);
            message.push_str(&format!(
                "MCP persistence initialized. Auto-started {} servers.",
                auto_start_count
            ));
        }

        if message.is_empty() {
            Ok("Persistence already initialized".to_string())
        } else {
            Ok(message)
        }
    }

    /// Reload tools from persistent storage
    async fn reload_tools(&mut self) -> Result<String> {
        if let Some(ref persistence) = self.persistence {
            // Clear existing custom tools from both maps
            for path in self.custom_tools.keys() {
                self.tool_registry.remove_tcl_tool(path);
            }
            self.custom_tools.clear();

            // Reload from storage
            let stored_tools = persistence.list_tools(None).await?;
            let loaded_count = stored_tools.len();

            // Add stored tools back to in-memory cache
            for tool in stored_tools {
                // Only load user tools, system tools are hardcoded
                if matches!(tool.path.namespace, Namespace::User(_)) {
                    self.custom_tools.insert(tool.path.clone(), tool.clone());
                    // Also add to the unified registry
                    self.tool_registry.add_tcl_tool(tool);
                }
            }

            tracing::info!("Reloaded {} tools from persistent storage", loaded_count);
            Ok(format!(
                "Reloaded {} tools from persistent storage",
                loaded_count
            ))
        } else {
            Ok("Persistence not initialized, cannot reload tools".to_string())
        }
    }

    /// Remove tool from persistent storage
    async fn remove_tool_from_storage(&mut self, path: &ToolPath) -> Result<bool> {
        if let Some(ref mut persistence) = self.persistence {
            return persistence.delete_tool(path).await;
        }
        Ok(false)
    }

    /// Execute a tool from the filesystem or custom tools
    async fn exec_tool(&mut self, tool_path: &str, params: serde_json::Value) -> Result<String> {
        // Parse the tool path
        let path = ToolPath::parse(tool_path)?;

        // Check custom tools first (added via tcl_tool_add)
        if self.custom_tools.contains_key(&path) {
            return self.execute_custom_tool(&path, params);
        }

        // Check if it's a discovered tool
        if let Some(discovered_tool) = self.discovered_tools.get(&path) {
            // Read and execute the tool file
            let script_content = tokio::fs::read_to_string(&discovered_tool.file_path).await?;

            // Set parameters as TCL variables
            if let Some(params_obj) = params.as_object() {
                for param_def in &discovered_tool.parameters {
                    if let Some(value) = params_obj.get(&param_def.name) {
                        let tcl_value = match value {
                            serde_json::Value::String(s) => s.to_string(),
                            _ => value.to_string(),
                        };
                        self.runtime.set_var(&param_def.name, &tcl_value)?;
                    } else if param_def.required {
                        return Err(anyhow!("Missing required parameter: {}", param_def.name));
                    }
                }
            }

            // Execute the tool script
            return self.execute_script(&script_content);
        }

        // Check if it's a custom tool
        if let Some(_custom_tool) = self.custom_tools.get(&path) {
            return self.execute_custom_tool(&path, params);
        }

        // Check if it's an MCP tool
        if let Namespace::Mcp(server_id) = &path.namespace {
            return self
                .execute_mcp_tool(server_id, &path.name, params, "json", 30000)
                .await;
        }

        // Check if it's a built-in system tool
        match tool_path {
            "/bin/tcl_execute" => {
                if let Some(script) = params.get("script").and_then(|s| s.as_str()) {
                    self.execute_script(script)
                } else {
                    Err(anyhow!("Missing required parameter: script"))
                }
            }
            "/bin/tcl_tool_list" => {
                let namespace = params
                    .get("namespace")
                    .and_then(|s| s.as_str())
                    .map(String::from);
                let filter = params
                    .get("filter")
                    .and_then(|s| s.as_str())
                    .map(String::from);
                let tools = self.list_tools(namespace, filter);
                Ok(tools.join("\n"))
            }
            "/bin/list_tools" => {
                let namespace = params
                    .get("namespace")
                    .and_then(|s| s.as_str())
                    .map(String::from);
                let server = params
                    .get("server")
                    .and_then(|s| s.as_str())
                    .map(String::from);
                let search = params
                    .get("search")
                    .and_then(|s| s.as_str())
                    .map(String::from);
                let include_schemas = params
                    .get("include_schemas")
                    .and_then(|b| b.as_bool())
                    .unwrap_or(false);
                let limit = params
                    .get("limit")
                    .and_then(|n| n.as_u64())
                    .map(|n| n as usize);
                let format = params
                    .get("format")
                    .and_then(|s| s.as_str())
                    .map(String::from);

                match advanced_tools::list_tools(
                    &self.tool_registry,
                    namespace,
                    server,
                    search,
                    include_schemas,
                    limit,
                    format,
                )
                .await
                {
                    Ok(result) => Ok(result),
                    Err(e) => Err(anyhow!("Failed to list tools: {}", e)),
                }
            }
            "/bin/inspect_tool" => {
                if let Some(tool_path) = params.get("tool_path").and_then(|s| s.as_str()) {
                    match advanced_tools::inspect_tool(&self.tool_registry, tool_path).await {
                        Ok(result) => Ok(result),
                        Err(e) => Err(anyhow!("Failed to inspect tool: {}", e)),
                    }
                } else {
                    Err(anyhow!("Missing required parameter: tool_path"))
                }
            }
            "/bin/list_namespaces" => {
                match advanced_tools::list_namespaces(&self.tool_registry).await {
                    Ok(result) => Ok(result),
                    Err(e) => Err(anyhow!("Failed to list namespaces: {}", e)),
                }
            }
            "/bin/search_tools" => {
                if let Some(query) = params.get("query").and_then(|s| s.as_str()) {
                    let limit = params
                        .get("limit")
                        .and_then(|n| n.as_u64())
                        .map(|n| n as usize);
                    match advanced_tools::search_tools(&self.tool_registry, query, limit).await {
                        Ok(result) => Ok(result),
                        Err(e) => Err(anyhow!("Failed to search tools: {}", e)),
                    }
                } else {
                    Err(anyhow!("Missing required parameter: query"))
                }
            }
            "/bin/list_xmcp_tools" => {
                let server_filter = params
                    .get("server")
                    .and_then(|s| s.as_str())
                    .map(String::from);
                match advanced_tools::list_xmcp_tools(&self.tool_registry, server_filter).await {
                    Ok(result) => Ok(result),
                    Err(e) => Err(anyhow!("Failed to list external MCP tools: {}", e)),
                }
            }
            _ => Err(anyhow!("Tool '{}' not found", tool_path)),
        }
    }

    /// Discover and index tools from the filesystem
    async fn discover_tools(&mut self) -> Result<String> {
        // Discover tools from the filesystem
        let discovered = self.tool_discovery.discover_tools().await?;
        let count = discovered.len();

        // Add discovered tools to our cache
        for tool in discovered {
            self.discovered_tools.insert(tool.path.clone(), tool);
        }

        // Register discovered tools as available for execution
        // Note: We don't add them as TCL commands directly since that would require
        // complex callback handling. Instead, they can be executed via exec_tool.

        Ok(format!("Discovered {} tools from filesystem", count))
    }

    // MCP server management methods
    async fn execute_mcp_tool(
        &mut self,
        server_id: &str,
        tool_name: &str,
        params: serde_json::Value,
        response_format: &str,
        timeout_ms: u64,
    ) -> Result<String> {
        // Execute the tool via MCP client
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(timeout_ms),
            self.mcp_client
                .execute_tool(server_id, tool_name, params.clone()),
        )
        .await
        .map_err(|_| anyhow!("Tool execution timeout after {}ms", timeout_ms))?
        .map_err(|e| anyhow!("MCP tool execution failed: {}", e))?;

        // Format response based on requested format
        match response_format {
            "json" => Ok(serde_json::to_string_pretty(&result)?),
            "text" => {
                // Extract text content from MCP response
                if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
                    let mut text_parts = Vec::new();
                    for item in content {
                        if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                            text_parts.push(text.to_string());
                        }
                    }
                    Ok(text_parts.join("\n"))
                } else {
                    Ok(result.to_string())
                }
            }
            "auto" | _ => {
                // Auto-detect best format
                if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
                    if content.len() == 1
                        && content[0].get("type")
                            == Some(&serde_json::Value::String("text".to_string()))
                    {
                        // Single text response - return as plain text
                        if let Some(text) = content[0].get("text").and_then(|t| t.as_str()) {
                            Ok(text.to_string())
                        } else {
                            Ok(serde_json::to_string_pretty(&result)?)
                        }
                    } else {
                        // Complex response - return as JSON
                        Ok(serde_json::to_string_pretty(&result)?)
                    }
                } else {
                    Ok(serde_json::to_string_pretty(&result)?)
                }
            }
        }
    }

    async fn add_mcp_server(&mut self, config: McpServerConfig) -> Result<String> {
        let server_id = config.id.clone();

        // Validate server configuration
        if server_id.is_empty() {
            return Err(anyhow!("Server ID cannot be empty"));
        }

        if config.command.is_empty() {
            return Err(anyhow!("Server command cannot be empty"));
        }

        // Register the server with the MCP client
        self.mcp_client
            .register_server(config.clone())
            .await
            .map_err(|e| anyhow!("Failed to register MCP server: {}", e))?;

        // Save to persistence
        if let Some(ref mut mcp_persistence) = self.mcp_persistence {
            if let Err(e) = mcp_persistence
                .save_server(server_id.clone(), config.clone(), config.auto_start)
                .await
            {
                tracing::warn!("Failed to persist MCP server configuration: {}", e);
            }
        }

        // Try to connect if auto_start is enabled
        if config.auto_start {
            match self.mcp_client.connect_server(&server_id).await {
                Ok(_) => {
                    // Get tool count
                    let tools = self.mcp_client.get_server_tools(&server_id).await
                        .unwrap_or_default();
                    Ok(format!(
                        "MCP server '{}' registered and connected successfully. {} tools available.",
                        server_id, tools.len()
                    ))
                }
                Err(e) => {
                    Ok(format!(
                        "MCP server '{}' registered but failed to connect: {}. Use '/bin/execute_mcp' to retry connection.",
                        server_id, e
                    ))
                }
            }
        } else {
            Ok(format!(
                "MCP server '{}' registered successfully (auto-start disabled)",
                server_id
            ))
        }
    }

    async fn remove_mcp_server(&mut self, server_id: &str, force: bool) -> Result<String> {
        if server_id.is_empty() {
            return Err(anyhow!("Server ID cannot be empty"));
        }

        // Remove the server
        self.mcp_client
            .remove_server(server_id, force)
            .await
            .map_err(|e| anyhow!("Failed to remove MCP server: {}", e))?;

        // Remove from persistence
        if let Some(ref mut mcp_persistence) = self.mcp_persistence {
            if let Err(e) = mcp_persistence.remove_server(server_id).await {
                tracing::warn!("Failed to remove MCP server from persistence: {}", e);
            }
        }

        if force {
            Ok(format!("MCP server '{}' forcibly removed", server_id))
        } else {
            Ok(format!("MCP server '{}' gracefully removed", server_id))
        }
    }

    async fn list_mcp_servers(&mut self) -> Result<String> {
        let servers = self.mcp_client.list_servers().await;
        if servers.is_empty() {
            return Ok("No MCP servers registered".to_string());
        }

        let mut result = vec!["Registered MCP servers:".to_string()];
        for (id, status) in servers {
            result.push(format!("  {} - Status: {:?}", id, status));
        }
        Ok(result.join("\n"))
    }

    async fn get_mcp_server_tools(
        &mut self,
    ) -> Result<Vec<(String, Vec<crate::mcp_client::McpToolDefinition>)>> {
        let mut all_tools = Vec::new();

        // Get list of all connected servers
        let servers = self.mcp_client.list_servers().await;

        for (server_id, status) in servers {
            // Only get tools from connected servers
            if matches!(status, crate::mcp_client::ConnectionStatus::Connected) {
                match self.mcp_client.get_server_tools(&server_id).await {
                    Ok(tools) => {
                        all_tools.push((server_id, tools));
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

        Ok(all_tools)
    }

    async fn debug_connect_mcp_server(&mut self, server_id: &str) -> Result<String> {
        self.mcp_client.debug_connect_server(server_id).await
    }

    async fn debug_disconnect_mcp_server(&mut self, server_id: &str) -> Result<String> {
        self.mcp_client.debug_disconnect_server(server_id).await
    }

    async fn debug_mcp_server_info(&mut self, server_id: &str) -> Result<String> {
        self.mcp_client.debug_server_info(server_id).await
    }

    async fn debug_ping_mcp_server(&mut self, server_id: &str) -> Result<String> {
        self.mcp_client.debug_ping_server(server_id).await
    }
}

/// Setup filesystem watcher for tools directory
async fn setup_filesystem_watcher(tx: mpsc::Sender<TclCommand>) -> Option<RecommendedWatcher> {
    use crate::platform_dirs;

    let tools_dir = match platform_dirs::tools_dir() {
        Ok(dir) => dir,
        Err(e) => {
            tracing::warn!("Failed to get tools directory: {}", e);
            return None;
        }
    };

    if !tools_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&tools_dir) {
            tracing::warn!("Failed to create tools directory: {}", e);
            return None;
        }
    }

    let (watch_tx, mut watch_rx) = mpsc::channel(100);

    let mut watcher = match notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
        match res {
            Ok(event) => {
                // Send event to our channel
                let _ = watch_tx.try_send(event);
            }
            Err(e) => {
                tracing::warn!("Watch error: {}", e);
            }
        }
    }) {
        Ok(w) => w,
        Err(e) => {
            tracing::warn!("Failed to create filesystem watcher: {}", e);
            return None;
        }
    };

    // Watch the tools directory recursively
    if let Err(e) = watcher.watch(&tools_dir, RecursiveMode::Recursive) {
        tracing::warn!("Failed to watch tools directory: {}", e);
        return None;
    }

    tracing::info!(
        "Filesystem watcher started for tools directory: {}",
        tools_dir.display()
    );

    // Spawn a task to handle filesystem events
    tokio::spawn(async move {
        let mut last_reload = std::time::Instant::now();
        let debounce_duration = Duration::from_millis(500); // 500ms debounce

        while let Some(event) = watch_rx.recv().await {
            // Check if this is a file modification/creation event
            let should_reload = match event.kind {
                EventKind::Create(_) | EventKind::Modify(_) => {
                    // Check if it's a .json file (our tool files)
                    event
                        .paths
                        .iter()
                        .any(|path| path.extension().map_or(false, |ext| ext == "json"))
                }
                _ => false,
            };

            if should_reload {
                let now = std::time::Instant::now();
                if now.duration_since(last_reload) > debounce_duration {
                    tracing::info!("Tools directory changed, triggering reload");
                    last_reload = now;

                    // Send reload command
                    let (response_tx, _response_rx) = oneshot::channel();
                    let _ = tx
                        .send(TclCommand::ReloadTools {
                            response: response_tx,
                        })
                        .await;
                }
            }
        }
    });

    Some(watcher)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcl_tools::ParameterDefinition;
    use std::time::Duration;
    use tokio::time::timeout;

    #[tokio::test]
    async fn test_tcl_script_execution() {
        let executor = TclExecutor::spawn_with_runtime(false, RuntimeConfig::default()).unwrap();
        let (tx, rx) = oneshot::channel();

        executor
            .send(TclCommand::Execute {
                script: "set result [expr {2 + 3}]; return $result".to_string(),
                response: tx,
            })
            .await
            .unwrap();

        let result = timeout(Duration::from_secs(5), rx)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(result.trim(), "5");
    }

    #[tokio::test]
    async fn test_tcl_error_handling() {
        let executor = TclExecutor::spawn_with_runtime(false, RuntimeConfig::default()).unwrap();
        let (tx, rx) = oneshot::channel();

        // Execute invalid TCL script
        executor
            .send(TclCommand::Execute {
                script: "invalid_command_that_does_not_exist".to_string(),
                response: tx,
            })
            .await
            .unwrap();

        let result = timeout(Duration::from_secs(5), rx).await.unwrap().unwrap();
        assert!(result.is_err(), "Expected error for invalid TCL command");
    }

    #[tokio::test]
    async fn test_special_character_handling() {
        let executor = TclExecutor::spawn_with_runtime(false, RuntimeConfig::default()).unwrap();
        let (tx, rx) = oneshot::channel();

        // Test string with special characters
        executor
            .send(TclCommand::Execute {
                script:
                    r#"set message "Hello \"World\" with \$pecial characters!"; return $message"#
                        .to_string(),
                response: tx,
            })
            .await
            .unwrap();

        let result = timeout(Duration::from_secs(5), rx)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(result.contains("Hello \"World\""));
    }

    #[tokio::test]
    async fn test_tool_addition_and_execution() {
        let executor = TclExecutor::spawn_with_runtime(true, RuntimeConfig::default()).unwrap();

        // Add a simple test tool
        let (add_tx, add_rx) = oneshot::channel();
        let tool_path = ToolPath::user("test", "math", "add", "latest");

        executor
            .send(TclCommand::AddTool {
                path: tool_path.clone(),
                description: "Simple addition tool".to_string(),
                script: "set result [expr {$a + $b}]; return \"Result: $a + $b = $result\""
                    .to_string(),
                parameters: vec![
                    ParameterDefinition {
                        name: "a".to_string(),
                        description: "First number".to_string(),
                        required: true,
                        type_name: "number".to_string(),
                    },
                    ParameterDefinition {
                        name: "b".to_string(),
                        description: "Second number".to_string(),
                        required: true,
                        type_name: "number".to_string(),
                    },
                ],
                response: add_tx,
            })
            .await
            .unwrap();

        let add_result = timeout(Duration::from_secs(5), add_rx)
            .await
            .unwrap()
            .unwrap();
        assert!(add_result.is_ok(), "Tool addition should succeed");

        // Execute the tool
        let (exec_tx, exec_rx) = oneshot::channel();
        let mut params = serde_json::Map::new();
        params.insert("a".to_string(), serde_json::Value::Number(5.into()));
        params.insert("b".to_string(), serde_json::Value::Number(3.into()));

        executor
            .send(TclCommand::ExecuteCustomTool {
                path: tool_path,
                params: serde_json::Value::Object(params),
                response: exec_tx,
            })
            .await
            .unwrap();

        let exec_result = timeout(Duration::from_secs(5), exec_rx)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(exec_result.contains("Result: 5 + 3 = 8"));
    }

    #[tokio::test]
    async fn test_parameter_validation() {
        let executor = TclExecutor::spawn_with_runtime(true, RuntimeConfig::default()).unwrap();

        // Add a tool with required parameters
        let (add_tx, add_rx) = oneshot::channel();
        let tool_path = ToolPath::user("test", "validation", "required_param", "latest");

        executor
            .send(TclCommand::AddTool {
                path: tool_path.clone(),
                description: "Tool with required parameter".to_string(),
                script: "return \"Got value: $value\"".to_string(),
                parameters: vec![ParameterDefinition {
                    name: "value".to_string(),
                    description: "Required value".to_string(),
                    required: true,
                    type_name: "string".to_string(),
                }],
                response: add_tx,
            })
            .await
            .unwrap();

        timeout(Duration::from_secs(5), add_rx)
            .await
            .unwrap()
            .unwrap()
            .unwrap();

        // Try to execute without required parameter
        let (exec_tx, exec_rx) = oneshot::channel();
        let params = serde_json::Map::new(); // Empty params

        executor
            .send(TclCommand::ExecuteCustomTool {
                path: tool_path,
                params: serde_json::Value::Object(params),
                response: exec_tx,
            })
            .await
            .unwrap();

        let exec_result = timeout(Duration::from_secs(5), exec_rx)
            .await
            .unwrap()
            .unwrap();
        assert!(
            exec_result.is_err(),
            "Should fail with missing required parameter"
        );
    }
}
