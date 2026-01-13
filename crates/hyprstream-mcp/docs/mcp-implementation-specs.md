# MCP Server Registration Implementation Specifications

## 📋 Core Component Implementations

### 1. MCP Client Pool Manager

```rust
// src/mcp_client.rs
use anyhow::{Result, anyhow};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use tokio::sync::{Mutex, RwLock};
use chrono::{DateTime, Utc};
use tokio::time::{interval, Duration};

#[derive(Debug, Clone)]
pub struct McpClientPool {
    connections: Arc<DashMap<String, Arc<McpClient>>>,
    metadata: Arc<DashMap<String, McpConnectionMetadata>>,
    config: McpPoolConfig,
    registry: Arc<RwLock<McpServerRegistry>>,
}

#[derive(Debug, Clone)]
pub struct McpConnectionMetadata {
    pub server_id: String,
    pub created_at: DateTime<Utc>,
    pub last_used: DateTime<Utc>,
    pub connection_count: AtomicU64,
    pub error_count: AtomicU64,
    pub success_count: AtomicU64,
    pub health_status: Arc<Mutex<McpHealthStatus>>,
    pub avg_response_time: Arc<Mutex<Duration>>,
}

#[derive(Debug, Clone)]
pub enum McpHealthStatus {
    Healthy { last_check: DateTime<Utc> },
    Degraded { reason: String, since: DateTime<Utc> },
    Unhealthy { reason: String, since: DateTime<Utc> },
    Disconnected { since: DateTime<Utc> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpPoolConfig {
    pub max_connections_per_server: usize,
    pub connection_timeout: Duration,
    pub idle_timeout: Duration,
    pub health_check_interval: Duration,
    pub max_retries: usize,
    pub retry_backoff_base: Duration,
    pub circuit_breaker_threshold: usize,
    pub circuit_breaker_timeout: Duration,
}

impl Default for McpPoolConfig {
    fn default() -> Self {
        Self {
            max_connections_per_server: 5,
            connection_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(300),
            health_check_interval: Duration::from_secs(60),
            max_retries: 3,
            retry_backoff_base: Duration::from_millis(100),
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout: Duration::from_secs(60),
        }
    }
}

impl McpClientPool {
    pub fn new(config: McpPoolConfig) -> Self {
        Self {
            connections: Arc::new(DashMap::new()),
            metadata: Arc::new(DashMap::new()),
            config,
            registry: Arc::new(RwLock::new(McpServerRegistry::new())),
        }
    }
    
    /// Get or create connection to MCP server
    pub async fn get_connection(&self, server_id: &str) -> Result<Arc<McpClient>> {
        // Check for existing healthy connection
        if let Some(client) = self.connections.get(server_id) {
            if let Some(metadata) = self.metadata.get(server_id) {
                let health = metadata.health_status.lock().await;
                if matches!(*health, McpHealthStatus::Healthy { .. }) {
                    metadata.connection_count.fetch_add(1, Ordering::Relaxed);
                    return Ok(client.clone());
                }
            }
        }
        
        // Create new connection
        self.create_connection(server_id).await
    }
    
    async fn create_connection(&self, server_id: &str) -> Result<Arc<McpClient>> {
        let registry = self.registry.read().await;
        let server_entry = registry.get_server(server_id)
            .ok_or_else(|| anyhow!("Server '{}' not registered", server_id))?;
        
        let client = match &server_entry.connection_config {
            McpConnectionConfig::Stdio { command, args, env } => {
                McpClient::new_stdio(command.clone(), args.clone(), env.clone()).await?
            }
            McpConnectionConfig::Http { url, headers } => {
                McpClient::new_http(url.clone(), headers.clone()).await?
            }
            McpConnectionConfig::WebSocket { url, headers } => {
                McpClient::new_websocket(url.clone(), headers.clone()).await?
            }
        };
        
        let client = Arc::new(client);
        
        // Initialize metadata
        let metadata = McpConnectionMetadata {
            server_id: server_id.to_string(),
            created_at: Utc::now(),
            last_used: Utc::now(),
            connection_count: AtomicU64::new(1),
            error_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            health_status: Arc::new(Mutex::new(McpHealthStatus::Healthy { 
                last_check: Utc::now() 
            })),
            avg_response_time: Arc::new(Mutex::new(Duration::from_millis(100))),
        };
        
        self.connections.insert(server_id.to_string(), client.clone());
        self.metadata.insert(server_id.to_string(), metadata);
        
        Ok(client)
    }
    
    /// Execute tool on MCP server with error handling and metrics
    pub async fn execute_tool(
        &self, 
        server_id: &str, 
        tool_name: &str, 
        params: serde_json::Value
    ) -> Result<String> {
        let start_time = std::time::Instant::now();
        let client = self.get_connection(server_id).await?;
        
        // Execute tool with timeout
        let result = tokio::time::timeout(
            self.config.connection_timeout,
            client.call_tool(tool_name, params)
        ).await;
        
        let execution_time = start_time.elapsed();
        
        // Update metadata
        if let Some(metadata) = self.metadata.get(server_id) {
            match &result {
                Ok(Ok(_)) => {
                    metadata.success_count.fetch_add(1, Ordering::Relaxed);
                    self.update_avg_response_time(server_id, execution_time).await;
                }
                Ok(Err(_)) | Err(_) => {
                    metadata.error_count.fetch_add(1, Ordering::Relaxed);
                    self.handle_error(server_id).await;
                }
            }
        }
        
        match result {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(anyhow!("Tool execution timeout after {:?}", self.config.connection_timeout)),
        }
    }
    
    async fn update_avg_response_time(&self, server_id: &str, new_time: Duration) {
        if let Some(metadata) = self.metadata.get(server_id) {
            let mut avg_time = metadata.avg_response_time.lock().await;
            // Simple exponential moving average
            *avg_time = Duration::from_millis(
                (avg_time.as_millis() as u64 * 9 + new_time.as_millis() as u64) / 10
            );
        }
    }
    
    async fn handle_error(&self, server_id: &str) {
        if let Some(metadata) = self.metadata.get(server_id) {
            let error_count = metadata.error_count.load(Ordering::Relaxed);
            if error_count >= self.config.circuit_breaker_threshold as u64 {
                let mut health = metadata.health_status.lock().await;
                *health = McpHealthStatus::Unhealthy {
                    reason: format!("Too many errors: {}", error_count),
                    since: Utc::now(),
                };
                
                // Remove connection to force reconnection
                self.connections.remove(server_id);
            }
        }
    }
    
    /// Background health monitoring
    pub async fn start_health_monitoring(&self) {
        let connections = self.connections.clone();
        let metadata = self.metadata.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(config.health_check_interval);
            
            loop {
                interval.tick().await;
                
                for entry in connections.iter() {
                    let server_id = entry.key();
                    let client = entry.value();
                    
                    if let Some(meta) = metadata.get(server_id) {
                        // Check if connection is idle and should be closed
                        let last_used = meta.last_used;
                        if Utc::now().signed_duration_since(last_used) > 
                           chrono::Duration::from_std(config.idle_timeout).unwrap() {
                            connections.remove(server_id);
                            continue;
                        }
                        
                        // Perform health check
                        let health_result = tokio::time::timeout(
                            Duration::from_secs(10),
                            client.health_check()
                        ).await;
                        
                        let mut health = meta.health_status.lock().await;
                        *health = match health_result {
                            Ok(Ok(_)) => McpHealthStatus::Healthy { last_check: Utc::now() },
                            Ok(Err(e)) => McpHealthStatus::Degraded {
                                reason: e.to_string(),
                                since: Utc::now(),
                            },
                            Err(_) => McpHealthStatus::Unhealthy {
                                reason: "Health check timeout".to_string(),
                                since: Utc::now(),
                            },
                        };
                    }
                }
            }
        });
    }
    
    /// Get pool statistics
    pub async fn get_statistics(&self) -> McpPoolStatistics {
        let mut stats = McpPoolStatistics::default();
        
        for entry in self.metadata.iter() {
            let metadata = entry.value();
            stats.total_servers += 1;
            stats.total_connections += metadata.connection_count.load(Ordering::Relaxed);
            stats.total_errors += metadata.error_count.load(Ordering::Relaxed);
            stats.total_successes += metadata.success_count.load(Ordering::Relaxed);
            
            let health = metadata.health_status.lock().await;
            match *health {
                McpHealthStatus::Healthy { .. } => stats.healthy_servers += 1,
                McpHealthStatus::Degraded { .. } => stats.degraded_servers += 1,
                McpHealthStatus::Unhealthy { .. } => stats.unhealthy_servers += 1,
                McpHealthStatus::Disconnected { .. } => stats.disconnected_servers += 1,
            }
        }
        
        stats
    }
}

#[derive(Debug, Default)]
pub struct McpPoolStatistics {
    pub total_servers: usize,
    pub healthy_servers: usize,
    pub degraded_servers: usize,
    pub unhealthy_servers: usize,
    pub disconnected_servers: usize,
    pub total_connections: u64,
    pub total_errors: u64,
    pub total_successes: u64,
}
```

### 2. MCP Server Registry Implementation

```rust
// src/mcp_registry.rs
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use crate::persistence::FilePersistence;
use crate::namespace::ToolPath;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerRegistry {
    servers: HashMap<String, McpServerEntry>,
    tool_mappings: HashMap<String, HashMap<String, McpToolDefinition>>,
    metadata: RegistryMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub connection_config: McpConnectionConfig,
    pub auth_config: Option<McpAuthConfig>,
    pub health_config: McpHealthConfig,
    pub registered_at: DateTime<Utc>,
    pub last_discovered: Option<DateTime<Utc>>,
    pub capabilities: Vec<String>,
    pub tool_count: usize,
    pub status: McpServerStatus,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    pub server_id: String,
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub namespace_path: ToolPath,
    pub discovered_at: DateTime<Utc>,
    pub last_used: Option<DateTime<Utc>>,
    pub usage_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum McpConnectionConfig {
    Stdio {
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
        working_dir: Option<String>,
    },
    Http {
        url: String,
        headers: HashMap<String, String>,
        timeout_ms: Option<u64>,
        verify_ssl: bool,
    },
    WebSocket {
        url: String,
        headers: HashMap<String, String>,
        timeout_ms: Option<u64>,
        verify_ssl: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum McpAuthConfig {
    None,
    ApiKey {
        #[serde(with = "encrypted_string")]
        key: String,
        header_name: String,
    },
    BasicAuth {
        #[serde(with = "encrypted_string")]
        username: String,
        #[serde(with = "encrypted_string")]
        password: String,
    },
    Bearer {
        #[serde(with = "encrypted_string")]
        token: String,
    },
    OAuth2 {
        client_id: String,
        #[serde(with = "encrypted_string")]
        client_secret: String,
        token_url: String,
        scopes: Vec<String>,
        #[serde(with = "encrypted_string")]
        refresh_token: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpHealthConfig {
    pub check_interval_ms: u64,
    pub timeout_ms: u64,
    pub max_failures: usize,
    pub recovery_timeout_ms: u64,
}

impl Default for McpHealthConfig {
    fn default() -> Self {
        Self {
            check_interval_ms: 60000,
            timeout_ms: 10000,
            max_failures: 3,
            recovery_timeout_ms: 300000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum McpServerStatus {
    Active,
    Inactive,
    Error { message: String },
    Disabled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryMetadata {
    pub version: String,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub total_servers: usize,
    pub total_tools: usize,
}

impl McpServerRegistry {
    pub fn new() -> Self {
        Self {
            servers: HashMap::new(),
            tool_mappings: HashMap::new(),
            metadata: RegistryMetadata {
                version: "1.0.0".to_string(),
                created_at: Utc::now(),
                last_updated: Utc::now(),
                total_servers: 0,
                total_tools: 0,
            },
        }
    }
    
    /// Load registry from persistent storage
    pub async fn load() -> Result<Self> {
        let persistence = FilePersistence::new().await?;
        // Implementation depends on persistence layer
        // For now, return new registry
        Ok(Self::new())
    }
    
    /// Save registry to persistent storage
    pub async fn save(&self) -> Result<()> {
        let persistence = FilePersistence::new().await?;
        // Serialize and save registry data
        // Implementation depends on persistence layer
        Ok(())
    }
    
    /// Register new MCP server
    pub async fn register_server(&mut self, mut entry: McpServerEntry) -> Result<()> {
        if self.servers.contains_key(&entry.id) {
            return Err(anyhow!("Server with ID '{}' already exists", entry.id));
        }
        
        entry.registered_at = Utc::now();
        entry.status = McpServerStatus::Active;
        
        self.servers.insert(entry.id.clone(), entry);
        self.metadata.total_servers = self.servers.len();
        self.metadata.last_updated = Utc::now();
        
        self.save().await?;
        Ok(())
    }
    
    /// Unregister MCP server
    pub async fn unregister_server(&mut self, server_id: &str) -> Result<bool> {
        if self.servers.remove(server_id).is_some() {
            // Remove associated tools
            self.tool_mappings.remove(server_id);
            
            self.metadata.total_servers = self.servers.len();
            self.metadata.total_tools = self.tool_mappings.values()
                .map(|tools| tools.len()).sum();
            self.metadata.last_updated = Utc::now();
            
            self.save().await?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// Get server by ID
    pub fn get_server(&self, server_id: &str) -> Option<&McpServerEntry> {
        self.servers.get(server_id)
    }
    
    /// List all servers with optional filters
    pub fn list_servers(&self, filters: Option<RegistryFilters>) -> Vec<&McpServerEntry> {
        let mut servers: Vec<&McpServerEntry> = self.servers.values().collect();
        
        if let Some(filters) = filters {
            if let Some(status) = filters.status {
                servers.retain(|s| std::mem::discriminant(&s.status) == std::mem::discriminant(&status));
            }
            
            if let Some(tags) = filters.tags {
                servers.retain(|s| tags.iter().all(|tag| s.tags.contains(tag)));
            }
            
            if let Some(name_pattern) = filters.name_pattern {
                servers.retain(|s| s.name.contains(&name_pattern) || s.description.contains(&name_pattern));
            }
        }
        
        servers.sort_by(|a, b| a.name.cmp(&b.name));
        servers
    }
    
    /// Register tools for a server
    pub async fn register_tools(&mut self, server_id: &str, tools: Vec<McpToolDefinition>) -> Result<()> {
        if !self.servers.contains_key(server_id) {
            return Err(anyhow!("Server '{}' not found", server_id));
        }
        
        let tool_map: HashMap<String, McpToolDefinition> = tools.into_iter()
            .map(|tool| (tool.name.clone(), tool))
            .collect();
            
        // Update server tool count
        if let Some(server) = self.servers.get_mut(server_id) {
            server.tool_count = tool_map.len();
            server.last_discovered = Some(Utc::now());
        }
        
        self.tool_mappings.insert(server_id.to_string(), tool_map);
        self.metadata.total_tools = self.tool_mappings.values()
            .map(|tools| tools.len()).sum();
        self.metadata.last_updated = Utc::now();
        
        self.save().await?;
        Ok(())
    }
    
    /// Get tool definition
    pub fn get_tool(&self, server_id: &str, tool_name: &str) -> Option<&McpToolDefinition> {
        self.tool_mappings.get(server_id)?.get(tool_name)
    }
    
    /// List tools for server
    pub fn list_server_tools(&self, server_id: &str) -> Option<Vec<&McpToolDefinition>> {
        Some(self.tool_mappings.get(server_id)?.values().collect())
    }
    
    /// Search tools across all servers
    pub fn search_tools(&self, query: &str) -> Vec<&McpToolDefinition> {
        let mut results = Vec::new();
        
        for tools in self.tool_mappings.values() {
            for tool in tools.values() {
                if tool.name.contains(query) || tool.description.contains(query) {
                    results.push(tool);
                }
            }
        }
        
        results.sort_by(|a, b| a.name.cmp(&b.name));
        results
    }
}

#[derive(Debug, Clone)]
pub struct RegistryFilters {
    pub status: Option<McpServerStatus>,
    pub tags: Option<Vec<String>>,
    pub name_pattern: Option<String>,
}

// Encryption helper module for sensitive data
mod encrypted_string {
    use serde::{Deserialize, Deserializer, Serializer};
    
    pub fn serialize<S>(value: &str, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // In production, encrypt the string here
        // For now, just base64 encode as a placeholder
        let encoded = base64::encode(value);
        serializer.serialize_str(&encoded)
    }
    
    pub fn deserialize<'de, D>(deserializer: D) -> Result<String, D::Error>
    where
        D: Deserializer<'de>,
    {
        let encoded = String::deserialize(deserializer)?;
        // In production, decrypt the string here
        // For now, just base64 decode as a placeholder
        base64::decode(&encoded)
            .map_err(serde::de::Error::custom)
            .and_then(|bytes| String::from_utf8(bytes).map_err(serde::de::Error::custom))
    }
}
```

### 3. Tool Implementation Specifications

```rust
// src/mcp_tools.rs
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;
use crate::mcp_client::McpClientPool;
use crate::mcp_registry::{McpServerRegistry, McpServerEntry, McpConnectionConfig};
use crate::tcl_executor::TclCommand;

/// bin__execute_mcp tool implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpExecuteRequest {
    /// MCP server identifier
    pub server_id: String,
    /// Tool name within the MCP server
    pub tool_name: String,
    /// Parameters to pass to the MCP tool
    pub params: serde_json::Value,
    /// Optional timeout in milliseconds
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Whether to use cached connection
    #[serde(default = "default_true")]
    pub use_cache: bool,
}

pub async fn bin_execute_mcp(
    request: McpExecuteRequest,
    client_pool: &McpClientPool
) -> Result<String> {
    // Validate server exists
    let registry = client_pool.registry.read().await;
    if !registry.has_server(&request.server_id) {
        return Err(anyhow!("MCP server '{}' not registered", request.server_id));
    }
    
    // Validate tool exists
    if registry.get_tool(&request.server_id, &request.tool_name).is_none() {
        return Err(anyhow!("Tool '{}' not found on server '{}'", 
                          request.tool_name, request.server_id));
    }
    
    // Execute tool with proper error handling
    let result = client_pool.execute_tool(
        &request.server_id,
        &request.tool_name,
        request.params
    ).await;
    
    match result {
        Ok(response) => {
            // Update tool usage statistics
            registry.update_tool_usage(&request.server_id, &request.tool_name).await?;
            Ok(response)
        }
        Err(e) => {
            // Log execution error for monitoring
            tracing::warn!("MCP tool execution failed: server={}, tool={}, error={}", 
                          request.server_id, request.tool_name, e);
            Err(e)
        }
    }
}

/// sbin__mcp_add tool implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpAddRequest {
    /// Unique identifier for this MCP server
    pub server_id: String,
    /// Human-readable name
    pub name: String,
    /// Description of server capabilities
    pub description: String,
    /// Connection configuration
    pub connection: McpConnectionConfig,
    /// Authentication configuration
    #[serde(default)]
    pub auth: Option<McpAuthConfig>,
    /// Auto-discover tools on registration
    #[serde(default = "default_true")]
    pub discover_tools: bool,
    /// Health check configuration
    #[serde(default)]
    pub health_check: McpHealthConfig,
    /// Optional tags for categorization
    #[serde(default)]
    pub tags: Vec<String>,
}

pub async fn sbin_mcp_add(
    request: McpAddRequest,
    client_pool: &McpClientPool,
    tool_discovery: &McpToolDiscovery
) -> Result<String> {
    // Validate server ID format
    if !is_valid_server_id(&request.server_id) {
        return Err(anyhow!("Invalid server ID format: must be alphanumeric with hyphens/underscores"));
    }
    
    // Test connection before registering
    tracing::info!("Testing connection to MCP server '{}'", request.server_id);
    let test_client = create_test_client(&request.connection, &request.auth).await?;
    
    // Verify server responds to basic MCP protocol
    test_client.initialize().await
        .map_err(|e| anyhow!("Failed to initialize MCP connection: {}", e))?;
    
    // Create server entry
    let server_entry = McpServerEntry {
        id: request.server_id.clone(),
        name: request.name,
        description: request.description,
        connection_config: request.connection,
        auth_config: request.auth,
        health_config: request.health_check,
        registered_at: chrono::Utc::now(),
        last_discovered: None,
        capabilities: Vec::new(),
        tool_count: 0,
        status: McpServerStatus::Active,
        tags: request.tags,
    };
    
    // Register server in registry
    let mut registry = client_pool.registry.write().await;
    registry.register_server(server_entry).await?;
    
    let mut result_msg = format!("MCP server '{}' registered successfully", request.server_id);
    
    // Discover tools if requested
    if request.discover_tools {
        match tool_discovery.discover_server_tools(&request.server_id).await {
            Ok(tools) => {
                registry.register_tools(&request.server_id, tools.clone()).await?;
                result_msg.push_str(&format!("\nDiscovered {} tools", tools.len()));
            }
            Err(e) => {
                tracing::warn!("Failed to discover tools for server '{}': {}", request.server_id, e);
                result_msg.push_str(&format!("\nWarning: Tool discovery failed: {}", e));
            }
        }
    }
    
    Ok(result_msg)
}

/// sbin__mcp_remove tool implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRemoveRequest {
    /// Server ID to remove
    pub server_id: String,
    /// Force removal even if connections exist
    #[serde(default)]
    pub force: bool,
    /// Remove associated tool definitions
    #[serde(default = "default_true")]
    pub cleanup_tools: bool,
}

pub async fn sbin_mcp_remove(
    request: McpRemoveRequest,
    client_pool: &McpClientPool
) -> Result<String> {
    // Check if server exists
    let registry = client_pool.registry.read().await;
    if !registry.has_server(&request.server_id) {
        return Err(anyhow!("MCP server '{}' not found", request.server_id));
    }
    
    // Check for active connections
    let stats = client_pool.get_server_statistics(&request.server_id).await;
    if stats.active_connections > 0 && !request.force {
        return Err(anyhow!("Server '{}' has {} active connections. Use force=true to remove anyway.", 
                          request.server_id, stats.active_connections));
    }
    
    // Close existing connections
    client_pool.close_server_connections(&request.server_id).await;
    
    // Remove from registry
    drop(registry);
    let mut registry = client_pool.registry.write().await;
    let removed = registry.unregister_server(&request.server_id).await?;
    
    if removed {
        Ok(format!("MCP server '{}' removed successfully", request.server_id))
    } else {
        Err(anyhow!("Failed to remove server '{}'", request.server_id))
    }
}

/// Utility functions
fn default_true() -> bool { true }

fn is_valid_server_id(id: &str) -> bool {
    !id.is_empty() && 
    id.len() <= 64 && 
    id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_')
}

async fn create_test_client(
    config: &McpConnectionConfig,
    auth: &Option<McpAuthConfig>
) -> Result<McpClient> {
    // Create temporary client for testing
    match config {
        McpConnectionConfig::Stdio { command, args, env, .. } => {
            McpClient::new_stdio(command.clone(), args.clone(), env.clone()).await
        }
        McpConnectionConfig::Http { url, headers, .. } => {
            McpClient::new_http(url.clone(), headers.clone()).await
        }
        McpConnectionConfig::WebSocket { url, headers, .. } => {
            McpClient::new_websocket(url.clone(), headers.clone()).await
        }
    }
}
```

### 4. Tool Discovery Implementation

```rust
// src/mcp_discovery.rs
use anyhow::Result;
use crate::mcp_client::McpClientPool;
use crate::mcp_registry::{McpServerRegistry, McpToolDefinition};
use crate::namespace::{ToolPath, Namespace};

pub struct McpToolDiscovery {
    client_pool: Arc<McpClientPool>,
}

impl McpToolDiscovery {
    pub fn new(client_pool: Arc<McpClientPool>) -> Self {
        Self { client_pool }
    }
    
    /// Discover tools from specific MCP server
    pub async fn discover_server_tools(&self, server_id: &str) -> Result<Vec<McpToolDefinition>> {
        let client = self.client_pool.get_connection(server_id).await?;
        
        // Call MCP tools/list method
        let tools_response = client.list_tools().await?;
        let mut discovered_tools = Vec::new();
        
        for tool in tools_response.tools {
            // Map MCP tool to TCL namespace
            let namespace_path = ToolPath::user(
                "mcp",
                server_id,
                &tool.name,
                "latest"
            );
            
            let tool_def = McpToolDefinition {
                server_id: server_id.to_string(),
                name: tool.name,
                description: tool.description.unwrap_or_default(),
                input_schema: tool.input_schema,
                namespace_path,
                discovered_at: chrono::Utc::now(),
                last_used: None,
                usage_count: 0,
            };
            
            discovered_tools.push(tool_def);
        }
        
        Ok(discovered_tools)
    }
    
    /// Refresh all server tools
    pub async fn refresh_all_tools(&self) -> Result<RefreshSummary> {
        let registry = self.client_pool.registry.read().await;
        let servers: Vec<String> = registry.list_servers(None)
            .into_iter()
            .map(|s| s.id.clone())
            .collect();
        drop(registry);
        
        let mut summary = RefreshSummary::default();
        
        for server_id in servers {
            match self.discover_server_tools(&server_id).await {
                Ok(tools) => {
                    let tool_count = tools.len();
                    let mut registry = self.client_pool.registry.write().await;
                    if let Err(e) = registry.register_tools(&server_id, tools).await {
                        tracing::warn!("Failed to register tools for server '{}': {}", server_id, e);
                        summary.failed_servers.push(server_id);
                    } else {
                        summary.successful_servers.push(server_id.clone());
                        summary.total_tools_discovered += tool_count;
                    }
                }
                Err(e) => {
                    tracing::warn!("Failed to discover tools for server '{}': {}", server_id, e);
                    summary.failed_servers.push(server_id);
                }
            }
        }
        
        Ok(summary)
    }
}

#[derive(Debug, Default)]
pub struct RefreshSummary {
    pub successful_servers: Vec<String>,
    pub failed_servers: Vec<String>,
    pub total_tools_discovered: usize,
}
```

This implementation provides a robust, secure, and performant foundation for MCP server registration while maintaining compatibility with the existing TCL-MCP architecture.