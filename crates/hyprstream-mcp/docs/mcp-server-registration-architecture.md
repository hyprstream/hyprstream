# MCP Server Registration Architecture

## Executive Summary

This document presents a comprehensive architecture for registering MCP servers as tools within the TCL-MCP system. The design enables seamless integration of external MCP servers while maintaining security, performance, and consistency with the existing namespace-based tool system.

## 🚨 Architecture Overview

### Core Components

1. **bin__execute_mcp** - Tool execution proxy for registered MCP servers
2. **sbin__mcp_add** - Secure MCP server registration tool 
3. **sbin__mcp_remove** - Safe MCP server deregistration tool
4. **MCP Client Pool** - Connection management and lifecycle
5. **MCP Server Registry** - Persistent storage and metadata management
6. **MCP Tool Discovery** - Capability introspection and tool mapping

### Design Principles

- **Security First**: Authentication, authorization, and privilege separation
- **Performance Optimized**: Connection pooling, caching, and async execution
- **Namespace Integration**: MCP servers as first-class tools in TCL namespace system
- **Fault Tolerance**: Health monitoring, reconnection, and graceful degradation

## 🔧 Core Tool Implementation Specifications

### 1. bin__execute_mcp Tool

**Purpose**: Execute tools from registered MCP servers with transparent parameter marshaling

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpExecuteRequest {
    /// MCP server identifier (from registry)
    pub server_id: String,
    /// Tool name within the MCP server
    pub tool_name: String,
    /// Parameters to pass to the MCP tool
    pub params: serde_json::Value,
    /// Optional timeout in milliseconds
    #[serde(default)]
    pub timeout_ms: Option<u64>,
    /// Whether to use cached connection or create new one
    #[serde(default)]
    pub use_cache: bool,
}

pub async fn bin_execute_mcp(request: McpExecuteRequest) -> Result<String> {
    // 1. Validate server_id exists in registry
    // 2. Get or create MCP client connection
    // 3. Call MCP server tool with parameters
    // 4. Handle response and error mapping
    // 5. Return formatted result
}
```

**Security Features**:
- Server ID validation against registry
- Parameter sanitization and type checking
- Timeout enforcement to prevent hanging
- Rate limiting per server
- Audit logging of all executions

### 2. sbin__mcp_add Tool

**Purpose**: Register new MCP servers with authentication and capability discovery

```rust
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
    pub auth: Option<McpAuthConfig>,
    /// Auto-discover tools on registration
    #[serde(default = "default_true")]
    pub discover_tools: bool,
    /// Health check configuration
    #[serde(default)]
    pub health_check: McpHealthConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum McpConnectionConfig {
    Stdio {
        command: String,
        args: Vec<String>,
        env: HashMap<String, String>,
    },
    Http {
        url: String,
        headers: HashMap<String, String>,
    },
    WebSocket {
        url: String,
        headers: HashMap<String, String>,
    },
}

pub async fn sbin_mcp_add(request: McpAddRequest) -> Result<String> {
    // 1. Validate server_id uniqueness
    // 2. Test connection and authentication
    // 3. Discover available tools and capabilities
    // 4. Store configuration in registry
    // 5. Initialize health monitoring
    // 6. Return registration summary
}
```

**Security Features**:
- Privileged operation (sbin namespace)
- Connection validation before storage
- Credential encryption at rest
- Capability-based access control
- Configuration integrity validation

### 3. sbin__mcp_remove Tool

**Purpose**: Safely deregister MCP servers with cleanup

```rust
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

pub async fn sbin_mcp_remove(request: McpRemoveRequest) -> Result<String> {
    // 1. Validate server exists
    // 2. Check for active connections
    // 3. Gracefully close connections
    // 4. Remove from registry
    // 5. Clean up cached tools and data
    // 6. Stop health monitoring
}
```

## 🔗 MCP Client Lifecycle Management

### Connection Pool Architecture

```rust
pub struct McpClientPool {
    /// Active connections by server ID
    connections: DashMap<String, Arc<McpClient>>,
    /// Connection metadata and health status
    metadata: DashMap<String, McpConnectionMetadata>,
    /// Background health monitoring task
    health_monitor: Option<JoinHandle<()>>,
    /// Configuration
    config: McpPoolConfig,
}

#[derive(Debug, Clone)]
pub struct McpConnectionMetadata {
    pub server_id: String,
    pub last_used: DateTime<Utc>,
    pub connection_count: AtomicU64,
    pub error_count: AtomicU64,
    pub health_status: Arc<Mutex<McpHealthStatus>>,
}

#[derive(Debug, Clone)]
pub enum McpHealthStatus {
    Healthy,
    Degraded { reason: String },
    Unhealthy { reason: String },
    Disconnected,
}
```

### Connection Management Strategy

1. **Connection Reuse**: Pool connections per server ID with configurable TTL
2. **Health Monitoring**: Periodic health checks with exponential backoff on failures
3. **Reconnection Logic**: Automatic reconnection with circuit breaker pattern
4. **Resource Cleanup**: Automatic cleanup of stale connections and resources
5. **Concurrency Control**: Connection limits per server to prevent resource exhaustion

### Connection Lifecycle

```rust
impl McpClientPool {
    /// Get or create connection to MCP server
    pub async fn get_connection(&self, server_id: &str) -> Result<Arc<McpClient>> {
        // 1. Check existing healthy connection
        // 2. If none, create new connection
        // 3. Update metadata and health status
        // 4. Return pooled connection
    }
    
    /// Health check for specific server
    pub async fn health_check(&self, server_id: &str) -> McpHealthStatus {
        // 1. Send ping/capabilities request
        // 2. Measure response time
        // 3. Update health status
        // 4. Trigger reconnection if needed
    }
    
    /// Background health monitoring
    async fn monitor_health(&self) {
        // 1. Periodic health checks for all servers
        // 2. Connection cleanup for failed servers
        // 3. Metrics collection and reporting
        // 4. Circuit breaker management
    }
}
```

## 🗄️ MCP Server Registry Design

### Registry Storage Schema

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerRegistry {
    /// Registered servers by ID
    servers: DashMap<String, McpServerEntry>,
    /// Tool mappings (server_id -> tool_name -> tool_definition)
    tool_mappings: DashMap<String, DashMap<String, McpToolDefinition>>,
    /// Registry metadata
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
    pub last_discovered: DateTime<Utc>,
    pub capabilities: Vec<String>,
    pub tool_count: usize,
    pub status: McpServerStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDefinition {
    pub server_id: String,
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub namespace_path: ToolPath, // Maps to TCL namespace
    pub discovered_at: DateTime<Utc>,
}
```

### Persistence Strategy

```rust
impl McpServerRegistry {
    /// Save registry to persistent storage
    pub async fn save(&self) -> Result<()> {
        // 1. Serialize registry data
        // 2. Encrypt sensitive data (credentials)
        // 3. Write to storage with atomic operation
        // 4. Update backup if needed
    }
    
    /// Load registry from persistent storage
    pub async fn load() -> Result<Self> {
        // 1. Read from storage
        // 2. Decrypt sensitive data
        // 3. Validate data integrity
        // 4. Reconstruct registry
    }
    
    /// Register new MCP server
    pub async fn register_server(&mut self, entry: McpServerEntry) -> Result<()> {
        // 1. Validate uniqueness
        // 2. Test connection
        // 3. Discover tools
        // 4. Store entry
        // 5. Persist changes
    }
}
```

## 🔍 Tool Discovery and Capability Introspection

### Discovery Process

```rust
pub struct McpToolDiscovery {
    client_pool: Arc<McpClientPool>,
    registry: Arc<RwLock<McpServerRegistry>>,
}

impl McpToolDiscovery {
    /// Discover tools from MCP server
    pub async fn discover_server_tools(&self, server_id: &str) -> Result<Vec<McpToolDefinition>> {
        // 1. Get connection to server
        // 2. Call tools/list MCP method
        // 3. Parse tool definitions
        // 4. Map to TCL namespace paths
        // 5. Store in registry
    }
    
    /// Refresh tool definitions for all servers
    pub async fn refresh_all_tools(&self) -> Result<RefreshSummary> {
        // 1. Iterate through registered servers
        // 2. Discover tools for each healthy server
        // 3. Update registry with new tools
        // 4. Remove stale tool definitions
        // 5. Return summary of changes
    }
}
```

### Namespace Integration

MCP server tools are mapped into the TCL namespace system:

```
/mcp/{server_id}/{tool_name}        # Direct tool path
/mcp/{server_id}/{tool_name}:latest # Versioned path
/bin/execute_mcp                    # Generic execution tool
/sbin/mcp_add                       # Registration tool
/sbin/mcp_remove                    # Deregistration tool
```

## 🔐 Security and Authentication Framework

### Authentication Configuration

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum McpAuthConfig {
    None,
    ApiKey {
        #[serde(serialize_with = "encrypt_credential")]
        key: String,
        header_name: Option<String>,
    },
    BasicAuth {
        #[serde(serialize_with = "encrypt_credential")]
        username: String,
        #[serde(serialize_with = "encrypt_credential")]
        password: String,
    },
    Bearer {
        #[serde(serialize_with = "encrypt_credential")]
        token: String,
    },
    OAuth2 {
        client_id: String,
        #[serde(serialize_with = "encrypt_credential")]
        client_secret: String,
        token_url: String,
        scopes: Vec<String>,
    },
}
```

### Security Features

1. **Credential Encryption**: All sensitive data encrypted at rest using platform keychain
2. **Connection Validation**: Certificate validation for HTTPS/WSS connections
3. **Privilege Separation**: MCP operations require sbin privileges
4. **Audit Logging**: All MCP operations logged for security monitoring
5. **Rate Limiting**: Per-server rate limits to prevent abuse
6. **Timeout Enforcement**: Configurable timeouts to prevent resource exhaustion

### Authorization Framework

```rust
pub struct McpAuthorizationPolicy {
    /// Allowed namespaces for MCP tools
    pub allowed_namespaces: Vec<String>,
    /// Maximum execution time per tool
    pub max_execution_time: Duration,
    /// Rate limits per server
    pub rate_limits: HashMap<String, RateLimit>,
    /// Blocked tool patterns
    pub blocked_tools: Vec<Regex>,
}
```

## ⚡ Performance Optimization

### Connection Pooling Strategy

1. **Connection Reuse**: Maintain persistent connections with configurable TTL
2. **Connection Limits**: Per-server connection limits to prevent resource exhaustion
3. **Health Monitoring**: Proactive health checks to avoid failed requests
4. **Circuit Breaker**: Automatic fallback for failed servers

### Caching Strategy

```rust
pub struct McpCacheManager {
    /// Tool definition cache
    tool_cache: Cache<String, McpToolDefinition>,
    /// Response cache for idempotent operations
    response_cache: Cache<String, CachedResponse>,
    /// Server capability cache
    capability_cache: Cache<String, ServerCapabilities>,
}
```

### Performance Metrics

1. **Connection Metrics**: Pool size, connection health, error rates
2. **Execution Metrics**: Response times, success rates, timeout rates
3. **Cache Metrics**: Hit rates, eviction rates, memory usage
4. **Resource Usage**: CPU, memory, network utilization

## 🏗️ Implementation Roadmap

### Phase 1: Core Infrastructure (Weeks 1-2)
- [ ] MCP client connection management
- [ ] Basic registry implementation
- [ ] Tool discovery framework
- [ ] Security foundation

### Phase 2: Tool Implementation (Weeks 3-4)
- [ ] bin__execute_mcp implementation
- [ ] sbin__mcp_add implementation
- [ ] sbin__mcp_remove implementation
- [ ] Namespace integration

### Phase 3: Advanced Features (Weeks 5-6)
- [ ] Connection pooling and health monitoring
- [ ] Performance optimization
- [ ] Comprehensive error handling
- [ ] Metrics and monitoring

### Phase 4: Testing and Documentation (Weeks 7-8)
- [ ] Unit test suite
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] User documentation

## 🧪 Testing Strategy

### Unit Tests
- MCP client connection handling
- Registry storage and retrieval
- Tool discovery and mapping
- Authentication and authorization

### Integration Tests
- End-to-end MCP server communication
- Namespace integration
- Error handling and recovery
- Performance under load

### Security Tests
- Authentication bypass attempts
- Authorization boundary testing
- Credential storage security
- Network security validation

## 📊 Monitoring and Observability

### Metrics Collection
- Connection pool status
- Tool execution statistics
- Error rates and types
- Performance characteristics

### Health Checks
- MCP server availability
- Connection pool health
- Registry integrity
- Tool discovery status

### Alerting
- Server disconnections
- High error rates
- Performance degradation
- Security events

## 🔗 Integration with Existing TCL-MCP System

### Namespace Compatibility
MCP tools integrate seamlessly with existing namespace system:
- System tools remain in `/bin` and `/sbin`
- MCP tools appear in `/mcp/{server_id}/`
- Tool listing includes MCP tools
- Execution model remains consistent

### Persistence Integration
MCP registry uses existing persistence framework:
- FilePersistence for configuration storage
- Encrypted credential storage
- Atomic updates and rollback
- Backup and recovery

### Discovery Integration
MCP tools extend existing tool discovery:
- ToolDiscovery framework extension
- Unified tool listing
- Cross-namespace tool search
- Capability-based filtering

## 🚀 Advanced Features

### Auto-Discovery
- Network scanning for MCP servers
- Service registration protocols
- Dynamic server registration
- Capability negotiation

### Load Balancing
- Multiple servers for same service
- Round-robin and weighted routing
- Health-based routing
- Failover capabilities

### Caching and Optimization
- Response caching for idempotent operations
- Tool definition caching
- Connection pooling
- Batch request optimization

This architecture provides a robust, secure, and performant foundation for integrating MCP servers as first-class tools in the TCL-MCP system while maintaining compatibility with existing patterns and ensuring excellent performance characteristics.