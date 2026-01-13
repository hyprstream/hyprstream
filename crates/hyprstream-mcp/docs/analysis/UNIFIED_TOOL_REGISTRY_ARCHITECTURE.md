# Unified Tool Registry Architecture

## Executive Summary

This document presents a comprehensive architecture design for a unified tool registry system that integrates Model Context Protocol (MCP), WebAssembly (WASM), and NPX capabilities into the existing TCL MCP server codebase. The design builds upon the current namespace-based tool management and runtime abstraction layers to create a flexible, performant, and extensible plugin ecosystem.

## Current Architecture Analysis

### Existing Strengths

1. **Modular Design**: Clean separation of concerns with distinct modules for runtime, namespace, persistence, and tool discovery
2. **Runtime Abstraction**: Pluggable TCL runtime system (Molt/TCL) via trait-based interface
3. **Namespace System**: Unix-like hierarchical tool organization (`/bin`, `/sbin`, `/users`)
4. **MCP Integration**: Native MCP server implementation with JSON-RPC protocol
5. **Persistence Layer**: File-based tool storage with metadata and indexing
6. **Tool Discovery**: Filesystem-based tool scanning and registration

### Extension Points Identified

1. **Runtime Trait (`TclRuntime`)**: Perfect foundation for adding WASM and NPX runtimes
2. **Tool Discovery System**: Can be extended to discover different tool types
3. **Persistence Layer**: Pluggable storage backend ready for multi-format support
4. **Command Processing (`TclCommand` enum)**: Extensible command routing system
5. **MCP Tool Registration**: Dynamic tool schema generation and registration

## Unified Architecture Design

### 1. Core Registry Architecture

```
┌─────────────────────────────────────────────────────────┐
│                 Unified Tool Registry                   │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐│
│  │ MCP Tools   │ │ WASM Tools  │ │ NPX Tools           ││
│  │ (.mcp)      │ │ (.wasm)     │ │ (.js/.ts/.json)     ││
│  └─────────────┘ └─────────────┘ └─────────────────────┘│
├─────────────────────────────────────────────────────────┤
│               Tool Registry Manager                     │
│  ┌─────────────────────────────────────────────────────┐│
│  │ • Tool Discovery & Registration                     ││
│  │ • Runtime Selection & Execution                     ││
│  │ • Schema Validation & Transformation               ││
│  │ • Performance Monitoring & Caching                 ││
│  └─────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────┤
│                Runtime Abstraction Layer               │
│  ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐ │
│  │ TCL Runtime  │ │ WASM Runtime │ │ NPX Runtime     │ │
│  │ (Molt/TCL)   │ │ (wasmtime)   │ │ (tokio process) │ │
│  └──────────────┘ └──────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                  Storage & Persistence                  │
│  ┌─────────────────────────────────────────────────────┐│
│  │ • Multi-format tool storage (TCL/WASM/NPX)          ││
│  │ • Unified metadata and indexing                     ││
│  │ • Performance caching and hot reloading             ││
│  │ • Backup and versioning                             ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
```

### 2. Enhanced Runtime System

#### Extended Runtime Trait

```rust
pub trait UnifiedRuntime: Send + Sync {
    /// Runtime identification
    fn runtime_type(&self) -> RuntimeType;
    fn name(&self) -> &'static str;
    fn version(&self) -> &'static str;
    
    /// Execution interface
    async fn execute(&mut self, tool: &ToolDefinition, params: Value) -> Result<ToolResult>;
    async fn validate_tool(&self, tool: &ToolDefinition) -> Result<ValidationResult>;
    
    /// Capabilities and performance
    fn capabilities(&self) -> RuntimeCapabilities;
    fn performance_profile(&self) -> PerformanceProfile;
    
    /// Lifecycle management
    async fn initialize(&mut self) -> Result<()>;
    async fn cleanup(&mut self) -> Result<()>;
    async fn health_check(&self) -> Result<HealthStatus>;
}

#[derive(Debug, Clone)]
pub enum RuntimeType {
    Tcl(TclVariant),
    Wasm(WasmVariant),
    Npx(NpxVariant),
    Hybrid(Vec<RuntimeType>), // For multi-runtime tools
}

#[derive(Debug, Clone)]
pub enum TclVariant { Molt, Official }

#[derive(Debug, Clone)]
pub enum WasmVariant { Wasmtime, Wasmer, WasmEdge }

#[derive(Debug, Clone)]
pub enum NpxVariant { Node, Deno, Bun }
```

#### Runtime-Specific Implementations

1. **WASM Runtime Integration**
   - `wasmtime` as primary WASM engine
   - WASI support for file system and network access
   - Component model for tool composition
   - Bytecode caching and JIT optimization

2. **NPX Runtime Integration**
   - Node.js process spawning with `tokio::process`
   - Package resolution via `npm/yarn/pnpm`
   - TypeScript compilation support
   - ESM and CommonJS module loading

3. **Enhanced TCL Runtime**
   - Keep existing Molt/TCL implementations
   - Add performance profiling and monitoring
   - Enhanced security sandboxing options

### 3. Tool Discovery and Registration System

#### Multi-Format Tool Discovery

```rust
pub struct UnifiedToolDiscovery {
    discoverers: HashMap<ToolFormat, Box<dyn ToolDiscoverer>>,
    cache: Arc<RwLock<DiscoveryCache>>,
    watchers: Vec<FileWatcher>,
}

pub trait ToolDiscoverer: Send + Sync {
    async fn discover(&self, path: &Path) -> Result<Vec<DiscoveredTool>>;
    fn supported_extensions(&self) -> &[&str];
    fn tool_format(&self) -> ToolFormat;
}

#[derive(Debug, Clone)]
pub enum ToolFormat {
    Tcl,
    Wasm,
    Npx,
}

// Specific discoverers
pub struct TclDiscoverer;
pub struct WasmDiscoverer;
pub struct NpxDiscoverer;
```

#### Tool Definition Evolution

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedToolDefinition {
    pub metadata: ToolMetadata,
    pub runtime: RuntimeRequirement,
    pub implementation: ToolImplementation,
    pub interface: ToolInterface,
    pub performance: PerformanceHints,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolImplementation {
    Tcl { script: String },
    Wasm { 
        module: Vec<u8>,
        entry_point: String,
        imports: Vec<WasmImport>,
    },
    Npx {
        package: Option<String>, // For external packages
        script: Option<String>,  // For inline scripts
        entry_point: String,
        dependencies: Vec<NpxDependency>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeRequirement {
    pub preferred: RuntimeType,
    pub fallbacks: Vec<RuntimeType>,
    pub constraints: RuntimeConstraints,
}
```

### 4. Performance Optimization Strategy

#### Caching and Hot Reloading

1. **Runtime Instance Pooling**
   - Pre-warmed runtime instances for common tool types
   - Connection pooling for Node.js processes
   - WASM module compilation caching

2. **Tool Execution Caching**
   - Result caching for deterministic tools
   - Incremental compilation for TypeScript/NPX tools
   - WASM bytecode caching with version invalidation

3. **Hot Reloading and Development Mode**
   - File system watchers for development
   - Incremental recompilation and reloading
   - Live tool updates without server restart

#### Execution Pipeline Optimization

```rust
pub struct ExecutionPipeline {
    stages: Vec<Box<dyn PipelineStage>>,
    metrics: Arc<ExecutionMetrics>,
}

pub trait PipelineStage: Send + Sync {
    async fn process(&self, context: &mut ExecutionContext) -> Result<()>;
    fn stage_name(&self) -> &'static str;
}

// Pipeline stages:
// 1. Parameter validation and transformation
// 2. Runtime selection and preparation
// 3. Tool execution
// 4. Result transformation and caching
// 5. Performance metrics collection
```

### 5. Plugin System Architecture

#### Plugin Interface

```rust
pub trait ToolPlugin: Send + Sync {
    fn plugin_info(&self) -> PluginInfo;
    fn supported_formats(&self) -> Vec<ToolFormat>;
    
    async fn register_tools(&self, registry: &mut ToolRegistry) -> Result<()>;
    async fn handle_execution(&self, request: ExecutionRequest) -> Result<ExecutionResponse>;
    
    // Optional hooks
    async fn on_tool_added(&self, tool: &UnifiedToolDefinition) -> Result<()> { Ok(()) }
    async fn on_tool_removed(&self, tool_id: &str) -> Result<()> { Ok(()) }
    async fn on_server_startup(&self) -> Result<()> { Ok(()) }
    async fn on_server_shutdown(&self) -> Result<()> { Ok(()) }
}

#[derive(Debug, Clone)]
pub struct PluginInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub capabilities: Vec<PluginCapability>,
}
```

#### Built-in Plugins

1. **TCL Plugin** (existing functionality)
2. **WASM Plugin** (new)
3. **NPX Plugin** (new)
4. **Hybrid Plugin** (for multi-runtime tools)

### 6. Backwards Compatibility and Migration

#### Migration Strategy

1. **Phase 1**: Extend existing interfaces without breaking changes
2. **Phase 2**: Add new runtime support alongside existing TCL support
3. **Phase 3**: Migrate existing tools to unified format (optional)
4. **Phase 4**: Deprecate old interfaces with long transition period

#### Compatibility Layer

```rust
pub struct CompatibilityLayer {
    legacy_server: Option<TclMcpServer>,
    unified_server: UnifiedMcpServer,
    migration_tools: MigrationToolset,
}

impl CompatibilityLayer {
    pub async fn handle_legacy_request(&self, request: &str) -> Result<String> {
        // Route legacy requests through compatibility shim
        // Gradually migrate tools to new format
    }
}
```

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)
1. Extract and enhance runtime abstraction layer
2. Create unified tool definition format
3. Implement basic plugin system architecture
4. Add performance monitoring infrastructure

### Phase 2: WASM Integration (Weeks 5-8)
1. Implement WASM runtime with wasmtime
2. Create WASM tool discoverer and validator
3. Add WASM compilation and caching pipeline
4. Implement WASI integration for system access

### Phase 3: NPX Integration (Weeks 9-12)
1. Implement NPX runtime with Node.js process management
2. Add package resolution and dependency management
3. Create TypeScript compilation pipeline
4. Implement NPM package integration

### Phase 4: Optimization and Polish (Weeks 13-16)
1. Performance profiling and optimization
2. Advanced caching and hot reloading
3. Comprehensive testing and documentation
4. Migration tools and compatibility validation

## Performance Benchmarks and Targets

### Target Performance Metrics

1. **Tool Discovery**: < 100ms for 1000 tools
2. **Tool Execution**: 
   - TCL: < 10ms overhead
   - WASM: < 50ms cold start, < 5ms warm start
   - NPX: < 200ms cold start, < 20ms warm start
3. **Memory Usage**: < 50MB base + 1-10MB per active runtime
4. **Concurrent Executions**: Support 100+ concurrent tool executions

### Optimization Strategies

1. **Runtime Pooling**: Pre-warmed runtime instances
2. **Compilation Caching**: Aggressive caching of compiled artifacts
3. **Lazy Loading**: Load runtimes and tools on demand
4. **Resource Limits**: Configurable limits for memory and execution time

## Security and Sandboxing

### Security Model

1. **Runtime Isolation**: Each runtime provides its own sandboxing
2. **Resource Limits**: CPU, memory, and time constraints per tool
3. **Network Access Control**: Configurable network policies
4. **File System Access**: WASI-based file system virtualization
5. **Privilege Escalation Prevention**: Tool execution in restricted contexts

### Security Features by Runtime

- **TCL**: Existing Molt safety + additional command filtering
- **WASM**: Full WASI sandboxing with capability-based security
- **NPX**: Node.js vm2 sandboxing + process isolation

## Conclusion

This unified tool registry architecture provides a comprehensive foundation for integrating MCP, WASM, and NPX capabilities while maintaining the existing TCL MCP server's strengths. The design emphasizes:

1. **Extensibility**: Plugin-based architecture for adding new runtimes
2. **Performance**: Optimization strategies for sub-100ms tool execution
3. **Compatibility**: Seamless migration path for existing tools
4. **Security**: Multi-layered sandboxing and isolation
5. **Developer Experience**: Rich tooling for development and debugging

The phased implementation approach ensures minimal disruption to existing functionality while progressively adding new capabilities, making this a practical and achievable enhancement to the current system.