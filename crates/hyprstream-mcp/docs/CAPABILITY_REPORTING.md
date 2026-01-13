# TCL MCP Runtime Capability Reporting Design

## Overview

This document outlines the design for communicating TCL runtime capabilities to LLM clients through the MCP protocol. The design addresses the need for LLMs to understand which TCL interpreter is active, available language features, command safety levels, and operational constraints.

## 1. Enhanced Server Info Structure

### Current Implementation
```json
{
  "protocolVersion": "2024-11-05",
  "capabilities": {
    "tools": {}
  },
  "serverInfo": {
    "name": "tcl-mcp-server",
    "version": "1.0.0"
  }
}
```

### Enhanced Implementation
```json
{
  "protocolVersion": "2024-11-05",
  "capabilities": {
    "tools": {},
    "tcl": {
      "runtime": {
        "type": "molt|tcl",
        "version": "0.4.0",
        "features": ["basic", "string", "list", "math", "control"],
        "limitations": ["no_exec", "no_file_io", "no_network"],
        "safety_level": "restricted|privileged"
      },
      "commands": {
        "available": 47,
        "unsafe": ["exec", "file", "socket"],
        "restricted": ["puts", "gets"],
        "safe": ["string", "list", "expr", "set"]
      },
      "extensions": {
        "custom_tools": true,
        "filesystem_discovery": true,
        "persistence": true
      }
    }
  },
  "serverInfo": {
    "name": "tcl-mcp-server",
    "version": "1.0.0",
    "tcl_runtime": "Molt 0.4.0",
    "build_features": ["molt"],
    "safety_mode": "restricted|privileged"
  }
}
```

## 2. Runtime Capability Metadata

### TCL Runtime Information
```rust
#[derive(Debug, Serialize, Clone)]
pub struct TclRuntimeCapabilities {
    /// Runtime type (molt, tcl)
    pub runtime_type: String,
    /// Runtime version
    pub version: String,
    /// Available language features
    pub features: Vec<String>,
    /// Known limitations
    pub limitations: Vec<String>,
    /// Safety level (restricted/privileged)
    pub safety_level: String,
    /// Total available commands
    pub command_count: usize,
    /// Command safety classification
    pub command_safety: CommandSafetyInfo,
}

#[derive(Debug, Serialize, Clone)]
pub struct CommandSafetyInfo {
    /// Commands that are completely safe
    pub safe: Vec<String>,
    /// Commands with restrictions
    pub restricted: Vec<String>,
    /// Commands that are potentially unsafe
    pub unsafe: Vec<String>,
    /// Commands not available in this runtime
    pub unavailable: Vec<String>,
}
```

### Feature Detection
```rust
pub trait TclRuntime {
    // Existing methods...
    
    /// Get runtime capabilities for MCP reporting
    fn get_capabilities(&self) -> TclRuntimeCapabilities;
    
    /// Get detailed command information
    fn get_command_info(&self) -> CommandSafetyInfo;
    
    /// Check if a specific feature is supported
    fn supports_feature(&self, feature: &str) -> bool;
    
    /// Get runtime-specific limitations
    fn get_limitations(&self) -> Vec<String>;
}
```

## 3. New MCP Methods for Capability Queries

### Method: `tcl/capabilities`
Returns detailed TCL runtime capability information.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tcl/capabilities",
  "params": {}
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "runtime": {
      "type": "molt",
      "version": "0.4.0",
      "name": "Molt TCL Interpreter",
      "implementation": "rust",
      "thread_safe": false,
      "memory_safe": true
    },
    "features": {
      "core_commands": ["set", "expr", "if", "for", "foreach", "while", "proc"],
      "string_operations": ["string", "append", "format"],
      "list_operations": ["list", "lindex", "llength", "lappend", "lrange"],
      "math_operations": ["expr", "incr"],
      "control_structures": ["if", "for", "foreach", "while", "switch"],
      "procedures": ["proc", "return", "uplevel", "upvar"],
      "variables": ["set", "unset", "global", "variable"]
    },
    "limitations": {
      "file_io": "No file I/O commands available",
      "network": "No network commands available", 
      "exec": "No external process execution",
      "unsafe_commands": "Commands like 'exec', 'open', 'socket' not available",
      "tcl_version": "Implements TCL 8.6 subset, not full compatibility"
    },
    "safety": {
      "level": "restricted",
      "sandboxed": true,
      "command_filtering": true,
      "privilege_escalation": false
    },
    "commands": {
      "total_available": 47,
      "safe": ["string", "list", "expr", "set", "puts", "incr"],
      "restricted": [],
      "unsafe": [],
      "unavailable": ["exec", "open", "file", "socket", "cd", "pwd"]
    }
  }
}
```

### Method: `tcl/commands`
Returns available TCL commands with safety classifications.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tcl/commands",
  "params": {
    "filter": "safe|restricted|unsafe|all",
    "category": "string|list|math|control|io|system"
  }
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "commands": [
      {
        "name": "string",
        "safety": "safe",
        "category": "string",
        "description": "String manipulation operations",
        "subcommands": ["length", "index", "range", "tolower", "toupper"],
        "available": true,
        "restrictions": []
      },
      {
        "name": "exec",
        "safety": "unsafe",
        "category": "system", 
        "description": "Execute external commands",
        "available": false,
        "reason": "Security restriction in current runtime"
      }
    ],
    "summary": {
      "total": 2,
      "safe": 1,
      "restricted": 0,
      "unsafe": 0,
      "unavailable": 1
    }
  }
}
```

## 4. Tool Description Enhancements

### Enhanced Tool Descriptions
Each tool description should include runtime context:

```json
{
  "name": "bin___tcl_execute",
  "description": "Execute a TCL script using Molt interpreter (safe subset)",
  "inputSchema": {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
      "script": {
        "type": "string",
        "description": "TCL script to execute (Molt runtime - safe commands only)"
      }
    },
    "required": ["script"]
  },
  "metadata": {
    "runtime": "molt",
    "safety_level": "restricted",
    "available_commands": 47,
    "limitations": ["no_file_io", "no_exec", "no_network"]
  }
}
```

## 5. Implementation Strategy

### Phase 1: Core Infrastructure
1. Extend `TclRuntime` trait with capability methods
2. Implement capability detection for Molt runtime
3. Add capability caching to avoid repeated introspection

### Phase 2: MCP Integration
1. Enhance `initialize` response with runtime info
2. Add new MCP methods (`tcl/capabilities`, `tcl/commands`)
3. Update tool descriptions with metadata

### Phase 3: Advanced Features
1. Runtime switching capability reporting
2. Dynamic capability updates
3. Performance metrics inclusion

## 6. Runtime-Specific Implementations

### Molt Runtime Capabilities
```rust
impl TclRuntime for MoltRuntime {
    fn get_capabilities(&self) -> TclRuntimeCapabilities {
        TclRuntimeCapabilities {
            runtime_type: "molt".to_string(),
            version: "0.4.0".to_string(),
            features: vec![
                "basic".to_string(),
                "string".to_string(), 
                "list".to_string(),
                "math".to_string(),
                "control".to_string(),
                "procedures".to_string(),
            ],
            limitations: vec![
                "no_file_io".to_string(),
                "no_exec".to_string(),
                "no_network".to_string(),
                "no_unsafe_commands".to_string(),
            ],
            safety_level: "restricted".to_string(),
            command_count: self.get_available_command_count(),
            command_safety: self.get_command_safety_info(),
        }
    }
    
    fn get_command_info(&self) -> CommandSafetyInfo {
        CommandSafetyInfo {
            safe: vec![
                "string".to_string(), "list".to_string(), "expr".to_string(),
                "set".to_string(), "puts".to_string(), "incr".to_string(),
                "if".to_string(), "for".to_string(), "foreach".to_string(),
                "while".to_string(), "proc".to_string(), "return".to_string(),
            ],
            restricted: vec![],
            unsafe: vec![],
            unavailable: vec![
                "exec".to_string(), "open".to_string(), "file".to_string(),
                "socket".to_string(), "cd".to_string(), "pwd".to_string(),
            ],
        }
    }
}
```

### Official TCL Runtime Capabilities
```rust
impl TclRuntime for TclInterpreter {
    fn get_capabilities(&self) -> TclRuntimeCapabilities {
        TclRuntimeCapabilities {
            runtime_type: "tcl".to_string(),
            version: "8.6.13".to_string(), // Detected dynamically
            features: vec![
                "full_tcl".to_string(),
                "file_io".to_string(),
                "network".to_string(), 
                "exec".to_string(),
                "packages".to_string(),
                "namespaces".to_string(),
            ],
            limitations: vec![], // Fewer limitations in full TCL
            safety_level: "privileged".to_string(),
            command_count: self.get_available_command_count(),
            command_safety: self.get_command_safety_info(),
        }
    }
}
```

## 7. LLM Code Generation Guidance

### Runtime Detection Patterns
LLMs should check runtime capabilities before generating code:

```tcl
# Check if file operations are available
if {[info commands file] ne ""} {
    # Use file operations
} else {
    # Use alternative approach
}

# Check for string operations
if {[info commands string] ne ""} {
    set result [string length $text]
} else {
    # Fallback method
}
```

### Safe Code Generation Rules
1. **Always check command availability** before using advanced features
2. **Prefer safe commands** like `string`, `list`, `expr` over system commands  
3. **Use capability metadata** to determine appropriate code patterns
4. **Provide fallbacks** for missing functionality
5. **Respect safety levels** - don't generate unsafe code in restricted mode

## 8. Benefits

### For LLMs
- **Better Code Generation**: Understanding available commands prevents invalid code
- **Safety Awareness**: Knowing safety levels helps generate appropriate code
- **Runtime Adaptation**: Can adapt code generation to specific TCL implementation
- **Error Prevention**: Avoid suggesting unavailable features

### For Users
- **Transparency**: Clear understanding of what's available in current mode
- **Debugging**: Better error messages when commands aren't available
- **Security**: Clear indication of safety levels and restrictions

### For Developers
- **Extensibility**: Easy to add new runtimes with capability reporting
- **Monitoring**: Track which features are being used
- **Optimization**: Focus development on most-used capabilities

## 9. Future Enhancements

1. **Dynamic Capability Updates**: Report changes when switching runtimes
2. **Performance Metrics**: Include execution time and memory usage data
3. **Custom Command Registration**: Report dynamically added commands
4. **Security Policy Integration**: Include security policy information
5. **Extension Discovery**: Report available TCL extensions and packages

This design provides comprehensive capability reporting while maintaining backward compatibility with existing MCP clients and enabling rich, context-aware TCL code generation by LLMs.