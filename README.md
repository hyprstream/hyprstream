# TCL MCP Server with Namespace Support

A Model Context Protocol (MCP) server that provides TCL (Tool Command Language) execution capabilities with namespace-based tool management and versioning. Supports multiple TCL runtime implementations including Molt (safe subset) and the official TCL interpreter.

**Caution this is "vibe coded" and allows for autonomous AI development and execution on target systems. Use with caution)**

<!-- GitHub repository badges -->
[![GitHub Release](https://img.shields.io/github/v/release/cyberdione/mcp-tcl-udf-server.svg)](https://github.com/cyberdione/mcp-tcl-udf-server/releases)
[![GitHub Issues](https://img.shields.io/github/issues/cyberdione/mcp-tcl-udf-server.svg)](https://github.com/cyberdione/mcp-tcl-udf-server/issues)

<!-- Crates.io crate badges -->
[![crates.io](https://img.shields.io/crates/v/tcl-mcp-server.svg)](https://crates.io/crates/tcl-mcp-server)
[![Documentation](https://docs.rs/tcl-mcp-server/badge.svg)](https://docs.rs/tcl-mcp-server)

## TCL Runtime Support

This server supports multiple TCL runtime implementations with intelligent capability reporting and MCP integration:

- **Molt** (default): A safe, embedded TCL interpreter written in Rust. Provides a subset of TCL functionality with memory safety.
- **TCL** (optional): The official TCL interpreter via Rust bindings. Provides full TCL functionality.

### Runtime Selection

```bash
# Choose runtime via CLI argument
tcl-mcp-server --runtime molt --privileged
tcl-mcp-server --runtime tcl --privileged

# Choose runtime via environment variable
export TCL_MCP_RUNTIME=molt
tcl-mcp-server --privileged

# Priority: CLI args > Environment > Smart defaults (prefers Molt for safety)
```

**IMPORTANT NOTE**: Wrapper scripts with integrated defaults

Normally, you can simply use `tcl-mcp-server-admin` manually or by an agent swarm orchestrator. Worker agents may either use the `tcl-mcp-server-admin` (privileged), or for safety, `tcl-mcp-server` (non-privileged)

The following scripts default to the settings as indicated by name:

* tcl-mcp-server-ctcl
* tcl-mcp-server-admin-ctcl
* tcl-mcp-server-molt
* tcl-mcp-server-admin-molt

### Building with Different Runtimes

```bash
# Default build with Molt (recommended)
cargo build --release

# Build with official TCL interpreter (requires TCL installed on system)
cargo build --release --no-default-features --features tcl

# Build with both runtimes for maximum flexibility
cargo build --release --features molt,tcl
```

## Overview

This server provides TCL script execution through MCP with a Unix-like namespace system for organizing tools:
- `/bin/` - System tools (read-only)
- `/sbin/` - System administration tools (privileged)
- `/docs/` - Documentation tools (read-only)
- `/<user>/<package>/<tool>:<version>` - User tools with versioning

## Features

- **Namespace Organization**: Unix-like path structure for tool organization
- **Version Support**: Tools can have specific versions or use "latest"
- **Protected System Tools**: System tools in `/bin` and `/sbin` cannot be removed
- **User Tool Management**: Users can create tools in their own namespaces
- **Built-in Documentation**: Access Molt TCL interpreter docs and examples via `docs__molt_book`
- **MCP Server Integration**: Bridge external MCP servers as TCL tools with persistent configuration
- **Cross-Platform Persistence**: Platform-appropriate storage directories (XDG, macOS, Windows)
- **Auto-Start Servers**: Automatically reconnect to configured MCP servers on startup
- **MCP-Compatible Naming**: Internal namespace paths are converted to MCP-compatible names
- **Thread-Safe Architecture**: Handles TCL's non-thread-safe interpreter safely

## Namespace System

### Tool Paths
Tools are organized using a path-like structure:
- `/bin/tcl_execute` - Execute TCL scripts (system tool)
- `/bin/tcl_tool_list` - List available tools
- `/docs/molt_book` - Access Molt TCL documentation
- `/sbin/tcl_tool_add` - Add new tools (admin)
- `/sbin/tcl_tool_remove` - Remove tools (admin)
- `/alice/utils/reverse_string:1.0` - User tool with version
- `/bob/math/calculate:latest` - User tool with latest version

### MCP Name Mapping
Since MCP doesn't support forward slashes in tool names, paths are converted:
- `/bin/tcl_execute` → `bin__tcl_execute`
- `/docs/molt_book` → `docs__molt_book`
- `/sbin/tcl_tool_add` → `sbin__tcl_tool_add`
- `/alice/utils/reverse_string:1.0` → `user__alice__utils__reverse_string__v1_0`
- `/bob/math/calculate:latest` → `user__bob__math__calculate`

## MCP Server Integration

External MCP servers can be integrated as TCL tools:

```bash
# Add an MCP server (requires privileged mode)
tcl-mcp-server run sbin__mcp_add '{
  "id": "context7",
  "name": "Context7 Server", 
  "command": "npx",
  "args": ["-y", "@modelcontextprotocol/server-everything"],
  "auto_start": true
}'

# List tools (includes MCP server tools)
tcl-mcp-server list
# Output: bin__tcl_execute, mcp__context7__get_library_docs, ...

# Execute MCP server tool
tcl-mcp-server run mcp__context7__get_library_docs '{"query": "rust async"}'
```

**Persistence**: Server configurations are saved to platform-appropriate directories:
- **Linux**: `~/.local/share/tcl-mcp-server/mcp-index.json`  
- **macOS**: `~/Library/Application Support/tcl-mcp-server/mcp-index.json`
- **Windows**: `%APPDATA%\tcl-mcp-server\mcp-index.json`

See [MCP Server Integration Guide](docs/MCP_SERVER_INTEGRATION.md) for details.

## Installation

```bash
cargo build --release
```

## Usage

### Running the Server

#### Restricted Mode (Default)
```bash
cargo run
# or
./target/release/tcl-mcp-server
```

In restricted mode:
- Only `tcl_execute`, `tcl_tool_list`, and `docs___molt_book` tools are available  
- Tool management commands (`tcl_tool_add`, `tcl_tool_remove`) are disabled
- Provides safer TCL execution environment

#### Privileged Mode
```bash
cargo run -- --privileged
# or  
./target/release/tcl-mcp-server --privileged
# or use the generated wrapper (recommended for MCP integration)
./target/release/tcl-mcp-server-admin
```

In privileged mode:
- All tools are available including tool management
- Full TCL language access via Molt interpreter
- Enables dynamic tool creation and removal
- **Use with caution** - provides full TCL access

#### Admin Wrapper Script
The build process automatically generates `tcl-mcp-server-admin` wrapper script that enables privileged mode. This is useful for MCP integration since Claude's MCP configuration doesn't support command-line arguments:

```bash
# Generated automatically during build
./target/release/tcl-mcp-server-admin  # Equivalent to: tcl-mcp-server --privileged
./target/debug/tcl-mcp-server-admin    # Debug version
```

#### Command Line Options
```bash
./target/release/tcl-mcp-server --help
```

```
TCL MCP Server - Execute TCL scripts via Model Context Protocol

Usage: tcl-mcp-server [OPTIONS]

Options:
      --privileged  Enable privileged mode with full TCL access and tool management capabilities
  -h, --help        Print help
  -V, --version     Print version
```

### System Tools

#### 1. bin___tcl_execute *(Available in both modes)*
Execute a TCL script (path: `/bin/tcl_execute`)

```json
{
  "tool": "bin___tcl_execute",
  "parameters": {
    "script": "expr {2 + 2}"
  }
}
```

#### 2. bin___tcl_tool_list *(Available in both modes)*
List all available TCL tools (path: `/bin/tcl_tool_list`)

```json
{
  "tool": "bin___tcl_tool_list",
  "parameters": {
    "namespace": "alice",  // optional filter
    "filter": "utils*"     // optional name pattern
  }
}
```

#### 3. sbin___tcl_tool_add *(Privileged mode only)*
Add a new tool to a user namespace (path: `/sbin/tcl_tool_add`)

```json
{
  "tool": "sbin___tcl_tool_add",
  "parameters": {
    "user": "alice",
    "package": "utils",
    "name": "reverse_string",
    "version": "1.0",
    "description": "Reverse a string",
    "script": "return [string reverse $text]",
    "parameters": [{
      "name": "text",
      "description": "Text to reverse",
      "required": true,
      "type_name": "string"
    }]
  }
}
```

#### 4. sbin___tcl_tool_remove *(Privileged mode only)*
Remove a user tool (path: `/sbin/tcl_tool_remove`)

```json
{
  "tool": "sbin___tcl_tool_remove",
  "parameters": {
    "path": "/alice/utils/reverse_string:1.0"
  }
}
```

#### 5. docs___molt_book *(Available in both modes)*
Access Molt TCL interpreter documentation and examples (path: `/docs/molt_book`)

```json
{
  "tool": "docs___molt_book",
  "parameters": {
    "topic": "overview"  // Available: overview, basic_syntax, commands, examples, links
  }
}
```

**Available topics:**
- `overview` - Introduction to Molt TCL interpreter
- `basic_syntax` - TCL syntax fundamentals (variables, lists, control structures)
- `commands` - Common TCL commands reference
- `examples` - Practical TCL code examples
- `links` - Links to official Molt documentation and resources

### Runtime Capability Queries

LLMs can query runtime capabilities for intelligent code generation:

```json
{
  "tool": "tcl_runtime_info",
  "parameters": {
    "include_examples": true,
    "category_filter": "safe"
  }
}
```

**Response for Molt Runtime:**
```json
{
  "runtime_name": "Molt",
  "features": ["safe_subset", "memory_safe", "no_file_io"],
  "limitations": ["No file I/O operations", "No system commands"],
  "command_categories": {
    "core": ["set", "expr", "if", "while", "proc"],
    "string": ["string", "format", "regexp"],
    "list": ["list", "lappend", "llength"]
  },
  "examples": [
    {
      "category": "arithmetic",
      "prompt": "Calculate the area of a circle with radius 5",
      "code": "set radius 5\nset pi 3.14159\nexpr {$pi * $radius * $radius}"
    }
  ]
}
```

### Custom Tool Example

1. **Add a runtime-aware tool:**
```json
{
  "tool": "sbin___tcl_tool_add",
  "parameters": {
    "user": "myuser",
    "package": "text",
    "name": "word_count",
    "version": "1.0", 
    "description": "Count words in text (Safe for Molt runtime)",
    "script": "return [llength [split $text]]",
    "parameters": [{
      "name": "text",
      "description": "Text to count words in",
      "required": true,
      "type_name": "string"
    }]
  }
}
```

2. **Use the tool with runtime context:**
```json
{
  "tool": "user_myuser__text___word_count__v1_0",
  "parameters": {
    "text": "Hello world from TCL"
  }
}
```

Result: `4` (works safely in both Molt and full TCL)

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  MCP Client ├────►│  MCP Server  ├────►│TCL Executor │
│  (stdio)    │     │  (jsonrpc)   │     │  (thread)   │
└─────────────┘     └──────────────┘     └─────────────┘
                          │                      │
                          │  Namespace Manager   │
                          │  - Path mapping      │
                          │  - Version control   │
                          │  - Access control    │
                          └──────────────────────┘
```

## Security Considerations

### Privileged vs Restricted Mode

**Restricted Mode (Default - Recommended)**:
- Only `tcl_execute` and `tcl_tool_list` tools are available
- Tool management commands are disabled
- Safer for production environments
- Limits potential attack surface

**Privileged Mode**:
- ⚠️ **Use with caution** - provides full TCL interpreter access
- Enables dynamic tool creation and removal
- Should only be used in trusted environments
- Consider additional sandboxing for untrusted input

### General Security

- System tools in `/bin` and `/sbin` cannot be removed
- Only `/sbin` tools can manage the tool registry (privileged mode only)
- User tools are isolated in user namespaces
- TCL execution is handled by the Molt interpreter (memory-safe Rust implementation)
- Consider running in a container or sandbox for production use
- Always validate input when accepting TCL scripts from untrusted sources

## MCP Configuration & Integration

### Claude Desktop Integration

Add to your Claude Desktop `settings.json`:

**Safe Mode with Molt (Recommended):**
```json
{
  "mcpServers": {
    "tcl-safe": {
      "command": "/path/to/tcl-mcp/target/release/tcl-mcp-server",
      "args": ["--runtime", "molt", "--privileged"],
      "env": {
        "TCL_MCP_RUNTIME": "molt",
        "RUST_LOG": "info"
      }
    }
  }
}
```

**Full TCL Runtime (Advanced):**
```json
{
  "mcpServers": {
    "tcl-full": {
      "command": "/path/to/tcl-mcp/target/release/tcl-mcp-server",
      "args": ["--runtime", "tcl", "--privileged"],
      "env": {
        "TCL_MCP_RUNTIME": "tcl",
        "RUST_LOG": "info"
      }
    }
  }
}
```

### MCP Tool Properties & Example Prompts

The server provides rich MCP metadata including runtime-aware example prompts:

```json
{
  "name": "bin___tcl_execute",
  "description": "Execute TCL scripts (Runtime: Molt, Safety: Sandboxed)",
  "metadata": {
    "runtime_context": {
      "active_runtime": "Molt",
      "safety_level": "safe",
      "available_features": ["core", "string", "list", "math"],
      "limitations": ["no_file_io", "no_system_commands"]
    },
    "example_prompts": [
      {
        "category": "arithmetic",
        "prompt": "Calculate compound interest: principal=1000, rate=0.05, time=3",
        "code": "set principal 1000\nset rate 0.05\nset time 3\nexpr {$principal * pow(1 + $rate, $time)}",
        "expected_output": "1157.625"
      },
      {
        "category": "string_processing",
        "prompt": "Extract domain from email 'user@example.com'",
        "code": "set email \"user@example.com\"\nset parts [split $email \"@\"]\nlindex $parts 1",
        "expected_output": "example.com"
      },
      {
        "category": "data_structures", 
        "prompt": "Create a key-value store and lookup a value",
        "code": "array set store {name John age 30}\nset store(name)",
        "expected_output": "John"
      }
    ],
    "limitation_examples": [
      {
        "forbidden_operation": "file_read",
        "forbidden_code": "set fp [open \"file.txt\" r]",
        "why_forbidden": "Molt runtime doesn't support file I/O",
        "safe_alternative": "set data \"embedded content here\"",
        "alternative_explanation": "Use embedded strings instead of file operations"
      }
    ]
  }
}
```

### Claude Code MCP Integration

```bash
# Add restricted server
claude mcp add tcl /path/to/tcl-mcp/target/release/tcl-mcp-server

# Add privileged server
claude mcp add tcl-admin /path/to/tcl-mcp/target/release/tcl-mcp-server-admin
```

## Testing

Test scripts are available in the tests directory:

```bash
# Run the comprehensive test suite
./scripts/run_mcp_tests.sh

# Or run specific Python tests
python3 tests/test_bin_exec_tool_mcp.py
python3 tests/test_capabilities.py
```

## Building a Container Image

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
# Copy both the main binary and admin wrapper
COPY --from=builder /app/target/release/tcl-mcp-server /usr/bin/
COPY --from=builder /app/target/release/tcl-mcp-server-admin /usr/bin/
# Default to restricted mode
CMD ["/usr/bin/tcl-mcp-server"]
```

For privileged mode container:
```dockerfile
CMD ["/usr/bin/tcl-mcp-server-admin"]
```

## LLM Integration & Smart Code Generation

The server is designed for intelligent LLM integration with runtime-aware capabilities:

### Automatic Capability Detection
LLMs can query runtime capabilities and adapt code generation:

```python
# LLM workflow example
capabilities = await mcp.call_tool("tcl_runtime_info")

if "file_io" in capabilities["limitations"]:
    # Generate safe code for Molt
    code = "set data \"embedded content\"\nprocessing..."
else:
    # Generate full-featured code for TCL
    code = "set fp [open file.txt r]\nset data [read $fp]\nclose $fp"
```

### Context-Aware Error Messages
When operations aren't supported, the server provides helpful alternatives:

```json
{
  "error": "File operations not available in Molt runtime",
  "alternatives": [
    "Use embedded data instead of file reading",
    "Switch to full TCL runtime with --runtime tcl",
    "Process data passed as script parameters"
  ],
  "example_alternative": "set data \"embedded content\" # instead of: set fp [open file.txt r]"
}
```

### Runtime Comparison for LLMs

| Feature | Molt Runtime | TCL Runtime |
|---------|-------------|--------------|
| **Safety** | ✅ Sandboxed, memory-safe | ⚠️ Full system access |
| **File I/O** | ❌ Not supported | ✅ Full file operations |
| **System Commands** | ❌ Blocked | ✅ exec, system integration |
| **Networking** | ❌ Not available | ✅ Socket operations |
| **Best For** | Data processing, algorithms | System administration, file processing |

### Tool Name Patterns
Tool names follow a consistent LLM-friendly pattern:
```
"name": "user_alice__utils___reverse_string__v1_0",
"description": "Reverse a string [/alice/utils/reverse_string:1.0] (Runtime: Molt, Safe: ✅)"
```

## License

MIT
