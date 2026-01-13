# TCL MCP Server

A Model Context Protocol (MCP) server that enables AI agents to execute TCL scripts and manage MCP tool ecosystems. Built with safety and developer experience in mind.

[![crates.io](https://img.shields.io/crates/v/tcl-mcp-server.svg)](https://crates.io/crates/tcl-mcp-server)
[![Documentation](https://docs.rs/tcl-mcp-server/badge.svg)](https://docs.rs/tcl-mcp-server)

## Quick Start

```bash
# Install and run (safe mode with Molt runtime)
cargo install tcl-mcp-server
tcl-mcp-server

# Or build from source
git clone https://github.com/cyberdione/mcp-tcl-udf-server
cd mcp-tcl-udf-server
cargo build --release
./target/release/tcl-mcp-server

# OPTIONAL: Build with full unsafe TCL runtime (requires system TCL installation)
# cargo build --release --features tcl
```

## What It Does

- **Execute TCL Scripts**: Run TCL code through MCP with intelligent safety controls
- **Manage MCP Ecosystem**: Add, remove, and orchestrate other MCP servers
- **Safe by Default**: Uses Molt (memory-safe TCL) with sandboxed execution
- **Tool Management**: Create, version, and organize custom tools
- **Cross-Platform**: Works on Linux, macOS, and Windows

## Runtime Options

Choose between two TCL runtime implementations:

### 🔒 **Molt Runtime (Default - Safe)**
- **Memory-safe**: Written in Rust with built-in safety guarantees
- **Sandboxed**: No file I/O, no system commands, no network access
- **Subset**: Core TCL functionality for data processing and algorithms
- **Recommended**: For production use and untrusted environments
- **Documentation**: [Molt TCL Book](https://wduquette.github.io/molt/)

### ⚠️ **TCL Runtime (Complete - Unsafe)**
- **Full functionality**: Complete TCL language with all features
- **System access**: File I/O, system commands, network operations
- **Powerful**: Advanced scripting capabilities and system integration
- **Risky**: Requires trusted environment and careful input validation
- **Documentation**: [Official TCL Documentation](https://www.tcl-lang.org/doc/)

## Core Features

### 🔒 **Safety First**
- **Restricted Mode** (default): Safe TCL execution with limited commands
- **Privileged Mode**: Full TCL access for advanced use cases
- **Runtime Selection**: Choose between safe (Molt) and complete (TCL) implementations

### 🛠️ **MCP Management**
```bash
# Add external MCP servers
tcl-mcp-server mcp add claude-flow "Claude Flow" -- npx claude-flow@alpha mcp start

# List all servers
tcl-mcp-server mcp list

# Test connectivity
tcl-mcp-server mcp ping claude-flow
```

### 📦 **Tool Organization**
Tools are organized using a namespace system with MCP-compatible naming:
- `bin__tcl_execute` - Execute TCL scripts
- `user__alice__utils__reverse_string` - User-created tools
- `mcp__context7__get_library_docs` - External MCP server tools

## Usage

### Running the Server

**Default (Read-only mode)**
```bash
tcl-mcp-server
```

**Privileged Mode (Save/store scripts)**
```bash
tcl-mcp-server --privileged
# or use the admin wrapper
tcl-mcp-server-admin
```

### Essential Commands

```bash
# Execute TCL directly
tcl-mcp-server run tcl_execute '{"script": "expr {2 + 2}"}'

# List available tools
tcl-mcp-server list

# Get tool information
tcl-mcp-server info tcl_execute

# Manage MCP servers
tcl-mcp-server mcp add my-server "My Server" -- node server.js
tcl-mcp-server mcp remove my-server
```

### MCP Client Integration

**Claude Desktop**
```json
{
  "mcpServers": {
    "tcl": {
      "command": "/path/to/tcl-mcp-server",
      "args": ["--runtime", "molt", "--privileged"]
    }
  }
}
```

**Claude Code**
```bash
claude mcp add tcl /path/to/tcl-mcp-server
```

## Built-in Tools

### Core Tools (Always Available)

**`bin__tcl_execute`** - Execute TCL scripts
```json
{
  "script": "set x 5; set y 10; expr {$x + $y}"
}
```

**`bin__list_tools`** - List available tools
```json
{
  "namespace": "user",
  "filter": "utils*"
}
```

**`docs__molt_book`** - Access TCL documentation
```json
{
  "topic": "basic_syntax"
}
```

### Management Tools (Privileged Mode Only)

**`sbin__tcl_tool_add`** - Create custom tools
```json
{
  "user": "alice",
  "package": "utils",
  "name": "reverse_string",
  "version": "1.0",
  "description": "Reverse a string",
  "script": "return [string reverse $text]",
  "parameters": [
    {
      "name": "text",
      "description": "Text to reverse",
      "required": true,
      "type_name": "string"
    }
  ]
}
```

**`sbin__mcp_add`** - Add MCP servers programmatically
```json
{
  "id": "context7",
  "name": "Context7 Server",
  "command": "npx",
  "args": ["@modelcontextprotocol/server-everything"],
  "auto_start": true
}
```

## Compilation and Runtime Configuration

### Build Options

The server supports two TCL runtime implementations that must be selected at compile time:

#### Default Build (Molt Runtime - Safe)
```bash
# Build with Molt runtime only (recommended)
cargo build --release

# The resulting binary uses Molt by default
./target/release/tcl-mcp-server
```

#### Build with TCL Runtime (Complete but Unsafe)
```bash
# Build with full TCL runtime (requires system TCL installation)
cargo build --release --no-default-features --features tcl

# The resulting binary uses full TCL
./target/release/tcl-mcp-server
```

#### Build with Both Runtimes
```bash
# Build with both runtimes available (maximum flexibility)
cargo build --release --features molt,tcl

# Select runtime at startup
./target/release/tcl-mcp-server --runtime molt   # Safe mode
./target/release/tcl-mcp-server --runtime tcl    # Complete mode
```

### Runtime Selection (Multi-Runtime Builds)

When built with multiple runtimes, you can choose at startup:

```bash
# Command line selection
tcl-mcp-server --runtime molt          # Safe: Molt runtime
tcl-mcp-server --runtime tcl           # Unsafe: Full TCL runtime

# Environment variable
export TCL_MCP_RUNTIME=molt
tcl-mcp-server

# Priority: CLI args > Environment > Default (Molt)
```

### System Requirements

**For Molt Runtime (default):**
- Rust toolchain only
- No external dependencies
- Works on all platforms

**For TCL Runtime:**
- System TCL installation required (8.6+)
- Development headers needed for compilation
- Platform-specific setup:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install tcl-dev

  # macOS
  brew install tcl-tk

  # Windows
  # Install TCL from https://www.tcl-lang.org/software/tcltk/
  ```

### Pre-built Wrapper Scripts

The build process automatically generates convenience wrappers:

```bash
# Generated during build
./target/release/tcl-mcp-server-admin      # Privileged mode
./target/release/tcl-mcp-server-molt       # Force Molt runtime
./target/release/tcl-mcp-server-admin-molt # Privileged + Molt
./target/release/tcl-mcp-server-ctcl       # Force TCL runtime
./target/release/tcl-mcp-server-admin-ctcl # Privileged + TCL
```

## MCP Server Management

### Adding Servers

```bash
# Basic server
tcl-mcp-server mcp add my-server "My Server" -- node server.js

# With environment variables
tcl-mcp-server mcp add my-server "My Server" \
  --env "NODE_ENV=production" \
  --env "API_KEY=secret" \
  -- node server.js

# Custom timeout and retry settings
tcl-mcp-server mcp add my-server "My Server" \
  --timeout-ms 60000 \
  --max-retries 5 \
  -- node server.js
```

### Server Information

```bash
# List all servers
tcl-mcp-server mcp list

# Detailed view
tcl-mcp-server mcp list --detailed

# Server details
tcl-mcp-server mcp info my-server
```

### Connection Management

```bash
# Manual connection
tcl-mcp-server mcp connect my-server

# Test connectivity
tcl-mcp-server mcp ping my-server

# Disconnect
tcl-mcp-server mcp disconnect my-server

# Remove server
tcl-mcp-server mcp remove my-server
```

## Security Model

### Default Security (Recommended)

- **Restricted Mode**: Only essential tools available
- **Molt Runtime**: Memory-safe, sandboxed execution (see [Molt docs](https://wduquette.github.io/molt/))
- **No File I/O**: Prevents unauthorized file access
- **No System Commands**: Blocks system-level operations

### Privileged Mode

⚠️ **Use with caution**
- Full TCL language access
- Tool management capabilities
- System-level operations possible (especially with TCL runtime)
- Recommended for trusted environments only
- **With TCL Runtime**: Complete system access (see [TCL docs](https://www.tcl-lang.org/doc/))

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  AI Agent   ├────►│  MCP Server  ├────►│TCL Executor │
│  (Claude)   │     │  (JSON-RPC)  │     │  (Molt)     │
└─────────────┘     └──────────────┘     └─────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │ MCP Manager │
                    │ (External   │
                    │  Servers)   │
                    └─────────────┘
```

## Advanced Usage

### Creating Custom Tools

1. **Add a tool** (requires privileged mode):
```bash
tcl-mcp-server run sbin__tcl_tool_add '{
  "user": "dev",
  "package": "math",
  "name": "fibonacci",
  "version": "1.0",
  "description": "Calculate Fibonacci number",
  "script": "proc fib {n} { if {$n <= 1} {return $n} else {return [expr {[fib [expr {$n-1}]] + [fib [expr {$n-2}]]}]} }; return [fib $n]",
  "parameters": [
    {
      "name": "n",
      "description": "Number to calculate Fibonacci for",
      "required": true,
      "type_name": "integer"
    }
  ]
}'
```

2. **Use the tool**:
```bash
tcl-mcp-server run user__dev__math__fibonacci '{"n": 10}'
```

### Runtime Capability Detection

Query runtime capabilities for intelligent code generation:
```bash
tcl-mcp-server run tcl_runtime_info '{
  "include_examples": true,
  "category_filter": "safe"
}'
```

**Runtime Feature Comparison:**

| Feature | Molt Runtime | TCL Runtime |
|---------|-------------|-------------|
| **Memory Safety** | ✅ Rust-based, memory-safe | ⚠️ C-based, manual memory management |
| **File I/O** | ❌ Blocked for security | ✅ Full file operations |
| **System Commands** | ❌ No `exec` or system calls | ✅ Complete system integration |
| **Networking** | ❌ No socket operations | ✅ Full network capabilities |
| **Performance** | ⚡ Fast startup, low overhead | 🐌 Slower startup, higher memory usage |
| **Compatibility** | 📚 Core TCL subset | 🔧 Full TCL language + extensions |
| **Use Cases** | Data processing, algorithms, safe scripting | System administration, complex applications |
| **Documentation** | [Molt Book](https://wduquette.github.io/molt/) | [TCL Documentation](https://www.tcl-lang.org/doc/) |

### Container Deployment

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/tcl-mcp-server /usr/bin/
COPY --from=builder /app/target/release/tcl-mcp-server-admin /usr/bin/
CMD ["/usr/bin/tcl-mcp-server"]
```

## Testing

```bash
# Run the test suite
./scripts/run_mcp_tests.sh

# Test specific functionality
python3 tests/test_bin_exec_tool_mcp.py
```

## Data Storage

Server configurations are stored in platform-appropriate locations:
- **Linux**: `~/.local/share/tcl-mcp-server/`
- **macOS**: `~/Library/Application Support/tcl-mcp-server/`
- **Windows**: `%APPDATA%\tcl-mcp-server\`

## Troubleshooting

### Common Issues

**Server won't start**
```bash
# Check runtime availability
tcl-mcp-server --runtime molt --privileged

# Enable debug logging
RUST_LOG=debug tcl-mcp-server
```

**MCP server connection fails**
```bash
# Test connectivity
tcl-mcp-server mcp ping server-id

# Check server logs
TCL_MCP_DEBUG_STDERR=1 tcl-mcp-server
```

**Tool not found**
```bash
# List available tools
tcl-mcp-server list

# Check specific namespace
tcl-mcp-server list --namespace user
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details
