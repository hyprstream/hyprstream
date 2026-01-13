# MCP Client Usage Guide

The TCL MCP Server includes a built-in MCP client that allows it to connect to and execute tools on other MCP servers. This enables powerful inter-server communication and tool composition.

## Overview

The MCP client provides:
- Dynamic registration of external MCP servers
- Automatic tool discovery
- JSON-RPC communication over stdio
- Connection management and health checking
- Tool execution with parameter validation

## Architecture

```
TCL MCP Server
├── MCP Server (handles incoming requests)
│   └── Tools: tcl_execute, exec_tool, execute_mcp, etc.
└── MCP Client (connects to external servers)
    ├── Server Registry
    ├── Connection Manager
    └── JSON-RPC Handler
```

## Using the MCP Client

### 1. Registering an MCP Server

Use the `/sbin/mcp_add` tool (requires privileged mode):

```json
{
  "id": "my-server",
  "name": "My MCP Server",
  "description": "Description of the server",
  "command": "npx",
  "args": ["my-mcp-server"],
  "env": {},
  "auto_start": true,
  "timeout_ms": 30000,
  "max_retries": 3
}
```

### 2. Executing Tools on Remote Servers

Use the `/bin/execute_mcp` tool:

```json
{
  "server_id": "my-server",
  "tool_name": "some_tool",
  "params": {
    "param1": "value1",
    "param2": 42
  },
  "response_format": "json",
  "timeout_ms": 5000
}
```

### 3. Managing MCP Servers

**List registered servers:**
```tcl
/sbin/mcp_list
```

**Remove a server:**
```json
{
  "server_id": "my-server",
  "force": false
}
```

## Connection Lifecycle

1. **Registration**: Server configuration is stored
2. **Connection**: Process spawned, MCP handshake performed
3. **Tool Discovery**: `tools/list` called to discover available tools
4. **Tool Execution**: `tools/call` used to execute specific tools
5. **Disconnection**: Graceful shutdown or forced termination

## Implementation Details

### Connection Management

The MCP client maintains persistent connections to external servers:
- Connections are reused for multiple tool calls
- Automatic reconnection on failure (configurable)
- Health checking to detect stale connections

### JSON-RPC Communication

All communication follows the MCP protocol specification:
- Request/response over newline-delimited JSON
- Proper error handling with JSON-RPC error codes
- Timeout protection for all operations

### Security Considerations

- MCP server management requires privileged mode
- Command injection protection through proper escaping
- Environment variable isolation
- Resource limits on spawned processes

## Example: Composing Tools

Here's how you might compose tools from multiple MCP servers:

```tcl
# In privileged mode, register two MCP servers
/sbin/mcp_add {
  "id": "data-processor",
  "command": "npx",
  "args": ["data-processor-mcp"],
  "auto_start": true
}

/sbin/mcp_add {
  "id": "ml-predictor", 
  "command": "python",
  "args": ["-m", "ml_predictor_mcp"],
  "auto_start": true
}

# Process data and make predictions
set processed_data [/bin/execute_mcp {
  "server_id": "data-processor",
  "tool_name": "clean_data",
  "params": {"input": $raw_data}
}]

set prediction [/bin/execute_mcp {
  "server_id": "ml-predictor",
  "tool_name": "predict",
  "params": {"data": $processed_data}
}]
```

## Error Handling

The MCP client provides detailed error messages:
- Connection failures
- Tool not found errors
- Parameter validation errors
- Timeout errors
- Protocol errors

## Performance Considerations

- Connection pooling reduces latency
- Configurable timeouts prevent hanging
- Async implementation allows concurrent operations
- Memory-efficient streaming for large responses

## Troubleshooting

### Server won't connect
- Check the command and args are correct
- Verify the MCP server is installed
- Look for error messages in the connection status
- Try running the command manually to test

### Tools not appearing
- Ensure the server implements `tools/list` correctly
- Check that the MCP handshake completed
- Verify the server is returning valid JSON schemas

### Timeouts
- Increase `timeout_ms` for slow operations
- Check network connectivity
- Verify the server is responsive

## API Reference

### McpServerConfig
- `id`: Unique identifier for the server
- `name`: Human-readable name
- `description`: Optional description
- `command`: Executable to run
- `args`: Command arguments
- `env`: Environment variables
- `auto_start`: Connect on registration
- `timeout_ms`: Operation timeout
- `max_retries`: Reconnection attempts

### ConnectionStatus
- `Disconnected`: Not connected
- `Connecting`: Connection in progress
- `Connected`: Ready for operations
- `Error(String)`: Connection failed

### Response Formats
- `json`: Parse as JSON
- `text`: Return as plain text
- `auto`: Detect format automatically