# bin__exec_tool - Tool Discovery and Execution System

## Overview

The `bin__exec_tool` is a powerful TCL MCP tool that enables dynamic discovery and execution of tools from the filesystem. It extends the TCL MCP server's capabilities by allowing tools to be loaded from disk rather than being hardcoded or dynamically added through the API.

## Features

### 1. Tool Discovery (`bin__discover_tools`)
- Automatically discovers TCL tools from the filesystem
- Scans predefined directories: `tools/bin`, `tools/sbin`, `tools/docs`
- Supports user tools in `tools/users/<username>/<package>/`
- Parses tool metadata from header comments
- Indexes tools for fast lookup

### 2. Tool Execution (`bin__exec_tool`)
- Executes any discovered tool by its path
- Validates required parameters
- Passes parameters as TCL variables
- Supports all namespace types (bin, sbin, docs, user)

### 3. Tool Metadata Format
Tools can include metadata in header comments:

```tcl
#!/usr/bin/env tclsh
# @description Brief description of what the tool does
# @version 1.0.0
# @param name:type:required Description of parameter
# @param optional_param:string:optional Description (optional params)
```

## Directory Structure

```
tools/
├── bin/          # System tools (available to all users)
│   ├── hello_world.tcl
│   └── list_dir.tcl
├── sbin/         # Administrative tools (privileged mode only)
│   └── system_config.tcl
├── docs/         # Documentation tools
│   └── api_docs.tcl
└── users/        # User-specific tools
    ├── alice/
    │   └── utils/
    │       └── calculator.tcl
    └── bob/
        └── scripts/
            └── backup.tcl
```

## Usage Examples

### 1. Discover Available Tools

```json
{
  "method": "tools/call",
  "params": {
    "name": "bin___discover_tools",
    "arguments": {}
  }
}
```

This will scan the filesystem and index all available tools.

### 2. Execute a Tool

```json
{
  "method": "tools/call",
  "params": {
    "name": "bin___exec_tool",
    "arguments": {
      "tool_path": "/bin/hello_world",
      "params": {
        "name": "Alice"
      }
    }
  }
}
```

### 3. List Tools in a Namespace

```json
{
  "method": "tools/call",
  "params": {
    "name": "bin___tcl_tool_list",
    "arguments": {
      "namespace": "bin"
    }
  }
}
```

## Tool Implementation Example

Here's a complete example of a discoverable tool:

```tcl
#!/usr/bin/env tclsh
# @description Calculate the factorial of a number
# @version 1.0.0
# @param n:integer:required The number to calculate factorial for

# Validate the parameter
if {![info exists n]} {
    error "Missing required parameter: n"
}

# Convert to integer and validate
if {![string is integer -strict $n] || $n < 0} {
    error "Parameter 'n' must be a non-negative integer"
}

# Calculate factorial
proc factorial {num} {
    if {$num <= 1} {
        return 1
    }
    return [expr {$num * [factorial [expr {$num - 1}]]}]
}

# Return the result
puts [factorial $n]
```

## Integration with Existing Tools

The `bin__exec_tool` integrates seamlessly with the existing TCL MCP infrastructure:

1. **Namespace System**: Discovered tools respect the existing namespace hierarchy
2. **Tool Listing**: Discovered tools appear in `bin___tcl_tool_list` output
3. **Persistence**: Discovered tools coexist with dynamically added tools
4. **Security**: Respects privileged mode for sbin tools

## Implementation Details

### Tool Discovery Process

1. **Scanning**: The system recursively scans the tools directory
2. **Metadata Parsing**: Extracts metadata from TCL header comments
3. **Indexing**: Creates an in-memory index for fast lookup
4. **Registration**: Makes tools available through the MCP interface

### Parameter Handling

1. **Validation**: Checks for required parameters before execution
2. **Type Conversion**: Handles JSON to TCL value conversion
3. **Variable Binding**: Sets parameters as TCL variables
4. **Error Handling**: Returns clear error messages for missing parameters

### Security Considerations

1. **Sandboxing**: Tools execute within the Molt interpreter's sandbox
2. **Privileged Mode**: Administrative tools require --privileged flag
3. **Path Validation**: Only allows execution of discovered tools
4. **Input Sanitization**: Parameters are properly escaped

## Performance Characteristics

- **Discovery**: O(n) where n is the number of files in tools directory
- **Execution**: O(1) lookup after discovery + tool execution time
- **Memory**: Minimal - only stores metadata, not tool content
- **Caching**: Tools are indexed in memory after discovery

## Error Handling

The system provides clear error messages for common issues:

- `Tool '/bin/unknown' not found` - Tool doesn't exist
- `Missing required parameter: name` - Required parameter not provided
- `TCL execution error: ...` - Runtime errors from the tool

## Future Enhancements

1. **Hot Reloading**: Automatically detect new tools without restart
2. **Tool Versioning**: Support multiple versions of the same tool
3. **Dependency Management**: Allow tools to declare dependencies
4. **Tool Testing**: Built-in test runner for tool validation
5. **Performance Metrics**: Track execution time and usage statistics