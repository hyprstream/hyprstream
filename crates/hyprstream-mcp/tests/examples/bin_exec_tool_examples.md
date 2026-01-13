# bin__exec_tool Examples

This document provides comprehensive examples of using the `bin__exec_tool` functionality in the TCL MCP server.

## Overview

The `bin__exec_tool` is a system tool that allows execution of custom user-defined tools that have been added to the TCL MCP server. It provides a standardized way to invoke tools with parameters and handle their results.

## Basic Usage

### 1. Adding a Simple Tool

First, add a custom tool using `tcl_tool_add`:

```json
{
  "method": "tools/call",
  "params": {
    "name": "mcp__tcl__sbin___tcl_tool_add",
    "arguments": {
      "user": "alice",
      "package": "utils",
      "name": "greet",
      "description": "A friendly greeting tool",
      "script": "return \"Hello, $name! Welcome to TCL MCP.\"",
      "parameters": [
        {
          "name": "name",
          "description": "Name of the person to greet",
          "required": true,
          "type_name": "string"
        }
      ]
    }
  }
}
```

### 2. Executing the Tool

Now execute it using `bin__exec_tool`:

```json
{
  "method": "tools/call",
  "params": {
    "name": "mcp__tcl__bin___exec_tool",
    "arguments": {
      "tool_path": "/alice/utils/greet:latest",
      "arguments": {
        "name": "Bob"
      }
    }
  }
}
```

Response:
```json
{
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Hello, Bob! Welcome to TCL MCP."
      }
    ]
  }
}
```

## Advanced Examples

### String Manipulation Tool

```tcl
# Tool definition
{
  "user": "alice",
  "package": "strings",
  "name": "manipulate",
  "description": "Advanced string manipulation",
  "script": "
    set result \"\"
    
    # Reverse the string
    if {$reverse} {
        set text [string reverse $text]
    }
    
    # Change case
    switch $case {
        \"upper\" { set text [string toupper $text] }
        \"lower\" { set text [string tolower $text] }
        \"title\" { set text [string totitle $text] }
    }
    
    # Repeat if specified
    if {[info exists repeat] && $repeat > 1} {
        set text [string repeat $text $repeat]
    }
    
    return $text
  ",
  "parameters": [
    {
      "name": "text",
      "description": "Text to manipulate",
      "required": true,
      "type_name": "string"
    },
    {
      "name": "reverse",
      "description": "Whether to reverse the string",
      "required": true,
      "type_name": "boolean"
    },
    {
      "name": "case",
      "description": "Case transformation (upper/lower/title/none)",
      "required": true,
      "type_name": "string"
    },
    {
      "name": "repeat",
      "description": "Number of times to repeat",
      "required": false,
      "type_name": "integer"
    }
  ]
}
```

Execution:
```json
{
  "tool_path": "/alice/strings/manipulate:latest",
  "arguments": {
    "text": "Hello World",
    "reverse": true,
    "case": "upper",
    "repeat": 2
  }
}
```

Result: `"DLROW OLLEHDLROW OLLEH"`

### Mathematical Calculator

```tcl
# Tool definition
{
  "user": "bob",
  "package": "math",
  "name": "calculator",
  "description": "Advanced mathematical operations",
  "script": "
    # Define mathematical functions
    proc factorial {n} {
        if {$n <= 1} { return 1 }
        return [expr {$n * [factorial [expr {$n - 1}]]}]
    }
    
    proc fibonacci {n} {
        if {$n <= 1} { return $n }
        return [expr {[fibonacci [expr {$n - 1}]] + [fibonacci [expr {$n - 2}]]}]
    }
    
    proc gcd {a b} {
        while {$b != 0} {
            set temp $b
            set b [expr {$a % $b}]
            set a $temp
        }
        return $a
    }
    
    # Execute requested operation
    switch $operation {
        \"factorial\" { return [factorial $a] }
        \"fibonacci\" { return [fibonacci $a] }
        \"gcd\" { return [gcd $a $b] }
        \"power\" { return [expr {pow($a, $b)}] }
        \"sqrt\" { return [expr {sqrt($a)}] }
        default { error \"Unknown operation: $operation\" }
    }
  ",
  "parameters": [
    {
      "name": "operation",
      "description": "Mathematical operation to perform",
      "required": true,
      "type_name": "string"
    },
    {
      "name": "a",
      "description": "First number",
      "required": true,
      "type_name": "number"
    },
    {
      "name": "b",
      "description": "Second number (for binary operations)",
      "required": false,
      "type_name": "number"
    }
  ]
}
```

### Data Processing Tool

```tcl
# Tool definition for CSV processing
{
  "user": "data",
  "package": "processing",
  "name": "csv_stats",
  "description": "Calculate statistics from CSV data",
  "script": "
    # Parse CSV data
    set lines [split $csv_data \\n]
    set headers [split [lindex $lines 0] ,]
    set data_rows [lrange $lines 1 end]
    
    # Find column index
    set col_idx [lsearch $headers $column]
    if {$col_idx == -1} {
        error \"Column '$column' not found\"
    }
    
    # Extract column values
    set values {}
    foreach row $data_rows {
        if {$row ne \"\"} {
            set fields [split $row ,]
            set value [lindex $fields $col_idx]
            if {[string is double $value]} {
                lappend values $value
            }
        }
    }
    
    # Calculate statistics
    set count [llength $values]
    if {$count == 0} {
        return \"No numeric values found\"
    }
    
    set sum 0
    set min [lindex $values 0]
    set max [lindex $values 0]
    
    foreach val $values {
        set sum [expr {$sum + $val}]
        if {$val < $min} { set min $val }
        if {$val > $max} { set max $val }
    }
    
    set avg [expr {$sum / double($count)}]
    
    # Format result
    set result \"\"
    append result \"Count: $count\\n\"
    append result \"Sum: $sum\\n\"
    append result \"Average: [format %.2f $avg]\\n\"
    append result \"Min: $min\\n\"
    append result \"Max: $max\"
    
    return $result
  ",
  "parameters": [
    {
      "name": "csv_data",
      "description": "CSV data to process",
      "required": true,
      "type_name": "string"
    },
    {
      "name": "column",
      "description": "Column name to analyze",
      "required": true,
      "type_name": "string"
    }
  ]
}
```

### File Template Generator

```tcl
# Tool for generating file contents from templates
{
  "user": "dev",
  "package": "templates",
  "name": "generate",
  "description": "Generate file content from templates",
  "script": "
    # Define templates
    set templates [dict create]
    
    dict set templates \"python\" {
#!/usr/bin/env python3
\"\"\"
$description
\"\"\"

import sys

def main():
    \"\"\"Main function.\"\"\"
    # TODO: Implement $name
    pass

if __name__ == \"__main__\":
    main()
}
    
    dict set templates \"javascript\" {
/**
 * $description
 */

'use strict';

/**
 * $name implementation
 */
function $name() {
    // TODO: Implement
}

module.exports = { $name };
}
    
    dict set templates \"dockerfile\" {
FROM $base_image

LABEL maintainer=\"$author\"
LABEL description=\"$description\"

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \\
    && rm -rf /var/lib/apt/lists/*

# Copy application
COPY . .

# Set entrypoint
ENTRYPOINT [\"/app/entrypoint.sh\"]
}
    
    # Get template
    if {![dict exists $templates $template_type]} {
        error \"Unknown template type: $template_type\"
    }
    
    set template [dict get $templates $template_type]
    
    # Substitute variables
    set result $template
    foreach {var value} [array get ::] {
        if {[string match \"*$var*\" $result]} {
            regsub -all \"\\$$var\" $result $value result
        }
    }
    
    return $result
  ",
  "parameters": [
    {
      "name": "template_type",
      "description": "Type of template (python/javascript/dockerfile)",
      "required": true,
      "type_name": "string"
    },
    {
      "name": "name",
      "description": "Name of the component",
      "required": true,
      "type_name": "string"
    },
    {
      "name": "description",
      "description": "Description of the component",
      "required": true,
      "type_name": "string"
    },
    {
      "name": "author",
      "description": "Author name (for dockerfile)",
      "required": false,
      "type_name": "string"
    },
    {
      "name": "base_image",
      "description": "Base image (for dockerfile)",
      "required": false,
      "type_name": "string"
    }
  ]
}
```

## Error Handling Examples

### 1. Missing Required Parameter

Request:
```json
{
  "tool_path": "/alice/utils/greet:latest",
  "arguments": {}  // Missing required 'name' parameter
}
```

Error Response:
```json
{
  "error": {
    "code": -32602,
    "message": "Missing required parameter: name"
  }
}
```

### 2. Tool Not Found

Request:
```json
{
  "tool_path": "/nonexistent/tool:1.0",
  "arguments": {}
}
```

Error Response:
```json
{
  "error": {
    "code": -32602,
    "message": "Tool '/nonexistent/tool:1.0' not found"
  }
}
```

### 3. Script Execution Error

When a tool's TCL script encounters an error:

```json
{
  "error": {
    "code": -32603,
    "message": "TCL execution error: invalid command name \"undefined_proc\""
  }
}
```

## Best Practices

1. **Parameter Validation**: Always define required parameters with appropriate types
2. **Error Handling**: Include error checking in your TCL scripts
3. **Documentation**: Provide clear descriptions for tools and parameters
4. **Versioning**: Use semantic versioning for your tools
5. **Namespacing**: Organize tools into logical packages
6. **Testing**: Test tools thoroughly before deployment

## Security Considerations

1. **Privileged Mode**: Some operations require the server to run in privileged mode
2. **Input Sanitization**: The server automatically escapes special characters in string parameters
3. **Resource Limits**: Consider implementing timeouts for long-running operations
4. **Access Control**: Use namespaces to organize and control access to tools

## Performance Tips

1. **Caching**: Tools are cached in memory for fast execution
2. **Async Operations**: The server handles concurrent requests efficiently
3. **Minimal Dependencies**: Keep TCL scripts lightweight
4. **Batch Operations**: Design tools to handle multiple items when appropriate