# MCP Integration Guide for TCL Runtime Abstraction

## Overview

This guide demonstrates how to use MCP (Model Context Protocol) properties and example prompts with the TCL runtime abstraction system. The server provides intelligent capability reporting that helps LLMs generate appropriate TCL code based on the active runtime.

## MCP Tool Properties Structure

### Basic Tool Definition with Runtime Context

```json
{
  "name": "bin___tcl_execute",
  "description": "Execute TCL scripts with runtime-aware capabilities",
  "inputSchema": {
    "type": "object",
    "properties": {
      "script": {
        "type": "string",
        "description": "TCL script to execute"
      }
    },
    "required": ["script"]
  },
  "metadata": {
    "runtime_context": {
      "active_runtime": "Molt",
      "safety_level": "safe",
      "available_features": ["core", "string", "list", "math"],
      "limitations": ["no_file_io", "no_system_commands"]
    },
    "examples": [
      {
        "name": "Basic arithmetic",
        "prompt": "Calculate the sum of 15 and 27",
        "code": "expr {15 + 27}"
      },
      {
        "name": "String manipulation",
        "prompt": "Convert 'hello world' to uppercase",
        "code": "string toupper \"hello world\""
      },
      {
        "name": "List operations",
        "prompt": "Create a list of fruits and get its length",
        "code": "set fruits {apple banana orange}; llength $fruits"
      }
    ]
  }
}
```

### Runtime-Specific Example Prompts

#### Molt Runtime Examples (Safe Subset)

```json
{
  "metadata": {
    "runtime_info": {
      "name": "Molt",
      "version": "0.3.1",
      "safety": "sandboxed"
    },
    "example_prompts": [
      {
        "category": "arithmetic",
        "description": "Safe mathematical operations",
        "examples": [
          {
            "prompt": "Calculate compound interest: principal=1000, rate=0.05, time=3",
            "tcl_code": "set principal 1000\nset rate 0.05\nset time 3\nexpr {$principal * pow(1 + $rate, $time)}",
            "expected_output": "1157.625",
            "explanation": "Uses expr for safe mathematical calculations"
          },
          {
            "prompt": "Find the average of numbers: 23, 45, 67, 12, 89",
            "tcl_code": "set numbers {23 45 67 12 89}\nset sum 0\nforeach num $numbers { set sum [expr {$sum + $num}] }\nexpr {$sum / [llength $numbers]}",
            "expected_output": "47.2",
            "explanation": "Demonstrates loops and list operations"
          }
        ]
      },
      {
        "category": "string_processing",
        "description": "Text manipulation and formatting",
        "examples": [
          {
            "prompt": "Parse a CSV line: 'John,Doe,25,Engineer' and extract the profession",
            "tcl_code": "set csv \"John,Doe,25,Engineer\"\nset fields [split $csv \",\"]\nlindex $fields 3",
            "expected_output": "Engineer",
            "explanation": "Safe string splitting and list indexing"
          },
          {
            "prompt": "Create a formatted name from first='Alice' and last='Johnson'",
            "tcl_code": "set first \"Alice\"\nset last \"Johnson\"\nformat \"%s, %s\" $last $first",
            "expected_output": "Johnson, Alice",
            "explanation": "String formatting without file I/O"
          }
        ]
      },
      {
        "category": "control_flow",
        "description": "Conditional logic and loops",
        "examples": [
          {
            "prompt": "Implement FizzBuzz for numbers 1-15",
            "tcl_code": "for {set i 1} {$i <= 15} {incr i} {\n  if {$i % 15 == 0} {\n    puts \"FizzBuzz\"\n  } elseif {$i % 3 == 0} {\n    puts \"Fizz\"\n  } elseif {$i % 5 == 0} {\n    puts \"Buzz\"\n  } else {\n    puts $i\n  }\n}",
            "expected_output": "1\\n2\\nFizz\\n4\\nBuzz\\n...",
            "explanation": "Safe control flow without system access"
          }
        ]
      }
    ],
    "limitations_examples": [
      {
        "category": "file_operations",
        "forbidden_prompt": "Read contents of /etc/passwd",
        "why_forbidden": "Molt runtime doesn't support file I/O for security",
        "safe_alternative": "Use embedded data or variables instead",
        "alternative_code": "set data \"user1:x:1000:1000:User One:/home/user1:/bin/bash\""
      },
      {
        "category": "system_commands",
        "forbidden_prompt": "Execute 'ls -la' system command",
        "why_forbidden": "Molt runtime blocks system command execution",
        "safe_alternative": "Use TCL's built-in commands for data processing",
        "alternative_code": "set files {file1.txt file2.log file3.conf}\nforeach file $files { puts \"Processing $file\" }"
      }
    ]
  }
}
```

#### TCL Official Runtime Examples (Full Features)

```json
{
  "metadata": {
    "runtime_info": {
      "name": "TCL (Official)",
      "version": "8.6+",
      "safety": "full_access"
    },
    "example_prompts": [
      {
        "category": "file_operations",
        "description": "File I/O and filesystem operations",
        "examples": [
          {
            "prompt": "Read a configuration file and parse key-value pairs",
            "tcl_code": "set config_file \"app.conf\"\nif {[file exists $config_file]} {\n  set fp [open $config_file r]\n  set content [read $fp]\n  close $fp\n  array set config {}\n  foreach line [split $content \"\\n\"] {\n    if {[regexp {^(\\w+)=(.+)$} $line -> key value]} {\n      set config($key) $value\n    }\n  }\n  parray config\n}",
            "explanation": "Full file I/O with error checking and regex parsing"
          },
          {
            "prompt": "Create a backup directory with timestamp",
            "tcl_code": "set timestamp [clock format [clock seconds] -format \"%Y%m%d_%H%M%S\"]\nset backup_dir \"backup_$timestamp\"\nfile mkdir $backup_dir\nputs \"Created backup directory: $backup_dir\"",
            "explanation": "Directory creation with timestamp formatting"
          }
        ]
      },
      {
        "category": "system_integration",
        "description": "System command execution and process management",
        "examples": [
          {
            "prompt": "Get disk usage for current directory",
            "tcl_code": "set result [exec du -sh .]\nputs \"Current directory size: $result\"",
            "explanation": "System command execution with output capture"
          },
          {
            "prompt": "Monitor system processes containing 'nginx'",
            "tcl_code": "if {[catch {exec ps aux | grep nginx | grep -v grep} processes]} {\n  puts \"No nginx processes found\"\n} else {\n  puts \"Nginx processes:\"\n  puts $processes\n}",
            "explanation": "Process monitoring with error handling"
          }
        ]
      },
      {
        "category": "network_operations",
        "description": "Socket programming and network communication",
        "examples": [
          {
            "prompt": "Create a simple HTTP client to fetch a webpage",
            "tcl_code": "package require http\nset token [http::geturl \"http://httpbin.org/json\"]\nset data [http::data $token]\nhttp::cleanup $token\nputs $data",
            "explanation": "HTTP client using TCL's http package"
          },
          {
            "prompt": "Create a TCP server that echoes messages",
            "tcl_code": "proc handle_client {sock addr port} {\n  puts \"Client connected: $addr:$port\"\n  fileevent $sock readable [list echo_data $sock]\n}\n\nproc echo_data {sock} {\n  if {[eof $sock]} {\n    close $sock\n    return\n  }\n  set data [read $sock]\n  puts -nonewline $sock $data\n  flush $sock\n}\n\nsocket -server handle_client 8080\nputs \"Echo server listening on port 8080\"\nvwait forever",
            "explanation": "TCP server with event-driven I/O"
          }
        ]
      }
    ],
    "privileged_examples": [
      {
        "category": "tool_management",
        "description": "Dynamic tool creation and management",
        "examples": [
          {
            "prompt": "Create a custom tool for temperature conversion",
            "tcl_code": "# This would use the sbin___tcl_tool_add MCP tool\n# Tool script:\nif {$unit eq \"C\"} {\n  return [expr {$temp * 9.0/5.0 + 32}]\n} else {\n  return [expr {($temp - 32) * 5.0/9.0}]\n}",
            "explanation": "Custom tool creation for privileged mode"
          }
        ]
      }
    ]
  }
}
```

## MCP Server Capability Reporting

### Runtime Information Tool

```json
{
  "name": "tcl_runtime_info",
  "description": "Get detailed information about the active TCL runtime and its capabilities",
  "inputSchema": {
    "type": "object",
    "properties": {
      "include_examples": {
        "type": "boolean",
        "default": true,
        "description": "Include example prompts and code snippets"
      },
      "category_filter": {
        "type": "string",
        "enum": ["all", "safe", "file_io", "system", "network"],
        "default": "all",
        "description": "Filter examples by capability category"
      }
    }
  },
  "metadata": {
    "example_prompts": [
      {
        "prompt": "What TCL commands are available for string processing?",
        "usage": {
          "tool": "tcl_runtime_info",
          "arguments": {
            "include_examples": true,
            "category_filter": "safe"
          }
        },
        "expected_response": {
          "runtime_name": "Molt",
          "available_commands": {
            "string": ["string", "format", "scan", "regexp", "regsub"],
            "examples": [
              {
                "command": "string",
                "usage": "string toupper \"hello\"",
                "result": "HELLO"
              }
            ]
          }
        }
      },
      {
        "prompt": "Can I use file operations with the current runtime?",
        "usage": {
          "tool": "tcl_runtime_info",
          "arguments": {
            "category_filter": "file_io"
          }
        },
        "expected_response": {
          "runtime_name": "Molt",
          "file_io_available": false,
          "limitations": ["No file I/O operations", "Use embedded data instead"],
          "alternatives": [
            {
              "instead_of": "set fp [open file.txt r]",
              "use": "set data \"embedded content\""
            }
          ]
        }
      }
    ]
  }
}
```

### Command Availability Tool

```json
{
  "name": "tcl_command_check",
  "description": "Check if specific TCL commands are available in the current runtime",
  "inputSchema": {
    "type": "object",
    "properties": {
      "commands": {
        "type": "array",
        "items": {"type": "string"},
        "description": "List of TCL commands to check"
      }
    },
    "required": ["commands"]
  },
  "metadata": {
    "example_prompts": [
      {
        "prompt": "Can I use 'exec', 'open', and 'socket' commands?",
        "usage": {
          "tool": "tcl_command_check",
          "arguments": {
            "commands": ["exec", "open", "socket", "string", "expr"]
          }
        },
        "expected_response": {
          "runtime": "Molt",
          "results": {
            "exec": {"available": false, "reason": "System commands disabled in Molt"},
            "open": {"available": false, "reason": "File I/O not supported in Molt"},
            "socket": {"available": false, "reason": "Network operations not supported"},
            "string": {"available": true, "category": "string_manipulation"},
            "expr": {"available": true, "category": "arithmetic"}
          },
          "safe_alternatives": {
            "exec": "Use TCL control flow instead of system commands",
            "open": "Use embedded data or variables",
            "socket": "Not available in safe mode"
          }
        }
      }
    ]
  }
}
```

## LLM Integration Examples

### Smart Code Generation Based on Runtime

```python
# Example: LLM adapts code based on runtime capabilities

async def generate_tcl_code(prompt, mcp_client):
    # 1. Check runtime capabilities
    capabilities = await mcp_client.call_tool("tcl_runtime_info", {})
    
    # 2. Adapt code generation based on capabilities
    if prompt == "read a file and process its contents":
        if capabilities["file_io_available"]:
            # Generate full TCL with file I/O
            code = """
set fp [open "data.txt" r]
set content [read $fp]
close $fp
foreach line [split $content "\\n"] {
    # Process each line
    puts "Processing: $line"
}
"""
        else:
            # Generate safe alternative for Molt
            code = """
# File I/O not available in Molt runtime
# Using embedded data instead
set content "line1\\nline2\\nline3"
foreach line [split $content "\\n"] {
    # Process each line
    puts "Processing: $line"
}
"""
    
    return {
        "code": code,
        "runtime": capabilities["runtime_name"],
        "explanation": f"Generated for {capabilities['runtime_name']} runtime"
    }
```

### Context-Aware Help System

```json
{
  "name": "tcl_help",
  "description": "Get context-aware help for TCL commands based on active runtime",
  "inputSchema": {
    "type": "object", 
    "properties": {
      "command": {
        "type": "string",
        "description": "TCL command to get help for"
      },
      "include_examples": {
        "type": "boolean",
        "default": true
      }
    },
    "required": ["command"]
  },
  "metadata": {
    "example_prompts": [
      {
        "prompt": "How do I use the 'string' command?",
        "usage": {
          "tool": "tcl_help",
          "arguments": {"command": "string", "include_examples": true}
        },
        "response_for_molt": {
          "command": "string",
          "available": true,
          "subcommands": ["length", "index", "range", "toupper", "tolower", "trim"],
          "examples": [
            {"usage": "string length \"hello\"", "result": "5"},
            {"usage": "string toupper \"world\"", "result": "WORLD"},
            {"usage": "string range \"hello\" 1 3", "result": "ell"}
          ],
          "runtime_notes": "All string operations are safe in Molt runtime"
        },
        "response_for_tcl": {
          "command": "string",
          "available": true,
          "subcommands": ["length", "index", "range", "toupper", "tolower", "trim", "map", "match", "compare"],
          "examples": [
            {"usage": "string map {old new} \"old text\"", "result": "new text"},
            {"usage": "string match \"*pattern*\" \"test pattern here\"", "result": "1"}
          ],
          "runtime_notes": "Full string command available in official TCL"
        }
      },
      {
        "prompt": "Can I use 'exec' to run system commands?",
        "usage": {
          "tool": "tcl_help", 
          "arguments": {"command": "exec"}
        },
        "response_for_molt": {
          "command": "exec",
          "available": false,
          "reason": "System command execution disabled for security",
          "alternatives": [
            "Use TCL's built-in commands for data processing",
            "Implement logic using control flow and variables"
          ],
          "safe_example": "# Instead of: exec ls\nset files {file1.txt file2.log}\nforeach file $files { puts $file }"
        },
        "response_for_tcl": {
          "command": "exec",
          "available": true,
          "privileged_only": true,
          "examples": [
            {"usage": "exec ls -la", "note": "List directory contents"},
            {"usage": "exec echo $env(USER)", "note": "Access environment variables"}
          ],
          "security_warning": "exec provides full system access - use carefully"
        }
      }
    ]
  }
}
```

## Best Practices for MCP Integration

### 1. Runtime-Aware Tool Descriptions

```json
{
  "name": "bin___tcl_execute",
  "description": "Execute TCL scripts (Runtime: {RUNTIME_NAME}, Safety: {SAFETY_LEVEL})",
  "metadata": {
    "runtime_template": true,
    "dynamic_description": "Execute TCL scripts (Runtime: {{runtime.name}}, Safety: {{runtime.safety_level}})"
  }
}
```

### 2. Capability-Based Example Selection

```python
def get_relevant_examples(runtime_capabilities, user_intent):
    """Select examples based on runtime and user needs"""
    examples = []
    
    if "string" in user_intent:
        examples.extend(runtime_capabilities["examples"]["string_processing"])
    
    if "file" in user_intent:
        if runtime_capabilities["file_io_available"]:
            examples.extend(runtime_capabilities["examples"]["file_operations"])
        else:
            examples.extend(runtime_capabilities["examples"]["file_alternatives"])
    
    return examples
```

### 3. Progressive Enhancement Pattern

```json
{
  "basic_functionality": {
    "available_in": ["molt", "tcl"],
    "examples": ["arithmetic", "string_ops", "lists"]
  },
  "enhanced_functionality": {
    "available_in": ["tcl"],
    "requires_privilege": false,
    "examples": ["advanced_regex", "clock_formatting"]
  },
  "privileged_functionality": {
    "available_in": ["tcl"],
    "requires_privilege": true,
    "examples": ["file_operations", "system_commands", "network"]
  }
}
```

This comprehensive MCP integration ensures that LLMs can generate appropriate, safe, and effective TCL code while understanding the capabilities and limitations of the active runtime.