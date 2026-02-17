@0x8c9b0a1d2e3f4a5b;

# Cap'n Proto schema for MCP (Model Context Protocol) service
#
# The MCP service provides an MCP-compliant interface for AI coding assistants
# to interact with hyprstream via stdio (Claude Code) or HTTP/SSE (web clients).
#
# This schema defines the ZMQ control plane for internal operations:
# - Service status and metrics
# - Tool discovery and introspection
# - Internal tool invocation for other hyprstream services

# Unified MCP request with union discriminator
struct McpRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    # Get service status
    getStatus @1 :Void;

    # List available tools
    listTools @2 :Void;

    # Get service metrics
    getMetrics @3 :Void;

    # Call an MCP tool internally
    callTool @4 :CallTool;
  }
}

# Call an MCP tool internally
struct CallTool {
  # Tool name to call (e.g., "model_load", "model_list")
  toolName @0 :Text;

  # Tool arguments as JSON string
  # (Matches MCP protocol's JSON-RPC arguments format)
  arguments @1 :Text;

  # Optional caller identity for authorization
  # If None, uses the ZMQ request's identity
  callerIdentity @2 :Text;
}

# Unified MCP response
struct McpResponse {
  # Request ID this response corresponds to
  requestId @0 :UInt64;

  # Response payload
  # Convention: response variant name = request variant name + "Result"
  union {
    # Service status information
    getStatusResult @1 :ServiceStatus;

    # List of available tools
    listToolsResult @2 :ToolList;

    # Service metrics
    getMetricsResult @3 :ServiceMetrics;

    # Tool call result
    callToolResult @4 :ToolResult;

    # Error occurred
    error @5 :ErrorInfo;
  }
}

# Service status information
struct ServiceStatus {
  # Whether the service is running
  isRunning @0 :Bool;

  # Number of loaded models
  loadedModelCount @1 :UInt32;

  # Authentication status
  isAuthenticated @2 :Bool;

  # Authenticated user (if any)
  authenticatedUser @3 :Text;

  # Available scopes (if authenticated)
  scopes @4 :List(Text);
}

# List of available tools
struct ToolList {
  # Tool definitions
  tools @0 :List(ToolDefinition);
}

# Tool definition
struct ToolDefinition {
  # Tool name (function name)
  name @0 :Text;

  # Human-readable description
  description @1 :Text;

  # Whether tool is read-only
  isReadOnly @2 :Bool;

  # Whether tool is destructive (modifies state)
  isDestructive @3 :Bool;

  # Required scope to call this tool
  requiredScope @4 :Text;

  # JSON Schema for tool arguments
  argumentSchema @5 :Text;
}

# Service metrics
struct ServiceMetrics {
  # Total number of tool calls
  totalCalls @0 :UInt64;

  # Number of calls per tool
  callsPerTool @1 :List(CallCount);

  # Average call duration in milliseconds
  averageCallDurationMs @2 :Float64;

  # Service uptime in seconds
  uptimeSeconds @3 :Float64;
}

# Call count for a specific tool
struct CallCount {
  # Tool name
  toolName @0 :Text;

  # Number of calls
  count @1 :UInt64;
}

# Tool call result
struct ToolResult {
  # Whether the call was successful
  success @0 :Bool;

  # Result data as JSON string
  # (Matches MCP protocol's CallToolResult format)
  result @1 :Text;

  # Error message (if success is false)
  errorMessage @2 :Text;
}

# Error information
struct ErrorInfo {
  # Human-readable error message
  message @0 :Text;

  # Error code (e.g., "INVALID_REQUEST", "INTERNAL_ERROR")
  code @1 :Text;

  # Additional error details
  details @2 :Text;
}
