@0xf1a2b3c4d5e6f708;

# Cap'n Proto schema for policy service
#
# The policy service handles authorization checks via Casbin.
# Uses REQ/REP pattern. Runs on multi-threaded runtime.

using import "/annotations.capnp".mcpScope;
using import "/annotations.capnp".mcpDescription;

# Unified policy request with union discriminator (follows RegistryRequest pattern)
struct PolicyRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    # Authorization check
    check @1 :PolicyCheck;

    # JWT token issuance
    issueToken @2 :IssueToken $mcpScope(manage);

    # List all supported authorization scopes discovered from service schemas
    listScopes @3 :Void $mcpScope(query) $mcpDescription("List all supported authorization scopes discovered from service schemas");
  }
}

# Authorization check parameters
struct PolicyCheck {
  # Subject making the request (e.g., "local:alice")
  subject @0 :Text;

  # Domain for policy lookup (e.g., "HfModel", "*" for any)
  domain @1 :Text;

  # Resource being accessed (e.g., "inference:qwen3-small")
  resource @2 :Text;

  # Operation being performed (e.g., "infer", "query", "write")
  operation @3 :Text;
}

# JWT token issuance parameters
struct IssueToken {
  # Structured scopes: action:resource:identifier
  requestedScopes @0 :List(Text);

  # Optional TTL in seconds (0 = use default)
  ttl @1 :UInt32;

  # RFC 8707 resource indicator for audience binding (empty = no binding)
  audience @2 :Text;

  # Explicit subject for token (empty = use envelope identity).
  # Requires caller to have `manage` permission on `policy:issue-token`.
  subject @3 :Text;
}

# Unified policy response (covers both check and token issuance)
struct PolicyResponse {
  # Request ID this response corresponds to
  requestId @0 :UInt64;

  # Response payload
  # Convention: response variant = request name + "Result"
  # This enables codegen to auto-unwrap typed returns.
  union {
    # Authorization result (for check)
    checkResult @1 :Bool;

    # Error occurred
    error @2 :ErrorInfo;

    # Token issuance result (for issueToken)
    issueTokenResult @3 :TokenInfo;

    # Supported scopes list (for listScopes)
    listScopesResult @4 :ScopeList;
  }
}

# Token information
struct TokenInfo {
  token @0 :Text;      # Signed JWT token (stateless validation)
  expiresAt @1 :Int64;
}

# Error Information
struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
  details @2 :Text;
}

# List of supported authorization scopes
struct ScopeList {
  scopes @0 :List(Text);
}
