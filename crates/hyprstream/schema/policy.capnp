@0xe7f8a9b0c1d2e3f4;

# Cap'n Proto schema for policy service
#
# The policy service handles authorization checks via Casbin.
# Uses REQ/REP pattern. Runs on multi-threaded runtime.

# Unified policy request with union discriminator (follows RegistryRequest pattern)
struct PolicyRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    # Authorization check
    check @1 :PolicyCheck;

    # JWT token issuance
    issueToken @2 :IssueToken;
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
}

# Response for authorization checks
struct PolicyResponse {
  # Request ID this response corresponds to
  requestId @0 :UInt64;

  # Response payload
  union {
    # Authorization result
    allowed @1 :Bool;

    # Error occurred
    error @2 :ErrorInfo;
  }
}

# Response for token issuance
struct IssueTokenResponse {
  requestId @0 :UInt64;
  union {
    success @1 :TokenInfo;
    error @2 :ErrorInfo;
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
