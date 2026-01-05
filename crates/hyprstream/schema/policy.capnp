@0xe7f8a9b0c1d2e3f4;

# Cap'n Proto schema for policy service
#
# The policy service handles authorization checks via Casbin.
# Uses REQ/REP pattern. Runs on multi-threaded runtime.

struct PolicyRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Subject making the request (e.g., "local:alice")
  subject @1 :Text;

  # Domain for policy lookup (e.g., "HfModel", "*" for any)
  domain @2 :Text;

  # Resource being accessed (e.g., "inference:qwen3-small")
  resource @3 :Text;

  # Operation being performed (e.g., "infer", "query", "write")
  operation @4 :Text;
}

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

# Error Information
struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
}
