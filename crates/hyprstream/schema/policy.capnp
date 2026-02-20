@0xf1a2b3c4d5e6f708;

# Cap'n Proto schema for policy service
#
# The policy service handles authorization checks via Casbin.
# Uses REQ/REP pattern. Runs on multi-threaded runtime.

using import "/annotations.capnp".mcpScope;
using import "/annotations.capnp".mcpDescription;
using import "/annotations.capnp".optional;

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

    # Get current policy rules and role assignments
    getPolicy @4 :Void $mcpScope(query) $mcpDescription("Get current policy rules and role assignments");

    # Apply a built-in template (overwrites policy.csv)
    applyTemplate @5 :ApplyTemplate $mcpScope(manage) $mcpDescription("Apply a built-in policy template");

    # Commit draft changes (uncommitted policy.csv edits)
    applyDraft @6 :ApplyDraft $mcpScope(manage) $mcpDescription("Commit draft policy changes");

    # Rollback to a previous policy version
    rollback @7 :RollbackPolicy $mcpScope(manage) $mcpDescription("Rollback to a previous policy version");

    # Get policy commit history
    getHistory @8 :GetHistory $mcpScope(query) $mcpDescription("Get policy commit history");

    # Get diff of uncommitted policy changes vs a ref (default HEAD)
    getDiff @9 :GetDiff $mcpScope(query) $mcpDescription("Get diff of uncommitted policy changes");

    # Check if there are uncommitted policy changes
    getDraftStatus @10 :Void $mcpScope(query) $mcpDescription("Check if there are uncommitted policy changes");
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
  requestedScopes @0 :List(Text) $optional;

  # Optional TTL in seconds (0 = use default)
  ttl @1 :UInt32 $optional;

  # RFC 8707 resource indicator for audience binding (empty = no binding)
  audience @2 :Text $optional;

  # Explicit subject for token (empty = use envelope identity).
  # Requires caller to have `manage` permission on `policy:issue-token`.
  subject @3 :Text $optional;
}

# Apply a built-in policy template
struct ApplyTemplate {
  # Template name (e.g., "local", "public-inference", "public-read")
  name @0 :Text;
}

# Commit draft changes to running policy
struct ApplyDraft {
  # Optional commit message (auto-generated if empty)
  message @0 :Text $optional;
}

# Rollback to a previous policy version
struct RollbackPolicy {
  # Git ref to rollback to (e.g., "HEAD~1")
  gitRef @0 :Text;
}

# Get policy commit history
struct GetHistory {
  # Maximum number of entries to return (0 = default 10)
  count @0 :UInt32;
}

# Get diff of uncommitted policy changes
struct GetDiff {
  # Git ref to diff against (empty = HEAD)
  gitRef @0 :Text $optional;
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

    # Current policy info (for getPolicy)
    getPolicyResult @5 :PolicyInfo;

    # Commit message from apply/rollback operations
    applyTemplateResult @6 :Text;

    # Commit message from apply draft
    applyDraftResult @7 :Text;

    # Commit message from rollback
    rollbackResult @8 :Text;

    # Policy history (for getHistory)
    getHistoryResult @9 :PolicyHistory;

    # Diff text (for getDiff)
    getDiffResult @10 :Text;

    # Draft status (for getDraftStatus)
    getDraftStatusResult @11 :DraftStatus;
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

# Current policy configuration
struct PolicyInfo {
  rules @0 :List(PolicyRule);
  groupings @1 :List(Grouping);
}

# A single policy rule (p = sub, dom, obj, act, eft)
struct PolicyRule {
  subject @0 :Text;
  domain @1 :Text;
  resource @2 :Text;
  action @3 :Text;
  effect @4 :Text;
}

# A role assignment (g = user, role)
struct Grouping {
  user @0 :Text;
  role @1 :Text;
}

# Policy commit history entry
struct PolicyHistoryEntry {
  hash @0 :Text;
  message @1 :Text;
  date @2 :Text;
}

# Whether there are uncommitted policy changes
struct DraftStatus {
  hasChanges @0 :Bool;
  summary @1 :Text;    # e.g. "2 files changed"
}

# Policy commit history
struct PolicyHistory {
  entries @0 :List(PolicyHistoryEntry);
}
