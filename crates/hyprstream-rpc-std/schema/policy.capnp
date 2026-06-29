@0xf1a2b3c4d5e6f708;

# Cap'n Proto schema for policy service
#
# The policy service handles authorization checks via Casbin.
# Uses REQ/REP pattern. Runs on multi-threaded runtime.

using import "/common.capnp".ErrorInfo;
using import "/annotations.capnp".scope;
using import "/annotations.capnp".mcpDescription;
using import "/annotations.capnp".optional;
using Opt = import "/optional.capnp";

# Unified policy request with union discriminator (follows RegistryRequest pattern)
struct PolicyRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    # Authorization check
    # Intentionally unscoped — authorization check cannot require authorization (circular dependency)
    check @1 :PolicyCheck
      $scopeExempt("the authz check itself cannot require authz — circular dependency");

    # JWT token issuance
    issueToken @2 :IssueToken $scope(manage);

    # List all supported authorization scopes discovered from service schemas
    listScopes @3 :Void $scope(query) $mcpDescription("List all supported authorization scopes discovered from service schemas");

    # Get current policy rules and role assignments
    getPolicy @4 :Void $scope(query) $mcpDescription("Get current policy rules and role assignments");

    # Apply a built-in template (overwrites policy.csv)
    applyTemplate @5 :ApplyTemplate $scope(manage) $mcpDescription("Apply a built-in policy template");

    # Commit draft changes (uncommitted policy.csv edits)
    applyDraft @6 :ApplyDraft $scope(manage) $mcpDescription("Commit draft policy changes");

    # Rollback to a previous policy version
    rollback @7 :RollbackPolicy $scope(manage) $mcpDescription("Rollback to a previous policy version");

    # Get policy commit history
    getHistory @8 :GetHistory $scope(query) $mcpDescription("Get policy commit history");

    # Get diff of uncommitted policy changes vs a ref (default HEAD)
    getDiff @9 :GetDiff $scope(query) $mcpDescription("Get diff of uncommitted policy changes");

    # Check if there are uncommitted policy changes
    getDraftStatus @10 :Void $scope(query) $mcpDescription("Check if there are uncommitted policy changes");

    # Assign a role to a user
    addGrouping @11 :AddGrouping $scope(manage) $mcpDescription("Assign a role to a user");

    # Remove a role from a user
    removeGrouping @12 :RemoveGrouping $scope(manage) $mcpDescription("Remove a role from a user");

    # Set a model branch as public or private
    setBranchVisibility @13 :SetBranchVisibility $scope(manage) $mcpDescription("Set a model branch as public or private");

    # Register an event prefix for publishing
    registerEventPrefix @14 :RegisterEventPrefix $scope(manage) $mcpDescription("Register an event prefix for publishing");

    # Subscribe to an event prefix
    subscribeEventPrefix @15 :SubscribeEventPrefix $scope(manage) $mcpDescription("Subscribe to an event prefix");

    # Get pending subscribers for a prefix
    getPendingSubscribers @16 :GetPendingSubscribers $scope(query) $mcpDescription("Get pending subscribers for a prefix");

    # Deposit wrapped keys for subscribers
    depositWrappedKeys @17 :DepositWrappedKeys $scope(manage) $mcpDescription("Deposit wrapped keys for subscribers");

    # Resolve a service name to its Ed25519 verifying key
    resolveServiceKey @18 :ResolveServiceKey $scope(query) $mcpDescription("Resolve a service name to its Ed25519 verifying key");

    # Register a service's verifying key with the CA
    # Internal CA operation — any caller with a valid CA-signed JWT can register.
    # No authorization scope required; the JWT itself proves CA attestation.
    registerServiceKey @19 :RegisterServiceKey
      $scopeExempt("gated by CA-signed JWT attestation, not by a scope")
      $mcpDescription("Register a service verifying key with the CA");

    # Renew the caller's service JWT. Identity is taken from the signed envelope —
    # no explicit subject field; the CA signs a fresh 30-day JWT for the caller.
    refreshServiceToken @20 :RefreshServiceTokenRequest
      $scope(manage) $mcpDescription("Renew the caller's service JWT; identity from signed envelope");

    # Exchange the caller's envelope WIT for an OAuth at+jwt.
    # Identity and cnf.jwk are read from the verified envelope — no credential submission.
    # Requires 'exchange' permission on 'policy:exchange-wit' in Casbin policy.
    exchangeWit @21 :ExchangeWit
      $scope(manage) $mcpDescription("Exchange the caller's envelope WIT for an OAuth at+jwt; identity from signed envelope");
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

  # Optional TTL in seconds (None = use default)
  ttl @1 :Opt.OptionUint32;

  # RFC 8707 resource indicator for audience binding (empty = no binding)
  audience @2 :Text $optional;

  # Explicit subject for token (empty = use envelope identity).
  # Requires caller to have `manage` permission on `policy:IssueToken` (capnp type name).
  # For service tokens: sub = "service:{name}" (e.g. "service:model").
  # The "pub" claim is derived from the root key by the CA — not caller-provided.
  subject @3 :Text $optional;

  # User's Ed25519 verifying key (base64url, 32 bytes) for user tokens.
  # When present, included in the JWT `pub_key` claim to bind the user's key identity.
  # Ignored for service tokens (pubkey is CA-derived).
  userPubKey @4 :Text $optional;

  # DPoP JWK thumbprint (RFC 7638 SHA-256 base64url) for DPoP user tokens.
  # When present, the issued token carries cnf.jkt instead of cnf.jwk.
  # Takes precedence over userPubKey.
  dpopJkt @5 :Text $optional;
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

    # Commit SHA from addGrouping
    addGroupingResult @12 :Text;

    # Commit SHA from removeGrouping
    removeGroupingResult @13 :Text;

    # Commit SHA from setBranchVisibility
    setBranchVisibilityResult @14 :Text;

    # Event prefix registration result
    registerEventPrefixResult @15 :Void;

    # Event prefix subscription result (returns access info)
    subscribeEventPrefixResult @16 :EventPrefixAccess;

    # Pending subscribers result
    getPendingSubscribersResult @17 :PendingSubscribers;

    # Wrapped keys deposit result
    depositWrappedKeysResult @18 :Void;

    # Service key resolution result
    resolveServiceKeyResult @19 :ServiceKeyResponse;

    # Service key registration result
    registerServiceKeyResult @20 :Void;

    # Fresh JWT from refreshServiceToken
    refreshServiceTokenResult @21 :TokenInfo;

    # at+jwt from exchangeWit
    exchangeWitResult @22 :TokenInfo;
  }
}

# Token information
struct TokenInfo {
  token @0 :Text;      # Signed JWT token (stateless validation)
  expiresAt @1 :Int64;
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

# Parameters for assigning a role to a user
struct AddGrouping {
  user @0 :Text;
  role @1 :Text;
}

# Parameters for removing a role from a user
struct RemoveGrouping {
  user @0 :Text;
  role @1 :Text;
}

# Parameters for setting a model branch's public/private visibility
struct SetBranchVisibility {
  modelName  @0 :Text;
  branchName @1 :Text;
  public     @2 :Bool;
}

# Register an event prefix for publishing
struct RegisterEventPrefix {
  prefix @0 :Text;
  publisherEphemeralPubkey @1 :Data;
  schema @2 :Text;
}

# Subscribe to an event prefix
struct SubscribeEventPrefix {
  prefix @0 :Text;
  subscriberEphemeralPubkey @1 :Data;
}

# Get pending subscribers for a prefix
struct GetPendingSubscribers {
  prefix @0 :Text;
}

# Deposit wrapped keys for pending subscribers
struct DepositWrappedKeys {
  prefix @0 :Text;
  entries @1 :List(WrappedKeyDeposit);
}

# A single wrapped key deposit for a subscriber
struct WrappedKeyDeposit {
  subPubkeyHash @0 :Data;
  wrappedBlob @1 :Data;
}

# Access info returned after subscribing to an event prefix
struct EventPrefixAccess {
  publisherEphemeralPubkey @0 :Data;
  wrappedGroupKey @1 :Data;
  schema @2 :Text;
}

# List of pending subscriber public keys
struct PendingSubscribers {
  pubkeys @0 :List(Data);
}

# Resolve a service name to its Ed25519 verifying key
struct ResolveServiceKey {
  serviceName @0 :Text;
}

# Response containing a service's verifying key and optional CA-signed attestation
struct ServiceKeyResponse {
  # Ed25519 verifying key (32 bytes)
  verifyingKey @0 :Data;
  # CA-signed JWT attesting this key (optional, for verification)
  serviceJwt @1 :Text $optional;
}

# Register a service's verifying key with the CA (PolicyService)
struct RegisterServiceKey {
  # Service name (e.g. "model", "registry")
  serviceName @0 :Text;
  # Ed25519 verifying key (32 bytes)
  verifyingKey @1 :Data;
  # CA-signed JWT proving key ownership (subject must be "service:{serviceName}")
  serviceJwt @2 :Text;
}

# Request to renew the caller's service JWT. Subject is taken from the
# signed envelope context — the caller does not specify it.
struct RefreshServiceTokenRequest {
  # Requested TTL in seconds. Server clamps to [3600, 2592000] (1h – 30d).
  ttlSeconds @0 :Int64 = 2592000;
}

# Exchange the caller's envelope WIT for an OAuth at+jwt (ZMQ-native token bridge).
# The caller's identity and cnf.jwk are read from the verified envelope WIT —
# no credential is submitted in the request body.
struct ExchangeWit {
  # RFC 8707 resource indicator for audience binding.
  # If absent, PolicyService applies the configured default audience.
  audience @0 :Text $optional;

  # Requested scopes (space-delimited).
  # PolicyService intersects with Casbin-permitted scopes for the caller.
  scopes @1 :Text $optional;

  # TTL override in seconds. Server clamps to configured [min, max].
  ttl @2 :Opt.OptionUint32;
}
