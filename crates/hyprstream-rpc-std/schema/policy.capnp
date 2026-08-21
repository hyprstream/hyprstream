@0xfa0212966c7dd7db;

# Cap'n Proto schema for policy service
#
# The policy service handles authorization checks via Casbin.
# Uses REQ/REP pattern. Runs on multi-threaded runtime.

using import "/common.capnp".ErrorInfo;
using import "/annotations.capnp".scope;
using import "/annotations.capnp".scopeExempt;
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

    # Publish a credential revocation to the canonical store owned by this service.
    # Restricted to the OAuth revocation authority (service:oauth) by Casbin —
    # the RFC 7009 endpoint is the only legitimate publisher.
    revokeCredential @22 :RevokeCredential
      $scope(manage);

    # Ask the revocation authority whether a credential has been revoked.
    # Service-identities only (service:*): every enrolled service process
    # checks revocations on its verification path and probes at startup;
    # anonymous/end-user callers have no legitimate read.
    checkCredentialRevocation @23 :CheckCredentialRevocation
      $scope(query);

    # Register a new session with the canonical session registry. Restricted
    # to the OAuth authority (service:oauth), which owns user-session
    # lifecycle at issuance. Session identifiers are never reassigned.
    registerSession @24 :RegisterSession
      $scope(manage);

    # Revoke a session: every credential carrying it is then rejected.
    # Restricted to the OAuth authority (service:oauth).
    revokeSession @25 :RevokeSession
      $scope(manage);

    # Ask the session authority whether a session is active.
    # Service-identities only (service:*); the result Bool is true = ACTIVE
    # and known — revoked, expired, unknown, or malformed all read false
    # (fail-closed; note the opposite polarity from
    # checkCredentialRevocationResult, which reports revoked=true).
    checkSession @26 :CheckSession
      $scope(query);

    # RFC 8693 §4 on-behalf-of delegated mint (v16 §8.1 AsOriginator). The
    # AUTHENTICATED RPC caller is the terminal actor: the authority derives the
    # actor subject and cnf from the verified envelope — never from a request
    # field — verifies the presented originator source credential, and mints a
    # NEW delegated credential (fresh jti, originator `sub`, nested terminal
    # `act`, terminal-actor `cnf`, fail-closed `meet(originator, every actor)`
    # clearance, attenuated scope, conditional `sid`). Reusable (no
    # consume-once). WS-E calls this for a derived AsOriginator dispatch.
    exchangeDelegated @27 :ExchangeDelegated
      $scope(manage);
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

# Issuance profile selected by the authority-owned caller. The default is the
# session-bound interactive profile so an older or malformed caller cannot turn
# a missing profile into an unsessioned user credential.
enum IssueTokenProfile {
  interactiveSession @0;
  rfc8693 @1;
  rfc7523 @2;
  service @3;
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

  # Optional issuer override for profile-specific OAuth tokens.
  # Empty uses PolicyService's configured default issuer.
  issuer @6 :Text $optional;

  # Target tenant/Casbin domain for the issued token.
  # Empty inherits the caller's verified tenant. Cross-tenant issuance requires
  # policy:IssueToken/manage in this target tenant.
  tenant @7 :Text $optional;

  # Require the PolicyService to resolve an authority-owned enrollment
  # clearance for the subject before minting. The resolved clearance is
  # clamped to Classical assurance and stamped into the signed claims.
  requireClearance @8 :Bool;

  # OIDC session ID (`sid` claim) to stamp on the minted token. Set by the
  # OAuth authority for interactive user sessions only; the caller owns the
  # session lifecycle (registration is a separate authority operation —
  # issuance never registers). Empty/absent = no session (standalone service
  # credentials carry no session — v16 §3.3).
  sessionId @9 :Text $optional;

  # Credential issuance profile. Interactive user/OIDC issuance MUST carry
  # `sessionId`; RFC 8693 and RFC 7523 are deliberate non-interactive
  # profiles; service issuance is limited to `service:*` subjects.
  issuanceProfile @10 :IssueTokenProfile;

  # RFC 9068 §2.2.1 `client_id`: the OAuth client the access token is issued to.
  # REQUIRED (non-empty) for the user `at+jwt` profiles (interactive/RFC 8693/
  # RFC 7523); the authority stamps it into the signed `client_id` claim. The
  # service profile mints a `wit+jwt` and carries no `client_id`. Empty/absent =
  # not supplied.
  clientId @11 :Text $optional;
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

    # Revocation publication acknowledged (durable)
    revokeCredentialResult @23 :Void;

    # Revocation check result (true = revoked or unknown — fail-closed)
    checkCredentialRevocationResult @24 :Bool;

    # Session registration acknowledged (durable)
    registerSessionResult @25 :Void;

    # Session revocation acknowledged (durable)
    revokeSessionResult @26 :Void;

    # Session check result (true = ACTIVE and known; false = revoked,
    # expired, unknown, or malformed — fail-closed)
    checkSessionResult @27 :Bool;

    # Minted delegated at+jwt/wit from exchangeDelegated (fresh jti).
    exchangeDelegatedResult @28 :TokenInfo;
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

# Resolve a service name to its published Ed25519 verification-key set
struct ResolveServiceKey {
  serviceName @0 :Text;
}

# One named, CA-attested service verification key.
struct ServiceKeyCandidate {
  # Stable identifier derived from the Ed25519 public key.
  keyId @0 :Text;
  # Ed25519 verifying key (32 bytes).
  verifyingKey @1 :Data;
  # CA-signed JWT attesting this key (optional for bootstrap entries).
  serviceJwt @2 :Text $optional;
  # Unix timestamp at which this candidate stops being accepted (0 = no expiry).
  notAfter @3 :Int64;
}

# Response containing every currently valid, named service verification key.
struct ServiceKeyResponse {
  # Deprecated singleton projection. New producers leave this empty so old
  # consumers fail closed instead of silently selecting an arbitrary key.
  verifyingKey @0 :Data;
  # Deprecated singleton projection; see `keys`.
  serviceJwt @1 :Text $optional;
  # All compatible candidates; order conveys no authority.
  keys @2 :List(ServiceKeyCandidate);
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

# RFC 8693 §4 delegated on-behalf-of mint (v16 §8.1 AsOriginator).
#
# The terminal actor is NEVER a field here: it is the authenticated RPC caller,
# derived from the verified policy envelope (subject + cnf), so E cannot supply
# an arbitrary actor identity, clearance, or key. Only the ORIGINATOR authority
# (the presented source credential) and the requested attenuation subset cross
# the wire. The authority derives originator/session/scope/clearance from the
# verified source credential and actor/cnf/tenant from the verified envelope,
# computes the fail-closed meet, and mints a fresh delegated credential.
struct ExchangeDelegated {
  # The originator's already-issued source credential (at+jwt / wit+jwt) whose
  # authority is being delegated. The authority verifies its signature, expiry,
  # revocation, and issuer/tenant/subject coherence — it is never trusted as
  # plaintext. `sub` becomes the delegated credential's originator.
  sourceCredential @0 :Text;

  # The derived-call OAuth scope subset (space-delimited). v16 §8.1 requires
  # EXPLICIT attenuation at every hop: a scope-bearing source requires an
  # explicit non-empty subset here (equality is allowed only when explicitly
  # requested) — an empty/absent value against a scope-bearing source DENIES (no
  # silent full inheritance). A source with no scope grants none: any requested
  # scope denies. Every requested scope MUST be held by the source; broadening
  # is rejected.
  requestedScopes @1 :Text $optional;

  # The derived-call MAC/UCAN capability subset (`ability@resource` tokens,
  # space-delimited). v16 derived authority is BOTH OAuth scope AND capability;
  # this attenuates the capability axis via the reviewed `Capability` cover
  # relation. Same explicit-attenuation rule as requestedScopes: a cap-bearing
  # source requires an explicit non-empty subset (equality allowed only if
  # explicitly requested); empty/absent against a cap-bearing source DENIES; a
  # source with no `cap` grants none. Any capability not covered by the source
  # is rejected as broadening.
  requestedCapabilities @4 :Text $optional;

  # RFC 8707 resource indicator for the derived call's target. REQUIRED and
  # non-empty: the authority binds it to the reviewed derived-call contract via
  # the fail-closed DelegationEdgeAuthorizer, never an arbitrary string, and
  # never defaulted to the issuer.
  audience @2 :Text $optional;

  # The generated method identifier of the derived call (e.g. "model.Infer").
  # REQUIRED and non-empty: the authority passes it to the
  # DelegationEdgeAuthorizer so the exact reviewed DispatchCallManifest method
  # edge is enforced. It is a request descriptor of the outbound call, not an
  # identity or clearance; an absent/empty value denies (no wildcard).
  targetMethodId @5 :Text $optional;

  # TTL override in seconds. Clamped to the configured [min, max] AND never
  # beyond the source credential's own remaining lifetime, the terminal actor's
  # authority, or a retained session bound.
  ttl @3 :Opt.OptionUint32;
}

# Issuer-scoped credential identifier (iss, jti/cti). JWT jti text and CWT cti
# bytes are disjoint typed namespaces — mirrors the verifier-side
# CredentialValue type; a CWT cti is never stringified into the JWT namespace.
struct CredentialIdRef {
  # The token `iss` claim identifying the credential issuer.
  issuer @0 :Text;
  union {
    # JWT `jti` claim (RFC 7519) — case-sensitive text.
    jwtJti @1 :Text;
    # CWT `cti` claim (RFC 8392) — raw bytes.
    cwtCti @2 :Data;
  }
}

# Publish a credential revocation. The entry may be garbage-collected once
# `expiresAt` passes (natural token expiry rejects it anyway).
struct RevokeCredential {
  credential @0 :CredentialIdRef;
  # The token's `exp` (Unix seconds) — GC hint for the durable store.
  expiresAt @1 :Int64;
}

# Query the revocation authority for a credential's revocation state.
struct CheckCredentialRevocation {
  credential @0 :CredentialIdRef;
}

# Issuer-scoped session identifier (iss, sid/workload_session_id). The two
# variants are disjoint typed namespaces — mirrors SessionIdentifier.
struct SessionKeyRef {
  # The token `iss` claim identifying the session's issuer.
  issuer @0 :Text;
  union {
    # OIDC user-session ID (the registered `sid` claim).
    oidcSid @1 :Text;
    # Workload credential family session ID.
    workloadSessionId @2 :Text;
  }
}

# Register a new session. `expiresAt` bounds the session's lifetime (checked
# against the authority's configured horizon, same as revocation entries).
struct RegisterSession {
  session @0 :SessionKeyRef;
  # Subject identifier (`sub`) the session belongs to.
  subject @1 :Text;
  # Verified tenant/domain the session is bound to.
  tenant @2 :Text;
  # Session expiry (Unix seconds).
  expiresAt @3 :Int64;
  # Clearance epoch at registration.
  clearanceEpoch @4 :UInt64;
}

# Revoke a session: every credential carrying it is then rejected.
struct RevokeSession {
  session @0 :SessionKeyRef;
}

# Query the session authority for a session's active state.
struct CheckSession {
  session @0 :SessionKeyRef;
}
