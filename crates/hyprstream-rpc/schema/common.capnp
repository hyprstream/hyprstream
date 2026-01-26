@0xb1c2d3e4f5a6b7c8;

# Common Cap'n Proto types for hyprstream RPC
#
# This schema defines identity and envelope types used across all services.
# Every request is wrapped in a SignedEnvelope that carries identity context.
#
# Security Model:
# - Transport layer: CURVE encryption (TCP only)
# - Application layer: Ed25519 signatures (survives message forwarding)
# - Streaming: Chained HMAC-SHA256 for authentication + ordering

# Identity context for request authorization
struct RequestIdentity {
  union {
    local @0 :LocalIdentity;       # Local process user (OS username)
    apiToken @1 :ApiTokenIdentity; # API token authenticated
    peer @2 :PeerIdentity;         # Remote peer (CURVE authenticated)
    anonymous @3 :Void;            # No authentication
  }
}

# Local process identity (trusted, same machine)
struct LocalIdentity {
  user @0 :Text;  # OS username (e.g., "alice")
}

# API token identity
struct ApiTokenIdentity {
  user @0 :Text;       # User name associated with token
  tokenName @1 :Text;  # Token identifier (e.g., "ci-pipeline")
}

# Remote peer identity (CURVE authenticated)
struct PeerIdentity {
  name @0 :Text;       # Registered peer name (e.g., "gpu-server-1")
  curveKey @1 :Data;   # CURVE public key (32 bytes)
}

# Unsigned request data - this is what gets signed
#
# Contains all request metadata and payload. The entire serialized RequestEnvelope
# is signed by SignedEnvelope.signature for clear signing scope.
struct RequestEnvelope {
  requestId @0 :UInt64;            # Unique request ID for correlation
  identity @1 :RequestIdentity;    # Who is making the request (service identity)
  payload @2 :Data;                # Serialized inner request (RegistryRequest, etc.)
  ephemeralPubkey @3 :Data;        # Ristretto255/P-256 public key for stream HMAC (optional, 32 bytes)
  nonce @4 :Data;                  # 16 random bytes for replay protection
  timestamp @5 :Int64;             # Unix millis, for expiration check
  claims @6 :Claims;               # User authorization claims (protected by envelope signature)
}

# Signed wrapper - signature covers serialized RequestEnvelope bytes
#
# All RPC requests should be wrapped in this envelope.
# The nested structure makes clear exactly what is being signed.
#
# Signing: signature = Ed25519.sign(signing_key, serialize(envelope))
# Verification: Ed25519.verify(signerPubkey, serialize(envelope), signature)
struct SignedEnvelope {
  envelope @0 :RequestEnvelope;    # The data being signed
  signature @1 :Data;              # Ed25519 signature (64 bytes) over serialized envelope
  signerPubkey @2 :Data;           # Ed25519 public key (32 bytes)
}

# Signed response envelope
#
# All RPC responses are signed for E2E authentication, preventing MITM attacks
# on response data (e.g., DH public keys in StreamInfo).
#
# Signing: signature = Ed25519.sign(server_key, requestId || payload)
# Verification: Ed25519.verify(signerPubkey, requestId || payload, signature)
struct ResponseEnvelope {
  requestId @0 :UInt64;    # Correlates with RequestEnvelope.requestId
  payload @1 :Data;        # Serialized inner response
  signature @2 :Data;      # Ed25519 signature (64 bytes)
  signerPubkey @3 :Data;   # Ed25519 public key (32 bytes)
}

# =============================================================================
# Streaming types moved to streaming.capnp
# =============================================================================
#
# The following types are now defined in streaming.capnp:
#   - StreamInfo, StreamRegister, StreamStartRequest, StreamAuthResponse
#   - StreamBlock, StreamPayload, StreamStats, StreamError, StreamResume
#
# Import with: using Streaming = import "streaming.capnp";

# =============================================================================
# Authorization
# =============================================================================

# Structured scope for fine-grained authorization
# Format: action:resource:identifier
# Examples:
#   infer:model:qwen-7b     - Specific model inference
#   subscribe:stream:abc    - Specific stream subscription
#   read:model:*            - Read any model (explicit wildcard)
#   admin:*:*               - Admin wildcard
struct Scope {
  action @0 :Text;       # read, write, infer, subscribe, admin
  resource @1 :Text;     # model, stream, policy
  identifier @2 :Text;   # specific ID or "*" for wildcard
}

# JWT claims with structured scopes
struct Claims {
  sub @0 :Text;          # Subject (user/service)
  exp @1 :Int64;         # Expiration timestamp
  iat @2 :Int64;         # Issued at timestamp
  scopes @3 :List(Scope); # Structured scopes
  admin @4 :Bool;        # Admin override
}
