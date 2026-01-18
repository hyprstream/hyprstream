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
  ephemeralPubkey @3 :Data;        # X25519/P-256 public key for stream HMAC (optional, 32 bytes)
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

# Response envelope (for non-streaming responses)
# Non-streaming responses don't need signatures (same connection, correlates with request)
struct ResponseEnvelope {
  requestId @0 :UInt64;  # Correlates with RequestEnvelope.requestId
  payload @1 :Data;      # Serialized inner response
}

# Stream chunk for XPUB/XSUB streaming responses
#
# Uses chained HMAC-SHA256 for authentication AND cryptographic ordering.
# No sequence field needed - ordering is enforced by the HMAC chain:
#   mac_0 = HMAC(key, request_id_bytes || data_0)  # First chunk
#   mac_n = HMAC(key, mac_{n-1} || data_n)         # Subsequent chunks
#
# Verification of chunk N requires mac_{N-1}, providing cryptographic ordering.
struct StreamChunk {
  requestId @0 :UInt64;   # Which request this chunk belongs to
  data @1 :Data;          # Token or chunk data
  hmac @2 :Data;          # Chained HMAC-SHA256 (32 bytes)
}

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
