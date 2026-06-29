@0x8e593b39473af5fa;

using import "annotations.capnp".fixedSize;
using import "annotations.capnp".optional;

# Common Cap'n Proto types for hyprstream RPC
#
# This schema defines authorization and envelope types used across all services.
# Every request is wrapped in a SignedEnvelope that carries authorization context.
#
# Security Model:
# - Transport layer: CURVE encryption (TCP only)
# - Application layer: Ed25519 signatures (survives message forwarding)
# - Streaming: Chained HMAC-SHA256 for authentication + ordering

# Authorization union — replaces legacy identity + jwtToken + claims fields.
struct Authorization {
  union {
    none @0 :Void;
    local @1 :TokenClaims;
    federated @2 :FederatedToken;
    idJag @3 :Text;
  }
}

# Verified token claims (local issuer).
struct TokenClaims {
  iss @0 :Text;
  sub @1 :Text;
  aud @2 :List(Text);
  exp @3 :Int64;
  iat @4 :Int64;
  jti @5 :Text;
  scope @6 :List(Scope);
  cnfJkt @7 :Text;
}

# Federated token from a foreign issuer.
struct FederatedToken {
  raw @0 :Text;
  claims @1 :TokenClaims;
  dpopProof @2 :Text $optional;
}

# Unsigned request data - this is what gets signed
#
# Contains all request metadata and payload. The entire serialized RequestEnvelope
# is signed by SignedEnvelope.sig for clear signing scope.
struct RequestEnvelope {
  requestId @0 :UInt64;            # Unique request ID for correlation
  payload @1 :Data;                # Serialized inner request (RegistryRequest, etc.)
  iat @2 :Int64;                   # Unix millis, for expiration check
  nonce @3 :Data $fixedSize(16);   # 16 random bytes for replay protection
  authorization @4 :Authorization; # Authorization context
  delegationToken @5 :Text $optional;  # Delegation token relayed by a trusted service
  wth @6 :Data $fixedSize(32) $optional;  # SHA-256 of jwtToken (WIT binding)
  clientDhPublic @7 :Data $fixedSize(32) $optional;  # Ephemeral DH public key for stream key derivation
}

# Signed wrapper - signature covers serialized RequestEnvelope bytes
#
# All RPC requests should be wrapped in this envelope.
# The nested structure makes clear exactly what is being signed.
#
# Cleartext mode:
#   sig = Ed25519.sign(signing_key, serialize(envelope))
#   Verification: Ed25519.verify(cnf, serialize(envelope), sig)
#
# Encrypted mode (when encryptedEnvelope is present):
#   sig = Ed25519.sign(signing_key, encryptedEnvelope || clientEphemeralPublic)
#   Verification: Ed25519.verify(cnf, encryptedEnvelope || clientEphemeralPublic, sig)
#   Decryption: X25519 DH(server_sk, clientEphemeralPublic) -> AES-256-GCM-SIV
struct SignedEnvelope {
  envelope @0 :RequestEnvelope;    # Cleartext envelope (used when encryptedEnvelope is absent)
  sig @1 :Data $fixedSize(64);     # Ed25519 signature (64 bytes) — cnf, sig retained for the
  cnf @2 :Data $fixedSize(32);     # signer-pubkey advertisement + transition; auth comes from `cose`.
  encryptedEnvelope @3 :Data;      # AES-256-GCM-SIV ciphertext of serialized RequestEnvelope
  clientEphemeralPublic @4 :Data $fixedSize(32);  # X25519 ephemeral public key for DH
  # M3 (#152): COSE composite signature (RFC 9052 COSE_Sign), detached over the
  # canonical RequestEnvelope (cleartext) or the encrypted signing-data.
  #   - Classical mode: ONE EdDSA COSE_Signature entry.
  #   - Hybrid mode: TWO entries (EdDSA + ML-DSA-65).
  # The ML-DSA-65 verifying key is NOT carried here; it is resolved by kid from
  # the trust store (kid-anchored), fixing the prior self-certification gap.
  cose @5 :Data;                   # CBOR-encoded COSE_Sign (composite signatures)
  pqKemCiphertext @6 :Data;        # ML-KEM-768 ciphertext (1088 bytes) for hybrid encryption
}

# Signed response envelope
#
# All RPC responses are signed for E2E authentication, preventing MITM attacks
# on response data (e.g., DH public keys in StreamInfo).
#
# Signing: sig = Ed25519.sign(server_key, requestId || payload)
# Verification: Ed25519.verify(cnf, requestId || payload, sig)
struct ResponseEnvelope {
  requestId @0 :UInt64;    # Correlates with RequestEnvelope.requestId
  payload @1 :Data;        # Serialized inner response
  sig @2 :Data $fixedSize(64);     # Ed25519 signature (64 bytes) — cnf, sig retained for the
  cnf @3 :Data $fixedSize(32);     # signer-pubkey advertisement + transition; auth comes from `cose`.
  # #275: COSE composite signature (RFC 9052 COSE_Sign1), detached over the
  # response signing-data (requestId || payload). Mirrors SignedEnvelope.cose.
  #   - Classical mode: ONE EdDSA COSE_Sign1 entry.
  #   - Hybrid mode: TWO entries (EdDSA + ML-DSA-65), nested (outer signs
  #     `signing-data || inner_eddsa_signature`).
  # The signing domain is bound to RESPONSE_ENVELOPE_TYPE_ID (distinct from the
  # request domain), so a response COSE signature can never verify as a request.
  # The ML-DSA-65 verifying key is NOT carried here; it is resolved by kid from
  # the trust store (kid-anchored), fixing the prior self-certification gap.
  cose @4 :Data;                   # CBOR-encoded nested COSE composite signature
}

# Authorization subject — bare username or "anonymous".
# All identity types produce bare usernames (no prefix).
struct Subject {
  name @0 :Text;  # Username, empty string = anonymous
}

# =============================================================================
# Streaming types moved to streaming.capnp
# =============================================================================
#
# The following types are now defined in streaming.capnp:
#   - StreamInfo, StreamRegister, StartStreamRequest, StreamAuthResponse
#   - StreamBlock, StreamPayload, StreamStats, StreamError
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
#   manage:*:*              - Manage all resources
struct Scope {
  action @0 :Text;       # read, write, infer, subscribe, manage
  resource @1 :Text;     # model, stream, policy
  identifier @2 :Text;   # specific ID or "*" for wildcard
}

# JWT claims with structured scopes
struct Claims {
  sub @0 :Text;          # Subject (user/service)
  exp @1 :Int64;         # Expiration timestamp
  iat @2 :Int64;         # Issued at timestamp
  scopes @3 :List(Scope); # Structured scopes
  admin @4 :Bool;        # Deprecated: always false. Use Casbin policies.
  aud @5 :Text;          # RFC 8707 audience (resource indicator)
  token @6 :Text;        # Original JWT for e2e verification by downstream services
  iss @7 :Text;          # Issuer URL (RFC 7519); hyprstream node that minted this token
  pubKey @8 :Text;       # Ed25519 public key (base64url) for service identity tokens
}

# UTC timestamp with nanosecond precision
struct Timestamp {
  seconds @0 :Int64;   # Seconds since Unix epoch (1970-01-01T00:00:00Z)
  nanos @1 :Int32;     # Nanosecond offset [0, 999999999]
}

# Standard error response used across all services
struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
  details @2 :Text;
}
