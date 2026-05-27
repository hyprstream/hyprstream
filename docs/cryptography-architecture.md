# Cryptography Architecture

Secure communication primitives for hyprstream RPC.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CRYPTOGRAPHY SUBSYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   Ed25519 Signing   │  │   Key Agreement     │  │   Chained HMAC      │  │
│  │                     │  │                     │  │                     │  │
│  │  • Envelope auth    │  │  • Ristretto255 DH  │  │  • Stream auth      │  │
│  │  • Non-repudiation  │  │  • blake3 KDF       │  │  • MAC chaining     │  │
│  │  • E2E integrity    │  │  • Envelope encrypt │  │  • Ordering proof   │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                              │
│  ┌─────────────────────┐                                                    │
│  │   ES256 (P-256)     │                                                    │
│  │                     │                                                    │
│  │  • DPoP verification│                                                    │
│  │  • atproto interop  │                                                    │
│  │  • JWK thumbprints  │                                                    │
│  └─────────────────────┘                                                    │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   AES-256-GCM-SIV   │  │   JWKS Rotation     │  │   PQ Migration      │  │
│  │   (planned)         │  │                     │  │   (planned)         │  │
│  │  • Envelope AEAD    │  │  • drain/active/lead│  │  • ML-DSA-65 sigs   │  │
│  │  • HW accelerated   │  │  • JWT signing keys │  │  • ML-KEM-768 KEM   │  │
│  │  • Nonce-misuse safe│  │  • 6h rotation check│  │  • Hybrid classical │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Component | Purpose | Performance |
|-----------|---------|-------------|
| Ed25519 Signatures | Envelope authentication, E2E integrity | ~10k/sec |
| ES256 (P-256) | DPoP proof verification, atproto interop | ~5k/sec |
| Key Agreement | Stream HMAC key + envelope encryption key derivation | One-time per request |
| Chained HMAC | Streaming response authentication + ordering | Millions/sec |
| AES-256-GCM-SIV (planned) | Envelope payload encryption (PFS) | ~6 GB/s (AES-NI) |
| JWKS Rotation | JWT signing key lifecycle (drain/active/lead) | Background, 6h interval |

## Two-Layer Security Model

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| **Transport** | CURVE/QUIC-TLS | Encrypts connection, authenticates immediate peer |
| **Application** | Signed envelope (Ed25519) | E2E integrity through brokers, authenticates originator |

The envelope signature is **not redundant** with transport security. Transport secures hop-by-hop; the envelope provides **end-to-end integrity through intermediaries** like the blind StreamService broker.

```
Client ──TLS──▶ Service ──TLS──▶ StreamService ──TLS──▶ Client
                    │                    │
                    └── envelope sig ────┘── E2E integrity
                         (survives relay)
```

## Ed25519 Digital Signatures

**Location:** `crates/hyprstream-rpc/src/crypto/signing.rs`

All RPC messages are signed with Ed25519. Each client and service has its own Ed25519 keypair. Identity is established via the TrustStore (key attestation) or JWT authorization — not by sharing a single key.

### Signing Flow

```
Client                                         Service
    │                                              │
    │  1. Build RequestEnvelope                    │
    │  2. Canonicalize via Cap'n Proto             │
    │  3. Sign canonical bytes with client key     │
    │                                              │
    │────────── SignedEnvelope ───────────────────►│
    │    { envelope, sig, cnf }                    │
    │                                              │
    │                               4. Verify sig against cnf
    │                               5. Resolve identity from authorization
    │                               6. Process request
    │                               7. Sign response with service key
    │                                              │
    │◄───────── ResponseEnvelope ─────────────────│
    │    { requestId, payload, sig, cnf }          │
    │                                              │
    │  8. Verify response sig against expected key │
    │  9. Process response                         │
```

### Envelope Structure (current)

**Schema:** `crates/hyprstream-rpc/schema/common.capnp`

```capnp
struct RequestEnvelope {
  requestId @0 :UInt64;
  payload @1 :Data;                         # Serialized inner request
  iat @2 :Int64;                            # Unix millis (replay window)
  nonce @3 :Data $fixedSize(16);            # 16 random bytes (replay protection)
  authorization @4 :Authorization;          # Auth context (union)
  delegationToken @5 :Text $optional;       # Bearer relay by trusted service
  wth @6 :Data $fixedSize(32) $optional;    # SHA-256(WIT) — WIMSE binding
}

struct SignedEnvelope {
  envelope @0 :RequestEnvelope;
  sig @1 :Data $fixedSize(64);              # Ed25519 over canonical(envelope)
  cnf @2 :Data $fixedSize(32);              # Signer's Ed25519 pubkey (RFC 7800)
}

struct ResponseEnvelope {
  requestId @0 :UInt64;
  payload @1 :Data;
  sig @2 :Data $fixedSize(64);
  cnf @3 :Data $fixedSize(32);
}
```

### Authorization Union

```capnp
struct Authorization {
  union {
    none @0 :Void;                # Anonymous
    local @1 :TokenClaims;       # Local issuer — covered by envelope sig
    federated @2 :FederatedToken; # Foreign issuer — JWKS-verified JWT
    idJag @3 :Text;              # Cross-domain grant (RFC 8693 / ID-JAG)
  }
}
```

- **Local**: Claims in Cap'n Proto, covered by envelope Ed25519 sig. No separate JWT verification needed.
- **Federated**: Raw JWT string for JWKS verification + optional DPoP proof (RFC 9449). DPoP accepts both EdDSA (Ed25519) and ES256 (P-256) — the latter required for atproto interop.
- **IdJag**: Cross-domain authorization grant for non-atproto federation.

### Subject Resolution

Identity comes from authorization, not from caller-asserted fields:

```
Authorization::Local(claims)    → claims.sub (covered by envelope sig)
Authorization::Federated(token) → token.claims.sub (after JWKS verification)
Authorization::IdJag(jwt)       → sub claim (after JWT verification + aud check)
Authorization::None             → key_derived_subject from TrustStore, or Anonymous
```

The `cnf` field (signer pubkey) is the canonical audit identifier. Its RFC 7638 thumbprint binds to `TokenClaims.cnfJkt` for proof-of-possession.

### Canonical Serialization

Signatures use Cap'n Proto canonical serialization for deterministic bytes:

```rust
let canonical = envelope.to_bytes();  // Cap'n Proto canonicalize()
let signature = signing_key.sign(&canonical);
```

This ensures signature verification succeeds across platforms and library versions.

### Security Properties

| Property | Guarantee |
|----------|-----------|
| **Authentication** | Ed25519 sig proves sender identity via cnf pubkey |
| **Bidirectional** | Both requests AND responses are signed |
| **Integrity** | Any modification invalidates the signature |
| **Non-repudiation** | Signature survives message forwarding through proxies |
| **Mandatory verification** | ZmqClient enforces response verification (no bypass) |
| **Canonical serialization** | Deterministic Cap'n Proto bytes for cross-platform signing |
| **Zeroization** | Secret keys securely erased from memory when dropped |

## Key Agreement

**Location:** `crates/hyprstream-rpc/src/crypto/key_exchange.rs`

Streaming responses use HMAC authentication derived from a Diffie-Hellman shared secret. The same DH exchange will also derive the envelope encryption key (when encrypt-then-sign lands).

### Algorithm

| Algorithm | Feature Flag | Description |
|-----------|--------------|-------------|
| Ristretto255 | Default | Prime-order group on Curve25519 |
| ECDH P-256 | `fips` | NIST curve for FIPS 140-2 compliance |

Ristretto255 eliminates common DH vulnerabilities: no cofactor issues, no small subgroups, canonical encodings only.

### Stream Key Derivation

`derive_stream_keys()` uses blake3 key derivation:

```
IKM  = DH(client_secret, server_public)     # Ristretto255 shared secret
Salt = XOR(client_pub, server_pub)           # Binds both parties

Topic   = blake3::derive_key("hyprstream-topic-v1", IKM || Salt)  → 32 bytes → 64 hex chars
MAC Key = blake3::derive_key("hyprstream-mac-v1", IKM || Salt)    → 32 bytes
```

### Key Exchange Protocol

```
Client                                         Server
  │                                              │
  │  1. Generate ephemeral DH keypair            │
  │     (client_secret, client_public)           │
  │                                              │
  │  2. Include client_public in                 │
  │     RequestEnvelope.clientDhPublic           │
  │                                              │
  │─────────── SignedEnvelope ──────────────────►│
  │                                              │
  │                               3. Extract clientDhPublic
  │                               4. Generate server ephemeral keypair
  │                               5. DH(server_secret, client_public) → shared
  │                               6. blake3 derive → (topic, mac_key)
  │                                              │
  │◄── ResponseEnvelope(StreamInfo) ────────────│
  │    { streamId, endpoint, dhPublic }          │
  │                                              │
  │  7. DH(client_secret, server_public) → shared│
  │  8. blake3 derive → same (topic, mac_key)    │
  │  9. Subscribe to topic on PUB/SUB            │
  │                                              │
  │◄══ StreamBlocks (HMAC-chained) ═════════════│
```

> **Note:** `clientDhPublic` is currently being restored to the `RequestEnvelope` as a signaling-plane field. It was temporarily removed during the envelope rearchitecture; the design decision is to treat the envelope as the signaling channel (analogous to Signal/WebRTC signaling), with DH key exchange as a natural signaling concern alongside authorization and replay protection.

## Chained HMAC

**Location:** `crates/hyprstream-rpc/src/crypto/hmac.rs`

Streaming responses use chained HMAC-SHA256 for authentication and ordering. Each block's HMAC depends on the previous block's HMAC, creating a cryptographic chain.

### Wire Format (StreamBlock)

```
ZMQ multipart:
  Frame 0:      topic (64 hex chars, DH-derived)
  Frame 1..N-1: capnp segments (StreamBlock)
  Frame N:      mac (16 bytes, truncated HMAC-SHA256)

MAC chain:
  Block 0: mac = HMAC(mac_key, topic_bytes || segments)[..16]
  Block N: mac = HMAC(mac_key, prev_mac || segments)[..16]
```

### Security Properties

| Property | Guarantee |
|----------|-----------|
| **Authentication** | Proves publisher holds the DH shared secret |
| **Ordering** | Reordering impossible — can't verify block N without mac_{N-1} |
| **Blind forwarding** | StreamService forwards without HMAC verification |
| **Request binding** | Topic cryptographically binds stream to DH exchange |
| **Timing safety** | Constant-time comparison (subtle crate) |
| **Key zeroization** | HMAC keys securely erased when dropped |

## JWKS Key Rotation

**Location:** `crates/hyprstream/src/auth/key_rotation.rs`

JWT signing keys rotate through three slots:

```
lead   — pre-published (nbf in future); clients see it in JWKS, no tokens use it yet
active — current issuance key
drain  — old active, still valid for verification until exp + drain_days
```

### Rotation Lifecycle

Background task checks every 6 hours:
1. `lead.nbf <= now` → promote lead → active, old active → drain
2. `drain.exp + drain_days < now` → remove drain, delete key material
3. `lead` is None and `active.exp - now < lead_days` → generate new lead, persist

### Configuration

| Setting | Default | Purpose |
|---------|---------|---------|
| `jwt_key_active_days` | 14 | How long a key is used for issuance |
| `jwt_key_lead_days` | 7 | How far before expiry to pre-generate successor |
| `jwt_key_drain_days` | 30 | How long old keys remain valid for verification |

### JWKS Endpoint

`GET /oauth/jwks` serves all rotation slots (drain + active + lead) plus the cluster CA key. Keys are identified by RFC 7638 JWK Thumbprint (`kid`). De-duplicated by `kid` to avoid serving the same key twice.

## Planned: Encrypt-then-Sign Envelope (PFS)

The envelope is the **signaling plane** — it carries authorization, key agreement, and replay protection alongside the application payload. Currently, the envelope is signed but not encrypted. A planned upgrade adds encryption for perfect forward secrecy.

### Design

```
┌──────────────────────────────────────────────────────────────┐
│  ENCRYPT-THEN-SIGN                                           │
│                                                              │
│  1. Client builds RequestEnvelope (all fields cleartext)     │
│  2. Serialize → plaintext bytes                              │
│  3. DH(client_ephemeral, server_static) → blake3 → key      │
│  4. Encrypt(AES-256-GCM-SIV, key, plaintext) → ciphertext   │
│  5. Sign(client_ed25519, ciphertext) → sig                   │
│                                                              │
│  Intermediary can:                                           │
│    ✓ Verify sig (proves sender identity)                     │
│    ✓ Read cnf (for routing, trust store lookup)              │
│    ✗ Read anything else                                      │
│                                                              │
│  Server can:                                                 │
│    ✓ Verify sig                                              │
│    ✓ DH → blake3 → key → decrypt → full RequestEnvelope     │
│                                                              │
│  PFS: ephemeral DH key destroyed after send.                 │
│       Recorded traffic: sig verifiable, content opaque.      │
└──────────────────────────────────────────────────────────────┘
```

### Why AES-256-GCM-SIV

| Factor | AES-256-GCM-SIV | XChaCha20-Poly1305 |
|--------|------------------|-------------------|
| HW acceleration | AES-NI + VAES (~6-10 GB/s) | Software only (~3 GB/s) |
| Nonce misuse | Safe (only leaks plaintext equality) | N/A (24B nonce, random safe) |
| NIC offload (RDMA) | ConnectX-6/7, BlueField-2+ | No NIC offload |
| Nonce strategy | Fixed zero nonce (one msg per key) | Random 24B nonce |

Each envelope derives a unique key from ephemeral DH, so only one message is ever encrypted per key. The nonce can be fixed (all zeros), eliminating nonce management entirely.

## Planned: Post-Quantum Migration

Google's quantum computing timeline targets 2029. Hyprstream plans a phased PQ migration using hybrid classical + post-quantum cryptography.

### PQ Envelope Schema (target)

```capnp
struct SignedEnvelope {
  # Key agreement (hybrid PQ + classical)
  kemCiphertext    @0 :Data;                # ML-KEM-768 encapsulation (1088B)
  clientDhPublic   @1 :Data $fixedSize(32); # X25519 ephemeral (classical)

  # Encrypted payload (AES-256-GCM-SIV)
  ciphertext       @2 :Data;               # AEAD(RequestEnvelope)

  # Dual signatures
  sigClassical     @3 :Data $fixedSize(64); # Ed25519
  sigPq            @4 :Data;               # ML-DSA-65 (3293B)

  # Signer identity
  cnf              @5 :Data $fixedSize(32); # Ed25519 pubkey
  cnfPqFingerprint @6 :Data $fixedSize(32); # blake3(ML-DSA-65 pubkey)
}
```

### Key Agreement: X25519 + ML-KEM-768 Hybrid

```
Client:
  x25519_ss = X25519(ephemeral_sk, server_x25519_pk)
  kem_ct, kem_ss = ML-KEM-768.Encaps(server_mlkem_pk)
  envelope_key = blake3::derive_key("hyprstream-envelope-v1", x25519_ss || kem_ss)

Server:
  x25519_ss = X25519(server_x25519_sk, clientDhPublic)
  kem_ss = ML-KEM-768.Decaps(server_mlkem_sk, kemCiphertext)
  envelope_key = blake3::derive_key("hyprstream-envelope-v1", x25519_ss || kem_ss)
```

If either X25519 or ML-KEM is unbroken, the envelope key is safe.

### Dual Signatures: Ed25519 + ML-DSA-65

- **Intermediaries verify Ed25519 only** — 32B pubkey, 64B sig, ~50μs. Cheap routing decisions.
- **Endpoints verify both** — ML-DSA pubkey looked up from TrustStore by `cnfPqFingerprint` (avoids 1952B pubkey in every message).
- **Gradual migration** — `sigPq = empty` during transition; enforcement via policy.

### Size Budget

| Design | Envelope overhead | CPU per request |
|--------|------------------|-----------------|
| Current (Ed25519, cleartext) | 128B | ~50μs |
| + Encrypt-then-sign | ~160B | ~100μs |
| + ML-DSA-65 + ML-KEM-768 | ~4.5KB | ~350μs |

### RDMA Impact

The RDMA hot path is **stream data** (StreamBlocks with symmetric HMAC chain) — already PQ-safe, no change needed. PQ overhead only applies to the signaling plane (REQ/REP), which runs at 10-100/sec and sets up streams that produce millions of tokens over symmetric crypto.

### OAuth / JOSE Compatibility

PQ migration on the internal envelope plane has **no IETF dependency** — it's an internal protocol. OAuth/JOSE PQ (ML-DSA in JWTs, DPoP) is blocked on:

- `draft-ietf-cose-dilithium-11` — ML-DSA for JOSE (RFC expected ~2027)
- `draft-ietf-jose-pq-composite-sigs-01` — ML-DSA-65-Ed25519 hybrid composite
- `draft-ietf-jose-pqc-kem-05` — ML-KEM in JWE

DPoP (RFC 9449) is algorithm-agnostic and will work with ML-DSA once registered. JWKS will use `kty: "AKP"` for PQ keys.

### Migration Phases

| Phase | Envelope | OAuth/JOSE | Timeline |
|-------|----------|------------|----------|
| 0 (current) | Ed25519, cleartext | Ed25519 (EdDSA) + ES256 DPoP verification | Now |
| 1 | + encrypt-then-sign + clientDhPublic | No change | Near-term |
| 2 | + ML-DSA-65 sigPq (optional) + ML-KEM-768 | No change | Pre-2029 |
| 3 | Require sigPq | No change | By 2029 |
| 4 | ML-DSA in JOSE | When RFCs land | ~2028 |

## Key Inventory

```
┌────────────────────────┬──────────────────┬──────────┬──────────────────┐
│ Key                    │ Purpose          │ Rotation │ Forward secrecy  │
├────────────────────────┼──────────────────┼──────────┼──────────────────┤
│ JWKS keys              │ Sign JWTs        │ 14-day   │ Yes (drain       │
│ (drain/active/lead)    │ (at+jwt)         │ active   │ keys deleted)    │
├────────────────────────┼──────────────────┼──────────┼──────────────────┤
│ Cluster CA key         │ Sign JWTs in     │ None     │ No (derived      │
│                        │ single-process   │ (static) │ from seed)       │
├────────────────────────┼──────────────────┼──────────┼──────────────────┤
│ Client Ed25519 keys    │ Sign request     │ None     │ No (persisted    │
│ (cnf in envelope)      │ envelopes        │          │ in keychain)     │
├────────────────────────┼──────────────────┼──────────┼──────────────────┤
│ Streaming DH keys      │ Derive HMAC      │ Per-     │ Yes (ephemeral,  │
│ (per-stream ephemeral) │ chain key        │ stream   │ destroyed)       │
├────────────────────────┼──────────────────┼──────────┼──────────────────┤
│ Transport keys         │ Encrypt          │ Per-     │ Yes (ECDHE in    │
│ (CURVE / QUIC TLS)     │ connection       │ conn     │ TLS, CURVE)      │
└────────────────────────┴──────────────────┴──────────┴──────────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `crates/hyprstream-rpc/src/crypto/mod.rs` | Module exports |
| `crates/hyprstream-rpc/src/crypto/signing.rs` | Ed25519 signatures |
| `crates/hyprstream-rpc/src/crypto/key_exchange.rs` | Ristretto255/ECDH P-256 key exchange |
| `crates/hyprstream-rpc/src/crypto/hmac.rs` | Chained HMAC for streaming |
| `crates/hyprstream-rpc/src/envelope.rs` | SignedEnvelope, ResponseEnvelope, Authorization |
| `crates/hyprstream-rpc/schema/common.capnp` | Cap'n Proto envelope + auth schema |
| `crates/hyprstream-rpc/schema/streaming.capnp` | StreamInfo, StreamBlock, StreamPayload |
| `crates/hyprstream/src/auth/key_rotation.rs` | JWKS key rotation (drain/active/lead) |
| `crates/hyprstream/src/services/oauth/dpop.rs` | DPoP proof verification (EdDSA + ES256) |
| `crates/hyprstream/src/services/oauth/jwks.rs` | JWKS endpoint |

## Feature Flags

| Flag | Effect |
|------|--------|
| (default) | Ristretto255 key exchange; ES256 DPoP verification always enabled |
| `fips` | ECDH P-256 key exchange (FIPS 140-2) |

## Dependencies

| Crate | Purpose |
|-------|---------|
| `ed25519-dalek` | Ed25519 signatures |
| `curve25519-dalek` | Ristretto255 (default) |
| `p256` | ECDH P-256 (FIPS mode) + ECDSA (ES256 DPoP verification) |
| `blake3` | Key derivation (stream keys, future envelope keys) |
| `sha2` | SHA-256 (wth binding, JWK thumbprints) |
| `hmac` | HMAC-SHA256 (stream chain) |
| `subtle` | Constant-time operations |
| `zeroize` | Secure memory cleanup |
| `rand` | Secure random generation |

## Related

- [[streaming-service-architecture]] — StreamService blind forwarding, HMAC chain verification
- [[rpc-architecture]] — Service topology, JWT authorization flow
- [[Envelope Rearchitecture - RFC-aligned Auth and E2E Integrity]] — Design doc for current envelope schema
- [[ES256 signing for atproto compat]] — P-256 signing for atproto interop
