# Cryptography Architecture

Secure communication primitives for hyprstream RPC.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CRYPTOGRAPHY SUBSYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   Ed25519 Signing   │  │    Key Exchange     │  │   Chained HMAC      │  │
│  │                     │  │                     │  │                     │  │
│  │  • Request auth     │  │  • Ristretto255     │  │  • Stream auth      │  │
│  │  • Integrity        │  │  • ECDH P-256 (FIPS)│  │  • MAC chaining     │  │
│  │  • Non-repudiation  │  │  • Key derivation   │  │  • Ordering proof   │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

The cryptography module (`crates/hyprstream-rpc/src/crypto/`) provides:

| Component | Purpose | Performance |
|-----------|---------|-------------|
| Ed25519 Signatures | Request envelope authentication | ~10k/sec |
| Key Exchange | Stream HMAC key derivation | One-time per stream |
| Chained HMAC | Streaming response authentication | Millions/sec |

## Shared Signing Key

All hyprstream services use a **single shared Ed25519 signing key** for bidirectional authentication.

**Location:** `~/.local/share/hyprstream/models/.registry/keys/signing.key`

### Key Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SHARED KEY MANAGEMENT                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  First Startup                      Subsequent Startups                  │
│       │                                    │                             │
│       ▼                                    ▼                             │
│  Key file exists?                    Key file exists?                    │
│       │                                    │                             │
│    No │                                Yes │                             │
│       ▼                                    ▼                             │
│  Generate 32 bytes                   Load from file                      │
│  Save to signing.key                 (same key for all)                  │
│  chmod 0600                                                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Why Shared Key?

All services (CLI, systemd units, spawned InferenceServices) must use the **same key** because:

1. **Request verification** - Services verify client signatures
2. **Response verification** - Clients verify service signatures
3. **Cross-service calls** - ModelService → InferenceService uses same key

If keys differ, response verification fails with "Response signed by unexpected key".

## Ed25519 Digital Signatures

**Location:** `crates/hyprstream-rpc/src/crypto/signing.rs`

All ZMQ messages are signed with Ed25519 to provide authentication, integrity, and non-repudiation.

### Bidirectional Signing Flow

```
Client                                         Service
    │                                              │
    │  1. Create RequestEnvelope                   │
    │  2. Sign with shared_key                     │
    │                                              │
    │────────── SignedEnvelope ───────────────────►│
    │                                              │
    │                               3. Verify signature
    │                               4. Process request
    │                               5. Create ResponseEnvelope
    │                               6. Sign with shared_key
    │                                              │
    │◄───────── ResponseEnvelope ─────────────────│
    │                                              │
    │  7. Verify signature                         │
    │  8. Process response                         │
```

**Mandatory verification**: ZmqClient requires `server_verifying_key` at construction. There is no way to receive unverified response data.

### Request Envelope Structure

The request signed data is computed as:

```
SHA-512(request_id || identity_bytes || payload)
```

Where:
- `request_id`: 8 bytes (u64, little-endian)
- `identity_bytes`: Canonical serialization of RequestIdentity
- `payload`: Raw request payload

### Response Envelope Structure

**Schema:** `crates/hyprstream-rpc/schema/common.capnp`

```capnp
struct ResponseEnvelope {
    requestId @0 :UInt64;           # Correlates with request
    payload @1 :Data;               # Service-specific response
    timestamp @2 :Int64;            # Unix timestamp (milliseconds)
    signature @3 :Data;             # Ed25519 signature (64 bytes)
    signerPubkey @4 :Data;          # Server's verifying key (32 bytes)
}
```

The response signed data is computed as:

```
SHA-512(request_id || payload || timestamp)
```

### Response Signing API

```rust
use hyprstream_rpc::envelope::ResponseEnvelope;

// Service side: sign response
let response = ResponseEnvelope::new_signed(
    request_id,
    payload_bytes,
    &signing_key,
);

// Client side: verify and unwrap
let (request_id, payload) = hyprstream_rpc::envelope::unwrap_response(
    &wire_bytes,
    Some(&expected_verifying_key),  // Mandatory verification
)?;
```

### API

```rust
use hyprstream_rpc::crypto::{
    generate_signing_keypair, sign_message, verify_message,
    signing_key_from_bytes, verifying_key_from_bytes,
    SigningKey, VerifyingKey,
};

// Generate new keypair
let (signing_key, verifying_key) = generate_signing_keypair();

// Sign a message
let signature: [u8; 64] = sign_message(
    &signing_key,
    request_id,      // u64
    identity_bytes,  // &[u8]
    payload,         // &[u8]
);

// Verify a signature
verify_message(
    &verifying_key,
    &signature,
    request_id,
    identity_bytes,
    payload,
)?;

// Serialize/deserialize keys
let secret_bytes = signing_key.to_bytes();
let restored = signing_key_from_bytes(&secret_bytes);

let public_bytes = verifying_key.to_bytes();
let restored = verifying_key_from_bytes(&public_bytes)?;
```

### Security Properties

| Property | Guarantee |
|----------|-----------|
| **Authentication** | Only holder of shared signing key can create valid signatures |
| **Bidirectional** | Both requests AND responses are signed (no MITM possible) |
| **Integrity** | Any modification invalidates the signature |
| **Non-repudiation** | Signature survives message forwarding through proxies |
| **Mandatory verification** | ZmqClient enforces response verification (no bypass) |
| **Pre-hashing** | Uses SHA-512 streaming hash (effectively Ed25519ph) |
| **Zeroization** | Secret keys are securely erased from memory when dropped |

### Key Loading

```rust
use hyprstream_core::cli::policy_handlers::load_or_generate_signing_key;

// Load shared key (generates if missing)
let keys_dir = models_dir.join(".registry").join("keys");
let signing_key = load_or_generate_signing_key(&keys_dir).await?;
let verifying_key = signing_key.verifying_key();

// Both signing_key and verifying_key are derived from the same 32-byte secret
// All services must load from the same file to interoperate
```

## Key Exchange

**Location:** `crates/hyprstream-rpc/src/crypto/key_exchange.rs`

Streaming responses use HMAC authentication instead of per-token Ed25519 signatures for performance. The HMAC key is derived from a Diffie-Hellman shared secret.

### Supported Algorithms

| Algorithm | Feature Flag | Description |
|-----------|--------------|-------------|
| Ristretto255 | Default | Prime-order group on Curve25519 |
| ECDH P-256 | `fips` | NIST curve for FIPS 140-2 compliance |

### Ristretto255 (Default)

Ristretto255 is preferred because it eliminates common DH vulnerabilities:

- **No cofactor issues** - All valid points are in the prime-order subgroup
- **No small subgroups** - Invalid encodings rejected at decode time
- **Canonical encodings** - Only one valid encoding per point

```rust
use hyprstream_rpc::crypto::{
    generate_ephemeral_keypair, ristretto_dh, derive_stream_keys,
    RistrettoSecret, RistrettoPublic, StreamKeys,
};

// Client generates ephemeral keypair
let (client_secret, client_public) = generate_ephemeral_keypair();

// Server generates ephemeral keypair
let (server_secret, server_public) = generate_ephemeral_keypair();

// Both compute shared secret
let client_shared = ristretto_dh(&client_secret, &server_public);
let server_shared = ristretto_dh(&server_secret, &client_public);
// client_shared == server_shared

// Derive stream keys for HMAC authentication
let keys: StreamKeys = derive_stream_keys(
    &client_shared,
    &client_public.to_bytes(),
    &server_public.to_bytes(),
)?;

// StreamKeys contains:
// - topic: 64-char hex string for ZMQ PUB/SUB routing
// - mac_key: 32-byte HMAC key for MAC chain
```

### ECDH P-256 (FIPS Mode)

When compiled with `--features fips`, uses NIST P-256 curve (SP 800-56A approved):

```rust
// Same API, different underlying algorithm
let (secret, public) = generate_ephemeral_keypair();  // P-256 keypair
```

### Stream Key Derivation

`derive_stream_keys()` uses HKDF-SHA256:

```
Salt = XOR(client_pub, server_pub)   // Binds both parties
IKM  = DH shared secret

Topic   = HKDF-Expand(IKM, Salt, info="topic")  // 32 bytes → 64 hex chars
MAC Key = HKDF-Expand(IKM, Salt, info="mac")    // 32 bytes
```

### Key Exchange Protocol

```
Client                                         Server
  │                                              │
  │  1. Generate ephemeral keypair               │  2. Generate ephemeral keypair
  │     (client_secret, client_public)           │     (server_secret, server_public)
  │                                              │
  │  3. Include client_public in                 │
  │     signed RequestEnvelope                   │
  │                                              │
  │─────────── RequestEnvelope ─────────────────►│
  │                                              │
  │                               4. Extract client_public
  │                               5. DH(server_secret, client_public)
  │                               6. derive_stream_keys()
  │                               7. Use topic for routing
  │                                              │
  │◄────────── Stream chunks with HMAC ──────────│
  │                                              │
  │  8. DH(client_secret, server_public)         │
  │  9. derive_stream_keys()                     │
  │ 10. Verify HMAC chain                        │
```

### Security Checks

```rust
fn derive_stream_keys(...) -> Result<StreamKeys> {
    // Reject self-connection attacks
    if client_pub.ct_eq(server_pub).into() {
        return Err("client and server keys are identical");
    }

    // Ristretto255: No low-order point checks needed!
    // Invalid encodings are rejected at decode time.
    ...
}
```

## Chained HMAC

**Location:** `crates/hyprstream-rpc/src/crypto/hmac.rs`

Streaming responses use chained HMAC-SHA256 instead of per-token signatures for performance (millions of MACs/sec vs ~10k signatures/sec).

### Chained HMAC Design

Each chunk's HMAC depends on the previous chunk's HMAC:

```
mac_0 = HMAC(key, request_id_bytes || data_0)  // First chunk
mac_n = HMAC(key, mac_{n-1} || data_n)         // Subsequent chunks
```

This provides cryptographic ordering without explicit sequence numbers.

### API

```rust
use hyprstream_rpc::crypto::{ChainedStreamHmac, HmacKey};

// Server side: create producer
let mut producer = ChainedStreamHmac::from_bytes(mac_key, request_id);

// Compute MACs for stream chunks
let mac1 = producer.compute_next(b"token 1");
let mac2 = producer.compute_next(b"token 2");
let mac3 = producer.compute_next(b"[DONE]");

// Client side: create verifier
let mut verifier = ChainedStreamHmac::from_bytes(mac_key, request_id);

// Verify in order
verifier.verify_next(b"token 1", &mac1)?;
verifier.verify_next(b"token 2", &mac2)?;
verifier.verify_next(b"[DONE]", &mac3)?;
```

### Security Properties

| Property | Guarantee |
|----------|-----------|
| Authentication | Proves server holds the DH shared secret |
| Ordering | Reordering impossible - can't verify chunk N without mac_{N-1} |
| Request binding | Request ID binds all chunks to their request |
| Timing safety | Uses constant-time comparison (subtle crate) |
| Key zeroization | HMAC keys securely erased when dropped |

### Why Chained HMAC?

```
Traditional approach:                 Chained HMAC:
┌──────────────────────┐             ┌──────────────────────┐
│ Chunk 1              │             │ Chunk 1              │
│ sequence: 1          │             │ data                 │
│ data                 │             │ mac = HMAC(prev||d)  │
│ mac                  │             └──────────────────────┘
└──────────────────────┘                       │
         │                                     ▼
         ▼                           ┌──────────────────────┐
┌──────────────────────┐             │ Chunk 2              │
│ Chunk 2              │             │ data                 │
│ sequence: 2          │             │ mac = HMAC(prev||d)  │
│ data                 │             └──────────────────────┘
│ mac                  │
└──────────────────────┘

Attacker can replay chunks           Attacker cannot:
from different requests              - Reorder chunks (breaks chain)
with same sequence numbers           - Replay (different prev_mac)
```

### State Management

```rust
// Get current chain state (for persistence)
let state = verifier.chain_state();  // &[u8; 32]

// Initial state is request_id padded to 32 bytes
// After first chunk, state becomes mac_0
// After second chunk, state becomes mac_1
// etc.
```

## End-to-End Integration

### Complete Request/Response Flow (REQ/REP)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                      BIDIRECTIONAL AUTHENTICATED RPC                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Client creates signed request                                           │
│     ┌─────────────────────────────────────────┐                            │
│     │ SignedEnvelope                           │                            │
│     │   envelope: RequestEnvelope {            │                            │
│     │     request_id: 12345                    │                            │
│     │     identity: local:alice                │                            │
│     │     payload: {...}                       │                            │
│     │   }                                      │                            │
│     │   signature: Ed25519(shared_key)         │ ◄── Signed with shared key│
│     └─────────────────────────────────────────┘                            │
│                                    │                                        │
│                                    ▼                                        │
│  2. Server verifies request signature, processes                            │
│     unwrap_envelope(wire_bytes, &verifying_key, &nonce_cache)?             │
│                                    │                                        │
│                                    ▼                                        │
│  3. Server creates signed response                                          │
│     ┌─────────────────────────────────────────┐                            │
│     │ ResponseEnvelope                         │                            │
│     │   request_id: 12345                      │ ◄── Correlates request    │
│     │   payload: {...}                         │                            │
│     │   timestamp: 1762974327000               │                            │
│     │   signature: Ed25519(shared_key)         │ ◄── Signed with shared key│
│     │   signer_pubkey: [32 bytes]              │                            │
│     └─────────────────────────────────────────┘                            │
│                                    │                                        │
│                                    ▼                                        │
│  4. Client verifies response signature                                      │
│     unwrap_response(wire_bytes, Some(&expected_verifying_key))?            │
│     (verification is MANDATORY - no bypass)                                 │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Complete Streaming Flow (PUB/SUB with HMAC)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         AUTHENTICATED STREAMING                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Client creates request with DH public key                               │
│     ┌─────────────────────────────────────────┐                            │
│     │ RequestEnvelope                          │                            │
│     │   request_id: 12345                      │                            │
│     │   identity: local:alice                  │                            │
│     │   payload: {prompt: "Hello"}             │                            │
│     │   ephemeral_pubkey: [32 bytes]           │ ◄── Client DH public      │
│     │   signature: Ed25519(shared_key)         │ ◄── Signed with shared key│
│     └─────────────────────────────────────────┘                            │
│                                    │                                        │
│                                    ▼                                        │
│  2. Server verifies signature, performs DH key exchange                     │
│     shared = DH(server_secret, client_pubkey)                               │
│     keys = derive_stream_keys(shared, client_pub, server_pub)               │
│                                    │                                        │
│                                    ▼                                        │
│  3. Server sends signed response with stream info                           │
│     ┌─────────────────────────────────────────┐                            │
│     │ ResponseEnvelope                         │                            │
│     │   payload: StreamStarted {               │                            │
│     │     stream_id, endpoint, server_pubkey   │ ◄── DH server public      │
│     │   }                                      │                            │
│     │   signature: Ed25519(shared_key)         │ ◄── Signed with shared key│
│     └─────────────────────────────────────────┘                            │
│                                    │                                        │
│                                    ▼                                        │
│  4. Server streams data with chained HMACs                                  │
│     ┌─────────────────────────────────────────┐                            │
│     │ StreamChunk                              │                            │
│     │   topic: keys.topic                      │ ◄── DH-derived topic      │
│     │   data: "Hello"                          │                            │
│     │   hmac: HMAC(mac_key, prev || data)      │ ◄── Chained MAC           │
│     └─────────────────────────────────────────┘                            │
│                                    │                                        │
│                                    ▼                                        │
│  5. Client derives same keys and verifies MAC chain                         │
│     keys = derive_stream_keys(shared, client_pub, server_pub)               │
│     verifier.verify_next(data, hmac)?                                       │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Example: REQ/REP with Mandatory Response Verification

```rust
use hyprstream_rpc::prelude::*;

// === Shared Key (loaded from file in real code) ===
let (signing_key, verifying_key) = generate_signing_keypair();

// === Client: Create and Send Signed Request ===
let request_id = 12345u64;
let envelope = RequestEnvelope {
    request_id,
    identity: RequestIdentity::local(),
    payload: b"request data".to_vec(),
    ephemeral_pubkey: None,
    nonce: generate_nonce(),
    timestamp: current_timestamp_ms(),
    claims: None,
};

let signed = SignedEnvelope::new_signed(envelope, &signing_key);
let wire_bytes = serialize(&signed);
// send via ZMQ...

// === Server: Verify Request, Create Signed Response ===
let (ctx, payload) = unwrap_envelope(&wire_bytes, &verifying_key, &nonce_cache)?;
let response_payload = process_request(&ctx, &payload)?;

let response = ResponseEnvelope::new_signed(
    ctx.request_id,
    response_payload,
    &signing_key,  // Same shared key
);
let response_bytes = serialize(&response);
// send via ZMQ...

// === Client: Verify Response (MANDATORY) ===
let (request_id, payload) = unwrap_response(
    &response_bytes,
    Some(&verifying_key),  // Verification required, no bypass
)?;
```

### Example: Full E2E Streaming with Authentication

```rust
use hyprstream_rpc::crypto::{
    generate_signing_keypair, sign_message, verify_message,
    generate_ephemeral_keypair, ristretto_dh, derive_stream_keys,
    ChainedStreamHmac,
};

// === Shared Key Setup ===
let (signing_key, verifying_key) = generate_signing_keypair();

// === Client: Create Signed Request with DH Public ===
let (client_dh_secret, client_dh_public) = generate_ephemeral_keypair();

let request_id = 12345u64;
let identity = b"local:alice";
let payload = b"{\"prompt\": \"Hello\"}";

let signature = sign_message(
    &signing_key,  // Shared signing key
    request_id,
    identity,
    payload,
);

// === Server: Verify and Setup Stream ===
verify_message(&verifying_key, &signature, request_id, identity, payload)?;

let (server_dh_secret, server_dh_public) = generate_ephemeral_keypair();
let shared = ristretto_dh(&server_dh_secret, &client_dh_public);
let keys = derive_stream_keys(
    &shared,
    &client_dh_public.to_bytes(),
    &server_dh_public.to_bytes(),
)?;

// Server sends signed response with stream info
let stream_response = ResponseEnvelope::new_signed(
    request_id,
    serialize_stream_started(&keys.topic, &server_dh_public),
    &signing_key,
);
// Client verifies response before subscribing to stream

// Server produces HMAC-authenticated stream
let mut producer = ChainedStreamHmac::from_bytes(*keys.mac_key, request_id);
let chunks = vec![b"Hello", b", ", b"world!", b"[DONE]"];
let macs: Vec<_> = chunks.iter().map(|c| producer.compute_next(c)).collect();

// === Client: Verify Response, Then Verify Stream ===
// 1. Verify response signature
let (_, stream_info) = unwrap_response(&stream_response_bytes, Some(&verifying_key))?;

// 2. Subscribe and verify HMAC chain
let client_shared = ristretto_dh(&client_dh_secret, &server_dh_public);
let client_keys = derive_stream_keys(
    &client_shared,
    &client_dh_public.to_bytes(),
    &server_dh_public.to_bytes(),
)?;

let mut verifier = ChainedStreamHmac::from_bytes(*client_keys.mac_key, request_id);
for (chunk, mac) in chunks.iter().zip(macs.iter()) {
    verifier.verify_next(chunk, mac)?;
}
```

## Files

| File | Purpose |
|------|---------|
| `crates/hyprstream-rpc/src/crypto/mod.rs` | Module exports, documentation |
| `crates/hyprstream-rpc/src/crypto/signing.rs` | Ed25519 signatures |
| `crates/hyprstream-rpc/src/crypto/key_exchange.rs` | Ristretto255/ECDH P-256 key exchange |
| `crates/hyprstream-rpc/src/crypto/hmac.rs` | Chained HMAC for streaming |
| `crates/hyprstream-rpc/src/envelope.rs` | SignedEnvelope, ResponseEnvelope, unwrap functions |
| `crates/hyprstream-rpc/schema/common.capnp` | Cap'n Proto schema for envelope types |
| `crates/hyprstream/src/cli/policy_handlers.rs` | `load_or_generate_signing_key()` |

## Feature Flags

| Flag | Effect |
|------|--------|
| (default) | Ristretto255 key exchange |
| `fips` | ECDH P-256 key exchange (FIPS 140-2) |

## Dependencies

| Crate | Purpose |
|-------|---------|
| `ed25519-dalek` | Ed25519 signatures |
| `curve25519-dalek` | Ristretto255 (default) |
| `p256` | ECDH P-256 (FIPS mode) |
| `sha2` | SHA-256/SHA-512 hashing |
| `hkdf` | Key derivation |
| `hmac` | HMAC-SHA256 |
| `subtle` | Constant-time operations |
| `zeroize` | Secure memory cleanup |
| `rand` | Secure random generation |
