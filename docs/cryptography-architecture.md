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
| Ed25519 Signatures | Classical envelope signature (inner COSE entry) | ~10k/sec |
| COSE Composite (EdDSA + ML-DSA-65) | Authoritative envelope authentication (see Hybrid PQ/T section) | Per request |
| Key Exchange | Stream HMAC key derivation | One-time per stream |
| Chained HMAC | Streaming response authentication | Millions/sec |
| Envelope Encryption | Optional envelope confidentiality (AES-256-GCM-SIV + hybrid KEM) | Per request |

## Signing Keys

The node's Ed25519 signing key lives at **`<secrets_dir>/signing-key`**, where the
secrets directory is resolved via `HyprConfig::resolve_secrets_dir()` (the
configured `secrets` directory, or the platform XDG default). It is loaded by
`load_or_generate_signing_key()` (`crates/hyprstream/src/cli/policy_handlers.rs`):

1. `HYPRSTREAM__SIGNING_KEY` env var / `config.signing_key` — **test-only bypass**
2. Load the key file at `<secrets_dir>/signing-key`
3. Generate a new key and write it to the file (writable directory only)

### Per-Service Keys

Services do not share a single blanket key. Each service obtains its signing key
via `ctx.service_signing_key(name)` in its factory; in multi-process mode each
service signs with its **own** Ed25519 key, and verifiers resolve the expected
key for a peer service through the service trust store
(`hyprstream-service::service::trust_store`). Response verification is still
mandatory: a client always pins the expected verifying key for the service it
calls, and a mismatched signer fails with "Response signed by unexpected key".

## Ed25519 Digital Signatures

**Location:** `crates/hyprstream-rpc/src/crypto/signing.rs`

All RPC envelopes (requests and responses) are signed to provide authentication,
integrity, and non-repudiation. The **authoritative** mechanism is the COSE
composite signature (`cose` field — see the Hybrid PQ/T section below); the raw
Ed25519 `sig`/`cnf` fields are retained for signer-key advertisement and the JWT
`cnf` key-binding path.

### Bidirectional Signing Flow

```
Client                                         Service
    │                                              │
    │  1. Create RequestEnvelope                   │
    │  2. Sign with client key                     │
    │                                              │
    │────────── SignedEnvelope ───────────────────►│
    │                                              │
    │                               3. Verify signature
    │                               4. Process request
    │                               5. Create ResponseEnvelope
    │                               6. Sign with service key
    │                                              │
    │◄───────── ResponseEnvelope ─────────────────│
    │                                              │
    │  7. Verify signature                         │
    │  8. Process response                         │
```

**Mandatory verification**: `RpcClientImpl` (behind all generated clients, via the `RpcClient` trait) requires the server's verifying key at construction. There is no way to receive unverified response data.

### Request Envelope Structure

The signature covers the **serialized bytes of the entire `RequestEnvelope`**
(`crates/hyprstream-rpc/schema/common.capnp`), making the signing scope
structurally explicit:

```
Cleartext mode:  sig = Ed25519.sign(signing_key, serialize(envelope))
Encrypted mode:  sig = Ed25519.sign(signing_key, encryptedEnvelope || clientEphemeralPublic)
```

The envelope also carries the COSE composite (`cose` field), detached over the
canonical envelope bytes (or the encrypted signing-data), which is what
verification enforces. Identity travels in the envelope's `authorization` union
(`none` / `local` claims / `federated` token / `idJag`) — there is no separate
identity field in the signed data.

### Response Envelope Structure

**Schema:** `crates/hyprstream-rpc/schema/common.capnp`

```capnp
struct ResponseEnvelope {
    requestId @0 :UInt64;        # Correlates with RequestEnvelope.requestId
    payload @1 :Data;            # Serialized inner response
    sig @2 :Data $fixedSize(64); # Ed25519 signature — retained for signer
    cnf @3 :Data $fixedSize(32); # pubkey advertisement; auth comes from `cose`
    cose @4 :Data;               # CBOR-encoded nested COSE composite signature
}
```

The response signing-data is `request_id || payload`, and the response COSE
composite is domain-separated from the request one (bound to a distinct
`RESPONSE_ENVELOPE_TYPE_ID` in the COSE `external_aad`), so a response signature
can never verify as a request signature or vice versa.

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
| **Authentication** | Only the holder of the signing key can create valid signatures |
| **Bidirectional** | Both requests AND responses are signed (no MITM possible) |
| **Integrity** | Any modification invalidates the signature |
| **Non-repudiation** | Signature survives message forwarding through proxies |
| **Mandatory verification** | `RpcClientImpl` enforces response verification (no bypass) |
| **Pre-hashing** | Uses SHA-512 streaming hash (effectively Ed25519ph) |
| **Zeroization** | Secret keys are securely erased from memory when dropped |

### Key Loading

```rust
use hyprstream::cli::policy_handlers::load_or_generate_signing_key;

// Load the node signing key (generates if missing).
// The path argument is ignored; the actual location is always
// <secrets_dir>/signing-key via HyprConfig::resolve_secrets_dir().
// HYPRSTREAM__SIGNING_KEY is a test-only env bypass.
let signing_key = load_or_generate_signing_key(Path::new("")).await?;
let verifying_key = signing_key.verifying_key();

// Both signing_key and verifying_key are derived from the same 32-byte secret.
// Per-service keys are obtained via ctx.service_signing_key(name) in factories;
// verifiers resolve peer keys through the service trust store.
```

## Hybrid (PQ/T) Signatures — WNS posture

**Location:** `crates/hyprstream-rpc/src/crypto/cose_sign.rs`, `envelope.rs`

Envelopes (request `SignedEnvelope` and response `ResponseEnvelope`) carry a nested
COSE composite: an **inner EdDSA** `COSE_Sign1` and an **outer ML-DSA-65** (FIPS 204)
`COSE_Sign1` over `payload ‖ inner_sig`. This is a **Weak Non-Separable (WNS)**
construction in the IETF PQUIP taxonomy: the inner classical signature stays
*independently verifiable by design*, enabling gradual PQ migration.

### Specs of record
- **`draft-ietf-pquip-hybrid-signature-spectrums`** (PQUIP WG) — the SNS/WNS spectrum +
  nested-construction non-separability. Our construction is **WNS**, not SNS (no fusion).
- **RFC 9794** — PQ/T hybrid terminology.
- **`draft-ietf-lamps-pq-composite-sigs`** — the composite alg-id `id-MLDSA65-Ed25519`,
  bound into the COSE `external_aad` (Hybrid mode only) for *signature-level*
  non-separability: an inner EdDSA lifted out of a Hybrid composite is not a valid
  standalone Classical signature.
- **`draft-ietf-cose-mldsa`** — ML-DSA in COSE. Multicodec `ml-dsa-65-pub` = `0x1211`
  (DID `#mesh-pq` Multikey).
- PQUIP WG: <https://datatracker.ietf.org/wg/pquip/about/>

### Verification rule (per-identity, NOT blanket fail-closed)
```rust
// envelope.rs :: verify_cose (both SignedEnvelope and ResponseEnvelope)
let anchored_pq = pq_store.and_then(|s| s.ml_dsa_key_for(&self.cnf));
let require_pq = verify_policy.uses_pq() && anchored_pq.is_some();
```
Enforce the ML-DSA-65 outer **only** for signer identities whose PQ key is anchored
out-of-band (`KeyedPqTrustStore`, keyed by the signer's Ed25519 `cnf`). For an
**unanchored** signer, fall back to the inner EdDSA (classical floor) rather than
failing closed.

This is safe because `ed_vk = VerifyingKey::from_bytes(&self.cnf)` — the PQ-lookup
identity *is* the EdDSA-verified identity — and the response path constant-time-pins
`cnf == expected_pubkey` when the server key is known:

| Signer state | Behavior | Guarantee |
|---|---|---|
| **anchored** | `require_pq = true` → outer enforced | un-downgradable Hybrid (a PQ adversary cannot forge ML-DSA) |
| **unanchored** | classical inner-EdDSA floor | no weaker than the pre-PQ baseline |

> **Never** reintroduce blanket fail-closed Hybrid for unanchored peers: it rejected
> all responses on a fresh install (empty `mesh_peers`), including a node verifying its
> own in-process services. PQ is also **never** resolved from the self-asserted COSE
> entry (that would reintroduce the self-cert weakness).

### Rollout / known limitation
The escape hatch `HYPRSTREAM_ENVELOPE_POLICY=classical` downgrades both directions in
lock-step for staged rollout. In multi-process mode each service signs with its own
Ed25519 key but only the node/OAuth `#mesh-pq` VM is published, so remote per-service
identities verify at the classical floor until each service publishes/anchors its own
`#mesh-pq` via DID resolution (tracked under #137 / #279).

### DPoP keys are a separate keyspace from DID identity keys (#698 Decision D)

`PqTrustStore` anchors ML-DSA-65 keys to a subject's **DID identity key** — the same
long-term key `GlobalPqUcanVerifier` uses to verify UCAN chain signatures
(`require_pq: true`, hardcoded — every chain-validated **issuer** is cryptographically
proven `PqHybrid` the moment `evaluate_grant` succeeds). DPoP proofs (RFC 9449), by
design, use an ephemeral, per-session key that lives in a completely different
keyspace and rotates freely — nothing in the protocol binds "this DPoP key" to "that
DID's PQ-anchored identity key".

This matters at the MAC grant path (see `docs/mac-architecture.md` §"Assurance is a
labeling requirement") because chain validation only proves **issuers**; the UCAN
**audience** (the delegated actor — e.g. MCP acting on a user's behalf) is only named,
never a signer over the chain. Its only proof of possession is the DPoP proof itself,
which proves classical-key possession and nothing about PQ anchoring.

**Ratified decision (2026-07-03, on issue #698):** rather than inventing an ad hoc
DPoP↔DID binding, a delegated actor's assurance is derived exclusively from what its
key material cryptographically proves — Classical, unconditionally — until a
sanctioned binding exists. Options considered and their disposition:

| Option | Shape | Disposition |
|---|---|---|
| **A** | DPoP proof signed with the DID's own long-term identity key | Degenerate case of B, worse key hygiene (loses DPoP's per-session rotation) |
| **B** | Enrollment-time registration: sign the enrolled DID's authorized actor key(s) into the enrollment table (`CompiledPolicy.enrollment`) | **Sanctioned upgrade path** — filed as #718; reuses the existing hybrid-PQC-signed policy artifact, no new trust surface |
| **C** | Per-session hybrid-COSE attestation binding DPoP jkt → DID | New wire-format surface; deferred unless ephemeral rotation *at* PqHybrid assurance becomes a real requirement |
| **D** | Floor delegated-actor assurance at Classical permanently | **Adopted now** — truthful, fail-closed, zero new protocol surface |

`EnrollmentSubjectContextResolver` (`crates/hyprstream/src/mac/mod.rs`) implements
Option D: it resolves clearance off the signed policy's enrollment table but clamps
the resulting `SecurityContext`'s assurance to `VerifiedKeyMaterial::Classical` via
`SecurityContext::from_clearance`'s `min(clearance.assurance, derived)` rule,
regardless of what assurance the enrollment table itself asserts. #718 (Option B)
extends this by letting an enrollment entry register a specific, PQ-anchored actor
key; a DPoP proof key matching a registered key earns `PqHybrid` for that actor only
— never as a blanket default.

## Envelope Confidentiality

**Location:** `crates/hyprstream-rpc/src/crypto/envelope_crypto.rs`, `envelope.rs`

Request envelopes support an optional **encrypted mode** (encrypt-then-sign). The
serialized `RequestEnvelope` bytes are encrypted with **AES-256-GCM-SIV** and
carried in `SignedEnvelope.encrypted_envelope`; the cleartext `envelope` field is
unused on the wire in this mode.

Key agreement is X25519 static-ephemeral DH: the client generates an ephemeral
X25519 keypair (`client_ephemeral_public`) and derives the AEAD key against the
server's Ed25519 key converted to X25519 via the birational map
(`to_montgomery()`), with the KDF context `hyprstream-envelope-v1`.

Under the `Hybrid` crypto policy, encryption is a **hybrid KEM**: an ML-KEM-768
encapsulation against the server's KEM key is combined with the X25519 secret,
and the 1088-byte ciphertext travels in `SignedEnvelope.pq_kem_ciphertext`.
`Hybrid` mode requires both the server ML-KEM key and an ML-DSA-65 signing key —
it fails closed rather than silently downgrading.

The signature (raw Ed25519 and the COSE composite) covers the encrypted
signing-data:

```
signing_data = ciphertext ‖ client_ephemeral_public ‖ [pq_kem_ciphertext]
```

Constructors: `SignedEnvelope::new_signed_encrypted()` (classical) and
`new_signed_encrypted_hybrid()` / `new_signed_encrypted_with_policy()` (hybrid).

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
// - topic: 64-char hex string, used as the moq broadcast-path component
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

### Identified hybrid stream epochs (#554)

`stream_epoch.rs` defines the production cryptographic profile that replaces the
legacy Ristretto bootstrap as individual stream producers migrate. The client
places a fresh, suite-complete `clientKemPublic` in the signed (and, on network
carriers, sealed) request. The server accepts only the policy-pinned
`HyKEM-X25519-MLKEM768` suite and returns its suite-identified component
ciphertexts in the signed `StreamInfo.kemCiphertexts` field.

The HyKEM combiner secret is mixed with one canonical identified-session binding:

- explicit `owned-hybrid-transport` or `standard-public-relay` profile and the
  direct/relay route role;
- producer and consumer endpoint DIDs;
- both accepted-state CID512 digests and epochs;
- service, capability, track, and both KEM key IDs; and
- the pinned suite and epoch block limit.

Each direction and epoch expands separate MAC, AEAD, control, rekey-authentication,
nonce-domain, and epoch-namespace material. AES-GCM nonces are deterministic
`nonce_domain || sequenceNumber`; the domain changes across direction, track, and
epoch. An authenticated `StreamEpochCommit` advances the one-way ratchet
atomically: invalid, skipped, replayed, or cross-stream commits leave the prior
epoch committed. The publisher resets the per-epoch sequence and MAC chain but
never resets its MOQT Group counter, so a cached track/group/object identity is
never republished with different ciphertext.

The profile is carrier-neutral. A standard MOQT relay forwards ordinary opaque
Object bytes and receives neither the transcript nor traffic keys. Network key
release is still a separate #726 authorization PEP; the generic
`StreamKeyReleaseGate`/`StreamKeyReleasePrincipal` seam does not implement or
claim restricted-anonymous #1060-#1062 authorization.

The legacy `from_dh` constructors remain during the dependency-safe migration and
are not evidence that #554 or the later global #556 downgrade-removal ticket is
closed.

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

## Group-Keyed Event Encryption

**Location:** `crates/hyprstream-rpc/src/events.rs`, `crypto/group_key.rs`, `crypto/event_crypto.rs`

Events published on the moq event plane can be encrypted at the **group** level:
a symmetric group key encrypts every event for a subscriber group (O(1) per
publish instead of O(N) per-subscriber encryption). The wire type is
`EncryptedEvent`; the group key is distributed by wrapping it per subscriber
(`wrap_group_key` / `unwrap_group_key`), and key rotation is driven by a
`RekeyEvent` published under a `RekeyPolicy` (rotation interval, max key
lifetime, grace period). See
[eventservice-architecture.md](eventservice-architecture.md) for the full
EventService design.

## End-to-End Integration

### Complete Request/Response Flow

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
│     │     authorization: local(claims)         │                            │
│     │     payload: {...}                       │                            │
│     │   }                                      │                            │
│     │   sig / cnf + cose composite             │ ◄── Signed w/ client key  │
│     └─────────────────────────────────────────┘                            │
│                                    │                                        │
│                                    ▼                                        │
│  2. Server verifies request signature, processes                            │
│     unwrap_envelope(wire_bytes, &unwrap_opts)?                              │
│                                    │                                        │
│                                    ▼                                        │
│  3. Server creates signed response                                          │
│     ┌─────────────────────────────────────────┐                            │
│     │ ResponseEnvelope                         │                            │
│     │   request_id: 12345                      │ ◄── Correlates request    │
│     │   payload: {...}                         │                            │
│     │   sig / cnf + cose composite             │ ◄── Signed w/ service key │
│     └─────────────────────────────────────────┘                            │
│                                    │                                        │
│                                    ▼                                        │
│  4. Client verifies response signature                                      │
│     unwrap_response(wire_bytes, Some(&expected_verifying_key))?            │
│     (verification is MANDATORY - no bypass)                                 │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Complete Streaming Flow (moq broadcast with HMAC)

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         AUTHENTICATED STREAMING                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. Client creates request with DH public key                               │
│     ┌─────────────────────────────────────────┐                            │
│     │ RequestEnvelope                          │                            │
│     │   request_id: 12345                      │                            │
│     │   authorization: local(claims)           │                            │
│     │   payload: {prompt: "Hello"}             │                            │
│     │   client_dh_public: [32 bytes]           │ ◄── Client DH public      │
│     │   sig / cnf + cose composite             │ ◄── Signed w/ client key  │
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
│     │   sig / cnf + cose composite             │ ◄── Signed w/ service key │
│     └─────────────────────────────────────────┘                            │
│                                    │                                        │
│                                    ▼                                        │
│  4. Server streams data with chained HMACs                                  │
│     ┌─────────────────────────────────────────┐                            │
│     │ StreamChunk (moq broadcast at keys.topic)│                            │
│     │   topic: keys.topic                      │ ◄── DH-derived path       │
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

### Example: Request/Reply with Mandatory Response Verification

```rust
use hyprstream_rpc::prelude::*;

// === Keys (loaded from the secrets dir / trust store in real code) ===
let (signing_key, verifying_key) = generate_signing_keypair();

// === Client: Create and Send Signed Request ===
// RequestEnvelope::anonymous() fills request_id, iat, nonce, and an empty
// authorization union; other constructors attach JWT/federated authorization.
let envelope = RequestEnvelope::anonymous(b"request data".to_vec());

let signed = SignedEnvelope::new_signed(envelope, &signing_key);
let wire_bytes = serialize(&signed);
// send over the transport (inproc / UDS / QUIC / iroh)...

// === Server: Verify Request, Create Signed Response ===
let (ctx, payload) = unwrap_envelope(&wire_bytes, &unwrap_opts)?;
let response_payload = process_request(&ctx, &payload)?;

let response = ResponseEnvelope::new_signed(
    ctx.request_id,
    response_payload,
    &signing_key,  // The service's signing key
);
let response_bytes = serialize(&response);
// send over the transport...

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

// === Key Setup ===
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
| `crates/hyprstream-rpc/src/crypto/cose_sign.rs` | COSE composite (EdDSA + ML-DSA-65) sign/verify |
| `crates/hyprstream-rpc/src/crypto/cose_sign1.rs` | COSE_Sign1 encoding + `external_aad` schema binding |
| `crates/hyprstream-rpc/src/crypto/pq.rs` | Post-quantum primitives: ML-DSA-65, ML-KEM-768 |
| `crates/hyprstream-rpc/src/crypto/envelope_crypto.rs` | Envelope encryption (X25519 DH + AES-256-GCM-SIV, hybrid ML-KEM) |
| `crates/hyprstream-rpc/src/crypto/key_exchange.rs` | Ristretto255/ECDH P-256 key exchange |
| `crates/hyprstream-rpc/src/crypto/hmac.rs` | Chained HMAC for streaming |
| `crates/hyprstream-rpc/src/crypto/group_key.rs` | Group-key wrapping for event confidentiality |
| `crates/hyprstream-rpc/src/crypto/event_crypto.rs` | Event encryption primitives |
| `crates/hyprstream-rpc/src/crypto/broadcast_primitives.rs` | Reusable broadcast crypto helpers (fingerprints, AAD, one-shot MAC) |
| `crates/hyprstream-rpc/src/events.rs` | `EncryptedEvent`, `RekeyEvent`, group-keyed event codec |
| `crates/hyprstream-rpc/src/envelope.rs` | SignedEnvelope, ResponseEnvelope, unwrap functions, `PqTrustStore` |
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
