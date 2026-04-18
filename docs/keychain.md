# Key Chain Architecture

## Intent

A hierarchical trust chain where a single root of trust per realm (PolicyService) certifies service identity keys. Clients verify the chain from root to service key, establishing authenticated communication without sharing secrets.

## Trust Model

```
                        PolicyService (Root CA)
                        holds: root Ed25519 keypair
                              │
                  signs ServiceCertificate for each service
                              │
         ┌────────────────────┼────────────────────────┐
         │                    │                         │
   DiscoveryService      RegistryService           ModelService
   cert: {pubkey,        cert: {pubkey,            cert: {pubkey,
         signer,                signer,                   signer,
         expiry}                expiry}                   expiry}
         │                    │                         │
    serves certs to       serves registry data      runs inference
    clients via RPC       to authorized users       for authorized users
```

### Analogy

| PKI Concept | Hyprstream Equivalent |
|---|---|
| Root CA | PolicyService root Ed25519 keypair |
| Intermediate CA | (not yet — single-tier chain) |
| Certificate | `ServiceCertificate` signed by root |
| Certificate Store | DiscoveryService (resolver, like DNSSEC) |
| Certificate Transparency | Service announcement log |
| Certificate Revocation | Expiry in certificate; future: CRL/OCSP |

### Chain of Trust (Client Perspective)

```
1. Bootstrap: Client knows the PolicyService root pubkey
   Source: certHash pin from host config, or TOFU with user confirmation

2. Discovery: Client calls DiscoveryService.getEndpoints("model")
   → Returns ServiceEndpoints with ServiceCertificate per endpoint

3. Verification: Client verifies ServiceCertificate
   → Check: ed25519_verify(root_pubkey, cert.canonical_bytes(), cert.signature)
   → Check: cert.expiry > now()
   → Check: cert.service_name == "model"

4. Connection: Client creates RpcClient with the certified service pubkey
   → All subsequent SignedEnvelope verification uses this certified key

5. TLS Binding (optional): cert.tls_endorsement cross-links TLS ↔ Ed25519
   → Sign(tls_key, "TLS_ENDORSEMENT_V1" || ed25519_pubkey || domain)
```

## ServiceCertificate (Proposed Capnp Schema)

```capnp
# A certificate binding a service name to an Ed25519 public key.
# Signed by the realm's root key (PolicyService).
struct ServiceCertificate {
  # The Ed25519 public key being certified (32 bytes)
  pubkey @0 :Data;

  # Human-readable service name (e.g. "model", "registry", "discovery")
  serviceName @1 :Text;

  # Unix timestamp (seconds) when certificate was created
  issuedAt @2 :Int64;

  # Unix timestamp (seconds) when certificate expires
  expiresAt @3 :Int64;

  # Ed25519 signature from root key over canonical serialization
  # of (pubkey || serviceName || issuedAt || expiresAt)
  signature @4 :Data;

  # Ed25519 public key of the signer (root key)
  signerPubkey @5 :Data;

  # TLS endorsement: Sign(tls_key, "TLS_ENDORSEMENT_V1" || ed25519_pubkey || domain)
  tlsEndorsement @6 :Data;
  tlsDomain @7 :Text;
}
```

## Key Lifecycle

### Service Key Generation

Each service derives its signing key from the node's root key via HKDF:

```
Root Ed25519 seed
  │
  ├── HKDF(purpose="node-root") → node identity key (current: signing_key)
  │
  ├── HKDF(purpose="service:discovery") → DiscoveryService signing key
  ├── HKDF(purpose="service:registry")  → RegistryService signing key
  ├── HKDF(purpose="service:model")     → ModelService signing key
  └── HKDF(purpose="service:policy")    → PolicyService signing key
```

This is already partially implemented via `NodeIdentityProvider::identity_open(purpose)` and `derive_purpose_key()`. The gap is that services currently all share the root key directly instead of using purpose-derived keys.

### Certificate Issuance Flow

```
Service Startup                        PolicyService
     │                                      │
     │  1. Derive service signing key       │
     │     key = HKDF(root, "service:{name}")
     │     pubkey = key.verifying_key()     │
     │                                      │
     │  2. Request certificate              │
     │  ──── certificate_request ──────────►│
     │      {pubkey, service_name}          │
     │                                      │
     │                      3. Validate request
     │                         (verify caller identity)
     │                      4. Sign certificate
     │                         signature = Ed25519(root_key, cert_bytes)
     │                      5. Set expiry (e.g. 7 days)
     │                                      │
     │  ◄──── ServiceCertificate ───────────│
     │      {pubkey, name, issued, exp, sig}
     │                                      │
     │  6. Register with DiscoveryService   │
     │  ──── announce(service_name,         │
     │            endpoint, certificate) ──► │
```

### Certificate Refresh

Certificates have bounded lifetimes. Services refresh before expiry:

```
Certificate valid for 7 days
  │
  ├── Day 5: Service requests renewal from PolicyService
  │         (grace period: refresh at ~70% of lifetime)
  │
  ├── Day 7: Certificate expires
  │         DiscoveryService stops serving it
  │         New clients can't verify the service
  │         Existing connections still work (key hasn't changed)
  │
  └── If renewal fails: Service logs error, continues with expired cert
              Admin alerted, manual intervention needed
```

## Client-Side Verification

### Browser (WASM)

```typescript
// 1. Bootstrap: know the root pubkey (from host config certHash)
const rootPubkey = hostConfig.rootPubkey; // pinned

// 2. Discovery: get service endpoints with certificates
const endpoints = await discoveryClient.getEndpoints("model");

// 3. Verify certificate (in Rust/WASM, not TypeScript)
for (const endpoint of endpoints) {
  const cert = endpoint.certificate;
  const valid = verifyCertificate(cert, rootPubkey);
  if (!valid || cert.expiresAt < Date.now() / 1000) {
    continue; // Skip unverified/expired
  }
  // 4. Create RpcClient with certified pubkey
  const client = new RpcClient(endpoint.url, certHash, myPubkey, signFn, cert.pubkey);
}
```

### Native (Rust)

```rust
// Verification built into generated DiscoveryClient
let endpoints = discovery_client.get_endpoints("model").await?;
for ep in &endpoints {
    if !ep.certificate.verify(&root_pubkey)? {
        continue;
    }
    if ep.certificate.expires_at < now {
        continue;
    }
    let client = RpcClient::new(transport, signer, ep.certificate.pubkey);
}
```

## Service-to-Service Communication

Services currently use `RequestIdentity::anonymous()` for cross-service calls. With the key chain:

```
ModelService → RegistryService
  │
  │  1. ModelService signs request with its purpose-derived key
  │  2. RequestIdentity includes service identity:
  │     Peer { name: "model", curve_key: model_pubkey }
  │
  │  3. RegistryService verifies:
  │     a. Check ServiceCertificate for "model" (from DiscoveryService)
  │     b. Verify envelope signature against certified pubkey
  │     c. Check Casbin policy for peer → resource authorization
```

## Federation (Multi-Node)

Multiple hyprstream nodes can federate. Each node has its own root key and PolicyService:

```
Node A (root_key_A)                      Node B (root_key_B)
  │                                        │
  │  PolicyService_A trusts root_key_B     │
  │  (configured via trusted_issuers)      │
  │                                        │
  │  ◄──── Cross-sign certificates ───────►│
  │                                        │
  │  Client on Node A can verify           │
  │  services on Node B via:               │
  │  1. root_key_B certifies B's services  │
  │  2. PolicyService_A trusts root_key_B  │
  │  3. Chain: A trusts B → B certifies X  │
```

This extends the existing `federation_key_source` in `ServiceContext` — currently used for JWT verification across nodes, will also cover certificate chain validation.

## Security Properties

| Property | Mechanism |
|---|---|
| **Authentication** | Service key certified by root → verified by client |
| **Integrity** | Ed25519 signatures on certificates and envelopes |
| **Confidentiality** | TLS/QUIC transport encryption (certHash pinning) |
| **Non-repudiation** | Root key signature on service certificates |
| **Key separation** | Each service has its own HKDF-derived key |
| **Key rotation** | Certificate expiry + refresh cycle |
| **No shared secrets** | Each party holds its own private key |
| **Forward secrecy** | Streaming uses DH key exchange (Ristretto255) per stream |
| **Replay protection** | Nonces in SignedEnvelope + timestamp checks |

## Comparison with Current Implementation

### What Exists Today

| Component | Status | Location |
|---|---|---|
| Root Ed25519 keypair | Shared by all services (one key) | `ServiceContext::signing_key()` |
| HKDF purpose derivation | Implemented, used for discovery self-proof only | `node_identity.rs:derive_purpose_key()` |
| Service registration | In-process `EndpointRegistry` (ZMQ bind) | `crates/hyprstream-rpc/src/registry.rs` |
| Service announcement | Cross-process via Discovery RPC `announce` | `discovery.capnp:ServiceAnnouncement` |
| Self-proof in EndpointInfo | `Sign(key, pubkey \|\| ts \|\| exp)` — redundant with envelope | `discovery.capnp:EndpointInfo` |
| TLS endorsement | Computed at startup, served via EndpointInfo | `DiscoveryService::tls_endorsement` |
| Client trust | certHash pin → TLS → envelope signature → self-proof verify | `HyprstreamRpcContext.tsx` |
| JWT authorization | PolicyService issues JWTs, services verify | `ServiceContext::jwt_token` |

### Gaps (Current → Target)

| # | Gap | Impact | Fix |
|---|---|---|---|
| G1 | **All services share one root key** | Key compromise exposes all services; no key separation | Each service uses `derive_purpose_key(root, "service:{name}")` |
| G2 | **No ServiceCertificate type** | No structured certificate binding key → service name | Add `ServiceCertificate` to common.capnp |
| G3 | **PolicyService does authorization only** | No root CA capability; can't sign service keys | Extend PolicyService with certificate signing RPC |
| G4 | **Self-proof is a "selfie"** | `Sign(key, own_pubkey)` proves nothing the envelope doesn't already prove | Remove self-proof fields; replace with ServiceCertificate |
| G5 | **Service-to-service uses anonymous identity** | No accountability for cross-service calls | Use `RequestIdentity::Peer` with certified pubkey |
| G6 | **Client has no root pubkey to verify against** | Trust anchored on certHash pin only (TLS layer) | Expose root pubkey via discovery; client pins at config time |
| G7 | **No certificate refresh/rotation** | Keys are static; no lifecycle management | Add certificate expiry + renewal flow |
| G8 | **DiscoveryService serves same pubkey for all services** | All services appear to use one key | DiscoveryService serves per-service certificates |
| G9 | **announce() carries no identity proof** | Any caller can announce endpoints for any service | Require ServiceCertificate in announce requests |

### What Can Be Reused (No Changes Needed)

| Component | Why |
|---|---|
| `SignedEnvelope` + `ResponseEnvelope` | Already carry `signerPubkey` — client verifies against certified key |
| `NodeIdentityProvider` + `derive_purpose_key()` | Ready for per-service key derivation |
| `Transport` trait | Transport-agnostic; works with any key |
| Generated clients (`DiscoveryClient`, etc.) | Take `Arc<dyn RpcClient>` — key passed at construction |
| TLS endorsement mechanism | Reusable as field in `ServiceCertificate` |
| `federation_key_source` | Already resolves trusted issuer keys for JWTs; extends to certificate chains |
| Chained HMAC streaming | Ephemeral DH keys per stream; independent of service key chain |

## Migration Path

### Phase 1: Per-Service Key Derivation (No Wire Changes)

- Each service factory calls `ctx.identity_provider().identity_open("service:{name}")`
- Service signing key is purpose-derived, not the root key
- Self-proof continues to work (signs with derived key)
- Rollback: services can revert to root key with no client changes

### Phase 2: ServiceCertificate + PolicyService Signing

- Add `ServiceCertificate` to common.capnp
- PolicyService gains `sign_certificate(pubkey, service_name) → ServiceCertificate`
- DiscoveryService serves certificates instead of raw pubkey + self-proof
- Client verifies certificate chain
- Remove self-proof fields from EndpointInfo

### Phase 3: Service-to-Service Identity

- Cross-service calls use `RequestIdentity::Peer` with certified pubkey
- Receiving service verifies peer certificate before processing
- Replace `anonymous()` calls with authenticated peer identity

### Phase 4: Certificate Lifecycle

- Certificate expiry + refresh (7-day lifetime, refresh at day 5)
- Auto-renewal on service startup
- Revocation: expired certificates not served by DiscoveryService

### Phase 5: Federation

- Cross-node certificate trust via `federation_key_source`
- PolicyService A trusts PolicyService B's root key
- Client verifies chain: root_A → trust_B → cert_B → service_B

## Related Documentation

- `docs/cryptography-architecture.md` — Ed25519 signatures, DH key exchange, chained HMAC
- `docs/rpc-architecture.md` — RPC message flow, envelope structure, JWT authorization
- `crates/hyprstream-rpc/schema/common.capnp` — SignedEnvelope, RequestIdentity
- `crates/hyprstream-rpc/schema/streaming.capnp` — StreamInfo (analogue: no self-proof)
- `crates/hyprstream-rpc/src/node_identity.rs` — HKDF purpose derivation
- `crates/hyprstream-service/src/service/factory.rs` — ServiceContext, shared key
- `crates/hyprstream-discovery/src/service.rs` — DiscoveryService, self-proof generation
- `crates/hyprstream-discovery/schema/discovery.capnp` — EndpointInfo schema
