# ZMQ Federation — Option B Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Option A `iss`-guard in `ZmqService::verify_claims()` with real federation: resolve the issuer's Ed25519 key via `FederationKeyResolver` and verify the JWT with it, matching the HTTP `auth_middleware` behavior exactly.

**Architecture:** A minimal `FederationKeySource` async trait is defined in `hyprstream-rpc/src/auth/` to avoid a circular crate dependency. `FederationKeyResolver` in the `hyprstream` crate implements this trait. `ZmqService` gains two new optional methods — `local_issuer_url()` and `federation_key_source()` — and `verify_claims()` is promoted to `async` so it can await key resolution.

**Tech Stack:** `async_trait`, `ed25519_dalek`, `anyhow`, `hyprstream-rpc::auth`, `hyprstream::auth::FederationKeyResolver`, `ServiceContext` (factory wiring), `process_request` in `zmtp_quic.rs`.

---

## Background

The current state (post Option A, commit `584d48da`) hard-rejects any JWT with a non-empty `iss` on the ZMQ transport. This closes the impersonation vulnerability but also blocks legitimate federated traffic. Option B replaces the guard with real key resolution.

### Inventory of all ZmqService implementors

| Implementor | Crate | Serves external traffic? | Needs federation? |
|---|---|---|---|
| `PolicyService` | `hyprstream` | Yes | Yes |
| `RegistryService` | `hyprstream` | Yes | Yes |
| `DiscoveryService` | `hyprstream` | Yes (already has `oauth_issuer_url` field) | Yes |
| `McpService` | `hyprstream` | Yes | Yes |
| `NotificationService` | `hyprstream` | Yes | Yes |
| `ModelService` | `hyprstream` | Yes | Yes |
| `InferenceZmqAdapter` | `hyprstream` | Yes | Yes |
| `WorkerService` | `hyprstream-workers` | Internal only | No — default `None` |
| `WorkflowService` | `hyprstream-workers` | Internal only | No — default `None` |
| `EchoService` (test) | `hyprstream-rpc` | Test only | No — default `None` |

---

## Task 1: FederationKeySource trait in hyprstream-rpc

**Files:**
- Create: `crates/hyprstream-rpc/src/auth/federation.rs`
- Modify: `crates/hyprstream-rpc/src/auth/mod.rs`

**Step 1.** Create `crates/hyprstream-rpc/src/auth/federation.rs`:

```rust
//! Abstraction layer for multi-issuer JWT key resolution on the ZMQ transport.
//!
//! Defines `FederationKeySource`, an async trait implemented by
//! `hyprstream::auth::FederationKeyResolver`. The indirection avoids a circular
//! crate dependency: `hyprstream-rpc` cannot import from `hyprstream` directly.

use anyhow::Result;
use ed25519_dalek::VerifyingKey;

/// Resolves external JWT issuer URLs to Ed25519 verifying keys.
///
/// Implemented by `hyprstream::auth::FederationKeyResolver`. Services that
/// serve external traffic return `Some(Arc<dyn FederationKeySource>)` from
/// `ZmqService::federation_key_source()` so the default `verify_claims()`
/// can perform real key resolution instead of rejecting federated JWTs.
#[async_trait::async_trait]
pub trait FederationKeySource: Send + Sync + 'static {
    /// Return true if `issuer` is in the configured trusted-issuer list.
    fn is_trusted(&self, issuer: &str) -> bool;

    /// Fetch (or return from cache) the Ed25519 verifying key for `issuer`.
    ///
    /// Returns `Err` if the issuer is untrusted or key fetch fails.
    async fn get_key(&self, issuer: &str) -> Result<VerifyingKey>;
}
```

**Step 2.** In `crates/hyprstream-rpc/src/auth/mod.rs`, add:

```rust
pub mod federation;
pub use federation::FederationKeySource;
```

**Step 3.** Run:
```bash
OPENSSL_NO_VENDOR=1 cargo check -p hyprstream-rpc --features download-libtorch
```
Expected: clean compile.

**Step 4.** Commit:
```bash
git add crates/hyprstream-rpc/src/auth/
git commit -m "feat(rpc): add FederationKeySource trait for ZMQ federation abstraction"
```

---

## Task 2: FederationKeyResolver implements FederationKeySource

**Files:**
- Modify: `crates/hyprstream/src/auth/federation.rs`

**Step 1.** At the top of the file, add the import:
```rust
use hyprstream_rpc::auth::FederationKeySource;
```

**Step 2.** After the closing `}` of the existing `impl FederationKeyResolver { ... }` block, add:

```rust
#[async_trait::async_trait]
impl FederationKeySource for FederationKeyResolver {
    fn is_trusted(&self, issuer: &str) -> bool {
        self.is_trusted(issuer)
    }

    async fn get_key(&self, issuer: &str) -> anyhow::Result<ed25519_dalek::VerifyingKey> {
        self.get_key(issuer).await
    }
}
```

**Step 3.** Run:
```bash
OPENSSL_NO_VENDOR=1 cargo test -p hyprstream auth::federation --features download-libtorch
```
Expected: `test_untrusted_issuer_rejected` and `test_trusted_issuer_recognized` both pass.

**Step 4.** Commit:
```bash
git add crates/hyprstream/src/auth/federation.rs
git commit -m "feat: FederationKeyResolver implements FederationKeySource trait"
```

---

## Task 3: ZmqService trait changes

**Files:**
- Modify: `crates/hyprstream-rpc/src/service/zmq.rs`

### Step 1 — Add two new optional trait methods

Immediately after `fn expected_audience()` in the `ZmqService` trait body, add:

```rust
/// Local OAuth issuer URL used to distinguish locally-issued JWTs from
/// federated ones. Return `None` to treat all tokens as local.
///
/// When `Some(url)`, a JWT whose `iss` matches `url` (or is empty) is
/// verified with `self.verifying_key()`. Any other non-empty `iss` falls
/// through to `federation_key_source()`.
fn local_issuer_url(&self) -> Option<&str> {
    None
}

/// Federation key resolver for multi-issuer JWT verification on the ZMQ path.
///
/// Return `None` (default) to disable federated JWT acceptance. Services that
/// accept external / network traffic should return `Some(Arc<dyn FederationKeySource>)`.
/// Internal-only services (workers, workflows) keep the default.
fn federation_key_source(
    &self,
) -> Option<std::sync::Arc<dyn crate::auth::FederationKeySource>> {
    None
}
```

### Step 2 — Promote verify_claims to async with full federation branching

Replace the current `fn verify_claims` (everything from the `fn verify_claims` signature through the closing `}`) with:

```rust
/// E2E JWT verification with federation and downgrade attack protection.
///
/// Called by `process_request` after envelope signature verification.
/// Async because federated key resolution may require an HTTP JWKS fetch.
async fn verify_claims(&self, ctx: &EnvelopeContext) -> anyhow::Result<()> {
    if let Some(claims) = ctx.claims() {
        // A token is local when iss is empty (legacy / intra-cluster) OR
        // matches the configured local OAuth issuer URL.
        let is_local = claims.iss.is_empty()
            || self
                .local_issuer_url()
                .map_or(false, |local| local == claims.iss);

        if is_local {
            // --- Local token path ---
            match claims.verify_token(&self.verifying_key(), self.expected_audience()) {
                Ok(Some(verified)) => {
                    if verified.sub != claims.sub {
                        tracing::warn!(
                            "Claims sub mismatch: envelope={} jwt={}",
                            claims.sub,
                            verified.sub
                        );
                        anyhow::bail!("Claims subject mismatch");
                    }
                    if verified.iss != claims.iss {
                        tracing::warn!(
                            "Claims iss mismatch: envelope={} jwt={}",
                            claims.iss,
                            verified.iss
                        );
                        anyhow::bail!("Claims issuer mismatch");
                    }
                }
                Ok(None) => {
                    // SECURITY: Non-local requests MUST carry a JWT.
                    if !ctx.identity.is_local() {
                        tracing::warn!(
                            "Non-local request with claims but no JWT token: \
                             sub={}, identity={:?}",
                            claims.sub,
                            ctx.identity
                        );
                        anyhow::bail!("JWT token required for non-local requests");
                    }
                }
                Err(e) => {
                    tracing::warn!("E2E JWT verification failed: {}", e);
                    anyhow::bail!("JWT verification failed");
                }
            }
        } else {
            // --- Federated token path ---
            let resolver = self.federation_key_source().ok_or_else(|| {
                anyhow::anyhow!(
                    "Federated JWT (iss={}) rejected: no FederationKeySource \
                     configured on this service",
                    claims.iss
                )
            })?;

            if !resolver.is_trusted(&claims.iss) {
                anyhow::bail!(
                    "Federated JWT issuer '{}' is not in the trusted-issuer list",
                    claims.iss
                );
            }

            let fed_key = resolver.get_key(&claims.iss).await.map_err(|e| {
                tracing::warn!(
                    "Federation key resolution failed for iss={}: {}",
                    claims.iss,
                    e
                );
                anyhow::anyhow!("Federation key resolution failed: {}", e)
            })?;

            let token = claims.token.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "Federated JWT (iss={}) present in claims but no raw token embedded",
                    claims.iss
                )
            })?;

            let verified = crate::auth::decode_with_key(token, &fed_key, self.expected_audience())
                .map_err(|e| {
                    anyhow::anyhow!("Federated JWT signature invalid: {}", e)
                })?;

            if verified.sub != claims.sub {
                tracing::warn!(
                    "Federated claims sub mismatch: envelope={} jwt={}",
                    claims.sub,
                    verified.sub
                );
                anyhow::bail!("Federated claims subject mismatch");
            }
            if verified.iss != claims.iss {
                tracing::warn!(
                    "Federated claims iss mismatch: envelope={} jwt={}",
                    claims.iss,
                    verified.iss
                );
                anyhow::bail!("Federated claims issuer mismatch");
            }
        }
    }
    Ok(())
}
```

**Step 3.** Run:
```bash
OPENSSL_NO_VENDOR=1 cargo check -p hyprstream-rpc --features download-libtorch
```
Expected: compile error at the `zmtp_quic.rs` call site (missing `.await`) — this is expected and fixed in Task 4.

**Step 4.** Commit together with Task 4.

---

## Task 4: Update call site in zmtp_quic.rs

**Files:**
- Modify: `crates/hyprstream-rpc/src/transport/zmtp_quic.rs`

**Step 1.** Find the line that reads:
```rust
if let Err(e) = service.verify_claims(&ctx) {
```

Change it to:
```rust
if let Err(e) = service.verify_claims(&ctx).await {
```

That is the complete change. The surrounding `process_request` function is already `async`.

**Step 2.** Run:
```bash
OPENSSL_NO_VENDOR=1 cargo check -p hyprstream-rpc --features download-libtorch
```
Expected: clean compile.

**Step 3.** Run tests:
```bash
OPENSSL_NO_VENDOR=1 cargo test -p hyprstream-rpc --lib --features download-libtorch
```
Expected: all tests pass.

**Step 4.** Commit Tasks 3 + 4 together:
```bash
git add crates/hyprstream-rpc/src/service/zmq.rs \
        crates/hyprstream-rpc/src/transport/zmtp_quic.rs
git commit -m "feat(rpc): async verify_claims with local/federated JWT branching"
```

---

## Task 5: Add federation_key_source to ServiceContext

**Files:**
- Modify: `crates/hyprstream-rpc/src/service/factory.rs` (or wherever `ServiceContext` is defined — search for `struct ServiceContext`)

**Step 1.** Add field to `ServiceContext`:
```rust
/// Shared federation key resolver (None when no trusted_issuers are configured).
federation_key_source: Option<std::sync::Arc<dyn crate::auth::FederationKeySource>>,
```

Initialize to `None` in `ServiceContext::new()` / any constructors.

**Step 2.** Add getter and builder methods:
```rust
/// Get the federation key source (if configured).
pub fn federation_key_source(
    &self,
) -> Option<std::sync::Arc<dyn crate::auth::FederationKeySource>> {
    self.federation_key_source.clone()
}

/// Set the shared federation key source for multi-issuer ZMQ token acceptance.
pub fn with_federation_key_source(
    mut self,
    src: std::sync::Arc<dyn crate::auth::FederationKeySource>,
) -> Self {
    self.federation_key_source = Some(src);
    self
}
```

**Step 3.** In `main.rs` (or wherever `ServiceContext` is constructed), wire `ServerState::federation_resolver` into it:
```rust
// federation_resolver is Arc<FederationKeyResolver>; it implements FederationKeySource.
let fed_src: std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource> =
    state.federation_resolver.clone();
service_ctx = service_ctx.with_federation_key_source(fed_src);
```

**Step 4.** Run:
```bash
OPENSSL_NO_VENDOR=1 cargo check -p hyprstream --features download-libtorch
```
Expected: clean compile.

**Step 5.** Commit:
```bash
git commit -m "feat: thread FederationKeySource through ServiceContext"
```

---

## Task 6: Wire federation into external-traffic service implementors

For each of the 7 services below, the pattern is identical:
1. Add `local_issuer_url: Option<String>` and `federation_key_source: Option<Arc<dyn FederationKeySource>>` fields
2. Initialize both to `None` in the constructor
3. Add `with_local_issuer_url` and `with_federation_key_source` builder methods
4. Override `fn local_issuer_url()` and `fn federation_key_source()` in the `ZmqService` impl
5. Wire from the factory using `ctx.federation_key_source()` and `ctx.oauth_issuer_url()` (or equivalent)

### 6a — PolicyService (`crates/hyprstream/src/services/policy.rs`)

`local_issuer_url` → `self.default_audience.as_deref()` (the field that already holds the issuer URL)

```rust
fn local_issuer_url(&self) -> Option<&str> {
    self.default_audience.as_deref()
}
fn federation_key_source(
    &self,
) -> Option<std::sync::Arc<dyn hyprstream_rpc::auth::FederationKeySource>> {
    self.federation_key_source.clone()
}
```

Factory wiring (`create_policy_service` in `factories.rs`):
```rust
if let Some(fed) = ctx.federation_key_source() {
    policy_service = policy_service.with_federation_key_source(fed);
}
```

### 6b — DiscoveryService (`crates/hyprstream/src/services/discovery.rs`)

`local_issuer_url` → `self.oauth_issuer_url.as_deref()` (field already exists).

Only add `federation_key_source` field; the `local_issuer_url` override already has a source.

### 6c — RegistryService (`crates/hyprstream/src/services/registry.rs`)

No existing issuer URL field — add `local_issuer_url: Option<String>` and builder.
Wire from factory: `ctx.oauth_issuer_url()` if that method exists, otherwise pass `config.oauth.issuer_url()`.

### 6d — McpService (`crates/hyprstream/src/services/mcp_service.rs`)

`McpService` already does federation inline for its HTTP path. For ZMQ, add fields and delegate through the standard pattern. `local_issuer_url` → the `oauth_issuer_url` captured at construction time.

### 6e — NotificationService (`crates/hyprstream/src/services/notification.rs`)

Add fields and builders. `local_issuer_url` → passed via factory from `config.oauth.issuer_url()`.

### 6f — ModelService (`crates/hyprstream/src/services/model.rs`)

`ModelServiceInner` holds `expected_audience: Option<String>`. Add the two fields there.
The `ZmqService` impl delegates: `self.inner.local_issuer_url.as_deref()` / `self.inner.federation_key_source.clone()`.

### 6g — InferenceZmqAdapter (`crates/hyprstream/src/services/inference.rs`)

Add fields to the private adapter struct. Wire in the adapter construction site.

**Step after all 7:** Run:
```bash
OPENSSL_NO_VENDOR=1 cargo clippy -p hyprstream --features download-libtorch -- -D warnings
OPENSSL_NO_VENDOR=1 cargo test -p hyprstream --lib --features download-libtorch
```
Expected: 308 tests pass, no clippy errors.

**Commit:**
```bash
git commit -m "feat: wire FederationKeySource into all external ZMQ service implementors"
```

---

## Task 7: Remove Option A guard, final cleanup + tests

**Files:**
- Modify: `crates/hyprstream-rpc/src/service/zmq.rs`

**Step 1.** Confirm the Option A guard block is gone (it was replaced wholesale in Task 3). Search for any remaining references:
```bash
grep -n "Option A\|option-a\|Federated JWT.*rejected" \
  crates/hyprstream-rpc/src/service/zmq.rs
```
Expected: no matches (guard was replaced, not left as dead code).

**Step 2.** Remove the stale TODO comment referencing Option B (it's now implemented):
```bash
grep -n "TODO(task-9" crates/hyprstream-rpc/src/service/zmq.rs
```
Delete any remaining TODO lines related to this feature.

**Step 3.** Full workspace build + tests:
```bash
OPENSSL_NO_VENDOR=1 cargo build --workspace --features download-libtorch
OPENSSL_NO_VENDOR=1 cargo clippy --workspace --features download-libtorch -- -D warnings
OPENSSL_NO_VENDOR=1 cargo test -p hyprstream --lib --features download-libtorch
OPENSSL_NO_VENDOR=1 cargo test -p hyprstream-rpc --lib --features download-libtorch
```
Expected: all tests pass, no warnings.

**Step 4.** Final commit:
```bash
git commit -m "feat(task-9): complete ZMQ federation — remove Option A guard"
```

---

## Sequencing summary

```
Task 1 (FederationKeySource trait in hyprstream-rpc)
  ↓
Task 2 (FederationKeyResolver implements FederationKeySource)
  ↓
Task 3 + 4 (ZmqService: async verify_claims + zmtp_quic call site)
  ↓
Task 5 (ServiceContext gains federation_key_source field + builder)
  ↓
Task 6 (Wire into 7 external-traffic service implementors)
  ↓
Task 7 (Remove Option A guard, full tests)
```

---

## Key design decisions

**Why `is_trusted()` on the trait?** Avoids an async round-trip for the common case of a clearly untrusted issuer, matching the fast-reject pattern in `auth_middleware`.

**Why `local_issuer_url()` as `Option<&str>`?** Services without an issuer URL (test services, WorkerService, WorkflowService) default to `None`, treating all tokens as local — which is safe because they sit behind the cluster boundary.

**Why keep `verify_claims` a default method?** The 3 internal-only services (WorkerService, WorkflowService, EchoService) need zero changes — they inherit the default impl and return `None` for `federation_key_source()`, which correctly hard-rejects foreign tokens.

**The `claims.token` field:** The raw JWT string is carried inside `Claims` via Cap'n Proto. The federated path extracts it with `claims.token.as_deref()`. No wire format changes needed.

**Circular dependency avoidance:** The `FederationKeySource` trait lives in `hyprstream-rpc/src/auth/` and only depends on `ed25519_dalek` + `anyhow`. `FederationKeyResolver` in `hyprstream` implements it. No cycle.
