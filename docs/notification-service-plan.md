# NotificationService — Blind Relay with Broadcast Encryption

## Context

Model loading in hyprstream takes ~60s but the ZMQ REQ/REP timeout is 30s. The CLI (`hyprstream quick infer`) needs to fire-and-forget a `load_model()` call and **wait asynchronously** for a notification that the model is ready. This requires a notification system that:

- Works across networked hosts (CLI, ModelService, InferenceService may be on different machines)
- Preserves the 3-layer security model (CURVE transport → Ed25519 signed envelopes → Casbin)
- Gives each subscriber its own DH-authenticated stream (per-subscriber topic via Ristretto255)
- NotificationService is an **untrusted relay** — cannot read payload content
- Supports future expansion to phone push / browser via QUIC/WebTransport

## Architecture — Blind Relay with Broadcast Encryption

```
Publisher (ModelService)                      Subscriber (CLI)
        │                                           │
        │ 1. publishIntent(claim, pub_pubkey)        │ 1. subscribe(claim, sub_pubkey)
        ▼                                           ▼
┌────────────────────────────────────────────────────────┐
│                  NotificationService                    │
│  ┌──────────────────┐  ┌─────────────────────────────┐ │
│  │ Casbin AuthZ      │  │ SubscriberRegistry           │ │
│  │ verify Claim sig  │  │  (scope, ephemeral_pubkey,   │ │
│  │ check policy      │  │   registered_topic, ttl)     │ │
│  └────────┬─────────┘  └──────────────┬──────────────┘ │
│           │                           │                 │
│  2. Rerandomize: blinded = sub_pub + r*G                │
│     Return BLINDED pubkeys (unlinkable across intents)  │
└───────────┼───────────────────────────┼─────────────────┘
            │                           │
            ▼                           │
   Publisher encrypts:                  │
   Payload includes Ed25519 attestation of ephemeral pubkey
   data_key = random AES-256 key        │
   ciphertext = AES-GCM(data_key, nonce, signed_payload, aad=intentId||scope)
   For each blinded_pubkey:             │
     DH(pub_secret, blinded_pubkey) → enc_key, mac_key
     capsule = AES-GCM(enc_key, data_key, aad=fingerprint)
     mac = one_shot_mac(mac_key, ciphertext)
            │                           │
   3. deliver(intentId, [(fingerprint, capsule, mac), ...], ciphertext)
            │                           │
            ▼                           │
     ┌──────────────┐                   │
     │ StreamService │ blind PULL→XPUB  │
     └──────┬───────┘                   │
            │ per-subscriber topic      │
            ▼                           ▼
       TMQ SUB (CLI)
       Receive r_i from notification_block header
       DH((sub_secret + r_i), pub_pubkey) → enc_key, mac_key
       Verify MAC → decrypt capsule → decrypt payload
       Verify Ed25519 attestation → confirm publisher identity
```

### What NS sees vs. doesn't see

| Visible to NS | NOT visible to NS |
|---|---|
| Verified Claim scope (from SignedEnvelope) | Payload content (AES-GCM encrypted) |
| Subscriber real pubkeys + blinding scalars | DH shared secrets |
| Registered XPUB topics (routing) | data_key (wrapped in per-subscriber capsule) |
| Casbin policy matches | Publisher identity (encrypted inside payload) |
| Subscriber count per scope | Physical destination |

### Linkability properties

| Scenario | Linkable? | Why |
|----------|----------|-----|
| Same publisher, same intent, multiple subscribers | No | Different blinded pubkeys per subscriber |
| Different publishers, same subscriber | **No** | Different blinding scalar `r_i` per intent → different blinded pubkeys |
| Colluding publishers | **No** | Each sees a different rerandomized point for the same subscriber |
| NS correlating across intents | Yes (NS knows real pubkeys) | NS sees the blinding relationship — trade-off for routing capability |

## Schema: `notification.capnp`

**File**: `crates/hyprstream/schema/notification.capnp`

Two-phase publish: `publishIntent` returns subscriber pubkeys → publisher encrypts → `deliver` sends encrypted capsules. NS never sees plaintext.

Requires `subscribe @7;` and `publish @8;` additions to `ScopeAction` enum in `annotations.capnp`. CGR reader discovers annotations by name — no code changes needed in `cgr_reader.rs`.

```capnp
@0xa1b2c3d4e5f6a7b8;

using import "/annotations.capnp".mcpScope;
using import "/annotations.capnp".mcpDescription;

struct NotificationRequest {
  id @0 :UInt64;
  union {
    subscribe @1 :SubscribeRequest
      $mcpScope(subscribe) $mcpDescription("Subscribe to notifications matching a claim scope");

    publishIntent @2 :PublishIntentRequest
      $mcpScope(publish) $mcpDescription("Get subscriber pubkeys for scope-matched broadcast encryption");

    deliver @3 :DeliverRequest
      $mcpScope(publish) $mcpDescription("Deliver encrypted broadcast to scope-matched subscribers");

    unsubscribe @4 :UnsubscribeRequest
      $mcpScope(subscribe) $mcpDescription("Tear down a notification stream");

    listSubscriptions @5 :Void
      $mcpScope(query) $mcpDescription("List active subscriptions for current identity");

    ping @6 :Void
      $mcpScope(query) $mcpDescription("Health check");
  }
}

struct SubscribeRequest {
  scopePattern @0 :Text;       # Claim scope pattern: "serve:model:*"
  ephemeralPubkey @1 :Data;    # Client Ristretto255 pubkey (32 bytes)
  ttlSeconds @2 :UInt32;       # Subscription TTL (default 600, max 3600)
}

struct PublishIntentRequest {
  scope @0 :Text;              # Claim scope: "serve:model:qwen3"
  publisherPubkey @1 :Data;    # Publisher's ephemeral Ristretto255 pubkey (32 bytes)
}

struct DeliverRequest {
  intentId @0 :Text;
  capsules @1 :List(RecipientCapsule);
  encryptedPayload @2 :Data;   # AES-256-GCM ciphertext (shared across recipients)
  nonce @3 :Data;              # AES-GCM nonce (12 bytes)
}

struct RecipientCapsule {
  pubkeyFingerprint @0 :Data;  # Blake3(BLINDED_pubkey)[..16] for routing (128-bit)
  wrappedKey @1 :Data;         # AES-GCM(enc_key, data_key, aad=blinded_fingerprint)
  keyNonce @2 :Data;           # AES-GCM nonce for key wrapping (12 bytes, random OsRng)
  mac @3 :Data;                # One-shot MAC(mac_key, ciphertext) — 32 bytes
}

struct UnsubscribeRequest {
  subscriptionId @0 :Text;
}

struct NotificationResponse {
  requestId @0 :UInt64;
  union {
    error @1 :ErrorInfo;
    subscribeResult @2 :SubscribeResponse;
    publishIntentResult @3 :PublishIntentResponse;
    deliverResult @4 :DeliverResponse;
    unsubscribeResult @5 :Void;
    listSubscriptionsResult @6 :SubscriptionList;
    pingResult @7 :PingInfo;
  }
}

struct SubscribeResponse {
  subscriptionId @0 :Text;
  assignedTopic @1 :Text;      # XPUB topic (pre-registered with StreamService)
  streamEndpoint @2 :Text;     # StreamService XPUB endpoint to connect to
}

struct PublishIntentResponse {
  intentId @0 :Text;           # UUID, valid for 30s
  recipientPubkeys @1 :List(Data);  # BLINDED pubkeys: sub_pub + r_i * G (unlinkable across intents)
}

struct DeliverResponse {
  deliveredCount @0 :UInt32;
}

# Wire format for messages forwarded through StreamService
# NS constructs this from DeliverRequest fields — subscriber parses it
# Sent as StreamPayload::data inside a StreamBlock (via StreamPublisher API)
struct NotificationBlock {
  publisherPubkey @0 :Data;     # Publisher's ephemeral Ristretto pubkey (32 bytes)
  blindingScalar @1 :Data;      # r_i (32 bytes) — subscriber needs for blinding-aware DH
  wrappedKey @2 :Data;          # AES-GCM(enc_key, data_key, aad=fingerprint)
  keyNonce @3 :Data;            # AES-GCM nonce for key wrapping (12 bytes, random OsRng)
  encryptedPayload @4 :Data;    # AES-GCM ciphertext (shared across recipients)
  nonce @5 :Data;               # AES-GCM nonce for payload (12 bytes, random OsRng)
  intentId @6 :Text;            # For length-prefixed AAD reconstruction
  scope @7 :Text;               # For length-prefixed AAD reconstruction
  publisherMac @8 :Data;        # One-shot MAC(mac_key, ciphertext) — 32 bytes
}

struct SubscriptionInfo {
  subscriptionId @0 :Text;
  scopePattern @1 :Text;
  createdAt @2 :Int64;
  expiresAt @3 :Int64;
}

struct SubscriptionList {
  subscriptions @0 :List(SubscriptionInfo);
}

struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
  details @2 :Text;
}

struct PingInfo {
  status @0 :Text;
  activeSubscriptions @1 :UInt32;
  totalDelivered @2 :UInt64;
}
```

## NotificationService Implementation

**File**: `crates/hyprstream/src/services/notification.rs`

NS is a blind relay — stores ephemeral pubkeys and topics for routing, never sees plaintext payloads or DH shared secrets.

```rust
pub struct NotificationService {
    subscribers: Arc<RwLock<SubscriberRegistry>>,
    pending_intents: Arc<RwLock<HashMap<String, PendingIntent>>>,
    delivery_counter: AtomicU64,
    started_at: Instant,
    // StreamChannel for registering topics + shared TMQ PUSH to StreamService
    stream_channel: StreamChannel,
    // Auth
    signing_key: Arc<SigningKey>,
    policy_client: Option<PolicyClient>,
    expected_audience: Option<String>,
    // Infrastructure
    context: Arc<zmq::Context>,
    transport: TransportConfig,
}

struct PendingIntent {
    scope: String,
    publisher_subject: String,   // CRITICAL: verified caller identity for deliver auth
    publisher_pubkey: Vec<u8>,
    matched_subscribers: Vec<MatchedSubscriber>,
    created_at: Instant,         // expires after 30s
}

struct MatchedSubscriber {
    id: Uuid,
    blinding_scalar: Scalar,     // r_i for Ristretto rerandomization
    blinded_fingerprint: [u8; 16], // Blake3(blinded_pubkey)[..16] — 128-bit, used for capsule routing
}
```

### Subscriber Registry

```rust
struct Subscriber {
    id: Uuid,
    subject: String,             // Verified identity from EnvelopeContext (for unsubscribe auth)
    scope_pattern: Scope,        // Parsed claim scope: action:resource:identifier
    ephemeral_pubkey: Vec<u8>,   // Ristretto255 pubkey (32 bytes)
    pubkey_fingerprint: [u8; 16], // Blake3(pubkey)[..16] for capsule routing (128-bit)
    registered_topic: String,    // Pre-registered with StreamService via StreamChannel
    created_at: Instant,
    expires_at: Instant,
}

struct SubscriberRegistry {
    by_id: HashMap<Uuid, Subscriber>,
    by_fingerprint: HashMap<[u8; 16], Uuid>,
    by_scope_prefix: HashMap<String, HashSet<Uuid>>,
}
```

**Design decisions:**
- **Topic registration at subscribe time**: NS pre-registers each subscriber's XPUB topic with StreamService via `StreamChannel::register_topic(topic, expiry, claims)` during `handle_subscribe` — NOT during `handle_deliver`. This eliminates the registration-delivery race condition. StreamService buffers messages for topics where the subscriber hasn't connected yet (`!state.subscribed`), flushing on subscribe.
- **Use `StreamPublisher` API**: NS does NOT construct raw 3-frame multipart. Instead, it uses `StreamPublisher::publish_data()` which handles `StreamBlock` framing, HMAC chaining, and 3-frame wire format automatically. The `NotificationBlock` is serialized as `StreamPayload::data` inside a `StreamBlock`.
- **Single shared PUSH socket**: NS uses one `StreamChannel` (one TMQ PUSH socket), differentiating by topic prefix per message. No per-subscriber sockets.
- **Pubkey fingerprint**: `Blake3::hash(pubkey)[..16]` (128-bit) instead of raw bytes for better collision resistance (~2^64 birthday bound).
- **Fingerprint collision**: Check on insert, reject with error if collision detected.
- **TTL**: Default 600s (10 min), max 3600s (1 hour). Background task every 30s removes expired entries.
- **Scope matching for routing**: Simple `keyMatch`-style glob over `SubscriberRegistry` (separate from Casbin authorization checks). `serve:model:*` matches `serve:model:qwen3`.
- **Continuation semaphore**: `RequestLoop` limits to 16 inflight continuations. For >16 concurrent subscribers per delivery, batch sends within the continuation.

### Handler Methods — Manual Dispatch

Generated dispatch produces normal async handlers, not `(response, Continuation)` pairs. NS must implement `ZmqService::handle_request` directly (following `InferenceZmqAdapter` pattern in `services/inference.rs`) to support Continuation on deliver.

| Handler | Auth Check | Action |
|---------|-----------|--------|
| `handle_subscribe` | Casbin: `(subject, *, notify:{scope_pattern}, subscribe)` | **Await** `StreamChannel::register_topic()` to pre-register XPUB topic, create `StreamPublisher` for topic, store subscriber, return `SubscribeResponse` |
| `handle_publish_intent` | Casbin: `(subject, *, notify:{scope}, publish)` | Scope-match subscribers, **rerandomize** each pubkey (`blinded = sub_pub + r_i * G`), store `PendingIntent` with blinding scalars + publisher_subject, return blinded pubkeys |
| `handle_deliver` | **Verify `ctx.subject() == intent.publisher_subject`** | Route capsules to subscriber topics via `StreamPublisher::publish_data()`. Returns `DeliverResponse` + `Continuation` for async sends. |
| `handle_unsubscribe` | `ctx.subject() == subscriber.subject` | Remove from registry |
| `handle_list_subscriptions` | Filter by `ctx.subject()` | Return matching entries |
| `handle_ping` | None (local always allowed) | Return stats |

### Deliver Fanout (Continuation Pattern)

`handle_deliver` returns the REP response immediately, then a `Continuation` future (following `InferenceService`'s `PendingWork` pattern) performs the sends via `StreamPublisher`:

1. Verify `ctx.subject()` matches `PendingIntent.publisher_subject` (prevent intent hijacking)
2. For each `RecipientCapsule`, look up `MatchedSubscriber` by `blinded_fingerprint` → get subscriber `id` and `blinding_scalar r_i`
3. Look up subscriber by `id` → get registered `topic` and `StreamPublisher`
4. Serialize `NotificationBlock` as Cap'n Proto bytes:
   - Fields: `publisher_pubkey`, **`blinding_scalar = r_i`**, `wrapped_key`, `key_nonce`, `encrypted_payload`, `nonce`, `intentId`, `scope`, `publisherMac`
5. Call `publisher.publish_data(&notification_block_bytes).await?` — StreamPublisher handles:
   - Wrapping as `StreamPayload::data` inside `StreamBlock`
   - Computing HMAC chain (StreamService wire format requirement)
   - 3-frame multipart: `[topic, StreamBlock capnp, 16-byte HMAC]`
6. Call `publisher.complete(b"").await?` to signal end of delivery for this topic
7. StreamService PULL → XPUB → subscriber SUB

The publisher's one-shot MAC is embedded inside `NotificationBlock.publisherMac` (not frame 2). Frame 2 is the StreamService HMAC (computed by `StreamPublisher` for wire format compliance). The subscriber verifies BOTH:
- StreamService HMAC (frame 2) → proves message came through authenticated StreamService path
- Publisher's one-shot MAC (inside `NotificationBlock.publisherMac`) → proves message from DH key holder

NS includes `r_i` in the `NotificationBlock` so subscriber can perform blinding-aware DH. NS forwards encrypted payload **as-is** — never verifies or constructs MACs.

## Casbin Policy Design

Authorization is claim-based: scope is a verified Claim from SignedEnvelope, checked against Casbin 5-tuple `(sub, dom, obj, act, eft)`.

Resource format: `notify:{action}:{resource}:{identifier}` with `keyMatch`.

| Policy | Subject | Domain | Resource | Action | Effect |
|--------|---------|--------|----------|--------|--------|
| ModelService publish | `local:model-service` | `*` | `notify:serve:model:*` | `publish` | allow |
| CLI subscribe to model events | `local:*` | `*` | `notify:serve:model:*` | `subscribe` | allow |
| Operators full access | `role:operator` | `*` | `notify:*:*:*` | `*` | allow |
| Trainers subscribe to training | `role:trainer` | `*` | `notify:train:model:*` | `subscribe` | allow |
| Inference events | `local:inference-service` | `*` | `notify:infer:model:*` | `publish` | allow |

### Claim-Policy matching

- **Authorization** (binary allow/deny): Casbin `check_with_domain(subject, "*", "notify:serve:model:qwen3", "publish")`
- **Routing** (which subscribers match): Simple `keyMatch` glob over `SubscriberRegistry.scope_pattern` — NOT Casbin. Separate concern from authorization.

## E2E Crypto Design

Payload encryption is **mandatory** for blind relay. NS cannot see content.

### MAC: One-shot (not chained)

`ChainedStreamHmac` is designed for multi-block `StreamBlock` sequences (e.g., inference token streaming). Notifications are single messages with fresh ephemeral keys per publish — the chain resets each time, degenerating to a plain MAC. Use one-shot MAC instead.

```rust
// One-shot MAC (not ChainedStreamHmac):
let mac = keyed_mac(&mac_key, &ciphertext);  // 32 bytes, from crypto::backend
// Verify:
keyed_mac_verify(&mac_key, &ciphertext, &received_mac)?;  // constant-time
```

Use existing `keyed_mac()` from `crypto/backend.rs` (Blake3 keyed hash or HMAC-SHA256 in FIPS mode).

### Key derivation: `derive_notification_keys()`

New function in `key_exchange.rs`, following `derive_stream_keys()` pattern:

```rust
pub struct NotificationKeys {
    pub enc_key: Zeroizing<[u8; 32]>,  // AES-256-GCM key for data_key wrapping
    pub mac_key: Zeroizing<[u8; 32]>,  // One-shot MAC key
}

pub fn derive_notification_keys(
    shared_secret: &[u8; 32],
    publisher_pub: &[u8; 32],
    subscriber_pub: &[u8; 32],
) -> EnvelopeResult<NotificationKeys> {
    // Reject self-connection
    if is_self_connection(publisher_pub, subscriber_pub) {
        return Err(EnvelopeError::KeyExchange("identical keys".into()));
    }
    // Salt = XOR(pub_pubkey, sub_pubkey)
    let mut salt = [0u8; 32];
    for i in 0..32 { salt[i] = publisher_pub[i] ^ subscriber_pub[i]; }
    // IKM = shared_secret || salt
    let mut ikm = [0u8; 64];
    ikm[..32].copy_from_slice(shared_secret);
    ikm[32..].copy_from_slice(&salt);
    // Derive with notification-specific context strings
    let enc_key = derive_key("hyprstream notification-keys v1 enc", &ikm);
    let mac_key = derive_key("hyprstream notification-keys v1 mac", &ikm);
    Ok(NotificationKeys {
        enc_key: Zeroizing::new(enc_key),
        mac_key: Zeroizing::new(mac_key),
    })
}
```

### AES-GCM with AAD (Associated Authenticated Data)

AAD must be **length-prefixed** to prevent ambiguity (e.g., `intentId="ab", scope="cd"` vs `intentId="abc", scope="d"`). All nonces must be **random from `OsRng`**, never derived from shared secrets.

- **Payload encryption**: `AAD = len(intentId):u32_le || intentId || len(scope):u32_le || scope` — binds ciphertext to the specific publish intent
- **Key wrapping**: `AAD = pubkey_fingerprint` (16 bytes, unambiguous) — binds wrapped key to specific subscriber

```rust
fn build_payload_aad(intent_id: &str, scope: &str) -> Vec<u8> {
    let mut aad = Vec::with_capacity(8 + intent_id.len() + scope.len());
    aad.extend_from_slice(&(intent_id.len() as u32).to_le_bytes());
    aad.extend_from_slice(intent_id.as_bytes());
    aad.extend_from_slice(&(scope.len() as u32).to_le_bytes());
    aad.extend_from_slice(scope.as_bytes());
    aad
}
```

### Ristretto255 Rerandomization (subscriber unlinkability)

On each `publishIntent`, NS generates random blinding scalar `r_i` per (subscriber, intent):

```rust
// NS generates blinding per subscriber per intent
let r_i = Scalar::random(&mut rng);
let blinded_pubkey = sub_pubkey_point + r_i * RISTRETTO_BASEPOINT_POINT;
let blinded_compressed = blinded_pubkey.compress().to_bytes();
```

Publisher receives `blinded_pubkey` — unlinkable to `sub_pubkey` without `r_i`. Different intents produce different blinded pubkeys for the same subscriber.

NS includes `r_i` in the `NotificationBlock` header (cleartext — safe because `r_i` alone is useless without `sub_secret` or `pub_secret`).

**Algebraic correctness**: Publisher computes `DH(s_pub, P_sub + r*G) = s_pub * P_sub + s_pub * r * G`. Subscriber computes `(s_sub + r) * P_pub = s_sub * s_pub * G + r * s_pub * G`. These are identical by DH commutativity.

### Ed25519 Publisher Identity Attestation

Publisher signs their ephemeral DH pubkey with their persistent Ed25519 signing key, binding it to the scope, intent, and subscriber:

```rust
struct SignedNotificationPayload {
    publisher_verifying_key: [u8; 32],  // Ed25519 verifying key
    attestation: [u8; 64],              // Ed25519_sign(signing_key, msg) where
                                        // msg = ephemeral_pubkey || blinded_sub_pubkey || scope || intentId
    event: EventEnvelope,               // The actual notification data
}
```

This is **inside the encrypted payload** (inside AES-GCM ciphertext), not in cleartext. NS cannot see the publisher's persistent identity.

Subscriber verification after decryption:
1. Reconstruct `blinded_pub = sub_pubkey + r_i * G`
2. `Ed25519_verify(publisher_verifying_key, ephemeral_pubkey || blinded_pub || scope || intentId, attestation)` → proves ephemeral key was issued by this Ed25519 identity, bound to this subscriber
3. Look up `publisher_verifying_key` in trust store (via DiscoveryService) → maps to `local:model-service`
4. Trust chain: DH MAC proves message is from the DH key holder → attestation proves DH key was generated by Ed25519 identity → DiscoveryService maps Ed25519 key to service name

### Encryption flow (per publish)

1. Publisher generates random 256-bit `data_key`, random 12-byte `nonce` (**`OsRng`**, never derived)
2. For each **blinded** subscriber pubkey from `publishIntent`:
   - `shared_secret = DH(pub_ephemeral_secret, blinded_pubkey)` (Ristretto255)
   - `(enc_key, mac_key) = derive_notification_keys(shared_secret, pub_pub, blinded_pub)`
3. Publisher creates per-subscriber `SignedNotificationPayload`:
   - `attestation = Ed25519_sign(signing_key, pub_ephemeral_pubkey || blinded_pub || scope || intentId)`
   - `payload = serialize(SignedNotificationPayload { verifying_key, attestation, event })`
4. `ciphertext = AES-256-GCM(data_key, nonce, payload, aad=build_payload_aad(intentId, scope))`
5. For each subscriber:
   - `key_nonce` = random 12 bytes from **`OsRng`**
   - `wrapped_key = AES-256-GCM(enc_key, key_nonce, data_key, aad=blinded_fingerprint)`
   - `mac = keyed_mac(mac_key, ciphertext)` — **one-shot**, not chained
6. Send `DeliverRequest` with capsules (keyed by `blinded_fingerprint`) + shared ciphertext

**Note**: Steps 3-4 produce the same ciphertext for all subscribers (same `data_key`). Only the attestation is per-subscriber (bound to `blinded_pub`). If all subscribers should see identical attestation, bind only to `scope || intentId` instead. Trade-off: per-subscriber attestation = stronger binding but `N` signatures; shared attestation = 1 signature but no subscriber-specific binding.

### Decryption flow (per subscriber)

1. Receive `NotificationBlock` on XPUB topic containing `pub_pubkey`, `r_i`, `wrapped_key`, `key_nonce`, `ciphertext`, `nonce`, `intentId`, `scope`
2. Receive `mac` in frame 2 (via StreamService 3-frame wire format)
3. **Blinding-aware DH**: `shared_secret = (sub_secret + r_i) * pub_pubkey` (Ristretto scalar add + mult)
   - Equivalent to publisher's `pub_secret * (sub_pubkey + r_i * G)` — same shared secret
4. `blinded_pub = sub_pubkey + r_i * G` (reconstruct to match publisher's derivation)
5. `blinded_fingerprint = Blake3(blinded_pub.compress())[..16]` (reconstruct 128-bit fingerprint)
6. `(enc_key, mac_key) = derive_notification_keys(shared_secret, pub_pub, blinded_pub)`
7. Verify: `keyed_mac_verify(mac_key, ciphertext, mac)?`
8. `data_key = AES-256-GCM-decrypt(enc_key, key_nonce, wrapped_key, aad=blinded_fingerprint)`
9. `payload = AES-256-GCM-decrypt(data_key, nonce, ciphertext, aad=build_payload_aad(intentId, scope))`
10. Parse `SignedNotificationPayload` → verify `Ed25519(verifying_key, pub_pubkey || blinded_pub || scope || intentId, attestation)`
11. Look up `verifying_key` in trust store → confirm publisher identity

### Replay protection

Include monotonic timestamp in plaintext payload. Subscribers reject messages older than TTL. Additionally, `intentId` is included in AES-GCM AAD — replayed ciphertexts from a different intent are rejected by AES-GCM authentication.

### Dependencies

- `aes-gcm` — **needs to be added** to `hyprstream-rpc/Cargo.toml` (not currently in any Cargo.toml despite being in Cargo.lock)
- `curve25519-dalek` — already used for Ristretto255
- `blake3` / HKDF — already used for key derivation

### Performance

- AES-256-GCM: ~1-2μs per payload (AES-NI)
- DH per subscriber: ~50μs per Ristretto255 scalar mult
- For 10 subscribers: ~500μs DH + ~20μs encryption — trivial

### Not viable (assessed)

- **Homomorphic encryption** (`tfhe-rs`, `concrete`): ms-scale, wrong problem domain
- **Attribute-Based Encryption** (`rabe`): unaudited, Casbin already gates access
- **Group keys**: NS must be honest-but-curious — weaker trust model

### Zero-knowledge assessment (future work)

NS currently sees publisher/subscriber identities during Casbin authorization. True ZK would require:
- **Ring signatures**: Publisher proves "I am one of N authorized publishers" without revealing which. Feasible with Ristretto Schnorr ring sigs, but no audited Rust crate. O(N) signature size.
- **Blind credentials**: Subscriber obtains blind signature from auth server proving authorization. Complex protocol, no mature Rust implementation.
- **Private Information Retrieval**: NS queries subscriber registry without learning the query. 100ms+ per query — impractical.
- **Pseudonymous Casbin**: Scope-bound pseudonyms (`Blake3(identity || scope)`) in place of real identities. Simple but just indirection — policy admin knows mapping.
- **Recommended stepping stone**: "Honest-but-forgetful" NS — verify identity for Casbin, immediately discard, use bearer tokens for deliver/unsubscribe auth. Deferred to Phase 2.

## Event Payload Additions

**File**: `crates/hyprstream/src/events/mod.rs`

Add `EventSource::Model` variant (currently missing — only `Inference`, `Metrics`, `Training`, `Git2db`).

Add `EventPayload` variants:
```rust
ModelLoaded { model_ref: String, endpoint: String },
ModelFailed { model_ref: String, error: String },
ModelUnloaded { model_ref: String },
```

## ModelService Integration

**File**: `crates/hyprstream/src/services/model.rs`

Crypto logic does NOT belong on generated `NotificationClient`. Create a separate `NotificationPublisher` struct that wraps `NotificationClient` and handles the two-phase encrypt-then-deliver flow. Matches how `StreamChannel` wraps ZMQ sockets with DH + signing.

```rust
/// Publisher-side helper for encrypted notification delivery.
/// Wraps NotificationClient with DH + AES-GCM broadcast encryption.
pub struct NotificationPublisher {
    client: NotificationClient,
    signing_key: SigningKey,
}

impl NotificationPublisher {
    /// Two-phase encrypted publish:
    /// 1. publishIntent → get subscriber pubkeys
    /// 2. Generate data_key, encrypt payload, wrap key per subscriber
    /// 3. deliver → NS routes capsules to subscriber topics
    pub async fn publish(&self, scope: &Scope, payload: &[u8]) -> Result<u32> { ... }
}
```

ModelService holds `notification_publisher: Option<NotificationPublisher>`:

```rust
// After successful load (~line 238):
if let Some(ref notif) = self.notification_publisher {
    let event = EventEnvelope::new(
        EventSource::Model, "model.loaded",
        EventPayload::ModelLoaded { model_ref, endpoint });
    let scope = Scope::new("serve", "model", &model_name);
    let _ = notif.publish(&scope, &serde_json::to_vec(&event)?).await;
}
```

## CLI Integration

**File**: `crates/hyprstream/src/cli/git_handlers.rs` (and CLI modules)

### `hyprstream model load <ref> --wait`

1. Generate ephemeral Ristretto255 keypair `(sub_secret, sub_pubkey)`
2. `NotificationClient::subscribe("serve:model:*", sub_pubkey, 120)` → `SubscribeResponse { subscriptionId, assignedTopic, streamEndpoint }`
3. Connect TMQ SUB to `streamEndpoint`, subscribe on `assignedTopic`
4. Call `ModelZmqClient::load(model_ref)` (fire and forget — don't wait for 30s timeout)
5. Receive on SUB → for each 3-frame message `[topic, StreamBlock, hmac]`:
   - Verify StreamService HMAC (frame 2) via `StreamVerifier`
   - Extract `StreamPayload::data` from `StreamBlock` → parse as `NotificationBlock`
   - Use `NotificationSubscriber` (hand-written wrapper) to decrypt:
     - Extract `r_i` (blinding scalar) and `pub_pubkey` from `NotificationBlock`
     - Blinding-aware DH: `shared_secret = (sub_secret + r_i) * pub_pubkey`
     - Reconstruct `blinded_pub = sub_pubkey + r_i * G`
     - `derive_notification_keys(shared_secret, pub_pub, blinded_pub)` → `(enc_key, mac_key)`
     - Verify publisher one-shot MAC: `keyed_mac_verify(mac_key, ciphertext, publisherMac)`
     - Decrypt capsule: `AES-GCM(enc_key, key_nonce, wrapped_key, aad=blinded_fingerprint)` → `data_key`
     - Decrypt payload: `AES-GCM(data_key, nonce, ciphertext, aad=build_payload_aad(intentId, scope))`
     - Verify Ed25519 attestation inside decrypted payload
   - Parse `EventEnvelope` → check if `model.loaded` for our `model_ref`
6. On match / failure / timeout(120s): print result, `unsubscribe(subscriptionId)`

### `hyprstream notify subscribe <pattern> [--json] [--timeout N]`

Standalone command. Same decrypt flow, prints all matching events.

### `hyprstream quick infer <model> <prompt>`

Subscribe to `serve:model:{model}` before `load_model()`. If already loaded, proceed immediately. If loading, wait for notification. Then inference.

## RPC Codegen Integration

### Annotations (`annotations.capnp`)

Extend `ScopeAction` enum:
```capnp
enum ScopeAction {
  query @0; write @1; manage @2; infer @3; train @4; serve @5; context @6;
  subscribe @7;   # NEW: notification subscriptions
  publish @8;     # NEW: notification publishing
}
```

CGR reader (`cgr_reader.rs`) discovers annotations by name via `find_annotation_id()` — no code changes needed. The new enum variants are forward-compatible.

### Generated Code (Rust)

`generate_rpc_service!` for `notification.capnp` produces:
- `NotificationClient` — typed methods for each request variant (subscribe, publishIntent, deliver, etc.)
- `NotificationHandler` trait — async handler methods
- `dispatch_notification()` — request routing
- `NotificationResponseVariant` — response union enum
- `schema_metadata()` — method info for MCP/CLI

**Hand-written wrappers** (NOT codegen'd):
- `NotificationPublisher` — wraps `NotificationClient` + DH + AES-GCM broadcast encryption
- `NotificationSubscriber` — wraps SUB socket + DH + decrypt + verify attestation

Rationale: Method-level crypto wrapping can't be auto-generated without significant macro complexity. The scoped client pattern in `codegen/scoped.rs` is the closest existing pattern but doesn't support conditional crypto. Follow `InferenceZmqAdapter` / `StreamChannel` pattern for hand-written wrappers.

### Generated Code (TypeScript)

`hyprstream-ts-codegen` produces:
- `notification.ts` — interfaces, builders, parsers, `NotificationClient` class
- All crypto delegates to WASM API (`wasm_api.rs`)

**Hand-written TS wrappers**:
- `NotificationCryptoClient` — calls WASM for `derive_notification_keys()`, `aes_gcm_encrypt()`, `ristretto_blinded_dh()`, etc.
- `NotificationCryptoSubscriber` — decrypts received `NotificationBlock` via WASM

### WASM API Additions (`wasm_api.rs`)

New exports alongside existing `ecdh_ristretto()`, `derive_stream_keys()`:
- `derive_notification_keys(shared_secret, pub_pub, sub_pub)` → `{ enc_key, mac_key }`
- `aes_gcm_encrypt(key, nonce, plaintext, aad)` → ciphertext
- `aes_gcm_decrypt(key, nonce, ciphertext, aad)` → plaintext
- `ristretto_rerandomize(pubkey, blinding_scalar)` → blinded pubkey
- `ristretto_blinded_dh(secret, blinding_scalar, other_pubkey)` → shared secret
- `keyed_mac_oneshot(key, data)` → 32-byte MAC
- `keyed_mac_verify(key, data, mac)` → bool
- `ed25519_sign_attestation(signing_key, ephemeral_pub, blinded_pub, scope, intent_id)` → signature
- `ed25519_verify_attestation(verifying_key, ephemeral_pub, blinded_pub, scope, intent_id, sig)` → bool
- `build_payload_aad(intent_id, scope)` → length-prefixed AAD bytes

## Files Modified

| File | Action | Description |
|------|--------|-------------|
| `crates/hyprstream-rpc/schema/annotations.capnp` | **Modify** | Add `subscribe @7;` and `publish @8;` to `ScopeAction` enum |
| `crates/hyprstream-rpc/src/crypto/key_exchange.rs` | **Modify** | Add `NotificationKeys`, `derive_notification_keys()`, `rerandomize_pubkey()`, `blinded_dh()` |
| `crates/hyprstream-rpc/src/crypto/notification.rs` | **Create** | `BroadcastEncryptor`/`BroadcastDecryptor` (AES-GCM, key wrap, one-shot MAC, attestation) |
| `crates/hyprstream-rpc/src/crypto/mod.rs` | **Modify** | Add `pub mod notification;` export |
| `crates/hyprstream-rpc/src/wasm_api.rs` | **Modify** | Add notification crypto WASM exports |
| `crates/hyprstream-rpc/Cargo.toml` | **Modify** | Add `aes-gcm` dependency |
| `crates/hyprstream/schema/notification.capnp` | **Create** | Schema with two-phase publish + broadcast encryption |
| `crates/hyprstream/src/services/notification.rs` | **Create** | Blind relay NS with manual ZmqService dispatch (InferenceZmqAdapter pattern) |
| `crates/hyprstream/src/services/mod.rs` | **Modify** | Add `pub mod notification;` |
| `crates/hyprstream/src/services/factories.rs` | **Modify** | Add `#[service_factory("notification")]` |
| `crates/hyprstream/src/services/model.rs` | **Modify** | Add `NotificationPublisher`, publish events on load/unload |
| `crates/hyprstream/src/events/mod.rs` | **Modify** | Add `EventSource::Model`, `EventPayload::ModelLoaded/Failed/Unloaded` |
| `crates/hyprstream/schema/events.capnp` | **Modify** | Add `model` to `EventSource` enum, model event variants |
| `crates/hyprstream/src/cli/git_handlers.rs` | **Modify** | `--wait` flag, `notify subscribe`, DH + decrypt on receive |

## Implementation Sequence

1. **Annotations**: Add `subscribe @7;` and `publish @8;` to `ScopeAction` in `annotations.capnp`. Rebuild to verify CGR reader picks them up.
2. **Crypto primitives**: `derive_notification_keys()` + `rerandomize_pubkey()` + `blinded_dh()` in `key_exchange.rs`. Add `aes-gcm` dep. Create `crypto/notification.rs` with `BroadcastEncryptor`/`BroadcastDecryptor` (AES-GCM with length-prefixed AAD, random `OsRng` nonces, one-shot MAC, Ed25519 attestation with subscriber binding).
3. **Schema**: `notification.capnp` (with `publisherMac` field in `NotificationBlock`) + capnpc build integration. Verify compilation.
4. **Service scaffolding**: `NotificationService` struct, manual `ZmqService::handle_request` dispatch (following `InferenceZmqAdapter`), factory registration.
5. **Subscriber registry**: Data structure, 128-bit `Blake3` fingerprinting, TTL expiry, topic pre-registration via `StreamChannel::register_topic()` at subscribe time (not deliver time), subscribe/unsubscribe handlers.
6. **Two-phase publish**: `publishIntent` (rerandomize pubkeys, store PendingIntent with publisher_subject) + `deliver` (verify caller identity, route capsules via `StreamPublisher::publish_data()` in Continuation).
7. **Client wrappers** (hand-written): `NotificationPublisher` (DH + AES-GCM broadcast encryption), `NotificationSubscriber` (DH + decrypt + verify attestation).
8. **WASM API**: Export notification crypto functions for browser/TypeScript clients.
9. **Event integration**: `EventSource::Model`, `EventPayload` variants, ModelService integration with `NotificationPublisher`.
10. **CLI**: `--wait` flag on `model load`, `notify subscribe` command, DH + decrypt on receive.

## Verification

1. `cargo build --release` — compiles (capnpc + generate_rpc_service! succeeds)
2. `cargo clippy --workspace --all-targets` — zero warnings
3. Unit test: `derive_notification_keys()` — symmetric derivation from both sides of DH, self-connection rejected
4. Unit test: `BroadcastEncryptor`/`BroadcastDecryptor` — encrypt → wrap per subscriber → unwrap → decrypt roundtrip
5. Unit test: One-shot MAC verify succeeds, tampered ciphertext rejected
6. Unit test: AES-GCM AAD binding — wrong intentId/fingerprint rejects decryption
7. Unit test: **Length-prefixed AAD** — different `intentId`/`scope` splits with same concatenation produce different AAD bytes
8. Unit test: **Ristretto rerandomization** — blinded DH produces same shared secret as subscriber's `(secret + r_i) * pub_pubkey`
9. Unit test: **Unlinkability** — two different `r_i` values for same subscriber produce different blinded pubkeys
10. Unit test: **Ed25519 attestation** — valid attestation passes, wrong pubkey/scope/intentId/blinded_pub rejected
11. Unit test: Scope matching (`serve:model:*` matches `serve:model:qwen3`, not `train:model:x`)
12. Unit test: Subscriber registry insert/lookup/TTL expiry/128-bit fingerprint collision rejection
13. Unit test: Intent hijacking rejected — deliver from different subject than intent creator
14. Unit test: Casbin deny on subscribe/publish without policy
15. Unit test: **128-bit fingerprint** — `Blake3(pubkey)[..16]` produces distinct fingerprints for similar pubkeys
16. Integration test: subscribe (topic pre-registered) → publishIntent (blinded pubkeys returned) → deliver (via StreamPublisher) → receive + decrypt + verify attestation on SUB socket
17. Integration test: **Subscription race** — deliver before subscriber connects → StreamService buffers → subscriber receives on connect
18. Manual: `hyprstream model load qwen3:main --wait` — prints success on load completion
19. Manual: `hyprstream notify subscribe "serve:model:*" --json` — prints decrypted events with verified publisher identity
20. WASM: `wasm-pack test --node` — notification crypto roundtrip in WASM environment

## Findings Addressed

### Round 1 (Initial Panel)

| Finding | Severity | Resolution |
|---------|----------|------------|
| Intent hijacking: deliver doesn't verify caller | CRITICAL | Store `publisher_subject` in `PendingIntent`, verify on deliver |
| HMAC chain pointless for single-message notifications | CRITICAL | Use one-shot `keyed_mac()` instead of `ChainedStreamHmac` |
| Cross-publisher subscriber linkability | HIGH | Ristretto255 rerandomization: `blinded = sub_pub + r_i * G` per intent |
| Publisher identity not verifiable by subscriber | HIGH | Ed25519 attestation of ephemeral pubkey, encrypted inside payload |
| NS zero-knowledge of identities | ASSESSED | Documented for future: ring sigs, blind credentials. Stepping stone: honest-but-forgetful NS |
| `$mcpScope(subscribe)` not a valid ScopeAction | BLOCKER | Add `subscribe @7;` and `publish @8;` to `ScopeAction` enum |
| StreamService rejects unregistered random topics | BLOCKER | Pre-register via `StreamChannel::register_topic()` |
| Sync `zmq::Socket` blocks Tokio | BLOCKER | Use shared `StreamChannel` with TMQ async PUSH |
| Per-subscriber PUSH sockets waste resources | DESIGN | Single shared PUSH via `StreamChannel` |
| Codegen won't produce Continuation handlers | DESIGN | Manual `ZmqService::handle_request` dispatch |
| AES-GCM without AAD | MEDIUM | Add AAD: `intentId\|\|scope` for payload, `blinded_fingerprint` for key wrap |
| No replay protection | MEDIUM | Timestamp in payload + intentId in AAD |
| Salt undefined for key derivation | MEDIUM | `XOR(publisher_pub, blinded_sub_pub)` explicit |
| Pubkey fingerprint from raw bytes | LOW | `Blake3::hash(blinded_pubkey)[..8]` |
| `aes-gcm` not in Cargo.toml | MEDIUM | Add to `hyprstream-rpc/Cargo.toml` |
| `EventEnvelope::new` takes enum not string | LOW | Use `EventSource::Model` (add variant) |
| Crypto on generated client breaks pattern | MEDIUM | Separate `NotificationPublisher` struct |
| Unused StreamInfo import | LOW | Removed |

### Round 2 (Crypto, Codegen, ZMQ Panel)

| Finding | Severity | Resolution |
|---------|----------|------------|
| **AAD ambiguity**: raw concat of `intentId\|\|scope` | MEDIUM | Length-prefix both fields: `u32_le(len) \|\| bytes` |
| **Fingerprint too small**: 8 bytes = 2^32 birthday bound | MEDIUM | Upgraded to 16 bytes (128-bit, 2^64 birthday bound) |
| **Attestation not subscriber-bound** | MEDIUM | Include `blinded_pub` in Ed25519 signed message |
| **AES-GCM nonce must be random** | HIGH | All nonces from `OsRng`, never derived from shared secrets |
| **Topic registration race**: register on deliver → messages lost | CRITICAL | Register topic on `handle_subscribe` (not `handle_deliver`) |
| **Wire format**: NS must use StreamBlock, not raw multipart | HIGH | Use `StreamPublisher::publish_data()` API, wraps `NotificationBlock` as `StreamPayload::data` |
| **No EventService exists**: only StreamService has buffering | HIGH | Confirmed: must use StreamService PULL→XPUB |
| **MAC field missing from NotificationBlock** | HIGH | Added `publisherMac @8 :Data` to `NotificationBlock` schema |
| **Continuation semaphore**: max 16 inflight | MEDIUM | Batch sends within continuation for >16 subscribers |
| **CryptoClient should be hand-written** | HIGH | Don't codegen — hand-write `NotificationPublisher`/`NotificationSubscriber` |
| **TS codegen lacks annotation access** | MEDIUM | Extend `ParsedSchema` → `UnionVariant` with crypto flags for future |
| **Rerandomization algebra**: crypto panel flagged as broken | FALSE POSITIVE | Math is correct: `s*(P+rG) = (s+r)*Q` via DH commutativity. Panel confused subscriber not knowing `s` (publisher secret) with inability to compute shared secret — subscriber uses own `s_sub + r` |
| **No delivery ACK from StreamService** | MEDIUM | `register_topic()` is fire-and-forget PUSH; acceptable because topic is registered well before delivery (at subscribe time) |

### Round 3 (Implementation Code Review)

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | **Attestation message lacks length prefixes**: `build_attestation_message` raw-concatenates variable-length `scope` and `intentId` — same ambiguity as payload AAD | MEDIUM | Length-prefix `scope` and `intentId` in attestation message (fixed-length pubkeys are fine) |
| 2 | **Manual zeroing in `BroadcastDecryptor::Drop`**: `iter_mut().for_each(\|b\| *b = 0)` may be optimized away by compiler | MEDIUM | Use `Zeroize` trait from `zeroize` crate (already imported) |
| 3 | **`data_key` not zeroized in `decrypt()`**: Unwrapped 32-byte `data_key` stays on stack without zeroization | MEDIUM | Wrap in `Zeroizing<[u8; 32]>` |
| 4 | **`match_scope` ignores `by_scope_prefix` index**: Linear scan over all subscribers instead of using the prefix index | LOW | Use `by_scope_prefix` for initial candidate set, then filter |
| 5 | **`scope_matches` trailing-colon edge case**: `"serve:model:"` with trailing colon and no wildcard doesn't match correctly | LOW | Add test, verify behavior (turns out correct — exact match handles it) |
| 6 | **`handle_list_subscriptions` relative timestamps**: Uses `elapsed()` and `duration_since()` which are relative to `Instant::now()` — not epoch timestamps | MEDIUM | Store `SystemTime` alongside `Instant` for serializable timestamps |
| 7 | **`started_at` field unused**: `NotificationService::started_at` set but never read after `uptime` was removed | LOW | Remove field |
| 8 | **Duplicate DH implementations**: `BroadcastEncryptor::ristretto_dh_raw` and `BroadcastDecryptor::blinded_dh_raw` duplicate logic from `key_exchange.rs` | MEDIUM | Add raw-bytes wrappers in `key_exchange.rs`, reuse from `notification.rs` |
| 9 | **CRITICAL: `handle_deliver` removes intent before auth**: Intent consumed from map before verifying caller identity — attacker can destroy valid intents | CRITICAL | Verify caller identity BEFORE removing from map; only remove on success |
| 10 | **`ikm` not zeroized in `derive_notification_keys`**: 64-byte `ikm` containing `shared_secret` stays on stack | MEDIUM | Zeroize `ikm` before return |
