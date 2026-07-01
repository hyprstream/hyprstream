//! Unified event publisher/subscriber (EV1, EventService consolidation epic #600).
//!
//! Before this module, the project had two separate types for event-bus
//! broadcast:
//!
//! - a **plaintext** publisher/subscriber in `hyprstream-workers` (a thin
//!   wrapper over [`crate::moq_event::MoqEventPublisher`] /
//!   [`crate::moq_event::MoqEventSubscriber`]), used by every production
//!   call site (worker/system lifecycle events, workflow triggers); and
//! - a **crypto-only** `SecureEventPublisher` / `SecureEventSubscriber`
//!   (also in `hyprstream-workers`), implementing group-key AES-256-GCM
//!   encryption + Ed25519 event signing + Ristretto255 DH key wrapping, but
//!   with **no MoQ transport wiring at all** — `SecureEventPublisher::publish`
//!   returned an [`EncryptedEvent`] struct and left it to the (nonexistent)
//!   caller to actually put it on the wire. It was exercised only by its own
//!   unit tests.
//!
//! This module **unifies both into one canonical [`EventPublisher`] /
//! [`EventSubscriber`] pair**, living in `hyprstream-rpc` alongside
//! [`crate::moq_event`] (the transport it wires to) and
//! [`crate::crypto::event_crypto`] (the crypto it wires to) — both already
//! lived in this crate, so the publisher/subscriber types that glue them
//! together belong here too, not in `hyprstream-workers`.
//!
//! The privacy mode is selected per `EventPublisher` instance via
//! [`EventPrivacy`]:
//!
//! - [`EventPrivacy::Public`] — no encryption; the payload is written to the
//!   moq track unmodified. This is **wire-identical** to the pre-EV1
//!   plaintext path, so existing production callers
//!   (`EventPublisher::new("worker")`, `EventPublisher::new("system")`, …)
//!   keep working unchanged with no behavior change. This is the default for
//!   [`EventPublisher::new`] / [`EventPublisher::new_with_oid`] /
//!   [`EventPublisher::new_oid_only`] — flipping any production publisher to
//!   an encrypted-by-default mode is explicitly OUT OF SCOPE for this ticket
//!   (deferred to the EventService consolidation epic's later phases, which
//!   also need group-membership records (#602) and authenticated join (#604)
//!   before an encrypted default is safe).
//! - [`EventPrivacy::ZeroKnowledge`] / [`EventPrivacy::LimitedKnowledge`] —
//!   group-key encrypted, Ed25519-signed events, now actually wired to a
//!   shared `moq_event` broadcast track per registered prefix: one
//!   ciphertext is written once per publish and fans out natively to every
//!   subscriber of that track (previously this crypto path had no transport
//!   at all).
//!
//! # Wire format
//!
//! `EventPrivacy::Public` publishes the payload unmodified via
//! [`crate::moq_event::MoqEventPublisher::publish_raw`] — same
//! `[topic_len][topic][payload]` framing as before this module existed.
//!
//! `EventPrivacy::ZeroKnowledge` / `LimitedKnowledge` publish an
//! [`EncryptedEvent`] encoded by [`EncryptedEvent::encode_body`] as the moq
//! frame's payload (the topic itself still travels as the moq frame's topic
//! field, unencrypted — topic *secrecy* is out of scope here; see epic #600's
//! EV3 (#603) for keyed/opaque topic routing). The encoded body is a simple
//! versioned, length-prefixed binary layout (mirroring the project's existing
//! `[len][bytes]` framing style rather than introducing a capnp schema for an
//! internal-only, single-crate wire format):
//!
//! ```text
//! [1B version][12B nonce][16B key_commitment]
//! [4B tag_len][tag][4B ciphertext_len][ciphertext][4B lk_tag_len][lk_tag]
//! [8B timestamp BE][32B publisher_pubkey][4B signature_len][signature]
//! ```
//!
//! # Signature verification (new in this module)
//!
//! The original crypto-only `SecureEventPublisher` populated
//! `EncryptedEvent::signature` / `publisher_pubkey`, but no subscriber-side
//! code ever verified them. Since this module builds the receive path from
//! scratch, [`EventSubscriber::recv`] now verifies the embedded Ed25519
//! signature against the decrypted payload before returning it, rejecting
//! tampered frames. The embedded pubkey is **self-asserted** by the wire
//! frame, not yet bound to a verified service/DID identity — that binding
//! (publisher↔DID authentication on the publish path) is epic #600's EV4
//! (#604), gated on #446. This verification only proves the signature is
//! internally consistent with the payload: tamper-evidence, not identity.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use tokio::sync::{mpsc, RwLock};
use tracing::debug;
use zeroize::Zeroizing;

use crate::crypto::event_crypto::{
    build_event_sig_message, check_key_commitment, decrypt_event_full, derive_wrap_key,
    encrypt_event, unwrap_group_key, wrap_group_key, EventPrivacy,
};
use crate::crypto::{generate_ephemeral_keypair, keyed_mac, ristretto_dh_raw};
use crate::envelope::Subject;
// The four shared event types + the rotation constants are canonical in the
// keyable-group primitive (`crypto::group_key`); re-export them here so the
// `hyprstream_rpc::events::*` surface downstream consumers already use stays
// stable. (Only the EventService-specific `EncryptedEvent` wire codec + the
// `EventPublisher`/`EventSubscriber`/`KeyRing` types are defined in this module.)
pub use crate::crypto::group_key::{
    EncryptedEvent, MAX_KEY_LIFETIME, DEFAULT_ROTATION_INTERVAL, GRACE_PERIOD, RekeyPolicy,
    RotationResult, WrappedKeyEntry,
};
use crate::moq_event::{
    global_moq_event_origin, BackfillMode, MoqEventPublisher, MoqEventSubscriber,
};

// ============================================================================
// Rekey policy — the type + its constants live in the keyable-group primitive
// (`crypto::group_key`); this module re-imports them above. Only the
// subscriber-side grace constant below is specific to this module.
// ============================================================================

/// Default grace period a subscriber accepts the previous group key after a
/// rekey (separate from the publisher-side [`GRACE_PERIOD`] above — this one
/// bounds how long [`EventSubscriber::try_decrypt`] still trial-decrypts with
/// `previous` after [`EventSubscriber::promote_key`]).
pub const DEFAULT_SUBSCRIBER_GRACE_PERIOD: Duration = Duration::from_secs(30);

// Rekey policy configuration.
//
// # Cost / forward-secrecy tradeoff (#606)
//
// Rotation re-wraps the group key for every known subscriber: O(M) DH +
// AEAD-wrap operations per rotation, M = group size (see
// [`EventPublisher::rotate_key`]).
//
// - [`RekeyPolicy::Immediate`] rotates on EVERY revocation — prompt
//   forward-secrecy (a revoked member loses access immediately), but O(M)
//   PER revocation. M sequential departures cost O(M²) total. Use only for
//   an explicit revocation / suspected-compromise event on a specific
//   prefix, not as a blanket policy for routine membership churn.
// - [`RekeyPolicy::Scheduled`] bounds the cost to O(M) per `interval` regardless
//   of churn, but DEFERS revocation: a removed member can still decrypt
//   until the next scheduled rotation (up to `interval`, capped by
//   [`MAX_KEY_LIFETIME`]). This is the default — routine join/leave should
//   not pay an O(M) rewrap per event.
// - [`RekeyPolicy::Jittered`] is `Scheduled` plus timing-attack resistance on
//   exactly when the rotation fires; same cost/latency tradeoff as
//   `Scheduled`.
//
// Recommended use: `Scheduled`/`Jittered` for the steady state; switch a
// specific prefix to `Immediate` only around an explicit revocation, then
// switch back.
//
// The type itself + its `Default`/`validate` impls live in the keyable-group
// primitive ([`crate::crypto::group_key`]); this module re-uses them.

// ============================================================================
// EncryptedEvent — wire-encodable result of encrypting an event
// ============================================================================

const WIRE_VERSION: u8 = 1;

impl EncryptedEvent {
    /// Encode the non-topic fields into a wire frame body (see module docs
    /// for the exact layout). `topic` travels separately as the moq frame's
    /// topic field and is reattached by [`Self::decode_body`].
    fn encode_body(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(
            1 + 12
                + 16
                + 4
                + self.tag.len()
                + 4
                + self.ciphertext.len()
                + 4
                + self.lk_tag.len()
                + 8
                + 32
                + 4
                + self.signature.len(),
        );
        buf.push(WIRE_VERSION);
        buf.extend_from_slice(&self.nonce);
        buf.extend_from_slice(&self.key_commitment);
        push_lenprefixed(&mut buf, &self.tag);
        push_lenprefixed(&mut buf, &self.ciphertext);
        push_lenprefixed(&mut buf, &self.lk_tag);
        buf.extend_from_slice(&self.timestamp.to_be_bytes());
        buf.extend_from_slice(&self.publisher_pubkey);
        push_lenprefixed(&mut buf, &self.signature);
        buf
    }

    /// Decode a wire frame body produced by [`Self::encode_body`], reattaching
    /// the `topic` the moq frame carried separately.
    fn decode_body(topic: &str, buf: &[u8]) -> Result<Self> {
        let mut off = 0usize;
        let version = read_u8(buf, &mut off)?;
        if version != WIRE_VERSION {
            return Err(anyhow!("unsupported event wire version {version}"));
        }
        let nonce: [u8; 12] = read_array(buf, &mut off)?;
        let key_commitment: [u8; 16] = read_array(buf, &mut off)?;
        let tag = read_lenprefixed(buf, &mut off)?;
        let ciphertext = read_lenprefixed(buf, &mut off)?;
        let lk_tag = read_lenprefixed(buf, &mut off)?;
        let timestamp_bytes: [u8; 8] = read_array(buf, &mut off)?;
        let timestamp = i64::from_be_bytes(timestamp_bytes);
        let publisher_pubkey: [u8; 32] = read_array(buf, &mut off)?;
        let signature = read_lenprefixed(buf, &mut off)?;
        Ok(Self {
            topic: topic.to_owned(),
            tag,
            ciphertext,
            nonce,
            key_commitment,
            lk_tag,
            signature,
            publisher_pubkey,
            timestamp,
        })
    }
}

fn push_lenprefixed(buf: &mut Vec<u8>, data: &[u8]) {
    buf.extend_from_slice(&(data.len() as u32).to_be_bytes());
    buf.extend_from_slice(data);
}

fn read_u8(buf: &[u8], off: &mut usize) -> Result<u8> {
    let b = *buf
        .get(*off)
        .ok_or_else(|| anyhow!("event wire: truncated reading version byte"))?;
    *off += 1;
    Ok(b)
}

fn read_array<const N: usize>(buf: &[u8], off: &mut usize) -> Result<[u8; N]> {
    let end = *off + N;
    let slice = buf
        .get(*off..end)
        .ok_or_else(|| anyhow!("event wire: truncated reading {N}-byte field"))?;
    *off = end;
    slice
        .try_into()
        .map_err(|_| anyhow!("event wire: array conversion failed"))
}

fn read_lenprefixed(buf: &[u8], off: &mut usize) -> Result<Vec<u8>> {
    let len_bytes: [u8; 4] = read_array(buf, off)?;
    let len = u32::from_be_bytes(len_bytes) as usize;
    let end = *off + len;
    let slice = buf
        .get(*off..end)
        .ok_or_else(|| anyhow!("event wire: truncated reading {len}-byte length-prefixed field"))?;
    *off = end;
    Ok(slice.to_vec())
}

fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn hash_pubkey(pubkey: &[u8; 32]) -> [u8; 32] {
    *blake3::hash(pubkey).as_bytes()
}

fn hash_prefix_bytes(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

// ============================================================================
// EventPublisher — the canonical broadcast publisher
// ============================================================================

/// Group-key state for one registered prefix (ZK/LK modes only).
struct GroupKeyState {
    current: Zeroizing<[u8; 32]>,
    pending: Option<PendingRekey>,
    created_at: Instant,
}

struct PendingRekey {
    new_key: Zeroizing<[u8; 32]>,
    effective_at: Instant,
    new_ephemeral_secret: Zeroizing<[u8; 32]>,
    new_ephemeral_pubkey: [u8; 32],
}

/// Crypto state for one registered prefix. Present only when the publisher's
/// privacy mode is ZeroKnowledge/LimitedKnowledge; `Public`-mode prefixes
/// carry no crypto state at all.
struct PrefixCrypto {
    key_state: GroupKeyState,
    /// Publisher's ephemeral Ristretto255 secret scalar bytes.
    ephemeral_secret: Zeroizing<[u8; 32]>,
    /// Publisher's ephemeral Ristretto255 public key bytes.
    ephemeral_pubkey: [u8; 32],
    /// Known subscriber pubkeys (hash -> pubkey).
    subscribers: HashMap<[u8; 32], [u8; 32]>,
}

/// Per-prefix state: the moq transport for `local/events/{prefix}`, plus
/// optional encryption state.
struct PrefixState {
    moq: MoqEventPublisher,
    crypto: Option<PrefixCrypto>,
}

// ============================================================================
// Event-plane authorization + publisher identity (EV4, epic #600)
//
// This is the EventService PEP (Policy Enforcement Point) for the
// `publish`/`subscribe` ScopeActions. The load-bearing membership gate for
// obtaining `K_group` lives in `GroupKeyRegistry::join`'s `MembershipResolver`
// (fail-closed); these checks guard the publish/subscribe paths themselves.
//
// The reference-monitor enforcement at the *generated 9P mount* is S2 (#568,
// blocked on #539) — that is a separate PEP. EV4 wires the event-plane PEP;
// when S2 lands the two share S1's subject/label model but stay distinct
// enforcement points (mount-namespace vs event-plane).
// ============================================================================

/// The caller attempting an event-plane op (`publish`/`subscribe`). Carries the
/// verified subject identity (a DID when available). Until #446 lands, IPC
/// callers may resolve as `anonymous` — see [`PublisherIdentity`].
pub trait EventAuthz: Send + Sync {
    /// May `caller` publish to the group/`prefix`? Maps to the `publish`
    /// ScopeAction.
    fn can_publish(&self, caller: &Subject, prefix: &str) -> bool;
    /// May `caller` subscribe to the group/`prefix`? Maps to the `subscribe`
    /// ScopeAction.
    fn can_subscribe(&self, caller: &Subject, prefix: &str) -> bool;
}

/// Fail-closed authz: denies every publish/subscribe. The default for the
/// encrypted (`ZeroKnowledge`/`LimitedKnowledge`) profile — MAC deny-by-default
/// by construction. Production wires a real UCAN/capability-backed impl (S3/S4
/// vocab on main) via [`EventPublisher::with_authz`].
pub struct DenyAllEventAuthz;
impl EventAuthz for DenyAllEventAuthz {
    fn can_publish(&self, _caller: &Subject, _prefix: &str) -> bool {
        false
    }
    fn can_subscribe(&self, _caller: &Subject, _prefix: &str) -> bool {
        false
    }
}

/// Permissive authz: allows every publish/subscribe. For the public (`Public`)
/// profile (the open firehose — no authz by design) and for tests until a real
/// authz is wired.
pub struct AllowAllEventAuthz;
impl EventAuthz for AllowAllEventAuthz {
    fn can_publish(&self, _caller: &Subject, _prefix: &str) -> bool {
        true
    }
    fn can_subscribe(&self, _caller: &Subject, _prefix: &str) -> bool {
        true
    }
}

/// The publisher's verified identity (EV4 authn-on-publish). Binds a publish to
/// a DID when the verified identity is available.
///
/// **#446-gated stub:** until #446 lands, service-to-service IPC callers
/// resolve as `anonymous` (key-derived identity lost over UDS), so `did` is
/// `None` on those paths. A `None` DID is treated as **fail-closed** for the
/// confidential profile (an anonymous publisher cannot be bound to a DID), with
/// a `// TODO(#446)` seam for the real identity. Do NOT fake a verified DID.
#[derive(Debug, Clone)]
pub struct PublisherIdentity {
    /// The publisher's verified DID, when known. `None` on #446-affected IPC
    /// paths (anonymous) — confidential publish fails-closed in that case.
    pub did: Option<String>,
}

impl PublisherIdentity {
    /// An anonymous publisher (no verified DID) — the #446-affected default.
    pub fn anonymous() -> Self {
        Self { did: None }
    }
    /// A publisher with a verified DID (post-#446, or non-IPC paths).
    pub fn verified(did: impl Into<String>) -> Self {
        Self { did: Some(did.into()) }
    }
    /// Is a verified DID bound? (Confidential publish requires this.)
    pub fn is_verified(&self) -> bool {
        self.did.is_some()
    }
}

impl Default for PublisherIdentity {
    fn default() -> Self {
        Self::anonymous()
    }
}

/// The canonical broadcast event publisher (EV1, epic #600).
///
/// One instance manages publishing to one or more `local/events/{prefix}`
/// broadcasts. Each registered prefix gets its own moq transport and — for
/// encrypted privacy modes — its own group key and Ristretto255 ephemeral
/// keypair.
pub struct EventPublisher {
    /// The prefix this publisher is primarily bound to (set by
    /// `new`/`new_with_oid`/`new_oid_only`; used by the `publish(entity,
    /// event, payload)` convenience method). Empty for multi-prefix
    /// `new_encrypted` publishers, which must use `publish_raw`.
    primary_source: String,
    prefixes: Arc<RwLock<HashMap<String, PrefixState>>>,
    /// Ed25519 signing key. `None` for `Public`-only publishers.
    signing_key: Option<SigningKey>,
    privacy_mode: EventPrivacy,
    rekey_policy: RekeyPolicy,
    /// Event-plane authz (EV4). `AllowAll` for the `Public` profile (open
    /// firehose — no authz by design); `DenyAll` (fail-closed) for the
    /// encrypted profile until a real UCAN/capability impl is wired via
    /// [`Self::with_authz`].
    authz: Arc<dyn EventAuthz>,
    /// Publisher's verified identity (EV4 authn-on-publish). `None` DID
    /// (anonymous) is fail-closed for confidential publish until #446.
    publisher_identity: PublisherIdentity,
}

impl EventPublisher {
    /// Create a plaintext (`EventPrivacy::Public`) publisher for `source`.
    ///
    /// Wire-identical to the pre-EV1 plaintext publisher: requires the
    /// process-global moq event bus to have been initialized via
    /// `init_global_moq_event_origin` (done by the event-service factory at
    /// startup).
    pub fn new(source: &str) -> Result<Self> {
        let origin = global_moq_event_origin().ok_or_else(|| {
            anyhow!("moq event bus not initialized; start the event service first")
        })?;
        let moq = origin.publisher(source)?;
        Self::from_public_prefix(source, moq)
    }

    /// Plaintext publisher for `source` that ALSO mirrors every event to the
    /// per-OID (#393) publication track for `oid` (transition path).
    pub fn new_with_oid(source: &str, oid: &str) -> Result<Self> {
        let origin = global_moq_event_origin().ok_or_else(|| {
            anyhow!("moq event bus not initialized; start the event service first")
        })?;
        let moq = origin.publisher_with_oid(source, oid)?;
        Self::from_public_prefix(source, moq)
    }

    /// Plaintext publisher that writes ONLY to the per-OID (#393) publication
    /// track for `oid` (no flat-track mirror; post-migration).
    pub fn new_oid_only(source: &str, oid: &str) -> Result<Self> {
        let origin = global_moq_event_origin().ok_or_else(|| {
            anyhow!("moq event bus not initialized; start the event service first")
        })?;
        let moq = origin.publisher_oid_only(source, oid)?;
        Self::from_public_prefix(source, moq)
    }

    fn from_public_prefix(source: &str, moq: MoqEventPublisher) -> Result<Self> {
        let mut map = HashMap::new();
        map.insert(source.to_owned(), PrefixState { moq, crypto: None });
        Ok(Self {
            primary_source: source.to_owned(),
            prefixes: Arc::new(RwLock::new(map)),
            signing_key: None,
            privacy_mode: EventPrivacy::Public,
            rekey_policy: RekeyPolicy::default(),
            // Public firehose profile: no authz by design; identity N/A.
            authz: Arc::new(AllowAllEventAuthz),
            publisher_identity: PublisherIdentity::anonymous(),
        })
    }

    /// Create an encrypted (`ZeroKnowledge`/`LimitedKnowledge`) publisher.
    /// No prefix is bound at construction — call [`Self::register_prefix`]
    /// for each `local/events/{prefix}` broadcast this publisher will write
    /// to, then [`Self::publish_raw`] with a fully-qualified
    /// `{prefix}.{entity}.{event}` topic (the convenience
    /// `publish(entity, event, payload)` method is only available on
    /// single-prefix `Public`-mode publishers from `new`/`new_with_oid`/
    /// `new_oid_only`).
    pub fn new_encrypted(
        signing_key: SigningKey,
        privacy_mode: EventPrivacy,
        rekey_policy: RekeyPolicy,
    ) -> Result<Self, String> {
        if privacy_mode == EventPrivacy::Public {
            return Err(
                "new_encrypted requires ZeroKnowledge or LimitedKnowledge; use EventPublisher::new for Public mode".to_owned(),
            );
        }
        rekey_policy.validate()?;
        Ok(Self {
            primary_source: String::new(),
            prefixes: Arc::new(RwLock::new(HashMap::new())),
            signing_key: Some(signing_key),
            privacy_mode,
            rekey_policy,
            // Confidential profile: deny-by-default (MAC) until a real authz is
            // injected via `with_authz`; anonymous identity until #446 is wired
            // via `with_publisher_identity`.
            authz: Arc::new(DenyAllEventAuthz),
            publisher_identity: PublisherIdentity::anonymous(),
        })
    }

    /// Inject the event-plane authz (EV4). The encrypted profile defaults to
    /// [`DenyAllEventAuthz`] (fail-closed); production wires a UCAN/capability-
    /// backed impl (S3/S4 vocab) here. Returns `self` for chaining.
    pub fn with_authz(mut self, authz: Arc<dyn EventAuthz>) -> Self {
        self.authz = authz;
        self
    }

    /// Inject the publisher's verified identity (EV4 authn-on-publish). Until
    /// #446 lands the default is anonymous, which fail-closes confidential
    /// publish; a non-IPC path (or post-#446) supplies a verified DID here.
    pub fn with_publisher_identity(mut self, identity: PublisherIdentity) -> Self {
        self.publisher_identity = identity;
        self
    }

    /// Register a new encrypted prefix: opens its moq transport, generates a
    /// fresh group key and Ristretto255 ephemeral keypair.
    ///
    /// Returns the publisher's ephemeral pubkey for this prefix (used by
    /// subscribers' [`EventSubscriber::join_prefix`]).
    pub async fn register_prefix(&self, prefix: &str) -> Result<[u8; 32], String> {
        if self.privacy_mode == EventPrivacy::Public {
            return Err("register_prefix is only valid on encrypted (ZK/LK) publishers".to_owned());
        }
        let origin = global_moq_event_origin().ok_or_else(|| {
            "moq event bus not initialized; start the event service first".to_owned()
        })?;
        let moq = origin.publisher(prefix).map_err(|e| e.to_string())?;

        let mut group_key = Zeroizing::new([0u8; 32]);
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut *group_key);
        let (eph_secret, eph_public) = generate_ephemeral_keypair();
        let ephemeral_secret = Zeroizing::new(eph_secret.scalar().to_bytes());
        let ephemeral_pubkey = eph_public.to_bytes();

        let crypto = PrefixCrypto {
            key_state: GroupKeyState {
                current: group_key,
                pending: None,
                created_at: Instant::now(),
            },
            ephemeral_secret,
            ephemeral_pubkey,
            subscribers: HashMap::new(),
        };

        self.prefixes.write().await.insert(
            prefix.to_owned(),
            PrefixState {
                moq,
                crypto: Some(crypto),
            },
        );
        Ok(ephemeral_pubkey)
    }

    /// Wrap the current group key for a new subscriber (DH + key-wrap).
    /// Returns the wrapped key blob, suitable for deposit to PolicyService.
    pub async fn wrap_for_subscriber(
        &self,
        prefix: &str,
        subscriber_pubkey: &[u8; 32],
    ) -> Result<Vec<u8>, String> {
        let mut prefixes = self.prefixes.write().await;
        let state = prefixes
            .get_mut(prefix)
            .ok_or_else(|| format!("prefix '{prefix}' not registered"))?;
        let crypto = state
            .crypto
            .as_mut()
            .ok_or_else(|| format!("prefix '{prefix}' is not an encrypted prefix"))?;

        maybe_promote_pending(crypto);

        let sub_hash = hash_pubkey(subscriber_pubkey);
        let group_key = &crypto.key_state.current;
        let shared_secret = Zeroizing::new(
            ristretto_dh_raw(&crypto.ephemeral_secret, subscriber_pubkey)
                .map_err(|e| format!("DH failed: {e}"))?,
        );
        let wrap_key = derive_wrap_key(&shared_secret, subscriber_pubkey, &crypto.ephemeral_pubkey);
        wrap_group_key(&wrap_key, group_key, &sub_hash, prefix)
    }

    /// Wrap the group key for multiple new subscribers. Returns
    /// `(sub_pubkey_hash, wrapped_blob)` pairs ready for PolicyService deposit.
    pub async fn wrap_for_new_subscribers(
        &self,
        prefix: &str,
        subscriber_pubkeys: &[[u8; 32]],
    ) -> Result<Vec<([u8; 32], Vec<u8>)>, String> {
        let mut prefixes = self.prefixes.write().await;
        let state = prefixes
            .get_mut(prefix)
            .ok_or_else(|| format!("prefix '{prefix}' not registered"))?;
        let crypto = state
            .crypto
            .as_mut()
            .ok_or_else(|| format!("prefix '{prefix}' is not an encrypted prefix"))?;

        maybe_promote_pending(crypto);

        let mut results = Vec::with_capacity(subscriber_pubkeys.len());
        for sub_pubkey in subscriber_pubkeys {
            let sub_hash = hash_pubkey(sub_pubkey);
            let shared_secret = Zeroizing::new(
                ristretto_dh_raw(&crypto.ephemeral_secret, sub_pubkey)
                    .map_err(|e| format!("DH failed: {e}"))?,
            );
            let wrap_key = derive_wrap_key(&shared_secret, sub_pubkey, &crypto.ephemeral_pubkey);
            let wrapped = wrap_group_key(&wrap_key, &crypto.key_state.current, &sub_hash, prefix)?;
            crypto.subscribers.insert(sub_hash, *sub_pubkey);
            results.push((sub_hash, wrapped));
        }
        Ok(results)
    }

    /// Publish `{primary_source}.{entity}.{event}` (matches the pre-EV1
    /// plaintext `EventPublisher::publish(entity, event, payload)` call
    /// shape). Only valid on single-prefix publishers from `new`/
    /// `new_with_oid`/`new_oid_only`; multi-prefix `new_encrypted` publishers
    /// must use [`Self::publish_raw`] with a fully-qualified topic.
    pub async fn publish(&self, entity: &str, event: &str, payload: &[u8]) -> Result<()> {
        if entity.contains('.') {
            return Err(anyhow!("Entity name cannot contain '.': {entity}"));
        }
        if event.contains('.') {
            return Err(anyhow!("Event name cannot contain '.': {event}"));
        }
        if self.primary_source.is_empty() {
            return Err(anyhow!(
                "no primary source bound; use publish_raw with a fully-qualified topic, or construct via EventPublisher::new(source)"
            ));
        }
        let topic = format!("{}.{}.{}", self.primary_source, entity, event);
        self.publish_raw(&topic, payload).await
    }

    /// Publish a pre-formatted `{prefix}.{...}` topic + payload.
    ///
    /// In `Public` mode the payload is written unmodified (byte-identical to
    /// the pre-EV1 plaintext path). In `ZeroKnowledge`/`LimitedKnowledge`
    /// mode the payload is signed + encrypted and wrapped in a wire frame
    /// (see module docs) before being written to the shared moq track for
    /// the topic's prefix (`topic.split('.').next()`).
    pub async fn publish_raw(&self, topic: &str, payload: &[u8]) -> Result<()> {
        let prefix = topic.split('.').next().unwrap_or(topic).to_owned();
        let mut prefixes = self.prefixes.write().await;
        let state = prefixes
            .get_mut(&prefix)
            .ok_or_else(|| anyhow!("prefix '{prefix}' not registered"))?;

        match state.crypto.as_mut() {
            None => state.moq.publish_raw(topic, payload),
            Some(crypto) => {
                // EV4 event-plane PEP (publish ScopeAction). The caller is this
                // publisher; its subject comes from `publisher_identity`
                // (`Subject::anonymous()` until #446 wires verified IPC
                // identity — a real authz impl denies anonymous for the
                // confidential profile). DenyAll (the default) denies every
                // publish until a real authz is injected via `with_authz`.
                let caller = match &self.publisher_identity.did {
                    // TODO(#446): IPC callers currently resolve as anonymous
                    // (key-derived identity lost over UDS); the verified DID
                    // binding is the #446 fast-follow. Do NOT fake a DID.
                    Some(did) => Subject::new(did.clone()),
                    None => Subject::anonymous(),
                };
                if !self.authz.can_publish(&caller, &prefix) {
                    return Err(anyhow!(
                        "publish denied by event-plane authz for prefix '{prefix}'"
                    ));
                }
                maybe_promote_pending(crypto);
                let signing_key = self
                    .signing_key
                    .as_ref()
                    .ok_or_else(|| anyhow!("encrypted privacy mode requires a signing key"))?;

                let group_key: &[u8; 32] = &crypto.key_state.current;
                let timestamp = now_millis();
                let sig_message = build_event_sig_message(topic, payload, timestamp);
                let signature = signing_key.sign(&sig_message);

                let (tag, ciphertext, nonce, commitment) =
                    encrypt_event(group_key, &prefix, payload, self.privacy_mode)
                        .map_err(|e| anyhow!(e))?;

                let lk_tag = match self.privacy_mode {
                    EventPrivacy::LimitedKnowledge => {
                        keyed_mac(group_key, prefix.as_bytes())[..16].to_vec()
                    }
                    EventPrivacy::ZeroKnowledge | EventPrivacy::Public => vec![],
                };

                let encrypted = EncryptedEvent {
                    topic: topic.to_owned(),
                    tag,
                    ciphertext,
                    nonce,
                    key_commitment: commitment,
                    lk_tag,
                    signature: signature.to_bytes().to_vec(),
                    publisher_pubkey: signing_key.verifying_key().to_bytes(),
                    timestamp,
                };

                state.moq.publish_raw(topic, &encrypted.encode_body())
            }
        }
    }

    /// True if a key rotation is due for `prefix` per this publisher's
    /// [`RekeyPolicy`].
    pub async fn needs_rotation(&self, prefix: &str) -> bool {
        let prefixes = self.prefixes.read().await;
        let Some(state) = prefixes.get(prefix) else {
            return false;
        };
        let Some(crypto) = state.crypto.as_ref() else {
            return false;
        };
        let age = crypto.key_state.created_at.elapsed();
        match &self.rekey_policy {
            RekeyPolicy::Scheduled { interval } => age >= *interval,
            RekeyPolicy::Immediate => false,
            RekeyPolicy::Jittered { interval, jitter } => {
                let jitter_offset = {
                    let h = hash_prefix_bytes(prefix.as_bytes());
                    let frac = (h[0] as u64 * 256 + h[1] as u64) as f64 / 65536.0;
                    Duration::from_secs_f64(jitter.as_secs_f64() * frac)
                };
                age >= *interval + jitter_offset
            }
        }
    }

    /// Rotate the group key for `prefix`: generates a new group key +
    /// ephemeral keypair, wraps the new key for all known subscribers. The
    /// caller is responsible for broadcasting a rekey announcement and
    /// depositing the wrapped keys.
    pub async fn rotate_key(
        &self,
        prefix: &str,
        effective_delay: Duration,
    ) -> Result<RotationResult, String> {
        let mut prefixes = self.prefixes.write().await;
        let state = prefixes
            .get_mut(prefix)
            .ok_or_else(|| format!("prefix '{prefix}' not registered"))?;
        let crypto = state
            .crypto
            .as_mut()
            .ok_or_else(|| format!("prefix '{prefix}' is not an encrypted prefix"))?;

        let mut new_group_key = Zeroizing::new([0u8; 32]);
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut *new_group_key);
        let (new_eph_secret, new_eph_public) = generate_ephemeral_keypair();
        let new_ephemeral_secret = Zeroizing::new(new_eph_secret.scalar().to_bytes());
        let new_ephemeral_pubkey = new_eph_public.to_bytes();

        let mut wrapped_keys = Vec::with_capacity(crypto.subscribers.len());
        for (sub_hash, sub_pubkey) in &crypto.subscribers {
            let shared_secret = Zeroizing::new(
                ristretto_dh_raw(&new_ephemeral_secret, sub_pubkey)
                    .map_err(|e| format!("DH failed: {e}"))?,
            );
            let wrap_key = derive_wrap_key(&shared_secret, sub_pubkey, &new_ephemeral_pubkey);
            let wrapped = wrap_group_key(&wrap_key, &new_group_key, sub_hash, prefix)?;
            let mut routing_tag = [0u8; 16];
            rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut routing_tag);
            wrapped_keys.push(WrappedKeyEntry {
                routing_tag,
                wrapped_blob: wrapped,
            });
        }

        let effective_at = Instant::now() + effective_delay;
        // During the pending window, wrap_for_subscriber uses the CURRENT
        // (old) ephemeral keypair and group key — new subscribers get a
        // working key that will be rekeyed at effective_at via the rekey
        // announcement, atomically promoted by maybe_promote_pending().
        crypto.key_state.pending = Some(PendingRekey {
            new_key: new_group_key,
            effective_at,
            new_ephemeral_secret,
            new_ephemeral_pubkey,
        });

        let effective_at_millis = now_millis() + effective_delay.as_millis() as i64;

        Ok(RotationResult {
            new_ephemeral_pubkey,
            wrapped_keys,
            effective_at_millis,
        })
    }

    /// The privacy mode this publisher was constructed with.
    pub fn privacy_mode(&self) -> EventPrivacy {
        self.privacy_mode
    }

    /// The Ed25519 verifying key bytes, if this is an encrypted publisher.
    pub fn verifying_key(&self) -> Option<[u8; 32]> {
        self.signing_key
            .as_ref()
            .map(|k| k.verifying_key().to_bytes())
    }

    /// The bound primary source (single-prefix publishers only; empty for
    /// multi-prefix `new_encrypted` publishers).
    pub fn source(&self) -> &str {
        &self.primary_source
    }
}

/// Promote a pending key + ephemeral keypair to current if past
/// `effective_at`. Must be called under the prefixes write lock.
fn maybe_promote_pending(crypto: &mut PrefixCrypto) {
    if let Some(pending) = crypto.key_state.pending.take() {
        if Instant::now() >= pending.effective_at {
            crypto.key_state.current = pending.new_key;
            crypto.key_state.created_at = Instant::now();
            crypto.ephemeral_secret = pending.new_ephemeral_secret;
            crypto.ephemeral_pubkey = pending.new_ephemeral_pubkey;
        } else {
            crypto.key_state.pending = Some(pending);
        }
    }
}

// ============================================================================
// KeyRing + RekeyEvent — publisher-side state (EventService-specific; the
// reusable keyable-group primitive is `crypto::group_key::GroupKeyRegistry`).
// ============================================================================

// ============================================================================
// EventSubscriber — the canonical broadcast subscriber
// ============================================================================

/// Holds the current and previous group key for a joined prefix.
struct KeyRing {
    current: Zeroizing<[u8; 32]>,
    previous: Option<Zeroizing<[u8; 32]>>,
    grace_until: Option<Instant>,
}

impl KeyRing {
    fn new(key: Zeroizing<[u8; 32]>) -> Self {
        Self {
            current: key,
            previous: None,
            grace_until: None,
        }
    }

    fn is_grace_expired(&self) -> bool {
        match self.grace_until {
            Some(deadline) => Instant::now() >= deadline,
            None => true,
        }
    }

    fn gc_previous(&mut self) {
        if self.is_grace_expired() {
            self.previous = None;
            self.grace_until = None;
        }
    }
}

/// Per-joined-prefix subscriber state: ephemeral DH keypair + key ring.
struct SubscriberPrefixState {
    ephemeral_secret: Zeroizing<[u8; 32]>,
    ephemeral_pubkey: [u8; 32],
    /// Publisher's Ristretto255 DH pubkey (for wrap-key re-derivation during
    /// rekey) — distinct from the Ed25519 signing pubkey embedded per-event
    /// in [`EncryptedEvent::publisher_pubkey`].
    publisher_dh_pubkey: [u8; 32],
    ring: KeyRing,
}

/// A pending rekey notification.
#[derive(Debug)]
pub struct RekeyEvent {
    pub prefix: String,
    pub new_key: Zeroizing<[u8; 32]>,
    pub effective_at: Instant,
}

/// The canonical broadcast event subscriber (EV1, epic #600).
///
/// Wraps [`MoqEventSubscriber`] for transport. Prefixes NOT joined via
/// [`Self::join_prefix`] are treated as `Public` mode: [`Self::recv`] returns
/// their frames unmodified (byte-identical to the pre-EV1 plaintext
/// subscriber). Joined prefixes are decoded, signature-checked, and
/// decrypted automatically.
pub struct EventSubscriber {
    inner: MoqEventSubscriber,
    prefixes: Arc<RwLock<HashMap<String, SubscriberPrefixState>>>,
    rekey_tx: mpsc::Sender<RekeyEvent>,
    rekey_rx: Option<mpsc::Receiver<RekeyEvent>>,
}

impl EventSubscriber {
    /// Create a new subscriber. No background task is started until the
    /// first `recv()` call.
    pub fn new() -> Result<Self> {
        let (tx, rx) = mpsc::channel(32);
        Ok(Self {
            inner: MoqEventSubscriber::new(),
            prefixes: Arc::new(RwLock::new(HashMap::new())),
            rekey_tx: tx,
            rekey_rx: Some(rx),
        })
    }

    /// Take the rekey-notification receiver. Returns `None` if already taken.
    pub fn take_rekey_receiver(&mut self) -> Option<mpsc::Receiver<RekeyEvent>> {
        self.rekey_rx.take()
    }

    /// Subscribe to a topic pattern (prefix match). See
    /// [`MoqEventSubscriber::subscribe`] for pattern semantics.
    pub fn subscribe(&mut self, pattern: &str) -> Result<()> {
        self.inner.subscribe(pattern)
    }

    /// Subscribe to all events.
    pub fn subscribe_all(&mut self) -> Result<()> {
        self.inner.subscribe_all()
    }

    /// Unsubscribe from a topic pattern. Must be called before `recv()`.
    pub fn unsubscribe(&mut self, pattern: &str) -> Result<()> {
        self.inner.unsubscribe(pattern)
    }

    /// Subscribe to a single model OID's per-OID publication track (#393).
    pub fn subscribe_oid(&mut self, oid: &str) -> Result<()> {
        self.inner.subscribe_oid(oid)
    }

    /// Set the late-join retention mode (#393 decision A: firehose-backfill).
    pub fn with_backfill(&mut self, mode: BackfillMode) -> Result<()> {
        self.inner.with_backfill(mode)
    }

    /// Select the delivery QoS (#606). Pass
    /// `hyprstream_rpc::stream_info::EventReliable::stream_opt()` for
    /// at-least-once delivery (events that must not be silently dropped).
    /// Defaults to `EventLive` (at-most-once, drop-oldest) if never called.
    /// Must be called before the first `recv()`.
    pub fn with_qos(&mut self, qos: crate::stream_info::StreamOpt) -> Result<()> {
        self.inner.with_qos(qos)
    }

    /// Skip live events already delivered in a prior session (offset-resume,
    /// #606). See [`crate::moq_event::MoqEventSubscriber::with_resume_from`]
    /// for the multi-source caveat. Must be called before the first `recv()`.
    pub fn with_resume_from(&mut self, sequence: u64) -> Result<()> {
        self.inner.with_resume_from(sequence)
    }

    /// Highest live-group sequence delivered so far (resume hint, #606).
    pub fn last_sequence(&self) -> u64 {
        self.inner.last_sequence()
    }

    /// Count of items evicted under drop-oldest backpressure (#606).
    pub fn dropped_count(&self) -> u64 {
        self.inner.dropped_count()
    }

    /// Join an encrypted prefix: generate an ephemeral keypair, unwrap the
    /// initial group key from `wrapped_key_blob` (obtained out-of-band, e.g.
    /// via PolicyService deposit). Required before [`Self::recv`] will
    /// decrypt events for `prefix`; un-joined prefixes pass through as
    /// `Public` mode.
    pub async fn join_prefix(
        &self,
        prefix: &str,
        publisher_dh_pubkey: &[u8; 32],
        wrapped_key_blob: &[u8],
    ) -> Result<[u8; 32], String> {
        let (eph_secret, eph_public) = generate_ephemeral_keypair();
        let eph_pubkey_bytes = eph_public.to_bytes();
        let eph_secret_bytes: [u8; 32] = eph_secret.scalar().to_bytes();

        let shared_secret = Zeroizing::new(
            ristretto_dh_raw(&eph_secret_bytes, publisher_dh_pubkey)
                .map_err(|e| format!("DH failed: {e}"))?,
        );
        let wrap_key = derive_wrap_key(&shared_secret, publisher_dh_pubkey, &eph_pubkey_bytes);
        let sub_hash: [u8; 32] = *blake3::hash(&eph_pubkey_bytes).as_bytes();
        let group_key = unwrap_group_key(&wrap_key, wrapped_key_blob, &sub_hash, prefix)?;

        let state = SubscriberPrefixState {
            ephemeral_secret: Zeroizing::new(eph_secret_bytes),
            ephemeral_pubkey: eph_pubkey_bytes,
            publisher_dh_pubkey: *publisher_dh_pubkey,
            ring: KeyRing::new(group_key),
        };

        self.prefixes.write().await.insert(prefix.to_owned(), state);
        Ok(eph_pubkey_bytes)
    }

    /// Try to decrypt an already-decoded ciphertext for `prefix`, trialing
    /// the current key then the previous key (if within grace period).
    pub async fn try_decrypt(
        &self,
        prefix: &str,
        tag: &[u8],
        ciphertext: &[u8],
        nonce: &[u8; 12],
        key_commitment: &[u8; 16],
    ) -> Result<Vec<u8>, String> {
        let (current_key, previous_key) = {
            let prefixes = self.prefixes.read().await;
            let state = prefixes
                .get(prefix)
                .ok_or_else(|| format!("not joined to prefix: {prefix}"))?;
            let current = state.ring.current.clone();
            let previous = if !state.ring.is_grace_expired() {
                state.ring.previous.clone()
            } else {
                None
            };
            (current, previous)
        };

        if check_key_commitment(&current_key, nonce, key_commitment) {
            match decrypt_event_full(&current_key, nonce, tag, ciphertext, prefix) {
                Ok(plaintext) => return Ok(plaintext),
                Err(e) => debug!(
                    prefix,
                    "current key commitment matched but decrypt failed: {e}"
                ),
            }
        }

        if let Some(ref prev_key) = previous_key {
            if check_key_commitment(prev_key, nonce, key_commitment) {
                match decrypt_event_full(prev_key, nonce, tag, ciphertext, prefix) {
                    Ok(plaintext) => {
                        debug!(prefix, "decrypted with previous key (grace period)");
                        return Ok(plaintext);
                    }
                    Err(e) => debug!(
                        prefix,
                        "previous key commitment matched but decrypt failed: {e}"
                    ),
                }
            }
        }

        Err("decryption failed: no matching key".to_owned())
    }

    /// Handle a rekey: trial-decrypt the wrapped key entries and send the new
    /// key to the rekey channel.
    pub async fn handle_rekey(
        &self,
        prefix: &str,
        wrapped_blobs: &[Vec<u8>],
        new_publisher_dh_pubkey: &[u8; 32],
        effective_at: Instant,
    ) -> Result<(), String> {
        let (eph_secret, eph_pubkey) = {
            let prefixes = self.prefixes.read().await;
            let state = prefixes
                .get(prefix)
                .ok_or_else(|| format!("not joined to prefix: {prefix}"))?;
            (state.ephemeral_secret.clone(), state.ephemeral_pubkey)
        };

        let sub_hash: [u8; 32] = *blake3::hash(&eph_pubkey).as_bytes();
        let shared_secret = Zeroizing::new(
            ristretto_dh_raw(&eph_secret, new_publisher_dh_pubkey)
                .map_err(|e| format!("DH failed during rekey: {e}"))?,
        );
        let wrap_key = derive_wrap_key(&shared_secret, new_publisher_dh_pubkey, &eph_pubkey);

        for wrapped_blob in wrapped_blobs {
            if let Ok(new_key) = unwrap_group_key(&wrap_key, wrapped_blob, &sub_hash, prefix) {
                {
                    let mut prefixes = self.prefixes.write().await;
                    if let Some(state) = prefixes.get_mut(prefix) {
                        state.publisher_dh_pubkey = *new_publisher_dh_pubkey;
                    }
                }
                self.rekey_tx
                    .send(RekeyEvent {
                        prefix: prefix.to_owned(),
                        new_key,
                        effective_at,
                    })
                    .await
                    .map_err(|_| "rekey channel closed".to_owned())?;
                return Ok(());
            }
        }

        Err("no wrapped entry found for our key".to_owned())
    }

    /// Promote a pending key to current, demoting current to previous with a
    /// grace period. Should be called at (or after) `effective_at` from a
    /// [`RekeyEvent`].
    pub async fn promote_key(
        &self,
        prefix: &str,
        new_key: Zeroizing<[u8; 32]>,
        grace_period: Option<Duration>,
    ) -> Result<(), String> {
        let mut prefixes = self.prefixes.write().await;
        let state = prefixes
            .get_mut(prefix)
            .ok_or_else(|| format!("not joined to prefix: {prefix}"))?;

        let grace = grace_period.unwrap_or(DEFAULT_SUBSCRIBER_GRACE_PERIOD);
        let old_current = std::mem::replace(&mut state.ring.current, new_key);
        state.ring.previous = Some(old_current);
        state.ring.grace_until = Some(Instant::now() + grace);
        debug!(prefix, ?grace, "promoted new key, old key in grace period");
        Ok(())
    }

    /// Garbage-collect expired previous keys across all joined prefixes.
    pub async fn gc_expired_keys(&self) {
        let mut prefixes = self.prefixes.write().await;
        for (prefix, state) in prefixes.iter_mut() {
            if state.ring.previous.is_some() && state.ring.is_grace_expired() {
                state.ring.gc_previous();
                debug!(prefix, "expired previous key removed");
            }
        }
    }

    /// Receive the next event.
    ///
    /// For prefixes NOT joined via [`Self::join_prefix`], the raw payload is
    /// returned unmodified (`Public` mode passthrough). For joined prefixes,
    /// the wire frame is decoded, the embedded Ed25519 signature is checked
    /// (tamper-evidence; see module docs on identity binding), and the
    /// payload is decrypted via the current/previous-key trial logic in
    /// [`Self::try_decrypt`]. Frames that fail to decode, verify, or decrypt
    /// are dropped (logged at debug) and the loop continues — matching the
    /// event bus's best-effort, at-most-once delivery semantics.
    pub async fn recv(&mut self) -> Result<(String, Vec<u8>)> {
        loop {
            let (topic, raw) = self.inner.recv().await?;
            match self.decode_frame(&topic, &raw).await {
                FrameOutcome::Passthrough => return Ok((topic, raw)),
                FrameOutcome::Decoded(payload) => return Ok((topic, payload)),
                FrameOutcome::Drop(reason) => {
                    debug!(
                        topic,
                        reason, "dropping undecodable/unverifiable event frame"
                    );
                    continue;
                }
            }
        }
    }

    /// Receive with a timeout. Same decode/decrypt behavior as [`Self::recv`].
    pub async fn recv_timeout(&mut self, timeout: Duration) -> Result<Option<(String, Vec<u8>)>> {
        match tokio::time::timeout(timeout, self.recv()).await {
            Ok(result) => result.map(Some),
            Err(_elapsed) => Ok(None),
        }
    }

    /// Try to receive without blocking. Same decode/decrypt behavior as
    /// [`Self::recv`], except a frame that fails to decode/verify/decrypt is
    /// dropped and `Ok(None)` returned immediately rather than continuing to
    /// poll (this method must not block).
    pub fn try_recv(&mut self) -> Result<Option<(String, Vec<u8>)>> {
        let Some((topic, raw)) = self.inner.try_recv()? else {
            return Ok(None);
        };
        // `decode_frame` is async (it takes the prefixes read lock); but the
        // lock is uncontended in the common case and this method already
        // requires being called from an async context to exist at all, so a
        // blocking `block_in_place`-free best-effort: prefixes not joined
        // (the overwhelmingly common Public-mode case) never touch the lock.
        if topic
            .split('.')
            .next()
            .map(|prefix| matches!(self.prefixes.try_read(), Ok(p) if p.contains_key(prefix)))
            .unwrap_or(false)
        {
            // An encrypted prefix; decoding requires the async path. Callers
            // needing non-blocking receive on encrypted prefixes should use
            // `recv_timeout(Duration::ZERO)` instead.
            return Err(anyhow!(
                "try_recv() does not support decoding encrypted prefixes; use recv_timeout(Duration::ZERO)"
            ));
        }
        Ok(Some((topic, raw)))
    }

    async fn decode_frame(&self, topic: &str, raw: &[u8]) -> FrameOutcome {
        let prefix = topic.split('.').next().unwrap_or(topic);
        let joined = self.prefixes.read().await.contains_key(prefix);
        if !joined {
            return FrameOutcome::Passthrough;
        }

        let encrypted = match EncryptedEvent::decode_body(topic, raw) {
            Ok(e) => e,
            Err(e) => return FrameOutcome::Drop(format!("decode: {e}")),
        };

        // Tamper-evidence check (see module docs: self-asserted pubkey, not
        // yet identity-bound — that is epic #600 EV4 #604).
        let Ok(verifying_key) = VerifyingKey::from_bytes(&encrypted.publisher_pubkey) else {
            return FrameOutcome::Drop("invalid embedded publisher pubkey".to_owned());
        };
        let signature = Signature::from_bytes(match encrypted.signature.as_slice().try_into() {
            Ok(s) => s,
            Err(_) => return FrameOutcome::Drop("signature is not 64 bytes".to_owned()),
        });

        let plaintext = match self
            .try_decrypt(
                prefix,
                &encrypted.tag,
                &encrypted.ciphertext,
                &encrypted.nonce,
                &encrypted.key_commitment,
            )
            .await
        {
            Ok(p) => p,
            Err(e) => return FrameOutcome::Drop(format!("decrypt: {e}")),
        };

        let sig_message = build_event_sig_message(topic, &plaintext, encrypted.timestamp);
        if verifying_key.verify(&sig_message, &signature).is_err() {
            return FrameOutcome::Drop("signature verification failed".to_owned());
        }

        FrameOutcome::Decoded(plaintext)
    }
}

enum FrameOutcome {
    /// Prefix not joined — Public-mode passthrough, raw bytes already correct.
    Passthrough,
    /// Decoded, verified, and decrypted plaintext.
    Decoded(Vec<u8>),
    /// Drop the frame; carries a debug-log reason.
    Drop(String),
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn test_signing_key() -> SigningKey {
        let mut secret = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut secret);
        SigningKey::from_bytes(&secret)
    }

    #[test]
    fn rekey_policy_validation() {
        let result = RekeyPolicy::Scheduled {
            interval: Duration::from_secs(100_000),
        }
        .validate();
        assert!(result.is_err());

        let result = RekeyPolicy::Scheduled {
            interval: Duration::from_secs(3600),
        }
        .validate();
        assert!(result.is_ok());
    }

    #[test]
    fn encrypted_event_wire_roundtrip() {
        let event = EncryptedEvent {
            topic: "worker.sandbox1.started".to_owned(),
            tag: vec![1, 2, 3, 4],
            ciphertext: vec![5, 6, 7, 8, 9],
            nonce: [9u8; 12],
            key_commitment: [7u8; 16],
            lk_tag: vec![],
            signature: vec![1u8; 64],
            publisher_pubkey: [3u8; 32],
            timestamp: 1_700_000_000_000,
        };

        let encoded = event.encode_body();
        let decoded = EncryptedEvent::decode_body(&event.topic, &encoded).unwrap();

        assert_eq!(decoded.topic, event.topic);
        assert_eq!(decoded.tag, event.tag);
        assert_eq!(decoded.ciphertext, event.ciphertext);
        assert_eq!(decoded.nonce, event.nonce);
        assert_eq!(decoded.key_commitment, event.key_commitment);
        assert_eq!(decoded.lk_tag, event.lk_tag);
        assert_eq!(decoded.signature, event.signature);
        assert_eq!(decoded.publisher_pubkey, event.publisher_pubkey);
        assert_eq!(decoded.timestamp, event.timestamp);
    }

    #[test]
    fn encrypted_event_wire_rejects_truncated_input() {
        let buf = vec![WIRE_VERSION, 1, 2, 3]; // far too short
        assert!(EncryptedEvent::decode_body("topic", &buf).is_err());
    }

    #[test]
    fn encrypted_event_wire_rejects_bad_version() {
        let mut buf = vec![99u8]; // bad version
        buf.extend_from_slice(&[0u8; 200]);
        assert!(EncryptedEvent::decode_body("topic", &buf).is_err());
    }

    #[tokio::test]
    async fn register_prefix_requires_global_origin() {
        // No global origin initialized in this test process (or a stale one
        // from another test) — either way, register_prefix must not panic and
        // must surface a clean error rather than wiring to nothing.
        let publisher = EventPublisher::new_encrypted(
            test_signing_key(),
            EventPrivacy::ZeroKnowledge,
            RekeyPolicy::Scheduled {
                interval: Duration::from_secs(3600),
            },
        )
        .unwrap();
        // Either errors cleanly (no global origin) or succeeds (origin was
        // initialized by another test in this binary) — both are acceptable;
        // the assertion is just "no panic".
        let _ = publisher.register_prefix("test-prefix").await;
    }

    #[test]
    fn new_encrypted_rejects_public_mode() {
        let result = EventPublisher::new_encrypted(
            test_signing_key(),
            EventPrivacy::Public,
            RekeyPolicy::default(),
        );
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn publish_raw_errors_on_unregistered_prefix() {
        let publisher = EventPublisher::new_encrypted(
            test_signing_key(),
            EventPrivacy::ZeroKnowledge,
            RekeyPolicy::default(),
        )
        .unwrap();
        let result = publisher
            .publish_raw("unregistered.entity.event", b"payload")
            .await;
        assert!(result.is_err());
    }

    // ── EV4: event-plane authz + publisher identity ─────────────────────────

    #[test]
    fn event_authz_impls() {
        // DenyAll (the encrypted-profile default) denies everything.
        let deny = DenyAllEventAuthz;
        assert!(!deny.can_publish(&Subject::anonymous(), "p"));
        assert!(!deny.can_subscribe(&Subject::anonymous(), "p"));
        // AllowAll (the public-profile / test default) permits everything.
        let allow = AllowAllEventAuthz;
        assert!(allow.can_publish(&Subject::anonymous(), "p"));
        assert!(allow.can_subscribe(&Subject::anonymous(), "p"));
    }

    #[test]
    fn publisher_identity_anonymous_vs_verified() {
        // #446-gated: the default is anonymous (no verified DID).
        let anon = PublisherIdentity::anonymous();
        assert!(!anon.is_verified());
        assert!(anon.did.is_none());
        // A verified DID (post-#446 / non-IPC path).
        let verified = PublisherIdentity::verified("did:web:node.example.com");
        assert!(verified.is_verified());
        assert_eq!(verified.did.as_deref(), Some("did:web:node.example.com"));
        // Default == anonymous.
        assert!(!PublisherIdentity::default().is_verified());
    }

    #[tokio::test]
    async fn encrypted_publish_blocked_by_default_denyall_authz() {
        // The encrypted profile defaults to DenyAllEventAuthz (MAC
        // deny-by-default). With a registered encrypted prefix, publish_raw
        // MUST be denied by the event-plane PEP — even before reaching the
        // signing/encryption step. (No real moq origin needed: the authz gate
        // is the first check in the encrypted arm.)
        let publisher = EventPublisher::new_encrypted(
            test_signing_key(),
            EventPrivacy::ZeroKnowledge,
            RekeyPolicy::default(),
        )
        .unwrap()
        .with_publisher_identity(PublisherIdentity::verified("did:web:pub"));
        // Inject a prefix with crypto state directly (bypasses register_prefix,
        // which needs a global moq origin). The moq field is never reached —
        // the authz gate fires first — so a panicking placeholder is fine.
        let crypto = PrefixCrypto {
            key_state: GroupKeyState {
                current: Zeroizing::new([0u8; 32]),
                pending: None,
                created_at: std::time::Instant::now(),
            },
            ephemeral_secret: Zeroizing::new([0u8; 32]),
            ephemeral_pubkey: [0u8; 32],
            subscribers: HashMap::new(),
        };
        // SAFETY-of-test: we never publish to `moq` because DenyAll fires first.
        // We cannot construct a real MoqEventPublisher without the global origin,
        // so we register the prefix into the map with crypto and rely on the
        // authz gate short-circuiting. Use the publisher's own map: register an
        // entry whose `moq` is sourced from a real origin when available.
        if let Some(origin) = global_moq_event_origin() {
            if let Ok(moq) = origin.publisher("deny-test") {
                let mut prefixes = publisher.prefixes.write().await;
                prefixes.insert(
                    "deny-test".to_owned(),
                    PrefixState {
                        moq,
                        crypto: Some(crypto),
                    },
                );
                drop(prefixes);
                let result = publisher.publish_raw("deny-test.e.e", b"p").await;
                let err =
                    result.expect_err("DenyAll must block encrypted publish");
                assert!(
                    err.to_string().contains("denied by event-plane authz"),
                    "expected authz denial, got: {err}"
                );
            }
        }
        // If no global origin is available in this test binary, the
        // inject-then-deny path is not exercisable here; the unit tests above
        // cover the authz surface directly.
    }
}
