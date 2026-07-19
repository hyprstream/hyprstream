//! Unified EventService publisher/subscriber (epic #600).
//!
//! Public publishers preserve the existing plaintext MoQ event path. Confidential
//! publishers use controller-managed group epochs: the controller atomically
//! prepares and commits each membership-version/epoch transition, then distributes
//! the fresh epoch secret separately to each accepted member with HyKEM
//! (X25519 + ML-KEM-768) and COSE_Encrypt0. The stock relay only forwards opaque
//! event bytes and never receives an epoch grant.
//!
//! Event objects derive distinct sender/track AEAD keys and deterministic nonce
//! domains from the epoch secret. Their AAD binds publisher, track, membership
//! version, epoch, and sequence; subscribers reject nonce misuse, replay, unknown
//! epochs, and retired objects. Publisher identity is anchored by the controller
//! grant and attested with mandatory Ed25519 + ML-DSA-65 composite signatures.
//! Ed25519 keys and subscriber-generated blinded Ristretto presentations are never
//! used as KEX keys.
//!
//! # Wire format
//!
//! `EventPrivacy::Public` publishes payloads unmodified. Confidential modes encode
//! an [`EncryptedEvent`] as a versioned, length-prefixed MoQ payload:
//!
//! ```text
//! [1B version][16B session_id][12B nonce][16B key_commitment]
//! [4B tag_len][tag][4B ciphertext_len][ciphertext][4B lk_tag_len][lk_tag]
//! [8B timestamp BE][8B membership_version][8B epoch][8B sequence]
//! [32B publisher_pubkey][4B signature_len][hybrid signature]
//! ```
//!
//! The topic remains the MoQ frame topic. It is authenticated by the hybrid
//! signature; deployments requiring routing opacity bind the controller's opaque
//! routing policy into each member grant.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, Result};
use ed25519_dalek::SigningKey;
use rand::RngCore;
use tokio::sync::RwLock;
use tracing::debug;
use zeroize::Zeroizing;

use crate::crypto::cose_sign::{sign_composite, verify_composite};
use crate::crypto::event_crypto::{
    build_epoch_event_sig_message, decrypt_epoch_event, derive_event_nonce, encrypt_epoch_event,
    EventPrivacy,
};
use crate::crypto::group_key::{open_epoch_grant, EpochGrant};
use crate::crypto::hybrid_kem::RecipientKeypair;
use crate::crypto::keyed_mac;
use crate::crypto::pq::{ml_dsa_vk_bytes, MlDsaSigningKey};
use crate::envelope::Subject;
// The four shared event types + the rotation constants are canonical in the
// keyable-group primitive (`crypto::group_key`); re-export them here so the
// `hyprstream_rpc::events::*` surface downstream consumers already use stays
// stable. (Only the EventService-specific `EncryptedEvent` wire codec + the
// `EventPublisher`/`EventSubscriber`/`KeyRing` types are defined in this module.)
pub use crate::crypto::group_key::{
    EncryptedEvent, RekeyPolicy, RotationResult, WrappedKeyEntry, DEFAULT_ROTATION_INTERVAL,
    GRACE_PERIOD, MAX_KEY_LIFETIME,
};
use crate::moq_event::{
    global_moq_event_origin, BackfillMode, MoqEventPublisher, MoqEventSubscriber,
};

// ============================================================================
// Rekey policy — the type + its constants live in the keyable-group primitive
// (`crypto::group_key`); this module re-imports them above. Only the
// subscriber-side grace constant below is specific to this module.
// ============================================================================

/// Maximum grace period in which a subscriber accepts authenticated objects
/// from the prior epoch, bounded by the controller-supplied last-issued
/// sequence. Security-triggered transitions install zero grace.
pub const DEFAULT_SUBSCRIBER_GRACE_PERIOD: Duration = Duration::from_secs(30);

// Rekey policy configuration remains an API-level scheduling hint. Every
// controller membership/security transition nevertheless prepares a fresh
// epoch and O(M) per-member HyKEM/COSE grants; EventPublisher cannot rotate or
// distribute epochs itself.
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

const WIRE_VERSION: u8 = 3;
const EVENT_ATTESTATION_SCHEMA_ID: u64 = 0x86fe_5f45_a782_3a11;
const EVENT_ATTESTATION_TYPE_ID: u64 = 0x4576_7448_7942_3032; // "EvtHyB02"

fn event_attestation_aad() -> Vec<u8> {
    crate::crypto::cose_sign1::build_external_aad(
        EVENT_ATTESTATION_SCHEMA_ID,
        EVENT_ATTESTATION_TYPE_ID,
    )
}

impl EncryptedEvent {
    /// Encode the non-topic fields into a wire frame body (see module docs
    /// for the exact layout). `topic` travels separately as the moq frame's
    /// topic field and is reattached by [`Self::decode_body`].
    fn encode_body(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(
            1 + 16
                + 12
                + 16
                + 4
                + self.tag.len()
                + 4
                + self.ciphertext.len()
                + 4
                + self.lk_tag.len()
                + 8
                + 8
                + 8
                + 8
                + 32
                + 4
                + self.signature.len(),
        );
        buf.push(WIRE_VERSION);
        buf.extend_from_slice(&self.session_id);
        buf.extend_from_slice(&self.nonce);
        buf.extend_from_slice(&self.key_commitment);
        push_lenprefixed(&mut buf, &self.tag);
        push_lenprefixed(&mut buf, &self.ciphertext);
        push_lenprefixed(&mut buf, &self.lk_tag);
        buf.extend_from_slice(&self.timestamp.to_be_bytes());
        buf.extend_from_slice(&self.membership_version.to_be_bytes());
        buf.extend_from_slice(&self.epoch.to_be_bytes());
        buf.extend_from_slice(&self.sequence.to_be_bytes());
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
        let session_id: [u8; 16] = read_array(buf, &mut off)?;
        let nonce: [u8; 12] = read_array(buf, &mut off)?;
        let key_commitment: [u8; 16] = read_array(buf, &mut off)?;
        let tag = read_lenprefixed(buf, &mut off)?;
        let ciphertext = read_lenprefixed(buf, &mut off)?;
        let lk_tag = read_lenprefixed(buf, &mut off)?;
        let timestamp_bytes: [u8; 8] = read_array(buf, &mut off)?;
        let timestamp = i64::from_be_bytes(timestamp_bytes);
        let membership_version = u64::from_be_bytes(read_array(buf, &mut off)?);
        let epoch = u64::from_be_bytes(read_array(buf, &mut off)?);
        let sequence = u64::from_be_bytes(read_array(buf, &mut off)?);
        let publisher_pubkey: [u8; 32] = read_array(buf, &mut off)?;
        let signature = read_lenprefixed(buf, &mut off)?;
        if off != buf.len() {
            return Err(anyhow!("event wire: trailing bytes"));
        }
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
            session_id,
            membership_version,
            epoch,
            sequence,
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

fn hash_prefix_bytes(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

// ============================================================================
// EventPublisher — the canonical broadcast publisher
// ============================================================================

/// Controller-installed crypto state for one encrypted prefix.
///
/// EventService never invents or distributes this key. The controller creates
/// the epoch through `GroupKeyRegistry`, commits membership+epoch atomically,
/// and installs the resulting secret here on an authorized publisher.
struct PrefixCrypto {
    membership_version: u64,
    epoch: u64,
    current: Zeroizing<[u8; 32]>,
    sequence: u64,
    session_id: [u8; 16],
    created_at: Instant,
}

/// Per-prefix state: the moq transport for `local/events/{prefix}`, plus
/// optional encryption state.
struct PrefixState {
    moq: MoqEventPublisher,
    confidential: bool,
    crypto: Option<PrefixCrypto>,
}

// ============================================================================
// Event-plane authorization + publisher identity (EV4, epic #600)
//
// This is the EventService PEP (Policy Enforcement Point) for the
// `publish`/`subscribe` ScopeActions. The load-bearing membership gate for
// obtaining `K_group` is the fail-closed `MembershipResolver` used by
// `GroupKeyRegistry::prepare_change`; these checks guard the event paths.
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
        Self {
            did: Some(did.into()),
        }
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
    /// Dedicated hybrid publisher attestation keys. Both are absent for Public.
    signing_key: Option<SigningKey>,
    pq_signing_key: Option<MlDsaSigningKey>,
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
        map.insert(
            source.to_owned(),
            PrefixState {
                moq,
                confidential: false,
                crypto: None,
            },
        );
        Ok(Self {
            primary_source: source.to_owned(),
            prefixes: Arc::new(RwLock::new(map)),
            signing_key: None,
            pq_signing_key: None,
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
        let pq_signing_key = crate::node_identity::derive_mesh_mldsa_key(&signing_key);
        Ok(Self {
            primary_source: String::new(),
            prefixes: Arc::new(RwLock::new(HashMap::new())),
            signing_key: Some(signing_key),
            pq_signing_key: Some(pq_signing_key),
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

    /// Register a confidential MOQT prefix. No key is generated here: publish
    /// remains fail-closed until the controller installs a committed epoch.
    pub async fn register_prefix(&self, prefix: &str) -> Result<(), String> {
        if self.privacy_mode == EventPrivacy::Public {
            return Err("register_prefix is only valid on encrypted (ZK/LK) publishers".to_owned());
        }
        let mut prefixes = self.prefixes.write().await;
        if prefixes.contains_key(prefix) {
            return Err(format!(
                "prefix '{prefix}' is already registered; replacement could reset a live crypto domain"
            ));
        }
        let origin = global_moq_event_origin().ok_or_else(|| {
            "moq event bus not initialized; start the event service first".to_owned()
        })?;
        let moq = origin.publisher(prefix).map_err(|e| e.to_string())?;
        prefixes.insert(
            prefix.to_owned(),
            PrefixState {
                moq,
                confidential: true,
                crypto: None,
            },
        );
        Ok(())
    }

    /// Install an epoch only after the controller atomically committed the same
    /// membership-version/epoch coordinates.
    pub async fn install_committed_epoch(
        &self,
        prefix: &str,
        membership_version: u64,
        epoch: u64,
        epoch_secret: Zeroizing<[u8; 32]>,
    ) -> Result<(), String> {
        if membership_version == 0 || epoch == 0 {
            return Err(
                "confidential EventService epochs start at committed version/epoch 1".to_owned(),
            );
        }
        if self.verifying_key().is_none() {
            return Err("encrypted publisher has no hybrid identity".to_owned());
        }
        let mut prefixes = self.prefixes.write().await;
        let state = prefixes
            .get_mut(prefix)
            .ok_or_else(|| format!("prefix '{prefix}' not registered"))?;
        if !state.confidential {
            return Err("cannot install an epoch on a public prefix".to_owned());
        }
        if let Some(current) = &state.crypto {
            if membership_version <= current.membership_version || epoch <= current.epoch {
                return Err(
                    "epoch installation must advance membership version and epoch".to_owned(),
                );
            }
        }
        let mut rng = rand::rngs::OsRng;
        let session_id =
            ((u128::from(rng.next_u64()) << 64) | u128::from(rng.next_u64())).to_be_bytes();
        state.crypto = Some(PrefixCrypto {
            membership_version,
            epoch,
            current: epoch_secret,
            sequence: 0,
            session_id,
            created_at: Instant::now(),
        });
        Ok(())
    }

    /// Classical Ristretto wrapping was removed. Epoch release is exclusively
    /// per-member HyKEM/COSE via `GroupKeyRegistry::prepare_change`.
    pub async fn wrap_for_subscriber(
        &self,
        _prefix: &str,
        _subscriber_pubkey: &[u8; 32],
    ) -> Result<Vec<u8>, String> {
        Err("classical event key wrapping removed; use controller HyKEM epoch grants".to_owned())
    }

    pub async fn wrap_for_new_subscribers(
        &self,
        _prefix: &str,
        _subscriber_pubkeys: &[[u8; 32]],
    ) -> Result<Vec<([u8; 32], Vec<u8>)>, String> {
        Err("classical event key wrapping removed; use controller HyKEM epoch grants".to_owned())
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

        if !state.confidential {
            return state.moq.publish_raw(topic, payload);
        }
        let crypto = state.crypto.as_mut().ok_or_else(|| {
            anyhow!("confidential prefix '{prefix}' has no committed controller epoch installed")
        })?;
        let caller = match &self.publisher_identity.did {
            Some(did) => Subject::new(did.clone()),
            None => Subject::anonymous(),
        };
        if !self.authz.can_publish(&caller, &prefix) {
            return Err(anyhow!(
                "publish denied by event-plane authz for prefix '{prefix}'"
            ));
        }
        let signing_key = self
            .signing_key
            .as_ref()
            .ok_or_else(|| anyhow!("encrypted privacy mode requires an Ed25519 signing key"))?;
        let pq_signing_key = self
            .pq_signing_key
            .as_ref()
            .ok_or_else(|| anyhow!("encrypted privacy mode requires an ML-DSA-65 signing key"))?;
        let publisher_pubkey = signing_key.verifying_key().to_bytes();
        let timestamp = now_millis();
        let sequence = crypto
            .sequence
            .checked_add(1)
            .ok_or_else(|| anyhow!("event sequence exhausted; rotate epoch"))?;
        let (tag, ciphertext, nonce, commitment) = encrypt_epoch_event(
            &crypto.current,
            &prefix,
            &publisher_pubkey,
            &crypto.session_id,
            crypto.membership_version,
            crypto.epoch,
            sequence,
            payload,
        )
        .map_err(|e| anyhow!(e))?;
        let sig_message = build_epoch_event_sig_message(
            topic,
            payload,
            timestamp,
            &crypto.session_id,
            crypto.membership_version,
            crypto.epoch,
            sequence,
        );
        let signature = sign_composite(
            signing_key,
            Some(pq_signing_key),
            &sig_message,
            &event_attestation_aad(),
        )?;
        crypto.sequence = sequence;
        let lk_tag = match self.privacy_mode {
            EventPrivacy::LimitedKnowledge => {
                keyed_mac(&crypto.current, prefix.as_bytes())[..16].to_vec()
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
            signature,
            publisher_pubkey,
            timestamp,
            session_id: crypto.session_id,
            membership_version: crypto.membership_version,
            epoch: crypto.epoch,
            sequence,
        };
        state.moq.publish_raw(topic, &encrypted.encode_body())
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
        let age = crypto.created_at.elapsed();
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

    /// EventService cannot rotate or distribute epochs by itself. The
    /// controller must prepare+commit through `GroupKeyRegistry`, deliver each
    /// HyKEM grant out of band, then call [`Self::install_committed_epoch`].
    pub async fn rotate_key(
        &self,
        _prefix: &str,
        _effective_delay: Duration,
    ) -> Result<RotationResult, String> {
        Err("controller-managed epochs cannot be rotated by EventPublisher".to_owned())
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

// ============================================================================
// KeyRing + RekeyEvent — publisher-side state (EventService-specific; the
// reusable keyable-group primitive is `crypto::group_key::GroupKeyRegistry`).
// ============================================================================

// ============================================================================
// EventSubscriber — the canonical broadcast subscriber
// ============================================================================

/// One installed controller epoch.
struct InstalledEpoch {
    membership_version: u64,
    epoch: u64,
    key: Zeroizing<[u8; 32]>,
}

struct PriorEpoch {
    installed: InstalledEpoch,
    grace_until: Instant,
    /// Only objects the publisher had already issued before the transition are
    /// eligible for grace. The controller supplies this authenticated cutoff.
    last_issued_sequence: u64,
}

/// Externally anchored controller-authorized hybrid publisher identity.
pub struct EventPublisherAnchor {
    pub ed25519: ed25519_dalek::VerifyingKey,
    pub ml_dsa_65: crate::crypto::pq::MlDsaVerifyingKey,
}

impl EventPublisherAnchor {
    fn matches(&self, other: &Self) -> bool {
        self.ed25519 == other.ed25519
            && ml_dsa_vk_bytes(&self.ml_dsa_65) == ml_dsa_vk_bytes(&other.ml_dsa_65)
    }
}

struct SubscriberPrefixState {
    current: InstalledEpoch,
    previous: Option<PriorEpoch>,
    publisher: EventPublisherAnchor,
    seen_objects: HashSet<(u64, [u8; 16], u64)>,
    seen_nonces: HashSet<([u8; 16], [u8; 12])>,
}

/// A pending controller epoch notification.
#[derive(Debug)]
pub struct RekeyEvent {
    pub prefix: String,
    pub membership_version: u64,
    pub epoch: u64,
    pub effective_at: Instant,
}

/// Canonical EventService subscriber. Public tracks pass through. Confidential
/// tracks are installed only from per-member HyKEM/COSE controller grants.
pub struct EventSubscriber {
    inner: MoqEventSubscriber,
    prefixes: Arc<RwLock<HashMap<String, SubscriberPrefixState>>>,
}

impl EventSubscriber {
    pub fn new() -> Result<Self> {
        Ok(Self {
            inner: MoqEventSubscriber::new(),
            prefixes: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn subscribe(&mut self, pattern: &str) -> Result<()> {
        self.inner.subscribe(pattern)
    }

    pub fn subscribe_all(&mut self) -> Result<()> {
        self.inner.subscribe_all()
    }

    pub fn unsubscribe(&mut self, pattern: &str) -> Result<()> {
        self.inner.unsubscribe(pattern)
    }

    pub fn subscribe_oid(&mut self, oid: &str) -> Result<()> {
        self.inner.subscribe_oid(oid)
    }

    pub fn with_backfill(&mut self, mode: BackfillMode) -> Result<()> {
        self.inner.with_backfill(mode)
    }

    pub fn with_qos(&mut self, qos: crate::stream_info::StreamOpt) -> Result<()> {
        self.inner.with_qos(qos)
    }

    pub fn with_resume_from(&mut self, sequence: u64) -> Result<()> {
        self.inner.with_resume_from(sequence)
    }

    pub fn last_sequence(&self) -> u64 {
        self.inner.last_sequence()
    }

    pub fn dropped_count(&self) -> u64 {
        self.inner.dropped_count()
    }

    /// Install a per-member epoch grant after policy/accepted-state admission.
    ///
    /// `prior_last_issued_sequence` is mandatory for an advance and bounds old
    /// epoch grace to objects issued before the transition. Grace is clamped to
    /// [`DEFAULT_SUBSCRIBER_GRACE_PERIOD`]; security-triggered transitions pass
    /// `Some(Duration::ZERO)`.
    pub async fn install_epoch_grant(
        &self,
        prefix: &str,
        recipient: &RecipientKeypair,
        grant: &EpochGrant,
        publisher: EventPublisherAnchor,
        prior_last_issued_sequence: Option<u64>,
        grace_period: Option<Duration>,
    ) -> Result<(), String> {
        if prefix.is_empty() || grant.membership_version == 0 || grant.epoch == 0 {
            return Err("invalid confidential event epoch coordinates".to_owned());
        }
        if grant.expires_at_millis <= now_millis() {
            return Err("epoch grant is expired".to_owned());
        }
        if grant.sender_ed25519 != publisher.ed25519.to_bytes()
            || grant.sender_ml_dsa_65 != ml_dsa_vk_bytes(&publisher.ml_dsa_65)
            || grant.sender_did.is_empty()
            || grant.retention_policy.is_empty()
            || grant.opaque_routing_policy.is_empty()
        {
            return Err(
                "publisher anchor or relay profile does not match the authenticated epoch grant"
                    .to_owned(),
            );
        }
        let key = open_epoch_grant(recipient, grant)?;
        let mut prefixes = self.prefixes.write().await;
        match prefixes.get_mut(prefix) {
            None => {
                if prior_last_issued_sequence.is_some() {
                    return Err("initial epoch install must not claim a prior cutoff".to_owned());
                }
                prefixes.insert(
                    prefix.to_owned(),
                    SubscriberPrefixState {
                        current: InstalledEpoch {
                            membership_version: grant.membership_version,
                            epoch: grant.epoch,
                            key,
                        },
                        previous: None,
                        publisher,
                        seen_objects: HashSet::new(),
                        seen_nonces: HashSet::new(),
                    },
                );
            }
            Some(state) => {
                if !state.publisher.matches(&publisher) {
                    return Err(
                        "publisher anchor changed without a controller-authorized rotation"
                            .to_owned(),
                    );
                }
                if grant.membership_version != state.current.membership_version + 1
                    || grant.epoch != state.current.epoch + 1
                {
                    return Err(
                        "epoch grant is unknown, stale, skipped, or future-before-install"
                            .to_owned(),
                    );
                }
                let last_issued_sequence = prior_last_issued_sequence.ok_or_else(|| {
                    "epoch advance requires a prior last-issued cutoff".to_owned()
                })?;
                let grace = grace_period
                    .unwrap_or(DEFAULT_SUBSCRIBER_GRACE_PERIOD)
                    .min(DEFAULT_SUBSCRIBER_GRACE_PERIOD);
                let old = std::mem::replace(
                    &mut state.current,
                    InstalledEpoch {
                        membership_version: grant.membership_version,
                        epoch: grant.epoch,
                        key,
                    },
                );
                state.previous = Some(PriorEpoch {
                    installed: old,
                    grace_until: Instant::now() + grace,
                    last_issued_sequence,
                });
                state.seen_objects.retain(|(epoch, _, _)| {
                    *epoch == state.current.epoch || *epoch + 1 == state.current.epoch
                });
                state.seen_nonces.clear();
            }
        }
        Ok(())
    }

    /// Removed classical admission API: an Ed25519/Ristretto key can never be
    /// substituted for an accepted `#mesh-kem` recipient.
    pub async fn join_prefix(
        &self,
        _prefix: &str,
        _publisher_dh_pubkey: &[u8; 32],
        _wrapped_key_blob: &[u8],
    ) -> Result<[u8; 32], String> {
        Err("classical event join removed; install a controller HyKEM epoch grant".to_owned())
    }

    pub async fn handle_rekey(
        &self,
        _prefix: &str,
        _wrapped_blobs: &[Vec<u8>],
        _new_publisher_dh_pubkey: &[u8; 32],
        _effective_at: Instant,
    ) -> Result<(), String> {
        Err("classical event rekey removed; install a controller HyKEM epoch grant".to_owned())
    }

    pub async fn promote_key(
        &self,
        _prefix: &str,
        _new_key: Zeroizing<[u8; 32]>,
        _grace_period: Option<Duration>,
    ) -> Result<(), String> {
        Err("raw symmetric epoch promotion removed; install an authenticated grant".to_owned())
    }

    pub async fn gc_expired_keys(&self) {
        let mut prefixes = self.prefixes.write().await;
        for state in prefixes.values_mut() {
            if state
                .previous
                .as_ref()
                .is_some_and(|previous| Instant::now() >= previous.grace_until)
            {
                state.previous = None;
                let current_epoch = state.current.epoch;
                state
                    .seen_objects
                    .retain(|(epoch, _, _)| *epoch == current_epoch);
            }
        }
    }

    /// Receive the next event.
    ///
    /// Prefixes without an installed grant remain byte-identical `Public`
    /// passthrough. For an installed confidential prefix, the wire frame must
    /// match a current or bounded prior epoch, use the required nonce domain,
    /// decrypt under the sender/track key, and verify against the externally
    /// anchored Ed25519 + ML-DSA-65 publisher identity. Invalid frames are
    /// dropped and the loop continues, matching best-effort event delivery.
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
        let mut prefixes = self.prefixes.write().await;
        let Some(state) = prefixes.get_mut(prefix) else {
            return FrameOutcome::Passthrough;
        };
        let encrypted = match EncryptedEvent::decode_body(topic, raw) {
            Ok(event) => event,
            Err(error) => return FrameOutcome::Drop(format!("decode: {error}")),
        };
        if encrypted.publisher_pubkey != state.publisher.ed25519.to_bytes() {
            return FrameOutcome::Drop(
                "publisher key is not the controller-authorized anchor".to_owned(),
            );
        }
        if state
            .seen_objects
            .contains(&(encrypted.epoch, encrypted.session_id, encrypted.sequence))
        {
            return FrameOutcome::Drop("replayed event object".to_owned());
        }
        if state
            .seen_nonces
            .contains(&(encrypted.session_id, encrypted.nonce))
        {
            return FrameOutcome::Drop("reused event nonce".to_owned());
        }

        let epoch_key = if encrypted.membership_version == state.current.membership_version
            && encrypted.epoch == state.current.epoch
        {
            &state.current.key
        } else if let Some(previous) = &state.previous {
            if encrypted.membership_version == previous.installed.membership_version
                && encrypted.epoch == previous.installed.epoch
                && Instant::now() < previous.grace_until
                && encrypted.sequence <= previous.last_issued_sequence
            {
                &previous.installed.key
            } else {
                return FrameOutcome::Drop(
                    "unknown, future-before-install, retired, or non-in-flight epoch object"
                        .to_owned(),
                );
            }
        } else {
            return FrameOutcome::Drop(
                "unknown, future-before-install, or retired epoch object".to_owned(),
            );
        };
        let expected_nonce = match derive_event_nonce(encrypted.sequence) {
            Ok(nonce) => nonce,
            Err(error) => return FrameOutcome::Drop(error),
        };
        if encrypted.nonce != expected_nonce {
            return FrameOutcome::Drop("event nonce is outside the sender/track domain".to_owned());
        }
        let plaintext = match decrypt_epoch_event(
            epoch_key,
            prefix,
            &encrypted.publisher_pubkey,
            &encrypted.session_id,
            encrypted.membership_version,
            encrypted.epoch,
            encrypted.sequence,
            &encrypted.nonce,
            &encrypted.tag,
            &encrypted.ciphertext,
            &encrypted.key_commitment,
        ) {
            Ok(plaintext) => plaintext,
            Err(error) => return FrameOutcome::Drop(format!("decrypt: {error}")),
        };
        let signed = build_epoch_event_sig_message(
            topic,
            &plaintext,
            encrypted.timestamp,
            &encrypted.session_id,
            encrypted.membership_version,
            encrypted.epoch,
            encrypted.sequence,
        );
        if let Err(error) = verify_composite(
            &encrypted.signature,
            &state.publisher.ed25519,
            Some(&state.publisher.ml_dsa_65),
            &signed,
            &event_attestation_aad(),
            true,
        ) {
            return FrameOutcome::Drop(format!("hybrid publisher attestation failed: {error}"));
        }
        state
            .seen_objects
            .insert((encrypted.epoch, encrypted.session_id, encrypted.sequence));
        state
            .seen_nonces
            .insert((encrypted.session_id, encrypted.nonce));
        FrameOutcome::Decoded(plaintext)
    }
}

#[derive(Debug)]
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
    use crate::crypto::group_key::{
        recipient_key_id, ControllerBinding, GroupKeyRegistry, GroupMembership, GroupRef,
        MembershipChange, MembershipResolver,
    };
    use crate::crypto::hybrid_kem::{generate_recipient, RecipientKeypair, SuiteId};
    use ml_dsa::Keypair;

    struct AllowExact;
    impl MembershipResolver for AllowExact {
        fn resolve(&self, requested: &GroupMembership) -> Result<GroupMembership, String> {
            Ok(requested.clone())
        }
    }

    fn signing_key() -> SigningKey {
        let mut secret = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut secret);
        SigningKey::from_bytes(&secret)
    }

    fn member(recipient: &RecipientKeypair, did: &str, blind: u8) -> GroupMembership {
        let public = recipient.public();
        GroupMembership {
            group_uri: "at://did:web:controller/ai.hyprstream.event.group/g1".to_owned(),
            subject_did: did.to_owned(),
            accepted_state: b"cid512:accepted-a".to_vec(),
            capability: format!("event:g1:subscribe:{did}").into_bytes(),
            recipient_key_id: recipient_key_id(&public),
            recipient: public,
            blinded_routing_key: [blind; 32],
            expires_at_millis: i64::MAX,
        }
    }

    async fn grant_for(
        recipient: &RecipientKeypair,
        ed: &SigningKey,
        pq: &MlDsaSigningKey,
    ) -> (EpochGrant, Zeroizing<[u8; 32]>) {
        let group = GroupRef::new(
            "at://did:web:controller/ai.hyprstream.event.group/g1",
            "ks-0",
        );
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowExact).unwrap();
        registry
            .register_group(
                group.clone(),
                ControllerBinding {
                    controller_did: "did:web:controller".to_owned(),
                    accepted_state: b"cid512:accepted-a".to_vec(),
                    sender_did: "did:web:publisher".to_owned(),
                    sender_ed25519: ed.verifying_key().to_bytes(),
                    sender_ml_dsa_65: crate::crypto::pq::ml_dsa_sk_to_vk_bytes(pq),
                    retention_policy: b"retention:bounded-v1".to_vec(),
                    opaque_routing_policy: b"routing:stock-moq-opaque-v1".to_vec(),
                    expires_at_millis: i64::MAX,
                },
            )
            .await
            .unwrap();
        let prepared = registry
            .prepare_change(
                &group,
                MembershipChange::Join(member(recipient, "did:web:member", 7)),
                "ks-1",
            )
            .await
            .unwrap();
        let grant = prepared.grants[0].clone();
        registry
            .commit_prepared(&group, prepared.membership_version, prepared.epoch)
            .await
            .unwrap();
        (
            grant,
            Zeroizing::new(registry.k_group(&group).await.unwrap()),
        )
    }

    fn event(
        epoch_key: &[u8; 32],
        ed_sk: &SigningKey,
        pq_sk: &MlDsaSigningKey,
        payload: &[u8],
        sequence: u64,
    ) -> EncryptedEvent {
        event_at(epoch_key, ed_sk, pq_sk, payload, 1, 1, sequence)
    }

    fn event_at(
        epoch_key: &[u8; 32],
        ed_sk: &SigningKey,
        pq_sk: &MlDsaSigningKey,
        payload: &[u8],
        membership_version: u64,
        epoch: u64,
        sequence: u64,
    ) -> EncryptedEvent {
        let topic = "worker.sandbox1.started";
        let prefix = "worker";
        let kid = ed_sk.verifying_key().to_bytes();
        let session_digest = blake3::hash(&epoch.to_be_bytes());
        let session_id: [u8; 16] = session_digest.as_bytes()[..16]
            .try_into()
            .expect("BLAKE3 output contains a 16-byte test session ID");
        let (tag, ciphertext, nonce, key_commitment) = encrypt_epoch_event(
            epoch_key,
            prefix,
            &kid,
            &session_id,
            membership_version,
            epoch,
            sequence,
            payload,
        )
        .unwrap();
        let timestamp = 1_700_000_000_000;
        let signed = build_epoch_event_sig_message(
            topic,
            payload,
            timestamp,
            &session_id,
            membership_version,
            epoch,
            sequence,
        );
        let signature =
            sign_composite(ed_sk, Some(pq_sk), &signed, &event_attestation_aad()).unwrap();
        EncryptedEvent {
            topic: topic.to_owned(),
            tag,
            ciphertext,
            nonce,
            key_commitment,
            lk_tag: Vec::new(),
            signature,
            publisher_pubkey: kid,
            timestamp,
            session_id,
            membership_version,
            epoch,
            sequence,
        }
    }

    fn anchor(ed_sk: &SigningKey, pq_sk: &MlDsaSigningKey) -> EventPublisherAnchor {
        EventPublisherAnchor {
            ed25519: ed_sk.verifying_key(),
            ml_dsa_65: pq_sk.verifying_key().clone(),
        }
    }

    #[test]
    fn encrypted_event_wire_roundtrip_and_trailing_bytes_rejected() {
        let ed = signing_key();
        let pq = crate::node_identity::derive_mesh_mldsa_key(&ed);
        let event = event(&[9; 32], &ed, &pq, b"payload", 4);
        let encoded = event.encode_body();
        let decoded = EncryptedEvent::decode_body(&event.topic, &encoded).unwrap();
        assert_eq!(decoded.membership_version, 1);
        assert_eq!(decoded.epoch, 1);
        assert_eq!(decoded.sequence, 4);
        assert_eq!(decoded.signature, event.signature);
        let mut trailing = encoded;
        trailing.push(0);
        assert!(EncryptedEvent::decode_body(&event.topic, &trailing).is_err());
    }

    #[tokio::test]
    async fn grant_install_relay_forward_and_hybrid_verify() {
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let ed = signing_key();
        let pq = crate::node_identity::derive_mesh_mldsa_key(&ed);
        let (grant, key) = grant_for(&recipient, &ed, &pq).await;
        let subscriber = EventSubscriber::new().unwrap();
        subscriber
            .install_epoch_grant("worker", &recipient, &grant, anchor(&ed, &pq), None, None)
            .await
            .unwrap();

        let object = event(&key, &ed, &pq, b"relay cannot read this", 1);
        let wire = object.encode_body();
        // A stock relay performs only byte-identical forward/cache operations.
        let relayed = wire.clone();
        assert_eq!(relayed, wire);
        assert!(!relayed
            .windows(b"relay cannot read this".len())
            .any(|window| window == b"relay cannot read this"));
        match subscriber.decode_frame(&object.topic, &relayed).await {
            FrameOutcome::Decoded(payload) => assert_eq!(payload, b"relay cannot read this"),
            other => panic!("expected decoded object, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stock_relay_multi_member_revocation_conformance() {
        let member_a = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let member_b = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let relay_recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let ed = signing_key();
        let pq = crate::node_identity::derive_mesh_mldsa_key(&ed);
        let group = GroupRef::new(
            "at://did:web:controller/ai.hyprstream.event.group/g1",
            "ks-0",
        );
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowExact).unwrap();
        registry
            .register_group(
                group.clone(),
                ControllerBinding {
                    controller_did: "did:web:controller".to_owned(),
                    accepted_state: b"cid512:accepted-a".to_vec(),
                    sender_did: "did:web:publisher".to_owned(),
                    sender_ed25519: ed.verifying_key().to_bytes(),
                    sender_ml_dsa_65: crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq),
                    retention_policy: b"retention:bounded-v1".to_vec(),
                    opaque_routing_policy: b"routing:stock-moq-opaque-v1".to_vec(),
                    expires_at_millis: i64::MAX,
                },
            )
            .await
            .unwrap();

        let joined_a = registry
            .prepare_change(
                &group,
                MembershipChange::Join(member(&member_a, "did:web:member-a", 7)),
                "ks-1",
            )
            .await
            .unwrap();
        registry
            .commit_prepared(&group, joined_a.membership_version, joined_a.epoch)
            .await
            .unwrap();
        let joined_b = registry
            .prepare_change(
                &group,
                MembershipChange::Join(member(&member_b, "did:web:member-b", 8)),
                "ks-2",
            )
            .await
            .unwrap();
        assert_eq!(joined_b.grants.len(), 2);
        registry
            .commit_prepared(&group, joined_b.membership_version, joined_b.epoch)
            .await
            .unwrap();
        let key_epoch_2 = Zeroizing::new(registry.k_group(&group).await.unwrap());

        let grant_a_2 = joined_b
            .grants
            .iter()
            .find(|grant| grant.subject_did == "did:web:member-a")
            .unwrap();
        let grant_b_2 = joined_b
            .grants
            .iter()
            .find(|grant| grant.subject_did == "did:web:member-b")
            .unwrap();
        assert!(crate::crypto::group_key::open_epoch_grant(&relay_recipient, grant_a_2).is_err());
        let subscriber_a = EventSubscriber::new().unwrap();
        let subscriber_b = EventSubscriber::new().unwrap();
        subscriber_a
            .install_epoch_grant("worker", &member_a, grant_a_2, anchor(&ed, &pq), None, None)
            .await
            .unwrap();
        subscriber_b
            .install_epoch_grant("worker", &member_b, grant_b_2, anchor(&ed, &pq), None, None)
            .await
            .unwrap();

        let before = event_at(&key_epoch_2, &ed, &pq, b"members only", 2, 2, 1);
        let wire = before.encode_body();
        let relayed = wire.clone();
        assert_eq!(relayed, wire);
        assert!(!relayed
            .windows(b"members only".len())
            .any(|window| window == b"members only"));
        for subscriber in [&subscriber_a, &subscriber_b] {
            assert!(matches!(
                subscriber.decode_frame(&before.topic, &relayed).await,
                FrameOutcome::Decoded(ref payload) if payload == b"members only"
            ));
        }

        let revoked = registry
            .prepare_change(
                &group,
                MembershipChange::Revoke {
                    subject_did: "did:web:member-b".to_owned(),
                },
                "ks-3",
            )
            .await
            .unwrap();
        assert_eq!(revoked.grants.len(), 1);
        assert_eq!(revoked.grants[0].subject_did, "did:web:member-a");
        registry
            .commit_prepared(&group, revoked.membership_version, revoked.epoch)
            .await
            .unwrap();
        assert!(!registry.contains_member(&group, "did:web:member-b").await);
        let key_epoch_3 = Zeroizing::new(registry.k_group(&group).await.unwrap());
        subscriber_a
            .install_epoch_grant(
                "worker",
                &member_a,
                &revoked.grants[0],
                anchor(&ed, &pq),
                Some(1),
                Some(Duration::ZERO),
            )
            .await
            .unwrap();

        let after = event_at(&key_epoch_3, &ed, &pq, b"after revoke", 3, 3, 1);
        let relayed_after = after.encode_body();
        assert!(!relayed_after
            .windows(b"after revoke".len())
            .any(|window| window == b"after revoke"));
        assert!(matches!(
            subscriber_a.decode_frame(&after.topic, &relayed_after).await,
            FrameOutcome::Decoded(ref payload) if payload == b"after revoke"
        ));
        assert!(matches!(
            subscriber_b.decode_frame(&after.topic, &relayed_after).await,
            FrameOutcome::Drop(ref reason) if reason.contains("future-before-install")
        ));
    }

    #[tokio::test]
    async fn replay_nonce_mutation_and_stripped_pq_fail_closed() {
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let ed = signing_key();
        let pq = crate::node_identity::derive_mesh_mldsa_key(&ed);
        let (grant, key) = grant_for(&recipient, &ed, &pq).await;
        let subscriber = EventSubscriber::new().unwrap();
        subscriber
            .install_epoch_grant("worker", &recipient, &grant, anchor(&ed, &pq), None, None)
            .await
            .unwrap();

        let first = event(&key, &ed, &pq, b"one", 1);
        assert!(matches!(
            subscriber
                .decode_frame(&first.topic, &first.encode_body())
                .await,
            FrameOutcome::Decoded(_)
        ));
        assert!(matches!(
            subscriber.decode_frame(&first.topic, &first.encode_body()).await,
            FrameOutcome::Drop(reason) if reason.contains("replayed")
        ));

        let mut bad_nonce = event(&key, &ed, &pq, b"two", 2);
        bad_nonce.nonce[0] ^= 1;
        assert!(matches!(
            subscriber.decode_frame(&bad_nonce.topic, &bad_nonce.encode_body()).await,
            FrameOutcome::Drop(reason) if reason.contains("nonce")
        ));

        let mut stripped = event(&key, &ed, &pq, b"three", 3);
        let signed = build_epoch_event_sig_message(
            &stripped.topic,
            b"three",
            stripped.timestamp,
            &stripped.session_id,
            1,
            1,
            3,
        );
        stripped.signature = sign_composite(&ed, None, &signed, &event_attestation_aad()).unwrap();
        assert!(matches!(
            subscriber.decode_frame(&stripped.topic, &stripped.encode_body()).await,
            FrameOutcome::Drop(reason) if reason.contains("attestation")
        ));
    }

    #[tokio::test]
    async fn duplicate_registration_preserves_live_crypto_domain() {
        let _ =
            crate::moq_event::init_global_moq_event_origin(crate::moq_event::MoqEventOrigin::new());
        let prefix = format!("hardening-duplicate-{}", rand::random::<u64>());
        let publisher = EventPublisher::new_encrypted(
            signing_key(),
            EventPrivacy::ZeroKnowledge,
            RekeyPolicy::Immediate,
        )
        .unwrap();
        publisher.register_prefix(&prefix).await.unwrap();
        publisher
            .install_committed_epoch(&prefix, 1, 1, Zeroizing::new([0x61; 32]))
            .await
            .unwrap();
        {
            let mut prefixes = publisher.prefixes.write().await;
            prefixes
                .get_mut(&prefix)
                .unwrap()
                .crypto
                .as_mut()
                .unwrap()
                .sequence = 7;
        }
        let (session_before, sequence_before) = {
            let prefixes = publisher.prefixes.read().await;
            let crypto = prefixes.get(&prefix).unwrap().crypto.as_ref().unwrap();
            (crypto.session_id, crypto.sequence)
        };

        let error = publisher.register_prefix(&prefix).await.unwrap_err();
        assert!(error.contains("already registered"));
        let prefixes = publisher.prefixes.read().await;
        let crypto = prefixes.get(&prefix).unwrap().crypto.as_ref().unwrap();
        assert_eq!(crypto.session_id, session_before);
        assert_eq!(crypto.sequence, sequence_before);
    }

    #[tokio::test]
    async fn independent_installations_cannot_reuse_key_nonce_pair() {
        let _ =
            crate::moq_event::init_global_moq_event_origin(crate::moq_event::MoqEventOrigin::new());
        let prefix = format!("hardening-failover-{}", rand::random::<u64>());
        let signing = signing_key();
        let secret = signing.to_bytes();
        let publisher_a = EventPublisher::new_encrypted(
            SigningKey::from_bytes(&secret),
            EventPrivacy::ZeroKnowledge,
            RekeyPolicy::Immediate,
        )
        .unwrap();
        let publisher_b = EventPublisher::new_encrypted(
            SigningKey::from_bytes(&secret),
            EventPrivacy::ZeroKnowledge,
            RekeyPolicy::Immediate,
        )
        .unwrap();
        publisher_a.register_prefix(&prefix).await.unwrap();
        publisher_b.register_prefix(&prefix).await.unwrap();
        let epoch_secret = [0x71u8; 32];
        publisher_a
            .install_committed_epoch(&prefix, 1, 1, Zeroizing::new(epoch_secret))
            .await
            .unwrap();
        publisher_b
            .install_committed_epoch(&prefix, 1, 1, Zeroizing::new(epoch_secret))
            .await
            .unwrap();

        let session_a = publisher_a
            .prefixes
            .read()
            .await
            .get(&prefix)
            .unwrap()
            .crypto
            .as_ref()
            .unwrap()
            .session_id;
        let session_b = publisher_b
            .prefixes
            .read()
            .await
            .get(&prefix)
            .unwrap()
            .crypto
            .as_ref()
            .unwrap()
            .session_id;
        let kid = publisher_a.verifying_key().unwrap();
        let key_a = crate::crypto::event_crypto::derive_sender_track_key(
            &epoch_secret,
            &kid,
            &prefix,
            &session_a,
        );
        let key_b = crate::crypto::event_crypto::derive_sender_track_key(
            &epoch_secret,
            &kid,
            &prefix,
            &session_b,
        );
        let nonce = derive_event_nonce(1).unwrap();

        assert_ne!(session_a, session_b);
        assert_ne!((&*key_a, nonce), (&*key_b, nonce));
    }

    #[tokio::test]
    async fn signing_key_cannot_substitute_for_mesh_kem_recipient() {
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let wrong = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let ed = signing_key();
        let pq = crate::node_identity::derive_mesh_mldsa_key(&ed);
        let (grant, _) = grant_for(&recipient, &ed, &pq).await;
        let subscriber = EventSubscriber::new().unwrap();
        assert!(subscriber
            .install_epoch_grant("worker", &wrong, &grant, anchor(&ed, &pq), None, None)
            .await
            .is_err());
        assert!(subscriber
            .join_prefix(
                "worker",
                &ed.verifying_key().to_bytes(),
                &grant.sealed_epoch_secret
            )
            .await
            .is_err());
    }

    #[test]
    fn rekey_policy_and_authz_are_fail_closed() {
        assert!(RekeyPolicy::Scheduled {
            interval: Duration::from_secs(100_000),
        }
        .validate()
        .is_err());
        let deny = DenyAllEventAuthz;
        assert!(!deny.can_publish(&Subject::anonymous(), "worker"));
        assert!(!deny.can_subscribe(&Subject::anonymous(), "worker"));
        assert!(EventPublisher::new_encrypted(
            signing_key(),
            EventPrivacy::Public,
            RekeyPolicy::default(),
        )
        .is_err());
    }
}
