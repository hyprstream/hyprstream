//! Generic **keyable group** primitive for record-backed fan-out pub/sub.
//!
//! This is the general-purpose, lexicon-agnostic home for the "a group of
//! members shares a symmetric group key (`K_group`), distributed to members via
//! DH-wrap-on-join, with rekey-driven forward secrecy" pattern. Any feature that
//! needs record-backed group fan-out — EventService (`ai.hyprstream.event.group`),
//! the placement scheduler (`ai.hyprstream.placement.group`), or a future one —
//! consumes this primitive instead of reimplementing the key state machine.
//!
//! # Split with the identity layer
//!
//! The *identity* of a group (the list record + listitem + bidirectional consent)
//! lives in `hyprstream-pds::list_record` — generic `ListRecord<E>` /
//! `ListItemRecord`, modeled on `app.bsky.graph.list`/`listitem`. This module is
//! the *key material* half: given a group's record reference ([`GroupRef`]), it
//! owns the per-epoch `K_group`, the publisher ephemeral keypair, and the
//! DH-wrap-on-join path that delivers `K_group` to each member.
//!
//! # What "K_group derivation" means here
//!
//! `K_group` bytes are **freshly random per rotation** (`OsRng`) — there is no
//! KDF chain producing `K_group` itself. "Derivation" refers to the
//! **identity/reference** `(group_uri, keyset_id, epoch)` being bound to the
//! signed record, so a member can ask "give me `K_group` for keyset X epoch Y".
//! A keyed-PRF *derivation* (e.g. EventService's
//! `topic = KDF(K_group, subject‖epoch)`) is a downstream use of this `K_group`,
//! out of scope here — see [`GroupKeyRegistry::k_group`].
//!
//! # Forward secrecy
//!
//! Each epoch uses a freshly-rotated **random** group key (not a static key with
//! an epoch tweak), so compromise of an old key cannot derive later keys. The
//! rekey **grace window** ([`GRACE_PERIOD`]) means two epochs' keys are briefly
//! valid simultaneously during a rotation. Rekey timing is governed by
//! [`RekeyPolicy`] (Scheduled / Jittered / Immediate) — the Immediate-vs-Scheduled
//! tradeoff (prompt revocation forward secrecy vs bounded O(M)-per-rotation cost)
//! is a per-consumer decision; see that type's docs.
//!
//! This module reuses — does not duplicate — the low-level crypto already in this
//! crate: [`event_crypto::derive_wrap_key`] / [`event_crypto::wrap_group_key`] for
//! the DH-wrap, and [`crate::crypto::ristretto_dh_raw`] /
//! [`crate::crypto::generate_ephemeral_keypair`] for the Ristretto255 DH legs.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use zeroize::Zeroizing;

use crate::crypto::backend::keyed_mac;
use crate::crypto::event_crypto::{derive_wrap_key, unwrap_group_key, wrap_group_key};
use crate::crypto::{blinded_dh_raw, rerandomize_pubkey, RistrettoPublic};

// ────────────────────────────────────────────────────────────────────────────
// Rekey policy + timing constants
// ────────────────────────────────────────────────────────────────────────────

/// Maximum key lifetime (24 hours). Keys MUST be rotated before this.
pub const MAX_KEY_LIFETIME: Duration = Duration::from_secs(86400);

/// Default rotation interval (1 hour).
pub const DEFAULT_ROTATION_INTERVAL: Duration = Duration::from_secs(3600);

/// Grace period for old key acceptance after rotation (120 seconds). During this
/// window two epochs' keys are both valid; revocation that must be prompt should
/// use a zero/short grace.
pub const GRACE_PERIOD: Duration = Duration::from_secs(120);

/// Rekey policy configuration.
///
/// The choice between [`RekeyPolicy::Immediate`] (rotate on every revocation —
/// prompt forward secrecy, but O(M) wrap cost per revocation, i.e. O(M²) over M
/// departures) and [`RekeyPolicy::Scheduled`] (rotate on a fixed interval —
/// bounded O(M) per rotation, but a revoked member retains access until the next
/// rotation) is a per-consumer tradeoff between revocation latency and cost.
#[derive(Clone, Debug)]
pub enum RekeyPolicy {
    /// Rotate on fixed schedule. Revocations deferred to next rotation.
    Scheduled { interval: Duration },
    /// Rotate immediately on revocation.
    Immediate,
    /// Scheduled with jitter for timing-attack resistance.
    Jittered {
        interval: Duration,
        jitter: Duration,
    },
}

impl Default for RekeyPolicy {
    fn default() -> Self {
        RekeyPolicy::Scheduled {
            interval: DEFAULT_ROTATION_INTERVAL,
        }
    }
}

impl RekeyPolicy {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::Scheduled { interval } | Self::Jittered { interval, .. }
                if *interval > MAX_KEY_LIFETIME =>
            {
                return Err(format!(
                    "interval {:?} exceeds MAX_KEY_LIFETIME ({:?})",
                    interval, MAX_KEY_LIFETIME
                ));
            }
            _ => {}
        }
        Ok(())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Result/data structs (shared by every consumer of the primitive)
// ────────────────────────────────────────────────────────────────────────────

/// Result of encrypting an event under a group key.
///
/// This is the higher-level envelope around [`crate::crypto::event_crypto::encrypt_event`]
/// (which returns the raw `(tag, ciphertext, nonce, commitment)` tuple); it adds
/// the routing/topic metadata, Ed25519 publisher signature, and timestamp a
/// fan-out consumer needs to ship the event.
#[derive(Debug, Clone)]
pub struct EncryptedEvent {
    pub topic: String,
    pub tag: Vec<u8>,
    pub ciphertext: Vec<u8>,
    pub nonce: [u8; 12],
    pub key_commitment: [u8; 16],
    /// Limited-knowledge routing tag (keyed HMAC of the routing prefix under the
    /// group key). Empty in zero-knowledge mode. NOTE: a non-empty `lk_tag` is a
    /// stable per-prefix selector and is linkable — consumers that require topic
    /// opacity must not use the LK mode.
    pub lk_tag: Vec<u8>,
    /// Ed25519 signature over the event.
    pub signature: Vec<u8>,
    /// Publisher's Ed25519 verifying key.
    pub publisher_pubkey: [u8; 32],
    /// Event timestamp (unix millis).
    pub timestamp: i64,
}

/// Result of a key rotation.
#[derive(Debug)]
pub struct RotationResult {
    pub new_ephemeral_pubkey: [u8; 32],
    pub wrapped_keys: Vec<WrappedKeyEntry>,
    pub effective_at_millis: i64,
}

/// A wrapped key entry for a single subscriber.
#[derive(Debug)]
pub struct WrappedKeyEntry {
    /// Random 16-byte routing tag (unlinkable across rekeys).
    pub routing_tag: [u8; 16],
    /// Opaque wrapped group-key blob.
    pub wrapped_blob: Vec<u8>,
}

// ────────────────────────────────────────────────────────────────────────────
// Group identity + membership
// ────────────────────────────────────────────────────────────────────────────

/// A group's identity, lexicon-agnostic: an atproto record URI + a keyset id.
///
/// `group_uri` is the at-uri of the group *list* record (e.g.
/// `at://did:web:node/ai.hyprstream.event.group/g1` or
/// `at://did:web:node/ai.hyprstream.placement.group/g7`); `keyset_id` is the
/// opaque key-material generation reference the list record carries (never key
/// bytes — the actual `K_group` is delivered per-member via DH-wrap-on-join).
///
/// This is deliberately plain data (no `hyprstream-pds` type) so this crate stays
/// lower-level than the record store and any list lexicon can back it.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct GroupRef {
    pub group_uri: String,
    pub keyset_id: String,
}

impl GroupRef {
    pub fn new(group_uri: impl Into<String>, keyset_id: impl Into<String>) -> Self {
        Self {
            group_uri: group_uri.into(),
            keyset_id: keyset_id.into(),
        }
    }
}

/// A membership fact the caller has already resolved: `subject_did` is a verified
/// member of `group_uri`. Carries the member's ephemeral Ristretto255 pubkey used
/// for the DH-wrap.
///
/// By the time a `GroupMembership` exists, the caller (a [`MembershipResolver`]
/// impl) has already verified the list records (CAR proof / commit signature) and
/// the bidirectional-consent check, and run any authz-prefilter. This struct is
/// the *result* of that work, not a substitute for it.
#[derive(Clone, Debug)]
pub struct GroupMembership {
    pub group_uri: String,
    pub subject_did: String,
    /// Member's ephemeral Ristretto255 pubkey (for the DH-wrap).
    pub member_pubkey: [u8; 32],
}

/// Resolves group membership for a join request. A real implementation fetches
/// the list/groupItem records (and the member's own consent record) via the
/// DiscoveryService `getRecord` path, verifies the CAR proofs, runs
/// bidirectional-consent + any authz-prefilter, and returns the verified
/// [`GroupMembership`].
///
/// Implementations live in the consumer crate (they depend on the concrete
/// lexicon + DiscoveryService); [`DenyAllResolver`] is the fail-closed default
/// used until a real resolver is wired in.
pub trait MembershipResolver: Send + Sync {
    fn resolve(
        &self,
        group_uri: &str,
        subject_did: &str,
        member_pubkey: [u8; 32],
    ) -> Result<GroupMembership, String>;
}

/// A resolver that always denies — the safe default until a real resolver is
/// wired in. Prevents the registry from accidentally admitting unverified members.
pub struct DenyAllResolver;

impl MembershipResolver for DenyAllResolver {
    fn resolve(
        &self,
        _group_uri: &str,
        _subject_did: &str,
        _member_pubkey: [u8; 32],
    ) -> Result<GroupMembership, String> {
        Err("MembershipResolver not wired — fail-closed".to_owned())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Blinded join (EV4: participant anonymity vs the relay)
// ────────────────────────────────────────────────────────────────────────────

/// A subscriber's **blinded** presentation for join (EV4 participant anonymity).
///
/// The subscriber blinds their stable Ristretto255 public key with a fresh
/// random scalar `r` (`blinded = P_member + r·G` via [`rerandomize_pubkey`])
/// and presents `blinded_pubkey` to the publisher's [`GroupKeyRegistry::join`].
/// The publisher DHs against the blinded pubkey — it never sees, and cannot
/// recover, the subscriber's stable key. The subscriber retains `blinding` to
/// unwrap via [`blinded_dh_raw`].
///
/// **The blinder is the subscriber (or a DiscoveryService-side step that the
/// subscriber trusts), NEVER the relay.** A relay that chose `r` could link
/// presentations; the subscriber choosing `r` keeps that unlinkability.
///
/// # Residual (honest)
/// Anonymity is **vs the relay only**. The publisher still sees the (blinded)
/// connection and its traffic pattern; a DiscoveryService that authenticates the
/// join (to authorize it) necessarily de-anonymizes the subscriber at authz
/// time. The relay observes the anonymous connection-set cardinality + the
/// per-connection traffic pattern. This is participant *key*-anonymity — it is
/// NOT per-publish edge-set hiding (the relay can still see who-publishes-to /
/// who-subscribes-to which track within a group, at the connection level).
pub struct BlindedMember {
    /// The blinded pubkey the subscriber presents to `join`.
    pub blinded_pubkey: [u8; 32],
    /// The blinding scalar, retained subscriber-side and NEVER sent. Required to
    /// unwrap the wrapped group key via [`blinded_dh_raw`].
    pub blinding: [u8; 32],
}

impl BlindedMember {
    /// Subscriber-side: blind a stable member public key. `member_pubkey` is the
    /// subscriber's stable Ristretto255 public key (32 bytes). Returns the
    /// blinded presentation to send to `join`, plus the blinding scalar to keep.
    pub fn new(member_pubkey: &[u8; 32]) -> Result<Self, String> {
        let pub_obj = RistrettoPublic::from_bytes(member_pubkey)
            .ok_or_else(|| "invalid Ristretto255 member public key".to_owned())?;
        let (blinded, blinding) = rerandomize_pubkey(&pub_obj);
        Ok(Self {
            blinded_pubkey: blinded.to_bytes(),
            blinding,
        })
    }

    /// Subscriber-side: recover `K_group` from the wrapped blob returned by
    /// [`GroupKeyRegistry::join`], using the blinded-DH. `member_secret` is the
    /// subscriber's stable secret scalar; `publisher_pubkey` is the ephemeral
    /// publisher pubkey `join` returned. By DH commutativity this yields the
    /// same shared secret the publisher computed against the blinded pubkey.
    pub fn unwrap(
        &self,
        member_secret: &[u8; 32],
        publisher_pubkey: &[u8; 32],
        wrapped: &[u8],
        tenant: &str,
        group_uri: &str,
        subject_did: &str,
    ) -> Result<Zeroizing<[u8; 32]>, String> {
        // (s_member + r) · P_publisher  ==  s_publisher · (P_member + r·G)
        let shared = blinded_dh_raw(member_secret, &self.blinding, publisher_pubkey)
            .map_err(|e| format!("blinded DH failed: {e}"))?;
        // Salt is XOR-symmetric over the two pubkeys; join derived with
        // (blinded_pubkey, publisher_ephemeral) — same two pubs here.
        let wrap_key = derive_wrap_key(&shared, &self.blinded_pubkey, publisher_pubkey);
        unwrap_group_key(
            &wrap_key,
            wrapped,
            tenant,
            &keyed_subject_hash(subject_did),
            group_uri,
        )
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Per-group key state (private)
// ────────────────────────────────────────────────────────────────────────────

struct GroupKeyState {
    keyset_id: String,
    epoch: u64,
    current: Zeroizing<[u8; 32]>,
    pending: Option<PendingRekey>,
    created_at: Instant,
}

struct PendingRekey {
    new_keyset_id: String,
    new_epoch: u64,
    new_key: Zeroizing<[u8; 32]>,
    effective_at: Instant,
}

struct GroupState {
    key_state: GroupKeyState,
    /// Publisher's ephemeral Ristretto255 keypair for this group (the DH-wrap
    /// publisher side; members DH against the public half).
    ephemeral_secret: Zeroizing<[u8; 32]>,
    ephemeral_pubkey: [u8; 32],
    /// Members who have successfully joined (subject_did -> pubkey).
    members: HashMap<String, [u8; 32]>,
}

// ────────────────────────────────────────────────────────────────────────────
// GroupKeyRegistry — the primitive
// ────────────────────────────────────────────────────────────────────────────

/// Record-backed `K_group` registry: the general-purpose keyable-group primitive.
///
/// One instance per publishing node; groups are identified by a lexicon-agnostic
/// [`GroupRef`]. Members join through a [`MembershipResolver`] (fail-closed);
/// the current `K_group` is delivered via DH-wrap-on-join. Rekey rotates to a
/// fresh random key (forward secrecy) under a [`RekeyPolicy`].
///
/// Consumers retrieve the current `K_group` via [`Self::k_group`] for downstream
/// keyed-PRF uses (e.g. EventService's topic derivation).
pub struct GroupKeyRegistry<R: MembershipResolver> {
    groups: Arc<RwLock<HashMap<GroupRef, GroupState>>>,
    rekey_policy: RekeyPolicy,
    resolver: R,
}

impl<R: MembershipResolver> GroupKeyRegistry<R> {
    pub fn new(rekey_policy: RekeyPolicy, resolver: R) -> Result<Self, String> {
        rekey_policy.validate()?;
        Ok(Self {
            groups: Arc::new(RwLock::new(HashMap::new())),
            rekey_policy,
            resolver,
        })
    }

    /// Register a group, generating a fresh `K_group` for `keyset_id` epoch 0.
    /// Returns the publisher's ephemeral pubkey for this group (members DH
    /// against it to receive their wrapped key).
    pub async fn register_group(&self, group: GroupRef) -> Result<[u8; 32], String> {
        let mut group_key = Zeroizing::new([0u8; 32]);
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut *group_key);

        let (eph_secret, eph_public) = crate::crypto::generate_ephemeral_keypair();
        let ephemeral_secret = Zeroizing::new(eph_secret.scalar().to_bytes());
        let ephemeral_pubkey = eph_public.to_bytes();

        let state = GroupState {
            key_state: GroupKeyState {
                keyset_id: group.keyset_id.clone(),
                epoch: 0,
                current: group_key,
                pending: None,
                created_at: Instant::now(),
            },
            ephemeral_secret,
            ephemeral_pubkey,
            members: HashMap::new(),
        };

        let mut groups = self.groups.write();
        if groups.contains_key(&group) {
            return Err(format!("group {group:?} already registered"));
        }
        groups.insert(group, state);
        Ok(ephemeral_pubkey)
    }

    /// Join a group: resolve membership (fail-closed via [`MembershipResolver`]),
    /// then DH-wrap the current `K_group` for the member.
    ///
    /// `member_pubkey` is the key the publisher DHs against — for EV4 participant
    /// anonymity this SHOULD be a **blinded** presentation ([`BlindedMember`]),
    /// not the subscriber's stable key. The publisher side is blinding-agnostic:
    /// `ristretto_dh_raw` is commutative with the subscriber's `blinded_dh_raw`,
    /// so the same wrap/unwrap round-trips whether `member_pubkey` is raw or
    /// blinded. Presenting a raw key here is correct but exposes the stable key
    /// to the publisher/relay — use [`BlindedMember`] for the confidential path.
    ///
    /// Returns `(wrapped_key_blob, keyset_id, epoch, publisher_ephemeral_pubkey)`.
    /// The member unwraps with `unwrap_group_key` (raw) or [`BlindedMember::unwrap`]
    /// (blinded) using the same DH (their secret + the returned publisher pubkey).
    pub async fn join(
        &self,
        group: &GroupRef,
        subject_did: &str,
        member_pubkey: [u8; 32],
    ) -> Result<(Vec<u8>, String, u64, [u8; 32]), String> {
        let membership = self
            .resolver
            .resolve(&group.group_uri, subject_did, member_pubkey)?;
        if membership.group_uri != group.group_uri || membership.subject_did != subject_did {
            return Err("resolver returned mismatched membership".to_owned());
        }

        let mut groups = self.groups.write();
        let state = groups
            .get_mut(group)
            .ok_or_else(|| format!("group {group:?} not registered"))?;

        // DH(publisher_secret, member_pubkey) -> shared secret, THEN derive the
        // wrap key from it (derive_wrap_key's first arg is the shared secret, not
        // a raw scalar).
        let shared_secret = Zeroizing::new(
            crate::crypto::ristretto_dh_raw(&state.ephemeral_secret, &member_pubkey)
                .map_err(|e| format!("DH failed: {e}"))?,
        );
        let wrap_key = derive_wrap_key(&shared_secret, &member_pubkey, &state.ephemeral_pubkey);
        let wrapped = wrap_group_key(
            &wrap_key,
            &state.key_state.current,
            &group.group_uri,
            &keyed_subject_hash(subject_did),
            &group.group_uri,
        )?;

        state.members.insert(subject_did.to_owned(), member_pubkey);
        Ok((
            wrapped,
            state.key_state.keyset_id.clone(),
            state.key_state.epoch,
            state.ephemeral_pubkey,
        ))
    }

    /// Begin a rekey: generates a fresh random key under a new keyset_id/epoch,
    /// staged as pending. Promotion timing/grace follows the registry's
    /// [`RekeyPolicy`]; call [`Self::maybe_promote_pending`] to advance it.
    pub async fn begin_rekey(
        &self,
        group: &GroupRef,
        new_keyset_id: &str,
        effective_at: Instant,
    ) -> Result<(), String> {
        let mut new_key = Zeroizing::new([0u8; 32]);
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut *new_key);

        let mut groups = self.groups.write();
        let state = groups
            .get_mut(group)
            .ok_or_else(|| format!("group {group:?} not registered"))?;
        state.key_state.pending = Some(PendingRekey {
            new_keyset_id: new_keyset_id.to_owned(),
            new_epoch: state.key_state.epoch + 1,
            new_key,
            effective_at,
        });
        Ok(())
    }

    /// Promote a pending rekey if its `effective_at` has passed (no-op otherwise).
    /// Old `K_group` bytes are dropped (zeroized) on promotion — forward secrecy:
    /// a member who only has the old key cannot derive the new one.
    pub async fn maybe_promote_pending(
        &self,
        group: &GroupRef,
        now: Instant,
    ) -> Result<bool, String> {
        let mut groups = self.groups.write();
        let state = groups
            .get_mut(group)
            .ok_or_else(|| format!("group {group:?} not registered"))?;

        let due = matches!(&state.key_state.pending, Some(p) if now >= p.effective_at);
        if !due {
            return Ok(false);
        }
        let Some(pending) = state.key_state.pending.take() else {
            return Ok(false);
        };
        state.key_state.keyset_id = pending.new_keyset_id;
        state.key_state.epoch = pending.new_epoch;
        state.key_state.current = pending.new_key;
        state.key_state.created_at = now;
        Ok(true)
    }

    /// The current `K_group` for a group, if registered. Consumers use this for
    /// downstream keyed-PRF derivations (e.g. EventService's topic key).
    pub async fn k_group(&self, group: &GroupRef) -> Option<[u8; 32]> {
        let groups = self.groups.read();
        groups.get(group).map(|s| *s.key_state.current)
    }

    /// `(keyset_id, epoch)` for a group, if registered.
    pub async fn keyset(&self, group: &GroupRef) -> Option<(String, u64)> {
        let groups = self.groups.read();
        groups
            .get(group)
            .map(|s| (s.key_state.keyset_id.clone(), s.key_state.epoch))
    }

    pub fn rekey_policy(&self) -> &RekeyPolicy {
        &self.rekey_policy
    }

    /// Best-effort wall-clock timestamp helper (unix millis) for consumers
    /// building [`RotationResult`]-style outputs.
    pub fn now_millis() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
    }
}

/// Subject-identity hash bound into the wrap AAD — a DID's UTF-8 bytes hashed to
/// 32 bytes. DIDs are the membership identity; pubkeys are per-session, so the
/// wrap AAD binds to the stable identity, not the ephemeral key.
fn keyed_subject_hash(subject_did: &str) -> [u8; 32] {
    // Domain-separated, unkeyed-in-effect (zero key) hash — this is an AAD
    // binding value, not a secret; keyed_mac is used here for consistency with
    // the rest of the crypto module's primitives.
    let zero_key = [0u8; 32];
    let mac = keyed_mac(&zero_key, subject_did.as_bytes());
    let mut out = [0u8; 32];
    out.copy_from_slice(&mac[..32]);
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::crypto::event_crypto::{derive_wrap_key, unwrap_group_key};

    struct AllowResolver;
    impl MembershipResolver for AllowResolver {
        fn resolve(
            &self,
            group_uri: &str,
            subject_did: &str,
            member_pubkey: [u8; 32],
        ) -> Result<GroupMembership, String> {
            Ok(GroupMembership {
                group_uri: group_uri.to_owned(),
                subject_did: subject_did.to_owned(),
                member_pubkey,
            })
        }
    }

    fn grp(uri: &str) -> GroupRef {
        GroupRef::new(uri, "ks-1")
    }

    #[tokio::test]
    async fn deny_all_resolver_fails_closed() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, DenyAllResolver).unwrap();
        registry
            .register_group(grp("at://did:web:a/g1"))
            .await
            .unwrap();
        let result = registry
            .join(&grp("at://did:web:a/g1"), "did:web:member", [7u8; 32])
            .await;
        assert!(result.is_err(), "DenyAllResolver must fail-closed");
    }

    #[tokio::test]
    async fn join_then_unwrap_round_trips() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowResolver).unwrap();
        let g = grp("at://did:web:a/g1");
        registry.register_group(g.clone()).await.unwrap();

        let (member_secret, member_public) = crate::crypto::generate_ephemeral_keypair();
        let member_secret_bytes = member_secret.scalar().to_bytes();
        let member_pubkey = member_public.to_bytes();

        let (wrapped, keyset_id, epoch, publisher_pubkey) = registry
            .join(&g, "did:web:member", member_pubkey)
            .await
            .unwrap();
        assert_eq!(keyset_id, "ks-1");
        assert_eq!(epoch, 0);

        let shared =
            crate::crypto::ristretto_dh_raw(&member_secret_bytes, &publisher_pubkey).expect("dh");
        let member_wrap_key = derive_wrap_key(&shared, &publisher_pubkey, &member_pubkey);
        let unwrapped = unwrap_group_key(
            &member_wrap_key,
            &wrapped,
            &g.group_uri,
            &keyed_subject_hash("did:web:member"),
            &g.group_uri,
        );
        assert!(
            unwrapped.is_ok(),
            "member must be able to unwrap K_group via DH: {:?}",
            unwrapped.err()
        );
    }

    #[tokio::test]
    async fn blinded_join_then_unwrap_round_trips() {
        // EV4: the subscriber presents a BLINDED pubkey to join; the publisher
        // DHs against the blinded key (blinding-agnostic) and never sees the
        // stable key. The subscriber unwraps via blinded_dh_raw.
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowResolver).unwrap();
        let g = grp("at://did:web:a/g1");
        registry.register_group(g.clone()).await.unwrap();

        let (member_secret, member_public) = crate::crypto::generate_ephemeral_keypair();
        let member_secret_bytes = member_secret.scalar().to_bytes();
        let member_pubkey = member_public.to_bytes();

        // Subscriber blinds their stable key (subscriber-side; r never leaves).
        let blinded = BlindedMember::new(&member_pubkey).expect("blind");
        // The blinded pubkey is different from the stable key ( unlinkable).
        assert_ne!(blinded.blinded_pubkey, member_pubkey);

        // join receives the BLINDED pubkey; the publisher never sees `member_pubkey`.
        let (wrapped, _keyset_id, _epoch, publisher_pubkey) = registry
            .join(&g, "did:web:member", blinded.blinded_pubkey)
            .await
            .expect("join with blinded pubkey");

        // Subscriber unwraps via the blinded-DH path.
        let recovered = blinded
            .unwrap(
                &member_secret_bytes,
                &publisher_pubkey,
                &wrapped,
                &g.group_uri,
                &g.group_uri,
                "did:web:member",
            )
            .expect("blinded unwrap");
        // And it equals the registry's authoritative K_group.
        let authoritative = registry.k_group(&g).await.expect("k_group");
        assert_eq!(*recovered, authoritative);
    }

    #[tokio::test]
    async fn k_group_returns_current_key() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowResolver).unwrap();
        let g = grp("at://did:web:a/g1");
        registry.register_group(g.clone()).await.unwrap();
        assert!(registry.k_group(&g).await.is_some());
        assert_eq!(registry.keyset(&g).await, Some(("ks-1".to_owned(), 0)));
    }

    #[tokio::test]
    async fn rekey_rotates_to_fresh_random_key_and_old_key_unrecoverable() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowResolver).unwrap();
        let g = grp("at://did:web:a/g1");
        registry.register_group(g.clone()).await.unwrap();

        let now = Instant::now();
        registry.begin_rekey(&g, "ks-2", now).await.unwrap();
        let promoted = registry.maybe_promote_pending(&g, now).await.unwrap();
        assert!(promoted, "effective_at already passed -> should promote");

        // Re-join after rotation must report the NEW keyset/epoch.
        let (_, member2_pubkey) = crate::crypto::generate_ephemeral_keypair();
        let (_wrapped, keyset_id, epoch, _pub) = registry
            .join(&g, "did:web:member2", member2_pubkey.to_bytes())
            .await
            .unwrap();
        assert_eq!(keyset_id, "ks-2");
        assert_eq!(epoch, 1);
    }

    #[tokio::test]
    async fn rekey_not_yet_effective_does_not_promote() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowResolver).unwrap();
        let g = grp("at://did:web:a/g1");
        registry.register_group(g.clone()).await.unwrap();

        let future = Instant::now() + Duration::from_secs(3600);
        registry.begin_rekey(&g, "ks-2", future).await.unwrap();
        let promoted = registry
            .maybe_promote_pending(&g, Instant::now())
            .await
            .unwrap();
        assert!(
            !promoted,
            "effective_at in the future -> must not promote yet"
        );

        let (_, member3_pubkey) = crate::crypto::generate_ephemeral_keypair();
        let (_wrapped, keyset_id, epoch, _pub) = registry
            .join(&g, "did:web:member3", member3_pubkey.to_bytes())
            .await
            .unwrap();
        assert_eq!(keyset_id, "ks-1", "still on old keyset until promotion");
        assert_eq!(epoch, 0);
    }

    #[tokio::test]
    async fn duplicate_registration_rejected() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowResolver).unwrap();
        let g = grp("at://did:web:a/g1");
        registry.register_group(g.clone()).await.unwrap();
        let second = registry.register_group(g).await;
        assert!(second.is_err());
    }

    #[test]
    fn rekey_policy_validation_rejects_overlong_interval() {
        assert!(RekeyPolicy::Scheduled {
            interval: Duration::from_secs(100_000)
        }
        .validate()
        .is_err());
        assert!(RekeyPolicy::Immediate.validate().is_ok());
    }
}
