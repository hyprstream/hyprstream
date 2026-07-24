//! Controller-managed hybrid group epochs for record-backed fan-out.
//!
//! Epoch secrets are fresh random values and are sealed independently to every
//! admitted member's accepted-state-bound HyKEM recipient with COSE_Encrypt0.
//! Relays receive only ordinary encrypted MOQT object bytes; epoch grants are
//! control-plane objects and MUST NOT be published on a relay track.
//!
//! Membership changes use an explicit prepare/commit transaction. Preparing a
//! change creates and seals the next epoch but does not expose it as current.
//! Committing swaps membership version, epoch, key, and member set atomically.
//! A crash before commit therefore leaves the prior committed epoch intact.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use zeroize::Zeroizing;

use crate::crypto::cose_encrypt;
use crate::crypto::hybrid_kem::{RecipientKeypair, RecipientPublic};

pub const MAX_KEY_LIFETIME: Duration = Duration::from_secs(86_400);
pub const DEFAULT_ROTATION_INTERVAL: Duration = Duration::from_secs(3_600);
pub const GRACE_PERIOD: Duration = Duration::from_secs(120);

#[derive(Clone, Debug)]
pub enum RekeyPolicy {
    Scheduled {
        interval: Duration,
    },
    Immediate,
    Jittered {
        interval: Duration,
        jitter: Duration,
    },
}

impl Default for RekeyPolicy {
    fn default() -> Self {
        Self::Scheduled {
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
                Err(format!(
                    "interval {interval:?} exceeds MAX_KEY_LIFETIME ({MAX_KEY_LIFETIME:?})"
                ))
            }
            _ => Ok(()),
        }
    }
}

/// Encrypted event payload. `publisher_pubkey` is an anchored Ed25519 key id;
/// `signature` is a mandatory hybrid COSE composite (Ed25519 + ML-DSA-65).
#[derive(Debug, Clone)]
pub struct EncryptedEvent {
    pub topic: String,
    pub tag: Vec<u8>,
    pub ciphertext: Vec<u8>,
    pub nonce: [u8; 12],
    pub key_commitment: [u8; 16],
    pub lk_tag: Vec<u8>,
    pub signature: Vec<u8>,
    pub publisher_pubkey: [u8; 32],
    pub timestamp: i64,
    /// CSPRNG lifecycle identifier that scopes one publisher/track epoch key.
    pub session_id: [u8; 16],
    /// Controller-committed membership version and epoch are sender-authenticated.
    pub membership_version: u64,
    pub epoch: u64,
    /// Monotonic per-sender/track sequence within the session-scoped key.
    pub sequence: u64,
}

#[derive(Debug, Clone)]
pub struct WrappedKeyEntry {
    /// Random recipient-local routing tag; regenerated every epoch.
    pub routing_tag: [u8; 16],
    /// Accepted `#mesh-kem` recipient key id.
    pub recipient_key_id: Vec<u8>,
    /// Canonical COSE_Encrypt0 carrying the epoch secret and HyKEM shares.
    pub wrapped_blob: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct RotationResult {
    pub wrapped_keys: Vec<WrappedKeyEntry>,
    pub effective_at_millis: i64,
    pub membership_version: u64,
    pub epoch: u64,
}

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

/// State that identifies the controller and the accepted CID512/did:at9p view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControllerBinding {
    pub controller_did: String,
    pub accepted_state: Vec<u8>,
    /// Controller-authorized sender identity for this epoch profile.
    pub sender_did: String,
    pub sender_ed25519: [u8; 32],
    pub sender_ml_dsa_65: Vec<u8>,
    /// Canonical policy identifiers, authenticated into every member grant.
    pub retention_policy: Vec<u8>,
    pub opaque_routing_policy: Vec<u8>,
    pub expires_at_millis: i64,
}

/// A member admission resolved from authenticated policy/record state.
///
/// `blinded_routing_key` is a fresh subscriber-generated Ristretto presentation.
/// It retains the old construction's unlinkability property, but is only an
/// authenticated opaque binding: it is never used to derive key material. Epoch
/// confidentiality comes solely from the separate HyKEM recipient.
#[derive(Clone, Debug)]
pub struct GroupMembership {
    pub group_uri: String,
    pub subject_did: String,
    pub accepted_state: Vec<u8>,
    pub capability: Vec<u8>,
    pub recipient_key_id: Vec<u8>,
    pub recipient: RecipientPublic,
    pub blinded_routing_key: [u8; 32],
    pub expires_at_millis: i64,
}

impl GroupMembership {
    fn validate(&self, group_uri: &str) -> Result<(), String> {
        if self.group_uri != group_uri {
            return Err("membership group does not match requested group".to_owned());
        }
        if self.subject_did.is_empty()
            || self.accepted_state.is_empty()
            || self.capability.is_empty()
            || self.recipient_key_id.is_empty()
        {
            return Err("membership binding is incomplete".to_owned());
        }
        if self.blinded_routing_key == [0; 32] {
            return Err("blinded routing key must not be all zero".to_owned());
        }
        self.recipient.validate().map_err(|e| e.to_string())?;
        let expected = recipient_key_id(&self.recipient);
        if self.recipient_key_id != expected {
            return Err("recipient key id does not match accepted HyKEM key".to_owned());
        }
        Ok(())
    }
}

/// The TCB membership gate. Implementations validate accepted state, capability,
/// expiry and the current `#mesh-kem` recipient before returning a binding.
pub trait MembershipResolver: Send + Sync {
    fn resolve(&self, requested: &GroupMembership) -> Result<GroupMembership, String>;
}

pub struct DenyAllResolver;
impl MembershipResolver for DenyAllResolver {
    fn resolve(&self, _requested: &GroupMembership) -> Result<GroupMembership, String> {
        Err("MembershipResolver not wired — fail-closed".to_owned())
    }
}

/// Subscriber-side unlinkable presentation retained from the prior blinded-DH
/// protocol. It is intentionally not a KEX key in the hybrid construction.
pub struct BlindedMember {
    pub blinded_pubkey: [u8; 32],
    pub blinding: [u8; 32],
}

impl BlindedMember {
    pub fn new(member_pubkey: &[u8; 32]) -> Result<Self, String> {
        let pub_obj = crate::crypto::RistrettoPublic::from_bytes(member_pubkey)
            .ok_or_else(|| "invalid Ristretto255 member public key".to_owned())?;
        let (blinded, blinding) = crate::crypto::rerandomize_pubkey(&pub_obj);
        Ok(Self {
            blinded_pubkey: blinded.to_bytes(),
            blinding,
        })
    }
}

#[derive(Clone, Debug)]
pub enum MembershipChange {
    Join(GroupMembership),
    Leave {
        subject_did: String,
    },
    Revoke {
        subject_did: String,
    },
    Expire {
        subject_did: String,
    },
    RotateRecipient(GroupMembership),
    ControllerChange(ControllerBinding),
    AcceptedStateAdvance {
        accepted_state: Vec<u8>,
        members: Vec<GroupMembership>,
    },
}

#[derive(Clone, Debug)]
pub struct EpochGrant {
    pub group_uri: String,
    pub controller_did: String,
    pub keyset_id: String,
    pub subject_did: String,
    pub accepted_state: Vec<u8>,
    pub sender_did: String,
    pub sender_ed25519: [u8; 32],
    pub sender_ml_dsa_65: Vec<u8>,
    pub retention_policy: Vec<u8>,
    pub opaque_routing_policy: Vec<u8>,
    pub capability: Vec<u8>,
    pub recipient_key_id: Vec<u8>,
    pub blinded_routing_key: [u8; 32],
    pub membership_version: u64,
    pub epoch: u64,
    pub expires_at_millis: i64,
    pub sealed_epoch_secret: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct PreparedEpoch {
    pub group_uri: String,
    pub keyset_id: String,
    pub membership_version: u64,
    pub epoch: u64,
    pub grants: Vec<EpochGrant>,
}

struct EpochState {
    keyset_id: String,
    membership_version: u64,
    epoch: u64,
    key: Zeroizing<[u8; 32]>,
    controller: ControllerBinding,
    members: HashMap<String, GroupMembership>,
    created_at: Instant,
}

struct PendingEpoch {
    state: EpochState,
    public: PreparedEpoch,
}

struct GroupState {
    current: EpochState,
    pending: Option<PendingEpoch>,
}

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

    pub async fn register_group(
        &self,
        group: GroupRef,
        controller: ControllerBinding,
    ) -> Result<(), String> {
        validate_controller(&controller)?;
        let state = EpochState {
            keyset_id: group.keyset_id.clone(),
            membership_version: 0,
            epoch: 0,
            key: random_epoch_key(),
            controller,
            members: HashMap::new(),
            created_at: Instant::now(),
        };
        let mut groups = self.groups.write();
        if groups.contains_key(&group) {
            return Err(format!("group {group:?} already registered"));
        }
        groups.insert(
            group,
            GroupState {
                current: state,
                pending: None,
            },
        );
        Ok(())
    }

    /// Prepare the next epoch and seal it independently to every resulting
    /// member. Current state remains visible until [`Self::commit_prepared`].
    pub async fn prepare_change(
        &self,
        group: &GroupRef,
        change: MembershipChange,
        new_keyset_id: impl Into<String>,
    ) -> Result<PreparedEpoch, String> {
        let mut groups = self.groups.write();
        let state = groups
            .get_mut(group)
            .ok_or_else(|| format!("group {group:?} not registered"))?;
        if state.pending.is_some() {
            return Err("a membership/epoch transaction is already pending".to_owned());
        }

        let mut controller = state.current.controller.clone();
        let mut members = state.current.members.clone();
        match change {
            MembershipChange::Join(requested) | MembershipChange::RotateRecipient(requested) => {
                let resolved = self.resolver.resolve(&requested)?;
                resolved.validate(&group.group_uri)?;
                if resolved.subject_did != requested.subject_did
                    || resolved.accepted_state != controller.accepted_state
                {
                    return Err("resolver returned mismatched or stale membership".to_owned());
                }
                members.insert(resolved.subject_did.clone(), resolved);
            }
            MembershipChange::Leave { subject_did }
            | MembershipChange::Revoke { subject_did }
            | MembershipChange::Expire { subject_did } => {
                if members.remove(&subject_did).is_none() {
                    return Err("member is not in the committed group".to_owned());
                }
            }
            MembershipChange::ControllerChange(next) => {
                validate_controller(&next)?;
                controller = next;
                // Controller/accepted-state changes invalidate every old grant.
                members.clear();
            }
            MembershipChange::AcceptedStateAdvance {
                accepted_state,
                members: requested_members,
            } => {
                if accepted_state.is_empty() || accepted_state == controller.accepted_state {
                    return Err(
                        "accepted-state advance must install a fresh non-empty state".to_owned(),
                    );
                }
                controller.accepted_state = accepted_state;
                members.clear();
                for requested in requested_members {
                    let resolved = self.resolver.resolve(&requested)?;
                    resolved.validate(&group.group_uri)?;
                    if resolved.subject_did != requested.subject_did
                        || resolved.accepted_state != controller.accepted_state
                    {
                        return Err("member is not bound to the advanced accepted state".to_owned());
                    }
                    members.insert(resolved.subject_did.clone(), resolved);
                }
            }
        }

        let membership_version =
            state
                .current
                .membership_version
                .checked_add(1)
                .ok_or_else(|| {
                    "membership version is exhausted; create a new group generation".to_owned()
                })?;
        let epoch = state
            .current
            .epoch
            .checked_add(1)
            .ok_or_else(|| "epoch is exhausted; create a new group generation".to_owned())?;
        let keyset_id = new_keyset_id.into();
        if keyset_id.is_empty() || keyset_id == state.current.keyset_id {
            return Err("new epoch requires a fresh keyset id".to_owned());
        }
        let key = random_epoch_key();
        let grants = members
            .values()
            .map(|member| {
                seal_epoch_grant(
                    group,
                    &keyset_id,
                    &controller,
                    member,
                    membership_version,
                    epoch,
                    &key,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let public = PreparedEpoch {
            group_uri: group.group_uri.clone(),
            keyset_id: keyset_id.clone(),
            membership_version,
            epoch,
            grants,
        };
        state.pending = Some(PendingEpoch {
            state: EpochState {
                keyset_id,
                membership_version,
                epoch,
                key,
                controller,
                members,
                created_at: Instant::now(),
            },
            public: public.clone(),
        });
        Ok(public)
    }

    /// Atomically make a prepared membership version and epoch visible.
    pub async fn commit_prepared(
        &self,
        group: &GroupRef,
        membership_version: u64,
        epoch: u64,
    ) -> Result<(), String> {
        let mut groups = self.groups.write();
        let state = groups
            .get_mut(group)
            .ok_or_else(|| format!("group {group:?} not registered"))?;
        let pending = state
            .pending
            .as_ref()
            .ok_or_else(|| "no pending epoch".to_owned())?;
        if pending.public.membership_version != membership_version || pending.public.epoch != epoch
        {
            return Err("commit coordinates do not match the pending transaction".to_owned());
        }
        let pending = state
            .pending
            .take()
            .ok_or_else(|| "no pending epoch".to_owned())?;
        state.current = pending.state;
        Ok(())
    }

    /// Discard an uncommitted transaction. The committed epoch is untouched.
    pub async fn abort_prepared(
        &self,
        group: &GroupRef,
        keyset_id: &str,
        membership_version: u64,
        epoch: u64,
    ) -> Result<(), String> {
        let mut groups = self.groups.write();
        let state = groups
            .get_mut(group)
            .ok_or_else(|| format!("group {group:?} not registered"))?;
        let pending = state
            .pending
            .as_ref()
            .ok_or_else(|| "no pending epoch".to_owned())?;
        if pending.public.keyset_id != keyset_id
            || pending.public.membership_version != membership_version
            || pending.public.epoch != epoch
        {
            return Err("abort coordinates do not match the pending transaction".to_owned());
        }
        state.pending = None;
        Ok(())
    }

    pub async fn committed_coordinates(&self, group: &GroupRef) -> Option<(String, u64, u64)> {
        let groups = self.groups.read();
        groups.get(group).map(|state| {
            (
                state.current.keyset_id.clone(),
                state.current.membership_version,
                state.current.epoch,
            )
        })
    }

    pub async fn pending_epoch(&self, group: &GroupRef) -> Option<PreparedEpoch> {
        self.groups
            .read()
            .get(group)
            .and_then(|state| state.pending.as_ref().map(|pending| pending.public.clone()))
    }

    pub async fn k_group(&self, group: &GroupRef) -> Option<[u8; 32]> {
        self.groups
            .read()
            .get(group)
            .map(|state| *state.current.key)
    }

    pub async fn contains_member(&self, group: &GroupRef, subject_did: &str) -> bool {
        self.groups
            .read()
            .get(group)
            .is_some_and(|state| state.current.members.contains_key(subject_did))
    }

    pub async fn needs_rotation(&self, group: &GroupRef) -> bool {
        let groups = self.groups.read();
        let Some(state) = groups.get(group) else {
            return false;
        };
        let age = state.current.created_at.elapsed();
        match self.rekey_policy {
            RekeyPolicy::Immediate => false,
            RekeyPolicy::Scheduled { interval } => age >= interval,
            RekeyPolicy::Jittered { interval, jitter } => {
                let digest = blake3::hash(group.group_uri.as_bytes());
                let fraction = u16::from_be_bytes([digest.as_bytes()[0], digest.as_bytes()[1]])
                    as f64
                    / u16::MAX as f64;
                age >= interval + Duration::from_secs_f64(jitter.as_secs_f64() * fraction)
            }
        }
    }

    pub fn rekey_policy(&self) -> &RekeyPolicy {
        &self.rekey_policy
    }

    pub fn now_millis() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64
    }
}

pub fn recipient_key_id(recipient: &RecipientPublic) -> Vec<u8> {
    blake3::derive_key(
        "hyprstream group HyKEM recipient key id v1",
        &recipient.encode(),
    )
    .to_vec()
}

pub fn open_epoch_grant(
    recipient: &RecipientKeypair,
    grant: &EpochGrant,
) -> Result<Zeroizing<[u8; 32]>, String> {
    if recipient_key_id(&recipient.public()) != grant.recipient_key_id {
        return Err("epoch grant is not addressed to this HyKEM recipient".to_owned());
    }
    let aad = epoch_grant_aad(grant);
    let plaintext = cose_encrypt::open_from_recipient(
        recipient,
        &grant.sealed_epoch_secret,
        &aad,
        grant.epoch,
        grant.membership_version,
    )
    .map_err(|e| format!("hybrid epoch unwrap failed: {e}"))?;
    let key: [u8; 32] = plaintext
        .try_into()
        .map_err(|_| "epoch secret must be exactly 32 bytes".to_owned())?;
    Ok(Zeroizing::new(key))
}

fn validate_controller(controller: &ControllerBinding) -> Result<(), String> {
    if controller.controller_did.is_empty()
        || controller.accepted_state.is_empty()
        || controller.sender_did.is_empty()
        || controller.sender_ed25519 == [0; 32]
        || controller.sender_ml_dsa_65.is_empty()
        || controller.retention_policy.is_empty()
        || controller.opaque_routing_policy.is_empty()
    {
        return Err("controller binding is incomplete".to_owned());
    }
    ed25519_dalek::VerifyingKey::from_bytes(&controller.sender_ed25519)
        .map_err(|_| "controller sender Ed25519 key is invalid".to_owned())?;
    crate::crypto::pq::ml_dsa_vk_from_bytes(&controller.sender_ml_dsa_65)
        .map_err(|_| "controller sender ML-DSA-65 key is invalid".to_owned())?;
    Ok(())
}

fn random_epoch_key() -> Zeroizing<[u8; 32]> {
    let mut key = Zeroizing::new([0; 32]);
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut *key);
    key
}

fn seal_epoch_grant(
    group: &GroupRef,
    keyset_id: &str,
    controller: &ControllerBinding,
    member: &GroupMembership,
    membership_version: u64,
    epoch: u64,
    key: &[u8; 32],
) -> Result<EpochGrant, String> {
    let mut grant = EpochGrant {
        group_uri: group.group_uri.clone(),
        controller_did: controller.controller_did.clone(),
        keyset_id: keyset_id.to_owned(),
        subject_did: member.subject_did.clone(),
        accepted_state: member.accepted_state.clone(),
        sender_did: controller.sender_did.clone(),
        sender_ed25519: controller.sender_ed25519,
        sender_ml_dsa_65: controller.sender_ml_dsa_65.clone(),
        retention_policy: controller.retention_policy.clone(),
        opaque_routing_policy: controller.opaque_routing_policy.clone(),
        capability: member.capability.clone(),
        recipient_key_id: member.recipient_key_id.clone(),
        blinded_routing_key: member.blinded_routing_key,
        membership_version,
        epoch,
        expires_at_millis: member.expires_at_millis.min(controller.expires_at_millis),
        sealed_epoch_secret: Vec::new(),
    };
    let aad = epoch_grant_aad(&grant);
    grant.sealed_epoch_secret =
        cose_encrypt::seal_to_recipient(&member.recipient, key, &aad, epoch, membership_version)
            .map_err(|e| format!("hybrid epoch seal failed: {e}"))?;
    Ok(grant)
}

fn epoch_grant_aad(grant: &EpochGrant) -> Vec<u8> {
    fn lp(out: &mut Vec<u8>, value: &[u8]) {
        out.extend_from_slice(&(value.len() as u32).to_be_bytes());
        out.extend_from_slice(value);
    }
    let mut out = b"hyprstream controller group epoch grant v1".to_vec();
    lp(&mut out, grant.group_uri.as_bytes());
    lp(&mut out, grant.controller_did.as_bytes());
    lp(&mut out, grant.keyset_id.as_bytes());
    lp(&mut out, grant.subject_did.as_bytes());
    lp(&mut out, &grant.accepted_state);
    lp(&mut out, grant.sender_did.as_bytes());
    lp(&mut out, &grant.sender_ed25519);
    lp(&mut out, &grant.sender_ml_dsa_65);
    lp(&mut out, &grant.retention_policy);
    lp(&mut out, &grant.opaque_routing_policy);
    lp(&mut out, &grant.capability);
    lp(&mut out, &grant.recipient_key_id);
    lp(&mut out, &grant.blinded_routing_key);
    out.extend_from_slice(&grant.membership_version.to_be_bytes());
    out.extend_from_slice(&grant.epoch.to_be_bytes());
    out.extend_from_slice(&grant.expires_at_millis.to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::crypto::hybrid_kem::{generate_recipient, SuiteId};

    struct AllowExact;
    impl MembershipResolver for AllowExact {
        fn resolve(&self, requested: &GroupMembership) -> Result<GroupMembership, String> {
            Ok(requested.clone())
        }
    }

    struct SwapsSubject;
    impl MembershipResolver for SwapsSubject {
        fn resolve(&self, requested: &GroupMembership) -> Result<GroupMembership, String> {
            let mut resolved = requested.clone();
            resolved.subject_did = "did:web:attacker".to_owned();
            Ok(resolved)
        }
    }

    fn group() -> GroupRef {
        GroupRef::new("at://did:web:controller/events/g1", "ks-0")
    }
    fn controller(state: &[u8]) -> ControllerBinding {
        let ed = ed25519_dalek::SigningKey::from_bytes(&[42; 32]);
        let pq = crate::node_identity::derive_mesh_mldsa_key(&ed);
        ControllerBinding {
            controller_did: "did:web:controller".to_owned(),
            accepted_state: state.to_vec(),
            sender_did: "did:web:publisher".to_owned(),
            sender_ed25519: ed.verifying_key().to_bytes(),
            sender_ml_dsa_65: crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq),
            retention_policy: b"retention:bounded-v1".to_vec(),
            opaque_routing_policy: b"routing:stock-moq-opaque-v1".to_vec(),
            expires_at_millis: i64::MAX,
        }
    }
    fn member(did: &str, recipient: &RecipientKeypair, blind: u8, state: &[u8]) -> GroupMembership {
        let public = recipient.public();
        GroupMembership {
            group_uri: group().group_uri,
            subject_did: did.to_owned(),
            accepted_state: state.to_vec(),
            capability: format!("cap:{did}:subscribe").into_bytes(),
            recipient_key_id: recipient_key_id(&public),
            recipient: public,
            blinded_routing_key: [blind; 32],
            expires_at_millis: i64::MAX,
        }
    }

    #[tokio::test]
    async fn deny_all_fails_before_epoch_release() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, DenyAllResolver).unwrap();
        registry
            .register_group(group(), controller(b"cid512:a"))
            .await
            .unwrap();
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let result = registry
            .prepare_change(
                &group(),
                MembershipChange::Join(member("did:web:m", &recipient, 7, b"cid512:a")),
                "ks-1",
            )
            .await;
        assert!(result.is_err());
        assert_eq!(registry.committed_coordinates(&group()).await.unwrap().1, 0);
    }

    #[tokio::test]
    async fn hybrid_join_unwraps_and_wrong_recipient_cannot() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowExact).unwrap();
        registry
            .register_group(group(), controller(b"cid512:a"))
            .await
            .unwrap();
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let wrong = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let prepared = registry
            .prepare_change(
                &group(),
                MembershipChange::Join(member("did:web:m", &recipient, 7, b"cid512:a")),
                "ks-1",
            )
            .await
            .unwrap();
        assert_eq!(prepared.grants.len(), 1);
        assert!(open_epoch_grant(&wrong, &prepared.grants[0]).is_err());
        let opened = open_epoch_grant(&recipient, &prepared.grants[0]).unwrap();
        registry.commit_prepared(&group(), 1, 1).await.unwrap();
        assert_eq!(*opened, registry.k_group(&group()).await.unwrap());
    }

    #[tokio::test]
    async fn prepare_is_invisible_and_abort_preserves_committed_epoch() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowExact).unwrap();
        registry
            .register_group(group(), controller(b"cid512:a"))
            .await
            .unwrap();
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        registry
            .prepare_change(
                &group(),
                MembershipChange::Join(member("did:web:m", &recipient, 8, b"cid512:a")),
                "ks-1",
            )
            .await
            .unwrap();
        assert_eq!(
            registry.committed_coordinates(&group()).await.unwrap(),
            ("ks-0".to_owned(), 0, 0)
        );
        assert!(!registry.contains_member(&group(), "did:web:m").await);
        registry
            .abort_prepared(&group(), "ks-1", 1, 1)
            .await
            .unwrap();
        assert_eq!(registry.committed_coordinates(&group()).await.unwrap().2, 0);
    }

    #[tokio::test]
    async fn stale_abort_cannot_discard_a_newer_pending_epoch() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowExact).unwrap();
        registry
            .register_group(group(), controller(b"cid512:a"))
            .await
            .unwrap();
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let first = registry
            .prepare_change(
                &group(),
                MembershipChange::Join(member("did:web:m", &recipient, 8, b"cid512:a")),
                "ks-1",
            )
            .await
            .unwrap();
        registry
            .abort_prepared(
                &group(),
                &first.keyset_id,
                first.membership_version,
                first.epoch,
            )
            .await
            .unwrap();
        let second = registry
            .prepare_change(
                &group(),
                MembershipChange::Join(member("did:web:m", &recipient, 8, b"cid512:a")),
                "ks-2",
            )
            .await
            .unwrap();
        assert!(registry
            .abort_prepared(
                &group(),
                &first.keyset_id,
                first.membership_version,
                first.epoch,
            )
            .await
            .is_err());
        assert_eq!(
            registry.pending_epoch(&group()).await.unwrap().keyset_id,
            second.keyset_id
        );
    }

    #[tokio::test]
    async fn accepted_state_advance_rejects_a_resolver_subject_swap() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, SwapsSubject).unwrap();
        registry
            .register_group(group(), controller(b"cid512:a"))
            .await
            .unwrap();
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let change = MembershipChange::AcceptedStateAdvance {
            accepted_state: b"cid512:b".to_vec(),
            members: vec![member("did:web:member", &recipient, 9, b"cid512:b")],
        };
        assert!(registry
            .prepare_change(&group(), change, "ks-1")
            .await
            .is_err());
        assert!(registry.pending_epoch(&group()).await.is_none());
    }

    #[tokio::test]
    async fn revoke_rotates_and_excludes_revoked_member() {
        let registry = GroupKeyRegistry::new(
            RekeyPolicy::Scheduled {
                interval: Duration::from_secs(3600),
            },
            AllowExact,
        )
        .unwrap();
        registry
            .register_group(group(), controller(b"cid512:a"))
            .await
            .unwrap();
        let a = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let b = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let p = registry
            .prepare_change(
                &group(),
                MembershipChange::Join(member("did:web:a", &a, 1, b"cid512:a")),
                "ks-1",
            )
            .await
            .unwrap();
        registry
            .commit_prepared(&group(), p.membership_version, p.epoch)
            .await
            .unwrap();
        let p = registry
            .prepare_change(
                &group(),
                MembershipChange::Join(member("did:web:b", &b, 2, b"cid512:a")),
                "ks-2",
            )
            .await
            .unwrap();
        registry
            .commit_prepared(&group(), p.membership_version, p.epoch)
            .await
            .unwrap();
        let p = registry
            .prepare_change(
                &group(),
                MembershipChange::Revoke {
                    subject_did: "did:web:a".to_owned(),
                },
                "ks-3",
            )
            .await
            .unwrap();
        assert_eq!(
            p.grants.len(),
            1,
            "revoked member receives no next-epoch material"
        );
        assert_eq!(p.grants[0].subject_did, "did:web:b");
        assert!(open_epoch_grant(&a, &p.grants[0]).is_err());
        assert!(open_epoch_grant(&b, &p.grants[0]).is_ok());
        registry
            .commit_prepared(&group(), p.membership_version, p.epoch)
            .await
            .unwrap();
        assert!(!registry.contains_member(&group(), "did:web:a").await);
    }

    #[tokio::test]
    async fn aad_mutation_and_classical_key_substitution_fail() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowExact).unwrap();
        registry
            .register_group(group(), controller(b"cid512:a"))
            .await
            .unwrap();
        let recipient = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
        let p = registry
            .prepare_change(
                &group(),
                MembershipChange::Join(member("did:web:m", &recipient, 3, b"cid512:a")),
                "ks-1",
            )
            .await
            .unwrap();
        let mut mutated = p.grants[0].clone();
        mutated.controller_did = "did:web:attacker".to_owned();
        assert!(
            open_epoch_grant(&recipient, &mutated).is_err(),
            "controller mutation must invalidate COSE AAD"
        );
        let mut substituted = p.grants[0].clone();
        // Mutation-effective negative control for the removed signing-key-as-KEX bug:
        // an arbitrary 32-byte classical signing public key cannot stand in for
        // the separately accepted suite-identified HyKEM recipient.
        substituted.recipient_key_id = vec![0xED; 32];
        assert!(open_epoch_grant(&recipient, &substituted).is_err());
    }

    #[test]
    fn blinded_presentations_remain_unlinkable_but_are_not_kex_keys() {
        let (_, stable) = crate::crypto::generate_ephemeral_keypair();
        let a = BlindedMember::new(&stable.to_bytes()).unwrap();
        let b = BlindedMember::new(&stable.to_bytes()).unwrap();
        assert_ne!(a.blinded_pubkey, stable.to_bytes());
        assert_ne!(a.blinded_pubkey, b.blinded_pubkey);
    }
}
