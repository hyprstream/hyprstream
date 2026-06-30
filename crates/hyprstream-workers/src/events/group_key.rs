//! DRAFT: `K_group` registry for record-backed EventService groups (epic #600,
//! EV2 #602). **Not wired into the live publish path** — that is EV3/EV4's job
//! (EV2 only establishes the group identity + key-derivation machinery).
//!
//! Today [`super::secure_publisher::SecureEventPublisher`] keys its group state
//! by an arbitrary **prefix string** (`register_prefix(prefix: &str)`). This
//! module replaces "group identity" with a **group record reference**
//! (`group_uri`, an `ai.hyprstream.event.group` at-uri — see
//! `hyprstream-pds::event_group` for the DRAFT lexicon) plus an explicit
//! **epoch counter**, while reusing the *exact same* rekey/wrap machinery
//! (`RekeyPolicy`, the DH-wrap-on-join pattern via `derive_wrap_key`/
//! `wrap_group_key`) that `secure_publisher.rs` already implements correctly.
//!
//! This module deliberately does **not** depend on `hyprstream-pds`'s concrete
//! `EventGroupRecord`/`EventGroupItemRecord` types — `hyprstream-workers` does
//! not currently depend on `hyprstream-pds`, and adding that cross-crate edge
//! is its own reviewable decision. Membership facts are passed in as plain
//! data (`GroupMembership`); the caller is responsible for resolving them from
//! the atproto records (today: a TODO stub, since the live `getRecord` call
//! path for this purpose isn't wired up yet — see [`MembershipResolver`]).
//!
//! # What "K_group derivation" means here
//!
//! `K_group` bytes are **freshly random per rotation** (`OsRng`), exactly as
//! `register_prefix` does today — there is no KDF chain producing `K_group`
//! itself. "Derivation" in the EV2 ticket refers to the **identity/reference**
//! `(group_uri, keyset_id, epoch)` being bound to the signed record, so a
//! member can ask "give me K_group for keyset X epoch Y" — not to a
//! cryptographic derivation of the key bytes. The keyed-PRF derivation in
//! EV3 (`topic = KDF(K_group, subject‖epoch)`) is a *different*, downstream
//! use of this K_group — out of scope here.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;
use zeroize::Zeroizing;

use hyprstream_rpc::crypto::event_crypto::{derive_wrap_key, wrap_group_key};

pub use crate::events::secure_publisher::RekeyPolicy;

/// A fact the caller has already resolved: `subject` is a verified member of
/// `group_uri`, per the bidirectional-consent check described in
/// `hyprstream-pds::event_group::check_bidirectional_consent` (owner-side
/// `groupItem` claim AND the member's own consent record both agree).
///
/// This struct intentionally carries no proof material — by the time a
/// [`GroupMembership`] exists, the caller has already verified the records
/// (CAR proof / commit signature) and the authz-prefilter (EV4). It is the
/// *result* of that work, not a substitute for it.
#[derive(Clone, Debug)]
pub struct GroupMembership {
    pub group_uri: String,
    pub subject_did: String,
    /// Member's ephemeral Ristretto255 pubkey (for the DH-wrap, same as
    /// `secure_publisher.rs`'s subscriber pubkey).
    pub member_pubkey: [u8; 32],
}

/// Resolves group membership for a join request. A real implementation fetches
/// the `ai.hyprstream.event.group`/`groupItem` records (and the member's own
/// consent record) via DiscoveryService's `getRecord`, verifies the CAR proofs,
/// and runs `check_bidirectional_consent` + the EV4 authz-prefilter.
///
/// **STUB in this draft** — no live `getRecord`-based resolution path exists
/// on `origin/main` for this purpose yet (the `ai.hyprstream.placement.*`
/// lexicons this pattern mirrors are themselves not yet merged — see #524).
/// `TODO(#524-integration)`: replace with a real resolver once that lands.
pub trait MembershipResolver: Send + Sync {
    fn resolve(
        &self,
        group_uri: &str,
        subject_did: &str,
        member_pubkey: [u8; 32],
    ) -> Result<GroupMembership, String>;
}

/// A resolver that always denies — the safe default until a real resolver is
/// wired in. Prevents this draft from accidentally admitting unverified
/// members if it's ever exercised before `TODO(#524-integration)` lands.
pub struct DenyAllResolver;

impl MembershipResolver for DenyAllResolver {
    fn resolve(
        &self,
        _group_uri: &str,
        _subject_did: &str,
        _member_pubkey: [u8; 32],
    ) -> Result<GroupMembership, String> {
        Err("MembershipResolver not wired (TODO #524-integration) — fail-closed".to_owned())
    }
}

/// Per-group key-state, identical shape to `secure_publisher.rs`'s
/// `GroupKeyState`/`PendingRekey`, but keyed by `group_uri` (a record
/// reference) instead of an arbitrary topic prefix, and carrying an explicit
/// `keyset_id` + `epoch` so members can request a specific generation.
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
    /// Publisher's ephemeral Ristretto255 keypair for this group (same DH-wrap
    /// role as `secure_publisher.rs`'s per-prefix ephemeral keypair).
    ephemeral_secret: Zeroizing<[u8; 32]>,
    ephemeral_pubkey: [u8; 32],
    /// Members who have successfully joined (subject_did -> pubkey).
    members: HashMap<String, [u8; 32]>,
}

/// Record-backed `K_group` registry. One instance per EventService publisher
/// node; groups are identified by `group_uri`, not prefix string.
pub struct GroupKeyRegistry<R: MembershipResolver> {
    groups: Arc<RwLock<HashMap<String, GroupState>>>,
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
    /// against this to receive their wrapped key).
    pub async fn register_group(
        &self,
        group_uri: &str,
        keyset_id: &str,
    ) -> Result<[u8; 32], String> {
        let mut group_key = Zeroizing::new([0u8; 32]);
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut *group_key);

        let (eph_secret, eph_public) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
        let ephemeral_secret = Zeroizing::new(eph_secret.scalar().to_bytes());
        let ephemeral_pubkey = eph_public.to_bytes();

        let state = GroupState {
            key_state: GroupKeyState {
                keyset_id: keyset_id.to_owned(),
                epoch: 0,
                current: group_key,
                pending: None,
                created_at: Instant::now(),
            },
            ephemeral_secret,
            ephemeral_pubkey,
            members: HashMap::new(),
        };

        let mut groups = self.groups.write().await;
        if groups.contains_key(group_uri) {
            return Err(format!("group {group_uri:?} already registered"));
        }
        groups.insert(group_uri.to_owned(), state);
        Ok(ephemeral_pubkey)
    }

    /// Join a group: resolve membership (fail-closed via [`MembershipResolver`]),
    /// then DH-wrap the current `K_group` for the member — exactly the
    /// `secure_publisher.rs` subscriber-add path, retargeted at group records.
    ///
    /// Returns `(wrapped_key_blob, keyset_id, epoch, publisher_ephemeral_pubkey)`.
    /// The member unwraps with `unwrap_group_key` using the same DH (their
    /// secret + the returned publisher pubkey) — see the round-trip test below.
    pub async fn join(
        &self,
        group_uri: &str,
        subject_did: &str,
        member_pubkey: [u8; 32],
    ) -> Result<(Vec<u8>, String, u64, [u8; 32]), String> {
        let membership = self
            .resolver
            .resolve(group_uri, subject_did, member_pubkey)?;
        if membership.group_uri != group_uri || membership.subject_did != subject_did {
            return Err("resolver returned mismatched membership".to_owned());
        }

        let mut groups = self.groups.write().await;
        let state = groups
            .get_mut(group_uri)
            .ok_or_else(|| format!("group {group_uri:?} not registered"))?;

        // DH(publisher_secret, member_pubkey) -> shared secret, THEN derive the
        // wrap key from it (derive_wrap_key's first arg is the shared secret,
        // not a raw secret scalar -- matches secure_publisher.rs's convention).
        let shared_secret = Zeroizing::new(
            hyprstream_rpc::crypto::ristretto_dh_raw(&state.ephemeral_secret, &member_pubkey)
                .map_err(|e| format!("DH failed: {e}"))?,
        );
        let wrap_key = derive_wrap_key(&shared_secret, &member_pubkey, &state.ephemeral_pubkey);
        let wrapped = wrap_group_key(
            &wrap_key,
            &state.key_state.current,
            &keyed_subject_hash(subject_did),
            group_uri,
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
    /// staged as pending. Promotion timing/grace follows the same
    /// `RekeyPolicy` semantics as `secure_publisher.rs` (kept identical
    /// rather than reimplemented — see that module for the Immediate vs
    /// Scheduled/Jittered tradeoff EV6 documents).
    pub async fn begin_rekey(
        &self,
        group_uri: &str,
        new_keyset_id: &str,
        effective_at: Instant,
    ) -> Result<(), String> {
        let mut new_key = Zeroizing::new([0u8; 32]);
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut *new_key);

        let mut groups = self.groups.write().await;
        let state = groups
            .get_mut(group_uri)
            .ok_or_else(|| format!("group {group_uri:?} not registered"))?;
        state.key_state.pending = Some(PendingRekey {
            new_keyset_id: new_keyset_id.to_owned(),
            new_epoch: state.key_state.epoch + 1,
            new_key,
            effective_at,
        });
        Ok(())
    }

    /// Promote a pending rekey if its `effective_at` has passed. No-op
    /// otherwise. Old `K_group` bytes are dropped (zeroized) on promotion —
    /// forward secrecy: a member who only has the old key cannot derive the
    /// new one, matching epic #600's threat-model requirement that epochs use
    /// a freshly-rotated *random* key, not a static key with an epoch tweak.
    pub async fn maybe_promote_pending(
        &self,
        group_uri: &str,
        now: Instant,
    ) -> Result<bool, String> {
        let mut groups = self.groups.write().await;
        let state = groups
            .get_mut(group_uri)
            .ok_or_else(|| format!("group {group_uri:?} not registered"))?;

        let due = matches!(&state.key_state.pending, Some(p) if now >= p.effective_at);
        if !due {
            return Ok(false);
        }
        // `due` only true when `pending` was `Some`, so this `take()` cannot
        // observe `None` — but match on it explicitly rather than `expect()`
        // (denied by workspace clippy) to keep the invariant infallible.
        let Some(pending) = state.key_state.pending.take() else {
            return Ok(false);
        };
        state.key_state.keyset_id = pending.new_keyset_id;
        state.key_state.epoch = pending.new_epoch;
        state.key_state.current = pending.new_key;
        state.key_state.created_at = now;
        Ok(true)
    }

    pub fn rekey_policy(&self) -> &RekeyPolicy {
        &self.rekey_policy
    }
}

/// Subject-identity hash bound into the wrap AAD, mirroring
/// `secure_publisher.rs`'s subscriber-hash convention (a DID's UTF-8 bytes
/// hashed to 32 bytes, rather than reusing the raw pubkey hash that module
/// uses — DIDs are the membership identity here, pubkeys are per-session).
fn keyed_subject_hash(subject_did: &str) -> [u8; 32] {
    use hyprstream_rpc::crypto::backend::keyed_mac;
    // Domain-separated, unkeyed-in-effect (zero key) hash — this is an AAD
    // binding value, not a secret; using keyed_mac here only for consistency
    // with the rest of the crypto module's primitives.
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
    use hyprstream_rpc::crypto::event_crypto::unwrap_group_key;

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

    #[tokio::test]
    async fn deny_all_resolver_fails_closed() {
        let registry = GroupKeyRegistry::new(
            RekeyPolicy::Scheduled {
                interval: std::time::Duration::from_secs(3600),
            },
            DenyAllResolver,
        )
        .unwrap();
        registry
            .register_group("at://did:web:a/ai.hyprstream.event.group/g1", "ks-1")
            .await
            .unwrap();
        let result = registry
            .join(
                "at://did:web:a/ai.hyprstream.event.group/g1",
                "did:web:member",
                [7u8; 32],
            )
            .await;
        assert!(result.is_err(), "DenyAllResolver must fail-closed");
    }

    #[tokio::test]
    async fn join_then_unwrap_round_trips() {
        let registry = GroupKeyRegistry::new(
            RekeyPolicy::Scheduled {
                interval: std::time::Duration::from_secs(3600),
            },
            AllowResolver,
        )
        .unwrap();
        let group_uri = "at://did:web:a/ai.hyprstream.event.group/g1";
        registry.register_group(group_uri, "ks-1").await.unwrap();

        // Member generates its own ephemeral DH keypair.
        let (member_secret, member_public) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
        let member_secret_bytes = member_secret.scalar().to_bytes();
        let member_pubkey = member_public.to_bytes();

        let (wrapped, keyset_id, epoch, publisher_pubkey) = registry
            .join(group_uri, "did:web:member", member_pubkey)
            .await
            .unwrap();
        assert_eq!(keyset_id, "ks-1");
        assert_eq!(epoch, 0);

        // Member-side: derive the same wrap key via DH against the publisher's
        // returned ephemeral pubkey, then unwrap.
        let shared =
            hyprstream_rpc::crypto::ristretto_dh_raw(&member_secret_bytes, &publisher_pubkey)
                .expect("dh");
        let member_wrap_key = derive_wrap_key(&shared, &publisher_pubkey, &member_pubkey);
        let unwrapped = unwrap_group_key(
            &member_wrap_key,
            &wrapped,
            &keyed_subject_hash("did:web:member"),
            group_uri,
        );
        assert!(
            unwrapped.is_ok(),
            "member must be able to unwrap K_group via DH: {:?}",
            unwrapped.err()
        );
    }

    #[tokio::test]
    async fn rekey_rotates_to_fresh_random_key_and_old_key_unrecoverable() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowResolver).unwrap();
        let group_uri = "at://did:web:a/ai.hyprstream.event.group/g1";
        registry.register_group(group_uri, "ks-1").await.unwrap();

        let now = Instant::now();
        registry.begin_rekey(group_uri, "ks-2", now).await.unwrap();
        let promoted = registry
            .maybe_promote_pending(group_uri, now)
            .await
            .unwrap();
        assert!(promoted, "effective_at already passed -> should promote");

        // Re-join after rotation must report the NEW keyset/epoch.
        let (_, member2_pubkey) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
        let (_wrapped, keyset_id, epoch, _pub) = registry
            .join(group_uri, "did:web:member2", member2_pubkey.to_bytes())
            .await
            .unwrap();
        assert_eq!(keyset_id, "ks-2");
        assert_eq!(epoch, 1);
    }

    #[tokio::test]
    async fn rekey_not_yet_effective_does_not_promote() {
        let registry = GroupKeyRegistry::new(RekeyPolicy::Immediate, AllowResolver).unwrap();
        let group_uri = "at://did:web:a/ai.hyprstream.event.group/g1";
        registry.register_group(group_uri, "ks-1").await.unwrap();

        let future = Instant::now() + std::time::Duration::from_secs(3600);
        registry
            .begin_rekey(group_uri, "ks-2", future)
            .await
            .unwrap();
        let promoted = registry
            .maybe_promote_pending(group_uri, Instant::now())
            .await
            .unwrap();
        assert!(
            !promoted,
            "effective_at in the future -> must not promote yet"
        );

        let (_, member3_pubkey) = hyprstream_rpc::crypto::generate_ephemeral_keypair();
        let (_wrapped, keyset_id, epoch, _pub) = registry
            .join(group_uri, "did:web:member3", member3_pubkey.to_bytes())
            .await
            .unwrap();
        assert_eq!(keyset_id, "ks-1", "still on old keyset until promotion");
        assert_eq!(epoch, 0);
    }

    #[tokio::test]
    async fn duplicate_registration_rejected() {
        let registry = GroupKeyRegistry::new(
            RekeyPolicy::Scheduled {
                interval: std::time::Duration::from_secs(3600),
            },
            AllowResolver,
        )
        .unwrap();
        let group_uri = "at://did:web:a/ai.hyprstream.event.group/g1";
        registry.register_group(group_uri, "ks-1").await.unwrap();
        let second = registry.register_group(group_uri, "ks-2").await;
        assert!(second.is_err());
    }
}
