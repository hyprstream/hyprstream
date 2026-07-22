//! C4 (#1170): the sealed op-log head — cross-process source of truth for the
//! active `#atproto` ES256 generation. **Closes #1123.**
//!
//! Plan t1124 §3.4: under `--ipc`, discovery/registry/oauth must agree on the
//! active generation with **no event-delivery mechanism**. They do it by
//! reading ONE sealed head written by the rotation task: the rotator (OAuth)
//! writes it after every promotion, every reader (the registry's
//! [`crate::services::discovery::PdsPublisher`] and, later, any other
//! cross-process consumer) reads it at use time. The process-local
//! `OnceLock`/`Es256SigningKeyStore` stays the rotator's own working state;
//! it is no longer the source of truth for *other* processes.
//!
//! ## Dependency honesty — C2 (#1168) is not built
//!
//! C4 formally depends on C2 (crash-atomic rotation). This module ships the
//! **read path** (real, signature-verified, fail-closed) plus the head
//! **format** and a [`seal_op_log_head`] **seam-write**. What C2 must
//! guarantee for this to be fully sound, and what C4 does *not* claim today:
//!
//!   * [`seal_op_log_head`] writes the head with an atomic rename — real
//!     file-level crash atomicity (a crash leaves either the old or the new
//!     head, never a torn one).
//!   * C2 replaces the promote-then-write sequence with the full rotation
//!     ordering — *re-sign repo head → durably write head → seal DidOp →
//!     publish DID doc* — as one crash-atomic unit, so no crash can leave a
//!     head naming a key whose repo head was not durably re-signed. Until C2
//!     lands, a crash inside the rotation task's promote→write window can
//!     leave the head pointing at a freshly promoted key whose commit has not
//!     yet been re-published; the reader trusts the head anyway (it is
//!     node-signed), so the next publish signs with that key. That is the
//!     window C2 closes.
//!
//! See the PR body for the full C2 contract.
//!
//! ## What this is NOT
//!
//! Not the federated `DidOp` witness chain (C1 #1167 / C5 #1171). The head is
//! a **node-attested projection** for cross-process readers on ONE node,
//! signed by a purpose-key derived from the node's CA root. It deliberately
//! does not encode rotation-key custody (epic #1158 Q2 / one-way door #4), so
//! it does not touch that one-way door. The projection carries its own
//! `version` so it can evolve independently of the DidOp format C1 finalizes.

use anyhow::{anyhow, Context as _, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{Signature, Signer, SigningKey as Ed25519SigningKey, Verifier};
use p256::ecdsa::{SigningKey as Es256SigningKey, VerifyingKey as Es256VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::path::{Path, PathBuf};

use super::jwt::es256_kid;
use super::key_rotation::Es256SigningKeyStore;

/// Purpose label for the Ed25519 key that signs the sealed head. Derived from
/// the node's CA root, so every process that can load the root can verify.
pub const HEAD_SIGNING_PURPOSE: &str = "hyprstream-oplog-head-v1";

/// Derive the head-signing key from the node root. Deterministic, so the
/// rotator (signs) and any cross-process reader (verifies) compute the same
/// keypair without key distribution. The CA derivation step keeps this key
/// distinct from the JWT CA key itself.
pub fn head_signing_key_from_root(root: &Ed25519SigningKey) -> Ed25519SigningKey {
    let ca = hyprstream_rpc::node_identity::derive_purpose_key(root, "hyprstream-jwt-v1");
    hyprstream_rpc::node_identity::derive_purpose_key(&ca, HEAD_SIGNING_PURPOSE)
}

/// Projection format version. Versions THIS PROJECTION only — NOT the DidOp
/// format (C1 #1167 owns that, gated on epic #1158 Q2 / one-way door #4).
pub const PROJECTION_VERSION: u16 = 1;

/// Filename of the sealed head inside the shared secrets dir.
pub const SEALED_HEAD_FILENAME: &str = "atproto-oplog-head.json";

/// Path of the sealed head inside a secrets dir.
pub fn sealed_head_path(secrets_dir: &Path) -> PathBuf {
    secrets_dir.join(SEALED_HEAD_FILENAME)
}

/// The on-disk sealed head: an authenticated, ordered projection of "which
/// `#atproto` ES256 generation is active right now."
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SealedOpLogHead {
    pub version: u16,
    /// Monotonic generation counter; strictly advances. Readers reject a
    /// regression so a stale head file can never shadow a newer one.
    pub seq: u64,
    /// `kid` of the active `#atproto` ES256 key.
    pub active_kid: String,
    /// Active P-256 verifying key, SEC1 compressed, base64url (33 raw bytes).
    pub active_vk_sec1: String,
    /// Repo head CID re-signed by the active key (`head_at_op` in plan §3.1).
    /// `None` until the first publish after this promotion. Forward-looking:
    /// C3 (#1169) consumes this for historical key validity.
    pub head_at_op: Option<String>,
    /// `sha256` of the canonical payload of the previous head (`None` at
    /// genesis). Forward-looking: C3 walks this chain. C4's reader checks the
    /// node signature + `seq` monotonicity, not the full chain.
    pub prev: Option<[u8; 32]>,
    /// Ed25519 signature by [`head_signing_key_from_root`] over
    /// [`SealedOpLogHead::signing_payload`].
    pub sig: Vec<u8>,
}

impl SealedOpLogHead {
    /// The bytes the head signature covers: canonical JSON of every field
    /// except `sig`. Field declaration order is stable, so `serde_json` output
    /// is deterministic across builds.
    pub fn signing_payload(&self) -> Result<Vec<u8>> {
        #[derive(Serialize)]
        struct Payload<'a> {
            version: u16,
            seq: u64,
            active_kid: &'a str,
            active_vk_sec1: &'a str,
            head_at_op: &'a Option<String>,
            prev: &'a Option<[u8; 32]>,
        }
        serde_json::to_vec(&Payload {
            version: self.version,
            seq: self.seq,
            active_kid: &self.active_kid,
            active_vk_sec1: &self.active_vk_sec1,
            head_at_op: &self.head_at_op,
            prev: &self.prev,
        })
        .context("serializing sealed-head payload")
    }

    /// Verify the node signature and return the active P-256 verifying key.
    pub fn verify(&self, head_vk: &ed25519_dalek::VerifyingKey) -> Result<Es256VerifyingKey> {
        let sig =
            Signature::from_slice(&self.sig).context("sealed head signature is malformed")?;
        head_vk
            .verify(&self.signing_payload()?, &sig)
            .context("sealed head signature does not verify under the node head key")?;
        if self.version != PROJECTION_VERSION {
            return Err(anyhow!(
                "sealed head projection version {}: reader supports {}",
                self.version,
                PROJECTION_VERSION
            ));
        }
        let vk_bytes = URL_SAFE_NO_PAD
            .decode(&self.active_vk_sec1)
            .context("active_vk_sec1 is not valid base64url")?;
        Es256VerifyingKey::from_sec1_bytes(&vk_bytes)
            .map_err(|e| anyhow!("invalid active_vk_sec1 in sealed head: {e}"))
    }
}

/// The active `#atproto` generation, as resolved by a reader. Carries the
/// signing key (the publisher signs commits with it) and the verifying key.
/// The signing key is resolved **by kid** from the shared secrets dir — the
/// sealed head AUTHENTICATES which kid is active, so a stale in-memory slot
/// cannot present a retired key as live.
#[derive(Clone, Debug)]
pub struct ActiveGeneration {
    pub seq: u64,
    pub kid: String,
    pub verifying_key: Es256VerifyingKey,
    pub signing_key: Es256SigningKey,
    /// `None` until C3 (#1169) records the re-signed repo head.
    pub head_at_op: Option<String>,
}

/// Source of truth for the active `#atproto` generation.
///
/// **Consistency guarantee (stated explicitly).** A reader is NOT guaranteed
/// to observe a completed rotation instantaneously: the head is rewritten
/// after the rotation task promotes a slot, so there is a bounded propagation
/// delay (one rotation tick plus filesystem visibility). The fail-closed
/// contract that makes "eventual" safe:
///
///   * the reader verifies the head's node signature and rejects a `seq`
///     regression, so it can never accept a forged or rolled-back head;
///   * the reader resolves the signing key **by kid** from the shared secrets
///     dir — a kid absent from the projection is never loaded, so a retired
///     key is never presented as active;
///   * if the head is absent, corrupt, or unsigned-by-this-node, the reader
///     returns `Ok(None)` and the publisher **declines to sign** rather than
///     fall back to an untrusted local slot.
///
/// The worst observable case is a brief window in which the reader still
/// presents the *previous* generation (which remains valid as drain) until
/// the new head becomes visible — never a retired key presented as active.
pub trait ActiveGenerationSource: Send + Sync {
    /// The active generation, or `Ok(None)` when no sealed generation is
    /// observable yet. `Err` is fail-closed: a corrupt/stale/unverifiable
    /// head that a reader must not trust.
    fn active_generation(&self) -> Result<Option<ActiveGeneration>>;
}

/// A single immutable generation — used to preserve the old
/// `PdsPublisher::new(store, did, signing_key)` call sites (tests, simple
/// embedders) behind the source abstraction.
pub struct FixedGenerationSource {
    kid: String,
    signing_key: Es256SigningKey,
}

impl FixedGenerationSource {
    pub fn new(signing_key: Es256SigningKey) -> Self {
        let kid = es256_kid(&signing_key);
        Self { kid, signing_key }
    }
}

impl ActiveGenerationSource for FixedGenerationSource {
    fn active_generation(&self) -> Result<Option<ActiveGeneration>> {
        Ok(Some(ActiveGeneration {
            seq: 0,
            kid: self.kid.clone(),
            verifying_key: *self.signing_key.verifying_key(),
            signing_key: self.signing_key.clone(),
            head_at_op: None,
        }))
    }
}

/// Cross-process reader: reads the sealed op-log head, verifies the node
/// signature, and resolves the signing key by kid from the shared secrets
/// dir. This is the source that closes #1123 — the registry process observes
/// OAuth's rotations through the head file, not an in-memory `OnceLock`.
pub struct SealedHeadEs256Source {
    head_path: PathBuf,
    head_verifying_key: ed25519_dalek::VerifyingKey,
    secrets_dir: PathBuf,
}

impl SealedHeadEs256Source {
    /// Construct from the node root: derives the head verifying key so the
    /// reader and the rotator agree without key distribution.
    pub fn from_root(secrets_dir: &Path, node_root: &Ed25519SigningKey) -> Self {
        Self {
            head_path: sealed_head_path(secrets_dir),
            head_verifying_key: head_signing_key_from_root(node_root).verifying_key(),
            secrets_dir: secrets_dir.to_path_buf(),
        }
    }

    /// Construct from a known head verifying key (tests).
    pub fn with_head_verifying_key(
        secrets_dir: &Path,
        head_verifying_key: ed25519_dalek::VerifyingKey,
    ) -> Self {
        Self {
            head_path: sealed_head_path(secrets_dir),
            head_verifying_key,
            secrets_dir: secrets_dir.to_path_buf(),
        }
    }

    /// Read and verify the head without resolving signing material.
    fn read_head(&self) -> Result<Option<SealedOpLogHead>> {
        let bytes = match std::fs::read(&self.head_path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).context("reading sealed op-log head"),
        };
        let head: SealedOpLogHead =
            serde_json::from_slice(&bytes).context("sealed op-log head is corrupt")?;
        head.verify(&self.head_verifying_key)?;
        Ok(Some(head))
    }
}

impl ActiveGenerationSource for SealedHeadEs256Source {
    fn active_generation(&self) -> Result<Option<ActiveGeneration>> {
        let head = match self.read_head()? {
            Some(h) => h,
            None => return Ok(None),
        };
        // Resolve the signing key BY KID from the shared secrets dir. The head
        // authenticates *which* kid is active; the secrets dir provides the
        // material. A kid missing from the projection is never loaded, so a
        // retired key cannot be presented as the active generation.
        let signing_key = super::key_rotation::es256_signing_key_for_kid(
            &self.secrets_dir,
            &head.active_kid,
        )
        .ok_or_else(|| {
            anyhow!(
                "sealed head names active kid {} but its signing key is not \
                 materialized in the secrets dir",
                head.active_kid
            )
        })?;
        Ok(Some(ActiveGeneration {
            seq: head.seq,
            kid: head.active_kid,
            verifying_key: *signing_key.verifying_key(),
            signing_key,
            head_at_op: head.head_at_op,
        }))
    }
}

/// Write the sealed op-log head after a promotion (or at boot). This is the
/// **C2 (#1168) seam**: the write is rename-atomic at the file level, but the
/// full crash-atomic rotation ordering (re-sign repo head → write head → seal
/// DidOp → publish) is C2's guarantee — see the module docs.
///
/// `composite_ca_key` is the node's CA Ed25519 key (the same one
/// [`super::key_rotation::RotationStores`] carries); the head is signed by a
/// dedicated purpose-key derived from it.
pub fn seal_op_log_head(
    secrets_dir: &Path,
    composite_ca_key: &Ed25519SigningKey,
    es256_store: &Es256SigningKeyStore,
) -> Result<()> {
    let active = es256_store
        .0
        .blocking_read()
        .active
        .clone()
        .ok_or_else(|| anyhow!("cannot seal op-log head: ES256 store has no active slot"))?;
    let signing_key = (*active.key).clone();
    let verifying_key = signing_key.verifying_key();
    let kid = active.kid();
    let vk_sec1 = URL_SAFE_NO_PAD.encode(verifying_key.to_sec1_bytes());

    // Carry forward the chain + monotonic seq from the prior head so a reader
    // can reject a regression. The prior head's signature was already verified
    // when it was written; we re-verify on read, not on carry-forward.
    let (seq, prev) = match std::fs::read(sealed_head_path(secrets_dir)) {
        Ok(prior_bytes) => match serde_json::from_slice::<SealedOpLogHead>(&prior_bytes) {
            Ok(prior) => {
                let prev_hash = sha256_fixed(&prior.signing_payload()?);
                (prior.seq.saturating_add(1), Some(prev_hash))
            }
            // A corrupt prior head: do not carry forward a junk chain; start
            // fresh. The new head is node-signed, so a reader still verifies.
            Err(_) => (1, None),
        },
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (1, None),
        Err(e) => return Err(e).context("reading prior sealed op-log head"),
    };

    let mut head = SealedOpLogHead {
        version: PROJECTION_VERSION,
        seq,
        active_kid: kid,
        active_vk_sec1: vk_sec1,
        // head_at_op is recorded when the repo head is re-signed (C2's full
        // sequence). C4's projection carries forward the prior value.
        head_at_op: None,
        prev,
        sig: Vec::new(),
    };
    head.head_at_op = prior_head_at_op(secrets_dir);

    let head_sk = hyprstream_rpc::node_identity::derive_purpose_key(
        composite_ca_key,
        HEAD_SIGNING_PURPOSE,
    );
    let sig = head_sk.sign(&head.signing_payload()?);
    head.sig = sig.to_bytes().to_vec();

    super::identity_store::write_secret(
        secrets_dir,
        SEALED_HEAD_FILENAME,
        &serde_json::to_vec(&head)?,
    )?;
    Ok(())
}

fn prior_head_at_op(secrets_dir: &Path) -> Option<String> {
    let bytes = std::fs::read(sealed_head_path(secrets_dir)).ok()?;
    serde_json::from_slice::<SealedOpLogHead>(&bytes)
        .ok()
        .and_then(|h| h.head_at_op)
}

fn sha256_fixed(bytes: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize().into()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn ca_key() -> Ed25519SigningKey {
        Ed25519SigningKey::from_bytes(&[0x5a; 32])
    }

    /// The CA key — what `seal_op_log_head` receives in production (the
    /// rotation task's `composite_ca_key`). The reader derives the matching
    /// head verifying key from the root via [`head_signing_key_from_root`].
    fn derived_ca() -> Ed25519SigningKey {
        hyprstream_rpc::node_identity::derive_purpose_key(&ca_key(), "hyprstream-jwt-v1")
    }

    #[test]
    fn sealed_head_round_trips_and_verifies() {
        let dir = TempDir::new().unwrap();
        let sk = Es256SigningKey::random(&mut rand::rngs::OsRng);
        let store = Es256SigningKeyStore::new(super::super::key_rotation::Es256KeySlots {
            active: Some(super::super::key_rotation::Es256KeySlot::new(sk, 0, 0)),
            drain: None,
            lead: None,
        });
        seal_op_log_head(dir.path(), &derived_ca(), &store).unwrap();

        let head: SealedOpLogHead =
            serde_json::from_slice(&std::fs::read(sealed_head_path(dir.path())).unwrap()).unwrap();
        let head_vk = head_signing_key_from_root(&ca_key()).verifying_key();
        assert_eq!(head.seq, 1);
        assert!(head.verify(&head_vk).is_ok());
        assert!(!head.sig.is_empty());
    }

    #[test]
    fn reader_fails_closed_on_tampered_head() {
        let dir = TempDir::new().unwrap();
        let sk = Es256SigningKey::random(&mut rand::rngs::OsRng);
        let store = Es256SigningKeyStore::new(super::super::key_rotation::Es256KeySlots {
            active: Some(super::super::key_rotation::Es256KeySlot::new(sk, 0, 0)),
            drain: None,
            lead: None,
        });
        seal_op_log_head(dir.path(), &derived_ca(), &store).unwrap();

        // Tamper with the on-disk head: flip a byte in the signature.
        let path = sealed_head_path(dir.path());
        let mut head: SealedOpLogHead =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        head.sig[0] ^= 0xff;
        std::fs::write(&path, serde_json::to_vec(&head).unwrap()).unwrap();

        let source = SealedHeadEs256Source::from_root(dir.path(), &ca_key());
        let err = source.active_generation().unwrap_err();
        assert!(
            err.to_string().contains("signature"),
            "tampered head must fail closed on signature: {err}"
        );
    }

    #[test]
    fn reader_returns_none_when_head_absent() {
        let dir = TempDir::new().unwrap();
        let source = SealedHeadEs256Source::from_root(dir.path(), &ca_key());
        assert!(source.active_generation().unwrap().is_none());
    }

    /// The #1123/#1170 acceptance test. Two "processes" share a secrets dir
    /// (the `--ipc` topology): process A is the rotator (holds the in-memory
    /// store, persists slots, seals the head); process B is the cross-process
    /// reader (reads the head + slot files only — no shared OnceLock, no
    /// event-delivery). B must observe A's rotation, and once it does it MUST
    /// NOT present the retired pre-rotation key.
    #[test]
    fn cross_process_rotation_observed_via_sealed_head() {
        use super::super::key_rotation::{
            persist_es256_slot, Es256KeySlot, Es256KeySlots, Es256SigningKeyStore,
        };
        let dir = TempDir::new().unwrap();
        let secrets = dir.path();
        let root = Ed25519SigningKey::from_bytes(&[0x5a; 32]);
        super::super::identity_store::write_ca_signing_key(secrets, &root).unwrap();
        // `seal_op_log_head` signs with a purpose key derived from the CA key
        // (the same key the rotation task carries as `composite_ca_key`); the
        // reader derives the matching verifying key from the root.
        let ca = hyprstream_rpc::node_identity::derive_purpose_key(&root, "hyprstream-jwt-v1");

        // ── Process A: active K1, persisted + head sealed ──────────────────
        let k1 = Es256SigningKey::random(&mut rand::rngs::OsRng);
        let k1_vk = *k1.verifying_key();
        let k1_slot = Es256KeySlot::new(k1.clone(), 0, 0);
        let store = Es256SigningKeyStore::new(Es256KeySlots {
            active: Some(k1_slot.clone()),
            drain: None,
            lead: None,
        });
        persist_es256_slot(secrets, "active", &k1_slot).unwrap();
        seal_op_log_head(secrets, &ca, &store).unwrap();

        // ── Process B: reads the sealed head ONLY ──────────────────────────
        let source = SealedHeadEs256Source::from_root(secrets, &root);
        let gen_b = source
            .active_generation()
            .unwrap()
            .expect("process B observes the initial generation K1");
        assert_eq!(gen_b.verifying_key, k1_vk);
        assert_eq!(gen_b.seq, 1);

        // ── Process A rotates: K1 → drain, K2 → active ────────────────────
        let k2 = Es256SigningKey::random(&mut rand::rngs::OsRng);
        let k2_vk = *k2.verifying_key();
        persist_es256_slot(secrets, "drain", &k1_slot).unwrap();
        let k2_slot = Es256KeySlot::new(k2.clone(), 0, 0);
        persist_es256_slot(secrets, "active", &k2_slot).unwrap();
        store.0.blocking_write().drain = Some(k1_slot);
        store.0.blocking_write().active = Some(k2_slot);
        seal_op_log_head(secrets, &ca, &store).unwrap();

        // ── Process B observes the rotation via a fresh head read ──────────
        let gen_b2 = source
            .active_generation()
            .unwrap()
            .expect("process B observes the rotated generation K2");
        assert_eq!(
            gen_b2.verifying_key,
            k2_vk,
            "B must observe the rotated key K2 — no propagation mechanism"
        );
        assert_eq!(gen_b2.seq, 2);

        // ── Negative case: once K2 is visible, B MUST NOT present K1 ───────
        assert_ne!(
            gen_b2.verifying_key, k1_vk,
            "a retired key must never be presented as the active generation"
        );
        assert_ne!(gen_b2.kid, gen_b.kid);
    }
}
