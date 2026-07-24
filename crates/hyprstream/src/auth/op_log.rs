//! C4 (#1170): the sealed op-log head — cross-process source of truth for the
//! active `#atproto` ES256 generation. **Closes #1123.**
//!
//! Plan t1124 §3.4: under `--ipc`, discovery/registry/oauth agree on the active
//! generation with **no event-delivery mechanism** — every process reads ONE
//! sealed head that the rotator (OAuth) writes after each promotion. The
//! process-local `OnceLock` stays the rotator's own working state; it is no
//! longer the source of truth for *other* processes.
//!
//! ## Credentials (split, least-privilege)
//!
//! The head is signed by a **dedicated** Ed25519 keypair — NOT a key derived
//! from the CA root. The private key (`oplog-head-key`) is a secret held only
//! by the rotator; the public verifying key (`atproto-oplog-head-pubkey`) is
//! published to the shared state dir and loaded by readers. A process that can
//! *verify* a head therefore **cannot forge** one (F3). Readers never touch
//! `ca-key` (which is PolicyService-only under systemd IPC).
//!
//! ## Head state directory (F4)
//!
//! The head is a signed, public artifact — it carries no secrecy, so it does
//! NOT live in the read-only per-unit credentials directory. It lives in a
//! dedicated **shared writable state dir** ([`resolve_oplog_state_dir`]),
//! env-overridable so a systemd `--ipc` deployment can point it at a shared
//! `StateDirectory=`. Where that shared storage is provisioned is deployment
//! plumbing (#808); this module takes the path, never assumes `secrets_dir`.
//!
//! ## Consistency guarantee (implemented)
//!
//! A reader is NOT guaranteed to observe a rotation instantaneously (bounded
//! by one rotation tick + filesystem visibility). The fail-closed contract:
//! the reader verifies the head's node signature, **rejects a `seq`
//! regression** against both the highest generation it has observed in memory
//! and the rotator's durable high-water mark, and resolves the signing key
//! **by kid** from the key-material store. A retired key is never presented as
//! active; replaying only an older-but-valid head is rejected even after a
//! reader restart. As with any node-local checkpoint, coordinated rollback of
//! the head and its high-water mark is outside this file-level guarantee.
//!
//! ## Dependency honesty — C2 (#1168) is not built
//!
//! [`seal_op_log_head`] is rename-atomic at the file level. C2 must harden
//! the full rotation ordering — *re-sign repo head → durably write head →
//! seal DidOp → publish* — as one crash-atomic unit, so no crash leaves a
//! head naming a key whose repo head was not durably re-signed. Full
//! crash-injection at every boundary is C2's job; the writer verifies every
//! prior head before carrying its `seq`/`prev` forward (F5). This is NOT the
//! DidOp format (C1 #1167) and does not encode rotation-key custody (epic
//! #1158 Q2 / one-way door #4); the projection carries its own `version`.

use anyhow::{anyhow, Context as _, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{
    Signature, Signer, SigningKey as Ed25519SigningKey, Verifier,
    VerifyingKey as Ed25519VerifyingKey,
};
use p256::ecdsa::{SigningKey as Es256SigningKey, VerifyingKey as Es256VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::jwt::es256_kid;
use super::key_rotation::Es256SigningKeyStore;

/// Projection format version (NOT the DidOp format — C1 #1167 owns that).
pub const PROJECTION_VERSION: u16 = 1;

/// Secret seed for the head-signing Ed25519 key (rotator only).
pub const HEAD_SIGNING_KEY_NAME: &str = "oplog-head-key";
/// Public verifying key for the head signature (all readers).
pub const HEAD_VERIFYING_KEY_FILENAME: &str = "atproto-oplog-head-pubkey";
/// The sealed head file.
pub const SEALED_HEAD_FILENAME: &str = "atproto-oplog-head.json";
/// Durable sequence high-water mark, written by the rotator after each head.
pub const SEALED_HEAD_MAX_SEQ_FILENAME: &str = "atproto-oplog-head-max-seq";
/// Env override for the shared head state dir (#808 systemd seam).
pub const OPLOG_STATE_DIR_ENV: &str = "HYPRSTREAM_OPLOG_STATE_DIR";

/// Resolve the shared, writable head state dir.
///
/// Default: a dedicated `oplog-state/` sibling of the secrets dir (shared and
/// writable in the single-process daemon). A systemd `--ipc` deployment sets
/// `HYPRSTREAM_OPLOG_STATE_DIR` to a shared `StateDirectory=` path —
/// provisioning that shared storage is #808.
pub fn resolve_oplog_state_dir(secrets_dir: &Path) -> Result<PathBuf> {
    if let Some(from_env) = std::env::var_os(OPLOG_STATE_DIR_ENV) {
        let path = PathBuf::from(from_env);
        std::fs::create_dir_all(&path)
            .with_context(|| format!("creating oplog state dir {}", path.display()))?;
        return Ok(path);
    }
    let parent = secrets_dir
        .parent()
        .ok_or_else(|| anyhow!("secrets dir has no parent for oplog state"))?;
    let path = parent.join("oplog-state");
    std::fs::create_dir_all(&path)
        .with_context(|| format!("creating oplog state dir {}", path.display()))?;
    Ok(path)
}

/// Load or initialize the dedicated head-signing Ed25519 key (rotator only).
pub fn load_or_init_head_signing_key(secrets_dir: &Path) -> Result<Ed25519SigningKey> {
    if let Some(bytes) = super::identity_store::read_secret(secrets_dir, HEAD_SIGNING_KEY_NAME)? {
        let mut seed = [0u8; 32];
        let v: Vec<u8> = bytes;
        seed.copy_from_slice(&v);
        return Ok(Ed25519SigningKey::from_bytes(&seed));
    }
    let key = Ed25519SigningKey::generate(&mut rand::rngs::OsRng);
    super::identity_store::write_secret(secrets_dir, HEAD_SIGNING_KEY_NAME, &key.to_bytes())?;
    Ok(key)
}

/// Publish the head-signing *verifying* key to the shared state dir (public).
pub fn publish_head_verifying_key(state_dir: &Path, head_sk: &Ed25519SigningKey) -> Result<()> {
    let vk = head_sk.verifying_key();
    atomic_write_public(&state_dir.join(HEAD_VERIFYING_KEY_FILENAME), &vk.to_bytes())
}

/// Atomic (rename) write with world-readable perms (0644). The head and its
/// verifying key are signed/public artifacts shared across services, which may
/// run as distinct users under systemd `--ipc` — so unlike secrets (0600)
/// they must be readable by every reader process.
fn atomic_write_public(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write as _;
    let dir = path
        .parent()
        .ok_or_else(|| anyhow!("head path has no parent directory"))?;
    let mut tmp = tempfile::NamedTempFile::new_in(dir)
        .with_context(|| format!("creating temp file in {}", dir.display()))?;
    tmp.write_all(bytes)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o644))?;
    }
    tmp.persist(path)
        .with_context(|| format!("persisting {}", path.display()))?;
    Ok(())
}

/// Load the head-signing verifying key from the shared state dir (readers).
pub fn load_head_verifying_key(state_dir: &Path) -> Result<Ed25519VerifyingKey> {
    let bytes = std::fs::read(state_dir.join(HEAD_VERIFYING_KEY_FILENAME))
        .context("head verifying key is not published (rotator has not run)")?;
    let arr: [u8; 32] = bytes
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("head verifying key must be 32 bytes"))?;
    Ed25519VerifyingKey::from_bytes(&arr).map_err(|e| anyhow!("invalid head verifying key: {e}"))
}

/// The on-disk sealed head: an authenticated, ordered projection of "which
/// `#atproto` ES256 generation is active right now."
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SealedOpLogHead {
    pub version: u16,
    /// Monotonic generation; strictly advances. Readers reject a regression.
    pub seq: u64,
    pub active_kid: String,
    /// Active P-256 verifying key, SEC1 compressed, base64url (33 raw bytes).
    pub active_vk_sec1: String,
    /// Repo head CID re-signed by the active key. `None` until C3 (#1169).
    pub head_at_op: Option<String>,
    /// sha256 of the canonical payload of the previous *verified* head.
    pub prev: Option<[u8; 32]>,
    /// Ed25519 signature by the head-signing key over [`Self::signing_payload`].
    pub sig: Vec<u8>,
}

impl SealedOpLogHead {
    /// Bytes the signature covers (every field except `sig`). Field order is
    /// stable, so `serde_json` output is deterministic.
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

    /// Verify the head signature under `head_vk` and return the active vk.
    pub fn verify(&self, head_vk: &Ed25519VerifyingKey) -> Result<Es256VerifyingKey> {
        let sig = Signature::from_slice(&self.sig).context("sealed head signature is malformed")?;
        head_vk
            .verify(&self.signing_payload()?, &sig)
            .context("sealed head signature does not verify under the head verifying key")?;
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

/// The active `#atproto` generation, as resolved by a reader.
#[derive(Clone, Debug)]
pub struct ActiveGeneration {
    pub seq: u64,
    pub kid: String,
    pub verifying_key: Es256VerifyingKey,
    pub signing_key: Es256SigningKey,
    pub head_at_op: Option<String>,
}

/// Source of truth for the active `#atproto` generation.
pub trait ActiveGenerationSource: Send + Sync {
    /// `Ok(None)` when no sealed generation is observable yet; `Err` is
    /// fail-closed (corrupt/stale/rolled-back/unverifiable head).
    fn active_generation(&self) -> Result<Option<ActiveGeneration>>;
}

/// A single immutable generation — preserves the old `PdsPublisher::new(store,
/// did, signing_key)` call sites behind the source abstraction.
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

/// Cross-process reader of the sealed op-log head. Holds NO private key
/// material — it loads the head-signing *verifying* key from the shared state
/// dir, so a process that can verify a head cannot forge one (F3). It tracks
/// the highest `seq` it has accepted and rejects any regression (F2).
pub struct SealedHeadEs256Source {
    state_dir: PathBuf,
    key_material_dir: PathBuf,
    /// Highest `seq` accepted by this process. A lower head is rejected as a
    /// rollback; the same head may be resolved repeatedly between rotations.
    max_seq: AtomicU64,
}

impl SealedHeadEs256Source {
    /// Construct from the shared head state dir + the ES256 key-material dir.
    /// Holds no credentials; the verifying key is loaded from the state dir at
    /// read time.
    pub fn new(state_dir: &Path, key_material_dir: &Path) -> Self {
        Self {
            state_dir: state_dir.to_path_buf(),
            key_material_dir: key_material_dir.to_path_buf(),
            max_seq: AtomicU64::new(0),
        }
    }

    fn head_path(&self) -> PathBuf {
        self.state_dir.join(SEALED_HEAD_FILENAME)
    }

    /// Read + verify the head (no signing-material resolution).
    fn read_head(&self) -> Result<Option<(SealedOpLogHead, Es256VerifyingKey)>> {
        let bytes = match std::fs::read(self.head_path()) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e).context("reading sealed op-log head"),
        };
        let head: SealedOpLogHead =
            serde_json::from_slice(&bytes).context("sealed op-log head is corrupt")?;
        let head_vk = load_head_verifying_key(&self.state_dir)?;
        let active_vk = head.verify(&head_vk)?;
        Ok(Some((head, active_vk)))
    }
}

impl ActiveGenerationSource for SealedHeadEs256Source {
    fn active_generation(&self) -> Result<Option<ActiveGeneration>> {
        let (head, _active_vk) = match self.read_head()? {
            Some(h) => h,
            None => return Ok(None),
        };
        // F2: reject a rolled-back head against the rotator's durable anchor,
        // including on a reader's first call after process restart.
        let durable_max = read_durable_max_seq(&self.state_dir)?.ok_or_else(|| {
            anyhow!(
                "sealed head exists without durable sequence anchor {}",
                SEALED_HEAD_MAX_SEQ_FILENAME
            )
        })?;
        if head.seq < durable_max {
            return Err(anyhow!(
                "sealed head seq {} is older than durable high-water mark {} \
                 (cold-start rollback or replay rejected)",
                head.seq,
                durable_max
            ));
        }

        // The same sealed generation is expected to serve many publishes.
        // Only a lower seq is a regression; equality is safe and necessary.
        let prev_max = self.max_seq.load(Ordering::Acquire);
        if head.seq < prev_max {
            return Err(anyhow!(
                "sealed head seq {} is older than the last accepted {} \
                 (rollback rejected)",
                head.seq,
                prev_max
            ));
        }
        // Resolve the signing key BY KID from the key-material store. The head
        // authenticates *which* kid is active; a kid it does not name is never
        // loaded, so a retired key cannot be presented as active.
        let signing_key = super::key_rotation::es256_signing_key_for_kid(
            &self.key_material_dir,
            &head.active_kid,
        )
        .ok_or_else(|| {
            anyhow!(
                "sealed head names active kid {} but its signing key is not \
                 materialized in the key-material dir",
                head.active_kid
            )
        })?;
        // Publish the new process-local high-water mark only after the key
        // resolves. `fetch_max` closes the race between concurrent readers.
        let concurrent_max = self.max_seq.fetch_max(head.seq, Ordering::AcqRel);
        if head.seq < concurrent_max {
            return Err(anyhow!(
                "sealed head seq {} is older than the concurrently accepted {} \
                 (rollback rejected)",
                head.seq,
                concurrent_max
            ));
        }
        Ok(Some(ActiveGeneration {
            seq: head.seq,
            kid: head.active_kid,
            verifying_key: *signing_key.verifying_key(),
            signing_key,
            head_at_op: head.head_at_op,
        }))
    }
}

/// Resolve the head state dir + dedicated head key, publish the verifying key,
/// and seal the current ES256 active generation in one call. This is the
/// rotator's full "advance the sealed head" step (boot + each rotation tick).
pub async fn advance_sealed_head(
    secrets_dir: &Path,
    es256_store: &Es256SigningKeyStore,
) -> Result<()> {
    let state_dir = resolve_oplog_state_dir(secrets_dir)?;
    let head_sk = load_or_init_head_signing_key(secrets_dir)?;
    publish_head_verifying_key(&state_dir, &head_sk)?;
    seal_op_log_head(&state_dir, &head_sk, es256_store).await
}

/// Write the sealed op-log head after a promotion (or at boot). **Async** — it
/// reads the ES256 store via `.read().await` (the store is a tokio RwLock);
/// the rotator calls it from its async context (F1). This is the **C2 (#1168)
/// seam**: rename-atomic at the file level; the full crash-atomic rotation
/// ordering is C2's guarantee (see module docs).
///
/// Prior state is carried forward only from a *signature-verified* prior head
/// (F5): an unverified/tampered prior does not inject `seq`/`prev`.
pub async fn seal_op_log_head(
    state_dir: &Path,
    head_signing_key: &Ed25519SigningKey,
    es256_store: &Es256SigningKeyStore,
) -> Result<()> {
    let active = es256_store
        .0
        .read()
        .await
        .active
        .clone()
        .ok_or_else(|| anyhow!("cannot seal op-log head: ES256 store has no active slot"))?;
    let signing_key = (*active.key).clone();
    let kid = active.kid();
    let vk_sec1 = URL_SAFE_NO_PAD.encode(signing_key.verifying_key().to_sec1_bytes());

    // F5: carry the chain forward only from a verified prior head.
    let (seq, prev) = prior_chain(state_dir, head_signing_key)?;

    let mut head = SealedOpLogHead {
        version: PROJECTION_VERSION,
        seq,
        active_kid: kid,
        active_vk_sec1: vk_sec1,
        head_at_op: None,
        prev,
        sig: Vec::new(),
    };
    let sig = head_signing_key.sign(&head.signing_payload()?);
    head.sig = sig.to_bytes().to_vec();

    atomic_write_public(
        &state_dir.join(SEALED_HEAD_FILENAME),
        &serde_json::to_vec(&head)?,
    )?;
    // Write the durable anchor after the head. If a crash lands between these
    // renames, the previous anchor already rejects every older head; the new
    // head remains acceptable because it is newer than that anchor.
    write_durable_max_seq(state_dir, head.seq)?;
    Ok(())
}

fn read_durable_max_seq(state_dir: &Path) -> Result<Option<u64>> {
    let path = state_dir.join(SEALED_HEAD_MAX_SEQ_FILENAME);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(e)
                .with_context(|| format!("reading sealed-head high-water mark {}", path.display()))
        }
    };
    let text = std::str::from_utf8(&bytes).context("sealed-head high-water mark is not UTF-8")?;
    let seq = text
        .parse::<u64>()
        .context("sealed-head high-water mark is not a u64")?;
    Ok(Some(seq))
}

fn write_durable_max_seq(state_dir: &Path, seq: u64) -> Result<()> {
    atomic_write_public(
        &state_dir.join(SEALED_HEAD_MAX_SEQ_FILENAME),
        seq.to_string().as_bytes(),
    )
}

/// Return `(seq, prev)` for the new head, carrying forward only from a
/// signature-verified prior head (F5). An absent or unverified prior breaks the
/// hash link, but the durable high-water mark still prevents `seq` reuse.
fn prior_chain(
    state_dir: &Path,
    head_signing_key: &Ed25519SigningKey,
) -> Result<(u64, Option<[u8; 32]>)> {
    let durable_max = read_durable_max_seq(state_dir)?.unwrap_or(0);
    let next_unlinked = || {
        durable_max
            .checked_add(1)
            .ok_or_else(|| anyhow!("sealed-head sequence exhausted at {durable_max}"))
            .map(|seq| (seq, None))
    };
    let head_vk = head_signing_key.verifying_key();
    let bytes = match std::fs::read(state_dir.join(SEALED_HEAD_FILENAME)) {
        Ok(b) => b,
        Err(_) => return next_unlinked(),
    };
    let prior: SealedOpLogHead = match serde_json::from_slice(&bytes) {
        Ok(h) => h,
        Err(_) => return next_unlinked(),
    };
    // F5: verify the prior head before trusting its seq/prev. A tampered prior
    // cannot inject an arbitrary sequence. The durable anchor still advances.
    if prior.verify(&head_vk).is_err() {
        return next_unlinked();
    }
    let prev_hash = match prior.signing_payload() {
        Ok(p) => sha256_fixed(&p),
        Err(_) => return next_unlinked(),
    };
    let seq = prior
        .seq
        .max(durable_max)
        .checked_add(1)
        .ok_or_else(|| anyhow!("sealed-head sequence exhausted at {}", u64::MAX))?;
    let prev = (prior.seq >= durable_max).then_some(prev_hash);
    Ok((seq, prev))
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
    use std::process::Command;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    const CROSS_PROCESS_CHILD_ENV: &str = "HYPRSTREAM_TEST_OPLOG_CHILD";
    const CROSS_PROCESS_STATE_ENV: &str = "HYPRSTREAM_TEST_OPLOG_STATE";
    const CROSS_PROCESS_SECRETS_ENV: &str = "HYPRSTREAM_TEST_OPLOG_SECRETS";
    const CROSS_PROCESS_READY_ENV: &str = "HYPRSTREAM_TEST_OPLOG_READY";
    const CROSS_PROCESS_CONTINUE_ENV: &str = "HYPRSTREAM_TEST_OPLOG_CONTINUE";
    const CROSS_PROCESS_RESULT_ENV: &str = "HYPRSTREAM_TEST_OPLOG_RESULT";

    /// A rotator fixture: dedicated head key + published verifying key + a
    /// populated ES256 store. `state_dir` is the shared head state dir;
    /// `secrets_dir` holds the head private key and ES256 key material.
    struct Rotator {
        state_dir: PathBuf,
        secrets_dir: PathBuf,
        head_sk: Ed25519SigningKey,
        store: Es256SigningKeyStore,
    }

    impl Rotator {
        fn new() -> Self {
            let state_dir = TempDir::new().unwrap().keep();
            let secrets_dir = TempDir::new().unwrap().keep();
            let head_sk = Ed25519SigningKey::generate(&mut rand::rngs::OsRng);
            publish_head_verifying_key(&state_dir, &head_sk).unwrap();
            Self {
                state_dir,
                secrets_dir,
                head_sk,
                store: Es256SigningKeyStore::new(empty_slots()),
            }
        }

        fn set_active(&self, sk: Es256SigningKey) {
            *self.store.0.blocking_write() = super::super::key_rotation::Es256KeySlots {
                active: Some(super::super::key_rotation::Es256KeySlot::new(sk, 0, 0)),
                drain: None,
                lead: None,
            };
        }

        fn seal(&self) {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(seal_op_log_head(
                    &self.state_dir,
                    &self.head_sk,
                    &self.store,
                ))
                .unwrap();
        }

        fn persist_slot(&self, name: &str, sk: &Es256SigningKey) {
            super::super::key_rotation::persist_es256_slot(
                &self.secrets_dir,
                name,
                &super::super::key_rotation::Es256KeySlot::new(sk.clone(), 0, 0),
            )
            .unwrap();
        }
    }

    fn empty_slots() -> super::super::key_rotation::Es256KeySlots {
        super::super::key_rotation::Es256KeySlots {
            active: None,
            drain: None,
            lead: None,
        }
    }

    // ── F1: the seal must work inside an async runtime (panics pre-fix) ─────

    #[test]
    fn seal_runs_inside_async_runtime() {
        // seal_op_log_head calls es256_store.0.read().await; pre-fix it used
        // blocking_read() which panics inside a runtime. Drive it under one.
        let r = Rotator::new();
        let k1 = Es256SigningKey::random(&mut rand::rngs::OsRng);
        r.set_active(k1.clone());
        r.persist_slot("active", &k1);
        r.seal();
        assert!(r.state_dir.join(SEALED_HEAD_FILENAME).exists());
    }

    // ── F2: a rolled-back head is rejected (replay-negative) ────────────────

    #[test]
    fn reader_rejects_rolled_back_head() {
        let r = Rotator::new();
        let k1 = Es256SigningKey::random(&mut rand::rngs::OsRng);
        let k1_vk = *k1.verifying_key();
        r.set_active(k1.clone());
        r.persist_slot("active", &k1);
        r.seal();
        let seq1_head = std::fs::read(r.state_dir.join(SEALED_HEAD_FILENAME)).unwrap();

        let reader = SealedHeadEs256Source::new(&r.state_dir, &r.secrets_dir);
        let gen1 = reader.active_generation().unwrap().unwrap();
        assert_eq!(gen1.verifying_key, k1_vk);

        // Rotate to K2.
        let k2 = Es256SigningKey::random(&mut rand::rngs::OsRng);
        let k2_vk = *k2.verifying_key();
        r.persist_slot("drain", &k1);
        r.persist_slot("active", &k2);
        r.set_active(k2);
        r.seal();
        let gen2 = reader.active_generation().unwrap().unwrap();
        assert_eq!(gen2.verifying_key, k2_vk);

        // Replay attack: overwrite the head with the still-valid signed seq=1.
        std::fs::write(r.state_dir.join(SEALED_HEAD_FILENAME), seq1_head).unwrap();
        let err = reader.active_generation().unwrap_err();
        assert!(
            err.to_string().contains("rollback"),
            "a replayed older signed head must be rejected: {err}"
        );
    }

    #[test]
    fn cold_start_reader_rejects_rolled_back_head() {
        let r = Rotator::new();
        let k1 = Es256SigningKey::random(&mut rand::rngs::OsRng);
        r.set_active(k1.clone());
        r.persist_slot("active", &k1);
        r.seal();
        let seq1_head = std::fs::read(r.state_dir.join(SEALED_HEAD_FILENAME)).unwrap();

        let k2 = Es256SigningKey::random(&mut rand::rngs::OsRng);
        r.persist_slot("drain", &k1);
        r.persist_slot("active", &k2);
        r.set_active(k2);
        r.seal();
        assert_eq!(read_durable_max_seq(&r.state_dir).unwrap(), Some(2));

        // Replay only the older, still-valid head, then construct a brand-new
        // source to model a process cold start. The durable anchor survives.
        std::fs::write(r.state_dir.join(SEALED_HEAD_FILENAME), seq1_head).unwrap();
        let restarted_reader = SealedHeadEs256Source::new(&r.state_dir, &r.secrets_dir);
        let err = restarted_reader.active_generation().unwrap_err();
        assert!(
            err.to_string().contains("cold-start rollback"),
            "a cold reader must reject an older signed head: {err}"
        );
    }

    #[test]
    fn reader_can_resolve_the_same_head_repeatedly() {
        let r = Rotator::new();
        let k1 = Es256SigningKey::random(&mut rand::rngs::OsRng);
        r.set_active(k1.clone());
        r.persist_slot("active", &k1);
        r.seal();

        let reader = SealedHeadEs256Source::new(&r.state_dir, &r.secrets_dir);
        assert_eq!(reader.active_generation().unwrap().unwrap().seq, 1);
        assert_eq!(reader.active_generation().unwrap().unwrap().seq, 1);
    }

    /// #1123/#1170 acceptance test: the rotator stays in this process while a
    /// child test process holds the reader across the K1 -> K2 rotation.
    #[test]
    fn cross_process_rotation_observed_via_sealed_head() {
        if std::env::var_os(CROSS_PROCESS_CHILD_ENV).is_some() {
            cross_process_reader_child();
            return;
        }

        let r = Rotator::new();
        let k1 = Es256SigningKey::random(&mut rand::rngs::OsRng);
        let k1_kid = es256_kid(&k1);
        r.set_active(k1.clone());
        r.persist_slot("active", &k1);
        r.seal();

        let coordination = TempDir::new().unwrap();
        let ready = coordination.path().join("ready");
        let continue_path = coordination.path().join("continue");
        let result = coordination.path().join("result");
        let test_name = "auth::op_log::tests::cross_process_rotation_observed_via_sealed_head";
        let mut child = Command::new(std::env::current_exe().unwrap())
            .arg("--exact")
            .arg(test_name)
            .arg("--nocapture")
            .env(CROSS_PROCESS_CHILD_ENV, "1")
            .env(CROSS_PROCESS_STATE_ENV, &r.state_dir)
            .env(CROSS_PROCESS_SECRETS_ENV, &r.secrets_dir)
            .env(CROSS_PROCESS_READY_ENV, &ready)
            .env(CROSS_PROCESS_CONTINUE_ENV, &continue_path)
            .env(CROSS_PROCESS_RESULT_ENV, &result)
            .spawn()
            .unwrap();

        wait_for_path(&ready);

        let k2 = Es256SigningKey::random(&mut rand::rngs::OsRng);
        let k2_kid = es256_kid(&k2);
        r.persist_slot("drain", &k1);
        r.persist_slot("active", &k2);
        r.set_active(k2);
        r.seal();
        std::fs::write(&continue_path, b"rotate").unwrap();

        let status = child.wait().unwrap();
        assert!(status.success(), "cross-process reader child failed");
        let observed = std::fs::read_to_string(result).unwrap();
        assert_eq!(observed, format!("{k1_kid}\n{k2_kid}\n"));
        assert_ne!(
            k1_kid, k2_kid,
            "the child must stop presenting the pre-rotation generation"
        );
    }

    fn cross_process_reader_child() {
        let state = PathBuf::from(std::env::var_os(CROSS_PROCESS_STATE_ENV).unwrap());
        let secrets = PathBuf::from(std::env::var_os(CROSS_PROCESS_SECRETS_ENV).unwrap());
        let ready = PathBuf::from(std::env::var_os(CROSS_PROCESS_READY_ENV).unwrap());
        let continue_path = PathBuf::from(std::env::var_os(CROSS_PROCESS_CONTINUE_ENV).unwrap());
        let result = PathBuf::from(std::env::var_os(CROSS_PROCESS_RESULT_ENV).unwrap());
        let reader = SealedHeadEs256Source::new(&state, &secrets);

        let before = reader.active_generation().unwrap().unwrap();
        std::fs::write(&ready, b"ready").unwrap();
        wait_for_path(&continue_path);
        let after = reader.active_generation().unwrap().unwrap();
        std::fs::write(result, format!("{}\n{}\n", before.kid, after.kid)).unwrap();
    }

    fn wait_for_path(path: &Path) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !path.exists() {
            assert!(
                Instant::now() < deadline,
                "timed out waiting for {}",
                path.display()
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    // ── F3: a reader (verifying key only) cannot forge a head ───────────────

    #[test]
    fn reader_rejects_head_signed_by_a_different_key() {
        let r = Rotator::new();
        let k1 = Es256SigningKey::random(&mut rand::rngs::OsRng);
        r.set_active(k1.clone());
        r.persist_slot("active", &k1);
        r.seal();

        // Attacker writes a head signed by a DIFFERENT key, but the reader
        // verifies against the published verifying key -> rejected.
        let attacker = Ed25519SigningKey::generate(&mut rand::rngs::OsRng);
        let mut forged = SealedOpLogHead {
            version: PROJECTION_VERSION,
            seq: 99,
            active_kid: es256_kid(&k1),
            active_vk_sec1: URL_SAFE_NO_PAD.encode(k1.verifying_key().to_sec1_bytes()),
            head_at_op: None,
            prev: None,
            sig: Vec::new(),
        };
        forged.sig = attacker
            .sign(&forged.signing_payload().unwrap())
            .to_bytes()
            .to_vec();
        std::fs::write(
            r.state_dir.join(SEALED_HEAD_FILENAME),
            serde_json::to_vec(&forged).unwrap(),
        )
        .unwrap();

        let reader = SealedHeadEs256Source::new(&r.state_dir, &r.secrets_dir);
        let err = reader.active_generation().unwrap_err();
        assert!(
            err.to_string().contains("signature"),
            "a head signed by a non-head key must fail signature verification: {err}"
        );
    }

    // ── F5: a tampered prior head does not inject an arbitrary seq ──────────

    #[test]
    fn tampered_prior_head_does_not_inject_seq() {
        let r = Rotator::new();
        let k1 = Es256SigningKey::random(&mut rand::rngs::OsRng);
        r.set_active(k1.clone());
        r.persist_slot("active", &k1);
        r.seal();

        // Tamper with the prior head's seq (without re-signing).
        let mut prior: SealedOpLogHead =
            serde_json::from_slice(&std::fs::read(r.state_dir.join(SEALED_HEAD_FILENAME)).unwrap())
                .unwrap();
        prior.seq = 999_999;
        std::fs::write(
            r.state_dir.join(SEALED_HEAD_FILENAME),
            serde_json::to_vec(&prior).unwrap(),
        )
        .unwrap();

        // Re-seal: the tampered prior does not verify, so the hash chain
        // restarts, but the durable sequence anchor still advances.
        let k2 = Es256SigningKey::random(&mut rand::rngs::OsRng);
        r.set_active(k2);
        r.seal();
        let head: SealedOpLogHead =
            serde_json::from_slice(&std::fs::read(r.state_dir.join(SEALED_HEAD_FILENAME)).unwrap())
                .unwrap();
        assert_eq!(
            head.seq, 2,
            "an unverified prior must not inject or regress the new head seq"
        );
        assert_eq!(head.prev, None);
    }

    // ── F4/torn: a corrupt head is rejected; absent head returns None ───────

    #[test]
    fn reader_fail_closed_on_corrupt_head() {
        let r = Rotator::new();
        std::fs::write(r.state_dir.join(SEALED_HEAD_FILENAME), b"not json").unwrap();
        let reader = SealedHeadEs256Source::new(&r.state_dir, &r.secrets_dir);
        assert!(reader.active_generation().is_err());
    }

    #[test]
    fn reader_returns_none_when_head_absent() {
        let r = Rotator::new();
        let reader = SealedHeadEs256Source::new(&r.state_dir, &r.secrets_dir);
        assert!(reader.active_generation().unwrap().is_none());
    }
}
