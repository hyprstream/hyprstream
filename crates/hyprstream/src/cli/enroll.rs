//! Shared user-signing-key enrollment routine (#438 wizard + #439 `user create`).
//!
//! Enrolling a usable client identity is a register → store → bind sequence
//! whose silent partial failure leaves the CLI authenticating as `anonymous`
//! (#438/#439). This module is the single place that sequence lives, so the
//! bootstrap wizard and the `hyprstream user create` command cannot drift.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ed25519_dalek::{SigningKey, VerifyingKey};
use hyprstream_rpc::crypto::CryptoPolicy;

use crate::auth::identity_store::{self, UserIdentityKeys};
use crate::auth::{KeyAlgorithm, RocksDbUserStore, UserStore, pubkey_fingerprint};

/// Where the client signing key comes from during enrollment.
#[derive(Debug, Clone)]
pub enum EnrollKeySource {
    /// Generate a fresh Ed25519 key, or reuse the existing on-disk
    /// `user-signing-key` if one is already present (the wizard path).
    Generate,
    /// Adopt a raw 32-byte Ed25519 seed, installing it as the client signing key.
    RawSeed([u8; 32]),
    /// Adopt an already-decoded Ed25519 signing key, installing it as the
    /// client signing key (e.g. parsed from an OpenSSH private key).
    SigningKey(SigningKey),
}

/// Result of enrolling a user identity.
#[derive(Debug, Clone)]
pub struct EnrollOutcome {
    pub username: String,
    /// SSH-style `SHA256:` fingerprint of the bound verifying key. Always the
    /// **Ed25519 anchor's** fingerprint (the kid) — a bound ML-DSA-65 key does
    /// not change it.
    pub fingerprint: String,
    /// Algorithm actually enrolled — [`KeyAlgorithm::HybridEd25519MlDsa65`]
    /// under a Hybrid policy, [`KeyAlgorithm::Ed25519`] under Classical. Never
    /// hardcoded; reflects what was really written.
    pub algorithm: KeyAlgorithm,
    /// Path of the prior `user-signing-key` backup, if an existing key was
    /// displaced by an adopt (`--ssh`/`--key`). `None` for `Generate`.
    pub key_backed_up: Option<PathBuf>,
    /// Path of the prior `user-signing-key.mldsa` backup, if a PQ sibling was
    /// displaced by an adopt. `None` otherwise.
    pub pq_key_backed_up: Option<PathBuf>,
    /// Human-facing notices the caller should print verbatim (e.g. a classical
    /// downgrade warning, or that a PQ component was generated for an adopted
    /// SSH/seed key). Never silent.
    pub notices: Vec<String>,
}

/// The one shared enrollment routine.
///
/// 1. **Register** the user record if it does not already exist (`register`
///    overwrites pubkeys on an existing record, so it is only called when the
///    profile is absent).
/// 2. **Store** the resolved signing key as the client's actual
///    `credentials/user-signing-key` (backing up any prior key on the adopt
///    paths), so the CLI signs with *this* key.
/// 3. **Bind** the derived public key(s) to the user, idempotently and
///    collision-safely (see [`bind_user_signing_key_material`]).
///
/// Returns the bound fingerprint and algorithm so callers can verify + print
/// enrollment actually took (`user create` shows it equals
/// `ssh-keygen -l -E sha256` of the key).
///
/// `policy` selects the crypto suite (see [`resolve_and_store_identity`]). Under
/// [`CryptoPolicy::Hybrid`] (the node default) enrollment mints an Ed25519
/// anchor **plus** a bound ML-DSA-65 key and writes a hybrid store record; a
/// missing/unpersistable PQ component is a hard error (fail closed). Under
/// The test-only [`CryptoPolicy::Classical`] variant writes a classical-only
/// identity with a loud notice — never a silent downgrade.
pub async fn enroll_user(
    store: &RocksDbUserStore,
    secrets_dir: &Path,
    username: &str,
    source: EnrollKeySource,
    policy: CryptoPolicy,
) -> Result<EnrollOutcome> {
    // 1. Register the user record if absent. Propagate a lookup *failure*
    //    rather than treating it as "absent" — `register` overwrites an
    //    existing record's bound pubkeys, so a swallowed error could drop a
    //    real user's keys.
    let existing = store
        .get_profile(username)
        .await
        .with_context(|| format!("Failed to look up user identity '{username}'"))?;
    if existing.is_none() {
        store
            .register(username)
            .await
            .with_context(|| format!("Failed to register user identity '{username}'"))?;
    }

    // 2. Resolve the identity key material (per source × policy) and install it
    //    as the client key(s).
    let resolved = resolve_and_store_identity(secrets_dir, source, policy)?;

    // 3. Bind the derived public key(s) to the user (hybrid record when a PQ
    //    component is present).
    let vk = resolved.keys.ed.verifying_key();
    let pq_vk = resolved.keys.pq_verifying_key_bytes();
    bind_user_signing_key_material(store, username, vk, pq_vk).await?;

    Ok(EnrollOutcome {
        username: username.to_owned(),
        fingerprint: pubkey_fingerprint(&vk),
        algorithm: resolved.keys.algorithm(),
        key_backed_up: resolved.ed_backup,
        pq_key_backed_up: resolved.pq_backup,
        notices: resolved.notices,
    })
}

/// Identity key material resolved from a source × policy, already installed on
/// disk as the client key(s).
struct ResolvedIdentity {
    keys: UserIdentityKeys,
    ed_backup: Option<PathBuf>,
    pq_backup: Option<PathBuf>,
    notices: Vec<String>,
}

/// The downgrade notice used by Classical-policy test fixtures.
fn classical_notice() -> String {
    "classical-only (policy=Classical): this identity is capped at Classical MAC \
     assurance. Production enrollment is pinned to Hybrid."
        .to_owned()
}

/// Resolve `source` × `policy` into installed identity key material.
///
/// - [`EnrollKeySource::Generate`]: load/generate the Ed25519 anchor and (under
///   Hybrid) the ML-DSA-65 sibling via
///   [`identity_store::load_or_generate_user_identity`] — this also covers the
///   in-place upgrade of a legacy Ed25519-only secrets dir.
/// - [`EnrollKeySource::RawSeed`] / [`EnrollKeySource::SigningKey`] (adopt): the
///   adopted key is the **classical component only**. Under Hybrid a fresh
///   ML-DSA-65 key is generated and bound to the same anchor (an SSH/seed key
///   can never make the identity hybrid by itself); a notice says so.
fn resolve_and_store_identity(
    secrets_dir: &Path,
    source: EnrollKeySource,
    policy: CryptoPolicy,
) -> Result<ResolvedIdentity> {
    match source {
        EnrollKeySource::Generate => {
            // load_or_generate persists (or reuses) the on-disk key(s) — no
            // adopt, no backup. Fails closed under Hybrid if the PQ component
            // cannot be provisioned.
            let keys = identity_store::load_or_generate_user_identity(secrets_dir, policy)?;
            let mut notices = Vec::new();
            if keys.mldsa.is_none() {
                notices.push(classical_notice());
            }
            Ok(ResolvedIdentity { keys, ed_backup: None, pq_backup: None, notices })
        }
        EnrollKeySource::RawSeed(seed) => {
            adopt_identity(secrets_dir, SigningKey::from_bytes(&seed), policy, &pq_generate)
        }
        EnrollKeySource::SigningKey(sk) => adopt_identity(secrets_dir, sk, policy, &pq_generate),
    }
}

/// Generate a fresh ML-DSA-65 signing key (indirection kept small so the adopt
/// path reads clearly).
fn pq_generate() -> hyprstream_rpc::crypto::pq::MlDsaSigningKey {
    hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair().0
}

/// Install an adopted Ed25519 key as the client anchor, minting + binding a
/// fresh ML-DSA-65 component under Hybrid.
fn adopt_identity(
    secrets_dir: &Path,
    ed_sk: SigningKey,
    policy: CryptoPolicy,
    pq_gen: &dyn Fn() -> hyprstream_rpc::crypto::pq::MlDsaSigningKey,
) -> Result<ResolvedIdentity> {
    let mut notices = Vec::new();
    let mldsa = if policy.uses_pq() {
        notices.push(
            "adopted SSH/seed key is classical; a post-quantum ML-DSA-65 component \
             was generated and bound to the same anchor."
                .to_owned(),
        );
        Some(pq_gen())
    } else {
        notices.push(classical_notice());
        None
    };

    let backup = identity_store::install_user_identity(secrets_dir, &ed_sk, mldsa.as_ref())?;
    let keys = UserIdentityKeys { ed: ed_sk, mldsa };
    Ok(ResolvedIdentity {
        keys,
        ed_backup: backup.ed,
        pq_backup: backup.pq,
        notices,
    })
}

/// Bind a user-signing-key public key to `username`, idempotently and
/// collision-safely.
///
/// `add_pubkey` rejects a fingerprint that is already bound (to the same or a
/// different user), so the current binding is resolved first:
/// - already bound to `username` → no-op (the key is already recognized),
/// - bound to a *different* user (e.g. an `anonymous` record left by a prior
///   partial run) → re-point it: remove from the old user, add to `username`,
/// - unbound → add it.
///
/// Re-pointing is the least-surprising behavior: there is exactly one local
/// user-signing-key, so it should map to exactly one identity — the one the
/// operator is enrolling.
///
/// Thin classical (no-PQ) wrapper over [`bind_user_signing_key_material`], used
/// by the wizard/enroll tests to bind a pre-existing on-disk key; production
/// enrollment goes through [`bind_user_signing_key_material`] directly.
#[cfg(test)]
pub(crate) async fn bind_user_signing_key(
    store: &RocksDbUserStore,
    username: &str,
    vk: VerifyingKey,
) -> Result<()> {
    bind_user_signing_key_material(store, username, vk, None).await
}

/// Bind a user-signing-key to `username`, carrying an optional bound ML-DSA-65
/// verifying key (a hybrid record when `pq_vk` is `Some`).
///
/// Collision handling is keyed by the **Ed25519 anchor fingerprint** (the PQ vk
/// never changes the kid), identical to the classical path:
/// - already bound to `username`: no-op for classical; for hybrid, perform the
///   in-place Ed25519 → Hybrid upgrade (idempotent) so a legacy classical record
///   for the same anchor gains its PQ binding;
/// - bound to a different (stale `anonymous`) user: re-point;
/// - unbound: add.
pub(crate) async fn bind_user_signing_key_material(
    store: &RocksDbUserStore,
    username: &str,
    vk: VerifyingKey,
    pq_vk: Option<Vec<u8>>,
) -> Result<()> {
    let fingerprint = pubkey_fingerprint(&vk);

    match store.get_pubkey_user(&fingerprint).await? {
        Some(existing) if existing == username => {
            // Already bound to this user. Classical → nothing to do. Hybrid →
            // upgrade in place (add_pubkey_hybrid is idempotent and lifts an
            // existing Ed25519 record to Hybrid for the same anchor).
            if let Some(pq) = pq_vk {
                store
                    .add_pubkey_hybrid(username, vk, pq, Some("enroll".to_owned()))
                    .await?;
            }
            return Ok(());
        }
        Some(existing) => {
            // Bound to a different (likely stale `anonymous`) user — re-point.
            tracing::info!("Re-pointing user-signing-key from '{existing}' to '{username}'");
            store.remove_pubkey(&existing, &fingerprint).await?;
        }
        None => {}
    }

    match pq_vk {
        Some(pq) => {
            store
                .add_pubkey_hybrid(username, vk, pq, Some("enroll".to_owned()))
                .await?;
        }
        None => {
            store
                .add_pubkey(username, vk, Some("enroll".to_owned()))
                .await?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn open(creds: &TempDir) -> RocksDbUserStore {
        RocksDbUserStore::open(creds.path()).unwrap()
    }

    #[tokio::test]
    async fn enroll_generate_registers_binds_and_tags_ed25519() {
        let creds = TempDir::new().unwrap();
        let secrets = TempDir::new().unwrap();
        let store = open(&creds);

        let outcome = enroll_user(
            &store,
            secrets.path(),
            "alice",
            EnrollKeySource::Generate,
            CryptoPolicy::Classical,
        )
        .await
        .unwrap();

        assert_eq!(outcome.username, "alice");
        assert_eq!(outcome.algorithm, KeyAlgorithm::Ed25519);
        assert!(outcome.fingerprint.starts_with("SHA256:"));
        assert!(outcome.key_backed_up.is_none(), "generate must not back up");

        // Profile exists and the fingerprint reverse-maps to the user.
        assert!(store.get_profile("alice").await.unwrap().is_some());
        assert_eq!(
            store.get_pubkey_user(&outcome.fingerprint).await.unwrap(),
            Some("alice".to_owned())
        );

        // The bound key equals the on-disk user-signing-key the CLI signs with.
        let (_sk, vk) = identity_store::load_or_generate_user_signing_key(secrets.path()).unwrap();
        assert_eq!(
            crate::auth::pubkey_fingerprint(&vk),
            outcome.fingerprint,
            "enrolled fingerprint must match the client signing key"
        );
    }

    #[tokio::test]
    async fn enroll_adopt_signing_key_installs_and_backs_up_existing() {
        let creds = TempDir::new().unwrap();
        let secrets = TempDir::new().unwrap();
        let store = open(&creds);

        // Seed the on-disk key first (simulate a prior key), then adopt a new one.
        let (_prior, _) =
            identity_store::load_or_generate_user_signing_key(secrets.path()).unwrap();
        let adopted = SigningKey::generate(&mut rand::rngs::OsRng);
        let outcome = enroll_user(
            &store,
            secrets.path(),
            "bob",
            EnrollKeySource::SigningKey(adopted),
            CryptoPolicy::Classical,
        )
        .await
        .unwrap();

        // A backup of the prior key must exist, and the installed key must be the adopted one.
        assert!(outcome.key_backed_up.as_ref().unwrap().exists());
        let (_sk, vk) = identity_store::load_or_generate_user_signing_key(secrets.path()).unwrap();
        assert_eq!(
            crate::auth::pubkey_fingerprint(&vk),
            outcome.fingerprint,
            "installed signing key must be the adopted key"
        );

        // Re-running enroll for bob is idempotent.
        enroll_user(
            &store,
            secrets.path(),
            "bob",
            EnrollKeySource::Generate,
            CryptoPolicy::Classical,
        )
        .await
        .unwrap();
        assert_eq!(store.list_pubkeys("bob").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn enroll_repoints_from_stale_anonymous_binding() {
        let creds = TempDir::new().unwrap();
        let secrets = TempDir::new().unwrap();
        let store = open(&creds);

        // Generate the on-disk key and bind it to a stale "anonymous" first.
        let (_sk, vk) =
            identity_store::load_or_generate_user_signing_key(secrets.path()).unwrap();
        store.register("anonymous").await.unwrap();
        bind_user_signing_key(&store, "anonymous", vk).await.unwrap();
        let fp = crate::auth::pubkey_fingerprint(&vk);
        assert_eq!(
            store.get_pubkey_user(&fp).await.unwrap(),
            Some("anonymous".to_owned())
        );

        // Enrolling the real user must re-point the key away from anonymous.
        let outcome = enroll_user(
            &store,
            secrets.path(),
            "carol",
            EnrollKeySource::Generate,
            CryptoPolicy::Classical,
        )
        .await
        .unwrap();
        assert_eq!(
            store.get_pubkey_user(&outcome.fingerprint).await.unwrap(),
            Some("carol".to_owned())
        );
    }

    #[tokio::test]
    async fn enroll_raw_seed_round_trips() {
        let creds = TempDir::new().unwrap();
        let secrets = TempDir::new().unwrap();
        let store = open(&creds);

        let sk = SigningKey::generate(&mut rand::rngs::OsRng);
        let seed = sk.to_bytes();

        let outcome = enroll_user(
            &store,
            secrets.path(),
            "dan",
            EnrollKeySource::RawSeed(seed),
            CryptoPolicy::Classical,
        )
        .await
        .unwrap();
        assert_eq!(
            outcome.fingerprint,
            crate::auth::pubkey_fingerprint(&sk.verifying_key())
        );
        // No prior key → no backup.
        assert!(outcome.key_backed_up.is_none());
    }

    // ─── Hybrid (post-quantum) enrollment (#439) ─────────────────────────────

    #[tokio::test]
    async fn enroll_generate_hybrid_writes_pq_record_and_sibling_file() {
        let creds = TempDir::new().unwrap();
        let secrets = TempDir::new().unwrap();
        let store = open(&creds);

        let outcome = enroll_user(
            &store,
            secrets.path(),
            "alice",
            EnrollKeySource::Generate,
            CryptoPolicy::Hybrid,
        )
        .await
        .unwrap();

        // The enrolled algorithm reflects reality: hybrid, not a hardcoded tag.
        assert_eq!(outcome.algorithm, KeyAlgorithm::HybridEd25519MlDsa65);
        assert!(outcome.notices.is_empty(), "hybrid generate has no downgrade notice");

        // Both on-disk key files exist, each mode 0600.
        let ed = secrets.path().join("user-signing-key");
        let pq = secrets.path().join("user-signing-key.mldsa");
        assert!(ed.exists() && pq.exists(), "both key files must be written");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for p in [&ed, &pq] {
                let mode = std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
                assert_eq!(mode, 0o600, "{} must be 0600", p.display());
            }
        }

        // The store record is hybrid and carries the bound ML-DSA-65 vk.
        let keys = store.list_pubkeys("alice").await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].algorithm, KeyAlgorithm::HybridEd25519MlDsa65);
        let pq_bytes = keys[0].pq_pubkey.as_ref().expect("hybrid record has pq_pubkey");
        // Matches the ML-DSA vk derived from the on-disk PQ seed.
        let loaded = identity_store::load_or_generate_user_identity(
            secrets.path(),
            CryptoPolicy::Hybrid,
        )
        .unwrap();
        assert_eq!(&loaded.pq_verifying_key_bytes().unwrap(), pq_bytes);
        // Fingerprint is the Ed25519 anchor's (kid), unchanged by the PQ key.
        assert_eq!(
            outcome.fingerprint,
            crate::auth::pubkey_fingerprint(&loaded.ed.verifying_key())
        );
    }

    #[tokio::test]
    async fn enroll_adopt_ssh_under_hybrid_generates_pq_and_notices() {
        let creds = TempDir::new().unwrap();
        let secrets = TempDir::new().unwrap();
        let store = open(&creds);

        // An adopted (SSH-style) key is the classical component; under Hybrid a
        // fresh PQ component is generated + bound, with a notice saying so.
        let adopted = SigningKey::generate(&mut rand::rngs::OsRng);
        let vk = adopted.verifying_key();
        let outcome = enroll_user(
            &store,
            secrets.path(),
            "erin",
            EnrollKeySource::SigningKey(adopted),
            CryptoPolicy::Hybrid,
        )
        .await
        .unwrap();

        assert_eq!(outcome.algorithm, KeyAlgorithm::HybridEd25519MlDsa65);
        assert_eq!(
            outcome.fingerprint,
            crate::auth::pubkey_fingerprint(&vk),
            "anchor fingerprint is the adopted Ed25519 key's"
        );
        assert!(
            outcome.notices.iter().any(|n| n.contains("post-quantum")),
            "must notify that a PQ component was generated for the adopted key"
        );
        assert!(secrets.path().join("user-signing-key.mldsa").exists());
        let keys = store.list_pubkeys("erin").await.unwrap();
        assert!(keys[0].pq_pubkey.is_some());
    }

    #[tokio::test]
    async fn enroll_hybrid_readonly_secrets_fails_closed() {
        let creds = TempDir::new().unwrap();
        let secrets = TempDir::new().unwrap();
        let store = open(&creds);

        // Pre-seed a classical Ed25519 key, then make the dir read-only so the
        // PQ sibling cannot be generated. Under Hybrid this MUST fail closed —
        // never mint a classical-only identity.
        let _ = identity_store::load_or_generate_user_signing_key(secrets.path()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(secrets.path(), std::fs::Permissions::from_mode(0o500))
                .unwrap();

            let err = enroll_user(
                &store,
                secrets.path(),
                "frank",
                EnrollKeySource::Generate,
                CryptoPolicy::Hybrid,
            )
            .await
            .expect_err("hybrid enrollment must fail closed on a read-only secrets dir");
            assert!(
                err.to_string().contains("fail closed")
                    || err.to_string().to_lowercase().contains("hybrid"),
                "error must name the fail-closed reason: {err}"
            );

            // Restore perms so TempDir cleanup works.
            std::fs::set_permissions(secrets.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
    }

    #[tokio::test]
    async fn enroll_legacy_ed25519_dir_upgrades_in_place_same_fingerprint() {
        let creds = TempDir::new().unwrap();
        let secrets = TempDir::new().unwrap();
        let store = open(&creds);

        // Legacy: classical enrollment first (Ed25519 file only, classical record).
        let classical = enroll_user(
            &store,
            secrets.path(),
            "grace",
            EnrollKeySource::Generate,
            CryptoPolicy::Classical,
        )
        .await
        .unwrap();
        assert_eq!(classical.algorithm, KeyAlgorithm::Ed25519);
        assert!(!secrets.path().join("user-signing-key.mldsa").exists());

        // Re-enroll under Hybrid: same anchor gains a PQ binding in place; the
        // fingerprint (kid) is unchanged, and the record becomes hybrid.
        let hybrid = enroll_user(
            &store,
            secrets.path(),
            "grace",
            EnrollKeySource::Generate,
            CryptoPolicy::Hybrid,
        )
        .await
        .unwrap();
        assert_eq!(
            hybrid.fingerprint, classical.fingerprint,
            "in-place upgrade must preserve the anchor fingerprint"
        );
        assert_eq!(hybrid.algorithm, KeyAlgorithm::HybridEd25519MlDsa65);
        let keys = store.list_pubkeys("grace").await.unwrap();
        assert_eq!(keys.len(), 1, "upgrade is in place, not a second key");
        assert!(keys[0].pq_pubkey.is_some());
    }
}
