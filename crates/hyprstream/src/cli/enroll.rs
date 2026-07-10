//! Shared user-signing-key enrollment routine (#438 wizard + #439 `user create`).
//!
//! Enrolling a usable client identity is a register → store → bind sequence
//! whose silent partial failure leaves the CLI authenticating as `anonymous`
//! (#438/#439). This module is the single place that sequence lives, so the
//! bootstrap wizard and the `hyprstream user create` command cannot drift.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ed25519_dalek::{SigningKey, VerifyingKey};

use crate::auth::identity_store;
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
    /// SSH-style `SHA256:` fingerprint of the bound verifying key.
    pub fingerprint: String,
    /// Algorithm tag of the enrolled key (Ed25519 today).
    pub algorithm: KeyAlgorithm,
    /// Path of the prior `user-signing-key` backup, if an existing key was
    /// displaced by an adopt (`--ssh`/`--key`). `None` for `Generate`.
    pub key_backed_up: Option<PathBuf>,
}

/// The one shared enrollment routine.
///
/// 1. **Register** the user record if it does not already exist (`register`
///    overwrites pubkeys on an existing record, so it is only called when the
///    profile is absent).
/// 2. **Store** the resolved signing key as the client's actual
///    `credentials/user-signing-key` (backing up any prior key on the adopt
///    paths), so the CLI signs with *this* key.
/// 3. **Bind** the derived public key to the user, idempotently and
///    collision-safely (see [`bind_user_signing_key`]).
///
/// Returns the bound fingerprint and algorithm so callers can verify + print
/// enrollment actually took (`user create` shows it equals
/// `ssh-keygen -l -E sha256` of the key).
pub async fn enroll_user(
    store: &RocksDbUserStore,
    secrets_dir: &Path,
    username: &str,
    source: EnrollKeySource,
) -> Result<EnrollOutcome> {
    // 1. Register the user record if absent.
    if store.get_profile(username).await.ok().flatten().is_none() {
        store
            .register(username)
            .await
            .with_context(|| format!("Failed to register user identity '{username}'"))?;
    }

    // 2. Resolve the signing key and ensure it is installed as the client key.
    let (vk, backed_up) = resolve_and_store_signing_key(secrets_dir, source)?;

    // 3. Bind the derived public key to the user.
    bind_user_signing_key(store, username, vk).await?;

    Ok(EnrollOutcome {
        username: username.to_owned(),
        fingerprint: pubkey_fingerprint(&vk),
        algorithm: KeyAlgorithm::Ed25519,
        key_backed_up: backed_up,
    })
}

/// Resolve `source` to a verifying key, installing adopt-path keys as the
/// client signing key (with backup). Returns `(verifying_key, backup_path)`.
fn resolve_and_store_signing_key(
    secrets_dir: &Path,
    source: EnrollKeySource,
) -> Result<(VerifyingKey, Option<PathBuf>)> {
    match source {
        EnrollKeySource::Generate => {
            // load_or_generate already persists (or reuses) the on-disk key —
            // no adopt, no backup.
            let (_sk, vk) = identity_store::load_or_generate_user_signing_key(secrets_dir)?;
            Ok((vk, None))
        }
        EnrollKeySource::RawSeed(seed) => {
            let sk = SigningKey::from_bytes(&seed);
            let backup = identity_store::install_user_signing_key(secrets_dir, &sk)?;
            Ok((sk.verifying_key(), backup))
        }
        EnrollKeySource::SigningKey(sk) => {
            let vk = sk.verifying_key();
            let backup = identity_store::install_user_signing_key(secrets_dir, &sk)?;
            Ok((vk, backup))
        }
    }
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
pub(crate) async fn bind_user_signing_key(
    store: &RocksDbUserStore,
    username: &str,
    vk: VerifyingKey,
) -> Result<()> {
    let fingerprint = pubkey_fingerprint(&vk);

    match store.get_pubkey_user(&fingerprint).await? {
        Some(existing) if existing == username => {
            // Already bound to this user — nothing to do.
            return Ok(());
        }
        Some(existing) => {
            // Bound to a different (likely stale `anonymous`) user — re-point.
            tracing::info!("Re-pointing user-signing-key from '{existing}' to '{username}'");
            store.remove_pubkey(&existing, &fingerprint).await?;
        }
        None => {}
    }

    store
        .add_pubkey(username, vk, Some("enroll".to_owned()))
        .await?;
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

        let outcome = enroll_user(&store, secrets.path(), "alice", EnrollKeySource::Generate)
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
        enroll_user(&store, secrets.path(), "bob", EnrollKeySource::Generate)
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
        let outcome = enroll_user(&store, secrets.path(), "carol", EnrollKeySource::Generate)
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

        let outcome =
            enroll_user(&store, secrets.path(), "dan", EnrollKeySource::RawSeed(seed))
                .await
                .unwrap();
        assert_eq!(
            outcome.fingerprint,
            crate::auth::pubkey_fingerprint(&sk.verifying_key())
        );
        // No prior key → no backup.
        assert!(outcome.key_backed_up.is_none());
    }
}
