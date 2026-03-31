//! Persistent file-based credential storage.
//!
//! This module provides atomic read/write of secret key material to a
//! configurable directory (`secrets.path` in config, or the systemd
//! credentials directory when running under systemd).
//!
//! # Design
//!
//! - **No env var checks** — callers must resolve config bypasses before calling
//!   these functions.  This module is pure file I/O.
//! - **Atomic writes** — secrets are written via tempfile + rename so partial
//!   writes are never visible.
//! - **Mode 0600 / 0700** — secret files and their parent directory are created
//!   with restrictive permissions on Unix.
//! - **Read-only detection** — when the secrets directory is not writable (e.g.
//!   the systemd credentials ramfs), missing secrets are a hard error rather than
//!   triggering key generation.

use age::secrecy::ExposeSecret;
use anyhow::{anyhow, Context, Result};
use ed25519_dalek::{SigningKey, VerifyingKey};
use zeroize::Zeroizing;

use crate::server::tls::TlsMaterials;

// ─── Low-level primitives ───────────────────────────────────────────────────

/// Read a named secret from `dir`.  Returns `None` if the file does not exist.
pub fn read_secret(dir: &std::path::Path, name: &str) -> Result<Option<Vec<u8>>> {
    let path = dir.join(name);
    match std::fs::read(&path) {
        Ok(bytes) => Ok(Some(bytes)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow!("failed to read secret '{}': {}", path.display(), e)),
    }
}

/// Write a named secret to `dir` atomically (tempfile + rename).
///
/// Creates `dir` with mode 0700 if it does not exist.  The resulting file has
/// mode 0600.  Returns an error if the directory is not writable.
pub fn write_secret(dir: &std::path::Path, name: &str, value: &[u8]) -> Result<()> {
    ensure_secrets_dir(dir)?;
    let path = dir.join(name);

    // Write to a tempfile in the same directory, then rename for atomicity.
    let tmp_path = dir.join(format!(".{name}.tmp"));
    std::fs::write(&tmp_path, value)
        .with_context(|| format!("failed to write temp secret '{}'", tmp_path.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp_path, std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("failed to chmod secret '{}'", tmp_path.display()))?;
    }

    std::fs::rename(&tmp_path, &path)
        .with_context(|| format!("failed to rename secret to '{}'", path.display()))?;

    tracing::debug!("wrote secret '{}'", path.display());
    Ok(())
}

/// Returns `true` if `dir` exists and is writable (or can be created).
pub fn is_writable(dir: &std::path::Path) -> bool {
    if !dir.exists() {
        // Check whether the parent is writable instead.
        return dir
            .parent()
            .map(is_writable)
            .unwrap_or(false);
    }
    // Probe by attempting to create (and immediately remove) a tempfile.
    let probe = dir.join(".hyprstream-write-probe");
    match std::fs::write(&probe, b"") {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// Ensure the secrets directory exists with mode 0700.
fn ensure_secrets_dir(dir: &std::path::Path) -> Result<()> {
    if dir.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dir)
        .with_context(|| format!("failed to create secrets directory '{}'", dir.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
            .with_context(|| format!("failed to chmod secrets directory '{}'", dir.display()))?;
    }

    Ok(())
}

/// Return a hard error indicating that a required secret is missing from a
/// read-only credentials directory (e.g. the systemd credentials ramfs).
fn missing_in_readonly(secrets_dir: &std::path::Path, name: &str) -> anyhow::Error {
    anyhow!(
        "Secret '{}' not found in credentials directory '{}'.\n\
         The directory is not writable, so automatic key generation is not possible.\n\
         Re-run 'hyprstream service install' to provision credentials.",
        name,
        secrets_dir.display()
    )
}

// ─── High-level key loaders ─────────────────────────────────────────────────

/// Load or generate the Ed25519 node signing key.
///
/// Callers **must** check the `config.signing_key` bypass before calling this.
///
/// 1. Read `secrets_dir/signing-key` → return if present.
/// 2. If writable: generate new key, write, return.
/// 3. If read-only and missing: hard error.
pub fn load_or_generate_signing_key(secrets_dir: &std::path::Path) -> Result<SigningKey> {
    const NAME: &str = "signing-key";

    if let Some(bytes) = read_secret(secrets_dir, NAME)? {
        let arr: [u8; 32] = bytes
            .try_into()
            .map_err(|_| anyhow!("secret '{}' must be 32 bytes (Ed25519 seed)", NAME))?;
        tracing::info!("Loaded Ed25519 signing key from '{}'", secrets_dir.display());
        return Ok(SigningKey::from_bytes(&arr));
    }

    if !is_writable(secrets_dir) {
        return Err(missing_in_readonly(secrets_dir, NAME));
    }

    let key = SigningKey::generate(&mut rand::rngs::OsRng);
    write_secret(secrets_dir, NAME, &key.to_bytes())?;
    tracing::info!(
        "Generated new Ed25519 signing key → '{}/{}'",
        secrets_dir.display(),
        NAME
    );
    Ok(key)
}

/// Load or generate the age credential-store key.
///
/// Callers **must** check the `config.oauth.credential_store_key` bypass before
/// calling this.
///
/// When `store_file_path` already exists on disk and the key is missing from a
/// read-only secrets directory, the error includes recovery instructions (since
/// generating a new key would make the existing store unreadable).
pub fn load_or_generate_credential_store_key(
    secrets_dir: &std::path::Path,
    store_file_path: &std::path::Path,
) -> Result<age::x25519::Identity> {
    const NAME: &str = "credential-store-key";

    if let Some(bytes) = read_secret(secrets_dir, NAME)? {
        let s = String::from_utf8(bytes)
            .context("credential-store-key is not valid UTF-8")?;
        return s
            .trim()
            .parse::<age::x25519::Identity>()
            .map_err(|e| anyhow!("failed to parse credential-store-key: {:?}", e));
    }

    let store_exists = store_file_path.exists();

    if !is_writable(secrets_dir) {
        if store_exists {
            return Err(anyhow!(
                "Credential store key not found in credentials directory '{}'.\n\
                 The encrypted store exists at '{}' but cannot be decrypted.\n\
                 Recovery options:\n\
                 - Set HYPRSTREAM__OAUTH__CREDENTIAL_STORE_KEY=<age-secret-key-...> \
                   if you have a backup of the original key.\n\
                 - Delete the store file to start fresh (existing users will be lost).",
                secrets_dir.display(),
                store_file_path.display()
            ));
        }
        return Err(missing_in_readonly(secrets_dir, NAME));
    }

    if store_exists {
        // Store exists but key is gone — generating a new key would make the
        // existing store unreadable (age "no matching keys" error).
        return Err(anyhow!(
            "Credential store key not found in secrets directory '{}'.\n\
             The encrypted store exists at '{}' but cannot be decrypted.\n\
             Recovery options:\n\
             - Set HYPRSTREAM__OAUTH__CREDENTIAL_STORE_KEY=<age-secret-key-...> \
               if you have a backup of the original key.\n\
             - Delete the store file to start fresh (existing users will be lost).",
            secrets_dir.display(),
            store_file_path.display()
        ));
    }

    // First run — no store file, safe to generate a new key.
    let identity = age::x25519::Identity::generate();
    let secret_str = identity.to_string();
    write_secret(
        secrets_dir,
        NAME,
        secret_str.expose_secret().as_bytes(),
    )?;
    tracing::info!(
        "Generated new age credential-store key → '{}/{}'",
        secrets_dir.display(),
        NAME
    );
    Ok(identity)
}

/// Load or generate the user signing key (Ed25519).
///
/// Callers **must** check the `config.oauth.user_signing_key` bypass before
/// calling this.
///
/// 1. Read `secrets_dir/user-signing-key` → return if present.
/// 2. If writable: generate, write, return.
/// 3. If read-only and missing: hard error.
pub fn load_or_generate_user_signing_key(
    secrets_dir: &std::path::Path,
) -> Result<(SigningKey, VerifyingKey)> {
    const NAME: &str = "user-signing-key";

    if let Some(bytes) = read_secret(secrets_dir, NAME)? {
        let arr: [u8; 32] = bytes
            .try_into()
            .map_err(|_| anyhow!("secret '{}' must be 32 bytes (Ed25519 seed)", NAME))?;
        let sk = SigningKey::from_bytes(&arr);
        tracing::info!("Loaded user signing key from '{}'", secrets_dir.display());
        return Ok((sk.clone(), sk.verifying_key()));
    }

    if !is_writable(secrets_dir) {
        return Err(missing_in_readonly(secrets_dir, NAME));
    }

    let sk = SigningKey::generate(&mut rand::rngs::OsRng);
    write_secret(secrets_dir, NAME, &sk.to_bytes())?;
    tracing::info!(
        "Generated new user signing key → '{}/{}'",
        secrets_dir.display(),
        NAME
    );
    Ok((sk.clone(), sk.verifying_key()))
}

/// Load or generate TLS materials using the default secret names (`tls-key`, `tls-cert`).
///
/// See [`load_or_generate_tls_materials_named`] for details.
pub fn load_or_generate_tls_materials(
    secrets_dir: &std::path::Path,
    server_name: &str,
    max_validity_days: u32,
) -> Result<TlsMaterials> {
    load_or_generate_tls_materials_named(secrets_dir, server_name, max_validity_days, "tls-key", "tls-cert")
}

/// Load or generate TLS materials (ECDSA P-256 key + self-signed cert) using
/// caller-specified secret names.
///
/// - If `secrets_dir` contains both `key_name` and `cert_name`, load them.
/// - If the cert's file mtime is older than `(max_validity_days - 1)` days,
///   regenerate the cert (reusing the same key) to keep the key hash stable.
/// - If writable and files are missing, generate key + cert and write both.
/// - If read-only and files are missing, hard error.
///
/// `max_validity_days` should be 365 for HTTP services and 14 for QUIC/WebTransport.
///
/// Returns a [`TlsMaterials`] with DER-encoded cert and key.
pub fn load_or_generate_tls_materials_named(
    secrets_dir: &std::path::Path,
    server_name: &str,
    max_validity_days: u32,
    key_name: &str,
    cert_name: &str,
) -> Result<TlsMaterials> {
    let key_bytes = read_secret(secrets_dir, key_name)?;
    let cert_bytes = read_secret(secrets_dir, cert_name)?;

    match (key_bytes, cert_bytes) {
        (Some(key_der), Some(cert_der)) => {
            // Check if cert needs renewal based on file mtime.
            let needs_renewal = cert_renewal_needed(secrets_dir, cert_name, max_validity_days);

            if needs_renewal && is_writable(secrets_dir) {
                tracing::info!(
                    "TLS cert '{}' approaching expiry; regenerating (reusing existing key)", cert_name
                );
                let new_cert_der = generate_cert_from_key_der(&key_der, server_name, max_validity_days)?;
                write_secret(secrets_dir, cert_name, &new_cert_der)?;
                return Ok(TlsMaterials {
                    cert_der: new_cert_der,
                    key_der: Zeroizing::new(key_der),
                });
            }

            tracing::info!("Loaded persisted TLS materials from '{}'", secrets_dir.display());
            Ok(TlsMaterials {
                cert_der,
                key_der: Zeroizing::new(key_der),
            })
        }

        (Some(key_der), None) => {
            // Key exists but cert is missing — regenerate cert from existing key.
            if !is_writable(secrets_dir) {
                return Err(missing_in_readonly(secrets_dir, cert_name));
            }
            let cert_der = generate_cert_from_key_der(&key_der, server_name, max_validity_days)?;
            write_secret(secrets_dir, cert_name, &cert_der)?;
            tracing::info!("Regenerated TLS cert '{}' from persisted key", cert_name);
            Ok(TlsMaterials {
                cert_der,
                key_der: Zeroizing::new(key_der),
            })
        }

        (None, _) => {
            // No key — generate from scratch.
            if !is_writable(secrets_dir) {
                return Err(missing_in_readonly(secrets_dir, key_name));
            }
            let key_pair = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256)?;
            let key_der = key_pair.serialize_der();
            let cert_der = generate_cert_from_rcgen_keypair(&key_pair, server_name, max_validity_days)?;
            write_secret(secrets_dir, key_name, &key_der)?;
            write_secret(secrets_dir, cert_name, &cert_der)?;
            tracing::info!(
                "Generated new TLS key '{}' + cert '{}' ({max_validity_days}d) → '{}'",
                key_name, cert_name, secrets_dir.display()
            );
            Ok(TlsMaterials {
                cert_der,
                key_der: Zeroizing::new(key_der),
            })
        }
    }
}

// ─── TLS helpers ────────────────────────────────────────────────────────────

/// Check whether the persisted cert needs renewal.
///
/// Returns `true` if the cert file's mtime is older than `(max_validity_days - 1)` days.
fn cert_renewal_needed(secrets_dir: &std::path::Path, cert_name: &str, max_validity_days: u32) -> bool {
    let cert_path = secrets_dir.join(cert_name);
    let renewal_threshold = std::time::Duration::from_secs(
        u64::from(max_validity_days.saturating_sub(1)) * 86_400,
    );
    match std::fs::metadata(&cert_path).and_then(|m| m.modified()) {
        Ok(mtime) => match mtime.elapsed() {
            Ok(age) => age >= renewal_threshold,
            Err(_) => false,
        },
        Err(_) => true,
    }
}

/// Generate a self-signed cert DER from a raw DER-encoded ECDSA P-256 key.
fn generate_cert_from_key_der(
    key_der: &[u8],
    server_name: &str,
    max_validity_days: u32,
) -> Result<Vec<u8>> {
    let pki_key = rustls::pki_types::PrivateKeyDer::try_from(key_der)
        .map_err(|e| anyhow!("invalid TLS key DER: {}", e))?;
    let key_pair = rcgen::KeyPair::from_der_and_sign_algo(&pki_key, &rcgen::PKCS_ECDSA_P256_SHA256)
        .context("failed to parse persisted TLS key DER")?;
    generate_cert_from_rcgen_keypair(&key_pair, server_name, max_validity_days)
}

/// Generate a self-signed cert DER from an rcgen KeyPair.
fn generate_cert_from_rcgen_keypair(
    key_pair: &rcgen::KeyPair,
    server_name: &str,
    max_validity_days: u32,
) -> Result<Vec<u8>> {
    let mut params = rcgen::CertificateParams::new(vec![server_name.to_owned()])?;
    params.not_before = time::OffsetDateTime::now_utc();
    params.not_after =
        time::OffsetDateTime::now_utc() + time::Duration::days(i64::from(max_validity_days));
    let cert = params.self_signed(key_pair)?;
    Ok(cert.der().to_vec())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_read_write_roundtrip() {
        let dir = TempDir::new().unwrap();
        let data = b"hello world";
        write_secret(dir.path(), "test-key", data).unwrap();
        let loaded = read_secret(dir.path(), "test-key").unwrap().unwrap();
        assert_eq!(loaded, data);
    }

    #[test]
    fn test_read_missing_returns_none() {
        let dir = TempDir::new().unwrap();
        let result = read_secret(dir.path(), "nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    #[cfg(unix)]
    fn test_write_sets_mode_0600() {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        write_secret(dir.path(), "test-key", b"secret").unwrap();
        let meta = std::fs::metadata(dir.path().join("test-key")).unwrap();
        let mode = meta.permissions().mode();
        assert_eq!(mode & 0o777, 0o600, "expected mode 0600, got {:o}", mode & 0o777);
    }

    #[test]
    fn test_generate_on_first_run_writes_file() {
        let dir = TempDir::new().unwrap();
        let key = load_or_generate_signing_key(dir.path()).unwrap();
        assert_eq!(key.to_bytes().len(), 32);
        // File should now exist
        let raw = read_secret(dir.path(), "signing-key").unwrap().unwrap();
        assert_eq!(raw, key.to_bytes().to_vec());
    }

    #[test]
    fn test_load_persisted_signing_key() {
        let dir = TempDir::new().unwrap();
        // Write a known key
        let known = SigningKey::generate(&mut rand::rngs::OsRng);
        write_secret(dir.path(), "signing-key", &known.to_bytes()).unwrap();
        let loaded = load_or_generate_signing_key(dir.path()).unwrap();
        assert_eq!(loaded.to_bytes(), known.to_bytes());
    }

    #[test]
    fn test_signing_key_readonly_dir_fails() {
        // Use a path that doesn't exist and whose parent is the root (unwritable).
        let dir = std::path::PathBuf::from("/proc/hyprstream-test-readonly");
        let result = load_or_generate_signing_key(&dir);
        assert!(result.is_err());
    }

    #[test]
    fn test_credential_store_key_roundtrip() {
        let dir = TempDir::new().unwrap();
        let store_path = dir.path().join("users.toml.age");
        let id1 = load_or_generate_credential_store_key(dir.path(), &store_path).unwrap();
        // Load again from disk
        let id2 = load_or_generate_credential_store_key(dir.path(), &store_path).unwrap();
        // Both identities should produce the same public key
        assert_eq!(
            id1.to_public().to_string(),
            id2.to_public().to_string()
        );
    }

    #[test]
    fn test_credential_store_key_missing_with_existing_store_errors() {
        let dir = TempDir::new().unwrap();
        let store_path = dir.path().join("users.toml.age");
        // Simulate: store exists but key file does not
        std::fs::write(&store_path, b"fake encrypted data").unwrap();
        match load_or_generate_credential_store_key(dir.path(), &store_path) {
            Ok(_) => panic!("expected error when store exists but key is missing"),
            Err(e) => assert!(
                e.to_string().contains("cannot be decrypted"),
                "unexpected error message: {e}"
            ),
        }
    }

    #[test]
    fn test_user_signing_key_roundtrip() {
        let dir = TempDir::new().unwrap();
        let (sk1, vk1) = load_or_generate_user_signing_key(dir.path()).unwrap();
        let (sk2, vk2) = load_or_generate_user_signing_key(dir.path()).unwrap();
        assert_eq!(sk1.to_bytes(), sk2.to_bytes());
        assert_eq!(vk1.as_bytes(), vk2.as_bytes());
    }

    #[test]
    fn test_tls_materials_cert_hash_stable() {
        let dir = TempDir::new().unwrap();
        let m1 = load_or_generate_tls_materials(dir.path(), "localhost", 365).unwrap();
        let m2 = load_or_generate_tls_materials(dir.path(), "localhost", 365).unwrap();
        // Same cert bytes → same hash
        assert_eq!(m1.cert_der, m2.cert_der);
        assert_eq!(*m1.key_der, *m2.key_der);
    }

    #[test]
    fn test_tls_materials_cert_regenerated_from_same_key() {
        use sha2::{Digest, Sha256};
        let dir = TempDir::new().unwrap();
        let m1 = load_or_generate_tls_materials(dir.path(), "localhost", 365).unwrap();
        // Remove only the cert file to force regeneration
        std::fs::remove_file(dir.path().join("tls-cert")).unwrap();
        let m2 = load_or_generate_tls_materials(dir.path(), "localhost", 365).unwrap();
        // Key must be identical (stable identity)
        assert_eq!(*m1.key_der, *m2.key_der);
        // New cert is a different DER (different not_before/not_after)
        // but has the same public key embedded
        let h1 = Sha256::digest(&m1.cert_der);
        let h2 = Sha256::digest(&m2.cert_der);
        // Hashes differ because cert timestamps differ
        assert_ne!(h1, h2);
    }
}
