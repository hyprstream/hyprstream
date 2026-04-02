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
use zeroize::{Zeroize, Zeroizing};

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
///
/// Uses `NamedTempFile` so the temporary file is automatically removed on any
/// failure path — no stale `.{name}.tmp` files are left on disk.
pub fn write_secret(dir: &std::path::Path, name: &str, value: &[u8]) -> Result<()> {
    use std::io::Write as _;
    ensure_secrets_dir(dir)?;
    let path = dir.join(name);

    let mut tmp = tempfile::NamedTempFile::new_in(dir)
        .with_context(|| format!("failed to create temp file in '{}'", dir.display()))?;

    tmp.write_all(value)
        .with_context(|| format!("failed to write secret in '{}'", dir.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o600))
            .with_context(|| format!("failed to chmod secret in '{}'", dir.display()))?;
    }

    tmp.persist(&path)
        .with_context(|| format!("failed to persist secret to '{}'", path.display()))?;

    tracing::debug!("wrote secret '{}'", path.display());
    Ok(())
}

/// Returns `true` if `dir` exists and is writable (or can be created).
///
/// Uses `tempfile::tempfile_in` so no named probe file is left on disk,
/// even if the process is killed during the check.
pub fn is_writable(dir: &std::path::Path) -> bool {
    if !dir.exists() {
        return dir.parent().map(is_writable).unwrap_or(false);
    }
    tempfile::tempfile_in(dir).is_ok()
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

/// Load or generate the Ed25519 **node** signing key (the root-of-trust key
/// that identifies this Hyprstream instance).
///
/// Callers **must** check `HyprConfig::node_signing_key_bypass()` before
/// calling this. This function is pure file I/O with no config awareness.
///
/// 1. Read `secrets_dir/signing-key` → return if present.
/// 2. If writable: generate new key, write, return.
/// 3. If read-only and missing: hard error.
pub fn load_or_generate_node_signing_key(secrets_dir: &std::path::Path) -> Result<SigningKey> {
    const NAME: &str = "signing-key";

    if let Some(mut bytes) = read_secret(secrets_dir, NAME)? {
        let mut arr: [u8; 32] = bytes.as_slice()
            .try_into()
            .map_err(|_| anyhow!("secret '{}' must be 32 bytes (Ed25519 seed)", NAME))?;
        let sk = SigningKey::from_bytes(&arr);
        bytes.zeroize();
        arr.zeroize();
        tracing::info!("Loaded Ed25519 signing key from '{}'", secrets_dir.display());
        return Ok(sk);
    }

    if !is_writable(secrets_dir) {
        return Err(missing_in_readonly(secrets_dir, NAME));
    }

    let key = SigningKey::generate(&mut rand::rngs::OsRng);
    let mut raw = key.to_bytes();
    let result = write_secret(secrets_dir, NAME, &raw);
    raw.zeroize();
    result?;
    tracing::info!(
        "Generated new Ed25519 signing key → '{}/{}'",
        secrets_dir.display(),
        NAME
    );
    Ok(key)
}

/// Deprecated alias for [`load_or_generate_node_signing_key`].
#[deprecated(since = "0.4.1", note = "renamed to load_or_generate_node_signing_key")]
pub fn load_or_generate_signing_key(secrets_dir: &std::path::Path) -> Result<SigningKey> {
    load_or_generate_node_signing_key(secrets_dir)
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
            .map_err(|e| anyhow!("failed to parse credential-store-key: {}", e));
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
    let result = write_secret(
        secrets_dir,
        NAME,
        secret_str.expose_secret().as_bytes(),
    );
    drop(secret_str); // SecretString zeroizes on drop
    result?;
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

    if let Some(mut bytes) = read_secret(secrets_dir, NAME)? {
        let mut arr: [u8; 32] = bytes.as_slice()
            .try_into()
            .map_err(|_| anyhow!("secret '{}' must be 32 bytes (Ed25519 seed)", NAME))?;
        let sk = SigningKey::from_bytes(&arr);
        bytes.zeroize();
        arr.zeroize();
        let vk = sk.verifying_key();
        tracing::info!("Loaded user signing key from '{}'", secrets_dir.display());
        return Ok((sk, vk));
    }

    if !is_writable(secrets_dir) {
        return Err(missing_in_readonly(secrets_dir, NAME));
    }

    let sk = SigningKey::generate(&mut rand::rngs::OsRng);
    let mut raw = sk.to_bytes();
    let result = write_secret(secrets_dir, NAME, &raw);
    raw.zeroize();
    result?;
    let vk = sk.verifying_key();
    tracing::info!(
        "Generated new user signing key → '{}/{}'",
        secrets_dir.display(),
        NAME
    );
    Ok((sk, vk))
}

/// Load or generate an RSA 2048 keypair for RS256 JWT signing.
///
/// Stored as PKCS#8 DER in `secrets_dir/rsa-key`. If the file doesn't exist
/// and the directory is writable, a new keypair is generated using `openssl`.
///
/// Returns the DER-encoded PKCS#8 private key bytes (suitable for
/// `jsonwebtoken::EncodingKey::from_rsa_der`).
pub fn load_or_generate_rsa_key(secrets_dir: &std::path::Path) -> Result<Vec<u8>> {
    const NAME: &str = "rsa-key";

    if let Some(bytes) = read_secret(secrets_dir, NAME)? {
        tracing::info!("Loaded RSA key from '{}'", secrets_dir.display());
        return Ok(bytes);
    }

    if !is_writable(secrets_dir) {
        return Err(missing_in_readonly(secrets_dir, NAME));
    }

    // Generate RSA 2048 keypair via openssl (avoids adding rsa crate dependency).
    // Output is PKCS#8 DER, compatible with jsonwebtoken::EncodingKey::from_rsa_der.
    let output = std::process::Command::new("openssl")
        .args(["genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-outform", "DER"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .context("Failed to run openssl for RSA key generation")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("openssl RSA key generation failed: {stderr}"));
    }

    let der_bytes = output.stdout;
    if der_bytes.len() < 100 {
        return Err(anyhow!("openssl produced unexpectedly small RSA key ({} bytes)", der_bytes.len()));
    }

    write_secret(secrets_dir, NAME, &der_bytes)?;
    tracing::info!("Generated new RSA 2048 key → '{}/{}'", secrets_dir.display(), NAME);
    Ok(der_bytes)
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
///
/// # Note on mtime-as-proxy
///
/// We use the cert file's mtime as a proxy for its `notBefore` date to avoid
/// parsing DER. This is a heuristic: over-renewing is safe (one extra I/O on
/// startup), while under-renewing could cause expired-cert errors in clients.
/// Clock skew (mtime in the future) is handled conservatively — we trigger
/// renewal rather than silently skipping it.
fn cert_renewal_needed(secrets_dir: &std::path::Path, cert_name: &str, max_validity_days: u32) -> bool {
    let cert_path = secrets_dir.join(cert_name);
    let renewal_threshold = std::time::Duration::from_secs(
        u64::from(max_validity_days.saturating_sub(1)) * 86_400,
    );
    match std::fs::metadata(&cert_path).and_then(|m| m.modified()) {
        Ok(mtime) => match mtime.elapsed() {
            Ok(age) => age >= renewal_threshold,
            Err(_) => {
                tracing::warn!(
                    "cert '{}': mtime is in the future (clock skew?); triggering renewal",
                    cert_path.display()
                );
                true
            }
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

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Back-date a file's mtime by `days` days using nix utimes.
    #[cfg(unix)]
    fn backdate_mtime(path: &std::path::Path, days: i64) {
        use nix::sys::time::TimeVal;
        let past_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64
            - days * 86_400;
        let tv = TimeVal::new(past_secs, 0);
        nix::sys::stat::utimes(path, &tv, &tv).unwrap();
    }

    #[test]
    fn test_is_writable_leaves_no_probe_file() {
        let dir = TempDir::new().unwrap();
        assert!(is_writable(dir.path()));
        let count = std::fs::read_dir(dir.path()).unwrap().count();
        assert_eq!(count, 0, "is_writable should not leave any files behind");
    }

    #[test]
    fn test_write_secret_no_tmp_remnant() {
        let dir = TempDir::new().unwrap();
        write_secret(dir.path(), "mykey", b"value").unwrap();
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
            .collect();
        assert_eq!(names, vec!["mykey"], "only the final file should exist");
    }

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
        let key = load_or_generate_node_signing_key(dir.path()).unwrap();
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
        let loaded = load_or_generate_node_signing_key(dir.path()).unwrap();
        assert_eq!(loaded.to_bytes(), known.to_bytes());
    }

    #[test]
    #[cfg(unix)]
    fn test_signing_key_readonly_dir_fails() {
        use std::os::unix::fs::PermissionsExt;
        // 0o500 = r-x------ (owner read+traverse, no write, no group/other access).
        // Mirrors a systemd credentials directory.  TempDir::drop uses rmdir which
        // only needs write on the *parent* dir, so cleanup succeeds without restoring.
        let parent = TempDir::new().unwrap();
        let secrets_dir = parent.path().join("secrets");
        std::fs::create_dir(&secrets_dir).unwrap();
        std::fs::set_permissions(&secrets_dir, std::fs::Permissions::from_mode(0o500)).unwrap();
        let result = load_or_generate_node_signing_key(&secrets_dir);
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

    // ── Directory mode 0700 ──────────────────────────────────────────────────

    #[test]
    #[cfg(unix)]
    fn test_ensure_dir_mode_0700() {
        use std::os::unix::fs::PermissionsExt;
        let parent = TempDir::new().unwrap();
        let secrets_dir = parent.path().join("new-secrets");
        // Dir does not exist yet; write_secret should create it with mode 0700.
        write_secret(&secrets_dir, "k", b"v").unwrap();
        let meta = std::fs::metadata(&secrets_dir).unwrap();
        assert_eq!(
            meta.permissions().mode() & 0o777,
            0o700,
            "secrets dir should be 0700, got {:o}",
            meta.permissions().mode() & 0o777
        );
    }

    // ── Binary data roundtrip ────────────────────────────────────────────────

    #[test]
    fn test_read_write_binary_roundtrip() {
        let dir = TempDir::new().unwrap();
        // Include null bytes and high-bit bytes to verify binary fidelity.
        let data: Vec<u8> = (0u8..=255).collect();
        write_secret(dir.path(), "binary-key", &data).unwrap();
        let loaded = read_secret(dir.path(), "binary-key").unwrap().unwrap();
        assert_eq!(loaded, data);
    }

    // ── Invalid signing key length ───────────────────────────────────────────

    #[test]
    fn test_load_signing_key_wrong_length_errors() {
        let dir = TempDir::new().unwrap();
        // Write 16 bytes instead of 32 — should produce a clear error.
        write_secret(dir.path(), "signing-key", &[0u8; 16]).unwrap();
        let result = load_or_generate_node_signing_key(dir.path());
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("32 bytes"),
            "error should mention expected 32-byte length"
        );
    }

    #[test]
    fn test_load_user_signing_key_wrong_length_errors() {
        let dir = TempDir::new().unwrap();
        write_secret(dir.path(), "user-signing-key", &[0u8; 64]).unwrap();
        let result = load_or_generate_user_signing_key(dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("32 bytes"));
    }

    // ── User signing key: generate + readonly paths ──────────────────────────

    #[test]
    fn test_user_signing_key_generates_on_first_run() {
        let dir = TempDir::new().unwrap();
        let (sk, vk) = load_or_generate_user_signing_key(dir.path()).unwrap();
        assert_eq!(sk.to_bytes().len(), 32);
        assert_eq!(vk.as_bytes().len(), 32);
        // Key file should now exist.
        let raw = read_secret(dir.path(), "user-signing-key").unwrap().unwrap();
        assert_eq!(raw, sk.to_bytes().as_slice());
    }

    #[test]
    #[cfg(unix)]
    fn test_user_signing_key_readonly_dir_fails() {
        use std::os::unix::fs::PermissionsExt;
        let parent = TempDir::new().unwrap();
        let secrets_dir = parent.path().join("secrets");
        std::fs::create_dir(&secrets_dir).unwrap();
        std::fs::set_permissions(&secrets_dir, std::fs::Permissions::from_mode(0o500)).unwrap();
        let result = load_or_generate_user_signing_key(&secrets_dir);
        assert!(result.is_err());
    }

    // ── TLS: initial key+cert generation (the (None, _) branch) ────────────

    #[test]
    fn test_tls_materials_initial_generation() {
        let dir = TempDir::new().unwrap();
        // No pre-existing key or cert.
        let m = load_or_generate_tls_materials(dir.path(), "localhost", 365).unwrap();
        assert!(!m.cert_der.is_empty(), "cert_der should be non-empty");
        assert!(!m.key_der.is_empty(), "key_der should be non-empty");
        // Both files should have been written.
        assert!(dir.path().join("tls-key").exists(), "tls-key file should exist");
        assert!(dir.path().join("tls-cert").exists(), "tls-cert file should exist");
    }

    #[test]
    #[cfg(unix)]
    fn test_tls_materials_readonly_no_key_fails() {
        use std::os::unix::fs::PermissionsExt;
        let parent = TempDir::new().unwrap();
        let secrets_dir = parent.path().join("secrets");
        std::fs::create_dir(&secrets_dir).unwrap();
        std::fs::set_permissions(&secrets_dir, std::fs::Permissions::from_mode(0o500)).unwrap();
        let result = load_or_generate_tls_materials(&secrets_dir, "localhost", 365);
        assert!(result.is_err(), "expected error for unwritable dir");
    }

    // ── cert_renewal_needed: mtime-based logic ───────────────────────────────

    #[test]
    #[cfg(unix)]
    fn test_cert_renewal_not_needed_for_fresh_cert() {
        let dir = TempDir::new().unwrap();
        write_secret(dir.path(), "tls-cert", b"fake-cert-data").unwrap();
        // A freshly created file should not need renewal under a 365-day window.
        assert!(!cert_renewal_needed(dir.path(), "tls-cert", 365));
    }

    #[test]
    #[cfg(unix)]
    fn test_cert_renewal_needed_for_old_cert() {
        let dir = TempDir::new().unwrap();
        write_secret(dir.path(), "tls-cert", b"fake-cert-data").unwrap();
        let cert_path = dir.path().join("tls-cert");
        // Backdate the cert file mtime by 365 days.
        backdate_mtime(&cert_path, 365);
        // Now renewal should be triggered (elapsed >= 364 days = threshold for 365d validity).
        assert!(cert_renewal_needed(dir.path(), "tls-cert", 365));
    }

    #[test]
    #[cfg(unix)]
    fn test_cert_renewal_triggers_regen_with_same_key() {
        use sha2::{Digest, Sha256};
        let dir = TempDir::new().unwrap();
        let m1 = load_or_generate_tls_materials(dir.path(), "localhost", 365).unwrap();
        // Backdate the cert file to trigger mtime-based renewal.
        backdate_mtime(&dir.path().join("tls-cert"), 365);
        let m2 = load_or_generate_tls_materials(dir.path(), "localhost", 365).unwrap();
        // Key bytes must be unchanged (stable key = stable cert hash lineage).
        assert_eq!(*m1.key_der, *m2.key_der, "key must be reused on cert renewal");
        // Cert is a different DER (different timestamps).
        let h1 = Sha256::digest(&m1.cert_der);
        let h2 = Sha256::digest(&m2.cert_der);
        assert_ne!(h1, h2, "renewed cert should have different DER due to new timestamps");
    }

    // ── Credential store key: readonly + no store ────────────────────────────

    #[test]
    #[cfg(unix)]
    fn test_credential_store_key_readonly_no_store_fails() {
        use std::os::unix::fs::PermissionsExt;
        let parent = TempDir::new().unwrap();
        let secrets_dir = parent.path().join("secrets");
        std::fs::create_dir(&secrets_dir).unwrap();
        std::fs::set_permissions(&secrets_dir, std::fs::Permissions::from_mode(0o500)).unwrap();
        let store_path = secrets_dir.join("users.toml.age");
        let result = load_or_generate_credential_store_key(&secrets_dir, &store_path);
        assert!(result.is_err());
    }
}
