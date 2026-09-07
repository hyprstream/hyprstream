//! Persistent file-based credential storage.
//!
//! This module provides atomic read/write of secret key material to a
//! configurable directory (`secrets.path` in config, or the systemd
//! credentials directory when running under systemd).
//!
//! # Design
//!
//! - **Centralized directory resolution** — callers use `credentials_dir()` so
//!   all secret-bearing paths fail closed consistently.
//! - **Atomic writes** — secrets are written via tempfile + rename so partial
//!   writes are never visible.
//! - **Mode 0600 / 0700** — secret files and their parent directory are created
//!   with restrictive permissions on Unix.
//! - **Read-only detection** — when the secrets directory is not writable (e.g.
//!   the systemd credentials ramfs), missing secrets are a hard error rather than
//!   triggering key generation.

use anyhow::{anyhow, ensure, Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{SigningKey, VerifyingKey};
use zeroize::{Zeroize, Zeroizing};

use crate::server::tls::TlsMaterials;

// ─── Low-level primitives ───────────────────────────────────────────────────

/// Resolve the credentials/secrets directory via the unified config resolver.
///
/// Precedence is `HYPRSTREAM__SECRETS__PATH`, config `[secrets].path`, XDG
/// `<config_dir>/credentials`, then error. Fails closed — NEVER falls back to a
/// world-writable `/tmp` path or implicit `/etc` path for a secret-bearing dir.
pub fn credentials_dir() -> anyhow::Result<std::path::PathBuf> {
    crate::config::HyprConfig::resolve_secrets_dir()
}

/// Resolve credentials from an already-loaded config handle.
///
/// This is the same resolver family as `credentials_dir()`, but lets CLI paths
/// that loaded config from `--config` avoid reloading default locations.
pub fn credentials_dir_for_config(
    config: Option<&crate::config::HyprConfig>,
) -> anyhow::Result<std::path::PathBuf> {
    crate::config::HyprConfig::resolve_secrets_dir_for(config)
}

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

/// Write a secret only if the file does not already exist (atomic O_EXCL).
///
/// Returns `Ok(true)` if the file was created, `Ok(false)` if it already
/// existed (another process won the race). This eliminates the TOCTOU race
/// between checking if a key exists and generating a new one.
pub fn write_secret_exclusive(dir: &std::path::Path, name: &str, value: &[u8]) -> Result<bool> {
    use std::io::Write as _;
    ensure_secrets_dir(dir)?;
    let path = dir.join(name);

    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true) // O_EXCL — atomic check-and-create
        .open(&path)
    {
        Ok(mut f) => {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))
                    .with_context(|| format!("failed to chmod '{}'", path.display()))?;
            }
            f.write_all(value)
                .with_context(|| format!("failed to write secret to '{}'", path.display()))?;
            tracing::debug!("exclusively wrote secret '{}'", path.display());
            Ok(true)
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => Ok(false),
        Err(e) => Err(e).with_context(|| format!("failed to create '{}'", path.display())),
    }
}

/// Write a named **public** trust artifact to `dir` atomically (tempfile +
/// rename).
///
/// Same atomic-write contract as [`write_secret`], but for non-secret material
/// (public keys, attestations): the resulting file has mode 0644 so other
/// local readers (e.g. a credential-mint unit mounting the directory) can
/// consume it. NEVER pass secret material here.
pub fn write_public(dir: &std::path::Path, name: &str, value: &[u8]) -> Result<()> {
    use std::io::Write as _;
    ensure_secrets_dir(dir)?;
    let path = dir.join(name);

    let mut tmp = tempfile::NamedTempFile::new_in(dir)
        .with_context(|| format!("failed to create temp file in '{}'", dir.display()))?;

    tmp.write_all(value)
        .with_context(|| format!("failed to write public file in '{}'", dir.display()))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(tmp.path(), std::fs::Permissions::from_mode(0o644))
            .with_context(|| format!("failed to chmod public file in '{}'", dir.display()))?;
    }

    tmp.persist(&path)
        .with_context(|| format!("failed to persist public file to '{}'", path.display()))?;

    tracing::debug!("wrote public file '{}'", path.display());
    Ok(())
}

/// Write a public artifact only when it is missing or its content differs.
///
/// Keeps re-provisioning and service restarts idempotent: an up-to-date
/// sidecar is left untouched (no mtime churn, no rewrite).
fn write_public_if_changed(dir: &std::path::Path, name: &str, value: &[u8]) -> Result<()> {
    if let Some(existing) = read_secret(dir, name)? {
        if existing == value {
            return Ok(());
        }
    }
    write_public(dir, name, value)
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

// ─── Per-service key management ──────────────────────────────────────────────

/// Per-service credential directory layout:
///
/// ```text
/// credentials/
///   ca-key            # CA private key (policy service only)
///   ca-pubkey         # CA verifying key (public, all services)
///   ca-mldsa-pubkey   # CA derived ML-DSA-65 verifying key (public, all services)
///   {service}/
///     signing-key     # service's own Ed25519 private key
///     signing-key.pub # service's Ed25519 verifying key (public sidecar, 0644)
///     service-pubkey.hybrid  # hybrid bootstrap entry (public sidecar, 0644)
///     service-jwt     # CA-signed JWT certificate
///   bootstrap-pubkeys # JSON: { "policy": "base64...", "discovery": "base64..." }
/// ```
///
/// Load or generate an independent Ed25519 signing key for a specific service.
///
/// Each service gets its own randomly-generated keypair stored in
/// `credentials/{service_name}/signing-key`. This is NOT derived from the root
/// key — it's an independent key that the CA (PolicyService) certifies via a
/// service JWT.
///
/// # Arguments
///
/// * `credentials_dir` — Base credentials directory (e.g., `~/.config/hyprstream/credentials`)
/// * `service_name` — Service name (e.g., "model", "discovery")
///
/// Validate a service name for use in filesystem paths.
///
/// Rejects names containing path separators, `..`, or characters
/// outside `[a-z0-9-]`. Prevents directory traversal when service
/// names are used in path construction.
pub fn validate_service_name(name: &str) -> Result<()> {
    if name.is_empty() {
        anyhow::bail!("service name cannot be empty");
    }
    if name.len() > 64 {
        anyhow::bail!("service name too long (max 64 chars): '{name}'");
    }
    if !name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-') {
        anyhow::bail!(
            "service name '{name}' contains invalid characters; \
             only lowercase ASCII, digits, and hyphens are allowed"
        );
    }
    if name.starts_with('-') {
        anyhow::bail!("service name cannot start with a hyphen: '{name}'");
    }
    Ok(())
}

/// File name of the public Ed25519 sidecar written alongside a service's
/// `signing-key` seed: the raw 32-byte verifying key.
pub const SIGNING_KEY_PUB_NAME: &str = "signing-key.pub";

/// File name of the public hybrid (Ed25519 ‖ ML-DSA-65) sidecar written
/// alongside a service's `signing-key` seed.
///
/// Contents are exactly the service's 1984-byte `bootstrap-pubkeys` entry
/// value (see `docs/bootstrap-pubkeys-format.md`), so a bootstrap-enrollment
/// mint can consume it without ever touching secret material — the ML-DSA-65
/// half is derived from the Ed25519 seed and cannot be recomputed from a
/// public key alone.
pub const SERVICE_PUBKEY_HYBRID_NAME: &str = "service-pubkey.hybrid";

/// The directory holding `service_name`'s signing key under `secrets_dir` for
/// the given profile.
///
/// Mirrors [`resolve_service_signing_key`]'s file layout: "policy" and any
/// per-service-scoped directory use the flat layout, everything else gets a
/// `{service_name}/` subdirectory.
pub fn service_signing_key_dir(
    secrets_dir: &std::path::Path,
    service_name: &str,
    profile: SecretsProfile,
) -> std::path::PathBuf {
    match (profile, service_name) {
        (_, "policy") | (SecretsProfile::PerServiceScoped, _) => secrets_dir.to_path_buf(),
        (SecretsProfile::SharedDirectory, _) => secrets_dir.join(service_name),
    }
}

/// Write the public sidecars for a service signing key next to its seed:
///
/// - [`SIGNING_KEY_PUB_NAME`] — the 32-byte Ed25519 verifying key.
/// - [`SERVICE_PUBKEY_HYBRID_NAME`] — the 1984-byte hybrid `bootstrap-pubkeys`
///   entry (Ed25519 ‖ derived ML-DSA-65) for bootstrap enrollment.
///
/// Both are public trust material (mode 0644) derived from the key's public
/// half only; the seed never appears in either. Idempotent: a sidecar whose
/// content already matches is left untouched.
pub fn ensure_service_key_sidecars(
    service_dir: &std::path::Path,
    service_key: &SigningKey,
) -> Result<()> {
    write_public_if_changed(
        service_dir,
        SIGNING_KEY_PUB_NAME,
        service_key.verifying_key().as_bytes(),
    )?;
    let hybrid = BootstrapPubkey::for_service_key(service_key)?.to_key_bytes();
    write_public_if_changed(service_dir, SERVICE_PUBKEY_HYBRID_NAME, &hybrid)?;
    Ok(())
}

/// Best-effort sidecar write from the key loader.
///
/// Never fails key loading over a public sidecar (e.g. on a read-only
/// credentials mount): the sidecar is backfilled on every writable load and
/// by `hyprstream service ensure-key`.
fn backfill_service_key_sidecars(
    service_dir: &std::path::Path,
    service_name: &str,
    service_key: &SigningKey,
) {
    if let Err(e) = ensure_service_key_sidecars(service_dir, service_key) {
        tracing::warn!(
            "could not write public key sidecars for service '{service_name}': {e:#}"
        );
    }
}

pub fn load_or_generate_service_signing_key(
    credentials_dir: &std::path::Path,
    service_name: &str,
) -> Result<SigningKey> {
    validate_service_name(service_name)?;
    let service_dir = credentials_dir.join(service_name);
    const NAME: &str = "signing-key";

    if let Some(mut bytes) = read_secret(&service_dir, NAME)? {
        let mut arr: [u8; 32] = bytes.as_slice()
            .try_into()
            .map_err(|_| anyhow!("service '{service_name}' signing-key must be 32 bytes (Ed25519 seed)"))?;
        let sk = SigningKey::from_bytes(&arr);
        bytes.zeroize();
        arr.zeroize();
        // Backfill the public sidecars for pre-existing seeds (written on
        // generate below, but a seed written by an older version has none).
        backfill_service_key_sidecars(&service_dir, service_name, &sk);
        tracing::info!("Loaded Ed25519 signing key for service '{service_name}'");
        return Ok(sk);
    }

    if !is_writable(&service_dir) && !is_writable(credentials_dir) {
        return Err(missing_in_readonly(&service_dir, NAME));
    }

    let key = SigningKey::generate(&mut rand::rngs::OsRng);
    let mut raw = key.to_bytes();
    match write_secret_exclusive(&service_dir, NAME, &raw) {
        Ok(true) => {
            raw.zeroize();
            backfill_service_key_sidecars(&service_dir, service_name, &key);
            tracing::info!("Generated new Ed25519 signing key for service '{service_name}'");
            Ok(key)
        }
        Ok(false) => {
            // Another process won the race — reload their key
            raw.zeroize();
            tracing::info!(
                "Key for '{service_name}' created by another process — reloading"
            );
            // Re-read (not recursive — file now exists)
            let mut bytes = read_secret(&service_dir, NAME)?
                .ok_or_else(|| anyhow!("race loser: key file disappeared"))?;
            let mut arr: [u8; 32] = bytes.as_slice()
                .try_into()
                .map_err(|_| anyhow!("service '{service_name}' signing-key must be 32 bytes"))?;
            let sk = SigningKey::from_bytes(&arr);
            bytes.zeroize();
            arr.zeroize();
            backfill_service_key_sidecars(&service_dir, service_name, &sk);
            Ok(sk)
        }
        Err(e) => {
            raw.zeroize();
            Err(e)
        }
    }
}

/// Load the CA signing key (PolicyService only).
///
/// The CA key is the PolicyService's own signing key — it signs service JWTs
/// that bind service names to their Ed25519 pubkeys.
pub fn load_ca_signing_key(credentials_dir: &std::path::Path) -> Result<SigningKey> {
    const NAME: &str = "ca-key";
    if let Some(mut bytes) = read_secret(credentials_dir, NAME)? {
        let mut arr: [u8; 32] = bytes.as_slice()
            .try_into()
            .map_err(|_| anyhow!("ca-key must be 32 bytes (Ed25519 seed)"))?;
        let sk = SigningKey::from_bytes(&arr);
        bytes.zeroize();
        arr.zeroize();
        tracing::info!("Loaded CA signing key");
        return Ok(sk);
    }
    Err(missing_in_readonly(credentials_dir, NAME))
}

/// Write the CA signing key to the credentials directory.
///
/// Used during wizard bootstrap to persist the generated CA key.
pub fn write_ca_signing_key(credentials_dir: &std::path::Path, key: &SigningKey) -> Result<()> {
    write_secret(credentials_dir, "ca-key", &key.to_bytes())
}

/// Load the CA verifying key (public, distributed to all services).
///
/// This is the trust anchor for verifying service JWTs.
pub fn load_ca_verifying_key(credentials_dir: &std::path::Path) -> Result<VerifyingKey> {
    const NAME: &str = "ca-pubkey";
    if let Some(bytes) = read_secret(credentials_dir, NAME)? {
        let arr: [u8; 32] = bytes.as_slice()
            .try_into()
            .map_err(|_| anyhow!("ca-pubkey must be 32 bytes (Ed25519 pubkey)"))?;
        VerifyingKey::from_bytes(&arr)
            .map_err(|e| anyhow!("invalid ca-pubkey: {e}"))
    } else {
        Err(missing_in_readonly(credentials_dir, NAME))
    }
}

/// Write the CA verifying key to the credentials directory.
pub fn write_ca_verifying_key(credentials_dir: &std::path::Path, key: &VerifyingKey) -> Result<()> {
    write_secret(credentials_dir, "ca-pubkey", key.as_bytes())
}

/// Load the CA's derived ML-DSA-65 verifying key (public, distributed to all
/// services).
///
/// This is the post-quantum half of the CA JWT composite pair: the CA signs
/// hybrid service WITs with `(derive_mesh_mldsa_key(ca_jwt_key), ca_jwt_key)`,
/// and verifiers resolve the composite kid to this key plus `ca-pubkey`.
pub fn load_ca_ml_dsa_verifying_key(
    credentials_dir: &std::path::Path,
) -> Result<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey> {
    const NAME: &str = "ca-mldsa-pubkey";
    if let Some(bytes) = read_secret(credentials_dir, NAME)? {
        hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&bytes)
            .map_err(|e| anyhow!("invalid ca-mldsa-pubkey: {e}"))
    } else {
        Err(missing_in_readonly(credentials_dir, NAME))
    }
}

/// Write the CA's derived ML-DSA-65 verifying key to the credentials directory.
pub fn write_ca_ml_dsa_verifying_key(
    credentials_dir: &std::path::Path,
    key: &hyprstream_rpc::crypto::pq::MlDsaVerifyingKey,
) -> Result<()> {
    write_secret(
        credentials_dir,
        "ca-mldsa-pubkey",
        &hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(key),
    )
}

fn provisioned_service_jwt_dir(
    credentials_dir: &std::path::Path,
    service_name: &str,
    profile: SecretsProfile,
) -> std::path::PathBuf {
    match profile {
        SecretsProfile::SharedDirectory => credentials_dir.join(service_name),
        SecretsProfile::PerServiceScoped => credentials_dir.to_path_buf(),
    }
}

fn writable_service_jwt_dir(
    credentials_dir: &std::path::Path,
    service_name: &str,
    profile: SecretsProfile,
) -> std::path::PathBuf {
    let state_dir = super::key_rotation::rotation_state_dir(credentials_dir);
    if profile == SecretsProfile::PerServiceScoped && state_dir == credentials_dir {
        state_dir
    } else {
        // `$STATE_DIRECTORY` is shared by the generated service units, so the
        // writable fallback must retain a service-name component even when the
        // read-only systemd credential directory was already service-scoped.
        state_dir.join(service_name)
    }
}

/// Load a service JWT using the active credential layout.
///
/// Renewed state is preferred. When a read-only credential provider caused
/// writes to fall back to `$STATE_DIRECTORY`/XDG state, the originally
/// provisioned credential remains the first-boot fallback.
pub fn load_service_jwt_for_profile(
    credentials_dir: &std::path::Path,
    service_name: &str,
    profile: SecretsProfile,
) -> Result<Option<String>> {
    validate_service_name(service_name)?;
    let state_dir = writable_service_jwt_dir(credentials_dir, service_name, profile);
    let provisioned_dir = provisioned_service_jwt_dir(credentials_dir, service_name, profile);
    let bytes = match read_secret(&state_dir, "service-jwt")? {
        Some(bytes) => Some(bytes),
        None if state_dir != provisioned_dir => read_secret(&provisioned_dir, "service-jwt")?,
        None => None,
    };
    match bytes {
        Some(bytes) => {
            let jwt = String::from_utf8(bytes)
                .context("service-jwt is not valid UTF-8")?;
            Ok(Some(jwt))
        }
        None => Ok(None),
    }
}

/// Persist a service JWT using the same profile-aware path startup reads.
pub fn write_service_jwt_for_profile(
    credentials_dir: &std::path::Path,
    service_name: &str,
    profile: SecretsProfile,
    jwt: &str,
) -> Result<()> {
    validate_service_name(service_name)?;
    let state_dir = writable_service_jwt_dir(credentials_dir, service_name, profile);
    write_secret(&state_dir, "service-jwt", jwt.as_bytes())
}

/// Load a service JWT from the shared-directory credential layout.
pub fn load_service_jwt(
    credentials_dir: &std::path::Path,
    service_name: &str,
) -> Result<Option<String>> {
    load_service_jwt_for_profile(
        credentials_dir,
        service_name,
        SecretsProfile::SharedDirectory,
    )
}

/// Write a service JWT to the shared-directory credential layout.
pub fn write_service_jwt(
    credentials_dir: &std::path::Path,
    service_name: &str,
    jwt: &str,
) -> Result<()> {
    write_service_jwt_for_profile(
        credentials_dir,
        service_name,
        SecretsProfile::SharedDirectory,
        jwt,
    )
}

/// Bootstrap pubkeys — the pubkeys of services needed before discovery is available.
///
/// Contains the pubkeys of PolicyService and DiscoveryService, which must be
/// known to all services so they can verify RPC responses from these bootstrap
/// services without querying discovery (chicken-and-egg).
///
/// Wire format (see `docs/bootstrap-pubkeys-format.md`): a flat JSON object
/// `{ "<service>": "<base64>" }`, base64 using the URL-safe-no-pad alphabet
/// (RFC 4648 §5, no `+`/`/`/`=`). Each value decodes either to 32 raw Ed25519
/// bytes (classical entry) or to 1984 bytes — 32 Ed25519 followed by 1952
/// ML-DSA-65 — for a hybrid entry, the same concatenation the deployment CA
/// root uses.
///
/// This projection returns only the Ed25519 anchor of each entry, so callers
/// that predate hybrid entries keep working unchanged. Callers that need the
/// bound post-quantum key use [`load_bootstrap_pubkeys_hybrid`].
pub fn load_bootstrap_pubkeys(
    credentials_dir: &std::path::Path,
) -> Result<std::collections::HashMap<String, VerifyingKey>> {
    Ok(load_bootstrap_pubkeys_hybrid(credentials_dir)?
        .into_iter()
        .map(|(name, entry)| (name, entry.ed25519))
        .collect())
}

/// Write bootstrap pubkeys to the credentials directory.
///
/// Every entry is written in the classical 32-byte form. Use
/// [`write_bootstrap_pubkeys_hybrid`] to persist entries that carry a bound
/// ML-DSA-65 key.
pub fn write_bootstrap_pubkeys(
    credentials_dir: &std::path::Path,
    pubkeys: &std::collections::HashMap<String, VerifyingKey>,
) -> Result<()> {
    let hybrid: std::collections::HashMap<String, BootstrapPubkey> = pubkeys
        .iter()
        .map(|(name, vk)| (name.clone(), BootstrapPubkey::classical(*vk)))
        .collect();
    write_bootstrap_pubkeys_hybrid(credentials_dir, &hybrid)
}

// ─── Hybrid bootstrap entries ────────────────────────────────────────────────

/// Length of a raw Ed25519 verifying key.
const BOOTSTRAP_ED25519_BYTES: usize = 32;
/// Length of a raw ML-DSA-65 verifying key.
const BOOTSTRAP_ML_DSA_65_BYTES: usize = 1952;
/// Length of the concatenated hybrid form (Ed25519 ‖ ML-DSA-65).
const BOOTSTRAP_HYBRID_BYTES: usize = BOOTSTRAP_ED25519_BYTES + BOOTSTRAP_ML_DSA_65_BYTES;

/// The on-disk file name of the bootstrap pubkeys seed.
const BOOTSTRAP_PUBKEYS_NAME: &str = "bootstrap-pubkeys";

/// One `bootstrap-pubkeys` entry: the Ed25519 anchor plus an optional bound
/// ML-DSA-65 verifying key.
///
/// The Ed25519 key is always the identity of the entry — the post-quantum key
/// is bound *to* it, mirroring the user identity and deployment-CA layouts.
/// Verification is per-identity: an entry that carries no PQ key verifies
/// classically, an entry that carries one requires both signatures.
#[derive(Clone, Debug)]
pub struct BootstrapPubkey {
    pub ed25519: VerifyingKey,
    pub ml_dsa_65: Option<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey>,
}

impl BootstrapPubkey {
    /// A classical (Ed25519-only) entry.
    pub fn classical(ed25519: VerifyingKey) -> Self {
        Self { ed25519, ml_dsa_65: None }
    }

    /// A hybrid entry binding an ML-DSA-65 key to an Ed25519 anchor.
    pub fn hybrid(
        ed25519: VerifyingKey,
        ml_dsa_65: hyprstream_rpc::crypto::pq::MlDsaVerifyingKey,
    ) -> Self {
        Self { ed25519, ml_dsa_65: Some(ml_dsa_65) }
    }

    /// The hybrid entry for a service that signs with `service_key`.
    ///
    /// The ML-DSA-65 half is DERIVED from the service's Ed25519 key with
    /// [`hyprstream_rpc::node_identity::derive_mesh_mldsa_key`] rather than
    /// generated independently. That is not merely convenient — it is required
    /// for the entry to be usable: every signer in the tree (the local signer,
    /// the service dispatch default, the published `#mesh-pq` verification
    /// method) produces its post-quantum signature from exactly that
    /// derivation, so an independently generated key would anchor a public key
    /// nothing ever signs with. It also keeps the service's secret material a
    /// single Ed25519 seed — nothing new to persist, protect, back up or
    /// rotate.
    pub fn for_service_key(service_key: &SigningKey) -> Result<Self> {
        let pq_sk = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(service_key);
        let pq_vk_bytes = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq_sk);
        let pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&pq_vk_bytes)
            .context("derived mesh ML-DSA-65 verifying key is malformed")?;
        Ok(Self::hybrid(service_key.verifying_key(), pq_vk))
    }

    /// Whether this entry carries a bound post-quantum key.
    pub fn is_hybrid(&self) -> bool {
        self.ml_dsa_65.is_some()
    }

    /// The canonical byte encoding: the 32 raw Ed25519 bytes, followed by the
    /// 1952 raw ML-DSA-65 bytes when the entry is hybrid.
    pub fn to_key_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(if self.is_hybrid() {
            BOOTSTRAP_HYBRID_BYTES
        } else {
            BOOTSTRAP_ED25519_BYTES
        });
        out.extend_from_slice(self.ed25519.as_bytes());
        if let Some(pq) = &self.ml_dsa_65 {
            out.extend_from_slice(&hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(pq));
        }
        out
    }

    /// Decode the canonical byte encoding, discriminating on length.
    ///
    /// `service` names the entry only so the error text is actionable.
    pub fn from_key_bytes(service: &str, bytes: &[u8]) -> Result<Self> {
        let (ed_bytes, pq_bytes) = match bytes.len() {
            BOOTSTRAP_ED25519_BYTES => (bytes, None),
            BOOTSTRAP_HYBRID_BYTES => {
                let (ed, pq) = bytes.split_at(BOOTSTRAP_ED25519_BYTES);
                (ed, Some(pq))
            }
            other => {
                return Err(anyhow!(
                    "bootstrap-pubkey for '{service}' must decode to either \
                     {BOOTSTRAP_ED25519_BYTES} bytes (raw Ed25519 verifying key) or \
                     {BOOTSTRAP_HYBRID_BYTES} bytes (32-byte Ed25519 followed by \
                     {BOOTSTRAP_ML_DSA_65_BYTES}-byte ML-DSA-65 verifying key); \
                     got {other} bytes"
                ))
            }
        };

        let arr: [u8; BOOTSTRAP_ED25519_BYTES] = ed_bytes
            .try_into()
            .map_err(|_| anyhow!("bootstrap-pubkey for '{service}' has a malformed Ed25519 part"))?;
        let ed25519 = VerifyingKey::from_bytes(&arr)
            .map_err(|e| anyhow!("invalid bootstrap-pubkey for '{service}': {e}"))?;

        let ml_dsa_65 = match pq_bytes {
            Some(pq) => Some(
                hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(pq).map_err(|e| {
                    anyhow!("bootstrap-pubkey for '{service}' has a malformed ML-DSA-65 part: {e}")
                })?,
            ),
            None => None,
        };

        Ok(Self { ed25519, ml_dsa_65 })
    }

    /// Verify a signature over `message` under this entry's per-identity policy.
    ///
    /// - Classical entry: the Ed25519 signature must verify. A post-quantum
    ///   signature cannot be checked against an entry with no bound PQ key, so
    ///   supplying one is an error rather than a silently ignored input.
    /// - Hybrid entry: **both** the Ed25519 and the ML-DSA-65 signature must be
    ///   present and verify.
    ///
    /// Post-quantum verification is never demanded of an entry that does not
    /// carry a post-quantum key — a classical entry keeps its classical floor.
    pub fn verify(
        &self,
        message: &[u8],
        ed25519_sig: &ed25519_dalek::Signature,
        ml_dsa_65_sig: Option<&[u8]>,
    ) -> Result<()> {
        use ed25519_dalek::Verifier;

        self.ed25519
            .verify(message, ed25519_sig)
            .map_err(|e| anyhow!("Ed25519 signature verification failed: {e}"))?;

        match (&self.ml_dsa_65, ml_dsa_65_sig) {
            (Some(pq_key), Some(sig)) => hyprstream_rpc::crypto::pq::ml_dsa_verify(pq_key, message, sig),
            (Some(_), None) => Err(anyhow!(
                "this bootstrap key is hybrid (Ed25519 + ML-DSA-65) but no ML-DSA-65 \
                 signature was supplied; both signatures are required"
            )),
            (None, Some(_)) => Err(anyhow!(
                "an ML-DSA-65 signature was supplied but this bootstrap key is \
                 Ed25519-only, so the signature cannot be verified; re-provision the \
                 entry in its 1984-byte hybrid form to enable post-quantum verification"
            )),
            (None, None) => Ok(()),
        }
    }
}

/// Load bootstrap pubkeys, preserving any bound post-quantum key material.
///
/// Accepts both the classical (32-byte value) and the hybrid (1984-byte value)
/// form in the same file; the two are distinguished by decoded length. Files
/// written before hybrid entries existed load unchanged.
pub fn load_bootstrap_pubkeys_hybrid(
    credentials_dir: &std::path::Path,
) -> Result<std::collections::HashMap<String, BootstrapPubkey>> {
    let Some(bytes) = read_secret(credentials_dir, BOOTSTRAP_PUBKEYS_NAME)? else {
        return Ok(std::collections::HashMap::new());
    };

    let json: std::collections::HashMap<String, String> = serde_json::from_slice(&bytes).context(
        "bootstrap-pubkeys is not valid JSON: expected a flat object \
         `{ \"<service>\": \"<base64>\" }` (see docs/bootstrap-pubkeys-format.md)",
    )?;

    let mut map = std::collections::HashMap::new();
    let mut classical_only = Vec::new();
    for (name, b64) in json {
        let key_bytes: Vec<u8> = URL_SAFE_NO_PAD.decode(&b64).with_context(|| {
            format!(
                "invalid base64 in bootstrap-pubkeys for '{name}': expected \
                 URL-safe-no-pad alphabet (RFC 4648 §5, no '+'/'/'/'='), got {b64:?}"
            )
        })?;
        let entry = BootstrapPubkey::from_key_bytes(&name, &key_bytes)?;
        if !entry.is_hybrid() {
            classical_only.push(name.clone());
        }
        map.insert(name, entry);
    }

    if !classical_only.is_empty() {
        classical_only.sort();
        tracing::debug!(
            services = %classical_only.join(", "),
            "bootstrap-pubkeys entries are Ed25519-only; they verify classically \
             until re-provisioned with a bound ML-DSA-65 key"
        );
    }

    Ok(map)
}

/// Reject a `bootstrap-pubkeys` map that still carries Ed25519-only service
/// entries.
///
/// Every service this node provisions gets a bound ML-DSA-65 key: there is no
/// classical-only install path, so a classical entry here is stale material
/// from a pre-hybrid provisioning run, not a supported configuration. Left
/// alone it does not degrade gracefully — the service is simply never anchored
/// in the post-quantum trust store, and its RPC is later refused with an
/// opaque "no anchored ML-DSA-65 signer key" at verification time. Failing
/// here converts that into an actionable provisioning error.
///
/// Deliberately NOT enforced inside [`load_bootstrap_pubkeys_hybrid`]: the
/// low-level loader must stay able to read a legacy file so tooling — and this
/// error itself — can report precisely which entries are stale.
pub fn ensure_bootstrap_pubkeys_hybrid(
    entries: &std::collections::HashMap<String, BootstrapPubkey>,
) -> Result<()> {
    let mut classical: Vec<&str> = entries
        .iter()
        .filter(|(_, entry)| !entry.is_hybrid())
        .map(|(name, _)| name.as_str())
        .collect();
    if classical.is_empty() {
        return Ok(());
    }
    classical.sort_unstable();
    Err(anyhow!(
        "bootstrap-pubkeys has Ed25519-only entries for service(s): {}. Service \
         identities must be hybrid (Ed25519 + ML-DSA-65); these were written by a \
         pre-hybrid provisioning run and cannot be anchored for post-quantum \
         verification. Re-provision this node by running 'hyprstream wizard' — \
         it re-provisions in place, preserving existing keys while binding the \
         ML-DSA-65 half for every service — to rewrite {BOOTSTRAP_PUBKEYS_NAME}.",
        classical.join(", ")
    ))
}

/// Write bootstrap pubkeys, encoding hybrid entries in their concatenated form.
///
/// Classical entries are written exactly as the pre-hybrid writer wrote them, so
/// a file round-trips byte-for-byte when no entry carries a PQ key.
pub fn write_bootstrap_pubkeys_hybrid(
    credentials_dir: &std::path::Path,
    pubkeys: &std::collections::HashMap<String, BootstrapPubkey>,
) -> Result<()> {
    let json: std::collections::HashMap<String, String> = pubkeys
        .iter()
        .map(|(name, entry)| (name.clone(), URL_SAFE_NO_PAD.encode(entry.to_key_bytes())))
        .collect();
    let data = serde_json::to_vec(&json).context("failed to serialize bootstrap-pubkeys")?;
    write_secret(credentials_dir, BOOTSTRAP_PUBKEYS_NAME, &data)
}

/// Directory sibling to `bootstrap-pubkeys` holding the per-service chain-signed
/// enrollment attestations (hyprstream#1562 H3): `{service}.json` per
/// allowlisted service.
pub const BOOTSTRAP_PUBKEYS_ENROLLMENT_DIR: &str = "bootstrap-pubkeys.enrollment";

/// Fail-closed enrollment check for OS-owned deployments (hyprstream#1562 H3).
///
/// Every bootstrap entry for a service in the fixed enrollment allowlist
/// (`hyprstream_discovery::SERVICE_KEY_ENROLLMENT_ALLOWED_SERVICES`) must be
/// backed by an attestation in
/// `{credentials_dir}/bootstrap-pubkeys.enrollment/{service}.json` that
/// verifies against the node's OS-owned deployment trust chain and names
/// exactly this entry's hybrid key — the unsigned-TOFU posture is refused.
/// Missing, malformed, expired, or mismatched attestations are fatal. Entries
/// outside the allowlist cannot be enrolled by design and keep their existing
/// local posture. Wizard/dev (non-OsOwnedFiles) deployments never call this.
pub fn ensure_bootstrap_pubkeys_enrolled(
    credentials_dir: &std::path::Path,
    entries: &std::collections::HashMap<String, BootstrapPubkey>,
) -> Result<()> {
    let mut names: Vec<&str> = entries.keys().map(String::as_str).collect();
    names.sort_unstable();
    for name in names {
        if !hyprstream_discovery::SERVICE_KEY_ENROLLMENT_ALLOWED_SERVICES.contains(&name) {
            continue;
        }
        let path = credentials_dir
            .join(BOOTSTRAP_PUBKEYS_ENROLLMENT_DIR)
            .join(format!("{name}.json"));
        let bytes = std::fs::read(&path).map_err(|error| {
            anyhow!(
                "OS-owned deployment requires a chain-signed enrollment attestation for \
                 service '{name}' at {}: {error} (re-run the service-key enrollment \
                 provisioning step)",
                path.display()
            )
        })?;
        let verified = hyprstream_discovery::verify_os_owned_service_key_enrollment(&bytes)
            .with_context(|| format!("enrollment attestation for service '{name}' rejected"))?;
        ensure!(
            verified.service == name,
            "enrollment attestation at {} is for service '{}', not '{name}'",
            path.display(),
            verified.service
        );
        let entry = &entries[name];
        ensure!(
            entry.is_hybrid() && verified.hybrid_public_key == entry.to_key_bytes(),
            "enrollment attestation for service '{name}' does not match its \
             bootstrap-pubkeys entry"
        );
    }
    Ok(())
}

// ─── Node-level key loaders ──────────────────────────────────────────────────

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

/// Describes how a service's credentials directory is scoped.
///
/// This is deliberately about directory layout rather than a particular
/// process manager. Both systemd `LoadCredential` directories and projected
/// Kubernetes Secret/CSI volumes can present a directory already scoped to a
/// single service, while standalone multi-process deployments share one
/// writable directory among services.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretsProfile {
    /// One base directory is shared by all services; non-policy credentials
    /// live under a `{service_name}/` subdirectory.
    SharedDirectory,
    /// The provider has already scoped the directory to one service, so its
    /// credentials use flat names such as `signing-key`.
    PerServiceScoped,
}

/// Explicit deployment signal for credential-directory layout.
///
/// The path override is deliberately separate: setting
/// `HYPRSTREAM__SECRETS__PATH` alone does not prove that a provider has already
/// scoped the directory to one service.
pub const SECRETS_PROFILE_ENV: &str = "HYPRSTREAM_SECRETS_PROFILE";

impl SecretsProfile {
    pub fn from_env() -> Result<Self> {
        match std::env::var(SECRETS_PROFILE_ENV) {
            Err(std::env::VarError::NotPresent) => Ok(Self::SharedDirectory),
            Ok(value) if value == "shared-directory" => Ok(Self::SharedDirectory),
            Ok(value) if value == "per-service-scoped" => Ok(Self::PerServiceScoped),
            Ok(value) => Err(anyhow!(
                "{SECRETS_PROFILE_ENV} must be 'shared-directory' or \
                 'per-service-scoped', got {value:?}"
            )),
            Err(error) => Err(anyhow!(
                "{SECRETS_PROFILE_ENV} is not valid Unicode: {error}"
            )),
        }
    }
}

/// Resolve the Ed25519 signing key a service process should use to sign its
/// own RPC responses in `--ipc` (multi-process) deployment.
///
/// # PolicyService is special
///
/// PolicyService's identity IS the root/CA key, not an independent
/// per-service key: `bootstrap_manager::do_bootstrap` registers
/// `bootstrap_pubkeys["policy"] = root_key.verifying_key()` (the same value
/// persisted flat at `secrets_dir/signing-key`), rather than generating a
/// `secrets_dir/policy/signing-key` like every other service gets.
///
/// Resolving "policy" via the generic per-service path
/// (`load_or_generate_service_signing_key`) reads/generates a *different*
/// file (`{secrets_dir}/policy/signing-key`) that bootstrap never populated —
/// so on first `--ipc` startup it silently mints a fresh, unrelated key. Every
/// peer that resolves "policy" from the trust store (seeded from
/// `bootstrap-pubkeys`) still expects `root_key.verifying_key()`, so
/// PolicyService's actual responses fail verification with "Response signed
/// by unexpected key" (#759) — a sibling gap to the CA-derivation fallback
/// #441 already removed from `resolve_service_key`.
///
/// - [`SecretsProfile::PerServiceScoped`]: the credential provider already
///   scopes `secrets_dir` to that one service, so the flat top-level file is
///   correct for every service, policy included.
/// - [`SecretsProfile::SharedDirectory`]: `secrets_dir` is shared across every
///   spawned service process, so non-policy services use a `{service_name}/`
///   subdirectory to avoid collisions — but policy must still resolve to the
///   flat root/CA key, matching what bootstrap registered.
pub fn resolve_service_signing_key(
    secrets_dir: &std::path::Path,
    service_name: &str,
    profile: SecretsProfile,
) -> Result<SigningKey> {
    match (profile, service_name) {
        (_, "policy") | (SecretsProfile::PerServiceScoped, _) => {
            load_or_generate_node_signing_key(secrets_dir)
        }
        (SecretsProfile::SharedDirectory, _) => {
            load_or_generate_service_signing_key(secrets_dir, service_name)
        }
    }
}

// The `#atproto` commit-signing key is NOT loaded here. It is the *active* key
// of the shared `Es256SigningKeyStore` (`auth::key_rotation`), the same P-256
// key `oauth::did_document` publishes as the `#atproto` verification method —
// one source of truth for signer and published key. The PDS writer
// (`services::discovery::PdsPublisher`) sources it from that store; there is no
// separate `atproto-signing-key` secret and no duplicate loader (#910a).

/// Decode the `exp` claim from a JWT without signature verification.
/// Returns `None` on any parse error (invalid base64, not JSON, missing exp).
pub fn decode_jwt_exp_raw(jwt: &str) -> Option<i64> {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;
    let payload_b64 = jwt.split('.').nth(1)?;
    let payload = URL_SAFE_NO_PAD.decode(payload_b64).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    value["exp"].as_i64()
}

/// Return the Unix mtime of `secrets_dir/signing-key`, or `now` if missing.
/// Used to populate the `nbf` field in JWKS entries (key was valid since written).
pub fn node_signing_key_mtime(secrets_dir: &std::path::Path) -> i64 {
    let path = secrets_dir.join("signing-key");
    path.metadata()
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64)
        .unwrap_or_else(|| chrono::Utc::now().timestamp())
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

/// Install `sk` as the client's user-signing-key (the file
/// `load_or_generate_user_signing_key` reads), backing up any existing key
/// first.
///
/// Used by `hyprstream user create --ssh`/`--key` (#439) to adopt an external
/// key as the *actual* signing key the CLI signs with — without this, an
/// imported-but-not-installed key leaves the client authenticating as
/// `anonymous`. The previous key, if any, is copied to `user-signing-key.bak`
/// (mode 0600) so the adopt is reversible; `None` is returned when no prior
/// key existed.
pub fn install_user_signing_key(
    secrets_dir: &std::path::Path,
    sk: &SigningKey,
) -> Result<Option<std::path::PathBuf>> {
    const NAME: &str = "user-signing-key";
    const BAK: &str = "user-signing-key.bak";

    if !is_writable(secrets_dir) {
        return Err(missing_in_readonly(secrets_dir, NAME));
    }

    let path = secrets_dir.join(NAME);
    let backup = if path.exists() {
        let bak = secrets_dir.join(BAK);
        // Back up via `write_secret` (atomic temp-file write with mode 0600 set
        // *before* persist) so the raw seed is never briefly world-readable, and
        // zeroize the in-memory copy before dropping it.
        let mut existing = std::fs::read(&path)
            .with_context(|| format!("failed to read existing {NAME} for backup"))?;
        let result = write_secret(secrets_dir, BAK, &existing);
        existing.zeroize();
        result
            .with_context(|| format!("failed to back up existing {NAME} to '{}'", bak.display()))?;
        tracing::info!("Backed up prior user signing key → '{}'", bak.display());
        Some(bak)
    } else {
        None
    };

    let mut raw = sk.to_bytes();
    let result = write_secret(secrets_dir, NAME, &raw);
    raw.zeroize();
    result?;
    tracing::info!("Installed adopted user signing key → '{}/{}'", secrets_dir.display(), NAME);
    Ok(backup)
}

// ─── Post-quantum hybrid user identity (#439) ────────────────────────────────

/// The on-disk name of the ML-DSA-65 sibling of `user-signing-key`.
const USER_PQ_NAME: &str = "user-signing-key.mldsa";
const USER_PQ_BAK: &str = "user-signing-key.mldsa.bak";

/// A user's client identity key material: the Ed25519 anchor plus an optional
/// bound ML-DSA-65 key (present under [`CryptoPolicy::Hybrid`], absent under
/// `Classical`). The fingerprint / kid is always the Ed25519 anchor's — the PQ
/// key is bound *to* it, never fingerprinted itself.
pub struct UserIdentityKeys {
    pub ed: SigningKey,
    pub mldsa: Option<hyprstream_rpc::crypto::pq::MlDsaSigningKey>,
}

impl UserIdentityKeys {
    /// The store algorithm tag this key material corresponds to.
    pub fn algorithm(&self) -> crate::auth::KeyAlgorithm {
        if self.mldsa.is_some() {
            crate::auth::KeyAlgorithm::HybridEd25519MlDsa65
        } else {
            crate::auth::KeyAlgorithm::Ed25519
        }
    }

    /// The bound ML-DSA-65 verifying key bytes (~1952B), if hybrid.
    pub fn pq_verifying_key_bytes(&self) -> Option<Vec<u8>> {
        self.mldsa
            .as_ref()
            .map(hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes)
    }
}

/// Load (or, under `Hybrid`, generate) the user's identity key material for the
/// given [`CryptoPolicy`].
///
/// - The Ed25519 anchor is loaded/generated exactly as before
///   ([`load_or_generate_user_signing_key`]) — the on-disk `user-signing-key`
///   file is unchanged, so existing loaders keep working.
/// - Under [`CryptoPolicy::Hybrid`] an ML-DSA-65 sibling
///   (`user-signing-key.mldsa`, a 32-byte FIPS 204 seed) is loaded if present,
///   or generated when absent. This covers the **in-place upgrade** case: a
///   legacy secrets dir that has only the Ed25519 key gains a bound PQ key for
///   the *same anchor* (same fingerprint) — no re-enrollment. If the PQ
///   component is missing and the directory is not writable, or if generating /
///   persisting it fails, this is a **hard error — never a classical fallback**
///   (fail closed, matching the node's default Hybrid policy).
/// - Under [`CryptoPolicy::Classical`] no PQ key is loaded or generated.
pub fn load_or_generate_user_identity(
    secrets_dir: &std::path::Path,
    policy: hyprstream_rpc::crypto::CryptoPolicy,
) -> Result<UserIdentityKeys> {
    use hyprstream_rpc::crypto::pq;

    let (ed, _vk) = load_or_generate_user_signing_key(secrets_dir)?;

    if !policy.uses_pq() {
        return Ok(UserIdentityKeys { ed, mldsa: None });
    }

    // Hybrid: load the PQ sibling if present, else generate (fail closed).
    if let Some(mut seed_bytes) = read_secret(secrets_dir, USER_PQ_NAME)? {
        let mut seed: [u8; 32] = seed_bytes
            .as_slice()
            .try_into()
            .map_err(|_| anyhow!("secret '{}' must be 32 bytes (ML-DSA-65 seed)", USER_PQ_NAME))?;
        let mldsa = pq::ml_dsa_sk_from_seed(&seed);
        seed_bytes.zeroize();
        seed.zeroize();
        tracing::info!("Loaded hybrid PQ (ML-DSA-65) user key from '{}'", secrets_dir.display());
        return Ok(UserIdentityKeys { ed, mldsa: Some(mldsa) });
    }

    // Missing PQ component. Under Hybrid this must be generated; if we cannot
    // write, fail closed rather than mint a classical-only identity.
    if !is_writable(secrets_dir) {
        return Err(anyhow!(
            "CryptoPolicy is Hybrid but the post-quantum component '{}' is missing \
             and '{}' is not writable — refusing to fall back to a classical-only \
             identity (fail closed). Provision the ML-DSA-65 key or set the policy \
             to Classical explicitly.",
            USER_PQ_NAME,
            secrets_dir.display()
        ));
    }

    let (mldsa, _mldsa_vk) = pq::ml_dsa_generate_keypair();
    let mut seed = pq::ml_dsa_sk_to_seed(&mldsa);
    let result = write_secret(secrets_dir, USER_PQ_NAME, &seed);
    seed.zeroize();
    result.with_context(|| {
        format!(
            "CryptoPolicy is Hybrid but persisting the ML-DSA-65 component '{}' failed \
             — refusing a classical-only fallback (fail closed)",
            USER_PQ_NAME
        )
    })?;
    tracing::info!(
        "Generated new hybrid PQ (ML-DSA-65) user key → '{}/{}'",
        secrets_dir.display(),
        USER_PQ_NAME
    );
    Ok(UserIdentityKeys { ed, mldsa: Some(mldsa) })
}

/// Paths of any prior key files displaced by an adopt, so the caller can report
/// the backup(s).
#[derive(Debug, Default)]
pub struct InstalledIdentityBackup {
    pub ed: Option<std::path::PathBuf>,
    pub pq: Option<std::path::PathBuf>,
}

/// Install adopted identity key material (the `--ssh`/`--key` path), backing up
/// any displaced key files first.
///
/// The adopted Ed25519 key becomes the client's `user-signing-key` (via
/// [`install_user_signing_key`]). Under `Hybrid`, a **freshly generated**
/// ML-DSA-65 key is installed as its sibling — an adopted SSH/seed key can never
/// make the identity hybrid by itself, so a new PQ component is minted and bound
/// to the same anchor. Under `Classical`, no PQ sibling is written, and any
/// pre-existing sibling is backed up and removed so the on-disk pair stays
/// consistent (the file layout never claims hybrid without a matching PQ key).
pub fn install_user_identity(
    secrets_dir: &std::path::Path,
    ed_sk: &SigningKey,
    mldsa_sk: Option<&hyprstream_rpc::crypto::pq::MlDsaSigningKey>,
) -> Result<InstalledIdentityBackup> {
    use hyprstream_rpc::crypto::pq;

    let ed_backup = install_user_signing_key(secrets_dir, ed_sk)?;

    let pq_backup = match mldsa_sk {
        Some(mldsa) => {
            // Back up any existing PQ sibling, then install the fresh one.
            let bak = backup_secret_if_present(secrets_dir, USER_PQ_NAME, USER_PQ_BAK)?;
            let mut seed = pq::ml_dsa_sk_to_seed(mldsa);
            let result = write_secret(secrets_dir, USER_PQ_NAME, &seed);
            seed.zeroize();
            result?;
            tracing::info!(
                "Installed adopted hybrid PQ (ML-DSA-65) user key → '{}/{}'",
                secrets_dir.display(),
                USER_PQ_NAME
            );
            bak
        }
        None => {
            // Classical adopt: retire any stale PQ sibling so the pair is
            // consistent (never leave a PQ seed bound to a displaced anchor).
            let bak = backup_secret_if_present(secrets_dir, USER_PQ_NAME, USER_PQ_BAK)?;
            if bak.is_some() {
                let live = secrets_dir.join(USER_PQ_NAME);
                std::fs::remove_file(&live).with_context(|| {
                    format!("failed to retire stale PQ key '{}'", live.display())
                })?;
                tracing::info!("Retired stale PQ user key (classical adopt) → backup kept");
            }
            bak
        }
    };

    Ok(InstalledIdentityBackup { ed: ed_backup, pq: pq_backup })
}

/// Back up `name` → `bak_name` (via [`write_secret`], mode 0600 before persist)
/// if `name` exists, zeroizing the read buffer. Returns the backup path if made.
fn backup_secret_if_present(
    secrets_dir: &std::path::Path,
    name: &str,
    bak_name: &str,
) -> Result<Option<std::path::PathBuf>> {
    let path = secrets_dir.join(name);
    if !path.exists() {
        return Ok(None);
    }
    let mut existing = std::fs::read(&path)
        .with_context(|| format!("failed to read existing {name} for backup"))?;
    let result = write_secret(secrets_dir, bak_name, &existing);
    existing.zeroize();
    result.with_context(|| format!("failed to back up existing {name}"))?;
    Ok(Some(secrets_dir.join(bak_name)))
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

            if needs_renewal {
                // Read-only credentials dir (e.g. systemd $CREDENTIALS_DIRECTORY):
                // the imported cert ages past validity with no on-disk remedy from
                // inside the service. Fail loud so operators re-provision (#808).
                tracing::error!(
                    "TLS cert '{}' in '{}' is past its renewal threshold but the \
                     credentials directory is read-only; renewal was SKIPPED and the \
                     cert will expire. Re-run 'hyprstream service install' to \
                     re-provision a fresh certificate.",
                    cert_name,
                    secrets_dir.display()
                );
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

    static SECRETS_ENV_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

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

    /// RAII guard to set/unset a process env var for the duration of a test and
    /// restore the previous value, keeping env-mutating tests from leaking state.
    struct EnvVarGuard {
        key: String,
        prev: Option<String>,
    }
    impl EnvVarGuard {
        fn set(key: &str, val: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::set_var(key, val);
            Self { key: key.to_owned(), prev }
        }
        fn unset(key: &str) -> Self {
            let prev = std::env::var(key).ok();
            std::env::remove_var(key);
            Self { key: key.to_owned(), prev }
        }
    }
    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var(&self.key, v),
                None => std::env::remove_var(&self.key),
            }
        }
    }

    /// `credentials_dir()` precedence: `HYPRSTREAM__SECRETS__PATH` wins when set;
    /// otherwise resolution falls back to a `<config_dir>/credentials` path.
    /// Both cases in one test — the env var is process-global and tests run in
    /// parallel, so splitting them would race.
    #[test]
    fn test_credentials_dir_precedence() {
        let _serial = SECRETS_ENV_LOCK.lock();
        const VAR: &str = "HYPRSTREAM__SECRETS__PATH";

        // (a) env var set → returns exactly that path.
        {
            let _g = EnvVarGuard::set(VAR, "/run/credentials/hyprstream");
            let dir = credentials_dir().unwrap();
            assert_eq!(dir, std::path::PathBuf::from("/run/credentials/hyprstream"));
        }

        // (b) env var unset → default resolution yields a path ending in
        // `credentials` (via config load or the XDG fallback).
        {
            let _g = EnvVarGuard::unset(VAR);
            let dir = credentials_dir().unwrap();
            assert_eq!(
                dir.file_name().and_then(|n| n.to_str()),
                Some("credentials"),
                "default resolution should end in 'credentials': {}",
                dir.display()
            );
        }
    }

    #[test]
    fn test_path_override_alone_keeps_shared_profile_and_service_key_isolation() {
        let _serial = SECRETS_ENV_LOCK.lock();
        let dir = TempDir::new().unwrap();
        let _path = EnvVarGuard::set("HYPRSTREAM__SECRETS__PATH", dir.path().to_str().unwrap());
        let _profile = EnvVarGuard::unset(SECRETS_PROFILE_ENV);

        let profile = SecretsProfile::from_env().unwrap();
        assert_eq!(
            profile,
            SecretsProfile::SharedDirectory,
            "a general path override is not proof of per-service scoping"
        );
        let model = resolve_service_signing_key(dir.path(), "model", profile).unwrap();
        let worker = resolve_service_signing_key(dir.path(), "worker", profile).unwrap();
        assert_ne!(
            model.to_bytes(),
            worker.to_bytes(),
            "services sharing an overridden directory must retain independent keys"
        );
        assert!(dir.path().join("model").join("signing-key").exists());
        assert!(dir.path().join("worker").join("signing-key").exists());
        assert!(!dir.path().join("signing-key").exists());
    }

    #[test]
    fn test_service_jwt_write_round_trips_through_startup_read_layouts() {
        let dir = TempDir::new().unwrap();

        write_service_jwt_for_profile(
            dir.path(),
            "model",
            SecretsProfile::SharedDirectory,
            "shared.jwt",
        )
        .unwrap();
        assert_eq!(
            load_service_jwt_for_profile(dir.path(), "model", SecretsProfile::SharedDirectory,)
                .unwrap()
                .as_deref(),
            Some("shared.jwt")
        );
        assert!(dir.path().join("model").join("service-jwt").exists());

        let scoped = dir.path().join("scoped");
        write_service_jwt_for_profile(
            &scoped,
            "model",
            SecretsProfile::PerServiceScoped,
            "scoped.jwt",
        )
        .unwrap();
        assert_eq!(
            load_service_jwt_for_profile(&scoped, "model", SecretsProfile::PerServiceScoped,)
                .unwrap()
                .as_deref(),
            Some("scoped.jwt")
        );
        assert!(scoped.join("service-jwt").exists());
        assert!(!scoped.join("model").exists());
    }

    #[cfg(unix)]
    #[test]
    fn test_scoped_service_jwt_renewal_survives_read_only_credentials_restart() {
        use std::os::unix::fs::PermissionsExt as _;

        let _serial = SECRETS_ENV_LOCK.lock();
        let dir = TempDir::new().unwrap();
        let credentials = dir.path().join("credentials");
        let state = dir.path().join("state");
        std::fs::create_dir_all(&credentials).unwrap();
        std::fs::create_dir_all(&state).unwrap();
        write_secret(&credentials, "service-jwt", b"provisioned.jwt").unwrap();

        std::fs::set_permissions(&credentials, std::fs::Permissions::from_mode(0o500)).unwrap();
        let _state = EnvVarGuard::set("STATE_DIRECTORY", state.to_str().unwrap());
        let _instance = EnvVarGuard::unset("HYPRSTREAM_INSTANCE");

        write_service_jwt_for_profile(
            &credentials,
            "model",
            SecretsProfile::PerServiceScoped,
            "renewed.jwt",
        )
        .unwrap();
        assert_eq!(
            load_service_jwt_for_profile(&credentials, "model", SecretsProfile::PerServiceScoped,)
                .unwrap()
                .as_deref(),
            Some("renewed.jwt"),
            "startup must prefer the renewed JWT written to writable state"
        );
        assert_eq!(
            std::fs::read_to_string(state.join("credentials").join("model").join("service-jwt"))
                .unwrap(),
            "renewed.jwt"
        );
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

    // ── resolve_service_signing_key (#759 regression) ────────────────────────

    #[test]
    fn test_resolve_service_signing_key_policy_matches_bootstrap_root_key() {
        // Reproduces the bootstrap flow in `bootstrap_manager::do_bootstrap`:
        // the root key is written flat as "signing-key" and its verifying key
        // is what gets registered as `bootstrap_pubkeys["policy"]` — PolicyService
        // never gets an independent `policy/signing-key` file.
        let dir = TempDir::new().unwrap();
        let root_key = load_or_generate_node_signing_key(dir.path()).unwrap();

        // Standalone (non-systemd) `--ipc` startup, resolving policy's own key —
        // this must equal the root key bootstrap registered, not a freshly
        // minted independent key.
        let resolved =
            resolve_service_signing_key(dir.path(), "policy", SecretsProfile::SharedDirectory)
                .unwrap();
        assert_eq!(
            resolved.to_bytes(), root_key.to_bytes(),
            "policy must resolve to the flat root/CA key, matching bootstrap_pubkeys[\"policy\"]"
        );

        // No independent `policy/signing-key` file should have been created.
        assert!(
            read_secret(&dir.path().join("policy"), "signing-key").unwrap().is_none(),
            "resolving policy's key must not mint an independent per-service key file"
        );
    }

    #[test]
    fn test_resolve_service_signing_key_policy_matches_root_key_when_scoped() {
        let dir = TempDir::new().unwrap();
        let root_key = load_or_generate_node_signing_key(dir.path()).unwrap();
        let resolved =
            resolve_service_signing_key(dir.path(), "policy", SecretsProfile::PerServiceScoped)
                .unwrap();
        assert_eq!(resolved.to_bytes(), root_key.to_bytes());
    }

    #[test]
    fn test_resolve_service_signing_key_non_policy_uses_flat_key_when_scoped() {
        let dir = TempDir::new().unwrap();
        let flat_key = load_or_generate_node_signing_key(dir.path()).unwrap();
        let resolved =
            resolve_service_signing_key(dir.path(), "model", SecretsProfile::PerServiceScoped)
                .unwrap();

        assert_eq!(resolved.to_bytes(), flat_key.to_bytes());
        assert!(!dir.path().join("model").exists());
    }

    #[test]
    fn test_resolve_service_signing_key_non_policy_uses_independent_subdir_key() {
        let dir = TempDir::new().unwrap();
        let root_key = load_or_generate_node_signing_key(dir.path()).unwrap();

        let model_key =
            resolve_service_signing_key(dir.path(), "model", SecretsProfile::SharedDirectory)
                .unwrap();
        assert_ne!(
            model_key.to_bytes(), root_key.to_bytes(),
            "non-policy services must keep their own independent key, distinct from the root/CA key"
        );

        // Independent key file lives under the service subdirectory.
        let from_disk = read_secret(&dir.path().join("model"), "signing-key").unwrap().unwrap();
        assert_eq!(from_disk, model_key.to_bytes().to_vec());
    }

    #[test]
    fn test_resolve_service_signing_key_is_stable_across_calls() {
        // Simulates repeated resolution (e.g. across a restart) — the same
        // key must come back every time, for both policy and non-policy.
        let dir = TempDir::new().unwrap();
        let _root_key = load_or_generate_node_signing_key(dir.path()).unwrap();

        let policy_1 =
            resolve_service_signing_key(dir.path(), "policy", SecretsProfile::SharedDirectory)
                .unwrap();
        let policy_2 =
            resolve_service_signing_key(dir.path(), "policy", SecretsProfile::SharedDirectory)
                .unwrap();
        assert_eq!(policy_1.to_bytes(), policy_2.to_bytes());

        let model_1 =
            resolve_service_signing_key(dir.path(), "model", SecretsProfile::SharedDirectory)
                .unwrap();
        let model_2 =
            resolve_service_signing_key(dir.path(), "model", SecretsProfile::SharedDirectory)
                .unwrap();
        assert_eq!(model_1.to_bytes(), model_2.to_bytes());
    }

    // ── service key public sidecars (#1562 H1) ──────────────────────────────

    #[cfg(unix)]
    fn file_mode(path: &std::path::Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[test]
    fn test_service_key_sidecars_written_on_generate() {
        let dir = TempDir::new().unwrap();
        let key = load_or_generate_service_signing_key(dir.path(), "discovery").unwrap();
        let svc = dir.path().join("discovery");

        let pub_bytes = std::fs::read(svc.join(SIGNING_KEY_PUB_NAME)).unwrap();
        assert_eq!(pub_bytes, key.verifying_key().as_bytes());

        let hybrid_bytes = std::fs::read(svc.join(SERVICE_PUBKEY_HYBRID_NAME)).unwrap();
        let expected = BootstrapPubkey::for_service_key(&key).unwrap().to_key_bytes();
        assert_eq!(hybrid_bytes, expected);
        assert_eq!(hybrid_bytes.len(), 1984);

        #[cfg(unix)]
        {
            assert_eq!(file_mode(&svc.join("signing-key")), 0o600);
            assert_eq!(file_mode(&svc.join(SIGNING_KEY_PUB_NAME)), 0o644);
            assert_eq!(file_mode(&svc.join(SERVICE_PUBKEY_HYBRID_NAME)), 0o644);
        }
    }

    #[test]
    fn test_service_key_sidecars_backfilled_on_load_of_preexisting_seed() {
        let dir = TempDir::new().unwrap();
        // Simulate a pre-H1 install: only the seed exists, no sidecars.
        let seed = SigningKey::generate(&mut rand::rngs::OsRng);
        write_secret(&dir.path().join("discovery"), "signing-key", &seed.to_bytes()).unwrap();
        let svc = dir.path().join("discovery");
        assert!(!svc.join(SIGNING_KEY_PUB_NAME).exists());
        assert!(!svc.join(SERVICE_PUBKEY_HYBRID_NAME).exists());

        let loaded = load_or_generate_service_signing_key(dir.path(), "discovery").unwrap();
        assert_eq!(
            loaded.to_bytes(),
            seed.to_bytes(),
            "load must adopt the existing seed, not rotate it"
        );

        let pub_bytes = std::fs::read(svc.join(SIGNING_KEY_PUB_NAME)).unwrap();
        assert_eq!(pub_bytes, seed.verifying_key().as_bytes());
        let hybrid_bytes = std::fs::read(svc.join(SERVICE_PUBKEY_HYBRID_NAME)).unwrap();
        assert_eq!(
            hybrid_bytes,
            BootstrapPubkey::for_service_key(&seed).unwrap().to_key_bytes()
        );
        #[cfg(unix)]
        {
            assert_eq!(file_mode(&svc.join(SIGNING_KEY_PUB_NAME)), 0o644);
            assert_eq!(file_mode(&svc.join(SERVICE_PUBKEY_HYBRID_NAME)), 0o644);
        }
    }

    #[test]
    fn test_service_key_sidecars_idempotent() {
        let dir = TempDir::new().unwrap();
        let first = load_or_generate_service_signing_key(dir.path(), "registry").unwrap();
        let svc = dir.path().join("registry");
        let pub1 = std::fs::read(svc.join(SIGNING_KEY_PUB_NAME)).unwrap();
        let hyb1 = std::fs::read(svc.join(SERVICE_PUBKEY_HYBRID_NAME)).unwrap();

        let second = load_or_generate_service_signing_key(dir.path(), "registry").unwrap();
        assert_eq!(
            first.to_bytes(),
            second.to_bytes(),
            "re-running must not rotate the key"
        );
        assert_eq!(std::fs::read(svc.join(SIGNING_KEY_PUB_NAME)).unwrap(), pub1);
        assert_eq!(std::fs::read(svc.join(SERVICE_PUBKEY_HYBRID_NAME)).unwrap(), hyb1);
        #[cfg(unix)]
        {
            assert_eq!(file_mode(&svc.join(SIGNING_KEY_PUB_NAME)), 0o644);
            assert_eq!(file_mode(&svc.join(SERVICE_PUBKEY_HYBRID_NAME)), 0o644);
        }
    }

    #[test]
    fn test_service_key_sidecars_contain_no_secret_material() {
        let dir = TempDir::new().unwrap();
        let key = load_or_generate_service_signing_key(dir.path(), "model").unwrap();
        let seed = key.to_bytes();
        let svc = dir.path().join("model");

        for name in [SIGNING_KEY_PUB_NAME, SERVICE_PUBKEY_HYBRID_NAME] {
            let bytes = std::fs::read(svc.join(name)).unwrap();
            assert_ne!(bytes, seed.as_slice());
            assert!(
                !bytes.windows(seed.len()).any(|w| w == seed.as_slice()),
                "{name} must not embed the seed anywhere"
            );
        }

        // Positive control: the sidecar bytes are exactly the public derivation
        // of the seed — Ed25519 verifying key, then the derived ML-DSA-65
        // verifying key.
        let pub_bytes = std::fs::read(svc.join(SIGNING_KEY_PUB_NAME)).unwrap();
        assert_eq!(pub_bytes, key.verifying_key().as_bytes());
        let pq_sk = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&key);
        let pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq_sk);
        let hybrid_bytes = std::fs::read(svc.join(SERVICE_PUBKEY_HYBRID_NAME)).unwrap();
        assert_eq!(&hybrid_bytes[..32], key.verifying_key().as_bytes());
        assert_eq!(&hybrid_bytes[32..], pq_vk.as_slice());
    }

    #[test]
    fn test_service_signing_key_dir_matches_resolve_layout() {
        let base = std::path::Path::new("/credentials");
        assert_eq!(
            service_signing_key_dir(base, "policy", SecretsProfile::SharedDirectory),
            base
        );
        assert_eq!(
            service_signing_key_dir(base, "policy", SecretsProfile::PerServiceScoped),
            base
        );
        assert_eq!(
            service_signing_key_dir(base, "model", SecretsProfile::PerServiceScoped),
            base
        );
        assert_eq!(
            service_signing_key_dir(base, "model", SecretsProfile::SharedDirectory),
            base.join("model")
        );
    }

    // ── bootstrap-pubkeys wire format ────────────────────────────────────────

    fn bootstrap_ed_key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    /// Write a `bootstrap-pubkeys` file verbatim, as an external provisioner
    /// (or an older release of this code) would have left it on disk.
    fn write_raw_bootstrap_pubkeys(dir: &std::path::Path, entries: &[(&str, &str)]) {
        let json: std::collections::BTreeMap<&str, &str> = entries.iter().copied().collect();
        write_secret(dir, "bootstrap-pubkeys", &serde_json::to_vec(&json).unwrap()).unwrap();
    }

    // ── mandatory hybrid service entries ─────────────────────────────────────

    /// A classical service entry is a hard, actionable error — never a silent
    /// downgrade to the classical floor.
    #[test]
    fn classical_service_entry_is_a_hard_actionable_error() {
        let dir = TempDir::new().unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert(
            "discovery".to_owned(),
            BootstrapPubkey::classical(bootstrap_ed_key(11).verifying_key()),
        );
        map.insert(
            "policy".to_owned(),
            BootstrapPubkey::for_service_key(&bootstrap_ed_key(12)).unwrap(),
        );
        write_bootstrap_pubkeys_hybrid(dir.path(), &map).unwrap();

        // The low-level loader still reads the file, so the error can name the
        // offending entries precisely.
        let loaded = load_bootstrap_pubkeys_hybrid(dir.path()).unwrap();
        assert_eq!(loaded.len(), 2);

        let err = format!("{:#}", ensure_bootstrap_pubkeys_hybrid(&loaded).unwrap_err());
        assert!(err.contains("discovery"), "error names the classical service: {err}");
        assert!(
            !err.contains("policy"),
            "error must not implicate the hybrid service: {err}"
        );
        assert!(err.contains("wizard"), "error names the working recovery: {err}");
        assert!(
            !err.contains("service repair"),
            "error must not name the dead-end 'service repair' command: {err}"
        );
        assert!(err.contains("ML-DSA-65"), "error names what is missing: {err}");
    }

    /// An all-hybrid file passes, and an unprovisioned (empty) node is not an
    /// error — there are no service identities to be wrong about yet.
    #[test]
    fn hybrid_service_entries_and_empty_file_both_pass() {
        let mut map = std::collections::HashMap::new();
        for (name, seed) in [("policy", 21u8), ("discovery", 22), ("inference", 23)] {
            map.insert(
                name.to_owned(),
                BootstrapPubkey::for_service_key(&bootstrap_ed_key(seed)).unwrap(),
            );
        }
        ensure_bootstrap_pubkeys_hybrid(&map).unwrap();
        ensure_bootstrap_pubkeys_hybrid(&std::collections::HashMap::new()).unwrap();
    }

    /// H3: in an OS-owned deployment an allowlisted service without a
    /// chain-signed enrollment attestation fails closed — the missing-file
    /// check fires before any chain access, so no trust dir is needed here.
    #[test]
    fn enrollment_attestation_missing_fails_closed() {
        let dir = TempDir::new().unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert(
            "discovery".to_owned(),
            BootstrapPubkey::for_service_key(&bootstrap_ed_key(41)).unwrap(),
        );
        let err = ensure_bootstrap_pubkeys_enrolled(dir.path(), &map)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(BOOTSTRAP_PUBKEYS_ENROLLMENT_DIR),
            "error names the enrollment directory: {err}"
        );
        assert!(err.contains("discovery"), "error names the service: {err}");
    }

    /// H3: services outside the fixed enrollment allowlist cannot be enrolled
    /// by design, so they require no attestation — and an empty (unprovisioned)
    /// node passes, matching the hybrid check's posture.
    #[test]
    fn unallowlisted_bootstrap_entries_need_no_attestation() {
        let dir = TempDir::new().unwrap();
        let mut map = std::collections::HashMap::new();
        map.insert(
            "inference".to_owned(),
            BootstrapPubkey::for_service_key(&bootstrap_ed_key(42)).unwrap(),
        );
        ensure_bootstrap_pubkeys_enrolled(dir.path(), &map).unwrap();
        ensure_bootstrap_pubkeys_enrolled(dir.path(), &std::collections::HashMap::new()).unwrap();
    }

    /// A provisioned service entry round-trips through the file and then
    /// verifies a real hybrid signature made with the keys the service actually
    /// signs with — the Ed25519 key plus the ML-DSA-65 key derived from it.
    #[test]
    fn provisioned_service_entry_verifies_a_hybrid_signature_end_to_end() {
        use ed25519_dalek::Signer;

        let dir = TempDir::new().unwrap();
        let service_key = bootstrap_ed_key(31);

        let mut map = std::collections::HashMap::new();
        map.insert(
            "discovery".to_owned(),
            BootstrapPubkey::for_service_key(&service_key).unwrap(),
        );
        write_bootstrap_pubkeys_hybrid(dir.path(), &map).unwrap();

        let loaded = load_bootstrap_pubkeys_hybrid(dir.path()).unwrap();
        let entry = &loaded["discovery"];
        assert!(entry.is_hybrid());
        ensure_bootstrap_pubkeys_hybrid(&loaded).unwrap();

        // Sign exactly as the service's own signer does.
        let msg = b"service response payload";
        let ed_sig = service_key.sign(msg);
        let pq_sk = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&service_key);
        let pq_sig = hyprstream_rpc::crypto::pq::ml_dsa_sign(&pq_sk, msg);

        entry.verify(msg, &ed_sig, Some(&pq_sig)).unwrap();

        // And the Ed25519 signature alone is not enough for a provisioned service.
        assert!(
            entry.verify(msg, &ed_sig, None).is_err(),
            "a provisioned service entry must require both signatures"
        );
    }

    #[test]
    fn bootstrap_pubkeys_classical_round_trip() {
        let dir = TempDir::new().unwrap();
        let vk = bootstrap_ed_key(7).verifying_key();

        let mut map = std::collections::HashMap::new();
        map.insert("discovery".to_owned(), vk);
        write_bootstrap_pubkeys(dir.path(), &map).unwrap();

        let loaded = load_bootstrap_pubkeys(dir.path()).unwrap();
        assert_eq!(loaded.get("discovery"), Some(&vk));

        let hybrid = load_bootstrap_pubkeys_hybrid(dir.path()).unwrap();
        let entry = &hybrid["discovery"];
        assert!(!entry.is_hybrid(), "an entry written from a bare Ed25519 key is classical");
        assert_eq!(entry.to_key_bytes().len(), 32);
    }

    #[test]
    fn bootstrap_pubkeys_hybrid_round_trip() {
        let dir = TempDir::new().unwrap();
        let ed = bootstrap_ed_key(9).verifying_key();
        let (_pq_sk, pq_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();

        let mut map = std::collections::HashMap::new();
        map.insert("discovery".to_owned(), BootstrapPubkey::hybrid(ed, pq_vk.clone()));
        write_bootstrap_pubkeys_hybrid(dir.path(), &map).unwrap();

        let loaded = load_bootstrap_pubkeys_hybrid(dir.path()).unwrap();
        let entry = &loaded["discovery"];
        assert!(entry.is_hybrid());
        assert_eq!(entry.ed25519, ed);
        assert_eq!(
            hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(entry.ml_dsa_65.as_ref().unwrap()),
            hyprstream_rpc::crypto::pq::ml_dsa_vk_bytes(&pq_vk)
        );
        assert_eq!(entry.to_key_bytes().len(), 1984);

        // The Ed25519-only projection still sees the anchor, so callers that
        // predate hybrid entries keep resolving the same key.
        assert_eq!(load_bootstrap_pubkeys(dir.path()).unwrap().get("discovery"), Some(&ed));
    }

    #[test]
    fn bootstrap_pubkeys_legacy_file_loads_unchanged() {
        let dir = TempDir::new().unwrap();
        let discovery = bootstrap_ed_key(0x11).verifying_key();
        let policy = bootstrap_ed_key(0x22).verifying_key();

        // Exactly the shape deployed nodes and external provisioning emit:
        // a flat map of URL-safe-no-pad base64 over 32 raw Ed25519 bytes.
        write_raw_bootstrap_pubkeys(
            dir.path(),
            &[
                ("discovery", &URL_SAFE_NO_PAD.encode(discovery.as_bytes())),
                ("policy", &URL_SAFE_NO_PAD.encode(policy.as_bytes())),
            ],
        );

        let loaded = load_bootstrap_pubkeys(dir.path()).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get("discovery"), Some(&discovery));
        assert_eq!(loaded.get("policy"), Some(&policy));

        let hybrid = load_bootstrap_pubkeys_hybrid(dir.path()).unwrap();
        assert!(hybrid.values().all(|e| !e.is_hybrid()));
    }

    #[test]
    fn bootstrap_pubkeys_mixed_file_verifies_per_identity() {
        let dir = TempDir::new().unwrap();
        let legacy_sk = bootstrap_ed_key(0x33);
        let hybrid_ed_sk = bootstrap_ed_key(0x44);
        let (pq_sk, pq_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();

        let hybrid_entry = BootstrapPubkey::hybrid(hybrid_ed_sk.verifying_key(), pq_vk);
        write_raw_bootstrap_pubkeys(
            dir.path(),
            &[
                ("policy", &URL_SAFE_NO_PAD.encode(legacy_sk.verifying_key().as_bytes())),
                ("discovery", &URL_SAFE_NO_PAD.encode(hybrid_entry.to_key_bytes())),
            ],
        );

        let loaded = load_bootstrap_pubkeys_hybrid(dir.path()).unwrap();
        assert_eq!(loaded.len(), 2);

        let msg = b"bootstrap trust seed";

        // Classical entry: an Ed25519 signature alone is sufficient, and no
        // post-quantum material is demanded of it.
        let legacy = &loaded["policy"];
        assert!(!legacy.is_hybrid());
        let legacy_sig = {
            use ed25519_dalek::Signer;
            legacy_sk.sign(msg)
        };
        legacy.verify(msg, &legacy_sig, None).unwrap();

        // Hybrid entry: both signatures are required, and the classical half
        // alone is rejected.
        let hybrid = &loaded["discovery"];
        assert!(hybrid.is_hybrid());
        let hybrid_ed_sig = {
            use ed25519_dalek::Signer;
            hybrid_ed_sk.sign(msg)
        };
        let pq_sig = hyprstream_rpc::crypto::pq::ml_dsa_sign(&pq_sk, msg);
        hybrid.verify(msg, &hybrid_ed_sig, Some(&pq_sig)).unwrap();
        assert!(
            hybrid.verify(msg, &hybrid_ed_sig, None).is_err(),
            "a hybrid entry must not verify on the Ed25519 signature alone"
        );

        // A post-quantum signature has nothing to check against on a classical
        // entry, so it is rejected rather than silently ignored.
        assert!(legacy.verify(msg, &legacy_sig, Some(&pq_sig)).is_err());
    }

    #[test]
    fn bootstrap_pubkeys_reject_wrong_length_values() {
        let dir = TempDir::new().unwrap();
        // 33 bytes: neither the classical nor the hybrid length.
        write_raw_bootstrap_pubkeys(dir.path(), &[("discovery", &URL_SAFE_NO_PAD.encode([0u8; 33]))]);

        let err = load_bootstrap_pubkeys(dir.path()).unwrap_err().to_string();
        assert!(err.contains("'discovery'"), "error names the offending service: {err}");
        assert!(err.contains("32 bytes"), "error states the classical shape: {err}");
        assert!(err.contains("1984 bytes"), "error states the hybrid shape: {err}");
        assert!(err.contains("got 33 bytes"), "error states what was found: {err}");
    }

    #[test]
    fn bootstrap_pubkeys_reject_standard_alphabet_base64() {
        let dir = TempDir::new().unwrap();
        // Standard-alphabet base64 with padding is not the documented alphabet.
        write_raw_bootstrap_pubkeys(dir.path(), &[("discovery", "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8=")]);

        let err = format!("{:#}", load_bootstrap_pubkeys(dir.path()).unwrap_err());
        assert!(err.contains("URL-safe-no-pad"), "error states the expected alphabet: {err}");
    }

    #[test]
    fn bootstrap_pubkeys_reject_malformed_pq_length() {
        let dir = TempDir::new().unwrap();
        // Correct Ed25519 prefix, truncated ML-DSA-65 tail.
        let mut bytes = bootstrap_ed_key(0x55).verifying_key().as_bytes().to_vec();
        bytes.extend_from_slice(&[0u8; 1951]);
        write_raw_bootstrap_pubkeys(dir.path(), &[("discovery", &URL_SAFE_NO_PAD.encode(&bytes))]);

        let err = load_bootstrap_pubkeys_hybrid(dir.path()).unwrap_err().to_string();
        assert!(err.contains("1984 bytes"), "error states the hybrid shape: {err}");
    }

    /// The hybrid `discovery` value published as the worked example in
    /// `docs/bootstrap-pubkeys-format.md`. Kept verbatim so the documented
    /// example cannot drift away from what the parser actually accepts.
    const DOC_EXAMPLE_HYBRID_VALUE: &str = "JIrL26-eBQGW3nBL6i1odw5RkVDRA7WH2uLZytU92TAXHtJkdXNEM19SNj1GPxnt-7mImdVPdRRENw5lnPLIVkDc3BiWsqDvmLI2x9roidQrSRArrQ-itYpEyQaLf-y6ckpa1NdkeN9ibwNu393M3E8PD2N3Vh9vx_sD2mbn532l0QXGV3i-9lQppfKcFwzinBkQN7eLLWRrInMfbE3IY_7FnC9Q0scH7bUMBvNY7k5jEDNtz1ZpEWgR-1u2M-iF221WgzT9AnQX5X88dd9gYs0SsIHLjx5o8-nnHk0XRxPq1JtegEupUKLf5a7e3IEm62VSXRJhH_V2IA9PAJMlSL8rJGYHdMwjscP1DyKCpCquSgYIH8zJDQCPqVUFBujtdq3x-eMFd0CxpVssPuLeoNb_4YZbtSwdk6iMMPupiDYIKOgknxymsnYGxR1PiA_7b1lZVaD62pCZHSCi9nJEDVoWuQ0_cSMB589zaDSTcI7r5Ks6l0MUPtVsa_yWQ4jwS8GnDMikXXmLvKegE5V6rZpG6AXx707kEvpxyESIzwnMU9UO2h4MGI1uWp36SXzHxF5RSZrY3Wr1ei1MwnJj7kgve4w1skjtZ32EE4A8r7R8mIl8L6LkXsD6n4FXFWQWocLdBerVvVBSDTDca8mQ3Lqr-3Q8wasUx-Ifqk6AAOZTmPseQ50uoiZnr0MCzxVkbAmvB5o_I5J_BqxKqQUimv0xiq69n9Vr__7FS8HTd9139LUPBkBDduZeH4qqJyIkDshEPJeXqOwBQZDIupnjc-PvDNKRM6o8Kg1OtaBjYwWWL6pVFRXYw3NeGb52N77-T9dEOCR659RHFrI-741xIaJ_gTJb6G43C7Go-EMI0aMkzakh-LSyTrv4C2s-KYksgVap-EsREaKmX1FAinmWsn0222V-tzJ-9dvPKKwCnTyTXpYRK3ltR-3LnP-onbg4ysP_tpVMg8K7jAeCzYsEEZCIjoJr5qpTkwGncfytwVvT6sHsoD988S10in7Dz9TNcNnFmb_foYqELht6jhAv_X6nf8IJEZgaLNaz9eV-_L87TdgQDF4iKrgHcuUZjPlfi-az1B6uV1Z7pqUsqJBT5XQkCqr3j4chZ2gNEw8SpconZVuXsT1tEtSWyo9GmvNntsiCvGM30faWGj-_cIsrCveMeMfLLsfNP3X7jG7Jgb7uYbj50Mwf7b-VqrdAmIJ0CMmCSygBaWV7YDx_qTEEuE7oL-R6_fXkhmXeZYyhFPF-As70CrVW-KwAJVp0mYasmPwu9YZ-Mlerg-QgHiAXnNpVjXE-wvnqP6JKbrbgaUZiLC4gXn0VTnB70y4e97Efl8ztGqE7AazPxwRMDyfher5oGpVm2lh9StBZYqaoAO7XQTUKvczld7imzqDugagVpWRW7ffE5qKp0ezq5PUwws4GB7UfTrmBm2gQ2HmT8aouV15b5zJaI1x9sOsNUjFrL1yL3V-Fi8adOXfRc5MR1JzdsaAki5YZhyjjYEHzdZfD8mwvN8jxN5U03tq495bR87ER_tHZNNd875gYZNYQ-pkD8zBlLWq54mDj5AKCwdTce3gOVTx9CtNYm3LFxsaB9VL_B93X81IR_Czaf_cHNQPsTpd8wWCWTZbc9zBhHUYtlrTrxhzXV68yBDcjnVF3LkbOcqapBCsUTrQ6XgloOp-fkUYd1LWTQN7ofcCXvR7x30BXXsM_qZnlUR4zminj2lWlJacV90Uqf7SL4HAEefqH8btyoVepUG_5JYQ7bp7NsUWUlw-Jc0hkumvNcaNX5uNFCLIIVTGgA9dG8ELuC3g-bhVjokIr4wSHVCyjFJs2qo93VdHy-abKdwpjeRwo6r66uaideZKEBxKUlZ_XHiZmT_FksJZIOHrq-8gQVPyjwo0oyI9Dq8xF-5aVN8NpHaJU7x4VcsndfHqsY5CGQUV0ZfuDpA94Srx-vvnvAJeM8ql63Ng2-0dNDra6fr8fgdEZeyJaIhdVBU8wBlshyeX32N2dbiJR7qWY5VqLbaKjePvqLa1LnnMtWdS2PqMX2UE_Q44Oa_FUdOB2sFdvKpxEbHDS74w5sd6C5AZzYPNUL1ICrV1F-c0Mxm9Z2j8PFEuYDz6fpR4Z1iDzLH9QKFUwus9fPzCU7b-ptTU3T3vnFTN4NLgxR6Ih78B-Mp3QrrCOxW3NtBNpnImuqHgRYBBPkg_dM659hgNtgCMYefiQuWJlpS22GG3kwPVuHoG8eqa-NDtxgMgFOPy30IKJcLh9ooG-QD9reZht_TBkLWbDW1JfeINAeJ86oZ5g3qtTdVAftVHaRCHLQN0ClLXPquM-v1GJm2F_tIXFJF-utsp0SgCDKo2VUymmMSgXQL3iP29lHQcp6Jzgisk6ZAEs6MqzrGV-T0cl0sQ4pKECzGHUL-8B8CKmMtshHZLCX4hvc3ALGd8ls72e6sh2JkFcU84I_8GLPgJu66dN5eZtOPR9CkmTKa2Ph1MgXqBGwmcYcvlg5GVZvzZC-UlhTrF7WAbObzDrGQbIGHBKG09dobSuSXpiSmZljJUh-88qsTXEVW12C4ZoWyDQ4OzH09JofMByTzb-k5Hn8cKIawcJBqCaAXElF5Z6MLrxxwagcL9XuMMdKG3UtEWY9HrsGY6qsA";

    #[test]
    fn documented_hybrid_example_loads_as_a_hybrid_entry() {
        let dir = TempDir::new().unwrap();
        write_raw_bootstrap_pubkeys(dir.path(), &[("discovery", DOC_EXAMPLE_HYBRID_VALUE)]);

        let loaded = load_bootstrap_pubkeys_hybrid(dir.path()).unwrap();
        let entry = &loaded["discovery"];
        assert!(entry.is_hybrid(), "the documented example must be a hybrid entry");
        assert_eq!(entry.to_key_bytes().len(), 1984);
        // The documented Ed25519 anchor is the leading 43 base64 characters.
        assert!(
            DOC_EXAMPLE_HYBRID_VALUE
                .starts_with(&URL_SAFE_NO_PAD.encode(entry.ed25519.as_bytes())),
            "the documented anchor prefix must be the entry's Ed25519 key"
        );
    }
}
