//! Handlers for `hyprstream user` CLI subcommands.
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use ed25519_dalek::{SigningKey, VerifyingKey};
use std::io::{Read, Write};
use std::path::Path;

use crate::auth::{RocksDbUserStore, UserStore};
use crate::cli::commands::UserKeysImportFormat;
use crate::cli::enroll::{enroll_user, EnrollKeySource};

fn open_store(credentials_dir: &Path) -> Result<RocksDbUserStore> {
    RocksDbUserStore::open(credentials_dir).context("Failed to open credential store")
}

/// Resolve key material: from -f <path>, -f -, or inline data.
fn resolve_key_input(file: Option<&str>, data: Option<&str>) -> Result<String> {
    if let Some(path) = file {
        if path == "-" {
            let mut buf = String::new();
            std::io::stdin().read_to_string(&mut buf)?;
            return Ok(buf.trim().to_owned());
        }
        return std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read key file: {path}"))
            .map(|s| s.trim().to_owned());
    }
    data.map(|s| s.trim().to_owned())
        .ok_or_else(|| anyhow::anyhow!("Provide key via -f <path> or inline as the last argument"))
}

/// Parse an SSH Ed25519 public key line into a VerifyingKey.
///
/// Accepts: `ssh-ed25519 <base64> [comment]`
/// Wire format: u32be(11) "ssh-ed25519" u32be(32) <32-byte-key> = 51 bytes total.
fn parse_ssh_ed25519(line: &str) -> Result<VerifyingKey> {
    let line = line.trim();
    let b64_part = if let Some(rest) = line.strip_prefix("ssh-ed25519 ") {
        rest.split_whitespace().next().unwrap_or(rest)
    } else {
        anyhow::bail!("Not an SSH Ed25519 key line (expected 'ssh-ed25519 AAAA...'). To import a raw key use 'keys import <username> raw'.");
    };

    let wire = STANDARD
        .decode(b64_part)
        .context("Invalid base64 in SSH public key")?;

    if wire.len() != 51 {
        anyhow::bail!(
            "SSH Ed25519 wire format must be 51 bytes, got {}. Ensure this is an ssh-ed25519 key (not another type).",
            wire.len()
        );
    }

    // Validate the embedded type string
    let type_len_bytes: [u8; 4] = wire[0..4]
        .try_into()
        .map_err(|_| anyhow::anyhow!("Malformed SSH wire prefix"))?;
    let type_len = u32::from_be_bytes(type_len_bytes) as usize;
    if type_len != 11 || &wire[4..15] != b"ssh-ed25519" {
        anyhow::bail!("Key type mismatch: only Ed25519 keys are supported; got ssh-rsa or similar. Use: ssh-keygen -t ed25519");
    }

    let key_bytes: [u8; 32] = wire[19..51]
        .try_into()
        .map_err(|_| anyhow::anyhow!("Malformed SSH Ed25519 wire payload"))?;

    VerifyingKey::from_bytes(&key_bytes).context("Invalid Ed25519 public key bytes")
}

/// Handle `user register <username>`
pub async fn handle_user_register(credentials_dir: &Path, username: &str) -> Result<()> {
    let store = open_store(credentials_dir)?;
    store.register(username).await.context("Failed to register user")?;
    println!("Registered user '{username}'");
    Ok(())
}

/// Handle `user list`
pub async fn handle_user_list(credentials_dir: &Path) -> Result<()> {
    let store = open_store(credentials_dir)?;
    let mut users = store.list_users().await;
    if users.is_empty() {
        println!("No users registered.");
    } else {
        users.sort();
        for u in &users {
            println!("{u}");
        }
    }
    Ok(())
}

/// Handle `user remove <username> [--force]`
pub async fn handle_user_remove(
    credentials_dir: &Path,
    username: &str,
    force: bool,
) -> Result<()> {
    if !force {
        print!("Remove user '{username}'? [y/N] ");
        std::io::stdout().flush()?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if input.trim().to_lowercase() != "y" {
            println!("Cancelled.");
            return Ok(());
        }
    }
    let store = open_store(credentials_dir)?;
    let removed = store.remove(username).await?;
    if removed {
        println!("Removed user '{username}'");
    } else {
        println!("User '{username}' not found");
    }
    Ok(())
}

/// Handle `user keys list <username>`
pub async fn handle_user_keys_list(credentials_dir: &Path, username: &str) -> Result<()> {
    let store = open_store(credentials_dir)?;
    let pubkeys = store.list_pubkeys(username).await?;
    if pubkeys.is_empty() {
        println!("No keys registered for '{username}'");
    } else {
        for pk in &pubkeys {
            let label = pk.label.as_deref().unwrap_or("(no label)");
            let ts = pk.created_at;
            println!("{}  {}  algorithm={}  (added {})", pk.fingerprint, label, pk.algorithm.as_str(), ts);
        }
    }
    Ok(())
}

/// Handle `user keys import <username> <format>`
pub async fn handle_user_keys_import(
    credentials_dir: &Path,
    username: &str,
    format: &UserKeysImportFormat,
) -> Result<()> {
    let (pubkey, label) = match format {
        UserKeysImportFormat::SshEd25519 { file, data, label } => {
            let input = resolve_key_input(file.as_deref(), data.as_deref())?;
            let vk = parse_ssh_ed25519(&input)?;
            (vk, label.clone())
        }
        UserKeysImportFormat::Raw { file, data, label } => {
            let input = resolve_key_input(file.as_deref(), data.as_deref())?;
            let raw = STANDARD
                .decode(input.trim())
                .context("Invalid base64 for raw key")?;
            let bytes: [u8; 32] = raw
                .try_into()
                .map_err(|_| anyhow::anyhow!("Raw key must be 32 bytes (Ed25519)"))?;
            let vk = VerifyingKey::from_bytes(&bytes).context("Invalid Ed25519 public key")?;
            (vk, label.clone())
        }
    };

    let store = open_store(credentials_dir)?;
    let fingerprint = store
        .add_pubkey(username, pubkey, label)
        .await
        .context("Failed to add key")?;
    println!("Added key {fingerprint}");
    Ok(())
}

/// Handle `user create <username> [--role] [--generate|--key|--ssh] [--no-role]`.
///
/// One-command enrollment (#439): generates (default) or adopts a signing key,
/// installs it as the client's actual signing key, registers the user record,
/// binds the derived public key, and grants a role — collapsing the multi-step
/// dance whose partial failure silently authenticates as `anonymous`.
pub async fn handle_user_create(
    credentials_dir: &Path,
    username: &str,
    role: &str,
    no_role: bool,
    generate: bool,
    key: Option<&str>,
    ssh: Option<&str>,
) -> Result<()> {
    let store = open_store(credentials_dir)?;
    let secrets_dir = crate::config::HyprConfig::resolve_secrets_dir()
        .context("Failed to resolve secrets directory")?;

    let source = resolve_create_key_source(generate, key, ssh)?;

    let outcome = enroll_user(&store, &secrets_dir, username, source)
        .await
        .context("Failed to enroll user")?;

    println!("✓ Enrolled user '{}'", outcome.username);
    if let Some(bak) = &outcome.key_backed_up {
        println!("  Backed up prior signing key to {}", bak.display());
    }
    println!("  Signing key installed as the client key — the CLI now signs with this key.");
    println!(
        "  Fingerprint: {}  (algorithm={})",
        outcome.fingerprint,
        outcome.algorithm.as_str()
    );
    println!(
        "  Verify: `ssh-keygen -l -E sha256` of the key prints {}",
        outcome.fingerprint
    );

    if !no_role {
        // Best-effort role grant via the running PolicyService. Enrollment
        // already succeeded; a down service (or first-run setup) must not undo
        // it — point the operator at the manual command instead.
        match grant_role(username, role).await {
            Ok(()) => {}
            Err(e) => {
                println!(
                    "  ⚠ Could not grant role '{role}' automatically ({e}).\n    \
                     Run `hyprstream policy role add {username} {role}` once services are up."
                );
            }
        }
    }

    Ok(())
}

/// Resolve the `user create` key source from the `--generate`/`--key`/`--ssh`
/// flags. `--generate` (or no key flag) → [`EnrollKeySource::Generate`].
fn resolve_create_key_source(
    _generate: bool,
    key: Option<&str>,
    ssh: Option<&str>,
) -> Result<EnrollKeySource> {
    if let Some(path) = ssh {
        let sk = parse_openssh_ed25519(path)?;
        return Ok(EnrollKeySource::SigningKey(sk));
    }
    if let Some(path) = key {
        let seed = read_seed_file(path)?;
        return Ok(EnrollKeySource::RawSeed(seed));
    }
    Ok(EnrollKeySource::Generate)
}

/// Read a raw 32-byte Ed25519 seed from a file (or `-` stdin). Accepts either
/// raw 32 bytes or standard base64 of 32 bytes.
fn read_seed_file(path: &str) -> Result<[u8; 32]> {
    let raw = if path == "-" {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf)?;
        buf
    } else {
        std::fs::read(path).with_context(|| format!("Failed to read seed file: {path}"))?
    };

    let seed: [u8; 32] = if raw.len() == 32 {
        raw.try_into()
            .map_err(|_| anyhow::anyhow!("Seed file is not 32 bytes (Ed25519)"))?
    } else {
        // Try base64 of 32 bytes.
        let s = std::str::from_utf8(&raw)
            .map_err(|_| anyhow::anyhow!("Seed file is neither 32 raw bytes nor base64 of 32 bytes"))?;
        let decoded = STANDARD.decode(s.trim())
            .map_err(|_| anyhow::anyhow!("Seed file is neither 32 raw bytes nor base64 of 32 bytes"))?;
        decoded
            .try_into()
            .map_err(|_| anyhow::anyhow!("Decoded seed is not 32 bytes (Ed25519)"))?
    };
    Ok(seed)
}

/// Parse an OpenSSH Ed25519 private key from `path`, prompting for a passphrase
/// if the key is encrypted.
fn parse_openssh_ed25519(path: &str) -> Result<SigningKey> {
    let pem = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read OpenSSH key: {path}"))?;
    let key = ssh_key::PrivateKey::from_openssh(&pem)
        .with_context(|| format!("Failed to parse OpenSSH private key: {path}"))?;
    let key = if key.is_encrypted() {
        let passphrase = prompt_passphrase()?;
        key.decrypt(&passphrase)
            .context("Failed to decrypt OpenSSH key (wrong passphrase?)")?
    } else {
        key
    };
    match key.key_data() {
        ssh_key::private::KeypairData::Ed25519(ed) => {
            Ok(SigningKey::from(&ed.private))
        }
        _ => anyhow::bail!(
            "{path} is not an Ed25519 key ({}); only ssh-ed25519 is supported. \
             Generate one with: ssh-keygen -t ed25519",
            key.algorithm().as_str()
        ),
    }
}

/// Prompt the terminal for an OpenSSH passphrase without echoing it.
fn prompt_passphrase() -> Result<String> {
    inquire::Password::new("Passphrase (OpenSSH key is encrypted): ")
        .with_display_mode(inquire::PasswordDisplayMode::Hidden)
        .without_confirmation()
        .prompt()
        .map_err(|e| anyhow::anyhow!("failed to read passphrase: {e}"))
}

/// Grant a role to `username` via the running PolicyService.
async fn grant_role(username: &str, role: &str) -> Result<()> {
    let keys_dir = std::path::PathBuf::from("."); // load_or_generate_signing_key resolves via config
    let signing_key = crate::cli::load_or_generate_signing_key(&keys_dir).await?;
    crate::cli::handle_policy_role_add(&signing_key, username, role, false).await
}

/// Handle `user keys remove <username> <fingerprint>`
pub async fn handle_user_keys_remove(
    credentials_dir: &Path,
    username: &str,
    fingerprint: &str,
) -> Result<()> {
    // Normalize: ensure SHA256: prefix is present regardless of whether user included it
    let owned;
    let fp = if fingerprint.starts_with("SHA256:") {
        fingerprint
    } else {
        owned = format!("SHA256:{fingerprint}");
        &owned
    };
    let store = open_store(credentials_dir)?;
    let removed = store.remove_pubkey(username, fp).await?;
    if removed {
        println!("Removed key {fp}");
    } else {
        println!("Key {fp} not found for '{username}'");
    }
    Ok(())
}
