//! Handlers for `hyprstream user` CLI subcommands.
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use ed25519_dalek::VerifyingKey;
use std::io::{Read, Write};
use std::path::Path;

use crate::auth::{RocksDbUserStore, UserStore};
use crate::cli::commands::UserKeysImportFormat;

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
            println!("{}  {}  (added {})", pk.fingerprint, label, ts);
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
