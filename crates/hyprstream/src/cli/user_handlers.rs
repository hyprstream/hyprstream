//! Handlers for `hyprstream user` CLI subcommands.
//!
//! These commands go through `PolicyService` RPC (`listUsers`/`registerUser`/
//! `removeUser`/`listUserKeys`/`addUserKey`/`removeUserKey`, #449) instead of
//! opening the RocksDB credential store directly — direct access from a
//! second process conflicted with any already-running service holding the
//! store's exclusive writer lock (`IO error: While lock file: .../LOCK:
//! Resource temporarily unavailable`).
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use ed25519_dalek::{SigningKey, VerifyingKey};
use std::io::{Read, Write};

use crate::cli::commands::UserKeysImportFormat;
use crate::cli::policy_handlers::create_policy_client;
use crate::services::generated::policy_client::{
    AddUserKey, RegisterUser, RemoveUser, RemoveUserKey, ListUserKeys,
};

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
///
/// This is pure client-side parsing — no store access — so it stays local to
/// the CLI; only the decoded raw key bytes are sent to PolicyService.
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
pub async fn handle_user_register(signing_key: &SigningKey, username: &str) -> Result<()> {
    let client = create_policy_client(signing_key)?;
    client
        .register_user(&RegisterUser { username: username.to_owned() })
        .await
        .context("Failed to register user via PolicyService. Are services running?")?;
    println!("Registered user '{username}'");
    Ok(())
}

/// Handle `user list`
pub async fn handle_user_list(signing_key: &SigningKey) -> Result<()> {
    let client = create_policy_client(signing_key)?;
    let mut users = client
        .list_users()
        .await
        .context("Failed to list users via PolicyService. Are services running?")?
        .usernames;
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
    signing_key: &SigningKey,
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
    let client = create_policy_client(signing_key)?;
    let removed = client
        .remove_user(&RemoveUser { username: username.to_owned() })
        .await
        .context("Failed to remove user via PolicyService. Are services running?")?;
    if removed {
        println!("Removed user '{username}'");
    } else {
        println!("User '{username}' not found");
    }
    Ok(())
}

/// Handle `user keys list <username>`
pub async fn handle_user_keys_list(signing_key: &SigningKey, username: &str) -> Result<()> {
    let client = create_policy_client(signing_key)?;
    let pubkeys = client
        .list_user_keys(&ListUserKeys { username: username.to_owned() })
        .await
        .context("Failed to list keys via PolicyService. Are services running?")?;
    if pubkeys.keys.is_empty() {
        println!("No keys registered for '{username}'");
    } else {
        for pk in &pubkeys.keys {
            let label = pk.label.as_deref().filter(|l| !l.is_empty()).unwrap_or("(no label)");
            println!("{}  {}  (added {})", pk.fingerprint, label, pk.created_at);
        }
    }
    Ok(())
}

/// Handle `user keys import <username> <format>`
pub async fn handle_user_keys_import(
    signing_key: &SigningKey,
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

    let client = create_policy_client(signing_key)?;
    let fingerprint = client
        .add_user_key(&AddUserKey {
            username: username.to_owned(),
            pubkey_raw: pubkey.as_bytes().to_vec(),
            label,
        })
        .await
        .context("Failed to add key via PolicyService. Are services running?")?;
    println!("Added key {fingerprint}");
    Ok(())
}

/// Handle `user keys remove <username> <fingerprint>`
pub async fn handle_user_keys_remove(
    signing_key: &SigningKey,
    username: &str,
    fingerprint: &str,
) -> Result<()> {
    // Normalize: ensure SHA256: prefix is present regardless of whether user included it
    let fp = if fingerprint.starts_with("SHA256:") {
        fingerprint.to_owned()
    } else {
        format!("SHA256:{fingerprint}")
    };
    let client = create_policy_client(signing_key)?;
    let removed = client
        .remove_user_key(&RemoveUserKey { username: username.to_owned(), fingerprint: fp.clone() })
        .await
        .context("Failed to remove key via PolicyService. Are services running?")?;
    if removed {
        println!("Removed key {fp}");
    } else {
        println!("Key {fp} not found for '{username}'");
    }
    Ok(())
}
