//! Handlers for `hyprstream user` CLI subcommands.
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use ed25519_dalek::VerifyingKey;
use std::io::Write;
use std::path::Path;

use crate::auth::{LocalKeyStore, UserStore};

/// Handle `user register <username> <pubkey_base64>`
pub async fn handle_user_register(
    credentials_dir: &Path,
    username: &str,
    pubkey_base64: &str,
) -> Result<()> {
    let raw = STANDARD
        .decode(pubkey_base64)
        .context("Invalid base64 for public key")?;
    let bytes: [u8; 32] = raw
        .try_into()
        .map_err(|_| anyhow::anyhow!("Public key must be 32 bytes (Ed25519)"))?;
    let pubkey =
        VerifyingKey::from_bytes(&bytes).context("Invalid Ed25519 public key")?;

    let store =
        LocalKeyStore::load(credentials_dir).context("Failed to open credential store")?;
    store.register(username).await
        .context("Failed to register user")?;
    store.add_pubkey(username, pubkey, None).await
        .context("Failed to add pubkey")?;
    println!("Registered user '{username}'");
    Ok(())
}

/// Handle `user list`
pub async fn handle_user_list(credentials_dir: &Path) -> Result<()> {
    let store =
        LocalKeyStore::load(credentials_dir).context("Failed to open credential store")?;
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
    let store =
        LocalKeyStore::load(credentials_dir).context("Failed to open credential store")?;
    let removed = store.remove(username).await?;
    if removed {
        println!("Removed user '{username}'");
    } else {
        println!("User '{username}' not found");
    }
    Ok(())
}

/// Handle `user show <username>`
pub async fn handle_user_show(credentials_dir: &Path, username: &str) -> Result<()> {
    let store =
        LocalKeyStore::load(credentials_dir).context("Failed to open credential store")?;
    let pubkeys = store.list_pubkeys(username).await?;
    if pubkeys.is_empty() {
        println!("User '{username}' has no pubkeys");
    } else {
        for pk in &pubkeys {
            let label = pk.label.as_deref().unwrap_or("(no label)");
            println!("{} {}", STANDARD.encode(pk.pubkey.as_bytes()), label);
        }
    }
    Ok(())
}
