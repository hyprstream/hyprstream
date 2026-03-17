//! User credential store abstraction.
//!
//! `UserStore` is the trait. `LocalKeyStore` is the implementation
//! backed by an age-encrypted TOML file, with the decryption key
//! in the OS keyring.

use age::secrecy::ExposeSecret;
use anyhow::{anyhow, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Abstraction over user credential stores.
pub trait UserStore: Send + Sync {
    /// Look up a user's Ed25519 public key by username.
    fn get_pubkey(&self, username: &str) -> Result<Option<VerifyingKey>>;
    /// Register a user with their Ed25519 public key.
    fn register(&mut self, username: &str, pubkey: VerifyingKey) -> Result<()>;
    /// Remove a user.
    fn remove(&mut self, username: &str) -> Result<bool>;
    /// List all registered usernames.
    fn list_users(&self) -> Vec<String>;
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct UsersFile {
    /// username -> base64-standard-encoded 32-byte Ed25519 public key
    users: HashMap<String, String>,
}

/// Credential store backed by an age-encrypted TOML file.
/// The age decryption key is stored in the OS keyring.
pub struct LocalKeyStore {
    path: PathBuf,
    /// In-memory state (loaded at startup, written on mutation)
    data: UsersFile,
    /// age identity for decrypting/re-encrypting the file
    identity: age::x25519::Identity,
}

impl LocalKeyStore {
    const KEYRING_SERVICE: &'static str = "hyprstream";
    const KEYRING_KEY_NAME: &'static str = "credential-store-key";

    /// Load (or initialize) the credential store.
    /// Creates the file and keyring entry if they don't exist.
    pub fn load(credentials_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(credentials_dir)?;
        let path = credentials_dir.join("users.toml.age");

        let identity = Self::load_or_generate_identity()?;

        let data = if path.exists() {
            Self::decrypt_and_parse(&path, &identity)?
        } else {
            UsersFile::default()
        };

        Ok(Self { path, data, identity })
    }

    fn load_or_generate_identity() -> Result<age::x25519::Identity> {
        // Check for test bypass via config before touching the OS keyring.
        // Set HYPRSTREAM__OAUTH__CREDENTIAL_STORE_KEY=<age-secret-key-...> to inject a key.
        if let Ok(cfg) = crate::config::HyprConfig::load() {
            if let Some(ref age_key) = cfg.oauth.credential_store_key {
                return age_key
                    .trim()
                    .parse::<age::x25519::Identity>()
                    .map_err(|e| anyhow!("HYPRSTREAM__OAUTH__CREDENTIAL_STORE_KEY: invalid age identity: {:?}", e));
            }
        }

        let entry = keyring::Entry::new(Self::KEYRING_SERVICE, Self::KEYRING_KEY_NAME)
            .context("Failed to access keyring for credential store key")?;

        match entry.get_secret() {
            Ok(bytes) => {
                let s = String::from_utf8(bytes)?;
                s.parse::<age::x25519::Identity>()
                    .map_err(|e| anyhow!("Failed to parse credential store key: {:?}", e))
            }
            Err(keyring::Error::NoEntry) => {
                let identity = age::x25519::Identity::generate();
                let secret_str = identity.to_string();
                entry
                    .set_secret(secret_str.expose_secret().as_bytes())
                    .context("Failed to store credential store key in keyring")?;
                tracing::info!("Generated new credential store key (in OS keyring)");
                Ok(identity)
            }
            Err(e) => Err(anyhow!("Keyring error loading credential store key: {}", e)),
        }
    }

    fn decrypt_and_parse(path: &Path, identity: &age::x25519::Identity) -> Result<UsersFile> {
        let ciphertext = std::fs::read(path)?;
        let decryptor = age::Decryptor::new_buffered(&ciphertext[..])
            .map_err(|e| anyhow!("Failed to init decryptor: {:?}", e))?;
        let mut plaintext = vec![];
        let mut reader = decryptor
            .decrypt(std::iter::once(identity as &dyn age::Identity))
            .map_err(|e| anyhow!("Decryption failed: {:?}", e))?;
        reader.read_to_end(&mut plaintext)?;
        toml::from_str(std::str::from_utf8(&plaintext)?).context("Failed to parse users.toml")
    }

    fn encrypt_and_write(&self) -> Result<()> {
        let plaintext =
            toml::to_string_pretty(&self.data).context("Failed to serialize users.toml")?;
        let recipient = self.identity.to_public();
        let encryptor =
            age::Encryptor::with_recipients(std::iter::once(&recipient as &dyn age::Recipient))
                .map_err(|e| anyhow!("Failed to create encryptor: {:?}", e))?;
        let mut ciphertext = vec![];
        let mut writer = encryptor
            .wrap_output(&mut ciphertext)
            .map_err(|e| anyhow!("Failed to init encryption: {:?}", e))?;
        writer.write_all(plaintext.as_bytes())?;
        writer
            .finish()
            .map_err(|e| anyhow!("Failed to finalize encryption: {:?}", e))?;
        std::fs::write(&self.path, ciphertext)?;
        Ok(())
    }
}

impl UserStore for LocalKeyStore {
    fn get_pubkey(&self, username: &str) -> Result<Option<VerifyingKey>> {
        match self.data.users.get(username) {
            None => Ok(None),
            Some(b64) => {
                let raw = STANDARD.decode(b64)?;
                let bytes: [u8; 32] = raw
                    .try_into()
                    .map_err(|_| anyhow!("Stored pubkey for {} is not 32 bytes", username))?;
                Ok(Some(VerifyingKey::from_bytes(&bytes)?))
            }
        }
    }

    fn register(&mut self, username: &str, pubkey: VerifyingKey) -> Result<()> {
        if username.contains(':') {
            anyhow::bail!("Username '{}' must not contain ':'", username);
        }
        let b64 = STANDARD.encode(pubkey.as_bytes());
        if self.data.users.contains_key(username) {
            tracing::warn!(
                "Overwriting existing public key for user '{}' in credential store",
                username
            );
        }
        self.data.users.insert(username.to_owned(), b64);
        self.encrypt_and_write()
    }

    fn remove(&mut self, username: &str) -> Result<bool> {
        let removed = self.data.users.remove(username).is_some();
        if removed {
            self.encrypt_and_write()?;
        }
        Ok(removed)
    }

    fn list_users(&self) -> Vec<String> {
        self.data.users.keys().cloned().collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use tempfile::TempDir;

    // Tests bypass the keyring by constructing LocalKeyStore directly with a generated identity.
    fn make_store_with_identity(dir: &Path, identity: age::x25519::Identity) -> LocalKeyStore {
        LocalKeyStore {
            path: dir.join("users.toml.age"),
            data: UsersFile::default(),
            identity,
        }
    }

    #[test]
    fn test_register_and_get_pubkey() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity);
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let pubkey = signing_key.verifying_key();
        store.register("alice", pubkey)?;
        let retrieved = store
            .get_pubkey("alice")?
            .ok_or_else(|| anyhow!("alice not found"))?;
        assert_eq!(retrieved.as_bytes(), pubkey.as_bytes());
        Ok(())
    }

    #[test]
    fn test_remove_user() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity);
        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        store.register("bob", key)?;
        assert!(store.remove("bob")?);
        assert!(store.get_pubkey("bob")?.is_none());
        Ok(())
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity.clone());
        let signing_key = SigningKey::generate(&mut rand::thread_rng());
        let pubkey = signing_key.verifying_key();
        store.register("carol", pubkey)?;

        // Re-load from disk using the same identity
        let loaded_data =
            LocalKeyStore::decrypt_and_parse(&dir.path().join("users.toml.age"), &identity)?;
        let b64 = loaded_data
            .users
            .get("carol")
            .ok_or_else(|| anyhow!("carol not found in loaded data"))?;
        let raw = STANDARD.decode(b64)?;
        let bytes: [u8; 32] = raw
            .try_into()
            .map_err(|_| anyhow!("wrong key length after roundtrip"))?;
        let loaded_key = VerifyingKey::from_bytes(&bytes)?;
        assert_eq!(loaded_key.as_bytes(), pubkey.as_bytes());
        Ok(())
    }

    #[test]
    fn test_list_users() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity);
        let key1 = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let key2 = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        store.register("alice", key1)?;
        store.register("bob", key2)?;
        let mut users = store.list_users();
        users.sort();
        assert_eq!(users, vec!["alice", "bob"]);
        Ok(())
    }

    #[test]
    fn test_register_rejects_colon_in_username() -> Result<()> {
        let dir = TempDir::new()?;
        let identity = age::x25519::Identity::generate();
        let mut store = make_store_with_identity(dir.path(), identity);
        let key = SigningKey::generate(&mut rand::thread_rng()).verifying_key();
        let result = store.register("bad:user", key);
        assert!(result.is_err(), "register should reject usernames with ':'");
        let err = result.unwrap_err();
        assert!(err.to_string().contains("must not contain"), "error message should mention colon restriction");
        Ok(())
    }
}
