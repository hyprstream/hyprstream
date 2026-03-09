//! Encrypted per-session history store.
//!
//! # Wire format (per session file)
//!
//! ```text
//! nonce(12B, OsRng) || AES-256-GCM(plaintext=serde_json(entries), aad=session_uuid.as_bytes())
//! ```
//!
//! - **Nonce**: 12 fresh random bytes from `OsRng` on every write. Never reused.
//! - **AAD**: the 16-byte session UUID binds the ciphertext to its filename,
//!   preventing file-swap attacks.
//! - **Key**: `Zeroizing<[u8; 32]>` derived by the caller (native: from signing
//!   key seed via `derive_key`; browser: session-scoped Web Crypto key).
//! - **Writes**: atomic — `{uuid}.enc.tmp` → `fs::rename` → `{uuid}.enc`.

use std::collections::HashMap;
use std::path::PathBuf;

#[cfg(not(target_os = "wasi"))]
use parking_lot::Mutex;
#[cfg(target_os = "wasi")]
#[allow(clippy::disallowed_types)]
use std::sync::Mutex;

use aes_gcm::{
    aead::{Aead, KeyInit, Payload},
    Aes256Gcm, Nonce,
};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use zeroize::Zeroizing;

// ============================================================================
// StorageKey
// ============================================================================

/// 32-byte AES-256 key, zeroed on drop.
pub type StorageKey = Zeroizing<[u8; 32]>;

// ============================================================================
// StorageBackend
// ============================================================================

/// Backend abstraction for raw byte reads and atomic writes.
pub trait StorageBackend: Send + Sync {
    /// Atomically write `data` to the file for `session_id`.
    fn write(&self, session_id: &Uuid, data: &[u8]) -> std::io::Result<()>;
    /// Read the raw bytes for `session_id`. Returns `None` if no file exists.
    fn read(&self, session_id: &Uuid) -> std::io::Result<Option<Vec<u8>>>;
}

// ============================================================================
// FsBackend
// ============================================================================

/// Filesystem backend.
///
/// Native path: `$XDG_DATA_HOME/hyprstream/private/v1/`
/// WASM path:   `/private/v1/` (requires OPFS preopen — Phase W1)
pub struct FsBackend {
    dir: PathBuf,
}

impl FsBackend {
    /// Create the backend and ensure the directory exists.
    /// Returns an error (not a fallback) if the directory cannot be created.
    pub fn new(dir: PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }
}

impl StorageBackend for FsBackend {
    fn write(&self, session_id: &Uuid, data: &[u8]) -> std::io::Result<()> {
        let tmp = self.dir.join(format!("{}.enc.tmp", session_id));
        let final_path = self.dir.join(format!("{}.enc", session_id));
        std::fs::write(&tmp, data)?;
        std::fs::rename(&tmp, &final_path)
    }

    fn read(&self, session_id: &Uuid) -> std::io::Result<Option<Vec<u8>>> {
        let path = self.dir.join(format!("{}.enc", session_id));
        match std::fs::read(&path) {
            Ok(data) => Ok(Some(data)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }
}

// ============================================================================
// MemoryBackend
// ============================================================================

/// In-memory backend for tests and ephemeral sessions.
#[derive(Default)]
pub struct MemoryBackend {
    store: Mutex<HashMap<Uuid, Vec<u8>>>,
}

impl MemoryBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

impl StorageBackend for MemoryBackend {
    fn write(&self, session_id: &Uuid, data: &[u8]) -> std::io::Result<()> {
        store_lock(&self.store).insert(*session_id, data.to_vec());
        Ok(())
    }

    fn read(&self, session_id: &Uuid) -> std::io::Result<Option<Vec<u8>>> {
        Ok(store_lock(&self.store).get(session_id).cloned())
    }
}

/// Acquire the store lock in a cfg-portable way.
/// - `parking_lot::Mutex::lock()` returns `MutexGuard<T>` directly.
/// - `std::sync::Mutex::lock()` returns `Result<MutexGuard<T>, PoisonError>`.
fn store_lock(m: &Mutex<HashMap<Uuid, Vec<u8>>>) -> impl std::ops::DerefMut<Target = HashMap<Uuid, Vec<u8>>> + '_ {
    #[cfg(not(target_os = "wasi"))]
    { m.lock() }
    #[cfg(target_os = "wasi")]
    { m.lock().unwrap_or_else(|e| e.into_inner()) }
}

// ============================================================================
// PrivateStore
// ============================================================================

/// Encrypted per-session history store.
///
/// Generic over `StorageBackend` so the same code works with `FsBackend`
/// (native/WASM) and `MemoryBackend` (tests).
pub struct PrivateStore<B: StorageBackend> {
    backend: B,
    key: StorageKey,
}

impl<B: StorageBackend> PrivateStore<B> {
    pub fn new(backend: B, key: StorageKey) -> Self {
        Self { backend, key }
    }

    /// Encrypt `entries` and atomically write to storage.
    ///
    /// A fresh 12-byte nonce is generated from `OsRng` on every call.
    pub fn save<T: Serialize>(
        &self,
        session_id: &Uuid,
        entries: &[T],
    ) -> anyhow::Result<()> {
        let plaintext = serde_json::to_vec(entries)?;
        let aad = session_id.as_bytes();

        let mut nonce_bytes = [0u8; 12];
        rand::rngs::OsRng.fill_bytes(&mut nonce_bytes);

        let cipher = Aes256Gcm::new(self.key.as_ref().into());
        let ciphertext = cipher
            .encrypt(
                Nonce::from_slice(&nonce_bytes),
                Payload { msg: &plaintext, aad },
            )
            .map_err(|e| anyhow::anyhow!("AES-GCM encrypt: {}", e))?;

        let mut file_data = Vec::with_capacity(12 + ciphertext.len());
        file_data.extend_from_slice(&nonce_bytes);
        file_data.extend_from_slice(&ciphertext);

        self.backend.write(session_id, &file_data)?;
        Ok(())
    }

    /// Load and decrypt entries for `session_id`.
    /// Returns `None` if no history file exists yet.
    pub fn load<T: for<'de> Deserialize<'de>>(
        &self,
        session_id: &Uuid,
    ) -> anyhow::Result<Option<Vec<T>>> {
        let Some(file_data) = self.backend.read(session_id)? else {
            return Ok(None);
        };
        if file_data.len() < 12 {
            return Err(anyhow::anyhow!(
                "private history file too short ({} bytes)",
                file_data.len()
            ));
        }
        let (nonce_bytes, ciphertext) = file_data.split_at(12);
        let aad = session_id.as_bytes();

        let cipher = Aes256Gcm::new(self.key.as_ref().into());
        let plaintext = cipher
            .decrypt(
                Nonce::from_slice(nonce_bytes),
                Payload { msg: ciphertext, aad },
            )
            .map_err(|e| anyhow::anyhow!("AES-GCM decrypt: {}", e))?;

        let entries: Vec<T> = serde_json::from_slice(&plaintext)?;
        Ok(Some(entries))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key() -> StorageKey {
        Zeroizing::new([0x42u8; 32])
    }

    #[test]
    fn round_trip_memory_backend() {
        let store = PrivateStore::new(MemoryBackend::new(), test_key());
        let session = Uuid::new_v4();
        let entries = vec!["hello", "world"];
        store.save(&session, &entries).unwrap();
        let loaded: Vec<String> = store.load(&session).unwrap().unwrap();
        assert_eq!(loaded, entries);
    }

    #[test]
    fn missing_session_returns_none() {
        let store = PrivateStore::new(MemoryBackend::new(), test_key());
        let session = Uuid::new_v4();
        let loaded: Option<Vec<String>> = store.load(&session).unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn wrong_key_fails_decrypt() {
        let store1 = PrivateStore::new(MemoryBackend::new(), test_key());
        let session = Uuid::new_v4();
        store1.save(&session, &["secret"]).unwrap();

        // Extract the ciphertext bytes from the MemoryBackend by saving with store1.
        let raw = store1.backend.read(&session).unwrap().unwrap();

        // Create a second store with a different key sharing the same backend.
        let wrong_key = Zeroizing::new([0x00u8; 32]);
        let store2 = PrivateStore {
            backend: MemoryBackend::new(),
            key: wrong_key,
        };
        // Manually write the ciphertext into store2's backend.
        store2.backend.write(&session, &raw).unwrap();

        let result: anyhow::Result<Option<Vec<String>>> = store2.load(&session);
        assert!(result.is_err(), "wrong key should fail decryption");
    }

    #[test]
    fn aad_binding_prevents_session_swap() {
        let backend = MemoryBackend::new();
        let key = test_key();

        let session_a = Uuid::new_v4();
        let session_b = Uuid::new_v4();

        let store = PrivateStore::new(MemoryBackend::new(), test_key());
        store.save(&session_a, &["data for A"]).unwrap();

        // Extract A's ciphertext and plant it under session_b's key.
        let raw_a = store.backend.read(&session_a).unwrap().unwrap();
        let store2 = PrivateStore { backend, key };
        store2.backend.write(&session_b, &raw_a).unwrap();

        let result: anyhow::Result<Option<Vec<String>>> = store2.load(&session_b);
        assert!(result.is_err(), "AAD mismatch should fail decryption");
    }
}
