//! Ed25519 cryptographic operations for GitTorrent

use crate::{types::MutableKey, Error, Result};
use ed25519_dalek::{SigningKey, VerifyingKey, Signature, Signer, Verifier};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Ed25519 keypair for GitTorrent mutable keys
#[derive(Debug)]
pub struct Ed25519KeyPair {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
}

/// Serializable key data for storage
#[derive(Debug, Serialize, Deserialize)]
pub struct KeyData {
    pub pub_key: String,
    pub priv_key: String,
}

impl Ed25519KeyPair {
    /// Generate a new random keypair
    pub fn generate() -> Result<Self> {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        Ok(Ed25519KeyPair { signing_key, verifying_key })
    }

    /// Create keypair from existing key data
    pub fn from_key_data(key_data: &KeyData) -> Result<Self> {
        let secret_bytes = hex::decode(&key_data.priv_key)?;
        if secret_bytes.len() != 32 {
            return Err(Error::crypto("Secret key must be 32 bytes"));
        }

        let mut secret_array = [0u8; 32];
        secret_array.copy_from_slice(&secret_bytes);

        let signing_key = SigningKey::from_bytes(&secret_array);
        let verifying_key = signing_key.verifying_key();

        Ok(Ed25519KeyPair { signing_key, verifying_key })
    }

    /// Load keypair from file, creating if it doesn't exist
    pub fn load_or_create<P: AsRef<Path>>(path: P) -> Result<Self> {
        if path.as_ref().exists() {
            Self::load_from_file(path)
        } else {
            let keypair = Self::generate()?;
            keypair.save_to_file(path)?;
            Ok(keypair)
        }
    }

    /// Load keypair from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let key_data: KeyData = serde_json::from_str(&content)?;
        Self::from_key_data(&key_data)
    }

    /// Save keypair to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let key_data = KeyData {
            pub_key: hex::encode(self.verifying_key.as_bytes()),
            priv_key: hex::encode(self.signing_key.as_bytes()),
        };

        let content = serde_json::to_string_pretty(&key_data)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get the public key bytes
    pub fn public_key_bytes(&self) -> &[u8] {
        self.verifying_key.as_bytes()
    }

    /// Get the secret key bytes
    pub fn secret_key_bytes(&self) -> &[u8] {
        self.signing_key.as_bytes()
    }

    /// Get the mutable key (SHA256 of public key)
    pub fn mutable_key(&self) -> Result<MutableKey> {
        let hash = crate::crypto::hash::sha256(self.public_key_bytes());
        MutableKey::new(hex::encode(hash))
    }

    /// Sign data
    pub fn sign(&self, data: &[u8]) -> Vec<u8> {
        let signature = self.signing_key.sign(data);
        signature.to_bytes().to_vec()
    }

    /// Verify signature
    pub fn verify(&self, data: &[u8], signature: &[u8]) -> Result<bool> {
        if signature.len() != 64 {
            return Err(Error::crypto("Signature must be 64 bytes"));
        }

        let mut sig_array = [0u8; 64];
        sig_array.copy_from_slice(signature);
        let signature = Signature::from_bytes(&sig_array);

        Ok(self.verifying_key.verify(data, &signature).is_ok())
    }

    /// Verify signature with a different public key
    pub fn verify_with_public_key(
        public_key: &[u8],
        data: &[u8],
        signature: &[u8],
    ) -> Result<bool> {
        if public_key.len() != 32 {
            return Err(Error::crypto("Public key must be 32 bytes"));
        }
        if signature.len() != 64 {
            return Err(Error::crypto("Signature must be 64 bytes"));
        }

        let mut pk_array = [0u8; 32];
        pk_array.copy_from_slice(public_key);
        let verifying_key = VerifyingKey::from_bytes(&pk_array)
            .map_err(|e| Error::crypto(format!("Invalid public key: {}", e)))?;

        let mut sig_array = [0u8; 64];
        sig_array.copy_from_slice(signature);
        let signature = Signature::from_bytes(&sig_array);

        Ok(verifying_key.verify(data, &signature).is_ok())
    }
}

impl Clone for Ed25519KeyPair {
    fn clone(&self) -> Self {
        // We need to recreate the keypair since ed25519_dalek keys don't implement Clone
        let key_data = KeyData {
            pub_key: hex::encode(self.verifying_key.as_bytes()),
            priv_key: hex::encode(self.signing_key.as_bytes()),
        };
        Self::from_key_data(&key_data).expect("Failed to clone keypair")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_keypair_generation() {
        let keypair = Ed25519KeyPair::generate().unwrap();
        assert_eq!(keypair.public_key_bytes().len(), 32);
        assert_eq!(keypair.secret_key_bytes().len(), 32);
    }

    #[test]
    fn test_sign_verify() {
        let keypair = Ed25519KeyPair::generate().unwrap();
        let data = b"hello world";

        let signature = keypair.sign(data);
        assert!(keypair.verify(data, &signature).unwrap());

        // Test with different data
        let bad_data = b"goodbye world";
        assert!(!keypair.verify(bad_data, &signature).unwrap());
    }

    #[test]
    fn test_mutable_key() {
        let keypair = Ed25519KeyPair::generate().unwrap();
        let mutable_key = keypair.mutable_key().unwrap();
        assert_eq!(mutable_key.as_str().len(), 64); // SHA256 hex string
    }

    #[test]
    fn test_save_load() {
        let keypair1 = Ed25519KeyPair::generate().unwrap();
        let temp_file = NamedTempFile::new().unwrap();

        // Save keypair
        keypair1.save_to_file(temp_file.path()).unwrap();

        // Load keypair
        let keypair2 = Ed25519KeyPair::load_from_file(temp_file.path()).unwrap();

        // Should have same keys
        assert_eq!(keypair1.public_key_bytes(), keypair2.public_key_bytes());
        assert_eq!(keypair1.secret_key_bytes(), keypair2.secret_key_bytes());

        // Should produce same signatures
        let data = b"test data";
        let sig1 = keypair1.sign(data);
        let sig2 = keypair2.sign(data);

        assert!(keypair1.verify(data, &sig2).unwrap());
        assert!(keypair2.verify(data, &sig1).unwrap());
    }

    #[test]
    fn test_load_or_create() {
        let temp_dir = tempfile::tempdir().unwrap();
        let key_path = temp_dir.path().join("test_key.json");

        // Should create new keypair since file doesn't exist
        let keypair1 = Ed25519KeyPair::load_or_create(&key_path).unwrap();

        // Should load existing keypair since file now exists
        let keypair2 = Ed25519KeyPair::load_or_create(&key_path).unwrap();

        assert_eq!(keypair1.public_key_bytes(), keypair2.public_key_bytes());
    }
}