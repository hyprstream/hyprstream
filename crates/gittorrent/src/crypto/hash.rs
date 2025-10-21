//! Hashing utilities for GitTorrent

use crate::types::Sha256Hash;
use sha2::{Digest, Sha256};

/// Calculate SHA256 hash of data
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Calculate SHA256 hash and return as hex string
pub fn sha256_hex(data: &[u8]) -> String {
    hex::encode(sha256(data))
}

/// Calculate SHA256 hash and return as Sha256Hash type
pub fn sha256_git(data: &[u8]) -> crate::Result<Sha256Hash> {
    let hash = sha256_hex(data);
    Sha256Hash::new(hash)
}

/// Calculate SHA256 hash of data and return as Sha256Hash (alias for sha256_git)
pub fn sha256_data(data: &[u8]) -> crate::Result<Sha256Hash> {
    sha256_git(data)
}


/// Hash a string to create an identifier
pub fn hash_string(s: &str) -> String {
    sha256_hex(s.as_bytes())
}

/// Verify that a given SHA256 matches the data
pub fn verify_sha256(data: &[u8], expected: &Sha256Hash) -> crate::Result<bool> {
    let actual = sha256_git(data)?;
    Ok(actual == *expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_basic() {
        let data = b"hello world";
        let hash = sha256(data);
        assert_eq!(hash.len(), 32);

        let hex = sha256_hex(data);
        assert_eq!(hex.len(), 64);
        assert_eq!(hex, "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9");
    }

    #[test]
    fn test_sha256_git() {
        let data = b"test data";
        let sha256 = sha256_git(data).unwrap();
        assert_eq!(sha256.as_str().len(), 64);
    }

    #[test]
    fn test_verify_sha256() {
        let data = b"verify me";
        let sha256 = sha256_git(data).unwrap();
        assert!(verify_sha256(data, &sha256).unwrap());

        let bad_data = b"different data";
        assert!(!verify_sha256(bad_data, &sha256).unwrap());
    }

    #[test]
    fn test_hash_string() {
        let s = "test string";
        let hash = hash_string(s);
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }
}