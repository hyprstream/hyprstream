//! Cryptographic backend abstraction for KDF and MAC operations.
//!
//! Provides unified API for key derivation and message authentication,
//! with compile-time selection between:
//! - **Default**: Blake3 (fast, modern, ~10+ GB/s with SIMD)
//! - **FIPS mode**: HKDF-SHA256 + HMAC-SHA256 (NIST-approved)
//!
//! # Feature Flags
//!
//! - Default: Blake3 `derive_key()` and `keyed_hash()`
//! - `fips`: HKDF-SHA256 (SP 800-56C) and HMAC-SHA256 (FIPS 198-1)

// ============================================================================
// Key Derivation Function (KDF)
// ============================================================================

/// Derive a 32-byte key from input material with domain separation.
///
/// # Arguments
///
/// * `context` - Static domain separation string (must be compile-time constant)
/// * `ikm` - Input key material (e.g., shared_secret || salt)
///
/// # Returns
///
/// 32-byte derived key.
///
/// # Backend Behavior
///
/// - **Blake3**: Uses `blake3::derive_key(context, ikm)`
/// - **FIPS**: Uses `HKDF-Expand(HKDF-Extract(ikm), context)`
#[cfg(not(feature = "fips"))]
pub fn derive_key(context: &str, ikm: &[u8]) -> [u8; 32] {
    blake3::derive_key(context, ikm)
}

#[cfg(feature = "fips")]
pub fn derive_key(context: &str, ikm: &[u8]) -> [u8; 32] {
    use hkdf::Hkdf;
    use sha2::Sha256;

    // HKDF-Extract with no salt (salt = None means use zeros)
    // Then HKDF-Expand with context as info
    let hk = Hkdf::<Sha256>::new(None, ikm);
    let mut output = [0u8; 32];
    hk.expand(context.as_bytes(), &mut output)
        .expect("32 bytes is valid HKDF-SHA256 output length");
    output
}

// ============================================================================
// Message Authentication Code (MAC)
// ============================================================================

/// Compute a 32-byte keyed MAC.
///
/// # Arguments
///
/// * `key` - 32-byte MAC key
/// * `data` - Data to authenticate
///
/// # Returns
///
/// 32-byte MAC tag.
///
/// # Backend Behavior
///
/// - **Blake3**: Uses `blake3::keyed_hash(key, data)`
/// - **FIPS**: Uses `HMAC-SHA256(key, data)`
#[cfg(not(feature = "fips"))]
pub fn keyed_mac(key: &[u8; 32], data: &[u8]) -> [u8; 32] {
    *blake3::keyed_hash(key, data).as_bytes()
}

#[cfg(feature = "fips")]
pub fn keyed_mac(key: &[u8; 32], data: &[u8]) -> [u8; 32] {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    let mut mac = Hmac::<Sha256>::new_from_slice(key)
        .expect("HMAC-SHA256 accepts any key size");
    mac.update(data);
    let result = mac.finalize();
    let mut output = [0u8; 32];
    output.copy_from_slice(&result.into_bytes());
    output
}

/// Compute a 16-byte truncated keyed MAC (for wire format).
///
/// Used by streaming protocol where 16-byte MACs reduce overhead
/// while still providing 128-bit security (adequate for stream auth).
///
/// # Arguments
///
/// * `key` - 32-byte MAC key
/// * `data` - Data to authenticate
///
/// # Returns
///
/// 16-byte truncated MAC tag.
pub fn keyed_mac_truncated(key: &[u8; 32], data: &[u8]) -> [u8; 16] {
    let full = keyed_mac(key, data);
    let mut output = [0u8; 16];
    output.copy_from_slice(&full[..16]);
    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_key_deterministic() {
        let ikm = [0x42u8; 32];
        let key1 = derive_key("test context", &ikm);
        let key2 = derive_key("test context", &ikm);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_derive_key_different_contexts() {
        let ikm = [0x42u8; 32];
        let key1 = derive_key("context a", &ikm);
        let key2 = derive_key("context b", &ikm);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_derive_key_different_ikm() {
        let key1 = derive_key("test context", &[0x01u8; 32]);
        let key2 = derive_key("test context", &[0x02u8; 32]);
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_keyed_mac_deterministic() {
        let key = [0x42u8; 32];
        let data = b"test data";
        let mac1 = keyed_mac(&key, data);
        let mac2 = keyed_mac(&key, data);
        assert_eq!(mac1, mac2);
    }

    #[test]
    fn test_keyed_mac_different_keys() {
        let data = b"test data";
        let mac1 = keyed_mac(&[0x01u8; 32], data);
        let mac2 = keyed_mac(&[0x02u8; 32], data);
        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_keyed_mac_different_data() {
        let key = [0x42u8; 32];
        let mac1 = keyed_mac(&key, b"data a");
        let mac2 = keyed_mac(&key, b"data b");
        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_keyed_mac_truncated() {
        let key = [0x42u8; 32];
        let data = b"test data";
        let full = keyed_mac(&key, data);
        let truncated = keyed_mac_truncated(&key, data);
        assert_eq!(&full[..16], &truncated[..]);
    }
}
