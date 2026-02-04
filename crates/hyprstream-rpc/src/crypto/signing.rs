//! Ed25519 digital signatures for request envelope authentication.
//!
//! All ZMQ messages are signed to provide:
//! - Authentication: Proves who created the request
//! - Integrity: Detects tampering
//! - Non-repudiation: Signature survives message forwarding
//!
//! # Signing Flow
//!
//! 1. API server validates bearer token or OS user
//! 2. Creates `RequestIdentity` with user info
//! 3. Signs `(request_id || identity || payload)` with server's Ed25519 key
//! 4. ZMQ service verifies signature before processing

use ed25519_dalek::{Signature, Signer, Verifier};
use sha2::{Digest, Sha512};
use zeroize::Zeroizing;

use crate::error::{EnvelopeError, EnvelopeResult};

// Re-export key types
pub use ed25519_dalek::{SigningKey, VerifyingKey};

/// Sign a message using Ed25519.
///
/// Uses streaming hash to avoid allocating the full message.
///
/// # Arguments
///
/// * `signing_key` - The Ed25519 signing key
/// * `request_id` - Unique request ID (included in signed data)
/// * `identity_bytes` - Canonical serialization of RequestIdentity
/// * `payload` - The request payload
///
/// # Returns
///
/// 64-byte Ed25519 signature
pub fn sign_message(
    signing_key: &SigningKey,
    request_id: u64,
    identity_bytes: &[u8],
    payload: &[u8],
) -> [u8; 64] {
    // Build the message to sign using streaming hash
    let mut hasher = Sha512::new();
    hasher.update(request_id.to_le_bytes());
    hasher.update(identity_bytes);
    hasher.update(payload);
    let hash = hasher.finalize();

    // Sign the hash
    // Note: ed25519-dalek internally hashes again, but we pre-hash for
    // consistency with the streaming approach. This is effectively Ed25519ph.
    let signature = signing_key.sign(&hash);
    signature.to_bytes()
}

/// Verify an Ed25519 signature.
///
/// # Arguments
///
/// * `verifying_key` - The Ed25519 public key
/// * `signature` - The 64-byte signature to verify
/// * `request_id` - Request ID that was signed
/// * `identity_bytes` - Canonical serialization of RequestIdentity
/// * `payload` - The request payload
///
/// # Errors
///
/// Returns `EnvelopeError::InvalidSignature` if verification fails.
pub fn verify_message(
    verifying_key: &VerifyingKey,
    signature: &[u8; 64],
    request_id: u64,
    identity_bytes: &[u8],
    payload: &[u8],
) -> EnvelopeResult<()> {
    // Reconstruct the hash that was signed
    let mut hasher = Sha512::new();
    hasher.update(request_id.to_le_bytes());
    hasher.update(identity_bytes);
    hasher.update(payload);
    let hash = hasher.finalize();

    // Verify
    let sig = Signature::from_bytes(signature);
    verifying_key.verify(&hash, &sig)?;
    Ok(())
}

/// Generate a new Ed25519 signing keypair.
///
/// Uses the system's secure random number generator.
pub fn generate_signing_keypair() -> (SigningKey, VerifyingKey) {
    let signing_key = SigningKey::generate(&mut rand::thread_rng());
    let verifying_key = signing_key.verifying_key();
    (signing_key, verifying_key)
}

/// Load a signing key from 32 secret bytes.
///
/// # Security
///
/// The input bytes are wrapped in `Zeroizing` to ensure they are
/// securely erased from memory when dropped.
pub fn signing_key_from_bytes(bytes: &[u8; 32]) -> SigningKey {
    let secret = Zeroizing::new(*bytes);
    SigningKey::from_bytes(&secret)
}

/// Load a verifying key from 32 public key bytes.
///
/// # Errors
///
/// Returns `EnvelopeError::InvalidPublicKey` if the bytes are not a valid
/// Ed25519 public key.
pub fn verifying_key_from_bytes(bytes: &[u8; 32]) -> EnvelopeResult<VerifyingKey> {
    VerifyingKey::from_bytes(bytes).map_err(EnvelopeError::InvalidSignature)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign_verify_roundtrip() -> crate::EnvelopeResult<()> {
        let (signing_key, verifying_key) = generate_signing_keypair();

        let request_id = 12345u64;
        let identity_bytes = b"local:alice";
        let payload = b"test payload data";

        let signature = sign_message(&signing_key, request_id, identity_bytes, payload);

        // Should verify successfully
        verify_message(&verifying_key, &signature, request_id, identity_bytes, payload)?;
        Ok(())
    }

    #[test]
    fn test_tampered_payload_fails() {
        let (signing_key, verifying_key) = generate_signing_keypair();

        let request_id = 12345u64;
        let identity_bytes = b"local:alice";
        let payload = b"original payload";

        let signature = sign_message(&signing_key, request_id, identity_bytes, payload);

        // Tampered payload should fail
        let tampered = b"tampered payload";
        let result = verify_message(&verifying_key, &signature, request_id, identity_bytes, tampered);
        assert!(result.is_err());
    }

    #[test]
    fn test_tampered_request_id_fails() {
        let (signing_key, verifying_key) = generate_signing_keypair();

        let request_id = 12345u64;
        let identity_bytes = b"local:alice";
        let payload = b"test payload";

        let signature = sign_message(&signing_key, request_id, identity_bytes, payload);

        // Different request_id should fail
        let result = verify_message(&verifying_key, &signature, 99999, identity_bytes, payload);
        assert!(result.is_err());
    }

    #[test]
    fn test_wrong_key_fails() {
        let (signing_key, _) = generate_signing_keypair();
        let (_, wrong_verifying_key) = generate_signing_keypair();

        let request_id = 12345u64;
        let identity_bytes = b"local:alice";
        let payload = b"test payload";

        let signature = sign_message(&signing_key, request_id, identity_bytes, payload);

        // Wrong key should fail
        let result = verify_message(&wrong_verifying_key, &signature, request_id, identity_bytes, payload);
        assert!(result.is_err());
    }

    #[test]
    fn test_key_serialization_roundtrip() -> crate::EnvelopeResult<()> {
        let (signing_key, verifying_key) = generate_signing_keypair();

        // Serialize and deserialize signing key
        let signing_bytes = signing_key.to_bytes();
        let restored_signing = signing_key_from_bytes(&signing_bytes);
        assert_eq!(signing_key.to_bytes(), restored_signing.to_bytes());

        // Serialize and deserialize verifying key
        let verifying_bytes = verifying_key.to_bytes();
        let restored_verifying = verifying_key_from_bytes(&verifying_bytes)?;
        assert_eq!(verifying_key.to_bytes(), restored_verifying.to_bytes());
        Ok(())
    }
}
