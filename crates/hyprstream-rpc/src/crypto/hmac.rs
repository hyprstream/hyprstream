//! Chained HMAC-SHA256 for streaming response authentication.
//!
//! Streaming responses (XPUB/XSUB) use HMAC instead of per-token Ed25519
//! signatures for performance reasons:
//! - Ed25519: ~10k signatures/sec
//! - HMAC-SHA256: millions of MACs/sec
//!
//! # Chained HMAC Design
//!
//! Instead of explicit sequence numbers, we use chained HMACs where each
//! chunk's HMAC depends on the previous chunk's HMAC:
//!
//! ```text
//! mac_0 = HMAC(key, request_id_bytes || data_0)  // First chunk: prev = request_id
//! mac_n = HMAC(key, mac_{n-1} || data_n)         // Subsequent chunks
//! ```
//!
//! # Security Properties
//!
//! - Authenticates server as holder of the DH shared secret
//! - Chained HMAC provides cryptographic ordering (no separate sequence field)
//! - Reordering is impossible - can't verify chunk N without mac_{N-1}
//! - Request ID binds chunks to their request
//! - TCP/ZMQ provides transport-level ordering (defense in depth)

use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;

use crate::error::{EnvelopeError, EnvelopeResult};

type HmacSha256 = Hmac<Sha256>;

/// HMAC key derived from DH shared secret.
#[derive(Clone)]
pub struct HmacKey([u8; 32]);

impl HmacKey {
    /// Create from raw bytes.
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the raw bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl Drop for HmacKey {
    fn drop(&mut self) {
        // Zeroize on drop
        self.0.iter_mut().for_each(|b| *b = 0);
    }
}

/// Chained stream HMAC context for authenticating streaming responses.
///
/// Each chunk's HMAC depends on the previous chunk's HMAC, providing
/// cryptographic ordering without explicit sequence numbers.
///
/// # Example
///
/// ```ignore
/// // Server side: create producer
/// let mut producer = ChainedStreamHmac::new_producer(hmac_key, request_id);
/// for chunk in chunks {
///     let mac = producer.compute_next(&chunk);
///     send(StreamChunk { request_id, data: chunk, hmac: mac });
/// }
///
/// // Client side: create verifier
/// let mut verifier = ChainedStreamHmac::new_verifier(hmac_key, request_id);
/// for chunk in received_chunks {
///     verifier.verify_next(&chunk.data, &chunk.hmac)?;
/// }
/// ```
pub struct ChainedStreamHmac {
    key: HmacKey,
    /// Previous MAC (or request_id bytes for first chunk)
    prev_mac: [u8; 32],
}

impl ChainedStreamHmac {
    /// Create a new chained HMAC producer/verifier.
    ///
    /// The initial "previous MAC" is the request_id as bytes (padded to 32 bytes).
    /// This ensures the first chunk is bound to the request.
    pub fn new(key: HmacKey, request_id: u64) -> Self {
        // Initialize prev_mac with request_id bytes (zero-padded)
        let mut prev_mac = [0u8; 32];
        prev_mac[..8].copy_from_slice(&request_id.to_le_bytes());
        Self { key, prev_mac }
    }

    /// Create from raw key bytes.
    pub fn from_bytes(key_bytes: [u8; 32], request_id: u64) -> Self {
        Self::new(HmacKey::new(key_bytes), request_id)
    }

    /// Compute HMAC for the next chunk in the stream.
    ///
    /// The MAC covers: `prev_mac || data`
    ///
    /// After computing, updates internal state to the new MAC.
    ///
    /// # Arguments
    ///
    /// * `data` - The chunk data
    ///
    /// # Returns
    ///
    /// 32-byte HMAC-SHA256 tag
    pub fn compute_next(&mut self, data: &[u8]) -> [u8; 32] {
        // SAFETY: Per RFC 2104, HMAC accepts keys of any size (keys > block size are hashed first)
        // HmacSha256::new_from_slice only fails for InvalidLength, which can't happen with any &[u8]
        let mut mac = HmacSha256::new_from_slice(self.key.as_bytes())
            .unwrap_or_else(|_| HmacSha256::new_from_slice(&[0u8; 32]).unwrap());

        // Chain: HMAC(key, prev_mac || data)
        mac.update(&self.prev_mac);
        mac.update(data);

        let result = mac.finalize();
        let mut output = [0u8; 32];
        output.copy_from_slice(&result.into_bytes());

        // Update chain state
        self.prev_mac = output;

        output
    }

    /// Verify HMAC for the next chunk in the stream.
    ///
    /// Uses constant-time comparison to prevent timing attacks.
    /// After verifying, updates internal state to the verified MAC.
    ///
    /// # Arguments
    ///
    /// * `data` - The chunk data
    /// * `expected_mac` - The MAC to verify against
    ///
    /// # Errors
    ///
    /// Returns `EnvelopeError::InvalidHmac` if verification fails.
    pub fn verify_next(&mut self, data: &[u8], expected_mac: &[u8; 32]) -> EnvelopeResult<()> {
        let computed = self.compute_next_peek(data);

        // Constant-time comparison to prevent timing attacks
        if computed.ct_eq(expected_mac).into() {
            // Update chain state only on success
            self.prev_mac = *expected_mac;
            Ok(())
        } else {
            Err(EnvelopeError::InvalidHmac)
        }
    }

    /// Compute the MAC for next chunk without updating state.
    ///
    /// Useful for verification where we don't want to update state on failure.
    fn compute_next_peek(&self, data: &[u8]) -> [u8; 32] {
        // SAFETY: Per RFC 2104, HMAC accepts keys of any size
        let mut mac = HmacSha256::new_from_slice(self.key.as_bytes())
            .unwrap_or_else(|_| HmacSha256::new_from_slice(&[0u8; 32]).unwrap());

        mac.update(&self.prev_mac);
        mac.update(data);

        let result = mac.finalize();
        let mut output = [0u8; 32];
        output.copy_from_slice(&result.into_bytes());
        output
    }

    /// Get the current chain state (previous MAC).
    ///
    /// This can be used to resume a stream if the state is persisted.
    pub fn chain_state(&self) -> &[u8; 32] {
        &self.prev_mac
    }
}

/// Legacy sequence-based HMAC (kept for compatibility).
///
/// Prefer `ChainedStreamHmac` for new code.
pub struct StreamHmac {
    key: HmacKey,
}

impl StreamHmac {
    /// Create a new StreamHmac with the given key.
    pub fn new(key: HmacKey) -> Self {
        Self { key }
    }

    /// Create from raw key bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self::new(HmacKey::new(bytes))
    }

    /// Compute HMAC for a stream chunk (sequence-based, legacy).
    pub fn compute(&self, request_id: u64, sequence: u64, data: &[u8]) -> [u8; 32] {
        // SAFETY: Per RFC 2104, HMAC accepts keys of any size
        let mut mac = HmacSha256::new_from_slice(self.key.as_bytes())
            .unwrap_or_else(|_| HmacSha256::new_from_slice(&[0u8; 32]).unwrap());

        mac.update(&request_id.to_le_bytes());
        mac.update(&sequence.to_le_bytes());
        mac.update(data);

        let result = mac.finalize();
        let mut output = [0u8; 32];
        output.copy_from_slice(&result.into_bytes());
        output
    }

    /// Verify HMAC for a stream chunk (sequence-based, legacy).
    pub fn verify(
        &self,
        request_id: u64,
        sequence: u64,
        data: &[u8],
        expected_mac: &[u8; 32],
    ) -> EnvelopeResult<()> {
        let computed = self.compute(request_id, sequence, data);

        if computed.ct_eq(expected_mac).into() {
            Ok(())
        } else {
            Err(EnvelopeError::InvalidHmac)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Chained HMAC tests
    // =========================================================================

    #[test]
    fn test_chained_hmac_compute_verify() {
        let key = [0x42u8; 32];
        let request_id = 12345u64;

        // Producer side
        let mut producer = ChainedStreamHmac::from_bytes(key, request_id);
        let mac1 = producer.compute_next(b"chunk 1");
        let mac2 = producer.compute_next(b"chunk 2");
        let mac3 = producer.compute_next(b"chunk 3");

        // Verifier side
        let mut verifier = ChainedStreamHmac::from_bytes(key, request_id);
        verifier.verify_next(b"chunk 1", &mac1).unwrap();
        verifier.verify_next(b"chunk 2", &mac2).unwrap();
        verifier.verify_next(b"chunk 3", &mac3).unwrap();
    }

    #[test]
    fn test_chained_hmac_tampered_data_fails() {
        let key = [0x42u8; 32];
        let request_id = 12345u64;

        let mut producer = ChainedStreamHmac::from_bytes(key, request_id);
        let mac = producer.compute_next(b"original data");

        let mut verifier = ChainedStreamHmac::from_bytes(key, request_id);
        let result = verifier.verify_next(b"tampered data", &mac);
        assert!(matches!(result, Err(EnvelopeError::InvalidHmac)));
    }

    #[test]
    fn test_chained_hmac_reordering_fails() {
        let key = [0x42u8; 32];
        let request_id = 12345u64;

        // Produce chunks in order
        let mut producer = ChainedStreamHmac::from_bytes(key, request_id);
        let mac1 = producer.compute_next(b"chunk 1");
        let mac2 = producer.compute_next(b"chunk 2");

        // Try to verify in wrong order - should fail
        let mut verifier = ChainedStreamHmac::from_bytes(key, request_id);

        // Skip chunk 1 and try to verify chunk 2 directly
        let result = verifier.verify_next(b"chunk 2", &mac2);
        assert!(matches!(result, Err(EnvelopeError::InvalidHmac)));

        // Now verify chunk 1 (should work)
        verifier.verify_next(b"chunk 1", &mac1).unwrap();

        // Now chunk 2 should work
        verifier.verify_next(b"chunk 2", &mac2).unwrap();
    }

    #[test]
    fn test_chained_hmac_different_request_ids() {
        let key = [0x42u8; 32];

        // Same key, different request IDs
        let mut producer1 = ChainedStreamHmac::from_bytes(key, 1);
        let mut producer2 = ChainedStreamHmac::from_bytes(key, 2);

        let mac1 = producer1.compute_next(b"same data");
        let mac2 = producer2.compute_next(b"same data");

        // Different request IDs should produce different MACs
        assert_ne!(mac1, mac2);

        // Cross-verification should fail
        let mut verifier1 = ChainedStreamHmac::from_bytes(key, 1);
        let result = verifier1.verify_next(b"same data", &mac2);
        assert!(matches!(result, Err(EnvelopeError::InvalidHmac)));
    }

    #[test]
    fn test_chained_hmac_different_keys() {
        let request_id = 12345u64;

        let mut producer1 = ChainedStreamHmac::from_bytes([0x01u8; 32], request_id);
        let mut producer2 = ChainedStreamHmac::from_bytes([0x02u8; 32], request_id);

        let mac1 = producer1.compute_next(b"same data");
        let mac2 = producer2.compute_next(b"same data");

        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_chained_hmac_chain_state() {
        let key = [0x42u8; 32];
        let request_id = 12345u64;

        let mut hmac = ChainedStreamHmac::from_bytes(key, request_id);

        // Initial state should be request_id padded to 32 bytes
        let initial_state = hmac.chain_state();
        assert_eq!(&initial_state[..8], &request_id.to_le_bytes());
        assert!(initial_state[8..].iter().all(|&b| b == 0));

        // After computing, state should change
        let mac = hmac.compute_next(b"data");
        assert_eq!(hmac.chain_state(), &mac);
    }

    // =========================================================================
    // Legacy sequence-based HMAC tests (for compatibility)
    // =========================================================================

    #[test]
    fn test_legacy_hmac_compute_verify() {
        let key = [0x42u8; 32];
        let hmac = StreamHmac::from_bytes(key);

        let request_id = 12345u64;
        let sequence = 1u64;
        let data = b"test chunk data";

        let mac = hmac.compute(request_id, sequence, data);
        hmac.verify(request_id, sequence, data, &mac).unwrap();
    }

    #[test]
    fn test_legacy_hmac_tampered_data_fails() {
        let key = [0x42u8; 32];
        let hmac = StreamHmac::from_bytes(key);

        let request_id = 12345u64;
        let sequence = 1u64;
        let data = b"original data";
        let tampered = b"tampered data";

        let mac = hmac.compute(request_id, sequence, data);
        let result = hmac.verify(request_id, sequence, tampered, &mac);
        assert!(matches!(result, Err(EnvelopeError::InvalidHmac)));
    }

    #[test]
    fn test_legacy_hmac_wrong_sequence_fails() {
        let key = [0x42u8; 32];
        let hmac = StreamHmac::from_bytes(key);

        let request_id = 12345u64;
        let data = b"test data";

        let mac = hmac.compute(request_id, 1, data);
        let result = hmac.verify(request_id, 2, data, &mac);
        assert!(matches!(result, Err(EnvelopeError::InvalidHmac)));
    }

    #[test]
    fn test_legacy_different_keys_different_macs() {
        let hmac1 = StreamHmac::from_bytes([0x01u8; 32]);
        let hmac2 = StreamHmac::from_bytes([0x02u8; 32]);

        let request_id = 12345u64;
        let sequence = 1u64;
        let data = b"test data";

        let mac1 = hmac1.compute(request_id, sequence, data);
        let mac2 = hmac2.compute(request_id, sequence, data);

        assert_ne!(mac1, mac2);
    }
}
