//! Stream block HMAC for authenticating streaming responses.
//!
//! Streaming responses (XPUB/XSUB) use chained MACs instead of per-token Ed25519
//! signatures for performance:
//! - Ed25519: ~10k signatures/sec
//! - Blake3/HMAC-SHA256: millions of MACs/sec
//!
//! # Wire format
//!
//! Each chunk's MAC depends on the previous chunk's MAC, providing cryptographic
//! ordering without explicit sequence numbers:
//!
//! ```text
//! mac_0 = HMAC(key, topic.as_bytes() || capnp_data_0)[..16]  // Block 0: seed = topic
//! mac_n = HMAC(key, mac_{n-1} || capnp_data_n)[..16]         // Subsequent blocks
//! ```
//!
//! # Security Properties
//!
//! - Authenticates server as holder of the DH-derived MAC key
//! - Per-stream key derivation binds chains to specific sessions
//! - Chained MAC provides cryptographic ordering (no separate sequence field)
//! - Reordering is impossible — can't verify block N without mac_{N-1}
//! - 16-byte truncation (128 bits) provides standard authentication strength
//! - Constant-time verification via `subtle::ConstantTimeEq` prevents timing attacks
//!
//! # Backend
//!
//! - Default: Blake3 `keyed_hash()` (~10+ GB/s with SIMD)
//! - FIPS mode: HMAC-SHA256 (FIPS 198-1)
//!
//! # Cryptographic notes (from external review)
//!
//! - 128-bit truncation is well above NIST recommendations and standard for stream auth
//! - First-N-bytes truncation is the safe pattern for both Blake3 and HMAC-SHA256
//! - Topic-as-seed is sound: the key is secret; topic publicness is fine
//! - Per-stream replay protection comes from the per-call ECDH ephemeral keypair,
//!   which makes the topic effectively per-stream-random

use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

/// Chained HMAC state for stream block authentication, matching wire format.
///
/// Used by both server (`StreamBuilder`) and client (`StreamMount`).
///
/// # Wire format
/// - Block 0: HMAC(key, topic.as_bytes() || capnp_data)[..16]
/// - Block N: HMAC(key, prev_mac (16 bytes) || capnp_data)[..16]
#[derive(Clone)]
pub struct StreamHmacState {
    key: Zeroizing<[u8; 32]>,
    prev_mac: Option<[u8; 16]>,
    topic: String,
}

impl StreamHmacState {
    /// Create new HMAC chain state.
    pub fn new(key: [u8; 32], topic: String) -> Self {
        Self {
            key: Zeroizing::new(key),
            prev_mac: None,
            topic,
        }
    }

    /// Compute 16-byte truncated MAC for next block, advancing the chain.
    pub fn compute_next(&mut self, capnp_data: &[u8]) -> [u8; 16] {
        let mut input = Vec::with_capacity(64 + capnp_data.len());
        match &self.prev_mac {
            None => input.extend_from_slice(self.topic.as_bytes()),
            Some(prev) => input.extend_from_slice(prev),
        }
        input.extend_from_slice(capnp_data);

        let truncated = crate::crypto::keyed_mac_truncated(&self.key, &input);
        self.prev_mac = Some(truncated);
        truncated
    }

    /// Compute the next MAC without advancing chain state (peek).
    fn compute_next_peek(&self, capnp_data: &[u8]) -> [u8; 16] {
        let mut input = Vec::with_capacity(64 + capnp_data.len());
        match &self.prev_mac {
            None => input.extend_from_slice(self.topic.as_bytes()),
            Some(prev) => input.extend_from_slice(prev),
        }
        input.extend_from_slice(capnp_data);
        crate::crypto::keyed_mac_truncated(&self.key, &input)
    }

    /// Verify the next block's MAC in constant time.
    ///
    /// Advances the chain only if verification succeeds.
    /// Returns true if the MAC matches, false otherwise.
    pub fn verify_next(&mut self, capnp_data: &[u8], expected_mac: &[u8]) -> bool {
        if expected_mac.len() != 16 {
            return false;
        }
        let computed = self.compute_next_peek(capnp_data);
        if computed.ct_eq(expected_mac).into() {
            self.prev_mac = Some(computed);
            true
        } else {
            false
        }
    }

    /// Get the topic.
    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Get previous MAC bytes (for prevMac field in StreamBlock).
    pub fn prev_mac_bytes(&self) -> &[u8] {
        match &self.prev_mac {
            Some(mac) => mac,
            None => &self.topic.as_bytes()[..16.min(self.topic.len())],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_hmac_compute_verify_roundtrip() {
        let key = [0x42u8; 32];
        let topic = "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff".to_owned();

        // Producer
        let mut producer = StreamHmacState::new(key, topic.clone());
        let mac0 = producer.compute_next(b"block 0 data");
        let mac1 = producer.compute_next(b"block 1 data");
        let mac2 = producer.compute_next(b"block 2 data");

        // Verifier
        let mut verifier = StreamHmacState::new(key, topic);
        assert!(verifier.verify_next(b"block 0 data", &mac0));
        assert!(verifier.verify_next(b"block 1 data", &mac1));
        assert!(verifier.verify_next(b"block 2 data", &mac2));
    }

    #[test]
    fn test_stream_hmac_rejects_wrong_mac() {
        let key = [0x42u8; 32];
        let topic = "topic".to_owned();

        let mut verifier = StreamHmacState::new(key, topic);
        let bad_mac = [0xffu8; 16];
        assert!(!verifier.verify_next(b"data", &bad_mac));
    }

    #[test]
    fn test_stream_hmac_rejects_wrong_length() {
        let key = [0x42u8; 32];
        let topic = "topic".to_owned();

        let mut verifier = StreamHmacState::new(key, topic);
        assert!(!verifier.verify_next(b"data", &[0u8; 32])); // 32 bytes is wrong
        assert!(!verifier.verify_next(b"data", &[0u8; 8]));  // 8 bytes is wrong
    }
}
