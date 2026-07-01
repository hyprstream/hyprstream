//! Broadcast crypto primitives preserved from the removed NotificationService.
//!
//! These are reused by the WASM client (`wasm_api.rs`) and remain public so the
//! browser binding is unaffected by the NotificationService removal (EV5/#605).
//! The blind-relay broadcast service itself is gone; only the reusable crypto
//! helpers (pubkey fingerprint, length-prefixed AAD, one-shot MAC, attestation
//! message) remain here.

use subtle::ConstantTimeEq;

use super::backend::keyed_mac;
use crate::error::{EnvelopeError, EnvelopeResult};

/// 128-bit pubkey fingerprint (Blake3 truncated).
pub type PubkeyFingerprint = [u8; 16];

/// Compute a 128-bit fingerprint of a public key: `Blake3(pubkey)[..16]`.
pub fn pubkey_fingerprint(pubkey_bytes: &[u8; 32]) -> PubkeyFingerprint {
    let hash = blake3::hash(pubkey_bytes);
    let mut fp = [0u8; 16];
    fp.copy_from_slice(&hash.as_bytes()[..16]);
    fp
}

/// Build length-prefixed AAD for payload encryption.
///
/// Format: `u32_le(len(intent_id)) || intent_id || u32_le(len(scope)) || scope`.
/// Length prefixing prevents ambiguity where different intent_id/scope splits
/// produce the same concatenated bytes.
pub fn build_payload_aad(intent_id: &str, scope: &str) -> Vec<u8> {
    let mut aad = Vec::with_capacity(8 + intent_id.len() + scope.len());
    aad.extend_from_slice(&(intent_id.len() as u32).to_le_bytes());
    aad.extend_from_slice(intent_id.as_bytes());
    aad.extend_from_slice(&(scope.len() as u32).to_le_bytes());
    aad.extend_from_slice(scope.as_bytes());
    aad
}

/// Compute a one-shot MAC: `keyed_mac(mac_key, ciphertext)`.
pub fn notification_mac(mac_key: &[u8; 32], ciphertext: &[u8]) -> [u8; 32] {
    keyed_mac(mac_key, ciphertext)
}

/// Verify a one-shot MAC in constant time.
pub fn notification_mac_verify(
    mac_key: &[u8; 32],
    ciphertext: &[u8],
    expected_mac: &[u8; 32],
) -> EnvelopeResult<()> {
    let computed = keyed_mac(mac_key, ciphertext);
    if bool::from(computed.ct_eq(expected_mac)) {
        Ok(())
    } else {
        Err(EnvelopeError::MacVerification)
    }
}

/// Build the message signed by the publisher's Ed25519 key for attestation.
///
/// Format: `ephemeral_pubkey || blinded_sub_pubkey || u32_le(len(scope)) ||
/// scope || u32_le(len(intent_id)) || intent_id`. Fixed-length pubkeys are not
/// length-prefixed; variable-length fields are, to prevent concatenation
/// ambiguity.
pub fn build_attestation_message(
    ephemeral_pubkey: &[u8; 32],
    blinded_sub_pubkey: &[u8; 32],
    scope: &str,
    intent_id: &str,
) -> Vec<u8> {
    let mut msg = Vec::with_capacity(64 + 8 + scope.len() + intent_id.len());
    msg.extend_from_slice(ephemeral_pubkey);
    msg.extend_from_slice(blinded_sub_pubkey);
    msg.extend_from_slice(&(scope.len() as u32).to_le_bytes());
    msg.extend_from_slice(scope.as_bytes());
    msg.extend_from_slice(&(intent_id.len() as u32).to_le_bytes());
    msg.extend_from_slice(intent_id.as_bytes());
    msg
}
