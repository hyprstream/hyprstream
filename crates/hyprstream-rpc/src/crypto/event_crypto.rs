//! Cryptographic primitives for EventService transport.
//!
//! The controller supplies a fresh random secret for every committed group
//! epoch. Confidential event objects never use that secret directly as an AEAD
//! key: they derive session-scoped sender/track keys, bind publisher/track/
//! session/membership-version/epoch/sequence into AAD, and use an injective
//! counter nonce within each CSPRNG-generated session key domain. The module
//! retains legacy low-level wrap helpers for compatibility, but the surviving
//! EventService path distributes epochs exclusively through per-member HyKEM/COSE
//! grants in `group_key`.
//!
//! Key commitment provides fast wrong-key rejection before AES-256-GCM
//! decryption. Publisher signature transcripts bind the plaintext and epoch
//! coordinates and are consumed by the mandatory Ed25519 + ML-DSA-65 composite
//! attestation in `events`.

use aes_gcm::{
    aead::{Aead, AeadCore, KeyInit, Payload},
    Aes256Gcm, Nonce,
};
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

use super::backend::{derive_key, keyed_mac};

// ============================================================================
// EventPrivacy Mode
// ============================================================================

/// Privacy mode for event transport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventPrivacy {
    /// No group-key encryption: payload is written to the moq track unmodified
    /// (wire-identical to the pre-EV1 plaintext event bus). Use for the
    /// public/interop firehose profile, or any event with no confidentiality
    /// requirement. `EventPublisher::new`/`new_with_oid`/`new_oid_only`
    /// default to this mode.
    Public,
    /// All events on one broadcast stream. StreamService learns nothing about interests.
    ZeroKnowledge,
    /// Per-prefix direct circuits. Efficient but reveals topology.
    LimitedKnowledge,
}

// ============================================================================
// AES-256-GCM Helpers
// ============================================================================

/// Encrypt with AES-256-GCM. Returns ciphertext (includes auth tag).
fn aes_gcm_encrypt(
    key: &[u8; 32],
    nonce: &[u8; 12],
    plaintext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, String> {
    let cipher = Aes256Gcm::new(key.into());
    let payload = Payload {
        msg: plaintext,
        aad,
    };
    cipher
        .encrypt(Nonce::from_slice(nonce), payload)
        .map_err(|_| "AES-GCM encrypt failed".to_owned())
}

/// Decrypt with AES-256-GCM. Returns plaintext.
fn aes_gcm_decrypt(
    key: &[u8; 32],
    nonce: &[u8; 12],
    ciphertext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, String> {
    let cipher = Aes256Gcm::new(key.into());
    let payload = Payload {
        msg: ciphertext,
        aad,
    };
    cipher
        .decrypt(Nonce::from_slice(nonce), payload)
        .map_err(|_| "AES-GCM decrypt failed".to_owned())
}

/// Generate a random 12-byte AES-GCM nonce from OsRng.
fn random_nonce() -> [u8; 12] {
    // Idiomatic aes-gcm nonce generation (CodeQL-recognized CSPRNG).
    // Equivalent to the prior OsRng.fill_bytes(&mut [0u8;12]) — the value
    // was always overwritten by OsRng — but without the init literal that
    // tripped rust/hard-coded-cryptographic-value (false-positive root fix).
    Aes256Gcm::generate_nonce(&mut rand::rngs::OsRng).into()
}

// ============================================================================
// Group Key Wrapping
// ============================================================================

/// Build length-prefixed AAD for group key wrapping.
///
/// Format: `u32_le(len(sub_hash)) || sub_hash || u32_le(len(prefix)) || prefix`
///
/// Length prefixing prevents ambiguity where different sub_hash/prefix splits
/// produce the same concatenated bytes.
pub fn build_wrap_aad(sub_hash: &[u8; 32], prefix: &str) -> Vec<u8> {
    let mut aad = Vec::with_capacity(8 + 32 + prefix.len());
    aad.extend_from_slice(&(sub_hash.len() as u32).to_le_bytes());
    aad.extend_from_slice(sub_hash);
    aad.extend_from_slice(&(prefix.len() as u32).to_le_bytes());
    aad.extend_from_slice(prefix.as_bytes());
    aad
}

// ============================================================================
// Keyed-topic AEAD (EV3 — confidential profile)
// ============================================================================

/// Domain separator for the keyed-topic AAD (EV3).
const TOPIC_AAD_DOMAIN: &str = "hyprstream event-topic-aad v1";

/// Build length-prefixed AAD binding the **keyed routing leaf + epoch** (EV3).
///
/// When the semantic topic moves under the payload AEAD (replacing the cleartext
/// prefix-as-AAD), the AAD MUST bind the keyed topic leaf and the epoch, so a
/// ciphertext from `(topic A, epoch N)` replayed onto `(topic B, epoch M)` is
/// rejected by AEAD authentication. Without this binding, replacing the prefix
/// with an opaque keyed leaf would invite cross-topic / cross-epoch reuse.
///
/// Format: `domain || u32_le(len(leaf)) || leaf || epoch_le`
/// (`leaf` is the hex [`crate::event_subject::topic_leaf`] segment).
pub fn build_topic_aad(keyed_leaf: &str, epoch: u64) -> Vec<u8> {
    let leaf = keyed_leaf.as_bytes();
    let mut aad = Vec::with_capacity(TOPIC_AAD_DOMAIN.len() + 4 + leaf.len() + 8);
    aad.extend_from_slice(TOPIC_AAD_DOMAIN.as_bytes());
    aad.extend_from_slice(&(leaf.len() as u32).to_le_bytes());
    aad.extend_from_slice(leaf);
    aad.extend_from_slice(&epoch.to_le_bytes());
    aad
}

/// Encrypt an event with the **keyed-topic + epoch** bound into the AAD (EV3
/// confidential profile). Otherwise identical to [`encrypt_event`].
///
/// Use when publishing under a [`crate::event_subject::topic_leaf`] routing key;
/// the recipient reconstructs the same `keyed_leaf` + `epoch` for
/// [`decrypt_event_keyed`]. The semantic topic name travels inside the encrypted
/// payload, not in the AAD.
pub fn encrypt_event_keyed(
    group_key: &[u8; 32],
    keyed_leaf: &str,
    epoch: u64,
    plaintext: &[u8],
    _privacy_mode: EventPrivacy,
) -> Result<(Vec<u8>, Vec<u8>, [u8; 12], [u8; 16]), String> {
    let nonce = random_nonce();
    let commitment = key_commitment(group_key, &nonce);
    let aad = build_topic_aad(keyed_leaf, epoch);
    let aead_output = aes_gcm_encrypt(group_key, &nonce, plaintext, &aad)?;
    if aead_output.len() < 16 {
        return Err("AEAD output too short".to_owned());
    }
    let split = aead_output.len() - 16;
    let ciphertext = aead_output[..split].to_vec();
    let tag = aead_output[split..].to_vec();
    Ok((tag, ciphertext, nonce, commitment))
}

/// Decrypt an event encrypted with [`encrypt_event_keyed`] (keyed-topic + epoch
/// AAD). The caller must supply the same `keyed_leaf` + `epoch` used at encrypt
/// time — a mismatch (cross-topic or cross-epoch replay) fails AEAD auth.
pub fn decrypt_event_keyed(
    group_key: &[u8; 32],
    nonce: &[u8; 12],
    tag: &[u8],
    ciphertext: &[u8],
    keyed_leaf: &str,
    epoch: u64,
) -> Result<Vec<u8>, String> {
    let mut combined = Vec::with_capacity(ciphertext.len() + tag.len());
    combined.extend_from_slice(ciphertext);
    combined.extend_from_slice(tag);
    let aad = build_topic_aad(keyed_leaf, epoch);
    aes_gcm_decrypt(group_key, nonce, &combined, &aad)
}

/// Derive a wrap key from DH shared secret + pubkey XOR salt.
///
/// Follows the `derive_notification_keys()` convention:
/// - IKM: shared_secret || salt (where salt = XOR of our_pubkey and their_pubkey)
/// - Context: event-transport-specific domain separation string
pub fn derive_wrap_key(
    our_secret: &[u8; 32],
    their_pubkey: &[u8; 32],
    our_pubkey: &[u8; 32],
) -> Zeroizing<[u8; 32]> {
    // XOR public keys for salt (binds both parties' keys to the derivation)
    let mut salt = [0u8; 32];
    for i in 0..32 {
        salt[i] = our_pubkey[i] ^ their_pubkey[i];
    }

    // Build IKM: shared_secret || salt
    let mut ikm = [0u8; 64];
    ikm[..32].copy_from_slice(our_secret);
    ikm[32..64].copy_from_slice(&salt);

    let key = derive_key("hyprstream event-transport v1 wrap-key", &ikm);

    // Zeroize IKM containing secret material (use zeroize crate to prevent elision)
    use zeroize::Zeroize;
    ikm.zeroize();

    Zeroizing::new(key)
}

/// Wrap a group key for a specific subscriber.
///
/// Returns `(nonce || ciphertext)` as an opaque blob suitable for wire transport.
///
/// # Arguments
///
/// * `wrap_key` - 32-byte key derived from `derive_wrap_key()`
/// * `group_key` - 32-byte group key to wrap
/// * `sub_hash` - 32-byte subscriber identity hash (bound via AAD)
/// * `prefix` - Topic prefix string (bound via AAD)
pub fn wrap_group_key(
    wrap_key: &[u8; 32],
    group_key: &[u8; 32],
    sub_hash: &[u8; 32],
    prefix: &str,
) -> Result<Vec<u8>, String> {
    let nonce = random_nonce();
    let aad = build_wrap_aad(sub_hash, prefix);
    let ciphertext = aes_gcm_encrypt(wrap_key, &nonce, group_key, &aad)?;

    // Prepend nonce to ciphertext
    let mut blob = Vec::with_capacity(12 + ciphertext.len());
    blob.extend_from_slice(&nonce);
    blob.extend_from_slice(&ciphertext);
    Ok(blob)
}

/// Unwrap a group key blob.
///
/// Expects `wrapped_blob` in the format produced by `wrap_group_key()`:
/// `nonce (12 bytes) || ciphertext`.
///
/// # Arguments
///
/// * `wrap_key` - 32-byte key derived from `derive_wrap_key()`
/// * `wrapped_blob` - Opaque blob from `wrap_group_key()`
/// * `sub_hash` - 32-byte subscriber identity hash (must match wrapping AAD)
/// * `prefix` - Topic prefix string (must match wrapping AAD)
pub fn unwrap_group_key(
    wrap_key: &[u8; 32],
    wrapped_blob: &[u8],
    sub_hash: &[u8; 32],
    prefix: &str,
) -> Result<Zeroizing<[u8; 32]>, String> {
    if wrapped_blob.len() < 12 {
        return Err("wrapped blob too short".to_owned());
    }

    let nonce: [u8; 12] = wrapped_blob[..12]
        .try_into()
        .map_err(|_| "nonce extraction failed")?;
    let ciphertext = &wrapped_blob[12..];

    let aad = build_wrap_aad(sub_hash, prefix);
    let plaintext = aes_gcm_decrypt(wrap_key, &nonce, ciphertext, &aad)?;

    if plaintext.len() != 32 {
        return Err(format!(
            "unwrapped key wrong length: expected 32, got {}",
            plaintext.len()
        ));
    }

    let mut key = Zeroizing::new([0u8; 32]);
    key.copy_from_slice(&plaintext);
    Ok(key)
}

// ============================================================================
// Event Encryption / Decryption
// ============================================================================

/// Compute key commitment: `keyed_mac(group_key, nonce)[0..16]`.
///
/// Provides a committing property for the AEAD scheme. Receivers check this
/// commitment before attempting decryption, enabling fast rejection of events
/// encrypted under a different group key.
pub fn key_commitment(group_key: &[u8; 32], nonce: &[u8; 12]) -> [u8; 16] {
    let mac = keyed_mac(group_key, nonce);
    let mut commitment = [0u8; 16];
    commitment.copy_from_slice(&mac[..16]);
    commitment
}

/// Check key commitment for fast rejection before AEAD.
///
/// Uses constant-time comparison to avoid timing side-channels.
pub fn check_key_commitment(group_key: &[u8; 32], nonce: &[u8; 12], expected: &[u8; 16]) -> bool {
    let computed = key_commitment(group_key, nonce);
    bool::from(computed.ct_eq(expected))
}

/// Encrypt an EventEnvelopeV2 payload with a group key.
///
/// Returns `(tag, ciphertext, nonce, key_commitment)` suitable for
/// populating a TaggedPayload on the wire.
///
/// # Arguments
///
/// * `group_key` - 32-byte group encryption key
/// * `prefix` - Topic prefix used as AAD for domain separation
/// * `plaintext` - Event payload bytes to encrypt
/// * `_privacy_mode` - Privacy mode (reserved for future routing decisions)
///
/// # Returns
///
/// * `tag` - 16-byte AES-GCM authentication tag (last 16 bytes of AEAD output)
/// * `ciphertext` - Encrypted payload (AEAD output minus trailing tag)
/// * `nonce` - 12-byte random nonce
/// * `key_commitment` - 16-byte commitment for fast key rejection
pub fn encrypt_event(
    group_key: &[u8; 32],
    prefix: &str,
    plaintext: &[u8],
    _privacy_mode: EventPrivacy,
) -> Result<(Vec<u8>, Vec<u8>, [u8; 12], [u8; 16]), String> {
    let nonce = random_nonce();
    let commitment = key_commitment(group_key, &nonce);

    // Use prefix as AAD for domain separation
    let aad = prefix.as_bytes();
    let aead_output = aes_gcm_encrypt(group_key, &nonce, plaintext, aad)?;

    // AES-GCM output is ciphertext || 16-byte tag
    if aead_output.len() < 16 {
        return Err("AEAD output too short".to_owned());
    }
    let split = aead_output.len() - 16;
    let ciphertext = aead_output[..split].to_vec();
    let tag = aead_output[split..].to_vec();

    Ok((tag, ciphertext, nonce, commitment))
}

/// Encrypt with an explicit nonce supplied by a stateful protocol.
///
/// This is crate-private because nonce uniqueness becomes the caller's
/// responsibility.  The identified stream epoch profile derives the nonce from
/// its direction/track/epoch domain plus the authenticated sequence number.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn encrypt_event_with_nonce(
    group_key: &[u8; 32],
    nonce: [u8; 12],
    aad: &str,
    plaintext: &[u8],
) -> Result<(Vec<u8>, Vec<u8>, [u8; 12], [u8; 16]), String> {
    let commitment = key_commitment(group_key, &nonce);
    let aead_output = aes_gcm_encrypt(group_key, &nonce, plaintext, aad.as_bytes())?;
    if aead_output.len() < 16 {
        return Err("AEAD output too short".to_owned());
    }
    let split = aead_output.len() - 16;
    Ok((
        aead_output[split..].to_vec(),
        aead_output[..split].to_vec(),
        nonce,
        commitment,
    ))
}

/// Decrypt an event with full AAD binding (tag + ciphertext + prefix).
///
/// This is the preferred decryption path for EventEnvelopeV2, which
/// binds the prefix as AAD.
///
/// # Arguments
///
/// * `group_key` - 32-byte group encryption key
/// * `nonce` - 12-byte nonce
/// * `tag` - 16-byte authentication tag
/// * `ciphertext` - Encrypted payload (without tag)
/// * `prefix` - Topic prefix (must match encryption AAD)
pub fn decrypt_event_full(
    group_key: &[u8; 32],
    nonce: &[u8; 12],
    tag: &[u8],
    ciphertext: &[u8],
    prefix: &str,
) -> Result<Vec<u8>, String> {
    // Reconstruct AEAD input: ciphertext || tag
    let mut combined = Vec::with_capacity(ciphertext.len() + tag.len());
    combined.extend_from_slice(ciphertext);
    combined.extend_from_slice(tag);

    let aad = prefix.as_bytes();
    aes_gcm_decrypt(group_key, nonce, &combined, aad)
}

// ============================================================================
// Controller-managed epoch object profile (#555)
// ============================================================================

const EPOCH_OBJECT_AAD_DOMAIN: &[u8] = b"hyprstream event epoch object aad v2";

/// Derive an AEAD key unique to one `(epoch secret, sender, track, session)`
/// domain. The CSPRNG session prevents a restarted or concurrent publisher from
/// reusing a key when its in-memory sequence restarts.
pub fn derive_sender_track_key(
    epoch_secret: &[u8; 32],
    publisher_kid: &[u8; 32],
    track: &str,
    session_id: &[u8; 16],
) -> Zeroizing<[u8; 32]> {
    let mut material = Vec::with_capacity(32 + 32 + 4 + track.len() + 16);
    material.extend_from_slice(epoch_secret);
    material.extend_from_slice(publisher_kid);
    material.extend_from_slice(&(track.len() as u32).to_be_bytes());
    material.extend_from_slice(track.as_bytes());
    material.extend_from_slice(session_id);
    Zeroizing::new(blake3::derive_key(
        "hyprstream event sender track session key v2",
        &material,
    ))
}

/// Return the injective 96-bit nonce for one sequence in a session-scoped key
/// domain. Sequence zero is reserved as the pre-publication state.
pub fn derive_event_nonce(sequence: u64) -> Result<[u8; 12], String> {
    if sequence == 0 {
        return Err("event sequence zero is reserved".to_owned());
    }
    let sequence = sequence.to_be_bytes();
    // b"EVN1" is a domain-separation label, not nonce entropy. Nonce uniqueness
    // comes from the injective big-endian sequence under a key scoped to a fresh
    // 128-bit OsRng session ID, so (key, nonce) cannot repeat across restart,
    // re-registration, or failover. Reviewed by two independent security
    // reviewers; see PR #1111 discussion.
    Ok([
        b'E', // codeql[rust/hard-coded-cryptographic-value]
        b'V', // codeql[rust/hard-coded-cryptographic-value]
        b'N', // codeql[rust/hard-coded-cryptographic-value]
        b'1', // codeql[rust/hard-coded-cryptographic-value]
        sequence[0],
        sequence[1],
        sequence[2],
        sequence[3],
        sequence[4],
        sequence[5],
        sequence[6],
        sequence[7],
    ])
}

/// Canonical authenticated coordinates kept inside the application payload/AAD;
/// a stock relay need not understand any of them.
pub fn build_epoch_object_aad(
    track: &str,
    publisher_kid: &[u8; 32],
    session_id: &[u8; 16],
    membership_version: u64,
    epoch: u64,
    sequence: u64,
) -> Vec<u8> {
    let mut aad = Vec::with_capacity(EPOCH_OBJECT_AAD_DOMAIN.len() + 4 + track.len() + 72);
    aad.extend_from_slice(EPOCH_OBJECT_AAD_DOMAIN);
    aad.extend_from_slice(&(track.len() as u32).to_be_bytes());
    aad.extend_from_slice(track.as_bytes());
    aad.extend_from_slice(publisher_kid);
    aad.extend_from_slice(session_id);
    aad.extend_from_slice(&membership_version.to_be_bytes());
    aad.extend_from_slice(&epoch.to_be_bytes());
    aad.extend_from_slice(&sequence.to_be_bytes());
    aad
}

pub fn encrypt_epoch_event(
    epoch_secret: &[u8; 32],
    track: &str,
    publisher_kid: &[u8; 32],
    session_id: &[u8; 16],
    membership_version: u64,
    epoch: u64,
    sequence: u64,
    plaintext: &[u8],
) -> Result<(Vec<u8>, Vec<u8>, [u8; 12], [u8; 16]), String> {
    let key = derive_sender_track_key(epoch_secret, publisher_kid, track, session_id);
    let nonce = derive_event_nonce(sequence)?;
    let aad = build_epoch_object_aad(
        track,
        publisher_kid,
        session_id,
        membership_version,
        epoch,
        sequence,
    );
    let commitment = key_commitment(&key, &nonce);
    let output = aes_gcm_encrypt(&key, &nonce, plaintext, &aad)?;
    if output.len() < 16 {
        return Err("AEAD output too short".to_owned());
    }
    let split = output.len() - 16;
    Ok((
        output[split..].to_vec(),
        output[..split].to_vec(),
        nonce,
        commitment,
    ))
}

pub fn decrypt_epoch_event(
    epoch_secret: &[u8; 32],
    track: &str,
    publisher_kid: &[u8; 32],
    session_id: &[u8; 16],
    membership_version: u64,
    epoch: u64,
    sequence: u64,
    nonce: &[u8; 12],
    tag: &[u8],
    ciphertext: &[u8],
    commitment: &[u8; 16],
) -> Result<Vec<u8>, String> {
    let key = derive_sender_track_key(epoch_secret, publisher_kid, track, session_id);
    if !check_key_commitment(&key, nonce, commitment) {
        return Err("sender/track/session key commitment mismatch".to_owned());
    }
    let aad = build_epoch_object_aad(
        track,
        publisher_kid,
        session_id,
        membership_version,
        epoch,
        sequence,
    );
    let mut combined = Vec::with_capacity(ciphertext.len() + tag.len());
    combined.extend_from_slice(ciphertext);
    combined.extend_from_slice(tag);
    aes_gcm_decrypt(&key, nonce, &combined, &aad)
}

/// Hybrid-attested event transcript. Session, epoch coordinates, and sequence
/// are signed, so possession of the symmetric epoch key is not publisher
/// identity evidence.
pub fn build_epoch_event_sig_message(
    topic: &str,
    payload: &[u8],
    timestamp: i64,
    session_id: &[u8; 16],
    membership_version: u64,
    epoch: u64,
    sequence: u64,
) -> Vec<u8> {
    let mut msg = build_event_sig_message(topic, payload, timestamp);
    msg.extend_from_slice(session_id);
    msg.extend_from_slice(&membership_version.to_be_bytes());
    msg.extend_from_slice(&epoch.to_be_bytes());
    msg.extend_from_slice(&sequence.to_be_bytes());
    msg
}
// Ed25519 Event Signing
// ============================================================================

/// Build the signing message for EventEnvelopeV2.
///
/// Format: `u32_le(len(topic)) || topic || u32_le(len(payload)) || payload || timestamp_i64_le`
///
/// Variable-length fields are length-prefixed to prevent concatenation ambiguity.
/// The fixed-length timestamp is appended without a length prefix.
pub fn build_event_sig_message(topic: &str, payload: &[u8], timestamp: i64) -> Vec<u8> {
    let mut msg = Vec::with_capacity(4 + topic.len() + 4 + payload.len() + 8);
    msg.extend_from_slice(&(topic.len() as u32).to_le_bytes());
    msg.extend_from_slice(topic.as_bytes());
    msg.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    msg.extend_from_slice(payload);
    msg.extend_from_slice(&timestamp.to_le_bytes());
    msg
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_unwrap_group_key_roundtrip() {
        let wrap_key = [0x42u8; 32];
        let group_key = [0xABu8; 32];
        let sub_hash = [0x01u8; 32];
        let prefix = "serve:model:qwen3";

        let wrapped = wrap_group_key(&wrap_key, &group_key, &sub_hash, prefix).unwrap();
        let unwrapped = unwrap_group_key(&wrap_key, &wrapped, &sub_hash, prefix).unwrap();

        assert_eq!(&*unwrapped, &group_key);
    }

    #[test]
    fn test_wrap_with_wrong_key_fails() {
        let wrap_key = [0x42u8; 32];
        let wrong_key = [0x99u8; 32];
        let group_key = [0xABu8; 32];
        let sub_hash = [0x01u8; 32];
        let prefix = "serve:model:qwen3";

        let wrapped = wrap_group_key(&wrap_key, &group_key, &sub_hash, prefix).unwrap();
        let result = unwrap_group_key(&wrong_key, &wrapped, &sub_hash, prefix);

        assert!(result.is_err());
    }

    #[test]
    fn test_wrap_cross_prefix_fails() {
        let wrap_key = [0x42u8; 32];
        let group_key = [0xABu8; 32];
        let sub_hash = [0x01u8; 32];

        let wrapped = wrap_group_key(&wrap_key, &group_key, &sub_hash, "prefix-a").unwrap();
        let result = unwrap_group_key(&wrap_key, &wrapped, &sub_hash, "prefix-b");

        assert!(
            result.is_err(),
            "cross-prefix unwrap must fail due to AAD mismatch"
        );
    }

    #[test]
    fn test_wrap_cross_subscriber_fails() {
        let wrap_key = [0x42u8; 32];
        let group_key = [0xABu8; 32];
        let sub_hash_a = [0x01u8; 32];
        let sub_hash_b = [0x02u8; 32];
        let prefix = "serve:model:qwen3";

        let wrapped = wrap_group_key(&wrap_key, &group_key, &sub_hash_a, prefix).unwrap();
        let result = unwrap_group_key(&wrap_key, &wrapped, &sub_hash_b, prefix);

        assert!(
            result.is_err(),
            "cross-subscriber unwrap must fail due to AAD mismatch"
        );
    }

    #[test]
    fn test_wrap_blob_too_short() {
        let wrap_key = [0x42u8; 32];
        let sub_hash = [0x01u8; 32];

        let result = unwrap_group_key(&wrap_key, &[0u8; 5], &sub_hash, "prefix");
        assert!(result.is_err());
    }

    #[test]
    fn test_encrypt_decrypt_event_roundtrip() {
        let group_key = [0x42u8; 32];
        let prefix = "serve:model:qwen3";
        let plaintext = b"hello, event world!";

        let (tag, ciphertext, nonce, commitment) =
            encrypt_event(&group_key, prefix, plaintext, EventPrivacy::ZeroKnowledge).unwrap();

        // Verify key commitment
        assert!(check_key_commitment(&group_key, &nonce, &commitment));

        // Decrypt with full AAD binding
        let decrypted = decrypt_event_full(&group_key, &nonce, &tag, &ciphertext, prefix).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_encrypt_decrypt_event_simple() {
        let group_key = [0x42u8; 32];
        let prefix = "test";
        let plaintext = b"simple test";

        let (tag, ciphertext, nonce, _commitment) = encrypt_event(
            &group_key,
            prefix,
            plaintext,
            EventPrivacy::LimitedKnowledge,
        )
        .unwrap();

        // Reconstruct combined ciphertext||tag for simple decrypt (no AAD)
        // Note: simple decrypt uses empty AAD, so this will fail because
        // encrypt uses prefix as AAD. Use decrypt_event_full instead.
        let decrypted = decrypt_event_full(&group_key, &nonce, &tag, &ciphertext, prefix).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_key_commitment_matches() {
        let group_key = [0x42u8; 32];
        let nonce = [0x01u8; 12];

        let commitment = key_commitment(&group_key, &nonce);
        assert!(check_key_commitment(&group_key, &nonce, &commitment));

        // Same inputs produce same commitment
        let commitment2 = key_commitment(&group_key, &nonce);
        assert_eq!(commitment, commitment2);
    }

    #[test]
    fn test_key_commitment_rejects_wrong_key() {
        let group_key_a = [0x42u8; 32];
        let group_key_b = [0x99u8; 32];
        let nonce = [0x01u8; 12];

        let commitment = key_commitment(&group_key_a, &nonce);

        // Wrong key must not match
        assert!(!check_key_commitment(&group_key_b, &nonce, &commitment));
    }

    #[test]
    fn test_key_commitment_rejects_wrong_nonce() {
        let group_key = [0x42u8; 32];
        let nonce_a = [0x01u8; 12];
        let nonce_b = [0x02u8; 12];

        let commitment = key_commitment(&group_key, &nonce_a);

        // Wrong nonce must not match
        assert!(!check_key_commitment(&group_key, &nonce_b, &commitment));
    }

    #[test]
    fn test_aad_binding_prevents_cross_prefix_confusion() {
        let group_key = [0x42u8; 32];
        let plaintext = b"sensitive data";

        let (tag, ciphertext, nonce, _) = encrypt_event(
            &group_key,
            "prefix-a",
            plaintext,
            EventPrivacy::ZeroKnowledge,
        )
        .unwrap();

        // Decrypt with wrong prefix must fail
        let result = decrypt_event_full(&group_key, &nonce, &tag, &ciphertext, "prefix-b");
        assert!(
            result.is_err(),
            "cross-prefix decryption must fail due to AAD mismatch"
        );
    }

    #[test]
    fn test_build_event_sig_message_deterministic() {
        let topic = "serve:model:qwen3";
        let payload = b"event payload";
        let timestamp = 1700000000i64;

        let msg1 = build_event_sig_message(topic, payload, timestamp);
        let msg2 = build_event_sig_message(topic, payload, timestamp);

        assert_eq!(msg1, msg2);
    }

    #[test]
    fn test_build_event_sig_message_structure() {
        let topic = "test-topic";
        let payload = b"data";
        let timestamp = 42i64;

        let msg = build_event_sig_message(topic, payload, timestamp);

        // Verify structure: u32_le(10) || "test-topic" || u32_le(4) || "data" || i64_le(42)
        let expected_len = 4 + 10 + 4 + 4 + 8;
        assert_eq!(msg.len(), expected_len);

        // Parse and verify
        let topic_len = u32::from_le_bytes(msg[0..4].try_into().unwrap()) as usize;
        assert_eq!(topic_len, 10);
        assert_eq!(&msg[4..14], b"test-topic");

        let payload_len = u32::from_le_bytes(msg[14..18].try_into().unwrap()) as usize;
        assert_eq!(payload_len, 4);
        assert_eq!(&msg[18..22], b"data");

        let ts = i64::from_le_bytes(msg[22..30].try_into().unwrap());
        assert_eq!(ts, 42);
    }

    #[test]
    fn test_build_event_sig_message_different_topics_differ() {
        let payload = b"same";
        let timestamp = 1i64;

        let msg1 = build_event_sig_message("topic-a", payload, timestamp);
        let msg2 = build_event_sig_message("topic-b", payload, timestamp);

        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_build_event_sig_message_prevents_concatenation_ambiguity() {
        // "ab" + "cd" vs "abc" + "d" must differ due to length prefixing
        let msg1 = build_event_sig_message("ab", b"cd", 0);
        let msg2 = build_event_sig_message("abc", b"d", 0);

        assert_ne!(msg1, msg2);
    }

    #[test]
    fn test_build_wrap_aad_prevents_ambiguity() {
        let hash_a = [0x01u8; 32];
        let hash_b = [0x02u8; 32];

        let aad1 = build_wrap_aad(&hash_a, "prefix");
        let aad2 = build_wrap_aad(&hash_b, "prefix");

        assert_ne!(aad1, aad2, "different sub_hash must produce different AAD");

        // Same inputs produce same AAD
        let aad3 = build_wrap_aad(&hash_a, "prefix");
        assert_eq!(aad1, aad3);
    }

    #[test]
    fn test_derive_wrap_key_deterministic() {
        let secret = [0x42u8; 32];
        let their_pub = [0x01u8; 32];
        let our_pub = [0x02u8; 32];

        let key1 = derive_wrap_key(&secret, &their_pub, &our_pub);
        let key2 = derive_wrap_key(&secret, &their_pub, &our_pub);

        assert_eq!(&*key1, &*key2);
    }

    #[test]
    fn test_derive_wrap_key_different_secrets_differ() {
        let their_pub = [0x01u8; 32];
        let our_pub = [0x02u8; 32];

        let key1 = derive_wrap_key(&[0x11u8; 32], &their_pub, &our_pub);
        let key2 = derive_wrap_key(&[0x22u8; 32], &their_pub, &our_pub);

        assert_ne!(&*key1, &*key2);
    }

    #[test]
    fn test_encrypt_empty_plaintext() {
        let group_key = [0x42u8; 32];
        let prefix = "test";

        let (tag, ciphertext, nonce, commitment) =
            encrypt_event(&group_key, prefix, b"", EventPrivacy::ZeroKnowledge).unwrap();

        assert!(check_key_commitment(&group_key, &nonce, &commitment));
        assert!(
            ciphertext.is_empty(),
            "empty plaintext produces empty ciphertext body"
        );

        let decrypted = decrypt_event_full(&group_key, &nonce, &tag, &ciphertext, prefix).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn test_privacy_modes_both_work() {
        let group_key = [0x42u8; 32];
        let prefix = "test";
        let plaintext = b"works in both modes";

        // ZeroKnowledge mode
        let (tag_zk, ct_zk, nonce_zk, _) =
            encrypt_event(&group_key, prefix, plaintext, EventPrivacy::ZeroKnowledge).unwrap();
        let dec_zk = decrypt_event_full(&group_key, &nonce_zk, &tag_zk, &ct_zk, prefix).unwrap();
        assert_eq!(dec_zk, plaintext);

        // LimitedKnowledge mode
        let (tag_lk, ct_lk, nonce_lk, _) = encrypt_event(
            &group_key,
            prefix,
            plaintext,
            EventPrivacy::LimitedKnowledge,
        )
        .unwrap();
        let dec_lk = decrypt_event_full(&group_key, &nonce_lk, &tag_lk, &ct_lk, prefix).unwrap();
        assert_eq!(dec_lk, plaintext);
    }

    #[test]
    fn keyed_aead_round_trips_and_binds_topic_epoch() {
        let group_key = [0x11u8; 32];
        let plaintext = b"payload-bytes";
        let leaf = "deadbeef".repeat(8); // 64 hex chars, like topic_leaf output
        let epoch = 7u64;

        let (tag, ct, nonce, _commitment) = encrypt_event_keyed(
            &group_key,
            &leaf,
            epoch,
            plaintext,
            EventPrivacy::ZeroKnowledge,
        )
        .unwrap();
        let recovered = decrypt_event_keyed(&group_key, &nonce, &tag, &ct, &leaf, epoch).unwrap();
        assert_eq!(recovered, plaintext);

        // Cross-epoch replay: same leaf, wrong epoch -> AEAD auth fails.
        assert!(decrypt_event_keyed(&group_key, &nonce, &tag, &ct, &leaf, epoch + 1).is_err());
        // Cross-topic replay: wrong leaf, same epoch -> fails.
        let other_leaf = "cafe".repeat(16);
        assert!(decrypt_event_keyed(&group_key, &nonce, &tag, &ct, &other_leaf, epoch).is_err());
        // Wrong group key -> fails.
        assert!(decrypt_event_keyed(&[0x22u8; 32], &nonce, &tag, &ct, &leaf, epoch).is_err());
    }

    #[test]
    fn session_scoped_keys_and_injective_nonces_prevent_pair_reuse() {
        let epoch_secret = [0x11u8; 32];
        let publisher_kid = [0x22u8; 32];
        let session_a = [0x33u8; 16];
        let session_b = [0x44u8; 16];
        let key_a = derive_sender_track_key(&epoch_secret, &publisher_kid, "worker", &session_a);
        let key_b = derive_sender_track_key(&epoch_secret, &publisher_kid, "worker", &session_b);
        let nonce_1 = derive_event_nonce(1).unwrap();
        let nonce_2 = derive_event_nonce(2).unwrap();

        assert_ne!(&*key_a, &*key_b);
        assert_eq!(&nonce_1[..4], b"EVN1");
        assert_eq!(&nonce_1[4..], &1u64.to_be_bytes());
        assert_eq!(&nonce_2[4..], &2u64.to_be_bytes());
        assert_ne!(nonce_1, nonce_2);
        assert!(derive_event_nonce(0).is_err());
    }

    #[test]
    fn session_epoch_and_sequence_mutation_fail_aead_authentication() {
        let epoch_secret = [0x51u8; 32];
        let publisher_kid = [0x52u8; 32];
        let session_id = [0x53u8; 16];
        let (tag, ciphertext, nonce, commitment) = encrypt_epoch_event(
            &epoch_secret,
            "worker",
            &publisher_kid,
            &session_id,
            7,
            9,
            11,
            b"session-bound payload",
        )
        .unwrap();

        assert_eq!(
            decrypt_epoch_event(
                &epoch_secret,
                "worker",
                &publisher_kid,
                &session_id,
                7,
                9,
                11,
                &nonce,
                &tag,
                &ciphertext,
                &commitment,
            )
            .unwrap(),
            b"session-bound payload"
        );

        let mut wrong_session = session_id;
        wrong_session[0] ^= 1;
        assert!(decrypt_epoch_event(
            &epoch_secret,
            "worker",
            &publisher_kid,
            &wrong_session,
            7,
            9,
            11,
            &nonce,
            &tag,
            &ciphertext,
            &commitment,
        )
        .is_err());
        assert!(decrypt_epoch_event(
            &epoch_secret,
            "worker",
            &publisher_kid,
            &session_id,
            7,
            10,
            11,
            &nonce,
            &tag,
            &ciphertext,
            &commitment,
        )
        .is_err());
        assert!(decrypt_epoch_event(
            &epoch_secret,
            "worker",
            &publisher_kid,
            &session_id,
            7,
            9,
            12,
            &nonce,
            &tag,
            &ciphertext,
            &commitment,
        )
        .is_err());
    }

    #[test]
    fn topic_aad_is_deterministic_and_distinct() {
        assert_eq!(build_topic_aad("abcd", 1), build_topic_aad("abcd", 1));
        assert_ne!(build_topic_aad("abcd", 1), build_topic_aad("abcd", 2));
        assert_ne!(build_topic_aad("abcd", 1), build_topic_aad("abce", 1));
        assert!(build_topic_aad("abcd", 1).starts_with(b"hyprstream event-topic-aad v1"));
    }
}
