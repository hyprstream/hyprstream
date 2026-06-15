//! WASM-bindgen API surface for browser clients AND browser-side services.
//!
//! Compiled via: wasm-pack build --target web crates/hyprstream-rpc
//!
//! This module provides the cryptographic primitives needed for:
//! - Generating ephemeral keypairs for streaming DH
//! - Deriving stream keys (topic + MAC)
//! - Verifying stream blocks (chained HMAC) — stateless (verify_stream_block) and
//!   stateful chain (init_stream_hmac + verify_stream_block_step)
//! - ECDH key exchange (Ristretto255)
//! - Verifying incoming envelopes (for browser-side services)
//! - ZMTP 3.1 framing (encode/decode frames, greeting, handshake commands)
//! - Notification broadcast encryption (blinded DH, AES-GCM, one-shot MAC)
//!
//! # Signing is handled externally
//!
//! Signing Ed25519 request envelopes is **not** in this module. It moved to
//! `hyprstream-rpc-std::wasm_exports::RpcSession`, which takes an external
//! signing callback (a JS async function) that produces signatures over
//! canonical envelope bytes. This lets consumers keep signing keys in an
//! isolated context — an aegis-vault wasm module, a cross-origin iframe, a
//! Wanix worker process, etc. The seed never enters this crate.
//!
//! All crypto is compiled from Rust - no TypeScript reimplementation.

#![cfg(target_arch = "wasm32")]

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsError;

thread_local! {
    static WASM_NONCE_CACHE: std::sync::Arc<crate::envelope::InMemoryNonceCache> =
        std::sync::Arc::new(crate::envelope::InMemoryNonceCache::new());

    // Per-stream HMAC chains for concurrent stream verification.
    // Each stream gets a unique handle from init_stream_hmac.
    // Uses the same StreamHmacState as the server for wire-format compatibility.
    static STREAM_HMAC_TABLE: std::cell::RefCell<std::collections::HashMap<u32, crate::crypto::StreamHmacState>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
    static NEXT_HMAC_HANDLE: std::cell::Cell<u32> = std::cell::Cell::new(1);

    // WASM-side PQ trust store: maps Ed25519 pubkey (32 bytes) → ML-DSA-65 vk
    // bytes. Populated by register_pq_trust(). When a signer is present here,
    // verify_signed_envelope enforces Hybrid (outer SNS layer required). (#158)
    static WASM_PQ_BINDINGS: std::cell::RefCell<std::collections::HashMap<[u8; 32], Vec<u8>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

// ============================================================================
// Key generation (ephemeral-only)
// ============================================================================

/// Generate an ephemeral Ristretto255 keypair for streaming DH.
///
/// # Returns
///
/// 64 bytes: [secret_scalar (32) || pubkey_bytes (32)]
#[wasm_bindgen]
pub fn generate_ephemeral_keypair() -> Result<Vec<u8>, JsError> {
    use crate::crypto::key_exchange::DefaultKeyExchange;
    use crate::crypto::KeyExchange;

    let (secret, public) = DefaultKeyExchange::generate_keypair();

    // Serialize: secret scalar bytes || public point bytes
    let mut result = Vec::with_capacity(64);
    result.extend_from_slice(secret.scalar().as_bytes());
    result.extend_from_slice(&DefaultKeyExchange::pubkey_to_bytes(&public));

    Ok(result)
}

// ============================================================================
// Stream key derivation
// ============================================================================

/// DH + derive stream keys.
///
/// # Arguments
///
/// * `shared_secret` - 32 bytes from ecdh_ristretto
/// * `client_pub` - 32 bytes client Ristretto255 pubkey
/// * `server_pub` - 32 bytes server Ristretto255 pubkey
///
/// # Returns
///
/// 128 bytes: [topic(32) || mac_key(32) || ctrl_topic(32) || ctrl_mac_key(32)]
#[wasm_bindgen]
pub fn derive_stream_keys(
    shared_secret: &[u8],
    client_pub: &[u8],
    server_pub: &[u8],
) -> Result<Vec<u8>, JsError> {
    use crate::crypto::key_exchange::derive_stream_keys;

    if shared_secret.len() != 32 {
        return Err(JsError::new("shared_secret must be 32 bytes"));
    }
    if client_pub.len() != 32 {
        return Err(JsError::new("client_pub must be 32 bytes"));
    }
    if server_pub.len() != 32 {
        return Err(JsError::new("server_pub must be 32 bytes"));
    }

    let mut ss = [0u8; 32];
    ss.copy_from_slice(shared_secret);
    let mut cp = [0u8; 32];
    cp.copy_from_slice(client_pub);
    let mut sp = [0u8; 32];
    sp.copy_from_slice(server_pub);

    let keys = derive_stream_keys(&ss, &cp, &sp)
        .map_err(|e| JsError::new(&format!("key derivation failed: {}", e)))?;

    let mut result = Vec::with_capacity(128);
    // topic is a hex string (64 chars = 32 bytes decoded)
    let topic_bytes = hex::decode(&keys.topic)
        .map_err(|e| JsError::new(&format!("invalid topic hex: {}", e)))?;
    result.extend_from_slice(&topic_bytes);
    result.extend_from_slice(&*keys.mac_key);
    let ctrl_topic_bytes = hex::decode(&keys.ctrl_topic)
        .map_err(|e| JsError::new(&format!("invalid ctrl_topic hex: {}", e)))?;
    result.extend_from_slice(&ctrl_topic_bytes);
    result.extend_from_slice(&*keys.ctrl_mac_key);

    Ok(result)
}

// ============================================================================
// Stream block verification
// ============================================================================

/// Initialize a per-stream HMAC chain for stateful block verification.
///
/// Returns a handle (u32) to identify this stream's HMAC state. Pass the handle
/// to `verify_stream_block_step` and `close_stream_hmac`. Supports concurrent streams.
///
/// Uses `StreamHmacState` which matches the server's wire-format implementation:
/// 16-byte truncated MACs with `topic.as_bytes()` as the initial chain state.
///
/// # Arguments
///
/// * `mac_key` - 32 bytes MAC key from derive_stream_keys
/// * `topic` - Hex topic string (64 chars) used as the initial chain state
#[wasm_bindgen]
pub fn init_stream_hmac(mac_key: &[u8], topic: &str) -> Result<u32, JsError> {
    use crate::crypto::StreamHmacState;

    let key: [u8; 32] = mac_key
        .try_into()
        .map_err(|_| JsError::new("mac_key must be 32 bytes"))?;
    let handle = NEXT_HMAC_HANDLE.with(|h| {
        let id = h.get();
        h.set(id + 1);
        id
    });
    STREAM_HMAC_TABLE.with(|t| {
        t.borrow_mut().insert(handle, StreamHmacState::new(key, topic.to_owned()));
    });
    Ok(handle)
}

/// Verify the next stream block in the stateful HMAC chain.
///
/// Uses constant-time comparison via `subtle::ConstantTimeEq` to prevent timing
/// attacks. Advances the chain state only on success.
///
/// # Arguments
///
/// * `handle` - HMAC chain handle from init_stream_hmac
/// * `capnp_data` - Cap'n Proto StreamBlock bytes (WITHOUT the trailing MAC)
/// * `mac` - 16-byte truncated MAC extracted from the WebTransport message
///
/// # Returns
///
/// true if MAC verification passes; false if MAC is wrong (stream integrity violation)
#[wasm_bindgen]
pub fn verify_stream_block_step(handle: u32, capnp_data: &[u8], mac: &[u8]) -> Result<bool, JsError> {
    STREAM_HMAC_TABLE.with(|t| {
        let mut table = t.borrow_mut();
        let hmac = table
            .get_mut(&handle)
            .ok_or_else(|| JsError::new(&format!("stream HMAC handle {} not found; call init_stream_hmac first", handle)))?;
        Ok(hmac.verify_next(capnp_data, mac))
    })
}

/// Close a stream HMAC chain, zeroing key material.
///
/// Call when the stream is complete or cancelled to clean up state.
#[wasm_bindgen]
pub fn close_stream_hmac(handle: u32) {
    STREAM_HMAC_TABLE.with(|t| {
        t.borrow_mut().remove(&handle);
        // StreamHmacState drops; key is wrapped in Zeroizing<[u8; 32]> and zeroed automatically
    });
}

// ============================================================================
// ECDH key exchange
// ============================================================================

/// Compute ECDH shared secret: Ristretto255 scalar * point.
///
/// # Arguments
///
/// * `client_privkey` - 32 bytes Ristretto255 scalar (from generate_ephemeral_keypair)
/// * `server_pubkey` - 32 bytes Ristretto255 compressed point
///
/// # Returns
///
/// 32 bytes shared secret
#[wasm_bindgen]
pub fn ecdh_ristretto(
    client_privkey: &[u8],
    server_pubkey: &[u8],
) -> Result<Vec<u8>, JsError> {
    use crate::crypto::key_exchange::DefaultKeyExchange;
    use crate::crypto::KeyExchange;

    if client_privkey.len() != 32 {
        return Err(JsError::new("client_privkey must be 32 bytes"));
    }
    if server_pubkey.len() != 32 {
        return Err(JsError::new("server_pubkey must be 32 bytes"));
    }

    // Reconstruct secret from scalar bytes
    let scalar = curve25519_dalek::scalar::Scalar::from_canonical_bytes(
        client_privkey.try_into().unwrap()
    );
    if scalar.is_none().into() {
        return Err(JsError::new("invalid Ristretto255 scalar"));
    }
    let secret = crate::crypto::key_exchange::RistrettoSecret::from_scalar(scalar.unwrap())
        .ok_or_else(|| JsError::new("Ristretto255 scalar is zero"))?;

    let public = DefaultKeyExchange::pubkey_from_bytes(server_pubkey)
        .map_err(|e| JsError::new(&format!("invalid server pubkey: {}", e)))?;

    let shared = DefaultKeyExchange::derive_shared(&secret, &public)
        .map_err(|e| JsError::new(&format!("ECDH failed: {}", e)))?;

    Ok(shared.as_bytes().to_vec())
}

// ============================================================================
// Envelope verification
// ============================================================================

/// Register a peer's ML-DSA-65 verifying key bound to their Ed25519 identity.
///
/// Once registered, `verify_signed_envelope` enforces Hybrid policy for
/// envelopes from this signer — the outer ML-DSA-65 layer is required and
/// stripping it causes verification to fail (closes #158).
///
/// Call this with both keys from the peer's hybrid key bundle (fetched from
/// JWKS or the key registry at startup). Replaces any prior binding for the
/// same Ed25519 pubkey.
///
/// # Arguments
///
/// * `ed25519_pubkey` - 32-byte Ed25519 verifying key
/// * `ml_dsa_vk` - ML-DSA-65 verifying key bytes (~1952 bytes)
#[wasm_bindgen]
pub fn register_pq_trust(ed25519_pubkey: &[u8], ml_dsa_vk: &[u8]) -> Result<(), JsError> {
    if ed25519_pubkey.len() != 32 {
        return Err(JsError::new("ed25519_pubkey must be 32 bytes"));
    }
    // Validate ML-DSA key before storing (fails fast on malformed input).
    crate::crypto::pq::ml_dsa_vk_from_bytes(ml_dsa_vk)
        .map_err(|e| JsError::new(&format!("invalid ML-DSA verifying key: {e}")))?;

    let mut ed = [0u8; 32];
    ed.copy_from_slice(ed25519_pubkey);
    WASM_PQ_BINDINGS.with(|bindings| {
        bindings.borrow_mut().insert(ed, ml_dsa_vk.to_vec());
    });
    Ok(())
}

/// Remove a peer's ML-DSA-65 binding. After this call, envelopes from that
/// signer are accepted under Classical (EdDSA-only) policy.
///
/// # Arguments
///
/// * `ed25519_pubkey` - 32-byte Ed25519 verifying key to unregister
#[wasm_bindgen]
pub fn unregister_pq_trust(ed25519_pubkey: &[u8]) -> Result<(), JsError> {
    if ed25519_pubkey.len() != 32 {
        return Err(JsError::new("ed25519_pubkey must be 32 bytes"));
    }
    let mut ed = [0u8; 32];
    ed.copy_from_slice(ed25519_pubkey);
    WASM_PQ_BINDINGS.with(|bindings| {
        bindings.borrow_mut().remove(&ed);
    });
    Ok(())
}

/// Clear all registered PQ trust bindings.
///
/// After this call, all signers fall back to Classical (EdDSA-only) policy.
#[wasm_bindgen]
pub fn clear_pq_trust() {
    WASM_PQ_BINDINGS.with(|bindings| bindings.borrow_mut().clear());
}

/// Verify a signed envelope (for browser-side services receiving requests from server).
///
/// When the signer's ML-DSA-65 key has been registered via `register_pq_trust`,
/// Hybrid policy is enforced: the outer ML-DSA-65 layer is required and an
/// envelope that omits it is rejected. For signers without a registered ML-DSA
/// key, Classical (EdDSA-only) policy applies — documented WNS until the
/// browser trust store is provisioned with the peer's PQ key bundle.
///
/// # Arguments
///
/// * `envelope_bytes` - Cap'n Proto serialized SignedEnvelope
/// * `expected_signer_pubkey` - 32-byte Ed25519 verifying key
///
/// # Returns
///
/// Payload bytes if verification passes (signature + nonce + timestamp valid)
#[wasm_bindgen]
pub fn verify_signed_envelope(
    envelope_bytes: &[u8],
    expected_signer_pubkey: &[u8],
) -> Result<Vec<u8>, JsError> {
    use crate::crypto::signing::verifying_key_from_bytes;
    use crate::envelope::{unwrap_and_verify, KeyedPqTrustStore, UnwrapOptions};

    if expected_signer_pubkey.len() != 32 {
        return Err(JsError::new("expected_signer_pubkey must be 32 bytes"));
    }

    let mut pubkey_bytes = [0u8; 32];
    pubkey_bytes.copy_from_slice(expected_signer_pubkey);
    let verifying_key = verifying_key_from_bytes(&pubkey_bytes)
        .map_err(|e| JsError::new(&format!("invalid verifying key: {}", e)))?;

    let nonce_cache = WASM_NONCE_CACHE.with(|c| c.clone());

    // Build a local PqTrustStore from WASM_PQ_BINDINGS. Track whether the
    // specific signer has a registered ML-DSA binding; only in that case do we
    // upgrade to Hybrid (enforcing the outer SNS layer). Signers without a
    // registered key continue under Classical — documented WNS until their PQ
    // key is provisioned in the browser trust store. (#158)
    let mut pq_store = KeyedPqTrustStore::new();
    let signer_has_pq = WASM_PQ_BINDINGS.with(|bindings| {
        let b = bindings.borrow();
        let mut found = false;
        for (ed_key, ml_dsa_bytes) in b.iter() {
            if let Ok(ml_dsa_vk) = crate::crypto::pq::ml_dsa_vk_from_bytes(ml_dsa_bytes) {
                pq_store.bind(*ed_key, &ml_dsa_vk);
                if ed_key == &pubkey_bytes {
                    found = true;
                }
            }
        }
        found
    });

    let opts = UnwrapOptions::fixed_signer(&verifying_key, &*nonce_cache);
    let opts = if signer_has_pq {
        // Hybrid: outer ML-DSA-65 required for this signer (policy default is already Hybrid).
        opts.with_pq_store(&pq_store)
    } else {
        // No PQ binding: classical fallback (documented WNS for unprovisioned signers).
        opts.classical()
    };

    let (_signed, payload) = unwrap_and_verify(envelope_bytes, &opts)
        .map_err(|e| JsError::new(&format!("envelope verification failed: {}", e)))?;

    Ok(payload)
}

// ============================================================================
// Utilities
// ============================================================================

/// Get current timestamp in Unix milliseconds.
#[wasm_bindgen]
pub fn current_timestamp_ms() -> i64 {
    crate::envelope::current_timestamp()
}

/// Generate a random 16-byte nonce.
#[wasm_bindgen]
pub fn generate_nonce() -> Vec<u8> {
    crate::envelope::generate_nonce().to_vec()
}

// ============================================================================
// ZMTP 3.1 framing
// ============================================================================

/// Build a 64-byte ZMTP 3.1 greeting (NULL mechanism).
#[wasm_bindgen]
pub fn zmtp_greeting() -> Vec<u8> {
    crate::zmtp_framing::build_greeting().to_vec()
}

/// Validate a 64-byte ZMTP 3.1 greeting from the peer.
#[wasm_bindgen]
pub fn zmtp_validate_greeting(greeting: &[u8]) -> Result<(), JsError> {
    crate::zmtp_framing::validate_greeting(greeting)
        .map_err(|e| JsError::new(&format!("{}", e)))
}

/// Build a ZMTP READY command frame for the given socket type.
///
/// # Arguments
///
/// * `socket_type` - One of: "REQ", "REP", "DEALER", "ROUTER", "PUB", "SUB", "PUSH", "PULL", "PAIR"
///
/// # Returns
///
/// Encoded ZMTP command frame bytes (ready to send on the wire)
#[wasm_bindgen]
pub fn zmtp_ready_command(socket_type: &str) -> Result<Vec<u8>, JsError> {
    use crate::zmtp_framing::ZmqSocketType;

    let st = match socket_type {
        "REQ" => ZmqSocketType::Req,
        "REP" => ZmqSocketType::Rep,
        "PUB" => ZmqSocketType::Pub,
        "SUB" => ZmqSocketType::Sub,
        "XPUB" => ZmqSocketType::XPub,
        "XSUB" => ZmqSocketType::XSub,
        "PUSH" => ZmqSocketType::Push,
        "PULL" => ZmqSocketType::Pull,
        _ => return Err(JsError::new(&format!("unknown socket type: {}", socket_type))),
    };

    let metadata = crate::zmtp_framing::build_ready_metadata(st);
    Ok(crate::zmtp_framing::encode_command("READY", &metadata))
}

/// Encode a single ZMTP data frame.
///
/// # Arguments
///
/// * `more` - true if more frames follow in this message
/// * `data` - frame payload bytes
///
/// # Returns
///
/// Wire-encoded ZMTP frame bytes
#[wasm_bindgen]
pub fn zmtp_encode_frame(more: bool, data: &[u8]) -> Vec<u8> {
    crate::zmtp_framing::encode_frame(more, false, data)
}

/// Encode a ZMTP multipart message from flat-encoded parts.
///
/// Flat encoding: `[4B-len-BE, data, 4B-len-BE, data, ...]`
///
/// # Returns
///
/// Wire-encoded ZMTP frames (each part as a frame, last frame has more=false)
#[wasm_bindgen]
pub fn zmtp_encode_multipart(parts_flat: &[u8]) -> Result<Vec<u8>, JsError> {
    crate::zmtp_framing::encode_multipart_flat(parts_flat)
        .map_err(|e| JsError::new(&format!("{}", e)))
}

/// Decode ZMTP frames from a buffer into flat-encoded parts.
///
/// # Returns
///
/// Flat encoding: `[4B-len-BE, data, 4B-len-BE, data, ...]`
///
/// Only data frames are returned (command frames are skipped).
#[wasm_bindgen]
pub fn zmtp_decode_frames(buf: &[u8]) -> Result<Vec<u8>, JsError> {
    crate::zmtp_framing::decode_frames_to_flat(buf)
        .map_err(|e| JsError::new(&format!("{}", e)))
}

/// Decode a ZMTP command frame.
///
/// # Returns
///
/// Flat encoding: `[1B-name-len, name-bytes, remaining-body-bytes]`
#[wasm_bindgen]
pub fn zmtp_decode_command(buf: &[u8]) -> Result<Vec<u8>, JsError> {
    // First decode as a frame
    let (frame, _consumed) = crate::zmtp_framing::decode_frame(buf)
        .map_err(|e| JsError::new(&format!("frame decode: {}", e)))?;

    if !frame.command {
        return Err(JsError::new("not a command frame"));
    }

    // Parse the command from frame data
    let cmd = crate::zmtp_framing::ZmtpCommand::parse(&frame.data)
        .map_err(|e| JsError::new(&format!("command parse: {}", e)))?;

    // Return: [name_len(1), name_bytes..., body_bytes...]
    let name_bytes = cmd.name.as_bytes();
    let mut result = Vec::with_capacity(1 + name_bytes.len() + cmd.body.len());
    result.push(name_bytes.len() as u8);
    result.extend_from_slice(name_bytes);
    result.extend_from_slice(&cmd.body);

    Ok(result)
}

// ============================================================================
// Notification broadcast encryption
// ============================================================================

/// Derive notification encryption keys from a DH shared secret.
///
/// # Arguments
///
/// * `shared_secret` - 32 bytes from blinded_dh or ecdh_ristretto
/// * `publisher_pub` - 32 bytes publisher's Ristretto255 pubkey
/// * `subscriber_pub` - 32 bytes subscriber's (blinded) Ristretto255 pubkey
///
/// # Returns
///
/// 64 bytes: [enc_key(32) || mac_key(32)]
#[wasm_bindgen]
pub fn derive_notification_keys(
    shared_secret: &[u8],
    publisher_pub: &[u8],
    subscriber_pub: &[u8],
) -> Result<Vec<u8>, JsError> {
    if shared_secret.len() != 32 {
        return Err(JsError::new("shared_secret must be 32 bytes"));
    }
    if publisher_pub.len() != 32 {
        return Err(JsError::new("publisher_pub must be 32 bytes"));
    }
    if subscriber_pub.len() != 32 {
        return Err(JsError::new("subscriber_pub must be 32 bytes"));
    }

    let mut ss = [0u8; 32];
    ss.copy_from_slice(shared_secret);
    let mut pp = [0u8; 32];
    pp.copy_from_slice(publisher_pub);
    let mut sp = [0u8; 32];
    sp.copy_from_slice(subscriber_pub);

    let keys = crate::crypto::key_exchange::derive_notification_keys(&ss, &pp, &sp)
        .map_err(|e| JsError::new(&format!("notification key derivation failed: {}", e)))?;

    let mut result = Vec::with_capacity(64);
    result.extend_from_slice(&*keys.enc_key);
    result.extend_from_slice(&*keys.mac_key);
    Ok(result)
}

/// Encrypt with AES-256-GCM.
///
/// # Arguments
///
/// * `key` - 32-byte AES-256 key
/// * `nonce` - 12-byte random nonce (must be from OsRng, never reused)
/// * `plaintext` - Data to encrypt
/// * `aad` - Associated authenticated data (from notification_build_payload_aad)
///
/// # Returns
///
/// Ciphertext bytes (plaintext.len() + 16 bytes GCM tag)
#[wasm_bindgen]
pub fn aes_gcm_encrypt(
    key: &[u8],
    nonce: &[u8],
    plaintext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, JsError> {
    use aes_gcm::{aead::{Aead, KeyInit, Payload}, Aes256Gcm, Nonce};

    if key.len() != 32 {
        return Err(JsError::new("key must be 32 bytes"));
    }
    if nonce.len() != 12 {
        return Err(JsError::new("nonce must be 12 bytes"));
    }

    let cipher = Aes256Gcm::new(key.into());
    let payload = Payload { msg: plaintext, aad };
    cipher
        .encrypt(Nonce::from_slice(nonce), payload)
        .map_err(|_| JsError::new("AES-GCM encrypt failed"))
}

/// Decrypt with AES-256-GCM.
///
/// # Arguments
///
/// * `key` - 32-byte AES-256 key
/// * `nonce` - 12-byte nonce used during encryption
/// * `ciphertext` - Data to decrypt (includes GCM tag)
/// * `aad` - Associated authenticated data (must match encryption AAD)
///
/// # Returns
///
/// Plaintext bytes
#[wasm_bindgen]
pub fn aes_gcm_decrypt(
    key: &[u8],
    nonce: &[u8],
    ciphertext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, JsError> {
    use aes_gcm::{aead::{Aead, KeyInit, Payload}, Aes256Gcm, Nonce};

    if key.len() != 32 {
        return Err(JsError::new("key must be 32 bytes"));
    }
    if nonce.len() != 12 {
        return Err(JsError::new("nonce must be 12 bytes"));
    }

    let cipher = Aes256Gcm::new(key.into());
    let payload = Payload { msg: ciphertext, aad };
    cipher
        .decrypt(Nonce::from_slice(nonce), payload)
        .map_err(|_| JsError::new("AES-GCM decrypt failed (wrong key, nonce, or AAD)"))
}

/// Blinding-aware ECDH: `(subscriber_secret + blinding_scalar) * publisher_pubkey`.
///
/// Used by notification subscribers to derive the same shared secret as the publisher
/// who encrypted against a blinded pubkey.
///
/// # Arguments
///
/// * `subscriber_secret` - 32-byte Ristretto255 scalar
/// * `blinding_scalar` - 32-byte blinding scalar `r` from NotificationBlock
/// * `publisher_pubkey` - 32-byte publisher's Ristretto255 pubkey
///
/// # Returns
///
/// 32-byte shared secret
#[wasm_bindgen]
pub fn notification_blinded_dh(
    subscriber_secret: &[u8],
    blinding_scalar: &[u8],
    publisher_pubkey: &[u8],
) -> Result<Vec<u8>, JsError> {
    if subscriber_secret.len() != 32 {
        return Err(JsError::new("subscriber_secret must be 32 bytes"));
    }
    if blinding_scalar.len() != 32 {
        return Err(JsError::new("blinding_scalar must be 32 bytes"));
    }
    if publisher_pubkey.len() != 32 {
        return Err(JsError::new("publisher_pubkey must be 32 bytes"));
    }

    let mut ss = [0u8; 32];
    ss.copy_from_slice(subscriber_secret);
    let mut bs = [0u8; 32];
    bs.copy_from_slice(blinding_scalar);
    let mut pp = [0u8; 32];
    pp.copy_from_slice(publisher_pubkey);

    let shared = crate::crypto::blinded_dh_raw(&ss, &bs, &pp)
        .map_err(|e| JsError::new(&format!("blinded DH failed: {}", e)))?;

    Ok(shared.to_vec())
}

/// Reconstruct a blinded pubkey: `subscriber_pubkey + blinding_scalar * G`.
///
/// Used by subscribers to reconstruct the blinded pubkey for fingerprint
/// computation and key derivation.
///
/// # Arguments
///
/// * `subscriber_pubkey` - 32-byte subscriber's Ristretto255 pubkey
/// * `blinding_scalar` - 32-byte blinding scalar `r` from NotificationBlock
///
/// # Returns
///
/// 32-byte blinded pubkey (compressed Ristretto point)
#[wasm_bindgen]
pub fn notification_reconstruct_blinded_pub(
    subscriber_pubkey: &[u8],
    blinding_scalar: &[u8],
) -> Result<Vec<u8>, JsError> {
    if subscriber_pubkey.len() != 32 {
        return Err(JsError::new("subscriber_pubkey must be 32 bytes"));
    }
    if blinding_scalar.len() != 32 {
        return Err(JsError::new("blinding_scalar must be 32 bytes"));
    }

    let mut sp = [0u8; 32];
    sp.copy_from_slice(subscriber_pubkey);
    let mut bs = [0u8; 32];
    bs.copy_from_slice(blinding_scalar);

    let blinded = crate::crypto::reconstruct_blinded_pub_raw(&sp, &bs)
        .map_err(|e| JsError::new(&format!("reconstruct blinded pub failed: {}", e)))?;

    Ok(blinded.to_vec())
}

/// Compute a one-shot MAC: `keyed_mac(mac_key, data)`.
///
/// Uses Blake3 keyed hash (or HMAC-SHA256 in FIPS mode).
///
/// # Arguments
///
/// * `mac_key` - 32-byte MAC key (from derive_notification_keys)
/// * `data` - Data to authenticate (typically the encrypted payload)
///
/// # Returns
///
/// 32-byte MAC
#[wasm_bindgen]
pub fn notification_mac(mac_key: &[u8], data: &[u8]) -> Result<Vec<u8>, JsError> {
    if mac_key.len() != 32 {
        return Err(JsError::new("mac_key must be 32 bytes"));
    }

    let mut key = [0u8; 32];
    key.copy_from_slice(mac_key);
    let mac = crate::crypto::notification::notification_mac(&key, data);
    Ok(mac.to_vec())
}

/// Verify a one-shot MAC in constant time.
///
/// # Arguments
///
/// * `mac_key` - 32-byte MAC key
/// * `data` - Authenticated data
/// * `expected_mac` - 32-byte expected MAC value
///
/// # Returns
///
/// `true` if MAC is valid, `false` otherwise. Never errors on mismatch.
#[wasm_bindgen]
pub fn notification_mac_verify(
    mac_key: &[u8],
    data: &[u8],
    expected_mac: &[u8],
) -> Result<bool, JsError> {
    if mac_key.len() != 32 {
        return Err(JsError::new("mac_key must be 32 bytes"));
    }
    if expected_mac.len() != 32 {
        return Err(JsError::new("expected_mac must be 32 bytes"));
    }

    let mut key = [0u8; 32];
    key.copy_from_slice(mac_key);
    let mut mac = [0u8; 32];
    mac.copy_from_slice(expected_mac);

    Ok(crate::crypto::notification::notification_mac_verify(&key, data, &mac).is_ok())
}

/// Build length-prefixed AAD for notification payload encryption/decryption.
///
/// Format: `u32_le(len(intent_id)) || intent_id || u32_le(len(scope)) || scope`
///
/// Must be called with matching `intent_id` and `scope` on both encrypt and decrypt sides.
///
/// # Arguments
///
/// * `intent_id` - Intent ID string from publishIntent response
/// * `scope` - Scope string (e.g., "serve:model:qwen3")
///
/// # Returns
///
/// AAD bytes for use with aes_gcm_encrypt/decrypt
#[wasm_bindgen]
pub fn notification_build_payload_aad(intent_id: &str, scope: &str) -> Vec<u8> {
    crate::crypto::notification::build_payload_aad(intent_id, scope)
}

/// Compute a 128-bit pubkey fingerprint: `Blake3(pubkey)[..16]`.
///
/// Used for capsule routing — matches the fingerprint format used by NotificationService.
///
/// # Arguments
///
/// * `pubkey` - 32-byte Ristretto255 pubkey (typically the blinded pubkey)
///
/// # Returns
///
/// 16-byte fingerprint
#[wasm_bindgen]
pub fn notification_pubkey_fingerprint(pubkey: &[u8]) -> Result<Vec<u8>, JsError> {
    if pubkey.len() != 32 {
        return Err(JsError::new("pubkey must be 32 bytes"));
    }
    let mut pk = [0u8; 32];
    pk.copy_from_slice(pubkey);
    Ok(crate::crypto::notification::pubkey_fingerprint(&pk).to_vec())
}

/// Build the Ed25519 attestation message for publisher identity verification.
///
/// Message format: `ephemeral_pubkey(32) || blinded_sub_pubkey(32) || u32_le(scope_len) || scope || u32_le(intent_len) || intent_id`
///
/// # Arguments
///
/// * `ephemeral_pubkey` - Publisher's 32-byte ephemeral Ristretto pubkey
/// * `blinded_sub_pubkey` - 32-byte blinded subscriber pubkey
/// * `scope` - Claim scope string
/// * `intent_id` - Intent ID string
///
/// # Returns
///
/// Message bytes suitable for Ed25519 sign/verify
#[wasm_bindgen]
pub fn notification_build_attestation_message(
    ephemeral_pubkey: &[u8],
    blinded_sub_pubkey: &[u8],
    scope: &str,
    intent_id: &str,
) -> Result<Vec<u8>, JsError> {
    if ephemeral_pubkey.len() != 32 {
        return Err(JsError::new("ephemeral_pubkey must be 32 bytes"));
    }
    if blinded_sub_pubkey.len() != 32 {
        return Err(JsError::new("blinded_sub_pubkey must be 32 bytes"));
    }

    let mut ep = [0u8; 32];
    ep.copy_from_slice(ephemeral_pubkey);
    let mut bp = [0u8; 32];
    bp.copy_from_slice(blinded_sub_pubkey);

    Ok(crate::crypto::notification::build_attestation_message(&ep, &bp, scope, intent_id))
}

/// Sign an Ed25519 attestation for publisher identity binding.
///
/// Signs the attestation message (from notification_build_attestation_message)
/// with the publisher's Ed25519 signing key.
///
/// # Arguments
///
/// * `privkey_seed` - 32-byte Ed25519 seed (from generate_signing_keypair)
/// * `message` - Attestation message bytes (from notification_build_attestation_message)
///
/// # Returns
///
/// 64-byte Ed25519 signature
#[wasm_bindgen]
pub fn ed25519_sign(privkey_seed: &[u8], message: &[u8]) -> Result<Vec<u8>, JsError> {
    use ed25519_dalek::{Signer, SigningKey};

    if privkey_seed.len() != 32 {
        return Err(JsError::new("privkey_seed must be 32 bytes"));
    }

    let mut seed = [0u8; 32];
    seed.copy_from_slice(privkey_seed);
    let signing_key = SigningKey::from_bytes(&seed);
    let sig = signing_key.sign(message);
    Ok(sig.to_bytes().to_vec())
}

/// Verify an Ed25519 signature.
///
/// # Arguments
///
/// * `pubkey` - 32-byte Ed25519 verifying key
/// * `message` - Message bytes that were signed
/// * `signature` - 64-byte Ed25519 signature
///
/// # Returns
///
/// `true` if signature is valid
#[wasm_bindgen]
pub fn ed25519_verify(
    pubkey: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<bool, JsError> {
    use ed25519_dalek::{Signature, Verifier, VerifyingKey};

    if pubkey.len() != 32 {
        return Err(JsError::new("pubkey must be 32 bytes"));
    }
    if signature.len() != 64 {
        return Err(JsError::new("signature must be 64 bytes"));
    }

    let vk = VerifyingKey::from_bytes(pubkey.try_into().unwrap())
        .map_err(|e| JsError::new(&format!("invalid verifying key: {}", e)))?;
    let sig = Signature::from_bytes(signature.try_into().unwrap());

    Ok(vk.verify(message, &sig).is_ok())
}

/// Generate a random 12-byte AES-GCM nonce from OsRng.
///
/// Every AES-GCM encryption MUST use a fresh random nonce. Never reuse nonces.
#[wasm_bindgen]
pub fn generate_aes_nonce() -> Vec<u8> {
    let mut nonce = [0u8; 12];
    rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut nonce);
    nonce.to_vec()
}

