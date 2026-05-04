//! Shared Ed25519 challenge-response verification for OAuth endpoints.
//!
//! Used by both the device flow (`device.rs`) and the authorization code
//! flow (`authorize.rs`) to verify that a user possesses the private key
//! corresponding to the public key registered in their UserStore entry.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use crate::auth::user_store::UserStore;

/// Error types for Ed25519 challenge-response verification.
#[allow(dead_code)]
pub(super) enum ChallengeError {
    /// Username contains ':', which would make the challenge string ambiguous.
    InvalidUsername,
    /// Signature is not valid base64.
    InvalidSignatureEncoding,
    /// Signature is not exactly 64 bytes (88 base64 chars).
    InvalidSignatureLength,
    /// No UserStore entry for this username.
    UserNotFound,
    /// User has no pubkeys registered.
    NoPubkeys,
    /// UserStore returned an error.
    UserStoreError(#[allow(dead_code)] anyhow::Error),
    /// Signature is syntactically valid but does not verify against any registered key.
    SignatureInvalid,
}

impl ChallengeError {
    /// Human-readable error message for display in HTML forms.
    pub(super) fn message(&self) -> &'static str {
        match self {
            ChallengeError::InvalidUsername => "Username must not contain ':'",
            ChallengeError::InvalidSignatureEncoding =>
                "Invalid signature encoding (expected base64).",
            ChallengeError::InvalidSignatureLength =>
                "Invalid signature length (expected 64 bytes / 88 base64 chars).",
            ChallengeError::UserNotFound =>
                "Unknown user. Please contact your administrator.",
            ChallengeError::NoPubkeys =>
                "No public keys registered for this user.",
            ChallengeError::UserStoreError(_) =>
                "Internal error looking up user credentials.",
            ChallengeError::SignatureInvalid =>
                "Invalid signature. Ensure you signed the correct challenge string.",
        }
    }
}

/// Verify an Ed25519 challenge-response.
///
/// The `challenge` parameter is the fully-constructed challenge string
/// (caller is responsible for constructing it correctly for the flow):
/// - Device flow:    `"{username}:{user_code_normalized}:{nonce}"`
/// - Auth code flow: `"{username}:{nonce}:{code_challenge}"`
///
/// `sig_b64` is the standard (non-URL-safe) base64-encoded 64-byte signature.
///
/// Returns the verified `VerifyingKey` on success — the key confirmed to have
/// signed the challenge. Tries all registered pubkeys for the user until one matches.
pub(super) async fn verify_ed25519_response(
    user_store: &dyn UserStore,
    username: &str,
    challenge: &str,
    sig_b64: &str,
) -> Result<ed25519_dalek::VerifyingKey, ChallengeError> {
    if username.contains(':') {
        return Err(ChallengeError::InvalidUsername);
    }

    let sig_bytes = STANDARD.decode(sig_b64)
        .map_err(|_| ChallengeError::InvalidSignatureEncoding)?;

    let sig_array: [u8; 64] = sig_bytes.try_into()
        .map_err(|_| ChallengeError::InvalidSignatureLength)?;

    let signature = ed25519_dalek::Signature::from_bytes(&sig_array);

    // Get all pubkeys for the user and try each one
    let pubkeys = user_store.list_pubkeys(username).await
        .map_err(ChallengeError::UserStoreError)?;

    if pubkeys.is_empty() {
        return Err(ChallengeError::NoPubkeys);
    }

    // Try to verify against each registered pubkey
    for entry in &pubkeys {
        if entry.pubkey.verify_strict(challenge.as_bytes(), &signature).is_ok() {
            // Optionally touch the pubkey to update last_used_at
            let _ = user_store.touch_pubkey(username, &entry.fingerprint).await;
            return Ok(entry.pubkey);
        }
    }

    Err(ChallengeError::SignatureInvalid)
}

/// HTML-escape a string for safe embedding in HTML attributes and text.
pub(crate) fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}
