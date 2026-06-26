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
    /// Fingerprint string is malformed (empty, contains ':' delimiter conflict, etc.).
    InvalidFingerprint,
    /// Signature is not valid base64.
    InvalidSignatureEncoding,
    /// Signature is not exactly 64 bytes (88 base64 chars).
    InvalidSignatureLength,
    /// No UserStore entry for this username.
    UserNotFound,
    /// User has no pubkeys registered.
    NoPubkeys,
    /// No user is registered with the supplied key fingerprint.
    UnknownFingerprint,
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
            ChallengeError::InvalidFingerprint =>
                "Invalid key fingerprint format. Expected `SHA256:...` from `hyprstream sign-challenge`.",
            ChallengeError::InvalidSignatureEncoding =>
                "Invalid signature encoding (expected base64).",
            ChallengeError::InvalidSignatureLength =>
                "Invalid signature length (expected 64 bytes / 88 base64 chars).",
            ChallengeError::UserNotFound =>
                "Unknown user. Please contact your administrator.",
            ChallengeError::NoPubkeys =>
                "No public keys registered for this user.",
            ChallengeError::UnknownFingerprint =>
                "Key fingerprint not registered to any user. Run `hyprstream wizard` or have an admin register your key.",
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

/// Verify an Ed25519 challenge-response identified by the signer's key
/// fingerprint rather than a client-declared username.
///
/// Resolves the user via the `pubkey:{fingerprint} → username` reverse index
/// the UserStore already maintains. Builds the challenge string as
/// `"{fingerprint}:{nonce}:{code_challenge}"` so the client and server agree
/// on the binding without the client ever asserting an identity.
///
/// Returns `(username, VerifyingKey)` on success: the resolved username (for
/// the JWT `sub` claim) and the key that verified the signature.
pub(super) async fn verify_ed25519_response_by_fingerprint(
    user_store: &dyn UserStore,
    fingerprint: &str,
    challenge: &str,
    sig_b64: &str,
) -> Result<(String, ed25519_dalek::VerifyingKey), ChallengeError> {
    // Accept only the canonical `SHA256:...` form. The client and server
    // build the challenge with the exact same string, so additional ':' inside
    // the fingerprint don't cause delimiter ambiguity — but rejecting other
    // shapes early gives a better error than "unknown fingerprint".
    if !fingerprint.starts_with("SHA256:") {
        return Err(ChallengeError::InvalidFingerprint);
    }

    let sig_bytes = STANDARD.decode(sig_b64)
        .map_err(|_| ChallengeError::InvalidSignatureEncoding)?;
    let sig_array: [u8; 64] = sig_bytes.try_into()
        .map_err(|_| ChallengeError::InvalidSignatureLength)?;
    let signature = ed25519_dalek::Signature::from_bytes(&sig_array);

    let Some(username) = user_store.get_pubkey_user(fingerprint).await
        .map_err(ChallengeError::UserStoreError)?
    else {
        return Err(ChallengeError::UnknownFingerprint);
    };

    let pubkeys = user_store.list_pubkeys(&username).await
        .map_err(ChallengeError::UserStoreError)?;
    let Some(entry) = pubkeys.into_iter().find(|e| e.fingerprint == fingerprint) else {
        // Reverse index pointed at this user but the key is gone — treat as unknown.
        return Err(ChallengeError::UnknownFingerprint);
    };

    if entry.pubkey.verify_strict(challenge.as_bytes(), &signature).is_err() {
        return Err(ChallengeError::SignatureInvalid);
    }

    let _ = user_store.touch_pubkey(&username, &entry.fingerprint).await;
    Ok((username, entry.pubkey))
}

/// HTML-escape a string for safe embedding in HTML attributes and text.
pub(crate) fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}
