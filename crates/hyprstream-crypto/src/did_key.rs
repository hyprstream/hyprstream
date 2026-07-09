//! Shared `did:key` (Ed25519) ⇄ raw-key codec.
//!
//! This is the **single source of truth** for the `ed25519-pub` multicodec
//! prefix, the base58btc multibase encoding, and DID-URL fragment/query
//! stripping used to convert between a raw 32-byte Ed25519 public key and its
//! `did:key:z6Mk…` / Multikey `publicKeyMultibase` forms.
//!
//! It is compiled on **all targets** (unlike `did:web` resolution, which is
//! native-only, and `iroh_peer`, which is wasm32-only) precisely so the native
//! `did:web` admission resolver and the wasm32 `iroh_peer` browser identity
//! helpers share one implementation and can never drift (#281/#475).
//! `hyprstream_rpc::did_web` re-exports `decode_ed25519_multikey`,
//! `did_key_to_ed25519`, and `ed25519_to_did_key` for source compatibility with
//! existing callers.
//!
//! This module is the **single canonical home** for the multicodec constants
//! ([`MULTICODEC_ED25519_PUB`], [`MULTICODEC_ML_DSA_65_PUB`]) and the generic
//! [`decode_multikey`] helper (#916); previously these were duplicated in
//! `hyprstream::auth::mesh_trust` and `hyprstream-rpc`'s `service_entry`.

use anyhow::{anyhow, bail, Result};

/// Multicodec `ed25519-pub` unsigned-varint prefix (`0xed01` → bytes `0xed 0x01`).
///
/// This is the one canonical definition every `did:key` / Multikey codepath
/// shares across the workspace.
pub const MULTICODEC_ED25519_PUB: [u8; 2] = [0xed, 0x01];

/// Multicodec `ml-dsa-65-pub` unsigned-varint prefix (`0x1211` → bytes `0x91 0x24`).
pub const MULTICODEC_ML_DSA_65_PUB: [u8; 2] = [0x91, 0x24];

/// Decode a `Multikey` `publicKeyMultibase` string into raw key bytes, verifying
/// the base58btc multibase prefix (`z`) and the expected multicodec header.
///
/// Returns the payload with the multicodec prefix stripped (variable length; the
/// caller validates the key length for its algorithm). For the fixed 32-byte
/// Ed25519 case, prefer [`decode_ed25519_multikey`].
///
/// Returns `Err` for a wrong multibase, an undecodable base58, or a multicodec
/// prefix that does not match `expected_codec`.
pub fn decode_multikey(multibase: &str, expected_codec: &[u8; 2]) -> Result<Vec<u8>> {
    let body = multibase
        .strip_prefix('z')
        .ok_or_else(|| anyhow!("Multikey must use base58btc multibase ('z') prefix"))?;
    let decoded = bs58::decode(body)
        .into_vec()
        .map_err(|e| anyhow!("invalid base58btc Multikey: {e}"))?;
    if decoded.len() < 2 || &decoded[..2] != expected_codec {
        bail!(
            "unexpected multicodec prefix (expected {expected_codec:02x?}, got {:02x?})",
            decoded.get(..2).unwrap_or(&decoded)
        );
    }
    Ok(decoded[2..].to_vec())
}

/// Decode a `Multikey` `publicKeyMultibase` string into raw Ed25519 key bytes.
///
/// Verifies the base58btc multibase prefix (`z`) and the `ed25519-pub` multicodec
/// header, returning the 32-byte payload.
///
/// Returns `Err` for a wrong multibase, an undecodable base58, a non-Ed25519
/// codec, or a payload that is not exactly 32 bytes.
pub fn decode_ed25519_multikey(multibase: &str) -> Result<[u8; 32]> {
    let payload = decode_multikey(multibase, &MULTICODEC_ED25519_PUB)?;
    let len = payload.len();
    payload
        .try_into()
        .map_err(|_| anyhow!("ed25519 Multikey payload is {len} bytes (expected 32)"))
}

/// Decode a `did:key` (Ed25519) identifier into its raw 32-byte Ed25519 key.
///
/// For Ed25519 a `did:key` is *exactly* `"did:key:" + multibase-base58btc(0xed01
/// ‖ pubkey)` — i.e. the method-specific identifier is the same `Multikey`
/// `publicKeyMultibase` value [`decode_ed25519_multikey`] decodes. A `did:key`
/// is therefore **self-contained**: the key *is* the identity, so this is a pure
/// decode with **no network fetch**.
///
/// Returns `Err` for a non-`did:key` identifier, a non-Ed25519 multicodec, a bad
/// multibase, or a wrong-length payload. A DID URL fragment / query is stripped
/// (`did:key:z6Mk…#z6Mk…` self-references the same key; we decode the base
/// method-specific identifier).
pub fn did_key_to_ed25519(did: &str) -> Result<[u8; 32]> {
    // Strip any DID URL fragment / query before decoding (a `#fragment` or
    // `?query` would otherwise corrupt the base58btc body).
    let did = did.split(['#', '?']).next().unwrap_or(did);
    let msi = did
        .strip_prefix("did:key:")
        .ok_or_else(|| anyhow!("not a did:key identifier: {did}"))?;
    if msi.is_empty() {
        bail!("did:key has empty method-specific identifier: {did}");
    }
    // The method-specific id IS a Multikey `publicKeyMultibase`; reuse the one
    // source of truth for the ed25519-pub multicodec (no duplicated 0xed01).
    decode_ed25519_multikey(msi)
        .map_err(|e| anyhow!("did:key {did} is not a valid Ed25519 Multikey: {e}"))
}

/// Encode a raw 32-byte Ed25519 public key as a `did:key` (Ed25519) identifier
/// (`did:key:z6Mk…`).
///
/// The produced string is `"did:key:" + ed25519_to_multibase(key)` and
/// round-trips with [`did_key_to_ed25519`]; the Multikey body is byte-identical
/// to the `publicKeyMultibase` our DID documents publish (one multicodec source
/// of truth over the same `0xed01 ‖ key` payload).
pub fn ed25519_to_did_key(key: &[u8; 32]) -> String {
    let mut payload = Vec::with_capacity(2 + 32);
    payload.extend_from_slice(&MULTICODEC_ED25519_PUB);
    payload.extend_from_slice(key);
    format!("did:key:z{}", bs58::encode(payload).into_string())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn rand_ed25519() -> [u8; 32] {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;
        SigningKey::generate(&mut OsRng).verifying_key().to_bytes()
    }

    #[test]
    fn did_key_roundtrips_random_key() {
        for _ in 0..16 {
            let raw = rand_ed25519();
            assert_eq!(did_key_to_ed25519(&ed25519_to_did_key(&raw)).unwrap(), raw);
        }
    }

    #[test]
    fn did_key_strips_fragment_and_query() {
        let raw = [3u8; 32];
        let base = ed25519_to_did_key(&raw);
        let mb = base.strip_prefix("did:key:").unwrap();
        // Self-referencing fragment (`#z6Mk…`) — the common DID-URL VM id form.
        assert_eq!(did_key_to_ed25519(&format!("{base}#{mb}")).unwrap(), raw);
        // Query parameter.
        assert_eq!(did_key_to_ed25519(&format!("{base}?versionId=1")).unwrap(), raw);
    }

    #[test]
    fn did_key_rejects_malformed() {
        assert!(did_key_to_ed25519("did:web:example.com").is_err());
        assert!(did_key_to_ed25519("did:key:").is_err());
        // ed25519-pub multicodec but a 31-byte payload (wrong length).
        let mut payload = vec![0xedu8, 0x01];
        payload.extend_from_slice(&[0u8; 31]);
        let did = format!("did:key:z{}", bs58::encode(payload).into_string());
        assert!(did_key_to_ed25519(&did).is_err());
    }
}
