//! L1 blob manifest + multihash address helpers.
//!
//! The manifest is the substrate's ingest result: it carries the canonical
//! self-describing content address (CIDv1 multihash), the legacy XET merkle hex
//! (for the current wire and provenance keying), the reconstruction xorb set, the
//! reconstructed byte length, and the `security_label` **carrier field** that
//! unblocks #699 carrier-(b).

use hyprstream_rpc::auth::mac::SecurityLabel;
use hyprstream_rpc::cid::{decode_cid, encode_cid, Codec, HashAlgo};
use serde::{Deserialize, Serialize};

use super::CasError;

/// Multicodec for a full-file reconstruction address.
///
/// A CAS blob is addressed by its XET reconstruction *shard* (merkle) hash, so the
/// content-type codec is [`Codec::XetShard`] — see `cid.rs`.
pub const FILE_RECONSTRUCTION_CODEC: Codec = Codec::XetShard;

/// The L1 manifest describing one stored blob.
///
/// This is the substrate-level result type. It supersedes `cas_serve::PutResult`
/// for substrate callers by adding the canonical [`Self::cid`] and the
/// [`Self::security_label`] carrier field, while keeping [`Self::merkle`] and
/// [`Self::xorb_hashes`] so existing wire/provenance behavior is preserved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlobManifest {
    /// Canonical self-describing content address (CIDv1, base32 multibase). Encodes
    /// the reconstruction codec + multihash algorithm + digest, so it is portable
    /// across the local registry, federation, and XET CAS uniformly.
    pub cid: String,

    /// Legacy XET merkle root (hex). Identical to `cas_serve::PutResult::merkle`;
    /// this is the digest embedded in [`Self::cid`]. Retained because the current
    /// `putBlob`/`getBlob` wire and the provenance store key on it.
    pub merkle: String,

    /// Reconstruction xorb set (hex xorb hashes), from the underlying store.
    pub xorb_hashes: Vec<String>,

    /// Bytes newly written after content-addressed dedup within the domain
    /// (`0` if fully deduplicated).
    pub bytes_stored: u64,

    /// Total byte length of the reconstructed content.
    pub byte_len: u64,

    /// **Carrier field (#699 carrier-(b)).** The MAC object label for this blob.
    ///
    /// Plumb-through only: the substrate never populates this with real policy,
    /// never derives it, and never enforces on it. Population + enforcement is
    /// #699/#767's job. `None` means *unlabeled carrier* — NOT "public".
    #[serde(default)]
    pub security_label: Option<SecurityLabel>,
}

/// Encode a legacy XET merkle hex digest as a canonical CIDv1 for the given
/// algorithm, using the reconstruction-shard codec.
pub fn cid_from_merkle(algorithm: HashAlgo, merkle_hex: &str) -> Result<String, CasError> {
    let digest = hex::decode(merkle_hex).map_err(|e| CasError::Hex(e.to_string()))?;
    encode_cid(FILE_RECONSTRUCTION_CODEC, algorithm, &digest).map_err(|e| CasError::Cid(e.to_string()))
}

/// Resolve a caller-supplied content address to the hex digest the underlying
/// `cas_serve::CasStore` keys on.
///
/// Accepts **either**:
/// - a canonical CIDv1 string (`b…` base32 multibase) — decoded to its digest, or
/// - a legacy bare hex merkle (40 or 64 hex chars) — returned lowercased.
///
/// This is the compatibility seam that lets existing callers keep passing a hex
/// merkle while new callers pass a self-describing multihash. The two address
/// spaces are unambiguous: our legacy merkles are exactly 20- or 32-byte digests
/// (40/64 hex chars), and a CID is longer and uses the base32 alphabet.
pub fn merkle_from_address(address: &str) -> Result<String, CasError> {
    let looks_like_legacy_hex =
        matches!(address.len(), 40 | 64) && address.bytes().all(|b| b.is_ascii_hexdigit());
    if looks_like_legacy_hex {
        return Ok(address.to_ascii_lowercase());
    }
    let cid = decode_cid(address).map_err(|e| CasError::Cid(e.to_string()))?;
    Ok(hex::encode(cid.multihash.digest))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn cid_from_merkle_round_trips_via_address() {
        // 32-byte blake3 merkle → CID → back to the same hex digest.
        let merkle = "ab".repeat(32);
        let cid = cid_from_merkle(HashAlgo::Blake3, &merkle).unwrap();
        assert!(cid.starts_with('b'), "canonical CIDv1 base32 prefix");
        let back = merkle_from_address(&cid).unwrap();
        assert_eq!(back, merkle);
    }

    #[test]
    fn legacy_hex_merkle_passes_through() {
        let merkle = "cd".repeat(32); // 64 hex chars
        assert_eq!(merkle_from_address(&merkle).unwrap(), merkle);
        // Uppercase legacy hex is normalized to lowercase.
        assert_eq!(merkle_from_address(&merkle.to_uppercase()).unwrap(), merkle);
        // A 40-char (sha1) legacy digest is also accepted.
        let sha1 = "ef".repeat(20);
        assert_eq!(merkle_from_address(&sha1).unwrap(), sha1);
    }

    #[test]
    fn cid_and_legacy_hex_resolve_identically() {
        let merkle = "12".repeat(32);
        let cid = cid_from_merkle(HashAlgo::Blake3, &merkle).unwrap();
        assert_eq!(
            merkle_from_address(&cid).unwrap(),
            merkle_from_address(&merkle).unwrap(),
            "a CID and its legacy hex must resolve to the same store key"
        );
    }

    #[test]
    fn rejects_garbage_address() {
        // Not 40/64 hex, not a valid CID.
        assert!(merkle_from_address("not-an-address").is_err());
        assert!(cid_from_merkle(HashAlgo::Blake3, "zz").is_err());
    }

    #[test]
    fn manifest_security_label_carrier_serializes() {
        use hyprstream_rpc::auth::mac::{Assurance, CompartmentSet, Level};
        let label = SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY);
        let m = BlobManifest {
            cid: "bexample".to_owned(),
            merkle: "ab".repeat(32),
            xorb_hashes: vec!["cd".repeat(32)],
            bytes_stored: 42,
            byte_len: 100,
            security_label: Some(label),
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: BlobManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.security_label, Some(label));
    }

    #[test]
    fn manifest_unlabeled_carrier_round_trips() {
        let m = BlobManifest {
            cid: "bexample".to_owned(),
            merkle: "ab".repeat(32),
            xorb_hashes: vec![],
            bytes_stored: 0,
            byte_len: 0,
            security_label: None,
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: BlobManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.security_label, None, "None = unlabeled carrier, not public");
    }
}
