//! L1 blob manifest + multihash address helpers.
//!
//! The manifest is the substrate's ingest result: it carries the canonical
//! self-describing content address (CIDv1 multihash), the legacy XET merkle hex
//! (for the current wire and provenance keying), the reconstruction xorb set, the
//! reconstructed byte length, and the `security_label` **carrier field** that
//! unblocks #699 carrier-(b).

use hyprstream_rpc::auth::mac::{ContentBoundLabel, LabeledObject, SecurityLabel};
use hyprstream_rpc::cid::{Codec, HashAlgo, decode_cid, encode_cid};
use serde::{Deserialize, Serialize};

use super::CasError;

/// Multicodec for a sealed labeled blob manifest.
///
/// The raw reconstruction shard remains keyed by its legacy merkle; the public
/// CAS CID identifies the manifest that binds that shard to its label.
pub const FILE_RECONSTRUCTION_CODEC: Codec = Codec::XetManifest;

/// The L1 manifest describing one stored blob.
///
/// This is the sealed substrate-level result type. It supersedes
/// `cas_serve::PutResult` for substrate callers by adding the canonical
/// [`Self::cid`] and required [`Self::security_label`], while keeping
/// [`Self::merkle`] and [`Self::xorb_hashes`] so existing wire/provenance
/// behavior is preserved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlobManifest {
    /// Canonical self-describing content address (CIDv1, base32 multibase). Encodes
    /// the reconstruction codec + multihash algorithm + digest, so it is portable
    /// across the local registry, federation, and XET CAS uniformly.
    pub cid: String,

    /// Legacy XET merkle root (hex). Identical to `cas_serve::PutResult::merkle`.
    /// Retained because the current `putBlob`/`getBlob` wire and the provenance
    /// store key on it.
    pub merkle: String,

    /// Reconstruction xorb set (hex xorb hashes), from the underlying store.
    pub xorb_hashes: Vec<String>,

    /// Bytes newly written after content-addressed dedup within the domain
    /// (`0` if fully deduplicated).
    pub bytes_stored: u64,

    /// Total byte length of the reconstructed content.
    pub byte_len: u64,

    /// The MAC object label for this sealed blob. It is required and covered by
    /// the manifest CID, so relabeling necessarily creates a distinct object.
    pub security_label: SecurityLabel,
}

/// The exact, canonical preimage addressed by a [`BlobManifest`] CID. `cid` is
/// intentionally excluded to avoid self-reference. `bytes_stored` is an ingest
/// receipt (dedup accounting), not content truth, so it is also excluded; all
/// semantic object fields, including the required label, are covered.
#[derive(Serialize)]
struct ManifestPreimage<'a> {
    merkle: &'a str,
    xorb_hashes: &'a [String],
    byte_len: u64,
    security_label: SecurityLabel,
}

impl BlobManifest {
    /// Construct a sealed, labeled manifest and derive its content-bound CID.
    pub fn new(
        merkle: String,
        xorb_hashes: Vec<String>,
        bytes_stored: u64,
        byte_len: u64,
        security_label: SecurityLabel,
    ) -> Result<Self, CasError> {
        let mut manifest = Self {
            cid: String::new(),
            merkle,
            xorb_hashes,
            bytes_stored,
            byte_len,
            security_label,
        };
        manifest.cid = manifest.recomputed_cid()?;
        Ok(manifest)
    }

    fn preimage(&self) -> ManifestPreimage<'_> {
        ManifestPreimage {
            merkle: &self.merkle,
            xorb_hashes: &self.xorb_hashes,
            byte_len: self.byte_len,
            security_label: self.security_label,
        }
    }

    /// Recompute the CID from the complete manifest preimage.
    pub fn recomputed_cid(&self) -> Result<String, CasError> {
        let bytes = serde_json::to_vec(&self.preimage())
            .map_err(|e| CasError::Manifest(format!("serialize manifest preimage: {e}")))?;
        let digest = blake3::hash(&bytes);
        encode_cid(
            FILE_RECONSTRUCTION_CODEC,
            HashAlgo::Blake3,
            digest.as_bytes(),
        )
        .map_err(|e| CasError::Cid(e.to_string()))
    }
}

impl LabeledObject for BlobManifest {
    fn security_label(&self) -> Option<SecurityLabel> {
        Some(self.security_label)
    }
}

impl ContentBoundLabel for BlobManifest {
    fn content_id(&self) -> &[u8] {
        self.cid.as_bytes()
    }

    fn verify_binding(&self) -> bool {
        self.recomputed_cid().is_ok_and(|cid| cid == self.cid)
    }
}

/// Encode a legacy XET merkle hex digest as a canonical CIDv1 for the given
/// algorithm, using the reconstruction-shard codec.
pub fn cid_from_merkle(algorithm: HashAlgo, merkle_hex: &str) -> Result<String, CasError> {
    let digest = hex::decode(merkle_hex).map_err(|e| CasError::Hex(e.to_string()))?;
    encode_cid(Codec::XetShard, algorithm, &digest).map_err(|e| CasError::Cid(e.to_string()))
}

/// Resolve a legacy caller-supplied reconstruction address to the hex digest the
/// underlying `cas_serve::CasStore` keys on.
///
/// Accepts **either**:
/// - a legacy reconstruction CIDv1 string (`b…` base32 multibase) — decoded to
///   its digest, or
/// - a legacy bare hex merkle (40 or 64 hex chars) — returned lowercased.
///
/// A labeled manifest CID deliberately cannot be reduced to a merkle: it must be
/// resolved through its persisted manifest so the label binding is verified.
pub fn merkle_from_address(address: &str) -> Result<String, CasError> {
    if looks_like_legacy_hex(address) {
        return Ok(address.to_ascii_lowercase());
    }
    let cid = decode_cid(address).map_err(|e| CasError::Cid(e.to_string()))?;
    if cid.codec != Codec::XetShard {
        return Err(CasError::Cid(
            "manifest CID must be resolved through the sealed manifest store".into(),
        ));
    }
    Ok(hex::encode(cid.multihash.digest))
}

/// Whether `address` is a legacy bare SHA-1 or BLAKE3 merkle digest.
///
/// This predicate selects the compatibility path that bypasses sealed-manifest
/// resolution, so all CAS callers must share this single definition.
pub(crate) fn looks_like_legacy_hex(address: &str) -> bool {
    matches!(address.len(), 40 | 64) && address.bytes().all(|b| b.is_ascii_hexdigit())
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
    fn sealed_manifest_serializes_with_a_required_label() {
        use hyprstream_rpc::auth::mac::{Assurance, CompartmentSet, Level};
        let label =
            SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY);
        let m = BlobManifest::new("ab".repeat(32), vec!["cd".repeat(32)], 42, 100, label).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        let back: BlobManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
        assert_eq!(back.security_label, label);
        assert!(back.verify_binding());
    }

    #[test]
    fn changing_a_label_rehashes_the_manifest_cid() {
        use hyprstream_rpc::auth::mac::{Assurance, CompartmentSet, Level};
        let internal =
            SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY);
        let secret = SecurityLabel::new(Level::Secret, Assurance::Classical, CompartmentSet::EMPTY);
        let a = BlobManifest::new("ab".repeat(32), vec![], 0, 100, internal).unwrap();
        let b = BlobManifest::new("ab".repeat(32), vec![], 0, 100, secret).unwrap();
        assert_ne!(a.cid, b.cid, "relabeling must rehash the manifest");
    }
}
