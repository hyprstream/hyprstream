//! [`CasSubstrate`] — the L1 unified content-addressed store.
//!
//! Wraps `cas_serve::CasStore` per dedup domain: each domain gets its own store
//! rooted at [`DedupDomain::relative_path`] under the substrate root, so
//! deduplication is structurally confined to a single `(compartment, algorithm,
//! trust_boundary)` domain. Addressing is multihash: ingest returns a canonical
//! CID, and reads accept either a CID or a legacy hex merkle.

use std::fs;
use std::path::{Path, PathBuf};

use cas_serve::CasStore;
use hyprstream_rpc::auth::mac::SecurityLabel;
use hyprstream_rpc::cid::{Codec, HashAlgo, decode_cid, encode_cid};

use super::CasError;
use super::domain::DedupDomain;
use super::manifest::{BlobManifest, merkle_from_address};

/// Hex-character length of a `digest_length`-byte digest (2 hex chars per byte).
const fn hex_digest_len(digest_length: u16) -> usize {
    digest_length as usize * 2
}

/// The L1 unified content-addressed store.
///
/// A thin, domain-partitioning facade over `cas_serve::CasStore`. Construct once
/// (`from_env`) and route every ingest/read through a [`DedupDomain`].
#[derive(Debug, Clone)]
pub struct CasSubstrate {
    root: PathBuf,
}

impl CasSubstrate {
    /// Open a substrate rooted at `root`.
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    /// Open a substrate using the standard XET storage-path resolution
    /// (`CAS_STORAGE` / `XET_CACHE_DIR` / `~/.cache/xet`), matching the legacy
    /// `CasStore::from_env()` root exactly so the default domain is layout-compatible.
    pub fn from_env() -> Self {
        Self::new(cas_serve::store::resolve_storage_path())
    }

    /// The `CasStore` backing a given dedup domain (rooted at the domain sub-path).
    fn store_for(&self, domain: &DedupDomain) -> CasStore {
        CasStore::new(self.root.join(domain.relative_path()))
    }

    fn manifest_dir(&self, domain: &DedupDomain) -> PathBuf {
        self.root.join(domain.relative_path()).join("manifests")
    }

    fn manifest_path(&self, domain: &DedupDomain, cid: &str) -> PathBuf {
        self.manifest_dir(domain).join(format!("{cid}.json"))
    }

    fn canonical_cid(address: &str) -> Result<String, CasError> {
        let decoded = decode_cid(address).map_err(|e| CasError::Cid(e.to_string()))?;
        encode_cid(
            decoded.codec,
            decoded.multihash.algo,
            &decoded.multihash.digest,
        )
        .map_err(|e| CasError::Cid(e.to_string()))
    }

    fn persist_manifest(
        &self,
        domain: &DedupDomain,
        manifest: &BlobManifest,
    ) -> Result<(), CasError> {
        let dir = self.manifest_dir(domain);
        fs::create_dir_all(&dir)
            .map_err(|e| CasError::Manifest(format!("create {}: {e}", dir.display())))?;
        let bytes = serde_json::to_vec(manifest)
            .map_err(|e| CasError::Manifest(format!("serialize manifest: {e}")))?;
        let path = self.manifest_path(domain, &manifest.cid);
        fs::write(&path, bytes)
            .map_err(|e| CasError::Manifest(format!("write {}: {e}", path.display())))
    }

    fn load_manifest(&self, domain: &DedupDomain, address: &str) -> Result<BlobManifest, CasError> {
        let cid = Self::canonical_cid(address)?;
        let decoded = decode_cid(&cid).map_err(|e| CasError::Cid(e.to_string()))?;
        if decoded.codec != Codec::XetManifest {
            return Err(CasError::Cid(
                "CID does not address a labeled CAS manifest".into(),
            ));
        }
        let path = self.manifest_path(domain, &cid);
        let bytes = fs::read(&path)
            .map_err(|e| CasError::Manifest(format!("read {}: {e}", path.display())))?;
        let manifest: BlobManifest = serde_json::from_slice(&bytes)
            .map_err(|e| CasError::Manifest(format!("decode {}: {e}", path.display())))?;
        if manifest.cid != cid
            || !hyprstream_rpc::auth::mac::ContentBoundLabel::verify_binding(&manifest)
        {
            return Err(CasError::Manifest(format!(
                "CID binding verification failed for {cid}"
            )));
        }
        Ok(manifest)
    }

    /// Ingest bytes into a dedup domain.
    ///
    /// Chunking (gearhash CDC), xorb aggregation, content-addressed dedup, and the
    /// server-computed merkle all happen inside the underlying `CasStore` — byte
    /// boundaries stay identical to xet-core. The substrate adds the canonical
    /// manifest CID with a required, content-bound `security_label`.
    ///
    /// Only **BLAKE3-256** ingest is implemented today: the merkle the store
    /// computes is the 32-byte BLAKE3 reconstruction hash, so the domain must
    /// declare `(Blake3, digest_length = 32)`. Any other `(algorithm, length)`
    /// pair is rejected rather than mislabeled with a CID that claims the wrong
    /// algorithm/width. The domain vocabulary still *keys* every algorithm and
    /// width (see [`DedupDomain`]) so future non-BLAKE3 / wider ingest (e.g. the
    /// BLAKE3-512 at9p capsule path, #881) never dedups against BLAKE3-256 content
    /// — it lands as a distinct, length-partitioned storage root.
    pub async fn put(
        &self,
        domain: &DedupDomain,
        data: &[u8],
        security_label: SecurityLabel,
    ) -> Result<BlobManifest, CasError> {
        if domain.algorithm != HashAlgo::Blake3 || domain.digest_length != 32 {
            return Err(CasError::UnsupportedIngestAlgorithm(domain.algorithm));
        }
        let store = self.store_for(domain);
        let put = store.put_file_bytes(data).await?;
        // The store's merkle is the 32-byte BLAKE3 digest the domain declared;
        // assert the contract rather than silently emitting a CID of the wrong width.
        debug_assert_eq!(
            put.merkle.len(),
            hex_digest_len(domain.digest_length),
            "store merkle must be the declared {}-byte digest",
            domain.digest_length
        );
        let manifest = BlobManifest::new(
            put.merkle,
            put.xorb_hashes,
            put.bytes_stored,
            data.len() as u64,
            security_label,
        )?;
        self.persist_manifest(domain, &manifest)?;
        Ok(manifest)
    }

    /// Reconstruct a blob's bytes by content address within a domain.
    ///
    /// `address` may be a canonical CID or a legacy hex merkle (see
    /// [`merkle_from_address`]).
    pub async fn get(&self, domain: &DedupDomain, address: &str) -> Result<Vec<u8>, CasError> {
        let merkle = if looks_like_legacy_hex(address) {
            merkle_from_address(address)?
        } else {
            self.load_manifest(domain, address)?.merkle
        };
        let store = self.store_for(domain);
        Ok(store.get_file_bytes(&merkle).await?)
    }

    /// Load and verify the sealed manifest named by a manifest CID.
    pub fn manifest(&self, domain: &DedupDomain, cid: &str) -> Result<BlobManifest, CasError> {
        self.load_manifest(domain, cid)
    }

    /// Read the raw bytes of a single stored xorb, keyed by its hex xorb hash,
    /// within a domain. Backs the HuggingFace-XET `GET /get_xorb/{hash}/` route.
    pub async fn read_xorb(
        &self,
        domain: &DedupDomain,
        xorb_hash_hex: &str,
    ) -> Result<Vec<u8>, CasError> {
        let store = self.store_for(domain);
        Ok(store.read_xorb(xorb_hash_hex).await?)
    }

    /// True if the substrate can serve this content address within the domain.
    pub fn exists(&self, domain: &DedupDomain, address: &str) -> bool {
        let merkle = if looks_like_legacy_hex(address) {
            merkle_from_address(address)
        } else {
            self.load_manifest(domain, address)
                .map(|manifest| manifest.merkle)
        };
        merkle.is_ok_and(|merkle| self.store_for(domain).exists(&merkle))
    }
}

fn looks_like_legacy_hex(address: &str) -> bool {
    matches!(address.len(), 40 | 64) && address.bytes().all(|b| b.is_ascii_hexdigit())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use hyprstream_rpc::auth::mac::{Assurance, Compartment, CompartmentSet, Level};

    fn label() -> SecurityLabel {
        SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY)
    }

    fn payload(seed: u8, len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| seed.wrapping_add((i as u8).wrapping_mul(31)))
            .collect()
    }

    #[tokio::test]
    async fn put_get_round_trips_via_cid_and_legacy_hex() {
        let dir = tempfile::tempdir().unwrap();
        let sub = CasSubstrate::new(dir.path());
        let domain = DedupDomain::local_default();
        let original = payload(7, 256 * 1024);

        let m = sub.put(&domain, &original, label()).await.unwrap();
        assert!(m.cid.starts_with('b'));
        assert_eq!(m.byte_len, original.len() as u64);
        assert!(m.bytes_stored > 0);

        // Reconstruct by the canonical CID.
        assert_eq!(sub.get(&domain, &m.cid).await.unwrap(), original);
        // And by the legacy hex merkle — same content.
        assert_eq!(sub.get(&domain, &m.merkle).await.unwrap(), original);
        assert!(sub.exists(&domain, &m.cid));
    }

    #[tokio::test]
    async fn default_domain_matches_legacy_casstore_layout() {
        // Parity guard: a blob written through the substrate's default domain must
        // be readable by a bare CasStore at the same root (byte-for-byte layout).
        let dir = tempfile::tempdir().unwrap();
        let sub = CasSubstrate::new(dir.path());
        let original = payload(3, 200 * 1024);

        let m = sub
            .put(&DedupDomain::local_default(), &original, label())
            .await
            .unwrap();

        let legacy = CasStore::new(dir.path());
        assert_eq!(legacy.get_file_bytes(&m.merkle).await.unwrap(), original);
    }

    #[tokio::test]
    async fn distinct_domains_do_not_dedup_together() {
        // Same bytes in two domains must produce the same merkle but be stored
        // independently: the second write is NOT deduplicated against the first.
        let dir = tempfile::tempdir().unwrap();
        let sub = CasSubstrate::new(dir.path());
        let original = payload(9, 200 * 1024);

        let default_domain = DedupDomain::local_default();
        let tenant_domain = DedupDomain::local_tenant(Compartment::new("tenant:acme"));

        let a = sub.put(&default_domain, &original, label()).await.unwrap();
        let b = sub.put(&tenant_domain, &original, label()).await.unwrap();

        // Content-addressed ⇒ identical merkle regardless of domain.
        assert_eq!(a.merkle, b.merkle);
        // But the tenant domain stored fresh bytes (no cross-domain dedup).
        assert!(
            b.bytes_stored > 0,
            "cross-domain content must not deduplicate: existence/content leak"
        );

        // The tenant blob is NOT visible from the default domain and vice-versa is
        // physically separate storage.
        assert!(sub.exists(&tenant_domain, &b.merkle));
    }

    #[tokio::test]
    async fn same_domain_reupload_dedups() {
        let dir = tempfile::tempdir().unwrap();
        let sub = CasSubstrate::new(dir.path());
        let domain = DedupDomain::local_default();
        let original = payload(11, 200 * 1024);

        let first = sub.put(&domain, &original, label()).await.unwrap();
        let second = sub.put(&domain, &original, label()).await.unwrap();
        assert_eq!(first.merkle, second.merkle);
        assert_eq!(second.bytes_stored, 0, "same-domain re-upload fully dedups");
    }

    #[tokio::test]
    async fn non_blake3_ingest_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let sub = CasSubstrate::new(dir.path());
        let domain = DedupDomain {
            algorithm: HashAlgo::Sha2_256,
            digest_length: 32,
            ..DedupDomain::local_default()
        };
        let err = sub.put(&domain, b"hello", label()).await.unwrap_err();
        assert!(matches!(err, CasError::UnsupportedIngestAlgorithm(_)));
    }

    #[tokio::test]
    async fn blake3_512_ingest_not_yet_supported() {
        // The domain vocabulary admits a BLAKE3-512 width (length-partitioned
        // storage root), but ingest of it is #881's job; the substrate must
        // reject a 512-bit ingest domain rather than silently addressing the
        // 32-byte merkle as if it were 64 bytes.
        let dir = tempfile::tempdir().unwrap();
        let sub = CasSubstrate::new(dir.path());
        let domain = DedupDomain {
            digest_length: 64,
            ..DedupDomain::local_default()
        };
        let err = sub.put(&domain, b"hello", label()).await.unwrap_err();
        assert!(
            matches!(err, CasError::UnsupportedIngestAlgorithm(_)),
            "BLAKE3-512 ingest must be rejected until #881 lands"
        );
    }

    #[tokio::test]
    async fn carrier_label_is_passed_through() {
        use hyprstream_rpc::auth::mac::{Assurance, CompartmentSet, Level};
        let dir = tempfile::tempdir().unwrap();
        let sub = CasSubstrate::new(dir.path());
        let label =
            SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY);
        let m = sub
            .put(&DedupDomain::local_default(), b"payload-bytes", label)
            .await
            .unwrap();
        assert_eq!(
            m.security_label, label,
            "required label plumbs through unchanged"
        );
    }

    #[tokio::test]
    async fn manifest_cid_remains_resolvable_after_reopening_the_substrate() {
        let dir = tempfile::tempdir().unwrap();
        let domain = DedupDomain::local_default();
        let manifest = CasSubstrate::new(dir.path())
            .put(&domain, b"durable-manifest", label())
            .await
            .unwrap();
        let reopened = CasSubstrate::new(dir.path());
        assert_eq!(
            reopened.get(&domain, &manifest.cid).await.unwrap(),
            b"durable-manifest"
        );
    }
}
