//! [`CasSubstrate`] — the L1 unified content-addressed store.
//!
//! Wraps `cas_serve::CasStore` per dedup domain: each domain gets its own store
//! rooted at [`DedupDomain::relative_path`] under the substrate root, so
//! deduplication is structurally confined to a single `(compartment, algorithm,
//! trust_boundary)` domain. Addressing is multihash: ingest returns a canonical
//! CID, and reads accept either a CID or a legacy hex merkle.

use std::path::{Path, PathBuf};

use cas_serve::CasStore;
use hyprstream_rpc::auth::mac::SecurityLabel;
use hyprstream_rpc::cid::HashAlgo;

use super::domain::DedupDomain;
use super::manifest::{cid_from_merkle, merkle_from_address, BlobManifest};
use super::CasError;

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

    /// Ingest bytes into a dedup domain.
    ///
    /// Chunking (gearhash CDC), xorb aggregation, content-addressed dedup, and the
    /// server-computed merkle all happen inside the underlying `CasStore` — byte
    /// boundaries stay identical to xet-core. The substrate adds the canonical
    /// multihash CID and attaches the `security_label` carrier (plumb-through only).
    ///
    /// Only BLAKE3-256 ingest is implemented today (the merkle the store computes
    /// is BLAKE3-based); a non-BLAKE3 domain is rejected rather than mislabeled
    /// with a CID that claims the wrong algorithm. The domain still *keys* every
    /// algorithm so future non-BLAKE3 content never dedups against BLAKE3 content.
    pub async fn put(
        &self,
        domain: &DedupDomain,
        data: &[u8],
        security_label: Option<SecurityLabel>,
    ) -> Result<BlobManifest, CasError> {
        if domain.algorithm != HashAlgo::Blake3 {
            return Err(CasError::UnsupportedIngestAlgorithm(domain.algorithm));
        }
        let store = self.store_for(domain);
        let put = store.put_file_bytes(data).await?;
        let cid = cid_from_merkle(domain.algorithm, &put.merkle)?;
        Ok(BlobManifest {
            cid,
            merkle: put.merkle,
            xorb_hashes: put.xorb_hashes,
            bytes_stored: put.bytes_stored,
            byte_len: data.len() as u64,
            security_label,
        })
    }

    /// Reconstruct a blob's bytes by content address within a domain.
    ///
    /// `address` may be a canonical CID or a legacy hex merkle (see
    /// [`merkle_from_address`]).
    pub async fn get(&self, domain: &DedupDomain, address: &str) -> Result<Vec<u8>, CasError> {
        let merkle = merkle_from_address(address)?;
        let store = self.store_for(domain);
        Ok(store.get_file_bytes(&merkle).await?)
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
        match merkle_from_address(address) {
            Ok(merkle) => self.store_for(domain).exists(&merkle),
            Err(_) => false,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use hyprstream_rpc::auth::mac::Compartment;

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

        let m = sub.put(&domain, &original, None).await.unwrap();
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
            .put(&DedupDomain::local_default(), &original, None)
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

        let a = sub.put(&default_domain, &original, None).await.unwrap();
        let b = sub.put(&tenant_domain, &original, None).await.unwrap();

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

        let first = sub.put(&domain, &original, None).await.unwrap();
        let second = sub.put(&domain, &original, None).await.unwrap();
        assert_eq!(first.merkle, second.merkle);
        assert_eq!(second.bytes_stored, 0, "same-domain re-upload fully dedups");
    }

    #[tokio::test]
    async fn non_blake3_ingest_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let sub = CasSubstrate::new(dir.path());
        let domain = DedupDomain {
            algorithm: HashAlgo::Sha2_256,
            ..DedupDomain::local_default()
        };
        let err = sub.put(&domain, b"hello", None).await.unwrap_err();
        assert!(matches!(err, CasError::UnsupportedIngestAlgorithm(_)));
    }

    #[tokio::test]
    async fn carrier_label_is_passed_through() {
        use hyprstream_rpc::auth::mac::{Assurance, CompartmentSet, Level};
        let dir = tempfile::tempdir().unwrap();
        let sub = CasSubstrate::new(dir.path());
        let label = SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY);
        let m = sub
            .put(&DedupDomain::local_default(), b"payload-bytes", Some(label))
            .await
            .unwrap();
        assert_eq!(m.security_label, Some(label), "carrier field plumbs through unchanged");
    }
}
