//! Dedup domains — the `(compartment, algorithm, trust_boundary)` key that
//! decides which content may be deduplicated against which.
//!
//! A domain is realized as a distinct filesystem sub-root under the substrate
//! root, so two blobs in different domains physically cannot share a xorb — the
//! dedup partition is structural, not a runtime check that could be forgotten.

use std::path::PathBuf;

use hyprstream_rpc::auth::mac::Compartment;
use hyprstream_rpc::cid::HashAlgo;

/// The storage layer / trust boundary a blob lives in.
///
/// Local content is **never** deduplicated against shared-remote content: doing so
/// would turn "is this xorb already present?" into an existence/content oracle
/// across the boundary (U5). The boundary is part of the dedup-domain key, so the
/// two tiers get physically separate storage roots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrustBoundary {
    /// Node-local durable serving tier — the default write destination.
    Local,
    /// Content that originates from, or is shared with, a remote/federated peer.
    SharedRemote,
}

impl TrustBoundary {
    /// Stable, path-safe token for this boundary (used in the domain sub-root).
    const fn token(self) -> &'static str {
        match self {
            TrustBoundary::Local => "local",
            TrustBoundary::SharedRemote => "shared-remote",
        }
    }
}

/// A dedup domain: content is only ever deduplicated against other content in the
/// **same** domain. Keyed by:
///
/// - `compartment` — tenant / need-to-know isolation, drawn from the existing MAC
///   [`Compartment`] vocabulary (e.g. `"tenant:acme"`). `None` = untenanted. This
///   field is READ-only vocabulary reuse; the substrate never authors, derives,
///   or enforces MAC policy from it.
/// - `algorithm` — the multihash algorithm; SHA-1 content is never deduplicated
///   against SHA-256 (the pair would lean on the weaker collision resistance).
/// - `digest_length` — the multihash digest length in bytes. **Length is part of
///   CID identity** (the multihash carries `uvarint(len)`), so it is part of the
///   dedup key too: BLAKE3 is an XOF and `BLAKE3-512(x)[0:32] == BLAKE3-256(x)`,
///   so a 256-bit and a 512-bit domain over the *same* content would alias if the
///   length were dropped from the key. Two widths get distinct storage roots
///   (#812/#881 joint decision).
/// - `trust_boundary` — see [`TrustBoundary`].
///
/// The `(algorithm, digest_length)` pair is validated against
/// [`HashAlgo::validate_digest_len`] by [`DedupDomain::new`]; the fields stay
/// `pub` so struct-literal construction with `..DedupDomain::local_default()`
/// remains ergonomic, but callers that vary the algorithm/length should go through
/// [`DedupDomain::new`] to keep the pair well-formed.
///
/// # On-disk mapping
///
/// The **default** domain `(None, Blake3, 32, Local)` maps to the substrate root
/// itself (no sub-directory), so the legacy `cas_serve::CasStore` layout and every
/// existing caller keep working byte-for-byte. Every other domain maps to a
/// deterministic, injective sub-path so distinct domains never collide.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DedupDomain {
    /// Need-to-know compartment (tenant scope). `None` = untenanted.
    pub compartment: Option<Compartment>,
    /// Multihash algorithm partition.
    pub algorithm: HashAlgo,
    /// Multihash digest length in bytes (part of the CID identity). Must be a
    /// length the `algorithm` accepts (see [`HashAlgo::validate_digest_len`]).
    pub digest_length: u16,
    /// Storage layer / trust boundary partition.
    pub trust_boundary: TrustBoundary,
}

impl DedupDomain {
    /// The default domain: untenanted, BLAKE3-256, node-local. This is the domain
    /// the current registry/XET call sites use, and it maps to the substrate root
    /// (legacy layout, byte-for-byte parity).
    pub fn local_default() -> Self {
        Self {
            compartment: None,
            algorithm: HashAlgo::Blake3,
            digest_length: 32,
            trust_boundary: TrustBoundary::Local,
        }
    }

    /// A node-local, BLAKE3-256 domain scoped to a single tenant compartment.
    pub fn local_tenant(compartment: Compartment) -> Self {
        Self {
            compartment: Some(compartment),
            algorithm: HashAlgo::Blake3,
            digest_length: 32,
            trust_boundary: TrustBoundary::Local,
        }
    }

    /// Construct a domain, validating that `(algorithm, digest_length)` is a
    /// well-formed multihash pair (the length is one the algorithm accepts).
    ///
    /// This is the constructor to use when varying the algorithm or width; the
    /// compartment/trust-boundary default to untenanted/local. Prefer
    /// [`local_default`](Self::local_default) /
    /// [`local_tenant`](Self::local_tenant) for the common BLAKE3-256 cases.
    pub fn new(
        algorithm: HashAlgo,
        digest_length: u16,
        compartment: Option<Compartment>,
        trust_boundary: TrustBoundary,
    ) -> Result<Self, InvalidDedupDomain> {
        if algorithm.validate_digest_len(digest_length as usize).is_err() {
            return Err(InvalidDedupDomain {
                algorithm,
                digest_length,
            });
        }
        Ok(Self {
            compartment,
            algorithm,
            digest_length,
            trust_boundary,
        })
    }

    /// True for the default domain (root-mapped, legacy-compatible).
    pub fn is_default(&self) -> bool {
        self.compartment.is_none()
            && self.algorithm == HashAlgo::Blake3
            && self.digest_length == 32
            && self.trust_boundary == TrustBoundary::Local
    }

    /// Relative sub-path (under the substrate root) isolating this domain's blobs.
    ///
    /// - Default domain → empty (the root itself), preserving the legacy layout.
    /// - Otherwise →
    ///   `domains/<boundary>/algo-<code>-len<N>/c-<hex(compartment)|none>/`.
    ///
    /// The `algo-<code>-len<N>` segment encodes **both** the algorithm code and
    /// the digest length, so two widths of the same algorithm (BLAKE3-256 vs
    /// BLAKE3-512) map to distinct sub-roots and cannot share xorbs. The
    /// compartment name is hex-encoded so the mapping is injective and path-safe
    /// regardless of the characters a compartment name contains (e.g. the `:` in
    /// `"tenant:acme"`), and can never traverse out of the root.
    pub fn relative_path(&self) -> PathBuf {
        if self.is_default() {
            return PathBuf::new();
        }
        let compartment = match &self.compartment {
            Some(c) => format!("c-{}", hex::encode(c.as_str().as_bytes())),
            None => "c-none".to_owned(),
        };
        PathBuf::from("domains")
            .join(self.trust_boundary.token())
            .join(format!(
                "algo-{:#x}-len{}",
                self.algorithm as u64, self.digest_length
            ))
            .join(compartment)
    }
}

/// Error returned by [`DedupDomain::new`] when the `(algorithm, digest_length)`
/// pair is not a well-formed multihash pair (e.g. sha1/32, blake3/48).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid dedup domain: algorithm {algorithm:?} does not accept digest length {digest_length}")]
pub struct InvalidDedupDomain {
    pub algorithm: HashAlgo,
    pub digest_length: u16,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn default_domain_maps_to_root() {
        let d = DedupDomain::local_default();
        assert!(d.is_default());
        assert_eq!(d.relative_path(), PathBuf::new(), "default domain is the root");
    }

    #[test]
    fn distinct_domains_map_to_distinct_paths() {
        // Every axis change must produce a distinct storage sub-root, or two
        // domains would share xorbs (the exact leak the domain key prevents).
        let base = DedupDomain::local_default();
        let other_algo = DedupDomain {
            algorithm: HashAlgo::Sha2_256,
            digest_length: 32,
            ..DedupDomain::local_default()
        };
        let other_boundary = DedupDomain {
            trust_boundary: TrustBoundary::SharedRemote,
            ..DedupDomain::local_default()
        };
        let tenant_a = DedupDomain::local_tenant(Compartment::new("tenant:acme"));
        let tenant_b = DedupDomain::local_tenant(Compartment::new("tenant:beta"));

        let paths = [
            base.relative_path(),
            other_algo.relative_path(),
            other_boundary.relative_path(),
            tenant_a.relative_path(),
            tenant_b.relative_path(),
        ];
        // All five must be pairwise distinct.
        for i in 0..paths.len() {
            for j in (i + 1)..paths.len() {
                assert_ne!(paths[i], paths[j], "domains {i} and {j} collide on disk");
            }
        }
    }

    #[test]
    fn blake3_256_and_512_widths_do_not_alias() {
        // The #812/#881 joint-decision guarantee: BLAKE3 is an XOF and
        // BLAKE3-512(x)[0:32] == BLAKE3-256(x), so the two widths MUST get
        // distinct storage roots or a 512-bit address would dedup against — and
        // leak the existence of — its 256-bit prefix domain.
        //
        // Use a tenant compartment so *neither* domain collapses to the default
        // root (Blake3/32/None/Local); we want to compare two real sub-paths and
        // see the length encoded in each.
        let tenant = Some(Compartment::new("tenant:acme"));
        let w256 = DedupDomain::new(HashAlgo::Blake3, 32, tenant.clone(), TrustBoundary::Local)
            .unwrap();
        let w512 = DedupDomain::new(HashAlgo::Blake3, 64, tenant, TrustBoundary::Local).unwrap();
        let p256 = w256.relative_path();
        let p512 = w512.relative_path();
        assert_ne!(
            p256, p512,
            "BLAKE3-256 and BLAKE3-512 domains must not share a storage root"
        );
        // The path must encode the length so the distinction is self-describing.
        assert!(
            p256.to_string_lossy().contains("len32"),
            "256-bit domain path must carry len32: {p256:?}"
        );
        assert!(
            p512.to_string_lossy().contains("len64"),
            "512-bit domain path must carry len64: {p512:?}"
        );
    }

    #[test]
    fn new_rejects_malformed_algo_length_pair() {
        // sha1 only accepts 20 bytes; 32 is not a valid sha1 multihash.
        let err = DedupDomain::new(HashAlgo::Sha1, 32, None, TrustBoundary::Local).unwrap_err();
        assert_eq!(err.algorithm, HashAlgo::Sha1);
        assert_eq!(err.digest_length, 32);
        // blake3 accepts 32 or 64, not 48.
        assert!(DedupDomain::new(HashAlgo::Blake3, 48, None, TrustBoundary::Local).is_err());
        // Valid pairs construct fine.
        assert!(DedupDomain::new(HashAlgo::Blake3, 64, None, TrustBoundary::Local).is_ok());
        assert!(DedupDomain::new(HashAlgo::Sha2_256, 32, None, TrustBoundary::Local).is_ok());
    }

    #[test]
    fn compartment_path_is_traversal_safe() {
        // A hostile-looking compartment name must not escape the root.
        let d = DedupDomain::local_tenant(Compartment::new("../../etc/passwd"));
        let p = d.relative_path();
        assert!(
            p.components().all(|c| !matches!(c, std::path::Component::ParentDir)),
            "compartment name must never yield a `..` component: {p:?}"
        );
    }

    #[test]
    fn same_domain_is_stable() {
        let a = DedupDomain::local_tenant(Compartment::new("tenant:acme"));
        let b = DedupDomain::local_tenant(Compartment::new("tenant:acme"));
        assert_eq!(a.relative_path(), b.relative_path());
    }
}
