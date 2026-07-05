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
/// - `trust_boundary` — see [`TrustBoundary`].
///
/// # On-disk mapping
///
/// The **default** domain `(None, Blake3, Local)` maps to the substrate root
/// itself (no sub-directory), so the legacy `cas_serve::CasStore` layout and every
/// existing caller keep working byte-for-byte. Every other domain maps to a
/// deterministic, injective sub-path so distinct domains never collide.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DedupDomain {
    /// Need-to-know compartment (tenant scope). `None` = untenanted.
    pub compartment: Option<Compartment>,
    /// Multihash algorithm partition.
    pub algorithm: HashAlgo,
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
            trust_boundary: TrustBoundary::Local,
        }
    }

    /// A node-local, BLAKE3-256 domain scoped to a single tenant compartment.
    pub fn local_tenant(compartment: Compartment) -> Self {
        Self {
            compartment: Some(compartment),
            algorithm: HashAlgo::Blake3,
            trust_boundary: TrustBoundary::Local,
        }
    }

    /// True for the default domain (root-mapped, legacy-compatible).
    pub fn is_default(&self) -> bool {
        self.compartment.is_none()
            && self.algorithm == HashAlgo::Blake3
            && self.trust_boundary == TrustBoundary::Local
    }

    /// Relative sub-path (under the substrate root) isolating this domain's blobs.
    ///
    /// - Default domain → empty (the root itself), preserving the legacy layout.
    /// - Otherwise →
    ///   `domains/<boundary>/algo-<code>/c-<hex(compartment)|none>/`.
    ///
    /// The compartment name is hex-encoded so the mapping is injective and
    /// path-safe regardless of the characters a compartment name contains (e.g.
    /// the `:` in `"tenant:acme"`), and can never traverse out of the root.
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
            .join(format!("algo-{:#x}", self.algorithm as u64))
            .join(compartment)
    }
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
