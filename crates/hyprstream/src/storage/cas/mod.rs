//! L1 unified content-addressed store (CAS) substrate — #812 (epic #809, W1).
//!
//! One content-addressed store under everything. Every artifact is addressed by a
//! self-describing **multihash** ([`hyprstream_rpc::cid`]), and every ingest is
//! partitioned into a **dedup domain** keyed `(compartment, algorithm,
//! trust_boundary)` so content is only ever deduplicated against other content in
//! the *same* domain.
//!
//! ## What this layer is
//!
//! [`CasSubstrate`] absorbs [`cas_serve::CasStore`] (the write/reconstruct core
//! left behind the retired `cas-serve` binary) behind a domain-partitioned,
//! multihash-addressed facade. The registry `putBlob`/`getBlob` paths and the XET
//! `get_xorb` route reach the store *through* the substrate now, never a bare
//! `CasStore::from_env()`.
//!
//! - **Chunking** is unchanged: gearhash CDC via `cas_serve::chunker`, byte-for-byte
//!   identical to xet-core (see that crate's `chunker.rs` golden test).
//! - **Default algorithm** is BLAKE3-256, aligning with the existing
//!   merklehash/xet substrate.
//! - **On-disk layout** for the *default* domain `(None, Blake3, Local)` is
//!   identical to the legacy `CasStore` root, so existing stores and callers keep
//!   working byte-for-byte. Non-default domains get their own sub-root.
//!
//! ## Dedup domains ([`DedupDomain`])
//!
//! Content in different domains is **never** deduplicated together, because each
//! domain is a distinct storage sub-root:
//!
//! - **compartment** — tenant / need-to-know isolation, keyed off the existing MAC
//!   [`Compartment`](hyprstream_rpc::auth::mac::Compartment) vocabulary
//!   (READ-only here — the substrate never authors or enforces MAC policy).
//! - **algorithm** — SHA-1 content is never deduplicated against SHA-256; the pair
//!   would lean on the weaker collision resistance.
//! - **trust_boundary** — node-local content is never deduplicated against
//!   shared-remote content (an existence/content oracle across the boundary, U5).
//!
//! ## MAC boundary (deliberate)
//!
//! [`BlobManifest`] carries a `security_label` **carrier field** (unblocks #699
//! carrier-(b)). This is *plumb-through only*: the substrate never populates it
//! with real policy, never derives clearance, and never enforces access. Object
//! labeling and enforcement are #699/#767's job. A `None` label is an *unlabeled
//! carrier*, NOT "public".

mod domain;
mod manifest;
mod mount;
mod substrate;

pub use domain::{DedupDomain, TrustBoundary};
pub use manifest::{BlobManifest, FILE_RECONSTRUCTION_CODEC, cid_from_merkle, merkle_from_address};
pub use mount::{
    AllowAllCasAuthorizer, CasMount, CasMountAuthorizer, CasMountAuthzRequest, CasMountObjectKind,
    DenyAllCasAuthorizer,
};
pub use substrate::CasSubstrate;

/// Errors from the L1 CAS substrate.
#[derive(Debug, thiserror::Error)]
pub enum CasError {
    /// The underlying content-addressed store failed (io, not-found, corrupt shard).
    #[error("cas store: {0}")]
    Store(#[from] cas_serve::StoreError),

    /// A content address (CID / multihash) could not be encoded or decoded.
    #[error("content address: {0}")]
    Cid(String),

    /// A hex digest was malformed.
    #[error("invalid hex digest: {0}")]
    Hex(String),

    /// Ingest was requested for an algorithm the substrate cannot currently
    /// *produce*. The domain vocabulary supports every algorithm for keying, but
    /// the ingest path only computes BLAKE3-256 today (see [`CasSubstrate::put`]).
    #[error("unsupported ingest algorithm: {0:?} (only BLAKE3-256 ingest is implemented)")]
    UnsupportedIngestAlgorithm(hyprstream_rpc::cid::HashAlgo),
}
