//! atproto PDS record store — the federated spine's signed-mutable-pointer layer (#392).
//!
//! Per-account record store hosting `ai.hyprstream.model` records in a Merkle
//! Search Tree (MST), emitting signed commits, and serving CAR proofs. This is
//! the consensus-free versioning primitive: a **signed mutable pointer → immutable
//! OID** that makes federation work without a coordinator.
//!
//! # What this crate provides
//!
//! - **DAG-CBOR** deterministic encode/decode ([`dag_cbor`]) — canonical CBOR
//!   (RFC 7049 §3.9 / §4.2.1): sorted map keys, minimal-length ints, no
//!   duplicate keys. CID links use CBOR tag 42 per the
//!   [DAG-CBOR spec](https://github.com/ipld/specs/blob/master/block-layer/codecs/dag-cbor.md).
//!   The same record produces identical bytes → same CID every time.
//! - **`ai.hyprstream.model` record** ([`record`]) — the confirmed 3-field
//!   lexicon: `repo` (at-uri), `currentOid` (cid), `createdAt` (datetime).
//!   DAG-CBOR encoded; `currentOid` is a `format: "cid"` *string* (not a link).
//! - **MST** ([`mst`]) — the atproto repo structure (Auvolat & Taïani, SRDS 2019).
//!   Insert/delete records keyed by TID; compute the MST root CID. Implemented
//!   as the full recursive tree (`NodeData { l, e: [TreeEntry { p, k, v, t }] }`)
//!   matching the atproto reference.
//! - **Signed commit** ([`commit`]) — `{did, version: 3, rev, data, prev, sig}`.
//!   `sig` is ES256 (P-256 ECDSA over SHA-256) of the DAG-CBOR of the
//!   *unsigned* commit (commit minus the `sig` field), signed with the DID's
//!   `#atproto` P-256 key.
//! - **CAR proof** ([`car`]) — serve `getRepo`/`getRecord` as a CAR (Content
//!   Addressable aRchive) containing the commit + MST path + record.
//! - **`verifyRecordProof`** ([`car::verify_record_proof`]) — offline
//!   verification: check commit sig, walk the MST proof to the record, verify
//!   the record's CID. Enforces the D5 untrusted-host posture.
//!
//! # What this crate deliberately is NOT
//!
//! - No libtorch / inference — this is a metadata/crypto layer.
//! - No networking — proofs are produced and verified in-process; a transport
//!   (HTTP/moq) wraps this separately.
//! - No on-disk storage — the store is in-memory; persistence is a caller concern.
//!
//! # Reused infrastructure
//!
//! ES256/P-256 key handling mirrors `hyprstream::auth::key_rotation`
//! (`Es256SigningKeyStore`) and the DID-document `#atproto` verification method
//! from `hyprstream::services::oauth::did_document`. The signing key type is the
//! same `p256::ecdsa::SigningKey`, so a key from the rotation store drops in
//! directly.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used)]

pub mod at9p;
pub mod at9p_chain;
pub mod at9p_gate;
pub mod at9p_resolver;
pub mod at9p_sign;
pub mod car;
pub mod cid;
pub mod commit;
pub mod dag_cbor;
pub mod event_group;
pub mod list_record;
pub mod mst;
pub mod placement;
pub mod record;
pub mod tid;

pub use cid::Cid;
pub use placement::{GroupItemRecord, GroupRecord, NodeRecord, WorkloadRecord};
pub use record::ModelRecord;
