//! v16 proof-CWT wire support — canonical parser, replay admission, response binding.
//!
//! This module implements the `hs-rpc-proof-v1` profile: a CWT claims set
//! (RFC 8392) signed as a COSE object (RFC 9052) that proves one RPC request's
//! integrity, freshness, credential binding, and response binding.
//!
//! The wire contract is the frozen gate-2 CDDL at
//! `docs/standards/v16/hyprstream-proof-cwt.cddl` and the private-label
//! registry at `docs/standards/v16/private-label-registry.md`. Canonical test
//! vectors are in `docs/standards/v16/vectors/`.
//!
//! # Layout
//!
//! - [`claims`] — proof CWT claims-set types and bounded CBOR decode.
//! - [`plan`] — canonical signature-plan types and validation.
//! - [`parser`] — bounded COSE object parse, header/`crit` validation, and
//!   proof-vector decode.
//! - [`replay`] — replay admission store, partitioned by disposition.
//! - [`challenge`] — rotating server challenge for unattributed proofs.
//! - [`response`] — response proof binding types.

pub mod claims;
pub mod challenge;
pub mod parser;
pub mod plan;
pub mod replay;
pub mod response;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Profile constants — frozen by the gate-2 CDDL.
// ---------------------------------------------------------------------------

/// Media type for the request-proof CWT (RFC 6838 vendor tree, `+cwt` suffix).
pub const PROOF_TYP: &str = "application/vnd.hyprstream.proof+cwt";

/// Media type for the response-proof CWT.
pub const RESPONSE_PROOF_TYP: &str = "application/vnd.hyprstream.response-proof+cwt";

/// Cryptographic domain separator for request proofs — inside every
/// `Sig_structure`.
pub const REQUEST_PROOF_DOMAIN: &str = "hs-rpc-request-proof-v1";

/// Cryptographic domain separator for response proofs.
pub const RESPONSE_PROOF_DOMAIN: &str = "hs-rpc-response-proof-v1";

/// Versioned suite ID for standalone Ed25519.
pub const SUITE_CLASSICAL: &str = "hs-cose-sign-ed25519-v1";

/// Versioned suite ID for the weakly-non-separable Ed25519 + ML-DSA-65 hybrid.
pub const SUITE_HYBRID: &str = "hs-cose-sign-ed25519-mldsa65-wns-v1";

// --- proof-v1 caps (exact; raising one is an incompatible profile revision) ---

/// Maximum signer groups per plan.
pub const MAX_SIGNER_GROUPS: usize = 8;
/// Maximum components per signer group.
pub const MAX_COMPONENTS_PER_GROUP: usize = 2;
/// Maximum signature entries per proof.
pub const MAX_SIGNATURE_ENTRIES: usize = 16;
/// Maximum encoded bytes for a suite ID.
pub const MAX_SUITE_ID_BYTES: usize = 64;
/// Maximum bytes for a key ID.
pub const MAX_KID_BYTES: usize = 64;
/// Maximum bytes for a canonical service domain (`aud`).
pub const MAX_AUD_BYTES: usize = 253;
/// Maximum Cap'n Proto body bytes signed by the proof.
pub const MAX_BODY_BYTES: usize = 1_048_576; // 1 MiB
/// Maximum total encoded COSE object size.
pub const MAX_COSE_OBJECT_BYTES: usize = 2_097_152; // 2 MiB
/// Request ID size in bytes (CWT `cti`).
pub const REQUEST_ID_SIZE: usize = 16;
/// Credential hash size in bytes (SHA-256).
pub const CREDENTIAL_HASH_SIZE: usize = 32;
/// Minimum server-challenge size in bytes.
pub const MIN_CHALLENGE_BYTES: usize = 16;
/// Maximum server-challenge size in bytes.
pub const MAX_CHALLENGE_BYTES: usize = 64;

// --- private-use CWT claim keys (checked registry) ---

pub const CLAIM_CREDENTIAL_HASH: i64 = -70001;
pub const CLAIM_CAPNP_SCHEMA_ID: i64 = -70002;
pub const CLAIM_CAPNP_BODY_BYTES: i64 = -70003;
pub const CLAIM_RESPONSE_BINDING: i64 = -70004;

// --- private-use COSE header parameters ---

pub const HEADER_HS_DOMAIN: i64 = -70100;
pub const HEADER_HS_SIGNATURE_PLAN: i64 = -70101;
pub const HEADER_HS_LOGICAL_SIGNER_GROUP: i64 = -70102;
pub const HEADER_HS_UNATTRIBUTED_KEY_SET: i64 = -70103;

// --- registered CWT claim keys used by this profile ---

pub const CWT_CLAIM_AUD: i64 = 3;
pub const CWT_CLAIM_EXP: i64 = 4;
pub const CWT_CLAIM_IAT: i64 = 6;
pub const CWT_CLAIM_CTI: i64 = 7;
pub const CWT_CLAIM_NONCE: i64 = 10; // RFC 9711

// --- registered COSE header parameters used by this profile ---

pub const COSE_HEADER_ALG: i64 = 1;
pub const COSE_HEADER_CRIT: i64 = 2;
pub const COSE_HEADER_KID: i64 = 4;
pub const COSE_HEADER_TYP: i64 = 16; // RFC 9596

// --- COSE algorithm identifiers ---

pub const ALG_ED25519: i64 = -19;
pub const ALG_ML_DSA_65: i64 = -49;

/// A 128-bit request identifier — the CWT `cti` claim.
pub type RequestId = [u8; REQUEST_ID_SIZE];

/// SHA-256 credential hash.
pub type CredentialHash = [u8; CREDENTIAL_HASH_SIZE];

/// The kind of proof, determined by which `typ`/`hs_domain` pair it carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofKind {
    /// Request proof (`hs-rpc-request-proof-v1`).
    Request,
    /// Response proof (`hs-rpc-response-proof-v1`).
    Response,
}

impl ProofKind {
    pub fn typ(self) -> &'static str {
        match self {
            Self::Request => PROOF_TYP,
            Self::Response => RESPONSE_PROOF_TYP,
        }
    }

    pub fn domain(self) -> &'static str {
        match self {
            Self::Request => REQUEST_PROOF_DOMAIN,
            Self::Response => RESPONSE_PROOF_DOMAIN,
        }
    }
}

/// Disposition of a proof relative to credential presence. Determines the
/// replay key and partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofDisposition {
    /// A credential was presented and the proof is `cnf`-bound.
    Authenticated,
    /// No credential — self-asserted key set, system-low clearance.
    Unattributed,
}
