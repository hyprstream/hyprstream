//! Settlement attestation verification (#1399).
//!
//! Verifies the PQ-hybrid-signed settlement attestation server-side.
//! Reuses the #1129 spend-auth verification path (`verify_composite`).

use hyprstream_pay::PayError;

/// A verified settlement attestation — the attestation blob was PQ-hybrid
/// signed by a trusted settlement authority (Stripe webhook processor)
/// and verifies against the issuer's key material.
pub struct VerifiedAttestation {
    pub settlement_id: String,
    pub issuer_did: String,
}

/// The seam for verifying settlement attestations. The production impl
/// calls `hyprstream_crypto::cose_sign::verify_composite`; tests use a
/// test double.
#[async_trait::async_trait]
pub trait SettlementAttestationVerifier: Send + Sync {
    /// Verify the attestation blob against the settlement id. Returns the
    /// verified attestation on success, or a fail-closed error.
    async fn verify(
        &self,
        attestation: &[u8],
        settlement_id: &str,
    ) -> Result<VerifiedAttestation, PayError>;
}

/// A test/bootstrap verifier that denies everything (fail-closed).
pub struct DenyAllAttestationVerifier;

#[async_trait::async_trait]
impl SettlementAttestationVerifier for DenyAllAttestationVerifier {
    async fn verify(
        &self,
        _attestation: &[u8],
        _settlement_id: &str,
    ) -> Result<VerifiedAttestation, PayError> {
        Err(PayError::AttestationInvalid(
            "no attestation verifier configured (deny-all bootstrap)".to_owned(),
        ))
    }
}
