//! Production issuance authority — the service-layer impl of
//! [`hyprstream_ledger::MintVerifier`].
//!
//! Issuance is sealed in the ledger crate: `credit` needs a capability that
//! only the backend's own verifier can produce (see `hyprstream_ledger::mint`).
//! That crate is MIT and WASM-clean, so it defines the signing input and the
//! verification seam but cannot depend on the hybrid-PQC COSE stack. This is
//! the TCB-side half that fills the seam, and it is the exact counterpart of
//! [`CoseCheckpointSigner`](super::CoseCheckpointSigner): the actor signs an
//! issuance authorization with the cell's signing key, and this verifies it
//! against the matching key material.
//!
//! An issuance authorization is a minted, authorizing artifact, so under the
//! Hybrid policy it must carry both arms (EdDSA + ML-DSA-65). `require_pq`
//! propagates from the ledger's `require_pq_signatures` setting, and a missing
//! PQ arm then fails verification rather than silently downgrading.

use std::sync::Arc;

use hyprstream_crypto::cose_sign::verify_composite;
use hyprstream_crypto::pq::MlDsaVerifyingKey;
use hyprstream_ledger::{LedgerError, MintVerifier};

/// Verifies hybrid-PQC composite issuance authorizations.
#[derive(Clone)]
pub struct CoseMintVerifier {
    ed_vk: ed25519_dalek::VerifyingKey,
    pq_vk: Option<Arc<MlDsaVerifyingKey>>,
    require_pq: bool,
}

impl std::fmt::Debug for CoseMintVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoseMintVerifier")
            .field("require_pq", &self.require_pq)
            .field("has_pq_key", &self.pq_vk.is_some())
            .finish()
    }
}

impl CoseMintVerifier {
    /// Build a verifier from the cell's issuance key material.
    ///
    /// `require_pq` must match the signer's policy: a Hybrid signer paired with
    /// a verifier that tolerates a missing PQ arm would accept classical-only
    /// authorizations, which is precisely the downgrade the policy forbids.
    pub fn new(
        ed_vk: ed25519_dalek::VerifyingKey,
        pq_vk: Option<Arc<MlDsaVerifyingKey>>,
        require_pq: bool,
    ) -> Self {
        Self {
            ed_vk,
            pq_vk,
            require_pq,
        }
    }
}

impl MintVerifier for CoseMintVerifier {
    fn verify(&self, signing_input: &[u8], sig: &[u8]) -> Result<(), LedgerError> {
        verify_composite(
            sig,
            &self.ed_vk,
            self.pq_vk.as_deref(),
            signing_input,
            &[],
            self.require_pq,
        )
        .map(|_| ())
        .map_err(|e| LedgerError::MintNotAuthorized(format!("issuance authorization invalid: {e}")))
    }
}

#[cfg(test)]
mod tests {
    // A test asserting a known-good value legitimately unwraps.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use hyprstream_ledger::{
        mint_signing_input, AccountId, Cid, Did, IssueTransfer, TransferId, UnitId,
    };

    fn xfer(amount: u128) -> IssueTransfer {
        IssueTransfer {
            id: TransferId(1),
            issuer_liability: AccountId(10),
            destination: AccountId(11),
            unit: UnitId {
                issuer: Did("did:web:issuer.test".to_owned()),
                resource_class: "gpu.h100.seconds".to_owned(),
            },
            amount,
            grant_cid: Some(Cid(vec![1, 2, 3])),
            user_data: [0u8; 32],
        }
    }

    /// A hybrid-signed authorization verifies, and the same signature does not
    /// carry over to a different issuance.
    #[test]
    fn hybrid_authorization_verifies_and_is_bound_to_its_transfer() {
        use hyprstream_crypto::cose_sign::sign_composite;
        use hyprstream_crypto::pq::ml_dsa_generate_keypair;

        let ed_sk = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let verifier = CoseMintVerifier::new(ed_sk.verifying_key(), Some(Arc::new(pq_vk)), true);

        let t = xfer(500);
        let input = mint_signing_input(&t).unwrap();
        let sig = sign_composite(&ed_sk, Some(&pq_sk), &input, &[]).unwrap();
        verifier.verify(&input, &sig).unwrap();

        // Re-aiming the authorization at a bigger issuance must not verify.
        let bigger = mint_signing_input(&xfer(500_000)).unwrap();
        assert!(
            verifier.verify(&bigger, &sig).is_err(),
            "an authorization must not cover a different issuance"
        );
    }

    /// Under the Hybrid policy a classical-only signature is refused rather
    /// than accepted as a downgrade.
    #[test]
    fn classical_only_authorization_is_refused_under_the_hybrid_policy() {
        use hyprstream_crypto::cose_sign::sign_composite;
        use hyprstream_crypto::pq::ml_dsa_generate_keypair;

        let ed_sk = ed25519_dalek::SigningKey::generate(&mut rand::rngs::OsRng);
        let (_pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let verifier = CoseMintVerifier::new(ed_sk.verifying_key(), Some(Arc::new(pq_vk)), true);

        let input = mint_signing_input(&xfer(500)).unwrap();
        let classical = sign_composite(&ed_sk, None, &input, &[]).unwrap();

        let err = verifier.verify(&input, &classical).unwrap_err();
        assert!(
            matches!(err, LedgerError::MintNotAuthorized(_)),
            "hybrid policy must reject a classical-only authorization, got {err:?}"
        );
    }
}
