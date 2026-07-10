//! Production checkpoint signer — the service-layer impl of
//! `hyprstream_ledger::CheckpointSigner` (plan §3, item 1.6).
//!
//! Wraps [`hyprstream_crypto::cose_sign::sign_composite`] (EdDSA + ML-DSA-65
//! nested COSE) and is `CryptoPolicy`-aware: under the Hybrid policy
//! (`require_pq = true`) a missing PQ key **fails closed** at sign time — it
//! never silently downgrades to a Classical-only signature. This is the same
//! rule the MAC audit path and the RPC envelope layer enforce for
//! security-critical artifacts.

use std::sync::Arc;

use hyprstream_crypto::cose_sign::sign_composite;
use hyprstream_crypto::pq::MlDsaSigningKey;
use hyprstream_ledger::{CheckpointSigner, Did, LedgerError};

/// Composite (Hybrid-PQC) checkpoint signer keyed by the cell ledger's service
/// identity. `Clone` so the actor and any emitter/task can share it cheaply
/// (the keys are held behind an `Arc`).
#[derive(Clone)]
pub struct CoseCheckpointSigner {
    ed_sk: Arc<ed25519_dalek::SigningKey>,
    pq_sk: Option<Arc<MlDsaSigningKey>>,
    ledger_id: Did,
    require_pq: bool,
}

impl std::fmt::Debug for CoseCheckpointSigner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoseCheckpointSigner")
            .field("ledger_id", &self.ledger_id)
            .field("require_pq", &self.require_pq)
            .field("has_pq_key", &self.pq_sk.is_some())
            .finish()
    }
}

impl CoseCheckpointSigner {
    /// Construct a Hybrid signer (Ed25519 + ML-DSA-65). `require_pq = true`
    /// is the production default.
    pub fn hybrid(
        ledger_id: Did,
        ed_sk: ed25519_dalek::SigningKey,
        pq_sk: MlDsaSigningKey,
    ) -> Self {
        Self {
            ed_sk: Arc::new(ed_sk),
            pq_sk: Some(Arc::new(pq_sk)),
            ledger_id,
            require_pq: true,
        }
    }

    /// Construct a Classical-only (Ed25519) signer. `require_pq` is forced to
    /// `false` — a Classical signer can never enforce the Hybrid policy. This
    /// path exists for tests and constrained environments; production
    /// deployments use [`Self::hybrid`].
    pub fn classical(ledger_id: Did, ed_sk: ed25519_dalek::SigningKey) -> Self {
        Self {
            ed_sk: Arc::new(ed_sk),
            pq_sk: None,
            ledger_id,
            require_pq: false,
        }
    }

    /// The Ed25519 verifying key for this signer (so callers can publish /
    /// register the cell ledger's classical identity).
    pub fn ed25519_verifying_key(&self) -> ed25519_dalek::VerifyingKey {
        self.ed_sk.verifying_key()
    }
}

impl CheckpointSigner for CoseCheckpointSigner {
    fn sign(&self, signing_input: &[u8]) -> Result<Vec<u8>, LedgerError> {
        // Fail-closed Hybrid policy: a Hybrid signer missing its PQ key must
        // never emit a Classical-only signature. `sign_composite` already
        // produces a Classical signature when `pq_sk` is `None`, so we gate it
        // explicitly here to make the policy decision local and visible.
        if self.require_pq && self.pq_sk.is_none() {
            return Err(LedgerError::Internal(
                "Hybrid checkpoint policy requires an ML-DSA-65 key (fail-closed)".to_owned(),
            ));
        }
        let pq_ref = self.pq_sk.as_deref();
        sign_composite(&self.ed_sk, pq_ref, signing_input, &[])
            .map_err(|e| LedgerError::Internal(format!("checkpoint sign failed: {e}")))
    }

    fn ledger_id(&self) -> &Did {
        &self.ledger_id
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use hyprstream_crypto::cose_sign::verify_composite;
    use hyprstream_crypto::pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes, ml_dsa_vk_from_bytes};
    use hyprstream_ledger::journal::{checkpoint_signing_input, SignedCheckpoint};

    fn checkpoint_with_sig(sig: Vec<u8>) -> SignedCheckpoint {
        SignedCheckpoint {
            ledger_id: Did("did:web:cell.test".to_owned()),
            seq: 1,
            head_hash: [0u8; 32],
            balances_root: [0u8; 32],
            pending_root: [0u8; 32],
            ts: 0,
            prev_checkpoint_hash: [0u8; 32],
            sig,
        }
    }

    #[test]
    fn hybrid_signer_produces_verifiable_composite() {
        let ed_sk = ed25519_dalek::SigningKey::from_bytes(&[1u8; 32]);
        let pq_sk = ml_dsa_sk_from_seed(&[2u8; 32]);
        let signer = CoseCheckpointSigner::hybrid(
            Did("did:web:cell.test".to_owned()),
            ed_sk.clone(),
            pq_sk.clone(),
        );
        // Build the sig-free checkpoint, derive its signing input, sign, then
        // assemble and verify — mirrors the journal roundtrip test.
        let mut cp = checkpoint_with_sig(Vec::new());
        let input = cp.signing_input().unwrap();
        cp.sig = signer.sign(&input).unwrap();

        let ed_vk = ed_sk.verifying_key();
        let pq_vk = ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&pq_sk)).unwrap();
        let verify_input = cp.signing_input().unwrap();
        assert!(verify_composite(&cp.sig, &ed_vk, Some(&pq_vk), &verify_input, &[], true,).is_ok());

        // Tamper: a changed field alters the signing input ⇒ verify fails.
        cp.balances_root = [0xFF; 32];
        let tampered_input = cp.signing_input().unwrap();
        assert!(
            verify_composite(&cp.sig, &ed_vk, Some(&pq_vk), &tampered_input, &[], true).is_err()
        );
    }

    #[test]
    fn classical_only_signer_fails_closed_when_pq_required() {
        let ed_sk = ed25519_dalek::SigningKey::from_bytes(&[1u8; 32]);
        // A classical signer cannot satisfy the Hybrid policy.
        let classical = CoseCheckpointSigner::classical(Did("did:web:cell.test".to_owned()), ed_sk);
        // require_pq is forced false for classical signers, so signing succeeds
        // (Classical policy). The fail-closed path is exercised by the next test.
        let input = checkpoint_signing_input(&[3u8; 32]);
        assert!(classical.sign(&input).is_ok());

        // Now simulate a Hybrid-policy signer that *should* have a PQ key but
        // lost it — constructed via the hybrid builder then stripped.
        let mut hybrid = CoseCheckpointSigner::hybrid(
            Did("did:web:cell.test".to_owned()),
            ed25519_dalek::SigningKey::from_bytes(&[1u8; 32]),
            ml_dsa_sk_from_seed(&[2u8; 32]),
        );
        hybrid.pq_sk = None; // simulate missing PQ key under a Hybrid policy
        assert!(
            hybrid.sign(&input).is_err(),
            "Hybrid policy with no PQ key must fail closed"
        );
    }
}
