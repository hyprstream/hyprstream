//! Sealing the mint: issuance requires an unforgeable, key-bound capability.
//!
//! Issuance is the only operation that grows a unit's supply (INV-1), so "who
//! may call this" is a monetary question rather than an access-control
//! convenience.
//!
//! ## What the seal is, and what defeated the previous attempt
//!
//! An earlier revision gated `credit` on a [`MintCapability`] with a private
//! field, but obtained that capability through an injectable `MintVerifier`
//! **trait** plus a public `with_mint_verifier` setter. That is not a seal: a
//! downstream crate could implement the trait as `fn verify(..) { Ok(()) }`,
//! install it on an existing backend, and mint arbitrary amounts from
//! meaningless bytes. Making the *capability* unforgeable is pointless if the
//! *authority that mints capabilities* is publicly replaceable.
//!
//! So there is no trait and no setter. Verification is a concrete in-crate
//! function over concrete key material:
//!
//! - [`MintAuthority`] holds **verifying keys**, not behaviour. There is no
//!   downstream-implementable extension point to swap the check for `Ok(())`.
//! - The authority is supplied **once, at backend construction**, and is
//!   immutable thereafter. Holding a `&mut` backend does not let you change
//!   what it will accept.
//! - [`MintCapability`] has no public constructor. The only route is
//!   [`LedgerBackend::authorize_mint`](crate::LedgerBackend::authorize_mint),
//!   which verifies a hybrid-PQC composite signature over the canonical,
//!   domain-separated encoding of **that exact transfer** against the
//!   authority the backend was built with.
//!
//! The residual power a downstream crate has is to stand up *its own* ledger
//! with *its own* keys, which is not an escalation — it mints only into its own
//! database, against a supply it already owns. What it cannot do is mint
//! against a ledger whose authority's signing key it does not hold.
//!
//! ## Why hybrid-PQC, concretely and not behind a seam
//!
//! Minted issuance authorizations are exactly the artifact class that must be
//! EdDSA + ML-DSA-65 composite rather than classical-only. `hyprstream-crypto`
//! is an unconditional, WASM-safe dependency of this crate (the same one
//! `verify_checkpoint_signature` already uses), so the real verifier lives
//! here. Nothing needs to be injected from outside, which is precisely what
//! removes the injection point.
//!
//! A backend built with **no** authority cannot mint at all — the fail-closed
//! direction. An operator who has not wired an issuance authority gets a
//! mint-disabled ledger, never a mint-open one.

use crate::errors::LedgerError;
use crate::types::IssueTransfer;
use ciborium::value::Value;
use hyprstream_crypto::pq::MlDsaVerifyingKey;

/// Domain-separation tag for the issuance-authorization signing input. Kept
/// distinct from every other signed body in the system so a signature over one
/// artifact can never be replayed as authorization for another.
const MINT_DOMAIN: &str = "hs-ledger-mint-authorization-v1";

/// The key material a ledger checks issuance authorizations against.
///
/// Deliberately **data, not behaviour**: there is no trait here, so there is
/// nothing for a downstream crate to implement permissively. Supplied at
/// backend construction and immutable afterwards.
#[derive(Clone)]
pub struct MintAuthority {
    ed_vk: ed25519_dalek::VerifyingKey,
    pq_vk: Option<MlDsaVerifyingKey>,
    require_pq: bool,
}

impl std::fmt::Debug for MintAuthority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MintAuthority")
            .field("require_pq", &self.require_pq)
            .field("has_pq_key", &self.pq_vk.is_some())
            .finish()
    }
}

impl MintAuthority {
    /// The production constructor: hybrid EdDSA + ML-DSA-65.
    ///
    /// A classical-only authorization is refused under this authority, so a
    /// missing PQ arm fails closed instead of silently downgrading.
    pub fn hybrid(ed_vk: ed25519_dalek::VerifyingKey, pq_vk: MlDsaVerifyingKey) -> Self {
        Self {
            ed_vk,
            pq_vk: Some(pq_vk),
            require_pq: true,
        }
    }

    /// Classical-only (Ed25519). For tests and constrained environments —
    /// production uses [`Self::hybrid`], matching the checkpoint-signer rule.
    pub fn classical(ed_vk: ed25519_dalek::VerifyingKey) -> Self {
        Self {
            ed_vk,
            pq_vk: None,
            require_pq: false,
        }
    }

    /// Verify an issuance authorization. Concrete, in-crate, not overridable.
    fn verify(&self, signing_input: &[u8], sig: &[u8]) -> Result<(), LedgerError> {
        hyprstream_crypto::cose_sign::verify_composite(
            sig,
            &self.ed_vk,
            self.pq_vk.as_ref(),
            signing_input,
            &[],
            self.require_pq,
        )
        .map(|_| ())
        .map_err(|e| LedgerError::MintNotAuthorized(format!("issuance authorization invalid: {e}")))
    }
}

/// The canonical, domain-separated bytes an issuance authorization signs.
///
/// Every field that determines what is minted, and where it lands, is bound:
/// changing the amount, the destination, the unit, or the idempotency key
/// changes the signing input and invalidates the signature. The encoding is
/// canonical DAG-CBOR in a fixed array order (the same discipline
/// [`crate::AccountId::derive`] uses), so an independent implementation can
/// reproduce these bytes exactly.
///
/// ```text
/// [
///   "hs-ledger-mint-authorization-v1",
///   <transfer_id  : bstr(16)>,   // big-endian u128
///   <issuer_liab  : bstr(16)>,   // big-endian u128
///   <destination  : bstr(16)>,   // big-endian u128
///   <unit_issuer  : tstr>,
///   <unit_class   : tstr>,
///   <amount       : bstr(16)>,   // big-endian u128
///   <grant_cid    : bstr>,       // empty when absent
///   <user_data    : bstr(32)>
/// ]
/// ```
pub fn mint_signing_input(t: &IssueTransfer) -> Result<Vec<u8>, LedgerError> {
    let value = Value::Array(vec![
        Value::Text(MINT_DOMAIN.to_owned()),
        Value::Bytes(t.id.0.to_be_bytes().to_vec()),
        Value::Bytes(t.issuer_liability.0.to_be_bytes().to_vec()),
        Value::Bytes(t.destination.0.to_be_bytes().to_vec()),
        Value::Text(t.unit.issuer.0.clone()),
        Value::Text(t.unit.resource_class.clone()),
        Value::Bytes(t.amount.to_be_bytes().to_vec()),
        Value::Bytes(
            t.grant_cid
                .as_ref()
                .map(|c| c.0.clone())
                .unwrap_or_default(),
        ),
        Value::Bytes(t.user_data.to_vec()),
    ]);
    let mut buf = Vec::new();
    ciborium::into_writer(&value, &mut buf)
        .map_err(|e| LedgerError::Internal(format!("mint signing input encode failed: {e}")))?;
    Ok(buf)
}

/// Proof that a specific issuance was authorized by the ledger's mint authority.
///
/// No public constructor, and no public route to one other than
/// [`LedgerBackend::authorize_mint`](crate::LedgerBackend::authorize_mint).
/// It borrows the transfer it authorized, so a capability obtained for one
/// issuance cannot be spent on another — that is a type error, not a runtime
/// check.
#[derive(Debug)]
pub struct MintCapability<'a> {
    transfer: &'a IssueTransfer,
}

impl MintCapability<'_> {
    /// The issuance this capability authorizes — the only one it can be spent
    /// on, since the backend mints exactly this value.
    pub fn transfer(&self) -> &IssueTransfer {
        self.transfer
    }
}

/// Verify an issuance authorization and mint the capability on success.
///
/// Crate-private on purpose: backends call it from `authorize_mint`, and there
/// is deliberately no public route to a [`MintCapability`]. An absent authority
/// refuses everything.
pub(crate) fn authorize<'a>(
    authority: Option<&MintAuthority>,
    transfer: &'a IssueTransfer,
    sig: &[u8],
) -> Result<MintCapability<'a>, LedgerError> {
    let authority = authority.ok_or_else(|| {
        LedgerError::MintNotAuthorized(
            "no issuance authority is configured for this ledger".to_owned(),
        )
    })?;
    let input = mint_signing_input(transfer)?;
    authority.verify(&input, sig)?;
    Ok(MintCapability { transfer })
}

#[cfg(test)]
mod tests {
    // A test asserting a known-good value legitimately unwraps.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::types::{AccountId, Cid, Did, TransferId, UnitId};
    use hyprstream_crypto::cose_sign::sign_composite;
    use hyprstream_crypto::pq::ml_dsa_generate_keypair;

    fn xfer(amount: u128) -> IssueTransfer {
        IssueTransfer {
            id: TransferId(7),
            issuer_liability: AccountId(1),
            destination: AccountId(2),
            unit: UnitId {
                issuer: Did("did:web:issuer.test".to_owned()),
                resource_class: "gpu.h100.seconds".to_owned(),
            },
            amount,
            grant_cid: Some(Cid(vec![9, 9])),
            user_data: [3u8; 32],
        }
    }

    #[test]
    fn signing_input_is_deterministic() {
        assert_eq!(
            mint_signing_input(&xfer(100)).unwrap(),
            mint_signing_input(&xfer(100)).unwrap()
        );
    }

    #[test]
    fn signing_input_binds_the_amount() {
        assert_ne!(
            mint_signing_input(&xfer(100)).unwrap(),
            mint_signing_input(&xfer(101)).unwrap(),
            "an authorization for one amount must not cover another"
        );
    }

    #[test]
    fn signing_input_binds_the_destination() {
        let mut other = xfer(100);
        other.destination = AccountId(3);
        assert_ne!(
            mint_signing_input(&xfer(100)).unwrap(),
            mint_signing_input(&other).unwrap(),
            "an authorization must not be redirectable to another account"
        );
    }

    #[test]
    fn an_absent_authority_refuses_everything() {
        let t = xfer(100);
        let err = authorize(None, &t, &[0u8; 64]).unwrap_err();
        assert!(matches!(err, LedgerError::MintNotAuthorized(_)));
    }

    /// The exact shape of the R4 exploit: arbitrary transfer, meaningless bytes.
    #[test]
    fn garbage_bytes_are_refused() {
        let ed_sk = ed25519_dalek::SigningKey::from_bytes(&[11u8; 32]);
        let (_pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let authority = MintAuthority::hybrid(ed_sk.verifying_key(), pq_vk);
        let t = xfer(1_000_000);
        for sig in [Vec::new(), vec![0u8; 64], b"not a signature".to_vec()] {
            let err = authorize(Some(&authority), &t, &sig).unwrap_err();
            assert!(
                matches!(err, LedgerError::MintNotAuthorized(_)),
                "meaningless bytes must not authorize a mint"
            );
        }
    }

    #[test]
    fn a_hybrid_signature_authorizes_exactly_its_own_transfer() {
        let ed_sk = ed25519_dalek::SigningKey::from_bytes(&[11u8; 32]);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let authority = MintAuthority::hybrid(ed_sk.verifying_key(), pq_vk);

        let t = xfer(100);
        let sig =
            sign_composite(&ed_sk, Some(&pq_sk), &mint_signing_input(&t).unwrap(), &[]).unwrap();
        assert_eq!(
            authorize(Some(&authority), &t, &sig).unwrap().transfer(),
            &t
        );

        // The same signature must not carry over to a larger issuance.
        let bigger = xfer(1_000_000);
        assert!(
            authorize(Some(&authority), &bigger, &sig).is_err(),
            "an authorization must not cover a different issuance"
        );
    }

    /// An attacker holding their own keypair gains nothing: the authority is
    /// fixed at construction and cannot be swapped.
    #[test]
    fn a_signature_from_another_key_is_refused() {
        let ed_sk = ed25519_dalek::SigningKey::from_bytes(&[11u8; 32]);
        let (_pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let authority = MintAuthority::hybrid(ed_sk.verifying_key(), pq_vk);

        let attacker_ed = ed25519_dalek::SigningKey::from_bytes(&[99u8; 32]);
        let (attacker_pq, _) = ml_dsa_generate_keypair();
        let t = xfer(1_000_000);
        let sig = sign_composite(
            &attacker_ed,
            Some(&attacker_pq),
            &mint_signing_input(&t).unwrap(),
            &[],
        )
        .unwrap();

        let err = authorize(Some(&authority), &t, &sig).unwrap_err();
        assert!(matches!(err, LedgerError::MintNotAuthorized(_)));
    }

    #[test]
    fn a_classical_only_signature_is_refused_under_a_hybrid_authority() {
        let ed_sk = ed25519_dalek::SigningKey::from_bytes(&[11u8; 32]);
        let (_pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let authority = MintAuthority::hybrid(ed_sk.verifying_key(), pq_vk);

        let t = xfer(100);
        let classical =
            sign_composite(&ed_sk, None, &mint_signing_input(&t).unwrap(), &[]).unwrap();

        let err = authorize(Some(&authority), &t, &classical).unwrap_err();
        assert!(
            matches!(err, LedgerError::MintNotAuthorized(_)),
            "a hybrid authority must reject a classical-only authorization"
        );
    }
}
