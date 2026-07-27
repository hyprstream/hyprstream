//! Sealing the mint: issuance requires an unforgeable capability.
//!
//! Issuance is the only operation that grows a unit's supply (INV-1), so it is
//! the one place where "who may call this" is a monetary question rather than
//! an access-control convenience. Before this module, [`LedgerBackend::credit`]
//! took a plain [`IssueTransfer`]: any code holding a backend could mint, and
//! the authority token in the service layer only narrowed one *particular*
//! caller rather than the operation itself.
//!
//! ## The seal
//!
//! [`MintCapability`] has no public constructor and cannot be built outside
//! this crate. The only way to obtain one is
//! [`LedgerBackend::authorize_mint`](crate::LedgerBackend::authorize_mint),
//! which checks a signature over the canonical, domain-separated encoding of
//! **that exact transfer** against the [`MintVerifier`] the backend was
//! constructed with. Two consequences follow:
//!
//! - A downstream crate cannot mint by depending on `hyprstream-ledger`. There
//!   is no path from `IssueTransfer` to `credit` that does not pass through the
//!   backend's own verifier.
//! - A capability cannot be moved between transfers. It borrows the transfer it
//!   authorized, so authorizing a 1-unit issuance and then minting 10^9 does
//!   not typecheck, let alone verify.
//!
//! ## Why a verifier seam rather than a fixed algorithm
//!
//! This crate is MIT and WASM-clean: it cannot reach the hybrid-PQC COSE stack
//! in `hyprstream-rpc`. It uses the same shape it already uses for signed
//! checkpoints — define the signing input and the verification seam here,
//! inject the concrete hybrid implementation from the TCB. Minted issuance
//! authorizations are exactly the class of artifact that must be
//! EdDSA + ML-DSA-65 composite rather than classical-only, so a production
//! backend must be built with a hybrid verifier. The default is
//! [`DenyAllMintVerifier`]: a backend nobody deliberately gave a verifier
//! refuses to mint at all, rather than minting freely.

use crate::errors::LedgerError;
use crate::types::IssueTransfer;
use ciborium::value::Value;

/// Domain-separation tag for the issuance-authorization signing input. Kept
/// distinct from every other signed body in the system so a signature over one
/// artifact can never be replayed as authorization for another.
const MINT_DOMAIN: &str = "hs-ledger-mint-authorization-v1";

/// Verifies issuance authorizations for a backend.
///
/// **Production implementations must verify a hybrid-PQC composite signature**
/// (EdDSA + ML-DSA-65), matching the rule for other minted/authorizing
/// artifacts. A classical-only verifier is acceptable only in tests.
pub trait MintVerifier: Send + std::fmt::Debug {
    /// Verify `sig` over `signing_input`, which is always the output of
    /// [`mint_signing_input`] for the transfer being authorized.
    ///
    /// Returns `Ok(())` only on a good signature. Any failure — bad signature,
    /// unknown key, missing PQ arm under a hybrid policy — must be an error.
    fn verify(&self, signing_input: &[u8], sig: &[u8]) -> Result<(), LedgerError>;
}

/// The default verifier: refuses everything.
///
/// A backend constructed without an explicit issuance verifier cannot mint.
/// This is the fail-closed direction: an operator who has not yet wired the
/// issuance authority gets a ledger that rejects minting, never one that mints
/// for anyone.
#[derive(Debug, Default, Clone, Copy)]
pub struct DenyAllMintVerifier;

impl MintVerifier for DenyAllMintVerifier {
    fn verify(&self, _signing_input: &[u8], _sig: &[u8]) -> Result<(), LedgerError> {
        Err(LedgerError::MintNotAuthorized(
            "no issuance verifier is configured for this ledger".to_owned(),
        ))
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

/// Proof that a specific issuance was authorized by the ledger's mint verifier.
///
/// Constructible only inside this crate, and only by verifying a signature over
/// the transfer it borrows. Passing one to
/// [`LedgerBackend::credit`](crate::LedgerBackend::credit) is what makes the
/// mint reachable at all.
#[derive(Debug)]
pub struct MintCapability<'a> {
    transfer: &'a IssueTransfer,
}

impl<'a> MintCapability<'a> {
    /// The issuance this capability authorizes — the only one it can be spent
    /// on, since the backend mints exactly this value.
    pub fn transfer(&self) -> &IssueTransfer {
        self.transfer
    }
}

/// Verify an issuance authorization and mint the capability on success.
///
/// Crate-private on purpose: backends call it from `authorize_mint`, and there
/// is deliberately no public route to a [`MintCapability`].
pub(crate) fn authorize<'a>(
    verifier: &dyn MintVerifier,
    transfer: &'a IssueTransfer,
    sig: &[u8],
) -> Result<MintCapability<'a>, LedgerError> {
    let input = mint_signing_input(transfer)?;
    verifier.verify(&input, sig)?;
    Ok(MintCapability { transfer })
}

#[cfg(test)]
mod tests {
    // A test asserting a known-good value legitimately unwraps.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::types::{AccountId, Cid, Did, TransferId, UnitId};

    #[derive(Debug)]
    struct AcceptAll;
    impl MintVerifier for AcceptAll {
        fn verify(&self, _i: &[u8], _s: &[u8]) -> Result<(), LedgerError> {
            Ok(())
        }
    }

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
    fn the_default_verifier_refuses() {
        let t = xfer(100);
        let err = authorize(&DenyAllMintVerifier, &t, &[0u8; 64]).unwrap_err();
        assert!(matches!(err, LedgerError::MintNotAuthorized(_)));
    }

    #[test]
    fn a_verified_authorization_yields_a_capability_for_that_transfer() {
        let t = xfer(100);
        let cap = authorize(&AcceptAll, &t, &[0u8; 64]).unwrap();
        assert_eq!(cap.transfer(), &t);
    }
}
