//! # Settlement/tariff service (AGPL, #1399)
//!
//! The authenticated mesh service that implements the MIT-licensed
//! [`hyprstream_pay`] traits: [`SettlementIssuer`] + [`TariffProvider`].
//!
//! ## Security model
//!
//! 1. **Peer auth**: transport-level (TLS 1.3 on QUIC, UDS peer credentials)
//! 2. **Capability gate**: `#[authorize]` on RPC handlers
//! 3. **Attestation verification**: the PQ-hybrid-signed settlement attestation
//!    is re-verified server-side — never trusts a client assertion
//! 4. **Settlement store**: the restricted DB row must exist and be in
//!    `settlement_committed` state
//! 5. **Restricted credit**: the only path to `LedgerHandle::credit` is through
//!    the [`SettlementAuthority`] sealed token (F7)
//!
//! ## Current status — scoped out to PAY-02
//!
//! This module implements the in-process trait logic (attestation verification
//! → settlement store lookup → restricted credit) but **does NOT yet wire**:
//! - Cap'n Proto-generated server/client bindings (build.rs + capnp deps)
//! - iroh/QUIC RPC handler registration
//! - `#[authorize]` scope enforcement on the RPC surface
//! - Concrete PQ-hybrid `verify_composite` attestation verifier
//! - Durable Postgres `SettlementStore` implementation
//! - Pay service `#[service_factory]` registration
//!
//! These are explicitly **PAY-02** scope. The current module provides the
//! verified-trait-logic skeleton that PAY-02's RPC/verifier wiring will
//! complete.
//!
//! This module is AGPL (part of the `hyprstream` crate). The protocol surface
//! (`hyprstream-pay`) is MIT.

pub mod attestation;
pub mod settlement_store;
pub mod tariff;

pub use attestation::SettlementAttestationVerifier;
pub use settlement_store::{SettlementRow, SettlementStore};
pub use tariff::StaticTariffProvider;

use std::sync::Arc;

use async_trait::async_trait;
use hyprstream_ledger::IssueTransfer;
use hyprstream_pay::{IssueRequest, IssueResponse, PayError, SettlementIssuer};

use super::ledger::{LedgerHandle, SettlementAuthority};

/// The production settlement-issuer service.
///
/// Wires the settlement store + attestation verifier + the restricted
/// ledger credit path. Only this service can mint purchased credits.
pub struct SettlementIssuerService {
    handle: LedgerHandle,
    auth: SettlementAuthority,
    store: Arc<dyn SettlementStore + Send + Sync>,
    verifier: Arc<dyn SettlementAttestationVerifier + Send + Sync>,
}

impl std::fmt::Debug for SettlementIssuerService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SettlementIssuerService").finish_non_exhaustive()
    }
}

impl SettlementIssuerService {
    /// Construct the settlement issuer. The `auth` token is created here —
    /// it's `pub(crate)` so only the factory (inside `hyprstream`) can wire it.
    pub fn new(
        handle: LedgerHandle,
        store: Arc<dyn SettlementStore + Send + Sync>,
        verifier: Arc<dyn SettlementAttestationVerifier + Send + Sync>,
    ) -> Self {
        SettlementIssuerService {
            handle,
            auth: SettlementAuthority::new(),
            store,
            verifier,
        }
    }
}

#[async_trait]
impl SettlementIssuer for SettlementIssuerService {
    async fn issue(&self, req: IssueRequest) -> Result<IssueResponse, PayError> {
        // 1. Verify the attestation server-side (PQ-hybrid signature check).
        //
        // The verified identity is then *bound* to the request rather than
        // discarded: the attestation is self-contained and carries its own
        // settlement id, so a blob that verifies but attests to a different
        // settlement must be refused here. Trusting the caller-supplied id
        // while only signature-checking the blob would let a valid attestation
        // for a cheap settlement authorize issuance against an expensive one.
        let verified = self
            .verifier
            .verify(&req.attestation, &req.settlement_id)
            .await?;
        if verified.settlement_id != req.settlement_id {
            return Err(PayError::AttestationInvalid(format!(
                "attestation attests to settlement {} but was presented for {}",
                verified.settlement_id, req.settlement_id
            )));
        }

        // 2. Load the committed settlement row from the restricted store.
        let row = self
            .store
            .get_settlement(&req.settlement_id)
            .await?
            .ok_or_else(|| PayError::SettlementNotFound(req.settlement_id.clone()))?;

        if !row.is_committed() {
            return Err(PayError::SettlementNotCommitted(
                req.settlement_id.clone(),
                row.state_label(),
            ));
        }

        // 3. Derive issuance fields from the settlement row (server-side,
        //    never client-supplied).
        let transfer_id = row.derive_transfer_id();
        let issuer_liability = row.issuer_liability_account;
        let destination = row.destination_account;
        let unit = row.unit.clone();
        let amount = row.amount_minor;
        let grant_cid = row.grant_cid.clone();

        // 4. Mark settlement as issuance-pending BEFORE attempting the credit.
        //    R2-F8: persist settlement state transitions with recovery semantics.
        self.store
            .mark_issuance_pending(&req.settlement_id, transfer_id)
            .await?;

        // 5. Construct the internal IssueTransfer and call the restricted
        //    credit path (F7: only SettlementAuthority can do this).
        let issue = IssueTransfer {
            id: transfer_id,
            issuer_liability,
            destination,
            unit: hyprstream_ledger::UnitId {
                issuer: hyprstream_ledger::Did(unit.issuer_did),
                resource_class: unit.resource_class,
            },
            amount,
            grant_cid: grant_cid.map(hyprstream_ledger::Cid),
            user_data: [0u8; 32], // correlation: settlement_id hash
        };

        let outcome = self.handle.credit(&self.auth, issue).await;

        // 6. If issuance succeeded, mark the settlement as issued.
        if outcome.is_ok() {
            self.store
                .mark_issued(&req.settlement_id)
                .await?;
        }

        Ok(IssueResponse {
            transfer_id_lo: (transfer_id.0 & u64::MAX as u128) as u64,
            transfer_id_hi: (transfer_id.0 >> 64) as u64,
            outcome_seq: outcome.seq,
            ok: outcome.is_ok(),
            error: outcome.result.err().map(|e| e.to_string()),
        })
    }

    async fn status(&self, settlement_id: &str) -> Result<IssueResponse, PayError> {
        let row = self
            .store
            .get_settlement(settlement_id)
            .await?
            .ok_or_else(|| PayError::SettlementNotFound(settlement_id.to_owned()))?;

        let transfer_id = row.derive_transfer_id();

        // R2-F8: report success ONLY if the settlement has actually been issued
        // (not merely committed). A committed-but-not-issued settlement is
        // NOT success — it means issuance has not completed yet.
        let is_issued = matches!(
            row.state,
            crate::services::pay::settlement_store::SettlementState::Issued
        );

        Ok(IssueResponse {
            transfer_id_lo: (transfer_id.0 & u64::MAX as u128) as u64,
            transfer_id_hi: (transfer_id.0 >> 64) as u64,
            outcome_seq: 0, // TODO(PAY-02): query the ledger outcome for this transfer_id
            ok: is_issued,
            error: if is_issued {
                None
            } else {
                Some(row.state_label())
            },
        })
    }
}
