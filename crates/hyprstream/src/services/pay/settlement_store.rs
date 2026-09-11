//! Restricted settlement store (#1399, PAY-06).
//!
//! The separate durable state machine for payment settlement, outside PDS.
//! Contains no KYC/legal-identity fields. This is the seam definition;
//! the Postgres impl lands with PAY-06 (Stripe integration).

use hyprstream_ledger::{AccountId, TransferId};
use hyprstream_pay::{PayError, UnitRef};

/// A committed settlement row in the restricted store.
///
/// All issuance fields are derived server-side from this row — the client
/// never supplies amount, destination, unit, or transfer id.
pub struct SettlementRow {
    /// Internal purchase/settlement id.
    pub settlement_id: String,
    /// The deterministic transfer id (blake3 of settlement_id + catalog_version + line_index).
    pub transfer_id: TransferId,
    /// The issuer liability account (debit side for issuance).
    pub issuer_liability_account: AccountId,
    /// The destination account (credit side).
    pub destination_account: AccountId,
    /// The unit being issued.
    pub unit: UnitRef,
    /// Amount in minor units.
    pub amount_minor: u128,
    /// The allocation grant CID this issuance backs (opaque bytes).
    pub grant_cid: Option<Vec<u8>>,
    /// Settlement state.
    pub state: SettlementState,
}

impl SettlementRow {
    /// Whether this settlement is committed (ready for issuance).
    pub fn is_committed(&self) -> bool {
        matches!(self.state, SettlementState::SettlementCommitted)
    }

    /// Human-readable state label.
    pub fn state_label(&self) -> String {
        match self.state {
            SettlementState::Received => "received".into(),
            SettlementState::SignatureVerified => "signature_verified".into(),
            SettlementState::ServerValidated => "server_validated".into(),
            SettlementState::SettlementCommitted => "settlement_committed".into(),
            SettlementState::IssuancePending => "issuance_pending".into(),
            SettlementState::Issued => "issued".into(),
            SettlementState::AllocationPublished => "allocation_published".into(),
            SettlementState::Refunded => "refunded".into(),
            SettlementState::Disputed => "disputed".into(),
        }
    }

    /// Derive the deterministic transfer id for this settlement.
    pub fn derive_transfer_id(&self) -> TransferId {
        self.transfer_id
    }
}

/// The settlement state machine (plan §2.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SettlementState {
    Received,
    SignatureVerified,
    ServerValidated,
    SettlementCommitted,
    IssuancePending,
    Issued,
    AllocationPublished,
    Refunded,
    Disputed,
}

/// The seam for the restricted settlement store. The production impl is a
/// Postgres table in the dedicated ledger database (not the identity store).
#[async_trait::async_trait]
pub trait SettlementStore: Send + Sync {
    /// Look up a settlement row by id.
    async fn get_settlement(&self, id: &str) -> Result<Option<SettlementRow>, PayError>;

    /// Mark a settlement as issuance-pending (after the ledger transfer is
    /// dispatched but before the outcome is confirmed).
    async fn mark_issuance_pending(&self, id: &str, transfer_id: TransferId) -> Result<(), PayError>;

    /// Mark a settlement as issued (after the ledger outcome confirms success).
    async fn mark_issued(&self, id: &str) -> Result<(), PayError>;
}

/// A deny-all store (bootstrap — production uses Postgres).
pub struct DenyAllSettlementStore;

#[async_trait::async_trait]
impl SettlementStore for DenyAllSettlementStore {
    async fn get_settlement(&self, id: &str) -> Result<Option<SettlementRow>, PayError> {
        Err(PayError::SettlementNotFound(id.to_owned()))
    }

    async fn mark_issuance_pending(&self, _id: &str, _tid: TransferId) -> Result<(), PayError> {
        Err(PayError::Internal("no settlement store configured".into()))
    }

    async fn mark_issued(&self, _id: &str) -> Result<(), PayError> {
        Err(PayError::Internal("no settlement store configured".into()))
    }
}
