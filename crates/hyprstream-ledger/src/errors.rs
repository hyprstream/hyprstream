//! The ledger error vocabulary.
//!
//! Errors are **data** (no wrapped source types) so an [`crate::Outcome`] is
//! `Clone + PartialEq + Eq` and a replay can be asserted byte-identical to the
//! original — the property the idempotency proptests lean on.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::types::{AccountId, TransferId, UnitId};

/// Every way a ledger operation can fail. All variants are deterministic
/// functions of `(current state, op)` so a replay reproduces them exactly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Error)]
pub enum LedgerError {
    /// A referenced account does not exist.
    #[error("unknown account {0:?}")]
    UnknownAccount(AccountId),

    /// The transfer's unit does not match a touched account's unit (INV-1(b)).
    #[error("unit mismatch: transfer unit {transfer:?} != account unit {account:?}")]
    UnitMismatch {
        /// The unit named on the transfer.
        transfer: UnitId,
        /// The unit denormalized on the account.
        account: UnitId,
    },

    /// `open_account` was called for an existing id but with a different unit.
    #[error("account {id:?} already exists with a different unit")]
    AccountUnitConflict {
        /// The account id whose unit conflicts.
        id: AccountId,
    },

    /// The debit account cannot cover the amount once pending holds are counted
    /// (overdraft floor, plan §2b.1).
    #[error("insufficient balance on {account:?}: need {needed}, available {available}")]
    InsufficientBalance {
        /// The debit account.
        account: AccountId,
        /// Amount requested.
        needed: u128,
        /// Amount actually available.
        available: u128,
    },

    /// Amount was zero (transfers must move a positive amount).
    #[error("amount must be greater than zero")]
    ZeroAmount,

    /// A transfer named the same account on both sides. Double-entry requires two
    /// distinct accounts (the TigerBeetle `accounts_must_be_different` rule) — a
    /// self-transfer is meaningless and would collapse two deltas onto one row.
    #[error("debit and credit accounts must differ ({0:?})")]
    AccountsMustDiffer(AccountId),

    /// A partial post named an amount larger than the reservation.
    #[error("post amount {requested} exceeds reserved {reserved}")]
    PostExceedsReservation {
        /// Requested post amount.
        requested: u128,
        /// The reserved amount.
        reserved: u128,
    },

    /// `credit` was called with a debit side that is not an issuer-liability
    /// account (INV-1: issuance must debit the issuer's own liability).
    #[error("issuance debit account {0:?} is not an issuer-liability account")]
    NotIssuerLiability(AccountId),

    /// A second-phase op named a pending id that does not exist.
    #[error("unknown pending transfer {0:?}")]
    UnknownPendingTransfer(TransferId),

    /// The reservation's deadline passed before this second phase committed
    /// (plan §2b.5 — the loser of the expiry-vs-post race).
    #[error("pending transfer {0:?} expired")]
    PendingTransferExpired(TransferId),

    /// The reservation was already posted by an earlier second phase.
    #[error("pending transfer {0:?} already posted")]
    PendingTransferAlreadyPosted(TransferId),

    /// The reservation was already voided by an earlier second phase.
    #[error("pending transfer {0:?} already voided")]
    PendingTransferAlreadyVoided(TransferId),

    /// A reservation timeout was outside the permitted `[1s, 24h]` band.
    #[error("reservation timeout {0}s out of bounds")]
    TimeoutOutOfBounds(u32),

    /// A replay arrived for a `TransferId` whose outcome row was already pruned
    /// past the retention horizon (plan §2c). Distinguishable from "never seen"
    /// so a caller escalates rather than blindly re-executing. (MemLedger never
    /// prunes, so it never returns this; retention lands with a later work item.)
    #[error("transfer id {0:?} is older than the outcome-retention horizon")]
    IdTooOld(TransferId),

    /// A replay was detected. The reference contract is **transparent replay**
    /// (the stored [`crate::Outcome`] is returned verbatim), so MemLedger does
    /// not surface this; it exists for backends that prefer an explicit signal,
    /// carrying the original outcome.
    #[error("duplicate transfer id {id:?}")]
    DuplicateTransferId {
        /// The replayed id.
        id: TransferId,
        /// The original recorded outcome, boxed to keep the enum small.
        original: Box<crate::types::Outcome>,
    },

    /// An internal invariant/encoding failure. Fail-closed: the op is rejected
    /// without mutating state, and is retryable.
    #[error("internal ledger error: {0}")]
    Internal(String),
}
