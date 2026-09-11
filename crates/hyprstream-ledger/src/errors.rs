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

    /// Another process already holds the single-writer lease for this cell.
    ///
    /// A durable backend admits **exactly one** writer per cell for the whole
    /// instance lifetime (the premise [`crate::engine`] is written against).
    /// A second instance is rejected at construction rather than allowed to
    /// stage from its own mirror.
    #[error("writer lease for this cell is held by another instance (epoch {epoch})")]
    WriterLeaseHeld {
        /// The epoch of the lease currently recorded in the database.
        epoch: u64,
    },

    /// This instance's writer lease was taken over by another instance.
    ///
    /// The fencing epoch recorded in the database no longer matches the epoch
    /// this instance acquired, so its mirror may be arbitrarily stale. The op
    /// is refused and the instance poisons itself — it never writes again.
    #[error("writer lease lost: acquired epoch {expected}, database now at {found}")]
    WriterLeaseLost {
        /// The epoch this instance acquired at startup.
        expected: u64,
        /// The epoch currently recorded in the database.
        found: u64,
    },

    /// The in-memory mirror did not describe the state that produced the
    /// committed head, so the staged deltas cannot be persisted.
    ///
    /// Absolute account counters are only meaningful relative to the state
    /// they were computed from. Committing deltas staged against a stale
    /// mirror is exactly the lost-update bug, so the commit is refused inside
    /// the ordering boundary instead.
    #[error("mirror stale: staged against seq {mirror_seq}, database head is seq {db_seq}")]
    MirrorStale {
        /// The head sequence the mirror staged against.
        mirror_seq: u64,
        /// The head sequence actually found in the database.
        db_seq: u64,
    },

    /// The backend refused the op because a prior integrity failure poisoned
    /// this instance. Fail-closed and terminal: the process must restart (and
    /// rebuild from the journal) to write again.
    #[error("ledger instance poisoned, refusing all writes: {0}")]
    Poisoned(String),

    /// A persisted row could not be decoded under the strict representation.
    ///
    /// 128-bit ids and amounts are exactly 16 bytes; anything else is a
    /// legacy or corrupt row. Decoding fails closed rather than truncating or
    /// substituting zero, which would silently alias distinct ids.
    #[error("corrupt persisted row: {0}")]
    CorruptRow(String),

    /// The database schema version is not one this build can serve.
    #[error("incompatible ledger schema: {0}")]
    SchemaIncompatible(String),

    /// An issuance was attempted without a valid authorization.
    ///
    /// Minting is the only supply-growing operation (INV-1), so it is gated on
    /// a verified, transfer-bound capability rather than on the caller merely
    /// holding a backend. See [`crate::mint`].
    #[error("issuance not authorized: {0}")]
    MintNotAuthorized(String),
}
