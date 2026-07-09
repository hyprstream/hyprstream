//! The `LedgerBackend` trait — the load-bearing accounting-plane surface every
//! backend (MemLedger now; RocksLedger, TigerBeetleLedger later) implements, and
//! the enforcer builds on (plan §3.2).
//!
//! PLAN DECISION 8: the trait is **synchronous and blocking**. Every backend is
//! single-writer anyway (plan §2a), so the service layer owns the actor thread
//! and exposes the async facade (`LedgerHandle::submit → oneshot`). Keeping the
//! core sync is what keeps it tokio-free (the WASM requirement), matches
//! RocksDB's sync API, and confines TigerBeetle's async client inside its feature
//! gate.

use crate::errors::LedgerError;
use crate::journal::{
    ChainHead, CheckpointSigner, JournalEntry, OutboxItem, OutboxSeq, SignedCheckpoint, TickReport,
};
use crate::types::{
    Account, AccountId, AccountSpec, BalanceView, IssueTransfer, Outcome, Transfer, TransferId,
};

/// A single-writer double-entry ledger over one consistency domain (one cell).
///
/// # Idempotency contract (plan §2c)
///
/// Every mutating method carries a client-supplied [`TransferId`]. Replaying a
/// `TransferId` returns the **original** [`Outcome`] verbatim — including the
/// original error. This is a trait contract, property-tested against every
/// backend against the MemLedger oracle. (Backends may return
/// [`LedgerError::IdTooOld`] once the outcome row is pruned past the retention
/// horizon — a later work item; MemLedger keeps everything and never prunes.)
pub trait LedgerBackend: Send {
    /// Idempotent. Creates the account iff absent and returns it; if it already
    /// exists the existing row is returned (its unit must match, else
    /// [`LedgerError::AccountUnitConflict`]).
    fn open_account(&mut self, spec: AccountSpec) -> Result<Account, LedgerError>;

    /// Single-phase issuance: issuer liability → destination. INV-1: the only
    /// entry point that grows a unit's supply; the debit side MUST be the
    /// issuer's `IssuerLiability` account for the unit (checked).
    fn credit(&mut self, t: IssueTransfer) -> Outcome;

    /// Single-phase spend. Overdraft-checked on the debit side (plan §2b.1).
    fn debit(&mut self, t: Transfer) -> Outcome;

    /// Two-phase phase 1: places pending holds on both sides (plan §2b). The
    /// transfer's `id` is also the reservation id used by `post`/`void`.
    fn reserve(&mut self, t: Transfer, timeout_s: u32) -> Outcome;

    /// Two-phase phase 2. `amount: None` = full post; `Some(p)` = partial, the
    /// remainder released. At most one second phase succeeds per reservation.
    fn post(&mut self, id: TransferId, pending: TransferId, amount: Option<u128>) -> Outcome;

    /// Two-phase phase 2 cancel: releases the pending holds.
    fn void(&mut self, id: TransferId, pending: TransferId) -> Outcome;

    /// Read-only balance projection (served from a consistent snapshot; safe
    /// concurrent with the writer).
    fn balance(&self, account: AccountId) -> Result<BalanceView, LedgerError>;

    /// Force a signed checkpoint now and return it. Periodic checkpoints are the
    /// backend's own duty (driven from `tick`).
    fn checkpoint(
        &mut self,
        signer: &dyn CheckpointSigner,
    ) -> Result<SignedCheckpoint, LedgerError>;

    /// Housekeeping: expiry sweep (plan §2b.4), outcome pruning (plan §2c — later
    /// item), and scheduled checkpoints. Called by the actor on an interval.
    fn tick(&mut self, signer: &dyn CheckpointSigner) -> Result<TickReport, LedgerError>;

    /// Ordered peek at committed-but-unemitted proof-plane items (plan §2e).
    fn outbox_peek(&self, max: usize) -> Result<Vec<OutboxItem>, LedgerError>;

    /// Acknowledge (drop) drained outbox items up to and including `up_to`.
    fn outbox_ack(&mut self, up_to: OutboxSeq) -> Result<(), LedgerError>;

    /// Replay/inspection window into the journal (for verify tools and
    /// settlement framing).
    fn journal_range(&self, from_seq: u64, max: usize) -> Result<Vec<JournalEntry>, LedgerError>;

    /// The current head of the hash chain.
    fn head(&self) -> ChainHead;
}
