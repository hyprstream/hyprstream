//! # hyprstream-ledger
//!
//! The **accounting plane** of the cellular-ledger stack (epic #922, plan item
//! 1.1): an embedded, single-writer, double-entry credit engine that
//! deliberately mirrors TigerBeetle's data model.
//!
//! ## The two invariants (D8)
//!
//! - **INV-1 — credits are issuer liabilities, not bearer tokens.** Every unit
//!   [`UnitId`] names its issuer; every account id bakes that unit in; issuance is
//!   the only supply-growing entry point and it debits the issuer's own liability.
//!   There is no representation of value that omits the issuer.
//! - **INV-2 — no ledger tier on the inference hot path.** This crate is
//!   synchronous and knows nothing of admission; the enforcer's `CreditGate`
//!   (a later item) depends on the *types* here, never on a backend, and talks to
//!   the ledger only through an async actor facade in the service layer.
//!
//! ## What is in this crate (item 1.1)
//!
//! - [`types`] — the account model, units, transfers, outcomes (plan §2.0).
//! - [`errors`] — the deterministic error vocabulary.
//! - [`engine`] — the pure, backend-agnostic transfer state machine (plan §2b),
//!   shared verbatim by every backend so "same ops ⇒ same outcomes" holds by
//!   construction.
//! - [`journal`] — journal / checkpoint / outbox types and the
//!   [`CheckpointSigner`](journal::CheckpointSigner) injection seam.
//! - [`backend::LedgerBackend`] — the load-bearing trait.
//! - [`mem::MemLedger`] — the complete in-memory reference implementation and
//!   proptest oracle; WASM-friendly (no tokio, no I/O, logical clock only).
//!
//! RocksLedger, the persisted hash-chained journal + offline verify tool, the
//! transactional-outbox emitter, and TigerBeetleLedger are later work items.
//!
//! ## Crate boundaries (D3)
//!
//! No RPC / VFS / PDS / tokio dependencies. `hyprstream-crypto` (#920, WASM-safe)
//! is an **unconditional** dependency — signed checkpoints are load-bearing, so a
//! build that cannot verify them must not exist (no feature gate, ratified).

pub mod backend;
pub mod engine;
pub mod errors;
pub mod journal;
pub mod mem;
pub mod types;

pub use backend::LedgerBackend;
pub use errors::LedgerError;
pub use journal::{
    balances_root, checkpoint_signing_input, pending_root, verify_checkpoint_signature, ChainHead,
    CheckpointSigner, JournalEntry, OutboxItem, OutboxKind, OutboxSeq, SignedCheckpoint, TickReport,
};
pub use mem::MemLedger;
pub use types::{
    Account, AccountFlags, AccountId, AccountSpec, BalanceView, Cid, Did, IssueTransfer, Outcome,
    PendingReservation, PendingState, Purpose, Transfer, TransferId, TransferResult, UnitId,
};
