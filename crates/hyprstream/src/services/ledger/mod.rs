//! Phase-1 cellular-ledger **local-enforcer** service (epic #922, issue #925).
//!
//! The scheduler stops owning quota and becomes the Phase-1 contract:
//! *verify a presented capability → spend a credit → emit a receipt*
//! (plan §5). This module is the service-layer home for the three Phase-1
//! work items:
//!
//! - **1.6 — ledger service** ([`actor`] + [`handle::LedgerHandle`],
//!   [`signer::CoseCheckpointSigner`], [`sink`] receipt emitter +
//!   [`sink::DebtBreaker`] ReceiptDebt fail-closed breaker, and the
//!   [`service::LedgerService`] `Spawnable` + `#[service_factory]`).
//! - **1.7 — [`CreditGate`]** (plan §5.2/.3): grant cache + balance cells +
//!   generation-counter revocation + spend-authorization verification.
//! - **1.8 — [`enforcer::LocalEnforcer`]** (plan §5.1/.4/.5): the realigned
//!   admission contract — verify → gate → `transfer_id` in the result,
//!   reject-don't-queue, with reserve/post/void wiring against the
//!   [`LedgerHandle`](handle::LedgerHandle).
//!
//! Everything here is gated behind the `ledger` cargo feature and the runtime
//! [`LedgerConfig::enabled`] flag, both **default off**. The
//! `hyprstream-workers` scheduler quota path is untouched until an operator
//! opts in; when off, the subsystem is inert (nothing is enforced, nothing is
//! bypassed). INV-1 (credits are issuer liabilities) is upheld by
//! `hyprstream-ledger`'s engine; INV-2 (no ledger tier on the hot path) is
//! upheld by [`CreditGate::try_hold`] operating on atomics only — the durable
//! reserve is async and off the admit path.
//!
//! The Phase-1 backend is [`hyprstream_ledger::MemLedger`] (the reference
//! oracle). RocksLedger (plan item 1.2) lands separately. The grant verifier
//! seam ([`credit_gate::GrantVerifier`]) defaults to a test double; wiring it
//! to `hyprstream_rpc::auth::ucan` chain validation + the
//! `ai.hyprstream.ledger.allocation` lexicon is a follow-up that the
//! `#[service_factory]` marks clearly.

pub mod actor;
pub mod credit_gate;
pub mod enforcer;
pub mod handle;
pub mod service;
pub mod signer;
pub mod sink;

pub use credit_gate::{
    CreditGate, DenyReason, GrantVerifier, Hold, SpendAuthorization, StaticGrantVerifier,
    VerifiedGrant,
};
pub use enforcer::{AdmissionRequest, AdmissionResult, LocalEnforcer, Rejection};
pub use handle::LedgerHandle;
pub use service::LedgerService;
pub use signer::CoseCheckpointSigner;
pub use sink::{DebtBreaker, LoggingReceiptSink, ReceiptPayload, ReceiptSink};

use serde::{Deserialize, Serialize};

/// Phase-1 local-enforcer configuration (epic #922 / #925).
///
/// `enabled` defaults to **false**: the whole subsystem is inert until an
/// operator opts in, so the scheduler quota path is byte-for-byte unchanged
/// for everyone who does not set `[ledger] enabled = true`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LedgerConfig {
    /// Master switch. When `false` (default) the ledger service is not
    /// started and the [`CreditGate`] never gates admission — the scheduler
    /// behaves exactly as before #925.
    pub enabled: bool,
    /// Receipt-debt fail-closed threshold (plan §2e / Appendix A.5): if the
    /// oldest unemitted receipt has been outstanding for longer than this, the
    /// enforcer's epoch is bumped with reason `ReceiptDebt`, flipping
    /// admission for receipt-requiring spends to fail-closed until the outbox
    /// drains. Default 15 minutes.
    pub receipt_debt_age_secs: u64,
    /// Receipt-debt depth threshold: outbox depth above this trips the same
    /// fail-closed breaker. Default 10 000.
    pub receipt_debt_max: usize,
    /// Periodic housekeeping interval for the actor's `tick` (expiry sweep +
    /// scheduled checkpoint) and the receipt-emitter drain. Default 10s.
    pub tick_interval_secs: u64,
    /// Default two-phase reservation timeout for an admitted spend (the hold
    /// an in-flight job places on capacity). Bounded `[1s, 24h]` by the
    /// engine. Default 5 minutes.
    pub reserve_timeout_secs: u32,
    /// Commit-count checkpoint cadence (plan §2d). Default 4096.
    pub checkpoint_every_n: u64,
    /// Wall-time checkpoint cadence. Default 60s.
    pub checkpoint_every_t_secs: u64,
    /// Require a PQ (ML-DSA-65) signature on checkpoints — the `CryptoPolicy`
    /// selection. When `true`, a missing PQ key **fails closed** at sign time
    /// (never silently downgrades to Classical). Default `true`.
    pub require_pq_signatures: bool,
}

impl Default for LedgerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            receipt_debt_age_secs: 15 * 60,
            receipt_debt_max: 10_000,
            tick_interval_secs: 10,
            reserve_timeout_secs: 5 * 60,
            checkpoint_every_n: 4096,
            checkpoint_every_t_secs: 60,
            require_pq_signatures: true,
        }
    }
}

impl LedgerConfig {
    /// Whether the local-enforcer is active. Convenience for call sites that
    /// only need the on/off answer.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}
