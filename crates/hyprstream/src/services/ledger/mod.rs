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
pub mod inference_spend;
pub mod service;
pub mod signer;
pub mod sink;

pub use credit_gate::{
    CreditGate, DenyReason, GrantVerifier, Hold, SpendAuthorization, StaticGrantVerifier,
    VerifiedGrant,
};
pub use enforcer::{
    AdmissionRequest, AdmissionResult, AuthenticatedSubject, AuthenticatedSubjectError,
    LocalEnforcer, Rejection,
};
pub use handle::{LedgerHandle, SettlementAuthority};
pub use inference_spend::{
    observe_spend_result, InferenceSpendEmitter, SpendDecline, SpendFailure, SpendInput,
    SpendResult,
};
pub use service::LedgerService;
pub use signer::CoseCheckpointSigner;
pub use sink::{DebtBreaker, LoggingReceiptSink, ReceiptPayload, ReceiptSink};

use serde::{Deserialize, Serialize};

/// Phase-1 local-enforcer configuration (epic #922 / #925).
///
/// `enabled` defaults to **false**: the whole subsystem is inert until an
/// operator opts in, so the scheduler quota path is byte-for-byte unchanged
/// for everyone who does not set `[ledger] enabled = true`.
///
/// **Production activation (PAY-01 F8):** when `enabled = true` and
/// `backend = Postgres`, the factory requires a successful Postgres connection,
/// strict rehydration, and integrity verification. A missing/unavailable
/// Postgres backend is **FATAL** — never silently falls back to MemLedger.
/// MemLedger is only valid behind `backend = Mem` (explicit dev/test).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LedgerConfig {
    /// Master switch. When `false` (default) the ledger service is not
    /// started and the [`CreditGate`] never gates admission — the scheduler
    /// behaves exactly as before #925.
    pub enabled: bool,
    /// Backend selection (PAY-01 F8). **`Postgres` is production; `Mem` is
    /// dev/test only.** When `Postgres` is selected and the connection fails,
    /// startup is **FATAL** — no silent MemLedger fallback.
    pub backend: BackendKind,
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
    /// Declare this deployment as production (PAY-01 R4 finding 4).
    ///
    /// The ledger's defaults are deliberately inert so that a developer machine
    /// behaves exactly as it did before the ledger existed. That inertness is
    /// wrong for a real deployment, where "the accounting plane quietly did not
    /// run" must be a startup failure rather than a silent state. Setting this
    /// converts the permissive defaults into hard requirements — see
    /// [`LedgerConfig::validate_for_production`].
    ///
    /// Also settable out-of-band with `HS_LEDGER_PRODUCTION=1`, so an operator
    /// can enforce it from the environment without editing config.
    pub production: bool,
}

/// Durable backend selection (PAY-01 F8).
///
/// `Postgres` is the only production-safe backend. `Mem` is volatile —
/// dev/test only. The factory **never** silently substitutes one for the
/// other: if `Postgres` is selected and unavailable, startup is FATAL.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    /// In-memory backend (MemLedger). **Dev/test only.** All state is lost
    /// on restart. The factory refuses to start with this backend if
    /// `enabled = true` and no explicit `dev_mode` flag is set (future:
    /// guard via a `--dev` CLI flag or `HS_DEV=1` env var).
    Mem,
    /// Production durable Postgres backend (PostgresLedger). Requires a
    /// valid `ledger_postgres_url` in HyprConfig. Connection/rehydration/
    /// chain-verification failures are **FATAL**.
    #[default]
    Postgres,
}

impl Default for LedgerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            backend: BackendKind::Postgres,
            receipt_debt_age_secs: 15 * 60,
            receipt_debt_max: 10_000,
            tick_interval_secs: 10,
            reserve_timeout_secs: 5 * 60,
            checkpoint_every_n: 4096,
            checkpoint_every_t_secs: 60,
            require_pq_signatures: true,
            production: false,
        }
    }
}

impl LedgerConfig {
    /// Whether the local-enforcer is active. Convenience for call sites that
    /// only need the on/off answer.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Whether this deployment is declared production, from config or from
    /// `HS_LEDGER_PRODUCTION`.
    pub fn is_production(&self) -> bool {
        if self.production {
            return true;
        }
        matches!(
            std::env::var("HS_LEDGER_PRODUCTION").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    }

    /// Convert the permissive development defaults into startup failures.
    ///
    /// Three things are inert-by-default and must not be inert in production:
    /// the ledger being switched off entirely, a volatile in-memory backend,
    /// and a Postgres backend with nowhere to connect. Each of them means the
    /// accounting plane is not actually accounting, so each is fatal here
    /// rather than a log line nobody reads.
    pub fn validate_for_production(&self, postgres_url: Option<&str>) -> anyhow::Result<()> {
        if !self.is_production() {
            return Ok(());
        }
        if !self.enabled {
            anyhow::bail!(
                "ledger: production mode requires [ledger] enabled = true (FATAL —                  credit enforcement would otherwise be bypassed entirely)"
            );
        }
        if self.backend != BackendKind::Postgres {
            anyhow::bail!(
                "ledger: production mode requires [ledger] backend = postgres (FATAL — \
                 backend = {:?} is volatile and would lose all ledger state on restart)",
                self.backend
            );
        }
        if postgres_url.map(str::trim).unwrap_or("").is_empty() {
            anyhow::bail!(
                "ledger: production mode requires a non-empty ledger_postgres_url (FATAL —                  the durable backend has nowhere to connect)"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod production_activation_tests {
    // A test asserting a known-good value legitimately unwraps.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// The development defaults must stay permissive — a developer machine
    /// behaves exactly as it did before the ledger existed.
    #[test]
    fn defaults_are_inert_outside_production() {
        let cfg = LedgerConfig::default();
        assert!(!cfg.enabled);
        assert!(cfg.validate_for_production(None).is_ok());
    }

    /// ...and must become startup failures once production is declared.
    #[test]
    fn production_requires_the_ledger_to_be_enabled() {
        let cfg = LedgerConfig {
            production: true,
            enabled: false,
            ..Default::default()
        };
        let err = cfg
            .validate_for_production(Some("postgres://localhost/ledger"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("enabled = true"), "unexpected error: {err}");
    }

    #[test]
    fn production_refuses_the_volatile_backend() {
        let cfg = LedgerConfig {
            production: true,
            enabled: true,
            backend: BackendKind::Mem,
            ..Default::default()
        };
        let err = cfg
            .validate_for_production(Some("postgres://localhost/ledger"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("backend = postgres"), "unexpected error: {err}");
    }

    #[test]
    fn production_requires_somewhere_to_connect() {
        let cfg = LedgerConfig {
            production: true,
            enabled: true,
            backend: BackendKind::Postgres,
            ..Default::default()
        };
        for url in [None, Some(""), Some("   ")] {
            let err = cfg.validate_for_production(url).unwrap_err().to_string();
            assert!(
                err.contains("ledger_postgres_url"),
                "unexpected error for {url:?}: {err}"
            );
        }
    }

    #[test]
    fn a_fully_configured_production_ledger_validates() {
        let cfg = LedgerConfig {
            production: true,
            enabled: true,
            backend: BackendKind::Postgres,
            ..Default::default()
        };
        cfg.validate_for_production(Some("postgres://localhost/ledger"))
            .unwrap();
    }
}
