//! Receipt emission + the **ReceiptDebt** fail-closed breaker (plan §2e, item
//! 1.6).
//!
//! The outbox is drained asynchronously by the actor's tick loop: each
//! committed-but-unemitted proof-plane item is handed to a [`ReceiptSink`],
//! and only acked in the ledger once the sink accepts it (at-least-once). The
//! debt breaker watches that drain: if the oldest unemitted receipt stays
//! outstanding past [`LedgerConfig::receipt_debt_age_secs`] or the outbox
//! depth passes [`LedgerConfig::receipt_debt_max`], it bumps the
//! [`CreditGate`](super::CreditGate)'s generation counter with reason
//! `ReceiptDebt`, flipping admission for receipt-requiring spends to
//! fail-closed until the outbox drains — the same shape as `AuditedAvc`
//! ("a decision that cannot be durably audited is a Deny"). Already-committed
//! spends stand; the debt only gates *new* spends.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use hyprstream_ledger::{Did, OutboxItem, TransferId};
use tokio::sync::Mutex;

use super::LedgerConfig;

/// What an emitted receipt/checkpoint evidences — the projection the emitter
/// reconstructs from an [`OutboxItem`] plus the cell identity, handed to a
/// [`ReceiptSink`]. Building the full `ai.hyprstream.ledger.receipt` PDS
/// record (lexicon from item 1.5) inside a sink impl is the production
/// emission path; the [`LoggingReceiptSink`] here is the inert default.
#[derive(Debug, Clone)]
pub struct ReceiptPayload {
    /// The cell ledger's service identity (the receipt host).
    pub ledger_id: Did,
    /// The outbox item being emitted.
    pub item: OutboxItem,
}

impl ReceiptPayload {
    /// The transfer this receipt evidences, if any (checkpoints carry `None`).
    pub fn transfer_id(&self) -> Option<TransferId> {
        self.item.transfer_id
    }
}

/// Where receipts/checkpoints are emitted (the proof plane). The production
/// impl writes the `ai.hyprstream.ledger.{receipt,checkpoint}` PDS records
/// (item 1.5's collection); [`LoggingReceiptSink`] is the inert default that
/// never fails — useful for single-cell tests and for deployments that opt
/// into the enforcer without PDS emission wired yet.
#[async_trait]
pub trait ReceiptSink: Send + Sync {
    /// Emit one proof-plane item. Returning `Err` leaves it in the outbox for
    /// at-least-once re-drain; the [`DebtBreaker`] turns sustained failure
    /// into fail-closed admission.
    async fn emit(&self, payload: &ReceiptPayload) -> anyhow::Result<()>;
}

/// Inert [`ReceiptSink`] that logs each emission via `tracing` and always
/// succeeds — the outbox drains to zero, so the debt breaker never trips.
/// This is the default when no PDS sink is wired.
#[derive(Debug, Default, Clone)]
pub struct LoggingReceiptSink;

#[async_trait]
impl ReceiptSink for LoggingReceiptSink {
    async fn emit(&self, payload: &ReceiptPayload) -> anyhow::Result<()> {
        tracing::info!(
            ledger = %payload.ledger_id.as_str(),
            seq = payload.item.journal_seq,
            kind = ?payload.item.kind,
            transfer = ?payload.item.transfer_id,
            "ledger receipt emitted (logging sink)"
        );
        Ok(())
    }
}

/// The state transition a [`DebtBreaker::observe`] call produced, so the
/// emitter loop can log / alarm on edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DebtTransition {
    /// Nothing changed (still healthy, or still in debt).
    Unchanged,
    /// The debt threshold was crossed this call — the generation counter was
    /// bumped, flipping admission to fail-closed for new receipt-requiring
    /// spends.
    Tripped { generation: u64 },
    /// The outbox drained back under threshold — admission re-opened.
    Cleared,
}

/// Watches the receipt-outbox drain and trips a fail-closed breaker when
/// receipts cannot be durably evidenced (plan §2e / Appendix A.5).
///
/// The breaker shares its `generation` counter with the
/// [`CreditGate`](super::CreditGate): a bump invalidates every cached grant
/// keyed to an older generation, so the next admit re-materializes against a
/// fresh epoch and — under debt — is denied (the gate observes the bumped
/// generation and the enforcer treats a cold gate as deny, plan §5.4).
pub struct DebtBreaker {
    age: std::time::Duration,
    depth_max: usize,
    /// Shared with the CreditGate. Bumped on trip.
    generation: Arc<AtomicU64>,
    in_debt: AtomicBool,
    oldest_unacked: Mutex<Option<Instant>>,
}

impl std::fmt::Debug for DebtBreaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DebtBreaker")
            .field("age", &self.age)
            .field("depth_max", &self.depth_max)
            .field("in_debt", &self.in_debt.load(Ordering::Relaxed))
            .finish()
    }
}

impl DebtBreaker {
    /// Construct from config, sharing the gate's generation counter.
    pub fn new(config: &LedgerConfig, generation: Arc<AtomicU64>) -> Self {
        DebtBreaker {
            age: std::time::Duration::from_secs(config.receipt_debt_age_secs),
            depth_max: config.receipt_debt_max,
            generation,
            in_debt: AtomicBool::new(false),
            oldest_unacked: Mutex::new(None),
        }
    }

    /// Whether admission is currently fail-closed under receipt debt.
    pub fn in_debt(&self) -> bool {
        self.in_debt.load(Ordering::Acquire)
    }

    /// Observe one drain cycle. `outbox_depth` is the depth remaining *after*
    /// this drain attempt; `drain_succeeded` is whether every item the sink
    /// was asked to emit was accepted (and thus acked). `now` is the
    /// observation instant.
    ///
    /// Returns the [`DebtTransition`] so the emitter can log edges. Idempotent
    /// on steady state.
    pub async fn observe(
        &self,
        outbox_depth: usize,
        drain_succeeded: bool,
        now: Instant,
    ) -> DebtTransition {
        let healthy = drain_succeeded && outbox_depth == 0;
        let mut oldest = self.oldest_unacked.lock().await;

        if healthy {
            let was = self.in_debt.swap(false, Ordering::AcqRel);
            if oldest.take().is_some() || was {
                return DebtTransition::Cleared;
            }
            return DebtTransition::Unchanged;
        }

        // Not healthy: there is unacked debt. Record when the oldest item first
        // became unacked if we had none tracked.
        if oldest.is_none() {
            *oldest = Some(now);
        }
        let age = now.duration_since(oldest.unwrap_or(now));
        let over_depth = outbox_depth > self.depth_max;
        let over_age = age >= self.age;

        if (over_depth || over_age) && !self.in_debt.load(Ordering::Acquire) {
            let gen = self.generation.fetch_add(1, Ordering::AcqRel) + 1;
            self.in_debt.store(true, Ordering::Release);
            DebtTransition::Tripped { generation: gen }
        } else {
            DebtTransition::Unchanged
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use hyprstream_ledger::{OutboxKind, OutboxSeq};

    fn breaker(age_secs: u64, depth_max: usize) -> (DebtBreaker, Arc<AtomicU64>) {
        let cfg = LedgerConfig {
            receipt_debt_age_secs: age_secs,
            receipt_debt_max: depth_max,
            ..LedgerConfig::default()
        };
        let gen = Arc::new(AtomicU64::new(1));
        (DebtBreaker::new(&cfg, Arc::clone(&gen)), gen)
    }

    fn payload() -> ReceiptPayload {
        ReceiptPayload {
            ledger_id: Did("did:web:cell.test".to_owned()),
            item: OutboxItem {
                seq: OutboxSeq(1),
                kind: OutboxKind::Receipt,
                transfer_id: Some(TransferId(7)),
                journal_seq: 3,
            },
        }
    }

    #[tokio::test]
    async fn healthy_drain_keeps_gate_open_and_clears() {
        let (b, gen) = breaker(60, 10);
        // A failed drain with items starts tracking age but does not trip yet.
        assert_eq!(
            b.observe(5, false, Instant::now()).await,
            DebtTransition::Unchanged
        );
        assert_eq!(gen.load(Ordering::Relaxed), 1);
        // Drain succeeds and outbox empties → Cleared.
        assert_eq!(
            b.observe(0, true, Instant::now()).await,
            DebtTransition::Cleared
        );
        assert!(!b.in_debt());
    }

    #[tokio::test]
    async fn depth_threshold_trips_generation_and_marks_debt() {
        let (b, gen) = breaker(3600, 10);
        let t = Instant::now();
        // Depth just over the max trips immediately (age irrelevant).
        assert_eq!(
            b.observe(11, false, t).await,
            DebtTransition::Tripped { generation: 2 }
        );
        assert!(b.in_debt());
        assert_eq!(
            gen.load(Ordering::Relaxed),
            2,
            "generation must bump on trip"
        );
        // A second over-threshold observation does not double-bump.
        assert_eq!(b.observe(12, false, t).await, DebtTransition::Unchanged);
        assert_eq!(gen.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn age_threshold_trips_after_window_elapses() {
        let (b, gen) = breaker(60, 10_000);
        let start = Instant::now();
        // Under depth max, young debt does not trip.
        assert_eq!(b.observe(5, false, start).await, DebtTransition::Unchanged);
        // Simulate the window elapsing by observing again with a `now` advanced
        // past the age threshold (oldest-unacked is still `start`).
        let later = start + std::time::Duration::from_secs(61);
        assert_eq!(
            b.observe(5, false, later).await,
            DebtTransition::Tripped { generation: 2 }
        );
        assert_eq!(gen.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn logging_sink_always_emits() {
        let sink = LoggingReceiptSink;
        assert!(sink.emit(&payload()).await.is_ok());
    }
}
