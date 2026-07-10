//! The background tick + receipt-emitter loop (plan §2b.4 / §2e, item 1.6).
//!
//! Runs on the [`super::service::LedgerService`]`s runtime: every
//! [`LedgerConfig::tick_interval_secs`] it (1) drives the backend's `tick`
//! (expiry sweep + scheduled checkpoint), (2) drains the committed-but-unemitted
//! outbox into the [`super::sink::ReceiptSink`], acking only what the sink
//! accepted (at-least-once), and (3) feeds the [`super::sink::DebtBreaker`] so
//! sustained emission failure flips admission to fail-closed.

use std::sync::Arc;
use std::time::Duration;

use hyprstream_ledger::Did;
use tokio::sync::Notify;

use super::handle::LedgerHandle;
use super::sink::{DebtBreaker, DebtTransition, ReceiptPayload, ReceiptSink};

/// One drain batch size. Bounded so a single tick makes bounded progress and
/// the breaker observes depth promptly.
const DRAIN_BATCH: usize = 256;

/// Run the tick + emitter loop until `shutdown` is notified.
///
/// The loop never panics on ledger/sink errors: a failed drain leaves items in
/// the outbox (at-least-once) and the [`DebtBreaker`] turns sustained failure
/// into the fail-closed admission policy.
pub async fn run_emitter_loop(
    handle: LedgerHandle,
    sink: Arc<dyn ReceiptSink>,
    breaker: Arc<DebtBreaker>,
    ledger_id: Did,
    tick: Duration,
    shutdown: Arc<Notify>,
) {
    let mut interval = tokio::time::interval(tick);
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = shutdown.notified() => {
                tracing::info!("ledger emitter loop shutting down");
                break;
            }
            _ = interval.tick() => {}
        }

        // 1. Housekeeping tick (expiry sweep + scheduled checkpoint).
        if let Err(e) = handle.tick().await {
            tracing::warn!(error = %e, "ledger tick failed (will retry next interval)");
        }

        // 2. Drain the outbox into the sink, acking only accepted items.
        let peeked = match handle.outbox_peek(DRAIN_BATCH).await {
            Ok(items) => items,
            Err(e) => {
                tracing::warn!(error = %e, "outbox peek failed");
                // Observe a failed drain with the unknown depth; the breaker
                // tracks age from the first failure.
                observe(&breaker, 0, false).await;
                continue;
            }
        };
        let mut last_ack: Option<hyprstream_ledger::OutboxSeq> = None;
        let mut all_ok = true;
        for item in &peeked {
            let payload = ReceiptPayload {
                ledger_id: ledger_id.clone(),
                item: item.clone(),
            };
            match sink.emit(&payload).await {
                Ok(()) => {
                    last_ack = Some(item.seq);
                }
                Err(e) => {
                    all_ok = false;
                    tracing::warn!(
                        error = %e,
                        seq = ?item.seq,
                        "receipt emit failed (left in outbox for at-least-once re-drain)"
                    );
                    break;
                }
            }
        }
        if let Some(up_to) = last_ack {
            if let Err(e) = handle.outbox_ack(up_to).await {
                tracing::warn!(error = %e, "outbox ack failed");
            }
        }

        // 3. Re-peek to report remaining depth to the breaker (0 if we drained
        //    everything and the backend had no more). A separate peek keeps the
        //    breaker honest about lingering debt rather than assuming the batch
        //    was the whole outbox.
        let remaining = handle.outbox_peek(1).await.map(|v| v.len()).unwrap_or(1);
        observe(&breaker, remaining, all_ok && remaining == 0).await;
    }
}

async fn observe(breaker: &DebtBreaker, depth: usize, drain_succeeded: bool) {
    match breaker
        .observe(depth, drain_succeeded, std::time::Instant::now())
        .await
    {
        DebtTransition::Unchanged => {}
        DebtTransition::Tripped { generation } => {
            tracing::warn!(
                generation,
                "receipt debt tripped: admission fail-closed for new receipt-requiring spends"
            );
        }
        DebtTransition::Cleared => {
            tracing::info!("receipt debt cleared: admission re-opened");
        }
    }
}
