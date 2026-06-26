//! Reconnect backoff for lazy transports (#156).
//!
//! [`ReconnectBackoff`] tracks consecutive failed dial attempts and computes an
//! exponential-backoff cooldown window between them.  The lazy transports hold one
//! `ReconnectBackoff` alongside their cached session so failed dials don't
//! tight-loop on a downed peer — matching the explicit `tokio::time::sleep`
//! backoff that `zmq_connection.rs` used to perform.
//!
//! ## Design
//!
//! - [`LazyState`] bundles the optional cached transport + backoff so they stay
//!   consistent under a single lock: "no cached session" always implies "may be in
//!   backoff."
//! - `invalidate()` calls `record_failure()` so a *send-path* transport error
//!   also counts against the backoff budget (not only dial failures).
//! - `INITIAL_BACKOFF` is 1 s; subsequent failures double up to `MAX_BACKOFF`
//!   (30 s), matching the ZMQ reconnect interval cap observed in practice.
//!
//! ## What #156 defers
//!
//! DID-doc re-resolve on re-dial (so the manager picks up a moved `service`
//! entry) requires async resolution which can't run inside the sync lock; that
//! layer is left for a future incremental PR and documented with a TODO below.

use std::time::{Duration, Instant};

const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const BACKOFF_MULTIPLIER: f64 = 2.0;
const MAX_BACKOFF: Duration = Duration::from_secs(30);

/// Exponential-backoff cooldown tracker for RPC reconnect attempts.
#[derive(Debug, Default)]
pub struct ReconnectBackoff {
    /// Number of consecutive failed dials (saturates at u32::MAX).
    failures: u32,
    /// Reconnect is blocked until this instant; `None` means "go ahead."
    cooldown_until: Option<Instant>,
}

impl ReconnectBackoff {
    /// How long the caller must wait before a reconnect attempt is allowed.
    /// Returns `None` if a dial may proceed immediately.
    pub fn cooldown_remaining(&self) -> Option<Duration> {
        let until = self.cooldown_until?;
        let now = Instant::now();
        if now >= until {
            None
        } else {
            Some(until - now)
        }
    }

    /// Record a failed dial (or send-path fatal) and schedule the next window.
    pub fn record_failure(&mut self) {
        self.failures = self.failures.saturating_add(1);
        let secs = INITIAL_BACKOFF.as_secs_f64()
            * BACKOFF_MULTIPLIER.powi(self.failures.saturating_sub(1) as i32);
        let delay = Duration::from_secs_f64(secs).min(MAX_BACKOFF);
        self.cooldown_until = Some(Instant::now() + delay);
    }

    /// Reset backoff after a successful connection.
    pub fn record_success(&mut self) {
        self.failures = 0;
        self.cooldown_until = None;
    }
}

/// Combined lazy-connect state: the optional cached transport + reconnect backoff.
///
/// Keeping both behind a single lock ensures "session = None" is always
/// consistent with the backoff state — a race cannot observe a stale cooldown
/// after a successful reconnect resets it.
pub struct LazyState<T> {
    /// `None` until the first successful dial; reset to `None` on transport error.
    pub cached: Option<T>,
    pub backoff: ReconnectBackoff,
}

impl<T> Default for LazyState<T> {
    fn default() -> Self {
        Self {
            cached: None,
            backoff: ReconnectBackoff::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_failure_sets_cooldown() {
        let mut b = ReconnectBackoff::default();
        assert!(b.cooldown_remaining().is_none(), "no cooldown before first failure");
        b.record_failure();
        assert!(b.cooldown_remaining().is_some(), "cooldown after first failure");
    }

    #[test]
    fn success_resets_backoff() {
        let mut b = ReconnectBackoff::default();
        b.record_failure();
        b.record_success();
        assert_eq!(b.failures, 0);
        assert!(b.cooldown_remaining().is_none());
    }

    #[test]
    fn backoff_caps_at_max() {
        let mut b = ReconnectBackoff::default();
        for _ in 0..64 {
            b.record_failure();
        }
        // Can't inspect the raw cooldown_until, but record_failure must not panic
        // on saturation and the remaining duration must be ≤ MAX_BACKOFF + epsilon.
        if let Some(remaining) = b.cooldown_remaining() {
            assert!(remaining <= MAX_BACKOFF + Duration::from_millis(50));
        }
    }
}
