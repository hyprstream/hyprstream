//! OpenTelemetry signals for fail-closed OAuth replay barriers.

use hyprstream_util::InsertIfAbsentNoEvictResult;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::time::{Duration, Instant};

pub const DPOP: &str = "dpop";
pub const CLIENT_ASSERTION: &str = "client_assertion";
pub const ATPROTO_SERVICE_ASSERTION: &str = "atproto_service_assertion";

const FULL_WARNING_INTERVAL: Duration = Duration::from_secs(60);
static LAST_FULL_WARNING: Lazy<Mutex<HashMap<&'static str, Instant>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(all(feature = "otel", not(target_arch = "wasm32")))]
mod metered {
    use once_cell::sync::Lazy;
    use opentelemetry::KeyValue;

    use super::*;

    const METER_NAME: &str = "hyprstream.oauth";
    const METRIC_REJECTIONS: &str = "oauth_replay_barrier_rejections_total";

    struct ReplayBarrierMeter {
        rejections: opentelemetry::metrics::Counter<u64>,
    }

    impl ReplayBarrierMeter {
        fn global() -> Self {
            let meter = opentelemetry::global::meter(METER_NAME);
            Self {
                rejections: meter
                    .u64_counter(METRIC_REJECTIONS)
                    .with_description(
                        "OAuth replay barrier rejections, by barrier and rejection reason",
                    )
                    .build(),
            }
        }

        fn record(&self, barrier: &'static str, result: InsertIfAbsentNoEvictResult) {
            let reason = match result {
                InsertIfAbsentNoEvictResult::Duplicate => "duplicate",
                InsertIfAbsentNoEvictResult::Full => "full",
                InsertIfAbsentNoEvictResult::Inserted => return,
            };
            self.rejections.add(
                1,
                &[
                    KeyValue::new("barrier", barrier),
                    KeyValue::new("reason", reason),
                ],
            );
        }
    }

    static METER: Lazy<ReplayBarrierMeter> = Lazy::new(ReplayBarrierMeter::global);

    pub(super) fn record(barrier: &'static str, result: InsertIfAbsentNoEvictResult) {
        METER.record(barrier, result);
    }
}

#[cfg(not(all(feature = "otel", not(target_arch = "wasm32"))))]
mod metered {
    use super::*;

    pub(super) fn record(_barrier: &'static str, _result: InsertIfAbsentNoEvictResult) {}
}

/// Record every failed replay-barrier insertion. The `reason` attribute makes
/// a blocked replay distinguishable from legitimate traffic refused because
/// the fail-closed barrier is saturated.
pub fn record_rejection(barrier: &'static str, result: InsertIfAbsentNoEvictResult) {
    metered::record(barrier, result);
}

/// Return `true` at most once per barrier per minute for a full-barrier
/// rejection. Metrics still record every rejection; this only prevents an
/// attacker from turning saturation into unbounded warning-log volume.
pub fn should_warn_full(barrier: &'static str, result: InsertIfAbsentNoEvictResult) -> bool {
    if result != InsertIfAbsentNoEvictResult::Full {
        return false;
    }
    let now = Instant::now();
    let mut last_warning = LAST_FULL_WARNING.lock();
    match last_warning.get_mut(barrier) {
        Some(last) if now.duration_since(*last) < FULL_WARNING_INTERVAL => false,
        Some(last) => {
            *last = now;
            true
        }
        None => {
            last_warning.insert(barrier, now);
            true
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_warnings_are_rate_limited_per_barrier() {
        let barrier = "replay_metrics_rate_limit_test";
        assert!(should_warn_full(barrier, InsertIfAbsentNoEvictResult::Full));
        assert!(!should_warn_full(barrier, InsertIfAbsentNoEvictResult::Full));
        assert!(!should_warn_full(
            barrier,
            InsertIfAbsentNoEvictResult::Duplicate
        ));
    }
}
