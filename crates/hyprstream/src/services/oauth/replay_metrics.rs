//! OpenTelemetry signals for fail-closed OAuth replay barriers.

use hyprstream_util::InsertIfAbsentNoEvictResult;

pub const DPOP: &str = "dpop";
pub const CLIENT_ASSERTION: &str = "client_assertion";
pub const ATPROTO_SERVICE_ASSERTION: &str = "atproto_service_assertion";

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
