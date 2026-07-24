//! OpenTelemetry token-burn metering for inference.
//!
//! Token burn — prompt tokens consumed plus generated tokens produced per
//! request — is accounting- and ops-relevant, so it is emitted as OpenTelemetry
//! **metrics** rather than process logs (#1253, #1261). The authoritative spend
//! ledger lives in #1264; the instruments here are the metering/ops signal.
//!
//! # Instruments
//!
//! - `inference_tokens_total` — `Counter<u64>` (unit "tokens"), cumulative tokens
//!   consumed, with attributes `kind` = `prompt` | `generated`, `model`, and an
//!   **opaque hashed** tenant id.
//! - `inference_request_tokens` — `Histogram<u64>` (unit "tokens"), total tokens
//!   (prompt + generated) per request, same `model` / `tenant` attributes.
//!
//! # Security
//!
//! The raw request subject is **never** used as a metric attribute: metrics are
//! aggregatable across tenants and must not carry tenant-identifying content. The
//! tenant is reduced to an opaque 64-bit hash via [`opaque_tenant_id`] before it
//! becomes an attribute. Prompt text, token text, and generated text are likewise
//! never in metrics — only integer counts. See the no-leak canary test in
//! [`crate::runtime::torch_engine`] which guards the same property for logs.

#[cfg(feature = "otel")]
mod metered {
    use opentelemetry::{KeyValue, metrics::Meter};

    /// OTel meter scope name for inference token-burn instruments.
    const METER_NAME: &str = "hyprstream.inference";
    const METRIC_TOKENS_TOTAL: &str = "inference_tokens_total";
    const METRIC_REQUEST_TOKENS: &str = "inference_request_tokens";
    const UNIT: &str = "tokens";

    const KIND_PROMPT: &str = "prompt";
    const KIND_GENERATED: &str = "generated";
    const KIND_TOTAL: &str = "total";

    /// Build the (kind, model, opaque-tenant) attribute set for a metric point.
    fn attrs(kind: &str, model: &str, tenant: Option<u64>) -> [KeyValue; 3] {
        let tenant_attr = match tenant {
            // Opaque hashed id; never the raw subject.
            Some(h) => KeyValue::new("tenant", format!("tenant:{h:016x}")),
            None => KeyValue::new("tenant", "anonymous"),
        };
        [
            KeyValue::new("kind", kind.to_owned()),
            KeyValue::new("model", model.to_owned()),
            tenant_attr,
        ]
    }

    /// Token-burn instruments built from an OTel [`Meter`].
    ///
    /// When constructed via [`TokenBurnMeter::global`] the process-global OTel
    /// meter is used, which is a no-op meter unless an SDK meter provider has been
    /// registered (e.g. by `init_telemetry`). Recording is therefore always safe:
    /// it never blocks or panics, and silently does nothing when no exporter is
    /// configured or the `otel` feature is off.
    pub struct TokenBurnMeter {
        tokens_total: opentelemetry::metrics::Counter<u64>,
        request_tokens: opentelemetry::metrics::Histogram<u64>,
    }

    impl TokenBurnMeter {
        /// Build from the process-global OTel meter (no-op when no provider set).
        pub fn global() -> Self {
            Self::from_meter(opentelemetry::global::meter(METER_NAME))
        }

        /// Build from an explicit meter — used in tests with a local SDK meter
        /// provider so recording can be asserted hermetically.
        pub fn from_meter(meter: Meter) -> Self {
            Self {
                tokens_total: meter
                    .u64_counter(METRIC_TOKENS_TOTAL)
                    .with_unit(UNIT)
                    .with_description(
                        "Inference tokens consumed (prompt + generated), by kind/model/tenant",
                    )
                    .build(),
                request_tokens: meter
                    .u64_histogram(METRIC_REQUEST_TOKENS)
                    .with_unit(UNIT)
                    .with_description("Total tokens per inference request (prompt + generated)")
                    .build(),
            }
        }

        /// Record prompt tokens consumed by tokenizing a request.
        pub fn record_prompt(&self, model: &str, tenant: Option<u64>, count: u64) {
            if count == 0 {
                return;
            }
            self.tokens_total
                .add(count, &attrs(KIND_PROMPT, model, tenant));
        }

        /// Record generated tokens produced by a completed request.
        pub fn record_generated(&self, model: &str, tenant: Option<u64>, count: u64) {
            if count == 0 {
                return;
            }
            self.tokens_total
                .add(count, &attrs(KIND_GENERATED, model, tenant));
        }

        /// Record the per-request token total (prompt + generated) on the histogram.
        pub fn record_request_total(&self, model: &str, tenant: Option<u64>, total: u64) {
            // Record even when total == 0 so an empty-generation request is still
            // observable as a sample; the counter carries the non-zero breakdown.
            self.request_tokens
                .record(total, &attrs(KIND_TOTAL, model, tenant));
        }
    }
}

#[cfg(feature = "otel")]
pub use metered::TokenBurnMeter;

/// No-op instruments when the `otel` feature is disabled.
///
/// Keeps the recording call sites compiling without an OTel SDK so non-`otel`
/// builds still meter nothing without conditionalizing every call site.
#[cfg(not(feature = "otel"))]
pub struct TokenBurnMeter;

#[cfg(not(feature = "otel"))]
impl TokenBurnMeter {
    pub fn global() -> Self {
        Self
    }
    pub fn record_prompt(&self, _model: &str, _tenant: Option<u64>, _count: u64) {}
    pub fn record_generated(&self, _model: &str, _tenant: Option<u64>, _count: u64) {}
    pub fn record_request_total(&self, _model: &str, _tenant: Option<u64>, _total: u64) {}
}

/// Reduce a raw tenant subject to an opaque, stable 64-bit id for metric attrs.
///
/// The tenant subject is never carried verbatim in metrics (it is aggregatable
/// across tenants and must not leak identity). A deterministic hash is emitted so
/// the same tenant maps to the same opaque id across process restarts, enabling
/// continuity in dashboards without ever revealing the subject. Returns `None`
/// when no tenant is in scope.
///
/// This is pure std and available regardless of the `otel` feature so it can be
/// unit-tested hermetically.
pub fn opaque_tenant_id(tenant: &Option<String>) -> Option<u64> {
    use std::hash::{Hash, Hasher};
    let subject = tenant.as_ref()?;
    // DefaultHasher is deterministic (fixed SipHash keys), not the per-process
    // randomized RandomState — chosen deliberately for cross-restart stability.
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    subject.hash(&mut hasher);
    Some(hasher.finish())
}

#[cfg(all(test, feature = "otel"))]
mod tests {
    use super::*;
    use opentelemetry::metrics::MeterProvider;
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};

    /// Drive the meter through a real SDK pipeline (in-memory exporter) and assert
    /// token burn lands on the right instruments with opaque, never-raw tenant attrs.
    #[test]
    fn token_burn_meter_records_prompt_and_generated() {
        let exporter = InMemoryMetricExporter::default();
        let provider = SdkMeterProvider::builder()
            .with_reader(PeriodicReader::builder(exporter.clone()).build())
            .build();
        let meter = provider.meter("hyprstream.inference");
        let m = TokenBurnMeter::from_meter(meter);

        let tenant_hash = opaque_tenant_id(&Some("did:web:acme.example".to_owned()));
        assert!(tenant_hash.is_some(), "tenant hashes to an opaque id");
        m.record_prompt("test-model", tenant_hash, 10);
        m.record_generated("test-model", tenant_hash, 7);
        m.record_request_total("test-model", tenant_hash, 17);

        provider.force_flush().unwrap();
        let collected = exporter.get_finished_metrics().unwrap();

        // Flatten (name, sum-or-count, serialized-attrs) across scopes/instruments.
        let mut counter_sum: u64 = 0;
        let mut histogram_count: u64 = 0;
        let mut histogram_sum: u64 = 0;
        let mut any_attrs = String::new();
        for rm in collected {
            for sm in rm.scope_metrics() {
                for metric in sm.metrics() {
                    let name = metric.name();
                    match metric.data() {
                        AggregatedMetrics::U64(MetricData::Sum(s)) => {
                            for dp in s.data_points() {
                                if name == "inference_tokens_total" {
                                    counter_sum += dp.value();
                                }
                                for kv in dp.attributes() {
                                    any_attrs.push_str(&format!("{kv:?}"));
                                }
                            }
                        }
                        AggregatedMetrics::U64(MetricData::Histogram(h)) => {
                            for dp in h.data_points() {
                                if name == "inference_request_tokens" {
                                    histogram_count += dp.count();
                                    histogram_sum += dp.sum();
                                }
                                for kv in dp.attributes() {
                                    any_attrs.push_str(&format!("{kv:?}"));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Counter: 10 (prompt) + 7 (generated) = 17 cumulative tokens.
        assert_eq!(counter_sum, 17, "inference_tokens_total should sum to 17");
        // Histogram: one request totaling 17 tokens.
        assert_eq!(histogram_count, 1, "one request recorded on the histogram");
        assert_eq!(histogram_sum, 17, "histogram total should be 17");

        // The raw tenant subject must never appear in metric attributes; only the
        // opaque hash form does.
        assert!(
            !any_attrs.contains("acme.example"),
            "raw tenant subject leaked into metric attrs: {any_attrs}"
        );
        assert!(
            any_attrs.contains("tenant:"),
            "opaque tenant attr missing: {any_attrs}"
        );
    }

    #[test]
    fn opaque_tenant_id_is_stable_and_anonymous_when_absent() {
        let a = opaque_tenant_id(&Some("did:web:acme.example".to_owned()));
        let b = opaque_tenant_id(&Some("did:web:acme.example".to_owned()));
        assert_eq!(a, b, "same subject must hash to the same opaque id");

        let other = opaque_tenant_id(&Some("did:web:other.example".to_owned()));
        assert_ne!(a, other, "different subjects must hash differently");

        assert_eq!(
            opaque_tenant_id(&None),
            None,
            "absent tenant yields no opaque id"
        );
    }

    #[test]
    fn zero_prompt_is_a_noop() {
        // A zero count should not add a data point to the counter.
        let exporter = InMemoryMetricExporter::default();
        let provider = SdkMeterProvider::builder()
            .with_reader(PeriodicReader::builder(exporter.clone()).build())
            .build();
        let meter = provider.meter("hyprstream.inference");
        let m = TokenBurnMeter::from_meter(meter);
        m.record_prompt("test-model", None, 0);
        m.record_generated("test-model", None, 0);
        provider.force_flush().unwrap();

        let mut any_counter: u64 = 0;
        for rm in exporter.get_finished_metrics().unwrap() {
            for sm in rm.scope_metrics() {
                for metric in sm.metrics() {
                    if let AggregatedMetrics::U64(MetricData::Sum(s)) = metric.data() {
                        any_counter += s.data_points().map(|dp| dp.value()).sum::<u64>();
                    }
                }
            }
        }
        assert_eq!(any_counter, 0, "zero-count records must not land on the counter");
    }
}
