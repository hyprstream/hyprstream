//! OpenTelemetry token-burn metering for inference.
//!
//! Token burn — prompt tokens consumed plus generated tokens produced per
//! request — is accounting- and ops-relevant, so it is emitted as OpenTelemetry
//! **metrics** rather than process logs (#1253, #1261). The authoritative spend
//! ledger lives in #1264; the instruments here are the metering/ops signal.
//!
//! # Instruments
//!
//! - `inference_tokens_total` — `Counter<u64>` (unit `{token}`, UCUM custom-unit
//!   form), cumulative tokens consumed, with attributes `kind` =
//!   `prompt` | `generated`, `model`, and an **opaque keyed-hash** tenant id.
//! - `inference_request_tokens` — `Histogram<u64>` (unit `{token}`), total tokens
//!   (prompt + generated) per request, with `model` / `tenant` attributes only.
//!
//! # Security
//!
//! The raw request subject is **never** used as a metric attribute: metrics are
//! aggregatable across tenants and must not carry tenant-identifying content. The
//! tenant is reduced to an opaque keyed hash via [`opaque_tenant_id`] before it
//! becomes an attribute. Prompt text, token text, generated text, and token IDs
//! are likewise never in metrics — only integer counts. See the no-leak canary
//! test in [`crate::runtime::torch_engine`] which guards the same property for
//! logs.
//!
//! # Platform gating
//!
//! The opentelemetry dependency family is not wasm32-compatible, so the crates
//! are target-gated in `Cargo.toml` and the instruments here compile only for
//! `all(feature = "otel", not(target_arch = "wasm32"))`. Everywhere else a no-op
//! [`TokenBurnMeter`] keeps the call sites compiling — the wasm/browser client
//! pays no metric cost and non-`otel` builds meter nothing.

#[cfg(all(feature = "otel", not(target_arch = "wasm32")))]
mod metered {
    use once_cell::sync::Lazy;
    use opentelemetry::{KeyValue, metrics::Meter};

    /// OTel meter scope name for inference token-burn instruments.
    const METER_NAME: &str = "hyprstream.inference";
    const METRIC_TOKENS_TOTAL: &str = "inference_tokens_total";
    const METRIC_REQUEST_TOKENS: &str = "inference_request_tokens";
    /// UCUM custom-unit form for token counts (OTel convention: `{unit}`).
    const UNIT: &str = "{token}";

    const KIND_PROMPT: &str = "prompt";
    const KIND_GENERATED: &str = "generated";

    /// The tenant attribute: opaque keyed-hash id, never the raw subject.
    fn tenant_attr(tenant: Option<u64>) -> KeyValue {
        match tenant {
            Some(h) => KeyValue::new("tenant", format!("tenant:{h:016x}")),
            None => KeyValue::new("tenant", "anonymous"),
        }
    }

    /// (kind, model, opaque-tenant) attribute set for the counter.
    fn counter_attrs(kind: &str, model: &str, tenant: Option<u64>) -> [KeyValue; 3] {
        [
            KeyValue::new("kind", kind.to_owned()),
            KeyValue::new("model", model.to_owned()),
            tenant_attr(tenant),
        ]
    }

    /// (model, opaque-tenant) attribute set for the per-request histogram.
    fn request_attrs(model: &str, tenant: Option<u64>) -> [KeyValue; 2] {
        [KeyValue::new("model", model.to_owned()), tenant_attr(tenant)]
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

        /// Build from an explicit meter — the test seam, used with a local SDK
        /// meter provider so recording can be asserted hermetically.
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
                .add(count, &counter_attrs(KIND_PROMPT, model, tenant));
        }

        /// Record generated tokens produced by a completed request.
        pub fn record_generated(&self, model: &str, tenant: Option<u64>, count: u64) {
            if count == 0 {
                return;
            }
            self.tokens_total
                .add(count, &counter_attrs(KIND_GENERATED, model, tenant));
        }

        /// Record the per-request token total (prompt + generated) on the histogram.
        pub fn record_request_total(&self, model: &str, tenant: Option<u64>, total: u64) {
            // Record even when total == 0 so an empty-generation request is still
            // observable as a sample; the counter carries the non-zero breakdown.
            self.request_tokens.record(total, &request_attrs(model, tenant));
        }
    }

    /// Process-global instrument bundle, built **once** on first recording — not
    /// per request (#1261 review: instrument construction must not be hot-path
    /// work). Built lazily so a provider installed at startup (`init_telemetry`)
    /// is already in place; if recording ever raced ahead of provider install,
    /// the global meter defers and connects the instruments once the provider is
    /// set, so no points are lost.
    static GLOBAL_METER: Lazy<TokenBurnMeter> = Lazy::new(TokenBurnMeter::global);

    /// Shared process-global token-burn instruments.
    pub fn global_meter() -> &'static TokenBurnMeter {
        &GLOBAL_METER
    }
}

#[cfg(all(feature = "otel", not(target_arch = "wasm32")))]
pub use metered::{TokenBurnMeter, global_meter};

/// No-op instruments when the `otel` feature is disabled or the target is wasm32
/// (the opentelemetry crates are not wasm-compatible and are target-gated out).
///
/// Keeps the recording call sites compiling without an OTel SDK so non-`otel`
/// and wasm builds meter nothing without conditionalizing every call site.
#[cfg(not(all(feature = "otel", not(target_arch = "wasm32"))))]
pub struct TokenBurnMeter;

#[cfg(not(all(feature = "otel", not(target_arch = "wasm32"))))]
impl TokenBurnMeter {
    pub fn global() -> Self {
        Self
    }
    pub fn record_prompt(&self, _model: &str, _tenant: Option<u64>, _count: u64) {}
    pub fn record_generated(&self, _model: &str, _tenant: Option<u64>, _count: u64) {}
    pub fn record_request_total(&self, _model: &str, _tenant: Option<u64>, _total: u64) {}
}

/// Shared no-op instruments for non-`otel` / wasm32 builds.
#[cfg(not(all(feature = "otel", not(target_arch = "wasm32"))))]
pub fn global_meter() -> &'static TokenBurnMeter {
    static NOOP_METER: TokenBurnMeter = TokenBurnMeter;
    &NOOP_METER
}

/// Reduce a raw tenant subject to an opaque 64-bit id for metric attrs and
/// tenant-safe logs.
///
/// The tenant subject is never carried verbatim in metrics or logs (metrics are
/// aggregatable across tenants and must not leak identity). The hash is a
/// SipHash keyed with **process-random** keys (`RandomState`, drawn from the OS
/// RNG): an unkeyed hash such as `DefaultHasher::new()` is offline-enumerable —
/// anyone can precompute the ids of guessed low-entropy subjects (#1261 review).
/// With a random key the id is stable within the process but cannot be
/// precomputed or correlated across deployments. Returns `None` when no tenant
/// is in scope.
///
/// This is pure std and available regardless of the `otel` feature or target so
/// it can be unit-tested hermetically and reused by tenant-safe logging.
pub fn opaque_tenant_id(tenant: &Option<String>) -> Option<u64> {
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;
    use std::sync::OnceLock;

    static TENANT_HASH_KEY: OnceLock<RandomState> = OnceLock::new();

    let subject = tenant.as_ref()?;
    Some(TENANT_HASH_KEY.get_or_init(RandomState::new).hash_one(subject))
}

#[cfg(all(test, feature = "otel", not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use opentelemetry::metrics::MeterProvider;
    use opentelemetry_sdk::metrics::data::{AggregatedMetrics, MetricData};
    use opentelemetry_sdk::metrics::{InMemoryMetricExporter, PeriodicReader, SdkMeterProvider};

    /// Render a data point's attributes as a sorted `k=v` list for exact
    /// set comparisons.
    fn attr_set<'a, I: Iterator<Item = &'a opentelemetry::KeyValue>>(attrs: I) -> Vec<String> {
        let mut v: Vec<String> = attrs.map(|kv| format!("{}={}", kv.key, kv.value)).collect();
        v.sort();
        v
    }

    /// Drive the meter through a real SDK pipeline (in-memory exporter) and assert
    /// token burn lands on the right instruments with the exact attribute contract
    /// and opaque, never-raw tenant attrs.
    #[test]
    fn token_burn_meter_records_prompt_and_generated() -> Result<(), Box<dyn std::error::Error>> {
        let exporter = InMemoryMetricExporter::default();
        let provider = SdkMeterProvider::builder()
            .with_reader(PeriodicReader::builder(exporter.clone()).build())
            .build();
        let meter = provider.meter("hyprstream.inference");
        let m = TokenBurnMeter::from_meter(meter);

        let tenant_hash = opaque_tenant_id(&Some("did:web:acme.example".to_owned()));
        assert!(tenant_hash.is_some(), "tenant hashes to an opaque id");
        let expected_tenant = format!("tenant=tenant:{:016x}", tenant_hash.unwrap_or(0));
        m.record_prompt("test-model", tenant_hash, 10);
        m.record_generated("test-model", tenant_hash, 7);
        m.record_request_total("test-model", tenant_hash, 17);

        provider.force_flush()?;
        let collected = exporter.get_finished_metrics()?;

        let mut counter_points: Vec<(Vec<String>, u64)> = Vec::new();
        let mut counter_unit = String::new();
        let mut histogram_points: Vec<(Vec<String>, u64, u64)> = Vec::new();
        let mut histogram_unit = String::new();
        let mut any_attrs = String::new();
        for rm in collected {
            for sm in rm.scope_metrics() {
                for metric in sm.metrics() {
                    match metric.data() {
                        AggregatedMetrics::U64(MetricData::Sum(s)) => {
                            assert_eq!(metric.name(), "inference_tokens_total");
                            counter_unit = metric.unit().to_owned();
                            for dp in s.data_points() {
                                let attrs = attr_set(dp.attributes());
                                any_attrs.push_str(&attrs.join(","));
                                counter_points.push((attrs, dp.value()));
                            }
                        }
                        AggregatedMetrics::U64(MetricData::Histogram(h)) => {
                            assert_eq!(metric.name(), "inference_request_tokens");
                            histogram_unit = metric.unit().to_owned();
                            for dp in h.data_points() {
                                let attrs = attr_set(dp.attributes());
                                any_attrs.push_str(&attrs.join(","));
                                histogram_points.push((attrs, dp.count(), dp.sum()));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Units use the UCUM custom-unit form.
        assert_eq!(counter_unit, "{token}", "counter unit");
        assert_eq!(histogram_unit, "{token}", "histogram unit");

        // Counter: exactly two points — kind=prompt (10) and kind=generated (7) —
        // each with the exact (kind, model, tenant) attribute set. The SDK does
        // not guarantee data-point order, so sort before comparing.
        counter_points.sort();
        let expected_prompt_attrs = vec![
            "kind=prompt".to_owned(),
            "model=test-model".to_owned(),
            expected_tenant.clone(),
        ];
        let expected_generated_attrs = vec![
            "kind=generated".to_owned(),
            "model=test-model".to_owned(),
            expected_tenant.clone(),
        ];
        assert_eq!(
            counter_points,
            vec![
                (expected_generated_attrs, 7),
                (expected_prompt_attrs, 10),
            ],
            "counter points must be exactly prompt=10 and generated=7 with exact attrs"
        );

        // Histogram: exactly one point (count 1, sum 17) with ONLY the
        // (model, tenant) attribute set — no kind.
        let expected_histogram_attrs = vec!["model=test-model".to_owned(), expected_tenant];
        assert_eq!(
            histogram_points,
            vec![(expected_histogram_attrs, 1, 17)],
            "histogram must carry one request of 17 tokens with (model, tenant) attrs only"
        );

        // The raw tenant subject must never appear in metric attributes.
        assert!(
            !any_attrs.contains("acme.example"),
            "raw tenant subject leaked into metric attrs: {any_attrs}"
        );
        Ok(())
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
    fn zero_prompt_is_a_noop() -> Result<(), Box<dyn std::error::Error>> {
        // A zero count should not add a data point to the counter.
        let exporter = InMemoryMetricExporter::default();
        let provider = SdkMeterProvider::builder()
            .with_reader(PeriodicReader::builder(exporter.clone()).build())
            .build();
        let meter = provider.meter("hyprstream.inference");
        let m = TokenBurnMeter::from_meter(meter);
        m.record_prompt("test-model", None, 0);
        m.record_generated("test-model", None, 0);
        provider.force_flush()?;

        let mut any_counter: u64 = 0;
        for rm in exporter.get_finished_metrics()? {
            for sm in rm.scope_metrics() {
                for metric in sm.metrics() {
                    if let AggregatedMetrics::U64(MetricData::Sum(s)) = metric.data() {
                        for dp in s.data_points() {
                            any_counter += dp.value();
                        }
                    }
                }
            }
        }
        assert_eq!(any_counter, 0, "zero-count records must not land on the counter");
        Ok(())
    }
}
