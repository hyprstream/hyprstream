//! Scheduling Substrate — the shared vocabulary + small primitives every
//! scheduling surface composes on (#523 P0 / #524 / #628).
//!
//! Five selection systems grew independently (`selection.rs` backend pick,
//! `SandboxPool::acquire`, `DevicePool`, `CellRouter`/HRW routing,
//! `queryCandidates`). They share one shape —
//! `candidates → filter(predicates) → rank → select → explain` — so this module
//! is the ONE place that shape lives. Each provider declares capability truth
//! once (as labels + resources via [`CapabilitySource`]); each consumer filters
//! and ranks through [`filter`] / [`rank`] / [`select`] and gets a trace
//! ([`SelectionReport`]) for free, because every [`Predicate`] returns *why* it
//! rejected.
//!
//! ## Guardrail (non-negotiable)
//! This is small on purpose: plain types + free functions. **No trait towers,
//! no `Scheduler<T>` abstraction-without-consumers.** If it grows past
//! "vocabulary + small helpers," it is wrong — extend by adding real consumers,
//! not by abstracting in a vacuum.

use std::cmp::Ordering;

// ─── Vocabulary (mirrors `discovery.capnp` #523 P0) ─────────────────────────

/// A named resource amount (`{name, quantity}`); `quantity` is a k8s-quantity
/// string e.g. `"8"`, `"512Gi"`, `"100m"`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resource {
    pub name: String,
    pub quantity: String,
}

impl Resource {
    pub fn new(name: impl Into<String>, quantity: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            quantity: quantity.into(),
        }
    }
}

/// A resource requirement: a candidate must have at least `min_quantity` of
/// `name` allocatable (declared − already-allocated).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceRequest {
    pub name: String,
    pub min_quantity: String,
}

impl ResourceRequest {
    pub fn new(name: impl Into<String>, min_quantity: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            min_quantity: min_quantity.into(),
        }
    }

    /// Satisfied iff `available` parses and is `>= min_quantity`. A missing/
    /// unparseable available value is *not* satisfiable (fail-closed).
    pub fn satisfied_by(&self, available: &str) -> bool {
        match (
            parse_k8s_quantity(available),
            parse_k8s_quantity(&self.min_quantity),
        ) {
            (Some(have), Some(need)) => have >= need,
            _ => false,
        }
    }
}

/// Label-selector match operator (k8s match-expressions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectorOp {
    In,
    NotIn,
    Exists,
    DoesNotExist,
    Gt,
    Lt,
}

/// One label-selector conjunct: `key <op> values`. A query carries a list of
/// these (all must match — AND). `values` is ignored for `Exists`/`DoesNotExist`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelSelector {
    pub key: String,
    pub op: SelectorOp,
    pub values: Vec<String>,
}

impl LabelSelector {
    pub fn new(key: impl Into<String>, op: SelectorOp, values: Vec<String>) -> Self {
        Self {
            key: key.into(),
            op,
            values,
        }
    }

    /// Evaluate this selector against a label set `(key, value)`.
    pub fn matches(&self, labels: &[(String, String)]) -> bool {
        let entry = labels.iter().find(|(k, _)| k == &self.key);
        match self.op {
            SelectorOp::Exists => entry.is_some(),
            SelectorOp::DoesNotExist => entry.is_none(),
            SelectorOp::In => entry.is_some_and(|(_, v)| self.values.iter().any(|w| w == v)),
            SelectorOp::NotIn => entry.is_none_or(|(_, v)| !self.values.iter().any(|w| w == v)),
            // Gt/Lt compare numerically over the k8s-quantity value (k8s restricts
            // these to integer resources; we extend to any parseable quantity).
            SelectorOp::Gt => {
                self.cmp_numeric(entry.map(|(_, v)| v.as_str()), |o| o == Ordering::Greater)
            }
            SelectorOp::Lt => {
                self.cmp_numeric(entry.map(|(_, v)| v.as_str()), |o| o == Ordering::Less)
            }
        }
    }

    fn cmp_numeric(&self, have: Option<&str>, want: impl Fn(Ordering) -> bool) -> bool {
        let have = match have.and_then(parse_k8s_quantity) {
            Some(h) => h,
            None => return false,
        };
        // Gt/Lt compare against `values[0]` (k8s: a single value for these ops).
        let need = match self.values.first().and_then(|v| parse_k8s_quantity(v)) {
            Some(n) => n,
            None => return false,
        };
        want(have.cmp(&need))
    }
}

/// A placement candidate returned by `queryCandidates`.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementCandidate {
    pub node: String,
    pub record_uri: String,
    pub load_fraction: f32,
    pub allocatable: Vec<Resource>,
    pub last_seen: i64,
}

/// Bounded result set. `total_matching` is the full match count before the
/// bound was applied.
#[derive(Debug, Clone)]
pub struct PlacementCandidateSet {
    pub candidates: Vec<PlacementCandidate>,
    pub total_matching: u32,
}

// ─── k8s-quantity parsing (minimal, for comparison) ────────────────────────

/// Parse a k8s-quantity string into a canonical comparable value (milli of the
/// base unit, as `i128`). Handles the common suffixes:
/// - binary: `Ki`/`Mi`/`Gi`/`Ti`/`Pi`/`Ei` (×1024^n)
/// - decimal: `k`/`M`/`G`/`T`/`P`/`E` (×1000^n)
/// - `m` (milli, ÷1000)
/// - plain decimal number
///
/// Returns `None` for unparseable input. This is deliberately **minimal** — it
/// covers the suffixes in real use here; full k8s quantity semantics (sign,
/// exponents, the binary/decimal distinction k8s preserves) are deferred to the
/// scheduler engine that actually does capacity math. Comparison only needs the
/// same input to canonicalize consistently.
pub fn parse_k8s_quantity(s: &str) -> Option<i128> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    // Split trailing non-digit suffix from the numeric prefix. k8s quantities are
    // `<number><suffix>`; a leading sign is allowed.
    let split = s
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == '+'))
        .unwrap_or(s.len());
    let (num, suffix) = s.split_at(split);
    let v: f64 = num.parse().ok()?;
    // Canonicalize to milli of the base unit (i128). One multiplier per suffix.
    // Sub-unit suffixes (`m`, `u`) divide instead of multiply.
    let milli: i128 = match suffix {
        "" => (v * 1_000.0) as i128,
        "m" => v as i128,                   // already milli
        "u" | "µ" => (v / 1_000.0) as i128, // micro → milli
        "k" => (v * 1_000.0 * 1_000.0) as i128,
        "M" => (v * 1_000_000.0 * 1_000.0) as i128,
        "G" => (v * 1_000_000_000.0 * 1_000.0) as i128,
        "T" => (v * 1_000_000_000_000.0 * 1_000.0) as i128,
        "Ki" => (v * 1_024.0 * 1_000.0) as i128,
        "Mi" => (v * 1_048_576.0 * 1_000.0) as i128,
        "Gi" => (v * 1_073_741_824.0 * 1_000.0) as i128,
        "Ti" => (v * 1_099_511_627_776.0 * 1_000.0) as i128,
        // P/E and Pi/Ei overflow i128 milli in real magnitudes; fall through to
        // the catch-all `None` (not in use here).
        _ => return None,
    };
    Some(milli)
}

// ─── filter → rank → select ────────────────────────────────────────────────

/// Why a predicate rejected a candidate (carried into the explain trace).
#[derive(Debug, Clone)]
pub struct RejectionReason(pub String);

/// A predicate over a candidate: returns `None` if it passes, `Some(reason)` if
/// rejected (and why). Because every predicate returns its reason, a
/// [`SelectionReport`] trace falls out of [`filter`] / [`select`] for free.
pub type Predicate<C> = Box<dyn Fn(&C) -> Option<RejectionReason>>;

/// Per-candidate outcome from [`filter`]: the candidate plus its first rejection
/// (if any). Short-circuits per candidate on the first rejecting predicate.
#[derive(Debug, Clone)]
pub struct Outcome<'c, C> {
    pub candidate: &'c C,
    pub rejected: Option<RejectionReason>,
}

impl<'c, C> Outcome<'c, C> {
    pub fn passed(&self) -> bool {
        self.rejected.is_none()
    }
}

/// Run every candidate through every predicate, recording the first rejection
/// (if any) per candidate. Predicates are evaluated in order; a candidate
/// rejected by predicate *i* is not tested against *i+1*.
#[allow(clippy::needless_lifetimes)] // two input refs; the `'c` ties the borrow unambiguously
pub fn filter<'c, C>(candidates: &'c [C], predicates: &[Predicate<C>]) -> Vec<Outcome<'c, C>> {
    candidates
        .iter()
        .map(|c| {
            let rejected = predicates.iter().find_map(|p| p(c));
            Outcome {
                candidate: c,
                rejected,
            }
        })
        .collect()
}

/// Order survivors by `by`. Stable. Consumes the slice of survivor references
/// (typically `filter(...).iter().filter(|o| o.passed()).map(|o| o.candidate)`).
pub fn rank<C>(mut survivors: Vec<&C>, by: impl Fn(&C, &C) -> Ordering) -> Vec<&C> {
    survivors.sort_by(|a, b| by(a, b));
    survivors
}

/// One candidate's verdict in a [`SelectionReport`] (generic — consumer-agnostic;
/// `selection.rs` adds its own backend-specific fields when it adopts this).
#[derive(Debug, Clone)]
pub struct CandidateOutcome<C> {
    pub candidate: C,
    pub selected: bool,
    pub rejection_reason: Option<String>,
}

/// Full trace of a selection decision: every candidate considered + the winner.
/// The explain shape (generalized from the backend-selection diagnostics).
#[derive(Debug, Clone)]
pub struct SelectionReport<C> {
    pub requested: String,
    pub candidates: Vec<CandidateOutcome<C>>,
    pub selected: Option<usize>, // index into `candidates`
    pub summary: String,
}

/// The full pipeline: filter → rank → bounded select, returning both the chosen
/// candidates and the explain trace. `identify` maps a `&C` to the name used in
/// the report's per-candidate entries (so the report stays generic over `C`).
///
/// `selected` in the report marks every returned (post-bound) candidate.
pub fn select<C>(
    candidates: &[C],
    predicates: &[Predicate<C>],
    rank_by: impl Fn(&C, &C) -> Ordering,
    max: usize,
) -> SelectionReport<C>
where
    C: Clone,
{
    let outcomes = filter(candidates, predicates);
    let mut survivors: Vec<&C> = outcomes
        .iter()
        .filter(|o| o.passed())
        .map(|o| o.candidate)
        .collect();
    survivors = rank(survivors, rank_by);
    let bound: Vec<&C> = survivors.into_iter().take(max).collect();
    let chosen: Vec<&C> = bound.clone();
    let summary = format!(
        "{} considered, {} survived filters, {} selected (bound {max})",
        candidates.len(),
        outcomes.iter().filter(|o| o.passed()).count(),
        chosen.len(),
    );
    let candidate_reports = candidates
        .iter()
        .map(|c| {
            let is_selected = chosen.iter().any(|s| std::ptr::eq(*s, c));
            let rejection = outcomes
                .iter()
                .find(|o| std::ptr::eq(o.candidate, c))
                .and_then(|o| o.rejected.as_ref().map(|r| r.0.clone()));
            CandidateOutcome {
                candidate: c.clone(),
                selected: is_selected,
                rejection_reason: rejection,
            }
        })
        .collect();
    let selected_idx = bound.first().and_then(|b| {
        candidates
            .iter()
            .position(|c| std::ptr::eq(c as *const _, *b as *const _))
    });
    SelectionReport {
        requested: String::new(),
        candidates: candidate_reports,
        selected: selected_idx,
        summary,
    }
}

// ─── Capability-source contract ────────────────────────────────────────────

/// A provider of scheduling capability truth: emits its stable capabilities +
/// declared capacity in the shared vocabulary so a `PlacementCandidate` / node
/// record can aggregate them (one source of truth — no hand-declaration, no
/// drift between e.g. a node record claiming `runtime=kata` and the live
/// `selection.rs` knowing `cloud-hypervisor` is missing).
///
/// Implementors: backend-selection registries, device pools, future NUMA/
/// topology providers. Aggregators (the node-record publisher) call
/// [`CapabilitySource::capability_labels`] / [`CapabilitySource::capability_resources`]
/// and fold the results into the published record.
pub trait CapabilitySource {
    /// Stable capability labels — e.g. `("runtime", "kata")`, `("gpu.arch", "hopper")`,
    /// `("topology.zone", "us-east-1a")`.
    fn capability_labels(&self) -> Vec<(String, String)>;
    /// Declared countable capacity — e.g. `Resource("nvidia.com/gpu", "8")`,
    /// `Resource("memory", "512Gi")`. (Live/allocatable is volatile; the node
    /// record combines declared capacity with a liveness report.)
    fn capability_resources(&self) -> Vec<Resource>;
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn sel(key: &str, op: SelectorOp, values: &[&str]) -> LabelSelector {
        LabelSelector::new(key, op, values.iter().map(|&s| s.to_owned()).collect())
    }

    #[test]
    fn selector_in_not_in() {
        let labels = vec![("env".into(), "prod".into()), ("gpu".into(), "h100".into())];
        assert!(sel("env", SelectorOp::In, &["prod", "staging"]).matches(&labels));
        assert!(!sel("env", SelectorOp::In, &["dev"]).matches(&labels));
        assert!(sel("env", SelectorOp::NotIn, &["dev"]).matches(&labels));
        assert!(!sel("env", SelectorOp::NotIn, &["prod"]).matches(&labels));
    }

    #[test]
    fn selector_exists_does_not_exist() {
        let labels = vec![("env".into(), "prod".into())];
        assert!(sel("env", SelectorOp::Exists, &[]).matches(&labels));
        assert!(!sel("missing", SelectorOp::Exists, &[]).matches(&labels));
        assert!(sel("missing", SelectorOp::DoesNotExist, &[]).matches(&labels));
        assert!(!sel("env", SelectorOp::DoesNotExist, &[]).matches(&labels));
    }

    #[test]
    fn selector_gt_lt_numeric() {
        let labels = vec![("gpu.count".into(), "8".into())];
        assert!(sel("gpu.count", SelectorOp::Gt, &["4"]).matches(&labels));
        assert!(!sel("gpu.count", SelectorOp::Gt, &["8"]).matches(&labels));
        assert!(sel("gpu.count", SelectorOp::Lt, &["16"]).matches(&labels));
        assert!(!sel("gpu.count", SelectorOp::Lt, &["8"]).matches(&labels));
        // missing key → not greater/less (fail-closed)
        assert!(!sel("absent", SelectorOp::Gt, &["1"]).matches(&labels));
    }

    #[test]
    fn k8s_quantity_compares_across_suffixes() {
        assert!(parse_k8s_quantity("8").unwrap() == parse_k8s_quantity("8000m").unwrap());
        assert!(parse_k8s_quantity("1Gi").unwrap() > parse_k8s_quantity("512Mi").unwrap());
        assert!(parse_k8s_quantity("2").unwrap() > parse_k8s_quantity("1").unwrap());
        assert!(parse_k8s_quantity("100m").unwrap() < parse_k8s_quantity("1").unwrap());
        assert!(parse_k8s_quantity("garbage").is_none());
        assert!(parse_k8s_quantity("").is_none());
    }

    #[test]
    fn resource_request_satisfied() {
        let req = ResourceRequest::new("memory", "256Gi");
        assert!(req.satisfied_by("512Gi"));
        assert!(req.satisfied_by("256Gi"));
        assert!(!req.satisfied_by("128Gi"));
        assert!(!req.satisfied_by("garbage")); // fail-closed
    }

    #[test]
    fn filter_rank_select_pipeline() {
        // candidates: (name, priority). predicate: keep priority >= 2.
        let cands = vec![
            ("a".to_owned(), 1),
            ("b".to_owned(), 3),
            ("c".to_owned(), 2),
            ("d".to_owned(), 2),
        ];
        let preds: Vec<Predicate<(String, i32)>> = vec![Box::new(|c| {
            if c.1 >= 2 {
                None
            } else {
                Some(RejectionReason("priority < 2".into()))
            }
        })];
        let report = select(
            &cands,
            &preds,
            |a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)), // priority desc, name asc
            2,                                        // bound
        );
        // survivors after filter: b(3), c(2), d(2) → ranked b, c, d → bound 2 → [b, c]
        let selected: Vec<&String> = report
            .candidates
            .iter()
            .filter(|o| o.selected)
            .map(|o| &o.candidate.0)
            .collect();
        assert_eq!(selected, vec![&"b".to_owned(), &"c".to_owned()]);
        // rejected candidate carries its reason
        let a = report
            .candidates
            .iter()
            .find(|o| o.candidate.0 == "a")
            .unwrap();
        assert_eq!(a.rejection_reason.as_deref(), Some("priority < 2"));
        assert!(report.summary.contains("4 considered") && report.summary.contains("3 survived"));
    }

    #[test]
    fn filter_records_first_rejection_only() {
        let cands = vec![1, 2, 3];
        let preds: Vec<Predicate<i32>> = vec![
            Box::new(|c| {
                if c % 2 == 0 {
                    Some(RejectionReason("even".into()))
                } else {
                    None
                }
            }),
            Box::new(|c| {
                if *c == 1 {
                    Some(RejectionReason("is-one".into()))
                } else {
                    None
                }
            }),
        ];
        let out = filter(&cands, &preds);
        // 1 rejected by FIRST predicate? no — 1 is odd, passes pred 0, rejected by pred 1 ("is-one")
        assert_eq!(out[0].rejected.as_ref().unwrap().0, "is-one");
        // 2 rejected by first predicate ("even"), pred 1 not evaluated
        assert_eq!(out[1].rejected.as_ref().unwrap().0, "even");
        // 3 passes both
        assert!(out[2].passed());
    }

    #[test]
    fn capability_source_trait() {
        struct Gpu;
        impl CapabilitySource for Gpu {
            fn capability_labels(&self) -> Vec<(String, String)> {
                vec![("gpu.arch".into(), "hopper".into())]
            }
            fn capability_resources(&self) -> Vec<Resource> {
                vec![Resource::new("nvidia.com/gpu", "8")]
            }
        }
        let g = Gpu;
        assert_eq!(
            g.capability_labels(),
            vec![("gpu.arch".into(), "hopper".into())]
        );
        assert_eq!(g.capability_resources().len(), 1);
    }
}
