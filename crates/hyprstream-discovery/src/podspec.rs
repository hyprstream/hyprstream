//! K4c (#787) — the placement→PodSpec vocabulary map.
//!
//! Two-level scheduling, the Argo/Ray meta-scheduler pattern: **hyprstream's
//! scheduler decides *what runs with what constraints*; Kubernetes decides
//! *which node*.** This module is the pure translation seam between the two. It
//! lowers a placement decision expressed in the [`crate::scheduling`] substrate
//! vocabulary (`LabelSelector` / `ResourceRequest`, both already k8s-shaped from
//! #523 P0) into the `PodSpec` scheduling fields Kubernetes consumes:
//! `nodeSelector` / `affinity` / `tolerations` / `resources` /
//! `priorityClassName`.
//!
//! ## Pure translation — no cluster, no client
//!
//! There is no `kube::Client` here and no I/O: it maps values to values. The
//! Kubernetes backend (K4a, #781) calls [`stamp_pod_spec`] to stamp a Pod it is
//! about to emit; everything is unit-testable offline.
//!
//! ## The narrowing invariant (never widen)
//!
//! The native scheduler's decision is *narrowed, never widened*, by the map.
//! Concretely, for the **fit predicates** (label selectors + resource requests):
//! any node the native [`LabelSelector::matches`] would reject is also rejected
//! by the emitted `nodeSelector`/`nodeAffinity`, and every emitted resource
//! request carries the native `min_quantity` unchanged (never lowered). This is
//! asserted by property tests. `SelectorOp`'s operator set is *identical* to the
//! Kubernetes `NodeSelectorRequirement` operator set (`In/NotIn/Exists/
//! DoesNotExist/Gt/Lt`), so each selector maps 1:1 with no semantic drift.
//!
//! `tolerations` and `priorityClassName` are *additive scheduling policy*, not
//! fit predicates — they are not derived from a rejecting predicate and so are
//! not part of the narrowing invariant. Tolerations exist precisely to *permit*
//! scheduling onto tainted nodes; they are carried through verbatim from the
//! placement decision.
//!
//! ## Stubbed input
//!
//! The #525 P2 scheduler engine (PR #761) that will produce a single typed
//! placement-decision struct is not landed yet. Until it is, [`Placement`] is
//! assembled by the caller from the substrate vocabulary that *is* on `main`
//! today ([`LabelSelector`] / [`ResourceRequest`]), plus a minimal local
//! [`TaintToleration`] input for the taints→tolerations rule (the substrate has
//! no taint vocabulary yet). When #525's decision type lands, [`Placement`]
//! becomes a thin `From` over it.

use std::collections::BTreeMap;

use crate::scheduling::{LabelSelector, ResourceRequest, SelectorOp};

// The exact `k8s-openapi` version the workspace is pinned to (`v1_32`), reached
// through `hyprstream-k8s` so exactly one `v1_*` is linked workspace-wide.
use hyprstream_k8s::k8s_openapi::api::core::v1::{
    Affinity, NodeAffinity, NodeSelector, NodeSelectorRequirement, NodeSelectorTerm, PodSpec,
    ResourceRequirements, Toleration,
};
use hyprstream_k8s::k8s_openapi::apimachinery::pkg::api::resource::Quantity;

// ─── Input vocabulary ──────────────────────────────────────────────────────

/// How a [`TaintToleration`] matches a node taint (mirrors the k8s toleration
/// operator).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TolerationOp {
    /// `key`'s taint value must equal [`TaintToleration::value`].
    Equal,
    /// `key` need only exist as a taint (value ignored).
    Exists,
}

/// A taint effect a toleration applies to (mirrors the k8s taint effect). `None`
/// on a [`TaintToleration`] tolerates *all* effects for the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaintEffect {
    NoSchedule,
    PreferNoSchedule,
    NoExecute,
}

impl TaintEffect {
    fn as_str(self) -> &'static str {
        match self {
            TaintEffect::NoSchedule => "NoSchedule",
            TaintEffect::PreferNoSchedule => "PreferNoSchedule",
            TaintEffect::NoExecute => "NoExecute",
        }
    }
}

/// A taint the placement tolerates. **Stubbed input** (see module docs): the
/// scheduling substrate has no taint vocabulary yet, so this local type carries
/// the taints-policy → `tolerations` rule until the #525 engine lands one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaintToleration {
    pub key: String,
    pub op: TolerationOp,
    /// Required only for [`TolerationOp::Equal`]; ignored for `Exists`.
    pub value: Option<String>,
    /// The effect tolerated; `None` tolerates every effect for `key`.
    pub effect: Option<TaintEffect>,
}

/// A placement decision expressed in the substrate vocabulary, ready to lower
/// onto a `PodSpec`. Assembled by the caller from a `queryCandidates`/scheduler
/// result (see module docs — this is the seam that becomes a `From<Decision>`
/// once #525 lands).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Placement {
    /// Node label constraints (the query's match-expressions). All must hold
    /// (AND). Lowered to `nodeSelector` + `nodeAffinity`.
    pub selectors: Vec<LabelSelector>,
    /// Resource requirements. Lowered to container `resources.requests` (and,
    /// for extended resources, `resources.limits`).
    pub resources: Vec<ResourceRequest>,
    /// Taints the placement tolerates. Lowered to `tolerations`.
    pub tolerations: Vec<TaintToleration>,
    /// Placement priority → `priorityClassName`. Optional.
    pub priority_class: Option<String>,
}

/// The lowered scheduling constraints, ready to stamp onto (or read off) a
/// `PodSpec`. Every field is `Option`/empty-aware so an empty placement stamps
/// nothing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PodScheduling {
    /// `PodSpec.nodeSelector` — equality constraints (single-value `In`).
    pub node_selector: Option<BTreeMap<String, String>>,
    /// `PodSpec.affinity` — the remaining match-expressions as `nodeAffinity`.
    pub affinity: Option<Affinity>,
    /// `PodSpec.tolerations`.
    pub tolerations: Option<Vec<Toleration>>,
    /// Per-container `resources` (requests + extended-resource limits).
    pub resources: ResourceRequirements,
    /// `PodSpec.priorityClassName`.
    pub priority_class_name: Option<String>,
}

// ─── Mapping ───────────────────────────────────────────────────────────────

/// A resource name is an *extended resource* (vendor-qualified, e.g.
/// `nvidia.com/gpu`) when it is namespaced with `/`. Kubernetes requires
/// extended resources to set `limits == requests`; standard resources
/// (`cpu`/`memory`/…) take a request only.
fn is_extended_resource(name: &str) -> bool {
    name.contains('/')
}

/// Lower [`ResourceRequest`]s to a `ResourceRequirements`. Each request's
/// `min_quantity` is copied **verbatim** into `requests[name]` (never lowered —
/// this is the resource half of the narrowing invariant). Extended resources
/// (GPUs, `nvidia.com/gpu`, …) additionally land in `limits` with the same
/// quantity, as Kubernetes mandates.
pub fn to_resource_requirements(requests: &[ResourceRequest]) -> ResourceRequirements {
    let mut req_map: BTreeMap<String, Quantity> = BTreeMap::new();
    let mut lim_map: BTreeMap<String, Quantity> = BTreeMap::new();
    for r in requests {
        let q = Quantity(r.min_quantity.clone());
        if is_extended_resource(&r.name) {
            lim_map.insert(r.name.clone(), q.clone());
        }
        req_map.insert(r.name.clone(), q);
    }
    ResourceRequirements {
        requests: (!req_map.is_empty()).then_some(req_map),
        limits: (!lim_map.is_empty()).then_some(lim_map),
        ..Default::default()
    }
}

fn selector_op_str(op: SelectorOp) -> &'static str {
    match op {
        SelectorOp::In => "In",
        SelectorOp::NotIn => "NotIn",
        SelectorOp::Exists => "Exists",
        SelectorOp::DoesNotExist => "DoesNotExist",
        SelectorOp::Gt => "Gt",
        SelectorOp::Lt => "Lt",
    }
}

/// Lower [`LabelSelector`]s to `(nodeSelector, affinity)`.
///
/// A pure single-value `In` is an equality constraint and lowers to a terse
/// `nodeSelector` entry. Every other selector — multi-value `In`, `NotIn`,
/// `Exists`, `DoesNotExist`, `Gt`, `Lt` — lowers to a `nodeAffinity`
/// match-expression, whose operator vocabulary is identical to `SelectorOp`.
/// `nodeSelector` and `nodeAffinity` are ANDed by Kubernetes, so splitting the
/// selectors across them still requires a node to satisfy *all* of them
/// (narrowing preserved). All affinity expressions go in a single
/// `NodeSelectorTerm` so they AND (separate terms would OR).
pub fn to_node_constraints(
    selectors: &[LabelSelector],
) -> (Option<BTreeMap<String, String>>, Option<Affinity>) {
    let mut node_selector: BTreeMap<String, String> = BTreeMap::new();
    let mut match_expressions: Vec<NodeSelectorRequirement> = Vec::new();

    for s in selectors {
        if s.op == SelectorOp::In && s.values.len() == 1 {
            node_selector.insert(s.key.clone(), s.values[0].clone());
            continue;
        }
        let values = match s.op {
            SelectorOp::Exists | SelectorOp::DoesNotExist => None,
            _ => Some(s.values.clone()),
        };
        match_expressions.push(NodeSelectorRequirement {
            key: s.key.clone(),
            operator: selector_op_str(s.op).to_owned(),
            values,
        });
    }

    let affinity = (!match_expressions.is_empty()).then(|| Affinity {
        node_affinity: Some(NodeAffinity {
            required_during_scheduling_ignored_during_execution: Some(NodeSelector {
                node_selector_terms: vec![NodeSelectorTerm {
                    match_expressions: Some(match_expressions),
                    ..Default::default()
                }],
            }),
            ..Default::default()
        }),
        ..Default::default()
    });

    (
        (!node_selector.is_empty()).then_some(node_selector),
        affinity,
    )
}

/// Lower [`TaintToleration`]s to `PodSpec.tolerations`. Additive policy, not a
/// fit predicate (see module docs).
pub fn to_tolerations(tolerations: &[TaintToleration]) -> Option<Vec<Toleration>> {
    if tolerations.is_empty() {
        return None;
    }
    Some(
        tolerations
            .iter()
            .map(|t| Toleration {
                key: Some(t.key.clone()),
                operator: Some(
                    match t.op {
                        TolerationOp::Equal => "Equal",
                        TolerationOp::Exists => "Exists",
                    }
                    .to_owned(),
                ),
                // A k8s toleration with operator `Exists` must not carry a value.
                value: match t.op {
                    TolerationOp::Equal => t.value.clone(),
                    TolerationOp::Exists => None,
                },
                effect: t.effect.map(|e| e.as_str().to_owned()),
                ..Default::default()
            })
            .collect(),
    )
}

/// Lower a whole [`Placement`] to [`PodScheduling`].
pub fn to_pod_scheduling(placement: &Placement) -> PodScheduling {
    let (node_selector, affinity) = to_node_constraints(&placement.selectors);
    PodScheduling {
        node_selector,
        affinity,
        tolerations: to_tolerations(&placement.tolerations),
        resources: to_resource_requirements(&placement.resources),
        priority_class_name: placement.priority_class.clone(),
    }
}

/// Stamp a placement decision onto a `PodSpec` a backend is about to emit.
///
/// Scheduling fields present in the placement *replace* the corresponding
/// `PodSpec` fields (the placement is the authoritative decision). An empty
/// placement leaves the pod untouched. The lowered `resources` are applied to
/// **every** container in the pod (a training/inference pod is single-container
/// in practice); a pod with no containers gets its scheduling fields set but no
/// resources (there is nowhere to put them).
pub fn stamp_pod_spec(placement: &Placement, pod: &mut PodSpec) {
    let sched = to_pod_scheduling(placement);

    if sched.node_selector.is_some() {
        pod.node_selector = sched.node_selector;
    }
    if sched.affinity.is_some() {
        pod.affinity = sched.affinity;
    }
    if sched.tolerations.is_some() {
        pod.tolerations = sched.tolerations;
    }
    if sched.priority_class_name.is_some() {
        pod.priority_class_name = sched.priority_class_name;
    }
    if !placement.resources.is_empty() {
        for c in pod.containers.iter_mut() {
            c.resources = Some(sched.resources.clone());
        }
    }
}

/// Human-readable lines describing the emitted Kubernetes constraints, for the
/// round-trip explainability deliverable: a [`crate::scheduling::SelectionReport`]
/// summary can embed these so a placement is explainable end-to-end (native
/// predicate → emitted k8s constraint).
pub fn describe_constraints(sched: &PodScheduling) -> Vec<String> {
    let mut lines = Vec::new();
    if let Some(ns) = &sched.node_selector {
        for (k, v) in ns {
            lines.push(format!("nodeSelector: {k}={v}"));
        }
    }
    if let Some(aff) = &sched.affinity {
        if let Some(na) = &aff.node_affinity {
            if let Some(sel) = &na.required_during_scheduling_ignored_during_execution {
                for term in &sel.node_selector_terms {
                    if let Some(exprs) = &term.match_expressions {
                        for e in exprs {
                            let vals = e.values.as_ref().map(|v| v.join(",")).unwrap_or_default();
                            lines
                                .push(format!("nodeAffinity: {} {} [{}]", e.key, e.operator, vals));
                        }
                    }
                }
            }
        }
    }
    if let Some(reqs) = &sched.resources.requests {
        for (k, v) in reqs {
            lines.push(format!("resources.requests: {k}={}", v.0));
        }
    }
    if let Some(lims) = &sched.resources.limits {
        for (k, v) in lims {
            lines.push(format!("resources.limits: {k}={}", v.0));
        }
    }
    if let Some(tols) = &sched.tolerations {
        for t in tols {
            lines.push(format!(
                "toleration: {} {} {}",
                t.key.as_deref().unwrap_or("*"),
                t.operator.as_deref().unwrap_or("Exists"),
                t.effect.as_deref().unwrap_or("*"),
            ));
        }
    }
    if let Some(pc) = &sched.priority_class_name {
        lines.push(format!("priorityClassName: {pc}"));
    }
    lines
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    fn sel(key: &str, op: SelectorOp, values: &[&str]) -> LabelSelector {
        LabelSelector::new(key, op, values.iter().map(|&s| s.to_owned()).collect())
    }

    #[test]
    fn single_value_in_becomes_node_selector() {
        let (ns, aff) = to_node_constraints(&[sel("zone", SelectorOp::In, &["us-east-1a"])]);
        let ns = ns.unwrap();
        assert_eq!(ns.get("zone").map(String::as_str), Some("us-east-1a"));
        assert!(aff.is_none(), "a single-value In should not emit affinity");
    }

    #[test]
    fn multi_value_and_ops_become_affinity_expressions() {
        let (ns, aff) = to_node_constraints(&[
            sel("gpu", SelectorOp::In, &["h100", "a100"]),
            sel("spot", SelectorOp::DoesNotExist, &[]),
            sel("mem", SelectorOp::Gt, &["64"]),
        ]);
        assert!(ns.is_none());
        let exprs = aff
            .unwrap()
            .node_affinity
            .unwrap()
            .required_during_scheduling_ignored_during_execution
            .unwrap()
            .node_selector_terms
            .into_iter()
            .next()
            .unwrap()
            .match_expressions
            .unwrap();
        assert_eq!(exprs.len(), 3);
        let gpu = exprs.iter().find(|e| e.key == "gpu").unwrap();
        assert_eq!(gpu.operator, "In");
        assert_eq!(gpu.values.as_deref().unwrap(), ["h100", "a100"]);
        let spot = exprs.iter().find(|e| e.key == "spot").unwrap();
        assert_eq!(spot.operator, "DoesNotExist");
        assert!(spot.values.is_none(), "Exists/DoesNotExist carry no values");
    }

    #[test]
    fn gpu_lands_as_extended_resource_with_matching_limit() {
        let rr = to_resource_requirements(&[
            ResourceRequest::new("nvidia.com/gpu", "2"),
            ResourceRequest::new("cpu", "4"),
            ResourceRequest::new("memory", "16Gi"),
        ]);
        let requests = rr.requests.unwrap();
        let limits = rr.limits.unwrap();
        // GPU request preserved verbatim, and mirrored into limits.
        assert_eq!(requests["nvidia.com/gpu"].0, "2");
        assert_eq!(limits["nvidia.com/gpu"].0, "2");
        // Standard resources: request only, no limit.
        assert_eq!(requests["cpu"].0, "4");
        assert_eq!(requests["memory"].0, "16Gi");
        assert!(!limits.contains_key("cpu"));
        assert!(!limits.contains_key("memory"));
    }

    #[test]
    fn toleration_exists_drops_value() {
        let tols = to_tolerations(&[TaintToleration {
            key: "nvidia.com/gpu".to_owned(),
            op: TolerationOp::Exists,
            value: Some("ignored".to_owned()),
            effect: Some(TaintEffect::NoSchedule),
        }])
        .unwrap();
        assert_eq!(tols.len(), 1);
        assert_eq!(tols[0].operator.as_deref(), Some("Exists"));
        assert!(
            tols[0].value.is_none(),
            "Exists toleration must not carry a value"
        );
        assert_eq!(tols[0].effect.as_deref(), Some("NoSchedule"));
    }

    #[test]
    fn empty_placement_stamps_nothing() {
        let mut pod = PodSpec::default();
        stamp_pod_spec(&Placement::default(), &mut pod);
        assert!(pod.node_selector.is_none());
        assert!(pod.affinity.is_none());
        assert!(pod.tolerations.is_none());
        assert!(pod.priority_class_name.is_none());
    }

    #[test]
    fn describe_covers_every_field() {
        let sched = to_pod_scheduling(&Placement {
            selectors: vec![
                sel("zone", SelectorOp::In, &["z1"]),
                sel("gpu", SelectorOp::Exists, &[]),
            ],
            resources: vec![ResourceRequest::new("nvidia.com/gpu", "1")],
            tolerations: vec![TaintToleration {
                key: "gpu".to_owned(),
                op: TolerationOp::Exists,
                value: None,
                effect: Some(TaintEffect::NoSchedule),
            }],
            priority_class: Some("high".to_owned()),
        });
        let lines = describe_constraints(&sched);
        assert!(lines.iter().any(|l| l.contains("nodeSelector: zone=z1")));
        assert!(lines
            .iter()
            .any(|l| l.starts_with("nodeAffinity: gpu Exists")));
        assert!(lines
            .iter()
            .any(|l| l.contains("resources.requests: nvidia.com/gpu=1")));
        assert!(lines
            .iter()
            .any(|l| l.contains("resources.limits: nvidia.com/gpu=1")));
        assert!(lines
            .iter()
            .any(|l| l.starts_with("toleration: gpu Exists NoSchedule")));
        assert!(lines.iter().any(|l| l.contains("priorityClassName: high")));
    }
}
