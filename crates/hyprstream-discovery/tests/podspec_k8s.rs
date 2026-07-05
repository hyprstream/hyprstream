//! K4c (#787) — property + golden tests for the placement→PodSpec map.
//!
//! The whole file is gated on the `k8s` feature (empty when it is off), so
//! `cargo test -p hyprstream-discovery` compiles either way; run with
//! `--features k8s` to exercise it.
//!
//! Two kinds of coverage, per the issue's acceptance criteria:
//!  * **Property (the narrowing invariant)** — an independent re-implementation
//!    of Kubernetes node admission over the *emitted* `nodeSelector`/`affinity`
//!    is checked, over many random inputs, to agree with the native
//!    `LabelSelector::matches` filter (⇒ no widening); resource requests are
//!    shown to be copied verbatim and GPUs to land as extended resources with a
//!    matching limit.
//!  * **Golden** — a representative training Pod is stamped and serialized to
//!    YAML and asserted byte-for-byte against a committed fixture.
#![cfg(feature = "k8s")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use hyprstream_discovery::podspec::{
    describe_constraints, stamp_pod_spec, to_node_constraints, to_pod_scheduling,
    to_resource_requirements, Placement, TaintEffect, TaintToleration, TolerationOp,
};
use hyprstream_discovery::scheduling::{LabelSelector, ResourceRequest, SelectorOp};

use hyprstream_k8s::k8s_openapi::api::core::v1::{
    Affinity, Container, NodeSelectorRequirement, PodSpec,
};

use proptest::prelude::*;

// ─── Independent Kubernetes node-admission oracle ──────────────────────────
//
// A from-scratch evaluation of the emitted constraints against a node's labels,
// implementing k8s node-affinity/nodeSelector semantics *without* calling any
// of the production mapping code's evaluation, so agreement is meaningful.

fn parse_int(s: &str) -> Option<i128> {
    s.trim().parse::<i128>().ok()
}

fn expr_matches(e: &NodeSelectorRequirement, labels: &[(String, String)]) -> bool {
    let val = labels
        .iter()
        .find(|(k, _)| k == &e.key)
        .map(|(_, v)| v.as_str());
    let vals = e.values.clone().unwrap_or_default();
    match e.operator.as_str() {
        "In" => val.is_some_and(|v| vals.iter().any(|w| w == v)),
        "NotIn" => val.is_none_or(|v| !vals.iter().any(|w| w == v)),
        "Exists" => val.is_some(),
        "DoesNotExist" => val.is_none(),
        "Gt" => match (
            val.and_then(parse_int),
            vals.first().and_then(|s| parse_int(s)),
        ) {
            (Some(h), Some(n)) => h > n,
            _ => false,
        },
        "Lt" => match (
            val.and_then(parse_int),
            vals.first().and_then(|s| parse_int(s)),
        ) {
            (Some(h), Some(n)) => h < n,
            _ => false,
        },
        _ => false,
    }
}

fn k8s_admits(
    node_selector: &Option<BTreeMap<String, String>>,
    affinity: &Option<Affinity>,
    labels: &[(String, String)],
) -> bool {
    if let Some(ns) = node_selector {
        for (k, v) in ns {
            match labels.iter().find(|(lk, _)| lk == k) {
                Some((_, lv)) if lv == v => {}
                _ => return false,
            }
        }
    }
    if let Some(aff) = affinity {
        let sel = aff
            .node_affinity
            .as_ref()
            .unwrap()
            .required_during_scheduling_ignored_during_execution
            .as_ref()
            .unwrap();
        // Terms OR; each term's expressions AND.
        let any = sel.node_selector_terms.iter().any(|term| {
            term.match_expressions
                .as_ref()
                .is_none_or(|exprs| exprs.iter().all(|e| expr_matches(e, labels)))
        });
        if !any {
            return false;
        }
    }
    true
}

// Native side: a node passes iff every selector matches.
fn native_admits(selectors: &[LabelSelector], labels: &[(String, String)]) -> bool {
    selectors.iter().all(|s| s.matches(labels))
}

// ─── Generators ────────────────────────────────────────────────────────────

const KEYS: &[&str] = &["a", "b", "c", "zone", "gpu"];
// Mix of numeric (for Gt/Lt) and non-numeric values.
const VALUES: &[&str] = &["1", "2", "10", "x", "y"];

fn op_strategy() -> impl Strategy<Value = SelectorOp> {
    prop_oneof![
        Just(SelectorOp::In),
        Just(SelectorOp::NotIn),
        Just(SelectorOp::Exists),
        Just(SelectorOp::DoesNotExist),
        Just(SelectorOp::Gt),
        Just(SelectorOp::Lt),
    ]
}

fn selector_strategy() -> impl Strategy<Value = LabelSelector> {
    (
        prop::sample::select(KEYS),
        op_strategy(),
        prop::collection::vec(prop::sample::select(VALUES), 0..3),
    )
        .prop_map(|(key, op, values)| {
            LabelSelector::new(key, op, values.into_iter().map(str::to_owned).collect())
        })
}

fn labels_strategy() -> impl Strategy<Value = Vec<(String, String)>> {
    // A subset of keys, each with some value. Dedup by key (a label set is a map).
    prop::collection::vec(
        (prop::sample::select(KEYS), prop::sample::select(VALUES)),
        0..5,
    )
    .prop_map(|pairs| {
        let mut m: BTreeMap<String, String> = BTreeMap::new();
        for (k, v) in pairs {
            m.entry(k.to_owned()).or_insert_with(|| v.to_owned());
        }
        m.into_iter().collect()
    })
}

proptest! {
    /// The narrowing invariant, as exact equivalence: the emitted k8s
    /// constraints admit a node **iff** the native selectors do. Equivalence is
    /// strictly stronger than "never widens" (which is only the ⇐ direction).
    #[test]
    fn selectors_map_without_semantic_drift(
        selectors in prop::collection::vec(selector_strategy(), 0..6),
        labels in labels_strategy(),
    ) {
        let (ns, aff) = to_node_constraints(&selectors);
        let k8s = k8s_admits(&ns, &aff, &labels);
        let native = native_admits(&selectors, &labels);
        prop_assert_eq!(
            k8s, native,
            "emitted k8s admission ({}) disagreed with native filter ({}) for selectors={:?} labels={:?}",
            k8s, native, selectors, labels
        );
    }
}

// A closed pool of resource names, each used at most once per case.
const RES_NAMES: &[&str] = &[
    "cpu",
    "memory",
    "nvidia.com/gpu",
    "amd.com/gpu",
    "hugepages-2Mi",
];
const QUANTITIES: &[&str] = &["1", "2", "500m", "8", "16Gi", "512Mi"];

proptest! {
    /// Resource requests are copied verbatim (never lowered — the resource half
    /// of the narrowing invariant), and vendor-qualified (extended) resources
    /// land in `limits` with a matching quantity, standard ones do not.
    #[test]
    fn resource_requests_are_verbatim_and_gpus_are_extended(
        names in prop::sample::subsequence(RES_NAMES.to_vec(), 0..RES_NAMES.len()),
        qtys in prop::collection::vec(prop::sample::select(QUANTITIES), RES_NAMES.len()),
    ) {
        let reqs: Vec<ResourceRequest> = names
            .iter()
            .zip(qtys.iter())
            .map(|(n, q)| ResourceRequest::new(*n, *q))
            .collect();
        let rr = to_resource_requirements(&reqs);

        for r in &reqs {
            let got = rr
                .requests
                .as_ref()
                .and_then(|m| m.get(&r.name))
                .map(|q| q.0.as_str());
            prop_assert_eq!(got, Some(r.min_quantity.as_str()), "request not verbatim for {}", r.name);

            if r.name.contains('/') {
                let lim = rr
                    .limits
                    .as_ref()
                    .and_then(|m| m.get(&r.name))
                    .map(|q| q.0.as_str());
                prop_assert_eq!(lim, Some(r.min_quantity.as_str()), "extended resource {} missing matching limit", r.name);
            } else {
                prop_assert!(
                    rr.limits.as_ref().is_none_or(|m| !m.contains_key(&r.name)),
                    "standard resource {} should carry no limit", r.name
                );
            }
        }
    }
}

// ─── Golden fixture ────────────────────────────────────────────────────────

/// A representative GPU training placement.
fn golden_placement() -> Placement {
    Placement {
        selectors: vec![
            LabelSelector::new(
                "topology.zone",
                SelectorOp::In,
                vec!["us-east-1a".to_owned()],
            ),
            LabelSelector::new(
                "gpu.arch",
                SelectorOp::In,
                vec!["hopper".to_owned(), "ampere".to_owned()],
            ),
            LabelSelector::new("spot", SelectorOp::DoesNotExist, vec![]),
            LabelSelector::new("gpu.memory.gib", SelectorOp::Gt, vec!["40".to_owned()]),
        ],
        resources: vec![
            ResourceRequest::new("cpu", "4"),
            ResourceRequest::new("memory", "32Gi"),
            ResourceRequest::new("nvidia.com/gpu", "2"),
        ],
        tolerations: vec![TaintToleration {
            key: "nvidia.com/gpu".to_owned(),
            op: TolerationOp::Exists,
            value: None,
            effect: Some(TaintEffect::NoSchedule),
        }],
        priority_class: Some("training-high".to_owned()),
    }
}

fn golden_pod() -> PodSpec {
    let mut pod = PodSpec {
        containers: vec![Container {
            name: "trainer".to_owned(),
            image: Some("hyprstream/trainer:latest".to_owned()),
            ..Default::default()
        }],
        ..Default::default()
    };
    stamp_pod_spec(&golden_placement(), &mut pod);
    pod
}

#[test]
fn golden_training_pod_spec() {
    let pod = golden_pod();
    let yaml = serde_yaml::to_string(&pod).unwrap();
    let golden = include_str!("golden/training_pod.yaml");
    assert_eq!(
        yaml, golden,
        "mapped PodSpec drifted from golden fixture.\n--- got ---\n{yaml}\n--- want ---\n{golden}"
    );
}

#[test]
fn golden_explain_lines() {
    let sched = to_pod_scheduling(&golden_placement());
    let lines = describe_constraints(&sched);
    // Round-trip explainability: the emitted constraints are enumerable so a
    // SelectionReport can reference them.
    assert!(lines
        .iter()
        .any(|l| l == "nodeSelector: topology.zone=us-east-1a"));
    assert!(lines
        .iter()
        .any(|l| l.starts_with("nodeAffinity: gpu.arch In")));
    assert!(lines
        .iter()
        .any(|l| l == "resources.requests: nvidia.com/gpu=2"));
    assert!(lines
        .iter()
        .any(|l| l == "resources.limits: nvidia.com/gpu=2"));
    assert!(lines
        .iter()
        .any(|l| l.starts_with("toleration: nvidia.com/gpu Exists NoSchedule")));
    assert!(lines
        .iter()
        .any(|l| l == "priorityClassName: training-high"));
}
