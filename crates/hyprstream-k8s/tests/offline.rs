//! Offline validation of the hyprstream CRDs.
//!
//! Two independent checks, neither needing a cluster:
//! 1. Structural validity — every CRD serializes and carries a group, a served
//!    + stored version, names, and an embedded openAPIv3 schema.
//! 2. CEL enforcement — the derive's client-side `validate_cel()` (kube `cel`
//!    feature) rejects the specs the `x-kubernetes-validations` rules forbid and
//!    accepts the ones they allow, so the acceptance-criteria "invalid specs
//!    rejected" holds without an apiserver.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::str_to_string)]

use hyprstream_k8s::{
    Adapter, AdapterSpec, InferenceService, InferenceServiceSpec, Model, ModelSpec, ModelStage,
    Statefulness, TenantBinding, TenantBindingSpec, TrainingRun, TrainingRunSpec,
};
use kube::CustomResourceExt;

#[test]
fn all_crds_are_structurally_valid() {
    let crds = hyprstream_k8s::all_crds();
    assert_eq!(crds.len(), 5, "expected 5 CRDs");

    for crd in &crds {
        // Round-trips through serde cleanly.
        let json = serde_json::to_value(crd).unwrap();
        let name = crd.metadata.name.as_deref().unwrap();
        assert!(name.contains(".hyprstream.io"), "bad group in {name}");

        let spec = &json["spec"];
        assert!(spec["group"].as_str().unwrap().ends_with("hyprstream.io"));
        assert!(spec["names"]["kind"].is_string());
        assert!(spec["names"]["plural"].is_string());

        let versions = spec["versions"].as_array().unwrap();
        assert_eq!(versions.len(), 1, "one version per CRD in {name}");
        let v = &versions[0];
        assert_eq!(v["name"], "v1alpha1");
        assert_eq!(v["served"], true);
        assert_eq!(v["storage"], true);
        // Schema is embedded (structural schema required by apiextensions/v1).
        assert!(
            v["schema"]["openAPIV3Schema"].is_object(),
            "missing schema in {name}"
        );
        // Every resource exposes a status subresource.
        assert!(
            v["subresources"]["status"].is_object(),
            "no status subresource in {name}"
        );
    }
}

#[test]
fn crds_declare_expected_groups_and_kinds() {
    let expected = [
        ("models.hyprstream.io", "Model"),
        ("models.hyprstream.io", "Adapter"),
        ("training.hyprstream.io", "TrainingRun"),
        ("serving.hyprstream.io", "InferenceService"),
        ("mesh.hyprstream.io", "TenantBinding"),
    ];
    let got: Vec<(String, String)> = hyprstream_k8s::all_crds()
        .iter()
        .map(|c| {
            let j = serde_json::to_value(c).unwrap();
            (
                j["spec"]["group"].as_str().unwrap().to_string(),
                j["spec"]["names"]["kind"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    for (g, k) in expected {
        assert!(
            got.iter().any(|(gg, kk)| gg == g && kk == k),
            "missing {g}/{k}"
        );
    }
}

#[test]
fn tenant_binding_is_cluster_scoped() {
    let crd = TenantBinding::crd();
    let j = serde_json::to_value(&crd).unwrap();
    // Cluster scope is the confused-deputy guard: a tenant in its own namespace
    // must not be able to forge its binding.
    assert_eq!(j["spec"]["scope"], "Cluster");
    assert_eq!(TenantBinding::crd().spec.scope, "Cluster");
}

#[test]
fn model_cel_rejects_empty_repo() {
    let bad = Model::new(
        "m",
        ModelSpec {
            repo: String::new(),
            git_ref: None,
            stage: ModelStage::Staged,
        },
    );
    assert!(bad.validate_cel().is_err(), "empty repo must be rejected");

    let good = Model::new(
        "m",
        ModelSpec {
            repo: "hf://org/model".into(),
            git_ref: Some("main".into()),
            stage: ModelStage::Promoted,
        },
    );
    assert!(good.validate_cel().is_ok(), "valid model must pass");
}

#[test]
fn model_cel_rejects_empty_git_ref() {
    let bad = Model::new(
        "m",
        ModelSpec {
            repo: "r".into(),
            git_ref: Some(String::new()),
            stage: ModelStage::Staged,
        },
    );
    assert!(bad.validate_cel().is_err(), "empty gitRef must be rejected");
}

#[test]
fn adapter_cel_enforces_filename_convention() {
    let bad = Adapter::new(
        "a",
        AdapterSpec {
            model_ref: "m".into(),
            file: "adapter.bin".into(),
            base_ref: None,
        },
    );
    assert!(
        bad.validate_cel().is_err(),
        "non-conforming filename must be rejected"
    );

    let good = Adapter::new(
        "a",
        AdapterSpec {
            model_ref: "m".into(),
            file: "00_style.safetensors".into(),
            base_ref: None,
        },
    );
    assert!(good.validate_cel().is_ok(), "conforming filename must pass");
}

#[test]
fn training_run_cel_requires_absolute_dataset_mount() {
    let bad = TrainingRun::new(
        "t",
        TrainingRunSpec {
            model_ref: "m".into(),
            dataset_mount: "relative/path".into(),
            adapter_name: None,
            runs_on: None,
            resources: None,
        },
    );
    assert!(
        bad.validate_cel().is_err(),
        "relative dataset mount must be rejected"
    );

    let good = TrainingRun::new(
        "t",
        TrainingRunSpec {
            model_ref: "m".into(),
            dataset_mount: "/data/train".into(),
            adapter_name: Some("00_ft.safetensors".into()),
            runs_on: Some("gpu".into()),
            resources: None,
        },
    );
    assert!(
        good.validate_cel().is_ok(),
        "absolute dataset mount must pass"
    );
}

#[test]
fn inference_service_cel_enforces_replica_bounds() {
    let bad = InferenceService::new(
        "i",
        InferenceServiceSpec {
            model: "m:main".into(),
            min_replicas: 5,
            max_replicas: 2,
            statefulness: Statefulness::Stateless,
        },
    );
    assert!(bad.validate_cel().is_err(), "min > max must be rejected");

    let good = InferenceService::new(
        "i",
        InferenceServiceSpec {
            model: "m:main".into(),
            min_replicas: 1,
            max_replicas: 4,
            statefulness: Statefulness::TttStateful,
        },
    );
    assert!(good.validate_cel().is_ok(), "valid bounds must pass");
}

#[test]
fn tenant_binding_cel_validates_namespace_and_tenant() {
    let bad_ns = TenantBinding::new(
        "tb",
        TenantBindingSpec {
            namespace: "Not_A_DNS_Label".into(),
            tenant: "did:web:acme".into(),
        },
    );
    assert!(
        bad_ns.validate_cel().is_err(),
        "invalid namespace must be rejected"
    );

    let bad_tenant = TenantBinding::new(
        "tb",
        TenantBindingSpec {
            namespace: "acme".into(),
            tenant: String::new(),
        },
    );
    assert!(
        bad_tenant.validate_cel().is_err(),
        "empty tenant must be rejected"
    );

    let good = TenantBinding::new(
        "tb",
        TenantBindingSpec {
            namespace: "acme".into(),
            tenant: "did:web:acme".into(),
        },
    );
    assert!(good.validate_cel().is_ok(), "valid binding must pass");
}

#[test]
fn committed_yaml_matches_generated() {
    // Guards against forgetting to re-run `gen-crds` after a schema change.
    for (crd, stem) in hyprstream_k8s::crd_manifests() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/crds/");
        let file = std::path::Path::new(path).join(format!("{stem}.yaml"));
        let on_disk = std::fs::read_to_string(&file)
            .unwrap_or_else(|e| panic!("read {}: {e}", file.display()));
        let regenerated = serde_yaml::to_string(&crd).unwrap();
        assert!(
            on_disk.contains(regenerated.trim()),
            "committed {} is stale; re-run `cargo run -p hyprstream-k8s --bin gen-crds`",
            file.display()
        );
    }
}
