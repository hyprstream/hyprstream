//! `serving.hyprstream.io/v1alpha1` — [`InferenceService`].
//!
//! Serving intent for a model:branch, with autoscaling bounds and a
//! statefulness class. The group is DNS-subspaced under `hyprstream.io` so it
//! never collides with `serving.kserve.io/InferenceService`; KServe is an
//! optional adapter target (K6b), never the owner of this noun.
//!
//! `spec` is desired intent; `status` is observed serving truth.

use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Statefulness class of a served model, which the serving controller (K6a)
/// uses to decide autoscaling behaviour.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Default, JsonSchema)]
pub enum Statefulness {
    /// Stateless inference; replicas are freely interchangeable.
    #[default]
    Stateless,
    /// Test-Time-Training-stateful: replicas hold per-tenant LoRA deltas, so
    /// scaling/routing must be TTT-aware (session affinity, drain-before-scale).
    TttStateful,
}

/// Desired state of an [`InferenceService`].
///
/// # Validation
/// - `model` must be non-empty.
/// - `minReplicas` must be `<= maxReplicas`.
/// - `maxReplicas` must be `>= 1`.
#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "serving.hyprstream.io",
    version = "v1alpha1",
    kind = "InferenceService",
    plural = "inferenceservices",
    singular = "inferenceservice",
    shortname = "hsisvc",
    namespaced,
    status = "InferenceServiceStatus",
    category = "hyprstream",
    doc = "Serving intent for a model:branch with autoscaling bounds and a statefulness class. Subspaced under serving.hyprstream.io so it never collides with serving.kserve.io.",
    printcolumn = r#"{"name":"Model","type":"string","jsonPath":".spec.model"}"#,
    printcolumn = r#"{"name":"Ready","type":"integer","jsonPath":".status.readyReplicas"}"#,
    printcolumn = r#"{"name":"URL","type":"string","jsonPath":".status.url"}"#,
    validation = "self.spec.model != ''",
    validation = "self.spec.minReplicas <= self.spec.maxReplicas",
    validation = "self.spec.maxReplicas >= 1",
    cel
)]
#[serde(rename_all = "camelCase")]
pub struct InferenceServiceSpec {
    /// Model to serve, in `model:branch` form (branch optional, defaults to the
    /// promoted ref).
    pub model: String,

    /// Minimum replica count. `0` permits scale-to-zero for stateless services.
    #[serde(default)]
    pub min_replicas: u32,

    /// Maximum replica count. Must be at least `1`.
    #[serde(default = "default_max_replicas")]
    pub max_replicas: u32,

    /// Statefulness class driving autoscaling behaviour.
    #[serde(default)]
    pub statefulness: Statefulness,
}

fn default_max_replicas() -> u32 {
    1
}

/// Observed serving truth for an [`InferenceService`]. Written only by the
/// operator.
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct InferenceServiceStatus {
    /// Coarse lifecycle phase (`Pending`, `Ready`, `Failed`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,

    /// Number of ready replicas currently serving.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ready_replicas: Option<u32>,

    /// Externally reachable URL, once serving.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,

    /// `metadata.generation` last reconciled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_generation: Option<i64>,
}
