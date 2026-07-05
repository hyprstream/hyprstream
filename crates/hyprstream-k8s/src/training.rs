//! `training.hyprstream.io/v1alpha1` — [`TrainingRun`].
//!
//! A [`TrainingRun`] is the intent to run Test-Time-Training / adapter training
//! against a model, producing a file-based adapter. It is *intent only*: the
//! STEP-loop reconciliation that dispatches it to workers and promotes results
//! via git merge is K5c. `spec` is desired intent; `status` is observed truth.

use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Coarse resource request for a training run. Lowered to a concrete PodSpec by
/// the placement→PodSpec vocabulary map (K4c); kept minimal and portable here.
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ResourceSpec {
    /// Number of GPUs requested (0 = CPU-only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpus: Option<u32>,

    /// CPU request in Kubernetes quantity form, e.g. `"2"` or `"500m"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu: Option<String>,

    /// Memory request in Kubernetes quantity form, e.g. `"8Gi"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<String>,
}

/// Desired state of a [`TrainingRun`].
///
/// # Validation
/// - `modelRef` must be non-empty.
/// - `datasetMount` must be an absolute path.
#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "training.hyprstream.io",
    version = "v1alpha1",
    kind = "TrainingRun",
    plural = "trainingruns",
    singular = "trainingrun",
    shortname = "hstrain",
    namespaced,
    status = "TrainingRunStatus",
    category = "hyprstream",
    doc = "A TTT / adapter training job intent. spec is desired intent; status is observed truth. Promotion of results happens via git merge in RegistryService (K5c), never by this operator writing weights.",
    printcolumn = r#"{"name":"Model","type":"string","jsonPath":".spec.modelRef"}"#,
    printcolumn = r#"{"name":"Phase","type":"string","jsonPath":".status.phase"}"#,
    validation = "self.spec.modelRef != ''",
    validation = "self.spec.datasetMount.startsWith('/')",
    cel
)]
#[serde(rename_all = "camelCase")]
pub struct TrainingRunSpec {
    /// Name of the [`crate::models::Model`] (same namespace) to train against.
    pub model_ref: String,

    /// Absolute mount path where the training dataset is delivered to the job
    /// (a hyprstream VFS mount, K4b/CSI).
    pub dataset_mount: String,

    /// Name of the output file-based adapter to produce, e.g.
    /// `00_finetune.safetensors`. Defaults are chosen by the reconciler when
    /// omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_name: Option<String>,

    /// Placement class / scheduling hint (`runs_on`) resolved by the scheduling
    /// substrate. Free-form label consumed by K4c.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runs_on: Option<String>,

    /// Coarse resource request for the training pod.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<ResourceSpec>,
}

/// Observed truth for a [`TrainingRun`]. Written only by the operator.
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TrainingRunStatus {
    /// Coarse lifecycle phase (`Pending`, `Running`, `Succeeded`, `Failed`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,

    /// Adapter file observed as produced by the run, once complete.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub produced_adapter: Option<String>,

    /// Human-readable detail for the current phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,

    /// `metadata.generation` last reconciled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_generation: Option<i64>,
}
