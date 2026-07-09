//! `models.hyprstream.io/v1alpha1` — [`Model`] and [`Adapter`].
//!
//! A [`Model`] is **git-ref intent**: which repository, which ref, and which
//! STEP stage the operator should observe. An [`Adapter`] is **file-based
//! adapter intent**: a `.safetensors` file under `adapters/` attached to a
//! model (adapters are files, never branches — see the project's adapter model).
//!
//! Single-writer rule: `spec` is desired intent, `status` is observed git
//! truth. The operator never writes weights; promotion/rollback happen in git
//! via `RegistryService`.

use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// STEP-loop stage a model ref should be observed at.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Default, JsonSchema)]
pub enum ModelStage {
    /// Candidate weights on a staging branch, not yet promoted.
    #[default]
    Staged,
    /// Weights merged/promoted onto the mainline (the served ref).
    Promoted,
}

/// Desired state of a [`Model`]: a git-ref intent.
///
/// # Validation
/// - `repo` must be non-empty.
/// - `gitRef` (when set) must be non-empty.
#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "models.hyprstream.io",
    version = "v1alpha1",
    kind = "Model",
    plural = "models",
    singular = "model",
    shortname = "hsmodel",
    namespaced,
    status = "ModelStatus",
    category = "hyprstream",
    doc = "A hyprstream model as a git-ref intent (Stage → Train → Evaluate → Promote). spec is desired intent; status is observed git truth. The operator never writes weights.",
    printcolumn = r#"{"name":"Repo","type":"string","jsonPath":".spec.repo"}"#,
    printcolumn = r#"{"name":"Ref","type":"string","jsonPath":".status.observedRef"}"#,
    printcolumn = r#"{"name":"Stage","type":"string","jsonPath":".spec.stage"}"#,
    validation = "self.spec.repo != ''",
    validation = "!has(self.spec.gitRef) || self.spec.gitRef != ''",
    cel
)]
#[serde(rename_all = "camelCase")]
pub struct ModelSpec {
    /// Git repository the model lives in (URL or registry id). Required.
    pub repo: String,

    /// Git ref (branch, tag, or commit) to observe. Defaults to the repository
    /// default branch when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_ref: Option<String>,

    /// STEP stage the operator should observe this model at.
    #[serde(default)]
    pub stage: ModelStage,
}

/// Observed git/runtime truth for a [`Model`]. Written only by the operator.
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct ModelStatus {
    /// Resolved commit (or ref) currently observed in git.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_ref: Option<String>,

    /// Coarse lifecycle phase (`Pending`, `Ready`, `Failed`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,

    /// Human-readable detail for the current phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,

    /// `metadata.generation` last reconciled, for staleness detection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_generation: Option<i64>,
}

/// Desired state of an [`Adapter`]: a file-based adapter intent.
///
/// # Validation
/// - `modelRef` must be non-empty.
/// - `file` must match the file-based adapter naming convention
///   `NN_name.safetensors` (two leading digits, then `_`, then a name,
///   ending in `.safetensors`).
#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "models.hyprstream.io",
    version = "v1alpha1",
    kind = "Adapter",
    plural = "adapters",
    singular = "adapter",
    shortname = "hsadapter",
    namespaced,
    status = "AdapterStatus",
    category = "hyprstream",
    doc = "A file-based LoRA adapter intent stored as adapters/NN_name.safetensors on a model. Adapters are files, not branches.",
    printcolumn = r#"{"name":"Model","type":"string","jsonPath":".spec.modelRef"}"#,
    printcolumn = r#"{"name":"File","type":"string","jsonPath":".spec.file"}"#,
    validation = "self.spec.modelRef != ''",
    validation = "self.spec.file.matches('^[0-9]{2}_[A-Za-z0-9._-]+\\\\.safetensors$')",
    cel
)]
#[serde(rename_all = "camelCase")]
pub struct AdapterSpec {
    /// Name of the [`Model`] (same namespace) this adapter attaches to.
    pub model_ref: String,

    /// Adapter filename under `adapters/`, e.g. `00_style.safetensors`.
    pub file: String,

    /// Optional base git ref the adapter was trained against, for provenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_ref: Option<String>,
}

/// Observed truth for an [`Adapter`]. Written only by the operator.
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct AdapterStatus {
    /// The adapter file the operator observed on the model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_file: Option<String>,

    /// Coarse lifecycle phase (`Pending`, `Ready`, `Failed`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,

    /// `metadata.generation` last reconciled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_generation: Option<i64>,
}
