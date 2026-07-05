//! Kubernetes CRD schemas for hyprstream resources under `*.hyprstream.io`.
//!
//! This is the foundation crate for the kube-rs track of the Kubernetes epic
//! (#778). It defines the **hyprstream → Kubernetes** projection: hyprstream
//! resources as declarative Kubernetes objects, subspaced under hyprstream-owned
//! API groups so nouns never collide with third-party CRDs
//! (`serving.hyprstream.io/InferenceService` ≠ `serving.kserve.io/InferenceService`
//! by construction).
//!
//! ## Single-writer rule (baked into every schema)
//!
//! Git owns the weights. For every resource here, `spec` is **desired STEP
//! intent** (Stage → Train → Evaluate → Promote) and `status` is **observed
//! git/runtime truth**. The operator that reconciles these (K5b/K5c) is a
//! *translator*, never a second writer of weights: promotion is a git merge and
//! rollback is a git checkout, both performed through `RegistryService`. Nothing
//! in these CRDs is authoritative over the weights themselves.
//!
//! ## Two-writer drift guard
//!
//! Objects owned by the hyprstream operator carry the [`MANAGED_BY_LABEL`]
//! (value [`MANAGED_BY_VALUE`]) and `ownerReferences` to their parent. Each
//! driver refuses to adopt objects it does not own.
//!
//! ## Confused-deputy guard
//!
//! [`mesh::TenantBinding`] is the explicit, admin-created mapping between a
//! Kubernetes namespace and a hyprstream tenant. It is gated by the
//! `federation:register` scope and is the anchor the reconcilers (K5b) use to
//! reject any cross-tenant reference that a binding does not cover. It is
//! deliberately **cluster-scoped** so a tenant confined to its own namespace
//! cannot forge its own binding.
//!
//! ## Feature layout
//!
//! - default (no features): CRD *type* definitions, schema derivation, CEL
//!   validation and YAML emission — all compile with no cluster present.
//! - `k8s`: pulls in the kube client + controller-runtime for code that needs a
//!   live API server (see [`install`]). CRD types never require it.
//!
//! All CRDs are `v1alpha1`; the conversion/upgrade path to a future stored
//! version is noted per-resource and is out of scope for this crate (K5b owns
//! the conversion webhook).

pub mod mesh;
pub mod models;
pub mod serving;
pub mod training;

/// Re-export of the exact `k8s-openapi` version this crate is built against, so
/// downstream crates (e.g. `hyprstream-discovery`'s placement→PodSpec map, K4c
/// #787) can name `PodSpec` / `Quantity` / `Affinity` at the single
/// workspace-pinned `v1_32` version without independently depending on
/// `k8s-openapi` (exactly one `v1_*` is permitted workspace-wide).
pub use k8s_openapi;

#[cfg(feature = "k8s")]
pub mod install;

// Re-export the underlying `kube`/`k8s-openapi` crates so downstream crates
// (e.g. `hyprstream-workers`' `k8s` sandbox backend) consume the *same*
// versions and the single `v1_*` k8s-openapi feature this crate pins, rather
// than declaring their own — the workspace allows exactly one k8s-openapi
// `v1_*` feature, and threading these through one crate keeps that invariant
// impossible to violate by accident.
pub use k8s_openapi;
pub use kube;

pub use mesh::{TenantBinding, TenantBindingSpec, TenantBindingStatus};
pub use models::{Adapter, AdapterSpec, AdapterStatus, Model, ModelSpec, ModelStage, ModelStatus};
pub use serving::{InferenceService, InferenceServiceSpec, InferenceServiceStatus, Statefulness};
pub use training::{ResourceSpec, TrainingRun, TrainingRunSpec, TrainingRunStatus};

use k8s_openapi::apiextensions_apiserver::pkg::apis::apiextensions::v1::CustomResourceDefinition;
use kube::CustomResourceExt;

/// Label applied to every object the hyprstream operator manages.
///
/// Drivers refuse to adopt an object that does not carry this label with the
/// expected value ([`MANAGED_BY_VALUE`]), preventing two-writer drift with
/// other controllers that might project onto the same nouns.
pub const MANAGED_BY_LABEL: &str = "hyprstream.io/managed-by";

/// Canonical value for [`MANAGED_BY_LABEL`].
pub const MANAGED_BY_VALUE: &str = "hyprstream-operator";

/// The API version shared by every CRD in this crate.
pub const API_VERSION: &str = "v1alpha1";

/// Every CRD defined by this crate, in a stable order.
///
/// This is the single source of truth consumed by the `gen-crds` binary (YAML
/// emission) and by the offline structural-validation tests.
pub fn all_crds() -> Vec<CustomResourceDefinition> {
    vec![
        Model::crd(),
        Adapter::crd(),
        TrainingRun::crd(),
        InferenceService::crd(),
        TenantBinding::crd(),
    ]
}

/// `(kind, plural_group, filename_stem)` triples for each CRD, used to name the
/// emitted YAML files deterministically.
pub fn crd_manifests() -> Vec<(CustomResourceDefinition, &'static str)> {
    vec![
        (Model::crd(), "models.hyprstream.io_models"),
        (Adapter::crd(), "models.hyprstream.io_adapters"),
        (TrainingRun::crd(), "training.hyprstream.io_trainingruns"),
        (
            InferenceService::crd(),
            "serving.hyprstream.io_inferenceservices",
        ),
        (TenantBinding::crd(), "mesh.hyprstream.io_tenantbindings"),
    ]
}
