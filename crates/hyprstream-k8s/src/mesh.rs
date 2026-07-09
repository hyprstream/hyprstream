//! `mesh.hyprstream.io/v1alpha1` — [`TenantBinding`].
//!
//! [`TenantBinding`] is the **confused-deputy fix** from the epic #778 threat
//! table: an explicit, admin-created mapping between a Kubernetes namespace and
//! a hyprstream tenant. Creating one is gated by the `federation:register`
//! scope. The reconcilers (K5b) treat a binding as the *only* authority that
//! lets a resource in namespace `N` reference tenant `T`; any cross-binding
//! reference a `TenantBinding` does not cover is refused at admission.
//!
//! It is deliberately **cluster-scoped**: a tenant confined to its own
//! namespace must not be able to author the very binding that would grant it a
//! tenant identity. Only a cluster admin (holding `federation:register`) may
//! create these.

use kube::CustomResource;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Desired state of a [`TenantBinding`]: namespace ↔ hyprstream tenant.
///
/// # Validation
/// - `namespace` must be a valid DNS-1123 label.
/// - `tenant` must be non-empty.
///
/// # Security
/// Creating/updating a binding requires the `federation:register` scope. This
/// object is the trust anchor for cross-tenant reference checks; it is
/// cluster-scoped so it cannot be forged from inside a tenant namespace.
#[derive(CustomResource, Serialize, Deserialize, Clone, Debug, JsonSchema)]
#[kube(
    group = "mesh.hyprstream.io",
    version = "v1alpha1",
    kind = "TenantBinding",
    plural = "tenantbindings",
    singular = "tenantbinding",
    shortname = "hstb",
    status = "TenantBindingStatus",
    category = "hyprstream",
    doc = "Explicit, admin-created namespace ↔ hyprstream-tenant binding (the confused-deputy guard, gated by federation:register). Cluster-scoped so a tenant cannot forge its own binding.",
    printcolumn = r#"{"name":"Namespace","type":"string","jsonPath":".spec.namespace"}"#,
    printcolumn = r#"{"name":"Tenant","type":"string","jsonPath":".spec.tenant"}"#,
    printcolumn = r#"{"name":"Bound","type":"boolean","jsonPath":".status.bound"}"#,
    // `namespace` is a CEL reserved word, so the apiserver requires the field
    // to be referenced with the `__namespace__` escape (same rule as the Go
    // apiserver's `apiserver/schema/cel/model` escaping).
    validation = "self.spec.__namespace__.size() <= 63 && self.spec.__namespace__.matches('^[a-z0-9]([-a-z0-9]*[a-z0-9])?$')",
    validation = "self.spec.tenant != ''",
    cel
)]
#[serde(rename_all = "camelCase")]
pub struct TenantBindingSpec {
    /// The Kubernetes namespace this binding authorizes.
    pub namespace: String,

    /// The hyprstream tenant identifier (e.g. an atproto DID) the namespace is
    /// bound to.
    pub tenant: String,
}

/// Observed truth for a [`TenantBinding`]. Written only by the operator.
#[derive(Serialize, Deserialize, Clone, Debug, Default, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct TenantBindingStatus {
    /// Whether the operator has accepted and activated this binding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bound: Option<bool>,

    /// Coarse lifecycle phase (`Pending`, `Bound`, `Rejected`, ...).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,

    /// Human-readable detail, e.g. why a binding was rejected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,

    /// `metadata.generation` last reconciled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_generation: Option<i64>,
}
