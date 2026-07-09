//! Live-cluster helpers, gated behind the `k8s` feature.
//!
//! Everything here needs a real API server / controller-runtime, so it is kept
//! out of the default build: the CRD *types* and YAML emission never require a
//! client. This module is intentionally small — the operator controller runtime
//! is K5b (#791); this is just enough to server-side-apply the CRDs.

use k8s_openapi::apiextensions_apiserver::pkg::apis::apiextensions::v1::CustomResourceDefinition;
use kube::api::{Patch, PatchParams};
use kube::{Api, Client};

/// Field manager used when server-side-applying hyprstream CRDs.
pub const FIELD_MANAGER: &str = "hyprstream-k8s";

/// Server-side-apply every hyprstream CRD to the connected cluster.
///
/// Idempotent: applying an unchanged CRD is a no-op. Requires cluster-admin
/// rights on `customresourcedefinitions`.
pub async fn apply_all_crds(client: Client) -> Result<Vec<CustomResourceDefinition>, kube::Error> {
    let api: Api<CustomResourceDefinition> = Api::all(client);
    let params = PatchParams::apply(FIELD_MANAGER).force();
    let mut applied = Vec::new();
    for crd in crate::all_crds() {
        let name = crd
            .metadata
            .name
            .clone()
            .unwrap_or_else(|| "<unnamed>".to_owned());
        let result = api.patch(&name, &params, &Patch::Apply(&crd)).await?;
        applied.push(result);
    }
    Ok(applied)
}
