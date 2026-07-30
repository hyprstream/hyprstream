//! Compile-only proof of the supported Apache substrate + AGPL adapter wiring.
//!
//! The adapter's dev dependency enables `hyprstream-k8s/k8s,grant`. These
//! generic functions are type-checked but never invoked, so no Kubernetes
//! client is created and no cluster is contacted.

use std::sync::Arc;

use hyprstream_k8s::kube::Client;
use hyprstream_k8s::operator::{
    run_operator_with_grant_service, HyprstreamOperatorRpc, OperatorConfig, OperatorError,
    OperatorState,
};
use hyprstream_k8s_pds::TenantGrantIssuer;

#[allow(dead_code)]
fn configured_state<R: HyprstreamOperatorRpc>(
    client: Client,
    rpc: Arc<R>,
    config: OperatorConfig,
    issuer: TenantGrantIssuer,
) -> OperatorState<R> {
    OperatorState::new(client, rpc, config).with_grant_service(Some(Arc::new(issuer)))
}

#[allow(dead_code)]
async fn configured_operator<R: HyprstreamOperatorRpc>(
    client: Client,
    rpc: Arc<R>,
    config: OperatorConfig,
    issuer: TenantGrantIssuer,
) -> Result<(), OperatorError> {
    run_operator_with_grant_service(client, rpc, config, Some(Arc::new(issuer))).await
}

#[test]
fn k8s_grant_adapter_operator_composition_type_checks() {
    // Compiling this integration target is the assertion. The functions above
    // exercise both supported public wiring surfaces without runtime effects.
}
