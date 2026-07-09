//! kube-rs operator runtime for hyprstream CRDs.
//!
//! This is K5b (#791): reconcilers translate Kubernetes intent into
//! hyprstream RPC calls, then reflect observed git/runtime truth back into
//! status. They are deliberately not a second writer of model weights.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::BoxFuture;
use futures::StreamExt;
use kube::api::{ListParams, Patch, PatchParams};
use kube::core::NamespaceResourceScope;
use kube::runtime::controller::Action;
use kube::runtime::{watcher, Controller};
use kube::{Api, Client, Resource, ResourceExt};
use serde::Serialize;
use serde_json::json;
use thiserror::Error;

use crate::{
    Adapter, AdapterStatus, Model, ModelStatus, TenantBinding, API_VERSION, MANAGED_BY_LABEL,
    MANAGED_BY_VALUE,
};

/// Field manager used by the operator when writing status.
pub const OPERATOR_FIELD_MANAGER: &str = "hyprstream-operator";

const MODEL_KIND: &str = "Model";
const ADAPTER_KIND: &str = "Adapter";

/// Runtime configuration shared by every reconciler.
#[derive(Clone, Debug)]
pub struct OperatorConfig {
    /// Requeue interval after a successful reconcile.
    pub requeue: Duration,
    /// Maximum age of the namespace -> tenant cache.
    pub tenant_binding_cache_ttl: Duration,
}

impl Default for OperatorConfig {
    fn default() -> Self {
        Self {
            requeue: Duration::from_secs(300),
            tenant_binding_cache_ttl: Duration::from_secs(60),
        }
    }
}

/// Result observed after a model reconcile through RegistryService/ModelService.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelObservation {
    pub observed_ref: String,
    pub message: Option<String>,
}

/// Result observed after an adapter reconcile through RegistryService.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterObservation {
    pub observed_file: String,
    pub message: Option<String>,
}

/// Thin abstraction over the hyprstream RPC clients used by K5b.
///
/// The production implementation is expected to wrap generated
/// RegistryService/ModelService clients. #848 is the remaining cluster-topology
/// dependency for constructing those clients over cross-pod QUIC/iroh.
pub trait HyprstreamOperatorRpc: Send + Sync + 'static {
    fn reconcile_model<'a>(
        &'a self,
        tenant: &'a str,
        model: &'a Model,
    ) -> BoxFuture<'a, Result<ModelObservation, OperatorError>>;

    fn reconcile_adapter<'a>(
        &'a self,
        tenant: &'a str,
        adapter: &'a Adapter,
    ) -> BoxFuture<'a, Result<AdapterObservation, OperatorError>>;
}

/// Shared controller state.
#[derive(Clone)]
pub struct OperatorState<R> {
    client: Client,
    rpc: Arc<R>,
    config: OperatorConfig,
    tenant_bindings: TenantBindingCache,
}

impl<R> OperatorState<R>
where
    R: HyprstreamOperatorRpc,
{
    pub fn new(client: Client, rpc: Arc<R>, config: OperatorConfig) -> Self {
        Self {
            client,
            rpc,
            config,
            tenant_bindings: TenantBindingCache::default(),
        }
    }
}

#[derive(Clone, Default)]
struct TenantBindingCache {
    inner: Arc<tokio::sync::RwLock<CachedTenantBindings>>,
}

#[derive(Default)]
struct CachedTenantBindings {
    refreshed_at: Option<Instant>,
    tenants_by_namespace: HashMap<String, String>,
}

/// Operator runtime errors surfaced into status conditions and logs.
#[derive(Debug, Error)]
pub enum OperatorError {
    #[error("resource {kind}/{name} is missing metadata.namespace")]
    MissingNamespace { kind: &'static str, name: String },

    #[error("namespace {namespace} is not bound to a hyprstream tenant")]
    TenantBindingMissing { namespace: String },

    #[error("namespace {namespace} has multiple TenantBindings: {bindings}")]
    TenantBindingDuplicate { namespace: String, bindings: String },

    #[error("kubernetes API error: {0}")]
    Kube(#[from] kube::Error),

    #[error("hyprstream RPC error: {0}")]
    Rpc(String),
}

/// Run Model and Adapter controllers until the process receives Ctrl-C.
pub async fn run_model_adapter_operator<R>(
    client: Client,
    rpc: Arc<R>,
    config: OperatorConfig,
) -> Result<(), OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    let state = Arc::new(OperatorState::new(client.clone(), rpc, config));
    let models = run_model_controller(client.clone(), Arc::clone(&state));
    let adapters = run_adapter_controller(client, state);
    tokio::select! {
        result = models => result,
        result = adapters => result,
        _ = tokio::signal::ctrl_c() => Ok(()),
    }
}

/// Run the Model controller stream.
pub async fn run_model_controller<R>(
    client: Client,
    state: Arc<OperatorState<R>>,
) -> Result<(), OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    Controller::new(Api::<Model>::all(client), watcher::Config::default())
        .run(reconcile_model, model_error_policy::<R>, state)
        .for_each(|result| async move {
            if let Err(error) = result {
                tracing::warn!(%error, "model reconcile failed");
            }
        })
        .await;
    Ok(())
}

/// Run the Adapter controller stream.
pub async fn run_adapter_controller<R>(
    client: Client,
    state: Arc<OperatorState<R>>,
) -> Result<(), OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    Controller::new(Api::<Adapter>::all(client), watcher::Config::default())
        .run(reconcile_adapter, adapter_error_policy::<R>, state)
        .for_each(|result| async move {
            if let Err(error) = result {
                tracing::warn!(%error, "adapter reconcile failed");
            }
        })
        .await;
    Ok(())
}

async fn reconcile_model<R>(
    model: Arc<Model>,
    state: Arc<OperatorState<R>>,
) -> Result<Action, OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    if model_status_observed_current(model.as_ref()) {
        return Ok(Action::requeue(state.config.requeue));
    }
    ensure_operator_label(&state.client, model.as_ref(), MODEL_KIND).await?;
    let outcome = evaluate_model(model.as_ref(), state.as_ref()).await;
    patch_model_status(
        &state.client,
        model.as_ref(),
        status_for_model_outcome(model.as_ref(), &outcome),
    )
    .await?;
    outcome.map(|_| Action::requeue(state.config.requeue))
}

async fn reconcile_adapter<R>(
    adapter: Arc<Adapter>,
    state: Arc<OperatorState<R>>,
) -> Result<Action, OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    if adapter_status_observed_current(adapter.as_ref()) {
        return Ok(Action::requeue(state.config.requeue));
    }
    ensure_operator_label(&state.client, adapter.as_ref(), ADAPTER_KIND).await?;
    let outcome = evaluate_adapter(adapter.as_ref(), state.as_ref()).await;
    patch_adapter_status(
        &state.client,
        adapter.as_ref(),
        status_for_adapter_outcome(adapter.as_ref(), &outcome),
    )
    .await?;
    outcome.map(|_| Action::requeue(state.config.requeue))
}

fn model_error_policy<R>(
    _model: Arc<Model>,
    _error: &OperatorError,
    state: Arc<OperatorState<R>>,
) -> Action
where
    R: HyprstreamOperatorRpc,
{
    Action::requeue(state.config.requeue)
}

fn adapter_error_policy<R>(
    _adapter: Arc<Adapter>,
    _error: &OperatorError,
    state: Arc<OperatorState<R>>,
) -> Action
where
    R: HyprstreamOperatorRpc,
{
    Action::requeue(state.config.requeue)
}

async fn evaluate_model<R>(
    model: &Model,
    state: &OperatorState<R>,
) -> Result<ModelObservation, OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    let tenant = tenant_for_resource(state, MODEL_KIND, model).await?;
    state.rpc.reconcile_model(&tenant, model).await
}

async fn evaluate_adapter<R>(
    adapter: &Adapter,
    state: &OperatorState<R>,
) -> Result<AdapterObservation, OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    let tenant = tenant_for_resource(state, ADAPTER_KIND, adapter).await?;
    state.rpc.reconcile_adapter(&tenant, adapter).await
}

async fn tenant_for_resource<R, K>(
    state: &OperatorState<R>,
    kind: &'static str,
    resource: &K,
) -> Result<String, OperatorError>
where
    R: HyprstreamOperatorRpc,
    K: Resource + ResourceExt,
{
    let name = resource.name_any();
    let namespace = resource
        .namespace()
        .ok_or_else(|| OperatorError::MissingNamespace { kind, name })?;
    state.tenant_for_namespace(&namespace).await
}

impl<R> OperatorState<R>
where
    R: HyprstreamOperatorRpc,
{
    async fn tenant_for_namespace(&self, namespace: &str) -> Result<String, OperatorError> {
        if let Some(tenant) = self
            .tenant_bindings
            .get_fresh(namespace, self.config.tenant_binding_cache_ttl)
            .await
        {
            return Ok(tenant);
        }

        let bindings: Api<TenantBinding> = Api::all(self.client.clone());
        let list = bindings.list(&ListParams::default()).await?;
        let tenants_by_namespace = tenant_map(list)?;
        self.tenant_bindings
            .replace(tenants_by_namespace, Instant::now())
            .await;
        self.tenant_bindings
            .get_any(namespace)
            .await
            .ok_or_else(|| OperatorError::TenantBindingMissing {
                namespace: namespace.to_owned(),
            })
    }
}

impl TenantBindingCache {
    async fn get_fresh(&self, namespace: &str, ttl: Duration) -> Option<String> {
        let cache = self.inner.read().await;
        let fresh = cache
            .refreshed_at
            .is_some_and(|refreshed_at| refreshed_at.elapsed() <= ttl);
        if fresh {
            cache.tenants_by_namespace.get(namespace).cloned()
        } else {
            None
        }
    }

    async fn get_any(&self, namespace: &str) -> Option<String> {
        self.inner
            .read()
            .await
            .tenants_by_namespace
            .get(namespace)
            .cloned()
    }

    async fn replace(&self, tenants_by_namespace: HashMap<String, String>, refreshed_at: Instant) {
        let mut cache = self.inner.write().await;
        cache.refreshed_at = Some(refreshed_at);
        cache.tenants_by_namespace = tenants_by_namespace;
    }
}

fn tenant_map(
    bindings: impl IntoIterator<Item = TenantBinding>,
) -> Result<HashMap<String, String>, OperatorError> {
    let mut tenants = HashMap::new();
    let mut owners: HashMap<String, Vec<String>> = HashMap::new();

    for binding in bindings {
        let namespace = binding.spec.namespace.clone();
        owners
            .entry(namespace.clone())
            .or_default()
            .push(binding.name_any());
        tenants
            .entry(namespace)
            .or_insert_with(|| binding.spec.tenant.clone());
    }

    if let Some((namespace, bindings)) = owners
        .into_iter()
        .find(|(_namespace, bindings)| bindings.len() > 1)
    {
        let mut bindings = bindings;
        bindings.sort();
        return Err(OperatorError::TenantBindingDuplicate {
            namespace,
            bindings: bindings.join(","),
        });
    }

    Ok(tenants)
}

fn model_status_observed_current(model: &Model) -> bool {
    model
        .status
        .as_ref()
        .and_then(|status| status.observed_generation)
        == model.meta().generation
}

fn adapter_status_observed_current(adapter: &Adapter) -> bool {
    adapter
        .status
        .as_ref()
        .and_then(|status| status.observed_generation)
        == adapter.meta().generation
}

fn status_for_model_outcome(
    model: &Model,
    outcome: &Result<ModelObservation, OperatorError>,
) -> ModelStatus {
    match outcome {
        Ok(observed) => ModelStatus {
            observed_ref: Some(observed.observed_ref.clone()),
            phase: Some("Ready".to_owned()),
            message: observed.message.clone(),
            observed_generation: model.meta().generation,
        },
        Err(error) => ModelStatus {
            observed_ref: None,
            phase: Some("Rejected".to_owned()),
            message: Some(error.to_string()),
            observed_generation: model.meta().generation,
        },
    }
}

fn status_for_adapter_outcome(
    adapter: &Adapter,
    outcome: &Result<AdapterObservation, OperatorError>,
) -> AdapterStatus {
    match outcome {
        Ok(observed) => AdapterStatus {
            observed_file: Some(observed.observed_file.clone()),
            phase: Some("Ready".to_owned()),
            observed_generation: adapter.meta().generation,
        },
        Err(_error) => AdapterStatus {
            observed_file: None,
            phase: Some("Rejected".to_owned()),
            observed_generation: adapter.meta().generation,
        },
    }
}

async fn ensure_operator_label<K>(
    client: &Client,
    resource: &K,
    kind: &'static str,
) -> Result<K, kube::Error>
where
    K: Clone
        + std::fmt::Debug
        + Resource<Scope = NamespaceResourceScope>
        + ResourceExt
        + serde::de::DeserializeOwned,
    <K as Resource>::DynamicType: Default,
{
    let namespace = resource.namespace().unwrap_or_default();
    let api: Api<K> = Api::namespaced(client.clone(), &namespace);
    api.patch(
        &resource.name_any(),
        &PatchParams::apply(OPERATOR_FIELD_MANAGER).force(),
        &Patch::Apply(&metadata_patch(resource, kind)),
    )
    .await
}

async fn patch_model_status(
    client: &Client,
    model: &Model,
    status: ModelStatus,
) -> Result<Model, kube::Error> {
    patch_namespaced_status(client, model, MODEL_KIND, status).await
}

async fn patch_adapter_status(
    client: &Client,
    adapter: &Adapter,
    status: AdapterStatus,
) -> Result<Adapter, kube::Error> {
    patch_namespaced_status(client, adapter, ADAPTER_KIND, status).await
}

async fn patch_namespaced_status<K, S>(
    client: &Client,
    resource: &K,
    kind: &'static str,
    status: S,
) -> Result<K, kube::Error>
where
    K: Clone + Resource<Scope = NamespaceResourceScope> + ResourceExt + serde::de::DeserializeOwned,
    <K as Resource>::DynamicType: Default,
    S: Serialize,
{
    let namespace = resource.namespace().unwrap_or_default();
    let api: Api<K> = Api::namespaced(client.clone(), &namespace);
    let patch = status_patch(resource, kind, status);
    api.patch_status(
        &resource.name_any(),
        &PatchParams::apply(OPERATOR_FIELD_MANAGER).force(),
        &Patch::Apply(&patch),
    )
    .await
}

fn status_patch<K, S>(resource: &K, kind: &'static str, status: S) -> serde_json::Value
where
    K: ResourceExt,
    S: Serialize,
{
    json!({
        "apiVersion": format!("{}.hyprstream.io/{API_VERSION}", group_for_kind(kind)),
        "kind": kind,
        "metadata": {
            "name": resource.name_any(),
            "namespace": resource.namespace(),
        },
        "status": status,
    })
}

fn metadata_patch<K>(resource: &K, kind: &'static str) -> serde_json::Value
where
    K: ResourceExt,
{
    json!({
        "apiVersion": format!("{}.hyprstream.io/{API_VERSION}", group_for_kind(kind)),
        "kind": kind,
        "metadata": {
            "name": resource.name_any(),
            "namespace": resource.namespace(),
            "labels": {
                MANAGED_BY_LABEL: MANAGED_BY_VALUE,
            },
        },
    })
}

fn group_for_kind(kind: &str) -> &'static str {
    match kind {
        MODEL_KIND | ADAPTER_KIND => "models",
        _ => "mesh",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdapterSpec, ModelSpec, ModelStage, TenantBindingSpec};

    #[test]
    fn model_success_status_reports_observed_ref() {
        let model = Model::new(
            "qwen",
            ModelSpec {
                repo: "hf://org/qwen".to_owned(),
                git_ref: Some("main".to_owned()),
                stage: ModelStage::Promoted,
            },
        );
        let status = status_for_model_outcome(
            &model,
            &Ok(ModelObservation {
                observed_ref: "refs/heads/main".to_owned(),
                message: Some("checked out".to_owned()),
            }),
        );

        assert_eq!(status.phase.as_deref(), Some("Ready"));
        assert_eq!(status.observed_ref.as_deref(), Some("refs/heads/main"));
        assert_eq!(status.message.as_deref(), Some("checked out"));
    }

    #[test]
    fn model_binding_failure_is_rejected_status() {
        let model = Model::new(
            "qwen",
            ModelSpec {
                repo: "hf://org/qwen".to_owned(),
                git_ref: Some("main".to_owned()),
                stage: ModelStage::Promoted,
            },
        );
        let status = status_for_model_outcome(
            &model,
            &Err(OperatorError::TenantBindingMissing {
                namespace: "tenant-a".to_owned(),
            }),
        );

        assert_eq!(status.phase.as_deref(), Some("Rejected"));
        assert!(status
            .message
            .as_deref()
            .unwrap_or_default()
            .contains("tenant-a"));
    }

    #[test]
    fn adapter_success_status_reports_file() {
        let adapter = Adapter::new(
            "style",
            AdapterSpec {
                model_ref: "qwen".to_owned(),
                file: "00_style.safetensors".to_owned(),
                base_ref: None,
            },
        );
        let status = status_for_adapter_outcome(
            &adapter,
            &Ok(AdapterObservation {
                observed_file: "00_style.safetensors".to_owned(),
                message: None,
            }),
        );

        assert_eq!(status.phase.as_deref(), Some("Ready"));
        assert_eq!(
            status.observed_file.as_deref(),
            Some("00_style.safetensors")
        );
    }

    #[test]
    fn model_metadata_patch_marks_operator_ownership() {
        let model = Model::new(
            "qwen",
            ModelSpec {
                repo: "hf://org/qwen".to_owned(),
                git_ref: Some("main".to_owned()),
                stage: ModelStage::Promoted,
            },
        );
        let patch = metadata_patch(&model, MODEL_KIND);

        assert_eq!(patch["apiVersion"], "models.hyprstream.io/v1alpha1");
        assert_eq!(patch["kind"], "Model");
        assert_eq!(
            patch["metadata"]["labels"][MANAGED_BY_LABEL],
            MANAGED_BY_VALUE
        );
    }

    #[test]
    fn adapter_status_patch_uses_models_group() {
        let adapter = Adapter::new(
            "style",
            AdapterSpec {
                model_ref: "qwen".to_owned(),
                file: "00_style.safetensors".to_owned(),
                base_ref: None,
            },
        );
        let patch = status_patch(
            &adapter,
            ADAPTER_KIND,
            AdapterStatus {
                observed_file: Some("00_style.safetensors".to_owned()),
                phase: Some("Ready".to_owned()),
                observed_generation: None,
            },
        );

        assert_eq!(patch["apiVersion"], "models.hyprstream.io/v1alpha1");
        assert_eq!(patch["kind"], "Adapter");
        assert_eq!(patch["status"]["observedFile"], "00_style.safetensors");
    }

    #[test]
    fn current_generation_status_short_circuits_reconcile() {
        let mut model = Model::new(
            "qwen",
            ModelSpec {
                repo: "hf://org/qwen".to_owned(),
                git_ref: Some("main".to_owned()),
                stage: ModelStage::Promoted,
            },
        );
        model.metadata.generation = Some(7);
        model.status = Some(ModelStatus {
            observed_generation: Some(7),
            ..Default::default()
        });

        assert!(model_status_observed_current(&model));

        model.status = Some(ModelStatus {
            observed_generation: Some(6),
            ..Default::default()
        });
        assert!(!model_status_observed_current(&model));
    }

    #[test]
    fn tenant_map_rejects_duplicate_namespace_bindings() {
        let first = TenantBinding::new(
            "tenant-a",
            TenantBindingSpec {
                namespace: "team-a".to_owned(),
                tenant: "did:web:tenant-a".to_owned(),
            },
        );
        let second = TenantBinding::new(
            "tenant-b",
            TenantBindingSpec {
                namespace: "team-a".to_owned(),
                tenant: "did:web:tenant-b".to_owned(),
            },
        );

        let error = match tenant_map([first, second]) {
            Ok(_) => panic!("duplicate must fail"),
            Err(error) => error,
        };
        assert!(matches!(
            error,
            OperatorError::TenantBindingDuplicate { namespace, .. } if namespace == "team-a"
        ));
    }

    #[test]
    fn tenant_map_is_namespace_keyed() {
        let binding = TenantBinding::new(
            "tenant-a",
            TenantBindingSpec {
                namespace: "team-a".to_owned(),
                tenant: "did:web:tenant-a".to_owned(),
            },
        );

        match tenant_map([binding]) {
            Ok(tenants) => assert_eq!(
                tenants.get("team-a").map(String::as_str),
                Some("did:web:tenant-a")
            ),
            Err(error) => panic!("valid binding failed: {error}"),
        }
    }
}
