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
use serde_json::Value;
use thiserror::Error;

use crate::{
    Adapter, AdapterStatus, InferenceService, InferenceServiceStatus, Model, ModelStatus,
    Statefulness, TenantBinding, TrainingRun, TrainingRunStatus, API_VERSION, MANAGED_BY_LABEL,
    MANAGED_BY_VALUE,
};

/// Field manager used by the operator when writing status.
pub const OPERATOR_FIELD_MANAGER: &str = "hyprstream-operator";

const MODEL_KIND: &str = "Model";
const ADAPTER_KIND: &str = "Adapter";
const TRAINING_RUN_KIND: &str = "TrainingRun";
const INFERENCE_SERVICE_KIND: &str = "InferenceService";
const TRAINING_RUN_FINALIZER: &str = "training.hyprstream.io/finalizer";
const SERVING_APP_LABEL: &str = "hyprstream.io/serving-app";

/// Runtime configuration shared by every reconciler.
#[derive(Clone, Debug)]
pub struct OperatorConfig {
    /// Requeue interval after a successful reconcile.
    pub requeue: Duration,
    /// Maximum age of the namespace -> tenant cache.
    pub tenant_binding_cache_ttl: Duration,
    /// Container image used by native serving Deployments.
    pub serving_image: String,
    /// Gateway API parentRef name for generated HTTPRoutes.
    pub gateway_parent_ref: String,
    /// Container port exposed by the OpenAI-compatible serving process.
    pub serving_port: u16,
    /// Prometheus base URL used by generated KEDA ScaledObjects.
    pub prometheus_server: String,
    /// Request-rate threshold used by generated autoscalers.
    pub request_rate_threshold: String,
    /// Token-throughput threshold used by generated autoscalers.
    pub tokens_per_second_threshold: String,
    /// Stream-backpressure threshold used by generated autoscalers.
    pub stream_backpressure_threshold: String,
    /// Extended resource key used for GPU scheduling.
    pub gpu_resource_name: String,
    /// GPU count requested by serving pods.
    pub gpu_count: u32,
    /// Memory request for serving pods.
    pub serving_memory_request: String,
}

impl Default for OperatorConfig {
    fn default() -> Self {
        Self {
            requeue: Duration::from_secs(300),
            tenant_binding_cache_ttl: Duration::from_secs(60),
            serving_image: "ghcr.io/hyprstream/hyprstream:latest".to_owned(),
            gateway_parent_ref: "hyprstream".to_owned(),
            serving_port: 8080,
            prometheus_server: "http://prometheus-server.monitoring.svc:9090".to_owned(),
            request_rate_threshold: "10".to_owned(),
            tokens_per_second_threshold: "100".to_owned(),
            stream_backpressure_threshold: "1".to_owned(),
            gpu_resource_name: "nvidia.com/gpu".to_owned(),
            gpu_count: 1,
            serving_memory_request: "16Gi".to_owned(),
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

/// Result observed after a TrainingRun STEP reconcile.
///
/// The RPC side owns the actual Stage -> Train -> Evaluate -> Promote actions:
/// worker dispatch, stream/event progress, quality gate, and RegistryService git
/// merge/checkout. The operator records the observed phase only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainingRunObservation {
    pub phase: TrainingRunPhase,
    pub produced_adapter: Option<String>,
    pub message: Option<String>,
}

/// Kubernetes objects projected for a native hyprstream serving endpoint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServingPlan {
    pub apply: Vec<Value>,
    pub prune: Vec<Value>,
}

/// Result observed after reconciling an InferenceService serving plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferenceServiceObservation {
    pub ready_replicas: u32,
    pub url: Option<String>,
    pub message: Option<String>,
}

/// Coarse TrainingRun phases persisted into `status.phase`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingRunPhase {
    Pending,
    Staged,
    Training,
    Evaluated,
    AwaitingPromotion,
    Promoted,
    RolledBack,
    Failed,
}

impl TrainingRunPhase {
    fn as_status(self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::Staged => "Staged",
            Self::Training => "Training",
            Self::Evaluated => "Evaluated",
            Self::AwaitingPromotion => "AwaitingPromotion",
            Self::Promoted => "Promoted",
            Self::RolledBack => "RolledBack",
            Self::Failed => "Failed",
        }
    }

    fn is_terminal_status(status: &str) -> bool {
        matches!(
            status,
            "AwaitingPromotion" | "Promoted" | "RolledBack" | "Failed"
        )
    }
}

/// Thin abstraction over the hyprstream RPC clients used by K5b.
///
/// The production implementation is expected to wrap generated
/// RegistryService/ModelService/WorkerService clients. #848 is the remaining
/// cluster-topology dependency for constructing those clients over cross-pod
/// QUIC/iroh.
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

    fn reconcile_training_run<'a>(
        &'a self,
        tenant: &'a str,
        run: &'a TrainingRun,
    ) -> BoxFuture<'a, Result<TrainingRunObservation, OperatorError>>;

    fn finalize_training_run<'a>(
        &'a self,
        _tenant: &'a str,
        _run: &'a TrainingRun,
    ) -> BoxFuture<'a, Result<(), OperatorError>> {
        Box::pin(async { Ok(()) })
    }

    fn reconcile_inference_service<'a>(
        &'a self,
        tenant: &'a str,
        service: &'a InferenceService,
        plan: &'a ServingPlan,
    ) -> BoxFuture<'a, Result<InferenceServiceObservation, OperatorError>>;
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

/// Run Model, Adapter, TrainingRun, and InferenceService controllers until Ctrl-C.
pub async fn run_operator<R>(
    client: Client,
    rpc: Arc<R>,
    config: OperatorConfig,
) -> Result<(), OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    let state = Arc::new(OperatorState::new(client.clone(), rpc, config));
    let models = run_model_controller(client.clone(), Arc::clone(&state));
    let adapters = run_adapter_controller(client.clone(), Arc::clone(&state));
    let training_runs = run_training_run_controller(client.clone(), Arc::clone(&state));
    let inference_services = run_inference_service_controller(client, state);
    tokio::select! {
        result = models => result,
        result = adapters => result,
        result = training_runs => result,
        result = inference_services => result,
        _ = tokio::signal::ctrl_c() => Ok(()),
    }
}

/// Run Model and Adapter controllers until Ctrl-C.
///
/// Kept for callers that want K5b core behavior without enabling K5c
/// TrainingRun reconciliation yet.
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

/// Run the TrainingRun controller stream.
pub async fn run_training_run_controller<R>(
    client: Client,
    state: Arc<OperatorState<R>>,
) -> Result<(), OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    Controller::new(Api::<TrainingRun>::all(client), watcher::Config::default())
        .run(
            reconcile_training_run,
            training_run_error_policy::<R>,
            state,
        )
        .for_each(|result| async move {
            if let Err(error) = result {
                tracing::warn!(%error, "trainingrun reconcile failed");
            }
        })
        .await;
    Ok(())
}

/// Run the InferenceService controller stream.
pub async fn run_inference_service_controller<R>(
    client: Client,
    state: Arc<OperatorState<R>>,
) -> Result<(), OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    Controller::new(
        Api::<InferenceService>::all(client),
        watcher::Config::default(),
    )
    .run(
        reconcile_inference_service,
        inference_service_error_policy::<R>,
        state,
    )
    .for_each(|result| async move {
        if let Err(error) = result {
            tracing::warn!(%error, "inferenceservice reconcile failed");
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

async fn reconcile_training_run<R>(
    run: Arc<TrainingRun>,
    state: Arc<OperatorState<R>>,
) -> Result<Action, OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    if run.meta().deletion_timestamp.is_some() {
        return finalize_training_run(run.as_ref(), state.as_ref()).await;
    }
    if training_run_terminal_observed_current(run.as_ref()) {
        return Ok(Action::requeue(state.config.requeue));
    }
    ensure_training_run_metadata(&state.client, run.as_ref()).await?;
    let outcome = evaluate_training_run(run.as_ref(), state.as_ref()).await;
    let status = status_for_training_run_outcome(run.as_ref(), &outcome);
    if training_run_status_changed(run.as_ref(), &status) {
        patch_training_run_status(&state.client, run.as_ref(), status).await?;
    }
    outcome.map(|_| Action::requeue(state.config.requeue))
}

async fn reconcile_inference_service<R>(
    service: Arc<InferenceService>,
    state: Arc<OperatorState<R>>,
) -> Result<Action, OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    ensure_operator_label(&state.client, service.as_ref(), INFERENCE_SERVICE_KIND).await?;
    let outcome = evaluate_inference_service(service.as_ref(), state.as_ref()).await;
    patch_inference_service_status(
        &state.client,
        service.as_ref(),
        status_for_inference_service_outcome(service.as_ref(), &outcome),
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

fn training_run_error_policy<R>(
    _run: Arc<TrainingRun>,
    _error: &OperatorError,
    state: Arc<OperatorState<R>>,
) -> Action
where
    R: HyprstreamOperatorRpc,
{
    Action::requeue(state.config.requeue)
}

fn inference_service_error_policy<R>(
    _service: Arc<InferenceService>,
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

async fn evaluate_training_run<R>(
    run: &TrainingRun,
    state: &OperatorState<R>,
) -> Result<TrainingRunObservation, OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    let tenant = tenant_for_resource(state, TRAINING_RUN_KIND, run).await?;
    state.rpc.reconcile_training_run(&tenant, run).await
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

async fn evaluate_inference_service<R>(
    service: &InferenceService,
    state: &OperatorState<R>,
) -> Result<InferenceServiceObservation, OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    let tenant = tenant_for_resource(state, INFERENCE_SERVICE_KIND, service).await?;
    let plan = serving_plan(service, &tenant, &state.config);
    state
        .rpc
        .reconcile_inference_service(&tenant, service, &plan)
        .await
}

async fn finalize_training_run<R>(
    run: &TrainingRun,
    state: &OperatorState<R>,
) -> Result<Action, OperatorError>
where
    R: HyprstreamOperatorRpc,
{
    if !has_finalizer(run, TRAINING_RUN_FINALIZER) {
        return Ok(Action::await_change());
    }
    let tenant = tenant_for_resource(state, TRAINING_RUN_KIND, run).await?;
    state.rpc.finalize_training_run(&tenant, run).await?;
    remove_training_run_finalizer(&state.client, run).await?;
    Ok(Action::await_change())
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

fn training_run_terminal_observed_current(run: &TrainingRun) -> bool {
    run.status.as_ref().is_some_and(|status| {
        status.observed_generation == run.meta().generation
            && status
                .phase
                .as_deref()
                .is_some_and(TrainingRunPhase::is_terminal_status)
    })
}

fn training_run_status_changed(run: &TrainingRun, next: &TrainingRunStatus) -> bool {
    run.status
        .as_ref()
        .is_none_or(|current| status_value(current) != status_value(next))
}

fn status_value<S: Serialize>(status: &S) -> serde_json::Value {
    serde_json::to_value(status).unwrap_or(serde_json::Value::Null)
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

fn status_for_training_run_outcome(
    run: &TrainingRun,
    outcome: &Result<TrainingRunObservation, OperatorError>,
) -> TrainingRunStatus {
    match outcome {
        Ok(observed) => TrainingRunStatus {
            phase: Some(observed.phase.as_status().to_owned()),
            produced_adapter: observed.produced_adapter.clone(),
            message: observed.message.clone(),
            observed_generation: run.meta().generation,
        },
        Err(error) => TrainingRunStatus {
            phase: Some(TrainingRunPhase::Failed.as_status().to_owned()),
            produced_adapter: None,
            message: Some(error.to_string()),
            observed_generation: run.meta().generation,
        },
    }
}

fn status_for_inference_service_outcome(
    service: &InferenceService,
    outcome: &Result<InferenceServiceObservation, OperatorError>,
) -> InferenceServiceStatus {
    match outcome {
        Ok(observed) => InferenceServiceStatus {
            phase: Some(inference_service_phase(service, observed.ready_replicas).to_owned()),
            ready_replicas: Some(observed.ready_replicas),
            url: observed.url.clone(),
            message: observed.message.clone(),
            observed_generation: service.meta().generation,
        },
        Err(error) => InferenceServiceStatus {
            phase: Some("Failed".to_owned()),
            ready_replicas: Some(0),
            url: None,
            message: Some(error.to_string()),
            observed_generation: service.meta().generation,
        },
    }
}

fn inference_service_phase(service: &InferenceService, ready_replicas: u32) -> &'static str {
    if ready_replicas > 0 {
        "Ready"
    } else if service.spec.statefulness == Statefulness::Stateless && service.spec.min_replicas == 0
    {
        "ScaledToZero"
    } else {
        "Pending"
    }
}

fn serving_plan(service: &InferenceService, tenant: &str, config: &OperatorConfig) -> ServingPlan {
    let name = service.name_any();
    let namespace = service.namespace().unwrap_or_default();
    let app = serving_app_name(&name);
    let labels = serving_labels(&app, tenant);
    let owner_references = owner_references(service, INFERENCE_SERVICE_KIND);
    let min_replicas = effective_min_replicas(service);
    let max_replicas = service.spec.max_replicas.max(min_replicas.max(1));
    let statefulness = statefulness_value(&service.spec.statefulness);

    let deployment = json!({
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": app,
            "namespace": namespace,
            "labels": labels,
            "ownerReferences": owner_references,
        },
        "spec": {
            "selector": {
                "matchLabels": {
                    SERVING_APP_LABEL: app,
                },
            },
            "template": {
                "metadata": {
                    "labels": labels,
                    "annotations": {
                        "hyprstream.io/model-ref": service.spec.model,
                        "hyprstream.io/statefulness": statefulness,
                        "hyprstream.io/drain-before-scale": drain_before_scale(&service.spec.statefulness),
                        "hyprstream.io/drain-prerequisite": "https://github.com/hyprstream/hyprstream/issues/869",
                    },
                },
                "spec": {
                    "terminationGracePeriodSeconds": 120,
                    "containers": [{
                        "name": "model-service",
                        "image": config.serving_image,
                        "args": [
                            "service",
                            "start",
                            "oai",
                            "--foreground",
                            "--model", service.spec.model,
                            "--tenant", tenant,
                        ],
                        "ports": [{
                            "name": "http",
                            "containerPort": config.serving_port,
                        }],
                        "resources": serving_resources(config),
                        "readinessProbe": {
                            "httpGet": {
                                "path": "/health",
                                "port": "http",
                            },
                            "periodSeconds": 10,
                            "failureThreshold": 18,
                        },
                        "livenessProbe": {
                            "httpGet": {
                                "path": "/health",
                                "port": "http",
                            },
                            "periodSeconds": 30,
                            "failureThreshold": 3,
                        },
                        "lifecycle": lifecycle_for_statefulness(&service.spec.statefulness),
                    }],
                },
            },
        },
    });

    let service_manifest = json!({
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": app,
            "namespace": namespace,
            "labels": labels,
            "ownerReferences": owner_references,
        },
        "spec": {
            "selector": {
                SERVING_APP_LABEL: app,
            },
            "ports": [{
                "name": "http",
                "port": 80,
                "targetPort": "http",
            }],
            "sessionAffinity": "None",
        },
    });

    let http_route = json!({
        "apiVersion": "gateway.networking.k8s.io/v1",
        "kind": "HTTPRoute",
        "metadata": {
            "name": app,
            "namespace": namespace,
            "labels": labels,
            "ownerReferences": owner_references,
            "annotations": http_route_annotations(&service.spec.statefulness),
        },
        "spec": {
            "parentRefs": [{
                "name": config.gateway_parent_ref,
            }],
            "rules": [{
                "matches": [{
                    "path": {
                        "type": "PathPrefix",
                        "value": "/v1",
                    },
                }],
                "backendRefs": [{
                    "name": app,
                    "port": 80,
                }],
            }],
        },
    });

    let (autoscaler, prunes) =
        if service.spec.statefulness == Statefulness::Stateless && min_replicas == 0 {
            (
                keda_scaled_object(
                    service,
                    &app,
                    &namespace,
                    labels,
                    owner_references,
                    max_replicas,
                    &config.prometheus_server,
                    &config.request_rate_threshold,
                ),
                vec![prune_manifest(
                    "autoscaling/v2",
                    "HorizontalPodAutoscaler",
                    &app,
                    &namespace,
                )],
            )
        } else {
            (
                hpa(
                    &app,
                    &namespace,
                    labels,
                    owner_references,
                    min_replicas,
                    max_replicas,
                    config,
                ),
                vec![prune_manifest(
                    "keda.sh/v1alpha1",
                    "ScaledObject",
                    &app,
                    &namespace,
                )],
            )
        };

    ServingPlan {
        apply: vec![deployment, service_manifest, http_route, autoscaler],
        prune: prunes,
    }
}

fn serving_app_name(name: &str) -> String {
    let candidate = format!("hs-serve-{name}");
    if candidate.len() <= 63 {
        return candidate;
    }

    let hash = fnv1a32(name.as_bytes());
    format!("hs-serve-{}-{hash:08x}", &name[..45])
}

fn fnv1a32(bytes: &[u8]) -> u32 {
    let mut hash = 0x811c9dc5u32;
    for byte in bytes {
        hash ^= u32::from(*byte);
        hash = hash.wrapping_mul(0x01000193);
    }
    hash
}

fn serving_labels(app: &str, tenant: &str) -> Value {
    json!({
        MANAGED_BY_LABEL: MANAGED_BY_VALUE,
        SERVING_APP_LABEL: app,
        "hyprstream.io/tenant": tenant,
    })
}

fn owner_references(service: &InferenceService, kind: &'static str) -> Value {
    match service.meta().uid.as_ref() {
        Some(uid) => json!([{
            "apiVersion": format!("{}.hyprstream.io/{API_VERSION}", group_for_kind(kind)),
            "kind": kind,
            "name": service.name_any(),
            "uid": uid,
            "controller": true,
            "blockOwnerDeletion": true,
        }]),
        None => json!([]),
    }
}

fn effective_min_replicas(service: &InferenceService) -> u32 {
    match service.spec.statefulness {
        Statefulness::TttStateful => service.spec.min_replicas.max(1),
        Statefulness::Stateless => service.spec.min_replicas,
    }
}

fn statefulness_value(statefulness: &Statefulness) -> &'static str {
    match statefulness {
        Statefulness::Stateless => "stateless",
        Statefulness::TttStateful => "ttt-stateful",
    }
}

fn http_route_annotations(statefulness: &Statefulness) -> Value {
    match statefulness {
        Statefulness::Stateless => json!({
            "hyprstream.io/hostname-source": "did-operated-domain",
        }),
        Statefulness::TttStateful => json!({
            "hyprstream.io/hostname-source": "did-operated-domain",
            "hyprstream.io/session-persistence": "subject-header-consistent-hash",
            "hyprstream.io/session-persistence-key": "Subject",
        }),
    }
}

fn drain_before_scale(statefulness: &Statefulness) -> &'static str {
    match statefulness {
        Statefulness::Stateless => "not-required",
        Statefulness::TttStateful => "required",
    }
}

fn lifecycle_for_statefulness(statefulness: &Statefulness) -> Value {
    match statefulness {
        Statefulness::Stateless => json!({}),
        Statefulness::TttStateful => json!({
            "preStop": {
                "exec": {
                    "command": [
                        "sh",
                        "-c",
                        "hyprstream service drain --export-once || true",
                    ],
                },
            },
        }),
    }
}

fn serving_resources(config: &OperatorConfig) -> Value {
    json!({
        "requests": {
            "memory": config.serving_memory_request,
            config.gpu_resource_name.as_str(): config.gpu_count.to_string(),
        },
        "limits": {
            config.gpu_resource_name.as_str(): config.gpu_count.to_string(),
        },
    })
}

fn hpa(
    app: &str,
    namespace: &str,
    labels: Value,
    owner_references: Value,
    min_replicas: u32,
    max_replicas: u32,
    config: &OperatorConfig,
) -> Value {
    json!({
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {
            "name": app,
            "namespace": namespace,
            "labels": labels,
            "ownerReferences": owner_references,
            "annotations": {
                "hyprstream.io/autoscaling-signals": "request_rate,stream_backpressure,tokens_per_second",
            },
        },
        "spec": {
            "scaleTargetRef": {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "name": app,
            },
            "minReplicas": min_replicas,
            "maxReplicas": max_replicas,
            "metrics": [{
                "type": "Pods",
                "pods": {
                    "metric": {
                        "name": "hyprstream_request_rate",
                    },
                    "target": {
                        "type": "AverageValue",
                        "averageValue": config.request_rate_threshold,
                    },
                },
            }, {
                "type": "Pods",
                "pods": {
                    "metric": {
                        "name": "hyprstream_tokens_per_second",
                    },
                    "target": {
                        "type": "AverageValue",
                        "averageValue": config.tokens_per_second_threshold,
                    },
                },
            }, {
                "type": "Pods",
                "pods": {
                    "metric": {
                        "name": "hyprstream_stream_backpressure",
                    },
                    "target": {
                        "type": "AverageValue",
                        "averageValue": config.stream_backpressure_threshold,
                    },
                },
            }],
        },
    })
}

fn keda_scaled_object(
    service: &InferenceService,
    app: &str,
    namespace: &str,
    labels: Value,
    owner_references: Value,
    max_replicas: u32,
    prometheus_server: &str,
    request_rate_threshold: &str,
) -> Value {
    json!({
        "apiVersion": "keda.sh/v1alpha1",
        "kind": "ScaledObject",
        "metadata": {
            "name": app,
            "namespace": namespace,
            "labels": labels,
            "ownerReferences": owner_references,
            "annotations": {
                "hyprstream.io/autoscaling-signals": "request_rate,stream_backpressure,tokens_per_second",
            },
        },
        "spec": {
            "scaleTargetRef": {
                "name": app,
            },
            "minReplicaCount": 0,
            "maxReplicaCount": max_replicas,
            "triggers": [{
                "type": "prometheus",
                "metadata": {
                    "serverAddress": prometheus_server,
                    "metricName": "hyprstream_request_rate",
                    "threshold": request_rate_threshold,
                    "query": format!("sum(rate(hyprstream_requests_total{{inference_service=\"{}\"}}[1m]))", service.name_any()),
                },
            }],
        },
    })
}

fn prune_manifest(api_version: &str, kind: &str, name: &str, namespace: &str) -> Value {
    json!({
        "apiVersion": api_version,
        "kind": kind,
        "metadata": {
            "name": name,
            "namespace": namespace,
        },
        "hyprstream.io/action": "delete",
    })
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

async fn patch_training_run_status(
    client: &Client,
    run: &TrainingRun,
    status: TrainingRunStatus,
) -> Result<TrainingRun, kube::Error> {
    patch_namespaced_status(client, run, TRAINING_RUN_KIND, status).await
}

async fn ensure_training_run_metadata(
    client: &Client,
    run: &TrainingRun,
) -> Result<TrainingRun, kube::Error> {
    let namespace = run.namespace().unwrap_or_default();
    let api: Api<TrainingRun> = Api::namespaced(client.clone(), &namespace);
    api.patch(
        &run.name_any(),
        &PatchParams::apply(OPERATOR_FIELD_MANAGER).force(),
        &Patch::Apply(&training_run_metadata_patch(run)),
    )
    .await
}

async fn remove_training_run_finalizer(
    client: &Client,
    run: &TrainingRun,
) -> Result<TrainingRun, kube::Error> {
    let namespace = run.namespace().unwrap_or_default();
    let api: Api<TrainingRun> = Api::namespaced(client.clone(), &namespace);
    let finalizers: Vec<String> = run
        .meta()
        .finalizers
        .clone()
        .unwrap_or_default()
        .into_iter()
        .filter(|finalizer| finalizer != TRAINING_RUN_FINALIZER)
        .collect();
    api.patch(
        &run.name_any(),
        &PatchParams::default(),
        &Patch::Merge(json!({
            "metadata": {
                "finalizers": finalizers,
            },
        })),
    )
    .await
}

async fn patch_inference_service_status(
    client: &Client,
    service: &InferenceService,
    status: InferenceServiceStatus,
) -> Result<InferenceService, kube::Error> {
    patch_namespaced_status(client, service, INFERENCE_SERVICE_KIND, status).await
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

fn training_run_metadata_patch(run: &TrainingRun) -> serde_json::Value {
    let mut finalizers = run.meta().finalizers.clone().unwrap_or_default();
    if !finalizers
        .iter()
        .any(|finalizer| finalizer == TRAINING_RUN_FINALIZER)
    {
        finalizers.push(TRAINING_RUN_FINALIZER.to_owned());
    }
    json!({
        "apiVersion": format!("training.hyprstream.io/{API_VERSION}"),
        "kind": TRAINING_RUN_KIND,
        "metadata": {
            "name": run.name_any(),
            "namespace": run.namespace(),
            "labels": {
                MANAGED_BY_LABEL: MANAGED_BY_VALUE,
            },
            "finalizers": finalizers,
        },
    })
}

fn has_finalizer<K>(resource: &K, finalizer: &str) -> bool
where
    K: ResourceExt,
{
    resource
        .meta()
        .finalizers
        .as_ref()
        .is_some_and(|finalizers| finalizers.iter().any(|item| item == finalizer))
}

fn group_for_kind(kind: &str) -> &'static str {
    match kind {
        MODEL_KIND | ADAPTER_KIND => "models",
        TRAINING_RUN_KIND => "training",
        INFERENCE_SERVICE_KIND => "serving",
        _ => "mesh",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AdapterSpec, InferenceServiceSpec, ModelSpec, ModelStage, TenantBindingSpec,
        TrainingRunSpec,
    };

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

    #[test]
    fn training_run_promoted_status_reports_adapter() {
        let run = TrainingRun::new(
            "train-qwen",
            TrainingRunSpec {
                model_ref: "qwen".to_owned(),
                dataset_mount: "/datasets/toy".to_owned(),
                adapter_name: Some("00_toy.safetensors".to_owned()),
                runs_on: None,
                resources: None,
            },
        );
        let status = status_for_training_run_outcome(
            &run,
            &Ok(TrainingRunObservation {
                phase: TrainingRunPhase::Promoted,
                produced_adapter: Some("00_toy.safetensors".to_owned()),
                message: Some("merged refs/heads/train-qwen".to_owned()),
            }),
        );

        assert_eq!(status.phase.as_deref(), Some("Promoted"));
        assert_eq!(
            status.produced_adapter.as_deref(),
            Some("00_toy.safetensors")
        );
        assert_eq!(
            status.message.as_deref(),
            Some("merged refs/heads/train-qwen")
        );
    }

    #[test]
    fn training_run_manual_gate_status_is_distinct() {
        let run = TrainingRun::new(
            "train-qwen",
            TrainingRunSpec {
                model_ref: "qwen".to_owned(),
                dataset_mount: "/datasets/toy".to_owned(),
                adapter_name: None,
                runs_on: None,
                resources: None,
            },
        );
        let status = status_for_training_run_outcome(
            &run,
            &Ok(TrainingRunObservation {
                phase: TrainingRunPhase::AwaitingPromotion,
                produced_adapter: Some("train-qwen.safetensors".to_owned()),
                message: Some("manual promotion required".to_owned()),
            }),
        );

        assert_eq!(status.phase.as_deref(), Some("AwaitingPromotion"));
        assert_eq!(
            status.produced_adapter.as_deref(),
            Some("train-qwen.safetensors")
        );
    }

    #[test]
    fn training_run_failure_status_retains_message() {
        let run = TrainingRun::new(
            "train-qwen",
            TrainingRunSpec {
                model_ref: "qwen".to_owned(),
                dataset_mount: "/datasets/toy".to_owned(),
                adapter_name: None,
                runs_on: None,
                resources: None,
            },
        );
        let status = status_for_training_run_outcome(
            &run,
            &Err(OperatorError::Rpc(
                "worker exited 1; branch retained".to_owned(),
            )),
        );

        assert_eq!(status.phase.as_deref(), Some("Failed"));
        assert_eq!(status.produced_adapter, None);
        assert!(status
            .message
            .as_deref()
            .unwrap_or_default()
            .contains("branch retained"));
    }

    #[test]
    fn training_run_status_patch_uses_training_group() {
        let run = TrainingRun::new(
            "train-qwen",
            TrainingRunSpec {
                model_ref: "qwen".to_owned(),
                dataset_mount: "/datasets/toy".to_owned(),
                adapter_name: None,
                runs_on: None,
                resources: None,
            },
        );
        let patch = status_patch(
            &run,
            TRAINING_RUN_KIND,
            TrainingRunStatus {
                phase: Some("Training".to_owned()),
                produced_adapter: None,
                message: None,
                observed_generation: None,
            },
        );

        assert_eq!(patch["apiVersion"], "training.hyprstream.io/v1alpha1");
        assert_eq!(patch["kind"], "TrainingRun");
        assert_eq!(patch["status"]["phase"], "Training");
    }

    fn inference_service(
        name: &str,
        statefulness: Statefulness,
        min_replicas: u32,
    ) -> InferenceService {
        let mut service = InferenceService::new(
            name,
            InferenceServiceSpec {
                model: "qwen:main".to_owned(),
                min_replicas,
                max_replicas: 4,
                statefulness,
            },
        );
        service.metadata.namespace = Some("default".to_owned());
        service
    }

    fn applied<'a>(plan: &'a ServingPlan, kind: &str) -> &'a Value {
        plan.apply
            .iter()
            .find(|manifest| manifest["kind"] == kind)
            .unwrap_or_else(|| panic!("missing apply manifest kind {kind}"))
    }

    #[test]
    fn serving_plan_stateful_never_scales_to_zero() {
        let service = inference_service("chat", Statefulness::TttStateful, 0);
        let plan = serving_plan(&service, "tenant-a", &OperatorConfig::default());
        let deployment = applied(&plan, "Deployment");
        let service_manifest = applied(&plan, "Service");
        let route = applied(&plan, "HTTPRoute");
        let autoscaler = applied(&plan, "HorizontalPodAutoscaler");

        assert!(deployment["spec"].get("replicas").is_none());
        assert_eq!(service_manifest["spec"]["sessionAffinity"], "None");
        assert_eq!(
            route["metadata"]["annotations"]["hyprstream.io/session-persistence"],
            "subject-header-consistent-hash"
        );
        assert_eq!(
            route["metadata"]["annotations"]["hyprstream.io/session-persistence-key"],
            "Subject"
        );
        assert!(route["spec"].get("hostnames").is_none());
        assert_eq!(autoscaler["spec"]["minReplicas"], 1);
        assert_eq!(plan.prune[0]["kind"], "ScaledObject");
        assert_eq!(
            deployment["spec"]["template"]["metadata"]["annotations"]["hyprstream.io/statefulness"],
            "ttt-stateful"
        );
        assert_eq!(
            deployment["spec"]["template"]["metadata"]["annotations"]
                ["hyprstream.io/drain-before-scale"],
            "required"
        );
        assert_eq!(
            deployment["spec"]["template"]["spec"]["containers"][0]["lifecycle"]["preStop"]["exec"]
                ["command"],
            json!(["sh", "-c", "hyprstream service drain --export-once || true"])
        );
    }

    #[test]
    fn serving_plan_stateless_zero_uses_keda_scaled_object() {
        let service = inference_service("embed", Statefulness::Stateless, 0);
        let plan = serving_plan(&service, "tenant-a", &OperatorConfig::default());
        let deployment = applied(&plan, "Deployment");
        let service_manifest = applied(&plan, "Service");
        let autoscaler = applied(&plan, "ScaledObject");

        assert!(deployment["spec"].get("replicas").is_none());
        assert_eq!(service_manifest["spec"]["sessionAffinity"], "None");
        assert_eq!(autoscaler["apiVersion"], "keda.sh/v1alpha1");
        assert_eq!(autoscaler["spec"]["minReplicaCount"], 0);
        assert_eq!(plan.prune[0]["kind"], "HorizontalPodAutoscaler");
        assert_eq!(
            autoscaler["spec"]["triggers"][0]["metadata"]["serverAddress"],
            OperatorConfig::default().prometheus_server
        );
    }

    #[test]
    fn serving_plan_routes_openai_surface_through_gateway() {
        let service = inference_service("chat", Statefulness::Stateless, 1);
        let plan = serving_plan(&service, "tenant-a", &OperatorConfig::default());
        let deployment = applied(&plan, "Deployment");
        let route = applied(&plan, "HTTPRoute");

        assert_eq!(route["apiVersion"], "gateway.networking.k8s.io/v1");
        assert_eq!(route["kind"], "HTTPRoute");
        assert!(route["spec"].get("hostnames").is_none());
        assert_eq!(
            route["metadata"]["annotations"]["hyprstream.io/hostname-source"],
            "did-operated-domain"
        );
        assert_eq!(
            route["spec"]["rules"][0]["matches"][0]["path"]["value"],
            "/v1"
        );
        assert_eq!(
            route["spec"]["rules"][0]["backendRefs"][0]["name"],
            "hs-serve-chat"
        );
        assert_eq!(
            deployment["spec"]["template"]["spec"]["containers"][0]["args"],
            json!([
                "service",
                "start",
                "oai",
                "--foreground",
                "--model",
                "qwen:main",
                "--tenant",
                "tenant-a"
            ])
        );
        assert_eq!(
            deployment["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]
                ["nvidia.com/gpu"],
            "1"
        );
        assert_eq!(
            deployment["spec"]["template"]["spec"]["containers"][0]["readinessProbe"]["httpGet"]
                ["path"],
            "/health"
        );
        assert_eq!(
            deployment["spec"]["template"]["spec"]["containers"][0]["livenessProbe"]["httpGet"]
                ["path"],
            "/health"
        );
    }

    #[test]
    fn serving_plan_app_name_fits_dns_label() {
        let name = "a".repeat(63);
        let app = serving_app_name(&name);

        assert!(app.len() <= 63);
        assert!(app.starts_with("hs-serve-"));
    }

    #[test]
    fn inference_service_status_patch_uses_serving_group() {
        let service = inference_service("chat", Statefulness::Stateless, 1);
        let patch = status_patch(
            &service,
            INFERENCE_SERVICE_KIND,
            InferenceServiceStatus {
                phase: Some("Ready".to_owned()),
                ready_replicas: Some(1),
                url: Some("https://chat.example.com/v1".to_owned()),
                message: Some("published via discovery".to_owned()),
                observed_generation: None,
            },
        );

        assert_eq!(patch["apiVersion"], "serving.hyprstream.io/v1alpha1");
        assert_eq!(patch["kind"], "InferenceService");
        assert_eq!(patch["status"]["readyReplicas"], 1);
        assert_eq!(patch["status"]["message"], "published via discovery");
    }

    #[test]
    fn inference_service_ready_status_uses_observed_url() {
        let service = inference_service("chat", Statefulness::Stateless, 1);
        let status = status_for_inference_service_outcome(
            &service,
            &Ok(InferenceServiceObservation {
                ready_replicas: 2,
                url: Some("https://chat.example.com/v1".to_owned()),
                message: Some("observed from DiscoveryService".to_owned()),
            }),
        );

        assert_eq!(status.phase.as_deref(), Some("Ready"));
        assert_eq!(status.ready_replicas, Some(2));
        assert_eq!(status.url.as_deref(), Some("https://chat.example.com/v1"));
        assert_eq!(
            status.message.as_deref(),
            Some("observed from DiscoveryService")
        );
    }

    #[test]
    fn inference_service_stateless_zero_status_is_scaled_to_zero() {
        let service = inference_service("embed", Statefulness::Stateless, 0);
        let status = status_for_inference_service_outcome(
            &service,
            &Ok(InferenceServiceObservation {
                ready_replicas: 0,
                url: Some("https://embed.example.com/v1".to_owned()),
                message: None,
            }),
        );

        assert_eq!(status.phase.as_deref(), Some("ScaledToZero"));
        assert_eq!(status.ready_replicas, Some(0));
    }
}
