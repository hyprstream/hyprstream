//! WorkflowService - ZmqService implementation for workflow orchestration

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use anyhow::Result as AnyhowResult;
use async_trait::async_trait;
use hyprstream_rpc::prelude::SigningKey;
use hyprstream_rpc::service::{AuthorizeFn, EnvelopeContext, ZmqService};
use hyprstream_rpc::transport::TransportConfig;

use crate::error::Result;
use crate::events::{EventSubscriber, ReceivedEvent};
use crate::generated::workflow_client::{
    WorkflowHandler, dispatch_workflow, WorkflowResponseVariant,
    WorkflowDef as WorkflowDefWire, WorkflowInfo, WorkflowRun as WorkflowRunWire,
    JobRun as JobRunWire, StepRun as StepRunWire,
    RunStatusEnum,
    // Request types
    DispatchRequest, SubscribeRequest,
};

use super::subscription::WorkflowSubscription;
use super::triggers::{EventHandler, EventTrigger, HandlerResult};
use super::{RunId, WorkflowId, Workflow};

// ═══════════════════════════════════════════════════════════════════════════════
// Internal domain types — NOT exposed in the public API.
// The public WorkflowClient trait uses generated *Data types from the schema.
// These exist for internal service state where the domain model differs
// from the wire format (e.g., HashMap vs Vec, Option<DateTime> vs i64).
// ═══════════════════════════════════════════════════════════════════════════════

/// Internal workflow definition (domain model, not wire format)
#[derive(Debug, Clone)]
pub(crate) struct WorkflowDef {
    pub path: String,
    pub repo_id: String,
    pub workflow: Workflow,
    pub triggers: Vec<EventTrigger>,
}

/// Internal workflow run state
#[derive(Debug, Clone)]
pub(crate) struct WorkflowRun {
    pub id: String,
    pub workflow_id: String,
    pub status: RunStatus,
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
    pub jobs: HashMap<String, JobRun>,
}

/// Internal run status
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RunStatus {
    Queued,
}

/// Internal job run state
#[derive(Debug, Clone)]
pub(crate) struct JobRun {
    pub name: String,
    pub status: RunStatus,
    pub steps: Vec<StepRun>,
}

/// Internal step run state
#[derive(Debug, Clone)]
pub(crate) struct StepRun {
    pub name: String,
    pub status: RunStatus,
    pub exit_code: Option<i32>,
}

/// Service name for endpoint registry
const SERVICE_NAME: &str = "workflow";

/// WorkflowService handles workflow orchestration
///
/// Discovers workflows from repositories, subscribes to events,
/// and spawns containers via WorkerService.
pub struct WorkflowService {
    /// Registered workflows
    workflows: RwLock<HashMap<WorkflowId, WorkflowDef>>,

    /// Active runs
    runs: RwLock<HashMap<RunId, WorkflowRun>>,

    /// Per-repo workflow subscriptions (O(1) lookup)
    repo_workflows: RwLock<HashMap<String, Vec<WorkflowSubscription>>>,

    /// ZMQ context for event subscription
    context: Arc<zmq::Context>,

    /// Event handlers for different event types
    handlers: RwLock<Vec<Box<dyn EventHandler>>>,

    /// Background event loop handle
    event_loop_handle: tokio::sync::Mutex<Option<JoinHandle<()>>>,

    // Infrastructure (for ZmqService / Spawnable)
    /// Transport configuration
    transport: TransportConfig,
    /// Signing key for message authentication
    signing_key: SigningKey,

    /// Optional authorization callback (injected by parent crate)
    authorize_fn: Option<AuthorizeFn>,
}

impl WorkflowService {
    /// Create a new WorkflowService
    ///
    /// # Arguments
    ///
    /// * `context` - ZMQ context for event subscription (must be same as EventService for inproc://)
    /// * `transport` - Transport configuration for ZMQ service binding
    /// * `signing_key` - Signing key for message authentication
    pub fn new(
        context: Arc<zmq::Context>,
        transport: TransportConfig,
        signing_key: SigningKey,
    ) -> Self {
        Self {
            workflows: RwLock::new(HashMap::new()),
            runs: RwLock::new(HashMap::new()),
            repo_workflows: RwLock::new(HashMap::new()),
            context,
            handlers: RwLock::new(Vec::new()),
            event_loop_handle: tokio::sync::Mutex::new(None),
            transport,
            signing_key,
            authorize_fn: None,
        }
    }

    /// Set the authorization callback for policy checks.
    pub fn set_authorize_fn(&mut self, authorize_fn: AuthorizeFn) {
        self.authorize_fn = Some(authorize_fn);
    }

    /// Initialize the service
    pub async fn initialize(&self) -> Result<()> {
        // TODO: Scan all registered repos for workflows
        // TODO: Create handlers for each workflow trigger
        tracing::info!("WorkflowService initialized");
        Ok(())
    }

    /// Start the event subscription loop
    ///
    /// Subscribes to worker events and routes them to registered handlers.
    /// This spawns a background task that runs until `stop()` is called.
    pub async fn start(self: Arc<Self>) -> Result<()> {
        let mut subscriber = EventSubscriber::new(&self.context)?;

        // Subscribe to worker events (sandbox/container lifecycle)
        subscriber.subscribe("worker.")?;

        tracing::info!("WorkflowService subscribed to worker.* events");

        let service = self.clone();
        let handle = tokio::spawn(async move {
            service.event_loop(subscriber).await;
        });

        let mut event_loop_handle = self.event_loop_handle.lock().await;
        *event_loop_handle = Some(handle);

        Ok(())
    }

    /// Stop the event subscription loop
    pub async fn stop(&self) {
        let mut event_loop_handle = self.event_loop_handle.lock().await;
        if let Some(handle) = event_loop_handle.take() {
            handle.abort();
            tracing::info!("WorkflowService event loop stopped");
        }
    }

    /// Internal event processing loop
    async fn event_loop(&self, mut subscriber: EventSubscriber) {
        loop {
            match subscriber.recv().await {
                Ok((topic, payload)) => {
                    if let Err(e) = self.handle_event(&topic, &payload).await {
                        tracing::warn!(topic = %topic, error = %e, "Failed to handle event");
                    }
                }
                Err(e) => {
                    tracing::error!(error = %e, "Event subscriber error, stopping loop");
                    break;
                }
            }
        }
    }

    /// Handle an incoming event
    ///
    /// Creates a ReceivedEvent from the raw topic and payload,
    /// then dispatches to registered handlers.
    async fn handle_event(&self, topic: &str, payload: &[u8]) -> Result<()> {
        // Parse topic and payload into ReceivedEvent (handles deserialization)
        let event = ReceivedEvent::from_message(topic, payload);

        tracing::debug!(
            topic = %event.topic,
            source = %event.source,
            entity_id = %event.entity_id,
            event_type = %event.event_type,
            has_worker_event = event.worker_event.is_some(),
            "Received event"
        );

        // Dispatch to handlers
        let handlers = self.handlers.read().await;
        for handler in handlers.iter() {
            if handler.matches(&event) {
                match handler.handle(&event).await {
                    Ok(HandlerResult::Dispatch { workflow_id, inputs }) => {
                        tracing::info!(
                            workflow_id = %workflow_id,
                            trigger_event = %topic,
                            "Dispatching workflow from event"
                        );
                        if let Err(e) = self.dispatch(&workflow_id, inputs).await {
                            tracing::error!(
                                workflow_id = %workflow_id,
                                error = %e,
                                "Failed to dispatch workflow"
                            );
                        }
                    }
                    Ok(HandlerResult::Rescan { repo_id }) => {
                        if let Err(e) = self.rescan_repo(&repo_id).await {
                            tracing::error!(repo_id = %repo_id, error = %e, "Failed to rescan repo");
                        }
                    }
                    Ok(HandlerResult::Ignored) => {}
                    Err(e) => {
                        tracing::warn!(error = %e, "Handler error");
                    }
                }
            }
        }

        Ok(())
    }

    /// Register an event handler
    pub async fn register_handler(&self, handler: Box<dyn EventHandler>) {
        let mut handlers = self.handlers.write().await;
        handlers.push(handler);
    }

    /// Scan a repository for workflows
    pub(crate) async fn scan_repo(&self, repo_id: &str) -> Result<Vec<WorkflowDef>> {
        // TODO: Read .github/workflows/*.yml from repo
        // TODO: Parse each workflow file
        // TODO: Return list of workflow definitions
        tracing::info!(repo_id = %repo_id, "Scanning repository for workflows");
        Ok(Vec::new())
    }

    /// Register a workflow
    pub(crate) async fn register_workflow(&self, workflow: WorkflowDef) -> Result<WorkflowId> {
        let workflow_id = format!("{}:{}", workflow.repo_id, workflow.path);

        let mut workflows = self.workflows.write().await;
        workflows.insert(workflow_id.clone(), workflow);

        tracing::info!(workflow_id = %workflow_id, "Registered workflow");
        Ok(workflow_id)
    }

    /// List registered workflows (returns wire-format types directly)
    pub async fn list_workflows(&self) -> Result<Vec<WorkflowInfo>> {
        let workflows = self.workflows.read().await;

        Ok(workflows
            .iter()
            .map(|(id, def)| WorkflowInfo {
                id: id.clone(),
                name: def.workflow.name.clone(),
                path: def.path.clone(),
                repo_id: def.repo_id.clone(),
                enabled: true,
            })
            .collect())
    }

    /// Dispatch a workflow manually
    pub(crate) async fn dispatch(
        &self,
        workflow_id: &WorkflowId,
        inputs: HashMap<String, String>,
    ) -> Result<RunId> {
        let workflows = self.workflows.read().await;
        let _workflow = workflows
            .get(workflow_id)
            .ok_or_else(|| crate::error::WorkerError::WorkflowNotFound(workflow_id.clone()))?;

        let run_id = uuid::Uuid::new_v4().to_string();

        let run = WorkflowRun {
            id: run_id.clone(),
            workflow_id: workflow_id.clone(),
            status: RunStatus::Queued,
            started_at: None,
            completed_at: None,
            jobs: HashMap::new(),
        };

        let mut runs = self.runs.write().await;
        runs.insert(run_id.clone(), run);

        // TODO: Start workflow execution
        tracing::info!(
            workflow_id = %workflow_id,
            run_id = %run_id,
            inputs = ?inputs,
            "Dispatched workflow"
        );

        Ok(run_id)
    }

    /// Get a workflow run
    pub(crate) async fn get_run(&self, run_id: &RunId) -> Result<WorkflowRun> {
        let runs = self.runs.read().await;
        runs.get(run_id)
            .cloned()
            .ok_or_else(|| crate::error::WorkerError::RunNotFound(run_id.clone()))
    }

    /// List runs for a workflow
    pub(crate) async fn list_runs(&self, workflow_id: &WorkflowId) -> Result<Vec<WorkflowRun>> {
        let runs = self.runs.read().await;
        Ok(runs
            .values()
            .filter(|r| &r.workflow_id == workflow_id)
            .cloned()
            .collect())
    }

    /// Rescan a repository for workflow changes
    pub async fn rescan_repo(&self, repo_id: &str) -> Result<()> {
        let workflows = self.scan_repo(repo_id).await?;

        // Update subscriptions for this repo
        let mut repo_workflows = self.repo_workflows.write().await;
        let mut subscriptions = Vec::new();

        for wf in workflows {
            let wf_id = self.register_workflow(wf.clone()).await?;

            for trigger in &wf.triggers {
                let subscription = WorkflowSubscription::new(wf_id.clone(), trigger.clone());
                subscriptions.push(subscription);
            }
        }

        repo_workflows.insert(repo_id.to_owned(), subscriptions);

        tracing::info!(repo_id = %repo_id, "Rescanned repository");
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// WorkflowHandler Implementation — bridges generated types to business logic
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert internal RunStatus to generated RunStatusEnum
fn to_status_enum(status: &RunStatus) -> RunStatusEnum {
    match status {
        RunStatus::Queued => RunStatusEnum::Queued,
    }
}

/// Convert internal WorkflowRun to generated WorkflowRunWire
fn to_run_data(run: &WorkflowRun) -> WorkflowRunWire {
    WorkflowRunWire {
        id: run.id.clone(),
        workflow_id: run.workflow_id.clone(),
        status: to_status_enum(&run.status),
        started_at: run.started_at.map(|t| t.timestamp()).unwrap_or(0),
        completed_at: run.completed_at.map(|t| t.timestamp()).unwrap_or(0),
        jobs: run.jobs.values().map(|j| JobRunWire {
            name: j.name.clone(),
            status: to_status_enum(&j.status),
            steps: j.steps.iter().map(|s| StepRunWire {
                name: s.name.clone(),
                status: to_status_enum(&s.status),
                exit_code: s.exit_code.unwrap_or(0),
            }).collect(),
        }).collect(),
    }
}

#[async_trait::async_trait(?Send)]
impl WorkflowHandler for WorkflowService {
    async fn authorize(&self, ctx: &EnvelopeContext, resource: &str, operation: &str) -> AnyhowResult<()> {
        if let Some(ref auth_fn) = self.authorize_fn {
            let subject = ctx.subject().to_string();
            let allowed = auth_fn(subject.clone(), resource.to_owned(), operation.to_owned()).await
                .unwrap_or_else(|e| {
                    tracing::warn!("Policy check failed for {} on {}: {} - denying access", subject, resource, e);
                    false
                });
            if allowed {
                Ok(())
            } else {
                anyhow::bail!("Unauthorized: {} cannot {} on {}", subject, operation, resource)
            }
        } else {
            // No authorization configured — fail-closed
            anyhow::bail!("Authorization not configured for workflow service")
        }
    }

    async fn handle_scan_repo(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        repo_id: &str,
    ) -> AnyhowResult<WorkflowResponseVariant> {
        let workflows = self.scan_repo(repo_id).await?;
        Ok(WorkflowResponseVariant::ScanRepoResult(
            workflows.iter().map(|wf| WorkflowDefWire {
                path: wf.path.clone(),
                repo_id: wf.repo_id.clone(),
                name: wf.workflow.name.clone(),
                triggers: Vec::new(), // EventTrigger is empty struct for now
                yaml: String::new(),  // TODO: serialize workflow back to YAML
            }).collect()
        ))
    }

    async fn handle_register(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        data: &WorkflowDefWire,
    ) -> AnyhowResult<WorkflowResponseVariant> {
        let workflow_def = WorkflowDef {
            path: data.path.clone(),
            repo_id: data.repo_id.clone(),
            workflow: Workflow {
                name: data.name.clone(),
                on: super::parser::WorkflowTrigger::List(Vec::new()),
                env: HashMap::new(),
                jobs: HashMap::new(),
            },
            triggers: Vec::new(), // TODO: convert EventTrigger to EventTrigger
        };
        let workflow_id = self.register_workflow(workflow_def).await?;
        Ok(WorkflowResponseVariant::RegisterResult(workflow_id))
    }

    async fn handle_list(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
    ) -> AnyhowResult<WorkflowResponseVariant> {
        let workflows = self.list_workflows().await?;
        Ok(WorkflowResponseVariant::ListResult(workflows))
    }

    async fn handle_dispatch(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        request: &DispatchRequest,
    ) -> AnyhowResult<WorkflowResponseVariant> {
        let inputs_map: HashMap<String, String> = request.inputs.iter()
            .map(|kv| (kv.key.clone(), kv.value.clone()))
            .collect();
        let run_id = self.dispatch(&request.workflow_id.clone(), inputs_map).await?;
        Ok(WorkflowResponseVariant::DispatchResult(run_id))
    }

    async fn handle_subscribe(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        request: &SubscribeRequest,
    ) -> AnyhowResult<WorkflowResponseVariant> {
        // TODO: implement subscription via event handler registration
        tracing::info!(workflow_id = %request.workflow_id, "Subscribe request (not yet implemented)");
        Ok(WorkflowResponseVariant::SubscribeResult(
            format!("sub-{}", request.workflow_id)
        ))
    }

    async fn handle_unsubscribe(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        sub_id: &str,
    ) -> AnyhowResult<WorkflowResponseVariant> {
        // TODO: implement unsubscription
        tracing::info!(sub_id = %sub_id, "Unsubscribe request (not yet implemented)");
        Ok(WorkflowResponseVariant::UnsubscribeResult)
    }

    async fn handle_get_run(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        run_id: &str,
    ) -> AnyhowResult<WorkflowResponseVariant> {
        let run = self.get_run(&run_id.to_owned()).await?;
        let data = to_run_data(&run);
        Ok(WorkflowResponseVariant::GetRunResult(data))
    }

    async fn handle_list_runs(
        &self,
        _ctx: &EnvelopeContext,
        _request_id: u64,
        workflow_id: &str,
    ) -> AnyhowResult<WorkflowResponseVariant> {
        let runs = self.list_runs(&workflow_id.to_owned()).await?;
        Ok(WorkflowResponseVariant::ListRunsResult(
            runs.iter().map(to_run_data).collect()
        ))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ZmqService Implementation — delegates to generated dispatch_workflow
// ═══════════════════════════════════════════════════════════════════════════════

#[async_trait(?Send)]
impl ZmqService for WorkflowService {
    async fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> AnyhowResult<(Vec<u8>, Option<hyprstream_rpc::service::Continuation>)> {
        tracing::debug!(
            "Workflow request from {} (request_id={})",
            ctx.subject(),
            ctx.request_id
        );
        dispatch_workflow(self, ctx, payload).await
    }

    fn name(&self) -> &str {
        SERVICE_NAME
    }

    fn context(&self) -> &Arc<zmq::Context> {
        &self.context
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }
}
