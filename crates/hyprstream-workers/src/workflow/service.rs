//! WorkflowService - ZmqService implementation for workflow orchestration

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use crate::error::Result;
use crate::events::{EventSubscriber, ReceivedEvent};

use super::client::{WorkflowDef, WorkflowInfo, WorkflowRun, RunStatus};
use super::subscription::WorkflowSubscription;
use super::triggers::{EventHandler, HandlerResult};
use super::{RunId, WorkflowId};

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
}

impl WorkflowService {
    /// Create a new WorkflowService
    ///
    /// # Arguments
    ///
    /// * `context` - ZMQ context for event subscription (must be same as EventService for inproc://)
    pub fn new(context: Arc<zmq::Context>) -> Self {
        Self {
            workflows: RwLock::new(HashMap::new()),
            runs: RwLock::new(HashMap::new()),
            repo_workflows: RwLock::new(HashMap::new()),
            context,
            handlers: RwLock::new(Vec::new()),
            event_loop_handle: tokio::sync::Mutex::new(None),
        }
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
    pub async fn scan_repo(&self, repo_id: &str) -> Result<Vec<WorkflowDef>> {
        // TODO: Read .github/workflows/*.yml from repo
        // TODO: Parse each workflow file
        // TODO: Return list of workflow definitions
        tracing::info!(repo_id = %repo_id, "Scanning repository for workflows");
        Ok(Vec::new())
    }

    /// Register a workflow
    pub async fn register_workflow(&self, workflow: WorkflowDef) -> Result<WorkflowId> {
        let workflow_id = format!("{}:{}", workflow.repo_id, workflow.path);

        let mut workflows = self.workflows.write().await;
        workflows.insert(workflow_id.clone(), workflow);

        tracing::info!(workflow_id = %workflow_id, "Registered workflow");
        Ok(workflow_id)
    }

    /// List registered workflows
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
    pub async fn dispatch(
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
    pub async fn get_run(&self, run_id: &RunId) -> Result<WorkflowRun> {
        let runs = self.runs.read().await;
        runs.get(run_id)
            .cloned()
            .ok_or_else(|| crate::error::WorkerError::RunNotFound(run_id.clone()))
    }

    /// List runs for a workflow
    pub async fn list_runs(&self, workflow_id: &WorkflowId) -> Result<Vec<WorkflowRun>> {
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

        repo_workflows.insert(repo_id.to_string(), subscriptions);

        tracing::info!(repo_id = %repo_id, "Rescanned repository");
        Ok(())
    }
}

// Note: Default implementation removed - WorkflowService requires zmq::Context
