//! WorkflowService - ZmqService implementation for workflow orchestration

use std::collections::HashMap;
use tokio::sync::RwLock;

use crate::error::Result;

use super::client::{WorkflowDef, WorkflowInfo, WorkflowRun, RunStatus};
use super::subscription::WorkflowSubscription;
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

    // TODO: Add these when implementing
    // worker_client: RuntimeZmq,
    // registry_client: RegistryZmq,
    // subscriber: zmq::Socket,
}

impl WorkflowService {
    /// Create a new WorkflowService
    pub fn new() -> Self {
        Self {
            workflows: RwLock::new(HashMap::new()),
            runs: RwLock::new(HashMap::new()),
            repo_workflows: RwLock::new(HashMap::new()),
        }
    }

    /// Initialize the service
    pub async fn initialize(&self) -> Result<()> {
        // TODO: Scan all registered repos for workflows
        // TODO: Create handlers for each workflow trigger
        // TODO: Subscribe to event topics
        tracing::info!("WorkflowService initialized");
        Ok(())
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

impl Default for WorkflowService {
    fn default() -> Self {
        Self::new()
    }
}
