//! WorkflowClient trait and ZMQ client

use crate::error::Result;
use async_trait::async_trait;
use std::collections::HashMap;

use super::{RunId, SubscriptionId, Workflow, WorkflowId};
use super::triggers::EventTrigger;

/// WorkflowClient trait for workflow operations
#[async_trait]
pub trait WorkflowClient: Send + Sync {
    /// Scan a repository for workflows
    async fn scan_repo(&self, repo_id: &str) -> Result<Vec<WorkflowDef>>;

    /// Register a workflow
    async fn register_workflow(&self, workflow: WorkflowDef) -> Result<WorkflowId>;

    /// List registered workflows
    async fn list_workflows(&self) -> Result<Vec<WorkflowInfo>>;

    /// Dispatch a workflow manually
    async fn dispatch(
        &self,
        workflow_id: &WorkflowId,
        inputs: HashMap<String, String>,
    ) -> Result<RunId>;

    /// Subscribe a workflow to an event
    async fn subscribe(
        &self,
        trigger: EventTrigger,
        workflow_id: &WorkflowId,
    ) -> Result<SubscriptionId>;

    /// Unsubscribe from events
    async fn unsubscribe(&self, sub_id: &SubscriptionId) -> Result<()>;

    /// Get a workflow run
    async fn get_run(&self, run_id: &RunId) -> Result<WorkflowRun>;

    /// List runs for a workflow
    async fn list_runs(&self, workflow_id: &WorkflowId) -> Result<Vec<WorkflowRun>>;
}

/// Workflow definition
#[derive(Debug, Clone)]
pub struct WorkflowDef {
    /// Path to workflow file (e.g., ".github/workflows/train.yml")
    pub path: String,

    /// Repository ID
    pub repo_id: String,

    /// Parsed workflow
    pub workflow: Workflow,

    /// Event triggers
    pub triggers: Vec<EventTrigger>,
}

/// Workflow info
#[derive(Debug, Clone)]
pub struct WorkflowInfo {
    /// Workflow ID
    pub id: WorkflowId,

    /// Workflow name
    pub name: String,

    /// Path in repository
    pub path: String,

    /// Repository ID
    pub repo_id: String,

    /// Whether workflow is enabled
    pub enabled: bool,
}

/// Workflow run status
#[derive(Debug, Clone)]
pub struct WorkflowRun {
    /// Run ID
    pub id: RunId,

    /// Workflow ID
    pub workflow_id: WorkflowId,

    /// Run status
    pub status: RunStatus,

    /// Started at
    pub started_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Completed at
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Job statuses
    pub jobs: HashMap<String, JobRun>,
}

/// Run status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    /// Queued
    Queued,
    /// In progress
    InProgress,
    /// Completed successfully
    Success,
    /// Failed
    Failure,
    /// Cancelled
    Cancelled,
}

/// Job run status
#[derive(Debug, Clone)]
pub struct JobRun {
    /// Job name
    pub name: String,

    /// Job status
    pub status: RunStatus,

    /// Step statuses
    pub steps: Vec<StepRun>,
}

/// Step run status
#[derive(Debug, Clone)]
pub struct StepRun {
    /// Step name
    pub name: String,

    /// Step status
    pub status: RunStatus,

    /// Exit code (if completed)
    pub exit_code: Option<i32>,
}

/// ZMQ client for WorkflowClient
pub struct WorkflowZmq {
    _endpoint: String,
}

impl WorkflowZmq {
    /// Create a new WorkflowZmq client
    pub fn new(endpoint: &str) -> Self {
        Self {
            _endpoint: endpoint.to_owned(),
        }
    }
}

#[async_trait]
impl WorkflowClient for WorkflowZmq {
    async fn scan_repo(&self, _repo_id: &str) -> Result<Vec<WorkflowDef>> {
        todo!("Implement ZMQ call")
    }

    async fn register_workflow(&self, _workflow: WorkflowDef) -> Result<WorkflowId> {
        todo!("Implement ZMQ call")
    }

    async fn list_workflows(&self) -> Result<Vec<WorkflowInfo>> {
        todo!("Implement ZMQ call")
    }

    async fn dispatch(
        &self,
        _workflow_id: &WorkflowId,
        _inputs: HashMap<String, String>,
    ) -> Result<RunId> {
        todo!("Implement ZMQ call")
    }

    async fn subscribe(
        &self,
        _trigger: EventTrigger,
        _workflow_id: &WorkflowId,
    ) -> Result<SubscriptionId> {
        todo!("Implement ZMQ call")
    }

    async fn unsubscribe(&self, _sub_id: &SubscriptionId) -> Result<()> {
        todo!("Implement ZMQ call")
    }

    async fn get_run(&self, _run_id: &RunId) -> Result<WorkflowRun> {
        todo!("Implement ZMQ call")
    }

    async fn list_runs(&self, _workflow_id: &WorkflowId) -> Result<Vec<WorkflowRun>> {
        todo!("Implement ZMQ call")
    }
}
