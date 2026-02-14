//! WorkflowClient trait and generated type re-exports
//!
//! The trait uses generated wire-format types from the Cap'n Proto schema.
//! Domain types for internal service state are defined in service.rs.

use crate::error::Result;
use async_trait::async_trait;

use super::{RunId, SubscriptionId, WorkflowId};

// Re-export generated wire types as the canonical types for the workflow API.
// These are produced by `generate_rpc_service!("workflow")` in crate::generated::workflow_client.
pub use crate::generated::workflow_client::{
    // Generated client
    WorkflowClient as GenWorkflowClient,
    // Wire-format data types
    WorkflowDef, WorkflowInfo, WorkflowRun,
    JobRun, StepRun, RunStatusEnum,
    KeyValue, EventTrigger,
    // Response variant enum (for handler return)
    WorkflowResponseVariant,
};

/// WorkflowClient trait for workflow operations
///
/// Uses generated wire-format types from the Cap'n Proto schema directly,
/// eliminating redundant domain type wrappers.
#[async_trait]
pub trait WorkflowClient: Send + Sync {
    /// Scan a repository for workflows
    async fn scan_repo(&self, repo_id: &str) -> Result<Vec<WorkflowDef>>;

    /// Register a workflow
    async fn register_workflow(&self, def: &WorkflowDef) -> Result<WorkflowId>;

    /// List registered workflows
    async fn list_workflows(&self) -> Result<Vec<WorkflowInfo>>;

    /// Dispatch a workflow manually
    async fn dispatch(
        &self,
        workflow_id: &WorkflowId,
        inputs: &[KeyValue],
    ) -> Result<RunId>;

    /// Subscribe a workflow to events
    async fn subscribe(&self, workflow_id: &WorkflowId) -> Result<SubscriptionId>;

    /// Unsubscribe from events
    async fn unsubscribe(&self, sub_id: &SubscriptionId) -> Result<()>;

    /// Get a workflow run
    async fn get_run(&self, run_id: &RunId) -> Result<WorkflowRun>;

    /// List runs for a workflow
    async fn list_runs(&self, workflow_id: &WorkflowId) -> Result<Vec<WorkflowRun>>;
}
