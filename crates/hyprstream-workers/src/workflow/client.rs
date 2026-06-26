//! Generated workflow client type re-exports

// Re-export generated wire types as the canonical types for the workflow API.
// These are produced by `generate_rpc_service!("workflow")` in crate::generated::workflow_client.
pub use crate::generated::workflow_client::{
    // Generated client
    WorkflowClient as GenWorkflowClient,
    // Wire-format data types
    WorkflowDef, WorkflowInfo, WorkflowRun,
    JobRun, StepRun, RunStatus,
    KeyValue, EventTrigger,
    // Response variant enum (for handler return)
    WorkflowResponseVariant,
};
