//! Generated workflow client type re-exports

// Re-export generated wire types as the canonical types for the workflow API.
// These are re-exported from the canonical `hyprstream-rpc-std` client module;
// crate::generated::workflow_client adds only the AGPL server dispatch.
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
