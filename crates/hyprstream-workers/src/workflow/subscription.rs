//! Workflow subscriptions for event routing

use super::triggers::EventTrigger;
use super::WorkflowId;

/// Workflow subscription linking a workflow to an event trigger
#[derive(Debug, Clone)]
pub struct WorkflowSubscription {
    /// Workflow ID
    pub workflow_id: WorkflowId,

    /// Event trigger
    pub trigger: EventTrigger,

    /// Whether subscription is active
    pub active: bool,
}

impl WorkflowSubscription {
    /// Create a new subscription
    pub fn new(workflow_id: WorkflowId, trigger: EventTrigger) -> Self {
        Self {
            workflow_id,
            trigger,
            active: true,
        }
    }

    /// Deactivate the subscription
    pub fn deactivate(&mut self) {
        self.active = false;
    }
}
