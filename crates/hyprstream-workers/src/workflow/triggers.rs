//! Event triggers and handlers for workflow execution
//!
//! Provides the EventHandler trait for testable, composable event handling.
//! Uses ReceivedEvent from the events module for EventService integration.

#![allow(dead_code)] // Handlers will be used when event bus is implemented

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Result;
use crate::events::{ReceivedEvent, WorkerEvent};

use super::parser::InputDef;
use super::WorkflowId;

// ═══════════════════════════════════════════════════════════════════════════════
// Trigger Configuration Types
// ═══════════════════════════════════════════════════════════════════════════════

/// Event trigger types for workflow configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventTrigger {
    /// Repository events - triggers rescan + registered workflows
    RepositoryEvent {
        /// Event type
        event_type: RepoEventType,
        /// Glob pattern for branch/path filtering
        pattern: Option<String>,
    },

    /// Training progress (for auto-checkpoint workflows)
    TrainingProgress {
        /// Model ID
        model_id: String,
        /// Minimum step to trigger
        min_step: Option<u32>,
    },

    /// Metrics threshold breach (for auto-tune workflows)
    MetricsBreach {
        /// Metric name
        metric_name: String,
        /// Threshold value
        threshold: f64,
    },

    /// Manual workflow dispatch
    WorkflowDispatch {
        /// Input definitions
        inputs: HashMap<String, InputDef>,
    },

    /// Worker lifecycle events (sandbox/container)
    WorkerLifecycle {
        /// Event filter: "started", "stopped", or None for all
        event_filter: Option<String>,
        /// Entity type: "sandbox", "container", or None for all
        entity_type: Option<String>,
    },

    /// Custom topic subscription
    Custom {
        /// Topic pattern
        topic: String,
        /// Message pattern
        pattern: String,
    },
}

/// Repository event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RepoEventType {
    /// Repository cloned
    Clone,
    /// Branch pushed
    Push,
    /// Commit created
    Commit,
    /// Branch merged
    Merge,
    /// Pull request opened/updated
    PullRequest,
    /// Tag created
    Tag,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Handler Result
// ═══════════════════════════════════════════════════════════════════════════════

/// Handler result actions
#[derive(Debug)]
pub enum HandlerResult {
    /// Trigger workflow dispatch
    Dispatch {
        workflow_id: WorkflowId,
        inputs: HashMap<String, String>,
    },

    /// Rescan repository for workflows
    Rescan { repo_id: String },

    /// No action needed
    Ignored,
}

// ═══════════════════════════════════════════════════════════════════════════════
// Event Handler Trait (uses ReceivedEvent from EventService)
// ═══════════════════════════════════════════════════════════════════════════════

/// Event handler trait for testable, composable handlers.
///
/// Handlers receive ReceivedEvent from the EventService subscriber
/// and decide whether to dispatch workflows based on the event.
#[async_trait]
pub trait EventHandler: Send + Sync {
    /// Check if this handler should process the event.
    ///
    /// Called for every event; should be fast (topic/type checking only).
    fn matches(&self, event: &ReceivedEvent) -> bool;

    /// Process the event and return an action.
    ///
    /// Only called if `matches()` returned true.
    async fn handle(&self, event: &ReceivedEvent) -> Result<HandlerResult>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Worker Lifecycle Handler
// ═══════════════════════════════════════════════════════════════════════════════

/// Handler for worker lifecycle events (sandbox/container started/stopped).
///
/// Triggers workflows when sandboxes or containers reach specific states.
pub struct WorkerLifecycleHandler {
    workflow_id: WorkflowId,
    /// Filter by event type: "started", "stopped", or None for all
    event_filter: Option<String>,
    /// Filter by entity type: "sandbox", "container", or None for all
    entity_type: Option<String>,
}

impl WorkerLifecycleHandler {
    /// Create a new worker lifecycle handler
    pub fn new(
        workflow_id: WorkflowId,
        event_filter: Option<String>,
        entity_type: Option<String>,
    ) -> Self {
        Self {
            workflow_id,
            event_filter,
            entity_type,
        }
    }

    /// Create a handler for all sandbox events
    pub fn sandboxes(workflow_id: WorkflowId) -> Self {
        Self::new(workflow_id, None, Some("sandbox".to_string()))
    }

    /// Create a handler for all container events
    pub fn containers(workflow_id: WorkflowId) -> Self {
        Self::new(workflow_id, None, Some("container".to_string()))
    }

    /// Create a handler for started events only
    pub fn on_started(workflow_id: WorkflowId) -> Self {
        Self::new(workflow_id, Some("started".to_string()), None)
    }

    /// Create a handler for stopped events only
    pub fn on_stopped(workflow_id: WorkflowId) -> Self {
        Self::new(workflow_id, Some("stopped".to_string()), None)
    }
}

#[async_trait]
impl EventHandler for WorkerLifecycleHandler {
    fn matches(&self, event: &ReceivedEvent) -> bool {
        // Only handle worker events
        if event.source != "worker" {
            return false;
        }

        // Check event type filter
        if let Some(ref filter) = self.event_filter {
            if &event.event_type != filter {
                return false;
            }
        }

        // Check entity type filter
        if let Some(ref entity_type) = self.entity_type {
            if let Some(ref worker_event) = event.worker_event {
                match (entity_type.as_str(), worker_event) {
                    ("sandbox", WorkerEvent::SandboxStarted(_))
                    | ("sandbox", WorkerEvent::SandboxStopped(_)) => {}
                    ("container", WorkerEvent::ContainerStarted(_))
                    | ("container", WorkerEvent::ContainerStopped(_)) => {}
                    _ => return false,
                }
            } else {
                return false;
            }
        }

        true
    }

    async fn handle(&self, event: &ReceivedEvent) -> Result<HandlerResult> {
        let inputs = event.extract_inputs();
        Ok(HandlerResult::Dispatch {
            workflow_id: self.workflow_id.clone(),
            inputs,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Topic Pattern Handler
// ═══════════════════════════════════════════════════════════════════════════════

/// Handler that matches events by topic pattern.
///
/// Useful for custom event routing or debugging.
pub struct TopicPatternHandler {
    workflow_id: WorkflowId,
    /// Topic prefix or exact match
    topic_pattern: String,
}

impl TopicPatternHandler {
    /// Create a new topic pattern handler
    pub fn new(workflow_id: WorkflowId, topic_pattern: String) -> Self {
        Self {
            workflow_id,
            topic_pattern,
        }
    }
}

#[async_trait]
impl EventHandler for TopicPatternHandler {
    fn matches(&self, event: &ReceivedEvent) -> bool {
        event.matches_pattern(&self.topic_pattern)
    }

    async fn handle(&self, event: &ReceivedEvent) -> Result<HandlerResult> {
        let inputs = event.extract_inputs();
        Ok(HandlerResult::Dispatch {
            workflow_id: self.workflow_id.clone(),
            inputs,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

/// Simple glob pattern matching
pub fn glob_match(pattern: &str, text: &str) -> bool {
    // Simple implementation: * matches any sequence
    if pattern == "*" {
        return true;
    }

    if pattern.contains('*') {
        let parts: Vec<&str> = pattern.split('*').collect();
        if parts.len() == 2 {
            // prefix*suffix
            let prefix = parts[0];
            let suffix = parts[1];
            return text.starts_with(prefix) && text.ends_with(suffix);
        }
    }

    pattern == text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glob_match() {
        assert!(glob_match("main", "main"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("feature/*", "feature/foo"));
        assert!(!glob_match("main", "develop"));
    }

    #[test]
    fn test_worker_lifecycle_handler_matches() {
        // Create a handler for sandbox started events
        let handler =
            WorkerLifecycleHandler::new("test:wf".to_string(), Some("started".to_string()), None);

        // Create test events
        let started_event = ReceivedEvent::from_message("worker.sandbox123.started", &[]);
        let stopped_event = ReceivedEvent::from_message("worker.sandbox123.stopped", &[]);
        let other_event = ReceivedEvent::from_message("registry.repo456.push", &[]);

        assert!(handler.matches(&started_event));
        assert!(!handler.matches(&stopped_event));
        assert!(!handler.matches(&other_event));
    }

    #[test]
    fn test_topic_pattern_handler_matches() {
        let handler = TopicPatternHandler::new("test:wf".to_string(), "worker.".to_string());

        let worker_event = ReceivedEvent::from_message("worker.sandbox123.started", &[]);
        let registry_event = ReceivedEvent::from_message("registry.repo456.push", &[]);

        assert!(handler.matches(&worker_event));
        assert!(!handler.matches(&registry_event));
    }
}
