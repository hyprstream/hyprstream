//! Event triggers and handlers for workflow execution
//!
//! Provides the EventHandler trait for testable, composable event handling.

#![allow(dead_code)] // Handlers will be used when event bus is implemented

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Result;

use super::WorkflowId;
use super::parser::InputDef;

/// Event trigger types
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

/// Event envelope for routing
#[derive(Debug, Clone)]
pub struct EventEnvelope {
    /// Event topic (e.g., "git2db.abc123.push")
    pub topic: String,

    /// Event payload
    pub payload: EventPayload,

    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Event payload types
#[derive(Debug, Clone)]
pub enum EventPayload {
    /// Repository cloned
    RepositoryCloned {
        repo_id: String,
        name: String,
        url: String,
    },

    /// Repository push
    RepositoryPush {
        repo_id: String,
        branch: String,
        commit_hash: String,
    },

    /// Repository commit
    RepositoryCommit {
        repo_id: String,
        hash: String,
        message: String,
    },

    /// Repository merge
    RepositoryMerge {
        repo_id: String,
        source_branch: String,
        target_branch: String,
    },

    /// Repository tag
    RepositoryTag {
        repo_id: String,
        tag_name: String,
        commit_hash: String,
    },

    /// Training progress
    TrainingProgress {
        model_id: String,
        step: u32,
        loss: f64,
    },

    /// Metrics breach
    MetricsBreach {
        metric_name: String,
        value: f64,
        threshold: f64,
    },
}

/// Handler result actions
#[derive(Debug)]
pub enum HandlerResult {
    /// Trigger workflow dispatch
    Dispatch {
        workflow_id: WorkflowId,
        inputs: HashMap<String, String>,
    },

    /// Rescan repository for workflows
    Rescan {
        repo_id: String,
    },

    /// No action needed
    Ignored,
}

/// Event handler trait for testable, composable handlers
#[async_trait]
pub trait EventHandler: Send + Sync {
    /// Event types this handler processes
    fn handles(&self) -> &[&str];

    /// Fine-grained matching (branch patterns, thresholds, etc.)
    fn matches(&self, event: &EventEnvelope) -> bool;

    /// Process the event
    async fn handle(&self, event: &EventEnvelope) -> Result<HandlerResult>;
}

/// Push event handler
pub struct PushHandler {
    workflow_id: WorkflowId,
    branch_pattern: Option<String>,
}

impl PushHandler {
    /// Create a new push handler
    pub fn new(workflow_id: WorkflowId, branch_pattern: Option<String>) -> Self {
        Self {
            workflow_id,
            branch_pattern,
        }
    }
}

#[async_trait]
impl EventHandler for PushHandler {
    fn handles(&self) -> &[&str] {
        &["push"]
    }

    fn matches(&self, event: &EventEnvelope) -> bool {
        if let EventPayload::RepositoryPush { branch, .. } = &event.payload {
            self.branch_pattern.as_ref().map_or(true, |pattern| {
                glob_match(pattern, branch)
            })
        } else {
            false
        }
    }

    async fn handle(&self, event: &EventEnvelope) -> Result<HandlerResult> {
        let inputs = extract_inputs(event);
        Ok(HandlerResult::Dispatch {
            workflow_id: self.workflow_id.clone(),
            inputs,
        })
    }
}

/// Metrics breach handler
pub struct MetricsBreachHandler {
    workflow_id: WorkflowId,
    metric_name: String,
    threshold: f64,
}

impl MetricsBreachHandler {
    /// Create a new metrics breach handler
    pub fn new(workflow_id: WorkflowId, metric_name: String, threshold: f64) -> Self {
        Self {
            workflow_id,
            metric_name,
            threshold,
        }
    }
}

#[async_trait]
impl EventHandler for MetricsBreachHandler {
    fn handles(&self) -> &[&str] {
        &["metrics_breach"]
    }

    fn matches(&self, event: &EventEnvelope) -> bool {
        if let EventPayload::MetricsBreach { metric_name, value, threshold } = &event.payload {
            metric_name == &self.metric_name && *value >= *threshold
        } else {
            false
        }
    }

    async fn handle(&self, event: &EventEnvelope) -> Result<HandlerResult> {
        let inputs = extract_inputs(event);
        Ok(HandlerResult::Dispatch {
            workflow_id: self.workflow_id.clone(),
            inputs,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Simple glob pattern matching
fn glob_match(pattern: &str, text: &str) -> bool {
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

/// Extract inputs from event for workflow dispatch
fn extract_inputs(event: &EventEnvelope) -> HashMap<String, String> {
    let mut inputs = HashMap::new();

    match &event.payload {
        EventPayload::RepositoryPush { repo_id, branch, commit_hash } => {
            inputs.insert("repo_id".to_string(), repo_id.clone());
            inputs.insert("branch".to_string(), branch.clone());
            inputs.insert("commit".to_string(), commit_hash.clone());
        }
        EventPayload::RepositoryCommit { repo_id, hash, message } => {
            inputs.insert("repo_id".to_string(), repo_id.clone());
            inputs.insert("commit".to_string(), hash.clone());
            inputs.insert("message".to_string(), message.clone());
        }
        EventPayload::TrainingProgress { model_id, step, loss } => {
            inputs.insert("model_id".to_string(), model_id.clone());
            inputs.insert("step".to_string(), step.to_string());
            inputs.insert("loss".to_string(), loss.to_string());
        }
        EventPayload::MetricsBreach { metric_name, value, threshold } => {
            inputs.insert("metric".to_string(), metric_name.clone());
            inputs.insert("value".to_string(), value.to_string());
            inputs.insert("threshold".to_string(), threshold.to_string());
        }
        _ => {}
    }

    inputs
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
}
