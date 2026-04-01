//! GitHub Actions compatible subscriber adapter.
//!
//! Parses `.github/workflows/*.yml` YAML files, registers workflows with
//! `WorkflowService`, builds `EventHandler`s from `on:` triggers, and
//! dispatches matching workflows when events arrive.
//!
//! This adapter uses the existing `EventSubscriber` (XPUB/XSUB proxy).
//! When Phase 7 (Secure Event Transport) is complete, this will be updated
//! to use `SecureEventSubscriber` for group-key encrypted event delivery.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use hyprstream_vfs::Subject;

use crate::error::Result;
use crate::events::{EventSubscriber, ReceivedEvent};

use super::adapter::SubscriberAdapter;
use super::service::WorkflowService;
use super::triggers::{EventHandler, HandlerResult, TopicPatternHandler, WorkerLifecycleHandler};
use super::WorkflowId;

/// GitHub Actions compatible subscriber adapter.
///
/// Scans repositories for workflow YAML files, creates event handlers
/// from the `on:` triggers, and dispatches workflows when matching
/// events are received.
pub struct GitHubActionsAdapter {
    /// Event subscriber for receiving events.
    /// Wrapped in Mutex because tmq::Subscribe is !Sync.
    subscriber: tokio::sync::Mutex<EventSubscriber>,
    /// Event handlers built from workflow triggers.
    handlers: Vec<Box<dyn EventHandler>>,
    /// Service identity for this adapter.
    subject: Subject,
    /// Topic prefixes this adapter is subscribed to.
    subscribed_prefixes: Vec<String>,
}

impl GitHubActionsAdapter {
    /// Create a new GitHubActionsAdapter.
    ///
    /// # Arguments
    /// * `subscriber` - Event subscriber for the event bus
    /// * `subject` - Service identity for workflow dispatch
    pub fn new(subscriber: EventSubscriber, subject: Subject) -> Self {
        Self {
            subscriber: tokio::sync::Mutex::new(subscriber),
            handlers: Vec::new(),
            subject,
            subscribed_prefixes: Vec::new(),
        }
    }

    /// Scan a repository for `.github/workflows/*.yml`, parse each,
    /// register with WorkflowService, and build EventHandlers from triggers.
    pub async fn load_repo(
        &mut self,
        repo_id: &str,
        service: &WorkflowService,
    ) -> Result<()> {
        // Scan the repository for workflow files.
        let workflow_defs = service.scan_repo(repo_id).await?;

        for def in workflow_defs {
            // Register the workflow definition.
            let workflow_id = service.register_workflow(def.clone()).await?;

            // Build handlers from triggers.
            for trigger in &def.triggers {
                if let Some(handler) = self.build_handler(&workflow_id, trigger) {
                    self.handlers.push(handler);
                }
            }
        }

        tracing::info!(
            repo_id = %repo_id,
            handler_count = self.handlers.len(),
            "Loaded repository workflows"
        );
        Ok(())
    }

    /// Build an EventHandler from a trigger configuration.
    fn build_handler(
        &mut self,
        workflow_id: &WorkflowId,
        trigger: &super::triggers::EventTrigger,
    ) -> Option<Box<dyn EventHandler>> {
        use super::triggers::EventTrigger;

        match trigger {
            EventTrigger::RepositoryEvent { event_type, .. } => {
                let topic = format!("repository.*.{:?}", event_type).to_lowercase();
                self.ensure_subscribed("repository.");
                Some(Box::new(TopicPatternHandler::new(
                    workflow_id.clone(),
                    topic,
                )))
            }
            EventTrigger::WorkerLifecycle {
                event_filter,
                entity_type,
            } => {
                self.ensure_subscribed("worker.");
                Some(Box::new(WorkerLifecycleHandler::new(
                    workflow_id.clone(),
                    event_filter.clone(),
                    entity_type.clone(),
                )))
            }
            EventTrigger::Custom { topic, .. } => {
                let prefix = topic.split('.').next().unwrap_or(topic);
                self.ensure_subscribed(&format!("{prefix}."));
                Some(Box::new(TopicPatternHandler::new(
                    workflow_id.clone(),
                    topic.clone(),
                )))
            }
            // WorkflowDispatch is triggered via RPC, not events.
            // Training/Metrics triggers need specialized handlers.
            EventTrigger::WorkflowDispatch { .. }
            | EventTrigger::TrainingProgress { .. }
            | EventTrigger::MetricsBreach { .. } => None,
        }
    }

    /// Track which prefixes need subscription.
    fn ensure_subscribed(&mut self, prefix: &str) {
        if !self.subscribed_prefixes.iter().any(|p| p == prefix) {
            self.subscribed_prefixes.push(prefix.to_owned());
        }
    }

    /// Get topic prefixes that need to be subscribed to.
    pub fn required_prefixes(&self) -> &[String] {
        &self.subscribed_prefixes
    }
}

#[async_trait]
impl SubscriberAdapter for GitHubActionsAdapter {
    fn name(&self) -> &str {
        "github-actions"
    }

    async fn run(
        &self,
        service: Arc<WorkflowService>,
        cancel: CancellationToken,
    ) -> Result<()> {
        tracing::info!(
            adapter = self.name(),
            prefixes = ?self.subscribed_prefixes,
            handlers = self.handlers.len(),
            "Starting adapter event loop"
        );

        let mut subscriber = self.subscriber.lock().await;

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    tracing::info!(adapter = self.name(), "Adapter cancelled, shutting down");
                    break;
                }
                event = subscriber.recv() => {
                    match event {
                        Ok((topic, payload)) => {
                            let received = ReceivedEvent::from_message(&topic, &payload);
                            for handler in &self.handlers {
                                if handler.matches(&received) {
                                    match handler.handle(&received).await {
                                        Ok(HandlerResult::Dispatch { workflow_id, inputs }) => {
                                            // Log event provenance (dispatch() rejects
                                            // _-prefixed keys to prevent injection, so
                                            // provenance is recorded in structured logs
                                            // rather than mixed into workflow inputs).
                                            tracing::info!(
                                                workflow_id = %workflow_id,
                                                event_topic = %topic,
                                                event_source = %received.source,
                                                event_entity = %received.entity_id,
                                                "Dispatching workflow from event"
                                            );

                                            if let Err(e) = service.dispatch(&workflow_id, inputs, &self.subject).await {
                                                tracing::error!(
                                                    workflow_id = %workflow_id,
                                                    error = %e,
                                                    "Failed to dispatch workflow"
                                                );
                                            }
                                        }
                                        Ok(HandlerResult::Rescan { repo_id }) => {
                                            if let Err(e) = service.rescan_repo(&repo_id).await {
                                                tracing::error!(
                                                    repo_id = %repo_id,
                                                    error = %e,
                                                    "Failed to rescan repo"
                                                );
                                            }
                                        }
                                        Ok(HandlerResult::Ignored) => {}
                                        Err(e) => {
                                            tracing::warn!(
                                                error = %e,
                                                "Handler error"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "Event subscriber error");
                            // Brief backoff before retrying.
                            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
