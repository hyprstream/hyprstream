//! GitHub Actions compatible subscriber adapter.
//!
//! Parses `.github/workflows/*.yml` YAML files, registers workflows with
//! `WorkflowService`, builds `EventHandler`s from `on:` triggers, and
//! dispatches matching workflows when events arrive.
//!
//! This adapter uses the canonical `EventSubscriber` (moq-lite backed, #167).
//! It currently subscribes in `Public` (plaintext) mode; group-key encrypted
//! delivery is available by calling `EventSubscriber::join_prefix` for the
//! prefixes this adapter cares about — not yet wired here.

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
    /// Wrapped in Mutex because EventSubscriber is !Sync.
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
    ///
    /// Generic (non-gate) loader: parses in [`Legacy`](super::parser::ParseMode)
    /// mode via [`WorkflowService::scan_repo`]. The merge gate opts into
    /// strict mode via [`Self::load_repo_with`] (#1432).
    pub async fn load_repo(
        &mut self,
        repo_id: &str,
        service: &WorkflowService,
    ) -> Result<()> {
        self.load_repo_with(repo_id, service, super::parser::ParseMode::Legacy).await
    }

    /// Load a repository's workflows with an explicit [`ParseMode`] (#1432).
    ///
    /// This is the **gate-specific boundary** for strict selection: a merge
    /// gate calls `load_repo_with(repo, svc, ParseMode::Strict)` so that a
    /// workflow file using unsupported semantics (`permissions:`,
    /// `concurrency:`, `services:`) is refused rather than silently loaded
    /// with dropped keys. In [`ParseMode::Legacy`] (the default for the
    /// generic adapter) unknown keys are tolerated, preserving non-gate
    /// caller compatibility (#1432 non-goal).
    ///
    /// The chosen mode is recorded per-repo on the service
    /// ([`WorkflowService::set_repo_mode`]) so that push-triggered rescans
    /// retain strict rather than silently dropping back to legacy. This load
    /// also **reconciles**: stale service registrations and adapter handlers
    /// for this repo that are absent from the fresh set are evicted/dropped,
    /// so a deleted or strict-rejected workflow can no longer be dispatched
    /// or triggered (fail-closed).
    pub async fn load_repo_with(
        &mut self,
        repo_id: &str,
        service: &WorkflowService,
        mode: super::parser::ParseMode,
    ) -> Result<()> {
        // Record the mode so event-triggered rescans retain strict.
        service.set_repo_mode(repo_id, mode).await;

        // Scan the repository for workflow files in the chosen mode.
        let workflow_defs = service.scan_repo_with(repo_id, mode).await?;

        // Reconcile the service registry: evict any prior registration for
        // this repo that did not survive the fresh scan.
        let fresh_ids: std::collections::HashSet<WorkflowId> = workflow_defs
            .iter()
            .map(|def| format!("{}:{}", def.repo_id, def.path))
            .collect();
        service.evict_stale_for_repo(repo_id, &fresh_ids).await;

        // Reconcile this adapter's handlers: drop handlers whose workflow
        // belongs to this repo, then rebuild from the fresh set. Without
        // this, a strict-rejected/deleted workflow's handler would persist
        // and keep firing (the fail-closed gap).
        let repo_prefix = format!("{repo_id}:");
        self.handlers
            .retain(|h| !h.workflow_id().starts_with(&repo_prefix));

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
            mode = ?mode,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventSubscriber;
    use hyprstream_vfs::Subject;
    use std::path::PathBuf;

    const KNOWN_YAML: &str = r#"
name: Gate
on: push
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test
"#;

    const UNKNOWN_KEY_YAML: &str = r#"
name: Gate
on: push
permissions:
  contents: read
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - run: cargo test
"#;

    async fn write_workflow(
        root: &std::path::Path,
        yaml: &str,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = root.join(".github").join("workflows");
        tokio::fs::create_dir_all(&dir).await?;
        tokio::fs::write(dir.join("gate.yml"), yaml).await?;
        Ok(())
    }

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    fn adapter() -> std::result::Result<GitHubActionsAdapter, Box<dyn std::error::Error>> {
        Ok(GitHubActionsAdapter::new(EventSubscriber::new()?, Subject::new("gate-test")))
    }

    fn service() -> super::super::service::WorkflowService {
        use hyprstream_rpc::prelude::SigningKey;
        use hyprstream_rpc::transport::TransportConfig;
        use rand::rngs::OsRng;
        super::super::service::WorkflowService::new(
            TransportConfig::inproc("test-gh-adapter"),
            SigningKey::generate(&mut OsRng),
        )
    }

    /// P1-B (rev 4): a repeated strict load must reconcile — a workflow that
    /// is now strict-rejected must have BOTH its registration evicted AND its
    /// adapter handler dropped, so it can no longer be triggered.
    #[tokio::test]
    async fn strict_load_reconciles_stale_handler() -> TestResult {
        let tmp = tempfile::tempdir()?;
        write_workflow(tmp.path(), KNOWN_YAML).await?;

        let svc = service();
        svc.register_repo_path("acme", PathBuf::from(tmp.path())).await;

        let mut adp = adapter()?;
        adp.load_repo_with("acme", &svc, super::super::parser::ParseMode::Strict)
            .await?;
        // `on: push` builds a TopicPatternHandler → at least one handler.
        assert!(
            !adp.handlers.is_empty(),
            "known workflow should register at least one handler"
        );
        assert!(
            svc.list_workflows().await?.iter().any(|i| i.id
                == "acme:.github/workflows/gate.yml"),
            "workflow should be registered after first strict load"
        );

        // Mutate to add an unknown key and reload in strict mode.
        write_workflow(tmp.path(), UNKNOWN_KEY_YAML).await?;
        adp.load_repo_with("acme", &svc, super::super::parser::ParseMode::Strict)
            .await?;

        // Adapter handlers for this repo must be dropped (reconcile).
        let repo_prefix = "acme:";
        let remaining = adp
            .handlers
            .iter()
            .filter(|h| h.workflow_id().starts_with(repo_prefix))
            .count();
        assert_eq!(
            remaining, 0,
            "strict-rejected workflow's adapter handlers must be dropped on reconcile"
        );
        // And the registration must be evicted from the service.
        assert!(
            !svc.list_workflows().await?.iter().any(|i| i.id
                == "acme:.github/workflows/gate.yml"),
            "strict-rejected workflow's registration must be evicted on reconcile"
        );
        Ok(())
    }

    /// Legacy reload keeps the workflow (and its handler) — guards against the
    /// reconcile path accidentally evicting under the generic/legacy loader.
    #[tokio::test]
    async fn legacy_load_keeps_unknown_key_workflow() -> TestResult {
        let tmp = tempfile::tempdir()?;
        write_workflow(tmp.path(), UNKNOWN_KEY_YAML).await?;

        let svc = service();
        svc.register_repo_path("acme", PathBuf::from(tmp.path())).await;

        let mut adp = adapter()?;
        adp.load_repo("acme", &svc).await?; // legacy
        assert!(
            !adp.handlers.is_empty(),
            "legacy load keeps the unknown-key workflow's handlers"
        );
        assert!(
            svc.list_workflows().await?.iter().any(|i| i.id
                == "acme:.github/workflows/gate.yml"),
            "legacy load keeps the unknown-key workflow registered"
        );
        Ok(())
    }
}
