//! Subscriber adapter pattern for bridging events to workflow dispatch.
//!
//! A `SubscriberAdapter` watches events and dispatches workflows. The
//! `WorkflowService` is a pure execution engine; adapters provide the
//! event-to-workflow routing logic.
//!
//! GitHub Actions is one adapter. Others (Tekton, Argo, raw capnp RPC)
//! could follow the same pattern.

use std::sync::Arc;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::error::Result;
use super::service::WorkflowService;

/// A subscriber adapter watches events and dispatches workflows.
///
/// Each adapter owns its event subscription and handler logic.
/// The `WorkflowService` provides only execution — adapters decide
/// which workflows to dispatch in response to which events.
#[async_trait]
pub trait SubscriberAdapter: Send + Sync {
    /// Human-readable name for logging.
    fn name(&self) -> &str;

    /// Start watching events. Runs until cancelled.
    ///
    /// The adapter should:
    /// 1. Subscribe to relevant event topics
    /// 2. Loop: receive event → match handlers → dispatch workflows
    /// 3. Exit cleanly when `cancel` is triggered
    async fn run(
        &self,
        service: Arc<WorkflowService>,
        cancel: CancellationToken,
    ) -> Result<()>;
}
