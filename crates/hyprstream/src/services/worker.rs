//! Worker service types and helpers.
//!
//! Worker-service authorization glue. Public worker RPC contracts and clients
//! are owned by `hyprstream-rpc-std`; this module contains no client facade.

use hyprstream_rpc::service::AuthorizeFn;
use std::sync::Arc;
use hyprstream_rpc_std::policy_client::PolicyCheck;

// ============================================================================
// Authorization Helper
// ============================================================================

use hyprstream_rpc_std::policy_client::PolicyClient;

/// Build an `AuthorizeFn` backed by a `PolicyClient`.
///
/// The returned closure is async-compatible (returns a boxed future) so it
/// works on single-threaded runtimes used by RequestService.
pub fn build_authorize_fn(policy_client: PolicyClient) -> AuthorizeFn {
    Arc::new(
        move |ctx: hyprstream_rpc::service::EnvelopeContext,
              resource: String,
              operation: String| {
            let client = policy_client.clone();
            Box::pin(async move {
                let upstream_subject = ctx.subject();
                let request = PolicyCheck {
                    subject: upstream_subject.to_string(),
                    // Compatibility fields only; Policy derives the domain
                    // from independently verified caller evidence.
                    domain: String::new(),
                    resource,
                    operation,
                };
                crate::services::policy::check_with_holder_evidence(
                    &client,
                    &request,
                    ctx.jwt_token(),
                    &upstream_subject,
                    ctx.original_holder_evidence(),
                )
                .await
            })
        },
    )
}

// WorkerZmqClient / attach_container removed — use
// `hyprstream_rpc_std::worker_client::WorkerClient` directly (DH key exchange
// is encapsulated by the canonical client codegen).
