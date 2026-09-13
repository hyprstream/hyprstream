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
        move |subject: String,
              domain: String,
              resource: String,
              operation: String,
              bearer: Option<String>| {
            let client = policy_client.clone();
            Box::pin(async move {
                let upstream_subject =
                    hyprstream_rpc::envelope::Subject::new(subject.clone());
                let request = PolicyCheck {
                    subject,
                    domain,
                    resource,
                    operation,
                };
                crate::services::policy::check_with_verified_bearer(
                    &client,
                    &request,
                    bearer.as_deref(),
                    &upstream_subject,
                )
                .await
            })
        },
    )
}

// WorkerZmqClient / attach_container removed — use
// `hyprstream_rpc_std::worker_client::WorkerClient` directly (DH key exchange
// is encapsulated by the canonical client codegen).
