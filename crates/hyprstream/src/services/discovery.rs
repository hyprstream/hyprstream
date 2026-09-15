//! Discovery service — re-exports from `hyprstream-discovery`.
//!
//! The DiscoveryService implementation has moved to the `hyprstream-discovery` crate.
//! This module provides a `PolicyAuthProvider` that wraps `PolicyClient` to implement
//! the `AuthorizationProvider` trait expected by DiscoveryService.

pub use hyprstream_discovery::{AuthorizationProvider, DiscoveryService};

use hyprstream_rpc_std::policy_client::PolicyCheck;
use async_trait::async_trait;



/// Policy-based authorization provider wrapping PolicyClient.
///
/// Bridges the `AuthorizationProvider` trait from `hyprstream-discovery`
/// to the `PolicyClient` generated in this crate.
pub struct PolicyAuthProvider {
    client: hyprstream_rpc_std::policy_client::PolicyClient,
}

impl PolicyAuthProvider {
    /// Create a new policy-based authorization provider.
    pub fn new(client: hyprstream_rpc_std::policy_client::PolicyClient) -> Self {
        Self { client }
    }
}

#[async_trait(?Send)]
impl AuthorizationProvider for PolicyAuthProvider {
    async fn check_batch(
        &self, subject: &str, domain: &str, resources: &[String],
        operation: &str, bearer: Option<&str>,
    ) -> anyhow::Result<Vec<bool>> {
        use hyprstream_rpc_std::policy_client::PolicyCheckBatch;
        anyhow::ensure!(resources.len() <= 256, "authorization batch exceeds 256");
        let client = match bearer {
            Some(token) => self.client.clone().with_delegated_bearer(token.to_owned()),
            None => {
                let upstream = hyprstream_rpc::envelope::Subject::new(subject);
                anyhow::ensure!(!upstream.is_federated() && upstream.name()
                    .is_some_and(|name| name == "system" || name.starts_with("service:")),
                    "service-mediated user policy check requires verified bearer");
                self.client.clone()
            }
        };
        let request = PolicyCheckBatch { checks: resources.iter().map(|resource| PolicyCheck {
            subject: subject.to_owned(), domain: domain.to_owned(),
            resource: resource.clone(), operation: operation.to_owned(),
        }).collect() };
        let result = client.check_batch(&request).await?;
        anyhow::ensure!(result.allowed.len() == resources.len(), "invalid policy batch decision count");
        Ok(result.allowed)
    }

    async fn check(
        &self,
        subject: &str,
        domain: &str,
        resource: &str,
        operation: &str,
        bearer: Option<&str>,
    ) -> anyhow::Result<bool> {
        let request = PolicyCheck {
            subject: subject.to_owned(),
            domain: domain.to_owned(),
            resource: resource.to_owned(),
            operation: operation.to_owned(),
        };
        crate::services::policy::check_with_verified_bearer(
            &self.client,
            &request,
            bearer,
            &hyprstream_rpc::envelope::Subject::new(subject),
        )
        .await
    }
}

// The durable PDS record store and its publisher/resolver/ingest stack only
// compile when the `rocksdb` feature carries the dependency. Everything is
// re-exported unchanged, so callers keep the `services::discovery::*` paths.
#[cfg(feature = "rocksdb")]
pub use crate::services::pds_record_rocksdb::*;
