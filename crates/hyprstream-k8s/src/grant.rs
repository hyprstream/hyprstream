//! Service-neutral tenant-grant boundary for the Kubernetes operator.
//!
//! This Apache-licensed module owns only the generic reconciliation contract:
//! an AGPL service adapter may compile an authored [`TenantBinding`]
//! entitlement and return opaque content-addressed references. Cryptographic
//! issuer keys, signed grant verification, PDS allocation records, and their
//! CID implementation belong downstream of this crate.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use kube::Resource;

use crate::mesh::{TenantBinding, TenantBindingStatus};

/// Opaque references produced by a service-specific grant compiler.
///
/// The Kubernetes substrate records these values as observed status but never
/// interprets their service-specific contents.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TenantGrantArtifacts {
    pub grant_cid: String,
    pub allocation_cid: String,
    pub epoch: u64,
}

/// A fail-closed service error safe to reflect in Kubernetes status.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TenantGrantServiceError {
    message: String,
}

impl TenantGrantServiceError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for TenantGrantServiceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for TenantGrantServiceError {}

/// Service behavior invoked by the generic `TenantBinding` reconciler.
///
/// Implementations are authority-bearing and therefore live in downstream
/// service crates. They must validate the binding, issue/sign the grant, bind
/// it to the allocation record, and fail closed on any error. The Apache
/// operator receives only the resulting opaque content references.
pub trait TenantGrantService: Send + Sync {
    fn compile_grant(
        &self,
        binding: &TenantBinding,
        epoch: u64,
        now: u64,
    ) -> Result<TenantGrantArtifacts, TenantGrantServiceError>;
}

/// Project a service result into the status written by the generic operator.
///
/// An authored entitlement without a configured service is rejected. A
/// service error is also rejected and never leaves partial content references.
pub fn compile_tenant_binding_status(
    binding: &TenantBinding,
    service: Option<&dyn TenantGrantService>,
    epoch: u64,
    now: u64,
) -> TenantBindingStatus {
    let generation = binding.meta().generation;
    match binding.spec.entitlement.as_ref() {
        None => TenantBindingStatus {
            bound: Some(true),
            phase: Some("Bound".to_owned()),
            message: Some("namespace↔tenant mapping active; no entitlement to compile".to_owned()),
            observed_generation: generation,
            grant_cid: None,
            allocation_cid: None,
            epoch: None,
        },
        Some(_) => match service {
            None => rejected_status(
                generation,
                "entitlement present but operator has no tenant-grant service configured",
            ),
            Some(service) => match service.compile_grant(binding, epoch, now) {
                Ok(compiled) if compiled.epoch != epoch => rejected_status(
                    generation,
                    &format!(
                        "grant compilation failed: service returned epoch {} for requested epoch {epoch}",
                        compiled.epoch
                    ),
                ),
                Ok(compiled) => TenantBindingStatus {
                    bound: Some(true),
                    phase: Some("Bound".to_owned()),
                    message: Some(format!(
                        "compiled grant {} (allocation {}); PDS publish pending #910",
                        compiled.grant_cid, compiled.allocation_cid
                    )),
                    observed_generation: generation,
                    grant_cid: Some(compiled.grant_cid),
                    allocation_cid: Some(compiled.allocation_cid),
                    epoch: Some(compiled.epoch),
                },
                Err(error) => {
                    rejected_status(generation, &format!("grant compilation failed: {error}"))
                }
            },
        },
    }
}

fn rejected_status(generation: Option<i64>, message: &str) -> TenantBindingStatus {
    TenantBindingStatus {
        bound: Some(false),
        phase: Some("Rejected".to_owned()),
        message: Some(message.to_owned()),
        observed_generation: generation,
        grant_cid: None,
        allocation_cid: None,
        epoch: None,
    }
}

/// Unix seconds used by the generic reconcile request.
///
/// Platforms without a usable wall clock return zero so downstream expiry
/// validation fails closed.
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;
    use crate::mesh::{TenantBindingSpec, TenantEntitlement, TenantGrantClass};

    struct RecordingService;

    impl TenantGrantService for RecordingService {
        fn compile_grant(
            &self,
            binding: &TenantBinding,
            epoch: u64,
            now: u64,
        ) -> Result<TenantGrantArtifacts, TenantGrantServiceError> {
            assert_eq!(binding.spec.tenant, "did:web:acme");
            assert_eq!(epoch, 7);
            assert_eq!(now, 1_700_000_000);
            Ok(TenantGrantArtifacts {
                grant_cid: "bafy-grant".to_owned(),
                allocation_cid: "bafy-allocation".to_owned(),
                epoch,
            })
        }
    }

    fn entitled_binding() -> TenantBinding {
        TenantBinding::new(
            "acme",
            TenantBindingSpec {
                namespace: "acme".to_owned(),
                tenant: "did:web:acme".to_owned(),
                entitlement: Some(TenantEntitlement {
                    unit: "compute-second".to_owned(),
                    amount: 100,
                    class: TenantGrantClass::Underwritten,
                    expiration: None,
                }),
            },
        )
    }

    #[test]
    fn tenant_grant_service_boundary_is_required_for_entitlements() {
        let rejected = compile_tenant_binding_status(&entitled_binding(), None, 7, 1_700_000_000);
        assert_eq!(rejected.bound, Some(false));
        assert_eq!(rejected.phase.as_deref(), Some("Rejected"));
        assert!(rejected.grant_cid.is_none());
        assert!(rejected.allocation_cid.is_none());

        let compiled = compile_tenant_binding_status(
            &entitled_binding(),
            Some(&RecordingService),
            7,
            1_700_000_000,
        );
        assert_eq!(compiled.bound, Some(true));
        assert_eq!(compiled.grant_cid.as_deref(), Some("bafy-grant"));
        assert_eq!(compiled.allocation_cid.as_deref(), Some("bafy-allocation"));
        assert_eq!(compiled.epoch, Some(7));
    }

    struct FailingService;

    impl TenantGrantService for FailingService {
        fn compile_grant(
            &self,
            _binding: &TenantBinding,
            _epoch: u64,
            _now: u64,
        ) -> Result<TenantGrantArtifacts, TenantGrantServiceError> {
            Err(TenantGrantServiceError::new("issuer unavailable"))
        }
    }

    #[test]
    fn tenant_grant_service_errors_fail_closed_without_partial_references() {
        let status = compile_tenant_binding_status(
            &entitled_binding(),
            Some(&FailingService),
            7,
            1_700_000_000,
        );
        assert_eq!(status.bound, Some(false));
        assert_eq!(status.phase.as_deref(), Some("Rejected"));
        assert!(status
            .message
            .as_deref()
            .is_some_and(|message| message.contains("issuer unavailable")));
        assert!(status.grant_cid.is_none());
        assert!(status.allocation_cid.is_none());
        assert!(status.epoch.is_none());
    }

    struct WrongEpochService;

    impl TenantGrantService for WrongEpochService {
        fn compile_grant(
            &self,
            _binding: &TenantBinding,
            epoch: u64,
            _now: u64,
        ) -> Result<TenantGrantArtifacts, TenantGrantServiceError> {
            Ok(TenantGrantArtifacts {
                grant_cid: "bafy-grant".to_owned(),
                allocation_cid: "bafy-allocation".to_owned(),
                epoch: epoch + 1,
            })
        }
    }

    #[test]
    fn tenant_grant_service_cannot_bypass_requested_revocation_epoch() {
        let status = compile_tenant_binding_status(
            &entitled_binding(),
            Some(&WrongEpochService),
            7,
            1_700_000_000,
        );
        assert_eq!(status.bound, Some(false));
        assert_eq!(status.phase.as_deref(), Some("Rejected"));
        assert!(status.grant_cid.is_none());
        assert!(status.allocation_cid.is_none());
        assert!(status.epoch.is_none());
    }
}
