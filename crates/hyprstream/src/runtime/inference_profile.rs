//! Deployment and scheduling contract for inference-engine instances.
//!
//! Engine isolation is deliberately separate from model architecture. A
//! `TorchEngine` keeps one contract whether an operator places it in the host
//! process, a tenant subprocess, or an isolated task/microVM.

use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Address-space isolation selected for an inference engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InferenceIsolationProfile {
    /// Dedicated engine thread in the model-service process.
    ///
    /// Valid only for a single-tenant development deployment.
    InProcess,
    /// One operating-system process per authority-verified tenant.
    PerTenantSubprocess,
    /// One sandbox task or microVM per authority-verified tenant.
    PerTenantMicroVmTask {
        /// Explicit worker backend (for example `kata`). An unavailable backend
        /// must fail rather than downgrade.
        backend: String,
    },
}

/// Tenancy posture of the deployment hosting inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InferenceTenancy {
    /// One tenant, intended for development and trusted appliances.
    SingleTenantDevelopment,
    /// More than one mutually untrusted tenant may reach the deployment.
    SharedMultiTenant,
}

/// Accelerator class reserved for each engine instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum InferenceCompute {
    /// CPU-only execution. This is the initial demo profile.
    Cpu,
    /// A specific GPU is assigned by the scheduler.
    Gpu,
    /// Preserve legacy auto-detection in single-tenant development.
    Auto,
}

/// Per-instance resource and lifecycle limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceResourceBudget {
    /// CPU reservation in millicores.
    pub cpu_millis: u32,
    /// Resident-memory ceiling.
    pub memory_bytes: u64,
    /// VRAM ceiling. Must be zero for [`InferenceCompute::Cpu`].
    pub gpu_vram_bytes: u64,
    /// Maximum time allowed for model preparation.
    pub startup_timeout_ms: u64,
    /// Maximum time a health probe may take.
    pub health_check_timeout_ms: u64,
    /// Maximum time allowed for drain and teardown.
    pub teardown_timeout_ms: u64,
}

impl Default for InferenceResourceBudget {
    fn default() -> Self {
        Self {
            cpu_millis: 1_000,
            memory_bytes: 8 * 1024 * 1024 * 1024,
            gpu_vram_bytes: 0,
            startup_timeout_ms: 300_000,
            health_check_timeout_ms: 5_000,
            teardown_timeout_ms: 30_000,
        }
    }
}

/// Explicit inference deployment profile validated before scheduling.
///
/// The current in-process path consumes the compute selector. Isolation and
/// resource budgets are the contract for an external launcher; selecting one
/// of those profiles fails closed until that launcher is present.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceDeploymentProfile {
    pub isolation: InferenceIsolationProfile,
    pub tenancy: InferenceTenancy,
    pub compute: InferenceCompute,
    pub resources: InferenceResourceBudget,
}

impl Default for InferenceDeploymentProfile {
    fn default() -> Self {
        Self {
            isolation: InferenceIsolationProfile::InProcess,
            tenancy: InferenceTenancy::SingleTenantDevelopment,
            compute: InferenceCompute::Auto,
            resources: InferenceResourceBudget::default(),
        }
    }
}

impl InferenceDeploymentProfile {
    /// CPU-only profile used by the demo.
    #[must_use]
    pub fn cpu_demo(isolation: InferenceIsolationProfile, tenancy: InferenceTenancy) -> Self {
        Self {
            isolation,
            tenancy,
            compute: InferenceCompute::Cpu,
            resources: InferenceResourceBudget::default(),
        }
    }

    /// Load the production scheduling profile from explicit environment
    /// signals. Unknown values fail startup; they never fall back to weaker
    /// isolation or automatic accelerator selection.
    pub fn from_env() -> Result<Self> {
        let mut profile = Self::default();
        if let Ok(value) = std::env::var("HYPRSTREAM_INFERENCE_ISOLATION") {
            profile.isolation = match value.as_str() {
                "in-process" => InferenceIsolationProfile::InProcess,
                "per-tenant-subprocess" => InferenceIsolationProfile::PerTenantSubprocess,
                "per-tenant-microvm-task" => {
                    let backend =
                        std::env::var("HYPRSTREAM_INFERENCE_SANDBOX_BACKEND").map_err(|_| {
                            anyhow::anyhow!(
                                "HYPRSTREAM_INFERENCE_SANDBOX_BACKEND is required for \
                                 per-tenant-microvm-task"
                            )
                        })?;
                    InferenceIsolationProfile::PerTenantMicroVmTask { backend }
                }
                other => anyhow::bail!(
                    "HYPRSTREAM_INFERENCE_ISOLATION must be 'in-process', \
                     'per-tenant-subprocess', or 'per-tenant-microvm-task', got {other:?}"
                ),
            };
        }
        if let Ok(value) = std::env::var("HYPRSTREAM_INFERENCE_TENANCY") {
            profile.tenancy = match value.as_str() {
                "single-tenant-development" => InferenceTenancy::SingleTenantDevelopment,
                "shared-multi-tenant" => InferenceTenancy::SharedMultiTenant,
                other => anyhow::bail!(
                    "HYPRSTREAM_INFERENCE_TENANCY must be 'single-tenant-development' \
                     or 'shared-multi-tenant', got {other:?}"
                ),
            };
        }
        if let Ok(value) = std::env::var("HYPRSTREAM_INFERENCE_COMPUTE") {
            profile.compute = match value.as_str() {
                "cpu" => InferenceCompute::Cpu,
                "gpu" => InferenceCompute::Gpu,
                "auto" => InferenceCompute::Auto,
                other => anyhow::bail!(
                    "HYPRSTREAM_INFERENCE_COMPUTE must be 'cpu', 'gpu', or 'auto', got {other:?}"
                ),
            };
        }
        apply_u32_env(
            "HYPRSTREAM_INFERENCE_CPU_MILLIS",
            &mut profile.resources.cpu_millis,
        )?;
        apply_u64_env(
            "HYPRSTREAM_INFERENCE_MEMORY_BYTES",
            &mut profile.resources.memory_bytes,
        )?;
        apply_u64_env(
            "HYPRSTREAM_INFERENCE_GPU_VRAM_BYTES",
            &mut profile.resources.gpu_vram_bytes,
        )?;
        apply_u64_env(
            "HYPRSTREAM_INFERENCE_STARTUP_TIMEOUT_MS",
            &mut profile.resources.startup_timeout_ms,
        )?;
        apply_u64_env(
            "HYPRSTREAM_INFERENCE_HEALTH_TIMEOUT_MS",
            &mut profile.resources.health_check_timeout_ms,
        )?;
        apply_u64_env(
            "HYPRSTREAM_INFERENCE_TEARDOWN_TIMEOUT_MS",
            &mut profile.resources.teardown_timeout_ms,
        )?;
        profile.validate()?;
        Ok(profile)
    }

    /// Validate the deployment contract before an engine is loaded.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.resources.cpu_millis > 0,
            "inference CPU budget must be non-zero"
        );
        ensure!(
            self.resources.memory_bytes > 0,
            "inference memory budget must be non-zero"
        );
        ensure!(
            self.resources.startup_timeout_ms > 0
                && self.resources.health_check_timeout_ms > 0
                && self.resources.teardown_timeout_ms > 0,
            "inference lifecycle timeouts must be non-zero"
        );
        if self.compute == InferenceCompute::Cpu {
            ensure!(
                self.resources.gpu_vram_bytes == 0,
                "CPU inference profile cannot reserve GPU VRAM"
            );
        }
        if self.compute == InferenceCompute::Gpu {
            ensure!(
                self.resources.gpu_vram_bytes > 0,
                "GPU inference profile requires an explicit VRAM budget"
            );
        }
        ensure!(
            !(self.tenancy == InferenceTenancy::SharedMultiTenant
                && self.isolation == InferenceIsolationProfile::InProcess),
            "shared multi-tenant inference requires per-tenant subprocess or microVM/task isolation"
        );
        if let InferenceIsolationProfile::PerTenantMicroVmTask { backend } = &self.isolation {
            ensure!(
                !backend.trim().is_empty() && backend != "auto",
                "microVM/task inference requires an explicit sandbox backend"
            );
        }
        Ok(())
    }

    /// Whether this profile needs an out-of-process launcher.
    #[must_use]
    pub fn requires_isolated_launcher(&self) -> bool {
        !matches!(self.isolation, InferenceIsolationProfile::InProcess)
    }
}

fn apply_u32_env(name: &str, target: &mut u32) -> Result<()> {
    if let Ok(value) = std::env::var(name) {
        *target = value.parse().map_err(|e| {
            anyhow::anyhow!("{name} must be an unsigned integer, got {value:?}: {e}")
        })?;
    }
    Ok(())
}

fn apply_u64_env(name: &str, target: &mut u64) -> Result<()> {
    if let Ok(value) = std::env::var(name) {
        *target = value.parse().map_err(|e| {
            anyhow::anyhow!("{name} must be an unsigned integer, got {value:?}: {e}")
        })?;
    }
    Ok(())
}

/// Tenant/model/replica identity used for placement and endpoint derivation.
///
/// The tenant is accepted only after it came from `EnvelopeContext::domain()`.
/// It is never derived from subject text or caller-supplied model input.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InferenceInstanceId {
    tenant: String,
    model_ref: String,
    replica: u16,
}

/// Domain-separated carrier-key purpose for one opaque service instance.
pub(crate) fn inference_iroh_key_purpose(service_name: &str) -> String {
    format!("hyprstream-inference-iroh-v1/{service_name}")
}

impl InferenceInstanceId {
    pub fn new(verified_tenant: &str, model_ref: &str, replica: u16) -> Result<Self> {
        let tenant = verified_tenant.trim();
        ensure!(
            !tenant.is_empty() && tenant != "*",
            "inference instance requires a non-empty, non-wildcard verified tenant"
        );
        ensure!(
            !model_ref.trim().is_empty(),
            "inference instance requires a model ref"
        );
        Ok(Self {
            tenant: tenant.to_owned(),
            model_ref: model_ref.to_owned(),
            replica,
        })
    }

    #[must_use]
    pub fn tenant(&self) -> &str {
        &self.tenant
    }

    #[must_use]
    pub fn model_ref(&self) -> &str {
        &self.model_ref
    }

    #[must_use]
    pub fn replica(&self) -> u16 {
        self.replica
    }

    /// Opaque, deterministic name safe for registries and process managers.
    #[must_use]
    pub fn service_name(&self) -> String {
        let mut digest = Sha256::new();
        digest.update(b"hyprstream-inference-instance-v1\0");
        digest.update(self.tenant.as_bytes());
        digest.update(b"\0");
        digest.update(self.model_ref.as_bytes());
        digest.update(b"\0");
        digest.update(self.replica.to_be_bytes());
        let suffix = hex::encode(&digest.finalize()[..12]);
        format!("inference-{suffix}")
    }
    #[must_use]
    pub fn iroh_key_purpose(&self) -> String {
        inference_iroh_key_purpose(&self.service_name())
    }

    /// Same-host cross-process dial target for a subprocess placement.
    #[must_use]
    pub fn ipc_transport(&self, runtime_dir: &Path) -> hyprstream_rpc::transport::TransportConfig {
        let path: PathBuf = runtime_dir.join(format!("{}.sock", self.service_name()));
        hyprstream_rpc::transport::TransportConfig::ipc(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_tenancy_rejects_in_process_profile() {
        let profile = InferenceDeploymentProfile::cpu_demo(
            InferenceIsolationProfile::InProcess,
            InferenceTenancy::SharedMultiTenant,
        );
        assert!(profile.validate().is_err());
    }

    #[test]
    fn shared_cpu_subprocess_profile_is_valid() {
        let profile = InferenceDeploymentProfile::cpu_demo(
            InferenceIsolationProfile::PerTenantSubprocess,
            InferenceTenancy::SharedMultiTenant,
        );
        assert!(profile.validate().is_ok());
        assert!(profile.requires_isolated_launcher());
    }

    #[test]
    fn two_tenants_get_distinct_opaque_dial_targets() -> Result<()> {
        let first = InferenceInstanceId::new("tenant-a", "tiny-llama:main", 0)?;
        let second = InferenceInstanceId::new("tenant-b", "tiny-llama:main", 0)?;
        assert_ne!(first.service_name(), second.service_name());
        assert_ne!(
            first.ipc_transport(Path::new("/run/hyprstream")),
            second.ipc_transport(Path::new("/run/hyprstream"))
        );
        assert_ne!(first.iroh_key_purpose(), second.iroh_key_purpose());
        assert!(!first.service_name().contains("tenant-a"));
        Ok(())
    }

    #[test]
    fn wildcard_or_blank_tenant_fails_closed() {
        assert!(InferenceInstanceId::new("*", "model:main", 0).is_err());
        assert!(InferenceInstanceId::new(" ", "model:main", 0).is_err());
    }

    #[test]
    fn replica_identity_changes_dial_target() -> Result<()> {
        let first = InferenceInstanceId::new("tenant-a", "tiny-llama:main", 0)?;
        let second = InferenceInstanceId::new("tenant-a", "tiny-llama:main", 1)?;
        assert_ne!(first.service_name(), second.service_name());
        Ok(())
    }

    #[test]
    fn two_cpu_models_get_distinct_service_addresses() -> Result<()> {
        let first = InferenceInstanceId::new("tenant-a", "model-one:main", 0)?;
        let second = InferenceInstanceId::new("tenant-a", "model-two:main", 0)?;
        assert_ne!(first.service_name(), second.service_name());
        assert_ne!(
            first.ipc_transport(Path::new("/run/hyprstream")),
            second.ipc_transport(Path::new("/run/hyprstream"))
        );
        let service_key =
            hyprstream_rpc::prelude::SigningKey::from_bytes(&[31u8; 32]);
        let first_carrier = hyprstream_rpc::node_identity::derive_purpose_key(
            &service_key,
            &first.iroh_key_purpose(),
        );
        let second_carrier = hyprstream_rpc::node_identity::derive_purpose_key(
            &service_key,
            &second.iroh_key_purpose(),
        );
        assert_ne!(
            first_carrier.verifying_key(),
            service_key.verifying_key(),
            "carrier address must not reuse application authority"
        );
        assert_ne!(first_carrier.verifying_key(), second_carrier.verifying_key());
        Ok(())
    }
}
