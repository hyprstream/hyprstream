//! Job → scheduled workload placement (#527).
//!
//! Maps a job's `runs_on:` label + `resources:` hints onto a placement
//! decision and, for jobs that request isolation, acquires/releases a real
//! sandbox reservation through the #525 P2 admission engine
//! (`SandboxPool::acquire`) instead of the pre-#527 fire-and-forget
//! `tokio::spawn` that ignored both.
//!
//! ## 🔒 Mapping convention (ratified: B′ group semantics, #921 decision 5)
//!
//! `runs_on: <label>` is matched, case-insensitively, against a small
//! reserved alias set ([`IN_PROC_LABELS`]) meaning "no isolation — run
//! in-process" (today's VFS/Tcl execution path, unchanged). Any other label
//! routes the job onto the [`JobScheduler`]'s configured [`SandboxPool`] and
//! is recorded as the `hyprstream_workers::runtime::ANN_GROUP` annotation.
//!
//! Under the ratified B′ semantics that annotation is a **capacity-partition
//! selector, never an authoritative quota/trust/billing key**: admission
//! (`runtime::admission`) resolves it against the verified Subject's
//! bidirectional-consent placement-group membership via the pool's
//! `GroupSelectorValidator` — a workflow naming a group its Subject is not a
//! member of is **rejected at admission**, not silently accepted. The
//! production default validator (`DenyUnknownGroupValidator`) fail-closes on
//! every non-empty selector until a membership source
//! (`PlacementIndex::is_member`) is wired, so an isolated `runs_on:` label is
//! deny-by-default today. Validated labels partition node-local capacity via
//! `AdmissionConfig::max_per_group` (fairness/organization only —
//! authoritative quota arrives via the ledger path, #922/#925). This module
//! adds **no group accounting of its own**: the label's only consumer is the
//! `ANN_GROUP` annotation read back by `admission::derive_group_selector`
//! inside `SandboxPool::acquire`.
//!
//! This is a **label → group** mapping, not a **label → backend** mapping:
//! one `JobScheduler` wraps exactly one pre-resolved [`SandboxPool`] (bound
//! to one [`crate::runtime::SandboxBackend`] at construction — see
//! `pool.rs`'s own documented invariant that a pool never re-selects its
//! backend per-request). `runs_on: kata` therefore does NOT pick a different
//! backend than `runs_on: nspawn` within a single scheduler; both go to
//! *this scheduler's* one backend, merely tagged with a different admission
//! group. Routing different labels to genuinely different backends would
//! mean a router over multiple backend-specific pools — the same
//! architectural change `pool.rs` already flags as a follow-up, not
//! something this ticket re-derives. Flagged here explicitly for reviewer
//! sign-off, per the issue's disclosure ask.
//!
//! ## What "runs in a sandbox" means today (also flagged)
//!
//! Placement acquires a *real* sandbox reservation — admission is checked,
//! the backend's `start()` actually runs (a real Kata VM boots when the pool
//! is Kata-backed) — closing the #520 "no isolation" gap structurally: a
//! job that requests isolation now really holds a sandbox for its duration.
//! Step *content* execution (the `uses:`/`run:` bodies) still runs through
//! the existing VFS `/bin/` ctl + TclShell bridge on the host, not inside the
//! guest's own namespace — routing individual step exec into the guest (via
//! `kata_agent::ExecProcess`, `nspawn`, etc.) needs the streaming/fd plane
//! `exec_mount.rs` itself documents as not yet built. That is a materially
//! larger follow-up, not attempted here.

use std::sync::Arc;

use hyprstream_vfs::Subject;

use crate::error::Result;
use crate::runtime::{
    KeyValue, PodSandboxConfig, PodSandboxMetadata, SandboxPool, ANN_GPU_REQUEST, ANN_GROUP,
};

use super::parser::{JobResources, RunsOn};

/// `runs_on:` labels (case-insensitive) meaning "no isolation — run
/// in-process" via the existing VFS/Tcl execution path. Anything else is
/// treated as a request for this scheduler's sandbox (see module docs).
pub const IN_PROC_LABELS: &[&str] = &["local", "self-hosted", "in-process", "tcl"];

/// The label that drives a job's placement decision.
///
/// GHA's `runs-on: [a, b]` means "match ALL labels" against a runner's
/// capability set. Hyprstream has no such per-runner capability set to
/// intersect against (yet) — `runs_on` here selects an admission *group*
/// (see module docs), a mapping for which multi-label AND-composition has no
/// defined meaning. Only the first label is consulted; additional labels in
/// `RunsOn::Labels` are accepted (not a parse error) but otherwise ignored.
/// This is a documented scope-limit, not a silent drop.
pub fn primary_label(runs_on: &RunsOn) -> &str {
    match runs_on {
        RunsOn::Label(l) => l.as_str(),
        RunsOn::Labels(v) => v.first().map(String::as_str).unwrap_or(""),
    }
}

/// True when `runs_on`'s primary label requests in-process (no sandbox)
/// execution — either explicitly (an [`IN_PROC_LABELS`] alias) or because no
/// label was usefully provided (empty string).
pub fn wants_in_proc(runs_on: &RunsOn) -> bool {
    let label = primary_label(runs_on);
    label.is_empty() || IN_PROC_LABELS.iter().any(|l| l.eq_ignore_ascii_case(label))
}

/// Build the `PodSandboxConfig` a job's `resources:` hints + `runs_on` label
/// translate to for admission (#525 P2): cpu/memory become
/// `linux.resources`, `gpu` becomes the `ANN_GPU_REQUEST` annotation (the
/// same vocabulary `runtime::admission::derive_demand` already reads), and
/// the `runs_on` label becomes the `ANN_GROUP` annotation — a B′ group
/// *selector*, validated against the verified Subject's membership by the
/// pool's `GroupSelectorValidator` at admission (see module docs).
pub fn job_pod_sandbox_config(
    job_name: &str,
    runs_on_label: &str,
    resources: &JobResources,
) -> PodSandboxConfig {
    let mut config = PodSandboxConfig {
        metadata: PodSandboxMetadata {
            name: format!("workflow-job-{job_name}"),
            ..Default::default()
        },
        ..Default::default()
    };

    if resources.cpu_millis > 0 {
        // CRI convention: a 100ms period is the common default; quota scales
        // with it so `(quota / period) * 1000` round-trips through
        // `derive_demand` back to `cpu_millis`.
        config.linux.resources.cpu_period = 100_000;
        config.linux.resources.cpu_quota = (resources.cpu_millis as i64) * 100;
    }
    if resources.memory_bytes > 0 {
        config.linux.resources.memory_limit_in_bytes = resources.memory_bytes as i64;
    }
    if resources.gpu > 0 {
        config.annotations.push(KeyValue {
            key: ANN_GPU_REQUEST.to_owned(),
            value: resources.gpu.to_string(),
        });
    }
    config.annotations.push(KeyValue {
        key: ANN_GROUP.to_owned(),
        value: runs_on_label.to_owned(),
    });

    config
}

/// A job's resolved placement. `Sandbox` must be released via
/// [`Placement::release`] once the job finishes (success, failure, timeout,
/// or cancellation) so the admission reservation and warm-pool slot are
/// given back — see `SandboxPool::release`'s own reservation-rollback
/// discipline.
pub enum Placement {
    /// No isolation — the job's steps run on the existing host VFS/Tcl path.
    InProc,
    /// A live sandbox reservation acquired from `pool`.
    Sandbox {
        pool: Arc<SandboxPool>,
        sandbox_id: String,
    },
}

impl Placement {
    /// Release a `Sandbox` placement back to its pool. A no-op for `InProc`.
    /// Errors are logged, not propagated: by the time a job finishes, its
    /// result is already decided, and swallowing a release failure here
    /// matches `SandboxPool::release`'s own callers elsewhere (best-effort
    /// cleanup, never fails the job it releases after).
    pub async fn release(self) {
        if let Placement::Sandbox { pool, sandbox_id } = self {
            if let Err(e) = pool.release(&sandbox_id).await {
                tracing::warn!(
                    sandbox_id = %sandbox_id,
                    error = %e,
                    "failed to release workflow job sandbox"
                );
            }
        }
    }
}

/// Routes workflow jobs through the #525 P2 scheduler.
///
/// See the module docs for the `runs_on` → group mapping convention and its
/// documented scope limits.
pub struct JobScheduler {
    pool: Arc<SandboxPool>,
}

impl JobScheduler {
    /// Wrap an already-constructed [`SandboxPool`] (bound to one backend,
    /// e.g. via `runtime::resolve_backend` — the same #516 fail-closed
    /// registry `worker.backend` uses) as the target for isolated jobs.
    pub fn new(pool: Arc<SandboxPool>) -> Self {
        Self { pool }
    }

    /// Decide placement for a job and, if it requests isolation, acquire a
    /// sandbox reservation for it (admitted via `subject`'s identity/quota —
    /// fail-closed if `subject` is anonymous, see `runtime::admission`).
    ///
    /// The `runs_on` label rides along as a B′ group selector; admission
    /// rejects it unless the pool's `GroupSelectorValidator` proves `subject`
    /// a consented member of that group (deny-unknown by default).
    pub async fn place(
        &self,
        job_name: &str,
        runs_on: &RunsOn,
        resources: &JobResources,
        subject: &Subject,
    ) -> Result<Placement> {
        if wants_in_proc(runs_on) {
            return Ok(Placement::InProc);
        }
        let label = primary_label(runs_on);
        let config = job_pod_sandbox_config(job_name, label, resources);
        let sandbox_id = self.pool.acquire(subject, &config).await?;
        Ok(Placement::Sandbox {
            pool: Arc::clone(&self.pool),
            sandbox_id,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::config::PoolConfig;
    use crate::runtime::{
        AdmissionConfig, LinuxContainerResources, PodSandbox, SandboxBackend, SandboxHandle,
        StaticGroupMembership,
    };
    use async_trait::async_trait;
    use std::any::Any;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn primary_label_picks_first_of_list() {
        assert_eq!(primary_label(&RunsOn::Label("kata".into())), "kata");
        assert_eq!(
            primary_label(&RunsOn::Labels(vec!["gpu".into(), "linux".into()])),
            "gpu"
        );
        assert_eq!(primary_label(&RunsOn::Labels(vec![])), "");
    }

    #[test]
    fn in_proc_labels_are_case_insensitive() {
        assert!(wants_in_proc(&RunsOn::Label("LOCAL".into())));
        assert!(wants_in_proc(&RunsOn::Label("Self-Hosted".into())));
        assert!(!wants_in_proc(&RunsOn::Label("kata".into())));
        assert!(!wants_in_proc(&RunsOn::Label("ubuntu-latest".into())));
    }

    /// Read back an annotation by key (mirrors
    /// `runtime::admission::annotation`, which is private to that module).
    fn find_annotation<'a>(config: &'a PodSandboxConfig, key: &str) -> Option<&'a str> {
        config
            .annotations
            .iter()
            .find(|kv| kv.key == key)
            .map(|kv| kv.value.as_str())
    }

    #[test]
    fn job_pod_sandbox_config_maps_resources_and_group() {
        let resources = JobResources {
            cpu_millis: 500,
            memory_bytes: 1024,
            gpu: 2,
        };
        let config = job_pod_sandbox_config("build", "gpu-workers", &resources);

        // CRI convention: (quota / period) * 1000 == requested cpu_millis
        // (see `runtime::admission::derive_demand`, which reads it back the
        // same way from a real acquire() call).
        assert_eq!(config.linux.resources.cpu_period, 100_000);
        assert_eq!(config.linux.resources.cpu_quota, 50_000);
        assert_eq!(config.linux.resources.memory_limit_in_bytes, 1024);
        assert_eq!(find_annotation(&config, ANN_GPU_REQUEST), Some("2"));
        assert_eq!(find_annotation(&config, ANN_GROUP), Some("gpu-workers"));
    }

    #[test]
    fn job_pod_sandbox_config_no_resources_is_zero_demand() {
        let config = job_pod_sandbox_config("noop", "kata", &JobResources::default());
        assert_eq!(config.linux.resources.cpu_period, 0);
        assert_eq!(config.linux.resources.cpu_quota, 0);
        assert_eq!(config.linux.resources.memory_limit_in_bytes, 0);
        assert_eq!(find_annotation(&config, ANN_GPU_REQUEST), None);
        // Group is still set even with no resource demand — `runs_on`
        // influences placement (quota) independent of resource requests.
        assert_eq!(find_annotation(&config, ANN_GROUP), Some("kata"));
    }

    #[derive(Debug)]
    struct FakeHandle;
    impl SandboxHandle for FakeHandle {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    /// Minimal in-memory fake backend so scheduler tests don't need a real
    /// Kata/nspawn toolchain. Tracks how many times `start()` was actually
    /// invoked, so tests can prove an isolated job really acquires a sandbox
    /// (#520) and an in-proc job never touches the backend at all.
    #[derive(Default)]
    struct FakeBackend {
        starts: AtomicUsize,
    }

    #[async_trait]
    impl SandboxBackend for FakeBackend {
        fn backend_type(&self) -> &'static str {
            "fake"
        }
        fn is_available(&self) -> bool {
            true
        }
        async fn initialize(&self, _config: &PoolConfig) -> Result<()> {
            Ok(())
        }
        async fn start(
            &self,
            _sandbox: &mut PodSandbox,
            _config: &PodSandboxConfig,
            _pool_config: &PoolConfig,
            _annotations: &HashMap<String, String>,
        ) -> Result<Arc<dyn SandboxHandle>> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(FakeHandle))
        }
        async fn stop(&self, _sandbox: &PodSandbox) -> Result<()> {
            Ok(())
        }
        async fn destroy(&self, _sandbox: &PodSandbox) -> Result<()> {
            Ok(())
        }
        async fn reset(&self, _sandbox: &mut PodSandbox) -> Result<bool> {
            Ok(true)
        }
        async fn get_pids(&self, _sandbox: &PodSandbox) -> Result<Vec<u32>> {
            Ok(vec![])
        }
        fn supports_exec(&self) -> bool {
            false
        }
        async fn exec_sync(
            &self,
            _sandbox: &PodSandbox,
            _command: &[String],
            _timeout_secs: u64,
        ) -> Result<(i32, Vec<u8>, Vec<u8>)> {
            Err(crate::error::WorkerError::ExecFailed(
                "not supported".into(),
            ))
        }
        async fn update_resources(
            &self,
            _sandbox: &PodSandbox,
            _resources: &LinuxContainerResources,
        ) -> Result<()> {
            Ok(())
        }
    }

    /// Scheduler over a pool with the production-default (fail-closed,
    /// deny-unknown-group) validator — what `SandboxPool::new` gives you.
    fn make_scheduler(admission: AdmissionConfig) -> (Arc<JobScheduler>, Arc<FakeBackend>) {
        let backend = Arc::new(FakeBackend::default());
        let pool_config = PoolConfig {
            warm_pool_size: 0,
            admission,
            ..Default::default()
        };
        let pool = Arc::new(SandboxPool::new(
            pool_config,
            backend.clone() as Arc<dyn SandboxBackend>,
        ));
        (Arc::new(JobScheduler::new(pool)), backend)
    }

    /// Scheduler over a pool with a membership-backed validator (B′): the
    /// runs_on selector is only honored for groups `subject` provably belongs
    /// to.
    fn make_scheduler_with_membership(
        admission: AdmissionConfig,
        membership: StaticGroupMembership,
    ) -> (Arc<JobScheduler>, Arc<FakeBackend>) {
        let backend = Arc::new(FakeBackend::default());
        let pool_config = PoolConfig {
            warm_pool_size: 0,
            admission,
            ..Default::default()
        };
        let pool = Arc::new(SandboxPool::with_group_validator(
            pool_config,
            backend.clone() as Arc<dyn SandboxBackend>,
            Arc::new(membership),
        ));
        (Arc::new(JobScheduler::new(pool)), backend)
    }

    #[tokio::test]
    async fn in_proc_runs_on_never_touches_the_backend() {
        let (scheduler, backend) = make_scheduler(AdmissionConfig::default());
        let subject = Subject::new("test-user");

        let placement = scheduler
            .place(
                "job-a",
                &RunsOn::Label("local".into()),
                &JobResources::default(),
                &subject,
            )
            .await
            .expect("in-proc placement never fails");

        assert!(matches!(placement, Placement::InProc));
        assert_eq!(backend.starts.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn isolated_runs_on_acquires_a_real_sandbox() {
        // B′: the runs_on label is a group selector, so the Subject must be a
        // consented member of that group for admission to honor it.
        let (scheduler, backend) = make_scheduler_with_membership(
            AdmissionConfig::default(),
            StaticGroupMembership::new().with_member("kata", "test-user"),
        );
        let subject = Subject::new("test-user");

        let placement = scheduler
            .place(
                "job-b",
                &RunsOn::Label("kata".into()),
                &JobResources::default(),
                &subject,
            )
            .await
            .expect("sandbox placement should succeed");

        assert!(matches!(placement, Placement::Sandbox { .. }));
        assert_eq!(backend.starts.load(Ordering::SeqCst), 1);

        placement.release().await;
    }

    #[tokio::test]
    async fn unverified_runs_on_group_is_rejected_at_admission() {
        // B′ fail-closed default: with no membership source wired
        // (`DenyUnknownGroupValidator`, what `SandboxPool::new` installs), a
        // runs_on label — an unverifiable group selector — must be REJECTED
        // at admission, never silently accepted as a fresh quota bucket. The
        // backend is never touched.
        let (scheduler, backend) = make_scheduler(AdmissionConfig::default());
        let subject = Subject::new("test-user");

        let placement = scheduler
            .place(
                "job-x",
                &RunsOn::Label("kata".into()),
                &JobResources::default(),
                &subject,
            )
            .await;

        assert!(
            placement.is_err(),
            "unverifiable group selector must be rejected fail-closed (B′)"
        );
        assert_eq!(backend.starts.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn runs_on_label_influences_admission_via_group_quota() {
        // max_per_group: 1 means a second job under the SAME runs_on label
        // cannot be admitted concurrently, while a job under a DIFFERENT
        // label is unaffected — proving the runs_on value actually reaches
        // and influences the scheduler's placement decision, not just that
        // *a* sandbox gets created. Both labels are consented memberships of
        // the Subject (B′) — the quota partition only applies to validated
        // selectors.
        let (scheduler, _backend) = make_scheduler_with_membership(
            AdmissionConfig {
                max_per_group: Some(1),
                ..AdmissionConfig::default()
            },
            StaticGroupMembership::new()
                .with_member("gpu-only", "test-user")
                .with_member("cpu-only", "test-user"),
        );
        let subject = Subject::new("test-user");

        let first = scheduler
            .place(
                "job-1",
                &RunsOn::Label("gpu-only".into()),
                &JobResources::default(),
                &subject,
            )
            .await
            .expect("first gpu-only job admitted");

        // Second job, SAME label, same subject: over the per-group quota.
        let second = scheduler
            .place(
                "job-2",
                &RunsOn::Label("gpu-only".into()),
                &JobResources::default(),
                &subject,
            )
            .await;
        assert!(
            second.is_err(),
            "second job under the same maxed-out group should be rejected"
        );

        // Third job, DIFFERENT label: its own group quota, unaffected by gpu-only's.
        let third = scheduler
            .place(
                "job-3",
                &RunsOn::Label("cpu-only".into()),
                &JobResources::default(),
                &subject,
            )
            .await
            .expect("different runs_on label has its own group quota");

        first.release().await;
        third.release().await;
    }
}
