//! Workflow runner — executes workflows through the VFS namespace.
//!
//! The WorkflowRunner is a general-purpose execution engine that:
//! - Topologically sorts jobs by `needs:` dependencies into execution waves
//! - Runs independent jobs in each wave concurrently via `tokio::JoinSet`
//! - Resolves `uses:` action steps through VFS `/bin/` ctl files
//! - Evaluates `run:` script steps via TclShell on `spawn_blocking`
//! - Propagates CancellationToken for cooperative cancellation
//! - Routes each job through a [`JobScheduler`] (#527) — `runs_on:` +
//!   `resources:` decide in-proc vs. a real P2-admitted sandbox reservation
//!   (see `workflow::scheduler`); no scheduler configured (`None`, the
//!   default) preserves the pre-#527 in-proc-only behavior exactly.
//! - Honors `WorkflowConfig::max_concurrent_runs` (a run-admission semaphore,
//!   distinct from `script_semaphore`'s per-job script concurrency) and
//!   `job_timeout_secs`/`step_timeout_secs` (#521 — wired for real, not
//!   superseded by P2: admission governs resource capacity/quota, this
//!   governs wall-clock run/step duration, a different axis)

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use hyprstream_workers_tcl::TclShell;
// VFS proxy no longer needed — TclShell awaits namespace operations directly.
use hyprstream_vfs::{Namespace, Subject};

use crate::config::WorkflowConfig;

use super::parser::{Job, Step, Workflow};
use super::scheduler::{JobScheduler, Placement};

// ─────────────────────────────────────────────────────────────────────────────
// Result types (local to runner, not wire-format)
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a complete workflow run.
#[derive(Debug, Clone)]
pub struct WorkflowRunResult {
    pub success: bool,
    pub jobs: HashMap<String, JobRunResult>,
}

/// Outcome of a single job.
#[derive(Debug, Clone)]
pub struct JobRunResult {
    pub name: String,
    pub success: bool,
    pub steps: Vec<StepRunResult>,
}

/// Outcome of a single step.
#[derive(Debug, Clone)]
pub struct StepRunResult {
    pub name: String,
    pub success: bool,
    pub output: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors specific to workflow execution.
#[derive(Debug, thiserror::Error)]
pub enum RunnerError {
    #[error("invalid action name: {0}")]
    InvalidActionName(String),

    #[error("invalid input key: {0}")]
    InvalidInputKey(String),

    #[error("invalid env var name: {0}")]
    InvalidEnvVarName(String),

    #[error("dependency cycle detected in jobs")]
    DependencyCycle,

    #[error("unknown job dependency: {0}")]
    UnknownDependency(String),

    #[error("step failed: {0}")]
    StepFailed(String),

    #[error("cancelled")]
    Cancelled,

    #[error("vfs error: {0}")]
    VfsError(String),

    #[error("tcl error: {0}")]
    TclError(String),

    #[error("join error: {0}")]
    JoinError(String),
}

type RunnerResult<T> = std::result::Result<T, RunnerError>;

// ─────────────────────────────────────────────────────────────────────────────
// WorkflowRunner
// ─────────────────────────────────────────────────────────────────────────────

/// General-purpose workflow execution engine.
///
/// Resolves action steps through VFS `/bin/` ctl files and evaluates
/// script steps via TclShell on `spawn_blocking` threads. Concurrent
/// script-executing jobs are capped by `script_semaphore`.
///
/// # Scheduling (#527)
///
/// Each job is routed through an optional [`JobScheduler`]: `runs_on:` +
/// `resources:` decide in-proc execution (unchanged) vs. a real sandbox
/// reservation acquired through the #525 P2 admission engine. With no
/// scheduler configured (the default), every job runs in-proc exactly as
/// before this change — no regression for existing callers.
///
/// # Run-level concurrency + timeouts (#521)
///
/// `run_semaphore` bounds concurrent workflow *runs* (`WorkflowConfig::
/// max_concurrent_runs`) — a different axis from `script_semaphore`, which
/// bounds concurrent script-executing *jobs* within/across runs.
/// `job_timeout`/`step_timeout` (`job_timeout_secs`/`step_timeout_secs`) wrap
/// job and step execution in `tokio::time::timeout`; a job's own
/// `timeout-minutes:` overrides `job_timeout` when present.
#[derive(Clone)]
pub struct WorkflowRunner {
    ns: Arc<Namespace>,
    script_semaphore: Arc<Semaphore>,
    run_semaphore: Arc<Semaphore>,
    job_timeout: Duration,
    step_timeout: Duration,
    scheduler: Option<Arc<JobScheduler>>,
}

impl WorkflowRunner {
    /// Create a new WorkflowRunner with `WorkflowConfig::default()` timeouts/
    /// concurrency and no job scheduler (every job runs in-proc).
    ///
    /// * `ns` — the root namespace (mounts for `/bin/`, `/env/`, `/srv/`, etc.)
    /// * Default concurrency cap: 64 concurrent script-executing jobs.
    pub fn new(ns: Arc<Namespace>) -> Self {
        Self::with_config(ns, &WorkflowConfig::default())
    }

    /// Create a WorkflowRunner honoring `config`'s `max_concurrent_runs` +
    /// `job_timeout_secs`/`step_timeout_secs` (#521).
    pub fn with_config(ns: Arc<Namespace>, config: &WorkflowConfig) -> Self {
        Self {
            ns,
            script_semaphore: Arc::new(Semaphore::new(64)),
            // A zero-permit semaphore would deadlock every run forever;
            // treat a misconfigured 0 as "at least 1" rather than a silent hang.
            run_semaphore: Arc::new(Semaphore::new(config.max_concurrent_runs.max(1))),
            job_timeout: Duration::from_secs(config.job_timeout_secs),
            step_timeout: Duration::from_secs(config.step_timeout_secs),
            scheduler: None,
        }
    }

    /// Attach a [`JobScheduler`] (#527): jobs whose `runs_on:` requests
    /// isolation are placed onto its sandbox pool instead of running in-proc.
    pub fn with_scheduler(mut self, scheduler: Arc<JobScheduler>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Run a complete workflow.
    ///
    /// Jobs are topologically sorted by `needs:` into execution waves.
    /// Independent jobs in each wave run concurrently. The `cancel` token
    /// is checked between waves and propagated to individual jobs.
    ///
    /// Bounded by `run_semaphore` (#521's `max_concurrent_runs`): a run that
    /// can't get a permit waits here, honoring `cancel` while waiting.
    pub async fn run(
        &self,
        workflow: &Workflow,
        inputs: HashMap<String, String>,
        subject: &Subject,
        cancel: CancellationToken,
    ) -> RunnerResult<WorkflowRunResult> {
        // Validate inputs: reject _-prefixed keys (provenance injection prevention).
        validate_inputs(&inputs)?;

        // #521: bound concurrent workflow RUNS. Held for the whole run.
        let _run_permit = tokio::select! {
            permit = Arc::clone(&self.run_semaphore).acquire_owned() => {
                permit.map_err(|_| RunnerError::Cancelled)?
            }
            _ = cancel.cancelled() => return Err(RunnerError::Cancelled),
        };

        // Topological sort jobs into execution waves.
        let waves = topological_sort(&workflow.jobs)?;

        let mut all_jobs = HashMap::new();
        let mut all_success = true;

        for wave in waves {
            if cancel.is_cancelled() {
                return Err(RunnerError::Cancelled);
            }

            let mut join_set = JoinSet::new();

            for job_name in wave {
                // Job names come from topological_sort which iterates jobs.keys(),
                // so this lookup cannot fail. Continue defensively to satisfy clippy.
                let Some(job) = workflow.jobs.get(&job_name).cloned() else {
                    continue;
                };
                let ns = Arc::clone(&self.ns);
                let sem = Arc::clone(&self.script_semaphore);
                let scheduler = self.scheduler.clone();
                let job_timeout = job
                    .timeout_minutes
                    .map(|m| Duration::from_secs(u64::from(m) * 60))
                    .unwrap_or(self.job_timeout);
                let step_timeout = self.step_timeout;
                let subject = subject.clone();
                let cancel = cancel.clone();
                let env = merge_env(&workflow.env, &job.env);
                let inputs = inputs.clone();
                let name = job_name.clone();

                join_set.spawn(async move {
                    let result = run_job(
                        ns, sem, scheduler, &job, &name, env, &inputs, &subject, cancel,
                        job_timeout, step_timeout,
                    ).await;
                    (name, result)
                });
            }

            while let Some(result) = join_set.join_next().await {
                match result {
                    Ok((name, Ok(job_result))) => {
                        if !job_result.success {
                            all_success = false;
                        }
                        all_jobs.insert(name, job_result);
                    }
                    Ok((name, Err(e))) => {
                        all_success = false;
                        all_jobs.insert(
                            name.clone(),
                            JobRunResult {
                                name,
                                success: false,
                                steps: vec![StepRunResult {
                                    name: "<error>".to_owned(),
                                    success: false,
                                    output: e.to_string(),
                                }],
                            },
                        );
                    }
                    Err(e) => {
                        all_success = false;
                        tracing::error!("Job task panicked: {e}");
                    }
                }
            }

            // If any job in this wave failed, stop executing subsequent waves.
            if !all_success {
                break;
            }
        }

        Ok(WorkflowRunResult {
            success: all_success,
            jobs: all_jobs,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Job execution
// ─────────────────────────────────────────────────────────────────────────────

/// Route `job` through the scheduler (#527), then run its steps within
/// `job_timeout`, releasing the placement afterward regardless of outcome.
#[allow(clippy::too_many_arguments)]
async fn run_job(
    ns: Arc<Namespace>,
    semaphore: Arc<Semaphore>,
    scheduler: Option<Arc<JobScheduler>>,
    job: &Job,
    job_name: &str,
    env: HashMap<String, String>,
    inputs: &HashMap<String, String>,
    subject: &Subject,
    cancel: CancellationToken,
    job_timeout: Duration,
    step_timeout: Duration,
) -> RunnerResult<JobRunResult> {
    // #527: decide + (if isolation is requested) acquire a sandbox
    // reservation before running any steps. `None` scheduler ⇒ always
    // in-proc (pre-#527 behavior, unchanged).
    let placement = match &scheduler {
        Some(scheduler) => scheduler
            .place(job_name, &job.runs_on, &job.resources, subject)
            .await
            .map_err(|e| RunnerError::VfsError(format!("job placement failed: {e}")))?,
        None => Placement::InProc,
    };

    // #521: bound job execution wall-clock time. A job's own
    // `timeout-minutes:` already folded into `job_timeout` by the caller.
    let outcome = tokio::time::timeout(
        job_timeout,
        run_job_steps(ns, semaphore, job, job_name, env, inputs, subject, cancel, step_timeout),
    )
    .await;

    placement.release().await;

    match outcome {
        Ok(result) => result,
        Err(_elapsed) => Ok(JobRunResult {
            name: job_name.to_owned(),
            success: false,
            steps: vec![StepRunResult {
                name: "<timeout>".to_owned(),
                success: false,
                output: format!("job timed out after {}s", job_timeout.as_secs()),
            }],
        }),
    }
}

/// Execute a single job's steps sequentially. For script steps (`run:`), a
/// single TclShell is reused across all sequential steps in the same job
/// (constructed inside `spawn_blocking`). Each step is individually bounded
/// by `step_timeout` (#521).
#[allow(clippy::too_many_arguments)]
async fn run_job_steps(
    ns: Arc<Namespace>,
    semaphore: Arc<Semaphore>,
    job: &Job,
    job_name: &str,
    env: HashMap<String, String>,
    inputs: &HashMap<String, String>,
    subject: &Subject,
    cancel: CancellationToken,
    step_timeout: Duration,
) -> RunnerResult<JobRunResult> {
    let mut step_results = Vec::new();
    let mut job_success = true;

    // Partition steps into action steps and script steps.
    // We process them sequentially, but script steps share a TclShell.
    let has_scripts = job.steps.iter().any(|s| s.run.is_some());

    if has_scripts {
        // Acquire semaphore permit for script execution.
        let _permit = semaphore
            .acquire()
            .await
            .map_err(|_| RunnerError::Cancelled)?;

        // Fork namespace for script isolation: remove /config, /private, /srv.
        let mut forked = ns.fork();
        forked.unmount("/config");
        forked.unmount("/private");
        forked.unmount("/srv");

        // Write env vars to /env/* in forked namespace.
        // We use an in-memory mount for env vars.
        let env_mount = Arc::new(EnvMount::new(&env, inputs));
        forked
            .mount("/env", env_mount)
            .map_err(|e| RunnerError::VfsError(e.to_string()))?;

        let forked = Arc::new(forked);

        // Collect steps to process on the blocking thread.
        let steps: Vec<Step> = job.steps.clone();
        let subject_clone = subject.clone();
        let ns_for_actions = Arc::clone(&ns);
        let subject_for_actions = subject.clone();

        // Spawn a single blocking thread with one TclShell for the job.
        // Script steps are sent to it via channel; action steps run on async runtime.
        // The thread runs a current-thread tokio runtime + LocalSet so TclShell
        // (which is !Send due to Rc in molt Value) can .await VFS operations directly.
        let (script_tx, mut script_rx) = tokio::sync::mpsc::channel::<(
            String,                                                          // script
            HashMap<String, String>,                                         // step env
            tokio::sync::oneshot::Sender<Result<String, String>>,            // reply
        )>(1);

        let shell_handle = {
            let ns = Arc::clone(&forked);
            let subject_clone = subject_clone.clone();
            #[allow(clippy::expect_used)] // Runtime creation failure is unrecoverable
            tokio::task::spawn_blocking(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("tcl runner runtime");
                let local = tokio::task::LocalSet::new();
                local.block_on(&rt, async {
                    let mut shell = TclShell::new(subject_clone, ns);
                    while let Some((script, step_env, reply)) = script_rx.recv().await {
                        for (k, v) in &step_env {
                            let _ = shell.set_env(k, v);
                        }
                        let result = shell.eval(&script).await;
                        let _ = reply.send(result);
                    }
                });
            })
        };

        for (i, step) in steps.iter().enumerate() {
            if cancel.is_cancelled() {
                drop(script_tx);
                let _ = shell_handle.await;
                return Err(RunnerError::Cancelled);
            }

            let step_name = step
                .name
                .clone()
                .unwrap_or_else(|| format!("step-{}", i));

            if let Some(ref action) = step.uses {
                // Action step — runs on async runtime, bounded by step_timeout (#521).
                let result = match tokio::time::timeout(
                    step_timeout,
                    run_action_step(action, &step.with, &ns_for_actions, &subject_for_actions),
                )
                .await
                {
                    Ok(r) => r,
                    Err(_elapsed) => Err(RunnerError::StepFailed(format!(
                        "step timed out after {}s",
                        step_timeout.as_secs()
                    ))),
                };
                match result {
                    Ok(output) => {
                        step_results.push(StepRunResult {
                            name: step_name,
                            success: true,
                            output,
                        });
                    }
                    Err(e) => {
                        let failed = StepRunResult {
                            name: step_name,
                            success: false,
                            output: e.to_string(),
                        };
                        step_results.push(failed);
                        if !step.continue_on_error {
                            job_success = false;
                            break;
                        }
                    }
                }
            } else if let Some(ref script) = step.run {
                // Script step — send to the shared TclShell on the blocking thread.
                // Uses tokio channels: async send + oneshot reply to avoid blocking
                // tokio worker threads.
                let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
                if script_tx
                    .send((script.clone(), step.env.clone(), reply_tx))
                    .await
                    .is_err()
                {
                    step_results.push(StepRunResult {
                        name: step_name,
                        success: false,
                        output: "TclShell thread exited".into(),
                    });
                    job_success = false;
                    break;
                }
                // Bounded by step_timeout (#521); on elapsed the reply is simply
                // dropped — the shell thread will finish the eval on its own time
                // and move on to the next queued script (or idle if none comes).
                match tokio::time::timeout(step_timeout, reply_rx).await {
                    Ok(Ok(Ok(output))) => {
                        step_results.push(StepRunResult {
                            name: step_name,
                            success: true,
                            output,
                        });
                    }
                    Ok(Ok(Err(e))) => {
                        let failed = StepRunResult {
                            name: step_name,
                            success: false,
                            output: e,
                        };
                        step_results.push(failed);
                        if !step.continue_on_error {
                            job_success = false;
                            break;
                        }
                    }
                    Ok(Err(_)) => {
                        step_results.push(StepRunResult {
                            name: step_name,
                            success: false,
                            output: "TclShell thread panicked".into(),
                        });
                        job_success = false;
                        break;
                    }
                    Err(_elapsed) => {
                        step_results.push(StepRunResult {
                            name: step_name,
                            success: false,
                            output: format!("step timed out after {}s", step_timeout.as_secs()),
                        });
                        job_success = false;
                        break;
                    }
                }
            }
        }

        // Drop the sender to signal the shell thread to exit, then wait for it.
        drop(script_tx);
        let _ = shell_handle.await;
    } else {
        // No script steps — only action steps, run on async runtime.
        for (i, step) in job.steps.iter().enumerate() {
            if cancel.is_cancelled() {
                return Err(RunnerError::Cancelled);
            }

            let step_name = step
                .name
                .clone()
                .unwrap_or_else(|| format!("step-{}", i));

            if let Some(ref action) = step.uses {
                // Bounded by step_timeout (#521).
                let result = match tokio::time::timeout(
                    step_timeout,
                    run_action_step(action, &step.with, &ns, subject),
                )
                .await
                {
                    Ok(r) => r,
                    Err(_elapsed) => Err(RunnerError::StepFailed(format!(
                        "step timed out after {}s",
                        step_timeout.as_secs()
                    ))),
                };
                match result {
                    Ok(output) => {
                        step_results.push(StepRunResult {
                            name: step_name,
                            success: true,
                            output,
                        });
                    }
                    Err(e) => {
                        let failed = StepRunResult {
                            name: step_name,
                            success: false,
                            output: e.to_string(),
                        };
                        step_results.push(failed);
                        if !step.continue_on_error {
                            job_success = false;
                            break;
                        }
                    }
                }
            }
        }
    }

    Ok(JobRunResult {
        name: job_name.to_owned(),
        success: job_success,
        steps: step_results,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Action step execution
// ─────────────────────────────────────────────────────────────────────────────

/// Run an action step by writing JSON of `with:` map to `/bin/{action}` ctl file.
///
/// `uses: solver/submit@v1` → strip `@v1`, validate name, write to `/bin/solver/submit`.
async fn run_action_step(
    action: &str,
    with: &HashMap<String, String>,
    ns: &Namespace,
    subject: &Subject,
) -> RunnerResult<String> {
    // Strip @version suffix.
    let action_name = action.split('@').next().unwrap_or(action);

    // Validate action name.
    validate_action_name(action_name)?;

    // Build JSON payload from `with:` map.
    let payload = serde_json::to_vec(with).map_err(|e| RunnerError::VfsError(e.to_string()))?;

    // Write to /bin/{action} ctl file.
    let ctl_path = format!("/bin/{action_name}");
    let response = ns
        .ctl(&ctl_path, &payload, subject)
        .await
        .map_err(|e| RunnerError::VfsError(e.to_string()))?;

    Ok(String::from_utf8_lossy(&response).into_owned())
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Validate action name: no `..`, only alphanumeric + `/` + `-` + `_`.
fn validate_action_name(name: &str) -> RunnerResult<()> {
    if name.contains("..") {
        return Err(RunnerError::InvalidActionName(format!(
            "path traversal in action name: {name}"
        )));
    }
    if name.is_empty() {
        return Err(RunnerError::InvalidActionName("empty action name".into()));
    }
    for ch in name.chars() {
        if !ch.is_alphanumeric() && ch != '/' && ch != '-' && ch != '_' {
            return Err(RunnerError::InvalidActionName(format!(
                "invalid character '{ch}' in action name: {name}"
            )));
        }
    }
    Ok(())
}

/// Validate input keys: reject `_`-prefixed keys (provenance injection prevention)
/// and non-alphanumeric characters (prevent path traversal when prefixed with `INPUT_`).
fn validate_inputs(inputs: &HashMap<String, String>) -> RunnerResult<()> {
    for key in inputs.keys() {
        if key.starts_with('_') {
            return Err(RunnerError::InvalidInputKey(format!(
                "input key must not start with '_': {key}"
            )));
        }
        // Validate characters so INPUT_{key} is safe as a VFS path component.
        for ch in key.chars() {
            if !ch.is_alphanumeric() && ch != '_' && ch != '-' {
                return Err(RunnerError::InvalidInputKey(format!(
                    "invalid character '{ch}' in input key: {key}"
                )));
            }
        }
    }
    Ok(())
}

/// Validate env var name: only alphanumeric + `_` + `-`.
fn validate_env_var_name(name: &str) -> RunnerResult<()> {
    if name.is_empty() {
        return Err(RunnerError::InvalidEnvVarName("empty env var name".into()));
    }
    for ch in name.chars() {
        if !ch.is_alphanumeric() && ch != '_' && ch != '-' {
            return Err(RunnerError::InvalidEnvVarName(format!(
                "invalid character '{ch}' in env var name: {name}"
            )));
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Topological sort
// ─────────────────────────────────────────────────────────────────────────────

/// Topological sort of jobs by `needs:` into execution waves.
///
/// Returns a Vec of waves, where each wave is a Vec of job names that
/// can execute concurrently. Jobs in wave N+1 depend on jobs in wave N or earlier.
fn topological_sort(jobs: &HashMap<String, Job>) -> RunnerResult<Vec<Vec<String>>> {
    let job_names: HashSet<&str> = jobs.keys().map(String::as_str).collect();

    // Validate all dependencies exist.
    for (name, job) in jobs {
        if let Some(ref needs) = job.needs {
            for dep in needs {
                if !job_names.contains(dep.as_str()) {
                    return Err(RunnerError::UnknownDependency(format!(
                        "job '{name}' depends on unknown job '{dep}'"
                    )));
                }
            }
        }
    }

    // Kahn's algorithm.
    let mut in_degree: HashMap<&str, usize> = HashMap::new();
    let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

    for name in jobs.keys() {
        in_degree.entry(name.as_str()).or_insert(0);
        dependents.entry(name.as_str()).or_default();
    }

    for (name, job) in jobs {
        if let Some(ref needs) = job.needs {
            *in_degree.entry(name.as_str()).or_insert(0) += needs.len();
            for dep in needs {
                dependents
                    .entry(dep.as_str())
                    .or_default()
                    .push(name.as_str());
            }
        }
    }

    let mut waves = Vec::new();
    let mut queue: VecDeque<&str> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&name, _)| name)
        .collect();

    let mut processed = 0;

    while !queue.is_empty() {
        let mut wave: Vec<String> = Vec::new();
        let mut next_queue = VecDeque::new();

        for &name in &queue {
            wave.push(name.to_owned());
            processed += 1;

            for &dependent in dependents.get(name).unwrap_or(&Vec::new()) {
                let Some(deg) = in_degree.get_mut(dependent) else {
                    continue;
                };
                *deg -= 1;
                if *deg == 0 {
                    next_queue.push_back(dependent);
                }
            }
        }

        // Sort wave for deterministic ordering in tests.
        wave.sort();
        waves.push(wave);
        queue = next_queue;
    }

    if processed != jobs.len() {
        return Err(RunnerError::DependencyCycle);
    }

    Ok(waves)
}

// ─────────────────────────────────────────────────────────────────────────────
// Env merge + mount
// ─────────────────────────────────────────────────────────────────────────────

/// Merge workflow-level and job-level env vars (job overrides workflow).
fn merge_env(
    workflow_env: &HashMap<String, String>,
    job_env: &HashMap<String, String>,
) -> HashMap<String, String> {
    let mut merged = workflow_env.clone();
    merged.extend(job_env.iter().map(|(k, v)| (k.clone(), v.clone())));
    merged
}

/// Simple in-memory Mount that serves env vars and inputs as files.
///
/// Files at `/env/{KEY}` return the value as bytes.
/// Inputs are stored with `INPUT_` prefix.
struct EnvMount {
    vars: HashMap<String, Vec<u8>>,
}

impl EnvMount {
    fn new(env: &HashMap<String, String>, inputs: &HashMap<String, String>) -> Self {
        let mut vars = HashMap::new();
        for (k, v) in env {
            if validate_env_var_name(k).is_ok() {
                vars.insert(k.clone(), v.as_bytes().to_vec());
            }
        }
        for (k, v) in inputs {
            vars.insert(format!("INPUT_{k}"), v.as_bytes().to_vec());
        }
        Self { vars }
    }
}

/// Fid type for EnvMount.
struct EnvFid {
    path: String,
}

#[async_trait::async_trait]
impl hyprstream_vfs::Mount for EnvMount {
    async fn walk(
        &self,
        components: &[&str],
        _caller: &Subject,
    ) -> Result<hyprstream_vfs::Fid, hyprstream_vfs::MountError> {
        let path = components.join("/");
        Ok(hyprstream_vfs::Fid::new(EnvFid { path }))
    }

    async fn open(
        &self,
        _fid: &mut hyprstream_vfs::Fid,
        _mode: u8,
        _caller: &Subject,
    ) -> Result<(), hyprstream_vfs::MountError> {
        Ok(())
    }

    async fn read(
        &self,
        fid: &hyprstream_vfs::Fid,
        offset: u64,
        _count: u32,
        _caller: &Subject,
    ) -> Result<Vec<u8>, hyprstream_vfs::MountError> {
        let inner = fid.downcast_ref::<EnvFid>().ok_or_else(|| {
            hyprstream_vfs::MountError::InvalidArgument("bad fid".into())
        })?;
        match self.vars.get(&inner.path) {
            Some(data) => {
                let start = offset as usize;
                if start >= data.len() {
                    Ok(vec![])
                } else {
                    Ok(data[start..].to_vec())
                }
            }
            None => Err(hyprstream_vfs::MountError::NotFound(inner.path.clone())),
        }
    }

    async fn write(
        &self,
        _fid: &hyprstream_vfs::Fid,
        _offset: u64,
        data: &[u8],
        _caller: &Subject,
    ) -> Result<u32, hyprstream_vfs::MountError> {
        Ok(data.len() as u32)
    }

    async fn readdir(
        &self,
        _fid: &hyprstream_vfs::Fid,
        _caller: &Subject,
    ) -> Result<Vec<hyprstream_vfs::DirEntry>, hyprstream_vfs::MountError> {
        Ok(self
            .vars
            .keys()
            .map(|k| hyprstream_vfs::DirEntry {
                name: k.clone(),
                is_dir: false,
                size: 0,
                stat: None,
            })
            .collect())
    }

    async fn stat(
        &self,
        fid: &hyprstream_vfs::Fid,
        _caller: &Subject,
    ) -> Result<hyprstream_vfs::Stat, hyprstream_vfs::MountError> {
        let inner = fid.downcast_ref::<EnvFid>().ok_or_else(|| {
            hyprstream_vfs::MountError::InvalidArgument("bad fid".into())
        })?;
        Ok(hyprstream_vfs::Stat::unknown_qid(0, 0, inner.path.clone(), 0))
    }

    async fn clunk(&self, _fid: hyprstream_vfs::Fid, _caller: &Subject) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_types)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
    use std::sync::Arc;

    /// In-memory mount that supports ctl pattern (write→read).
    struct MemMount {
        files: HashMap<String, Vec<u8>>,
    }

    impl MemMount {
        fn new(files: Vec<(&str, &[u8])>) -> Self {
            Self {
                files: files
                    .into_iter()
                    .map(|(k, v)| (k.to_owned(), v.to_vec()))
                    .collect(),
            }
        }
    }

    struct MemFid {
        path: String,
        write_buf: std::sync::Mutex<Vec<u8>>,
    }

    #[async_trait]
    impl Mount for MemMount {
        async fn walk(
            &self,
            components: &[&str],
            _caller: &Subject,
        ) -> Result<Fid, MountError> {
            let path = components.join("/");
            Ok(Fid::new(MemFid {
                path,
                write_buf: std::sync::Mutex::new(Vec::new()),
            }))
        }

        async fn open(
            &self,
            _fid: &mut Fid,
            _mode: u8,
            _caller: &Subject,
        ) -> Result<(), MountError> {
            Ok(())
        }

        async fn read(
            &self,
            fid: &Fid,
            offset: u64,
            _count: u32,
            _caller: &Subject,
        ) -> Result<Vec<u8>, MountError> {
            let inner = fid
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            // If there's a write buffer (ctl pattern), return that.
            let wb = inner.write_buf.lock().unwrap();
            if !wb.is_empty() {
                let start = offset as usize;
                if start >= wb.len() {
                    return Ok(vec![]);
                }
                return Ok(wb[start..].to_vec());
            }
            drop(wb);
            match self.files.get(&inner.path) {
                Some(data) => {
                    let start = offset as usize;
                    if start >= data.len() {
                        Ok(vec![])
                    } else {
                        Ok(data[start..].to_vec())
                    }
                }
                None => Err(MountError::NotFound(inner.path.clone())),
            }
        }

        async fn write(
            &self,
            fid: &Fid,
            _offset: u64,
            data: &[u8],
            _caller: &Subject,
        ) -> Result<u32, MountError> {
            let inner = fid
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            *inner.write_buf.lock().unwrap() =
                format!("ok: {}", String::from_utf8_lossy(data)).into_bytes();
            Ok(data.len() as u32)
        }

        async fn readdir(
            &self,
            fid: &Fid,
            _caller: &Subject,
        ) -> Result<Vec<DirEntry>, MountError> {
            let inner = fid
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let prefix = if inner.path.is_empty() {
                String::new()
            } else {
                format!("{}/", inner.path)
            };
            let mut entries = Vec::new();
            for key in self.files.keys() {
                if let Some(rest) = key.strip_prefix(&prefix) {
                    if !rest.contains('/') {
                        entries.push(DirEntry {
                            name: rest.to_owned(),
                            is_dir: false,
                            size: 0,
                            stat: None,
                        });
                    }
                } else if inner.path.is_empty() && !key.contains('/') {
                    entries.push(DirEntry {
                        name: key.clone(),
                        is_dir: false,
                        size: 0,
                        stat: None,
                    });
                }
            }
            Ok(entries)
        }

        async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            let inner = fid
                .downcast_ref::<MemFid>()
                .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            Ok(Stat::unknown_qid(0, 0, inner.path.clone(), 0))
        }

        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    /// Ctl-pattern mount whose `write` actually awaits (`tokio::time::sleep`)
    /// before answering "ok" — used to make `step_timeout`/`job_timeout`
    /// (#521) deterministically fire in tests: a step routed through this
    /// mount really pends, so `tokio::time::timeout` races it for real
    /// instead of the inner future completing synchronously on first poll.
    struct SlowMount {
        delay: std::time::Duration,
    }

    struct SlowFid;

    #[async_trait]
    impl Mount for SlowMount {
        async fn walk(&self, _components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            Ok(Fid::new(SlowFid))
        }
        async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
            Ok(())
        }
        async fn write(
            &self,
            _fid: &Fid,
            _offset: u64,
            _data: &[u8],
            _caller: &Subject,
        ) -> Result<u32, MountError> {
            tokio::time::sleep(self.delay).await;
            Ok(0)
        }
        async fn read(
            &self,
            _fid: &Fid,
            offset: u64,
            _count: u32,
            _caller: &Subject,
        ) -> Result<Vec<u8>, MountError> {
            if offset == 0 {
                Ok(b"ok".to_vec())
            } else {
                Ok(vec![])
            }
        }
        async fn readdir(&self, _fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            Ok(vec![])
        }
        async fn stat(&self, _fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            Ok(Stat::unknown_qid(0, 0, String::new(), 0))
        }
        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    /// Ctl-pattern mount that tracks concurrently-in-flight `write` calls (for
    /// the `max_concurrent_runs` / #521 test): bumps a shared counter, records
    /// the observed peak, sleeps briefly (so overlapping calls actually
    /// overlap), then decrements.
    struct TrackingMount {
        current: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
        delay: std::time::Duration,
    }

    struct TrackingFid;

    #[async_trait]
    impl Mount for TrackingMount {
        async fn walk(&self, _components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            Ok(Fid::new(TrackingFid))
        }
        async fn open(&self, _fid: &mut Fid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
            Ok(())
        }
        async fn write(
            &self,
            _fid: &Fid,
            _offset: u64,
            _data: &[u8],
            _caller: &Subject,
        ) -> Result<u32, MountError> {
            use std::sync::atomic::Ordering;
            let cur = self.current.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(cur, Ordering::SeqCst);
            tokio::time::sleep(self.delay).await;
            self.current.fetch_sub(1, Ordering::SeqCst);
            Ok(0)
        }
        async fn read(
            &self,
            _fid: &Fid,
            offset: u64,
            _count: u32,
            _caller: &Subject,
        ) -> Result<Vec<u8>, MountError> {
            if offset == 0 {
                Ok(b"ok".to_vec())
            } else {
                Ok(vec![])
            }
        }
        async fn readdir(&self, _fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            Ok(vec![])
        }
        async fn stat(&self, _fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            Ok(Stat::unknown_qid(0, 0, String::new(), 0))
        }
        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    fn test_subject() -> Subject {
        Subject::new("test-runner")
    }

    fn make_namespace() -> Arc<Namespace> {
        let mut ns = Namespace::new();

        // /bin/ with ctl files for action step testing.
        let bin_mount = Arc::new(MemMount::new(vec![
            ("solver/submit", b""),
            ("deploy/run", b""),
        ]));
        ns.mount("/bin", bin_mount).unwrap();

        // /config/ mount (will be unmounted in forked namespaces).
        let config_mount = Arc::new(MemMount::new(vec![("secret", b"hunter2")]));
        ns.mount("/config", config_mount).unwrap();

        Arc::new(ns)
    }

    fn make_namespace_with_slow_action(delay: std::time::Duration) -> Arc<Namespace> {
        let mut ns = Namespace::new();
        let bin_mount = Arc::new(SlowMount { delay });
        ns.mount("/bin", bin_mount).unwrap();
        Arc::new(ns)
    }

    fn make_namespace_with_tracking_action(
        current: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
        delay: std::time::Duration,
    ) -> Arc<Namespace> {
        let mut ns = Namespace::new();
        let bin_mount = Arc::new(TrackingMount {
            current,
            peak,
            delay,
        });
        ns.mount("/bin", bin_mount).unwrap();
        Arc::new(ns)
    }

    fn single_action_job(action: &str) -> Job {
        Job {
            runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
            needs: None,
            env: HashMap::new(),
            steps: vec![Step {
                name: Some("step".into()),
                id: None,
                uses: Some(action.to_owned()),
                run: None,
                shell: None,
                working_directory: None,
                with: HashMap::new(),
                env: HashMap::new(),
                condition: None,
                continue_on_error: false,
            }],
            condition: None,
            timeout_minutes: None,
            resources: super::super::parser::JobResources::default(),
        }
    }

    fn make_simple_workflow(jobs: HashMap<String, Job>) -> Workflow {
        Workflow {
            name: "test-workflow".to_owned(),
            on: super::super::parser::WorkflowTrigger::Simple("push".to_owned()),
            env: HashMap::new(),
            jobs,
        }
    }

    // ── Topological sort tests ───────────────────────────────────────────

    #[test]
    fn topo_sort_independent_jobs() {
        let mut jobs = HashMap::new();
        jobs.insert(
            "a".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: None,
                env: HashMap::new(),
                steps: vec![],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );
        jobs.insert(
            "b".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: None,
                env: HashMap::new(),
                steps: vec![],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );

        let waves = topological_sort(&jobs).unwrap();
        assert_eq!(waves.len(), 1);
        assert_eq!(waves[0].len(), 2);
    }

    #[test]
    fn topo_sort_with_dependency() {
        let mut jobs = HashMap::new();
        jobs.insert(
            "build".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: None,
                env: HashMap::new(),
                steps: vec![],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );
        jobs.insert(
            "test".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: Some(vec!["build".to_owned()]),
                env: HashMap::new(),
                steps: vec![],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );
        jobs.insert(
            "deploy".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: Some(vec!["test".to_owned()]),
                env: HashMap::new(),
                steps: vec![],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );

        let waves = topological_sort(&jobs).unwrap();
        assert_eq!(waves.len(), 3);
        assert_eq!(waves[0], vec!["build"]);
        assert_eq!(waves[1], vec!["test"]);
        assert_eq!(waves[2], vec!["deploy"]);
    }

    #[test]
    fn topo_sort_cycle_detected() {
        let mut jobs = HashMap::new();
        jobs.insert(
            "a".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: Some(vec!["b".to_owned()]),
                env: HashMap::new(),
                steps: vec![],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );
        jobs.insert(
            "b".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: Some(vec!["a".to_owned()]),
                env: HashMap::new(),
                steps: vec![],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );

        let result = topological_sort(&jobs);
        assert!(matches!(result, Err(RunnerError::DependencyCycle)));
    }

    // ── Validation tests ─────────────────────────────────────────────────

    #[test]
    fn validate_action_name_rejects_dotdot() {
        assert!(validate_action_name("../etc/passwd").is_err());
        assert!(validate_action_name("foo/../bar").is_err());
    }

    #[test]
    fn validate_action_name_accepts_valid() {
        assert!(validate_action_name("solver/submit").is_ok());
        assert!(validate_action_name("deploy-prod").is_ok());
        assert!(validate_action_name("my_action").is_ok());
    }

    #[test]
    fn validate_action_name_rejects_special_chars() {
        assert!(validate_action_name("foo bar").is_err());
        assert!(validate_action_name("foo;bar").is_err());
    }

    #[test]
    fn validate_inputs_rejects_underscore_prefix() {
        let mut inputs = HashMap::new();
        inputs.insert("_internal".to_owned(), "value".to_owned());
        assert!(matches!(
            validate_inputs(&inputs),
            Err(RunnerError::InvalidInputKey(_))
        ));
    }

    #[test]
    fn validate_inputs_accepts_normal_keys() {
        let mut inputs = HashMap::new();
        inputs.insert("model".to_owned(), "qwen3".to_owned());
        inputs.insert("temperature".to_owned(), "0.7".to_owned());
        assert!(validate_inputs(&inputs).is_ok());
    }

    // ── Integration tests ────────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn two_job_workflow_with_needs() {
        let ns = make_namespace();
        let runner = WorkflowRunner::new(Arc::clone(&ns));

        let mut jobs = HashMap::new();
        jobs.insert(
            "build".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: None,
                env: HashMap::new(),
                steps: vec![Step {
                    name: Some("build-action".into()),
                    id: None,
                    uses: Some("solver/submit".into()),
                    run: None,
                    shell: None,
                    working_directory: None,
                    with: {
                        let mut m = HashMap::new();
                        m.insert("model".into(), "qwen3".into());
                        m
                    },
                    env: HashMap::new(),
                    condition: None,
                    continue_on_error: false,
                }],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );
        jobs.insert(
            "deploy".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: Some(vec!["build".to_owned()]),
                env: HashMap::new(),
                steps: vec![Step {
                    name: Some("deploy-action".into()),
                    id: None,
                    uses: Some("deploy/run".into()),
                    run: None,
                    shell: None,
                    working_directory: None,
                    with: HashMap::new(),
                    env: HashMap::new(),
                    condition: None,
                    continue_on_error: false,
                }],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );

        let workflow = make_simple_workflow(jobs);
        let cancel = CancellationToken::new();
        let result = runner
            .run(&workflow, HashMap::new(), &test_subject(), cancel)
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(result.jobs.len(), 2);
        assert!(result.jobs.contains_key("build"));
        assert!(result.jobs.contains_key("deploy"));
        assert!(result.jobs["build"].success);
        assert!(result.jobs["deploy"].success);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn uses_step_resolves_to_vfs_bin() {
        let ns = make_namespace();
        let runner = WorkflowRunner::new(Arc::clone(&ns));

        let mut jobs = HashMap::new();
        jobs.insert(
            "test".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: None,
                env: HashMap::new(),
                steps: vec![Step {
                    name: Some("submit".into()),
                    id: None,
                    uses: Some("solver/submit@v1".into()),
                    run: None,
                    shell: None,
                    working_directory: None,
                    with: {
                        let mut m = HashMap::new();
                        m.insert("problem".into(), "sat".into());
                        m
                    },
                    env: HashMap::new(),
                    condition: None,
                    continue_on_error: false,
                }],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );

        let workflow = make_simple_workflow(jobs);
        let cancel = CancellationToken::new();
        let result = runner
            .run(&workflow, HashMap::new(), &test_subject(), cancel)
            .await
            .unwrap();

        assert!(result.success);
        let job = &result.jobs["test"];
        assert!(job.success);
        // The MemMount ctl pattern returns "ok: {payload}".
        assert!(job.steps[0].output.starts_with("ok: "));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_step_evaluates_tcl_on_spawn_blocking() {
        let ns = make_namespace();
        let runner = WorkflowRunner::new(Arc::clone(&ns));

        let mut jobs = HashMap::new();
        jobs.insert(
            "script-job".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: None,
                env: HashMap::new(),
                steps: vec![Step {
                    name: Some("compute".into()),
                    id: None,
                    uses: None,
                    run: Some("expr {2 + 3}".into()),
                    shell: None,
                    working_directory: None,
                    with: HashMap::new(),
                    env: HashMap::new(),
                    condition: None,
                    continue_on_error: false,
                }],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );

        let workflow = make_simple_workflow(jobs);
        let cancel = CancellationToken::new();
        let result = runner
            .run(&workflow, HashMap::new(), &test_subject(), cancel)
            .await
            .unwrap();

        assert!(result.success);
        let job = &result.jobs["script-job"];
        assert!(job.success);
        assert_eq!(job.steps[0].output, "5");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_token_stops_workflow() {
        let ns = make_namespace();
        let runner = WorkflowRunner::new(Arc::clone(&ns));

        let mut jobs = HashMap::new();
        // Two waves: "first" runs, then "second" should be skipped.
        jobs.insert(
            "first".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: None,
                env: HashMap::new(),
                steps: vec![Step {
                    name: Some("noop".into()),
                    id: None,
                    uses: Some("solver/submit".into()),
                    run: None,
                    shell: None,
                    working_directory: None,
                    with: HashMap::new(),
                    env: HashMap::new(),
                    condition: None,
                    continue_on_error: false,
                }],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );
        jobs.insert(
            "second".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: Some(vec!["first".to_owned()]),
                env: HashMap::new(),
                steps: vec![Step {
                    name: Some("should-not-run".into()),
                    id: None,
                    uses: Some("solver/submit".into()),
                    run: None,
                    shell: None,
                    working_directory: None,
                    with: HashMap::new(),
                    env: HashMap::new(),
                    condition: None,
                    continue_on_error: false,
                }],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );

        let cancel = CancellationToken::new();

        // Cancel after first wave completes — we do this by spawning the run
        // and cancelling before the second wave.
        let workflow = make_simple_workflow(jobs);

        // Cancel immediately — the first wave may or may not complete.
        cancel.cancel();

        let result = runner
            .run(&workflow, HashMap::new(), &test_subject(), cancel.clone())
            .await;

        // Either Cancelled error or the second job never ran.
        match result {
            Err(RunnerError::Cancelled) => { /* expected */ }
            Ok(r) => {
                // If first wave completed before cancel was checked,
                // the second job should not be present.
                assert!(!r.jobs.contains_key("second"));
            }
            Err(_) => panic!("unexpected error variant"),
        }
    }

    #[test]
    fn action_name_dotdot_rejected() {
        assert!(validate_action_name("../../etc/passwd").is_err());
        assert!(validate_action_name("foo/..bar").is_err());
        assert!(validate_action_name("..").is_err());
    }

    #[test]
    fn input_key_underscore_prefix_rejected() {
        let mut inputs = HashMap::new();
        inputs.insert("_provenance".into(), "injected".into());
        assert!(validate_inputs(&inputs).is_err());

        // Normal key OK.
        let mut inputs2 = HashMap::new();
        inputs2.insert("normal_key".into(), "value".into());
        assert!(validate_inputs(&inputs2).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn step_failure_stops_job() {
        // Use a script step that fails (no VFS mount ambiguity).
        let ns = make_namespace();
        let runner = WorkflowRunner::new(Arc::clone(&ns));

        let mut jobs = HashMap::new();
        jobs.insert(
            "fail-job".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: None,
                env: HashMap::new(),
                steps: vec![
                    Step {
                        name: Some("failing-step".into()),
                        id: None,
                        uses: None,
                        run: Some("error {intentional failure}".into()),
                        shell: None,
                        working_directory: None,
                        with: HashMap::new(),
                        env: HashMap::new(),
                        condition: None,
                        continue_on_error: false,
                    },
                    Step {
                        name: Some("should-not-run".into()),
                        id: None,
                        uses: None,
                        run: Some("expr {1 + 1}".into()),
                        shell: None,
                        working_directory: None,
                        with: HashMap::new(),
                        env: HashMap::new(),
                        condition: None,
                        continue_on_error: false,
                    },
                ],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );

        let workflow = make_simple_workflow(jobs);
        let cancel = CancellationToken::new();
        let result = runner
            .run(&workflow, HashMap::new(), &test_subject(), cancel)
            .await
            .unwrap();

        assert!(!result.success);
        let job = &result.jobs["fail-job"];
        assert!(!job.success);
        // First step failed, second should not have run.
        assert_eq!(job.steps.len(), 1);
        assert!(!job.steps[0].success);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn continue_on_error_allows_next_step() {
        let ns = make_namespace();
        let runner = WorkflowRunner::new(Arc::clone(&ns));

        let mut jobs = HashMap::new();
        jobs.insert(
            "resilient-job".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: None,
                env: HashMap::new(),
                steps: vec![
                    Step {
                        name: Some("failing-step".into()),
                        id: None,
                        uses: None,
                        run: Some("error {intentional failure}".into()),
                        shell: None,
                        working_directory: None,
                        with: HashMap::new(),
                        env: HashMap::new(),
                        condition: None,
                        continue_on_error: true, // Should continue
                    },
                    Step {
                        name: Some("should-run".into()),
                        id: None,
                        uses: None,
                        run: Some("expr {1 + 1}".into()),
                        shell: None,
                        working_directory: None,
                        with: HashMap::new(),
                        env: HashMap::new(),
                        condition: None,
                        continue_on_error: false,
                    },
                ],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );

        let workflow = make_simple_workflow(jobs);
        let cancel = CancellationToken::new();
        let result = runner
            .run(&workflow, HashMap::new(), &test_subject(), cancel)
            .await
            .unwrap();

        let job = &result.jobs["resilient-job"];
        // Both steps should have run.
        assert_eq!(job.steps.len(), 2);
        assert!(!job.steps[0].success); // First failed
        assert!(job.steps[1].success); // Second ran and succeeded
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tcl_state_persists_across_steps_in_job() {
        // Verify that TclShell is reused across steps in the same job
        let ns = make_namespace();
        let runner = WorkflowRunner::new(Arc::clone(&ns));

        let mut jobs = HashMap::new();
        jobs.insert(
            "state-job".to_owned(),
            Job {
                runs_on: super::super::parser::RunsOn::Label("ubuntu".into()),
                needs: None,
                env: HashMap::new(),
                steps: vec![
                    Step {
                        name: Some("set-var".into()),
                        id: None,
                        uses: None,
                        run: Some("set myvar hello".into()),
                        shell: None,
                        working_directory: None,
                        with: HashMap::new(),
                        env: HashMap::new(),
                        condition: None,
                        continue_on_error: false,
                    },
                    Step {
                        name: Some("read-var".into()),
                        id: None,
                        uses: None,
                        run: Some("set myvar".into()),
                        shell: None,
                        working_directory: None,
                        with: HashMap::new(),
                        env: HashMap::new(),
                        condition: None,
                        continue_on_error: false,
                    },
                ],
                condition: None,
                timeout_minutes: None,
                resources: super::super::parser::JobResources::default(),
            },
        );

        let workflow = make_simple_workflow(jobs);
        let cancel = CancellationToken::new();
        let result = runner
            .run(&workflow, HashMap::new(), &test_subject(), cancel)
            .await
            .unwrap();

        assert!(result.success);
        let job = &result.jobs["state-job"];
        assert_eq!(job.steps.len(), 2);
        // Second step should see the variable set by first step
        assert_eq!(job.steps[1].output, "hello");
    }

    // ── #521: run-level concurrency + job/step timeouts ──────────────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn step_timeout_fails_the_step_and_job() {
        // The mount actually sleeps 2s; step_timeout of 1s must trip first.
        let ns = make_namespace_with_slow_action(Duration::from_secs(2));
        let config = WorkflowConfig {
            step_timeout_secs: 1,
            ..Default::default()
        };
        let runner = WorkflowRunner::with_config(ns, &config);

        let mut jobs = HashMap::new();
        jobs.insert("slow-job".to_owned(), single_action_job("slow/step"));
        let workflow = make_simple_workflow(jobs);

        let result = runner
            .run(&workflow, HashMap::new(), &test_subject(), CancellationToken::new())
            .await
            .unwrap();

        assert!(!result.success);
        let job = &result.jobs["slow-job"];
        assert!(!job.success);
        assert!(
            job.steps[0].output.contains("timed out"),
            "expected a timeout message, got: {:?}",
            job.steps[0].output
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn job_timeout_fails_the_whole_job() {
        // step_timeout is generous; job_timeout (wrapping the whole job) is
        // the one that must trip.
        let ns = make_namespace_with_slow_action(Duration::from_secs(2));
        let config = WorkflowConfig {
            job_timeout_secs: 1,
            step_timeout_secs: 600,
            ..Default::default()
        };
        let runner = WorkflowRunner::with_config(ns, &config);

        let mut jobs = HashMap::new();
        jobs.insert("slow-job".to_owned(), single_action_job("slow/step"));
        let workflow = make_simple_workflow(jobs);

        let result = runner
            .run(&workflow, HashMap::new(), &test_subject(), CancellationToken::new())
            .await
            .unwrap();

        assert!(!result.success);
        let job = &result.jobs["slow-job"];
        assert!(!job.success);
        assert_eq!(job.steps[0].name, "<timeout>");
        assert!(job.steps[0].output.contains("job timed out"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn max_concurrent_runs_serializes_concurrent_runs() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let current = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let ns = make_namespace_with_tracking_action(
            current.clone(),
            peak.clone(),
            Duration::from_millis(150),
        );
        let config = WorkflowConfig {
            max_concurrent_runs: 1,
            ..Default::default()
        };
        let runner = Arc::new(WorkflowRunner::with_config(ns, &config));

        let make_workflow = || {
            let mut jobs = HashMap::new();
            jobs.insert("job".to_owned(), single_action_job("track/step"));
            make_simple_workflow(jobs)
        };

        let r1 = {
            let runner = Arc::clone(&runner);
            let wf = make_workflow();
            tokio::spawn(async move {
                runner
                    .run(&wf, HashMap::new(), &test_subject(), CancellationToken::new())
                    .await
            })
        };
        // Give r1 a head start so it acquires the sole run permit first.
        tokio::time::sleep(Duration::from_millis(30)).await;
        let r2 = {
            let runner = Arc::clone(&runner);
            let wf = make_workflow();
            tokio::spawn(async move {
                runner
                    .run(&wf, HashMap::new(), &test_subject(), CancellationToken::new())
                    .await
            })
        };

        let (res1, res2) = tokio::join!(r1, r2);
        assert!(res1.unwrap().unwrap().success);
        assert!(res2.unwrap().unwrap().success);
        assert_eq!(
            peak.load(Ordering::SeqCst),
            1,
            "max_concurrent_runs=1 should have serialized the two runs, never letting \
             both execute their job step at once"
        );
    }
}
