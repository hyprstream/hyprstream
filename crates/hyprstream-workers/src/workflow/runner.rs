//! Workflow runner — executes workflows through the VFS namespace.
//!
//! The WorkflowRunner is a general-purpose execution engine that:
//! - Topologically sorts jobs by `needs:` dependencies into execution waves
//! - Runs independent jobs in each wave concurrently via `tokio::JoinSet`
//! - Resolves `uses:` action steps through VFS `/bin/` ctl files
//! - Evaluates `run:` script steps via TclShell on `spawn_blocking`
//! - Propagates CancellationToken for cooperative cancellation

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;

use hyprstream_tcl::TclShell;
use hyprstream_vfs::proxy::spawn_vfs_proxy;
use hyprstream_vfs::{Namespace, Subject};

use super::parser::{Job, Step, Workflow};

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
#[derive(Clone)]
pub struct WorkflowRunner {
    ns: Arc<Namespace>,
    script_semaphore: Arc<Semaphore>,
}

impl WorkflowRunner {
    /// Create a new WorkflowRunner.
    ///
    /// * `ns` — the root namespace (mounts for `/bin/`, `/env/`, `/srv/`, etc.)
    /// * Default concurrency cap: 64 concurrent script-executing jobs.
    pub fn new(ns: Arc<Namespace>) -> Self {
        Self {
            ns,
            script_semaphore: Arc::new(Semaphore::new(64)),
        }
    }

    /// Run a complete workflow.
    ///
    /// Jobs are topologically sorted by `needs:` into execution waves.
    /// Independent jobs in each wave run concurrently. The `cancel` token
    /// is checked between waves and propagated to individual jobs.
    pub async fn run(
        &self,
        workflow: &Workflow,
        inputs: HashMap<String, String>,
        subject: &Subject,
        cancel: CancellationToken,
    ) -> RunnerResult<WorkflowRunResult> {
        // Validate inputs: reject _-prefixed keys (provenance injection prevention).
        validate_inputs(&inputs)?;

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
                let subject = subject.clone();
                let cancel = cancel.clone();
                let env = merge_env(&workflow.env, &job.env);
                let inputs = inputs.clone();
                let name = job_name.clone();

                join_set.spawn(async move {
                    let result = run_job(ns, sem, &job, &name, env, &inputs, &subject, cancel).await;
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

/// Run a single job. Steps execute sequentially within a job.
///
/// For script steps (`run:`), a single TclShell is reused across all
/// sequential steps in the same job (constructed inside `spawn_blocking`).
async fn run_job(
    ns: Arc<Namespace>,
    semaphore: Arc<Semaphore>,
    job: &Job,
    job_name: &str,
    env: HashMap<String, String>,
    inputs: &HashMap<String, String>,
    subject: &Subject,
    cancel: CancellationToken,
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

        // Spawn VFS proxy BEFORE spawn_blocking (needs tokio runtime).
        let vfs_tx = spawn_vfs_proxy(Arc::clone(&forked), subject.clone());

        // Collect steps to process on the blocking thread.
        let steps: Vec<Step> = job.steps.clone();
        let subject_clone = subject.clone();
        let ns_for_actions = Arc::clone(&ns);
        let subject_for_actions = subject.clone();

        // For jobs with mixed action + script steps, we process action steps
        // on the async runtime and script steps on spawn_blocking.
        // Since steps are sequential, we interleave.
        // Spawn a single blocking thread with one TclShell for the job.
        // Script steps are sent to it via channel; action steps run on async runtime.
        // This reuses the interpreter across sequential steps (Tcl state carries over).
        //
        // Channel types: tokio::sync::mpsc for the dispatch channel (async send,
        // blocking_recv on spawn_blocking side). tokio::sync::oneshot for replies
        // (blocking send on spawn_blocking, async recv on the runtime). This avoids
        // blocking tokio worker threads with std::sync::mpsc::recv().
        let (script_tx, mut script_rx) = tokio::sync::mpsc::channel::<(
            String,                                                          // script
            HashMap<String, String>,                                         // step env
            tokio::sync::oneshot::Sender<Result<String, String>>,            // reply
        )>(1);

        let shell_handle = {
            let vfs_tx = vfs_tx.clone();
            let subject_clone = subject_clone.clone();
            tokio::task::spawn_blocking(move || {
                let mut shell = TclShell::new(subject_clone, vfs_tx);
                while let Some((script, step_env, reply)) = script_rx.blocking_recv() {
                    // Set step-level env vars via the interpreter API (not eval)
                    // to prevent Tcl injection from brace-unbalancing in values.
                    for (k, v) in &step_env {
                        let _ = shell.set_env(k, v);
                    }
                    let result = shell.eval(&script);
                    let _ = reply.send(result);
                }
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
                // Action step — runs on async runtime.
                let result =
                    run_action_step(action, &step.with, &ns_for_actions, &subject_for_actions)
                        .await;
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
                match reply_rx.await {
                    Ok(Ok(output)) => {
                        step_results.push(StepRunResult {
                            name: step_name,
                            success: true,
                            output,
                        });
                    }
                    Ok(Err(e)) => {
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
                    Err(_) => {
                        step_results.push(StepRunResult {
                            name: step_name,
                            success: false,
                            output: "TclShell thread panicked".into(),
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
                let result = run_action_step(action, &step.with, &ns, subject).await;
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
        Ok(hyprstream_vfs::Stat {
            qtype: 0,
            size: 0,
            name: inner.path.clone(),
            mtime: 0,
        })
    }

    async fn clunk(&self, _fid: hyprstream_vfs::Fid, _caller: &Subject) {}
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::unwrap_used)]
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
            Ok(Stat {
                qtype: 0,
                size: 0,
                name: inner.path.clone(),
                mtime: 0,
            })
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
}
