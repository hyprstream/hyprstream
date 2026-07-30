//! GitHub Actions YAML workflow parser
//!
//! Parses `.github/workflows/*.yml` files into Workflow structs.
//!
//! ## Strict vs legacy mode (#1432)
//!
//! Two parse entry points exist:
//!
//! - [`Workflow::parse`] — **legacy / lenient** mode. Unknown YAML keys are
//!   silently dropped (the historical behavior). This is the compatibility
//!   path for non-gate callers and is unchanged by #1432.
//! - [`Workflow::parse_strict`] — **strict / fail-closed** mode. Unknown
//!   top-level, job-level, and step-level keys are rejected with
//!   [`WorkerError::WorkflowParseError`](crate::error::WorkerError). This is
//!   the mode the merge gate's workflow loader must use: a silently-ignored
//!   `permissions:`, `concurrency:`, or `services:` would be a security gap.
//!
//! ### Explicitly supported subset (strict mode)
//!
//! Strict mode accepts *only* the keys this parser models:
//!
//! | Level    | Allowed keys |
//! |----------|--------------|
//! | workflow | `name`, `on`, `env`, `jobs` |
//! | job      | `runs-on`, `needs`, `env`, `steps`, `if`, `timeout-minutes`, `resources` |
//! | step     | `name`, `id`, `uses`, `run`, `shell`, `working-directory`, `with`, `env`, `if`, `continue-on-error` |
//!
//! `resources` is a hyprstream-specific extension (see [`JobResources`]),
//! not a GHA key; it is permitted in strict mode so operator gate workflows
//! may declare resource hints.
//!
//! ### Non-goals (#1432 Phase 2)
//!
//! Strict mode rejects **unknown structural keys** — it does **not** claim
//! full GitHub Actions compatibility, and it deliberately does not implement
//! `matrix`/`strategy`, `services`, `permissions`, `concurrency`, expression
//! evaluation, environment scopes, or forge-specific checks. Those features
//! remain Phase 2; until then they correctly fail closed in strict mode and
//! are silently ignored in legacy mode.
//!
//! Free-form data maps (`env`, `with`, and trigger sub-keys like `branches`,
//! `tags`, `paths`, `inputs`) are not key-checked — they carry arbitrary
//! data, not parser-modeled schema.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::Result;

/// GitHub Actions compatible workflow definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Workflow name
    pub name: String,

    /// Trigger configuration
    #[serde(rename = "on")]
    pub on: WorkflowTrigger,

    /// Global environment variables
    #[serde(default)]
    pub env: HashMap<String, String>,

    /// Workflow jobs
    pub jobs: HashMap<String, Job>,
}

/// Workflow trigger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WorkflowTrigger {
    /// Simple trigger (single event)
    Simple(String),

    /// List of triggers
    List(Vec<String>),

    /// Complex trigger configuration
    Complex(HashMap<String, TriggerConfig>),
}

/// Trigger configuration for an event type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TriggerConfig {
    /// No additional config
    None,

    /// Branch/tag filters
    BranchFilter {
        #[serde(default)]
        branches: Vec<String>,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        paths: Vec<String>,
    },

    /// Workflow dispatch inputs
    WorkflowDispatch {
        #[serde(default)]
        inputs: HashMap<String, InputDef>,
    },
}

/// Input definition for workflow_dispatch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputDef {
    /// Description
    #[serde(default)]
    pub description: String,

    /// Required input
    #[serde(default)]
    pub required: bool,

    /// Default value
    #[serde(default)]
    pub default: Option<String>,

    /// Input type
    #[serde(rename = "type", default)]
    pub input_type: Option<String>,
}

/// Job definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    /// Runner label (e.g., "ubuntu-latest", "hyprstream-gpu")
    #[serde(rename = "runs-on")]
    pub runs_on: RunsOn,

    /// Job dependencies
    #[serde(default)]
    pub needs: Option<Vec<String>>,

    /// Job environment variables
    #[serde(default)]
    pub env: HashMap<String, String>,

    /// Job steps
    pub steps: Vec<Step>,

    /// Condition for running this job
    #[serde(rename = "if", default)]
    pub condition: Option<String>,

    /// Timeout in minutes
    #[serde(rename = "timeout-minutes", default)]
    pub timeout_minutes: Option<u32>,

    /// Resource hints for scheduling (hyprstream extension, not part of the
    /// GHA syntax). Consumed by the P2 admission engine (#525) via
    /// `workflow::scheduler` to derive a `Demand` for the job's sandbox
    /// reservation. All-zero (the default) means "no resource requirement" —
    /// existing workflow YAML with no `resources:` key is unaffected.
    #[serde(default)]
    pub resources: JobResources,
}

/// Runner specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RunsOn {
    /// Single runner label
    Label(String),

    /// Multiple runner labels (AND logic)
    Labels(Vec<String>),
}

/// Resource hints a job declares for scheduling (hyprstream extension).
///
/// GHA has no native `resources:` job key — this is a hyprstream-specific
/// addition, parsed only when present (`#[serde(default)]` on the `Job`
/// field), so it never affects existing workflow YAML. See
/// `workflow::scheduler::job_pod_sandbox_config` for how these map onto the
/// #525 admission engine's `Demand`.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct JobResources {
    /// Requested CPU, in millicores (e.g. `500` = half a core).
    pub cpu_millis: u64,
    /// Requested memory, in bytes.
    pub memory_bytes: u64,
    /// Requested GPU count.
    pub gpu: usize,
}

/// Step definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Step {
    /// Step name
    #[serde(default)]
    pub name: Option<String>,

    /// Step ID for outputs
    #[serde(default)]
    pub id: Option<String>,

    /// Action reference (uses)
    #[serde(default)]
    pub uses: Option<String>,

    /// Shell command (run)
    #[serde(default)]
    pub run: Option<String>,

    /// Shell to use
    #[serde(default)]
    pub shell: Option<String>,

    /// Working directory
    #[serde(rename = "working-directory", default)]
    pub working_directory: Option<String>,

    /// Action inputs
    #[serde(default, rename = "with")]
    pub with: HashMap<String, String>,

    /// Step environment variables
    #[serde(default)]
    pub env: HashMap<String, String>,

    /// Condition for running this step
    #[serde(rename = "if", default)]
    pub condition: Option<String>,

    /// Continue on error
    #[serde(rename = "continue-on-error", default)]
    pub continue_on_error: bool,
}

/// Parse strictness.
///
/// The merge gate **must** parse operator-authored workflows in
/// [`ParseMode::Strict`]: a silently-dropped `permissions:`, `concurrency:`,
/// or `services:` key is a security gap (the workflow appears to parse cleanly
/// while quietly losing semantics). Existing non-gate callers stay on
/// [`ParseMode::Legacy`] until they explicitly opt in — see the non-goals of
/// <https://github.com/hyprstream/hyprstream/issues/1432>.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseMode {
    /// Legacy / lenient compatibility mode (the historical `Workflow::parse`
    /// behavior). Unknown top-level, job-level, and step-level YAML keys are
    /// silently ignored. Use only for callers that must tolerate arbitrary
    /// upstream GitHub Actions YAML; do **not** use this for the merge gate.
    Legacy,

    /// Strict / fail-closed mode. Unknown keys at the top level, inside a job,
    /// or inside a step are rejected with
    /// [`WorkerError::WorkflowParseError`](crate::error::WorkerError). This is
    /// the mode the gate's workflow loader uses by default.
    ///
    /// Note: this rejects *unknown structural keys* — it does **not** claim
    /// full GitHub Actions compatibility. Unsupported-but-recognized GHA
    /// features (`matrix`, `services`, `permissions`, `concurrency`,
    /// expression evaluation, forge-specific checks) are deliberately out of
    /// scope (#1432 Phase 2).
    Strict,
}

/// Top-level workflow keys the parser knows about (wire names, post-`rename`).
const KNOWN_WORKFLOW_KEYS: &[&str] = &["name", "on", "env", "jobs"];

/// Job-level keys the parser knows about (wire names, post-`rename`).
///
/// `resources` is a hyprstream-specific extension, not a GHA key; it is
/// included here so operator gate workflows that declare resource hints still
/// parse in strict mode.
const KNOWN_JOB_KEYS: &[&str] = &[
    "runs-on",
    "needs",
    "env",
    "steps",
    "if",
    "timeout-minutes",
    "resources",
];

/// Step-level keys the parser knows about (wire names, post-`rename`).
const KNOWN_STEP_KEYS: &[&str] = &[
    "name",
    "id",
    "uses",
    "run",
    "shell",
    "working-directory",
    "with",
    "env",
    "if",
    "continue-on-error",
];

impl Workflow {
    /// Parse a workflow from YAML content in **legacy** mode.
    ///
    /// This is the historical entry point: unknown keys are silently dropped.
    /// Existing non-gate callers should keep using this until they explicitly
    /// opt into strict mode. The merge gate should use [`Workflow::parse_strict`]
    /// (or [`Workflow::parse_with`] with [`ParseMode::Strict`]).
    pub fn parse(yaml: &str) -> Result<Self> {
        Self::parse_with(yaml, ParseMode::Legacy)
    }

    /// Parse a workflow from YAML content in **strict / fail-closed** mode.
    ///
    /// Rejects unknown top-level, job-level, and step-level keys. See
    /// [`ParseMode::Strict`] for the exact contract and non-goals.
    pub fn parse_strict(yaml: &str) -> Result<Self> {
        Self::parse_with(yaml, ParseMode::Strict)
    }

    /// Parse a workflow with an explicit [`ParseMode`].
    ///
    /// In [`ParseMode::Strict`] the document is first validated against the
    /// known-key sets at the top level, per-job, and per-step (the three
    /// structural levels the parser models) before being handed to
    /// `serde_yaml`. Validation runs on the raw [`serde_yaml::Value`] tree so
    /// that an unknown key can never be silently absorbed by
    /// `#[serde(default)]`.
    pub fn parse_with(yaml: &str, mode: ParseMode) -> Result<Self> {
        let value: serde_yaml::Value = serde_yaml::from_str(yaml)
            .map_err(|e| crate::error::WorkerError::WorkflowParseError(e.to_string()))?;

        if mode == ParseMode::Strict {
            validate_known_keys(&value)?;
        }

        let workflow: Workflow = serde_yaml::from_value(value)
            .map_err(|e| crate::error::WorkerError::WorkflowParseError(e.to_string()))?;
        Ok(workflow)
    }

    /// Parse a workflow from a file in **legacy** mode.
    pub fn parse_file(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::parse(&content)
    }

    /// Parse a workflow from a file with an explicit [`ParseMode`].
    pub fn parse_file_with(path: &std::path::Path, mode: ParseMode) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::parse_with(&content, mode)
    }
}

/// Reject unknown structural keys at the top level, per-job, and per-step.
///
/// `env`, `with`, and trigger sub-keys (`branches`, `tags`, `paths`,
/// `inputs`, …) are free-form data maps, not schema levels, so they are not
/// key-checked here.
fn validate_known_keys(value: &serde_yaml::Value) -> Result<()> {
    let mapping = value
        .as_mapping()
        .ok_or_else(|| crate::error::WorkerError::WorkflowParseError(
            String::from("workflow root must be a mapping"),
        ))?;

    for (key, val) in mapping {
        let key_str = match key.as_str() {
            Some(s) => s,
            None => {
                return Err(crate::error::WorkerError::WorkflowParseError(format!(
                    "top-level key is not a string: {key:?}"
                )))
            }
        };

        if !KNOWN_WORKFLOW_KEYS.contains(&key_str) {
            return Err(crate::error::WorkerError::WorkflowParseError(format!(
                "unknown top-level key `{key_str}` (strict mode: allowed keys are {})",
                KNOWN_WORKFLOW_KEYS.join(", ")
            )));
        }

        // Only `jobs` has a structural sub-level to recurse into;
        // `on`, `env`, `name` are free-form / scalar.
        if key_str == "jobs" {
            validate_jobs(val)?;
        }
    }
    Ok(())
}

/// Validate each job mapping and its steps.
fn validate_jobs(jobs: &serde_yaml::Value) -> Result<()> {
    let mapping = jobs
        .as_mapping()
        .ok_or_else(|| crate::error::WorkerError::WorkflowParseError(
            String::from("`jobs` must be a mapping of job-id → job"),
        ))?;

    for (job_id, job) in mapping {
        let job_id_str = job_id.as_str().unwrap_or("<non-string job id>");
        let job_map = job.as_mapping().ok_or_else(|| {
            crate::error::WorkerError::WorkflowParseError(format!(
                "job `{job_id_str}` must be a mapping"
            ))
        })?;

        for (key, val) in job_map {
            let key_str = match key.as_str() {
                Some(s) => s,
                None => {
                    return Err(crate::error::WorkerError::WorkflowParseError(format!(
                        "job `{job_id_str}` has a non-string key: {key:?}"
                    )))
                }
            };

            if !KNOWN_JOB_KEYS.contains(&key_str) {
                return Err(crate::error::WorkerError::WorkflowParseError(format!(
                    "unknown key `{key_str}` in job `{job_id_str}` (strict mode: allowed keys are {})",
                    KNOWN_JOB_KEYS.join(", ")
                )));
            }

            if key_str == "steps" {
                validate_steps(job_id_str, val)?;
            }
        }
    }
    Ok(())
}

/// Validate each step in a job's `steps` sequence.
fn validate_steps(job_id: &str, steps: &serde_yaml::Value) -> Result<()> {
    let seq = steps.as_sequence().ok_or_else(|| {
        crate::error::WorkerError::WorkflowParseError(format!(
            "`steps` in job `{job_id}` must be a sequence"
        ))
    })?;

    for (idx, step) in seq.iter().enumerate() {
        let step_map = step.as_mapping().ok_or_else(|| {
            crate::error::WorkerError::WorkflowParseError(format!(
                "step[{idx}] in job `{job_id}` must be a mapping"
            ))
        })?;

        for key in step_map.keys() {
            let key_str = match key.as_str() {
                Some(s) => s,
                None => {
                    return Err(crate::error::WorkerError::WorkflowParseError(format!(
                        "step[{idx}] in job `{job_id}` has a non-string key: {key:?}"
                    )))
                }
            };

            if !KNOWN_STEP_KEYS.contains(&key_str) {
                return Err(crate::error::WorkerError::WorkflowParseError(format!(
                    "unknown key `{key_str}` in step[{idx}] of job `{job_id}` (strict mode: allowed keys are {})",
                    KNOWN_STEP_KEYS.join(", ")
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_workflow() -> Result<()> {
        let yaml = r#"
name: Test Workflow
on: push
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: Build
        run: cargo build
"#;

        let workflow = Workflow::parse(yaml)?;
        assert_eq!(workflow.name, "Test Workflow");
        assert!(workflow.jobs.contains_key("build"));
        Ok(())
    }

    #[test]
    fn test_parse_complex_trigger() -> Result<()> {
        let yaml = r#"
name: Complex Workflow
on:
  push:
    branches:
      - main
      - 'release/*'
  workflow_dispatch:
    inputs:
      model:
        description: Model to train
        required: true
jobs:
  train:
    runs-on: hyprstream-gpu
    steps:
      - uses: hyprstream/model-load@v1
        with:
          model: ${{ inputs.model }}
"#;

        let workflow = Workflow::parse(yaml)?;
        assert_eq!(workflow.name, "Complex Workflow");
        Ok(())
    }

    // ─── strict-mode acceptance: the known subset parses cleanly ───────────

    fn minimal_gate_yaml() -> &'static str {
        // The operator-authored gate workflow subset: only keys the parser
        // models. Must parse without error in strict mode.
        r#"
name: Gate
on:
  pull_request:
    branches:
      - main
env:
  RUST_BACKTRACE: "1"
jobs:
  gate:
    runs-on: ubuntu-latest
    needs: []
    if: ${{ github.event.pull_request.draft == false }}
    timeout-minutes: 30
    env:
      CARGO_TERM_COLOR: always
    steps:
      - name: Checkout
        id: checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: "0"
        env:
          GHA: "true"
        if: ${{ true }}
        continue-on-error: false
        working-directory: .
        shell: bash
      - name: Build
        run: cargo build
"#
    }

    #[test]
    fn test_strict_accepts_known_subset() -> Result<()> {
        let workflow = Workflow::parse_strict(minimal_gate_yaml())?;
        assert_eq!(workflow.name, "Gate");
        assert!(workflow.jobs.contains_key("gate"));
        // Legacy mode agrees.
        let legacy = Workflow::parse(minimal_gate_yaml())?;
        assert_eq!(legacy.jobs.len(), workflow.jobs.len());
        Ok(())
    }

    #[test]
    fn test_strict_accepts_resource_hints_extension() -> Result<()> {
        // `resources:` is a hyprstream extension; must be permitted in strict.
        let yaml = r#"
name: GPU
on: push
jobs:
  train:
    runs-on: hyprstream-gpu
    resources:
      cpu_millis: 4000
      memory_bytes: 8589934592
      gpu: 1
    steps:
      - run: python train.py
"#;
        let workflow = Workflow::parse_strict(yaml)?;
        assert_eq!(workflow.jobs["train"].resources.gpu, 1);
        Ok(())
    }

    #[test]
    fn test_parse_with_legacy_mode_matches_parse() -> Result<()> {
        let a = Workflow::parse(minimal_gate_yaml())?;
        let b = Workflow::parse_with(minimal_gate_yaml(), ParseMode::Legacy)?;
        assert_eq!(a.name, b.name);
        assert_eq!(a.jobs.len(), b.jobs.len());
        Ok(())
    }

    // ─── strict-mode rejections: unknown keys at each structural level ─────

    fn assert_strict_rejects(yaml: &str, needle: &str) {
        match Workflow::parse_strict(yaml) {
            Err(crate::error::WorkerError::WorkflowParseError(msg)) => {
                assert!(
                    msg.contains(needle),
                    "expected strict error to mention `{needle}`, got: {msg}"
                );
            }
            other => panic!("strict mode should reject, got: {other:?}"),
        }
    }

    #[test]
    fn test_strict_rejects_unknown_top_level_services() {
        let yaml = r#"
name: T
on: push
services:
  db:
    image: postgres
jobs:
  b:
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
"#;
        assert_strict_rejects(yaml, "unknown top-level key `services`");
    }

    #[test]
    fn test_strict_rejects_unknown_top_level_permissions() {
        let yaml = r#"
name: T
on: push
permissions:
  contents: read
jobs:
  b:
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
"#;
        assert_strict_rejects(yaml, "unknown top-level key `permissions`");
    }

    #[test]
    fn test_strict_rejects_unknown_top_level_concurrency() {
        let yaml = r#"
name: T
on: push
concurrency:
  group: deploy
jobs:
  b:
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
"#;
        assert_strict_rejects(yaml, "unknown top-level key `concurrency`");
    }

    #[test]
    fn test_strict_rejects_unknown_job_level_matrix() {
        let yaml = r#"
name: T
on: push
jobs:
  b:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        os: [ubuntu, macos]
    steps:
      - run: echo hi
"#;
        assert_strict_rejects(yaml, "unknown key `strategy` in job `b`");
    }

    #[test]
    fn test_strict_rejects_unknown_job_level_services() {
        let yaml = r#"
name: T
on: push
jobs:
  b:
    runs-on: ubuntu-latest
    services:
      db:
        image: postgres
    steps:
      - run: echo hi
"#;
        assert_strict_rejects(yaml, "unknown key `services` in job `b`");
    }

    #[test]
    fn test_strict_rejects_unknown_step_level_key() {
        let yaml = r#"
name: T
on: push
jobs:
  b:
    runs-on: ubuntu-latest
    steps:
      - name: x
        run: echo hi
        timeout-minutes: 5
"#;
        // `timeout-minutes` is a known *job* key, not a known step key here.
        assert_strict_rejects(yaml, "unknown key `timeout-minutes` in step[0]");
    }

    #[test]
    fn test_strict_rejects_unknown_step_level_tty() {
        let yaml = r#"
name: T
on: push
jobs:
  b:
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
        with:
          key: value
"#;
        // `with` IS a known step key — this should actually be accepted.
        // Sanity: ensure we don't over-reject known step keys.
        match Workflow::parse_strict(yaml) {
            Ok(w) => assert_eq!(w.name, "T"),
            Err(e) => panic!("strict mode must accept `with` step key: {e}"),
        }
    }

    // ─── legacy mode still silently tolerates the same inputs ──────────────

    #[test]
    fn test_legacy_silently_drops_unknown_top_level() -> Result<()> {
        let yaml = r#"
name: T
on: push
permissions:
  contents: read
jobs:
  b:
    runs-on: ubuntu-latest
    steps:
      - run: echo hi
"#;
        // Legacy mode must NOT reject — that is its defined contract.
        let workflow = Workflow::parse(yaml)?;
        assert_eq!(workflow.name, "T");
        Ok(())
    }

    #[test]
    fn test_legacy_silently_drops_unknown_job_and_step_keys() -> Result<()> {
        let yaml = r#"
name: T
on: push
jobs:
  b:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
    steps:
      - run: echo hi
        bogus-key: oops
"#;
        let workflow = Workflow::parse(yaml)?;
        assert!(workflow.jobs.contains_key("b"));
        Ok(())
    }

    // ─── structural-error edge cases ───────────────────────────────────────

    #[test]
    fn test_strict_rejects_non_mapping_root() {
        assert_strict_rejects("not a mapping", "workflow root must be a mapping");
    }

    #[test]
    fn test_strict_rejects_non_mapping_job() {
        let yaml = r#"
name: T
on: push
jobs:
  b: not-a-mapping
"#;
        assert_strict_rejects(yaml, "job `b` must be a mapping");
    }
}
