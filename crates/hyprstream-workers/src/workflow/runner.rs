//! Workflow runner - executes workflows via WorkerService
//!
//! Creates PodSandboxes and runs steps as containers.

use std::collections::HashMap;

use crate::error::Result;
use crate::runtime::{
    ContainerConfig, KeyValue, PodSandboxConfig, WorkerService,
};

use super::client::{JobRun, RunStatus, StepRun, WorkflowRun};
use super::parser::{Job, Step, Workflow};
use super::WorkflowId;

/// Workflow runner that executes workflows using WorkerService
pub struct WorkflowRunner {
    /// Reference to WorkerService for container operations
    worker_service: std::sync::Arc<WorkerService>,
}

impl WorkflowRunner {
    /// Create a new workflow runner
    pub fn new(worker_service: std::sync::Arc<WorkerService>) -> Self {
        Self { worker_service }
    }

    /// Execute a workflow
    pub async fn run(
        &self,
        workflow_id: &WorkflowId,
        workflow: &Workflow,
        inputs: HashMap<String, String>,
    ) -> Result<WorkflowRun> {
        let run_id = uuid::Uuid::new_v4().to_string();

        let mut run = WorkflowRun {
            id: run_id.clone(),
            workflow_id: workflow_id.clone(),
            status: RunStatus::InProgress,
            started_at: Some(chrono::Utc::now()),
            completed_at: None,
            jobs: HashMap::new(),
        };

        tracing::info!(
            workflow_id = %workflow_id,
            run_id = %run_id,
            "Starting workflow execution"
        );

        // Execute jobs in dependency order
        let job_order = self.resolve_job_order(workflow)?;

        for job_name in job_order {
            let job = &workflow.jobs[&job_name];
            let job_run = self.run_job(&job_name, job, &inputs).await?;

            // Check if job failed
            if job_run.status == RunStatus::Failure {
                run.status = RunStatus::Failure;
                run.completed_at = Some(chrono::Utc::now());
                run.jobs.insert(job_name, job_run);
                return Ok(run);
            }

            run.jobs.insert(job_name, job_run);
        }

        run.status = RunStatus::Success;
        run.completed_at = Some(chrono::Utc::now());

        tracing::info!(
            workflow_id = %workflow_id,
            run_id = %run_id,
            "Workflow execution completed"
        );

        Ok(run)
    }

    /// Run a single job
    async fn run_job(
        &self,
        job_name: &str,
        job: &Job,
        inputs: &HashMap<String, String>,
    ) -> Result<JobRun> {
        tracing::info!(job = %job_name, "Starting job");

        // Create a sandbox for this job
        let sandbox_config = PodSandboxConfig {
            metadata: crate::runtime::PodSandboxMetadata {
                name: format!("job-{job_name}"),
                uid: uuid::Uuid::new_v4().to_string(),
                namespace: "default".to_owned(),
                attempt: 0,
            },
            labels: HashMap::from([
                ("workflow.job".to_owned(), job_name.to_owned()),
            ]),
            ..Default::default()
        };

        let sandbox_id = self.worker_service.run_pod_sandbox(&sandbox_config).await?;

        let mut job_run = JobRun {
            name: job_name.to_owned(),
            status: RunStatus::InProgress,
            steps: Vec::new(),
        };

        // Execute each step
        for (idx, step) in job.steps.iter().enumerate() {
            let step_name = step.name.clone().unwrap_or_else(|| format!("step-{idx}"));

            let step_run = self.run_step(&sandbox_id, &step_name, step, inputs).await?;

            // Check if step failed
            if step_run.status == RunStatus::Failure && !step.continue_on_error {
                job_run.status = RunStatus::Failure;
                job_run.steps.push(step_run);
                break;
            }

            job_run.steps.push(step_run);
        }

        if job_run.status != RunStatus::Failure {
            job_run.status = RunStatus::Success;
        }

        // Clean up sandbox
        self.worker_service.stop_pod_sandbox(&sandbox_id).await?;
        self.worker_service.remove_pod_sandbox(&sandbox_id).await?;

        tracing::info!(job = %job_name, status = ?job_run.status, "Job completed");

        Ok(job_run)
    }

    /// Run a single step
    async fn run_step(
        &self,
        sandbox_id: &str,
        step_name: &str,
        step: &Step,
        inputs: &HashMap<String, String>,
    ) -> Result<StepRun> {
        tracing::debug!(step = %step_name, "Running step");

        let mut step_run = StepRun {
            name: step_name.to_owned(),
            status: RunStatus::InProgress,
            exit_code: None,
        };

        // Determine step type and execute
        if let Some(run_cmd) = &step.run {
            // Shell command
            let expanded_cmd = self.expand_variables(run_cmd, inputs);
            step_run = self.run_shell_step(sandbox_id, step_name, &expanded_cmd, step).await?;
        } else if let Some(uses) = &step.uses {
            // Action
            step_run = self.run_action_step(sandbox_id, step_name, uses, step, inputs).await?;
        }

        Ok(step_run)
    }

    /// Run a shell command step
    async fn run_shell_step(
        &self,
        sandbox_id: &str,
        step_name: &str,
        command: &str,
        step: &Step,
    ) -> Result<StepRun> {
        let sandbox_config = PodSandboxConfig::default();

        // Create container for the shell command
        let container_config = ContainerConfig {
            metadata: crate::runtime::ContainerMetadata {
                name: format!("step-{step_name}"),
                attempt: 0,
            },
            command: vec![
                step.shell.clone().unwrap_or_else(|| "/bin/sh".to_owned()),
                "-c".to_owned(),
            ],
            args: vec![command.to_owned()],
            working_dir: step.working_directory.clone().unwrap_or_default(),
            envs: step.env.iter().map(|(k, v)| {
                KeyValue {
                    key: k.clone(),
                    value: v.clone(),
                }
            }).collect(),
            ..Default::default()
        };

        let container_id = self.worker_service
            .create_container(sandbox_id, &container_config, &sandbox_config)
            .await?;

        self.worker_service.start_container(&container_id).await?;

        // Wait for completion via exec_sync
        // In practice, we'd monitor the container state
        let result = self.worker_service
            .exec_sync(&container_id, &["/bin/true".to_owned()], 600)
            .await?;

        // Clean up container
        self.worker_service.stop_container(&container_id, 30).await?;
        self.worker_service.remove_container(&container_id).await?;

        Ok(StepRun {
            name: step_name.to_owned(),
            status: if result.exit_code == 0 { RunStatus::Success } else { RunStatus::Failure },
            exit_code: Some(result.exit_code),
        })
    }

    /// Run an action step
    async fn run_action_step(
        &self,
        _sandbox_id: &str,
        step_name: &str,
        action: &str,
        step: &Step,
        _inputs: &HashMap<String, String>,
    ) -> Result<StepRun> {
        // TODO: Implement action loading and execution
        // 1. Parse action reference (org/repo@version)
        // 2. Clone action repository
        // 3. Parse action.yml
        // 4. Execute action entrypoint

        tracing::info!(
            step = %step_name,
            action = %action,
            inputs = ?step.with,
            "Running action step (stub)"
        );

        Ok(StepRun {
            name: step_name.to_owned(),
            status: RunStatus::Success,
            exit_code: Some(0),
        })
    }

    /// Resolve job execution order based on dependencies
    fn resolve_job_order(&self, workflow: &Workflow) -> Result<Vec<String>> {
        // Simple topological sort based on 'needs'
        let mut result = Vec::new();
        let mut remaining: Vec<_> = workflow.jobs.keys().cloned().collect();
        let mut completed: std::collections::HashSet<String> = std::collections::HashSet::new();

        while !remaining.is_empty() {
            let mut made_progress = false;

            remaining.retain(|job_name| {
                let job = &workflow.jobs[job_name];
                let needs_satisfied = job.needs.as_ref().is_none_or(|needs| {
                    needs.iter().all(|n| completed.contains(n))
                });

                if needs_satisfied {
                    result.push(job_name.clone());
                    completed.insert(job_name.clone());
                    made_progress = true;
                    false // Remove from remaining
                } else {
                    true // Keep in remaining
                }
            });

            if !made_progress && !remaining.is_empty() {
                return Err(crate::error::WorkerError::WorkflowParseError(
                    "Circular dependency in job 'needs'".to_owned(),
                ));
            }
        }

        Ok(result)
    }

    /// Expand variables in a string (e.g., ${{ inputs.model }})
    fn expand_variables(&self, template: &str, inputs: &HashMap<String, String>) -> String {
        let mut result = template.to_owned();

        // Simple variable expansion for ${{ inputs.NAME }}
        for (key, value) in inputs {
            let pattern = format!("${{{{ inputs.{key} }}}}");
            result = result.replace(&pattern, value);
        }

        result
    }
}
