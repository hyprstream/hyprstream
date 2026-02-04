//! GitHub Actions YAML workflow parser
//!
//! Parses `.github/workflows/*.yml` files into Workflow structs.

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

impl Workflow {
    /// Parse a workflow from YAML content
    pub fn parse(yaml: &str) -> Result<Self> {
        let workflow: Workflow = serde_yaml::from_str(yaml)
            .map_err(|e| crate::error::WorkerError::WorkflowParseError(e.to_string()))?;
        Ok(workflow)
    }

    /// Parse a workflow from a file
    pub fn parse_file(path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Self::parse(&content)
    }
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
}
