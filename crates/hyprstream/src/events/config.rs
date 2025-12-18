//! Sink configuration loading from YAML
//!
//! Configuration file format:
//! ```yaml
//! sinks:
//!   - name: self-supervised-trainer
//!     type: in_process
//!     handler: training::self_supervised::on_generation_complete
//!     subscribe: "inference.generation_complete"
//!
//!   - name: quality-alerts
//!     type: webhook
//!     url: "https://hooks.slack.com/..."
//!     headers:
//!       Authorization: "Bearer ${SLACK_TOKEN}"
//!     subscribe: "metrics.threshold_breach"
//! ```

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info, warn};

/// Top-level sinks configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SinksConfig {
    /// List of sink configurations
    #[serde(default)]
    pub sinks: Vec<SinkConfig>,
}

impl SinksConfig {
    /// Load configuration from a YAML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("failed to read config file {}: {}", path.display(), e))?;

        Self::from_yaml(&content)
    }

    /// Parse configuration from YAML string
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        let config: SinksConfig = serde_yaml::from_str(yaml)
            .map_err(|e| anyhow!("failed to parse YAML config: {}", e))?;

        // Expand environment variables in config
        let config = config.expand_env_vars();

        Ok(config)
    }

    /// Expand environment variables in configuration values
    fn expand_env_vars(mut self) -> Self {
        for sink in &mut self.sinks {
            sink.expand_env_vars();
        }
        self
    }

    /// Load configuration from default locations
    ///
    /// Searches in order:
    /// 1. `.registry/event_sinks.yaml` (project-local)
    /// 2. `$XDG_CONFIG_HOME/hyprstream/event_sinks.yaml`
    /// 3. `~/.config/hyprstream/event_sinks.yaml`
    pub fn load_default() -> Result<Self> {
        let paths = [
            Some(".registry/event_sinks.yaml".into()),
            xdg::BaseDirectories::new()
                .ok()
                .and_then(|xdg| xdg.find_config_file("hyprstream/event_sinks.yaml")),
            dirs::config_dir().map(|p| p.join("hyprstream/event_sinks.yaml")),
        ];

        for path in paths.into_iter().flatten() {
            if path.exists() {
                info!("loading sink config from {}", path.display());
                return Self::load(&path);
            }
        }

        debug!("no sink configuration found, using defaults");
        Ok(Self::default())
    }
}

/// Configuration for a single sink
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinkConfig {
    /// Unique sink name
    pub name: String,

    /// Sink type and type-specific configuration
    #[serde(flatten)]
    pub sink_type: SinkType,

    /// Topic filter for ZeroMQ prefix matching
    /// - `""` subscribes to all events
    /// - `"inference"` subscribes to all inference events
    /// - `"inference.generation_complete"` subscribes to specific event
    pub subscribe: String,
}

impl SinkConfig {
    /// Expand environment variables in configuration values
    fn expand_env_vars(&mut self) {
        match &mut self.sink_type {
            SinkType::Webhook { url, headers } => {
                *url = expand_env(url);
                if let Some(headers) = headers {
                    for value in headers.values_mut() {
                        *value = expand_env(value);
                    }
                }
            }
            SinkType::Nats { url, .. } => {
                *url = expand_env(url);
            }
            SinkType::Mcp { endpoint, .. } => {
                *endpoint = expand_env(endpoint);
            }
            _ => {}
        }
    }
}

/// Sink type configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SinkType {
    /// In-process handler (for self-supervised training, etc.)
    InProcess {
        /// Handler identifier (e.g., "training::self_supervised::on_generation_complete")
        handler: String,
    },

    /// HTTP webhook sink
    Webhook {
        /// Webhook URL
        url: String,
        /// Optional HTTP headers
        #[serde(default)]
        headers: Option<HashMap<String, String>>,
    },

    /// NATS JetStream forwarder
    Nats {
        /// NATS server URL
        url: String,
        /// Subject prefix (events published as `{prefix}.{topic}`)
        subject_prefix: String,
    },

    /// MCP tool invocation
    Mcp {
        /// Tool name to invoke
        tool: String,
        /// MCP endpoint (IPC or TCP)
        endpoint: String,
    },

    /// Container execution
    Container {
        /// Container image
        image: String,
        /// Optional container runtime (default: auto-detect)
        #[serde(default)]
        runtime: Option<String>,
    },
}

/// Expand environment variables in a string
///
/// Supports `${VAR}` and `$VAR` syntax.
fn expand_env(s: &str) -> String {
    shellexpand::env(s).unwrap_or_else(|_| s.into()).into_owned()
}

/// Helper module for dirs (XDG fallback)
mod dirs {
    use std::path::PathBuf;

    pub fn config_dir() -> Option<PathBuf> {
        std::env::var("HOME")
            .ok()
            .map(|h| PathBuf::from(h).join(".config"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_yaml() {
        let yaml = r#"
sinks:
  - name: test-webhook
    type: webhook
    url: "https://example.com/hook"
    subscribe: "inference"

  - name: test-in-process
    type: in_process
    handler: "test::handler"
    subscribe: ""
"#;

        let config = SinksConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.sinks.len(), 2);
        assert_eq!(config.sinks[0].name, "test-webhook");
        assert_eq!(config.sinks[1].name, "test-in-process");
    }

    #[test]
    fn test_webhook_config() {
        let yaml = r#"
sinks:
  - name: slack-alerts
    type: webhook
    url: "https://hooks.slack.com/services/xxx"
    headers:
      Content-Type: "application/json"
      Authorization: "Bearer token"
    subscribe: "metrics.threshold_breach"
"#;

        let config = SinksConfig::from_yaml(yaml).unwrap();
        let sink = &config.sinks[0];

        assert_eq!(sink.name, "slack-alerts");
        assert_eq!(sink.subscribe, "metrics.threshold_breach");

        match &sink.sink_type {
            SinkType::Webhook { url, headers } => {
                assert_eq!(url, "https://hooks.slack.com/services/xxx");
                let headers = headers.as_ref().unwrap();
                assert_eq!(headers.get("Content-Type").unwrap(), "application/json");
            }
            _ => panic!("expected webhook sink"),
        }
    }

    #[test]
    fn test_nats_config() {
        let yaml = r#"
sinks:
  - name: nats-bridge
    type: nats
    url: "nats://localhost:4222"
    subject_prefix: "hyprstream.events"
    subscribe: ""
"#;

        let config = SinksConfig::from_yaml(yaml).unwrap();
        let sink = &config.sinks[0];

        match &sink.sink_type {
            SinkType::Nats {
                url,
                subject_prefix,
            } => {
                assert_eq!(url, "nats://localhost:4222");
                assert_eq!(subject_prefix, "hyprstream.events");
            }
            _ => panic!("expected nats sink"),
        }
    }

    #[test]
    fn test_mcp_config() {
        let yaml = r#"
sinks:
  - name: mcp-validator
    type: mcp
    tool: "validate_output"
    endpoint: "ipc:///tmp/mcp-validator.sock"
    subscribe: "git2db.commit_created"
"#;

        let config = SinksConfig::from_yaml(yaml).unwrap();
        let sink = &config.sinks[0];

        match &sink.sink_type {
            SinkType::Mcp { tool, endpoint } => {
                assert_eq!(tool, "validate_output");
                assert_eq!(endpoint, "ipc:///tmp/mcp-validator.sock");
            }
            _ => panic!("expected mcp sink"),
        }
    }

    #[test]
    fn test_container_config() {
        let yaml = r#"
sinks:
  - name: sandbox-processor
    type: container
    image: "hyprstream/event-processor:latest"
    runtime: "podman"
    subscribe: "training"
"#;

        let config = SinksConfig::from_yaml(yaml).unwrap();
        let sink = &config.sinks[0];

        match &sink.sink_type {
            SinkType::Container { image, runtime } => {
                assert_eq!(image, "hyprstream/event-processor:latest");
                assert_eq!(runtime.as_deref(), Some("podman"));
            }
            _ => panic!("expected container sink"),
        }
    }

    #[test]
    fn test_env_expansion() {
        std::env::set_var("TEST_URL", "https://test.example.com");

        let yaml = r#"
sinks:
  - name: test
    type: webhook
    url: "${TEST_URL}/webhook"
    subscribe: "test"
"#;

        let config = SinksConfig::from_yaml(yaml).unwrap();
        let sink = &config.sinks[0];

        match &sink.sink_type {
            SinkType::Webhook { url, .. } => {
                assert_eq!(url, "https://test.example.com/webhook");
            }
            _ => panic!("expected webhook sink"),
        }
    }

    #[test]
    fn test_empty_config() {
        let yaml = "sinks: []";
        let config = SinksConfig::from_yaml(yaml).unwrap();
        assert!(config.sinks.is_empty());
    }

    #[test]
    fn test_default_config() {
        let config = SinksConfig::default();
        assert!(config.sinks.is_empty());
    }
}
