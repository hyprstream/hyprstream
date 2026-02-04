//! Worker event types for EventService pub/sub.
//!
//! These types mirror the Cap'n Proto WorkerEvent schema and provide:
//! - Rust-friendly struct representations with ToCapnp/FromCapnp derives
//! - A WorkerEvent enum for type-safe event handling
//! - ReceivedEvent wrapper for EventSubscriber integration
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_workers::events::{WorkerEvent, ReceivedEvent, SandboxStarted};
//!
//! // Publishing (in WorkerService)
//! let event = SandboxStarted {
//!     sandbox_id: "abc123".to_owned(),
//!     metadata: "{}".to_owned(),
//!     vm_pid: 1234,
//! };
//! let payload = event.to_bytes()?;
//! publisher.publish("abc123", "started", &payload).await?;
//!
//! // Subscribing (in WorkflowService)
//! let (topic, payload) = subscriber.recv().await?;
//! let event = ReceivedEvent::from_message(&topic, &payload);
//! if let Some(WorkerEvent::SandboxStarted(ss)) = &event.worker_event {
//!     println!("Sandbox {} started with PID {}", ss.sandbox_id, ss.vm_pid);
//! }
//! ```

use anyhow::Result;
use hyprstream_rpc::prelude::{FromCapnp, ToCapnp};

use crate::workers_capnp;

// ═══════════════════════════════════════════════════════════════════════════════
// Individual Event Types (with derive macros)
// ═══════════════════════════════════════════════════════════════════════════════

/// Sandbox started event.
///
/// Published when a Kata VM sandbox is successfully started.
/// Topic format: `worker.{sandbox_id}.started`
#[derive(Debug, Clone, ToCapnp, FromCapnp)]
#[capnp(workers_capnp::sandbox_started)]
pub struct SandboxStarted {
    /// Unique sandbox identifier
    pub sandbox_id: String,
    /// JSON metadata about the sandbox
    pub metadata: String,
    /// VM process ID
    pub vm_pid: u32,
}

/// Sandbox stopped event.
///
/// Published when a Kata VM sandbox stops (gracefully or due to error).
/// Topic format: `worker.{sandbox_id}.stopped`
#[derive(Debug, Clone, ToCapnp, FromCapnp)]
#[capnp(workers_capnp::sandbox_stopped)]
pub struct SandboxStopped {
    /// Unique sandbox identifier
    pub sandbox_id: String,
    /// Reason for stopping (e.g., "completed", "error", "timeout")
    pub reason: String,
    /// VM exit code
    pub exit_code: i32,
}

/// Container started event.
///
/// Published when an OCI container starts within a sandbox.
/// Topic format: `worker.{container_id}.started`
#[derive(Debug, Clone, ToCapnp, FromCapnp)]
#[capnp(workers_capnp::container_started)]
pub struct ContainerStarted {
    /// Unique container identifier
    pub container_id: String,
    /// Parent sandbox identifier
    pub sandbox_id: String,
    /// Container image reference
    pub image: String,
}

/// Container stopped event.
///
/// Published when an OCI container stops within a sandbox.
/// Topic format: `worker.{container_id}.stopped`
#[derive(Debug, Clone, ToCapnp, FromCapnp)]
#[capnp(workers_capnp::container_stopped)]
pub struct ContainerStopped {
    /// Unique container identifier
    pub container_id: String,
    /// Parent sandbox identifier
    pub sandbox_id: String,
    /// Container exit code
    pub exit_code: i32,
    /// Reason for stopping
    pub reason: String,
}

// ═══════════════════════════════════════════════════════════════════════════════
// WorkerEvent Union (manual serialization for enum)
// ═══════════════════════════════════════════════════════════════════════════════

/// Union of all worker event types.
///
/// Represents the Cap'n Proto WorkerEvent union for type-safe event handling.
#[derive(Debug, Clone)]
pub enum WorkerEvent {
    SandboxStarted(SandboxStarted),
    SandboxStopped(SandboxStopped),
    ContainerStarted(ContainerStarted),
    ContainerStopped(ContainerStopped),
}

impl WorkerEvent {
    /// Deserialize from Cap'n Proto bytes.
    ///
    /// Reads the WorkerEvent union and returns the appropriate variant.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        use capnp::message::ReaderOptions;
        use capnp::serialize;
        use workers_capnp::worker_event;

        let reader = serialize::read_message(&mut std::io::Cursor::new(bytes), ReaderOptions::new())?;
        let event = reader.get_root::<worker_event::Reader>()?;

        match event.which()? {
            worker_event::Which::SandboxStarted(r) => {
                Ok(Self::SandboxStarted(SandboxStarted::read_from(r?)?))
            }
            worker_event::Which::SandboxStopped(r) => {
                Ok(Self::SandboxStopped(SandboxStopped::read_from(r?)?))
            }
            worker_event::Which::ContainerStarted(r) => {
                Ok(Self::ContainerStarted(ContainerStarted::read_from(r?)?))
            }
            worker_event::Which::ContainerStopped(r) => {
                Ok(Self::ContainerStopped(ContainerStopped::read_from(r?)?))
            }
        }
    }

    /// Get the event type name (e.g., "started", "stopped").
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::SandboxStarted(_) | Self::ContainerStarted(_) => "started",
            Self::SandboxStopped(_) | Self::ContainerStopped(_) => "stopped",
        }
    }

    /// Get the entity ID (sandbox_id or container_id).
    pub fn entity_id(&self) -> &str {
        match self {
            Self::SandboxStarted(e) => &e.sandbox_id,
            Self::SandboxStopped(e) => &e.sandbox_id,
            Self::ContainerStarted(e) => &e.container_id,
            Self::ContainerStopped(e) => &e.container_id,
        }
    }

    /// Check if this is a sandbox event.
    pub fn is_sandbox_event(&self) -> bool {
        matches!(self, Self::SandboxStarted(_) | Self::SandboxStopped(_))
    }

    /// Check if this is a container event.
    pub fn is_container_event(&self) -> bool {
        matches!(self, Self::ContainerStarted(_) | Self::ContainerStopped(_))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Serialization helpers for publishing
// ═══════════════════════════════════════════════════════════════════════════════

/// Serialize a SandboxStarted event to Cap'n Proto bytes.
pub fn serialize_sandbox_started(event: &SandboxStarted) -> Result<Vec<u8>> {
    hyprstream_rpc::serialize_message(|msg| {
        let builder = msg.init_root::<workers_capnp::worker_event::Builder>();
        let mut started = builder.init_sandbox_started();
        event.write_to(&mut started);
    })
}

/// Serialize a SandboxStopped event to Cap'n Proto bytes.
pub fn serialize_sandbox_stopped(event: &SandboxStopped) -> Result<Vec<u8>> {
    hyprstream_rpc::serialize_message(|msg| {
        let builder = msg.init_root::<workers_capnp::worker_event::Builder>();
        let mut stopped = builder.init_sandbox_stopped();
        event.write_to(&mut stopped);
    })
}

/// Serialize a ContainerStarted event to Cap'n Proto bytes.
pub fn serialize_container_started(event: &ContainerStarted) -> Result<Vec<u8>> {
    hyprstream_rpc::serialize_message(|msg| {
        let builder = msg.init_root::<workers_capnp::worker_event::Builder>();
        let mut started = builder.init_container_started();
        event.write_to(&mut started);
    })
}

/// Serialize a ContainerStopped event to Cap'n Proto bytes.
pub fn serialize_container_stopped(event: &ContainerStopped) -> Result<Vec<u8>> {
    hyprstream_rpc::serialize_message(|msg| {
        let builder = msg.init_root::<workers_capnp::worker_event::Builder>();
        let mut stopped = builder.init_container_stopped();
        event.write_to(&mut stopped);
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// ReceivedEvent - EventSubscriber integration
// ═══════════════════════════════════════════════════════════════════════════════

/// Event received from EventService.
///
/// Wraps the raw topic and payload from EventSubscriber and provides
/// parsed fields for handler dispatch. This replaces the duplicate
/// EventEnvelope type in workflow/triggers.rs.
///
/// # Topic Format
///
/// Topics follow the pattern: `{source}.{entity_id}.{event_type}`
/// - `worker.sandbox123.started` - Sandbox started event
/// - `worker.container456.stopped` - Container stopped event
#[derive(Debug, Clone)]
pub struct ReceivedEvent {
    /// Full topic string (e.g., "worker.sandbox123.started")
    pub topic: String,

    /// Source service (e.g., "worker", "registry", "model")
    pub source: String,

    /// Entity identifier (e.g., sandbox_id, container_id)
    pub entity_id: String,

    /// Event type (e.g., "started", "stopped")
    pub event_type: String,

    /// Deserialized worker event (if source is "worker" and payload is valid)
    pub worker_event: Option<WorkerEvent>,

    /// Event timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl ReceivedEvent {
    /// Create a ReceivedEvent from raw topic and payload.
    ///
    /// Parses the topic into components and attempts to deserialize
    /// the payload for worker events.
    pub fn from_message(topic: &str, payload: &[u8]) -> Self {
        let parts: Vec<&str> = topic.split('.').collect();
        let (source, entity_id, event_type) = if parts.len() >= 3 {
            (
                parts[0].to_owned(),
                parts[1].to_owned(),
                parts[2].to_owned(),
            )
        } else if parts.len() == 2 {
            (parts[0].to_owned(), parts[1].to_owned(), String::new())
        } else {
            (topic.to_owned(), String::new(), String::new())
        };

        // Attempt to deserialize worker events
        let worker_event = if source == "worker" && !payload.is_empty() {
            WorkerEvent::from_bytes(payload).ok()
        } else {
            None
        };

        Self {
            topic: topic.to_owned(),
            source,
            entity_id,
            event_type,
            worker_event,
            timestamp: chrono::Utc::now(),
        }
    }

    /// Check if this event matches a topic pattern.
    ///
    /// Supports prefix matching (e.g., "worker." matches "worker.abc.started").
    pub fn matches_pattern(&self, pattern: &str) -> bool {
        if pattern.ends_with('.') {
            self.topic.starts_with(pattern)
        } else {
            self.topic == pattern
        }
    }

    /// Extract inputs for workflow dispatch.
    ///
    /// Returns a map of input values extracted from the event payload.
    pub fn extract_inputs(&self) -> std::collections::HashMap<String, String> {
        let mut inputs = std::collections::HashMap::new();

        inputs.insert("topic".to_owned(), self.topic.clone());
        inputs.insert("source".to_owned(), self.source.clone());
        inputs.insert("entity_id".to_owned(), self.entity_id.clone());
        inputs.insert("event_type".to_owned(), self.event_type.clone());

        // Add event-specific fields
        if let Some(ref event) = self.worker_event {
            match event {
                WorkerEvent::SandboxStarted(e) => {
                    inputs.insert("sandbox_id".to_owned(), e.sandbox_id.clone());
                    inputs.insert("metadata".to_owned(), e.metadata.clone());
                    inputs.insert("vm_pid".to_owned(), e.vm_pid.to_string());
                }
                WorkerEvent::SandboxStopped(e) => {
                    inputs.insert("sandbox_id".to_owned(), e.sandbox_id.clone());
                    inputs.insert("reason".to_owned(), e.reason.clone());
                    inputs.insert("exit_code".to_owned(), e.exit_code.to_string());
                }
                WorkerEvent::ContainerStarted(e) => {
                    inputs.insert("container_id".to_owned(), e.container_id.clone());
                    inputs.insert("sandbox_id".to_owned(), e.sandbox_id.clone());
                    inputs.insert("image".to_owned(), e.image.clone());
                }
                WorkerEvent::ContainerStopped(e) => {
                    inputs.insert("container_id".to_owned(), e.container_id.clone());
                    inputs.insert("sandbox_id".to_owned(), e.sandbox_id.clone());
                    inputs.insert("exit_code".to_owned(), e.exit_code.to_string());
                    inputs.insert("reason".to_owned(), e.reason.clone());
                }
            }
        }

        inputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_received_event_parsing() {
        let event = ReceivedEvent::from_message("worker.sandbox123.started", &[]);

        assert_eq!(event.source, "worker");
        assert_eq!(event.entity_id, "sandbox123");
        assert_eq!(event.event_type, "started");
        assert!(event.worker_event.is_none()); // Empty payload
    }

    #[test]
    fn test_received_event_pattern_matching() {
        let event = ReceivedEvent::from_message("worker.sandbox123.started", &[]);

        assert!(event.matches_pattern("worker."));
        assert!(event.matches_pattern("worker.sandbox123.started"));
        assert!(!event.matches_pattern("registry."));
    }

    #[test]
    fn test_sandbox_started_roundtrip() -> Result<()> {
        let original = SandboxStarted {
            sandbox_id: "test-123".to_owned(),
            metadata: r#"{"cpu":2}"#.to_owned(),
            vm_pid: 9876,
        };

        let bytes = serialize_sandbox_started(&original)?;
        let event = WorkerEvent::from_bytes(&bytes)?;

        match event {
            WorkerEvent::SandboxStarted(ss) => {
                assert_eq!(ss.sandbox_id, "test-123");
                assert_eq!(ss.metadata, r#"{"cpu":2}"#);
                assert_eq!(ss.vm_pid, 9876);
            }
            _ => panic!("Expected SandboxStarted"),
        }
        Ok(())
    }

    #[test]
    fn test_container_stopped_roundtrip() -> Result<()> {
        let original = ContainerStopped {
            container_id: "cont-456".to_owned(),
            sandbox_id: "sb-123".to_owned(),
            exit_code: 0,
            reason: "completed".to_owned(),
        };

        let bytes = serialize_container_stopped(&original)?;
        let event = WorkerEvent::from_bytes(&bytes)?;

        match event {
            WorkerEvent::ContainerStopped(cs) => {
                assert_eq!(cs.container_id, "cont-456");
                assert_eq!(cs.sandbox_id, "sb-123");
                assert_eq!(cs.exit_code, 0);
                assert_eq!(cs.reason, "completed");
            }
            _ => panic!("Expected ContainerStopped"),
        }
        Ok(())
    }

    #[test]
    fn test_extract_inputs() -> Result<()> {
        let original = SandboxStarted {
            sandbox_id: "sb-789".to_owned(),
            metadata: "{}".to_owned(),
            vm_pid: 1234,
        };
        let bytes = serialize_sandbox_started(&original)?;

        let event = ReceivedEvent::from_message("worker.sb-789.started", &bytes);
        let inputs = event.extract_inputs();

        assert_eq!(inputs.get("sandbox_id"), Some(&"sb-789".to_owned()));
        assert_eq!(inputs.get("vm_pid"), Some(&"1234".to_owned()));
        assert_eq!(inputs.get("event_type"), Some(&"started".to_owned()));
        Ok(())
    }
}
