//! Event types for hyprstream
//!
//! This module defines event types for P2P messaging between hyprstream instances.
//! Events will be delivered via DHT-based peer discovery (future work).
//!
//! # Topics
//!
//! Events use dot-notation topics:
//! - `inference.generation_complete` - Generation completed with metrics
//! - `inference.generation_failed` - Generation failed with error
//! - `metrics.threshold_breach` - Quality metric threshold breached
//! - `metrics.window_rollover` - Time window closed with stats
//! - `training.started` - Training job started
//! - `training.completed` - Training job completed
//! - `training.checkpoint_saved` - Checkpoint committed to git
//! - `git2db.repository_cloned` - Repository cloned
//! - `git2db.commit_created` - Commit created
//! - `git2db.adapter_saved` - Adapter saved to repository

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Source of an event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSource {
    /// LLM inference events
    Inference,
    /// Metrics aggregation events
    Metrics,
    /// Training job events
    Training,
    /// Git repository events
    Git2db,
}

impl std::fmt::Display for EventSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventSource::Inference => write!(f, "inference"),
            EventSource::Metrics => write!(f, "metrics"),
            EventSource::Training => write!(f, "training"),
            EventSource::Git2db => write!(f, "git2db"),
        }
    }
}

/// Envelope containing event metadata and payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEnvelope {
    /// Unique event ID
    pub id: Uuid,

    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Optional correlation ID for event chains
    /// (e.g., generation_complete → threshold_breach → training_started)
    pub correlation_id: Option<Uuid>,

    /// Event source
    pub source: EventSource,

    /// Topic for routing (e.g., "inference.generation_complete")
    pub topic: String,

    /// Event payload
    pub payload: EventPayload,
}

impl EventEnvelope {
    /// Create a new event envelope
    pub fn new(source: EventSource, topic: impl Into<String>, payload: EventPayload) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            correlation_id: None,
            source,
            topic: topic.into(),
            payload,
        }
    }

    /// Create a new event with correlation to a previous event
    pub fn with_correlation(
        source: EventSource,
        topic: impl Into<String>,
        payload: EventPayload,
        correlation_id: Uuid,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            correlation_id: Some(correlation_id),
            source,
            topic: topic.into(),
            payload,
        }
    }
}

/// Event payload variants
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventPayload {
    // ─────────────────────────────────────────────────────────────────────
    // Inference Events
    // ─────────────────────────────────────────────────────────────────────
    /// Generation completed successfully
    GenerationComplete {
        /// Model identifier
        model_id: String,
        /// Session ID (for multi-turn conversations)
        session_id: Option<String>,
        /// Quality metrics from generation
        metrics: GenerationMetrics,
    },

    /// Generation failed
    GenerationFailed {
        /// Model identifier
        model_id: String,
        /// Session ID
        session_id: Option<String>,
        /// Error message
        error: String,
        /// Error code (optional)
        error_code: Option<String>,
    },

    /// Tool was executed during generation
    ToolExecuted {
        /// Model identifier
        model_id: String,
        /// Tool name
        tool_name: String,
        /// Tool execution result (success/failure)
        success: bool,
        /// Execution time in milliseconds
        duration_ms: u64,
    },

    // ─────────────────────────────────────────────────────────────────────
    // Metrics Events
    // ─────────────────────────────────────────────────────────────────────
    /// Quality threshold breached
    ThresholdBreach {
        /// Model identifier
        model_id: String,
        /// Metric name (perplexity, entropy, repetition_ratio)
        metric: String,
        /// Configured threshold
        threshold: f64,
        /// Actual value
        actual: f64,
        /// Z-score (standard deviations from baseline)
        z_score: f64,
    },

    /// Time window rolled over with statistics
    WindowRollover {
        /// Model identifier
        model_id: String,
        /// Window statistics
        stats: WindowedStats,
    },

    /// Baseline established for model
    BaselineReady {
        /// Model identifier
        model_id: String,
        /// Number of samples in baseline
        sample_count: u64,
        /// Baseline statistics
        stats: WindowedStats,
    },

    // ─────────────────────────────────────────────────────────────────────
    // Training Events
    // ─────────────────────────────────────────────────────────────────────
    /// Training job started
    TrainingStarted {
        /// Model identifier
        model_id: String,
        /// Adapter identifier
        adapter_id: String,
        /// Training configuration
        config: TrainingConfig,
    },

    /// Training job completed
    TrainingCompleted {
        /// Model identifier
        model_id: String,
        /// Adapter identifier
        adapter_id: String,
        /// Total training steps
        steps: u64,
        /// Final loss value
        final_loss: f32,
    },

    /// Checkpoint saved to repository
    CheckpointSaved {
        /// Model identifier
        model_id: String,
        /// Checkpoint identifier
        checkpoint_id: String,
        /// Git commit hash
        commit_hash: String,
    },

    // ─────────────────────────────────────────────────────────────────────
    // Git2db Events
    // ─────────────────────────────────────────────────────────────────────
    /// Repository cloned
    RepositoryCloned {
        /// Repository ID in registry
        repo_id: String,
        /// Repository name
        name: String,
        /// Source URL
        url: String,
    },

    /// Branch created
    BranchCreated {
        /// Repository ID
        repo_id: String,
        /// Branch name
        branch_name: String,
        /// Base commit hash
        base_commit: String,
    },

    /// Worktree created
    WorktreeCreated {
        /// Repository ID
        repo_id: String,
        /// Branch name
        branch_name: String,
        /// Worktree path
        path: String,
    },

    /// Commit created
    CommitCreated {
        /// Repository ID
        repo_id: String,
        /// Commit hash
        hash: String,
        /// Commit message
        message: String,
    },

    /// Adapter saved to repository
    AdapterSaved {
        /// Repository ID
        repo_id: String,
        /// Adapter name
        adapter_name: String,
        /// Adapter index
        index: u32,
    },

    /// Adapter loaded from repository
    AdapterLoaded {
        /// Repository ID
        repo_id: String,
        /// Adapter name
        adapter_name: String,
        /// Adapter index
        index: u32,
    },
}

/// Generation quality metrics (matches GenerationQualityMetrics)
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct GenerationMetrics {
    /// Perplexity: exp(-avg_log_prob)
    pub perplexity: f32,
    /// Average entropy of probability distributions
    pub avg_entropy: f32,
    /// Variance of entropy across tokens
    pub entropy_variance: f32,
    /// Ratio of repeated n-grams
    pub repetition_ratio: f32,
    /// Number of tokens generated
    pub token_count: u32,
    /// Tokens generated per second
    pub tokens_per_second: f32,
    /// Total generation time in milliseconds
    pub generation_time_ms: u64,
}

impl From<crate::runtime::generation_metrics::GenerationQualityMetrics> for GenerationMetrics {
    fn from(m: crate::runtime::generation_metrics::GenerationQualityMetrics) -> Self {
        Self {
            perplexity: m.perplexity,
            avg_entropy: m.avg_entropy,
            entropy_variance: m.entropy_variance,
            repetition_ratio: m.repetition_ratio,
            token_count: m.token_count,
            // These fields are set separately at generation time
            tokens_per_second: 0.0,
            generation_time_ms: 0,
        }
    }
}

impl GenerationMetrics {
    /// Create metrics with timing information from a GenerationQualityMetrics
    pub fn with_timing(
        m: crate::runtime::generation_metrics::GenerationQualityMetrics,
        tokens_per_second: f32,
        generation_time_ms: u64,
    ) -> Self {
        Self {
            perplexity: m.perplexity,
            avg_entropy: m.avg_entropy,
            entropy_variance: m.entropy_variance,
            repetition_ratio: m.repetition_ratio,
            token_count: m.token_count,
            tokens_per_second,
            generation_time_ms,
        }
    }
}

/// Windowed statistics from Welford's algorithm
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct WindowedStats {
    /// Window duration in seconds
    pub window_seconds: u64,
    /// Sample count
    pub sample_count: u64,
    /// Mean perplexity
    pub perplexity_mean: f64,
    /// Perplexity standard deviation
    pub perplexity_stddev: f64,
    /// Mean entropy
    pub entropy_mean: f64,
    /// Entropy standard deviation
    pub entropy_stddev: f64,
}

/// Training configuration summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// LoRA rank
    pub rank: u32,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: u32,
    /// Maximum steps
    pub max_steps: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_envelope_creation() {
        let event = EventEnvelope::new(
            EventSource::Inference,
            "inference.generation_complete",
            EventPayload::GenerationComplete {
                model_id: "qwen3-small".to_owned(),
                session_id: None,
                metrics: GenerationMetrics::default(),
            },
        );

        assert_eq!(event.source, EventSource::Inference);
        assert_eq!(event.topic, "inference.generation_complete");
        assert!(event.correlation_id.is_none());
    }

    #[test]
    fn test_event_serialization() {
        let event = EventEnvelope::new(
            EventSource::Metrics,
            "metrics.threshold_breach",
            EventPayload::ThresholdBreach {
                model_id: "qwen3-small".to_owned(),
                metric: "perplexity".to_owned(),
                threshold: 50.0,
                actual: 75.0,
                z_score: 2.5,
            },
        );

        let json = serde_json::to_string(&event).expect("test: serialize event");
        let deserialized: EventEnvelope = serde_json::from_str(&json).expect("test: deserialize event");

        assert_eq!(deserialized.topic, "metrics.threshold_breach");
    }

    #[test]
    fn test_event_correlation() {
        let first = EventEnvelope::new(
            EventSource::Metrics,
            "metrics.threshold_breach",
            EventPayload::ThresholdBreach {
                model_id: "qwen3-small".to_owned(),
                metric: "perplexity".to_owned(),
                threshold: 50.0,
                actual: 75.0,
                z_score: 2.5,
            },
        );

        let second = EventEnvelope::with_correlation(
            EventSource::Training,
            "training.started",
            EventPayload::TrainingStarted {
                model_id: "qwen3-small".to_owned(),
                adapter_id: "auto-fix".to_owned(),
                config: TrainingConfig {
                    rank: 16,
                    learning_rate: 1e-4,
                    batch_size: 4,
                    max_steps: Some(100),
                },
            },
            first.id,
        );

        assert_eq!(second.correlation_id, Some(first.id));
    }
}
