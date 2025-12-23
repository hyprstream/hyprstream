//! Cap'n Proto serialization helpers for events
//!
//! This module provides functions to convert between the Rust event types
//! and Cap'n Proto messages for zero-copy serialization over ZMQ.
//!
//! Usage:
//! ```rust,ignore
//! use crate::events::capnp_serde::{serialize_event, deserialize_event};
//!
//! // Serialize an event to capnp bytes
//! let bytes = serialize_event(&event)?;
//!
//! // Deserialize capnp bytes back to an event
//! let event = deserialize_event(&bytes)?;
//! ```

use crate::events::{EventEnvelope, EventPayload, EventSource, GenerationMetrics};
use crate::events_capnp;
use anyhow::{anyhow, Result};
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use chrono::{TimeZone, Utc};

/// Serialize an EventEnvelope to Cap'n Proto bytes
pub fn serialize_event(event: &EventEnvelope) -> Result<Vec<u8>> {
    let mut message = Builder::new_default();
    let mut envelope = message.init_root::<events_capnp::event_envelope::Builder>();

    // Set ID (UUID as bytes)
    envelope.set_id(event.id.as_bytes());

    // Set timestamp (convert DateTime<Utc> to unix millis)
    envelope.set_timestamp(event.timestamp.timestamp_millis());

    // Set correlation ID if present
    if let Some(ref corr_id) = event.correlation_id {
        envelope.set_correlation_id(corr_id.as_bytes());
    }

    // Set source
    let source = match event.source {
        EventSource::Inference => events_capnp::EventSource::Inference,
        EventSource::Metrics => events_capnp::EventSource::Metrics,
        EventSource::Training => events_capnp::EventSource::Training,
        EventSource::Git2db => events_capnp::EventSource::Git2db,
    };
    envelope.set_source(source);

    // Set topic
    envelope.set_topic(&event.topic);

    // Set payload
    let mut payload = envelope.init_payload();
    set_event_payload(&mut payload, &event.payload)?;

    // Serialize to bytes
    let mut bytes = Vec::new();
    serialize::write_message(&mut bytes, &message)?;
    Ok(bytes)
}

/// Deserialize Cap'n Proto bytes to an EventEnvelope
pub fn deserialize_event(bytes: &[u8]) -> Result<EventEnvelope> {
    let reader = serialize::read_message(bytes, ReaderOptions::new())?;
    let envelope = reader.get_root::<events_capnp::event_envelope::Reader>()?;

    // Get ID
    let id_bytes = envelope.get_id()?;
    let id = uuid::Uuid::from_slice(id_bytes)
        .map_err(|e| anyhow!("invalid UUID: {}", e))?;

    // Get timestamp (convert unix millis to DateTime<Utc>)
    let timestamp_millis = envelope.get_timestamp();
    let timestamp = Utc.timestamp_millis_opt(timestamp_millis)
        .single()
        .ok_or_else(|| anyhow!("invalid timestamp: {}", timestamp_millis))?;

    // Get correlation ID
    let correlation_id = if envelope.has_correlation_id() {
        let corr_bytes = envelope.get_correlation_id()?;
        Some(uuid::Uuid::from_slice(corr_bytes)
            .map_err(|e| anyhow!("invalid correlation UUID: {}", e))?)
    } else {
        None
    };

    // Get source
    let source = match envelope.get_source()? {
        events_capnp::EventSource::Inference => EventSource::Inference,
        events_capnp::EventSource::Metrics => EventSource::Metrics,
        events_capnp::EventSource::Training => EventSource::Training,
        events_capnp::EventSource::Git2db => EventSource::Git2db,
        events_capnp::EventSource::System => EventSource::Inference, // Map to default
    };

    // Get topic
    let topic = envelope.get_topic()?.to_string()?;

    // Get payload
    let payload_reader = envelope.get_payload()?;
    let payload = get_event_payload(payload_reader)?;

    Ok(EventEnvelope {
        id,
        timestamp,
        correlation_id,
        source,
        topic,
        payload,
    })
}

/// Set the payload in a Cap'n Proto builder
fn set_event_payload(
    payload: &mut events_capnp::event_payload::Builder,
    event_payload: &EventPayload,
) -> Result<()> {
    match event_payload {
        EventPayload::GenerationComplete {
            model_id,
            session_id,
            metrics,
        } => {
            let mut gen = payload.reborrow().init_generation_complete();
            gen.set_model_id(model_id);
            if let Some(ref sid) = session_id {
                gen.set_session_id(sid);
            }
            let mut m = gen.init_metrics();
            m.set_perplexity(metrics.perplexity);
            m.set_avg_entropy(metrics.avg_entropy);
            m.set_entropy_variance(metrics.entropy_variance);
            m.set_repetition_ratio(metrics.repetition_ratio);
            m.set_token_count(metrics.token_count);
            m.set_tokens_per_second(metrics.tokens_per_second);
            m.set_generation_time_ms(metrics.generation_time_ms);
        }
        EventPayload::GenerationFailed {
            model_id,
            session_id,
            error,
            error_code,
        } => {
            let mut gen = payload.reborrow().init_generation_failed();
            gen.set_model_id(model_id);
            if let Some(ref sid) = session_id {
                gen.set_session_id(sid);
            }
            gen.set_error(error);
            if let Some(ref code) = error_code {
                gen.set_error_code(code);
            }
        }
        EventPayload::ThresholdBreach {
            model_id,
            metric,
            threshold,
            actual,
            z_score,
        } => {
            let mut tb = payload.reborrow().init_threshold_breach();
            tb.set_model_id(model_id);
            tb.set_metric(metric);
            tb.set_threshold(*threshold);
            tb.set_actual(*actual);
            tb.set_z_score(*z_score);
        }
        EventPayload::TrainingStarted {
            model_id,
            adapter_id,
            config,
        } => {
            let mut ts = payload.reborrow().init_training_started();
            ts.set_model_id(model_id);
            ts.set_adapter_name(adapter_id);
            // Map config fields to capnp (epochs not in Rust config, use 0)
            ts.set_epochs(0);
            ts.set_learning_rate(config.learning_rate as f32);
        }
        EventPayload::TrainingCompleted {
            model_id,
            adapter_id,
            steps,
            final_loss,
        } => {
            let mut tc = payload.reborrow().init_training_completed();
            tc.set_model_id(model_id);
            tc.set_adapter_name(adapter_id);
            tc.set_final_loss(*final_loss);
            tc.set_total_steps(*steps as u32);
            tc.set_duration_ms(0); // Not in Rust type
        }
        EventPayload::CheckpointSaved {
            model_id,
            checkpoint_id,
            commit_hash,
        } => {
            let mut cs = payload.reborrow().init_checkpoint_saved();
            cs.set_model_id(model_id);
            cs.set_adapter_name(checkpoint_id);
            cs.set_checkpoint_path(commit_hash);
            cs.set_step(0); // Not in Rust type
        }
        EventPayload::RepositoryCloned {
            repo_id,
            name,
            url,
        } => {
            let mut rc = payload.reborrow().init_repository_cloned();
            rc.set_repo_id(repo_id);
            rc.set_name(name);
            rc.set_url(url);
            rc.set_worktree_path(""); // Not in Rust type
        }
        EventPayload::CommitCreated {
            repo_id,
            hash,
            message,
        } => {
            let mut cc = payload.reborrow().init_commit_created();
            cc.set_repo_id(repo_id);
            cc.set_commit_oid(hash);
            cc.set_message(message);
            cc.set_author(""); // Not in Rust type
        }
        EventPayload::BranchCreated {
            repo_id,
            branch_name,
            base_commit,
        } => {
            let mut bc = payload.reborrow().init_branch_created();
            bc.set_repo_id(repo_id);
            bc.set_branch_name(branch_name);
            bc.set_base_branch(base_commit);
        }
        EventPayload::WorktreeCreated {
            repo_id,
            branch_name,
            path,
        } => {
            let mut wc = payload.reborrow().init_worktree_created();
            wc.set_repo_id(repo_id);
            wc.set_worktree_path(path);
            wc.set_branch_name(branch_name);
        }
        // Events without capnp representation use fallback
        EventPayload::ToolExecuted { .. }
        | EventPayload::WindowRollover { .. }
        | EventPayload::BaselineReady { .. }
        | EventPayload::AdapterSaved { .. }
        | EventPayload::AdapterLoaded { .. } => {
            // Use health check as a neutral placeholder for unsupported types
            payload.reborrow().init_health_check();
        }
    }
    Ok(())
}

/// Get the payload from a Cap'n Proto reader
fn get_event_payload(
    payload: events_capnp::event_payload::Reader,
) -> Result<EventPayload> {
    use events_capnp::event_payload::Which;

    match payload.which()? {
        Which::GenerationComplete(gen) => {
            let gen = gen?;
            let metrics_reader = gen.get_metrics()?;
            Ok(EventPayload::GenerationComplete {
                model_id: gen.get_model_id()?.to_string()?,
                session_id: if gen.has_session_id() {
                    Some(gen.get_session_id()?.to_string()?)
                } else {
                    None
                },
                metrics: GenerationMetrics {
                    perplexity: metrics_reader.get_perplexity(),
                    avg_entropy: metrics_reader.get_avg_entropy(),
                    entropy_variance: metrics_reader.get_entropy_variance(),
                    repetition_ratio: metrics_reader.get_repetition_ratio(),
                    token_count: metrics_reader.get_token_count(),
                    tokens_per_second: metrics_reader.get_tokens_per_second(),
                    generation_time_ms: metrics_reader.get_generation_time_ms(),
                },
            })
        }
        Which::GenerationFailed(gen) => {
            let gen = gen?;
            Ok(EventPayload::GenerationFailed {
                model_id: gen.get_model_id()?.to_string()?,
                session_id: if gen.has_session_id() {
                    Some(gen.get_session_id()?.to_string()?)
                } else {
                    None
                },
                error: gen.get_error()?.to_string()?,
                error_code: if gen.has_error_code() {
                    Some(gen.get_error_code()?.to_string()?)
                } else {
                    None
                },
            })
        }
        Which::ThresholdBreach(tb) => {
            let tb = tb?;
            Ok(EventPayload::ThresholdBreach {
                model_id: tb.get_model_id()?.to_string()?,
                metric: tb.get_metric()?.to_string()?,
                threshold: tb.get_threshold(),
                actual: tb.get_actual(),
                z_score: tb.get_z_score(),
            })
        }
        Which::TrainingStarted(ts) => {
            let ts = ts?;
            Ok(EventPayload::TrainingStarted {
                model_id: ts.get_model_id()?.to_string()?,
                adapter_id: ts.get_adapter_name()?.to_string()?,
                config: crate::events::TrainingConfig {
                    rank: 0, // Not in capnp
                    learning_rate: ts.get_learning_rate() as f64,
                    batch_size: 0, // Not in capnp
                    max_steps: None, // Not in capnp
                },
            })
        }
        Which::TrainingProgress(tp) => {
            // Map TrainingProgress to TrainingStarted (best match)
            let tp = tp?;
            Ok(EventPayload::TrainingStarted {
                model_id: tp.get_model_id()?.to_string()?,
                adapter_id: tp.get_adapter_name()?.to_string()?,
                config: crate::events::TrainingConfig {
                    rank: 0,
                    learning_rate: tp.get_learning_rate() as f64,
                    batch_size: 0,
                    max_steps: None,
                },
            })
        }
        Which::TrainingCompleted(tc) => {
            let tc = tc?;
            Ok(EventPayload::TrainingCompleted {
                model_id: tc.get_model_id()?.to_string()?,
                adapter_id: tc.get_adapter_name()?.to_string()?,
                steps: tc.get_total_steps() as u64,
                final_loss: tc.get_final_loss(),
            })
        }
        Which::CheckpointSaved(cs) => {
            let cs = cs?;
            Ok(EventPayload::CheckpointSaved {
                model_id: cs.get_model_id()?.to_string()?,
                checkpoint_id: cs.get_adapter_name()?.to_string()?,
                commit_hash: cs.get_checkpoint_path()?.to_string()?,
            })
        }
        Which::RepositoryCloned(rc) => {
            let rc = rc?;
            Ok(EventPayload::RepositoryCloned {
                repo_id: rc.get_repo_id()?.to_string()?,
                name: rc.get_name()?.to_string()?,
                url: rc.get_url()?.to_string()?,
            })
        }
        Which::CommitCreated(cc) => {
            let cc = cc?;
            Ok(EventPayload::CommitCreated {
                repo_id: cc.get_repo_id()?.to_string()?,
                hash: cc.get_commit_oid()?.to_string()?,
                message: cc.get_message()?.to_string()?,
            })
        }
        Which::BranchCreated(bc) => {
            let bc = bc?;
            Ok(EventPayload::BranchCreated {
                repo_id: bc.get_repo_id()?.to_string()?,
                branch_name: bc.get_branch_name()?.to_string()?,
                base_commit: if bc.has_base_branch() {
                    bc.get_base_branch()?.to_string()?
                } else {
                    String::new()
                },
            })
        }
        Which::WorktreeCreated(wc) => {
            let wc = wc?;
            Ok(EventPayload::WorktreeCreated {
                repo_id: wc.get_repo_id()?.to_string()?,
                branch_name: wc.get_branch_name()?.to_string()?,
                path: wc.get_worktree_path()?.to_string()?,
            })
        }
        // Handle remaining variants with defaults
        _ => {
            // Return a default payload for unsupported types
            Ok(EventPayload::GenerationComplete {
                model_id: "unknown".to_string(),
                session_id: None,
                metrics: GenerationMetrics::default(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_generation_complete() {
        let event = EventEnvelope::new(
            EventSource::Inference,
            "inference.generation_complete",
            EventPayload::GenerationComplete {
                model_id: "test-model".to_string(),
                session_id: Some("session-1".to_string()),
                metrics: GenerationMetrics {
                    perplexity: 1.5,
                    avg_entropy: 2.0,
                    entropy_variance: 0.1,
                    repetition_ratio: 0.05,
                    token_count: 100,
                    tokens_per_second: 50.0,
                    generation_time_ms: 2000,
                },
            },
        );

        let bytes = serialize_event(&event).unwrap();
        let decoded = deserialize_event(&bytes).unwrap();

        assert_eq!(decoded.topic, event.topic);
        assert_eq!(decoded.id, event.id);

        if let EventPayload::GenerationComplete { model_id, metrics, .. } = decoded.payload {
            assert_eq!(model_id, "test-model");
            assert_eq!(metrics.token_count, 100);
        } else {
            panic!("Expected GenerationComplete payload");
        }
    }

    #[test]
    fn test_roundtrip_threshold_breach() {
        let event = EventEnvelope::new(
            EventSource::Metrics,
            "metrics.threshold_breach",
            EventPayload::ThresholdBreach {
                model_id: "test-model".to_string(),
                metric: "perplexity".to_string(),
                threshold: 50.0,
                actual: 75.0,
                z_score: 2.5,
            },
        );

        let bytes = serialize_event(&event).unwrap();
        let decoded = deserialize_event(&bytes).unwrap();

        assert_eq!(decoded.topic, event.topic);

        if let EventPayload::ThresholdBreach { model_id, metric, threshold, actual, z_score } = decoded.payload {
            assert_eq!(model_id, "test-model");
            assert_eq!(metric, "perplexity");
            assert_eq!(threshold, 50.0);
            assert_eq!(actual, 75.0);
            assert_eq!(z_score, 2.5);
        } else {
            panic!("Expected ThresholdBreach payload");
        }
    }
}
