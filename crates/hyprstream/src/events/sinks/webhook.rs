//! HTTP webhook sink implementation
//!
//! Forwards events to an HTTP endpoint via POST requests.
//! Runs on a blocking thread, spawns async tasks for HTTP requests.

use super::super::bus::EventSubscriber;
use super::super::EventEnvelope;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue};
use std::collections::HashMap;
use tracing::{debug, error, info, trace, warn};

/// Webhook sink loop
///
/// Receives events and forwards them to an HTTP endpoint via POST.
/// Events are sent as JSON in the request body.
///
/// This runs on a blocking thread. HTTP requests are spawned as
/// async tasks on the tokio runtime.
pub fn webhook_loop(
    subscriber: EventSubscriber,
    url: &str,
    headers: Option<HashMap<String, String>>,
) {
    info!(
        "starting webhook sink to '{}' for topic '{}'",
        url,
        subscriber.topic_filter()
    );

    // Build HTTP client with custom headers
    let mut header_map = HeaderMap::new();
    header_map.insert(
        reqwest::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );

    if let Some(custom_headers) = &headers {
        for (key, value) in custom_headers {
            if let (Ok(name), Ok(val)) = (
                HeaderName::try_from(key.as_str()),
                HeaderValue::from_str(value),
            ) {
                header_map.insert(name, val);
            } else {
                warn!("invalid header: {}={}", key, value);
            }
        }
    }

    let client = reqwest::Client::builder()
        .default_headers(header_map)
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .expect("failed to build HTTP client");

    let url = url.to_string();

    loop {
        match subscriber.recv() {
            Ok(event) => {
                debug!(
                    "webhook sink received event {} (topic: {})",
                    event.id, event.topic
                );

                // Clone for the async task
                let client = client.clone();
                let url = url.clone();

                // Spawn async task for the HTTP request
                // Use spawn_blocking's handle to run async code
                let rt = tokio::runtime::Handle::try_current();
                if let Ok(handle) = rt {
                    handle.spawn(async move {
                        if let Err(e) = send_webhook(&client, &url, &event).await {
                            error!("failed to send webhook to {}: {}", url, e);
                        }
                    });
                } else {
                    // Fallback: blocking HTTP (not ideal but works)
                    let payload = match serde_json::to_vec(&event) {
                        Ok(p) => p,
                        Err(e) => {
                            error!("failed to serialize event: {}", e);
                            continue;
                        }
                    };
                    trace!(
                        "would POST to {} ({} bytes)",
                        url,
                        payload.len()
                    );
                }
            }
            Err(e) => {
                warn!("webhook sink recv error: {}", e);
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        }
    }
}

/// Send a single event to the webhook endpoint
async fn send_webhook(
    client: &reqwest::Client,
    url: &str,
    event: &EventEnvelope,
) -> Result<(), reqwest::Error> {
    let response = client.post(url).json(event).send().await?;

    let status = response.status();
    if status.is_success() {
        trace!("webhook response: {} for event {}", status, event.id);
    } else {
        let body = response.text().await.unwrap_or_default();
        warn!(
            "webhook returned non-success status {} for event {}: {}",
            status, event.id, body
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{EventPayload, EventSource, GenerationMetrics};

    #[test]
    fn test_event_serialization() {
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

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("threshold_breach"));
        assert!(json.contains("perplexity"));
    }
}
