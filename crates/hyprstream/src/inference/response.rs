//! Response types for the inference service.
//!
//! Provides types for handling streaming generation responses
//! and generation statistics.

use futures::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::{mpsc, oneshot};

use super::client::InferenceError;
use crate::config::FinishReason;
use crate::runtime::generation_metrics::GenerationQualityMetrics;

/// Handle for receiving streaming generation output.
///
/// This type wraps the channels used to communicate streaming
/// generation results from the service to the client.
pub struct StreamHandle {
    /// Receiver for text chunks.
    receiver: mpsc::Receiver<Result<String, InferenceError>>,
    /// Receiver for final statistics (sent once after stream ends).
    stats_receiver: Option<oneshot::Receiver<StreamStats>>,
}

impl StreamHandle {
    /// Create a new stream handle.
    pub fn new(
        receiver: mpsc::Receiver<Result<String, InferenceError>>,
        stats_receiver: oneshot::Receiver<StreamStats>,
    ) -> Self {
        Self {
            receiver,
            stats_receiver: Some(stats_receiver),
        }
    }

    /// Get the next text chunk.
    pub async fn next(&mut self) -> Option<Result<String, InferenceError>> {
        self.receiver.recv().await
    }

    /// Convert to an async Stream.
    pub fn into_stream(self) -> impl Stream<Item = Result<String, InferenceError>> {
        StreamHandleStream { handle: self }
    }

    /// Wait for final statistics (call after stream exhausted).
    ///
    /// This consumes self since stats are only available after generation completes.
    pub async fn stats(mut self) -> Result<StreamStats, InferenceError> {
        // Drain any remaining chunks
        while self.receiver.recv().await.is_some() {}

        self.stats_receiver
            .take()
            .ok_or_else(|| InferenceError::channel("Stats already consumed"))?
            .await
            .map_err(|_| InferenceError::channel("Stats channel closed"))
    }
}

/// Wrapper to implement Stream trait for StreamHandle.
struct StreamHandleStream {
    handle: StreamHandle,
}

impl Stream for StreamHandleStream {
    type Item = Result<String, InferenceError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.handle.receiver).poll_recv(cx)
    }
}

/// Statistics from a streaming generation.
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Total generation time in milliseconds.
    pub generation_time_ms: u64,
    /// Tokens per second rate.
    pub tokens_per_second: f32,
    /// Why generation finished.
    pub finish_reason: Option<FinishReason>,
    /// Quality metrics (if captured).
    pub quality_metrics: Option<GenerationQualityMetrics>,
}

impl Default for StreamStats {
    fn default() -> Self {
        Self {
            tokens_generated: 0,
            generation_time_ms: 0,
            tokens_per_second: 0.0,
            finish_reason: None,
            quality_metrics: None,
        }
    }
}

impl StreamStats {
    /// Create stats from generation statistics.
    pub fn from_generation_stats(stats: &crate::runtime::GenerationStats) -> Self {
        Self {
            tokens_generated: stats.tokens_generated,
            generation_time_ms: stats.generation_time_ms,
            tokens_per_second: stats.tokens_per_second,
            finish_reason: stats.finish_reason.clone(),
            quality_metrics: stats.quality_metrics,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_stream_handle_basic() {
        let (chunk_tx, chunk_rx) = mpsc::channel(32);
        let (stats_tx, stats_rx) = oneshot::channel();

        let mut handle = StreamHandle::new(chunk_rx, stats_rx);

        // Send some chunks
        chunk_tx.send(Ok("Hello".to_owned())).await.expect("test: send chunk");
        chunk_tx.send(Ok(" World".to_owned())).await.expect("test: send chunk");
        drop(chunk_tx); // Close channel

        // Receive chunks
        assert_eq!(handle.next().await, Some(Ok("Hello".to_owned())));
        assert_eq!(handle.next().await, Some(Ok(" World".to_owned())));
        assert_eq!(handle.next().await, None);

        // Send stats
        let _ = stats_tx.send(StreamStats::default());

        // Get stats
        let stats = handle.stats().await.expect("test: get stats");
        assert_eq!(stats.tokens_generated, 0);
    }
}
