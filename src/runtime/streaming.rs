//! Streaming support for token-by-token generation
//!
//! This module provides async streaming callbacks for real-time token generation,
//! enabling SSE/WebSocket streaming with proper backpressure and cancellation.

use anyhow::Result;
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use std::time::Duration;
use serde_json::json;
use crate::runtime::FinishReason;

/// Control flow for generation continuation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ContinueGeneration {
    /// Continue generating tokens
    Continue,
    /// Stop generation gracefully
    Stop,
    /// Pause generation temporarily (for backpressure)
    Pause(Duration),
}

/// Async callback trait for streaming token generation
#[async_trait]
pub trait StreamingCallback: Send {
    /// Called for each generated token
    async fn on_token(&mut self, token: &str) -> Result<ContinueGeneration>;
    
    /// Called when generation completes
    async fn on_complete(&mut self, reason: FinishReason);
    
    /// Called on error
    async fn on_error(&mut self, error: anyhow::Error);
    
    /// Called at start of generation
    async fn on_start(&mut self) {}
}

/// SSE-specific streaming callback for OpenAI-compatible streaming
pub struct SseStreamingCallback {
    /// Channel sender for SSE events
    sender: mpsc::Sender<Result<serde_json::Value, anyhow::Error>>,
    /// Buffer for batching tokens
    buffer: String,
    /// Minimum tokens to batch before sending
    chunk_size: usize,
    /// Model name for responses
    model: String,
    /// Stream ID
    stream_id: String,
    /// Token counter
    tokens_sent: usize,
}

impl SseStreamingCallback {
    /// Create a new SSE streaming callback
    pub fn new(
        sender: mpsc::Sender<Result<serde_json::Value, anyhow::Error>>,
        model: String,
    ) -> Self {
        Self {
            sender,
            buffer: String::new(),
            chunk_size: 1, // Send every token immediately for minimal latency
            model,
            stream_id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            tokens_sent: 0,
        }
    }
    
    /// Flush buffered tokens
    async fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        
        let response = json!({
            "id": self.stream_id,
            "object": "chat.completion.chunk",
            "created": chrono::Utc::now().timestamp(),
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": self.buffer.clone()
                },
                "finish_reason": null
            }]
        });
        
        // Try to send with backpressure handling
        match self.sender.try_send(Ok(response)) {
            Ok(_) => {
                self.tokens_sent += 1;
                self.buffer.clear();
                Ok(())
            }
            Err(mpsc::error::TrySendError::Full(_)) => {
                // Channel is full, apply backpressure
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                Err(anyhow::anyhow!("Stream closed by client"))
            }
        }
    }
}

#[async_trait]
impl StreamingCallback for SseStreamingCallback {
    async fn on_start(&mut self) {
        // Send initial role message
        let response = json!({
            "id": self.stream_id,
            "object": "chat.completion.chunk",
            "created": chrono::Utc::now().timestamp(),
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": ""
                },
                "finish_reason": null
            }]
        });
        
        let _ = self.sender.send(Ok(response)).await;
    }
    
    async fn on_token(&mut self, token: &str) -> Result<ContinueGeneration> {
        self.buffer.push_str(token);
        
        // Check if we should flush
        if self.buffer.len() >= self.chunk_size || 
           token.contains('\n') || // Flush on newlines for better UX
           token.ends_with('.') || token.ends_with('!') || token.ends_with('?') {
            self.flush().await?;
        }
        
        // Check for backpressure
        if self.sender.capacity() == 0 {
            // Channel is at capacity, pause briefly
            Ok(ContinueGeneration::Pause(Duration::from_millis(10)))
        } else {
            Ok(ContinueGeneration::Continue)
        }
    }
    
    async fn on_complete(&mut self, reason: FinishReason) {
        // Flush any remaining tokens
        let _ = self.flush().await;
        
        // Send completion message
        let finish_reason = match reason {
            FinishReason::MaxTokens => "length",
            FinishReason::StopToken(_) => "stop",
            FinishReason::EndOfSequence => "stop",
            FinishReason::Stop => "stop",
            FinishReason::Error(_) => "stop",
        };
        
        let response = json!({
            "id": self.stream_id,
            "object": "chat.completion.chunk",
            "created": chrono::Utc::now().timestamp(),
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason
            }]
        });
        
        let _ = self.sender.send(Ok(response)).await;
    }
    
    async fn on_error(&mut self, error: anyhow::Error) {
        let _ = self.sender.send(Err(error)).await;
    }
}

/// Simple callback for collecting generated text
pub struct CollectingCallback {
    pub text: String,
}

impl CollectingCallback {
    pub fn new() -> Self {
        Self {
            text: String::new(),
        }
    }
}

#[async_trait]
impl StreamingCallback for CollectingCallback {
    async fn on_token(&mut self, token: &str) -> Result<ContinueGeneration> {
        self.text.push_str(token);
        Ok(ContinueGeneration::Continue)
    }
    
    async fn on_complete(&mut self, _reason: FinishReason) {}
    
    async fn on_error(&mut self, _error: anyhow::Error) {}
}

/// Cancellable generation context
pub struct GenerationContext {
    /// Cancellation token for stopping generation
    pub cancel_token: CancellationToken,
    /// Maximum time to wait for generation
    pub timeout: Duration,
}

impl Default for GenerationContext {
    fn default() -> Self {
        Self {
            cancel_token: CancellationToken::new(),
            timeout: Duration::from_secs(300), // 5 minute default timeout
        }
    }
}