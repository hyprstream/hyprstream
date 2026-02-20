//! OpenAI API compatibility layer

use serde::{Deserialize, Serialize};

// Type aliases for API compatibility
pub use ChatCompletionRequest as OpenAIRequest;
pub use ChatCompletionResponse as OpenAIResponse;

/// Chat completion streaming response
pub type ChatCompletionStreamResponse = OpenAIStreamResponse;

/// OpenAI Chat Completion Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<i32>,
    pub stream: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub max_tokens: Option<usize>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub top_k: Option<usize>,
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    pub user: Option<String>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
}

/// Chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
    pub function_call: Option<FunctionCall>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub tool_call_id: Option<String>,
}

impl From<&ChatCompletionRequest> for crate::config::SamplingParams {
    fn from(req: &ChatCompletionRequest) -> Self {
        Self {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            max_tokens: req.max_tokens,
            repeat_penalty: req.repeat_penalty,
            repeat_last_n: req.repeat_last_n,
            stop_tokens: req.stop.clone(),
            seed: None,
            length_penalty: None,
            typical_p: None,
            epsilon_cutoff: None,
            eta_cutoff: None,
            do_sample: None,
            timeout_ms: None,
        }
    }
}

/// Function call in chat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Tool definition (OpenAI format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: ToolFunction,
}

/// Tool function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value, // JSON Schema
}

/// Tool choice parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    String(String), // "none", "auto", "required"
    Specific { 
        #[serde(rename = "type")]
        tool_type: String, // "function"
        function: ToolChoiceFunction 
    },
}

/// Specific tool choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

/// Tool call made by the model (in response)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String, // "function"
    pub function: ToolCallFunction,
}

/// Tool call function details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String, // JSON string
}

/// Chat Completion Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<Usage>,
}

/// Chat choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: i32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

impl ChatChoice {
    /// Check if this choice contains tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.message.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty())
    }
}

/// Text Completion Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub suffix: Option<String>,
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub n: Option<i32>,
    pub stream: Option<bool>,
    pub logprobs: Option<i32>,
    pub echo: Option<bool>,
    pub stop: Option<Vec<String>>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub top_k: Option<usize>,
    pub best_of: Option<i32>,
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    pub user: Option<String>,
}

impl From<&CompletionRequest> for crate::config::SamplingParams {
    fn from(req: &CompletionRequest) -> Self {
        Self {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            max_tokens: req.max_tokens,
            repeat_penalty: req.repeat_penalty,
            repeat_last_n: req.repeat_last_n,
            stop_tokens: req.stop.clone(),
            seed: None,
            length_penalty: None,
            typical_p: None,
            epsilon_cutoff: None,
            eta_cutoff: None,
            do_sample: None,
            timeout_ms: None,
        }
    }
}

/// Completion Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: Option<Usage>,
}

/// Completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: i32,
    pub logprobs: Option<LogProbs>,
    pub finish_reason: Option<String>,
}

/// Log probabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogProbs {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    pub top_logprobs: Vec<std::collections::HashMap<String, f32>>,
    pub text_offset: Vec<i32>,
}

/// Embedding Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    pub user: Option<String>,
}

/// Embedding Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: Usage,
}

/// Embedding data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// Online training details (hyprstream extension, follows OpenAI's *_details pattern)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub online_training_details: Option<OnlineTrainingDetails>,
}

/// Online training (TTT) adaptation summary for usage reporting.
///
/// Nested inside `usage` following OpenAI's `completion_tokens_details` pattern.
/// Contains the essential metrics identified by TTT literature:
/// - Loss trajectory (avg_loss, loss_improvement) -- convergence signal
/// - Perplexity change (initial/final) -- primary quality metric
/// - Compute budget (steps, time) -- overhead transparency
/// - Action signal (pending, recommendation) -- commit/rollback decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineTrainingDetails {
    pub avg_loss: f32,
    pub loss_improvement: f32,
    pub initial_perplexity: f32,
    pub final_perplexity: f32,
    pub steps_performed: usize,
    pub adaptation_time_ms: u64,
    pub pending: bool,
    pub recommendation: bool,
    pub skipped: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skip_reason: Option<String>,
}

impl From<&crate::config::TTTMetrics> for OnlineTrainingDetails {
    fn from(m: &crate::config::TTTMetrics) -> Self {
        Self {
            avg_loss: m.avg_loss,
            loss_improvement: m.loss_improvement,
            initial_perplexity: m.initial_perplexity,
            final_perplexity: m.final_perplexity,
            steps_performed: m.steps_performed,
            adaptation_time_ms: m.adaptation_time_ms,
            pending: m.pending,
            recommendation: m.recommendation,
            skipped: m.skipped,
            skip_reason: m.skip_reason.clone(),
        }
    }
}

/// List Models Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModelsResponse {
    pub object: String,
    pub data: Vec<Model>,
}

/// Model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

/// Streaming response for SSE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIStreamResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
}

/// Streaming choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoice {
    pub index: i32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

/// Delta content for streaming
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delta {
    pub role: Option<String>,
    pub content: Option<String>,
}

/// Error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: ErrorDetail,
}

/// Error detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub param: Option<String>,
    pub code: Option<String>,
}




