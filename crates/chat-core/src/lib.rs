//! Chat orchestration state machine.
//!
//! Owns conversation state, token parsing, tool call detection, and the
//! agentic loop. Used by both the TUI and web GUI — they are thin renderers
//! over this core.
//!
//! This crate has no I/O, no platform deps, no async runtime. It is pure
//! state machine logic that can be compiled to any target (native, WASI, wasm32).

use serde::{Deserialize, Serialize};

// ─── Tool Call Format ────────────────────────────────────────────────

/// Model-specific tool call syntax.
///
/// Determines which markers to scan for in streaming token output.
/// Unified enum — replaces duplicates in chat_app.rs and api/tools.rs.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub enum ToolCallFormat {
    /// Qwen3 XML: `<tool_call>{"name":…,"arguments":…}</tool_call>`
    Qwen3Xml,
    /// Qwen3.5 XML parameter format:
    /// `<tool_call><function=NAME><parameter=KEY>value</parameter>…</function></tool_call>`
    Qwen35XmlParam,
    /// Llama 3.1+: `<|python_tag|>` prefix + JSON
    LlamaJson,
    /// Mistral: `[TOOL_CALLS]` prefix + JSON array
    MistralJson,
    /// Model does not support tool calling.
    #[default]
    None,
}

impl ToolCallFormat {
    /// Determine tool call format from model architecture string.
    pub fn from_architecture(arch: &str) -> Self {
        let lower = arch.to_lowercase();
        if lower.contains("qwen3_5") || lower.contains("qwen3.5") {
            Self::Qwen35XmlParam
        } else if lower.contains("qwen3") || lower.contains("qwen") {
            Self::Qwen3Xml
        } else if lower.contains("llama") {
            Self::LlamaJson
        } else if lower.contains("mistral") {
            Self::MistralJson
        } else {
            Self::None
        }
    }

    /// Returns the (open, close) marker pair for this format.
    pub fn markers(&self) -> Option<(&'static str, &'static str)> {
        match self {
            Self::Qwen3Xml => Some(("<tool_call>", "</tool_call>")),
            Self::Qwen35XmlParam => Some(("<tool_call>", "</tool_call>")),
            Self::LlamaJson => Some(("<|python_tag|>", "")),
            Self::MistralJson => Some(("[TOOL_CALLS]", "")),
            Self::None => None,
        }
    }
}

// ─── Chat History Types ──────────────────────────────────────────────

/// Role of a chat history entry.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum ChatRole {
    User,
    Assistant,
    /// Tool result injected back into the conversation.
    Tool,
}

/// A recorded tool call and its result (or in-flight state).
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ToolCallRecord {
    /// Per-invocation correlation ID (UUID v4).
    #[serde(default)]
    pub id: String,
    /// The UUID name used on the wire (what the model outputs as `name`).
    pub uuid: String,
    /// Human-readable label from `list_tools()`.
    pub description: String,
    /// JSON-encoded arguments string.
    pub arguments: String,
    /// Execution result. `None` while the call is in-flight.
    pub result: Option<String>,
}

/// A single message in the conversation history.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ChatHistoryEntry {
    pub role: ChatRole,
    /// Visible response content.
    pub content: String,
    /// Reasoning / thinking tokens (hidden by default).
    #[serde(default)]
    pub thinking: String,
    /// Tool calls made by this assistant turn.
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRecord>,
    /// For Tool messages: the UUID of the corresponding tool call.
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

// ─── Events ──────────────────────────────────────────────────────────

/// Events emitted by the chat state machine.
/// These are structured — clients never see raw `<think>` or `<tool_call>` tags.
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// Display text content chunk.
    Content(String),
    /// Thinking/reasoning block chunk.
    Thinking(String),
    /// A complete tool call was detected in the model output.
    ToolCallDetected {
        id: String,
        uuid: String,
        description: String,
        arguments: String,
    },
    /// The model finished generating (stream complete).
    Complete,
    /// Stream error.
    Error(String),
}

/// Commands sent to the chat state machine.
#[derive(Debug, Clone)]
pub enum ChatCommand {
    /// User submitted a message.
    UserMessage(String),
    /// An inference token arrived from the backend.
    InferenceToken(String),
    /// Inference stream completed.
    InferenceComplete,
    /// Inference stream errored.
    InferenceError(String),
    /// Tool execution result.
    ToolCallResult { id: String, result: String },
    /// Cancel current generation.
    Cancel,
}

// ─── Chat State Machine ──────────────────────────────────────────────

/// Core chat orchestration state.
///
/// Owns conversation history, token parsing, tool call detection,
/// and the agentic loop. No I/O — all communication is via
/// `handle_command()` → `Vec<ChatEvent>`.
pub struct ChatState {
    pub history: Vec<ChatHistoryEntry>,
    pub tool_call_format: ToolCallFormat,
    pub tool_descriptions: std::collections::HashMap<String, String>,

    // Streaming parser state
    in_thinking: bool,
    in_tool_call: bool,
    tool_call_buf: String,

    // Agentic loop state
    pub pending_tool_calls: usize,
    tool_call_counter: usize,
    awaiting_reinvoke: bool,
}

impl ChatState {
    pub fn new(format: ToolCallFormat) -> Self {
        Self {
            history: Vec::new(),
            tool_call_format: format,
            tool_descriptions: std::collections::HashMap::new(),
            in_thinking: false,
            in_tool_call: false,
            tool_call_buf: String::new(),
            pending_tool_calls: 0,
            tool_call_counter: 0,
            awaiting_reinvoke: false,
        }
    }

    /// Process a command and return events to emit.
    pub fn handle_command(&mut self, cmd: ChatCommand) -> Vec<ChatEvent> {
        match cmd {
            ChatCommand::UserMessage(text) => {
                self.history.push(ChatHistoryEntry {
                    role: ChatRole::User,
                    content: text,
                    thinking: String::new(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                });
                // Add empty assistant entry for streaming
                self.history.push(ChatHistoryEntry {
                    role: ChatRole::Assistant,
                    content: String::new(),
                    thinking: String::new(),
                    tool_calls: Vec::new(),
                    tool_call_id: None,
                });
                self.in_thinking = false;
                self.in_tool_call = false;
                self.tool_call_buf.clear();
                Vec::new() // Caller should trigger inference
            }
            ChatCommand::InferenceToken(token) => {
                self.awaiting_reinvoke = false;
                self.ingest_token(&token)
            }
            ChatCommand::InferenceComplete => {
                let mut events = Vec::new();

                // Flush buffered tool call for formats without closing tags (Llama/Mistral)
                if self.in_tool_call && !self.tool_call_buf.is_empty() {
                    self.in_tool_call = false;
                    let buf = std::mem::take(&mut self.tool_call_buf);
                    if let Some((uuid, arguments)) = parse_tool_call_buf(&buf) {
                        let description = self.tool_descriptions
                            .get(&uuid).cloned().unwrap_or_else(|| uuid.clone());
                        self.tool_call_counter += 1;
                        let id = format!("tc-{}", self.tool_call_counter);
                        if let Some(last) = self.history.last_mut() {
                            last.tool_calls.push(ToolCallRecord {
                                id: id.clone(), uuid: uuid.clone(),
                                description: description.clone(),
                                arguments: arguments.clone(), result: None,
                            });
                        }
                        self.pending_tool_calls += 1;
                        events.push(ChatEvent::ToolCallDetected { id, uuid, description, arguments });
                    }
                }

                if self.pending_tool_calls == 0 {
                    events.push(ChatEvent::Complete);
                }
                events
            }
            ChatCommand::InferenceError(msg) => {
                vec![ChatEvent::Error(msg)]
            }
            ChatCommand::ToolCallResult { id, result } => {
                // Inject tool result into history
                if let Some(last_assistant) = self.history.iter_mut().rev()
                    .find(|e| e.role == ChatRole::Assistant)
                {
                    if let Some(tc) = last_assistant.tool_calls.iter_mut().find(|t| t.id == id) {
                        tc.result = Some(result.clone());
                    }
                }
                // Add tool message to history
                self.history.push(ChatHistoryEntry {
                    role: ChatRole::Tool,
                    content: result,
                    thinking: String::new(),
                    tool_calls: Vec::new(),
                    tool_call_id: Some(id),
                });

                self.pending_tool_calls = self.pending_tool_calls.saturating_sub(1);

                // If all tool calls resolved, set up for re-invocation
                if self.pending_tool_calls == 0 {
                    self.awaiting_reinvoke = true;
                    // Add new empty assistant entry for the next generation
                    self.history.push(ChatHistoryEntry {
                        role: ChatRole::Assistant,
                        content: String::new(),
                        thinking: String::new(),
                        tool_calls: Vec::new(),
                        tool_call_id: None,
                    });
                    self.in_thinking = false;
                    self.in_tool_call = false;
                    self.tool_call_buf.clear();
                    // Caller should re-trigger inference with updated history
                }
                Vec::new()
            }
            ChatCommand::Cancel => {
                self.in_thinking = false;
                self.in_tool_call = false;
                self.tool_call_buf.clear();
                Vec::new()
            }
        }
    }

    /// Process a streaming token. Handles `<think>`, `<tool_call>`, and
    /// content separation. Returns events to emit.
    ///
    /// Mirrors the TUI's `ChatApp::ingest_token` but returns events
    /// instead of mutating UI state.
    pub fn ingest_token(&mut self, token: &str) -> Vec<ChatEvent> {
        let mut events: Vec<ChatEvent> = Vec::new();
        let markers = self.tool_call_format.markers();

        let last = match self.history.last_mut() {
            Some(e) => e,
            None => return events,
        };

        let mut remaining = token;

        while !remaining.is_empty() {
            if self.in_thinking {
                if let Some(end) = remaining.find("</think>") {
                    let chunk = &remaining[..end];
                    last.thinking.push_str(chunk);
                    events.push(ChatEvent::Thinking(chunk.to_owned()));
                    self.in_thinking = false;
                    remaining = &remaining[end + "</think>".len()..];
                } else {
                    last.thinking.push_str(remaining);
                    events.push(ChatEvent::Thinking(remaining.to_owned()));
                    break;
                }
            } else if self.in_tool_call {
                let close = markers.map(|(_, c)| c).unwrap_or("");
                if close.is_empty() {
                    // Llama/Mistral: no closing tag — buffer everything
                    self.tool_call_buf.push_str(remaining);
                    break;
                }
                if let Some(end) = remaining.find(close) {
                    self.tool_call_buf.push_str(&remaining[..end]);
                    remaining = &remaining[end + close.len()..];
                    self.in_tool_call = false;
                    let buf = std::mem::take(&mut self.tool_call_buf);
                    if let Some((uuid, arguments)) = parse_tool_call_buf(&buf) {
                        let description = self.tool_descriptions
                            .get(&uuid)
                            .cloned()
                            .unwrap_or_else(|| uuid.clone());
                        self.tool_call_counter += 1;
                        let id = format!("tc-{}", self.tool_call_counter);
                        last.tool_calls.push(ToolCallRecord {
                            id: id.clone(),
                            uuid: uuid.clone(),
                            description: description.clone(),
                            arguments: arguments.clone(),
                            result: None,
                        });
                        self.pending_tool_calls += 1;
                        events.push(ChatEvent::ToolCallDetected {
                            id,
                            uuid,
                            description,
                            arguments,
                        });
                    }
                } else {
                    self.tool_call_buf.push_str(remaining);
                    break;
                }
            } else if let Some((open, _)) = markers {
                if let Some(start) = remaining.find(open) {
                    let content = &remaining[..start];
                    if !content.is_empty() {
                        last.content.push_str(content);
                        events.push(ChatEvent::Content(content.to_owned()));
                    }
                    self.in_tool_call = true;
                    self.tool_call_buf.clear();
                    remaining = &remaining[start + open.len()..];
                } else if let Some(start) = remaining.find("<think>") {
                    let content = &remaining[..start];
                    if !content.is_empty() {
                        last.content.push_str(content);
                        events.push(ChatEvent::Content(content.to_owned()));
                    }
                    self.in_thinking = true;
                    remaining = &remaining[start + "<think>".len()..];
                } else {
                    last.content.push_str(remaining);
                    events.push(ChatEvent::Content(remaining.to_owned()));
                    break;
                }
            } else if let Some(start) = remaining.find("<think>") {
                let content = &remaining[..start];
                if !content.is_empty() {
                    last.content.push_str(content);
                    events.push(ChatEvent::Content(content.to_owned()));
                }
                self.in_thinking = true;
                remaining = &remaining[start + "<think>".len()..];
            } else {
                last.content.push_str(remaining);
                events.push(ChatEvent::Content(remaining.to_owned()));
                break;
            }
        }

        events
    }

    /// Whether the state machine needs inference to be re-invoked
    /// (e.g. after all tool results have been received).
    /// Whether the state machine needs inference to be re-invoked
    /// (after all tool results received and a new empty assistant entry is ready).
    pub fn needs_reinvoke(&self) -> bool {
        self.awaiting_reinvoke
    }
}

// ─── Tool Call Parsers ───────────────────────────────────────────────

/// Parse a tool call buffer. Tries JSON first (Qwen3/Hermes), then
/// Qwen3.5 XML parameter format.
pub fn parse_tool_call_buf(buf: &str) -> Option<(String, String)> {
    let trimmed = buf.trim();

    // Try JSON format: {"name": "...", "arguments": {...}}
    if let Ok(call_data) = serde_json::from_str::<serde_json::Value>(trimmed) {
        let name = call_data["name"].as_str()?.to_owned();
        let arguments = serde_json::to_string(&call_data["arguments"]).ok()?;
        return Some((name, arguments));
    }

    // Fall back to Qwen3.5 XML parameter format:
    //   <function=get_weather>
    //   <parameter=location>NYC</parameter>
    //   </function>
    if trimmed.contains("<function=") {
        return parse_xml_param_tool_call(trimmed);
    }

    None
}

/// Parse a Qwen3.5 XML parameter-format tool call block.
pub fn parse_xml_param_tool_call(buf: &str) -> Option<(String, String)> {
    // Extract function name from <function=NAME>
    let func_start = buf.find("<function=")?;
    let name_start = func_start + "<function=".len();
    let name_end = buf[name_start..].find('>')? + name_start;
    let name = buf[name_start..name_end].trim().to_owned();
    if name.is_empty() {
        return None;
    }

    // Extract body between <function=NAME> and </function>
    let body_start = name_end + 1;
    let body_end = buf.find("</function>").unwrap_or(buf.len());
    let body = &buf[body_start..body_end];

    // Extract <parameter=KEY>VALUE</parameter> pairs
    let mut args = serde_json::Map::new();
    let mut search_from = 0;
    while let Some(param_start) = body[search_from..].find("<parameter=") {
        let abs_start = search_from + param_start;
        let key_start = abs_start + "<parameter=".len();
        let key_end = match body[key_start..].find('>') {
            Some(p) => key_start + p,
            None => break,
        };
        let key = body[key_start..key_end].trim().to_owned();

        let value_start = key_end + 1;
        let value_end = match body[value_start..].find("</parameter>") {
            Some(p) => value_start + p,
            None => break,
        };
        let value = body[value_start..value_end].trim();

        if !key.is_empty() {
            // Try JSON parse for numbers/bools/objects, fall back to string
            let json_val = serde_json::from_str::<serde_json::Value>(value)
                .unwrap_or_else(|_| serde_json::Value::String(value.to_owned()));
            args.insert(key, json_val);
        }

        search_from = value_end + "</parameter>".len();
    }

    let arguments = serde_json::to_string(&serde_json::Value::Object(args)).ok()?;
    Some((name, arguments))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_format_from_architecture() {
        assert_eq!(ToolCallFormat::from_architecture("qwen3"), ToolCallFormat::Qwen3Xml);
        assert_eq!(ToolCallFormat::from_architecture("qwen3_5"), ToolCallFormat::Qwen35XmlParam);
        assert_eq!(ToolCallFormat::from_architecture("llama3"), ToolCallFormat::LlamaJson);
        assert_eq!(ToolCallFormat::from_architecture("mistral"), ToolCallFormat::MistralJson);
        assert_eq!(ToolCallFormat::from_architecture("gemma"), ToolCallFormat::None);
    }

    #[test]
    fn test_parse_json_tool_call() {
        let buf = r#"{"name": "get_weather", "arguments": {"location": "NYC"}}"#;
        let (name, args) = parse_tool_call_buf(buf).unwrap();
        assert_eq!(name, "get_weather");
        assert!(args.contains("NYC"));
    }

    #[test]
    fn test_parse_xml_tool_call() {
        let buf = "<function=get_weather>\n<parameter=location>NYC</parameter>\n</function>";
        let (name, args) = parse_xml_param_tool_call(buf).unwrap();
        assert_eq!(name, "get_weather");
        assert!(args.contains("NYC"));
    }

    #[test]
    fn test_ingest_token_content() {
        let mut state = ChatState::new(ToolCallFormat::Qwen3Xml);
        state.handle_command(ChatCommand::UserMessage("hi".into()));
        let events = state.ingest_token("Hello world");
        assert_eq!(events.len(), 1);
        match &events[0] {
            ChatEvent::Content(text) => assert_eq!(text, "Hello world"),
            _ => panic!("expected Content event"),
        }
    }

    #[test]
    fn test_ingest_token_thinking() {
        let mut state = ChatState::new(ToolCallFormat::Qwen3Xml);
        state.handle_command(ChatCommand::UserMessage("hi".into()));
        let events = state.ingest_token("<think>reasoning here</think>answer");
        // Should produce: Thinking("reasoning here"), Content("answer")
        assert_eq!(events.len(), 2);
        match &events[0] {
            ChatEvent::Thinking(text) => assert_eq!(text, "reasoning here"),
            _ => panic!("expected Thinking event"),
        }
        match &events[1] {
            ChatEvent::Content(text) => assert_eq!(text, "answer"),
            _ => panic!("expected Content event"),
        }
    }

    #[test]
    fn test_ingest_token_tool_call_json() {
        let mut state = ChatState::new(ToolCallFormat::Qwen3Xml);
        state.handle_command(ChatCommand::UserMessage("list repos".into()));
        let events = state.ingest_token(
            r#"<tool_call>{"name": "list_repos", "arguments": {}}</tool_call>"#
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            ChatEvent::ToolCallDetected { uuid, .. } => assert_eq!(uuid, "list_repos"),
            _ => panic!("expected ToolCallDetected event"),
        }
        assert_eq!(state.pending_tool_calls, 1);
    }

    #[test]
    fn test_ingest_token_tool_call_xml() {
        let mut state = ChatState::new(ToolCallFormat::Qwen35XmlParam);
        state.handle_command(ChatCommand::UserMessage("weather?".into()));
        let events = state.ingest_token(
            "<tool_call><function=get_weather><parameter=location>NYC</parameter></function></tool_call>"
        );
        assert_eq!(events.len(), 1);
        match &events[0] {
            ChatEvent::ToolCallDetected { uuid, arguments, .. } => {
                assert_eq!(uuid, "get_weather");
                assert!(arguments.contains("NYC"));
            }
            _ => panic!("expected ToolCallDetected"),
        }
    }
}
