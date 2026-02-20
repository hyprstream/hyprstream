//! Tool calling: format detection and per-model response parsing.
//!
//! Prompt formatting is handled by HuggingFace chat templates (the `tools` variable
//! is passed to the template engine). This module handles the *output* side:
//! detecting and parsing tool calls from model responses in various formats.

use serde_json::Value;
use regex::Regex;
use once_cell::sync::Lazy;

use super::openai_compat::{ToolCall, ToolCallFunction};
use crate::runtime::model_config::ModelArchitecture;

// =============================================================================
// ToolCallFormat — enum-dispatched per-model parsing
// =============================================================================

/// Tool-call output format used by a model family.
///
/// Selected from the model's architecture string at request time.
/// Parsing is stateless (free functions), so an enum + match is simpler than a trait.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// Qwen3 XML: `<tool_call>{"name":…,"arguments":…}</tool_call>`
    Qwen3Xml,
    /// Llama 3.1+: `<|python_tag|>` prefix + JSON `{"name":…,"parameters":…}`
    LlamaJson,
    /// Mistral: `[TOOL_CALLS]` prefix + JSON array
    MistralJson,
    /// Model does not support tool calling.
    None,
}

impl ToolCallFormat {
    /// Select the tool-call format from the model architecture.
    pub fn from_architecture(arch: &ModelArchitecture) -> Self {
        match arch {
            ModelArchitecture::Qwen => Self::Qwen3Xml,
            ModelArchitecture::Llama => Self::LlamaJson,
            ModelArchitecture::Mistral => Self::MistralJson,
            ModelArchitecture::Gemma | ModelArchitecture::Janus | ModelArchitecture::Unknown(_) => Self::None,
        }
    }

    /// Select the tool-call format from an architecture name string.
    pub fn from_architecture_str(arch: &str) -> Self {
        match arch.to_lowercase().as_str() {
            "qwen" | "qwen2" | "qwen3" => Self::Qwen3Xml,
            "llama" | "llama2" | "llama3" => Self::LlamaJson,
            "mistral" | "mixtral" => Self::MistralJson,
            _ => Self::None,
        }
    }
}

// =============================================================================
// Format-aware dispatch functions
// =============================================================================

/// Check if text contains tool calls in the given format.
pub fn has_tool_calls_for_format(format: ToolCallFormat, text: &str) -> bool {
    match format {
        ToolCallFormat::Qwen3Xml => text.contains("<tool_call>"),
        ToolCallFormat::LlamaJson => text.contains("<|python_tag|>"),
        ToolCallFormat::MistralJson => text.contains("[TOOL_CALLS]"),
        ToolCallFormat::None => false,
    }
}

/// Parse tool calls from model output using the given format.
pub fn parse_tool_calls_for_format(format: ToolCallFormat, text: &str) -> Result<Vec<ToolCall>, anyhow::Error> {
    match format {
        ToolCallFormat::Qwen3Xml => parse_qwen3_tool_calls(text),
        ToolCallFormat::LlamaJson => parse_llama_tool_calls(text),
        ToolCallFormat::MistralJson => parse_mistral_tool_calls(text),
        ToolCallFormat::None => Ok(vec![]),
    }
}

/// Extract text content from response, removing tool call markers for the given format.
pub fn extract_text_content_for_format(format: ToolCallFormat, text: &str) -> String {
    match format {
        ToolCallFormat::Qwen3Xml => extract_qwen3_text_content(text),
        ToolCallFormat::LlamaJson => extract_llama_text_content(text),
        ToolCallFormat::MistralJson => extract_mistral_text_content(text),
        ToolCallFormat::None => text.to_owned(),
    }
}

// =============================================================================
// Legacy API (backwards-compatible, delegates to Qwen3 format)
// =============================================================================

/// Check if text contains Qwen3 tool calls.
///
/// Kept for backwards compatibility. Prefer `has_tool_calls_for_format()`.
pub fn has_tool_calls(text: &str) -> bool {
    has_tool_calls_for_format(ToolCallFormat::Qwen3Xml, text)
}

/// Extract text content from response, removing Qwen3 tool call XML tags.
///
/// Kept for backwards compatibility. Prefer `extract_text_content_for_format()`.
pub fn extract_text_content(text: &str) -> String {
    extract_qwen3_text_content(text)
}

// =============================================================================
// Qwen3 XML parser
// =============================================================================

/// Parse Qwen3 XML tool calls from model output and convert to OpenAI format.
///
/// Uses delimiter-based extraction instead of a single regex so that nested
/// JSON braces inside `arguments` are handled correctly.  Handles malformed
/// output where the model emits duplicate `<tool_call>` tags by searching
/// backward from `</tool_call>` to find the nearest `<tool_call>` (and also
/// skipping blocks that don't contain valid JSON).
pub fn parse_qwen3_tool_calls(text: &str) -> Result<Vec<ToolCall>, anyhow::Error> {
    let mut tool_calls = Vec::new();
    let mut search_from = 0;

    while let Some(pos) = text[search_from..].find("</tool_call>") {
        let close_tag = search_from + pos;

        let region = &text[search_from..close_tag];
        let open_tag = match region.rfind("<tool_call>") {
            Some(pos) => search_from + pos,
            None => {
                search_from = close_tag + "</tool_call>".len();
                continue;
            }
        };

        let inner = text[open_tag + "<tool_call>".len()..close_tag].trim();
        search_from = close_tag + "</tool_call>".len();

        if inner.is_empty() {
            continue;
        }

        let call_data: Value = match serde_json::from_str(inner) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let name = match call_data["name"].as_str() {
            Some(n) => n.to_owned(),
            None => continue,
        };

        let arguments = call_data["arguments"].clone();
        let arguments_str = serde_json::to_string(&arguments)?;

        tool_calls.push(ToolCall {
            id: format!("call_{}", uuid::Uuid::new_v4()),
            tool_type: "function".to_owned(),
            function: ToolCallFunction {
                name,
                arguments: arguments_str,
            },
        });
    }

    Ok(tool_calls)
}

/// Extract text content from Qwen3 response, removing `<tool_call>…</tool_call>` blocks.
fn extract_qwen3_text_content(text: &str) -> String {
    static TOOL_CALL_REGEX: Lazy<Regex> = Lazy::new(|| {
        #[allow(clippy::unwrap_used)]
        Regex::new(r#"(?s)<tool_call>.*?</tool_call>"#).unwrap()
    });
    TOOL_CALL_REGEX.replace_all(text, "").trim().to_owned()
}

// =============================================================================
// Llama JSON parser
// =============================================================================

/// Parse Llama 3.1+ tool calls.
///
/// Format: `<|python_tag|>{"name": "func", "parameters": {...}}`
/// May contain multiple JSON objects separated by newlines.
pub fn parse_llama_tool_calls(text: &str) -> Result<Vec<ToolCall>, anyhow::Error> {
    let mut tool_calls = Vec::new();

    // Find everything after <|python_tag|>
    let tool_text = match text.find("<|python_tag|>") {
        Some(pos) => &text[pos + "<|python_tag|>".len()..],
        None => return Ok(tool_calls),
    };

    // Try parsing each line as a JSON tool call
    for line in tool_text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || !trimmed.starts_with('{') {
            continue;
        }

        let call_data: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let name = match call_data["name"].as_str() {
            Some(n) => n.to_owned(),
            None => continue,
        };

        // Llama uses "parameters" instead of "arguments"
        let arguments = if call_data.get("parameters").is_some() {
            call_data["parameters"].clone()
        } else {
            call_data["arguments"].clone()
        };
        let arguments_str = serde_json::to_string(&arguments)?;

        tool_calls.push(ToolCall {
            id: format!("call_{}", uuid::Uuid::new_v4()),
            tool_type: "function".to_owned(),
            function: ToolCallFunction {
                name,
                arguments: arguments_str,
            },
        });
    }

    Ok(tool_calls)
}

/// Extract text content from Llama response, removing everything after `<|python_tag|>`.
fn extract_llama_text_content(text: &str) -> String {
    match text.find("<|python_tag|>") {
        Some(pos) => text[..pos].trim().to_owned(),
        None => text.to_owned(),
    }
}

// =============================================================================
// Mistral JSON parser
// =============================================================================

/// Parse Mistral tool calls.
///
/// Format: `[TOOL_CALLS] [{"name": "func", "arguments": {...}}]`
pub fn parse_mistral_tool_calls(text: &str) -> Result<Vec<ToolCall>, anyhow::Error> {
    let mut tool_calls = Vec::new();

    // Find everything after [TOOL_CALLS]
    let tool_text = match text.find("[TOOL_CALLS]") {
        Some(pos) => text[pos + "[TOOL_CALLS]".len()..].trim(),
        None => return Ok(tool_calls),
    };

    // Try parsing as a JSON array
    let calls: Vec<Value> = match serde_json::from_str(tool_text) {
        Ok(v) => v,
        Err(_) => return Ok(tool_calls),
    };

    for call_data in calls {
        let name = match call_data["name"].as_str() {
            Some(n) => n.to_owned(),
            None => continue,
        };

        let arguments = call_data["arguments"].clone();
        let arguments_str = serde_json::to_string(&arguments)?;

        tool_calls.push(ToolCall {
            id: format!("call_{}", uuid::Uuid::new_v4()),
            tool_type: "function".to_owned(),
            function: ToolCallFunction {
                name,
                arguments: arguments_str,
            },
        });
    }

    Ok(tool_calls)
}

/// Extract text content from Mistral response, removing everything after `[TOOL_CALLS]`.
fn extract_mistral_text_content(text: &str) -> String {
    match text.find("[TOOL_CALLS]") {
        Some(pos) => text[..pos].trim().to_owned(),
        None => text.to_owned(),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // --- ToolCallFormat selection ---

    #[test]
    fn test_format_from_architecture() {
        assert_eq!(ToolCallFormat::from_architecture(&ModelArchitecture::Qwen), ToolCallFormat::Qwen3Xml);
        assert_eq!(ToolCallFormat::from_architecture(&ModelArchitecture::Llama), ToolCallFormat::LlamaJson);
        assert_eq!(ToolCallFormat::from_architecture(&ModelArchitecture::Mistral), ToolCallFormat::MistralJson);
        assert_eq!(ToolCallFormat::from_architecture(&ModelArchitecture::Gemma), ToolCallFormat::None);
    }

    #[test]
    fn test_format_from_architecture_str() {
        assert_eq!(ToolCallFormat::from_architecture_str("qwen"), ToolCallFormat::Qwen3Xml);
        assert_eq!(ToolCallFormat::from_architecture_str("llama3"), ToolCallFormat::LlamaJson);
        assert_eq!(ToolCallFormat::from_architecture_str("mistral"), ToolCallFormat::MistralJson);
        assert_eq!(ToolCallFormat::from_architecture_str("unknown"), ToolCallFormat::None);
    }

    // --- Qwen3 XML tests (existing, preserved) ---

    #[test]
    fn test_parse_tool_calls() {
        let text = r#"Let me search for that.
<tool_call>
{"name": "search_web", "arguments": {"query": "rust programming"}}
</tool_call>"#;

        let calls = parse_qwen3_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search_web");
    }

    #[test]
    fn test_parse_tool_calls_nested_json() {
        let text = r#"<tool_call>
{"name": "execute_command", "arguments": {"command": "ls -la", "options": {"timeout": 30, "env": {"HOME": "/root"}}}}
</tool_call>"#;

        let calls = parse_qwen3_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "execute_command");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["command"], "ls -la");
        assert_eq!(args["options"]["timeout"], 30);
        assert_eq!(args["options"]["env"]["HOME"], "/root");
    }

    #[test]
    fn test_parse_multiple_tool_calls() {
        let text = r#"I'll do both.
<tool_call>
{"name": "search", "arguments": {"q": "hello"}}
</tool_call>
<tool_call>
{"name": "fetch", "arguments": {"url": "https://example.com"}}
</tool_call>"#;

        let calls = parse_qwen3_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "search");
        assert_eq!(calls[1].function.name, "fetch");
    }

    #[test]
    fn test_parse_tool_calls_duplicate_opening_tag() {
        let text = "<tool_call>\n\n<tool_call>\n{\"name\": \"search_web\", \"arguments\": {\"query\": \"Windermere\"}}\n</tool_call>";

        let calls = parse_qwen3_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search_web");
    }

    #[test]
    fn test_extract_text() {
        let text = r#"Here is the answer: <tool_call>
{"name": "test", "arguments": {}}
</tool_call> More text."#;

        let extracted = extract_text_content(text);
        assert!(!extracted.contains("<tool_call>"));
        assert!(extracted.contains("Here is the answer:"));
    }

    // --- Llama JSON tests ---

    #[test]
    fn test_parse_llama_tool_calls() {
        let text = "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"London\"}}";

        let calls = parse_llama_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "London");
    }

    #[test]
    fn test_llama_has_tool_calls() {
        assert!(has_tool_calls_for_format(ToolCallFormat::LlamaJson, "text <|python_tag|>{...}"));
        assert!(!has_tool_calls_for_format(ToolCallFormat::LlamaJson, "no tool calls here"));
    }

    #[test]
    fn test_llama_extract_text() {
        let text = "Here is my answer.<|python_tag|>{\"name\": \"test\", \"parameters\": {}}";
        let extracted = extract_llama_text_content(text);
        assert_eq!(extracted, "Here is my answer.");
    }

    // --- Mistral JSON tests ---

    #[test]
    fn test_parse_mistral_tool_calls() {
        let text = "[TOOL_CALLS] [{\"name\": \"search\", \"arguments\": {\"query\": \"rust\"}}]";

        let calls = parse_mistral_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");
    }

    #[test]
    fn test_mistral_has_tool_calls() {
        assert!(has_tool_calls_for_format(ToolCallFormat::MistralJson, "[TOOL_CALLS] [...]"));
        assert!(!has_tool_calls_for_format(ToolCallFormat::MistralJson, "no tools"));
    }

    #[test]
    fn test_mistral_extract_text() {
        let text = "Let me help.[TOOL_CALLS] [{\"name\": \"test\", \"arguments\": {}}]";
        let extracted = extract_mistral_text_content(text);
        assert_eq!(extracted, "Let me help.");
    }

    // --- Format-aware dispatch tests ---

    #[test]
    fn test_dispatch_has_tool_calls() {
        assert!(has_tool_calls_for_format(ToolCallFormat::Qwen3Xml, "<tool_call>...</tool_call>"));
        assert!(!has_tool_calls_for_format(ToolCallFormat::None, "<tool_call>...</tool_call>"));
    }
}
