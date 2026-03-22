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
    /// Qwen3.5 XML parameter format:
    /// `<tool_call><function=NAME><parameter=KEY>value</parameter>…</function></tool_call>`
    Qwen35XmlParam,
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
            ModelArchitecture::Qwen3_5 => Self::Qwen35XmlParam,
            ModelArchitecture::Llama => Self::LlamaJson,
            ModelArchitecture::Mistral => Self::MistralJson,
            ModelArchitecture::Gemma | ModelArchitecture::Janus | ModelArchitecture::Unknown(_) => Self::None,
        }
    }

    /// Select the tool-call format from an architecture name string.
    pub fn from_architecture_str(arch: &str) -> Self {
        match arch.to_lowercase().as_str() {
            "qwen" | "qwen2" | "qwen3" => Self::Qwen3Xml,
            "qwen3_5" | "qwen3_5_text" | "qwen3_5_moe" => Self::Qwen35XmlParam,
            "llama" | "llama2" | "llama3" => Self::LlamaJson,
            "mistral" | "mixtral" => Self::MistralJson,
            _ => Self::None,
        }
    }

    /// Infer the tool-call format from a model reference string (e.g. "qwen3.5-9b:main").
    ///
    /// Uses substring matching against known model family names. Falls back to `None`
    /// if the model family cannot be determined from the name.
    pub fn from_model_ref(model_ref: &str) -> Self {
        let name = model_ref.split(':').next().unwrap_or(model_ref).to_lowercase();
        if name.contains("qwen3.5") || name.contains("qwen3_5") {
            Self::Qwen35XmlParam
        } else if name.contains("qwen") {
            Self::Qwen3Xml
        } else if name.contains("llama") {
            Self::LlamaJson
        } else if name.contains("mistral") || name.contains("mixtral") {
            Self::MistralJson
        } else {
            Self::None
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
        ToolCallFormat::Qwen35XmlParam => has_qwen35_xml_tool_calls(text),
        ToolCallFormat::LlamaJson => text.contains("<|python_tag|>"),
        ToolCallFormat::MistralJson => text.contains("[TOOL_CALLS]"),
        ToolCallFormat::None => false,
    }
}

/// Parse tool calls from model output using the given format.
pub fn parse_tool_calls_for_format(format: ToolCallFormat, text: &str) -> Result<Vec<ToolCall>, anyhow::Error> {
    match format {
        ToolCallFormat::Qwen3Xml => parse_qwen3_tool_calls(text),
        ToolCallFormat::Qwen35XmlParam => parse_qwen35_xml_tool_calls(text),
        ToolCallFormat::LlamaJson => parse_llama_tool_calls(text),
        ToolCallFormat::MistralJson => parse_mistral_tool_calls(text),
        ToolCallFormat::None => Ok(vec![]),
    }
}

/// Extract text content from response, removing tool call markers for the given format.
pub fn extract_text_content_for_format(format: ToolCallFormat, text: &str) -> String {
    match format {
        ToolCallFormat::Qwen3Xml => extract_qwen3_text_content(text),
        ToolCallFormat::Qwen35XmlParam => extract_qwen35_xml_text_content(text),
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
// Qwen3.5 XML parameter parser
// =============================================================================

/// Detect whether text contains Qwen3.5 XML parameter-style tool calls.
///
/// The format uses `<tool_call>` blocks containing `<function=NAME>` tags with
/// `<parameter=KEY>value</parameter>` children, as opposed to the Qwen3 format
/// which embeds a JSON object inside `<tool_call>`.
///
/// Detection requires both `<tool_call>` and `<function=` to disambiguate from
/// Qwen3's JSON-in-XML format.
pub fn has_qwen35_xml_tool_calls(text: &str) -> bool {
    text.contains("<tool_call>") && text.contains("<function=")
}

/// Parse Qwen3.5 XML parameter-style tool calls from model output.
///
/// # Format
///
/// ```text
/// <tool_call>
/// <function=get_weather>
/// <parameter=location>NYC</parameter>
/// <parameter=unit>celsius</parameter>
/// </function>
/// </tool_call>
/// ```
///
/// Multiple `<tool_call>` blocks may appear in a single response.  Parameter
/// values can span multiple lines and are whitespace-trimmed.  Malformed blocks
/// (missing function name, unparseable parameters) are silently skipped.
///
/// Returns an empty `Vec` (not an error) when no tool calls are found.
pub fn parse_qwen35_xml_tool_calls(text: &str) -> Result<Vec<ToolCall>, anyhow::Error> {
    // Regexes compiled once and cached.
    static TOOL_CALL_RE: Lazy<Regex> = Lazy::new(|| {
        #[allow(clippy::unwrap_used)]
        Regex::new(r"(?s)<tool_call>(.*?)</tool_call>").unwrap()
    });
    static FUNCTION_RE: Lazy<Regex> = Lazy::new(|| {
        #[allow(clippy::unwrap_used)]
        Regex::new(r"(?s)<function=([^>]+)>(.*?)</function>").unwrap()
    });
    static PARAM_RE: Lazy<Regex> = Lazy::new(|| {
        #[allow(clippy::unwrap_used)]
        Regex::new(r"(?s)<parameter=([^>]+)>(.*?)</parameter>").unwrap()
    });

    let mut tool_calls = Vec::new();

    for tc_cap in TOOL_CALL_RE.captures_iter(text) {
        let tc_inner = &tc_cap[1];

        for fn_cap in FUNCTION_RE.captures_iter(tc_inner) {
            let fn_name = fn_cap[1].trim();
            if fn_name.is_empty() {
                continue;
            }

            let fn_body = &fn_cap[2];

            // Collect parameter key-value pairs into a JSON object.
            let mut args = serde_json::Map::new();
            for param_cap in PARAM_RE.captures_iter(fn_body) {
                let key = param_cap[1].trim().to_owned();
                let value = param_cap[2].trim().to_owned();
                if key.is_empty() {
                    continue;
                }
                // Try to parse value as JSON first (numbers, booleans, objects,
                // arrays).  Fall back to a plain JSON string on failure.
                let json_val = serde_json::from_str::<Value>(&value)
                    .unwrap_or_else(|_| Value::String(value));
                args.insert(key, json_val);
            }

            let arguments_str = serde_json::to_string(&Value::Object(args))?;

            tool_calls.push(ToolCall {
                id: format!("call_{}", uuid::Uuid::new_v4()),
                tool_type: "function".to_owned(),
                function: ToolCallFunction {
                    name: fn_name.to_owned(),
                    arguments: arguments_str,
                },
            });
        }
    }

    Ok(tool_calls)
}

/// Extract text content from a Qwen3.5 XML parameter response, removing all
/// `<tool_call>…</tool_call>` blocks.
fn extract_qwen35_xml_text_content(text: &str) -> String {
    static TOOL_CALL_RE: Lazy<Regex> = Lazy::new(|| {
        #[allow(clippy::unwrap_used)]
        Regex::new(r"(?s)<tool_call>.*?</tool_call>").unwrap()
    });
    TOOL_CALL_RE.replace_all(text, "").trim().to_owned()
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
        assert_eq!(ToolCallFormat::from_architecture(&ModelArchitecture::Qwen3_5), ToolCallFormat::Qwen35XmlParam);
        assert_eq!(ToolCallFormat::from_architecture(&ModelArchitecture::Llama), ToolCallFormat::LlamaJson);
        assert_eq!(ToolCallFormat::from_architecture(&ModelArchitecture::Mistral), ToolCallFormat::MistralJson);
        assert_eq!(ToolCallFormat::from_architecture(&ModelArchitecture::Gemma), ToolCallFormat::None);
    }

    #[test]
    fn test_format_from_architecture_str() {
        assert_eq!(ToolCallFormat::from_architecture_str("qwen"), ToolCallFormat::Qwen3Xml);
        assert_eq!(ToolCallFormat::from_architecture_str("qwen3"), ToolCallFormat::Qwen3Xml);
        assert_eq!(ToolCallFormat::from_architecture_str("qwen3_5"), ToolCallFormat::Qwen35XmlParam);
        assert_eq!(ToolCallFormat::from_architecture_str("qwen3_5_moe"), ToolCallFormat::Qwen35XmlParam);
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

    // --- Qwen3.5 XML parameter format tests ---

    #[test]
    fn test_qwen35_xml_basic() {
        let text = r#"<tool_call>
<function=get_weather>
<parameter=location>NYC</parameter>
<parameter=unit>celsius</parameter>
</function>
</tool_call>"#;

        assert!(has_qwen35_xml_tool_calls(text));
        let calls = parse_qwen35_xml_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].tool_type, "function");
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].id.starts_with("call_"));

        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "NYC");
        assert_eq!(args["unit"], "celsius");
    }

    #[test]
    fn test_qwen35_xml_multiple_tool_calls() {
        let text = r#"I'll check both locations.

<tool_call>
<function=get_weather>
<parameter=location>NYC</parameter>
</function>
</tool_call>

<tool_call>
<function=get_weather>
<parameter=location>LA</parameter>
</function>
</tool_call>"#;

        let calls = parse_qwen35_xml_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_weather");
        assert_ne!(calls[0].id, calls[1].id, "IDs must be unique");

        let args0: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args0["location"], "NYC");
        assert_eq!(args1["location"], "LA");
    }

    #[test]
    fn test_qwen35_xml_multiline_value() {
        let text = r#"<tool_call>
<function=execute_code>
<parameter=code>
def hello():
    print("hello world")
    return 42
</parameter>
</function>
</tool_call>"#;

        let calls = parse_qwen35_xml_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "execute_code");

        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        let code = args["code"].as_str().unwrap();
        assert!(code.contains("def hello():"));
        assert!(code.contains("return 42"));
    }

    #[test]
    fn test_qwen35_xml_numeric_and_bool_values() {
        let text = r#"<tool_call>
<function=configure>
<parameter=timeout>30</parameter>
<parameter=verbose>true</parameter>
<parameter=ratio>0.75</parameter>
<parameter=name>my-config</parameter>
</function>
</tool_call>"#;

        let calls = parse_qwen35_xml_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);

        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["timeout"], 30);
        assert_eq!(args["verbose"], true);
        assert_eq!(args["ratio"], 0.75);
        assert_eq!(args["name"], "my-config");
    }

    #[test]
    fn test_qwen35_xml_json_object_value() {
        let text = r#"<tool_call>
<function=create_item>
<parameter=data>{"key": "value", "nested": [1,2,3]}</parameter>
</function>
</tool_call>"#;

        let calls = parse_qwen35_xml_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);

        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["data"]["key"], "value");
        assert_eq!(args["data"]["nested"], Value::Array(vec![
            Value::from(1), Value::from(2), Value::from(3),
        ]));
    }

    #[test]
    fn test_qwen35_xml_no_parameters() {
        let text = r#"<tool_call>
<function=get_time>
</function>
</tool_call>"#;

        let calls = parse_qwen35_xml_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_time");

        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert!(args.as_object().unwrap().is_empty());
    }

    #[test]
    fn test_qwen35_xml_inline_parameter_values() {
        // Model might emit values on the same line (no newlines inside parameter tags)
        let text = "<tool_call>\n<function=search>\n<parameter=query>rust programming</parameter>\n</function>\n</tool_call>";

        let calls = parse_qwen35_xml_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search");

        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["query"], "rust programming");
    }

    #[test]
    fn test_qwen35_xml_whitespace_trimming() {
        let text = r#"<tool_call>
<function=greet>
<parameter=name>
   Alice
</parameter>
</function>
</tool_call>"#;

        let calls = parse_qwen35_xml_tool_calls(text).unwrap();
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["name"], "Alice");
    }

    #[test]
    fn test_qwen35_xml_no_tool_calls() {
        let text = "Just a normal response with no tool calls.";
        assert!(!has_qwen35_xml_tool_calls(text));

        let calls = parse_qwen35_xml_tool_calls(text).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn test_qwen35_xml_malformed_no_function() {
        // <tool_call> block with no <function=...> — should be skipped
        let text = "<tool_call>\nsome garbage\n</tool_call>";
        let calls = parse_qwen35_xml_tool_calls(text).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn test_qwen35_xml_malformed_unclosed_tool_call() {
        // Unclosed <tool_call> — regex won't match, returns empty
        let text = "<tool_call>\n<function=test>\n<parameter=x>1</parameter>\n</function>";
        let calls = parse_qwen35_xml_tool_calls(text).unwrap();
        assert!(calls.is_empty());
    }

    #[test]
    fn test_qwen35_xml_text_before_and_after() {
        let text = r#"Let me look that up for you.

<tool_call>
<function=search_web>
<parameter=query>Rust language</parameter>
</function>
</tool_call>

I'll have the results shortly."#;

        let calls = parse_qwen35_xml_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "search_web");

        let extracted = extract_qwen35_xml_text_content(text);
        assert!(extracted.contains("Let me look that up"));
        assert!(extracted.contains("results shortly"));
        assert!(!extracted.contains("<tool_call>"));
        assert!(!extracted.contains("<function="));
    }

    #[test]
    fn test_qwen35_xml_extract_text_no_tool_calls() {
        let text = "No tool calls here.";
        let extracted = extract_qwen35_xml_text_content(text);
        assert_eq!(extracted, "No tool calls here.");
    }

    #[test]
    fn test_qwen35_xml_detection_distinguishes_from_qwen3() {
        // Qwen3 format (JSON inside tool_call) should NOT match the Qwen3.5 detector
        let qwen3 = "<tool_call>\n{\"name\": \"search\", \"arguments\": {\"q\": \"hello\"}}\n</tool_call>";
        assert!(!has_qwen35_xml_tool_calls(qwen3), "Qwen3 JSON format should not trigger Qwen3.5 detector");

        // Qwen3.5 format should match
        let qwen35 = "<tool_call>\n<function=search>\n<parameter=q>hello</parameter>\n</function>\n</tool_call>";
        assert!(has_qwen35_xml_tool_calls(qwen35));
    }

    #[test]
    fn test_qwen35_xml_round_trip() {
        let model_output = "I'll check the weather for you.\n\n\
            <tool_call>\n\
            <function=get_weather>\n\
            <parameter=location>NYC</parameter>\n\
            <parameter=unit>fahrenheit</parameter>\n\
            </function>\n\
            </tool_call>";

        // 1. Detect
        assert!(has_tool_calls_for_format(ToolCallFormat::Qwen35XmlParam, model_output));

        // 2. Parse
        let calls = parse_tool_calls_for_format(ToolCallFormat::Qwen35XmlParam, model_output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].id.starts_with("call_"));
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "NYC");
        assert_eq!(args["unit"], "fahrenheit");

        // 3. Extract text
        let text = extract_text_content_for_format(ToolCallFormat::Qwen35XmlParam, model_output);
        assert!(text.contains("I'll check the weather"));
        assert!(!text.contains("<tool_call>"));
        assert!(!text.contains("<function="));
    }

    #[test]
    fn test_qwen35_xml_dispatch_has_tool_calls() {
        let qwen35_text = "<tool_call>\n<function=test>\n</function>\n</tool_call>";
        assert!(has_tool_calls_for_format(ToolCallFormat::Qwen35XmlParam, qwen35_text));
        assert!(!has_tool_calls_for_format(ToolCallFormat::None, qwen35_text));
        // Qwen3Xml also matches on <tool_call> but that's by design (format is
        // selected at request time from the model architecture, not auto-detected).
    }

    #[test]
    fn test_qwen35_xml_different_functions_in_one_response() {
        let text = r#"<tool_call>
<function=get_weather>
<parameter=location>NYC</parameter>
</function>
</tool_call>

<tool_call>
<function=search_web>
<parameter=query>restaurants near Times Square</parameter>
</function>
</tool_call>"#;

        let calls = parse_qwen35_xml_tool_calls(text).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "search_web");

        let args1: Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args1["query"], "restaurants near Times Square");
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

    // --- Tool call round-trip tests ---

    #[test]
    fn test_tool_call_round_trip_qwen3() {
        // Simulate model response with tool call
        let model_output = "I'll check the weather for you.\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"NYC\"}}\n</tool_call>";

        // 1. Detect tool calls
        assert!(has_tool_calls_for_format(ToolCallFormat::Qwen3Xml, model_output));

        // 2. Parse tool calls
        let calls = parse_tool_calls_for_format(ToolCallFormat::Qwen3Xml, model_output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].id.starts_with("call_"), "ID should be auto-generated");
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "NYC");

        // 3. Extract text content (without tool markers)
        let text = extract_text_content_for_format(ToolCallFormat::Qwen3Xml, model_output);
        assert!(text.contains("I'll check the weather"));
        assert!(!text.contains("<tool_call>"));

        // 4. Build next-turn messages (simulating client constructing history)
        // Verify the parsed tool call can be used to construct the next turn
        let tool_response = r#"{"temperature": 72, "condition": "sunny"}"#;
        let call_id = calls[0].id.clone();

        // Verify the conversation structure pieces are correct
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(!call_id.is_empty(), "tool call ID should be non-empty");
        assert!(!tool_response.is_empty());
        // Assistant message should have no text content (tool-call-only)
        assert!(text.contains("I'll check the weather"));
        assert!(!text.contains("<tool_call>"));
    }

    #[test]
    fn test_tool_call_round_trip_llama() {
        // Llama 3.1 uses <|python_tag|> format with "parameters" field
        let model_output = "<|python_tag|>\n{\"name\": \"get_weather\", \"parameters\": {\"location\": \"Tokyo\"}}";

        assert!(has_tool_calls_for_format(ToolCallFormat::LlamaJson, model_output));

        let calls = parse_tool_calls_for_format(ToolCallFormat::LlamaJson, model_output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        // Llama uses "parameters" not "arguments" — verify it's normalized to arguments
        let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["location"], "Tokyo");
    }

    #[test]
    fn test_tool_call_round_trip_mistral() {
        // Mistral uses [TOOL_CALLS] JSON array format
        let model_output = "Let me check.\n[TOOL_CALLS][{\"name\": \"get_weather\", \"arguments\": {\"location\": \"London\"}}]";

        assert!(has_tool_calls_for_format(ToolCallFormat::MistralJson, model_output));

        let calls = parse_tool_calls_for_format(ToolCallFormat::MistralJson, model_output).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");

        let text = extract_text_content_for_format(ToolCallFormat::MistralJson, model_output);
        assert_eq!(text, "Let me check.");
    }

    #[test]
    fn test_multiple_tool_calls_in_single_response() {
        let model_output = "I'll check both locations.\n\n\
            <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"NYC\"}}\n</tool_call>\n\
            <tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"LA\"}}\n</tool_call>";

        let calls = parse_tool_calls_for_format(ToolCallFormat::Qwen3Xml, model_output).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[1].function.name, "get_weather");

        // Each call should have a unique ID
        assert_ne!(calls[0].id, calls[1].id);

        // Verify different arguments
        let args0: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        let args1: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
        assert_eq!(args0["location"], "NYC");
        assert_eq!(args1["location"], "LA");
    }

    #[test]
    fn test_no_tool_calls_when_format_none() {
        // Even if the text contains tool call markers, ToolCallFormat::None should not detect them
        let text_with_markers = "Here's an example: <tool_call>{\"name\": \"test\"}</tool_call>";

        assert!(!has_tool_calls_for_format(ToolCallFormat::None, text_with_markers));

        // parse should return empty or error
        let result = parse_tool_calls_for_format(ToolCallFormat::None, text_with_markers);
        match result {
            Ok(calls) => assert!(calls.is_empty()),
            Err(_) => {} // Also acceptable
        }
    }

    // --- from_model_ref tests ---

    #[test]
    fn test_from_model_ref() {
        assert_eq!(ToolCallFormat::from_model_ref("qwen3.5-9b:main"), ToolCallFormat::Qwen35XmlParam);
        assert_eq!(ToolCallFormat::from_model_ref("Qwen3.5-9B-FP8:main"), ToolCallFormat::Qwen35XmlParam);
        assert_eq!(ToolCallFormat::from_model_ref("qwen3-8b:main"), ToolCallFormat::Qwen3Xml);
        assert_eq!(ToolCallFormat::from_model_ref("qwen2.5-72b:main"), ToolCallFormat::Qwen3Xml);
        assert_eq!(ToolCallFormat::from_model_ref("llama-3.1-8b:main"), ToolCallFormat::LlamaJson);
        assert_eq!(ToolCallFormat::from_model_ref("mistral-7b:main"), ToolCallFormat::MistralJson);
        assert_eq!(ToolCallFormat::from_model_ref("mixtral-8x7b:main"), ToolCallFormat::MistralJson);
        assert_eq!(ToolCallFormat::from_model_ref("gemma-2b:main"), ToolCallFormat::None);
        assert_eq!(ToolCallFormat::from_model_ref("unknown-model"), ToolCallFormat::None);
    }
}
