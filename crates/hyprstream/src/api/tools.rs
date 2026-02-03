//! Tool calling translation between OpenAI JSON and Qwen3 XML formats

use serde_json::Value;
use regex::Regex;
use once_cell::sync::Lazy;

use super::openai_compat::{Tool, ToolCall, ToolCallFunction};

/// Translate OpenAI tools to Qwen3 XML format for inclusion in prompt.
///
/// Produces the exact text that Qwen3's chat template generates when the
/// `tools` variable is set, so the model recognises the tool-calling
/// instructions regardless of whether they arrive via the template variable
/// or via the system message content.
pub fn tools_to_qwen3_xml(tools: &[Tool]) -> String {
    let mut tool_lines = Vec::new();
    for tool in tools {
        let json = serde_json::to_string(tool).unwrap_or_default();
        tool_lines.push(json);
    }

    format!(
        "# Tools\n\
         \n\
         You may call one or more functions to assist with the user query.\n\
         \n\
         You are provided with function signatures within <tools></tools> XML tags:\n\
         <tools>\n\
         {}\n\
         </tools>\n\
         \n\
         For each function call, return a json object with function name and arguments within \
         <tool_call></tool_call> XML tags:\n\
         <tool_call>\n\
         {{\"name\": <function-name>, \"arguments\": <args-json-object>}}\n\
         </tool_call>",
        tool_lines.join("\n")
    )
}

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

    loop {
        // Find the next </tool_call> first, then search backward for the nearest <tool_call>.
        // This handles cases where the model emits duplicate opening tags:
        //   <tool_call>\n<tool_call>\n{"name": ...}\n</tool_call>
        let close_tag = match text[search_from..].find("</tool_call>") {
            Some(pos) => search_from + pos,
            None => break,
        };

        // Search backward from close_tag for the nearest <tool_call>
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

        // Parse the JSON object (may contain nested braces).
        // Skip blocks that don't parse as valid JSON rather than failing entirely.
        let call_data: Value = match serde_json::from_str(inner) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let name = match call_data["name"].as_str() {
            Some(n) => n.to_string(),
            None => continue,
        };

        let arguments = call_data["arguments"].clone();
        let arguments_str = serde_json::to_string(&arguments)?;

        tool_calls.push(ToolCall {
            id: format!("call_{}", uuid::Uuid::new_v4()),
            tool_type: "function".to_string(),
            function: ToolCallFunction {
                name,
                arguments: arguments_str,
            },
        });
    }

    Ok(tool_calls)
}

/// Extract text content from response, removing tool call XML tags
pub fn extract_text_content(text: &str) -> String {
    static TOOL_CALL_REGEX: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r#"(?s)<tool_call>.*?</tool_call>"#).unwrap()
    });
    
    TOOL_CALL_REGEX.replace_all(text, "").trim().to_string()
}

/// Check if text contains Qwen3 tool calls
pub fn has_tool_calls(text: &str) -> bool {
    text.contains("<tool_call>")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::openai_compat::{Tool, ToolFunction};

    #[test]
    fn test_tools_to_xml() {
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "search_web".to_string(),
                description: Some("Search the web".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }),
            },
        }];

        let xml = tools_to_qwen3_xml(&tools);
        assert!(xml.contains("<tools>"));
        assert!(xml.contains("</tools>"));
        assert!(xml.contains("search_web"));
        // Must include the full Qwen3 tool-calling instructions
        assert!(xml.contains("# Tools"));
        assert!(xml.contains("You may call one or more functions"));
        assert!(xml.contains("<tool_call>"));
        assert!(xml.contains("</tool_call>"));
    }

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
        // Model sometimes emits duplicate <tool_call> tags
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
}
