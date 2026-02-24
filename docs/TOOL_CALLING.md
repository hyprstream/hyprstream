# Tool Calling Implementation

## Overview

HyprStream now supports OpenAI-compatible tool calling (function calling) for compatibility with Cline and other coding frontends. This implementation bridges the gap between OpenAI's JSON-based tool format and Qwen3's native XML-based tool format.

## Architecture

### Format Translation Layer

The implementation uses a translation layer that:

1. **Input**: Converts OpenAI JSON tool definitions → Qwen3 XML format (injected into system message)
2. **Output**: Parses Qwen3 XML tool calls → OpenAI JSON ToolCall format

### Files

- **`crates/hyprstream/src/api/tools.rs`**: Core translation logic
- **`crates/hyprstream/src/server/routes/openai.rs`**: Integration into chat completion endpoints

## How It Works

### 1. Tool Definition Injection (Request Phase)

When a client sends tools in the request:

```json
{
  "model": "qwen3:main",
  "messages": [...],
  "tools": [{
    "type": "function",
    "function": {
      "name": "execute_command",
      "description": "Execute a shell command",
      "parameters": { ... }
    }
  }]
}
```

The server:
1. Converts tools to Qwen3 XML format using `tools::tools_to_qwen3_xml()`
2. Injects XML into system message (or creates new system message)
3. Applies chat template with augmented messages

Example XML injection:
```xml
<tools>
  <tool name="execute_command">
    <description>Execute a shell command</description>
    <parameters>
      {JSON schema}
    </parameters>
  </tool>
</tools>
```

### 2. Tool Call Parsing (Response Phase)

When Qwen3 generates a tool call:

```xml
<tool_call>
{"name": "execute_command", "arguments": {"command": "ls -la"}}
</tool_call>
```

The server:
1. Detects tool calls using `tools::has_tool_calls()`
2. Parses XML using `tools::parse_qwen3_tool_calls()`
3. Extracts text content using `tools::extract_text_content()`
4. Returns OpenAI-format response:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "execute_command",
          "arguments": "{\"command\": \"ls -la\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

## Implementation Details

### Non-Streaming (`/v1/chat/completions`)

1. Inject tools into system message
2. Apply chat template
3. Generate response
4. Parse tool calls from accumulated text
5. Return with `finish_reason: "tool_calls"` if tool calls detected

### Streaming (`/v1/chat/completions` with `stream: true`)

1. Inject tools into system message
2. Apply chat template
3. Stream tokens as they're generated (accumulate text internally)
4. On completion:
   - Parse accumulated text for tool calls
   - Send tool calls in a separate delta chunk
   - Send final completion with `finish_reason: "tool_calls"`

## Key Functions

### `tools::tools_to_qwen3_xml(tools: &[Tool]) -> String`
Converts OpenAI tool definitions to Qwen3 XML format for prompt injection.

### `tools::parse_qwen3_tool_calls(text: &str) -> Result<Vec<ToolCall>>`
Parses `<tool_call>` XML tags from model output and converts to OpenAI ToolCall format.

### `tools::extract_text_content(text: &str) -> String`
Strips `<tool_call>` tags from response, leaving only text content.

### `tools::has_tool_calls(text: &str) -> bool`
Fast check for presence of tool call tags.

## Compatibility

- ✅ Cline (v3.35+)
- ✅ Any OpenAI-compatible client expecting function/tool calling
- ✅ Models using Qwen3-style XML tool format

## Testing

To test tool calling:

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:main",
    "messages": [{"role": "user", "content": "List files in current directory"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "execute_command",
        "description": "Execute a shell command",
        "parameters": {
          "type": "object",
          "properties": {
            "command": {"type": "string"}
          },
          "required": ["command"]
        }
      }
    }]
  }'
```

Expected: Response with `tool_calls` array and `finish_reason: "tool_calls"`.

## Future Enhancements

- [ ] Support for `tool_choice` parameter (force/disable tool use)
- [ ] Parallel tool calling (multiple tools in one response)
- [ ] Tool call result handling in conversation history
- [ ] Support for other XML-based tool formats
