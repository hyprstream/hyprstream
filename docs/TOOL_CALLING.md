# Tool Calling

## Overview

HyprStream supports OpenAI-compatible tool calling (function calling) for
compatibility with Cline and other coding frontends, across multiple model
families (Qwen3, Qwen3.5, Llama 3.1+, Mistral).

The design splits into two independent halves:

- **Input (prompt) side** — tool definitions are formatted by the model's own
  HuggingFace chat template. There is no hand-rolled prompt injection.
- **Output (parsing) side** — model output is parsed into OpenAI `ToolCall`
  JSON using a per-model-family `ToolCallFormat`.

### Files

- **`crates/hyprstream/src/api/tools.rs`** — output-side format detection and parsing
- **`crates/hyprstream/src/runtime/template_engine.rs`** — chat template application (input side)
- **`crates/hyprstream/src/server/routes/openai.rs`** — integration into the chat completion endpoints

## Input Side: Chat Templates

When a client sends `tools` in the request:

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

the route serializes the tool array to JSON and passes it through the
`apply_chat_template` RPC (`server/routes/openai.rs:393-418`). The template
engine (`runtime/template_engine.rs:155-218`) exposes it to the model's
HuggingFace chat template as the standard `tools` minijinja variable — the same
variable HF tokenizers use — so each model family's own template formats the
tool descriptions natively (Qwen's `<tools>` block, Llama 3.1's JSON preamble,
Mistral's `[AVAILABLE_TOOLS]`, etc.). The server never constructs
format-specific tool markup itself.

## Output Side: `ToolCallFormat`

Parsing is per-model-family, selected from the model architecture at request
time (`api/tools.rs:23-35`):

```rust
pub enum ToolCallFormat {
    /// Qwen3 XML: <tool_call>{"name":…,"arguments":…}</tool_call>
    Qwen3Xml,
    /// Qwen3.5 XML parameter format:
    /// <tool_call><function=NAME><parameter=KEY>value</parameter>…</function></tool_call>
    Qwen35XmlParam,
    /// Llama 3.1+: <|python_tag|> prefix + JSON {"name":…,"parameters":…}
    LlamaJson,
    /// Mistral: [TOOL_CALLS] prefix + JSON array
    MistralJson,
    /// Model does not support tool calling.
    None,
}
```

Selection: `ToolCallFormat::from_architecture()` (from the detected
`ModelArchitecture`), with `from_architecture_str()` / `from_model_ref()`
variants for string inputs.

The format-aware API (`api/tools.rs:85-115`) is what the routes use:

- `has_tool_calls_for_format(format, text)` — fast marker check
- `parse_tool_calls_for_format(format, text)` — parse into OpenAI `ToolCall`s
- `extract_text_content_for_format(format, text)` — strip tool-call markup,
  leaving the plain text content

The bare `has_tool_calls()` / `extract_text_content()` functions are
backwards-compatibility shims that hardcode the Qwen3 format — prefer the
`*_for_format` variants.

## Request Flow

### Non-streaming (`/v1/chat/completions`)

1. Serialize `tools` → pass to the chat template as the `tools` variable
2. Determine `ToolCallFormat` from the model architecture
3. Generate response
4. Parse tool calls from the accumulated text with the selected format
5. Return with `finish_reason: "tool_calls"` if any were found:

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

### Streaming (`stream: true`)

1. Same input handling
2. Stream tokens as generated (accumulating text internally)
3. On completion: parse the accumulated text, send tool calls in a delta chunk,
   then a final chunk with `finish_reason: "tool_calls"`

## Parallel Tool Calls

Multiple tool calls in one response are supported: every parser returns a
`Vec<ToolCall>`, extracting each `<tool_call>` block (or each element of
Mistral's JSON array) in order. The parsers are also tolerant of common model
misbehavior (duplicated `<tool_call>` tags, non-JSON blocks are skipped).

## Compatibility

- Cline (v3.35+) and any OpenAI-compatible client expecting function/tool calling
- Qwen3-family (XML/JSON), Qwen3.5 (XML parameter format), Llama 3.1+
  (`<|python_tag|>` JSON), Mistral (`[TOOL_CALLS]` JSON array)
- Models without a known tool format (`ToolCallFormat::None`) pass text through
  unchanged

## Testing

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

Expected: response with a `tool_calls` array and `finish_reason: "tool_calls"`.

## Known Limitations

- The `tool_choice` parameter is accepted but ignored — tool use cannot yet be
  forced or disabled per-request.
