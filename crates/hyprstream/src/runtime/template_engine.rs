//! Jinja2 template engine for chat templates using minijinja

use anyhow::{anyhow, Result};
use minijinja::{context, Environment, Value};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Chat message structure for template rendering.
///
/// Fields match what HuggingFace chat templates expect:
/// - `content` is Optional (can be null for tool-call-only assistant messages)
/// - `tool_calls` carries tool call objects on assistant messages
/// - `tool_call_id` identifies which tool call a "tool" role message responds to
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Template configuration loaded from tokenizer_config.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateConfig {
    /// The Jinja2 chat template string
    pub chat_template: Option<String>,
    /// Special tokens mapping
    pub special_tokens: HashMap<String, String>,
    /// Whether to add generation prompt
    pub add_generation_prompt: bool,
    /// BOS token
    pub bos_token: Option<String>,
    /// EOS token
    pub eos_token: Option<String>,
    /// PAD token
    pub pad_token: Option<String>,
    /// UNK token
    pub unk_token: Option<String>,
    /// SEP token
    pub sep_token: Option<String>,
    /// CLS token
    pub cls_token: Option<String>,
    /// Additional tokens
    pub additional_special_tokens: Vec<String>,
}

impl Default for TemplateConfig {
    fn default() -> Self {
        Self {
            chat_template: None,
            special_tokens: HashMap::new(),
            add_generation_prompt: true,
            bos_token: None,
            eos_token: None,
            pad_token: None,
            unk_token: None,
            sep_token: None,
            cls_token: None,
            additional_special_tokens: Vec::new(),
        }
    }
}

/// Template engine for rendering chat templates
pub struct TemplateEngine {
    env: Environment<'static>,
    config: TemplateConfig,
}

impl TemplateEngine {
    /// Create a new template engine with the given configuration
    pub fn new(config: TemplateConfig) -> Result<Self> {
        let mut env = Environment::new();

        // Register common filters that HuggingFace templates might use
        env.add_filter("length", length_filter);
        env.add_filter("tojson", tojson_filter);
        env.add_filter("strip", strip_filter);
        env.add_filter("rstrip", rstrip_filter);
        env.add_filter("lstrip", lstrip_filter);
        env.add_filter("split_first", split_first_filter);
        env.add_filter("split_last", split_last_filter);

        // Add custom tests for string operations
        // These can be used as: {% if value is startswith("prefix") %}
        env.add_test("startswith", |value: &str, prefix: &str| -> bool {
            value.starts_with(prefix)
        });
        env.add_test("endswith", |value: &str, suffix: &str| -> bool {
            value.ends_with(suffix)
        });

        // Also add as filters for compatibility: {{ value|startswith("prefix") }}
        env.add_filter("startswith", |value: &str, prefix: &str| -> bool {
            value.starts_with(prefix)
        });
        env.add_filter("endswith", |value: &str, suffix: &str| -> bool {
            value.ends_with(suffix)
        });

        // Register raise_exception() — used by Mistral templates for validation
        env.add_function("raise_exception", raise_exception_fn);

        // We'll add the template dynamically when applying it
        // to avoid lifetime issues

        Ok(Self { env, config })
    }

    /// Apply chat template to messages.
    ///
    /// `tools` is an optional JSON value (array of tool definitions) that will be
    /// passed to the template as the `tools` variable. HuggingFace chat templates
    /// for tool-calling models (Qwen3, Llama 3.1, Mistral, etc.) use this variable
    /// to format tool descriptions natively.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: Option<bool>,
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        // Use provided template or fall back to a default
        let template_str = self
            .config
            .chat_template
            .as_ref()
            .ok_or_else(|| anyhow!("No chat template configured"))?;

        // Transform Python-style method calls to minijinja syntax if needed
        // HuggingFace templates use Python/Jinja2 syntax but minijinja uses test syntax
        let transformed = template_str
            .replace(".startswith(", " is startswith(")
            .replace(".endswith(", " is endswith(");

        // Transform .split('sep')[0] → |split_first('sep') and .split('sep')[-1] → |split_last('sep')
        // Must use regex because the pattern includes a subscript index after the call.
        let transformed = transform_split_calls(&transformed);
        // Transform .strip(...), .rstrip(...), .lstrip(...) → filter syntax
        // Must do .strip( before .rstrip(/.lstrip( since the latter contain "strip("
        let transformed = transform_strip_calls(&transformed);

        // Compile the template
        let tmpl = self.env.template_from_str(&transformed)
            .map_err(|e| anyhow!("Template compilation failed: {}. Original template may use unsupported Python syntax.", e))?;

        // Prepare context with all special tokens and variables
        let add_gen = add_generation_prompt.unwrap_or(self.config.add_generation_prompt);

        // Convert tools to minijinja Value (or undefined if None)
        let tools_value = match tools {
            Some(t) => Value::from_serialize(t),
            None => Value::UNDEFINED,
        };

        // Render the template
        let rendered = tmpl.render(context! {
            messages => messages,
            tools => tools_value,
            bos_token => self.config.bos_token.as_deref().unwrap_or(""),
            eos_token => self.config.eos_token.as_deref().unwrap_or(""),
            pad_token => self.config.pad_token.as_deref().unwrap_or(""),
            unk_token => self.config.unk_token.as_deref().unwrap_or(""),
            sep_token => self.config.sep_token.as_deref().unwrap_or(""),
            cls_token => self.config.cls_token.as_deref().unwrap_or(""),
            additional_special_tokens => &self.config.additional_special_tokens,
            add_generation_prompt => add_gen,
        })?;

        Ok(rendered)
    }

    /// Get a fallback template for common model architectures
    pub fn get_fallback_template(model_type: &str) -> String {
        match model_type.to_lowercase().as_str() {
            "llama" | "llama2" | "llama3" => {
                // Llama-style template
                r#"{% for message in messages %}
{% if message['role'] == 'system' %}{{ message['content'] }}

{% elif message['role'] == 'user' %}### Human: {{ message['content'] }}

{% elif message['role'] == 'assistant' %}### Assistant: {{ message['content'] }}

{% endif %}{% endfor %}{% if add_generation_prompt %}### Assistant: {% endif %}"#.to_owned()
            }
            "qwen" | "qwen2" => {
                // Qwen2-style template with special tokens
                r#"{% for message in messages %}
{%- if message['role'] == 'system' -%}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#.to_owned()
            }
            "mistral" | "mixtral" => {
                // Mistral/Mixtral template
                r#"{% for message in messages %}
{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]
{% elif message['role'] == 'assistant' %}{{ message['content'] }}</s>
{% endif %}{% endfor %}"#.to_owned()
            }
            "gemma" | "gemma2" => {
                // Gemma-style template
                r#"{% for message in messages %}
{% if message['role'] == 'user' %}<start_of_turn>user
{{ message['content'] }}<end_of_turn>
{% elif message['role'] == 'assistant' %}<start_of_turn>model
{{ message['content'] }}<end_of_turn>
{% endif %}{% endfor %}
{% if add_generation_prompt %}<start_of_turn>model
{% endif %}"#.to_owned()
            }
            "chatml" | "chatgpt" => {
                // ChatML format (GPT-style)
                r#"{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#.to_owned()
            }
            "janus" | "janus-1.3b" | "janus-pro" => {
                // DeepSeek/Janus format - simple User:/Assistant: format
                r#"{% for message in messages %}
{% if message['role'] == 'user' %}User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
{% endif %}{% endfor %}
{% if add_generation_prompt %}Assistant: {% endif %}"#.to_owned()
            }
            _ => {
                // Default simple template
                r#"{% for message in messages %}
{{ message['role'] }}: {{ message['content'] }}
{% endfor %}
{% if add_generation_prompt %}assistant: {% endif %}"#.to_owned()
            }
        }
    }

    /// Parse template configuration from tokenizer_config.json
    pub fn from_tokenizer_config(config_json: &serde_json::Value) -> Result<TemplateConfig> {
        let mut template_config = TemplateConfig::default();

        // Extract chat template
        if let Some(template) = config_json.get("chat_template").and_then(|v| v.as_str()) {
            template_config.chat_template = Some(template.to_owned());
        }

        // Extract special tokens
        if let Some(bos) = config_json.get("bos_token") {
            template_config.bos_token = extract_token_value(bos);
        }
        if let Some(eos) = config_json.get("eos_token") {
            template_config.eos_token = extract_token_value(eos);
        }
        if let Some(pad) = config_json.get("pad_token") {
            template_config.pad_token = extract_token_value(pad);
        }
        if let Some(unk) = config_json.get("unk_token") {
            template_config.unk_token = extract_token_value(unk);
        }
        if let Some(sep) = config_json.get("sep_token") {
            template_config.sep_token = extract_token_value(sep);
        }
        if let Some(cls) = config_json.get("cls_token") {
            template_config.cls_token = extract_token_value(cls);
        }

        // Extract additional special tokens
        if let Some(additional) = config_json.get("additional_special_tokens") {
            if let Some(arr) = additional.as_array() {
                for token in arr {
                    if let Some(token_str) = extract_token_value(token) {
                        template_config.additional_special_tokens.push(token_str);
                    }
                }
            }
        }

        // Check for add_generation_prompt setting
        if let Some(add_gen) = config_json
            .get("add_generation_prompt")
            .and_then(serde_json::Value::as_bool)
        {
            template_config.add_generation_prompt = add_gen;
        }

        // Build special tokens map for convenience
        if let Some(ref bos) = template_config.bos_token {
            template_config
                .special_tokens
                .insert("bos_token".to_owned(), bos.clone());
        }
        if let Some(ref eos) = template_config.eos_token {
            template_config
                .special_tokens
                .insert("eos_token".to_owned(), eos.clone());
        }

        Ok(template_config)
    }
}

/// Extract token value from JSON (handles both string and object formats)
fn extract_token_value(value: &serde_json::Value) -> Option<String> {
    if let Some(s) = value.as_str() {
        Some(s.to_owned())
    } else if let Some(obj) = value.as_object() {
        obj.get("content")
            .and_then(|v| v.as_str())
            .map(std::borrow::ToOwned::to_owned)
    } else {
        None
    }
}

/// Custom filter for getting length
fn length_filter(value: &Value) -> Result<Value, minijinja::Error> {
    // In minijinja 2.12.0, try_iter returns a Result
    if let Ok(iter) = value.try_iter() {
        Ok(Value::from(iter.count()))
    } else if let Some(s) = value.as_str() {
        // Use char count instead of byte length for proper Unicode support
        Ok(Value::from(s.chars().count()))
    } else {
        Ok(Value::from(0))
    }
}

/// Custom filter for JSON serialization.
///
/// Supports optional `indent` keyword argument for pretty-printing, matching
/// Jinja2's `tojson(indent=4)` syntax used by Llama 3.1 templates.
fn tojson_filter(value: &Value, kwargs: minijinja::value::Kwargs) -> Result<Value, minijinja::Error> {
    let indent: Option<usize> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;

    let json_str = if let Some(n) = indent {
        // Pretty-print with the requested indentation
        let v: serde_json::Value = serde_json::from_str(
            &serde_json::to_string(&value).map_err(|e| {
                minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    format!("Failed to serialize to JSON: {e}"),
                )
            })?,
        )
        .map_err(|e| {
            minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                format!("Failed to re-parse JSON for indentation: {e}"),
            )
        })?;
        let indent_bytes = b" ".repeat(n);
        let formatter = serde_json::ser::PrettyFormatter::with_indent(&indent_bytes);
        let mut buf = Vec::new();
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
        serde::Serialize::serialize(&v, &mut ser).map_err(|e| {
            minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                format!("Failed to pretty-print JSON: {e}"),
            )
        })?;
        String::from_utf8(buf).unwrap_or_default()
    } else {
        serde_json::to_string(&value).map_err(|e| {
            minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                format!("Failed to serialize to JSON: {e}"),
            )
        })?
    };
    Ok(Value::from(json_str))
}

/// Custom function: `raise_exception(message)` — used by Mistral templates for validation.
fn raise_exception_fn(msg: String) -> Result<Value, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

/// Custom filter for Python's .strip() / .strip(chars)
fn strip_filter(value: &Value, chars: Option<&str>) -> Result<Value, minijinja::Error> {
    if let Some(s) = value.as_str() {
        match chars {
            Some(c) => {
                let char_list: Vec<char> = c.chars().collect();
                Ok(Value::from(s.trim_matches(char_list.as_slice())))
            }
            None => Ok(Value::from(s.trim())),
        }
    } else {
        Ok(value.clone())
    }
}

/// Custom filter for Python's .rstrip(chars)
fn rstrip_filter(value: &Value, chars: &str) -> Result<Value, minijinja::Error> {
    if let Some(s) = value.as_str() {
        let char_list: Vec<char> = chars.chars().collect();
        Ok(Value::from(s.trim_end_matches(char_list.as_slice())))
    } else {
        Ok(value.clone())
    }
}

/// Custom filter for Python's .lstrip(chars)
fn lstrip_filter(value: &Value, chars: &str) -> Result<Value, minijinja::Error> {
    if let Some(s) = value.as_str() {
        let char_list: Vec<char> = chars.chars().collect();
        Ok(Value::from(s.trim_start_matches(char_list.as_slice())))
    } else {
        Ok(value.clone())
    }
}

/// Custom filter: equivalent to Python's .split(sep)[0]
fn split_first_filter(value: &Value, sep: &str) -> Result<Value, minijinja::Error> {
    if let Some(s) = value.as_str() {
        Ok(Value::from(s.split(sep).next().unwrap_or("")))
    } else {
        Ok(value.clone())
    }
}

/// Custom filter: equivalent to Python's .split(sep)[-1]
fn split_last_filter(value: &Value, sep: &str) -> Result<Value, minijinja::Error> {
    if let Some(s) = value.as_str() {
        Ok(Value::from(s.rsplit(sep).next().unwrap_or("")))
    } else {
        Ok(value.clone())
    }
}

/// Transform Python `.split('sep')[idx]` calls to minijinja filter syntax.
///
/// - `.split('sep')[0]`  → `|split_first('sep')`
/// - `.split('sep')[-1]` → `|split_last('sep')`
fn transform_split_calls(template: &str) -> String {
    // Rust regex doesn't support backreferences, so handle single and double quotes separately
    let re_single = Regex::new(r"\.split\('([^']*)'\)\[(-?\d+)\]").unwrap();
    let re_double = Regex::new(r#"\.split\("([^"]*)"\)\[(-?\d+)\]"#).unwrap();

    let result = re_single.replace_all(template, |caps: &regex::Captures| {
        split_replacement(&caps[1], &caps[2])
    });
    let result = re_double.replace_all(&result, |caps: &regex::Captures| {
        split_replacement(&caps[1], &caps[2])
    });
    result.into_owned()
}

/// Transform Python `.strip(...)`, `.rstrip(...)`, `.lstrip(...)` calls to filter syntax.
///
/// Handles both no-arg (`.strip()`) and parameterized (`.strip('\n')`) forms.
/// Must be applied carefully: `.rstrip(` and `.lstrip(` are handled first to avoid
/// the `.strip(` replacement matching the suffix of `.rstrip(` / `.lstrip(`.
fn transform_strip_calls(template: &str) -> String {
    // Order matters: replace .rstrip/.lstrip before .strip to avoid partial matches
    template
        .replace(".rstrip(", "|rstrip(")
        .replace(".lstrip(", "|lstrip(")
        .replace(".strip(", "|strip(")
}

fn split_replacement(sep: &str, idx_str: &str) -> String {
    let idx: i64 = idx_str.parse().unwrap_or(0);
    match idx {
        0 => format!("|split_first('{sep}')"),
        -1 => format!("|split_last('{sep}')"),
        _ => format!(".split('{sep}')[{idx}]"), // leave unsupported indices unchanged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwen_template() -> Result<()> {
        let config = TemplateConfig {
            chat_template: Some(
                r#"{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"#.to_owned(),
            ),
            ..Default::default()
        };

        let engine = TemplateEngine::new(config)?;
        let messages = vec![ChatMessage { role: "user".into(), content: Some("Hello".into()), ..Default::default() }];

        let result = engine.apply_chat_template(&messages, Some(true), None)?;
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello"));
        assert!(result.contains("<|im_start|>assistant"));
        Ok(())
    }

    #[test]
    fn test_fallback_templates() {
        for model_type in &["llama", "qwen", "mistral", "gemma", "chatml"] {
            let template = TemplateEngine::get_fallback_template(model_type);
            assert!(!template.is_empty());
        }
    }

    #[test]
    fn test_huggingface_template_with_python_methods() -> Result<()> {
        // Real HuggingFace template that uses Python-style .startswith() method
        let config = TemplateConfig {
            chat_template: Some(
                r#"{% for message in messages %}
{% if message['role'].startswith('sys') %}System: {{ message['content'] }}
{% elif message['role'].endswith('er') %}User: {{ message['content'] }}
{% elif message['role'].startswith('assist') %}Assistant: {{ message['content'] }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"#.to_owned(),
            ),
            ..Default::default()
        };

        let engine = TemplateEngine::new(config)?;
        let messages = vec![
            ChatMessage { role: "system".into(), content: Some("You are a helpful assistant.".into()), ..Default::default() },
            ChatMessage { role: "user".into(), content: Some("Hello!".into()), ..Default::default() },
            ChatMessage { role: "assistant".into(), content: Some("Hi there!".into()), ..Default::default() },
        ];

        let result = engine.apply_chat_template(&messages, Some(true), None)?;

        // Verify the template was processed correctly
        assert!(result.contains("System: You are a helpful assistant."));
        assert!(result.contains("User: Hello!"));
        assert!(result.contains("Assistant: Hi there!"));
        assert!(result.ends_with("Assistant: "));
        Ok(())
    }

    #[test]
    fn test_complex_template_with_conditions() -> Result<()> {
        // Complex template with nested conditions and multiple Python methods
        let config = TemplateConfig {
            chat_template: Some(
                r#"{% for message in messages %}
{% if message['role'].startswith('sys') and message['content'] != 'test' %}
[SYSTEM] {{ message['content'] }}
{% elif message['role'].startswith('u') %}
[USER] {{ message['content'] }}
{% elif message['role'].endswith('ant') %}
[ASSISTANT] {{ message['content'] }}
{% endif %}
{% endfor %}"#.to_owned(),
            ),
            ..Default::default()
        };

        let engine = TemplateEngine::new(config)?;
        let messages = vec![
            ChatMessage { role: "system".into(), content: Some("Configure the model".into()), ..Default::default() },
            ChatMessage { role: "user".into(), content: Some("What's 2+2?".into()), ..Default::default() },
            ChatMessage { role: "assistant".into(), content: Some("4".into()), ..Default::default() },
        ];

        let result = engine.apply_chat_template(&messages, Some(false), None)?;
        assert!(result.contains("[SYSTEM] Configure the model"));
        assert!(result.contains("[USER] What's 2+2?"));
        assert!(result.contains("[ASSISTANT] 4"));
        Ok(())
    }

    #[test]
    fn test_strip_filter() -> Result<()> {
        // Test template with .strip() method (Python-style)
        let config = TemplateConfig {
            chat_template: Some(
                r#"{% for message in messages %}
{{ message['content'].strip() }}
{% endfor %}"#.to_owned(),
            ),
            ..Default::default()
        };

        let engine = TemplateEngine::new(config)?;
        let messages = vec![ChatMessage { role: "user".into(), content: Some("  Hello World  ".into()), ..Default::default() }];

        let result = engine.apply_chat_template(&messages, Some(false), None)?;
        assert!(result.contains("Hello World"));
        assert!(!result.contains("  Hello World  "));
        Ok(())
    }

    #[test]
    fn test_split_and_strip_transforms() {
        // Simulates the Qwen3 template's <think> tag extraction:
        //   content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
        //   content.split('</think>')[-1].lstrip('\n')
        let config = TemplateConfig {
            chat_template: Some(
                r#"{%- for message in messages -%}
{%- set content = message['content'] -%}
{%- if '</think>' in content -%}
{%- set reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') -%}
{%- set content = content.split('</think>')[-1].lstrip('\n') -%}
THINK:{{ reasoning }}
CONTENT:{{ content }}
{%- else -%}
CONTENT:{{ content }}
{%- endif -%}
{%- endfor -%}"#.to_owned(),
            ),
            ..Default::default()
        };

        let engine = TemplateEngine::new(config).expect("test: create template engine");

        // Message with think tags
        let messages = vec![ChatMessage {
            role: "assistant".into(),
            content: Some("<think>\nI should search.\n</think>\nHere is the answer.".into()),
            ..Default::default()
        }];

        let result = engine.apply_chat_template(&messages, Some(false), None).expect("test: apply template");
        assert!(result.contains("THINK:I should search."));
        assert!(result.contains("CONTENT:Here is the answer."));

        // Message without think tags
        let messages_no_think = vec![ChatMessage { role: "user".into(), content: Some("Hello".into()), ..Default::default() }];

        let result = engine.apply_chat_template(&messages_no_think, Some(false), None).expect("test: apply no-think");
        assert!(result.contains("CONTENT:Hello"));
    }

    #[test]
    fn test_transform_split_calls() {
        assert_eq!(
            transform_split_calls("x.split('</think>')[0]"),
            "x|split_first('</think>')"
        );
        assert_eq!(
            transform_split_calls("x.split('<think>')[-1]"),
            "x|split_last('<think>')"
        );
        assert_eq!(
            transform_split_calls(r#"x.split("sep")[0]"#),
            "x|split_first('sep')"
        );
    }
}
