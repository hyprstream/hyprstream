//! Jinja2 template engine for chat templates using minijinja

use anyhow::{Result, anyhow};
use minijinja::{Environment, Value, context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Chat message structure for template rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
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
        
        // We'll add the template dynamically when applying it
        // to avoid lifetime issues
        
        Ok(Self { env, config })
    }
    
    /// Apply chat template to messages
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: Option<bool>,
    ) -> Result<String> {
        // Use provided template or fall back to a default
        let template_str = self.config.chat_template.as_ref()
            .ok_or_else(|| anyhow!("No chat template configured"))?;
        
        // If the template uses Python-style method calls, we need to transform it
        // This is necessary because HuggingFace templates use Python/Jinja2 syntax
        // but minijinja doesn't support method calls on strings
        let transformed = if template_str.contains(".startswith(") || template_str.contains(".endswith(") {
            // Transform Python method calls to minijinja test syntax
            // This is the least brittle approach - we only touch specific patterns
            template_str
                .replace(".startswith(", " is startswith(")
                .replace(".endswith(", " is endswith(")
        } else {
            template_str.to_string()
        };
        
        // Compile the template
        let tmpl = self.env.template_from_str(&transformed)
            .map_err(|e| anyhow!("Template compilation failed: {}. Original template may use unsupported Python syntax.", e))?;
        
        // Prepare context with all special tokens and variables
        let add_gen = add_generation_prompt.unwrap_or(self.config.add_generation_prompt);
        
        // Render the template
        let rendered = tmpl.render(context! {
            messages => messages,
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

{% endif %}{% endfor %}{% if add_generation_prompt %}### Assistant: {% endif %}"#.to_string()
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
{% endif %}"#.to_string()
            }
            "mistral" | "mixtral" => {
                // Mistral/Mixtral template
                r#"{% for message in messages %}
{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]
{% elif message['role'] == 'assistant' %}{{ message['content'] }}</s>
{% endif %}{% endfor %}"#.to_string()
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
{% endif %}"#.to_string()
            }
            "chatml" | "chatgpt" => {
                // ChatML format (GPT-style)
                r#"{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"#.to_string()
            }
            _ => {
                // Default simple template
                r#"{% for message in messages %}
{{ message['role'] }}: {{ message['content'] }}
{% endfor %}
{% if add_generation_prompt %}assistant: {% endif %}"#.to_string()
            }
        }
    }
    
    /// Parse template configuration from tokenizer_config.json
    pub fn from_tokenizer_config(config_json: &serde_json::Value) -> Result<TemplateConfig> {
        let mut template_config = TemplateConfig::default();
        
        // Extract chat template
        if let Some(template) = config_json.get("chat_template").and_then(|v| v.as_str()) {
            template_config.chat_template = Some(template.to_string());
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
        if let Some(add_gen) = config_json.get("add_generation_prompt").and_then(|v| v.as_bool()) {
            template_config.add_generation_prompt = add_gen;
        }
        
        // Build special tokens map for convenience
        if let Some(ref bos) = template_config.bos_token {
            template_config.special_tokens.insert("bos_token".to_string(), bos.clone());
        }
        if let Some(ref eos) = template_config.eos_token {
            template_config.special_tokens.insert("eos_token".to_string(), eos.clone());
        }
        
        Ok(template_config)
    }
}

/// Extract token value from JSON (handles both string and object formats)
fn extract_token_value(value: &serde_json::Value) -> Option<String> {
    if let Some(s) = value.as_str() {
        Some(s.to_string())
    } else if let Some(obj) = value.as_object() {
        obj.get("content").and_then(|v| v.as_str()).map(|s| s.to_string())
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

/// Custom filter for JSON serialization
fn tojson_filter(value: &Value) -> Result<Value, minijinja::Error> {
    let json_str = serde_json::to_string(&value)
        .map_err(|e| minijinja::Error::new(
            minijinja::ErrorKind::InvalidOperation,
            format!("Failed to serialize to JSON: {}", e)
        ))?;
    Ok(Value::from(json_str))
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_qwen_template() {
        let config = TemplateConfig {
            chat_template: Some(r#"{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"#.to_string()),
            ..Default::default()
        };
        
        let engine = TemplateEngine::new(config).unwrap();
        let messages = vec![
            ChatMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
        ];
        
        let result = engine.apply_chat_template(&messages, Some(true)).unwrap();
        assert!(result.contains("<|im_start|>user"));
        assert!(result.contains("Hello"));
        assert!(result.contains("<|im_start|>assistant"));
    }
    
    #[test]
    fn test_fallback_templates() {
        for model_type in &["llama", "qwen", "mistral", "gemma", "chatml"] {
            let template = TemplateEngine::get_fallback_template(model_type);
            assert!(!template.is_empty());
        }
    }
    
    #[test]
    fn test_huggingface_template_with_python_methods() {
        // Real HuggingFace template that uses Python-style .startswith() method
        let config = TemplateConfig {
            chat_template: Some(r#"{% for message in messages %}
{% if message['role'].startswith('sys') %}System: {{ message['content'] }}
{% elif message['role'].endswith('er') %}User: {{ message['content'] }}
{% elif message['role'].startswith('assist') %}Assistant: {{ message['content'] }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}"#.to_string()),
            ..Default::default()
        };
        
        let engine = TemplateEngine::new(config).unwrap();
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello!".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "Hi there!".to_string(),
            },
        ];
        
        let result = engine.apply_chat_template(&messages, Some(true)).unwrap();
        
        // Verify the template was processed correctly
        assert!(result.contains("System: You are a helpful assistant."));
        assert!(result.contains("User: Hello!"));
        assert!(result.contains("Assistant: Hi there!"));
        assert!(result.ends_with("Assistant: "));
    }
    
    #[test]
    fn test_complex_template_with_conditions() {
        // Complex template with nested conditions and multiple Python methods
        let config = TemplateConfig {
            chat_template: Some(r#"{% for message in messages %}
{% if message['role'].startswith('sys') and message['content'] != 'test' %}
[SYSTEM] {{ message['content'] }}
{% elif message['role'].startswith('u') %}
[USER] {{ message['content'] }}
{% elif message['role'].endswith('ant') %}
[ASSISTANT] {{ message['content'] }}
{% endif %}
{% endfor %}"#.to_string()),
            ..Default::default()
        };
        
        let engine = TemplateEngine::new(config).unwrap();
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "Configure the model".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "What's 2+2?".to_string(),
            },
            ChatMessage {
                role: "assistant".to_string(),
                content: "4".to_string(),
            },
        ];
        
        let result = engine.apply_chat_template(&messages, Some(false)).unwrap();
        assert!(result.contains("[SYSTEM] Configure the model"));
        assert!(result.contains("[USER] What's 2+2?"));
        assert!(result.contains("[ASSISTANT] 4"));
    }
}