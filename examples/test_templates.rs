//! Test template engine functionality

use hyprstream_core::runtime::template_engine::{TemplateEngine, TemplateConfig, ChatMessage};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing Jinja2 template engine for chat templates\n");
    
    // Test 1: Qwen2 template
    test_qwen2_template()?;
    
    // Test 2: Llama template
    test_llama_template()?;
    
    // Test 3: Gemma template
    test_gemma_template()?;
    
    // Test 4: ChatML template
    test_chatml_template()?;
    
    println!("\n✅ All template tests passed!");
    Ok(())
}

fn test_qwen2_template() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Qwen2 Template ===");
    
    let config = TemplateConfig {
        chat_template: Some(r#"{% for message in messages %}{{'<|im_start|>' + message['role'] + '
' + message['content'] + '<|im_end|>' + '
'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant
' }}{% endif %}"#.to_string()),
        ..Default::default()
    };
    
    let engine = TemplateEngine::new(config)?;
    
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Hello, how are you?".to_string(),
        },
    ];
    
    let result = engine.apply_chat_template(&messages, Some(true))?;
    println!("Output:\n{}", result);
    
    assert!(result.contains("<|im_start|>system"));
    assert!(result.contains("<|im_start|>user"));
    assert!(result.contains("<|im_start|>assistant"));
    assert!(result.contains("<|im_end|>"));
    
    Ok(())
}

fn test_llama_template() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing Llama Template ===");
    
    let template = TemplateEngine::get_fallback_template("llama");
    let config = TemplateConfig {
        chat_template: Some(template),
        ..Default::default()
    };
    
    let engine = TemplateEngine::new(config)?;
    
    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "What is 2+2?".to_string(),
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "2+2 equals 4.".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Are you sure?".to_string(),
        },
    ];
    
    let result = engine.apply_chat_template(&messages, Some(true))?;
    println!("Output:\n{}", result);
    
    assert!(result.contains("### Human:"));
    assert!(result.contains("### Assistant:"));
    
    Ok(())
}

fn test_gemma_template() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing Gemma Template ===");
    
    let template = TemplateEngine::get_fallback_template("gemma");
    let config = TemplateConfig {
        chat_template: Some(template),
        ..Default::default()
    };
    
    let engine = TemplateEngine::new(config)?;
    
    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Explain quantum computing".to_string(),
        },
    ];
    
    let result = engine.apply_chat_template(&messages, Some(true))?;
    println!("Output:\n{}", result);
    
    assert!(result.contains("<start_of_turn>user"));
    assert!(result.contains("<end_of_turn>"));
    assert!(result.contains("<start_of_turn>model"));
    
    Ok(())
}

fn test_chatml_template() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing ChatML Template ===");
    
    let template = TemplateEngine::get_fallback_template("chatml");
    let config = TemplateConfig {
        chat_template: Some(template),
        bos_token: Some("<s>".to_string()),
        eos_token: Some("</s>".to_string()),
        ..Default::default()
    };
    
    let engine = TemplateEngine::new(config)?;
    
    let messages = vec![
        ChatMessage {
            role: "system".to_string(),
            content: "You are ChatGPT".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: "Tell me a joke".to_string(),
        },
    ];
    
    let result = engine.apply_chat_template(&messages, Some(true))?;
    println!("Output:\n{}", result);
    
    assert!(result.contains("<|im_start|>system"));
    assert!(result.contains("<|im_start|>user"));
    assert!(result.contains("<|im_start|>assistant"));
    assert!(result.contains("<|im_end|>"));
    
    Ok(())
}

fn test_template_with_special_tokens() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n=== Testing Template with Special Tokens ===");
    
    let mut special_tokens = HashMap::new();
    special_tokens.insert("bos_token".to_string(), "<s>".to_string());
    special_tokens.insert("eos_token".to_string(), "</s>".to_string());
    
    let config = TemplateConfig {
        chat_template: Some(r#"{{ bos_token }}{% for message in messages %}{{ message.role }}: {{ message.content }}{{ eos_token }}{% endfor %}"#.to_string()),
        bos_token: Some("<s>".to_string()),
        eos_token: Some("</s>".to_string()),
        special_tokens,
        ..Default::default()
    };
    
    let engine = TemplateEngine::new(config)?;
    
    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "Test message".to_string(),
        },
    ];
    
    let result = engine.apply_chat_template(&messages, Some(false))?;
    println!("Output:\n{}", result);
    
    assert!(result.starts_with("<s>"));
    assert!(result.contains("</s>"));
    
    Ok(())
}