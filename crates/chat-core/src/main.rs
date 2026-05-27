//! chat-core WASI binary.
//!
//! Reads ChatCommand messages from stdin (length-prefixed JSON),
//! processes them through ChatState, writes ChatEvent messages to stdout.
//!
//! In the Wanix environment, stdin/stdout are connected to DMA ring buffers
//! via the VFS pipe abstraction. The host orchestrator reads/writes the
//! other end.
//!
//! Wire format: [4-byte LE length][JSON payload]
//! Each message is a complete JSON object (ChatCommand in, ChatEvent out).
//!
//! This is a minimal WASI wrapper — all logic lives in the chat_core library.

use chat_core::{ChatCommand, ChatEvent, ChatState, ToolCallFormat};
use std::io::{self, Read, Write};

fn read_message(reader: &mut impl Read) -> io::Result<Option<String>> {
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    }
    let len = u32::from_le_bytes(len_buf) as usize;
    if len == 0 || len > 10 * 1024 * 1024 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid message length"));
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map(Some).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn write_message(writer: &mut impl Write, msg: &str) -> io::Result<()> {
    let bytes = msg.as_bytes();
    let len = (bytes.len() as u32).to_le_bytes();
    writer.write_all(&len)?;
    writer.write_all(bytes)?;
    writer.flush()
}

fn event_to_json(event: &ChatEvent) -> String {
    match event {
        ChatEvent::Content(text) => {
            serde_json::json!({"type": "content", "text": text}).to_string()
        }
        ChatEvent::Thinking(text) => {
            serde_json::json!({"type": "thinking", "text": text}).to_string()
        }
        ChatEvent::ToolCallDetected { id, uuid, description, arguments } => {
            serde_json::json!({
                "type": "toolCallDetected",
                "id": id, "uuid": uuid,
                "description": description, "arguments": arguments
            }).to_string()
        }
        ChatEvent::Complete => {
            serde_json::json!({"type": "complete"}).to_string()
        }
        ChatEvent::Error(msg) => {
            serde_json::json!({"type": "error", "message": msg}).to_string()
        }
    }
}

fn json_to_command(json: &str) -> Result<ChatCommand, String> {
    let v: serde_json::Value = serde_json::from_str(json).map_err(|e| e.to_string())?;
    let cmd_type = v["type"].as_str().ok_or("missing type field")?;
    match cmd_type {
        "userMessage" => {
            let text = v["text"].as_str().ok_or("missing text")?.to_owned();
            Ok(ChatCommand::UserMessage(text))
        }
        "inferenceToken" => {
            let token = v["token"].as_str().ok_or("missing token")?.to_owned();
            Ok(ChatCommand::InferenceToken(token))
        }
        "inferenceComplete" => Ok(ChatCommand::InferenceComplete),
        "inferenceError" => {
            let msg = v["message"].as_str().unwrap_or("unknown error").to_owned();
            Ok(ChatCommand::InferenceError(msg))
        }
        "toolCallResult" => {
            let id = v["id"].as_str().ok_or("missing id")?.to_owned();
            let result = v["result"].as_str().ok_or("missing result")?.to_owned();
            Ok(ChatCommand::ToolCallResult { id, result })
        }
        "cancel" => Ok(ChatCommand::Cancel),
        other => Err(format!("unknown command type: {other}")),
    }
}

fn main() {
    // Determine tool call format from environment or default
    let format_str = std::env::var("TOOL_CALL_FORMAT").unwrap_or_default();
    let format = ToolCallFormat::from_architecture(&format_str);

    let mut state = ChatState::new(format);

    // Load tool descriptions from environment if provided
    if let Ok(tools_json) = std::env::var("TOOL_DESCRIPTIONS") {
        if let Ok(tools) = serde_json::from_str::<serde_json::Value>(&tools_json) {
            if let Some(obj) = tools.as_object() {
                for (uuid, desc) in obj {
                    if let Some(desc_str) = desc.as_str() {
                        state.tool_descriptions.insert(uuid.clone(), desc_str.to_owned());
                    }
                }
            }
        }
    }

    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();

    // Send ready signal
    let ready = serde_json::json!({"type": "ready"}).to_string();
    if write_message(&mut stdout, &ready).is_err() {
        return;
    }

    // Main event loop
    loop {
        match read_message(&mut stdin) {
            Ok(Some(json)) => {
                match json_to_command(&json) {
                    Ok(cmd) => {
                        let events = state.handle_command(cmd);
                        for event in &events {
                            let msg = event_to_json(event);
                            if write_message(&mut stdout, &msg).is_err() {
                                return;
                            }
                        }

                        // Check if we need to signal for re-invocation
                        if state.needs_reinvoke() {
                            let reinvoke = serde_json::json!({
                                "type": "reinvoke",
                                "history": serde_json::to_value(&state.history).unwrap_or_default()
                            }).to_string();
                            if write_message(&mut stdout, &reinvoke).is_err() {
                                return;
                            }
                        }
                    }
                    Err(e) => {
                        let err = serde_json::json!({"type": "error", "message": e}).to_string();
                        let _ = write_message(&mut stdout, &err);
                    }
                }
            }
            Ok(None) => break, // EOF
            Err(e) => {
                eprintln!("chat-core: read error: {e}");
                break;
            }
        }
    }
}
