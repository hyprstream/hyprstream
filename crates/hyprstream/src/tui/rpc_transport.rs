//! RPC-backed StreamSpawner and ToolCaller for ChatApp inference.
//!
//! Deduplicated from `shell_handlers.rs` and `service.rs` — both callers now
//! use `make_chat_spawner` and (optionally) `make_tool_caller` from this module.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use ed25519_dalek::SigningKey;

/// Build a `StreamSpawner` that drives inference via `ModelClient`.
///
/// Spawns a dedicated OS thread with a single-threaded Tokio runtime per
/// user message.  Applies the chat template, starts an authenticated inference
/// stream, and sends `ChatEvent` variants back through the provided channel.
///
/// `available_tools` is closure-captured and passed to `apply_chat_template`
/// so the model's chat template can inject the tool list into the prompt.
pub fn make_chat_spawner(
    signing_key: &SigningKey,
    model_ref: &str,
    available_tools: Option<Vec<serde_json::Value>>,
    gen_config: Arc<RwLock<hyprstream_tui::chat_app::ChatGenConfig>>,
) -> hyprstream_tui::chat_app::StreamSpawner {
    use hyprstream_rpc::streaming::StreamPayload;

    use crate::services::generated::inference_client::{ChatMessage, ToolCall, ToolCallFunction};
    use crate::runtime::GenerationRequest;
    use crate::services::generated::model_client::ModelClient;
    use hyprstream_tui::chat_app::{ChatEvent, ChatHistoryEntry, ChatRole};

    let sk = signing_key.clone();
    let mr = model_ref.to_owned();
    let tools_json = available_tools.map(serde_json::Value::Array);
    let gen_config_arc = gen_config;

    Box::new(move |history: Vec<ChatHistoryEntry>, event_tx| {
        use hyprstream_tui::chat_app::CancelHandle;
        use tokio::sync::oneshot;

        let sk_inner = sk.clone();
        let mr_inner = mr.clone();
        let tx = event_tx.clone();
        let tools_inner = tools_json.clone();
        let gen_cfg = gen_config_arc.read().clone();

        let (cancel_tx, cancel_rx) = oneshot::channel::<()>();

        std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(ChatEvent::StreamError(e.to_string()));
                    return;
                }
            };

            rt.block_on(async move {
                // Resolve model service key via PolicyClient.
                let policy_vk = sk_inner.verifying_key();
                let policy_client = match crate::services::PolicyClient::from_installed_resolver(
                    sk_inner.clone(),
                    policy_vk,
                    None,
                ) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(ChatEvent::StreamError(format!("Failed to create PolicyClient: {e}")));
                        return;
                    }
                };
                let model_key_resp = match policy_client.resolve_service_key(
                    &crate::services::generated::policy_client::ResolveServiceKey {
                        service_name: "model".to_owned(),
                    },
                ).await {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx.send(ChatEvent::StreamError(format!("Failed to resolve model key: {e}")));
                        return;
                    }
                };
                let model_vk = match <[u8; 32]>::try_from(model_key_resp.verifying_key.as_slice()) {
                    Ok(bytes) => match hyprstream_rpc::crypto::VerifyingKey::from_bytes(&bytes) {
                        Ok(vk) => vk,
                        Err(e) => {
                            let _ = tx.send(ChatEvent::StreamError(format!("Invalid model key: {e}")));
                            return;
                        }
                    },
                    Err(_) => {
                        let _ = tx.send(ChatEvent::StreamError("Invalid model key length".to_owned()));
                        return;
                    }
                };
                let model_client =
                    match ModelClient::from_installed_resolver(sk_inner.clone(), model_vk, None) {
                        Ok(c) => c,
                        Err(e) => {
                            let _ = tx.send(ChatEvent::StreamError(format!("Failed to create ModelClient: {e}")));
                            return;
                        }
                    };

                // Map ChatHistoryEntry → ChatMessage, handling all roles.
                // Skip the trailing empty assistant placeholder — it's only a
                // generation marker for the TUI; `add_generation_prompt: true`
                // already injects the assistant-start token into the prompt.
                // Including it causes a double `<|im_start|>assistant` prefix.
                let history_slice = match history.last() {
                    Some(e) if matches!(e.role, ChatRole::Assistant)
                        && e.content.is_empty()
                        && e.tool_calls.is_empty() => &history[..history.len() - 1],
                    _ => &history[..],
                };

                let messages: Vec<ChatMessage> = history_slice
                    .iter()
                    .map(|e| match e.role {
                        ChatRole::Tool => ChatMessage {
                            role: "tool".to_owned(),
                            content: e.content.clone(),
                            tool_calls: vec![],
                            tool_call_id: e.tool_call_id.clone().unwrap_or_default(),
                        },
                        ChatRole::Assistant if !e.tool_calls.is_empty() => ChatMessage {
                            role: "assistant".to_owned(),
                            content: e.content.clone(),
                            tool_calls: e.tool_calls
                                .iter()
                                .map(|tc| ToolCall {
                                    // Use the per-invocation correlation ID (not the
                                    // function-name UUID) so two calls to the same
                                    // tool have distinct tool_call_id values.
                                    id: tc.id.clone(),
                                    tool_type: "function".to_owned(),
                                    function: ToolCallFunction {
                                        name: tc.uuid.clone(),
                                        arguments: tc.arguments.clone(),
                                    },
                                })
                                .collect(),
                            tool_call_id: String::new(),
                        },
                        ChatRole::User | ChatRole::Assistant => ChatMessage {
                            role: match e.role {
                                ChatRole::User => "user",
                                _ => "assistant",
                            }
                            .to_owned(),
                            content: e.content.clone(),
                            tool_calls: vec![],
                            tool_call_id: String::new(),
                        },
                    })
                    .collect();

                let tools_json = tools_inner.as_ref()
                    .map(|t| serde_json::to_string(t).unwrap_or_default())
                    .unwrap_or_default();
                let template_result = tokio::time::timeout(
                    std::time::Duration::from_secs(15),
                    model_client
                        .infer(&mr_inner)
                        .apply_chat_template(&crate::services::generated::model_client::ChatTemplateRequest {
                            messages: messages.clone(),
                            add_generation_prompt: true,
                            tools_json: Some(tools_json.clone()).filter(|s| !s.is_empty()),
                            max_tokens: Some(gen_cfg.max_tokens as u32),
                            enable_thinking: None,
                            template_vars_json: None,
                        }),
                ).await;
                let prompt = match template_result {
                    Ok(Ok(p)) => crate::config::TemplatedPrompt::new(p),
                    Ok(Err(e)) => {
                        let _ = tx.send(ChatEvent::TemplateError(e.to_string()));
                        return;
                    }
                    Err(_elapsed) => {
                        let _ = tx.send(ChatEvent::TemplateError(
                            "chat template timed out after 15s — model service may be busy".to_owned(),
                        ));
                        return;
                    }
                };

                let req = GenerationRequest {
                    prompt: prompt.into_inner(),
                    max_tokens: Some(gen_cfg.max_tokens as u32),
                    temperature: Some(gen_cfg.temperature),
                    top_p: Some(gen_cfg.top_p),
                    top_k: gen_cfg.top_k.map(|v| v as u32),
                    ..Default::default()
                };

                use crate::services::generated::model_client::InferRpc;
                let mut handle = match InferRpc::generate_stream(&model_client.infer(&mr_inner), &req).await {
                    Ok(h) => h,
                    Err(e) => {
                        let _ = tx.send(ChatEvent::StreamError(e.to_string()));
                        return;
                    }
                };

                let mut cancel_rx = cancel_rx;
                loop {
                    tokio::select! {
                        biased;
                        _ = &mut cancel_rx => {
                            let _ = tx.send(ChatEvent::StreamCancelled);
                            break;
                        }
                        result = handle.recv_next() => {
                            match result {
                                Ok(Some(StreamPayload::Data(b))) => {
                                    let _ = tx.send(ChatEvent::Token(
                                        String::from_utf8_lossy(&b).into_owned(),
                                    ));
                                }
                                Ok(Some(StreamPayload::Complete(_))) | Ok(None) => {
                                    let _ = tx.send(ChatEvent::StreamComplete);
                                    break;
                                }
                                Ok(Some(StreamPayload::Error(m))) => {
                                    let _ = tx.send(ChatEvent::StreamError(m));
                                    break;
                                }
                                Ok(Some(StreamPayload::Tagged { .. })) => {
                                    // encrypted event payload, skip
                                    continue;
                                }
                                Err(e) => {
                                    let _ = tx.send(ChatEvent::StreamError(e.to_string()));
                                    break;
                                }
                            }
                        }
                    }
                }
            });
        });

        let handle: CancelHandle = Box::new(move || {
            let _ = cancel_tx.send(());
        });
        handle
    })
}

/// Build a `ToolCaller` backed by the MCP ZMQ service.
///
/// Eagerly fetches the tool list so that `tool_descriptions` and `openai_tools`
/// can be returned alongside the caller.  If the MCP service is unreachable,
/// returns empty collections and a caller that immediately sends an error result.
///
/// Returns `(caller, descriptions, openai_tools)` where:
/// - `descriptions` maps UUID → human label (for TUI display)
/// - `openai_tools` is the full OpenAI-format tool list for `apply_chat_template`
pub fn make_tool_caller(
    signing_key: &SigningKey,
) -> (hyprstream_tui::chat_app::ToolCaller, HashMap<String, String>, Vec<serde_json::Value>) {
    use crate::services::generated::mcp_client::McpClient as GenMcpClient;
    use hyprstream_tui::chat_app::ChatEvent;

    let sk = signing_key.clone();

    // ── Eagerly fetch tool list ───────────────────────────────────────────────
    // Spawned on a dedicated OS thread so `block_on` is safe regardless of
    // whether the caller is inside an existing Tokio runtime.
    let sk_fetch = sk.clone();
    let (descriptions, openai_tools): (HashMap<String, String>, Vec<serde_json::Value>) =
        std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .ok()?;
            rt.block_on(async move {
                // Resolve MCP service key via PolicyClient.
                let policy_vk = sk_fetch.verifying_key();
                let policy_client = crate::services::PolicyClient::from_installed_resolver(
                    sk_fetch.clone(),
                    policy_vk,
                    None,
                ).ok()?;
                let mcp_key_resp = policy_client.resolve_service_key(
                    &crate::services::generated::policy_client::ResolveServiceKey {
                        service_name: "mcp".to_owned(),
                    },
                ).await.ok()?;
                let mcp_vk = hyprstream_rpc::crypto::VerifyingKey::from_bytes(
                    mcp_key_resp.verifying_key.as_slice().try_into().ok()?
                ).ok()?;
                let gen: GenMcpClient =
                    GenMcpClient::from_installed_resolver(sk_fetch, mcp_vk, None).ok()?;
                let tool_list = gen.list_tools().await.ok()?;
                let mut descs = HashMap::new();
                let mut tools = Vec::new();
                for t in tool_list.tools {
                    descs.insert(t.name.clone(), t.description.clone());
                    // Build OpenAI-format tool definition for apply_chat_template.
                    // Omit `parameters` (full JSON Schema) to keep the tool list
                    // compact — the schema can add hundreds of tokens per tool and
                    // quickly exhaust the context window.  Name + description is
                    // sufficient for the model to know when and how to call a tool.
                    tools.push(serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                        }
                    }));
                }
                Some((descs, tools))
            })
        })
        .join()
        .unwrap_or(None)
        .unwrap_or_default();

    // ── Build the caller closure ──────────────────────────────────────────────
    let caller: hyprstream_tui::chat_app::ToolCaller =
        std::sync::Arc::new(move |id: String, uuid: String, arguments: String, event_tx: std::sync::mpsc::SyncSender<ChatEvent>| {
            let sk_c = sk.clone();
            let uuid_c = uuid.clone();
            std::thread::spawn(move || {
                let result_str: String = (move || {
                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(r) => r,
                        Err(e) => return format!("error: {e}"),
                    };
                    rt.block_on(async move {
                        // Resolve MCP service key via PolicyClient.
                        let policy_vk = sk_c.verifying_key();
                        let policy_client = match crate::services::PolicyClient::from_installed_resolver(
                            sk_c.clone(),
                            policy_vk,
                            None,
                        ) {
                            Ok(c) => c,
                            Err(e) => return format!("error: failed to create PolicyClient: {e}"),
                        };
                        let mcp_vk = match policy_client.resolve_service_key(
                            &crate::services::generated::policy_client::ResolveServiceKey {
                                service_name: "mcp".to_owned(),
                            },
                        ).await {
                            Ok(resp) => {
                                let bytes: [u8; 32] = match resp.verifying_key.as_slice().try_into() {
                                    Ok(b) => b,
                                    Err(_) => return "error: invalid MCP key length".to_owned(),
                                };
                                match hyprstream_rpc::crypto::VerifyingKey::from_bytes(&bytes) {
                                    Ok(vk) => vk,
                                    Err(e) => return format!("error: invalid MCP key: {e}"),
                                }
                            },
                            Err(e) => return format!("error: failed to resolve MCP key: {e}"),
                        };
                        let mcp_client = match GenMcpClient::from_installed_resolver(sk_c, mcp_vk, None) {
                            Ok(c) => c,
                            Err(e) => return format!("error: failed to create McpClient: {e}"),
                        };
                        match mcp_client
                            .call_tool(&crate::services::generated::mcp_client::CallTool {
                                tool_name: uuid_c,
                                arguments,
                                caller_identity: hyprstream_rpc::identity::Did::new(String::new()),
                            })
                            .await
                        {
                            Ok(res) => {
                                if res.success {
                                    res.result
                                } else {
                                    format!("error: {}", res.error_message)
                                }
                            }
                            Err(e) => format!("error: {e}"),
                        }
                    })
                })();
                let _ = event_tx.send(ChatEvent::ToolCallResult {
                    id,
                    uuid,
                    result: result_str,
                });
            });
        });

    (caller, descriptions, openai_tools)
}
