//! ZMQ-based StreamSpawner for ChatApp inference.
//!
//! Deduplicated from `shell_handlers.rs` and `service.rs` — both callers now
//! use `make_chat_spawner` from this module.

use ed25519_dalek::SigningKey;

/// Build a `StreamSpawner` that drives inference via `ModelZmqClient`.
///
/// Spawns a dedicated OS thread with a single-threaded Tokio runtime per
/// user message.  Applies the chat template, starts an authenticated inference
/// stream, and sends `ChatEvent` variants back through the provided channel.
pub fn make_chat_spawner(
    signing_key: &SigningKey,
    model_ref: &str,
) -> hyprstream_tui::chat_app::StreamSpawner {
    use hyprstream_rpc::crypto::generate_ephemeral_keypair;
    use hyprstream_rpc::envelope::RequestIdentity;
    use hyprstream_rpc::streaming::StreamPayload;

    use crate::api::openai_compat::ChatMessage;
    use crate::config::GenerationRequest;
    use crate::services::model::ModelZmqClient;
    use crate::services::rpc_types::StreamHandle;
    use crate::zmq::global_context;
    use hyprstream_tui::chat_app::ChatEvent;

    let sk = signing_key.clone();
    let mr = model_ref.to_owned();

    Box::new(move |pairs, event_tx| {
        use hyprstream_tui::chat_app::CancelHandle;
        use tokio::sync::oneshot;

        let sk_inner = sk.clone();
        let mr_inner = mr.clone();
        let tx = event_tx.clone();

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
                let model_client =
                    ModelZmqClient::new(sk_inner.clone(), RequestIdentity::local());

                let messages: Vec<ChatMessage> = pairs
                    .into_iter()
                    .map(|(role, content)| ChatMessage {
                        role,
                        content: Some(content),
                        function_call: None,
                        tool_calls: None,
                        tool_call_id: None,
                    })
                    .collect();

                let prompt = match model_client
                    .apply_chat_template(&mr_inner, &messages, true, None)
                    .await
                {
                    Ok(p) => p,
                    Err(e) => {
                        let _ = tx.send(ChatEvent::TemplateError(e.to_string()));
                        return;
                    }
                };

                let req = GenerationRequest {
                    prompt,
                    max_tokens: 2048,
                    temperature: 0.7,
                    ..Default::default()
                };

                let (client_secret, client_pubkey) = generate_ephemeral_keypair();
                let client_pubkey_bytes: [u8; 32] = client_pubkey.to_bytes();

                let stream_info = match model_client
                    .infer_stream(&mr_inner, &req, client_pubkey_bytes)
                    .await
                {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = tx.send(ChatEvent::StreamError(e.to_string()));
                        return;
                    }
                };

                let mut handle = match StreamHandle::new(
                    &global_context(),
                    stream_info.stream_id,
                    &stream_info.endpoint,
                    &stream_info.server_pubkey,
                    &client_secret,
                    &client_pubkey_bytes,
                ) {
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
