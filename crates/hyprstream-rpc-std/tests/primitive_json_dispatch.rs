//! Exercise the shipped JSON dispatcher through its public client/transport seam.
#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::expect_used)]

use async_trait::async_trait;
use hyprstream_rpc::stream_consumer::StreamHandle;
use hyprstream_rpc::{CallOptions, RpcClient};
use hyprstream_rpc_std::{tui_capnp, tui_client::TuiClient};
use parking_lot::Mutex;
use serde_json::json;
use std::sync::Arc;

#[derive(Default)]
struct RecordingClient {
    calls: Mutex<Vec<(String, u32)>>,
}

#[async_trait]
impl RpcClient for RecordingClient {
    async fn call(&self, _payload: Vec<u8>) -> anyhow::Result<Vec<u8>> {
        panic!("unexpected transport entry point")
    }
    async fn call_for_service(
        &self,
        _service_domain: &str,
        _payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        panic!("unexpected transport entry point")
    }
    async fn call_for_service_with_method(
        &self,
        service: &str,
        method: u16,
        payload: Vec<u8>,
    ) -> anyhow::Result<Vec<u8>> {
        let message = capnp::serialize::read_message(
            &mut payload.as_slice(),
            capnp::message::ReaderOptions::new(),
        )?;
        let request = message.get_root::<tui_capnp::tui_request::Reader<'_>>()?;
        use tui_capnp::tui_request::Which;
        let (name, value, expected_method) = match request.which()? {
            Which::Disconnect(value) => ("disconnect", value, 1),
            Which::CloseWindow(value) => ("close_window", value, 3),
            Which::ListWindows(value) => ("list_windows", value, 4),
            Which::FocusWindow(value) => ("focus_window", value, 5),
            Which::ClosePane(value) => ("close_pane", value, 7),
            Which::FocusPane(value) => ("focus_pane", value, 8),
            Which::Snapshot(value) => ("snapshot", value, 9),
            Which::PollStdin(value) => ("poll_stdin", value, 13),
            _ => anyhow::bail!("unexpected TUI request"),
        };
        assert_eq!(service, "tui");
        assert_eq!(method, expected_method, "method binding for {name}");
        self.calls.lock().push((name.to_owned(), value));
        hyprstream_rpc::serialize_message(|message| {
            let mut response = message.init_root::<tui_capnp::tui_response::Builder<'_>>();
            response.set_request_id(request.get_id());
            match name {
                "disconnect" => response.set_disconnect_result(()),
                "close_window" => response.set_close_window_result(()),
                "list_windows" => {
                    response.init_list_windows_result();
                }
                "focus_window" => response.set_focus_window_result(()),
                "close_pane" => response.set_close_pane_result(()),
                "focus_pane" => response.set_focus_pane_result(()),
                "snapshot" => {
                    response.init_snapshot_result();
                }
                "poll_stdin" => response.set_poll_stdin_result(b"input"),
                _ => unreachable!(),
            }
        })
    }
    async fn call_with_options(
        &self,
        _payload: Vec<u8>,
        _options: CallOptions,
    ) -> anyhow::Result<Vec<u8>> {
        panic!("unexpected transport entry point")
    }
    async fn call_with_options_for_service(
        &self,
        _service_domain: &str,
        _payload: Vec<u8>,
        _options: CallOptions,
    ) -> anyhow::Result<Vec<u8>> {
        panic!("unexpected transport entry point")
    }
    async fn call_streaming(
        &self,
        _payload: Vec<u8>,
        _ephemeral_pubkey: [u8; 32],
    ) -> anyhow::Result<Vec<u8>> {
        panic!("unexpected transport entry point")
    }
    async fn call_streaming_for_service(
        &self,
        _service_domain: &str,
        _payload: Vec<u8>,
        _ephemeral_pubkey: [u8; 32],
    ) -> anyhow::Result<Vec<u8>> {
        panic!("unexpected transport entry point")
    }
    async fn call_streaming_for_service_with_method(
        &self,
        _service_domain: &str,
        _method_discriminator: u16,
        _payload: Vec<u8>,
        _ephemeral_pubkey: [u8; 32],
    ) -> anyhow::Result<Vec<u8>> {
        panic!("unexpected transport entry point")
    }
    async fn open_stream(&self, _payload: Vec<u8>) -> anyhow::Result<Box<dyn StreamHandle>> {
        panic!("unexpected transport entry point")
    }
    async fn open_stream_from_info(
        &self,
        _stream_info: hyprstream_rpc::stream_info::StreamInfo,
        _client_secret: [u8; 32],
        _client_pubkey: [u8; 32],
    ) -> anyhow::Result<Box<dyn StreamHandle>> {
        panic!("unexpected transport entry point")
    }
    fn next_id(&self) -> u64 {
        0
    }
}

const METHODS: &[&str] = &[
    "disconnect",
    "close_window",
    "list_windows",
    "focus_window",
    "close_pane",
    "focus_pane",
    "snapshot",
    "poll_stdin",
];

#[tokio::test]
async fn primitive_json_methods_preserve_values_and_method_binding() {
    let transport = Arc::new(RecordingClient::default());
    let client = TuiClient::new(transport.clone());
    for &method in METHODS {
        for value in [0, 7, u32::MAX] {
            for args in [json!({"value": value}), json!({method: value, "value": 19})] {
                transport.calls.lock().clear();
                let response = client
                    .call_method(method, &args)
                    .await
                    .expect("valid primitive JSON call must complete");
                assert_eq!(*transport.calls.lock(), vec![(method.to_owned(), value)]);
                if method == "poll_stdin" {
                    assert_eq!(response, json!([105, 110, 112, 117, 116]));
                }
            }
        }
    }
}

#[tokio::test]
async fn invalid_primitive_arguments_never_reach_transport() {
    let transport = Arc::new(RecordingClient::default());
    let client = TuiClient::new(transport.clone());
    for &method in METHODS {
        let invalid = [
            json!({}),
            json!(null),
            json!([]),
            json!(7),
            json!({"value": null}),
            json!({"value": true}),
            json!({"value": "7"}),
            json!({"value": 1.5}),
            json!({"value": -1}),
            json!({"value": 4_294_967_296_u64}),
            json!({method: null, "value": 7}),
            json!({method: "invalid", "value": 7}),
        ];
        for args in invalid {
            assert!(
                client.call_method(method, &args).await.is_err(),
                "{method}: {args}"
            );
            assert!(
                transport.calls.lock().is_empty(),
                "invalid argument sent: {method}: {args}"
            );
        }
    }
}
