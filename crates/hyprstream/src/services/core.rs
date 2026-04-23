//! Core service infrastructure — re-exports from `hyprstream-rpc`.

pub use hyprstream_rpc::service::{Continuation, EnvelopeContext, ZmqService};

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]
mod tests {
    use super::*;
    use crate::zmq::global_context;
    use anyhow::Result;
    use std::sync::Arc;
    use hyprstream_rpc::crypto::generate_signing_keypair;
    use hyprstream_rpc::prelude::*;
    use hyprstream_rpc::service::RequestLoop;
    use hyprstream_rpc::transport::TransportConfig;
    use hyprstream_rpc::rpc_client::RpcClientImpl;
    use hyprstream_rpc::signer::LocalSigner;
    use hyprstream_rpc::zmq_connection::ZmqConnection;

    /// Test service with infrastructure (new pattern)
    struct EchoService {
        context: Arc<zmq::Context>,
        transport: TransportConfig,
        signing_key: SigningKey,
    }

    impl EchoService {
        fn new(context: Arc<zmq::Context>, transport: TransportConfig, signing_key: SigningKey) -> Self {
            Self { context, transport, signing_key }
        }
    }

    #[async_trait::async_trait(?Send)]
    impl ZmqService for EchoService {
        async fn handle_request(&self, ctx: &EnvelopeContext, payload: &[u8]) -> Result<(Vec<u8>, Option<crate::services::Continuation>)> {
            let user = ctx.user();
            let mut response = format!("from {}:", user).into_bytes();
            response.extend_from_slice(payload);
            Ok((response, None))
        }

        fn name(&self) -> &str {
            "echo"
        }

        fn context(&self) -> &Arc<zmq::Context> {
            &self.context
        }

        fn transport(&self) -> &TransportConfig {
            &self.transport
        }

        fn signing_key(&self) -> SigningKey {
            self.signing_key.clone()
        }
    }

    fn make_client(endpoint: &str, signing_key: SigningKey, server_verifying_key: VerifyingKey) -> RpcClientImpl<LocalSigner, ZmqConnection> {
        let signer = LocalSigner::new(signing_key, RequestIdentity::anonymous());
        let transport = ZmqConnection::new(endpoint, global_context());
        RpcClientImpl::new(signer, transport, Some(server_verifying_key))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_request_loop() {
        let local = tokio::task::LocalSet::new();
        local.run_until(async {
            let transport = TransportConfig::inproc("test-echo-service-core");
            let endpoint = transport.zmq_endpoint();

            let (signing_key, verifying_key) = generate_signing_keypair();
            let service = EchoService::new(global_context(), transport.clone(), signing_key.clone());

            let runner = RequestLoop::new(transport, global_context(), signing_key.clone());
            let mut handle = runner.run(service).await.expect("test: start service");

            let client = make_client(&endpoint, signing_key, verifying_key);
            let response = client.call(b"hello".to_vec()).await.expect("test: call");

            let response_str = String::from_utf8_lossy(&response);
            assert!(response_str.contains("hello"), "Response should contain 'hello': {}", response_str);

            handle.stop().await;
        }).await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_invalid_request_signature_rejected() {
        let local = tokio::task::LocalSet::new();
        local.run_until(async {
            let transport = TransportConfig::inproc("test-invalid-req-sig-core");
            let endpoint = transport.zmq_endpoint();

            let (server_signing_key, server_verifying_key) = generate_signing_keypair();
            let (client_signing_key, _client_verifying_key) = generate_signing_keypair();

            let service = EchoService::new(global_context(), transport.clone(), server_signing_key.clone());

            let runner = RequestLoop::new(transport, global_context(), server_signing_key);
            let mut handle = runner.run(service).await.expect("test: start service");

            // Sign request with different key than service expects
            let client = make_client(&endpoint, client_signing_key, server_verifying_key);
            let result = client.call(b"should fail".to_vec()).await;

            match result {
                Ok(response) => {
                    assert!(response.is_empty(), "Invalid request signature should return empty response");
                }
                Err(_) => {
                    // Error is also acceptable
                }
            }

            handle.stop().await;
        }).await;
    }
}
