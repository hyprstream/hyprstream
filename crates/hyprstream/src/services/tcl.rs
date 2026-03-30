//! TclService — ZMQ RPC service wrapping a `TclExecutor`.
//!
//! This service accepts Tcl eval requests over ZMQ (REQ/REP pattern).
//! The payload is treated as a raw UTF-8 script; the response is the
//! evaluation result (or error prefixed with "ERROR: ").
//!
//! # Protocol (minimal, pre-Cap'n Proto)
//!
//! **Request**: raw UTF-8 bytes — the Tcl script to evaluate.
//! **Response**: raw UTF-8 bytes — the evaluation result, or `"ERROR: <msg>"` on failure.
//!
//! A proper `tcl.capnp` schema with `generate_rpc_service!` integration
//! is planned for a future step. This minimal service proves the pattern:
//! factory registration, TclExecutor ownership, and async dispatch.
//!
//! # Thread model
//!
//! TclExecutor isolates the `!Send` molt interpreter on a dedicated OS thread.
//! The service itself is `Send + Sync` and runs on the tokio async runtime.
//! `handle_request` uses `TclExecutor::eval()` which internally calls
//! `spawn_blocking` for the response channel recv.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use hyprstream_rpc::prelude::*;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_tcl::TclExecutor;
use hyprstream_vfs::{Namespace, Subject};
use tracing::{debug, warn};

use super::{Continuation, EnvelopeContext};

/// Default inproc endpoint for the Tcl service.
pub const TCL_ENDPOINT: &str = "inproc://hyprstream/tcl";

/// ZMQ RPC service that owns a `TclExecutor`.
///
/// Registered via `#[service_factory("tcl")]` in `factories.rs`.
/// Accepts raw UTF-8 eval requests and returns results.
pub struct TclService {
    /// The Tcl interpreter handle (Send + Sync).
    executor: TclExecutor,
    /// ZMQ context for socket creation.
    context: Arc<zmq::Context>,
    /// Transport configuration (endpoint binding).
    transport: TransportConfig,
    /// Ed25519 signing key for signing responses.
    signing_key: SigningKey,
}

impl TclService {
    /// Create a new TclService.
    ///
    /// Spawns a `TclExecutor` on a dedicated thread with the given VFS namespace
    /// and default subject. The executor lives as long as this service.
    pub fn new(
        ns: Arc<Namespace>,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
        signing_key: SigningKey,
    ) -> Self {
        let rt = tokio::runtime::Handle::current();
        let executor = TclExecutor::spawn(ns, Subject::new("system"), rt);
        Self {
            executor,
            context,
            transport,
            signing_key,
        }
    }

    /// Create a TclService with a pre-built executor (for testing).
    #[cfg(test)]
    pub fn with_executor(
        executor: TclExecutor,
        context: Arc<zmq::Context>,
        transport: TransportConfig,
        signing_key: SigningKey,
    ) -> Self {
        Self {
            executor,
            context,
            transport,
            signing_key,
        }
    }
}

#[async_trait(?Send)]
impl super::ZmqService for TclService {
    async fn handle_request(
        &self,
        ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        let script = std::str::from_utf8(payload)
            .map_err(|e| anyhow::anyhow!("invalid UTF-8 in eval request: {e}"))?;

        debug!(
            service = "tcl",
            user = ctx.user(),
            script_len = script.len(),
            "eval request"
        );

        match self.executor.eval(script).await {
            Ok(result) => Ok((result.into_bytes(), None)),
            Err(err) => {
                warn!(service = "tcl", user = ctx.user(), error = %err, "eval failed");
                // Return the error as a response (not a transport-level error)
                // so the client gets a clean message.
                Ok((format!("ERROR: {err}").into_bytes(), None))
            }
        }
    }

    fn name(&self) -> &str {
        "tcl"
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use hyprstream_rpc::crypto::generate_signing_keypair;

    fn make_ctx() -> EnvelopeContext {
        EnvelopeContext::from_callback_system(1)
    }

    fn make_service() -> TclService {
        let (signing_key, _) = generate_signing_keypair();
        let ns = Arc::new(Namespace::new());
        let context = Arc::new(zmq::Context::new());
        let transport = TransportConfig::inproc("test-tcl-service");
        TclService::new(ns, context, transport, signing_key)
    }

    #[tokio::test]
    async fn eval_simple_expr() {
        let svc = make_service();
        let ctx = make_ctx();
        let (resp, cont) = svc.handle_request(&ctx, b"expr {2 + 3}").await.unwrap();
        assert_eq!(String::from_utf8_lossy(&resp), "5");
        assert!(cont.is_none());
    }

    #[tokio::test]
    async fn eval_error_returns_error_prefix() {
        let svc = make_service();
        let ctx = make_ctx();
        let (resp, _) = svc
            .handle_request(&ctx, b"nonexistent_command")
            .await
            .unwrap();
        let s = String::from_utf8_lossy(&resp);
        assert!(s.starts_with("ERROR: "), "expected ERROR prefix: {s}");
    }

    #[tokio::test]
    async fn invalid_utf8_rejected() {
        let svc = make_service();
        let ctx = make_ctx();
        let result = svc.handle_request(&ctx, &[0xFF, 0xFE]).await;
        assert!(result.is_err());
    }
}
