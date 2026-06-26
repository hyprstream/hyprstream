//! End-to-end test for the default (inproc) flipped serve path — the #136 cut's
//! most-used path, previously untested (the UDS canary covers ipc only).
//!
//! Wires a `RequestService` through the exact bridged inproc path the daemon uses:
//! `LocalServiceBridge` → `register_inproc` (in-memory dial registry) → `dial()`
//! (InMemoryTransport) → `RpcClientImpl`. A real SignedEnvelope round-trips, and —
//! crucially — the service returns a streaming **continuation** that must be
//! spawned by the bridge (the #186 StreamInfo pump). This guards against the
//! regression where the bridge dropped/rejected continuations, silently breaking
//! streaming services (model / tui).

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::{SigningKey, VerifyingKey};
use rand::RngCore;
use tokio::sync::Mutex;

use hyprstream_rpc::dial::{dial, register_inproc};
use hyprstream_rpc::envelope::InMemoryNonceCache;
use hyprstream_rpc::service::{Continuation, EnvelopeContext, RequestService};
use hyprstream_rpc::signer::LocalSigner;
use hyprstream_rpc::transport::iroh_rpc::LocalServiceBridge;
use hyprstream_rpc::transport::rpc_session::IrohRequestProcessor;
use hyprstream_rpc::transport::TransportConfig;

/// Echoes the payload AND returns a continuation that fires a oneshot — so the
/// test can assert the bridge actually spawned the streaming pump.
struct StreamingEcho {
    name: String,
    transport: TransportConfig,
    signing_key: SigningKey,
    fired: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

#[async_trait(?Send)]
impl RequestService for StreamingEcho {
    async fn handle_request(
        &self,
        _ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        let tx = self.fired.lock().await.take();
        let cont: Continuation = Box::pin(async move {
            if let Some(tx) = tx {
                let _ = tx.send(());
            }
        });
        Ok((payload.to_vec(), Some(cont)))
    }

    fn name(&self) -> &str {
        &self.name
    }
    fn transport(&self) -> &TransportConfig {
        &self.transport
    }
    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }
}

fn fresh_key() -> SigningKey {
    let mut b = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut b);
    SigningKey::from_bytes(&b)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inproc_bridged_round_trip_and_continuation_spawned() -> Result<()> {
    // Integration test → hyprstream-rpc in non-test mode → verify policy
    // fail-closes to Hybrid (#160); opt into the Classical policy this exercises.
    let _ = hyprstream_rpc::envelope::install_verify_config(
        hyprstream_rpc::envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
            pq_store: None,
        },
    );

    const NAME: &str = "test/inproc-stream-echo";

    let server_key = fresh_key();
    let server_vk: VerifyingKey = server_key.verifying_key();
    let (fired_tx, fired_rx) = tokio::sync::oneshot::channel();

    let svc = StreamingEcho {
        name: NAME.to_owned(),
        transport: TransportConfig::inproc(NAME),
        signing_key: server_key,
        fired: Mutex::new(Some(fired_tx)),
    };

    // Server side: bridge + register in the in-memory dial registry. Keep the
    // strong processor Arc alive for the duration (registry holds only a Weak).
    let nonce = Arc::new(InMemoryNonceCache::new());
    let bridge = LocalServiceBridge::spawn(svc, nonce, 0)?;
    let processor: Arc<dyn IrohRequestProcessor> = Arc::new(bridge);
    register_inproc(NAME, &processor);

    // Client side: dial inproc → InMemoryTransport → real SignedEnvelope.
    let client = dial(
        &TransportConfig::inproc(NAME),
        LocalSigner::new(fresh_key()),
        Some(server_vk),
        None,
    )?;
    let resp = client.call(b"hello-inproc".to_vec()).await?;
    assert_eq!(&resp[..], b"hello-inproc");

    // The streaming continuation must have been spawned by the bridge and run.
    tokio::time::timeout(std::time::Duration::from_secs(3), fired_rx)
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "continuation did not run — regression: the bridge dropped/rejected \
                 the streaming continuation, which silently breaks streaming services"
            )
        })??;

    drop(processor);
    Ok(())
}
