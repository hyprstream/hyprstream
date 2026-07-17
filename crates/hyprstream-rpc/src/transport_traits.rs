//! Transport and Signer traits for unified RPC communication.
//!
//! These traits abstract over the wire transport (ZMQ vs WebTransport) and
//! signing mechanism (local key vs external callback), enabling a single
//! `RpcClient<S, T>` implementation that compiles to both native and wasm32.

use anyhow::Result;
use async_trait::async_trait;
use futures::Stream;

/// A transport failure proven to have occurred before any request bytes were
/// dispatched. Only this marker is safe for automatic endpoint retry.
#[derive(Debug)]
pub struct PreDispatchTransportError(anyhow::Error);

impl PreDispatchTransportError {
    pub fn new(error: anyhow::Error) -> Self {
        Self(error)
    }
}

impl std::fmt::Display for PreDispatchTransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "pre-dispatch transport failure: {}", self.0)
    }
}

impl std::error::Error for PreDispatchTransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref())
    }
}

pub fn is_pre_dispatch_transport_error(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|cause| cause.downcast_ref::<PreDispatchTransportError>().is_some())
}

/// Signing abstraction for Ed25519 envelope signatures.
///
/// Native: `LocalSigner` owns the `SigningKey` and signs synchronously.
/// WASM: `JsSigner` delegates to a JS callback (aegis-vault) and awaits the result.
///
/// # Safety (wasm32)
///
/// `JsSigner` holds `js_sys::Function` which is `!Send`. We use
/// `unsafe impl Send + Sync` on it because wasm32 is single-threaded.
/// The `Send + Sync` bounds on this trait are required so that `RpcClient<S, T>`
/// can be used with `Arc` and `tokio::spawn` on native.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait Signer: Send + Sync {
    /// Ed25519 public key (32 bytes).
    fn pubkey(&self) -> [u8; 32];

    /// Sign canonical envelope bytes. Returns 64-byte Ed25519 signature.
    async fn sign(&self, canonical_bytes: &[u8]) -> Result<[u8; 64]>;

    /// ML-DSA-65 verifying key bytes (1952 bytes) when PQ hybrid is available.
    fn pq_pubkey(&self) -> Option<Vec<u8>> {
        None
    }

    /// ML-DSA-65 signature over canonical envelope bytes.
    async fn pq_sign(&self, _canonical_bytes: &[u8]) -> Result<Option<Vec<u8>>> {
        Ok(None)
    }
}

/// Wire transport abstraction.
///
/// Native: `ZmqConnection` wraps ZMQ REQ sockets with auto-reconnect.
/// WASM: `WtConnection` wraps WebTransport bidi streams with ZMTP framing.
///
/// Both transports carry ZMTP-framed messages. The `Sub` and `Pub` associated
/// types provide streaming (server→client) and control (client→server) channels.
///
/// `Sub` must implement `futures::Stream` yielding frames as `Vec<Vec<u8>>`,
/// matching the ZMTP multipart format `[topic, capnp_data, mac]`.
///
/// Uses cfg-gated async_trait: Send on native, ?Send on wasm32.
/// The trait types themselves are `Send + Sync` (via unsafe impl on wasm32),
/// so `Arc<RpcClient<S, T>>` can be shared across threads on native.
/// Individual call futures are awaited in place, not spawned.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait Transport: Send + Sync {
    /// Subscriber type for data streams (server→client).
    /// Must yield ZMTP multipart frames as `Vec<Vec<u8>>`.
    type Sub: Stream<Item = Result<Vec<Vec<u8>>>> + Unpin + Send;

    /// Publisher type for control channel (client→server).
    type Pub: PublishSink + Send;

    /// Send a request and receive a response (REQ/REP pattern).
    ///
    /// `timeout_ms`: Optional timeout in milliseconds. Implementations should
    /// use a default (e.g., 30s) when `None`. On native, this caps the ZMQ
    /// recv timeout. On WASM, WebTransport stream lifetime handles this.
    async fn send(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>>;

    /// Whether this carrier forbids cleartext request envelopes.
    ///
    /// Networked/untrusted carriers require the RPC client to populate
    /// `SignedEnvelope.encrypted_envelope` with a HyKEM/COSE_Encrypt0 payload.
    ///
    /// **Fail-closed default (`true`).** A carrier is treated as untrusted for
    /// envelope confidentiality unless it explicitly opts out — so a new or
    /// out-of-tree `Transport` cannot silently inherit cleartext permission
    /// (epic #550 principle 1: no silent downgrade). Only same-process and
    /// same-host IPC transports (`InMemoryTransport`, `LazyUdsTransport`)
    /// override this to `false`. Every QUIC carrier
    /// (`QuinnTransport`, `LazyQuinnTransport`) and iroh carrier forbids
    /// cleartext unconditionally — a loopback address earns NO exemption, since
    /// it is not evidence the bytes stay inside the trust boundary (it can
    /// terminate at a proxy/tunnel). This mirrors
    /// [`EndpointType::forbids_cleartext_envelope`](crate::transport::EndpointType::forbids_cleartext_envelope),
    /// the parallel config-time classification; the two are intentionally
    /// layered (pre-dial classification vs runtime enforcement) rather than
    /// delegated, so a runtime transport need not reconstruct an `EndpointType`.
    fn forbids_cleartext_envelope(&self) -> bool {
        true
    }

    /// Subscribe to a topic (SUB pattern). Returns a stream of multipart frames.
    async fn subscribe(&self, topic: &[u8]) -> Result<Self::Sub>;

    /// Open a publish channel to a topic (PUB/PUSH pattern for ctrl messages).
    async fn publish(&self, topic: &[u8]) -> Result<Self::Pub>;
}

/// Trait for sending frames on a publish/control channel.
///
/// Abstracted because native uses sync `zmq::Socket::send` with `DONTWAIT`,
/// while WASM uses async WebTransport writes.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait PublishSink: Send {
    /// Send ZMTP multipart frames (e.g., `[ctrl_topic, capnp, mac]`).
    async fn send_frames(&self, frames: &[&[u8]]) -> Result<()>;
}
