//! Transport and Signer traits for unified RPC communication.
//!
//! These traits abstract over the wire transport (ZMQ vs WebTransport) and
//! signing mechanism (local key vs external callback), enabling a single
//! `RpcClient<S, T>` implementation that compiles to both native and wasm32.

use anyhow::Result;
use async_trait::async_trait;
use futures::Stream;

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
#[allow(dead_code)]
#[async_trait]
pub trait Signer: Send + Sync {
    /// Ed25519 public key (32 bytes).
    fn pubkey(&self) -> [u8; 32];

    /// Sign canonical envelope bytes. Returns 64-byte Ed25519 signature.
    async fn sign(&self, canonical_bytes: &[u8]) -> Result<[u8; 64]>;
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
#[allow(dead_code)]
#[async_trait]
pub trait Transport: Send + Sync {
    /// Subscriber type for data streams (server→client).
    /// Must yield ZMTP multipart frames as `Vec<Vec<u8>>`.
    type Sub: Stream<Item = Result<Vec<Vec<u8>>>> + Unpin + Send;

    /// Publisher type for control channel (client→server).
    type Pub: Send;

    /// Send a request and receive a response (REQ/REP pattern).
    ///
    /// `timeout_ms`: Optional timeout in milliseconds. Implementations should
    /// use a default (e.g., 30s) when `None`. On native, this caps the ZMQ
    /// recv timeout. On WASM, WebTransport stream lifetime handles this.
    async fn send(&self, payload: Vec<u8>, timeout_ms: Option<i32>) -> Result<Vec<u8>>;

    /// Subscribe to a topic (SUB pattern). Returns a stream of multipart frames.
    async fn subscribe(&self, topic: &[u8]) -> Result<Self::Sub>;

    /// Open a publish channel to a topic (PUB/PUSH pattern for ctrl messages).
    async fn publish(&self, topic: &[u8]) -> Result<Self::Pub>;
}

/// Trait for sending frames on a publish/control channel.
///
/// Abstracted because native uses sync `zmq::Socket::send` with `DONTWAIT`,
/// while WASM uses async WebTransport writes.
#[allow(dead_code)]
#[async_trait]
pub trait PublishSink: Send {
    /// Send ZMTP multipart frames (e.g., `[ctrl_topic, capnp, mac]`).
    async fn send_frames(&self, frames: &[&[u8]]) -> Result<()>;
}
