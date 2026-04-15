//! Unified RPC client parameterized over Signer and Transport.
//!
//! `RpcClientImpl<S, T>` handles envelope construction, signing, response verification,
//! and streaming setup. Same code compiles to native and wasm32 — only the type
//! parameters differ:
//!
//! - Native: `RpcClient<LocalSigner, ZmqConnection>`
//! - WASM:   `RpcClient<JsSigner, WtConnection>`
//!
//! Generated service clients (e.g., `RegistryClient`) use `Arc<dyn RpcClient>`
//! for object-safe dynamic dispatch.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use async_trait::async_trait;

use crate::crypto::VerifyingKey;
use crate::envelope::{
    self, RequestEnvelope, SignedEnvelope,
};
use crate::transport_traits::{Signer, Transport};
use crate::capnp::{FromCapnp, ToCapnp};
use crate::stream_consumer::{StreamHandle, StreamHandleImpl};

/// Unified RPC client. Evolved from ZmqClient — same envelope, timeout,
/// and resilience logic, parameterized over transport and signing.
pub struct RpcClientImpl<S: Signer, T: Transport + 'static> {
    pub signer: S,
    pub transport: T,
    pub server_verifying_key: VerifyingKey,
    jwt_token: parking_lot::RwLock<Option<String>>,
    request_id: AtomicU64,
}

impl<S: Signer, T: Transport + 'static> RpcClientImpl<S, T> {
    /// Create a new RPC client.
    pub fn new(signer: S, transport: T, server_verifying_key: VerifyingKey) -> Self {
        Self {
            signer,
            transport,
            server_verifying_key,
            jwt_token: parking_lot::RwLock::new(None),
            request_id: AtomicU64::new(1),
        }
    }

    /// Send a request and return the verified, unwrapped response payload.
    pub async fn call(&self, payload: Vec<u8>) -> Result<Vec<u8>> {
        let request_id = self.next_id();
        let signed_bytes = self.sign_envelope(request_id, payload, None).await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = envelope::unwrap_response(
            &response_bytes,
            Some(&self.server_verifying_key),
        )?;
        Ok(inner)
    }

    /// Send a streaming request with ephemeral DH pubkey.
    /// Returns the verified, unwrapped response payload (typically StreamInfo).
    pub async fn call_streaming(
        &self,
        payload: Vec<u8>,
        ephemeral_pubkey: [u8; 32],
    ) -> Result<Vec<u8>> {
        let request_id = self.next_id();
        let signed_bytes = self
            .sign_envelope(request_id, payload, Some(ephemeral_pubkey))
            .await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = envelope::unwrap_response(
            &response_bytes,
            Some(&self.server_verifying_key),
        )?;
        Ok(inner)
    }

    /// Set the opaque JWT token. Server decodes and verifies.
    pub fn set_jwt(&self, token: Option<String>) {
        *self.jwt_token.write() = token;
    }

    /// Get the next monotonically increasing request ID.
    pub fn next_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Build, sign, and serialize a request envelope.
    async fn sign_envelope(
        &self,
        request_id: u64,
        payload: Vec<u8>,
        ephemeral_pubkey: Option<[u8; 32]>,
    ) -> Result<Vec<u8>> {
        let jwt_token = self.jwt_token.read().clone();

        let mut envelope = RequestEnvelope::new(self.signer.identity(), payload);
        envelope.request_id = request_id;
        if let Some(token) = jwt_token {
            envelope = envelope.with_jwt_token(token);
        }
        if let Some(pubkey) = ephemeral_pubkey {
            envelope = envelope.with_ephemeral_pubkey(pubkey);
        }

        let canonical = envelope.to_bytes();
        let signature = self.signer.sign(&canonical).await?;
        let signed = SignedEnvelope {
            envelope,
            signature,
            signer_pubkey: self.signer.pubkey(),
        };

        let mut message = capnp::message::Builder::new_default();
        {
            let mut builder =
                message.init_root::<crate::common_capnp::signed_envelope::Builder>();
            signed.write_to(&mut builder);
        }
        let mut bytes = Vec::new();
        capnp::serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    /// Calculate timeout, defaulting to 30s.
    fn calculate_timeout(&self) -> Option<i32> {
        // JWT is opaque — no client-side expiry capping.
        // Server enforces JWT expiry on its side.
        Some(30_000)
    }

    /// Open a verified streaming subscription.
    ///
    /// Generates an ephemeral keypair, sends the streaming request,
    /// performs ECDH + key derivation, subscribes to the data topic,
    /// opens the ctrl channel, and creates a verified StreamHandle.
    ///
    /// All crypto happens in Rust — the caller just gets a handle
    /// that yields verified payloads.
    #[cfg(not(feature = "fips"))]
    pub async fn open_stream(&self, payload: Vec<u8>) -> Result<StreamHandleImpl<T>> {
        use crate::crypto::generate_ephemeral_keypair;

        let (secret, public) = generate_ephemeral_keypair();
        let pub_bytes = public.to_bytes();

        // Send streaming request via signed envelope
        let response = self.call_streaming(payload, pub_bytes).await?;

        // Parse StreamInfo from response
        let stream_info = parse_stream_info(&response)?;

        // Full streaming setup via Transport
        let secret_bytes = secret.scalar().to_bytes();
        StreamHandleImpl::open(&self.transport, stream_info, &secret_bytes, &pub_bytes).await
    }
    /// Open a verified stream from pre-parsed StreamInfo + ephemeral keys.
    ///
    /// Use when the streaming RPC was already sent (e.g., via dispatch) and
    /// you have the StreamInfo + ephemeral keypair. Skips the send step.
    #[cfg(not(feature = "fips"))]
    pub async fn open_stream_from_info(
        &self,
        stream_info: crate::stream_info::StreamInfo,
        client_secret: &[u8; 32],
        client_pubkey: &[u8; 32],
    ) -> Result<StreamHandleImpl<T>> {
        StreamHandleImpl::open(&self.transport, stream_info, client_secret, client_pubkey).await
    }
}

/// Parse StreamInfo from Cap'n Proto response bytes.
fn parse_stream_info(bytes: &[u8]) -> Result<crate::stream_info::StreamInfo> {
    let reader = capnp::serialize::read_message(
        &mut std::io::Cursor::new(bytes),
        capnp::message::ReaderOptions::default(),
    )?;
    // StreamInfo may be nested inside a service response variant.
    // Try direct parse first (for portable client path).
    let si_reader = reader.get_root::<crate::streaming_capnp::stream_info::Reader>()?;
    crate::stream_info::StreamInfo::read_from(si_reader)
}

// ============================================================================
// Object-safe trait for generated portable clients
// ============================================================================

/// Object-safe RPC client trait for dynamic dispatch.
///
/// Generated portable clients use `Arc<dyn RpcClient>` so they work
/// with any `RpcClientImpl<S, T>` regardless of concrete signer/transport.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
pub trait RpcClient: Send + Sync {
    /// Send a request and return the verified response payload.
    async fn call(&self, payload: Vec<u8>) -> Result<Vec<u8>>;

    /// Send a streaming request with ephemeral DH pubkey.
    async fn call_streaming(&self, payload: Vec<u8>, ephemeral_pubkey: [u8; 32]) -> Result<Vec<u8>>;

    /// Open a verified streaming subscription.
    async fn open_stream(&self, payload: Vec<u8>) -> Result<Box<dyn StreamHandle>>;

    /// Open a verified stream from pre-parsed StreamInfo + ephemeral keys.
    async fn open_stream_from_info(
        &self,
        stream_info: crate::stream_info::StreamInfo,
        client_secret: [u8; 32],
        client_pubkey: [u8; 32],
    ) -> Result<Box<dyn StreamHandle>>;

    /// Get the next request ID.
    fn next_id(&self) -> u64;

    /// Set the JWT token for authenticated requests.
    fn set_jwt(&self, token: Option<String>);
}

/// Blanket impl: any `RpcClientImpl<S, T>` satisfies `RpcClient`.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<S: Signer, T: Transport + 'static> RpcClient for RpcClientImpl<S, T> {
    async fn call(&self, payload: Vec<u8>) -> Result<Vec<u8>> {
        RpcClientImpl::call(self, payload).await
    }

    async fn call_streaming(&self, payload: Vec<u8>, ephemeral_pubkey: [u8; 32]) -> Result<Vec<u8>> {
        RpcClientImpl::call_streaming(self, payload, ephemeral_pubkey).await
    }

    #[cfg(not(feature = "fips"))]
    async fn open_stream(&self, payload: Vec<u8>) -> Result<Box<dyn StreamHandle>> {
        let handle = RpcClientImpl::open_stream(self, payload).await?;
        Ok(Box::new(handle))
    }

    #[cfg(feature = "fips")]
    async fn open_stream(&self, _payload: Vec<u8>) -> Result<Box<dyn StreamHandle>> {
        anyhow::bail!("streaming not available under FIPS")
    }

    #[cfg(not(feature = "fips"))]
    async fn open_stream_from_info(
        &self,
        stream_info: crate::stream_info::StreamInfo,
        client_secret: [u8; 32],
        client_pubkey: [u8; 32],
    ) -> Result<Box<dyn StreamHandle>> {
        let handle = RpcClientImpl::open_stream_from_info(self, stream_info, &client_secret, &client_pubkey).await?;
        Ok(Box::new(handle))
    }

    #[cfg(feature = "fips")]
    async fn open_stream_from_info(
        &self,
        _stream_info: crate::stream_info::StreamInfo,
        _client_secret: [u8; 32],
        _client_pubkey: [u8; 32],
    ) -> Result<Box<dyn StreamHandle>> {
        anyhow::bail!("streaming not available under FIPS")
    }

    fn next_id(&self) -> u64 {
        RpcClientImpl::next_id(self)
    }

    fn set_jwt(&self, token: Option<String>) {
        RpcClientImpl::set_jwt(self, token)
    }
}
