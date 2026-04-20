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
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;

use crate::crypto::VerifyingKey;
use crate::envelope::{
    self, RequestEnvelope, SignedEnvelope,
};
use crate::transport_traits::{Signer, Transport};
use crate::capnp::{FromCapnp, ToCapnp};
use crate::stream_consumer::{StreamHandle, StreamHandleImpl};

// ============================================================================
// Per-call options (non-mutating, Send+Sync, owned data)
// ============================================================================

/// Per-call options for RPC requests.
///
/// Used with `call_with_options()` and `RequestBuilder` to pass per-call
/// authentication context without mutating shared client state.
/// Safe for concurrent use — each call gets its own owned `CallOptions`.
///
/// # Security
///
/// - `jwt`: Overrides the client's default JWT for this call.
///   Used when a service needs to present a specific token.
/// - `delegated_bearer`: Bearer token relayed on behalf of a user.
///   Only trusted services (OAI, MCP) may set this. Policy gates which
///   services can relay bearer tokens. The server resolves the subject
///   from the bearer token, not the service identity.
#[derive(Debug, Clone, Default)]
pub struct CallOptions {
    /// Override the client's default JWT for this call.
    pub jwt: Option<String>,
    /// Bearer token relayed on behalf of a user (delegation).
    pub delegated_bearer: Option<String>,
}

impl CallOptions {
    /// Create empty options (no overrides).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a per-call JWT override.
    pub fn jwt(mut self, token: impl Into<String>) -> Self {
        self.jwt = Some(token.into());
        self
    }

    /// Set a delegated bearer token for relay.
    pub fn delegated_bearer(mut self, token: impl Into<String>) -> Self {
        self.delegated_bearer = Some(token.into());
        self
    }
}

// ============================================================================
// Per-call request builder
// ============================================================================

/// Per-call request builder for custom authentication options.
///
/// Created by `client.request()`, captures per-call auth context
/// without mutating the shared `Arc<dyn RpcClient>`. Safe for
/// concurrent use with connection pooling.
///
/// # Example
///
/// ```ignore
/// let response = model_client.request()
///     .delegated_bearer(user_bearer_token)
///     .call(payload)
///     .await?;
/// ```
pub struct RequestBuilder<'a> {
    client: &'a Arc<dyn RpcClient>,
    jwt: Option<String>,
    delegated_bearer: Option<String>,
}

impl<'a> std::fmt::Debug for RequestBuilder<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestBuilder")
            .field("jwt", &self.jwt.as_ref().map(|_| "***"))
            .field("delegated_bearer", &self.delegated_bearer.as_ref().map(|_| "***"))
            .finish()
    }
}

impl<'a> RequestBuilder<'a> {
    /// Create a builder wrapping the given client.
    pub fn new(client: &'a Arc<dyn RpcClient>) -> Self {
        Self {
            client,
            jwt: None,
            delegated_bearer: None,
        }
    }

    /// Set a per-call JWT override.
    pub fn jwt(mut self, token: impl Into<String>) -> Self {
        self.jwt = Some(token.into());
        self
    }

    /// Set a delegated bearer token for relay.
    pub fn delegated_bearer(mut self, token: impl Into<String>) -> Self {
        self.delegated_bearer = Some(token.into());
        self
    }

    /// Send a raw request with these options and return the raw response bytes.
    pub async fn call(self, payload: Vec<u8>) -> Result<Vec<u8>> {
        self.client.call_with_options(payload, CallOptions {
            jwt: self.jwt,
            delegated_bearer: self.delegated_bearer,
        }).await
    }
}

/// Unified RPC client. Evolved from ZmqClient — same envelope, timeout,
/// and resilience logic, parameterized over transport and signing.
pub struct RpcClientImpl<S: Signer, T: Transport + 'static> {
    pub signer: S,
    pub transport: T,
    pub server_verifying_key: VerifyingKey,
    default_jwt: Option<String>,
    request_id: AtomicU64,
}

impl<S: Signer, T: Transport + 'static> RpcClientImpl<S, T> {
    /// Create a new RPC client.
    pub fn new(signer: S, transport: T, server_verifying_key: VerifyingKey) -> Self {
        Self {
            signer,
            transport,
            server_verifying_key,
            default_jwt: None,
            request_id: AtomicU64::new(1),
        }
    }

    /// Create a new RPC client with a default JWT token.
    ///
    /// The token is stored immutably and used for all calls unless
    /// overridden by `call_with_options()`.
    pub fn with_default_jwt(mut self, token: String) -> Self {
        self.default_jwt = Some(token);
        self
    }

    /// Send a request and return the verified, unwrapped response payload.
    /// Uses the client's default JWT.
    pub async fn call(&self, payload: Vec<u8>) -> Result<Vec<u8>> {
        let request_id = self.next_id();
        let signed_bytes = self.sign_envelope(request_id, payload, None, self.default_jwt.clone(), None).await?;
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
    /// Uses the client's default JWT.
    pub async fn call_streaming(
        &self,
        payload: Vec<u8>,
        ephemeral_pubkey: [u8; 32],
    ) -> Result<Vec<u8>> {
        let request_id = self.next_id();
        let signed_bytes = self
            .sign_envelope(request_id, payload, Some(ephemeral_pubkey), self.default_jwt.clone(), None)
            .await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = envelope::unwrap_response(
            &response_bytes,
            Some(&self.server_verifying_key),
        )?;
        Ok(inner)
    }

    /// Send a streaming request with per-call authentication options.
    ///
    /// Uses `options.jwt` if provided, otherwise falls back to `default_jwt`.
    pub async fn call_streaming_with_options(
        &self,
        payload: Vec<u8>,
        ephemeral_pubkey: [u8; 32],
        options: CallOptions,
    ) -> Result<Vec<u8>> {
        let request_id = self.next_id();
        let jwt = options.jwt.or_else(|| self.default_jwt.clone());
        let signed_bytes = self
            .sign_envelope(request_id, payload, Some(ephemeral_pubkey), jwt, options.delegated_bearer)
            .await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = envelope::unwrap_response(
            &response_bytes,
            Some(&self.server_verifying_key),
        )?;
        Ok(inner)
    }

    /// Send a request with per-call authentication options.
    ///
    /// Uses `options.jwt` if provided, otherwise falls back to `default_jwt`.
    /// Includes `options.delegated_bearer` in the envelope for relay.
    pub async fn call_with_options(
        &self,
        payload: Vec<u8>,
        options: CallOptions,
    ) -> Result<Vec<u8>> {
        let request_id = self.next_id();
        let jwt = options.jwt.or_else(|| self.default_jwt.clone());
        let signed_bytes = self
            .sign_envelope(request_id, payload, None, jwt, options.delegated_bearer)
            .await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = envelope::unwrap_response(
            &response_bytes,
            Some(&self.server_verifying_key),
        )?;
        Ok(inner)
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
        jwt: Option<String>,
        delegated_bearer: Option<String>,
    ) -> Result<Vec<u8>> {
        let mut envelope = RequestEnvelope::new(self.signer.identity(), payload);
        envelope.request_id = request_id;
        if let Some(token) = jwt {
            envelope = envelope.with_jwt_token(token);
        }
        if let Some(bearer) = delegated_bearer {
            envelope = envelope.with_delegated_bearer(bearer);
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

    /// Open a verified streaming subscription with per-call authentication options.
    ///
    /// Same as `open_stream` but passes per-call JWT through the streaming
    /// request envelope. Used by WASM client where JWT is set post-construction.
    #[cfg(not(feature = "fips"))]
    pub async fn open_stream_with_options(
        &self,
        payload: Vec<u8>,
        options: CallOptions,
    ) -> Result<StreamHandleImpl<T>> {
        use crate::crypto::generate_ephemeral_keypair;

        let (secret, public) = generate_ephemeral_keypair();
        let pub_bytes = public.to_bytes();

        let response = self.call_streaming_with_options(payload, pub_bytes, options).await?;
        let stream_info = parse_stream_info(&response)?;

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

    /// Send a request with per-call authentication options.
    ///
    /// Uses `options.jwt` if provided, otherwise falls back to the client's
    /// default JWT. Includes `options.delegated_bearer` in the envelope.
    async fn call_with_options(&self, payload: Vec<u8>, options: CallOptions) -> Result<Vec<u8>>;

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
}

/// Blanket impl: any `RpcClientImpl<S, T>` satisfies `RpcClient`.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl<S: Signer, T: Transport + 'static> RpcClient for RpcClientImpl<S, T> {
    async fn call(&self, payload: Vec<u8>) -> Result<Vec<u8>> {
        RpcClientImpl::call(self, payload).await
    }

    async fn call_with_options(&self, payload: Vec<u8>, options: CallOptions) -> Result<Vec<u8>> {
        RpcClientImpl::call_with_options(self, payload, options).await
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
}
