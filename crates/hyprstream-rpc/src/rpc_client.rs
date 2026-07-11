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

type TokenProviderBox = Arc<dyn Fn() -> Option<String> + Send + Sync>;

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
    pub server_verifying_key: Option<VerifyingKey>,
    token_provider: Option<TokenProviderBox>,
    request_id: AtomicU64,
    /// Per-client response-verify policy override (#275/#277). `None` means "not
    /// explicitly set" — in that case `unwrap_response` consults the process-
    /// global response verify config ([`crate::envelope::global_response_verify_policy`]),
    /// which is fail-closed `Hybrid` when the daemon installed it (or when
    /// uninstalled in production). `Some(_)` pins the policy for this client,
    /// bypassing the global default (advanced / tests).
    response_verify_policy: Option<crate::crypto::CryptoPolicy>,
    /// Per-client kid-anchored ML-DSA-65 trust store used to resolve the server's
    /// PQ key for `ResponseEnvelope` verification (required under `Hybrid`). `None`
    /// falls back to the process-global response store
    /// ([`crate::envelope::global_response_pq_store`]).
    response_pq_store: Option<std::sync::Arc<dyn crate::envelope::PqTrustStore>>,
    /// INV-2 (ADR #1023): `true` when this client dials an **untrusted carrier**
    /// (iroh / relay / cross-host QUIC), on which a cleartext
    /// (`encrypted_envelope = None`) `SignedEnvelope` is forbidden. Set at dial
    /// time from [`crate::transport::EndpointType::forbids_cleartext_envelope`].
    /// Defaults to `false` (permissive) for directly-constructed clients (tests,
    /// wasm) — `dial()` is the production path that classifies correctly.
    carrier_forbids_cleartext: bool,
}

impl<S: Signer, T: Transport + 'static> RpcClientImpl<S, T> {
    /// Create a new RPC client.
    ///
    /// `server_verifying_key`: `None` skips response signature verification
    /// (TLS cert pinning still protects the connection).
    pub fn new(signer: S, transport: T, server_verifying_key: Option<VerifyingKey>) -> Self {
        Self {
            signer,
            transport,
            server_verifying_key,
            token_provider: None,
            request_id: AtomicU64::new(1),
            response_verify_policy: None,
            response_pq_store: None,
            carrier_forbids_cleartext: false,
        }
    }

    /// INV-2 (ADR #1023): mark this client's carrier as one that forbids a
    /// cleartext `SignedEnvelope` (iroh / relay / cross-host QUIC). Called by
    /// [`crate::dial::dial`] with
    /// [`crate::transport::EndpointType::forbids_cleartext_envelope`]. When set,
    /// the send path refuses to emit a cleartext envelope per the process INV-2
    /// mode (see [`crate::inv2`]) — fail-closed once the HyKEM tunnel is wired.
    pub fn with_carrier_cleartext_forbidden(mut self, forbidden: bool) -> Self {
        self.carrier_forbids_cleartext = forbidden;
        self
    }

    /// Enforce Hybrid (EdDSA + ML-DSA-65) verification of responses (#275),
    /// anchoring the server's ML-DSA-65 key via `pq_store`. Under this policy a
    /// classical-only, stripped-outer, self-cert, or unanchored response is
    /// REJECTED (fail-closed, no silent downgrade). Mirrors the request-side
    /// Hybrid verify config.
    pub fn with_response_pq_store(
        mut self,
        pq_store: std::sync::Arc<dyn crate::envelope::PqTrustStore>,
    ) -> Self {
        self.response_pq_store = Some(pq_store);
        self.response_verify_policy = Some(crate::crypto::CryptoPolicy::Hybrid);
        self
    }

    /// Set the response-verify policy explicitly (advanced). Under `Hybrid` a
    /// PQ trust store MUST also be set via [`Self::with_response_pq_store`].
    /// Setting this pins the policy for this client and bypasses the process-
    /// global response default consulted by [`Self::unwrap_response`].
    pub fn with_response_verify_policy(mut self, policy: crate::crypto::CryptoPolicy) -> Self {
        self.response_verify_policy = Some(policy);
        self
    }

    /// Set a dynamic token provider called on every RPC request.
    ///
    /// The closure is called once per `call()` / `call_with_options()` invocation
    /// so short-lived tokens (OAuth at+jwt, WIT) are always fresh without
    /// reconstructing the client. On WASM, wrap `js_sys::Function` in a
    /// `Send + Sync` newtype before passing it here (WASM is single-threaded).
    pub fn with_token_provider<F>(mut self, f: F) -> Self
    where
        F: Fn() -> Option<String> + Send + Sync + 'static,
    {
        self.token_provider = Some(Arc::new(f));
        self
    }

    /// Set a static default JWT. Sugar over `with_token_provider`.
    pub fn with_default_jwt(self, token: String) -> Self {
        self.with_token_provider(move || Some(token.clone()))
    }

    fn effective_jwt(&self) -> Option<String> {
        self.token_provider.as_ref().and_then(|f| f())
    }

    /// Send a request and return the verified, unwrapped response payload.
    /// Uses the client's default JWT.
    pub async fn call(&self, payload: Vec<u8>) -> Result<Vec<u8>> {
        let request_id = self.next_id();
        let signed_bytes = self.sign_envelope(request_id, payload, None, self.effective_jwt(), None).await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = self.unwrap_response(&response_bytes)?;
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
            .sign_envelope(request_id, payload, Some(ephemeral_pubkey), self.effective_jwt(), None)
            .await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = self.unwrap_response(&response_bytes)?;
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
        let jwt = options.jwt.or_else(|| self.effective_jwt());
        let signed_bytes = self
            .sign_envelope(request_id, payload, Some(ephemeral_pubkey), jwt, options.delegated_bearer)
            .await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = self.unwrap_response(&response_bytes)?;
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
        let jwt = options.jwt.or_else(|| self.effective_jwt());
        let signed_bytes = self
            .sign_envelope(request_id, payload, None, jwt, options.delegated_bearer)
            .await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = self.unwrap_response(&response_bytes)?;
        Ok(inner)
    }

    /// Get the next monotonically increasing request ID.
    pub fn next_id(&self) -> u64 {
        self.request_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Unwrap + verify a `ResponseEnvelope` (#275/#277).
    ///
    /// Resolves the effective response-verify policy and PQ trust store: a
    /// per-client override (set via [`Self::with_response_pq_store`] /
    /// [`Self::with_response_verify_policy`]) wins; otherwise the process-global
    /// response default ([`envelope::global_response_verify_policy`] +
    /// [`envelope::global_response_pq_store`]) is consulted. That global default
    /// is fail-closed `Hybrid` in production — symmetric to the request-side
    /// `global_verify_policy` — so a classical-only / stripped response is
    /// rejected by default unless an operator opted into the `classical` escape
    /// hatch. Under `Hybrid` the server's kid-anchored ML-DSA-65 layer is
    /// required (no silent downgrade).
    #[cfg(not(target_arch = "wasm32"))]
    fn unwrap_response(&self, response_bytes: &[u8]) -> Result<(u64, Vec<u8>)> {
        let policy = self
            .response_verify_policy
            .unwrap_or_else(envelope::global_response_verify_policy);
        // Prefer the per-client store; else the process-global one. Hold the Arc
        // so the borrow passed to `unwrap_response_with` outlives the call.
        let global_store = if self.response_pq_store.is_none() {
            envelope::global_response_pq_store()
        } else {
            None
        };
        let store: Option<&dyn envelope::PqTrustStore> = self
            .response_pq_store
            .as_deref()
            .or(global_store.as_deref());
        envelope::unwrap_response_with(
            response_bytes,
            self.server_verifying_key.as_ref(),
            store,
            policy,
        )
    }

    /// WASM response unwrap (#277 scope boundary): no process-global response
    /// config exists on wasm, so this keeps the per-client behavior. The wasm
    /// verify policy is owned by #158 (`wasm_api.rs`) and is intentionally NOT
    /// changed here.
    #[cfg(target_arch = "wasm32")]
    fn unwrap_response(&self, response_bytes: &[u8]) -> Result<(u64, Vec<u8>)> {
        envelope::unwrap_response_with(
            response_bytes,
            self.server_verifying_key.as_ref(),
            self.response_pq_store.as_deref(),
            self.response_verify_policy
                .unwrap_or(crate::crypto::CryptoPolicy::Classical),
        )
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
        let mut envelope = RequestEnvelope::new(payload);
        envelope.request_id = request_id;
        if let Some(token) = jwt {
            envelope = envelope.with_jwt_token(token);
        }
        if let Some(bearer) = delegated_bearer {
            envelope = envelope.with_delegation_token(bearer);
        }
        if let Some(key) = ephemeral_pubkey {
            envelope = envelope.with_client_dh_public(key);
        }

        let canonical = envelope.to_bytes();
        let ed_pubkey = self.signer.pubkey();

        // Raw EdDSA signature (sig/cnf) retained for signer-pubkey advertisement
        // + the JWT cnf key-binding path.
        let signature = self.signer.sign(&canonical).await?;

        // Build the COSE composite (M3 #152) by signing each entry's
        // Sig_structure out-of-band via the (possibly async/WASM) Signer.
        let aad = crate::crypto::cose_sign1::build_external_aad(
            crate::envelope::ENVELOPE_SCHEMA_ID,
            crate::envelope::REQUEST_ENVELOPE_TYPE_ID,
        );
        // Nested SNS composite: sign the inner EdDSA layer first, then sign the
        // outer ML-DSA-65 layer over `canonical ‖ inner_eddsa_signature` so the
        // inner signature is bound into the outer (Strong-Non-Separable).
        //
        // Hybrid iff the signer exposes an ML-DSA-65 key. In Hybrid mode the inner
        // EdDSA layer binds the hybrid-composite alg-id into its AAD (#278), making
        // it byte-distinct from a Classical inner layer.
        let hybrid = self.signer.pq_pubkey().is_some();
        let ed_kid = ed_pubkey.to_vec();
        let ed_tbs = crate::crypto::cose_sign::inner_tbs(ed_kid.clone(), &canonical, &aad, hybrid);
        let ed_sig = self.signer.sign(&ed_tbs).await?.to_vec();

        // Hybrid component when the signer exposes an ML-DSA-65 key.
        let pq_entry = if let Some(pq_kid) = self.signer.pq_pubkey() {
            let pq_tbs = crate::crypto::cose_sign::outer_tbs(
                pq_kid.clone(),
                &canonical,
                &ed_sig,
                &aad,
            );
            // The inner EdDSA above was already bound to the hybrid-composite
            // alg-id (`hybrid = pq_pubkey().is_some()`), so it cannot fall back to
            // a classical inner. A signer that advertises a PQ key but declines to
            // sign must therefore be a HARD ERROR, not a silent downgrade that
            // would desync inner-AAD (hybrid) from outer-presence (none) and make
            // the peer's inner verification fail (#278 review).
            let pq_sig = self
                .signer
                .pq_sign(&pq_tbs)
                .await?
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "signer exposes a PQ public key but pq_sign() returned no \
                         signature; refusing to emit a hybrid-bound inner without \
                         its ML-DSA-65 outer (#278)"
                    )
                })?;
            Some((pq_kid, pq_sig))
        } else {
            None
        };
        let policy = if pq_entry.is_some() {
            crate::crypto::CryptoPolicy::Hybrid
        } else {
            crate::crypto::CryptoPolicy::Classical
        };
        let cose = crate::crypto::cose_sign::assemble_composite_nested((ed_kid, ed_sig), pq_entry)?;

        let signed = SignedEnvelope {
            envelope,
            sig: signature,
            cnf: ed_pubkey,
            encrypted_envelope: None,
            client_ephemeral_public: None,
            cose,
            policy,
            pq_kem_ciphertext: None,
        };

        // INV-2 (ADR #1023): structural, fail-closed refusal to hand a cleartext
        // `SignedEnvelope` (encrypted_envelope = None, above) to an untrusted
        // carrier. This is the single chokepoint for the RPC request send path —
        // every `call*` variant funnels through `sign_envelope` before touching
        // the transport — so the guard cannot be bypassed by a call flavor.
        // Under the interim WarnOnly mode (HyKEM tunnel #551 not yet wired) it
        // logs and proceeds; under Enforce it returns Err before any byte ships.
        crate::inv2::guard_cleartext_envelope(
            self.carrier_forbids_cleartext,
            signed.is_encrypted(),
        )?;

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
    // Borrowing (zero-copy) reader: the capnp `Reader` references `bytes` directly.
    let mut slice: &[u8] = bytes;
    let reader = capnp::serialize::read_message_from_flat_slice(
        &mut slice,
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod inv2_send_path_tests {
    //! INV-2 (ADR #1023) end-to-end wiring test: prove the structural guard
    //! fires on the real `RpcClientImpl` send path (`sign_envelope`) BEFORE any
    //! byte is handed to the transport — i.e. a cleartext envelope can never
    //! reach an untrusted carrier when enforcement is on. This is the
    //! adversarial check that the fix is not merely cosmetic.

    use super::*;
    use crate::signer::LocalSigner;
    use crate::transport_traits::{PublishSink, Transport};
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A transport that records whether `send` was ever reached. If the INV-2
    /// guard works, an untrusted-carrier client under Enforce must return an
    /// error WITHOUT `send` being called.
    struct SentinelTransport {
        sent: Arc<AtomicBool>,
    }

    struct NoopSink;

    #[async_trait]
    impl PublishSink for NoopSink {
        async fn send_frames(&self, _frames: &[&[u8]]) -> Result<()> {
            Ok(())
        }
    }

    #[async_trait]
    impl Transport for SentinelTransport {
        type Sub = futures::stream::Empty<Result<Vec<Vec<u8>>>>;
        type Pub = NoopSink;

        async fn send(&self, _payload: Vec<u8>, _timeout_ms: Option<i32>) -> Result<Vec<u8>> {
            self.sent.store(true, Ordering::SeqCst);
            // Return bogus bytes: if the guard failed to fire, response unwrap
            // would fail here too — but we assert on `sent` to be unambiguous.
            Ok(Vec::new())
        }

        async fn subscribe(&self, _topic: &[u8]) -> Result<Self::Sub> {
            Ok(futures::stream::empty())
        }

        async fn publish(&self, _topic: &[u8]) -> Result<Self::Pub> {
            Ok(NoopSink)
        }
    }

    fn test_signer() -> LocalSigner {
        let (sk, _vk) = crate::crypto::signing::generate_signing_keypair();
        LocalSigner::new(sk)
    }

    #[tokio::test]
    async fn cleartext_over_untrusted_carrier_is_refused_before_send() {
        // Force fail-closed enforcement for this process (OnceLock: the only
        // setter; Enforce is harmless to trusted-carrier tests which classify
        // carrier_forbids = false and are therefore always Allowed).
        crate::inv2::set_inv2_mode(crate::inv2::Inv2Mode::Enforce);

        let sent = Arc::new(AtomicBool::new(false));
        let transport = SentinelTransport { sent: sent.clone() };
        // carrier_forbids_cleartext = true == the iroh / relay / cross-host case.
        let client = RpcClientImpl::new(test_signer(), transport, None)
            .with_carrier_cleartext_forbidden(true);

        let result = client.call(b"payload".to_vec()).await;

        assert!(
            result.is_err(),
            "INV-2: a cleartext envelope over an untrusted carrier must be refused"
        );
        assert!(
            result.unwrap_err().to_string().contains("INV-2"),
            "the refusal must be the INV-2 guard, not an unrelated error"
        );
        assert!(
            !sent.load(Ordering::SeqCst),
            "INV-2: refusal MUST happen before any byte reaches the transport"
        );
    }

    #[tokio::test]
    async fn cleartext_over_trusted_carrier_is_allowed() {
        // A trusted carrier (inproc/UDS) classifies carrier_forbids = false, so
        // the same cleartext send proceeds to the transport even under Enforce.
        crate::inv2::set_inv2_mode(crate::inv2::Inv2Mode::Enforce);

        let sent = Arc::new(AtomicBool::new(false));
        let transport = SentinelTransport { sent: sent.clone() };
        let client = RpcClientImpl::new(test_signer(), transport, None)
            .with_carrier_cleartext_forbidden(false);

        // Response unwrap fails on the empty bogus reply, but that is AFTER send:
        // the point is the guard did not block the trusted-carrier path.
        let _ = client.call(b"payload".to_vec()).await;
        assert!(
            sent.load(Ordering::SeqCst),
            "trusted-carrier cleartext must NOT be blocked by INV-2"
        );
    }
}
