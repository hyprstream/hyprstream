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

use crate::capnp::{FromCapnp, ToCapnp};
use crate::crypto::hybrid_kem::KemTrustStore;
use crate::crypto::VerifyingKey;
use crate::envelope::{self, RequestEnvelope, SignedEnvelope};
use crate::stream_consumer::{StreamHandle, StreamHandleImpl};
use crate::transport_traits::{Signer, Transport};

type TokenProviderBox = Arc<dyn Fn() -> Option<String> + Send + Sync>;

/// Non-cloneable, non-serializable one-call response decapsulation material.
/// The component secret bytes are `Zeroizing` inside `RecipientKeypair` and
/// are dropped immediately after the sole response-open attempt.
struct ResponseRecipientSecret {
    keypair: crate::crypto::hybrid_kem::RecipientKeypair,
    #[cfg(test)]
    drop_probe: Option<Arc<std::sync::atomic::AtomicUsize>>,
}

#[cfg(test)]
impl Drop for ResponseRecipientSecret {
    fn drop(&mut self) {
        if let Some(probe) = &self.drop_probe {
            probe.fetch_add(1, Ordering::SeqCst);
        }
    }
}

struct PendingResponse {
    request_id: u64,
    request_iat: i64,
    request_nonce: [u8; 16],
    server_identity: [u8; 32],
    recipient_public: crate::crypto::hybrid_kem::RecipientPublic,
    service_domain: String,
    secret: ResponseRecipientSecret,
}

impl std::fmt::Debug for PendingResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingResponse")
            .field("request_id", &self.request_id)
            .field("server_identity", &hex::encode(self.server_identity))
            .field("service_domain", &self.service_domain)
            .finish_non_exhaustive()
    }
}

impl PendingResponse {
    fn open(self, envelope: &crate::envelope::ResponseEnvelope) -> Result<Vec<u8>> {
        if envelope.request_id != self.request_id {
            anyhow::bail!(
                "response request id {} does not match pending request {}",
                envelope.request_id,
                self.request_id
            );
        }
        envelope.open_encrypted(
            &self.secret.keypair,
            &self.recipient_public,
            self.request_iat,
            &self.request_nonce,
            &self.server_identity,
            &self.service_domain,
        )
    }
}

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
    service_domain: Option<&'static str>,
    jwt: Option<String>,
    delegated_bearer: Option<String>,
}

impl<'a> std::fmt::Debug for RequestBuilder<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RequestBuilder")
            .field("jwt", &self.jwt.as_ref().map(|_| "***"))
            .field(
                "delegated_bearer",
                &self.delegated_bearer.as_ref().map(|_| "***"),
            )
            .finish()
    }
}

impl<'a> RequestBuilder<'a> {
    /// Create a builder wrapping the given client.
    pub fn new(client: &'a Arc<dyn RpcClient>) -> Self {
        Self {
            client,
            service_domain: None,
            jwt: None,
            delegated_bearer: None,
        }
    }

    /// Create a builder bound to a generated client's canonical service.
    pub fn for_service(client: &'a Arc<dyn RpcClient>, service_domain: &'static str) -> Self {
        Self {
            client,
            service_domain: Some(service_domain),
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
        let options = CallOptions {
            jwt: self.jwt,
            delegated_bearer: self.delegated_bearer,
        };
        match self.service_domain {
            Some(service_domain) => {
                self.client
                    .call_with_options_for_service(service_domain, payload, options)
                    .await
            }
            None => self.client.call_with_options(payload, options).await,
        }
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
    /// Per-client kid-anchored `#mesh-kem` trust store used to encrypt request
    /// envelopes for carriers that forbid cleartext. Bindings are established
    /// out-of-band by DID keyAgreement / peer attestation; absence fails closed.
    request_kem_store: Option<Arc<dyn KemTrustStore>>,
    #[cfg(test)]
    response_secret_drop_probe: Option<Arc<std::sync::atomic::AtomicUsize>>,
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
            request_kem_store: None,
            #[cfg(test)]
            response_secret_drop_probe: None,
        }
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

    /// Set the anchored `#mesh-kem` recipient key store used to encrypt request
    /// envelopes when the selected transport forbids cleartext.
    pub fn with_request_kem_store(mut self, kem_store: Arc<dyn KemTrustStore>) -> Self {
        self.request_kem_store = Some(kem_store);
        self
    }

    #[cfg(test)]
    fn with_response_secret_drop_probe(
        mut self,
        probe: Arc<std::sync::atomic::AtomicUsize>,
    ) -> Self {
        self.response_secret_drop_probe = Some(probe);
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
        self.call_bound(payload, None).await
    }

    /// Send a request bound to a canonical destination service. Generated
    /// clients always use this path with their `SERVICE_NAME`.
    pub async fn call_for_service(
        &self,
        service_domain: &str,
        payload: Vec<u8>,
    ) -> Result<Vec<u8>> {
        self.call_bound(payload, Some(service_domain)).await
    }

    async fn call_bound(&self, payload: Vec<u8>, service_domain: Option<&str>) -> Result<Vec<u8>> {
        let request_id = self.next_id();
        let (signed_bytes, pending) = self
            .sign_envelope(
                request_id,
                payload,
                None,
                None,
                self.effective_jwt(),
                None,
                service_domain,
            )
            .await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = self.unwrap_response(&response_bytes, request_id, pending)?;
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
        self.call_streaming_bound(payload, ephemeral_pubkey, None)
            .await
    }

    /// Send a streaming setup request bound to a canonical destination.
    pub async fn call_streaming_for_service(
        &self,
        service_domain: &str,
        payload: Vec<u8>,
        ephemeral_pubkey: [u8; 32],
    ) -> Result<Vec<u8>> {
        self.call_streaming_bound(payload, ephemeral_pubkey, Some(service_domain))
            .await
    }

    async fn call_streaming_bound(
        &self,
        payload: Vec<u8>,
        ephemeral_pubkey: [u8; 32],
        service_domain: Option<&str>,
    ) -> Result<Vec<u8>> {
        let request_id = self.next_id();
        let (signed_bytes, pending) = self
            .sign_envelope(
                request_id,
                payload,
                Some(ephemeral_pubkey),
                None,
                self.effective_jwt(),
                None,
                service_domain,
            )
            .await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = self.unwrap_response(&response_bytes, request_id, pending)?;
        Ok(inner)
    }

    /// Send an identified stream setup using the pinned classical+ML-KEM
    /// recipient.  The caller retains the matching keypair and combines the
    /// returned `StreamInfo.kemCiphertexts` with its accepted-current
    /// [`IdentifiedStreamBinding`] before opening the stream.
    pub async fn call_streaming_identified_for_service(
        &self,
        service_domain: &str,
        payload: Vec<u8>,
        recipient: crate::crypto::hybrid_kem::RecipientPublic,
    ) -> Result<Vec<u8>> {
        if recipient.suite_id != crate::stream_epoch::IDENTIFIED_STREAM_SUITE {
            anyhow::bail!("identified stream recipient does not use the pinned HyKEM suite");
        }
        let request_id = self.next_id();
        let (signed_bytes, pending) = self
            .sign_envelope(
                request_id,
                payload,
                None,
                Some(recipient),
                self.effective_jwt(),
                None,
                Some(service_domain),
            )
            .await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = self.unwrap_response(&response_bytes, request_id, pending)?;
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
        self.call_streaming_with_options_bound(payload, ephemeral_pubkey, options, None)
            .await
    }

    async fn call_streaming_with_options_bound(
        &self,
        payload: Vec<u8>,
        ephemeral_pubkey: [u8; 32],
        options: CallOptions,
        service_domain: Option<&str>,
    ) -> Result<Vec<u8>> {
        let request_id = self.next_id();
        let jwt = options.jwt.or_else(|| self.effective_jwt());
        let (signed_bytes, pending) = self
            .sign_envelope(
                request_id,
                payload,
                Some(ephemeral_pubkey),
                None,
                jwt,
                options.delegated_bearer,
                service_domain,
            )
            .await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = self.unwrap_response(&response_bytes, request_id, pending)?;
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
        self.call_with_options_bound(payload, options, None).await
    }

    /// Send an option-bearing request bound to a canonical destination.
    pub async fn call_with_options_for_service(
        &self,
        service_domain: &str,
        payload: Vec<u8>,
        options: CallOptions,
    ) -> Result<Vec<u8>> {
        self.call_with_options_bound(payload, options, Some(service_domain))
            .await
    }

    async fn call_with_options_bound(
        &self,
        payload: Vec<u8>,
        options: CallOptions,
        service_domain: Option<&str>,
    ) -> Result<Vec<u8>> {
        let request_id = self.next_id();
        let jwt = options.jwt.or_else(|| self.effective_jwt());
        let (signed_bytes, pending) = self
            .sign_envelope(
                request_id,
                payload,
                None,
                None,
                jwt,
                options.delegated_bearer,
                service_domain,
            )
            .await?;
        let timeout = self.calculate_timeout();
        let response_bytes = self.transport.send(signed_bytes, timeout).await?;
        let (_req_id, inner) = self.unwrap_response(&response_bytes, request_id, pending)?;
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
    fn unwrap_response(
        &self,
        response_bytes: &[u8],
        request_id: u64,
        pending: Option<PendingResponse>,
    ) -> Result<(u64, Vec<u8>)> {
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
        self.unwrap_response_with_pending(response_bytes, request_id, pending, store, policy)
    }

    /// WASM response unwrap (#277 scope boundary): no process-global response
    /// config exists on wasm, so this keeps the per-client behavior. The wasm
    /// verify policy is owned by #158 (`wasm_api.rs`) and is intentionally NOT
    /// changed here.
    #[cfg(target_arch = "wasm32")]
    fn unwrap_response(
        &self,
        response_bytes: &[u8],
        request_id: u64,
        pending: Option<PendingResponse>,
    ) -> Result<(u64, Vec<u8>)> {
        self.unwrap_response_with_pending(
            response_bytes,
            request_id,
            pending,
            self.response_pq_store.as_deref(),
            self.response_verify_policy
                .unwrap_or(crate::crypto::CryptoPolicy::Classical),
        )
    }

    fn unwrap_response_with_pending(
        &self,
        response_bytes: &[u8],
        request_id: u64,
        pending: Option<PendingResponse>,
        pq_store: Option<&dyn envelope::PqTrustStore>,
        policy: crate::crypto::CryptoPolicy,
    ) -> Result<(u64, Vec<u8>)> {
        if self.transport.forbids_cleartext_envelope() {
            let pending = pending.ok_or_else(|| {
                anyhow::anyhow!("network response has no one-shot pending recipient")
            })?;
            // Carrier gate first: inspect only the bounded encrypted field and
            // reject cleartext before copying or using the outer payload.
            let response = envelope::read_response_envelope(response_bytes, true)?;
            let server = self
                .server_verifying_key
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("network response has no pinned server identity"))?;
            let store = pq_store.ok_or_else(|| {
                anyhow::anyhow!("network response has no anchored ML-DSA-65 trust store")
            })?;
            if store.ml_dsa_key_for(&server.to_bytes()).is_none() {
                anyhow::bail!("network response server has no anchored ML-DSA-65 key");
            }
            response.verify_encrypted_with_service_domain(
                server,
                store,
                &pending.service_domain,
            )?;
            let payload = pending.open(&response)?;
            Ok((response.request_id, payload))
        } else {
            let (response_id, payload) = envelope::unwrap_response_with(
                response_bytes,
                self.server_verifying_key.as_ref(),
                pq_store,
                policy,
            )?;
            if response_id != request_id {
                anyhow::bail!(
                    "response request id {} does not match pending request {}",
                    response_id,
                    request_id
                );
            }
            Ok((response_id, payload))
        }
    }

    /// Build, sign, and serialize a request envelope.
    async fn sign_envelope(
        &self,
        request_id: u64,
        payload: Vec<u8>,
        ephemeral_pubkey: Option<[u8; 32]>,
        stream_kem_recipient: Option<crate::crypto::hybrid_kem::RecipientPublic>,
        jwt: Option<String>,
        delegated_bearer: Option<String>,
        service_domain: Option<&str>,
    ) -> Result<(Vec<u8>, Option<PendingResponse>)> {
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
        if let Some(recipient) = stream_kem_recipient {
            envelope = envelope.with_client_kem_public(recipient)?;
        }
        if let Some(service_domain) = service_domain {
            envelope = envelope.with_service_domain(service_domain)?;
        }

        let (pending, encrypted_envelope) = if self.transport.forbids_cleartext_envelope() {
            let service_domain = service_domain.ok_or_else(|| {
                anyhow::anyhow!(
                    "network request has no canonical service domain; use a generated client or explicit service-bound call"
                )
            })?;
            crate::envelope::validate_service_domain(service_domain)?;
            let server = self.server_verifying_key.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "transport forbids cleartext envelopes but no server verifying key is pinned; refusing to choose a #mesh-kem recipient"
                )
            })?;
            let kem_store = self.request_kem_store.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "transport forbids cleartext envelopes but no #mesh-kem trust store is configured; refusing cleartext downgrade"
                )
            })?;
            let recipient = kem_store
                .kem_recipient_for(&server.to_bytes())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "transport forbids cleartext envelopes but server {} has no anchored #mesh-kem recipient key",
                        hex::encode(server.to_bytes())
                    )
                })?;
            let keypair = crate::crypto::hybrid_kem::generate_recipient(
                crate::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
            )?;
            let public = keypair.public();
            envelope = envelope.with_response_kem_recipient(public.clone());
            let pending = Some(PendingResponse {
                request_id,
                request_iat: envelope.iat,
                request_nonce: envelope.nonce,
                server_identity: server.to_bytes(),
                recipient_public: public,
                service_domain: service_domain.to_owned(),
                secret: ResponseRecipientSecret {
                    keypair,
                    #[cfg(test)]
                    drop_probe: self.response_secret_drop_probe.clone(),
                },
            });
            let sealed = Some(crate::envelope::seal_request_envelope(
                &envelope, &recipient,
            )?);
            (pending, sealed)
        } else {
            (None, None)
        };

        let canonical = envelope.to_bytes();
        let ed_pubkey = self.signer.pubkey();
        let signing_data: &[u8] = encrypted_envelope.as_deref().unwrap_or(&canonical);

        if encrypted_envelope.is_some() && self.signer.pq_pubkey().is_none() {
            anyhow::bail!(
                "transport forbids cleartext envelopes but signer has no ML-DSA-65 \
                 key; refusing to emit encrypted envelope without hybrid signature"
            );
        }

        // Raw EdDSA signature (sig/cnf) retained for signer-pubkey advertisement
        // + the JWT cnf key-binding path.
        let signature = self.signer.sign(signing_data).await?;

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
        let ed_tbs =
            crate::crypto::cose_sign::inner_tbs(ed_kid.clone(), signing_data, &aad, hybrid);
        let ed_sig = self.signer.sign(&ed_tbs).await?.to_vec();

        // Hybrid component when the signer exposes an ML-DSA-65 key.
        let pq_entry = if let Some(pq_kid) = self.signer.pq_pubkey() {
            let pq_tbs =
                crate::crypto::cose_sign::outer_tbs(pq_kid.clone(), signing_data, &ed_sig, &aad);
            // The inner EdDSA above was already bound to the hybrid-composite
            // alg-id (`hybrid = pq_pubkey().is_some()`), so it cannot fall back to
            // a classical inner. A signer that advertises a PQ key but declines to
            // sign must therefore be a HARD ERROR, not a silent downgrade that
            // would desync inner-AAD (hybrid) from outer-presence (none) and make
            // the peer's inner verification fail (#278 review).
            let pq_sig = self.signer.pq_sign(&pq_tbs).await?.ok_or_else(|| {
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
            encrypted_envelope,
            client_ephemeral_public: None,
            cose,
            policy,
            pq_kem_ciphertext: None,
        };

        let mut message = capnp::message::Builder::new_default();
        {
            let mut builder = message.init_root::<crate::common_capnp::signed_envelope::Builder>();
            signed.write_to(&mut builder);
        }
        let mut bytes = Vec::new();
        capnp::serialize::write_message(&mut bytes, &message)?;
        Ok((bytes, pending))
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

        let response = self
            .call_streaming_with_options(payload, pub_bytes, options)
            .await?;
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

    /// Send a request bound to a canonical destination service.
    async fn call_for_service(&self, service_domain: &str, payload: Vec<u8>) -> Result<Vec<u8>>;

    /// Send a request with per-call authentication options.
    ///
    /// Uses `options.jwt` if provided, otherwise falls back to the client's
    /// default JWT. Includes `options.delegated_bearer` in the envelope.
    async fn call_with_options(&self, payload: Vec<u8>, options: CallOptions) -> Result<Vec<u8>>;

    /// Send an option-bearing request bound to a canonical destination.
    async fn call_with_options_for_service(
        &self,
        service_domain: &str,
        payload: Vec<u8>,
        options: CallOptions,
    ) -> Result<Vec<u8>>;

    /// Send a streaming request with ephemeral DH pubkey.
    async fn call_streaming(&self, payload: Vec<u8>, ephemeral_pubkey: [u8; 32])
        -> Result<Vec<u8>>;

    /// Send a streaming setup request bound to a canonical destination.
    async fn call_streaming_for_service(
        &self,
        service_domain: &str,
        payload: Vec<u8>,
        ephemeral_pubkey: [u8; 32],
    ) -> Result<Vec<u8>>;

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

    async fn call_for_service(&self, service_domain: &str, payload: Vec<u8>) -> Result<Vec<u8>> {
        RpcClientImpl::call_for_service(self, service_domain, payload).await
    }

    async fn call_with_options(&self, payload: Vec<u8>, options: CallOptions) -> Result<Vec<u8>> {
        RpcClientImpl::call_with_options(self, payload, options).await
    }

    async fn call_with_options_for_service(
        &self,
        service_domain: &str,
        payload: Vec<u8>,
        options: CallOptions,
    ) -> Result<Vec<u8>> {
        RpcClientImpl::call_with_options_for_service(self, service_domain, payload, options).await
    }

    async fn call_streaming(
        &self,
        payload: Vec<u8>,
        ephemeral_pubkey: [u8; 32],
    ) -> Result<Vec<u8>> {
        RpcClientImpl::call_streaming(self, payload, ephemeral_pubkey).await
    }

    async fn call_streaming_for_service(
        &self,
        service_domain: &str,
        payload: Vec<u8>,
        ephemeral_pubkey: [u8; 32],
    ) -> Result<Vec<u8>> {
        RpcClientImpl::call_streaming_for_service(self, service_domain, payload, ephemeral_pubkey)
            .await
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
        let handle =
            RpcClientImpl::open_stream_from_info(self, stream_info, &client_secret, &client_pubkey)
                .await?;
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod request_kem_tests {
    //! Coverage for the request KEM-store plumbing that `dial_wasm::dial` /
    //! `dial_wasm::dial_with_kem_store` delegate to.
    //!
    //! `dial_wasm` is `#![cfg(target_arch = "wasm32")]` and its bodies call
    //! `WtConnection::connect()` (the browser WebTransport API), so they cannot
    //! be exercised by native `cargo test`, and the workspace has no
    //! `wasm-bindgen-test` browser runner. The security-relevant behavior those
    //! functions delegate to, however — "a cleartext-forbidding carrier with a
    //! forwarded `#mesh-kem` store encrypts, and with NO store fails closed" —
    //! lives entirely in `RpcClientImpl::sign_envelope` and is fully testable
    //! here over a mock forbidding transport. `dial_wasm`'s own logic is the
    //! same `Some(store) => with_request_kem_store(store)` / `None => client`
    //! match as native `dial` (already covered by the iroh e2e round-trip).

    use super::*;
    use crate::crypto::hybrid_kem::KeyedKemTrustStore;
    use crate::crypto::signing::generate_signing_keypair;
    use crate::node_identity::derive_mesh_kem_recipient;
    use crate::signer::LocalSigner;
    use crate::transport::rpc_session::{RpcPendingStream, RpcPublishStub};
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
    use tokio::sync::{oneshot, Mutex, Notify};

    /// A carrier that forbids cleartext envelopes; `send` must never be reached
    /// in these tests (they stop at `sign_envelope`).
    struct ForbidsCleartextMock;

    struct CleartextResponseMock(Vec<u8>);

    const LIFECYCLE_SUCCESS: u8 = 0;
    const LIFECYCLE_FAIL_SEND: u8 = 1;
    const LIFECYCLE_BLOCK: u8 = 2;
    const LIFECYCLE_SWAP: u8 = 3;

    struct LifecycleState {
        behavior: AtomicU8,
        recipients: parking_lot::Mutex<Vec<Vec<u8>>>,
        swap_waiters: Mutex<Vec<(Vec<u8>, oneshot::Sender<Vec<u8>>)>>,
        entered: Notify,
    }

    struct LifecycleTransport {
        server_sk: crate::crypto::SigningKey,
        state: Arc<LifecycleState>,
    }

    impl LifecycleTransport {
        fn response_for(&self, payload: &[u8]) -> Result<Vec<u8>> {
            let mut signed = signed_envelope_from_bytes(payload);
            let server_recipient = derive_mesh_kem_recipient(&self.server_sk)?;
            signed.decrypt_in_place_mesh_kem(&server_recipient)?;
            let request = signed.envelope;
            let recipient = request
                .response_kem_recipient
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("test request omitted response recipient"))?;
            let service_domain = request
                .service_domain
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("test request omitted service domain"))?;
            self.state.recipients.lock().push(recipient.encode());
            let pq_sk = crate::node_identity::derive_mesh_mldsa_key(&self.server_sk);
            let response = crate::envelope::ResponseEnvelope::new_signed_encrypted(
                request.request_id,
                request.payload,
                &self.server_sk,
                &pq_sk,
                recipient,
                request.iat,
                &request.nonce,
                service_domain,
            )?;
            let mut message = capnp::message::Builder::new_default();
            response.write_to(
                &mut message.init_root::<crate::common_capnp::response_envelope::Builder>(),
            );
            let mut bytes = Vec::new();
            capnp::serialize::write_message(&mut bytes, &message)?;
            Ok(bytes)
        }
    }

    #[async_trait]
    impl Transport for LifecycleTransport {
        type Sub = RpcPendingStream;
        type Pub = RpcPublishStub;

        fn forbids_cleartext_envelope(&self) -> bool {
            true
        }

        async fn send(&self, payload: Vec<u8>, _timeout_ms: Option<i32>) -> Result<Vec<u8>> {
            let response = self.response_for(&payload)?;
            match self.state.behavior.load(Ordering::SeqCst) {
                LIFECYCLE_SUCCESS => Ok(response),
                LIFECYCLE_FAIL_SEND => anyhow::bail!("controlled send failure"),
                LIFECYCLE_BLOCK => {
                    self.state.entered.notify_waiters();
                    std::future::pending().await
                }
                LIFECYCLE_SWAP => {
                    let (tx, rx) = oneshot::channel();
                    let mut waiters = self.state.swap_waiters.lock().await;
                    waiters.push((response, tx));
                    if waiters.len() == 2 {
                        let (second_response, second_tx) = waiters.pop().unwrap();
                        let (first_response, first_tx) = waiters.pop().unwrap();
                        first_tx
                            .send(second_response)
                            .map_err(|_| anyhow::anyhow!("first swapped caller was cancelled"))?;
                        second_tx
                            .send(first_response)
                            .map_err(|_| anyhow::anyhow!("second swapped caller was cancelled"))?;
                    }
                    drop(waiters);
                    rx.await
                        .map_err(|_| anyhow::anyhow!("swap response channel closed"))
                }
                other => anyhow::bail!("unknown lifecycle behavior {other}"),
            }
        }

        async fn subscribe(&self, _topic: &[u8]) -> Result<Self::Sub> {
            Ok(futures::stream::pending())
        }

        async fn publish(&self, _topic: &[u8]) -> Result<Self::Pub> {
            Ok(RpcPublishStub)
        }
    }

    fn lifecycle_client() -> (
        Arc<RpcClientImpl<LocalSigner, LifecycleTransport>>,
        Arc<LifecycleState>,
        Arc<AtomicUsize>,
    ) {
        let (server_sk, server_vk) = generate_signing_keypair();
        let (client_sk, _client_vk) = generate_signing_keypair();
        let mut kem_store = KeyedKemTrustStore::new();
        kem_store.bind(
            server_vk.to_bytes(),
            derive_mesh_kem_recipient(&server_sk).unwrap().public(),
        );
        let pq_sk = crate::node_identity::derive_mesh_mldsa_key(&server_sk);
        let pq_vk = crate::crypto::pq::ml_dsa_vk_from_bytes(
            &crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq_sk),
        )
        .unwrap();
        let mut pq_store = crate::envelope::KeyedPqTrustStore::new();
        pq_store.bind(server_vk.to_bytes(), &pq_vk);
        let state = Arc::new(LifecycleState {
            behavior: AtomicU8::new(LIFECYCLE_SUCCESS),
            recipients: parking_lot::Mutex::new(Vec::new()),
            swap_waiters: Mutex::new(Vec::new()),
            entered: Notify::new(),
        });
        let drops = Arc::new(AtomicUsize::new(0));
        let client = RpcClientImpl::new(
            LocalSigner::new(client_sk),
            LifecycleTransport {
                server_sk,
                state: Arc::clone(&state),
            },
            Some(server_vk),
        )
        .with_request_kem_store(Arc::new(kem_store))
        .with_response_pq_store(Arc::new(pq_store))
        .with_response_secret_drop_probe(Arc::clone(&drops));
        (Arc::new(client), state, drops)
    }

    #[async_trait]
    impl Transport for ForbidsCleartextMock {
        type Sub = RpcPendingStream;
        type Pub = RpcPublishStub;

        fn forbids_cleartext_envelope(&self) -> bool {
            true
        }

        async fn send(&self, _payload: Vec<u8>, _timeout_ms: Option<i32>) -> Result<Vec<u8>> {
            anyhow::bail!("mock transport: send must not be reached in sign_envelope tests")
        }

        async fn subscribe(&self, _topic: &[u8]) -> Result<Self::Sub> {
            anyhow::bail!("mock transport: no subscribe")
        }

        async fn publish(&self, _topic: &[u8]) -> Result<Self::Pub> {
            anyhow::bail!("mock transport: no publish")
        }
    }

    #[async_trait]
    impl Transport for CleartextResponseMock {
        type Sub = RpcPendingStream;
        type Pub = RpcPublishStub;

        fn forbids_cleartext_envelope(&self) -> bool {
            true
        }

        async fn send(&self, _payload: Vec<u8>, _timeout_ms: Option<i32>) -> Result<Vec<u8>> {
            Ok(self.0.clone())
        }

        async fn subscribe(&self, _topic: &[u8]) -> Result<Self::Sub> {
            anyhow::bail!("unused")
        }

        async fn publish(&self, _topic: &[u8]) -> Result<Self::Pub> {
            anyhow::bail!("unused")
        }
    }

    fn response_wire(response: &crate::envelope::ResponseEnvelope) -> Vec<u8> {
        let mut message = capnp::message::Builder::new_default();
        let mut builder = message.init_root::<crate::common_capnp::response_envelope::Builder>();
        response.write_to(&mut builder);
        let mut bytes = Vec::new();
        capnp::serialize::write_message(&mut bytes, &message).unwrap();
        bytes
    }

    /// A carrier that records whether `send` was ever reached, with a
    /// configurable cleartext-forbidding classification. Used by the full
    /// `RpcClient::call()` sentinel tests (ported from #1032, adapted to
    /// #1036's `forbids_cleartext_envelope` model — no WarnOnly / runtime
    /// downgrade). If the fail-closed guard works, a forbidding-carrier client
    /// with no KEM store must error WITHOUT `send` being called.
    struct SentinelTransport {
        sent: Arc<AtomicBool>,
        forbids: bool,
    }

    #[async_trait]
    impl Transport for SentinelTransport {
        type Sub = RpcPendingStream;
        type Pub = RpcPublishStub;

        fn forbids_cleartext_envelope(&self) -> bool {
            self.forbids
        }

        async fn send(&self, _payload: Vec<u8>, _timeout_ms: Option<i32>) -> Result<Vec<u8>> {
            self.sent.store(true, Ordering::SeqCst);
            // Bogus empty reply: response unwrap fails after this, but the tests
            // assert on `sent` to be unambiguous about WHERE the failure occurs.
            Ok(Vec::new())
        }

        async fn subscribe(&self, _topic: &[u8]) -> Result<Self::Sub> {
            Ok(futures::stream::pending())
        }

        async fn publish(&self, _topic: &[u8]) -> Result<Self::Pub> {
            Ok(RpcPublishStub)
        }
    }

    fn signed_envelope_from_bytes(bytes: &[u8]) -> SignedEnvelope {
        let reader = capnp::serialize::read_message(
            &mut std::io::Cursor::new(bytes),
            capnp::message::ReaderOptions::new(),
        )
        .expect("read signed envelope");
        let sr = reader
            .get_root::<crate::common_capnp::signed_envelope::Reader>()
            .expect("signed envelope root");
        SignedEnvelope::read_from(sr).expect("decode signed envelope")
    }

    /// `None` store (what `dial_wasm::dial` passes) MUST fail closed on a
    /// cleartext-forbidding carrier — never a cleartext downgrade.
    #[tokio::test]
    async fn cleartext_forbidding_carrier_without_kem_store_fails_closed() {
        let (_server_sk, server_vk) = generate_signing_keypair();
        let (client_sk, _client_vk) = generate_signing_keypair();

        // Pinned server key present, but NO request KEM store (the `None` arm).
        let rpc = RpcClientImpl::new(
            LocalSigner::new(client_sk),
            ForbidsCleartextMock,
            Some(server_vk),
        );

        let err = rpc
            .sign_envelope(
                1,
                b"payload".to_vec(),
                None,
                None,
                None,
                None,
                Some("test-service"),
            )
            .await
            .expect_err("must fail closed without a #mesh-kem store");
        assert!(
            err.to_string().contains("mesh-kem trust"),
            "expected fail-closed on missing KEM store, got: {err}"
        );
    }

    /// `Some(store)` (what `dial_wasm::dial_with_kem_store` forwards) with an
    /// anchored recipient MUST produce an encrypted envelope — proving the
    /// forwarded store is actually used to seal.
    #[tokio::test]
    async fn cleartext_forbidding_carrier_with_kem_store_encrypts() {
        let (server_sk, server_vk) = generate_signing_keypair();
        let (client_sk, _client_vk) = generate_signing_keypair();

        // Anchor the server's #mesh-kem recipient key (out-of-band binding).
        let mut kem_store = KeyedKemTrustStore::new();
        kem_store.bind(
            server_vk.to_bytes(),
            derive_mesh_kem_recipient(&server_sk).unwrap().public(),
        );

        let rpc = RpcClientImpl::new(
            LocalSigner::new(client_sk),
            ForbidsCleartextMock,
            Some(server_vk),
        )
        .with_request_kem_store(Arc::new(kem_store));

        let (bytes, pending) = rpc
            .sign_envelope(
                1,
                b"payload".to_vec(),
                None,
                None,
                None,
                None,
                Some("test-service"),
            )
            .await
            .expect("forwarded store must seal the envelope");
        assert!(
            pending.is_some(),
            "network call must retain a response secret"
        );
        let signed = signed_envelope_from_bytes(&bytes);
        assert!(
            signed.is_encrypted(),
            "forwarded #mesh-kem store must yield an encrypted_envelope"
        );
        // The redacted outer payload must not carry the cleartext.
        assert!(
            !bytes.windows(b"payload".len()).any(|w| w == b"payload"),
            "cleartext payload must not appear on the wire"
        );
    }

    #[tokio::test]
    async fn hostile_cleartext_response_is_rejected_before_payload_use() {
        let (server_sk, server_vk) = generate_signing_keypair();
        let (client_sk, _) = generate_signing_keypair();
        let server_pq_sk = crate::node_identity::derive_mesh_mldsa_key(&server_sk);
        let clear = crate::envelope::ResponseEnvelope::new_signed_hybrid(
            1,
            b"cleartext-response-sentinel".to_vec(),
            &server_sk,
            &server_pq_sk,
        );

        let mut kem_store = KeyedKemTrustStore::new();
        kem_store.bind(
            server_vk.to_bytes(),
            derive_mesh_kem_recipient(&server_sk).unwrap().public(),
        );
        let server_pq_vk = crate::crypto::pq::ml_dsa_vk_from_bytes(
            &crate::crypto::pq::ml_dsa_sk_to_vk_bytes(&server_pq_sk),
        )
        .unwrap();
        let mut pq_store = crate::envelope::KeyedPqTrustStore::new();
        pq_store.bind(server_vk.to_bytes(), &server_pq_vk);

        let rpc = RpcClientImpl::new(
            LocalSigner::new(client_sk),
            CleartextResponseMock(response_wire(&clear)),
            Some(server_vk),
        )
        .with_request_kem_store(Arc::new(kem_store))
        .with_response_pq_store(Arc::new(pq_store));

        let err = rpc
            .call_for_service("test-service", b"request".to_vec())
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("cleartext response rejected before payload use"),
            "unexpected error: {err}"
        );
    }

    /// Full-`call()` sentinel (ported from #1032): a forbidding carrier with no
    /// KEM store must be refused BEFORE any byte reaches `Transport::send()`.
    /// This proves the guard is on the real send path, not merely on the
    /// `sign_envelope` helper in isolation.
    #[tokio::test]
    async fn call_over_forbidding_carrier_without_kem_store_refused_before_send() {
        let (_server_sk, server_vk) = generate_signing_keypair();
        let (client_sk, _client_vk) = generate_signing_keypair();
        let sent = Arc::new(AtomicBool::new(false));
        let rpc = RpcClientImpl::new(
            LocalSigner::new(client_sk),
            SentinelTransport {
                sent: sent.clone(),
                forbids: true,
            },
            Some(server_vk),
        );

        let result = rpc.call(b"payload".to_vec()).await;
        assert!(
            result.is_err(),
            "forbidding carrier + no KEM store must be refused"
        );
        assert!(
            !sent.load(Ordering::SeqCst),
            "refusal MUST happen before any byte reaches the transport"
        );
    }

    /// The mirror: a trusted carrier (forbids == false) is NOT blocked — the
    /// same store-less cleartext send proceeds to the transport. Proves the ban
    /// is not over-broadened onto trusted inproc/UDS paths. (Response unwrap
    /// fails on the bogus empty reply, but that is AFTER `send`.)
    #[tokio::test]
    async fn call_over_trusted_carrier_reaches_send() {
        let (_server_sk, server_vk) = generate_signing_keypair();
        let (client_sk, _client_vk) = generate_signing_keypair();
        let sent = Arc::new(AtomicBool::new(false));
        let rpc = RpcClientImpl::new(
            LocalSigner::new(client_sk),
            SentinelTransport {
                sent: sent.clone(),
                forbids: false,
            },
            Some(server_vk),
        );

        let _ = rpc.call(b"payload".to_vec()).await;
        assert!(
            sent.load(Ordering::SeqCst),
            "trusted-carrier cleartext must NOT be blocked — send must be reached"
        );
    }

    #[tokio::test]
    async fn concurrent_calls_use_distinct_recipients_and_swapped_live_responses_fail() {
        let (client, state, _drops) = lifecycle_client();
        let control = client
            .call_for_service("service-a", b"control".to_vec())
            .await
            .expect("unmutated control must succeed before response swapping");
        assert_eq!(control, b"control");

        state.behavior.store(LIFECYCLE_SWAP, Ordering::SeqCst);
        let first = {
            let client = Arc::clone(&client);
            tokio::spawn(async move {
                client
                    .call_for_service("service-a", b"first".to_vec())
                    .await
            })
        };
        let second = {
            let client = Arc::clone(&client);
            tokio::spawn(async move {
                client
                    .call_for_service("service-a", b"second".to_vec())
                    .await
            })
        };
        assert!(
            first.await.unwrap().is_err(),
            "first swapped response must fail"
        );
        assert!(
            second.await.unwrap().is_err(),
            "second swapped response must fail"
        );

        let recipients = state.recipients.lock();
        assert_eq!(recipients.len(), 3);
        assert_ne!(
            recipients[1], recipients[2],
            "concurrent calls must carry distinct response recipients"
        );
    }

    #[tokio::test]
    async fn cancellation_drops_one_shot_response_state() {
        let (client, state, drops) = lifecycle_client();
        client
            .call_for_service("service-a", b"control".to_vec())
            .await
            .expect("unmutated control must succeed before cancellation");
        assert_eq!(drops.load(Ordering::SeqCst), 1);

        state.behavior.store(LIFECYCLE_BLOCK, Ordering::SeqCst);
        let entered = state.entered.notified();
        let call = {
            let client = Arc::clone(&client);
            tokio::spawn(async move {
                client
                    .call_for_service("service-a", b"cancelled".to_vec())
                    .await
            })
        };
        entered.await;
        call.abort();
        assert!(call.await.unwrap_err().is_cancelled());
        assert_eq!(
            drops.load(Ordering::SeqCst),
            2,
            "cancelling a live send must drop its response secret"
        );
    }

    #[tokio::test]
    async fn send_failure_drops_state_and_caller_retry_uses_fresh_recipient() {
        let (client, state, drops) = lifecycle_client();
        client
            .call_for_service("service-a", b"control".to_vec())
            .await
            .expect("unmutated control must succeed before send failure");

        state.behavior.store(LIFECYCLE_FAIL_SEND, Ordering::SeqCst);
        assert!(client
            .call_for_service("service-a", b"retry-me".to_vec())
            .await
            .is_err());
        assert_eq!(drops.load(Ordering::SeqCst), 2);

        state.behavior.store(LIFECYCLE_SUCCESS, Ordering::SeqCst);
        let retried = client
            .call_for_service("service-a", b"retry-me".to_vec())
            .await
            .expect("caller retry control succeeds");
        assert_eq!(retried, b"retry-me");
        assert_eq!(drops.load(Ordering::SeqCst), 3);

        let recipients = state.recipients.lock();
        assert_eq!(recipients.len(), 3);
        assert_ne!(
            recipients[1], recipients[2],
            "caller retry must generate a fresh response recipient"
        );
    }
}
