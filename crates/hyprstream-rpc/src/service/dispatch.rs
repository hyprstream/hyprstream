//! Transport-neutral request dispatch core (#148).
//!
//! `process_request` is the single envelope-verify → JWT/DPoP → Casbin →
//! `handle_request` → signed-response pipeline shared by every front-end:
//! the ZMQ `RequestLoop`, the WebTransport server, and the generic RPC plane's
//! `LocalServiceBridge`. It contains no transport-specific code (no ZMQ, no
//! quinn) — extracted out of `transport/zmtp_quic.rs` so the ZMQ-specific
//! remainder of that file can be deleted in #138 without touching this.

use anyhow::{Context, Result};
use capnp::message::Builder;
use capnp::serialize;
use tracing::{debug, error, warn};

use crate::envelope::ResponseEnvelope;
use crate::ToCapnp;

/// Envelope signer verification mode (re-exported from `crate::envelope`).
///
/// - **FixedSigner**: internal service-to-service — envelope signer must match
///   a known `server_pubkey` for mutual authentication. Peers pre-share keys.
/// - **AnySigner**: external clients (e.g. browser over WebTransport) — any
///   valid signer accepted; transport TLS provides peer authentication.
pub use crate::envelope::EnvelopeVerification;

/// Process a request through the full envelope verification pipeline.
///
/// Unified handler for all transport front-ends. The only difference between
/// them is envelope signer verification, controlled by `verification`.
///
/// # Pipeline
///
/// 1. Unwrap `SignedEnvelope` and verify the signature (mode-dependent), under
///    the process-global verify policy + kid-anchored PQ trust store.
/// 2. Verify JWT claims (`sub`, `exp`, `aud`, `scope`, downgrade protection).
/// 3. Dispatch to `service.handle_request()` with a verified `EnvelopeContext`.
/// 4. Sign the response with the server's `signing_key`.
///
/// # Streaming
///
/// A streaming handler returns a `Continuation` (the server-side streaming
/// response that runs after the reply). As of #186 that task is spawned
/// **here**, via [`crate::streaming::spawn_streaming_response`], rather than
/// handed back to the transport front-end — so every front-end (ZMQ
/// `RequestLoop`, WebTransport server, generic-plane `LocalServiceBridge`) is
/// uniform "bytes in → bytes out" and the generic plane no longer has to spawn
/// (or, worse, reject) continuations itself. **Invariant:** `process_request`
/// must therefore run on a `tokio::task::LocalSet` (it already did —
/// `RequestService` is `?Send`); the spawned task is `?Send`.
///
/// # Returns
///
/// * `Ok(response_bytes)` - Signed response. Any streaming pump has already been
///   spawned onto the current `LocalSet`.
/// * `Err(e)` - Processing error (already logged)
pub async fn process_request<S>(
    raw_bytes: &[u8],
    service: &S,
    verification: EnvelopeVerification<'_>,
    signing_key: &ed25519_dalek::SigningKey,
    nonce_cache: &crate::envelope::InMemoryNonceCache,
    carrier: crate::transport::carrier::CarrierContext,
) -> Result<Vec<u8>>
where
    S: crate::service::RequestService,
{
    // 1. Unwrap, verify, and optionally decrypt the SignedEnvelope.
    //    The verify policy + kid-anchored PQ trust store come from the
    //    process-global verify configuration installed at startup (Hybrid
    //    ENFORCED in the daemon). This closes the prior fail-open where the
    //    site hardcoded Classical / no PQ store.
    let pq_store_holder = crate::envelope::global_pq_store();
    let base = match verification {
        EnvelopeVerification::FixedSigner(pubkey) => {
            crate::envelope::UnwrapOptions::fixed_signer(pubkey, nonce_cache)
        }
        EnvelopeVerification::AnySigner => crate::envelope::UnwrapOptions::any_signer(nonce_cache),
    }
    .with_decryption_key(signing_key)
    .require_encrypted(carrier.forbids_cleartext_envelope());
    let opts = crate::envelope::apply_global_verify_config(base, &pq_store_holder);

    let (mut ctx, payload) = match crate::envelope::unwrap_envelope(raw_bytes, &opts) {
        Ok(result) => result,
        Err(e) => {
            warn!("{} envelope verification failed: {}", service.name(), e);
            // Never sign a response to unauthenticated input. The transport
            // boundary treats this error as a silent stream reset/drop on
            // untrusted carriers, preventing a signing oracle/amplifier.
            return Err(e).with_context(|| format!("{} envelope admission failed", service.name()));
        }
    };

    let request_id = ctx.request_id;
    let actual_service_domain = service.name();
    crate::envelope::validate_service_domain(actual_service_domain).with_context(|| {
        format!("service exposes non-canonical domain '{actual_service_domain}'")
    })?;
    match ctx.service_domain.as_deref() {
        Some(expected) if expected != actual_service_domain => {
            anyhow::bail!(
                "authenticated request service domain '{expected}' does not match dispatcher '{actual_service_domain}'"
            );
        }
        None if carrier.forbids_cleartext_envelope() => {
            anyhow::bail!(
                "authenticated network request omitted serviceDomain; dropping without response"
            );
        }
        _ => {}
    }
    let transcript_policy = if carrier.requires_browser_provisioning() {
        crate::browser_provisioning::BrowserTranscriptPolicy::Required {
            request_id,
            service_name: actual_service_domain,
            carrier_profile:
                crate::browser_provisioning::BrowserCarrierProfile::OwnedHybridWebTransport,
        }
    } else {
        crate::browser_provisioning::BrowserTranscriptPolicy::NotBrowserCarrier
    };
    let (browser_transcript, payload) =
        crate::browser_provisioning::recover_request_payload(&payload, transcript_policy)?;
    ctx.browser_method_discriminator = browser_transcript
        .as_ref()
        .map(|transcript| transcript.method_discriminator);
    if carrier.forbids_cleartext_envelope() && ctx.response_kem_recipient.is_none() {
        anyhow::bail!(
            "authenticated network request omitted responseKemRecipient; dropping without response"
        );
    }
    // Refuse before dispatch if the service cannot emit the mandatory pinned
    // hybrid response suite. Missing key material is never a signal to
    // construct a classical response. Deliberately checked only after envelope
    // authentication: the default `pq_signing_key()` derives an ML-DSA key per
    // call, and unauthenticated input must not be able to trigger that work.
    let response_pq_key = service.pq_signing_key().ok_or_else(|| {
        anyhow::anyhow!("service has no ML-DSA-65 response signing key (mandatory Hybrid suite)")
    })?;
    if carrier.requires_browser_provisioning() {
        let binding = &browser_transcript
            .as_ref()
            .ok_or_else(|| {
                anyhow::anyhow!("authenticated WebTransport request omitted browser binding")
            })?
            .binding;
        anyhow::ensure!(
            binding.service_name == actual_service_domain,
            "browser provisioning service '{}' does not match dispatcher '{}'",
            binding.service_name,
            actual_service_domain
        );
        anyhow::ensure!(
            binding.capability == "hyprstream-rpc/1"
                && binding.scope == actual_service_domain
                && binding.carrier_profile
                    == crate::browser_provisioning::BrowserCarrierProfile::OwnedHybridWebTransport,
            "browser provisioning capability/scope/carrier misclassification"
        );
        let verifier = crate::envelope::global_browser_currentness_verifier().ok_or_else(|| {
            anyhow::anyhow!(
                "checkpoint-backed browser currentness verifier is not installed; dropping without response"
            )
        })?;
        verifier
            .ensure_current(binding)
            .await
            .context("browser accepted-current evidence rejected at dispatch")?;
    }
    let response_recipient = ctx.response_kem_recipient.clone();
    let request_iat = ctx.request_iat;
    let request_nonce = ctx.request_nonce;
    let sign_response = |payload: Vec<u8>| -> Result<ResponseEnvelope> {
        if carrier.forbids_cleartext_envelope() {
            let recipient = response_recipient
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing authenticated response recipient"))?;
            let pq_key = response_pq_key
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing response ML-DSA-65 signing key"))?;
            ResponseEnvelope::new_signed_encrypted(
                request_id,
                payload,
                signing_key,
                pq_key,
                recipient,
                request_iat,
                &request_nonce,
                actual_service_domain,
            )
            .map_err(Into::into)
        } else {
            Ok(ResponseEnvelope::new_signed_with_policy(
                request_id,
                payload,
                signing_key,
                response_pq_key.as_ref(),
                response_policy,
            ))
        }
    };
    debug!(
        "{} verified request from {} (id={})",
        service.name(),
        ctx.subject(),
        request_id
    );

    // 2. Verify claims (E2E JWT, downgrade protection)
    if let Err(e) = service.verify_claims(&mut ctx).await {
        warn!(
            "{} claims verification failed for {} (id={}): {}",
            service.name(),
            ctx.subject(),
            request_id,
            e
        );
        let error_payload = service.build_error_payload(request_id, &e.to_string());
        let signed_response = sign_response(error_payload)?;

        let mut message = Builder::new_default();
        let mut builder = message.init_root::<crate::common_capnp::response_envelope::Builder>();
        signed_response.write_to(&mut builder);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        return Ok(bytes);
    }

    // 3. Handle request
    let (response_payload, continuation) = match service.handle_request(&ctx, &payload).await {
        Ok((resp, cont)) => (resp, cont),
        Err(e) => {
            error!("{} request handling error: {}", service.name(), e);
            (
                service.build_error_payload(request_id, &e.to_string()),
                None,
            )
        }
    };

    // 4. Sign and serialize response
    let signed_response = sign_response(response_payload)?;

    let mut message = Builder::new_default();
    let mut builder = message.init_root::<crate::common_capnp::response_envelope::Builder>();
    signed_response.write_to(&mut builder);

    let mut bytes = Vec::new();
    serialize::write_message(&mut bytes, &message)?;

    // 5. Spawn the server-side streaming response (if any) onto the current
    //    LocalSet, so the reply is all the transport front-end has to deal with
    //    (#186). Bounded by a per-service admission permit; see
    //    spawn_streaming_response.
    if let Some(cont) = continuation {
        crate::streaming::spawn_streaming_response(service.name(), cont);
    }

    Ok(bytes)
}
