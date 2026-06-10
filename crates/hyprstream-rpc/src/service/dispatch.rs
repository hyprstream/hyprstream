//! Transport-neutral request dispatch core (#148).
//!
//! `process_request` is the single envelope-verify → JWT/DPoP → Casbin →
//! `handle_request` → signed-response pipeline shared by every front-end:
//! the ZMQ `RequestLoop`, the WebTransport server, and the generic RPC plane's
//! `LocalServiceBridge`. It contains no transport-specific code (no ZMQ, no
//! quinn) — extracted out of `transport/zmtp_quic.rs` so the ZMQ-specific
//! remainder of that file can be deleted in #138 without touching this.

use anyhow::Result;
use tracing::{debug, error, warn};

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
/// # Returns
///
/// * `Ok((response_bytes, continuation))` - Signed response and optional continuation
/// * `Err(e)` - Processing error (already logged)
pub async fn process_request<S>(
    raw_bytes: &[u8],
    service: &S,
    verification: EnvelopeVerification<'_>,
    signing_key: &ed25519_dalek::SigningKey,
    nonce_cache: &crate::envelope::InMemoryNonceCache,
) -> Result<(Vec<u8>, Option<crate::service::Continuation>)>
where
    S: crate::service::RequestService,
{
    use crate::ToCapnp;
    use crate::envelope::ResponseEnvelope;
    use capnp::message::Builder;
    use capnp::serialize;

    // 1. Unwrap, verify, and optionally decrypt the SignedEnvelope.
    //    The verify policy + kid-anchored PQ trust store come from the
    //    process-global verify configuration installed at startup (Hybrid
    //    ENFORCED in the daemon). This closes the prior fail-open where the
    //    site hardcoded Classical / no PQ store.
    let pq_store_holder = crate::envelope::global_pq_store();
    let base = match verification {
        EnvelopeVerification::FixedSigner(pubkey) =>
            crate::envelope::UnwrapOptions::fixed_signer(pubkey, nonce_cache),
        EnvelopeVerification::AnySigner =>
            crate::envelope::UnwrapOptions::any_signer(nonce_cache),
    }.with_decryption_key(signing_key);
    let opts =
        crate::envelope::apply_global_verify_config(base, &pq_store_holder);

    let (mut ctx, payload) = match crate::envelope::unwrap_envelope(raw_bytes, &opts) {
        Ok(result) => result,
        Err(e) => {
            warn!("{} envelope verification failed: {}", service.name(), e);
            // Build error response with request_id=0 (envelope is invalid)
            let error_payload = service.build_error_payload(0, &format!("envelope verification failed: {}", e));
            let signed_response = ResponseEnvelope::new_signed(0, error_payload, signing_key);

            let mut message = Builder::new_default();
            let mut builder = message.init_root::<crate::common_capnp::response_envelope::Builder>();
            signed_response.write_to(&mut builder);

            let mut bytes = Vec::new();
            serialize::write_message(&mut bytes, &message)?;
            return Ok((bytes, None));
        }
    };

    let request_id = ctx.request_id;
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
            service.name(), ctx.subject(), request_id, e
        );
        let error_payload = service.build_error_payload(request_id, &e.to_string());
        let signed_response = ResponseEnvelope::new_signed(request_id, error_payload, signing_key);

        let mut message = Builder::new_default();
        let mut builder = message.init_root::<crate::common_capnp::response_envelope::Builder>();
        signed_response.write_to(&mut builder);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        return Ok((bytes, None));
    }

    // 3. Handle request
    let (response_payload, continuation) = match service.handle_request(&ctx, &payload).await {
        Ok((resp, cont)) => (resp, cont),
        Err(e) => {
            error!("{} request handling error: {}", service.name(), e);
            (service.build_error_payload(request_id, &e.to_string()), None)
        }
    };

    // 4. Sign and serialize response
    let signed_response = ResponseEnvelope::new_signed(request_id, response_payload, signing_key);

    let mut message = Builder::new_default();
    let mut builder = message.init_root::<crate::common_capnp::response_envelope::Builder>();
    signed_response.write_to(&mut builder);

    let mut bytes = Vec::new();
    serialize::write_message(&mut bytes, &message)?;

    Ok((bytes, continuation))
}
