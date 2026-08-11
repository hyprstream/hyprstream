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

    // Proof-CWT structural parse (v16 §5.2 pipeline: parse canonical COSE
    // and proof payload under bounds). This runs the bounded parser which
    // validates the profile's structural rules (typ, hs_domain, crit,
    // signature plan, claims, key set) but does NOT verify cryptographic
    // signatures. Signature verification and replay admission run after
    // policy evaluation, immediately before handler entry.
    let parsed_proof = if let Some(proof_cwt) = &ctx.envelope_proof_cwt {
        let proof = crate::proof::parser::ParsedProof::parse(proof_cwt)
            .with_context(|| format!("{} proof-CWT parse failed", service.name()))?;

        // CRITICAL: only request proofs are valid in request dispatch.
        if proof.kind != crate::proof::ProofKind::Request {
            warn!(
                "{} rejected {} proof in request dispatch",
                service.name(),
                match proof.kind {
                    crate::proof::ProofKind::Response => "response",
                    crate::proof::ProofKind::Request => "request",
                }
            );
            anyhow::bail!("proof kind mismatch: only request proofs accepted in dispatch");
        }

        // The proof's aud MUST match the service domain.
        if proof.claims.aud != actual_service_domain {
            warn!(
                "{} proof aud mismatch: '{}' vs '{}'",
                service.name(), proof.claims.aud, actual_service_domain
            );
            anyhow::bail!("proof aud mismatch");
        }

        // Freshness: per-disposition bounds against verifier clock.
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let skew_tolerance_secs: u64 = 30;
        let max_lifetime_secs: u64 = match proof.disposition {
            crate::proof::ProofDisposition::Authenticated => 300,  // 5 min
            crate::proof::ProofDisposition::Unattributed => 30,   // seconds-scale
        };
        if proof.claims.iat.abs_diff(now_secs) > skew_tolerance_secs {
            anyhow::bail!("proof freshness: iat outside {skew_tolerance_secs}s tolerance");
        }
        if proof.claims.exp <= now_secs {
            anyhow::bail!("proof freshness: exp expired");
        }
        if proof.claims.exp > now_secs + max_lifetime_secs {
            anyhow::bail!(
                "proof freshness: exp exceeds {max_lifetime_secs}s max lifetime for {:?}",
                proof.disposition
            );
        }

        debug!(
            "{} proof-CWT parsed (not yet crypto-verified): disposition={:?} request_id={}",
            service.name(),
            proof.disposition,
            hex::encode(proof.claims.request_id)
        );
        Some(proof)
    } else {
        None
    };
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

    // After carrier recovery: verify proof body bytes match the ONE decoded
    // request body that feeds both PEP and handler (v16 §5.1 invariant).
    if let Some(ref proof) = parsed_proof {
        if proof.claims.capnp_request_bytes != payload {
            warn!(
                "{} proof body mismatch after carrier recovery: {} vs {} bytes",
                service.name(),
                proof.claims.capnp_request_bytes.len(),
                payload.len()
            );
            anyhow::bail!("proof body bytes do not match decoded request body");
        }
    }

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
            ResponseEnvelope::new_signed_encrypted(
                request_id,
                payload,
                signing_key,
                &response_pq_key,
                recipient,
                request_iat,
                &request_nonce,
                actual_service_domain,
            )
            .map_err(Into::into)
        } else {
            ResponseEnvelope::new_signed_with_policy(
                request_id,
                payload,
                signing_key,
                Some(&response_pq_key),
                crate::crypto::CryptoPolicy::Hybrid,
            )
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

    // 2b. Mandatory native-MAC dispatch PEP (epic #1267 T3, #1268).
    //
    // This is the mandatory, unavoidable gate between claims verification
    // and handler invocation — the RPC-plane analogue of the 9P
    // `NinePAccessDecider`. It composes the S1 two-input clearance derivation
    // (Claims × VerifiedKeyMaterial via `EnvelopeContext::security_context`)
    // with a trusted object-label resolver and the intrinsic `can_access`
    // lattice floor. Once installed, the PEP fails closed: missing clearance,
    // missing label, or lattice-floor deny ⇒ handler is never called.
    //
    // **Activation gate (#1267):** identity-aware subject selection remains
    // operator-gated, but mediation is mandatory at rest. Until a node installs
    // a PEP process-globally via `install_mac_dispatch_pep`, this gate denies
    // with `NoPepInstalled`; after installation its decision is authoritative.
    //
    // Streaming continuations: the continuation produced by a permitted
    // handler inherits this dispatch-time Permit. Explicit re-check of
    // long-running continuations against revoked authority is a DEFERRED
    // follow-up (#1267 scope expansion — `StaleAuthority` variant is reserved
    // for that future gate, not used today).
    let mac_decision = crate::auth::mac::check_dispatch_mac(
        &ctx,
        actual_service_domain,
        ctx.browser_method_discriminator,
    );
    if let crate::auth::mac::MacDecision::Deny(reason) = mac_decision {
        let mac_resource = match ctx.browser_method_discriminator {
            Some(method) => format!("{actual_service_domain}:method:{method}"),
            None => format!("{actual_service_domain}:*"),
        };
        // S7/#1274: a mandatory-MAC rejection is an authorization decision,
        // not merely a diagnostic warning. Emit it on the unified MAC audit
        // target before returning the signed deny response so the RPC and 9P
        // planes have the same fail-closed audit contract.
        warn!(
            target: "hyprstream.mac.audit",
            decision = "deny",
            subject = %ctx.subject(),
            resource = %mac_resource,
            action = "rpc-dispatch",
            request_id,
            plane = "rpc",
            reason = ?reason,
            "authorization decision"
        );
        warn!(
            "{} MAC dispatch PEP denied {} (id={}, reason={:?})",
            service.name(),
            ctx.subject(),
            request_id,
            reason,
        );
        let error_payload =
            service.build_error_payload(request_id, &format!("MAC deny: {reason:?}"));
        let signed_response = sign_response(error_payload)?;

        let mut message = Builder::new_default();
        let mut builder = message.init_root::<crate::common_capnp::response_envelope::Builder>();
        signed_response.write_to(&mut builder);

        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        return Ok(bytes);
    }

    // 2c. Proof-CWT verification gate (v16 §5.2 pipeline: after policy,
    // before handler). Every required gate MUST pass before handler entry.
    // No gate is skipped — unimplemented checks deny fail-closed.
    if let Some(ref proof) = parsed_proof {
        // Gate 1: COSE signature verification — every required component
        // MUST cryptographically verify. ML-DSA-65 components deny until
        // verification is wired (not silently skipped).
        let cnf_key = ctx.authenticated_signer_key();
        crate::proof::verify::verify_proof_signatures(proof, cnf_key.as_ref())
            .with_context(|| format!("{} proof signature verification failed", service.name()))?;

        // Gate 2: Credential hash binding (authenticated proofs only).
        // Missing credential MUST deny — no skip/downgrade.
        if proof.disposition == crate::proof::ProofDisposition::Authenticated {
            let expected_hash = proof.claims.credential_hash
                .ok_or_else(|| anyhow::anyhow!("authenticated proof missing credential_hash"))?;
            let jwt = ctx.jwt_token()
                .ok_or_else(|| anyhow::anyhow!("authenticated proof: no credential (JWT) resolved from claims"))?;
            use sha2::{Digest, Sha256};
            let actual_hash = Sha256::digest(jwt.as_bytes());
            if actual_hash.as_slice() != expected_hash {
                warn!("{} proof credential_hash mismatch", service.name());
                anyhow::bail!("proof credential hash does not match presented credential");
            }
        }

        // Gate 3: Challenge validation (unattributed proofs only).
        // validate() atomically returns the matched challenge's accept_until
        // so we can compute replay expiry without a TOCTOU race.
        let challenge_accept_until: Option<u64> = if proof.disposition == crate::proof::ProofDisposition::Unattributed {
            let challenge = proof.claims.nonce.as_ref()
                .ok_or_else(|| anyhow::anyhow!("unattributed proof missing Nonce claim"))?;
            let now_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            match crate::proof::admission::global_challenge_manager() {
                Some(mgr) => {
                    let accept_until = mgr.validate(challenge, now_secs)
                        .ok_or_else(|| {
                            warn!("{} unattributed proof challenge invalid/expired", service.name());
                            anyhow::anyhow!("proof challenge validation failed")
                        })?;
                    Some(accept_until)
                }
                None => {
                    warn!("{} no challenge manager installed; denying unattributed proof", service.name());
                    anyhow::bail!("unattributed proof requires a challenge manager to be installed");
                }
            }
        } else {
            None
        };

        // Gate 4: Replay admission using the process-global ProofReplayStore.
        // No auto-install: the store must be explicitly installed at startup.
        // If absent, deny fail-closed.
        let replay_store = crate::proof::admission::global_proof_replay_store()
            .ok_or_else(|| {
                anyhow::anyhow!("no ProofReplayStore installed; install via set_global_proof_replay_store at startup")
            })?;

        // Compute replay expiry using the atomically-returned challenge
        // accept_until (no TOCTOU race).
        let replay_expiry = match challenge_accept_until {
            Some(au) => proof.claims.exp.min(au),
            None => proof.claims.exp,
        };

        let replay_key = match proof.disposition {
            crate::proof::ProofDisposition::Unattributed => {
                let thumbprint = proof.unattributed_replay_thumbprint()
                    .ok_or_else(|| anyhow::anyhow!("cannot compute unattributed thumbprint"))?;
                crate::proof::admission::ProofReplayKey {
                    signer_thumbprint: thumbprint,
                    request_id: proof.claims.request_id,
                }
            }
            crate::proof::ProofDisposition::Authenticated => {
                let cnf_key = ctx.authenticated_signer_key()
                    .ok_or_else(|| anyhow::anyhow!("authenticated proof: no cnf signer key"))?;
                use sha2::{Digest, Sha256};
                crate::proof::admission::ProofReplayKey {
                    signer_thumbprint: Sha256::digest(cnf_key.as_bytes()).into(),
                    request_id: proof.claims.request_id,
                }
            }
        };

        match replay_store.check_and_insert(
            proof.disposition,
            &replay_key,
            replay_expiry,
        ) {
            crate::proof::admission::ProofAdmissionResult::Admitted => {
                debug!("{} proof replay admission: admitted", service.name());
            }
            crate::proof::admission::ProofAdmissionResult::Replayed => {
                warn!("{} proof replay detected", service.name());
                anyhow::bail!("proof replay detected");
            }
            crate::proof::admission::ProofAdmissionResult::Failed => {
                warn!("{} proof replay store at capacity (fail-closed)", service.name());
                anyhow::bail!("proof replay store at capacity");
            }
        }
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
