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

/// The one externally visible pre-handler denial (§14.2 `DispatchDenied`).
///
/// Every pre-handler denial carries exactly this text, so the external surface
/// is uniform and the internal cause is confined to the operator's log. It is
/// paired with the server's current challenge in the response's fixed slot,
/// which is likewise attached to every denial regardless of cause.
pub const DISPATCH_DENIED: &str = "dispatch denied";

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
    let sign_response_with_challenge = |payload: Vec<u8>,
                                        server_challenge: Option<Vec<u8>>|
     -> Result<ResponseEnvelope> {
        if carrier.forbids_cleartext_envelope() {
            let recipient = response_recipient
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing authenticated response recipient"))?;
            ResponseEnvelope::new_signed_encrypted_with_challenge(
                request_id,
                payload,
                server_challenge,
                signing_key,
                &response_pq_key,
                recipient,
                request_iat,
                &request_nonce,
                actual_service_domain,
            )
            .map_err(Into::into)
        } else {
            ResponseEnvelope::new_signed_with_challenge(
                request_id,
                payload,
                server_challenge,
                signing_key,
                Some(&response_pq_key),
                crate::crypto::CryptoPolicy::Hybrid,
            )
        }
    };
    let sign_response = |payload: Vec<u8>| sign_response_with_challenge(payload, None);

    /// Serialize a signed response envelope to wire bytes.
    fn serialize_response(signed: &ResponseEnvelope) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        let mut builder = message.init_root::<crate::common_capnp::response_envelope::Builder>();
        signed.write_to(&mut builder);
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    // Every pre-handler denial goes out through this one path (§14.2): the
    // same shape, and always carrying the server's current challenge in the
    // fixed response slot (§4.7). Because every denial carries one, its
    // presence reveals nothing about the internal cause, and a client that
    // has no challenge yet obtains a usable one from its first denial and
    // retries exactly once with a fresh request_id.
    //
    // When rotation cannot produce a usable challenge the denial still goes
    // out — it simply advertises none, which is service refusal rather than
    // an unusable challenge the client would burn its one retry on.
    let dispatch_denied = |cause: &str| -> Result<Vec<u8>> {
        // The cause is for the operator's log, never for the payload. Unknown
        // service/method, malformed body, invalid credential, missing session,
        // revocation, replay, signature-threshold failure, resource limit, and
        // policy absence are externally indistinguishable (§14.2): no response
        // reveals whether a credential ID, session, subject, signer, or label
        // exists.
        debug!(
            "{} pre-handler denial (id={request_id}): {cause}",
            service.name()
        );
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let challenge = crate::proof::admission::global_challenge_manager()
            .and_then(|mgr| mgr.current_or_rotate(now_secs))
            .map(|c| c.value);
        let payload = service.build_error_payload(request_id, DISPATCH_DENIED);
        let signed = sign_response_with_challenge(payload, challenge)?;
        serialize_response(&signed)
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
        return dispatch_denied(&format!("claims verification: {e}"));
    }

    // Proof-CWT structural parse and profile gates (§5.2: parse canonical COSE
    // and proof payload under bounds, then bind the service coordinate and
    // freshness). Cryptographic verification and replay admission follow.
    //
    // Deliberately placed after the denial surface exists: a structural,
    // audience, freshness, or body-binding failure is a pre-handler denial
    // like any other, and must leave through the same uniform response rather
    // than dropping the connection and telling the client nothing.
    let parsed_proof = {
        let parsed = (|| -> Result<Option<crate::proof::parser::ParsedProof>> {
        // Proof-CWT structural parse (v16 §5.2 pipeline: parse canonical COSE
        // and proof payload under bounds). This runs the bounded parser which
        // validates the profile's structural rules (typ, hs_domain, crit,
        // signature plan, claims, key set) but does NOT verify cryptographic
        // signatures. Signature verification and replay admission run after
        // policy evaluation, immediately before handler entry.
        let parsed = if let Some(proof_cwt) = &ctx.envelope_proof_cwt {
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
        // After carrier recovery: verify proof body bytes match the ONE decoded
        // request body that feeds both PEP and handler (v16 §5.1 invariant).
        if let Some(ref proof) = parsed {
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
            Ok(parsed)
        })();
        match parsed {
            Ok(parsed) => parsed,
            Err(e) => {
                warn!(
                    "{} proof admission denied (id={}): {e:#}",
                    service.name(),
                    request_id
                );
                return dispatch_denied(&format!("proof gate: {e:#}"));
            }
        }
    };

    // 2c. Proof authority (v16 §5.2 pipeline step 3). The credential was
    // parsed and verified above (step 2); this cryptographically verifies the
    // signer entries and the credential hash/cnf binding BEFORE the decoded
    // leaf's generated policy and before dispatch MAC, so no unverified proof
    // claim can influence authorization.
    //
    // Denials leave through the one uniform DispatchDenied surface, so a proof
    // denial is externally indistinguishable from any other pre-handler denial
    // and still hands the client a usable challenge to retry with.
    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let verified_proof = match parsed_proof.as_ref() {
        None => None,
        Some(proof) => {
            let authority = (|| -> Result<crate::proof::verify::VerifiedProof> {
                // Every required component MUST cryptographically verify under
                // the key its own enrolled signer-suite record pins.
                // Verification also yields the replay namespace this proof is
                // admitted under; it is never derived from wire material.
                let cnf_key = ctx.authenticated_signer_key();
                let verified = crate::proof::verify::verify_proof_signatures(
                    proof,
                    cnf_key.as_ref(),
                    crate::proof::enrollment::global_enrollment_resolver(),
                    now_secs,
                )
                .with_context(|| {
                    format!("{} proof signature verification failed", service.name())
                })?;

                // Credential hash binding. Presenting a credential never
                // leaves the proof in the unattributed branch (§4.4): hash
                // absence is permitted only when no credential is presented,
                // so a credential alongside an unattributed proof denies
                // rather than silently dropping to system-low.
                if proof.disposition == crate::proof::ProofDisposition::Unattributed
                    && ctx.jwt_token().is_some()
                {
                    anyhow::bail!("credential presented with a proof carrying no credential_hash");
                }
                // Missing credential MUST deny — no skip/downgrade.
                if proof.disposition == crate::proof::ProofDisposition::Authenticated {
                    let expected_hash = proof.claims.credential_hash.ok_or_else(|| {
                        anyhow::anyhow!("authenticated proof missing credential_hash")
                    })?;
                    let jwt = ctx.jwt_token().ok_or_else(|| {
                        anyhow::anyhow!(
                            "authenticated proof: no credential (JWT) resolved from claims"
                        )
                    })?;
                    use sha2::{Digest, Sha256};
                    let actual_hash = Sha256::digest(jwt.as_bytes());
                    if actual_hash.as_slice() != expected_hash {
                        anyhow::bail!("proof credential hash does not match presented credential");
                    }

                    // A proof must not outlive the credential that authorizes
                    // it (§4.5). The enrollment record bounds the enrollment's
                    // own validity; this bounds the presented credential
                    // instance, so an admin-anchored enrollment with no expiry
                    // cannot extend a short-lived token's reach.
                    if let Some(credential_exp) =
                        ctx.claims().map(|claims| claims.exp).filter(|exp| *exp > 0)
                    {
                        let credential_exp = credential_exp as u64;
                        if proof.claims.exp > credential_exp {
                            anyhow::bail!(
                                "proof exp {} exceeds the presented credential's exp {credential_exp}",
                                proof.claims.exp
                            );
                        }
                    }
                }
                Ok(verified)
            })();
            match authority {
                Ok(verified) => Some(verified),
                Err(e) => {
                    warn!(
                        "{} proof authority denied (id={}): {e:#}",
                        service.name(),
                        request_id
                    );
                    return dispatch_denied("dispatch denied");
                }
            }
        }
    };

    // 2d. Generated per-method signature policy (§5.2 step 5, §4.4). The
    // decoded leaf — not the caller — decides whether this method may be
    // dispatched unattributed at all, which cryptographic suite its primary
    // logical signer must use, and which enrolled approvers must sign.
    // Enrollment says who signed; this says what the method requires.
    //
    // A leaf the generated table does not list is unlisted and denies. This
    // gate runs only for proof-bearing requests: the pre-v16 path has no
    // signed leaf to resolve, and is governed by the legacy checks below until
    // the migration completes.
    if let (Some(proof), Some(verified)) = (parsed_proof.as_ref(), verified_proof.as_ref()) {
        let leaf_path = match ctx.browser_method_discriminator {
            Some(method) => method.to_string(),
            // No derived leaf path: an empty path denies rather than falling
            // back to a coarser, more permissive row.
            None => {
                warn!(
                    "{} proof-bearing request has no derived leaf path (id={})",
                    service.name(),
                    request_id
                );
                return dispatch_denied("dispatch denied");
            }
        };
        let decision = match crate::proof::policy::global_method_policy() {
            Some(table) => match table.policy_for(actual_service_domain, &leaf_path) {
                Some(policy) => {
                    crate::proof::policy::evaluate(&policy, proof.disposition, verified)
                }
                None => Err(anyhow::anyhow!(
                    "unlisted (service, leaf) row for '{actual_service_domain}':'{leaf_path}'"
                )),
            },
            None => Err(anyhow::anyhow!(
                "no generated method policy table installed; proof-bearing dispatch denies"
            )),
        };
        if let Err(e) = decision {
            warn!(
                "{} generated method policy denied (id={}): {e:#}",
                service.name(),
                request_id
            );
            return dispatch_denied("dispatch denied");
        }
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
        return dispatch_denied(&format!("MAC deny: {reason:?}"));
    }

    // 2e. Replay admission (§5.2 step 7). Deliberately last: the pipeline
    // admits the replay key only after policy evaluation, so rejected and
    // denied requests never consume store capacity.
    if let (Some(proof), Some(verified)) = (parsed_proof.as_ref(), verified_proof.as_ref()) {
      let proof_admission = (|| -> Result<()> {
        // Gate 3: Challenge validation (unattributed proofs only).
        // validate() atomically returns the matched challenge's accept_until
        // so we can compute replay expiry without a TOCTOU race.
        let challenge_accept_until: Option<u64> = if proof.disposition == crate::proof::ProofDisposition::Unattributed {
            let challenge = proof.claims.nonce.as_ref()
                .ok_or_else(|| anyhow::anyhow!("unattributed proof missing Nonce claim"))?;
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

        // The issuer-declared credential profile selects the admission key
        // (§3.4, §4.5). A one-shot credential's ID is consumed atomically and
        // domain-wide as its SINGLE admission action: request_id still
        // correlates the response but creates no second replay entry. The
        // profile is read only from verified, issuer-signed claims — a caller
        // cannot mark its own token one-shot, and a method cannot reinterpret
        // a reusable token as one-shot.
        let one_shot = ctx
            .claims()
            .and_then(|claims| {
                claims
                    .credential_use
                    .filter(|use_profile| use_profile.is_one_shot())
                    .map(|_| claims)
            })
            .map(|claims| -> Result<crate::proof::admission::OneShotCredentialId> {
                // A one-shot credential with no credential ID cannot be
                // consumed, so it can never be admitted.
                let value = claims.jti.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("one-shot credential carries no credential ID (jti/cti)")
                })?;
                Ok(crate::proof::admission::OneShotCredentialId {
                    issuer: claims.iss.clone(),
                    value: value.as_bytes().to_vec(),
                })
            })
            .transpose()?;

        if let Some(credential_id) = one_shot {
            return match crate::proof::admission::consume_one_shot_credential(
                replay_store,
                &credential_id,
                replay_expiry,
            ) {
                crate::proof::admission::ProofAdmissionResult::Admitted => {
                    debug!("{} one-shot credential consumed", service.name());
                    Ok(())
                }
                crate::proof::admission::ProofAdmissionResult::Replayed => {
                    anyhow::bail!("one-shot credential already consumed")
                }
                crate::proof::admission::ProofAdmissionResult::Failed => {
                    // Including a deployment whose routing cannot make the
                    // consume domain-wide linearizable.
                    anyhow::bail!("one-shot credential consumption could not be guaranteed")
                }
            };
        }

        // The replay namespace comes from verification, not from wire
        // material: the credential-bound primary signer-suite thumbprint
        // (exact suite ID, ordered pinned component keys, enrollment epoch)
        // for authenticated proofs, and the (plan, key set) thumbprint for
        // unattributed ones.
        let replay_key = crate::proof::admission::ProofReplayKey {
            signer_thumbprint: verified.replay_thumbprint,
            request_id: proof.claims.request_id,
        };

        match crate::proof::admission::admit_request_proof(
            replay_store,
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
                // Capacity, backend unavailability, or a namespace this node
                // holds no history for. Every one of them is a denial.
                warn!(
                    "{} proof replay admission failed closed (capacity, backend, or namespace affinity)",
                    service.name()
                );
                anyhow::bail!("proof replay admission could not be guaranteed");
            }
        }
        Ok(())
      })();

      if let Err(e) = proof_admission {
        warn!(
            "{} proof admission denied (id={}): {e:#}",
            service.name(),
            request_id
        );
        return dispatch_denied("dispatch denied");
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
