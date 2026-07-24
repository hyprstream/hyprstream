//! `hyprstream`-side wiring for the federation admission gate (#137, M3).
//!
//! The transport-agnostic two-stage gate lives in
//! [`hyprstream_rpc::admission`] (lower crate, no `PolicyService` client). This
//! module supplies the **stage-1 origin-admission decision** over the real
//! `PolicyService`, reusing the existing unified federation trust gate
//! (`federation:register`) verbatim — the same Casbin resource checked by CIMD
//! client registration and [`crate::auth::federation::FederationKeyResolver`].
//!
//! It also exposes a constructor that assembles a ready-to-use
//! [`hyprstream_rpc::admission::FederationAdmissionGate`] from a `PolicyClient`
//! plus a `did:web` resolver, so the live accept/identity path can construct the
//! gate once and call `admit(...)` per inbound federated peer.
//!
//! ## Integration seam (honest status)
//!
//! Stage 1 here is wired against the real `PolicyService`. **Stage 2's live peer
//! key** is the documented seam: in this architecture the peer's authenticated
//! identity is the verified Ed25519 *envelope signer key* (`cnf`), surfaced at
//! the application/service layer ([`hyprstream_rpc::service`]'s
//! `verify_claims`/`resolve_key_subject`), **not** at the transport accept loop
//! (the QUIC/WebTransport server has only carrier metadata, which is never an
//! application signer proof). The accept
//! loop also does not know the peer's *origin* (it arrives in the app-layer
//! JWT/envelope), so even stage 1 cannot run at the raw accept loop. The gate is
//! therefore designed to be invoked from the service layer once that path passes
//! the verified signer key + issuer origin in; see the integration TODO below.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use hyprstream_rpc::admission::{FederationAdmissionGate, OriginAdmission};
use hyprstream_rpc::did_web::{DidWebResolver, HttpDidDocFetcher};

use crate::services::PolicyClient;

/// Stage-1 origin admission over `PolicyService` `federation:register`.
///
/// Wraps the same decision as [`crate::auth::federation::FederationKeyResolver`]'s
/// peer trust gate: an origin is admitted iff Casbin returns `allow` for
/// `(subject = origin, domain = "*", resource = "federation:register", op =
/// "check")`. Fail-closed on denial **and** on `PolicyService` outage — never
/// default-allow.
pub struct PolicyOriginAdmission {
    policy_client: Arc<PolicyClient>,
}
impl PolicyOriginAdmission {
    /// Construct over a shared `PolicyService` client.
    pub fn new(policy_client: Arc<PolicyClient>) -> Self {
        Self { policy_client }
    }
}
#[async_trait]
impl OriginAdmission for PolicyOriginAdmission {
    async fn admit_origin(&self, origin: &str) -> Result<()> {
        use crate::services::generated::policy_client::PolicyCheck;
        match self
            .policy_client
            .check(&PolicyCheck {
                subject: origin.to_owned(),
                domain: origin.to_owned(),
                resource: "federation:register".to_owned(),
                operation: "check".to_owned(),
            })
            .await
        {
            Ok(true) => Ok(()),
            Ok(false) => anyhow::bail!(
                "origin {origin} is not permitted by policy (federation:register denied)"
            ),
            Err(e) => {
                // Fail-closed: an unreachable PolicyService must reject, never
                // default-allow. Same posture as the CIMD / FederationKeyResolver
                // gate.
                tracing::error!(
                    origin = %origin,
                    error = %e,
                    "PolicyService unreachable during federation admission stage 1 — failing closed"
                );
                anyhow::bail!("PolicyService unreachable; federation admission rejected (fail-closed): {e}")
            }
        }
    }
}

/// Assemble a ready-to-use [`FederationAdmissionGate`] over the real
/// `PolicyService` (stage 1) and a native `did:web` resolver (stage 2).
///
/// The resolver reuses the #279 [`HttpDidDocFetcher`] (TTL-cached, SSRF/DoS
/// hardened) — no new fetch/cache infrastructure. Construct this once and call
/// [`FederationAdmissionGate::admit`] per inbound federated peer with the
/// verified envelope-signer key and the peer's origin + DID.
///
/// # Live proof seam
///
/// The caller must pass the peer's *authenticated* Ed25519 key
/// (`ApplicationSignerKey`). Today that key is the verified COSE envelope signer key,
/// available at the service layer after `verify_claims`. A transport NodeId is
/// never a substitute for that application proof (#1031/#1027).
pub fn build_federation_admission_gate(
    policy_client: Arc<PolicyClient>,
) -> Result<FederationAdmissionGate<PolicyOriginAdmission, DidWebResolver<HttpDidDocFetcher>>> {
    // One-hour TTL, matching the JWKS / DID-doc cache default.
    let fetcher = HttpDidDocFetcher::new(std::time::Duration::from_secs(3600))?;
    let resolver = DidWebResolver::new(fetcher);
    Ok(FederationAdmissionGate::new(
        PolicyOriginAdmission::new(policy_client),
        resolver,
    ))
}
