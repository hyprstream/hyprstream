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
//! (the QUIC/WebTransport server has only the channel cert; client raw-public-key
//! binding is #200 and iroh live `node_id` is #282, neither wired). The accept
//! loop also does not know the peer's *origin* (it arrives in the app-layer
//! JWT/envelope), so even stage 1 cannot run at the raw accept loop. The gate is
//! therefore designed to be invoked from the service layer once that path passes
//! the verified signer key + issuer origin in; see `TODO(#200/#185/#282)` below.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use hyprstream_rpc::admission::{
    DidDocResolve, FederationAdmissionGate, OriginAdmission, PeerChannelKey,
};
use hyprstream_rpc::did_web::{ed25519_to_did_key, DidWebResolver, HttpDidDocFetcher};
use hyprstream_rpc::transport::iroh_admission::IrohPeerAdmission;

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
                domain: "*".to_owned(),
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
/// # TODO(#200/#185/#282) — live peer-key seam
///
/// The caller must pass the peer's *authenticated* Ed25519 key
/// (`PeerChannelKey`). Today that key is the verified COSE envelope signer key,
/// available at the service layer after `verify_claims`. Once RFC 7250
/// raw-public-key QUIC (#200) and/or iroh live `node_id` (#282) are wired, the
/// same key can be sourced from the channel; the gate's match logic is
/// unchanged.
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

/// #282: adapt the #137 [`FederationAdmissionGate`] to the iroh accept path's
/// [`IrohPeerAdmission`] hook.
///
/// Bridges the iroh `remote_id()` (the inbound peer's authenticated Ed25519
/// `node_id`) into the gate's three inputs:
///
/// - **DID** — the self-certifying `did:key` of `node_id` (#281). The key *is*
///   the identity, so the gate's key-binding stage 2 is satisfied without any
///   network fetch.
/// - **`PeerChannelKey`** — `node_id` itself: iroh's QUIC TLS already bound the
///   channel to this key, so it is the authenticated channel key the gate matches
///   (this is exactly the live peer-key the #137 quinn path lacked).
/// - **origin (RFC 6454)** — the residual seam. A raw inbound iroh peer carries
///   no http origin on the channel; the app-layer envelope/JWT `iss` does, but
///   that is not available at the accept loop. We therefore pass the `did:key`
///   string as the admission **subject** — the same identifier `did:key` peers
///   register under (#281) — so stage 1 (`federation:register`) is the
///   load-bearing decision. The per-request service-layer path still re-verifies
///   the envelope signer key == `remote_id()`, closing the loop.
///
/// Fail-closed: any gate error rejects the connection (the iroh handler drops it).
pub struct IrohFederationAdmission<O: OriginAdmission, R: DidDocResolve> {
    gate: Arc<FederationAdmissionGate<O, R>>,
}

impl<O: OriginAdmission, R: DidDocResolve> IrohFederationAdmission<O, R> {
    /// Wrap a shared #137 gate as the iroh accept-path admission hook.
    pub fn new(gate: Arc<FederationAdmissionGate<O, R>>) -> Self {
        Self { gate }
    }
}

#[async_trait]
impl<O: OriginAdmission + 'static, R: DidDocResolve + 'static> IrohPeerAdmission
    for IrohFederationAdmission<O, R>
{
    async fn admit_peer(&self, node_id: &[u8; 32]) -> Result<()> {
        // Self-certifying did:key of the authenticated node_id (#281): the key is
        // the identity, used both as the DID (stage 2 self-cert) and the stage-1
        // admission subject (origin seam — see struct docs).
        let did = ed25519_to_did_key(node_id);
        self.gate
            .admit(&did, PeerChannelKey(*node_id), &did, None)
            .await
            .map(|_admitted| ())
    }
}

/// Build the iroh accept-path admission hook over the real `PolicyService`
/// (stage 1) + native `did:web`/`did:key` resolver (stage 2), ready to install
/// on an iroh substrate via `QuicLoopConfig::iroh_admission`.
pub fn build_iroh_admission(
    policy_client: Arc<PolicyClient>,
) -> Result<Arc<dyn IrohPeerAdmission>> {
    let gate = Arc::new(build_federation_admission_gate(policy_client)?);
    Ok(Arc::new(IrohFederationAdmission::new(gate)))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod iroh_admission_tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use hyprstream_rpc::admission::FederationAdmissionGate;
    use rand::rngs::OsRng;
    use serde_json::Value;

    struct AllowAll;
    #[async_trait]
    impl OriginAdmission for AllowAll {
        async fn admit_origin(&self, _origin: &str) -> Result<()> {
            Ok(())
        }
    }
    struct DenyAll;
    #[async_trait]
    impl OriginAdmission for DenyAll {
        async fn admit_origin(&self, origin: &str) -> Result<()> {
            anyhow::bail!("origin {origin} denied")
        }
    }
    /// did:key is self-certifying — the resolver must never be called.
    struct NeverResolve;
    #[async_trait]
    impl DidDocResolve for NeverResolve {
        async fn resolve_doc(&self, did: &str) -> Result<Value> {
            panic!("did:key admission must not resolve (called for {did})")
        }
    }

    fn random_node_id() -> [u8; 32] {
        SigningKey::generate(&mut OsRng).verifying_key().to_bytes()
    }

    #[tokio::test]
    async fn iroh_peer_admitted_when_origin_policy_allows() {
        // The peer's remote_id() drives a self-certifying did:key; stage 1 allows
        // the subject → admitted. NeverResolve proves stage 2 is self-certifying.
        let gate = Arc::new(FederationAdmissionGate::new(AllowAll, NeverResolve));
        let hook = IrohFederationAdmission::new(gate);
        assert!(hook.admit_peer(&random_node_id()).await.is_ok());
    }

    #[tokio::test]
    async fn iroh_peer_rejected_fail_closed_when_policy_denies() {
        // Stage 1 (federation:register) denies the did:key subject → reject.
        let gate = Arc::new(FederationAdmissionGate::new(DenyAll, NeverResolve));
        let hook = IrohFederationAdmission::new(gate);
        assert!(hook.admit_peer(&random_node_id()).await.is_err());
    }

    #[tokio::test]
    async fn node_id_binds_to_its_did_key_iroh_vm() {
        // The DID the gate matches is exactly the did:key of the node_id, and its
        // key-binding stage trivially holds (self-certifying). This is the
        // node_id ↔ #iroh VM binding invariant (#282): the key advertised as the
        // `#iroh` VM is the node_id, and a peer presenting that node_id binds to
        // that DID.
        let node_id = random_node_id();
        let did = ed25519_to_did_key(&node_id);
        let decoded = hyprstream_rpc::did_web::did_key_to_ed25519(&did).unwrap();
        assert_eq!(decoded, node_id, "did:key(node_id) must round-trip to node_id");
    }
}
