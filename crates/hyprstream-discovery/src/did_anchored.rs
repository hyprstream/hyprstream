//! DID-anchored deployment trust bootstrap (#1136).
//!
//! This module resolves two public operator pins into verification-only
//! deployment material. Network responses are never authority by themselves:
//! the fetched capsule must pass the canon -> hash -> signature GATE for the
//! configured `did:at9p`, and the fetched `did:web` document must reciprocally
//! name that exact identity before the pair is trusted. The deployment CA and
//! Discovery reach come exclusively from the GATE-verified capsule; document
//! keys and services remain advisory.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use hyprstream_pds::at9p::{ServiceType, Transport as At9pTransport};
use hyprstream_pds::at9p_alias::AuthoritativeIdentity;
use hyprstream_pds::at9p_gate::VerifiedCapsule;
use hyprstream_rpc::did_web::{did_web_to_url, DidWebResolver, HttpDidDocFetcher};
use hyprstream_rpc::identity::Did;
use hyprstream_rpc::transport::{EndpointType, TransportConfig};
use serde_json::Value;

use crate::at9p_alias::At9pAliasResolver;
use crate::at9p_resolver::CapsuleSource;
use crate::service::{DeploymentAuthorityLog, HybridDeploymentCa};

const MAX_CAPSULE_BYTES: usize = 4 * 1024 * 1024;
const DEPLOYMENT_REACH_SERVICE: &str = "#ns";

/// The two public, non-secret anchors for DID-backed deployment trust.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DidAnchors {
    pub cluster_at9p_did: String,
    pub cluster_did_web: String,
    /// Optional extra TLS root (PEM) for private-PKI deployments whose
    /// did:web host terminates with an internal CA. ADDITIVE — the public
    /// WebPKI roots remain enabled; this never disables verification.
    pub extra_root_cert_pem: Option<Vec<u8>>,
}

impl DidAnchors {
    /// Attach an extra TLS root for private-PKI did:web termination.
    pub fn with_root_cert_pem(mut self, pem: Vec<u8>) -> Self {
        self.extra_root_cert_pem = Some(pem);
        self
    }
}

/// Explicit startup trust-source selection.
///
/// There is deliberately no fallback between variants: once `DidAnchored` is
/// selected, any fetch, verification, or liveness failure is terminal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DeploymentTrustSource {
    OsOwnedFiles,
    DidAnchored(DidAnchors),
}

impl DeploymentTrustSource {
    /// Select a trust source from the public anchor pair.
    ///
    /// Both unset preserves the historical OS-owned-file behavior exactly.
    /// Supplying only one anchor is rejected rather than silently downgrading.
    pub fn from_anchors(
        cluster_at9p_did: Option<&str>,
        cluster_did_web: Option<&str>,
    ) -> Result<Self> {
        match (cluster_at9p_did, cluster_did_web) {
            (None, None) => Ok(Self::OsOwnedFiles),
            (Some(at9p), Some(web)) => {
                anyhow::ensure!(
                    at9p.starts_with(hyprstream_pds::at9p_gate::DID_AT9P_PREFIX),
                    "cluster_at9p_did must be a did:at9p identifier"
                );
                anyhow::ensure!(
                    web.starts_with("did:web:"),
                    "cluster_did_web must be a did:web identifier"
                );
                anyhow::ensure!(
                    at9p.trim() == at9p && web.trim() == web,
                    "DID deployment anchors must not contain surrounding whitespace"
                );
                // Validate the did:web method-specific identifier before any I/O.
                did_web_to_url(web).context("cluster_did_web is malformed")?;
                Ok(Self::DidAnchored(DidAnchors {
                    cluster_at9p_did: at9p.to_owned(),
                    cluster_did_web: web.to_owned(),
                    extra_root_cert_pem: None,
                }))
            }
            _ => bail!(
                "cluster_at9p_did and cluster_did_web must be configured together; partial DID trust configuration is forbidden"
            ),
        }
    }
}

/// Verified public material extracted from a mutually-attested identity pair.
pub(crate) struct DidAnchoredTrust {
    pub ca_verifying_key: HybridDeploymentCa,
    pub discovery_transport: TransportConfig,
    pub authoritative_identity: AuthoritativeIdentity,
    /// The current CA-signed registry deployment credential, fetched from the
    /// same well-known endpoint family as the capsule. Like the capsule, it
    /// is integrity-protected by design (CA signature + one-hour freshness
    /// profile, enforced by `validate_registry_deployment_credential_profile`),
    /// so the byte channel needs no trust of its own.
    pub registry_credential: String,
    /// Current root-anchored authority log fetched beside the credential.
    /// Its verified head must match the independently provisioned local
    /// checkpoint before the credential is accepted.
    pub authority_log: DeploymentAuthorityLog,
    /// The document's `#mesh-kem` hybrid-KEM recipient public, when published.
    /// Required by the REMOTE-node bootstrap arm (QUIC forbids cleartext
    /// envelopes); the same-node arm never encrypts over the local fabric.
    pub mesh_kem_recipient: Option<hyprstream_rpc::crypto::hybrid_kem::RecipientPublic>,
    /// The document's ML-DSA-65 verification methods (`#mesh-pq`). The
    /// remote-node arm requires exactly one — the discovery service's mesh
    /// PQ verifying key — to anchor response authentication.
    pub ml_dsa_65_keys: Vec<hyprstream_rpc::crypto::pq::MlDsaVerifyingKey>,
}

/// Fetch capsules from the deployment's static well-known content endpoint.
///
/// A root `did:web:host` document at `/.well-known/did.json` maps capsule CIDs
/// to `/.well-known/at9p/<cid>.cbor`; a path-form DID maps them beside its
/// `did.json`. This endpoint is an untrusted byte transport only: the configured
/// CID and GATE pipeline decide whether the bytes are accepted.
struct HttpWellKnownCapsuleSource {
    http: reqwest::Client,
    document_url: String,
}

/// Maximum accepted size of the registry deployment credential response.
/// A compact EdDSA JWT is well under 4 KiB; 64 KiB is generous headroom.
const MAX_CREDENTIAL_BYTES: usize = 64 * 1024;

impl HttpWellKnownCapsuleSource {
    fn new(did_web: &str, extra_root_pem: Option<&[u8]>) -> Result<Self> {
        let document_url = did_web_to_url(did_web)?;
        let mut builder = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(10));
        if let Some(pem) = extra_root_pem {
            let cert = reqwest::Certificate::from_pem(pem)
                .context("extra well-known TLS root is not a valid PEM certificate")?;
            builder = builder.add_root_certificate(cert);
        }
        let http = builder
            .build()
            .context("failed to build at9p capsule HTTPS client")?;
        Ok(Self { http, document_url })
    }

    /// Fetch the current registry deployment credential from beside the
    /// capsule (`{prefix}deployment/registry-service.jwt`). The bytes are an
    /// untrusted transport only: `validate_registry_deployment_credential_profile`
    /// decides whether they are accepted (CA signature + freshness window).
    async fn fetch_registry_credential(&self) -> Result<String> {
        let prefix = self
            .document_url
            .strip_suffix("did.json")
            .ok_or_else(|| anyhow::anyhow!("derived did:web URL does not end in did.json"))?;
        let url = format!("{prefix}deployment/registry-service.jwt");
        let mut response = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("failed to fetch registry deployment credential from {url}"))?
            .error_for_status()
            .with_context(|| format!("registry deployment credential endpoint rejected {url}"))?;
        if let Some(length) = response.content_length() {
            anyhow::ensure!(
                length <= MAX_CREDENTIAL_BYTES as u64,
                "registry deployment credential exceeds {MAX_CREDENTIAL_BYTES}-byte limit"
            );
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .context("failed to read registry deployment credential body")?
        {
            anyhow::ensure!(
                bytes.len().saturating_add(chunk.len()) <= MAX_CREDENTIAL_BYTES,
                "registry deployment credential exceeds {MAX_CREDENTIAL_BYTES}-byte limit"
            );
            bytes.extend_from_slice(&chunk);
        }
        String::from_utf8(bytes).context("registry deployment credential is not UTF-8")
    }

    /// Fetch the current root-anchored authority log from beside the registry
    /// credential. HTTPS transports untrusted bytes; the capsule-derived root
    /// and independent local head checkpoint authenticate them later.
    async fn fetch_authority_log(&self) -> Result<DeploymentAuthorityLog> {
        let prefix = self
            .document_url
            .strip_suffix("did.json")
            .ok_or_else(|| anyhow::anyhow!("derived did:web URL does not end in did.json"))?;
        let url = format!("{prefix}deployment/deployment-authority.log.json");
        let mut response = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("failed to fetch deployment authority log from {url}"))?
            .error_for_status()
            .with_context(|| format!("deployment authority-log endpoint rejected {url}"))?;
        if let Some(length) = response.content_length() {
            anyhow::ensure!(
                length <= MAX_CREDENTIAL_BYTES as u64,
                "deployment authority log exceeds {MAX_CREDENTIAL_BYTES}-byte limit"
            );
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .context("failed to read deployment authority-log body")?
        {
            anyhow::ensure!(
                bytes.len().saturating_add(chunk.len()) <= MAX_CREDENTIAL_BYTES,
                "deployment authority log exceeds {MAX_CREDENTIAL_BYTES}-byte limit"
            );
            bytes.extend_from_slice(&chunk);
        }
        serde_json::from_slice(&bytes).context("deployment authority log is malformed")
    }

    fn capsule_url(&self, did: &str) -> Result<String> {
        let cid = did
            .strip_prefix(hyprstream_pds::at9p_gate::DID_AT9P_PREFIX)
            .ok_or_else(|| anyhow::anyhow!("capsule fetch requested for a non-at9p DID"))?;
        // The full CID parser runs in GATE after fetch. Constrain the URL path
        // before I/O so a malformed configured identifier cannot inject path,
        // query, or authority syntax.
        anyhow::ensure!(
            !cid.is_empty()
                && cid
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit()),
            "did:at9p CID contains URL-unsafe characters"
        );
        let prefix = self
            .document_url
            .strip_suffix("did.json")
            .ok_or_else(|| anyhow::anyhow!("derived did:web URL does not end in did.json"))?;
        Ok(format!("{prefix}at9p/{cid}.cbor"))
    }
}

#[async_trait]
impl CapsuleSource for HttpWellKnownCapsuleSource {
    async fn fetch_capsule(&self, did: &str) -> Result<Vec<u8>> {
        let url = self.capsule_url(did)?;
        let mut response = self
            .http
            .get(&url)
            .send()
            .await
            .with_context(|| format!("failed to fetch at9p capsule from {url}"))?
            .error_for_status()
            .with_context(|| format!("at9p capsule endpoint rejected {url}"))?;
        if let Some(length) = response.content_length() {
            anyhow::ensure!(
                length <= MAX_CAPSULE_BYTES as u64,
                "at9p capsule exceeds {MAX_CAPSULE_BYTES}-byte limit"
            );
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            anyhow::ensure!(
                bytes.len().saturating_add(chunk.len()) <= MAX_CAPSULE_BYTES,
                "at9p capsule exceeds {MAX_CAPSULE_BYTES}-byte limit"
            );
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }
}

fn document_names_at9p(document: &Value, at9p_did: &str) -> bool {
    document
        .get("alsoKnownAs")
        .and_then(Value::as_array)
        .is_some_and(|aliases| aliases.iter().any(|alias| alias.as_str() == Some(at9p_did)))
}

/// Enforce the deployment-specific, closed anchor-capsule profile.
///
/// Generic at9p capsules intentionally have set semantics and may carry
/// rotation or delegation material. A deployment anchor is narrower: every
/// signed claim must be one the anchor minter and this resolver consume.
fn validate_deployment_anchor_profile(
    verified: &VerifiedCapsule,
    configured_did_web: &str,
) -> Result<()> {
    let body = &verified.capsule().body;
    anyhow::ensure!(
        body.subject_keys.len() == 1,
        "closed deployment-anchor profile violation: subjectKeys must contain exactly one pinned-Hybrid signer (got {})",
        body.subject_keys.len()
    );
    anyhow::ensure!(
        body.services.len() == 1,
        "closed deployment-anchor profile violation: services must contain exactly one #ns NinePExport entry (got {})",
        body.services.len()
    );
    let service = &body.services[0];
    anyhow::ensure!(
        service.id == DEPLOYMENT_REACH_SERVICE
            && service.service_type == ServiceType::NinePExport,
        "closed deployment-anchor profile violation: sole service must be #ns with type NinePExport"
    );
    let aliases = body.also_known_as.as_deref().unwrap_or_default();
    anyhow::ensure!(
        aliases.len() == 1 && aliases[0] == configured_did_web,
        "closed deployment-anchor profile violation: alsoKnownAs must contain exactly the configured did:web {configured_did_web:?}"
    );
    anyhow::ensure!(
        body.next_key_commitments.is_empty(),
        "closed deployment-anchor profile violation: nextKeyCommitments are forbidden"
    );
    anyhow::ensure!(
        body.label_hints.is_none(),
        "closed deployment-anchor profile violation: labelHints are forbidden"
    );
    anyhow::ensure!(
        body.delegations.is_none(),
        "closed deployment-anchor profile violation: delegations are forbidden"
    );
    anyhow::ensure!(
        body.witnesses.is_none(),
        "closed deployment-anchor profile violation: witnesses are forbidden"
    );
    anyhow::ensure!(
        service.endpoint.export.is_none(),
        "closed deployment-anchor profile violation: the #ns endpoint export field is forbidden"
    );
    match service.endpoint.transport {
        At9pTransport::Iroh => {}
        At9pTransport::Quic => anyhow::ensure!(
            service.endpoint.node_id.is_none() && service.endpoint.relay.is_none(),
            "closed deployment-anchor profile violation: QUIC #ns endpoints must not carry iroh nodeId or relay fields"
        ),
        ref other => bail!(
            "closed deployment-anchor profile violation: #ns transport must be iroh or quic (got {other:?})"
        ),
    }
    Ok(())
}

/// Extract the deployment CA from the sole hybrid subject key that signed the
/// GATE-verified, closed-profile capsule.
fn ca_key_from_capsule(verified: &VerifiedCapsule) -> Result<HybridDeploymentCa> {
    let [subject] = verified.capsule().body.subject_keys.as_slice() else {
        bail!("closed deployment-anchor profile violation: subjectKeys must contain exactly one pinned-Hybrid signer");
    };
    HybridDeploymentCa::from_public_key_bytes(&subject.ed25519_pub, &subject.mldsa65_pub)
        .context("capsule sole subject key is not a valid hybrid deployment CA")
}

/// Extract Discovery reach from the capsule's typed `#ns` service. The
/// independent nodeId is transport reach only; the signed ping remains pinned
/// to the separately authenticated Discovery application key.
fn reach_from_capsule(verified: &VerifiedCapsule, document: &Value) -> Result<TransportConfig> {
    let entry = verified
        .capsule()
        .body
        .services
        .iter()
        .find(|service| {
            service.id == DEPLOYMENT_REACH_SERVICE
                && service.service_type == ServiceType::NinePExport
        })
        .ok_or_else(|| {
            anyhow::anyhow!(
"capsule has no NinePExport service entry {DEPLOYMENT_REACH_SERVICE:?} for deployment reach; \
                 the pinned did:at9p must be an ANCHOR capsule signed by the deployment CA, not a node's \
                 own identity capsule (e.g. the `#pds` capsule a node's OAuth service renders for \
                 itself) — mint one with `hyprstream trust mint-anchor-capsule` and publish it \
                 (with its did.json) under the deployment well-known directory"
            )
        })?;
    match entry.endpoint.transport {
        At9pTransport::Iroh => {
            let node_id_multibase = entry.endpoint.node_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "capsule deployment reach {DEPLOYMENT_REACH_SERVICE:?} carries no independent iroh nodeId"
                )
            })?;
            let node_id = hyprstream_rpc::did_key::decode_ed25519_multikey(node_id_multibase)
                .context("capsule deployment reach nodeId is not a valid Ed25519 Multikey")?;
            Ok(TransportConfig::iroh(
                node_id,
                Vec::new(),
                entry.endpoint.relay.clone(),
            ))
        }
        At9pTransport::Quic => {
            let carrier = entry
                .endpoint
                .address
                .strip_prefix("quic://")
                .ok_or_else(|| {
                    anyhow::anyhow!("capsule QUIC reach must use a quic:// socket carrier")
                })?;
            let address = carrier
                .parse()
                .context("capsule QUIC reach is not an IP socket address")?;
            // The capsule remains authoritative for WHERE to dial. The
            // did:web service may only contribute channel mechanics (SNI,
            // WebPKI policy, and certificate hashes) for that exact
            // capsule-bound socket. Those mechanics cannot select application
            // identity: the signed ping is still pinned independently.
            //
            // A hostname-based document URI (`https://host:port`) decodes to
            // an unspecified address with only the port populated, so the match
            // accepts either an exact address equality or a port-only match
            // when the document entry carries an unspecified IP.
            let document_transport = hyprstream_rpc::did_web::transport_entries(document)
                .into_iter()
                .map(|decoded| decoded.config)
                .find(|config| {
                    matches!(
                        &config.endpoint,
                        EndpointType::Quic { addr, .. }
                            if *addr == address
                                || (addr.ip().is_unspecified() && addr.port() == address.port())
                    )
                })
                .map(|mut config| {
                    // A hostname-based document URI decodes to an unspecified
                    // IP with only the port populated. The capsule carries the
                    // real dial address, so substitute it: without this rewrite
                    // the config would dial 0.0.0.0:port and never connect,
                    // turning a verified capsule into an unusable one.
                    if let EndpointType::Quic { addr, .. } = &mut config.endpoint {
                        if addr.ip().is_unspecified() {
                            *addr = address;
                        }
                    }
                    config
                });
            Ok(document_transport.unwrap_or_else(|| {
                tracing::warn!(
                    "capsule QUIC reach {} has no matching QuicTransport entry in the did:web \
                     document; falling back to bare-IP dial without SNI or certificate pins — \
                     the deployment will not be reachable via hostname or WebPKI-validated TLS",
                    address
                );
                TransportConfig::quic(address, address.ip().to_string()).with_connect_mode()
            }))
        }
        ref other => bail!(
            "capsule deployment reach {DEPLOYMENT_REACH_SERVICE:?} is not an iroh or QUIC endpoint (got {other:?})"
        ),
    }
}

/// Deployment material an anchor capsule contributes, once the capsule and the
/// `did:web` document have mutually attested to each other.
///
/// This is the verification-only half of the DID-anchored bootstrap — the part
/// a minting ceremony can self-check offline, before publication and before any
/// registry credential or authority log exists.
#[derive(Clone, Debug)]
pub struct VerifiedAnchorMaterial {
    /// The GATE-verified `did:at9p` the document reciprocally names.
    pub at9p_did: String,
    /// Raw hybrid deployment root taken from the capsule's primary subject
    /// key: the 32-byte Ed25519 key followed by the 1952-byte ML-DSA-65 key,
    /// the same layout the OS-owned `deployment-ca.hybrid` pin uses.
    pub deployment_ca_public: Vec<u8>,
    /// Deployment reach decoded from the capsule's `#ns` NinePExport entry.
    pub discovery_transport: TransportConfig,
}

/// Verify an anchor capsule against its `did:web` document through the
/// production resolution path, without needing the registry credential or
/// authority log the live bootstrap also fetches.
///
/// Minting ceremonies call this on their own output: material this rejects is
/// material a node would refuse at boot.
pub async fn verify_anchor_material(
    anchors: &DidAnchors,
    document: &Value,
    capsule_source: Arc<dyn CapsuleSource>,
) -> Result<VerifiedAnchorMaterial> {
    let (identity, ca, discovery_transport) =
        resolve_anchor_pair(anchors, document, capsule_source).await?;
    let mut deployment_ca_public = ca.ed25519_bytes().to_vec();
    deployment_ca_public.extend_from_slice(&ca.ml_dsa_65_bytes());
    Ok(VerifiedAnchorMaterial {
        at9p_did: identity.at9p_did.as_str().to_owned(),
        deployment_ca_public,
        discovery_transport,
    })
}

/// Shared core of the DID-anchored trust decision: reciprocal naming, the
/// capsule GATE, and the CA + reach the GATE-verified capsule carries.
async fn resolve_anchor_pair(
    anchors: &DidAnchors,
    document: &Value,
    capsule_source: Arc<dyn CapsuleSource>,
) -> Result<(AuthoritativeIdentity, HybridDeploymentCa, TransportConfig)> {
    anyhow::ensure!(
        document.get("id").and_then(Value::as_str) == Some(anchors.cluster_did_web.as_str()),
        "did:web document id does not match configured cluster_did_web"
    );
    anyhow::ensure!(
        document_names_at9p(document, &anchors.cluster_at9p_did),
        "did:web document does not name the configured did:at9p in alsoKnownAs"
    );

    let classical = Did::new(anchors.cluster_did_web.clone());
    let at9p = Did::new(anchors.cluster_at9p_did.clone());
    let (authoritative_identity, verified) = At9pAliasResolver::new(capsule_source)
        .resolve_authoritative_with_capsule(&classical, &at9p)
        .await
        .context("DID deployment anchors failed mutual-alias verification")?;
    anyhow::ensure!(
        authoritative_identity.at9p_did == at9p,
        "mutual-alias resolver did not preserve configured at9p authority"
    );

    validate_deployment_anchor_profile(&verified, &anchors.cluster_did_web)?;

    // The document contributes only the reciprocal identifier vouch above.
    // Everything installed is content-bound to the configured did:at9p pin.
    let ca_verifying_key = ca_key_from_capsule(&verified)?;
    let discovery_transport = reach_from_capsule(&verified, document)?;
    anyhow::ensure!(
        matches!(
            discovery_transport.endpoint,
            EndpointType::Iroh { .. } | EndpointType::Quic { .. }
        ),
        "capsule deployment reach is not a network transport"
    );
    Ok((
        authoritative_identity,
        ca_verifying_key,
        discovery_transport,
    ))
}

pub(crate) async fn verify_did_anchored_document(
    anchors: &DidAnchors,
    document: &Value,
    capsule_source: Arc<dyn CapsuleSource>,
    registry_credential: String,
    authority_log: DeploymentAuthorityLog,
) -> Result<DidAnchoredTrust> {
    let (authoritative_identity, ca_verifying_key, discovery_transport) =
        resolve_anchor_pair(anchors, document, capsule_source).await?;
    Ok(DidAnchoredTrust {
        ca_verifying_key,
        discovery_transport,
        authoritative_identity,
        registry_credential,
        authority_log,
        mesh_kem_recipient: hyprstream_rpc::did_web::mesh_kem_recipient(document),
        ml_dsa_65_keys: hyprstream_rpc::did_web::verification_method_ml_dsa_65_keys(document),
    })
}

pub(crate) async fn resolve_did_anchored_trust(anchors: &DidAnchors) -> Result<DidAnchoredTrust> {
    let extra_root = anchors.extra_root_cert_pem.as_deref();
    let document = DidWebResolver::new(match extra_root {
        Some(pem) => HttpDidDocFetcher::with_extra_root(Duration::from_secs(3600), pem)?,
        None => HttpDidDocFetcher::new(Duration::from_secs(3600))?,
    })
    .resolve_document(&anchors.cluster_did_web)
    .await
    .context("failed to fetch configured cluster did:web document")?;
    let capsule_source = Arc::new(HttpWellKnownCapsuleSource::new(
        &anchors.cluster_did_web,
        extra_root,
    )?);
    // Fetch the registry credential from the same well-known family BEFORE
    // trusting anything: a missing/refusing endpoint must fail the bootstrap
    // even when the document and capsule are otherwise valid.
    let registry_credential = capsule_source
        .fetch_registry_credential()
        .await
        .context("failed to fetch registry deployment credential")?;
    let authority_log = capsule_source
        .fetch_authority_log()
        .await
        .context("failed to fetch deployment authority log")?;
    let trust = verify_did_anchored_document(
        anchors,
        &document,
        capsule_source,
        registry_credential,
        authority_log,
    )
    .await?;
    tracing::info!(
        at9p = %trust.authoritative_identity.at9p_did,
        did_web = %trust.authoritative_identity.classical_did,
        "verified mutually-attested DID deployment trust anchors from GATE-verified capsule"
    );
    Ok(trust)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use base64::{
        engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
        Engine as _,
    };
    use ed25519_dalek::{Signer as _, SigningKey};
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};
    use hyprstream_pds::at9p::{
        CapsuleBody, Delegation, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType,
        Transport, ML_DSA65_PUBLIC_KEY_LEN,
    };
    use hyprstream_pds::at9p_gate::verify_did_at9p;
    use hyprstream_pds::at9p_sign::sign_capsule;
    use hyprstream_pds::dag_cbor::DagCbor;
    use hyprstream_rpc::{
        auth::ucan::{
            Ability, Capability, CaveatValue, Caveats, Did as UcanDid, Resource, Ucan, UcanPayload,
        },
        crypto::{
            cose_sign::{assemble_composite_nested, inner_tbs, outer_tbs, split_composite},
            pq::{ml_dsa_sign, ml_dsa_sk_to_vk_bytes},
        },
    };
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::net::SocketAddr;

    struct FixedCapsuleSource(Vec<u8>);

    #[async_trait]
    impl CapsuleSource for FixedCapsuleSource {
        async fn fetch_capsule(&self, _did: &str) -> Result<Vec<u8>> {
            Ok(self.0.clone())
        }
    }

    struct CapsuleSigner {
        ed: SigningKey,
        pq: MlDsaSigningKey,
        pair: HybridKeyPair,
    }

    fn capsule_signer(tag: u8) -> CapsuleSigner {
        let ed = SigningKey::from_bytes(&[tag; 32]);
        let (pq, pq_vk) = ml_dsa_generate_keypair();
        let pair = HybridKeyPair::new(
            ed.verifying_key().to_bytes().to_vec(),
            ml_dsa_vk_bytes(&pq_vk),
        )
        .unwrap();
        CapsuleSigner { ed, pq, pair }
    }

    fn multikey(key: &[u8; 32]) -> String {
        hyprstream_rpc::did_key::ed25519_to_did_key(key)
            .strip_prefix("did:key:")
            .unwrap()
            .to_owned()
    }

    fn capsule_ca(tag: u8) -> [u8; 32] {
        SigningKey::from_bytes(&[tag; 32])
            .verifying_key()
            .to_bytes()
    }

    fn unused_authority_log() -> DeploymentAuthorityLog {
        DeploymentAuthorityLog {
            schema: "unused-test-schema".to_owned(),
            deployment_domain: "unused-test-domain".to_owned(),
            did: "did:plc:unusedtestauthoritylog".to_owned(),
            operations_b64: vec![],
        }
    }

    fn sign_nested_for_test(
        payload: &[u8],
        aad: &[u8],
        ed: &SigningKey,
        pq: &MlDsaSigningKey,
    ) -> Vec<u8> {
        let ed_signature = ed.sign(&inner_tbs(
            ed.verifying_key().to_bytes().to_vec(),
            payload,
            aad,
            true,
        ));
        let pq_public =
            hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(pq)).unwrap();
        let pq_signature = ml_dsa_sign(
            pq,
            &outer_tbs(
                ml_dsa_vk_bytes(&pq_public),
                payload,
                &ed_signature.to_bytes(),
                aad,
            ),
        );
        assemble_composite_nested(
            (
                ed.verifying_key().to_bytes().to_vec(),
                ed_signature.to_bytes().to_vec(),
            ),
            Some((ml_dsa_vk_bytes(&pq_public), pq_signature)),
        )
        .unwrap()
    }

    fn authority_log_and_checkpoint(
        root: &CapsuleSigner,
        deployment_domain: &str,
    ) -> (
        DeploymentAuthorityLog,
        crate::service::DeploymentAuthorityCheckpoint,
    ) {
        let mut genesis = crate::did_op::DidOp {
            sequence: 0,
            prev: None,
            rotation_keys: vec![crate::did_op::HybridRotationKey::new(
                root.ed.verifying_key().to_bytes(),
                ml_dsa_sk_to_vk_bytes(&root.pq),
            )
            .unwrap()],
            signature: crate::did_op::HybridDidOpSignature {
                ed25519: vec![0; 64],
                mldsa65: vec![0; 3_309],
            },
        };
        let composite = sign_nested_for_test(
            &genesis.signable_bytes(),
            crate::did_op::DID_OP_SIGNATURE_CONTEXT,
            &root.ed,
            &root.pq,
        );
        let (ed25519, mldsa65) = split_composite(&composite).unwrap();
        genesis.signature = crate::did_op::HybridDidOpSignature {
            ed25519,
            mldsa65: mldsa65.unwrap(),
        };
        let did = genesis.genesis_did().unwrap();
        let verified = crate::did_op::verify_did_op_log(&did, &[genesis.clone()]).unwrap();
        (
            DeploymentAuthorityLog {
                schema: "hyprstream.deployment-authority-log.v1".to_owned(),
                deployment_domain: deployment_domain.to_owned(),
                did: did.clone(),
                operations_b64: vec![STANDARD.encode(genesis.to_dag_cbor())],
            },
            crate::service::DeploymentAuthorityCheckpoint {
                schema: "hyprstream.deployment-authority-checkpoint.v1".to_owned(),
                deployment_domain: deployment_domain.to_owned(),
                did,
                sequence: verified.sequence,
                head_cid: verified.head_cid,
            },
        )
    }

    fn delegated_registry_credential(
        root: &CapsuleSigner,
        authority_log: &DeploymentAuthorityLog,
        deployment_domain: &str,
        registry: &SigningKey,
    ) -> String {
        let delegated_ed = SigningKey::from_bytes(&[0xA2; 32]);
        let (delegated_pq, _) = ml_dsa_generate_keypair();
        let mut delegated_public = delegated_ed.verifying_key().to_bytes().to_vec();
        delegated_public.extend_from_slice(&ml_dsa_sk_to_vk_bytes(&delegated_pq));
        let now = u64::try_from(chrono::Utc::now().timestamp()).unwrap();
        let mut caveats = BTreeMap::new();
        caveats.insert(
            "audience".to_owned(),
            CaveatValue::Text("urn:hyprstream:service:registry".to_owned()),
        );
        caveats.insert(
            "deployment_domain".to_owned(),
            CaveatValue::Text(deployment_domain.to_owned()),
        );
        caveats.insert(
            "delegated_public_key_b64".to_owned(),
            CaveatValue::Text(STANDARD.encode(&delegated_public)),
        );
        caveats.insert("max_ttl_seconds".to_owned(), CaveatValue::Int(3_600));
        caveats.insert(
            "profile".to_owned(),
            CaveatValue::Text("hyprstream.registry-deployment.v1".to_owned()),
        );
        let payload = UcanPayload {
            issuer: UcanDid::from_ed25519(&root.ed.verifying_key().to_bytes()),
            audience: UcanDid::from_ed25519(&delegated_ed.verifying_key().to_bytes()),
            capabilities: vec![Capability::with_caveats(
                Resource::new(format!(
                    "hyprstream://deployment/{deployment_domain}/service/registry"
                )),
                Ability::new("mint-registry-jwt"),
                Caveats(caveats),
            )],
            not_before: Some(now),
            expiration: Some(now + 3_600),
            nonce: vec![0xA3; 16],
        };
        let ucan = Ucan {
            signature: sign_nested_for_test(
                &payload.signing_bytes().unwrap(),
                hyprstream_rpc::auth::ucan::token::UCAN_AAD,
                &root.ed,
                &root.pq,
            ),
            payload,
            proofs: vec![],
        };
        let artifact = crate::service::RegistryDelegationArtifact {
            schema: "hyprstream.registry-delegation.v1".to_owned(),
            deployment_domain: deployment_domain.to_owned(),
            authority_log_did: authority_log.did.clone(),
            delegated_public_key_b64: STANDARD.encode(&delegated_public),
            ucan_b64: STANDARD.encode(ucan.to_cbor().unwrap()),
        };
        let exp = i64::try_from(now + 60).unwrap();
        let now = i64::try_from(now).unwrap();
        let kid = hyprstream_rpc::auth::composite_kid(
            &hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(
                &delegated_pq,
            ))
            .unwrap(),
            &delegated_ed.verifying_key(),
        );
        let protected = URL_SAFE_NO_PAD.encode(
            serde_json::to_vec(&json!({
                "alg": "ML-DSA-65-Ed25519",
                "typ": "wit+jwt",
                "kid": kid,
            }))
            .unwrap(),
        );
        let claims = URL_SAFE_NO_PAD.encode(
            serde_json::to_vec(&json!({
                "iss": format!("urn:hyprstream:deployment:{deployment_domain}"),
                "sub": "service:registry",
                "aud": "urn:hyprstream:service:registry",
                "exp": exp,
                "nbf": now,
                "iat": now,
                "deployment_domain": deployment_domain,
                "profile": "hyprstream.registry-deployment.v1",
                "cnf": {"jwk": {
                    "kty": "OKP",
                    "crv": "Ed25519",
                    "x": URL_SAFE_NO_PAD.encode(registry.verifying_key().as_bytes()),
                }},
                "delegation": URL_SAFE_NO_PAD.encode(serde_json::to_vec(&artifact).unwrap()),
            }))
            .unwrap(),
        );
        let signing_input = format!("{protected}.{claims}");
        let mut signature = ml_dsa_sign(&delegated_pq, signing_input.as_bytes());
        signature.extend_from_slice(&delegated_ed.sign(signing_input.as_bytes()).to_bytes());
        format!("{signing_input}.{}", URL_SAFE_NO_PAD.encode(signature))
    }

    fn capsule_with_carrier(
        classical_alias: &str,
        tag: u8,
        carrier: Option<[u8; 32]>,
    ) -> (Vec<u8>, String) {
        let signer = capsule_signer(tag);
        let mut endpoint =
            ServiceEndpoint::new(Transport::Iroh, format!("iroh://node{tag}")).unwrap();
        endpoint.node_id = carrier.map(|key| multikey(&key));
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        let mut body = CapsuleBody::new(vec![signer.pair], vec![service]).unwrap();
        body.also_known_as = Some(vec![classical_alias.to_owned()]);
        let capsule = sign_capsule(body, &signer.ed, &signer.pq).unwrap();
        let bytes = capsule.to_dag_cbor().unwrap();
        let did = format!("did:at9p:{}", capsule.cid512().unwrap());
        (bytes, did)
    }

    fn capsule(classical_alias: &str, tag: u8) -> (Vec<u8>, String) {
        capsule_with_carrier(classical_alias, tag, Some([0xC0; 32]))
    }

    fn anchor_body(classical_alias: &str, signer: &CapsuleSigner) -> CapsuleBody {
        let mut endpoint = ServiceEndpoint::new(Transport::Iroh, "iroh://anchor").unwrap();
        endpoint.node_id = Some(multikey(&[0xC0; 32]));
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        let mut body = CapsuleBody::new(vec![signer.pair.clone()], vec![service]).unwrap();
        body.also_known_as = Some(vec![classical_alias.to_owned()]);
        body
    }

    fn signed_capsule(body: CapsuleBody, signer: &CapsuleSigner) -> (Vec<u8>, String) {
        let capsule = sign_capsule(body, &signer.ed, &signer.pq).unwrap();
        let bytes = capsule.to_dag_cbor().unwrap();
        let did = format!("did:at9p:{}", capsule.cid512().unwrap());
        (bytes, did)
    }

    async fn closed_profile_error(web: &str, bytes: Vec<u8>, at9p: String) -> anyhow::Error {
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
            extra_root_cert_pem: None,
        };
        verify_anchor_material(
            &anchors,
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .expect_err("non-profile anchor capsule unexpectedly accepted")
    }

    fn assert_closed_profile_error(error: &anyhow::Error, field: &str) {
        let chain = format!("{error:#}");
        assert!(
            chain.contains("closed deployment-anchor profile violation"),
            "failure did not come from the closed profile: {chain}"
        );
        assert!(
            chain.contains(field),
            "closed-profile failure did not identify {field}: {chain}"
        );
    }

    fn document(web: &str, at9p: Option<&str>) -> Value {
        let mut document = json!({ "id": web });
        if let Some(at9p) = at9p {
            document["alsoKnownAs"] = json!([at9p]);
        }
        document
    }

    fn document_with_ca_and_reach(
        web: &str,
        at9p: Option<&str>,
        ca: [u8; 32],
        iroh_node: [u8; 32],
    ) -> Value {
        let mut document = json!({
            "id": web,
            "verificationMethod": [{
                "id": format!("{web}#deployment-ca"),
                "type": "Multikey",
                "controller": web,
                "publicKeyMultibase": multikey(&ca),
            }],
            "service": [{
                "id": format!("{web}#iroh"),
                "type": "IrohTransport",
                "serviceEndpoint": hyprstream_rpc::service_entry::encode_iroh(
                    &iroh_node,
                    &[],
                    &["hyprstream-rpc/1"],
                ),
            }],
        });
        if let Some(at9p) = at9p {
            document["alsoKnownAs"] = json!([at9p]);
        }
        document
    }

    fn remove_primary_ml_dsa_key(capsule: &[u8]) -> Vec<u8> {
        let mut value = DagCbor::decode(capsule).expect("test capsule DAG-CBOR");
        let DagCbor::Map(capsule_members) = &mut value else {
            panic!("capsule map");
        };
        let body = capsule_members
            .iter_mut()
            .find_map(|(key, value)| {
                matches!(key, DagCbor::Text(name) if name == "body").then_some(value)
            })
            .expect("capsule body");
        let DagCbor::Map(body_members) = body else {
            panic!("body map");
        };
        let subject_keys = body_members
            .iter_mut()
            .find_map(|(key, value)| {
                matches!(key, DagCbor::Text(name) if name == "subjectKeys").then_some(value)
            })
            .expect("subject keys");
        let DagCbor::List(subject_keys) = subject_keys else {
            panic!("subject key list");
        };
        let DagCbor::Map(primary_members) = subject_keys.first_mut().expect("primary subject key")
        else {
            panic!("primary subject key map");
        };
        primary_members
            .retain(|(key, _)| !matches!(key, DagCbor::Text(name) if name == "mldsa65Pub"));
        value.encode()
    }

    #[test]
    fn unset_anchors_preserve_os_owned_files_selection() {
        assert_eq!(
            DeploymentTrustSource::from_anchors(None, None).unwrap(),
            DeploymentTrustSource::OsOwnedFiles
        );
        assert!(DeploymentTrustSource::from_anchors(Some("did:at9p:x"), None).is_err());
        assert!(DeploymentTrustSource::from_anchors(None, Some("did:web:example.com")).is_err());
    }

    #[tokio::test]
    async fn capsule_hash_mismatch_is_rejected() {
        let web = "did:web:cluster.example";
        let (served_bytes, _served_did) = capsule(web, 1);
        let (_other_bytes, configured_did) = capsule(web, 2);
        let anchors = DidAnchors {
            cluster_at9p_did: configured_did.clone(),
            cluster_did_web: web.to_owned(),
            extra_root_cert_pem: None,
        };
        let error = verify_did_anchored_document(
            &anchors,
            &document(web, Some(&configured_did)),
            Arc::new(FixedCapsuleSource(served_bytes)),
            "unused-test-credential".to_owned(),
            unused_authority_log(),
        )
        .await
        .err()
        .expect("mismatched capsule unexpectedly accepted");
        assert!(error.to_string().contains("mutual-alias"), "{error:#}");
        assert!(format!("{error:#}").contains("hash-gate"), "{error:#}");
    }

    #[tokio::test]
    async fn one_way_alias_is_rejected() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule("did:web:someone-else.example", 3);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
            extra_root_cert_pem: None,
        };
        let error = verify_did_anchored_document(
            &anchors,
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(bytes)),
            "unused-test-credential".to_owned(),
            unused_authority_log(),
        )
        .await
        .err()
        .expect("one-way alias unexpectedly accepted");
        assert!(format!("{error:#}").contains("does not name"), "{error:#}");
    }

    #[tokio::test]
    async fn closed_anchor_profile_preserves_the_exact_minted_deployment_ca() {
        let web = "did:web:cluster.example";
        let signer = capsule_signer(4);
        let mut deployment_ca = signer.pair.ed25519_pub.clone();
        deployment_ca.extend_from_slice(&signer.pair.mldsa65_pub);
        let (bytes, at9p) = signed_capsule(anchor_body(web, &signer), &signer);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
            extra_root_cert_pem: None,
        };
        let verified = verify_anchor_material(
            &anchors,
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .unwrap();
        assert_eq!(verified.at9p_did, at9p);
        assert_eq!(
            verified.deployment_ca_public, deployment_ca,
            "resolved deployment CA must be byte-identical to deployment-ca.hybrid"
        );
        assert_eq!(
            verified.deployment_ca_public.len(),
            32 + ML_DSA65_PUBLIC_KEY_LEN
        );
        match verified.discovery_transport.endpoint {
            EndpointType::Iroh { node_id, .. } => assert_eq!(node_id, [0xC0; 32]),
            other => panic!("expected iroh reach from capsule, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn closed_anchor_profile_rejects_two_subject_keys_signed_by_the_second() {
        let web = "did:web:cluster.example";
        let untrusted_first = capsule_signer(0x41);
        let actual_signer = capsule_signer(0x42);
        let mut body = anchor_body(web, &actual_signer);
        body.subject_keys = vec![untrusted_first.pair, actual_signer.pair.clone()];
        let (bytes, at9p) = signed_capsule(body, &actual_signer);

        verify_did_at9p(&at9p, &bytes)
            .expect("generic set-semantic GATE must accept the second subject as signer");
        let error = closed_profile_error(web, bytes, at9p).await;
        assert_closed_profile_error(&error, "subjectKeys");
    }

    #[tokio::test]
    async fn closed_anchor_profile_rejects_an_extra_service() {
        let web = "did:web:cluster.example";
        let signer = capsule_signer(0x43);
        let mut body = anchor_body(web, &signer);
        let endpoint = ServiceEndpoint::new(Transport::Https, "https://pds.example").unwrap();
        body.services
            .push(ServiceEntry::new("#pds", ServiceType::AtprotoPds, endpoint).unwrap());
        let (bytes, at9p) = signed_capsule(body, &signer);

        verify_did_at9p(&at9p, &bytes).expect("extra-service capsule must pass generic GATE");
        let error = closed_profile_error(web, bytes, at9p).await;
        assert_closed_profile_error(&error, "services");
    }

    #[tokio::test]
    async fn closed_anchor_profile_rejects_an_extra_alias() {
        let web = "did:web:cluster.example";
        let signer = capsule_signer(0x44);
        let mut body = anchor_body(web, &signer);
        body.also_known_as = Some(vec![
            web.to_owned(),
            "did:web:unexpected.example".to_owned(),
        ]);
        let (bytes, at9p) = signed_capsule(body, &signer);

        verify_did_at9p(&at9p, &bytes).expect("extra-alias capsule must pass generic GATE");
        let error = closed_profile_error(web, bytes, at9p).await;
        assert_closed_profile_error(&error, "alsoKnownAs");
    }

    #[tokio::test]
    async fn closed_anchor_profile_rejects_forbidden_signed_authority_metadata() {
        let web = "did:web:cluster.example";
        let signer = capsule_signer(0x45);

        let mut next_key = anchor_body(web, &signer);
        next_key
            .next_key_commitments
            .push(signer.pair.commitment_digest());
        let mut label_hints = anchor_body(web, &signer);
        label_hints.label_hints = Some(vec!["deployment".to_owned()]);
        let mut delegations = anchor_body(web, &signer);
        delegations.delegations = Some(vec![Delegation::new(
            "operator",
            "did:web:delegate.example",
            vec!["admin".to_owned()],
        )
        .unwrap()]);
        let mut witnesses = anchor_body(web, &signer);
        witnesses.witnesses = Some(vec!["did:web:witness.example".to_owned()]);

        for (field, body) in [
            ("nextKeyCommitments", next_key),
            ("labelHints", label_hints),
            ("delegations", delegations),
            ("witnesses", witnesses),
        ] {
            let (bytes, at9p) = signed_capsule(body, &signer);
            verify_did_at9p(&at9p, &bytes)
                .unwrap_or_else(|error| panic!("{field} capsule must pass generic GATE: {error}"));
            let error = closed_profile_error(web, bytes, at9p).await;
            assert_closed_profile_error(&error, field);
        }
    }

    #[tokio::test]
    async fn delegated_registry_token_bootstraps_through_did_anchored_log_and_checkpoint() {
        let web = "did:web:cluster.example";
        let root = capsule_signer(0x51);
        let mut endpoint =
            ServiceEndpoint::new(Transport::Iroh, "iroh://delegated-bootstrap").unwrap();
        endpoint.node_id = Some(multikey(&[0xC1; 32]));
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        let mut body = CapsuleBody::new(vec![root.pair.clone()], vec![service]).unwrap();
        body.also_known_as = Some(vec![web.to_owned()]);
        let capsule = sign_capsule(body, &root.ed, &root.pq).unwrap();
        let capsule_bytes = capsule.to_dag_cbor().unwrap();
        let at9p = format!("did:at9p:{}", capsule.cid512().unwrap());
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
            extra_root_cert_pem: None,
        };
        let root_pq =
            hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&root.pq))
                .unwrap();
        let deployment_domain =
            hyprstream_rpc::auth::composite_kid(&root_pq, &root.ed.verifying_key());
        let (authority_log, checkpoint) = authority_log_and_checkpoint(&root, &deployment_domain);
        let registry = SigningKey::from_bytes(&[0x52; 32]);
        let credential =
            delegated_registry_credential(&root, &authority_log, &deployment_domain, &registry);

        let trust = verify_did_anchored_document(
            &anchors,
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(capsule_bytes)),
            credential,
            authority_log,
        )
        .await
        .unwrap();
        let verifier = crate::service::authenticate_resolved_did_trust(trust, checkpoint)
            .expect("delegated DID-anchored deployment credential");
        assert!(verifier.matches(&registry.verifying_key()));
    }

    #[tokio::test]
    async fn substituted_document_ca_and_reach_are_ignored() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule(web, 4);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
            extra_root_cert_pem: None,
        };

        let honest = verify_did_anchored_document(
            &anchors,
            &document_with_ca_and_reach(web, Some(&at9p), [0x07; 32], [0x45; 32]),
            Arc::new(FixedCapsuleSource(bytes.clone())),
            "unused-test-credential".to_owned(),
            unused_authority_log(),
        )
        .await
        .unwrap();
        let substituted = verify_did_anchored_document(
            &anchors,
            &document_with_ca_and_reach(web, Some(&at9p), [0x66; 32], [0xEE; 32]),
            Arc::new(FixedCapsuleSource(bytes)),
            "unused-test-credential".to_owned(),
            unused_authority_log(),
        )
        .await
        .unwrap();

        assert_eq!(
            honest.ca_verifying_key.ed25519_bytes(),
            substituted.ca_verifying_key.ed25519_bytes()
        );
        assert_eq!(
            honest.ca_verifying_key.ml_dsa_65_bytes(),
            substituted.ca_verifying_key.ml_dsa_65_bytes()
        );
        assert_eq!(honest.discovery_transport, substituted.discovery_transport);
        assert_eq!(honest.ca_verifying_key.ed25519_bytes(), capsule_ca(4));
        match honest.discovery_transport.endpoint {
            EndpointType::Iroh { node_id, .. } => assert_eq!(node_id, [0xC0; 32]),
            other => panic!("expected capsule-bound iroh reach, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn classical_only_capsule_cannot_supply_the_deployment_root() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule(web, 0x46);
        let classical_only = remove_primary_ml_dsa_key(&bytes);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
            extra_root_cert_pem: None,
        };
        let error = verify_did_anchored_document(
            &anchors,
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(classical_only)),
            "unused-test-credential".to_owned(),
            unused_authority_log(),
        )
        .await
        .err()
        .expect("classical-only capsule unexpectedly supplied deployment root");
        assert!(
            format!("{error:#}").contains("mldsa65Pub"),
            "failure did not identify the missing PQ root half: {error:#}"
        );
    }

    #[tokio::test]
    async fn reach_without_carrier_node_id_fails_closed() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule_with_carrier(web, 4, None);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
            extra_root_cert_pem: None,
        };
        let error = verify_did_anchored_document(
            &anchors,
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(bytes)),
            "unused-test-credential".to_owned(),
            unused_authority_log(),
        )
        .await
        .err()
        .expect("carrier-less reach unexpectedly accepted");
        assert!(format!("{error:#}").contains("no independent iroh nodeId"));
    }

    #[tokio::test]
    async fn capsule_bound_quic_reach_is_accepted() {
        let web = "did:web:cluster.example";
        let signer = capsule_signer(5);
        let endpoint = ServiceEndpoint::new(Transport::Quic, "quic://127.0.0.1:7443").unwrap();
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        let mut body = CapsuleBody::new(vec![signer.pair], vec![service]).unwrap();
        body.also_known_as = Some(vec![web.to_owned()]);
        let capsule = sign_capsule(body, &signer.ed, &signer.pq).unwrap();
        let bytes = capsule.to_dag_cbor().unwrap();
        let at9p = format!("did:at9p:{}", capsule.cid512().unwrap());
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
            extra_root_cert_pem: None,
        };
        let channel_auth =
            hyprstream_rpc::transport::QuicServerAuth::pinned(vec![[0xA5; 32]]).unwrap();
        let mut document = document(web, Some(&at9p));
        document["service"] = json!([{
            "id": format!("{web}#quic"),
            "type": "QuicTransport",
            "serviceEndpoint": hyprstream_rpc::service_entry::encode_quic(
                "https://127.0.0.1:7443",
                &channel_auth,
                &["hyprstream-rpc/1"],
            ),
        }]);

        let trust = verify_did_anchored_document(
            &anchors,
            &document,
            Arc::new(FixedCapsuleSource(bytes)),
            "unused-test-credential".to_owned(),
            unused_authority_log(),
        )
        .await
        .unwrap();
        match trust.discovery_transport.endpoint {
            EndpointType::Quic { addr, auth, .. } => {
                assert_eq!(addr, "127.0.0.1:7443".parse().unwrap());
                assert!(!auth.require_web_pki());
                assert_eq!(auth.accept_cert_hashes(), &[[0xA5; 32]]);
            }
            other => panic!("expected capsule-bound QUIC reach, got {other:?}"),
        }
    }

    /// When the document QUIC entry uses a hostname URI (decoding to an
    /// unspecified IP) and the port matches the capsule's real address, the
    /// port-match relaxation must fire AND the returned config must dial the
    /// capsule's real socket address — not `0.0.0.0:port`.
    ///
    /// Removing the address-rewrite in `reach_from_capsule` turns this test
    /// red: the config would dial an unspecified address and never connect.
    #[tokio::test]
    async fn quic_port_match_with_hostname_uri_rewrites_dial_address() {
        let web = "did:web:cluster.example";
        let signer = capsule_signer(7);
        let capsule_addr: SocketAddr = "203.0.113.10:443".parse().unwrap();
        let endpoint =
            ServiceEndpoint::new(Transport::Quic, format!("quic://{capsule_addr}")).unwrap();
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        let mut body = CapsuleBody::new(vec![signer.pair], vec![service]).unwrap();
        body.also_known_as = Some(vec![web.to_owned()]);
        let capsule = sign_capsule(body, &signer.ed, &signer.pq).unwrap();
        let bytes = capsule.to_dag_cbor().unwrap();
        let at9p = format!("did:at9p:{}", capsule.cid512().unwrap());
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
            extra_root_cert_pem: None,
        };
        let channel_auth =
            hyprstream_rpc::transport::QuicServerAuth::pinned(vec![[0xB7; 32]]).unwrap();
        let mut document = document(web, Some(&at9p));
        // Hostname URI decodes to 0.0.0.0:443 — port matches, IP does not.
        document["service"] = json!([{
            "id": format!("{web}#quic"),
            "type": "QuicTransport",
            "serviceEndpoint": hyprstream_rpc::service_entry::encode_quic(
                "https://staging.example.com:443",
                &channel_auth,
                &["hyprstream-rpc/1"],
            ),
        }]);

        let trust = verify_did_anchored_document(
            &anchors,
            &document,
            Arc::new(FixedCapsuleSource(bytes)),
            "unused-test-credential".to_owned(),
            unused_authority_log(),
        )
        .await
        .expect("hostname-URI port match must verify");
        match trust.discovery_transport.endpoint {
            EndpointType::Quic { addr, auth, .. } => {
                assert_eq!(
                    addr, capsule_addr,
                    "dial address must be the capsule's real socket, not 0.0.0.0"
                );
                assert!(!auth.require_web_pki());
                assert_eq!(auth.accept_cert_hashes(), &[[0xB7; 32]]);
            }
            other => panic!("expected capsule-bound QUIC reach, got {other:?}"),
        }
    }

    /// When the document QUIC entry's port does NOT match the capsule's address
    /// (and the IP is unspecified from a hostname URI), no entry matches and the
    /// resolver falls back to a bare-IP dial without SNI or certificate pins.
    #[tokio::test]
    async fn quic_neither_port_nor_address_match_falls_back_to_bare_ip() {
        let web = "did:web:cluster.example";
        let signer = capsule_signer(9);
        let capsule_addr: SocketAddr = "203.0.113.20:443".parse().unwrap();
        let endpoint =
            ServiceEndpoint::new(Transport::Quic, format!("quic://{capsule_addr}")).unwrap();
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        let mut body = CapsuleBody::new(vec![signer.pair], vec![service]).unwrap();
        body.also_known_as = Some(vec![web.to_owned()]);
        let capsule = sign_capsule(body, &signer.ed, &signer.pq).unwrap();
        let bytes = capsule.to_dag_cbor().unwrap();
        let at9p = format!("did:at9p:{}", capsule.cid512().unwrap());
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
            extra_root_cert_pem: None,
        };
        let channel_auth =
            hyprstream_rpc::transport::QuicServerAuth::pinned(vec![[0xC9; 32]]).unwrap();
        let mut document = document(web, Some(&at9p));
        // Port 8443 does not match the capsule's port 443.
        document["service"] = json!([{
            "id": format!("{web}#quic"),
            "type": "QuicTransport",
            "serviceEndpoint": hyprstream_rpc::service_entry::encode_quic(
                "https://staging.example.com:8443",
                &channel_auth,
                &["hyprstream-rpc/1"],
            ),
        }]);

        let trust = verify_did_anchored_document(
            &anchors,
            &document,
            Arc::new(FixedCapsuleSource(bytes)),
            "unused-test-credential".to_owned(),
            unused_authority_log(),
        )
        .await
        .expect("fallback to bare-IP dial must still verify");
        match trust.discovery_transport.endpoint {
            EndpointType::Quic { addr, auth, .. } => {
                // Fallback: bare capsule address, WebPKI default, no cert pins.
                assert_eq!(addr, capsule_addr);
                assert!(auth.require_web_pki());
                assert!(auth.accept_cert_hashes().is_empty());
            }
            other => panic!("expected fallback QUIC reach, got {other:?}"),
        }
    }
}
