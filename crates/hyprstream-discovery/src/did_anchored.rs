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
use ed25519_dalek::VerifyingKey;
use hyprstream_pds::at9p::{ServiceType, Transport as At9pTransport};
use hyprstream_pds::at9p_alias::AuthoritativeIdentity;
use hyprstream_pds::at9p_gate::VerifiedCapsule;
use hyprstream_rpc::auth::mac::Assurance;
use hyprstream_rpc::did_web::{did_web_to_url, DidWebResolver, HttpDidDocFetcher};
use hyprstream_rpc::identity::Did;
use hyprstream_rpc::transport::{EndpointType, TransportConfig};
use serde_json::Value;

use crate::at9p_alias::At9pAliasResolver;
use crate::at9p_resolver::CapsuleSource;

const MAX_CAPSULE_BYTES: usize = 4 * 1024 * 1024;
const DEPLOYMENT_REACH_SERVICE: &str = "#ns";

/// The two public, non-secret anchors for DID-backed deployment trust.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DidAnchors {
    pub cluster_at9p_did: String,
    pub cluster_did_web: String,
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
                }))
            }
            _ => bail!(
                "cluster_at9p_did and cluster_did_web must be configured together; partial DID trust configuration is forbidden"
            ),
        }
    }
}

/// Verified public material extracted from a mutually-attested identity pair.
///
/// The CA and transport are sourced from the GATE-verified capsule, never the
/// `did:web` document.
pub(crate) struct DidAnchoredTrust {
    pub ca_verifying_key: VerifyingKey,
    pub discovery_transport: TransportConfig,
    pub authoritative_identity: AuthoritativeIdentity,
    /// Assurance carried by the capsule's hybrid subject key.
    pub assurance: Assurance,
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

impl HttpWellKnownCapsuleSource {
    fn new(did_web: &str) -> Result<Self> {
        let document_url = did_web_to_url(did_web)?;
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(10))
            .build()
            .context("failed to build at9p capsule HTTPS client")?;
        Ok(Self { http, document_url })
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

/// Extract the deployment CA from the primary hybrid subject key that signed
/// the GATE-verified capsule.
fn ca_key_from_capsule(verified: &VerifiedCapsule) -> Result<VerifyingKey> {
    let primary = verified
        .capsule()
        .body
        .subject_keys
        .first()
        .ok_or_else(|| anyhow::anyhow!("GATE-verified capsule has no subject key"))?;
    let ed: [u8; 32] = primary
        .ed25519_pub
        .as_slice()
        .try_into()
        .map_err(|_| anyhow::anyhow!("capsule subject Ed25519 key is not 32 bytes"))?;
    VerifyingKey::from_bytes(&ed).context("capsule subject Ed25519 key is malformed")
}

/// Extract Discovery reach from the capsule's typed `#ns` service. The
/// independent nodeId is transport reach only; the signed ping remains pinned
/// to the separately authenticated Discovery application key.
fn reach_from_capsule(verified: &VerifiedCapsule) -> Result<TransportConfig> {
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
                "capsule has no NinePExport service entry {DEPLOYMENT_REACH_SERVICE:?} for deployment reach"
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
            Ok(TransportConfig::quic(address, address.ip().to_string()).with_connect_mode())
        }
        ref other => bail!(
            "capsule deployment reach {DEPLOYMENT_REACH_SERVICE:?} is not an iroh or QUIC endpoint (got {other:?})"
        ),
    }
}

pub(crate) async fn verify_did_anchored_document(
    anchors: &DidAnchors,
    document: &Value,
    capsule_source: Arc<dyn CapsuleSource>,
) -> Result<DidAnchoredTrust> {
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

    // The document contributes only the reciprocal identifier vouch above.
    // Everything installed is content-bound to the configured did:at9p pin.
    let ca_verifying_key = ca_key_from_capsule(&verified)?;
    let discovery_transport = reach_from_capsule(&verified)?;
    anyhow::ensure!(
        matches!(
            discovery_transport.endpoint,
            EndpointType::Iroh { .. } | EndpointType::Quic { .. }
        ),
        "capsule deployment reach is not a network transport"
    );

    Ok(DidAnchoredTrust {
        ca_verifying_key,
        discovery_transport,
        assurance: authoritative_identity.assurance,
        authoritative_identity,
    })
}

pub(crate) async fn resolve_did_anchored_trust(anchors: &DidAnchors) -> Result<DidAnchoredTrust> {
    let document = DidWebResolver::new(HttpDidDocFetcher::new(Duration::from_secs(3600))?)
        .resolve_document(&anchors.cluster_did_web)
        .await
        .context("failed to fetch configured cluster did:web document")?;
    let capsule_source: Arc<dyn CapsuleSource> =
        Arc::new(HttpWellKnownCapsuleSource::new(&anchors.cluster_did_web)?);
    let trust = verify_did_anchored_document(anchors, &document, capsule_source).await?;
    tracing::info!(
        at9p = %trust.authoritative_identity.at9p_did,
        did_web = %trust.authoritative_identity.classical_did,
        assurance = ?trust.assurance,
        "verified mutually-attested DID deployment trust anchors from GATE-verified capsule"
    );
    Ok(trust)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};
    use hyprstream_pds::at9p::{
        CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
    };
    use hyprstream_pds::at9p_sign::sign_capsule;
    use serde_json::json;

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
        };
        let error = verify_did_anchored_document(
            &anchors,
            &document(web, Some(&configured_did)),
            Arc::new(FixedCapsuleSource(served_bytes)),
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
        };
        let error = verify_did_anchored_document(
            &anchors,
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .err()
        .expect("one-way alias unexpectedly accepted");
        assert!(format!("{error:#}").contains("does not name"), "{error:#}");
    }

    #[tokio::test]
    async fn mutual_alias_accepts_at9p_as_authoritative() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule(web, 4);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
        };
        let trust = verify_did_anchored_document(
            &anchors,
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .unwrap();
        assert_eq!(trust.authoritative_identity.at9p_did.as_str(), at9p);
        assert_eq!(trust.authoritative_identity.classical_did.as_str(), web);
        assert_eq!(trust.ca_verifying_key.to_bytes(), capsule_ca(4));
        match trust.discovery_transport.endpoint {
            EndpointType::Iroh { node_id, .. } => assert_eq!(node_id, [0xC0; 32]),
            other => panic!("expected iroh reach from capsule, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn installed_authority_is_pqhybrid_not_classical() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule(web, 4);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
        };
        let trust = verify_did_anchored_document(
            &anchors,
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .unwrap();
        assert_ne!(trust.assurance, Assurance::Classical);
        assert_eq!(trust.assurance, Assurance::PqHybrid);
    }

    #[tokio::test]
    async fn rotation_with_overlapping_document_cas_still_bootstraps() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule(web, 4);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
        };
        let mut doc = document(web, Some(&at9p));
        doc["verificationMethod"] = json!([
            {
                "id": format!("{web}#deployment-ca-old"),
                "type": "Multikey",
                "controller": web,
                "publicKeyMultibase": multikey(&[0x11; 32]),
            },
            {
                "id": format!("{web}#deployment-ca-new"),
                "type": "Multikey",
                "controller": web,
                "publicKeyMultibase": multikey(&[0x22; 32]),
            },
        ]);
        let trust =
            verify_did_anchored_document(&anchors, &doc, Arc::new(FixedCapsuleSource(bytes)))
                .await
                .expect("overlapping document CAs must not break bootstrap");
        assert_eq!(trust.ca_verifying_key.to_bytes(), capsule_ca(4));
    }

    #[tokio::test]
    async fn substituted_document_ca_and_reach_are_ignored() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule(web, 4);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
        };

        let honest = verify_did_anchored_document(
            &anchors,
            &document_with_ca_and_reach(web, Some(&at9p), [0x07; 32], [0x45; 32]),
            Arc::new(FixedCapsuleSource(bytes.clone())),
        )
        .await
        .unwrap();
        let substituted = verify_did_anchored_document(
            &anchors,
            &document_with_ca_and_reach(web, Some(&at9p), [0x66; 32], [0xEE; 32]),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .unwrap();

        assert_eq!(honest.ca_verifying_key, substituted.ca_verifying_key);
        assert_eq!(honest.discovery_transport, substituted.discovery_transport);
        assert_eq!(honest.ca_verifying_key.to_bytes(), capsule_ca(4));
        match honest.discovery_transport.endpoint {
            EndpointType::Iroh { node_id, .. } => assert_eq!(node_id, [0xC0; 32]),
            other => panic!("expected capsule-bound iroh reach, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn ca_is_bound_to_verified_capsule_not_document() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule(web, 4);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
        };
        let document_ca = [0x07; 32];
        assert_ne!(capsule_ca(4), document_ca);
        let trust = verify_did_anchored_document(
            &anchors,
            &document_with_ca_and_reach(web, Some(&at9p), document_ca, [0x45; 32]),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .unwrap();
        assert_eq!(trust.ca_verifying_key.to_bytes(), capsule_ca(4));
    }

    #[tokio::test]
    async fn reach_without_carrier_node_id_fails_closed() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule_with_carrier(web, 4, None);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
        };
        let error = verify_did_anchored_document(
            &anchors,
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .err()
        .expect("carrier-less reach unexpectedly accepted");
        assert!(format!("{error:#}").contains("no independent iroh nodeId"));
    }

    #[tokio::test]
    async fn capsule_carries_authoritative_keys_and_services() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule(web, 4);
        let verified = hyprstream_pds::at9p_gate::verify_did_at9p(&at9p, &bytes)
            .expect("capsule GATE-verifies");
        assert!(!verified.capsule().body.subject_keys.is_empty());
        assert!(!verified.capsule().body.services.is_empty());
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
        };

        let trust = verify_did_anchored_document(
            &anchors,
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .unwrap();
        match trust.discovery_transport.endpoint {
            EndpointType::Quic { addr, .. } => {
                assert_eq!(addr, "127.0.0.1:7443".parse().unwrap());
            }
            other => panic!("expected capsule-bound QUIC reach, got {other:?}"),
        }
    }
}
