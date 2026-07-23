//! DID-anchored deployment trust bootstrap (#1136).
//!
//! This module resolves two public operator pins into verification-only
//! deployment material. Network responses are never authority by themselves:
//! the fetched capsule must pass the canon -> hash -> signature GATE for the
//! configured `did:at9p`, and the fetched `did:web` document must reciprocally
//! name that exact identity before the pair is trusted.
//!
//! # Trust material comes from the GATE-verified capsule, not the document (#1157 / option C)
//!
//! The deployment CA key and the discovery reach are taken from the
//! **GATE-verified `did:at9p` capsule** (`body.subject_keys` / `body.services`),
//! never from the classical `did:web` document. This is what ratified #905 §8
//! requires — *"everything authoritative is created at the GATE (capsule
//! content) ... addresses select; verification asserts"*. The `did:web`
//! document is used **only** for the #905 §2 leg-2 reciprocal identifier vouch
//! (its `alsoKnownAs` must name the configured `did:at9p`) and, otherwise, as an
//! advisory discovery/rotation hint. An adversary who controls the `did:web`
//! origin can no longer substitute the CA or reach: both are content-bound to
//! the hash-pinned capsule (Erica's decision, issue #1157, 2026-07-22).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use ed25519_dalek::VerifyingKey;
use hyprstream_crypto::pq::{ml_dsa_vk_from_bytes, MlDsaVerifyingKey};
use hyprstream_pds::at9p::{ServiceType, Transport as At9pTransport};
use hyprstream_pds::at9p_alias::AuthoritativeIdentity;
use hyprstream_pds::at9p_duplicity::AcceptedAt9pState;
use hyprstream_pds::at9p_gate::VerifiedCapsule;
use hyprstream_rpc::auth::mac::Assurance;
use hyprstream_rpc::did_web::{did_web_to_url, DidWebResolver, HttpDidDocFetcher};
use hyprstream_rpc::identity::Did;
use hyprstream_rpc::transport::{EndpointType, TransportConfig};
use serde_json::Value;

use crate::at9p_alias::At9pAliasResolver;
use crate::at9p_resolver::CapsuleSource;

const MAX_CAPSULE_BYTES: usize = 4 * 1024 * 1024;

/// The at9p service selector whose typed reach is installed as the deployment
/// Discovery transport. `#ns` is the canonical `NinePExport` entry (mirrors
/// [`crate::at9p_resolver::DEFAULT_AT9P_SERVICE`]).
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

/// An atomic Ed25519 + ML-DSA-65 deployment-credential authority.
///
/// The pair is intentionally indivisible at the consumer boundary.  Carrying
/// only `ed25519` would let a later EdDSA-only credential erase the GATE's PQ
/// half while retaining a misleading `PqHybrid` label.
#[derive(Clone)]
pub(crate) struct HybridDeploymentCa {
    pub ed25519: VerifyingKey,
    pub ml_dsa_65: MlDsaVerifyingKey,
}

impl HybridDeploymentCa {
    /// Stable identity for one complete Hybrid authority pair.  This must not
    /// be derived from only Ed25519: a bounded overlap may rotate ML-DSA-65
    /// while intentionally retaining the classical component.
    pub(crate) fn key_id(&self) -> String {
        hyprstream_rpc::auth::composite_kid(&self.ml_dsa_65, &self.ed25519)
    }
}

/// Verified public material extracted from a mutually-attested identity pair.
///
/// Both `ca_verifying_key` and `discovery_transport` are sourced from the
/// GATE-verified capsule, never the `did:web` document (#1157 / option C).
pub(crate) struct DidAnchoredTrust {
    /// Every currently published capsule CA pair.  Credential `kid` selects a
    /// member; this is deliberately a key set, never a positional singleton.
    pub ca_keys: Vec<HybridDeploymentCa>,
    pub discovery_transport: TransportConfig,
    pub authoritative_identity: AuthoritativeIdentity,
    /// The assurance of the trust material actually installed. Because the CA is
    /// selected from the capsule's hybrid `subject_keys`, this is `PqHybrid` —
    /// carried, not reconstructed (#556 / F5). A capsule GATE-verifies only under
    /// pinned Hybrid, so this is never `Classical`.
    pub assurance: Assurance,
    /// The GATE witness for the immutable genesis. Bootstrap replays the
    /// daemon-owned accepted history from this anchor before choosing a live CA.
    pub genesis: VerifiedCapsule,
}

impl DidAnchoredTrust {
    /// Replace the genesis seed projection with the daemon-authenticated live
    /// accepted state.  Genesis establishes the identity only; current CA keys
    /// and reach are always projected from the checkpointed state that follows.
    pub(crate) fn from_accepted_state(seed: Self, state: &AcceptedAt9pState) -> Result<Self> {
        anyhow::ensure!(
            state.did == seed.authoritative_identity.at9p_did.as_str(),
            "accepted did:at9p state does not match the configured deployment anchor"
        );
        let ca_keys = ca_keys_from_body(&state.current)?;
        let discovery_transport = reach_from_body(&state.current)?;
        Ok(Self {
            ca_keys,
            discovery_transport,
            authoritative_identity: seed.authoritative_identity,
            assurance: seed.assurance,
            genesis: seed.genesis,
        })
    }
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

/// Decode every GATE-authenticated deployment CA pair from the capsule.
///
/// `subject_keys` has set semantics for consumers: multiple live keys are
/// allowed during overlap and the credential's authenticated `kid` selects the
/// exact pair.  Neither this function nor its callers use `.first()`.
fn ca_keys_from_body(body: &hyprstream_pds::at9p::CapsuleBody) -> Result<Vec<HybridDeploymentCa>> {
    let keys =
        body.subject_keys
            .iter()
            .map(|pair| {
                let ed: [u8; 32] =
                    pair.ed25519_pub.as_slice().try_into().map_err(|_| {
                        anyhow::anyhow!("capsule subject Ed25519 key is not 32 bytes")
                    })?;
                Ok(HybridDeploymentCa {
                    ed25519: VerifyingKey::from_bytes(&ed)
                        .context("capsule subject Ed25519 key is malformed")?,
                    ml_dsa_65: ml_dsa_vk_from_bytes(&pair.mldsa65_pub)
                        .context("capsule subject ML-DSA-65 key is malformed")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(!keys.is_empty(), "GATE-verified capsule has no subject key");
    for (index, key) in keys.iter().enumerate() {
        anyhow::ensure!(
            !keys[..index]
                .iter()
                .any(|prior| prior.key_id() == key.key_id()),
            "GATE-verified capsule repeats a deployment CA Hybrid key pair"
        );
    }
    Ok(keys)
}

fn ca_keys_from_capsule(verified: &VerifiedCapsule) -> Result<Vec<HybridDeploymentCa>> {
    ca_keys_from_body(&verified.capsule().body)
}

/// The discovery reach is the `#ns` `NinePExport` service entry's typed iroh
/// endpoint, taken from the GATE-verified capsule.  Iroh requires an
/// independent `nodeId`; QUIC uses its signed socket carrier directly.
///
/// # Reach vs identity (#1031)
///
/// Reach is an **explicit carrier address** (`ServiceEndpoint.node_id`), never
/// derived from the subject identity key — the invariant #1031 protects and the
/// reason the D1 `capsule_to_iroh_reach` fails closed when only a relay claim is
/// present. A capsule that publishes an independent `nodeId` carrier supplies a
/// dialable address; the installed reach is still an untrusted carrier, so
/// `bootstrap_deployment_process` pins the response to the separately resolved
/// Discovery key and requires a signed liveness `ping` before the resolver is
/// installed. A missing/non-iroh/carrier-less entry fails closed.
fn reach_from_body(body: &hyprstream_pds::at9p::CapsuleBody) -> Result<TransportConfig> {
    let entry = body
        .services
        .iter()
        .find(|s| s.id == DEPLOYMENT_REACH_SERVICE && s.service_type == ServiceType::NinePExport)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "capsule has no NinePExport service entry {DEPLOYMENT_REACH_SERVICE:?} for deployment reach"
            )
        })?;
    match entry.endpoint.transport {
        At9pTransport::Iroh => {
            // The independent carrier EndpointId — an explicit transport
            // address, not the subject identity key (#1031).
            let node_id_multibase = entry.endpoint.node_id.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "capsule deployment reach {DEPLOYMENT_REACH_SERVICE:?} carries no independent iroh \
                     nodeId carrier; refusing to derive reach from the subject identity key (#1031)"
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
            // The schema has one signed carrier string.  Requiring an IP socket
            // avoids a second unauthenticated DNS resolution surface; the TLS
            // certificate must therefore contain that IP as a SAN.
            Ok(TransportConfig::quic(address, address.ip().to_string()).with_connect_mode())
        }
        ref other => bail!(
            "capsule deployment reach {DEPLOYMENT_REACH_SERVICE:?} is not an iroh or QUIC endpoint (got {other:?})"
        ),
    }
}

fn reach_from_capsule(verified: &VerifiedCapsule) -> Result<TransportConfig> {
    reach_from_body(&verified.capsule().body)
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
    // The mutual-alias rule returns the authoritative identity AND the
    // GATE-verified capsule the attestation was proven against. All trust
    // material below comes from that capsule — never the did:web document.
    let (authoritative_identity, verified) = At9pAliasResolver::new(capsule_source)
        .resolve_authoritative_with_capsule(&classical, &at9p)
        .await
        .context("DID deployment anchors failed mutual-alias verification")?;
    anyhow::ensure!(
        authoritative_identity.at9p_did == at9p,
        "mutual-alias resolver did not preserve configured at9p authority"
    );

    // CA + reach from the GATE-verified capsule (option C). The did:web document
    // is used only for the reciprocal identifier vouch checked above.
    let ca_keys = ca_keys_from_capsule(&verified)?;
    let discovery_transport = reach_from_capsule(&verified)?;
    anyhow::ensure!(
        matches!(
            discovery_transport.endpoint,
            EndpointType::Iroh { .. } | EndpointType::Quic { .. }
        ),
        "capsule deployment reach is not a network transport"
    );

    Ok(DidAnchoredTrust {
        ca_keys,
        discovery_transport,
        // The capsule leg was walked (GATE = pinned Hybrid) and the CA is the
        // capsule's hybrid subject key ⇒ PqHybrid, carried not reconstructed.
        assurance: authoritative_identity.assurance,
        authoritative_identity,
        genesis: verified,
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
        "verified mutually-attested DID deployment trust anchors (CA + reach from GATE-verified capsule)"
    );
    Ok(trust)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};
    use hyprstream_pds::at9p::h512;
    use hyprstream_pds::at9p::{
        CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
    };
    use hyprstream_pds::at9p_duplicity::{DuplicityGuard, InMemoryWatermarkStore};
    use hyprstream_pds::at9p_gate::verify_did_at9p;
    use hyprstream_pds::at9p_sign::{sign_capsule, sign_update_record};
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

    /// Encode a raw Ed25519 key as a `Multikey` `publicKeyMultibase` string (the
    /// carrier `nodeId` form the capsule schema stores).
    fn multikey(key: &[u8; 32]) -> String {
        hyprstream_rpc::did_key::ed25519_to_did_key(key)
            .strip_prefix("did:key:")
            .unwrap()
            .to_owned()
    }

    /// The Ed25519 verifying key of the capsule signer for `tag` — i.e. the
    /// capsule's primary `subject_keys[0]`, which option C installs as the CA.
    fn capsule_ca(tag: u8) -> [u8; 32] {
        SigningKey::from_bytes(&[tag; 32])
            .verifying_key()
            .to_bytes()
    }

    /// A signed capsule that names `classical_alias` (leg 2) and publishes an
    /// `#ns` NinePExport iroh service. `carrier` sets the independent iroh
    /// `nodeId` on that service (the #1031 explicit reach carrier); `None`
    /// leaves it absent so reach derivation must fail closed.
    fn capsule_with_carrier(
        classical_alias: &str,
        tag: u8,
        carrier: Option<[u8; 32]>,
    ) -> (Vec<u8>, String) {
        let signer = capsule_signer(tag);
        let mut endpoint =
            ServiceEndpoint::new(Transport::Iroh, format!("iroh://node{tag}")).unwrap();
        endpoint.node_id = carrier.map(|c| multikey(&c));
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        let mut body = CapsuleBody::new(vec![signer.pair], vec![service]).unwrap();
        body.also_known_as = Some(vec![classical_alias.to_owned()]);
        let capsule = sign_capsule(body, &signer.ed, &signer.pq).unwrap();
        let bytes = capsule.to_dag_cbor().unwrap();
        let did = format!("did:at9p:{}", capsule.cid512().unwrap());
        (bytes, did)
    }

    /// The mainline fixture: capsule with a valid reach carrier (tag `0xC0`).
    fn capsule(classical_alias: &str, tag: u8) -> (Vec<u8>, String) {
        capsule_with_carrier(classical_alias, tag, Some([0xC0; 32]))
    }

    fn capsule_with_ca_overlap(
        classical_alias: &str,
        old: &CapsuleSigner,
        new: &CapsuleSigner,
    ) -> (Vec<u8>, String) {
        let mut endpoint = ServiceEndpoint::new(Transport::Iroh, "iroh://overlap").unwrap();
        endpoint.node_id = Some(multikey(&[0xC0; 32]));
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        let mut body =
            CapsuleBody::new(vec![old.pair.clone(), new.pair.clone()], vec![service]).unwrap();
        body.also_known_as = Some(vec![classical_alias.to_owned()]);
        let capsule = sign_capsule(body, &old.ed, &old.pq).unwrap();
        let bytes = capsule.to_dag_cbor().unwrap();
        let did = format!("did:at9p:{}", capsule.cid512().unwrap());
        (bytes, did)
    }

    fn capsule_with_quic_reach(classical_alias: &str, signer: &CapsuleSigner) -> (Vec<u8>, String) {
        let endpoint = ServiceEndpoint::new(Transport::Quic, "quic://127.0.0.1:4433").unwrap();
        let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
        let mut body = CapsuleBody::new(vec![signer.pair.clone()], vec![service]).unwrap();
        body.also_known_as = Some(vec![classical_alias.to_owned()]);
        let capsule = sign_capsule(body, &signer.ed, &signer.pq).unwrap();
        let bytes = capsule.to_dag_cbor().unwrap();
        let did = format!("did:at9p:{}", capsule.cid512().unwrap());
        (bytes, did)
    }

    /// A minimal did:web document: only the fields option C still reads — the
    /// `id` and the reciprocal `alsoKnownAs` identifier vouch. No key material.
    fn document(web: &str, at9p: Option<&str>) -> Value {
        let mut document = json!({ "id": web });
        if let Some(at9p) = at9p {
            document["alsoKnownAs"] = json!([at9p]);
        }
        document
    }

    /// A did:web document that ALSO carries CA key material and reach — the
    /// shape the pre-option-C code read from. Used to prove those fields are now
    /// ignored (the F1 substitution regression test).
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
        // The singleton fixture publishes the capsule's one complete hybrid CA
        // pair, never a document key.
        assert_eq!(trust.ca_keys.len(), 1);
        assert_eq!(trust.ca_keys[0].ed25519.to_bytes(), capsule_ca(4));
        // Reach is the capsule's carrier nodeId.
        match trust.discovery_transport.endpoint {
            EndpointType::Iroh { node_id, .. } => assert_eq!(node_id, [0xC0; 32]),
            other => panic!("expected iroh reach from capsule, got {other:?}"),
        }
    }

    // ── option C properties (issue #1157) ────────────────────────────────────

    /// F5 regression guard (#556): the installed authority carries the capsule's
    /// hybrid assurance and is NEVER `Classical`. A capsule GATE-verifies only
    /// under pinned Hybrid, so its subject key carries a bound ML-DSA-65 half;
    /// selecting the CA from it means `PqHybrid` is carried, not reconstructed.
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
        assert_ne!(
            trust.assurance,
            Assurance::Classical,
            "deployment CA must not land classical — it anchors the registry/audit/checkpoint chain (#556)"
        );
        assert_eq!(trust.assurance, Assurance::PqHybrid);
    }

    /// F4 subsumed: the `did:web` document may publish two Ed25519 verification
    /// methods (new CA alongside old — the normal overlap rotation) and bootstrap
    /// still succeeds. Under the old `ca_keys.len() == 1` rule this made
    /// `len() == 2` and broke bootstrap for every node at once. Option C never
    /// reads document keys, so overlap is a non-event.
    #[tokio::test]
    async fn rotation_with_overlapping_document_cas_still_bootstraps() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule(web, 4);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
        };
        // Document publishes TWO Ed25519 VMs (old + new CA overlapping).
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
                .expect("overlapping document CAs must not break bootstrap (F4 subsumed)");
        // The installed CA is still the capsule key, unaffected by the document.
        assert_eq!(trust.ca_keys.len(), 1);
        assert_eq!(trust.ca_keys[0].ed25519.to_bytes(), capsule_ca(4));
    }

    /// The capsule key set itself, rather than document noise, is the live
    /// authority set consumed by credential verification.  The companion
    /// service test authenticates under both pairs and then rejects the
    /// retired pair after the set is advanced.
    #[tokio::test]
    async fn capsule_ca_overlap_preserves_both_atomic_hybrid_pairs() {
        let web = "did:web:cluster.example";
        let old = capsule_signer(0x41);
        let new = capsule_signer(0x42);
        let (bytes, at9p) = capsule_with_ca_overlap(web, &old, &new);
        let trust = verify_did_anchored_document(
            &DidAnchors {
                cluster_at9p_did: at9p.clone(),
                cluster_did_web: web.to_owned(),
            },
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .expect("two published capsule CAs must remain a usable key set");
        assert_eq!(trust.ca_keys.len(), 2);
        assert!(trust
            .ca_keys
            .iter()
            .any(|key| key.ed25519 == old.ed.verifying_key()));
        assert!(trust
            .ca_keys
            .iter()
            .any(|key| key.ed25519 == new.ed.verifying_key()));
    }

    #[tokio::test]
    async fn accepted_current_state_retires_genesis_ca_and_updates_reach() {
        let web = "did:web:cluster.example";
        let old = capsule_signer(0x51);
        let new = capsule_signer(0x52);
        let service = |address: &str| {
            let endpoint = ServiceEndpoint::new(Transport::Quic, address).unwrap();
            ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap()
        };
        let mut genesis_body = CapsuleBody::new(
            vec![old.pair.clone()],
            vec![service("quic://127.0.0.1:4433")],
        )
        .unwrap();
        genesis_body.also_known_as = Some(vec![web.to_owned()]);
        genesis_body.next_key_commitments = vec![new.pair.commitment_digest()];
        let genesis = sign_capsule(genesis_body, &old.ed, &old.pq).unwrap();
        let genesis_bytes = genesis.to_dag_cbor().unwrap();
        let did = format!("did:at9p:{}", genesis.cid512().unwrap());
        let anchors = DidAnchors {
            cluster_at9p_did: did.clone(),
            cluster_did_web: web.to_owned(),
        };
        let seed = verify_did_anchored_document(
            &anchors,
            &document(web, Some(&did)),
            Arc::new(FixedCapsuleSource(genesis_bytes.clone())),
        )
        .await
        .expect("GATE-verified genesis seed");

        let mut current = CapsuleBody::new(
            vec![new.pair.clone()],
            vec![service("quic://127.0.0.2:4434")],
        )
        .unwrap();
        current.also_known_as = Some(vec![web.to_owned()]);
        let update = sign_update_record(
            genesis.cid512().unwrap(),
            1,
            h512(&genesis_bytes),
            current,
            "2099-01-01T00:00:00Z".to_owned(),
            &new.ed,
            &new.pq,
        )
        .expect("signed successor");
        let guard = DuplicityGuard::new(InMemoryWatermarkStore::new());
        let verified = verify_did_at9p(&did, &genesis_bytes).expect("GATE genesis");
        guard.seed_genesis(&verified).expect("seed genesis");
        guard
            .admit_successor(&update, "2026-07-23T00:00:00Z")
            .expect("advance accepted state");
        let state = guard
            .accepted_state(
                did.strip_prefix(hyprstream_pds::at9p_gate::DID_AT9P_PREFIX)
                    .expect("at9p DID"),
            )
            .expect("read accepted state")
            .expect("state present after advance");
        let live =
            DidAnchoredTrust::from_accepted_state(seed, &state).expect("live state projection");

        assert_eq!(live.ca_keys.len(), 1);
        assert_eq!(live.ca_keys[0].ed25519, new.ed.verifying_key());
        assert_ne!(live.ca_keys[0].ed25519, old.ed.verifying_key());
        assert!(matches!(
            live.discovery_transport.endpoint,
            EndpointType::Quic { ref addr, .. } if addr.port() == 4434
        ));
    }

    #[tokio::test]
    async fn signed_capsule_quic_reach_remains_accepted() {
        let web = "did:web:cluster.example";
        let signer = capsule_signer(0x43);
        let (bytes, at9p) = capsule_with_quic_reach(web, &signer);
        let trust = verify_did_anchored_document(
            &DidAnchors {
                cluster_at9p_did: at9p.clone(),
                cluster_did_web: web.to_owned(),
            },
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .expect("signed QUIC deployment reach must retain the previous transport contract");
        assert!(matches!(
            trust.discovery_transport.endpoint,
            EndpointType::Quic { .. }
        ));
    }

    /// F1 fix regression: substituting the `did:web` document's CA key AND reach
    /// no longer changes the installed trust material, because neither is read
    /// from the document. This is the opus-verdict attack, now neutralized.
    #[tokio::test]
    async fn substituted_document_ca_and_reach_are_ignored() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule(web, 4);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
        };

        // Honest document carries one CA + reach; attacker document carries
        // entirely different ones. Same capsule bytes (the public pin) both times.
        let honest = verify_did_anchored_document(
            &anchors,
            &document_with_ca_and_reach(web, Some(&at9p), [0x7; 32], [0x45; 32]),
            Arc::new(FixedCapsuleSource(bytes.clone())),
        )
        .await
        .unwrap();
        let evil = verify_did_anchored_document(
            &anchors,
            &document_with_ca_and_reach(web, Some(&at9p), [0x66; 32], [0xEE; 32]),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .expect("document contents are advisory; verification still succeeds");

        // Identical installed trust despite different document CA + reach.
        assert_eq!(honest.ca_keys.len(), 1);
        assert_eq!(honest.ca_keys[0].ed25519, evil.ca_keys[0].ed25519);
        assert_eq!(honest.discovery_transport, evil.discovery_transport);
        // And it is the capsule's material, not either document's.
        assert_eq!(honest.ca_keys[0].ed25519.to_bytes(), capsule_ca(4));
        assert_ne!(honest.ca_keys[0].ed25519.to_bytes(), [0x7; 32]);
    }

    /// F1 target (was #[ignore]'d expected-fail on #1143; now passes): the
    /// installed CA is bound to the GATE-verified capsule, even when the document
    /// publishes a different key.
    #[tokio::test]
    async fn ca_is_bound_to_verified_capsule_not_document() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule(web, 4);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
        };
        // Document publishes an unrelated CA; it must be ignored.
        let doc_ca = [0x7; 32];
        assert_ne!(capsule_ca(4), doc_ca, "fixture sanity");
        let trust = verify_did_anchored_document(
            &anchors,
            &document_with_ca_and_reach(web, Some(&at9p), doc_ca, [0x45; 32]),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .unwrap();
        assert_eq!(
            trust.ca_keys[0].ed25519.to_bytes(),
            capsule_ca(4),
            "installed CA must derive from the GATE-verified capsule, not the did:web document"
        );
    }

    /// Reach is an explicit carrier address, never the subject identity key
    /// (#1031). A capsule whose `#ns` service has no independent `nodeId` carrier
    /// fails closed rather than deriving reach from the genesis subject key.
    #[tokio::test]
    async fn reach_without_carrier_node_id_fails_closed() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule_with_carrier(web, 4, None);
        let anchors = DidAnchors {
            cluster_at9p_did: at9p.clone(),
            cluster_did_web: web.to_owned(),
        };
        let err = verify_did_anchored_document(
            &anchors,
            &document(web, Some(&at9p)),
            Arc::new(FixedCapsuleSource(bytes)),
        )
        .await
        .err()
        .expect("carrier-less reach must fail closed (#1031)");
        assert!(
            format!("{err:#}").contains("no independent iroh"),
            "{err:#}"
        );
    }

    /// Evidence retained from the #1143 analysis: the capsule carries the
    /// material the document should not own (subject_keys + services). This is
    /// why option C is wiring, not new crypto.
    #[tokio::test]
    async fn f1_capsule_carries_the_material_the_document_should_not_own() {
        let web = "did:web:cluster.example";
        let (bytes, at9p) = capsule(web, 4);
        let verified = hyprstream_pds::at9p_gate::verify_did_at9p(&at9p, &bytes)
            .expect("capsule GATE-verifies");
        assert!(
            !verified.capsule().body.subject_keys.is_empty(),
            "capsule carries hybrid subject keys — an in-band CA source"
        );
        assert!(
            !verified.capsule().body.services.is_empty(),
            "capsule carries typed services — an in-band reach source"
        );
    }
}
