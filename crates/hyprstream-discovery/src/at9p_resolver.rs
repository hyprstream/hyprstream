//! `did:at9p` capsule resolver — GATE-verified reach as a `TransportConfig`
//! (#893, design #879 §9; epic #880 Track D / D1).
//!
//! This is the **D1 sibling to `hyprstream_rpc::service_entry::decode_iroh`**:
//! where `decode_iroh` turns a DID-document `IrohTransport` service entry into a
//! dialable [`TransportConfig`], this module turns a *GATE-verified `did:at9p`
//! capsule* into the same shape. It is explicitly **not** a [`RecordResolver`]
//! impl — a `RecordResolver` returns an atproto CAR (the wrong output shape for
//! "how do I dial this peer"). The capsule→reach path is a separate resolution
//! arm, injected at the network-profile resolver seam (`Resolver::set_global`,
//! the future `NetworkDiscoveryResolver` of #873).
//!
//! # Why reach comes from the capsule, not an endpoint list
//!
//! A `did:at9p:<cid512>` is self-certifying: the identity *is* the BLAKE3-512
//! hash of the genesis capsule. After [`verify_did_at9p`] passes all three gates
//! (canon → hash → sig), every field on the capsule is content-bound to that
//! identity — an attacker cannot substitute keys or endpoints without changing
//! the cid. The resolver therefore trusts **only** what the GATE verified, and
//! fails closed otherwise: no GATE witness, no reach.
//!
//! The reach it emits is the iroh dial the existing transport already speaks:
//!
//! - `node_id` — the capsule's **primary subject Ed25519 key**. On iroh the
//!   `EndpointId` *is* the Ed25519 pubkey, so this is the authoritative NodeId,
//!   taken from the GATE-verified [`Capsule`] (not from a self-declared string
//!   on the endpoint). The ed25519 half already drives the existing iroh dial
//!   path verbatim — this is additive.
//! - `relay_url` — the relay hint carried by the selected 9P-export service
//!   entry's iroh endpoint.
//!
//! # Service selection (#893 update; design #905 §4, #879 §5.1a)
//!
//! A capsule carries a **map** of named services (the `#882` `services` field),
//! each typed (`NinePExport`, `AtprotoPds`). The resolver takes an optional
//! service selector (default [`DEFAULT_AT9P_SERVICE`] = `#ns`, the 9P export a
//! client attaches to) and emits reach from **that selected `NinePExport`
//! service entry only**. A missing id, a wrong-type entry, or a non-iroh
//! transport **fails closed** — never an unrequested or wrong-shape endpoint.
//! This is the seam #906 (the DID-URL dereferencer, G1) pins against, and the
//! one #877's attach profile consumes.

use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};

use hyprstream_pds::at9p::{
    Capsule, ServiceType, Transport as At9pTransport, ED25519_PUBLIC_KEY_LEN,
};
use hyprstream_pds::at9p_gate::{verify_did_at9p, VerifiedCapsule};
use hyprstream_rpc::transport::TransportConfig;

/// Default at9p service selector — the `NinePExport` entry a client attaches to.
///
/// Matches the canonical 9P export name (`#ns`) used across the namespace
/// delivery layer (#877 attach `aname` / `exportRef`).
pub const DEFAULT_AT9P_SERVICE: &str = "#ns";

/// Map a **GATE-verified** `did:at9p` capsule to a dialable iroh
/// [`TransportConfig`], selecting the `NinePExport` service entry named `service`
/// (default [`DEFAULT_AT9P_SERVICE`]).
///
/// This is the pure, I/O-free core of D1 — a sibling to
/// `service_entry::decode_iroh`. It takes a [`VerifiedCapsule`] (so GATE has
/// already passed before this is reachable) and:
///
/// 1. selects the requested `NinePExport` service entry — fail closed if the id
///    is absent or the entry is the wrong type;
/// 2. requires that entry's transport to be iroh — fail closed otherwise;
/// 3. emits `TransportConfig::iroh(node_id = primary subject Ed25519,
///    relay_url = entry.relay)`.
///
/// The `node_id` is the **verified** primary subject key, not the endpoint's
/// self-declared `nodeId` string: on iroh the `EndpointId` *is* the Ed25519
/// pubkey, so the capsule's content-bound subject key is the authoritative dial
/// target. Direct addresses are not carried (iroh discovers direct paths; the
/// relay is a NAT-traversal hint), matching `decode_iroh`.
pub fn capsule_to_iroh_reach(
    verified: &VerifiedCapsule,
    service: Option<&str>,
) -> Result<TransportConfig> {
    let capsule = verified.capsule();
    let svc_id = service.unwrap_or(DEFAULT_AT9P_SERVICE);

    // Select the requested NinePExport service entry. Wrong type / missing id
    // ⇒ fail closed: we never emit reach from an unrequested or non-9P entry.
    let entry = capsule
        .body
        .services
        .iter()
        .find(|s| s.id == svc_id && s.service_type == ServiceType::NinePExport)
        .ok_or_else(|| {
            anyhow!("at9p capsule has no NinePExport service entry for selector {svc_id:?}")
        })?;

    // The selected entry must be iroh-reachable. A QUIC/MoQ/HTTPS entry under a
    // NinePExport is a wrong-shape capsule for this resolver — fail closed.
    ensure!(
        entry.endpoint.transport == At9pTransport::Iroh,
        "at9p service {svc_id:?} is not an iroh endpoint (got {:?})",
        entry.endpoint.transport,
    );

    // node_id = the GATE-verified primary subject Ed25519 key (the iroh NodeId).
    let node_id = primary_ed25519(capsule)?;
    let relay_url = entry.endpoint.relay.clone();

    Ok(TransportConfig::iroh(node_id, Vec::new(), relay_url))
}

/// Extract the primary subject Ed25519 key as a 32-byte iroh `EndpointId`.
///
/// The capsule schema guarantees `subject_keys` is non-empty and each keypair's
/// ed25519 half is exactly 32 bytes, so this only fails if a future schema
/// change breaks that invariant — kept fallible for defense in depth.
fn primary_ed25519(capsule: &Capsule) -> Result<[u8; ED25519_PUBLIC_KEY_LEN]> {
    let primary = capsule
        .body
        .subject_keys
        .first()
        .ok_or_else(|| anyhow!("verified capsule has no subject keys"))?;
    let node_id: [u8; ED25519_PUBLIC_KEY_LEN] = primary
        .ed25519_pub
        .as_slice()
        .try_into()
        .map_err(|_| anyhow!("at9p primary subject ed25519 key is not 32 bytes"))?;
    Ok(node_id)
}

/// Untrusted source of capsule bytes for a `did:at9p:<cid>` identifier.
///
/// The mainline locator (#890 / C2) is the production implementation; for D1
/// the trait abstracts *how* the bytes are fetched so the resolver only depends
/// on the GATE decision over them. The locator derives **zero authority**: the
/// [`verify_did_at9p`] gate is what makes the bytes trustworthy, not the source.
#[async_trait::async_trait]
pub trait CapsuleSource: Send + Sync {
    /// Fetch the raw (canonical DAG-CBOR) capsule bytes advertised for `did`.
    ///
    /// The bytes are attacker-controlled until GATE verifies them.
    async fn fetch_capsule(&self, did: &str) -> Result<Vec<u8>>;
}

/// `did:at9p` capsule resolver — turns a `did:at9p:<cid>` into a dialable
/// [`TransportConfig`] over a [`CapsuleSource`] + the GATE pipeline.
///
/// This is the injectable capsule source for the `Resolver::set_global` /
/// `NetworkDiscoveryResolver` seam (#873 network profile): a future network
/// resolver that recognizes `did:at9p:` peer addresses composes this to turn
/// them into reach. Everything here is additive — the ed25519 half already
/// drives the existing iroh dial path verbatim.
///
/// # Fail-closed posture
///
/// Reach is emitted **only** from a capsule that passes [`verify_did_at9p`]
/// (canon → hash → sig). Any GATE failure, a missing/wrong-type service entry,
/// or a non-iroh transport returns `Err` — never a partial or unverified reach.
pub struct At9pResolver {
    source: Arc<dyn CapsuleSource>,
}

impl At9pResolver {
    /// Build a resolver over an untrusted capsule source.
    pub fn new(source: Arc<dyn CapsuleSource>) -> Self {
        Self { source }
    }

    /// Resolve `did:at9p:<cid>` to a dialable [`TransportConfig`], selecting the
    /// `NinePExport` service entry named `service` (default `#ns`).
    ///
    /// Fetches the capsule bytes from the (untrusted) [`CapsuleSource`], runs the
    /// full GATE pipeline, and maps the verified capsule to iroh reach. Fails
    /// closed at the first gate that rejects the input.
    pub async fn resolve_did(&self, did: &str, service: Option<&str>) -> Result<TransportConfig> {
        let bytes = self.source.fetch_capsule(did).await?;
        // GATE — the only place authority is granted. Fail closed on any gate.
        let verified = verify_did_at9p(did, &bytes)?;
        capsule_to_iroh_reach(&verified, service)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::indexing_slicing)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes, MlDsaSigningKey};
    use hyprstream_pds::at9p::{
        CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, Transport,
    };
    use hyprstream_pds::at9p_gate::DID_AT9P_PREFIX;
    use hyprstream_pds::at9p_sign::sign_capsule;
    use hyprstream_rpc::transport::{BindMode, EndpointType};

    /// A test signer with matching hybrid subject keys.
    struct Signer {
        ed_sk: SigningKey,
        pq_sk: MlDsaSigningKey,
        keypair: HybridKeyPair,
    }

    fn signer(tag: u8) -> Signer {
        let mut seed = [0u8; 32];
        seed[0] = tag;
        seed[31] = tag.wrapping_add(7);
        let ed_sk = SigningKey::from_bytes(&seed);
        let (pq_sk, pq_vk) = ml_dsa_generate_keypair();
        let keypair = HybridKeyPair::new(
            ed_sk.verifying_key().to_bytes().to_vec(),
            ml_dsa_vk_bytes(&pq_vk),
        )
        .unwrap();
        Signer {
            ed_sk,
            pq_sk,
            keypair,
        }
    }

    /// A capsule body exposing three services: `#ns` (NinePExport/iroh — the
    /// dialable one), `#quic` (NinePExport but QUIC — wrong transport), and
    /// `#pds` (AtprotoPds/https — wrong type). Lets both selection gates be
    /// exercised.
    fn body_with_services(s: &Signer, tag: u8) -> CapsuleBody {
        let mut ns = ServiceEndpoint::new(Transport::Iroh, format!("iroh://node{tag}")).unwrap();
        ns.relay = Some(format!("https://relay{tag}.example"));
        let ns_entry = ServiceEntry::new("#ns", ServiceType::NinePExport, ns).unwrap();
        let quic = ServiceEndpoint::new(Transport::Quic, format!("quic://node{tag}")).unwrap();
        let quic_entry = ServiceEntry::new("#quic", ServiceType::NinePExport, quic).unwrap();
        let pds =
            ServiceEndpoint::new(Transport::Https, format!("https://pds{tag}.example")).unwrap();
        let pds_entry = ServiceEntry::new("#pds", ServiceType::AtprotoPds, pds).unwrap();
        CapsuleBody::new(
            vec![s.keypair.clone()],
            vec![ns_entry, quic_entry, pds_entry],
        )
        .unwrap()
    }

    /// A fully-signed capsule, its canonical bytes, its cid512, and its DID.
    fn signed(tag: u8) -> (Capsule, Vec<u8>, String, String) {
        let s = signer(tag);
        let body = body_with_services(&s, tag);
        let capsule = sign_capsule(body, &s.ed_sk, &s.pq_sk).unwrap();
        let bytes = capsule.to_dag_cbor().unwrap();
        let cid = capsule.cid512().unwrap();
        let did = format!("{DID_AT9P_PREFIX}{cid}");
        (capsule, bytes, cid, did)
    }

    #[test]
    fn valid_capsule_emits_iroh_reach_from_verified_subject_key() {
        let (capsule, bytes, _cid, did) = signed(1);

        // Reach comes from the GATE witness, not the raw capsule.
        let verified = verify_did_at9p(&did, &bytes).unwrap();
        let cfg = capsule_to_iroh_reach(&verified, None).unwrap();

        let expected_node = capsule.body.subject_keys[0].ed25519_pub.clone();
        match cfg.endpoint {
            EndpointType::Iroh {
                node_id,
                direct_addrs,
                relay_url,
            } => {
                assert_eq!(node_id.as_slice(), &expected_node);
                assert!(direct_addrs.is_empty());
                assert_eq!(relay_url.as_deref(), Some("https://relay1.example"));
            }
            // Bind mode is Connect — this is a client dial.
            other => panic!("expected Iroh endpoint, got {other:?}"),
        }
        assert_eq!(cfg.bind_mode, BindMode::Connect);
    }

    #[test]
    fn explicit_service_selector_selects_named_entry() {
        let (capsule, bytes, _cid, did) = signed(2);
        let verified = verify_did_at9p(&did, &bytes).unwrap();
        // Default `#ns` succeeds.
        assert!(capsule_to_iroh_reach(&verified, None).is_ok());
        let _ = capsule; // keep capsule alive for clarity
                         // `#quic` is a NinePExport but QUIC → wrong transport, fail closed.
        let err = capsule_to_iroh_reach(&verified, Some("#quic")).unwrap_err();
        assert!(
            err.to_string().contains("not an iroh endpoint"),
            "expected non-iroh rejection, got: {err}"
        );
    }

    #[test]
    fn missing_service_selector_fails_closed() {
        let (_capsule, bytes, _cid, did) = signed(3);
        let verified = verify_did_at9p(&did, &bytes).unwrap();
        // `#pds` is AtprotoPds → wrong type, treated as a missing NinePExport.
        let err = capsule_to_iroh_reach(&verified, Some("#pds")).unwrap_err();
        assert!(
            err.to_string().contains("no NinePExport service entry"),
            "expected wrong-type rejection, got: {err}"
        );
        // A totally absent selector id fails closed the same way.
        let err = capsule_to_iroh_reach(&verified, Some("#nope")).unwrap_err();
        assert!(
            err.to_string().contains("no NinePExport service entry"),
            "expected missing-service rejection, got: {err}"
        );
    }

    /// A capsule that fails GATE yields no reach: pass valid bytes but claim a
    /// different identity → hash-gate rejects before reach is ever computed.
    #[test]
    fn capsule_failing_gate_emits_no_reach() {
        let (_capsule, bytes, _cid_a, _did_a) = signed(10);
        let (_capsule_b, _bytes_b, _cid_b, did_b) = signed(11);
        // bytes are genuinely self-signed and canonical, but we claim the
        // identity of a *different* capsule → GATE fails closed.
        let gate = verify_did_at9p(&did_b, &bytes);
        assert!(gate.is_err(), "GATE must reject the mismatched identity");
        // No VerifiedCapsule exists, so capsule_to_iroh_reach is unreachable —
        // i.e. no reach can be emitted. (Mirrored at the resolver level below.)
    }

    /// `At9pResolver::resolve_did` threads GATE through: a valid capsule
    /// resolves; a GATE failure (wrong identity) emits no reach.
    struct FixedSource {
        bytes: Vec<u8>,
    }

    #[async_trait::async_trait]
    impl CapsuleSource for FixedSource {
        async fn fetch_capsule(&self, _did: &str) -> Result<Vec<u8>> {
            Ok(self.bytes.clone())
        }
    }

    #[tokio::test]
    async fn resolver_valid_capsule_resolves_to_iroh() {
        let (capsule, bytes, _cid, did) = signed(20);
        let resolver = At9pResolver::new(Arc::new(FixedSource {
            bytes: bytes.clone(),
        }));
        let cfg = resolver.resolve_did(&did, None).await.unwrap();
        let expected_node = capsule.body.subject_keys[0].ed25519_pub.clone();
        match cfg.endpoint {
            EndpointType::Iroh {
                node_id, relay_url, ..
            } => {
                assert_eq!(node_id.as_slice(), &expected_node);
                assert_eq!(relay_url.as_deref(), Some("https://relay20.example"));
            }
            other => panic!("expected Iroh endpoint, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn resolver_capsule_failing_gate_emits_no_reach() {
        // Serve capsule A's bytes under B's identity — GATE must fail closed.
        let (_a, bytes_a, _cid_a, _did_a) = signed(30);
        let (_b, _bytes_b, _cid_b, did_b) = signed(31);
        let resolver = At9pResolver::new(Arc::new(FixedSource { bytes: bytes_a }));
        let err = resolver.resolve_did(&did_b, None).await.unwrap_err();
        assert!(
            err.to_string().contains("hash-gate"),
            "expected a GATE rejection surfaced, got: {err}"
        );
    }
}
