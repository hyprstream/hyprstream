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
//! # Why a genesis capsule cannot mint a dial target
//!
//! GATE makes genesis fields content-bound claims; it does not prove current
//! state, liveness, or possession. The schema carries a relay claim but no
//! independent carrier `EndpointId`, so this resolver fails closed rather than
//! deriving reach from the subject Ed25519 key (#1031).
//!
//! # Service selection (#893 update; design #905 §4, #879 §5.1a)
//!
//! A capsule carries a **map** of named services (the `#882` `services` field),
//! each typed (`NinePExport`, `AtprotoPds`). The resolver takes an optional
//! service selector (default [`DEFAULT_AT9P_SERVICE`] = `#ns`, the 9P export a
//! client attaches to) and validates **that selected `NinePExport` service
//! entry only**. A missing id, a wrong-type entry, a non-iroh transport, or the
//! absence of an independent carrier EndpointId **fails closed** — never an
//! unrequested, wrong-shape, or identity-derived endpoint.
//! This is the seam #906 (the DID-URL dereferencer, G1) pins against, and the
//! one #877's attach profile consumes.

use std::sync::Arc;

use anyhow::{anyhow, ensure, Result};

use hyprstream_pds::at9p::{ServiceType, Transport as At9pTransport};
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
/// 3. rejects the capsule because the current schema has no independent
///    carrier EndpointId. The signed relay remains a reach claim, but cannot
///    turn the genesis subject identity key into a live dial target.
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

    let relay = entry.endpoint.relay.as_deref().unwrap_or("<none>");
    Err(anyhow!(
        "at9p service {svc_id:?} carries relay claim {relay:?} but no independent iroh EndpointId; refusing to derive transport reach from the genesis identity key (#1031)"
    ))
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
/// resolver that recognizes `did:at9p:` peer addresses may compose this after
/// the capsule schema carries independently authenticated live reach.
///
/// # Fail-closed posture
///
/// The capsule must pass [`verify_did_at9p`] (canon → hash → sig), but that
/// proves only the genesis claim. Until an independent EndpointId is available,
/// even a valid capsule returns `Err`; other GATE and service-selection failures
/// also return `Err`, never partial or identity-derived reach.
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
    /// full GATE pipeline, and validates the selected reach claim. It fails
    /// closed because the current capsule has no independent live EndpointId.
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
        Capsule, CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, Transport,
    };
    use hyprstream_pds::at9p_gate::DID_AT9P_PREFIX;
    use hyprstream_pds::at9p_sign::sign_capsule;

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
    fn signed_capsule_relay_claim_does_not_mint_live_reach() {
        let (_capsule, bytes, _cid, did) = signed(1);

        // Reach comes from the GATE witness, not the raw capsule.
        let verified = verify_did_at9p(&did, &bytes).unwrap();
        let err = capsule_to_iroh_reach(&verified, None).unwrap_err();
        assert!(err.to_string().contains("no independent iroh EndpointId"));
    }

    #[test]
    fn explicit_service_selector_selects_named_entry() {
        let (capsule, bytes, _cid, did) = signed(2);
        let verified = verify_did_at9p(&did, &bytes).unwrap();
        // Default `#ns` is signed, but cannot mint live reach.
        assert!(capsule_to_iroh_reach(&verified, None).is_err());
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
    async fn resolver_valid_capsule_without_transport_id_fails_closed() {
        let (_capsule, bytes, _cid, did) = signed(20);
        let resolver = At9pResolver::new(Arc::new(FixedSource {
            bytes: bytes.clone(),
        }));
        let err = resolver.resolve_did(&did, None).await.unwrap_err();
        assert!(err.to_string().contains("no independent iroh EndpointId"));
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
