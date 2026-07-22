//! #1137 end-to-end: the DID-anchored trust bootstrap against the REAL
//! serving side.
//!
//! This boots the production `bootstrap_deployment_process` against:
//! - a REAL HTTPS server (rustls, test-CA-signed `localhost` certificate)
//!   mounting the REAL `deployment_well_known` router from a real
//!   operator-laid-out directory (`did.json`, `at9p/<cid>.cbor`,
//!   `deployment/registry-service.jwt`), and
//! - a REAL DiscoveryService served over REAL QUIC (self-signed cert, pinned
//!   by SHA-256 in the DID document's `QuicTransport` entry).
//!
//! The positive test exercises resolve → mutual-alias verify → GATE →
//! credential fetch + validation → dial → signed liveness ping → resolver
//! install. Every negative test removes or corrupts one leg and asserts the
//! bootstrap REFUSES — a reachable-but-wrong endpoint must never downgrade
//! trust.
//!
//! Process-global note: `bootstrap_deployment_process` seals a one-shot
//! authority per process, so exactly one test drives the full boot; the
//! negatives drive `resolve_and_authenticate_did_anchors` (the identical
//! trust decision minus the seal/install). The same-node (local) arm has its
//! own test file (`did_trust_e2e_local.rs`) for the same reason.

#![allow(clippy::expect_used, clippy::unwrap_used)]

#[path = "fixtures/did_trust.rs"]
mod did_trust;
use did_trust::*;

use ed25519_dalek::SigningKey;
use hyprstream_discovery::{DeploymentTrustSource, DidAnchors};
use hyprstream_rpc::did_key::ed25519_to_did_key;
use hyprstream_rpc::service_entry::encode_quic;
use hyprstream_rpc::transport::{QuicServerAuth, TransportConfig};
use hyprstream_service::{InprocManager, ServiceManager as _};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;
use tempfile::TempDir;

struct Fixture {
    _dir: TempDir,
    _discovery: hyprstream_service::SpawnedService,
    well_known: std::path::PathBuf,
    did_web: String,
    capsule: CapsuleMaterial,
    ca: SigningKey,
    registry: SigningKey,
    tls: TlsMaterial,
    discovery_sk: SigningKey,
    quic_addr: SocketAddr,
    quic_cert_der: Vec<u8>,
}

impl Fixture {
    fn anchors(&self) -> DidAnchors {
        DidAnchors {
            cluster_at9p_did: self.capsule.at9p_did.clone(),
            cluster_did_web: self.did_web.clone(),
            extra_root_cert_pem: None,
        }
        .with_root_cert_pem(self.tls.root_pem.clone())
    }

    fn trust_source(&self) -> DeploymentTrustSource {
        DeploymentTrustSource::DidAnchored(self.anchors())
    }

    fn did_document(&self) -> serde_json::Value {
        let ca_vk = self.ca.verifying_key();
        let multikey = ed25519_to_did_key(ca_vk.as_bytes())
            .strip_prefix("did:key:")
            .unwrap()
            .to_owned();
        let pin = hyprstream_rpc::transport::quinn_transport::cert_sha256(&self.quic_cert_der);
        let auth = QuicServerAuth::pinned(vec![pin]).unwrap();
        // The discovery service's #mesh-kem hybrid-KEM recipient, derived
        // from its signing key exactly as the QUIC loop derives it — the
        // remote bootstrap arm binds the pinned discovery key to this.
        let kem = hyprstream_rpc::node_identity::derive_mesh_kem_recipient(&self.discovery_sk)
            .unwrap()
            .public();
        let multibase = |codec: [u8; 2], key: &[u8]| {
            let mut payload = Vec::with_capacity(2 + key.len());
            payload.extend_from_slice(&codec);
            payload.extend_from_slice(key);
            format!("z{}", bs58::encode(payload).into_string())
        };
        // The discovery service's mesh ML-DSA-65 verifying key — the remote
        // bootstrap arm anchors response authentication to it.
        let pq_sk = hyprstream_rpc::node_identity::derive_mesh_mldsa_key(&self.discovery_sk);
        let pq_vk_bytes = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq_sk);
        json!({
            "id": self.did_web,
            "alsoKnownAs": [self.capsule.at9p_did],
            "verificationMethod": [
                {
                    "id": format!("{}#deployment-ca", self.did_web),
                    "type": "Multikey",
                    "controller": self.did_web,
                    "publicKeyMultibase": multikey,
                },
                {
                    "id": format!("{}#mesh-pq", self.did_web),
                    "type": "Multikey",
                    "controller": self.did_web,
                    "publicKeyMultibase": multibase([0x91, 0x24], &pq_vk_bytes),
                },
            ],
            "keyAgreement": [
                {
                    "id": format!("{}#mesh-kem-x25519", self.did_web),
                    "type": "Multikey",
                    "controller": self.did_web,
                    "publicKeyMultibase": multibase([0xec, 0x01], &kem.eks[0]),
                },
                {
                    "id": format!("{}#mesh-kem-mlkem768", self.did_web),
                    "type": "Multikey",
                    "controller": self.did_web,
                    "publicKeyMultibase": multibase([0x8c, 0x24], &kem.eks[1]),
                },
            ],
            "service": [{
                "id": format!("{}#quic", self.did_web),
                "type": "QuicTransport",
                "serviceEndpoint": encode_quic(
                    &format!("https://{}", self.quic_addr),
                    &auth,
                    &["hyprstream-rpc/1"],
                ),
            }],
        })
    }

    /// Lay out (or refresh) the deployment well-known directory contents.
    fn write_material(&self, capsule_bytes: &[u8], credential: &str, document: &serde_json::Value) {
        let at9p_dir = self.well_known.join("at9p");
        let deployment_dir = self.well_known.join("deployment");
        std::fs::create_dir_all(&at9p_dir).unwrap();
        std::fs::create_dir_all(&deployment_dir).unwrap();
        std::fs::write(self.well_known.join("did.json"), document.to_string()).unwrap();
        let cid = self
            .capsule
            .at9p_did
            .strip_prefix("did:at9p:")
            .unwrap()
            .to_owned();
        std::fs::write(at9p_dir.join(format!("{cid}.cbor")), capsule_bytes).unwrap();
        std::fs::write(deployment_dir.join("registry-service.jwt"), credential).unwrap();
    }

    fn write_valid(&self) {
        let credential = mint_credential(&self.ca, &self.registry);
        self.write_material(&self.capsule.bytes, &credential, &self.did_document());
    }
}

async fn build_fixture() -> Fixture {
    ensure_process_globals();

    // Bind before publishing the DID so the first bootstrap fetch cannot race
    // the spawned HTTPS task for an otherwise-free port.
    let https_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let https_addr = https_listener.local_addr().unwrap();
    let https_port = https_addr.port();
    let did_web = format!("did:web:localhost%3A{https_port}");
    let capsule = make_capsule(&did_web, 0x51);

    // Real DiscoveryService over real QUIC (WebTransport, h3 — the same
    // service loop production uses; self-signed cert, SHA-256 pinned).
    let discovery_sk = SigningKey::from_bytes(&[0x52; 32]);
    register_discovery_key(discovery_sk.verifying_key());
    let quic_cert =
        rcgen::generate_simple_self_signed(vec!["hyprstream.local".to_owned()]).unwrap();
    let quic_cert_der = quic_cert.cert.der().to_vec();
    let quic_key_der = quic_cert.key_pair.serialize_der();
    let jwt_vk = SigningKey::from_bytes(&[0x53; 32]).verifying_key();
    let discovery_service = hyprstream_discovery::DiscoveryService::new(
        Arc::new(discovery_sk.clone()),
        jwt_vk,
        TransportConfig::inproc("did-trust-e2e-discovery"),
    );
    let (addr_tx, addr_rx) = tokio::sync::oneshot::channel::<SocketAddr>();
    let quic_config = hyprstream_rpc::service::QuicLoopConfig {
        cert_chain: vec![quic_cert_der.clone()],
        key_der: zeroize::Zeroizing::new(quic_key_der),
        bind_addr: "127.0.0.1:0".parse().unwrap(),
        server_name: "hyprstream.local".to_owned(),
        protected_resource_json: None,
        on_quic_bound: Some(Box::new(move |_name, addr, _server| {
            let _ = addr_tx.send(addr);
        })),
        iroh_enabled: false,
        on_iroh_bound: None,
        moq_relay: None,
    };
    let service =
        hyprstream_service::UnifiedServiceConfig::new(discovery_service, Some(quic_config));
    let manager = InprocManager::new();
    let spawned = manager
        .spawn(Box::new(service))
        .await
        .expect("discovery QUIC loop must bind");
    let quic_addr = addr_rx.await.expect("QUIC loop must report its bound addr");

    let dir = TempDir::new().unwrap();
    let well_known = dir.path().join("well-known");
    std::fs::create_dir_all(&well_known).unwrap();

    let fixture = Fixture {
        well_known,
        _discovery: spawned,
        did_web,
        capsule,
        ca: SigningKey::from_bytes(&[0x54; 32]),
        registry: SigningKey::from_bytes(&[0x55; 32]),
        tls: make_tls(),
        discovery_sk,
        quic_addr,
        quic_cert_der,
        _dir: dir,
    };
    fixture.write_valid();

    // Real HTTPS serving side: the REAL deployment well-known router.
    let rustls = axum_server::tls_rustls::RustlsConfig::from_der(
        fixture.tls.chain_der.clone(),
        fixture.tls.key_der.clone(),
    )
    .await
    .expect("test RustlsConfig must build");
    let app = hyprstream_core::services::oauth::deployment_well_known::router::<()>(Some(
        fixture.well_known.clone(),
    ));
    tokio::spawn(async move {
        axum_server::from_tcp_rustls(https_listener, rustls)
            .serve(app.into_make_service())
            .await
            .expect("deployment well-known HTTPS server failed");
    });

    fixture
}

/// THE end-to-end boot: real serving side, real GATE, real credential, real
/// QUIC discovery, real liveness ping, real resolver install.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn did_anchored_bootstrap_boots_end_to_end() {
    let fixture = build_fixture().await;
    let node_key = SigningKey::from_bytes(&[0x56; 32]);
    hyprstream_discovery::bootstrap_deployment_process(node_key, fixture.trust_source(), true)
        .await
        .expect("DID-anchored bootstrap must boot against the real serving side");
}

/// A reachable serving side that does NOT serve the capsule must refuse.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_capsule_refuses() {
    let fixture = build_fixture().await;
    let cid = fixture.capsule.at9p_did.strip_prefix("did:at9p:").unwrap();
    std::fs::remove_file(fixture.well_known.join("at9p").join(format!("{cid}.cbor"))).unwrap();
    let error = hyprstream_discovery::resolve_and_authenticate_did_anchors(&fixture.anchors())
        .await
        .err()
        .expect("bootstrap must refuse when the capsule endpoint 404s");
    assert!(format!("{error:#}").contains("capsule"), "{error:#}");
}

/// A DIFFERENT (validly-signed, wrong-hash) capsule must refuse at the
/// hash-gate — a reachable-but-hostile capsule endpoint cannot downgrade.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreign_capsule_refuses_at_hash_gate() {
    let fixture = build_fixture().await;
    let foreign = make_capsule(&fixture.did_web, 0x77);
    let credential = mint_credential(&fixture.ca, &fixture.registry);
    fixture.write_material(&foreign.bytes, &credential, &fixture.did_document());
    let error = hyprstream_discovery::resolve_and_authenticate_did_anchors(&fixture.anchors())
        .await
        .err()
        .expect("a capsule that does not hash to the configured DID must refuse");
    assert!(format!("{error:#}").contains("hash-gate"), "{error:#}");
}

/// A capsule that does not reciprocate the did:web alias must refuse.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_sided_alias_refuses() {
    let fixture = build_fixture().await;
    // Doc-side one-sided claim: the capsule reciprocates the did:web, but the
    // served document does not name the configured at9p DID. Mutual
    // attestation is bidirectional at this layer too — refuse.
    let mut document = fixture.did_document();
    document.as_object_mut().unwrap().remove("alsoKnownAs");
    let credential = mint_credential(&fixture.ca, &fixture.registry);
    fixture.write_material(&fixture.capsule.bytes, &credential, &document);
    let error = hyprstream_discovery::resolve_and_authenticate_did_anchors(&fixture.anchors())
        .await
        .err()
        .expect("a document that does not name the at9p DID must refuse");
    assert!(format!("{error:#}").contains("alsoKnownAs"), "{error:#}");
}

/// No registry credential at the endpoint → refuse (never degrade to a
/// weaker or stale credential source).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_credential_refuses() {
    let fixture = build_fixture().await;
    std::fs::remove_file(
        fixture
            .well_known
            .join("deployment")
            .join("registry-service.jwt"),
    )
    .unwrap();
    let error = hyprstream_discovery::resolve_and_authenticate_did_anchors(&fixture.anchors())
        .await
        .err()
        .expect("bootstrap must refuse when the credential endpoint 404s");
    assert!(format!("{error:#}").contains("credential"), "{error:#}");
}

/// A credential minted under a DIFFERENT CA must refuse — the DID-derived CA
/// is the only authority, exactly the attack my #1143 review repro exercised.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreign_ca_credential_refuses() {
    let fixture = build_fixture().await;
    let attacker_ca = SigningKey::from_bytes(&[0xAA; 32]);
    let credential = mint_credential(&attacker_ca, &fixture.registry);
    fixture.write_material(&fixture.capsule.bytes, &credential, &fixture.did_document());
    let error = hyprstream_discovery::resolve_and_authenticate_did_anchors(&fixture.anchors())
        .await
        .err()
        .expect("a credential under a foreign CA must refuse");
    assert!(format!("{error:#}").contains("credential"), "{error:#}");
}

/// The liveness leg: a DID-advertised transport whose endpoint does not hold
/// the pinned discovery key (or is dead) must fail the signed ping.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dead_or_wrong_key_discovery_fails_liveness() {
    let fixture = build_fixture().await;

    // Dead endpoint — nothing listening.
    let dead: SocketAddr = "127.0.0.1:9".parse().unwrap();
    let pin = hyprstream_rpc::transport::quinn_transport::cert_sha256(&fixture.quic_cert_der);
    let dead_transport = TransportConfig::quic_pinned(dead, "hyprstream.local", pin);
    let signer = hyprstream_rpc::signer::LocalSigner::new(SigningKey::from_bytes(&[0x57; 32]));
    let discovery_vk = fixture.discovery_sk.verifying_key();
    let rpc =
        hyprstream_rpc::dial::dial(&dead_transport, signer, Some(discovery_vk), None).unwrap();
    let client = hyprstream_discovery::DiscoveryClient::new(rpc);
    assert!(
        client.ping().await.is_err(),
        "ping against a dead endpoint must fail"
    );

    // Wrong key at a LIVE endpoint: dial the real discovery but pin a key it
    // does not hold — response verification must fail.
    let wrong_vk = SigningKey::from_bytes(&[0x58; 32]).verifying_key();
    let live_transport = TransportConfig::quic_pinned(fixture.quic_addr, "hyprstream.local", pin);
    let signer = hyprstream_rpc::signer::LocalSigner::new(SigningKey::from_bytes(&[0x59; 32]));
    let rpc = hyprstream_rpc::dial::dial(&live_transport, signer, Some(wrong_vk), None).unwrap();
    let client = hyprstream_discovery::DiscoveryClient::new(rpc);
    assert!(
        client.ping().await.is_err(),
        "ping pinned to a key the endpoint does not hold must fail"
    );
}
