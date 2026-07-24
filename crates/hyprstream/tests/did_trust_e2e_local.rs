//! #1137 end-to-end, same-node arm: the DID-anchored bootstrap with a lazy
//! local discovery client (the metal stack's `service start --ipc` shape —
//! containers sharing the IPC fabric, Discovery hosted in-process).
//!
//! Separate test FILE from `did_trust_e2e.rs` because
//! `bootstrap_deployment_process` seals a one-shot process-global authority;
//! each arm gets its own process.

#![allow(clippy::expect_used, clippy::unwrap_used)]

#[path = "fixtures/did_trust.rs"]
mod did_trust;
use did_trust::*;

use ed25519_dalek::SigningKey;
use hyprstream_discovery::{DeploymentTrustSource, DidAnchors};
use hyprstream_rpc::did_key::ed25519_to_did_key;
use hyprstream_rpc::service_entry::encode_quic;
use hyprstream_rpc::transport::QuicServerAuth;
use serde_json::json;
use tempfile::TempDir;

struct LocalFixture {
    _dir: TempDir,
    well_known: std::path::PathBuf,
    did_web: String,
    capsule: CapsuleMaterial,
    ca: SigningKey,
    registry: SigningKey,
    tls: TlsMaterial,
}

impl LocalFixture {
    fn anchors(&self) -> DidAnchors {
        DidAnchors {
            cluster_at9p_did: self.capsule.at9p_did.clone(),
            cluster_did_web: self.did_web.clone(),
            extra_root_cert_pem: None,
        }
        .with_root_cert_pem(self.tls.root_pem.clone())
    }

    fn did_document(&self) -> serde_json::Value {
        let ca_vk = self.ca.verifying_key();
        let multikey = ed25519_to_did_key(ca_vk.as_bytes())
            .strip_prefix("did:key:")
            .unwrap()
            .to_owned();
        // The local arm never dials the advertised transport; a pinned
        // loopback entry stands in for reach.
        let auth = QuicServerAuth::pinned(vec![[7u8; 32]]).unwrap();
        json!({
            "id": self.did_web,
            "alsoKnownAs": [self.capsule.at9p_did],
            "verificationMethod": [{
                "id": format!("{}#deployment-ca", self.did_web),
                "type": "Multikey",
                "controller": self.did_web,
                "publicKeyMultibase": multikey,
            }],
            "service": [{
                "id": format!("{}#quic", self.did_web),
                "type": "QuicTransport",
                "serviceEndpoint": encode_quic(
                    "https://127.0.0.1:4433",
                    &auth,
                    &["hyprstream-rpc/1"],
                ),
            }],
        })
    }
}

/// The same-node boot: full resolve → verify → credential validation against
/// the real serving side, then a LAZY local discovery client and resolver
/// install — no network liveness required (the metal stack shape, where
/// Discovery starts later in the same process).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn did_anchored_bootstrap_boots_same_node() {
    ensure_process_globals();

    // Bind before publishing the DID so the first bootstrap fetch cannot race
    // the spawned HTTPS task for an otherwise-free port.
    let https_listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let https_port = https_listener.local_addr().unwrap().port();
    let did_web = format!("did:web:localhost%3A{https_port}");
    let capsule = make_capsule(&did_web, 0x61);

    let dir = TempDir::new().unwrap();
    let well_known = dir.path().join("well-known");
    std::fs::create_dir_all(well_known.join("at9p")).unwrap();
    std::fs::create_dir_all(well_known.join("deployment")).unwrap();

    let fixture = LocalFixture {
        well_known,
        did_web,
        capsule,
        ca: SigningKey::from_bytes(&[0x62; 32]),
        registry: SigningKey::from_bytes(&[0x63; 32]),
        tls: make_tls(),
        _dir: dir,
    };

    // Lay out the deployment material.
    let cid = fixture.capsule.at9p_did.strip_prefix("did:at9p:").unwrap();
    std::fs::write(
        fixture.well_known.join("did.json"),
        fixture.did_document().to_string(),
    )
    .unwrap();
    std::fs::write(
        fixture.well_known.join("at9p").join(format!("{cid}.cbor")),
        &fixture.capsule.bytes,
    )
    .unwrap();
    std::fs::write(
        fixture
            .well_known
            .join("deployment")
            .join("registry-service.jwt"),
        mint_credential(&fixture.ca, &fixture.registry),
    )
    .unwrap();

    // Real HTTPS serving side (the REAL deployment well-known router).
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

    // The discovery key the local trust store pins (the local client is lazy;
    // no live discovery needed for this arm).
    let discovery_sk = SigningKey::from_bytes(&[0x64; 32]);
    register_discovery_key(discovery_sk.verifying_key());

    let node_key = SigningKey::from_bytes(&[0x65; 32]);
    hyprstream_discovery::bootstrap_deployment_process(
        node_key,
        DeploymentTrustSource::DidAnchored(fixture.anchors()),
        false,
    )
    .await
    .expect("same-node DID-anchored bootstrap must boot without network liveness");
}
