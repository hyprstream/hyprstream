//! Shared fixture helpers for the DID-anchored trust e2e tests (#1137).
//! Included via `#[path]` from the per-process test files.

#![allow(clippy::expect_used, clippy::unwrap_used, dead_code, unused_imports)]

use std::net::SocketAddr;
use std::sync::Arc;

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use ed25519_dalek::{Signer as _, SigningKey};
use hyprstream_discovery::{DeploymentTrustSource, DidAnchors};
use hyprstream_pds::at9p::{
    CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
};
use hyprstream_pds::at9p_sign::sign_capsule;
use hyprstream_rpc::did_key::ed25519_to_did_key;
use hyprstream_rpc::service_entry::encode_quic;
use hyprstream_rpc::transport::{QuicServerAuth, TransportConfig};
use hyprstream_service::{InprocManager, ServiceManager as _};
use serde_json::json;
use tempfile::TempDir;

pub const AUD: &str = "urn:hyprstream:service:registry";
pub const PROFILE: &str = "hyprstream.registry-deployment.v1";

/// Reserve a loopback port (bind + release). Tiny race, acceptable in tests.
pub fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

pub fn ensure_process_globals() {
    use hyprstream_rpc::crypto::CryptoPolicy;
    use hyprstream_rpc::registry::{init as init_registry, EndpointMode};
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    // Isolate the deployment data dir (the bootstrap opens the checkpointed
    // PDS store under it) from any real node state.
    std::env::set_var(
        "HYPRSTREAM_INSTANCE",
        format!("did-trust-e2e-{}", std::process::id()),
    );
    // Idempotent: registry init and verify-config installs are first-write-wins.
    init_registry(EndpointMode::Ipc, Some(std::env::temp_dir()));
    // rustls needs an explicit process default when both aws-lc-rs and ring
    // features are linked; use the repo's declared interop provider.
    hyprstream_rpc::transport::pq_provider::install_pq_crypto_provider()
        .expect("process crypto provider must install");
    let _ = hyprstream_rpc::envelope::install_verify_config(
        hyprstream_rpc::envelope::EnvelopeVerifyConfig {
            policy: CryptoPolicy::Classical,
            pq_store: None,
        },
    );
    let _ = hyprstream_rpc::envelope::install_response_verify_config(
        hyprstream_rpc::envelope::ResponseVerifyConfig {
            policy: CryptoPolicy::Classical,
            pq_store: None,
        },
    );
}

pub fn register_discovery_key(vk: ed25519_dalek::VerifyingKey) {
    hyprstream_service::global_trust_store().insert(
        vk,
        hyprstream_service::Attestation {
            scopes: std::iter::once("discovery".to_owned()).collect(),
            subject: None,
            jwt: None,
            expires_at: chrono::Utc::now().timestamp() + 86_400,
            attested_by: None,
        },
    );
}

pub fn deployment_domain(ca_vk: &ed25519_dalek::VerifyingKey) -> String {
    hyprstream_rpc::auth::jwk_thumbprint(&hyprstream_rpc::auth::JwkThumbprintInput::Ed25519 {
        x: ca_vk.as_bytes(),
    })
}

/// Mint a registry deployment credential with the exact production profile.
pub fn mint_credential(ca: &SigningKey, registry: &SigningKey) -> String {
    let domain = deployment_domain(&ca.verifying_key());
    let now = chrono::Utc::now().timestamp();
    let protected = json!({"alg": "EdDSA", "typ": "wit+jwt", "kid": domain});
    let claims = json!({
        "iss": format!("urn:hyprstream:deployment:{domain}"),
        "sub": "service:registry",
        "aud": AUD,
        "exp": now + 3600,
        "nbf": now,
        "iat": now,
        "deployment_domain": domain,
        "profile": PROFILE,
        "cnf": {"jwk": {
            "kty": "OKP",
            "crv": "Ed25519",
            "x": URL_SAFE_NO_PAD.encode(registry.verifying_key().as_bytes()),
        }},
    });
    let protected = URL_SAFE_NO_PAD.encode(protected.to_string().as_bytes());
    let claims = URL_SAFE_NO_PAD.encode(claims.to_string().as_bytes());
    let input = format!("{protected}.{claims}");
    let sig = ca.sign(input.as_bytes());
    format!("{input}.{}", URL_SAFE_NO_PAD.encode(sig.to_bytes()))
}

pub struct CapsuleMaterial {
    pub bytes: Vec<u8>,
    pub at9p_did: String,
}

pub fn make_capsule(did_web: &str, tag: u8) -> CapsuleMaterial {
    let ed = SigningKey::from_bytes(&[tag; 32]);
    let (pq, pq_vk) = hyprstream_crypto::pq::ml_dsa_generate_keypair();
    let pair = HybridKeyPair::new(
        ed.verifying_key().to_bytes().to_vec(),
        hyprstream_crypto::pq::ml_dsa_vk_bytes(&pq_vk),
    )
    .unwrap();
    let endpoint = ServiceEndpoint::new(Transport::Iroh, format!("iroh://node{tag}")).unwrap();
    let service = ServiceEntry::new("#ns", ServiceType::NinePExport, endpoint).unwrap();
    let mut body = CapsuleBody::new(vec![pair], vec![service]).unwrap();
    body.also_known_as = Some(vec![did_web.to_owned()]);
    let capsule = sign_capsule(body, &ed, &pq).unwrap();
    let bytes = capsule.to_dag_cbor().unwrap();
    let did = format!("did:at9p:{}", capsule.cid512().unwrap());
    CapsuleMaterial {
        bytes,
        at9p_did: did,
    }
}

pub struct TlsMaterial {
    /// PEM of the test root CA — the client's extra trust anchor.
    pub root_pem: Vec<u8>,
    /// DER chain (leaf, CA) + DER key for the HTTPS server.
    pub chain_der: Vec<Vec<u8>>,
    pub key_der: Vec<u8>,
}

pub fn make_tls() -> TlsMaterial {
    let ca_key = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
    let mut ca_params = rcgen::CertificateParams::default();
    ca_params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Unconstrained);
    // Distinct issuer/subject DNs are load-bearing: with empty DNs OpenSSL
    // (reqwest's native-tls here) classifies the leaf as self-signed
    // (issuer == subject) and chain building to the extra root fails.
    ca_params
        .distinguished_name
        .push(rcgen::DnType::CommonName, "hyprstream-test-root");
    let ca = ca_params.self_signed(&ca_key).unwrap();

    let leaf_key = rcgen::KeyPair::generate_for(&rcgen::PKCS_ECDSA_P256_SHA256).unwrap();
    let mut leaf_params = rcgen::CertificateParams::new(vec!["localhost".to_owned()]).unwrap();
    leaf_params
        .distinguished_name
        .push(rcgen::DnType::CommonName, "localhost");
    let leaf = leaf_params.signed_by(&leaf_key, &ca, &ca_key).unwrap();

    TlsMaterial {
        root_pem: ca.pem().into_bytes(),
        chain_der: vec![leaf.der().to_vec(), ca.der().to_vec()],
        key_der: leaf_key.serialize_der(),
    }
}
