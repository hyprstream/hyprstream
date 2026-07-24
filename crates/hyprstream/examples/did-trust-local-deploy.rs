//! #1137 local boot-repro fixture: stand up a REAL local deployment — the
//! same material the e2e tests use — and serve it over REAL HTTPS, then
//! print the exact environment a same-node `service start … --ipc`
//! invocation needs to DID-anchor against it.
//!
//! Usage:
//!   cargo run -p hyprstream --example did-trust-local-deploy -- <workdir>
//!
//! Then, in another shell:
//!   export HYPRSTREAM__CLUSTER_AT9P_DID=...   (printed)
//!   export HYPRSTREAM__CLUSTER_DID_WEB=...    (printed)
//!   export HYPRSTREAM__CLUSTER_ANCHOR_ROOT_CERT=...  (printed)
//!   export HYPRSTREAM_INSTANCE=...            (printed)
//!   hyprstream service start registry --foreground --ipc

#![allow(clippy::expect_used, clippy::print_stdout, clippy::unwrap_used)]

#[path = "../tests/fixtures/did_trust.rs"]
mod did_trust;
use did_trust::*;

use ed25519_dalek::SigningKey;
use hyprstream_rpc::did_key::ed25519_to_did_key;
use hyprstream_rpc::service_entry::encode_quic;
use hyprstream_rpc::transport::QuicServerAuth;
use serde_json::json;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> anyhow::Result<()> {
    // rustls needs an explicit process default (both aws-lc-rs and ring are
    // linked in this workspace); use the repo's declared interop provider.
    hyprstream_rpc::transport::pq_provider::install_pq_crypto_provider()
        .map_err(|e| anyhow::anyhow!("crypto provider: {e}"))?;
    let workdir = std::env::args()
        .nth(1)
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp/did-trust-local-deploy"));
    std::fs::create_dir_all(&workdir)?;

    // Reserve the listener before emitting the DID so callers can connect as
    // soon as this process reports the deployment ready.
    let https_listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let https_port = https_listener.local_addr()?.port();
    let did_web = format!("did:web:localhost%3A{https_port}");
    let capsule = make_capsule(&did_web, 0x71);
    let ca = SigningKey::from_bytes(&[0x72; 32]);
    let registry = SigningKey::from_bytes(&[0x73; 32]);
    let tls = make_tls();

    // Deployment well-known directory layout.
    let well_known = workdir.join("well-known");
    std::fs::create_dir_all(well_known.join("at9p"))?;
    std::fs::create_dir_all(well_known.join("deployment"))?;

    let ca_vk = ca.verifying_key();
    let multikey = ed25519_to_did_key(ca_vk.as_bytes())
        .strip_prefix("did:key:")
        .unwrap()
        .to_owned();
    let auth = QuicServerAuth::pinned(vec![[9u8; 32]]).unwrap();
    let document = json!({
        "id": did_web,
        "alsoKnownAs": [capsule.at9p_did],
        "verificationMethod": [{
            "id": format!("{did_web}#deployment-ca"),
            "type": "Multikey",
            "controller": did_web,
            "publicKeyMultibase": multikey,
        }],
        "service": [{
            "id": format!("{did_web}#quic"),
            "type": "QuicTransport",
            "serviceEndpoint": encode_quic("https://127.0.0.1:4433", &auth, &["hyprstream-rpc/1"]),
        }],
    });
    std::fs::write(well_known.join("did.json"), document.to_string())?;
    let cid = capsule.at9p_did.strip_prefix("did:at9p:").unwrap();
    std::fs::write(
        well_known.join("at9p").join(format!("{cid}.cbor")),
        &capsule.bytes,
    )?;
    std::fs::write(
        well_known.join("deployment").join("registry-service.jwt"),
        mint_credential(&ca, &registry),
    )?;
    let root_cert_path = workdir.join("test-root.pem");
    std::fs::write(&root_cert_path, &tls.root_pem)?;

    // Real HTTPS serving side (the REAL deployment well-known router).
    let rustls =
        axum_server::tls_rustls::RustlsConfig::from_der(tls.chain_der, tls.key_der).await?;
    let app = hyprstream_core::services::oauth::deployment_well_known::router::<()>(Some(
        well_known.clone(),
    ));
    tokio::spawn(async move {
        axum_server::from_tcp_rustls(https_listener, rustls)
            .serve(app.into_make_service())
            .await
            .expect("deployment well-known HTTPS server failed");
    });

    let instance = format!("did-trust-boot-{}", std::process::id());
    println!("deployment ready at {}", workdir.display());
    println!("export HYPRSTREAM__CLUSTER_AT9P_DID={}", capsule.at9p_did);
    println!("export HYPRSTREAM__CLUSTER_DID_WEB={did_web}");
    println!(
        "export HYPRSTREAM__CLUSTER_ANCHOR_ROOT_CERT={}",
        root_cert_path.display()
    );
    println!("export HYPRSTREAM_INSTANCE={instance}");
    println!("serving https://localhost:{https_port}/.well-known/ — Ctrl-C to stop");

    // Park forever.
    tokio::signal::ctrl_c().await?;
    Ok(())
}
