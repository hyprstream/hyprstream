//! Cross-process credential revocation over a real transport.
//!
//! The policy service is the canonical revocation authority: it owns the one
//! durable store; every other process holds a `PolicyAuthorityRevocationStore`
//! that checks and publishes through the policy service over the RPC bus.
//! These tests drive that topology end-to-end: a real `PolicyService` served
//! over iroh (same harness as `policy_over_iroh.rs`), with two independent
//! RPC client stores simulating two non-policy processes.

mod support;

use std::sync::Arc;

use anyhow::Result;
use ed25519_dalek::SigningKey;
use rand::RngCore;
use tempfile::TempDir;
use tokio::sync::RwLock;

use git2db::Git2DB;
use hyprstream_core::auth::PolicyManager;
use hyprstream_core::config::TokenConfig;
use hyprstream_core::services::PolicyService;
use hyprstream_core::services::generated::policy_client::PolicyClient;
use hyprstream_core::services::revocation::PolicyAuthorityRevocationStore;

use hyprstream_rpc::auth::{
    CredentialId, CredentialRevocationStore, FileBackedCredentialRevocationStore,
    global_credential_revocation_store, set_global_credential_revocation_store,
};
use hyprstream_rpc::crypto::CryptoPolicy;
use hyprstream_rpc::crypto::hybrid_kem::{KemTrustStore, KeyedKemTrustStore};
use hyprstream_rpc::envelope::{
    EnvelopeVerifyConfig, InMemoryNonceCache, KeyedPqTrustStore, PqTrustStore, install_verify_config,
};
use hyprstream_rpc::node_identity::{derive_mesh_kem_recipient, derive_mesh_mldsa_key};
use hyprstream_rpc::rpc_client::RpcClientImpl;
use hyprstream_rpc::signer::LocalSigner;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::transport::iroh_rpc::{IrohRpcProtocolHandler, LocalServiceBridge};
use hyprstream_rpc::transport::iroh_substrate::{ALPN_HYPRSTREAM_RPC, IrohSubstrate, NoopHandler};
use hyprstream_rpc::transport::iroh_transport::IrohTransport;

use iroh::{EndpointAddr, TransportAddr};

const SERVER_SIGNING_SEED: [u8; 32] = [0x61; 32];
const PROCESS_A_SIGNING_SEED: [u8; 32] = [0xA1; 32];
const PROCESS_B_SIGNING_SEED: [u8; 32] = [0xB1; 32];

fn server_signing_key() -> SigningKey {
    SigningKey::from_bytes(&SERVER_SIGNING_SEED)
}

fn process_a_signing_key() -> SigningKey {
    SigningKey::from_bytes(&PROCESS_A_SIGNING_SEED)
}

fn process_b_signing_key() -> SigningKey {
    SigningKey::from_bytes(&PROCESS_B_SIGNING_SEED)
}

fn bind_mesh_anchor(store: &mut KeyedPqTrustStore, signing_key: &SigningKey) {
    let pq_signing_key = derive_mesh_mldsa_key(signing_key);
    store.bind(
        signing_key.verifying_key().to_bytes(),
        &ml_dsa::Keypair::verifying_key(&pq_signing_key),
    );
}

fn pq_trust_store() -> Arc<dyn PqTrustStore> {
    let mut store = KeyedPqTrustStore::new();
    bind_mesh_anchor(&mut store, &server_signing_key());
    bind_mesh_anchor(&mut store, &process_a_signing_key());
    bind_mesh_anchor(&mut store, &process_b_signing_key());
    Arc::new(store)
}

fn request_kem_store(server_signing: &SigningKey) -> Result<Arc<dyn KemTrustStore>> {
    let mut store = KeyedKemTrustStore::new();
    store.bind(
        server_signing.verifying_key().to_bytes(),
        derive_mesh_kem_recipient(server_signing)?.public(),
    );
    Ok(Arc::new(store))
}

fn install_hybrid_verify_config() -> Arc<dyn PqTrustStore> {
    let pq_store = pq_trust_store();
    let _ = install_verify_config(EnvelopeVerifyConfig {
        policy: CryptoPolicy::Hybrid,
        pq_store: Some(Arc::clone(&pq_store)),
    });
    pq_store
}

fn fresh_node_key() -> [u8; 32] {
    let mut k = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut k);
    k
}

fn direct_addr(substrate: &IrohSubstrate) -> EndpointAddr {
    EndpointAddr::from_parts(
        substrate.endpoint_id(),
        substrate
            .endpoint()
            .bound_sockets()
            .into_iter()
            .map(TransportAddr::Ip),
    )
}

async fn make_policy_service() -> Result<(PolicyService, SigningKey, TempDir)> {
    let temp = TempDir::new()?;
    let models_dir = temp.path().to_path_buf();
    let policies_dir = models_dir.join(".registry").join("policies");
    let policy_manager = Arc::new(PolicyManager::new(&policies_dir).await?);
    let git2db = Arc::new(RwLock::new(Git2DB::open(&models_dir).await?));
    let signing_key = server_signing_key();
    let service = PolicyService::new(
        Arc::clone(&policy_manager),
        Arc::new(signing_key.clone()),
        TokenConfig::default(),
        git2db,
        TransportConfig::inproc("policy-revocation-unused"),
    );
    Ok((service, signing_key, temp))
}

async fn serve_over_iroh(
    service: PolicyService,
    server_signing: &SigningKey,
) -> Result<(IrohSubstrate, EndpointAddr)> {
    let nonce_cache = Arc::new(InMemoryNonceCache::new());
    let bridge = LocalServiceBridge::spawn(service, nonce_cache, 0)?;
    let substrate = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("moq-not-wired"),
        IrohRpcProtocolHandler::new(bridge, server_signing.clone()),
    )
    .await?;
    let addr = direct_addr(&substrate);
    Ok((substrate, addr))
}

/// Build one simulated non-policy process: an iroh client whose
/// `PolicyAuthorityRevocationStore` delegates to the authority over RPC.
async fn revocation_store_for(
    server_addr: EndpointAddr,
    server_vk: ed25519_dalek::VerifyingKey,
    signing_key: SigningKey,
) -> Result<(IrohSubstrate, PolicyAuthorityRevocationStore)> {
    let client_substrate = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("c-moq"),
        NoopHandler::new("c-rpc"),
    )
    .await?;
    let conn = client_substrate
        .connect(server_addr, ALPN_HYPRSTREAM_RPC)
        .await?;
    let transport = IrohTransport::new(conn);
    let rpc = RpcClientImpl::new(LocalSigner::new(signing_key), transport, Some(server_vk))
        .with_request_kem_store(request_kem_store(&server_signing_key())?)
        .with_response_pq_store(pq_trust_store());
    let client = PolicyClient::new(Arc::new(rpc));
    Ok((client_substrate, PolicyAuthorityRevocationStore::new(client)))
}

/// Publish the authority-side durable store (tempdir-backed) as the
/// process-global store — the policy handlers resolve the store through the
/// global. Get-or-init: the OnceLock is per test-binary process.
fn install_authority_store(temp: &TempDir) -> Result<Arc<FileBackedCredentialRevocationStore>> {
    let path = temp.path().join("credential-revocations.jsonl");
    let store = Arc::new(FileBackedCredentialRevocationStore::open(&path)?);
    if global_credential_revocation_store().is_none() {
        set_global_credential_revocation_store(store.clone())
            .map_err(|e| anyhow::anyhow!("no other test in this binary publishes the global store: {e}"))?;
    }
    Ok(store)
}

/// THE cross-process proof: process A publishes a revocation over RPC;
/// process B observes it over RPC. Issuer scoping and the JWT/CWT typed
/// namespaces survive the wire round-trip.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn revocation_is_visible_across_processes() -> Result<()> {
    support::install_explicit_dispatch_pep();
    install_hybrid_verify_config();

    let store_temp = TempDir::new()?;
    install_authority_store(&store_temp)?;

    let (service, server_signing, _temp) = make_policy_service().await?;
    let server_vk = server_signing.verifying_key();
    let (server, server_addr) = serve_over_iroh(service, &server_signing).await?;

    let (substrate_a, process_a) =
        revocation_store_for(server_addr.clone(), server_vk, process_a_signing_key()).await?;
    let (substrate_b, process_b) =
        revocation_store_for(server_addr, server_vk, process_b_signing_key()).await?;

    let issuer = "https://issuer.example";
    let id = CredentialId::jwt(issuer, "jti-cross-process");
    let exp = chrono::Utc::now().timestamp() + 3600;

    // Unrevoked ID reads false from both processes.
    assert!(
        !process_b.is_revoked(&id).await,
        "unrevoked credential must read false"
    );

    // Process A revokes; process B observes it.
    process_a.revoke_credential(id.clone(), exp).await?;
    assert!(
        process_b.is_revoked(&id).await,
        "process B must observe process A's revocation"
    );

    // Cross-issuer same-jti is NOT revoked (issuer scoping over the wire).
    let other_issuer = CredentialId::jwt("https://other.example", "jti-cross-process");
    assert!(
        !process_b.is_revoked(&other_issuer).await,
        "cross-issuer same-jti must NOT be revoked"
    );

    // CWT cti bytes vs the same bytes as a JWT jti: disjoint namespaces over
    // the wire.
    let cwt_id = CredentialId::cwt(issuer, b"same-identifier-bytes".to_vec());
    process_a.revoke_credential(cwt_id.clone(), exp).await?;
    assert!(
        process_b.is_revoked(&cwt_id).await,
        "CWT cti revocation must be visible cross-process"
    );
    let jwt_same_bytes = CredentialId::jwt(issuer, "same-identifier-bytes");
    assert!(
        !process_b.is_revoked(&jwt_same_bytes).await,
        "JWT jti with the same bytes must NOT collide with the CWT cti"
    );

    drop(process_a);
    drop(process_b);
    substrate_a.shutdown().await?;
    substrate_b.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}

/// A revocation client store whose authority is unreachable fails closed:
/// checks report revoked and publications return an error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unreachable_authority_fails_closed() -> Result<()> {
    use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;

    let signing_key = process_a_signing_key();
    let remote_vk = server_signing_key().verifying_key();
    let rpc = RpcClientImpl::new(
        LocalSigner::new(signing_key),
        LazyUdsTransport::new("/dev/null/policy-revocation-dead.sock".into()),
        Some(remote_vk),
    );
    let store = PolicyAuthorityRevocationStore::new(PolicyClient::new(Arc::new(rpc)));

    let id = CredentialId::jwt("https://issuer.example", "jti-unreachable");
    assert!(
        store.is_revoked(&id).await,
        "check against a dead authority must fail closed"
    );
    let result = store
        .revoke_credential(id, chrono::Utc::now().timestamp() + 3600)
        .await;
    assert!(
        result.is_err(),
        "publication against a dead authority must fail"
    );
    Ok(())
}

/// The authority's durable store: revocations survive a process restart
/// (drop + re-open from the same path), expired entries are dropped on load,
/// and a corrupt file fails closed at open.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn file_backed_store_is_durable() -> Result<()> {
    let temp = TempDir::new()?;
    let path = temp.path().join("credential-revocations.jsonl");
    let now = chrono::Utc::now().timestamp();

    let live = CredentialId::jwt("https://issuer.example", "jti-durable");
    let live_cwt = CredentialId::cwt("https://issuer.example", b"cti-durable".to_vec());
    let expired = CredentialId::jwt("https://issuer.example", "jti-expired");
    {
        let store = FileBackedCredentialRevocationStore::open(&path)?;
        store.revoke_credential(live.clone(), now + 3600).await?;
        store.revoke_credential(live_cwt.clone(), now + 3600).await?;
        store.revoke_credential(expired.clone(), now - 10).await?;
    }

    // Re-open from the same path: live entries survive, expired are dropped.
    let reopened = FileBackedCredentialRevocationStore::open(&path)?;
    assert!(reopened.is_revoked(&live).await, "JWT revocation must survive reopen");
    assert!(
        reopened.is_revoked(&live_cwt).await,
        "CWT revocation must survive reopen"
    );
    assert!(
        !reopened.is_revoked(&expired).await,
        "expired entries are dropped on load"
    );
    drop(reopened);

    // Corrupt the log: open must fail closed.
    std::fs::write(&path, b"{not json\n")?;
    assert!(
        FileBackedCredentialRevocationStore::open(&path).is_err(),
        "corrupt revocation log must fail open"
    );
    Ok(())
}
