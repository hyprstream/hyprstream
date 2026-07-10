//! Phase 2 part 3: real `PolicyService` exercised end-to-end over iroh.
//!
//! Constructs a real `PolicyService` (PolicyManager + Git2DB + signing key,
//! all in a `TempDir`), serves it through a `LocalServiceBridge` behind an
//! `IrohRpcProtocolHandler` on ALPN `hyprstream-rpc/1`, and drives the
//! generated `PolicyClient` against it via `IrohTransport` + `RpcClientImpl`.
//!
//! This validates the canary path for #133: every byte that the production
//! `create_policy_service` factory would terminate on the ZMTP wire also
//! works unchanged over iroh, because we reuse the same `process_request`
//! envelope-verify/JWT/Casbin path via the bridge.
//!
//! Factory wiring (so deployments actually serve over iroh) is a separate
//! follow-up under Phase 5 (#136), since every service will need the same
//! change at the same place.

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
use hyprstream_core::services::generated::policy_client::{PolicyCheck, PolicyClient};

use hyprstream_rpc::crypto::CryptoPolicy;
use hyprstream_rpc::envelope::{
    EnvelopeVerifyConfig, InMemoryNonceCache, KeyedPqTrustStore, PqTrustStore,
    install_verify_config,
};
use hyprstream_rpc::node_identity::derive_mesh_mldsa_key;
use hyprstream_rpc::rpc_client::RpcClientImpl;
use hyprstream_rpc::signer::LocalSigner;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::transport::iroh_rpc::{IrohRpcProtocolHandler, LocalServiceBridge};
use hyprstream_rpc::transport::iroh_substrate::{ALPN_HYPRSTREAM_RPC, IrohSubstrate, NoopHandler};
use hyprstream_rpc::transport::iroh_transport::IrohTransport;

use iroh::{EndpointAddr, TransportAddr};

const SERVICE_SIGNING_SEED: [u8; 32] = [0x51; 32];
const CLIENT_SIGNING_SEED: [u8; 32] = [0xC1; 32];

fn policy_service_signing_key() -> SigningKey {
    SigningKey::from_bytes(&SERVICE_SIGNING_SEED)
}

fn policy_client_signing_key() -> SigningKey {
    SigningKey::from_bytes(&CLIENT_SIGNING_SEED)
}

fn bind_mesh_anchor(store: &mut KeyedPqTrustStore, signing_key: &SigningKey) {
    let pq_signing_key = derive_mesh_mldsa_key(signing_key);
    store.bind(
        signing_key.verifying_key().to_bytes(),
        &ml_dsa::Keypair::verifying_key(&pq_signing_key),
    );
}

fn policy_pq_trust_store() -> Arc<dyn PqTrustStore> {
    let mut store = KeyedPqTrustStore::new();
    bind_mesh_anchor(&mut store, &policy_service_signing_key());
    bind_mesh_anchor(&mut store, &policy_client_signing_key());
    Arc::new(store)
}

fn install_hybrid_verify_config() -> Arc<dyn PqTrustStore> {
    let pq_store = policy_pq_trust_store();
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

/// Stand up a real `PolicyService` in a fresh tempdir and return it
/// alongside its signing key and the tempdir guard (which must outlive
/// the service).
async fn make_policy_service() -> Result<(PolicyService, SigningKey, TempDir)> {
    let temp = TempDir::new()?;
    let models_dir = temp.path().to_path_buf();
    let policies_dir = models_dir.join(".registry").join("policies");

    // PolicyManager bootstraps with SERVICE_BASE_POLICIES so service:*
    // authorization works without any extra rule writes.
    let policy_manager = Arc::new(PolicyManager::new(&policies_dir).await?);

    // Git2DB::open initializes .registry as a git repo at this path.
    let git2db = Arc::new(RwLock::new(Git2DB::open(&models_dir).await?));

    let signing_key = policy_service_signing_key();
    let service = PolicyService::new(
        policy_manager,
        Arc::new(signing_key.clone()),
        TokenConfig::default(),
        git2db,
        TransportConfig::inproc("policy-over-iroh-unused"),
    );

    Ok((service, signing_key, temp))
}

/// Build the iroh server side: PolicyService → LocalServiceBridge →
/// IrohRpcProtocolHandler → IrohSubstrate.
async fn serve_over_iroh(
    service: PolicyService,
    server_signing: SigningKey,
) -> Result<(IrohSubstrate, EndpointAddr)> {
    let nonce_cache = Arc::new(InMemoryNonceCache::new());
    let bridge = LocalServiceBridge::spawn(service, nonce_cache, 0)?;

    let substrate = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("moq-not-wired"),
        IrohRpcProtocolHandler::new(bridge, server_signing),
    )
    .await?;
    let addr = direct_addr(&substrate);
    Ok((substrate, addr))
}

/// Build the client side: iroh dial → IrohTransport → RpcClientImpl →
/// PolicyClient.
async fn client_for(
    server_addr: EndpointAddr,
    server_vk: ed25519_dalek::VerifyingKey,
    response_pq_store: Arc<dyn PqTrustStore>,
) -> Result<(IrohSubstrate, PolicyClient)> {
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
    let rpc = RpcClientImpl::new(
        LocalSigner::new(policy_client_signing_key()),
        transport,
        Some(server_vk),
    )
    .with_response_pq_store(response_pq_store);
    let policy_client = PolicyClient::new(Arc::new(rpc));
    Ok((client_substrate, policy_client))
}

/// Real PolicyService over iroh — verifies the canary path end-to-end with
/// a representative `check` RPC, including both ALLOW and DENY outcomes
/// produced by the bootstrap `SERVICE_BASE_POLICIES`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn policy_check_allow_and_deny_over_iroh() -> Result<()> {
    let pq_store = install_hybrid_verify_config();

    let (service, server_signing, _temp) = make_policy_service().await?;
    let server_vk = server_signing.verifying_key();

    let (server, server_addr) = serve_over_iroh(service, server_signing).await?;
    let (client_substrate, policy) = client_for(server_addr, server_vk, pq_store).await?;

    // ALLOW: service:oauth → policy:ResolveServiceKey:query (from
    // SERVICE_BASE_POLICIES, asserted in policy_manager.rs base-rules test).
    let allow_resp = policy
        .check(&PolicyCheck {
            subject: "service:oauth".to_owned(),
            domain: "*".to_owned(),
            resource: "policy:ResolveServiceKey".to_owned(),
            operation: "query".to_owned(),
        })
        .await?;
    assert!(allow_resp, "service:oauth must be allowed: {allow_resp:?}");

    // DENY: service:unknown is not in the base policies.
    let deny_resp = policy
        .check(&PolicyCheck {
            subject: "service:unknown".to_owned(),
            domain: "*".to_owned(),
            resource: "policy:ResolveServiceKey".to_owned(),
            operation: "query".to_owned(),
        })
        .await?;
    assert!(!deny_resp, "service:unknown must be denied: {deny_resp:?}");

    client_substrate.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}

/// Concurrent `check` calls on the same `PolicyClient` (one iroh connection,
/// many bidi streams) — verifies the wire layer's per-stream correlation
/// holds under a real service's response shape.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn policy_concurrent_checks_over_iroh() -> Result<()> {
    let pq_store = install_hybrid_verify_config();

    let (service, server_signing, _temp) = make_policy_service().await?;
    let server_vk = server_signing.verifying_key();

    let (server, server_addr) = serve_over_iroh(service, server_signing).await?;
    let (client_substrate, policy) = client_for(server_addr, server_vk, pq_store).await?;
    let policy = Arc::new(policy);

    // Cases chosen so wildcard base rules don't interfere:
    //   - policy:ResolveServiceKey:query is granted ONLY to specific services
    //     (service:oauth, service:policy, service:model, service:registry,
    //     service:worker, service:mcp), so unknown service:* subjects deny.
    //   - policy:Check:check is similarly per-service.
    //   - Non-`service:` subjects (e.g. "alice") deny on everything below
    //     unless explicitly granted.
    let cases: Vec<(&str, &str, &str, bool)> = vec![
        ("service:oauth", "policy:ResolveServiceKey", "query", true),
        ("service:policy", "policy:ResolveServiceKey", "query", true),
        ("service:model", "policy:Check", "check", true),
        ("service:oauth", "policy:IssueToken", "manage", true),
        ("service:bogus0", "policy:ResolveServiceKey", "query", false),
        ("service:bogus1", "policy:Check", "check", false),
        ("alice", "policy:ResolveServiceKey", "query", false),
        ("alice", "policy:IssueToken", "manage", false),
    ];
    let mut handles = Vec::new();
    for (subject, resource, operation, expect_allow) in cases {
        let policy = Arc::clone(&policy);
        let subject = subject.to_owned();
        let resource = resource.to_owned();
        let operation = operation.to_owned();
        handles.push(tokio::spawn(async move {
            let resp = policy
                .check(&PolicyCheck {
                    subject: subject.clone(),
                    domain: "*".to_owned(),
                    resource: resource.clone(),
                    operation,
                })
                .await?;
            assert_eq!(
                resp, expect_allow,
                "{subject} -> {resource}: expected allow={expect_allow}, got {resp:?}"
            );
            anyhow::Ok(())
        }));
    }
    for h in handles {
        h.await??;
    }

    client_substrate.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}
