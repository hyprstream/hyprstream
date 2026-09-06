//! #1499 focused boot-labeling acceptance: the fresh-state `registerServiceKey`
//! path through the **real production dispatch PEP**.
//!
//! This is the executable contract of the focused staging boot slice:
//!
//! 1. With the production PEP installed (`install_production_rpc_dispatch_pep`,
//!    the exact seam `service start` runs at `main.rs`), a fresh-state
//!    `registerServiceKey` from a declared bootstrap service (`discovery`)
//!    passes the PEP and reaches the real `PolicyService` handler — without
//!    `UnlabeledObject` and without a fabricated anonymous/public clearance
//!    (the caller presents a verified CA-signed service identity, and the PEP
//!    requires the declared service clearance).
//! 2. The causal twin — the identical caller and service with an undeclared
//!    leaf (`resolveServiceKey`) — denies `UnlabeledObject` before handler
//!    entry, and the PEP remains installed afterwards.
//! 3. A call to an undeclared service domain from the same declared caller
//!    denies before handler entry (handler invocation counter stays zero).

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::{SigningKey, VerifyingKey};

use hyprstream_core::auth::service_jwt::issue_or_load_service_jwt;
use hyprstream_core::auth::PolicyManager;
use hyprstream_core::config::TokenConfig;
use hyprstream_core::services::generated::policy_client::{RegisterServiceKey, ResolveServiceKey};
use hyprstream_core::services::{PolicyClient, PolicyService};
use hyprstream_rpc::auth::mac::global_mac_dispatch_pep;
use hyprstream_rpc::auth::ClusterKeySource;
use hyprstream_rpc::dial::{dial_with_crypto_stores, register_inproc};
use hyprstream_rpc::envelope::{InMemoryNonceCache, KeyedPqTrustStore};
use hyprstream_rpc::node_identity::{derive_mesh_mldsa_key, derive_purpose_key};
use hyprstream_rpc::service::{Continuation, EnvelopeContext, RequestService};
use hyprstream_rpc::signer::LocalSigner;
use hyprstream_rpc::transport::iroh_rpc::LocalServiceBridge;
use hyprstream_rpc::transport::rpc_session::IrohRequestProcessor;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_service::{InprocManager, ServiceManager as _};

const POLICY_ROOT_KEY: [u8; 32] = [0x52; 32];
const DISCOVERY_KEY: [u8; 32] = [0x42; 32];
const GHOST_CLIENT_KEY: [u8; 32] = [0x43; 32];
const ISSUER: &str = "http://127.0.0.1:6791";

/// Install this binary's process-wide hybrid trust view: envelope signature
/// verification (Hybrid policy) with the PQ anchors of the fixture keys.
/// These anchors authenticate keys; they grant no authorization.
fn install_crypto() {
    let mut store = KeyedPqTrustStore::new();
    for bytes in [POLICY_ROOT_KEY, DISCOVERY_KEY, GHOST_CLIENT_KEY] {
        let ed = SigningKey::from_bytes(&bytes);
        let pq = derive_mesh_mldsa_key(&ed);
        let pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
            &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq),
        )
        .expect("fixture ML-DSA key");
        store.bind(ed.verifying_key().to_bytes(), &pq_vk);
    }
    let _ = hyprstream_rpc::envelope::install_verify_config(
        hyprstream_rpc::envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(Arc::new(store)),
        },
    );
    let _ = hyprstream_rpc::envelope::install_response_verify_config(
        hyprstream_rpc::envelope::ResponseVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
            pq_store: None,
        },
    );
}

/// A real CA-signed service JWT for `service_name`, minted exactly as the
/// wizard/bootstrap path mints it (`service_jwt::issue_or_load_service_jwt`)
/// from the CA JWT key the PolicyService purpose-derives from its root key.
fn mint_service_jwt(
    dir: &tempfile::TempDir,
    service_name: &str,
    ca_jwt_key: &SigningKey,
    service_vk: &VerifyingKey,
) -> String {
    let now = chrono::Utc::now().timestamp();
    issue_or_load_service_jwt(
        dir.path(),
        service_name,
        ca_jwt_key,
        service_vk,
        ISSUER,
        now,
    )
    .expect("mint service JWT")
}

/// The PolicyService's JWT key source, wired the way the service factory wires
/// `ServiceContext::cluster_key_source()` for the offline (no JWKS fetcher)
/// path: the purpose-derived CA JWT key plus its composite pair.
fn cluster_key_source(ca_jwt_key: &SigningKey) -> Arc<ClusterKeySource> {
    let ca_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&derive_mesh_mldsa_key(ca_jwt_key));
    Arc::new(
        ClusterKeySource::new(ca_jwt_key.verifying_key(), ISSUER.to_owned())
            .with_ca_composite_key(ca_pq_vk),
    )
}

/// Spawn a real PolicyService on a unique inproc endpoint and return a client
/// bound to the given caller key + service JWT.
async fn spawn_policy_and_client(
    tag: &str,
    caller_key: &SigningKey,
    service_jwt: Option<String>,
) -> Result<PolicyClient> {
    let root_key = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca_jwt_key = derive_purpose_key(&root_key, "hyprstream-jwt-v1");

    let policy_dir = tempfile::TempDir::new()?;
    let git2db = Arc::new(tokio::sync::RwLock::new(
        git2db::Git2DB::open(policy_dir.path()).await?,
    ));
    let policy_service = PolicyService::new(
        Arc::new(PolicyManager::permissive().await?),
        Arc::new(root_key.clone()),
        TokenConfig::default(),
        git2db,
        TransportConfig::inproc(tag),
    )
    .with_jwt_key_source(cluster_key_source(&ca_jwt_key));
    let manager = InprocManager::new();
    let _handle = manager.spawn(Box::new(policy_service)).await?;

    PolicyClient::for_local_endpoint_bootstrap(
        &format!("inproc://{tag}"),
        caller_key.clone(),
        root_key.verifying_key(),
        service_jwt,
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fresh_state_register_service_key_passes_production_dispatch_pep() -> Result<()> {
    install_crypto();

    // The exact production install seam `service start` runs (main.rs).
    hyprstream_core::mac::install_production_rpc_dispatch_pep();
    assert!(
        global_mac_dispatch_pep().is_some(),
        "the production dispatch PEP must be installed"
    );

    let root_key = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca_jwt_key = derive_purpose_key(&root_key, "hyprstream-jwt-v1");
    let discovery_key = SigningKey::from_bytes(&DISCOVERY_KEY);
    let creds = tempfile::TempDir::new()?;
    let discovery_jwt = mint_service_jwt(
        &creds,
        "discovery",
        &ca_jwt_key,
        &discovery_key.verifying_key(),
    );

    let tag = format!("mac-1499-policy-{}", uuid::Uuid::new_v4());
    let client = spawn_policy_and_client(&tag, &discovery_key, Some(discovery_jwt.clone())).await?;

    // Positive: the exact fresh-state boot call — discovery registers its
    // signing key with the PolicyService CA. This must pass the production
    // PEP (typed declared (policy, registerServiceKey) + deliberate
    // service:discovery clearance) AND the handler's own CA-JWT checks.
    //
    // An `Ok` here is the proof the handler ran: the PEP deny, a claims
    // failure, and a handler error all surface as `Err` (the PEP's deny is
    // the signed error payload the client maps to `Err`, per the #1499
    // staging log "registerServiceKey RPC failed ... MAC deny: ...").
    let request = RegisterServiceKey {
        service_name: "discovery".to_owned(),
        verifying_key: discovery_key.verifying_key().as_bytes().to_vec(),
        service_jwt: discovery_jwt,
    };
    client
        .register_service_key(&request)
        .await
        .expect("declared registerServiceKey must pass the production dispatch PEP on fresh state");

    assert!(
        global_mac_dispatch_pep().is_some(),
        "the dispatch PEP must remain installed after a permit"
    );

    // Causal twin: identical caller, identical service, undeclared leaf.
    // `resolveServiceKey` is a real policy method (discriminant 17) that this
    // slice deliberately does NOT declare — declaration, not schema, is the
    // authority. It must deny before handler entry with UnlabeledObject.
    let undeclared = client
        .resolve_service_key(&ResolveServiceKey {
            service_name: "registry".to_owned(),
        })
        .await;
    let error = undeclared.expect_err("undeclared leaf must deny");
    assert!(
        format!("{error:?}").contains("UnlabeledObject"),
        "undeclared leaf must deny UnlabeledObject, got: {error:?}"
    );

    assert!(
        global_mac_dispatch_pep().is_some(),
        "the dispatch PEP must remain installed after a deny"
    );
    Ok(())
}

/// An undeclared service domain denies before handler entry, with the handler
/// never invoked — the staging failure shape (`subject=service:discovery`,
/// unknown object) now failing closed for the right reason.
struct CountingEchoService {
    name: &'static str,
    transport: TransportConfig,
    signing_key: SigningKey,
    invocations: Arc<AtomicUsize>,
    key_source: Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
}

#[async_trait(?Send)]
impl RequestService for CountingEchoService {
    async fn handle_request(
        &self,
        _ctx: &EnvelopeContext,
        payload: &[u8],
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        Ok((payload.to_vec(), None))
    }

    fn name(&self) -> &str {
        self.name
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    fn jwt_key_source(&self) -> Option<Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
        Some(Arc::clone(&self.key_source))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn undeclared_service_domain_denies_before_handler_entry() -> Result<()> {
    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep();

    let root_key = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca_jwt_key = derive_purpose_key(&root_key, "hyprstream-jwt-v1");
    let discovery_key = SigningKey::from_bytes(&DISCOVERY_KEY);
    let creds = tempfile::TempDir::new()?;
    let discovery_jwt = mint_service_jwt(
        &creds,
        "discovery",
        &ca_jwt_key,
        &discovery_key.verifying_key(),
    );

    // A live service whose domain is NOT in the declared table.
    let invocations = Arc::new(AtomicUsize::new(0));
    let service = CountingEchoService {
        name: "ghost",
        transport: TransportConfig::inproc("ghost"),
        signing_key: SigningKey::from_bytes(&GHOST_CLIENT_KEY),
        invocations: Arc::clone(&invocations),
        key_source: cluster_key_source(&ca_jwt_key),
    };
    let bridge = LocalServiceBridge::spawn(service, Arc::new(InMemoryNonceCache::new()), 0)?;
    let processor: Arc<dyn IrohRequestProcessor> = Arc::new(bridge);
    register_inproc("ghost", &processor);

    let ghost_signing = SigningKey::from_bytes(&GHOST_CLIENT_KEY);
    let ghost_pq = derive_mesh_mldsa_key(&ghost_signing);
    let ghost_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
        &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&ghost_pq),
    )?;
    let mut response_store = KeyedPqTrustStore::new();
    response_store.bind(ghost_signing.verifying_key().to_bytes(), &ghost_pq_vk);
    let client = dial_with_crypto_stores(
        &TransportConfig::inproc("ghost"),
        LocalSigner::new(discovery_key),
        Some(ghost_signing.verifying_key()),
        Some(discovery_jwt),
        None,
        Some(Arc::new(response_store)),
    )?;

    // A declared caller (service:discovery, verified JWT) calling an
    // undeclared service domain: deny before handler entry.
    let response = client
        .call_for_service("ghost", b"must-not-reach-handler".to_vec())
        .await?;
    assert!(
        response.is_empty(),
        "the MAC denial must return the service's signed error payload, never handler bytes"
    );
    assert_eq!(
        invocations.load(Ordering::SeqCst),
        0,
        "an undeclared service domain must deny before handler invocation"
    );
    assert!(global_mac_dispatch_pep().is_some());
    drop(processor);
    Ok(())
}
