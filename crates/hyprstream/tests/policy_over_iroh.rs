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
use hyprstream_core::services::generated::policy_client::{
    PolicyCheck, PolicyClient, RegisterEventPrefix, SubscribeEventPrefix,
};

use hyprstream_rpc::crypto::CryptoPolicy;
use hyprstream_rpc::crypto::hybrid_kem::{KemTrustStore, KeyedKemTrustStore};
use hyprstream_rpc::envelope::{
    EnvelopeVerifyConfig, InMemoryNonceCache, KeyedPqTrustStore, PqTrustStore,
    install_verify_config,
};
use hyprstream_rpc::node_identity::{derive_mesh_kem_recipient, derive_mesh_mldsa_key};
use hyprstream_rpc::rpc_client::RpcClientImpl;
use hyprstream_rpc::signer::LocalSigner;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::transport::iroh_rpc::{IrohRpcProtocolHandler, LocalServiceBridge};
use hyprstream_rpc::transport::iroh_substrate::{ALPN_HYPRSTREAM_RPC, IrohSubstrate, NoopHandler};
use hyprstream_rpc::transport::iroh_transport::IrohTransport;

use iroh::{EndpointAddr, TransportAddr};

const SERVICE_SIGNING_SEED: [u8; 32] = [0x51; 32];
const CLIENT_SIGNING_SEED: [u8; 32] = [0xC1; 32];
/// #1128 spike: two more fixed-seed client identities so tests can drive
/// per-subject authorization outcomes over the wire.
const CLIENT_A_SIGNING_SEED: [u8; 32] = [0xC2; 32];
const CLIENT_B_SIGNING_SEED: [u8; 32] = [0xC3; 32];
const TEST_ISSUER: &str = "https://policy.test";

/// Subjects the test trust-store attestations bind to client A / B keys.
const TENANT_A_SUBJECT: &str = "tenant-a-user";
const TENANT_B_SUBJECT: &str = "tenant-b-user";

fn policy_service_signing_key() -> SigningKey {
    SigningKey::from_bytes(&SERVICE_SIGNING_SEED)
}

fn policy_client_signing_key() -> SigningKey {
    SigningKey::from_bytes(&CLIENT_SIGNING_SEED)
}

fn policy_client_a_signing_key() -> SigningKey {
    SigningKey::from_bytes(&CLIENT_A_SIGNING_SEED)
}

fn policy_client_b_signing_key() -> SigningKey {
    SigningKey::from_bytes(&CLIENT_B_SIGNING_SEED)
}

/// Bind the fixed-seed client keys to bare user subjects in the global trust
/// store (`hyprstream-service`), which `PolicyService::resolve_key_subject`
/// consults on the networked (AnySigner) path. Without this, over-iroh
/// callers resolve as `anonymous` and every policy-gated RPC is denied.
/// Idempotent: inserting the same fixed-seed attestation twice keeps the
/// first entry (equal-expiry first-writer wins).
fn install_test_key_subjects() {
    let store = hyprstream_service::global_trust_store();
    for (key, subject) in [
        (policy_client_a_signing_key(), TENANT_A_SUBJECT),
        (policy_client_b_signing_key(), TENANT_B_SUBJECT),
    ] {
        store.insert(
            key.verifying_key(),
            hyprstream_service::Attestation {
                scopes: std::collections::HashSet::new(),
                subject: Some(subject.to_owned()),
                jwt: None,
                expires_at: 0, // never expires
                attested_by: None,
            },
        );
    }
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
    bind_mesh_anchor(&mut store, &policy_client_a_signing_key());
    bind_mesh_anchor(&mut store, &policy_client_b_signing_key());
    Arc::new(store)
}

fn policy_request_kem_store(server_signing: &SigningKey) -> Result<Arc<dyn KemTrustStore>> {
    let mut store = KeyedKemTrustStore::new();
    store.bind(
        server_signing.verifying_key().to_bytes(),
        derive_mesh_kem_recipient(server_signing)?.public(),
    );
    Ok(Arc::new(store))
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

struct TestTokenAuthority {
    ml_dsa: Arc<hyprstream_rpc::crypto::pq::MlDsaSigningKey>,
    ed25519: Arc<SigningKey>,
}

impl TestTokenAuthority {
    fn new(
        root_signing_key: &SigningKey,
    ) -> Result<(
        Self,
        Arc<hyprstream_rpc::auth::CompositeKeySet>,
    )> {
        use hyprstream_rpc::auth::{
            CompositeKeyPair, CompositeKeySet, CompositePairRole, CompositePairState,
        };

        let ed25519 = Arc::new(hyprstream_rpc::node_identity::derive_purpose_key(
            root_signing_key,
            "hyprstream-jwt-v1",
        ));
        let (ml_dsa, ml_dsa_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let ml_dsa = Arc::new(ml_dsa);
        let kid = hyprstream_core::auth::jwt::composite_kid(
            &ml_dsa_vk,
            &ed25519.verifying_key(),
        );
        let pair = CompositeKeyPair::verifying(
            kid,
            ml_dsa_vk,
            ed25519.verifying_key(),
            CompositePairRole::Policy,
            CompositePairState::Active,
            0,
            i64::MAX,
        );
        let key_set = Arc::new(CompositeKeySet::default());
        key_set.publish(1, "policy-over-iroh".to_owned(), vec![pair])?;
        Ok((Self { ml_dsa, ed25519 }, key_set))
    }

    fn token(&self, subject: &str, tenant: &str, signer: &SigningKey) -> String {
        let now = chrono::Utc::now().timestamp();
        let claims = hyprstream_rpc::auth::Claims::new(subject.to_owned(), now, now + 300)
            .with_issuer(TEST_ISSUER.to_owned())
            .with_tenant(tenant.to_owned())
            .with_cnf_jwk(signer.verifying_key().as_bytes());
        hyprstream_core::auth::jwt::encode_composite_ml_dsa_65_ed25519(
            &claims,
            &self.ml_dsa,
            &self.ed25519,
        )
    }
}

/// Stand up a real `PolicyService` in a fresh tempdir and return it
/// alongside its signing key and the tempdir guard (which must outlive
/// the service).
async fn make_policy_service() -> Result<(PolicyService, SigningKey, TempDir)> {
    let (service, _manager, signing_key, temp, _authority) =
        make_policy_service_with_manager_and_authority().await?;
    Ok((service, signing_key, temp))
}

/// Like [`make_policy_service`] but also returns the `Arc<PolicyManager>` so
/// tests can write grants (e.g. per-tenant domain rules for #1128) before
/// serving.
async fn make_policy_service_with_manager(
) -> Result<(PolicyService, Arc<PolicyManager>, SigningKey, TempDir)> {
    let (service, manager, signing_key, temp, _authority) =
        make_policy_service_with_manager_and_authority().await?;
    Ok((service, manager, signing_key, temp))
}

async fn make_policy_service_with_manager_and_authority(
) -> Result<(
    PolicyService,
    Arc<PolicyManager>,
    SigningKey,
    TempDir,
    TestTokenAuthority,
)> {
    let temp = TempDir::new()?;
    let models_dir = temp.path().to_path_buf();
    let policies_dir = models_dir.join(".registry").join("policies");

    // PolicyManager bootstraps with SERVICE_BASE_POLICIES so service:*
    // authorization works without any extra rule writes.
    let policy_manager = Arc::new(PolicyManager::new(&policies_dir).await?);

    // Git2DB::open initializes .registry as a git repo at this path.
    let git2db = Arc::new(RwLock::new(Git2DB::open(&models_dir).await?));

    let signing_key = policy_service_signing_key();
    let (authority, composite_keys) = TestTokenAuthority::new(&signing_key)?;
    let key_source = hyprstream_rpc::auth::ClusterKeySource::new(
        authority.ed25519.verifying_key(),
        TEST_ISSUER.to_owned(),
    )
    .with_composite_key_set(composite_keys);
    let service = PolicyService::new(
        Arc::clone(&policy_manager),
        Arc::new(signing_key.clone()),
        TokenConfig::default(),
        git2db,
        TransportConfig::inproc("policy-over-iroh-unused"),
    )
    .with_jwt_key_source(Arc::new(key_source));

    Ok((service, policy_manager, signing_key, temp, authority))
}

/// Build the iroh server side: PolicyService → LocalServiceBridge →
/// IrohRpcProtocolHandler → IrohSubstrate.
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

/// Build the client side: iroh dial → IrohTransport → RpcClientImpl →
/// PolicyClient.
async fn client_for(
    server_addr: EndpointAddr,
    server_vk: ed25519_dalek::VerifyingKey,
    request_kem_store: Arc<dyn KemTrustStore>,
    response_pq_store: Arc<dyn PqTrustStore>,
) -> Result<(IrohSubstrate, PolicyClient)> {
    client_for_key(
        server_addr,
        server_vk,
        request_kem_store,
        response_pq_store,
        policy_client_signing_key(),
    )
    .await
}

/// Like [`client_for`] but the client signs with `signing_key` — used by the
/// #1128 spike tests to distinguish subject A from subject B on the wire.
async fn client_for_key(
    server_addr: EndpointAddr,
    server_vk: ed25519_dalek::VerifyingKey,
    request_kem_store: Arc<dyn KemTrustStore>,
    response_pq_store: Arc<dyn PqTrustStore>,
    signing_key: SigningKey,
) -> Result<(IrohSubstrate, PolicyClient)> {
    client_for_key_with_jwt(
        server_addr,
        server_vk,
        request_kem_store,
        response_pq_store,
        signing_key,
        None,
    )
    .await
}

async fn client_for_key_with_jwt(
    server_addr: EndpointAddr,
    server_vk: ed25519_dalek::VerifyingKey,
    request_kem_store: Arc<dyn KemTrustStore>,
    response_pq_store: Arc<dyn PqTrustStore>,
    signing_key: SigningKey,
    jwt: Option<String>,
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
    let mut rpc = RpcClientImpl::new(
        LocalSigner::new(signing_key),
        transport,
        Some(server_vk),
    )
    .with_request_kem_store(request_kem_store)
    .with_response_pq_store(response_pq_store);
    if let Some(jwt) = jwt {
        rpc = rpc.with_default_jwt(jwt);
    }
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
    let request_kem_store = policy_request_kem_store(&server_signing)?;

    let (server, server_addr) = serve_over_iroh(service, &server_signing).await?;
    let (client_substrate, policy) =
        client_for(server_addr, server_vk, request_kem_store, pq_store).await?;

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
    let request_kem_store = policy_request_kem_store(&server_signing)?;

    let (server, server_addr) = serve_over_iroh(service, &server_signing).await?;
    let (client_substrate, policy) =
        client_for(server_addr, server_vk, request_kem_store, pq_store).await?;
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

// ============================================================================
// Issue #1128 spike: tenancy-isolation evidence tests
//
// These tests document the CURRENT behavior of the PolicyService RPC path
// with respect to tenants (Casbin domains) and event-prefix ownership.
// Tests named `*_gap1128` assert behavior that is known-broken for
// multi-tenancy; they are expected to PASS against today's code and to be
// inverted when the fix lands. Every gap assertion names the production
// site responsible.
// ============================================================================

/// Baseline (expected green, not a gap): two clients with distinct signing
/// keys resolve to distinct subjects over iroh, and a Casbin grant for
/// subject A does not leak to subject B. Proves per-subject isolation works
/// on the RPC path — the missing piece for #1128 is domains, not subjects.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_subjects_per_subject_isolation_over_rpc() -> Result<()> {
    let pq_store = install_hybrid_verify_config();
    install_test_key_subjects();

    let (service, manager, server_signing, _temp, authority) =
        make_policy_service_with_manager_and_authority().await?;
    // Grant subject A (and only A): (1) the transport-level dispatch gate for
    // the method itself ($scope(manage) on `policy:RegisterEventPrefix`,
    // policy.capnp:64 — generated dispatch calls authorize() before the
    // handler) and (2) the handler-level publish scope for one event prefix.
    manager
        .add_policy_with_domain(
            TENANT_A_SUBJECT,
            "*",
            "policy:RegisterEventPrefix",
            "manage",
            "allow",
        )
        .await?;
    manager
        .add_policy_with_domain(
            TENANT_A_SUBJECT,
            "*",
            "publish:events:alpha.*",
            "register",
            "allow",
        )
        .await?;

    let server_vk = server_signing.verifying_key();
    let (server, server_addr) = serve_over_iroh(service, &server_signing).await?;

    let client_a_key = policy_client_a_signing_key();
    let client_a_jwt = authority.token(TENANT_A_SUBJECT, "tenant-a", &client_a_key);
    let (client_a_substrate, policy_a) = client_for_key_with_jwt(
        server_addr.clone(),
        server_vk,
        policy_request_kem_store(&server_signing)?,
        Arc::clone(&pq_store),
        client_a_key,
        Some(client_a_jwt),
    )
    .await?;
    let client_b_key = policy_client_b_signing_key();
    let client_b_jwt = authority.token(TENANT_B_SUBJECT, "tenant-b", &client_b_key);
    let (client_b_substrate, policy_b) = client_for_key_with_jwt(
        server_addr,
        server_vk,
        policy_request_kem_store(&server_signing)?,
        pq_store,
        client_b_key,
        Some(client_b_jwt),
    )
    .await?;

    // A is allowed: the grant names A's key-derived subject.
    policy_a
        .register_event_prefix(&RegisterEventPrefix {
            prefix: "alpha".to_owned(),
            publisher_ephemeral_pubkey: vec![0x0A; 32],
            schema: String::new(),
        })
        .await
        .map_err(|e| anyhow::anyhow!("subject A must be allowed to register: {e}"))?;

    // B has no grant: same class of operation is denied for B's subject.
    let err = policy_b
        .register_event_prefix(&RegisterEventPrefix {
            prefix: "beta".to_owned(),
            publisher_ephemeral_pubkey: vec![0x0B; 32],
            schema: String::new(),
        })
        .await;
    let Err(err) = err else {
        panic!("subject B must be denied (no grant)");
    };
    assert!(
        err.to_string().contains("Unauthorized"),
        "denial should be an authorization failure, got: {err}"
    );

    client_a_substrate.shutdown().await?;
    client_b_substrate.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}

/// GAP (#1128): a grant written in Casbin domain "tenant-a" is INERT on the
/// RPC path. The RPC authorize path hardcodes the request domain to "*"
/// (registry.rs:1701, policy.rs:271 — every `check_with_domain(caller, "*",
/// ...)` call), and the matcher only fires when `p.dom == "*" || r.dom ==
/// p.dom` (policy_manager.rs:152-155), so a non-wildcard-domain policy can
/// never authorize an RPC call. This test asserts the current broken
/// behavior; a fix must make the first assertion ALLOW.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn domain_grant_is_inert_over_rpc_gap1128() -> Result<()> {
    let pq_store = install_hybrid_verify_config();
    install_test_key_subjects();

    let (service, manager, server_signing, _temp) = make_policy_service_with_manager().await?;
    // Grant exists ONLY in domain "tenant-a".
    manager
        .add_policy_with_domain(
            TENANT_A_SUBJECT,
            "tenant-a",
            "data:orders",
            "read",
            "allow",
        )
        .await?;

    let server_vk = server_signing.verifying_key();
    let (server, server_addr) = serve_over_iroh(service, &server_signing).await?;
    let (client_substrate, policy) = client_for_key(
        server_addr,
        server_vk,
        policy_request_kem_store(&server_signing)?,
        pq_store,
        policy_client_a_signing_key(),
    )
    .await?;

    // The enforce path used by RPC authorization always passes r.dom="*":
    // with r.dom="*" the tenant-a grant does not match → DENIED. This is the
    // gap: the domain-scoped grant cannot take effect over RPC.
    let denied = policy
        .check(&PolicyCheck {
            subject: TENANT_A_SUBJECT.to_owned(),
            domain: "*".to_owned(),
            resource: "data:orders".to_owned(),
            operation: "read".to_owned(),
        })
        .await?;
    assert!(
        !denied,
        "GAP #1128: tenant-a domain grant must be inert when r.dom=\"*\" (RPC path)"
    );

    // Control: the grant itself is real — when the domain is evaluated as
    // "tenant-a" (which no RPC authorize call ever does today), it allows.
    let allowed = policy
        .check(&PolicyCheck {
            subject: TENANT_A_SUBJECT.to_owned(),
            domain: "tenant-a".to_owned(),
            resource: "data:orders".to_owned(),
            operation: "read".to_owned(),
        })
        .await?;
    assert!(
        allowed,
        "control: the tenant-a grant must match when the domain is actually evaluated"
    );

    client_substrate.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}

/// A verified tenant claim is the request domain. Two pairwise identities may
/// use different policy domains on one server, but cannot use each other's.
///
/// This is deliberately end-to-end: the server derives the domain from the
/// signed, verified JWT tenant claim rather than the subject or request data.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn verified_tenant_domains_isolate_pairwise_rpc_callers() -> Result<()> {
    let pq_store = install_hybrid_verify_config();
    install_test_key_subjects();

    let (service, manager, server_signing, _temp, authority) =
        make_policy_service_with_manager_and_authority().await?;
    for (subject, tenant, prefix) in [
        (TENANT_A_SUBJECT, "tenant-a", "alpha"),
        (TENANT_B_SUBJECT, "tenant-b", "beta"),
    ] {
        manager
            .add_policy_with_domain(
                subject,
                tenant,
                "policy:RegisterEventPrefix",
                "manage",
                "allow",
            )
            .await?;
        manager
            .add_policy_with_domain(
                subject,
                tenant,
                &format!("publish:events:{prefix}.*"),
                "register",
                "allow",
            )
            .await?;
    }
    // This would authorize an anonymous request under the old wildcard path.
    manager.add_policy_with_domain("anonymous", "*", "policy:RegisterEventPrefix", "manage", "allow").await?;
    manager.add_policy_with_domain("anonymous", "*", "publish:events:alpha.*", "register", "allow").await?;

    let server_vk = server_signing.verifying_key();
    let (server, server_addr) = serve_over_iroh(service, &server_signing).await?;
    let client_a_key = policy_client_a_signing_key();
    let client_a_jwt = authority.token(TENANT_A_SUBJECT, "tenant-a", &client_a_key);
    let (client_a_substrate, policy_a) = client_for_key_with_jwt(
        server_addr.clone(),
        server_vk,
        policy_request_kem_store(&server_signing)?,
        Arc::clone(&pq_store),
        client_a_key,
        Some(client_a_jwt),
    )
    .await?;
    let client_b_key = policy_client_b_signing_key();
    let client_b_jwt = authority.token(TENANT_B_SUBJECT, "tenant-b", &client_b_key);
    let (client_b_substrate, policy_b) = client_for_key_with_jwt(
        server_addr.clone(),
        server_vk,
        policy_request_kem_store(&server_signing)?,
        Arc::clone(&pq_store),
        client_b_key,
        Some(client_b_jwt),
    )
    .await?;
    let (anonymous_substrate, anonymous_policy) = client_for(server_addr, server_vk, policy_request_kem_store(&server_signing)?, pq_store).await?;

    policy_a
        .register_event_prefix(&RegisterEventPrefix {
            prefix: "alpha".to_owned(),
            publisher_ephemeral_pubkey: vec![0x0A; 32],
            schema: String::new(),
        })
        .await?;
    policy_b
        .register_event_prefix(&RegisterEventPrefix {
            prefix: "beta".to_owned(),
            publisher_ephemeral_pubkey: vec![0x0B; 32],
            schema: String::new(),
        })
        .await?;

    let err = policy_b
        .register_event_prefix(&RegisterEventPrefix {
            prefix: "alpha".to_owned(),
            publisher_ephemeral_pubkey: vec![0x0C; 32],
            schema: String::new(),
        })
        .await;
    let Err(err) = err else {
        panic!("tenant B must be denied in tenant A's domain");
    };
    assert!(
        err.to_string()
            .contains("cannot register on publish:events:alpha.*"),
        "cross-domain denial must come from Casbin, got: {err}"
    );

    let err = anonymous_policy
        .register_event_prefix(&RegisterEventPrefix {
            prefix: "alpha".to_owned(),
            publisher_ephemeral_pubkey: vec![0x0D; 32],
            schema: String::new(),
        })
        .await;
    let Err(err) = err else {
        panic!("domain-less request must not fall back to wildcard policy");
    };
    assert!(err.to_string().contains("no verified tenant domain"), "domain-less caller must fail closed, got: {err}");

    client_a_substrate.shutdown().await?;
    client_b_substrate.shutdown().await?;
    anonymous_substrate.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}

/// Equal raw event prefixes are independent when their verified tenant differs.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn event_prefixes_are_isolated_across_verified_tenants() -> Result<()> {
    let pq_store = install_hybrid_verify_config();
    install_test_key_subjects();

    const PUBKEY_A: [u8; 32] = [0x0A; 32];
    const PUBKEY_B: [u8; 32] = [0x0B; 32];

    let (service, manager, server_signing, _temp, authority) =
        make_policy_service_with_manager_and_authority().await?;
    // Both subjects can register the same raw prefix, but only in their own
    // verified tenant. A can subscribe there to verify B cannot replace it.
    for (subject, tenant) in [
        (TENANT_A_SUBJECT, "tenant-a"),
        (TENANT_B_SUBJECT, "tenant-b"),
    ] {
        manager
            .add_policy_with_domain(
                subject,
                tenant,
                "policy:RegisterEventPrefix",
                "manage",
                "allow",
            )
            .await?;
        manager
            .add_policy_with_domain(
                subject,
                tenant,
                "publish:events:orders.*",
                "register",
                "allow",
            )
            .await?;
    }
    manager
        .add_policy_with_domain(
            TENANT_A_SUBJECT,
            "tenant-a",
            "policy:SubscribeEventPrefix",
            "manage",
            "allow",
        )
        .await?;
    manager
        .add_policy_with_domain(
            TENANT_A_SUBJECT,
            "tenant-a",
            "subscribe:events:orders.*",
            "subscribe",
            "allow",
        )
        .await?;

    let server_vk = server_signing.verifying_key();
    let (server, server_addr) = serve_over_iroh(service, &server_signing).await?;
    let client_a_key = policy_client_a_signing_key();
    let client_a_jwt = authority.token(TENANT_A_SUBJECT, "tenant-a", &client_a_key);
    let (client_a_substrate, policy_a) = client_for_key_with_jwt(
        server_addr.clone(),
        server_vk,
        policy_request_kem_store(&server_signing)?,
        Arc::clone(&pq_store),
        client_a_key,
        Some(client_a_jwt),
    )
    .await?;
    let client_b_key = policy_client_b_signing_key();
    let client_b_jwt = authority.token(TENANT_B_SUBJECT, "tenant-b", &client_b_key);
    let (client_b_substrate, policy_b) = client_for_key_with_jwt(
        server_addr,
        server_vk,
        policy_request_kem_store(&server_signing)?,
        pq_store,
        client_b_key,
        Some(client_b_jwt),
    )
    .await?;

    // A registers the prefix.
    policy_a
        .register_event_prefix(&RegisterEventPrefix {
            prefix: "orders".to_owned(),
            publisher_ephemeral_pubkey: PUBKEY_A.to_vec(),
            schema: "schema-v1".to_owned(),
        })
        .await?;

    // Sanity: subscribers see A's publisher key.
    let access = policy_a
        .subscribe_event_prefix(&SubscribeEventPrefix {
            prefix: "orders".to_owned(),
            subscriber_ephemeral_pubkey: vec![0x5A; 32],
        })
        .await?;
    assert_eq!(
        access.publisher_ephemeral_pubkey,
        PUBKEY_A.to_vec(),
        "before takeover the stored publisher key must be A's"
    );

    // B registers the same raw prefix in tenant-b; this must not replace A's
    // tenant-a registry or transport identity.
    policy_b
        .register_event_prefix(&RegisterEventPrefix {
            prefix: "orders".to_owned(),
            publisher_ephemeral_pubkey: PUBKEY_B.to_vec(),
            schema: "schema-evil".to_owned(),
        })
        .await?;

    // Tenant A still observes its own publisher key and schema.
    let access = policy_a
        .subscribe_event_prefix(&SubscribeEventPrefix {
            prefix: "orders".to_owned(),
            subscriber_ephemeral_pubkey: vec![0x5B; 32],
        })
        .await?;
    assert_eq!(
        access.publisher_ephemeral_pubkey,
        PUBKEY_A.to_vec(),
        "tenant-b registration must not replace tenant-a publisher state"
    );
    assert_eq!(
        access.schema, "schema-v1",
        "tenant-b registration must not replace tenant-a schema"
    );

    client_a_substrate.shutdown().await?;
    client_b_substrate.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}
