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
use ed25519_dalek::SigningKey;

use hyprstream_core::auth::identity_store::BootstrapPubkey;
use hyprstream_core::auth::service_jwt::issue_or_load_service_jwt;
use hyprstream_core::auth::PolicyManager;
use hyprstream_core::config::TokenConfig;
use hyprstream_rpc_std::policy_client::{
    IssueToken, IssueTokenProfile, PolicyCheck, RegisterServiceKey, ResolveServiceKey,
};
use hyprstream_core::services::PolicyService;
use hyprstream_rpc_std::policy_client::PolicyClient;
use hyprstream_rpc::auth::mac::global_mac_dispatch_pep;
use hyprstream_rpc::auth::ClusterKeySource;
use hyprstream_rpc::dial::{dial_with_crypto_stores, register_inproc};
use hyprstream_rpc::envelope::{InMemoryNonceCache, KeyedPqTrustStore};
use hyprstream_rpc::node_identity::{derive_mesh_mldsa_key, derive_purpose_key};
use hyprstream_rpc::service::{Continuation, DecodedRequestBody, EnvelopeContext, RequestService};
use hyprstream_rpc::signer::LocalSigner;
use hyprstream_rpc::transport::iroh_rpc::LocalServiceBridge;
use hyprstream_rpc::transport::rpc_session::IrohRequestProcessor;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_service::{InprocManager, ServiceManager as _};

const POLICY_ROOT_KEY: [u8; 32] = [0x52; 32];
const DISCOVERY_KEY: [u8; 32] = [0x42; 32];
const GHOST_CLIENT_KEY: [u8; 32] = [0x43; 32];
const OAUTH_KEY: [u8; 32] = [0x44; 32];
const ISSUER: &str = "http://127.0.0.1:6791";

/// Install this binary's process-wide hybrid trust view: envelope signature
/// verification (Hybrid policy) with the PQ anchors of the fixture keys.
/// These anchors authenticate keys; they grant no authorization.
fn install_crypto() {
    // Match the Policy-host startup authority: the dispatch plane fails
    // closed on jti-bearing service credentials without the process-global
    // revocation store, even on a fresh deployment. Get-or-init an in-memory
    // authority for this binary.
    if hyprstream_rpc::auth::global_credential_revocation_store().is_none() {
        let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
            hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
        ));
    }
    let mut store = KeyedPqTrustStore::new();
    for bytes in [POLICY_ROOT_KEY, DISCOVERY_KEY, GHOST_CLIENT_KEY, OAUTH_KEY] {
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
    service_key: &SigningKey,
) -> String {
    let now = chrono::Utc::now().timestamp();
    let bootstrap = BootstrapPubkey::for_service_key(service_key)
        .expect("fixture service hybrid enrollment");
    issue_or_load_service_jwt(
        dir.path(),
        service_name,
        ca_jwt_key,
        &bootstrap,
        ISSUER,
        now,
        Some(&hyprstream_core::mac::dispatch_labels::BOOTSTRAP_SERVICE_CLEARANCE),
    )
    .expect("mint service JWT")
}

/// Register a distinct active interactive session through the same process-global
/// authority the production OAuth flow uses before it calls PolicyService.
async fn register_active_session(
    issuer: &str,
    sid: &str,
    subject: &str,
    tenant: &str,
) -> Result<()> {
    if hyprstream_rpc::auth::global_session_registry().is_none() {
        let _ = hyprstream_rpc::auth::set_global_session_registry(Arc::new(
            hyprstream_rpc::auth::InMemorySessionRegistry::new(),
        ));
    }
    let registry = hyprstream_rpc::auth::global_session_registry()
        .ok_or_else(|| anyhow::anyhow!("session registry was not initialized"))?;
    let now = chrono::Utc::now().timestamp();
    registry
        .register_session(
            hyprstream_rpc::auth::SessionKey::oidc(issuer, sid),
            hyprstream_rpc::auth::SessionState {
                subject: subject.to_owned(),
                tenant: tenant.to_owned(),
                kind: hyprstream_rpc::auth::SessionKind::Interactive,
                created_at: now,
                expires_at: now + 3600,
                status: hyprstream_rpc::auth::ActiveOrRevoked::Active,
                clearance_epoch: 0,
            },
        )
        .await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn authenticated_oauth_policy_check_reaches_the_real_handler() -> Result<()> {
    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep();

    let root_key = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca_jwt_key = derive_purpose_key(&root_key, "hyprstream-jwt-v1");
    let oauth_key = SigningKey::from_bytes(&OAUTH_KEY);
    let creds = tempfile::TempDir::new()?;
    let oauth_jwt = mint_service_jwt(&creds, "oauth", &ca_jwt_key, &oauth_key);
    let tag = format!("mac-policy-check-{}", uuid::Uuid::new_v4());
    let client = spawn_policy_and_client(&tag, &oauth_key, Some(oauth_jwt)).await?;

    // This is the same authenticated PolicyClient check used by OAuth PAR
    // registration. Empty wire subject/domain are deliberate: PolicyService
    // derives them from the verified service envelope before Casbin evaluates
    // the federation registration resource.
    let allowed = client
        .check(&PolicyCheck {
            subject: String::new(),
            domain: String::new(),
            resource: "federation:register:https://signup.example.test".to_owned(),
            operation: "check".to_owned(),
        })
        .await?;
    assert!(allowed, "the permissive real PolicyService handler must run");
    assert!(
        global_mac_dispatch_pep().is_some(),
        "the production dispatch PEP must remain installed after policy.check"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn authenticated_hybrid_oauth_issue_token_reaches_the_real_authorization_boundary() -> Result<()> {
    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep();

    // The production PEP is activation-controlled and every process starts
    // FloorOnly: the anonymous-floor subject context can dominate the floor
    // `check` row but can never dominate the schema-declared Internal/PqHybrid
    // `issueToken` row, even with a perfect service clearance. Widen the
    // operator control identity-aware for this test (the same mechanism the
    // mac_enforcing_gate T8 contract exercises) and narrow on drop. This
    // synthetic evidence tests the control mechanism only; production staging
    // activation stays an operator gate.
    let coverage = hyprstream_rpc::auth::mac::GenesisReport {
        labeled: vec!["policy.issueToken".to_owned()],
        unlabeled: Vec::new(),
        ill_formed: Vec::new(),
    };
    let evidence = hyprstream_rpc::auth::mac::MacActivationEvidence {
        genesis: &coverage,
        mediation_integrity_g2: true,
        denial_handling_g4: true,
        observability_g5: true,
        runbook_signoff_g6: true,
        revocation_reload_g7: true,
    };
    hyprstream_rpc::auth::mac::global_mac_activation_control()
        .widen_identity_aware(&evidence)?;
    struct NarrowOnDrop;
    impl Drop for NarrowOnDrop {
        fn drop(&mut self) {
            hyprstream_rpc::auth::mac::global_mac_activation_control().narrow_to_floor();
        }
    }
    let _narrow = NarrowOnDrop;

    let root_key = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca_jwt_key = derive_purpose_key(&root_key, "hyprstream-jwt-v1");
    let oauth_key = SigningKey::from_bytes(&OAUTH_KEY);
    let credentials = tempfile::TempDir::new()?;
    let oauth_jwt = mint_service_jwt(&credentials, "oauth", &ca_jwt_key, &oauth_key);
    let tag = format!("mac-policy-issue-token-{}", uuid::Uuid::new_v4());
    let client = spawn_policy_and_client(&tag, &oauth_key, Some(oauth_jwt)).await?;

    // Match the authorization-code cold-signup issuance shape: OAuth asks the
    // PolicyService to mint for an explicit hosted account subject and tenant,
    // so the handler must apply target-tenant policy:IssueToken/manage as well
    // as the interactive session, client ID, audience, and signing guards.
    let subject = format!("cold-signup-{}", uuid::Uuid::new_v4());
    let tenant = "did:web:tilde.staging.lab.hyprstream.com";
    let issuer = "https://tilde.staging.lab.hyprstream.com";
    let sid = format!("sid-{}", uuid::Uuid::new_v4());
    register_active_session(issuer, &sid, &subject, tenant).await?;

    let minted = client
        .issue_token(&IssueToken {
            requested_scopes: Some(vec!["openid".to_owned(), "profile".to_owned()]),
            ttl: Some(300),
            audience: Some(issuer.to_owned()),
            subject: Some(subject),
            user_pub_key: None,
            dpop_jkt: None,
            issuer: Some(issuer.to_owned()),
            tenant: Some(tenant.to_owned()),
            require_clearance: false,
            session_id: Some(sid.clone()),
            issuance_profile: IssueTokenProfile::InteractiveSession,
            client_id: Some("cold-signup-browser".to_owned()),
        })
        .await
        .expect("verified hybrid service:oauth must pass the declared issueToken PEP and real PolicyService authorization boundary");
    assert!(!minted.token.is_empty(), "the real handler must mint a token");
    let claims = hyprstream_rpc::auth::decode_unverified(&minted.token)?;
    assert_eq!(claims.sid.as_deref(), Some(sid.as_str()));
    assert_eq!(claims.tenant.as_deref(), Some(tenant));
    assert_eq!(claims.client_id.as_deref(), Some("cold-signup-browser"));

    // A signed, hybrid bootstrap service with no scoped token-issuer
    // clearance cannot borrow OAuth's authority. The uniform dispatch denial
    // occurs before the handler, so it cannot mint or expose its policy state.
    let discovery_key = SigningKey::from_bytes(&DISCOVERY_KEY);
    let discovery_credentials = tempfile::TempDir::new()?;
    let discovery_jwt = mint_service_jwt(
        &discovery_credentials,
        "discovery",
        &ca_jwt_key,
        &discovery_key,
    );
    let denied_tag = format!("mac-policy-issue-token-deny-{}", uuid::Uuid::new_v4());
    let denied_client =
        spawn_policy_and_client(&denied_tag, &discovery_key, Some(discovery_jwt)).await?;
    let denied = denied_client
        .issue_token(&IssueToken {
            requested_scopes: Some(vec!["openid".to_owned()]),
            ttl: Some(300),
            audience: Some(issuer.to_owned()),
            subject: Some("other-subject".to_owned()),
            user_pub_key: None,
            dpop_jkt: None,
            issuer: Some(issuer.to_owned()),
            tenant: Some(tenant.to_owned()),
            require_clearance: false,
            session_id: Some(sid),
            issuance_profile: IssueTokenProfile::InteractiveSession,
            client_id: Some("cold-signup-browser".to_owned()),
        })
        .await;
    let error = denied.expect_err("a non-OAuth service must not reach issueToken");
    assert!(
        format!("{error:?}").contains(hyprstream_rpc::service::dispatch::DISPATCH_DENIED),
        "non-OAuth issueToken must fail at the uniform dispatch boundary: {error:?}"
    );
    Ok(())
}

/// The PolicyService's JWT key source, wired the way the service factory wires
/// `ServiceContext::cluster_key_source()` for the offline (no JWKS fetcher)
/// path: the purpose-derived CA JWT key plus its composite pair.
fn cluster_key_source(ca_jwt_key: &SigningKey) -> Arc<ClusterKeySource> {
    let ca_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&derive_mesh_mldsa_key(ca_jwt_key));
    // The OAuth fixture mints real session tokens through PolicyService's
    // shared signing boundary (`sign_token`), so the source must carry a
    // mint-capable composite ledger with an active Policy-role pair — the same
    // shape the production factory wires. Without it the issuance path fails
    // closed with SIGNING_NOT_CONFIGURED even when every MAC gate permits.
    let mut role_pairs = Vec::new();
    for (seed, role) in [
        (0x2Fu8, hyprstream_rpc::auth::CompositePairRole::OAuth),
        (0x30u8, hyprstream_rpc::auth::CompositePairRole::Policy),
    ] {
        let ed = SigningKey::from_bytes(&[seed; 32]);
        let (pq, pq_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let kid =
            hyprstream_core::auth::jwt::composite_kid(&pq_vk, &ed.verifying_key());
        role_pairs.push(hyprstream_rpc::auth::CompositeKeyPair::signing(
            kid,
            std::sync::Arc::new(pq),
            std::sync::Arc::new(ed),
            role,
            hyprstream_rpc::auth::CompositePairState::Active,
            0,
            i64::MAX,
        ));
    }
    // mint_snapshot() reads the COMMITTED on-disk ledger generation (flock +
    // committed marker + committed ledger file) and requires the in-memory
    // publication to match it exactly, so a bare publish() is not enough.
    let dir = std::env::temp_dir().join(format!(
        "hyprstream-mac-dispatch-fixture-{}",
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).expect("fixture composite authority dir");
    let ledger = dir.join("ledger.json");
    let committed = dir.join("committed");
    let committed_ledger_prefix = dir.join("committed-ledger");
    let key_set = std::sync::Arc::new(hyprstream_rpc::auth::CompositeKeySet::default());
    key_set.configure_authority(
        ledger.clone(),
        committed.clone(),
        committed_ledger_prefix.clone(),
        dir.join("lock"),
    );
    const GENERATION: u64 = 1;
    const DIGEST: &str = "mac-dispatch-boot-labeling-fixture";
    key_set
        .publish(GENERATION, DIGEST.to_owned(), role_pairs)
        .expect("fixture composite ledger publication");
    std::fs::write(
        &committed,
        format!(r#"{{"version":{GENERATION},"component_digest":"{DIGEST}"}}"#),
    )
    .expect("fixture committed marker");
    std::fs::write(
        committed_ledger_prefix.with_file_name(format!(
            "committed-ledger-{GENERATION}-{DIGEST}.json"
        )),
        format!(r#"{{"version":{GENERATION},"component_digest":"{DIGEST}"}}"#),
    )
    .expect("fixture committed ledger");
    Arc::new(
        ClusterKeySource::new(ca_jwt_key.verifying_key(), ISSUER.to_owned())
            .with_ca_composite_key(ca_pq_vk)
            .with_composite_key_set(key_set),
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
        &discovery_key,
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
    // authority. It must deny before handler entry. Per the v16 §14.2
    // uniform-denial rule the wire error is opaque ("dispatch denied") so the
    // response cannot leak which gate fired; the specific UnlabeledObject
    // reason is asserted at the PEP unit level (mac::dispatch_labels and
    // mac::cas_pep tests) and in the audit trail.
    let undeclared = client
        .resolve_service_key(&ResolveServiceKey {
            service_name: "registry".to_owned(),
        })
        .await;
    let error = undeclared.expect_err("undeclared leaf must deny");
    assert!(
        format!("{error:?}").contains(hyprstream_rpc::service::dispatch::DISPATCH_DENIED),
        "undeclared leaf must deny through the uniform dispatch denial, got: {error:?}"
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
        body: &DecodedRequestBody,
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        Ok((body.bytes().to_vec(), None))
    }

    fn decode_request_body(
        &self,
        signed_body: &[u8],
    ) -> Result<DecodedRequestBody> {
        // Byte-oriented echo: no Cap'n Proto request schema, no derivable
        // method leaf — the same affirmative `opaque` choice as the in-crate
        // mock services (v16 §5.2).
        Ok(DecodedRequestBody::opaque(signed_body.to_vec()))
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
        &discovery_key,
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
