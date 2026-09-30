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
//! 2. Other schema-declared leaves resolve through the same full inventory;
//!    unknown paths still deny before handler entry.
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
const BROWSER_KEY: [u8; 32] = [0x45; 32];
const REGISTRY_KEY: [u8; 32] = [0x46; 32];
const REGISTRATION_KEY: [u8; 32] = [0x47; 32];
const OAUTH_RELAY_KEY: [u8; 32] = [0x48; 32];
const REPLAY_POLICY_KEY: [u8; 32] = [0x49; 32];
const REPLAY_RELAY_KEY: [u8; 32] = [0x4a; 32];
const MEDIATED_MODEL_KEY: [u8; 32] = [0x4b; 32];
const MEDIATED_DENIED_KEY: [u8; 32] = [0x4c; 32];
const ISSUER: &str = "http://127.0.0.1:6791";

/// Exercise the installed production PEP over every actual generated leaf,
/// including nested Registry paths, with a user rather than service subject.
/// Signature/JWT verification is exercised separately through real RPC below;
/// these explicit context fixtures isolate the mandatory label decision.
#[test]
fn generated_dispatch_accepts_verified_user_clearance_without_service_prefix() -> Result<()> {
    use hyprstream_rpc::auth::mac::{MacDecision, MacDenyReason};
    use hyprstream_rpc::proof::policy::{collect_generated_rows, AuthenticationRequirement};
    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep()?;
    let pep = global_mac_dispatch_pep().unwrap();
    let hybrid = SigningKey::from_bytes(&DISCOVERY_KEY).verifying_key();
    let classical = SigningKey::from_bytes(&[0xA5; 32]).verifying_key();
    let rows = collect_generated_rows()?;
    let registry: Vec<_> = rows.iter().filter(|row| row.service == "registry").collect();
    assert!(registry.iter().any(|row| row.leaf_path.len() == 3));
    assert!(registry.iter().any(|row| row.symbolic_path == "list"));
    for row in registry {
        assert_eq!(row.authentication, AuthenticationRequirement::CredentialRequired);
        let claims = hyprstream_rpc::auth::Claims::new("browser-user".to_owned(), 1, i64::MAX)
            .with_clearance(row.target_label);
        let context = |signer, claims| EnvelopeContext::for_test_authenticated_subject_with_claims(
            hyprstream_rpc::Subject::new("browser-user"), "staging-test", signer, claims,
        );
        assert_eq!(pep.check(&context(hybrid, claims.clone()), row.service, Some(row.leaf_path)), MacDecision::Permit,
            "verified user must reach declared {}", row.symbolic_path);
        assert_eq!(pep.check(&context(classical, claims), row.service, Some(row.leaf_path)), MacDecision::Deny(MacDenyReason::FloorDeny),
            "a JWT cannot upgrade classical envelope assurance");
        let absent = hyprstream_rpc::auth::Claims::new("browser-user".to_owned(), 1, i64::MAX);
        let low = absent.clone().with_clearance(hyprstream_rpc::auth::mac::SecurityLabel::bottom());
        assert_eq!(pep.check(&context(hybrid, low), row.service, Some(row.leaf_path)), MacDecision::Deny(MacDenyReason::FloorDeny));
        assert_eq!(pep.check(&context(hybrid, absent), row.service, Some(row.leaf_path)), MacDecision::Deny(MacDenyReason::NoClearance));
        if row.leaf_path.len() > 1 {
            assert_eq!(pep.check(&bearerless_context(hybrid), row.service, Some(&row.leaf_path[..row.leaf_path.len() - 1])), MacDecision::Deny(MacDenyReason::UnlabeledObject));
        }
    }
    let bearerless = EnvelopeContext::for_test_authenticated_subject(hyprstream_rpc::Subject::new("browser-user"), hybrid);
    for (service, path) in [("registry", &[u16::MAX][..]), ("registry", &[0, 0][..]), ("/registry", &[0][..]), ("policy", &[u16::MAX][..])] {
        assert_eq!(pep.check(&bearerless, service, Some(path)), MacDecision::Deny(MacDenyReason::UnlabeledObject));
    }
    assert_eq!(pep.check(&bearerless, "registry", None), MacDecision::Deny(MacDenyReason::UnlabeledObject));
    Ok(())
}

#[test]
fn generated_dispatch_restricts_all_credential_authority_leaves() -> Result<()> {
    use hyprstream_rpc::auth::mac::{MacDecision, MacDenyReason};
    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep()?;
    let pep = global_mac_dispatch_pep().unwrap();
    let signer = SigningKey::from_bytes(&DISCOVERY_KEY).verifying_key();
    let names = ["issueToken", "registerSession", "revokeSession", "revokeCredential", "exchangeDelegated", "exchangeWit"];
    let rows = hyprstream_rpc::proof::policy::collect_generated_rows()?;
    for name in names {
        let row = rows.iter().find(|row| row.service == "policy" && row.symbolic_path == name)
            .ok_or_else(|| anyhow::anyhow!("missing authority leaf {name}"))?;
        for (subject, allowed) in [("service:discovery", false), ("service:mcp", false),
            ("service:registry", false), ("service:unknown", false),
            ("service:oauth", true), ("service:policy", true)] {
            let claims = hyprstream_rpc::auth::Claims::new(subject.to_owned(), 1, i64::MAX)
                .with_clearance(row.target_label);
            let context = EnvelopeContext::for_test_authenticated_subject_with_claims(
                hyprstream_rpc::Subject::new(subject), "staging-test", signer, claims,
            );
            assert_eq!(pep.check(&context, "policy", Some(row.leaf_path)),
                if allowed { MacDecision::Permit } else { MacDecision::Deny(MacDenyReason::NoClearance) },
                "{subject} on {name}");
        }
    }
    Ok(())
}

fn bearerless_context(signer: ed25519_dalek::VerifyingKey) -> EnvelopeContext {
    EnvelopeContext::for_test_authenticated_subject(hyprstream_rpc::Subject::new("browser-user"), signer)
}

/// Install this binary's process-wide hybrid trust view: envelope signature
/// verification (Hybrid policy) with the PQ anchors of the fixture keys.
/// These anchors authenticate keys; they grant no authorization.
fn install_crypto() {
    let _ = hyprstream_rpc::proof::admission::set_global_proof_replay_store(Box::new(
        hyprstream_rpc::proof::admission::InMemoryProofReplayStore::single_verifier_instance(10_000),
    ));
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
    for bytes in [POLICY_ROOT_KEY, DISCOVERY_KEY, GHOST_CLIENT_KEY, OAUTH_KEY, BROWSER_KEY, REGISTRY_KEY, REGISTRATION_KEY, OAUTH_RELAY_KEY, REPLAY_POLICY_KEY, REPLAY_RELAY_KEY, MEDIATED_MODEL_KEY, MEDIATED_DENIED_KEY] {
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
async fn hybrid_user_registry_rpc_passes_and_missing_clearance_or_wrong_signer_deny() -> Result<()> {
    use hyprstream_core::services::RegistryService;
    use hyprstream_rpc::auth::mac::{Assurance, CompartmentSet, Level, SecurityLabel};
    use hyprstream_rpc_std::registry_client::RegistryClient;
    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep()?;
    let root_key = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca_jwt_key = derive_purpose_key(&root_key, "hyprstream-jwt-v1");
    let ca_pq = derive_mesh_mldsa_key(&ca_jwt_key);
    let user_key = SigningKey::from_bytes(&BROWSER_KEY);
    let registry_key = SigningKey::from_bytes(&REGISTRY_KEY);
    // Relay authorization comes from the enrolled service key, not the
    // browser's bearer or a caller-supplied subject string.
    hyprstream_service::global_trust_store().insert(registry_key.verifying_key(), hyprstream_service::Attestation {
        scopes: std::iter::once("registry".to_owned()).collect(),
        subject: Some("service:registry".to_owned()),
        jwt: None,
        expires_at: 0,
        attested_by: None,
    });
    let tag = format!("mac-user-registry-{}", uuid::Uuid::new_v4());
    let policy = spawn_policy_and_client(&format!("{tag}-policy"), &registry_key, None).await?;
    let data = tempfile::TempDir::new()?;
    let registry = RegistryService::new(data.path(), policy, TransportConfig::inproc(&tag), registry_key.clone())
        .await?
        .with_jwt_key_source(cluster_key_source(&ca_jwt_key));
    let _handle = InprocManager::new().spawn(Box::new(registry)).await?;
    let sid = format!("registry-browser-{}", uuid::Uuid::new_v4());
    register_active_session(ISSUER, &sid, "browser-user", "staging-test").await?;
    let now = chrono::Utc::now().timestamp();
    let base = hyprstream_rpc::auth::Claims::new("browser-user".to_owned(), now, now + 300)
        .with_issuer(ISSUER.to_owned())
        .with_audience(Some(ISSUER.to_owned()))
        .with_tenant("staging-test".to_owned())
        .with_client_id("staging-browser-test")
        .with_sid(sid)
        .with_scope(Some("query:registry:*".to_owned()))
        .with_cnf_jwk(user_key.verifying_key().as_bytes());
    let cleared = base.clone().with_clearance(SecurityLabel::new(Level::Internal, Assurance::PqHybrid, CompartmentSet::EMPTY));
    let mint = |claims: &hyprstream_rpc::auth::Claims| {
        hyprstream_core::auth::jwt::encode_composite_ml_dsa_65_ed25519(claims, &ca_pq, &ca_jwt_key)
    };
    let token = mint(&cleared);
    let client = |key, token| RegistryClient::for_local_endpoint_bootstrap(
        &format!("inproc://{tag}"), key, registry_key.verifying_key(), token,
    );
    let listed = client(user_key.clone(), Some(token.clone()))?.list().await?;
    let persisted = git2db::Git2DB::open(data.path()).await?;
    let mut expected: Vec<_> = persisted.list().filter(|repo| repo.name.as_ref().is_some_and(|name| !name.is_empty())).map(|repo| repo.id.to_string()).collect();
    let mut actual: Vec<_> = listed.iter().map(|repo| repo.id.clone()).collect();
    expected.sort();
    actual.sort();
    assert_eq!(actual, expected, "RPC returns the fixture registry, including its bootstrap repositories");
    for (key, bearer) in [
        (user_key.clone(), None),
        (user_key, Some(mint(&base))),
        (SigningKey::from_bytes(&OAUTH_KEY), Some(token)),
    ] {
        let error = client(key, bearer)?.list().await.expect_err("invalid caller must not list Registry");
        assert!(format!("{error:?}").contains(hyprstream_rpc::service::dispatch::DISPATCH_DENIED), "uniform denial: {error:?}");
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn authenticated_oauth_policy_check_reaches_the_real_handler() -> Result<()> {
    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep()?;

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
async fn hybrid_service_registry_policy_mediation_preserves_caller() -> Result<()> {
    use hyprstream_core::services::RegistryService;
    use hyprstream_rpc_std::registry_client::RegistryClient;
    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep()?;
    let root = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca = derive_purpose_key(&root, "hyprstream-jwt-v1");
    let caller = SigningKey::from_bytes(&MEDIATED_MODEL_KEY);
    let registry_key = SigningKey::from_bytes(&REGISTRY_KEY);
    let credentials = tempfile::TempDir::new()?;
    let token = mint_service_jwt(&credentials, "model", &ca, &caller);
    let registry_token = mint_service_jwt(&credentials, "registry", &ca, &registry_key);
    hyprstream_service::global_trust_store().insert(registry_key.verifying_key(), hyprstream_service::Attestation {
        scopes: std::iter::once("registry".to_owned()).collect(),
        subject: Some("service:registry".to_owned()), jwt: None, expires_at: 0, attested_by: None,
    });
    let tag = format!("mac-service-mediation-{}", uuid::Uuid::new_v4());
    let policy_tag = format!("{tag}-policy");
    let rules = tempfile::TempDir::new()?;
    let policies = Arc::new(PolicyManager::new(rules.path().join("policies")).await?);
    policies.add_policy_with_domain("service:discovery", "*", "registry:List", "query", "deny").await?;
    let policy = spawn_policy_with_manager(&policy_tag, &registry_key, Some(registry_token.clone()), policies).await?;
    let direct = PolicyClient::for_local_endpoint_bootstrap(
        &format!("inproc://{policy_tag}"), caller.clone(), root.verifying_key(), Some(token.clone()),
    )?;
    assert!(direct.check(&PolicyCheck {
        subject: "service:model".to_owned(), domain: "*".to_owned(),
        resource: "registry:*".to_owned(), operation: "query".to_owned(),
    }).await?, "the original holder must authenticate and reach Policy");

    // Exercise the real mediated handler, not merely the evidence decoder.
    // A valid proof can answer a read-only check once; it cannot change the
    // requested operation, target, bound credential, or its freshness.
    use hyprstream_rpc::envelope::{RequestEnvelope, SignedEnvelope};
    use hyprstream_rpc::ToCapnp;
    use hyprstream_rpc_std::policy_client::MediatedPolicyCheck;
    let mut body = capnp::message::Builder::new_default();
    body.init_root::<hyprstream_rpc_std::registry_capnp::registry_request::Builder>().set_list(());
    let original = RequestEnvelope::new(capnp::serialize::write_message_to_words(&body))
        .with_jwt_token(token.clone()).with_service_domain("registry")?;
    let query = |request: RequestEnvelope, operation: &str| {
        let signed = SignedEnvelope::new_signed_hybrid(request, &caller, &derive_mesh_mldsa_key(&caller));
        let mut message = capnp::message::Builder::new_default();
        signed.write_to(&mut message.init_root::<hyprstream_rpc::common_capnp::signed_envelope::Builder>());
        MediatedPolicyCheck { evidence: capnp::serialize::write_message_to_words(&message).into(),
            resource: "registry:List".into(), operation: operation.into() }
    };
    let changed_operation = policy.check_mediated(&query(original.clone(), "manage")).await
        .expect_err("a query proof cannot authorize manage");
    assert!(format!("{changed_operation:?}").contains("mediated operation"));
    let wrong_target = original.clone().with_service_domain("model")?;
    let error = policy.check_mediated(&query(wrong_target, "query")).await.expect_err("wrong mediator target");
    assert!(format!("{error:?}").contains("target mismatch"));
    let swapped = original.clone().with_jwt_token(registry_token);
    policy.check_mediated(&query(swapped, "query")).await.expect_err("credential must belong to original signer");
    let mut stale = original.clone();
    stale.iat -= hyprstream_rpc::envelope::MAX_TIMESTAMP_AGE_MS + 60_000;
    let error = policy.check_mediated(&query(stale, "query")).await.expect_err("expired evidence");
    assert!(format!("{error:?}").contains("timestamp too old"));
    let valid_query = query(original, "query");
    assert!(format!("{valid_query:?}").contains("SensitiveBytes([REDACTED])"));
    assert!(!format!("{valid_query:?}").contains(&format!("{:?}", &*valid_query.evidence)));
    assert!(policy.check_mediated(&valid_query).await?, "invalid evidence must not consume valid admission");
    let error = policy.check_mediated(&valid_query).await.expect_err("same derived query cannot be replayed");
    assert!(format!("{error:?}").contains("replay admission denied"));
    let data = tempfile::TempDir::new()?;
    let registry = RegistryService::new(data.path(), policy, TransportConfig::inproc(&tag), registry_key.clone())
        .await?.with_jwt_key_source(cluster_key_source(&ca));
    let _handle = InprocManager::new().spawn(Box::new(registry)).await?;
    let client = RegistryClient::for_local_endpoint_bootstrap(
        &format!("inproc://{tag}"), caller.clone(), registry_key.verifying_key(), Some(token.clone()),
    )?;
    client.list().await.expect("authenticated service caller must survive Registry-to-Policy mediation");
    let denied_key = SigningKey::from_bytes(&MEDIATED_DENIED_KEY);
    let denied_token = mint_service_jwt(&credentials, "discovery", &ca, &denied_key);
    let denied_client = RegistryClient::for_local_endpoint_bootstrap(
        &format!("inproc://{tag}"), denied_key.clone(), registry_key.verifying_key(), Some(denied_token.clone()),
    )?;
    let error = denied_client.list().await.expect_err("the mediator's broad grants must not replace caller policy");
    assert!(format!("{error:?}").contains("Unauthorized: service:discovery cannot query on registry:List"),
        "expected caller-specific authorization denial, not a transport failure: {error:?}");

    // Same live Registry, policies and caller identities over encrypted Iroh.
    // Only the carrier changes: native clients must emit AQW rather than
    // forwarding an unverifiable decrypted ciphertext envelope.
    use hyprstream_rpc::transport::iroh_rpc::IrohRpcProtocolHandler;
    use hyprstream_rpc::transport::iroh_substrate::{IrohSubstrate, NoopHandler, ALPN_HYPRSTREAM_RPC};
    use hyprstream_rpc::transport::iroh_transport::IrohTransport;
    use hyprstream_rpc::rpc_client::RpcClientImpl;
    hyprstream_rpc::transport::pq_provider::install_pq_crypto_provider()?;
    let processor = hyprstream_rpc::dial::lookup_inproc(&tag).expect("same live Registry processor");
    let server = IrohSubstrate::new(
        derive_purpose_key(&registry_key, "mediation-test-server").to_bytes(),
        NoopHandler::new("unused events"),
        IrohRpcProtocolHandler::with_stream_limit(processor, registry_key.clone(), 16),
    ).await?;
    let network = IrohSubstrate::new(
        derive_purpose_key(&caller, "mediation-test-client").to_bytes(),
        NoopHandler::new("unused events"), NoopHandler::new("unused inbound RPC"),
    ).await?;
    let address = iroh::EndpointAddr::from_parts(server.endpoint_id(),
        server.endpoint().bound_sockets().into_iter().map(iroh::TransportAddr::Ip));
    let mut kem = hyprstream_rpc::crypto::hybrid_kem::KeyedKemTrustStore::new();
    kem.bind(registry_key.verifying_key().to_bytes(),
        hyprstream_rpc::node_identity::derive_mesh_kem_recipient(&registry_key)?.public());
    let kem = Arc::new(kem);
    let pq = hyprstream_rpc::envelope::global_pq_store().expect("fixture PQ anchors");
    for (key, credential, allowed) in [(caller, token, true), (denied_key, denied_token, false)] {
        let connection = network.connect(address.clone(), ALPN_HYPRSTREAM_RPC).await?;
        let rpc = RpcClientImpl::new(LocalSigner::new(key), IrohTransport::new(connection), Some(registry_key.verifying_key()))
            .with_default_jwt(credential).with_request_kem_store(kem.clone()).with_response_pq_store(pq.clone());
        let client = RegistryClient::new(Arc::new(rpc));
        let result = client.list().await;
        if allowed {
            result.expect("native hybrid holder witness must preserve model's grant");
        } else {
            let error = result.expect_err("native mediation must preserve discovery's denial");
            assert!(format!("{error:?}").contains("Unauthorized: service:discovery cannot query on registry:List"),
                "native denial must be caller policy, not transport: {error:?}");
        }
    }
    network.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}

#[test]
fn mediated_original_envelope_requires_hybrid_target_and_untampered_evidence() -> Result<()> {
    use hyprstream_rpc::envelope::{RequestEnvelope, SignedEnvelope, verify_mediated_envelope_evidence, MAX_MEDIATED_EVIDENCE_BYTES};
    use hyprstream_rpc::ToCapnp;
    install_crypto();
    let holder = SigningKey::from_bytes(&REPLAY_POLICY_KEY);
    // This layer verifies the signed transcript, not the credential. Policy
    // must separately reject this deliberately non-JWT placeholder.
    let request = RequestEnvelope::anonymous(vec![1, 2, 3])
        .with_jwt_token("credential-placeholder".to_owned())
        .with_service_domain("registry")?;
    let classical = SignedEnvelope::new_signed(request.clone(), &holder);
    let signed = SignedEnvelope::new_signed_hybrid(request, &holder, &derive_mesh_mldsa_key(&holder));
    let encode = |value: &SignedEnvelope| {
        let mut message = capnp::message::Builder::new_default();
        value.write_to(&mut message.init_root::<hyprstream_rpc::common_capnp::signed_envelope::Builder<'_>>());
        capnp::serialize::write_message_to_words(&message)
    };
    let evidence = encode(&signed);
    assert!(verify_mediated_envelope_evidence(&encode(&classical), "registry").is_err());
    assert_eq!(verify_mediated_envelope_evidence(&evidence, "registry")?.cnf, holder.verifying_key().to_bytes());
    assert!(verify_mediated_envelope_evidence(&evidence, "model").is_err());
    let context = EnvelopeContext::from_verified_as_system(&signed);
    assert_eq!(context.original_holder_evidence(), Some(evidence.as_slice()));
    let mut tampered = signed.clone();
    tampered.envelope.payload.push(4);
    assert!(verify_mediated_envelope_evidence(&encode(&tampered), "registry").is_err());
    let nested = SignedEnvelope::new_signed_hybrid(
        signed.envelope.clone().with_delegation_token("nested".to_owned()), &holder, &derive_mesh_mldsa_key(&holder),
    );
    assert!(verify_mediated_envelope_evidence(&encode(&nested), "registry").is_err());
    let mut trailing = evidence.clone();
    trailing.extend_from_slice(&[0; 8]);
    assert!(verify_mediated_envelope_evidence(&trailing, "registry").is_err());
    assert!(verify_mediated_envelope_evidence(&vec![0; MAX_MEDIATED_EVIDENCE_BYTES + 1], "registry").is_err());
    assert!(verify_mediated_envelope_evidence(&[], "registry").is_err());
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn authenticated_hybrid_oauth_issue_token_reaches_the_real_authorization_boundary() -> Result<()> {
    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep()?;
    // Use the production IdentityAware default. Do not mutate process-global
    // activation under parallel tests or manufacture coverage evidence.

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

    // A signed, hybrid service is not an issuance authority, even with the
    // permissive/legacy policy used here. No fixture-specific deny rule.
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
    let error = denied.expect_err("ordinary service clearance must not grant token issuance");
    assert!(
        format!("{error:?}").contains("dispatch denied"),
        "token issuance must fail at the principal dispatch boundary: {error:?}"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resolved_hybrid_policy_credential_cannot_be_replayed_for_issuance() -> Result<()> {
    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep()?;
    let root = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca = derive_purpose_key(&root, "hyprstream-jwt-v1");
    let holder = SigningKey::from_bytes(&REPLAY_POLICY_KEY);
    let relay = SigningKey::from_bytes(&REPLAY_RELAY_KEY);
    let credentials = tempfile::TempDir::new()?;
    let holder_token = mint_service_jwt(&credentials, "policy", &ca, &holder);
    let relay_credentials = tempfile::TempDir::new()?;
    let relay_token = mint_service_jwt(&relay_credentials, "registry", &ca, &relay);
    let tag = format!("hybrid-public-wit-replay-{}", uuid::Uuid::new_v4());
    let holder_client = spawn_policy_and_client(&tag, &holder, Some(holder_token.clone())).await?;
    holder_client.register_service_key(&RegisterServiceKey {
        service_name: "policy".to_owned(),
        verifying_key: holder.verifying_key().to_bytes().to_vec(),
        service_jwt: holder_token.clone(),
    }).await?;
    let relay_client = PolicyClient::for_local_endpoint_bootstrap(
        &format!("inproc://{tag}"), relay.clone(), root.verifying_key(), Some(relay_token.clone()),
    )?;
    relay_client.register_service_key(&RegisterServiceKey {
        service_name: "registry".to_owned(),
        verifying_key: relay.verifying_key().to_bytes().to_vec(),
        service_jwt: relay_token,
    }).await?;
    let resolved = relay_client.resolve_service_key(&ResolveServiceKey {
        service_name: "policy".to_owned(),
    }).await?;
    let disclosed = resolved.keys.into_iter()
        .find(|entry| entry.verifying_key == holder.verifying_key().as_bytes())
        .and_then(|entry| entry.service_jwt).expect("real resolution returns the published policy WIT");
    assert_eq!(disclosed, holder_token);
    let subject = format!("replay-target-{}", uuid::Uuid::new_v4());
    let sid = format!("replay-session-{}", uuid::Uuid::new_v4());
    register_active_session(ISSUER, &sid, &subject, "staging-test").await?;
    let request = IssueToken {
        requested_scopes: Some(vec!["openid".to_owned()]), ttl: Some(300),
        audience: Some(ISSUER.to_owned()), subject: Some(subject), user_pub_key: None,
        dpop_jkt: None, issuer: Some(ISSUER.to_owned()), tenant: Some("staging-test".to_owned()),
        require_clearance: false, session_id: Some(sid),
        issuance_profile: IssueTokenProfile::InteractiveSession, client_id: Some("replay-test".to_owned()),
    };
    assert!(!holder_client.issue_token(&request).await?.token.is_empty(),
        "the real holder must be able to issue this exact request");
    let error = relay_client.with_delegated_bearer(disclosed).issue_token(&request).await
        .expect_err("an admitted relay must not convert a public policy WIT into issuance authority");
    assert!(format!("{error:?}").contains("dispatch denied"), "wrong denial boundary: {error:?}");
    Ok(())
}

/// Real OAuth authorization hook + real Policy RPC with production base rules.
/// A local service cannot borrow the OAuth deputy's account-management grant.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn oauth_management_preserves_caller_authority_over_local_transport() -> Result<()> {
    let _ = tracing_subscriber::fmt().with_env_filter(tracing_subscriber::EnvFilter::from_default_env()).try_init();
    use hyprstream_core::services::generated::oauth_client::OauthHandler;
    use hyprstream_core::services::oauth::{rpc_handler::OAuthRpcHandler, state::OAuthState};
    use hyprstream_rpc::envelope::{RequestEnvelope, SignedEnvelope};
    use hyprstream_rpc_std::discovery_client::DiscoveryClient;

    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep()?;
    let root = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca = derive_purpose_key(&root, "hyprstream-jwt-v1");
    let relay = SigningKey::from_bytes(&OAUTH_RELAY_KEY);
    hyprstream_service::global_trust_store().insert(relay.verifying_key(), hyprstream_service::Attestation {
        scopes: std::iter::once("oauth".to_owned()).collect(),
        subject: Some("service:oauth".to_owned()), jwt: None, expires_at: 0, attested_by: None,
    });
    let dir = tempfile::TempDir::new()?;
    let tag = format!("oauth-management-policy-{}", uuid::Uuid::new_v4());
    let manager = Arc::new(PolicyManager::new(dir.path().join("policies")).await?);
    let verifier = PolicyService::new(
        manager, Arc::new(root.clone()), TokenConfig::default(),
        Arc::new(tokio::sync::RwLock::new(git2db::Git2DB::open(dir.path().join("registry")).await?)),
        TransportConfig::inproc(&tag),
    ).with_jwt_key_source(cluster_key_source(&ca));
    let mut contexts = Vec::new();
    for (name, bytes, allowed) in [("discovery", DISCOVERY_KEY, false), ("oauth", OAUTH_RELAY_KEY, true)] {
        let key = SigningKey::from_bytes(&bytes);
        let credentials = tempfile::TempDir::new()?;
        let token = mint_service_jwt(&credentials, name, &ca, &key);
        let envelope = SignedEnvelope::new_signed(
            RequestEnvelope::anonymous(Vec::new()).with_jwt_token(token), &key,
        );
        // Explicit local provenance fixture; JWT/cnf/issuer/audience/clearance
        // are checked by the production verifier, not injected as test claims.
        let mut ctx = EnvelopeContext::from_verified_as_system(&envelope);
        verifier.verify_claims(&mut ctx).await?;
        assert_eq!(ctx.claims().map(|claims| claims.sub.as_str()), Some(format!("service:{name}").as_str()));
        // RPC deliberately redacts verifier errors to the uniform denial.
        // The verifier unit regression asserts the precise holder mismatch.
        contexts.push((ctx, allowed, Some("dispatch denied")));
    }
    // A user bearer is not a public service attestation: the admitted OAuth
    // relay must still carry it to a real Policy decision as the user, never
    // borrowing its own account-management grant.
    let user_key = SigningKey::from_bytes(&BROWSER_KEY);
    let sid = format!("oauth-management-user-{}", uuid::Uuid::new_v4());
    register_active_session(ISSUER, &sid, "unprivileged-user", "staging-test").await?;
    let now = chrono::Utc::now().timestamp();
    let claims = hyprstream_rpc::auth::Claims::new("unprivileged-user".to_owned(), now, now + 300)
        .with_issuer(ISSUER.to_owned())
        .with_audience(Some(ISSUER.to_owned()))
        .with_tenant("staging-test".to_owned())
        .with_client_id("oauth-management-test")
        .with_sid(sid)
        .with_cnf_jwk(user_key.verifying_key().as_bytes());
    let token = hyprstream_core::auth::jwt::encode_composite_ml_dsa_65_ed25519(
        &claims, &derive_mesh_mldsa_key(&ca), &ca,
    );
    let envelope = SignedEnvelope::new_signed(
        RequestEnvelope::anonymous(Vec::new()).with_jwt_token(token), &user_key,
    );
    let mut user_ctx = EnvelopeContext::from_verified_as_system(&envelope);
    verifier.verify_claims(&mut user_ctx).await?;
    contexts.push((user_ctx, false, Some("Unauthorized OAuth user management operation")));
    let _handle = InprocManager::new().spawn(Box::new(verifier)).await?;
    let endpoint = format!("inproc://{tag}");
    let policy = PolicyClient::for_local_endpoint_bootstrap(&endpoint, relay.clone(), root.verifying_key(), None)?;
    // Unused in this authorization-only fixture; no Discovery request occurs.
    let discovery = DiscoveryClient::for_local_endpoint_bootstrap(&endpoint, relay.clone(), root.verifying_key(), None)?;
    let state = OAuthState::new(&hyprstream_core::config::OAuthConfig::default(), policy, discovery, relay.verifying_key().to_bytes());
    let handler = OAuthRpcHandler::new(Arc::new(state), TransportConfig::inproc("unused-oauth-handler"), relay);
    for (ctx, allowed, expected_denial) in contexts {
        for resource in ["oauth:AddPubkey", "oauth:RemoveUser"] {
            let result = handler.authorize(&ctx, resource, "manage").await;
            assert_eq!(result.is_ok(), allowed, "caller {} on {resource}: {result:?}", ctx.subject());
            if !allowed {
                let error = format!("{:?}", result.unwrap_err());
                assert!(error.contains(expected_denial.unwrap()), "wrong denial boundary: {error}");
            }
        }
    }
    assert!(handler.authorize(&EnvelopeContext::from_callback_service(1, "oauth"), "oauth:AddPubkey", "manage").await.is_err(),
        "locality without an upstream credential must not borrow deputy authority");
    Ok(())
}

/// A real generated OAuth request must not turn an enrolled ordinary service
/// into the OAuth account-management deputy.  The earlier authorization test
/// above drives the handler with an already-verified context; this regression
/// deliberately traverses the production decode -> dispatch PEP -> JWT ->
/// handler sequence used by the in-process carrier, then proves both targeted
/// mutations stop at OAuth's authenticated-local-control-plane gate before a
/// mutator is selected.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ordinary_service_cannot_mutate_oauth_through_generated_dispatch() -> Result<()> {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    use hyprstream_core::services::oauth::{rpc_handler::OAuthRpcHandler, state::OAuthState};
    use hyprstream_rpc_std::discovery_client::DiscoveryClient;
    use hyprstream_rpc_std::oauth_client::{AddPubkey, OauthClient};

    // OAuthRpcHandler deliberately contains only the generated handler and
    // authorization logic.  The production factory supplies its cluster JWT
    // verifier.  Keep that wiring explicit in this integration fixture so the
    // generated client reaches the real verifier rather than a test context.
    struct VerifiedOAuthDispatch {
        inner: OAuthRpcHandler,
        transport: TransportConfig,
        signing_key: SigningKey,
        jwt_key_source: Arc<ClusterKeySource>,
        handler_entries: Arc<AtomicUsize>,
    }

    #[async_trait(?Send)]
    impl RequestService for VerifiedOAuthDispatch {
        fn decode_request_body(&self, signed_body: &[u8]) -> Result<DecodedRequestBody> {
            self.inner.decode_request_body(signed_body)
        }

        async fn handle_request(
            &self,
            ctx: &EnvelopeContext,
            body: &DecodedRequestBody,
        ) -> Result<(Vec<u8>, Option<Continuation>)> {
            self.handler_entries.fetch_add(1, Ordering::SeqCst);
            self.inner.handle_request(ctx, body).await
        }

        fn name(&self) -> &str {
            "oauth"
        }

        fn transport(&self) -> &TransportConfig {
            &self.transport
        }

        fn signing_key(&self) -> SigningKey {
            self.signing_key.clone()
        }

        fn jwt_key_source(&self) -> Option<Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
            Some(self.jwt_key_source.clone())
        }

        fn build_error_payload(&self, request_id: u64, error: &str) -> Vec<u8> {
            self.inner.build_error_payload(request_id, error)
        }
    }

    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep()?;

    let root = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca = derive_purpose_key(&root, "hyprstream-jwt-v1");
    let relay = SigningKey::from_bytes(&OAUTH_RELAY_KEY);
    let caller = SigningKey::from_bytes(&DISCOVERY_KEY);
    let credentials = tempfile::TempDir::new()?;
    let caller_jwt = mint_service_jwt(&credentials, "discovery", &ca, &caller);
    hyprstream_service::global_trust_store().insert(relay.verifying_key(), hyprstream_service::Attestation {
        scopes: std::iter::once("oauth".to_owned()).collect(),
        subject: Some("service:oauth".to_owned()),
        jwt: None,
        expires_at: 0,
        attested_by: None,
    });

    let tag = format!("oauth-generated-mutation-{}", uuid::Uuid::new_v4());
    let policy_tag = format!("{tag}-policy");
    let policy_dir = tempfile::TempDir::new()?;
    let policies = Arc::new(PolicyManager::new(policy_dir.path().join("policies")).await?);
    for resource in ["oauth:AddPubkey", "oauth:RemoveUser"] {
        policies
            .add_policy_with_domain("service:discovery", "*", resource, "manage", "deny")
            .await?;
    }
    let policy = spawn_policy_with_manager(&policy_tag, &relay, None, policies).await?;
    // No Discovery request occurs in these mutation calls; the real OAuth
    // state still receives its normal generated client shape.
    let discovery = DiscoveryClient::for_local_endpoint_bootstrap(
        &format!("inproc://{policy_tag}"),
        relay.clone(),
        root.verifying_key(),
        None,
    )?;

    let state = OAuthState::new(
        &hyprstream_core::config::OAuthConfig::default(),
        policy,
        discovery,
        relay.verifying_key().to_bytes(),
    );
    let handler_entries = Arc::new(AtomicUsize::new(0));
    let fixture = VerifiedOAuthDispatch {
        inner: OAuthRpcHandler::new(
            Arc::new(state),
            TransportConfig::inproc(&tag),
            relay.clone(),
        ),
        transport: TransportConfig::inproc(&tag),
        signing_key: relay.clone(),
        jwt_key_source: cluster_key_source(&ca),
        handler_entries: handler_entries.clone(),
    };
    let _handle = InprocManager::new().spawn(Box::new(fixture)).await?;
    let client = OauthClient::for_local_endpoint_bootstrap(
        &format!("inproc://{tag}"),
        caller,
        relay.verifying_key(),
        Some(caller_jwt),
    )?;

    let attempted_key = SigningKey::from_bytes(&GHOST_CLIENT_KEY);
    let add_error = client
        .add_pubkey(&AddPubkey {
            username: "mutation-victim".to_owned(),
            pubkey_base64: URL_SAFE_NO_PAD.encode(attempted_key.verifying_key().as_bytes()),
            label: Some("must-not-write".to_owned()),
        })
        .await
        .expect_err("ordinary enrolled service must not add an OAuth user key");
    let add_error = format!("{add_error:?}");
    assert!(add_error.contains("authenticated local service boundary"),
        "addPubkey must stop at the OAuth locality gate before its mutator: {add_error}");
    let remove_error = client
        .remove_user("mutation-victim")
        .await
        .expect_err("ordinary enrolled service must not remove an OAuth user");
    let remove_error = format!("{remove_error:?}");
    assert!(remove_error.contains("authenticated local service boundary"),
        "removeUser must stop at the OAuth locality gate before its mutator: {remove_error}");

    assert_eq!(handler_entries.load(Ordering::SeqCst), 2,
        "both generated mutation requests must reach real dispatch and stop in OAuth authorization");
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
    spawn_policy_with_manager(tag, caller_key, service_jwt, Arc::new(PolicyManager::permissive().await?)).await
}

async fn spawn_policy_with_manager(
    tag: &str,
    caller_key: &SigningKey,
    service_jwt: Option<String>,
    policy_manager: Arc<PolicyManager>,
) -> Result<PolicyClient> {
    let root_key = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca_jwt_key = derive_purpose_key(&root_key, "hyprstream-jwt-v1");

    let policy_dir = tempfile::TempDir::new()?;
    let git2db = Arc::new(tokio::sync::RwLock::new(
        git2db::Git2DB::open(policy_dir.path()).await?,
    ));
    let policy_service = PolicyService::new(
        policy_manager,
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
    hyprstream_core::mac::install_production_rpc_dispatch_pep()?;
    assert!(
        global_mac_dispatch_pep().is_some(),
        "the production dispatch PEP must be installed"
    );

    let root_key = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca_jwt_key = derive_purpose_key(&root_key, "hyprstream-jwt-v1");
    // Other parallel tests mint different credentials for their discovery
    // caller. Keep this fresh registration key out of their identity cache.
    let discovery_key = SigningKey::from_bytes(&REGISTRATION_KEY);
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

    // verify_claims cached this same JWT before the handler ran. Registration
    // must complete that entry's service scope even at identical expiry.
    let published = client.resolve_service_key(&ResolveServiceKey {
        service_name: "discovery".to_owned(),
    }).await?;
    assert!(published.keys.iter().any(|candidate| candidate.verifying_key == discovery_key.verifying_key().as_bytes()));

    // The schema, not a partial second table, now declares this leaf. The
    // same authenticated caller reaches the real resolution handler. Asking
    // for an unregistered name must produce the handler's missing-key result,
    // not the old incomplete-table dispatch denial.
    let resolved = client
        .resolve_service_key(&ResolveServiceKey {
            service_name: "not-enrolled-in-fixture".to_owned(),
        })
        .await;
    let error = resolved.expect_err("unregistered name has no resolution");
    assert!(format!("{error:?}").contains("service key 'not-enrolled-in-fixture' not registered"), "must reach resolution rather than dispatch denial: {error:?}");

    assert!(
        global_mac_dispatch_pep().is_some(),
        "the dispatch PEP must remain installed after key resolution"
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
    hyprstream_core::mac::install_production_rpc_dispatch_pep()?;

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
