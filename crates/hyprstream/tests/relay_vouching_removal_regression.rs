//! REGRESSION (PR1678 scoped repair): the relay-vouching exception removed at
//! this head must keep the harvested-public-WIT attack DENIED end-to-end while
//! the two honest relay shapes keep their documented outcomes.
//!
//! Ported from the isolated spike reproduction (spike_relay_repro.rs at the
//! 3c72353b6 head) with the malicious case INVERTED: the admitted relay
//! harvests service:oauth's PUBLIC WIT through the real resolveServiceKey RPC
//! and relays it as a delegated service credential over the encrypted Iroh
//! hop. After the removal there is no vouching exception, so Policy's holder
//! binding denies BEFORE any foreign claims are published, and no token is
//! minted. Controls A1/A2 prove the uniform dispatch denial and the approved
//! user-delegation shape are unchanged.
//!
//! Real hybrid signatures, recipient-sealed AEAD over Iroh, KEM + PQ anchors,
//! production PEP, real PolicyService with production base policies.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;

use anyhow::Result;
use ed25519_dalek::SigningKey;

use hyprstream_core::auth::identity_store::BootstrapPubkey;
use hyprstream_core::auth::service_jwt::issue_or_load_service_jwt;
use hyprstream_core::auth::PolicyManager;
use hyprstream_core::config::TokenConfig;
use hyprstream_core::services::PolicyService;
use hyprstream_rpc::auth::ClusterKeySource;
use hyprstream_rpc::envelope::KeyedPqTrustStore;
use hyprstream_rpc::node_identity::{derive_mesh_mldsa_key, derive_purpose_key};
use hyprstream_rpc::signer::LocalSigner;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc_std::policy_client::{
    IssueToken, IssueTokenProfile, PolicyCheck, PolicyClient, RegisterServiceKey,
    ResolveServiceKey,
};
use hyprstream_service::{InprocManager, ServiceManager as _};

const POLICY_ROOT_KEY: [u8; 32] = [0x52; 32];
const OAUTH_KEY: [u8; 32] = [0x44; 32];
const RELAY_MODEL_KEY: [u8; 32] = [0x61; 32];
const USER_KEY: [u8; 32] = [0x62; 32];
const OUTSIDER_KEY: [u8; 32] = [0x77; 32];
const ISSUER: &str = "http://127.0.0.1:6791";

fn install_crypto() {
    let _ = hyprstream_rpc::proof::admission::set_global_proof_replay_store(Box::new(
        hyprstream_rpc::proof::admission::InMemoryProofReplayStore::single_verifier_instance(
            10_000,
        ),
    ));
    if hyprstream_rpc::auth::global_credential_revocation_store().is_none() {
        let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
            hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
        ));
    }
    let mut store = KeyedPqTrustStore::new();
    // OUTSIDER_KEY is anchored ONLY as a PQ identity (so its envelopes verify
    // and the test reaches the relay-admission boundary); it receives NO
    // trust-store attestation and registers NO service, so relay admission
    // still denies — which is the boundary this suite exercises.
    for bytes in [POLICY_ROOT_KEY, OAUTH_KEY, RELAY_MODEL_KEY, USER_KEY, OUTSIDER_KEY] {
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

fn mint_service_jwt(
    dir: &tempfile::TempDir,
    service_name: &str,
    ca_jwt_key: &SigningKey,
    service_key: &SigningKey,
) -> String {
    let now = chrono::Utc::now().timestamp();
    let bootstrap =
        BootstrapPubkey::for_service_key(service_key).expect("fixture service hybrid enrollment");
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

fn cluster_key_source(ca_jwt_key: &SigningKey) -> Arc<ClusterKeySource> {
    let ca_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk(&derive_mesh_mldsa_key(ca_jwt_key));
    let mut role_pairs = Vec::new();
    for (seed, role) in [
        (0x2Fu8, hyprstream_rpc::auth::CompositePairRole::OAuth),
        (0x30u8, hyprstream_rpc::auth::CompositePairRole::Policy),
    ] {
        let ed = SigningKey::from_bytes(&[seed; 32]);
        let (pq, pq_vk) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
        let kid = hyprstream_core::auth::jwt::composite_kid(&pq_vk, &ed.verifying_key());
        role_pairs.push(hyprstream_rpc::auth::CompositeKeyPair::signing(
            kid,
            Arc::new(pq),
            Arc::new(ed),
            role,
            hyprstream_rpc::auth::CompositePairState::Active,
            0,
            i64::MAX,
        ));
    }
    let dir = std::env::temp_dir().join(format!(
        "hyprstream-spike-repro-fixture-{}",
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
    const DIGEST: &str = "spike-relay-repro-fixture";
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

async fn spawn_policy(tag: &str) -> Result<PolicyClient> {
    let root_key = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca_jwt_key = derive_purpose_key(&root_key, "hyprstream-jwt-v1");
    let rules = tempfile::TempDir::new()?;
    // Production base policies (SERVICE_BASE_POLICIES) — not a permissive fixture.
    let manager = Arc::new(PolicyManager::new(rules.path().join("policies")).await?);
    let policy_dir = tempfile::TempDir::new()?;
    let git2db = Arc::new(tokio::sync::RwLock::new(
        git2db::Git2DB::open(policy_dir.path()).await?,
    ));
    let policy_service = PolicyService::new(
        manager,
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
        SigningKey::from_bytes(&POLICY_ROOT_KEY),
        root_key.verifying_key(),
        None,
    )
}

struct NativeHop {
    network: hyprstream_rpc::transport::iroh_substrate::IrohSubstrate,
    server: hyprstream_rpc::transport::iroh_substrate::IrohSubstrate,
    address: iroh::EndpointAddr,
    kem: Arc<hyprstream_rpc::crypto::hybrid_kem::KeyedKemTrustStore>,
    pq: Arc<dyn hyprstream_rpc::envelope::PqTrustStore>,
}

async fn native_hop(tag: &str, server_key: &SigningKey, client_key: &SigningKey) -> Result<NativeHop> {
    use hyprstream_rpc::transport::iroh_rpc::IrohRpcProtocolHandler;
    use hyprstream_rpc::transport::iroh_substrate::{IrohSubstrate, NoopHandler};
    hyprstream_rpc::transport::pq_provider::install_pq_crypto_provider()?;
    let processor = hyprstream_rpc::dial::lookup_inproc(tag).expect("same live Policy processor");
    let server = IrohSubstrate::new(
        derive_purpose_key(server_key, "spike-repro-server").to_bytes(),
        NoopHandler::new("unused events"),
        IrohRpcProtocolHandler::with_stream_limit(processor, server_key.clone(), 16),
    )
    .await?;
    let network = IrohSubstrate::new(
        derive_purpose_key(client_key, "spike-repro-client").to_bytes(),
        NoopHandler::new("unused events"),
        NoopHandler::new("unused inbound RPC"),
    )
    .await?;
    let address = iroh::EndpointAddr::from_parts(
        server.endpoint_id(),
        server.endpoint().bound_sockets().into_iter().map(iroh::TransportAddr::Ip),
    );
    let mut kem = hyprstream_rpc::crypto::hybrid_kem::KeyedKemTrustStore::new();
    kem.bind(
        server_key.verifying_key().to_bytes(),
        hyprstream_rpc::node_identity::derive_mesh_kem_recipient(server_key)?.public(),
    );
    let pq = hyprstream_rpc::envelope::global_pq_store().expect("fixture PQ anchors");
    Ok(NativeHop { network, server, address, kem: Arc::new(kem), pq })
}

impl NativeHop {
    async fn policy_client(&self, key: &SigningKey, server_vk: ed25519_dalek::VerifyingKey, default_jwt: Option<String>) -> Result<PolicyClient> {
        use hyprstream_rpc::rpc_client::RpcClientImpl;
        use hyprstream_rpc::transport::iroh_substrate::ALPN_HYPRSTREAM_RPC;
        use hyprstream_rpc::transport::iroh_transport::IrohTransport;
        let connection = self.network.connect(self.address.clone(), ALPN_HYPRSTREAM_RPC).await?;
        let rpc = RpcClientImpl::new(LocalSigner::new(key.clone()), IrohTransport::new(connection), Some(server_vk))
            .with_request_kem_store(self.kem.clone())
            .with_response_pq_store(self.pq.clone());
        let rpc = match default_jwt {
            Some(token) => rpc.with_default_jwt(token),
            None => rpc,
        };
        Ok(PolicyClient::new(Arc::new(rpc)))
    }

    async fn shutdown(self) -> Result<()> {
        self.network.shutdown().await?;
        self.server.shutdown().await?;
        Ok(())
    }
}

fn mint_user_token(
    ca_jwt_key: &SigningKey,
    user_key: &SigningKey,
    sid: &str,
) -> Result<String> {
    let now = chrono::Utc::now().timestamp();
    let claims = hyprstream_rpc::auth::Claims::new("alice".to_owned(), now, now + 300)
        .with_issuer(ISSUER.to_owned())
        .with_audience(Some(ISSUER.to_owned()))
        .with_tenant("staging-test".to_owned())
        .with_client_id("spike-honest-relay")
        .with_sid(sid.to_owned())
        .with_cnf_jwk(user_key.verifying_key().as_bytes());
    let ca_pq = derive_mesh_mldsa_key(ca_jwt_key);
    Ok(hyprstream_core::auth::jwt::encode_composite_ml_dsa_65_ed25519(
        &claims, &ca_pq, ca_jwt_key,
    ))
}


#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn harvested_public_wit_relay_denies_after_vouching_removal() -> Result<()> {
    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep()?;

    let root_key = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca_jwt_key = derive_purpose_key(&root_key, "hyprstream-jwt-v1");
    let oauth_key = SigningKey::from_bytes(&OAUTH_KEY);
    let relay_key = SigningKey::from_bytes(&RELAY_MODEL_KEY);
    let user_key = SigningKey::from_bytes(&USER_KEY);

    let credentials = tempfile::TempDir::new()?;
    let oauth_wit = mint_service_jwt(&credentials, "oauth", &ca_jwt_key, &oauth_key);
    let model_wit = mint_service_jwt(&credentials, "model", &ca_jwt_key, &relay_key);

    let tag = format!("relay-removal-policy-{}", uuid::Uuid::new_v4());
    let _policy = spawn_policy(&tag).await?;

    let oauth_client = PolicyClient::for_local_endpoint_bootstrap(
        &format!("inproc://{tag}"), oauth_key.clone(), root_key.verifying_key(), Some(oauth_wit.clone()),
    )?;
    oauth_client
        .register_service_key(&RegisterServiceKey {
            service_name: "oauth".to_owned(),
            verifying_key: oauth_key.verifying_key().as_bytes().to_vec(),
            service_jwt: oauth_wit.clone(),
        })
        .await?;
    let model_client = PolicyClient::for_local_endpoint_bootstrap(
        &format!("inproc://{tag}"), relay_key.clone(), root_key.verifying_key(), Some(model_wit.clone()),
    )?;
    model_client
        .register_service_key(&RegisterServiceKey {
            service_name: "model".to_owned(),
            verifying_key: relay_key.verifying_key().as_bytes().to_vec(),
            service_jwt: model_wit.clone(),
        })
        .await?;
    hyprstream_service::global_trust_store().insert(
        relay_key.verifying_key(),
        hyprstream_service::Attestation {
            scopes: std::iter::once("model".to_owned()).collect(),
            subject: Some("service:model".to_owned()),
            jwt: None,
            expires_at: 0,
            attested_by: None,
        },
    );

    let hop = native_hop(&tag, &root_key, &relay_key).await?;
    let policy_vk = root_key.verifying_key();

    // A1. The relay presents its OWN credential: the issuer guard denies at
    // dispatch, before any handler. (Unchanged control.)
    let honest_self = hop.policy_client(&relay_key, policy_vk, Some(model_wit.clone())).await?;
    let error = honest_self
        .issue_token(&IssueToken {
            requested_scopes: Some(vec!["openid".to_owned()]),
            ttl: Some(300),
            audience: Some(ISSUER.to_owned()),
            subject: Some("honest-self".to_owned()),
            user_pub_key: None,
            dpop_jkt: None,
            issuer: Some(ISSUER.to_owned()),
            tenant: None,
            require_clearance: false,
            session_id: None,
            issuance_profile: IssueTokenProfile::Rfc8693,
            client_id: Some("relay-removal-honest-self".to_owned()),
        })
        .await
        .expect_err("the relay's own service:model identity must not issue tokens");
    assert!(
        format!("{error:?}").contains("dispatch denied"),
        "A1: own identity must hit the uniform denial: {error:?}"
    );

    // A2. The relay relays a USER token: Policy evaluates the user.
    let sid = format!("relay-removal-sid-{}", uuid::Uuid::new_v4());
    register_active_session(ISSUER, &sid, "alice", "staging-test").await?;
    let user_token = mint_user_token(&ca_jwt_key, &user_key, &sid)?;
    let honest_user = hop
        .policy_client(&relay_key, policy_vk, Some(model_wit.clone()))
        .await?
        .with_delegated_bearer(user_token);
    let allowed = honest_user
        .check(&PolicyCheck {
            subject: String::new(),
            domain: String::new(),
            resource: "policy:IssueToken".to_owned(),
            operation: "manage".to_owned(),
        })
        .await?;
    assert!(!allowed, "A2: a relayed user must not acquire oauth's grant");

    // B. MALICIOUS (inverted): harvest oauth's PUBLIC WIT through the real
    // resolution RPC and relay it as a delegated service credential. With the
    // vouching exception removed, the holder binding denies BEFORE foreign
    // claims are published and nothing is minted.
    let harvested = model_client
        .resolve_service_key(&ResolveServiceKey {
            service_name: "oauth".to_owned(),
        })
        .await?
        .keys
        .into_iter()
        .find(|entry| entry.verifying_key == oauth_key.verifying_key().as_bytes())
        .and_then(|entry| entry.service_jwt)
        .expect("real resolution returns the published oauth WIT");
    assert_eq!(harvested, oauth_wit, "the public attestation IS the live credential");

    let malicious = hop
        .policy_client(&relay_key, policy_vk, Some(model_wit.clone()))
        .await?
        .with_delegated_bearer(harvested);
    let error = malicious
        .check(&PolicyCheck {
            subject: String::new(),
            domain: String::new(),
            resource: "policy:IssueToken".to_owned(),
            operation: "manage".to_owned(),
        })
        .await
        .expect_err(
            "B: a harvested public WIT must deny once the vouching exception is removed",
        );
    // The denial is the uniform dispatch boundary: the relayed credential
    // never authenticates, so no oauth claims are ever published and the
    // caller-specific decision evaluates the relay's own (insufficient)
    // identity. This is the required fail-closed entry point.
    assert!(
        error.to_string().contains("dispatch denied")
            || error.to_string().contains("requires its bound holder signer"),
        "B: expected fail-closed denial before foreign claims entry, got: {error:?}"
    );

    // No authority conversion: the mint attempt with the same relayed
    // credential is denied by dispatch (the credential never authenticated).
    let mint = malicious
        .issue_token(&IssueToken {
            requested_scopes: Some(vec!["openid".to_owned()]),
            ttl: Some(300),
            audience: Some(ISSUER.to_owned()),
            subject: Some("harvested-victim".to_owned()),
            user_pub_key: None,
            dpop_jkt: None,
            issuer: Some(ISSUER.to_owned()),
            tenant: None,
            require_clearance: false,
            session_id: None,
            issuance_profile: IssueTokenProfile::Rfc8693,
            client_id: Some("relay-removal-attack".to_owned()),
        })
        .await;
    assert!(
        mint.is_err(),
        "B: no token may be minted for the relayed harvested credential"
    );

    hop.shutdown().await?;
    Ok(())
}

/// A relay that is NOT admitted (its key is absent from the trust store and
/// it registered no service) cannot even present a delegated bearer at
/// dispatch: the admission gate denies before any claims verification. This
/// is the dispatch-level complement to the unit-level
/// `delegated_bearer_is_denied_by_default` (Sol/operational note: verify
/// non-controller admission on the whole dispatch path, never infer it from
/// the accept_delegated_bearer callback alone).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn non_admitted_relay_delegated_bearer_denies_at_dispatch() -> Result<()> {
    install_crypto();
    hyprstream_core::mac::install_production_rpc_dispatch_pep()?;

    let root_key = SigningKey::from_bytes(&POLICY_ROOT_KEY);
    let ca_jwt_key = derive_purpose_key(&root_key, "hyprstream-jwt-v1");
    let outsider = SigningKey::from_bytes(&OUTSIDER_KEY);
    let victim = SigningKey::from_bytes(&[0x78; 32]);

    let credentials = tempfile::TempDir::new()?;
    let victim_wit = mint_service_jwt(&credentials, "oauth", &ca_jwt_key, &victim);

    let tag = format!("relay-nonadmitted-policy-{}", uuid::Uuid::new_v4());
    let _policy = spawn_policy(&tag).await?;
    let hop = native_hop(&tag, &root_key, &outsider).await?;
    let policy_vk = root_key.verifying_key();

    // The outsider registered NOTHING and holds no trust-store attestation.
    // It forwards a credential belonging to service:oauth (harvested or
    // captured — provenance is irrelevant at this boundary).
    let intruder = hop
        .policy_client(&outsider, policy_vk, Some(victim_wit.clone()))
        .await?
        .with_delegated_bearer(victim_wit);
    let error = intruder
        .check(&PolicyCheck {
            subject: String::new(),
            domain: String::new(),
            resource: "policy:IssueToken".to_owned(),
            operation: "manage".to_owned(),
        })
        .await
        .expect_err("a non-admitted relay must be denied at the admission gate");
    // The unregistered signer is denied by the whole-dispatch admission
    // enforcement BEFORE claims verification — the operator-required
    // dispatch-level check, not merely the accept_delegated_bearer callback.
    // The deeper verifier-level boundary is unit-covered by
    // delegated_bearer_is_denied_by_default.
    assert!(
        error.to_string().contains("dispatch denied"),
        "expected the whole-dispatch admission denial for the unregistered signer, got: {error:?}"
    );

    hop.shutdown().await?;
    Ok(())
}
