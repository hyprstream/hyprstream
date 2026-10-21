#![allow(clippy::unwrap_used, clippy::expect_used)]
use super::*;
use ed25519_dalek::{SigningKey, VerifyingKey};
use hyprstream_rpc::Subject;

const STAGING_MODEL_REF: &str = "qwen2.5-0.5b-instruct:main";
const STAGING_SCOPE: &str = "infer:model:qwen2.5-0.5b-instruct:main";
const REGISTRY_LIST_SCOPE: &str = "query:registry:List";
const ATPROTO_DID: &str = "did:plc:abcdefghijklmnopqrstuvwx";

#[test]
fn candidate_tenants_are_scoped_to_verified_did_memberships() {
    let policies = vec![
        vec![
            ATPROTO_DID.into(),
            "tenant-a".into(),
            "registry:*".into(),
            "query".into(),
            "allow".into(),
        ],
        vec![
            "other-user".into(),
            "unrelated-tenant".into(),
            "model:*".into(),
            "infer".into(),
            "allow".into(),
        ],
        vec![
            ATPROTO_DID.into(),
            "*".into(),
            "registry:*".into(),
            "query".into(),
            "allow".into(),
        ],
    ];
    let groups = vec![
        vec![
            ATPROTO_DID.into(),
            "registry-reader".into(),
            "tenant-b".into(),
        ],
        vec![
            "another-user".into(),
            "model-user".into(),
            "unrelated-group-tenant".into(),
        ],
        vec![ATPROTO_DID.into(), "global-role".into(), "*".into()],
    ];

    let tenants = candidate_tenants(ATPROTO_DID, &policies, &groups).unwrap();
    assert_eq!(
        tenants,
        BTreeSet::from(["tenant-a".into(), "tenant-b".into()])
    );
}

#[test]
fn candidate_tenant_memberships_have_a_hard_ceiling() {
    let policies = (0..=MAX_FEDERATE_TENANT_MEMBERSHIPS)
        .map(|index| {
            vec![
                ATPROTO_DID.into(),
                format!("tenant-{index:03}"),
                "registry:*".into(),
                "query".into(),
                "allow".into(),
            ]
        })
        .collect::<Vec<_>>();

    assert!(candidate_tenants(ATPROTO_DID, &policies, &[]).is_err());
}

fn source() -> Source {
    let now = chrono::Utc::now().timestamp();
    Source {
        issuer: "https://issuer.test".into(),
        subject: "source-user".into(),
        atproto_did: ATPROTO_DID.into(),
        jti: "jti".into(),
        nonce: "nonce".into(),
        token_hash: [1; 32],
        issued_at: now,
        expires_at: now + 120,
    }
}

struct TestSharedReplayStore(hyprstream_rpc::proof::admission::InMemoryProofReplayStore);

impl hyprstream_rpc::proof::admission::ProofReplayStore for TestSharedReplayStore {
    fn domain_guarantee(&self) -> hyprstream_rpc::proof::admission::ReplayDomainGuarantee {
        hyprstream_rpc::proof::admission::ReplayDomainGuarantee::LinearizableSharedStore
    }

    fn check_and_insert(
        &self,
        partition: hyprstream_rpc::proof::ProofDisposition,
        key: &hyprstream_rpc::proof::admission::ProofReplayKey,
        expires_at: u64,
    ) -> hyprstream_rpc::proof::admission::ProofAdmissionResult {
        self.0.check_and_insert(partition, key, expires_at)
    }

    fn check_and_insert_mediated(
        &self,
        key: &hyprstream_rpc::proof::admission::MediatedQueryReplayKey,
        expires_at: u64,
    ) -> hyprstream_rpc::proof::admission::ProofAdmissionResult {
        self.0.check_and_insert_mediated(key, expires_at)
    }

    fn check_and_insert_credential(
        &self,
        key: &hyprstream_rpc::proof::admission::CredentialReplayKey,
        expires_at: u64,
    ) -> hyprstream_rpc::proof::admission::ProofAdmissionResult {
        self.0.check_and_insert_credential(key, expires_at)
    }
}

async fn fixture() -> (AdmissionService, EnvelopeContext) {
    let signer = SigningKey::from_bytes(&[24; 32]).verifying_key();
    use crate::auth::service_enrollment::ServiceEnrollment;
    use hyprstream_rpc::auth::mac::{Assurance, CompartmentSet, Level, SecurityLabel};
    let service_entry = |key: VerifyingKey| ServiceEnrollment {
        ed25519_pubkey: URL_SAFE_NO_PAD.encode(key.to_bytes()),
        ml_dsa_pubkey: None,
        clearance: SecurityLabel::new(Level::Internal, Assurance::Classical, CompartmentSet::EMPTY),
        allowed_audiences: None,
        workload_session: false,
    };
    let enrollment = Arc::new(ServiceEnrollmentManifest {
        version: 1,
        services: std::collections::BTreeMap::from([
            ("oauth".into(), service_entry(signer)),
            (
                "model".into(),
                service_entry(SigningKey::from_bytes(&[25; 32]).verifying_key()),
            ),
            (
                "registry".into(),
                service_entry(SigningKey::from_bytes(&[26; 32]).verifying_key()),
            ),
        ]),
    });
    let collision_inventory = CollisionInventory::from_sources(
        &["model".into(), "oauth".into(), "registry".into()],
        vec![
            hyprstream_session_store::primary::InventorySource {
                id: "model".into(),
                keys: vec![
                    SigningKey::from_bytes(&[25; 32])
                        .verifying_key()
                        .to_bytes()
                        .to_vec(),
                    vec![27; 1952],
                ],
            },
            hyprstream_session_store::primary::InventorySource {
                id: "oauth".into(),
                keys: vec![signer.to_bytes().to_vec()],
            },
            hyprstream_session_store::primary::InventorySource {
                id: "registry".into(),
                keys: vec![SigningKey::from_bytes(&[26; 32])
                    .verifying_key()
                    .to_bytes()
                    .to_vec()],
            },
        ],
    )
    .unwrap();
    let inventory_id = collision_inventory.id();
    let scopes = BTreeSet::from([STAGING_SCOPE.into()]);
    let policy = Arc::new(PolicyManager::new_in_memory().await.unwrap());
    policy
        .add_policy_with_domain(
            ATPROTO_DID,
            "tenant",
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
            "allow",
        )
        .await
        .unwrap();
    let a = Authorities {
        policy,
        enrollment,
        profile: Profile {
            issuer: "https://issuer.test".into(),
            host: "https://host.test".into(),
            client: "client".into(),
            resource: "https://host.test".into(),
            client_scopes: scopes.clone(),
            resource_scopes: scopes,
        },
        tui_profile: None,
        serving_generation: inventory_id,
        local_collision_inventory_id: inventory_id,
        collision_inventory,
    };
    (
        AdmissionService {
            authority: Some(a),
            capacity: Some(Semaphore::new(1)),
            source_replay: Some(Arc::new(
                hyprstream_rpc::proof::admission::InMemoryProofReplayStore::single_verifier_instance(128),
            )),
            ..Default::default()
        },
        EnvelopeContext::for_test_authenticated_subject_with_claims(
            Subject::new("service:oauth"),
            "*",
            signer,
            hyprstream_rpc::auth::Claims::new("service:oauth".into(), 0, i64::MAX),
        ),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn oauth_runtime_client_authenticates_real_prepare_and_commit() -> Result<()> {
    // Crypto configuration and the trust store are process-lifetime globals.
    // Keep this full wire fixture independent of parallel unit tests.
    const CHILD: &str = "HYPRSTREAM_OAUTH_POLICY_CLIENT_FIXTURE_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "services::policy::federate_admission::tests::oauth_runtime_client_authenticates_real_prepare_and_commit", "--nocapture"])
            .env(CHILD, "1").status()?;
        anyhow::ensure!(
            status.success(),
            "isolated OAuth Policy client regression failed"
        );
        return Ok(());
    }
    hyprstream_rpc::proof::admission::set_global_proof_replay_store(Box::new(
        TestSharedReplayStore(
            hyprstream_rpc::proof::admission::InMemoryProofReplayStore::single_verifier_instance(
                128,
            ),
        ),
    ))
    .map_err(|_| anyhow::anyhow!("isolated replay fixture already installed"))?;
    use hyprstream_rpc::auth::{jwt, Claims, ClusterKeySource};
    use hyprstream_rpc::transport::TransportConfig;
    use hyprstream_rpc_std::policy_client::{
        CommitFederateSession, FederateVerifiedSource, PolicyClient, PrepareFederateSession,
    };
    use hyprstream_service::{InprocManager, ServiceManager};

    // Exercise the real client, signed envelope verification and Policy
    // handlers. The dispatch PEP is an explicit test fixture; this test's
    // boundary is the OAuth service credential and holder binding.
    crate::mac::install_explicit_test_dispatch_pep();
    let _ = hyprstream_rpc::envelope::install_verify_config(
        hyprstream_rpc::envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
            pq_store: None,
        },
    );
    let _ = hyprstream_rpc::envelope::install_response_verify_config(
        hyprstream_rpc::envelope::ResponseVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
            pq_store: None,
        },
    );
    if hyprstream_rpc::auth::global_credential_revocation_store().is_none() {
        let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
            hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
        ));
    }
    let (admission, _) = fixture().await;
    let policy = admission.authority.as_ref().unwrap().policy.clone();
    for rule in crate::auth::policy_templates::SERVICE_BASE_POLICIES {
        policy
            .add_policy_with_domain(
                rule.subject,
                rule.domain,
                rule.resource,
                rule.action,
                rule.effect,
            )
            .await?;
    }
    let authority_key = SigningKey::from_bytes(&[94; 32]);
    let oauth_key = SigningKey::from_bytes(&[24; 32]);
    let issuer = "https://federate-policy-client.test";
    let transport = TransportConfig::inproc(format!("oauth-policy-{}", uuid::Uuid::new_v4()));
    let root = tempfile::tempdir()?;
    let git = Arc::new(tokio::sync::RwLock::new(
        git2db::Git2DB::open(root.path()).await?,
    ));
    let mut service = super::super::PolicyService::new(
        policy,
        Arc::new(authority_key.clone()),
        crate::config::TokenConfig::default(),
        git,
        transport.clone(),
    )
    .with_jwt_key_source(Arc::new(ClusterKeySource::new(
        authority_key.verifying_key(),
        issuer.to_owned(),
    )));
    service.federate_admission = Some(Arc::new(admission));
    let manager = InprocManager::new();
    manager.spawn(Box::new(service)).await?;

    let now = chrono::Utc::now().timestamp();
    let claims = Claims::new("service:oauth".into(), now, now + 120)
        .with_issuer(issuer.to_owned())
        .with_cnf_jwk(oauth_key.verifying_key().as_bytes());
    let token = jwt::encode_service_jwt(&claims, &authority_key);
    hyprstream_service::global_trust_store().insert(
        oauth_key.verifying_key(),
        hyprstream_service::Attestation {
            scopes: ["oauth".to_owned()].into_iter().collect(),
            subject: None,
            jwt: Some(token),
            expires_at: now + 120,
            attested_by: Some(authority_key.verifying_key().to_bytes()),
        },
    );
    let client = crate::services::oauth::runtime_policy_client(
        &transport,
        oauth_key.clone(),
        authority_key.verifying_key(),
    )?;
    let source = source();
    let verified = FederateVerifiedSource {
        issuer: source.issuer,
        subject: source.subject,
        atproto_did: source.atproto_did,
        jti: source.jti,
        nonce: source.nonce,
        token_hash: source.token_hash.to_vec(),
        issued_at: source.issued_at,
        expires_at: source.expires_at,
        device_flow: false,
    };
    let request = PrepareFederateSession {
        source: verified.clone(),
        requested: vec![STAGING_SCOPE.into()],
        challenge_id: vec![11; 32],
        challenge_created_at: now,
        ed_public: SigningKey::from_bytes(&[31; 32])
            .verifying_key()
            .to_bytes()
            .to_vec(),
        pq_public: vec![31; 1952],
        client: "client".into(),
    };
    let prepared = client.prepare_federate_session(&request).await?;
    assert_eq!(prepared.subject, ATPROTO_DID);
    let commit = CommitFederateSession {
        source: verified,
        policy_handle: prepared.policy_handle,
        challenge_id: request.challenge_id.clone(),
        challenge_created_at: now,
        challenge_expires_at: now + 60,
        ed_public: request.ed_public.clone(),
        pq_public: request.pq_public.clone(),
        client: "client".into(),
    };
    let committed = client.commit_federate_session(&commit).await?;
    assert_eq!(committed.subject, ATPROTO_DID);
    assert!(client.commit_federate_session(&commit).await.is_err());

    // The SAME long-lived local client must read its current exact-key
    // credential on every request, never falling back to its initial JWT.
    let trust = hyprstream_service::global_trust_store();
    let install = |claims: &Claims, expires_at| {
        trust.insert(
            oauth_key.verifying_key(),
            hyprstream_service::Attestation {
                scopes: ["oauth".to_owned()].into_iter().collect(),
                subject: None,
                jwt: Some(jwt::encode_service_jwt(claims, &authority_key)),
                expires_at,
                attested_by: Some(authority_key.verifying_key().to_bytes()),
            },
        );
    };
    let mut fresh_request = request.clone();
    fresh_request.challenge_id = vec![12; 32];
    fresh_request.source.jti = "renewed-source".to_owned();
    fresh_request.source.token_hash = vec![44; 32];
    install(
        &Claims {
            sub: "service:model".into(),
            ..claims.clone()
        },
        now + 121,
    );
    assert!(
        client
            .prepare_federate_session(&fresh_request)
            .await
            .is_err(),
        "a replaced invalid credential must not reuse the initial valid token"
    );
    let renewed = claims.clone().with_jti();
    install(&renewed, now + 122);
    assert_eq!(
        client
            .prepare_federate_session(&fresh_request)
            .await?
            .subject,
        ATPROTO_DID
    );
    // Generic trust insertion is monotonic in expiry. Remove before seeding
    // an already-expired entry to test expiry without sleeping or clock tricks.
    trust.remove(&oauth_key.verifying_key());
    install(&renewed, now - 1);
    assert!(
        client
            .prepare_federate_session(&fresh_request)
            .await
            .is_err(),
        "expired trust-store enrollment must not reuse an earlier token"
    );
    trust.remove(&oauth_key.verifying_key());
    assert!(
        client
            .prepare_federate_session(&fresh_request)
            .await
            .is_err(),
        "removed enrollment must not reuse an earlier token"
    );
    let expired = Claims {
        iat: now - 300,
        exp: now - 120,
        ..renewed.clone()
    };
    install(&expired, now + 120);
    assert!(
        client
            .prepare_federate_session(&fresh_request)
            .await
            .is_err(),
        "an expired current JWT must be rejected even with an unexpired attestation"
    );
    install(&renewed, now + 121);
    assert_eq!(
        client
            .prepare_federate_session(&fresh_request)
            .await?
            .subject,
        ATPROTO_DID
    );

    let missing = PolicyClient::for_local_transport_bootstrap(
        &transport,
        oauth_key.clone(),
        authority_key.verifying_key(),
        None,
    )?;
    assert!(missing.prepare_federate_session(&request).await.is_err());
    for bad_claims in [
        Claims {
            sub: "service:model".into(),
            ..claims.clone()
        },
        claims
            .clone()
            .with_cnf_jwk(SigningKey::from_bytes(&[32; 32]).verifying_key().as_bytes()),
    ] {
        let bad = PolicyClient::for_local_transport_bootstrap(
            &transport,
            oauth_key.clone(),
            authority_key.verifying_key(),
            Some(jwt::encode_service_jwt(&bad_claims, &authority_key)),
        )?;
        assert!(bad.prepare_federate_session(&request).await.is_err());
    }
    assert!(
        crate::services::oauth::runtime_policy_client(
            &transport,
            SigningKey::from_bytes(&[93; 32]),
            authority_key.verifying_key(),
        )
        .is_err(),
        "a different key cannot borrow OAuth's credential"
    );
    manager.stop("policy").await?;
    Ok(())
}

#[tokio::test]
async fn stateless_issuance_rejects_either_proof_component_if_in_trusted_inventory() {
    let (service, ctx) = fixture().await;
    let source = source();
    let requested = vec![STAGING_SCOPE.to_owned()];
    let now = chrono::Utc::now().timestamp();
    let model_ed = SigningKey::from_bytes(&[25; 32]).verifying_key().to_bytes();

    assert!(
        service
            .prepare_rpc(
                &ctx,
                "client",
                source.clone(),
                false,
                requested.clone(),
                [11; 32],
                now,
                model_ed,
                vec![31; 1952]
            )
            .await
            .is_err(),
        "foreign Ed25519 key was admitted as a Federate proof key"
    );
    assert!(
        service
            .prepare_rpc(
                &ctx,
                "client",
                source.clone(),
                false,
                requested.clone(),
                [12; 32],
                now,
                [31; 32],
                vec![27; 1952]
            )
            .await
            .is_err(),
        "foreign ML-DSA key was admitted as a Federate proof key"
    );

    let challenge_id = [13; 32];
    let ed_public = [31; 32];
    let pq_public = vec![32; 1952];
    let prepared = service
        .prepare_rpc(
            &ctx,
            "client",
            source.clone(),
            false,
            requested,
            challenge_id,
            now,
            ed_public,
            pq_public.clone(),
        )
        .await
        .expect("fresh proof-key pair should prepare");
    let expires_at = (now + 60)
        .min(source.expires_at)
        .min(source.issued_at + 120);
    // Simulate a corrupted/legacy pending handle so commit independently
    // rechecks the current trusted inventory instead of trusting prepare.
    service
        .pending
        .lock()
        .await
        .get_mut(&prepared.handle)
        .unwrap()
        .ed_public = model_ed;
    assert!(
        service
            .commit_rpc(
                &ctx,
                prepared.handle,
                "client",
                source,
                false,
                challenge_id,
                now,
                expires_at,
                model_ed,
                pq_public
            )
            .await
            .is_err(),
        "commit failed to recheck the key collision policy"
    );
}

// ── D2/D4: replay identity and tenant syntax ─────────────────────────────

#[test]
fn replay_identity_ignores_commitment_nonce_and_pins_the_client_class() {
    // D2: consumption identity derives only from immutable authenticated
    // fields — (issuer, client) namespace plus (jti, token_hash). The
    // commitment nonce/login_random never enters the key, so exactly one
    // consumption of a signed source token succeeds regardless of
    // client-supplied commitment inputs.
    let a = source();
    let mut b = source();
    b.nonce = "another-commitment-nonce".into();
    assert_eq!(
        super::source_credential_replay_key(&a, "client"),
        super::source_credential_replay_key(&b, "client")
    );
    let mut other_jti = source();
    other_jti.jti = "other-jti".into();
    assert_ne!(
        super::source_credential_replay_key(&a, "client"),
        super::source_credential_replay_key(&other_jti, "client")
    );
    assert_ne!(
        super::source_credential_replay_key(&a, "client"),
        super::source_credential_replay_key(&a, TUI_CLIENT_ID)
    );
}

#[test]
fn legacy_replay_key_differs_by_nonce_for_mixed_version_consumption() {
    // D2 transition: the legacy derivation includes the nonce so pre-upgrade
    // consumption records are recognized by post-upgrade consumers (the
    // consume path admits both keys); the current derivation ignores it.
    let a = source();
    let mut b = source();
    b.nonce = "another-commitment-nonce".into();
    assert_ne!(
        super::legacy_source_credential_replay_key(&a, "client"),
        super::legacy_source_credential_replay_key(&b, "client")
    );
    assert_eq!(
        super::source_credential_replay_key(&a, "client"),
        super::source_credential_replay_key(&b, "client")
    );
}

#[tokio::test]
async fn pre_d2_consumption_record_is_recognized_by_the_dual_key_consume() {
    let (service, _ctx) = fixture().await;
    let consumed = source();
    let expiry = u64::try_from(consumed.expires_at).unwrap();
    let replay = service.source_replay.as_ref().unwrap();
    // Simulate a pre-upgrade (nonce-bearing) consumption record.
    let legacy = super::legacy_source_credential_replay_key(&consumed, "client");
    assert!(matches!(
        replay.admit_source_credential(&legacy, expiry),
        hyprstream_rpc::proof::admission::ProofAdmissionResult::Admitted
    ));
    // The post-upgrade consume recognizes the token as consumed even though
    // its current-derivation key was never recorded.
    assert!(service
        .consume_source_credential(&consumed, "client", false)
        .is_err());
    // A fresh token still consumes exactly once (under both keys).
    let mut fresh = source();
    fresh.jti = "fresh-jti".into();
    fresh.token_hash = [9; 32];
    service
        .consume_source_credential(&fresh, "client", false)
        .unwrap();
    assert!(service
        .consume_source_credential(&fresh, "client", false)
        .is_err());
}

// ── W1b/D2: device-flow replay consumption ────────────────────────────────

/// Credential-insert fault injection over the in-memory store: the Nth
/// credential insert (1-based) returns Failed; 0 never fails.
struct FaultInjectingReplayStore {
    inner: hyprstream_rpc::proof::admission::InMemoryProofReplayStore,
    fail_on: std::sync::atomic::AtomicUsize,
    inserts: std::sync::atomic::AtomicUsize,
}

impl FaultInjectingReplayStore {
    fn new(fail_on: usize) -> Self {
        Self {
            inner:
                hyprstream_rpc::proof::admission::InMemoryProofReplayStore::single_verifier_instance(
                    128,
                ),
            fail_on: std::sync::atomic::AtomicUsize::new(fail_on),
            inserts: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn credential_inserts(&self) -> usize {
        self.inserts.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl hyprstream_rpc::proof::admission::ProofReplayStore for FaultInjectingReplayStore {
    fn domain_guarantee(&self) -> hyprstream_rpc::proof::admission::ReplayDomainGuarantee {
        hyprstream_rpc::proof::admission::ReplayDomainGuarantee::LinearizableSharedStore
    }

    fn check_and_insert(
        &self,
        partition: hyprstream_rpc::proof::ProofDisposition,
        key: &hyprstream_rpc::proof::admission::ProofReplayKey,
        expires_at: u64,
    ) -> hyprstream_rpc::proof::admission::ProofAdmissionResult {
        self.inner.check_and_insert(partition, key, expires_at)
    }

    fn check_and_insert_mediated(
        &self,
        key: &hyprstream_rpc::proof::admission::MediatedQueryReplayKey,
        expires_at: u64,
    ) -> hyprstream_rpc::proof::admission::ProofAdmissionResult {
        self.inner.check_and_insert_mediated(key, expires_at)
    }

    fn check_and_insert_credential(
        &self,
        key: &hyprstream_rpc::proof::admission::CredentialReplayKey,
        expires_at: u64,
    ) -> hyprstream_rpc::proof::admission::ProofAdmissionResult {
        let n = self
            .inserts
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        if n == self.fail_on.load(std::sync::atomic::Ordering::SeqCst) {
            return hyprstream_rpc::proof::admission::ProofAdmissionResult::Failed;
        }
        self.inner.check_and_insert_credential(key, expires_at)
    }
}

fn install_replay(
    service: &mut AdmissionService,
    fail_on: usize,
) -> Arc<FaultInjectingReplayStore> {
    let store = Arc::new(FaultInjectingReplayStore::new(fail_on));
    service.source_replay = Some(store.clone());
    store
}

#[tokio::test]
async fn device_flow_consumes_only_the_nonce_free_identity() {
    // W1b: a device-flow source consumes under the CURRENT identity only.
    // The legacy key is never recorded, so no fresh-legacy-key replay
    // window exists even for a token re-presented under a different
    // (client-chosen) device commitment nonce.
    let (mut service, _ctx) = fixture().await;
    install_replay(&mut service, 0);
    let device = source();
    let expiry = u64::try_from(device.expires_at).unwrap();
    service
        .consume_source_credential(&device, "client", true)
        .unwrap();
    // Same token under a DIFFERENT device commitment nonce (same
    // jti/token_hash): the current key is recorded -> Replayed.
    let mut varied = source();
    varied.nonce = "another-device-commitment".into();
    assert!(service
        .consume_source_credential(&varied, "client", true)
        .is_err());
    // The legacy (nonce-bearing) key was never written: it still admits
    // standalone, proving the device consume performed a single write.
    let replay = service.source_replay.as_ref().unwrap();
    let legacy = super::legacy_source_credential_replay_key(&device, "client");
    assert!(matches!(
        replay.admit_source_credential(&legacy, expiry),
        hyprstream_rpc::proof::admission::ProofAdmissionResult::Admitted
    ));
    // Code-flow control, same service: the dual write still applies, and a
    // pre-recorded legacy record denies a later code-flow consume.
    let mut code = source();
    code.jti = "code-jti".into();
    code.token_hash = [2; 32];
    let legacy_code = super::legacy_source_credential_replay_key(&code, "client");
    assert!(matches!(
        replay.admit_source_credential(&legacy_code, expiry),
        hyprstream_rpc::proof::admission::ProofAdmissionResult::Admitted
    ));
    assert!(service
        .consume_source_credential(&code, "client", false)
        .is_err());
}

#[tokio::test]
async fn device_flow_write_failure_cannot_enable_double_consume() {
    // Fault injection: the single (current-key) write FAILS on the first
    // attempt -> consumption aborts fail-closed, nothing recorded. A retry
    // under a DIFFERENT device commitment re-attempts the SAME current key:
    // it admits exactly once; every later attempt is Replayed. No pair of
    // attempts can both consume (the W1b fix invariant).
    let (mut service, _ctx) = fixture().await;
    let store = install_replay(&mut service, 1);
    let device = source();
    assert!(service
        .consume_source_credential(&device, "client", true)
        .is_err());
    assert_eq!(store.credential_inserts(), 1);
    let mut varied = source();
    varied.nonce = "different-device-commitment".into();
    service
        .consume_source_credential(&varied, "client", true)
        .unwrap();
    assert!(service
        .consume_source_credential(&device, "client", true)
        .is_err());
    assert!(service
        .consume_source_credential(&varied, "client", true)
        .is_err());
    assert_eq!(store.credential_inserts(), 4);
}

#[tokio::test]
async fn code_flow_failure_between_legacy_and_current_writes_denies_replay() {
    // Fault injection on the CODE path (dual write retained): legacy
    // admitted, current write FAILS -> consumption aborts. Retrying the
    // SAME token hits the already-recorded legacy key: Replayed. The
    // nonce is issuer-signed on this flow, so no different-commitment
    // variant of the same token can exist.
    let (mut service, _ctx) = fixture().await;
    let store = install_replay(&mut service, 2);
    let code = source();
    assert!(service
        .consume_source_credential(&code, "client", false)
        .is_err());
    assert_eq!(store.credential_inserts(), 2);
    assert!(service
        .consume_source_credential(&code, "client", false)
        .is_err());
    assert_eq!(store.credential_inserts(), 3);
}

#[tokio::test]
async fn malformed_policy_tenant_rejects_the_federate_decision() {
    // D4: a Policy grant carrying a tenant the HTTP middleware syntax would
    // reject must fail the admission decision outright, for both classes.
    for bad_tenant in ["bad/tenant", "..", "a b"] {
        let (service, ctx) = fixture().await;
        let authority = service.authority.as_ref().unwrap();
        authority
            .policy
            .add_policy_with_domain(
                ATPROTO_DID,
                bad_tenant,
                "model:qwen2.5-0.5b-instruct:main",
                "infer",
                "allow",
            )
            .await
            .unwrap();
        assert!(
            service
                .prepare_rpc(
                    &ctx,
                    "client",
                    source(),
                    false,
                    vec![STAGING_SCOPE.to_owned()],
                    [31; 32],
                    chrono::Utc::now().timestamp(),
                    [51; 32],
                    vec![52; 1952]
                )
                .await
                .is_err(),
            "malformed Policy tenant {bad_tenant} was admitted"
        );
    }
}

// ── W1.2: the per-client grant/profile table beside the fixed www pin ─────

const TUI_CLIENT_ID: &str = "hyprstream-tui-staging";
const TUI_CONNECT_SCOPE: &str = "connect:tui:session";
const TUI_QUERY_SCOPE: &str = "query:tui:session";

/// Fixture variant whose Authorities carries the configured TUI table entry
/// (the same fixed shape `AdmissionService::configured` builds from the
/// trusted-issuer `federate_tui_client` class) plus a Casbin grant of the
/// TUI session resource to the verified DID.
async fn tui_fixture() -> (AdmissionService, EnvelopeContext) {
    let (mut service, ctx) = fixture().await;
    let authority = service.authority.as_mut().unwrap();
    authority.tui_profile = Some(Profile {
        issuer: "https://issuer.test".into(),
        host: "https://host.test".into(),
        client: TUI_CLIENT_ID.into(),
        resource: "https://host.test".into(),
        client_scopes: BTreeSet::from([TUI_CONNECT_SCOPE.into(), TUI_QUERY_SCOPE.into()]),
        resource_scopes: BTreeSet::from([TUI_CONNECT_SCOPE.into(), TUI_QUERY_SCOPE.into()]),
    });
    authority
        .policy
        .add_policy_with_domain(ATPROTO_DID, "tenant", "tui:session", "connect", "allow")
        .await
        .unwrap();
    authority
        .policy
        .add_policy_with_domain(ATPROTO_DID, "tenant", "tui:session", "query", "allow")
        .await
        .unwrap();
    (service, ctx)
}

#[tokio::test]
async fn w12_tui_class_prepares_and_commits_under_its_own_profile_table_entry() {
    let (service, ctx) = tui_fixture().await;
    let now = chrono::Utc::now().timestamp();
    let prepared = service
        .prepare_rpc(
            &ctx,
            TUI_CLIENT_ID,
            source(),
            false,
            vec![TUI_CONNECT_SCOPE.into()],
            [21; 32],
            now,
            [51; 32],
            vec![52; 1952],
        )
        .await
        .unwrap();
    assert_eq!(prepared.subject, ATPROTO_DID);
    assert_eq!(prepared.granted, [TUI_CONNECT_SCOPE.to_owned()]);
    let session = service
        .commit_rpc(
            &ctx,
            prepared.handle,
            TUI_CLIENT_ID,
            source(),
            false,
            [21; 32],
            now,
            now + 60,
            [51; 32],
            vec![52; 1952],
        )
        .await
        .unwrap();
    assert_eq!(session.client_id, TUI_CLIENT_ID);
    assert_eq!(session.scopes, [TUI_CONNECT_SCOPE.to_owned()]);
    // A commit naming any other client consumes the one-use handle.
    let prepared = service
        .prepare_rpc(
            &ctx,
            TUI_CLIENT_ID,
            source(),
            false,
            vec![TUI_CONNECT_SCOPE.into()],
            [22; 32],
            now,
            [51; 32],
            vec![52; 1952],
        )
        .await
        .unwrap();
    assert!(
        service
            .commit_rpc(
                &ctx,
                prepared.handle,
                "client",
                source(),
                false,
                [22; 32],
                now,
                now + 60,
                [51; 32],
                vec![52; 1952]
            )
            .await
            .is_err(),
        "a www-class commit must not redeem a TUI-class challenge"
    );
    assert!(service.pending.lock().await.is_empty());
}

#[tokio::test]
async fn w12_per_client_scope_ceilings_do_not_intersect() {
    let (service, ctx) = tui_fixture().await;
    // The TUI class cannot be granted any www staging scope, even with a
    // live Casbin grant for it: the TUI table entry's ceiling excludes it.
    for www_scope in [STAGING_SCOPE, REGISTRY_LIST_SCOPE] {
        assert!(
            service
                .prepare_rpc(
                    &ctx,
                    TUI_CLIENT_ID,
                    source(),
                    false,
                    vec![www_scope.into()],
                    [23; 32],
                    chrono::Utc::now().timestamp(),
                    [51; 32],
                    vec![52; 1952]
                )
                .await
                .is_err(),
            "TUI class granted the www scope {www_scope}"
        );
    }
    // Symmetrically the www class cannot be granted a TUI session scope
    // even though the fixture just installed a live tui:session grant.
    assert!(
        service
            .prepare_rpc(
                &ctx,
                "client",
                source(),
                false,
                vec![TUI_CONNECT_SCOPE.into()],
                [24; 32],
                chrono::Utc::now().timestamp(),
                [51; 32],
                vec![52; 1952]
            )
            .await
            .is_err(),
        "www class granted the TUI scope"
    );
}

#[tokio::test]
async fn w12_unconfigured_tui_class_default_denies_at_admission() {
    // fixture() builds Authorities with tui_profile: None — the exact
    // default-deny state when `federate_tui_client` is unconfigured.
    let (service, ctx) = fixture().await;
    assert!(
        service
            .prepare_rpc(
                &ctx,
                TUI_CLIENT_ID,
                source(),
                false,
                vec![TUI_CONNECT_SCOPE.into()],
                [25; 32],
                chrono::Utc::now().timestamp(),
                [51; 32],
                vec![52; 1952]
            )
            .await
            .is_err(),
        "unconfigured TUI class must default-deny before any grant work"
    );
    // The www pin still resolves and admits exactly as before.
    let decision = service
        .prepare(&ctx, "client", &source(), &[STAGING_SCOPE.to_owned()])
        .await
        .unwrap();
    assert_eq!(decision.scopes, [STAGING_SCOPE]);
}

#[tokio::test]
async fn w12_deferred_request_use_reader_is_www_only() {
    let (service, _) = tui_fixture().await;
    let authority = service.authority.as_ref().unwrap();
    let mut request = AdmitFederateRequest {
        issuer: "https://host.test".into(),
        profile: hyprstream_session_store::PROFILE.into(),
        sid: "correlation-only".into(),
        subject: ATPROTO_DID.into(),
        tenant: "tenant".into(),
        client: TUI_CLIENT_ID.into(),
        audience: "https://host.test".into(),
        scopes: vec![TUI_CONNECT_SCOPE.into()],
        ed_public: vec![1; 32],
        suite_thumbprint: vec![2; 32],
        generation: authority.serving_generation.to_vec(),
        collision_inventory_id: authority.local_collision_inventory_id.to_vec(),
        expires_at: chrono::Utc::now().timestamp() + 60,
        proof_epoch: 1,
        verified_namespace: vec![4; 32],
        request_id: vec![5; 16],
        resource: "tui:session".into(),
        operation: "connect".into(),
    };
    let reader = RequestUseReader::new(Arc::new(service), None);
    let model_ctx = EnvelopeContext::for_test_authenticated_subject_with_claims(
        Subject::new("service:model"),
        "*",
        SigningKey::from_bytes(&[25; 32]).verifying_key(),
        hyprstream_rpc::auth::Claims::new("service:model".into(), 0, i64::MAX),
    );
    assert!(
        reader.admit(&model_ctx, &request).await.is_err(),
        "the Registry/Model deferred consumer must reject a TUI-class session"
    );
    request.client = "client".into();
    // The www-class shape still fails on the tui resource/scope ceiling
    // (TUI scopes are outside the www ceiling), then succeeds for its own.
    assert!(reader.admit(&model_ctx, &request).await.is_err());
    request.scopes = vec![STAGING_SCOPE.into()];
    request.resource = format!("model:{STAGING_MODEL_REF}");
    request.operation = "infer".into();
    reader.admit(&model_ctx, &request).await.unwrap();
}

#[tokio::test]
async fn rpc_prepare_and_commit_are_stateless_and_consume_mismatched_handles() {
    let (service, ctx) = fixture().await;
    let now = chrono::Utc::now().timestamp();
    let verified_source = source();
    let requested = vec![STAGING_SCOPE.to_owned()];
    let prepared = service
        .prepare_rpc(
            &ctx,
            "client",
            verified_source.clone(),
            false,
            requested,
            [9; 32],
            now,
            [1; 32],
            vec![2; 1952],
        )
        .await
        .unwrap();
    assert_eq!(service.pending.lock().await.len(), 1);
    // A wrong key consumes the one-use handle before any token can be minted.
    assert!(service
        .commit_rpc(
            &ctx,
            prepared.handle,
            "client",
            verified_source.clone(),
            false,
            [9; 32],
            now,
            now + 60,
            [3; 32],
            vec![2; 1952]
        )
        .await
        .is_err());
    assert!(service.pending.lock().await.is_empty());
    assert!(service
        .commit_rpc(
            &ctx,
            prepared.handle,
            "client",
            verified_source,
            false,
            [9; 32],
            now,
            now + 60,
            [1; 32],
            vec![2; 1952]
        )
        .await
        .is_err());

    let prepared = service
        .prepare_rpc(
            &ctx,
            "client",
            source(),
            false,
            vec![STAGING_SCOPE.into()],
            [10; 32],
            now,
            [1; 32],
            vec![2; 1952],
        )
        .await
        .unwrap();
    let mut substituted_source = source();
    substituted_source.token_hash = [7; 32];
    assert!(service
        .commit_rpc(
            &ctx,
            prepared.handle,
            "client",
            substituted_source,
            false,
            [10; 32],
            now,
            now + 60,
            [1; 32],
            vec![2; 1952]
        )
        .await
        .is_err());
    assert!(service.pending.lock().await.is_empty());

    let prepared = service
        .prepare_rpc(
            &ctx,
            "client",
            source(),
            false,
            vec![STAGING_SCOPE.into()],
            [11; 32],
            now,
            [1; 32],
            vec![2; 1952],
        )
        .await
        .unwrap();
    assert!(service
        .commit_rpc(
            &ctx,
            prepared.handle,
            "client",
            source(),
            false,
            [12; 32],
            now,
            now + 60,
            [1; 32],
            vec![2; 1952]
        )
        .await
        .is_err());
    assert!(service.pending.lock().await.is_empty());

    let prepared = service
        .prepare_rpc(
            &ctx,
            "client",
            source(),
            false,
            vec![STAGING_SCOPE.into()],
            [13; 32],
            now,
            [1; 32],
            vec![2; 1952],
        )
        .await
        .unwrap();
    let session = service
        .commit_rpc(
            &ctx,
            prepared.handle,
            "client",
            source(),
            false,
            [13; 32],
            now,
            now + 60,
            [1; 32],
            vec![2; 1952],
        )
        .await
        .unwrap();
    assert_eq!(session.subject, ATPROTO_DID);
    assert_eq!(session.proof_epoch, 1);
    assert_eq!(session.scopes, [STAGING_SCOPE.to_owned()]);

    let replayed = source();
    let prepared_again = service
        .prepare_rpc(
            &ctx,
            "client",
            replayed.clone(),
            false,
            vec![STAGING_SCOPE.into()],
            [14; 32],
            now,
            [1; 32],
            vec![2; 1952],
        )
        .await
        .unwrap();
    assert!(
        service
            .commit_rpc(
                &ctx,
                prepared_again.handle,
                "client",
                replayed,
                false,
                [14; 32],
                now,
                now + 60,
                [1; 32],
                vec![2; 1952]
            )
            .await
            .is_err(),
        "a fresh challenge must not redeem the same source credential twice"
    );
}

#[tokio::test]
async fn h2_request_use_requires_exact_enrolled_serving_holder() {
    let (service, _) = fixture().await;
    let manifest = &service.authority.as_ref().unwrap().enrollment;
    for (name, seed, expected) in [
        ("service:model", 25, "model"),
        ("service:registry", 26, "registry"),
    ] {
        let ctx = EnvelopeContext::for_test_authenticated_subject_with_claims(
            Subject::new(name),
            "*",
            SigningKey::from_bytes(&[seed; 32]).verifying_key(),
            hyprstream_rpc::auth::Claims::new(name.into(), 0, i64::MAX),
        );
        assert_eq!(serving_caller(&ctx, manifest).unwrap(), expected);
        let wrong_key = EnvelopeContext::for_test_authenticated_subject_with_claims(
            Subject::new(name),
            "*",
            SigningKey::from_bytes(&[27; 32]).verifying_key(),
            hyprstream_rpc::auth::Claims::new(name.into(), 0, i64::MAX),
        );
        assert!(serving_caller(&wrong_key, manifest).is_err());
        let wrong_subject = EnvelopeContext::for_test_authenticated_subject_with_claims(
            Subject::new(name),
            "*",
            SigningKey::from_bytes(&[seed; 32]).verifying_key(),
            hyprstream_rpc::auth::Claims::new("service:oauth".into(), 0, i64::MAX),
        );
        assert!(serving_caller(&wrong_subject, manifest).is_err());
    }
    let oauth = EnvelopeContext::for_test_authenticated_subject_with_claims(
        Subject::new("service:oauth"),
        "*",
        SigningKey::from_bytes(&[24; 32]).verifying_key(),
        hyprstream_rpc::auth::Claims::new("service:oauth".into(), 0, i64::MAX),
    );
    assert!(serving_caller(&oauth, manifest).is_err());
}

#[tokio::test]
async fn stateless_request_use_rechecks_the_live_did_grant_each_call() {
    let (service, _) = fixture().await;
    let authority = service.authority.as_ref().unwrap();
    let generation = authority.serving_generation.to_vec();
    let collision_inventory_id = authority.local_collision_inventory_id.to_vec();
    let reader = RequestUseReader::new(Arc::new(service), None);
    let model_ctx = EnvelopeContext::for_test_authenticated_subject_with_claims(
        Subject::new("service:model"),
        "*",
        SigningKey::from_bytes(&[25; 32]).verifying_key(),
        hyprstream_rpc::auth::Claims::new("service:model".into(), 0, i64::MAX),
    );
    let mut request = AdmitFederateRequest {
        issuer: "https://host.test".into(),
        profile: hyprstream_session_store::PROFILE.into(),
        sid: "correlation-only-sid".into(),
        subject: ATPROTO_DID.into(),
        tenant: "tenant".into(),
        client: "client".into(),
        audience: "https://host.test".into(),
        scopes: vec![STAGING_SCOPE.into()],
        ed_public: vec![1; 32],
        suite_thumbprint: vec![2; 32],
        generation,
        collision_inventory_id,
        expires_at: chrono::Utc::now().timestamp() + 60,
        proof_epoch: 1,
        verified_namespace: vec![4; 32],
        request_id: vec![5; 16],
        resource: format!("model:{STAGING_MODEL_REF}"),
        operation: "infer".into(),
    };
    reader.admit(&model_ctx, &request).await.unwrap();
    request.sid = "different-correlation-only-sid".into();
    reader.admit(&model_ctx, &request).await.unwrap();

    let policy = Arc::clone(&reader.admission.authority.as_ref().unwrap().policy);
    assert!(policy
        .remove_policy_with_domain(
            ATPROTO_DID,
            "tenant",
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
            "allow",
        )
        .await
        .unwrap());
    assert!(reader.admit(&model_ctx, &request).await.is_err());
}

#[tokio::test]
async fn configured_host_policy_admits_only_its_exact_issuer_and_resource() {
    for local in ["https://vm-a.example", "https://vm-b.example:8443"] {
        let (mut fixture, _) = fixture().await;
        let authority = fixture.authority.take().unwrap();
        let service = AdmissionService::configured(
            authority.policy,
            authority.enrollment,
            authority.collision_inventory,
            None,
            crate::services::oauth::federate_source::FederateHost::new(local).unwrap(),
            None,
        );
        let authority = service.authority.as_ref().unwrap();
        assert_eq!(authority.profile.host, local);
        assert_eq!(authority.profile.resource, local);
        let mut request = AdmitFederateRequest {
            issuer: local.into(),
            profile: hyprstream_session_store::PROFILE.into(),
            sid: "correlation-only".into(),
            subject: ATPROTO_DID.into(),
            tenant: "tenant".into(),
            client: crate::services::oauth::federate_source::CLIENT.into(),
            audience: local.into(),
            scopes: vec![STAGING_SCOPE.into()],
            ed_public: vec![1; 32],
            suite_thumbprint: vec![2; 32],
            generation: authority.serving_generation.to_vec(),
            collision_inventory_id: authority.local_collision_inventory_id.to_vec(),
            expires_at: chrono::Utc::now().timestamp() + 60,
            proof_epoch: 1,
            verified_namespace: vec![4; 32],
            request_id: vec![5; 16],
            resource: format!("model:{STAGING_MODEL_REF}"),
            operation: "infer".into(),
        };
        let reader = RequestUseReader::new(Arc::new(service), None);
        let ctx = EnvelopeContext::for_test_authenticated_subject_with_claims(
            Subject::new("service:model"),
            "*",
            SigningKey::from_bytes(&[25; 32]).verifying_key(),
            hyprstream_rpc::auth::Claims::new("service:model".into(), 0, i64::MAX),
        );
        reader.admit(&ctx, &request).await.unwrap();
        request.issuer = "https://other.example".into();
        assert!(reader.admit(&ctx, &request).await.is_err());
        request.issuer = local.into();
        request.audience = "https://other.example".into();
        assert!(reader.admit(&ctx, &request).await.is_err());
    }
}

#[tokio::test]
async fn h2_admission_disabled_caller_and_scope_boundaries() {
    let (service, ctx) = fixture().await;
    let requested = vec![STAGING_SCOPE.into(), "read:model:other".into()];
    assert!(AdmissionService::default()
        .prepare(&ctx, "client", &source(), &requested)
        .await
        .is_err());
    let decision = service
        .prepare(&ctx, "client", &source(), &requested)
        .await
        .unwrap();
    assert_eq!(decision.scopes, [STAGING_SCOPE]);
    for who in [
        Subject::new("alice"),
        Subject::new("service:model"),
        Subject::federated("https://foreign.test", "service:oauth"),
    ] {
        let wrong = EnvelopeContext::for_test_authenticated_subject(
            who,
            SigningKey::from_bytes(&[24; 32]).verifying_key(),
        );
        assert!(service
            .prepare(&wrong, "client", &source(), &requested)
            .await
            .is_err());
    }
    let wrong = EnvelopeContext::for_test_authenticated_subject_with_claims(
        Subject::new("service:oauth"),
        "*",
        SigningKey::from_bytes(&[25; 32]).verifying_key(),
        hyprstream_rpc::auth::Claims::new("service:oauth".into(), 0, i64::MAX),
    );
    assert!(service
        .prepare(&wrong, "client", &source(), &requested)
        .await
        .is_err());
    assert!(service
        .prepare(&ctx, "client", &source(), &["read:model:other".into()])
        .await
        .is_err());
    for nonmatching in [
        "infer:model:qwen2.5-0.5b-instruct:other",
        "infer:model:other:main",
        "read:model:qwen2.5-0.5b-instruct:main",
        "infer:model:qwen2.5-0.5b-instruct",
    ] {
        assert!(
            service
                .prepare(&ctx, "client", &source(), &[nonmatching.into()])
                .await
                .is_err(),
            "nonmatching scope unexpectedly intersected: {nonmatching}"
        );
    }
    assert!(canonical_requested(&["infer:model:*".into()]).is_err());
    assert!(canonical_requested(&["infer:model:qwen".into(), "infer:model:qwen".into()]).is_err());
    for malformed in [
        "infer:model:qwen2.5-0.5b-instruct::main",
        "infer:model:qwen2.5-0.5b-instruct:main:other",
        "infer:registry:qwen2.5-0.5b-instruct:main",
    ] {
        assert!(
            canonical_requested(&[malformed.into()]).is_err(),
            "accepted malformed or ambiguous scope: {malformed}"
        );
    }
}

#[tokio::test]
async fn h2_exact_model_ref_grant_matches_model_dispatch_resource() {
    let (service, ctx) = fixture().await;
    let requested = vec![STAGING_SCOPE.into()];
    canonical_requested(&requested).unwrap();
    let parsed = Scope::parse(STAGING_SCOPE).unwrap();
    let dispatch_resource = format!("model:{STAGING_MODEL_REF}");
    assert_eq!(parsed.policy_resource(), dispatch_resource);

    let decision = service
        .prepare(&ctx, "client", &source(), &requested)
        .await
        .unwrap();
    assert_eq!(decision.subject, ATPROTO_DID);
    assert_eq!(decision.scopes, [STAGING_SCOPE]);
    assert_eq!(decision.scopes[0], STAGING_SCOPE);
}

#[tokio::test]
async fn h2_registry_only_session_admits_list_but_not_model_infer_or_get() {
    let (mut service, ctx) = fixture().await;
    let authority = service.authority.as_mut().unwrap();
    authority
        .profile
        .client_scopes
        .insert(REGISTRY_LIST_SCOPE.into());
    authority
        .profile
        .resource_scopes
        .insert(REGISTRY_LIST_SCOPE.into());
    authority
        .policy
        .add_policy_with_domain(ATPROTO_DID, "tenant", "registry:List", "query", "allow")
        .await
        .unwrap();
    let source_record = source();
    let decision = service
        .prepare(
            &ctx,
            "client",
            &source_record,
            &[REGISTRY_LIST_SCOPE.into()],
        )
        .await
        .unwrap();
    assert_eq!(decision.scopes, [REGISTRY_LIST_SCOPE]);
    let session = session(&decision);
    let authority = service.authority.as_ref().unwrap();
    let now = chrono::Utc::now().timestamp();
    authority
        .authorize_use(
            &session,
            &source_record.issuer,
            &source_record.atproto_did,
            "registry:List",
            "query",
            now,
        )
        .await
        .unwrap();
    for (resource, operation) in [
        ("registry:Get", "query"),
        ("registry:List", "write"),
        ("model:qwen2.5-0.5b-instruct:main", "infer"),
    ] {
        assert!(
            authority
                .authorize_use(
                    &session,
                    &source_record.issuer,
                    &source_record.atproto_did,
                    resource,
                    operation,
                    now
                )
                .await
                .is_err(),
            "Registry-only session admitted {resource}/{operation}"
        );
    }
}

#[tokio::test]
async fn h2_admission_uses_verified_did_without_a_local_account() {
    let (service, ctx) = fixture().await;
    let requested = vec![STAGING_SCOPE.into()];
    let first = service
        .prepare(&ctx, "client", &source(), &requested)
        .await
        .unwrap();
    let again = service
        .prepare(&ctx, "client", &source(), &requested)
        .await
        .unwrap();
    assert!(first == again);
    assert_eq!(first.subject, ATPROTO_DID);
    assert_eq!(
        first.account_id,
        federate_account_id("https://host.test", ATPROTO_DID)
    );
    let policy = &service.authority.as_ref().unwrap().policy;
    assert!(policy
        .remove_policy_with_domain(
            ATPROTO_DID,
            "tenant",
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
            "allow",
        )
        .await
        .unwrap());
    assert!(service
        .prepare(&ctx, "client", &source(), &requested)
        .await
        .is_err());
    policy
        .add_policy_with_domain(
            ATPROTO_DID,
            "tenant",
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
            "allow",
        )
        .await
        .unwrap();
    let decision = service
        .prepare(&ctx, "client", &source(), &requested)
        .await
        .unwrap();
    assert_eq!(decision.subject, ATPROTO_DID);
    assert_eq!(decision.account_id, first.account_id);
    let _permit = service.capacity.as_ref().unwrap().try_acquire().unwrap();
    assert!(service
        .prepare(&ctx, "client", &source(), &requested)
        .await
        .is_err());
}

#[tokio::test]
async fn h2_ambiguous_did_tenant_grants_fail_closed() {
    let (service, ctx) = fixture().await;
    service
        .authority
        .as_ref()
        .unwrap()
        .policy
        .add_policy_with_domain(
            ATPROTO_DID,
            "another-tenant",
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
            "allow",
        )
        .await
        .unwrap();
    assert!(service
        .prepare(&ctx, "client", &source(), &[STAGING_SCOPE.into()])
        .await
        .is_err());
}

async fn connect(socket: &str, database: &str) -> tokio_postgres::Client {
    let (client, connection) = tokio_postgres::Config::new()
        .host_path(socket)
        .user("postgres")
        .dbname(database)
        .connect(tokio_postgres::NoTls)
        .await
        .unwrap();
    tokio::spawn(async move {
        let _ = connection.await;
    });
    client
}
fn evidence(challenge: Decision, id: &str) -> PossessionEvidence {
    let source = source();
    PossessionEvidence {
        client: "client".into(),
        device_flow: false,
        created_at: source.issued_at,
        expires_at: source.issued_at + 60,
        session_expires_at: source.expires_at,
        source,
        ed_public: [1; 32],
        pq_public: vec![2; 1952],
        sid: id.into(),
        requested: vec![STAGING_SCOPE.into()],
        challenge,
    }
}

fn session(challenge: &Decision) -> Session {
    Session {
        host: "https://host.test".into(),
        sid: "sid-1".into(),
        account_id: challenge.account_id.clone(),
        subject: challenge.subject.clone(),
        tenant: challenge.tenant.clone(),
        client_id: "client".into(),
        resource: "https://host.test".into(),
        scopes: challenge.scopes.clone(),
        grant_revision: challenge.revision.clone(),
        ed_public: [1; 32],
        pq_public: vec![2; 1952],
        generation: challenge.generation,
        collision_inventory_id: challenge.collision_inventory_id,
        proof_epoch: 1,
        expires_at: chrono::Utc::now().timestamp() + 60,
    }
}

#[tokio::test]
async fn h2_current_use_is_fresh_scoped_and_revocation_denies_next_boundary() {
    let (service, ctx) = fixture().await;
    let source_record = source();
    let decision = service
        .prepare(&ctx, "client", &source_record, &[STAGING_SCOPE.into()])
        .await
        .unwrap();
    let session = session(&decision);
    let authority = service.authority.as_ref().unwrap();
    let resource = format!("model:{STAGING_MODEL_REF}");

    authority
        .authorize_use(
            &session,
            &source_record.issuer,
            &source_record.atproto_did,
            &resource,
            "infer",
            chrono::Utc::now().timestamp(),
        )
        .await
        .unwrap();
    assert!(authority
        .authorize_use(
            &session,
            &source_record.issuer,
            &source_record.atproto_did,
            "model:qwen2.5-0.5b-instruct:other",
            "infer",
            chrono::Utc::now().timestamp()
        )
        .await
        .is_err());
    assert!(authority
        .authorize_use(
            &session,
            &source_record.issuer,
            &source_record.atproto_did,
            &resource,
            "query",
            chrono::Utc::now().timestamp()
        )
        .await
        .is_err());
    let mut expired = session.clone();
    expired.expires_at = chrono::Utc::now().timestamp();
    assert!(authority
        .authorize_use(
            &expired,
            &source_record.issuer,
            &source_record.atproto_did,
            &resource,
            "infer",
            chrono::Utc::now().timestamp()
        )
        .await
        .is_err());
    let mut wrong_scope = session.clone();
    wrong_scope.scopes = vec!["infer:model:qwen2.5-0.5b-instruct:other".into()];
    assert!(authority
        .authorize_use(
            &wrong_scope,
            &source_record.issuer,
            &source_record.atproto_did,
            &resource,
            "infer",
            chrono::Utc::now().timestamp()
        )
        .await
        .is_err());
    let mut wrong_source = source();
    wrong_source.atproto_did = "did:plc:differentprincipal".into();
    assert!(authority
        .authorize_use(
            &session,
            &wrong_source.issuer,
            &wrong_source.atproto_did,
            &resource,
            "infer",
            chrono::Utc::now().timestamp()
        )
        .await
        .is_err());
    let mut wrong_session = session.clone();
    wrong_session.account_id = uuid::Uuid::new_v4().to_string();
    assert!(authority
        .authorize_use(
            &wrong_session,
            &source_record.issuer,
            &source_record.atproto_did,
            &resource,
            "infer",
            chrono::Utc::now().timestamp()
        )
        .await
        .is_err());

    // The first decision is the authorization point for a streaming request;
    // revocation does not poll/cancel that request. Its next request/tool
    // boundary does a fresh read and denies before starting the operation.
    let removed = authority
        .policy
        .remove_policy_with_domain(ATPROTO_DID, "tenant", &resource, "infer", "allow")
        .await
        .unwrap();
    assert!(removed);
    assert!(
        authority
            .authorize_use(
                &session,
                &source_record.issuer,
                &source_record.atproto_did,
                &resource,
                "infer",
                chrono::Utc::now().timestamp()
            )
            .await
            .is_err(),
        "next distinct boundary must observe revoke"
    );

    // A Policy deny is the suspension mechanism and overrides an existing
    // allow on the next fresh check, without depending on token refresh.
    authority
        .policy
        .add_policy_with_domain(ATPROTO_DID, "tenant", &resource, "infer", "allow")
        .await
        .unwrap();
    authority
        .policy
        .add_policy_with_domain(ATPROTO_DID, "tenant", &resource, "infer", "deny")
        .await
        .unwrap();
    assert!(authority
        .authorize_use(
            &session,
            &source_record.issuer,
            &source_record.atproto_did,
            &resource,
            "infer",
            chrono::Utc::now().timestamp()
        )
        .await
        .is_err());
    assert!(authority
        .policy
        .remove_policy_with_domain(ATPROTO_DID, "tenant", &resource, "infer", "deny")
        .await
        .unwrap());
    assert!(authority
        .authorize_use(
            &session,
            &source_record.issuer,
            &source_record.atproto_did,
            &resource,
            "infer",
            chrono::Utc::now().timestamp()
        )
        .await
        .is_ok());
}

/// Only a disposable local Unix-socket fixture; never an environment DSN.
#[tokio::test]
#[ignore = "requires H2_TEST_SOCKET disposable PostgreSQL fixture"]
async fn h2_admission_pg_race_replay_revocation_and_outage() {
    let socket = std::env::var("H2_TEST_SOCKET").expect("local fixture socket");
    assert!(socket.starts_with("/home/") && !socket.contains("://"));
    let db = format!("h2_{}", uuid::Uuid::new_v4().simple());
    let admin = connect(&socket, "postgres").await;
    admin
        .batch_execute(&format!("CREATE DATABASE {db}"))
        .await
        .unwrap();
    let mut control = connect(&socket, &db).await;
    control
        .batch_execute(hyprstream_session_store::MIGRATION)
        .await
        .unwrap();
    control
        .batch_execute(hyprstream_session_store::MIGRATION_V2)
        .await
        .unwrap();
    control
        .batch_execute(hyprstream_session_store::MIGRATION_V3)
        .await
        .unwrap();
    control
        .batch_execute(hyprstream_session_store::MIGRATION_V4)
        .await
        .unwrap();
    control
        .batch_execute(hyprstream_session_store::MIGRATION_V5)
        .await
        .unwrap();
    let mut client = connect(&socket, &db).await;
    let (mut service, ctx) = fixture().await;
    let authority = service.authority.as_ref().unwrap();
    let serving_generation = authority.serving_generation;
    let collision_inventory_id = authority.local_collision_inventory_id;
    control.execute("INSERT INTO federate_session.profile_state(host,profile,enabled,authority_generation,collision_inventory_id) VALUES ($1,$2,true,$3,$4)",
        &[&"https://host.test", &hyprstream_session_store::PROFILE, &&serving_generation[..], &&collision_inventory_id[..]]).await.unwrap();
    let mut session_config = tokio_postgres::Config::new();
    session_config
        .host_path(&socket)
        .user("postgres")
        .dbname(&db);
    service.session_pool = Some(
        deadpool_postgres::Pool::builder(deadpool_postgres::Manager::new(
            session_config,
            tokio_postgres::NoTls,
        ))
        .runtime(deadpool_postgres::Runtime::Tokio1)
        .max_size(2)
        .build()
        .unwrap(),
    );
    let service = Arc::new(service);
    let requested = vec![STAGING_SCOPE.into()];
    let mut rpc_source = source();
    rpc_source.jti = "rpc-jti".into();
    rpc_source.nonce = "rpc-nonce".into();
    rpc_source.token_hash = [8; 32];
    let created_at = chrono::Utc::now().timestamp();
    let prepared = service
        .prepare_rpc(
            &ctx,
            "client",
            rpc_source.clone(),
            false,
            requested.clone(),
            [10; 32],
            created_at,
            [44; 32],
            vec![55; 1952],
        )
        .await
        .unwrap();
    let parallel = service
        .prepare_rpc(
            &ctx,
            "client",
            rpc_source.clone(),
            false,
            requested.clone(),
            [11; 32],
            created_at,
            [44; 32],
            vec![55; 1952],
        )
        .await
        .unwrap();
    let committed = service
        .commit_rpc(
            &ctx,
            prepared.handle,
            "client",
            rpc_source.clone(),
            false,
            [10; 32],
            created_at,
            created_at + 60,
            [44; 32],
            vec![55; 1952],
        )
        .await
        .unwrap();
    assert!(Store::lookup_primary(
        &mut client,
        "https://host.test",
        &committed.sid,
        &serving_generation,
        &collision_inventory_id
    )
    .await
    .unwrap()
    .is_some());
    assert!(
        service
            .commit_rpc(
                &ctx,
                prepared.handle,
                "client",
                rpc_source.clone(),
                false,
                [10; 32],
                created_at,
                created_at + 60,
                [44; 32],
                vec![55; 1952]
            )
            .await
            .is_err(),
        "one-use handle must not commit twice"
    );
    assert!(
        service
            .commit_rpc(
                &ctx,
                parallel.handle,
                "client",
                rpc_source,
                false,
                [11; 32],
                created_at,
                created_at + 60,
                [44; 32],
                vec![55; 1952]
            )
            .await
            .is_err(),
        "PostgreSQL must reject the same source through a second handle"
    );
    let challenge = service
        .prepare(&ctx, "client", &source(), &requested)
        .await
        .unwrap();
    let mut stale = challenge.clone();
    stale.tenant = "other".into();
    assert!(service
        .redeem(&ctx, &mut client, evidence(stale, "mismatch"))
        .await
        .is_err());
    let mut stale = challenge.clone();
    stale.collision_inventory_id = [8; 32];
    assert!(service
        .redeem(&ctx, &mut client, evidence(stale, "inventory-mismatch"))
        .await
        .is_err());
    let mut stale = challenge.clone();
    stale.generation = [4; 32];
    assert!(service
        .redeem(&ctx, &mut client, evidence(stale, "generation-mismatch"))
        .await
        .is_err());
    let challenge_policy = Arc::clone(&service.authority.as_ref().unwrap().policy);
    assert!(challenge_policy
        .remove_policy_with_domain(
            ATPROTO_DID,
            "tenant",
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
            "allow",
        )
        .await
        .unwrap());
    assert!(service
        .redeem(&ctx, &mut client, evidence(challenge, "revision"))
        .await
        .is_err());
    challenge_policy
        .add_policy_with_domain(
            ATPROTO_DID,
            "tenant",
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
            "allow",
        )
        .await
        .unwrap();
    let challenge = service
        .prepare(&ctx, "client", &source(), &requested)
        .await
        .unwrap();
    // Hold the H1 row lock to causally place a grant revocation AFTER the
    // Policy read and BEFORE admission commit. This lock is fixture-only.
    let tx = control.transaction().await.unwrap();
    tx.query_one(
        "SELECT host FROM federate_session.profile_state FOR UPDATE",
        &[],
    )
    .await
    .unwrap();
    let pid: i32 = client
        .query_one("SELECT pg_backend_pid()", &[])
        .await
        .unwrap()
        .get(0);
    let observer = connect(&socket, &db).await;
    let race_policy = Arc::clone(&service.authority.as_ref().unwrap().policy);
    let mutate = async {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                let waiting: bool = observer
                    .query_one(
                        "SELECT EXISTS(SELECT 1 FROM pg_locks WHERE pid=$1 AND NOT granted)",
                        &[&pid],
                    )
                    .await
                    .unwrap()
                    .get(0);
                if waiting {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("admission reached PG after authority read");
        race_policy
            .remove_policy_with_domain(
                ATPROTO_DID,
                "tenant",
                &format!("model:{STAGING_MODEL_REF}"),
                "infer",
                "allow",
            )
            .await
            .unwrap();
        tx.commit().await.unwrap();
    };
    let (receipt, ()) = tokio::join!(
        service.redeem(
            &ctx,
            &mut client,
            evidence(challenge.clone(), "accepted-race")
        ),
        mutate
    );
    let receipt = receipt.expect("accepted read-to-commit race must not be rejected");
    assert!(service
        .lookup(&ctx, &client, &receipt.sid, &serving_generation)
        .await
        .unwrap()
        .is_some());
    // The NEXT authority read denies. This is not a Registry/Model dispatch test.
    assert!(service
        .prepare(&ctx, "client", &source(), &requested)
        .await
        .is_err());
    race_policy
        .add_policy_with_domain(
            ATPROTO_DID,
            "tenant",
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
            "allow",
        )
        .await
        .unwrap();
    let primary = Store::lookup_primary(
        &mut client,
        "https://host.test",
        &receipt.sid,
        &serving_generation,
        &collision_inventory_id,
    )
    .await
    .unwrap()
    .expect("active primary record includes its unique source binding");
    assert_eq!(
        primary.source_identity(),
        ("https://issuer.test", "source-user")
    );
    let s = primary.session();
    let ordered = [s.ed_public.to_vec(), s.pq_public.clone()];
    let mut request_use = AdmitFederateRequest {
        issuer: s.host.clone(),
        profile: hyprstream_session_store::PROFILE.into(),
        sid: s.sid.clone(),
        subject: s.subject.clone(),
        tenant: s.tenant.clone(),
        client: s.client_id.clone(),
        audience: s.resource.clone(),
        scopes: s.scopes.clone(),
        ed_public: s.ed_public.to_vec(),
        suite_thumbprint: signer_suite_thumbprint(
            hyprstream_session_store::SUITE,
            &[&s.ed_public, &s.pq_public],
        )
        .to_vec(),
        generation: s.generation.to_vec(),
        collision_inventory_id: s.collision_inventory_id.to_vec(),
        expires_at: s.expires_at,
        proof_epoch: s.proof_epoch as u64,
        verified_namespace: authenticated_replay_namespace(
            hyprstream_session_store::SUITE,
            &ordered,
            s.proof_epoch as u64,
        )
        .to_vec(),
        request_id: vec![11; 16],
        resource: format!("model:{STAGING_MODEL_REF}"),
        operation: "infer".into(),
    };
    let mut pool_config = tokio_postgres::Config::new();
    pool_config.host_path(&socket).user("postgres").dbname(&db);
    let pool = deadpool_postgres::Pool::builder(deadpool_postgres::Manager::new(
        pool_config,
        tokio_postgres::NoTls,
    ))
    .runtime(deadpool_postgres::Runtime::Tokio1)
    .max_size(2)
    .build()
    .unwrap();
    let reader = RequestUseReader {
        admission: Arc::clone(&service),
        pool: Some(pool),
        capacity: Semaphore::new(2),
        clock_skew_secs: 30,
    };
    let model_ctx = EnvelopeContext::for_test_authenticated_subject_with_claims(
        Subject::new("service:model"),
        "*",
        SigningKey::from_bytes(&[25; 32]).verifying_key(),
        hyprstream_rpc::auth::Claims::new("service:model".into(), 0, i64::MAX),
    );
    reader.admit(&model_ctx, &request_use).await.unwrap();
    assert!(
        reader.admit(&model_ctx, &request_use).await.is_err(),
        "same request ID must not dispatch twice"
    );
    request_use.request_id = vec![12; 16];
    request_use.subject = "other-account".into();
    assert!(reader.admit(&model_ctx, &request_use).await.is_err());
    request_use.subject = s.subject.clone();
    request_use.resource = "model:other:main".into();
    assert!(reader.admit(&model_ctx, &request_use).await.is_err());
    request_use.resource = format!("model:{STAGING_MODEL_REF}");
    service
        .authority
        .as_ref()
        .unwrap()
        .policy
        .add_policy_with_domain(
            ATPROTO_DID,
            "tenant",
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
            "deny",
        )
        .await
        .unwrap();
    assert!(reader.admit(&model_ctx, &request_use).await.is_err());
    assert!(service
        .authority
        .as_ref()
        .unwrap()
        .policy
        .remove_policy_with_domain(
            ATPROTO_DID,
            "tenant",
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
            "deny",
        )
        .await
        .unwrap());
    reader.admit(&model_ctx, &request_use).await.unwrap();
    service
        .authorize_use(
            &ctx,
            &mut client,
            &primary,
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
        )
        .await
        .unwrap();
    control
        .execute(
            "UPDATE federate_session.profile_state SET enabled=false WHERE host=$1 AND profile=$2",
            &[&"https://host.test", &hyprstream_session_store::PROFILE],
        )
        .await
        .unwrap();
    assert!(service
        .authorize_use(
            &ctx,
            &mut client,
            &primary,
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
        )
        .await
        .is_err());
    request_use.request_id = vec![13; 16];
    assert!(reader.admit(&model_ctx, &request_use).await.is_err());
    control
        .execute(
            "UPDATE federate_session.profile_state SET enabled=true WHERE host=$1 AND profile=$2",
            &[&"https://host.test", &hyprstream_session_store::PROFILE],
        )
        .await
        .unwrap();
    reader.admit(&model_ctx, &request_use).await.unwrap();
    service
        .authorize_use(
            &ctx,
            &mut client,
            &primary,
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
        )
        .await
        .unwrap();
    control
        .execute(
            "UPDATE federate_session.profile_state SET enabled=false WHERE host=$1 AND profile=$2",
            &[&"https://host.test", &hyprstream_session_store::PROFILE],
        )
        .await
        .unwrap();
    control
        .execute(
            "UPDATE federate_session.profile_state SET authority_generation=$3 WHERE host=$1 AND profile=$2",
            &[&"https://host.test", &hyprstream_session_store::PROFILE, &&[4u8; 32][..]],
        )
        .await
        .unwrap();
    assert!(service
        .authorize_use(
            &ctx,
            &mut client,
            &primary,
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
        )
        .await
        .is_err());
    control
        .execute(
            "UPDATE federate_session.profile_state SET authority_generation=$3 WHERE host=$1 AND profile=$2",
            &[&"https://host.test", &hyprstream_session_store::PROFILE, &&serving_generation[..]],
        )
        .await
        .unwrap();
    control
        .execute(
            "UPDATE federate_session.profile_state SET enabled=true WHERE host=$1 AND profile=$2",
            &[&"https://host.test", &hyprstream_session_store::PROFILE],
        )
        .await
        .unwrap();
    service
        .authorize_use(
            &ctx,
            &mut client,
            &primary,
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
        )
        .await
        .unwrap();
    let mut second = connect(&socket, &db).await;
    let (a, b) = tokio::join!(
        service.redeem(&ctx, &mut client, evidence(challenge.clone(), "replay-a")),
        service.redeem(&ctx, &mut second, evidence(challenge, "replay-b"))
    );
    assert!(a.is_err() && b.is_err());
    assert!(service
        .revoke(&ctx, &client, &receipt.sid, &serving_generation)
        .await
        .unwrap());
    request_use.request_id = vec![14; 16];
    assert!(reader.admit(&model_ctx, &request_use).await.is_err());
    // Reusing the same request-local primary after revocation must fail at
    // the next distinct use boundary; there is deliberately no chunk polling.
    assert!(service
        .authorize_use(
            &ctx,
            &mut client,
            &primary,
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
        )
        .await
        .is_err());
    assert!(service
        .lookup(&ctx, &client, &receipt.sid, &serving_generation)
        .await
        .unwrap()
        .is_none());
    assert!(Store::lookup_primary(
        &mut client,
        "https://host.test",
        &receipt.sid,
        &serving_generation,
        &collision_inventory_id
    )
    .await
    .unwrap()
    .is_none());
    assert!(service
        .lookup(&ctx, &client, &receipt.sid, &[4; 32])
        .await
        .is_err());
    Store::cleanup(&mut client).await.unwrap();
    observer
        .execute("SELECT pg_terminate_backend($1)", &[&pid])
        .await
        .unwrap();
    assert!(service
        .lookup(&ctx, &client, &receipt.sid, &serving_generation)
        .await
        .is_err());
    drop(client);
    drop(second);
    drop(observer);
    drop(control);
    admin
        .batch_execute(&format!("DROP DATABASE {db} WITH (FORCE)"))
        .await
        .unwrap();
}
