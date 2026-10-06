#![allow(clippy::unwrap_used, clippy::expect_used)]
use super::*;
use crate::auth::user_store::*;
use ed25519_dalek::{SigningKey, VerifyingKey};
use hyprstream_pds_service::{AccountRecordReadAuthorizer, AccountRecordStore};
use hyprstream_rpc::auth::mac::{MacDecision, SecurityContext};
use hyprstream_vfs::{SyntheticMount, SyntheticNode};
use parking_lot::RwLock;

const STAGING_MODEL_REF: &str = "qwen2.5-0.5b-instruct:main";
const STAGING_SCOPE: &str = "infer:model:qwen2.5-0.5b-instruct:main";

struct Accounts(RwLock<UserProfile>);
#[async_trait::async_trait]
impl UserStore for Accounts {
    async fn get_profile(&self, _: &str) -> Result<Option<UserProfile>> {
        Ok(Some(self.0.read().clone()))
    }
    async fn get_external_identity_user(&self, issuer: &str, sub: &str) -> Result<Option<String>> {
        Ok((issuer == "https://issuer.test" && sub == "source-user").then(|| "alice".into()))
    }
    async fn register(&self, _: &str) -> Result<String> {
        anyhow::bail!("forbidden")
    }
    async fn set_profile(&self, _: &str, _: UserProfilePatch) -> Result<()> {
        anyhow::bail!("forbidden")
    }
    async fn remove(&self, _: &str) -> Result<bool> {
        anyhow::bail!("forbidden")
    }
    async fn list_users(&self) -> Result<Vec<String>> {
        anyhow::bail!("forbidden")
    }
    async fn search(&self, _: &UserFilter) -> Result<Vec<(String, UserProfile)>> {
        anyhow::bail!("forbidden")
    }
    async fn set_active(&self, _: &str, _: bool) -> Result<()> {
        anyhow::bail!("forbidden")
    }
    async fn list_pubkeys(&self, _: &str) -> Result<Vec<PubkeyEntry>> {
        anyhow::bail!("forbidden")
    }
    async fn add_pubkey(&self, _: &str, _: VerifyingKey, _: Option<String>) -> Result<String> {
        anyhow::bail!("forbidden")
    }
    async fn add_pubkey_hybrid(
        &self,
        _: &str,
        _: VerifyingKey,
        _: Vec<u8>,
        _: Option<String>,
    ) -> Result<String> {
        anyhow::bail!("forbidden")
    }
    async fn remove_pubkey(&self, _: &str, _: &str) -> Result<bool> {
        anyhow::bail!("forbidden")
    }
    async fn get_pubkey_user(&self, _: &str) -> Result<Option<String>> {
        anyhow::bail!("forbidden")
    }
    async fn touch_pubkey(&self, _: &str, _: &str) -> Result<()> {
        anyhow::bail!("forbidden")
    }
}
struct Permit;
impl AccountRecordReadAuthorizer for Permit {
    fn check_read(
        &self,
        _: &Subject,
        _: Option<&str>,
        _: Option<&SecurityContext>,
        _: &str,
    ) -> MacDecision {
        MacDecision::Permit
    }
}
fn source() -> Source {
    let now = chrono::Utc::now().timestamp();
    Source {
        issuer: "https://issuer.test".into(),
        subject: "source-user".into(),
        jti: "jti".into(),
        nonce: "nonce".into(),
        token_hash: [1; 32],
        issued_at: now,
        expires_at: now + 120,
    }
}
fn record() -> Vec<u8> {
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes};
    use hyprstream_pds::did_op::{
        sign_genesis, GenesisRepoHead, GenesisRotationKeys, HostKeyEnrollment, HybridRotationKey,
        RecoveryKeyEnrollment, UserRotationKey,
    };
    use hyprstream_pds::{AllocatedAccountName, HostedAccountMint};
    let ed = SigningKey::from_bytes(&[22; 32]);
    let (pq, vk) = ml_dsa_generate_keypair();
    let rotations = GenesisRotationKeys::new(
        UserRotationKey::new(
            HybridRotationKey::new(ed.verifying_key().to_bytes(), ml_dsa_vk_bytes(&vk)).unwrap(),
        ),
        RecoveryKeyEnrollment::Declined,
        HostKeyEnrollment::Absent,
    )
    .unwrap();
    let mint = HostedAccountMint::begin(
        AllocatedAccountName::new("alice", "did:web:alice.example.test").unwrap(),
        rotations,
    )
    .unwrap();
    let doc = mint.seal_did_document("https://pds.test").unwrap();
    let pending = mint
        .prepare_genesis(doc, GenesisRepoHead::EmptyRepo)
        .unwrap();
    let signature = sign_genesis(pending.unsigned_genesis(), &ed, &pq).unwrap();
    pending.seal(signature).unwrap().record_bytes().to_vec()
}
async fn fixture() -> (AdmissionService, Arc<Accounts>, EnvelopeContext) {
    let users = Arc::new(Accounts(RwLock::new(UserProfile {
        sub: Some(uuid::Uuid::new_v4().to_string()),
        active: Some(true),
        atproto_did: Some("did:web:alice.example.test".into()),
        ..Default::default()
    })));
    let root = SyntheticNode::dir().with_child(
        "tenant",
        SyntheticNode::dir().with_child(
            "accounts",
            SyntheticNode::dir().with_child(
                "alice",
                SyntheticNode::dir().with_child(
                    hyprstream_pds_service::PDS_ACCOUNT_RECORD_FILE,
                    SyntheticNode::file(record()),
                ),
            ),
        ),
    );
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
    let scopes = BTreeSet::from([STAGING_SCOPE.into()]);
    let policy = Arc::new(PolicyManager::new_in_memory().await.unwrap());
    policy
        .add_policy_with_domain(
            "alice",
            "tenant",
            &format!("model:{STAGING_MODEL_REF}"),
            "infer",
            "allow",
        )
        .await
        .unwrap();
    let a = Authorities {
        users: ProductionUserStore::for_test(users.clone()),
        accounts: Arc::new(AccountRecordStore::new(
            Arc::new(SyntheticMount::new(root)),
            Arc::new(Permit),
        )),
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
        serving_generation: [3; 32],
        local_collision_inventory_id: [7; 32],
    };
    (
        AdmissionService {
            authority: Some(a),
            capacity: Some(Semaphore::new(1)),
        },
        users,
        EnvelopeContext::for_test_authenticated_subject_with_claims(
            Subject::new("service:oauth"),
            "*",
            signer,
            hyprstream_rpc::auth::Claims::new("service:oauth".into(), 0, i64::MAX),
        ),
    )
}

#[tokio::test]
async fn h2_request_use_requires_exact_enrolled_serving_holder() {
    let (service, _, _) = fixture().await;
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
async fn h2_admission_disabled_caller_and_scope_boundaries() {
    let (service, _, ctx) = fixture().await;
    let requested = vec![STAGING_SCOPE.into(), "read:model:other".into()];
    assert!(AdmissionService::default()
        .prepare(&ctx, &source(), &requested)
        .await
        .is_err());
    let decision = service.prepare(&ctx, &source(), &requested).await.unwrap();
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
            .prepare(&wrong, &source(), &requested)
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
        .prepare(&wrong, &source(), &requested)
        .await
        .is_err());
    assert!(service
        .prepare(&ctx, &source(), &["read:model:other".into()])
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
                .prepare(&ctx, &source(), &[nonmatching.into()])
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
    let (service, _, ctx) = fixture().await;
    let requested = vec![STAGING_SCOPE.into()];
    canonical_requested(&requested).unwrap();
    let parsed = Scope::parse(STAGING_SCOPE).unwrap();
    let dispatch_resource = format!("model:{STAGING_MODEL_REF}");
    assert_eq!(parsed.policy_resource(), dispatch_resource);

    let decision = service.prepare(&ctx, &source(), &requested).await.unwrap();
    assert_eq!(decision.scopes, [STAGING_SCOPE]);
    assert_eq!(decision.scopes[0], STAGING_SCOPE);
}

#[tokio::test]
async fn h2_admission_fresh_account_tenant_revision_and_capacity() {
    let (service, users, ctx) = fixture().await;
    let requested = vec![STAGING_SCOPE.into()];
    let first = service.prepare(&ctx, &source(), &requested).await.unwrap();
    let again = service.prepare(&ctx, &source(), &requested).await.unwrap();
    assert!(first == again);
    users.0.write().active = Some(false);
    assert!(service.prepare(&ctx, &source(), &requested).await.is_err());
    users.0.write().active = Some(true);
    users.0.write().sub = Some(uuid::Uuid::new_v4().to_string());
    assert!(first != service.prepare(&ctx, &source(), &requested).await.unwrap());
    users.0.write().atproto_did = Some("did:web:missing.example.test".into());
    assert!(service.prepare(&ctx, &source(), &requested).await.is_err());
    let _permit = service.capacity.as_ref().unwrap().try_acquire().unwrap();
    assert!(service.prepare(&ctx, &source(), &requested).await.is_err());
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
    let (service, users, ctx) = fixture().await;
    let source_record = source();
    let decision = service
        .prepare(&ctx, &source_record, &[STAGING_SCOPE.into()])
        .await
        .unwrap();
    let session = session(&decision);
    let authority = service.authority.as_ref().unwrap();
    let resource = format!("model:{STAGING_MODEL_REF}");

    authority
        .authorize_use(
            &session,
            &source_record.issuer,
            &source_record.subject,
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
            &source_record.subject,
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
            &source_record.subject,
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
            &source_record.subject,
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
            &source_record.subject,
            &resource,
            "infer",
            chrono::Utc::now().timestamp()
        )
        .await
        .is_err());
    let mut wrong_source = source();
    wrong_source.subject = "different-source-user".into();
    assert!(authority
        .authorize_use(
            &session,
            &wrong_source.issuer,
            &wrong_source.subject,
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
            &source_record.subject,
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
        .remove_policy_with_domain("alice", "tenant", &resource, "infer", "allow")
        .await
        .unwrap();
    assert!(removed);
    assert!(
        authority
            .authorize_use(
                &session,
                &source_record.issuer,
                &source_record.subject,
                &resource,
                "infer",
                chrono::Utc::now().timestamp()
            )
            .await
            .is_err(),
        "next distinct boundary must observe revoke"
    );

    // Account suspension and tenant remapping also take effect on the next
    // fresh check, without depending on session/token refresh.
    authority
        .policy
        .add_policy_with_domain("alice", "tenant", &resource, "infer", "allow")
        .await
        .unwrap();
    users.0.write().active = Some(false);
    assert!(authority
        .authorize_use(
            &session,
            &source_record.issuer,
            &source_record.subject,
            &resource,
            "infer",
            chrono::Utc::now().timestamp()
        )
        .await
        .is_err());
    users.0.write().active = Some(true);
    users.0.write().atproto_did = Some("did:web:missing.example.test".into());
    assert!(authority
        .authorize_use(
            &session,
            &source_record.issuer,
            &source_record.subject,
            &resource,
            "infer",
            chrono::Utc::now().timestamp()
        )
        .await
        .is_err());
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
    control.execute("INSERT INTO federate_session.profile_state(host,profile,enabled,authority_generation,collision_inventory_id) VALUES ($1,$2,true,$3,$4)",
        &[&"https://host.test", &hyprstream_session_store::PROFILE, &&[3u8;32][..], &&[7u8;32][..]]).await.unwrap();
    let mut client = connect(&socket, &db).await;
    let (service, users, ctx) = fixture().await;
    let service = Arc::new(service);
    let requested = vec![STAGING_SCOPE.into()];
    let challenge = service.prepare(&ctx, &source(), &requested).await.unwrap();
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
    users.0.write().sub = Some(uuid::Uuid::new_v4().to_string());
    assert!(service
        .redeem(&ctx, &mut client, evidence(challenge, "revision"))
        .await
        .is_err());
    let challenge = service.prepare(&ctx, &source(), &requested).await.unwrap();
    // Hold the H1 row lock to causally place suspension AFTER the Policy read
    // and BEFORE admission commit. This lock is fixture instrumentation only.
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
        users.0.write().active = Some(false);
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
        .lookup(&ctx, &client, &receipt.sid, &[3; 32])
        .await
        .unwrap()
        .is_some());
    // The NEXT authority read denies. This is not a Registry/Model dispatch test.
    assert!(service.prepare(&ctx, &source(), &requested).await.is_err());
    users.0.write().active = Some(true);
    let primary = Store::lookup_primary(
        &mut client,
        "https://host.test",
        &receipt.sid,
        &[3; 32],
        &[7; 32],
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
        pool,
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
    users.0.write().active = Some(false);
    assert!(reader.admit(&model_ctx, &request_use).await.is_err());
    users.0.write().active = Some(true);
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
            &[&"https://host.test", &hyprstream_session_store::PROFILE, &&[3u8; 32][..]],
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
        .revoke(&ctx, &client, &receipt.sid, &[3; 32])
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
        .lookup(&ctx, &client, &receipt.sid, &[3; 32])
        .await
        .unwrap()
        .is_none());
    assert!(Store::lookup_primary(
        &mut client,
        "https://host.test",
        &receipt.sid,
        &[3; 32],
        &[7; 32]
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
        .lookup(&ctx, &client, &receipt.sid, &[3; 32])
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
