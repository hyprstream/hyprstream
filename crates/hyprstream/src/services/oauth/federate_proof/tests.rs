#![allow(clippy::unwrap_used, clippy::expect_used)]

use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use hyprstream_rpc::{
    crypto::pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes},
    proof::{
        build::{
            build_authenticated_hybrid_request_proof, AuthenticatedHybridProofSigner,
            AuthenticatedRequestProofInput,
        },
        enrollment::{authenticated_replay_namespace, InMemoryEnrollmentResolver},
        recipient_binding::FederateRecipientBinding,
    },
    crypto::hybrid_kem::{generate_recipient, SuiteId},
};
use hyprstream_session_store::primary::InventorySource;

use super::*;

const NOW: u64 = 1_800_000_000;
const TOKEN: &[u8] = b"fixture-only.verified-host-jwt.original-holder";
const BODY: &[u8] = b"fixture canonical request bytes, not a dispatch";
const SCHEMA: u64 = 0xd4d0_f2a1_b3c5_8e67;

fn host_claims() -> Claims {
    let ed = ed25519_dalek::SigningKey::from_bytes(&[41; 32])
        .verifying_key()
        .to_bytes();
    let pq = ml_dsa_sk_to_vk_bytes(&ml_dsa_sk_from_seed(&[42; 32]));
    let mut claims = Claims::new("local-account".into(), NOW as i64, NOW as i64 + 60)
        .with_issuer(HOST.into())
        .with_audience(Some(HOST.into()))
        .with_client_id(CLIENT)
        .with_tenant("tenant-a".into())
        .with_sid("sid-a")
        .with_scope(Some("query:registry:List".into()))
        .with_cnf_jwk(&ed)
        .with_federate_signer_suite(
            URL_SAFE_NO_PAD.encode(signer_suite_thumbprint(SUITE, &[&ed, &pq])),
        )
        .with_session_authority_generation([7; 32]);
    claims.jti = Some("jti-a".into());
    claims
}

#[test]
fn h3b_verified_host_snapshot_requires_complete_exact_signed_facts() {
    let valid = host_claims();
    let handle = CredentialHandle::from_claims_snapshot(&valid, "fixture-token").unwrap();
    assert_eq!(handle.expected.sid, "sid-a");
    assert_eq!(handle.expected.generation, [7; 32]);
    assert_eq!(handle.expected.scopes, ["query:registry:List"]);
    let token_hash: [u8; 32] = Sha256::digest(b"fixture-token").into();
    assert_eq!(handle.credential_hash, token_hash);

    let mut bad = valid.clone();
    bad.hs_session_authority_generation = None;
    assert!(CredentialHandle::from_claims_snapshot(&bad, "fixture-token").is_err());
    let mut bad = valid.clone();
    bad.iss = "https://other.example".into();
    assert!(CredentialHandle::from_claims_snapshot(&bad, "fixture-token").is_err());
    let mut bad = valid.clone();
    bad.aud = Some("registry".into());
    assert!(CredentialHandle::from_claims_snapshot(&bad, "fixture-token").is_err());
    let mut bad = valid.clone();
    bad.client_id = Some("other".into());
    assert!(CredentialHandle::from_claims_snapshot(&bad, "fixture-token").is_err());
    let mut bad = valid.clone();
    bad.scope = Some("query:registry:List query:registry:List".into());
    assert!(CredentialHandle::from_claims_snapshot(&bad, "fixture-token").is_err());
    let mut bad = valid.clone();
    bad.scope = Some("infer:model:qwen2.5-0.5b-instruct:main:extra".into());
    assert!(CredentialHandle::from_claims_snapshot(&bad, "fixture-token").is_err());
    let mut bad = valid.clone();
    bad.cnf.as_mut().unwrap().jkt = Some("alternate-holder".into());
    assert!(CredentialHandle::from_claims_snapshot(&bad, "fixture-token").is_err());
    let mut bad = valid.clone();
    bad.hs_profile = None;
    assert!(CredentialHandle::from_claims_snapshot(&bad, "fixture-token").is_err());
    let mut bad = valid.clone();
    bad.hs_signer_suite_v1 = None;
    assert!(CredentialHandle::from_claims_snapshot(&bad, "fixture-token").is_err());
    let mut bad = valid.clone();
    bad.cnf.as_mut().unwrap().hs_signer_suite = bad.hs_signer_suite_v1.take();
    assert!(CredentialHandle::from_claims_snapshot(&bad, "fixture-token").is_err());
    let mut bad = valid.clone();
    bad.workload_session_id = Some("other-session".into());
    assert!(CredentialHandle::from_claims_snapshot(&bad, "fixture-token").is_err());
    let callback = EnvelopeContext::from_callback_service(1, "oauth");
    assert!(CredentialHandle::from_verified_context(&callback).is_err());
}

struct Provider {
    record: Mutex<SessionPrimary>,
    calls: AtomicUsize,
    deny_sid: Mutex<Option<String>>,
    delay: Duration,
}

#[async_trait::async_trait]
impl CurrentPrimary for Provider {
    async fn resolve_current(&self, h: &CredentialHandle) -> Result<SessionPrimary> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }
        if self.deny_sid.lock().as_ref() == Some(&h.expected.sid) {
            return Err(Error::Denied);
        }
        Ok(self.record.lock().clone())
    }
}

fn inventory(keys: Vec<Vec<u8>>) -> CollisionInventory {
    CollisionInventory::from_sources(
        &["fixture:complete".into()],
        vec![InventorySource {
            id: "fixture:complete".into(),
            keys,
        }],
    )
    .unwrap()
}

fn fixture() -> (CredentialHandle, SessionPrimary, CollisionInventory) {
    let ed = ed25519_dalek::SigningKey::from_bytes(&[41; 32])
        .verifying_key()
        .to_bytes();
    let pq = ml_dsa_sk_to_vk_bytes(&ml_dsa_sk_from_seed(&[42; 32]));
    let inventory = inventory(vec![vec![91; 32], vec![92; 1952]]);
    let expected = ExpectedPrimary {
        issuer: HOST.into(),
        profile: PROFILE.into(),
        sid: "sid-a".into(),
        subject: "local-account".into(),
        tenant: "tenant-a".into(),
        client: CLIENT.into(),
        audience: HOST.into(),
        scopes: vec!["model:query".into()],
        ed_public: ed,
        suite_thumbprint: signer_suite_thumbprint(SUITE, &[&ed, &pq]),
        generation: [7; 32],
        expires_at: (NOW + 60) as i64,
    };
    let record = SessionPrimary {
        host: HOST.into(),
        profile: PROFILE.into(),
        suite: SUITE.into(),
        sid: expected.sid.clone(),
        account_id: "account-a".into(),
        subject: expected.subject.clone(),
        tenant: expected.tenant.clone(),
        client: CLIENT.into(),
        resource: HOST.into(),
        scopes: expected.scopes.clone(),
        grant_revision: "revision-a".into(),
        ed_public: ed.to_vec(),
        pq_public: pq,
        generation: expected.generation.to_vec(),
        created_at: NOW as i64 - 1,
        expires_at: expected.expires_at,
        proof_epoch: 37,
        collision_inventory_id: inventory.id().to_vec(),
    };
    (
        CredentialHandle {
            expected,
            credential_id: "jti-a".into(),
            credential_hash: Sha256::digest(TOKEN).into(),
        },
        record,
        inventory,
    )
}

#[test]
fn h3b_policy_lookup_request_is_exact_credential_tuple() {
    let (handle, _, _) = fixture();
    let request = PolicyPrimaryProvider::request(&handle);
    let e = &handle.expected;
    assert_eq!(request.issuer, e.issuer);
    assert_eq!(request.profile, e.profile);
    assert_eq!(request.sid, e.sid);
    assert_eq!(request.subject, e.subject);
    assert_eq!(request.tenant, e.tenant);
    assert_eq!(request.client, e.client);
    assert_eq!(request.audience, e.audience);
    assert_eq!(request.scopes, e.scopes);
    assert_eq!(request.ed_public, e.ed_public);
    assert_eq!(request.suite_thumbprint, e.suite_thumbprint);
    assert_eq!(request.generation, e.generation);
    assert_eq!(request.expires_at, e.expires_at);
}

fn provider(record: SessionPrimary) -> Arc<Provider> {
    Arc::new(Provider {
        record: Mutex::new(record),
        calls: AtomicUsize::new(0),
        deny_sid: Mutex::new(None),
        delay: Duration::ZERO,
    })
}

fn proof(ed_seed: u8, pq_seed: u8, credential: &[u8]) -> Vec<u8> {
    let no_forwarded_recipients =
        FederateRecipientBinding::from_recipients(None, None, None).unwrap();
    proof_with_binding(ed_seed, pq_seed, credential, Some(&no_forwarded_recipients))
}

fn proof_with_binding(
    ed_seed: u8,
    pq_seed: u8,
    credential: &[u8],
    binding: Option<&FederateRecipientBinding>,
) -> Vec<u8> {
    let ed = ed25519_dalek::SigningKey::from_bytes(&[ed_seed; 32]);
    let pq = ml_dsa_sk_from_seed(&[pq_seed; 32]);
    let kids = session_primary_kids(&ed.verifying_key().to_bytes(), &ml_dsa_sk_to_vk_bytes(&pq));
    let signer = AuthenticatedHybridProofSigner::new(ed, kids[0], pq, kids[1]).unwrap();
    build_authenticated_hybrid_request_proof(
        &AuthenticatedRequestProofInput {
            service_domain: "registry",
            credential,
            issued_at: NOW,
            expires_at: NOW + 20,
            capnp_schema_id: SCHEMA,
            capnp_body: BODY,
            response_binding: None,
            federate_recipient_binding: binding,
        },
        &signer,
    )
    .unwrap()
}

fn proof_with_deferred_recipient_extension(credential: &[u8]) -> Vec<u8> {
    let ed = ed25519_dalek::SigningKey::from_bytes(&[41; 32]);
    let pq = ml_dsa_sk_from_seed(&[42; 32]);
    let kids = session_primary_kids(&ed.verifying_key().to_bytes(), &ml_dsa_sk_to_vk_bytes(&pq));
    let signer = AuthenticatedHybridProofSigner::new(ed, kids[0], pq, kids[1]).unwrap();
    let response = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
    let stream = generate_recipient(SuiteId::HyKemX25519MlKem768).unwrap();
    let response_public = response.public();
    let stream_public = stream.public();
    let binding = FederateRecipientBinding::from_recipients(
        Some(&response_public),
        Some(&stream_public),
        Some([0x5a; 32]),
    )
    .unwrap();
    build_authenticated_hybrid_request_proof(
        &AuthenticatedRequestProofInput {
            service_domain: "registry",
            credential,
            issued_at: NOW,
            expires_at: NOW + 20,
            capnp_schema_id: SCHEMA,
            capnp_body: BODY,
            response_binding: None,
            federate_recipient_binding: Some(&binding),
        },
        &signer,
    )
    .unwrap()
}

fn request(proof: &[u8]) -> Request<'_> {
    Request {
        proof,
        credential: TOKEN,
        service: "registry",
        schema_id: SCHEMA,
        body: BODY,
    }
}

#[tokio::test]
async fn h3b_consumer_accepts_only_deferred_parser_recipient_extension() {
    let (handle, record, inventory) = fixture();
    let bytes = proof_with_deferred_recipient_extension(TOKEN);
    assert!(
        hyprstream_rpc::proof::parser::ParsedProof::parse(&bytes).is_err(),
        "generic v16 parser must remain closed to the source extension"
    );
    let static_enrollments = InMemoryEnrollmentResolver::new();
    let permits = Semaphore::new(1);
    let policy = LocalPolicy {
        inventory: &inventory,
        static_enrollments: &static_enrollments,
        in_flight: &permits,
    };
    let consumer = Consumer {
        provider: Some(provider(record)),
    };

    consumer
        .verify(&handle, &request(&bytes), &policy, || NOW)
        .await
        .expect("H3b Consumer uses the narrow deferred-Federate parser");
    let no_binding = proof_with_binding(41, 42, TOKEN, None);
    assert_eq!(
        consumer
            .verify(&handle, &request(&no_binding), &policy, || NOW)
            .await
            .unwrap_err(),
        Error::Denied,
        "a deferred Federate proof without -70009 must deny"
    );
}

#[tokio::test]
async fn h3b_disabled_without_provider_and_fresh_lookup_on_every_reuse() {
    let (handle, record, inventory) = fixture();
    let static_enrollments = InMemoryEnrollmentResolver::new();
    let permits = Semaphore::new(16);
    let policy = LocalPolicy {
        inventory: &inventory,
        static_enrollments: &static_enrollments,
        in_flight: &permits,
    };
    let bytes = proof(41, 42, TOKEN);
    assert_eq!(
        Consumer::default()
            .verify(&handle, &request(&bytes), &policy, || NOW)
            .await
            .unwrap_err(),
        Error::Disabled
    );
    let expected_replay_namespace = authenticated_replay_namespace(
        SUITE,
        &[record.ed_public.clone(), record.pq_public.clone()],
        record.proof_epoch,
    );
    let provider = provider(record);
    let consumer = Consumer {
        provider: Some(provider.clone()),
    };
    let first = consumer
        .verify(&handle, &request(&bytes), &policy, || NOW)
        .await
        .unwrap();
    let second = consumer
        .verify(&handle, &request(&bytes), &policy, || NOW)
        .await
        .unwrap();
    // Crypto success is deliberately NOT replay admission; duplicate proof still
    // verifies here. F3 must be tested at the later dispatch/replay boundary.
    assert_eq!(first.replay_thumbprint, second.replay_thumbprint);
    assert_eq!(first.replay_thumbprint, expected_replay_namespace);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    // The authority await must not retain the initial verifier timestamp.
    let clock_reads = AtomicUsize::new(0);
    assert!(consumer
        .verify(&handle, &request(&bytes), &policy, || {
            if clock_reads.fetch_add(1, Ordering::SeqCst) == 0 {
                NOW
            } else {
                NOW + 21
            }
        })
        .await
        .is_err());
    assert_eq!(clock_reads.load(Ordering::SeqCst), 2);
    *provider.deny_sid.lock() = Some(handle.expected.sid.clone());
    assert_eq!(
        consumer
            .verify(&handle, &request(&bytes), &policy, || NOW)
            .await
            .unwrap_err(),
        Error::Denied
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn h3b_each_authority_binding_and_epoch_is_checked() {
    let (handle, base, inventory) = fixture();
    let static_enrollments = InMemoryEnrollmentResolver::new();
    let permits = Semaphore::new(16);
    let policy = LocalPolicy {
        inventory: &inventory,
        static_enrollments: &static_enrollments,
        in_flight: &permits,
    };
    let bytes = proof(41, 42, TOKEN);
    let valid = Consumer {
        provider: Some(provider(base.clone())),
    };
    valid
        .verify(&handle, &request(&bytes), &policy, || NOW)
        .await
        .expect("unmodified primary must admit before testing mutations");
    let mutations: &[fn(&mut SessionPrimary)] = &[
        |r| r.host.push('x'),
        |r| r.profile.push('x'),
        |r| r.suite.push('x'),
        |r| r.sid.push('x'),
        |r| r.subject.push('x'),
        |r| r.tenant.push('x'),
        |r| r.client.push('x'),
        |r| r.resource.push('x'),
        |r| r.scopes.push("worker:write".into()),
        |r| r.ed_public[0] ^= 1,
        |r| r.pq_public[0] ^= 1,
        |r| r.generation[0] ^= 1,
        |r| r.expires_at -= 1,
        |r| r.created_at = NOW as i64 + 1,
        |r| r.proof_epoch = 0,
        |r| r.proof_epoch = u64::MAX,
        |r| r.collision_inventory_id[0] ^= 1,
        |r| r.account_id.clear(),
        |r| r.grant_revision.clear(),
    ];
    for (index, mutate) in mutations.iter().enumerate() {
        let mut altered = base.clone();
        mutate(&mut altered);
        let consumer = Consumer {
            provider: Some(provider(altered)),
        };
        assert!(
            consumer
                .verify(&handle, &request(&bytes), &policy, || NOW)
                .await
                .is_err(),
            "mutation {index}"
        );
    }
}

#[tokio::test]
async fn h3b_holder_components_and_exact_credential_body_schema_service() {
    let (handle, record, inventory) = fixture();
    let static_enrollments = InMemoryEnrollmentResolver::new();
    let permits = Semaphore::new(16);
    let policy = LocalPolicy {
        inventory: &inventory,
        static_enrollments: &static_enrollments,
        in_flight: &permits,
    };
    let consumer = Consumer {
        provider: Some(provider(record)),
    };
    let valid_bytes = proof(41, 42, TOKEN);
    consumer
        .verify(&handle, &request(&valid_bytes), &policy, || NOW)
        .await
        .expect("unmodified holder proof must admit before testing substitutions");
    for bytes in [
        proof(51, 42, TOKEN),
        proof(41, 52, TOKEN),
        proof(51, 52, TOKEN),
        proof(41, 42, b"substituted-token"),
    ] {
        assert!(consumer
            .verify(&handle, &request(&bytes), &policy, || NOW)
            .await
            .is_err());
    }
    let bytes = proof(41, 42, TOKEN);
    for r in [
        Request {
            body: b"different method bytes",
            ..request(&bytes)
        },
        Request {
            schema_id: 1,
            ..request(&bytes)
        },
        Request {
            service: "model",
            ..request(&bytes)
        },
        Request {
            credential: b"relay-token",
            ..request(&bytes)
        },
    ] {
        assert!(consumer.verify(&handle, &r, &policy, || NOW).await.is_err());
    }
    assert!(consumer
        .verify(&handle, &request(&bytes), &policy, || NOW + 61)
        .await
        .is_err());
    let mut corrupted = bytes.clone();
    let last = corrupted.len() - 1;
    corrupted[last] ^= 1;
    assert!(consumer
        .verify(&handle, &request(&corrupted), &policy, || NOW)
        .await
        .is_err());
    assert!(consumer
        .verify(
            &handle,
            &request(&bytes[..bytes.len() - 1]),
            &policy,
            || NOW
        )
        .await
        .is_err());
}

#[tokio::test]
async fn h3b_deadline_and_capacity_deny_without_fallback() {
    let (handle, record, inventory) = fixture();
    let static_enrollments = InMemoryEnrollmentResolver::new();
    let permits = Semaphore::new(1);
    let policy = LocalPolicy {
        inventory: &inventory,
        static_enrollments: &static_enrollments,
        in_flight: &permits,
    };
    let bytes = proof(41, 42, TOKEN);
    assert!(
        hyprstream_rpc::proof::parser::ParsedProof::parse_deferred_federate_request(&bytes)
            .is_ok(),
        "capacity and deadline test requires a valid deferred proof"
    );
    let provider = Arc::new(Provider {
        record: Mutex::new(record),
        calls: AtomicUsize::new(0),
        deny_sid: Mutex::new(None),
        delay: Duration::from_secs(60),
    });
    let consumer = Consumer {
        provider: Some(provider.clone()),
    };
    let held = permits.acquire().await.unwrap();
    assert_eq!(
        consumer
            .verify(&handle, &request(&bytes), &policy, || NOW)
            .await
            .unwrap_err(),
        Error::Unavailable
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    drop(held);
    assert_eq!(
        consumer
            .verify(&handle, &request(&bytes), &policy, || NOW)
            .await
            .unwrap_err(),
        Error::Unavailable
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    assert_eq!(permits.available_permits(), 1);
}

#[test]
fn h3b_request_local_resolver_never_falls_back_to_static_primary() {
    let (handle, record, inventory) = fixture();
    let static_key = ed25519_dalek::SigningKey::from_bytes(&[71; 32]).verifying_key();
    let mut statics = InMemoryEnrollmentResolver::new();
    let permits = Semaphore::new(16);
    let policy = LocalPolicy {
        inventory: &inventory,
        static_enrollments: &statics,
        in_flight: &permits,
    };
    let mut static_record = local_resolver(&handle.expected, &record, &policy)
        .unwrap()
        .primary;
    static_record.components[0] =
        EnrolledComponent::new(b"static-kid".to_vec(), ComponentKey::Ed25519(static_key));
    statics
        .enrol_primary(&static_key, static_record.clone())
        .unwrap();
    static_record.role = SignerRole::Approver;
    statics.enrol_approver(static_record.clone()).unwrap();
    static_record.role = SignerRole::Service;
    statics.enrol_service("registry", static_record).unwrap();
    let policy = LocalPolicy {
        inventory: &inventory,
        static_enrollments: &statics,
        in_flight: &permits,
    };
    let resolver = local_resolver(&handle.expected, &record, &policy).unwrap();
    assert!(resolver.resolve_primary(&static_key).is_none());
    assert!(statics.resolve_primary(&static_key).is_some());
    assert!(resolver.resolve_approver(b"static-kid").is_some());
    assert!(resolver.resolve_service("registry").is_some());
    let mut next_epoch = record.clone();
    next_epoch.proof_epoch += 1;
    let next = local_resolver(&handle.expected, &next_epoch, &policy).unwrap();
    assert_ne!(
        resolver.primary.replay_thumbprint(),
        next.primary.replay_thumbprint()
    );
    // Inventory equality alone cannot permit a component collision.
    let collision = inventory_with_key(&handle.expected.ed_public);
    let mut record = record;
    record.collision_inventory_id = collision.id().to_vec();
    let policy = LocalPolicy {
        inventory: &collision,
        static_enrollments: &statics,
        in_flight: &permits,
    };
    assert!(local_resolver(&handle.expected, &record, &policy).is_err());
}

fn inventory_with_key(key: &[u8; 32]) -> CollisionInventory {
    inventory(vec![key.to_vec()])
}

#[tokio::test]
async fn h3b_two_handles_for_one_subject_do_not_share_positive_authority() {
    let (a, record_a, inventory) = fixture();
    let (mut b, mut record_b, _) = fixture();
    b.expected.sid = "sid-b".into();
    b.credential_id = "jti-b".into();
    b.expected.ed_public = ed25519_dalek::SigningKey::from_bytes(&[51; 32])
        .verifying_key()
        .to_bytes();
    record_b.ed_public = b.expected.ed_public.to_vec();
    record_b.pq_public = ml_dsa_sk_to_vk_bytes(&ml_dsa_sk_from_seed(&[52; 32]));
    b.expected.suite_thumbprint =
        signer_suite_thumbprint(SUITE, &[&record_b.ed_public, &record_b.pq_public]);
    record_b.sid = b.expected.sid.clone();
    record_b.proof_epoch += 1;
    // In this primitive the credential is opaque; JWT parsing/verification is
    // not exercised. Separate exact bytes still bind each fixture handle.
    let token_b = b"fixture-only.second-host-jwt";
    b.credential_hash = Sha256::digest(token_b).into();
    let static_enrollments = InMemoryEnrollmentResolver::new();
    let permits = Semaphore::new(16);
    let policy = LocalPolicy {
        inventory: &inventory,
        static_enrollments: &static_enrollments,
        in_flight: &permits,
    };
    let provider = provider(record_a);
    let consumer = Consumer {
        provider: Some(provider.clone()),
    };
    let proof_a = proof(41, 42, TOKEN);
    let proof_b = proof(51, 52, token_b);
    let request_b = Request {
        credential: token_b,
        ..request(&proof_b)
    };
    consumer
        .verify(&a, &request(&proof_a), &policy, || NOW)
        .await
        .unwrap();
    *provider.deny_sid.lock() = Some(a.expected.sid.clone());
    *provider.record.lock() = record_b;
    consumer
        .verify(&b, &request_b, &policy, || NOW)
        .await
        .unwrap();
    assert!(consumer
        .verify(&a, &request(&proof_a), &policy, || NOW)
        .await
        .is_err());
    consumer
        .verify(&b, &request_b, &policy, || NOW)
        .await
        .unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 4);
}
