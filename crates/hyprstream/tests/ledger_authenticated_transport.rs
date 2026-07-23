//! Ledger admission identity binding through the real RPC boundary.
//!
//! This is deliberately an integration-test binary: its Hybrid envelope
//! verification policy is process-global, and must not be shared with the
//! package lib-test binary's Classical service fixtures.

#![cfg(feature = "ledger")]
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::sync::Arc;

use async_trait::async_trait;
use hyprstream_core::services::ledger::{
    AdmissionRequest, AdmissionResult, AuthenticatedSubject, CoseCheckpointSigner, CreditGate,
    DebtBreaker, DenyReason, LedgerConfig, LedgerHandle, LocalEnforcer, SpendAuthorization,
    StaticGrantVerifier, VerifiedGrant,
};
use hyprstream_crypto::{
    cose_sign::sign_composite,
    pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes, ml_dsa_vk_from_bytes},
};
use hyprstream_ledger::{
    AccountId, AccountSpec, Cid, Did, IssueTransfer, LedgerBackend, MemLedger, Purpose, TransferId,
    UnitId,
};
use parking_lot::Mutex;

fn unit() -> UnitId {
    UnitId {
        issuer: Did("did:web:issuer.test".to_owned()),
        resource_class: "gpu.h100.seconds".to_owned(),
    }
}

fn cid(b: u8) -> Cid {
    Cid(vec![b])
}

fn rand_nonce() -> u128 {
    use rand::RngCore;

    let mut buf = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut buf);
    u128::from_be_bytes(buf)
}

async fn fixture(
    grant_cap: u128,
) -> (
    LocalEnforcer,
    Did,
    Cid,
    ed25519_dalek::SigningKey,
    hyprstream_crypto::pq::MlDsaSigningKey,
) {
    let cell = Did("did:web:cell.test".to_owned());
    let ed_sk = ed25519_dalek::SigningKey::from_bytes(&[1u8; 32]);
    let pq_sk = ml_dsa_sk_from_seed(&[2u8; 32]);
    let holder = Did(hyprstream_crypto::did_key::ed25519_to_did_key(
        &ed_sk.verifying_key().to_bytes(),
    ));

    let mut backend = MemLedger::new(cell.clone());
    let issuer_liab =
        AccountId::derive(&cell, &unit().issuer, &unit(), &Purpose::IssuerLiability).unwrap();
    let holder_acct = AccountId::derive(&cell, &holder, &unit(), &Purpose::Available).unwrap();
    backend
        .open_account(AccountSpec::new(
            cell.clone(),
            unit().issuer.clone(),
            unit(),
            Purpose::IssuerLiability,
        ))
        .unwrap();
    backend
        .open_account(AccountSpec::new(
            cell.clone(),
            holder.clone(),
            unit(),
            Purpose::Available,
        ))
        .unwrap();
    backend
        .open_account(AccountSpec::new(
            cell.clone(),
            cell.clone(),
            unit(),
            Purpose::Available,
        ))
        .unwrap();
    backend
        .credit(IssueTransfer {
            id: TransferId(1),
            issuer_liability: issuer_liab,
            destination: holder_acct,
            unit: unit(),
            amount: grant_cap,
            grant_cid: Some(cid(1)),
            user_data: [0u8; 32],
        })
        .result
        .unwrap();

    let signer = Arc::new(CoseCheckpointSigner::classical(cell.clone(), ed_sk.clone()));
    let handle = LedgerHandle::spawn(Box::new(backend), signer);
    let grant = VerifiedGrant {
        holder: holder.clone(),
        unit: unit(),
        cap_amount: grant_cap,
        exp: u64::MAX,
        epoch: 0,
    };
    let gate = Arc::new(CreditGate::new(Arc::new(
        StaticGrantVerifier::new().with(cid(1), grant),
    )));
    let balance = handle.balance(holder_acct).await.unwrap();
    gate.materialize(&cid(1), balance.available);
    let config = LedgerConfig {
        enabled: true,
        require_pq_signatures: true,
        ..LedgerConfig::default()
    };
    let breaker = Arc::new(DebtBreaker::new(&config, gate.generation_handle()));
    (
        LocalEnforcer::new(gate, handle, breaker, cell, &config),
        holder,
        cid(1),
        ed_sk,
        pq_sk,
    )
}

fn signed_authz(
    grant_cid: &Cid,
    host: &Did,
    nonce: u128,
    ed_sk: &ed25519_dalek::SigningKey,
    pq_sk: &hyprstream_crypto::pq::MlDsaSigningKey,
) -> SpendAuthorization {
    let mut authz = SpendAuthorization {
        grant_cid: grant_cid.clone(),
        host: host.clone(),
        transfer_id: hyprstream_core::services::ledger::enforcer::mint_transfer_id(
            grant_cid, nonce,
        ),
        max_amount: 100,
        exp: u64::MAX,
        signature: Vec::new(),
    };
    authz.signature = sign_composite(ed_sk, Some(pq_sk), &authz.digest(), &[]).unwrap();
    authz
}

#[tokio::test]
async fn verified_user_transport_did_admits_its_grant_and_refuses_another_signer() {
    struct AdmissionProbe {
        enforcer: Arc<LocalEnforcer>,
        grant_cid: Cid,
        nonce: u128,
        authz: SpendAuthorization,
        signing_key: ed25519_dalek::SigningKey,
        key_source: Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
        observed: Arc<Mutex<Vec<(String, Did, Option<DenyReason>)>>>,
    }

    #[async_trait(?Send)]
    impl hyprstream_rpc::RequestService for AdmissionProbe {
        async fn handle_request(
            &self,
            ctx: &hyprstream_rpc::EnvelopeContext,
            _payload: &[u8],
        ) -> anyhow::Result<(Vec<u8>, Option<hyprstream_rpc::Continuation>)> {
            let signer_did = Did(ctx
                .authenticated_pairwise_did()
                .expect("verified envelope has an authenticated signer")
                .to_string());
            let subject = AuthenticatedSubject::from_verified_envelope(ctx).map_err(|error| {
                anyhow::anyhow!("ledger identity conversion refused: {error:?}")
            })?;
            let result = self
                .enforcer
                .admit(
                    &AdmissionRequest {
                        authenticated_subject: Some(subject.clone()),
                        grant_cid: self.grant_cid.clone(),
                        unit: unit(),
                        amount: 100,
                        nonce: self.nonce,
                        spend_authz: Some(self.authz.clone()),
                    },
                    0,
                )
                .await;
            let reason = match &result {
                AdmissionResult::Admitted { .. } => None,
                AdmissionResult::Rejected(rejection) => Some(rejection.reason.clone()),
            };
            self.observed
                .lock()
                .push((ctx.subject().to_string(), signer_did, reason));
            Ok((Vec::new(), None))
        }

        fn name(&self) -> &str {
            "ledger-admission-probe"
        }

        fn transport(&self) -> &hyprstream_rpc::transport::TransportConfig {
            static TRANSPORT: std::sync::OnceLock<hyprstream_rpc::transport::TransportConfig> =
                std::sync::OnceLock::new();
            TRANSPORT.get_or_init(|| {
                hyprstream_rpc::transport::TransportConfig::inproc("ledger-admission-probe")
            })
        }

        fn signing_key(&self) -> ed25519_dalek::SigningKey {
            self.signing_key.clone()
        }

        fn jwt_key_source(&self) -> Option<Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
            Some(Arc::clone(&self.key_source))
        }

        fn jwt_verify_policy(&self) -> hyprstream_rpc::crypto::CryptoPolicy {
            hyprstream_rpc::crypto::CryptoPolicy::Classical
        }
    }

    fn signed_wire(
        client: &ed25519_dalek::SigningKey,
        client_pq: &hyprstream_crypto::pq::MlDsaSigningKey,
        jwt: String,
        server_kem: &hyprstream_rpc::crypto::hybrid_kem::RecipientPublic,
    ) -> Vec<u8> {
        use hyprstream_rpc::ToCapnp as _;

        let response_recipient = hyprstream_rpc::crypto::hybrid_kem::generate_recipient(
            hyprstream_rpc::crypto::hybrid_kem::SuiteId::HyKemX25519MlKem768,
        )
        .expect("response recipient");
        let request = hyprstream_rpc::envelope::RequestEnvelope::new(Vec::new())
            .with_jwt_token(jwt)
            .with_service_domain("ledger-admission-probe")
            .expect("valid service domain")
            .with_response_kem_recipient(response_recipient.public());
        let signed = hyprstream_rpc::envelope::SignedEnvelope::new_signed_encrypted_mesh_kem(
            request, client, client_pq, server_kem,
        )
        .expect("hybrid request envelope");
        let mut message = capnp::message::Builder::new_default();
        signed.write_to(
            &mut message.init_root::<hyprstream_core::common_capnp::signed_envelope::Builder>(),
        );
        let mut wire = Vec::new();
        capnp::serialize::write_message(&mut wire, &message).expect("serialize request");
        wire
    }

    let (enforcer, holder, grant_cid, alice_ed, alice_pq) = fixture(1_000).await;
    let enforcer = Arc::new(enforcer);
    let nonce = rand_nonce();
    let authz = signed_authz(
        &grant_cid,
        enforcer.cell_identity(),
        nonce,
        &alice_ed,
        &alice_pq,
    );
    let bob_ed = ed25519_dalek::SigningKey::from_bytes(&[3u8; 32]);
    let bob_pq = ml_dsa_sk_from_seed(&[4u8; 32]);

    let alice_pq_vk = ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&alice_pq)).unwrap();
    let bob_pq_vk = ml_dsa_vk_from_bytes(&ml_dsa_sk_to_vk_bytes(&bob_pq)).unwrap();
    let mut store = hyprstream_rpc::envelope::KeyedPqTrustStore::new();
    store.bind(alice_ed.verifying_key().to_bytes(), &alice_pq_vk);
    store.bind(bob_ed.verifying_key().to_bytes(), &bob_pq_vk);
    hyprstream_rpc::envelope::install_verify_config(
        hyprstream_rpc::envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(Arc::new(store)),
        },
    )
    .expect("install test-binary verification config");

    let ca = ed25519_dalek::SigningKey::from_bytes(&[7u8; 32]);
    let issuer = "https://ledger-user.test".to_owned();
    let key_source: Arc<dyn hyprstream_rpc::auth::JwtKeySource> = Arc::new(
        hyprstream_rpc::auth::ClusterKeySource::new(ca.verifying_key(), issuer.clone()),
    );
    let observed = Arc::new(Mutex::new(Vec::new()));
    let probe = AdmissionProbe {
        enforcer,
        grant_cid,
        nonce,
        authz,
        signing_key: ed25519_dalek::SigningKey::from_bytes(&[9u8; 32]),
        key_source,
        observed: Arc::clone(&observed),
    };
    let server_kem = hyprstream_rpc::node_identity::derive_mesh_kem_recipient(&probe.signing_key)
        .expect("server KEM recipient");
    let jwt_for = |user: &str, key: &ed25519_dalek::SigningKey| {
        let now = chrono::Utc::now().timestamp();
        hyprstream_rpc::auth::jwt::encode(
            &hyprstream_rpc::auth::Claims::new(user.to_owned(), now, now + 3_600)
                .with_issuer(issuer.clone())
                .with_cnf_jwk(&key.verifying_key().to_bytes()),
            &ca,
        )
    };

    for (user, ed, pq) in [("alice", &alice_ed, &alice_pq), ("bob", &bob_ed, &bob_pq)] {
        hyprstream_rpc::service::dispatch::process_request(
            &signed_wire(ed, pq, jwt_for(user, ed), &server_kem.public()),
            &probe,
            hyprstream_rpc::envelope::EnvelopeVerification::AnySigner,
            &probe.signing_key,
            &hyprstream_rpc::envelope::InMemoryNonceCache::new(),
            hyprstream_rpc::transport::carrier::CarrierContext::iroh(),
        )
        .await
        .expect("verified user request reaches the ledger PEP");
    }

    let observed = observed.lock();
    assert_eq!(observed.len(), 2);
    assert_eq!(
        observed[0].0, "alice",
        "JWT/Casbin user still reaches the handler"
    );
    assert_eq!(
        observed[0].1, holder,
        "holder is the envelope signer's did:key"
    );
    assert_eq!(observed[0].2, None, "holder's authenticated grant admits");
    assert_eq!(observed[1].0, "bob");
    assert_ne!(
        observed[1].1, holder,
        "different signer derives a different DID"
    );
    assert_eq!(observed[1].2, Some(DenyReason::HolderMismatch));
}
