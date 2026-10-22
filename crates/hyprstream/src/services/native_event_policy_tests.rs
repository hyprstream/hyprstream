use super::*;
use anyhow::{anyhow, Result};
use ed25519_dalek::SigningKey;
use hyprstream_rpc::auth::mac::{
    DeclaredTrackPolicyResolver, MoqEventPep, MoqMacAuditRecord, MoqMacAuditSink,
};
use hyprstream_rpc::crypto::pq::{
    ml_dsa_sk_from_seed, ml_dsa_sk_to_vk, ml_dsa_sk_to_vk_bytes, MlDsaSigningKey,
};
use hyprstream_rpc::moq_event::{MoqEventOrigin, EVENT_TRACK};
use hyprstream_rpc::transport::iroh_moq::{IrohMoqProtocolHandler, MoqAuthzConfig, OriginShared};
use hyprstream_rpc::transport::iroh_substrate::{IrohSubstrate, RefuseHandler, ALPN_MOQ_LITE};
use hyprstream_rpc::transport::moql_admission::*;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(10);
type States = Arc<RwLock<HashMap<String, AcceptedIdentityState>>>;
struct Identity {
    did: String,
    ed: SigningKey,
    pq: MlDsaSigningKey,
}
struct Fixture {
    identities: HashMap<String, Identity>,
    states: States,
    authority: Arc<EnrolledEventAuthority>,
}
impl Fixture {
    fn new() -> Result<Self> {
        let mut identities = HashMap::new();
        let mut states = HashMap::new();
        let mut bootstrap = HashMap::new();
        for (name, seed) in [("event", 81), ("model", 83), ("registry", 85)] {
            let ed = SigningKey::from_bytes(&[seed; 32]);
            let pq = ml_dsa_sk_from_seed(&[seed + 1; 32]);
            let did = format!("did:at9p:{name}");
            bootstrap.insert(
                name.to_owned(),
                crate::auth::identity_store::BootstrapPubkey {
                    ed25519: ed.verifying_key(),
                    ml_dsa_65: Some(ml_dsa_sk_to_vk(&pq)),
                },
            );
            states.insert(
                did.clone(),
                AcceptedIdentityState {
                    epoch: 1,
                    head_digest: [seed; 64],
                    subject_keys: vec![AcceptedSubjectKey {
                        ed25519: ed.verifying_key().to_bytes(),
                        ml_dsa_65: ml_dsa_sk_to_vk_bytes(&pq),
                    }],
                    service_ids: vec![format!("#{name}")],
                    expires_at_unix_ms: Some(
                        hyprstream_rpc::envelope::current_timestamp() + 120_000,
                    ),
                },
            );
            identities.insert(name.to_owned(), Identity { did, ed, pq });
        }
        let manifest = ServiceEnrollmentManifest::from_bootstrap(&bootstrap);
        manifest.validate_key_agreement(&bootstrap)?;
        let states = Arc::new(RwLock::new(states));
        let accepted_states = states.clone();
        let roster_states = states.clone();
        let authority = Arc::new(EnrolledEventAuthority {
            manifest: Arc::new(manifest),
            accepted: Arc::new(move |did: &str| accepted_states.read().get(did).cloned()),
            roster: Arc::new(move || {
                roster_states
                    .read()
                    .iter()
                    .map(|(d, s)| (d.clone(), s.clone()))
                    .collect()
            }),
        });
        Ok(Self {
            identities,
            states,
            authority,
        })
    }
    fn authz(&self, audit: Arc<Audit>) -> Arc<NativeEventAuthz> {
        Arc::new(NativeEventAuthz {
            authority: self.authority.clone(),
            mac: MacEventAuthz::new(MoqEventPep::new(
                Arc::new(DeclaredTrackPolicyResolver::new(
                    internal_event_policy().expect("finite policy"),
                )),
                self.authority.clone(),
                audit,
            )),
        })
    }
    fn server_identity(&self) -> MoqlServerIdentityProof {
        let id = &self.identities["event"];
        let state = &self.states.read()[&id.did];
        MoqlServerIdentityProof {
            identity: hyprstream_rpc::stream_info::MoqlServerIdentity {
                did: id.did.clone(),
                epoch: state.epoch,
                head_digest: state.head_digest.to_vec(),
                expires_at_unix_ms: state.expires_at_unix_ms.expect("bounded successor"),
                ed25519: id.ed.verifying_key().to_bytes(),
                ml_dsa65: ml_dsa_sk_to_vk_bytes(&id.pq),
            },
            ed25519: id.ed.clone(),
            ml_dsa_65: id.pq.clone(),
        }
    }
}
#[derive(Default)]
struct Audit(parking_lot::Mutex<Vec<MoqMacAuditRecord>>);
impl MoqMacAuditSink for Audit {
    fn record_deny(&self, record: &MoqMacAuditRecord) -> std::result::Result<(), String> {
        self.0.lock().push(record.clone());
        Ok(())
    }
}

#[tokio::test]
async fn native_event_full_mac_policy_rechecks_never_write_permit_audits() -> Result<()> {
    let f = Fixture::new()?;
    let audit = Arc::new(Audit::default());
    let mac = MacEventAuthz::new(MoqEventPep::new(
        Arc::new(DeclaredTrackPolicyResolver::new(internal_event_policy()?)),
        f.authority.clone(),
        audit.clone(),
    ));
    let subject = Subject::new(f.identities["model"].did.clone());
    for _ in 0..100 {
        for source in INTERNAL_EVENT_SOURCES {
            assert!(mac.revalidate_publish(&subject, source).await);
            assert!(mac.revalidate_subscribe(&subject, source).await);
        }
    }
    assert!(
        audit.0.lock().is_empty(),
        "every sink call is counted, including hypothetical permit writes"
    );
    f.states.write().remove(&f.identities["model"].did);
    assert!(!mac.revalidate_subscribe(&subject, "system").await);
    assert!(
        audit.0.lock().is_empty(),
        "revoked watchdog grants close without hypothetical operation audits"
    );
    assert!(!mac.can_subscribe(&subject, "system").await);
    assert_eq!(
        audit.0.lock().len(),
        1,
        "fresh revocation must deny and audit once"
    );
    Ok(())
}

#[tokio::test]
async fn native_event_enrollment_currentness_and_exact_source_matrix() -> Result<()> {
    let f = Fixture::new()?;
    let audit = Arc::new(Audit::default());
    let authz = f.authz(audit.clone());
    let subject = Subject::new(f.identities["model"].did.clone());
    let state = f.states.read()[&f.identities["model"].did].clone();
    let witness = AdmittedMoqPeer {
        peer: hyprstream_rpc::moq_authz::PeerIdentity::authenticated(
            f.identities["model"].did.clone(),
        ),
        tenant: "local".to_owned(),
        epoch: state.epoch,
        head_digest: state.head_digest,
        subject_ed25519: f.identities["model"].ed.verifying_key().to_bytes(),
        carrier_node_id: [0; 32],
    };
    assert!(authz.admission_is_current(&witness));
    for changed in [
        AdmittedMoqPeer {
            tenant: "foreign".to_owned(),
            ..witness.clone()
        },
        AdmittedMoqPeer {
            epoch: 2,
            ..witness.clone()
        },
        AdmittedMoqPeer {
            head_digest: [0; 64],
            ..witness.clone()
        },
        AdmittedMoqPeer {
            subject_ed25519: [1; 32],
            ..witness.clone()
        },
    ] {
        assert!(!authz.admission_is_current(&changed));
    }
    // Another accepted key under the same DID must not inherit the enrolled
    // key's clearance, even though generic carrier admission can accept it.
    f.states
        .write()
        .get_mut(&f.identities["model"].did)
        .expect("model")
        .subject_keys
        .push(AcceptedSubjectKey {
            ed25519: [1; 32],
            ml_dsa_65: vec![1; 1952],
        });
    assert!(!authz.admission_is_current(&AdmittedMoqPeer {
        subject_ed25519: [1; 32],
        ..witness
    }));
    assert!(authz.can_publish(&subject, "model").await);
    assert!(authz.can_publish(&subject, "system").await);
    assert!(authz.can_subscribe(&subject, "system").await);
    assert!(!authz.can_publish(&subject, "worker").await);
    assert!(!authz.can_subscribe(&subject, "model").await);
    let registry = Subject::new(f.identities["registry"].did.clone());
    assert!(!authz.can_publish(&registry, "model").await);
    assert!(
        !authz
            .can_publish(&Subject::new("did:at9p:foreign"), "system")
            .await
    );
    let did = &f.identities["model"].did;
    let original = f.states.read()[did].clone();
    for replacement in [
        None,
        Some(AcceptedIdentityState {
            expires_at_unix_ms: Some(0),
            ..original.clone()
        }),
        Some(AcceptedIdentityState {
            head_digest: [0; 64],
            subject_keys: vec![],
            ..original.clone()
        }),
        Some(AcceptedIdentityState {
            service_ids: vec!["#registry".to_owned()],
            ..original.clone()
        }),
    ] {
        if let Some(state) = replacement {
            f.states.write().insert(did.clone(), state);
        } else {
            f.states.write().remove(did);
        }
        assert!(!authz.can_publish(&subject, "system").await);
    }
    f.states.write().insert(did.clone(), original.clone());
    let mut no_enrollment = (*f.authority.manifest).clone();
    no_enrollment.services.remove("model");
    let missing = EnrolledEventAuthority {
        manifest: Arc::new(no_enrollment),
        accepted: f.authority.accepted.clone(),
        roster: f.authority.roster.clone(),
    };
    assert!(missing.clearance(&subject).await.is_none());
    let mut low = (*f.authority.manifest).clone();
    low.services.get_mut("model").expect("enrolled").clearance =
        SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY);
    let low = Arc::new(EnrolledEventAuthority {
        manifest: Arc::new(low),
        accepted: f.authority.accepted.clone(),
        roster: f.authority.roster.clone(),
    });
    let mac = MacEventAuthz::new(MoqEventPep::new(
        Arc::new(DeclaredTrackPolicyResolver::new(internal_event_policy()?)),
        low,
        audit.clone(),
    ));
    assert!(!mac.can_publish(&subject, "system").await);
    assert!(!audit.0.lock().is_empty(), "real denials must be audited");
    Ok(())
}

async fn connect(
    f: &Fixture,
    server: &IrohSubstrate,
    name: &str,
) -> Result<(IrohSubstrate, moq_net::Session, MoqEventOrigin)> {
    let endpoint = IrohSubstrate::new(
        rand::random(),
        RefuseHandler::new("client"),
        RefuseHandler::new("client"),
    )
    .await?;
    let address = iroh::EndpointAddr::from_parts(
        server.endpoint_id(),
        server
            .endpoint()
            .bound_sockets()
            .into_iter()
            .map(iroh::TransportAddr::Ip),
    );
    let conn = endpoint.connect(address, ALPN_MOQ_LITE).await?;
    let identity = &f.identities[name];
    prove_moql_admission(
        &conn,
        &MoqlAdmissionProof {
            did: identity.did.clone(),
            ed25519: identity.ed.clone(),
            ml_dsa_65: identity.pq.clone(),
            expected_server: f.server_identity().identity,
        },
        *endpoint.endpoint_id().as_bytes(),
        WAIT,
    )
    .await?;
    let origin = MoqEventOrigin::new();
    let session = tokio::time::timeout(
        WAIT,
        moq_net::Client::new()
            .with_origin(origin.producer())
            .connect(web_transport_iroh::Session::raw(conn)),
    )
    .await??;
    Ok((endpoint, session, origin))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_event_admitted_producer_delivers_frame_to_distinct_consumer() -> Result<()> {
    let f = Fixture::new()?;
    let audit = Arc::new(Audit::default());
    let origin = MoqEventOrigin::new();
    let secret = rand::random();
    let carrier = *iroh::SecretKey::from_bytes(&secret).public().as_bytes();
    let admission = MoqlAdmissionAuthenticator::new(
        f.authority.accepted.clone(),
        admission_roster::derived_tenant_resolver(f.authority.roster.clone()),
    )
    .with_server_identity_and_carrier(f.server_identity(), carrier);
    let handler = IrohMoqProtocolHandler::with_origin(OriginShared::from_pair(
        origin.producer(),
        origin.consumer(),
    ))
    .with_event_authz(f.authz(audit.clone()))
    .with_authz(
        MoqAuthzConfig::default()
            .with_admission(Arc::new(admission))
            .with_ingress_authorizer(super::super::event_network::event_ingress_authorizer(
                f.authority.roster.clone(),
                Arc::new(["model".to_owned()].into_iter().collect()),
            )),
    );
    let server = IrohSubstrate::new(secret, handler, RefuseHandler::new("Event only")).await?;
    let (producer, producer_session, producer_origin) = connect(&f, &server, "model").await?;
    let (consumer, consumer_session, consumer_origin) = connect(&f, &server, "registry").await?;
    let mut publisher = producer_origin.publisher("system")?;
    publisher.publish_raw("system.model.ready", b"native-frame-proof")?;
    let frame = tokio::time::timeout(WAIT, async {
        let broadcast = consumer_origin
            .consumer()
            .announced_broadcast("local/events/system")
            .await
            .ok_or_else(|| anyhow!("missing native broadcast"))?;
        let mut track = broadcast.subscribe_track(&moq_net::Track::new(EVENT_TRACK))?;
        let mut group = track
            .next_group()
            .await?
            .ok_or_else(|| anyhow!("missing group"))?;
        group
            .read_frame()
            .await?
            .ok_or_else(|| anyhow!("missing frame"))
    })
    .await??;
    assert!(frame.ends_with(b"native-frame-proof"));
    let idle_denials = audit.0.lock().len();
    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(
        audit.0.lock().len(),
        idle_denials,
        "unchanged hypothetical denied scopes are not re-audited"
    );
    // Registry is not a configured publisher, despite possessing real enrollment.
    let mut forbidden = consumer_origin.publisher("model")?;
    forbidden.publish_raw("model.foreign.ready", b"must-not-arrive")?;
    assert!(tokio::time::timeout(
        Duration::from_millis(250),
        origin.consumer().announced_broadcast("local/events/model")
    )
    .await
    .is_err());
    f.states.write().remove(&f.identities["model"].did);
    let _ = tokio::time::timeout(WAIT, producer_session.closed()).await?;
    f.states
        .write()
        .get_mut(&f.identities["registry"].did)
        .expect("live registry")
        .expires_at_unix_ms = Some(0);
    let _ = tokio::time::timeout(WAIT, consumer_session.closed()).await?;
    producer.shutdown().await?;
    consumer.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}
