//! Real Iroh Event boundary: admission, explicit ingress, MAC source scope and revocation.
#![allow(clippy::expect_used, clippy::unwrap_used)]
use anyhow::{anyhow, Result};
use ed25519_dalek::SigningKey;
use hyprstream_rpc::auth::mac::*;
use hyprstream_rpc::crypto::pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes};
use hyprstream_rpc::envelope::Subject;
use hyprstream_rpc::events::{EventAuthz, MacEventAuthz};
use hyprstream_rpc::moq_authz::PeerIdentity;
use hyprstream_rpc::moq_event::{MoqEventOrigin, EVENT_TRACK};
use hyprstream_rpc::transport::iroh_moq::{IrohMoqProtocolHandler, MoqAuthzConfig, OriginShared};
use hyprstream_rpc::transport::iroh_substrate::{IrohSubstrate, RefuseHandler, ALPN_MOQ_LITE};
use hyprstream_rpc::transport::moql_admission::*;
use moq_net::{Client, Track};
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

const WAIT: Duration = Duration::from_secs(5);

// These labels/clearances are fixture policy only, never deployed defaults.
struct Clearance;
#[async_trait::async_trait]
impl ClearanceSource for Clearance {
    async fn clearance(&self, subject: &Subject) -> Option<SecurityContext> {
        if subject.name() != Some("did:at9p:event-client") {
            return None;
        }
        Some(SecurityContext::from_clearance(
            fixture_label(),
            VerifiedKeyMaterial::Classical,
        ))
    }
}
fn fixture_label() -> SecurityLabel {
    SecurityLabel::new(Level::Public, Assurance::Classical, CompartmentSet::EMPTY)
}
struct Audit;
impl MoqMacAuditSink for Audit {
    fn record_deny(&self, _: &MoqMacAuditRecord) -> std::result::Result<(), String> {
        Ok(())
    }
}
struct Policy {
    enabled: AtomicBool,
    mac: MacEventAuthz,
}
#[async_trait::async_trait]
impl EventAuthz for Policy {
    async fn can_publish(&self, subject: &Subject, source: &str) -> bool {
        self.enabled.load(Ordering::SeqCst) && self.mac.can_publish(subject, source).await
    }
    async fn can_subscribe(&self, subject: &Subject, source: &str) -> bool {
        self.enabled.load(Ordering::SeqCst) && self.mac.can_subscribe(subject, source).await
    }
}

struct Fixture {
    server: IrohSubstrate,
    origin: MoqEventOrigin,
    proof: MoqlAdmissionProof,
    policy: Arc<Policy>,
    ingress: Arc<AtomicBool>,
}

impl Fixture {
    async fn new(tenant: &str, ingress: bool, policy_enabled: bool) -> Result<Self> {
        let client_ed = SigningKey::from_bytes(&[23; 32]);
        let client_pq = ml_dsa_sk_from_seed(&[24; 32]);
        let server_ed = SigningKey::from_bytes(&[25; 32]);
        let server_pq = ml_dsa_sk_from_seed(&[26; 32]);
        let did = "did:at9p:event-client".to_owned();
        let server_did = "did:at9p:event-server".to_owned();
        let expiry = hyprstream_rpc::envelope::current_timestamp() + 60_000;
        let server_identity = hyprstream_rpc::stream_info::MoqlServerIdentity {
            did: server_did.clone(),
            epoch: 1,
            head_digest: vec![9; 64],
            expires_at_unix_ms: expiry,
            ed25519: server_ed.verifying_key().to_bytes(),
            ml_dsa65: ml_dsa_sk_to_vk_bytes(&server_pq),
        };
        let proof = MoqlAdmissionProof {
            did: did.clone(),
            ed25519: client_ed.clone(),
            ml_dsa_65: client_pq.clone(),
            expected_server: server_identity.clone(),
        };
        let states: HashMap<_, _> = [
            (
                did.clone(),
                AcceptedIdentityState {
                    epoch: 1,
                    head_digest: [8; 64],
                    subject_keys: vec![AcceptedSubjectKey {
                        ed25519: client_ed.verifying_key().to_bytes(),
                        ml_dsa_65: ml_dsa_sk_to_vk_bytes(&client_pq),
                    }],
                    expires_at_unix_ms: Some(expiry),
                },
            ),
            (
                server_did,
                AcceptedIdentityState {
                    epoch: 1,
                    head_digest: [9; 64],
                    subject_keys: vec![AcceptedSubjectKey {
                        ed25519: server_ed.verifying_key().to_bytes(),
                        ml_dsa_65: server_identity.ml_dsa65.clone(),
                    }],
                    expires_at_unix_ms: Some(expiry),
                },
            ),
        ]
        .into_iter()
        .collect();
        let secret: [u8; 32] = rand::random();
        let carrier = *iroh::SecretKey::from_bytes(&secret).public().as_bytes();
        let tenant = tenant.to_owned();
        let admission = MoqlAdmissionAuthenticator::new(
            Arc::new(move |did: &str| states.get(did).cloned()),
            Arc::new(move |peer: &PeerIdentity| {
                (peer.subject.as_deref() == Some(&did)).then(|| tenant.clone())
            }),
        )
        .with_server_identity_and_carrier(
            MoqlServerIdentityProof {
                identity: server_identity,
                ed25519: server_ed,
                ml_dsa_65: server_pq,
            },
            carrier,
        );
        let origin = MoqEventOrigin::new();
        let table = if policy_enabled {
            MoqEventPolicyTable::build(
                SUPPORTED_TRACK_POLICY_REVISION,
                ["worker", "system"].map(|source| {
                    MoqEventPolicyRow::new(MoqEventPlane::Event, source, fixture_label()).unwrap()
                }),
            )?
        } else {
            MoqEventPolicyTable::empty()
        };
        let policy = Arc::new(Policy {
            enabled: AtomicBool::new(true),
            mac: MacEventAuthz::new(MoqEventPep::new(
                Arc::new(DeclaredTrackPolicyResolver::new(table)),
                Arc::new(Clearance),
                Arc::new(Audit),
            )),
        });
        let ingress = Arc::new(AtomicBool::new(ingress));
        let grant = ingress.clone();
        let handler = IrohMoqProtocolHandler::with_origin(OriginShared::from_pair(
            origin.producer(),
            origin.consumer(),
        ))
        .with_event_authz(policy.clone())
        .with_authz(
            MoqAuthzConfig::default()
                .with_admission(Arc::new(admission))
                .with_ingress_authorizer(Arc::new(move |_: &PeerIdentity, _: &str| {
                    grant.load(Ordering::SeqCst)
                })),
        );
        let server = IrohSubstrate::new(secret, handler, RefuseHandler::new("MoQ only")).await?;
        Ok(Self {
            server,
            origin,
            proof,
            policy,
            ingress,
        })
    }

    async fn connect(&self) -> Result<(IrohSubstrate, moq_net::Session, MoqEventOrigin)> {
        let endpoint = IrohSubstrate::new(
            rand::random(),
            RefuseHandler::new("client"),
            RefuseHandler::new("client"),
        )
        .await?;
        let addr = iroh::EndpointAddr::from_parts(
            self.server.endpoint_id(),
            self.server
                .endpoint()
                .bound_sockets()
                .into_iter()
                .map(iroh::TransportAddr::Ip),
        );
        let conn = endpoint.connect(addr, ALPN_MOQ_LITE).await?;
        prove_moql_admission(&conn, &self.proof, *endpoint.endpoint_id().as_bytes(), WAIT).await?;
        let origin = MoqEventOrigin::new();
        let session = tokio::time::timeout(
            WAIT,
            Client::new()
                .with_origin(origin.producer())
                .connect(web_transport_iroh::Session::raw(conn)),
        )
        .await??;
        Ok((endpoint, session, origin))
    }
}

async fn read_event(origin: &MoqEventOrigin, source: &str) -> Result<Vec<u8>> {
    let path = format!("local/events/{source}");
    let broadcast = tokio::time::timeout(WAIT, origin.consumer().announced_broadcast(&path))
        .await?
        .ok_or_else(|| anyhow!("missing Event broadcast"))?;
    let mut track = broadcast.subscribe_track(&Track::new(EVENT_TRACK))?;
    let mut group = tokio::time::timeout(WAIT, track.next_group())
        .await??
        .ok_or_else(|| anyhow!("missing group"))?;
    let frame = tokio::time::timeout(WAIT, group.read_frame())
        .await??
        .ok_or_else(|| anyhow!("missing frame"))?;
    Ok(frame.to_vec())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn authenticated_event_ingests_and_serves_declared_sources() -> Result<()> {
    let fixture = Fixture::new("local", true, true).await?;
    let mut publisher = fixture.origin.publisher("system")?;
    publisher.publish_raw("system.event.ready", b"ready")?;
    let (client, session, origin) = fixture.connect().await?;
    assert!(read_event(&origin, "system").await?.ends_with(b"ready"));
    let mut publisher = origin.publisher("worker")?;
    publisher.publish_raw("worker.job.ready", b"ingested")?;
    assert!(read_event(&fixture.origin, "worker")
        .await?
        .ends_with(b"ingested"));
    // This source is absent from the server inventory, even though raw local
    // MoqEventOrigin publishers themselves intentionally have no policy layer.
    let mut unknown = origin.publisher("undeclared")?;
    unknown.publish_raw("undeclared.job.ready", b"must not arrive")?;
    assert!(tokio::time::timeout(
        Duration::from_millis(300),
        fixture
            .origin
            .consumer()
            .announced_broadcast("local/events/undeclared")
    )
    .await
    .is_err());
    fixture.policy.enabled.store(false, Ordering::SeqCst);
    let _ = tokio::time::timeout(WAIT, session.closed()).await?;
    client.shutdown().await?;
    fixture.server.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn event_ingress_requires_separate_grant() -> Result<()> {
    let fixture = Fixture::new("local", false, true).await?;
    let mut publisher = fixture.origin.publisher("system")?;
    publisher.publish_raw("system.event.ready", b"read-only")?;
    let (client, _session, origin) = fixture.connect().await?;
    assert!(read_event(&origin, "system").await?.ends_with(b"read-only"));
    let mut publisher = origin.publisher("worker")?;
    publisher.publish_raw("worker.job.ready", b"unauthorized")?;
    assert!(tokio::time::timeout(
        Duration::from_millis(300),
        fixture
            .origin
            .consumer()
            .announced_broadcast("local/events/worker")
    )
    .await
    .is_err());
    client.shutdown().await?;
    fixture.server.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn event_other_tenant_and_empty_policy_fail_closed() -> Result<()> {
    for (tenant, enabled) in [("other", true), ("local", false)] {
        let fixture = Fixture::new(tenant, true, enabled).await?;
        if let Ok((client, session, _origin)) = fixture.connect().await {
            let _ = tokio::time::timeout(WAIT, session.closed()).await?;
            client.shutdown().await?;
        }
        fixture.server.shutdown().await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn event_live_ingress_grant_revocation_closes_session() -> Result<()> {
    let fixture = Fixture::new("local", true, true).await?;
    let (client, session, _origin) = fixture.connect().await?;
    fixture.ingress.store(false, Ordering::SeqCst);
    let _ = tokio::time::timeout(WAIT, session.closed()).await?;
    client.shutdown().await?;
    fixture.server.shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_local_publishers_receive_checkpointed_identity_without_a_mac_grant() -> Result<()> {
    let fixture = Fixture::new("local", true, true).await?;
    hyprstream_rpc::moq_event::init_global_moq_event_origin(MoqEventOrigin::new());
    // The fixture MAC recognizes only the actual client DID; anonymous source
    // construction alone cannot publish, even with a declared worker row.
    let anonymous =
        hyprstream_rpc::events::EventPublisher::new("worker")?.with_authz(fixture.policy.clone());
    assert!(anonymous
        .publish_raw("worker.job.ready", b"anonymous")
        .await
        .is_err());
    hyprstream_rpc::events::install_network_event_identity(&fixture.proof)?;
    let publisher =
        hyprstream_rpc::events::EventPublisher::new("worker")?.with_authz(fixture.policy.clone());
    publisher
        .publish_raw("worker.job.ready", b"identified")
        .await?;
    fixture.policy.enabled.store(false, Ordering::SeqCst);
    assert!(publisher
        .publish_raw("worker.job.ready", b"revoked")
        .await
        .is_err());
    let mut other = fixture.proof.clone();
    other.did = "did:at9p:other".to_owned();
    assert!(hyprstream_rpc::events::install_network_event_identity(&other).is_err());
    fixture.server.shutdown().await?;
    Ok(())
}
