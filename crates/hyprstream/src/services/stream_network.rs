//! Native Streams origins. Carrier admission is independent of stream payload
//! authentication and MAC authorization performed by the producing RPC service.
use crate::config::QuicConfig;
use anyhow::{Context, Result};
use hyprstream_rpc::transport::iroh_moq::{MoqAuthzConfig, SharedIngressAuthorizer};
use hyprstream_rpc::transport::moql_admission::MoqlAdmissionAuthenticator;
use std::sync::Arc;

/// Current accepted identity and explicit tenant assignment, never carrier identity.
pub fn production_stream_admission(config: &QuicConfig) -> Result<Arc<MoqlAdmissionAuthenticator>> {
    Ok(stream_admission(
        config,
        hyprstream_discovery::production_moql_accepted_state_authority()?,
    ))
}

fn stream_admission(
    config: &QuicConfig,
    authority: Arc<dyn hyprstream_rpc::transport::moql_admission::AcceptedStateAuthority>,
) -> Arc<MoqlAdmissionAuthenticator> {
    let tenants = config.moql_subject_tenants.clone();
    Arc::new(MoqlAdmissionAuthenticator::new(
        authority,
        Arc::new(move |peer| {
            peer.subject
                .as_ref()
                .and_then(|did| tenants.get(did))
                .cloned()
        }),
    ))
}

/// Tenant admission never grants publishing. Configuration is immutable for the
/// process lifetime; removing a roster row requires a service reload/restart.
/// Accepted-state/key revocation is rechecked live by the admission authority.
pub fn stream_ingress_authorizer(config: &QuicConfig) -> SharedIngressAuthorizer {
    let publishers = config.stream_publishers.clone();
    let tenants = config.moql_subject_tenants.clone();
    Arc::new(
        move |peer: &hyprstream_rpc::moq_authz::PeerIdentity, tenant: &str| {
            peer.subject.as_ref().is_some_and(|did| {
                publishers.contains(did) && tenants.get(did).is_some_and(|bound| bound == tenant)
            })
        },
    )
}

fn stream_handler(
    config: &QuicConfig,
    origin: &hyprstream_rpc::moq_stream::MoqStreamOrigin,
    admission: Arc<MoqlAdmissionAuthenticator>,
) -> hyprstream_rpc::transport::iroh_moq::IrohMoqProtocolHandler {
    use hyprstream_rpc::transport::iroh_moq::{IrohMoqProtocolHandler, OriginShared};
    IrohMoqProtocolHandler::with_origin(OriginShared::from_pair(
        origin.producer().clone(),
        origin.consumer().clone(),
    ))
    .with_authz(
        MoqAuthzConfig::default()
            .with_admission(admission)
            .with_ingress_authorizer(stream_ingress_authorizer(config)),
    )
}

/// Owns an independent rendezvous origin. It advertises transport capability,
/// never ownership of tracks living on Registry/Model/TUI producer endpoints.
pub struct StreamsNetworkService {
    secret: [u8; 32],
    node_id: [u8; 32],
    handler: hyprstream_rpc::transport::iroh_moq::IrohMoqProtocolHandler,
    announce: hyprstream_service::service::factory::NativeIrohAnnouncementCallback,
}

impl StreamsNetworkService {
    pub fn new(ctx: &hyprstream_service::ServiceContext) -> Result<Self> {
        let config = crate::config::HyprConfig::load()?;
        let proof = ctx
            .moql_admission_proof("streams")?
            .context("native Streams server proof missing")?;
        anyhow::ensure!(
            config
                .quic
                .moql_subject_tenants
                .get(&proof.did)
                .is_some_and(|tenant| tenant == "local"),
            "native Streams server requires explicit admitted DID binding to local tenant"
        );
        let key = hyprstream_rpc::node_identity::derive_purpose_key(
            &ctx.service_signing_key("streams"),
            "hyprstream-iroh-transport-v1",
        );
        let node_id = key.verifying_key().to_bytes();
        let admission = production_stream_admission(&config.quic)?;
        admission.install_server_identity(
            hyprstream_rpc::transport::moql_admission::MoqlServerIdentityProof::from_local_admission_proof(&proof)?, node_id)?;
        let origin = hyprstream_rpc::moq_stream::MoqStreamOrigin::standalone()
            .with_prefix(hyprstream_rpc::moq_stream::DEFAULT_PREFIX)
            .build();
        Ok(Self {
            secret: key.to_bytes(),
            node_id,
            handler: stream_handler(&config.quic, &origin, admission),
            announce: ctx.native_iroh_announcement_callback("streams")?,
        })
    }
}

impl hyprstream_rpc::service::Spawnable for StreamsNetworkService {
    fn name(&self) -> &str {
        "streams"
    }
    fn registrations(
        &self,
    ) -> Vec<(
        hyprstream_rpc::registry::SocketKind,
        hyprstream_rpc::transport::TransportConfig,
    )> {
        vec![]
    }
    fn run(
        self: Box<Self>,
        shutdown: Arc<tokio::sync::Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> hyprstream_rpc::error::Result<()> {
        use hyprstream_rpc::error::RpcError;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| RpcError::SpawnFailed(e.to_string()))?;
        runtime.block_on(async move {
            let substrate = hyprstream_rpc::transport::iroh_substrate::IrohSubstrate::new(
                self.secret, self.handler,
                hyprstream_rpc::transport::iroh_substrate::RefuseHandler::new("Streams provides only authenticated moql"),
            ).await.map_err(|e| RpcError::SpawnFailed(format!("native Streams bind: {e}")))?;
            let cancellation = tokio_util::sync::CancellationToken::new();
            let _cancel_on_exit = cancellation.clone().drop_guard();
            let announce_cancel = cancellation.clone();
            let announce = self.announce;
            let node_id = self.node_id;
            let initial = tokio::select! {
                biased;
                _ = shutdown.notified() => Err(RpcError::SpawnFailed("Streams stopped before announcement".into())),
                result = tokio::task::spawn_blocking(move || announce(announce_cancel, node_id)) => {
                    result.map_err(|e| RpcError::SpawnFailed(format!("Streams announcement task: {e}")))
                        .and_then(|r| r.map_err(|e| RpcError::SpawnFailed(format!("Streams initial announcement: {e}"))))
                }
            };
            if let Err(error) = initial {
                cancellation.cancel();
                let _ = substrate.shutdown().await;
                return Err(error);
            }
            if let Some(ready) = on_ready { let _ = ready.send(()); }
            let _ = hyprstream_rpc::notify::ready();
            tokio::select! {
                _ = shutdown.notified() => {},
                _ = cancellation.cancelled() => {},
                _ = async {
                    while !substrate.router().is_shutdown() && !substrate.endpoint().is_closed() {
                        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    }
                } => {},
            }
            cancellation.cancel();
            substrate.shutdown().await.map_err(|e| RpcError::SpawnFailed(format!("Streams shutdown: {e}")))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use hyprstream_rpc::crypto::pq::{ml_dsa_sk_from_seed, ml_dsa_sk_to_vk_bytes};
    use hyprstream_rpc::transport::iroh_substrate::{IrohSubstrate, RefuseHandler, ALPN_MOQ_LITE};
    use hyprstream_rpc::transport::moql_admission::{
        prove_moql_admission, AcceptedIdentityState, AcceptedSubjectKey, MoqlAdmissionProof,
        MoqlServerIdentityProof,
    };
    use moq_net::{Client, Group, Origin, Track};
    use std::time::Duration;

    fn identity(name: &str, seed: u8) -> (MoqlAdmissionProof, AcceptedIdentityState) {
        let ed25519 = SigningKey::from_bytes(&[seed; 32]);
        let ml_dsa_65 = ml_dsa_sk_from_seed(&[seed; 32]);
        let expires_at_unix_ms = hyprstream_rpc::envelope::current_timestamp() + 60_000;
        let state = AcceptedIdentityState {
            epoch: 1,
            head_digest: [seed; 64],
            expires_at_unix_ms: Some(expires_at_unix_ms),
            subject_keys: vec![AcceptedSubjectKey {
                ed25519: ed25519.verifying_key().to_bytes(),
                ml_dsa_65: ml_dsa_sk_to_vk_bytes(&ml_dsa_65),
            }],
        };
        let expected_server = hyprstream_rpc::stream_info::MoqlServerIdentity {
            did: format!("did:at9p:{name}"),
            epoch: 1,
            head_digest: state.head_digest.to_vec(),
            expires_at_unix_ms,
            ed25519: ed25519.verifying_key().to_bytes(),
            ml_dsa65: ml_dsa_sk_to_vk_bytes(&ml_dsa_65),
        };
        (
            MoqlAdmissionProof {
                did: expected_server.did.clone(),
                ed25519,
                ml_dsa_65,
                expected_server,
            },
            state,
        )
    }

    fn direct(server: &IrohSubstrate) -> iroh::EndpointAddr {
        iroh::EndpointAddr::from_parts(
            server.endpoint_id(),
            server
                .endpoint()
                .bound_sockets()
                .into_iter()
                .map(iroh::TransportAddr::Ip),
        )
    }

    async fn peer(seed: u8) -> Result<IrohSubstrate> {
        IrohSubstrate::new(
            [seed; 32],
            RefuseHandler::new("no moq"),
            RefuseHandler::new("no rpc"),
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streams_readiness_requires_announcement_and_shutdown_cancels_it() -> Result<()> {
        use hyprstream_rpc::service::Spawnable;
        for fail_announcement in [true, false] {
            let secret = [if fail_announcement { 71 } else { 72 }; 32];
            let node_id = *iroh::SecretKey::from_bytes(&secret).public().as_bytes();
            let observed = Arc::new(parking_lot::Mutex::new(None));
            let callback_observed = Arc::clone(&observed);
            let service = StreamsNetworkService {
                secret,
                node_id,
                handler: hyprstream_rpc::transport::iroh_moq::IrohMoqProtocolHandler::new(),
                announce: Arc::new(move |cancel, actual| {
                    assert_eq!(actual, node_id);
                    *callback_observed.lock() = Some(cancel);
                    anyhow::ensure!(!fail_announcement, "fixture announcement refusal");
                    Ok(())
                }),
            };
            let shutdown = Arc::new(tokio::sync::Notify::new());
            let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
            let stop = Arc::clone(&shutdown);
            let task =
                tokio::task::spawn_blocking(move || Box::new(service).run(stop, Some(ready_tx)));
            let ready = tokio::time::timeout(Duration::from_secs(5), ready_rx).await?;
            if fail_announcement {
                assert!(ready.is_err());
                assert!(task.await?.is_err());
            } else {
                assert!(ready.is_ok());
                shutdown.notify_one();
                tokio::time::timeout(Duration::from_secs(5), task).await???;
            }
            assert!(observed
                .lock()
                .as_ref()
                .is_some_and(tokio_util::sync::CancellationToken::is_cancelled));
        }
        Ok(())
    }

    /// Exercises the exact production stream handler and config-to-authority
    /// wiring, using independent real Iroh peers and mutable accepted state.
    /// Fixture identity acceptance grants no Event labels or production policy.
    #[test]
    fn native_stream_payload_tenant_ingress_and_currentness() -> Result<()> {
        const CHILD: &str = "HYPRSTREAM_NATIVE_STREAM_PAYLOAD_TEST";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "services::stream_network::tests::native_stream_payload_tenant_ingress_and_currentness", "--nocapture"])
                .env(CHILD, "1").status()?;
            anyhow::ensure!(
                status.success(),
                "isolated native stream payload regression failed"
            );
            return Ok(());
        }
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?
            .block_on(stream_roundtrip())
    }

    async fn stream_roundtrip() -> Result<()> {
        let (server_proof, server_state) = identity("streams", 61);
        let (mut publisher, publisher_state) = identity("publisher", 62);
        let (mut reader, reader_state) = identity("reader", 63);
        let (mut other, other_state) = identity("other", 64);
        let server_identity = server_proof.expected_server.clone();
        for proof in [&mut publisher, &mut reader, &mut other] {
            proof.expected_server = server_identity.clone();
        }
        let states = Arc::new(parking_lot::RwLock::new(std::collections::BTreeMap::from(
            [
                (server_proof.did.clone(), server_state),
                (publisher.did.clone(), publisher_state),
                (reader.did.clone(), reader_state),
                (other.did.clone(), other_state),
            ],
        )));
        let authority = Arc::clone(&states);
        let mut config = QuicConfig::default();
        config.moql_subject_tenants = std::collections::BTreeMap::from([
            (server_proof.did.clone(), "local".into()),
            (publisher.did.clone(), "local".into()),
            (reader.did.clone(), "local".into()),
            (other.did.clone(), "other".into()),
        ]);
        config.stream_publishers.insert(publisher.did.clone());
        config.stream_publishers.insert(other.did.clone());
        // An Event ingress row must never authorize stream injection.
        config.event_publishers.insert(reader.did.clone());
        let admission = stream_admission(
            &config,
            Arc::new(move |did: &str| authority.read().get(did).cloned()),
        );
        let secret = [65; 32];
        admission.install_server_identity(
            MoqlServerIdentityProof::from_local_admission_proof(&server_proof)?,
            *iroh::SecretKey::from_bytes(&secret).public().as_bytes(),
        )?;
        let origin = hyprstream_rpc::moq_stream::MoqStreamOrigin::standalone().build();
        let server = IrohSubstrate::new(
            secret,
            stream_handler(&config, &origin, admission),
            RefuseHandler::new("streams no rpc"),
        )
        .await?;
        let writer = peer(66).await?;
        let subscriber = peer(67).await?;
        let outsider = peer(68).await?;
        let writer_conn = writer.connect(direct(&server), ALPN_MOQ_LITE).await?;
        prove_moql_admission(
            &writer_conn,
            &publisher,
            *writer.endpoint_id().as_bytes(),
            Duration::from_secs(2),
        )
        .await?;
        let write_origin = Origin::random().produce();
        let writer_session = Client::new()
            .with_origin(write_origin.clone())
            .connect(web_transport_iroh::Session::raw(writer_conn.clone()))
            .await?;
        let reader_conn = subscriber.connect(direct(&server), ALPN_MOQ_LITE).await?;
        prove_moql_admission(
            &reader_conn,
            &reader,
            *subscriber.endpoint_id().as_bytes(),
            Duration::from_secs(2),
        )
        .await?;
        let read_origin = Origin::random().produce();
        let read_consumer = read_origin.consume();
        let reader_session = Client::new()
            .with_origin(read_origin.clone())
            .with_consume(read_origin.clone())
            .connect(web_transport_iroh::Session::raw(reader_conn))
            .await?;
        let other_conn = outsider.connect(direct(&server), ALPN_MOQ_LITE).await?;
        prove_moql_admission(
            &other_conn,
            &other,
            *outsider.endpoint_id().as_bytes(),
            Duration::from_secs(2),
        )
        .await?;
        let other_origin = Origin::random().produce();
        let other_consumer = other_origin.consume();
        let other_session = Client::new()
            .with_origin(other_origin.clone())
            .with_consume(other_origin.clone())
            .connect(web_transport_iroh::Session::raw(other_conn))
            .await?;
        // Exercise the production required-profile reach dialer too, with the
        // authenticated producer witness and an independently provisioned peer.
        let lookup = iroh::address_lookup::memory::MemoryLookup::new();
        lookup.add_endpoint_info(direct(&server));
        subscriber.endpoint().address_lookup()?.add(lookup);
        assert!(
            hyprstream_rpc::transport::lazy_iroh::install_iroh_client_endpoint(
                subscriber.owned_client_endpoint()
            )
            .is_ok()
        );
        assert!(hyprstream_rpc::moq_stream::init_global_moq_admission_proof(
            reader.clone()
        ));
        hyprstream_rpc::moq_stream::require_native_iroh();
        let reach = hyprstream_rpc::moq_stream::ProducerReachConfig {
            iroh_node_id: Some(*server.endpoint_id().as_bytes()),
            moql_server_identity: Some(server_identity.clone()),
            ..Default::default()
        }
        .reach();
        let network_reader = tokio::time::timeout(
            Duration::from_secs(3),
            hyprstream_rpc::moq_stream::connect_moq_reach_for_profile(
                &reach,
                &server_identity,
                true,
            ),
        )
        .await??;
        let name = "local/streams/authorized-payload";
        let mut broadcast = write_origin
            .create_broadcast(name)
            .context("writer broadcast")?;
        let mut track = broadcast.create_track(Track::new("tokens"))?;
        let mut group = track.create_group(Group::from(0u64))?;
        group.write_frame(bytes::Bytes::from_static(b"producer-owned frame"))?;
        drop(group);
        let received = tokio::time::timeout(
            Duration::from_secs(3),
            read_consumer.announced_broadcast(name),
        )
        .await?
        .context("reader announcement")?;
        let received_track = received.subscribe_track(&Track::new("tokens"))?;
        let mut received_group =
            tokio::time::timeout(Duration::from_secs(3), received_track.get_group(0))
                .await??
                .context("payload group")?;
        assert_eq!(
            received_group.read_frame().await?,
            Some(bytes::Bytes::from_static(b"producer-owned frame"))
        );
        let network_broadcast = tokio::time::timeout(
            Duration::from_secs(3),
            network_reader.consumer.announced_broadcast(name),
        )
        .await?
        .context("required-profile reader announcement")?;
        let network_track = network_broadcast.subscribe_track(&Track::new("tokens"))?;
        let mut network_group =
            tokio::time::timeout(Duration::from_secs(3), network_track.get_group(0))
                .await??
                .context("required-profile group")?;
        assert_eq!(
            network_group.read_frame().await?,
            Some(bytes::Bytes::from_static(b"producer-owned frame"))
        );
        let _injected = read_origin
            .create_broadcast("local/streams/reader-injected")
            .context("reader injection")?;
        let _cross = other_origin
            .create_broadcast("local/streams/cross-tenant")
            .context("cross tenant injection")?;
        for denied in [
            "local/streams/reader-injected",
            "local/streams/cross-tenant",
        ] {
            assert!(tokio::time::timeout(
                Duration::from_millis(250),
                origin.consumer().announced_broadcast(denied)
            )
            .await
            .is_err());
        }
        assert!(tokio::time::timeout(
            Duration::from_millis(250),
            other_consumer.announced_broadcast(name)
        )
        .await
        .is_err());
        // Advancing authoritative state revokes an already writable session.
        states.write().remove(&publisher.did);
        tokio::time::timeout(Duration::from_secs(3), writer_conn.closed()).await?;
        let rejected = writer.connect(direct(&server), ALPN_MOQ_LITE).await?;
        assert!(prove_moql_admission(
            &rejected,
            &publisher,
            *writer.endpoint_id().as_bytes(),
            Duration::from_secs(2)
        )
        .await
        .is_err());
        drop((writer_session, reader_session, other_session));
        writer.shutdown().await?;
        subscriber.shutdown().await?;
        outsider.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }
}
