//! Native Streams origins. Carrier admission is independent of stream payload
//! authentication and MAC authorization performed by the producing RPC service.
use crate::config::QuicConfig;
use anyhow::{Context, Result};
use hyprstream_discovery::admission_roster::{self, DeploymentRosterSource};
use hyprstream_rpc::transport::iroh_moq::{MoqAuthzConfig, SharedIngressAuthorizer};
use hyprstream_rpc::transport::moql_admission::{
    AcceptedStateAuthority, MoqlAdmissionAuthenticator,
};
use std::sync::Arc;

/// Current accepted identity and derived tenant assignment, never carrier
/// identity. Authentication runs on the per-DID accepted-state authority;
/// the tenant resolver runs over the roster-wide derivation (#1652): a
/// factory service entry held UNIQUELY across the store ⇒ `local`,
/// everything else — foreign records, ambiguous names — unresolved/deny. No
/// DID-valued config to go stale at identity churn.
pub fn production_stream_admission() -> Result<Arc<MoqlAdmissionAuthenticator>> {
    Ok(stream_admission_over(
        hyprstream_discovery::production_moql_accepted_state_authority()?,
        hyprstream_discovery::production_deployment_roster()?,
    ))
}

pub(crate) fn stream_admission_over(
    authority: Arc<dyn AcceptedStateAuthority>,
    roster: Arc<dyn DeploymentRosterSource>,
) -> Arc<MoqlAdmissionAuthenticator> {
    Arc::new(MoqlAdmissionAuthenticator::new(
        authority,
        admission_roster::derived_tenant_resolver(roster),
    ))
}

/// Test/fixture entry: both halves over one roster (the roster carries every
/// projected state, so the per-DID authority is a lookup into it).
#[cfg(test)]
pub(crate) fn stream_admission(
    roster: Arc<dyn DeploymentRosterSource>,
) -> Arc<MoqlAdmissionAuthenticator> {
    let authority: Arc<dyn AcceptedStateAuthority> = {
        let roster = Arc::clone(&roster);
        Arc::new(move |did: &str| {
            let snapshot = roster.roster();
            snapshot
                .iter()
                .find(|(d, _)| d == did)
                .map(|(_, state)| state.clone())
        })
    };
    stream_admission_over(authority, roster)
}

/// Tenant admission never grants publishing. `quic.stream_publishers` is a
/// deliberately-empty DID set (no production writer; empty = read-only, and
/// an empty set cannot go stale — #1652 leaves it untouched). The tenant
/// itself is re-derived from the live roster on every check, so removing a
/// peer from the deployment roster — or a squatter making its name ambiguous
/// — is effective on the next recheck without a restart.
pub fn stream_ingress_authorizer(
    config: &QuicConfig,
    roster: Arc<dyn DeploymentRosterSource>,
) -> SharedIngressAuthorizer {
    let publishers = config.stream_publishers.clone();
    Arc::new(
        move |peer: &hyprstream_rpc::moq_authz::PeerIdentity, tenant: &str| {
            peer.subject.as_deref().is_some_and(|did| {
                publishers.contains(did)
                    && {
                        let snapshot = roster.roster();
                        admission_roster::roster_tenant(
                            &snapshot,
                            did,
                            hyprstream_rpc::envelope::current_timestamp(),
                        )
                        .as_deref()
                            == Some(tenant)
                    }
            })
        },
    )
}

fn stream_handler(
    config: &QuicConfig,
    roster: Arc<dyn DeploymentRosterSource>,
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
            .with_ingress_authorizer(stream_ingress_authorizer(config, roster)),
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
        // Self-binding (#1652): the checkpoint store must resolve THIS
        // process's own DID to the Streams service. Refusal fails the spawn
        // (fail-closed) — the same refuse-to-start shape as the former empty
        // tenant map, now bound to live store state instead of config.
        let roster = hyprstream_discovery::production_deployment_roster()?;
        admission_roster::require_service_self_binding(&roster, "streams", &proof)?;
        let key = hyprstream_rpc::node_identity::derive_purpose_key(
            &ctx.service_signing_key("streams"),
            "hyprstream-iroh-transport-v1",
        );
        let node_id = key.verifying_key().to_bytes();
        let admission = stream_admission_over(
            hyprstream_discovery::production_moql_accepted_state_authority()?,
            Arc::clone(&roster),
        );
        admission.install_server_identity(
            hyprstream_rpc::transport::moql_admission::MoqlServerIdentityProof::from_local_admission_proof(&proof)?, node_id)?;
        let origin = hyprstream_rpc::moq_stream::MoqStreamOrigin::standalone()
            .with_prefix(hyprstream_rpc::moq_stream::DEFAULT_PREFIX)
            .build();
        Ok(Self {
            secret: key.to_bytes(),
            node_id,
            handler: stream_handler(&config.quic, roster, &origin, admission),
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

    fn identity(
        name: &str,
        seed: u8,
        service_ids: &[&str],
    ) -> (MoqlAdmissionProof, AcceptedIdentityState) {
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
            service_ids: service_ids.iter().map(ToString::to_string).collect(),
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

    fn live_roster(
        states: &Arc<parking_lot::RwLock<std::collections::BTreeMap<String, AcceptedIdentityState>>>,
    ) -> Arc<dyn hyprstream_discovery::admission_roster::DeploymentRosterSource> {
        let states = Arc::clone(states);
        Arc::new(move || states.read().clone().into_iter().collect::<Vec<_>>())
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
        let (server_proof, server_state) = identity("streams", 61, &["#streams"]);
        let (mut publisher, publisher_state) = identity("publisher", 62, &["#model"]);
        let (mut reader, reader_state) = identity("reader", 63, &["#policy"]);
        let (mut foreign, foreign_state) = identity("foreign-record", 64, &["#ns"]);
        let server_identity = server_proof.expected_server.clone();
        for proof in [&mut publisher, &mut reader, &mut foreign] {
            proof.expected_server = server_identity.clone();
        }
        let states = Arc::new(parking_lot::RwLock::new(std::collections::BTreeMap::from(
            [
                (server_proof.did.clone(), server_state),
                (publisher.did.clone(), publisher_state),
                (reader.did.clone(), reader_state),
                (foreign.did.clone(), foreign_state),
            ],
        )));
        // #1652: no tenant map in config at all — admission derives tenants
        // from the live authority. The Event ingress set is name-keyed; an
        // Event grant on the reader's service must never authorize stream
        // injection.
        let mut config = QuicConfig::default();
        config.stream_publishers.insert(publisher.did.clone());
        config.event_publishers.insert("policy".to_owned());
        let admission = stream_admission(live_roster(&states));
        let secret = [65; 32];
        admission.install_server_identity(
            MoqlServerIdentityProof::from_local_admission_proof(&server_proof)?,
            *iroh::SecretKey::from_bytes(&secret).public().as_bytes(),
        )?;
        let origin = hyprstream_rpc::moq_stream::MoqStreamOrigin::standalone().build();
        let server = IrohSubstrate::new(
            secret,
            stream_handler(&config, live_roster(&states), &origin, admission),
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
        // A foreign at9p record present in the store (arbitrary `#ns`
        // service entry, not a factory service) must be denied at the
        // admission exchange itself — the ∩ get_factory filter.
        let foreign_conn = outsider.connect(direct(&server), ALPN_MOQ_LITE).await?;
        assert!(
            prove_moql_admission(
                &foreign_conn,
                &foreign,
                *outsider.endpoint_id().as_bytes(),
                Duration::from_secs(2),
            )
            .await
            .is_err(),
            "foreign store record must not derive a tenant"
        );
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
        // An admitted-but-unlisted peer stays read-only: the reader (tenant
        // local via #policy, absent from stream_publishers, Event grant
        // notwithstanding) cannot inject into the shared origin.
        assert!(tokio::time::timeout(
            Duration::from_millis(250),
            origin.consumer().announced_broadcast("local/streams/reader-injected")
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
        drop((writer_session, reader_session));
        writer.shutdown().await?;
        subscriber.shutdown().await?;
        outsider.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }

    /// #1652 THE CHURN TEST (Streams, full admission exchange): a service's
    /// DID changes — same service name re-bound to a new DID in the live
    /// store, exactly the R1 store re-init shape — and admission for that
    /// service keeps working in the SAME process, no restart, no config
    /// change. Under the former DID-keyed `quic.moql_subject_tenants` map the
    /// successor DID had no binding and every admission for it failed until
    /// config.toml was re-projected and every container restarted.
    #[test]
    fn native_stream_admission_survives_did_churn_without_restart() -> Result<()> {
        const CHILD: &str = "HYPRSTREAM_NATIVE_STREAM_CHURN_TEST";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "services::stream_network::tests::native_stream_admission_survives_did_churn_without_restart", "--nocapture"])
                .env(CHILD, "1").status()?;
            anyhow::ensure!(
                status.success(),
                "isolated native stream churn regression failed"
            );
            return Ok(());
        }
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?
            .block_on(stream_churn())
    }

    async fn stream_churn() -> Result<()> {
        let (server_proof, server_state) = identity("streams", 71, &["#streams"]);
        let (mut original, original_state) = identity("original-did", 72, &["#model"]);
        let (mut successor, successor_state) = identity("successor-did", 73, &["#model"]);
        let server_identity = server_proof.expected_server.clone();
        original.expected_server = server_identity.clone();
        successor.expected_server = server_identity.clone();
        let states = Arc::new(parking_lot::RwLock::new(std::collections::BTreeMap::from(
            [
                (server_proof.did.clone(), server_state),
                (original.did.clone(), original_state),
            ],
        )));
        // The config object is built ONCE, before the churn, and never
        // touched again — there is no tenant map to touch.
        let config = QuicConfig::default();
        let admission = stream_admission(live_roster(&states));
        let secret = [75; 32];
        admission.install_server_identity(
            MoqlServerIdentityProof::from_local_admission_proof(&server_proof)?,
            *iroh::SecretKey::from_bytes(&secret).public().as_bytes(),
        )?;
        let origin = hyprstream_rpc::moq_stream::MoqStreamOrigin::standalone().build();
        let server = IrohSubstrate::new(
            secret,
            stream_handler(&config, live_roster(&states), &origin, admission),
            RefuseHandler::new("streams no rpc"),
        )
        .await?;
        let client = peer(76).await?;

        // Pre-churn: the original DID admits through derived resolution.
        let conn = client.connect(direct(&server), ALPN_MOQ_LITE).await?;
        prove_moql_admission(
            &conn,
            &original,
            *client.endpoint_id().as_bytes(),
            Duration::from_secs(2),
        )
        .await?;
        drop(conn);

        // Identity churn: the same service name (#model) is re-bound to a new
        // DID; the superseded DID leaves the store.
        states.write().remove(&original.did);
        states
            .write()
            .insert(successor.did.clone(), successor_state);

        // Same process, same config: the successor DID admits.
        let churned = client.connect(direct(&server), ALPN_MOQ_LITE).await?;
        prove_moql_admission(
            &churned,
            &successor,
            *client.endpoint_id().as_bytes(),
            Duration::from_secs(2),
        )
        .await
        .context("successor DID must admit after churn without restart or config change")?;
        drop(churned);

        // The superseded DID is denied.
        let stale = client.connect(direct(&server), ALPN_MOQ_LITE).await?;
        assert!(
            prove_moql_admission(
                &stale,
                &original,
                *client.endpoint_id().as_bytes(),
                Duration::from_secs(2),
            )
            .await
            .is_err(),
            "superseded DID must stop admitting"
        );
        drop(stale);
        client.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }

    /// #1652 revocation-on-recheck: `is_still_current` re-runs the tenant
    /// resolver on every recheck (moql_admission.rs "removal or reassignment
    /// closes a previously admitted scoped session"). With the former
    /// config-cloned map that clause was inert — the map could never change
    /// within a process. Under derivation a store-side roster edit is
    /// effective on the next recheck: re-assigning the subject's capsule
    /// service entry OFF the deployment roster (same DID, same keys, same
    /// epoch, still live — only the `#service` id changes to a non-factory
    /// entry) closes the previously admitted session. Reassignment to another
    /// deployment service keeps the tenant, so the session legitimately
    /// survives — the tenant, not the service name, is the session binding.
    #[test]
    fn derived_tenant_recheck_closes_session_on_roster_removal() -> Result<()> {
        use hyprstream_rpc::moq_authz::PeerIdentity;
        use hyprstream_rpc::transport::moql_admission::AdmittedMoqPeer;
        let (server_proof, server_state) = identity("streams", 81, &["#streams"]);
        let (model_proof, model_state) = identity("model-service", 82, &["#model"]);
        let states = Arc::new(parking_lot::RwLock::new(std::collections::BTreeMap::from(
            [
                (server_proof.did.clone(), server_state),
                (model_proof.did.clone(), model_state.clone()),
            ],
        )));
        let admission = stream_admission(live_roster(&states));
        admission.install_server_identity(
            MoqlServerIdentityProof::from_local_admission_proof(&server_proof)?,
            [83; 32],
        )?;
        let admitted = AdmittedMoqPeer {
            peer: PeerIdentity::authenticated(model_proof.did.clone()),
            tenant: "local".to_owned(),
            epoch: model_state.epoch,
            head_digest: model_state.head_digest,
            subject_ed25519: model_proof.ed25519.verifying_key().to_bytes(),
            carrier_node_id: [84; 32],
        };
        assert!(
            admission.is_still_current(&admitted),
            "live deployment identity must remain current"
        );
        // Same-tenant reassignment: another deployment service name — the
        // derived tenant is unchanged, so the session stays current.
        let mut moved = model_state.clone();
        moved.service_ids = vec!["#policy".to_owned()];
        states.write().insert(model_proof.did.clone(), moved);
        assert!(
            admission.is_still_current(&admitted),
            "same-tenant service reassignment must not close the session"
        );
        // Off-roster reassignment: a foreign `#ns` entry (public ingest
        // shape). Keys, epoch, digest and liveness are all unchanged — only
        // roster membership is gone — and the session must close.
        let mut off_roster = model_state.clone();
        off_roster.service_ids = vec!["#ns".to_owned()];
        states.write().insert(model_proof.did.clone(), off_roster);
        assert!(
            !admission.is_still_current(&admitted),
            "roster removal must close the previously admitted session on recheck"
        );
        Ok(())
    }
}
