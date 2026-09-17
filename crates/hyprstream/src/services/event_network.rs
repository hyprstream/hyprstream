//! Native Event transport bootstrap. Identity admission and MAC are independent.
use anyhow::{Context, Result};
use hyprstream_discovery::admission_roster::{
    self, DeploymentRosterSource,
};
use hyprstream_rpc::moq_authz::PeerIdentity;
use hyprstream_rpc::transport::iroh_moq::SharedIngressAuthorizer;
use hyprstream_rpc::transport::moql_admission::MoqlAdmissionProof;
use std::collections::BTreeSet;
use std::sync::Arc;

/// The fixed Event tree belongs only to the provisioned local tenant. The
/// process's own proof must be a current accepted deployment service identity
/// — resolved from the live checkpoint store (#1652), never from a DID-valued
/// config binding that could go stale at identity churn.
pub fn require_event_tenant(proof: &MoqlAdmissionProof) -> Result<()> {
    let roster = hyprstream_discovery::production_deployment_roster()?;
    admission_roster::require_admitted_service_binding(&roster, proof)
}

/// Initialize both CLI and service Event clients according to the deployment
/// profile. Required mode never opens a UDS, including when credentials are absent.
pub fn ensure_event_origin_for_profile() -> Result<()> {
    let config = crate::config::HyprConfig::load()?;
    if !config.quic.iroh_required() && !hyprstream_discovery::native_network_required() {
        hyprstream_rpc::moq_event::ensure_event_client_origin(hyprstream_rpc::paths::event_socket());
        return Ok(());
    }
    let proof = hyprstream_rpc::moq_stream::global_moq_admission_proof()
        .context("native Event client requires an installed checkpointed admission proof")?
        .clone();
    require_event_tenant(&proof)?;
    hyprstream_rpc::events::install_network_event_identity(&proof)?;
    if let Some(origin) = hyprstream_rpc::moq_event::install_event_network_client_origin() {
        tokio::spawn(async move {
            loop {
                let result: Result<()> = async {
                    let (transport, identity) =
                        hyprstream_discovery::production_moq_event_target().await?;
                    let mut proof = proof.clone();
                    proof.expected_server = identity;
                    hyprstream_rpc::moq_event::run_event_network_link(&origin, &transport, &proof)
                        .await
                }
                .await;
                if let Err(error) = result {
                    tracing::warn!(%error, "native Event link unavailable");
                }
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
        });
    }
    Ok(())
}

/// The Event ingress grant: the already-admitted peer's verified DID resolves
/// (through the same live accepted-state roster admission used) to a
/// deployment service NAME, which must be a member of the operator's
/// name-keyed `quic.event_publishers` set (#1652). Names survive identity
/// churn, so the grant never goes stale when a service's DID changes; an
/// empty set denies every peer — read-only, never all-publishers.
pub fn event_ingress_authorizer(
    roster: Arc<dyn DeploymentRosterSource>,
    publishers: Arc<BTreeSet<String>>,
) -> SharedIngressAuthorizer {
    Arc::new(move |peer: &PeerIdentity, tenant: &str| {
        tenant == admission_roster::LOCAL_TENANT
            && peer
                .subject
                .as_deref()
                .and_then(|did| admission_roster::roster_service_name(&roster.roster(), did))
                .is_some_and(|name| publishers.contains(name.as_str()))
    })
}

/// Build the Event-owned admission authenticator and the independent explicit
/// ingress grant. An absent MAC policy still denies every Event source.
pub fn event_handler(
    ctx: &hyprstream_service::ServiceContext,
    origin: &hyprstream_rpc::moq_event::MoqEventOrigin,
    node_id: [u8; 32],
) -> Result<hyprstream_rpc::transport::iroh_moq::IrohMoqProtocolHandler> {
    use hyprstream_rpc::transport::iroh_moq::{
        IrohMoqProtocolHandler, MoqAuthzConfig, OriginShared,
    };
    use hyprstream_rpc::transport::moql_admission::{
        MoqlAdmissionAuthenticator, MoqlServerIdentityProof,
    };
    let config = crate::config::HyprConfig::load()?;
    let proof = ctx
        .moql_admission_proof("event")?
        .context("native Event server proof missing")?;
    // Self-binding: the checkpoint store must resolve THIS process's own DID
    // to THIS service, and every configured publisher name must be carried by
    // a live deployment identity. Refusal fails the spawn (fail-closed),
    // replacing the former empty-tenant-map refusal.
    let roster = hyprstream_discovery::production_deployment_roster()?;
    admission_roster::require_service_self_binding(&roster, "event", &proof)?;
    admission_roster::require_publisher_roster(&roster, &config.quic.event_publishers)?;
    hyprstream_rpc::events::install_network_event_identity(&proof)?;
    // Authentication stays on the per-DID authority; only grant resolution
    // runs over the roster (uniqueness needs the whole store).
    let admission = MoqlAdmissionAuthenticator::new(
        hyprstream_discovery::production_moql_accepted_state_authority()?,
        admission_roster::derived_tenant_resolver(Arc::clone(&roster)),
    );
    admission.install_server_identity(
        MoqlServerIdentityProof::from_local_admission_proof(&proof)?,
        node_id,
    )?;
    let authz = hyprstream_rpc::events::installed_event_authz()
        .context("Event MAC reference monitor missing")?;
    Ok(IrohMoqProtocolHandler::with_origin(OriginShared::from_pair(
        origin.producer(),
        origin.consumer(),
    ))
    .with_event_authz(authz)
    .with_authz(
        MoqAuthzConfig::default()
            .with_admission(Arc::new(admission))
            .with_ingress_authorizer(event_ingress_authorizer(
                roster,
                Arc::new(config.quic.event_publishers),
            )),
    ))
}

/// Owns the Event carrier and announcement cancellation for the service lifetime.
pub struct EventNetworkService {
    secret: [u8; 32],
    node_id: [u8; 32],
    handler: hyprstream_rpc::transport::iroh_moq::IrohMoqProtocolHandler,
    announce: hyprstream_service::service::factory::NativeIrohAnnouncementCallback,
}

impl EventNetworkService {
    pub fn new(
        ctx: &hyprstream_service::ServiceContext,
        origin: &hyprstream_rpc::moq_event::MoqEventOrigin,
    ) -> Result<Self> {
        let key = hyprstream_rpc::node_identity::derive_purpose_key(
            &ctx.service_signing_key("event"),
            "hyprstream-iroh-transport-v1",
        );
        let node_id = key.verifying_key().to_bytes();
        Ok(Self {
            secret: key.to_bytes(),
            node_id,
            handler: event_handler(ctx, origin, node_id)?,
            announce: ctx.native_iroh_announcement_callback("event")?,
        })
    }

    async fn run_until_shutdown(
        self,
        shutdown: Arc<tokio::sync::Notify>,
        ready: impl FnOnce(),
    ) -> hyprstream_rpc::error::Result<()> {
        use hyprstream_rpc::error::RpcError;
        // Keep the same registered waiter through bind, publication and readiness.
        // Replacing it after READY can lose notify_waiters() from an owner that
        // stops the service as soon as readiness is observed.
        let stopped = shutdown.notified();
        tokio::pin!(stopped);
        stopped.as_mut().enable();
        let substrate = hyprstream_rpc::transport::iroh_substrate::IrohSubstrate::new(
            self.secret, self.handler,
            hyprstream_rpc::transport::iroh_substrate::RefuseHandler::new("Event provides only authenticated moql"),
        ).await.map_err(|e| RpcError::SpawnFailed(format!("native Event bind: {e}")))?;
        let cancellation = tokio_util::sync::CancellationToken::new();
        let _cancel_on_exit = cancellation.clone().drop_guard();
        let announce_cancel = cancellation.clone();
        let announce = self.announce;
        let node_id = self.node_id;
        let initial = tokio::select! {
            biased;
            _ = &mut stopped => Err(RpcError::SpawnFailed("Event stopped before announcement".into())),
            result = tokio::task::spawn_blocking(move || announce(announce_cancel, node_id)) => {
                result.map_err(|e| RpcError::SpawnFailed(format!("Event announcement task: {e}")))
                    .and_then(|r| r.map_err(|e| RpcError::SpawnFailed(format!("Event initial announcement: {e}"))))
            }
        };
        if let Err(error) = initial {
            cancellation.cancel();
            let _ = substrate.shutdown().await;
            return Err(error);
        }
        ready();
        tokio::select! {
            _ = &mut stopped => {},
            _ = cancellation.cancelled() => {},
            _ = async {
                while !substrate.router().is_shutdown() && !substrate.endpoint().is_closed() {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            } => {},
        }
        cancellation.cancel();
        substrate.shutdown().await.map_err(|e| RpcError::SpawnFailed(format!("Event shutdown: {e}")))
    }
}

impl hyprstream_rpc::service::Spawnable for EventNetworkService {
    fn name(&self) -> &str {
        "event"
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
        runtime.block_on((*self).run_until_shutdown(shutdown, move || {
            if let Some(ready) = on_ready { let _ = ready.send(()); }
            let _ = hyprstream_rpc::notify::ready();
        }))
    }
}

/// Probe the native Event capability: authenticate to the freshly resolved
/// carrier and receive an actual frame from the authorized system source.
/// A carrier handshake alone, or a local retained Event, cannot satisfy it.
pub async fn probe_event_network(timeout: std::time::Duration) -> Result<()> {
    tokio::time::timeout(timeout, async {
        let config = crate::config::HyprConfig::load()?;
        anyhow::ensure!(config.quic.iroh_required() || hyprstream_discovery::native_network_required(), "Event network probe requires network-iroh-required");
        let mut proof = hyprstream_rpc::moq_stream::global_moq_admission_proof()
            .context("Event probe requires a checkpointed client proof")?.clone();
        require_event_tenant(&proof)?;
        let (transport, identity) = hyprstream_discovery::production_moq_event_target().await?;
        proof.expected_server = identity;
        let stream = hyprstream_rpc::dial::dial_stream_authenticated(&transport, &proof).await?;
        let origin = hyprstream_rpc::moq_event::MoqEventOrigin::new();
        // Subscriber-only: a health probe never announces or publishes.
        let client = moq_net::Client::new().with_consume(origin.producer());
        let session = stream.connect_moq(&client).await?;
        let consumer = origin.consumer();
        let evidence = async {
            let broadcast = consumer.announced_broadcast("local/events/system").await.context("system Event source unavailable")?;
            let mut track = broadcast.subscribe_track(&moq_net::Track::new(hyprstream_rpc::moq_event::EVENT_TRACK))?;
            let mut group = track.next_group().await?.context("system Event track has no group")?;
            group.read_frame().await?.context("system Event track has no frame")?;
            Ok(())
        };
        tokio::select! {
            result = evidence => result,
            _ = session.closed() => anyhow::bail!("Event probe session closed before authorized track delivery"),
        }
    }).await.context("native Event capability probe timed out")?
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_event_service(
        announce: hyprstream_service::service::factory::NativeIrohAnnouncementCallback,
    ) -> EventNetworkService {
        use hyprstream_rpc::transport::iroh_moq::{IrohMoqProtocolHandler, OriginShared};
        let key = ed25519_dalek::SigningKey::from_bytes(&rand::random());
        let origin = hyprstream_rpc::moq_event::MoqEventOrigin::new();
        EventNetworkService {
            secret: key.to_bytes(),
            node_id: key.verifying_key().to_bytes(),
            handler: IrohMoqProtocolHandler::with_origin(OriginShared::from_pair(
                origin.producer(),
                origin.consumer(),
            )),
            announce,
        }
    }

    #[tokio::test]
    async fn event_shutdown_at_readiness_is_retained() -> Result<()> {
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let cancellation = tokio_util::sync::CancellationToken::new();
        let announced = Arc::new(parking_lot::Mutex::new(cancellation.clone()));
        let observed = Arc::clone(&announced);
        let service = test_event_service(Arc::new(move |token, _| {
            *observed.lock() = token;
            Ok(())
        }));
        let ready = std::cell::Cell::new(false);
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            service.run_until_shutdown(Arc::clone(&shutdown), || {
                ready.set(true);
                // Deterministically stop inside readiness, before the final
                // serving select is polled. A newly created waiter loses this.
                shutdown.notify_waiters();
            }),
        )
        .await
        .context("Event lost shutdown at readiness")??;
        anyhow::ensure!(ready.get(), "Event never reached readiness");
        anyhow::ensure!(announced.lock().is_cancelled(), "announcement was not cancelled");
        Ok(())
    }

    #[tokio::test]
    async fn event_shutdown_during_announcement_prevents_readiness() -> Result<()> {
        let shutdown = Arc::new(tokio::sync::Notify::new());
        let stopping = Arc::clone(&shutdown);
        let service = test_event_service(Arc::new(move |_, _| {
            stopping.notify_waiters();
            Ok(())
        }));
        let ready = std::cell::Cell::new(false);
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            service.run_until_shutdown(shutdown, || ready.set(true)),
        )
        .await
        .context("Event did not stop during announcement")?;
        anyhow::ensure!(result.is_err(), "startup shutdown must fail readiness");
        anyhow::ensure!(!ready.get(), "Event became ready after startup shutdown");
        Ok(())
    }

    /// A required CLI/service initializer must reject absent proof before it
    /// installs an origin or spawns a compatibility socket task. Isolate process
    /// globals so this tests the real initializer without fixture order effects.
    #[test]
    fn required_event_client_missing_proof_has_no_local_fallback() -> Result<()> {
        const CHILD: &str = "HYPRSTREAM_EVENT_REQUIRED_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let status = std::process::Command::new(std::env::current_exe()?)
                .args(["--exact", "services::event_network::tests::required_event_client_missing_proof_has_no_local_fallback", "--nocapture"])
                .env(CHILD, "1")
                .env("HYPRSTREAM__QUIC__NATIVE_NETWORK_PROFILE", "network-iroh-required")
                .status()?;
            anyhow::ensure!(
                status.success(),
                "required Event initializer isolation failed"
            );
            return Ok(());
        }
        anyhow::ensure!(hyprstream_rpc::moq_event::global_moq_event_origin().is_none());
        let error = ensure_event_origin_for_profile()
            .err()
            .context("missing proof did not reject required Event initialization")?;
        anyhow::ensure!(
            error.to_string().contains("checkpointed admission proof"),
            "unexpected error: {error}"
        );
        anyhow::ensure!(
            hyprstream_rpc::moq_event::global_moq_event_origin().is_none(),
            "required failure installed a local origin"
        );
        Ok(())
    }

    fn state_with_services(service_ids: &[&str]) -> hyprstream_rpc::transport::moql_admission::AcceptedIdentityState {
        use hyprstream_rpc::transport::moql_admission::{AcceptedIdentityState, AcceptedSubjectKey};
        AcceptedIdentityState {
            epoch: 1,
            head_digest: [1; 64],
            subject_keys: vec![AcceptedSubjectKey {
                ed25519: [2; 32],
                ml_dsa_65: vec![3; 1952],
            }],
            service_ids: service_ids.iter().map(ToString::to_string).collect(),
            // Deployment invariants: successor epoch + bounded expiry — the
            // derivation rejects genesis-only/unbounded states.
            expires_at_unix_ms: Some(hyprstream_rpc::envelope::current_timestamp() + 3_600_000),
        }
    }

    fn peer_of(did: &str) -> hyprstream_rpc::moq_authz::PeerIdentity {
        hyprstream_rpc::moq_authz::PeerIdentity::authenticated(did.to_owned())
    }

    /// #1652 churn test (Event ingress grant): when a service's DID changes —
    /// same service name re-bound to a new DID, as at an R1 store re-init —
    /// the name-keyed grant keeps admitting that service in the SAME process,
    /// no restart, no config change. Under the former DID-keyed set the new
    /// DID was silently unlisted (read-only) until config was re-projected
    /// and every container restarted. A foreign record squatting the service
    /// name beside the genuine identity denies BOTH (fail-closed), instead of
    /// self-selecting into the grant.
    #[test]
    fn event_ingress_follows_service_names_across_did_churn() {
        use hyprstream_discovery::admission_roster::DeploymentRosterSource;
        use std::collections::BTreeMap;
        let original = "did:at9p:original";
        let successor = "did:at9p:successor";
        let foreign = "did:at9p:foreign-record";
        let squatter = "did:at9p:squatter";
        let states = Arc::new(parking_lot::RwLock::new(BTreeMap::from([
            (original.to_owned(), state_with_services(&["#event"])),
            (foreign.to_owned(), state_with_services(&["#ns"])),
        ])));
        let roster: Arc<dyn DeploymentRosterSource> = {
            let states = Arc::clone(&states);
            Arc::new(move || states.read().clone().into_iter().collect::<Vec<_>>())
        };
        let publishers: Arc<std::collections::BTreeSet<String>> =
            Arc::new(["event".to_owned()].into());
        let authorizer = event_ingress_authorizer(roster, publishers);

        assert!(authorizer.authorize_ingress(&peer_of(original), "local"));
        // Fail-closed matrix: foreign store record, unknown DID, wrong tenant.
        assert!(!authorizer.authorize_ingress(&peer_of(foreign), "local"));
        assert!(!authorizer.authorize_ingress(&peer_of("did:at9p:unknown"), "local"));
        assert!(!authorizer.authorize_ingress(&peer_of(original), "other"));

        // Identity churn: the same service name now lives at a new DID.
        states.write().remove(original);
        states.write().insert(successor.to_owned(), state_with_services(&["#event"]));
        assert!(
            !authorizer.authorize_ingress(&peer_of(original), "local"),
            "superseded DID must lose the grant"
        );
        assert!(
            authorizer.authorize_ingress(&peer_of(successor), "local"),
            "successor DID keeps ingress through the name-keyed grant, no restart"
        );

        // Name squatting: a foreign record claiming "#event" beside the
        // genuine identity makes the name ambiguous — deny both, escalate to
        // neither (the store cannot distinguish provenance; uniqueness is the
        // discriminator).
        states.write().insert(squatter.to_owned(), state_with_services(&["#event"]));
        assert!(!authorizer.authorize_ingress(&peer_of(successor), "local"));
        assert!(!authorizer.authorize_ingress(&peer_of(squatter), "local"));

        // Empty set = read-only for everyone, including deployment identities.
        states.write().remove(squatter);
        let empty: Arc<std::collections::BTreeSet<String>> = Arc::default();
        let deny_all = event_ingress_authorizer(
            {
                let states = Arc::clone(&states);
                Arc::new(move || states.read().clone().into_iter().collect::<Vec<_>>())
            },
            empty,
        );
        assert!(!deny_all.authorize_ingress(&peer_of(successor), "local"));
    }
}
