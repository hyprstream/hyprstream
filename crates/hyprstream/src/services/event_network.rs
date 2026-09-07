//! Native Event transport bootstrap. Identity admission and MAC are independent.
use anyhow::{Context, Result};
use hyprstream_rpc::transport::moql_admission::MoqlAdmissionProof;
use std::sync::Arc;

/// The fixed Event tree belongs only to the explicitly provisioned local tenant.
pub fn require_event_tenant(
    config: &crate::config::QuicConfig,
    proof: &MoqlAdmissionProof,
) -> Result<()> {
    anyhow::ensure!(
        config
            .moql_subject_tenants
            .get(&proof.did)
            .is_some_and(|t| t == "local"),
        "Event identity {} requires explicit quic.moql_subject_tenants binding to local",
        proof.did
    );
    Ok(())
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
    require_event_tenant(&config.quic, &proof)?;
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
    require_event_tenant(&config.quic, &proof)?;
    hyprstream_rpc::events::install_network_event_identity(&proof)?;
    let tenants = config.quic.moql_subject_tenants.clone();
    let publishers = config.quic.event_publishers.clone();
    let admission = MoqlAdmissionAuthenticator::new(
        hyprstream_discovery::production_moql_accepted_state_authority()?,
        Arc::new(move |peer| {
            peer.subject
                .as_ref()
                .and_then(|did| tenants.get(did))
                .cloned()
        }),
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
            .with_ingress_authorizer(Arc::new(
                move |peer: &hyprstream_rpc::moq_authz::PeerIdentity, tenant: &str| {
                    tenant == "local"
                        && peer
                            .subject
                            .as_ref()
                            .is_some_and(|did| publishers.contains(did))
                },
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
        runtime.block_on(async move {
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
                _ = shutdown.notified() => Err(RpcError::SpawnFailed("Event stopped before announcement".into())),
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
            substrate.shutdown().await.map_err(|e| RpcError::SpawnFailed(format!("Event shutdown: {e}")))
        })
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
        require_event_tenant(&config.quic, &proof)?;
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
}
