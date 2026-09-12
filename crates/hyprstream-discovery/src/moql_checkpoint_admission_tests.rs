//! Real process-bootstrap checkpoint authority composed with Iroh/MoQ payload
//! admission. Kept process-isolated because the production authority is a
//! one-shot process singleton.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use crate::checkpointed_pds::write_test_state;
use bytes::Bytes;
use hyprstream_pds::at9p::{
    CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport,
};
use hyprstream_pds::at9p_duplicity::{AcceptedAt9pState, DuplicityGuard, InMemoryWatermarkStore};
use hyprstream_pds::at9p_gate::verify_genesis_capsule;
use hyprstream_pds::at9p_sign::{sign_capsule, sign_update_record};
use hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes;
use hyprstream_rpc::moq_authz::PeerIdentity;
use hyprstream_rpc::node_identity::{derive_mesh_mldsa_key, derive_purpose_key};
use hyprstream_rpc::rpc_client::{CallOptions, RpcClient};
use hyprstream_rpc::stream_consumer::StreamHandle;
use hyprstream_rpc::transport::iroh_moq::{IrohMoqProtocolHandler, MoqAuthzConfig, OriginShared};
use hyprstream_rpc::transport::iroh_substrate::{IrohSubstrate, RefuseHandler, ALPN_MOQ_LITE};
use hyprstream_rpc::transport::moql_admission::{
    prove_moql_admission, MoqlAdmissionAuthenticator, MoqlAdmissionProof, MoqlServerIdentityProof,
};
use iroh::{EndpointAddr, TransportAddr};
use moq_net::{Client, Group, Origin, Track};
use std::collections::BTreeMap;
use web_transport_iroh::Session;

const IO_TIMEOUT: Duration = Duration::from_secs(5);

struct NoopBootstrapClient;

#[async_trait::async_trait]
impl RpcClient for NoopBootstrapClient {
    async fn call(&self, _: Vec<u8>) -> Result<Vec<u8>> {
        anyhow::bail!("bootstrap transport was unexpectedly dispatched")
    }
    async fn call_for_service(&self, _: &str, _: Vec<u8>) -> Result<Vec<u8>> {
        anyhow::bail!("bootstrap transport was unexpectedly dispatched")
    }
    async fn call_for_service_with_method(&self, _: &str, _: u16, _: Vec<u8>) -> Result<Vec<u8>> {
        anyhow::bail!("bootstrap transport was unexpectedly dispatched")
    }
    async fn call_with_options(&self, _: Vec<u8>, _: CallOptions) -> Result<Vec<u8>> {
        anyhow::bail!("bootstrap transport was unexpectedly dispatched")
    }
    async fn call_with_options_for_service(
        &self,
        _: &str,
        _: Vec<u8>,
        _: CallOptions,
    ) -> Result<Vec<u8>> {
        anyhow::bail!("bootstrap transport was unexpectedly dispatched")
    }
    async fn call_streaming(&self, _: Vec<u8>, _: [u8; 32]) -> Result<Vec<u8>> {
        anyhow::bail!("bootstrap transport was unexpectedly dispatched")
    }
    async fn call_streaming_for_service(
        &self,
        _: &str,
        _: Vec<u8>,
        _: [u8; 32],
    ) -> Result<Vec<u8>> {
        anyhow::bail!("bootstrap transport was unexpectedly dispatched")
    }
    async fn call_streaming_for_service_with_method(
        &self,
        _: &str,
        _: u16,
        _: Vec<u8>,
        _: [u8; 32],
    ) -> Result<Vec<u8>> {
        anyhow::bail!("bootstrap transport was unexpectedly dispatched")
    }
    async fn open_stream(&self, _: Vec<u8>) -> Result<Box<dyn StreamHandle>> {
        anyhow::bail!("bootstrap transport was unexpectedly dispatched")
    }
    async fn open_stream_from_info(
        &self,
        _: hyprstream_rpc::stream_info::StreamInfo,
        _: [u8; 32],
        _: [u8; 32],
    ) -> Result<Box<dyn StreamHandle>> {
        anyhow::bail!("bootstrap transport was unexpectedly dispatched")
    }
    fn next_id(&self) -> u64 {
        1
    }
}

fn service_body(service: &str, signer: &SigningKey) -> Result<CapsuleBody> {
    let pq = derive_mesh_mldsa_key(signer);
    let key = HybridKeyPair::new(
        signer.verifying_key().to_bytes().to_vec(),
        ml_dsa_sk_to_vk_bytes(&pq),
    )?;
    let carrier = derive_purpose_key(signer, "hyprstream-iroh-transport-v1");
    let endpoint = ServiceEndpoint::new(
        Transport::Iroh,
        format!("iroh://{}", hex::encode(carrier.verifying_key().to_bytes())),
    )?;
    let mut body = CapsuleBody::new(
        vec![key.clone()],
        vec![ServiceEntry::new(
            format!("#{service}"),
            ServiceType::NinePExport,
            endpoint,
        )?],
    )?;
    body.next_key_commitments = vec![key.commitment_digest()];
    Ok(body)
}

fn admitted(service: &str, signer: &SigningKey) -> Result<AcceptedAt9pState> {
    let pq = derive_mesh_mldsa_key(signer);
    let body = service_body(service, signer)?;
    let genesis = sign_capsule(body.clone(), signer, &pq)?;
    let bytes = genesis.to_dag_cbor()?;
    let verified = verify_genesis_capsule(&genesis.cid512()?, &bytes)?;
    let guard = DuplicityGuard::new(InMemoryWatermarkStore::default());
    guard.seed_genesis(&verified)?;
    let predecessor = guard.accepted_state(verified.cid512())?.unwrap();
    let update = sign_update_record(
        verified.cid512().to_owned(),
        1,
        predecessor.head_digest,
        body,
        expiry(),
        signer,
        &pq,
    )?;
    guard.admit_successor(&update, &now())?;
    Ok(guard.accepted_state(verified.cid512())?.unwrap())
}

fn admitted_with_rotation(
    service: &str,
    signer: &SigningKey,
    rotated: &SigningKey,
) -> Result<(AcceptedAt9pState, AcceptedAt9pState)> {
    let pq = derive_mesh_mldsa_key(signer);
    let genesis = sign_capsule(service_body(service, signer)?, signer, &pq)?;
    let bytes = genesis.to_dag_cbor()?;
    let verified = verify_genesis_capsule(&genesis.cid512()?, &bytes)?;
    let guard = DuplicityGuard::new(InMemoryWatermarkStore::default());
    guard.seed_genesis(&verified)?;
    let predecessor = guard.accepted_state(verified.cid512())?.unwrap();
    let mut first_body = service_body(service, signer)?;
    let rotated_key = HybridKeyPair::new(
        rotated.verifying_key().to_bytes().to_vec(),
        ml_dsa_sk_to_vk_bytes(&derive_mesh_mldsa_key(rotated)),
    )?;
    first_body.next_key_commitments = vec![rotated_key.commitment_digest()];
    let first = sign_update_record(
        verified.cid512().to_owned(),
        1,
        predecessor.head_digest,
        first_body,
        expiry(),
        signer,
        &pq,
    )?;
    guard.admit_successor(&first, &now())?;
    let current = guard.accepted_state(verified.cid512())?.unwrap();
    let rotated_pq = derive_mesh_mldsa_key(rotated);
    let second = sign_update_record(
        verified.cid512().to_owned(),
        2,
        current.head_digest,
        service_body(service, rotated)?,
        expiry(),
        rotated,
        &rotated_pq,
    )?;
    guard.admit_successor(&second, &now())?;
    let advanced = guard.accepted_state(verified.cid512())?.unwrap();
    Ok((current, advanced))
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn expiry() -> String {
    (chrono::Utc::now() + chrono::Duration::hours(1))
        .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn public_identity(
    state: &AcceptedAt9pState,
    signer: &SigningKey,
) -> Result<hyprstream_rpc::stream_info::MoqlServerIdentity> {
    Ok(hyprstream_rpc::stream_info::MoqlServerIdentity {
        did: state.did.clone(),
        epoch: state.epoch,
        head_digest: state.head_digest.to_vec(),
        expires_at_unix_ms: accepted_expiry_unix_ms(state)?,
        ed25519: signer.verifying_key().to_bytes(),
        ml_dsa65: ml_dsa_sk_to_vk_bytes(&derive_mesh_mldsa_key(signer)),
    })
}

fn proof(
    state: &AcceptedAt9pState,
    signer: &SigningKey,
    expected_server: hyprstream_rpc::stream_info::MoqlServerIdentity,
) -> MoqlAdmissionProof {
    MoqlAdmissionProof {
        did: state.did.clone(),
        ed25519: signer.clone(),
        ml_dsa_65: derive_mesh_mldsa_key(signer),
        expected_server,
    }
}

fn direct(server: &IrohSubstrate) -> EndpointAddr {
    EndpointAddr::from_parts(
        server.endpoint_id(),
        server
            .endpoint()
            .bound_sockets()
            .into_iter()
            .map(TransportAddr::Ip),
    )
}

async fn client(seed: u8) -> Result<IrohSubstrate> {
    IrohSubstrate::new(
        [seed; 32],
        RefuseHandler::new("client has no moq server"),
        RefuseHandler::new("client has no rpc server"),
    )
    .await
}

#[test]
fn installed_checkpoint_authority_controls_real_moq_payload() -> Result<()> {
    const CHILD: &str = "HYPRSTREAM_CHECKPOINT_MOQL_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let deployment = tempfile::tempdir()?;
        let status = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "service::moql_checkpoint_admission_tests::installed_checkpoint_authority_controls_real_moq_payload",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("XDG_DATA_HOME", deployment.path())
            .status()?;
        anyhow::ensure!(
            status.success(),
            "isolated checkpoint MoQ regression failed"
        );
        return Ok(());
    }

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?
        .block_on(roundtrip())
}

async fn roundtrip() -> Result<()> {
    let registry = SigningKey::from_bytes(&[0x71; 32]);
    let server_signer = SigningKey::from_bytes(&[0x72; 32]);
    let client_signer = SigningKey::from_bytes(&[0x73; 32]);
    let rotated_client = SigningKey::from_bytes(&[0x74; 32]);
    let unmapped_signer = SigningKey::from_bytes(&[0x75; 32]);
    let unknown_signer = SigningKey::from_bytes(&[0x76; 32]);
    let wrong_signer = SigningKey::from_bytes(&[0x77; 32]);

    let server_state = admitted("streams", &server_signer)?;
    let (client_state, advanced_client_state) =
        admitted_with_rotation("reader", &client_signer, &rotated_client)?;
    let unmapped_state = admitted("unmapped", &unmapped_signer)?;
    let unknown_state = admitted("unknown", &unknown_signer)?;
    let store = hyprstream_service::deployment_data_dir()?.join("pds-store");
    std::fs::create_dir_all(&store)?;
    for state in [&server_state, &client_state, &unmapped_state] {
        write_test_state(&store, state, &registry)?;
    }

    let mut bootstrap = authenticate_discovery_bootstrap_identity(registry.verifying_key())?;
    bootstrap.network_required = true;
    DiscoveryService::bootstrap_authenticated_process(
        bootstrap,
        crate::DiscoveryClient::new(Arc::new(NoopBootstrapClient)),
    )?;
    let authority = production_moql_accepted_state_authority()?;
    let tenants = BTreeMap::from([
        (server_state.did.clone(), "local".to_owned()),
        (client_state.did.clone(), "local".to_owned()),
    ]);
    let resolver = Arc::new(move |peer: &PeerIdentity| {
        peer.subject
            .as_deref()
            .and_then(|did| tenants.get(did))
            .cloned()
    });
    let server_identity = public_identity(&server_state, &server_signer)?;
    let server_identity_proof = MoqlServerIdentityProof {
        identity: server_identity.clone(),
        ed25519: server_signer.clone(),
        ml_dsa_65: derive_mesh_mldsa_key(&server_signer),
    };
    let carrier = derive_purpose_key(&server_signer, "hyprstream-iroh-transport-v1");
    let carrier_secret = carrier.to_bytes();
    let carrier_id = carrier.verifying_key().to_bytes();
    let admission = Arc::new(
        MoqlAdmissionAuthenticator::new(authority, resolver)
            .with_server_identity_and_carrier(server_identity_proof, carrier_id)
            .with_timeout(IO_TIMEOUT),
    );
    let origin = OriginShared::new();
    let producer = origin.producer().clone();
    let handler = IrohMoqProtocolHandler::with_origin(origin)
        .with_authz(MoqAuthzConfig::default().with_admission(admission));
    let server = IrohSubstrate::new(
        carrier_secret,
        handler,
        RefuseHandler::new("Streams exposes only authenticated moql"),
    )
    .await?;
    anyhow::ensure!(
        server.endpoint_id().as_bytes() == &carrier_id,
        "server carrier does not use the service-owned transport key"
    );
    let address = direct(&server);

    for (name, candidate) in [
        (
            "unmapped",
            proof(&unmapped_state, &unmapped_signer, server_identity.clone()),
        ),
        (
            "unknown",
            proof(&unknown_state, &unknown_signer, server_identity.clone()),
        ),
        (
            "wrong-current-key",
            proof(&client_state, &wrong_signer, server_identity.clone()),
        ),
    ] {
        let rejected_client = client(match name {
            "unmapped" => 0x31,
            "unknown" => 0x32,
            _ => 0x33,
        })
        .await?;
        let connection = rejected_client
            .connect(address.clone(), ALPN_MOQ_LITE)
            .await?;
        anyhow::ensure!(
            prove_moql_admission(
                &connection,
                &candidate,
                *rejected_client.endpoint_id().as_bytes(),
                IO_TIMEOUT,
            )
            .await
            .is_err(),
            "{name} identity unexpectedly admitted"
        );
        rejected_client.shutdown().await?;
    }

    let admitted_client = client(0x34).await?;
    let connection = admitted_client.connect(address, ALPN_MOQ_LITE).await?;
    let admitted_proof = proof(&client_state, &client_signer, server_identity.clone());
    prove_moql_admission(
        &connection,
        &admitted_proof,
        *admitted_client.endpoint_id().as_bytes(),
        IO_TIMEOUT,
    )
    .await?;
    let client_origin = Origin::random().produce();
    let consumer = client_origin.consume();
    let _session = tokio::time::timeout(
        IO_TIMEOUT,
        Client::new()
            .with_consume(client_origin)
            .connect(Session::raw(connection.clone())),
    )
    .await??;

    let mut broadcast = producer
        .create_broadcast("local/streams/checkpointed")
        .context("create server-owned broadcast")?;
    let mut track = broadcast.create_track(Track::new("tokens"))?;
    let mut group = track.create_group(Group::from(0u64))?;
    group.write_frame(Bytes::from_static(b"checkpoint-authorized payload"))?;
    drop(group);
    let received = tokio::time::timeout(
        IO_TIMEOUT,
        consumer.announced_broadcast("local/streams/checkpointed"),
    )
    .await?
    .context("admitted client did not see server-owned broadcast")?;
    let received_track = received.subscribe_track(&Track::new("tokens"))?;
    let mut received_group = tokio::time::timeout(IO_TIMEOUT, received_track.get_group(0))
        .await??
        .context("admitted client did not receive payload group")?;
    assert_eq!(
        tokio::time::timeout(IO_TIMEOUT, received_group.read_frame()).await??,
        Some(Bytes::from_static(b"checkpoint-authorized payload"))
    );

    anyhow::ensure!(
        tokio::time::timeout(Duration::from_millis(100), connection.closed())
            .await
            .is_err(),
        "admitted connection closed before the checkpoint advanced"
    );
    write_test_state(&store, &advanced_client_state, &registry)?;
    tokio::time::timeout(IO_TIMEOUT, connection.closed()).await?;
    let retry = client(0x35).await?;
    let retry_connection = retry.connect(direct(&server), ALPN_MOQ_LITE).await?;
    assert!(
        prove_moql_admission(
            &retry_connection,
            &admitted_proof,
            *retry.endpoint_id().as_bytes(),
            IO_TIMEOUT,
        )
        .await
        .is_err(),
        "proof using the pre-advance current key was admitted"
    );

    retry.shutdown().await?;
    let rotated = client(0x36).await?;
    let rotated_connection = rotated.connect(direct(&server), ALPN_MOQ_LITE).await?;
    let rotated_proof = proof(&advanced_client_state, &rotated_client, server_identity);
    prove_moql_admission(
        &rotated_connection,
        &rotated_proof,
        *rotated.endpoint_id().as_bytes(),
        IO_TIMEOUT,
    )
    .await?;
    let rotated_origin = Origin::random().produce();
    let rotated_consumer = rotated_origin.consume();
    let _rotated_session = tokio::time::timeout(
        IO_TIMEOUT,
        Client::new()
            .with_consume(rotated_origin)
            .connect(Session::raw(rotated_connection)),
    )
    .await??;
    let rotated_broadcast = tokio::time::timeout(
        IO_TIMEOUT,
        rotated_consumer.announced_broadcast("local/streams/checkpointed"),
    )
    .await?
    .context("rotated client did not see server-owned broadcast")?;
    let rotated_track = rotated_broadcast.subscribe_track(&Track::new("tokens"))?;
    let mut renewed_group = track.create_group(Group::from(1u64))?;
    renewed_group.write_frame(Bytes::from_static(b"checkpoint-authorized payload"))?;
    drop(renewed_group);
    let mut rotated_group = tokio::time::timeout(IO_TIMEOUT, rotated_track.get_group(1))
        .await??
        .context("rotated client did not receive payload group")?;
    assert_eq!(
        tokio::time::timeout(IO_TIMEOUT, rotated_group.read_frame()).await??,
        Some(Bytes::from_static(b"checkpoint-authorized payload"))
    );

    rotated.shutdown().await?;
    admitted_client.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}
