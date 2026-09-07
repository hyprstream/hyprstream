//! Real signed admission/checkpoints and hybrid Iroh RPC, isolated from the
//! process-global test fixtures used by other transport tests.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use super::*;
use crate::checkpointed_pds::{write_test_state, CheckpointedPdsAcceptedStateSource};
use hyprstream_pds::at9p::{CapsuleBody, HybridKeyPair, ServiceEndpoint, ServiceEntry, ServiceType, Transport};
use hyprstream_pds::at9p_duplicity::{AcceptedAt9pState, DuplicityGuard, InMemoryWatermarkStore};
use hyprstream_pds::at9p_gate::verify_genesis_capsule;
use hyprstream_pds::at9p_sign::{sign_capsule, sign_update_record};
use hyprstream_rpc::node_identity::{derive_mesh_kem_recipient, derive_mesh_mldsa_key, derive_purpose_key};
use hyprstream_rpc::transport::iroh_substrate::{IrohSubstrate, RefuseHandler};

fn admitted(service: &str, signer: &SigningKey) -> Result<AcceptedAt9pState> {
    let pq = derive_mesh_mldsa_key(signer);
    let key = HybridKeyPair::new(signer.verifying_key().to_bytes().to_vec(),
        hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq))?;
    let carrier = derive_purpose_key(signer, "hyprstream-iroh-transport-v1");
    let mut endpoint = ServiceEndpoint::new(Transport::Iroh,
        format!("iroh://{}", hex::encode(carrier.verifying_key().to_bytes())))?;
    endpoint.request_kem = Some(derive_mesh_kem_recipient(signer)?.public().encode());
    let mut body = CapsuleBody::new(vec![key.clone()],
        vec![ServiceEntry::new(format!("#{service}"), ServiceType::NinePExport, endpoint)?])?;
    body.next_key_commitments = vec![key.commitment_digest()];
    let genesis = sign_capsule(body.clone(), signer, &pq)?;
    let bytes = genesis.to_dag_cbor()?;
    let verified = verify_genesis_capsule(&genesis.cid512()?, &bytes)?;
    let guard = DuplicityGuard::new(InMemoryWatermarkStore::default());
    guard.seed_genesis(&verified)?;
    let predecessor = guard.accepted_state(verified.cid512())?.unwrap();
    let update = sign_update_record(verified.cid512().to_owned(), 1, predecessor.head_digest,
        body, (chrono::Utc::now() + chrono::Duration::hours(1)).to_rfc3339_opts(chrono::SecondsFormat::Secs, true), signer, &pq)?;
    guard.admit_successor(&update, &chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true))?;
    Ok(guard.accepted_state(verified.cid512())?.unwrap())
}

fn announcement(state: &AcceptedAt9pState, name: &str, signer: &SigningKey, ca: &SigningKey) -> ServiceAnnouncement {
    let claims = hyprstream_rpc::auth::Claims::new(format!("service:{name}"),
        chrono::Utc::now().timestamp(), chrono::Utc::now().timestamp() + 3600)
        .with_cnf_jwk(signer.verifying_key().as_bytes());
    ServiceAnnouncement {
        service_name: name.to_owned(), socket_kind: "iroh".to_owned(),
        endpoint: state.current.services[0].endpoint.address.clone(),
        service_jwt: Some(hyprstream_rpc::auth::jwt::encode_service_jwt(&claims, ca)),
        service_did: Did::from(state.did.clone()), capabilities: vec!["hyprstream-rpc/1".to_owned()],
        accepted_state_digest: state.head_digest.to_vec(), accepted_state_epoch: state.epoch,
        response_key_id: format!("{}#response", state.did), request_kem_key_id: format!("{}#mesh-kem", state.did),
        request_kem_recipient: state.current.services[0].endpoint.request_kem.clone().unwrap(),
        expires_at_unix_ms: accepted_expiry_unix_ms(state).unwrap(),
    }
}

#[test]
fn required_network_bootstrap_real_checkpoint_and_iroh() -> Result<()> {
    const CHILD: &str = "HYPRSTREAM_NATIVE_BOOTSTRAP_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "service::network_bootstrap_tests::required_network_bootstrap_real_checkpoint_and_iroh", "--nocapture"])
            .env(CHILD, "1").status()?;
        anyhow::ensure!(status.success(), "isolated Iroh bootstrap regression failed");
        return Ok(());
    }
    PROCESS_NATIVE_NETWORK_REQUIRED.set(true).expect("isolated required profile");
    tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?
        .block_on(network_roundtrip())
}

async fn network_roundtrip() -> Result<()> {
    hyprstream_rpc::registry::init(hyprstream_rpc::registry::EndpointMode::Inproc, None);
    use hyprstream_rpc::auth::mac::{MacDispatchPep, MacDecision, install_mac_dispatch_pep};
    struct FixturePep;
    impl MacDispatchPep for FixturePep {
        fn check(&self, _ctx: &EnvelopeContext, service: &str, _method: Option<u16>) -> MacDecision {
            if service == "discovery" { MacDecision::Permit } else { MacDecision::Deny(hyprstream_rpc::auth::mac::MacDenyReason::UnlabeledObject) }
        }
    }
    install_mac_dispatch_pep(Arc::new(FixturePep));
    let discovery = SigningKey::from_bytes(&[0x51; 32]);
    let policy = SigningKey::from_bytes(&[0x52; 32]);
    let model = SigningKey::from_bytes(&[0x53; 32]);
    let registry = SigningKey::from_bytes(&[0x54; 32]);
    let event = SigningKey::from_bytes(&[0x55; 32]);
    let event_state = admitted("event", &event)?;
    let discovery_state = admitted("discovery", &discovery)?;
    let policy_state = admitted("policy", &policy)?;
    let model_state = admitted("model", &model)?;
    let dir = tempfile::tempdir()?;
    for state in [&discovery_state, &policy_state, &model_state, &event_state] {
        write_test_state(dir.path(), state, &registry)?;
    }
    let source = Arc::new(CheckpointedPdsAcceptedStateSource::open_test(dir.path(), registry.verifying_key())?
        .with_network_bootstrap(true));
    for (name, key) in [("discovery", &discovery), ("policy", &policy)] {
        hyprstream_service::global_trust_store().insert(key.verifying_key(), hyprstream_service::Attestation {
            scopes: [name.to_owned()].into_iter().collect(), subject: None, jwt: None,
            expires_at: 0, attested_by: None,
        });
    }
    assert!(CheckpointedPdsAcceptedStateSource::open_test(dir.path(), model.verifying_key())?
        .accepted_state(&discovery_state.did).is_err());
    assert!(project_bootstrap_endpoint(&[], "discovery", &discovery.verifying_key()).is_err());
    assert!(project_bootstrap_endpoint(std::slice::from_ref(&discovery_state), "discovery", &model.verifying_key()).is_err());
    let mut invalid = discovery_state.clone();
    invalid.current.services[0].endpoint.request_kem = None;
    assert!(project_bootstrap_endpoint(&[invalid], "discovery", &discovery.verifying_key()).is_err());

    let state_store = MemoryStateStore::production_default();
    let mut service = DiscoveryService::new(Arc::new(discovery.clone()), policy.verifying_key(),
        TransportConfig::ipc(dir.path().join("must-not-exist.sock")))
        .with_accepted_state_source(source.clone());
    service.state_store = state_store.clone();
    let owner = service.self_announcer()?;
    let bootstrap_resolver = Arc::new(DiscoveryServiceResolver {
        state_store: MemoryStateStore::production_default(), accepted_state_source: source.clone(), discovery_client: None,
    });
    // Resolving fixed roles is authenticated offline work, with no sockets or
    // Policy service yet running; ordinary services have no such bypass.
    bootstrap_resolver.resolve_service(ServiceQuery::network("policy")?).await?;
    assert!(bootstrap_resolver.resolve_service(ServiceQuery::network("model")?).await.is_err());
    assert!(state_store.announcements_for("discovery", unix_millis_now()).await?.is_empty());

    let bridge = hyprstream_rpc::transport::iroh_rpc::LocalServiceBridge::spawn(service,
        Arc::new(hyprstream_rpc::envelope::InMemoryNonceCache::new()), 0)?;
    let carrier_key = derive_purpose_key(&discovery, "hyprstream-iroh-transport-v1");
    let server = IrohSubstrate::new(carrier_key.to_bytes(), RefuseHandler::new("test has no event runtime"),
        hyprstream_rpc::transport::iroh_rpc::IrohRpcProtocolHandler::new(bridge, discovery.clone())).await?;
    let self_announcement = announcement(&discovery_state, "discovery", &discovery, &policy);
    assert_eq!(server.endpoint_id().as_bytes(), &carrier_key.verifying_key().to_bytes());
    owner.publish(&self_announcement).await?;
    assert_eq!(state_store.announcements_for("discovery", unix_millis_now()).await?.len(), 1);
    assert!(!dir.path().join("must-not-exist.sock").exists());
    let mut bad = self_announcement.clone();
    bad.service_jwt = announcement(&discovery_state, "discovery", &discovery, &model).service_jwt;
    assert!(owner.publish(&bad).await.is_err());
    bad = self_announcement.clone();
    bad.endpoint = format!("iroh://{}", hex::encode([0x62; 32]));
    assert!(owner.publish(&bad).await.is_err());

    let client = IrohSubstrate::new([0x63; 32], RefuseHandler::new("client only"), RefuseHandler::new("client only")).await?;
    let lookup = iroh::address_lookup::memory::MemoryLookup::new();
    lookup.add_endpoint_info(iroh::EndpointAddr::from_parts(server.endpoint_id(),
        server.endpoint().bound_sockets().into_iter().map(iroh::TransportAddr::Ip)));
    client.endpoint().address_lookup()?.add(lookup);
    let _ = hyprstream_rpc::transport::lazy_iroh::install_iroh_client_endpoint(client.owned_client_endpoint());
    let mut request_keys = hyprstream_rpc::envelope::KeyedPqTrustStore::new();
    let model_pq = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
        &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&derive_mesh_mldsa_key(&model)))?;
    request_keys.bind(model.verifying_key().to_bytes(), &model_pq);
    let event_pq = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
        &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&derive_mesh_mldsa_key(&event)))?;
    request_keys.bind(event.verifying_key().to_bytes(), &event_pq);
    let _ = hyprstream_rpc::envelope::install_verify_config(hyprstream_rpc::envelope::EnvelopeVerifyConfig {
        policy: hyprstream_rpc::crypto::CryptoPolicy::Hybrid, pq_store: Some(Arc::new(request_keys)),
    });
    let discovery_client = crate::DiscoveryClient::new(Arc::new(ProductionRpcClient::new(
        "discovery", "discovery", None, model.clone(), None, bootstrap_resolver.clone(),
    )?));
    tokio::time::timeout(Duration::from_secs(10), discovery_client.announce(&announcement(&model_state, "model", &model, &policy))).await??;
    let resolver = DiscoveryServiceResolver {
        state_store: MemoryStateStore::production_default(), accepted_state_source: source.clone(),
        discovery_client: Some(discovery_client),
    };
    let resolved = tokio::time::timeout(Duration::from_secs(10), resolver.resolve_service(ServiceQuery::network("model")?)).await??;
    assert_eq!(resolved.service_did().as_str(), model_state.did);
    assert_eq!(resolved.response_verifying_key(), model.verifying_key());
    assert_eq!(resolved.request_kem_recipient.recipient.encode(), model_state.current.services[0].endpoint.request_kem.clone().unwrap());
    // The Event barrier advertises only MoQ. Its authenticated Discovery
    // announcement resolves under the required carrier profile, and cannot
    // accidentally satisfy the RPC capability query used by ordinary services.
    let event_carrier = derive_purpose_key(&event, "hyprstream-iroh-transport-v1");
    let event_endpoint = IrohSubstrate::new(event_carrier.to_bytes(), RefuseHandler::new("reach fixture"), RefuseHandler::new("Event has no RPC")).await?;
    let mut event_announcement = announcement(&event_state, "event", &event, &policy);
    event_announcement.capabilities = vec!["hyprstream-moq/1".to_owned()];
    let event_discovery_client = crate::DiscoveryClient::new(Arc::new(ProductionRpcClient::new(
        "discovery", "discovery", None, event.clone(), None, bootstrap_resolver,
    )?));
    event_discovery_client.announce(&event_announcement).await?;
    assert!(resolver.resolve_service(ServiceQuery::network("event")?).await.is_err());
    let resolved_event = resolver.resolve_service(ServiceQuery::network_moq("event")?).await?;
    resolver.ensure_current(&resolved_event).await?;
    assert_eq!(resolved_event.service_did().as_str(), event_state.did);
    assert!(matches!(resolved_event.transport().endpoint, hyprstream_rpc::transport::EndpointType::Iroh { node_id, .. }
        if node_id == *event_endpoint.endpoint_id().as_bytes()));
    event_endpoint.shutdown().await?;
    // Loss of the authenticated checkpoint is not repaired by a cached dial
    // or a fresh self heartbeat.
    let db = rocksdb::DB::open_for_read_only(&rocksdb::Options::default(), dir.path(), false)?;
    drop(db);
    let db = rocksdb::DB::open(&rocksdb::Options::default(), dir.path())?;
    db.delete(format!("at9p-checkpoint\0{}", discovery_state.subject_cid512))?;
    drop(db);
    assert!(owner.publish(&self_announcement).await.is_err());
    client.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}
