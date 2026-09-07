//! Exercise the actual OAuth runtime client seam with authenticated deployment
//! checkpoints, hybrid RPC, and no process-local Policy or Discovery peers.
#![allow(clippy::expect_used, clippy::unwrap_used)]

#[path = "../../../tests/fixtures/did_trust.rs"]
mod did_trust;

use super::runtime_clients;
use anyhow::Result;
use ed25519_dalek::SigningKey;
use hyprstream_rpc::node_identity::{derive_mesh_mldsa_key, derive_purpose_key};
use hyprstream_rpc::transport::iroh_rpc::{IrohRpcProtocolHandler, LocalServiceBridge};
use hyprstream_rpc::transport::iroh_substrate::{IrohSubstrate, RefuseHandler};
use hyprstream_rpc::transport::TransportConfig;
use std::sync::Arc;

#[test]
fn required_oauth_runtime_clients_reach_policy_and_discovery_over_iroh() -> Result<()> {
    const CHILD: &str = "HYPRSTREAM_REQUIRED_OAUTH_CLIENT_TEST";
    if std::env::var_os(CHILD).is_none() {
        let status = std::process::Command::new(std::env::current_exe()?)
            .args([
                "--exact",
                "services::oauth::required_consumer_tests::required_oauth_runtime_clients_reach_policy_and_discovery_over_iroh",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .status()?;
        anyhow::ensure!(
            status.success(),
            "isolated OAuth consumer regression failed"
        );
        return Ok(());
    }
    let directory = tempfile::Builder::new()
        .prefix(".required-oauth-clients-")
        .tempdir_in(env!("CARGO_MANIFEST_DIR"))?;
    let trust = directory.path().join("trust");
    let credentials = directory.path().join("credentials");
    let secrets = directory.path().join("secrets");
    std::fs::create_dir_all(&trust)?;
    std::fs::create_dir_all(&credentials)?;
    std::env::set_var("HYPRSTREAM_DEPLOYMENT_TRUST_DIR", &trust);
    std::env::set_var("CREDENTIALS_DIRECTORY", &credentials);
    std::env::set_var("XDG_DATA_HOME", directory.path().join("data"));
    std::env::set_var("HYPRSTREAM_INSTANCE", "required-oauth-clients");
    std::env::set_var("HYPRSTREAM__SECRETS__PATH", &secrets);

    let ca = SigningKey::from_bytes(&[0x71; 32]);
    let registry = SigningKey::from_bytes(&[0x72; 32]);
    let policy = SigningKey::from_bytes(&[0x73; 32]);
    let discovery = SigningKey::from_bytes(&[0x74; 32]);
    let oauth = SigningKey::from_bytes(&[0x75; 32]);
    let mut root = ca.verifying_key().to_bytes().to_vec();
    root.extend(hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(
        &hyprstream_rpc::crypto::pq::ml_dsa_sk_from_seed(&ca.to_bytes()),
    ));
    std::fs::write(trust.join("deployment-ca.hybrid"), root)?;
    let (log, checkpoint) = did_trust::authority_log_and_checkpoint(&ca);
    std::fs::write(
        trust.join("deployment-authority.log.json"),
        serde_json::to_vec(&log)?,
    )?;
    std::fs::write(
        trust.join("deployment-authority.head.json"),
        serde_json::to_vec(&checkpoint)?,
    )?;
    std::fs::write(
        credentials.join("registry-service.jwt"),
        did_trust::mint_credential(&ca, &registry),
    )?;
    for (name, signer) in [
        ("registry", &registry),
        ("policy", &policy),
        ("discovery", &discovery),
        ("oauth", &oauth),
    ] {
        let key_dir = if name == "policy" {
            secrets.clone()
        } else {
            secrets.join(name)
        };
        crate::auth::identity_store::write_secret(&key_dir, "signing-key", &signer.to_bytes())?;
        hyprstream_service::global_trust_store().insert(
            signer.verifying_key(),
            hyprstream_service::Attestation {
                scopes: [name.to_owned()].into_iter().collect(),
                subject: None,
                jwt: None,
                expires_at: 0,
                attested_by: None,
            },
        );
    }
    hyprstream_rpc::registry::init(
        hyprstream_rpc::registry::EndpointMode::Ipc,
        Some(directory.path().join("rpc")),
    );
    hyprstream_rpc::transport::pq_provider::install_pq_crypto_provider()?;
    crate::mac::install_explicit_test_dispatch_pep();
    let mut request_keys = hyprstream_rpc::envelope::KeyedPqTrustStore::new();
    for signer in [&oauth, &policy, &discovery] {
        let pq = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
            &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&derive_mesh_mldsa_key(signer)),
        )?;
        request_keys.bind(signer.verifying_key().to_bytes(), &pq);
    }
    let _ = hyprstream_rpc::envelope::install_verify_config(
        hyprstream_rpc::envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(Arc::new(request_keys)),
        },
    );
    hyprstream_discovery::initialize_deployment_checkpoint_store()?;
    let mut config = crate::config::HyprConfig::default();
    config.secrets.path = Some(secrets);
    crate::cli::deployment_bootstrap::provision_services(
        &config,
        &[
            "policy".to_owned(),
            "discovery".to_owned(),
            "oauth".to_owned(),
        ],
        3600,
    )?;

    // Match OAuthService::run: clients are created and used on its own
    // current-thread runtime/LocalSet, after process bootstrap is installed.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    tokio::task::LocalSet::new().block_on(&runtime, async {
        let mcp_config = crate::services::McpConfig {
            verifying_key: policy.verifying_key(),
            signing_key: oauth.clone(),
            transport: TransportConfig::ipc(directory.path().join("mcp-unused.sock")),
            ctx: None,
            policy_transport: TransportConfig::ipc(directory.path().join("policy-unused.sock")),
            policy_verifying_key: policy.verifying_key(),
            expected_audience: None,
            jwt_key_source: None,
        };
        assert!(!hyprstream_discovery::native_network_required());
        assert!(
            runtime_clients(&oauth).is_err(),
            "Compatibility still requires explicitly registered local peers"
        );
        // Compatibility dials the factory-resolved typed IPC transport
        // directly, so construction is registry-free; the process-local
        // registry fallbacks below stay refused.
        assert!(crate::services::McpService::new(mcp_config.clone()).is_ok());
        let client_carrier = IrohSubstrate::new(
            [0x76; 32],
            RefuseHandler::new("client only"),
            RefuseHandler::new("client only"),
        )
        .await?;
        // Deterministic loopback address hints affect transport reach only;
        // all endpoint authority comes from the real OS-authenticated store.
        let lookup = iroh::address_lookup::memory::MemoryLookup::new();
        client_carrier
            .endpoint()
            .address_lookup()?
            .add(lookup.clone());
        let _ = hyprstream_rpc::transport::lazy_iroh::install_iroh_client_endpoint(
            client_carrier.owned_client_endpoint(),
        );
        hyprstream_discovery::bootstrap_deployment_process(
            oauth.clone(),
            hyprstream_discovery::DeploymentTrustSource::OsOwnedFiles,
            false,
            true,
        )
        .await?;
        assert!(hyprstream_discovery::native_network_required());

        let manager = Arc::new(crate::auth::PolicyManager::new_in_memory().await?);
        manager
            .add_policy_with_domain("service:oauth", "*", "model:allowed", "query", "allow")
            .await?;
        let database = Arc::new(tokio::sync::RwLock::new(
            git2db::Git2DB::open(&directory.path().join("models")).await?,
        ));
        let policy_socket = directory.path().join("policy-must-not-exist.sock");
        let discovery_socket = directory.path().join("discovery-must-not-exist.sock");
        let policy_service = crate::services::PolicyService::new(
            manager,
            Arc::new(policy.clone()),
            crate::config::TokenConfig::default(),
            database,
            TransportConfig::ipc(&policy_socket),
        );
        let discovery_service = hyprstream_discovery::DiscoveryService::new(
            Arc::new(discovery.clone()),
            policy.verifying_key(),
            TransportConfig::ipc(&discovery_socket),
        );
        let policy_bridge = LocalServiceBridge::spawn(
            policy_service,
            Arc::new(hyprstream_rpc::envelope::InMemoryNonceCache::new()),
            0,
        )?;
        let discovery_bridge = LocalServiceBridge::spawn(
            discovery_service,
            Arc::new(hyprstream_rpc::envelope::InMemoryNonceCache::new()),
            0,
        )?;
        let policy_server = IrohSubstrate::new(
            derive_purpose_key(&policy, "hyprstream-iroh-transport-v1").to_bytes(),
            RefuseHandler::new("no events"),
            IrohRpcProtocolHandler::new(policy_bridge, policy.clone()),
        )
        .await?;
        let discovery_server = IrohSubstrate::new(
            derive_purpose_key(&discovery, "hyprstream-iroh-transport-v1").to_bytes(),
            RefuseHandler::new("no events"),
            IrohRpcProtocolHandler::new(discovery_bridge, discovery.clone()),
        )
        .await?;
        for server in [&policy_server, &discovery_server] {
            lookup.add_endpoint_info(iroh::EndpointAddr::from_parts(
                server.endpoint_id(),
                server
                    .endpoint()
                    .bound_sockets()
                    .into_iter()
                    .map(iroh::TransportAddr::Ip),
            ));
        }
        assert_no_local_peers();
        assert!(crate::services::PolicyClient::for_local_bootstrap(
            oauth.clone(),
            policy.verifying_key(),
            None
        )
        .is_err());
        assert!(crate::services::DiscoveryClient::for_local_bootstrap(
            oauth.clone(),
            discovery.verifying_key(),
            None
        )
        .is_err());
        let (policy_client, discovery_client) = runtime_clients(&oauth)?;
        let _mcp = crate::services::McpService::new(mcp_config)?;
        tokio::time::timeout(std::time::Duration::from_secs(15), async {
            let check = |resource: &str| crate::services::generated::policy_client::PolicyCheck {
                subject: "forged-caller".to_owned(),
                domain: "forged-domain".to_owned(),
                resource: resource.to_owned(),
                operation: "query".to_owned(),
            };
            assert!(policy_client.check(&check("model:allowed")).await?);
            assert!(!policy_client.check(&check("model:denied")).await?);
            let _ = discovery_client.list_services().await?;
            anyhow::Ok(())
        })
        .await??;
        assert_no_local_peers();
        assert!(!policy_socket.exists());
        assert!(!discovery_socket.exists());
        client_carrier.shutdown().await?;
        policy_server.shutdown().await?;
        discovery_server.shutdown().await?;
        anyhow::Ok(())
    })
}

fn assert_no_local_peers() {
    let registry = hyprstream_rpc::registry::global();
    for name in ["policy", "discovery"] {
        assert!(registry
            .registered_endpoint(name, hyprstream_rpc::SocketKind::Rep)
            .is_none());
    }
}
