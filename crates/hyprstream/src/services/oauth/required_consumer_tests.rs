//! Exercise the actual OAuth runtime client seam with authenticated deployment
//! checkpoints, hybrid RPC, and no process-local Policy or Discovery peers.
#![allow(clippy::expect_used, clippy::unwrap_used)]

#[path = "../../../tests/fixtures/did_trust.rs"]
pub(crate) mod did_trust;

use super::runtime_clients;
use super::{build_oauth_substrate_profile, classify_oauth_endpoint_install, OAuthEndpointInstall};
use anyhow::Result;
use ed25519_dalek::SigningKey;
use hyprstream_rpc::node_identity::{derive_mesh_mldsa_key, derive_purpose_key};
use hyprstream_rpc::transport::iroh_rpc::{IrohRpcProtocolHandler, LocalServiceBridge};
use hyprstream_rpc::transport::iroh_substrate::{IrohSubstrate, RefuseHandler};
use hyprstream_rpc::transport::TransportConfig;
use parking_lot::Mutex;
use std::sync::Arc;

struct OaiTestNinePDecider;

impl hyprstream_9p::AccessDecider for OaiTestNinePDecider {
    fn check(
        &self,
        _ctx: &hyprstream_rpc::auth::mac::SecurityContext,
        _object: hyprstream_rpc::auth::mac::ObjectRef<'_>,
        _action: hyprstream_9p::Action,
    ) -> bool {
        true
    }

    fn audit_denial(
        &self,
        _ctx: &hyprstream_rpc::auth::mac::SecurityContext,
        _object: hyprstream_rpc::auth::mac::ObjectRef<'_>,
        _object_label: Option<hyprstream_rpc::auth::mac::SecurityLabel>,
        _action: hyprstream_9p::Action,
        _reason: hyprstream_9p::ReferenceMonitorDenyReason,
    ) {
    }
}

#[derive(Clone)]
enum ModelReadinessResponse {
    ExpectedTenantDenial,
    Error {
        code: String,
        message: String,
        details: String,
    },
    Healthy,
    Block,
}

/// A real encrypted Model RPC responder for causal controls around OAI's
/// startup response classifier. The affirmative tenant-denial case is covered
/// separately with the production `ModelService` handler.
struct ModelReadinessResponder {
    signing_key: Arc<Mutex<SigningKey>>,
    transport: TransportConfig,
    jwt_key_source: Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
    response: Arc<Mutex<ModelReadinessResponse>>,
    entered: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

#[async_trait::async_trait(?Send)]
impl hyprstream_rpc::service::RequestService for ModelReadinessResponder {
    fn decode_request_body(
        &self,
        signed_body: &[u8],
    ) -> Result<hyprstream_rpc::service::DecodedRequestBody> {
        crate::services::generated::model_client::decode_model_request_body(signed_body)
    }

    async fn handle_request(
        &self,
        ctx: &hyprstream_rpc::service::EnvelopeContext,
        body: &hyprstream_rpc::service::DecodedRequestBody,
    ) -> Result<(Vec<u8>, Option<hyprstream_rpc::service::Continuation>)> {
        anyhow::ensure!(
            ctx.subject().name() == Some("service:oai"),
            "readiness caller is not authenticated as service:oai"
        );
        let request = body.root::<crate::model_capnp::model_request::Reader<'_>>()?;
        anyhow::ensure!(
            matches!(
                request.which()?,
                crate::model_capnp::model_request::Which::HealthCheck(())
            ),
            "readiness responder received a non-health Model request"
        );

        let response = self.response.lock().clone();
        if matches!(response, ModelReadinessResponse::Block) {
            self.entered.notify_one();
            self.release.notified().await;
        }
        let variant = match response {
            ModelReadinessResponse::ExpectedTenantDenial => {
                let denial = ctx
                    .domain()
                    .expect_err("tenantless service JWT must not acquire a Model tenant");
                anyhow::ensure!(
                    denial.to_string() == "authorization denied: no verified tenant domain",
                    "unexpected Model tenant denial: {denial}"
                );
                crate::services::generated::model_client::ModelResponseVariant::Error(
                    crate::services::generated::model_client::ErrorInfo {
                        code: "INTERNAL".to_owned(),
                        message: denial.to_string(),
                        details: String::new(),
                    },
                )
            }
            ModelReadinessResponse::Error {
                code,
                message,
                details,
            } => crate::services::generated::model_client::ModelResponseVariant::Error(
                crate::services::generated::model_client::ErrorInfo {
                    code,
                    message,
                    details,
                },
            ),
            ModelReadinessResponse::Healthy => {
                crate::services::generated::model_client::ModelResponseVariant::HealthCheckResult(
                    crate::services::generated::model_client::ModelHealthStatus {
                        status: "healthy".to_owned(),
                        loaded_model_count: 0,
                        max_models: 0,
                        total_memory_bytes: 0,
                    },
                )
            }
            ModelReadinessResponse::Block => {
                crate::services::generated::model_client::ModelResponseVariant::Error(
                    crate::services::generated::model_client::ErrorInfo {
                        code: "INTERNAL".to_owned(),
                        message: "authorization denied: no verified tenant domain".to_owned(),
                        details: String::new(),
                    },
                )
            }
        };
        let response = crate::services::generated::model_client::serialize_response(
            request.get_id(),
            &variant,
        )?;
        Ok((response, None))
    }

    fn name(&self) -> &str {
        "model"
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.lock().clone()
    }

    fn jwt_key_source(&self) -> Option<Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
        Some(Arc::clone(&self.jwt_key_source))
    }
}

/// Minimal real Registry RPC responder for the OAI readiness boundary. It
/// decodes the generated Registry request, receives a verified network
/// envelope through `LocalServiceBridge`, and emits the generated health
/// response; it is not a callback standing in for an RPC.
struct RegistryReadinessResponder {
    signing_key: SigningKey,
    transport: TransportConfig,
    jwt_key_source: Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
    status: Arc<Mutex<String>>,
}

#[async_trait::async_trait(?Send)]
impl hyprstream_rpc::service::RequestService for RegistryReadinessResponder {
    fn decode_request_body(
        &self,
        signed_body: &[u8],
    ) -> Result<hyprstream_rpc::service::DecodedRequestBody> {
        crate::services::generated::registry_client::decode_registry_request_body(signed_body)
    }

    async fn handle_request(
        &self,
        ctx: &hyprstream_rpc::service::EnvelopeContext,
        body: &hyprstream_rpc::service::DecodedRequestBody,
    ) -> Result<(Vec<u8>, Option<hyprstream_rpc::service::Continuation>)> {
        anyhow::ensure!(
            ctx.subject().name() == Some("service:oai"),
            "readiness caller is not authenticated as service:oai"
        );
        let request = body.root::<crate::registry_capnp::registry_request::Reader<'_>>()?;
        anyhow::ensure!(
            matches!(
                request.which()?,
                crate::registry_capnp::registry_request::Which::HealthCheck(())
            ),
            "readiness responder received a non-health Registry request"
        );
        let status = self.status.lock().clone();
        let response = crate::services::generated::registry_client::serialize_response(
            request.get_id(),
            &crate::services::generated::registry_client::RegistryResponseVariant::HealthCheckResult(
                crate::services::generated::registry_client::HealthStatus {
                    status,
                    repository_count: 0,
                    worktree_count: 0,
                    cache_hits: 0,
                    cache_misses: 0,
                },
            ),
        )?;
        Ok((response, None))
    }

    fn name(&self) -> &str {
        "registry"
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    fn jwt_key_source(&self) -> Option<Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
        Some(Arc::clone(&self.jwt_key_source))
    }
}

fn service_announcement(
    name: &str,
    state: &hyprstream_pds::at9p_duplicity::AcceptedAt9pState,
    signer: &SigningKey,
    jwt_signer: &SigningKey,
) -> hyprstream_discovery::ServiceAnnouncement {
    let entry = state
        .current
        .services
        .iter()
        .find(|entry| entry.id == format!("#{name}"))
        .expect("accepted service entry");
    let claims = hyprstream_rpc::auth::Claims::new(
        format!("service:{name}"),
        chrono::Utc::now().timestamp(),
        chrono::Utc::now().timestamp() + 3600,
    )
    .with_cnf_jwk(signer.verifying_key().as_bytes());
    hyprstream_discovery::ServiceAnnouncement {
        service_name: name.to_owned(),
        socket_kind: "iroh".to_owned(),
        endpoint: entry.endpoint.address.clone(),
        service_jwt: Some(hyprstream_rpc::auth::jwt::encode_service_jwt(
            &claims, jwt_signer,
        )),
        service_did: hyprstream_rpc::identity::Did::from(state.did.clone()),
        capabilities: vec!["hyprstream-rpc/1".to_owned()],
        accepted_state_digest: state.head_digest.to_vec(),
        accepted_state_epoch: state.epoch,
        response_key_id: format!("{}#response", state.did),
        request_kem_key_id: format!("{}#mesh-kem", state.did),
        request_kem_recipient: entry
            .endpoint
            .request_kem
            .clone()
            .expect("accepted request KEM"),
        expires_at_unix_ms: chrono::DateTime::parse_from_rfc3339(
            state.expires_at.as_deref().expect("bounded accepted state"),
        )
        .expect("accepted expiry")
        .timestamp_millis(),
    }
}

#[cfg(test)]
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
    let model = SigningKey::from_bytes(&[0x76; 32]);
    let oai = SigningKey::from_bytes(&[0x77; 32]);
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
        ("model", &model),
        ("oai", &oai),
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
    // Production publishes the authority stores before constructing any
    // service factory. This isolated child bypasses main, so install the same
    // fail-closed revocation seam before presenting the jti-bearing OAI WIT.
    if hyprstream_rpc::auth::global_credential_revocation_store().is_none() {
        let _ = hyprstream_rpc::auth::set_global_credential_revocation_store(Arc::new(
            hyprstream_rpc::auth::InMemoryCredentialRevocationStore::new(),
        ));
    }
    assert!(hyprstream_rpc::auth::global_credential_revocation_store().is_some());
    let mut request_keys = hyprstream_rpc::envelope::KeyedPqTrustStore::new();
    for signer in [&oauth, &policy, &discovery, &registry, &model, &oai] {
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
            "registry".to_owned(),
            "policy".to_owned(),
            "discovery".to_owned(),
            "oauth".to_owned(),
            "model".to_owned(),
            "oai".to_owned(),
        ],
        3600,
        None,
    )?;

    const TEST_ISSUER: &str = "https://required-oai.test";
    let oai_jwt = crate::auth::service_jwt::issue_or_load_service_jwt(
        &credentials,
        "oai",
        &ca,
        &crate::auth::identity_store::BootstrapPubkey::for_service_key(&oai)?,
        TEST_ISSUER,
        chrono::Utc::now().timestamp(),
        None,
    )?;
    let registry_jwt = crate::auth::service_jwt::issue_or_load_service_jwt(
        &credentials,
        "registry",
        &ca,
        &crate::auth::identity_store::BootstrapPubkey::for_service_key(&registry)?,
        TEST_ISSUER,
        chrono::Utc::now().timestamp(),
        None,
    )?;
    let ca_pq = derive_mesh_mldsa_key(&ca);
    let ca_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
        &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&ca_pq),
    )?;
    let jwt_key_source: Arc<dyn hyprstream_rpc::auth::JwtKeySource> = Arc::new(
        hyprstream_rpc::auth::ClusterKeySource::new(ca.verifying_key(), TEST_ISSUER.to_owned())
            .with_ca_composite_key(ca_pq_vk),
    );
    hyprstream_service::global_trust_store().insert(
        oai.verifying_key(),
        hyprstream_service::Attestation {
            scopes: ["oai".to_owned()].into_iter().collect(),
            subject: None,
            jwt: Some(oai_jwt.clone()),
            expires_at: chrono::Utc::now().timestamp() + 3600,
            attested_by: None,
        },
    );

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
            [0x78; 32],
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

        // Sol carrier disposition (PR 1585): bind OAuth's own inbound
        // reach-only substrate via the production profile helper AFTER the
        // process-global outbound slot is already occupied. Both install
        // outcomes are valid; the occupied slot is the positive control —
        // classify it ExistingGlobalRetained, RETAIN the substrate as the
        // independent inbound owner, and leave the existing global endpoint
        // untouched. The two carriers' endpoint IDs are deliberately
        // different (distinct transport purpose keys) and are never compared
        // in production.
        let oauth_substrate = build_oauth_substrate_profile(&oauth, true)
            .await?
            .expect("Required profile must bind its mandatory inbound substrate");
        assert!(matches!(
            classify_oauth_endpoint_install(&oauth_substrate),
            OAuthEndpointInstall::ExistingGlobalRetained
        ));
        assert_ne!(
            oauth_substrate.endpoint_id(),
            client_carrier.endpoint_id(),
            "the two carriers' transport purpose keys are deliberately distinct"
        );

        let manager = Arc::new(
            crate::auth::PolicyManager::new(directory.path().join("policies")).await?,
        );
        manager
            .add_policy_with_domain("service:oauth", "*", "model:allowed", "query", "allow")
            .await?;
        manager
            .add_policy_with_domain(
                "service:registry",
                "*",
                "discovery:Announce",
                "write",
                "allow",
            )
            .await?;
        manager
            .add_policy_with_domain("service:model", "*", "discovery:Announce", "write", "allow")
            .await?;
        let database = Arc::new(tokio::sync::RwLock::new(
            git2db::Git2DB::open(&directory.path().join("models")).await?,
        ));
        let policy_socket = directory.path().join("policy-must-not-exist.sock");
        let discovery_socket = directory.path().join("discovery-must-not-exist.sock");
        let policy_service = crate::services::PolicyService::new(
            Arc::clone(&manager),
            Arc::new(policy.clone()),
            crate::config::TokenConfig::default(),
            database,
            TransportConfig::ipc(&policy_socket),
        )
        .with_jwt_key_source(Arc::clone(&jwt_key_source));
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
        let _ = hyprstream_rpc::moq_event::init_global_moq_event_origin(
            hyprstream_rpc::moq_event::MoqEventOrigin::new(),
        );
        let model_service = crate::services::ModelService::new(
            crate::services::ModelServiceConfig::default(),
            model.clone(),
            crate::services::PolicyClient::from_resolver(model.clone(), None)?,
            crate::services::RegistryClient::from_resolver(model.clone(), None)?,
            TransportConfig::ipc(directory.path().join("model-must-not-exist.sock")),
            TransportConfig::ipc(&policy_socket),
        )
        .await?
        .with_jwt_key_source(Arc::clone(&jwt_key_source));
        let model_bridge = LocalServiceBridge::spawn(
            model_service,
            Arc::new(hyprstream_rpc::envelope::InMemoryNonceCache::new()),
            0,
        )?;
        let model_server = IrohSubstrate::new(
            derive_purpose_key(&model, "hyprstream-iroh-transport-v1").to_bytes(),
            RefuseHandler::new("no events"),
            IrohRpcProtocolHandler::new(model_bridge, model.clone()),
        )
        .await?;
        let registry_service = crate::services::RegistryService::new(
            directory.path().join("registry-data"),
            crate::services::PolicyClient::from_resolver(
                registry.clone(),
                Some(registry_jwt),
            )?,
            TransportConfig::ipc(directory.path().join("registry-must-not-exist.sock")),
            registry.clone(),
        )
        .await?
        .with_jwt_key_source(Arc::clone(&jwt_key_source));
        let registry_bridge = LocalServiceBridge::spawn(
            registry_service,
            Arc::new(hyprstream_rpc::envelope::InMemoryNonceCache::new()),
            0,
        )?;
        let registry_server = IrohSubstrate::new(
            derive_purpose_key(&registry, "hyprstream-iroh-transport-v1").to_bytes(),
            RefuseHandler::new("no events"),
            IrohRpcProtocolHandler::new(registry_bridge, registry.clone()),
        )
        .await?;
        for server in [
            &policy_server,
            &discovery_server,
            &registry_server,
            &model_server,
        ] {
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
        let accepted_store = crate::services::discovery::PdsRecordStore::open_readonly(
            &hyprstream_service::deployment_data_dir()?.join("pds-store"),
        )?
        .with_at9p_deployment_verifier(hyprstream_discovery::deployment_registry_verifier()?);
        let registry_state = accepted_store
            .accepted_at9p_states()?
            .into_iter()
            .find(|state| {
                state
                    .current
                    .services
                    .iter()
                    .any(|entry| entry.id == "#registry")
            })
            .expect("checkpoint-accepted Registry state");
        let model_state = accepted_store
            .accepted_at9p_states()?
            .into_iter()
            .find(|state| {
                state
                    .current
                    .services
                    .iter()
                    .any(|entry| entry.id == "#model")
            })
            .expect("checkpoint-accepted Model state");
        let registry_announcer =
            crate::services::DiscoveryClient::from_resolver(registry.clone(), None)?;
        registry_announcer
            .announce(&service_announcement(
                "registry",
                &registry_state,
                &registry,
                &policy,
            ))
            .await?;
        let model_announcer = crate::services::DiscoveryClient::from_resolver(model.clone(), None)?;
        model_announcer
            .announce(&service_announcement(
                "model",
                &model_state,
                &model,
                &policy,
            ))
            .await?;
        let registry_client = crate::services::RegistryClient::from_resolver(
            oai.clone(),
            Some(oai_jwt.clone()),
        )?;
        let model_client = crate::services::generated::model_client::ModelClient::from_resolver(
            oai.clone(),
            Some(oai_jwt),
        )?;
        let _mcp = crate::services::McpService::new(mcp_config)?;
        let server_state = crate::server::state::ServerState::new(
            crate::config::ServerConfig::default(),
            model_client.clone(),
            policy_client.clone(),
            std::clone::Clone::clone(&registry_client),
            oai.clone(),
            ca.verifying_key(),
            "http://127.0.0.1/oai".to_owned(),
            TEST_ISSUER.to_owned(),
            &std::collections::HashMap::new(),
            Arc::new(OaiTestNinePDecider),
        )
        .await?;
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
            crate::services::oai_prove_model_reachability_denial(&model_client).await?;
            crate::services::oai_healthy_registry(&registry_client).await?;
            anyhow::Ok(())
        })
        .await??;

        let reservation = std::net::TcpListener::bind("127.0.0.1:0")?;
        let ready_addr = reservation.local_addr()?;
        drop(reservation);
        let mut oai_config = crate::config::OAIConfig::default();
        oai_config.host = ready_addr.ip().to_string();
        oai_config.port = ready_addr.port();
        let mut tls_config = crate::config::TlsConfig::default();
        tls_config.enabled = false;
        let ready_shutdown = Arc::new(tokio::sync::Notify::new());
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let oai_service = crate::services::OAIService::new(
            oai_config,
            tls_config.clone(),
            crate::account::AccountZoneConfig::default(),
            server_state.clone(),
            true,
        );
        assert!(
            hyprstream_service::Spawnable::registrations(&oai_service).is_empty(),
            "HTTP-only OAI must not register an unserved native endpoint",
        );
        let ready_task = tokio::task::spawn_local(
            Box::new(oai_service).run_inner(Arc::clone(&ready_shutdown), Some(ready_tx)),
        );
        tokio::time::timeout(std::time::Duration::from_secs(10), ready_rx).await??;
        assert!(
            std::net::TcpListener::bind(ready_addr).is_err(),
            "OAI must retain its prebound listener after signaling readiness",
        );
        ready_shutdown.notify_waiters();
        tokio::time::timeout(std::time::Duration::from_secs(5), ready_task).await???;
        let rebound = std::net::TcpListener::bind(ready_addr)?;
        drop(rebound);

        let occupied = std::net::TcpListener::bind("127.0.0.1:0")?;
        let occupied_addr = occupied.local_addr()?;
        let mut occupied_config = crate::config::OAIConfig::default();
        occupied_config.host = occupied_addr.ip().to_string();
        occupied_config.port = occupied_addr.port();
        let (occupied_ready_tx, occupied_ready_rx) = tokio::sync::oneshot::channel();
        Box::new(crate::services::OAIService::new(
            occupied_config,
            tls_config.clone(),
            crate::account::AccountZoneConfig::default(),
            server_state.clone(),
            true,
        ))
        .run_inner(
            Arc::new(tokio::sync::Notify::new()),
            Some(occupied_ready_tx),
        )
        .await
        .expect_err("an occupied OAI listener must fail before readiness");
        assert!(
            occupied_ready_rx.await.is_err(),
            "an occupied listener must never emit OAI readiness",
        );
        drop(occupied);
        let rebound = std::net::TcpListener::bind(occupied_addr)?;
        drop(rebound);

        assert!(
            manager
                .add_policy_with_domain(
                    "service:oai",
                    "*",
                    "registry:HealthCheck",
                    "query",
                    "deny",
                )
                .await?,
            "test-only Registry denial must be installed alongside the base allow",
        );
        let denied = crate::services::oai_healthy_registry(&registry_client)
            .await
            .expect_err(
                "an exact persisted denial must override the OAI Registry base allow and withhold readiness",
            );
        assert!(
            denied.to_string().contains(
                "Unauthorized: service:oai cannot query on registry:HealthCheck",
            ),
            "the control must fail at Registry authorization, not transport: {denied}",
        );
        assert!(
            manager
                .remove_policy_with_domain(
                    "service:oai",
                    "*",
                    "registry:HealthCheck",
                    "query",
                    "deny",
                )
                .await?,
            "test-only Registry denial must be removed without altering base grants",
        );
        crate::services::oai_healthy_registry(&registry_client).await?;

        // The affirmative Model result above came from the production
        // `ModelService`: its generated dispatcher reached the real
        // tenant-authority boundary and returned the exact authenticated
        // tenantless denial. Replace that server with a protocol-equivalent
        // responder only for the classifier and liveness controls below.
        model_server.shutdown().await?;
        registry_server.shutdown().await?;
        let model_response = Arc::new(Mutex::new(ModelReadinessResponse::ExpectedTenantDenial));
        let model_signing_key = Arc::new(Mutex::new(model.clone()));
        let model_entered = Arc::new(tokio::sync::Notify::new());
        let model_release = Arc::new(tokio::sync::Notify::new());
        let model_bridge = LocalServiceBridge::spawn(
            ModelReadinessResponder {
                signing_key: Arc::clone(&model_signing_key),
                transport: TransportConfig::ipc(directory.path().join("model-must-not-exist.sock")),
                jwt_key_source: Arc::clone(&jwt_key_source),
                response: Arc::clone(&model_response),
                entered: Arc::clone(&model_entered),
                release: Arc::clone(&model_release),
            },
            Arc::new(hyprstream_rpc::envelope::InMemoryNonceCache::new()),
            0,
        )?;
        let model_server = IrohSubstrate::new(
            derive_purpose_key(&model, "hyprstream-iroh-transport-v1").to_bytes(),
            RefuseHandler::new("no events"),
            IrohRpcProtocolHandler::new(model_bridge, model.clone()),
        )
        .await?;
        lookup.add_endpoint_info(iroh::EndpointAddr::from_parts(
            model_server.endpoint_id(),
            model_server
                .endpoint()
                .bound_sockets()
                .into_iter()
                .map(iroh::TransportAddr::Ip),
        ));
        let registry_status = Arc::new(Mutex::new("healthy".to_owned()));
        let registry_bridge = LocalServiceBridge::spawn(
            RegistryReadinessResponder {
                signing_key: registry.clone(),
                transport: TransportConfig::ipc(
                    directory.path().join("registry-must-not-exist.sock"),
                ),
                jwt_key_source: Arc::clone(&jwt_key_source),
                status: Arc::clone(&registry_status),
            },
            Arc::new(hyprstream_rpc::envelope::InMemoryNonceCache::new()),
            0,
        )?;
        let registry_server = IrohSubstrate::new(
            derive_purpose_key(&registry, "hyprstream-iroh-transport-v1").to_bytes(),
            RefuseHandler::new("no events"),
            IrohRpcProtocolHandler::new(registry_bridge, registry.clone()),
        )
        .await?;
        lookup.add_endpoint_info(iroh::EndpointAddr::from_parts(
            registry_server.endpoint_id(),
            registry_server
                .endpoint()
                .bound_sockets()
                .into_iter()
                .map(iroh::TransportAddr::Ip),
        ));

        for response in [
            ModelReadinessResponse::Error {
                code: "UNAUTHORIZED".to_owned(),
                message: "authorization denied: no verified tenant domain".to_owned(),
                details: String::new(),
            },
            ModelReadinessResponse::Error {
                code: "INTERNAL".to_owned(),
                message: "authorization denied for another reason".to_owned(),
                details: String::new(),
            },
            ModelReadinessResponse::Error {
                code: "INTERNAL".to_owned(),
                message: "authorization denied: no verified tenant domain".to_owned(),
                details: "unexpected detail".to_owned(),
            },
            ModelReadinessResponse::Healthy,
        ] {
            *model_response.lock() = response;
            crate::services::oai_prove_model_reachability_denial(&model_client)
                .await
                .expect_err("only the exact authenticated tenant denial may satisfy readiness");
        }

        *model_response.lock() = ModelReadinessResponse::Block;
        let reservation = std::net::TcpListener::bind("127.0.0.1:0")?;
        let cancelled_addr = reservation.local_addr()?;
        drop(reservation);
        let mut cancelled_config = crate::config::OAIConfig::default();
        cancelled_config.host = cancelled_addr.ip().to_string();
        cancelled_config.port = cancelled_addr.port();
        let cancelled = Arc::new(tokio::sync::Notify::new());
        let (cancelled_ready_tx, mut cancelled_ready_rx) = tokio::sync::oneshot::channel();
        let cancellation_task = tokio::task::spawn_local(
            Box::new(crate::services::OAIService::new(
                cancelled_config,
                tls_config,
                crate::account::AccountZoneConfig::default(),
                server_state,
                true,
            ))
            .run_inner(Arc::clone(&cancelled), Some(cancelled_ready_tx)),
        );
        tokio::select! {
            _ = model_entered.notified() => {}
            _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {
                panic!("OAI did not enter its blocked Model readiness probe");
            }
        }
        assert!(
            std::net::TcpListener::bind(cancelled_addr).is_err(),
            "OAI must prebind HTTP before waiting on native dependencies",
        );
        assert!(
            matches!(
                cancelled_ready_rx.try_recv(),
                Err(tokio::sync::oneshot::error::TryRecvError::Empty)
            ),
            "OAI must withhold readiness while its Model probe is blocked",
        );
        cancelled.notify_waiters();
        let cancellation_result =
            tokio::time::timeout(std::time::Duration::from_secs(5), cancellation_task)
                .await??;
        let cancellation = cancellation_result
            .expect_err("shutdown must cancel OAI native readiness");
        assert!(cancellation.to_string().contains("cancelled"));
        assert!(
            cancelled_ready_rx.await.is_err(),
            "cancelled startup must drop its readiness sender without signaling",
        );
        model_release.notify_one();
        let rebound = std::net::TcpListener::bind(cancelled_addr)?;
        drop(rebound);

        let timeout_check = crate::services::oai_await_required_native_dependencies(
            &model_client,
            &registry_client,
            Arc::new(tokio::sync::Notify::new()),
            std::time::Duration::from_millis(250),
        );
        tokio::pin!(timeout_check);
        tokio::select! {
            _ = model_entered.notified() => {}
            result = &mut timeout_check => {
                panic!("blocked Model probe completed before timeout: {result:?}");
            }
        }
        let timeout = timeout_check
            .await
            .expect_err("bounded OAI native readiness must time out");
        assert!(timeout.to_string().contains("timed out"));
        model_release.notify_one();

        *model_response.lock() = ModelReadinessResponse::ExpectedTenantDenial;
        *model_signing_key.lock() = SigningKey::from_bytes(&[0x7e; 32]);
        crate::services::oai_await_required_native_dependencies(
            &model_client,
            &registry_client,
            Arc::new(tokio::sync::Notify::new()),
            std::time::Duration::from_secs(2),
        )
        .await
        .expect_err(
            "a server unable to authenticate as the accepted Model cannot satisfy bounded readiness",
        );
        *model_signing_key.lock() = model.clone();
        *registry_status.lock() = "degraded".to_owned();
        let unhealthy = crate::services::oai_healthy_registry(&registry_client)
            .await
            .expect_err("non-healthy Registry response must withhold OAI readiness");
        assert!(unhealthy.to_string().contains("degraded"));
        *registry_status.lock() = "healthy".to_owned();
        let unauthenticated = crate::services::RegistryClient::from_resolver(
            SigningKey::from_bytes(&[0x7f; 32]),
            None,
        )?;
        assert!(
            unauthenticated.health_check().await.is_err(),
            "an unregistered network signer must not satisfy Registry readiness"
        );
        assert_no_local_peers();
        // Ownership independence (Sol disposition case 6): shutting down ONLY
        // the OAuth inbound substrate must not disturb the process-global
        // outbound carrier — a subsequent real dial still succeeds.
        oauth_substrate.shutdown().await?;
        let post_shutdown = crate::services::generated::policy_client::PolicyCheck {
            subject: "forged-caller".to_owned(),
            domain: "forged-domain".to_owned(),
            resource: "model:allowed".to_owned(),
            operation: "query".to_owned(),
        };
        assert!(
            policy_client.check(&post_shutdown).await?,
            "outbound dials must survive OAuth substrate shutdown via the \
             untouched process-global carrier"
        );
        assert!(!policy_socket.exists());
        assert!(!discovery_socket.exists());
        model_server.shutdown().await?;
        crate::services::oai_await_required_native_dependencies(
            &model_client,
            &registry_client,
            Arc::new(tokio::sync::Notify::new()),
            std::time::Duration::from_secs(2),
        )
        .await
        .expect_err("an unreachable Model cannot satisfy bounded readiness");
        client_carrier.shutdown().await?;
        registry_server.shutdown().await?;
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
