//! #1274 MAC-enforcing end-to-end acceptance gate.
//!
//! The unconditional test keeps the currently deployable floor-only state
//! honest: an unlabeled RPC caller and an anonymous-floor 9P caller are denied
//! at the real dispatch/translator choke points, neither handler/backend read
//! runs, and both denials reach their audit surfaces.
//!
//! The positive test exercises the atproto → `exchangeUcan` token exchange →
//! RPC/9P acceptance path with a DPoP-bound, composite authority credential.
//! It widens only a test-local synthetic evidence fixture and narrows on drop;
//! production remains operator-gated and floor-only.

#![allow(clippy::expect_used, clippy::unwrap_used)]
#![cfg_attr(feature = "credential-pds", allow(dead_code, unused_imports))]

use std::io::{self, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use ed25519_dalek::{SigningKey, VerifyingKey};
use parking_lot::Mutex;
use rand::RngCore;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

use hyprstream_9p::memory::MemoryBackend;
use hyprstream_9p::msg::{self, Response};
use hyprstream_9p::{
    AccessDecider, Action as NinePAction, AttachAuthenticator, Backend, OpenResult,
    ReferenceMonitorDenyReason, SessionContext, StatResult, Translator, VerifiedAttach,
    VerifiedAttachIdentity, VerifiedTokenScope, WalkResult,
};
use hyprstream_core::mac::audit::{AuditError, AuditRecord, AuditSink, DecisionReason};
use hyprstream_core::mac::NinePAccessDecider;
use hyprstream_rpc::auth::mac::{
    install_mac_dispatch_pep, Assurance, CompartmentSet, DefaultMacDispatchPep, Level, MacDecision,
    MacDenyReason, MacDispatchPep, ObjectLabelResolver, ObjectRef, RpcObjectLabelResolver,
    SecurityContext, SecurityLabel,
};
use hyprstream_rpc::dial::{dial_with_crypto_stores, register_inproc};
use hyprstream_rpc::envelope::{InMemoryNonceCache, KeyedPqTrustStore};
use hyprstream_rpc::node_identity::derive_mesh_mldsa_key;
use hyprstream_rpc::service::{Continuation, EnvelopeContext, RequestService};
use hyprstream_rpc::signer::LocalSigner;
use hyprstream_rpc::transport::iroh_rpc::LocalServiceBridge;
use hyprstream_rpc::transport::rpc_session::IrohRequestProcessor;
use hyprstream_rpc::transport::TransportConfig;

const RPC_FLOOR_SERVICE: &str = "mac-gate-floor";
const RPC_T8_SERVICE: &str = "mac-gate-t8";
const SECRET_FILE: &str = "secret.txt";
const SECRET_BYTES: &[u8] = b"identity-aware MAC payload";
const FLOOR_CLIENT_KEY: [u8; 32] = [0x41; 32];
const T8_RPC_CLIENT_KEY: [u8; 32] = [0x42; 32];
const POLICY_SERVICE_KEY: [u8; 32] = [0x52; 32];

// Both paths install process-global RPC enforcement/crypto state. Keep their
// future concurrent execution deterministic when the T8 gate is unignored.
static GATE_GLOBALS: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn fresh_signing_key() -> SigningKey {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    SigningKey::from_bytes(&bytes)
}

fn label(level: Level, assurance: Assurance) -> SecurityLabel {
    SecurityLabel::new(level, assurance, CompartmentSet::EMPTY)
}

/// Install the fixture identities' process-wide hybrid trust view.
///
/// These anchors authenticate fixture keys; they grant no authorization.
/// Sharing one view also keeps the first-write-wins verifier deterministic
/// when the T8 test is eventually unignored beside the negative gate.
fn install_gate_crypto() -> Result<()> {
    let mut store = KeyedPqTrustStore::new();
    for bytes in [FLOOR_CLIENT_KEY, T8_RPC_CLIENT_KEY, POLICY_SERVICE_KEY] {
        let ed = SigningKey::from_bytes(&bytes);
        let pq = derive_mesh_mldsa_key(&ed);
        let pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
            &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&pq),
        )?;
        store.bind(ed.verifying_key().to_bytes(), &pq_vk);
    }
    let _ = hyprstream_rpc::envelope::install_verify_config(
        hyprstream_rpc::envelope::EnvelopeVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Hybrid,
            pq_store: Some(Arc::new(store)),
        },
    );
    let _ = hyprstream_rpc::envelope::install_response_verify_config(
        hyprstream_rpc::envelope::ResponseVerifyConfig {
            policy: hyprstream_rpc::crypto::CryptoPolicy::Classical,
            pq_store: None,
        },
    );
    Ok(())
}

struct RpcFloorLabels(&'static str);

impl RpcObjectLabelResolver for RpcFloorLabels {
    fn resolve(&self, service_domain: &str, _method: Option<&[u16]>) -> Option<SecurityLabel> {
        (service_domain == self.0).then(|| label(Level::Public, Assurance::Classical))
    }
}

/// Explicit, test-only bootstrap authority for the fixture policy service.
///
/// The OAuth fixture's local policy client intentionally carries no end-user
/// clearance. It must still install a narrowly scoped PEP now that the
/// uninstalled dispatch state denies at rest.
struct ExactServiceBootstrapPep(String);

impl MacDispatchPep for ExactServiceBootstrapPep {
    fn check(
        &self,
        _ctx: &EnvelopeContext,
        service_domain: &str,
        _method: Option<&[u16]>,
    ) -> MacDecision {
        if service_domain == self.0 {
            MacDecision::Permit
        } else {
            MacDecision::Deny(MacDenyReason::UnlabeledObject)
        }
    }
}

struct GateEchoService {
    name: &'static str,
    transport: TransportConfig,
    signing_key: SigningKey,
    invocations: Arc<AtomicUsize>,
}

#[async_trait(?Send)]
impl RequestService for GateEchoService {
    fn decode_request_body(
        &self,
        signed_body: &[u8],
    ) -> Result<hyprstream_rpc::service::DecodedRequestBody> {
        Ok(hyprstream_rpc::service::DecodedRequestBody::opaque(
            signed_body.to_vec(),
        ))
    }

    async fn handle_request(
        &self,
        _ctx: &EnvelopeContext,
        body: &hyprstream_rpc::service::DecodedRequestBody,
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        Ok((body.bytes().to_vec(), None))
    }

    fn name(&self) -> &str {
        self.name
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }
}

#[derive(Clone, Default)]
struct SharedLog {
    bytes: Arc<Mutex<Vec<u8>>>,
}

struct SharedWriter(SharedLog);

impl Write for SharedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.bytes.lock().extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SharedLog {
    type Writer = SharedWriter;

    fn make_writer(&'a self) -> Self::Writer {
        SharedWriter(self.clone())
    }
}

async fn rpc_floor_denial(log: &SharedLog) -> Result<()> {
    install_gate_crypto()?;
    let client_signing = SigningKey::from_bytes(&FLOOR_CLIENT_KEY);

    let pep: Arc<dyn MacDispatchPep> = Arc::new(DefaultMacDispatchPep::new(Box::new(
        RpcFloorLabels(RPC_FLOOR_SERVICE),
    )));
    install_mac_dispatch_pep(pep);

    let server_signing = fresh_signing_key();
    let server_verifying: VerifyingKey = server_signing.verifying_key();
    let invocations = Arc::new(AtomicUsize::new(0));
    let service = GateEchoService {
        name: RPC_FLOOR_SERVICE,
        transport: TransportConfig::inproc(RPC_FLOOR_SERVICE),
        signing_key: server_signing.clone(),
        invocations: Arc::clone(&invocations),
    };
    let bridge = LocalServiceBridge::spawn(service, Arc::new(InMemoryNonceCache::new()), 0)?;
    let processor: Arc<dyn IrohRequestProcessor> = Arc::new(bridge);
    register_inproc(RPC_FLOOR_SERVICE, &processor);

    let server_pq = derive_mesh_mldsa_key(&server_signing);
    let server_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
        &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&server_pq),
    )?;
    let mut response_store = KeyedPqTrustStore::new();
    response_store.bind(server_verifying.to_bytes(), &server_pq_vk);
    let client = dial_with_crypto_stores(
        &TransportConfig::inproc(RPC_FLOOR_SERVICE),
        LocalSigner::new(client_signing),
        Some(server_verifying),
        None,
        None,
        Some(Arc::new(response_store)),
    )?;

    let response = client
        .call_for_service(RPC_FLOOR_SERVICE, b"must-not-reach-handler".to_vec())
        .await?;
    assert!(
        response.is_empty(),
        "RPC MAC denial must return the service's signed error payload, never handler bytes"
    );
    assert_eq!(
        invocations.load(Ordering::SeqCst),
        0,
        "fail-closed RPC denial must happen before handler invocation"
    );

    let emitted = String::from_utf8_lossy(&log.bytes.lock()).into_owned();
    assert!(
        emitted.contains("hyprstream.mac.audit")
            && emitted.contains("rpc-dispatch")
            && emitted.contains("deny"),
        "RPC MAC denial must be present on the unified audit target: {emitted}"
    );
    drop(processor);
    Ok(())
}

struct NinePLabels;

impl ObjectLabelResolver for NinePLabels {
    fn resolve(&self, object: ObjectRef<'_>) -> Option<SecurityLabel> {
        match object {
            ObjectRef::Path([]) => Some(label(Level::Public, Assurance::Unverified)),
            ObjectRef::Path([SECRET_FILE]) => Some(label(Level::Secret, Assurance::Classical)),
            ObjectRef::Path(_) | ObjectRef::Cid(_) => None,
        }
    }
}

#[derive(Default)]
struct SpyAudit {
    records: Mutex<Vec<AuditRecord>>,
}

impl AuditSink for SpyAudit {
    fn record(&self, record: &AuditRecord) -> Result<(), AuditError> {
        self.records.lock().push(record.clone());
        Ok(())
    }
}

/// Preserve the current floor-only behavior while allowing the wire harness to
/// reach the concrete `Tread` choke point. T8 replaces this adapter with the
/// full verified-token reference monitor.
struct FloorReadGate {
    audited: NinePAccessDecider,
}

impl AccessDecider for FloorReadGate {
    fn check(&self, ctx: &SecurityContext, object: ObjectRef<'_>, action: NinePAction) -> bool {
        if action == NinePAction::Read {
            self.audited.check(ctx, object, action)
        } else {
            true
        }
    }

    fn audit_denial(
        &self,
        ctx: &SecurityContext,
        object: ObjectRef<'_>,
        object_label: Option<SecurityLabel>,
        action: NinePAction,
        reason: ReferenceMonitorDenyReason,
    ) {
        self.audited
            .audit_denial(ctx, object, object_label, action, reason);
    }
}

struct ReadCountingBackend {
    inner: MemoryBackend,
    reads: Arc<AtomicUsize>,
}

#[async_trait]
impl Backend for ReadCountingBackend {
    async fn attach(
        &self,
        uname: &str,
        aname: &str,
        verified: Option<hyprstream_9p::VerifiedAttach>,
    ) -> Result<Option<hyprstream_9p::VerifiedAttach>> {
        self.inner.attach(uname, aname, verified).await
    }

    async fn walk(&self, fid: u32, newfid: u32, components: &[String]) -> Result<WalkResult> {
        self.inner.walk(fid, newfid, components).await
    }

    async fn open(&self, fid: u32, flags: u32) -> Result<OpenResult> {
        self.inner.open(fid, flags).await
    }

    async fn read(&self, fid: u32, offset: u64, count: u32) -> Result<Vec<u8>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.read(fid, offset, count).await
    }

    async fn write(&self, fid: u32, offset: u64, data: &[u8]) -> Result<u32> {
        self.inner.write(fid, offset, data).await
    }

    async fn stat(&self, fid: u32) -> Result<StatResult> {
        self.inner.stat(fid).await
    }

    async fn readdir(&self, fid: u32, offset: u64, count: u32) -> Result<Vec<u8>> {
        self.inner.readdir(fid, offset, count).await
    }

    async fn clunk(&self, fid: u32) -> Result<()> {
        self.inner.clunk(fid).await
    }
}

async fn recv_9p(stream: &mut TcpStream) -> Vec<u8> {
    let mut len = [0u8; 4];
    stream.read_exact(&mut len).await.unwrap();
    let total = u32::from_le_bytes(len) as usize;
    let mut bytes = vec![0u8; total];
    bytes[..4].copy_from_slice(&len);
    stream.read_exact(&mut bytes[4..]).await.unwrap();
    bytes
}

async fn ninep_rpc(stream: &mut TcpStream, request: Vec<u8>) -> Response {
    stream.write_all(&request).await.unwrap();
    let response = recv_9p(stream).await;
    msg::parse_response(&response).unwrap().1
}

async fn ninep_floor_denial() -> Result<()> {
    let inner = MemoryBackend::default();
    inner.add_file(&format!("/{SECRET_FILE}"), SECRET_BYTES);
    let backend_reads = Arc::new(AtomicUsize::new(0));
    let backend = ReadCountingBackend {
        inner,
        reads: Arc::clone(&backend_reads),
    };
    let audit = Arc::new(SpyAudit::default());
    let decider = FloorReadGate {
        audited: NinePAccessDecider::new(Arc::new(NinePLabels), audit.clone()),
    };

    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let translator = Translator::new(Arc::new(backend), Arc::new(decider));
    let server = tokio::spawn(async move {
        let _ = translator.serve(listener).await;
    });
    let mut client = TcpStream::connect(address).await?;

    assert!(matches!(
        ninep_rpc(&mut client, msg::tversion(1, 4096, "9P2000.L")).await,
        Response::Version { .. }
    ));
    assert!(matches!(
        ninep_rpc(&mut client, msg::tattach(2, 0, u32::MAX, "attacker", "/")).await,
        Response::Attach { .. }
    ));
    assert!(matches!(
        ninep_rpc(&mut client, msg::twalk(3, 0, 1, &[SECRET_FILE])).await,
        Response::Walk { .. }
    ));
    assert!(matches!(
        ninep_rpc(&mut client, msg::tlopen(4, 1, 0)).await,
        Response::Lopen { .. }
    ));
    assert!(
        matches!(
            ninep_rpc(&mut client, msg::tread(5, 1, 0, 4096)).await,
            Response::Error { .. }
        ),
        "anonymous-floor 9P caller must be denied at the concrete read operation"
    );
    assert_eq!(
        backend_reads.load(Ordering::SeqCst),
        0,
        "fail-closed 9P denial must happen before backend read invocation"
    );

    let records = audit.records.lock();
    let denial = records
        .iter()
        .find(|record| record.reason == DecisionReason::FloorDeny)
        .expect("9P floor denial must be audited");
    assert_eq!(
        denial.subject_clearance,
        *hyprstream_9p::anonymous_floor().clearance(),
        "the current 9P gate must exercise the real anonymous-floor subject"
    );
    assert_eq!(
        denial.object_label,
        label(Level::Secret, Assurance::Classical),
        "the audited denial must name the trusted secret object label"
    );
    assert_eq!(
        denial.action,
        hyprstream_core::mac::te::Action::from_scope_action(
            hyprstream_core::mac::te::ScopeAction::Query,
        )
    );
    drop(records);
    server.abort();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unauthorized_subject_is_denied_and_audited_on_rpc_and_9p() -> Result<()> {
    let _globals = GATE_GLOBALS.lock().await;
    let log = SharedLog::default();
    tracing::subscriber::set_global_default(
        tracing_subscriber::fmt()
            .with_writer(log.clone())
            .with_ansi(false)
            .finish(),
    )
    .expect("gate test binary owns the global tracing subscriber");

    rpc_floor_denial(&log).await?;
    ninep_floor_denial().await?;
    Ok(())
}

// Positive acceptance path retained as the executable T8 contract.
//
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[cfg(not(feature = "credential-pds"))]
async fn verified_atproto_identity_authorizes_rpc_and_9p() -> Result<()> {
    let _globals = GATE_GLOBALS.lock().await;
    let coverage = hyprstream_rpc::auth::mac::GenesisReport {
        labeled: vec!["/srv".to_owned(), RPC_T8_SERVICE.to_owned()],
        unlabeled: Vec::new(),
        ill_formed: Vec::new(),
    };
    let evidence = hyprstream_rpc::auth::mac::MacActivationEvidence {
        genesis: &coverage,
        mediation_integrity_g2: true,
        denial_handling_g4: true,
        observability_g5: true,
        runbook_signoff_g6: true,
        revocation_reload_g7: true,
    };
    hyprstream_rpc::auth::mac::global_mac_activation_control().widen_identity_aware(&evidence)?;
    struct NarrowOnDrop;
    impl Drop for NarrowOnDrop {
        fn drop(&mut self) {
            hyprstream_rpc::auth::mac::global_mac_activation_control().narrow_to_floor();
        }
    }
    let _narrow = NarrowOnDrop;
    // The complete OAuth/PAR + DPoP + exchangeUcan fixture below uses the
    // production router and wire clients. The synthetic evidence above tests
    // the control mechanism; it is not production activation evidence.
    let credential = t8_atproto_session_credential().await?;

    assert_eq!(
        credential.subject, "did:web:alice.acct.example.test",
        "the session principal must be the DID verified by the atproto login"
    );
    assert_eq!(
        credential.tenant, "tenant-demo",
        "tenant must come from the DID's authority-owned hosted-account binding"
    );
    assert_eq!(
        credential.clearance,
        label(Level::Secret, Assurance::Classical),
        "clearance must be the authority-stamped Claims value, not anonymous_floor"
    );
    assert!(
        credential.composite_signed,
        "RFC 8693/UCAN exchange must yield the composite session credential"
    );
    assert_eq!(
        credential.rpc_bytes, SECRET_BYTES,
        "authorized RPC call must reach the protected handler"
    );
    assert_eq!(
        credential.ninep_bytes, SECRET_BYTES,
        "authorized 9P Tread must return protected bytes"
    );
    Ok(())
}

#[test]
fn verified_attach_denies_mac_identity_vfs_subject_divergence() {
    let identity = VerifiedAttachIdentity::from_verified_credential("alice", "tenant-a");
    let context = SecurityContext::from_clearance(
        label(Level::Secret, Assurance::Classical),
        hyprstream_rpc::auth::mac::VerifiedKeyMaterial::Classical,
    );
    let session = SessionContext::from_verified_clearance(identity.clone(), context);
    let result = VerifiedAttach::try_new(identity, hyprstream_rpc::Subject::new("bob"), session);
    assert!(
        result.is_err(),
        "a MAC identity and VFS Subject mismatch must fail closed before attach"
    );
}

struct T8SessionCredential {
    subject: String,
    tenant: String,
    clearance: SecurityLabel,
    composite_signed: bool,
    rpc_bytes: Vec<u8>,
    ninep_bytes: Vec<u8>,
}

#[cfg(not(feature = "credential-pds"))]
async fn t8_atproto_session_credential() -> Result<T8SessionCredential> {
    use base64::{
        engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
        Engine as _,
    };
    use ed25519_dalek::Signer as _;
    use hyprstream_core::auth::{PolicyManager, ProductionUserStore, UserProfile};
    use hyprstream_core::config::{CorsConfig, OAuthConfig, TokenConfig};
    use hyprstream_core::services::oauth::state::{
        AtprotoDidDocumentResolver, OAuthState, RegisteredClient,
    };
    use hyprstream_core::services::{DiscoveryClient, PolicyClient, PolicyService};
    use hyprstream_rpc::auth::ClusterKeySource;
    use hyprstream_rpc::crypto::CryptoPolicy;
    use hyprstream_rpc::rpc_client::RpcClientImpl;
    use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;
    use hyprstream_service::{InprocManager, ServiceManager as _};
    use hyprstream_vfs::{SyntheticMount, SyntheticNode};
    use sha2::{Digest as _, Sha256};

    const SUBJECT_DID: &str = "did:web:alice.acct.example.test";
    const TENANT: &str = "tenant-demo";
    const CLIENT_ID: &str = "mac-gate-client";
    const REDIRECT_URI: &str = "https://client.example.test/callback";
    const PKCE_VERIFIER: &str = "mac-gate-pkce-verifier-abcdefghijklmnopqrstuvwxyz0123456789";

    install_gate_crypto()?;
    configure_policy_signing_authority()?;

    // Reuse the OAuth conformance suite's real in-process PolicyService rather
    // than replacing the signing boundary with a fixture token.
    let policy_signing = SigningKey::from_bytes(&POLICY_SERVICE_KEY);
    let policy_tag = format!("mac-gate-policy-{}", uuid::Uuid::new_v4());
    install_mac_dispatch_pep(Arc::new(ExactServiceBootstrapPep("policy".to_owned())));
    let policy_dir = tempfile::TempDir::new()?;
    let git2db = Arc::new(tokio::sync::RwLock::new(
        git2db::Git2DB::open(policy_dir.path()).await?,
    ));
    let policy_service = PolicyService::new(
        Arc::new(PolicyManager::permissive().await?),
        Arc::new(policy_signing.clone()),
        TokenConfig::default(),
        git2db,
        TransportConfig::inproc(&policy_tag),
    )
    .with_token_clearance_resolver(Arc::new(|subject| {
        (subject == SUBJECT_DID).then(|| label(Level::Secret, Assurance::Classical))
    }));
    let manager = InprocManager::new();
    let mut policy_handle = manager.spawn(Box::new(policy_service)).await?;
    let policy_client = PolicyClient::for_local_endpoint_bootstrap(
        &format!("inproc://{policy_tag}"),
        policy_signing.clone(),
        policy_signing.verifying_key(),
        None,
    )?;

    let user_dir = tempfile::TempDir::new()?;
    let user_store = ProductionUserStore::open(user_dir.path()).await?;
    let login_key = SigningKey::from_bytes(&[0x53; 32]);
    user_store.register("alice").await?;
    let fingerprint = user_store
        .add_pubkey(
            "alice",
            login_key.verifying_key(),
            Some("mac-gate-login".to_owned()),
        )
        .await?;
    user_store
        .set_profile(
            "alice",
            UserProfile {
                atproto_did: Some(SUBJECT_DID.to_owned()),
                ..Default::default()
            }
            .into(),
        )
        .await?;

    // Build the authority-owned hosted account record. Its generated
    // #atproto key both signs the service assertion and anchors the tenant
    // lookup, so no client-supplied tenant can satisfy the assertion below.
    let account_ed = SigningKey::from_bytes(&[0x54; 32]);
    let (account_pq, account_pq_vk) = hyprstream_crypto::pq::ml_dsa_generate_keypair();
    let account_hybrid = hyprstream_pds::did_op::HybridRotationKey::new(
        account_ed.verifying_key().to_bytes(),
        hyprstream_crypto::pq::ml_dsa_vk_bytes(&account_pq_vk),
    )?;
    let rotations = hyprstream_pds::did_op::GenesisRotationKeys::new(
        hyprstream_pds::did_op::UserRotationKey::new(account_hybrid),
        hyprstream_pds::did_op::RecoveryKeyEnrollment::Declined,
        hyprstream_pds::did_op::HostKeyEnrollment::Absent,
    )?;
    let mint = hyprstream_pds::HostedAccountMint::begin(
        hyprstream_pds::AllocatedAccountName::new("alice", SUBJECT_DID)?,
        rotations,
    )?;
    let document = mint.seal_did_document("https://pds.example.com")?;
    let pending =
        mint.prepare_genesis(document, hyprstream_pds::did_op::GenesisRepoHead::EmptyRepo)?;
    let signature =
        hyprstream_pds::did_op::sign_genesis(pending.unsigned_genesis(), &account_ed, &account_pq)?;
    let sealed = pending.seal(signature)?;
    let atproto_signing_key = sealed.atproto_signing_key().clone();
    let pds_root = SyntheticNode::dir().with_child(
        TENANT,
        SyntheticNode::dir().with_child(
            "accounts",
            SyntheticNode::dir().with_child(
                "alice",
                SyntheticNode::dir().with_child(
                    "account-record.cbor",
                    SyntheticNode::file(sealed.record_bytes().to_vec()),
                ),
            ),
        ),
    );
    let hosted_store = Arc::new(hyprstream_pds_service::AccountRecordStore::new(
        Arc::new(SyntheticMount::new(pds_root)),
        hyprstream_core::mac::production_pds_account_read_authorizer(Arc::new(SpyAudit::default())),
    ));

    let mut atproto_multikey = vec![0x80, 0x24];
    atproto_multikey.extend_from_slice(
        atproto_signing_key
            .verifying_key()
            .to_encoded_point(true)
            .as_bytes(),
    );
    let atproto_document = serde_json::json!({
        "id": SUBJECT_DID,
        "verificationMethod": [{
            "id": format!("{SUBJECT_DID}#atproto"),
            "type": "Multikey",
            "controller": SUBJECT_DID,
            "publicKeyMultibase": format!(
                "z{}",
                bs58::encode(atproto_multikey).into_string()
            )
        }]
    });

    struct FixtureDidResolver(serde_json::Value);

    #[async_trait]
    impl AtprotoDidDocumentResolver for FixtureDidResolver {
        async fn resolve_document(&self, did: &str) -> Result<serde_json::Value> {
            anyhow::ensure!(self.0["id"].as_str() == Some(did), "fixture DID mismatch");
            Ok(self.0.clone())
        }
    }

    let oauth_listener = TcpListener::bind("127.0.0.1:0").await?;
    let oauth_address = oauth_listener.local_addr()?;
    let oauth_origin = format!("http://{oauth_address}");
    let dummy_key = SigningKey::from_bytes(&[0x55; 32]);
    let dummy_rpc = Arc::new(
        RpcClientImpl::new(
            LocalSigner::new(dummy_key.clone()),
            LazyUdsTransport::new("/dev/null/mac-gate-discovery.sock".into()),
            Some(dummy_key.verifying_key()),
        )
        .with_response_verify_policy(CryptoPolicy::Classical),
    );
    let mut oauth_config = OAuthConfig {
        external_url: Some(oauth_origin.clone()),
        ..OAuthConfig::default()
    };
    oauth_config.require_pushed_authorization_requests = true;
    let state = Arc::new(
        OAuthState::new(
            &oauth_config,
            policy_client,
            DiscoveryClient::new(dummy_rpc),
            policy_signing.verifying_key().to_bytes(),
        )
        .with_user_store(user_store)
        .with_hosted_account_store(hosted_store)
        .with_atproto_did_resolver(Arc::new(FixtureDidResolver(atproto_document)))
        .with_hosted_account_zone(hyprstream_core::account::AccountZone::new(
            "acct.example.test",
        )?)
        .with_audit_sink(Arc::new(SpyAudit::default())),
    );
    state.clients.write().await.insert(
        CLIENT_ID.to_owned(),
        RegisteredClient {
            client_id: CLIENT_ID.to_owned(),
            redirect_uris: vec![REDIRECT_URI.to_owned()],
            client_name: Some("MAC gate client".to_owned()),
            client_uri: None,
            logo_uri: None,
            grant_types: vec![
                "authorization_code".to_owned(),
                "urn:ietf:params:oauth:grant-type:token-exchange".to_owned(),
            ],
            response_types: vec!["code".to_owned()],
            token_endpoint_auth_method: Some("none".to_owned()),
            jwks: None,
            jwks_uri: None,
            hyprstream_node_did: None,
            scope: Some("atproto transition:generic".to_owned()),
            dpop_bound_access_tokens: Some(true),
            is_cimd: false,
            registered_at: std::time::Instant::now(),
        },
    );

    let oauth_app = hyprstream_core::services::oauth::create_app(
        Arc::clone(&state),
        &CorsConfig {
            enabled: false,
            ..CorsConfig::default()
        },
    );
    let oauth_server = tokio::spawn(async move {
        let _ = axum::serve(oauth_listener, oauth_app).await;
    });
    let http = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()?;

    // atproto login: PAR binds the browser session to the DPoP key; the real
    // authorization page verifies the registered user key before issuing the
    // code; the token exchange proves PKCE and the server-issued DPoP nonce.
    let atproto_dpop_key = p256::ecdsa::SigningKey::random(&mut rand::rngs::OsRng);
    let rpc_client_signing = SigningKey::from_bytes(&T8_RPC_CLIENT_KEY);
    let code_challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(PKCE_VERIFIER.as_bytes()));
    let par_proof = es256_dpop_proof(
        &atproto_dpop_key,
        &format!("{oauth_origin}/oauth/par"),
        "mac-gate-par",
        None,
    );
    let par = http
        .post(format!("{oauth_origin}/oauth/par"))
        .header("DPoP", par_proof)
        .form(&[
            ("client_id", CLIENT_ID),
            ("redirect_uri", REDIRECT_URI),
            ("code_challenge", &code_challenge),
            ("code_challenge_method", "S256"),
            ("response_type", "code"),
            ("state", "mac-gate-state"),
            ("scope", "atproto"),
            ("resource", &oauth_origin),
        ])
        .send()
        .await?;
    anyhow::ensure!(
        par.status() == reqwest::StatusCode::CREATED,
        "PAR failed: {}",
        par.status()
    );
    let par_nonce = par
        .headers()
        .get("DPoP-Nonce")
        .and_then(|value| value.to_str().ok())
        .context("PAR response omitted DPoP-Nonce")?
        .to_owned();
    let request_uri = par.json::<serde_json::Value>().await?["request_uri"]
        .as_str()
        .context("PAR response omitted request_uri")?
        .to_owned();
    let authorize = http
        .get(format!("{oauth_origin}/oauth/authorize"))
        .query(&[
            ("request_uri", request_uri.as_str()),
            ("client_id", CLIENT_ID),
        ])
        .send()
        .await?;
    anyhow::ensure!(authorize.status().is_success(), "authorize GET failed");
    let html = authorize.text().await?;
    let authorize_nonce = html_hidden_value(&html, "nonce")?;
    let challenge = format!("{fingerprint}:{authorize_nonce}:{code_challenge}");
    let login_signature = STANDARD.encode(login_key.sign(challenge.as_bytes()).to_bytes());
    let callback = http
        .post(format!("{oauth_origin}/oauth/authorize"))
        .form(&[
            ("client_id", CLIENT_ID),
            ("redirect_uri", REDIRECT_URI),
            ("code_challenge", &code_challenge),
            ("scope", "atproto"),
            ("state", "mac-gate-state"),
            ("resource", &oauth_origin),
            ("nonce", authorize_nonce),
            ("fingerprint", &fingerprint),
            ("signature", &login_signature),
        ])
        .send()
        .await?;
    anyhow::ensure!(
        callback.status() == reqwest::StatusCode::SEE_OTHER,
        "authorize POST failed: {}",
        callback.status()
    );
    let location = callback
        .headers()
        .get(reqwest::header::LOCATION)
        .and_then(|value| value.to_str().ok())
        .context("authorize POST omitted redirect")?;
    let callback_url = url::Url::parse(location)?;
    let code = callback_url
        .query_pairs()
        .find_map(|(key, value)| (key == "code").then(|| value.into_owned()))
        .context("authorize redirect omitted code")?;
    let token_proof = es256_dpop_proof(
        &atproto_dpop_key,
        &format!("{oauth_origin}/oauth/token"),
        "mac-gate-token",
        Some(&par_nonce),
    );
    let login_token = http
        .post(format!("{oauth_origin}/oauth/token"))
        .header("DPoP", token_proof)
        .form(&[
            ("grant_type", "authorization_code"),
            ("client_id", CLIENT_ID),
            ("code", &code),
            ("redirect_uri", REDIRECT_URI),
            ("code_verifier", PKCE_VERIFIER),
        ])
        .send()
        .await?;
    let login_status = login_token.status();
    if !login_status.is_success() {
        let body = login_token.text().await.unwrap_or_default();
        anyhow::bail!("authorization-code exchange failed: {login_status} {body}");
    }
    let login_json = login_token.json::<serde_json::Value>().await?;
    let login_access_token = login_json["access_token"]
        .as_str()
        .context("atproto token response omitted access_token")?;
    let login_claims = hyprstream_rpc::auth::decode_unverified(login_access_token)?;
    anyhow::ensure!(
        login_claims.sub == SUBJECT_DID,
        "atproto OAuth token was not bound to the verified DID"
    );

    // First obtain the verified atproto actor credential. The assertion's
    // #atproto key is resolved from the DID document, and the tenant comes only
    // from the hosted-account record constructed above.
    let now = chrono::Utc::now().timestamp();
    let header = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&serde_json::json!({
        "alg": "ES256", "typ": "JWT", "kid": "#atproto"
    }))?);
    let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&serde_json::json!({
        "iss": SUBJECT_DID,
        "aud": state.atproto_service_did().context("issuer has no service DID")?,
        "iat": now,
        "exp": now + 60,
        "lxm": hyprstream_core::services::oauth::token_exchange::ATPROTO_EXCHANGE_NSID,
        "jti": "mac-gate-service-auth"
    }))?);
    let signing_input = format!("{header}.{payload}");
    let service_signature: p256::ecdsa::Signature =
        atproto_signing_key.sign(signing_input.as_bytes());
    let service_jwt = format!(
        "{signing_input}.{}",
        URL_SAFE_NO_PAD.encode(service_signature.to_bytes())
    );
    let atproto_exchange_endpoint =
        format!("{oauth_origin}/xrpc/ai.hyprstream.identity.exchangeUcan");
    let exchange_proof = ed25519_dpop_proof(
        &rpc_client_signing,
        &atproto_exchange_endpoint,
        "mac-gate-atproto-exchange",
        None,
    );
    let exchange = http
        .post(&atproto_exchange_endpoint)
        .bearer_auth(service_jwt)
        .header("DPoP", exchange_proof)
        .json(&serde_json::json!({
            "tenant": TENANT,
            "scope": "transition:generic",
            "audience": oauth_origin.clone(),
        }))
        .send()
        .await?;
    let exchange_status = exchange.status();
    if !exchange_status.is_success() {
        let body = exchange.text().await.unwrap_or_default();
        anyhow::bail!("atproto actor-token exchange failed: {exchange_status} {body}");
    }
    let exchange_json = exchange.json::<serde_json::Value>().await?;
    let atproto_actor_token = exchange_json["access_token"]
        .as_str()
        .context("atproto exchange response omitted access_token")?
        .to_owned();
    let atproto_actor_claims = hyprstream_rpc::auth::decode_unverified(&atproto_actor_token)?;
    anyhow::ensure!(
        atproto_actor_claims.sub == SUBJECT_DID
            && atproto_actor_claims.tenant.as_deref() == Some(TENANT)
            && atproto_actor_claims.clearance
                == Some(label(Level::Secret, Assurance::Classical)),
        "verified atproto actor credential lost its authority-resolved subject, tenant, or clearance"
    );
    let session_token = atproto_actor_token;
    let claims = hyprstream_rpc::auth::decode_unverified(&session_token)?;
    let clearance = claims
        .clearance
        .context("authority-minted session credential omitted Claims.clearance")?;
    let tenant = claims
        .tenant
        .clone()
        .context("authority-minted session credential omitted tenant")?;
    let protected = hyprstream_rpc::auth::parse_protected_header(&session_token)?;
    let expected_rpc_jkt =
        hyprstream_rpc::auth::jwk_thumbprint(&hyprstream_rpc::auth::JwkThumbprintInput::Ed25519 {
            x: &rpc_client_signing.verifying_key().to_bytes(),
        });
    assert_eq!(
        claims.cnf_jkt(),
        Some(expected_rpc_jkt.as_str()),
        "exchanged credential must be sender-bound to the downstream RPC envelope signer"
    );

    // RPC production dispatch. T8 must bind the exchanged credential to the
    // envelope signer and install the Claims-fed PEP; no test-only context is
    // injected here.
    install_mac_dispatch_pep(Arc::new(DefaultMacDispatchPep::new(Box::new(
        RpcFloorLabels(RPC_T8_SERVICE),
    ))));
    let rpc_server_signing = fresh_signing_key();
    let rpc_server_vk = rpc_server_signing.verifying_key();
    let rpc_invocations = Arc::new(AtomicUsize::new(0));
    let rpc_key_source = Arc::new(ClusterKeySource::new(
        policy_signing.verifying_key(),
        oauth_origin.clone(),
    ));
    let rpc_service = AuthenticatedGateEcho {
        name: RPC_T8_SERVICE,
        transport: TransportConfig::inproc(RPC_T8_SERVICE),
        signing_key: rpc_server_signing.clone(),
        invocations: Arc::clone(&rpc_invocations),
        key_source: rpc_key_source,
    };
    let bridge = LocalServiceBridge::spawn(rpc_service, Arc::new(InMemoryNonceCache::new()), 0)?;
    let rpc_processor: Arc<dyn IrohRequestProcessor> = Arc::new(bridge);
    register_inproc(RPC_T8_SERVICE, &rpc_processor);
    let rpc_server_pq = derive_mesh_mldsa_key(&rpc_server_signing);
    let rpc_server_pq_vk = hyprstream_rpc::crypto::pq::ml_dsa_vk_from_bytes(
        &hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(&rpc_server_pq),
    )?;
    let mut response_store = KeyedPqTrustStore::new();
    response_store.bind(rpc_server_vk.to_bytes(), &rpc_server_pq_vk);
    let rpc_client = dial_with_crypto_stores(
        &TransportConfig::inproc(RPC_T8_SERVICE),
        LocalSigner::new(rpc_client_signing),
        Some(rpc_server_vk),
        Some(session_token.clone()),
        None,
        Some(Arc::new(response_store)),
    )?;
    let rpc_bytes = rpc_client
        .call_for_service(RPC_T8_SERVICE, SECRET_BYTES.to_vec())
        .await?;

    // 9P production monitor constructor. The fixture authenticator below
    // represents the route's already-successful JWT verification and derives
    // both the MAC session and VFS Subject from those same verified Claims.
    struct VerifiedClaimsAttach {
        token: String,
        claims: hyprstream_rpc::auth::Claims,
    }
    #[async_trait]
    impl AttachAuthenticator for VerifiedClaimsAttach {
        async fn authenticate(
            &self,
            uname: &str,
            aname: &str,
        ) -> Result<VerifiedAttach, hyprstream_vfs::MountError> {
            use hyprstream_rpc::auth::mac::SubjectContextClaims as _;
            if uname != self.token || aname != "/" {
                return Err(hyprstream_vfs::MountError::PermissionDenied(
                    "credential or export mismatch".to_owned(),
                ));
            }
            let tenant = self.claims.tenant.as_deref().ok_or_else(|| {
                hyprstream_vfs::MountError::PermissionDenied(
                    "verified claims omitted tenant".to_owned(),
                )
            })?;
            let context = self
                .claims
                .security_context(hyprstream_rpc::auth::mac::VerifiedKeyMaterial::Classical)
                .ok_or_else(|| {
                    hyprstream_vfs::MountError::PermissionDenied(
                        "verified claims omitted clearance".to_owned(),
                    )
                })?;
            let identity = VerifiedAttachIdentity::from_verified_credential(
                self.claims.sub.clone(),
                tenant.to_owned(),
            );
            let token_scope = VerifiedTokenScope::from_verified_token(
                *context.clearance(),
                Arc::from([
                    NinePAction::Attach,
                    NinePAction::Walk,
                    NinePAction::Open,
                    NinePAction::Read,
                    NinePAction::Getattr,
                    NinePAction::Readdir,
                ]),
                std::time::Instant::now() + std::time::Duration::from_secs(300),
            );
            let session =
                SessionContext::from_verified_token(identity.clone(), context, token_scope);
            VerifiedAttach::try_new(
                identity,
                hyprstream_rpc::Subject::new(self.claims.sub.clone()),
                session,
            )
        }
    }
    let ninep_backend = MemoryBackend::default();
    ninep_backend.add_file("/srv", SECRET_BYTES);
    let ninep_audit = Arc::new(SpyAudit::default());
    let ninep_decider: Arc<dyn AccessDecider> = Arc::new(NinePAccessDecider::new(
        Arc::new(hyprstream_core::mac::GenesisGate::production().into_resolver()),
        ninep_audit,
    ));
    let monitor = hyprstream_core::mac::production_ninep_reference_monitor(
        Arc::clone(&ninep_decider),
        Arc::new(VerifiedClaimsAttach {
            token: session_token.clone(),
            claims: claims.clone(),
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let translator = Translator::new(Arc::new(ninep_backend), ninep_decider)
        .with_reference_monitor(monitor)
        .with_activation_control();
    let ninep_server = tokio::spawn(async move {
        let _ = translator.serve(listener).await;
    });
    let mut ninep_client = TcpStream::connect(address).await?;
    let _ = ninep_rpc(&mut ninep_client, msg::tversion(11, 4096, "9P2000.L")).await;
    let attach = ninep_rpc(
        &mut ninep_client,
        msg::tattach(12, 0, u32::MAX, &session_token, "/"),
    )
    .await;
    let ninep_bytes = if matches!(attach, Response::Attach { .. }) {
        let walked = ninep_rpc(&mut ninep_client, msg::twalk(13, 0, 1, &["srv"])).await;
        if matches!(walked, Response::Walk { .. })
            && matches!(
                ninep_rpc(&mut ninep_client, msg::tlopen(14, 1, 0)).await,
                Response::Lopen { .. }
            )
        {
            match ninep_rpc(&mut ninep_client, msg::tread(15, 1, 0, 4096)).await {
                Response::Read { data } => data,
                Response::Error { .. } => Vec::new(),
                other => anyhow::bail!("unexpected 9P read response: {other:?}"),
            }
        } else {
            Vec::new()
        }
    } else {
        Vec::new()
    };

    ninep_server.abort();
    oauth_server.abort();
    policy_handle.stop().await?;
    drop(rpc_processor);

    Ok(T8SessionCredential {
        subject: claims.sub,
        tenant,
        clearance,
        composite_signed: protected.alg == "ML-DSA-65-Ed25519" && protected.typ == "at+jwt",
        rpc_bytes,
        ninep_bytes,
    })
}

struct AuthenticatedGateEcho {
    name: &'static str,
    transport: TransportConfig,
    signing_key: SigningKey,
    invocations: Arc<AtomicUsize>,
    key_source: Arc<dyn hyprstream_rpc::auth::JwtKeySource>,
}

#[async_trait(?Send)]
impl RequestService for AuthenticatedGateEcho {
    fn decode_request_body(
        &self,
        signed_body: &[u8],
    ) -> Result<hyprstream_rpc::service::DecodedRequestBody> {
        Ok(hyprstream_rpc::service::DecodedRequestBody::opaque(
            signed_body.to_vec(),
        ))
    }

    async fn handle_request(
        &self,
        _ctx: &EnvelopeContext,
        body: &hyprstream_rpc::service::DecodedRequestBody,
    ) -> Result<(Vec<u8>, Option<Continuation>)> {
        self.invocations.fetch_add(1, Ordering::SeqCst);
        Ok((body.bytes().to_vec(), None))
    }

    fn name(&self) -> &str {
        self.name
    }

    fn transport(&self) -> &TransportConfig {
        &self.transport
    }

    fn signing_key(&self) -> SigningKey {
        self.signing_key.clone()
    }

    fn jwt_key_source(&self) -> Option<Arc<dyn hyprstream_rpc::auth::JwtKeySource>> {
        Some(Arc::clone(&self.key_source))
    }
}

fn html_hidden_value<'a>(html: &'a str, name: &str) -> Result<&'a str> {
    let prefix = format!(r#"name="{name}" value=""#);
    let rest = html
        .split_once(&prefix)
        .with_context(|| format!("authorize page omitted hidden {name}"))?
        .1;
    rest.split_once('"')
        .map(|(value, _)| value)
        .with_context(|| format!("authorize page malformed hidden {name}"))
}

fn es256_dpop_proof(
    signing_key: &p256::ecdsa::SigningKey,
    htu: &str,
    jti: &str,
    nonce: Option<&str>,
) -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    use p256::ecdsa::signature::Signer as _;

    let point = signing_key.verifying_key().to_encoded_point(false);
    let header = serde_json::json!({
        "typ": "dpop+jwt",
        "alg": "ES256",
        "jwk": {
            "kty": "EC",
            "crv": "P-256",
            "x": URL_SAFE_NO_PAD.encode(point.x().unwrap()),
            "y": URL_SAFE_NO_PAD.encode(point.y().unwrap()),
        }
    });
    let mut payload = serde_json::json!({
        "jti": jti,
        "htm": "POST",
        "htu": htu,
        "iat": chrono::Utc::now().timestamp(),
    });
    if let Some(nonce) = nonce {
        payload["nonce"] = serde_json::Value::String(nonce.to_owned());
    }
    let header = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header).unwrap());
    let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap());
    let signing_input = format!("{header}.{payload}");
    let signature: p256::ecdsa::Signature = signing_key.sign(signing_input.as_bytes());
    format!(
        "{signing_input}.{}",
        URL_SAFE_NO_PAD.encode(signature.to_bytes())
    )
}

fn ed25519_dpop_proof(
    signing_key: &SigningKey,
    htu: &str,
    jti: &str,
    nonce: Option<&str>,
) -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
    use ed25519_dalek::Signer as _;

    let header = serde_json::json!({
        "typ": "dpop+jwt",
        "alg": "EdDSA",
        "jwk": {
            "kty": "OKP",
            "crv": "Ed25519",
            "x": URL_SAFE_NO_PAD.encode(signing_key.verifying_key().to_bytes()),
        }
    });
    let mut payload = serde_json::json!({
        "jti": jti,
        "htm": "POST",
        "htu": htu,
        "iat": chrono::Utc::now().timestamp(),
    });
    if let Some(nonce) = nonce {
        payload["nonce"] = serde_json::Value::String(nonce.to_owned());
    }
    let header = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header).unwrap());
    let payload = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap());
    let signing_input = format!("{header}.{payload}");
    let signature: ed25519_dalek::Signature = signing_key.sign(signing_input.as_bytes());
    format!(
        "{signing_input}.{}",
        URL_SAFE_NO_PAD.encode(signature.to_bytes())
    )
}

fn configure_policy_signing_authority() -> Result<()> {
    use hyprstream_rpc::auth::{CompositeKeyPair, CompositePairRole, CompositePairState};

    let authority_dir = tempfile::TempDir::new()?.keep();
    let ledger = authority_dir.join("ledger.json");
    let committed = authority_dir.join("committed");
    let prefix = authority_dir.join("committed-ledger");
    let lock = authority_dir.join("ledger.lock");
    let key_set = hyprstream_rpc::auth::global_composite_key_set();
    let version = key_set.snapshot().version() + 1;
    let digest = format!("mac-gate-{version}");
    let generation = serde_json::to_vec(&serde_json::json!({
        "version": version,
        "component_digest": digest,
    }))?;
    std::fs::write(&ledger, &generation)?;
    std::fs::write(&committed, &generation)?;
    std::fs::write(
        authority_dir.join(format!("committed-ledger-{version}-{digest}.json")),
        &generation,
    )?;
    let (ml_signing, ml_verifying) = hyprstream_rpc::crypto::pq::ml_dsa_generate_keypair();
    let ed_signing = Arc::new(SigningKey::from_bytes(&[0x51; 32]));
    let kid = hyprstream_rpc::auth::composite_kid(&ml_verifying, &ed_signing.verifying_key());
    let now = chrono::Utc::now().timestamp();
    let pair = CompositeKeyPair::signing(
        kid,
        Arc::new(ml_signing),
        ed_signing,
        CompositePairRole::Policy,
        CompositePairState::Active,
        now - 60,
        now + 86_400,
    );
    key_set.configure_authority(ledger, committed, prefix, lock);
    key_set.publish(version, digest, vec![pair])?;
    Ok(())
}
