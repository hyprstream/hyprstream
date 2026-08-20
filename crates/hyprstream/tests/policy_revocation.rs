//! Cross-process credential revocation over a real transport.
//!
//! The policy service is the canonical revocation authority: it owns the one
//! durable store; every other process holds a `PolicyAuthorityRevocationStore`
//! that checks and publishes through the policy service over the RPC bus.
//! These tests drive that topology end-to-end: a real `PolicyService` served
//! over iroh (same harness as `policy_over_iroh.rs`), with two independent
//! RPC client stores simulating two non-policy processes.

mod support;

use std::sync::Arc;

use anyhow::Result;
use ed25519_dalek::SigningKey;
use rand::RngCore;
use tempfile::TempDir;
use tokio::sync::RwLock;

use git2db::Git2DB;
use hyprstream_core::auth::PolicyManager;
use hyprstream_core::config::TokenConfig;
use hyprstream_core::services::PolicyService;
use hyprstream_core::services::generated::policy_client::PolicyClient;
use hyprstream_core::services::revocation::PolicyAuthorityRevocationStore;

use hyprstream_rpc::auth::{
    CredentialId, CredentialRevocationStore, FileBackedCredentialRevocationStore,
    global_credential_revocation_store, set_global_credential_revocation_store,
};
use hyprstream_rpc::crypto::CryptoPolicy;
use hyprstream_rpc::crypto::hybrid_kem::{KemTrustStore, KeyedKemTrustStore};
use hyprstream_rpc::envelope::{
    EnvelopeVerifyConfig, InMemoryNonceCache, KeyedPqTrustStore, PqTrustStore, install_verify_config,
};
use hyprstream_rpc::node_identity::{derive_mesh_kem_recipient, derive_mesh_mldsa_key};
use hyprstream_rpc::rpc_client::RpcClientImpl;
use hyprstream_rpc::signer::LocalSigner;
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::transport::iroh_rpc::{IrohRpcProtocolHandler, LocalServiceBridge};
use hyprstream_rpc::transport::iroh_substrate::{ALPN_HYPRSTREAM_RPC, IrohSubstrate, NoopHandler};
use hyprstream_rpc::transport::iroh_transport::IrohTransport;

use iroh::{EndpointAddr, TransportAddr};

const SERVER_SIGNING_SEED: [u8; 32] = [0x61; 32];
const PROCESS_A_SIGNING_SEED: [u8; 32] = [0xA1; 32];
const PROCESS_B_SIGNING_SEED: [u8; 32] = [0xB1; 32];
const PROCESS_C_SIGNING_SEED: [u8; 32] = [0xC1; 32];

fn server_signing_key() -> SigningKey {
    SigningKey::from_bytes(&SERVER_SIGNING_SEED)
}

fn process_a_signing_key() -> SigningKey {
    SigningKey::from_bytes(&PROCESS_A_SIGNING_SEED)
}

fn process_b_signing_key() -> SigningKey {
    SigningKey::from_bytes(&PROCESS_B_SIGNING_SEED)
}

/// Process C is never seeded into the subject trust store: its envelope
/// verifies (its mesh anchor is bound below) but it resolves to no service
/// identity — the anonymous-caller case for the scope gates.
fn process_c_signing_key() -> SigningKey {
    SigningKey::from_bytes(&PROCESS_C_SIGNING_SEED)
}

fn bind_mesh_anchor(store: &mut KeyedPqTrustStore, signing_key: &SigningKey) {
    let pq_signing_key = derive_mesh_mldsa_key(signing_key);
    store.bind(
        signing_key.verifying_key().to_bytes(),
        &ml_dsa::Keypair::verifying_key(&pq_signing_key),
    );
}

fn pq_trust_store() -> Arc<dyn PqTrustStore> {
    let mut store = KeyedPqTrustStore::new();
    bind_mesh_anchor(&mut store, &server_signing_key());
    bind_mesh_anchor(&mut store, &process_a_signing_key());
    bind_mesh_anchor(&mut store, &process_b_signing_key());
    bind_mesh_anchor(&mut store, &process_c_signing_key());
    Arc::new(store)
}

fn request_kem_store(server_signing: &SigningKey) -> Result<Arc<dyn KemTrustStore>> {
    let mut store = KeyedKemTrustStore::new();
    store.bind(
        server_signing.verifying_key().to_bytes(),
        derive_mesh_kem_recipient(server_signing)?.public(),
    );
    Ok(Arc::new(store))
}

fn install_hybrid_verify_config() -> Arc<dyn PqTrustStore> {
    let pq_store = pq_trust_store();
    let _ = install_verify_config(EnvelopeVerifyConfig {
        policy: CryptoPolicy::Hybrid,
        pq_store: Some(Arc::clone(&pq_store)),
    });
    pq_store
}

fn fresh_node_key() -> [u8; 32] {
    let mut k = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut k);
    k
}

fn direct_addr(substrate: &IrohSubstrate) -> EndpointAddr {
    EndpointAddr::from_parts(
        substrate.endpoint_id(),
        substrate
            .endpoint()
            .bound_sockets()
            .into_iter()
            .map(TransportAddr::Ip),
    )
}

async fn make_policy_service() -> Result<(PolicyService, SigningKey, TempDir)> {
    let temp = TempDir::new()?;
    let models_dir = temp.path().to_path_buf();
    let policies_dir = models_dir.join(".registry").join("policies");
    let policy_manager = Arc::new(PolicyManager::new(&policies_dir).await?);
    let git2db = Arc::new(RwLock::new(Git2DB::open(&models_dir).await?));
    let signing_key = server_signing_key();
    let service = PolicyService::new(
        Arc::clone(&policy_manager),
        Arc::new(signing_key.clone()),
        TokenConfig::default(),
        git2db,
        TransportConfig::inproc("policy-revocation-unused"),
    );
    Ok((service, signing_key, temp))
}

async fn serve_over_iroh(
    service: PolicyService,
    server_signing: &SigningKey,
) -> Result<(IrohSubstrate, EndpointAddr)> {
    let nonce_cache = Arc::new(InMemoryNonceCache::new());
    let bridge = LocalServiceBridge::spawn(service, nonce_cache, 0)?;
    let substrate = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("moq-not-wired"),
        IrohRpcProtocolHandler::new(bridge, server_signing.clone()),
    )
    .await?;
    let addr = direct_addr(&substrate);
    Ok((substrate, addr))
}

/// Build one simulated non-policy process: an iroh client whose
/// `PolicyAuthorityRevocationStore` delegates to the authority over RPC.
async fn revocation_store_for(
    server_addr: EndpointAddr,
    server_vk: ed25519_dalek::VerifyingKey,
    signing_key: SigningKey,
) -> Result<(IrohSubstrate, PolicyAuthorityRevocationStore)> {
    let client_substrate = IrohSubstrate::new(
        fresh_node_key(),
        NoopHandler::new("c-moq"),
        NoopHandler::new("c-rpc"),
    )
    .await?;
    let conn = client_substrate
        .connect(server_addr, ALPN_HYPRSTREAM_RPC)
        .await?;
    let transport = IrohTransport::new(conn);
    let rpc = RpcClientImpl::new(LocalSigner::new(signing_key), transport, Some(server_vk))
        .with_request_kem_store(request_kem_store(&server_signing_key())?)
        .with_response_pq_store(pq_trust_store());
    let client = PolicyClient::new(Arc::new(rpc));
    Ok((client_substrate, PolicyAuthorityRevocationStore::new(client)))
}

/// Publish the authority-side durable store (tempdir-backed) as the
/// process-global store — the policy handlers resolve the store through the
/// global. Get-or-init: the OnceLock is per test-binary process. The returned
/// bool is `true` when THIS call installed the global (its file is then the
/// authoritative log); when `false`, an earlier test's store is the live one.
fn install_authority_store(temp: &TempDir) -> Result<(Arc<FileBackedCredentialRevocationStore>, bool)> {
    let path = temp.path().join("credential-revocations.jsonl");
    let store = Arc::new(FileBackedCredentialRevocationStore::open(&path)?);
    let mut installed_here = false;
    if global_credential_revocation_store().is_none() {
        set_global_credential_revocation_store(store.clone())
            .map_err(|e| anyhow::anyhow!("no other test in this binary publishes the global store: {e}"))?;
        installed_here = true;
    }
    Ok((store, installed_here))
}

/// Bind the simulated process identities in the process-global trust store,
/// which `PolicyService::resolve_key_subject` consults on the networked path:
/// process A resolves as `service:oauth` (the authorized revocation
/// publisher via the mandatory base policy rule — no explicit grant needed),
/// process B as `service:other` (no grant). Idempotent fixed-seed inserts.
fn install_test_service_identities() {
    let store = hyprstream_service::global_trust_store();
    for (key, scope) in [
        (process_a_signing_key(), "oauth"),
        (process_b_signing_key(), "other"),
    ] {
        store.insert(
            key.verifying_key(),
            hyprstream_service::Attestation {
                scopes: std::iter::once(scope.to_owned()).collect(),
                subject: None,
                jwt: None,
                expires_at: 0, // never expires
                attested_by: None,
            },
        );
    }
}

/// Serializes the cache-mutating tests in this binary (the verified-subject
/// cache and the MAC activation control are process-global).
static CACHE_TEST_GATE: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// Widen the process-global MAC activation control to IdentityAware so a
/// cache miss is observable as `None` (FloorOnly maps every miss to the
/// anonymous floor). One-shot; widening flushes the cache generation.
fn ensure_identity_aware_activation() {
    use hyprstream_rpc::auth::mac::{
        MacActivationEvidence, MacActivationMode, global_mac_activation_control,
    };

    let control = global_mac_activation_control();
    if control.mode() == MacActivationMode::IdentityAware {
        return;
    }
    let genesis = hyprstream_rpc::auth::GenesisReport {
        labeled: vec!["/".to_owned()],
        unlabeled: Vec::new(),
        ill_formed: Vec::new(),
    };
    if let Err(e) = control.widen_identity_aware(&MacActivationEvidence {
        genesis: &genesis,
        mediation_integrity_g2: true,
        denial_handling_g4: true,
        observability_g5: true,
        runbook_signoff_g6: true,
        revocation_reload_g7: true,
    }) {
        panic!("complete activation evidence widens: {e}");
    }
}

/// Seed a credential-bearing verified-subject cache entry for `subject_name`.
fn remember_cached_subject(issuer: &str, jti: &str, subject_name: &str) -> CredentialId {
    use hyprstream_rpc::auth::mac::{
        Assurance, CompartmentSet, Level, SecurityLabel, VerifiedKeyMaterial,
        remember_verified_claims_with_credential,
    };

    let now = chrono::Utc::now().timestamp();
    let cred_id = CredentialId::jwt(issuer, jti);
    let mut claims = hyprstream_rpc::auth::Claims::new(subject_name.to_owned(), now, now + 300)
        .with_clearance(SecurityLabel::new(
            Level::Secret,
            Assurance::Classical,
            CompartmentSet::EMPTY,
        ));
    claims.iss = issuer.to_owned();
    claims.jti = Some(jti.to_owned());
    let subject = hyprstream_rpc::envelope::Subject::new(subject_name);
    remember_verified_claims_with_credential(
        &subject,
        &claims,
        VerifiedKeyMaterial::Classical,
        None,
        cred_id.clone(),
    );
    cred_id
}

/// THE cross-process proof: process A publishes a revocation over RPC;
/// process B observes it over RPC. Issuer scoping and the JWT/CWT typed
/// namespaces survive the wire round-trip. Publication is gated: only the
/// OAuth revocation authority identity (`service:oauth`) may publish.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn revocation_is_visible_across_processes() -> Result<()> {
    support::install_explicit_dispatch_pep();
    install_hybrid_verify_config();
    install_test_service_identities();

    let store_temp = TempDir::new()?;
    let (_authority, installed_here) = install_authority_store(&store_temp)?;

    let (service, server_signing, _temp) = make_policy_service().await?;
    let server_vk = server_signing.verifying_key();
    let (server, server_addr) = serve_over_iroh(service, &server_signing).await?;

    let (substrate_a, process_a) =
        revocation_store_for(server_addr.clone(), server_vk, process_a_signing_key()).await?;
    let (substrate_b, process_b) =
        revocation_store_for(server_addr.clone(), server_vk, process_b_signing_key()).await?;

    let issuer = "https://issuer.example";
    let id = CredentialId::jwt(issuer, "jti-cross-process");
    let exp = chrono::Utc::now().timestamp() + 3600;

    // Unrevoked ID reads false from both processes.
    assert!(
        !process_b.is_revoked(&id).await,
        "unrevoked credential must read false"
    );

    // Process A revokes; process B observes it.
    process_a.revoke_credential(id.clone(), exp).await?;
    assert!(
        process_b.is_revoked(&id).await,
        "process B must observe process A's revocation"
    );

    // Cross-issuer same-jti is NOT revoked (issuer scoping over the wire).
    let other_issuer = CredentialId::jwt("https://other.example", "jti-cross-process");
    assert!(
        !process_b.is_revoked(&other_issuer).await,
        "cross-issuer same-jti must NOT be revoked"
    );

    // CWT cti bytes vs the same bytes as a JWT jti: disjoint namespaces over
    // the wire.
    let cwt_id = CredentialId::cwt(issuer, b"same-identifier-bytes".to_vec());
    process_a.revoke_credential(cwt_id.clone(), exp).await?;
    assert!(
        process_b.is_revoked(&cwt_id).await,
        "CWT cti revocation must be visible cross-process"
    );
    let jwt_same_bytes = CredentialId::jwt(issuer, "same-identifier-bytes");
    assert!(
        !process_b.is_revoked(&jwt_same_bytes).await,
        "JWT jti with the same bytes must NOT collide with the CWT cti"
    );

    // NEGATIVE: process B (service:other, no grant) cannot publish — the
    // Casbin gate on `policy:RevokeCredential` rejects it before the handler.
    let denied_id = CredentialId::jwt(issuer, "jti-denied-publish");
    let Err(err) = process_b.revoke_credential(denied_id.clone(), exp).await else {
        panic!("service:other must be denied by the publication gate");
    };
    assert!(
        err.to_string().contains("Unauthorized"),
        "denial must be an authorization failure, got: {err}"
    );
    assert!(
        !process_b.is_revoked(&denied_id).await,
        "a denied publication must not become visible"
    );
    if installed_here {
        let log = std::fs::read_to_string(store_temp.path().join("credential-revocations.jsonl"))?;
        assert!(
            !log.contains("jti-denied-publish"),
            "a denied publication must never reach the durable log"
        );
    }

    // Server-side publication bounds: a caller-controlled far-future exp
    // (never GC'd, never dropped on load) or an oversized ID field must be
    // rejected, not stored — even for the authorized publisher.
    let permanent_id = CredentialId::jwt(issuer, "jti-permanent");
    assert!(
        process_a
            .revoke_credential(permanent_id.clone(), i64::MAX)
            .await
            .is_err(),
        "expires_at beyond the 45-day bound must be rejected"
    );
    assert!(
        !process_b.is_revoked(&permanent_id).await,
        "a rejected publication must not become visible"
    );
    let oversized_id = CredentialId::jwt(issuer, "x".repeat(2048));
    assert!(
        process_a
            .revoke_credential(oversized_id.clone(), exp)
            .await
            .is_err(),
        "an oversized jti must be rejected"
    );
    assert!(
        !process_b.is_revoked(&oversized_id).await,
        "an oversized-id publication must not become visible"
    );

    // Anonymous caller (envelope verifies, but no service identity): both the
    // publication gate and the service-only check gate must deny.
    let (substrate_c, process_c) =
        revocation_store_for(server_addr.clone(), server_vk, process_c_signing_key()).await?;
    let anon_id = CredentialId::jwt(issuer, "jti-anonymous");
    assert!(
        process_c
            .revoke_credential(anon_id.clone(), exp)
            .await
            .is_err(),
        "anonymous publication must be denied"
    );
    // The check gate denies anonymous callers; the client store maps that
    // denial to fail-closed `true` — indistinguishable from "revoked" for a
    // never-issued ID, and the authorized read of the same ID stays false.
    assert!(
        process_c.is_revoked(&anon_id).await,
        "anonymous check must fail closed (gate denial → true)"
    );
    assert!(
        !process_b.is_revoked(&anon_id).await,
        "authorized check of the same never-issued ID reads false"
    );
    drop(process_c);
    substrate_c.shutdown().await?;

    drop(process_a);
    drop(process_b);
    substrate_a.shutdown().await?;
    substrate_b.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}

/// A revocation client store whose authority is unreachable fails closed:
/// checks report revoked and publications return an error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unreachable_authority_fails_closed() -> Result<()> {
    use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;

    let signing_key = process_a_signing_key();
    let remote_vk = server_signing_key().verifying_key();
    let rpc = RpcClientImpl::new(
        LocalSigner::new(signing_key),
        LazyUdsTransport::new("/dev/null/policy-revocation-dead.sock".into()),
        Some(remote_vk),
    );
    let store = PolicyAuthorityRevocationStore::new(PolicyClient::new(Arc::new(rpc)));

    let id = CredentialId::jwt("https://issuer.example", "jti-unreachable");
    assert!(
        store.is_revoked(&id).await,
        "check against a dead authority must fail closed"
    );
    let result = store
        .revoke_credential(id, chrono::Utc::now().timestamp() + 3600)
        .await;
    assert!(
        result.is_err(),
        "publication against a dead authority must fail"
    );
    Ok(())
}

/// The authority's durable store: revocations survive a process restart
/// (drop + re-open from the same path), expired entries are dropped on load,
/// and a corrupt file fails closed at open.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn file_backed_store_is_durable() -> Result<()> {
    let temp = TempDir::new()?;
    let path = temp.path().join("credential-revocations.jsonl");
    let now = chrono::Utc::now().timestamp();

    let live = CredentialId::jwt("https://issuer.example", "jti-durable");
    let live_cwt = CredentialId::cwt("https://issuer.example", b"cti-durable".to_vec());
    let expired = CredentialId::jwt("https://issuer.example", "jti-expired");
    {
        let store = FileBackedCredentialRevocationStore::open(&path)?;
        store.revoke_credential(live.clone(), now + 3600).await?;
        store.revoke_credential(live_cwt.clone(), now + 3600).await?;
        store.revoke_credential(expired.clone(), now - 10).await?;
    }

    // Re-open from the same path: live entries survive, expired are dropped.
    let reopened = FileBackedCredentialRevocationStore::open(&path)?;
    assert!(reopened.is_revoked(&live).await, "JWT revocation must survive reopen");
    assert!(
        reopened.is_revoked(&live_cwt).await,
        "CWT revocation must survive reopen"
    );
    assert!(
        !reopened.is_revoked(&expired).await,
        "expired entries are dropped on load"
    );
    drop(reopened);

    // Corrupt the log: open must fail closed.
    std::fs::write(&path, b"{not json\n")?;
    assert!(
        FileBackedCredentialRevocationStore::open(&path).is_err(),
        "corrupt revocation log must fail open"
    );
    Ok(())
}

/// Crash recovery: a crash mid-append leaves a torn tail (partial final
/// record, no newline). Open must truncate to the last complete record and
/// recover — not brick every subsequent boot. A parseable tail fragment (the
/// newline itself was torn off) is salvaged and re-appended; a malformed
/// MID-FILE line is genuine corruption and still fails closed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn file_backed_store_recovers_torn_tail() -> Result<()> {
    let temp = TempDir::new()?;
    let path = temp.path().join("credential-revocations.jsonl");
    let now = chrono::Utc::now().timestamp();

    let id1 = CredentialId::jwt("https://issuer.example", "jti-complete");
    let id2 = CredentialId::jwt("https://issuer.example", "jti-torn-newline");

    // Learn the canonical serialized line for id2 from a scratch store.
    let scratch = TempDir::new()?;
    let id2_line = {
        let scratch_store =
            FileBackedCredentialRevocationStore::open(&scratch.path().join("s.jsonl"))?;
        scratch_store
            .revoke_credential(id2.clone(), now + 3600)
            .await?;
        std::fs::read_to_string(scratch.path().join("s.jsonl"))?
    };

    // One complete record, then a crash tears the second append mid-record.
    let complete_content = {
        let store = FileBackedCredentialRevocationStore::open(&path)?;
        store.revoke_credential(id1.clone(), now + 3600).await?;
        std::fs::read(&path)?
    };
    {
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new().append(true).open(&path)?;
        f.write_all(b"{\"iss\":\"https://issuer.example\",\"jwt\":\"jti-part")?; // torn
    }
    let recovered = FileBackedCredentialRevocationStore::open(&path)?;
    assert!(
        recovered.is_revoked(&id1).await,
        "complete records survive torn-tail recovery"
    );
    assert_eq!(
        std::fs::read(&path)?,
        complete_content,
        "the torn fragment is truncated to the last complete newline"
    );
    drop(recovered);

    // A tail that is a complete record missing only its newline is salvaged:
    // kept, and re-appended with the newline restored.
    {
        use std::io::Write as _;
        let mut f = std::fs::OpenOptions::new().append(true).open(&path)?;
        f.write_all(id2_line.trim_end_matches('\n').as_bytes())?;
    }
    let salvaged = FileBackedCredentialRevocationStore::open(&path)?;
    assert!(
        salvaged.is_revoked(&id1).await && salvaged.is_revoked(&id2).await,
        "the newline-torn record must be salvaged, not lost"
    );
    let content = std::fs::read_to_string(&path)?;
    assert!(content.ends_with('\n'), "salvage restores the trailing newline");
    assert_eq!(
        content.matches("jti-torn-newline").count(),
        1,
        "the salvaged record appears exactly once"
    );
    drop(salvaged);

    // A malformed MID-FILE line is corruption, not a crash artifact: fail.
    std::fs::write(&path, b"{\"iss\":\"x\",\"jwt\":\"a\",\"exp\":9999999999}\ngarbage\n")?;
    assert!(
        FileBackedCredentialRevocationStore::open(&path).is_err(),
        "mid-file corruption must fail closed"
    );
    Ok(())
}

/// A subject context cached in a distinct verifier process must stop
/// resolving once the credential it was derived from is revoked through the
/// authority. The cache is process-global in this test binary, so "process B"
/// is logical; the authority round-trip (A publishes over RPC, the
/// revalidating read observes the canonical store) is what's proven.
// await_holding_lock: CACHE_TEST_GATE deliberately serializes this cache-mutating
// test for its whole duration, including the RPC round trips.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cached_subject_revalidation_observes_cross_process_revocation() -> Result<()> {
    let _cache_guard = CACHE_TEST_GATE.lock();
    support::install_explicit_dispatch_pep();
    install_hybrid_verify_config();
    install_test_service_identities();
    ensure_identity_aware_activation();
    hyprstream_rpc::auth::mac::flush_verified_subject_cache_generation();

    let store_temp = TempDir::new()?;
    let (_authority, _installed) = install_authority_store(&store_temp)?;

    let (service, server_signing, _temp) = make_policy_service().await?;
    let server_vk = server_signing.verifying_key();
    let (server, server_addr) = serve_over_iroh(service, &server_signing).await?;
    let (substrate_a, process_a) =
        revocation_store_for(server_addr.clone(), server_vk, process_a_signing_key()).await?;
    let (substrate_b, process_b) =
        revocation_store_for(server_addr, server_vk, process_b_signing_key()).await?;

    // B-side cache entry derived from a credential-bearing token.
    let issuer = "https://issuer.example";
    let cred_id = remember_cached_subject(issuer, "jti-cached-subject", "did:web:cached");
    let subject = hyprstream_rpc::envelope::Subject::new("did:web:cached");
    let exp = chrono::Utc::now().timestamp() + 300;

    // Positive control: the authority has no revocation for this credential,
    // so the revalidating read returns the cached context.
    let ctx = hyprstream_rpc::auth::mac::subject_context(&subject, None).await;
    assert!(
        ctx.is_some(),
        "live credential's cached context must resolve"
    );

    // Process A (service:oauth) publishes the revocation over RPC; process
    // B's client store observes it (the canonical store is shared).
    process_a.revoke_credential(cred_id.clone(), exp).await?;
    assert!(
        process_b.is_revoked(&cred_id).await,
        "the revocation must be visible over RPC before the read-side check"
    );

    // The next revalidating read evicts the cached entry and misses.
    assert!(
        hyprstream_rpc::auth::mac::subject_context(&subject, None)
            .await
            .is_none(),
        "revoked credential's cached context must not resolve"
    );
    assert_eq!(
        hyprstream_rpc::auth::mac::revoke_verified_subject_credential(&cred_id),
        0,
        "entry was already evicted by the revalidating read"
    );

    drop(process_a);
    drop(process_b);
    substrate_a.shutdown().await?;
    substrate_b.shutdown().await?;
    server.shutdown().await?;
    Ok(())
}

/// A credential-bearing cached entry read against a DEAD revocation authority
/// denies: the client store's own fail-closed `true` drives the revalidating
/// read to evict the entry and miss.
// await_holding_lock: see cached_subject_revalidation_observes_cross_process_revocation.
#[allow(clippy::await_holding_lock)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cached_subject_read_fails_closed_with_dead_authority() -> Result<()> {
    use hyprstream_rpc::transport::lazy_uds::LazyUdsTransport;

    let _cache_guard = CACHE_TEST_GATE.lock();
    ensure_identity_aware_activation();
    hyprstream_rpc::auth::mac::flush_verified_subject_cache_generation();

    let cred_id = remember_cached_subject(
        "https://issuer.example",
        "jti-dead-authority",
        "did:web:dead-authority",
    );
    let subject = hyprstream_rpc::envelope::Subject::new("did:web:dead-authority");

    let rpc = RpcClientImpl::new(
        LocalSigner::new(process_a_signing_key()),
        LazyUdsTransport::new("/dev/null/policy-revocation-dead.sock".into()),
        Some(server_signing_key().verifying_key()),
    );
    let dead_store = PolicyAuthorityRevocationStore::new(PolicyClient::new(Arc::new(rpc)));

    assert!(
        hyprstream_rpc::auth::mac::subject_context_with(Some(&dead_store), &subject, None)
            .await
            .is_none(),
        "dead authority must fail the credential-bearing cached read closed"
    );
    assert_eq!(
        hyprstream_rpc::auth::mac::revoke_verified_subject_credential(&cred_id),
        0,
        "entry was evicted by the failed revalidation"
    );
    Ok(())
}
