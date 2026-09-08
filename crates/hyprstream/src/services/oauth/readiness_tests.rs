//! Causal readiness tests for the OAuth startup transaction (#1585 YuI7).
//!
//! Every run()-level case drives the real production startup seam — the same
//! `Spawnable::run` the foreground launcher calls — against real sockets.
//! Process-global state (credentials/config env, the install-once iroh client
//! endpoint) is only touched inside isolated child processes (the same
//! re-exec pattern as [`super::required_consumer_tests`]); the parent-side
//! cases touch no process globals.
//!
//! Scope against the Sol carrier disposition (pr1585-oauth-bootstrap-
//! carrier-disposition-sol): the helper-level carrier cases — real
//! bootstrap-first install classification (`ExistingGlobalRetained`),
//! endpoint-ID distinctness, retain-and-independent-ownership with real
//! outbound RPC, and the empty-slot `InstalledHere` control — live in
//! [`super::required_consumer_tests`] and
//! [`oauth_endpoint_install_wins_empty_slot`]. Those helpers are the exact
//! seam `run` calls, but helper coverage does not itself drive
//! `OAuthService::run` through READY; that run()-level Required positive
//! control is [`run_required_bootstrap_first_ready_then_http`]. The plain
//! positive HTTP transaction ([`run_ready_then_real_http_and_bounded_shutdown`])
//! is Compatibility-profile by construction.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;
use ed25519_dalek::SigningKey;
use hyprstream_rpc::node_identity::derive_mesh_mldsa_key;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::time::Duration;

/// Reserve an ephemeral loopback port for a later real bind. There is an
/// inherent TOCTOU window between the probe and the service bind; that is
/// accepted (and bounded) by keeping the reserve short-lived.
fn reserve_ephemeral_port() -> SocketAddr {
    let l = std::net::TcpListener::bind("127.0.0.1:0").expect("reserve ephemeral port");
    l.local_addr().expect("local addr")
}

/// Hold a loopback port so a service bind on it fails with a real collision.
/// The binding lives until the end of the calling scope.
fn occupy_port(addr: SocketAddr) -> std::net::TcpListener {
    std::net::TcpListener::bind(addr).expect("occupy port")
}

/// Insert the policy/discovery verifying keys `run()` requires from the
/// global trust store, as the launcher's depends_on population would.
fn seed_trust_store() {
    use hyprstream_service::Attestation;
    let store = hyprstream_service::global_trust_store();
    for (name, key) in [
        ("policy", SigningKey::from_bytes(&[0xB1; 32])),
        ("discovery", SigningKey::from_bytes(&[0xB2; 32])),
    ] {
        store.insert(
            key.verifying_key(),
            Attestation {
                scopes: [name.to_owned()].into_iter().collect(),
                subject: None,
                jwt: None,
                expires_at: 0,
                attested_by: None,
            },
        );
    }
}

/// Build an OAuthService at `addr` exactly as the production factory does,
/// with TLS OFF (plain-HTTP arm) and an IPC control transport.
fn http_service_at(
    oauth: &SigningKey,
    addr: SocketAddr,
    control: TransportConfig,
) -> OAuthService {
    let mut tls = crate::config::TlsConfig::default();
    tls.enabled = false;
    service_at(oauth, addr, control, tls)
}

fn service_at(
    oauth: &SigningKey,
    addr: SocketAddr,
    control: TransportConfig,
    tls: crate::config::TlsConfig,
) -> OAuthService {
    let mut oauth_config = crate::config::OAuthConfig::default();
    oauth_config.host = addr.ip().to_string();
    oauth_config.port = addr.port();
    let peer = TransportConfig::ipc("/tmp/oauth-readiness-unused-peer.sock");
    OAuthService::new(
        oauth_config,
        tls,
        crate::account::AccountZoneConfig::default(),
        oauth.clone(),
        control,
        peer.clone(),
        peer,
        oauth.verifying_key(),
        oauth.verifying_key(),
    )
}

/// Test-only `Spawnable` wrapper: delegates name/registrations/run to the
/// real OAuthService and signals `done_tx` only AFTER the inner `run` has
/// returned and dropped its state, so a bounded wait on the receiver proves
/// the actual service-thread teardown completed before the child's tempdir
/// is dropped.
struct RunCompletionService {
    inner: OAuthService,
    done_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl hyprstream_service::Spawnable for RunCompletionService {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn registrations(&self) -> Vec<(SocketKind, TransportConfig)> {
        self.inner.registrations()
    }

    fn run(
        mut self: Box<Self>,
        shutdown: Arc<Notify>,
        on_ready: Option<tokio::sync::oneshot::Sender<()>>,
    ) -> Result<(), hyprstream_rpc::error::RpcError> {
        let done_tx = self.done_tx.take();
        let result = Box::new(self.inner).run(shutdown, on_ready);
        if let Some(done_tx) = done_tx {
            let _ = done_tx.send(());
        }
        result
    }
}

/// Isolated-child preamble: fresh credentials/XDG/config env so the service's
/// startup filesystem surface (RocksDB token store, key stores, config load)
/// lands inside this test's tempdir and cannot race sibling tests. Each child
/// process runs exactly one `--exact` case.
struct ChildEnv {
    dir: tempfile::TempDir,
    /// Unix-socket paths are capped at `sun_len` (108 bytes); a control
    /// socket under the deep worktree tempdir exceeds that, so sockets live
    /// in a short-lived dir directly under the system temp root.
    sock_dir: tempfile::TempDir,
}

impl ChildEnv {
    fn new(tag: &str) -> Self {
        let dir = tempfile::Builder::new()
            .prefix(format!(".oauth-readiness-{tag}-").as_str())
            .tempdir_in(env!("CARGO_MANIFEST_DIR"))
            .expect("child tempdir");
        let sock_dir = tempfile::Builder::new()
            .prefix(format!(".oar-{tag}-").as_str())
            .tempdir_in(std::env::temp_dir())
            .expect("short child socket dir");
        let credentials = dir.path().join("credentials");
        std::fs::create_dir_all(&credentials).expect("credentials dir");
        std::env::set_var("CREDENTIALS_DIRECTORY", &credentials);
        std::env::set_var("HYPRSTREAM__SECRETS__PATH", &credentials);
        // The mandatory encrypted production profile requires the pglite
        // UserStore backend with real deployment age key material (the same
        // external-`age` seam production uses; both binaries are on PATH on
        // the Linux lanes this suite targets).
        std::env::set_var("HYPRSTREAM__CREDENTIALS__BACKEND", "pglite");
        let key_path = dir.path().join("userstore-age-identity.txt");
        let keygen = std::process::Command::new("age-keygen")
            .arg("-o")
            .arg(&key_path)
            .output()
            .expect("run age-keygen for the child UserStore keypair");
        assert!(
            keygen.status.success(),
            "age-keygen failed: {keygen:?}"
        );
        let stderr = String::from_utf8(keygen.stderr).expect("age-keygen stderr");
        let key_file =
            std::fs::read_to_string(&key_path).expect("read age-keygen identity file");
        let recipient = stderr
            .lines()
            .chain(key_file.lines())
            .find_map(|l| {
                l.strip_prefix("Public key: ")
                    .or_else(|| l.strip_prefix("# public key: "))
            })
            .expect("age-keygen must report the public key recipient")
            .to_owned();
        std::env::set_var("HYPRSTREAM_USERSTORE_AGE_RECIPIENTS", recipient);
        std::env::set_var("HYPRSTREAM_USERSTORE_AGE_IDENTITIES", &key_path);
        std::env::set_var("XDG_CONFIG_HOME", dir.path().join("xdg-config"));
        std::env::set_var("XDG_DATA_HOME", dir.path().join("xdg-data"));
        std::env::set_var("HYPRSTREAM_INSTANCE", format!("oauth-readiness-{tag}"));
        // Per-child pglite runtime cache: keeps every child's extracted
        // native runtime inside its own tempdir so no child depends on (or
        // observes) state left by another process.
        std::env::set_var("PGLITE_RUNTIME_DIR", dir.path().join("pglite-runtime"));
        Self { dir, sock_dir }
    }

    fn ipc_control(&self) -> TransportConfig {
        TransportConfig::ipc(self.sock_dir.path().join("control.sock"))
    }
}

/// Re-exec this test binary so `case` runs in its own process. Process
/// globals — the install-once iroh client endpoint OnceLock, the global trust
/// store, and the config env — are never reset cross-test.
fn run_isolated(case: &str, guard: &str) -> anyhow::Result<()> {
    if std::env::var_os(guard).is_some() {
        return Ok(());
    }
    // The pglite-patched standalone backend registers fd1 (stdout) as its
    // client socket with epoll while the UserStore opens
    // (pgl_startPGlite → pq_init): a regular file or /dev/null on stdout
    // makes epoll_ctl fail and the native boot raise a top-level ereport
    // that escapes as process exit(100) (no longjmp trampoline outside the
    // engine pump loop). The child therefore always gets PIPE stdio, which
    // is also what the foreground launcher and systemd (journald sockets)
    // give real services. Output is drained by the parent and surfaced only
    // on failure.
    let output = std::process::Command::new(std::env::current_exe()?)
        .args(["--exact", case, "--nocapture"])
        .env(guard, "1")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()?;
    anyhow::ensure!(
        output.status.success(),
        "isolated child {case} failed: {}\n--- child stderr (tail) ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr[output.stderr.len().saturating_sub(4096)..])
    );
    Ok(())
}

// ── Parent-side cases: no process-global state is touched. ──

/// Builder-failure propagation at `LocalServiceBridge::spawn_with`: a build
/// failure resolves the readiness receiver with the preserved error, so the
/// production ordering (await the receiver before serve_bridged) observes a
/// fatal init instead of readiness.
///
/// (OAuth's own production builder — `OAuthRpcHandler::new` — is infallible,
/// so a run()-level bridge-build failure has no injectable seam; this proves
/// error propagation at the exact API the transaction gates on.)
#[test]
fn bridge_spawn_with_propagates_builder_failure() -> anyhow::Result<()> {
    let (bridge, mut ready) = hyprstream_rpc::transport::iroh_rpc::LocalServiceBridge::spawn_with(
        "readiness-fail-echo",
        || async {
            Err::<rpc_handler::OAuthRpcHandler, _>(anyhow::anyhow!(
                "injected bridge build failure"
            ))
        },
        Arc::new(hyprstream_rpc::envelope::InMemoryNonceCache::new()),
        0,
    )?;
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        // The production ordering awaits THIS receiver before serve_bridged;
        // a build failure must resolve here as Err — before any READY.
        match tokio::time::timeout(Duration::from_secs(30), &mut ready).await {
            Ok(Ok(Err(build_err))) => {
                assert!(
                    build_err.to_string().contains("injected bridge build failure"),
                    "builder error must be preserved: {build_err}"
                );
            }
            Ok(Ok(Ok(()))) => panic!("failing builder must not report readiness"),
            Ok(Err(_)) => panic!("readiness receiver closed without a result"),
            Err(_) => panic!("readiness receiver never resolved"),
        }
    });
    let _ = bridge.begin_shutdown(tokio::time::Instant::now() + Duration::from_secs(5));
    drop(bridge);
    Ok(())
}

/// The profile-aware bind helper maps an injected bind failure at the real
/// production boundary: fatal for Required, warn-and-continue for
/// Compatibility. (This is the narrow substrate-builder seam — the failure is
/// injected into the builder run() itself calls, never simulated via the
/// install OnceLock.)
#[test]
fn substrate_bind_failure_profile_mapping() -> anyhow::Result<()> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        let required = bind_oauth_substrate_profile(true, || async {
            Err(anyhow::anyhow!("injected substrate bind failure"))
        })
        .await;
        let err = match required {
            Ok(_) => panic!("Required bind failure must be fatal"),
            Err(e) => e,
        };
        assert!(
            err.to_string()
                .contains("network-iroh-required OAuth iroh substrate bind failed"),
            "fatal Required mapping lost: {err}"
        );
        let compat = bind_oauth_substrate_profile(false, || async {
            Err(anyhow::anyhow!("injected substrate bind failure"))
        })
        .await
        .expect("Compatibility bind failure warns and continues");
        assert!(compat.is_none(), "degraded Compatibility has no substrate owner");
    });
    Ok(())
}

// ── Child-isolated cases: real run() startup seams. ──

/// Occupied plain-HTTP listener: run() fails at the real bind, before READY,
/// leaving no control socket (the serve phase never started) and a released
/// listener (RAII prebound handle closed on the failure path).
#[test]
fn run_occupied_http_listener_fails_before_ready() -> anyhow::Result<()> {
    const CASE: &str = "services::oauth::readiness_tests::run_occupied_http_listener_fails_before_ready";
    run_isolated(CASE, "HYPRSTREAM_OAUTH_READINESS_OCCUPIED_HTTP")?;
    if std::env::var_os("HYPRSTREAM_OAUTH_READINESS_OCCUPIED_HTTP").is_none() {
        return Ok(());
    }
    let env = ChildEnv::new("occupied-http");
    let oauth = SigningKey::from_bytes(&[0xA1; 32]);
    seed_trust_store();
    let addr = reserve_ephemeral_port();
    let _holder = occupy_port(addr);
    let control_path = env.sock_dir.path().join("control.sock");
    let service = http_service_at(&oauth, addr, env.ipc_control());
    let (ready_tx, mut ready_rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let handle = std::thread::spawn(move || Box::new(service).run(shutdown, Some(ready_tx)));
    let err = handle
        .join()
        .expect("run thread must not panic")
        .expect_err("occupied listener must fail startup");
    assert!(
        err.to_string().contains("bind failed"),
        "failure must come from the real listener bind: {err}"
    );
    assert!(
        ready_rx.try_recv().is_err(),
        "READY must never be signalled for a failed bind"
    );
    assert!(
        !control_path.exists(),
        "control transport must never be bound after a failed HTTP bind"
    );
    // The prebound listener is RAII: with the collision holder gone, the
    // port must be rebindable — run()'s failed bind released its handle.
    drop(_holder);
    drop(std::net::TcpListener::bind(addr).expect("port must be released after failed startup"));
    Ok(())
}

/// Occupied HTTPS listener: the same real collision through the prebound-TLS
/// branch (real RustlsConfig from a generated self-signed cert), no READY,
/// no second bind attempt.
#[test]
fn run_occupied_https_listener_fails_before_ready() -> anyhow::Result<()> {
    const CASE: &str = "services::oauth::readiness_tests::run_occupied_https_listener_fails_before_ready";
    run_isolated(CASE, "HYPRSTREAM_OAUTH_READINESS_OCCUPIED_HTTPS")?;
    if std::env::var_os("HYPRSTREAM_OAUTH_READINESS_OCCUPIED_HTTPS").is_none() {
        return Ok(());
    }
    let env = ChildEnv::new("occupied-https");
    let oauth = SigningKey::from_bytes(&[0xA2; 32]);
    seed_trust_store();
    let generated = rcgen::generate_simple_self_signed(vec!["localhost".to_owned()])?;
    let cert_path = env.dir.path().join("cert.pem");
    let key_path = env.dir.path().join("key.pem");
    std::fs::write(&cert_path, generated.cert.pem())?;
    std::fs::write(&key_path, generated.key_pair.serialize_pem())?;
    let mut tls = crate::config::TlsConfig::default();
    tls.enabled = true;
    let addr = reserve_ephemeral_port();
    let _holder = occupy_port(addr);
    let control_path = env.sock_dir.path().join("control.sock");
    let mut oauth_config = crate::config::OAuthConfig::default();
    oauth_config.host = addr.ip().to_string();
    oauth_config.port = addr.port();
    oauth_config.tls_cert = Some(cert_path);
    oauth_config.tls_key = Some(key_path);
    let peer = TransportConfig::ipc(env.dir.path().join("peer.sock"));
    let service = OAuthService::new(
        oauth_config,
        tls,
        crate::account::AccountZoneConfig::default(),
        oauth.clone(),
        env.ipc_control(),
        peer.clone(),
        peer,
        oauth.verifying_key(),
        oauth.verifying_key(),
    );
    let (ready_tx, mut ready_rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let handle = std::thread::spawn(move || Box::new(service).run(shutdown, Some(ready_tx)));
    let err = handle
        .join()
        .expect("run thread must not panic")
        .expect_err("occupied HTTPS listener must fail startup");
    assert!(
        err.to_string().contains("bind failed"),
        "failure must come from the real TLS listener bind: {err}"
    );
    assert!(
        ready_rx.try_recv().is_err(),
        "READY must never be signalled for a failed TLS bind"
    );
    assert!(!control_path.exists());
    Ok(())
}

/// Positive transaction: READY fires only after the listener is bound (a real
/// HTTP request then succeeds through the actual handler) and after the
/// control transport is bound (the IPC socket exists at READY time); the
/// shutdown signal produces a bounded, clean completion.
#[test]
fn run_ready_then_real_http_and_bounded_shutdown() -> anyhow::Result<()> {
    const CASE: &str =
        "services::oauth::readiness_tests::run_ready_then_real_http_and_bounded_shutdown";
    run_isolated(CASE, "HYPRSTREAM_OAUTH_READINESS_POSITIVE")?;
    if std::env::var_os("HYPRSTREAM_OAUTH_READINESS_POSITIVE").is_none() {
        return Ok(());
    }
    let env = ChildEnv::new("positive");
    let oauth = SigningKey::from_bytes(&[0xA3; 32]);
    seed_trust_store();
    let addr = reserve_ephemeral_port();
    let control_path = env.sock_dir.path().join("control.sock");
    let service = http_service_at(&oauth, addr, env.ipc_control());
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let shutdown_signal = Arc::clone(&shutdown);
    let handle = std::thread::spawn(move || Box::new(service).run(shutdown, Some(ready_tx)));

    // Wait for the EXTERNAL readiness signal, then prove the exact claim: at
    // signal time the listener was bound and the control transport registered.
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let ready_outcome = rt.block_on(async {
        tokio::time::timeout(Duration::from_secs(60), ready_rx).await
    });
    let ready_fired = match ready_outcome {
        Ok(Ok(())) => true,
        // CLOSED READY: the run thread ended without ever signalling. Join it
        // and surface the ORIGINAL run error before unwinding ChildEnv — the
        // later native-teardown noise must not mask the primary failure.
        Ok(Err(_)) | Err(_) => {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let _ = tx.send(handle.join());
            });
            let joined = rx
                .recv_timeout(Duration::from_secs(90))
                .expect("run() must terminate within the shutdown budget");
            let run_result = joined.expect("run thread must not panic");
            panic!(
                "on_ready never fired; the actual run() result was: {run_result:?}"
            );
        }
    };
    assert!(ready_fired, "on_ready must fire on the success path");
    assert!(control_path.exists(), "control UDS must be bound before READY");

    // Real HTTP request — proves the handler loop executes, not merely that
    // the backlog accepts.
    let mut stream = std::net::TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    write!(
        stream,
        "GET /.well-known/oauth-authorization-server HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "metadata endpoint must really serve: got {response:?}"
    );
    assert!(
        response.contains("issuer"),
        "authorization-server metadata body must be present"
    );

    // Bounded shutdown: notifying the serve path ends the whole transaction —
    // this join completing at all is the bounded-teardown claim.
    shutdown_signal.notify_one();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(handle.join());
    });
    let joined = rx
        .recv_timeout(Duration::from_secs(90))
        .expect("run() must terminate within the shutdown budget");
    let run_result = joined.expect("run thread must not panic");
    run_result.expect("clean shutdown must complete the transaction with Ok");
    Ok(())
}

/// Compatibility with Iroh DISABLED: run() reaches READY over real HTTP and
/// never touches the global endpoint — proven by classifying a freshly bound
/// substrate as `InstalledHere` afterwards (the slot would read occupied if
/// run() had bound or installed anything).
#[test]
fn run_compatibility_disabled_iroh_skips_substrate() -> anyhow::Result<()> {
    const CASE: &str =
        "services::oauth::readiness_tests::run_compatibility_disabled_iroh_skips_substrate";
    run_isolated(CASE, "HYPRSTREAM_OAUTH_READINESS_COMPAT_DISABLED")?;
    if std::env::var_os("HYPRSTREAM_OAUTH_READINESS_COMPAT_DISABLED").is_none() {
        return Ok(());
    }
    let env = ChildEnv::new("compat-disabled");
    let oauth = SigningKey::from_bytes(&[0xA9; 32]);
    seed_trust_store();
    let addr = reserve_ephemeral_port();
    let service = http_service_at(&oauth, addr, env.ipc_control());
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let shutdown_signal = Arc::clone(&shutdown);
    let handle = std::thread::spawn(move || Box::new(service).run(shutdown, Some(ready_tx)));
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        tokio::time::timeout(Duration::from_secs(90), ready_rx)
            .await
            .expect("READY must arrive within the test budget")
            .expect("on_ready must fire with Iroh disabled");
    });
    // The global slot is EMPTY: run() neither bound nor installed anything.
    let (probe_tx, probe_rx) = tokio::sync::oneshot::channel();
    rt.spawn(async move {
        let probe = build_oauth_substrate_profile(&oauth, false)
            .await?
            .expect("fresh substrate binds");
        let disposition = classify_oauth_endpoint_install(&probe);
        probe.shutdown().await?;
        probe_tx.send(disposition).ok();
        anyhow::Ok(())
    });
    let disposition = rt
        .block_on(async { tokio::time::timeout(Duration::from_secs(90), probe_rx).await })
        .expect("probe must finish within the budget")
        .expect("probe task must not fail");
    assert!(
        matches!(disposition, OAuthEndpointInstall::InstalledHere),
        "disabled-Compatibility run() must not have installed a global endpoint"
    );
    // Real HTTP still serving, then bounded shutdown.
    let mut stream = std::net::TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    write!(
        stream,
        "GET /.well-known/oauth-authorization-server HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "metadata endpoint must really serve: got {response:?}"
    );
    shutdown_signal.notify_one();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(handle.join());
    });
    let joined = rx
        .recv_timeout(Duration::from_secs(90))
        .expect("run() must terminate within the shutdown budget");
    joined
        .expect("run thread must not panic")
        .expect("clean shutdown must complete the transaction with Ok");
    Ok(())
}

/// Required profile with Iroh disabled is fatal before READY — defense in
/// depth behind `validate_native_network_profile`.
#[test]
fn run_required_profile_disabled_iroh_fails_before_ready() -> anyhow::Result<()> {
    const CASE: &str = "services::oauth::readiness_tests::run_required_profile_disabled_iroh_fails_before_ready";
    run_isolated(CASE, "HYPRSTREAM_OAUTH_READINESS_REQUIRED_DISABLED")?;
    if std::env::var_os("HYPRSTREAM_OAUTH_READINESS_REQUIRED_DISABLED").is_none() {
        return Ok(());
    }
    let env = ChildEnv::new("required-disabled");
    let oauth = SigningKey::from_bytes(&[0xA4; 32]);
    seed_trust_store();
let addr = reserve_ephemeral_port();
    let control_path = env.sock_dir.path().join("control.sock");
    let mut quic = crate::config::QuicConfig::default();
    quic.native_network_profile = crate::config::NativeNetworkProfile::NetworkIrohRequired;
    quic.enabled = false;
    quic.iroh = true;
    let service = http_service_at(&oauth, addr, env.ipc_control())
        .with_quic_config(quic);
    let (ready_tx, mut ready_rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let handle = std::thread::spawn(move || Box::new(service).run(shutdown, Some(ready_tx)));
    let err = handle
        .join()
        .expect("run thread must not panic")
        .expect_err("Required with disabled Iroh must fail startup");
    assert!(
        err.to_string()
            .contains("network-iroh-required OAuth service requires [quic] enabled"),
        "failure must come from the Required carrier guard: {err}"
    );
    assert!(ready_rx.try_recv().is_err(), "no READY on fatal startup");
    assert!(!control_path.exists());
    Ok(())
}

/// The real spawner chain the foreground launcher uses maps a pre-READY
/// startup failure to the spawn error (`service thread exited before ready`)
/// that `bin/main.rs` propagates via `?` to a nonzero child exit.
#[test]
fn manager_spawn_pre_ready_failure_maps_to_spawn_error() -> anyhow::Result<()> {
    const CASE: &str =
        "services::oauth::readiness_tests::manager_spawn_pre_ready_failure_maps_to_spawn_error";
    run_isolated(CASE, "HYPRSTREAM_OAUTH_READINESS_MANAGER_CHAIN")?;
    if std::env::var_os("HYPRSTREAM_OAUTH_READINESS_MANAGER_CHAIN").is_none() {
        return Ok(());
    }
    let env = ChildEnv::new("manager-chain");
    let oauth = SigningKey::from_bytes(&[0xA5; 32]);
    seed_trust_store();
    let addr = reserve_ephemeral_port();
    let _holder = occupy_port(addr);
    // Completion wrapper: done_rx fires only AFTER the inner OAuthService
    // run() returned and dropped its state on the manager's service thread.
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
    let service = RunCompletionService {
        inner: http_service_at(&oauth, addr, env.ipc_control()),
        done_tx: Some(done_tx),
    };
    let manager = hyprstream_service::InprocManager::new();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let spawned = rt.block_on(hyprstream_service::ServiceManager::spawn(
        &manager,
        Box::new(service),
    ));
    let err = match spawned {
        Ok(_) => panic!("pre-READY failure must reject manager.spawn"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("service thread exited before ready"),
        "failure must traverse the real ready-channel chain: {err}"
    );
    // Bounded wait for the REAL teardown: the manager rejects the spawn, but
    // its service thread still runs the inner run() to completion; done_rx
    // resolves only once that happened and the state dropped.
    let rt_wait = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt_wait
        .block_on(async {
            tokio::time::timeout(Duration::from_secs(90), done_rx).await
        })
        .expect("service teardown must finish within the budget")
        .expect("completion sender must outlive the inner run");
    Ok(())
}

/// Compatibility child with Iroh ENABLED and an EMPTY install-once slot:
/// OAuth's substrate wins the process-global outbound slot (`InstalledHere`);
/// a second, differently-derived substrate then observes the slot as occupied
/// (`ExistingGlobalRetained`) — the observable side effect of the first
/// install, without reaching into the rpc crate's private accessor.
#[test]
fn oauth_endpoint_install_wins_empty_slot() -> anyhow::Result<()> {
    const CASE: &str = "services::oauth::readiness_tests::oauth_endpoint_install_wins_empty_slot";
    run_isolated(CASE, "HYPRSTREAM_OAUTH_READINESS_WINS_EMPTY_SLOT")?;
    if std::env::var_os("HYPRSTREAM_OAUTH_READINESS_WINS_EMPTY_SLOT").is_none() {
        return Ok(());
    }
    let _env = ChildEnv::new("wins-empty-slot");
    let oauth = SigningKey::from_bytes(&[0xA6; 32]);
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        let winner = build_oauth_substrate_profile(&oauth, false)
            .await?
            .expect("successful Compatibility bind retains the substrate");
        assert!(
            matches!(
                classify_oauth_endpoint_install(&winner),
                OAuthEndpointInstall::InstalledHere
            ),
            "an empty slot must classify InstalledHere"
        );
        // A differently-derived substrate now finds the slot occupied: the
        // first install really did fill the process-global outbound slot.
        let other_key = SigningKey::from_bytes(&[0xA7; 32]);
        let loser = build_oauth_substrate_profile(&other_key, false)
            .await?
            .expect("second successful Compatibility bind retains its substrate");
        assert!(
            matches!(
                classify_oauth_endpoint_install(&loser),
                OAuthEndpointInstall::ExistingGlobalRetained
            ),
            "an occupied slot must classify ExistingGlobalRetained, never error"
        );
        assert_ne!(
            winner.endpoint_id(),
            loser.endpoint_id(),
            "distinct transport purpose keys yield deliberately distinct endpoints"
        );
        loser.shutdown().await?;
        winner.shutdown().await?;
        anyhow::Ok(())
    })?;
    Ok(())
}

/// Required positive control through the REAL bootstrap-first startup: the
/// authenticated OS-owned bootstrap runs first (installing the process-global
/// outbound carrier, Sol carrier case 1's precondition), then
/// `OAuthService::run` — under `network-iroh-required` with Iroh enabled —
/// reaches READY, serves a real HTTP request, and shuts down within bounds.
/// The carrier classification and endpoint-ID distinctness assertions remain
/// at the helper seam ([`super::required_consumer_tests`], the exact seam
/// this run calls internally).
#[test]
fn run_required_bootstrap_first_ready_then_http() -> anyhow::Result<()> {
    const CASE: &str =
        "services::oauth::readiness_tests::run_required_bootstrap_first_ready_then_http";
    run_isolated(CASE, "HYPRSTREAM_OAUTH_READINESS_REQUIRED_POSITIVE")?;
    if std::env::var_os("HYPRSTREAM_OAUTH_READINESS_REQUIRED_POSITIVE").is_none() {
        return Ok(());
    }
    let env = ChildEnv::new("required-positive");
    let oauth = SigningKey::from_bytes(&[0xA8; 32]);

    // ── Real OS-owned Required bootstrap fixtures (the did_trust recipe from
    // required_consumer_tests): deployment CA + authority log/checkpoint,
    // registry credential, per-service signing keys, trust store, and
    // provisioning for policy/discovery/oauth. ──
    let directory = env.dir.path().to_path_buf();
    let trust = directory.join("trust");
    let credentials = directory.join("credentials");
    let secrets = directory.join("secrets");
    std::fs::create_dir_all(&trust)?;
    std::fs::create_dir_all(&secrets)?;
    // The secrets resolver is env-first: point it at the provisioning dir
    // (where the per-service signing keys live) for the rest of the child.
    std::env::set_var("HYPRSTREAM__SECRETS__PATH", &secrets);
    std::env::set_var("HYPRSTREAM_DEPLOYMENT_TRUST_DIR", &trust);
    let ca = SigningKey::from_bytes(&[0x71; 32]);
    let registry = SigningKey::from_bytes(&[0x72; 32]);
    let policy = SigningKey::from_bytes(&[0x73; 32]);
    let discovery = SigningKey::from_bytes(&[0x74; 32]);
    let mut root = ca.verifying_key().to_bytes().to_vec();
    root.extend(hyprstream_rpc::crypto::pq::ml_dsa_sk_to_vk_bytes(
        &hyprstream_rpc::crypto::pq::ml_dsa_sk_from_seed(&ca.to_bytes()),
    ));
    std::fs::write(trust.join("deployment-ca.hybrid"), root)?;
    let (log, checkpoint) = super::required_consumer_tests::did_trust::authority_log_and_checkpoint(&ca);
    std::fs::write(
        trust.join("deployment-authority.log.json"),
        serde_json::to_vec(&log)?,
    )?;
    std::fs::write(
        trust.join("deployment-authority.head.json"),
        serde_json::to_vec(&checkpoint)?,
    )?;
    std::fs::write(credentials.join("registry-service.jwt"), super::required_consumer_tests::did_trust::mint_credential(&ca, &registry))?;
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
        Some(directory.join("rpc")),
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
        &["policy".to_owned(), "discovery".to_owned(), "oauth".to_owned()],
        3600,
        None,
    )?;

    // Bootstrap FIRST: the process-global outbound carrier is installed and
    // the process flips to the Required profile before the service starts.
    // The carrier's lazy transports live on THIS runtime; keep it driven for
    // the whole test (multi-thread stays polled after block_on returns) so
    // the retained endpoint is an operating carrier, not an unparked one.
    let bootstrap_rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    bootstrap_rt.block_on(hyprstream_discovery::bootstrap_deployment_process(
        oauth.clone(),
        hyprstream_discovery::DeploymentTrustSource::OsOwnedFiles,
        false,
        true,
    ))?;
    assert!(
        hyprstream_discovery::native_network_required(),
        "the Os-owned bootstrap must have activated the Required profile"
    );

    // Now the production run() seam under network-iroh-required: it binds its
    // own inbound substrate, classifies the occupied global slot (bootstrap
    // first-write-wins), and proceeds through bridge → inproc control
    // registration → READY. Production Required uses the local in-process
    // control (no service-RPC UDS); this test mirrors that configuration.
    let addr = reserve_ephemeral_port();
    let mut quic = crate::config::QuicConfig::default();
    quic.native_network_profile = crate::config::NativeNetworkProfile::NetworkIrohRequired;
    quic.enabled = true;
    quic.iroh = true;
    let service = http_service_at(
        &oauth,
        addr,
        TransportConfig::inproc("oauth-user-crud-readiness"),
    )
    .with_quic_config(quic);
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<()>();
    let shutdown = Arc::new(tokio::sync::Notify::new());
    let shutdown_signal = Arc::clone(&shutdown);
    let handle = std::thread::spawn(move || Box::new(service).run(shutdown, Some(ready_tx)));

    let rt_wait = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt_wait.block_on(async {
        tokio::time::timeout(Duration::from_secs(90), ready_rx)
            .await
            .expect("READY must arrive within the test budget")
            .expect("on_ready must fire on the Required success path");
    });
    // Real HTTP request through the actually serving handler. (This proves
    // READY ordering and HTTP serving only — it makes no claim about
    // protected-API authorization, and real outbound Policy/Discovery RPC
    // coverage lives in required_consumer_tests.)
    let mut stream = std::net::TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    write!(
        stream,
        "GET /.well-known/oauth-authorization-server HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n"
    )?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "metadata endpoint must really serve: got {response:?}"
    );

    // Bounded shutdown.
    shutdown_signal.notify_one();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(handle.join());
    });
    let joined = rx
        .recv_timeout(Duration::from_secs(90))
        .expect("run() must terminate within the shutdown budget");
    joined
        .expect("run thread must not panic")
        .expect("clean shutdown must complete the transaction with Ok");
    drop(bootstrap_rt);
    Ok(())
}
