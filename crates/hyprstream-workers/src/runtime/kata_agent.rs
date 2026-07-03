//! kata-agent ttrpc/vsock client — T1-C (#344) of the Worker GA epic (#341)
//!
//! Implements a real client for the `kata-agent` running *inside* a Kata
//! guest VM, so [`KataBackend::exec_sync`](super::kata_backend::KataBackend)
//! can create/start/exec a container inside the VM instead of returning the
//! historical "not supported" error.
//!
//! # Wire protocol: REAL, not hand-rolled
//!
//! The ttrpc request/response message types ([`protocols::agent`],
//! [`protocols::oci`]) and the generated async ttrpc client
//! ([`protocols::agent_ttrpc_async::AgentServiceClient`]) come from the
//! `protocols` crate, which is vendored verbatim from the upstream
//! `kata-containers` Rust workspace (`src/libs/protocols`, tag matching our
//! `kata-hypervisor` dependency). It is *already* present in our dependency
//! graph transitively (`kata-hypervisor` depends on `protocols` with the
//! `async` ttrpc feature enabled) — this module simply adds it as a direct
//! dependency and uses it, rather than reimplementing the kata-agent
//! protobuf/ttrpc wire format by hand. The `CreateContainer` / `StartContainer`
//! / `ExecProcess` / `WaitProcess` RPCs used below are the real generated
//! client methods (`agent_ttrpc_async.rs`), framed and (de)serialized by the
//! real `ttrpc` crate (`ttrpc::r#async::Client`).
//!
//! What IS hand-written here: the vsock/hybrid-vsock *transport* dialer
//! (connecting to the guest's kata-agent listen socket and handing the raw
//! fd to `ttrpc::r#async::Client::new`), and the mapping from our
//! `SandboxBackend::exec_sync` shape to the agent RPC sequence. This mirrors
//! (does not vendor) the dialer in upstream
//! `runtime-rs/crates/agent/src/sock/{vsock,hybrid_vsock}.rs`.
//!
//! # Connection convention (verified against upstream kata-containers source,
//! tag `3.31.0`, fetched into the local cargo git checkout cache — see
//! `~/.cargo/git/checkouts/kata-containers-*/*/src/runtime-rs/crates/agent/src/sock/`)
//!
//! - **Cloud Hypervisor** (the only hypervisor this crate wires up — see
//!   `kata_backend.rs`) exposes the kata-agent over *hybrid vsock*: a Unix
//!   domain socket at `<sandbox_dir>/ch-vm.sock`
//!   (`hypervisor::ch::utils::get_vsock_path`), where the host writes
//!   `"connect <port>\n"` and expects a response line containing `"OK"`
//!   before the same fd becomes the ttrpc transport
//!   (`runtime-rs/crates/agent/src/sock/hybrid_vsock.rs::connect_helper`).
//!   `kata_hypervisor::Hypervisor::get_agent_socket()` returns this address
//!   as `hvsock://<path>`, matching the `Hypervisor` trait already used by
//!   `KataBackend`.
//! - **Firecracker/Dragonball** use real `AF_VSOCK` (`vsock://<cid>:<port>`),
//!   dialed via `socket(AF_VSOCK) + connect(VsockAddr{cid, port})`
//!   (`runtime-rs/crates/agent/src/sock/vsock.rs`). We don't boot these
//!   hypervisors yet, but the dialer below supports the scheme for parity.
//! - The kata-agent's *default* ttrpc listen port is **1024**
//!   (`src/runtime/virtcontainers/hypervisor.go: vSockPort = 1024`), used
//!   whenever a caller doesn't override it.
//!
//! # What's REAL vs what's UNVALIDATED in this pass
//!
//! REAL: the message types, the ttrpc client/wire framing, the hybrid-vsock
//! handshake implementation, and the RPC sequencing
//! (`CreateContainer` → `StartContainer` → `ExecProcess` → `WaitProcess` (→
//! `ReadStdout`/`ReadStderr`)). Unit tests in this module run a *real*
//! `ttrpc::r#async::Server` (the same crate a real kata-agent links against)
//! bound to a Unix socket, with a minimal `AgentService` impl that returns
//! canned responses — so the request encoding / response decoding is
//! exercised through the genuine ttrpc+protobuf wire path, not mocked at the
//! byte level.
//!
//! UNVALIDATED (this sandbox has no bootable kata-agent guest image): that a
//! *real* kata-agent's `CreateContainer`/`ExecProcess` semantics (OCI bundle
//! layout expectations, namespace/cgroup setup it performs) are satisfied by
//! the minimal `oci::Spec` this module constructs. The `CreateContainerRequest.OCI`
//! field needs a real OCI runtime spec (rootfs, mounts, namespaces) wired up
//! from `PodSandbox`/container config — this module supplies a minimal
//! placeholder `Spec` sufficient to exercise the RPC, not a complete OCI bundle
//! translation (that's a larger follow-up, see `build_minimal_spec`).
//!
//! # Exit-value structuring (issue #608 / future #610)
//!
//! `exec_sync`'s return shape `(exit_code, stdout, stderr)` is built from
//! `WaitProcessResponse.status` (a real terminal exit code from the guest
//! agent) plus drained `ReadStdout`/`ReadStderr` output — i.e. the exit code
//! is already a true terminal value by construction, not synthesized. A
//! future `/exec/instances/<id>/exit` projection (#610) can latch directly
//! onto the `(i32, Vec<u8>, Vec<u8>)` returned by [`KataAgentClient::exec`]
//! without restructuring this client.

use std::os::unix::io::IntoRawFd;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use protocols::agent_ttrpc_async::AgentServiceClient;
use protocols::{agent, empty, oci};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;

/// Default kata-agent ttrpc vsock port.
///
/// Source: upstream kata-containers `src/runtime/virtcontainers/hypervisor.go`
/// (`vSockPort = 1024`), tag `3.31.0` — the same tag pinned for our
/// `kata-hypervisor`/`protocols` dependencies.
pub const KATA_AGENT_VSOCK_PORT: u32 = 1024;

/// Default per-dial-attempt timeout.
const DIAL_TIMEOUT: Duration = Duration::from_millis(500);

/// Default overall connect timeout (covers retries).
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Default per-RPC ttrpc timeout (nanoseconds, per `ttrpc::context::with_timeout`).
const RPC_TIMEOUT_NS: i64 = 30 * 1_000_000_000;

/// A parsed kata-agent connection address.
///
/// Mirrors the two schemes upstream kata-containers' own
/// `runtime-rs/crates/agent/src/sock` dialer accepts. Cloud Hypervisor (the
/// only hypervisor `KataBackend` boots today) always returns
/// [`AgentAddress::HybridVsock`] from `Hypervisor::get_agent_socket()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentAddress {
    /// `hvsock://<unix-socket-path>` — Cloud Hypervisor / Dragonball hybrid
    /// vsock: a host-side Unix socket multiplexes guest vsock ports via a
    /// `CONNECT <port>\n` text handshake.
    HybridVsock { path: String },
    /// `vsock://<cid>` — real `AF_VSOCK`, used by Firecracker. Not exercised
    /// by `KataBackend` today (it only boots Cloud Hypervisor/Dragonball),
    /// but parsed/dialable for parity with upstream.
    Vsock { cid: u32 },
}

impl AgentAddress {
    /// Parse a `hvsock://<path>` or `vsock://<cid>` URI, as returned by
    /// `kata_hypervisor::Hypervisor::get_agent_socket()`.
    pub fn parse(address: &str) -> Result<Self> {
        if let Some(path) = address.strip_prefix("hvsock://") {
            return Ok(AgentAddress::HybridVsock {
                path: path.to_owned(),
            });
        }
        if let Some(cid) = address.strip_prefix("vsock://") {
            let cid: u32 = cid
                .parse()
                .with_context(|| format!("invalid vsock cid in {address:?}"))?;
            return Ok(AgentAddress::Vsock { cid });
        }
        Err(anyhow!("unsupported kata-agent socket scheme: {address:?}"))
    }
}

/// Dial the kata-agent's ttrpc socket and return a raw, connected fd.
///
/// For [`AgentAddress::HybridVsock`], performs the hybrid-vsock handshake
/// (`connect <port>\n` → response line containing `OK`) over the host Unix
/// socket, exactly as upstream's `hybrid_vsock::connect_helper` does. For
/// [`AgentAddress::Vsock`], opens a real `AF_VSOCK` connection via the
/// `vsock` crate.
pub async fn dial(address: &AgentAddress, port: u32) -> Result<std::os::unix::io::RawFd> {
    let deadline = tokio::time::Instant::now() + CONNECT_TIMEOUT;
    let mut last_err = None;

    while tokio::time::Instant::now() < deadline {
        let attempt = dial_once(address, port).await;
        match attempt {
            Ok(fd) => return Ok(fd),
            Err(e) => {
                last_err = Some(e);
                tokio::time::sleep(DIAL_TIMEOUT).await;
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow!("dial timed out with no attempts made")))
}

async fn dial_once(address: &AgentAddress, port: u32) -> Result<std::os::unix::io::RawFd> {
    match address {
        AgentAddress::HybridVsock { path } => hybrid_vsock_connect(path, port).await,
        AgentAddress::Vsock { cid } => vsock_connect(*cid, port),
    }
}

/// Hybrid-vsock handshake: connect to the host UDS, write `"connect
/// <port>\n"`, read one response line, require it to contain `"OK"`.
///
/// Matches `runtime-rs/crates/agent/src/sock/hybrid_vsock.rs::connect_helper`
/// in upstream kata-containers (tag `3.31.0`).
async fn hybrid_vsock_connect(uds_path: &str, port: u32) -> Result<std::os::unix::io::RawFd> {
    let mut stream = UnixStream::connect(uds_path)
        .await
        .with_context(|| format!("connect to hybrid-vsock UDS {uds_path:?}"))?;

    stream
        .write_all(format!("connect {port}\n").as_bytes())
        .await
        .context("write hybrid-vsock CONNECT handshake")?;

    let mut response = Vec::new();
    let mut byte = [0u8; 1];
    loop {
        let n = stream
            .read(&mut byte)
            .await
            .context("read hybrid-vsock handshake response")?;
        if n == 0 || byte[0] == b'\n' {
            break;
        }
        response.push(byte[0]);
    }
    let response = String::from_utf8_lossy(&response);
    if !response.contains("OK") {
        return Err(anyhow!(
            "hybrid-vsock handshake error: malformed response {response:?}"
        ));
    }

    let std_stream = stream.into_std().context("into_std")?;
    std_stream
        .set_nonblocking(false)
        .context("set blocking for ttrpc handoff")?;
    Ok(std_stream.into_raw_fd())
}

/// Real `AF_VSOCK` connect via the `vsock` crate (Firecracker/Dragonball
/// convention: `vsock://<cid>`, default port 1024).
fn vsock_connect(cid: u32, port: u32) -> Result<std::os::unix::io::RawFd> {
    use vsock::{VsockAddr, VsockStream};

    let addr = VsockAddr::new(cid, port);
    let stream = VsockStream::connect(&addr)
        .with_context(|| format!("AF_VSOCK connect to cid={cid} port={port}"))?;
    Ok(stream.into_raw_fd())
}

/// A connected kata-agent ttrpc client.
///
/// Thin wrapper around the real, generated
/// [`protocols::agent_ttrpc_async::AgentServiceClient`], scoped to the CRI
/// subset `SandboxBackend::exec_sync` needs: create → start → exec → wait
/// (+ drain stdout/stderr).
pub struct KataAgentClient {
    client: AgentServiceClient,
    timeout_ns: i64,
}

impl KataAgentClient {
    /// Dial the agent at `address` (kata-agent's well-known port 1024 unless
    /// the caller has a reason to override it) and wrap it in an
    /// `AgentServiceClient`.
    pub async fn connect(address: &AgentAddress) -> Result<Self> {
        Self::connect_with_port(address, KATA_AGENT_VSOCK_PORT).await
    }

    pub async fn connect_with_port(address: &AgentAddress, port: u32) -> Result<Self> {
        let fd = dial(address, port).await?;
        let ttrpc_client = ttrpc::r#async::Client::new(fd);
        Ok(Self::from_ttrpc_client(ttrpc_client))
    }

    /// Wrap an already-connected `ttrpc::r#async::Client` (used directly by
    /// tests against a local mock server, and available to callers who
    /// already have a connected client from elsewhere).
    pub fn from_ttrpc_client(client: ttrpc::r#async::Client) -> Self {
        Self {
            client: AgentServiceClient::new(client),
            timeout_ns: RPC_TIMEOUT_NS,
        }
    }

    fn ctx(&self) -> ttrpc::context::Context {
        ttrpc::context::with_timeout(self.timeout_ns)
    }

    /// `CreateContainer` — minimal OCI spec wired from `container_id` +
    /// `argv`. See module docs: this is NOT a full OCI bundle translation
    /// (no rootfs/mounts/namespaces from `PodSandbox` yet), just enough to
    /// exercise the real RPC end to end.
    pub async fn create_container(
        &self,
        container_id: &str,
        exec_id: &str,
        argv: &[String],
    ) -> Result<()> {
        let req = agent::CreateContainerRequest {
            container_id: container_id.to_owned(),
            exec_id: exec_id.to_owned(),
            OCI: protobuf::MessageField::some(build_minimal_spec(container_id, argv)),
            ..Default::default()
        };
        self.client
            .create_container(self.ctx(), &req)
            .await
            .context("CreateContainer ttrpc call")?;
        Ok(())
    }

    pub async fn start_container(&self, container_id: &str) -> Result<()> {
        let req = agent::StartContainerRequest {
            container_id: container_id.to_owned(),
            ..Default::default()
        };
        self.client
            .start_container(self.ctx(), &req)
            .await
            .context("StartContainer ttrpc call")?;
        Ok(())
    }

    /// Idempotently ensure `container_id` exists and is started inside the
    /// guest, tolerating "already exists"/"already started" ttrpc errors
    /// from a prior call (kata-agent has no native "create or no-op" RPC, so
    /// this papers over that at the client level — matching what a real CRI
    /// shim does when `exec_sync` is called against a container it didn't
    /// just create itself).
    ///
    /// `argv` is only used if the container needs to be created (it seeds
    /// the OCI `Process.Args` — see [`build_minimal_spec`]); an already
    /// existing container's original entrypoint is left untouched.
    async fn ensure_container(&self, container_id: &str, argv: &[String]) -> Result<()> {
        let create_exec_id = format!("{container_id}-init");
        match self.create_container(container_id, &create_exec_id, argv).await {
            Ok(()) => {}
            Err(e) if is_already_exists(&e) => {
                tracing::debug!(container_id, "container already exists, skipping create");
            }
            Err(e) => return Err(e.context("ensure_container: create_container")),
        }

        match self.start_container(container_id).await {
            Ok(()) => Ok(()),
            Err(e) if is_already_exists(&e) => {
                tracing::debug!(container_id, "container already started, skipping start");
                Ok(())
            }
            Err(e) => Err(e.context("ensure_container: start_container")),
        }
    }

    /// `ExecProcess` — run `argv` inside `container_id` under a fresh
    /// `exec_id`. Does not block for completion; pair with [`Self::wait_process`].
    pub async fn exec_process(&self, container_id: &str, exec_id: &str, argv: &[String]) -> Result<()> {
        let process = oci::Process {
            Args: argv.to_vec(),
            Cwd: "/".to_owned(),
            ..Default::default()
        };
        let req = agent::ExecProcessRequest {
            container_id: container_id.to_owned(),
            exec_id: exec_id.to_owned(),
            process: protobuf::MessageField::some(process),
            ..Default::default()
        };
        self.client
            .exec_process(self.ctx(), &req)
            .await
            .context("ExecProcess ttrpc call")?;
        Ok(())
    }

    /// `WaitProcess` — blocks (ttrpc-side) until the process exits, returning
    /// its real terminal exit status from the guest agent.
    pub async fn wait_process(&self, container_id: &str, exec_id: &str) -> Result<i32> {
        let req = agent::WaitProcessRequest {
            container_id: container_id.to_owned(),
            exec_id: exec_id.to_owned(),
            ..Default::default()
        };
        let resp = self
            .client
            .wait_process(self.ctx(), &req)
            .await
            .context("WaitProcess ttrpc call")?;
        Ok(resp.status)
    }

    /// `UpdateContainer` — apply new cgroup resource limits to a running
    /// container inside the guest.
    ///
    /// This is the RPC a CRI `UpdateContainerResources` maps onto: the
    /// kata-agent rewrites the container's in-guest cgroups (CPU shares/quota/
    /// period, memory limit, cpuset, …). It updates the container's slice of
    /// the *existing* VM; growing the VM itself past its boot size (vCPU /
    /// memory hotplug) is a separate hypervisor concern and not attempted
    /// here — matching upstream kata, whose `update_container` handler drives
    /// the guest cgroup layer, with VM resize handled independently by the
    /// runtime's resource manager.
    pub async fn update_container(
        &self,
        container_id: &str,
        resources: oci::LinuxResources,
    ) -> Result<()> {
        let req = agent::UpdateContainerRequest {
            container_id: container_id.to_owned(),
            resources: protobuf::MessageField::some(resources),
            ..Default::default()
        };
        self.client
            .update_container(self.ctx(), &req)
            .await
            .context("UpdateContainer ttrpc call")?;
        Ok(())
    }

    /// `StatsContainer` — fetch the guest cgroup (+ network) stats for a
    /// container. The returned [`agent::StatsContainerResponse`] carries the
    /// real in-guest cgroup counters (`cgroup_stats.cpu_stats` /
    /// `memory_stats`); mapping them into the CRI `ContainerStats` shape is
    /// the caller's job (see `KataBackend::container_stats`).
    pub async fn stats_container(
        &self,
        container_id: &str,
    ) -> Result<agent::StatsContainerResponse> {
        let req = agent::StatsContainerRequest {
            container_id: container_id.to_owned(),
            ..Default::default()
        };
        let resp = self
            .client
            .stats_container(self.ctx(), &req)
            .await
            .context("StatsContainer ttrpc call")?;
        Ok(resp)
    }

    /// Drain available stdout via repeated `ReadStdout` calls until the
    /// agent reports no more data (empty response).
    pub async fn read_stdout_all(&self, container_id: &str, exec_id: &str) -> Result<Vec<u8>> {
        self.read_stream_all(container_id, exec_id, StreamKind::Stdout)
            .await
    }

    pub async fn read_stderr_all(&self, container_id: &str, exec_id: &str) -> Result<Vec<u8>> {
        self.read_stream_all(container_id, exec_id, StreamKind::Stderr)
            .await
    }

    async fn read_stream_all(
        &self,
        container_id: &str,
        exec_id: &str,
        kind: StreamKind,
    ) -> Result<Vec<u8>> {
        const CHUNK: u32 = 4096;
        let mut out = Vec::new();
        loop {
            let req = agent::ReadStreamRequest {
                container_id: container_id.to_owned(),
                exec_id: exec_id.to_owned(),
                len: CHUNK,
                ..Default::default()
            };
            let resp = match kind {
                StreamKind::Stdout => self.client.read_stdout(self.ctx(), &req).await,
                StreamKind::Stderr => self.client.read_stderr(self.ctx(), &req).await,
            }
            .context("ReadStdout/ReadStderr ttrpc call")?;
            if resp.data.is_empty() {
                break;
            }
            let got = resp.data.len();
            out.extend_from_slice(&resp.data);
            if got < CHUNK as usize {
                break;
            }
        }
        Ok(out)
    }

    /// Full ensure-container → exec → wait → drain sequence, matching the
    /// `(exit_code, stdout, stderr)` shape `SandboxBackend::exec_sync` needs.
    ///
    /// Calls [`Self::ensure_container`] first (create + start, idempotent)
    /// so callers don't need to separately track whether `container_id` has
    /// already been created in this guest — matching the issue's "real
    /// CRI create/start/exec_sync" scope.
    ///
    /// The returned `i32` is `WaitProcessResponse.status` — a real terminal
    /// exit value from the guest agent (see module docs re: #608/#610).
    pub async fn exec(
        &self,
        container_id: &str,
        argv: &[String],
        timeout: Duration,
    ) -> Result<(i32, Vec<u8>, Vec<u8>)> {
        let exec_id = uuid::Uuid::new_v4().to_string();

        tokio::time::timeout(timeout, async {
            self.ensure_container(container_id, argv).await?;
            self.exec_process(container_id, &exec_id, argv).await?;
            let status = self.wait_process(container_id, &exec_id).await?;
            let stdout = self.read_stdout_all(container_id, &exec_id).await.unwrap_or_default();
            let stderr = self.read_stderr_all(container_id, &exec_id).await.unwrap_or_default();
            Ok::<_, anyhow::Error>((status, stdout, stderr))
        })
        .await
        .map_err(|_| anyhow!("exec timed out after {timeout:?}"))?
    }
}

#[derive(Clone, Copy)]
enum StreamKind {
    Stdout,
    Stderr,
}

/// Best-effort detection of "the container/process already
/// exists/started" ttrpc errors, used by [`KataAgentClient::ensure_container`]
/// to make create+start idempotent.
///
/// Prefers the structured gRPC-style status code (`ALREADY_EXISTS = 6`, per
/// `ttrpc.proto` — the same code space kata-agent's `CreateContainer`/
/// `StartContainer` handlers use to reject duplicate ids upstream), falling
/// back to a substring match on the error chain for cases where the error
/// was wrapped before reaching here. NOTE: the exact error this would
/// produce against a *real* kata-agent has not been observed in this sandbox
/// (no bootable guest available) — this is the best-effort mapping a real
/// instance is expected to need, not a verified one.
fn is_already_exists(err: &anyhow::Error) -> bool {
    for cause in err.chain() {
        if let Some(ttrpc::Error::RpcStatus(status)) = cause.downcast_ref::<ttrpc::Error>() {
            if status.code() == ttrpc::Code::ALREADY_EXISTS {
                return true;
            }
        }
    }
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("already exists") || msg.contains("already started")
}

/// Build a minimal `oci::Spec` sufficient to drive `CreateContainer` against
/// a kata-agent. NOT a complete OCI bundle translation — see module docs.
fn build_minimal_spec(container_id: &str, argv: &[String]) -> oci::Spec {
    oci::Spec {
        Version: "1.0.2".to_owned(),
        Process: protobuf::MessageField::some(oci::Process {
            Args: argv.to_vec(),
            Cwd: "/".to_owned(),
            ..Default::default()
        }),
        Root: protobuf::MessageField::some(oci::Root {
            Path: format!("/run/kata-containers/{container_id}/rootfs"),
            Readonly: false,
            ..Default::default()
        }),
        Hostname: container_id.to_owned(),
        ..Default::default()
    }
}

// Re-export so `kata_backend.rs` doesn't need a separate `protocols`/`empty`
// import just to satisfy trait bounds on `Empty` responses it may inspect.
#[allow(unused_imports)]
pub(crate) use empty::Empty as AgentEmpty;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use protocols::agent_ttrpc_async::{create_agent_service, AgentService};
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::Arc;
    use tempfile::TempDir;

    // ─────────────────────────────────────────────────────────────────────
    // AgentAddress parsing
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_parse_hybrid_vsock() {
        let addr = AgentAddress::parse("hvsock:///run/sandbox/ch-vm.sock").unwrap();
        assert_eq!(
            addr,
            AgentAddress::HybridVsock {
                path: "/run/sandbox/ch-vm.sock".to_owned()
            }
        );
    }

    #[test]
    fn test_parse_vsock() {
        let addr = AgentAddress::parse("vsock://42").unwrap();
        assert_eq!(addr, AgentAddress::Vsock { cid: 42 });
    }

    #[test]
    fn test_parse_unsupported_scheme() {
        let err = AgentAddress::parse("tcp://127.0.0.1:1024").unwrap_err();
        assert!(err.to_string().contains("unsupported"));
    }

    #[test]
    fn test_parse_invalid_vsock_cid() {
        let err = AgentAddress::parse("vsock://not-a-number").unwrap_err();
        assert!(err.to_string().contains("invalid vsock cid"));
    }

    #[test]
    fn test_kata_agent_vsock_port_matches_upstream_convention() {
        // src/runtime/virtcontainers/hypervisor.go: vSockPort = 1024
        assert_eq!(KATA_AGENT_VSOCK_PORT, 1024);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Hybrid-vsock handshake (real UDS, fake "guest" side)
    // ─────────────────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_hybrid_vsock_handshake_ok() {
        let dir = TempDir::new().unwrap();
        let sock_path = dir.path().join("ch-vm.sock");
        let listener = tokio::net::UnixListener::bind(&sock_path).unwrap();

        let accept_task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 64];
            let n = stream.read(&mut buf).await.unwrap();
            let req = String::from_utf8_lossy(&buf[..n]);
            assert_eq!(req, "connect 1024\n");
            stream.write_all(b"OK 1234\n").await.unwrap();
            // keep stream alive for the duration of the test
            tokio::time::sleep(Duration::from_millis(50)).await;
        });

        let fd = hybrid_vsock_connect(sock_path.to_str().unwrap(), 1024)
            .await
            .unwrap();
        assert!(fd >= 0);
        unsafe {
            libc_close(fd);
        }

        accept_task.await.unwrap();
    }

    #[tokio::test]
    async fn test_hybrid_vsock_handshake_rejects_malformed_response() {
        let dir = TempDir::new().unwrap();
        let sock_path = dir.path().join("ch-vm.sock");
        let listener = tokio::net::UnixListener::bind(&sock_path).unwrap();

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 64];
            let _ = stream.read(&mut buf).await.unwrap();
            stream.write_all(b"ERROR\n").await.unwrap();
        });

        let result = hybrid_vsock_connect(sock_path.to_str().unwrap(), 1024).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("handshake error"));
    }

    unsafe fn libc_close(fd: std::os::unix::io::RawFd) {
        // Avoid pulling in libc just for a test cleanup `close()`; nix
        // re-exports it and is already a workspace dependency.
        let _ = nix::unistd::close(fd);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Real ttrpc server/client round-trip (fake AgentService, real wire format)
    // ─────────────────────────────────────────────────────────────────────

    /// A minimal `AgentService` impl that proves request decoding +
    /// response encoding work end-to-end over the real ttrpc transport.
    /// Records the last request seen for assertions and returns canned,
    /// deterministic replies — this is the "mock/fake ttrpc server" the
    /// issue asks for, built on the *real* generated service trait rather
    /// than hand-rolled framing.
    struct FakeAgent {
        wait_status: AtomicI32,
        last_create: parking_lot::Mutex<Option<agent::CreateContainerRequest>>,
        last_exec: parking_lot::Mutex<Option<agent::ExecProcessRequest>>,
        /// Artificial delay before `wait_process` resolves, used to
        /// deterministically exercise `KataAgentClient::exec`'s timeout path
        /// (a near-zero `tokio::time::timeout` duration is not reliable: a
        /// fast local round-trip can race the timer's first poll).
        wait_process_delay: Duration,
        /// When set, `create_container`/`start_container` always reply with
        /// a `Code::ALREADY_EXISTS` ttrpc status, to exercise
        /// `KataAgentClient::ensure_container`'s idempotent-retry path.
        already_exists: bool,
    }

    impl FakeAgent {
        fn new(wait_status: i32) -> Self {
            Self {
                wait_status: AtomicI32::new(wait_status),
                last_create: parking_lot::Mutex::new(None),
                last_exec: parking_lot::Mutex::new(None),
                wait_process_delay: Duration::ZERO,
                already_exists: false,
            }
        }

        fn with_wait_process_delay(wait_status: i32, delay: Duration) -> Self {
            Self {
                wait_process_delay: delay,
                ..Self::new(wait_status)
            }
        }

        fn always_already_exists(wait_status: i32) -> Self {
            Self {
                already_exists: true,
                ..Self::new(wait_status)
            }
        }
    }

    fn already_exists_status() -> ::ttrpc::Error {
        ::ttrpc::Error::RpcStatus(::ttrpc::get_status(
            ::ttrpc::Code::ALREADY_EXISTS,
            "already exists".to_owned(),
        ))
    }

    #[async_trait]
    impl AgentService for FakeAgent {
        async fn create_container(
            &self,
            _ctx: &::ttrpc::r#async::TtrpcContext,
            req: agent::CreateContainerRequest,
        ) -> ::ttrpc::Result<empty::Empty> {
            if self.already_exists {
                return Err(already_exists_status());
            }
            *self.last_create.lock() = Some(req);
            Ok(empty::Empty::new())
        }

        async fn start_container(
            &self,
            _ctx: &::ttrpc::r#async::TtrpcContext,
            _req: agent::StartContainerRequest,
        ) -> ::ttrpc::Result<empty::Empty> {
            if self.already_exists {
                return Err(already_exists_status());
            }
            Ok(empty::Empty::new())
        }

        async fn exec_process(
            &self,
            _ctx: &::ttrpc::r#async::TtrpcContext,
            req: agent::ExecProcessRequest,
        ) -> ::ttrpc::Result<empty::Empty> {
            *self.last_exec.lock() = Some(req);
            Ok(empty::Empty::new())
        }

        async fn wait_process(
            &self,
            _ctx: &::ttrpc::r#async::TtrpcContext,
            _req: agent::WaitProcessRequest,
        ) -> ::ttrpc::Result<agent::WaitProcessResponse> {
            if !self.wait_process_delay.is_zero() {
                tokio::time::sleep(self.wait_process_delay).await;
            }
            Ok(agent::WaitProcessResponse {
                status: self.wait_status.load(Ordering::SeqCst),
                ..Default::default()
            })
        }

        async fn read_stdout(
            &self,
            _ctx: &::ttrpc::r#async::TtrpcContext,
            _req: agent::ReadStreamRequest,
        ) -> ::ttrpc::Result<agent::ReadStreamResponse> {
            Ok(agent::ReadStreamResponse {
                data: b"hello from stdout".to_vec(),
                ..Default::default()
            })
        }

        async fn read_stderr(
            &self,
            _ctx: &::ttrpc::r#async::TtrpcContext,
            _req: agent::ReadStreamRequest,
        ) -> ::ttrpc::Result<agent::ReadStreamResponse> {
            Ok(agent::ReadStreamResponse {
                data: Vec::new(),
                ..Default::default()
            })
        }
    }

    /// Spin up a real `ttrpc::r#async::Server` bound to a temp Unix socket,
    /// serving `FakeAgent`, and return a connected `KataAgentClient` plus
    /// the server's join handle (kept alive for the test's duration) and
    /// the `Arc<FakeAgent>` for post-call assertions.
    async fn spin_up_fake_agent(
        wait_status: i32,
    ) -> (KataAgentClient, Arc<FakeAgent>, tokio::task::JoinHandle<()>, TempDir) {
        spin_up_fake_agent_with(FakeAgent::new(wait_status)).await
    }

    async fn spin_up_fake_agent_with(
        fake: FakeAgent,
    ) -> (KataAgentClient, Arc<FakeAgent>, tokio::task::JoinHandle<()>, TempDir) {
        let dir = TempDir::new().unwrap();
        let sock_path = dir.path().join("agent.sock");
        let sockaddr = format!("unix://{}", sock_path.display());

        let fake = Arc::new(fake);
        let service = create_agent_service(fake.clone() as Arc<dyn AgentService + Send + Sync>);

        let mut server = ttrpc::r#async::Server::new()
            .bind(&sockaddr)
            .unwrap()
            .register_service(service);

        let server_task = tokio::spawn(async move {
            let _ = server.start().await;
            // Hold the server alive; in a real shutdown path we'd call
            // server.shutdown().await, omitted here since the test process
            // exit reclaims the socket.
            std::future::pending::<()>().await;
        });

        // Give the listener a moment to bind before dialing.
        tokio::time::sleep(Duration::from_millis(50)).await;

        let ttrpc_client = ttrpc::r#async::Client::connect(&sockaddr).unwrap();
        let client = KataAgentClient::from_ttrpc_client(ttrpc_client);

        (client, fake, server_task, dir)
    }

    #[tokio::test]
    async fn test_create_container_round_trip() {
        let (client, fake, _server, _dir) = spin_up_fake_agent(0).await;

        client
            .create_container("ctr-1", "exec-1", &["echo".into(), "hi".into()])
            .await
            .unwrap();

        let seen = fake.last_create.lock().clone().unwrap();
        assert_eq!(seen.container_id, "ctr-1");
        assert_eq!(seen.exec_id, "exec-1");
        assert!(seen.OCI.is_some());
        assert_eq!(seen.OCI.Process.Args, vec!["echo".to_owned(), "hi".to_owned()]);
    }

    #[tokio::test]
    async fn test_exec_process_round_trip() {
        let (client, fake, _server, _dir) = spin_up_fake_agent(0).await;

        client
            .exec_process("ctr-1", "exec-1", &["true".into()])
            .await
            .unwrap();

        let seen = fake.last_exec.lock().clone().unwrap();
        assert_eq!(seen.container_id, "ctr-1");
        assert_eq!(seen.process.Args, vec!["true".to_owned()]);
    }

    #[tokio::test]
    async fn test_wait_process_returns_real_status() {
        let (client, _fake, _server, _dir) = spin_up_fake_agent(7).await;

        let status = client.wait_process("ctr-1", "exec-1").await.unwrap();
        assert_eq!(status, 7);
    }

    #[tokio::test]
    async fn test_read_stdout_all_drains_until_empty() {
        let (client, _fake, _server, _dir) = spin_up_fake_agent(0).await;

        let out = client.read_stdout_all("ctr-1", "exec-1").await.unwrap();
        assert_eq!(out, b"hello from stdout");
    }

    #[tokio::test]
    async fn test_full_exec_sequence_parses_exit_stdout_stderr() {
        let (client, _fake, _server, _dir) = spin_up_fake_agent(42).await;

        let (status, stdout, stderr) = client
            .exec("ctr-1", &["echo".into(), "hi".into()], Duration::from_secs(5))
            .await
            .unwrap();

        assert_eq!(status, 42, "exit code must be the real WaitProcessResponse.status");
        assert_eq!(stdout, b"hello from stdout");
        assert_eq!(stderr, b"");
    }

    #[tokio::test]
    async fn test_exec_times_out() {
        // FakeAgent's wait_process deliberately stalls longer than the exec
        // timeout, so this deterministically exercises the timeout path
        // (rather than racing a near-zero `tokio::time::timeout` against a
        // fast local round-trip, which can flake).
        let fake = FakeAgent::with_wait_process_delay(0, Duration::from_secs(10));
        let (client, _fake, _server, _dir) = spin_up_fake_agent_with(fake).await;

        let result = client
            .exec("ctr-1", &["echo".into()], Duration::from_millis(50))
            .await;
        assert!(result.is_err(), "exec must time out when wait_process stalls");
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn test_exec_tolerates_already_existing_container() {
        // A container that was already created+started by a prior exec_sync
        // call must not make a later exec_sync fail — `ensure_container`
        // should swallow ALREADY_EXISTS from both CreateContainer and
        // StartContainer and proceed straight to ExecProcess.
        let fake = FakeAgent::always_already_exists(5);
        let (client, _fake, _server, _dir) = spin_up_fake_agent_with(fake).await;

        let (status, _stdout, _stderr) = client
            .exec("ctr-1", &["true".into()], Duration::from_secs(5))
            .await
            .expect("exec must succeed despite ALREADY_EXISTS on create/start");
        assert_eq!(status, 5);
    }

    #[test]
    fn test_is_already_exists_matches_structured_status_code() {
        let err = anyhow::Error::new(already_exists_status());
        assert!(is_already_exists(&err));
    }

    #[test]
    fn test_is_already_exists_matches_message_fallback() {
        let err = anyhow!("container ctr-1 already exists");
        assert!(is_already_exists(&err));
    }

    #[test]
    fn test_is_already_exists_false_for_unrelated_error() {
        let err = anyhow!("connection refused");
        assert!(!is_already_exists(&err));
    }
}
