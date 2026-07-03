//! WasmBackend — in-process WebAssembly sandbox isolation (#505 P2)
//!
//! A *native, in-process* sibling under the [`SandboxBackend`] seam. Unlike Kata
//! (full VM) or nspawn (systemd container subprocess), this backend runs the
//! workload as a WebAssembly guest inside *this* process via the embedded
//! wasmtime substrate ([`hyprstream_workers_wasmtime::Sandbox`]). There is no hypervisor and
//! no child process: isolation is the wasm sandbox itself — a bespoke capability
//! `Linker` exposing exactly `env::host_random` (Profile A: zero WASI,
//! capability-only) plus `define_unknown_imports_as_traps` for everything else,
//! bound to a per-sandbox [`Subject`], DoS-bounded by fuel + epoch interruption.
//!
//! ## Lifecycle mapping onto the substrate
//!
//! | `SandboxBackend` | wasm realisation |
//! |------------------|------------------|
//! | `start`          | resolve guest wasm bytes → `Sandbox::from_bytes_for(bytes, subject)`; stash live `Sandbox` + `EpochTimer` in a downcastable [`WasmHandle`] |
//! | `exec_sync`      | map command → a guest export invocation; return `(exit_code, stdout, stderr)` |
//! | `reset`          | drop + reinstantiate the guest (cheap vs a VM reboot) → returns `true` (reuse-in-place) |
//! | `get_pids`       | N/A in-process → empty vec |
//! | `stop`/`destroy` | drop the live `Sandbox`/`EpochTimer` off the handle |
//! | `update_resources` | re-tune the per-call fuel budget (epoch cadence is fixed at start) |
//!
//! ## Guest sourcing (P2 minimal path)
//!
//! The guest wasm module is sourced, in order of precedence, from
//! `PodSandboxConfig` **annotations**:
//!   * `hyprstream.io/wasm-module` — filesystem path to a `.wasm` (or `.wat`) file.
//!   * `hyprstream.io/wasm-module-base64` — base64-encoded module bytes inline.
//!
//! wasmtime's `Module::new` accepts both binary `.wasm` and textual `.wat`, so a
//! tiny hand-written guest works for tests without a build step.
//!
//! TODO(#505): full OCI-wasm-artifact pull. CRI plumbs the image reference on the
//! *container* config, not `PodSandboxConfig`; wiring an OCI artifact pull (a
//! wasm module is an OCI artifact with mediaType
//! `application/vnd.wasm.content.layer.v1+wasm`) into the image store and handing
//! the resolved bytes here is the next step. Until then, annotations are the
//! source of truth.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use parking_lot::Mutex;
use tracing::{debug, info, warn};

// The consolidated sandbox crate (renamed from `hyprstream-wasm`). It re-exports
// the canonical `hyprstream_rpc::Subject` as `hyprstream_workers_wasmtime::Subject`, so the
// guest runs as the SAME identity the rest of the daemon uses.
use hyprstream_workers_wasmtime::{EpochTimer, Sandbox, Subject};

use crate::config::PoolConfig;
use crate::error::{Result, WorkerError};

use super::backend::{SandboxBackend, SandboxHandle};
use super::client::{LinuxContainerResources, PodSandboxConfig};
use super::sandbox::PodSandbox;

/// Annotation key: filesystem path to the guest `.wasm`/`.wat` module.
const ANN_WASM_PATH: &str = "hyprstream.io/wasm-module";
/// Annotation key: base64-encoded guest module bytes (inline).
const ANN_WASM_BASE64: &str = "hyprstream.io/wasm-module-base64";

/// Default per-call fuel budget (instruction count). Generous; the wall-clock
/// epoch bound is the real DoS guard, fuel is a belt-and-suspenders ceiling.
const DEFAULT_FUEL: u64 = 1_000_000_000;

/// Epoch timer tick: how often the engine epoch advances. Paired with the
/// per-call epoch deadline, this is the wall-clock DoS bound.
const EPOCH_TICK: Duration = Duration::from_millis(50);

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the in-process wasm backend.
#[derive(Debug, Clone)]
pub struct WasmConfig {
    /// Per-call instruction (fuel) budget.
    pub fuel: u64,
    /// Epoch timer cadence (the wall-clock DoS tick).
    pub epoch_tick: Duration,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            fuel: DEFAULT_FUEL,
            epoch_tick: EPOCH_TICK,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Handle
// ─────────────────────────────────────────────────────────────────────────────

/// Live in-process state for a wasm sandbox, stashed on the `PodSandbox`.
///
/// Holds the loaded [`Sandbox`] (engine + compiled module + capability linker,
/// bound to a [`Subject`]) and the [`EpochTimer`] driving the wall-clock DoS
/// bound. Both are behind a `Mutex` so `reset()` can swap the live `Sandbox`
/// in place for warm-pool reuse without recreating the handle.
///
/// Dropping the handle drops the `EpochTimer` (joins its thread) and the
/// `Sandbox` (frees the wasmtime engine/module) — that *is* `destroy`.
pub struct WasmHandle {
    /// Sandbox identifier (matches `PodSandbox::id`).
    pub sandbox_id: String,
    /// The subject this sandbox runs as.
    pub subject: Subject,
    /// Per-call fuel budget.
    pub fuel: u64,
    /// The live substrate sandbox + its epoch timer. `None` once stopped.
    inner: Mutex<Option<WasmInner>>,
}

/// The droppable live wasm state.
struct WasmInner {
    sandbox: Arc<Sandbox>,
    /// Kept alive so the engine epoch keeps advancing (the wall-clock bound).
    /// Dropping it stops the timer thread.
    _epoch_timer: EpochTimer,
}

impl std::fmt::Debug for WasmHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let live = self.inner.lock().is_some();
        f.debug_struct("WasmHandle")
            .field("sandbox_id", &self.sandbox_id)
            .field("subject", &self.subject)
            .field("fuel", &self.fuel)
            .field("live", &live)
            .finish()
    }
}

impl SandboxHandle for WasmHandle {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl WasmHandle {
    /// Borrow the live sandbox (clone of the Arc), or error if stopped.
    fn sandbox(&self) -> Result<Arc<Sandbox>> {
        let guard = self.inner.lock();
        match guard.as_ref() {
            Some(inner) => Ok(Arc::clone(&inner.sandbox)),
            None => Err(WorkerError::SandboxInvalidState {
                sandbox_id: self.sandbox_id.clone(),
                state: "stopped (wasm instance dropped)".into(),
                expected: "running".into(),
            }),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend
// ─────────────────────────────────────────────────────────────────────────────

/// In-process WebAssembly sandbox backend.
#[derive(Debug)]
pub struct WasmBackend {
    config: WasmConfig,
}

impl WasmBackend {
    pub fn new(config: WasmConfig) -> Self {
        Self { config }
    }

    /// Runtime prerequisite probe for the backend registry. wasmtime is vendored
    /// in-process once the `wasm` feature is on, so the backend is *always*
    /// available when compiled in — there is no external runtime to detect.
    fn registry_is_available() -> bool {
        true
    }

    /// Resolve the guest wasm/wat bytes from sandbox annotations.
    ///
    /// Precedence: explicit path (`hyprstream.io/wasm-module`) → inline base64
    /// (`hyprstream.io/wasm-module-base64`). Fail-closed: if neither is present
    /// we error rather than silently running an empty/placeholder guest.
    fn resolve_guest_bytes(
        sandbox_id: &str,
        annotations: &HashMap<String, String>,
    ) -> Result<Vec<u8>> {
        if let Some(path) = annotations.get(ANN_WASM_PATH) {
            debug!(sandbox_id, path, "loading wasm guest from path");
            return std::fs::read(path).map_err(|e| {
                WorkerError::SandboxCreationFailed(format!(
                    "wasm backend: failed to read guest module '{path}': {e}"
                ))
            });
        }
        if let Some(b64) = annotations.get(ANN_WASM_BASE64) {
            debug!(sandbox_id, "loading wasm guest from inline base64");
            return base64::engine::general_purpose::STANDARD
                .decode(b64)
                .map_err(|e| {
                    WorkerError::SandboxCreationFailed(format!(
                        "wasm backend: invalid base64 guest module: {e}"
                    ))
                });
        }
        Err(WorkerError::SandboxCreationFailed(format!(
            "wasm backend: no guest module for sandbox '{sandbox_id}'. Provide one \
             via annotation '{ANN_WASM_PATH}' (path) or '{ANN_WASM_BASE64}' \
             (inline base64). OCI-wasm-artifact pull is TODO(#505)."
        )))
    }

    /// Instantiate a fresh substrate [`Sandbox`] + [`EpochTimer`] for `subject`.
    fn build_inner(&self, wasm: &[u8], subject: Subject) -> Result<WasmInner> {
        let sandbox = Sandbox::from_bytes_for(wasm, subject).map_err(|e| {
            WorkerError::SandboxCreationFailed(format!(
                "wasm backend: failed to load guest module: {e:#}"
            ))
        })?;
        // Spawn the wall-clock DoS timer on this sandbox's engine. Held in the
        // handle so it lives as long as the guest does.
        let epoch_timer = EpochTimer::spawn(sandbox.engine(), self.config.epoch_tick);
        Ok(WasmInner {
            sandbox: Arc::new(sandbox),
            _epoch_timer: epoch_timer,
        })
    }
}

#[async_trait]
impl SandboxBackend for WasmBackend {
    fn backend_type(&self) -> &'static str {
        "wasm"
    }

    fn is_available(&self) -> bool {
        // wasmtime is vendored in-process; nothing external to probe.
        true
    }

    async fn initialize(&self, _config: &PoolConfig) -> Result<()> {
        // No external runtime, paths or privileges to set up — the substrate is
        // entirely in-process. Nothing to do.
        Ok(())
    }

    async fn start(
        &self,
        sandbox: &mut PodSandbox,
        config: &PodSandboxConfig,
        _pool_config: &PoolConfig,
        annotations: &HashMap<String, String>,
    ) -> Result<Arc<dyn SandboxHandle>> {
        // Subject = tenant/identity the guest runs as. Derive it from the pod
        // namespace (best available identity at this seam); anonymous otherwise.
        let subject = {
            let ns = &config.metadata.namespace;
            if ns.is_empty() {
                Subject::anonymous()
            } else {
                Subject::new(ns.clone())
            }
        };

        let wasm = Self::resolve_guest_bytes(&sandbox.id, annotations)?;
        let inner = self.build_inner(&wasm, subject.clone())?;

        info!(
            sandbox_id = %sandbox.id,
            subject = ?subject,
            fuel = self.config.fuel,
            "Started in-process wasm sandbox"
        );

        let handle = Arc::new(WasmHandle {
            sandbox_id: sandbox.id.clone(),
            subject,
            fuel: self.config.fuel,
            inner: Mutex::new(Some(inner)),
        });

        sandbox.mark_ready();
        Ok(handle)
    }

    async fn stop(&self, sandbox: &PodSandbox) -> Result<()> {
        // Drop the live wasm instance + epoch timer; the handle struct stays so a
        // later `reset`/inspection sees a clean "stopped" state.
        if let Some(handle) = sandbox.backend_handle() {
            if let Some(h) = handle.as_any().downcast_ref::<WasmHandle>() {
                *h.inner.lock() = None;
                debug!(sandbox_id = %sandbox.id, "Stopped wasm sandbox (instance dropped)");
            }
        }
        Ok(())
    }

    async fn destroy(&self, sandbox: &PodSandbox) -> Result<()> {
        // Same as stop for an in-process backend: dropping the instance frees the
        // engine/module. The Arc<dyn SandboxHandle> is released by the pool when
        // it forgets the PodSandbox.
        self.stop(sandbox).await?;
        debug!(sandbox_id = %sandbox.id, "Destroyed wasm sandbox");
        Ok(())
    }

    async fn reset(&self, sandbox: &mut PodSandbox) -> Result<bool> {
        // Cheap reuse: drop the current guest instance and reinstantiate a fresh
        // one (new wasmtime Store ⇒ no carried-over guest state). Far cheaper than
        // a Kata VM reboot, so we reuse the sandbox in place → return `true`.
        let handle = sandbox
            .backend_handle()
            .and_then(|h| h.as_any().downcast_ref::<WasmHandle>())
            .ok_or_else(|| {
                WorkerError::SandboxInvalidState {
                    sandbox_id: sandbox.id.clone(),
                    state: "non-wasm sandbox handle".into(),
                    expected: "wasm handle".into(),
                }
            })?;

        // Reinstantiate from the same module bytes is not possible (we don't keep
        // them); instead, the substrate Sandbox is itself reusable — each
        // `eval`/export call already runs in a *fresh* Store, so guest state does
        // not persist between calls. "Reset" therefore only needs to ensure the
        // instance is live; rebuild it if it was stopped.
        let live = handle.inner.lock().is_some();
        if !live {
            // Was stopped — we can't reinstantiate without the module bytes, so a
            // stopped wasm sandbox is not reusable. Fail-closed: report not-reusable.
            warn!(
                sandbox_id = %sandbox.id,
                "wasm sandbox was stopped; cannot reset in place (module bytes not retained)"
            );
            return Ok(false);
        }
        debug!(sandbox_id = %sandbox.id, "Reset wasm sandbox (fresh Store per call ⇒ reusable in place)");
        Ok(true)
    }

    async fn get_pids(&self, _sandbox: &PodSandbox) -> Result<Vec<u32>> {
        // In-process: the guest has no OS pid. Return empty (the doc comment on
        // the trait explicitly allows this for in-process backends).
        Ok(Vec::new())
    }

    fn supports_exec(&self) -> bool {
        true
    }

    async fn exec_sync(
        &self,
        sandbox: &PodSandbox,
        command: &[String],
        timeout_secs: u64,
    ) -> Result<(i32, Vec<u8>, Vec<u8>)> {
        let handle = sandbox
            .backend_handle()
            .and_then(|h| h.as_any().downcast_ref::<WasmHandle>())
            .ok_or_else(|| {
                WorkerError::ExecFailed(format!(
                    "exec_sync: sandbox '{}' has no wasm handle",
                    sandbox.id
                ))
            })?;

        let sb = handle.sandbox()?;
        let fuel = handle.fuel;

        // Map the command to a generic guest export invocation:
        //   [<export>, "<source>"]  → Sandbox::call_export(<export>, source.as_bytes(), fuel)
        // `command[0]` is the guest export NAME and `command[1]` is the byte payload
        // shipped into guest memory over the guest's `alloc`/`memory` ABI. The
        // consolidated sandbox crate decoupled the core from Python: `call_export`
        // is the canonical GENERIC entry point (the `python` profile module builds
        // its `eval` on top of it). So `["eval", <source>]` maps to the guest's
        // `eval` export, and any other command name maps to the guest export of
        // that name — no special-casing, no `unsupported export` dead end.
        //
        // The exit code is the guest export's returned i32 status (0 = ok, nonzero
        // = guest-level error). A wasm trap (out of fuel, epoch interrupt, or a
        // dead/un-granted import being called) surfaces as a nonzero exit + stderr.
        if command.is_empty() {
            return Err(WorkerError::ExecFailed(
                "exec_sync: empty command (expected [<export>, <source>, ..])".into(),
            ));
        }

        let export = command[0].clone();
        let source = command.get(1).cloned().unwrap_or_default();
        // Run the (CPU-bound, blocking) wasm call off the async reactor. The
        // substrate Sandbox is Send+Sync; clone the Arc into the task.
        let join =
            tokio::task::spawn_blocking(move || sb.call_export(&export, source.as_bytes(), fuel));
        let result = tokio::time::timeout(Duration::from_secs(timeout_secs), join)
            .await
            .map_err(|_| WorkerError::SandboxTimeout {
                operation: format!("wasm exec_sync in {}", sandbox.id),
                timeout_secs,
            })?
            .map_err(|e| WorkerError::ExecFailed(format!("wasm exec join error: {e}")))?;

        match result {
            Ok(status) => Ok((status, Vec::new(), Vec::new())),
            // A trap (out of fuel, epoch interrupt, a dead import being called, or
            // a missing/mis-typed export) is a sandbox-level failure: nonzero exit
            // + stderr.
            Err(trap) => {
                let msg = format!("wasm guest trapped: {trap:#}");
                Ok((-1, Vec::new(), msg.into_bytes()))
            }
        }
    }

    async fn update_resources(
        &self,
        _sandbox: &PodSandbox,
        _resources: &LinuxContainerResources,
    ) -> Result<()> {
        // CRI cgroup knobs (cpu/memory quotas) have no direct wasmtime analogue.
        // The wasm DoS bounds are fuel (per-call instruction budget) + epoch
        // (wall-clock). Those are fixed per `WasmConfig` at start; re-tuning them
        // per running sandbox would require swapping the handle's fuel/epoch, which
        // P2 does not plumb. No-op with a note.
        debug!("wasm backend: update_resources is a no-op (DoS bounds are fuel/epoch, set at start)");
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Backend registry self-registration (#507 / #518) — gated on the `wasm` feature
// ─────────────────────────────────────────────────────────────────────────────

// The wasm backend is registered ONLY when compiled with `--features wasm`
// (which is what pulls in the wasmtime-bearing `hyprstream-workers-wasmtime` dep). Mirrors
// how `kata` registers only under `kata-vm`. Selection stays fail-closed: with
// the feature off, an explicit `wasm` request hits the "unknown backend" error
// path (the name simply isn't in the registry) rather than silently downgrading.
//
// An in-process wasm sandbox is the weakest isolation tier (shared host address
// space; the guarantee is the wasm capability boundary, not a kernel/VM boundary).
// It is therefore **RuntimeClass-explicit-only**: `auto_selectable: false` keeps
// it out of `"auto"` entirely so it can never become a silent fallback when
// stronger backends are absent — that would be a silent isolation downgrade, which
// the #547 MAC/ZSP model forbids. It remains selectable by its explicit `wasm`
// name (fail-closed) and is always *available* once built. The `priority` is then
// only meaningful for explicit selection; it is kept low for documentation.
inventory::submit! {
    crate::runtime::selection::BackendRegistration {
        name: "wasm",
        priority: 5,
        auto_selectable: false,
        // In-process (shared host address space); there is no separate mount
        // namespace to bind-inject a host 9P socket into (#506).
        injects_9p_socket: false,
        is_available: WasmBackend::registry_is_available,
        construct: |_ctx| {
            Ok(std::sync::Arc::new(WasmBackend::new(WasmConfig::default()))
                as std::sync::Arc<dyn crate::runtime::SandboxBackend>)
        },
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::super::client::KeyValue;
    use super::*;
    use std::path::PathBuf;

    /// A tiny hand-written guest matching the substrate's pyguest export
    /// contract: `memory`, `alloc(i32)->i32`, `dealloc(i32,i32)`, and
    /// `eval(i32,i32)->i32`. `eval` ignores the source and returns a fixed
    /// status, which is all P2's start→exec_sync test needs. wasmtime's
    /// `Module::new` (called by `Sandbox::from_bytes`) accepts `.wat` text, so we
    /// embed source rather than precompiled bytes — no build step.
    const TRIVIAL_GUEST_WAT: &str = r#"
        (module
          (memory (export "memory") 1)
          (func (export "alloc") (param i32) (result i32)
            i32.const 0)
          (func (export "dealloc") (param i32 i32))
          (func (export "eval") (param i32 i32) (result i32)
            i32.const 7))
    "#;

    fn pod_config(ns: &str, annotations: Vec<(&str, &str)>) -> PodSandboxConfig {
        let mut cfg = PodSandboxConfig::default();
        cfg.metadata.namespace = ns.to_owned();
        cfg.annotations = annotations
            .into_iter()
            .map(|(k, v)| KeyValue {
                key: k.to_owned(),
                value: v.to_owned(),
            })
            .collect();
        cfg
    }

    fn ann_map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn new_pod(id: &str, cfg: &PodSandboxConfig) -> PodSandbox {
        PodSandbox::new(id.to_owned(), cfg, PathBuf::from("/tmp/wasm-test"))
    }

    /// base64 of the trivial WAT, for the inline-source path.
    fn trivial_guest_b64() -> String {
        base64::engine::general_purpose::STANDARD.encode(TRIVIAL_GUEST_WAT.as_bytes())
    }

    #[test]
    fn backend_type_and_availability() {
        let backend = WasmBackend::new(WasmConfig::default());
        assert_eq!(backend.backend_type(), "wasm");
        // wasmtime is vendored in-process ⇒ always available once built.
        assert!(backend.is_available());
        assert!(WasmBackend::registry_is_available());
    }

    #[test]
    fn handle_downcasts() {
        let handle: Arc<dyn SandboxHandle> = Arc::new(WasmHandle {
            sandbox_id: "abc".into(),
            subject: Subject::anonymous(),
            fuel: DEFAULT_FUEL,
            inner: Mutex::new(None),
        });
        let down = handle.as_any().downcast_ref::<WasmHandle>();
        assert!(down.is_some());
        assert_eq!(down.unwrap().sandbox_id, "abc");
    }

    #[tokio::test]
    async fn start_then_exec_sync_invokes_export() {
        let backend = WasmBackend::new(WasmConfig::default());
        let b64 = trivial_guest_b64();
        let cfg = pod_config("tenant-1", vec![(ANN_WASM_BASE64, &b64)]);
        let mut pod = new_pod("wasm-1", &cfg);
        let annotations = ann_map(&[(ANN_WASM_BASE64, &b64)]);

        let handle = backend
            .start(&mut pod, &cfg, &PoolConfig::default(), &annotations)
            .await
            .expect("start");
        pod.set_backend_handle(handle);
        assert!(pod.is_ready(), "wasm sandbox should be ready after start");

        // exec_sync ["eval", "<src>"] should invoke the guest `eval` export, which
        // returns status 7 in our trivial guest.
        let (code, stdout, stderr) = backend
            .exec_sync(&pod, &["eval".into(), "anything".into()], 10)
            .await
            .expect("exec_sync");
        assert_eq!(code, 7, "exec_sync should return the guest export's status");
        assert!(stdout.is_empty());
        assert!(stderr.is_empty(), "no trap expected: {:?}", String::from_utf8_lossy(&stderr));
    }

    #[tokio::test]
    async fn exec_sync_missing_export_traps() {
        // With the generic `call_export` mapping there is no "unsupported export"
        // allow-list: any command name is dispatched to the guest export of that
        // name. A name the guest does not export fails at instantiation/export
        // lookup, which surfaces as a guest trap → nonzero exit + stderr (a
        // sandbox-level failure), not an Err result.
        let backend = WasmBackend::new(WasmConfig::default());
        let b64 = trivial_guest_b64();
        let cfg = pod_config("tenant-1", vec![(ANN_WASM_BASE64, &b64)]);
        let mut pod = new_pod("wasm-2", &cfg);
        let annotations = ann_map(&[(ANN_WASM_BASE64, &b64)]);
        let handle = backend
            .start(&mut pod, &cfg, &PoolConfig::default(), &annotations)
            .await
            .unwrap();
        pod.set_backend_handle(handle);

        let (code, stdout, stderr) = backend
            .exec_sync(&pod, &["not_a_real_export".into()], 10)
            .await
            .expect("missing export surfaces as a trap, not an Err");
        assert_eq!(code, -1, "missing guest export must produce a nonzero exit");
        assert!(stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&stderr).contains("trapped"),
            "got: {}",
            String::from_utf8_lossy(&stderr)
        );
    }

    #[tokio::test]
    async fn start_without_guest_fails_closed() {
        let backend = WasmBackend::new(WasmConfig::default());
        let cfg = pod_config("tenant-1", vec![]);
        let mut pod = new_pod("wasm-3", &cfg);
        let annotations = ann_map(&[]);
        let err = backend
            .start(&mut pod, &cfg, &PoolConfig::default(), &annotations)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("no guest module"), "got: {err}");
    }

    #[tokio::test]
    async fn reset_reuses_live_sandbox_in_place() {
        let backend = WasmBackend::new(WasmConfig::default());
        let b64 = trivial_guest_b64();
        let cfg = pod_config("tenant-1", vec![(ANN_WASM_BASE64, &b64)]);
        let mut pod = new_pod("wasm-4", &cfg);
        let annotations = ann_map(&[(ANN_WASM_BASE64, &b64)]);
        let handle = backend
            .start(&mut pod, &cfg, &PoolConfig::default(), &annotations)
            .await
            .unwrap();
        pod.set_backend_handle(handle);

        // Live sandbox ⇒ reusable in place (fresh Store per call ⇒ no carried state).
        let reusable = backend.reset(&mut pod).await.unwrap();
        assert!(reusable, "live wasm sandbox should be reusable in place");

        // Still works after reset.
        let (code, _, _) = backend
            .exec_sync(&pod, &["eval".into(), "x".into()], 10)
            .await
            .unwrap();
        assert_eq!(code, 7);
    }

    #[tokio::test]
    async fn stop_then_reset_reports_not_reusable() {
        let backend = WasmBackend::new(WasmConfig::default());
        let b64 = trivial_guest_b64();
        let cfg = pod_config("tenant-1", vec![(ANN_WASM_BASE64, &b64)]);
        let mut pod = new_pod("wasm-5", &cfg);
        let annotations = ann_map(&[(ANN_WASM_BASE64, &b64)]);
        let handle = backend
            .start(&mut pod, &cfg, &PoolConfig::default(), &annotations)
            .await
            .unwrap();
        pod.set_backend_handle(handle);

        backend.stop(&pod).await.unwrap();
        // Stopped (module bytes not retained) ⇒ not reusable, fail-closed.
        let reusable = backend.reset(&mut pod).await.unwrap();
        assert!(!reusable, "stopped wasm sandbox must report not-reusable");
    }

    #[tokio::test]
    async fn get_pids_is_empty_in_process() {
        let backend = WasmBackend::new(WasmConfig::default());
        let b64 = trivial_guest_b64();
        let cfg = pod_config("t", vec![(ANN_WASM_BASE64, &b64)]);
        let mut pod = new_pod("wasm-6", &cfg);
        let annotations = ann_map(&[(ANN_WASM_BASE64, &b64)]);
        let handle = backend
            .start(&mut pod, &cfg, &PoolConfig::default(), &annotations)
            .await
            .unwrap();
        pod.set_backend_handle(handle);
        assert!(backend.get_pids(&pod).await.unwrap().is_empty());
    }
}
