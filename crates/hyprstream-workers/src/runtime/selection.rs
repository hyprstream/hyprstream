//! Inventory-based sandbox backend registry + fail-closed selection (#507).
//!
//! This is the *spine* of the unified sandbox taxonomy: one [`SandboxBackend`]
//! seam, one selection model shared by every call site (no per-site ad-hoc
//! `match` with a silent `_ => nspawn` catch-all).
//!
//! Backends are **not** a hardcoded enum. Each concrete backend self-registers a
//! [`BackendRegistration`] via `inventory::submit!` next to its implementation,
//! feature-gated by the `#[cfg]` that compiles that backend in (#518). The
//! registry is therefore *exactly* the set of backends built into this binary:
//!
//! * `nspawn` — always registered (systemd-nspawn lightweight container).
//! * `kata`   — registered only under the `kata-vm` feature (full VM isolation).
//! * `oci`    — registered only under the `oci` feature (rootless OCI/podman
//!   container, #346).
//! * `wasm`   — registered only under the `wasm` feature (in-process WebAssembly,
//!   explicit-name-only).
//!
//! YAGNI: still-speculative tiers (e.g. a raw `cri` shim) are intentionally
//! **not** registered. They gain a `submit!` when their phase actually builds
//! them; until then they simply do not exist as selectable names (an explicit
//! request errors).
//!
//! Selection is **config-driven** via `worker.backend` (a string: a registered
//! backend name, or `"auto"`):
//!
//! * `"auto"` → among registrations that are
//!   [`auto_selectable`](BackendRegistration::auto_selectable), pick the
//!   highest-[`priority`](BackendRegistration::priority) one whose
//!   [`is_available`](BackendRegistration::is_available) probe passes. The choice
//!   is logged. If none qualifies, error — never run a workload without isolation,
//!   and never auto-pick a backend that opted out of auto-selection (a silent
//!   isolation downgrade; see the in-process `wasm` tier, #547 ZSP).
//! * a concrete name → that registration, **iff** it is registered *and* its
//!   prerequisites are present. Otherwise error. We never substitute a different
//!   (weaker) backend (the #486 fail-open bug). An explicit name resolves
//!   regardless of `auto_selectable`, so an auto-excluded backend (e.g. `wasm`)
//!   is still reachable when deliberately requested by name.
//!
//! The cardinal rule is **fail-closed**: an unavailable or unknown backend
//! returns an error, never a silent downgrade to weaker isolation.

use std::sync::Arc;

use crate::config::PoolConfig;
use crate::error::{Result, WorkerError};

#[cfg(feature = "oci-image")]
use crate::config::ImageConfig;
#[cfg(feature = "oci-image")]
use crate::image::RafsStore;

use super::SandboxBackend;

/// Per-call construction context threaded to a backend's `construct` fn.
///
/// `inventory` items are `'static`, so [`BackendRegistration::construct`] must be
/// a bare `fn` pointer that cannot capture per-call state. This struct carries
/// everything a backend constructor needs at selection time.
pub struct BackendCtx {
    /// Sandbox pool configuration (paths, hypervisor, limits).
    pub pool_config: PoolConfig,

    /// Image storage configuration. Any backend that composes the RAFS image
    /// filesystem (kata over virtio-fs, or nspawn over FUSE — Model B #715)
    /// consumes it, so it exists on the `oci-image` build, not just `kata-vm`.
    /// The Cargo manifest is explicit that the RAFS image service "lives on
    /// oci-image, not kata-vm".
    #[cfg(feature = "oci-image")]
    pub image_config: ImageConfig,

    /// RAFS/nydus image store — available whenever the image filesystem service
    /// is compiled in (`oci-image`), so both the kata (virtio-fs) and nspawn
    /// (FUSE, #715) backends can compose a per-sandbox VFS from it.
    #[cfg(feature = "oci-image")]
    pub rafs_store: Arc<RafsStore>,
}

/// A compile-time sandbox-backend registration, collected via `inventory`.
///
/// Each backend submits one of these next to its implementation, gated by the
/// `#[cfg]` feature that compiles the backend in. The registry is enumerated by
/// [`resolve_backend`]; there is no central enum to keep in sync.
#[derive(Debug)]
pub struct BackendRegistration {
    /// Stable selector name, e.g. `"kata"`, `"nspawn"`. Matched against
    /// `worker.backend` (case-insensitively).
    pub name: &'static str,

    /// Auto-selection precedence: higher is preferred. `"auto"` picks the
    /// highest-priority *available* registration (among `auto_selectable` ones).
    pub priority: i32,

    /// Whether this backend is eligible for `"auto"` selection.
    ///
    /// `true` → `"auto"` may pick it (subject to `priority` and `is_available`).
    /// `false` → **explicit-name-only**: `"auto"` must never choose it, even if it
    /// is the highest-priority available (or only) registration. It remains
    /// selectable by its concrete name (fail-closed).
    ///
    /// This exists so weaker-isolation tiers cannot become a silent `"auto"`
    /// fallback when stronger backends are absent — auto-selecting them would be a
    /// silent isolation downgrade, which the #547 MAC/ZSP model forbids. The
    /// in-process `wasm` sandbox (shared host address space) sets this `false`.
    pub auto_selectable: bool,

    /// Whether this backend can **inject a 9P socket endpoint** — a host Unix
    /// domain socket — into a workload's mount namespace (#506).
    ///
    /// The Wanix-guest workload (`runtime::wanix_workload`) works by having
    /// hyprstream serve a tenant's Subject-scoped VFS `Mount` as a 9P2000.L
    /// server on a host UDS, then making that socket reachable *inside* the
    /// sandbox (bind-mounting the host socket into the container's mount
    /// namespace and setting `HYPRSTREAM_9P_SOCK`) so the guest dials back. That
    /// requires the backend to be able to bind-mount an arbitrary host path (the
    /// socket) into the workload, which the tiers do differently:
    ///
    /// * `oci` / `nspawn` → `true`: both bind-mount host paths into the
    ///   container (`podman --volume` / `nspawn --bind`), so a host UDS injects
    ///   cleanly.
    /// * `kata` → `false`: a full VM does not share the host mount namespace, so
    ///   a host UDS cannot be bind-injected the same way (the VM path uses
    ///   virtio-fs, a different mechanism — see `sandbox_fs`); it therefore does
    ///   **not** advertise this capability.
    /// * `wasm` → `false`: in-process, no separate mount namespace to inject a
    ///   host socket into.
    ///
    /// Selection is **fail-closed** on this flag: a workload that requires 9P
    /// socket injection ([`resolve_backend_9p_capable`] /
    /// [`require_9p_socket_capability`]) never resolves to a backend that does
    /// not advertise it — it errors rather than silently dropping the injection.
    pub injects_9p_socket: bool,
    /// Whether this backend **FUSE-mounts the composed per-sandbox tenant VFS**
    /// as a real host directory and POSIX-roots its workload there (Model B,
    /// #715 — the non-VM sibling of the kata virtio-fs path).
    ///
    /// `true` → the backend consumes `SandboxFs::serve_local_at` to serve the
    /// composed namespace over `/dev/fuse` and give that host directory to the
    /// container. `nspawn` sets this. `kata` does **not** — it already shares the
    /// composed VFS as a virtio-fs device, not a host FUSE mount. In-process
    /// `wasm` has no host filesystem to mount into. `oci` is deferred pending a
    /// rootless `/dev/fuse` feasibility spike and advertises `false` for now.
    ///
    /// [`require_fuse_mount_capability`] gates the mount fail-closed: a backend
    /// that does not advertise this capability is never handed a FUSE-rooted
    /// namespace (no silent POSIX-rooting in a namespace a backend can't mount).
    pub mounts_fuse_vfs: bool,

    /// Runtime prerequisite probe (PATH lookups, socket existence, …). Returns
    /// `true` when this backend can actually run on this host *right now*. Used
    /// to inform `"auto"` and to fail-close explicit requests whose prereqs are
    /// missing.
    pub is_available: fn() -> bool,

    /// Construct the concrete backend from per-call [`BackendCtx`] state.
    pub construct: fn(&BackendCtx) -> anyhow::Result<Arc<dyn SandboxBackend>>,
}

inventory::collect!(BackendRegistration);

/// Comma-separated registered backend names, for error messages.
fn registered_names(regs: &[&BackendRegistration]) -> String {
    let mut names: Vec<&str> = regs.iter().map(|r| r.name).collect();
    names.sort_unstable();
    names.join(", ")
}

/// Pure selection over a registration set, with an injectable availability
/// oracle. Factored out of [`resolve_backend`] so the fail-closed / auto-priority
/// / unknown-name logic is unit-testable without depending on which runtimes
/// happen to be installed on the test host.
///
/// Capability-agnostic; delegates to [`select_registration_with_cap`] with no
/// required capability.
fn select_registration<'a>(
    name: &str,
    regs: &[&'a BackendRegistration],
    is_available: impl Fn(&BackendRegistration) -> bool,
) -> Result<&'a BackendRegistration> {
    select_registration_with_cap(name, regs, is_available, false)
}

/// Pure selection with an additional **required-capability** gate for 9P socket
/// injection (#506), fail-closed.
///
/// When `require_9p_socket` is `true`, the chosen registration must advertise
/// [`BackendRegistration::injects_9p_socket`]:
///
/// * `"auto"` → only backends that are *both* `auto_selectable` **and**
///   `injects_9p_socket` are candidates; the highest-priority available one
///   wins. If none qualifies, error — never fall back to a backend that cannot
///   inject the socket (that would silently drop the tenant's namespace export).
/// * a concrete name → resolves the named backend, then rejects it if it does
///   not advertise the capability. We never substitute a different (capable)
///   backend for an explicitly-named incapable one — the explicit request is
///   authoritative, and the correct answer is a clean error, not a swap.
///
/// With `require_9p_socket == false` this is exactly the prior behaviour.
fn select_registration_with_cap<'a>(
    name: &str,
    regs: &[&'a BackendRegistration],
    is_available: impl Fn(&BackendRegistration) -> bool,
    require_9p_socket: bool,
) -> Result<&'a BackendRegistration> {
    // ── Auto: highest-priority *available* registration wins ──
    //
    // Only `auto_selectable` registrations are eligible. A backend that opts out
    // (e.g. the in-process `wasm` tier) must never be auto-picked, even as a last
    // resort when nothing stronger is available — that would be a silent isolation
    // downgrade (#547 ZSP). Such backends remain reachable by explicit name below.
    if name.eq_ignore_ascii_case("auto") {
        let mut candidates: Vec<&'a BackendRegistration> = regs
            .iter()
            .copied()
            .filter(|r| r.auto_selectable)
            // Capability gate: when a 9P-socket-injecting workload is being
            // placed, an incapable backend is not even a candidate — auto must
            // never downgrade to one that would silently drop the injection.
            .filter(|r| !require_9p_socket || r.injects_9p_socket)
            .collect();
        // Highest priority first; ties broken by name for determinism.
        candidates.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.name.cmp(b.name)));
        for reg in candidates {
            if is_available(reg) {
                return Ok(reg);
            }
        }
        let cap_note = if require_9p_socket {
            " that can inject a 9P socket endpoint (host UDS bind-mount)"
        } else {
            ""
        };
        return Err(WorkerError::ConfigError(format!(
            "auto backend selection found no available sandbox backend{cap_note} among [{}]. \
             Install a supported runtime or set `worker.backend` to an explicit \
             backend; refusing to run a workload without isolation (fail-closed).",
            registered_names(regs)
        )));
    }

    // ── Explicit request: authoritative, fail-closed ──
    match regs.iter().find(|r| r.name.eq_ignore_ascii_case(name)) {
        // Named + available, but cannot inject the required 9P socket → reject.
        // We do NOT substitute a capable backend for an explicitly-named one.
        Some(reg) if require_9p_socket && !reg.injects_9p_socket => {
            Err(WorkerError::ConfigError(format!(
                "sandbox backend '{}' was requested for a workload that requires 9P \
                 socket injection (a host Unix socket bind-mounted into the sandbox), \
                 but '{}' cannot inject a host UDS into a workload's mount namespace. \
                 Choose a backend that can (e.g. oci, nspawn); refusing to silently \
                 drop the namespace export (fail-closed).",
                reg.name, reg.name
            )))
        }
        Some(reg) if is_available(reg) => Ok(reg),
        Some(reg) => Err(WorkerError::ConfigError(format!(
            "sandbox backend '{}' was requested but its runtime prerequisites are \
             missing. Refusing to silently downgrade isolation to a weaker backend \
             (fail-closed).",
            reg.name
        ))),
        None => {
            let mut msg = format!(
                "unknown sandbox backend '{name}'; registered backends are [{}] (or \"auto\")",
                registered_names(regs)
            );
            // Helpful hint for the most common misconfiguration: requesting the
            // VM backend in a binary built without it.
            if name.eq_ignore_ascii_case("kata") {
                msg.push_str(
                    ". The `kata` backend is only present when built with \
                     `--features kata-vm`",
                );
            }
            Err(WorkerError::ConfigError(msg))
        }
    }
}

/// Resolve `worker.backend` to a constructed [`SandboxBackend`], fail-closed.
///
/// `name` is either a registered backend name or `"auto"`. This is the single
/// function both `factories.rs` and `bin/main.rs` route through — no scattered
/// `#[cfg]`, no `_ => nspawn` fallback.
///
/// * **`"auto"`** → among [`auto_selectable`](BackendRegistration::auto_selectable)
///   registrations, the highest-[`priority`](BackendRegistration::priority) one
///   whose [`is_available`](BackendRegistration::is_available) is true; error if
///   none (auto-excluded backends are never picked here — no silent downgrade).
/// * **concrete name** → that registration if present *and* available, regardless
///   of `auto_selectable`; otherwise error (unavailable → fail-closed; unknown →
///   error listing the registry).
pub fn resolve_backend(name: &str, ctx: &BackendCtx) -> Result<Arc<dyn SandboxBackend>> {
    let regs: Vec<&'static BackendRegistration> =
        inventory::iter::<BackendRegistration>().collect();

    let reg = select_registration(name, &regs, |r| (r.is_available)())?;

    tracing::info!(
        backend = reg.name,
        requested = name,
        priority = reg.priority,
        "sandbox backend selected (fail-closed)"
    );

    (reg.construct)(ctx).map_err(|e| {
        WorkerError::ConfigError(format!(
            "failed to construct sandbox backend '{}': {e:#}",
            reg.name
        ))
    })
}

/// Resolve `worker.backend` to a [`SandboxBackend`] that is **guaranteed capable
/// of 9P socket injection** (#506), fail-closed.
///
/// Identical to [`resolve_backend`] but with the capability gate engaged: the
/// resolved backend always advertises
/// [`injects_9p_socket`](BackendRegistration::injects_9p_socket). Use this when
/// placing a workload (e.g. the Wanix guest) that needs a host UDS injected into
/// its mount namespace. `"auto"` picks the highest-priority *capable* backend;
/// an explicit name that is incapable errors rather than being swapped for a
/// capable one.
pub fn resolve_backend_9p_capable(name: &str, ctx: &BackendCtx) -> Result<Arc<dyn SandboxBackend>> {
    let regs: Vec<&'static BackendRegistration> =
        inventory::iter::<BackendRegistration>().collect();

    let reg = select_registration_with_cap(name, &regs, |r| (r.is_available)(), true)?;

    tracing::info!(
        backend = reg.name,
        requested = name,
        priority = reg.priority,
        "sandbox backend selected for 9P-socket-injecting workload (fail-closed)"
    );

    (reg.construct)(ctx).map_err(|e| {
        WorkerError::ConfigError(format!(
            "failed to construct sandbox backend '{}': {e:#}",
            reg.name
        ))
    })
}

/// Does the registered backend named `backend_type` advertise 9P socket
/// injection? Returns `false` for an unknown/unregistered name.
///
/// This queries the *registry* (not a live backend instance), so a caller
/// holding an already-constructed `Arc<dyn SandboxBackend>` can check its
/// capability via [`SandboxBackend::backend_type`](super::SandboxBackend::backend_type).
pub fn backend_injects_9p_socket(backend_type: &str) -> bool {
    inventory::iter::<BackendRegistration>()
        .find(|r| r.name.eq_ignore_ascii_case(backend_type))
        .is_some_and(|r| r.injects_9p_socket)
}

/// Fail-closed guard used at the workload seam: error unless `backend_type`
/// advertises 9P socket injection (#506).
///
/// A caller that already holds a constructed backend (e.g. from the pool) uses
/// this to refuse — cleanly, before spawning any 9P server or building any
/// injection annotations — to place a socket-injecting workload on a backend
/// that cannot receive the socket. It never downgrades; it only reports.
pub fn require_9p_socket_capability(backend_type: &str) -> Result<()> {
    if backend_injects_9p_socket(backend_type) {
        Ok(())
    } else {
        Err(WorkerError::ConfigError(format!(
            "sandbox backend '{backend_type}' cannot inject a 9P socket endpoint \
             (a host Unix socket bind-mounted into the sandbox), which this workload \
             requires. Select a capable backend (e.g. oci, nspawn); refusing to run \
             the workload without its namespace export (fail-closed)."
        )))
    }
}

/// Pure capability check over a registration set, with the same fail-closed
/// contract as [`select_registration`]. Factored out so the Model B (#715)
/// FUSE-mount gate is unit-testable without depending on which backends the
/// build's inventory happens to contain.
fn check_fuse_mount_capability(name: &str, regs: &[&BackendRegistration]) -> Result<()> {
    match regs.iter().find(|r| r.name.eq_ignore_ascii_case(name)) {
        Some(reg) if reg.mounts_fuse_vfs => Ok(()),
        Some(reg) => Err(WorkerError::ConfigError(format!(
            "sandbox backend '{}' does not advertise the FUSE tenant-VFS mount \
             capability (mounts_fuse_vfs); refusing to POSIX-root a workload in a \
             namespace this backend was not declared able to mount (fail-closed).",
            reg.name
        ))),
        None => Err(WorkerError::ConfigError(format!(
            "unknown sandbox backend '{name}'; cannot check FUSE tenant-VFS mount \
             capability"
        ))),
    }
}

/// Fail-closed gate for Model B (#715): assert `backend_type` advertises the
/// [`mounts_fuse_vfs`](BackendRegistration::mounts_fuse_vfs) capability before a
/// caller FUSE-mounts the composed tenant VFS as that backend's root.
///
/// Parallels [`resolve_backend`]'s fail-closed contract: an unknown backend, or
/// a registered one that does not advertise the capability, errors rather than
/// proceeding — we never POSIX-root a workload in a namespace a backend was not
/// declared able to mount, and never silently substitute a different transport.
pub fn require_fuse_mount_capability(backend_type: &str) -> Result<()> {
    let regs: Vec<&'static BackendRegistration> =
        inventory::iter::<BackendRegistration>().collect();
    check_fuse_mount_capability(backend_type, &regs)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // ── Synthetic registrations for deterministic selection-logic tests ──
    //
    // These never touch the real `is_available` probes or `construct` fns; they
    // exercise the pure `select_registration` spine (auto-priority, fail-closed,
    // unknown-name) independent of what runtimes are installed.

    fn never_available(_: &BackendCtx) -> anyhow::Result<Arc<dyn SandboxBackend>> {
        unreachable!("construct is not called in selection-logic tests")
    }

    const HIGH: BackendRegistration = BackendRegistration {
        name: "high-tier",
        priority: 100,
        auto_selectable: true,
        injects_9p_socket: false,
        mounts_fuse_vfs: false,
        is_available: || true,
        construct: never_available,
    };
    const LOW: BackendRegistration = BackendRegistration {
        name: "low-tier",
        priority: 10,
        auto_selectable: true,
        injects_9p_socket: false,
        mounts_fuse_vfs: false,
        is_available: || true,
        construct: never_available,
    };

    /// A FUSE-mount-capable backend, modeling the Model B (#715) nspawn target:
    /// it advertises `mounts_fuse_vfs: true`.
    const FUSE_CAP: BackendRegistration = BackendRegistration {
        name: "fuse-tier",
        priority: 50,
        auto_selectable: true,
        injects_9p_socket: false,
        mounts_fuse_vfs: true,
        is_available: || true,
        construct: never_available,
    };

    /// An auto-excluded (explicit-name-only) backend, modeling the in-process
    /// `wasm` tier: even at the highest priority and always available, `"auto"`
    /// must never pick it.
    const EXPLICIT_ONLY: BackendRegistration = BackendRegistration {
        name: "explicit-only",
        priority: 1000,
        auto_selectable: false,
        injects_9p_socket: false,
        mounts_fuse_vfs: false,
        is_available: || true,
        construct: never_available,
    };

    /// A high-priority backend that CAN inject a 9P socket (models `oci`).
    const NINEP_HIGH: BackendRegistration = BackendRegistration {
        name: "ninep-high",
        priority: 100,
        auto_selectable: true,
        injects_9p_socket: true,
        mounts_fuse_vfs: false,
        is_available: || true,
        construct: never_available,
    };
    /// A lower-priority backend that CAN inject a 9P socket (models `nspawn`).
    const NINEP_LOW: BackendRegistration = BackendRegistration {
        name: "ninep-low",
        priority: 10,
        auto_selectable: true,
        injects_9p_socket: true,
        mounts_fuse_vfs: false,
        is_available: || true,
        construct: never_available,
    };

    fn regs() -> Vec<&'static BackendRegistration> {
        vec![&HIGH, &LOW]
    }

    #[test]
    fn auto_picks_highest_priority_available() {
        // Both available → strongest (highest priority) wins.
        let r = select_registration("auto", &regs(), |_| true).unwrap();
        assert_eq!(r.name, "high-tier");
    }

    #[test]
    fn auto_skips_unavailable_to_next_priority() {
        // Strongest unavailable → auto must fall to the next available tier,
        // never error while a usable backend remains.
        let r = select_registration("auto", &regs(), |reg| reg.name != "high-tier").unwrap();
        assert_eq!(r.name, "low-tier");
    }

    #[test]
    fn auto_errors_when_nothing_available() {
        let err = select_registration("auto", &regs(), |_| false).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no available sandbox backend"), "got: {msg}");
        assert!(msg.contains("fail-closed"), "got: {msg}");
    }

    #[test]
    fn auto_never_picks_explicit_only_even_as_only_backend() {
        // An auto-excluded backend that is the *only* registration (and available)
        // must NOT be auto-selected — `"auto"` errors instead of downgrading.
        let only_explicit = vec![&EXPLICIT_ONLY];
        let err = select_registration("auto", &only_explicit, |_| true).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no available sandbox backend"), "got: {msg}");
        assert!(msg.contains("fail-closed"), "got: {msg}");
    }

    #[test]
    fn auto_skips_explicit_only_for_a_weaker_auto_eligible_backend() {
        // Even though EXPLICIT_ONLY has the highest priority and is available,
        // `"auto"` must prefer a lower-priority *auto-eligible* backend over it —
        // never an isolation downgrade to the auto-excluded tier.
        let regs = vec![&EXPLICIT_ONLY, &LOW];
        let r = select_registration("auto", &regs, |_| true).unwrap();
        assert_eq!(r.name, "low-tier", "auto must skip the auto-excluded backend");
    }

    #[test]
    fn explicit_only_still_selectable_by_name() {
        // Auto-exclusion does not remove the backend from the registry: an
        // explicit request resolves it (fail-closed when unavailable).
        let regs = vec![&EXPLICIT_ONLY, &LOW];
        let r = select_registration("explicit-only", &regs, |_| true).unwrap();
        assert_eq!(r.name, "explicit-only");
    }

    #[test]
    fn explicit_available_resolves_exactly() {
        let r = select_registration("low-tier", &regs(), |_| true).unwrap();
        assert_eq!(r.name, "low-tier");
        // Case-insensitive.
        let r = select_registration("LOW-TIER", &regs(), |_| true).unwrap();
        assert_eq!(r.name, "low-tier");
    }

    #[test]
    fn explicit_unavailable_fails_closed_no_downgrade() {
        // "high-tier" requested but unavailable → error, must NOT downgrade to
        // the available "low-tier" backend.
        let err =
            select_registration("high-tier", &regs(), |reg| reg.name == "low-tier").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("prerequisites are missing"), "got: {msg}");
        assert!(msg.contains("fail-closed"), "got: {msg}");
        // Must not name the weaker backend it could have silently fallen back to.
        assert!(!msg.contains("low-tier"), "must not mention a downgrade target: {msg}");
    }

    #[test]
    fn unknown_name_errors_listing_registry() {
        let err = select_registration("bogus", &regs(), |_| true).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("unknown sandbox backend 'bogus'"), "got: {msg}");
        assert!(msg.contains("high-tier"), "should list registered names: {msg}");
        assert!(msg.contains("low-tier"), "should list registered names: {msg}");
    }

    // ── 9P-socket-injection capability gate (#506), fail-closed ──

    #[test]
    fn cap_auto_picks_highest_priority_capable_backend() {
        // Both capable → strongest (highest priority) wins.
        let regs = vec![&NINEP_HIGH, &NINEP_LOW];
        let r = select_registration_with_cap("auto", &regs, |_| true, true).unwrap();
        assert_eq!(r.name, "ninep-high");
    }

    #[test]
    fn cap_auto_skips_incapable_backend_even_if_higher_priority() {
        // HIGH (priority 100) cannot inject; NINEP_LOW (priority 10) can. With
        // the capability required, auto MUST pick the lower-priority *capable*
        // backend, never the higher-priority incapable one.
        let regs = vec![&HIGH, &NINEP_LOW];
        let r = select_registration_with_cap("auto", &regs, |_| true, true).unwrap();
        assert_eq!(r.name, "ninep-low", "auto must pick the capable backend");
    }

    #[test]
    fn cap_auto_errors_when_no_capable_backend_available() {
        // Only incapable backends present → auto errors rather than downgrading
        // to one that would silently drop the 9P injection.
        let regs = vec![&HIGH, &LOW];
        let err = select_registration_with_cap("auto", &regs, |_| true, true).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("9P socket endpoint"), "got: {msg}");
        assert!(msg.contains("fail-closed"), "got: {msg}");
    }

    #[test]
    fn cap_auto_errors_when_capable_backend_unavailable() {
        // The only capable backend is unavailable → auto fails closed (no
        // fallback to an available-but-incapable backend).
        let regs = vec![&NINEP_HIGH, &LOW];
        let err = select_registration_with_cap("auto", &regs, |r| r.name == "low-tier", true)
            .unwrap_err();
        assert!(err.to_string().contains("9P socket endpoint"), "got: {err}");
    }

    #[test]
    fn cap_explicit_capable_resolves() {
        let regs = vec![&NINEP_HIGH, &LOW];
        let r = select_registration_with_cap("ninep-high", &regs, |_| true, true).unwrap();
        assert_eq!(r.name, "ninep-high");
    }

    #[test]
    fn cap_explicit_incapable_errors_no_substitution() {
        // Explicitly asking for an incapable backend when the capability is
        // required errors — and must NOT silently swap in a capable one.
        let regs = vec![&HIGH, &NINEP_LOW];
        let err =
            select_registration_with_cap("high-tier", &regs, |_| true, true).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("9P socket injection"), "got: {msg}");
        assert!(msg.contains("fail-closed"), "got: {msg}");
        // Must not name the capable backend it could have swapped to.
        assert!(!msg.contains("ninep-low"), "must not suggest a silent swap: {msg}");
    }

    #[test]
    fn cap_off_ignores_incapable_flag() {
        // With the capability NOT required, an incapable backend resolves fine
        // (the gate is opt-in; default behaviour is unchanged).
        let regs = vec![&HIGH, &LOW];
        let r = select_registration_with_cap("high-tier", &regs, |_| true, false).unwrap();
        assert_eq!(r.name, "high-tier");
        let r = select_registration_with_cap("auto", &regs, |_| true, false).unwrap();
        assert_eq!(r.name, "high-tier");
    }

    #[test]
    fn require_capability_helper_is_fail_closed() {
        // The registry-backed helper the workload seam uses: real registered
        // backends reflect reality (oci/nspawn capable; kata/wasm not), and an
        // unknown name is treated as incapable (fail-closed).
        assert!(!backend_injects_9p_socket("does-not-exist"));
        assert!(require_9p_socket_capability("does-not-exist").is_err());

        // nspawn is always registered and IS capable (host --bind).
        assert!(backend_injects_9p_socket("nspawn"), "nspawn can bind a host UDS");
        assert!(require_9p_socket_capability("nspawn").is_ok());
        // Case-insensitive, mirroring name matching.
        assert!(backend_injects_9p_socket("NSPAWN"));
    }

    #[cfg(feature = "oci")]
    #[test]
    fn oci_advertises_9p_socket_injection() {
        let oci = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "oci")
            .expect("oci registered under the oci feature");
        assert!(oci.injects_9p_socket, "oci bind-mounts a host UDS → capable");
    }

    #[cfg(feature = "kata-vm")]
    #[test]
    fn kata_does_not_advertise_9p_socket_injection() {
        // A VM does not share the host mount namespace, so it cannot bind-inject
        // a host UDS the way oci/nspawn do — it must NOT advertise the capability.
        let kata = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "kata")
            .expect("kata registered under the kata-vm feature");
        assert!(
            !kata.injects_9p_socket,
            "kata (VM, no shared host mount ns) must not advertise host-UDS injection"
        );
    }

    #[cfg(feature = "wasm")]
    #[test]
    fn wasm_does_not_advertise_9p_socket_injection() {
        let wasm = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "wasm")
            .expect("wasm registered under the wasm feature");
        assert!(
            !wasm.injects_9p_socket,
            "in-process wasm has no separate mount ns to inject a host socket into"
        );
    }

    #[test]
    fn nspawn_advertises_9p_socket_injection() {
        let nspawn = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "nspawn")
            .expect("nspawn always registered");
        assert!(nspawn.injects_9p_socket, "nspawn --bind can inject a host UDS");
    }

    #[test]
    fn unknown_kata_hints_about_feature() {
        // Asking for kata when it isn't registered (e.g. kata-vm off) gives a
        // build-feature hint rather than a bare unknown error.
        let err = select_registration("kata", &regs(), |_| true).unwrap_err();
        assert!(err.to_string().contains("kata-vm"), "got: {err}");
    }

    // ── The real inventory must reflect the build's feature set ──

    // ── Model B (#715): FUSE tenant-VFS mount capability gate ──

    #[test]
    fn fuse_capability_gate_allows_advertised_backend() {
        let regs = vec![&FUSE_CAP, &LOW];
        assert!(check_fuse_mount_capability("fuse-tier", &regs).is_ok());
        // Case-insensitive, mirroring backend selection.
        assert!(check_fuse_mount_capability("FUSE-TIER", &regs).is_ok());
    }

    #[test]
    fn fuse_capability_gate_fails_closed_for_non_advertising_backend() {
        // A registered backend that does NOT advertise the capability must be
        // refused — no silent POSIX-rooting in a namespace it can't mount.
        let regs = vec![&FUSE_CAP, &LOW];
        let err = check_fuse_mount_capability("low-tier", &regs).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("does not advertise"), "got: {msg}");
        assert!(msg.contains("fail-closed"), "got: {msg}");
    }

    #[test]
    fn fuse_capability_gate_unknown_backend_errors() {
        let regs = vec![&FUSE_CAP];
        let err = check_fuse_mount_capability("bogus", &regs).unwrap_err();
        assert!(
            err.to_string().contains("unknown sandbox backend"),
            "got: {err}"
        );
    }

    #[test]
    fn nspawn_advertises_fuse_mount_capability_in_real_inventory() {
        // The real inventory: nspawn is the Model B FUSE-root target (#715), so
        // the public gate resolves it Ok. nspawn is always registered.
        assert!(
            require_fuse_mount_capability("nspawn").is_ok(),
            "nspawn must advertise mounts_fuse_vfs"
        );
    }

    #[cfg(feature = "kata-vm")]
    #[test]
    fn kata_does_not_advertise_fuse_mount_capability() {
        // kata already shares the composed VFS via virtio-fs, not a host FUSE
        // mount, so it must NOT advertise the capability (fail-closed).
        let err = require_fuse_mount_capability("kata").unwrap_err();
        assert!(err.to_string().contains("does not advertise"), "got: {err}");
    }

    #[test]
    fn registry_contains_nspawn_always() {
        let names: Vec<&str> = inventory::iter::<BackendRegistration>()
            .map(|r| r.name)
            .collect();
        assert!(names.contains(&"nspawn"), "nspawn must always register; got {names:?}");
    }

    #[test]
    fn registry_contains_kata_only_under_feature() {
        let has_kata = inventory::iter::<BackendRegistration>().any(|r| r.name == "kata");
        assert_eq!(
            has_kata,
            cfg!(feature = "kata-vm"),
            "kata registration must track the kata-vm feature"
        );
    }

    #[test]
    fn no_speculative_backends_registered() {
        // YAGNI: still-speculative tiers must not be in the registry until built.
        // `wasm` (#505 P2) and `oci` (#346) are now real backends, each handled by
        // its own feature-tracking test below; only `cri` remains speculative.
        let name = "cri";
        assert!(
            !inventory::iter::<BackendRegistration>().any(|r| r.name == name),
            "speculative backend '{name}' must not be registered (YAGNI)"
        );
    }

    #[test]
    fn registry_contains_oci_only_under_feature() {
        // The rootless OCI (podman) backend (#346) registers iff built with
        // `--features oci`. With the feature off it must NOT be a selectable name
        // (fail-closed), mirroring `wasm`/`kata`.
        let has_oci = inventory::iter::<BackendRegistration>().any(|r| r.name == "oci");
        assert_eq!(
            has_oci,
            cfg!(feature = "oci"),
            "oci registration must track the oci feature"
        );
    }

    #[cfg(feature = "oci")]
    #[test]
    fn oci_is_auto_selectable_and_outranks_nspawn() {
        // A rootless container (user namespaces + seccomp + its own image rootfs)
        // is a real isolation tier, so — unlike the in-process `wasm` backend — it
        // is auto-selectable and outranks nspawn (host-rootfs sharing).
        let oci = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "oci")
            .expect("oci registered under the oci feature");
        let nspawn = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "nspawn")
            .expect("nspawn always registered");
        assert!(oci.auto_selectable, "oci is a real isolation tier → auto-selectable");
        assert!(
            oci.priority > nspawn.priority,
            "rootless OCI must outrank nspawn for auto-selection"
        );
    }

    #[test]
    fn registry_contains_wasm_only_under_feature() {
        // The in-process wasm backend (#505 P2) registers iff built with
        // `--features wasm` (which vendors the wasmtime substrate). With the
        // feature off it must NOT be a selectable name (fail-closed).
        let has_wasm = inventory::iter::<BackendRegistration>().any(|r| r.name == "wasm");
        assert_eq!(
            has_wasm,
            cfg!(feature = "wasm"),
            "wasm registration must track the wasm feature"
        );
    }

    #[cfg(feature = "wasm")]
    #[test]
    fn wasm_is_auto_excluded() {
        // The in-process wasm backend (shared host address space) must be flagged
        // explicit-name-only so it can never be auto-picked (#547 ZSP — no silent
        // isolation downgrade).
        let wasm = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "wasm")
            .expect("wasm registered under the wasm feature");
        assert!(
            !wasm.auto_selectable,
            "wasm (in-process, shared address space) must be excluded from auto-selection"
        );
    }

    #[cfg(feature = "wasm")]
    #[test]
    fn auto_never_returns_wasm_even_when_only_available_backend() {
        // With *only* the real wasm registration present (registered + available),
        // `"auto"` must still error rather than downgrade to it.
        let wasm = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "wasm")
            .expect("wasm registered under the wasm feature");
        let only_wasm = vec![wasm];
        let err = select_registration("auto", &only_wasm, |r| (r.is_available)()).unwrap_err();
        assert!(
            err.to_string().contains("no available sandbox backend"),
            "auto must not pick wasm; got: {err}"
        );

        // And with nspawn alongside it, `"auto"` resolves to nspawn — never wasm.
        // Inject a deterministic availability oracle (everything available) so this
        // does NOT depend on the runner actually having nspawn installed: nspawn is
        // auto_selectable (and outranks), wasm is auto_selectable:false (excluded), so
        // auto must return nspawn. (Real-runtime availability is exercised elsewhere;
        // here we assert the selection *logic*, env-independently.)
        let nspawn = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "nspawn")
            .expect("nspawn always registered");
        let both = vec![wasm, nspawn];
        let r = select_registration("auto", &both, |_| true).unwrap();
        assert_eq!(r.name, "nspawn", "auto must pick nspawn, never wasm");
    }

    #[cfg(feature = "wasm")]
    #[test]
    fn wasm_selectable_by_name_and_available() {
        // Explicit `wasm` request resolves to the wasm registration, and it is
        // always available once compiled in (wasmtime is vendored in-process).
        let regs: Vec<&'static BackendRegistration> =
            inventory::iter::<BackendRegistration>().collect();
        let reg = select_registration("wasm", &regs, |r| (r.is_available)())
            .expect("wasm selectable by name when feature on");
        assert_eq!(reg.name, "wasm");
        assert!((reg.is_available)(), "wasm is always available once built");
    }

    #[cfg(feature = "kata-vm")]
    #[test]
    fn kata_outranks_nspawn_in_priority() {
        let kata = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "kata")
            .unwrap();
        let nspawn = inventory::iter::<BackendRegistration>()
            .find(|r| r.name == "nspawn")
            .unwrap();
        assert!(
            kata.priority > nspawn.priority,
            "kata (VM) must outrank nspawn for auto-selection"
        );
    }
}
