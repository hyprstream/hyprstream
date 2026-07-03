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

#[cfg(feature = "kata-vm")]
use crate::config::ImageConfig;
#[cfg(feature = "kata-vm")]
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

    /// Image storage configuration. Only the VM (`kata-vm`) path consumes the
    /// RAFS/nydus image store, so this field only exists on that build.
    #[cfg(feature = "kata-vm")]
    pub image_config: ImageConfig,

    /// RAFS/nydus image store — VM-only (`kata-vm`).
    #[cfg(feature = "kata-vm")]
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
fn select_registration<'a>(
    name: &str,
    regs: &[&'a BackendRegistration],
    is_available: impl Fn(&BackendRegistration) -> bool,
) -> Result<&'a BackendRegistration> {
    // ── Auto: highest-priority *available* registration wins ──
    //
    // Only `auto_selectable` registrations are eligible. A backend that opts out
    // (e.g. the in-process `wasm` tier) must never be auto-picked, even as a last
    // resort when nothing stronger is available — that would be a silent isolation
    // downgrade (#547 ZSP). Such backends remain reachable by explicit name below.
    if name.eq_ignore_ascii_case("auto") {
        let mut candidates: Vec<&'a BackendRegistration> =
            regs.iter().copied().filter(|r| r.auto_selectable).collect();
        // Highest priority first; ties broken by name for determinism.
        candidates.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.name.cmp(b.name)));
        for reg in candidates {
            if is_available(reg) {
                return Ok(reg);
            }
        }
        return Err(WorkerError::ConfigError(format!(
            "auto backend selection found no available sandbox backend among [{}]. \
             Install a supported runtime or set `worker.backend` to an explicit \
             backend; refusing to run a workload without isolation (fail-closed).",
            registered_names(regs)
        )));
    }

    // ── Explicit request: authoritative, fail-closed ──
    match regs.iter().find(|r| r.name.eq_ignore_ascii_case(name)) {
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
        is_available: || true,
        construct: never_available,
    };
    const LOW: BackendRegistration = BackendRegistration {
        name: "low-tier",
        priority: 10,
        auto_selectable: true,
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

    #[test]
    fn unknown_kata_hints_about_feature() {
        // Asking for kata when it isn't registered (e.g. kata-vm off) gives a
        // build-feature hint rather than a bare unknown error.
        let err = select_registration("kata", &regs(), |_| true).unwrap_err();
        assert!(err.to_string().contains("kata-vm"), "got: {err}");
    }

    // ── The real inventory must reflect the build's feature set ──

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
