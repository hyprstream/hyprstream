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
//!
//! YAGNI: speculative tiers (Oci/Cri/Wasm) are intentionally **not** registered.
//! They gain a `submit!` when their phase actually builds them; until then they
//! simply do not exist as selectable names (an explicit request errors).
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

use hyprstream_discovery::scheduling::{
    self, CapabilitySource, Predicate, RejectionReason, Resource, SelectionReport,
};

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

// ─────────────────────────────────────────────────────────────────────────────
// Wizard / CLI diagnostics (#348)
// ─────────────────────────────────────────────────────────────────────────────
//
// `resolve_backend` answers "which backend do I get"; the functions below add
// the backend-local "why not" knowledge a `hyprstream service` CLI subcommand
// or an interactive setup wizard needs — per-backend prerequisite sub-reasons
// and a status snapshot of the compiled-in registry. None of this changes
// selection semantics; it is read-only introspection over the same registry.
//
// The generic filter→rank→select explain trace (`SelectionReport<C>`) is owned
// by the #628 scheduling substrate (`hyprstream-discovery::scheduling`);
// `explain_selection` below composes on it (see the "Scheduling-substrate
// adoption" section) rather than carrying a second, backend-specific copy.

/// Backend-specific sub-reason for why `is_available()` returned `false`,
/// e.g. "cloud-hypervisor not on PATH" for `kata`. Falls back to a generic
/// message for backends this function doesn't have specific knowledge of.
///
/// This mirrors the prerequisite probes in each backend's
/// `registry_is_available()` (see `kata_backend.rs`, `nspawn.rs`,
/// `wasm_backend.rs`) but is allowed to drift from them in detail since it is
/// diagnostic-only — it never gates selection itself.
fn unavailable_reason(name: &str) -> String {
    match name {
        "kata" => "runtime prerequisites missing: `cloud-hypervisor` not found on PATH".to_owned(),
        "nspawn" => {
            "runtime prerequisites missing: `systemd-nspawn` and/or `machinectl` not found on PATH"
                .to_owned()
        }
        "podman" => "runtime prerequisites missing: `podman` not found on PATH".to_owned(),
        "wasm" => "wasm runtime prerequisites are not satisfied".to_owned(),
        other => format!("runtime prerequisites for '{other}' are not satisfied"),
    }
}

/// Wizard/CLI-facing snapshot of one registered backend's status. A future
/// `hyprstream service` subcommand can render `Vec<BackendStatus>` as a table
/// (name / priority / auto-eligible / available / why-not).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendStatus {
    pub name: &'static str,
    pub priority: i32,
    pub auto_selectable: bool,
    pub available: bool,
    pub reason_if_unavailable: Option<String>,
}

/// List every compiled-in backend with its live availability, for wizard/CLI
/// display. Ordered by descending priority (matching `"auto"`'s preference
/// order), name ascending as a tiebreak.
pub fn list_backends_for_wizard() -> Vec<BackendStatus> {
    let mut regs: Vec<&'static BackendRegistration> =
        inventory::iter::<BackendRegistration>().collect();
    regs.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.name.cmp(b.name)));

    regs.into_iter()
        .map(|reg| {
            let available = (reg.is_available)();
            BackendStatus {
                name: reg.name,
                priority: reg.priority,
                auto_selectable: reg.auto_selectable,
                available,
                reason_if_unavailable: if available {
                    None
                } else {
                    Some(unavailable_reason(reg.name))
                },
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Scheduling-substrate adoption (#628)
// ─────────────────────────────────────────────────────────────────────────────
//
// The explain surface #348 needs is the #628 substrate's generic
// `filter → rank → select → SelectionReport<C>`. Backend selection is one
// consumer of that shape: candidates are the compiled-in registrations,
// availability is a `Predicate` whose `RejectionReason` is sourced from
// `unavailable_reason`, and rank is by descending priority. `explain_selection`
// runs the substrate pipeline and returns its `SelectionReport`, so the trace
// falls out of `select` for free — no bespoke report type here.
//
// `resolve_backend`/`select_registration` above remain the authoritative
// construction path (they encode explicit-name and unknown-name semantics that
// are not a single generic `select` call); the substrate governs the "auto"
// explain trace, keeping one explain shape across every scheduling surface.
//
// Full resource-aware ranking (GPU/memory/NUMA via `ResourceRequest`) plugs in
// here as additional predicates + a resource-aware `rank_by`; that is downstream
// consumer work (#341/#525). This adoption wires availability-as-predicate +
// explain + the capability-source seam.

/// A cloneable projection of a [`BackendRegistration`] for the substrate's
/// generic pipeline (`SelectionReport<C>` requires `C: Clone`; a registration
/// holds `fn` pointers and is not cloneable). Carries exactly what selection
/// filtering, ranking, and the explain trace need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackendCandidate {
    /// Registered backend name (`"kata"`, `"nspawn"`, …).
    pub name: &'static str,
    /// Auto-selection precedence (higher preferred).
    pub priority: i32,
    /// Whether `"auto"` may pick this backend.
    pub auto_selectable: bool,
    /// Live availability probe result at the time the candidate was captured.
    pub available: bool,
}

impl BackendCandidate {
    fn from_registration(reg: &BackendRegistration) -> Self {
        Self {
            name: reg.name,
            priority: reg.priority,
            auto_selectable: reg.auto_selectable,
            available: (reg.is_available)(),
        }
    }
}

/// Runtime availability + capability truth a backend contributes to the shared
/// scheduling vocabulary. A node-record publisher aggregates these across every
/// registered backend so a placement record's `runtime=<name>` labels reflect
/// what is actually available here (no hand-declaration, no drift — #628).
impl CapabilitySource for BackendCandidate {
    fn capability_labels(&self) -> Vec<(String, String)> {
        vec![
            ("runtime".to_owned(), self.name.to_owned()),
            (
                format!("runtime.{}.available", self.name),
                self.available.to_string(),
            ),
        ]
    }

    fn capability_resources(&self) -> Vec<Resource> {
        // Backend selection contributes runtime-availability labels, not
        // countable capacity; GPU/memory/NUMA capacity comes from other sources
        // (DevicePool, node liveness). See the module note on resource-aware rank.
        Vec::new()
    }
}

/// Availability predicate: rejects a candidate whose runtime prerequisites are
/// missing, carrying the backend-specific [`unavailable_reason`] as the
/// [`RejectionReason`] so it lands in the [`SelectionReport`] trace.
fn availability_predicate() -> Predicate<BackendCandidate> {
    Box::new(|c: &BackendCandidate| {
        if c.available {
            None
        } else {
            Some(RejectionReason(unavailable_reason(c.name)))
        }
    })
}

/// Auto-eligibility predicate: rejects backends excluded from `"auto"` (the
/// isolation-downgrade guard, #547 ZSP), with the reason recorded in the trace.
fn auto_eligibility_predicate() -> Predicate<BackendCandidate> {
    Box::new(|c: &BackendCandidate| {
        if c.auto_selectable {
            None
        } else {
            Some(RejectionReason(format!(
                "backend '{}' is explicit-name-only (excluded from auto-selection; \
                 no silent isolation downgrade)",
                c.name
            )))
        }
    })
}

/// Explain what `"auto"` selection would decide over the compiled-in registry,
/// as the shared [`SelectionReport`]. The winning candidate is the highest-
/// priority backend that is both auto-eligible and available; every other
/// candidate carries the [`RejectionReason`] that excluded it.
///
/// This is diagnostics only (a wizard/CLI "why this backend?" trace); the
/// authoritative construction path is [`resolve_backend`]. Both agree on the
/// same winner for `"auto"` because both rank by descending priority over the
/// auto-eligible, available set.
pub fn explain_selection() -> SelectionReport<BackendCandidate> {
    let candidates: Vec<BackendCandidate> = inventory::iter::<BackendRegistration>()
        .map(BackendCandidate::from_registration)
        .collect();

    let predicates = vec![auto_eligibility_predicate(), availability_predicate()];

    let mut report = scheduling::select(
        &candidates,
        &predicates,
        // Highest priority first; name ascending as a deterministic tiebreak —
        // matching `select_registration`'s "auto" ordering exactly.
        |a, b| b.priority.cmp(&a.priority).then_with(|| a.name.cmp(b.name)),
        1, // "auto" selects a single backend
    );
    report.requested = "auto".to_owned();
    report
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
        // YAGNI: still-speculative tiers (Oci/Cri) must not be in the registry
        // until built. `wasm` is now a real backend (#505 P2) and is handled
        // separately below — it registers iff the `wasm` feature is on.
        for name in ["oci", "cri"] {
            assert!(
                !inventory::iter::<BackendRegistration>().any(|r| r.name == name),
                "speculative backend '{name}' must not be registered (YAGNI)"
            );
        }
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

    // ── N-backend fallback-chain semantics (#348) ──
    //
    // Synthetic 3-tier registry modeling a future kata > podman > nspawn world
    // (podman lands under #346; until then this just proves the *logic*
    // generalizes past two backends — it does not depend on #346 landing).

    const MID: BackendRegistration = BackendRegistration {
        name: "mid-tier",
        priority: 50,
        auto_selectable: true,
        is_available: || true,
        construct: never_available,
    };

    fn three_tier_regs() -> Vec<&'static BackendRegistration> {
        vec![&HIGH, &MID, &LOW]
    }

    #[test]
    fn auto_three_tier_all_available_picks_highest() {
        let r = select_registration("auto", &three_tier_regs(), |_| true).unwrap();
        assert_eq!(r.name, "high-tier");
    }

    #[test]
    fn auto_three_tier_degrades_through_priority_order_one_by_one() {
        // Simulate backends disappearing one at a time, strongest first —
        // exactly the "kata missing -> try podman -> try nspawn" chain a
        // wizard cares about. Each step must fall exactly one tier, never
        // skip ahead and never error while something remains available.
        let regs = three_tier_regs();

        let r = select_registration("auto", &regs, |reg| reg.name != "high-tier").unwrap();
        assert_eq!(r.name, "mid-tier", "first degradation: high unavailable -> mid");

        let r = select_registration("auto", &regs, |reg| reg.name == "low-tier").unwrap();
        assert_eq!(r.name, "low-tier", "second degradation: only low remains");
    }

    #[test]
    fn auto_three_tier_errors_only_when_all_unavailable() {
        let err = select_registration("auto", &three_tier_regs(), |_| false).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no available sandbox backend"), "got: {msg}");
        // Must list all three registered names so the operator knows what to install.
        assert!(msg.contains("high-tier") && msg.contains("mid-tier") && msg.contains("low-tier"));
    }

    #[test]
    fn auto_three_tier_skips_middle_unavailable_tier() {
        // Mid-tier specifically down (e.g. podman not installed) must not
        // wrongly fall all the way to low; high still wins if available, and
        // if high is also down it must land on low (skipping mid), not error.
        let regs = three_tier_regs();
        let r = select_registration("auto", &regs, |reg| reg.name != "mid-tier").unwrap();
        assert_eq!(r.name, "high-tier", "high still available, mid down -> high wins");

        let r =
            select_registration("auto", &regs, |reg| reg.name == "low-tier").unwrap();
        assert_eq!(r.name, "low-tier", "high and mid down -> falls through mid to low");
    }

    // ── unavailable_reason (#348 wizard diagnostics) ──

    #[test]
    fn unavailable_reason_has_per_backend_specifics() {
        assert!(unavailable_reason("kata").contains("cloud-hypervisor"));
        assert!(unavailable_reason("nspawn").contains("systemd-nspawn"));
        assert!(unavailable_reason("podman").contains("podman"));
        // Unknown backend still gets a non-empty, non-panicking message.
        assert!(unavailable_reason("future-backend").contains("future-backend"));
    }

    // ── list_backends_for_wizard (#348) ──

    #[test]
    fn wizard_list_reflects_the_real_compiled_registry() {
        let statuses = list_backends_for_wizard();
        let real_count = inventory::iter::<BackendRegistration>().count();
        assert_eq!(statuses.len(), real_count);

        let nspawn = statuses.iter().find(|s| s.name == "nspawn");
        assert!(nspawn.is_some(), "nspawn must always appear in the wizard list");

        // Ordered by descending priority.
        for pair in statuses.windows(2) {
            assert!(pair[0].priority >= pair[1].priority, "not sorted by priority: {statuses:?}");
        }

        // Unavailable backends must carry a reason; available ones must not.
        for s in &statuses {
            assert_eq!(s.reason_if_unavailable.is_some(), !s.available);
        }
    }

    // ── Scheduling-substrate explain adoption (#628) ──
    //
    // These exercise the substrate pipeline over `BackendCandidate` directly (a
    // deterministic candidate set), proving the migrated explain produces the
    // shared `SelectionReport<C>` and that `unavailable_reason` flows through as
    // the `RejectionReason`. `explain_selection()` itself is a thin wrapper that
    // feeds the real inventory into this same pipeline.

    fn candidate(name: &'static str, priority: i32, auto: bool, avail: bool) -> BackendCandidate {
        BackendCandidate { name, priority, auto_selectable: auto, available: avail }
    }

    /// Reusable copy of `explain_selection`'s pipeline over an explicit candidate
    /// set (so tests don't depend on the host's installed runtimes).
    fn explain_over(candidates: &[BackendCandidate]) -> SelectionReport<BackendCandidate> {
        let predicates = vec![auto_eligibility_predicate(), availability_predicate()];
        let mut report = scheduling::select(
            candidates,
            &predicates,
            |a, b| b.priority.cmp(&a.priority).then_with(|| a.name.cmp(b.name)),
            1,
        );
        report.requested = "auto".to_owned();
        report
    }

    #[test]
    fn explain_selects_highest_priority_available_auto_eligible() {
        let cands = vec![
            candidate("kata", 100, true, true),
            candidate("nspawn", 10, true, true),
        ];
        let report = explain_over(&cands);
        let selected: Vec<&str> =
            report.candidates.iter().filter(|o| o.selected).map(|o| o.candidate.name).collect();
        assert_eq!(selected, vec!["kata"], "highest-priority available wins");
        assert_eq!(report.requested, "auto");
    }

    #[test]
    fn explain_reports_unavailable_reason_as_rejection() {
        // kata unavailable → excluded, and its RejectionReason must be the
        // backend-specific `unavailable_reason` (the #628 integration point).
        let cands = vec![
            candidate("kata", 100, true, false),
            candidate("nspawn", 10, true, true),
        ];
        let report = explain_over(&cands);
        let kata = report.candidates.iter().find(|o| o.candidate.name == "kata").unwrap();
        assert!(!kata.selected);
        assert_eq!(kata.rejection_reason.as_deref(), Some(unavailable_reason("kata").as_str()));
        assert!(kata.rejection_reason.as_deref().unwrap().contains("cloud-hypervisor"));
        // nspawn (available, auto-eligible) is the winner.
        let nspawn = report.candidates.iter().find(|o| o.candidate.name == "nspawn").unwrap();
        assert!(nspawn.selected);
    }

    #[test]
    fn explain_excludes_auto_ineligible_with_reason() {
        // An explicit-name-only backend (e.g. wasm) at the highest priority and
        // available must be rejected in the trace, and a weaker auto-eligible
        // backend selected — matching `select_registration`'s auto semantics.
        let cands = vec![
            candidate("wasm", 1000, false, true),
            candidate("nspawn", 10, true, true),
        ];
        let report = explain_over(&cands);
        let wasm = report.candidates.iter().find(|o| o.candidate.name == "wasm").unwrap();
        assert!(!wasm.selected);
        assert!(wasm.rejection_reason.as_deref().unwrap().contains("explicit-name-only"));
        let nspawn = report.candidates.iter().find(|o| o.candidate.name == "nspawn").unwrap();
        assert!(nspawn.selected, "auto must pick the auto-eligible backend, never the excluded one");
    }

    #[test]
    fn explain_agrees_with_select_registration_on_the_winner() {
        // The explain trace and the authoritative `select_registration` path must
        // agree on which backend `"auto"` yields, over the same candidate set.
        const A: BackendRegistration = BackendRegistration {
            name: "high-tier", priority: 100, auto_selectable: true,
            is_available: || true, construct: never_available,
        };
        const B: BackendRegistration = BackendRegistration {
            name: "low-tier", priority: 10, auto_selectable: true,
            is_available: || true, construct: never_available,
        };
        let regs = vec![&A, &B];
        let winner = select_registration("auto", &regs, |_| true).unwrap();

        let cands: Vec<BackendCandidate> =
            regs.iter().map(|r| BackendCandidate::from_registration(r)).collect();
        let report = explain_over(&cands);
        let explained = report
            .candidates
            .iter()
            .find(|o| o.selected)
            .map(|o| o.candidate.name)
            .unwrap();
        assert_eq!(explained, winner.name, "explain winner must match select_registration");
    }

    #[test]
    fn backend_candidate_capability_labels_emit_runtime_truth() {
        // The capability-source seam: a backend contributes `runtime=<name>` +
        // availability so a node record can aggregate it (#628, no drift).
        let c = candidate("kata", 100, true, false);
        let labels = c.capability_labels();
        assert!(labels.contains(&("runtime".to_owned(), "kata".to_owned())));
        assert!(labels.contains(&("runtime.kata.available".to_owned(), "false".to_owned())));
        assert!(c.capability_resources().is_empty());
    }
}
