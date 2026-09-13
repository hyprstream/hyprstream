//! Handlers for service management commands
//!
//! Provides lifecycle management for hyprstream services including
//! installation, upgrade, start/stop, and status display.
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::ffi::OsString;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::info;

/// Persist requested built-in templates into the same policy store loaded by
/// PolicyService, without starting a resolver or loading service credentials.
pub async fn handle_service_provision_policy_templates(
    models_dir: &Path,
    template_names: &[String],
) -> Result<()> {
    provision_policy_templates(
        models_dir,
        template_names,
        |_| Ok(()),
        |_| Ok(()),
        |_| Ok(()),
    )
    .await
}

async fn provision_policy_templates(
    models_dir: &Path,
    template_names: &[String],
    after_staged_save: impl FnOnce(&Path) -> Result<()>,
    after_snapshot: impl FnOnce(&Path) -> Result<()>,
    mut after_publication_write: impl FnMut(&Path) -> Result<()>,
) -> Result<()> {
    use crate::auth::{get_template, PolicyManager, PolicyTemplate};
    use anyhow::{bail, ensure};
    use std::collections::BTreeSet;

    ensure!(
        !template_names.is_empty(),
        "at least one policy template is required"
    );

    // Resolve and validate the complete request before PolicyManager::new can
    // create or migrate anything on disk.
    let mut unique = BTreeSet::new();
    let mut templates: Vec<&'static PolicyTemplate> = Vec::with_capacity(template_names.len());
    for name in template_names {
        ensure!(
            unique.insert(name.as_str()),
            "duplicate policy template: {name}"
        );
        let Some(template) = get_template(name) else {
            bail!("unknown policy template: {name}");
        };
        templates.push(template);
    }

    let registry_dir = models_dir.join(".registry");
    tokio::fs::create_dir_all(&registry_dir)
        .await
        .context("create registry directory for policy provisioning")?;
    let staging = tempfile::Builder::new()
        .prefix(".policy-provision-")
        .tempdir_in(&registry_dir)
        .context("create policy staging directory")?;
    let policies_dir = registry_dir.join("policies");
    let staged_policies_dir = staging.path().join("policies");
    tokio::fs::create_dir(&staged_policies_dir)
        .await
        .context("create staged policy directory")?;
    for name in ["model.conf", "policy.csv"] {
        let source = policies_dir.join(name);
        match tokio::fs::read(&source).await {
            Ok(content) => {
                crate::auth::write_policy_file(&staged_policies_dir.join(name), content)
                    .await
                    .with_context(|| format!("stage retained policy file {name}"))?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("read retained policy file {}", source.display()));
            }
        }
    }

    // PolicyManager's FileAdapter truncates files when it saves. Keep every
    // constructor migration and template save in this disposable directory.
    let manager = PolicyManager::new(&staged_policies_dir)
        .await
        .context("open staged policy store")?;
    for template in &templates {
        manager
            .apply_template(template)
            .await
            .with_context(|| format!("apply policy template '{}'", template.name))?;
    }

    let requested: BTreeSet<&str> = templates.iter().map(|template| template.name).collect();
    let public_staging: BTreeSet<&str> = ["public-inference", "public-read"].into_iter().collect();
    verify_requested_templates(&manager, &templates).await?;
    if requested == public_staging {
        verify_public_staging_policy(&manager).await?;
    }
    let intended_state = complete_policy_state(&manager).await;
    drop(manager);

    let staged_policy_path = staged_policies_dir.join("policy.csv");
    after_staged_save(&staged_policy_path)?;

    // Capture once, validate these immutable bytes, and publish these same
    // buffers. A late write to staging cannot change the selected replacement.
    let staged_model = tokio::fs::read(staged_policies_dir.join("model.conf"))
        .await
        .context("capture staged policy model")?;
    let staged_policy = tokio::fs::read(&staged_policy_path)
        .await
        .context("capture staged policy")?;
    anyhow::ensure!(
        parse_complete_policy_state(&staged_model, &staged_policy).await? == intended_state,
        "captured staged policy does not match the complete intended policy state"
    );
    after_snapshot(&staged_policy_path)?;
    tokio::fs::create_dir_all(&policies_dir)
        .await
        .context("create live policy directory")?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        tokio::fs::set_permissions(&policies_dir, std::fs::Permissions::from_mode(0o750))
            .await
            .context("set live policy directory permissions")?;
    }

    // The model is compatible with both the retained and replacement policy.
    // Publish policy.csv last: its rename is the authority commit point.
    atomic_publish_policy_file(
        &policies_dir.join("model.conf"),
        &staged_model,
        &mut after_publication_write,
    )
        .context("publish verified policy model")?;
    atomic_publish_policy_file(
        &policies_dir.join("policy.csv"),
        &staged_policy,
        &mut after_publication_write,
    )
        .context("publish verified policy")?;

    let published = PolicyManager::new(&policies_dir)
        .await
        .context("reopen published policy store")?;
    verify_requested_templates(&published, &templates).await?;
    if requested == public_staging {
        verify_public_staging_policy(&published).await?;
    }

    println!(
        "verified {} policy template(s): {}",
        templates.len(),
        template_names.join(",")
    );
    Ok(())
}

#[derive(Debug, Eq, PartialEq)]
struct CompletePolicyState {
    policies: std::collections::BTreeSet<Vec<String>>,
    groupings: std::collections::BTreeSet<Vec<String>>,
    domain_groupings: std::collections::BTreeSet<Vec<String>>,
}

async fn complete_policy_state(manager: &crate::auth::PolicyManager) -> CompletePolicyState {
    CompletePolicyState {
        policies: manager.get_policy().await.into_iter().collect(),
        groupings: manager.get_grouping_policy().await.into_iter().collect(),
        domain_groupings: manager
            .get_domain_grouping_policy()
            .await
            .into_iter()
            .collect(),
    }
}

async fn parse_complete_policy_state(
    model: &[u8],
    policy: &[u8],
) -> Result<CompletePolicyState> {
    use casbin::{CoreApi as _, DefaultModel, Enforcer, MgmtApi as _, StringAdapter};

    let model = std::str::from_utf8(model).context("captured policy model is not UTF-8")?;
    let policy = std::str::from_utf8(policy).context("captured policy is not UTF-8")?;
    let model = DefaultModel::from_str(model)
        .await
        .context("parse captured policy model")?;
    let enforcer = Enforcer::new(model, StringAdapter::new(policy))
        .await
        .context("parse captured policy")?;
    Ok(CompletePolicyState {
        policies: enforcer.get_policy().into_iter().collect(),
        groupings: enforcer.get_grouping_policy().into_iter().collect(),
        domain_groupings: enforcer
            .get_named_grouping_policy("g2")
            .into_iter()
            .collect(),
    })
}

fn atomic_publish_policy_file(
    path: &Path,
    content: &[u8],
    after_write: &mut impl FnMut(&Path) -> Result<()>,
) -> Result<()> {
    let parent = path
        .parent()
        .context("policy publication path has no parent")?;
    let mut staged = tempfile::NamedTempFile::new_in(parent)
        .with_context(|| format!("create publication file for {}", path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        staged
            .as_file()
            .set_permissions(std::fs::Permissions::from_mode(0o640))
            .with_context(|| format!("set publication permissions for {}", path.display()))?;
    }
    staged
        .write_all(content)
        .with_context(|| format!("write publication file for {}", path.display()))?;
    after_write(path)?;
    staged
        .as_file_mut()
        .flush()
        .with_context(|| format!("flush publication file for {}", path.display()))?;
    staged
        .as_file()
        .sync_all()
        .with_context(|| format!("sync publication file for {}", path.display()))?;
    staged
        .persist(path)
        .map_err(|error| error.error)
        .with_context(|| format!("publish policy file {}", path.display()))?;
    #[cfg(unix)]
    std::fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("sync policy directory {}", parent.display()))?;
    Ok(())
}

async fn verify_requested_templates(
    manager: &crate::auth::PolicyManager,
    templates: &[&crate::auth::PolicyTemplate],
) -> Result<()> {
    use anyhow::ensure;
    use std::collections::BTreeSet;

    let policies = manager.get_policy().await;
    let groupings = manager.get_grouping_policy().await;
    let domain_groupings = manager.get_domain_grouping_policy().await;
    for template in templates {
        let expanded = template.expanded_policies();
        let tenant_domains: BTreeSet<&str> = expanded
            .iter()
            .map(|policy| policy.domain)
            .filter(|domain| *domain != "*")
            .collect();
        for expected in expanded {
            ensure!(
                policies.contains(&expected.to_vec()),
                "policy template '{}' did not persist its canonical rule",
                template.name
            );
        }
        if let Some(expected_groupings) = template.groupings {
            for expected in expected_groupings {
                if let Some(domain) = tenant_domains.first() {
                    ensure!(
                        domain_groupings.contains(&vec![
                            expected.user.to_owned(),
                            expected.role.to_owned(),
                            (*domain).to_owned(),
                        ]),
                        "policy template '{}' did not persist its canonical domain grouping",
                        template.name
                    );
                } else {
                    ensure!(
                        groupings.contains(&expected.to_vec()),
                        "policy template '{}' did not persist its canonical grouping",
                        template.name
                    );
                }
            }
        }
    }
    Ok(())
}

async fn verify_public_staging_policy(manager: &crate::auth::PolicyManager) -> Result<()> {
    use anyhow::ensure;

    let policies = manager.get_policy().await;
    let mut expected = Vec::new();
    for name in ["public-inference", "public-read"] {
        let template = crate::auth::get_template(name)
            .with_context(|| format!("compiled-in public staging template missing: {name}"))?;
        expected.extend(
            template
                .expanded_policies()
                .into_iter()
                .map(|rule| rule.to_vec()),
        );
    }
    ensure!(
        expected.len() == 3,
        "public staging templates must define exactly three rules"
    );
    for rule in &expected {
        ensure!(
            policies
                .iter()
                .filter(|candidate| candidate.as_slice() == rule.as_slice())
                .count()
                == 1,
            "public staging policy must persist each of its three canonical rules exactly once"
        );
    }

    let checks = [
        ("model:policy-bootstrap-probe", "infer.generate"),
        ("model:policy-bootstrap-probe", "query.status"),
        ("registry:policy-bootstrap-probe", "query.status"),
    ];
    for (resource, action) in checks {
        ensure!(
            manager
                .check_with_domain("anonymous", "*", resource, action)
                .await,
            "public staging policy is not effective for {action} on {resource}"
        );
    }
    ensure!(
        !manager
            .check_with_domain(
                "anonymous",
                "*",
                "model:policy-bootstrap-probe",
                "ttt.writeback",
            )
            .await,
        "public staging policy must not grant anonymous ttt.writeback"
    );
    Ok(())
}

/// Handle `service install` - Idempotent setup and optional restart
///
/// 1. Run repair checks (dirs, registry, policy, signing key, git identity)
/// 2. Install command alias (~/.local/bin/hyprstream)
/// 3. Install/update systemd units (if systemd available)
/// 4. If `start`: stop → start all target services
pub async fn handle_service_install(
    models_dir: &Path,
    config_services: &[String],
    services_filter: Option<Vec<String>>,
    start: bool,
    enable: bool,
    target: hyprstream_service::ServiceTarget,
    verbose: bool,
    explicit_config: Option<&Path>,
    iroh_required: bool,
) -> Result<()> {
    let target_services = services_filter.unwrap_or_else(|| config_services.to_vec());

    println!("Installing hyprstream...\n");

    // 1. Run repair checks to bootstrap the environment
    run_repair_checks(models_dir, verbose).await?;
    println!();

    // 2. Install command alias to user's executable directory
    println!("  Installing command...");
    match InstallPlan::prepare() {
        Ok(plan) => {
            println!("    Source: {} ({})", plan.source.display(), plan.type_label());
            match plan.execute() {
                Ok(result) => {
                    println!("    {} ({})", result.version_dir.display(), result.type_label());
                    println!(
                        "    {} -> ...",
                        result.bin_dir.join("hyprstream.appimage").display()
                    );
                    println!(
                        "    {} -> hyprstream.appimage",
                        result.bin_dir.join("hyprstream").display()
                    );
                    if !result.updated_profiles.is_empty() {
                        println!("    PATH updated: {}", result.updated_profiles.join(" "));
                    }
                }
                Err(e) => println!("    install failed ({})", e),
            }
        }
        Err(e) => println!("    skipped ({})", e),
    }
    println!();

    // 3. Install/update systemd units if available
    if hyprstream_rpc::has_systemd() {
        // Installed units run their own stored configuration (fixed ExecStart,
        // no config provenance, raw dependency order). Installing, enabling,
        // or starting such a unit under an explicit --config selector or the
        // required-native profile would silently run a different identity
        // than this process loaded. Refuse before any unit mutation; the
        // direct launch path is the supported route.
        if explicit_config.is_some() || iroh_required {
            anyhow::bail!(
                "installed hyprstream service units cannot receive an explicit --config \
                 selector or the required-native profile, so this command cannot install \
                 or start units for that configuration. Launch provisioned services \
                 directly instead, e.g. `hyprstream{} service start <service> --daemon`. \
                 (Plain `hyprstream service install` without the custom selector or \
                 required profile still installs default-configuration units.)",
                match explicit_config {
                    Some(path) => format!(" --config {}", path.display()),
                    None => String::new(),
                }
            );
        }

        let manager = hyprstream_service::detect_service_manager_with_mode(target).await?;

        // Encrypt secrets into the systemd user credstore before generating units.
        // This must happen before manager.install() so that install() can see the
        // .cred files when deciding whether to emit ImportCredential= directives.
        #[cfg(feature = "systemd")]
        {
            let secrets_dir = crate::config::HyprConfig::load()
                .ok()
                .and_then(|c| crate::config::HyprConfig::resolve_secrets_dir_for(Some(&c)).ok());
            hyprstream_service::encrypt_credentials_if_available(secrets_dir.as_deref());
        }

        // If --start, stop all target services first so they pick up changes.
        // Stop is legitimately idempotent here (already-stopped units are the
        // common case on reinstall), so failures stay non-fatal.
        if start {
            println!("  Stopping services...");
            for service in &target_services {
                let _ = manager.stop(service).await;
            }
        }

        println!("  Installing systemd units...");
        for service in &target_services {
            print!("    \u{25CB} {}... ", service);
            manager
                .install(service)
                .await
                .map_err(|e| anyhow::anyhow!("installing unit for {service}: {e}"))?;
            println!("\u{2713}");
        }

        // If --enable, register units for autostart at boot
        if enable {
            println!("  Enabling services for autostart...");
            for service in &target_services {
                print!("    \u{25CB} {}... ", service);
                manager
                    .enable(service)
                    .await
                    .map_err(|e| anyhow::anyhow!("enabling unit for {service}: {e}"))?;
                println!("\u{2713}");
            }
        }

        // 4. If --start, start all target services. A queued start request is
        // not success: require the unit to reach the active state (Type=notify)
        // within the bounded startup budget (#1585).
        if start {
            println!("  Starting services...");
            start_units_to_active(&*manager, &target_services, CHILD_READINESS_TIMEOUT).await?;
        }
    } else if start {
        // No systemd: the same factored direct launch path as `service start`
        // — profile-aware ordering, config forwarding, notification readiness.
        println!("  Starting services (standalone)...\n");

        let exe = hyprstream_rpc::paths::installed_executable_path()
            .unwrap_or_else(|| hyprstream_rpc::paths::executable_path().unwrap_or_default());

        let spawner = hyprstream_service::ProcessSpawner::standalone();

        launch_direct_children(
            &target_services,
            iroh_required,
            explicit_config,
            &exe,
            &spawner,
        )
        .await?;
    }

    println!("\n\u{2713} Install complete");
    if !start {
        println!();
        println!("Next steps:");
        println!("  1. Open a new shell to use the 'hyprstream' command");
        println!("  2. Start services with: hyprstream service start");
        println!("     Or reinstall with:   hyprstream service install --start");
    }
    Ok(())
}

/// Handle `service uninstall` - Stop and remove units
pub async fn handle_service_uninstall(
    config_services: &[String],
    services_filter: Option<Vec<String>>,
) -> Result<()> {
    let target_services = services_filter.unwrap_or_else(|| config_services.to_vec());

    if !hyprstream_rpc::has_systemd() {
        println!("Systemd not available. No units to uninstall.");
        return Ok(());
    }

    let manager = hyprstream_service::detect_service_manager().await?;

    println!("Uninstalling hyprstream services...\n");

    for service in &target_services {
        print!("  \u{25CB} {}... ", service);

        // Stop first, then uninstall
        let _ = manager.stop(service).await;

        match manager.uninstall(service).await {
            Ok(_) => println!("\u{2713}"),
            Err(e) => println!("\u{2717} {}", e),
        }
    }

    // Reload daemon
    manager.reload().await?;

    println!("\n\u{2713} Uninstall complete");
    Ok(())
}

/// Budget a spawned child gets to report genuine readiness (#1585).
const CHILD_READINESS_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Build one child invocation for the direct launch path (#1585).
///
/// Required-native children run `hyprstream [--config PATH] service start NAME
/// --foreground` — no `--ipc`, no local endpoint fallback; their readiness is
/// the child's own notification boundary, never a UDS socket. Compatibility
/// children keep the historical `--ipc` shape. The explicit config path (when
/// the operator supplied one) is forwarded so the child loads, validates, and
/// pins the same configuration the launcher did; it is canonical absolute, so
/// it stays meaningful regardless of the child's working directory, and it is
/// one argv element, so spaces need no quoting. Config contents, keys, and
/// JWTs never enter argv or the environment.
pub fn direct_child_process_config(
    service: &str,
    iroh_required: bool,
    explicit_config: Option<&Path>,
    exe: &Path,
) -> Result<hyprstream_service::ProcessConfig> {
    let mut args: Vec<OsString> = Vec::new();
    if let Some(config_path) = explicit_config {
        args.push(OsString::from("--config"));
        args.push(config_path.as_os_str().to_owned());
    }
    args.push(OsString::from("service"));
    args.push(OsString::from("start"));
    args.push(OsString::from(service));
    args.push(OsString::from("--foreground"));
    if !iroh_required {
        args.push(OsString::from("--ipc"));
    }

    let mut config = hyprstream_service::ProcessConfig::new(service, exe);
    config.args = args;
    if iroh_required {
        // The child's own authenticated startup boundary is the readiness
        // signal (sd_notify READY after Iroh bind and first publication).
        config.readiness =
            hyprstream_service::ProcessReadiness::Notify { timeout: CHILD_READINESS_TIMEOUT };
    }
    Ok(config)
}

/// Launch children directly in profile-aware dependency order (#1585).
///
/// Shared by `service start --daemon`, no-systemd `service start`, and the
/// no-systemd `service install --start` fallback, so every direct launch gets
/// the same ordering, config forwarding, and readiness contract.
///
/// Required-native: every child must report genuine readiness before the next
/// child (and stage) starts; any spawn, early-exit, or timeout failure stops
/// already-started children in reverse start order and returns `Err` — the
/// caller never prints success for a partial launch. Compatibility keeps the
/// historical best-effort behavior (per-service error lines, UDS-presence
/// stage barrier with warn-and-continue) so existing deployments are
/// unchanged.
pub async fn launch_direct_children(
    targets: &[String],
    iroh_required: bool,
    explicit_config: Option<&Path>,
    exe: &Path,
    spawner: &hyprstream_service::ProcessSpawner,
) -> Result<()> {
    if iroh_required {
        for service in targets {
            anyhow::ensure!(
                hyprstream_service::get_factory(service).is_some(),
                "unknown native service: {service}"
            );
        }
    }
    let stages = hyprstream_service::startup_stages_for_profile(targets, iroh_required);
    // Flatten the ordered stages into the serial launch plan (same order the
    // loop below would spawn in), so the launch core is injectably testable
    // without the real service binary.
    let mut plans: Vec<Vec<(String, hyprstream_service::ProcessConfig)>> = Vec::new();
    for stage in &stages {
        let mut stage_plans = Vec::new();
        for service in stage {
            stage_plans.push((
                service.clone(),
                direct_child_process_config(service, iroh_required, explicit_config, exe)?,
            ));
        }
        plans.push(stage_plans);
    }
    launch_planned_children(plans, iroh_required, spawner).await
}

/// Cancellation-safe ownership of the children of a Required launch
/// transaction (#1585).
///
/// Every child that spawns successfully — including children that already
/// reached READY and were adopted by the backend — is recorded here until the
/// WHOLE launch commits. If the orchestration future is cancelled (dropped at
/// any await: a later child's readiness, or a rollback stop), `Drop` runs the
/// synchronous bounded tracked-child stop
/// ([`hyprstream_service::ProcessSpawner::stop_tracked_child_sync`]) over the
/// children it still owns, newest first: the backend stops them through the
/// retained `Child` handles (never a blind by-PID signal) and removes their
/// PID artifacts. That cleanup is plain synchronous code on the drop path —
/// it does not depend on any async cleanup future surviving runtime shutdown.
/// On whole-launch success [`Self::commit`] disarms the guard so the adopted
/// daemons intentionally keep running, stoppable as before. Compatibility
/// launches are never armed: their historical behavior, including a
/// cancelled launch's historical leak, is unchanged.
struct RequiredLaunchGuard {
    started: Vec<(String, hyprstream_service::SpawnedProcess)>,
    /// Children whose async rollback stop FAILED: still fully owned by the
    /// guard (a local collection would recreate the ownership-transfer hole
    /// across the remaining rollback awaits), retried by the synchronous
    /// bounded pass.
    failed: Vec<(String, hyprstream_service::SpawnedProcess)>,
    /// Clone of the launching spawner: the tracked `Child` handles live in
    /// its backend, so `Drop` cleanup must go through THIS backend to keep
    /// the stronger tracked ownership.
    spawner: hyprstream_service::ProcessSpawner,
    armed: bool,
    committed: bool,
}

impl RequiredLaunchGuard {
    /// Arm only for Required launches; Compatibility ownership stays exactly
    /// as it was before #1585's transaction guard.
    fn arm(iroh_required: bool, spawner: &hyprstream_service::ProcessSpawner) -> Self {
        Self {
            started: Vec::new(),
            failed: Vec::new(),
            spawner: spawner.clone(),
            armed: iroh_required,
            committed: false,
        }
    }

    /// Record an adopted child under transaction ownership.
    fn adopt(&mut self, service: String, process: hyprstream_service::SpawnedProcess) {
        if self.armed {
            self.started.push((service, process));
        }
    }

    /// The newest owned child, without surrendering ownership: the async stop
    /// below works on cloned metadata, so a cancellation mid-await leaves the
    /// original owned and the guard's `Drop` responsible for it.
    fn newest(&self) -> Option<&(String, hyprstream_service::SpawnedProcess)> {
        self.started.last()
    }

    /// Surrender ownership of the newest child — ONLY after a CONFIRMED
    /// successful stop.
    fn release_newest(&mut self) {
        self.started.pop();
    }

    /// Demote the newest child after a FAILED async stop: it stays
    /// guard-owned (moved into `failed`) so every later rollback await, and
    /// cancellation during any of them, still leaves it under `Drop` cleanup.
    fn demote_newest_to_failed(&mut self) {
        if let Some(entry) = self.started.pop() {
            self.failed.push(entry);
        }
    }

    /// Reverse-order rollback with cancellation-safe ownership: a stop that
    /// errors demotes the child to guard-owned `failed` state — never a
    /// local unguarded collection — while confirmed successes are released.
    /// The `stop` seam is the production async rollback stop (the tracked
    /// path terminates via the retained handle with SIGKILL and reaps within
    /// a bounded budget — it is not a graceful TERM shutdown); tests inject
    /// a controllable boundary.
    async fn rollback_owned_children<S, F>(&mut self, mut stop: S)
    where
        S: FnMut(hyprstream_service::SpawnedProcess) -> F,
        F: std::future::Future<Output = anyhow::Result<()>>,
    {
        while let Some((service, process)) = self.newest().cloned() {
            print!("  \u{25CB} stopping {} after failed launch... ", service);
            match stop(process).await {
                Ok(()) => {
                    println!("\u{2713}");
                    self.release_newest();
                }
                Err(e) => {
                    println!("\u{2717} {}", e);
                    self.demote_newest_to_failed();
                }
            }
        }
    }

    /// Synchronously stop everything still owned — unreleased children and
    /// failed-stop demotions, newest first — and disarm. Returns one
    /// residual-report string per child whose bounded stop did not confirm
    /// termination or whose PID-artifact cleanup failed (the artifact is then
    /// deliberately retained — it may still name a live process).
    fn stop_all_sync_and_disarm(&mut self) -> Vec<String> {
        let mut residuals = Vec::new();
        // Deterministic reverse-start-order contract across BOTH ownership
        // collections: failed demotions were appended newest-first during the
        // async rollback, so they are revisited in insertion order first,
        // then the still-unattempted started entries in reverse order.
        let owned = self
            .failed
            .iter()
            .chain(self.started.iter().rev())
            .collect::<Vec<_>>();
        for (service, process) in owned {
            print!("  \u{25CB} stopping {} (synchronous pass)... ", service);
            match self.spawner.stop_tracked_child_sync(process) {
                Ok(()) => println!("\u{2713}"),
                Err(e) => {
                    println!("\u{2717} {}", e);
                    residuals.push(format!("{service}: {e}"));
                    tracing::error!(
                        service = %service,
                        error = %e,
                        "aborted required launch left residual child state; PID artifact retained"
                    );
                }
            }
        }
        self.started.clear();
        self.failed.clear();
        self.committed = true;
        residuals
    }

    /// Mark the whole launch committed: adopted daemons stay intentionally
    /// alive and the `Drop` path becomes a no-op.
    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for RequiredLaunchGuard {
    fn drop(&mut self) {
        if self.committed
            || !self.armed
            || (self.started.is_empty() && self.failed.is_empty())
        {
            return;
        }
        // Cancellation (task abort at any await — later-child readiness or a
        // rollback stop) lands here: the synchronous bounded pass is the
        // cleanup owner of last resort for everything not yet released,
        // including children demoted to `failed` by earlier stop errors.
        self.stop_all_sync_and_disarm();
    }
}

/// Serially launch pre-built child plans (stage-ordered) with the
/// required-native readiness contract and reverse-order rollback. Narrow seam
/// (#1585): plan building is separate so causal tests can inject concrete
/// children without the real service binary.
///
/// Required-native cancellation safety: started children stay owned by the
/// [`RequiredLaunchGuard`] until the whole launch commits, so a cancelled
/// future still cleans them through the backend's retained child handles; an
/// explicit failure runs the async rollback stop first (tracked path
/// terminates via SIGKILL through the retained handle — not a graceful TERM
/// shutdown), with the synchronous tracked-child pass as the
/// accurate-residual fallback.
async fn launch_planned_children(
    plans: Vec<Vec<(String, hyprstream_service::ProcessConfig)>>,
    iroh_required: bool,
    spawner: &hyprstream_service::ProcessSpawner,
) -> Result<()> {
    let mut guard = RequiredLaunchGuard::arm(iroh_required, spawner);
    let mut launch_error: Option<anyhow::Error> = None;

    'stages: for stage in &plans {
        for (service, config) in stage {
            print!("  \u{25CB} {}... ", service);

            match spawner.spawn(config.clone()).await {
                Ok(process) => {
                    info!("Spawned {} service: {:?}", service, process.kind);
                    println!("\u{2713} (pid {:?})", process.pid());
                    guard.adopt(service.clone(), process);
                }
                Err(e) => {
                    println!("\u{2717} {}", e);
                    if iroh_required {
                        // A required-native failure aborts the WHOLE launch:
                        // later stages depend on earlier ones, so none of
                        // them may spawn before the rollback runs.
                        launch_error = Some(e.into());
                        break 'stages;
                    }
                }
            }
        }

        // Compatibility stage barrier: wait for IPC sockets before the next
        // stage. Required-native readiness was already enforced per child
        // above; it never depends on a service UDS socket.
        if !iroh_required {
            let runtime_dir = hyprstream_rpc::paths::runtime_dir();
            for (service, _) in stage {
                let sock = runtime_dir.join(format!("{service}.sock"));
                let deadline = std::time::Instant::now() + CHILD_READINESS_TIMEOUT;
                while !sock.exists() && std::time::Instant::now() < deadline {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
                if !sock.exists() {
                    tracing::warn!("Timeout waiting for {service} socket; continuing");
                }
            }
        }
    }

    if let Some(error) = launch_error {
        if iroh_required {
            // Deterministic rollback: stop what we started, newest first; each
            // stop removes that child's PID artifact. Every child stays
            // guard-owned until its stop CONFIRMS — a failed stop is demoted
            // to guard-owned `failed` state, so a cancellation during ANY of
            // these awaits (or the synchronous pass) still cleans everything
            // unconfirmed through the guard's `Drop`.
            guard
                .rollback_owned_children(|process| async move {
                    spawner.stop(&process).await.map_err(anyhow::Error::from)
                })
                .await;
            // Synchronous bounded pass over guard-owned state: retries failed
            // stops and sweeps unreleased children. Only children whose
            // bounded stop did not confirm termination (or whose PID-artifact
            // cleanup failed) are residual — reported honestly, never
            // silently dropped.
            let residuals = guard.stop_all_sync_and_disarm();
            if !residuals.is_empty() {
                return Err(error.context(format!(
                    "launch rollback left residual state: {}",
                    residuals.join("; ")
                )));
            }
        }
        return Err(error);
    }
    // A child may exit after its readiness notification while a later stage
    // is still starting. Recheck every adopted child immediately before the
    // transaction commits so we never report success for a deployment whose
    // earlier service has already died.
    if iroh_required {
        let liveness_error = {
            let mut failure = None;
            for (service, process) in &guard.started {
                match spawner.is_running(process).await {
                    Ok(true) => {}
                    Ok(false) => {
                        failure = Some(anyhow::anyhow!(
                            "required-native child '{service}' exited before launch commit"
                        ));
                        break;
                    }
                    Err(e) => {
                        failure = Some(anyhow::anyhow!(
                            "could not verify required-native child '{service}' before launch commit: {e}"
                        ));
                        break;
                    }
                }
            }
            failure
        };
        if let Some(error) = liveness_error {
            guard
                .rollback_owned_children(|process| async move {
                    spawner.stop(&process).await.map_err(anyhow::Error::from)
                })
                .await;
            let residuals = guard.stop_all_sync_and_disarm();
            if !residuals.is_empty() {
                return Err(error.context(format!(
                    "launch rollback left residual state: {}",
                    residuals.join("; ")
                )));
            }
            return Err(error);
        }
    }
    guard.commit();
    Ok(())
}

/// Handle `service start` (non-foreground) - Start via systemd or spawn
///
/// `explicit_config` is the operator's canonical absolute `--config` selector
/// (CLI value or the `HYPRSTREAM_CONFIG` env selector); `iroh_required` is the
/// loaded config's native-network profile. Required-native or custom-config
/// starts never select an installed systemd unit — those units encode their
/// own stored configuration and cannot receive this process's configuration —
/// and are routed to the direct launch path instead.
pub async fn handle_service_start(
    config_services: &[String],
    name: Option<String>,
    daemon: bool,
    explicit_config: Option<&Path>,
    iroh_required: bool,
) -> Result<()> {
    let target_services: Vec<String> = if let Some(name) = name {
        vec![name]
    } else {
        config_services.to_vec()
    };

    // MAC genesis coverage gate: logged by the shared `ServiceAction::Start`
    // dispatch in `bin/main.rs` (before the foreground/systemd branching), so
    // every startup mode — including foreground boots that bypass this
    // handler — emits the report exactly once per process.

    // Use systemd if available and --daemon not specified
    if hyprstream_rpc::has_systemd() && !daemon {
        if explicit_config.is_some() || iroh_required {
            anyhow::bail!(
                "installed hyprstream service units run their own stored configuration; \
                 an explicit --config selector or the required-native profile cannot be \
                 applied through them. Launch provisioned services directly instead, \
                 e.g. `hyprstream{} service start <service> --daemon`.",
                match explicit_config {
                    Some(path) => format!(" --config {}", path.display()),
                    None => String::new(),
                }
            );
        }

        let manager = hyprstream_service::detect_service_manager().await?;

        println!("Starting services (systemd)...\n");

        start_units_to_active(&*manager, &target_services, CHILD_READINESS_TIMEOUT).await?;
    } else {
        // Direct launch: profile-aware ordering, config forwarding, and the
        // per-child notification readiness contract.
        println!("Starting services (standalone)...\n");

        let spawner = hyprstream_service::ProcessSpawner::standalone();
        let exe = hyprstream_rpc::paths::executable_path()?;
        launch_direct_children(
            &target_services,
            iroh_required,
            explicit_config,
            &exe,
            &spawner,
        )
        .await?;
    }

    println!("\n\u{2713} Start complete");
    Ok(())
}


/// Start installed units and require each to reach the active state within
/// `active_timeout` before reporting success (#1585).
///
/// The permitted Compatibility-only existing-unit path: mutation failures
/// propagate (a pre-existing active unit must never mask a failed start), and
/// a queued start request is not completion — the unit is Type=notify.
/// Injectable over [`hyprstream_service::ServiceManager`] for causal tests.
async fn start_units_to_active(
    manager: &dyn hyprstream_service::ServiceManager,
    targets: &[String],
    active_timeout: std::time::Duration,
) -> Result<()> {
    // Submit every start request concurrently before waiting on any unit.
    // Compatibility services may probe Policy during their own startup;
    // waiting for an earlier Event unit to become active before submitting
    // Policy would deadlock when the units were stopped. Systemd's
    // StartUnit call waits for its job result, so concurrent requests are
    // required to let Policy and Event make progress together. Mutation
    // failures still propagate before any active-state success is reported.
    for service in targets {
        print!("  \u{25CB} {}... ", service);
    }
    futures::future::try_join_all(targets.iter().map(|service| async move {
        manager
            .start(service)
            .await
            .map_err(|e| anyhow::anyhow!("starting {service}: {e}"))
    }))
    .await?;

    for service in targets {
        let deadline = std::time::Instant::now() + active_timeout;
        loop {
            match manager.is_active(service).await {
                Ok(true) => break,
                Ok(false) => {
                    if std::time::Instant::now() >= deadline {
                        anyhow::bail!(
                            "service {service} unit did not reach the active state within {}s",
                            active_timeout.as_secs()
                        );
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
                Err(e) => return Err(e),
            }
        }
        println!("\u{2713}");
    }
    Ok(())
}

/// Handle `service stop` - Stop services
///
/// Attempts to stop services via both systemd and PID files.
pub async fn handle_service_stop(
    config_services: &[String],
    name: Option<String>,
) -> Result<()> {
    let target_services: Vec<String> = if let Some(name) = name {
        vec![name]
    } else {
        config_services.to_vec()
    };

    println!("Stopping services...\n");

    // Try systemd if available
    if hyprstream_rpc::has_systemd() {
        let manager = hyprstream_service::detect_service_manager().await?;

        for service in &target_services {
            print!("  \u{25CB} {} (systemd)... ", service);
            match manager.stop(service).await {
                Ok(_) => println!("\u{2713}"),
                Err(_) => println!("-"),
            }
        }
    }

    // Stop daemon processes via PID files
    let runtime_dir = hyprstream_rpc::paths::runtime_dir();
    for service in &target_services {
        let pid_file = runtime_dir.join(format!("{}.pid", service));

        if pid_file.exists() {
            print!("  \u{25CB} {} (daemon)... ", service);
            if let Ok(pid_str) = std::fs::read_to_string(&pid_file) {
                if let Ok(pid) = pid_str.trim().parse::<i32>() {
                    let nix_pid = nix::unistd::Pid::from_raw(pid);
                    match nix::sys::signal::kill(nix_pid, nix::sys::signal::Signal::SIGTERM) {
                        Ok(_) => {
                            let _ = std::fs::remove_file(&pid_file);
                            println!("\u{2713}");
                        }
                        Err(nix::errno::Errno::ESRCH) => {
                            let _ = std::fs::remove_file(&pid_file);
                            println!("- (stale)");
                        }
                        Err(e) => println!("\u{2717} {}", e),
                    }
                } else {
                    println!("\u{2717} invalid pid");
                }
            } else {
                println!("\u{2717} read error");
            }
        }
    }

    println!("\n\u{2713} Stop complete");
    Ok(())
}

/// Handle `service status` - Show current state
pub async fn handle_service_status(
    config_services: &[String],
    verbose: bool,
) -> Result<()> {
    println!("Hyprstream Service Status");
    println!("{}", "=".repeat(50));

    // Execution context
    println!("\nExecution Context:");
    if let Ok(appimage) = std::env::var("APPIMAGE") {
        println!("  Executable:    {} (AppImage)", appimage);
    } else if let Ok(exe) = std::env::current_exe() {
        println!("  Executable:    {}", exe.display());
    }

    // Service manager type
    println!("\nService Manager:");
    if hyprstream_rpc::has_systemd() {
        println!("  Type:          systemd (user session)");

        if let Some(config_dir) = dirs::config_dir() {
            let units_dir = config_dir.join("systemd/user");
            println!("  Units dir:     {}", units_dir.display());
        }
    } else {
        println!("  Type:          standalone (process spawner)");
    }

    // Service status - check both systemd and daemon PID files
    println!("\nService Status:");
    println!("  {:<15} {:<10} MODE", "SERVICE", "STATUS");
    println!("  {}", "-".repeat(45));

    let runtime_dir = hyprstream_rpc::paths::runtime_dir();
    let manager = if hyprstream_rpc::has_systemd() {
        Some(hyprstream_service::detect_service_manager().await?)
    } else {
        None
    };

    for service in config_services {
        // Check systemd status
        let systemd_running = if let Some(ref mgr) = manager {
            mgr.is_active(service).await.unwrap_or(false)
        } else {
            false
        };

        // Check daemon PID file
        let daemon_running = {
            let pid_file = runtime_dir.join(format!("{}.pid", service));
            if pid_file.exists() {
                if let Ok(pid_str) = std::fs::read_to_string(&pid_file) {
                    if let Ok(pid) = pid_str.trim().parse::<i32>() {
                        let nix_pid = nix::unistd::Pid::from_raw(pid);
                        // Signal 0 checks if process exists
                        matches!(
                            nix::sys::signal::kill(nix_pid, None),
                            Ok(()) | Err(nix::errno::Errno::EPERM)
                        )
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        };

        let (icon, status, mode) = match (systemd_running, daemon_running) {
            (true, true) => ("\u{2713}", "running", "systemd+daemon"),
            (true, false) => ("\u{2713}", "running", "systemd"),
            (false, true) => ("\u{2713}", "running", "daemon"),
            (false, false) => ("\u{25CB}", "stopped", ""),
        };
        println!("  {:<15} {} {:<10} {}", service, icon, status, mode);
    }

    // Verbose: show unit file contents
    if verbose && hyprstream_rpc::has_systemd() {
        println!("\n{}", "=".repeat(50));
        println!("Unit File Contents:\n");

        if let Some(config_dir) = dirs::config_dir() {
            let units_dir = config_dir.join("systemd/user");
            for service in config_services {
                let unit_path = units_dir.join(format!("hyprstream-{}.service", service));
                if unit_path.exists() {
                    println!("--- {} ---", unit_path.display());
                    if let Ok(content) = std::fs::read_to_string(&unit_path) {
                        println!("{}", content);
                    }
                }
            }
        }
    }

    Ok(())
}

/// Run repair checks: directories, registry, policy, signing key, git identity.
///
/// Extracted from the old `handle_service_repair` so it can be called as part
/// of `handle_service_install` without the surrounding summary chrome.
pub async fn run_repair_checks(
    models_dir: &Path,
    verbose: bool,
) -> Result<()> {
    use crate::auth::PolicyManager;
    use crate::cli::policy_handlers::load_or_generate_signing_key;

    println!("  Repair checks\n");

    let mut all_passed = true;
    let mut warnings = Vec::new();

    // 1. Directories
    {
        let label = "Directories";
        let registry_path = models_dir.join(".registry");
        let policies_dir = registry_path.join("policies");
        let keys_dir = registry_path.join("keys");

        let dirs_to_create = [
            models_dir,
            registry_path.as_path(),
            policies_dir.as_path(),
            keys_dir.as_path(),
        ];

        let mut fixed = false;
        for dir in &dirs_to_create {
            if !dir.exists() {
                std::fs::create_dir_all(dir)
                    .with_context(|| format!("Failed to create directory: {}", dir.display()))?;
                fixed = true;
            }
        }

        if fixed {
            print_check(label, CheckStatus::Fixed, &format!("created missing directories under {}", models_dir.display()));
        } else {
            print_check(label, CheckStatus::Ok, &format!("{}", models_dir.display()));
        }

        if verbose {
            for dir in &dirs_to_create {
                println!("      {}", dir.display());
            }
        }
    }

    // 2. Registry initialized
    {
        let label = "Registry";
        let git_dir = models_dir.join(".registry").join(".git");

        if git_dir.exists() {
            // Verify it's a valid git repo
            match git2::Repository::open(models_dir.join(".registry")) {
                Ok(repo) => {
                    let count = match repo.revwalk() {
                        Ok(mut walk) => {
                            let _ = walk.push_head();
                            walk.count()
                        }
                        Err(_) => 0,
                    };
                    print_check(label, CheckStatus::Ok, &format!(".registry initialized ({count} commits)"));
                }
                Err(e) => {
                    print_check(label, CheckStatus::Fail, &format!(".registry git repo corrupt: {e}"));
                    all_passed = false;
                }
            }
        } else {
            // Initialize via Git2DB
            match git2db::Git2DB::open(models_dir).await {
                Ok(_) => {
                    print_check(label, CheckStatus::Fixed, ".registry initialized via Git2DB");
                }
                Err(e) => {
                    // Fall back to raw git init
                    match git2::Repository::init(models_dir.join(".registry")) {
                        Ok(_) => print_check(label, CheckStatus::Fixed, ".registry initialized (git init)"),
                        Err(e2) => {
                            print_check(label, CheckStatus::Fail, &format!("failed to init: Git2DB: {e}, git init: {e2}"));
                            all_passed = false;
                        }
                    }
                }
            }
        }
    }

    // 3. Policy files
    {
        let label = "Policy files";
        let policies_dir = models_dir.join(".registry").join("policies");

        let model_conf = policies_dir.join("model.conf");
        let policy_csv = policies_dir.join("policy.csv");

        if model_conf.exists() && policy_csv.exists() {
            print_check(label, CheckStatus::Ok, "model.conf + policy.csv present");
        } else {
            // PolicyManager::new creates defaults
            match PolicyManager::new(&policies_dir).await {
                Ok(_) => print_check(label, CheckStatus::Fixed, "created default policy files"),
                Err(e) => {
                    print_check(label, CheckStatus::Fail, &format!("failed to create: {e}"));
                    all_passed = false;
                }
            }
        }
    }

    // 4. Signing key
    {
        let label = "Signing key";
        let keys_dir = models_dir.join(".registry").join("keys");
        let key_path = keys_dir.join("signing.key");

        if key_path.exists() {
            match tokio::fs::read(&key_path).await {
                Ok(bytes) if bytes.len() == 32 => {
                    print_check(label, CheckStatus::Ok, "Ed25519 key loaded (32 bytes)");
                }
                Ok(bytes) => {
                    print_check(label, CheckStatus::Fail, &format!("invalid key: {} bytes (expected 32)", bytes.len()));
                    all_passed = false;
                }
                Err(e) => {
                    print_check(label, CheckStatus::Fail, &format!("read error: {e}"));
                    all_passed = false;
                }
            }
        } else {
            match load_or_generate_signing_key(&keys_dir).await {
                Ok(_) => print_check(label, CheckStatus::Fixed, "generated new Ed25519 key"),
                Err(e) => {
                    print_check(label, CheckStatus::Fail, &format!("failed to generate: {e}"));
                    all_passed = false;
                }
            }
        }
    }

    // 4b. TLS materials (HTTP + QUIC) — generate into secrets dir so they are
    //     available for systemd-creds encryption in phase 6.
    {
        match crate::config::HyprConfig::resolve_secrets_dir() {
            Ok(secrets_dir) => {
                // HTTP TLS (365-day self-signed)
                match crate::auth::identity_store::load_or_generate_tls_materials(&secrets_dir, "localhost", 365) {
                    Ok(_) => print_check("TLS key+cert", CheckStatus::Ok, "HTTP (365d)"),
                    Err(e) => {
                        print_check("TLS key+cert", CheckStatus::Fail, &format!("{e}"));
                        all_passed = false;
                    }
                }
                // QUIC TLS (14-day per WebTransport spec)
                match crate::auth::identity_store::load_or_generate_tls_materials_named(
                    &secrets_dir, "localhost", 14, "quic-key", "quic-cert",
                ) {
                    Ok(_) => print_check("QUIC key+cert", CheckStatus::Ok, "WebTransport (14d)"),
                    Err(e) => {
                        print_check("QUIC key+cert", CheckStatus::Fail, &format!("{e}"));
                        all_passed = false;
                    }
                }
            }
            Err(e) => {
                print_check("Secrets directory", CheckStatus::Fail, &format!("{e}"));
                all_passed = false;
            }
        }
    }

    // 4c. RSA key for RS256 JWT signing (OIDC interop)
    {
        match crate::config::HyprConfig::resolve_secrets_dir() {
            Ok(secrets_dir) => {
                match crate::auth::identity_store::load_or_generate_rsa_key(&secrets_dir) {
                    Ok(_) => print_check("RSA key", CheckStatus::Ok, "RS256 (2048-bit)"),
                    Err(e) => {
                        // Non-fatal: EdDSA still works, RS256 is for interop
                        print_check("RSA key", CheckStatus::Warn, &format!("{e}"));
                    }
                }
            }
            Err(e) => {
                print_check("Secrets directory", CheckStatus::Warn, &format!("{e}"));
            }
        }
    }

    // 5. Git config (warning only, don't modify)
    {
        let label = "Git identity";
        match git2::Config::open_default() {
            Ok(config) => {
                let has_name = config.get_string("user.name").is_ok();
                let has_email = config.get_string("user.email").is_ok();

                if has_name && has_email {
                    let name = config.get_string("user.name").unwrap_or_default();
                    print_check(label, CheckStatus::Ok, &name);
                } else {
                    let mut missing = Vec::new();
                    if !has_name { missing.push("user.name"); }
                    if !has_email { missing.push("user.email"); }
                    let msg = format!("{} not set", missing.join(", "));
                    print_check(label, CheckStatus::Warn, &msg);
                    warnings.push(
                        "Set git identity: git config --global user.name \"Your Name\" && git config --global user.email \"you@example.com\"".to_owned()
                    );
                }
            }
            Err(_) => {
                print_check(label, CheckStatus::Warn, "could not read git config");
                warnings.push("Set git identity for policy versioning".to_owned());
            }
        }
    }

    // 6. Service JWT presence
    {
        let credentials_dir = crate::auth::identity_store::credentials_dir()?;

        let mut missing_jwts = Vec::new();
        for factory in hyprstream_service::list_factories() {
            let svc = factory.name;
            if svc == "policy" {
                continue; // PolicyService uses the root CA key, not a per-service JWT
            }
            match crate::auth::identity_store::load_service_jwt(&credentials_dir, svc) {
                Ok(Some(_)) => {}
                _ => missing_jwts.push(svc),
            }
        }

        if missing_jwts.is_empty() {
            print_check("Service JWTs", CheckStatus::Ok, "all present");
        } else {
            print_check(
                "Service JWTs",
                CheckStatus::Warn,
                &format!("missing for: {}. Run: hyprstream wizard", missing_jwts.join(", ")),
            );
            warnings.push("Run 'hyprstream wizard' to generate missing service JWTs".to_owned());
        }
    }

    // 7. Policy active
    {
        let label = "Policy active";
        let policies_dir = models_dir.join(".registry").join("policies");
        let policy_csv = policies_dir.join("policy.csv");

        if policy_csv.exists() {
            match tokio::fs::read_to_string(&policy_csv).await {
                Ok(content) => {
                    let rule_count = content.lines()
                        .filter(|l| l.starts_with("p,") || l.starts_with("p "))
                        .count();

                    if rule_count > 0 {
                        // Get first subject for display
                        let first_subject = content.lines()
                            .find(|l| l.starts_with("p,") || l.starts_with("p "))
                            .and_then(|l| l.split(',').nth(1))
                            .map(|s| s.trim().to_owned())
                            .unwrap_or_default();
                        print_check(label, CheckStatus::Ok, &format!("{rule_count} allow rule(s) ({first_subject})"));
                    } else {
                        print_check(label, CheckStatus::Warn, "no allow rules (deny-by-default)");
                        warnings.push("Apply a template: hyprstream quick policy list-templates".to_owned());
                    }
                }
                Err(e) => {
                    print_check(label, CheckStatus::Fail, &format!("read error: {e}"));
                    all_passed = false;
                }
            }
        } else {
            print_check(label, CheckStatus::Warn, "policy.csv not found");
            warnings.push("Run 'hyprstream service install' again after fixing policy files".to_owned());
        }
    }

    // 8. Bootstrap-pubkeys hybrid posture
    //
    // A stale Ed25519-only file (from a pre-hybrid provisioning run) blocks
    // startup: service identities can never be anchored for post-quantum
    // verification. Detect it here and name the recovery, rather than letting
    // the operator discover it from a startup refusal.
    {
        let label = "Hybrid bootstrap";
        match crate::config::HyprConfig::resolve_secrets_dir() {
            Ok(secrets_dir) => {
                match crate::auth::identity_store::load_bootstrap_pubkeys_hybrid(&secrets_dir) {
                    Ok(entries) if entries.is_empty() => {
                        print_check(label, CheckStatus::Ok, "unprovisioned (clean)");
                    }
                    Ok(entries) => {
                        match crate::auth::identity_store::ensure_bootstrap_pubkeys_hybrid(&entries) {
                            Ok(()) => {
                                let n = entries.len();
                                let noun = if n == 1 { "entry" } else { "entries" };
                                print_check(label, CheckStatus::Ok,
                                    &format!("{n} hybrid {noun} bound"));
                            }
                            Err(e) => {
                                print_check(label, CheckStatus::Fail, &format!("{e}"));
                                all_passed = false;
                                warnings.push(
                                    "Re-provision with 'hyprstream wizard' to bind \
                                     ML-DSA-65 keys for every service".to_owned()
                                );
                            }
                        }
                    }
                    Err(e) => {
                        print_check(label, CheckStatus::Fail, &format!("parse error: {e}"));
                        all_passed = false;
                        warnings.push(
                            "Fix the file or re-provision with 'hyprstream wizard'".to_owned()
                        );
                    }
                }
            }
            Err(e) => {
                print_check(label, CheckStatus::Warn, &format!("secrets dir: {e}"));
            }
        }
    }

    // Summary
    println!();
    if !warnings.is_empty() {
        println!("  Suggestions:");
        for w in &warnings {
            println!("    {w}");
        }
        println!();
    }

    if all_passed && warnings.is_empty() {
        println!("    All checks passed.");
    } else if all_passed {
        println!("    All checks passed (with warnings).");
    } else {
        println!("    Some checks failed. Review output above.");
    }

    if all_passed {
        Ok(())
    } else {
        Err(anyhow::anyhow!("one or more repair checks failed; review output above"))
    }
}

#[allow(dead_code)]
pub(crate) enum CheckStatus {
    Ok,
    Fixed,
    Warn,
    Fail,
    Info,
}

pub(crate) fn print_check(label: &str, status: CheckStatus, detail: &str) {
    let (icon, color) = match status {
        CheckStatus::Ok => ("\u{2713}", "\x1b[32m"),    // green checkmark
        CheckStatus::Fixed => ("\u{2713}", "\x1b[33m"),  // yellow checkmark (fixed)
        CheckStatus::Warn => ("\u{26A0}", "\x1b[33m"),   // yellow warning
        CheckStatus::Fail => ("\u{2717}", "\x1b[31m"),   // red X
        CheckStatus::Info => ("\u{25CB}", "\x1b[36m"),   // cyan circle (informational)
    };
    println!("  {color}{icon}\x1b[0m {:<20} {detail}", label);
}

// =============================================================================
// Version helpers
// =============================================================================

/// Get the full build version string
///
/// Format: `{cargo_version}+{branch}.g{sha7}[.dirty]`
/// Example: `0.1.0-alpha-7+main.gabc1234.dirty`
///
/// Uses BUILD_VERSION from build.rs, falls back to CARGO_PKG_VERSION.
pub(crate) fn build_version() -> &'static str {
    option_env!("BUILD_VERSION").unwrap_or(env!("CARGO_PKG_VERSION"))
}

// =============================================================================
// Command alias installation helpers
// =============================================================================

// =============================================================================
// InstallPlan: unified binary installation pipeline
// =============================================================================

/// Plan for installing the hyprstream binary/AppImage.
///
/// Separates detection and validation (`prepare()`) from side effects (`execute()`).
/// Both `service install` and the wizard share this pipeline.
pub(crate) struct InstallPlan {
    pub(crate) source: PathBuf,
    pub(crate) is_appimage: bool,
    pub(crate) source_size: u64,
    pub(crate) version: &'static str,
    pub(crate) filename: &'static str,
    pub(crate) version_dir: PathBuf,
    pub(crate) bin_dir: PathBuf,
    pub(crate) available_space: u64,
}

/// Result of a successful `InstallPlan::execute()`.
pub(crate) struct InstallResult {
    pub(crate) bin_dir: PathBuf,
    pub(crate) version_dir: PathBuf,
    pub(crate) is_appimage: bool,
    pub(crate) updated_profiles: Vec<String>,
}

impl InstallPlan {
    /// Detect source, validate it, resolve paths, check disk space.
    /// No side effects — safe to call and discard.
    pub(crate) fn prepare() -> Result<Self> {
        let (source, is_appimage) = binary_copy_source()?;
        let source_size = validate_source(&source)?;

        let version = build_version();
        let filename = if is_appimage { "hyprstream.appimage" } else { "hyprstream" };

        let bin_dir = hyprstream_rpc::paths::bin_dir()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine user executable directory"))?;
        let version_dir = hyprstream_rpc::paths::version_dir(version)
            .ok_or_else(|| anyhow::anyhow!("Cannot determine version directory"))?;

        let available_space = available_space(&version_dir).unwrap_or(0);

        Ok(Self {
            source,
            is_appimage,
            source_size,
            version,
            filename,
            version_dir,
            bin_dir,
            available_space,
        })
    }

    /// Execute the install: copy, symlink, update shell profiles.
    /// Consumes the plan to prevent double-execution.
    pub(crate) fn execute(self) -> Result<InstallResult> {
        std::fs::create_dir_all(&self.bin_dir)
            .with_context(|| format!("Failed to create directory: {}", self.bin_dir.display()))?;
        std::fs::create_dir_all(&self.version_dir)
            .with_context(|| format!("Failed to create directory: {}", self.version_dir.display()))?;

        // Copy binary to versioned directory (skip if already in place)
        let versioned_binary = self.version_dir.join(self.filename);
        let same_file = std::fs::canonicalize(&self.source).ok()
            == std::fs::canonicalize(&versioned_binary).ok();
        if !same_file {
            remove_if_exists(&versioned_binary)?;
            std::fs::copy(&self.source, &versioned_binary).with_context(|| {
                format!(
                    "Failed to copy {} -> {}",
                    self.source.display(),
                    versioned_binary.display()
                )
            })?;
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&versioned_binary, std::fs::Permissions::from_mode(0o755))?;
        }

        // Create symlinks in bin_dir
        let bin_appimage = self.bin_dir.join("hyprstream.appimage");
        let bin_hyprstream = self.bin_dir.join("hyprstream");

        // Calculate relative path from bin to versioned binary
        let relative_path = Path::new("..")
            .join("share")
            .join("hyprstream")
            .join("versions")
            .join(self.version)
            .join(self.filename);

        remove_if_exists(&bin_appimage)?;
        remove_if_exists(&bin_hyprstream)?;

        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&relative_path, &bin_appimage)
                .with_context(|| format!("Failed to create symlink: {}", bin_appimage.display()))?;
            std::os::unix::fs::symlink(Path::new("hyprstream.appimage"), &bin_hyprstream)
                .with_context(|| format!("Failed to create symlink: {}", bin_hyprstream.display()))?;
        }

        let updated_profiles = if let Some(home) = dirs::home_dir() {
            update_shell_profiles(&home, &self.bin_dir).unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(InstallResult {
            bin_dir: self.bin_dir,
            version_dir: self.version_dir,
            is_appimage: self.is_appimage,
            updated_profiles,
        })
    }

    pub(crate) fn has_sufficient_space(&self) -> bool {
        self.available_space == 0 || self.available_space >= self.source_size + 1024 * 1024
    }

    pub(crate) fn type_label(&self) -> &'static str {
        if self.is_appimage { "AppImage" } else { "binary" }
    }
}

impl InstallResult {
    pub(crate) fn type_label(&self) -> &'static str {
        if self.is_appimage { "AppImage" } else { "binary" }
    }
}

// =============================================================================
// Source detection and validation helpers
// =============================================================================

/// Get the copy source: `$APPIMAGE` if set, otherwise `argv[0]`.
///
/// `$APPIMAGE` is set by the AppImage runtime and points to the stable
/// AppImage file (not the temporary FUSE mount). `argv[0]` preserves
/// what the shell resolved.
fn binary_copy_source() -> Result<(PathBuf, bool)> {
    if let Ok(appimage) = std::env::var("APPIMAGE") {
        let path = PathBuf::from(&appimage);
        if path.exists() {
            return Ok((path, true));
        }
    }

    // Fall back to argv[0]
    let argv0 = std::env::args_os()
        .next()
        .context("No argv[0] available")?;
    let path = PathBuf::from(&argv0);

    // Resolve relative paths against CWD
    let path = if path.is_relative() {
        std::env::current_dir()?.join(&path)
    } else {
        path
    };

    let is_appimage = is_appimage_file(&path).unwrap_or(false);
    Ok((path, is_appimage))
}

/// Validate the source file before copying:
/// - Must be a regular file (not symlink, directory, device, etc.)
/// - Must not be empty
/// - Must be owned by current user or root
fn validate_source(path: &Path) -> Result<u64> {
    use std::os::unix::fs::MetadataExt;

    let meta = std::fs::symlink_metadata(path)
        .with_context(|| format!("Cannot stat source: {}", path.display()))?;

    if !meta.file_type().is_file() {
        anyhow::bail!(
            "Source is not a regular file: {} (type: {:?})",
            path.display(),
            meta.file_type()
        );
    }

    let size = meta.len();
    if size == 0 {
        anyhow::bail!("Source file is empty: {}", path.display());
    }

    // Ownership: must be current user or root
    let file_uid = meta.uid();
    let my_uid = nix::unistd::getuid().as_raw();
    if file_uid != my_uid && file_uid != 0 {
        anyhow::bail!(
            "Source file owned by uid {} (expected {} or root): {}",
            file_uid,
            my_uid,
            path.display()
        );
    }

    Ok(size)
}

/// Check available disk space at the given path using `statvfs`.
pub(crate) fn available_space(path: &Path) -> Result<u64> {
    let check_path = if path.exists() {
        path.to_path_buf()
    } else {
        path.ancestors()
            .find(|p| p.exists())
            .unwrap_or_else(|| Path::new("/"))
            .to_path_buf()
    };
    let stat = nix::sys::statvfs::statvfs(&check_path)
        .with_context(|| format!("statvfs failed on {}", check_path.display()))?;
    Ok(stat.blocks_available() as u64 * stat.fragment_size() as u64)
}

/// Format a byte count as a human-readable string (powers of 1024, matching `df -h`).
pub(crate) fn format_size(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;

    if bytes >= GIB {
        let val = bytes as f64 / GIB as f64;
        if val >= 100.0 { format!("{:.0} GB", val) } else { format!("{:.1} GB", val) }
    } else if bytes >= MIB {
        let val = bytes as f64 / MIB as f64;
        if val >= 100.0 { format!("{:.0} MB", val) } else { format!("{:.1} MB", val) }
    } else if bytes >= KIB {
        format!("{:.0} KB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Check if the running binary is already in a known installed location.
///
/// Uses `current_exe()` (reads `/proc/self/exe` on Linux — kernel-maintained,
/// cannot be spoofed) to determine the real binary path, then checks if it
/// lives under any of the standard install locations.
pub(crate) fn is_binary_installed() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let canonical = exe.canonicalize().ok()?;

    // Check 1: Under the XDG data dir version store
    if let Some(versions_dir) = hyprstream_rpc::paths::versions_dir() {
        if let Ok(versions_canonical) = versions_dir.canonicalize() {
            if canonical.starts_with(&versions_canonical) {
                return Some(canonical);
            }
        }
    }

    // Check 2: Under the XDG executable dir
    if let Some(bin_dir) = dirs::executable_dir() {
        if let Ok(bin_canonical) = bin_dir.canonicalize() {
            if canonical.starts_with(&bin_canonical) {
                return Some(canonical);
            }
        }
    }

    // Check 3: ~/bin (traditional Unix)
    if let Some(home) = dirs::home_dir() {
        let home_bin = home.join("bin");
        if let Ok(home_bin_canonical) = home_bin.canonicalize() {
            if canonical.starts_with(&home_bin_canonical) {
                return Some(canonical);
            }
        }

        // Check 4: ~/Applications (AppImage community convention)
        let applications = home.join("Applications");
        if let Ok(app_canonical) = applications.canonicalize() {
            if canonical.starts_with(&app_canonical) {
                return Some(canonical);
            }
        }
    }

    // Check 5: Inode match against any PATH entry
    if let Ok(exe_meta) = std::fs::metadata(&canonical) {
        use std::os::unix::fs::MetadataExt;
        let exe_dev = exe_meta.dev();
        let exe_ino = exe_meta.ino();

        if let Ok(path_var) = std::env::var("PATH") {
            for dir in path_var.split(':') {
                if dir.is_empty() || !Path::new(dir).is_absolute() {
                    continue;
                }
                let candidate = Path::new(dir).join("hyprstream");
                if let Ok(meta) = std::fs::metadata(&candidate) {
                    if meta.dev() == exe_dev && meta.ino() == exe_ino {
                        return Some(canonical);
                    }
                }
            }
        }
    }

    None
}

/// Check if a file is an AppImage by reading its magic bytes
///
/// AppImage Type 2 has:
/// - ELF magic at offset 0: 0x7f 'E' 'L' 'F'
/// - AppImage magic at offset 8: 'A' 'I' 0x02
pub(crate) fn is_appimage_file(path: &Path) -> Result<bool> {
    use std::io::Read;

    let mut file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open: {}", path.display()))?;

    let mut header = [0u8; 11];
    if file.read_exact(&mut header).is_err() {
        return Ok(false);
    }

    // Check ELF magic
    let is_elf = header[0..4] == [0x7f, b'E', b'L', b'F'];

    // Check AppImage Type 2 magic at offset 8
    let is_appimage = header[8..11] == [b'A', b'I', 0x02];

    Ok(is_elf && is_appimage)
}

/// Remove a file if it exists (handles both regular files and symlinks)
fn remove_if_exists(path: &Path) -> Result<()> {
    if path.symlink_metadata().is_ok() {
        std::fs::remove_file(path)
            .with_context(|| format!("Failed to remove: {}", path.display()))?;
    }
    Ok(())
}

/// Handle `--print-cert-hash` — output the SHA-256 hash of the QUIC certificate.
///
/// Loads or generates the TLS certificate from QuicConfig and prints
/// the base64-encoded SHA-256 hash, suitable for use in the browser's
/// `serverCertificateHashes` WebTransport option.
pub fn handle_print_cert_hash(quic_config: &crate::config::QuicConfig) -> Result<()> {
    let (cert_chain, _key_der) = quic_config.load_tls_materials()
        .context("Failed to load/generate QUIC TLS certificate")?;

    let hash = hyprstream_rpc::transport::zmtp_quic::cert_hash(&cert_chain[0]);
    println!("{}", hash);
    Ok(())
}

/// Handle `service ensure-key` — materialize a service's signing key and its
/// public sidecars without starting any services.
///
/// Runs the same loader the service itself uses (`resolve_service_signing_key`,
/// so `policy` resolves to the flat node/CA key), then guarantees the public
/// sidecars exist next to the seed:
///
/// - `signing-key.pub` — 32-byte Ed25519 verifying key (mode 0644)
/// - `service-pubkey.hybrid` — 1984-byte hybrid bootstrap entry (mode 0644)
///
/// Idempotent: an existing key is loaded, never rotated, and up-to-date
/// sidecars are left untouched. Prints the base64 public key and the sidecar
/// paths for provisioning units that wire them into a credential mint.
pub fn handle_service_ensure_key(config: Option<&crate::config::HyprConfig>, name: &str) -> Result<()> {
    use base64::Engine as _;
    use crate::auth::identity_store;

    identity_store::validate_service_name(name)?;
    let secrets_dir = identity_store::credentials_dir_for_config(config)?;
    let profile = identity_store::SecretsProfile::from_env()?;
    let key = identity_store::resolve_service_signing_key(&secrets_dir, name, profile)?;
    let key_dir = identity_store::service_signing_key_dir(&secrets_dir, name, profile);
    identity_store::ensure_service_key_sidecars(&key_dir, &key)?;

    let pubkey_b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(key.verifying_key().as_bytes());
    println!("service '{name}' signing key ensured");
    println!("  public key (base64): {pubkey_b64}");
    println!("  {}", key_dir.join(identity_store::SIGNING_KEY_PUB_NAME).display());
    println!("  {}", key_dir.join(identity_store::SERVICE_PUBKEY_HYBRID_NAME).display());
    Ok(())
}

/// Update shell profiles to include bin_dir in PATH
fn update_shell_profiles(home: &Path, bin_dir: &Path) -> Result<Vec<String>> {
    let path_line = format!(r#"export PATH="{}:$PATH""#, bin_dir.display());
    let bin_dir_str = bin_dir.to_string_lossy();
    let mut updated = Vec::new();

    for profile in &[".bashrc", ".zshrc", ".profile"] {
        let profile_path = home.join(profile);
        if profile_path.exists() {
            let content = std::fs::read_to_string(&profile_path)
                .with_context(|| format!("Failed to read: {}", profile_path.display()))?;
            // Check if bin_dir already in PATH
            if !content.contains(bin_dir_str.as_ref()) {
                let mut file = std::fs::OpenOptions::new()
                    .append(true)
                    .open(&profile_path)
                    .with_context(|| format!("Failed to open: {}", profile_path.display()))?;
                writeln!(file, "\n# Added by hyprstream\n{}", path_line)?;
                updated.push((*profile).to_owned());
            }
        }
    }

    Ok(updated)
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod offline_policy_provision_tests {
    use super::*;
    use crate::auth::{get_template, PolicyManager};

    fn public_templates() -> Vec<String> {
        vec!["public-inference".to_owned(), "public-read".to_owned()]
    }

    #[tokio::test]
    async fn fresh_and_repeat_public_staging_policy_is_canonical() {
        let root = tempfile::tempdir().expect("temporary models root");
        handle_service_provision_policy_templates(root.path(), &public_templates())
            .await
            .expect("fresh provision");
        let policy_path = root.path().join(".registry/policies/policy.csv");
        let first = tokio::fs::read(&policy_path).await.expect("first policy");

        handle_service_provision_policy_templates(root.path(), &public_templates())
            .await
            .expect("repeat provision");
        let second = tokio::fs::read(&policy_path)
            .await
            .expect("repeated policy");
        assert_eq!(first, second, "repeat must leave serialized policy stable");

        let manager = PolicyManager::new(root.path().join(".registry/policies"))
            .await
            .expect("reopen policy store");
        let policies = manager.get_policy().await;
        let expected = public_templates()
            .iter()
            .flat_map(|name| get_template(name).expect("template").expanded_policies())
            .map(|rule| rule.to_vec())
            .collect::<Vec<_>>();
        assert_eq!(expected.len(), 3);
        for rule in expected {
            assert_eq!(
                policies
                    .iter()
                    .filter(|candidate| candidate.as_slice() == rule.as_slice())
                    .count(),
                1
            );
        }
        assert!(
            !manager
                .check_with_domain(
                    "anonymous",
                    "*",
                    "model:policy-bootstrap-probe",
                    "ttt.writeback",
                )
                .await
        );
    }

    #[tokio::test]
    async fn retained_partial_template_converges() {
        let root = tempfile::tempdir().expect("temporary models root");
        let policies_dir = root.path().join(".registry/policies");
        let manager = PolicyManager::new(&policies_dir)
            .await
            .expect("policy manager");
        manager
            .add_policy_with_domain("anonymous", "*", "model:*", "infer.generate", "allow")
            .await
            .expect("partial rule");
        manager.save().await.expect("save partial state");
        drop(manager);

        handle_service_provision_policy_templates(root.path(), &public_templates())
            .await
            .expect("partial state must converge");
        let verified = PolicyManager::new(&policies_dir)
            .await
            .expect("reopen converged policy");
        assert!(
            verified
                .check_with_domain(
                    "anonymous",
                    "*",
                    "model:policy-bootstrap-probe",
                    "query.status",
                )
                .await
        );
        assert!(
            verified
                .check_with_domain(
                    "anonymous",
                    "*",
                    "registry:policy-bootstrap-probe",
                    "query.status",
                )
                .await
        );
    }

    #[tokio::test]
    async fn retained_partial_domain_grouping_converges() {
        let root = tempfile::tempdir().expect("temporary models root");
        let policies_dir = root.path().join(".registry/policies");
        let manager = PolicyManager::new(&policies_dir)
            .await
            .expect("policy manager");
        manager
            .add_role_for_user_in_domain("service:inference:host-1", "mesh-readers", "acme")
            .await
            .expect("partial domain grouping");
        manager.save().await.expect("save partial grouping");
        drop(manager);

        handle_service_provision_policy_templates(root.path(), &["mesh-host-group".to_owned()])
            .await
            .expect("partial grouping must converge");
        let verified = PolicyManager::new(&policies_dir)
            .await
            .expect("reopen grouping policy");
        let groupings = verified.get_domain_grouping_policy().await;
        for host in ["service:inference:host-1", "service:inference:host-2"] {
            assert!(groupings.contains(&vec![
                host.to_owned(),
                "mesh-readers".to_owned(),
                "acme".to_owned(),
            ]));
        }
    }

    #[tokio::test]
    async fn invalid_requests_fail_before_storage_mutation() {
        for templates in [
            vec!["not-a-template".to_owned()],
            vec!["public-read".to_owned(), "public-read".to_owned()],
        ] {
            let root = tempfile::tempdir().expect("temporary models root");
            assert!(
                handle_service_provision_policy_templates(root.path(), &templates)
                    .await
                    .is_err()
            );
            assert!(!root.path().join(".registry").exists());
        }
    }

    #[tokio::test]
    async fn malformed_or_explicitly_denied_policy_fails_closed() {
        let malformed_root = tempfile::tempdir().expect("malformed models root");
        let malformed_dir = malformed_root.path().join(".registry/policies");
        PolicyManager::new(&malformed_dir)
            .await
            .expect("initialize policy");
        let malformed_path = malformed_dir.join("policy.csv");
        let malformed = b"p, malformed\n";
        tokio::fs::write(&malformed_path, malformed)
            .await
            .expect("write malformed policy");
        assert!(handle_service_provision_policy_templates(
            malformed_root.path(),
            &public_templates(),
        )
        .await
        .is_err());
        assert_eq!(
            tokio::fs::read(&malformed_path)
                .await
                .expect("read malformed"),
            malformed
        );

        let denied_root = tempfile::tempdir().expect("denied models root");
        let denied_dir = denied_root.path().join(".registry/policies");
        let denied = PolicyManager::new(&denied_dir)
            .await
            .expect("initialize denied policy");
        denied
            .add_policy_with_domain("anonymous", "*", "model:*", "infer.generate", "deny")
            .await
            .expect("add explicit deny");
        denied.save().await.expect("persist explicit deny");
        drop(denied);
        let denied_path = denied_dir.join("policy.csv");
        let denied_original = tokio::fs::read(&denied_path)
            .await
            .expect("retained anonymous deny");
        assert!(
            handle_service_provision_policy_templates(denied_root.path(), &public_templates(),)
                .await
                .is_err(),
            "explicit deny must block verified staging readiness"
        );
        assert_eq!(
            tokio::fs::read(&denied_path)
                .await
                .expect("anonymous deny after refused retry"),
            denied_original
        );
        let reopened = PolicyManager::new(&denied_dir)
            .await
            .expect("reopen denied policy");
        assert!(
            !reopened
                .check_with_domain(
                    "anonymous",
                    "*",
                    "model:policy-bootstrap-probe",
                    "infer.generate",
                )
                .await
        );
    }

    #[tokio::test]
    async fn prepublication_failure_and_retry_preserve_retained_deny() {
        let root = tempfile::tempdir().expect("retained-policy models root");
        let policies_dir = root.path().join(".registry/policies");
        let denied = PolicyManager::new(&policies_dir)
            .await
            .expect("initialize policy");
        let deny = vec![
            "service:retained".to_owned(),
            "*".to_owned(),
            "model:*".to_owned(),
            "ttt.writeback".to_owned(),
            "deny".to_owned(),
        ];
        denied
            .add_policy_with_domain(
                "service:retained",
                "*",
                "model:*",
                "ttt.writeback",
                "deny",
            )
            .await
            .expect("add retained deny");
        denied.save().await.expect("persist retained deny");
        drop(denied);
        let policy_path = policies_dir.join("policy.csv");
        let original = tokio::fs::read(&policy_path)
            .await
            .expect("retained policy");
        assert!(!original.is_empty());

        let mut fault_reached = false;
        assert!(provision_policy_templates(
                root.path(),
                &public_templates(),
                |_| {
                    fault_reached = true;
                    anyhow::bail!("injected failure after staged write before publication")
                },
                |_| Ok(()),
                |_| Ok(()),
            )
            .await
            .is_err());
        assert!(fault_reached, "prepublication fault seam must be reached");
        assert_eq!(
            tokio::fs::read(&policy_path)
                .await
                .expect("policy after injected failure"),
            original
        );

        let retained = PolicyManager::new(&policies_dir)
            .await
            .expect("reopen retained policy");
        assert!(retained.get_policy().await.contains(&deny));
        drop(retained);

        handle_service_provision_policy_templates(root.path(), &public_templates())
            .await
            .expect("retry must converge");
        let after_retry = PolicyManager::new(&policies_dir)
            .await
            .expect("reopen retry policy");
        assert!(after_retry.get_policy().await.contains(&deny));
    }

    #[tokio::test]
    async fn atomic_policy_publication_failure_preserves_retained_deny_and_retry() {
        let root = tempfile::tempdir().expect("retained-policy models root");
        let policies_dir = root.path().join(".registry/policies");
        let retained = PolicyManager::new(&policies_dir)
            .await
            .expect("initialize policy");
        let deny = vec![
            "service:retained".to_owned(),
            "*".to_owned(),
            "model:*".to_owned(),
            "ttt.writeback".to_owned(),
            "deny".to_owned(),
        ];
        retained
            .add_policy_with_domain(
                "service:retained",
                "*",
                "model:*",
                "ttt.writeback",
                "deny",
            )
            .await
            .expect("add retained deny");
        retained.save().await.expect("persist retained deny");
        drop(retained);
        let policy_path = policies_dir.join("policy.csv");
        let original = tokio::fs::read(&policy_path)
            .await
            .expect("retained policy bytes");

        let mut wrote_model_temp = false;
        let mut failed_policy_write = false;
        let result = provision_policy_templates(
            root.path(),
            &public_templates(),
            |_| Ok(()),
            |_| Ok(()),
            |destination| {
                if destination.file_name().is_some_and(|name| name == "model.conf") {
                    wrote_model_temp = true;
                } else if destination.file_name().is_some_and(|name| name == "policy.csv") {
                    failed_policy_write = true;
                    anyhow::bail!("injected failure after policy temp-file write")
                }
                Ok(())
            },
        )
        .await;
        assert!(result.is_err());
        assert!(wrote_model_temp, "model publication must precede policy");
        assert!(failed_policy_write, "fault must reach policy publication");
        assert_eq!(
            tokio::fs::read(&policy_path)
                .await
                .expect("policy after publication failure"),
            original
        );
        let after_failure = PolicyManager::new(&policies_dir)
            .await
            .expect("reopen policy after failure");
        assert!(after_failure.get_policy().await.contains(&deny));
        drop(after_failure);

        handle_service_provision_policy_templates(root.path(), &public_templates())
            .await
            .expect("retry converges without losing retained deny");
        let after_retry = PolicyManager::new(&policies_dir)
            .await
            .expect("reopen policy after retry");
        assert!(after_retry.get_policy().await.contains(&deny));
        assert!(
            !after_retry
                .check_with_domain(
                    "service:retained",
                    "*",
                    "model:policy-bootstrap-probe",
                    "ttt.writeback",
                )
                .await
        );
    }

    #[tokio::test]
    async fn incomplete_staged_snapshot_cannot_drop_retained_deny() {
        let root = tempfile::tempdir().expect("incomplete-snapshot models root");
        let policies_dir = root.path().join(".registry/policies");
        let retained = PolicyManager::new(&policies_dir)
            .await
            .expect("initialize policy");
        retained
            .add_policy_with_domain(
                "service:retained",
                "*",
                "model:*",
                "ttt.writeback",
                "deny",
            )
            .await
            .expect("add retained deny");
        retained.save().await.expect("persist retained deny");
        drop(retained);
        let policy_path = policies_dir.join("policy.csv");
        let original = tokio::fs::read(&policy_path)
            .await
            .expect("retained policy bytes");

        let result = provision_policy_templates(
            root.path(),
            &public_templates(),
            |staged_path| {
                let staged = std::fs::read_to_string(staged_path)?;
                for name in public_templates() {
                    let template = get_template(&name).expect("public template");
                    for rule in template.expanded_policies() {
                        assert!(staged.contains(&format!("p, {}", rule.to_vec().join(", "))));
                    }
                }
                let mut removed = 0;
                let incomplete = staged
                    .lines()
                    .filter(|line| {
                        let keep = !line.contains("service:retained")
                            || !line.contains("ttt.writeback")
                            || !line.ends_with("deny");
                        if !keep {
                            removed += 1;
                        }
                        keep
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n";
                assert_eq!(removed, 1, "fault must omit exactly the retained DENY");
                std::fs::write(staged_path, incomplete)?;
                Ok(())
            },
            |_| Ok(()),
            |_| Ok(()),
        )
        .await;
        assert!(
            result
                .expect_err("incomplete snapshot must be rejected")
                .to_string()
                .contains("complete intended policy state")
        );
        assert_eq!(
            tokio::fs::read(&policy_path)
                .await
                .expect("live policy after rejected snapshot"),
            original
        );
    }

    #[tokio::test]
    async fn mutation_after_snapshot_cannot_change_published_policy() {
        let root = tempfile::tempdir().expect("post-snapshot-mutation models root");
        let mut captured = None;
        provision_policy_templates(
            root.path(),
            &public_templates(),
            |_| Ok(()),
            |staged_path| {
                captured = Some(std::fs::read(staged_path)?);
                std::fs::write(staged_path, b"p, malformed\n")?;
                Ok(())
            },
            |_| Ok(()),
        )
        .await
        .expect("immutable captured policy must publish successfully");
        assert_eq!(
            tokio::fs::read(root.path().join(".registry/policies/policy.csv"))
                .await
                .expect("published policy"),
            captured.expect("snapshot callback captured bytes")
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unreadable_retained_policy_is_not_replaced_with_defaults() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = tempfile::tempdir().expect("unreadable-policy models root");
        let policies_dir = root.path().join(".registry/policies");
        PolicyManager::new(&policies_dir)
            .await
            .expect("initialize policy");
        let policy_path = policies_dir.join("policy.csv");
        let original = tokio::fs::read(&policy_path)
            .await
            .expect("retained policy bytes");
        tokio::fs::set_permissions(&policy_path, std::fs::Permissions::from_mode(0o000))
            .await
            .expect("make retained policy unreadable");

        assert!(
            handle_service_provision_policy_templates(root.path(), &public_templates())
                .await
                .is_err()
        );
        tokio::fs::set_permissions(&policy_path, std::fs::Permissions::from_mode(0o640))
            .await
            .expect("restore retained policy permissions");
        assert_eq!(
            tokio::fs::read(&policy_path)
                .await
                .expect("policy after read failure"),
            original
        );
    }
}

#[cfg(test)]
mod launcher_tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use hyprstream_service::ProcessReadiness;

    #[test]
    fn direct_child_process_config_builds_profile_specific_invocation() -> anyhow::Result<()> {
        let root = tempfile::tempdir()?;
        // A relative-looking config directory containing spaces: the launcher
        // hands the child one canonical absolute argv element that keeps the
        // spaces (no shell splitting, no quoting).
        let config_path = root.path().join("my configs/custom.toml");
        std::fs::create_dir_all(config_path.parent().expect("parent"))?;
        std::fs::write(&config_path, "[secrets]\n")?;
        let canonical = std::fs::canonicalize(&config_path)?;

        // Required-native: no --ipc, config forwarded, notification readiness.
        let required = direct_child_process_config(
            "model",
            true,
            Some(&canonical),
            Path::new("/usr/local/bin/hyprstream"),
        )?;
        let expected_tail = [
            OsString::from("service"),
            OsString::from("start"),
            OsString::from("model"),
            OsString::from("--foreground"),
        ];
        assert_eq!(&required.args[required.args.len() - 4..], &expected_tail);
        assert!(
            !required.args.iter().any(|arg| arg == "--ipc"),
            "required-native child must not be forced onto the local IPC endpoint"
        );
        assert!(
            required
                .args
                .windows(2)
                .any(|pair| pair[0] == "--config" && pair[1] == canonical.to_str().expect("utf-8")),
            "canonical config path (spaces intact) must be forwarded as one argv element"
        );
        assert!(matches!(
            required.readiness,
            ProcessReadiness::Notify { .. }
        ));

        // Compatibility keeps the historical shape and immediate reporting.
        let compat_with_config = direct_child_process_config(
            "registry",
            false,
            Some(&canonical),
            Path::new("/usr/local/bin/hyprstream"),
        )?;
        assert_eq!(
            &compat_with_config.args[compat_with_config.args.len() - 5..],
            &[
                OsString::from("service"),
                OsString::from("start"),
                OsString::from("registry"),
                OsString::from("--foreground"),
                OsString::from("--ipc"),
            ]
        );
        assert_eq!(compat_with_config.readiness, ProcessReadiness::Immediate);

        // No explicit selector: argv identical to the legacy launcher.
        let compat_default =
            direct_child_process_config("policy", false, None, Path::new("/bin/hyprstream"))?;
        assert_eq!(
            compat_default.args,
            [
                OsString::from("service"),
                OsString::from("start"),
                OsString::from("policy"),
                OsString::from("--foreground"),
                OsString::from("--ipc"),
            ]
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn direct_child_preserves_non_utf8_explicit_config_path() -> anyhow::Result<()> {
        use std::os::unix::ffi::{OsStrExt, OsStringExt};

        let root = tempfile::tempdir()?;
        let config_path = root
            .path()
            .join(OsString::from_vec(b"custom-\xff.toml".to_vec()));
        std::fs::write(&config_path, "[secrets]\n")?;
        let canonical = std::fs::canonicalize(&config_path)?;
        assert!(canonical.to_str().is_none(), "fixture path must be non-UTF-8");

        let loaded = crate::config::HyprConfig::from_file(&canonical)?;
        loaded.validate()?;

        for (iroh_required, expects_ipc) in [(true, false), (false, true)] {
            let plan = direct_child_process_config(
                "model",
                iroh_required,
                Some(&canonical),
                Path::new("/usr/local/bin/hyprstream"),
            )?;
            let selectors: Vec<_> = plan
                .args
                .windows(2)
                .filter(|pair| pair[0] == "--config")
                .collect();
            assert_eq!(selectors.len(), 1, "one explicit selector must be forwarded");
            assert_eq!(
                selectors[0][1].as_os_str().as_bytes(),
                canonical.as_os_str().as_bytes(),
                "the canonical selector must be preserved byte for byte"
            );
            assert_eq!(
                plan.args.iter().any(|arg| arg == "--ipc"),
                expects_ipc,
                "Required and Compatibility invocation shapes must remain distinct"
            );
            assert_eq!(
                matches!(plan.readiness, ProcessReadiness::Notify { .. }),
                iroh_required,
                "Required and Compatibility readiness policies must remain distinct"
            );
        }
        Ok(())
    }

    #[test]
    fn known_required_roster_retains_dependency_order() {
        let targets = ["model", "registry", "policy", "discovery"];
        for service in targets {
            assert!(
                hyprstream_service::get_factory(service).is_some(),
                "test roster service {service} must be compiled"
            );
        }
        assert_eq!(
            hyprstream_service::startup_stages_for_profile(&targets, true),
            vec![
                vec!["discovery".to_owned()],
                vec!["policy".to_owned()],
                vec!["registry".to_owned()],
                vec!["model".to_owned()],
            ]
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unknown_required_roster_is_rejected_before_any_child_spawns()
    -> anyhow::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let root = tempfile::tempdir()?;
        let executable = root.path().join("spawn-sentinel.sh");
        let marker = root.path().join("spawn-sentinel.sh.spawned");
        std::fs::write(&executable, b"#!/bin/sh\ntouch \"$0.spawned\"\n")?;
        let mut permissions = std::fs::metadata(&executable)?.permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&executable, permissions)?;

        let unknown = "not-a-compiled-native-service";
        let targets = vec!["policy".to_owned(), unknown.to_owned()];
        let spawner = hyprstream_service::ProcessSpawner::standalone();
        let error = launch_direct_children(&targets, true, None, &executable, &spawner)
            .await
            .expect_err("an unknown Required target must reject the whole plan");
        assert_eq!(error.to_string(), format!("unknown native service: {unknown}"));
        assert!(
            !marker.exists(),
            "validation must finish before any configured executable runs"
        );
        Ok(())
    }

    /// Mock manager: scripted start/is_active behavior for the permitted
    /// Compatibility unit-start lifecycle contract.
    struct MockManager {
        start_error: Option<&'static str>,
        active_after: std::sync::atomic::AtomicU32,
        calls: parking_lot::Mutex<Vec<String>>,
    }

    impl MockManager {
        fn failing() -> Self {
            Self {
                start_error: Some("unit is masked"),
                active_after: std::sync::atomic::AtomicU32::new(0),
                calls: parking_lot::Mutex::new(Vec::new()),
            }
        }
        fn never_active() -> Self {
            Self {
                start_error: None,
                active_after: std::sync::atomic::AtomicU32::new(u32::MAX),
                calls: parking_lot::Mutex::new(Vec::new()),
            }
        }
        fn immediate() -> Self {
            Self {
                start_error: None,
                active_after: std::sync::atomic::AtomicU32::new(0),
                calls: parking_lot::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl hyprstream_service::ServiceManager for MockManager {
        async fn install(&self, _service: &str) -> anyhow::Result<()> {
            Ok(())
        }
        async fn uninstall(&self, _service: &str) -> anyhow::Result<()> {
            Ok(())
        }
        async fn start(&self, service: &str) -> anyhow::Result<()> {
            self.calls
                .lock()
                .push(format!("start:{service}"));
            match self.start_error {
                Some(reason) => anyhow::bail!("{reason}"),
                None => Ok(()),
            }
        }
        async fn stop(&self, _service: &str) -> anyhow::Result<()> {
            Ok(())
        }
        async fn is_active(&self, service: &str) -> anyhow::Result<bool> {
            self.calls
                .lock()
                .push(format!("active:{service}"));
            use std::sync::atomic::Ordering;
            let remaining = self.active_after.load(Ordering::SeqCst);
            if remaining > 0 && remaining != u32::MAX {
                self.active_after.store(remaining - 1, Ordering::SeqCst);
                return Ok(false);
            }
            Ok(remaining == 0)
        }
        async fn reload(&self) -> anyhow::Result<()> {
            Ok(())
        }

        async fn spawn(
            &self,
            _spawnable: Box<dyn hyprstream_rpc::Spawnable>,
        ) -> anyhow::Result<hyprstream_service::SpawnedService> {
            anyhow::bail!("mock manager does not host services")
        }
    }

    #[tokio::test]
    async fn unit_start_mutation_failure_propagates_without_active_poll() {
        let manager = MockManager::failing();
        let error = start_units_to_active(
            &manager,
            &["model".to_owned()],
            std::time::Duration::from_secs(1),
        )
        .await
        .expect_err("start mutation failure must propagate");
        assert!(
            error.to_string().contains("unit is masked"),
            "real failure expected, got: {error}"
        );
    }

    #[tokio::test]
    async fn unit_start_requires_bounded_active_state() {
        let manager = MockManager::never_active();
        let error = start_units_to_active(
            &manager,
            &["model".to_owned()],
            std::time::Duration::from_millis(300),
        )
        .await
        .expect_err("a unit that never activates must fail the bounded wait");
        assert!(
            error
                .to_string()
                .contains("did not reach the active state"),
            "active-state timeout expected, got: {error}"
        );

        let manager = MockManager::immediate();
        start_units_to_active(
            &manager,
            &["model".to_owned()],
            std::time::Duration::from_secs(1),
        )
        .await
        .expect("immediately-active unit must complete");
    }

    #[tokio::test]
    async fn unit_starts_are_queued_before_active_waits() {
        let manager = MockManager::immediate();
        start_units_to_active(
            &manager,
            &["event".to_owned(), "policy".to_owned()],
            std::time::Duration::from_secs(1),
        )
        .await
        .expect("queued unit starts must complete");

        assert_eq!(
            *manager.calls.lock(),
            [
                "start:event".to_owned(),
                "start:policy".to_owned(),
                "active:event".to_owned(),
                "active:policy".to_owned(),
            ]
        );
    }

    /// Required roster rollback: a later child failing must stop earlier
    /// children in reverse order — proven with real supervised processes via
    /// the injected-plan seam.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[tokio::test]
    async fn required_rollback_stops_started_children_on_later_failure() -> anyhow::Result<()> {
        use hyprstream_service::{ProcessConfig, ProcessReadiness, ProcessSpawner};
        #[allow(unused_imports)]
        use ProcessReadiness as _ProcessReadinessMarker;

        const ROLLBACK_HELPER: &str = "HYPRSTREAM_LAUNCHER_ROLLBACK_HELPER";

        // Helper child used as the first plan: READY then stays alive.
        if std::env::var_os(ROLLBACK_HELPER).is_some() {
            hyprstream_rpc::notify::ready()?;
            std::thread::sleep(std::time::Duration::from_secs(30));
            return Ok(());
        }

        let exe = std::env::current_exe()?;
        let spawner = ProcessSpawner::standalone();
        let supervisor_a = "rollback-alive";
        let supervisor_b = "rollback-fail";

        let plans = vec![vec![
            (
                supervisor_a.to_owned(),
                ProcessConfig::new(supervisor_a, &exe)
                    .args([
                        "--exact",
                        "cli::service_handlers::launcher_tests::required_rollback_stops_started_children_on_later_failure",
                        "--nocapture",
                    ])
                    .env(ROLLBACK_HELPER, "1")
                    .with_notify_ready(std::time::Duration::from_secs(30)),
            ),
            (
                supervisor_b.to_owned(),
                ProcessConfig::new(supervisor_b, Path::new("/bin/sh"))
                    .args(["-c", "exit 7"])
                    .with_notify_ready(std::time::Duration::from_secs(10)),
            ),
        ]];

        let error = launch_planned_children(plans, true, &spawner)
            .await
            .expect_err("later-child failure must fail the launch");
        assert!(
            error.to_string().contains("exited during startup"),
            "launch failure expected, got: {error}"
        );

        // The alive first child was stopped by the rollback (reverse order):
        // its PID file was removed and the process is reaped. The failing
        // child never published anything.
        assert!(
            !hyprstream_rpc::paths::service_pid_file(supervisor_a).exists(),
            "rolled-back child must leave no PID file"
        );
        assert!(
            !hyprstream_rpc::paths::service_pid_file(supervisor_b).exists(),
            "failed child must leave no PID file"
        );
        Ok(())
    }

    /// Multi-stage required launch: a failure in one stage must abort the
    /// WHOLE launch — later dependent stages must never spawn — and every
    /// already-started child must be genuinely reaped.
    ///
    /// Fixture observes, per root dispatch `pr1585-launch-stage-failure-root.md`:
    /// two live predecessors (each records its own PID to a marker file, so
    /// termination is confirmed by `kill(pid, 0)` → ESRCH, not by PID-file
    /// absence alone), the original launch error surfaced unchanged, and a
    /// sentinel in a LATER stage whose spawn marker must never appear (the
    /// single-stage test cannot distinguish a full abort from a per-stage
    /// break). Stop ORDER is not externally observable here: rollback kills
    /// are SIGKILL, so reverse ordering remains a construction guarantee of
    /// `started.iter().rev()`, not an observed event.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[tokio::test]
    async fn required_stage_failure_aborts_launch_and_never_spawns_later_stages()
    -> anyhow::Result<()> {
        use hyprstream_service::{ProcessConfig, ProcessReadiness, ProcessSpawner};
        #[allow(unused_imports)]
        use ProcessReadiness as _ProcessReadinessMarker;

        const MARKER_HELPER: &str = "HYPRSTREAM_LAUNCHER_MARKER_HELPER";

        // Helper child used as a live predecessor: records its PID to the
        // marker BEFORE its READY send, so the parent's continuation past
        // this child deterministically proves the marker exists.
        if let Some(path) = std::env::var_os(MARKER_HELPER) {
            std::fs::write(&path, std::process::id().to_string())?;
            hyprstream_rpc::notify::ready()?;
            std::thread::sleep(std::time::Duration::from_secs(30));
            return Ok(());
        }

        let exe = std::env::current_exe()?;
        let spawner = ProcessSpawner::standalone();
        let dir = tempfile::tempdir()?;
        // Unique supervisor names per test process: PID files land in the
        // shared runtime dir, and libtest runs suites in parallel.
        let unique = std::process::id();
        let stage0a = format!("launch-stage-a-{unique}");
        let stage0b = format!("launch-stage-b-{unique}");
        let stage1fail = format!("launch-stage-fail-{unique}");
        let stage2sentinel = format!("launch-stage-sentinel-{unique}");

        let marker = |name: &str| dir.path().join(format!("{name}.marker"));
        let marker_pid = |path: &Path| -> anyhow::Result<u32> {
            std::fs::read_to_string(path)?
                .trim()
                .parse::<u32>()
                .context("marker must hold the child pid")
        };
        // Reap confirmation: ESRCH, not merely an absent PID file.
        let assert_reaped = |pid: u32| {
            let gone = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
                None,
            )
            .is_err_and(|e| e == nix::errno::Errno::ESRCH);
            assert!(gone, "child pid {pid} must be reaped after rollback");
        };

        let plans = vec![
            // Stage 0: two live predecessors (marker→READY→stay alive) —
            // both must be rolled back and genuinely reaped.
            vec![
                (
                    stage0a.clone(),
                    ProcessConfig::new(&stage0a, &exe)
                        .args([
                            "--exact",
                            "cli::service_handlers::launcher_tests::required_stage_failure_aborts_launch_and_never_spawns_later_stages",
                            "--nocapture",
                        ])
                        .env(MARKER_HELPER, marker(&stage0a).display().to_string())
                        .with_notify_ready(std::time::Duration::from_secs(30)),
                ),
                (
                    stage0b.clone(),
                    ProcessConfig::new(&stage0b, &exe)
                        .args([
                            "--exact",
                            "cli::service_handlers::launcher_tests::required_stage_failure_aborts_launch_and_never_spawns_later_stages",
                            "--nocapture",
                        ])
                        .env(MARKER_HELPER, marker(&stage0b).display().to_string())
                        .with_notify_ready(std::time::Duration::from_secs(30)),
                ),
            ],
            // Stage 1: required child fails during startup.
            vec![(
                stage1fail.clone(),
                ProcessConfig::new(&stage1fail, Path::new("/bin/sh"))
                    .args(["-c", "exit 7"])
                    .with_notify_ready(std::time::Duration::from_secs(10)),
            )],
            // Stage 2 sentinel: same marker-before-READY helper with Notify
            // readiness. If the abort ever leaked past the failing stage, the
            // buggy path would WAIT for this child's READY — which the helper
            // sends only after writing the marker — so an incorrect later-stage
            // spawn deterministically leaves a marker (an Immediate child could
            // be SIGKILLed by the buggy rollback before it ever wrote one,
            // giving a false pass).
            vec![(
                stage2sentinel.clone(),
                ProcessConfig::new(&stage2sentinel, &exe)
                    .args([
                        "--exact",
                        "cli::service_handlers::launcher_tests::required_stage_failure_aborts_launch_and_never_spawns_later_stages",
                        "--nocapture",
                    ])
                    .env(MARKER_HELPER, marker(&stage2sentinel).display().to_string())
                    .with_notify_ready(std::time::Duration::from_secs(30)),
            )],
        ];

        let error = launch_planned_children(plans, true, &spawner)
            .await
            .expect_err("stage-1 failure must fail the whole launch");
        assert!(
            error.to_string().contains("exited during startup"),
            "original launch error expected, got: {error}"
        );

        // Both started predecessors: actually reaped (ESRCH), PID files gone.
        for (name, path) in [(&stage0a, marker(&stage0a)), (&stage0b, marker(&stage0b))] {
            assert!(path.exists(), "predecessor {name} must have run");
            assert_reaped(marker_pid(&path)?);
            assert!(
                !hyprstream_rpc::paths::service_pid_file(name).exists(),
                "rolled-back {name} must leave no PID file"
            );
        }
        // The later-stage sentinel never spawned.
        assert!(
            !marker(&stage2sentinel).exists(),
            "sentinel stage must never spawn after a required stage failure"
        );
        assert!(
            !hyprstream_rpc::paths::service_pid_file(&stage2sentinel).exists(),
            "sentinel must publish no PID file"
        );
        Ok(())
    }

    /// Cancellation while a later Required child is pending must retain
    /// cleanup ownership of the already-adopted READY children: aborting the
    /// orchestration future (the real cancellation boundary — `BootstrapManager::drop`
    /// aborts this exact task shape) reaps the adopted child through the
    /// backend's retained handle and removes its PID artifact. The later
    /// child is proven genuinely pending: its marker proves it spawned (so
    /// the first child was adopted), and it never sends READY.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[tokio::test]
    async fn required_cancellation_reaps_adopted_children_and_artifacts() -> anyhow::Result<()> {
        use hyprstream_service::{ProcessConfig, ProcessReadiness, ProcessSpawner};
        #[allow(unused_imports)]
        use ProcessReadiness as _ProcessReadinessMarker;

        const HELPER: &str = "HYPRSTREAM_LAUNCHER_CANCEL_HELPER";
        const PENDING: &str = "HYPRSTREAM_LAUNCHER_CANCEL_PENDING";

        // Helper child: writes its PID marker first; the READY variant then
        // sends READY and stays alive, the pending variant never sends READY.
        if let Some(path) = std::env::var_os(HELPER) {
            std::fs::write(&path, std::process::id().to_string())?;
            if std::env::var_os(PENDING).is_none() {
                hyprstream_rpc::notify::ready()?;
            }
            std::thread::sleep(std::time::Duration::from_secs(30));
            return Ok(());
        }

        let exe = std::env::current_exe()?;
        let dir = tempfile::tempdir()?;
        let unique = std::process::id();
        let ready_name = format!("cancel-ready-{unique}");
        let pending_name = format!("cancel-pending-{unique}");
        let marker = |name: &str| dir.path().join(format!("{name}.marker"));
        let marker_pid = |path: &Path| -> anyhow::Result<u32> {
            std::fs::read_to_string(path)?
                .trim()
                .parse::<u32>()
                .context("marker must hold the child pid")
        };
        let assert_reaped = |pid: u32| {
            let gone = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
                None,
            )
            .is_err_and(|e| e == nix::errno::Errno::ESRCH);
            assert!(gone, "cancelled child pid {pid} must be genuinely reaped");
        };
        let helper_cfg = |name: &str, pending: bool| {
            let mut config = ProcessConfig::new(name, &exe)
                .args([
                    "--exact",
                    "cli::service_handlers::launcher_tests::required_cancellation_reaps_adopted_children_and_artifacts",
                    "--nocapture",
                ])
                .env(HELPER, marker(name).display().to_string())
                .with_notify_ready(std::time::Duration::from_secs(30));
            if pending {
                config = config.env(PENDING, "1");
            }
            config
        };
        let plans = vec![vec![
            (ready_name.clone(), helper_cfg(&ready_name, false)),
            (pending_name.clone(), helper_cfg(&pending_name, true)),
        ]];

        // The spawner lives inside the task exactly as in production
        // (`handle_service_start` owns it), so cancellation also drops the
        // backend while the guard's synchronous cleanup runs.
        let task = tokio::spawn(async move {
            let spawner = ProcessSpawner::standalone();
            launch_planned_children(plans, true, &spawner).await
        });

        // Bounded wait for the cancellation point: the pending child's
        // marker proves the launcher adopted the READY first child and is
        // now awaiting the second child's readiness.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while !marker(&ready_name).exists() || !marker(&pending_name).exists() {
            anyhow::ensure!(
                std::time::Instant::now() < deadline,
                "launch never reached the pending second child"
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        task.abort();
        let joined = task.await;
        assert!(
            matches!(&joined, Err(join) if join.is_cancelled()),
            "the orchestration future must have been cancelled, got: {joined:?}"
        );

        // Both children reaped (ESRCH — observed termination, not an
        // artifact absence) and no PID artifacts remain: the adopted READY
        // child through the transaction guard's synchronous tracked-child
        // pass, the pending child through its own armed startup guard.
        assert_reaped(marker_pid(&marker(&ready_name))?);
        assert_reaped(marker_pid(&marker(&pending_name))?);
        assert!(
            !hyprstream_rpc::paths::service_pid_file(&ready_name).exists(),
            "cancelled READY child must leave no PID artifact"
        );
        assert!(
            !hyprstream_rpc::paths::service_pid_file(&pending_name).exists(),
            "cancelled pending child must leave no PID artifact"
        );
        Ok(())
    }

    /// Whole-launch success must disarm the transaction guard: adopted
    /// daemons intentionally survive — both a normal launcher/spawner drop
    /// and later `stop` through the established contract. (Pair A proves
    /// stoppability through the launching backend's tracked stop; pair C is
    /// dropped with its spawner and must still be running afterwards.)
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[tokio::test]
    async fn required_success_commit_leaves_adopted_daemons_alive_and_stoppable()
    -> anyhow::Result<()> {
        use hyprstream_service::{
            ProcessConfig, ProcessKind, ProcessReadiness, ProcessSpawner, SpawnedProcess,
        };
        #[allow(unused_imports)]
        use ProcessReadiness as _ProcessReadinessMarker;

        const HELPER: &str = "HYPRSTREAM_LAUNCHER_COMMIT_HELPER";

        if let Some(path) = std::env::var_os(HELPER) {
            std::fs::write(&path, std::process::id().to_string())?;
            hyprstream_rpc::notify::ready()?;
            std::thread::sleep(std::time::Duration::from_secs(30));
            return Ok(());
        }

        let exe = std::env::current_exe()?;
        let dir = tempfile::tempdir()?;
        let unique = std::process::id();
        let marker = |name: &str| dir.path().join(format!("{name}.marker"));
        let marker_pid = |path: &Path| -> anyhow::Result<u32> {
            std::fs::read_to_string(path)?
                .trim()
                .parse::<u32>()
                .context("marker must hold the child pid")
        };
        let alive = |pid: u32| {
            nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), None).is_ok()
        };
        let helper_cfg = |name: &str| {
            ProcessConfig::new(name, &exe)
                .args([
                    "--exact",
                    "cli::service_handlers::launcher_tests::required_success_commit_leaves_adopted_daemons_alive_and_stoppable",
                    "--nocapture",
                ])
                .env(HELPER, marker(name).display().to_string())
                .with_notify_ready(std::time::Duration::from_secs(30))
        };

        // Pair A: committed, then stopped through the launching backend.
        let spawner = ProcessSpawner::standalone();
        let a = format!("commit-a-{unique}");
        let b = format!("commit-b-{unique}");
        launch_planned_children(
            vec![vec![(a.clone(), helper_cfg(&a)), (b.clone(), helper_cfg(&b))]],
            true,
            &spawner,
        )
        .await
        .expect("whole-launch success must commit");
        // Bounded wait until both helpers recorded their PIDs (the launch
        // only returns after both are READY, so the markers already exist).
        for name in [&a, &b] {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            while !marker(name).exists() {
                anyhow::ensure!(
                    std::time::Instant::now() < deadline,
                    "committed child {name} never recorded its PID"
                );
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            }
        }
        // Stop through the launching backend's established tracked contract.
        for (name, pid) in [(&a, marker_pid(&marker(&a))?), (&b, marker_pid(&marker(&b))?)] {
            assert!(alive(pid), "committed daemon {name} must still be running");
            let meta = SpawnedProcess::new(format!("{name}-{pid}"), ProcessKind::Direct(pid))
                .with_pid_file(hyprstream_rpc::paths::service_pid_file(name));
            spawner
                .stop(&meta)
                .await
                .expect("committed daemon must remain stoppable");
            assert!(
                nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), None)
                    .is_err_and(|e| e == nix::errno::Errno::ESRCH),
                "stopped committed daemon {name} must be reaped"
            );
            assert!(
                !hyprstream_rpc::paths::service_pid_file(name).exists(),
                "stopped committed daemon {name} must leave no PID artifact"
            );
        }

        // Pair C: committed, then the launching spawner is dropped normally —
        // the adopted daemons must survive it (kill_on_drop stays false).
        let c = format!("commit-c-{unique}");
        let d = format!("commit-d-{unique}");
        let drop_spawner = ProcessSpawner::standalone();
        launch_planned_children(
            vec![vec![(c.clone(), helper_cfg(&c)), (d.clone(), helper_cfg(&d))]],
            true,
            &drop_spawner,
        )
        .await
        .expect("whole-launch success must commit");
        let dropped_pids = [marker_pid(&marker(&c))?, marker_pid(&marker(&d))?];
        drop(drop_spawner);
        for (name, pid) in [(&c, dropped_pids[0]), (&d, dropped_pids[1])] {
            assert!(
                alive(pid),
                "committed daemon {name} must survive a normal launcher drop"
            );
        }
        // Fixture hygiene: reap the surviving pair directly (the test is the
        // parent); not a product-contract assertion. Afterwards remove only
        // the matching fixture-owned PID artifacts and establish their
        // absence, so a passing test leaves no stale files; any replacement
        // artifact would be preserved.
        for (name, pid) in [(&c, dropped_pids[0]), (&d, dropped_pids[1])] {
            let raw = nix::unistd::Pid::from_raw(pid as i32);
            let _ = nix::sys::signal::kill(raw, nix::sys::signal::Signal::SIGKILL);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            loop {
                match nix::sys::wait::waitpid(raw, Some(nix::sys::wait::WaitPidFlag::WNOHANG)) {
                    Ok(status) if status != nix::sys::wait::WaitStatus::StillAlive => break,
                    Err(_) => break,
                    Ok(_) => {}
                }
                anyhow::ensure!(
                    std::time::Instant::now() < deadline,
                    "fixture child {pid} could not be reaped"
                );
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            let pid_file = hyprstream_rpc::paths::service_pid_file(name);
            if let Ok(content) = std::fs::read_to_string(&pid_file) {
                if content.trim() == pid.to_string() {
                    let _ = std::fs::remove_file(&pid_file);
                }
            }
            assert!(
                !pid_file.exists(),
                "fixture hygiene: {name} must leave no stale PID artifact"
            );
        }
        Ok(())
    }

    /// Cancellation around a required startup failure cleans the spawned
    /// predecessor at the point where abort lands. Its PID marker is written
    /// before READY, so marker presence proves the child ran and wrote its
    /// PID; it does not prove READY was sent, that the launcher adopted it,
    /// or that rollback was entered. The landing is unspecified. The
    /// deterministic rollback-boundary evidence is provided by
    /// `rollback_failed_and_pending_stops_stay_guard_owned_through_cancellation`.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[tokio::test]
    async fn required_cancellation_around_startup_failure_cleans_predecessor_at_any_landing()
    -> anyhow::Result<()> {
        use hyprstream_service::{ProcessConfig, ProcessReadiness, ProcessSpawner};
        #[allow(unused_imports)]
        use ProcessReadiness as _ProcessReadinessMarker;

        const HELPER: &str = "HYPRSTREAM_LAUNCHER_CANCEL_ROLLBACK_HELPER";

        if let Some(path) = std::env::var_os(HELPER) {
            std::fs::write(&path, std::process::id().to_string())?;
            hyprstream_rpc::notify::ready()?;
            std::thread::sleep(std::time::Duration::from_secs(30));
            return Ok(());
        }

        let exe = std::env::current_exe()?;
        let dir = tempfile::tempdir()?;
        let unique = std::process::id();
        let ready_name = format!("cancel-rollback-ready-{unique}");
        let fail_name = format!("cancel-rollback-fail-{unique}");
        let marker = dir.path().join(format!("{ready_name}.marker"));
        let marker_pid = |path: &Path| -> anyhow::Result<u32> {
            std::fs::read_to_string(path)?
                .trim()
                .parse::<u32>()
                .context("marker must hold the child pid")
        };

        let plans = vec![vec![
            (
                ready_name.clone(),
                ProcessConfig::new(&ready_name, &exe)
                    .args([
                        "--exact",
                        "cli::service_handlers::launcher_tests::required_cancellation_around_startup_failure_cleans_predecessor_at_any_landing",
                        "--nocapture",
                    ])
                    .env(HELPER, marker.display().to_string())
                    .with_notify_ready(std::time::Duration::from_secs(30)),
            ),
            (
                fail_name.clone(),
                ProcessConfig::new(&fail_name, Path::new("/bin/sh"))
                    .args(["-c", "exit 7"])
                    .with_notify_ready(std::time::Duration::from_secs(10)),
            ),
        ]];

        let task = tokio::spawn(async move {
            let spawner = ProcessSpawner::standalone();
            launch_planned_children(plans, true, &spawner).await
        });

        // Abort after the predecessor's PID marker appears. The marker
        // precedes READY; it does not prove notification, adoption, or
        // rollback entry. This test checks cleanup at the observed,
        // unspecified cancellation point.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
        while !marker.exists() {
            anyhow::ensure!(
                std::time::Instant::now() < deadline,
                "READY predecessor never recorded its PID"
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        task.abort();
        let joined = task.await;
        // Either the future was cancelled (cancellation cleanup ran) or the
        // rollback completed before the abort landed (explicit-failure
        // rollback ran) — both paths must end with the predecessor cleaned.
        match &joined {
            Err(join) => assert!(join.is_cancelled(), "unexpected join error: {join:?}"),
            Ok(Err(error)) => assert!(
                error.to_string().contains("exited during startup"),
                "completed rollback must carry the launch error, got: {error}"
            ),
            Ok(Ok(())) => panic!("a required launch with a failing child must not succeed"),
        }
        let pid = marker_pid(&marker)?;
        let gone = nix::sys::signal::kill(nix::unistd::Pid::from_raw(pid as i32), None)
            .is_err_and(|e| e == nix::errno::Errno::ESRCH);
        assert!(gone, "predecessor pid {pid} must be cleaned at every landing point");
        assert!(
            !hyprstream_rpc::paths::service_pid_file(&ready_name).exists(),
            "predecessor must leave no PID artifact"
        );
        Ok(())
    }

    /// Rollback ownership-boundary regression (root-requested): a stop that
    /// FAILS must leave its child guard-owned — never moved into a local
    /// unguarded collection — and a cancellation while a LATER stop is still
    /// PENDING must attempt synchronous cleanup of BOTH. The failure/pending
    /// boundary is injected at the narrow `rollback_owned_children` stop seam; the
    /// two children are REAL spawned processes, so ownership and cleanup are
    /// evidenced by observed reaps (ESRCH) and PID-artifact removal.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[tokio::test]
    async fn rollback_failed_and_pending_stops_stay_guard_owned_through_cancellation()
    -> anyhow::Result<()> {
        use hyprstream_service::{ProcessConfig, ProcessSpawner};
        use std::sync::atomic::{AtomicUsize, Ordering};

        const HELPER: &str = "HYPRSTREAM_LAUNCHER_RBOWN_HELPER";

        if let Some(path) = std::env::var_os(HELPER) {
            std::fs::write(&path, std::process::id().to_string())?;
            hyprstream_rpc::notify::ready()?;
            std::thread::sleep(std::time::Duration::from_secs(30));
            return Ok(());
        }

        let exe = std::env::current_exe()?;
        let dir = tempfile::tempdir()?;
        let unique = std::process::id();
        let older = format!("rbown-older-{unique}");
        let newer = format!("rbown-newer-{unique}");
        let marker = |name: &str| dir.path().join(format!("{name}.marker"));
        let marker_pid = |path: &Path| -> anyhow::Result<u32> {
            std::fs::read_to_string(path)?
                .trim()
                .parse::<u32>()
                .context("marker must hold the child pid")
        };
        let assert_reaped = |pid: u32| {
            let gone = nix::sys::signal::kill(
                nix::unistd::Pid::from_raw(pid as i32),
                None,
            )
            .is_err_and(|e| e == nix::errno::Errno::ESRCH);
            assert!(gone, "guard-owned child pid {pid} must be genuinely reaped");
        };
        let helper_cfg = |name: &str| {
            ProcessConfig::new(name, &exe)
                .args([
                    "--exact",
                    "cli::service_handlers::launcher_tests::rollback_failed_and_pending_stops_stay_guard_owned_through_cancellation",
                    "--nocapture",
                ])
                .env(HELPER, marker(name).display().to_string())
                .with_notify_ready(std::time::Duration::from_secs(30))
        };

        let spawner = ProcessSpawner::standalone();
        let mut guard = RequiredLaunchGuard::arm(true, &spawner);
        // Real adoption, start order older → newer; the rollback walks
        // newest first, so the NEWER child hits the injected failed stop and
        // the OLDER child the injected pending stop.
        for name in [&older, &newer] {
            let process = spawner.spawn(helper_cfg(name)).await?;
            guard.adopt(name.to_owned(), process);
        }

        let calls = std::sync::Arc::new(AtomicUsize::new(0));
        let boundary: std::sync::Arc<parking_lot::Mutex<Vec<&'static str>>> =
            std::sync::Arc::default();
        let calls_task = calls.clone();
        let boundary_task = boundary.clone();
        let task = tokio::spawn(async move {
            guard
                .rollback_owned_children(move |_process| {
                    let call = calls_task.fetch_add(1, Ordering::SeqCst);
                    let boundary = boundary_task.clone();
                    async move {
                        if call == 0 {
                            boundary.lock().push("first-stop-failed");
                            Err(anyhow::anyhow!("injected stop failure"))
                        } else {
                            boundary.lock().push("next-stop-pending-entered");
                            // Pending forever: only the task abort below
                            // ends this await.
                            std::future::pending::<()>().await;
                            Ok(())
                        }
                    }
                })
                .await;
            // Unreachable under the abort; the guard's Drop owns cleanup.
        });

        // Deterministic boundary, no sleep-only timing assumption: abort
        // only after BOTH stop invocations happened — the first returned the
        // injected failure (its child demoted to guard-owned `failed`) and
        // the second is suspended inside the pending stop.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while calls.load(Ordering::SeqCst) < 2 {
            anyhow::ensure!(
                std::time::Instant::now() < deadline,
                "rollback never reached the pending second stop"
            );
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        task.abort();
        let joined = task.await;
        assert!(
            matches!(&joined, Err(join) if join.is_cancelled()),
            "the rollback future must have been cancelled, got: {joined:?}"
        );
        // The claimed causal boundary happened: the first stop FAILED and
        // the next stop was entered and PENDING at cancellation time.
        assert_eq!(
            *boundary.lock(),
            vec!["first-stop-failed", "next-stop-pending-entered"],
            "the first stop must have failed and the next pending stop must have been \
             entered before cancellation"
        );

        // BOTH children — the failed-stop one and the pending-stop one — are
        // reaped with their PID artifacts removed by the guard's synchronous
        // pass through the backend's retained handles.
        assert_reaped(marker_pid(&marker(&newer))?);
        assert_reaped(marker_pid(&marker(&older))?);
        for name in [&older, &newer] {
            assert!(
                !hyprstream_rpc::paths::service_pid_file(name).exists(),
                "cancelled rollback child {name} must leave no PID artifact"
            );
        }
        Ok(())
    }

    /// Installed units encode their own stored configuration; an explicit
    /// config selector or the required-native profile must be refused on the
    /// existing-unit path before any unit is started.
    #[tokio::test]
    async fn installed_unit_start_rejects_explicit_config_and_required_profile() {
        if !hyprstream_rpc::has_systemd() {
            return;
        }
        let root = tempfile::tempdir().unwrap();
        let config_path = root.path().join("custom.toml");
        std::fs::write(&config_path, "[secrets]\n").unwrap();

        let explicit_error = handle_service_start(
            &["model".to_owned()],
            Some("model".to_owned()),
            false,
            Some(config_path.as_path()),
            false,
        )
        .await
        .expect_err("explicit config selector must not start an installed unit");
        assert!(
            explicit_error.to_string().contains("installed hyprstream service units"),
            "actionable rejection expected, got: {explicit_error}"
        );

        let required_error = handle_service_start(
            &["model".to_owned()],
            Some("model".to_owned()),
            false,
            None,
            true,
        )
        .await
        .expect_err("required-native profile must not start an installed unit");
        assert!(
            required_error.to_string().contains("--daemon"),
            "rejection must direct the operator to the direct launch path"
        );
    }
}
