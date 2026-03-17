//! Handlers for service management commands
//!
//! Provides lifecycle management for hyprstream services including
//! installation, upgrade, start/stop, and status display.
// CLI handlers intentionally print to stdout/stderr for user interaction
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::info;

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
    verbose: bool,
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
        let manager = hyprstream_service::detect_service_manager().await?;

        // If --start, stop all target services first so they pick up changes
        if start {
            println!("  Stopping services...");
            for service in &target_services {
                let _ = manager.stop(service).await;
            }
        }

        println!("  Installing systemd units...");
        for service in &target_services {
            print!("    \u{25CB} {}... ", service);
            match manager.install(service).await {
                Ok(_) => println!("\u{2713}"),
                Err(e) => println!("\u{2717} {}", e),
            }
        }

        // 4. If --start, start all target services
        if start {
            println!("  Starting services...");
            for service in &target_services {
                print!("    \u{25CB} {}... ", service);
                match manager.start(service).await {
                    Ok(_) => println!("\u{2713}"),
                    Err(e) => println!("\u{2717} {}", e),
                }
            }
        }
    } else if start {
        // Standalone mode: use installed binary to spawn processes
        println!("  Starting services (standalone)...\n");

        let exe = hyprstream_rpc::paths::installed_executable_path()
            .unwrap_or_else(|| hyprstream_rpc::paths::executable_path().unwrap_or_default());

        let spawner = hyprstream_service::ProcessSpawner::standalone();

        for service in &target_services {
            print!("    \u{25CB} {}... ", service);

            let config = hyprstream_service::ProcessConfig::new(service, &exe)
                .args(["service", "start", service, "--foreground", "--ipc"]);

            match spawner.spawn(config).await {
                Ok(process) => {
                    info!("Spawned {} service: {:?}", service, process.kind);
                    println!("\u{2713}");
                }
                Err(e) => println!("\u{2717} {}", e),
            }
        }
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

/// Handle `service start` (non-foreground) - Start via systemd or spawn
pub async fn handle_service_start(
    config_services: &[String],
    name: Option<String>,
    daemon: bool,
) -> Result<()> {
    let target_services: Vec<String> = if let Some(name) = name {
        vec![name]
    } else {
        config_services.to_vec()
    };

    // Use systemd if available and --daemon not specified
    if hyprstream_rpc::has_systemd() && !daemon {
        let manager = hyprstream_service::detect_service_manager().await?;

        println!("Starting services (systemd)...\n");

        for service in &target_services {
            print!("  \u{25CB} {}... ", service);
            match manager.start(service).await {
                Ok(_) => println!("\u{2713}"),
                Err(e) => println!("\u{2717} {}", e),
            }
        }
    } else {
        // Standalone mode: spawn processes in background
        println!("Starting services (standalone)...\n");

        let spawner = hyprstream_service::ProcessSpawner::standalone();
        let exe = hyprstream_rpc::paths::executable_path()?;

        for service in &target_services {
            print!("  \u{25CB} {}... ", service);

            let config = hyprstream_service::ProcessConfig::new(service, &exe)
                .args(["service", "start", service, "--foreground", "--ipc"]);

            match spawner.spawn(config).await {
                Ok(process) => {
                    info!("Spawned {} service: {:?}", service, process.kind);
                    println!("\u{2713}");
                }
                Err(e) => println!("\u{2717} {}", e),
            }
        }
    }

    println!("\n\u{2713} Start complete");
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
pub(crate) async fn run_repair_checks(
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

    // 6. Policy active
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
                        warnings.push("Apply a template: hyprstream quick policy apply-template local".to_owned());
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

    Ok(())
}

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
    let (cert_der, _key_der) = quic_config.load_tls_materials()
        .context("Failed to load/generate QUIC TLS certificate")?;

    let hash = hyprstream_rpc::transport::zmtp_quic::cert_hash(&cert_der);
    println!("{}", hash);
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
