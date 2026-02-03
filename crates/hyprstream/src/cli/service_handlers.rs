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

/// Handle `service install` - Install units and start services
pub async fn handle_service_install(
    config_services: &[String],
    services_filter: Option<Vec<String>>,
) -> Result<()> {
    let target_services = services_filter.unwrap_or_else(|| config_services.to_vec());

    println!("Installing hyprstream...\n");

    // Show executable path being used
    if let Ok(appimage) = std::env::var("APPIMAGE") {
        println!("  Executable: {} (AppImage)", appimage);
    } else if let Ok(exe) = std::env::current_exe() {
        println!("  Executable: {}", exe.display());
    }
    println!();

    // Install command alias to user's executable directory
    println!("  Installing command...");
    match install_command_alias() {
        Ok((bin_dir, version_dir, is_appimage, updated_profiles)) => {
            let type_str = if is_appimage { "AppImage" } else { "binary" };
            println!("    {} ({})", version_dir.display(), type_str);
            println!(
                "    {} -> ...",
                bin_dir.join("hyprstream.appimage").display()
            );
            println!(
                "    {} -> hyprstream.appimage",
                bin_dir.join("hyprstream").display()
            );
            if !updated_profiles.is_empty() {
                println!("    PATH updated: {}", updated_profiles.join(" "));
            }
        }
        Err(e) => println!("    skipped ({})", e),
    }
    println!();

    // Install systemd units if available
    if hyprstream_rpc::has_systemd() {
        let manager = hyprstream_rpc::detect_service_manager().await?;

        println!("  Installing systemd units...");
        for service in &target_services {
            print!("    \u{25CB} {}... ", service);
            match manager.install(service).await {
                Ok(_) => println!("\u{2713}"),
                Err(e) => println!("\u{2717} {}", e),
            }
        }
    }

    println!("\n\u{2713} Install complete");
    println!();
    println!("Next steps:");
    println!("  1. Open a new shell to use the 'hyprstream' command");
    println!("  2. Start services with: hyprstream service start");
    println!("     Check status with:   hyprstream service status");
    Ok(())
}

/// Handle `service upgrade` - Reinstall units with current binary path
pub async fn handle_service_upgrade(
    config_services: &[String],
    services_filter: Option<Vec<String>>,
) -> Result<()> {
    let target_services = services_filter.unwrap_or_else(|| config_services.to_vec());

    if !hyprstream_rpc::has_systemd() {
        println!("Systemd not available. No units to upgrade.");
        return Ok(());
    }

    let manager = hyprstream_rpc::detect_service_manager().await?;

    println!("Upgrading hyprstream services...\n");

    if let Ok(appimage) = std::env::var("APPIMAGE") {
        println!("  New executable: {} (AppImage)", appimage);
    } else if let Ok(exe) = std::env::current_exe() {
        println!("  New executable: {}", exe.display());
    }
    println!();

    for service in &target_services {
        print!("  \u{25CB} {}... ", service);

        // Reinstall unit (will update ExecStart path)
        if let Err(e) = manager.install(service).await {
            println!("\u{2717} install failed: {}", e);
            continue;
        }

        // Stop and restart to pick up new binary
        let _ = manager.stop(service).await; // Ignore stop errors

        match manager.start(service).await {
            Ok(_) => println!("\u{2713}"),
            Err(e) => println!("\u{2717} start failed: {}", e),
        }
    }

    println!("\n\u{2713} Upgrade complete");
    Ok(())
}

/// Handle `service reinstall` - Full reset
pub async fn handle_service_reinstall(
    config_services: &[String],
    services_filter: Option<Vec<String>>,
) -> Result<()> {
    let target_services = services_filter.unwrap_or_else(|| config_services.to_vec());

    if !hyprstream_rpc::has_systemd() {
        println!("Systemd not available. No units to reinstall.");
        return Ok(());
    }

    let manager = hyprstream_rpc::detect_service_manager().await?;

    println!("Reinstalling hyprstream services (full reset)...\n");

    // Stop all first
    println!("  Stopping services...");
    for service in &target_services {
        let _ = manager.stop(service).await;
    }

    // Uninstall
    println!("  Removing unit files...");
    for service in &target_services {
        let _ = manager.uninstall(service).await;
    }

    // Reload daemon
    manager.reload().await?;

    // Reinstall
    println!("  Installing unit files...");
    for service in &target_services {
        print!("    \u{25CB} {}... ", service);
        match manager.install(service).await {
            Ok(_) => println!("\u{2713}"),
            Err(e) => println!("\u{2717} {}", e),
        }
    }

    // Start all
    println!("  Starting services...");
    for service in &target_services {
        print!("    \u{25CB} {}... ", service);
        match manager.start(service).await {
            Ok(_) => println!("\u{2713}"),
            Err(e) => println!("\u{2717} {}", e),
        }
    }

    println!("\n\u{2713} Reinstall complete");
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

    let manager = hyprstream_rpc::detect_service_manager().await?;

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
        let manager = hyprstream_rpc::detect_service_manager().await?;

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

        let spawner = hyprstream_rpc::ProcessSpawner::standalone();
        let exe = hyprstream_rpc::paths::executable_path()?;

        for service in &target_services {
            print!("  \u{25CB} {}... ", service);

            let config = hyprstream_rpc::ProcessConfig::new(service, &exe)
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
        let manager = hyprstream_rpc::detect_service_manager().await?;

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
        Some(hyprstream_rpc::detect_service_manager().await?)
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

// =============================================================================
// Version helpers
// =============================================================================

/// Get the full build version string
///
/// Format: `{cargo_version}+{branch}.g{sha7}[.dirty]`
/// Example: `0.1.0-alpha-7+main.gabc1234.dirty`
///
/// Uses BUILD_VERSION from build.rs, falls back to CARGO_PKG_VERSION.
fn build_version() -> &'static str {
    option_env!("BUILD_VERSION").unwrap_or(env!("CARGO_PKG_VERSION"))
}

// =============================================================================
// Command alias installation helpers
// =============================================================================

/// Install hyprstream binary/AppImage to versioned directory with symlinks
///
/// Structure:
/// ```text
/// ~/.local/share/hyprstream/versions/$VERSION/hyprstream[.appimage]
/// ~/.local/bin/hyprstream.appimage -> ../share/hyprstream/versions/$VERSION/hyprstream.appimage
/// ~/.local/bin/hyprstream -> hyprstream.appimage
/// ```
///
/// Returns (bin_dir, version_dir, is_appimage, updated_profiles) for output.
fn install_command_alias() -> Result<(PathBuf, PathBuf, bool, Vec<String>)> {
    let version = build_version();

    // Get directories using paths module
    let local_bin = hyprstream_rpc::paths::bin_dir()
        .ok_or_else(|| anyhow::anyhow!("Cannot determine user executable directory"))?;
    let ver_dir = hyprstream_rpc::paths::version_dir(version)
        .ok_or_else(|| anyhow::anyhow!("Cannot determine version directory"))?;

    std::fs::create_dir_all(&local_bin)
        .with_context(|| format!("Failed to create directory: {}", local_bin.display()))?;
    std::fs::create_dir_all(&ver_dir)
        .with_context(|| format!("Failed to create directory: {}", ver_dir.display()))?;

    // Get source: $APPIMAGE if available (stable path), otherwise current_exe()
    let (source, is_appimage) = if let Ok(appimage_env) = std::env::var("APPIMAGE") {
        (PathBuf::from(&appimage_env), true)
    } else {
        let exe = std::env::current_exe().context("Failed to get current executable path")?;
        let is_appimage = is_appimage_file(&exe).unwrap_or(false);
        (exe, is_appimage)
    };

    // Determine filenames based on type
    let filename = if is_appimage {
        "hyprstream.appimage"
    } else {
        "hyprstream"
    };

    // Copy binary to versioned directory
    let versioned_binary = ver_dir.join(filename);
    remove_if_exists(&versioned_binary)?;
    std::fs::copy(&source, &versioned_binary).with_context(|| {
        format!(
            "Failed to copy {} -> {}",
            source.display(),
            versioned_binary.display()
        )
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&versioned_binary, std::fs::Permissions::from_mode(0o755))?;
    }

    // Create symlinks in ~/.local/bin/
    let bin_appimage = local_bin.join("hyprstream.appimage");
    let bin_hyprstream = local_bin.join("hyprstream");

    // Calculate relative path from bin to versioned binary
    // ~/.local/bin/ -> ~/.local/share/hyprstream/versions/$VERSION/
    let relative_path = Path::new("..")
        .join("share")
        .join("hyprstream")
        .join("versions")
        .join(version)
        .join(filename);

    // Remove old symlinks/files
    remove_if_exists(&bin_appimage)?;
    remove_if_exists(&bin_hyprstream)?;

    // Create symlinks
    #[cfg(unix)]
    {
        // hyprstream.appimage -> ../share/hyprstream/versions/$VERSION/hyprstream[.appimage]
        std::os::unix::fs::symlink(&relative_path, &bin_appimage)
            .with_context(|| format!("Failed to create symlink: {}", bin_appimage.display()))?;

        // hyprstream -> hyprstream.appimage
        std::os::unix::fs::symlink(Path::new("hyprstream.appimage"), &bin_hyprstream)
            .with_context(|| format!("Failed to create symlink: {}", bin_hyprstream.display()))?;
    }

    // Update shell profiles for PATH
    let updated_profiles = if let Some(home) = dirs::home_dir() {
        update_shell_profiles(&home, &local_bin).unwrap_or_default()
    } else {
        Vec::new()
    };

    Ok((local_bin, ver_dir, is_appimage, updated_profiles))
}

/// Check if a file is an AppImage by reading its magic bytes
///
/// AppImage Type 2 has:
/// - ELF magic at offset 0: 0x7f 'E' 'L' 'F'
/// - AppImage magic at offset 8: 'A' 'I' 0x02
fn is_appimage_file(path: &Path) -> Result<bool> {
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
