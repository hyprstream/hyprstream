//! Handlers for service management commands
//!
//! Provides lifecycle management for hyprstream services including
//! installation, upgrade, start/stop, and status display.

use anyhow::Result;
use tracing::info;

/// Handle `service install` - Install units and start services
pub async fn handle_service_install(
    config_services: &[String],
    services_filter: Option<Vec<String>>,
) -> Result<()> {
    let target_services = services_filter.unwrap_or_else(|| config_services.to_vec());

    if !hyprstream_rpc::has_systemd() {
        println!("Systemd not available. Services will be spawned directly when started.");
        return Ok(());
    }

    let manager = hyprstream_rpc::detect_service_manager().await?;

    println!("Installing hyprstream services...\n");

    // Show executable path being used
    if let Ok(appimage) = std::env::var("APPIMAGE") {
        println!("  Executable: {} (AppImage)", appimage);
    } else if let Ok(exe) = std::env::current_exe() {
        println!("  Executable: {}", exe.display());
    }
    println!();

    for service in &target_services {
        print!("  {} {}... ", "\u{25CB}", service);
        match manager.ensure(service).await {
            Ok(_) => println!("\u{2713}"),
            Err(e) => println!("\u{2717} {}", e),
        }
    }

    println!("\n\u{2713} Install complete");
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
        print!("  {} {}... ", "\u{25CB}", service);

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
        print!("    {} {}... ", "\u{25CB}", service);
        match manager.install(service).await {
            Ok(_) => println!("\u{2713}"),
            Err(e) => println!("\u{2717} {}", e),
        }
    }

    // Start all
    println!("  Starting services...");
    for service in &target_services {
        print!("    {} {}... ", "\u{25CB}", service);
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
        print!("  {} {}... ", "\u{25CB}", service);

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
            print!("  {} {}... ", "\u{25CB}", service);
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
            print!("  {} {}... ", "\u{25CB}", service);

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
            print!("  {} {} (systemd)... ", "\u{25CB}", service);
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
            print!("  {} {} (daemon)... ", "\u{25CB}", service);
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

    // Service status
    println!("\nService Status:");
    println!("  {:<15} {:<10} {}", "SERVICE", "STATUS", "NOTES");
    println!("  {}", "-".repeat(45));

    if hyprstream_rpc::has_systemd() {
        let manager = hyprstream_rpc::detect_service_manager().await?;

        for service in config_services {
            let (icon, status, notes) = match manager.is_active(service).await {
                Ok(true) => ("\u{2713}", "running", String::new()),
                Ok(false) => ("\u{25CB}", "stopped", String::new()),
                Err(e) => ("\u{2717}", "error", format!("{}", e)),
            };
            println!("  {:<15} {} {:<8} {}", service, icon, status, notes);
        }
    } else {
        // For standalone, check if processes are running
        for service in config_services {
            println!("  {:<15} {} {:<8}", service, "\u{2014}", "unknown");
        }
        println!("\n  Note: Process status detection not available in standalone mode");
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
