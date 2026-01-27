//! Systemd unit file generation
//!
//! Generates socket and service unit files for hyprstream services.

use anyhow::{Context, Result};

use crate::paths;

/// Generate a systemd socket unit for a service
///
/// The socket listens on `$XDG_RUNTIME_DIR/hyprstream/{service}.sock`
/// and activates the corresponding service unit on connection.
pub fn socket_unit(service: &str) -> String {
    format!(
        r#"[Unit]
Description=Hyprstream {service} Socket

[Socket]
ListenStream=%t/hyprstream/{service}.sock
SocketMode=0600

[Install]
WantedBy=sockets.target
"#
    )
}

/// Generate a systemd service unit for a service
///
/// The service manages its own socket binding (via ZMQ) and notifies systemd
/// when ready via sd_notify.
///
/// Environment variables (LD_LIBRARY_PATH, LIBTORCH) are captured from
/// the process environment and forwarded to the service unit.
///
/// Executable path priority for systemd units:
/// 1. Installed binary at `~/.local/bin/hyprstream` (stable, survives updates)
/// 2. `$APPIMAGE` path (when running from AppImage)
/// 3. `current_exe()` fallback
pub fn service_unit(service: &str) -> Result<String> {
    // Prefer installed binary for systemd units (stable location)
    let exec = paths::installed_executable_path()
        .map(Ok)
        .unwrap_or_else(|| paths::executable_path())
        .context("Failed to get executable path")?;

    // Capture environment variables to forward to the service
    let ld_library_path = std::env::var("LD_LIBRARY_PATH").ok();
    let libtorch = std::env::var("LIBTORCH").ok();

    // Build Environment= directives
    let env_directives = vec![
        ld_library_path.map(|v| format!("Environment=LD_LIBRARY_PATH={v}")),
        libtorch.map(|v| format!("Environment=LIBTORCH={v}")),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join("\n");

    let env_section = if env_directives.is_empty() {
        String::new()
    } else {
        format!("\n{env_directives}")
    };

    Ok(format!(
        r#"[Unit]
Description=Hyprstream {service} Service

[Service]
Type=notify
ExecStart={exec} service start {service} --foreground{env_section}
Restart=on-failure

[Install]
WantedBy=default.target
"#,
        exec = exec.display(),
        env_section = env_section
    ))
}
