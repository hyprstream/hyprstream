//! Systemd unit file generation
//!
//! Generates socket and service unit files for hyprstream services.

use anyhow::Result;

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
/// The service is started by socket activation and notifies systemd
/// when ready via sd_notify.
pub fn service_unit(service: &str) -> Result<String> {
    let exec = std::env::current_exe()?;
    Ok(format!(
        r#"[Unit]
Description=Hyprstream {service} Service
Requires=hyprstream-{service}.socket
After=hyprstream-{service}.socket

[Service]
Type=notify
ExecStart={exec} service {service} --ipc
Restart=on-failure

[Install]
WantedBy=default.target
"#,
        exec = exec.display()
    ))
}
