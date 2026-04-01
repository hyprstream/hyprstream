//! Systemd unit file generation
//!
//! Generates socket and service unit files for hyprstream services.

use anyhow::{Context, Result};

use hyprstream_rpc::paths;

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

/// Secrets managed by systemd credentials (via `ImportCredential=`).
///
/// These are the five secrets that `hyprstream service install` encrypts into
/// `~/.config/credstore.encrypted/` using `systemd-creds encrypt`.  When the
/// service starts, systemd decrypts them into `$CREDENTIALS_DIRECTORY` and the
/// unit sets `HYPRSTREAM__SECRETS__PATH=%d` so hyprstream reads from there.
pub const SYSTEMD_CREDENTIAL_NAMES: &[&str] = &[
    "signing-key",
    "credential-store-key",
    "user-signing-key",
    "tls-key",
    "tls-cert",
    "quic-key",
    "quic-cert",
];

/// Generate a systemd service unit for a service.
///
/// When `use_systemd_creds` is `true`, the unit includes:
/// - `ImportCredential=<name>` for each of the five managed secrets
/// - `Environment=HYPRSTREAM__SECRETS__PATH=%d` to point hyprstream at the
///   systemd credentials directory (non-swappable ramfs, access-restricted)
/// - `PrivateMounts=yes` per systemd security recommendation for credential users
///
/// The service manages its own socket binding (via ZMQ) and notifies systemd
/// when ready via sd_notify.
///
/// Environment variables (LD_LIBRARY_PATH, LIBTORCH) are captured from
/// the process environment and forwarded to the service unit, **unless** the
/// target executable is an AppImage. AppImages bundle their own libtorch and
/// set up library paths via `AppRun` at launch time — hardcoding the mount path
/// from the installer process would produce a stale `/tmp/.mount_hyprst*/` path
/// that no longer exists when the service starts later.
///
/// Executable path priority for systemd units:
/// 1. Installed binary at `~/.local/bin/hyprstream` (stable, survives updates)
/// 2. `$APPIMAGE` path (when running from AppImage)
/// 3. `current_exe()` fallback
pub fn service_unit(service: &str, use_systemd_creds: bool) -> Result<String> {
    // Prefer installed binary for systemd units (stable location)
    let exec = paths::installed_executable_path()
        .map(Ok)
        .unwrap_or_else(paths::executable_path)
        .context("Failed to get executable path")?;

    // AppImages bundle libtorch and configure LD_LIBRARY_PATH themselves via
    // AppRun. Emitting LD_LIBRARY_PATH/LIBTORCH from the installer's environment
    // would hardcode a stale /tmp/.mount_hyprst*/ path that is invalid when
    // services start. Skip those vars if the target binary is an AppImage.
    let is_appimage = std::env::var("APPIMAGE").is_ok()
        || exec
            .extension()
            .map(|e| e.eq_ignore_ascii_case("appimage"))
            .unwrap_or(false)
        || exec
            .read_link()
            .ok()
            .and_then(|t| t.extension().map(|e| e.eq_ignore_ascii_case("appimage")))
            .unwrap_or(false);

    let hyprstream_instance = std::env::var("HYPRSTREAM_INSTANCE").ok();

    // Build Environment= directives
    let env_directives = if is_appimage {
        // AppImage manages its own libtorch — only forward instance namespace.
        vec![hyprstream_instance.map(|v| format!("Environment=HYPRSTREAM_INSTANCE={v}"))]
    } else {
        vec![
            std::env::var("LD_LIBRARY_PATH")
                .ok()
                .map(|v| format!("Environment=LD_LIBRARY_PATH={v}")),
            std::env::var("LIBTORCH")
                .ok()
                .map(|v| format!("Environment=LIBTORCH={v}")),
            hyprstream_instance.map(|v| format!("Environment=HYPRSTREAM_INSTANCE={v}")),
        ]
    }
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join("\n");

    let env_section = if env_directives.is_empty() {
        String::new()
    } else {
        format!("\n{env_directives}")
    };

    // Build systemd credentials section — only import credentials that
    // actually exist in the credstore (TLS/QUIC keys may not have been
    // generated at install time).
    let creds_section = if use_systemd_creds {
        let credstore = dirs::config_dir()
            .map(|d| d.join("credstore.encrypted"))
            .unwrap_or_default();
        let import_lines: String = SYSTEMD_CREDENTIAL_NAMES
            .iter()
            .filter(|name| credstore.join(name).exists())
            .map(|name| format!("ImportCredential={name}\n"))
            .collect();
        if import_lines.is_empty() {
            String::new()
        } else {
            // PrivateMounts=yes is recommended for credential users, but it
            // creates a private mount namespace that prevents AppImage FUSE
            // mounts.  Only enable it for non-AppImage executables.
            let private_mounts = if is_appimage { "" } else { "\nPrivateMounts=yes" };
            format!(
                "\n{}Environment=HYPRSTREAM__SECRETS__PATH=%d{private_mounts}",
                import_lines
            )
        }
    } else {
        String::new()
    };

    Ok(format!(
        r#"[Unit]
Description=Hyprstream {service} Service

[Service]
Type=notify
ExecStart={exec} service start {service} --foreground{env_section}{creds_section}
Restart=on-failure

[Install]
WantedBy=default.target
"#,
        exec = exec.display(),
        env_section = env_section,
        creds_section = creds_section,
    ))
}
