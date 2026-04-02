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

/// Node-level credentials — every service on this node needs these.
///
/// These are infrastructure secrets that define the node's identity:
/// signing key (root of trust), TLS materials (HTTP/QUIC transport),
/// RSA key (RS256 JWT signing for OIDC interop).
pub const NODE_CREDENTIAL_NAMES: &[&str] = &[
    "signing-key",
    "rsa-key",
    "tls-key",
    "tls-cert",
    "quic-key",
    "quic-cert",
];

/// Application-level credentials — only the oauth service needs these.
///
/// The credential-store key encrypts the user database (users.toml.age).
/// The user-signing key is used for Ed25519 challenge-response auth.
pub const OAUTH_CREDENTIAL_NAMES: &[&str] = &[
    "credential-store-key",
    "user-signing-key",
];

/// All managed credential names (node + application).
///
/// Used by `encrypt_credentials_if_available` to encrypt all secrets
/// from the secrets directory into the systemd credstore.
pub const ALL_CREDENTIAL_NAMES: &[&str] = &[
    "signing-key",
    "rsa-key",
    "credential-store-key",
    "user-signing-key",
    "tls-key",
    "tls-cert",
    "quic-key",
    "quic-cert",
];

/// Generate a systemd service unit for a service.
///
/// # Arguments
///
/// * `service` — service name (e.g., "policy", "oauth")
/// * `use_systemd_creds` — whether to emit `ImportCredential=` directives
/// * `depends_on` — service names that must start before this one
///
/// # Credential scoping
///
/// All services receive node-level credentials (signing key, TLS/QUIC
/// materials). Only the `oauth` service additionally receives application-
/// level secrets (credential-store key, user-signing key).
///
/// # Startup ordering
///
/// When `depends_on` is non-empty, the unit includes `After=` and
/// `Requires=` directives so systemd enforces the dependency graph.
pub fn service_unit(service: &str, use_systemd_creds: bool, depends_on: &[&str]) -> Result<String> {
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
    // actually exist in the credstore.
    let creds_section = if use_systemd_creds {
        let credstore = dirs::config_dir()
            .map(|d| d.join("credstore.encrypted"))
            .unwrap_or_default();

        // Node-level credentials for all services; add application-level
        // credentials for oauth only.
        let names: Vec<&&str> = if service == "oauth" {
            NODE_CREDENTIAL_NAMES.iter()
                .chain(OAUTH_CREDENTIAL_NAMES.iter())
                .collect()
        } else {
            NODE_CREDENTIAL_NAMES.iter().collect()
        };

        let import_lines: String = names
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

    // Build dependency directives for [Unit] section.
    let dep_section = if depends_on.is_empty() {
        String::new()
    } else {
        let units: Vec<String> = depends_on
            .iter()
            .map(|dep| format!("hyprstream-{dep}.service"))
            .collect();
        let unit_list = units.join(" ");
        format!("\nAfter={unit_list}\nRequires={unit_list}")
    };

    // Security hardening directives.
    let hardening = format!(
        "\nLimitCORE=0\nProtectProc=invisible\nProcSubset=pid{}",
        if is_appimage { "" } else { "\nPrivateTmp=yes" }
    );

    Ok(format!(
        r#"[Unit]
Description=Hyprstream {service} Service{dep_section}

[Service]
Type=notify
ExecStart={exec} service start {service} --foreground{env_section}{creds_section}{hardening}
Restart=on-failure

[Install]
WantedBy=default.target
"#,
        exec = exec.display(),
        dep_section = dep_section,
        env_section = env_section,
        creds_section = creds_section,
        hardening = hardening,
    ))
}
