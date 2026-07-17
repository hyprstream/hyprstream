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
/// These are infrastructure secrets shared across services:
/// signing-key (CA key, also used by PolicyService), RSA key, TLS materials.
/// Per-service signing keys are stored in per-service subdirectories and loaded
/// via SetLoadCredential.
pub const NODE_CREDENTIAL_NAMES: &[&str] = &[
    "rsa-key",
    "tls-key",
    "tls-cert",
    "quic-key",
    "quic-cert",
    "ca-pubkey",
    "bootstrap-pubkeys",
];

/// Per-service credentials — each service gets its own subdirectory.
///
/// Stored under `credentials/{service_name}/`:
/// - `signing-key` — service's own independent Ed25519 private key
///   (loaded via SetLoadCredential from per-service credstore subdirectory)
/// - `service-jwt` — CA-signed JWT certificate binding name → pubkey
///
/// Note: `signing-key` is also in NODE_CREDENTIAL_NAMES (flat credstore) as
/// the CA key for PolicyService. Non-policy services use SetLoadCredential
/// to override with their independent per-service key.
pub const SERVICE_CREDENTIAL_NAMES: &[&str] = &[
    "signing-key",
    "service-jwt",
];

/// Application-level credentials — only the oauth service needs these.
///
/// The credential-store key encrypts the user database (users.toml.age).
/// The user-signing key is used for Ed25519 challenge-response auth.
pub const OAUTH_CREDENTIAL_NAMES: &[&str] = &[
    "credential-store-key",
    "user-signing-key",
];

/// Policy-service-only credentials.
///
/// The CA private key is only available to PolicyService (the CA).
pub const POLICY_CREDENTIAL_NAMES: &[&str] = &[
    "ca-key",
];

/// All managed credential names (node + service + application).
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
    "ca-key",
    "ca-pubkey",
    "bootstrap-pubkeys",
    "service-jwt",
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

        // Node-level credentials for all services; add per-service credentials
        // from the service's subdirectory; add application-level credentials
        // for oauth; add CA private key for policy.
        let mut names: Vec<&'static str> = NODE_CREDENTIAL_NAMES.to_vec();
        names.extend(SERVICE_CREDENTIAL_NAMES.iter().copied());
        if service == "oauth" {
            names.extend(OAUTH_CREDENTIAL_NAMES.iter().copied());
        }
        if service == "policy" {
            names.extend(POLICY_CREDENTIAL_NAMES.iter().copied());
        }

        // Plaintext credential directory (wizard-written files):
        //   ~/.config/hyprstream/credentials/{service}/signing-key
        let plain_creds = dirs::config_dir()
            .map(|d| d.join("hyprstream").join("credentials"))
            .unwrap_or_default();

        let mut import_lines = String::new();

        for name in &names {
            // Per-service credentials (signing-key, service-jwt): each service
            // has its own independent Ed25519 key stored in a subdirectory.
            //
            // ImportCredential requires the credstore file name to match the
            // runtime name, but per-service creds are encrypted with prefixed
            // names (e.g. model-signing-key).  Use LoadCredential to load from
            // the plaintext file directly, making it available as the unprefixed
            // name the runtime expects.
            //
            // Policy is special: its "signing-key" IS the CA key, stored flat
            // in the encrypted credstore, so ImportCredential works for it.
            if SERVICE_CREDENTIAL_NAMES.contains(name) && service != "policy" {
                let plain_path = plain_creds.join(service).join(name);
                if plain_path.exists() {
                    import_lines.push_str(&format!(
                        "LoadCredential={name}:{}\n",
                        plain_path.display()
                    ));
                    continue;
                }
                // Fall through: no per-service plaintext file, try encrypted credstore
            }
            if credstore.join(name).exists() {
                import_lines.push_str(&format!("ImportCredential={name}\n"));
            }
        }

        let all_cred_lines = import_lines;

        if all_cred_lines.is_empty() {
            String::new()
        } else {
            // PrivateMounts=yes is recommended for credential users, but it
            // creates a private mount namespace that prevents AppImage FUSE
            // mounts.  Only enable it for non-AppImage executables.
            let private_mounts = if is_appimage { "" } else { "\nPrivateMounts=yes" };
            format!(
                "\n{}Environment=HYPRSTREAM__SECRETS__PATH=%d{private_mounts}",
                all_cred_lines
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
RestartSec=2s

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

/// Generate a systemd socket unit for system-wide (root) installation.
///
/// Socket listens on `/run/hyprstream/{service}.sock` and activates the
/// corresponding service unit on connection.
pub fn system_socket_unit(service: &str) -> String {
    format!(
        r#"[Unit]
Description=Hyprstream {service} Socket (system)

[Socket]
ListenStream=/run/hyprstream/{service}.sock
SocketMode=0660
SocketGroup=hyprstream

[Install]
WantedBy=sockets.target
"#
    )
}

/// Generate a systemd service unit for system-wide (root) installation.
///
/// The unit runs as the `hyprstream` system user and uses
/// `RuntimeDirectory=hyprstream` (→ `/run/hyprstream/`) for sockets.
/// Boots automatically via `WantedBy=multi-user.target`.
pub fn system_service_unit(service: &str, depends_on: &[&str]) -> Result<String> {
    let exec = paths::installed_executable_path()
        .map(Ok)
        .unwrap_or_else(paths::executable_path)
        .context("Failed to get executable path")?;

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

    let env_directives = if is_appimage {
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

    let hardening = format!(
        "\nLimitCORE=0\nProtectProc=invisible\nProcSubset=pid{}",
        if is_appimage { "" } else { "\nPrivateTmp=yes" }
    );

    Ok(format!(
        r#"[Unit]
Description=Hyprstream {service} Service (system){dep_section}

[Service]
Type=notify
User=hyprstream
RuntimeDirectory=hyprstream
RuntimeDirectoryMode=0750
ExecStart={exec} service start {service} --foreground{env_section}{hardening}
Restart=on-failure
RestartSec=2s

[Install]
WantedBy=multi-user.target
"#,
        exec = exec.display(),
    ))
}
