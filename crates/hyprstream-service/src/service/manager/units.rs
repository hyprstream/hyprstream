//! Systemd unit file generation
//!
//! Generates socket and service unit files for hyprstream services.

use anyhow::{anyhow, Context, Result};

use hyprstream_rpc::paths;

const SECRETS_PROFILE_ENV: &str = "HYPRSTREAM_SECRETS_PROFILE";
const PER_SERVICE_SCOPED_PROFILE: &str = "per-service-scoped";

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

/// Policy-service-only credentials.
///
/// The CA private key is only available to PolicyService (the CA).
pub const POLICY_CREDENTIAL_NAMES: &[&str] = &[
    "ca-key",
];

/// All managed credential names (node + service).
///
/// Used by `encrypt_credentials_if_available` to encrypt all secrets
/// from the secrets directory into the systemd credstore.
///
/// `user-signing-key` is deliberately absent: it is a CLI-side client
/// credential (challenge-response enrollment) with no service-side consumer,
/// so no unit imports it. The `credential-store-key` name was removed — it had
/// no producer or consumer anywhere in the tree (#808).
pub const ALL_CREDENTIAL_NAMES: &[&str] = &[
    "signing-key",
    "rsa-key",
    "tls-key",
    "tls-cert",
    "quic-key",
    "quic-cert",
    "ca-key",
    "ca-pubkey",
    "bootstrap-pubkeys",
    "service-jwt",
];

/// Build credential directives for a service unit.
///
/// This is the single, path-independent fail-closed guard shared by both the
/// user-unit (`service_unit`) and system-unit (`system_service_unit`) emission
/// paths. A non-policy service must have its own plaintext signing key
/// available for `LoadCredential=`. Falling through to
/// `ImportCredential=signing-key` would import the node/CA root key from the
/// flat credstore into that service.
fn credential_directives(
    service: &str,
    names: &[&str],
    credstore: &std::path::Path,
    plain_creds: &std::path::Path,
) -> Result<String> {
    let mut import_lines = String::new();

    for name in names {
        // Per-service credentials (signing-key, service-jwt): each service
        // has its own independent Ed25519 key stored in a subdirectory.
        //
        // ImportCredential requires the credstore file name to match the
        // runtime name, but per-service creds are encrypted with prefixed
        // names (e.g. model-signing-key). Use LoadCredential to load from
        // the plaintext file directly, making it available as the unprefixed
        // name the runtime expects.
        //
        // Policy is special: its "signing-key" IS the CA key, stored flat
        // in the encrypted credstore, so ImportCredential works for it.
        if SERVICE_CREDENTIAL_NAMES.contains(name) && service != "policy" {
            let plain_path = plain_creds.join(service).join(name);
            if plain_path.is_file() {
                import_lines.push_str(&format!(
                    "LoadCredential={name}:{}\n",
                    plain_path.display()
                ));
                continue;
            }
            if *name == "signing-key" {
                anyhow::bail!(
                    "per-service signing key '{}' is missing; refusing to import the node/CA root key as service '{}'. Re-run 'hyprstream service install' after provisioning service credentials",
                    plain_path.display(),
                    service
                );
            }
        }
        if credstore.join(name).exists() {
            import_lines.push_str(&format!("ImportCredential={name}\n"));
        }
    }

    Ok(import_lines)
}

/// Build the credential section for a system-wide service unit.
///
/// Kept separate from `system_service_unit` so the system path can be tested
/// with injectable paths while still routing every credential through the
/// shared [`credential_directives`] fail-closed guard.
fn system_creds_section(
    service: &str,
    is_appimage: bool,
    credstore: &std::path::Path,
    plain_creds: &std::path::Path,
) -> Result<String> {
    let mut names: Vec<&'static str> = NODE_CREDENTIAL_NAMES.to_vec();
    names.extend(SERVICE_CREDENTIAL_NAMES.iter().copied());
    if service == "policy" {
        names.extend(POLICY_CREDENTIAL_NAMES.iter().copied());
    }

    let import_lines = credential_directives(service, &names, credstore, plain_creds)?;

    Ok(if import_lines.is_empty() {
        String::new()
    } else {
        let private_mounts = if is_appimage {
            ""
        } else {
            "\nPrivateMounts=yes"
        };
        format!(
            "\n{import_lines}Environment=HYPRSTREAM__SECRETS__PATH=%d\n\
             Environment={SECRETS_PROFILE_ENV}={PER_SERVICE_SCOPED_PROFILE}{private_mounts}"
        )
    })
}

/// Instance-namespaced hyprstream config directory.
///
/// Mirrors the `StoragePaths` prefixing used by the bootstrap/config writers
/// (`$XDG_CONFIG_HOME/hyprstream/instances/{inst}` when `HYPRSTREAM_INSTANCE`
/// is set) so generated units reference the same credential paths that were
/// actually provisioned (#808).
///
/// Note: the systemd *credstore* itself (`$XDG_CONFIG_HOME/credstore.encrypted`)
/// is a fixed, systemd-owned location and is intentionally NOT namespaced.
fn validated_instance_name() -> Result<Option<String>> {
    match std::env::var("HYPRSTREAM_INSTANCE") {
        Ok(inst) if !inst.is_empty() => {
            if inst.contains('/') || inst.contains("..") {
                return Err(anyhow!(
                    "HYPRSTREAM_INSTANCE must not contain '/' or '..': {:?}",
                    inst
                ));
            }
            Ok(Some(inst))
        }
        _ => Ok(None),
    }
}

pub(crate) fn hyprstream_config_dir() -> Result<Option<std::path::PathBuf>> {
    let Some(base) = dirs::config_dir().map(|dir| dir.join("hyprstream")) else {
        return Ok(None);
    };
    Ok(match validated_instance_name()? {
        Some(inst) => Some(base.join("instances").join(inst)),
        None => Some(base),
    })
}

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
/// materials) plus their per-service credentials; only `policy` additionally
/// receives the CA private key.
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

    let hyprstream_instance = validated_instance_name()?;

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
        // from the service's subdirectory; add CA private key for policy.
        let mut names: Vec<&'static str> = NODE_CREDENTIAL_NAMES.to_vec();
        names.extend(SERVICE_CREDENTIAL_NAMES.iter().copied());
        if service == "policy" {
            names.extend(POLICY_CREDENTIAL_NAMES.iter().copied());
        }

        // Plaintext credential directory (wizard-written files), instance-
        // namespaced to match StoragePaths (#808):
        //   ~/.config/hyprstream[/instances/{inst}]/credentials/{service}/signing-key
        let plain_creds = hyprstream_config_dir()?
            .map(|d| d.join("credentials"))
            .unwrap_or_default();

        let all_cred_lines = credential_directives(service, &names, &credstore, &plain_creds)?;

        if all_cred_lines.is_empty() {
            String::new()
        } else {
            // PrivateMounts=yes is recommended for credential users, but it
            // creates a private mount namespace that prevents AppImage FUSE
            // mounts.  Only enable it for non-AppImage executables.
            let private_mounts = if is_appimage {
                ""
            } else {
                "\nPrivateMounts=yes"
            };
            format!(
                "\n{}Environment=HYPRSTREAM__SECRETS__PATH=%d\n\
                 Environment={SECRETS_PROFILE_ENV}={PER_SERVICE_SCOPED_PROFILE}{private_mounts}",
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
# Writable state home ($STATE_DIRECTORY) for JWT rotation state — the
# credentials ramfs ($CREDENTIALS_DIRECTORY) is read-only (#803).
StateDirectory=hyprstream
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

    let hyprstream_instance = validated_instance_name()?;

    let env_directives = if is_appimage {
        vec![hyprstream_instance
            .as_ref()
            .map(|v| format!("Environment=HYPRSTREAM_INSTANCE={v}"))]
    } else {
        vec![
            std::env::var("LD_LIBRARY_PATH")
                .ok()
                .map(|v| format!("Environment=LD_LIBRARY_PATH={v}")),
            std::env::var("LIBTORCH")
                .ok()
                .map(|v| format!("Environment=LIBTORCH={v}")),
            hyprstream_instance
                .as_ref()
                .map(|v| format!("Environment=HYPRSTREAM_INSTANCE={v}")),
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

    // System-wide credential plumbing (#808): mirrors the user-unit logic with
    // FHS system paths. `LoadCredential=`/`ImportCredential=` sources are read
    // by PID 1 (root) and passed to the service through the private, access-
    // restricted credentials ramfs ($CREDENTIALS_DIRECTORY), so the
    // `hyprstream` user never needs read access to the on-disk originals.
    let creds_section = {
        let credstore = std::path::PathBuf::from("/etc/credstore.encrypted");
        let plain_creds = match hyprstream_instance {
            Some(ref inst) => std::path::PathBuf::from("/etc/hyprstream")
                .join("instances")
                .join(inst)
                .join("credentials"),
            _ => std::path::PathBuf::from("/etc/hyprstream/credentials"),
        };

        system_creds_section(service, is_appimage, &credstore, &plain_creds)?
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
# Writable state home ($STATE_DIRECTORY) for JWT rotation state — the
# credentials ramfs ($CREDENTIALS_DIRECTORY) is read-only (#803).
StateDirectory=hyprstream
ExecStart={exec} service start {service} --foreground{env_section}{creds_section}{hardening}
Restart=on-failure
RestartSec=2s

[Install]
WantedBy=multi-user.target
"#,
        exec = exec.display(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    static INSTANCE_ENV_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    struct EnvVarGuard {
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn set(value: &str) -> Self {
            let previous = std::env::var("HYPRSTREAM_INSTANCE").ok();
            std::env::set_var("HYPRSTREAM_INSTANCE", value);
            Self { previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                Some(value) => std::env::set_var("HYPRSTREAM_INSTANCE", value),
                None => std::env::remove_var("HYPRSTREAM_INSTANCE"),
            }
        }
    }

    #[test]
    fn config_dir_rejects_instance_path_traversal() -> Result<()> {
        let _serial = INSTANCE_ENV_LOCK.lock();
        for invalid in ["../other", "nested/name", "name..suffix"] {
            let _instance = EnvVarGuard::set(invalid);
            let error = match hyprstream_config_dir() {
                Err(error) => error,
                Ok(path) => anyhow::bail!("invalid instance resolved to {path:?}"),
            };
            assert!(error.to_string().contains("must not contain '/' or '..'"));
        }
        Ok(())
    }

    #[test]
    fn missing_service_signing_key_never_falls_back_to_node_key() -> Result<()> {
        let temp = tempfile::TempDir::new()?;
        let credstore = temp.path().join("credstore.encrypted");
        let plain_creds = temp.path().join("credentials");
        std::fs::create_dir_all(&credstore)?;
        std::fs::create_dir_all(plain_creds.join("model"))?;

        // This is the dangerous fallback credential: the node/CA root key is
        // present under the runtime name expected by the service.
        std::fs::write(credstore.join("signing-key"), b"encrypted-root-key")?;

        let error = match credential_directives(
            "model",
            SERVICE_CREDENTIAL_NAMES,
            &credstore,
            &plain_creds,
        ) {
            Err(error) => error,
            Ok(lines) => anyhow::bail!("dangerous signing-key fallback returned: {lines}"),
        };
        let message = error.to_string();
        assert!(message.contains("per-service signing key"));
        assert!(message.contains("refusing to import the node/CA root key"));
        Ok(())
    }

    #[test]
    fn system_unit_missing_service_signing_key_never_falls_back_to_node_key() -> Result<()> {
        let temp = tempfile::TempDir::new()?;
        let credstore = temp.path().join("credstore.encrypted");
        let plain_creds = temp.path().join("credentials");
        std::fs::create_dir_all(&credstore)?;
        std::fs::create_dir_all(plain_creds.join("model"))?;

        // The dangerous fallback credential: node/CA root key present under
        // the flat runtime name a non-policy system service could import.
        std::fs::write(credstore.join("signing-key"), b"encrypted-root-key")?;

        let error = match system_creds_section("model", false, &credstore, &plain_creds) {
            Err(error) => error,
            Ok(section) => {
                anyhow::bail!("system-unit signing-key fallback returned: {section:?}")
            }
        };
        let message = error.to_string();
        assert!(message.contains("per-service signing key"));
        assert!(message.contains("refusing to import the node/CA root key"));
        Ok(())
    }

    #[test]
    fn system_unit_loads_present_per_service_key_without_root_fallback() -> Result<()> {
        let temp = tempfile::TempDir::new()?;
        let credstore = temp.path().join("credstore.encrypted");
        let plain_creds = temp.path().join("credentials");
        std::fs::create_dir_all(&credstore)?;
        std::fs::create_dir_all(plain_creds.join("model"))?;
        std::fs::write(credstore.join("signing-key"), b"encrypted-root-key")?;
        std::fs::write(plain_creds.join("model").join("signing-key"), b"svc-key")?;

        let section = system_creds_section("model", false, &credstore, &plain_creds)?;
        assert!(section.contains("LoadCredential=signing-key:"));
        assert!(section.contains("Environment=HYPRSTREAM_SECRETS_PROFILE=per-service-scoped"));
        assert!(
            !section.contains("ImportCredential=signing-key"),
            "must never import the flat node/CA root key for a non-policy service: {section}"
        );
        Ok(())
    }
}
