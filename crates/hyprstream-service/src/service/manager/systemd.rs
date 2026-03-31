//! Systemd service manager implementation
//!
//! Uses D-Bus to control systemd units via zbus_systemd.

use super::{units, ServiceManager};
use crate::service::spawner::{Spawnable, SpawnedService};
use anyhow::{anyhow, Context as _, Result};
use async_trait::async_trait;
use std::path::PathBuf;
use tracing::{debug, info};
use zbus_systemd::login1::ManagerProxy as LoginProxy;
use zbus_systemd::systemd1::ManagerProxy;
use zbus_systemd::zbus::Connection;

/// Systemd-based service manager
///
/// Manages services via D-Bus calls to systemd.
pub struct SystemdManager {
    #[allow(dead_code)]
    connection: Connection,
    systemd: ManagerProxy<'static>,
    #[allow(dead_code)]
    login: LoginProxy<'static>,
}

impl SystemdManager {
    /// Create a new SystemdManager
    ///
    /// Connects to the user session D-Bus and enables user lingering
    /// so services persist after logout.
    pub async fn new() -> Result<Self> {
        let connection = Connection::session().await?;
        let systemd = ManagerProxy::new(&connection).await?;
        let login = LoginProxy::new(&connection).await?;

        // Enable linger (services persist after logout)
        let uid = nix::unistd::getuid().as_raw();
        if let Err(e) = login.set_user_linger(uid, true, false).await {
            debug!("Failed to enable user linger (may already be enabled): {}", e);
        } else {
            debug!("User lingering enabled");
        }

        Ok(Self {
            connection,
            systemd,
            login,
        })
    }

    #[allow(dead_code)]
    fn socket_unit(service: &str) -> String {
        format!("hyprstream-{service}.socket")
    }

    fn service_unit(service: &str) -> String {
        format!("hyprstream-{service}.service")
    }

    fn units_dir() -> Result<PathBuf> {
        dirs::config_dir()
            .map(|d| d.join("systemd/user"))
            .ok_or_else(|| anyhow!("cannot determine config directory"))
    }
}

#[async_trait]
impl ServiceManager for SystemdManager {
    async fn install(&self, service: &str) -> Result<()> {
        let units_dir = Self::units_dir()?;
        std::fs::create_dir_all(&units_dir)?;

        // Check whether credentials have already been encrypted into the systemd
        // credstore. The caller (handle_service_install in hyprstream) is expected
        // to call encrypt_credentials_if_available() BEFORE calling install(),
        // so by the time we generate the unit the .cred files should be present.
        let use_creds = credstore_has_credentials();

        let service_content = units::service_unit(service, use_creds)?;
        let service_path = units_dir.join(Self::service_unit(service));

        // Write service unit if changed (idempotent)
        if std::fs::read_to_string(&service_path).ok().as_deref() != Some(&service_content) {
            std::fs::write(&service_path, &service_content)?;
            info!("Installed service unit: {}", service_path.display());
            self.reload().await?;
        }

        Ok(())
    }

    async fn start(&self, service: &str) -> Result<()> {
        self.systemd
            .start_unit(Self::service_unit(service), "replace".into())
            .await?;
        info!("Started service unit: {}", Self::service_unit(service));
        Ok(())
    }

    async fn stop(&self, service: &str) -> Result<()> {
        self.systemd
            .stop_unit(Self::service_unit(service), "replace".into())
            .await?;
        info!("Stopped service unit: {}", Self::service_unit(service));
        Ok(())
    }

    async fn is_active(&self, service: &str) -> Result<bool> {
        // Check if the SERVICE unit is active, not the socket unit
        match self.systemd.get_unit(Self::service_unit(service)).await {
            Ok(unit_path) => {
                // Query the unit's ActiveState property via D-Bus
                let unit = zbus_systemd::systemd1::UnitProxy::builder(self.systemd.inner().connection())
                    .path(unit_path)?
                    .build()
                    .await?;
                let state = unit.active_state().await?;
                Ok(state == "active")
            }
            Err(_) => Ok(false),
        }
    }

    async fn reload(&self) -> Result<()> {
        self.systemd.reload().await?;
        debug!("Reloaded systemd daemon");
        Ok(())
    }

    async fn uninstall(&self, service: &str) -> Result<()> {
        // Stop the service first
        let _ = self.stop(service).await;

        // Remove service unit file
        let units_dir = Self::units_dir()?;
        let service_path = units_dir.join(Self::service_unit(service));

        if service_path.exists() {
            std::fs::remove_file(&service_path)?;
            info!("Removed: {}", service_path.display());
        }

        self.reload().await
    }

    async fn spawn(&self, service: Box<dyn Spawnable>) -> Result<SpawnedService> {
        // Call existing ensure() to install and start systemd unit
        self.ensure(service.name()).await?;
        Ok(SpawnedService::dummy())
    }
}

// ─── Systemd credential helpers ─────────────────────────────────────────────

/// Check whether any of the managed credentials exist in the systemd user
/// credstore (`~/.config/credstore.encrypted/`).
///
/// Used by `install()` to decide whether to generate units with
/// `ImportCredential=` directives.  Callers are expected to invoke
/// [`encrypt_credentials_if_available`] first so the `.cred` files are present.
fn credstore_has_credentials() -> bool {
    let Some(credstore) = dirs::config_dir().map(|d| d.join("credstore.encrypted")) else {
        return false;
    };
    if !credstore.exists() {
        return false;
    }
    units::SYSTEMD_CREDENTIAL_NAMES
        .iter()
        .any(|name| credstore.join(name).exists())
}

// ─── Systemd credential encryption ──────────────────────────────────────────

/// Check whether `systemd-creds` is available on `$PATH`.
fn has_systemd_creds() -> bool {
    std::process::Command::new("systemd-creds")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Encrypt a single secret into the systemd user credstore.
///
/// Runs: `systemd-creds encrypt --user --name=<name> --with-key=auto - <output>`
/// reading the plaintext from stdin and writing the encrypted `.cred` file.
fn systemd_creds_encrypt(name: &str, plaintext: &[u8], credstore: &std::path::Path) -> Result<()> {
    use std::io::Write as _;
    let output_path = credstore.join(name);
    let mut child = std::process::Command::new("systemd-creds")
        .args([
            "encrypt",
            "--user",
            &format!("--name={name}"),
            "--with-key=auto",
            "-",               // read from stdin
            output_path.to_str().unwrap_or("-"),
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("failed to spawn systemd-creds for '{name}'"))?;

    child
        .stdin
        .as_mut()
        .ok_or_else(|| anyhow!("no stdin on systemd-creds"))?
        .write_all(plaintext)
        .context("failed to write plaintext to systemd-creds stdin")?;

    let output = child
        .wait_with_output()
        .with_context(|| format!("systemd-creds failed for '{name}'"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("systemd-creds encrypt failed for '{name}': {stderr}");
    }

    debug!("Encrypted credential '{name}' → {}", output_path.display());
    Ok(())
}

/// Encrypt all secrets from `secrets_dir` into the systemd user credstore
/// (`~/.config/credstore.encrypted/`).
///
/// Returns `true` if at least one credential was encrypted successfully and the
/// generated service unit should include `ImportCredential=` directives.
/// Returns `false` if `systemd-creds` is unavailable or no secrets were found.
///
/// Pass `None` for `secrets_dir` to use the default: `~/.config/hyprstream/credentials`.
pub fn encrypt_credentials_if_available(secrets_dir: Option<&std::path::Path>) -> bool {
    if !has_systemd_creds() {
        tracing::warn!(
            "systemd-creds not found on $PATH; service unit will not use ImportCredential=.\n\
             Secret files will be stored as plain files (mode 0600).\n\
             Install systemd 250+ and re-run 'hyprstream service install' to enable \
             TPM2/host-key-backed credential protection."
        );
        return false;
    }

    let default_dir;
    let dir: &std::path::Path = match secrets_dir {
        Some(d) => d,
        None => {
            default_dir = dirs::config_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("/etc/hyprstream"))
                .join("hyprstream")
                .join("credentials");
            &default_dir
        }
    };

    let credstore = match dirs::config_dir() {
        Some(d) => d.join("credstore.encrypted"),
        None => {
            tracing::warn!("Could not determine config dir for credstore; skipping credential encryption");
            return false;
        }
    };

    if let Err(e) = std::fs::create_dir_all(&credstore) {
        tracing::warn!("Could not create credstore directory '{}': {e}", credstore.display());
        return false;
    }

    let mut encrypted_count = 0usize;
    for name in units::SYSTEMD_CREDENTIAL_NAMES {
        let secret_path = dir.join(name);
        if !secret_path.exists() {
            debug!("Skipping credential '{}' (not yet generated)", name);
            continue;
        }
        let plaintext = match std::fs::read(&secret_path) {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!("Could not read secret '{}': {e}", secret_path.display());
                continue;
            }
        };
        match systemd_creds_encrypt(name, &plaintext, &credstore) {
            Ok(()) => encrypted_count += 1,
            Err(e) => tracing::warn!("Failed to encrypt credential '{name}': {e}"),
        }
    }

    if encrypted_count > 0 {
        info!(
            "Encrypted {encrypted_count} credential(s) into '{}'",
            credstore.display()
        );
        true
    } else {
        tracing::warn!("No credentials found to encrypt; units will not use ImportCredential=");
        false
    }
}
