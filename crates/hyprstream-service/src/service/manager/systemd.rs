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

/// Connect to the user D-Bus instance for systemd management.
///
/// Tries `$DBUS_SESSION_BUS_ADDRESS` first (set by PAM on both desktop and
/// headless SSH logins when `systemd --user` is running). Falls back to the
/// well-known socket path `/run/user/$UID/bus` for environments where the env
/// var is absent but the socket exists (e.g. running from a cron job or
/// service that didn't inherit the PAM environment).
///
/// Does NOT use the systemd private bus (`/run/user/$UID/systemd/private/bus`)
/// — that path is an internal bind-mount not accessible from outside systemd.
async fn user_systemd_connection() -> Result<Connection> {
    // Primary: $DBUS_SESSION_BUS_ADDRESS (set by pam_systemd on any login).
    if let Ok(conn) = Connection::session().await {
        return Ok(conn);
    }

    // Fallback: well-known socket path that systemd --user maintains.
    let uid = nix::unistd::getuid().as_raw();
    let rt = std::env::var("XDG_RUNTIME_DIR")
        .unwrap_or_else(|_| format!("/run/user/{uid}"));
    let bus_path = format!("{rt}/bus");
    zbus_systemd::zbus::ConnectionBuilder::address(
        format!("unix:path={bus_path}").as_str(),
    )?
    .build()
    .await
    .with_context(|| format!(
        "could not connect to user D-Bus (tried $DBUS_SESSION_BUS_ADDRESS and {bus_path})"
    ))
}

/// Systemd-based user service manager
///
/// Manages per-user services via D-Bus calls to the systemd user instance.
/// Units are installed to `~/.config/systemd/user/`.
pub struct SystemdManager {
    #[allow(dead_code)]
    connection: Connection,
    systemd: ManagerProxy<'static>,
}

impl SystemdManager {
    /// Create a new SystemdManager.
    ///
    /// Connects to the user session D-Bus for systemd1 and the system D-Bus
    /// for login1 (logind). logind lives on the system bus only — connecting
    /// LoginProxy to the session bus caused set_user_linger() to hang forever
    /// waiting for a response that never arrives. Enabling user linger keeps
    /// services persisting across logouts on headless machines.
    pub async fn new() -> Result<Self> {
        let connection = user_systemd_connection().await?;
        let systemd = ManagerProxy::new(&connection).await?;

        // Enable linger via the system bus (org.freedesktop.login1 is system-only).
        // This keeps the user systemd instance running after logout so services
        // registered with WantedBy=default.target survive reboots.
        let system_connection = Connection::system().await?;
        let login = LoginProxy::new(&system_connection).await?;
        let uid = nix::unistd::getuid().as_raw();
        if let Err(e) = login.set_user_linger(uid, true, false).await {
            debug!("Failed to enable user linger (may already be enabled): {}", e);
        } else {
            debug!("User lingering enabled");
        }

        Ok(Self {
            connection,
            systemd,
        })
    }

    /// Create a new SystemdManager using the D-Bus session bus.
    ///
    /// Only for session-scoped services (e.g. the TUI multiplexer) that should
    /// be tied to an active login session.  Do NOT use for long-lived public
    /// service daemons — the session bus disappears on logout.
    pub async fn new_session() -> Result<Self> {
        let connection = Connection::session().await
            .context("could not connect to D-Bus session bus (is $DBUS_SESSION_BUS_ADDRESS set?)")?;
        let systemd = ManagerProxy::new(&connection).await?;
        Ok(Self { connection, systemd })
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

/// Systemd-based system-wide service manager (requires root)
///
/// Manages system services via the system D-Bus. Units are installed to
/// `/etc/systemd/system/` and run under a dedicated `hyprstream` system user.
pub struct SystemdSystemManager {
    #[allow(dead_code)]
    connection: Connection,
    systemd: ManagerProxy<'static>,
}

impl SystemdSystemManager {
    pub async fn new() -> Result<Self> {
        let connection = Connection::system().await?;
        let systemd = ManagerProxy::new(&connection).await?;
        Ok(Self { connection, systemd })
    }

    fn service_unit(service: &str) -> String {
        format!("hyprstream-{service}.service")
    }

    fn units_dir() -> PathBuf {
        PathBuf::from("/etc/systemd/system")
    }
}

#[async_trait]
impl ServiceManager for SystemdSystemManager {
    async fn install(&self, service: &str) -> Result<()> {
        let units_dir = Self::units_dir();
        std::fs::create_dir_all(&units_dir)?;

        let depends_on = crate::service::factory::get_factory(service)
            .map(|f| f.depends_on)
            .unwrap_or(&[]);

        let service_content = units::system_service_unit(service, depends_on)?;
        let service_path = units_dir.join(Self::service_unit(service));

        if std::fs::read_to_string(&service_path).ok().as_deref() != Some(&service_content) {
            std::fs::write(&service_path, &service_content)?;
            info!("Installed system service unit: {}", service_path.display());
            self.reload().await?;
        }

        Ok(())
    }

    async fn uninstall(&self, service: &str) -> Result<()> {
        let _ = self.stop(service).await;
        let path = Self::units_dir().join(Self::service_unit(service));
        if path.exists() {
            std::fs::remove_file(&path)?;
            info!("Removed: {}", path.display());
        }
        self.reload().await
    }

    async fn start(&self, service: &str) -> Result<()> {
        self.systemd
            .start_unit(Self::service_unit(service), "replace".into())
            .await?;
        info!("Started system service unit: {}", Self::service_unit(service));
        Ok(())
    }

    async fn stop(&self, service: &str) -> Result<()> {
        self.systemd
            .stop_unit(Self::service_unit(service), "replace".into())
            .await?;
        info!("Stopped system service unit: {}", Self::service_unit(service));
        Ok(())
    }

    async fn is_active(&self, service: &str) -> Result<bool> {
        match self.systemd.get_unit(Self::service_unit(service)).await {
            Ok(unit_path) => {
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
        let status = tokio::process::Command::new("systemctl")
            .arg("daemon-reload")
            .status()
            .await
            .context("failed to spawn systemctl daemon-reload")?;
        if !status.success() {
            anyhow::bail!("systemctl daemon-reload failed (exit {})", status);
        }
        debug!("Reloaded systemd system daemon");
        Ok(())
    }

    async fn enable(&self, service: &str) -> Result<()> {
        let status = tokio::process::Command::new("systemctl")
            .args(["enable", &Self::service_unit(service)])
            .status()
            .await
            .context("failed to spawn systemctl enable")?;
        if !status.success() {
            anyhow::bail!("systemctl enable {} failed (exit {})", Self::service_unit(service), status);
        }
        debug!("Enabled system service unit: {}", Self::service_unit(service));
        Ok(())
    }

    async fn spawn(&self, service: Box<dyn Spawnable>) -> Result<SpawnedService> {
        self.ensure(service.name()).await?;
        Ok(SpawnedService::dummy())
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

        // Look up the factory to get depends_on for this service.
        let depends_on = crate::service::factory::get_factory(service)
            .map(|f| f.depends_on)
            .unwrap_or(&[]);

        let service_content = units::service_unit(service, use_creds, depends_on)?;
        let service_path = units_dir.join(Self::service_unit(service));

        // Write service unit if changed (idempotent), then reload so systemd
        // sees the new unit before any subsequent start() or enable() call.
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
        // Use systemctl subprocess rather than the D-Bus Reload() call.
        // The D-Bus call hangs on systemd 258+ (Fedora 43) during daemon-reload
        // when called from a non-session context.
        let status = tokio::process::Command::new("systemctl")
            .args(["--user", "daemon-reload"])
            .status()
            .await
            .context("failed to spawn systemctl --user daemon-reload")?;
        if !status.success() {
            anyhow::bail!("systemctl --user daemon-reload failed (exit {})", status);
        }
        debug!("Reloaded systemd user daemon");
        Ok(())
    }

    async fn enable(&self, service: &str) -> Result<()> {
        let status = tokio::process::Command::new("systemctl")
            .args(["--user", "enable", &Self::service_unit(service)])
            .status()
            .await
            .context("failed to spawn systemctl --user enable")?;
        if !status.success() {
            anyhow::bail!("systemctl --user enable {} failed (exit {})", Self::service_unit(service), status);
        }
        debug!("Enabled service unit: {}", Self::service_unit(service));
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
    units::ALL_CREDENTIAL_NAMES
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
/// Pass `None` for `secrets_dir` to use the default: the instance-namespaced
/// `~/.config/hyprstream[/instances/{inst}]/credentials`.
///
/// # Best-effort defense-in-depth (#808)
///
/// The plaintext 0600 files in `secrets_dir` remain the authoritative copy and
/// are NOT deleted: per-service `LoadCredential=` sources and CLI tooling read
/// them directly. The encrypted credstore is an additional TPM2/host-key-bound
/// copy consumed via `ImportCredential=` — protection here is best-effort
/// defense-in-depth, not the sole at-rest control.
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
            default_dir = units::hyprstream_config_dir().map(|d| d.join("credentials"));
            match default_dir.as_deref() {
                Some(d) => d,
                None => {
                    tracing::warn!(
                        "Could not determine config dir for secrets; skipping credential encryption"
                    );
                    return false;
                }
            }
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

    // Encrypt node-level credentials (flat: signing-key, ca-pubkey, bootstrap-pubkeys, rsa-key, TLS)
    // Note: signing-key is written as a copy of the CA key so PolicyService can load it
    for name in units::NODE_CREDENTIAL_NAMES.iter().chain(std::iter::once(&"signing-key")) {
        let secret_path = dir.join(name);
        if !secret_path.exists() {
            debug!("Skipping node credential '{}' (not yet generated)", name);
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

    // Encrypt per-service credentials with service-prefixed names
    // e.g., credentials/model/signing-key → credstore.encrypted/model-signing-key
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let service_name = match entry.file_name().to_str() {
                Some(n) => n.to_owned(),
                None => continue,
            };
            for name in units::SERVICE_CREDENTIAL_NAMES {
                let secret_path = path.join(name);
                if !secret_path.exists() {
                    continue;
                }
                let plaintext = match std::fs::read(&secret_path) {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::warn!("Could not read secret '{}': {e}", secret_path.display());
                        continue;
                    }
                };
                let prefixed_name = format!("{service_name}-{name}");
                match systemd_creds_encrypt(&prefixed_name, &plaintext, &credstore) {
                    Ok(()) => encrypted_count += 1,
                    Err(e) => tracing::warn!(
                        "Failed to encrypt credential '{prefixed_name}': {e}"
                    ),
                }
            }
        }
    }
    // Note: per-service credentials are now stored with prefixed names in the
    // flat credstore (e.g., model-signing-key, model-service-jwt) so that
    // ImportCredential can decrypt them. Subdirectory encryption was removed
    // because SetLoadCredential doesn't decrypt systemd-creds encrypted files.

    // Encrypt policy-only credentials (ca-key)
    for name in units::POLICY_CREDENTIAL_NAMES {
        let secret_path = dir.join(name);
        if !secret_path.exists() {
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
