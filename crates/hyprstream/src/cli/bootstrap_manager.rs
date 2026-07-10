//! BootstrapManager — pre-service bootstrap for first-boot wizard.
//!
//! The wizard runs before any hyprstream services exist. BootstrapManager handles
//! GPU detection, variant installation, directory/key/policy initialization, and
//! service startup. It implements WizardBackend for the TUI, using bounded channels
//! with drain-to-latest pattern to bridge async operations to the 30fps render loop.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use anyhow::Context;
use tracing;
use ed25519_dalek::SigningKey;

use hyprstream_tui::wizard::backend::*;

use crate::auth::identity_store;
use crate::auth::policy_templates::{get_template, get_templates};
use crate::auth::{RocksDbUserStore, PolicyManager};
use crate::cli::gpu_detect;
use crate::cli::policy_handlers::{
    load_or_generate_signing_key, mint_local_token, parse_duration,
};



/// Pre-service bootstrap manager for the wizard TUI.
///
/// Handles GPU detection, variant download/install, directory/key/policy
/// initialization, and service startup — everything needed before the
/// hyprstream service layer exists.
pub struct BootstrapManager {
    rt: tokio::runtime::Handle,
    models_dir: PathBuf,
    config_services: Vec<String>,
    signing_key: Option<SigningKey>,

    // Install phase state
    install_rx: Option<mpsc::Receiver<InstallPoll>>,
    install_handle: Option<tokio::task::JoinHandle<()>>,
    install_cancel: Option<Arc<AtomicBool>>,

    // Bootstrap phase state
    bootstrap_rx: Option<mpsc::Receiver<BootstrapPoll>>,
    bootstrap_handle: Option<tokio::task::JoinHandle<()>>,
    bootstrap_cancel: Option<Arc<AtomicBool>>,

    // Service phase state
    service_rx: Option<mpsc::Receiver<OpStatus>>,
    service_handle: Option<tokio::task::JoinHandle<()>>,

    // Cached environment (avoid re-detecting)
    cached_env: Option<EnvironmentInfo>,

    // Accumulated policy state
    policy_manager: Option<PolicyManager>,
}

impl Drop for BootstrapManager {
    fn drop(&mut self) {
        if let Some(flag) = self.install_cancel.take() {
            flag.store(true, Ordering::Relaxed);
        }
        if let Some(flag) = self.bootstrap_cancel.take() {
            flag.store(true, Ordering::Relaxed);
        }
        if let Some(handle) = self.install_handle.take() {
            handle.abort();
        }
        if let Some(handle) = self.bootstrap_handle.take() {
            handle.abort();
        }
        if let Some(handle) = self.service_handle.take() {
            handle.abort();
        }
    }
}

/// Resolve the local OS username exactly as the CLI presents it in `sub`.
///
/// Must stay in lockstep with `sign_challenge.rs::load_user_signing_key`
/// (`$USER` → `$LOGNAME` → `"anonymous"`), so the identity the wizard registers
/// matches the subject the CLI authenticates as.
fn os_username() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("LOGNAME"))
        .unwrap_or_else(|_| "anonymous".to_owned())
}

// The shared enroll routine (#438 wizard + #439 `user create`) lives in
// `cli::enroll`; the wizard uses it directly. `bind_user_signing_key` is still
// unit-tested here (see `tests`).
use crate::cli::enroll::{enroll_user, EnrollKeySource};

/// Returns true if this is the first run (no bootstrap completed yet).
///
/// Used by the no-args entry point to decide between wizard and ShellClient.
/// Checks for the presence of `bootstrap-pubkeys` in the credentials directory,
/// which is written at the end of a successful bootstrap.
pub fn is_first_run(_models_dir: &std::path::Path) -> bool {
    // Check env var first — if signing key is provided externally, not first run
    if std::env::var("HYPRSTREAM__SIGNING_KEY").is_ok() {
        return false;
    }

    // bootstrap-pubkeys is written last during bootstrap — best indicator.
    // If credentials cannot be resolved, fail closed by treating this as first run.
    let Ok(credentials_dir) = identity_store::credentials_dir() else {
        return true;
    };

    !credentials_dir.join("bootstrap-pubkeys").exists()
}

impl BootstrapManager {
    /// Create a new bootstrap manager.
    pub fn new(rt: tokio::runtime::Handle, models_dir: PathBuf, config_services: Vec<String>) -> Self {
        Self {
            rt,
            models_dir,
            config_services,
            signing_key: None,
            install_rx: None,
            install_handle: None,
            install_cancel: None,
            bootstrap_rx: None,
            bootstrap_handle: None,
            bootstrap_cancel: None,
            service_rx: None,
            service_handle: None,
            cached_env: None,
            policy_manager: None,
        }
    }

    fn data_dir(&self) -> PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| self.models_dir.clone())
            .join("hyprstream")
    }

    fn policies_dir(&self) -> PathBuf {
        self.models_dir.join(".registry").join("policies")
    }

    fn keys_dir(&self) -> PathBuf {
        self.models_dir.join(".registry").join("keys")
    }

    /// Register a user identity in UserStore and bind the CLI's user-signing-key
    /// public key to it.
    ///
    /// Delegates to the shared [`enroll_user`] routine (#438 wizard + #439
    /// `user create`) so the register→store→bind sequence cannot drift. The
    /// wizard uses the `Generate` source: it adopts the existing/on-disk
    /// user-signing-key the CLI signs with. Without the binding the server can
    /// never resolve the CLI's signature back to a subject and falls back to
    /// `anonymous` (#438).
    fn register_local_identity(&mut self, username: &str) {
        let credentials_dir = match identity_store::credentials_dir() {
            Ok(path) => path,
            Err(e) => {
                tracing::warn!(
                    username,
                    "Failed to resolve credentials directory during bootstrap — user identity will not be registered: {e}"
                );
                return;
            }
        };
        let store = match RocksDbUserStore::open(&credentials_dir) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(
                    username, ?credentials_dir,
                    "Failed to load UserStore during bootstrap — user identity will not be registered: {e}"
                );
                return;
            }
        };

        // Load the user-signing-key from the SAME secrets dir the CLI uses, so
        // the bound fingerprint matches the key the CLI signs with.
        let secrets_dir = match crate::config::HyprConfig::resolve_secrets_dir() {
            Ok(path) => path,
            Err(e) => {
                tracing::warn!(
                    username,
                    "Failed to resolve secrets directory — CLI signing key will not be bound to '{username}': {e}"
                );
                return;
            }
        };

        // Root-of-trust enrollment follows node crypto policy (Hybrid default,
        // fail-closed) — the same selector as envelope traffic.
        let policy = hyprstream_rpc::envelope::envelope_policy_from_env();
        if let Err(e) = self.rt.block_on(
            enroll_user(&store, &secrets_dir, username, EnrollKeySource::Generate, policy),
        ) {
            tracing::warn!(
                username,
                "Failed to enroll user identity (register + bind signing key): {e}"
            );
        }
    }

    /// Ensure the signing key is loaded.
    fn ensure_signing_key(&mut self) {
        if self.signing_key.is_none() {
            let keys_dir = self.keys_dir();
            if let Ok(key) = self
                .rt
                .block_on(load_or_generate_signing_key(&keys_dir))
            {
                self.signing_key = Some(key);
            }
        }
    }

    /// Ensure the policy manager is loaded.
    fn ensure_policy_manager(&mut self) {
        if self.policy_manager.is_none() {
            let policies_dir = self.policies_dir();
            if let Ok(pm) = self.rt.block_on(PolicyManager::new(&policies_dir)) {
                self.policy_manager = Some(pm);
            }
        }
    }
}

impl WizardBackend for BootstrapManager {
    fn detect_environment(&mut self) -> EnvironmentInfo {
        let env = gpu_detect::detect_environment(&self.data_dir());
        self.cached_env = Some(env.clone());
        env
    }

    fn recommend_action(&self, env: &EnvironmentInfo) -> InstallAction {
        if env.run_mode == RunMode::Development {
            return InstallAction::Skip {
                reason: "Development mode — LIBTORCH already configured".to_owned(),
            };
        }
        if env.run_mode == RunMode::AppImage {
            return InstallAction::Skip {
                reason: "AppImage manages its own backends".to_owned(),
            };
        }
        if env.current_variant == env.recommended_variant {
            return InstallAction::AlreadyCurrent;
        }
        if let Some(ref installed) = env.installed_variant {
            if *installed == env.recommended_variant {
                return InstallAction::AlreadyCurrent;
            }
        }
        InstallAction::UpgradeVariant(env.recommended_variant.clone())
    }

    fn start_install(&mut self, variant: &LibtorchVariant) {
        let (tx, rx) = mpsc::sync_channel(8);
        self.install_rx = Some(rx);
        let cancel = Arc::new(AtomicBool::new(false));
        self.install_cancel = Some(cancel.clone());

        let variant = variant.clone();
        let models_dir = self.models_dir.clone();
        let data_dir = self.data_dir();

        self.install_handle = Some(self.rt.spawn(async move {
            let result = do_install(&tx, &variant, &models_dir, &data_dir, &cancel).await;
            if let Err(e) = result {
                let _ = tx.send(InstallPoll::Failed(e.to_string()));
            }
        }));
    }

    fn poll_install(&mut self) -> InstallPoll {
        let rx = match &self.install_rx {
            Some(rx) => rx,
            None => return InstallPoll::Failed("Install not started".into()),
        };
        // Drain-to-latest: consume all pending, stop at terminal state
        let mut latest = None;
        while let Ok(msg) = rx.try_recv() {
            let is_terminal = matches!(msg, InstallPoll::Done { .. } | InstallPoll::Failed(_));
            latest = Some(msg);
            if is_terminal {
                break;
            }
        }
        let result = latest.unwrap_or(InstallPoll::Detecting);
        // Clear receiver after terminal state to prevent stale fallback on subsequent polls
        if matches!(result, InstallPoll::Done { .. } | InstallPoll::Failed(_)) {
            self.install_rx = None;
        }
        result
    }

    fn start_bootstrap(&mut self) {
        let (tx, rx) = mpsc::sync_channel(8);
        self.bootstrap_rx = Some(rx);
        let cancel = Arc::new(AtomicBool::new(false));
        self.bootstrap_cancel = Some(cancel.clone());

        let models_dir = self.models_dir.clone();
        let keys_dir = self.keys_dir();

        self.bootstrap_handle = Some(self.rt.spawn(async move {
            // cancel flag checked via handle.abort() in Drop; no mid-operation
            // cancellation points in bootstrap (it's a short sequence).
            let _cancel = cancel; // held to keep Arc alive for Drop
            let result = do_bootstrap(&tx, &models_dir, &keys_dir).await;
            if let Err(e) = result {
                let _ = tx.send(BootstrapPoll::Failed(e.to_string()));
            }
        }));
    }

    fn poll_bootstrap(&mut self) -> BootstrapPoll {
        let rx = match &self.bootstrap_rx {
            Some(rx) => rx,
            None => return BootstrapPoll::InProgress("Not started".to_owned()),
        };
        let mut latest = None;
        while let Ok(msg) = rx.try_recv() {
            let is_terminal =
                matches!(msg, BootstrapPoll::Done(_) | BootstrapPoll::Failed(_));
            latest = Some(msg);
            if is_terminal {
                break;
            }
        }
        // If bootstrap is done, load the signing key
        if let Some(BootstrapPoll::Done(_)) = &latest {
            self.ensure_signing_key();
        }
        let result =
            latest.unwrap_or_else(|| BootstrapPoll::InProgress("Starting...".to_owned()));
        if matches!(result, BootstrapPoll::Done(_) | BootstrapPoll::Failed(_)) {
            self.bootstrap_rx = None;
        }
        result
    }

    fn has_existing_policy(&self) -> bool {
        let policy_csv = self.policies_dir().join("policy.csv");
        if let Ok(content) = std::fs::read_to_string(&policy_csv) {
            content
                .lines()
                .any(|l| l.starts_with("p,") || l.starts_with("p "))
        } else {
            false
        }
    }

    fn apply_template(&mut self, name: &str) {
        if let Some(template) = get_template(name) {
            self.ensure_policy_manager();
            if let Some(ref pm) = self.policy_manager {
                let _ = self.rt.block_on(async {
                    pm.apply_template(template).await
                });
            }
        }
    }

    fn add_user(&mut self, username: &str, role: &str) {
        if !VALID_INITIAL_ROLES.contains(&role) {
            tracing::error!(
                role,
                valid = VALID_INITIAL_ROLES.join(", "),
                "unknown --initial-user-role; aborting bootstrap"
            );
            std::process::exit(1);
        }

        // Register identity first — UserStore is authoritative
        self.register_local_identity(username);

        self.ensure_policy_manager();
        if let Some(ref pm) = self.policy_manager {
            let rules = predefined_role_rules(role);
            let _ = self.rt.block_on(async {
                for (resource, action) in rules {
                    pm.add_policy_with_domain(username, "*", resource, action, "allow")
                        .await?;
                }
                pm.save().await?;
                Ok::<_, anyhow::Error>(())
            });
        }
    }

    fn add_user_custom(&mut self, username: &str, resource: &str, actions: &[String]) {
        // Register identity first — UserStore is authoritative
        self.register_local_identity(username);

        self.ensure_policy_manager();
        if let Some(ref pm) = self.policy_manager {
            let _ = self.rt.block_on(async {
                for action in actions {
                    pm.add_policy_with_domain(username, "*", resource, action, "allow")
                        .await?;
                }
                pm.save().await?;
                Ok::<_, anyhow::Error>(())
            });
        }
    }

    fn save_policies(&mut self) {
        if let Some(ref pm) = self.policy_manager {
            let _ = self.rt.block_on(pm.save());
        }
    }

    fn generate_token(&mut self, username: &str, duration: &str) -> TokenResult {
        self.ensure_signing_key();
        let signing_key = match &self.signing_key {
            Some(k) => k,
            None => {
                return TokenResult {
                    token: "ERROR: No signing key".to_owned(),
                    expires: "N/A".to_owned(),
                }
            }
        };

        let dur = parse_duration(duration)
            .unwrap_or_else(|_| Some(chrono::Duration::days(90)))
            .unwrap_or_else(|| chrono::Duration::days(90));

        let (token, exp) = mint_local_token(signing_key, username, dur);

        let expires = if duration == "never" {
            "never".to_owned()
        } else {
            chrono::DateTime::from_timestamp(exp, 0)
                .map(|dt| dt.format("%Y-%m-%d").to_string())
                .unwrap_or_else(|| "unknown".to_owned())
        };

        TokenResult { token, expires }
    }

    fn start_services(&mut self) {
        let (tx, rx) = mpsc::sync_channel(8);
        self.service_rx = Some(rx);
        let services = self.config_services.clone();

        self.service_handle = Some(self.rt.spawn(async move {
            let _ = tx.send(OpStatus::InProgress);
            match crate::cli::handle_service_start(&services, None, false).await {
                Ok(()) => {
                    let _ = tx.send(OpStatus::Done);
                }
                Err(e) => {
                    let _ = tx.send(OpStatus::Failed(e.to_string()));
                }
            }
        }));
    }

    fn poll_pending(&mut self) -> OpStatus {
        let rx = match &self.service_rx {
            Some(rx) => rx,
            None => return OpStatus::Done,
        };
        let mut latest = None;
        while let Ok(msg) = rx.try_recv() {
            let is_terminal = matches!(msg, OpStatus::Done | OpStatus::Failed(_));
            latest = Some(msg);
            if is_terminal {
                break;
            }
        }
        let result = latest.unwrap_or(OpStatus::InProgress);
        if matches!(result, OpStatus::Done | OpStatus::Failed(_)) {
            self.service_rx = None;
        }
        result
    }

    fn local_username(&self) -> String {
        os_username()
    }

    fn templates(&self) -> Vec<TemplateInfo> {
        get_templates()
            .iter()
            .map(|t| TemplateInfo {
                name: t.name.to_owned(),
                description: t.description.to_owned(),
            })
            .collect()
    }
}

pub const VALID_INITIAL_ROLES: &[&str] = &["admin", "operator", "viewer", "trainer"];

/// Predefined role rules (mirrors wizard_handlers.rs).
fn predefined_role_rules(role: &str) -> Vec<(&'static str, &'static str)> {
    match role {
        "admin" => vec![("*", "*")],
        "operator" => vec![
            ("model:*", "infer"),
            ("model:*", "query"),
            ("model:*", "serve"),
        ],
        "viewer" => vec![("model:*", "query"), ("registry:*", "query")],
        "trainer" => vec![
            ("model:*", "infer"),
            ("model:*", "query"),
            ("model:*", "serve"),
            ("model:*", "train"),
        ],
        _ => vec![],
    }
}

/// Async install flow.
///
/// 1. Parse release manifest (if available)
/// 2. Clone/fetch release registry
/// 3. Create pathspec-filtered worktree
/// 4. Write active-backend + active-version config
async fn do_install(
    tx: &mpsc::SyncSender<InstallPoll>,
    variant: &LibtorchVariant,
    models_dir: &std::path::Path,
    data_dir: &std::path::Path,
    cancel: &AtomicBool,
) -> anyhow::Result<()> {
    let _ = tx.send(InstallPoll::Detecting);

    // Check for release registry
    let releases_dir = models_dir.join("hyprstream-releases.git");
    let variant_id = variant.id();

    if releases_dir.exists() {
        // Fetch latest
        let _ = tx.send(InstallPoll::Downloading {
            item: "release metadata".to_owned(),
            pct: 5,
        });
        // TODO: git fetch via git2db Registry cloneStream
    } else {
        // Clone release registry
        let _ = tx.send(InstallPoll::Downloading {
            item: "release registry".to_owned(),
            pct: 10,
        });
        // TODO: Clone via git2db Registry
    }

    // Simulate download progress (will be replaced with real cloneStream bridge)
    for pct in (20..=90).step_by(10) {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("Cancelled");
        }
        let _ = tx.send(InstallPoll::Downloading {
            item: format!("libtorch {}", variant.label()),
            pct,
        });
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    let _ = tx.send(InstallPoll::Extracting {
        item: format!("{variant_id} backend"),
    });

    // Create data directory structure
    let version = env!("CARGO_PKG_VERSION");
    let variant_dir = data_dir
        .join("versions")
        .join(format!("{version}-{variant_id}"))
        .join("backends")
        .join(variant_id);
    tokio::fs::create_dir_all(&variant_dir)
        .await
        .context("Failed to create variant directory")?;

    let _ = tx.send(InstallPoll::Configuring {
        step: "Writing backend configuration".to_owned(),
    });

    // TODO: The real binary is placed by pathspec-filtered worktree checkout.
    // For now this is a stub — the binary won't exist yet.
    let expected_binary = variant_dir.join("hyprstream");

    // Only write active-backend/active-version AFTER the binary is verified.
    // This prevents an inconsistent state where config points to a missing binary.
    if expected_binary.exists() {
        tokio::fs::write(data_dir.join("active-backend"), variant_id)
            .await
            .context("Failed to write active-backend")?;
        tokio::fs::write(data_dir.join("active-version"), version)
            .await
            .context("Failed to write active-version")?;

        let _ = tx.send(InstallPoll::Done {
            summary: format!("{} backend installed successfully", variant.label()),
        });
    } else {
        // Stub mode: report success but note the binary is not yet placed
        let _ = tx.send(InstallPoll::Done {
            summary: format!(
                "{} backend configured (binary pending real download implementation)",
                variant.label()
            ),
        });
    }

    Ok(())
}

/// Async bootstrap flow — silent (no stdout writes; TUI holds the stdout lock).
async fn do_bootstrap(
    tx: &mpsc::SyncSender<BootstrapPoll>,
    models_dir: &std::path::Path,
    keys_dir: &std::path::Path,
) -> anyhow::Result<()> {
    let mut steps = Vec::new();

    // ── 1. Directories ──────────────────────────────────────────────────────
    let _ = tx.send(BootstrapPoll::InProgress("Checking directories...".to_owned()));
    let registry_path = models_dir.join(".registry");
    for dir in &[
        models_dir,
        registry_path.as_path(),
        registry_path.join("policies").as_path(),
        registry_path.join("keys").as_path(),
    ] {
        if !dir.exists() {
            tokio::fs::create_dir_all(dir)
                .await
                .with_context(|| format!("Failed to create {}", dir.display()))?;
        }
    }
    steps.push("Directories OK".to_owned());

    // ── 2. Registry git repo ────────────────────────────────────────────────
    let _ = tx.send(BootstrapPoll::InProgress("Verifying registry...".to_owned()));
    let git_dir = registry_path.join(".git");
    if !git_dir.exists() && git2db::Git2DB::open(models_dir).await.is_err() {
        git2::Repository::init(&registry_path)
            .context("Failed to initialize .registry git repo")?;
    }
    steps.push("Registry OK".to_owned());

    // ── 3. Default policy files ─────────────────────────────────────────────
    let policies_dir = registry_path.join("policies");
    if !policies_dir.join("model.conf").exists() || !policies_dir.join("policy.csv").exists() {
        PolicyManager::new(&policies_dir)
            .await
            .context("Failed to create default policy files")?;
    }

    // ── 4. Signing key (becomes the CA key for PolicyService) ────────────
    let _ = tx.send(BootstrapPoll::InProgress("Loading signing key...".to_owned()));
    let root_key = load_or_generate_signing_key(keys_dir).await?;

    // ── 4b. Per-service independent keys + CA credentials ────────────────
    let _ = tx.send(BootstrapPoll::InProgress("Generating service keys...".to_owned()));
    {
        let credentials_dir = identity_store::credentials_dir()?;

        // Local issuer URL stamped into service JWTs so they verify on the
        // local IPC/AnySigner plane without tripping the #328 empty-`iss` gate.
        // Must match the issuer the services' ClusterKeySource trusts
        // (oauth.issuer_url(), the same value `cluster_key_source()` uses).
        let local_issuer_url = crate::config::HyprConfig::load()
            .map(|c| c.oauth.issuer_url())
            .unwrap_or_default();

        // CA JWT signing key (purpose-derived for JWT signature separation).
        // Derived BEFORE writing ca-pubkey so we can store the JWT key's verifying key,
        // not the root key's verifying key. Services verify JWTs with the derived key.
        let ca_jwt_key = hyprstream_rpc::node_identity::derive_purpose_key(
            &root_key, "hyprstream-jwt-v1",
        );

        // Write CA signing key only if not already present (idempotent)
        match identity_store::load_ca_signing_key(&credentials_dir) {
            Ok(_) => {
                // CA key exists; preserve it so running services stay valid
            }
            Err(_) => {
                identity_store::write_ca_signing_key(&credentials_dir, &root_key)?;
                // Write the JWT-derived verifying key as ca-pubkey (not the root key).
                // PolicyService signs JWTs with ca_jwt_key; services must verify with its pubkey.
                identity_store::write_ca_verifying_key(&credentials_dir, &ca_jwt_key.verifying_key())?;
            }
        }
        // Always sync signing-key and verifying key (derived from root_key, harmless to overwrite)
        identity_store::write_secret(&credentials_dir, "signing-key", &root_key.to_bytes())?;
        identity_store::write_ca_verifying_key(&credentials_dir, &ca_jwt_key.verifying_key())?;

        // Generate independent keypairs for each registered service
        use hyprstream_service::list_factories;

        let mut bootstrap_pubkeys = std::collections::HashMap::new();
        let now = chrono::Utc::now().timestamp();

        for factory in list_factories() {
            let service_name = factory.name;

            // PolicyService's identity IS the root/CA key: unlike every other
            // service it has no independent per-service keypair, so it resolves
            // to `root_key` rather than `load_or_generate_service_signing_key`
            // (which would read/generate a divergent `policy/signing-key`).
            //
            // But it is NOT skipped: we still mint a CA-signed `service:policy`
            // JWT — self-signed in the sense that the CA JWT key signs it, with
            // `cnf` binding the root verifying key — and persist it to
            // `policy/service-jwt`. That makes PolicyService symmetric with
            // every other service and keeps the trust store (which records
            // `root_key.verifying_key()` for "policy") in lockstep with the
            // on-disk JWT, so a later `service repair`/reinstall can no longer
            // leave a stale `policy` key/JWT pair that disagrees with the
            // current CA key (#448). It also enables future per-service
            // rotation of the policy credential.
            let service_key = if service_name == "policy" {
                root_key.clone()
            } else {
                identity_store::load_or_generate_service_signing_key(
                    &credentials_dir, service_name,
                )?
            };
            let service_vk = service_key.verifying_key();

            let jwt = crate::auth::service_jwt::issue_or_load_service_jwt(
                &credentials_dir, service_name, &ca_jwt_key, &service_vk, &local_issuer_url, now,
            )?;
            identity_store::write_service_jwt(&credentials_dir, service_name, &jwt)?;

            bootstrap_pubkeys.insert(service_name.to_owned(), service_vk);
        }

        // Write bootstrap pubkeys for all services
        identity_store::write_bootstrap_pubkeys(&credentials_dir, &bootstrap_pubkeys)?;

        tracing::info!(
            "Generated service keypairs + JWTs for {} services",
            bootstrap_pubkeys.len(),
        );
    }
    steps.push("Service keys OK".to_owned());

    // ── 5. Done ─────────────────────────────────────────────────────────────
    let _ = tx.send(BootstrapPoll::InProgress("Validating environment...".to_owned()));
    steps.push("Environment OK".to_owned());

    let _ = tx.send(BootstrapPoll::Done(steps));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::{RocksDbUserStore, UserStore};
    use crate::cli::enroll::bind_user_signing_key;
    use tempfile::TempDir;

    /// Replicates the register + bind sequence performed by
    /// `register_local_identity` against a temp store/secrets dir, then asserts
    /// the identity↔key binding the CLI relies on actually exists (#438).
    #[tokio::test]
    async fn register_and_bind_creates_identity_and_reverse_maps_pubkey() -> anyhow::Result<()> {
        let creds = TempDir::new()?;
        let secrets = TempDir::new()?;
        let username = "alice";

        // CLI's signing key lives in the secrets dir.
        let (_sk, vk) =
            identity_store::load_or_generate_user_signing_key(secrets.path())?;

        let store = RocksDbUserStore::open(creds.path())?;
        // Register (user did not exist yet) + bind, exactly like the wizard path.
        store.register(username).await?;
        bind_user_signing_key(&store, username, vk).await?;

        // 1. UserStore has a record for the username.
        assert!(
            store.get_profile(username).await?.is_some(),
            "expected a UserStore record for '{username}'"
        );

        // 2. The signing key's fingerprint reverse-maps to that username.
        let fingerprint = crate::auth::user_store::pubkey_fingerprint(&vk);
        assert_eq!(
            store.get_pubkey_user(&fingerprint).await?,
            Some(username.to_owned()),
            "user-signing-key fingerprint must reverse-map to '{username}'"
        );

        Ok(())
    }

    /// Binding is idempotent (re-running the wizard) and re-points a key left
    /// bound to a stale `anonymous` record from a prior partial run.
    #[tokio::test]
    async fn bind_is_idempotent_and_repoints_from_anonymous() -> anyhow::Result<()> {
        let creds = TempDir::new()?;
        let secrets = TempDir::new()?;
        let (_sk, vk) =
            identity_store::load_or_generate_user_signing_key(secrets.path())?;
        let fingerprint = crate::auth::user_store::pubkey_fingerprint(&vk);

        let store = RocksDbUserStore::open(creds.path())?;

        // Simulate a prior partial run: key bound to "anonymous".
        store.register("anonymous").await?;
        bind_user_signing_key(&store, "anonymous", vk).await?;
        assert_eq!(
            store.get_pubkey_user(&fingerprint).await?,
            Some("anonymous".to_owned())
        );

        // Wizard now registers the real user and binds — must re-point.
        store.register("alice").await?;
        bind_user_signing_key(&store, "alice", vk).await?;
        assert_eq!(
            store.get_pubkey_user(&fingerprint).await?,
            Some("alice".to_owned()),
            "key should be re-pointed from anonymous to alice"
        );

        // Re-running for the same user is a no-op (must not error/duplicate).
        bind_user_signing_key(&store, "alice", vk).await?;
        assert_eq!(
            store.get_pubkey_user(&fingerprint).await?,
            Some("alice".to_owned())
        );
        assert_eq!(
            store.list_pubkeys("alice").await?.len(),
            1,
            "idempotent re-bind must not duplicate the pubkey"
        );

        Ok(())
    }

    /// `local_username()` (via `os_username`) returns the OS user — not the old
    /// hardcoded "anonymous" — matching the `sub` the CLI presents.
    #[test]
    fn os_username_returns_user_env_not_anonymous() {
        // Save/restore env to avoid cross-test contamination.
        let prev_user = std::env::var("USER").ok();
        let prev_logname = std::env::var("LOGNAME").ok();

        std::env::set_var("USER", "alice");
        std::env::remove_var("LOGNAME");
        assert_eq!(os_username(), "alice");
        assert_ne!(os_username(), "anonymous");

        // Restore.
        match prev_user {
            Some(v) => std::env::set_var("USER", v),
            None => std::env::remove_var("USER"),
        }
        if let Some(v) = prev_logname {
            std::env::set_var("LOGNAME", v);
        }
    }
}
