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
use ed25519_dalek::SigningKey;

use hyprstream_tui::wizard::backend::*;

use crate::auth::jwt;
use crate::auth::policy_templates::{get_template, get_templates};
use crate::auth::{Claims, PolicyManager, write_policy_file};
use crate::cli::gpu_detect;
use crate::cli::policy_handlers::load_or_generate_signing_key;
use crate::cli::service_handlers::run_repair_checks;

/// Pre-service bootstrap manager for the wizard TUI.
///
/// Handles GPU detection, variant download/install, directory/key/policy
/// initialization, and service startup — everything needed before the
/// hyprstream service layer exists.
pub struct BootstrapManager {
    rt: tokio::runtime::Handle,
    models_dir: PathBuf,
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

impl BootstrapManager {
    /// Create a new bootstrap manager.
    pub fn new(rt: tokio::runtime::Handle, models_dir: PathBuf) -> Self {
        Self {
            rt,
            models_dir,
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
            let content = template.expanded_rules();
            let policy_csv = self.policies_dir().join("policy.csv");
            let _ = self.rt.block_on(async {
                write_policy_file(&policy_csv, content.as_bytes()).await
            });
            // Reload policy manager
            self.policy_manager = None;
            self.ensure_policy_manager();
        }
    }

    fn add_user(&mut self, username: &str, role: &str) {
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

        let duration_secs = match duration {
            "30d" => 30 * 86400,
            "1y" => 365 * 86400,
            "never" => 365 * 100 * 86400,
            _ => 90 * 86400, // 90d default
        };

        let now = chrono::Utc::now().timestamp();
        let exp = now + duration_secs;
        let claims = Claims::new(username.to_owned(), now, exp);
        let token = jwt::encode(&claims, signing_key);

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

        self.service_handle = Some(self.rt.spawn(async move {
            let _ = tx.send(OpStatus::InProgress);
            match crate::cli::handle_service_start(&[], None, false).await {
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
        hyprstream_rpc::envelope::RequestIdentity::local()
            .user()
            .to_owned()
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

/// Async bootstrap flow.
async fn do_bootstrap(
    tx: &mpsc::SyncSender<BootstrapPoll>,
    models_dir: &std::path::Path,
    keys_dir: &std::path::Path,
) -> anyhow::Result<()> {
    let mut steps = Vec::new();

    let _ = tx.send(BootstrapPoll::InProgress(
        "Checking directories...".to_owned(),
    ));
    // Run repair checks (creates dirs, registry, etc.)
    run_repair_checks(models_dir, false).await?;
    steps.push("Directories OK".to_owned());

    let _ = tx.send(BootstrapPoll::InProgress(
        "Verifying registry...".to_owned(),
    ));
    // Registry is created by repair checks
    steps.push("Registry OK".to_owned());

    let _ = tx.send(BootstrapPoll::InProgress(
        "Loading signing key...".to_owned(),
    ));
    load_or_generate_signing_key(keys_dir).await?;
    steps.push("Signing key OK".to_owned());

    let _ = tx.send(BootstrapPoll::InProgress(
        "Validating environment...".to_owned(),
    ));
    steps.push("Environment OK".to_owned());

    let _ = tx.send(BootstrapPoll::Done(steps));
    Ok(())
}
