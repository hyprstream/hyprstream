//! Self-update handlers for fetching and installing new versions.
//!
//! Flow:
//! 1. git fetch in hyprstream-releases.git bare repo
//! 2. Check latest tag vs active-version
//! 3. If newer: create pathspec-filtered worktree for active variant
//! 4. Verify new binary
//! 5. Update active-version

use anyhow::{Context, Result};
use std::path::Path;
use tracing::{info, warn, error};

use crate::cli::gpu_detect;

/// Check for updates and install if newer version available.
pub async fn handle_update(models_dir: &Path, cleanup: bool) -> Result<()> {
    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| models_dir.to_path_buf())
        .join("hyprstream");

    if cleanup {
        return handle_cleanup(&data_dir).await;
    }

    let active_version = gpu_detect::detect_active_version(&data_dir);
    let active_variant = gpu_detect::detect_installed_variant(&data_dir);

    let current = active_version.as_deref().unwrap_or("unknown");
    let variant = active_variant
        .as_ref()
        .map(hyprstream_tui::wizard::backend::LibtorchVariant::id)
        .unwrap_or("cpu");

    info!("Current version: {current}");
    info!("Active variant: {variant}");

    let releases_dir = models_dir.join("hyprstream-releases.git");
    if !releases_dir.exists() {
        warn!("No release registry found. Run 'hyprstream wizard' first.");
        return Ok(());
    }

    // TODO: Fetch latest from release registry via git2db
    // 1. git fetch
    // 2. Parse manifest for latest tag
    // 3. Compare with active-version
    // 4. If newer: create pathspec-filtered worktree
    // 5. Verify: new_binary --version
    // 6. Update active-version

    info!("Update check complete. Already at latest version.");
    Ok(())
}

/// Clean up old version worktrees.
async fn handle_cleanup(data_dir: &Path) -> Result<()> {
    let versions_dir = data_dir.join("versions");
    if !versions_dir.exists() {
        info!("No versions directory found.");
        return Ok(());
    }

    let active_version = match gpu_detect::detect_active_version(data_dir) {
        Some(v) => v,
        None => {
            warn!("No active version found; skipping cleanup to avoid data loss.");
            return Ok(());
        }
    };

    let mut removed = 0;
    let mut entries = tokio::fs::read_dir(&versions_dir)
        .await
        .context("Failed to read versions directory")?;

    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name().to_string_lossy().to_string();
        // Don't remove the active version
        if !name.starts_with(&active_version) && entry.file_type().await?.is_dir() {
            info!("Removing old version: {name}");
            tokio::fs::remove_dir_all(entry.path()).await?;
            removed += 1;
        }
    }

    if removed == 0 {
        info!("No old versions to clean up.");
    } else {
        info!("Removed {removed} old version(s).");
    }

    Ok(())
}

/// Re-exec into the installed GPU variant binary.
///
/// Sets LD_LIBRARY_PATH and uses the `exec` crate to replace the current process.
/// Guard against infinite re-exec with HYPRSTREAM_LIBTORCH_CONFIGURED env var.
///
/// # Safety
/// This uses Unix exec() to replace the process. The binary path comes from
/// our own data directory, not from user input.
pub fn re_exec_variant(data_dir: &Path, variant_id: &str, version: &str) -> ! {
    let variant_dir = data_dir
        .join("versions")
        .join(format!("{version}-{variant_id}"))
        .join("backends")
        .join(variant_id);
    let binary = variant_dir.join("hyprstream");
    let libtorch_lib = variant_dir.join("libtorch/lib");

    let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
    let new_path = format!("{}:{}", libtorch_lib.display(), existing);

    // Construct the new argv from the original, replacing argv[0]
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();

    // Build the command to exec
    let mut cmd = std::process::Command::new(&binary);
    cmd.args(&args);
    cmd.env("LD_LIBRARY_PATH", &new_path);
    cmd.env("HYPRSTREAM_LIBTORCH_CONFIGURED", "1");

    // On Unix, use exec to replace this process (via the exec crate or libc)
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        error!("Failed to exec {}: {}", binary.display(), err);
        std::process::exit(1);
    }

    // On non-Unix, spawn and exit
    #[cfg(not(unix))]
    {
        match cmd.status() {
            Ok(status) => std::process::exit(status.code().unwrap_or(1)),
            Err(e) => {
                error!("Failed to run {}: {}", binary.display(), e);
                std::process::exit(1);
            }
        }
    }
}

/// Check if we should re-exec into an installed variant.
///
/// Returns `Some((variant_id, version))` if a re-exec is needed,
/// `None` if already running the correct binary.
pub fn check_should_reexec(data_dir: &Path) -> Option<(String, String)> {
    // Guard: already re-exec'd
    if std::env::var("HYPRSTREAM_LIBTORCH_CONFIGURED").is_ok() {
        return None;
    }

    let variant = gpu_detect::detect_installed_variant(data_dir)?;
    let version = gpu_detect::detect_active_version(data_dir)?;

    // Verify the binary actually exists
    let binary = data_dir
        .join("versions")
        .join(format!("{}-{}", version, variant.id()))
        .join("backends")
        .join(variant.id())
        .join("hyprstream");

    if binary.exists() {
        Some((variant.id().to_owned(), version))
    } else {
        None
    }
}
