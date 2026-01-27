//! Runtime path resolution for system vs user mode
//!
//! Provides FHS/XDG-compliant path resolution based on execution context:
//! - Root/system service: `/run/hyprstream/`
//! - User service: `$XDG_RUNTIME_DIR/hyprstream/`
//! - Fallback: `/tmp/hyprstream-<uid>/`

use std::path::PathBuf;

/// Get the appropriate runtime directory based on execution context
///
/// Path selection follows FHS 3.0 and XDG Base Directory specifications:
/// - Root/system service: `/run/hyprstream/`
/// - User service: `$XDG_RUNTIME_DIR/hyprstream/`
/// - Fallback (no systemd): `/tmp/hyprstream-<uid>/`
pub fn runtime_dir() -> PathBuf {
    if nix::unistd::geteuid().is_root() {
        PathBuf::from("/run/hyprstream")
    } else if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
        PathBuf::from(xdg).join("hyprstream")
    } else {
        PathBuf::from(format!("/tmp/hyprstream-{}", nix::unistd::getuid()))
    }
}

/// Socket directory for EventService
pub fn events_dir() -> PathBuf {
    runtime_dir().join("events")
}

/// Publisher socket path for EventService
pub fn events_pub_socket() -> PathBuf {
    events_dir().join("pub.sock")
}

/// Subscriber socket path for EventService
pub fn events_sub_socket() -> PathBuf {
    events_dir().join("sub.sock")
}

/// Sandbox runtime directory
pub fn sandboxes_dir() -> PathBuf {
    runtime_dir().join("sandboxes")
}

/// RegistryService IPC socket
pub fn registry_socket() -> PathBuf {
    runtime_dir().join("registry.sock")
}

/// PolicyService IPC socket
pub fn policy_socket() -> PathBuf {
    runtime_dir().join("policy.sock")
}

/// WorkerService IPC socket
pub fn worker_socket() -> PathBuf {
    runtime_dir().join("worker.sock")
}

/// WorkflowService IPC socket
pub fn workflow_socket() -> PathBuf {
    runtime_dir().join("workflow.sock")
}

/// EventService IPC socket
pub fn event_socket() -> PathBuf {
    runtime_dir().join("event.sock")
}

/// Nydus runtime directory
pub fn nydus_dir() -> PathBuf {
    runtime_dir().join("nydus")
}

/// PID file location for main process
pub fn pid_file() -> PathBuf {
    runtime_dir().join("hyprstream.pid")
}

/// PID file for a named service (subprocess mode)
///
/// Path: `<runtime_dir>/<name>.pid`
/// Example: `/run/user/1000/hyprstream/events.pid`
pub fn service_pid_file(name: &str) -> PathBuf {
    runtime_dir().join(format!("{}.pid", name))
}

/// Set appropriate permissions on a directory
///
/// - System (root): 0755
/// - User: 0700
#[cfg(unix)]
fn set_runtime_permissions(dir: &std::path::Path) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = if nix::unistd::geteuid().is_root() {
        0o755
    } else {
        0o700
    };
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(mode))
}

/// Ensure runtime directory exists with correct permissions
///
/// Creates the directory tree and sets appropriate permissions:
/// - System (root): 0755
/// - User: 0700
pub fn ensure_runtime_dir() -> std::io::Result<()> {
    let dir = runtime_dir();
    std::fs::create_dir_all(&dir)?;

    #[cfg(unix)]
    set_runtime_permissions(&dir)?;

    Ok(())
}

/// Ensure events directory exists with correct permissions
pub fn ensure_events_dir() -> std::io::Result<()> {
    ensure_runtime_dir()?;
    let dir = events_dir();
    std::fs::create_dir_all(&dir)?;

    #[cfg(unix)]
    set_runtime_permissions(&dir)?;

    Ok(())
}

/// Check if running as root
pub fn is_root() -> bool {
    nix::unistd::geteuid().is_root()
}

/// Get the stable executable path for spawning subprocesses.
///
/// When running from an AppImage, `current_exe()` returns the temporary mount path
/// (e.g., `/tmp/.mount_hyprXXX/usr/bin/hyprstream`) which becomes invalid when the
/// AppImage exits. Instead, we use the `$APPIMAGE` environment variable which
/// points to the stable AppImage file path.
///
/// See: <https://docs.appimage.org/packaging-guide/environment-variables.html>
///
/// # Returns
/// - `$APPIMAGE` path if set and the file exists
/// - `current_exe()` otherwise
pub fn executable_path() -> std::io::Result<PathBuf> {
    // Check for AppImage environment variable first
    if let Ok(appimage_path) = std::env::var("APPIMAGE") {
        let path = PathBuf::from(&appimage_path);
        if path.exists() {
            return Ok(path);
        }
    }

    // Fall back to current executable
    std::env::current_exe()
}

// =============================================================================
// Install paths
// =============================================================================

/// User's executable directory (`~/.local/bin`)
pub fn bin_dir() -> Option<PathBuf> {
    dirs::executable_dir()
}

/// Hyprstream data directory (`~/.local/share/hyprstream`)
pub fn data_dir() -> Option<PathBuf> {
    dirs::data_local_dir().map(|d| d.join("hyprstream"))
}

/// Versions directory (`~/.local/share/hyprstream/versions`)
pub fn versions_dir() -> Option<PathBuf> {
    data_dir().map(|d| d.join("versions"))
}

/// Version-specific directory (`~/.local/share/hyprstream/versions/$VERSION`)
pub fn version_dir(version: &str) -> Option<PathBuf> {
    versions_dir().map(|d| d.join(version))
}

/// Path to installed binary in version directory
///
/// Returns path to `hyprstream` or `hyprstream.appimage` in version dir.
/// Checks for both filenames, preferring `.appimage` if both exist.
pub fn version_binary(version: &str) -> Option<PathBuf> {
    let dir = version_dir(version)?;
    let appimage = dir.join("hyprstream.appimage");
    if appimage.is_file() {
        return Some(appimage);
    }
    let binary = dir.join("hyprstream");
    if binary.is_file() {
        return Some(binary);
    }
    None
}

/// Get the path where hyprstream is installed (if installed)
///
/// Returns `~/.local/bin/hyprstream` if it exists (file or symlink).
pub fn installed_executable_path() -> Option<PathBuf> {
    let path = bin_dir()?.join("hyprstream");
    if path.symlink_metadata().is_ok() {
        Some(path)
    } else {
        None
    }
}

/// Get the current user's UID
pub fn current_uid() -> u32 {
    nix::unistd::getuid().as_raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_dir_structure() {
        let rt = runtime_dir();
        assert!(rt.to_string_lossy().contains("hyprstream"));
    }

    #[test]
    fn test_events_paths() {
        let pub_sock = events_pub_socket();
        let sub_sock = events_sub_socket();

        assert!(pub_sock.to_string_lossy().contains("events"));
        assert!(pub_sock.to_string_lossy().ends_with("pub.sock"));
        assert!(sub_sock.to_string_lossy().ends_with("sub.sock"));
    }

    #[test]
    fn test_sandboxes_dir() {
        let dir = sandboxes_dir();
        assert!(dir.to_string_lossy().contains("sandboxes"));
    }
}
