/// Cross-platform directory management for TCL MCP Server
///
/// Uses the `dirs` crate to handle platform-specific data directories
/// following OS conventions:
/// - Linux/Unix: XDG Base Directory Specification
/// - macOS: Apple directory guidelines
/// - Windows: Windows directory standards
use anyhow::{anyhow, Result};
use std::fs;
use std::path::{Path, PathBuf};

/// Get the data directory for TCL MCP Server
///
/// Returns platform-specific data directory:
/// - Linux: `$XDG_DATA_HOME/tcl-mcp-server` or `~/.local/share/tcl-mcp-server`
/// - macOS: `~/Library/Application Support/tcl-mcp-server`
/// - Windows: `%APPDATA%\tcl-mcp-server` or `%USERPROFILE%\AppData\Roaming\tcl-mcp-server`
pub fn data_dir() -> Result<PathBuf> {
    let base_dir =
        dirs::data_local_dir().ok_or_else(|| anyhow!("Unable to determine data directory"))?;
    Ok(base_dir.join("tcl-mcp-server"))
}

/// Get the config directory for TCL MCP Server
///
/// Returns platform-specific config directory:
/// - Linux: `$XDG_CONFIG_HOME/tcl-mcp-server` or `~/.config/tcl-mcp-server`
/// - macOS: `~/Library/Preferences/tcl-mcp-server`
/// - Windows: Same as data directory
pub fn config_dir() -> Result<PathBuf> {
    let base_dir =
        dirs::config_dir().ok_or_else(|| anyhow!("Unable to determine config directory"))?;
    Ok(base_dir.join("tcl-mcp-server"))
}

/// Get the cache directory for TCL MCP Server
///
/// Returns platform-specific cache directory:
/// - Linux: `$XDG_CACHE_HOME/tcl-mcp-server` or `~/.cache/tcl-mcp-server`
/// - macOS: `~/Library/Caches/tcl-mcp-server`
/// - Windows: `%LOCALAPPDATA%\tcl-mcp-server\cache` or `%USERPROFILE%\AppData\Local\tcl-mcp-server\cache`
pub fn cache_dir() -> Result<PathBuf> {
    let base_dir =
        dirs::cache_dir().ok_or_else(|| anyhow!("Unable to determine cache directory"))?;
    Ok(base_dir.join("tcl-mcp-server"))
}

/// Ensure a directory exists, creating it if necessary
pub fn ensure_dir(path: &Path) -> Result<()> {
    if !path.exists() {
        fs::create_dir_all(path)?;
    }
    Ok(())
}

/// Get the path for storing TCL tools
pub fn tools_dir() -> Result<PathBuf> {
    let data = data_dir()?;
    Ok(data.join("tools"))
}

/// Get the path for the MCP server index
pub fn mcp_index_path() -> Result<PathBuf> {
    let data = data_dir()?;
    Ok(data.join("mcp-index.json"))
}

/// Get the path for storing TCL scripts
pub fn scripts_dir() -> Result<PathBuf> {
    let data = data_dir()?;
    Ok(data.join("scripts"))
}

/// Initialize all required directories
pub fn init_directories() -> Result<()> {
    ensure_dir(&data_dir()?)?;
    ensure_dir(&config_dir()?)?;
    ensure_dir(&cache_dir()?)?;
    ensure_dir(&tools_dir()?)?;
    ensure_dir(&scripts_dir()?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_dir() {
        let dir = data_dir().unwrap();
        assert!(dir.to_string_lossy().contains("tcl-mcp-server"));
    }

    #[test]
    fn test_platform_specific_paths() {
        let data = data_dir().unwrap();
        let config = config_dir().unwrap();
        let cache = cache_dir().unwrap();

        if cfg!(target_os = "linux") {
            // Should follow XDG spec
            let home = std::env::var("HOME").unwrap();
            assert!(
                data.starts_with(home.as_str()) || data.to_string_lossy().contains(".local/share")
            );
            assert!(
                config.starts_with(home.as_str()) || config.to_string_lossy().contains(".config")
            );
            assert!(cache.starts_with(home.as_str()) || cache.to_string_lossy().contains(".cache"));
        } else if cfg!(target_os = "macos") {
            // Should use Library directories
            assert!(data
                .to_string_lossy()
                .contains("Library/Application Support"));
            assert!(config.to_string_lossy().contains("Library/Preferences"));
            assert!(cache.to_string_lossy().contains("Library/Caches"));
        } else if cfg!(target_os = "windows") {
            // Should use AppData directories
            assert!(data.to_string_lossy().contains("AppData"));
        }
    }

    #[test]
    fn test_subdirectories() {
        let tools = tools_dir().unwrap();
        let scripts = scripts_dir().unwrap();
        let mcp_index = mcp_index_path().unwrap();

        assert!(tools.ends_with("tools"));
        assert!(scripts.ends_with("scripts"));
        assert!(mcp_index.file_name().unwrap() == "mcp-index.json");
    }
}
