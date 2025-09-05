//! Validation utilities for paths, filenames, and security checks

use std::path::{Path, PathBuf};
use crate::constants::validation::*;

/// Validate and sanitize a filename to prevent path traversal attacks
/// 
/// Returns true if the filename is safe, false otherwise
pub fn is_valid_filename(filename: &str) -> bool {
    // Reject if contains path traversal patterns
    for pattern in PATH_TRAVERSAL_PATTERNS {
        if filename.contains(pattern) {
            return false;
        }
    }
    
    // Reject absolute paths
    if filename.starts_with('/') || filename.starts_with('\\') {
        return false;
    }
    
    // Reject unsafe characters
    for unsafe_char in UNSAFE_PATH_CHARS {
        if filename.contains(*unsafe_char) {
            return false;
        }
    }
    
    // Only allow safe characters
    filename.chars().all(|c| {
        c.is_alphanumeric() || 
        c == '-' || 
        c == '_' || 
        c == '.' || 
        c == '/'  // Allow forward slash for subdirectories
    })
}

/// Sanitize a path component by removing dangerous characters
/// 
/// This is used for user-provided organization and model names
pub fn sanitize_path_component(component: &str) -> String {
    component
        .replace("..", "_")
        .replace('/', "_")
        .replace('\\', "_")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
        .collect::<String>()
}

/// Verify that a constructed path stays within the base directory
/// 
/// Returns Ok(canonical_path) if safe, Err if path traversal detected
pub fn verify_path_safety(base_dir: &Path, target_path: &Path) -> Result<PathBuf, String> {
    // Get canonical paths to resolve any symlinks or .. components
    let canonical_base = base_dir.canonicalize()
        .map_err(|e| format!("Failed to canonicalize base directory: {}", e))?;
    
    // If target doesn't exist yet, check its parent
    let canonical_target = if target_path.exists() {
        target_path.canonicalize()
            .map_err(|e| format!("Failed to canonicalize target path: {}", e))?
    } else {
        // For non-existent paths, resolve the parent and append the filename
        if let Some(parent) = target_path.parent() {
            if parent.exists() {
                let canonical_parent = parent.canonicalize()
                    .map_err(|e| format!("Failed to canonicalize parent: {}", e))?;
                
                if let Some(file_name) = target_path.file_name() {
                    canonical_parent.join(file_name)
                } else {
                    return Err("Invalid target path: no filename".to_string());
                }
            } else {
                // If parent doesn't exist, just ensure the path would be under base
                let mut result = canonical_base.clone();
                for component in target_path.components() {
                    match component {
                        std::path::Component::Normal(name) => {
                            result.push(name);
                        }
                        std::path::Component::ParentDir => {
                            return Err("Path contains '..' component".to_string());
                        }
                        _ => {}
                    }
                }
                result
            }
        } else {
            return Err("Invalid target path: no parent directory".to_string());
        }
    };
    
    // Verify the target is under the base directory
    if !canonical_target.starts_with(&canonical_base) {
        return Err(format!(
            "Path traversal detected: {} is not under {}",
            canonical_target.display(),
            canonical_base.display()
        ));
    }
    
    Ok(canonical_target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    
    #[test]
    fn test_valid_filenames() {
        assert!(is_valid_filename("model.safetensors"));
        assert!(is_valid_filename("config.json"));
        assert!(is_valid_filename("model-0001-of-0002.safetensors"));
        assert!(is_valid_filename("subfolder/model.bin"));
    }
    
    #[test]
    fn test_invalid_filenames() {
        assert!(!is_valid_filename("../etc/passwd"));
        assert!(!is_valid_filename("./../../etc/passwd"));
        assert!(!is_valid_filename("/etc/passwd"));
        assert!(!is_valid_filename("\\windows\\system32"));
        assert!(!is_valid_filename("file\0name"));
        assert!(!is_valid_filename(".."));
    }
    
    #[test]
    fn test_sanitize_path_component() {
        assert_eq!(sanitize_path_component("normal-name_123"), "normal-name_123");
        assert_eq!(sanitize_path_component("../../../etc"), "_etc");
        assert_eq!(sanitize_path_component("path/with/slashes"), "path_with_slashes");
        assert_eq!(sanitize_path_component("path\\with\\backslashes"), "path_with_backslashes");
    }
}