//! Validation utilities for paths, filenames, and security checks

use std::path::{Path, PathBuf};
use crate::constants::validation::*;
use anyhow::{Result, bail};
use uuid::Uuid;

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

/// Validate a model reference (model:ref format)
///
/// Ensures the model reference is safe to use
pub fn validate_model_ref(model_ref: &str) -> Result<String> {
    // Check for empty
    if model_ref.is_empty() {
        bail!("Model reference cannot be empty");
    }

    // Check length limits
    if model_ref.len() > 256 {
        bail!("Model reference too long (max 256 characters)");
    }

    // Check for null bytes
    if model_ref.contains('\0') {
        bail!("Model reference contains null byte");
    }

    // Check for path traversal
    if model_ref.contains("..") {
        bail!("Model reference contains directory traversal");
    }

    // Parse and validate components
    let parts: Vec<&str> = model_ref.split(':').collect();
    if parts.len() > 2 {
        bail!("Model reference can only have one colon separator");
    }

    Ok(model_ref.to_string())
}

/// Validate an adapter name
///
/// Ensures adapter names are safe and follow naming conventions
pub fn validate_adapter_name(name: &str) -> Result<String> {
    // Check for empty name
    if name.is_empty() {
        bail!("Adapter name cannot be empty");
    }

    // Check length limits
    if name.len() > 256 {
        bail!("Adapter name too long (max 256 characters)");
    }

    // Check for invalid characters (only allow alphanumeric, dash, underscore)
    if !name.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_') {
        bail!("Adapter name contains invalid characters. Only alphanumeric, dash, and underscore allowed");
    }

    // Check it doesn't start with a dash or underscore
    if name.starts_with('-') || name.starts_with('_') {
        bail!("Adapter name cannot start with dash or underscore");
    }

    Ok(name.to_string())
}

/// Validate a Git reference (branch, tag, or commit)
///
/// Ensures Git references are safe to use
pub fn validate_git_ref(git_ref: &str) -> Result<String> {
    // Check for empty ref
    if git_ref.is_empty() {
        bail!("Git reference cannot be empty");
    }

    // Check length limits
    if git_ref.len() > 256 {
        bail!("Git reference too long (max 256 characters)");
    }

    // Check for dangerous characters that could be interpreted as Git revision syntax
    if git_ref.contains("..") || git_ref.contains("~") || git_ref.contains("^") {
        bail!("Git reference contains potentially dangerous characters");
    }

    // Check for null bytes
    if git_ref.contains('\0') {
        bail!("Git reference contains null byte");
    }

    Ok(git_ref.to_string())
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

    #[test]
    fn test_validate_model_ref() {
        // Valid references
        assert!(validate_model_ref("model-name").is_ok());
        assert!(validate_model_ref("model:main").is_ok());
        assert!(validate_model_ref("model:v1.0").is_ok());
        assert!(validate_model_ref("550e8400-e29b-41d4-a716-446655440000").is_ok());

        // Invalid references
        assert!(validate_model_ref("").is_err());
        assert!(validate_model_ref("../../etc/passwd").is_err());
        assert!(validate_model_ref("model\0name").is_err());
        assert!(validate_model_ref(&"x".repeat(257)).is_err());
        assert!(validate_model_ref("model:ref:extra").is_err());
    }

    #[test]
    fn test_validate_adapter_name() {
        // Valid names
        assert!(validate_adapter_name("my-adapter").is_ok());
        assert!(validate_adapter_name("adapter_v1").is_ok());
        assert!(validate_adapter_name("LoRA123").is_ok());

        // Invalid names
        assert!(validate_adapter_name("").is_err());
        assert!(validate_adapter_name("-adapter").is_err());
        assert!(validate_adapter_name("_adapter").is_err());
        assert!(validate_adapter_name("adapter/name").is_err());
        assert!(validate_adapter_name("adapter name").is_err());
        assert!(validate_adapter_name(&"x".repeat(257)).is_err());
    }

    #[test]
    fn test_validate_git_ref() {
        // Valid refs
        assert!(validate_git_ref("main").is_ok());
        assert!(validate_git_ref("feature-branch").is_ok());
        assert!(validate_git_ref("v1.0.0").is_ok());
        assert!(validate_git_ref("abc123def").is_ok());

        // Invalid refs
        assert!(validate_git_ref("").is_err());
        assert!(validate_git_ref("refs/heads/../../../etc").is_err());
        assert!(validate_git_ref("HEAD~1").is_err());
        assert!(validate_git_ref("main^").is_err());
        assert!(validate_git_ref("branch\0name").is_err());
    }
}