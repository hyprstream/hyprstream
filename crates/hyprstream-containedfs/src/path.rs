//! Path construction and validation utilities.

use crate::error::FsError;
use std::path::{Component, Path, PathBuf};

/// Join an untrusted path to a base directory with traversal clamping.
///
/// Component-walk normalization:
/// - Skips `.` components
/// - Clamps `..` at root (never goes above base)
/// - Strips `RootDir` and `Prefix` components
/// - Rejects NUL bytes in the raw `&str` (checked before Path conversion)
///
/// Use this for **constructing** paths where `..` should be harmless (clamped).
/// For I/O where traversal is never valid, use [`validate_relative_path`] instead.
pub fn contained_join(base: &Path, untrusted: &str) -> Result<PathBuf, FsError> {
    // Check for NUL bytes in raw string (before Path conversion which may strip them)
    if untrusted.as_bytes().contains(&0) {
        return Err(FsError::path_escape("NUL byte in path"));
    }

    let mut result = base.to_path_buf();

    for component in Path::new(untrusted).components() {
        match component {
            Component::Normal(c) => result.push(c),
            Component::CurDir | Component::RootDir | Component::Prefix(_) => {} // skip ".", strip absolute/prefix
            Component::ParentDir => {
                // Clamp at base — never go above it
                if result != base {
                    result.pop();
                }
            }
        }
    }

    Ok(result)
}

/// Validate that a relative path doesn't contain traversal components.
///
/// **Rejects** (not clamps) `..` and absolute paths. Use this for I/O
/// operations where traversal is never valid.
///
/// Returns the validated path as a `PathBuf` on success.
pub fn validate_relative_path(path: &str) -> Result<PathBuf, FsError> {
    // Check for NUL bytes
    if path.as_bytes().contains(&0) {
        return Err(FsError::path_escape("NUL byte in path"));
    }

    let rel = Path::new(path);
    if rel.is_absolute() {
        return Err(FsError::path_escape("absolute path rejected"));
    }
    for component in rel.components() {
        match component {
            Component::ParentDir => {
                return Err(FsError::path_escape("path traversal via '..' rejected"));
            }
            Component::RootDir => {
                return Err(FsError::path_escape("root dir component rejected"));
            }
            Component::Prefix(_) => {
                return Err(FsError::path_escape("prefix component rejected"));
            }
            _ => {}
        }
    }
    Ok(rel.to_path_buf())
}

/// Validate a git ref name (branch/tag) for safe use in paths.
///
/// Rules matching `git check-ref-format`:
/// - No `..` anywhere
/// - No ASCII control chars or ` ~ ^ : ? * [ \`
/// - No leading/trailing `.` per component
/// - No consecutive dots `..`
/// - No `@{` sequence
/// - No trailing `.lock`
/// - Cannot be `@` alone
/// - No NUL bytes
/// - Non-empty
pub fn validate_ref_name(name: &str) -> Result<(), FsError> {
    if name.is_empty() {
        return Err(FsError::path_escape("ref name cannot be empty"));
    }

    // Check for NUL bytes
    if name.as_bytes().contains(&0) {
        return Err(FsError::path_escape("NUL byte in ref name"));
    }

    // Cannot be "@" alone
    if name == "@" {
        return Err(FsError::path_escape("ref name cannot be '@' alone"));
    }

    // No ".." anywhere
    if name.contains("..") {
        return Err(FsError::path_escape("ref name cannot contain '..'"));
    }

    // No "@{" sequence
    if name.contains("@{") {
        return Err(FsError::path_escape("ref name cannot contain '@{'"));
    }

    // No trailing ".lock"
    if name.ends_with(".lock") {
        return Err(FsError::path_escape("ref name cannot end with '.lock'"));
    }

    // Check per-character and per-component constraints
    for component in name.split('/') {
        if component.is_empty() {
            return Err(FsError::path_escape("ref name has empty component"));
        }
        // No leading or trailing '.' per component
        if component.starts_with('.') || component.ends_with('.') {
            return Err(FsError::path_escape("ref name component cannot start/end with '.'"));
        }
    }

    // Check for forbidden characters
    for byte in name.bytes() {
        match byte {
            // ASCII control characters (0x00-0x1F, 0x7F)
            0x00..=0x1F | 0x7F => {
                return Err(FsError::path_escape("ref name contains control character"));
            }
            // Forbidden special characters
            b' ' | b'~' | b'^' | b':' | b'?' | b'*' | b'[' | b'\\' => {
                return Err(FsError::path_escape(format!(
                    "ref name contains forbidden character '{}'",
                    byte as char
                )));
            }
            _ => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // ── contained_join tests ─────────────────────────────────────────────

    #[test]
    fn test_contained_join_normal() {
        let base = Path::new("/base");
        assert_eq!(
            contained_join(base, "subdir/file.txt").unwrap_or_default(),
            PathBuf::from("/base/subdir/file.txt")
        );
    }

    #[test]
    fn test_contained_join_traversal_clamped() {
        let base = Path::new("/base");
        // ".." should be clamped at base, never escaping
        assert_eq!(
            contained_join(base, "../../../etc/passwd").unwrap_or_default(),
            PathBuf::from("/base/etc/passwd")
        );
    }

    #[test]
    fn test_contained_join_absolute_stripped() {
        let base = Path::new("/base");
        assert_eq!(
            contained_join(base, "/tmp/evil").unwrap_or_default(),
            PathBuf::from("/base/tmp/evil")
        );
    }

    #[test]
    fn test_contained_join_dot_skipped() {
        let base = Path::new("/base");
        assert_eq!(
            contained_join(base, "./subdir/../other").unwrap_or_default(),
            PathBuf::from("/base/other")
        );
    }

    #[test]
    fn test_contained_join_nul_rejected() {
        let base = Path::new("/base");
        assert!(contained_join(base, "foo\0bar").is_err());
    }

    #[test]
    fn test_contained_join_branch_with_slash() {
        let base = Path::new("/base");
        assert_eq!(
            contained_join(base, "feature/my-branch").unwrap_or_default(),
            PathBuf::from("/base/feature/my-branch")
        );
    }

    #[test]
    fn test_contained_join_unicode() {
        let base = Path::new("/base");
        assert_eq!(
            contained_join(base, "日本語/ファイル").unwrap_or_default(),
            PathBuf::from("/base/日本語/ファイル")
        );
    }

    // ── validate_relative_path tests ─────────────────────────────────────

    #[test]
    fn test_validate_relative_path_clean() {
        assert!(validate_relative_path("subdir/file.txt").is_ok());
    }

    #[test]
    fn test_validate_relative_path_rejects_dotdot() {
        assert!(validate_relative_path("../escape").is_err());
        assert!(validate_relative_path("a/../b").is_err());
    }

    #[test]
    fn test_validate_relative_path_rejects_absolute() {
        assert!(validate_relative_path("/tmp/evil").is_err());
    }

    #[test]
    fn test_validate_relative_path_rejects_nul() {
        assert!(validate_relative_path("foo\0bar").is_err());
    }

    // ── validate_ref_name tests ──────────────────────────────────────────

    #[test]
    fn test_validate_ref_name_valid() {
        assert!(validate_ref_name("main").is_ok());
        assert!(validate_ref_name("feature/my-branch").is_ok());
        assert!(validate_ref_name("v1.0.0").is_ok());
        assert!(validate_ref_name("release/2024-01").is_ok());
    }

    #[test]
    fn test_validate_ref_name_dotdot() {
        assert!(validate_ref_name("a..b").is_err());
    }

    #[test]
    fn test_validate_ref_name_nul() {
        assert!(validate_ref_name("a\0b").is_err());
    }

    #[test]
    fn test_validate_ref_name_at_brace() {
        assert!(validate_ref_name("a@{b").is_err());
    }

    #[test]
    fn test_validate_ref_name_lock() {
        assert!(validate_ref_name("refs/heads/main.lock").is_err());
    }

    #[test]
    fn test_validate_ref_name_control_chars() {
        assert!(validate_ref_name("a\x01b").is_err());
    }

    #[test]
    fn test_validate_ref_name_empty() {
        assert!(validate_ref_name("").is_err());
    }

    #[test]
    fn test_validate_ref_name_at_alone() {
        assert!(validate_ref_name("@").is_err());
    }

    #[test]
    fn test_validate_ref_name_leading_dot() {
        assert!(validate_ref_name(".hidden").is_err());
    }

    #[test]
    fn test_validate_ref_name_forbidden_chars() {
        assert!(validate_ref_name("a~b").is_err());
        assert!(validate_ref_name("a^b").is_err());
        assert!(validate_ref_name("a:b").is_err());
        assert!(validate_ref_name("a?b").is_err());
        assert!(validate_ref_name("a*b").is_err());
        assert!(validate_ref_name("a[b").is_err());
        assert!(validate_ref_name("a\\b").is_err());
        assert!(validate_ref_name("a b").is_err());
    }
}
