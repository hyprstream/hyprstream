//! Utility functions for git2db

use std::path::Path;

// Note: Use safe_path::scoped_join directly for path operations

/// Format file size in human-readable format
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB", "EB"];
    const THRESHOLD: f64 = 1024.0;

    if bytes == 0 {
        return "0 B".to_string();
    }

    let mut size = bytes as f64;
    let mut unit_index = 0;

    while size >= THRESHOLD && unit_index < UNITS.len() - 1 {
        size /= THRESHOLD;
        unit_index += 1;
    }

    if unit_index == 0 {
        format!("{} {}", bytes, UNITS[unit_index])
    } else {
        format!("{:.1} {}", size, UNITS[unit_index])
    }
}

/// Calculate directory size recursively
pub fn calculate_directory_size<P: AsRef<Path>>(path: P) -> std::io::Result<u64> {
    let mut total_size = 0;

    fn visit_dir(dir: &Path, total_size: &mut u64) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            if metadata.is_file() {
                *total_size += metadata.len();
            } else if metadata.is_dir() {
                visit_dir(&entry.path(), total_size)?;
            }
        }
        Ok(())
    }

    visit_dir(path.as_ref(), &mut total_size)?;
    Ok(total_size)
}

/// Extract repository name from URL
pub fn extract_repo_name(url: &str) -> String {
    let name = url
        .trim_end_matches('/')
        .trim_end_matches(".git")
        .split('/')
        .next_back()
        .unwrap_or("unknown");

    // If we get an empty string, use the fallback
    if name.is_empty() {
        "unknown".to_string()
    } else {
        name.to_string()
    }
}

/// Check if a string looks like a commit hash
pub fn is_commit_hash(s: &str) -> bool {
    s.len() >= 7 && s.len() <= 40 && s.chars().all(|c| c.is_ascii_hexdigit())
}

/// Retry function with exponential backoff
pub async fn retry_with_backoff<F, Fut, T, E>(
    mut f: F,
    max_retries: usize,
    base_delay: std::time::Duration,
    max_delay: std::time::Duration,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    let mut delay = base_delay;

    for attempt in 0..max_retries {
        match f().await {
            Ok(result) => return Ok(result),
            Err(err) => {
                if attempt == max_retries - 1 {
                    return Err(err);
                }

                tracing::debug!(
                    "Attempt {} failed: {}. Retrying in {:?}...",
                    attempt + 1,
                    err,
                    delay
                );

                tokio::time::sleep(delay).await;

                // Exponential backoff with jitter
                delay = std::cmp::min(delay * 2, max_delay);
                let jitter = delay / 10; // 10% jitter
                                         // Simple jitter without external crate
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                let mut hasher = DefaultHasher::new();
                std::time::SystemTime::now().hash(&mut hasher);
                let jitter_millis = jitter.as_millis() as u64;
                let random = if jitter_millis > 0 {
                    hasher.finish() % jitter_millis
                } else {
                    0
                };
                delay += std::time::Duration::from_millis(random);
            }
        }
    }

    unreachable!()
}

/// Create a temporary directory with a specific prefix
pub fn create_temp_dir(prefix: &str) -> std::io::Result<tempfile::TempDir> {
    tempfile::Builder::new()
        .prefix(&format!("git2db-{}-", prefix))
        .tempdir()
}

/// Check if git is available in PATH
pub fn check_git_availability() -> bool {
    std::process::Command::new("git")
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_extract_repo_name() {
        assert_eq!(
            extract_repo_name("https://github.com/user/repo.git"),
            "repo"
        );
        assert_eq!(extract_repo_name("https://github.com/user/repo"), "repo");
        assert_eq!(extract_repo_name("git@github.com:user/repo.git"), "repo");
        assert_eq!(
            extract_repo_name("https://example.com/path/to/repo/"),
            "repo"
        );
    }

    #[test]
    fn test_is_commit_hash() {
        assert!(is_commit_hash("1234567"));
        assert!(is_commit_hash("1234567890abcdef1234567890abcdef12345678"));
        assert!(is_commit_hash("abcdef1234567890"));

        assert!(!is_commit_hash("123456")); // Too short
        assert!(!is_commit_hash("1234567890abcdef1234567890abcdef123456789")); // Too long
        assert!(!is_commit_hash("123456g")); // Invalid character
        assert!(!is_commit_hash("main")); // Not hex
    }

    #[test]
    fn test_safe_path_usage() {
        let temp_dir = tempdir().unwrap();
        let base = temp_dir.path();

        // Create a subdirectory
        let sub_dir = base.join("subdir");
        std::fs::create_dir(&sub_dir).unwrap();

        // All safe_path::scoped_join calls should succeed because they constrain paths within base
        let result1 = safe_path::scoped_join(base, "subdir").unwrap();
        assert!(result1.starts_with(base));
        assert!(result1.ends_with("subdir"));

        let result2 = safe_path::scoped_join(base, "./subdir").unwrap();
        assert!(result2.starts_with(base));

        // Even traversal attempts are constrained to the base directory
        let result3 = safe_path::scoped_join(base, "../outside").unwrap();
        assert!(result3.starts_with(base));
        assert!(result3.ends_with("outside"));

        // Absolute paths get scoped relative to base
        let result4 = safe_path::scoped_join(base, "/tmp").unwrap();
        assert!(result4.starts_with(base));
        assert!(result4.ends_with("tmp"));
    }

    #[test]
    fn test_calculate_directory_size() {
        let temp_dir = tempdir().unwrap();
        let test_file = temp_dir.path().join("test.txt");

        std::fs::write(&test_file, "Hello, world!").unwrap();

        let size = calculate_directory_size(temp_dir.path()).unwrap();
        assert_eq!(size, 13); // "Hello, world!" is 13 bytes
    }

    #[tokio::test]
    async fn test_retry_with_backoff() {
        let mut attempts = 0;

        let result = retry_with_backoff(
            || {
                attempts += 1;
                async move {
                    if attempts < 3 {
                        Err("fail")
                    } else {
                        Ok("success")
                    }
                }
            },
            5,
            std::time::Duration::from_millis(1),
            std::time::Duration::from_millis(100),
        )
        .await;

        assert_eq!(result, Ok("success"));
        assert_eq!(attempts, 3);
    }

    #[tokio::test]
    async fn test_retry_with_backoff_max_retries() {
        let mut attempts = 0;

        let result = retry_with_backoff(
            || {
                attempts += 1;
                async move { Err::<&str, &str>("always fail") }
            },
            3,
            std::time::Duration::from_millis(1),
            std::time::Duration::from_millis(10),
        )
        .await;

        assert_eq!(result, Err("always fail"));
        assert_eq!(attempts, 3);
    }

    #[test]
    fn test_create_temp_dir() {
        let temp_dir = create_temp_dir("test").unwrap();
        let path = temp_dir.path();

        assert!(path.exists());
        assert!(path.is_dir());
        assert!(path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .contains("git2db-test-"));
    }

    #[test]
    fn test_check_git_availability() {
        // This might pass or fail depending on whether git is installed
        // We just test that the function doesn't panic
        let _is_available = check_git_availability();
    }

    #[test]
    fn test_format_bytes_edge_cases() {
        // Test extreme values
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1025), "1.0 KB");
        assert_eq!(format_bytes(u64::MAX), "16.0 EB"); // Should use exabytes for readability
    }

    #[test]
    fn test_extract_repo_name_edge_cases() {
        assert_eq!(extract_repo_name(""), "unknown"); // Empty string should use fallback
        assert_eq!(extract_repo_name("/"), "unknown"); // Just slash should use fallback
        assert_eq!(extract_repo_name("repo"), "repo");
        assert_eq!(extract_repo_name("path/to/repo/"), "repo");
        assert_eq!(
            extract_repo_name("https://github.com/user/repo.git/"),
            "repo"
        );
        assert_eq!(
            extract_repo_name("ssh://git@example.com/user/repo.git"),
            "repo"
        );
    }

    #[test]
    fn test_is_commit_hash_edge_cases() {
        // Boundary cases
        assert!(is_commit_hash("1234567")); // Minimum length
        assert!(is_commit_hash("1234567890abcdef1234567890abcdef12345678")); // Maximum length
        assert!(!is_commit_hash("123456")); // Below minimum
        assert!(!is_commit_hash("1234567890abcdef1234567890abcdef123456789")); // Above maximum

        // Mixed case
        assert!(is_commit_hash("1234567890ABCDEF"));
        assert!(is_commit_hash("1234567890aBcDeF"));

        // Edge characters
        assert!(!is_commit_hash("1234567G")); // 'G' is not hex
        assert!(!is_commit_hash("1234567 ")); // Space
    }

    #[test]
    fn test_safe_path_edge_cases() {
        let temp_dir = tempdir().unwrap();
        let base = temp_dir.path();

        // safe_path::scoped_join constrains all paths to remain within base
        let result1 = safe_path::scoped_join(base, "non_existent_subdir").unwrap();
        assert!(result1.starts_with(base));

        let result2 = safe_path::scoped_join(base, "future/nested/path").unwrap();
        assert!(result2.starts_with(base));

        // Even potentially dangerous paths get constrained
        let result3 = safe_path::scoped_join(base, "/etc/passwd").unwrap();
        assert!(result3.starts_with(base));
        assert!(result3.ends_with("etc/passwd"));

        let result4 = safe_path::scoped_join(base, "../../../etc").unwrap();
        assert!(result4.starts_with(base));
    }

    #[test]
    fn test_calculate_directory_size_nested() {
        let temp_dir = tempdir().unwrap();
        let base = temp_dir.path();

        // Create nested structure
        let sub_dir = base.join("subdir");
        std::fs::create_dir(&sub_dir).unwrap();

        let file1 = base.join("file1.txt");
        let file2 = sub_dir.join("file2.txt");

        std::fs::write(&file1, "Hello").unwrap(); // 5 bytes
        std::fs::write(&file2, "World!").unwrap(); // 6 bytes

        let total_size = calculate_directory_size(base).unwrap();
        assert_eq!(total_size, 11); // 5 + 6 bytes
    }

    #[test]
    fn test_calculate_directory_size_empty() {
        let temp_dir = tempdir().unwrap();
        let size = calculate_directory_size(temp_dir.path()).unwrap();
        assert_eq!(size, 0);
    }

    #[tokio::test]
    async fn test_retry_with_backoff_immediate_success() {
        let mut attempts = 0;

        let result = retry_with_backoff(
            || {
                attempts += 1;
                async move { Ok::<&str, &str>("immediate success") }
            },
            3,
            std::time::Duration::from_millis(1),
            std::time::Duration::from_millis(10),
        )
        .await;

        assert_eq!(result, Ok("immediate success"));
        assert_eq!(attempts, 1);
    }

    #[tokio::test]
    async fn test_retry_with_backoff_delay_progression() {
        use std::time::Instant;
        let mut attempts = 0;
        let start = Instant::now();

        let _result = retry_with_backoff(
            || {
                attempts += 1;
                async move { Err::<&str, &str>("fail") }
            },
            3,
            std::time::Duration::from_millis(10),
            std::time::Duration::from_millis(100),
        )
        .await;

        let elapsed = start.elapsed();
        // Should take at least the base delay times (with some tolerance)
        assert!(elapsed >= std::time::Duration::from_millis(5)); // Allow for jitter and overhead
        assert_eq!(attempts, 3);
    }
}
