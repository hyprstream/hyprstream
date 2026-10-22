#[path = "../src/build_info.rs"]
mod build_info;

use build_info::{build_version, resolve, source_archive_info, GitInfo};

const VALID_COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

#[test]
fn valid_source_commit_uses_seven_character_prefix() {
    assert_eq!(
        source_archive_info(VALID_COMMIT).ok(),
        Some(GitInfo {
            sha: "0123456".to_owned(),
            branch: String::new(),
            dirty: false
        })
    );
}

#[test]
fn invalid_source_commit_is_rejected() {
    for value in [
        "",
        "0123456",
        "0123456789abcdef0123456789abcdef0123456G",
        "0123456789ABCDEF0123456789abcdef01234567",
    ] {
        assert!(
            source_archive_info(value).is_err(),
            "accepted invalid commit {value:?}"
        );
    }
}

#[test]
fn source_commit_takes_precedence_over_git_probe() {
    let fallback = GitInfo {
        sha: "githead".to_owned(),
        branch: "feature/x".to_owned(),
        dirty: true,
    };
    assert_eq!(
        resolve(Some(VALID_COMMIT), fallback).map(|info| info.sha),
        Ok("0123456".to_owned())
    );
    assert_eq!(
        resolve(
            Some(VALID_COMMIT),
            GitInfo {
                sha: "githead".to_owned(),
                branch: "feature/x".to_owned(),
                dirty: true
            }
        )
        .map(|info| info.branch),
        Ok(String::new())
    );
}

#[test]
fn source_archive_is_branchless_and_clean() {
    let info = source_archive_info(VALID_COMMIT);
    assert_eq!(
        info.as_ref().ok().map(|value| value.branch.as_str()),
        Some("")
    );
    assert_eq!(info.as_ref().ok().map(|value| value.dirty), Some(false));
    assert_eq!(
        source_archive_info(VALID_COMMIT).map(|value| build_version("0.5.0", &value, "")),
        Ok("0.5.0+g0123456".to_owned())
    );
}

#[test]
fn absent_source_commit_preserves_git_fallback() {
    let fallback = GitInfo {
        sha: "abc1234".to_owned(),
        branch: "feature-auth".to_owned(),
        dirty: true,
    };
    assert_eq!(resolve(None, fallback.clone()), Ok(fallback.clone()));
    assert_eq!(
        build_version("0.5.0", &fallback, "feature-auth"),
        "0.5.0+feature-auth.gabc1234.dirty"
    );
}
