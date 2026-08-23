//! Pure build provenance resolution shared by the build script and tests.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GitInfo {
    pub sha: String,
    pub branch: String,
    pub dirty: bool,
}

/// Validate and shorten the trusted source-archive commit override.
pub fn source_archive_info(value: &str) -> Result<GitInfo, String> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(
            "HYPRSTREAM_SOURCE_COMMIT must be exactly 40 lowercase hexadecimal characters"
                .to_owned(),
        );
    }

    Ok(GitInfo {
        sha: value[..7].to_owned(),
        branch: String::new(),
        dirty: false,
    })
}

/// Apply the source-archive override, preserving the live-Git fallback when absent.
pub fn resolve(source_commit: Option<&str>, git_fallback: GitInfo) -> Result<GitInfo, String> {
    source_commit.map_or(Ok(git_fallback), source_archive_info)
}

/// Construct the version string emitted by the build script.
pub fn build_version(cargo_version: &str, info: &GitInfo, sanitized_branch: &str) -> String {
    if info.sha.is_empty() {
        return cargo_version.to_owned();
    }

    let mut version = format!("{cargo_version}+");
    if !sanitized_branch.is_empty() {
        version.push_str(sanitized_branch);
        version.push('.');
    }
    version.push('g');
    version.push_str(&info.sha);
    if info.dirty {
        version.push_str(".dirty");
    }
    version
}
