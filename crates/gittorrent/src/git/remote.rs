//! Git remote operations for GitTorrent

use crate::{
    types::{GitHash, GitRef},
    Result,
};
use git2::Repository;
use std::collections::HashMap;
use tokio::process::Command;

/// List references from a remote Git repository
/// This is equivalent to `git ls-remote <url>`
pub async fn ls_remote(url: &str) -> Result<Vec<GitRef>> {
    let output = Command::new("git")
        .args(["ls-remote", url])
        .output()
        .await?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(crate::Error::other(format!("git ls-remote failed: {stderr}")));
    }

    let stdout = String::from_utf8(output.stdout)?;
    parse_ls_remote_output(&stdout)
}

/// Parse the output of `git ls-remote`
fn parse_ls_remote_output(output: &str) -> Result<Vec<GitRef>> {
    let mut refs = Vec::new();

    for line in output.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let hash = GitHash::from_hex(parts[0])?;
            let name = parts[1].to_owned();

            refs.push(GitRef { name, hash });
        }
    }

    Ok(refs)
}

/// List references from a local repository
pub fn list_local_refs(repo_path: &str) -> Result<Vec<GitRef>> {
    let repo = Repository::open(repo_path)?;
    let mut refs = Vec::new();

    for reference in repo.references()? {
        let reference = reference?;

        if let Some(name) = reference.name() {
            if let Some(target) = reference.target() {
                let hash = GitHash::from_hex(&target.to_string())?;
                refs.push(GitRef {
                    name: name.to_owned(),
                    hash,
                });
            }
        }
    }

    Ok(refs)
}

/// Convert references to a map for easier lookup
pub fn refs_to_map(refs: Vec<GitRef>) -> HashMap<String, GitHash> {
    refs.into_iter()
        .map(|git_ref| (git_ref.name, git_ref.hash))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ls_remote_output() -> crate::error::Result<()> {
        // Test with SHA1 hashes (40 hex chars) - current git format
        let output = r#"
0123456789abcdef0123456789abcdef01234567	HEAD
0123456789abcdef0123456789abcdef01234567	refs/heads/main
1123456789abcdef0123456789abcdef01234567	refs/heads/develop
2123456789abcdef0123456789abcdef01234567	refs/tags/v1.0.0
"#;

        let refs = parse_ls_remote_output(output.trim())?;
        assert_eq!(refs.len(), 4);

        assert_eq!(refs[0].name, "HEAD");
        assert_eq!(refs[0].hash.to_hex(), "0123456789abcdef0123456789abcdef01234567");
        assert!(refs[0].hash.is_sha1());

        assert_eq!(refs[1].name, "refs/heads/main");
        assert_eq!(refs[1].hash.to_hex(), "0123456789abcdef0123456789abcdef01234567");

        assert_eq!(refs[2].name, "refs/heads/develop");
        assert_eq!(refs[2].hash.to_hex(), "1123456789abcdef0123456789abcdef01234567");

        assert_eq!(refs[3].name, "refs/tags/v1.0.0");
        assert_eq!(refs[3].hash.to_hex(), "2123456789abcdef0123456789abcdef01234567");
        Ok(())
    }

    #[test]
    fn test_refs_to_map() -> crate::error::Result<()> {
        let refs = vec![
            GitRef {
                name: "refs/heads/main".to_owned(),
                hash: GitHash::from_hex("0123456789abcdef0123456789abcdef01234567")?,
            },
            GitRef {
                name: "refs/heads/develop".to_owned(),
                hash: GitHash::from_hex("1123456789abcdef0123456789abcdef01234567")?,
            },
        ];

        let map = refs_to_map(refs);
        assert_eq!(map.len(), 2);
        assert!(map.contains_key("refs/heads/main"));
        assert!(map.contains_key("refs/heads/develop"));
        Ok(())
    }
}