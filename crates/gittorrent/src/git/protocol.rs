//! Git remote helper protocol implementation

use crate::types::GitRef;
use std::collections::HashMap;

/// Git remote helper capabilities
pub const CAPABILITIES: &[&str] = &["fetch"];

/// Git remote helper command
#[derive(Debug, Clone)]
pub enum GitCommand {
    Capabilities,
    List,
    Fetch { commit_hash: String, refname: String },
}

/// Parse a git remote helper command from a line
pub fn parse_command(line: &str) -> Option<GitCommand> {
    let line = line.trim();

    if line == "capabilities" {
        Some(GitCommand::Capabilities)
    } else if line == "list" {
        Some(GitCommand::List)
    } else if line.starts_with("fetch ") {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 {
            Some(GitCommand::Fetch {
                commit_hash: parts[1].to_owned(),
                refname: parts[2].to_owned(),
            })
        } else {
            None
        }
    } else {
        None
    }
}

/// Handle git remote helper protocol
pub struct GitRemoteHelper {
    refs: HashMap<String, String>,
}

impl GitRemoteHelper {
    /// Create a new git remote helper with the given references
    pub fn new(refs: Vec<GitRef>) -> Self {
        let refs = refs
            .into_iter()
            .map(|git_ref| (git_ref.name, git_ref.hash.to_hex()))
            .collect();

        GitRemoteHelper { refs }
    }

    /// Handle a git remote helper command and return the response
    pub fn handle_command(&self, command: &GitCommand) -> String {
        match command {
            GitCommand::Capabilities => {
                let mut response = String::new();
                for capability in CAPABILITIES {
                    response.push_str(capability);
                    response.push('\n');
                }
                response.push('\n');
                response
            }
            GitCommand::List => {
                let mut response = String::new();
                for (refname, commit_hash) in &self.refs {
                    response.push_str(&format!("{commit_hash} {refname}\n"));
                }
                response.push('\n');
                response
            }
            GitCommand::Fetch { .. } => {
                // For fetch commands, we don't return anything immediately
                // The actual fetching will be handled separately
                String::new()
            }
        }
    }

    /// Get reference by name
    pub fn get_ref(&self, refname: &str) -> Option<&String> {
        self.refs.get(refname)
    }

    /// Get all references
    pub fn refs(&self) -> &HashMap<String, String> {
        &self.refs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::GitHash;

    #[test]
    fn test_parse_command() {
        assert!(matches!(
            parse_command("capabilities"),
            Some(GitCommand::Capabilities)
        ));

        assert!(matches!(parse_command("list"), Some(GitCommand::List)));

        if let Some(GitCommand::Fetch { commit_hash, refname }) =
            parse_command("fetch abc123 refs/heads/main")
        {
            assert_eq!(commit_hash, "abc123");
            assert_eq!(refname, "refs/heads/main");
        } else {
            panic!("Failed to parse fetch command");
        }

        assert!(parse_command("invalid").is_none());
    }

    #[test]
    fn test_git_remote_helper() -> crate::error::Result<()> {
        // Test with SHA1 hashes (40 hex chars) - current git format
        let refs = vec![
            GitRef {
                name: "refs/heads/main".to_owned(),
                hash: GitHash::from_hex("0123456789abcdef0123456789abcdef01234567")?,
            },
            GitRef {
                name: "HEAD".to_owned(),
                hash: GitHash::from_hex("0123456789abcdef0123456789abcdef01234567")?,
            },
        ];

        let helper = GitRemoteHelper::new(refs);

        // Test capabilities
        let caps_response = helper.handle_command(&GitCommand::Capabilities);
        assert!(caps_response.contains("fetch"));

        // Test list
        let list_response = helper.handle_command(&GitCommand::List);
        assert!(list_response.contains("0123456789abcdef0123456789abcdef01234567 refs/heads/main"));
        assert!(list_response.contains("0123456789abcdef0123456789abcdef01234567 HEAD"));

        // Test get_ref
        assert!(helper.get_ref("refs/heads/main").is_some());
        assert!(helper.get_ref("nonexistent").is_none());
        Ok(())
    }
}