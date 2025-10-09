//! Core types for GitTorrent

use serde::{Deserialize, Serialize};

/// A mutable key derived from a public key (SHA256 hash)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MutableKey(pub String);

impl MutableKey {
    /// Create a new mutable key from a hex string
    pub fn new(s: impl Into<String>) -> crate::Result<Self> {
        let s = s.into();
        if s.len() != 64 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(crate::Error::InvalidMutableKey(s));
        }
        Ok(MutableKey(s))
    }

    /// Get the hex string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        hex::decode(&self.0).expect("MutableKey should always be valid hex")
    }
}


/// A Git SHA256 hash (64 hex characters)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Sha256Hash(pub String);

impl Sha256Hash {
    /// Create a new SHA256 from a hex string
    pub fn new(s: impl Into<String>) -> crate::Result<Self> {
        let s = s.into();
        if s.len() != 64 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(crate::Error::InvalidSha256(s));
        }
        Ok(Sha256Hash(s))
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        if bytes.len() != 32 {
            return Err(crate::Error::InvalidSha256(format!("Expected 32 bytes, got {}", bytes.len())));
        }
        Ok(Sha256Hash(hex::encode(bytes)))
    }

    /// Get the hex string representation
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Convert to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        hex::decode(&self.0).expect("SHA256 hash should always be valid hex")
    }
}

impl std::fmt::Display for Sha256Hash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A Git reference (branch/tag) with its SHA256
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitRef {
    pub name: String,
    pub sha256: Sha256Hash,
}

/// Git references metadata for a repository
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitRefs {
    /// Git references (branches, tags)
    pub refs: std::collections::HashMap<String, Sha256Hash>,
    /// Current HEAD reference
    pub head: String,
    /// Creation timestamp
    pub created_at: u64,
}



/// GitTorrent URL variants
#[derive(Debug, Clone)]
pub enum GitTorrentUrl {
    /// gittorrent://COMMIT_SHA256 (pure commit-based cloning)
    Commit { hash: Sha256Hash },
    /// gittorrent://COMMIT_SHA256?refs (commit + references)
    CommitWithRefs { hash: Sha256Hash },
    /// gittorrent://server/repo (legacy: uses git:// for discovery)
    GitServer { server: String, repo: String },
    /// gittorrent://username (External lookup)
    Username { username: String },
}

impl GitTorrentUrl {
    /// Parse a gittorrent:// URL
    pub fn parse(url: &str) -> crate::Result<Self> {
        if !url.starts_with("gittorrent://") {
            return Err(crate::Error::InvalidUrl(format!(
                "URL must start with gittorrent://: {}",
                url
            )));
        }

        let url = &url[13..]; // Remove "gittorrent://"

        // Check for query parameters
        let (main_part, query) = if let Some(q_pos) = url.find('?') {
            (&url[..q_pos], Some(&url[q_pos + 1..]))
        } else {
            (url, None)
        };

        // Check if it's a 64-char hex string (SHA256 hash)
        if main_part.len() == 64 && main_part.chars().all(|c| c.is_ascii_hexdigit()) {
            let hash = Sha256Hash::new(main_part)?;

            // Check for refs parameter
            if let Some(query_str) = query {
                if query_str == "refs" {
                    return Ok(GitTorrentUrl::CommitWithRefs { hash });
                }
            }

            return Ok(GitTorrentUrl::Commit { hash });
        }

        // Handle legacy formats
        if let Some(slash_pos) = main_part.find('/') {
            let (server, repo) = main_part.split_at(slash_pos);
            let repo = &repo[1..]; // Remove leading slash

            return Ok(GitTorrentUrl::GitServer {
                server: server.to_string(),
                repo: repo.to_string(),
            });
        } else {
            // No slash, assume it's a username
            return Ok(GitTorrentUrl::Username {
                username: main_part.to_string(),
            });
        }
    }

    /// Convert back to URL string
    pub fn to_string(&self) -> String {
        match self {
            GitTorrentUrl::Commit { hash } => {
                format!("gittorrent://{}", hash)
            }
            GitTorrentUrl::CommitWithRefs { hash } => {
                format!("gittorrent://{}?refs", hash)
            }
            GitTorrentUrl::GitServer { server, repo } => {
                format!("gittorrent://{}/{}", server, repo)
            }
            GitTorrentUrl::Username { username } => {
                format!("gittorrent://{}", username)
            }
        }
    }

    /// Get the commit hash if this is a commit-based URL
    pub fn commit_hash(&self) -> Option<&Sha256Hash> {
        match self {
            GitTorrentUrl::Commit { hash } | GitTorrentUrl::CommitWithRefs { hash } => Some(hash),
            _ => None,
        }
    }

    /// Check if this URL includes references
    pub fn includes_refs(&self) -> bool {
        matches!(self, GitTorrentUrl::CommitWithRefs { .. })
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_creation() {
        let sha256 = Sha256Hash::new("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef").unwrap();
        assert_eq!(sha256.as_str(), "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
    }

    #[test]
    fn test_invalid_sha256() {
        assert!(Sha256Hash::new("invalid").is_err());
        assert!(Sha256Hash::new("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdeg").is_err()); // invalid char
        assert!(Sha256Hash::new("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcde").is_err()); // too short
    }

    #[test]
    fn test_gittorrent_url_parsing() {
        // Test commit hash URL (new format)
        let url = GitTorrentUrl::parse("gittorrent://0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef").unwrap();
        match url {
            GitTorrentUrl::Commit { hash } => {
                assert_eq!(hash.as_str(), "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
            }
            _ => panic!("Expected Commit variant"),
        }

        // Test commit hash with refs URL
        let url = GitTorrentUrl::parse("gittorrent://0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef?refs").unwrap();
        match url {
            GitTorrentUrl::CommitWithRefs { hash } => {
                assert_eq!(hash.as_str(), "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
            }
            _ => panic!("Expected CommitWithRefs variant"),
        }

        // Test git server URL (legacy)
        let url = GitTorrentUrl::parse("gittorrent://github.com/user/repo").unwrap();
        match url {
            GitTorrentUrl::GitServer { server, repo } => {
                assert_eq!(server, "github.com");
                assert_eq!(repo, "user/repo");
            }
            _ => panic!("Expected GitServer variant"),
        }

        // Test username URL
        let url = GitTorrentUrl::parse("gittorrent://username").unwrap();
        match url {
            GitTorrentUrl::Username { username } => {
                assert_eq!(username, "username");
            }
            _ => panic!("Expected Username variant"),
        }
    }

    #[test]
    fn test_gittorrent_url_roundtrip() {
        let test_hash = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

        // Test commit URL
        let url1 = GitTorrentUrl::parse(&format!("gittorrent://{}", test_hash)).unwrap();
        assert_eq!(url1.to_string(), format!("gittorrent://{}", test_hash));

        // Test commit with refs URL
        let url2 = GitTorrentUrl::parse(&format!("gittorrent://{}?refs", test_hash)).unwrap();
        assert_eq!(url2.to_string(), format!("gittorrent://{}?refs", test_hash));
    }

}