//! Core types for GitTorrent

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Fixed-width domain prefixes for DHT key derivation (4 bytes each)
/// These ensure SHA1 and SHA256 git objects are stored in separate DHT keyspaces
const DOMAIN_SHA1: &[u8; 4] = b"sha1";
const DOMAIN_S256: &[u8; 4] = b"s256";

/// A Git object hash - either SHA1 (legacy) or SHA256 (modern)
///
/// Git traditionally uses SHA1 (20 bytes / 40 hex chars) for object hashes.
/// Modern git supports SHA256 (32 bytes / 64 hex chars) but tooling support varies.
/// This enum supports both, auto-detecting by length when parsing.
///
/// For DHT storage, use `to_dht_key()` which derives a 256-bit key with domain
/// separation per the libp2p Kademlia spec.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GitHash {
    /// SHA1 hash (20 bytes / 40 hex chars) - legacy git format
    Sha1([u8; 20]),
    /// SHA256 hash (32 bytes / 64 hex chars) - modern git format
    Sha256([u8; 32]),
}

impl GitHash {
    /// Parse from hex string (auto-detects algorithm by length)
    ///
    /// - 40 hex chars -> SHA1
    /// - 64 hex chars -> SHA256
    /// - Other lengths -> Error
    pub fn from_hex(s: &str) -> crate::Result<Self> {
        let bytes = hex::decode(s).map_err(|_| crate::Error::InvalidHash(s.to_string()))?;
        match bytes.len() {
            20 => {
                let mut arr = [0u8; 20];
                arr.copy_from_slice(&bytes);
                Ok(GitHash::Sha1(arr))
            }
            32 => {
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                Ok(GitHash::Sha256(arr))
            }
            _ => Err(crate::Error::InvalidHash(s.to_string())),
        }
    }

    /// Create from raw SHA1 bytes
    pub fn from_sha1_bytes(bytes: [u8; 20]) -> Self {
        GitHash::Sha1(bytes)
    }

    /// Create from raw SHA256 bytes
    pub fn from_sha256_bytes(bytes: [u8; 32]) -> Self {
        GitHash::Sha256(bytes)
    }

    /// Derive DHT key using domain separation per libp2p Kademlia spec
    ///
    /// The DHT key is: `sha256(DOMAIN_PREFIX || RAW_HASH_BYTES)`
    ///
    /// This ensures:
    /// - Both SHA1 and SHA256 hashes map to 256-bit DHT keys
    /// - Keys are in separate DHT namespaces (no collisions)
    /// - Forward-compatible when git switches to SHA256
    pub fn to_dht_key(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        match self {
            GitHash::Sha1(bytes) => {
                hasher.update(DOMAIN_SHA1);
                hasher.update(bytes);
            }
            GitHash::Sha256(bytes) => {
                hasher.update(DOMAIN_S256);
                hasher.update(bytes);
            }
        }
        hasher.finalize().into()
    }

    /// Convert to hex string representation
    pub fn to_hex(&self) -> String {
        match self {
            GitHash::Sha1(b) => hex::encode(b),
            GitHash::Sha256(b) => hex::encode(b),
        }
    }

    /// Get the raw bytes
    pub fn as_bytes(&self) -> &[u8] {
        match self {
            GitHash::Sha1(b) => b,
            GitHash::Sha256(b) => b,
        }
    }

    /// Check if this is a SHA256 hash
    pub fn is_sha256(&self) -> bool {
        matches!(self, GitHash::Sha256(_))
    }

    /// Check if this is a SHA1 hash
    pub fn is_sha1(&self) -> bool {
        matches!(self, GitHash::Sha1(_))
    }
}

impl std::fmt::Display for GitHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

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
    ///
    /// Returns empty Vec if hex decoding fails (should never happen as
    /// MutableKey is validated in constructor)
    pub fn to_bytes(&self) -> Vec<u8> {
        hex::decode(&self.0).unwrap_or_default()
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
    ///
    /// Returns empty Vec if hex decoding fails (should never happen as
    /// Sha256Hash is validated in constructor)
    pub fn to_bytes(&self) -> Vec<u8> {
        hex::decode(&self.0).unwrap_or_default()
    }
}

impl std::fmt::Display for Sha256Hash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A Git reference (branch/tag) with its hash (SHA1 or SHA256)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitRef {
    pub name: String,
    pub hash: GitHash,
}

/// Git references metadata for a repository
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitRefs {
    /// Git references (branches, tags)
    pub refs: std::collections::HashMap<String, GitHash>,
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

            Ok(GitTorrentUrl::GitServer {
                server: server.to_string(),
                repo: repo.to_string(),
            })
        } else {
            // No slash, assume it's a username
            Ok(GitTorrentUrl::Username {
                username: main_part.to_string(),
            })
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

impl std::fmt::Display for GitTorrentUrl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GitTorrentUrl::Commit { hash } => {
                write!(f, "gittorrent://{}", hash)
            }
            GitTorrentUrl::CommitWithRefs { hash } => {
                write!(f, "gittorrent://{}?refs", hash)
            }
            GitTorrentUrl::GitServer { server, repo } => {
                write!(f, "gittorrent://{}/{}", server, repo)
            }
            GitTorrentUrl::Username { username } => {
                write!(f, "gittorrent://{}", username)
            }
        }
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

    #[test]
    fn test_git_hash_sha1() {
        // 40 hex chars = SHA1
        let sha1_hex = "0123456789abcdef0123456789abcdef01234567";
        let hash = GitHash::from_hex(sha1_hex).unwrap();

        assert!(hash.is_sha1());
        assert!(!hash.is_sha256());
        assert_eq!(hash.to_hex(), sha1_hex);
        assert_eq!(hash.as_bytes().len(), 20);
    }

    #[test]
    fn test_git_hash_sha256() {
        // 64 hex chars = SHA256
        let sha256_hex = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        let hash = GitHash::from_hex(sha256_hex).unwrap();

        assert!(hash.is_sha256());
        assert!(!hash.is_sha1());
        assert_eq!(hash.to_hex(), sha256_hex);
        assert_eq!(hash.as_bytes().len(), 32);
    }

    #[test]
    fn test_git_hash_invalid() {
        // Too short
        assert!(GitHash::from_hex("0123456789").is_err());
        // Too long
        assert!(GitHash::from_hex("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef00").is_err());
        // Invalid hex chars
        assert!(GitHash::from_hex("ghijklmnopqrstuvwxyz0123456789abcdef01234567").is_err());
        // Wrong length (not 40 or 64)
        assert!(GitHash::from_hex("0123456789abcdef0123456789abcdef012345678").is_err()); // 41 chars
    }

    #[test]
    fn test_git_hash_dht_key_domain_separation() {
        // Same first 20 bytes, different algorithm -> different DHT keys
        // SHA1: 40 hex chars, SHA256: 64 hex chars (same prefix + 24 zero chars)
        let sha1_hex = "0123456789abcdef0123456789abcdef01234567";
        let sha256_hex = "0123456789abcdef0123456789abcdef01234567000000000000000000000000";

        let sha1_hash = GitHash::from_hex(sha1_hex).unwrap();
        let sha256_hash = GitHash::from_hex(sha256_hex).unwrap();

        let sha1_dht_key = sha1_hash.to_dht_key();
        let sha256_dht_key = sha256_hash.to_dht_key();

        // DHT keys should be different due to domain separation
        assert_ne!(sha1_dht_key, sha256_dht_key);

        // Both DHT keys should be 32 bytes (256-bit)
        assert_eq!(sha1_dht_key.len(), 32);
        assert_eq!(sha256_dht_key.len(), 32);
    }

    #[test]
    fn test_git_hash_dht_key_deterministic() {
        let sha1_hex = "0123456789abcdef0123456789abcdef01234567";
        let hash1 = GitHash::from_hex(sha1_hex).unwrap();
        let hash2 = GitHash::from_hex(sha1_hex).unwrap();

        // Same input -> same DHT key
        assert_eq!(hash1.to_dht_key(), hash2.to_dht_key());
    }

    #[test]
    fn test_git_hash_display() {
        let sha1_hex = "0123456789abcdef0123456789abcdef01234567";
        let hash = GitHash::from_hex(sha1_hex).unwrap();
        assert_eq!(format!("{}", hash), sha1_hex);
    }

    #[test]
    fn test_git_hash_from_bytes() {
        let sha1_bytes = [0u8; 20];
        let sha256_bytes = [0u8; 32];

        let sha1 = GitHash::from_sha1_bytes(sha1_bytes);
        let sha256 = GitHash::from_sha256_bytes(sha256_bytes);

        assert!(sha1.is_sha1());
        assert!(sha256.is_sha256());
    }

}