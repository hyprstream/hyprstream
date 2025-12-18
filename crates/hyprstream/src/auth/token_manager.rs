//! API Token Manager for Bearer authentication
//!
//! Implements OpenAI-compatible token authentication:
//! - Token format: `hypr_{32 alphanumeric chars}`
//! - Tokens hashed (SHA-256) before storage
//! - Stored in `.registry/policies/tokens.csv`
//!
//! Usage:
//! ```text
//! Authorization: Bearer hypr_k8Jx9mPqR2vLnW4tY6uI0oA3sD5fG7hJ
//! ```

use chrono::{DateTime, Duration, Utc};
use rand::Rng;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Token prefix for standard API keys
pub const TOKEN_PREFIX: &str = "hypr_";

/// Token prefix for admin keys
pub const ADMIN_TOKEN_PREFIX: &str = "hypr_admin_";

/// Length of random portion of token
const TOKEN_RANDOM_LENGTH: usize = 32;

/// Errors from token operations
#[derive(Error, Debug)]
pub enum TokenError {
    #[error("Invalid token format")]
    InvalidFormat,

    #[error("Token not found")]
    NotFound,

    #[error("Token expired")]
    Expired,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("CSV error: {0}")]
    CsvError(String),

    #[error("Token already exists for this user with this name")]
    DuplicateName,
}

/// A stored token record
#[derive(Debug, Clone)]
pub struct TokenRecord {
    /// SHA-256 hash of the token
    pub hash: String,
    /// User this token authenticates as
    pub user: String,
    /// Human-readable name for the token
    pub name: String,
    /// When the token was created
    pub created_at: DateTime<Utc>,
    /// When the token expires (None = never)
    pub expires_at: Option<DateTime<Utc>>,
    /// Resource scopes (empty or ["*"] = all resources)
    pub scopes: Vec<String>,
}

impl TokenRecord {
    /// Check if the token has expired
    pub fn is_expired(&self) -> bool {
        match self.expires_at {
            Some(expires) => Utc::now() > expires,
            None => false,
        }
    }

    /// Check if token has access to a resource
    pub fn has_scope(&self, resource: &str) -> bool {
        if self.scopes.is_empty() || self.scopes.contains(&"*".to_string()) {
            return true;
        }
        self.scopes.iter().any(|scope| {
            if scope.ends_with('*') {
                let prefix = &scope[..scope.len() - 1];
                resource.starts_with(prefix)
            } else {
                scope == resource
            }
        })
    }
}

/// Summary of a token for listing (no sensitive data)
#[derive(Debug, Clone)]
pub struct TokenSummary {
    /// First 8 chars of hash for identification
    pub hash_prefix: String,
    /// User this token authenticates as
    pub user: String,
    /// Human-readable name
    pub name: String,
    /// When created
    pub created_at: DateTime<Utc>,
    /// When expires
    pub expires_at: Option<DateTime<Utc>>,
    /// Scopes
    pub scopes: Vec<String>,
}

/// Token manager for API authentication
pub struct TokenManager {
    /// Hash -> TokenRecord mapping
    tokens: HashMap<String, TokenRecord>,
    /// Path to tokens.csv file
    tokens_path: PathBuf,
}

impl TokenManager {
    /// Create a new TokenManager, loading tokens from disk
    pub async fn new(tokens_path: impl AsRef<Path>) -> Result<Self, TokenError> {
        let tokens_path = tokens_path.as_ref().to_path_buf();
        let mut manager = Self {
            tokens: HashMap::new(),
            tokens_path,
        };

        if manager.tokens_path.exists() {
            manager.load().await?;
        } else {
            info!("No tokens file found, starting with empty token store");
        }

        Ok(manager)
    }

    /// Create a TokenManager with no backing file (for testing)
    pub fn in_memory() -> Self {
        Self {
            tokens: HashMap::new(),
            tokens_path: PathBuf::new(),
        }
    }

    /// Load tokens from CSV file
    async fn load(&mut self) -> Result<(), TokenError> {
        let content = tokio::fs::read_to_string(&self.tokens_path).await?;
        self.tokens.clear();

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            match self.parse_csv_line(line) {
                Ok(record) => {
                    self.tokens.insert(record.hash.clone(), record);
                }
                Err(e) => {
                    warn!("Failed to parse token line: {}", e);
                }
            }
        }

        info!("Loaded {} tokens from {}", self.tokens.len(), self.tokens_path.display());
        Ok(())
    }

    /// Parse a CSV line into a TokenRecord
    fn parse_csv_line(&self, line: &str) -> Result<TokenRecord, TokenError> {
        let parts: Vec<&str> = line.splitn(6, ',').map(|s| s.trim()).collect();
        if parts.len() < 4 {
            return Err(TokenError::CsvError(format!("Invalid line: {}", line)));
        }

        let hash = parts[0].to_string();
        let user = parts[1].to_string();
        let name = parts.get(2).unwrap_or(&"").trim_matches('"').to_string();

        let created_at = parts.get(3)
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_else(Utc::now);

        let expires_at = parts.get(4)
            .filter(|s| !s.is_empty())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.with_timezone(&Utc));

        let scopes: Vec<String> = parts.get(5)
            .map(|s| s.split(';').map(|sc| sc.trim().to_string()).filter(|sc| !sc.is_empty()).collect())
            .unwrap_or_else(|| vec!["*".to_string()]);

        Ok(TokenRecord {
            hash,
            user,
            name,
            created_at,
            expires_at,
            scopes,
        })
    }

    /// Save tokens to CSV file
    pub async fn save(&self) -> Result<(), TokenError> {
        if self.tokens_path.as_os_str().is_empty() {
            return Ok(()); // In-memory mode
        }

        // Ensure parent directory exists
        if let Some(parent) = self.tokens_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }

        let mut content = String::new();
        content.push_str("# Hyprstream API Tokens\n");
        content.push_str("# Format: hash, user, name, created_at, expires_at, scopes\n");
        content.push_str("# WARNING: Do not edit manually - use 'hyprstream policy token' commands\n\n");

        for record in self.tokens.values() {
            let expires = record.expires_at
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default();
            let scopes = record.scopes.join(";");
            content.push_str(&format!(
                "{}, {}, \"{}\", {}, {}, {}\n",
                record.hash,
                record.user,
                record.name,
                record.created_at.to_rfc3339(),
                expires,
                scopes
            ));
        }

        tokio::fs::write(&self.tokens_path, content).await?;
        debug!("Saved {} tokens to {}", self.tokens.len(), self.tokens_path.display());
        Ok(())
    }

    /// Validate a token and return the record if valid
    pub fn validate(&self, token: &str) -> Option<&TokenRecord> {
        // Check format
        if !token.starts_with(TOKEN_PREFIX) {
            debug!("Token rejected: invalid prefix");
            return None;
        }

        let hash = Self::hash_token(token);

        match self.tokens.get(&hash) {
            Some(record) => {
                if record.is_expired() {
                    debug!("Token rejected: expired for user {}", record.user);
                    None
                } else {
                    debug!("Token validated for user {}", record.user);
                    Some(record)
                }
            }
            None => {
                debug!("Token rejected: not found");
                None
            }
        }
    }

    /// Create a new token for a user
    ///
    /// Returns the plaintext token (only shown once, never stored)
    pub async fn create_token(
        &mut self,
        user: &str,
        name: &str,
        expires_in: Option<Duration>,
        scopes: Vec<String>,
        admin: bool,
    ) -> Result<String, TokenError> {
        // Check for duplicate name for this user
        if self.tokens.values().any(|t| t.user == user && t.name == name) {
            return Err(TokenError::DuplicateName);
        }

        // Generate random token
        let prefix = if admin { ADMIN_TOKEN_PREFIX } else { TOKEN_PREFIX };
        let random_part: String = rand::thread_rng()
            .sample_iter(&rand::distributions::Alphanumeric)
            .take(TOKEN_RANDOM_LENGTH)
            .map(char::from)
            .collect();
        let token = format!("{}{}", prefix, random_part);

        // Hash for storage
        let hash = Self::hash_token(&token);

        // Calculate expiration
        let expires_at = expires_in.map(|d| Utc::now() + d);

        // Create record
        let record = TokenRecord {
            hash: hash.clone(),
            user: user.to_string(),
            name: name.to_string(),
            created_at: Utc::now(),
            expires_at,
            scopes: if scopes.is_empty() { vec!["*".to_string()] } else { scopes },
        };

        self.tokens.insert(hash, record);
        self.save().await?;

        info!("Created token '{}' for user '{}'", name, user);
        Ok(token)
    }

    /// Revoke a token by prefix match
    ///
    /// Matches against the beginning of the plaintext token
    pub async fn revoke(&mut self, token_prefix: &str) -> Result<TokenRecord, TokenError> {
        // If it looks like a full token, hash it directly
        let hash_to_remove = if token_prefix.starts_with(TOKEN_PREFIX) && token_prefix.len() > 10 {
            Some(Self::hash_token(token_prefix))
        } else {
            None
        };

        // Find and remove the token
        let removed = if let Some(hash) = hash_to_remove {
            self.tokens.remove(&hash)
        } else {
            // Can't match by prefix without storing original tokens
            // This is a limitation - users should provide the full token or use token name
            None
        };

        match removed {
            Some(record) => {
                self.save().await?;
                info!("Revoked token '{}' for user '{}'", record.name, record.user);
                Ok(record)
            }
            None => Err(TokenError::NotFound),
        }
    }

    /// Revoke a token by user and name
    pub async fn revoke_by_name(&mut self, user: &str, name: &str) -> Result<TokenRecord, TokenError> {
        let hash = self.tokens.iter()
            .find(|(_, r)| r.user == user && r.name == name)
            .map(|(h, _)| h.clone());

        match hash {
            Some(h) => {
                let record = self.tokens.remove(&h).unwrap();
                self.save().await?;
                info!("Revoked token '{}' for user '{}'", name, user);
                Ok(record)
            }
            None => Err(TokenError::NotFound),
        }
    }

    /// List all tokens (without sensitive data)
    pub fn list(&self) -> Vec<TokenSummary> {
        self.tokens.values()
            .map(|r| TokenSummary {
                hash_prefix: r.hash.chars().take(12).collect(),
                user: r.user.clone(),
                name: r.name.clone(),
                created_at: r.created_at,
                expires_at: r.expires_at,
                scopes: r.scopes.clone(),
            })
            .collect()
    }

    /// List tokens for a specific user
    pub fn list_for_user(&self, user: &str) -> Vec<TokenSummary> {
        self.list().into_iter().filter(|t| t.user == user).collect()
    }

    /// Get path to tokens file
    pub fn tokens_path(&self) -> &Path {
        &self.tokens_path
    }

    /// Hash a token for storage
    fn hash_token(token: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(token.as_bytes());
        let result = hasher.finalize();
        format!("sha256:{:x}", result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_token() {
        let token = "hypr_testtoken12345678901234567890";
        let hash = TokenManager::hash_token(token);
        assert!(hash.starts_with("sha256:"));
        assert_eq!(hash.len(), 7 + 64); // "sha256:" + 64 hex chars
    }

    #[test]
    fn test_token_record_expired() {
        let mut record = TokenRecord {
            hash: "sha256:test".to_string(),
            user: "alice".to_string(),
            name: "test".to_string(),
            created_at: Utc::now(),
            expires_at: Some(Utc::now() - Duration::hours(1)),
            scopes: vec!["*".to_string()],
        };
        assert!(record.is_expired());

        record.expires_at = Some(Utc::now() + Duration::hours(1));
        assert!(!record.is_expired());

        record.expires_at = None;
        assert!(!record.is_expired());
    }

    #[test]
    fn test_token_record_scope() {
        let record = TokenRecord {
            hash: "sha256:test".to_string(),
            user: "alice".to_string(),
            name: "test".to_string(),
            created_at: Utc::now(),
            expires_at: None,
            scopes: vec!["model:qwen3-*".to_string()],
        };

        assert!(record.has_scope("model:qwen3-small"));
        assert!(record.has_scope("model:qwen3-large"));
        assert!(!record.has_scope("model:llama3"));
        assert!(!record.has_scope("data:test"));
    }

    #[test]
    fn test_token_record_wildcard_scope() {
        let record = TokenRecord {
            hash: "sha256:test".to_string(),
            user: "alice".to_string(),
            name: "test".to_string(),
            created_at: Utc::now(),
            expires_at: None,
            scopes: vec!["*".to_string()],
        };

        assert!(record.has_scope("anything"));
        assert!(record.has_scope("model:test"));
    }

    #[tokio::test]
    async fn test_in_memory_manager() {
        let mut manager = TokenManager::in_memory();

        let token = manager.create_token(
            "alice",
            "test-token",
            Some(Duration::days(30)),
            vec![],
            false,
        ).await.unwrap();

        assert!(token.starts_with(TOKEN_PREFIX));
        assert_eq!(token.len(), TOKEN_PREFIX.len() + TOKEN_RANDOM_LENGTH);

        let record = manager.validate(&token);
        assert!(record.is_some());
        assert_eq!(record.unwrap().user, "alice");

        // Invalid token
        assert!(manager.validate("hypr_invalidtoken").is_none());
        assert!(manager.validate("invalid").is_none());
    }

    #[tokio::test]
    async fn test_admin_token() {
        let mut manager = TokenManager::in_memory();

        let token = manager.create_token(
            "admin",
            "admin-key",
            None,
            vec![],
            true,
        ).await.unwrap();

        assert!(token.starts_with(ADMIN_TOKEN_PREFIX));
    }
}
