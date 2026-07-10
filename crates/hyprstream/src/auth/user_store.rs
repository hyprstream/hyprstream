//! User credential store abstraction.
//!
//! `UserStore` is the trait. `RocksDbUserStore` is the production implementation
//! with atomic updates and multi-pubkey support.

use anyhow::Result;
use async_trait::async_trait;
use ed25519_dalek::VerifyingKey;
use serde::{Deserialize, Serialize};

/// User profile data (OIDC standard claims + SCIM-informed fields).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserProfile {
    /// Stable subject identifier (UUID). Generated at registration, never changes.
    #[serde(default)]
    pub sub: Option<String>,
    /// Display name (SCIM: displayName / OIDC: name).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Email address (SCIM: emails[0].value / OIDC: email).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    /// Whether the email is verified (OIDC: email_verified).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub email_verified: Option<bool>,
    /// Whether the user is active (SCIM: active). None defaults to true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active: Option<bool>,
    /// External identity ID from upstream IdP (SCIM: externalId).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub external_id: Option<String>,
}

/// The public-key algorithm of a stored [`PubkeyEntry`].
///
/// Ed25519 is the only implemented variant today; the tag exists from day one
/// (#439) so that adding ML-DSA-65 / hybrid user keys later (mirroring the
/// #280 Multikey path) is *additive* — every record already carries its
/// algorithm, so widening `UserStore::add_pubkey` past Ed25519 will not be a
/// store-schema migration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KeyAlgorithm {
    Ed25519,
}

impl Default for KeyAlgorithm {
    /// Ed25519 is the classical floor for local human identity keys (the same
    /// call already made for the iroh NodeId); PQ user keys are deferred.
    fn default() -> Self {
        Self::Ed25519
    }
}

impl KeyAlgorithm {
    /// Multicodec-style lowercase algorithm name written into stored records.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Ed25519 => "ed25519",
        }
    }

    /// Parse an algorithm tag read back from a stored record.
    ///
    /// Unknown tags error rather than silently falling back, so a future PQ
    /// tag written by a newer build is never misread as Ed25519 by this one.
    pub fn parse(s: &str) -> anyhow::Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ed25519" => Ok(Self::Ed25519),
            other => anyhow::bail!(
                "Unknown key algorithm tag '{other}'; only 'ed25519' is implemented in this build"
            ),
        }
    }
}

/// A pubkey entry associated with a user (like GitHub SSH keys).
#[derive(Debug, Clone)]
pub struct PubkeyEntry {
    /// Base64url SHA-256 fingerprint of the pubkey bytes.
    pub fingerprint: String,
    /// The actual Ed25519 public key.
    pub pubkey: VerifyingKey,
    /// User-provided label (e.g., "laptop", "work").
    pub label: Option<String>,
    /// Unix timestamp when the key was added.
    pub created_at: i64,
    /// Unix timestamp when the key was last used for auth, or None if never.
    pub last_used_at: Option<i64>,
    /// Algorithm tag (#439). Defaults to [`KeyAlgorithm::Ed25519`] for records
    /// written before the tag existed.
    pub algorithm: KeyAlgorithm,
}

/// Filter parameters for user search (SCIM-aligned).
#[derive(Debug, Clone, Default)]
pub struct UserFilter {
    /// SCIM filter expression: `userName eq "alice"`, `active pr`, etc.
    /// Minimum viable: `eq` and `pr` on userName, id, externalId, active.
    pub filter: Option<String>,
    /// If true, only return active users.
    pub active_only: Option<bool>,
    /// SCIM: max results per page (default 100).
    pub count: Option<usize>,
    /// SCIM: 1-indexed start position (converted to 0-indexed internally).
    pub start_index: Option<usize>,
    /// SCIM: attribute name to sort by (e.g., "userName", "id").
    pub sort_by: Option<String>,
    /// SCIM: sort order ("ascending" or "descending"). Default: ascending.
    pub sort_order: Option<String>,
}

/// Abstraction over user credential stores.
///
/// Supports profile CRUD and multi-pubkey management (like GitHub SSH keys).
#[async_trait]
pub trait UserStore: Send + Sync {
    // ─── Profile CRUD ────────────────────────────────────────────────────────

    /// Get a user's profile (OIDC claims).
    async fn get_profile(&self, username: &str) -> Result<Option<UserProfile>>;

    /// Register a new user. Returns the generated subject UUID.
    async fn register(&self, username: &str) -> Result<String>;

    /// Update a user's profile fields (merge semantics).
    async fn set_profile(&self, username: &str, profile: UserProfile) -> Result<()>;

    /// Remove a user and all their pubkeys.
    async fn remove(&self, username: &str) -> Result<bool>;

    /// List all registered usernames.
    async fn list_users(&self) -> Vec<String>;

    /// Search users with SCIM-aligned filtering, sorting, and pagination.
    async fn search(&self, filter: &UserFilter) -> Result<Vec<(String, UserProfile)>>;

    /// Set a user's active status.
    async fn set_active(&self, username: &str, active: bool) -> Result<()>;

    // ─── Pubkey Management ───────────────────────────────────────────────────

    /// List all pubkeys for a user.
    async fn list_pubkeys(&self, username: &str) -> Result<Vec<PubkeyEntry>>;

    /// Add a pubkey to a user. Returns the fingerprint.
    async fn add_pubkey(
        &self,
        username: &str,
        pubkey: VerifyingKey,
        label: Option<String>,
    ) -> Result<String>;

    /// Remove a pubkey by fingerprint.
    async fn remove_pubkey(&self, username: &str, fingerprint: &str) -> Result<bool>;

    /// Reverse lookup: find username by pubkey fingerprint (for auth).
    async fn get_pubkey_user(&self, fingerprint: &str) -> Result<Option<String>>;

    /// Update last_used_at timestamp for a pubkey.
    async fn touch_pubkey(&self, username: &str, fingerprint: &str) -> Result<()>;
}

/// Compute the SSH fingerprint of a pubkey (matches `ssh-keygen -l -E sha256`).
///
/// Hashes the SSH wire-format blob (u32be(11) "ssh-ed25519" u32be(32) <key>)
/// with SHA-256 and encodes as standard base64 without padding, prefixed "SHA256:".
pub fn pubkey_fingerprint(pubkey: &VerifyingKey) -> String {
    use base64::engine::general_purpose::STANDARD_NO_PAD;
    use base64::Engine;
    use sha2::{Digest, Sha256};
    let wire = ssh_wire_encode(pubkey);
    format!("SHA256:{}", STANDARD_NO_PAD.encode(Sha256::digest(wire)))
}

/// Encode an Ed25519 verifying key in SSH wire format (51 bytes).
///
/// Format: u32be(11) "ssh-ed25519" u32be(32) <32-byte key>
fn ssh_wire_encode(pubkey: &VerifyingKey) -> [u8; 51] {
    let mut wire = [0u8; 51];
    wire[0..4].copy_from_slice(&11u32.to_be_bytes());
    wire[4..15].copy_from_slice(b"ssh-ed25519");
    wire[15..19].copy_from_slice(&32u32.to_be_bytes());
    wire[19..51].copy_from_slice(pubkey.as_bytes());
    wire
}

/// Decode a base64-encoded Ed25519 public key.
///
/// Accepts two formats:
/// - SSH wire format (base64 of 51-byte blob): validates the "ssh-ed25519" type tag.
/// - Raw Ed25519 (base64 of 32-byte key).
pub fn decode_pubkey_base64(s: &str) -> anyhow::Result<VerifyingKey> {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;

    let raw = STANDARD
        .decode(s.trim())
        .map_err(|e| anyhow::anyhow!("Invalid base64: {e}"))?;

    let key_bytes: [u8; 32] = if raw.len() == 51 {
        let type_len = u32::from_be_bytes(
            raw[0..4]
                .try_into()
                .map_err(|_| anyhow::anyhow!("Malformed SSH wire prefix"))?,
        ) as usize;
        if type_len != 11 || &raw[4..15] != b"ssh-ed25519" {
            anyhow::bail!(
                "Unsupported SSH key type: only Ed25519 is accepted. \
                 Use: ssh-keygen -t ed25519"
            );
        }
        raw[19..51]
            .try_into()
            .map_err(|_| anyhow::anyhow!("Malformed SSH wire payload"))?
    } else if raw.len() == 32 {
        raw.try_into()
            .map_err(|_| anyhow::anyhow!("Expected 32-byte raw Ed25519 key"))?
    } else {
        anyhow::bail!(
            "Expected 32-byte raw or 51-byte SSH wire Ed25519 key, got {} bytes",
            raw.len()
        );
    };

    VerifyingKey::from_bytes(&key_bytes).map_err(|e| anyhow::anyhow!("Invalid Ed25519 key: {e}"))
}

/// A parsed SCIM filter expression (RFC 7644 §3.4.2.2).
///
/// Only the subset actually sent by IdP SCIM clients is parsed.
/// Unsupported expressions are captured as `Unrecognised` — RocksDB falls
/// through to in-memory evaluation; Valkey returns `invalidFilter` (HTTP 400).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScimFilter {
    UserNameEq(String),
    IdEq(String),
    ExternalIdEq(String),
    ActiveEq(bool),
    Presence(String),
    Unrecognised(String),
}

impl ScimFilter {
    pub fn parse(expr: &str) -> Self {
        let expr = expr.trim();
        if let Some(attr) = expr.strip_suffix(" pr") {
            return ScimFilter::Presence(attr.trim().to_owned());
        }
        if let Some(rest) = expr.strip_prefix("userName eq ") {
            return ScimFilter::UserNameEq(unquote(rest).to_owned());
        }
        if let Some(rest) = expr.strip_prefix("id eq ").or_else(|| expr.strip_prefix("sub eq ")) {
            return ScimFilter::IdEq(unquote(rest).to_owned());
        }
        if let Some(rest) = expr.strip_prefix("externalId eq ") {
            return ScimFilter::ExternalIdEq(unquote(rest).to_owned());
        }
        if let Some(rest) = expr.strip_prefix("active eq ") {
            return ScimFilter::ActiveEq(rest.trim() == "true");
        }
        ScimFilter::Unrecognised(expr.to_owned())
    }
}

/// Evaluate a simple SCIM filter expression against a user entry.
///
/// Supports:
/// - `userName eq "value"` — exact match on username
/// - `id eq "value"` or `sub eq "value"` — exact match on subject UUID
/// - `externalId eq "value"` — exact match on external ID
/// - `active eq true/false` — match on active status
/// - `userName pr` — presence (non-empty/non-None)
/// - `active pr` — presence check
pub(crate) fn matches_filter(
    expr: &str,
    username: &str,
    sub: &Option<String>,
    external_id: &Option<String>,
    active: Option<bool>,
) -> bool {
    let expr = expr.trim();

    // Presence operator: `attribute pr`
    if let Some(attr) = expr.strip_suffix(" pr") {
        let attr = attr.trim();
        return match attr {
            "userName" => !username.is_empty(),
            "id" | "sub" => sub.is_some(),
            "externalId" => external_id.is_some(),
            "active" => active.is_some(),
            "displayName" | "name" | "email" => true, // always "present" even if None conceptually
            _ => false,
        };
    }

    // Equality operator: `attribute eq "value"` or `attribute eq true/false`
    if let Some(rest) = expr.strip_prefix("userName eq ") {
        return username == unquote(rest);
    }
    if let Some(rest) = expr.strip_prefix("id eq ") {
        return sub.as_deref() == Some(unquote(rest));
    }
    if let Some(rest) = expr.strip_prefix("sub eq ") {
        return sub.as_deref() == Some(unquote(rest));
    }
    if let Some(rest) = expr.strip_prefix("externalId eq ") {
        return external_id.as_deref() == Some(unquote(rest));
    }
    if let Some(rest) = expr.strip_prefix("active eq ") {
        let val = rest.trim();
        let expected = val == "true";
        return active.unwrap_or(true) == expected;
    }

    tracing::warn!(%expr, "Unsupported SCIM filter expression");
    false
}

/// Strip surrounding double-quotes from a SCIM filter value.
fn unquote(s: &str) -> &str {
    s.trim().trim_matches('"')
}

// ────────────────────────────────────────────────────────────────────────────
// Anonymous device identity
// ────────────────────────────────────────────────────────────────────────────

/// An anonymous device identity record.
///
/// Created by the device enrollment flow (`POST /api/device/enroll`). Exists
/// independently before any user logs in. `user_sub` is populated when the
/// device cookie accompanies an OAuth token request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceRecord {
    /// Raw 32-byte Ed25519 public key.
    pub pubkey: [u8; 32],
    /// Base58 encoding of `pubkey` — used as the RocksDB key suffix and JWT `sub`.
    pub fingerprint: String,
    /// Optional human-readable label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Unix timestamp of first enrollment.
    pub enrolled_at: i64,
    /// Unix timestamp of most recent activity (`touch_device`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_seen_at: Option<i64>,
    /// Subject of the linked user, if OAuth login has been observed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_sub: Option<String>,
}

/// Storage operations for anonymous device records.
#[async_trait]
pub trait DeviceStore: Send + Sync {
    /// Upsert a device record. On conflict (same fingerprint), updates `enrolled_at`
    /// but preserves any existing `user_sub` and `label`.
    async fn enroll_device(&self, record: DeviceRecord) -> Result<()>;

    /// Associate a user subject with an existing device record.
    async fn link_device_user(&self, fingerprint: &str, user_sub: &str) -> Result<()>;

    /// Retrieve a device record by fingerprint.
    async fn get_device(&self, fingerprint: &str) -> Result<Option<DeviceRecord>>;

    /// Update `last_seen_at` to now.
    async fn touch_device(&self, fingerprint: &str) -> Result<()>;

    /// Delete a device record. Returns `true` if the record existed.
    async fn revoke_device(&self, fingerprint: &str) -> Result<bool>;
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_matches_filter_username_eq() {
        assert!(matches_filter(r#"userName eq "alice""#, "alice", &None, &None, None));
        assert!(!matches_filter(r#"userName eq "alice""#, "bob", &None, &None, None));
    }

    #[test]
    fn test_matches_filter_active_eq() {
        assert!(matches_filter("active eq true", "u", &None, &None, Some(true)));
        assert!(!matches_filter("active eq true", "u", &None, &None, Some(false)));
    }

    #[test]
    fn test_matches_filter_presence() {
        let sub = Some("uuid-1".to_owned());
        assert!(matches_filter("userName pr", "alice", &None, &None, None));
        assert!(matches_filter("id pr", "alice", &sub, &None, None));
        assert!(!matches_filter("active pr", "alice", &None, &None, None));
    }

    #[test]
    fn test_key_algorithm_parse_and_roundtrip() {
        assert_eq!(KeyAlgorithm::Ed25519.as_str(), "ed25519");
        assert_eq!(KeyAlgorithm::default(), KeyAlgorithm::Ed25519);
        // Accepts case/whitespace variants.
        assert_eq!(KeyAlgorithm::parse("ed25519").unwrap(), KeyAlgorithm::Ed25519);
        assert_eq!(KeyAlgorithm::parse("  Ed25519 ").unwrap(), KeyAlgorithm::Ed25519);
        // Unknown tags error (never silently fall back to Ed25519).
        assert!(KeyAlgorithm::parse("ml-dsa-65").is_err());
        assert!(KeyAlgorithm::parse("").is_err());
    }
}
