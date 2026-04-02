//! JWT claims for authentication.
//!
//! Two claim types:
//! - [`Claims`] — access token claims used throughout the RPC layer.
//!   Authorization is enforced server-side via Casbin policies, not JWT scopes.
//! - [`IdTokenClaims`] — OIDC ID Token claims (Section 2 of OpenID Connect
//!   Core 1.0). Only used by the OAuth token endpoint when `scope=openid`.

use crate::common_capnp;
use crate::capnp::{ToCapnp, FromCapnp};
use anyhow::Result;
use serde::{Deserialize, Serialize, Serializer, Deserializer};

/// Returns true if `iss` belongs to a local node.
///
/// Used for pre-decode key routing (before a `Claims` object exists) and by
/// [`Claims::is_local_to`] for post-decode subject derivation — both use
/// identical criteria so routing and subject resolution are always consistent.
///
/// Rules:
/// - If `local_issuers` is non-empty: `iss` must exactly match one entry.
/// - If `local_issuers` is empty (unconfigured node): only an empty `iss` is
///   accepted as local; any non-empty `iss` is treated as federated.
pub fn is_local_iss(iss: &str, local_issuers: &[&str]) -> bool {
    if local_issuers.is_empty() {
        iss.is_empty()
    } else {
        local_issuers.contains(&iss)
    }
}

/// JWT claims for authentication.
///
/// Note: Authorization is enforced via Casbin policies server-side.
/// Scopes are NOT embedded in JWTs.
#[derive(Clone, Serialize, Deserialize)]
pub struct Claims {
    /// Issuer URL — the hyprstream node that minted this token.
    /// Matches the OAuth issuer URL (e.g. "https://cloud-a.example.com").
    /// Required for federation: receiving nodes use this to fetch JWKS.
    /// Defaults to empty string for backward compat (treated as local-only).
    #[serde(default)]
    pub iss: String,
    pub sub: String,
    pub exp: i64,
    pub iat: i64,
    /// RFC 8707 audience claim for resource indicator binding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,
    /// Original JWT token for end-to-end verification.
    /// When present, downstream services MUST verify this token
    /// independently rather than trusting the envelope claims alone.
    ///
    /// SECURITY:
    /// - `#[serde(skip)]` prevents recursive JWT-in-JWT embedding and JSON exposure.
    /// - This field is ONLY populated via Cap'n Proto (`ToCapnp`/`FromCapnp`).
    /// - Custom Debug impl redacts this field as `[REDACTED]`.
    /// - Do NOT change to `skip_serializing_if` — that would leak the token in JSON.
    #[serde(skip)]
    pub token: Option<String>,
}

// Custom Debug impl — NEVER log the bearer token
impl std::fmt::Debug for Claims {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Claims")
            .field("iss", &self.iss)
            .field("sub", &self.sub)
            .field("exp", &self.exp)
            .field("iat", &self.iat)
            .field("aud", &self.aud)
            .field("token", &self.token.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

impl ToCapnp for Claims {
    type Builder<'a> = common_capnp::claims::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_sub(&self.sub);
        builder.set_exp(self.exp);
        builder.set_iat(self.iat);
        builder.set_admin(false); // Deprecated: always false
        if let Some(ref aud) = self.aud {
            builder.set_aud(aud);
        }
        if let Some(ref token) = self.token {
            builder.set_token(token);
        }
        if !self.iss.is_empty() {
            builder.set_iss(&self.iss);
        }
        // Write empty scopes list for wire compatibility
        builder.reborrow().init_scopes(0);
    }
}

impl FromCapnp for Claims {
    type Reader<'a> = common_capnp::claims::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        // Scopes on the wire are ignored - authorization is via Casbin
        let aud = reader.get_aud().ok()
            .and_then(|s| s.to_str().ok())
            .map(std::borrow::ToOwned::to_owned)
            .filter(|s| !s.is_empty());

        let token = reader.get_token().ok()
            .and_then(|s| s.to_str().ok())
            .map(std::borrow::ToOwned::to_owned)
            .filter(|s| !s.is_empty());

        let iss = reader.get_iss().ok()
            .and_then(|s| s.to_str().ok())
            .map(std::borrow::ToOwned::to_owned)
            .unwrap_or_default();

        Ok(Self {
            iss,
            sub: reader.get_sub()?.to_str()?.to_owned(),
            exp: reader.get_exp(),
            iat: reader.get_iat(),
            aud,
            token,
        })
    }
}

impl Claims {
    /// Create new claims.
    pub fn new(sub: String, iat: i64, exp: i64) -> Self {
        Self {
            iss: String::new(),
            sub,
            exp,
            iat,
            aud: None,
            token: None,
        }
    }

    /// Set the issuer URL (RFC 7519 `iss` claim).
    /// Should be the OAuth issuer URL of the hyprstream node that issued this token.
    pub fn with_issuer(mut self, issuer: String) -> Self {
        self.iss = issuer;
        self
    }

    /// Create new claims with an audience (RFC 8707 resource indicator).
    pub fn with_audience(mut self, audience: Option<String>) -> Self {
        self.aud = audience;
        self
    }

    /// Attach original JWT token for e2e verification by downstream services.
    pub fn with_token(mut self, token: String) -> Self {
        self.token = Some(token);
        self
    }

    /// Independently verify the embedded JWT token.
    ///
    /// Returns the verified claims if the token is present and valid.
    /// Returns None if no token is embedded.
    /// Returns Err if token is present but verification fails (MUST reject request).
    ///
    /// Note: `jwt::decode()` validates both signature and `exp` (via `is_expired()`).
    pub fn verify_token(
        &self,
        verifying_key: &ed25519_dalek::VerifyingKey,
        expected_aud: Option<&str>,
    ) -> std::result::Result<Option<Claims>, super::jwt::JwtError> {
        match &self.token {
            Some(token) => {
                let verified = super::jwt::decode(token, verifying_key, expected_aud)?;
                Ok(Some(verified))
            }
            None => Ok(None),
        }
    }

    /// Check if token is expired (with 5-second leeway for clock skew).
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.exp + 5
    }

    /// True if this token was issued by a local node.
    ///
    /// Delegates to [`is_local_iss`] using this token's `iss` field.
    pub fn is_local_to(&self, local_issuers: &[&str]) -> bool {
        is_local_iss(&self.iss, local_issuers)
    }

    /// Derive the Casbin authorization subject from these claims.
    ///
    /// Local tokens (issued by this node) produce bare subjects (`"alice"`)
    /// matching existing Casbin rules.  Federated tokens produce namespaced
    /// subjects (`"https://other.node:alice"`) to prevent cross-node spoofing.
    pub fn subject(&self, local_issuers: &[&str]) -> crate::envelope::Subject {
        if self.sub.is_empty() {
            return crate::envelope::Subject::anonymous();
        }
        if self.is_local_to(local_issuers) {
            crate::envelope::Subject::new(self.sub.clone())
        } else {
            crate::envelope::Subject::federated(&self.iss, &self.sub)
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_claims_new() {
        let claims = Claims::new("alice".to_owned(), 1000, 2000);
        assert_eq!(claims.sub, "alice");
        assert_eq!(claims.iat, 1000);
        assert_eq!(claims.exp, 2000);
        assert_eq!(claims.iss, "");
        assert!(claims.aud.is_none());
        assert!(claims.token.is_none());
    }

    #[test]
    fn test_claims_with_issuer() {
        let claims = Claims::new("alice".to_owned(), 1000, 2000)
            .with_issuer("https://a.example.com".to_owned());
        assert_eq!(claims.iss, "https://a.example.com");
    }

    #[test]
    fn test_claims_with_audience() {
        let claims = Claims::new("alice".to_owned(), 1000, 2000)
            .with_audience(Some("https://api.example.com".to_owned()));
        assert_eq!(claims.aud, Some("https://api.example.com".to_owned()));
    }

    #[test]
    fn test_claims_with_token() {
        let claims = Claims::new("alice".to_owned(), 1000, 2000)
            .with_token("eyJ.test.token".to_owned());
        assert_eq!(claims.token, Some("eyJ.test.token".to_owned()));
    }

    #[test]
    fn test_claims_token_not_in_json() {
        let claims = Claims::new("alice".to_owned(), 1000, 2000)
            .with_token("eyJ.secret.token".to_owned());
        let json = serde_json::to_string(&claims).unwrap();
        assert!(!json.contains("secret"));
        assert!(!json.contains("token"));
    }

    #[test]
    fn test_claims_debug_redacts_token() {
        let claims = Claims::new("alice".to_owned(), 1000, 2000)
            .with_token("eyJ.secret.token".to_owned());
        let debug = format!("{:?}", claims);
        assert!(debug.contains("[REDACTED]"));
        assert!(!debug.contains("secret"));
    }

    #[test]
    fn test_claims_subject() {
        use crate::envelope::Subject;
        let claims = Claims::new("alice".to_owned(), 1000, 2000);
        assert_eq!(Subject::from(&claims), Subject::new("alice"));
    }

    #[test]
    fn test_claims_is_local_to() {
        let local = Claims::new("alice".to_owned(), 0, 9999)
            .with_issuer("https://local.example.com".to_owned());
        let federated = Claims::new("bob".to_owned(), 0, 9999)
            .with_issuer("https://other.example.com".to_owned());
        let legacy = Claims::new("carol".to_owned(), 0, 9999); // empty iss

        assert!(local.is_local_to(&["https://local.example.com"]));
        assert!(!local.is_local_to(&["https://other.example.com"]));
        assert!(!federated.is_local_to(&["https://local.example.com"]));
        // Empty iss is local only when no local issuers configured
        assert!(legacy.is_local_to(&[]));
        assert!(!legacy.is_local_to(&["https://local.example.com"]));
    }

    #[test]
    fn test_claims_subject_method() {
        use crate::envelope::Subject;

        let local = Claims::new("alice".to_owned(), 0, 9999)
            .with_issuer("https://node.example.com".to_owned());
        let federated = Claims::new("bob".to_owned(), 0, 9999)
            .with_issuer("https://other.example.com".to_owned());
        let no_sub = Claims::new(String::new(), 0, 9999);

        assert_eq!(local.subject(&["https://node.example.com"]), Subject::new("alice"));
        assert_eq!(federated.subject(&["https://node.example.com"]), Subject::federated("https://other.example.com", "bob"));
        assert_eq!(no_sub.subject(&[]), Subject::anonymous());
    }

    #[test]
    fn test_claims_roundtrip_with_iss() {
        use super::super::jwt;
        use ed25519_dalek::SigningKey;

        let signing_key = SigningKey::from_bytes(&[42u8; 32]);
        let verifying_key = signing_key.verifying_key();

        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999)
            .with_issuer("https://a.example.com".to_owned());

        let token = jwt::encode(&claims, &signing_key);
        let decoded = jwt::decode(&token, &verifying_key, None).expect("decode failed");

        assert_eq!(decoded.iss, "https://a.example.com");
        assert_eq!(decoded.sub, "alice");
        assert_eq!(decoded.exp, 9_999_999_999);
    }

    #[test]
    fn test_claims_iss_absent_defaults_to_empty() {
        // Simulate an old token without the `iss` field.
        // `#[serde(default)]` must ensure deserialization succeeds with iss = "".
        let json = r#"{"sub":"bob","exp":9999999999,"iat":0}"#;
        let claims: Claims = serde_json::from_str(json)
            .expect("old token without iss should deserialize successfully");
        assert_eq!(claims.iss, "");
        assert_eq!(claims.sub, "bob");
    }

    #[test]
    fn test_claims_iss_not_in_json_when_empty() {
        // Empty iss is serialized as `"iss":""` (no skip_serializing_if),
        // but we do NOT skip it — that is intentional per the field design.
        // This test just confirms iss is present in JSON output.
        let claims = Claims::new("alice".to_owned(), 0, 9_999_999_999)
            .with_issuer("https://cloud.example.com".to_owned());
        let json = serde_json::to_string(&claims).unwrap();
        assert!(json.contains("\"iss\":\"https://cloud.example.com\""));
    }
}

// ─── OIDC ID Token Claims ──────────────────────────────────────────────────

/// A value that serializes as either a single string or an array of strings.
///
/// OIDC Core Section 2 says `aud` is a JSON string or array. Many RPs break
/// if they receive an unexpected format, so we serialize as a string when there
/// is exactly one value and as an array otherwise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OneOrMany {
    /// Single value — serialized as a JSON string.
    One(String),
    /// Multiple values — serialized as a JSON array.
    Many(Vec<String>),
}

impl OneOrMany {
    /// Create from a single value.
    pub fn one(value: impl Into<String>) -> Self {
        Self::One(value.into())
    }

    /// Create from a list. Collapses to `One` when the list has a single element.
    pub fn from_vec(mut values: Vec<String>) -> Self {
        if values.len() == 1 {
            Self::One(values.swap_remove(0))
        } else {
            Self::Many(values)
        }
    }

    /// Get the values as a slice-like iterator.
    pub fn as_slice(&self) -> &[String] {
        match self {
            Self::One(v) => std::slice::from_ref(v),
            Self::Many(v) => v,
        }
    }
}

impl Serialize for OneOrMany {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::One(v) => serializer.serialize_str(v),
            Self::Many(v) => v.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for OneOrMany {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::String(s) => Ok(Self::One(s)),
            serde_json::Value::Array(arr) => {
                let strings: std::result::Result<Vec<String>, _> = arr
                    .into_iter()
                    .map(|v| {
                        v.as_str()
                            .map(String::from)
                            .ok_or_else(|| serde::de::Error::custom("aud array must contain strings"))
                    })
                    .collect();
                Ok(Self::from_vec(strings?))
            }
            _ => Err(serde::de::Error::custom("aud must be a string or array")),
        }
    }
}

/// OIDC ID Token claims (OpenID Connect Core 1.0, Section 2).
///
/// Separate from [`Claims`] because:
/// - `aud` is the `client_id` (not the resource indicator)
/// - Includes user profile claims (name, email)
/// - Only issued by the OAuth token endpoint, not used in the RPC layer
/// - Different `typ` header (`"JWT"` vs `"at+jwt"` for access tokens)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdTokenClaims {
    // ── Required (OIDC Core Section 2) ──────────────────────────────────
    /// Issuer URL.
    pub iss: String,
    /// Subject identifier — a stable UUID, not a username.
    pub sub: String,
    /// Audience — the client_id. Supports single string or array.
    pub aud: OneOrMany,
    /// Expiration time (Unix timestamp).
    pub exp: i64,
    /// Issued at (Unix timestamp).
    pub iat: i64,

    // ── Conditional ─────────────────────────────────────────────────────
    /// OIDC nonce echoed from the authorization request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nonce: Option<String>,
    /// Time of user authentication (Unix timestamp).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auth_time: Option<i64>,
    /// Authorized party (REQUIRED when aud has multiple values).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub azp: Option<String>,

    // ── Profile claims (included based on requested scopes) ─────────────
    /// Full name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Email address.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<String>,
    /// Whether the email is verified.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email_verified: Option<bool>,
    /// Preferred display username.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preferred_username: Option<String>,
}

impl IdTokenClaims {
    /// Create minimal id_token claims (required fields only).
    pub fn new(iss: String, sub: String, aud: String, iat: i64, exp: i64) -> Self {
        Self {
            iss,
            sub,
            aud: OneOrMany::one(aud),
            exp,
            iat,
            nonce: None,
            auth_time: None,
            azp: None,
            name: None,
            email: None,
            email_verified: None,
            preferred_username: None,
        }
    }

    /// Set the OIDC nonce (echoed from the authorization request).
    pub fn with_nonce(mut self, nonce: Option<String>) -> Self {
        self.nonce = nonce;
        self
    }

    /// Set the authentication time.
    pub fn with_auth_time(mut self, auth_time: i64) -> Self {
        self.auth_time = Some(auth_time);
        self
    }

    /// Set profile claims from a user profile.
    pub fn with_profile(
        mut self,
        name: Option<String>,
        email: Option<String>,
        email_verified: Option<bool>,
        preferred_username: Option<String>,
    ) -> Self {
        self.name = name;
        self.email = email;
        self.email_verified = email_verified;
        self.preferred_username = preferred_username;
        self
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod id_token_tests {
    use super::*;

    #[test]
    fn test_id_token_claims_serialization() {
        let claims = IdTokenClaims::new(
            "https://example.com".into(),
            "f47ac10b-58cc-4372-a567-0e02b2c3d479".into(),
            "my-client".into(),
            1000,
            2000,
        );
        let json = serde_json::to_string(&claims).unwrap();
        // aud should be a single string, not array
        assert!(json.contains("\"aud\":\"my-client\""));
        assert!(json.contains("\"sub\":\"f47ac10b-"));
        // Optional fields should not appear
        assert!(!json.contains("nonce"));
        assert!(!json.contains("name"));
        assert!(!json.contains("email"));
    }

    #[test]
    fn test_id_token_claims_with_profile() {
        let claims = IdTokenClaims::new(
            "https://example.com".into(),
            "uuid-123".into(),
            "client-1".into(),
            1000,
            2000,
        )
        .with_nonce(Some("abc123".into()))
        .with_auth_time(999)
        .with_profile(
            Some("Alice".into()),
            Some("alice@example.com".into()),
            Some(true),
            Some("alice".into()),
        );
        let json = serde_json::to_string(&claims).unwrap();
        assert!(json.contains("\"nonce\":\"abc123\""));
        assert!(json.contains("\"auth_time\":999"));
        assert!(json.contains("\"name\":\"Alice\""));
        assert!(json.contains("\"email\":\"alice@example.com\""));
        assert!(json.contains("\"email_verified\":true"));
        assert!(json.contains("\"preferred_username\":\"alice\""));
    }

    #[test]
    fn test_one_or_many_single() {
        let aud = OneOrMany::one("client-1");
        let json = serde_json::to_string(&aud).unwrap();
        assert_eq!(json, "\"client-1\"");
    }

    #[test]
    fn test_one_or_many_multiple() {
        let aud = OneOrMany::from_vec(vec!["a".into(), "b".into()]);
        let json = serde_json::to_string(&aud).unwrap();
        assert_eq!(json, "[\"a\",\"b\"]");
    }

    #[test]
    fn test_one_or_many_deserialize_string() {
        let aud: OneOrMany = serde_json::from_str("\"client-1\"").unwrap();
        assert_eq!(aud, OneOrMany::One("client-1".into()));
    }

    #[test]
    fn test_one_or_many_deserialize_array() {
        let aud: OneOrMany = serde_json::from_str("[\"a\",\"b\"]").unwrap();
        assert_eq!(aud, OneOrMany::Many(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn test_one_or_many_collapse_single_element() {
        let aud = OneOrMany::from_vec(vec!["only-one".into()]);
        assert_eq!(aud, OneOrMany::One("only-one".into()));
    }
}
