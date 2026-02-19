//! JWT claims for authentication.
//!
//! Authorization is enforced server-side via Casbin policies, not JWT scopes.

use crate::common_capnp;
use crate::capnp::{ToCapnp, FromCapnp};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// JWT claims for authentication.
///
/// Note: Authorization is enforced via Casbin policies server-side.
/// Scopes are NOT embedded in JWTs.
#[derive(Clone, Serialize, Deserialize)]
pub struct Claims {
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

        Ok(Self {
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
            sub,
            exp,
            iat,
            aud: None,
            token: None,
        }
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
    ) -> std::result::Result<Option<Claims>, super::jwt::JwtError> {
        match &self.token {
            Some(token) => {
                let verified = super::jwt::decode(token, verifying_key)?;
                Ok(Some(verified))
            }
            None => Ok(None),
        }
    }

    /// Check if token is expired.
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.exp
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
        assert!(claims.aud.is_none());
        assert!(claims.token.is_none());
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
}
