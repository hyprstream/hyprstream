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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub exp: i64,
    pub iat: i64,
    /// RFC 8707 audience claim for resource indicator binding.
    /// When present, resource servers MUST validate this matches their canonical URI.
    /// Tokens without `aud` (legacy/env-var tokens) continue to work for backward compatibility.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,
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

        Ok(Self {
            sub: reader.get_sub()?.to_str()?.to_owned(),
            exp: reader.get_exp(),
            iat: reader.get_iat(),
            aud,
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
        }
    }

    /// Create new claims with an audience (RFC 8707 resource indicator).
    pub fn with_audience(mut self, audience: Option<String>) -> Self {
        self.aud = audience;
        self
    }

    /// Check if token is expired.
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.exp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claims_new() {
        let claims = Claims::new("alice".to_owned(), 1000, 2000);
        assert_eq!(claims.sub, "alice");
        assert_eq!(claims.iat, 1000);
        assert_eq!(claims.exp, 2000);
        assert!(claims.aud.is_none());
    }

    #[test]
    fn test_claims_with_audience() {
        let claims = Claims::new("alice".to_owned(), 1000, 2000)
            .with_audience(Some("https://api.example.com".to_owned()));
        assert_eq!(claims.aud, Some("https://api.example.com".to_owned()));
    }

    #[test]
    fn test_claims_subject() {
        use crate::envelope::Subject;
        let claims = Claims::new("alice".to_owned(), 1000, 2000);
        assert_eq!(Subject::from(&claims), Subject::new("alice"));
    }
}
