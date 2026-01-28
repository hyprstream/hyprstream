//! JWT claims with structured scopes.

use super::Scope;
use crate::common_capnp;
use crate::capnp::{ToCapnp, FromCapnp};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// JWT claims with structured scopes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claims {
    pub sub: String,
    pub exp: i64,
    pub iat: i64,
    pub scopes: Vec<Scope>,
    pub admin: bool,
}

impl ToCapnp for Claims {
    type Builder<'a> = common_capnp::claims::Builder<'a>;

    fn write_to(&self, builder: &mut Self::Builder<'_>) {
        builder.set_sub(&self.sub);
        builder.set_exp(self.exp);
        builder.set_iat(self.iat);
        builder.set_admin(self.admin);

        let mut scopes_builder = builder.reborrow().init_scopes(self.scopes.len() as u32);
        for (i, scope) in self.scopes.iter().enumerate() {
            let mut scope_builder = scopes_builder.reborrow().get(i as u32);
            scope.write_to(&mut scope_builder);
        }
    }
}

impl FromCapnp for Claims {
    type Reader<'a> = common_capnp::claims::Reader<'a>;

    fn read_from(reader: Self::Reader<'_>) -> Result<Self> {
        let scopes_reader = reader.get_scopes()?;
        let mut scopes = Vec::with_capacity(scopes_reader.len() as usize);
        for scope_reader in scopes_reader.iter() {
            scopes.push(Scope::read_from(scope_reader)?);
        }

        Ok(Self {
            sub: reader.get_sub()?.to_str()?.to_owned(),
            exp: reader.get_exp(),
            iat: reader.get_iat(),
            scopes,
            admin: reader.get_admin(),
        })
    }
}

impl Claims {
    /// Create new claims.
    pub fn new(sub: String, iat: i64, exp: i64, scopes: Vec<Scope>, admin: bool) -> Self {
        Self {
            sub,
            exp,
            iat,
            scopes,
            admin,
        }
    }

    /// Check if claims grant required scope.
    ///
    /// FAIL-SECURE: Empty scopes deny all (not allow all)
    pub fn has_scope(&self, required: &Scope) -> bool {
        // FAIL-SECURE: Empty scopes deny all
        if self.scopes.is_empty() && !self.admin {
            return false;
        }

        // Admin override
        if self.admin {
            return true;
        }

        // Check if any granted scope permits required scope
        self.scopes.iter().any(|s| s.grants(required))
    }

    /// Check if token is expired.
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.exp
    }

    /// Get Casbin subject string.
    pub fn casbin_subject(&self) -> String {
        format!("user:{}", self.sub)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_claims_has_scope_exact() {
        let claims = Claims::new(
            "alice".to_owned(),
            1000,
            2000,
            vec![Scope::parse("infer:model:qwen-7b").unwrap()],
            false,
        );
        let required = Scope::parse("infer:model:qwen-7b").unwrap();
        assert!(claims.has_scope(&required));
    }

    #[test]
    fn test_claims_has_scope_wildcard() {
        let claims = Claims::new(
            "alice".to_owned(),
            1000,
            2000,
            vec![Scope::parse("infer:model:*").unwrap()],
            false,
        );
        let required = Scope::parse("infer:model:qwen-7b").unwrap();
        assert!(claims.has_scope(&required));
    }

    #[test]
    fn test_claims_fail_secure_empty_scopes() {
        let claims = Claims::new("alice".to_owned(), 1000, 2000, vec![], false);
        let required = Scope::parse("infer:model:qwen-7b").unwrap();
        assert!(!claims.has_scope(&required));
    }

    #[test]
    fn test_claims_admin_override() {
        let claims = Claims::new("admin".to_owned(), 1000, 2000, vec![], true);
        let required = Scope::parse("infer:model:qwen-7b").unwrap();
        assert!(claims.has_scope(&required));
    }
}
