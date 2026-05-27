//! User identity mapping from external OIDC providers to local hyprstream subjects.

use anyhow::{anyhow, Result};

use crate::config::{UserMappingStrategy, ProvisioningMode};

/// Map an external OIDC identity to a local hyprstream subject.
///
/// The returned string becomes the `sub` claim in the hyprstream-issued JWT.
/// Casbin policies grant permissions to this mapped subject.
pub fn map_external_identity(
    provider_slug: &str,
    external_claims: &serde_json::Value,
    strategy: &UserMappingStrategy,
) -> Result<String> {
    match strategy {
        UserMappingStrategy::Namespaced => {
            let sub = external_claims["sub"]
                .as_str()
                .ok_or_else(|| anyhow!("External id_token missing 'sub' claim"))?;
            Ok(format!("{provider_slug}:{sub}"))
        }
        UserMappingStrategy::Email => {
            let email = external_claims["email"]
                .as_str()
                .ok_or_else(|| anyhow!("External id_token missing 'email' claim"))?;
            let verified = external_claims["email_verified"].as_bool().unwrap_or(false);
            if !verified {
                return Err(anyhow!(
                    "Email '{}' is not verified by the external provider (email_verified=false)",
                    email
                ));
            }
            Ok(email.to_owned())
        }
        UserMappingStrategy::Claim { name } => {
            let val = external_claims[name]
                .as_str()
                .ok_or_else(|| anyhow!("External id_token missing '{}' claim", name))?;
            Ok(val.to_owned())
        }
    }
}

/// Check if a user should be provisioned based on the provisioning mode.
///
/// Returns `Ok(true)` if the user should be auto-provisioned.
/// Returns `Ok(false)` if the user must already exist.
/// Returns `Err` if the user is denied by the allowlist.
pub fn should_provision(
    mode: &ProvisioningMode,
    mapped_subject: &str,
    allowed_domains: &[String],
) -> Result<bool> {
    match mode {
        ProvisioningMode::Deny => Ok(false),
        ProvisioningMode::Auto => Ok(true),
        ProvisioningMode::Allowlist => {
            // Check if the subject matches any allowed domain.
            // For email-based subjects: check the domain part.
            // For namespaced subjects: check the provider slug.
            let domain = mapped_subject
                .rsplit_once('@')
                .map(|(_, d)| d)
                .or_else(|| mapped_subject.split_once(':').map(|(p, _)| p));

            if let Some(domain) = domain {
                if allowed_domains.iter().any(|d| d == domain) {
                    return Ok(true);
                }
            }

            Err(anyhow!(
                "User '{}' is not in the allowed domains: {:?}",
                mapped_subject, allowed_domains
            ))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_namespaced_mapping() {
        let claims = serde_json::json!({"sub": "12345", "email": "alice@example.com"});
        let result = map_external_identity("keycloak", &claims, &UserMappingStrategy::Namespaced).unwrap();
        assert_eq!(result, "keycloak:12345");
    }

    #[test]
    fn test_email_mapping_verified() {
        let claims = serde_json::json!({"sub": "12345", "email": "alice@corp.com", "email_verified": true});
        let result = map_external_identity("google", &claims, &UserMappingStrategy::Email).unwrap();
        assert_eq!(result, "alice@corp.com");
    }

    #[test]
    fn test_email_mapping_unverified_fails() {
        let claims = serde_json::json!({"sub": "12345", "email": "alice@corp.com", "email_verified": false});
        let result = map_external_identity("google", &claims, &UserMappingStrategy::Email);
        assert!(result.is_err());
    }

    #[test]
    fn test_claim_mapping() {
        let claims = serde_json::json!({"sub": "12345", "preferred_username": "alice"});
        let strategy = UserMappingStrategy::Claim { name: "preferred_username".into() };
        let result = map_external_identity("dex", &claims, &strategy).unwrap();
        assert_eq!(result, "alice");
    }

    #[test]
    fn test_provisioning_deny() {
        assert!(!should_provision(&ProvisioningMode::Deny, "keycloak:123", &[]).unwrap());
    }

    #[test]
    fn test_provisioning_auto() {
        assert!(should_provision(&ProvisioningMode::Auto, "keycloak:123", &[]).unwrap());
    }

    #[test]
    fn test_provisioning_allowlist_email() {
        assert!(should_provision(
            &ProvisioningMode::Allowlist,
            "alice@corp.com",
            &["corp.com".into()],
        ).unwrap());
    }

    #[test]
    fn test_provisioning_allowlist_denied() {
        let result = should_provision(
            &ProvisioningMode::Allowlist,
            "alice@other.com",
            &["corp.com".into()],
        );
        assert!(result.is_err());
    }
}
