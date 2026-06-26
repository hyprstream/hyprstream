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
    local_issuer_url: &str,
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
        UserMappingStrategy::DidWeb => {
            // Phase 0.5 / Phase 1a — did:web subject format per architecture
            // doc Subject Identity Format. Format:
            //   did:web:{authority}:users:{external_sub}
            // where {authority} = the local OAuth issuer URL's host[:port].
            let sub = external_claims["sub"]
                .as_str()
                .ok_or_else(|| anyhow!("External id_token missing 'sub' claim"))?;
            let authority = extract_authority(local_issuer_url)?;
            // External sub may contain characters; for now we require it to
            // be a stable opaque identifier (most OIDC providers use UUIDs
            // or stable usernames). Operators using providers with weird sub
            // formats should use a different strategy (Email / Claim).
            if sub.contains(['#', '?', '/']) {
                return Err(anyhow!(
                    "External 'sub' contains characters incompatible with did:web path: {}",
                    sub
                ));
            }
            Ok(format!("did:web:{authority}:users:{sub}"))
        }
    }
}

/// Extract the authority component (host[:port]) from a URL, preserving
/// the port if present. Used for did:web construction per the did:web spec.
fn extract_authority(url: &str) -> Result<String> {
    let after_scheme = url
        .split_once("://")
        .map(|(_, rest)| rest)
        .ok_or_else(|| anyhow!("issuer URL missing scheme: {}", url))?;
    let authority = after_scheme.split('/').next().unwrap_or(after_scheme);
    if authority.is_empty() {
        return Err(anyhow!("issuer URL has empty authority: {}", url));
    }
    Ok(authority.to_owned())
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
        let result = map_external_identity("keycloak", &claims, &UserMappingStrategy::Namespaced, "").unwrap();
        assert_eq!(result, "keycloak:12345");
    }

    #[test]
    fn test_email_mapping_verified() {
        let claims = serde_json::json!({"sub": "12345", "email": "alice@corp.com", "email_verified": true});
        let result = map_external_identity("google", &claims, &UserMappingStrategy::Email, "").unwrap();
        assert_eq!(result, "alice@corp.com");
    }

    #[test]
    fn test_email_mapping_unverified_fails() {
        let claims = serde_json::json!({"sub": "12345", "email": "alice@corp.com", "email_verified": false});
        let result = map_external_identity("google", &claims, &UserMappingStrategy::Email, "");
        assert!(result.is_err());
    }

    #[test]
    fn test_claim_mapping() {
        let claims = serde_json::json!({"sub": "12345", "preferred_username": "alice"});
        let strategy = UserMappingStrategy::Claim { name: "preferred_username".into() };
        let result = map_external_identity("dex", &claims, &strategy, "").unwrap();
        assert_eq!(result, "alice");
    }

    #[test]
    fn test_didweb_mapping_basic() {
        let claims = serde_json::json!({"sub": "alice"});
        let result = map_external_identity(
            "keycloak",
            &claims,
            &UserMappingStrategy::DidWeb,
            "https://hyprstream.example.com",
        ).unwrap();
        assert_eq!(result, "did:web:hyprstream.example.com:users:alice");
    }

    #[test]
    fn test_didweb_mapping_preserves_port() {
        let claims = serde_json::json!({"sub": "12345"});
        let result = map_external_identity(
            "keycloak",
            &claims,
            &UserMappingStrategy::DidWeb,
            "http://127.0.0.1:6791",
        ).unwrap();
        assert_eq!(result, "did:web:127.0.0.1:6791:users:12345");
    }

    #[test]
    fn test_didweb_mapping_rejects_path_chars_in_sub() {
        let claims = serde_json::json!({"sub": "alice/bob"});
        let result = map_external_identity(
            "keycloak",
            &claims,
            &UserMappingStrategy::DidWeb,
            "https://hyprstream.example.com",
        );
        assert!(result.is_err(), "sub with '/' must reject");
    }

    #[test]
    fn test_didweb_mapping_requires_scheme() {
        let claims = serde_json::json!({"sub": "alice"});
        let result = map_external_identity(
            "keycloak",
            &claims,
            &UserMappingStrategy::DidWeb,
            "hyprstream.example.com",
        );
        assert!(result.is_err(), "missing scheme must reject");
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
