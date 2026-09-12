//! Native authority adapter for the protected ATProto `getSession` route.
//!
//! The resolver is deliberately narrower than a generic DID resolver. It only
//! answers for a host-form DID whose account label is covered by the configured
//! account zone, and it requires both durable account-record authority and the
//! credential profile's exact DID binding. There is no directory enumeration or
//! request-supplied tenant input on this path.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;

use crate::account::AccountZone;
use crate::auth::UserStore;
use crate::services::oauth::xrpc::{AtprotoSessionInfo, AtprotoSessionResolver};
use hyprstream_pds_service::{AccountRecordStore, OAUTH_ACCOUNT_RESOLVER_SUBJECT};
use hyprstream_rpc::Subject;

/// Production native authority for the standard protected `getSession` seam.
///
/// `user_store` is the admitted credential/profile backend. `account_store`
/// proves that the same DID has a durable, authority-owned hosted account
/// record. The zone is deployment configuration and is never inferred from a
/// request or profile field.
pub(crate) struct NativeAtprotoSessionResolver {
    user_store: Arc<dyn UserStore>,
    account_store: Arc<AccountRecordStore>,
    zone: AccountZone,
}

impl NativeAtprotoSessionResolver {
    pub(crate) fn new(
        user_store: Arc<dyn UserStore>,
        account_store: Arc<AccountRecordStore>,
        zone: AccountZone,
    ) -> Self {
        Self {
            user_store,
            account_store,
            zone,
        }
    }

    /// Extract the one account label represented by a configured-zone did:web.
    ///
    /// ATProto host-form DIDs use the DNS host directly for a single-label
    /// account zone. Reject ports, paths, percent escapes, extra labels and
    /// case changes instead of normalizing an ambiguous identity.
    fn label_for_did(&self, did: &str) -> Result<Option<String>> {
        hosted_label_for_did(&self.zone, did)
    }
}

fn hosted_label_for_did(zone: &AccountZone, did: &str) -> Result<Option<String>> {
    let Some(host) = did.strip_prefix("did:web:") else {
        return Ok(None);
    };
    if host.is_empty()
        || host != host.to_ascii_lowercase()
        || host.contains([':', '/', '%'])
        || !host.is_ascii()
    {
        return Err(anyhow!("invalid host-form hosted account DID"));
    }
    let suffix = format!(".{}", zone.apex());
    let Some(label) = host.strip_suffix(&suffix) else {
        return Ok(None);
    };
    if label.is_empty() || label.contains('.') || zone.host_for_label(label)? != host {
        return Err(anyhow!(
            "hosted account DID is outside the configured account zone"
        ));
    }
    Ok(Some(label.to_owned()))
}

#[async_trait]
impl AtprotoSessionResolver for NativeAtprotoSessionResolver {
    async fn resolve_session(&self, did: &str) -> Result<Option<AtprotoSessionInfo>> {
        let Some(label) = self.label_for_did(did)? else {
            // did:plc and foreign did:web identities are federated identities
            // for this PDS and must not be answered from local profile data.
            return Ok(None);
        };

        let Some(profile) = self.user_store.get_profile(&label).await? else {
            return Ok(None);
        };
        if profile.atproto_did.as_deref() != Some(did) {
            return Ok(None);
        }

        let authority = Subject::new(OAUTH_ACCOUNT_RESOLVER_SUBJECT);
        if self
            .account_store
            .resolve_tenant_for_hosted_did(&authority, did)
            .await?
            .is_none()
        {
            return Ok(None);
        }

        let active = profile.active.unwrap_or(true);
        Ok(Some(AtprotoSessionInfo {
            handle: self.zone.host_for_label(&label)?,
            did_doc: None,
            email: profile.email,
            email_confirmed: profile.email_verified,
            email_auth_factor: None,
            active,
            status: (!active).then(|| "deactivated".to_owned()),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_exact_single_label_zone_hosts_are_accepted() -> Result<()> {
        let zone = AccountZone::new("accounts.example.test")?;
        assert_eq!(
            hosted_label_for_did(&zone, "did:web:alice.accounts.example.test")?,
            Some("alice".to_owned())
        );
        assert!(hosted_label_for_did(&zone, "did:web:a.b.accounts.example.test").is_err());
        assert!(hosted_label_for_did(&zone, "did:web:alice.accounts.example.test:8443").is_err());
        assert!(hosted_label_for_did(&zone, "did:web:Alice.accounts.example.test").is_err());
        assert_eq!(
            hosted_label_for_did(&zone, "did:web:other.example.test")?,
            None
        );
        assert_eq!(hosted_label_for_did(&zone, "did:plc:abc")?, None);
        Ok(())
    }
}
