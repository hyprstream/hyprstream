//! Internal Policy lookup. No credential verification, grants or public route.
use std::collections::BTreeSet;
use std::time::Duration;

use hyprstream_rpc::auth::det_cbor::{det_cbor, DetCborValue};
use hyprstream_rpc::auth::signer_suite::signer_suite_thumbprint;
use sha2::{Digest, Sha256};
use tokio::sync::Semaphore;
use tokio_postgres::Client;

use crate::{bounded, Error, Session, Store, PROFILE, SUITE};

/// Complete public input from a trusted configuration loader, not a browser.
/// Required IDs must enumerate ALL configured static/foreign protocol sources,
/// including unusable static records. Read/parse failures must never become [].
pub struct InventorySource {
    pub id: String,
    pub keys: Vec<Vec<u8>>,
}

#[derive(Clone)]
pub struct CollisionInventory {
    id: [u8; 32],
    keys: BTreeSet<Vec<u8>>,
}

impl CollisionInventory {
    pub fn from_sources(required: &[String], sources: Vec<InventorySource>) -> Result<Self, Error> {
        let ids: BTreeSet<_> = required.iter().cloned().collect();
        if ids.is_empty()
            || ids.len() != required.len()
            || ids.len() > 256
            || ids.iter().any(|id| !bounded(id, 256))
            || sources.len() != ids.len()
        {
            return Err(Error::Invalid);
        }
        let mut seen = BTreeSet::new();
        let mut keys = BTreeSet::new();
        for source in sources {
            if !ids.contains(&source.id) || !seen.insert(source.id) || source.keys.len() > 4096 {
                return Err(Error::Invalid);
            }
            for key in source.keys {
                if !matches!(key.len(), 32 | 1952) {
                    return Err(Error::Invalid);
                }
                keys.insert(key);
                if keys.len() > 4096 {
                    return Err(Error::Invalid);
                }
            }
        }
        let value = DetCborValue::Array(vec![
            DetCborValue::Text("hyprstream.session-collision-inventory.v1"),
            DetCborValue::Array(ids.iter().map(|id| DetCborValue::Text(id)).collect()),
            DetCborValue::Array(keys.iter().map(|key| DetCborValue::Bytes(key)).collect()),
        ]);
        Ok(Self {
            id: Sha256::digest(det_cbor(&value)).into(),
            keys,
        })
    }

    pub fn id(&self) -> [u8; 32] {
        self.id
    }

    pub fn permits(&self, ed: &[u8; 32], pq: &[u8]) -> bool {
        pq.len() == 1952 && !self.keys.contains(ed.as_slice()) && !self.keys.contains(pq)
    }
}

/// Returned only for an active row matching current generation AND inventory.
/// profile/suite are constants checked by SQL; grant_revision is provenance only.
#[derive(Clone, PartialEq, Eq)]
pub struct ActiveSessionPrimary {
    pub session: Session,
    pub created_at: i64,
}

/// Derived from an already verified host JWT by the future H3b consumer.
/// issuer is an equality check, never a caller-selected database locator.
#[derive(Clone)]
pub struct ExpectedPrimary {
    pub issuer: String,
    pub profile: String,
    pub sid: String,
    pub subject: String,
    pub tenant: String,
    pub client: String,
    pub audience: String,
    pub scopes: Vec<String>,
    pub ed_public: [u8; 32],
    pub suite_thumbprint: [u8; 32],
    pub generation: [u8; 32],
    pub expires_at: i64,
}

/// Trusted configuration only. No implicit wildcard/service-wide permission.
pub struct LookupCapability {
    pub service: String,
    pub service_key: [u8; 32],
    pub tenant: String,
    pub resource: String,
}

/// Scoped to one configured Policy authority. No default capability or cache.
pub struct PrimaryLookup {
    host: String,
    serving_generation: [u8; 32],
    inventory: CollisionInventory,
    capabilities: Vec<LookupCapability>,
    in_flight: Semaphore,
}

impl PrimaryLookup {
    pub fn new(
        host: String,
        serving_generation: [u8; 32],
        inventory: CollisionInventory,
        capabilities: Vec<LookupCapability>,
    ) -> Result<Self, Error> {
        if !bounded(&host, 2048)
            || capabilities.len() > 256
            || capabilities.iter().any(|c| {
                !c.service.starts_with("service:")
                    || !bounded(&c.service, 256)
                    || c.service_key == [0; 32]
                    || !bounded(&c.tenant, 256)
                    || !bounded(&c.resource, 2048)
            })
        {
            return Err(Error::Invalid);
        }
        Ok(Self {
            host,
            serving_generation,
            inventory,
            capabilities,
            in_flight: Semaphore::new(16),
        })
    }

    /// Policy must authenticate the static service/key before calling. This
    /// additional explicit capability gate is not a substitute for signatures.
    pub async fn resolve(
        &self,
        client: &mut Client,
        service: &str,
        service_key: &[u8; 32],
        expected: &ExpectedPrimary,
    ) -> Result<ActiveSessionPrimary, Error> {
        if expected.issuer != self.host
            || expected.profile != PROFILE
            || expected.generation != self.serving_generation
            || !bounded(&expected.sid, 128)
            || !bounded(&expected.subject, 256)
            || !bounded(&expected.client, 256)
            || !bounded(&expected.tenant, 256)
            || !bounded(&expected.audience, 2048)
            || expected.scopes.is_empty()
            || expected.scopes.len() > 64
            || expected
                .scopes
                .iter()
                .any(|s| !bounded(s, 256) || !s.bytes().all(|b| (0x21..=0x7e).contains(&b)))
            || expected.scopes.windows(2).any(|s| s[0] >= s[1])
            || !self.capabilities.iter().any(|c| {
                c.service == service
                    && &c.service_key == service_key
                    && c.tenant == expected.tenant
                    && c.resource == expected.audience
            })
        {
            return Err(Error::Inactive);
        }
        let _permit = self
            .in_flight
            .try_acquire()
            .map_err(|_| Error::Unavailable)?;
        tokio::time::timeout(Duration::from_secs(2), async {
            // The v3 lookup owns the bounded read transaction and binds the
            // unique committed source replay row to this active session.
            let primary = Store::lookup_primary(
                client,
                &self.host,
                &expected.sid,
                &expected.generation,
                &self.inventory.id(),
            )
                .await?
                .ok_or(Error::Inactive)?;
            let active = primary.session();
            let created_at = primary.created_at();
            if active.collision_inventory_id != self.inventory.id()
                || !self.inventory.permits(&active.ed_public, &active.pq_public)
                || active.subject != expected.subject
                || active.tenant != expected.tenant
                || active.client_id != expected.client
                || active.resource != expected.audience
                || active.scopes != expected.scopes
                || active.ed_public != expected.ed_public
                || signer_suite_thumbprint(SUITE, &[&active.ed_public, &active.pq_public])
                    != expected.suite_thumbprint
                || expected.expires_at > active.expires_at
                || expected.expires_at <= created_at
            {
                return Err(Error::Inactive);
            }
            let now = i64::try_from(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(|_| Error::Unavailable)?
                    .as_secs(),
            )
            .map_err(|_| Error::Unavailable)?;
            if expected.expires_at <= now {
                return Err(Error::Expired);
            }
            Ok(ActiveSessionPrimary { session: active.clone(), created_at })
        })
        .await
        .map_err(|_| Error::Unavailable)?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_inventory_is_order_independent_and_rejects_missing_sources() -> Result<(), Error> {
        let required = vec!["static:all".into(), "envelope:local".into()];
        let source = |id: &str, key: u8| InventorySource {
            id: id.into(),
            keys: vec![vec![key; 32]],
        };
        let a = CollisionInventory::from_sources(
            &required,
            vec![source("static:all", 1), source("envelope:local", 2)],
        )?;
        let b = CollisionInventory::from_sources(
            &required,
            vec![source("envelope:local", 2), source("static:all", 1)],
        )?;
        assert_eq!(a.id(), b.id());
        assert!(!a.permits(&[1; 32], &[3; 1952]));
        assert!(a.permits(&[3; 32], &[3; 1952]));
        assert!(
            CollisionInventory::from_sources(&required, vec![source("static:all", 1)]).is_err()
        );
        assert!(CollisionInventory::from_sources(
            &required,
            vec![source("static:all", 1), source("static:all", 2)]
        )
        .is_err());
        assert!(CollisionInventory::from_sources(&[], vec![]).is_err());
        let c = CollisionInventory::from_sources(
            &required,
            vec![source("static:all", 3), source("envelope:local", 2)],
        )?;
        assert_ne!(a.id(), c.id());
        Ok(())
    }
}
