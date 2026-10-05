//! Disabled internal read path. No runtime factory or public provider setter.
//! Future wiring must supply a role-scoped TLS pool and complete inventory.
use super::EnvelopeContext;
use hyprstream_rpc_std::policy_client::{ResolveSessionPrimary, SessionPrimary};
use hyprstream_session_store::primary::{ExpectedPrimary, PrimaryLookup};
use hyprstream_session_store::{Error, PROFILE, SUITE};

pub(super) struct SessionPrimaryReader {
    pub(super) pool: deadpool_postgres::Pool,
    pub(super) authority: PrimaryLookup,
    pub(super) in_flight: tokio::sync::Semaphore,
}

impl SessionPrimaryReader {
    pub(super) async fn resolve(
        &self,
        ctx: &EnvelopeContext,
        data: &ResolveSessionPrimary,
    ) -> Result<SessionPrimary, Error> {
        let subject = ctx.subject();
        let service = subject.name().ok_or(Error::Inactive)?;
        let name = service.strip_prefix("service:").ok_or(Error::Inactive)?;
        let key = ed25519_dalek::VerifyingKey::from_bytes(&ctx.cnf).map_err(|_| Error::Inactive)?;
        if ctx.cnf == [0; 32] || !hyprstream_service::global_trust_store().is_authorized(&key, name)
        {
            return Err(Error::Inactive);
        }
        let expected = ExpectedPrimary {
            issuer: data.issuer.clone(),
            profile: data.profile.clone(),
            sid: data.sid.clone(),
            subject: data.subject.clone(),
            tenant: data.tenant.clone(),
            client: data.client.clone(),
            audience: data.audience.clone(),
            scopes: data.scopes.clone(),
            ed_public: data
                .ed_public
                .as_slice()
                .try_into()
                .map_err(|_| Error::Invalid)?,
            suite_thumbprint: data
                .suite_thumbprint
                .as_slice()
                .try_into()
                .map_err(|_| Error::Invalid)?,
            generation: data
                .generation
                .as_slice()
                .try_into()
                .map_err(|_| Error::Invalid)?,
            expires_at: data.expires_at,
        };
        // Includes pool wait; no unbounded queue behind the two-second SQL gate.
        let _permit = self
            .in_flight
            .try_acquire()
            .map_err(|_| Error::Unavailable)?;
        let active = tokio::time::timeout(std::time::Duration::from_secs(2), async {
            let mut client = self.pool.get().await.map_err(|_| Error::Unavailable)?;
            self.authority
                .resolve(&mut client, service, &ctx.cnf, &expected)
                .await
        })
        .await
        .map_err(|_| Error::Unavailable)??;
        let s = active.session;
        Ok(SessionPrimary {
            host: s.host,
            profile: PROFILE.into(),
            suite: SUITE.into(),
            sid: s.sid,
            account_id: s.account_id,
            subject: s.subject,
            tenant: s.tenant,
            client: s.client_id,
            resource: s.resource,
            scopes: s.scopes,
            grant_revision: s.grant_revision,
            ed_public: s.ed_public.to_vec(),
            pq_public: s.pq_public,
            generation: s.generation.to_vec(),
            created_at: active.created_at,
            expires_at: s.expires_at,
            proof_epoch: active.proof_epoch,
            collision_inventory_id: active.collision_inventory_id.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyprstream_session_store::primary::{
        CollisionInventory, InventorySource, LookupCapability,
    };

    #[tokio::test]
    async fn h3a_unregistered_service_cannot_use_a_lookup_capability() -> anyhow::Result<()> {
        let key = ed25519_dalek::SigningKey::from_bytes(&[77; 32])
            .verifying_key()
            .to_bytes();
        // No DB exists at this fixture endpoint. Denial MUST precede a pool fetch.
        let mut config = tokio_postgres::Config::new();
        config.host("127.0.0.1").port(1);
        let pool = deadpool_postgres::Pool::builder(deadpool_postgres::Manager::new(
            config,
            tokio_postgres::NoTls,
        ))
        .max_size(1)
        .build()?;
        let inventory = CollisionInventory::from_sources(
            &["fixture:static".into()],
            vec![InventorySource {
                id: "fixture:static".into(),
                keys: vec![],
            }],
        )?;
        let reader = SessionPrimaryReader {
            pool,
            authority: PrimaryLookup::new(
                "https://host.test".into(),
                inventory,
                vec![LookupCapability {
                    service: "service:h3a-unregistered".into(),
                    service_key: key,
                    tenant: "tenant".into(),
                    resource: "https://host.test".into(),
                }],
            )?,
            in_flight: tokio::sync::Semaphore::new(1),
        };
        let data = ResolveSessionPrimary {
            issuer: "https://host.test".into(),
            profile: PROFILE.into(),
            sid: "sid".into(),
            subject: "alice".into(),
            tenant: "tenant".into(),
            client: "client".into(),
            audience: "https://host.test".into(),
            scopes: vec!["model:query".into()],
            ed_public: vec![1; 32],
            suite_thumbprint: vec![2; 32],
            generation: vec![3; 32],
            expires_at: 1,
        };
        for subject in ["service:h3a-unregistered", "alice"] {
            let ctx = EnvelopeContext::for_test_authenticated_subject(
                hyprstream_rpc::envelope::Subject::new(subject),
                ed25519_dalek::VerifyingKey::from_bytes(&key)?,
            );
            assert!(matches!(
                reader.resolve(&ctx, &data).await,
                Err(Error::Inactive)
            ));
            assert_eq!(
                reader.pool.status().size,
                0,
                "unauthorized request must not acquire DB connection"
            );
        }
        Ok(())
    }
}
