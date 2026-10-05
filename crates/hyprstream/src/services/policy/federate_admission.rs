//! Disabled Policy-owned admission core. No RPC route, factory or token signer.
//! OAuth's future H4 adapter must supply typed verified possession evidence.
//! All serving requests/tool calls still require the separate H2 authority-at-use
//! gate: admission deliberately does not lock account/PDS/Casbin through commit.
#![allow(dead_code)] // Source-only slice; no runtime installer is exposed.

use std::{collections::BTreeSet, sync::Arc, time::Duration};
use anyhow::{ensure, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use hyprstream_rpc::{auth::Scope, Subject};
use hyprstream_session_store::{Admission, Session, Source, Store};
use tokio::sync::Semaphore;
use super::{EnvelopeContext, PolicyManager};
use crate::auth::{ProductionUserStore, service_enrollment::ServiceEnrollmentManifest};

/// Fixed, authority-owned ceilings; not browser input or dynamic client metadata.
struct Profile {
    issuer: String,
    host: String,
    client: String,
    resource: String,
    client_scopes: BTreeSet<String>,
    resource_scopes: BTreeSet<String>,
}

/// Immutable result of Policy's prepare read. No public field/serde constructor.
#[derive(Clone, PartialEq, Eq)]
struct Decision {
    account_id: String,
    subject: String,
    tenant: String,
    scopes: Vec<String>,
    revision: String,
}

/// Internal upstream evidence, not a browser DTO or verification boolean.
/// There is intentionally no production constructor until H4 implements BOTH
/// signatures over the exact challenge and strict source-token validation.
struct PossessionEvidence {
    source: Source,
    ed_public: [u8; 32],
    pq_public: Vec<u8>,
    sid: String,
    generation: [u8; 32],
    requested: Vec<String>,
    challenge: Decision,
    created_at: i64,
    expires_at: i64,
    session_expires_at: i64,
}

struct Authorities {
    users: ProductionUserStore,
    accounts: Arc<hyprstream_pds_service::AccountRecordStore>,
    policy: Arc<PolicyManager>,
    enrollment: Arc<ServiceEnrollmentManifest>,
    profile: Profile,
}

/// Default denies before any authority or DB access. Future wiring must admit a
/// writer-endpoint UserStore, fresh signed PDS mount and sole serving Policy.
#[derive(Default)]
struct AdmissionService {
    authority: Option<Authorities>,
    capacity: Option<Semaphore>,
}

fn oauth_caller(ctx: &EnvelopeContext, manifest: &ServiceEnrollmentManifest) -> Result<()> {
    let subject = ctx.subject();
    ensure!(ctx.is_authenticated() && ctx.claims().is_some()
        && !subject.is_federated() && subject.name() == Some("service:oauth"),
        "enrolled OAuth caller required");
    let claims = ctx.claims().ok_or_else(|| anyhow::anyhow!("service claims missing"))?;
    ensure!(claims.sub == "service:oauth" && claims.act.is_none() && claims.sid.is_none(),
        "direct OAuth service credential required");
    let entry = manifest.services.get("oauth")
        .ok_or_else(|| anyhow::anyhow!("OAuth enrollment missing"))?;
    let enrolled = URL_SAFE_NO_PAD.decode(&entry.ed25519_pubkey)?;
    ensure!(enrolled.as_slice() == ctx.cnf, "OAuth holder mismatch");
    Ok(())
}

fn canonical_requested(requested: &[String]) -> Result<()> {
    ensure!(!requested.is_empty() && requested.len() <= 64, "invalid scope count");
    ensure!(requested.windows(2).all(|w| w[0] < w[1]), "noncanonical scopes");
    for text in requested {
        ensure!(text.len() <= 256 && text.bytes().all(|b| (0x21..=0x7e).contains(&b)), "invalid scope");
        let scope = Scope::parse(text)?;
        ensure!([&scope.action, &scope.resource, &scope.identifier].iter()
            .all(|s| !s.is_empty() && !s.contains('*')), "concrete scopes required");
    }
    Ok(())
}

impl Authorities {
    async fn decision(&self, source: &Source, requested: &[String]) -> Result<Decision> {
        ensure!(source.issuer == self.profile.issuer, "issuer mismatch");
        canonical_requested(requested)?;
        let username = self.users.get_external_identity_user(&source.issuer, &source.subject)
            .await?.ok_or_else(|| anyhow::anyhow!("account not admitted"))?;
        ensure!(!username.is_empty() && !username.starts_with("service:") && username != "anonymous",
            "local account subject required");
        let profile = self.users.get_profile(&username).await?
            .ok_or_else(|| anyhow::anyhow!("account not admitted"))?;
        ensure!(profile.active == Some(true), "inactive account");
        let account_id = profile.sub.ok_or_else(|| anyhow::anyhow!("account UUID missing"))?;
        uuid::Uuid::parse_str(&account_id)?;
        let did = profile.atproto_did.ok_or_else(|| anyhow::anyhow!("hosted DID missing"))?;
        let tenant = self.accounts.resolve_current_tenant_for_hosted_did(
            &Subject::new(hyprstream_pds_service::OAUTH_ACCOUNT_RESOLVER_SUBJECT), &did,
        ).await?.ok_or_else(|| anyhow::anyhow!("signed tenant missing"))?;
        ensure!(!tenant.is_empty() && tenant != "*", "invalid tenant");
        let mut scopes = Vec::new();
        for text in requested {
            if !self.profile.client_scopes.contains(text) || !self.profile.resource_scopes.contains(text) {
                continue;
            }
            let scope = Scope::parse(text)?;
            if self.policy.check_with_domain(&username, &tenant,
                &format!("{}:{}", scope.resource, scope.identifier), &scope.action).await {
                scopes.push(text.clone());
            }
        }
        ensure!(!scopes.is_empty(), "no grants");
        // Revision of this observed decision, not a global mutation epoch. A
        // change after these reads is accepted; use-time authorization is vital.
        let bytes = serde_json::to_vec(&(
            "federate-policy-decision-v1", &self.profile.issuer, &self.profile.host,
            &self.profile.client, &self.profile.resource, &self.profile.client_scopes,
            &self.profile.resource_scopes, &source.subject, &account_id, &username,
            &did, &tenant, requested, &scopes,
        ))?;
        let revision = blake3::hash(&bytes).to_hex().to_string();
        Ok(Decision { account_id, subject: username, tenant, scopes, revision })
    }
}

#[cfg(test)]
mod tests;

impl AdmissionService {
    fn authority(&self, ctx: &EnvelopeContext) -> Result<&Authorities> {
        let authority = self.authority.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Federate admission disabled"))?;
        oauth_caller(ctx, &authority.enrollment)?;
        Ok(authority)
    }

    async fn prepare(&self, ctx: &EnvelopeContext, source: &Source, requested: &[String]) -> Result<Decision> {
        let a = self.authority(ctx)?;
        let _permit = self.capacity.as_ref().ok_or_else(|| anyhow::anyhow!("disabled"))?
            .try_acquire()?;
        tokio::time::timeout(Duration::from_secs(2), a.decision(source, requested)).await?
    }

    async fn redeem(&self, ctx: &EnvelopeContext, client: &mut tokio_postgres::Client,
        evidence: PossessionEvidence) -> Result<Session> {
        let a = self.authority(ctx)?;
        let _permit = self.capacity.as_ref().ok_or_else(|| anyhow::anyhow!("disabled"))?
            .try_acquire()?;
        tokio::time::timeout(Duration::from_secs(2), async {
            let current = a.decision(&evidence.source, &evidence.requested).await?;
            ensure!(current == evidence.challenge, "challenge decision changed");
            let session = Session {
                host: a.profile.host.clone(), sid: evidence.sid, account_id: current.account_id,
                subject: current.subject, tenant: current.tenant, client_id: a.profile.client.clone(),
                resource: a.profile.resource.clone(), scopes: current.scopes,
                grant_revision: current.revision, ed_public: evidence.ed_public,
                pq_public: evidence.pq_public, generation: evidence.generation,
                expires_at: evidence.session_expires_at,
            };
            // No cross-store lock: a mutation here may leave a stale session.
            // No token signer or protected handler is reachable from this core.
            Ok(Store::admit(client, &Admission {
                session, grant_expires_at: evidence.source.expires_at, source: evidence.source,
                challenge_created_at: evidence.created_at, challenge_expires_at: evidence.expires_at,
            }).await?)
        }).await?
    }

    async fn lookup(&self, ctx: &EnvelopeContext, client: &tokio_postgres::Client,
        sid: &str, generation: &[u8; 32]) -> Result<Option<Session>> {
        let a = self.authority(ctx)?;
        let _permit = self.capacity.as_ref().ok_or_else(|| anyhow::anyhow!("disabled"))?
            .try_acquire()?;
        ensure!(!sid.is_empty() && sid.len() <= 128, "invalid sid");
        Ok(tokio::time::timeout(Duration::from_secs(2), Store::lookup(client, &a.profile.host, sid, generation)).await??)
    }

    async fn revoke(&self, ctx: &EnvelopeContext, client: &tokio_postgres::Client,
        sid: &str, generation: &[u8; 32]) -> Result<bool> {
        let a = self.authority(ctx)?;
        let _permit = self.capacity.as_ref().ok_or_else(|| anyhow::anyhow!("disabled"))?
            .try_acquire()?;
        ensure!(!sid.is_empty() && sid.len() <= 128, "invalid sid");
        Ok(tokio::time::timeout(Duration::from_secs(2), Store::revoke(client, &a.profile.host, sid, generation)).await??)
    }
}
