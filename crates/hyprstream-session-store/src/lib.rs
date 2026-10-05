//! Source-only internal persistence foundation. No network listener, token signer,
//! runtime factory, migration-on-start or account authority is installed here.
//! Policy must supply an admitted account/tenant/grant decision after authenticated
//! OAuth has verified the source token and both possession signatures. This API
//! is not a verifier: never pass a deserialized browser request directly to it.
//! The caller supplies a dedicated, TLS-verified PostgreSQL connection under the
//! narrow runtime role. No DSN or token bodies are accepted or logged here.

use tokio_postgres::{error::SqlState, Client, IsolationLevel, Row};
pub mod primary;

pub const PROFILE: &str = "federate-session-v1";
pub const SUITE: &str = "hs-cose-sign-ed25519-mldsa65-wns-v1";
pub const MIGRATION: &str = include_str!("../sql/001_admission.sql");
pub const ROLE_GRANTS: &str = include_str!("../sql/roles.sql");
pub const MIGRATION_V2: &str = include_str!("../sql/002_epoch_inventory.sql");
pub const ROLE_GRANTS_V2: &str = include_str!("../sql/002_roles.sql");

/// Sanitized failures: PostgreSQL detail strings can include bound values.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Error {
    #[error("admission input invalid")]
    Invalid,
    #[error("profile disabled, missing or generation mismatch")]
    Inactive,
    #[error("source, challenge or grant expired")]
    Expired,
    #[error("source or session already consumed")]
    Conflict,
    #[error("session authority unavailable")]
    Unavailable,
}

impl From<tokio_postgres::Error> for Error {
    fn from(e: tokio_postgres::Error) -> Self {
        if e.code() == Some(&SqlState::UNIQUE_VIOLATION) {
            Self::Conflict
        } else {
            Self::Unavailable
        }
    }
}

/// Upstream-verified source metadata; never the raw ID token.
pub struct Source {
    pub issuer: String,
    pub subject: String,
    pub jti: String,
    pub nonce: String,
    pub token_hash: [u8; 32],
    pub issued_at: i64,
    pub expires_at: i64,
}

/// Policy's authoritative pre-admission decision. No epoch can be supplied:
/// PostgreSQL generates it only for a committed session.
#[derive(Clone, PartialEq, Eq)]
pub struct PendingSession {
    pub host: String,
    pub sid: String,
    pub account_id: String,
    pub subject: String,
    pub tenant: String,
    pub client_id: String,
    pub resource: String,
    pub scopes: Vec<String>,
    pub grant_revision: String,
    pub ed_public: [u8; 32],
    pub pq_public: Vec<u8>,
    pub generation: [u8; 32],
    /// Server-owned ID of the complete, locally validated collision inventory.
    pub collision_inventory_id: [u8; 32],
    pub expires_at: i64,
}

/// An admitted, immutable record returned only after the transaction commits.
#[derive(Clone, PartialEq, Eq)]
pub struct Session {
    pub host: String,
    pub sid: String,
    pub account_id: String,
    pub subject: String,
    pub tenant: String,
    pub client_id: String,
    pub resource: String,
    pub scopes: Vec<String>,
    pub grant_revision: String,
    pub ed_public: [u8; 32],
    pub pq_public: Vec<u8>,
    pub generation: [u8; 32],
    pub collision_inventory_id: [u8; 32],
    pub proof_epoch: i64,
    pub expires_at: i64,
}

impl PendingSession {
    fn admitted(&self, proof_epoch: i64) -> Result<Session, Error> {
        if proof_epoch <= 0 {
            return Err(Error::Unavailable);
        }
        Ok(Session {
            host: self.host.clone(),
            sid: self.sid.clone(),
            account_id: self.account_id.clone(),
            subject: self.subject.clone(),
            tenant: self.tenant.clone(),
            client_id: self.client_id.clone(),
            resource: self.resource.clone(),
            scopes: self.scopes.clone(),
            grant_revision: self.grant_revision.clone(),
            ed_public: self.ed_public,
            pq_public: self.pq_public.clone(),
            generation: self.generation,
            collision_inventory_id: self.collision_inventory_id,
            proof_epoch,
            expires_at: self.expires_at,
        })
    }
}

pub struct Admission {
    pub session: PendingSession,
    pub source: Source,
    /// Independently loaded, complete local inventory. Never browser input.
    pub local_collision_inventory_id: [u8; 32],
    /// Expiry of the server-owned pending challenge, not supplied by a browser.
    pub challenge_created_at: i64,
    pub challenge_expires_at: i64,
    pub grant_expires_at: i64,
}

fn bounded(s: &str, max: usize) -> bool {
    !s.is_empty() && s.len() <= max && !s.contains('\0')
}

impl Admission {
    fn validate(&self, now: i64) -> Result<i64, Error> {
        let s = &self.session;
        let src = &self.source;
        if !bounded(&s.host, 2048)
            || !bounded(&s.resource, 2048)
            || !bounded(&src.issuer, 2048)
            || !bounded(&s.sid, 128)
            || [
                &s.account_id,
                &s.subject,
                &s.tenant,
                &s.client_id,
                &s.grant_revision,
                &src.subject,
                &src.jti,
                &src.nonce,
            ]
            .iter()
            .any(|v| !bounded(v, 256))
            || s.pq_public.len() != 1952
            || s.scopes.is_empty()
            || s.scopes.len() > 64
            || s.scopes
                .iter()
                .any(|v| !bounded(v, 256) || !v.bytes().all(|b| (0x21..=0x7e).contains(&b)))
            || s.scopes.windows(2).any(|w| w[0] >= w[1])
            || src.issued_at < 0
            || src.expires_at <= src.issued_at
            || src
                .expires_at
                .checked_sub(src.issued_at)
                .ok_or(Error::Invalid)?
                > 300
        {
            return Err(Error::Invalid);
        }
        let age = now.checked_sub(src.issued_at).ok_or(Error::Invalid)?;
        if !(-30..=120).contains(&age)
            || now >= src.expires_at
            || self.challenge_created_at > now
            || self.challenge_created_at < 0
            || self
                .challenge_expires_at
                .checked_sub(self.challenge_created_at)
                .ok_or(Error::Invalid)?
                > 60
            || now >= self.challenge_expires_at
            || now >= s.expires_at
            || s.expires_at > self.grant_expires_at
            || s.expires_at > src.expires_at
            || s.expires_at.checked_sub(now).ok_or(Error::Invalid)? > 300
            || self.challenge_expires_at > src.expires_at
            || self.challenge_expires_at > src.issued_at.checked_add(120).ok_or(Error::Invalid)?
        {
            return Err(Error::Expired);
        }
        Ok(src
            .issued_at
            .checked_add(330)
            .ok_or(Error::Invalid)?
            .max(src.expires_at.checked_add(30).ok_or(Error::Invalid)?)
            .max(s.expires_at))
    }
}

/// Owns no pool or fallback authority. A dedicated connection is borrowed per
/// operation; caller cancellation drops/rolls back the transaction. Commit errors
/// (including ambiguous commit outcome) never return a usable admission receipt.
pub struct Store;

impl Store {
    pub async fn admit(client: &mut Client, admission: &Admission) -> Result<Session, Error> {
        let tx = client
            .build_transaction()
            .isolation_level(IsolationLevel::ReadCommitted)
            .start()
            .await?;
        // Bound lock/statement time. No parameter values are included in errors.
        tx.batch_execute("SET LOCAL lock_timeout = '2s'; SET LOCAL statement_timeout = '2s'")
            .await?;
        let version: i32 = tx
            .query_one("SELECT version FROM federate_session.schema_version", &[])
            .await?
            .try_get(0)?;
        if version != 2 {
            return Err(Error::Unavailable);
        }
        let s = &admission.session;
        let row = tx.query_opt(
            "SELECT enabled, authority_generation, collision_inventory_id FROM federate_session.profile_state WHERE host=$1 AND profile=$2 FOR UPDATE",
            &[&s.host, &PROFILE],
        ).await?.ok_or(Error::Inactive)?;
        let enabled: bool = row.try_get(0)?;
        let generation: Vec<u8> = row.try_get(1)?;
        let inventory: Vec<u8> = row.try_get(2)?;
        if !enabled
            || generation != s.generation
            || inventory != s.collision_inventory_id
            || inventory != admission.local_collision_inventory_id
        {
            return Err(Error::Inactive);
        }
        // PostgreSQL transaction timestamps precede lock waits; use clock_timestamp.
        let now: i64 = tx
            .query_one(
                "SELECT floor(extract(epoch FROM clock_timestamp()))::bigint",
                &[],
            )
            .await?
            .try_get(0)?;
        let retain_until = admission.validate(now)?;
        let proof_epoch: i64 = tx.query_one(
            "INSERT INTO federate_session.sessions (host,profile,sid,account_id,subject,tenant,client_id,resource,scopes,grant_revision,suite,ed_public,pq_public,generation,collision_inventory_id,created_at,expires_at) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17) RETURNING proof_epoch",
            &[&s.host,&PROFILE,&s.sid,&s.account_id,&s.subject,&s.tenant,&s.client_id,&s.resource,&s.scopes,&s.grant_revision,&SUITE,&&s.ed_public[..],&s.pq_public,&generation,&inventory,&now,&s.expires_at],
        ).await?.try_get(0)?;
        let admitted = s.admitted(proof_epoch)?;
        let src = &admission.source;
        tx.execute(
            "INSERT INTO federate_session.replay (issuer,client_id,jti,nonce,token_hash,source_iat,source_exp,retain_until,host,sid,source_subject) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
            &[&src.issuer,&s.client_id,&src.jti,&src.nonce,&&src.token_hash[..],&src.issued_at,&src.expires_at,&retain_until,&s.host,&s.sid,&src.subject],
        ).await?;
        tx.commit().await?;
        Ok(admitted)
    }

    /// Resolve by issuer/sid and compare the complete expected credential binding.
    /// No positive cache and no static-primary fallback. SQL/schema errors deny.
    pub async fn active(client: &Client, expected: &Session) -> Result<bool, Error> {
        Ok(Self::lookup(
            client,
            &expected.host,
            &expected.sid,
            &expected.generation,
            &expected.collision_inventory_id,
        )
        .await?
        .as_ref()
            == Some(expected))
    }

    /// Authoritative full-record lookup for the future request-local proof resolver.
    /// Caller must still compare all credential fields and verify the RPC proof.
    pub async fn lookup(
        client: &(impl tokio_postgres::GenericClient + Sync),
        host: &str,
        sid: &str,
        generation: &[u8; 32],
        local_collision_inventory_id: &[u8; 32],
    ) -> Result<Option<Session>, Error> {
        let row = client.query_opt(
            "SELECT s.* FROM federate_session.sessions s JOIN federate_session.profile_state p ON (s.host=p.host AND s.profile=p.profile) WHERE s.host=$1 AND s.sid=$2 AND s.profile=$3 AND s.suite=$4 AND s.status='active' AND p.enabled AND s.generation=p.authority_generation AND s.generation=$5 AND s.collision_inventory_id=p.collision_inventory_id AND s.collision_inventory_id=$6 AND s.proof_epoch>0 AND s.expires_at>floor(extract(epoch FROM clock_timestamp()))::bigint",
            &[&host,&sid,&PROFILE,&SUITE,&&generation[..],&&local_collision_inventory_id[..]],
        ).await?;
        row.as_ref().map(read_session).transpose()
    }

    /// Atomically consumes an already-verified v16 `(namespace, request_id)`
    /// against the current exact session. The caller must derive `namespace`
    /// from the verified suite, ordered public components and stored epoch;
    /// this store neither parses proofs nor offers a runtime verifier factory.
    /// Unknown commit outcomes deny; no local-cache fallback is possible.
    pub async fn consume_request_replay(
        client: &mut Client,
        expected: &Session,
        verified_namespace: &[u8; 32],
        request_id: &[u8; 16],
        configured_clock_skew_secs: i64,
    ) -> Result<(), Error> {
        if expected.proof_epoch <= 0 || configured_clock_skew_secs < 0 {
            return Err(Error::Invalid);
        }
        let retain_until = expected
            .expires_at
            .checked_add(configured_clock_skew_secs)
            .ok_or(Error::Invalid)?;
        let tx = client
            .build_transaction()
            .isolation_level(IsolationLevel::ReadCommitted)
            .start()
            .await?;
        tx.batch_execute("SET LOCAL lock_timeout = '2s'; SET LOCAL statement_timeout = '2s'")
            .await?;
        let version: i32 = tx
            .query_one("SELECT version FROM federate_session.schema_version", &[])
            .await?
            .try_get(0)?;
        if version != 2 {
            return Err(Error::Unavailable);
        }
        let profile = tx.query_opt(
            "SELECT enabled, authority_generation, collision_inventory_id FROM federate_session.profile_state WHERE host=$1 AND profile=$2 FOR UPDATE",
            &[&expected.host, &PROFILE],
        ).await?.ok_or(Error::Inactive)?;
        let enabled: bool = profile.try_get(0)?;
        let generation: Vec<u8> = profile.try_get(1)?;
        let inventory: Vec<u8> = profile.try_get(2)?;
        if !enabled
            || generation != expected.generation
            || inventory != expected.collision_inventory_id
        {
            return Err(Error::Inactive);
        }
        let row = tx.query_opt(
            "SELECT * FROM federate_session.sessions WHERE host=$1 AND sid=$2 AND profile=$3 AND suite=$4 AND status='active' FOR SHARE",
            &[&expected.host, &expected.sid, &PROFILE, &SUITE],
        ).await?.ok_or(Error::Inactive)?;
        if &read_session(&row)? != expected {
            return Err(Error::Inactive);
        }
        let now: i64 = tx
            .query_one(
                "SELECT floor(extract(epoch FROM clock_timestamp()))::bigint",
                &[],
            )
            .await?
            .try_get(0)?;
        if now >= expected.expires_at {
            return Err(Error::Expired);
        }
        tx.execute(
            "INSERT INTO federate_session.request_replay (verified_namespace,request_id,retain_until) VALUES ($1,$2,$3)",
            &[&&verified_namespace[..], &&request_id[..], &retain_until],
        ).await?;
        tx.commit().await?;
        Ok(())
    }

    pub async fn revoke(
        client: &Client,
        host: &str,
        sid: &str,
        generation: &[u8; 32],
    ) -> Result<bool, Error> {
        Ok(client.execute(
            "UPDATE federate_session.sessions SET status='revoked' WHERE host=$1 AND sid=$2 AND generation=$3 AND profile=$4",
            &[&host,&sid,&&generation[..],&PROFILE],
        ).await? == 1)
    }

    /// Maintenance-role only; one transaction, expired replay first. Runtime
    /// has no DELETE privilege. No active or replay-protected session is removed.
    pub async fn cleanup(client: &mut Client) -> Result<(), Error> {
        let tx = client.transaction().await?;
        tx.execute("DELETE FROM federate_session.request_replay WHERE retain_until<floor(extract(epoch FROM clock_timestamp()))::bigint", &[]).await?;
        tx.execute("DELETE FROM federate_session.replay WHERE retain_until<floor(extract(epoch FROM clock_timestamp()))::bigint", &[]).await?;
        tx.execute("DELETE FROM federate_session.sessions s WHERE expires_at<floor(extract(epoch FROM clock_timestamp()))::bigint AND NOT EXISTS (SELECT 1 FROM federate_session.replay r WHERE r.host=s.host AND r.sid=s.sid)", &[]).await?;
        tx.commit().await?;
        Ok(())
    }
}

fn read_session(row: &Row) -> Result<Session, Error> {
    Ok(Session {
        host: row.try_get("host")?,
        sid: row.try_get("sid")?,
        account_id: row.try_get("account_id")?,
        subject: row.try_get("subject")?,
        tenant: row.try_get("tenant")?,
        client_id: row.try_get("client_id")?,
        resource: row.try_get("resource")?,
        scopes: row.try_get("scopes")?,
        grant_revision: row.try_get("grant_revision")?,
        ed_public: row
            .try_get::<_, Vec<u8>>("ed_public")?
            .try_into()
            .map_err(|_| Error::Unavailable)?,
        pq_public: row.try_get("pq_public")?,
        generation: row
            .try_get::<_, Vec<u8>>("generation")?
            .try_into()
            .map_err(|_| Error::Unavailable)?,
        collision_inventory_id: row
            .try_get::<_, Vec<u8>>("collision_inventory_id")?
            .try_into()
            .map_err(|_| Error::Unavailable)?,
        proof_epoch: {
            let epoch: i64 = row.try_get("proof_epoch")?;
            if epoch <= 0 {
                return Err(Error::Unavailable);
            }
            epoch
        },
        expires_at: row.try_get("expires_at")?,
    })
}
