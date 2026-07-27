//! Relational UserStore on the shared #1351 PGlite/Postgres substrate.
//!
//! The database handle is injected so AppView inventory and credentials use one
//! embedded database.  Server PostgreSQL can use the same schema/repository.

use super::{
    HostedAccountProvision, HostedAccountProvisionResult, PubkeyEntry, UserFilter, UserProfile,
    UserProfilePatch, UserStore,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use ed25519_dalek::VerifyingKey;
use pglite::{PGlite, Row};
use std::{path::Path, sync::Arc};
use uuid::Uuid;

pub const USERSTORE_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS users (
 username TEXT PRIMARY KEY, sub TEXT NOT NULL UNIQUE, profile BYTEA NOT NULL,
 active BOOLEAN NOT NULL DEFAULT TRUE, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS user_did_bindings (
 username TEXT PRIMARY KEY REFERENCES users(username) ON DELETE CASCADE,
 atproto_did TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS oidc_bindings (
 issuer TEXT NOT NULL, issuer_sub TEXT NOT NULL,
 username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
 PRIMARY KEY (issuer, issuer_sub), UNIQUE (issuer, username)
);
CREATE TABLE IF NOT EXISTS pubkeys (
 fingerprint TEXT PRIMARY KEY, username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
 pubkey BYTEA NOT NULL, label BYTEA, algorithm TEXT NOT NULL DEFAULT 'ed25519', pq_pubkey BYTEA,
 created_at BIGINT NOT NULL, last_used_at BIGINT,
 CHECK ((algorithm = 'ed25519') = (pq_pubkey IS NULL))
);
CREATE INDEX IF NOT EXISTS pubkeys_username_idx ON pubkeys(username);
"#;

#[derive(Clone)]
pub struct PgliteUserStore {
    pub(crate) database: Arc<PGlite>,
}

impl PgliteUserStore {
    pub async fn open(data_dir: impl AsRef<Path>) -> Result<Self> {
        let db = Arc::new(PGlite::open(data_dir).await.context("open PGlite")?);
        Self::from_database(db).await
    }
    pub async fn from_database(database: Arc<PGlite>) -> Result<Self> {
        database
            .exec(USERSTORE_SCHEMA)
            .await
            .context("create UserStore schema")?;
        Ok(Self { database })
    }
    pub fn database(&self) -> Arc<PGlite> {
        Arc::clone(&self.database)
    }
}

fn blob<T: serde::Serialize>(v: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(v)?)
}
fn decode_profile(row: &Row) -> Result<UserProfile> {
    let bytes = row.get::<Vec<u8>>(0)?;
    Ok(serde_json::from_slice(&bytes)?)
}

#[async_trait]
impl UserStore for PgliteUserStore {
    async fn provision_hosted_account(
        &self,
        request: HostedAccountProvision,
    ) -> Result<HostedAccountProvisionResult> {
        let tx = self.database.transaction().await?;
        let existing = tx.query("SELECT username, sub FROM users WHERE username=$1 OR sub=$2", &[&request.username, &request.sub]).await?;
        if !existing.is_empty() {
            let same = existing.iter().any(|r| r.get::<String>(0).ok().as_deref() == Some(&request.username) && r.get::<String>(1).ok().as_deref() == Some(&request.sub));
            if same { tx.commit().await?; return Ok(HostedAccountProvisionResult::Resumed); }
            tx.rollback().await?;
            anyhow::bail!("hosted account conflict is not an exact trusted match")
        }
        let profile = UserProfile { sub: Some(request.sub.clone()), atproto_did: Some(request.atproto_did.clone()), active: Some(false), ..Default::default() };
        let bytes = blob(&profile)?;
        tx.query("INSERT INTO users(username,sub,profile,active) VALUES($1,$2,$3,FALSE)", &[&request.username, &request.sub, &bytes]).await?;
        tx.query("INSERT INTO user_did_bindings(username,atproto_did) VALUES($1,$2)", &[&request.username, &request.atproto_did]).await?;
        tx.query("INSERT INTO pubkeys(fingerprint,username,pubkey,pq_pubkey,algorithm,created_at) VALUES($1,$2,$3,$4,$5,$6)", &[&request.fingerprint, &request.username, &request.pubkey, &request.pq_pubkey, &if request.pq_pubkey.is_some() { "ed25519+ml-dsa-65" } else { "ed25519" }, &chrono::Utc::now().timestamp()]).await?;
        tx.commit().await?;
        Ok(HostedAccountProvisionResult::Provisioned)
    }
    async fn bind_external_idp(&self, issuer: &str, issuer_subject: &str, username: &str) -> Result<()> {
        let tx = self.database.transaction().await?;
        tx.query("INSERT INTO oidc_bindings(issuer,issuer_sub,username) VALUES($1,$2,$3)", &[&issuer,&issuer_subject,&username]).await?;
        tx.commit().await?; Ok(())
    }
    async fn resolve_external_idp(&self, issuer: &str, issuer_subject: &str) -> Result<Option<String>> {
        let rows=self.database.query("SELECT username FROM oidc_bindings WHERE issuer=$1 AND issuer_sub=$2", &[&issuer,&issuer_subject]).await?;
        rows.first().map(|r| r.get::<String>(0)).transpose().map_err(Into::into)
    }
    async fn get_profile(&self, username: &str) -> Result<Option<UserProfile>> {
        let rows = self
            .database
            .query("SELECT profile FROM users WHERE username=$1", &[&username])
            .await?;
        rows.first().map(decode_profile).transpose()
    }
    async fn register(&self, username: &str) -> Result<String> {
        let sub = Uuid::new_v4().to_string();
        let profile = UserProfile {
            sub: Some(sub.clone()),
            active: Some(true),
            ..Default::default()
        };
        let bytes = blob(&profile)?;
        self.database
            .query(
                "INSERT INTO users(username,sub,profile) VALUES($1,$2,$3)",
                &[&username, &sub, &bytes],
            )
            .await?;
        Ok(sub)
    }
    async fn set_profile(&self, username: &str, patch: UserProfilePatch) -> Result<()> {
        let mut p = self.get_profile(username).await?.context("unknown user")?;
        macro_rules! apply {
            ($f:ident) => {
                if let Some(v) = patch.$f {
                    p.$f = v;
                }
            };
        }
        apply!(sub);
        apply!(name);
        apply!(email);
        apply!(email_verified);
        apply!(active);
        apply!(external_id);
        apply!(atproto_did);
        let bytes = blob(&p)?;
        self.database.query("UPDATE users SET sub=COALESCE($2,sub), profile=$3, active=COALESCE($4,active), updated_at=now() WHERE username=$1", &[&username, &p.sub, &bytes, &p.active]).await?;
        Ok(())
    }
    async fn remove(&self, username: &str) -> Result<bool> {
        let rows = self
            .database
            .query(
                "DELETE FROM users WHERE username=$1 RETURNING username",
                &[&username],
            )
            .await?;
        Ok(!rows.is_empty())
    }
    async fn list_users(&self) -> Vec<String> {
        self.database
            .query("SELECT username FROM users ORDER BY username", &[])
            .await
            .unwrap_or_default()
            .iter()
            .filter_map(|r| r.get::<String>(0).ok())
            .collect()
    }
    async fn search(&self, _filter: &UserFilter) -> Result<Vec<(String, UserProfile)>> {
        let rows = self
            .database
            .query("SELECT username,profile FROM users ORDER BY username", &[])
            .await?;
        rows.into_iter()
            .map(|r| {
                Ok((
                    r.get::<String>(0)?,
                    serde_json::from_slice(&r.get::<Vec<u8>>(1)?)?,
                ))
            })
            .collect()
    }
    async fn set_active(&self, username: &str, active: bool) -> Result<()> {
        self.database
            .query(
                "UPDATE users SET active=$2, updated_at=now() WHERE username=$1",
                &[&username, &active],
            )
            .await?;
        Ok(())
    }
    async fn list_pubkeys(&self, _username: &str) -> Result<Vec<PubkeyEntry>> {
        anyhow::bail!("PGlite pubkey repository is pending schema adapter")
    }
    async fn add_pubkey(
        &self,
        _username: &str,
        _pubkey: VerifyingKey,
        _label: Option<String>,
    ) -> Result<String> {
        anyhow::bail!("PGlite pubkey repository is pending schema adapter")
    }
    async fn add_pubkey_hybrid(
        &self,
        _username: &str,
        _pubkey: VerifyingKey,
        _ml_dsa_vk: Vec<u8>,
        _label: Option<String>,
    ) -> Result<String> {
        anyhow::bail!("PGlite pubkey repository is pending schema adapter")
    }
    async fn remove_pubkey(&self, _: &str, _: &str) -> Result<bool> {
        anyhow::bail!("PGlite pubkey repository is pending schema adapter")
    }
    async fn get_pubkey_user(&self, _: &str) -> Result<Option<String>> {
        anyhow::bail!("PGlite pubkey repository is pending schema adapter")
    }
    async fn touch_pubkey(&self, _: &str, _: &str) -> Result<()> {
        anyhow::bail!("PGlite pubkey repository is pending schema adapter")
    }
}
