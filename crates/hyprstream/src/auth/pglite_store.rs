//! Relational UserStore on the shared #1351 PGlite/Postgres substrate.
//!
//! The database handle is injected so AppView inventory and credentials use one
//! embedded database.  Server PostgreSQL can use the same schema/repository.

use super::{
    AccountKeyCustody, ExternalIdentityResolution, HostedAccountProvisionError,
    HostedAccountProvisioning, PubkeyEntry, UserFilter, UserProfile, UserProfilePatch, UserStore,
};
use anyhow::{Context, Result};
use async_trait::async_trait;
use ed25519_dalek::VerifyingKey;
use pglite::{PGlite, Row};
use std::{path::Path, sync::Arc};
use uuid::Uuid;

pub const USERSTORE_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY CHECK (username <> ''),
    sub TEXT NOT NULL UNIQUE CHECK (sub <> ''),
    name BYTEA,
    email BYTEA,
    email_verified BOOLEAN,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    external_id BYTEA,
    key_custody TEXT
        CHECK (key_custody IS NULL OR key_custody IN ('self_custody', 'managed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS user_did_bindings (
    username TEXT PRIMARY KEY REFERENCES users(username) ON DELETE CASCADE,
    atproto_did TEXT NOT NULL UNIQUE CHECK (atproto_did <> '')
);
CREATE TABLE IF NOT EXISTS oidc_bindings (
    issuer TEXT NOT NULL CHECK (issuer <> ''),
    issuer_sub TEXT NOT NULL CHECK (issuer_sub <> ''),
    username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    PRIMARY KEY (issuer, issuer_sub),
    UNIQUE (issuer, username)
);
CREATE TABLE IF NOT EXISTS pubkeys (
    fingerprint TEXT PRIMARY KEY CHECK (fingerprint <> ''),
    username TEXT NOT NULL REFERENCES users(username) ON DELETE CASCADE,
    pubkey BYTEA NOT NULL CHECK (octet_length(pubkey) = 32),
    label BYTEA,
    algorithm TEXT NOT NULL DEFAULT 'ed25519'
        CHECK (algorithm IN ('ed25519', 'ed25519+ml-dsa-65')),
    pq_pubkey BYTEA,
    created_at BIGINT NOT NULL,
    last_used_at BIGINT,
    CHECK (
        (algorithm = 'ed25519' AND pq_pubkey IS NULL)
        OR
        (algorithm = 'ed25519+ml-dsa-65'
            AND pq_pubkey IS NOT NULL
            AND octet_length(pq_pubkey) = 1952)
    )
);
CREATE INDEX IF NOT EXISTS pubkeys_username_idx ON pubkeys(username);
CREATE INDEX IF NOT EXISTS oidc_bindings_username_idx ON oidc_bindings(username);
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

fn blob<T: serde::Serialize>(value: &T) -> Result<Vec<u8>> {
    Ok(serde_json::to_vec(value)?)
}

fn decode_profile(row: &Row) -> Result<UserProfile> {
    let bytes = row.get::<Vec<u8>>(0)?;
    Ok(serde_json::from_slice(&bytes)?)
}

#[async_trait]
impl UserStore for PgliteUserStore {
    async fn provision_hosted_account(
        &self,
        _username: &str,
        _atproto_did: &str,
        _pubkey: VerifyingKey,
        _custody: AccountKeyCustody,
    ) -> std::result::Result<HostedAccountProvisioning, HostedAccountProvisionError> {
        Err(HostedAccountProvisionError::Backend(anyhow::anyhow!(
            "PGlite hosted-account repository implementation is pending"
        )))
    }

    async fn activate_hosted_account(
        &self,
        _username: &str,
        _atproto_did: &str,
        _fingerprint: &str,
        _custody: AccountKeyCustody,
    ) -> std::result::Result<(), HostedAccountProvisionError> {
        Err(HostedAccountProvisionError::Backend(anyhow::anyhow!(
            "PGlite hosted-account repository implementation is pending"
        )))
    }

    async fn resolve_or_bind_external_idp(
        &self,
        _issuer: &str,
        _subject: &str,
        _username: &str,
    ) -> Result<ExternalIdentityResolution> {
        anyhow::bail!("PGlite external IdP repository implementation is pending")
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
    async fn list_users(&self) -> Result<Vec<String>> {
        Ok(self
            .database
            .query("SELECT username FROM users ORDER BY username", &[])
            .await?
            .iter()
            .map(|row| row.get::<String>(0).map_err(Into::into))
            .collect::<Result<Vec<_>>>()?)
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
