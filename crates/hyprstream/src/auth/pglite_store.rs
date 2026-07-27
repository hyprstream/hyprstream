//! Relational [`UserStore`] on the shared #1351 embedded PostgreSQL handle.
//!
//! # Architecture
//!
//! [`PgliteUserStore`] is the embedded backend; the SQL in
//! [`USERSTORE_SCHEMA`] is the authoritative, PostgreSQL-compatible DDL that a
//! future `PostgresUserStore` (server / metal deployment, #1378) consumes
//! against its own connection pool. PGlite and Postgres share one schema so
//! relational identities are portable between local and server deployments.
//!
//! The shared `Arc<PGlite>` handle (#1351) is injected via
//! [`PgliteUserStore::from_database`]; AppView inventory and credential
//! records then share one embedded database. Opening a second PGlite handle
//! against the same directory is unsafe — there is exactly one connection
//! owner, and #1378's server-side wiring must not create a competing one.
//!
//! # #1370 R4 hardening preserved
//!
//! Cold-signup staging (`provision_hosted_account`) and activation
//! (`activate_hosted_account`) implement the exact #1370 R4 contract also
//! implemented by [`RocksDbUserStore`](super::RocksDbUserStore) and
//! [`ValkeyUserStore`](super::valkey::ValkeyUserStore): orphan-genesis
//! ordering (inactive stage → PDS genesis → exact activation), exact
//! fingerprint recompute, full PQ-metadata validation, and the trustworthy
//! resume-vs-409 classification. Partial/corrupt state is a backend error
//! and fails closed; only a complete, usable, active hosted binding yields
//! `AccountAlreadyExists` / `KeyAlreadyBound`.

use super::{
    user_store::matches_filter, AccountKeyCustody, ExternalIdentityBinding,
    ExternalIdentityResolution, HostedAccountProvisionError, HostedAccountProvisioning,
    KeyAlgorithm, PubkeyEntry, UserFilter, UserProfile, UserProfilePatch, UserStore,
};
use anyhow::{anyhow, bail, ensure, Context, Result};
use async_trait::async_trait;
use ed25519_dalek::VerifyingKey;
use pglite::{PGlite, Row};
use std::{path::Path, sync::Arc};

/// PostgreSQL-compatible schema shared by embedded PGlite and server Postgres.
///
/// Lookup keys stay queryable. PII, labels, and key material use `BYTEA` so
/// envelope encryption can be added without changing relational identities.
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

const PROFILE_SELECT: &str = r#"
SELECT u.sub, u.name, u.email, u.email_verified, u.active, u.external_id,
       u.key_custody, d.atproto_did
FROM users u
LEFT JOIN user_did_bindings d ON d.username = u.username
WHERE u.username = $1
"#;

const KEY_SELECT: &str = r#"
SELECT fingerprint, pubkey, label, created_at, last_used_at, algorithm, pq_pubkey
FROM pubkeys WHERE username = $1 ORDER BY fingerprint
"#;

#[derive(Clone)]
pub struct PgliteUserStore {
    database: Arc<PGlite>,
}

impl PgliteUserStore {
    /// Standalone substrate factory. Application wiring should normally open
    /// PGlite once and call [`Self::from_database`] for every repository.
    pub async fn open(data_dir: impl AsRef<Path>) -> Result<Self> {
        let database = Arc::new(PGlite::open(data_dir).await.context("opening PGlite")?);
        Self::from_database(database).await
    }

    pub async fn from_database(database: Arc<PGlite>) -> Result<Self> {
        database
            .exec(USERSTORE_SCHEMA)
            .await
            .context("applying UserStore schema")?;
        Ok(Self { database })
    }

    pub fn database(&self) -> Arc<PGlite> {
        Arc::clone(&self.database)
    }

    async fn external_identities(&self, username: &str) -> Result<Vec<ExternalIdentityBinding>> {
        let rows = self
            .database
            .query(
                "SELECT issuer, issuer_sub FROM oidc_bindings \
                 WHERE username=$1 ORDER BY issuer, issuer_sub",
                &[&username],
            )
            .await?;
        rows.iter().map(decode_external_identity).collect()
    }
}

fn bytes_to_string(value: Option<Vec<u8>>, field: &str) -> Result<Option<String>> {
    value
        .map(|bytes| String::from_utf8(bytes).with_context(|| format!("decoding {field}")))
        .transpose()
}

fn string_to_bytes(value: Option<String>) -> Option<Vec<u8>> {
    value.map(String::into_bytes)
}

fn decode_external_identity(row: &Row) -> Result<ExternalIdentityBinding> {
    Ok(ExternalIdentityBinding {
        issuer: row.get(0).context("decoding external identity issuer")?,
        subject: row.get(1).context("decoding external identity subject")?,
    })
}

fn decode_profile_row(
    row: &Row,
    external_identities: Vec<ExternalIdentityBinding>,
) -> Result<UserProfile> {
    let custody = row
        .get::<Option<String>>(6)
        .context("decoding key custody")?
        .map(|value| AccountKeyCustody::parse(&value))
        .transpose()?;
    Ok(UserProfile {
        sub: Some(row.get(0).context("decoding stable subject")?),
        name: bytes_to_string(row.get(1).context("decoding name")?, "name")?,
        email: bytes_to_string(row.get(2).context("decoding email")?, "email")?,
        email_verified: row.get(3).context("decoding email verification")?,
        active: Some(row.get(4).context("decoding active state")?),
        external_id: bytes_to_string(row.get(5).context("decoding external ID")?, "external ID")?,
        atproto_did: row.get(7).context("decoding ATProto DID")?,
        key_custody: custody,
        external_identities,
    })
}

fn decode_key_row(row: &Row) -> Result<PubkeyEntry> {
    let fingerprint: String = row.get(0).context("decoding fingerprint")?;
    let raw: Vec<u8> = row.get(1).context("decoding Ed25519 key")?;
    let raw: [u8; 32] = raw
        .try_into()
        .map_err(|_| anyhow!("stored Ed25519 key {fingerprint} is not 32 bytes"))?;
    let pubkey = VerifyingKey::from_bytes(&raw)
        .with_context(|| format!("decoding Ed25519 key {fingerprint}"))?;
    ensure!(
        super::pubkey_fingerprint(&pubkey) == fingerprint,
        "stored key does not match fingerprint {fingerprint}"
    );
    let algorithm_raw: String = row.get(5).context("decoding key algorithm")?;
    let algorithm = KeyAlgorithm::parse(&algorithm_raw)?;
    let pq_pubkey: Option<Vec<u8>> = row.get(6).context("decoding ML-DSA-65 key")?;
    match (algorithm.is_hybrid(), pq_pubkey.as_ref()) {
        (false, None) => {}
        (true, Some(key)) if key.len() == 1952 => {}
        (true, _) => bail!("hybrid key {fingerprint} has invalid ML-DSA-65 material"),
        (false, Some(_)) => bail!("classical key {fingerprint} carries ML-DSA-65 material"),
    }
    Ok(PubkeyEntry {
        fingerprint,
        pubkey,
        label: bytes_to_string(row.get(2).context("decoding key label")?, "key label")?,
        created_at: row.get(3).context("decoding key creation time")?,
        last_used_at: row.get(4).context("decoding key last-used time")?,
        algorithm,
        pq_pubkey,
    })
}

fn hosted_backend(error: impl Into<anyhow::Error>) -> HostedAccountProvisionError {
    HostedAccountProvisionError::Backend(error.into())
}

#[async_trait]
impl UserStore for PgliteUserStore {
    async fn get_profile(&self, username: &str) -> Result<Option<UserProfile>> {
        let rows = self.database.query(PROFILE_SELECT, &[&username]).await?;
        ensure!(rows.len() <= 1, "username primary key returned duplicates");
        let Some(row) = rows.first() else {
            return Ok(None);
        };
        let bindings = self.external_identities(username).await?;
        Ok(Some(decode_profile_row(row, bindings)?))
    }

    async fn register(&self, username: &str) -> Result<String> {
        ensure!(!username.is_empty(), "username must be non-empty");
        let sub = uuid::Uuid::new_v4().to_string();
        self.database
            .query(
                "INSERT INTO users(username, sub, active) VALUES($1, $2, TRUE)",
                &[&username, &sub],
            )
            .await?;
        Ok(sub)
    }

    async fn resolve_or_bind_external_idp(
        &self,
        issuer: &str,
        subject: &str,
        username: &str,
    ) -> Result<ExternalIdentityResolution> {
        ensure!(
            !issuer.is_empty() && !subject.is_empty() && !username.is_empty(),
            "issuer, subject, and username must be non-empty"
        );
        let tx = self.database.transaction().await?;
        let existing = tx
            .query(
                "SELECT b.username, u.sub FROM oidc_bindings b \
                 JOIN users u ON u.username=b.username \
                 WHERE b.issuer=$1 AND b.issuer_sub=$2",
                &[&issuer, &subject],
            )
            .await?;
        ensure!(
            existing.len() <= 1,
            "external identity primary key returned duplicates"
        );
        if let Some(row) = existing.first() {
            let resolved_username: String = row.get(0)?;
            let sub: String = row.get(1)?;
            ensure!(
                !resolved_username.is_empty() && !sub.is_empty(),
                "external identity points at a corrupt local user"
            );
            tx.commit().await?;
            return Ok(ExternalIdentityResolution {
                username: resolved_username,
                sub,
                provisioned: false,
            });
        }

        let users = tx
            .query("SELECT sub FROM users WHERE username=$1", &[&username])
            .await?;
        ensure!(users.len() <= 1, "username primary key returned duplicates");
        let (sub, provisioned) = if let Some(row) = users.first() {
            let sub: String = row.get(0)?;
            ensure!(
                !sub.is_empty(),
                "candidate local user has no stable subject"
            );
            (sub, false)
        } else {
            let sub = uuid::Uuid::new_v4().to_string();
            tx.query(
                "INSERT INTO users(username, sub, active) VALUES($1, $2, TRUE)",
                &[&username, &sub],
            )
            .await?;
            (sub, true)
        };
        tx.query(
            "INSERT INTO oidc_bindings(issuer, issuer_sub, username) VALUES($1, $2, $3)",
            &[&issuer, &subject, &username],
        )
        .await?;
        tx.commit().await?;
        Ok(ExternalIdentityResolution {
            username: username.to_owned(),
            sub,
            provisioned,
        })
    }

    async fn provision_hosted_account(
        &self,
        username: &str,
        atproto_did: &str,
        pubkey: VerifyingKey,
        custody: AccountKeyCustody,
    ) -> std::result::Result<HostedAccountProvisioning, HostedAccountProvisionError> {
        if username.is_empty() || atproto_did.is_empty() {
            return Err(hosted_backend(anyhow!(
                "hosted username and DID must be non-empty"
            )));
        }
        let fingerprint = super::pubkey_fingerprint(&pubkey);
        let tx = self.database.transaction().await.map_err(hosted_backend)?;

        let profiles = tx
            .query(PROFILE_SELECT, &[&username])
            .await
            .map_err(hosted_backend)?;
        if let Some(profile_row) = profiles.first() {
            if profiles.len() != 1 {
                return Err(hosted_backend(anyhow!(
                    "username primary key returned duplicates"
                )));
            }
            let bindings = tx
                .query(
                    "SELECT issuer, issuer_sub FROM oidc_bindings WHERE username=$1",
                    &[&username],
                )
                .await
                .map_err(hosted_backend)?;
            let bindings = bindings
                .iter()
                .map(decode_external_identity)
                .collect::<Result<Vec<_>>>()
                .map_err(hosted_backend)?;
            let profile = decode_profile_row(profile_row, bindings).map_err(hosted_backend)?;
            let key_rows = tx
                .query(KEY_SELECT, &[&username])
                .await
                .map_err(hosted_backend)?;
            let keys = key_rows
                .iter()
                .map(decode_key_row)
                .collect::<Result<Vec<_>>>()
                .map_err(hosted_backend)?;
            let exact_key = keys.iter().any(|key| {
                key.fingerprint == fingerprint
                    && key.pubkey.as_bytes() == pubkey.as_bytes()
                    && key.algorithm == KeyAlgorithm::Ed25519
                    && key.pq_pubkey.is_none()
            });
            let exact = profile.sub.as_deref().is_some_and(|sub| !sub.is_empty())
                && profile.active.is_some()
                && profile.atproto_did.as_deref() == Some(atproto_did)
                && profile.key_custody == Some(custody)
                && profile.external_identities.is_empty()
                && exact_key;
            if exact {
                let sub = profile.sub.expect("checked above");
                tx.commit().await.map_err(hosted_backend)?;
                return Ok(HostedAccountProvisioning {
                    sub,
                    fingerprint,
                    resumed: true,
                });
            }
            let usable = profile.active == Some(true)
                && profile.sub.as_deref().is_some_and(|sub| !sub.is_empty())
                && profile
                    .atproto_did
                    .as_deref()
                    .is_some_and(|did| did.starts_with("did:web:"))
                && profile.key_custody.is_some()
                && !keys.is_empty();
            if usable {
                tx.commit().await.map_err(hosted_backend)?;
                return Err(HostedAccountProvisionError::AccountAlreadyExists);
            }
            return Err(hosted_backend(anyhow!(
                "existing hosted-account username has incomplete or inactive state"
            )));
        }

        let owner_rows = tx
            .query(
                "SELECT username FROM pubkeys WHERE fingerprint=$1",
                &[&fingerprint],
            )
            .await
            .map_err(hosted_backend)?;
        if let Some(owner_row) = owner_rows.first() {
            if owner_rows.len() != 1 {
                return Err(hosted_backend(anyhow!(
                    "fingerprint primary key returned duplicates"
                )));
            }
            let owner: String = owner_row.get(0).map_err(hosted_backend)?;
            let owner_profiles = tx
                .query(PROFILE_SELECT, &[&owner])
                .await
                .map_err(hosted_backend)?;
            let owner_keys = tx
                .query(KEY_SELECT, &[&owner])
                .await
                .map_err(hosted_backend)?;
            let keys = owner_keys
                .iter()
                .map(decode_key_row)
                .collect::<Result<Vec<_>>>()
                .map_err(hosted_backend)?;
            let profile = owner_profiles
                .first()
                .ok_or_else(|| hosted_backend(anyhow!("key owner profile is missing")))
                .and_then(|row| decode_profile_row(row, Vec::new()).map_err(hosted_backend))?;
            let exact_usable_key = keys.iter().any(|key| key.fingerprint == fingerprint);
            let usable = profile.active == Some(true)
                && profile.sub.as_deref().is_some_and(|sub| !sub.is_empty())
                && profile
                    .atproto_did
                    .as_deref()
                    .is_some_and(|did| did.starts_with("did:web:"))
                && profile.key_custody.is_some()
                && exact_usable_key;
            if usable {
                tx.commit().await.map_err(hosted_backend)?;
                return Err(HostedAccountProvisionError::KeyAlreadyBound);
            }
            return Err(hosted_backend(anyhow!(
                "existing hosted-account key owner is incomplete or inactive"
            )));
        }

        let sub = uuid::Uuid::new_v4().to_string();
        let now = chrono::Utc::now().timestamp();
        tx.query(
            "INSERT INTO users(username, sub, active, key_custody) \
             VALUES($1, $2, FALSE, $3)",
            &[&username, &sub, &custody.as_str()],
        )
        .await
        .map_err(hosted_backend)?;
        tx.query(
            "INSERT INTO user_did_bindings(username, atproto_did) VALUES($1, $2)",
            &[&username, &atproto_did],
        )
        .await
        .map_err(hosted_backend)?;
        tx.query(
            "INSERT INTO pubkeys(fingerprint, username, pubkey, label, algorithm, \
             pq_pubkey, created_at) VALUES($1, $2, $3, $4, 'ed25519', NULL, $5)",
            &[
                &fingerprint,
                &username,
                &pubkey.as_bytes().as_slice(),
                &Some(b"aegis-vault".to_vec()),
                &now,
            ],
        )
        .await
        .map_err(hosted_backend)?;
        tx.commit().await.map_err(hosted_backend)?;
        Ok(HostedAccountProvisioning {
            sub,
            fingerprint,
            resumed: false,
        })
    }

    async fn activate_hosted_account(
        &self,
        username: &str,
        atproto_did: &str,
        fingerprint: &str,
        custody: AccountKeyCustody,
    ) -> std::result::Result<(), HostedAccountProvisionError> {
        let tx = self.database.transaction().await.map_err(hosted_backend)?;
        let rows = tx
            .query(PROFILE_SELECT, &[&username])
            .await
            .map_err(hosted_backend)?;
        let profile = rows
            .first()
            .ok_or_else(|| hosted_backend(anyhow!("staged hosted account is missing")))
            .and_then(|row| decode_profile_row(row, Vec::new()).map_err(hosted_backend))?;
        let keys = tx
            .query(KEY_SELECT, &[&username])
            .await
            .map_err(hosted_backend)?;
        let keys = keys
            .iter()
            .map(decode_key_row)
            .collect::<Result<Vec<_>>>()
            .map_err(hosted_backend)?;
        let exact_key = keys.iter().any(|key| {
            key.fingerprint == fingerprint
                && key.algorithm == KeyAlgorithm::Ed25519
                && key.pq_pubkey.is_none()
        });
        if profile.sub.as_deref().is_none_or(str::is_empty)
            || profile.atproto_did.as_deref() != Some(atproto_did)
            || profile.key_custody != Some(custody)
            || !exact_key
        {
            return Err(hosted_backend(anyhow!(
                "staged hosted-account binding changed before activation"
            )));
        }
        match profile.active {
            Some(true) => {
                tx.commit().await.map_err(hosted_backend)?;
                Ok(())
            }
            Some(false) => {
                let updated = tx
                    .query(
                        "UPDATE users SET active=TRUE, updated_at=now() \
                         WHERE username=$1 AND active=FALSE RETURNING username",
                        &[&username],
                    )
                    .await
                    .map_err(hosted_backend)?;
                if updated.len() != 1 {
                    return Err(hosted_backend(anyhow!(
                        "staged hosted-account activation updated no exact row"
                    )));
                }
                tx.commit().await.map_err(hosted_backend)
            }
            None => Err(hosted_backend(anyhow!(
                "staged hosted-account binding has invalid activation state"
            ))),
        }
    }

    async fn set_profile(&self, username: &str, patch: UserProfilePatch) -> Result<()> {
        ensure!(
            patch.external_identities.is_none(),
            "set_profile cannot modify normalized external identity bindings"
        );
        let tx = self.database.transaction().await?;
        let rows = tx.query(PROFILE_SELECT, &[&username]).await?;
        ensure!(rows.len() <= 1, "username primary key returned duplicates");
        let row = rows.first().context("unknown user")?;
        let mut profile = decode_profile_row(row, Vec::new())?;
        if let Some(Some(sub)) = patch.sub {
            ensure!(!sub.is_empty(), "stable subject must be non-empty");
            profile.sub = Some(sub);
        }
        if let Some(value) = patch.name {
            profile.name = value;
        }
        if let Some(value) = patch.email {
            profile.email = value;
        }
        if let Some(value) = patch.email_verified {
            profile.email_verified = value;
        }
        if let Some(value) = patch.active {
            profile.active = Some(value.unwrap_or(true));
        }
        if let Some(value) = patch.external_id {
            profile.external_id = value;
        }
        let did_update = patch.atproto_did;
        if let Some(value) = did_update.as_ref() {
            profile.atproto_did = value.clone();
        }
        if let Some(value) = patch.key_custody {
            profile.key_custody = value;
        }
        let sub = profile.sub.context("user has no stable subject")?;
        let name = string_to_bytes(profile.name);
        let email = string_to_bytes(profile.email);
        let external_id = string_to_bytes(profile.external_id);
        let custody = profile.key_custody.map(AccountKeyCustody::as_str);
        tx.query(
            "UPDATE users SET sub=$2, name=$3, email=$4, email_verified=$5, \
             active=$6, external_id=$7, key_custody=$8, updated_at=now() \
             WHERE username=$1 RETURNING username",
            &[
                &username,
                &sub,
                &name,
                &email,
                &profile.email_verified,
                &profile.active.unwrap_or(true),
                &external_id,
                &custody,
            ],
        )
        .await?;
        if let Some(did) = did_update {
            tx.query(
                "DELETE FROM user_did_bindings WHERE username=$1",
                &[&username],
            )
            .await?;
            if let Some(did) = did {
                tx.query(
                    "INSERT INTO user_did_bindings(username, atproto_did) VALUES($1, $2)",
                    &[&username, &did],
                )
                .await?;
            }
        }
        tx.commit().await?;
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
        Ok(rows.len() == 1)
    }

    async fn list_users(&self) -> Result<Vec<String>> {
        self.database
            .query("SELECT username FROM users ORDER BY username", &[])
            .await?
            .iter()
            .map(|row| row.get(0).map_err(Into::into))
            .collect()
    }

    async fn search(&self, filter: &UserFilter) -> Result<Vec<(String, UserProfile)>> {
        let mut results = Vec::new();
        for username in self.list_users().await? {
            let profile = self
                .get_profile(&username)
                .await?
                .context("listed user profile is missing")?;
            if filter.active_only == Some(true) && profile.active == Some(false) {
                continue;
            }
            if filter.filter.as_ref().is_some_and(|expr| {
                !matches_filter(
                    expr,
                    &username,
                    &profile.sub,
                    &profile.external_id,
                    profile.active,
                )
            }) {
                continue;
            }
            results.push((username, profile));
        }
        if let Some(sort_by) = filter.sort_by.as_deref() {
            let descending = filter.sort_order.as_deref() == Some("descending");
            results.sort_by(|left, right| {
                let ordering = match sort_by {
                    "userName" => left.0.cmp(&right.0),
                    "id" | "sub" => left.1.sub.cmp(&right.1.sub),
                    "active" => left.1.active.cmp(&right.1.active),
                    "displayName" | "name" => left.1.name.cmp(&right.1.name),
                    "externalId" => left.1.external_id.cmp(&right.1.external_id),
                    _ => std::cmp::Ordering::Equal,
                };
                if descending {
                    ordering.reverse()
                } else {
                    ordering
                }
            });
        }
        let start = filter.start_index.unwrap_or(1).saturating_sub(1);
        let count = filter.count.unwrap_or(100);
        Ok(results.into_iter().skip(start).take(count).collect())
    }

    async fn set_active(&self, username: &str, active: bool) -> Result<()> {
        let rows = self
            .database
            .query(
                "UPDATE users SET active=$2, updated_at=now() \
                 WHERE username=$1 RETURNING username",
                &[&username, &active],
            )
            .await?;
        ensure!(rows.len() == 1, "unknown user");
        Ok(())
    }

    async fn list_pubkeys(&self, username: &str) -> Result<Vec<PubkeyEntry>> {
        let exists = self
            .database
            .query(
                "SELECT 1::BIGINT FROM users WHERE username=$1",
                &[&username],
            )
            .await?;
        ensure!(exists.len() == 1, "unknown user");
        self.database
            .query(KEY_SELECT, &[&username])
            .await?
            .iter()
            .map(decode_key_row)
            .collect()
    }

    async fn add_pubkey(
        &self,
        username: &str,
        pubkey: VerifyingKey,
        label: Option<String>,
    ) -> Result<String> {
        let fingerprint = super::pubkey_fingerprint(&pubkey);
        let now = chrono::Utc::now().timestamp();
        let label = string_to_bytes(label);
        self.database
            .query(
                "INSERT INTO pubkeys(fingerprint, username, pubkey, label, algorithm, \
                 pq_pubkey, created_at) VALUES($1, $2, $3, $4, 'ed25519', NULL, $5)",
                &[
                    &fingerprint,
                    &username,
                    &pubkey.as_bytes().as_slice(),
                    &label,
                    &now,
                ],
            )
            .await?;
        Ok(fingerprint)
    }

    async fn add_pubkey_hybrid(
        &self,
        username: &str,
        pubkey: VerifyingKey,
        ml_dsa_vk: Vec<u8>,
        label: Option<String>,
    ) -> Result<String> {
        ensure!(
            ml_dsa_vk.len() == 1952,
            "ML-DSA-65 verifying key must be exactly 1952 bytes"
        );
        let fingerprint = super::pubkey_fingerprint(&pubkey);
        let tx = self.database.transaction().await?;
        let existing = tx
            .query(
                "SELECT username, pubkey, algorithm, pq_pubkey FROM pubkeys \
                 WHERE fingerprint=$1",
                &[&fingerprint],
            )
            .await?;
        if let Some(row) = existing.first() {
            let owner: String = row.get(0)?;
            let stored_key: Vec<u8> = row.get(1)?;
            let algorithm: String = row.get(2)?;
            let pq: Option<Vec<u8>> = row.get(3)?;
            ensure!(owner == username, "pubkey is already bound to another user");
            ensure!(
                stored_key.as_slice() == pubkey.as_bytes(),
                "fingerprint row carries different Ed25519 bytes"
            );
            ensure!(
                algorithm == "ed25519" && pq.is_none(),
                "only a classical key can be upgraded to hybrid"
            );
            let label = string_to_bytes(label);
            tx.query(
                "UPDATE pubkeys SET algorithm='ed25519+ml-dsa-65', pq_pubkey=$2, \
                 label=COALESCE($3, label) WHERE fingerprint=$1 RETURNING fingerprint",
                &[&fingerprint, &ml_dsa_vk, &label],
            )
            .await?;
        } else {
            let now = chrono::Utc::now().timestamp();
            let label = string_to_bytes(label);
            tx.query(
                "INSERT INTO pubkeys(fingerprint, username, pubkey, label, algorithm, \
                 pq_pubkey, created_at) VALUES($1, $2, $3, $4, \
                 'ed25519+ml-dsa-65', $5, $6)",
                &[
                    &fingerprint,
                    &username,
                    &pubkey.as_bytes().as_slice(),
                    &label,
                    &ml_dsa_vk,
                    &now,
                ],
            )
            .await?;
        }
        tx.commit().await?;
        Ok(fingerprint)
    }

    async fn remove_pubkey(&self, username: &str, fingerprint: &str) -> Result<bool> {
        let rows = self
            .database
            .query(
                "DELETE FROM pubkeys WHERE username=$1 AND fingerprint=$2 \
                 RETURNING fingerprint",
                &[&username, &fingerprint],
            )
            .await?;
        Ok(rows.len() == 1)
    }

    async fn get_pubkey_user(&self, fingerprint: &str) -> Result<Option<String>> {
        let rows = self
            .database
            .query(
                "SELECT username FROM pubkeys WHERE fingerprint=$1",
                &[&fingerprint],
            )
            .await?;
        ensure!(
            rows.len() <= 1,
            "fingerprint primary key returned duplicates"
        );
        rows.first()
            .map(|row| row.get(0).map_err(Into::into))
            .transpose()
    }

    async fn touch_pubkey(&self, username: &str, fingerprint: &str) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        let rows = self
            .database
            .query(
                "UPDATE pubkeys SET last_used_at=$3 WHERE username=$1 AND fingerprint=$2 \
                 RETURNING fingerprint",
                &[&username, &fingerprint, &now],
            )
            .await?;
        ensure!(rows.len() == 1, "unknown user or fingerprint");
        Ok(())
    }

    async fn list_external_identities(
        &self,
        username: &str,
    ) -> Result<Vec<ExternalIdentityBinding>> {
        ensure!(self.get_profile(username).await?.is_some(), "unknown user");
        self.external_identities(username).await
    }

    async fn get_external_identity_user(
        &self,
        issuer: &str,
        subject: &str,
    ) -> Result<Option<String>> {
        ensure!(
            !issuer.is_empty() && !subject.is_empty(),
            "external identity issuer and subject must be non-empty"
        );
        let rows = self
            .database
            .query(
                "SELECT b.username FROM oidc_bindings b \
                 JOIN users u ON u.username=b.username \
                 WHERE b.issuer=$1 AND b.issuer_sub=$2",
                &[&issuer, &subject],
            )
            .await?;
        ensure!(
            rows.len() <= 1,
            "external identity primary key returned duplicates"
        );
        rows.first()
            .map(|row| row.get(0).map_err(Into::into))
            .transpose()
    }
}

#[cfg(all(test, feature = "pglite"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    use std::sync::OnceLock;

    /// PGlite is a process singleton — it cannot be reopened after close in
    /// the same process. All tests share this one instance, serialized by a
    /// mutex so parallel `cargo test` is safe without `--test-threads=1`.
    static SHARED_DB: OnceLock<PgliteUserStore> = OnceLock::new();
    static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    async fn shared() -> &'static PgliteUserStore {
        if let Some(s) = SHARED_DB.get() {
            return s;
        }
        let dir = tempfile::tempdir_in(std::env::current_dir().unwrap()).unwrap();
        let path = dir.path().to_owned();
        std::mem::forget(dir); // keep the data dir alive for the process lifetime
        let s = PgliteUserStore::open(&path).await.unwrap();
        let _ = SHARED_DB.set(s);
        SHARED_DB.get().unwrap()
    }

    /// Acquire the test serialization lock AND wipe all rows. The returned
    /// guard must be held (`_guard`) for the entire test body so no two
    /// tests touch the shared database concurrently.
    async fn fresh() -> tokio::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().await;
        let s = shared().await;
        s.database.query("DELETE FROM users", &[]).await.unwrap();
        guard
    }

    fn make_key() -> VerifyingKey {
        SigningKey::generate(&mut OsRng).verifying_key()
    }

    // ── Basic CRUD ───────────────────────────────────────────────────────

    #[tokio::test]
    async fn register_get_profile_round_trip() {
        let _guard = fresh().await;
    let store = shared().await;
        let sub = store.register("alice").await.unwrap();
        let profile = store.get_profile("alice").await.unwrap().unwrap();
        assert_eq!(profile.sub.as_deref(), Some(sub.as_str()));
        assert_eq!(profile.active, Some(true));
        assert!(profile.external_identities.is_empty());
    }

    #[tokio::test]
    async fn get_profile_missing_user_returns_none() {
        let _guard = fresh().await;
    let store = shared().await;
        assert!(store.get_profile("nobody").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn register_empty_username_fails() {
        let _guard = fresh().await;
    let store = shared().await;
        assert!(store.register("").await.is_err());
    }

    #[tokio::test]
    async fn set_profile_updates_columns() {
        let _guard = fresh().await;
    let store = shared().await;
        store.register("alice").await.unwrap();
        store
            .set_profile(
                "alice",
                UserProfilePatch {
                    name: Some(Some("Alice".to_owned())),
                    email: Some(Some("alice@example.com".to_owned())),
                    email_verified: Some(Some(true)),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        let profile = store.get_profile("alice").await.unwrap().unwrap();
        assert_eq!(profile.name.as_deref(), Some("Alice"));
        assert_eq!(profile.email.as_deref(), Some("alice@example.com"));
        assert_eq!(profile.email_verified, Some(true));
    }

    #[tokio::test]
    async fn set_profile_atproto_did_clears_binding() {
        let _guard = fresh().await;
    let store = shared().await;
        store.register("alice").await.unwrap();
        store
            .set_profile(
                "alice",
                UserProfilePatch {
                    atproto_did: Some(Some("did:web:alice.example".to_owned())),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            store.get_profile("alice").await.unwrap().unwrap().atproto_did,
            Some("did:web:alice.example".to_owned())
        );
        // Clear the DID.
        store
            .set_profile(
                "alice",
                UserProfilePatch {
                    atproto_did: Some(None),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(
            store.get_profile("alice").await.unwrap().unwrap().atproto_did,
            None
        );
    }

    #[tokio::test]
    async fn set_profile_unknown_user_fails() {
        let _guard = fresh().await;
    let store = shared().await;
        assert!(store
            .set_profile("nobody", UserProfilePatch::default())
            .await
            .is_err());
    }

    #[tokio::test]
    async fn remove_cascades_pubkeys_and_did_binding() {
        let _guard = fresh().await;
    let store = shared().await;
        store.register("alice").await.unwrap();
        let key = make_key();
        let fp = store.add_pubkey("alice", key, None).await.unwrap();
        store
            .set_profile(
                "alice",
                UserProfilePatch {
                    atproto_did: Some(Some("did:web:alice.example".to_owned())),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert!(store.remove("alice").await.unwrap());
        assert!(store.get_profile("alice").await.unwrap().is_none());
        assert!(store.get_pubkey_user(&fp).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn set_active_toggles() {
        let _guard = fresh().await;
    let store = shared().await;
        store.register("alice").await.unwrap();
        store.set_active("alice", false).await.unwrap();
        assert_eq!(
            store.get_profile("alice").await.unwrap().unwrap().active,
            Some(false)
        );
        store.set_active("alice", true).await.unwrap();
        assert_eq!(
            store.get_profile("alice").await.unwrap().unwrap().active,
            Some(true)
        );
    }

    // ── Search ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn search_filters_and_paginates() {
        let _guard = fresh().await;
    let store = shared().await;
        store.register("alice").await.unwrap();
        store.register("bob").await.unwrap();
        store.register("carol").await.unwrap();
        store.set_active("bob", false).await.unwrap();

        let active = store
            .search(&UserFilter {
                active_only: Some(true),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(active.len(), 2);

        let filtered = store
            .search(&UserFilter {
                filter: Some(r#"userName eq "alice""#.to_owned()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].0, "alice");

        let paged = store
            .search(&UserFilter {
                count: Some(1),
                start_index: Some(2),
                sort_by: Some("userName".to_owned()),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(paged.len(), 1);
        assert_eq!(paged[0].0, "bob");
    }

    // ── Pubkey operations ────────────────────────────────────────────────

    #[tokio::test]
    async fn add_list_remove_pubkey() {
        let _guard = fresh().await;
    let store = shared().await;
        store.register("alice").await.unwrap();
        let key = make_key();
        let fp = store
            .add_pubkey("alice", key, Some("laptop".to_owned()))
            .await
            .unwrap();
        let keys = store.list_pubkeys("alice").await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].fingerprint, fp);
        assert_eq!(keys[0].label.as_deref(), Some("laptop"));
        assert_eq!(keys[0].algorithm, KeyAlgorithm::Ed25519);
        assert!(keys[0].pq_pubkey.is_none());

        assert_eq!(
            store.get_pubkey_user(&fp).await.unwrap().as_deref(),
            Some("alice")
        );

        store.touch_pubkey("alice", &fp).await.unwrap();
        let keys = store.list_pubkeys("alice").await.unwrap();
        assert!(keys[0].last_used_at.is_some());

        assert!(store.remove_pubkey("alice", &fp).await.unwrap());
        assert!(store.list_pubkeys("alice").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn add_pubkey_hybrid_upgrades_classical() {
        let _guard = fresh().await;
    let store = shared().await;
        store.register("alice").await.unwrap();
        let key = make_key();
        let fp = store.add_pubkey("alice", key, None).await.unwrap();

        let pq_vk = vec![0u8; 1952];
        store
            .add_pubkey_hybrid("alice", key, pq_vk, Some("hybrid".to_owned()))
            .await
            .unwrap();
        let keys = store.list_pubkeys("alice").await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].fingerprint, fp);
        assert_eq!(keys[0].algorithm, KeyAlgorithm::HybridEd25519MlDsa65);
        assert!(keys[0].pq_pubkey.is_some());
        assert_eq!(keys[0].pq_pubkey.as_ref().unwrap().len(), 1952);
    }

    #[tokio::test]
    async fn add_pubkey_hybrid_wrong_length_fails() {
        let _guard = fresh().await;
    let store = shared().await;
        store.register("alice").await.unwrap();
        let key = make_key();
        assert!(store
            .add_pubkey_hybrid("alice", key, vec![0u8; 100], None)
            .await
            .is_err());
    }

    // ── External identity ────────────────────────────────────────────────

    #[tokio::test]
    async fn resolve_or_bind_creates_and_resolves() {
        let _guard = fresh().await;
    let store = shared().await;
        let res1 = store
            .resolve_or_bind_external_idp("https://idp.example", "sub-123", "alice")
            .await
            .unwrap();
        assert!(res1.provisioned);
        assert_eq!(res1.username, "alice");

        // Same (issuer, subject) resolves to the same user even with a
        // different candidate username.
        let res2 = store
            .resolve_or_bind_external_idp("https://idp.example", "sub-123", "bob")
            .await
            .unwrap();
        assert!(!res2.provisioned);
        assert_eq!(res2.username, "alice");
        assert_eq!(res2.sub, res1.sub);
    }

    #[tokio::test]
    async fn list_and_get_external_identities() {
        let _guard = fresh().await;
    let store = shared().await;
        store
            .resolve_or_bind_external_idp("https://idp.example", "sub-1", "alice")
            .await
            .unwrap();
        let bindings = store.list_external_identities("alice").await.unwrap();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].issuer, "https://idp.example");
        assert_eq!(bindings[0].subject, "sub-1");

        let user = store
            .get_external_identity_user("https://idp.example", "sub-1")
            .await
            .unwrap();
        assert_eq!(user.as_deref(), Some("alice"));
    }

    // ── #1370 hosted-account provisioning (R4 hardening) ────────────────

    #[tokio::test]
    async fn provision_stages_inactive_account() {
        let _guard = fresh().await;
    let store = shared().await;
        let key = make_key();
        let result = store
            .provision_hosted_account(
                "alice",
                "did:web:alice.example",
                key,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
        assert!(!result.resumed);
        let profile = store.get_profile("alice").await.unwrap().unwrap();
        assert_eq!(profile.active, Some(false));
        assert_eq!(
            profile.atproto_did.as_deref(),
            Some("did:web:alice.example")
        );
        assert_eq!(profile.key_custody, Some(AccountKeyCustody::SelfCustody));
        let keys = store.list_pubkeys("alice").await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0].fingerprint, result.fingerprint);
    }

    #[tokio::test]
    async fn provision_exact_repeat_resumes() {
        let _guard = fresh().await;
    let store = shared().await;
        let key = make_key();
        let first = store
            .provision_hosted_account(
                "alice",
                "did:web:alice.example",
                key,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
        let second = store
            .provision_hosted_account(
                "alice",
                "did:web:alice.example",
                key,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
        assert!(second.resumed);
        assert_eq!(first.sub, second.sub);
        assert_eq!(first.fingerprint, second.fingerprint);
    }

    #[tokio::test]
    async fn activate_flips_staged_to_active() {
        let _guard = fresh().await;
    let store = shared().await;
        let key = make_key();
        let provisioned = store
            .provision_hosted_account(
                "alice",
                "did:web:alice.example",
                key,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
        store
            .activate_hosted_account(
                "alice",
                "did:web:alice.example",
                &provisioned.fingerprint,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
        let profile = store.get_profile("alice").await.unwrap().unwrap();
        assert_eq!(profile.active, Some(true));
    }

    #[tokio::test]
    async fn activate_idempotent_on_already_active() {
        let _guard = fresh().await;
    let store = shared().await;
        let key = make_key();
        let fp = super::super::pubkey_fingerprint(&key);
        store
            .provision_hosted_account(
                "alice",
                "did:web:alice.example",
                key,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
        store
            .activate_hosted_account(
                "alice",
                "did:web:alice.example",
                &fp,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
        // Second activation is Ok (idempotent).
        store
            .activate_hosted_account(
                "alice",
                "did:web:alice.example",
                &fp,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn activate_rejects_changed_did() {
        let _guard = fresh().await;
    let store = shared().await;
        let key = make_key();
        let fp = super::super::pubkey_fingerprint(&key);
        store
            .provision_hosted_account(
                "alice",
                "did:web:alice.example",
                key,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
        assert!(store
            .activate_hosted_account(
                "alice",
                "did:web:wrong.example",
                &fp,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn provision_account_already_exists() {
        let _guard = fresh().await;
    let store = shared().await;
        let key = make_key();
        // Stage and fully activate a hosted account.
        let provisioned = store
            .provision_hosted_account(
                "alice",
                "did:web:alice.example",
                key,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
        store
            .activate_hosted_account(
                "alice",
                "did:web:alice.example",
                &provisioned.fingerprint,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
        // A second provision with a DIFFERENT key must see AccountAlreadyExists.
        let other_key = make_key();
        let err = store
            .provision_hosted_account(
                "alice",
                "did:web:alice.example",
                other_key,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            HostedAccountProvisionError::AccountAlreadyExists
        ));
    }

    #[tokio::test]
    async fn provision_key_already_bound() {
        let _guard = fresh().await;
    let store = shared().await;
        let key = make_key();
        // Stage+activate alice with this key.
        let provisioned = store
            .provision_hosted_account(
                "alice",
                "did:web:alice.example",
                key,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
        store
            .activate_hosted_account(
                "alice",
                "did:web:alice.example",
                &provisioned.fingerprint,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
        // Provisioning bob with alice's key must see KeyAlreadyBound.
        let err = store
            .provision_hosted_account(
                "bob",
                "did:web:bob.example",
                key,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            HostedAccountProvisionError::KeyAlreadyBound
        ));
    }

    #[tokio::test]
    async fn provision_corrupt_inactive_state_is_backend_error() {
        let _guard = fresh().await;
    let store = shared().await;
        let key = make_key();
        store
            .provision_hosted_account(
                "alice",
                "did:web:alice.example",
                key,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap();
        // A second provision with a different key against the still-INACTIVE
        // staged account is corrupt/incomplete (not a trusted 409).
        let other_key = make_key();
        let err = store
            .provision_hosted_account(
                "alice",
                "did:web:alice.example",
                other_key,
                AccountKeyCustody::SelfCustody,
            )
            .await
            .unwrap_err();
        // Must be Backend, NOT AccountAlreadyExists (account is not active).
        assert!(matches!(
            err,
            HostedAccountProvisionError::Backend(_)
        ));
    }

    // ── Shared handle (#1351) ────────────────────────────────────────────

    #[tokio::test]
    async fn from_database_shares_one_handle() {
        let _guard = fresh().await;
        let s = shared().await;
        let db = s.database();
        let store2 = PgliteUserStore::from_database(Arc::clone(&db))
            .await
            .unwrap();
        // The handle returned by database() is the same Arc.
        assert!(Arc::ptr_eq(&store2.database(), &db));
    }
}
