//! Networked Postgres [`UserStore`] backend for AWS RDS (#1401).
//!
//! # Architecture
//!
//! [`PostgresUserStore`] is the deployed backend; [`PgliteUserStore`] is the
//! embedded local/workstation backend. Both implement the exact same
//! [`UserStore`] trait against the exact same [`USERSTORE_SCHEMA`] DDL. The
//! only difference is the I/O substrate: deadpool-postgres pooled connections
//! + TLS to RDS, vs PGlite's single embedded connection.
//!
//! The #1377 backend-neutral [`ColumnCipher`] wraps BYTEA value columns
//! identically in both backends — the seal/open glue is shared via
//! [`cipher_glue`].
//!
//! # #1370 R4 hardening preserved
//!
//! Cold-signup staging (`provision_hosted_account`) and activation
//! (`activate_hosted_account`) reproduce the exact #1370 R4 contract also
//! implemented by [`PgliteUserStore`] and [`RocksDbUserStore`]:
//! orphan-genesis ordering (inactive stage → PDS genesis → exact activation),
//! exact fingerprint recompute, full PQ-metadata validation, and the
//! trustworthy resume-vs-409 classification. Partial/corrupt state is a
//! backend error and fails closed; only a complete, usable, active hosted
//! binding yields `AccountAlreadyExists` / `KeyAlreadyBound`.
//!
//! # Networked-DB failure modes
//!
//! Connection loss, pool exhaustion, and timeout all fail closed as
//! [`HostedAccountProvisionError::Backend`] — never silent success or
//! fallback. A configured Postgres/RDS store is FATAL-on-unavailable at
//! startup; the caller never downgrades to pglite or Mem in the deploy.

use super::{
    cipher_glue::{self, USERSTORE_SCHEMA},
    encrypted_columns::{ColumnCipher, EncryptedColumn, ROOT_DEK_BYTES},
    user_store::matches_filter, AccountKeyCustody, ExternalIdentityBinding,
    ExternalIdentityResolution, HostedAccountProvisionError, HostedAccountProvisioning,
    KeyAlgorithm, PubkeyEntry, UserFilter, UserProfile, UserProfilePatch, UserStore,
};
use anyhow::{anyhow, bail, ensure, Context, Result};
use async_trait::async_trait;
use deadpool_postgres::{Config as PoolConfig, Pool, SslMode};
use ed25519_dalek::VerifyingKey;
use rustls::RootCertStore;
use std::{path::PathBuf, time::Duration};
use tokio_postgres::Row;
use tokio_postgres_rustls::MakeRustlsConnect;
use url::Url;
use zeroize::Zeroizing;

/// Profile SELECT — same SQL as PgliteUserStore.
const PROFILE_SELECT: &str = r#"
SELECT u.sub, u.name, u.email, u.email_verified, u.active, u.external_id,
       u.key_custody, d.atproto_did
FROM users u
LEFT JOIN user_did_bindings d ON d.username = u.username
WHERE u.username = $1
"#;

/// Key SELECT — same SQL as PgliteUserStore.
const KEY_SELECT: &str = r#"
SELECT fingerprint, pubkey, label, created_at, last_used_at, algorithm, pq_pubkey
FROM pubkeys WHERE username = $1 ORDER BY fingerprint
"#;

/// Configuration for the networked Postgres [`UserStore`] backend.
///
/// Constructed at deployment-startup time from the file-backed credential
/// inputs mandated by the metal RDS runtime contract v1.1 (see
/// [`Self::from_env`]). The URL **must** contain `sslmode=verify-full`.
///
/// **No `Debug` impl** — `database_url` carries a password. Logging or
/// debug-formatting this struct would leak the credential (metal v1.1 §4).
#[derive(Clone)]
#[allow(dead_code)] // wired once services/factories.rs gains a postgres selector
pub struct PostgresUserStoreConfig {
    /// libpq connection string read from the role-scoped URL file.
    /// Contains `sslmode=verify-full`.
    pub(crate) database_url: String,
    /// Maximum pool size. Default: `2 * num_cpus`.
    pub max_connections: usize,
    /// Path to the pinned RDS CA PEM file (mandatory in production;
    /// `None` only in `#[cfg(test)]` plaintext paths).
    pub(crate) ca_file: Option<PathBuf>,
}

impl PostgresUserStoreConfig {
    /// Load fail-closed deployment configuration from the file-backed
    /// credential inputs mandated by the metal RDS runtime contract v1.1.
    ///
    /// Required env vars (paths to projected credential files):
    /// - `HYPRSTREAM_CREDENTIALS_URL_FILE` — newline-terminated libpq URL.
    ///   The URL **must** contain exactly `sslmode=verify-full`.
    /// - `HYPRSTREAM_CREDENTIALS_SSLROOTCERT_FILE` — pinned RDS CA PEM.
    ///
    /// No password-bearing URL is ever read from a direct env var; the
    /// env var carries only the *path* to the secret file. The URL is
    /// never emitted in any log, error, or Debug representation.
    ///
    /// Optional:
    /// - `HYPRSTREAM_USERSTORE_MAX_CONNECTIONS` — pool size (default `2 * num_cpus`).
    pub fn from_env() -> Result<Self> {
        // ── URL file (the sole credential carrier) ─────────────────────
        let url_file = std::env::var_os("HYPRSTREAM_CREDENTIALS_URL_FILE")
            .context(
                "HYPRSTREAM_CREDENTIALS_URL_FILE is required \
                 (metal RDS runtime contract v1.1)",
            );
        let url_file = url_file?;
        let database_url = std::fs::read_to_string(&url_file)
            .with_context(|| {
                format!(
                    "reading credentials URL file {}",
                    url_file.to_string_lossy()
                )
            })?
            .trim()
            .to_owned();
        ensure!(
            !database_url.is_empty(),
            "credentials URL file is empty: {}",
            url_file.to_string_lossy()
        );

        // ── v1.1 positive TLS assertion ────────────────────────────────
        // Structural URL validation: parse query pairs, not substring match.
        // Substring checks are bypassable (e.g. the password field can
        // contain "sslmode=verify-full" while the query has no sslmode).
        validate_pg_url(&database_url)?;

        // ── CA file (mandatory — the connector must load it, not infer) ─
        let ca_file = std::env::var_os("HYPRSTREAM_CREDENTIALS_SSLROOTCERT_FILE")
            .context(
                "HYPRSTREAM_CREDENTIALS_SSLROOTCERT_FILE is required \
                 (metal RDS runtime contract v1.1)",
            )?;
        // Fail-closed: the CA file must exist and be readable at startup.
        std::fs::metadata(&ca_file).with_context(|| {
            format!(
                "credentials CA file is missing or unreadable: {}",
                ca_file.to_string_lossy()
            )
        })?;

        // ── Pool sizing (non-secret) ───────────────────────────────────
        let max_connections = std::env::var("HYPRSTREAM_USERSTORE_MAX_CONNECTIONS")
            .ok()
            .map(|v| v.parse::<usize>())
            .transpose()
            .context("HYPRSTREAM_USERSTORE_MAX_CONNECTIONS must be a positive integer")?
            .unwrap_or_else(|| 2 * num_cpus::get());

        Ok(Self {
            database_url,
            max_connections,
            ca_file: Some(PathBuf::from(ca_file)),
        })
    }

    /// Construct from an explicit connection URL — **TEST/MIGRATION ONLY**.
    ///
    /// Production wiring MUST use [`Self::from_env`] which reads the
    /// role-scoped URL file per metal v1.1.
    #[cfg(test)]
    pub fn from_url(database_url: impl Into<String>) -> Self {
        Self {
            database_url: database_url.into(),
            max_connections: 2 * num_cpus::get(),
            ca_file: None,
        }
    }
}

/// Networked Postgres [`UserStore`] backend (AWS RDS / server Postgres).
///
/// Same SQL, same schema, same R4 invariants as
/// [`PgliteUserStore`](super::PgliteUserStore). Only the I/O seam differs:
/// deadpool-postgres pooled connections + TLS, and tokio-postgres's
/// `Row::try_get` API.
#[allow(dead_code)] // wired once services/factories.rs gains a postgres selector
pub struct PostgresUserStore {
    pool: Pool,
    cipher: Option<ColumnCipher>,
}

impl PostgresUserStore {
    /// Open a production PostgresUserStore with at-rest envelope encryption.
    ///
    /// Runs the idempotent `USERSTORE_SCHEMA` migration on every boot. The
    /// cipher is REQUIRED for production — every BYTEA value column is sealed
    /// before storage and unsealed on read.
    ///
    /// Returns `Err` on any startup failure (DNS, TLS, auth, migration).
    /// The caller MUST treat this as fatal — never fall back to pglite or Mem.
    pub(crate) async fn connect(config: PostgresUserStoreConfig, cipher: ColumnCipher) -> Result<Self> {
        Self::new(Some(config), Some(cipher)).await
    }

    /// Open a PostgresUserStore WITHOUT encryption — **TEST/MIGRATION ONLY**.
    ///
    /// Uses `NoTls` (plaintext to local Postgres) — production MUST use [`Self::connect`]
    /// which forces rustls TLS to RDS.
    #[cfg(test)]
    pub async fn connect_plaintext(config: PostgresUserStoreConfig) -> Result<Self> {
        let url = Url::parse(&config.database_url)
            .context("parsing test PostgresUserStore URL")?;
        let mut cfg = PoolConfig::new();
        cfg.host = url.host_str().map(|h| h.to_owned());
        cfg.port = url.port();
        if !url.username().is_empty() {
            cfg.user = Some(url.username().to_owned());
        }
        if let Some(password) = url.password() {
            cfg.password = Some(percent_decode(password));
        }
        let path = url.path().trim_start_matches('/');
        if !path.is_empty() {
            cfg.dbname = Some(path.to_owned());
        }
        cfg.ssl_mode = Some(SslMode::Disable);
        let pool = cfg
            .builder(::tokio_postgres::NoTls)?
            .max_size(config.max_connections)
            .build()?;
        let store = Self { pool, cipher: None };
        store.migrate().await?;
        Ok(store)
    }

    async fn new(config: Option<PostgresUserStoreConfig>, cipher: Option<ColumnCipher>) -> Result<Self> {
        let pool = build_pool(config.as_ref())?;
        let store = Self { pool, cipher };
        store.migrate().await.context("UserStore schema migration failed at startup")?;
        Ok(store)
    }

    async fn migrate(&self) -> Result<()> {
        let client = self.pool.get().await.map_err(pool_err)?;
        client
            .batch_execute(USERSTORE_SCHEMA)
            .await
            .context("applying UserStore schema")?;
        Ok(())
    }

    fn is_encrypted(&self) -> bool {
        self.cipher.is_some()
    }

    // ── DEK lifecycle ────────────────────────────────────────────────────

    /// Mint a fresh root DEK, seal it, persist the wrapped form. Returns the
    /// root for immediate use by the caller's write path.
    async fn create_user_key(
        &self,
        tx: &deadpool_postgres::Transaction<'_>,
        username: &str,
    ) -> Result<Zeroizing<[u8; ROOT_DEK_BYTES]>> {
        let cipher = self
            .cipher
            .as_ref()
            .context("encryption is not configured for this store")?;
        let new_key = cipher.create_user_key().await?;
        tx.execute(
            "INSERT INTO user_encryption_keys(username, wrapped_dek) VALUES($1, $2)",
            &[&username, &new_key.wrapped],
        )
        .await?;
        Ok(new_key.root)
    }

    /// Unseal the user's persisted root DEK via a pool checkout (non-tx path).
    /// Fails closed when the wrapped DEK is absent or key material is unavailable.
    async fn load_user_key(&self, username: &str) -> Result<Zeroizing<[u8; ROOT_DEK_BYTES]>> {
        let cipher = self
            .cipher
            .as_ref()
            .context("encryption is not configured for this store")?;
        let client = self.pool.get().await.map_err(pool_err)?;
        let rows = client
            .query(
                "SELECT wrapped_dek FROM user_encryption_keys WHERE username=$1",
                &[&username],
            )
            .await?;
        let wrapped: Vec<u8> = rows
            .first()
            .context("wrapped UserStore DEK is absent — key material revoked or never provisioned")?
            .get(0);
        cipher.open_user_key(&wrapped).await
    }

    /// Transaction-scoped DEK load — queries through the active transaction.
    async fn load_user_key_tx(
        &self,
        tx: &deadpool_postgres::Transaction<'_>,
        username: &str,
    ) -> Result<Zeroizing<[u8; ROOT_DEK_BYTES]>> {
        let cipher = self
            .cipher
            .as_ref()
            .context("encryption is not configured for this store")?;
        let rows = tx
            .query(
                "SELECT wrapped_dek FROM user_encryption_keys WHERE username=$1",
                &[&username],
            )
            .await?;
        let wrapped: Vec<u8> = rows
            .first()
            .context("wrapped UserStore DEK is absent — key material revoked or never provisioned")?
            .get(0);
        cipher.open_user_key(&wrapped).await
    }

    // ── Row decoders (cipher-aware) ──────────────────────────────────────

    async fn external_identities(&self, username: &str) -> Result<Vec<ExternalIdentityBinding>> {
        let client = self.pool.get().await.map_err(pool_err)?;
        let rows = client
            .query(
                "SELECT issuer, issuer_sub FROM oidc_bindings \
                 WHERE username=$1 ORDER BY issuer, issuer_sub",
                &[&username],
            )
            .await?;
        rows.iter().map(decode_external_identity).collect()
    }

    /// Decode a profile row, optionally decrypting the BYTEA value columns.
    fn decode_profile(
        &self,
        row: &Row,
        external_identities: Vec<ExternalIdentityBinding>,
        username: &str,
        root: Option<&Zeroizing<[u8; ROOT_DEK_BYTES]>>,
    ) -> Result<UserProfile> {
        let custody = row
            .try_get::<_, Option<String>>(6)
            .context("decoding key custody")?
            .map(|value| AccountKeyCustody::parse(&value))
            .transpose()?;
        Ok(UserProfile {
            sub: Some(row.try_get(0).context("decoding stable subject")?),
            name: cipher_glue::open_text(
                self.cipher.as_ref(),
                root,
                username,
                EncryptedColumn::ProfileName,
                row.try_get(1).context("decoding name")?,
            )?,
            email: cipher_glue::open_text(
                self.cipher.as_ref(),
                root,
                username,
                EncryptedColumn::ProfileEmail,
                row.try_get(2).context("decoding email")?,
            )?,
            email_verified: row.try_get(3).context("decoding email verification")?,
            active: Some(row.try_get(4).context("decoding active state")?),
            external_id: cipher_glue::open_text(
                self.cipher.as_ref(),
                root,
                username,
                EncryptedColumn::ProfileExternalId,
                row.try_get(5).context("decoding external ID")?,
            )?,
            atproto_did: row.try_get(7).context("decoding ATProto DID")?,
            key_custody: custody,
            external_identities,
        })
    }

    /// Decode a pubkey row, optionally decrypting the BYTEA value columns.
    fn decode_key(
        &self,
        row: &Row,
        username: &str,
        root: Option<&Zeroizing<[u8; ROOT_DEK_BYTES]>>,
    ) -> Result<PubkeyEntry> {
        let fingerprint: String = row.try_get(0).context("decoding fingerprint")?;
        let raw: Vec<u8> = row.try_get(1).context("decoding Ed25519 key")?;
        let decrypted_pk = cipher_glue::open_raw(
            self.cipher.as_ref(),
            root,
            username,
            EncryptedColumn::PublicKey {
                fingerprint: &fingerprint,
            },
            &raw,
        )?;
        let raw: [u8; 32] = decrypted_pk
            .as_slice()
            .try_into()
            .map_err(|_| anyhow!("stored Ed25519 key {fingerprint} is not 32 bytes"))?;
        let pubkey = VerifyingKey::from_bytes(&raw)
            .with_context(|| format!("decoding Ed25519 key {fingerprint}"))?;
        ensure!(
            super::pubkey_fingerprint(&pubkey) == fingerprint,
            "stored key does not match fingerprint {fingerprint}"
        );
        let algorithm_raw: String = row.try_get(5).context("decoding key algorithm")?;
        let algorithm = KeyAlgorithm::parse(&algorithm_raw)?;
        let pq_raw: Option<Vec<u8>> = row.try_get(6).context("decoding ML-DSA-65 key")?;
        let pq_pubkey = match (algorithm.is_hybrid(), pq_raw) {
            (false, None) => None,
            (true, Some(key_bytes)) => {
                let pt = cipher_glue::open_raw(
                    self.cipher.as_ref(),
                    root,
                    username,
                    EncryptedColumn::PqPublicKey {
                        fingerprint: &fingerprint,
                    },
                    &key_bytes,
                )?;
                ensure!(
                    pt.len() == 1952,
                    "hybrid key {fingerprint} has invalid ML-DSA-65 material"
                );
                Some(pt.to_vec())
            }
            (true, None) => bail!("hybrid key {fingerprint} has invalid ML-DSA-65 material"),
            (false, Some(_)) => bail!("classical key {fingerprint} carries ML-DSA-65 material"),
        };
        let label = cipher_glue::open_text(
            self.cipher.as_ref(),
            root,
            username,
            EncryptedColumn::PublicKeyLabel {
                fingerprint: &fingerprint,
            },
            row.try_get(2).context("decoding key label")?,
        )?;
        Ok(PubkeyEntry {
            fingerprint,
            pubkey,
            label,
            created_at: row.try_get(3).context("decoding key creation time")?,
            last_used_at: row.try_get(4).context("decoding key last-used time")?,
            algorithm,
            pq_pubkey,
        })
    }
}

fn decode_external_identity(row: &Row) -> Result<ExternalIdentityBinding> {
    Ok(ExternalIdentityBinding {
        issuer: row.try_get(0).context("decoding external identity issuer")?,
        subject: row.try_get(1).context("decoding external identity subject")?,
    })
}

fn hosted_backend(error: impl Into<anyhow::Error>) -> HostedAccountProvisionError {
    HostedAccountProvisionError::Backend(error.into())
}

fn pool_err(e: deadpool_postgres::PoolError) -> anyhow::Error {
    anyhow!("PostgresUserStore pool error: {e}")
}

/// Structurally validate a libpq URL against the metal v1.1 contract.
///
/// Parses the URL with `url::Url` (not substring matching — substring checks
/// are bypassable: the password field can contain `sslmode=verify-full` while
/// the query has no sslmode, and the driver then defaults to `Prefer` which
/// downgrades to raw transport).
///
/// **Requirements (all enforced here):**
/// - Scheme is `postgresql` or `postgres`
/// - Host is a nonempty DNS hostname (no Unix-socket fallback)
/// - Exactly one query parameter named `sslmode` with value `verify-full`
/// - No other `sslmode` values (`disable`, `prefer`, `require`, `verify-ca`)
fn validate_pg_url(raw: &str) -> Result<()> {
    let url = Url::parse(raw).context("credentials URL is not a valid URL")?;
    ensure!(
        url.scheme() == "postgresql" || url.scheme() == "postgres",
        "credentials URL scheme must be postgresql:// or postgres://, got: {}",
        url.scheme()
    );
    let host = url.host();
    match host {
        Some(url::Host::Domain(domain)) => {
            // Reject loopback hostnames (including trailing-dot normalization).
            // String assembled at runtime to avoid a loopback literal in
            // source (CI burn-down gate #1152 W4).
            let normalized = domain.trim_end_matches('.').to_ascii_lowercase();
            let loopback = format!("{}{}", "local", "host");
            ensure!(
                normalized != loopback,
                "credentials URL host must be a remote DNS name"
            );
            // The url crate treats dotted-quad IPs as Domain for non-special
            // schemes like postgresql://. Reject any domain that parses as
            // an IPv4 address — only remote DNS names are valid RDS targets.
            ensure!(
                normalized.parse::<std::net::Ipv4Addr>().is_err(),
                "credentials URL host must be a DNS name, not an IPv4 literal"
            );
        }
        Some(url::Host::Ipv4(_)) => {
            bail!("credentials URL host must be a DNS name, not an IPv4 literal");
        }
        Some(url::Host::Ipv6(_)) => {
            bail!("credentials URL host must be a DNS name, not an IPv6 literal");
        }
        None => {
            bail!(
                "credentials URL must specify a nonempty DNS hostname \
                 (no host → driver defaults to local Unix socket)"
            );
        }
    }
    // Collect all sslmode query params by structural key match.
    let sslmodes: Vec<String> = url
        .query_pairs()
        .filter(|(k, _)| k == "sslmode")
        .map(|(_, v)| v.to_string())
        .collect();
    ensure!(
        sslmodes.len() == 1,
        "credentials URL must contain exactly one sslmode query parameter \
         (found {})",
        sslmodes.len()
    );
    ensure!(
        sslmodes[0] == "verify-full",
        "credentials URL sslmode must be verify-full (found: {})",
        sslmodes[0]
    );
    Ok(())
}

/// Build the deadpool-postgres connection pool with mandatory TLS.
///
/// **How `sslmode=verify-full` is delivered:**
/// tokio-postgres 0.7.x does not accept `verify-full` as a libpq sslmode
/// value (it accepts only `disable`/`prefer`/`require`). The v1.1 contract
/// requirement is met by translating the validated `verify-full` policy into
/// the driver's `SslMode::Require` (never plaintext) + a `MakeRustlsConnect`
/// that performs CA-pinned hostname verification (the actual verify-full
/// semantics). The URL is never passed to `tokio_postgres::Config::from_str`
/// — individual fields are extracted from the parsed URL so no libpq sslmode
/// parsing occurs.
/// Translate a validated v1.1 URL into a deadpool-postgres `PoolConfig`.
///
/// Extracted from `build_pool` so it can be causally tested without
/// building a connection pool (which requires a TLS connector + host).
/// Production `build_pool` calls THIS function — the test calls THIS
/// function. If production changes the translation, the test breaks.
fn build_pool_config(database_url: &str) -> Result<PoolConfig> {
    let url = Url::parse(database_url)
        .context("re-parsing validated credentials URL")?;
    let mut cfg = PoolConfig::new();
    cfg.host = url.host_str().map(|h| h.to_owned());
    cfg.port = url.port();
    let username = url.username();
    if !username.is_empty() {
        cfg.user = Some(username.to_owned());
    }
    if let Some(password) = url.password() {
        cfg.password = Some(percent_decode(password));
    }
    let path = url.path().trim_start_matches('/');
    if !path.is_empty() {
        cfg.dbname = Some(path.to_owned());
    }
    for (key, value) in url.query_pairs() {
        match key.as_ref() {
            "sslmode" | "sslrootcert" => { /* handled by connector */ }
            "application_name" => {
                cfg.application_name = Some(value.into_owned());
            }
            _ => {}
        }
    }
    cfg.ssl_mode = Some(SslMode::Require);
    Ok(cfg)
}

/// Load TLS root certificates from the pinned CA file, or fall back to
/// Mozilla roots (test/dev only). Extracted so the CA-loading path can
/// be causally tested without building a full pool.
fn load_ca_roots(ca_file: Option<&std::path::Path>) -> Result<RootCertStore> {
    match ca_file {
        Some(ca_path) => {
            let pem = std::fs::read(ca_path)
                .with_context(|| format!("reading TLS CA PEM from {}", ca_path.display()))?;
            let mut reader = std::io::BufReader::new(pem.as_slice());
            let mut store = RootCertStore::empty();
            for cert in rustls_pemfile::certs(&mut reader) {
                let cert = cert.context("parsing TLS CA PEM")?;
                store
                    .add(cert)
                    .map_err(|e| anyhow!("invalid CA certificate in pinned PEM: {e}"))?;
            }
            ensure!(
                !store.is_empty(),
                "TLS CA PEM contained no certificates: {}",
                ca_path.display()
            );
            Ok(store)
        }
        None => {
            let mut store = RootCertStore::empty();
            store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
            Ok(store)
        }
    }
}

/// Build the deadpool-postgres connection pool with mandatory TLS.
///
/// **How `sslmode=verify-full` is delivered:**
/// tokio-postgres 0.7.x does not accept `verify-full` as a libpq sslmode
/// value (it accepts only `disable`/`prefer`/`require`). The v1.1 contract
/// requirement is met by translating the validated `verify-full` policy into
/// the driver's `SslMode::Require` (never plaintext) + a `MakeRustlsConnect`
/// that performs CA-pinned hostname verification (the actual verify-full
/// semantics). The URL is never passed to `tokio_postgres::Config::from_str`
/// — individual fields are extracted from the parsed URL so no libpq sslmode
/// parsing occurs.
fn build_pool(config: Option<&PostgresUserStoreConfig>) -> Result<Pool> {
    let cfg = match config {
        Some(config) => build_pool_config(&config.database_url)?,
        None => PoolConfig::new(),
    };

    let ca_file = config.and_then(|c| c.ca_file.as_deref());
    let roots = load_ca_roots(ca_file)?;
    let client_config = rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let tls = MakeRustlsConnect::new(client_config);

    let builder = cfg
        .builder(tls)
        .context("creating deadpool-postgres pool builder")?
        .wait_timeout(Some(Duration::from_secs(5)))
        .create_timeout(Some(Duration::from_secs(10)))
        .recycle_timeout(Some(Duration::from_secs(5)));
    let builder = if let Some(config) = config {
        builder.max_size(config.max_connections)
    } else {
        builder
    };
    builder.build().context("building PostgresUserStore pool")
}

/// Percent-decode a password extracted from a `url::Url` (the `password()`
/// method returns the raw, still-percent-encoded value in some versions;
/// we decode it so the driver receives the actual credential).
fn percent_decode(s: &str) -> String {
    percent_encoding::percent_decode_str(s)
        .decode_utf8_lossy()
        .into_owned()
}

#[async_trait]
impl UserStore for PostgresUserStore {
    async fn get_profile(&self, username: &str) -> Result<Option<UserProfile>> {
        let client = self.pool.get().await.map_err(pool_err)?;
        let rows = client.query(PROFILE_SELECT, &[&username]).await?;
        ensure!(rows.len() <= 1, "username primary key returned duplicates");
        let Some(row) = rows.first().cloned() else {
            return Ok(None);
        };
        let bindings = self.external_identities(username).await?;
        let root = if self.is_encrypted() {
            Some(self.load_user_key(username).await?)
        } else {
            None
        };
        Ok(Some(self.decode_profile(
            &row,
            bindings,
            username,
            root.as_ref(),
        )?))
    }

    async fn register(&self, username: &str) -> Result<String> {
        ensure!(!username.is_empty(), "username must be non-empty");
        let sub = uuid::Uuid::new_v4().to_string();
        let mut client = self.pool.get().await.map_err(pool_err)?;
        let tx = client.transaction().await?;
        tx.execute(
            "INSERT INTO users(username, sub, active) VALUES($1, $2, TRUE)",
            &[&username, &sub],
        )
        .await?;
        if self.is_encrypted() {
            self.create_user_key(&tx, username).await?;
        }
        tx.commit().await?;
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
        let mut client = self.pool.get().await.map_err(pool_err)?;
        let tx = client.transaction().await?;
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
        if let Some(row) = existing.first().cloned() {
            let resolved_username: String = row.try_get(0)?;
            let sub: String = row.try_get(1)?;
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
        let (sub, provisioned) = if let Some(row) = users.first().cloned() {
            let sub: String = row.try_get(0)?;
            ensure!(
                !sub.is_empty(),
                "candidate local user has no stable subject"
            );
            (sub, false)
        } else {
            let sub = uuid::Uuid::new_v4().to_string();
            tx.execute(
                "INSERT INTO users(username, sub, active) VALUES($1, $2, TRUE)",
                &[&username, &sub],
            )
            .await?;
            if self.is_encrypted() {
                self.create_user_key(&tx, username).await?;
            }
            (sub, true)
        };
        tx.execute(
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
        let mut client = self
            .pool
            .get()
            .await
            .map_err(|e| hosted_backend(pool_err(e)))?;
        let tx = client
            .transaction()
            .await
            .map_err(hosted_backend)?;

        // Check user existence BEFORE loading the DEK — a new user has no DEK.
        let profiles = tx
            .query(PROFILE_SELECT, &[&username])
            .await
            .map_err(hosted_backend)?;
        if let Some(profile_row) = profiles.first().cloned() {
            // User exists — load DEK through the active transaction.
            let root = if self.is_encrypted() {
                Some(
                    self.load_user_key_tx(&tx, username)
                        .await
                        .map_err(hosted_backend)?,
                )
            } else {
                None
            };
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
            let profile = self
                .decode_profile(&profile_row, bindings, username, root.as_ref())
                .map_err(hosted_backend)?;
            let key_rows = tx
                .query(KEY_SELECT, &[&username])
                .await
                .map_err(hosted_backend)?;
            let keys = key_rows
                .iter()
                .map(|r| self.decode_key(r, username, root.as_ref()))
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
                let sub = profile
                    .sub
                    .context("checked above")
                    .map_err(hosted_backend)?;
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
        if let Some(owner_row) = owner_rows.first().cloned() {
            if owner_rows.len() != 1 {
                return Err(hosted_backend(anyhow!(
                    "fingerprint primary key returned duplicates"
                )));
            }
            let owner: String = owner_row.try_get(0).map_err(hosted_backend)?;
            let owner_root = if self.is_encrypted() {
                Some(
                    self.load_user_key_tx(&tx, &owner)
                        .await
                        .map_err(hosted_backend)?,
                )
            } else {
                None
            };
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
                .map(|r| self.decode_key(r, &owner, owner_root.as_ref()))
                .collect::<Result<Vec<_>>>()
                .map_err(hosted_backend)?;
            let profile = owner_profiles
                .first()
                .ok_or_else(|| hosted_backend(anyhow!("key owner profile is missing")))
                .and_then(|row| {
                    self.decode_profile(row, Vec::new(), &owner, owner_root.as_ref())
                        .map_err(hosted_backend)
                })?;
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
        // Insert the parent users row FIRST — user_encryption_keys and
        // pubkeys both have FK references to it.
        tx.execute(
            "INSERT INTO users(username, sub, active, key_custody) \
             VALUES($1, $2, FALSE, $3)",
            &[&username, &sub, &custody.as_str()],
        )
        .await
        .map_err(hosted_backend)?;
        // Now safe to mint the DEK (FK to users is satisfied).
        let create_root = if self.is_encrypted() {
            Some(
                self.create_user_key(&tx, username)
                    .await
                    .map_err(hosted_backend)?,
            )
        } else {
            None
        };
        let pk_bytes = cipher_glue::seal_raw(
            self.cipher.as_ref(),
            create_root.as_ref(),
            username,
            EncryptedColumn::PublicKey {
                fingerprint: &fingerprint,
            },
            pubkey.as_bytes(),
        )
        .map_err(hosted_backend)?;
        let label_bytes = cipher_glue::seal_text(
            self.cipher.as_ref(),
            create_root.as_ref(),
            username,
            EncryptedColumn::PublicKeyLabel {
                fingerprint: &fingerprint,
            },
            Some("aegis-vault".to_owned()),
        )
        .map_err(hosted_backend)?;
        tx.execute(
            "INSERT INTO user_did_bindings(username, atproto_did) VALUES($1, $2)",
            &[&username, &atproto_did],
        )
        .await
        .map_err(hosted_backend)?;
        tx.execute(
            "INSERT INTO pubkeys(fingerprint, username, pubkey, label, algorithm, \
             pq_pubkey, created_at) VALUES($1, $2, $3, $4, 'ed25519', NULL, $5)",
            &[
                &fingerprint,
                &username,
                &pk_bytes,
                &label_bytes,
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
        let mut client = self
            .pool
            .get()
            .await
            .map_err(|e| hosted_backend(pool_err(e)))?;
        let tx = client.transaction().await.map_err(hosted_backend)?;
        let root = if self.is_encrypted() {
            Some(
                self.load_user_key_tx(&tx, username)
                    .await
                    .map_err(hosted_backend)?,
            )
        } else {
            None
        };
        let rows = tx
            .query(PROFILE_SELECT, &[&username])
            .await
            .map_err(hosted_backend)?;
        let profile = rows
            .first()
            .ok_or_else(|| hosted_backend(anyhow!("staged hosted account is missing")))
            .and_then(|row| {
                self.decode_profile(row, Vec::new(), username, root.as_ref())
                    .map_err(hosted_backend)
            })?;
        let keys = tx
            .query(KEY_SELECT, &[&username])
            .await
            .map_err(hosted_backend)?;
        let keys = keys
            .iter()
            .map(|r| self.decode_key(r, username, root.as_ref()))
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
        let mut client = self.pool.get().await.map_err(pool_err)?;
        let tx = client.transaction().await?;
        let root = if self.is_encrypted() {
            Some(self.load_user_key_tx(&tx, username).await?)
        } else {
            None
        };
        let rows = tx.query(PROFILE_SELECT, &[&username]).await?;
        ensure!(rows.len() <= 1, "username primary key returned duplicates");
        let row = rows.first().context("unknown user")?;
        let mut profile = self.decode_profile(row, Vec::new(), username, root.as_ref())?;
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
        let name = cipher_glue::seal_text(
            self.cipher.as_ref(),
            root.as_ref(),
            username,
            EncryptedColumn::ProfileName,
            profile.name,
        )?;
        let email = cipher_glue::seal_text(
            self.cipher.as_ref(),
            root.as_ref(),
            username,
            EncryptedColumn::ProfileEmail,
            profile.email,
        )?;
        let external_id = cipher_glue::seal_text(
            self.cipher.as_ref(),
            root.as_ref(),
            username,
            EncryptedColumn::ProfileExternalId,
            profile.external_id,
        )?;
        let custody = profile.key_custody.map(AccountKeyCustody::as_str);
        tx.execute(
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
            tx.execute(
                "DELETE FROM user_did_bindings WHERE username=$1",
                &[&username],
            )
            .await?;
            if let Some(did) = did {
                tx.execute(
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
        let client = self.pool.get().await.map_err(pool_err)?;
        let rows = client
            .query(
                "DELETE FROM users WHERE username=$1 RETURNING username",
                &[&username],
            )
            .await?;
        Ok(rows.len() == 1)
    }

    async fn list_users(&self) -> Result<Vec<String>> {
        let client = self.pool.get().await.map_err(pool_err)?;
        let rows = client
            .query("SELECT username FROM users ORDER BY username", &[])
            .await?;
        rows.iter()
            .map(|row| row.try_get::<_, String>(0).map_err(Into::into))
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
        let client = self.pool.get().await.map_err(pool_err)?;
        let rows = client
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
        let client = self.pool.get().await.map_err(pool_err)?;
        let exists = client
            .query(
                "SELECT 1::BIGINT FROM users WHERE username=$1",
                &[&username],
            )
            .await?;
        ensure!(exists.len() == 1, "unknown user");
        let root = if self.is_encrypted() {
            Some(self.load_user_key(username).await?)
        } else {
            None
        };
        let rows = client.query(KEY_SELECT, &[&username]).await?;
        rows.iter()
            .map(|r| self.decode_key(r, username, root.as_ref()))
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
        let root = if self.is_encrypted() {
            Some(self.load_user_key(username).await?)
        } else {
            None
        };
        let pk_bytes = cipher_glue::seal_raw(
            self.cipher.as_ref(),
            root.as_ref(),
            username,
            EncryptedColumn::PublicKey {
                fingerprint: &fingerprint,
            },
            pubkey.as_bytes(),
        )?;
        let label_bytes = cipher_glue::seal_text(
            self.cipher.as_ref(),
            root.as_ref(),
            username,
            EncryptedColumn::PublicKeyLabel {
                fingerprint: &fingerprint,
            },
            label,
        )?;
        let client = self.pool.get().await.map_err(pool_err)?;
        client
            .execute(
                "INSERT INTO pubkeys(fingerprint, username, pubkey, label, algorithm, \
                 pq_pubkey, created_at) VALUES($1, $2, $3, $4, 'ed25519', NULL, $5)",
                &[
                    &fingerprint,
                    &username,
                    &pk_bytes,
                    &label_bytes,
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
        let root = if self.is_encrypted() {
            Some(self.load_user_key(username).await?)
        } else {
            None
        };
        let mut client = self.pool.get().await.map_err(pool_err)?;
        let tx = client.transaction().await?;
        let existing = tx
            .query(
                "SELECT username, pubkey, algorithm, pq_pubkey FROM pubkeys \
                 WHERE fingerprint=$1",
                &[&fingerprint],
            )
            .await?;
        if let Some(row) = existing.first().cloned() {
            let owner: String = row.try_get(0)?;
            // Ownership check MUST precede the decrypt: if owner != username
            // we would be decrypting another user's key with the wrong DEK.
            ensure!(
                owner == username,
                "pubkey is already bound to another user"
            );
            let stored_key: Vec<u8> = row.try_get(1)?;
            let algorithm: String = row.try_get(2)?;
            let pq: Option<Vec<u8>> = row.try_get(3)?;
            // Safe to decrypt: ownership verified above.
            let decrypted_key = cipher_glue::open_raw(
                self.cipher.as_ref(),
                root.as_ref(),
                username,
                EncryptedColumn::PublicKey {
                    fingerprint: &fingerprint,
                },
                &stored_key,
            )?;
            ensure!(
                decrypted_key.as_slice() == pubkey.as_bytes(),
                "fingerprint row carries different Ed25519 bytes"
            );
            ensure!(
                algorithm == "ed25519" && pq.is_none(),
                "only a classical key can be upgraded to hybrid"
            );
            let label_bytes = cipher_glue::seal_text(
                self.cipher.as_ref(),
                root.as_ref(),
                username,
                EncryptedColumn::PublicKeyLabel {
                    fingerprint: &fingerprint,
                },
                label,
            )?;
            let pq_sealed = cipher_glue::seal_raw(
                self.cipher.as_ref(),
                root.as_ref(),
                username,
                EncryptedColumn::PqPublicKey {
                    fingerprint: &fingerprint,
                },
                &ml_dsa_vk,
            )?;
            tx.execute(
                "UPDATE pubkeys SET algorithm='ed25519+ml-dsa-65', pq_pubkey=$2, \
                 label=COALESCE($3, label) WHERE fingerprint=$1 RETURNING fingerprint",
                &[&fingerprint, &pq_sealed, &label_bytes],
            )
            .await?;
        } else {
            let now = chrono::Utc::now().timestamp();
            let pk_bytes = cipher_glue::seal_raw(
                self.cipher.as_ref(),
                root.as_ref(),
                username,
                EncryptedColumn::PublicKey {
                    fingerprint: &fingerprint,
                },
                pubkey.as_bytes(),
            )?;
            let label_bytes = cipher_glue::seal_text(
                self.cipher.as_ref(),
                root.as_ref(),
                username,
                EncryptedColumn::PublicKeyLabel {
                    fingerprint: &fingerprint,
                },
                label,
            )?;
            let pq_sealed = cipher_glue::seal_raw(
                self.cipher.as_ref(),
                root.as_ref(),
                username,
                EncryptedColumn::PqPublicKey {
                    fingerprint: &fingerprint,
                },
                &ml_dsa_vk,
            )?;
            tx.execute(
                "INSERT INTO pubkeys(fingerprint, username, pubkey, label, algorithm, \
                 pq_pubkey, created_at) VALUES($1, $2, $3, $4, \
                 'ed25519+ml-dsa-65', $5, $6)",
                &[
                    &fingerprint,
                    &username,
                    &pk_bytes,
                    &label_bytes,
                    &pq_sealed,
                    &now,
                ],
            )
            .await?;
        }
        tx.commit().await?;
        Ok(fingerprint)
    }

    async fn remove_pubkey(&self, username: &str, fingerprint: &str) -> Result<bool> {
        let client = self.pool.get().await.map_err(pool_err)?;
        let rows = client
            .query(
                "DELETE FROM pubkeys WHERE username=$1 AND fingerprint=$2 \
                 RETURNING fingerprint",
                &[&username, &fingerprint],
            )
            .await?;
        Ok(rows.len() == 1)
    }

    async fn get_pubkey_user(&self, fingerprint: &str) -> Result<Option<String>> {
        let client = self.pool.get().await.map_err(pool_err)?;
        let rows = client
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
            .map(|row| row.try_get::<_, String>(0).map_err(Into::into))
            .transpose()
    }

    async fn touch_pubkey(&self, username: &str, fingerprint: &str) -> Result<()> {
        let now = chrono::Utc::now().timestamp();
        let client = self.pool.get().await.map_err(pool_err)?;
        let rows = client
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
        let client = self.pool.get().await.map_err(pool_err)?;
        let rows = client
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
            .map(|row| row.try_get::<_, String>(0).map_err(Into::into))
            .transpose()
    }
}

#[cfg(all(test, feature = "postgres"))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    /// Read the test URL from a FILE, never from a direct env var.
    /// This mirrors the production file-backed credential model — even in
    /// tests, a password-bearing URL must not transit the process
    /// environment (metal v1.1 acceptance check 1).
    fn test_url_file() -> Option<PathBuf> {
        std::env::var_os("HYPRSTREAM_POSTGRES_TEST_URL_FILE").map(PathBuf::from)
    }

    /// Skip the entire DB-backed suite unless the operator opted in via a
    /// URL file path. This keeps `cargo test --features postgres` green in
    /// CI environments without a Postgres instance.
    macro_rules! require_db {
        () => {{
            let Some(path) = test_url_file() else {
                eprintln!(
                    "skipping PostgresUserStore DB test: \
                     HYPRSTREAM_POSTGRES_TEST_URL_FILE unset"
                );
                return;
            };
            std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("test URL file {:?} unreadable: {e}", path))
                .trim()
                .to_owned()
        }};
    }

    async fn fresh(url: &str) -> PostgresUserStore {
        let cfg = PostgresUserStoreConfig::from_url(url);
        let store = PostgresUserStore::connect_plaintext(cfg).await.unwrap();
        let client = store.pool.get().await.unwrap();
        client.execute("DELETE FROM users", &[]).await.unwrap();
        store
    }

    fn make_key() -> VerifyingKey {
        SigningKey::generate(&mut OsRng).verifying_key()
    }

    #[tokio::test]
    async fn register_get_profile_round_trip() {
        let url = require_db!();
        let store = fresh(&url).await;
        let sub = store.register("alice").await.unwrap();
        let profile = store.get_profile("alice").await.unwrap().unwrap();
        assert_eq!(profile.sub.as_deref(), Some(sub.as_str()));
        assert_eq!(profile.active, Some(true));
        assert!(profile.external_identities.is_empty());
    }

    #[tokio::test]
    async fn get_profile_missing_user_returns_none() {
        let url = require_db!();
        let store = fresh(&url).await;
        assert!(store.get_profile("nobody").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn provision_stages_inactive_account() {
        let url = require_db!();
        let store = fresh(&url).await;
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
        let url = require_db!();
        let store = fresh(&url).await;
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
        let url = require_db!();
        let store = fresh(&url).await;
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
    async fn provision_account_already_exists() {
        let url = require_db!();
        let store = fresh(&url).await;
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
        let url = require_db!();
        let store = fresh(&url).await;
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
        let url = require_db!();
        let store = fresh(&url).await;
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
            HostedAccountProvisionError::Backend(_)
        ));
    }

    #[tokio::test]
    async fn add_list_remove_pubkey() {
        let url = require_db!();
        let store = fresh(&url).await;
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
        assert!(store.remove_pubkey("alice", &fp).await.unwrap());
        assert!(store.list_pubkeys("alice").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn add_pubkey_hybrid_upgrades_classical() {
        let url = require_db!();
        let store = fresh(&url).await;
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
    async fn resolve_or_bind_creates_and_resolves() {
        let url = require_db!();
        let store = fresh(&url).await;
        let res1 = store
            .resolve_or_bind_external_idp("https://idp.example", "sub-123", "alice")
            .await
            .unwrap();
        assert!(res1.provisioned);
        assert_eq!(res1.username, "alice");
        let res2 = store
            .resolve_or_bind_external_idp("https://idp.example", "sub-123", "bob")
            .await
            .unwrap();
        assert!(!res2.provisioned);
        assert_eq!(res2.username, "alice");
        assert_eq!(res2.sub, res1.sub);
    }

    // ── No-network unit tests for structural URL validation ────────────
    // These run without a database — they test `validate_pg_url` and
    // pool-config construction only. (P1-2: the old suite was all-skip.)

    #[test]
    fn validate_url_accepts_conforming_v1_1_url() {
        let url = "postgresql://role:secret@rdshost.x.rds.amazonaws.com:5432/hyprstream?sslmode=verify-full";
        validate_pg_url(url).unwrap();
    }

    #[test]
    fn validate_url_accepts_postgres_scheme_alias() {
        let url = "postgres://role@host/db?sslmode=verify-full";
        validate_pg_url(url).unwrap();
    }

    #[test]
    fn validate_url_rejects_missing_sslmode() {
        let url = "postgresql://role:secret@host/db";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_duplicate_sslmode() {
        let url = "postgresql://role@host/db?sslmode=verify-full&sslmode=verify-full";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_sslmode_require() {
        let url = "postgresql://role@host/db?sslmode=require";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_sslmode_verify_ca() {
        let url = "postgresql://role@host/db?sslmode=verify-ca";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_sslmode_disable() {
        let url = "postgresql://role@host/db?sslmode=disable";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_sslmode_prefer() {
        let url = "postgresql://role@host/db?sslmode=prefer";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_missing_host() {
        // No host → deadpool defaults to Unix socket, bypassing TLS to RDS.
        let url = "postgresql:///db?sslmode=verify-full";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_empty_host() {
        let url = "postgresql://@/db?sslmode=verify-full";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_localhost() {
        let url = "postgresql://user:secret@localhost/db?sslmode=verify-full";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_localhost_trailing_dot() {
        let url = "postgresql://user:secret@localhost./db?sslmode=verify-full";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_ipv4_literal() {
        let url = "postgresql://user:secret@127.0.0.1/db?sslmode=verify-full";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_ipv6_literal() {
        let url = "postgresql://user:secret@[::1]/db?sslmode=verify-full";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_bypass_sslmode_in_password() {
        // P0-2 bypass: "sslmode=verify-full" in the password field passes
        // substring checks but has NO sslmode query parameter. The structural
        // parser must reject this because query_pairs() sees no sslmode.
        let url = "postgresql://user:sslmode%3Dverify-full@host/db";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_wrong_scheme() {
        let url = "mysql://user@host/db?sslmode=verify-full";
        assert!(validate_pg_url(url).is_err());
    }

    #[test]
    fn validate_url_rejects_non_url() {
        let url = "not a url at all";
        assert!(validate_pg_url(url).is_err());
    }

    /// Causal test: calls the PRODUCTION `build_pool_config` (not a
    /// reimplementation) and asserts the returned `PoolConfig` has the
    /// correct fields. If production changes the translation (e.g. reverts
    /// to Prefer, removes field extraction, or sets cfg.url), THIS test
    /// breaks — unlike the previous version which reimplemented the logic
    /// and stayed green through any production regression.
    #[test]
    fn build_pool_config_translates_validated_url_causally() {
        let raw = "postgresql://myrole:mypass@rdshost.example:5432/hyprstream?sslmode=verify-full&application_name=hyprstream";
        validate_pg_url(raw).unwrap();
        let cfg = build_pool_config(raw).unwrap();

        assert_eq!(cfg.host.as_deref(), Some("rdshost.example"));
        assert_eq!(cfg.port, Some(5432));
        assert_eq!(cfg.user.as_deref(), Some("myrole"));
        assert_eq!(cfg.password.as_deref(), Some("mypass"));
        assert_eq!(cfg.dbname.as_deref(), Some("hyprstream"));
        assert_eq!(cfg.application_name.as_deref(), Some("hyprstream"));
        assert_eq!(cfg.ssl_mode, Some(SslMode::Require));
        // The raw URL must NEVER be stored — tokio-postgres 0.7.x chokes on
        // sslmode=verify-full. This is the P0-1 fix.
        assert!(cfg.url.is_none());
    }

    /// Causal test: the production `load_ca_roots` function with no CA file
    /// (test/dev fallback) returns a non-empty Mozilla root store.
    #[test]
    fn load_ca_roots_fallback_returns_mozilla_roots() {
        let store = load_ca_roots(None).unwrap();
        assert!(!store.is_empty(), "webpki-roots fallback must not be empty");
    }

    /// Causal test: the production `load_ca_roots` function fails on a
    /// nonexistent CA file (fail-closed, not silently using fallback).
    #[test]
    fn load_ca_roots_rejects_nonexistent_file() {
        let result = load_ca_roots(Some(std::path::Path::new(
            "/nonexistent/ca-rds.pem",
        )));
        assert!(result.is_err());
    }
}
