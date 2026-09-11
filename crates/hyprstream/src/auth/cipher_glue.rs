//! Shared glue between relational [`UserStore`] backends and the
//! backend-neutral `ColumnCipher`.
//!
//! `USERSTORE_SCHEMA` is the one source of truth for the relational DDL and is
//! consumed by every relational backend, so it is compiled whenever any of them
//! is enabled.
//!
//! The four seal/open free functions below serve the server-Postgres backend
//! specifically: they take the cipher and root key as `Option`s, because that
//! backend can run with value-column encryption absent. `PgliteUserStore` holds
//! a non-optional `ColumnCipher` — encryption is unconditional there — so it
//! seals the same columns through the inherent `ColumnCipher::seal_*`/`open_*`
//! methods instead, and never calls these. They are therefore compiled only
//! with the `postgres` backend, matching their callers.
//!
//! Neither this module nor [`encrypted_columns`] has any backend-specific
//! (pglite / tokio-postgres) dependency — the cipher is pure AES-GCM-SIV +
//! HKDF + age CLI, and the DDL is standard PostgreSQL.

#[cfg(feature = "postgres")]
use super::encrypted_columns::{ColumnCipher, EncryptedColumn, ROOT_DEK_BYTES};
#[cfg(feature = "postgres")]
use anyhow::{bail, Result};
#[cfg(feature = "postgres")]
use zeroize::Zeroizing;

/// PostgreSQL-compatible schema shared by embedded PGlite and server Postgres.
///
/// Lookup keys stay queryable. PII, labels, and key material use `BYTEA` so
/// envelope encryption can be added without changing relational identities.
///
/// This const is the **single source of truth** for the relational UserStore
/// DDL (#1351). Both `PgliteUserStore` and `PostgresUserStore` consume it
/// verbatim — they do not define their own copies.
pub const USERSTORE_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY CHECK (username <> ''),
    sub TEXT NOT NULL UNIQUE CHECK (sub <> ''),
    name BYTEA CONSTRAINT users_name_storage_check CHECK (
        name IS NULL OR (
            octet_length(name) >= 32
            AND substring(name from 1 for 4) = decode('48534331', 'hex')
        )
    ),
    email BYTEA CONSTRAINT users_email_storage_check CHECK (
        email IS NULL OR (
            octet_length(email) >= 32
            AND substring(email from 1 for 4) = decode('48534331', 'hex')
        )
    ),
    email_verified BOOLEAN,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    external_id BYTEA CONSTRAINT users_external_id_storage_check CHECK (
        external_id IS NULL OR (
            octet_length(external_id) >= 32
            AND substring(external_id from 1 for 4) = decode('48534331', 'hex')
        )
    ),
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
    pubkey BYTEA NOT NULL
        CONSTRAINT pubkeys_pubkey_storage_check CHECK (
            octet_length(pubkey) = 64
            AND substring(pubkey from 1 for 4) = decode('48534331', 'hex')
        ),
    label BYTEA CONSTRAINT pubkeys_label_storage_check CHECK (
        label IS NULL OR (
            octet_length(label) >= 32
            AND substring(label from 1 for 4) = decode('48534331', 'hex')
        )
    ),
    algorithm TEXT NOT NULL DEFAULT 'ed25519'
        CHECK (algorithm IN ('ed25519', 'ed25519+ml-dsa-65')),
    pq_pubkey BYTEA,
    created_at BIGINT NOT NULL,
    last_used_at BIGINT,
    CONSTRAINT pubkeys_algorithm_pq_storage_check CHECK (
        (algorithm = 'ed25519' AND pq_pubkey IS NULL)
        OR (
            algorithm = 'ed25519+ml-dsa-65'
            AND pq_pubkey IS NOT NULL
            AND octet_length(pq_pubkey) = 1984
            AND substring(pq_pubkey from 1 for 4) = decode('48534331', 'hex')
        )
    )
);
CREATE INDEX IF NOT EXISTS pubkeys_username_idx ON pubkeys(username);
CREATE INDEX IF NOT EXISTS oidc_bindings_username_idx ON oidc_bindings(username);
CREATE TABLE IF NOT EXISTS user_encryption_keys (
    username TEXT PRIMARY KEY REFERENCES users(username) ON DELETE CASCADE,
    wrapped_dek BYTEA NOT NULL
);
"#;

// ── Field-level seal/open free functions ─────────────────────────────
//
// These bridge the backend-neutral `ColumnCipher` and the server-Postgres
// store's read/write paths. They are parameterized by
// `(cipher, root, username, column, value)` — no `self`, no backend type —
// and are `Option`-taking because that backend may run without value-column
// encryption configured. Backends whose cipher is unconditional call the
// inherent `ColumnCipher` methods directly rather than these.

/// Seal an optional text field for storage. Returns `None` for `None`
/// input. When encryption is disabled (`cipher` and `root` both `None`),
/// returns plaintext bytes.
#[cfg(feature = "postgres")]
pub(crate) fn seal_text(
    cipher: Option<&ColumnCipher>,
    root: Option<&Zeroizing<[u8; ROOT_DEK_BYTES]>>,
    username: &str,
    column: EncryptedColumn<'_>,
    value: Option<String>,
) -> Result<Option<Vec<u8>>> {
    match (value, cipher, root) {
        (None, _, _) => Ok(None),
        (Some(text), None, None) => Ok(Some(text.into_bytes())),
        (Some(text), Some(cipher), Some(root)) => Ok(Some(cipher.encrypt(
            root,
            username,
            column,
            text.as_bytes(),
        )?)),
        _ => bail!("cipher/root state mismatch in seal_text"),
    }
}

/// Open an optional text field from stored bytes. When encryption is
/// disabled, interprets bytes as UTF-8 directly.
#[cfg(feature = "postgres")]
pub(crate) fn open_text(
    cipher: Option<&ColumnCipher>,
    root: Option<&Zeroizing<[u8; ROOT_DEK_BYTES]>>,
    username: &str,
    column: EncryptedColumn<'_>,
    raw: Option<Vec<u8>>,
) -> Result<Option<String>> {
    match (raw, cipher, root) {
        (None, _, _) => Ok(None),
        (Some(bytes), None, None) => Ok(Some(String::from_utf8(bytes)?)),
        (Some(bytes), Some(cipher), Some(root)) => {
            let pt = cipher.decrypt(root, username, column, &bytes)?;
            Ok(Some(String::from_utf8(pt.to_vec())?))
        }
        _ => bail!("cipher/root state mismatch in open_text"),
    }
}

/// Seal raw bytes (e.g. pubkey material) for storage.
#[cfg(feature = "postgres")]
pub(crate) fn seal_raw(
    cipher: Option<&ColumnCipher>,
    root: Option<&Zeroizing<[u8; ROOT_DEK_BYTES]>>,
    username: &str,
    column: EncryptedColumn<'_>,
    value: &[u8],
) -> Result<Vec<u8>> {
    match (cipher, root) {
        (None, None) => Ok(value.to_vec()),
        (Some(cipher), Some(root)) => cipher.encrypt(root, username, column, value),
        _ => bail!("cipher/root state mismatch in seal_raw"),
    }
}

/// Open raw bytes from storage.
#[cfg(feature = "postgres")]
pub(crate) fn open_raw(
    cipher: Option<&ColumnCipher>,
    root: Option<&Zeroizing<[u8; ROOT_DEK_BYTES]>>,
    username: &str,
    column: EncryptedColumn<'_>,
    raw: &[u8],
) -> Result<Zeroizing<Vec<u8>>> {
    match (cipher, root) {
        (None, None) => Ok(Zeroizing::new(raw.to_vec())),
        (Some(cipher), Some(root)) => cipher.decrypt(root, username, column, raw),
        _ => bail!("cipher/root state mismatch in open_raw"),
    }
}
