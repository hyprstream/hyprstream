//! Persistent refresh token store abstraction.
//!
//! `TokenStore` is the trait. `RocksDbTokenStore` wraps the existing rocksdb::DB.
//! A Valkey implementation is provided when the `valkey` feature is enabled.

use std::sync::Arc;
use std::path::Path;
use anyhow::Result;
use async_trait::async_trait;

use super::state::RefreshTokenEntry;

/// Persistent storage for refresh tokens.
#[async_trait]
pub trait TokenStore: Send + Sync {
    async fn put(&self, token: &str, entry: &RefreshTokenEntry, ttl_secs: u64) -> Result<()>;
    async fn get(&self, token: &str) -> Result<Option<RefreshTokenEntry>>;
    async fn delete(&self, token: &str) -> Result<()>;
}

pub struct RocksDbTokenStore {
    db: Arc<rocksdb::DB>,
}

impl RocksDbTokenStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut opts = rocksdb::Options::default();
        opts.create_if_missing(true);
        let db = rocksdb::DB::open(&opts, path)?;
        Ok(Self { db: Arc::new(db) })
    }
}

#[async_trait]
impl TokenStore for RocksDbTokenStore {
    async fn put(&self, token: &str, entry: &RefreshTokenEntry, _ttl_secs: u64) -> Result<()> {
        let bytes = serde_json::to_vec(entry)?;
        self.db.put(token.as_bytes(), bytes)?;
        Ok(())
    }

    async fn get(&self, token: &str) -> Result<Option<RefreshTokenEntry>> {
        match self.db.get(token.as_bytes())? {
            None => Ok(None),
            Some(bytes) => {
                let entry: RefreshTokenEntry = serde_json::from_slice(&bytes)?;
                if entry.is_expired() {
                    let _ = self.db.delete(token.as_bytes());
                    Ok(None)
                } else {
                    Ok(Some(entry))
                }
            }
        }
    }

    async fn delete(&self, token: &str) -> Result<()> {
        self.db.delete(token.as_bytes())?;
        Ok(())
    }
}

/// Valkey-backed refresh token store (requires `valkey` feature).
#[cfg(feature = "valkey")]
pub struct ValkeyTokenStore {
    pool: fred::prelude::RedisPool,
    key_prefix: String,
}

#[cfg(feature = "valkey")]
impl ValkeyTokenStore {
    pub async fn connect(url: &str) -> Result<Self> {
        use fred::prelude::*;
        let config = RedisConfig::from_url(url)?;
        let pool = Builder::from_config(config).build_pool(8)?;
        pool.connect();
        pool.wait_for_connect().await?;
        Ok(Self { pool, key_prefix: "hs:token:".to_owned() })
    }

    fn key(&self, token: &str) -> String {
        format!("{}{}", self.key_prefix, token)
    }
}

#[cfg(feature = "valkey")]
#[async_trait]
impl TokenStore for ValkeyTokenStore {
    async fn put(&self, token: &str, entry: &RefreshTokenEntry, ttl_secs: u64) -> Result<()> {
        use fred::prelude::*;
        let bytes = serde_json::to_string(entry)?;
        self.pool
            .set::<(), _, _>(self.key(token), bytes, Some(Expiration::EX(ttl_secs as i64)), None, false)
            .await?;
        Ok(())
    }

    async fn get(&self, token: &str) -> Result<Option<RefreshTokenEntry>> {
        use fred::prelude::*;
        let val: Option<String> = self.pool.get(self.key(token)).await?;
        match val {
            None => Ok(None),
            Some(s) => Ok(Some(serde_json::from_str(&s)?)),
        }
    }

    async fn delete(&self, token: &str) -> Result<()> {
        use fred::prelude::*;
        self.pool.del::<i64, _>(self.key(token)).await?;
        Ok(())
    }
}
