//! Postgres-backed KV store for `PdsRecordStore` (#1257).
//!
//! This module implements the RDS Multi-AZ backend for the PDS record store.
//! It is selected when the `[rds]` config section is set and the
//! `pds-postgres` cargo feature is enabled. When unset, the store falls back
//! to its local RocksDB backend.
//!
//! ## D2 invariant — signed bytes are the source of truth
//!
//! The schema is a **projection-free BYTEA-keyed KV shell**: one table,
//! `(key BYTEA PRIMARY KEY, value BYTEA NOT NULL)`. The values are the exact
//! signed DAG-CBOR bytes the publisher commits — records, signed commits, and
//! daemon-authenticated at9p state envelopes. Nothing is normalized, parsed, or
//! reconstructed from SQL; SQL indexes only accelerate key-range scans over the
//! verbatim bytes. This makes the store "ship the already-signed evidence across
//! the seam" with zero rework when region federation (Stage 2/3 of
//! `ARCH-recursive-federation-verdict-fable.md`) ships rows to another cell.
//!
//! ## Fail-closed
//!
//! Every backend/connection error propagates as `Err` — the store never
//! returns empty-as-absent on a Postgres failure.
//!
//! ## Async bridge
//!
//! `deadpool-postgres` is async-only, but the existing `PdsRecordStore` API is
//! sync (the publisher runs in a thread-mode service; the resolver is called
//! from both sync and async contexts). `PgKv` owns a dedicated OS thread that
//! runs a single-worker tokio runtime + the deadpool pool. Sync callers send a
//! command over an `mpsc` channel and block on the reply — this is
//! runtime-context-proof (no `block_on` / `blocking_recv` panics regardless of
//! whether the caller is on a tokio thread, a plain OS thread, or inside
//! `block_in_place`).

use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

use anyhow::{Context as _, Result as AnyResult};

/// Commands sent from sync callers to the async-bridge thread.
enum PgCmd {
    /// Fetch one key. `None` = key absent.
    Get {
        key: Vec<u8>,
        reply: Sender<AnyResult<Option<Vec<u8>>>>,
    },
    /// Fetch multiple keys in one READ-ONLY transaction (snapshot consistency,
    /// matching `rocksdb::DB::snapshot().get()` in `load_at9p_state_from_db`).
    GetBatch {
        keys: Vec<Vec<u8>>,
        reply: Sender<AnyResult<Vec<Option<Vec<u8>>>>>,
    },
    /// Upsert one key/value.
    Put {
        key: Vec<u8>,
        value: Vec<u8>,
        reply: Sender<AnyResult<()>>,
    },
    /// Atomic upsert of multiple key/value pairs (single transaction).
    PutBatch {
        ops: Vec<(Vec<u8>, Vec<u8>)>,
        reply: Sender<AnyResult<()>>,
    },
    /// Range scan: all `(key, value)` pairs where `key >= start AND key < end`,
    /// ordered by key ascending. Mirrors RocksDB's `prefix_iterator`.
    RangeScan {
        start: Vec<u8>,
        end: Vec<u8>,
        reply: Sender<AnyResult<Vec<(Vec<u8>, Vec<u8>)>>>,
    },
    /// Full scan: all `(key, value)` pairs, ordered by key ascending. Mirrors
    /// RocksDB's `IteratorMode::Start`.
    AllPairs {
        reply: Sender<AnyResult<Vec<(Vec<u8>, Vec<u8>)>>>,
    },
    /// Ping (connection liveness check).
    Ping {
        reply: Sender<AnyResult<()>>,
    },
}

/// Sync KV store backed by a deadpool-postgres pool running on a dedicated
/// thread. Clone-safe (the inner state is `Arc`).
pub(crate) struct PgKv {
    cmd_tx: Sender<PgCmd>,
    /// Keep the bridge thread alive for the life of the store.
    _thread: JoinHandle<()>,
}

impl PgKv {
    /// Connect to RDS, run the schema migration, and spawn the async bridge.
    ///
    /// `cell_id` is stamped into the schema for the honorable-mention cell guard
    /// (recursive-federation arch verdict) — it does not gate any query; it is
    /// metadata only.
    pub(crate) fn connect(
        url: &str,
        root_cert_pem: Option<&std::path::Path>,
        cell_id: &str,
    ) -> AnyResult<Self> {
        let (cmd_tx, cmd_rx) = mpsc::channel::<PgCmd>();
        let url = url.to_owned();
        let root_cert = root_cert_pem.map(std::path::PathBuf::from);
        let cell_id = cell_id.to_owned();

        let thread = std::thread::Builder::new()
            .name("pds-pg-bridge".into())
            .spawn(move || {
                bridge_main(url, root_cert.as_deref(), &cell_id, cmd_rx);
            })
            .context("failed to spawn pds-pg-bridge thread")?;

        let kv = Self {
            cmd_tx,
            _thread: thread,
        };

        // Verify connectivity + migration by pinging. This is the
        // FATAL-on-unavailable boundary: a failed ping here propagates as an
        // error from `connect`, refusing startup rather than degrading to local.
        kv.ping().context("RDS connection check failed at startup")?;

        Ok(kv)
    }

    // ---- sync API (runtime-context-proof via mpsc round-trip) ----

    pub(crate) fn get(&self, key: &[u8]) -> AnyResult<Option<Vec<u8>>> {
        self.round_trip(|reply| PgCmd::Get {
            key: key.to_owned(),
            reply,
        })
    }

    pub(crate) fn get_batch(&self, keys: &[Vec<u8>]) -> AnyResult<Vec<Option<Vec<u8>>>> {
        self.round_trip(|reply| PgCmd::GetBatch {
            keys: keys.to_owned(),
            reply,
        })
    }

    pub(crate) fn put(&self, key: &[u8], value: &[u8]) -> AnyResult<()> {
        self.round_trip(|reply| PgCmd::Put {
            key: key.to_owned(),
            value: value.to_owned(),
            reply,
        })
    }

    pub(crate) fn put_batch(&self, ops: &[(Vec<u8>, Vec<u8>)]) -> AnyResult<()> {
        self.round_trip(|reply| PgCmd::PutBatch {
            ops: ops.to_owned(),
            reply,
        })
    }

    pub(crate) fn range_scan(&self, start: &[u8], end: &[u8]) -> AnyResult<Vec<(Vec<u8>, Vec<u8>)>> {
        self.round_trip(|reply| PgCmd::RangeScan {
            start: start.to_owned(),
            end: end.to_owned(),
            reply,
        })
    }

    pub(crate) fn all_pairs(&self) -> AnyResult<Vec<(Vec<u8>, Vec<u8>)>> {
        self.round_trip(|reply| PgCmd::AllPairs { reply })
    }

    pub(crate) fn ping(&self) -> AnyResult<()> {
        self.round_trip(|reply| PgCmd::Ping { reply })
    }

    fn round_trip<R, F>(&self, make_cmd: F) -> AnyResult<R>
    where
        R: Send + 'static,
        F: FnOnce(Sender<AnyResult<R>>) -> PgCmd,
    {
        let (reply_tx, reply_rx) = mpsc::channel::<AnyResult<R>>();
        self.cmd_tx
            .send(make_cmd(reply_tx))
            .map_err(|_| anyhow::anyhow!("pds-pg-bridge thread has exited"))?;
        reply_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("pds-pg-bridge thread dropped reply channel"))?
    }
}

/// Compute the exclusive upper bound for a byte-prefix range scan.
///
/// Given `prefix`, returns the lexicographically smallest byte string strictly
/// greater than all strings starting with `prefix` — or `None` if the prefix is
/// all `0xFF` (in which case the range extends to the end of the keyspace and
/// no upper bound is needed).
pub(crate) fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut v = prefix.to_vec();
    // Walk back from the last byte, incrementing the last non-0xFF byte and
    // truncating the rest.
    while let Some(last) = v.last() {
        if *last == 0xFF {
            v.pop();
        } else {
            let len = v.len();
            v[len - 1] += 1;
            return Some(v);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Bridge thread — owns the tokio runtime + deadpool pool.
// ---------------------------------------------------------------------------

fn bridge_main(
    url: String,
    root_cert: Option<&std::path::Path>,
    cell_id: &str,
    cmd_rx: Receiver<PgCmd>,
) {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!("pds-pg-bridge: failed to create runtime: {e}");
            // Drain all pending commands with the error so callers fail closed.
            for cmd in cmd_rx.iter() {
                let _ = send_err(cmd, anyhow::anyhow!("runtime creation failed: {e}"));
            }
            return;
        }
    };

    rt.block_on(async move {
        let pool = match build_pool(&url, root_cert).await {
            Ok(pool) => pool,
            Err(e) => {
                tracing::error!("pds-pg-bridge: failed to create pool: {e}");
                for cmd in cmd_rx.iter() {
                    let _ = send_err(cmd, anyhow::anyhow!("pool creation failed: {e}"));
                }
                return;
            }
        };

        // Run the idempotent schema migration. This is the FATAL boundary.
        if let Err(e) = migrate(&pool, cell_id).await {
            tracing::error!("pds-pg-bridge: schema migration failed: {e}");
            for cmd in cmd_rx.iter() {
                let _ = send_err(cmd, anyhow::anyhow!("schema migration failed: {e}"));
            }
            return;
        }

        tracing::info!("pds-pg-bridge: connected to RDS, schema migrated");

        // Serve commands until the sender half is dropped.
        while let Ok(cmd) = cmd_rx.recv() {
            if let Err(e) = handle_cmd(&pool, cmd).await {
                tracing::error!("pds-pg-bridge: command handler error: {e}");
                // The error was already sent to the caller inside handle_cmd;
                // this is just a log.
            }
        }
    });
}

async fn build_pool(
    url: &str,
    root_cert: Option<&std::path::Path>,
) -> AnyResult<deadpool_postgres::Pool> {
    let cfg = deadpool_postgres::Config {
        url: Some(url.to_owned()),
        ..Default::default()
    };

    let tls = tokio_postgres_rustls::MakeRustlsConnect::new(build_rustls_config(root_cert)?);
    let pool = cfg
        .create_pool(Some(deadpool_postgres::Runtime::Tokio1), tls)
        .map_err(|e| anyhow::anyhow!("deadpool-postgres pool creation failed: {e}"))?;
    Ok(pool)
}

/// Build a rustls client config for TLS to RDS. If `root_cert_pem` is given,
/// load it additively alongside the WebPKI roots.
fn build_rustls_config(
    root_cert_pem: Option<&std::path::Path>,
) -> AnyResult<rustls::ClientConfig> {
    let mut root_store = rustls::RootCertStore::empty();

    // Add the platform's trusted roots.
    let native = rustls_native_certs::load_native_certs();
    for cert in native.certs {
        let _ = root_store.add(cert);
    }
    for e in native.errors {
        tracing::warn!("failed to load a native TLS root for RDS: {e}");
    }

    // Additively load an operator-provided root (private-CA RDS).
    if let Some(path) = root_cert_pem {
        let pem = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("failed to read RDS root_cert at {path:?}: {e}"))?;
        for cert in rustls_pemfile::certs(&mut pem.as_slice())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("failed to parse RDS root_cert PEM at {path:?}: {e}"))?
        {
            root_store
                .add(cert)
                .map_err(|e| anyhow::anyhow!("failed to add RDS root cert: {e}"))?;
        }
    }

    Ok(
        rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth(),
    )
}

/// Idempotent schema migration.
///
/// One table: `pds_kv (key BYTEA PRIMARY KEY, value BYTEA NOT NULL)`. The
/// `cell_id` is stamped into a `pds_meta` row for the honorable-mention cell
/// guard. No queryable columns are derived from the signed bytes — D2.
async fn migrate(pool: &deadpool_postgres::Pool, cell_id: &str) -> AnyResult<()> {
    let conn = pool
        .get()
        .await
        .map_err(|e| anyhow::anyhow!("RDS connection acquisition failed during migration: {e}"))?;

    conn.batch_execute(
        "CREATE TABLE IF NOT EXISTS pds_kv (
            key   BYTEA PRIMARY KEY,
            value BYTEA NOT NULL
        );
        CREATE TABLE IF NOT EXISTS pds_meta (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        );",
    )
    .await
    .map_err(|e| anyhow::anyhow!("RDS schema migration (create tables) failed: {e}"))?;

    // Stamp the cell_id.
    conn.execute(
        "INSERT INTO pds_meta (k, v) VALUES ('cell_id', $1)
         ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v",
        &[&cell_id],
    )
    .await
    .map_err(|e| anyhow::anyhow!("RDS cell_id stamp failed: {e}"))?;

    Ok(())
}

async fn handle_cmd(pool: &deadpool_postgres::Pool, cmd: PgCmd) -> AnyResult<()> {
    match cmd {
        PgCmd::Get { key, reply } => {
            let r = cmd_get(pool, &key).await;
            let _ = reply.send(r);
        }
        PgCmd::GetBatch { keys, reply } => {
            let r = cmd_get_batch(pool, &keys).await;
            let _ = reply.send(r);
        }
        PgCmd::Put { key, value, reply } => {
            let r = cmd_put(pool, &key, &value).await;
            let _ = reply.send(r);
        }
        PgCmd::PutBatch { ops, reply } => {
            let r = cmd_put_batch(pool, &ops).await;
            let _ = reply.send(r);
        }
        PgCmd::RangeScan { start, end, reply } => {
            let r = cmd_range_scan(pool, &start, &end).await;
            let _ = reply.send(r);
        }
        PgCmd::AllPairs { reply } => {
            let r = cmd_all_pairs(pool).await;
            let _ = reply.send(r);
        }
        PgCmd::Ping { reply } => {
            let r = cmd_ping(pool).await;
            let _ = reply.send(r);
        }
    }
    Ok(())
}

async fn cmd_get(pool: &deadpool_postgres::Pool, key: &[u8]) -> AnyResult<Option<Vec<u8>>> {
    let conn = pool
        .get()
        .await
        .map_err(|e| anyhow::anyhow!("RDS get: connection acquisition failed: {e}"))?;
    let row = conn
        .query_opt("SELECT value FROM pds_kv WHERE key = $1", &[&key])
        .await
        .map_err(|e| anyhow::anyhow!("RDS get: query failed: {e}"))?;
    Ok(row.map(|r| r.get::<_, Vec<u8>>(0)))
}

async fn cmd_get_batch(
    pool: &deadpool_postgres::Pool,
    keys: &[Vec<u8>],
) -> AnyResult<Vec<Option<Vec<u8>>>> {
    let mut conn = pool
        .get()
        .await
        .map_err(|e| anyhow::anyhow!("RDS get_batch: connection acquisition failed: {e}"))?;
    // Single READ-ONLY transaction for snapshot consistency (mirrors
    // rocksdb::DB::snapshot().get() in load_at9p_state_from_db).
    let tx = conn
        .transaction()
        .await
        .map_err(|e| anyhow::anyhow!("RDS get_batch: begin transaction failed: {e}"))?;
    let mut results = Vec::with_capacity(keys.len());
    for key in keys {
        let row = tx
            .query_opt("SELECT value FROM pds_kv WHERE key = $1", &[&key])
            .await
            .map_err(|e| anyhow::anyhow!("RDS get_batch: query failed: {e}"))?;
        results.push(row.map(|r| r.get::<_, Vec<u8>>(0)));
    }
    tx.commit()
        .await
        .map_err(|e| anyhow::anyhow!("RDS get_batch: commit failed: {e}"))?;
    Ok(results)
}

async fn cmd_put(pool: &deadpool_postgres::Pool, key: &[u8], value: &[u8]) -> AnyResult<()> {
    let conn = pool
        .get()
        .await
        .map_err(|e| anyhow::anyhow!("RDS put: connection acquisition failed: {e}"))?;
    conn.execute(
        "INSERT INTO pds_kv (key, value) VALUES ($1, $2)
         ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
        &[&key, &value],
    )
    .await
    .map_err(|e| anyhow::anyhow!("RDS put: execute failed: {e}"))?;
    Ok(())
}

async fn cmd_put_batch(
    pool: &deadpool_postgres::Pool,
    ops: &[(Vec<u8>, Vec<u8>)],
) -> AnyResult<()> {
    let mut conn = pool
        .get()
        .await
        .map_err(|e| anyhow::anyhow!("RDS put_batch: connection acquisition failed: {e}"))?;
    let tx = conn
        .transaction()
        .await
        .map_err(|e| anyhow::anyhow!("RDS put_batch: begin transaction failed: {e}"))?;
    {
        let stmt = tx
            .prepare_typed(
                "INSERT INTO pds_kv (key, value) VALUES ($1, $2)
                 ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                &[tokio_postgres::types::Type::BYTEA, tokio_postgres::types::Type::BYTEA],
            )
            .await
            .map_err(|e| anyhow::anyhow!("RDS put_batch: prepare statement failed: {e}"))?;
        for (key, value) in ops {
            tx.execute(&stmt, &[&key, &value])
                .await
                .map_err(|e| anyhow::anyhow!("RDS put_batch: execute failed: {e}"))?;
        }
    }
    tx.commit()
        .await
        .map_err(|e| anyhow::anyhow!("RDS put_batch: commit failed: {e}"))?;
    Ok(())
}

async fn cmd_range_scan(
    pool: &deadpool_postgres::Pool,
    start: &[u8],
    end: &[u8],
) -> AnyResult<Vec<(Vec<u8>, Vec<u8>)>> {
    let conn = pool
        .get()
        .await
        .map_err(|e| anyhow::anyhow!("RDS range_scan: connection acquisition failed: {e}"))?;
    let rows = conn
        .query(
            "SELECT key, value FROM pds_kv WHERE key >= $1 AND key < $2 ORDER BY key ASC",
            &[&start, &end],
        )
        .await
        .map_err(|e| anyhow::anyhow!("RDS range_scan: query failed: {e}"))?;
    Ok(rows
        .into_iter()
        .map(|r| (r.get::<_, Vec<u8>>(0), r.get::<_, Vec<u8>>(1)))
        .collect())
}

async fn cmd_all_pairs(pool: &deadpool_postgres::Pool) -> AnyResult<Vec<(Vec<u8>, Vec<u8>)>> {
    let conn = pool
        .get()
        .await
        .map_err(|e| anyhow::anyhow!("RDS all_pairs: connection acquisition failed: {e}"))?;
    let rows = conn
        .query("SELECT key, value FROM pds_kv ORDER BY key ASC", &[])
        .await
        .map_err(|e| anyhow::anyhow!("RDS all_pairs: query failed: {e}"))?;
    Ok(rows
        .into_iter()
        .map(|r| (r.get::<_, Vec<u8>>(0), r.get::<_, Vec<u8>>(1)))
        .collect())
}

async fn cmd_ping(pool: &deadpool_postgres::Pool) -> AnyResult<()> {
    let conn = pool
        .get()
        .await
        .map_err(|e| anyhow::anyhow!("RDS ping: connection acquisition failed: {e}"))?;
    conn.execute("SELECT 1", &[])
        .await
        .map_err(|e| anyhow::anyhow!("RDS ping: SELECT 1 failed: {e}"))?;
    Ok(())
}

/// Send an error result on whatever reply channel the command carries.
#[allow(clippy::match_same_arms)] // each arm's reply channel has a different payload type
fn send_err(cmd: PgCmd, err: anyhow::Error) -> AnyResult<()> {
    match cmd {
        PgCmd::Get { reply, .. } => {
            let _ = reply.send(Err(err));
        }
        PgCmd::GetBatch { reply, .. } => {
            let _ = reply.send(Err(err));
        }
        PgCmd::Put { reply, .. } => {
            let _ = reply.send(Err(err));
        }
        PgCmd::PutBatch { reply, .. } => {
            let _ = reply.send(Err(err));
        }
        PgCmd::RangeScan { reply, .. } => {
            let _ = reply.send(Err(err));
        }
        PgCmd::AllPairs { reply, .. } => {
            let _ = reply.send(Err(err));
        }
        PgCmd::Ping { reply, .. } => {
            let _ = reply.send(Err(err));
        }
    }
    Ok(())
}

// We need rustls-native-certs for loading the platform trust store; it's
// declared as an optional dep under pds-postgres.
