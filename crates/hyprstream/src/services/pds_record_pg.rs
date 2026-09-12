//! Postgres-backed KV store for `PdsRecordStore` (#1257).
//!
//! This module implements the RDS Multi-AZ backend for the PDS record store.
//! It is selected when the `[rds]` config section (or its role-scoped env
//! binding) is set and the `pds-postgres` cargo feature is enabled. When
//! unset, the store falls back to its local RocksDB backend.
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
//! ## Cross-AZ write safety
//!
//! The deployment premise is two PDS processes in two AZs sharing one RDS
//! instance, so process-local mutexes guard nothing. Conditional writes go
//! through [`PgKv::cas_put`], which folds the compare into SQL
//! (`UPDATE … WHERE value = $expected`, rows-affected checked) inside one
//! transaction with the rest of the write set: the row lock serializes the
//! publishers, the loser's compare matches zero rows after the winner
//! commits, and the loser reports a conflict instead of silently
//! overwriting. Snapshot-consistent multi-reads go through
//! [`PgKv::read_snapshot`] (one read-only REPEATABLE READ transaction on one
//! pooled connection — a single MVCC snapshot for every statement, never a
//! new snapshot per statement), mirroring `rocksdb::DB::snapshot()`.
//!
//! ## Failover bounds
//!
//! RDS Multi-AZ failover can leave a connection hung mid-query. Every wait
//! is bounded: deadpool `wait`/`create`/`recycle` timeouts, driver
//! `connect_timeout`, TCP keepalive + `TCP_USER_TIMEOUT`, and a server-side
//! `statement_timeout`. A dead peer surfaces as an error; it never parks a
//! caller indefinitely.
//!
//! ## Async bridge
//!
//! `deadpool-postgres` is async-only, but the existing `PdsRecordStore` API is
//! sync (the publisher runs in a thread-mode service; the resolver is called
//! from both sync and async contexts). `PgKv` owns a dedicated OS thread that
//! runs a single-worker tokio runtime + the deadpool pool. Sync callers send a
//! command over a `tokio::sync::mpsc` unbounded channel (non-async `send`) and
//! block on a `std::sync::mpsc` reply — this is runtime-context-proof (no
//! `block_on` / `blocking_recv` panics regardless of whether the caller is on
//! a tokio thread, a plain OS thread, or inside `block_in_place`). The bridge
//! `recv().await`s commands and spawns each onto the runtime, so one slow
//! statement does not serialize all PDS I/O process-wide; the bounded pool
//! caps concurrency and Postgres row locks serialize conflicting writes.

use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context as _, Result as AnyResult};

/// Pool acquisition bounds (RDS Multi-AZ failover contract): a hung failover
/// connection must surface as an error, never block `pool.get()` forever.
const POOL_WAIT_TIMEOUT: Duration = Duration::from_secs(5);
const POOL_CREATE_TIMEOUT: Duration = Duration::from_secs(10);
const POOL_RECYCLE_TIMEOUT: Duration = Duration::from_secs(5);
/// Driver-level bounds: TCP connect, keepalive detection of a dead peer, and
/// a server-side per-statement ceiling. KV operations are point gets/puts and
/// prefix-bounded scans, so 30s per statement is generous headroom, not a
/// tuning target.
const PG_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const PG_KEEPALIVE_IDLE: Duration = Duration::from_secs(30);
const PG_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(10);
const PG_KEEPALIVE_RETRIES: u32 = 3;
const PG_TCP_USER_TIMEOUT: Duration = Duration::from_secs(30);
const PG_STATEMENT_TIMEOUT_MS: u32 = 30_000;

/// Commands sent from sync callers to the async-bridge thread.
enum PgCmd {
    /// Fetch one key. `None` = key absent.
    Get {
        key: Vec<u8>,
        reply: mpsc::Sender<AnyResult<Option<Vec<u8>>>>,
    },
    /// Fetch multiple keys in one READ-ONLY REPEATABLE READ transaction
    /// (snapshot consistency, matching `rocksdb::DB::snapshot().get()` in
    /// `load_at9p_state_from_db`).
    GetBatch {
        keys: Vec<Vec<u8>>,
        reply: mpsc::Sender<AnyResult<Vec<Option<Vec<u8>>>>>,
    },
    /// Upsert one key/value.
    Put {
        key: Vec<u8>,
        value: Vec<u8>,
        reply: mpsc::Sender<AnyResult<()>>,
    },
    /// Atomic compare-and-swap: in ONE transaction, verify `cas_key`'s current
    /// value equals `expected` (`None` = key must be absent), replace it with
    /// `new_value`, and apply `upserts`/`deletes`. The compare is folded into
    /// SQL (`UPDATE … WHERE key = $1 AND value = $3` / `INSERT … ON CONFLICT
    /// DO NOTHING`), so two publishers in two AZs serialize on the row lock
    /// and the loser observes zero rows changed. Replies `false` on
    /// precondition failure (concurrent advance), `true` when committed.
    CasPut {
        cas_key: Vec<u8>,
        expected: Option<Vec<u8>>,
        new_value: Vec<u8>,
        upserts: Vec<(Vec<u8>, Vec<u8>)>,
        deletes: Vec<Vec<u8>>,
        reply: mpsc::Sender<AnyResult<bool>>,
    },
    /// Full scan: all `(key, value)` pairs, ordered by key ascending. Mirrors
    /// RocksDB's `IteratorMode::Start`.
    AllPairs {
        reply: mpsc::Sender<AnyResult<Vec<(Vec<u8>, Vec<u8>)>>>,
    },
    /// Snapshot-consistent multi-read: every range scan and point get runs in
    /// ONE read-only REPEATABLE READ transaction on ONE pooled connection, so
    /// a concurrent writer cannot interleave between the scans (the Postgres
    /// counterpart of reading a repo's records and its signed commit from one
    /// RocksDB snapshot).
    ReadSnapshot {
        ranges: Vec<(Vec<u8>, Vec<u8>)>,
        keys: Vec<Vec<u8>>,
        reply: mpsc::Sender<AnyResult<SnapshotRead>>,
    },
    /// Ping (connection liveness check).
    Ping { reply: mpsc::Sender<AnyResult<()>> },
}

/// Result of a [`PgCmd::ReadSnapshot`]: per-range ordered pairs, then per-key
/// optional values, both positional with the request.
pub(crate) struct SnapshotRead {
    pub(crate) ranges: Vec<Vec<(Vec<u8>, Vec<u8>)>>,
    pub(crate) values: Vec<Option<Vec<u8>>>,
}

/// Sync KV store backed by a deadpool-postgres pool running on a dedicated
/// thread. Clone-safe (the inner state is `Arc`).
pub(crate) struct PgKv {
    cmd_tx: tokio::sync::mpsc::UnboundedSender<PgCmd>,
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
        url: &crate::config::ValidatedRdsUrl,
        root_cert_pem: &std::path::Path,
        cell_id: &str,
    ) -> AnyResult<Self> {
        let pg_config = build_driver_config(url.driver_url(), url.dns_hostname())?;
        let pool = assemble_pool(pg_config, root_cert_pem)?;
        Self::start(pool, cell_id)
    }

    /// TEST-ONLY: connect with no TLS and no URL validation.
    ///
    /// Live tests point this at an operator-provided scratch database via
    /// `HYPRSTREAM_POSTGRES_TEST_URL_FILE` — a file path, never a direct env
    /// var, mirroring the production file-backed credential model (metal v1.1
    /// acceptance check 1). Production connections MUST go through
    /// [`PgKv::connect`], which enforces the verify-full contract.
    #[cfg(test)]
    pub(crate) fn connect_test(driver_url: &str, cell_id: &str) -> AnyResult<Self> {
        let mut pg_config: tokio_postgres::Config = driver_url
            .parse()
            .map_err(|e| anyhow::anyhow!("test Postgres URL did not parse: {e}"))?;
        apply_driver_timeouts(&mut pg_config);
        let manager = deadpool_postgres::Manager::new(pg_config, tokio_postgres::NoTls);
        let pool = build_bounded_pool(manager)?;
        Self::start(pool, cell_id)
    }

    /// Spawn the bridge over an assembled pool, then verify connectivity +
    /// migration by pinging. The ping is the FATAL-on-unavailable boundary: a
    /// failure propagates as an error, refusing startup rather than degrading
    /// to local.
    fn start(pool: deadpool_postgres::Pool, cell_id: &str) -> AnyResult<Self> {
        let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel::<PgCmd>();
        let cell_id = cell_id.to_owned();
        let thread = std::thread::Builder::new()
            .name("pds-pg-bridge".into())
            .spawn(move || bridge_main(pool, cell_id, cmd_rx))
            .context("failed to spawn pds-pg-bridge thread")?;

        let kv = Self {
            cmd_tx,
            _thread: thread,
        };
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

    /// Compare-and-swap `cas_key` to `new_value`, gated on its current value
    /// being exactly `expected` (`None` = the key must be absent), applying
    /// `upserts` and `deletes` atomically in the same transaction. Returns
    /// `Ok(false)` when the precondition does not hold — i.e. a concurrent
    /// publisher advanced the key first. This is the cross-AZ lost-update
    /// guard: the mutex-free SQL compare is the only correct serialization
    /// point when two PDS processes share one RDS instance.
    pub(crate) fn cas_put(
        &self,
        cas_key: &[u8],
        expected: Option<&[u8]>,
        new_value: &[u8],
        upserts: &[(Vec<u8>, Vec<u8>)],
        deletes: &[Vec<u8>],
    ) -> AnyResult<bool> {
        self.round_trip(|reply| PgCmd::CasPut {
            cas_key: cas_key.to_owned(),
            expected: expected.map(<[u8]>::to_owned),
            new_value: new_value.to_owned(),
            upserts: upserts.to_owned(),
            deletes: deletes.to_owned(),
            reply,
        })
    }

    pub(crate) fn all_pairs(&self) -> AnyResult<Vec<(Vec<u8>, Vec<u8>)>> {
        self.round_trip(|reply| PgCmd::AllPairs { reply })
    }

    /// Run every range scan and point get in one read-only REPEATABLE READ
    /// transaction, so a reader never observes a record scan from before a
    /// write paired with a commit from after it (or any other torn
    /// combination).
    pub(crate) fn read_snapshot(
        &self,
        ranges: &[(Vec<u8>, Vec<u8>)],
        keys: &[Vec<u8>],
    ) -> AnyResult<SnapshotRead> {
        self.round_trip(|reply| PgCmd::ReadSnapshot {
            ranges: ranges.to_owned(),
            keys: keys.to_owned(),
            reply,
        })
    }

    pub(crate) fn ping(&self) -> AnyResult<()> {
        self.round_trip(|reply| PgCmd::Ping { reply })
    }

    fn round_trip<R, F>(&self, make_cmd: F) -> AnyResult<R>
    where
        R: Send + 'static,
        F: FnOnce(mpsc::Sender<AnyResult<R>>) -> PgCmd,
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
    pool: deadpool_postgres::Pool,
    cell_id: String,
    mut cmd_rx: tokio::sync::mpsc::UnboundedReceiver<PgCmd>,
) {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            tracing::error!("pds-pg-bridge: failed to create runtime: {e}");
            // Drain all pending commands with the error so callers fail
            // closed. No runtime exists here, so `blocking_recv` is legal.
            while let Some(cmd) = cmd_rx.blocking_recv() {
                let _ = send_err(cmd, anyhow::anyhow!("runtime creation failed: {e}"));
            }
            return;
        }
    };

    rt.block_on(async move {
        // Run the idempotent schema migration. This is the FATAL boundary.
        if let Err(e) = migrate(&pool, &cell_id).await {
            tracing::error!("pds-pg-bridge: schema migration failed: {e}");
            while let Some(cmd) = cmd_rx.recv().await {
                let _ = send_err(cmd, anyhow::anyhow!("schema migration failed: {e}"));
            }
            return;
        }

        tracing::info!("pds-pg-bridge: connected to RDS, schema migrated");

        // Serve commands until the sender half is dropped. Each command runs
        // on its own spawned task so one slow statement does not serialize
        // all PDS I/O process-wide; the bounded pool caps concurrency and
        // Postgres row locks serialize conflicting writes (see `cmd_cas_put`).
        while let Some(cmd) = cmd_rx.recv().await {
            let pool = pool.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_cmd(&pool, cmd).await {
                    tracing::error!("pds-pg-bridge: command handler error: {e}");
                    // The error was already sent to the caller inside
                    // handle_cmd; this is just a log.
                }
            });
        }
    });
}

/// Build a bounded pool with the failover contract applied: capped size and
/// `wait`/`create`/`recycle` timeouts so a hung RDS failover connection
/// surfaces as an error instead of blocking `pool.get()` forever.
fn build_bounded_pool(manager: deadpool_postgres::Manager) -> AnyResult<deadpool_postgres::Pool> {
    deadpool_postgres::Pool::builder(manager)
        .runtime(deadpool_postgres::Runtime::Tokio1)
        // Bounded pool: each PDS process holds at most this many server
        // connections; callers queue behind deadpool's wait timeout instead
        // of growing the RDS connection count unboundedly.
        .max_size(2 * num_cpus::get())
        .wait_timeout(Some(POOL_WAIT_TIMEOUT))
        .create_timeout(Some(POOL_CREATE_TIMEOUT))
        .recycle_timeout(Some(POOL_RECYCLE_TIMEOUT))
        .build()
        .map_err(|e| anyhow::anyhow!("deadpool-postgres pool creation failed: {e}"))
}

/// Driver-level failover bounds: a dead peer must be detected by the
/// transport, and no single statement may run unboundedly server-side.
fn apply_driver_timeouts(pg_config: &mut tokio_postgres::Config) {
    pg_config
        .connect_timeout(PG_CONNECT_TIMEOUT)
        .keepalives(true)
        .keepalives_idle(PG_KEEPALIVE_IDLE)
        .keepalives_interval(PG_KEEPALIVE_INTERVAL)
        .keepalives_retries(PG_KEEPALIVE_RETRIES)
        .tcp_user_timeout(PG_TCP_USER_TIMEOUT)
        .options(format!("-c statement_timeout={PG_STATEMENT_TIMEOUT_MS}"));
}

/// Assemble the production rustls connector and deadpool pool. The live TLS
/// tests below use it with an explicit loopback `hostaddr` only to direct the
/// test TCP peer while retaining the production hostname as SNI and
/// certificate-verification name.
fn assemble_pool(
    pg_config: tokio_postgres::Config,
    root_cert: &std::path::Path,
) -> AnyResult<deadpool_postgres::Pool> {
    let tls = tokio_postgres_rustls::MakeRustlsConnect::new(build_rustls_config(root_cert)?);
    let manager = deadpool_postgres::Manager::new(pg_config, tls);
    build_bounded_pool(manager)
}

/// Parse the translated URL with the pinned driver and reassert the effective
/// transport policy. This prevents query-level `host`/`hostaddr` overrides or
/// hostless defaults from changing the DNS endpoint validated by `RdsConfig`.
fn build_driver_config(driver_url: &str, dns_hostname: &str) -> AnyResult<tokio_postgres::Config> {
    // Match the #1401 user-store path: never pass verify-full to tokio-postgres
    // (0.7 accepts only disable/prefer/require). Extract only the connection
    // fields from the already-validated URL and set Require directly. This also
    // prevents query parameters such as host/hostaddr from overriding the
    // validated endpoint or SNI name.
    let url = url::Url::parse(driver_url)
        .map_err(|e| anyhow::anyhow!("failed to reparse validated RDS URL: {e}"))?;
    // Mirror `validate_url`'s arms exactly: domain verbatim, IP literals in
    // BARE form. (`Host`'s own `Display` adds brackets for IPv6 — comparing
    // against it rejected every contract-valid IPv6 endpoint.)
    let reparsed_host = match url.host() {
        Some(url::Host::Domain(domain)) => domain.to_owned(),
        Some(url::Host::Ipv4(addr)) => addr.to_string(),
        Some(url::Host::Ipv6(addr)) => addr.to_string(),
        None => anyhow::bail!("validated RDS URL lost its host"),
    };
    anyhow::ensure!(
        reparsed_host == dns_hostname,
        "validated RDS URL changed the contract hostname"
    );

    let mut config = tokio_postgres::Config::new();
    config.host(dns_hostname);
    if let Some(port) = url.port() {
        config.port(port);
    }
    if !url.username().is_empty() {
        config.user(url.username());
    }
    if let Some(password) = url.password() {
        let password = urlencoding::decode(password).map_err(|e| {
            anyhow::anyhow!("validated RDS password has invalid percent encoding: {e}")
        })?;
        config.password(password.into_owned());
    }
    let dbname = url.path().trim_start_matches('/');
    if !dbname.is_empty() {
        config.dbname(dbname);
    }
    for (key, value) in url.query_pairs() {
        if key == "application_name" {
            config.application_name(value.into_owned());
        }
    }
    config.ssl_mode(tokio_postgres::config::SslMode::Require);
    apply_driver_timeouts(&mut config);

    Ok(config)
}

/// Build the rustls connector used by PostgreSQL. The explicit records CA is
/// mandatory and is the pinned trust store. Standard WebPKI certificate and
/// hostname verification remains enabled.
fn build_rustls_config(root_cert_pem: &std::path::Path) -> AnyResult<rustls::ClientConfig> {
    let mut root_store = rustls::RootCertStore::empty();

    let pem = std::fs::read(root_cert_pem).map_err(|e| {
        anyhow::anyhow!("failed to read RDS root_cert_file at {root_cert_pem:?}: {e}")
    })?;
    let certs = rustls_pemfile::certs(&mut pem.as_slice())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| {
            anyhow::anyhow!("failed to parse RDS root_cert_file PEM at {root_cert_pem:?}: {e}")
        })?;
    anyhow::ensure!(
        !certs.is_empty(),
        "RDS root_cert_file at {root_cert_pem:?} contains no certificates"
    );
    for cert in certs {
        root_store
            .add(cert)
            .map_err(|e| anyhow::anyhow!("failed to add RDS root certificate: {e}"))?;
    }

    let roots = Arc::new(root_store);
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let verifier = rustls::client::WebPkiServerVerifier::builder_with_provider(
        Arc::clone(&roots),
        Arc::clone(&provider),
    )
    .build()
    .map_err(|e| anyhow::anyhow!("failed to build standard RDS WebPKI verifier: {e}"))?;
    Ok(rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| anyhow::anyhow!("failed to select safe RDS TLS protocol versions: {e}"))?
        .with_webpki_verifier(verifier)
        .with_no_client_auth())
}

/// Idempotent schema migration.
///
/// One table: `pds_kv (key BYTEA PRIMARY KEY, value BYTEA NOT NULL)`. The
/// `cell_id` is stamped into a `pds_meta` row for the honorable-mention cell
/// guard. No queryable columns are derived from the signed bytes — D2.
///
/// Runs under a transaction-scoped advisory lock: two PDS processes booting
/// against a fresh RDS instance (the two-AZ deployment, and parallel live
/// tests) otherwise race `CREATE TABLE IF NOT EXISTS`, which Postgres can
/// fail with a catalog duplicate-key error. The lock serializes the
/// migrators; the loser proceeds once the winner's transaction commits.
async fn migrate(pool: &deadpool_postgres::Pool, cell_id: &str) -> AnyResult<()> {
    /// Advisory lock ID for the pds_kv schema migration (arbitrary, stable).
    const MIGRATION_LOCK_ID: i64 = 0x1257_1257_1257_1257;

    let mut conn = pool
        .get()
        .await
        .map_err(|e| anyhow::anyhow!("RDS connection acquisition failed during migration: {e}"))?;
    let tx = conn
        .transaction()
        .await
        .map_err(|e| anyhow::anyhow!("RDS migration: begin transaction failed: {e}"))?;

    tx.execute("SELECT pg_advisory_xact_lock($1)", &[&MIGRATION_LOCK_ID])
        .await
        .map_err(|e| anyhow::anyhow!("RDS migration: advisory lock failed: {e}"))?;

    tx.batch_execute(
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
    tx.execute(
        "INSERT INTO pds_meta (k, v) VALUES ('cell_id', $1)
         ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v",
        &[&cell_id],
    )
    .await
    .map_err(|e| anyhow::anyhow!("RDS cell_id stamp failed: {e}"))?;

    tx.commit()
        .await
        .map_err(|e| anyhow::anyhow!("RDS migration: commit failed: {e}"))?;
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
        PgCmd::CasPut {
            cas_key,
            expected,
            new_value,
            upserts,
            deletes,
            reply,
        } => {
            let r = cmd_cas_put(pool, &cas_key, expected, &new_value, &upserts, &deletes).await;
            let _ = reply.send(r);
        }
        PgCmd::AllPairs { reply } => {
            let r = cmd_all_pairs(pool).await;
            let _ = reply.send(r);
        }
        PgCmd::ReadSnapshot {
            ranges,
            keys,
            reply,
        } => {
            let r = cmd_read_snapshot(pool, &ranges, &keys).await;
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

/// Begin the single read-only transaction every multi-statement read runs in.
///
/// REPEATABLE READ is the load-bearing part: READ COMMITTED (the Postgres
/// default) takes a **new MVCC snapshot per statement**, so a record range
/// scan and a subsequent commit point-get could observe different commits —
/// a torn read that would let a publisher CAS a head that does not cover a
/// competitor's concurrently-committed record. One snapshot per transaction
/// is the exact counterpart of `rocksdb::DB::snapshot()`.
async fn begin_snapshot_tx<'a>(
    conn: &'a mut deadpool_postgres::Object,
    op: &str,
) -> AnyResult<deadpool_postgres::Transaction<'a>> {
    conn.build_transaction()
        .read_only(true)
        .isolation_level(tokio_postgres::IsolationLevel::RepeatableRead)
        .start()
        .await
        .map_err(|e| anyhow::anyhow!("RDS {op}: begin snapshot transaction failed: {e}"))
}

async fn cmd_get_batch(
    pool: &deadpool_postgres::Pool,
    keys: &[Vec<u8>],
) -> AnyResult<Vec<Option<Vec<u8>>>> {
    let mut conn = pool
        .get()
        .await
        .map_err(|e| anyhow::anyhow!("RDS get_batch: connection acquisition failed: {e}"))?;
    // Single READ-ONLY, REPEATABLE READ transaction for snapshot consistency
    // (mirrors rocksdb::DB::snapshot().get() in load_at9p_state_from_db).
    let tx = begin_snapshot_tx(&mut conn, "get_batch").await?;
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

/// The SQL heart of cross-AZ write safety. The compare rides the same row
/// lock as the write, so two publishers in two AZs cannot both pass the
/// precondition: whoever commits first flips the bytes, and the loser's
/// conditional `UPDATE`/`INSERT` matches zero rows.
async fn cmd_cas_put(
    pool: &deadpool_postgres::Pool,
    cas_key: &[u8],
    expected: Option<Vec<u8>>,
    new_value: &[u8],
    upserts: &[(Vec<u8>, Vec<u8>)],
    deletes: &[Vec<u8>],
) -> AnyResult<bool> {
    let mut conn = pool
        .get()
        .await
        .map_err(|e| anyhow::anyhow!("RDS cas_put: connection acquisition failed: {e}"))?;
    let tx = conn
        .transaction()
        .await
        .map_err(|e| anyhow::anyhow!("RDS cas_put: begin transaction failed: {e}"))?;
    let changed = match &expected {
        Some(expected) => tx
            .execute(
                "UPDATE pds_kv SET value = $2 WHERE key = $1 AND value = $3",
                &[&cas_key, &new_value, expected],
            )
            .await
            .map_err(|e| anyhow::anyhow!("RDS cas_put: conditional update failed: {e}"))?,
        None => tx
            .execute(
                "INSERT INTO pds_kv (key, value) VALUES ($1, $2)
                 ON CONFLICT (key) DO NOTHING",
                &[&cas_key, &new_value],
            )
            .await
            .map_err(|e| anyhow::anyhow!("RDS cas_put: insert-if-absent failed: {e}"))?,
    };
    if changed == 0 {
        tx.rollback()
            .await
            .map_err(|e| anyhow::anyhow!("RDS cas_put: conflict rollback failed: {e}"))?;
        return Ok(false);
    }
    if !upserts.is_empty() {
        let stmt = tx
            .prepare_typed(
                "INSERT INTO pds_kv (key, value) VALUES ($1, $2)
                 ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                &[
                    tokio_postgres::types::Type::BYTEA,
                    tokio_postgres::types::Type::BYTEA,
                ],
            )
            .await
            .map_err(|e| anyhow::anyhow!("RDS cas_put: prepare upsert failed: {e}"))?;
        for (key, value) in upserts {
            tx.execute(&stmt, &[&key, &value])
                .await
                .map_err(|e| anyhow::anyhow!("RDS cas_put: upsert failed: {e}"))?;
        }
    }
    for key in deletes {
        tx.execute("DELETE FROM pds_kv WHERE key = $1", &[&key])
            .await
            .map_err(|e| anyhow::anyhow!("RDS cas_put: delete failed: {e}"))?;
    }
    tx.commit()
        .await
        .map_err(|e| anyhow::anyhow!("RDS cas_put: commit failed: {e}"))?;
    Ok(true)
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

async fn cmd_read_snapshot(
    pool: &deadpool_postgres::Pool,
    ranges: &[(Vec<u8>, Vec<u8>)],
    keys: &[Vec<u8>],
) -> AnyResult<SnapshotRead> {
    let mut conn = pool
        .get()
        .await
        .map_err(|e| anyhow::anyhow!("RDS read_snapshot: connection acquisition failed: {e}"))?;
    let tx = begin_snapshot_tx(&mut conn, "read_snapshot").await?;
    let mut range_results = Vec::with_capacity(ranges.len());
    for (start, end) in ranges {
        let rows = tx
            .query(
                "SELECT key, value FROM pds_kv WHERE key >= $1 AND key < $2 ORDER BY key ASC",
                &[&start, &end],
            )
            .await
            .map_err(|e| anyhow::anyhow!("RDS read_snapshot: range scan failed: {e}"))?;
        range_results.push(
            rows.into_iter()
                .map(|r| (r.get::<_, Vec<u8>>(0), r.get::<_, Vec<u8>>(1)))
                .collect(),
        );
    }
    let mut values = Vec::with_capacity(keys.len());
    for key in keys {
        let row = tx
            .query_opt("SELECT value FROM pds_kv WHERE key = $1", &[&key])
            .await
            .map_err(|e| anyhow::anyhow!("RDS read_snapshot: point get failed: {e}"))?;
        values.push(row.map(|r| r.get::<_, Vec<u8>>(0)));
    }
    tx.commit()
        .await
        .map_err(|e| anyhow::anyhow!("RDS read_snapshot: commit failed: {e}"))?;
    Ok(SnapshotRead {
        ranges: range_results,
        values,
    })
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
        PgCmd::CasPut { reply, .. } => {
            let _ = reply.send(Err(err));
        }
        PgCmd::AllPairs { reply, .. } => {
            let _ = reply.send(Err(err));
        }
        PgCmd::ReadSnapshot { reply, .. } => {
            let _ = reply.send(Err(err));
        }
        PgCmd::Ping { reply, .. } => {
            let _ = reply.send(Err(err));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read the test database URL from a FILE, never from a direct env var.
    /// This mirrors the production file-backed credential model — even in
    /// tests, a password-bearing URL must not transit the process
    /// environment (metal v1.1 acceptance check 1).
    fn test_url() -> Option<String> {
        let path = std::env::var_os("HYPRSTREAM_POSTGRES_TEST_URL_FILE")?;
        let url = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read test URL file {}: {e}", path.to_string_lossy()));
        let url = url.trim();
        assert!(!url.is_empty(), "test URL file is empty");
        Some(url.to_owned())
    }

    /// Skip a DB-backed test unless the operator opted in via a URL file.
    /// Keeps `cargo test --features pds-postgres` green with no Postgres.
    macro_rules! require_db {
        () => {{
            let Some(url) = test_url() else {
                eprintln!("skipping pds_record_pg DB test: HYPRSTREAM_POSTGRES_TEST_URL_FILE unset");
                return;
            };
            url
        }};
    }

    /// A key prefix unique to this process run, so tests against a shared
    /// scratch database never collide with prior runs or each other.
    fn run_unique_key(label: &str) -> Vec<u8> {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("test-kv\0{label}\0{}\0{nanos}\0", std::process::id()).into_bytes()
    }

    fn translated_url(url: &str) -> crate::config::ValidatedRdsUrl {
        crate::config::RdsConfig::validate_url(url)
            .unwrap_or_else(|e| panic!("valid contract URL rejected: {e}"))
    }

    #[test]
    fn valid_contract_url_builds_usable_require_driver_config_without_network() {
        let validated = translated_url(
            "postgresql://records:secret@db.internal.example:5432/records?sslmode=verify-full",
        );
        let config = build_driver_config(validated.driver_url(), validated.dns_hostname())
            .unwrap_or_else(|e| panic!("driver config rejected: {e}"));

        assert_eq!(
            config.get_ssl_mode(),
            tokio_postgres::config::SslMode::Require
        );
        assert_eq!(
            config.get_hosts(),
            [tokio_postgres::config::Host::Tcp(
                "db.internal.example".to_owned()
            )]
        );
        assert!(config.get_hostaddrs().is_empty());
        assert_eq!(
            config.get_connect_timeout(),
            Some(&PG_CONNECT_TIMEOUT),
            "failover contract: connect timeout must be bounded"
        );
        let options = config
            .get_options()
            .unwrap_or_else(|| panic!("statement_timeout options missing"));
        assert!(
            options.contains("statement_timeout="),
            "failover contract: server-side statement timeout must be set: {options}"
        );
    }

    #[test]
    fn driver_config_ignores_query_host_and_hostaddr_overrides() {
        for url in [
            "postgresql://records:secret@db.internal.example/records?host=/tmp&sslmode=verify-full",
            "postgresql://records:secret@db.internal.example/records?hostaddr=127.0.0.1&sslmode=verify-full",
        ] {
            let validated = translated_url(url);
            let config = build_driver_config(validated.driver_url(), validated.dns_hostname())
                .unwrap_or_else(|e| panic!("driver config rejected valid URL: {e}"));
            assert_eq!(
                config.get_hosts(),
                [tokio_postgres::config::Host::Tcp(
                    "db.internal.example".to_owned()
                )]
            );
            assert!(config.get_hostaddrs().is_empty());
        }
    }

    /// Contract and driver must agree on non-loopback IP endpoints
    /// (review P2-7): `validate_url` accepts them and stores the BARE host,
    /// so `build_driver_config` must not reject them on a bracketed
    /// `host_str()` comparison, and the connector must target the bare host
    /// (DNS/SNI form for IPs).
    #[test]
    fn contract_ip_endpoints_build_driver_config_on_the_bare_host() {
        for (url, bare_host) in [
            (
                "postgresql://records:secret@192.0.2.10:5432/records?sslmode=verify-full",
                "192.0.2.10",
            ),
            (
                "postgresql://records:secret@[2001:db8::10]:5432/records?sslmode=verify-full",
                "2001:db8::10",
            ),
        ] {
            let validated = translated_url(url);
            assert_eq!(validated.dns_hostname(), bare_host);
            let config = build_driver_config(validated.driver_url(), validated.dns_hostname())
                .unwrap_or_else(|e| panic!("driver config rejected contract IP URL {url}: {e}"));
            assert_eq!(
                config.get_hosts(),
                [tokio_postgres::config::Host::Tcp(bare_host.to_owned())],
                "the connector must target the validated bare host for {url}"
            );
            assert!(config.get_hostaddrs().is_empty());
        }
    }

    #[test]
    fn pool_builder_carries_bounded_failover_timeouts() {
        // Construction only — no connection is opened. This locks the
        // failover contract: max_size + wait/create/recycle timeouts.
        let pg_config = tokio_postgres::Config::new();
        let manager = deadpool_postgres::Manager::new(pg_config, tokio_postgres::NoTls);
        let pool = build_bounded_pool(manager).unwrap_or_else(|e| panic!("pool rejected: {e}"));
        let timeouts = pool.timeouts();
        assert_eq!(timeouts.wait, Some(POOL_WAIT_TIMEOUT));
        assert_eq!(timeouts.create, Some(POOL_CREATE_TIMEOUT));
        assert_eq!(timeouts.recycle, Some(POOL_RECYCLE_TIMEOUT));
        assert!(pool.status().max_size >= 2);
        drop(pool);
    }

    #[test]
    fn prefix_upper_bound_computes_successor_keys() {
        assert_eq!(prefix_upper_bound(b"abc"), Some(b"abd".to_vec()));
        assert_eq!(prefix_upper_bound(b"ab\xff"), Some(b"ac".to_vec()));
        assert_eq!(prefix_upper_bound(b"\x00"), Some(b"\x01".to_vec()));
        assert_eq!(prefix_upper_bound(&[0x61, 0x00]), Some(vec![0x61, 0x01]));
        // All-0xFF prefixes (and the empty prefix) cover the rest of the
        // keyspace — no upper bound exists.
        assert_eq!(prefix_upper_bound(b"\xff\xff"), None);
        assert_eq!(prefix_upper_bound(b""), None);
    }

    #[test]
    fn live_put_get_roundtrip_and_absent() {
        let url = require_db!();
        let kv = PgKv::connect_test(&url, "test-cell")
            .unwrap_or_else(|e| panic!("connect: {e}"));
        let key = run_unique_key("put-get");
        assert_eq!(
            kv.get(&key).unwrap_or_else(|e| panic!("get absent: {e}")),
            None
        );
        kv.put(&key, b"value-1")
            .unwrap_or_else(|e| panic!("put: {e}"));
        assert_eq!(
            kv.get(&key).unwrap_or_else(|e| panic!("get: {e}")),
            Some(b"value-1".to_vec())
        );
        // Upsert overwrites.
        kv.put(&key, b"value-2")
            .unwrap_or_else(|e| panic!("re-put: {e}"));
        assert_eq!(
            kv.get(&key).unwrap_or_else(|e| panic!("re-get: {e}")),
            Some(b"value-2".to_vec())
        );
    }

    #[test]
    fn live_get_batch_is_positional_and_reports_absence() {
        let url = require_db!();
        let kv = PgKv::connect_test(&url, "test-cell")
            .unwrap_or_else(|e| panic!("connect: {e}"));
        let base = run_unique_key("batch");
        let key_a = [&base[..], b"a"].concat();
        let key_b = [&base[..], b"b"].concat();
        let key_absent = [&base[..], b"absent"].concat();
        kv.put(&key_a, b"va").unwrap_or_else(|e| panic!("put a: {e}"));
        kv.put(&key_b, b"vb").unwrap_or_else(|e| panic!("put b: {e}"));
        let got = kv
            .get_batch(&[key_b.clone(), key_absent, key_a.clone()])
            .unwrap_or_else(|e| panic!("get_batch: {e}"));
        assert_eq!(
            got,
            vec![Some(b"vb".to_vec()), None, Some(b"va".to_vec())],
            "get_batch must be positional and report absence per key"
        );
    }

    #[test]
    fn live_cas_put_first_writer_wins() {
        let url = require_db!();
        let writer_a = PgKv::connect_test(&url, "test-cell")
            .unwrap_or_else(|e| panic!("connect A: {e}"));
        // A second pool = a second PDS process in another AZ.
        let writer_b = PgKv::connect_test(&url, "test-cell")
            .unwrap_or_else(|e| panic!("connect B: {e}"));
        let key = run_unique_key("cas");

        // Genesis: both expect the key absent — exactly one wins.
        assert!(
            writer_a
                .cas_put(&key, None, b"h1", &[], &[])
                .unwrap_or_else(|e| panic!("cas A genesis: {e}")),
            "first genesis CAS must commit"
        );
        assert!(
            !writer_b
                .cas_put(&key, None, b"h1-b", &[], &[])
                .unwrap_or_else(|e| panic!("cas B genesis: {e}")),
            "second genesis CAS must report the lost race"
        );

        // Advance with the correct expected bytes commits.
        assert!(
            writer_b
                .cas_put(&key, Some(b"h1"), b"h2", &[], &[])
                .unwrap_or_else(|e| panic!("cas B advance: {e}")),
            "CAS with the current expected value must commit"
        );
        // Advance with a stale expectation is rejected — no lost update.
        assert!(
            !writer_a
                .cas_put(&key, Some(b"h1"), b"h2-stale", &[], &[])
                .unwrap_or_else(|e| panic!("cas A stale: {e}")),
            "CAS with a stale expected value must be rejected"
        );
        assert_eq!(
            writer_a.get(&key).unwrap_or_else(|e| panic!("get: {e}")),
            Some(b"h2".to_vec()),
            "the winner's value — and only the winner's — is durable"
        );
    }

    #[test]
    fn live_cas_put_atomic_side_effects() {
        let url = require_db!();
        let kv = PgKv::connect_test(&url, "test-cell")
            .unwrap_or_else(|e| panic!("connect: {e}"));
        let key = run_unique_key("cas-atomic");
        let side = run_unique_key("cas-atomic-side");
        let marker = run_unique_key("cas-atomic-marker");
        kv.put(&marker, b"pending")
            .unwrap_or_else(|e| panic!("seed marker: {e}"));

        // Committing CAS: upsert and delete ride the same transaction.
        assert!(
            kv.cas_put(
                &key,
                None,
                b"h1",
                &[(side.clone(), b"side-v".to_vec())],
                std::slice::from_ref(&marker),
            )
            .unwrap_or_else(|e| panic!("cas commit: {e}")),
            "genesis CAS with side effects must commit"
        );
        assert_eq!(
            kv.get(&side).unwrap_or_else(|e| panic!("get side: {e}")),
            Some(b"side-v".to_vec()),
            "the upsert landed with the CAS"
        );
        assert_eq!(
            kv.get(&marker).unwrap_or_else(|e| panic!("get marker: {e}")),
            None,
            "the delete landed with the CAS"
        );

        // Failing CAS: no side effects leak.
        let side2 = run_unique_key("cas-atomic-side2");
        let marker2 = run_unique_key("cas-atomic-marker2");
        kv.put(&marker2, b"pending")
            .unwrap_or_else(|e| panic!("seed marker2: {e}"));
        assert!(
            !kv.cas_put(
                &key,
                Some(b"wrong-expected"),
                b"h2",
                &[(side2.clone(), b"side2-v".to_vec())],
                std::slice::from_ref(&marker2),
            )
            .unwrap_or_else(|e| panic!("cas conflict: {e}")),
            "stale CAS must be rejected"
        );
        assert_eq!(
            kv.get(&side2).unwrap_or_else(|e| panic!("get side2: {e}")),
            None,
            "a rejected CAS must not apply its upserts"
        );
        assert_eq!(
            kv.get(&marker2).unwrap_or_else(|e| panic!("get marker2: {e}")),
            Some(b"pending".to_vec()),
            "a rejected CAS must not apply its deletes"
        );
    }

    #[test]
    fn live_two_handles_share_one_store() {
        let url = require_db!();
        // Two independent bridges over two pools: the in-process model of the
        // publisher (AZ-a) and resolver (AZ-b) sharing one RDS instance.
        let az_a = PgKv::connect_test(&url, "test-cell")
            .unwrap_or_else(|e| panic!("connect AZ-a: {e}"));
        let az_b = PgKv::connect_test(&url, "test-cell")
            .unwrap_or_else(|e| panic!("connect AZ-b: {e}"));
        let key = run_unique_key("two-handle");
        az_a.put(&key, b"cross-az-value")
            .unwrap_or_else(|e| panic!("write via AZ-a: {e}"));
        assert_eq!(
            az_b.get(&key).unwrap_or_else(|e| panic!("read via AZ-b: {e}")),
            Some(b"cross-az-value".to_vec()),
            "a write through one handle must be visible through the other"
        );
    }

    #[test]
    fn live_read_snapshot_is_consistent() {
        let url = require_db!();
        let kv = PgKv::connect_test(&url, "test-cell")
            .unwrap_or_else(|e| panic!("connect: {e}"));
        let base = run_unique_key("snap");
        let mk = |suffix: &[u8]| [&base[..], suffix].concat();
        kv.put(&mk(b"\x01"), b"v1")
            .unwrap_or_else(|e| panic!("seed 1: {e}"));
        kv.put(&mk(b"\x02"), b"v2")
            .unwrap_or_else(|e| panic!("seed 2: {e}"));
        kv.put(&mk(b"\x03"), b"v3")
            .unwrap_or_else(|e| panic!("seed 3: {e}"));
        let head_key = run_unique_key("snap-head");
        kv.put(&head_key, b"head")
            .unwrap_or_else(|e| panic!("seed head: {e}"));

        // Prefix scan via prefix_upper_bound, plus a point get — one snapshot.
        let upper = prefix_upper_bound(&base)
            .unwrap_or_else(|| panic!("test prefix must have an upper bound"));
        let snap = kv
            .read_snapshot(
                &[(base.clone(), upper.clone())],
                std::slice::from_ref(&head_key),
            )
            .unwrap_or_else(|e| panic!("read_snapshot: {e}"));
        assert_eq!(snap.ranges.len(), 1);
        assert_eq!(
            snap.ranges.first().unwrap_or(&Vec::new()),
            &vec![
                (mk(b"\x01"), b"v1".to_vec()),
                (mk(b"\x02"), b"v2".to_vec()),
                (mk(b"\x03"), b"v3".to_vec()),
            ],
            "range scan must return exactly the prefixed keys in ascending order"
        );
        assert_eq!(snap.values, vec![Some(b"head".to_vec())]);

        // Sub-range semantics: start inclusive, end exclusive.
        let snap = kv
            .read_snapshot(&[(mk(b"\x02"), mk(b"\x04"))], &[])
            .unwrap_or_else(|e| panic!("sub-range: {e}"));
        assert_eq!(
            snap.ranges.first().unwrap_or(&Vec::new()),
            &vec![
                (mk(b"\x02"), b"v2".to_vec()),
                (mk(b"\x03"), b"v3".to_vec()),
            ]
        );
        let snap = kv
            .read_snapshot(&[(mk(b"\x04"), upper)], &[])
            .unwrap_or_else(|e| panic!("empty range: {e}"));
        assert!(snap.ranges.first().unwrap_or(&Vec::new()).is_empty());

        // Absent range + absent key read as empty/None, not as an error.
        let other = run_unique_key("snap-empty");
        let other_upper = prefix_upper_bound(&other)
            .unwrap_or_else(|| panic!("test prefix must have an upper bound"));
        let snap = kv
            .read_snapshot(&[(other, other_upper)], &[run_unique_key("snap-none")])
            .unwrap_or_else(|e| panic!("read_snapshot empty: {e}"));
        assert!(snap.ranges.first().unwrap_or(&Vec::new()).is_empty());
        assert_eq!(snap.values, vec![None]);
    }

    /// Deterministic isolation pin: inside one `begin_snapshot_tx`
    /// transaction, a commit from another connection landing BETWEEN two
    /// identical reads must not change the second read. READ COMMITTED (the
    /// Postgres default) fails this test — it takes a new snapshot per
    /// statement. This is the seam every multi-statement read
    /// (`cmd_get_batch`, `cmd_read_snapshot`) shares.
    #[tokio::test]
    async fn live_snapshot_transaction_is_repeatable_read() {
        let url = require_db!();
        // The writer handle also guarantees the schema exists.
        let writer = PgKv::connect_test(&url, "test-cell")
            .unwrap_or_else(|e| panic!("connect writer: {e}"));
        let key = run_unique_key("rr-tx");
        writer
            .put(&key, b"v1")
            .unwrap_or_else(|e| panic!("seed: {e}"));

        let pg_config: tokio_postgres::Config = url
            .parse()
            .unwrap_or_else(|e| panic!("test Postgres URL did not parse: {e}"));
        let pool = build_bounded_pool(deadpool_postgres::Manager::new(
            pg_config,
            tokio_postgres::NoTls,
        ))
        .unwrap_or_else(|e| panic!("pool: {e}"));
        let mut conn = pool
            .get()
            .await
            .unwrap_or_else(|e| panic!("pool get: {e}"));
        let tx = begin_snapshot_tx(&mut conn, "test")
            .await
            .unwrap_or_else(|e| panic!("begin snapshot tx: {e}"));
        let first = tx
            .query_opt("SELECT value FROM pds_kv WHERE key = $1", &[&key])
            .await
            .unwrap_or_else(|e| panic!("first read: {e}"));
        // A writer on another connection commits a new value mid-transaction.
        writer
            .put(&key, b"v2")
            .unwrap_or_else(|e| panic!("concurrent put: {e}"));
        let second = tx
            .query_opt("SELECT value FROM pds_kv WHERE key = $1", &[&key])
            .await
            .unwrap_or_else(|e| panic!("second read: {e}"));
        assert_eq!(
            first.as_ref().map(|r| r.get::<_, Vec<u8>>(0)),
            Some(b"v1".to_vec()),
            "the snapshot must start from the pre-write value"
        );
        let second = second.map(|r| r.get::<_, Vec<u8>>(0));
        assert_eq!(
            second,
            Some(b"v1".to_vec()),
            "REPEATABLE READ must not observe a mid-transaction commit; \
             READ COMMITTED would return v2 here (a torn read)"
        );
        tx.commit().await.unwrap_or_else(|e| panic!("commit: {e}"));
    }

    /// Command-level pin: `get_batch` repeats the same key 32 times inside
    /// ONE snapshot transaction while a second handle flips the value as fast
    /// as the wire allows. Every copy within a single batch response must be
    /// identical. This can never fail spuriously under REPEATABLE READ; under
    /// READ COMMITTED a flip lands mid-batch with overwhelming probability.
    #[test]
    fn live_get_batch_never_tears_under_a_concurrent_writer() {
        let url = require_db!();
        let reader = PgKv::connect_test(&url, "test-cell")
            .unwrap_or_else(|e| panic!("connect reader: {e}"));
        let writer = PgKv::connect_test(&url, "test-cell")
            .unwrap_or_else(|e| panic!("connect writer: {e}"));
        let key = run_unique_key("rr-batch");
        writer
            .put(&key, b"v1")
            .unwrap_or_else(|e| panic!("seed: {e}"));

        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let writer_thread = {
            let stop = std::sync::Arc::clone(&stop);
            let key = key.clone();
            std::thread::spawn(move || {
                let mut flip = false;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let value: &[u8] = if flip { b"v1" } else { b"v2" };
                    writer
                        .put(&key, value)
                        .unwrap_or_else(|e| panic!("flip write: {e}"));
                    flip = !flip;
                }
            })
        };
        let keys: Vec<Vec<u8>> = std::iter::repeat_with(|| key.clone()).take(32).collect();
        let mut seen = std::collections::BTreeSet::new();
        for round in 0..50 {
            let batch = reader
                .get_batch(&keys)
                .unwrap_or_else(|e| panic!("get_batch round {round}: {e}"));
            assert!(
                batch.windows(2).all(|pair| pair[0] == pair[1]),
                "torn read in round {round}: one batch observed both sides of \
                 a concurrent commit: {batch:?}"
            );
            seen.insert(batch.into_iter().next().flatten());
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        writer_thread
            .join()
            .unwrap_or_else(|_| panic!("writer thread panicked"));
        assert!(
            seen.len() > 1,
            "the concurrent writer never became visible — the race was not exercised"
        );
    }

    #[test]
    fn explicit_ca_is_loaded_into_rustls_connector() {
        let dir = tempfile::TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        let ca_path = dir.path().join("rds-ca.pem");
        let certified = rcgen::generate_simple_self_signed(vec!["db.internal.example".to_owned()])
            .unwrap_or_else(|e| panic!("{e}"));
        std::fs::write(&ca_path, certified.cert.pem()).unwrap_or_else(|e| panic!("{e}"));

        let client_config = build_rustls_config(&ca_path)
            .unwrap_or_else(|e| panic!("explicit CA was not loadable: {e}"));
        let _connector = tokio_postgres_rustls::MakeRustlsConnect::new(client_config);
    }

    #[test]
    fn rustls_connector_rejects_missing_empty_and_malformed_ca() {
        let dir = tempfile::TempDir::new().unwrap_or_else(|e| panic!("{e}"));
        assert!(build_rustls_config(&dir.path().join("missing.pem")).is_err());

        let empty = dir.path().join("empty.pem");
        std::fs::write(&empty, b"").unwrap_or_else(|e| panic!("{e}"));
        assert!(build_rustls_config(&empty).is_err());

        let malformed = dir.path().join("malformed.pem");
        std::fs::write(&malformed, b"not a certificate").unwrap_or_else(|e| panic!("{e}"));
        assert!(build_rustls_config(&malformed).is_err());
    }

    fn tls_server_config(certified: &rcgen::CertifiedKey) -> AnyResult<rustls::ServerConfig> {
        let key = rustls::pki_types::PrivatePkcs8KeyDer::from(certified.key_pair.serialize_der());
        rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .map_err(|e| anyhow::anyhow!("test TLS provider rejected safe versions: {e}"))?
        .with_no_client_auth()
        .with_single_cert(vec![certified.cert.der().clone()], key.into())
        .map_err(|e| anyhow::anyhow!("test TLS server config rejected certificate: {e}"))
    }

    async fn run_adversarial_tls_peer(
        listener: tokio::net::TcpListener,
        server_config: rustls::ServerConfig,
        sni_tx: tokio::sync::oneshot::Sender<Option<String>>,
    ) -> AnyResult<()> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let (mut tcp, _) = listener
            .accept()
            .await
            .context("test peer accepts client")?;
        let mut ssl_request = [0_u8; 8];
        tcp.read_exact(&mut ssl_request)
            .await
            .context("client sends PostgreSQL SSLRequest")?;
        anyhow::ensure!(
            ssl_request == [0, 0, 0, 8, 4, 210, 22, 47],
            "client did not request PostgreSQL TLS"
        );
        tcp.write_all(b"S")
            .await
            .context("test peer accepts PostgreSQL TLS upgrade")?;

        let start = tokio_rustls::LazyConfigAcceptor::new(rustls::server::Acceptor::default(), tcp)
            .await
            .context("client sends TLS ClientHello")?;
        let sni = start.client_hello().server_name().map(str::to_owned);
        sni_tx
            .send(sni)
            .map_err(|_| anyhow::anyhow!("test could not report observed SNI"))?;
        // Certificate verification occurs on the client. An alert is expected
        // here once it rejects this adversarial peer.
        let _ = start.into_stream(Arc::new(server_config)).await;
        Ok(())
    }

    #[tokio::test]
    async fn production_connector_sends_sni_and_rejects_an_unpinned_self_signed_peer(
    ) -> AnyResult<()> {
        let pinned_ca = rcgen::generate_simple_self_signed(vec!["rds.example.test".to_owned()])
            .unwrap_or_else(|e| panic!("generate pinned CA fixture: {e}"));
        let adversarial = rcgen::generate_simple_self_signed(vec!["rds.example.test".to_owned()])
            .unwrap_or_else(|e| panic!("generate adversarial certificate: {e}"));
        let temp = tempfile::tempdir().unwrap_or_else(|e| panic!("tempdir: {e}"));
        let ca_path = temp.path().join("pinned-rds-ca.pem");
        std::fs::write(&ca_path, pinned_ca.cert.pem())
            .unwrap_or_else(|e| panic!("write pinned CA: {e}"));

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .context("bind adversarial TLS peer")?;
        let port = listener
            .local_addr()
            .context("read adversarial TLS peer address")?
            .port();
        let (sni_tx, sni_rx) = tokio::sync::oneshot::channel();
        let peer = tokio::spawn(run_adversarial_tls_peer(
            listener,
            tls_server_config(&adversarial)?,
            sni_tx,
        ));

        let validated = translated_url(&format!(
            "postgresql://records:secret@rds.example.test:{port}/records?sslmode=verify-full"
        ));
        let mut config = build_driver_config(validated.driver_url(), validated.dns_hostname())
            .unwrap_or_else(|e| panic!("production config construction: {e}"));
        // Test transport only: preserve `rds.example.test` as the configured
        // host/SNI while connecting to our local adversarial fixture.
        config.hostaddr("127.0.0.1".parse().context("parse test loopback address")?);
        let pool = assemble_pool(config, &ca_path)
            .unwrap_or_else(|e| panic!("production connector assembly: {e}"));

        let error = match pool.get().await {
            Ok(_) => anyhow::bail!("an unpinned self-signed certificate was accepted"),
            Err(error) => error,
        };
        let chain = format!("{error:?}").to_ascii_lowercase();
        assert!(
            chain.contains("certificate") || chain.contains("issuer"),
            "connection failed before certificate verification: {chain}"
        );
        anyhow::ensure!(
            sni_rx.await.context("test TLS peer reports SNI")?
                == Some("rds.example.test".to_owned()),
            "connector did not send the validated hostname as SNI"
        );
        peer.await.context("test TLS peer joins")??;
        Ok(())
    }
}
