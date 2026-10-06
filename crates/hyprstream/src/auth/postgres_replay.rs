//! Durable, domain-wide replay admission for proof CWTs and mediated queries.
//!
//! The synchronous RPC admission trait is bridged to a dedicated Tokio thread.
//! Its bounded queue and reply deadline deny on overload or database failure;
//! no process-local replay map is used in this deployment mode.

use std::{
    path::PathBuf,
    sync::mpsc::{self, Receiver, SyncSender, TrySendError},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, ensure, Context, Result};
use deadpool_postgres::Pool;
use hyprstream_rpc::proof::{
    admission::{
        MediatedQueryReplayKey, ProofAdmissionResult, ProofReplayKey, ProofReplayStore,
        ReplayDomainGuarantee,
    },
    ProofDisposition,
};
use sha2::{Digest, Sha256};

use super::postgres_store::build_replay_pool;

// A caller may time out before the worker's pool wait plus statement deadline.
// It denies in that case even if the worker later commits; a retry then sees
// Replayed. This is a deliberate fail-closed availability tradeoff.
const REPLY_TIMEOUT: Duration = Duration::from_secs(15);
const DB_TIMEOUT: Duration = Duration::from_secs(10);
const QUEUE_CAPACITY: usize = 64;
const CLEAN_BATCH: i64 = 512;

// A single statement gives the unique-index arbiter the row lock. The conflict
// predicate is evaluated after that lock is acquired: an unexpired admission
// is never replaced, while an expired one can be reused. DB statement time is
// authoritative; an already expired proposed admission creates no row.
const ADMIT_SQL: &str = r#"
INSERT INTO replay_admission.entries_v1 AS replay
    (service_domain, partition, key_digest, expires_at)
SELECT $1::text, $2::smallint, $3::bytea, $4::bigint
WHERE $4::bigint > floor(extract(epoch FROM statement_timestamp()))::bigint
ON CONFLICT (service_domain, partition, key_digest)
DO UPDATE SET expires_at = EXCLUDED.expires_at
WHERE replay.expires_at <= floor(extract(epoch FROM statement_timestamp()))::bigint
RETURNING 1
"#;

// SKIP LOCKED avoids blocking admissions. The outer expiry predicate ensures
// cleanup cannot delete a row renewed by a concurrent admission.
const CLEAN_SQL: &str = r#"
WITH expired AS (
    SELECT ctid FROM replay_admission.entries_v1
    WHERE expires_at <= floor(extract(epoch FROM statement_timestamp()))::bigint
    ORDER BY expires_at LIMIT $1 FOR UPDATE SKIP LOCKED
)
DELETE FROM replay_admission.entries_v1 AS replay USING expired
WHERE replay.ctid = expired.ctid
  AND replay.expires_at <= floor(extract(epoch FROM statement_timestamp()))::bigint
"#;

struct Admission {
    partition: i16,
    digest: [u8; 32],
    expires_at: i64,
    answer: SyncSender<ProofAdmissionResult>,
}

/// Stable service-domain name must be identical on every verifier serving the
/// same domain, and distinct for independently admitting services.
pub struct PostgresProofReplayStore {
    tx: SyncSender<Admission>,
}

impl PostgresProofReplayStore {
    /// Open through a role-scoped URL file and a pinned CA file. No schema is
    /// created here: operators apply the reviewed migration before startup.
    pub fn from_env() -> Result<Self> {
        let domain = std::env::var("HYPRSTREAM_REPLAY_SERVICE_DOMAIN")
            .context("HYPRSTREAM_REPLAY_SERVICE_DOMAIN is required")?;
        Self::validate_domain(&domain)?;
        let url_path = PathBuf::from(
            std::env::var_os("HYPRSTREAM_REPLAY_POSTGRES_URL_FILE")
                .context("HYPRSTREAM_REPLAY_POSTGRES_URL_FILE is required")?,
        );
        let ca_path = PathBuf::from(
            std::env::var_os("HYPRSTREAM_REPLAY_POSTGRES_SSLROOTCERT_FILE")
                .context("HYPRSTREAM_REPLAY_POSTGRES_SSLROOTCERT_FILE is required")?,
        );
        let url = std::fs::read_to_string(&url_path).context("reading replay Postgres URL file")?;
        let url = url.trim();
        ensure!(!url.is_empty(), "replay Postgres URL file is empty");
        // build_replay_pool enforces remote DNS, verify-full, CA-pinned TLS.
        let pool =
            build_replay_pool(url, &ca_path, 2).context("building replay Postgres TLS pool")?;
        Self::start(domain, pool)
    }

    fn validate_domain(domain: &str) -> Result<()> {
        ensure!(
            (1..=128).contains(&domain.len()),
            "invalid replay service-domain length"
        );
        ensure!(
            domain.bytes().all(|b| b.is_ascii_lowercase()
                || b.is_ascii_digit()
                || matches!(b, b'.' | b'-' | b'_'))
                && domain.as_bytes()[0].is_ascii_alphanumeric(),
            "replay service-domain must be lower-case ASCII DNS-like text"
        );
        Ok(())
    }

    fn start(domain: String, pool: Pool) -> Result<Self> {
        let (tx, rx) = mpsc::sync_channel(QUEUE_CAPACITY);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name("postgres-replay-admission".into())
            .spawn(move || worker(domain, pool, rx, ready_tx))
            .context("starting replay Postgres worker")?;
        match ready_rx.recv_timeout(REPLY_TIMEOUT) {
            Ok(Ok(())) => Ok(Self { tx }),
            Ok(Err(())) => bail!("replay Postgres schema, privileges, or connectivity unavailable"),
            Err(_) => bail!("replay Postgres startup timed out"),
        }
    }

    fn admit(&self, partition: i16, digest: [u8; 32], expires_at: u64) -> ProofAdmissionResult {
        let Ok(expires_at) = i64::try_from(expires_at) else {
            return ProofAdmissionResult::Failed;
        };
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(u64::MAX);
        if (expires_at as u64) <= now {
            return ProofAdmissionResult::Failed;
        }
        let (answer, rx) = mpsc::sync_channel(1);
        let item = Admission {
            partition,
            digest,
            expires_at,
            answer,
        };
        match self.tx.try_send(item) {
            Ok(()) => rx
                .recv_timeout(REPLY_TIMEOUT)
                .unwrap_or(ProofAdmissionResult::Failed),
            Err(TrySendError::Full(_) | TrySendError::Disconnected(_)) => {
                ProofAdmissionResult::Failed
            }
        }
    }
}

impl ProofReplayStore for PostgresProofReplayStore {
    fn domain_guarantee(&self) -> ReplayDomainGuarantee {
        ReplayDomainGuarantee::LinearizableSharedStore
    }

    fn check_and_insert(
        &self,
        partition: ProofDisposition,
        key: &ProofReplayKey,
        expires_at: u64,
    ) -> ProofAdmissionResult {
        let partition = match partition {
            ProofDisposition::Authenticated => 1,
            ProofDisposition::Unattributed => 2,
        };
        let mut digest = Sha256::new();
        digest.update(b"hyprstream-proof-replay-v1\0");
        digest.update(key.signer_thumbprint);
        digest.update(key.request_id);
        self.admit(partition, digest.finalize().into(), expires_at)
    }

    fn check_and_insert_mediated(
        &self,
        key: &MediatedQueryReplayKey,
        expires_at: u64,
    ) -> ProofAdmissionResult {
        if hyprstream_rpc::proof::admission::validate_mediated_query_dimensions(
            &key.mediator,
            &key.resource,
            &key.operation,
        )
        .is_err()
        {
            return ProofAdmissionResult::Failed;
        }
        let mut digest = Sha256::new();
        digest.update(b"hyprstream-mediated-replay-v1\0");
        digest.update(key.signer_thumbprint);
        digest.update(key.request_id.to_be_bytes());
        digest.update(key.request_nonce);
        for field in [&key.mediator, &key.resource, &key.operation] {
            digest.update((field.len() as u32).to_be_bytes());
            digest.update(field.as_bytes());
        }
        self.admit(3, digest.finalize().into(), expires_at)
    }
}

fn worker(
    domain: String,
    pool: Pool,
    rx: Receiver<Admission>,
    ready: SyncSender<std::result::Result<(), ()>>,
) {
    let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        let _ = ready.send(Err(()));
        return;
    };
    let init = runtime.block_on(async {
        let client = tokio::time::timeout(DB_TIMEOUT, pool.get()).await.map_err(|_| ())?.map_err(|_| ())?;
        let row = tokio::time::timeout(DB_TIMEOUT, client.query_one(
            "SELECT has_table_privilege(current_user, 'replay_admission.entries_v1', 'SELECT') AND has_table_privilege(current_user, 'replay_admission.entries_v1', 'INSERT') AND has_table_privilege(current_user, 'replay_admission.entries_v1', 'UPDATE') AND has_table_privilege(current_user, 'replay_admission.entries_v1', 'DELETE')",
            &[],
        )).await.map_err(|_| ())?.map_err(|_| ())?;
        if !row.get::<_, bool>(0) { return Err(()); }
        tokio::time::timeout(DB_TIMEOUT, client.query("SELECT key_digest FROM replay_admission.entries_v1 LIMIT 0", &[]))
            .await.map_err(|_| ())?.map_err(|_| ())?;
        Ok(())
    });
    let initialized = init.is_ok();
    let _ = ready.send(init);
    if !initialized {
        return;
    }
    let mut count = 0u64;
    loop {
        match rx.recv_timeout(Duration::from_secs(60)) {
            Ok(item) => {
                count = count.wrapping_add(1);
                let result = runtime
                    .block_on(async {
                        let client = tokio::time::timeout(DB_TIMEOUT, pool.get())
                            .await
                            .map_err(|_| ())?
                            .map_err(|_| ())?;
                        if count.is_multiple_of(64) {
                            tokio::time::timeout(
                                DB_TIMEOUT,
                                client.execute(CLEAN_SQL, &[&CLEAN_BATCH]),
                            )
                            .await
                            .map_err(|_| ())?
                            .map_err(|_| ())?;
                        }
                        let row = tokio::time::timeout(
                            DB_TIMEOUT,
                            client.query_opt(
                                ADMIT_SQL,
                                &[
                                    &domain,
                                    &item.partition,
                                    &item.digest.as_slice(),
                                    &item.expires_at,
                                ],
                            ),
                        )
                        .await
                        .map_err(|_| ())?
                        .map_err(|_| ())?;
                        Ok::<_, ()>(if row.is_some() {
                            ProofAdmissionResult::Admitted
                        } else {
                            ProofAdmissionResult::Replayed
                        })
                    })
                    .unwrap_or(ProofAdmissionResult::Failed);
                let _ = item.answer.send(result);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = runtime.block_on(async {
                    let client = tokio::time::timeout(DB_TIMEOUT, pool.get())
                        .await
                        .map_err(|_| ())?
                        .map_err(|_| ())?;
                    tokio::time::timeout(DB_TIMEOUT, client.execute(CLEAN_SQL, &[&CLEAN_BATCH]))
                        .await
                        .map_err(|_| ())?
                        .map_err(|_| ())?;
                    Ok::<_, ()>(())
                });
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    fn is_local_scratch_host(host: Option<&str>) -> bool {
        // OCI qualification attaches this test container and its disposable
        // PostgreSQL fixture to the same isolated Podman network. `postgres`
        // is the exact network alias assigned to that fixture there; do not
        // accept arbitrary DNS names or private IPs (which could be RDS).
        matches!(host, Some("localhost" | "127.0.0.1" | "::1" | "postgres"))
    }

    fn proof_key(id: u8) -> ProofReplayKey {
        ProofReplayKey {
            signer_thumbprint: [9; 32],
            request_id: [id; 16],
        }
    }

    fn mediated_key(id: u64) -> MediatedQueryReplayKey {
        MediatedQueryReplayKey {
            signer_thumbprint: [9; 32],
            request_id: id,
            request_nonce: [3; 16],
            mediator: "registry".into(),
            resource: "model:one".into(),
            operation: "read".into(),
        }
    }

    fn future(seconds: u64) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + seconds
    }

    // This is deliberately ignored unless an operator supplies a disposable
    // loopback database or the isolated OCI test's exact `postgres` network
    // alias. It must never point at shared RDS.
    #[test]
    #[ignore = "requires isolated local PostgreSQL and HYPRSTREAM_REPLAY_TEST_URL_FILE"]
    fn scratch_postgres_replay_contract() {
        let file =
            std::env::var_os("HYPRSTREAM_REPLAY_TEST_URL_FILE").expect("scratch URL file required");
        let url = std::fs::read_to_string(file).expect("read scratch URL file");
        let url = url.trim();
        let parsed = url::Url::parse(url).expect("parse scratch URL");
        assert!(is_local_scratch_host(parsed.host_str()));
        assert_eq!(parsed.port_or_known_default(), Some(5432));
        assert!(parsed
            .path()
            .trim_start_matches('/')
            .starts_with("hyprstream_replay_test_"));
        let rt = tokio::runtime::Runtime::new().unwrap();
        let nonce = uuid::Uuid::new_v4().simple().to_string();
        let limited_role = format!("replay_limited_{nonce}");
        let missing_role = format!("replay_nogrant_{nonce}");
        let role_password = uuid::Uuid::new_v4().simple().to_string();
        rt.block_on(async {
            let (client, connection) = tokio_postgres::connect(url, tokio_postgres::NoTls)
                .await
                .unwrap();
            tokio::spawn(async move {
                let _ = connection.await;
            });
            client
                .batch_execute(include_str!("../../sql/replay_admission_v1.sql"))
                .await
                .unwrap();
            client
                .batch_execute("ALTER ROLE hyprstream_replay_runtime LOGIN")
                .await
                .unwrap();
            assert!(
                client
                    .batch_execute(include_str!("../../sql/replay_admission_v1.sql"))
                    .await
                    .is_err(),
                "migration must reject any preexisting runtime role"
            );
            client.batch_execute("ROLLBACK").await.unwrap();
            client
                .batch_execute("ALTER ROLE hyprstream_replay_runtime NOLOGIN")
                .await
                .unwrap();
            let role = client
                .query_one(
                    "SELECT rolcanlogin FROM pg_roles WHERE rolname = 'hyprstream_replay_runtime'",
                    &[],
                )
                .await
                .unwrap();
            assert!(
                !role.get::<_, bool>(0),
                "runtime privilege group must remain NOLOGIN"
            );
            // The scratch admin applies the migration; admission below uses
            // a distinct login that inherits only the migration's NOLOGIN
            // replay group. A login without that group must fail startup.
            client
                .batch_execute(&format!(
                    "CREATE ROLE {limited_role} LOGIN PASSWORD '{role_password}'; \
                     GRANT hyprstream_replay_runtime TO {limited_role}; \
                     CREATE ROLE {missing_role} LOGIN PASSWORD '{role_password}';"
                ))
                .await
                .unwrap();
        });

        let mut limited_url = parsed.clone();
        limited_url.set_username(&limited_role).unwrap();
        limited_url.set_password(Some(&role_password)).unwrap();
        let mut missing_url = parsed.clone();
        missing_url.set_username(&missing_role).unwrap();
        missing_url.set_password(Some(&role_password)).unwrap();

        fn pool(url: &str) -> Pool {
            let config: tokio_postgres::Config = url.parse().unwrap();
            Pool::builder(deadpool_postgres::Manager::new(
                config,
                tokio_postgres::NoTls,
            ))
            .runtime(deadpool_postgres::Runtime::Tokio1)
            .max_size(2)
            .build()
            .unwrap()
        }
        let domain = format!("test.{}", uuid::Uuid::new_v4().simple());
        assert!(
            PostgresProofReplayStore::start(domain.clone(), pool(missing_url.as_str())).is_err()
        );
        rt.block_on(async {
            let (missing_client, connection) = tokio_postgres::connect(missing_url.as_str(), tokio_postgres::NoTls)
                .await
                .unwrap();
            tokio::spawn(async move { let _ = connection.await; });
            let missing_grants = missing_client.query_one(
                "SELECT has_schema_privilege(current_user, 'replay_admission', 'USAGE')",
                &[],
            ).await.unwrap();
            assert!(!missing_grants.get::<_, bool>(0));
            let (client, connection) = tokio_postgres::connect(limited_url.as_str(), tokio_postgres::NoTls)
                .await
                .unwrap();
            tokio::spawn(async move { let _ = connection.await; });
            let grants = client.query_one(
                "SELECT has_schema_privilege(current_user, 'replay_admission', 'USAGE'), \
                        has_table_privilege(current_user, 'replay_admission.entries_v1', 'SELECT, INSERT, UPDATE, DELETE')",
                &[],
            ).await.unwrap();
            assert!(grants.get::<_, bool>(0));
            assert!(grants.get::<_, bool>(1));
        });
        let a = Arc::new(
            PostgresProofReplayStore::start(domain.clone(), pool(limited_url.as_str())).unwrap(),
        );
        let b = Arc::new(
            PostgresProofReplayStore::start(domain.clone(), pool(limited_url.as_str())).unwrap(),
        );
        let key = proof_key(1);
        let barrier = Arc::new(Barrier::new(2));
        let handles: Vec<_> = [a.clone(), b.clone()]
            .into_iter()
            .map(|store| {
                let key = key.clone();
                let barrier = barrier.clone();
                std::thread::spawn(move || {
                    barrier.wait();
                    store.check_and_insert(ProofDisposition::Authenticated, &key, future(60))
                })
            })
            .collect();
        let outcomes: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(
            outcomes
                .iter()
                .filter(|r| **r == ProofAdmissionResult::Admitted)
                .count(),
            1
        );
        assert_eq!(
            outcomes
                .iter()
                .filter(|r| **r == ProofAdmissionResult::Replayed)
                .count(),
            1
        );

        // A new verifier process with the same service domain sees prior rows.
        drop(a);
        drop(b);
        let restarted =
            PostgresProofReplayStore::start(domain.clone(), pool(limited_url.as_str())).unwrap();
        assert_eq!(
            restarted.check_and_insert(ProofDisposition::Authenticated, &key, future(60)),
            ProofAdmissionResult::Replayed
        );
        assert_eq!(
            restarted.check_and_insert(ProofDisposition::Unattributed, &key, future(60)),
            ProofAdmissionResult::Admitted
        );
        assert_eq!(
            restarted.check_and_insert_mediated(&mediated_key(1), future(60)),
            ProofAdmissionResult::Admitted
        );
        assert_eq!(
            restarted.check_and_insert_mediated(&mediated_key(1), future(60)),
            ProofAdmissionResult::Replayed
        );
        assert_eq!(
            restarted.check_and_insert_mediated(&mediated_key(2), future(60)),
            ProofAdmissionResult::Admitted
        );
        let other =
            PostgresProofReplayStore::start(format!("{domain}.other"), pool(limited_url.as_str()))
                .unwrap();
        assert_eq!(
            other.check_and_insert(ProofDisposition::Authenticated, &key, future(60)),
            ProofAdmissionResult::Admitted
        );

        // Race expiry cleanup with renewal of an expired key. A newly live row
        // must survive whichever statement wins the row lock.
        let expiring = proof_key(2);
        assert_eq!(
            restarted.check_and_insert(ProofDisposition::Authenticated, &expiring, future(3)),
            ProofAdmissionResult::Admitted
        );
        std::thread::sleep(Duration::from_secs(4));
        let cleanup_pool = pool(limited_url.as_str());
        let cleanup = std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let client = cleanup_pool.get().await.unwrap();
                client.execute(CLEAN_SQL, &[&CLEAN_BATCH]).await.unwrap();
            });
        });
        let raced =
            restarted.check_and_insert(ProofDisposition::Authenticated, &expiring, future(60));
        cleanup.join().unwrap();
        // Controlled PostgreSQL 16 interleavings showed that a statement
        // waiting on cleanup's delete inserts after that delete commits.
        // Either lock order admits exactly one renewal; the next is a replay.
        assert_eq!(raced, ProofAdmissionResult::Admitted);
        assert_eq!(
            restarted.check_and_insert(ProofDisposition::Authenticated, &expiring, future(60)),
            ProofAdmissionResult::Replayed,
        );

        // A role can lose database permission after startup. The already
        // installed verifier must deny its next admission, never fall back to
        // process-local state or report success from the old pool session.
        rt.block_on(async {
            let (client, connection) = tokio_postgres::connect(url, tokio_postgres::NoTls)
                .await
                .unwrap();
            tokio::spawn(async move {
                let _ = connection.await;
            });
            client
                .batch_execute(&format!(
                    "REVOKE hyprstream_replay_runtime FROM {limited_role}"
                ))
                .await
                .unwrap();
        });
        assert_eq!(
            restarted.check_and_insert(ProofDisposition::Authenticated, &proof_key(3), future(60)),
            ProofAdmissionResult::Failed,
        );

        // A missing schema/connection prevents store construction; no local
        // fallback is manufactured by this constructor.
        let mut unavailable = tokio_postgres::Config::new();
        unavailable
            .host("127.0.0.1")
            .port(1)
            .user("scratch")
            .dbname("missing");
        let unavailable_pool = Pool::builder(deadpool_postgres::Manager::new(
            unavailable,
            tokio_postgres::NoTls,
        ))
        .runtime(deadpool_postgres::Runtime::Tokio1)
        .max_size(1)
        .build()
        .unwrap();
        assert!(PostgresProofReplayStore::start(domain, unavailable_pool).is_err());
    }

    #[test]
    fn scratch_host_allowlist_is_local_or_exact_oci_fixture_alias() {
        for host in [
            Some("localhost"),
            Some("127.0.0.1"),
            Some("::1"),
            Some("postgres"),
        ] {
            assert!(
                is_local_scratch_host(host),
                "expected local fixture host: {host:?}"
            );
        }
        for host in [
            None,
            Some("postgres.example.test"),
            Some("db.example.test"),
            Some("10.0.0.8"),
            Some("192.168.1.20"),
            Some("rds.amazonaws.com"),
        ] {
            assert!(
                !is_local_scratch_host(host),
                "unexpected remote host allowed: {host:?}"
            );
        }
    }

    #[test]
    fn domains_are_explicit_and_bounded() {
        for valid in ["staging.policy", "staging_registry-1"] {
            PostgresProofReplayStore::validate_domain(valid).unwrap();
        }
        for invalid in ["", "UPPER", ".leading", "a/b", "x y"] {
            assert!(PostgresProofReplayStore::validate_domain(invalid).is_err());
        }
    }

    #[test]
    fn shared_domain_has_no_process_local_unattributed_challenge() {
        use hyprstream_rpc::proof::challenge::{
            ChallengeManager, DEFAULT_CHALLENGE_OVERLAP_SECS, DEFAULT_CHALLENGE_WINDOW_SECS,
        };
        let (tx, rx) = mpsc::sync_channel(1);
        drop(rx);
        let store = PostgresProofReplayStore { tx };
        assert_eq!(
            store.domain_guarantee(),
            ReplayDomainGuarantee::LinearizableSharedStore
        );
        assert!(ChallengeManager::rotating_for_domain(
            store.domain_guarantee(),
            DEFAULT_CHALLENGE_WINDOW_SECS,
            DEFAULT_CHALLENGE_OVERLAP_SECS,
            future(0),
        )
        .is_none());
    }

    #[test]
    fn replay_migration_rejects_any_preexisting_runtime_role() {
        let migration = include_str!("../../sql/replay_admission_v1.sql");
        let role_guard = migration
            .find("IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hyprstream_replay_runtime')");
        let create_schema = migration
            .find("CREATE SCHEMA IF NOT EXISTS replay_admission");
        assert!(
            matches!(
                (role_guard, create_schema),
                (Some(role_guard), Some(create_schema)) if role_guard < create_schema
            ),
            "preexisting role must be rejected before any schema/table DDL"
        );
        assert!(migration.contains(
            "hyprstream_replay_runtime already exists; audit and provision a fresh migration target"
        ));
        assert!(!migration.contains("ALTER ROLE hyprstream_replay_runtime"));
        assert!(migration.starts_with("-- Reviewed migration artifact"));
        assert!(migration.contains("BEGIN;"));
        assert!(migration.ends_with("COMMIT;\n"));
    }

    #[test]
    fn disconnected_worker_denies() {
        let (tx, rx) = mpsc::sync_channel(1);
        drop(rx);
        let store = PostgresProofReplayStore { tx };
        assert_eq!(
            store.check_and_insert(ProofDisposition::Authenticated, &proof_key(1), future(60)),
            ProofAdmissionResult::Failed
        );
        assert_eq!(
            store.check_and_insert_mediated(&mediated_key(1), future(60)),
            ProofAdmissionResult::Failed
        );
    }
}
