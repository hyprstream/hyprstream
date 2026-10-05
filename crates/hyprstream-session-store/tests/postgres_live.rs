//! Real PostgreSQL 16 causal tests. Explicit opt-in; no staging/cloud connections.
//! Set HS_SESSION_TEST_SOCKET to the Unix socket directory of a disposable local
//! PostgreSQL cluster. Each run creates/drops its own database and NOLOGIN roles.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use hyprstream_session_store::Session;
use hyprstream_session_store::{Admission, Error, PendingSession, Source, Store,
    MIGRATION, MIGRATION_V2, PROFILE, ROLE_GRANTS, ROLE_GRANTS_V2};
use std::time::Duration;
use tokio_postgres::{Client, NoTls};

const HOST: &str = "https://host.test";

async fn connect(socket: &str, database: &str, role: Option<&str>) -> Client {
    let (client, conn) = tokio_postgres::Config::new()
        .host_path(socket)
        .user("postgres")
        .dbname(database)
        .connect(NoTls)
        .await
        .unwrap();
    tokio::spawn(async move {
        let _ = conn.await;
    });
    if let Some(r) = role {
        // Only compile-time fixture role identifiers are used here.
        client
            .batch_execute(&format!("SET ROLE {r}"))
            .await
            .unwrap();
    }
    client
}

async fn now(c: &Client) -> i64 {
    c.query_one(
        "SELECT floor(extract(epoch FROM clock_timestamp()))::bigint",
        &[],
    )
    .await
    .unwrap()
    .get(0)
}

fn input(now: i64, id: u8) -> Admission {
    Admission {
        session: PendingSession {
            host: HOST.into(),
            sid: format!("session-{id}"),
            account_id: "fixture-account".into(),
            subject: "fixture-subject".into(),
            tenant: "fixture-tenant".into(),
            client_id: "fixture-client".into(),
            resource: HOST.into(),
            scopes: vec!["model:query".into()],
            grant_revision: "revision-1".into(),
            // Storage fixtures only: not valid cryptographic identities/proofs.
            ed_public: [id; 32],
            pq_public: vec![id; 1952],
            generation: [3; 32],
            collision_inventory_id: [7; 32],
            expires_at: now + 90,
        },
        local_collision_inventory_id: [7; 32],
        source: Source {
            issuer: "https://issuer.test".into(),
            subject: "opaque-upstream-fixture".into(),
            jti: format!("jti-{id}"),
            nonce: format!("nonce-{id}"),
            token_hash: [id; 32],
            issued_at: now,
            expires_at: now + 300,
        },
        challenge_created_at: now,
        challenge_expires_at: now + 60,
        grant_expires_at: now + 300,
    }
}

fn input_for(now: i64, id: u8, generation: [u8;32], inventory: [u8;32]) -> Admission {
    let mut admission = input(now, id);
    admission.session.generation = generation;
    admission.session.collision_inventory_id = inventory;
    admission.local_collision_inventory_id = inventory;
    admission
}

async fn count(c: &Client, table: &str) -> i64 {
    c.query_one(
        &format!("SELECT count(*) FROM federate_session.{table}"),
        &[],
    )
    .await
    .unwrap()
    .get(0)
}

async fn wait_blocked(admin: &Client, pid: i32) {
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            let blocked: bool = admin
                .query_one("SELECT cardinality(pg_blocking_pids($1)) > 0", &[&pid])
                .await
                .unwrap()
                .get(0);
            if blocked {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("expected a causal database lock wait");
}

async fn pid(c: &Client) -> i32 {
    c.query_one("SELECT pg_backend_pid()", &[])
        .await
        .unwrap()
        .get(0)
}

async fn scenarios(socket: &str, database: &str) {
    let admin = connect(socket, database, None).await;
    let mut runtime = connect(socket, database, Some("hs_policy_runtime")).await;
    let mut control = connect(socket, database, Some("hs_profile_control")).await;
    let n = now(&admin).await;
    assert!(matches!(
        Store::admit(&mut runtime, &input(n, 1)).await,
        Err(Error::Inactive)
    ));
    admin.execute("INSERT INTO federate_session.profile_state(host,profile,authority_generation,collision_inventory_id) VALUES ($1,$2,$3,$4)", &[&HOST,&PROFILE,&&[3u8;32][..],&&[7u8;32][..]]).await.unwrap();
    assert!(matches!(
        Store::admit(&mut runtime, &input(n, 1)).await,
        Err(Error::Inactive)
    ));
    assert_eq!(count(&admin, "sessions").await, 0);
    control
        .execute(
            "UPDATE federate_session.profile_state SET enabled=true",
            &[],
        )
        .await
        .unwrap();

    // Original contradiction is caught: SELECT alone cannot take FOR UPDATE.
    admin
        .batch_execute(
            "REVOKE UPDATE (lock_version) ON federate_session.profile_state FROM hs_policy_runtime",
        )
        .await
        .unwrap();
    assert!(matches!(
        Store::admit(&mut runtime, &input(n, 1)).await,
        Err(Error::Unavailable)
    ));
    admin
        .batch_execute(
            "GRANT UPDATE (lock_version) ON federate_session.profile_state TO hs_policy_runtime",
        )
        .await
        .unwrap();
    for sql in ["UPDATE federate_session.profile_state SET enabled=false", "UPDATE federate_session.profile_state SET authority_generation=decode(repeat('00',32),'hex')", "UPDATE federate_session.profile_state SET collision_inventory_id=decode(repeat('00',32),'hex')", "DELETE FROM federate_session.profile_state", "CREATE TABLE federate_session.forbidden(id int)"] {
        let err=runtime.batch_execute(sql).await.unwrap_err();
        assert_eq!(err.code(),Some(&tokio_postgres::error::SqlState::INSUFFICIENT_PRIVILEGE));
    }
    assert!(!admin.query_one(
        "SELECT has_column_privilege('hs_policy_runtime', 'federate_session.sessions', 'proof_epoch', 'UPDATE')",
        &[],
    ).await.unwrap().get::<_, bool>(0));
    assert!(runtime.batch_execute(
        "UPDATE federate_session.sessions SET proof_epoch=proof_epoch+1"
    ).await.is_err());
    for sql in [
        "SELECT * FROM federate_session.request_replay",
        "DELETE FROM federate_session.request_replay",
    ] {
        let err = runtime.batch_execute(sql).await.unwrap_err();
        assert_eq!(err.code(), Some(&tokio_postgres::error::SqlState::INSUFFICIENT_PRIVILEGE));
    }
    runtime
        .batch_execute("UPDATE federate_session.profile_state SET lock_version=999")
        .await
        .unwrap();
    // Two processes redeem exactly the same source to different sids.
    let mut other = connect(socket, database, Some("hs_policy_runtime")).await;
    let a = input(n, 1);
    let mut b = input(n, 1);
    b.session.sid = "other-sid".into();
    let (a_result, b_result) =
        tokio::join!(Store::admit(&mut runtime, &a), Store::admit(&mut other, &b));
    let committed = match (a_result, b_result) {
        (Ok(s), Err(Error::Conflict)) | (Err(Error::Conflict), Ok(s)) => s,
        _ => panic!("exactly one committed admission required"),
    };
    assert_eq!(count(&admin, "sessions").await, 1);
    assert_eq!(count(&admin, "replay").await, 1);
    let mut colliding_ed = input(n, 7);
    colliding_ed.session.ed_public = committed.ed_public;
    assert!(matches!(Store::admit(&mut runtime, &colliding_ed).await, Err(Error::Conflict)));
    let mut colliding_pq = input(n, 8);
    colliding_pq.session.pq_public = committed.pq_public.clone();
    assert!(matches!(Store::admit(&mut runtime, &colliding_pq).await, Err(Error::Conflict)));
    assert_eq!(count(&admin, "replay").await, 1);
    assert!(committed.proof_epoch > 0);
    assert_eq!(committed.collision_inventory_id, [7;32]);
    assert!(Store::active(&runtime, &committed).await.unwrap());
    let mut wrong = committed.clone();
    wrong.tenant = "another-tenant".into();
    assert!(!Store::active(&runtime, &wrong).await.unwrap());
    wrong = committed.clone();
    wrong.pq_public[0] ^= 1;
    assert!(!Store::active(&runtime, &wrong).await.unwrap());
    wrong = committed.clone();
    wrong.generation[0] ^= 1;
    assert!(!Store::active(&runtime, &wrong).await.unwrap());
    wrong = committed.clone();
    wrong.collision_inventory_id[0] ^= 1;
    assert!(!Store::active(&runtime, &wrong).await.unwrap());
    wrong = committed.clone();
    wrong.proof_epoch += 1;
    assert!(!Store::active(&runtime, &wrong).await.unwrap());
    let mut stale_inventory = input(n, 9);
    stale_inventory.local_collision_inventory_id = [8;32];
    assert!(matches!(Store::admit(&mut runtime, &stale_inventory).await, Err(Error::Inactive)));
    assert_eq!(count(&admin, "replay").await, 1);
    // Each replay index independently rejects reuse and rolls back session insert.
    for id in 2..5 {
        let mut a = input(n, id);
        match id {
            2 => a.source.jti = "jti-1".into(),
            3 => a.source.nonce = "nonce-1".into(),
            _ => a.source.token_hash = [1; 32],
        }
        assert!(matches!(
            Store::admit(&mut runtime, &a).await,
            Err(Error::Conflict)
        ));
    }
    assert_eq!(count(&admin, "sessions").await, 1);
    Store::revoke(&runtime, HOST, &committed.sid, &committed.generation)
        .await
        .unwrap();
    assert!(!Store::active(&runtime, &committed).await.unwrap());
    assert!(matches!(
        Store::admit(&mut runtime, &input(n, 1)).await,
        Err(Error::Conflict)
    ));
    let mut cleanup = connect(socket, database, Some("hs_session_cleanup")).await;
    assert!(matches!(
        Store::cleanup(&mut runtime).await,
        Err(Error::Unavailable)
    ));
    Store::cleanup(&mut cleanup).await.unwrap();
    assert_eq!(count(&admin, "replay").await, 1);
    // Retention formula and cleanup only after the entire source window.
    let retain: i64 = admin
        .query_one("SELECT retain_until FROM federate_session.replay", &[])
        .await
        .unwrap()
        .get(0);
    assert_eq!(retain, n + 330);
    admin.batch_execute("UPDATE federate_session.sessions SET created_at=created_at-1000, expires_at=expires_at-1000; UPDATE federate_session.replay SET source_iat=source_iat-1000,source_exp=source_exp-1000,retain_until=retain_until-1000").await.unwrap();
    Store::cleanup(&mut cleanup).await.unwrap();
    assert_eq!(count(&admin, "replay").await, 0);
    assert_eq!(count(&admin, "sessions").await, 0);

    // Invalid/expired inputs cannot consume replay state.
    let mut bad = input(n, 6);
    bad.source.issued_at = n - 121;
    bad.source.expires_at = n + 179;
    assert!(matches!(
        Store::admit(&mut runtime, &bad).await,
        Err(Error::Expired)
    ));
    bad = input(n, 6);
    bad.session.pq_public.pop();
    assert!(matches!(
        Store::admit(&mut runtime, &bad).await,
        Err(Error::Invalid)
    ));
    assert_eq!(count(&admin, "replay").await, 0);

    // Admission-first: trigger blocks after production row-lock, before insert.
    // A separate advisory lock is only test instrumentation, never production.
    admin.batch_execute("CREATE FUNCTION federate_session.test_barrier() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN PERFORM pg_advisory_xact_lock(741901); RETURN NEW; END $$; CREATE TRIGGER test_barrier BEFORE INSERT ON federate_session.sessions FOR EACH ROW EXECUTE FUNCTION federate_session.test_barrier()").await.unwrap();
    let mut serving_generation = [3u8;32];
    let mut serving_inventory = [7u8;32];
    for rotate in [false, true] {
        control
            .execute(
                "UPDATE federate_session.profile_state SET enabled=true WHERE authority_generation=$1 AND collision_inventory_id=$2",
                &[&&serving_generation[..], &&serving_inventory[..]],
            )
            .await
            .unwrap();
        admin
            .query_one("SELECT pg_advisory_lock(741901)", &[])
            .await
            .unwrap();
        let mut worker = connect(socket, database, Some("hs_policy_runtime")).await;
        let worker_pid = pid(&worker).await;
        let a = input_for(now(&admin).await, if rotate { 11 } else { 10 }, serving_generation, serving_inventory);
        let task = tokio::spawn(async move { Store::admit(&mut worker, &a).await });
        wait_blocked(&admin, worker_pid).await;
        let ctl = connect(socket, database, Some("hs_profile_control")).await;
        let ctl_pid = pid(&ctl).await;
        let control_task = tokio::spawn(async move {
            ctl.batch_execute(if rotate {"BEGIN; UPDATE federate_session.profile_state SET enabled=false; UPDATE federate_session.profile_state SET authority_generation=decode(repeat('04',32),'hex'), collision_inventory_id=decode(repeat('08',32),'hex') WHERE enabled=false; COMMIT"}else{"UPDATE federate_session.profile_state SET enabled=false"}).await.unwrap();
        });
        wait_blocked(&admin, ctl_pid).await;
        admin
            .query_one("SELECT pg_advisory_unlock(741901)", &[])
            .await
            .unwrap();
        let session = task.await.unwrap().unwrap();
        control_task.await.unwrap();
        if rotate { serving_generation = [4;32]; serving_inventory = [8;32]; }
        assert!(!Store::active(&runtime, &session).await.unwrap());
        assert!(matches!(
            Store::admit(&mut runtime, &input_for(now(&admin).await, 12, serving_generation, serving_inventory)).await,
            Err(Error::Inactive)
        ));
    }
    admin.batch_execute("DROP TRIGGER test_barrier ON federate_session.sessions; DROP FUNCTION federate_session.test_barrier()").await.unwrap();

    // Control-first: actual admission waits and observes committed control values.
    let mut rollback_session: Option<Session> = None;
    for (id, rotate, rollback) in [(20, false, false), (21, true, false), (22, false, true)] {
        control
            .execute(
                "UPDATE federate_session.profile_state SET enabled=true WHERE authority_generation=$1 AND collision_inventory_id=$2",
                &[&&serving_generation[..], &&serving_inventory[..]],
            )
            .await
            .unwrap();
        let before = count(&admin, "sessions").await;
        let before_replay = count(&admin, "replay").await;
        let tx = control.transaction().await.unwrap();
        tx.batch_execute(if rotate {"UPDATE federate_session.profile_state SET enabled=false; UPDATE federate_session.profile_state SET authority_generation=decode(repeat('05',32),'hex'), collision_inventory_id=decode(repeat('09',32),'hex') WHERE enabled=false"}else{"UPDATE federate_session.profile_state SET enabled=false"}).await.unwrap();
        let mut worker = connect(socket, database, Some("hs_policy_runtime")).await;
        let worker_pid = pid(&worker).await;
        let a = input_for(now(&admin).await, id, serving_generation, serving_inventory);
        let task = tokio::spawn(async move { Store::admit(&mut worker, &a).await });
        wait_blocked(&admin, worker_pid).await;
        if rollback {
            tx.rollback().await.unwrap();
        } else {
            tx.commit().await.unwrap();
        }
        let result = task.await.unwrap();
        if rollback {
            rollback_session = Some(result.unwrap());
        } else {
            assert!(matches!(result, Err(Error::Inactive)));
            assert_eq!(count(&admin, "sessions").await, before);
            assert_eq!(count(&admin, "replay").await, before_replay);
            if rotate { serving_generation = [5;32]; serving_inventory = [9;32]; }
        }
    }
    // Reconnect/restart preserves active session, revoke remains durable.
    let reconnect = connect(socket, database, Some("hs_policy_runtime")).await;
    let stored = rollback_session.expect("rollback admitted one session");
    assert!(Store::active(&reconnect, &stored).await.unwrap());
    assert!(stored.proof_epoch > committed.proof_epoch);
    assert!(admin.execute(
        "UPDATE federate_session.sessions SET proof_epoch=proof_epoch+1 WHERE sid=$1",
        &[&stored.sid],
    ).await.is_err());
    let namespace = [42u8;32];
    let request_id = [43u8;16];
    Store::consume_request_replay(&mut runtime, &stored, &namespace, &request_id,
        30).await.unwrap();
    // A second Policy connection sees the same durable uniqueness gate.
    assert!(matches!(Store::consume_request_replay(&mut other, &stored, &namespace,
        &request_id, 30).await, Err(Error::Conflict)));
    Store::consume_request_replay(&mut runtime, &stored, &namespace, &[44;16],
        30).await.unwrap();
    assert_eq!(count(&admin, "request_replay").await, 2);
    assert!(runtime.batch_execute("DELETE FROM federate_session.request_replay").await.is_err());
    let mut wrong_epoch = stored.clone();
    wrong_epoch.proof_epoch += 1;
    assert!(matches!(Store::consume_request_replay(&mut runtime, &wrong_epoch, &namespace,
        &[45;16], 30).await, Err(Error::Inactive)));
    assert_eq!(count(&admin, "request_replay").await, 2);
    admin
        .query_one("SELECT pg_terminate_backend($1)", &[&pid(&reconnect).await])
        .await
        .unwrap();
    assert!(matches!(
        Store::active(&reconnect, &stored).await,
        Err(Error::Unavailable)
    ));
    control.batch_execute("UPDATE federate_session.profile_state SET enabled=false").await.unwrap();
    assert!(control.batch_execute("UPDATE federate_session.profile_state SET collision_inventory_id=decode(repeat('0a',32),'hex')").await.is_err());
    // Paired generation-only and generation+inventory rotations both succeed
    // only while disabled. The old session is never reactivated.
    control.batch_execute("UPDATE federate_session.profile_state SET authority_generation=decode(repeat('06',32),'hex'), collision_inventory_id=decode(repeat('09',32),'hex') WHERE enabled=false").await.unwrap();
    control.batch_execute("UPDATE federate_session.profile_state SET authority_generation=decode(repeat('07',32),'hex'), collision_inventory_id=decode(repeat('0a',32),'hex') WHERE enabled=false").await.unwrap();
    assert!(!Store::active(&runtime, &stored).await.unwrap());
}

#[tokio::test]
#[ignore = "requires disposable local PostgreSQL 16 Unix socket; never staging"]
async fn postgres_admission_causal() {
    let socket = std::env::var("HS_SESSION_TEST_SOCKET").expect("local disposable socket required");
    assert!(
        socket.starts_with('/'),
        "Unix socket only; no remote DSN accepted"
    );
    let maintenance = connect(&socket, "postgres", None).await;
    let db = format!("hs_session_test_{}", std::process::id());
    maintenance.batch_execute("CREATE ROLE hs_policy_runtime NOLOGIN; CREATE ROLE hs_profile_control NOLOGIN; CREATE ROLE hs_session_cleanup NOLOGIN; CREATE ROLE hs_session_migration NOLOGIN").await.unwrap();
    maintenance
        .batch_execute(&format!("CREATE DATABASE {db};"))
        .await
        .unwrap();
    let admin = connect(&socket, &db, None).await;
    admin
        .batch_execute(&format!(
            "GRANT CREATE ON DATABASE {db} TO hs_session_migration; SET ROLE hs_session_migration;"
        ))
        .await
        .unwrap();
    admin.batch_execute(MIGRATION).await.unwrap();
    admin.execute(
        "INSERT INTO federate_session.profile_state(host,profile,authority_generation) VALUES ($1,$2,$3)",
        &[&"https://migration-refusal.test", &PROFILE, &&[1u8;32][..]],
    ).await.unwrap();
    assert!(admin.batch_execute(MIGRATION_V2).await.is_err());
    admin.batch_execute("ROLLBACK").await.unwrap();
    admin.execute(
        "DELETE FROM federate_session.profile_state WHERE host=$1 AND profile=$2",
        &[&"https://migration-refusal.test", &PROFILE],
    ).await.unwrap();
    admin.batch_execute(MIGRATION_V2).await.unwrap();
    admin.batch_execute(ROLE_GRANTS).await.unwrap();
    admin.batch_execute(ROLE_GRANTS_V2).await.unwrap();
    admin.batch_execute("RESET ROLE").await.unwrap();
    let socket_copy = socket.clone();
    let db_copy = db.clone();
    // Catch task panic so teardown still runs on failed assertions.
    let outcome = tokio::spawn(async move { scenarios(&socket_copy, &db_copy).await }).await;
    drop(admin);
    maintenance
        .batch_execute(&format!("DROP DATABASE {db} WITH (FORCE)"))
        .await
        .unwrap();
    maintenance.batch_execute("DROP ROLE hs_policy_runtime; DROP ROLE hs_profile_control; DROP ROLE hs_session_cleanup; DROP ROLE hs_session_migration;").await.unwrap();
    outcome.unwrap();
}
