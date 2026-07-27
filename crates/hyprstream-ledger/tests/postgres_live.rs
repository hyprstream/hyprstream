//! Live-PostgreSQL tests for the durable backend (PAY-01 #1389).
//!
//! These exercise the properties that cannot be shown against an in-memory
//! double: single-writer admission and fencing, restart rebuilding the mirror
//! from the committed journal, integrity detection when the projection or the
//! journal is tampered with, and legacy-schema migration.
//!
//! They require a real PostgreSQL. Set `HS_LEDGER_TEST_PG` to a maintenance DSN
//! (a superuser-ish connection to an existing database — each test creates and
//! drops its own throwaway database off it):
//!
//! ```sh
//! podman run -d --name hs-ledger-test -e POSTGRES_PASSWORD=test \
//!     -p 55432:5432 docker.io/library/postgres:16-alpine
//! export HS_LEDGER_TEST_PG='postgres://postgres:test@127.0.0.1:55432/postgres'
//! cargo test -p hyprstream-ledger --features postgres --test postgres_live
//! ```
//!
//! Without that variable every test in this file skips, so the default
//! `cargo test` run stays hermetic.

#![cfg(feature = "postgres")]
// A test harness legitimately unwraps known-good values.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use hyprstream_ledger::journal::CheckpointSigner;
use hyprstream_ledger::postgres::{PostgresConfig, PostgresLedger};
use hyprstream_ledger::{
    AccountId, AccountSpec, Did, IssueTransfer, LedgerBackend, LedgerError, Purpose, Transfer,
    TransferId, TransferResult, UnitId,
};
use std::sync::atomic::{AtomicU32, Ordering};

// ─── Harness ────────────────────────────────────────────────────────────────

/// The maintenance DSN, or `None` when live tests are disabled.
fn maintenance_dsn() -> Option<String> {
    std::env::var("HS_LEDGER_TEST_PG")
        .ok()
        .filter(|s| !s.is_empty())
}

/// Skip-with-notice. Returns `None` if live testing is not configured.
macro_rules! require_pg {
    () => {
        match maintenance_dsn() {
            Some(dsn) => dsn,
            None => {
                eprintln!("skipping: HS_LEDGER_TEST_PG not set");
                return;
            }
        }
    };
}

static DB_SEQ: AtomicU32 = AtomicU32::new(0);

/// A throwaway database, dropped when the guard goes out of scope.
///
/// Each test gets its own database rather than sharing tables, so tests remain
/// independent under cargo's default parallel execution.
struct TestDb {
    maintenance: String,
    name: String,
}

impl TestDb {
    fn create(maintenance: &str) -> Self {
        let n = DB_SEQ.fetch_add(1, Ordering::SeqCst);
        let name = format!("hs_ledger_t{}_{}", std::process::id(), n);
        run_sql(maintenance, &format!("DROP DATABASE IF EXISTS {name}"));
        run_sql(maintenance, &format!("CREATE DATABASE {name}"));
        TestDb {
            maintenance: maintenance.to_owned(),
            name,
        }
    }

    /// DSN for the throwaway database.
    fn dsn(&self) -> String {
        swap_db(&self.maintenance, &self.name)
    }

    fn config(&self) -> PostgresConfig {
        PostgresConfig {
            url: self.dsn(),
            pool_size: 4,
        }
    }
}

impl Drop for TestDb {
    fn drop(&mut self) {
        // Terminate stragglers so DROP DATABASE cannot block on a lingering
        // background connection.
        run_sql(
            &self.maintenance,
            &format!(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity \
                 WHERE datname = '{}' AND pid <> pg_backend_pid()",
                self.name
            ),
        );
        run_sql(
            &self.maintenance,
            &format!("DROP DATABASE IF EXISTS {}", self.name),
        );
    }
}

/// Replace the database component of a DSN.
fn swap_db(dsn: &str, db: &str) -> String {
    match dsn.rfind('/') {
        Some(i) => {
            let (head, tail) = dsn.split_at(i + 1);
            // Preserve any query string (e.g. sslmode).
            let query = tail.find('?').map(|q| &tail[q..]).unwrap_or("");
            format!("{head}{db}{query}")
        }
        None => dsn.to_owned(),
    }
}

/// Execute a statement on its own short-lived connection.
fn run_sql(dsn: &str, sql: &str) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let (client, conn) = tokio_postgres::connect(dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let handle = tokio::spawn(async move {
            let _ = conn.await;
        });
        let _ = client.batch_execute(sql).await;
        drop(client);
        let _ = handle.await;
    });
}

/// Execute a statement and require it to succeed.
fn must_sql(dsn: &str, sql: &str) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let (client, conn) = tokio_postgres::connect(dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let handle = tokio::spawn(async move {
            let _ = conn.await;
        });
        client
            .batch_execute(sql)
            .await
            .unwrap_or_else(|e| panic!("sql failed: {sql}\n{e}"));
        drop(client);
        let _ = handle.await;
    });
}

/// Scalar text query helper.
fn query_one_text(dsn: &str, sql: &str) -> String {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let (client, conn) = tokio_postgres::connect(dsn, tokio_postgres::NoTls)
            .await
            .unwrap();
        let handle = tokio::spawn(async move {
            let _ = conn.await;
        });
        let row = client.query_one(sql, &[]).await.unwrap();
        let v: String = row.get(0);
        drop(client);
        let _ = handle.await;
        v
    })
}

// ─── Ledger fixtures ────────────────────────────────────────────────────────

fn ledger_id() -> Did {
    Did("did:web:cell.test".to_owned())
}

fn unit() -> UnitId {
    UnitId {
        issuer: Did("did:web:issuer.test".to_owned()),
        resource_class: "gpu.h100.seconds".to_owned(),
    }
}

/// Test issuance authority. Minting is sealed behind a verified,
/// transfer-bound capability, so tests must supply an authority exactly the way
/// production does.
#[derive(Debug)]
struct TestMintVerifier;

impl hyprstream_ledger::MintVerifier for TestMintVerifier {
    fn verify(&self, _signing_input: &[u8], sig: &[u8]) -> Result<(), LedgerError> {
        if sig == TEST_MINT_SIG {
            Ok(())
        } else {
            Err(LedgerError::MintNotAuthorized(
                "bad test signature".to_owned(),
            ))
        }
    }
}

const TEST_MINT_SIG: &[u8] = b"test-mint-authorization";

/// Connect a ledger with the test issuance authority installed.
fn connect(db: &TestDb) -> Result<PostgresLedger, LedgerError> {
    PostgresLedger::connect(db.config(), ledger_id())
        .map(|l| l.with_mint_verifier(Box::new(TestMintVerifier)))
}

/// Authorize and issue in one step, the way the settlement issuer does.
fn mint(l: &mut PostgresLedger, t: IssueTransfer) -> hyprstream_ledger::Outcome {
    match l.authorize_mint(&t, TEST_MINT_SIG) {
        Ok(cap) => l.credit(cap),
        Err(e) => hyprstream_ledger::Outcome {
            result: Err(e),
            seq: l.head().seq,
        },
    }
}

struct NoopSigner(Did);
impl CheckpointSigner for NoopSigner {
    fn sign(&self, _input: &[u8]) -> Result<Vec<u8>, LedgerError> {
        Ok(vec![0u8; 64])
    }
    fn ledger_id(&self) -> &Did {
        &self.0
    }
}

/// Open the issuer-liability account plus `n` spendable accounts.
fn open_accounts(l: &mut PostgresLedger, n: usize) -> Vec<AccountId> {
    let mut ids = Vec::new();
    let liability = l
        .open_account(AccountSpec::new(
            ledger_id(),
            unit().issuer.clone(),
            unit(),
            Purpose::IssuerLiability,
        ))
        .unwrap();
    ids.push(liability.id);
    for i in 0..n {
        let a = l
            .open_account(AccountSpec::new(
                ledger_id(),
                Did(format!("did:key:owner{i}")),
                unit(),
                Purpose::Available,
            ))
            .unwrap();
        ids.push(a.id);
    }
    ids
}

fn issue(id: u128, liability: AccountId, dest: AccountId, amount: u128) -> IssueTransfer {
    IssueTransfer {
        id: TransferId(id),
        issuer_liability: liability,
        destination: dest,
        unit: unit(),
        amount,
        grant_cid: None,
        user_data: [0u8; 32],
    }
}

fn spend(id: u128, from: AccountId, to: AccountId, amount: u128) -> Transfer {
    Transfer {
        id: TransferId(id),
        debit_account: from,
        credit_account: to,
        unit: unit(),
        amount,
        grant_cid: None,
        user_data: [0u8; 32],
    }
}

// ─── 1. Single writer per cell ──────────────────────────────────────────────

#[test]
fn second_instance_is_refused_while_the_first_holds_the_lease() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    let first = connect(&db).unwrap();

    // A second writer for the same cell must not be constructed at all — the
    // mirror design is only sound with one writer, so admission is where that
    // is enforced.
    let second = connect(&db);
    match second {
        Err(LedgerError::WriterLeaseHeld { epoch }) => assert!(epoch >= 1),
        Err(e) => panic!("expected WriterLeaseHeld, got {e:?}"),
        Ok(_) => panic!("a second writer was admitted for the same cell"),
    }

    drop(first);
}

#[test]
fn a_clean_shutdown_releases_the_lease_for_the_next_instance() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    let first = connect(&db).unwrap();
    drop(first);

    // Restart must not have to wait out the heartbeat TTL.
    let second = connect(&db).expect("a released lease should be immediately claimable");
    drop(second);
}

#[test]
fn a_fenced_out_writer_cannot_commit_and_loses_no_update() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    // Instance A establishes state.
    let mut a = connect(&db).unwrap();
    let ids = open_accounts(&mut a, 2);
    let (liability, acct1, acct2) = (ids[0], ids[1], ids[2]);
    assert!(mint(&mut a, issue(1, liability, acct1, 1_000)).is_ok());
    let a_balance_before = a.balance(acct1).unwrap();

    // Simulate A crashing without releasing: expire its heartbeat in place.
    // The lease row still names A's epoch, so this models "A is gone" purely
    // from the database's point of view, while A is in fact still running.
    must_sql(
        &db.dsn(),
        "UPDATE ledger_meta \
         SET value = substring(value from 1 for 24) || '\\x0000000000000001'::bytea \
         WHERE key = 'writer_lease'",
    );

    // Instance B legitimately takes over, bumping the fencing epoch, and does
    // real work that moves acct1's balance.
    let mut b = connect(&db).expect("takeover of an expired lease should succeed");
    assert!(mint(&mut b, issue(2, liability, acct1, 500)).is_ok());
    let b_balance = b.balance(acct1).unwrap();

    // Now the zombie writes. Its mirror still reflects pre-takeover state, so
    // if this were allowed to commit, its absolute counters would erase B's
    // issuance. It must be refused.
    let zombie = mint(&mut a, issue(3, liability, acct2, 700));
    match zombie.result {
        Err(LedgerError::WriterLeaseLost { .. }) => {}
        other => panic!("expected the fenced-out writer to be refused, got {other:?}"),
    }

    // ...and it must stay refused: a poisoned instance never writes again.
    let again = a.debit(spend(4, acct1, acct2, 1));
    assert!(
        matches!(
            again.result,
            Err(LedgerError::Poisoned(_)) | Err(LedgerError::WriterLeaseLost { .. })
        ),
        "a fenced-out writer must remain refused, got {:?}",
        again.result
    );

    // B's update survived intact — no lost update.
    assert_eq!(
        b.balance(acct1).unwrap().available,
        b_balance.available,
        "B's committed balance changed underneath it"
    );
    assert!(
        b_balance.available > a_balance_before.available,
        "B's issuance should have increased the balance"
    );

    // And the durable state agrees with B, not with the zombie.
    drop(a);
    drop(b);
    let c = connect(&db).unwrap();
    assert_eq!(
        c.balance(acct1).unwrap().available,
        b_balance.available,
        "restart disagrees with the writer that actually held the lease"
    );
    // The zombie's transfer never landed.
    assert!(
        c.journal_range(0, 1000)
            .unwrap()
            .iter()
            .all(|e| e.op.idempotency_id() != Some(TransferId(3))),
        "a fenced-out writer's transfer reached the journal"
    );
}

#[test]
fn two_instances_with_conflicting_updates_do_not_lose_one() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    let mut a = connect(&db).unwrap();
    let ids = open_accounts(&mut a, 2);
    let (liability, acct1, acct2) = (ids[0], ids[1], ids[2]);
    mint(&mut a, issue(10, liability, acct1, 1_000))
        .result
        .unwrap();
    mint(&mut a, issue(11, liability, acct2, 1_000))
        .result
        .unwrap();

    // Both instances will touch the SAME accounts with DIFFERENT transfer ids,
    // which is precisely the shape that absolute-counter upserts lose.
    must_sql(
        &db.dsn(),
        "UPDATE ledger_meta \
         SET value = substring(value from 1 for 24) || '\\x0000000000000001'::bytea \
         WHERE key = 'writer_lease'",
    );
    let mut b = connect(&db).unwrap();

    // B moves 100 from acct1 to acct2.
    b.debit(spend(12, acct1, acct2, 100)).result.unwrap();
    let expected_1 = b.balance(acct1).unwrap().available;
    let expected_2 = b.balance(acct2).unwrap().available;

    // A, staging from its pre-takeover mirror, attempts a conflicting move.
    let lost = a.debit(spend(13, acct1, acct2, 250));
    assert!(
        lost.result.is_err(),
        "the stale writer's conflicting update was accepted: {:?}",
        lost.result
    );

    // The journal must describe exactly the state the tables hold. A restart
    // rebuilds the mirror from the journal and cross-checks the projection, so
    // a successful reconnect IS the agreement assertion.
    drop(a);
    drop(b);
    let c = connect(&db).expect("journal and materialized state disagree after concurrent writers");
    assert_eq!(c.balance(acct1).unwrap().available, expected_1);
    assert_eq!(c.balance(acct2).unwrap().available, expected_2);
}

// ─── 2. Journal-authoritative restart ───────────────────────────────────────

#[test]
fn restart_rebuilds_the_mirror_from_the_committed_journal() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    let (acct1, acct2, bal1, bal2, head) = {
        let mut l = connect(&db).unwrap();
        let ids = open_accounts(&mut l, 2);
        let (liability, a1, a2) = (ids[0], ids[1], ids[2]);
        mint(&mut l, issue(100, liability, a1, 5_000))
            .result
            .unwrap();
        l.debit(spend(101, a1, a2, 1_250)).result.unwrap();
        l.reserve(spend(102, a1, a2, 300), 3_600).result.unwrap();
        (
            a1,
            a2,
            l.balance(a1).unwrap(),
            l.balance(a2).unwrap(),
            l.head(),
        )
    };

    let l = connect(&db).unwrap();
    assert_eq!(l.head(), head, "head did not survive restart");
    assert_eq!(l.balance(acct1).unwrap(), bal1);
    assert_eq!(l.balance(acct2).unwrap(), bal2);
    // The pending hold must survive too, or a reserve could be double-spent.
    assert!(bal1.available < 5_000 - 1_250, "reserve should hold funds");
}

#[test]
fn a_replayed_transfer_id_returns_the_original_outcome_without_double_applying() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    let mut l = connect(&db).unwrap();
    let ids = open_accounts(&mut l, 1);
    let (liability, acct) = (ids[0], ids[1]);

    let first = mint(&mut l, issue(200, liability, acct, 400));
    assert!(matches!(first.result, Ok(TransferResult::Issued)));
    let after_first = l.balance(acct).unwrap();

    let replay = mint(&mut l, issue(200, liability, acct, 400));
    assert_eq!(replay.result, first.result, "replay changed the outcome");
    assert_eq!(
        replay.seq, first.seq,
        "replay changed the recorded sequence"
    );
    assert_eq!(
        l.balance(acct).unwrap(),
        after_first,
        "replay applied the transfer a second time"
    );

    // Replay across a restart must behave identically.
    drop(l);
    let mut l = connect(&db).unwrap();
    let replay2 = mint(&mut l, issue(200, liability, acct, 400));
    assert_eq!(replay2.result, first.result);
    assert_eq!(l.balance(acct).unwrap(), after_first);
}

#[test]
fn a_journal_gap_is_fatal_on_restart() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    {
        let mut l = connect(&db).unwrap();
        let ids = open_accounts(&mut l, 1);
        mint(&mut l, issue(300, ids[0], ids[1], 100))
            .result
            .unwrap();
        mint(&mut l, issue(301, ids[0], ids[1], 100))
            .result
            .unwrap();
    }

    // Remove an interior entry. Each surviving entry still re-hashes correctly
    // on its own, so this is caught by the cross-entry checks (contiguity and
    // chain linkage), never by per-entry hashing.
    must_sql(
        &db.dsn(),
        "DELETE FROM ledger_journal WHERE seq = (SELECT min(seq) + 1 FROM ledger_journal)",
    );

    let err = connect(&db);
    assert!(
        err.is_err(),
        "a journal with a missing entry must not be served"
    );
}

#[test]
fn a_truncated_journal_tail_is_fatal_on_restart() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    {
        let mut l = connect(&db).unwrap();
        let ids = open_accounts(&mut l, 1);
        mint(&mut l, issue(310, ids[0], ids[1], 100))
            .result
            .unwrap();
        mint(&mut l, issue(311, ids[0], ids[1], 100))
            .result
            .unwrap();
    }

    // Drop the LAST entry. Everything that remains is contiguous and chains
    // perfectly — this is only detectable by requiring the replayed tail to
    // equal the recorded head, which is why that check exists separately from
    // the hash walk.
    must_sql(
        &db.dsn(),
        "DELETE FROM ledger_journal WHERE seq = (SELECT max(seq) FROM ledger_journal)",
    );

    let err = connect(&db);
    assert!(
        err.is_err(),
        "a journal truncated at the tail must not be served"
    );
}

#[test]
fn materialized_state_diverging_from_the_journal_is_fatal_on_restart() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    {
        let mut l = connect(&db).unwrap();
        let ids = open_accounts(&mut l, 1);
        mint(&mut l, issue(400, ids[0], ids[1], 1_000))
            .result
            .unwrap();
    }

    // Tamper with the projection only — the journal is untouched and still
    // verifies, so this is caught solely by the journal-vs-materialized check.
    must_sql(
        &db.dsn(),
        "UPDATE ledger_accounts \
         SET credits_posted = '\\x000000000000000000000000deadbeef'::bytea \
         WHERE credits_posted <> '\\x00000000000000000000000000000000'::bytea",
    );

    let err = connect(&db);
    assert!(
        err.is_err(),
        "a projection that disagrees with the journal must not be served"
    );
}

#[test]
fn a_head_pointer_ahead_of_the_journal_is_fatal_on_restart() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    {
        let mut l = connect(&db).unwrap();
        let ids = open_accounts(&mut l, 1);
        mint(&mut l, issue(500, ids[0], ids[1], 100))
            .result
            .unwrap();
    }

    // Advance the recorded head past the last journal entry.
    must_sql(
        &db.dsn(),
        "UPDATE ledger_meta SET value = '\\x00000000000000ff'::bytea \
             || substring(value from 9 for 32) \
         WHERE key = 'head'",
    );

    let err = connect(&db);
    assert!(
        err.is_err(),
        "a head pointer ahead of the journal must not be served"
    );
}

#[test]
fn tick_and_checkpoint_survive_restart() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);
    let signer = NoopSigner(ledger_id());

    let head = {
        let mut l = connect(&db).unwrap();
        let ids = open_accounts(&mut l, 1);
        mint(&mut l, issue(600, ids[0], ids[1], 100))
            .result
            .unwrap();
        l.tick(&signer).unwrap();
        l.checkpoint(&signer).unwrap();
        l.head()
    };

    let l = connect(&db).unwrap();
    assert_eq!(l.head(), head);
    l.verify_chain().unwrap();
}

// ─── 3. The mint is sealed ──────────────────────────────────────────────────

#[test]
fn issuance_without_a_valid_authorization_is_refused() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    let mut l = connect(&db).unwrap();
    let ids = open_accounts(&mut l, 1);
    let (liability, acct) = (ids[0], ids[1]);

    let t = issue(800, liability, acct, 1_000);

    // A caller holding the backend still cannot mint: the capability `credit`
    // requires can only come from a verified, transfer-bound authorization.
    let err = l.authorize_mint(&t, b"forged").unwrap_err();
    assert!(
        matches!(err, LedgerError::MintNotAuthorized(_)),
        "expected the mint to be sealed, got {err:?}"
    );

    // Nothing was issued.
    assert!(
        l.balance(acct).unwrap().credits_posted == 0,
        "a refused issuance still moved value"
    );
}

#[test]
fn a_ledger_with_no_issuance_authority_cannot_mint_at_all() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    // Deliberately NOT installing a verifier: the default must be deny, so an
    // operator who has not wired an issuance authority gets a mint-disabled
    // ledger rather than a mint-open one.
    let mut l = PostgresLedger::connect(db.config(), ledger_id()).unwrap();
    let ids = open_accounts(&mut l, 1);
    let t = issue(801, ids[0], ids[1], 1_000);

    let err = l.authorize_mint(&t, TEST_MINT_SIG).unwrap_err();
    assert!(
        matches!(err, LedgerError::MintNotAuthorized(_)),
        "an unconfigured ledger must refuse to mint, got {err:?}"
    );
}

#[test]
fn an_authorization_cannot_be_moved_to_a_different_issuance() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    let mut l = connect(&db).unwrap();
    let ids = open_accounts(&mut l, 2);
    let (liability, acct1) = (ids[0], ids[1]);

    // The authorization binds the exact transfer, so the signature that covers
    // a 1-unit issuance does not cover a larger one. (The type system also
    // prevents reusing the capability itself, since it borrows its transfer.)
    let small = issue(802, liability, acct1, 1);
    let large = issue(802, liability, acct1, 1_000_000);
    assert_ne!(
        hyprstream_ledger::mint_signing_input(&small).unwrap(),
        hyprstream_ledger::mint_signing_input(&large).unwrap(),
        "authorization must bind the amount"
    );
}

// ─── 4. Migration of an existing database ───────────────────────────────────

#[test]
fn an_existing_64_bit_database_is_migrated_to_the_strict_representation() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    // Recreate the pre-migration schema: 128-bit values in BIGINT columns, plus
    // the obsolete denormalized result flag.
    must_sql(
        &db.dsn(),
        "CREATE TABLE ledger_journal (
             seq BIGSERIAL PRIMARY KEY, prev_hash BYTEA NOT NULL, ts BIGINT NOT NULL,
             op_cbor BYTEA NOT NULL, result_cbor BYTEA NOT NULL, head_hash BYTEA NOT NULL,
             result_ok BOOLEAN NOT NULL DEFAULT TRUE);
         CREATE TABLE ledger_accounts (
             id BIGINT PRIMARY KEY, unit_issuer TEXT NOT NULL, unit_resource TEXT NOT NULL,
             purpose_cbor BYTEA NOT NULL, debits_pending BIGINT NOT NULL,
             debits_posted BIGINT NOT NULL, credits_pending BIGINT NOT NULL,
             credits_posted BIGINT NOT NULL, flags INTEGER NOT NULL DEFAULT 0);
         CREATE TABLE ledger_pending (
             transfer_id BIGINT PRIMARY KEY, transfer_cbor BYTEA NOT NULL,
             deadline BIGINT NOT NULL, state SMALLINT NOT NULL DEFAULT 0);
         CREATE TABLE ledger_outcomes (
             transfer_id BIGINT PRIMARY KEY, result_cbor BYTEA NOT NULL, seq BIGINT NOT NULL);
         CREATE TABLE ledger_outbox (
             seq BIGSERIAL PRIMARY KEY, kind SMALLINT NOT NULL, transfer_id BIGINT,
             journal_seq BIGINT NOT NULL, emitted BOOLEAN NOT NULL DEFAULT FALSE);
         CREATE TABLE ledger_checkpoints (
             seq BIGSERIAL PRIMARY KEY, ledger_id TEXT NOT NULL, journal_seq BIGINT NOT NULL,
             head_hash BYTEA NOT NULL, balances_root BYTEA NOT NULL, pending_root BYTEA NOT NULL,
             ts BIGINT NOT NULL, prev_checkpoint_hash BYTEA NOT NULL, sig BYTEA NOT NULL,
             digest BYTEA NOT NULL);
         CREATE TABLE ledger_meta (key TEXT PRIMARY KEY, value BYTEA NOT NULL);",
    );

    // Connecting must upgrade the catalog rather than leave it as-is.
    let l = connect(&db).expect("connecting to a legacy database should migrate it");
    drop(l);

    for (table, column) in [
        ("ledger_accounts", "id"),
        ("ledger_accounts", "credits_posted"),
        ("ledger_pending", "transfer_id"),
        ("ledger_outcomes", "transfer_id"),
        ("ledger_outbox", "transfer_id"),
    ] {
        let ty = query_one_text(
            &db.dsn(),
            &format!(
                "SELECT data_type FROM information_schema.columns \
                 WHERE table_name = '{table}' AND column_name = '{column}'"
            ),
        );
        assert_eq!(ty, "bytea", "{table}.{column} was not migrated");
    }

    // The obsolete column is gone.
    let remaining = query_one_text(
        &db.dsn(),
        "SELECT count(*)::text FROM information_schema.columns \
         WHERE table_name = 'ledger_journal' AND column_name = 'result_ok'",
    );
    assert_eq!(
        remaining, "0",
        "obsolete result_ok column survived migration"
    );

    // The length constraints are actually installed.
    let checks = query_one_text(
        &db.dsn(),
        "SELECT count(*)::text FROM pg_constraint WHERE conname LIKE 'ck_ledger_%'",
    );
    assert!(
        checks.parse::<i64>().unwrap() >= 10,
        "expected the strict length constraints to be installed, found {checks}"
    );
}

#[test]
fn a_migrated_database_serves_traffic() {
    let dsn = require_pg!();
    let db = TestDb::create(&dsn);

    // Fresh install, some traffic, then reconnect: exercises the path where the
    // migration runs against an already-current catalog and must be a no-op.
    let bal = {
        let mut l = connect(&db).unwrap();
        let ids = open_accounts(&mut l, 1);
        mint(&mut l, issue(700, ids[0], ids[1], 9_000))
            .result
            .unwrap();
        (ids[1], l.balance(ids[1]).unwrap())
    };

    let l = connect(&db).unwrap();
    assert_eq!(l.balance(bal.0).unwrap(), bal.1);

    let version = query_one_text(
        &db.dsn(),
        "SELECT count(*)::text FROM ledger_meta WHERE key = 'schema_version'",
    );
    assert_eq!(version, "1", "schema version was not stamped");
}
