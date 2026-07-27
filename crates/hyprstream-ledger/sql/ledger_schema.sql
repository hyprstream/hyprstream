-- hyprstream-ledger durable Postgres schema (PAY-01 #1389, R2-F6 strict BYTEA)
-- Dedicated database per PAY-00 #6 (isolated from the identity store).
-- All CREATE TABLE IF NOT EXISTS — idempotent on every startup.
--
-- All 128-bit identifiers and amounts are strict 16-byte BYTEA with CHECK
-- constraints. tokio-postgres has no native i128, and BIGINT would
-- truncate/flip-sign u128 values.
--
-- Constraints are **named**, and the names match the ones `ledger_migrate.sql`
-- installs on an upgraded database. That is what makes the two paths converge:
-- a fresh database and a migrated database end up with the same catalog, and
-- the migration's `pg_constraint` guard correctly sees a fresh database as
-- already-constrained instead of adding redundant duplicates.

CREATE TABLE IF NOT EXISTS ledger_journal (
    seq         BIGSERIAL PRIMARY KEY,
    prev_hash   BYTEA NOT NULL,
    ts          BIGINT NOT NULL,
    op_cbor     BYTEA NOT NULL,
    result_cbor BYTEA NOT NULL,
    head_hash   BYTEA NOT NULL,
    CONSTRAINT ck_ledger_journal_prev_len CHECK (octet_length(prev_hash) = 32),
    CONSTRAINT ck_ledger_journal_head_len CHECK (octet_length(head_hash) = 32)
);

CREATE TABLE IF NOT EXISTS ledger_accounts (
    id               BYTEA PRIMARY KEY,
    unit_issuer      TEXT NOT NULL,
    unit_resource    TEXT NOT NULL,
    purpose_cbor     BYTEA NOT NULL,
    debits_pending   BYTEA NOT NULL,
    debits_posted    BYTEA NOT NULL,
    credits_pending  BYTEA NOT NULL,
    credits_posted   BYTEA NOT NULL,
    flags            INTEGER NOT NULL DEFAULT 0,
    CONSTRAINT ck_ledger_accounts_id_len  CHECK (octet_length(id) = 16),
    CONSTRAINT ck_ledger_accounts_dp_len  CHECK (octet_length(debits_pending) = 16),
    CONSTRAINT ck_ledger_accounts_dpo_len CHECK (octet_length(debits_posted) = 16),
    CONSTRAINT ck_ledger_accounts_cp_len  CHECK (octet_length(credits_pending) = 16),
    CONSTRAINT ck_ledger_accounts_cpo_len CHECK (octet_length(credits_posted) = 16)
);

CREATE TABLE IF NOT EXISTS ledger_pending (
    transfer_id   BYTEA PRIMARY KEY,
    transfer_cbor BYTEA NOT NULL,
    deadline      BIGINT NOT NULL,
    state         SMALLINT NOT NULL DEFAULT 0,
    CONSTRAINT ck_ledger_pending_tid_len CHECK (octet_length(transfer_id) = 16)
);

CREATE TABLE IF NOT EXISTS ledger_outcomes (
    transfer_id  BYTEA PRIMARY KEY,
    result_cbor  BYTEA NOT NULL,
    seq          BIGINT NOT NULL,
    CONSTRAINT ck_ledger_outcomes_tid_len CHECK (octet_length(transfer_id) = 16)
);

CREATE TABLE IF NOT EXISTS ledger_outbox (
    seq          BIGSERIAL PRIMARY KEY,
    kind         SMALLINT NOT NULL,
    transfer_id  BYTEA,
    journal_seq  BIGINT NOT NULL,
    emitted      BOOLEAN NOT NULL DEFAULT FALSE,
    CONSTRAINT ck_ledger_outbox_tid_len
        CHECK (transfer_id IS NULL OR octet_length(transfer_id) = 16)
);

CREATE TABLE IF NOT EXISTS ledger_checkpoints (
    seq                  BIGSERIAL PRIMARY KEY,
    ledger_id            TEXT NOT NULL,
    journal_seq          BIGINT NOT NULL,
    head_hash            BYTEA NOT NULL,
    balances_root        BYTEA NOT NULL,
    pending_root         BYTEA NOT NULL,
    ts                   BIGINT NOT NULL,
    prev_checkpoint_hash BYTEA NOT NULL,
    sig                  BYTEA NOT NULL,
    digest               BYTEA NOT NULL,
    CONSTRAINT ck_ledger_cp_head_len   CHECK (octet_length(head_hash) = 32),
    CONSTRAINT ck_ledger_cp_bal_len    CHECK (octet_length(balances_root) = 32),
    CONSTRAINT ck_ledger_cp_pen_len    CHECK (octet_length(pending_root) = 32),
    CONSTRAINT ck_ledger_cp_prev_len   CHECK (octet_length(prev_checkpoint_hash) = 32),
    CONSTRAINT ck_ledger_cp_digest_len CHECK (octet_length(digest) = 32)
);

-- Generic key/value metadata. Carries the chain head ('head'), the logical
-- clock ('clock'), the schema version ('schema_version'), and the
-- single-writer fencing lease ('writer_lease').
CREATE TABLE IF NOT EXISTS ledger_meta (
    key   TEXT PRIMARY KEY,
    value BYTEA NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outbox_unemitted ON ledger_outbox (seq) WHERE emitted = FALSE;
