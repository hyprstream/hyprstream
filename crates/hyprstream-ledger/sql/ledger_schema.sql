-- hyprstream-ledger durable Postgres schema (PAY-01 #1389)
-- Dedicated database per PAY-00 #6 (isolated from the identity store).
-- All CREATE TABLE IF NOT EXISTS — idempotent on every startup.
--
-- Amounts are stored as BYTEA (16-byte big-endian u128) because
-- tokio-postgres has no native i128 support.

CREATE TABLE IF NOT EXISTS ledger_journal (
    seq         BIGSERIAL PRIMARY KEY,
    prev_hash   BYTEA NOT NULL DEFAULT '\x00',
    ts          BIGINT NOT NULL,
    op_cbor     BYTEA NOT NULL,
    result_ok   BOOLEAN NOT NULL,
    result_cbor BYTEA NOT NULL,
    head_hash   BYTEA NOT NULL
);

CREATE TABLE IF NOT EXISTS ledger_accounts (
    id               BIGINT PRIMARY KEY,
    unit_issuer      TEXT NOT NULL,
    unit_resource    TEXT NOT NULL,
    purpose_cbor     BYTEA NOT NULL,
    debits_pending   BYTEA NOT NULL,
    debits_posted    BYTEA NOT NULL,
    credits_pending  BYTEA NOT NULL,
    credits_posted   BYTEA NOT NULL,
    flags            INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ledger_pending (
    transfer_id   BIGINT PRIMARY KEY,
    transfer_cbor BYTEA NOT NULL,
    deadline      BIGINT NOT NULL,
    state         SMALLINT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ledger_outcomes (
    transfer_id  BIGINT PRIMARY KEY,
    result_ok    BOOLEAN NOT NULL,
    result_cbor  BYTEA NOT NULL,
    seq          BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS ledger_outbox (
    seq          BIGSERIAL PRIMARY KEY,
    kind         SMALLINT NOT NULL,
    transfer_id  BIGINT,
    journal_seq  BIGINT NOT NULL,
    emitted      BOOLEAN NOT NULL DEFAULT FALSE
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
    digest               BYTEA NOT NULL
);

CREATE TABLE IF NOT EXISTS ledger_meta (
    key   TEXT PRIMARY KEY,
    value BYTEA NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_outbox_unemitted ON ledger_outbox (seq) WHERE emitted = FALSE;
