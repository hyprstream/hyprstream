-- hyprstream-ledger schema migration (PAY-01 #1389, R4 finding 4).
--
-- `ledger_schema.sql` only ever runs CREATE TABLE IF NOT EXISTS, so it is a
-- no-op against a database created by an earlier build. This file brings such
-- a database up to the strict 128-bit representation, and is idempotent: every
-- step is guarded on the current catalog state, so re-running it does nothing.
--
-- ## What changed, and what a migrated row means
--
-- The original schema stored 128-bit ids and amounts in BIGINT. The writer
-- cast `u128 as i64`, so any value that did not fit in 64 bits was **already
-- truncated when it was written** — that loss happened at write time and is
-- not recoverable here. `int8send` reproduces exactly the 8 bytes that were
-- stored and zero-extends them into the 16-byte big-endian representation, so
-- migration is faithful to what is actually on disk. It does not invent the
-- lost high bits. Operators upgrading a database that carried values above
-- 2^63 must treat those rows as suspect; the loud NOTICE below reports how
-- many rows were converted.
--
-- After migration the CHECK constraints plus the strict Rust decoder make any
-- further silent truncation impossible: a row that is not exactly 16 bytes
-- fails closed instead of being zero-filled.

DO $migrate$
DECLARE
    legacy_rows BIGINT := 0;
    n BIGINT;
BEGIN
    -- ── 1. 128-bit id / amount columns: BIGINT → BYTEA(16) ──────────────────
    --
    -- (table, column) pairs that carry a 128-bit value. `int8send` yields the
    -- 8-byte big-endian encoding of the stored BIGINT; prefixing 8 zero bytes
    -- zero-extends it to the 16-byte big-endian representation the current
    -- writer emits.
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ledger_accounts' AND column_name = 'id'
          AND data_type = 'bigint'
    ) THEN
        SELECT count(*) INTO n FROM ledger_accounts;
        legacy_rows := legacy_rows + n;
        ALTER TABLE ledger_accounts
            ALTER COLUMN id TYPE BYTEA
            USING ('\x0000000000000000'::bytea || int8send(id));
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ledger_accounts' AND column_name = 'debits_pending'
          AND data_type = 'bigint'
    ) THEN
        ALTER TABLE ledger_accounts
            ALTER COLUMN debits_pending  TYPE BYTEA
                USING ('\x0000000000000000'::bytea || int8send(debits_pending)),
            ALTER COLUMN debits_posted   TYPE BYTEA
                USING ('\x0000000000000000'::bytea || int8send(debits_posted)),
            ALTER COLUMN credits_pending TYPE BYTEA
                USING ('\x0000000000000000'::bytea || int8send(credits_pending)),
            ALTER COLUMN credits_posted  TYPE BYTEA
                USING ('\x0000000000000000'::bytea || int8send(credits_posted));
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ledger_pending' AND column_name = 'transfer_id'
          AND data_type = 'bigint'
    ) THEN
        SELECT count(*) INTO n FROM ledger_pending;
        legacy_rows := legacy_rows + n;
        ALTER TABLE ledger_pending
            ALTER COLUMN transfer_id TYPE BYTEA
            USING ('\x0000000000000000'::bytea || int8send(transfer_id));
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ledger_outcomes' AND column_name = 'transfer_id'
          AND data_type = 'bigint'
    ) THEN
        SELECT count(*) INTO n FROM ledger_outcomes;
        legacy_rows := legacy_rows + n;
        ALTER TABLE ledger_outcomes
            ALTER COLUMN transfer_id TYPE BYTEA
            USING ('\x0000000000000000'::bytea || int8send(transfer_id));
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ledger_outbox' AND column_name = 'transfer_id'
          AND data_type = 'bigint'
    ) THEN
        ALTER TABLE ledger_outbox
            ALTER COLUMN transfer_id TYPE BYTEA
            USING (CASE WHEN transfer_id IS NULL THEN NULL
                        ELSE '\x0000000000000000'::bytea || int8send(transfer_id) END);
    END IF;

    -- ── 2. Drop the obsolete denormalized result flag ───────────────────────
    --
    -- `result_cbor` carries the full Result<TransferResult, LedgerError> (F2);
    -- a separate NOT NULL boolean is both redundant and a write hazard.
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'ledger_journal' AND column_name = 'result_ok'
    ) THEN
        ALTER TABLE ledger_journal DROP COLUMN result_ok;
    END IF;

    IF legacy_rows > 0 THEN
        RAISE NOTICE
            'hyprstream-ledger: migrated % legacy 64-bit row(s) to the strict 128-bit representation. Values written above 2^63 by the pre-migration writer were truncated at write time and cannot be recovered; audit these rows.',
            legacy_rows;
    END IF;
END
$migrate$;

-- ── 3. Strict length constraints ────────────────────────────────────────────
--
-- Added separately (and idempotently) so they also land on databases whose
-- columns were already BYTEA but were created before the CHECKs existed. A
-- NOT VALID + VALIDATE pair would let bad rows survive, so these are validating
-- constraints: if an existing row violates one, migration fails closed and the
-- operator must reconcile before the ledger will serve traffic.
DO $checks$
DECLARE
    c RECORD;
BEGIN
    FOR c IN
        SELECT * FROM (VALUES
            ('ledger_accounts',   'ck_ledger_accounts_id_len',        'octet_length(id) = 16'),
            ('ledger_accounts',   'ck_ledger_accounts_dp_len',        'octet_length(debits_pending) = 16'),
            ('ledger_accounts',   'ck_ledger_accounts_dpo_len',       'octet_length(debits_posted) = 16'),
            ('ledger_accounts',   'ck_ledger_accounts_cp_len',        'octet_length(credits_pending) = 16'),
            ('ledger_accounts',   'ck_ledger_accounts_cpo_len',       'octet_length(credits_posted) = 16'),
            ('ledger_pending',    'ck_ledger_pending_tid_len',        'octet_length(transfer_id) = 16'),
            ('ledger_outcomes',   'ck_ledger_outcomes_tid_len',       'octet_length(transfer_id) = 16'),
            ('ledger_outbox',     'ck_ledger_outbox_tid_len',         'transfer_id IS NULL OR octet_length(transfer_id) = 16'),
            ('ledger_journal',    'ck_ledger_journal_prev_len',       'octet_length(prev_hash) = 32'),
            ('ledger_journal',    'ck_ledger_journal_head_len',       'octet_length(head_hash) = 32'),
            ('ledger_checkpoints','ck_ledger_cp_head_len',            'octet_length(head_hash) = 32'),
            ('ledger_checkpoints','ck_ledger_cp_bal_len',             'octet_length(balances_root) = 32'),
            ('ledger_checkpoints','ck_ledger_cp_pen_len',             'octet_length(pending_root) = 32'),
            ('ledger_checkpoints','ck_ledger_cp_prev_len',            'octet_length(prev_checkpoint_hash) = 32'),
            ('ledger_checkpoints','ck_ledger_cp_digest_len',          'octet_length(digest) = 32')
        ) AS t(tbl, name, expr)
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint WHERE conname = c.name
        ) THEN
            EXECUTE format('ALTER TABLE %I ADD CONSTRAINT %I CHECK (%s)',
                           c.tbl, c.name, c.expr);
        END IF;
    END LOOP;
END
$checks$;
