-- Reviewed migration artifact. Apply once with an authorized migration role;
-- application startup never runs DDL. No live database is touched by this file.
-- The login role used by HYPRSTREAM_REPLAY_POSTGRES_URL_FILE must be granted
-- membership in hyprstream_replay_runtime by the database operator.

CREATE SCHEMA IF NOT EXISTS replay_admission;
REVOKE ALL ON SCHEMA replay_admission FROM PUBLIC;

CREATE TABLE IF NOT EXISTS replay_admission.entries_v1 (
    service_domain text NOT NULL CHECK (
        length(service_domain) BETWEEN 1 AND 128
        AND service_domain ~ '^[a-z0-9][a-z0-9._-]*$'
    ),
    partition smallint NOT NULL CHECK (partition IN (1, 2, 3)),
    key_digest bytea NOT NULL CHECK (octet_length(key_digest) = 32),
    expires_at bigint NOT NULL CHECK (expires_at > 0),
    PRIMARY KEY (service_domain, partition, key_digest)
);

CREATE INDEX IF NOT EXISTS entries_v1_expiry_idx
    ON replay_admission.entries_v1 (expires_at);

REVOKE ALL ON replay_admission.entries_v1 FROM PUBLIC;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hyprstream_replay_runtime') THEN
        CREATE ROLE hyprstream_replay_runtime NOLOGIN;
    ELSE
        IF EXISTS (
            SELECT 1 FROM pg_roles
            WHERE rolname = 'hyprstream_replay_runtime'
              AND (rolcanlogin OR rolsuper OR rolcreaterole OR rolcreatedb
                   OR rolbypassrls OR rolreplication)
        ) OR EXISTS (
            SELECT 1 FROM pg_auth_members
            WHERE member = 'hyprstream_replay_runtime'::regrole
        ) THEN
            RAISE EXCEPTION 'existing hyprstream_replay_runtime role is not an unprivileged NOLOGIN group';
        END IF;
    END IF;
END $$;

GRANT USAGE ON SCHEMA replay_admission TO hyprstream_replay_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON replay_admission.entries_v1
    TO hyprstream_replay_runtime;

-- No sequence, credential table, proof body, request body, or key material is
-- stored. key_digest is SHA-256 of fixed identifiers and length-prefixed
-- mediated dimensions. expires_at is an exclusive Unix-second deadline.
