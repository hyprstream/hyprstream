-- Migration-owner only, applied explicitly; never executed by the runtime crate.
-- Roles are provisioned by the deployment operator (see roles.sql).
BEGIN;
CREATE SCHEMA federate_session;
REVOKE ALL ON SCHEMA federate_session FROM PUBLIC;
CREATE TABLE federate_session.schema_version (
    version integer PRIMARY KEY CHECK (version = 1)
);
INSERT INTO federate_session.schema_version VALUES (1);
CREATE TABLE federate_session.profile_state (
    host text NOT NULL CHECK (length(host) BETWEEN 1 AND 2048),
    profile text NOT NULL CHECK (profile = 'federate-session-v1'),
    enabled boolean NOT NULL DEFAULT false,
    authority_generation bytea NOT NULL CHECK (octet_length(authority_generation) = 32),
    lock_version bigint NOT NULL DEFAULT 0,
    PRIMARY KEY (host, profile)
);
-- No profile is seeded/enabled by migration. An absent row denies admission.
CREATE TABLE federate_session.sessions (
    host text NOT NULL,
    profile text NOT NULL,
    sid text NOT NULL CHECK (length(sid) BETWEEN 1 AND 128),
    account_id text NOT NULL CHECK (length(account_id) BETWEEN 1 AND 256),
    subject text NOT NULL CHECK (length(subject) BETWEEN 1 AND 256),
    tenant text NOT NULL CHECK (length(tenant) BETWEEN 1 AND 256),
    client_id text NOT NULL CHECK (length(client_id) BETWEEN 1 AND 256),
    resource text NOT NULL CHECK (length(resource) BETWEEN 1 AND 2048),
    scopes text[] NOT NULL CHECK (cardinality(scopes) BETWEEN 1 AND 64),
    grant_revision text NOT NULL CHECK (length(grant_revision) BETWEEN 1 AND 256),
    suite text NOT NULL CHECK (suite = 'hs-cose-sign-ed25519-mldsa65-wns-v1'),
    ed_public bytea NOT NULL CHECK (octet_length(ed_public) = 32),
    pq_public bytea NOT NULL CHECK (octet_length(pq_public) = 1952),
    generation bytea NOT NULL CHECK (octet_length(generation) = 32),
    created_at bigint NOT NULL CHECK (created_at >= 0),
    expires_at bigint NOT NULL CHECK (expires_at > created_at AND expires_at - created_at <= 300),
    status text NOT NULL DEFAULT 'active' CHECK (status IN ('active','revoked')),
    PRIMARY KEY (host, sid),
    FOREIGN KEY (host, profile) REFERENCES federate_session.profile_state(host, profile)
);
CREATE TABLE federate_session.replay (
    issuer text NOT NULL CHECK (length(issuer) BETWEEN 1 AND 2048),
    client_id text NOT NULL CHECK (length(client_id) BETWEEN 1 AND 256),
    source_subject text NOT NULL CHECK (length(source_subject) BETWEEN 1 AND 256),
    jti text NOT NULL CHECK (length(jti) BETWEEN 1 AND 256),
    nonce text NOT NULL CHECK (length(nonce) BETWEEN 1 AND 256),
    token_hash bytea NOT NULL UNIQUE CHECK (octet_length(token_hash) = 32),
    source_iat bigint NOT NULL CHECK (source_iat >= 0),
    source_exp bigint NOT NULL CHECK (source_exp > source_iat AND source_exp - source_iat <= 300),
    retain_until bigint NOT NULL CHECK (retain_until >= source_iat + 330 AND retain_until >= source_exp + 30),
    host text NOT NULL,
    sid text NOT NULL,
    PRIMARY KEY (issuer, client_id, jti),
    UNIQUE (issuer, client_id, nonce),
    FOREIGN KEY (host, sid) REFERENCES federate_session.sessions(host, sid)
);
CREATE INDEX replay_cleanup ON federate_session.replay(retain_until);
CREATE INDEX session_cleanup ON federate_session.sessions(expires_at);
REVOKE ALL ON ALL TABLES IN SCHEMA federate_session FROM PUBLIC;
COMMIT;
