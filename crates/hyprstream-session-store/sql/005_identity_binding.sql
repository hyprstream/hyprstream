-- Migration-owner only. Permanently binds one Dex source identity, one verified
-- ATProto DID, and one local account UUID within each host/profile boundary.
BEGIN;
DO $$
BEGIN
  IF (SELECT count(*) FROM federate_session.schema_version WHERE version = 4) <> 1
  THEN RAISE EXCEPTION 'expected federate session schema v4'; END IF;
END $$;
CREATE TABLE federate_session.identity_bindings (
  host text NOT NULL,
  profile text NOT NULL CHECK (profile = 'federate-session-v1'),
  issuer text NOT NULL CHECK (length(issuer) BETWEEN 1 AND 2048),
  source_subject text NOT NULL CHECK (length(source_subject) BETWEEN 1 AND 256),
  atproto_did text NOT NULL CHECK (length(atproto_did) BETWEEN 7 AND 255),
  account_id text NOT NULL CHECK (length(account_id) BETWEEN 1 AND 256),
  first_seen_at bigint NOT NULL CHECK (first_seen_at >= 0),
  PRIMARY KEY (host, profile, issuer, source_subject),
  UNIQUE (host, profile, atproto_did),
  UNIQUE (host, profile, account_id),
  FOREIGN KEY (host, profile)
    REFERENCES federate_session.profile_state(host, profile)
);
-- Preserve consistent v4 bindings, but reject any history that would require
-- choosing which account or DID was canonical. Legacy rows without a verified
-- DID remain unusable as primary records and are intentionally not guessed.
INSERT INTO federate_session.identity_bindings
  (host, profile, issuer, source_subject, atproto_did, account_id, first_seen_at)
SELECT s.host, s.profile, r.issuer, r.source_subject, r.source_atproto_did,
       s.account_id, min(s.created_at)
FROM federate_session.sessions s
JOIN federate_session.replay r ON (r.host=s.host AND r.sid=s.sid)
WHERE r.source_atproto_did IS NOT NULL
  AND s.subject = r.source_atproto_did
GROUP BY s.host, s.profile, r.issuer, r.source_subject,
         r.source_atproto_did, s.account_id;
REVOKE ALL ON federate_session.identity_bindings FROM PUBLIC;
ALTER TABLE federate_session.schema_version
  DROP CONSTRAINT schema_version_version_check;
UPDATE federate_session.schema_version SET version = 5 WHERE version = 4;
ALTER TABLE federate_session.schema_version
  ADD CONSTRAINT schema_version_version_check CHECK (version = 5);
COMMIT;
