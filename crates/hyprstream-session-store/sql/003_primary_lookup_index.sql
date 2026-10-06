-- Migration-owner only; required before using bounded per-use primary lookup.
-- The foreign key is keyed by (host, sid), but use-time source binding joins
-- from the session to replay on (host, sid, client_id). This index keeps that
-- exact lookup bounded as replay history grows.
BEGIN;
DO $$
BEGIN
  IF (SELECT count(*) FROM federate_session.schema_version WHERE version = 2) <> 1
  THEN RAISE EXCEPTION 'expected federate session schema v2'; END IF;
END $$;
CREATE INDEX replay_primary_lookup
  ON federate_session.replay(host, sid, client_id);
ALTER TABLE federate_session.schema_version
  DROP CONSTRAINT schema_version_version_check;
UPDATE federate_session.schema_version SET version = 3 WHERE version = 2;
ALTER TABLE federate_session.schema_version
  ADD CONSTRAINT schema_version_version_check CHECK (version = 3);
COMMIT;
