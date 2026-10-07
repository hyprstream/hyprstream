-- Migration-owner only; binds the signed Dex upstream ATProto DID to durable
-- source provenance. Old replay rows have no recoverable DID and therefore
-- remain unreadable as active primary records until their sessions expire.
BEGIN;
DO $$
BEGIN
  IF (SELECT count(*) FROM federate_session.schema_version WHERE version = 3) <> 1
  THEN RAISE EXCEPTION 'expected federate session schema v3'; END IF;
END $$;
ALTER TABLE federate_session.replay
  ADD COLUMN source_atproto_did text
    CHECK (source_atproto_did IS NULL OR length(source_atproto_did) BETWEEN 7 AND 255);
ALTER TABLE federate_session.schema_version
  DROP CONSTRAINT schema_version_version_check;
UPDATE federate_session.schema_version SET version = 4 WHERE version = 3;
ALTER TABLE federate_session.schema_version
  ADD CONSTRAINT schema_version_version_check CHECK (version = 4);
COMMIT;
