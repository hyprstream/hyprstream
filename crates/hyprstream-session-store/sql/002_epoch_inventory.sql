-- Migration-owner only. Apply explicitly after 001_admission.sql and before
-- 002_roles.sql. No runtime path executes migrations or creates a profile.
-- Existing dynamic rows cannot be assigned a trustworthy inventory or epoch.
BEGIN;
DO $$
BEGIN
  IF (SELECT count(*) FROM federate_session.schema_version WHERE version = 1) <> 1
     OR EXISTS (SELECT 1 FROM federate_session.profile_state)
     OR EXISTS (SELECT 1 FROM federate_session.sessions)
     OR EXISTS (SELECT 1 FROM federate_session.replay) THEN
    RAISE EXCEPTION 'federate session v2 requires empty v1 profile and replay state';
  END IF;
END;
$$;

ALTER TABLE federate_session.profile_state
  ADD COLUMN collision_inventory_id bytea NOT NULL
    CHECK (octet_length(collision_inventory_id) = 32);

ALTER TABLE federate_session.sessions
  ADD COLUMN collision_inventory_id bytea NOT NULL
    CHECK (octet_length(collision_inventory_id) = 32),
  ADD COLUMN proof_epoch bigint GENERATED ALWAYS AS IDENTITY
    (START WITH 1 INCREMENT BY 1 NO CYCLE)
    CHECK (proof_epoch > 0),
  ADD CONSTRAINT sessions_proof_epoch_unique UNIQUE (proof_epoch),
  ADD CONSTRAINT sessions_host_ed_unique UNIQUE (host, ed_public),
  ADD CONSTRAINT sessions_host_pq_unique UNIQUE (host, pq_public);

-- Generation-only rotation with the same complete inventory ID is valid while
-- disabled. The control procedure must still issue one paired UPDATE setting
-- both columns; a row trigger cannot inspect which unchanged columns were SET.
CREATE FUNCTION federate_session.guard_profile_rotation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.collision_inventory_id IS DISTINCT FROM OLD.collision_inventory_id
     AND NEW.authority_generation IS NOT DISTINCT FROM OLD.authority_generation THEN
    RAISE EXCEPTION 'inventory change requires generation rotation';
  END IF;
  IF (NEW.authority_generation IS DISTINCT FROM OLD.authority_generation
      OR NEW.collision_inventory_id IS DISTINCT FROM OLD.collision_inventory_id)
     AND (OLD.enabled OR NEW.enabled) THEN
    RAISE EXCEPTION 'rotation requires disabled profile';
  END IF;
  RETURN NEW;
END;
$$;
CREATE TRIGGER guard_profile_rotation BEFORE UPDATE ON federate_session.profile_state
  FOR EACH ROW EXECUTE FUNCTION federate_session.guard_profile_rotation();

CREATE FUNCTION federate_session.guard_proof_epoch() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF NEW.proof_epoch IS DISTINCT FROM OLD.proof_epoch THEN
    RAISE EXCEPTION 'proof epoch is immutable';
  END IF;
  RETURN NEW;
END;
$$;
CREATE TRIGGER guard_proof_epoch BEFORE UPDATE ON federate_session.sessions
  FOR EACH ROW EXECUTE FUNCTION federate_session.guard_proof_epoch();

CREATE TABLE federate_session.request_replay (
  verified_namespace bytea NOT NULL CHECK (octet_length(verified_namespace) = 32),
  request_id bytea NOT NULL CHECK (octet_length(request_id) = 16),
  retain_until bigint NOT NULL CHECK (retain_until > 0),
  PRIMARY KEY (verified_namespace, request_id)
);
CREATE INDEX request_replay_cleanup ON federate_session.request_replay(retain_until);
REVOKE ALL ON federate_session.request_replay FROM PUBLIC;

ALTER TABLE federate_session.schema_version
  DROP CONSTRAINT schema_version_version_check;
UPDATE federate_session.schema_version SET version = 2 WHERE version = 1;
ALTER TABLE federate_session.schema_version
  ADD CONSTRAINT schema_version_version_check CHECK (version = 2);
COMMIT;
