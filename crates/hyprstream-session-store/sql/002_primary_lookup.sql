-- Explicit migration-owner operation, never runtime boot behavior.
-- Legacy sessions have no collision provenance: refuse to fabricate it.
BEGIN;
DO $$ BEGIN
  IF EXISTS (SELECT 1 FROM federate_session.sessions)
     OR EXISTS (SELECT 1 FROM federate_session.profile_state WHERE enabled) THEN
    RAISE EXCEPTION 'disable and drain legacy session authority before migration';
  END IF;
END $$;
ALTER TABLE federate_session.profile_state ADD COLUMN collision_inventory_id bytea
  CHECK (octet_length(collision_inventory_id) = 32);
ALTER TABLE federate_session.profile_state ADD CHECK (NOT enabled OR collision_inventory_id IS NOT NULL);
ALTER TABLE federate_session.sessions
  ADD COLUMN proof_epoch bigint GENERATED ALWAYS AS IDENTITY (NO CYCLE) CHECK (proof_epoch > 0),
  ADD COLUMN collision_inventory_id bytea NOT NULL CHECK (octet_length(collision_inventory_id) = 32),
  ADD UNIQUE (host, ed_public), ADD UNIQUE (host, pq_public);
CREATE FUNCTION federate_session.guard_primary_lifecycle() RETURNS trigger
LANGUAGE plpgsql AS $$ BEGIN
  IF TG_TABLE_NAME = 'profile_state' THEN
    IF NEW.collision_inventory_id IS DISTINCT FROM OLD.collision_inventory_id
       AND NEW.authority_generation = OLD.authority_generation THEN
      RAISE EXCEPTION 'inventory change requires generation rotation';
    END IF;
  ELSIF NEW.proof_epoch <> OLD.proof_epoch
     OR NEW.collision_inventory_id <> OLD.collision_inventory_id
     OR NEW.ed_public <> OLD.ed_public OR NEW.pq_public <> OLD.pq_public
     OR NEW.generation <> OLD.generation THEN
    RAISE EXCEPTION 'session enrollment is immutable';
  END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER profile_inventory_rotation BEFORE UPDATE ON federate_session.profile_state
FOR EACH ROW EXECUTE FUNCTION federate_session.guard_primary_lifecycle();
CREATE TRIGGER immutable_primary BEFORE UPDATE ON federate_session.sessions
FOR EACH ROW EXECUTE FUNCTION federate_session.guard_primary_lifecycle();
ALTER TABLE federate_session.schema_version DROP CONSTRAINT schema_version_version_check;
UPDATE federate_session.schema_version SET version=2 WHERE version=1;
ALTER TABLE federate_session.schema_version ADD CHECK (version=2);
COMMIT;
