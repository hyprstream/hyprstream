-- Migration-owner only. Apply after 002_epoch_inventory.sql and the v1 grants.
-- The operator must audit effective privileges/role membership separately:
-- these narrow grants do not cancel a pre-existing broader privilege.
BEGIN;
GRANT UPDATE (collision_inventory_id)
  ON federate_session.profile_state TO hs_profile_control;
GRANT INSERT ON federate_session.request_replay TO hs_policy_runtime;
GRANT SELECT, DELETE ON federate_session.request_replay TO hs_session_cleanup;
COMMIT;
