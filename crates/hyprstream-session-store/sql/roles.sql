-- Apply explicitly as migration owner AFTER creating dedicated NOLOGIN roles
-- hs_policy_runtime, hs_profile_control, hs_session_cleanup. Login identities
-- inherit only their required role. No role may inherit migration ownership.
-- Do not use these roles for the existing account store.
BEGIN;
GRANT USAGE ON SCHEMA federate_session
  TO hs_policy_runtime, hs_profile_control, hs_session_cleanup;
GRANT SELECT ON federate_session.schema_version TO hs_policy_runtime;
GRANT SELECT ON federate_session.profile_state TO hs_policy_runtime, hs_profile_control;
GRANT UPDATE (lock_version) ON federate_session.profile_state TO hs_policy_runtime;
GRANT UPDATE (enabled, authority_generation, collision_inventory_id) ON federate_session.profile_state TO hs_profile_control;
GRANT SELECT ON federate_session.sessions TO hs_policy_runtime;
REVOKE INSERT ON federate_session.sessions FROM hs_policy_runtime;
GRANT INSERT (host,profile,sid,account_id,subject,tenant,client_id,resource,scopes,grant_revision,suite,ed_public,pq_public,generation,created_at,expires_at,collision_inventory_id) ON federate_session.sessions TO hs_policy_runtime;
GRANT USAGE ON SEQUENCE federate_session.sessions_proof_epoch_seq TO hs_policy_runtime;
GRANT SELECT, INSERT ON federate_session.replay TO hs_policy_runtime;
GRANT UPDATE (status) ON federate_session.sessions TO hs_policy_runtime;
GRANT SELECT, DELETE ON federate_session.replay, federate_session.sessions TO hs_session_cleanup;
COMMIT;
