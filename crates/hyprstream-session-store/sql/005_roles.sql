-- Migration-owner only; apply after 005_identity_binding.sql.
BEGIN;
GRANT SELECT, INSERT ON federate_session.identity_bindings TO hs_policy_runtime;
COMMIT;
