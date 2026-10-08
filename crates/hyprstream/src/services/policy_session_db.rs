//! Role-scoped Federate session connection for Policy.
//!
//! This module never migrates or enables a profile. A migration owner applies
//! the reviewed SQL before deployment; a separate profile-control identity
//! enables the exact generation. Policy's runtime login is checked here before
//! any session or request-replay reader can be installed.

use anyhow::{ensure, Context, Result};
use deadpool_postgres::Pool;
use std::path::Path;

use crate::auth::postgres_store::build_verified_postgres_pool;

const POLICY_SESSION_ROLE_SQL: &str =
    "SELECT \
     has_schema_privilege(current_user, 'federate_session', 'USAGE') AND \
     NOT has_schema_privilege(current_user, 'federate_session', 'CREATE') AND \
     NOT has_database_privilege(current_user, current_database(), 'CREATE') AND \
     has_table_privilege(current_user, 'federate_session.profile_state', 'SELECT') AND \
     has_table_privilege(current_user, 'federate_session.sessions', 'SELECT') AND \
     has_table_privilege(current_user, 'federate_session.sessions', 'INSERT') AND \
     has_table_privilege(current_user, 'federate_session.replay', 'SELECT') AND \
     has_table_privilege(current_user, 'federate_session.replay', 'INSERT') AND \
     has_table_privilege(current_user, 'federate_session.identity_bindings', 'SELECT') AND \
     has_table_privilege(current_user, 'federate_session.identity_bindings', 'INSERT') AND \
     has_table_privilege(current_user, 'federate_session.request_replay', 'INSERT') AND \
     has_column_privilege(current_user, 'federate_session.profile_state', 'lock_version', 'UPDATE') AND \
     has_column_privilege(current_user, 'federate_session.sessions', 'status', 'UPDATE') AND \
     NOT has_any_column_privilege(current_user, 'federate_session.profile_state', 'INSERT') AND \
     NOT has_table_privilege(current_user, 'federate_session.profile_state', 'DELETE,TRUNCATE') AND \
     NOT has_table_privilege(current_user, 'federate_session.sessions', 'DELETE,TRUNCATE') AND \
     NOT has_any_column_privilege(current_user, 'federate_session.replay', 'UPDATE') AND \
     NOT has_table_privilege(current_user, 'federate_session.replay', 'DELETE,TRUNCATE') AND \
     NOT has_any_column_privilege(current_user, 'federate_session.identity_bindings', 'UPDATE') AND \
     NOT has_table_privilege(current_user, 'federate_session.identity_bindings', 'DELETE,TRUNCATE') AND \
     NOT has_any_column_privilege(current_user, 'federate_session.request_replay', 'SELECT,UPDATE') AND \
     NOT has_table_privilege(current_user, 'federate_session.request_replay', 'DELETE,TRUNCATE') AND \
     NOT EXISTS (SELECT 1 FROM information_schema.columns c \
       WHERE c.table_schema='federate_session' AND c.table_name='profile_state' \
         AND c.column_name <> 'lock_version' \
         AND has_column_privilege(current_user, 'federate_session.profile_state', c.column_name, 'UPDATE')) AND \
     NOT EXISTS (SELECT 1 FROM information_schema.columns c \
       WHERE c.table_schema='federate_session' AND c.table_name='sessions' \
         AND c.column_name <> 'status' \
         AND has_column_privilege(current_user, 'federate_session.sessions', c.column_name, 'UPDATE'))";

#[allow(dead_code)] // The Policy factory installs this with the trusted profile loader.
pub(super) struct PolicySessionPool(Pool);

#[allow(dead_code)]
impl PolicySessionPool {
    pub(super) async fn from_env() -> Result<Self> {
        let url_file = std::env::var_os("HYPRSTREAM_FEDERATE_SESSION_URL_FILE")
            .context("HYPRSTREAM_FEDERATE_SESSION_URL_FILE is required")?;
        let ca_file = std::env::var_os("HYPRSTREAM_FEDERATE_SESSION_SSLROOTCERT_FILE")
            .context("HYPRSTREAM_FEDERATE_SESSION_SSLROOTCERT_FILE is required")?;
        Self::from_files(Path::new(&url_file), Path::new(&ca_file)).await
    }

    async fn from_files(url_file: &Path, ca_file: &Path) -> Result<Self> {
        ensure!(ca_file.is_file(), "Federate session CA path is not a file");
        let url = std::fs::read_to_string(url_file).context("read Federate session URL file")?;
        let url = url.trim();
        ensure!(!url.is_empty(), "Federate session URL file is empty");
        // The shared pool builder rejects plaintext or hostname-unverified
        // connections and pins the supplied CA. The URL is never logged.
        let pool = build_verified_postgres_pool(url, ca_file, 4)?;
        let client = pool
            .get()
            .await
            .context("connect Federate session database")?;
        let version: i32 = client
            .query_one("SELECT version FROM federate_session.schema_version", &[])
            .await
            .context("read Federate session schema version")?
            .get(0);
        ensure!(version == 5, "Federate session schema version mismatch");
        let role: bool = client
            .query_one(POLICY_SESSION_ROLE_SQL, &[])
            .await
            .context("verify Federate Policy runtime privileges")?
            .get(0);
        ensure!(role, "Federate Policy runtime privileges invalid");
        drop(client);
        Ok(Self(pool))
    }

    pub(super) fn into_pool(self) -> Pool {
        self.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn missing_tls_material_denies_before_database_contact() {
        let dir = tempfile::tempdir().unwrap();
        let url = dir.path().join("url");
        std::fs::write(
            &url,
            "postgresql://user:password@db.example.test/db?sslmode=verify-full",
        )
        .unwrap();
        let error = PolicySessionPool::from_files(&url, &dir.path().join("missing-ca"))
            .await
            .err()
            .unwrap();
        assert!(error.to_string().contains("CA path"));
        assert!(!error.to_string().contains("password"));
    }

    #[tokio::test]
    #[ignore = "requires an isolated hyprstream_federate_test_* PostgreSQL database"]
    async fn runtime_role_can_admit_but_cannot_control_profile_or_schema() {
        let path = std::env::var_os("HYPRSTREAM_FEDERATE_SESSION_TEST_URL_FILE").unwrap();
        let url = std::fs::read_to_string(path).unwrap();
        let (client, connection) = tokio_postgres::connect(url.trim(), tokio_postgres::NoTls)
            .await
            .unwrap();
        tokio::spawn(async move { connection.await.unwrap() });
        let database: String = client
            .query_one("SELECT current_database()", &[])
            .await
            .unwrap()
            .get(0);
        assert!(database.starts_with("hyprstream_federate_test_"));
        client
            .batch_execute(
                "CREATE ROLE hs_policy_runtime NOLOGIN; \
                 CREATE ROLE hs_profile_control NOLOGIN; \
                 CREATE ROLE hs_session_cleanup NOLOGIN; \
                 CREATE ROLE hs_session_migration NOLOGIN",
            )
            .await
            .unwrap();
        client
            .batch_execute(hyprstream_session_store::MIGRATION)
            .await
            .unwrap();
        client
            .batch_execute(hyprstream_session_store::ROLE_GRANTS)
            .await
            .unwrap();
        client
            .batch_execute(hyprstream_session_store::MIGRATION_V2)
            .await
            .unwrap();
        client
            .batch_execute(hyprstream_session_store::ROLE_GRANTS_V2)
            .await
            .unwrap();
        client
            .batch_execute(hyprstream_session_store::MIGRATION_V3)
            .await
            .unwrap();
        client
            .batch_execute(hyprstream_session_store::MIGRATION_V4)
            .await
            .unwrap();
        client
            .batch_execute(hyprstream_session_store::MIGRATION_V5)
            .await
            .unwrap();
        client
            .batch_execute(hyprstream_session_store::ROLE_GRANTS_V5)
            .await
            .unwrap();
        client
            .batch_execute("SET ROLE hs_policy_runtime")
            .await
            .unwrap();
        let allowed: bool = client
            .query_one(POLICY_SESSION_ROLE_SQL, &[])
            .await
            .unwrap()
            .get(0);
        assert!(allowed, "the narrow runtime role must pass");
        assert!(client
            .batch_execute("CREATE TABLE federate_session.forbidden(id integer)")
            .await
            .is_err());
        client.batch_execute("RESET ROLE").await.unwrap();
        client
            .batch_execute(
                "GRANT UPDATE(atproto_did) ON federate_session.identity_bindings TO hs_policy_runtime; \
                 SET ROLE hs_policy_runtime",
            )
            .await
            .unwrap();
        let allowed: bool = client
            .query_one(POLICY_SESSION_ROLE_SQL, &[])
            .await
            .unwrap()
            .get(0);
        assert!(!allowed, "identity bindings must be immutable to runtime");
        client
            .batch_execute(
                "RESET ROLE; \
                 REVOKE UPDATE(atproto_did) ON federate_session.identity_bindings FROM hs_policy_runtime; \
                 SET ROLE hs_policy_runtime",
            )
            .await
            .unwrap();
        client
            .batch_execute(
                "INSERT INTO federate_session.request_replay \
                 (verified_namespace, request_id, retain_until) \
                 VALUES (decode(repeat('01',32),'hex'), decode(repeat('02',16),'hex'), 4102444800)",
            )
            .await
            .unwrap();
        let move_error = client
            .batch_execute(
                "UPDATE federate_session.request_replay \
                 SET request_id = decode(repeat('03',16),'hex')",
            )
            .await
            .unwrap_err();
        assert_eq!(
            move_error.code(),
            Some(&tokio_postgres::error::SqlState::INSUFFICIENT_PRIVILEGE)
        );
        let duplicate_error = client
            .batch_execute(
                "INSERT INTO federate_session.request_replay \
                 (verified_namespace, request_id, retain_until) \
                 VALUES (decode(repeat('01',32),'hex'), decode(repeat('02',16),'hex'), 4102444800)",
            )
            .await
            .unwrap_err();
        assert_eq!(
            duplicate_error.code(),
            Some(&tokio_postgres::error::SqlState::UNIQUE_VIOLATION)
        );
        client.batch_execute("RESET ROLE").await.unwrap();
        client
            .batch_execute(
                "GRANT UPDATE(request_id) ON federate_session.request_replay TO hs_policy_runtime; \
                 SET ROLE hs_policy_runtime",
            )
            .await
            .unwrap();
        let grants = client
            .query_one(
                "SELECT has_table_privilege(current_user, 'federate_session.request_replay', 'UPDATE'), \
                        has_any_column_privilege(current_user, 'federate_session.request_replay', 'UPDATE')",
                &[],
            )
            .await
            .unwrap();
        assert!(!grants.get::<_, bool>(0));
        assert!(grants.get::<_, bool>(1));
        let allowed: bool = client
            .query_one(POLICY_SESSION_ROLE_SQL, &[])
            .await
            .unwrap()
            .get(0);
        assert!(!allowed, "replay-key column UPDATE must deny startup");
        client.batch_execute("RESET ROLE").await.unwrap();
        client
            .batch_execute(
                "REVOKE UPDATE(request_id) ON federate_session.request_replay FROM hs_policy_runtime; \
                 GRANT SELECT(request_id) ON federate_session.request_replay TO hs_policy_runtime; \
                 SET ROLE hs_policy_runtime",
            )
            .await
            .unwrap();
        let allowed: bool = client
            .query_one(POLICY_SESSION_ROLE_SQL, &[])
            .await
            .unwrap()
            .get(0);
        assert!(!allowed, "request replay column SELECT must deny startup");
        client.batch_execute("RESET ROLE").await.unwrap();
        client
            .batch_execute(
                "REVOKE SELECT(request_id) ON federate_session.request_replay FROM hs_policy_runtime; \
                 GRANT INSERT(host, profile, enabled, authority_generation, collision_inventory_id) \
                   ON federate_session.profile_state TO hs_policy_runtime; \
                 SET ROLE hs_policy_runtime",
            )
            .await
            .unwrap();
        let allowed: bool = client
            .query_one(POLICY_SESSION_ROLE_SQL, &[])
            .await
            .unwrap()
            .get(0);
        assert!(!allowed, "enabled-profile column INSERT must deny startup");
        client.batch_execute("RESET ROLE").await.unwrap();
        client
            .batch_execute(
                "REVOKE INSERT(host, profile, enabled, authority_generation, collision_inventory_id) \
                   ON federate_session.profile_state FROM hs_policy_runtime; \
                 GRANT UPDATE(jti) ON federate_session.replay TO hs_policy_runtime; \
                 SET ROLE hs_policy_runtime",
            )
            .await
            .unwrap();
        let allowed: bool = client
            .query_one(POLICY_SESSION_ROLE_SQL, &[])
            .await
            .unwrap()
            .get(0);
        assert!(!allowed, "source replay column UPDATE must deny startup");
        client.batch_execute("RESET ROLE").await.unwrap();
        client
            .batch_execute("REVOKE UPDATE(jti) ON federate_session.replay FROM hs_policy_runtime")
            .await
            .unwrap();
        client
            .batch_execute(
                "CREATE ROLE hs_policy_runtime_inherited NOLOGIN; \
                 GRANT UPDATE(request_id) ON federate_session.request_replay \
                   TO hs_policy_runtime_inherited; \
                 GRANT hs_policy_runtime_inherited TO hs_policy_runtime; \
                 SET ROLE hs_policy_runtime",
            )
            .await
            .unwrap();
        let allowed: bool = client
            .query_one(POLICY_SESSION_ROLE_SQL, &[])
            .await
            .unwrap()
            .get(0);
        assert!(!allowed, "inherited replay-key UPDATE must deny startup");
        client.batch_execute("RESET ROLE").await.unwrap();
        client
            .batch_execute(
                "REVOKE hs_policy_runtime_inherited FROM hs_policy_runtime; \
                 REVOKE ALL ON federate_session.request_replay FROM hs_policy_runtime_inherited; \
                 DROP ROLE hs_policy_runtime_inherited",
            )
            .await
            .unwrap();
        client
            .batch_execute(
                "GRANT UPDATE(enabled) ON federate_session.profile_state TO hs_policy_runtime; \
                 SET ROLE hs_policy_runtime",
            )
            .await
            .unwrap();
        let allowed: bool = client
            .query_one(POLICY_SESSION_ROLE_SQL, &[])
            .await
            .unwrap()
            .get(0);
        assert!(!allowed, "profile-control authority must deny startup");
        client.batch_execute("RESET ROLE").await.unwrap();
        client
            .batch_execute(
                "DROP SCHEMA federate_session CASCADE; \
                 DROP ROLE hs_policy_runtime, hs_profile_control, hs_session_cleanup, hs_session_migration",
            )
            .await
            .unwrap();
    }
}
