use std::sync::Arc;

#[cfg(feature = "pglite")]
use std::path::Path;

#[cfg(feature = "pglite")]
use anyhow::{bail, Context};
use anyhow::{ensure, Result};
use async_trait::async_trait;
use hyprstream_pds_service::federation_intake::{
    IdentityInventoryReadModel, InventoryEntry, InventoryKind,
};
use hyprstream_rpc::auth::mac::SecurityLabel;
#[cfg(feature = "pglite")]
use hyprstream_rpc::auth::mac::{Assurance, Level, SecurityContext};
#[cfg(feature = "pglite")]
use hyprstream_rpc::identity::UNAUTHENTICATED_DID_SENTINEL;
#[cfg(feature = "pglite")]
use pglite::{PGlite, Row};

#[cfg(feature = "pglite")]
const CREATE_SCHEMA_TEMPLATE: &str = r#"
CREATE TABLE IF NOT EXISTS appview_inventory (
    did                 TEXT PRIMARY KEY,
    handle              TEXT,
    kind                TEXT NOT NULL,
    tenant              TEXT,
    pds_endpoint        TEXT,
    label_level         SMALLINT NOT NULL,
    label_assurance     SMALLINT NOT NULL,
    label_compartments  BIGINT NOT NULL,
    CHECK (did <> '__UNAUTHENTICATED_DID_SENTINEL__'),
    CHECK (kind IN ('local', 'federated')),
    CHECK (
        (kind = 'local' AND tenant IS NOT NULL AND tenant <> '')
        OR
        (kind = 'federated' AND tenant IS NULL)
    )
);
CREATE INDEX IF NOT EXISTS appview_inventory_handle_idx
    ON appview_inventory (lower(handle));
"#;

#[cfg(feature = "pglite")]
const UPSERT: &str = r#"
INSERT INTO appview_inventory (
    did, handle, kind, tenant, pds_endpoint,
    label_level, label_assurance, label_compartments
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
ON CONFLICT (did) DO UPDATE SET
    handle = EXCLUDED.handle,
    kind = EXCLUDED.kind,
    tenant = EXCLUDED.tenant,
    pds_endpoint = EXCLUDED.pds_endpoint,
    label_level = EXCLUDED.label_level,
    label_assurance = EXCLUDED.label_assurance,
    label_compartments = EXCLUDED.label_compartments
WHERE appview_inventory.kind = EXCLUDED.kind
  AND appview_inventory.tenant IS NOT DISTINCT FROM EXCLUDED.tenant
  AND EXCLUDED.label_level >= appview_inventory.label_level
  AND EXCLUDED.label_assurance >= appview_inventory.label_assurance
  AND (
      appview_inventory.label_compartments & EXCLUDED.label_compartments
  ) = appview_inventory.label_compartments
RETURNING did
"#;

#[cfg(feature = "pglite")]
const VISIBLE_QUERY: &str = r#"
SELECT did, handle, kind, tenant, pds_endpoint
FROM appview_inventory
WHERE did <> $1
  AND label_level <= $2
  AND label_assurance <= $3
  AND (label_compartments & $4) = label_compartments
  AND (
      $5::text IS NULL
      OR strpos(lower(did), lower($5)) > 0
      OR strpos(lower(COALESCE(handle, '')), lower($5)) > 0
      OR strpos(lower(kind), lower($5)) > 0
      OR strpos(lower(COALESCE(tenant, '')), lower($5)) > 0
      OR strpos(lower(COALESCE(pds_endpoint, '')), lower($5)) > 0
  )
  AND ($6::text IS NULL OR did > $6)
ORDER BY did
LIMIT $7
"#;

#[cfg(feature = "pglite")]
const VISIBLE_GET: &str = r#"
SELECT did, handle, kind, tenant, pds_endpoint
FROM appview_inventory
WHERE did = $1
  AND did <> $2
  AND label_level <= $3
  AND label_assurance <= $4
  AND (label_compartments & $5) = label_compartments
"#;

/// PGlite-backed derived identity inventory.
///
/// The schema and queries use PostgreSQL types/operators. A metal deployment
/// can implement [`IdentityInventoryReadModel`] against its Postgres pool
/// without changing the projection or HTTP contracts.
#[cfg(feature = "pglite")]
#[derive(Clone)]
pub struct PGliteIdentityInventory {
    database: Arc<PGlite>,
}

#[cfg(feature = "pglite")]
impl PGliteIdentityInventory {
    /// Open the embedded Postgres data directory and apply the idempotent schema.
    pub async fn open(data_dir: impl AsRef<Path>) -> Result<Self> {
        let database = Arc::new(
            PGlite::open(data_dir)
                .await
                .context("opening AppView PGlite database")?,
        );
        Self::from_database(database).await
    }

    /// Apply the inventory schema to an already-open shared PGlite handle.
    pub async fn from_database(database: Arc<PGlite>) -> Result<Self> {
        let schema = CREATE_SCHEMA_TEMPLATE.replace(
            "__UNAUTHENTICATED_DID_SENTINEL__",
            UNAUTHENTICATED_DID_SENTINEL,
        );
        database
            .exec(&schema)
            .await
            .context("creating AppView inventory schema")?;
        Ok(Self { database })
    }

    /// Return the shared embedded Postgres handle for sibling repositories.
    pub fn database(&self) -> Arc<PGlite> {
        Arc::clone(&self.database)
    }

    async fn visible_rows(
        &self,
        viewer: &SecurityContext,
        filter: Option<&str>,
        after: Option<&str>,
        limit: usize,
    ) -> Result<Vec<InventoryEntry>> {
        ensure!(limit > 0, "inventory page limit must be positive");
        let limit = i64::try_from(limit).context("inventory page limit exceeds BIGINT")?;
        let clearance = viewer.clearance();
        let level = level_code(clearance.level);
        let assurance = assurance_code(clearance.assurance);
        let compartments = clearance.compartments.0 as i64;
        let normalized_filter = filter.map(str::trim).filter(|value| !value.is_empty());
        let rows = self
            .database
            .query(
                VISIBLE_QUERY,
                &[
                    &UNAUTHENTICATED_DID_SENTINEL,
                    &level,
                    &assurance,
                    &compartments,
                    &normalized_filter,
                    &after,
                    &limit,
                ],
            )
            .await
            .context("querying visible AppView inventory")?;
        rows.iter().map(decode_entry).collect()
    }
}

#[cfg(feature = "pglite")]
#[async_trait]
impl IdentityInventoryReadModel for PGliteIdentityInventory {
    async fn upsert_derived(&self, identity: InventoryEntry, label: SecurityLabel) -> Result<()> {
        identity.validate()?;
        ensure!(
            identity.kind() != InventoryKind::Unauthenticated,
            "the unauthenticated floor is not a listable inventory host"
        );

        let kind = kind_name(identity.kind());
        let did = identity.did();
        let handle = identity.handle();
        let tenant = identity.tenant();
        let pds_endpoint = identity.pds_endpoint();
        let level = level_code(label.level);
        let assurance = assurance_code(label.assurance);
        let compartments = label.compartments.0 as i64;
        let rows = self
            .database
            .query(
                UPSERT,
                &[
                    &did,
                    &handle,
                    &kind,
                    &tenant,
                    &pds_endpoint,
                    &level,
                    &assurance,
                    &compartments,
                ],
            )
            .await
            .context("upserting derived AppView inventory entry")?;
        ensure!(
            rows.len() == 1,
            "inventory refresh rejected a kind, tenant, or label write-down for {}",
            identity.did()
        );
        Ok(())
    }

    async fn query_page(
        &self,
        viewer: &SecurityContext,
        filter: Option<&str>,
        after: Option<&str>,
        limit: usize,
    ) -> Result<Vec<InventoryEntry>> {
        self.visible_rows(viewer, filter, after, limit).await
    }

    async fn get(
        &self,
        viewer: Option<&SecurityContext>,
        did: &str,
    ) -> Result<Option<InventoryEntry>> {
        let Some(viewer) = viewer else {
            return Ok(None);
        };
        if did == UNAUTHENTICATED_DID_SENTINEL {
            return Ok(None);
        }
        let clearance = viewer.clearance();
        let level = level_code(clearance.level);
        let assurance = assurance_code(clearance.assurance);
        let compartments = clearance.compartments.0 as i64;
        let rows = self
            .database
            .query(
                VISIBLE_GET,
                &[
                    &did,
                    &UNAUTHENTICATED_DID_SENTINEL,
                    &level,
                    &assurance,
                    &compartments,
                ],
            )
            .await
            .context("fetching visible AppView inventory entry")?;
        ensure!(
            rows.len() <= 1,
            "inventory DID primary key returned duplicates"
        );
        rows.first().map(decode_entry).transpose()
    }
}

#[cfg(feature = "pglite")]
fn decode_entry(row: &Row) -> Result<InventoryEntry> {
    let did: String = row.get(0).context("decoding inventory DID")?;
    let handle: Option<String> = row.get(1).context("decoding inventory handle")?;
    let kind: String = row.get(2).context("decoding inventory kind")?;
    let tenant: Option<String> = row.get(3).context("decoding inventory tenant")?;
    let pds_endpoint: Option<String> = row.get(4).context("decoding inventory PDS endpoint")?;
    match kind.as_str() {
        "local" => InventoryEntry::local(
            did,
            handle,
            tenant.context("stored local inventory entry has no tenant")?,
            pds_endpoint,
        ),
        "federated" => {
            ensure!(
                tenant.is_none(),
                "stored federated inventory entry carries a tenant"
            );
            InventoryEntry::indexed_federated(did, handle, pds_endpoint)
        }
        _ => bail!("stored inventory entry has invalid kind {kind:?}"),
    }
}

#[cfg(feature = "pglite")]
const fn level_code(level: Level) -> i16 {
    level as i16
}

#[cfg(feature = "pglite")]
const fn assurance_code(assurance: Assurance) -> i16 {
    assurance as i16
}

#[cfg(feature = "pglite")]
const fn kind_name(kind: InventoryKind) -> &'static str {
    match kind {
        InventoryKind::Local => "local",
        InventoryKind::Federated => "federated",
        InventoryKind::Unauthenticated => "unauthenticated",
    }
}

/// One projection emitted by an authority-owned source.
///
/// Construction routes through [`InventoryEntry`] invariant checks. The label
/// is trusted object metadata supplied by the source, not by a client or SQL.
#[derive(Clone, Debug)]
pub struct LabeledInventoryEntry {
    entry: InventoryEntry,
    label: SecurityLabel,
}

impl LabeledInventoryEntry {
    /// Create a projection from the directory/federation authority boundary.
    pub fn federated(
        did: impl Into<String>,
        handle: Option<String>,
        pds_endpoint: Option<String>,
        label: SecurityLabel,
    ) -> Result<Self> {
        Ok(Self {
            entry: InventoryEntry::indexed_federated(did, handle, pds_endpoint)?,
            label,
        })
    }

    /// Create a local projection from a hosted-account authority boundary.
    pub fn local(
        did: impl Into<String>,
        handle: Option<String>,
        tenant: impl Into<String>,
        pds_endpoint: Option<String>,
        label: SecurityLabel,
    ) -> Result<Self> {
        Ok(Self {
            entry: InventoryEntry::local(did, handle, tenant, pds_endpoint)?,
            label,
        })
    }
}

/// Bulk/bootstrap source for directory-derived foreign projections.
///
/// Live federation intake writes directly through
/// [`IdentityInventoryReadModel::upsert_derived`]; this source is the clean seam
/// for the directory lane's later snapshot/bootstrap adapter.
#[async_trait]
pub trait DirectoryInventorySource: Send + Sync {
    async fn entries(&self) -> Result<Vec<LabeledInventoryEntry>>;
}

/// Authority-owned source for local hosted-account projections.
///
/// Implementations must derive tenant from the hosted-account binding. A
/// client payload or database column is never an acceptable source.
#[async_trait]
pub trait HostedAccountInventorySource: Send + Sync {
    async fn entries(&self) -> Result<Vec<LabeledInventoryEntry>>;
}

/// Explicit empty directory adapter used until the directory lane lands.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubDirectoryInventorySource;

#[async_trait]
impl DirectoryInventorySource for StubDirectoryInventorySource {
    async fn entries(&self) -> Result<Vec<LabeledInventoryEntry>> {
        Ok(Vec::new())
    }
}

/// Explicit empty hosted-account enumeration adapter.
///
/// The merged hosted-PDS mint lane has no global, authority-safe listing API,
/// so the AppView does not infer tenant from its registration response.
#[derive(Clone, Copy, Debug, Default)]
pub struct StubHostedAccountInventorySource;

#[async_trait]
impl HostedAccountInventorySource for StubHostedAccountInventorySource {
    async fn entries(&self) -> Result<Vec<LabeledInventoryEntry>> {
        Ok(Vec::new())
    }
}

/// Refresh orchestration for separately landing directory/hosted adapters.
pub struct InventoryIngestor {
    inventory: Arc<dyn IdentityInventoryReadModel>,
}

impl InventoryIngestor {
    pub fn new(inventory: Arc<dyn IdentityInventoryReadModel>) -> Self {
        Self { inventory }
    }

    /// Ingest derived snapshots without exposing source totals to query callers.
    pub async fn refresh(
        &self,
        directory: &dyn DirectoryInventorySource,
        hosted_accounts: &dyn HostedAccountInventorySource,
    ) -> Result<()> {
        for projection in directory.entries().await? {
            ensure!(
                projection.entry.kind() == InventoryKind::Federated,
                "directory source emitted a non-federated projection"
            );
            self.inventory
                .upsert_derived(projection.entry, projection.label)
                .await?;
        }
        for projection in hosted_accounts.entries().await? {
            ensure!(
                projection.entry.kind() == InventoryKind::Local,
                "hosted-account source emitted a non-local projection"
            );
            self.inventory
                .upsert_derived(projection.entry, projection.label)
                .await?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use hyprstream_pds_service::federation_intake::InMemoryIdentityInventory;
    #[cfg(feature = "pglite")]
    use hyprstream_rpc::auth::mac::Assurance;
    use hyprstream_rpc::auth::mac::{CompartmentSet, Level, SecurityContext, VerifiedKeyMaterial};

    fn viewer(
        level: Level,
        assurance: VerifiedKeyMaterial,
        compartments: CompartmentSet,
    ) -> SecurityContext {
        SecurityContext::new(level, compartments, assurance)
    }

    #[cfg(feature = "pglite")]
    #[tokio::test]
    async fn pglite_prefilters_before_materialization_and_preserves_contract() {
        let temp = tempfile::tempdir_in(std::env::current_dir().unwrap()).unwrap();
        let inventory = PGliteIdentityInventory::open(temp.path()).await.unwrap();
        let public = InventoryEntry::indexed_federated(
            "did:web:public.example",
            Some("alice.example".to_owned()),
            Some("https://pds.public.example".to_owned()),
        )
        .unwrap();
        let secret = InventoryEntry::indexed_federated(
            "did:web:secret.example",
            Some("secret.example".to_owned()),
            None,
        )
        .unwrap();
        let local = InventoryEntry::local(
            "did:web:bob.accounts.example",
            Some("bob".to_owned()),
            "tenant-authority",
            Some("https://pds.local.example".to_owned()),
        )
        .unwrap();
        inventory
            .upsert_derived(public.clone(), SecurityLabel::bottom())
            .await
            .unwrap();
        inventory
            .upsert_derived(
                secret.clone(),
                SecurityLabel::new(
                    Level::Secret,
                    Assurance::PqHybrid,
                    CompartmentSet::single(7),
                ),
            )
            .await
            .unwrap();
        inventory
            .upsert_derived(
                local.clone(),
                SecurityLabel::new(
                    Level::Internal,
                    Assurance::Classical,
                    CompartmentSet::single(4),
                ),
            )
            .await
            .unwrap();
        let demotion = inventory
            .upsert_derived(
                InventoryEntry::indexed_federated(
                    local.did(),
                    Some("attacker-refresh.example".to_owned()),
                    None,
                )
                .unwrap(),
                SecurityLabel::bottom(),
            )
            .await
            .unwrap_err();
        assert!(demotion.to_string().contains("write-down"));
        let label_write_down = inventory
            .upsert_derived(secret.clone(), SecurityLabel::bottom())
            .await
            .unwrap_err();
        assert!(label_write_down.to_string().contains("write-down"));

        let floor = viewer(
            Level::Public,
            VerifiedKeyMaterial::Unverified,
            CompartmentSet::EMPTY,
        );
        let floor_visible = inventory.query(&floor, None).await.unwrap();
        assert_eq!(floor_visible, vec![public.clone()]);
        assert_eq!(
            inventory.query(&floor, Some("secret")).await.unwrap(),
            Vec::<InventoryEntry>::new()
        );
        assert_eq!(
            inventory.query(&floor, Some("ALICE")).await.unwrap(),
            vec![public.clone()]
        );
        assert!(
            inventory.query(&floor, Some("%")).await.unwrap().is_empty(),
            "SQL filters must treat wildcard characters as literal substrings"
        );
        assert_eq!(
            inventory.get(Some(&floor), secret.did()).await.unwrap(),
            None
        );
        assert_eq!(
            inventory
                .get(Some(&floor), "did:web:missing.example")
                .await
                .unwrap(),
            None
        );

        let privileged = viewer(
            Level::Secret,
            VerifiedKeyMaterial::PqHybrid,
            CompartmentSet::single(4).union(CompartmentSet::single(7)),
        );
        assert_eq!(
            inventory.query(&privileged, None).await.unwrap(),
            vec![local.clone(), public.clone(), secret.clone()]
        );
        assert_eq!(
            inventory
                .query_page(&privileged, None, None, 2)
                .await
                .unwrap(),
            vec![local, public.clone()]
        );
        assert_eq!(
            inventory
                .query_page(&privileged, None, Some(public.did()), 2)
                .await
                .unwrap(),
            vec![secret]
        );

        let wire = serde_json::to_value(&floor_visible).unwrap();
        let row = &wire.as_array().unwrap()[0];
        assert!(row.get("quicUrl").is_none());
        assert!(row.get("certHash").is_none());
        assert!(wire.get("total").is_none());
        assert!(wire.get("hiddenCount").is_none());

        let rejected = InventoryEntry::indexed_federated(UNAUTHENTICATED_DID_SENTINEL, None, None)
            .unwrap_err();
        assert!(rejected.to_string().contains("unauthenticated floor"));
    }

    #[tokio::test]
    async fn source_stubs_are_explicitly_empty() {
        assert!(StubDirectoryInventorySource
            .entries()
            .await
            .unwrap()
            .is_empty());
        assert!(StubHostedAccountInventorySource
            .entries()
            .await
            .unwrap()
            .is_empty());
    }

    struct StaticDirectory(Vec<LabeledInventoryEntry>);

    #[async_trait]
    impl DirectoryInventorySource for StaticDirectory {
        async fn entries(&self) -> Result<Vec<LabeledInventoryEntry>> {
            Ok(self.0.clone())
        }
    }

    struct StaticHostedAccounts(Vec<LabeledInventoryEntry>);

    #[async_trait]
    impl HostedAccountInventorySource for StaticHostedAccounts {
        async fn entries(&self) -> Result<Vec<LabeledInventoryEntry>> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn source_refresh_preserves_local_tenant_authority_boundary() {
        let inventory = Arc::new(InMemoryIdentityInventory::default());
        let ingestor = InventoryIngestor::new(inventory.clone());
        let directory = StaticDirectory(vec![LabeledInventoryEntry::federated(
            "did:web:foreign.example",
            Some("foreign.example".to_owned()),
            Some("https://pds.foreign.example".to_owned()),
            SecurityLabel::bottom(),
        )
        .unwrap()]);
        let hosted = StaticHostedAccounts(vec![LabeledInventoryEntry::local(
            "did:web:alice.accounts.example",
            Some("alice".to_owned()),
            "tenant-authority",
            Some("https://pds.local.example".to_owned()),
            SecurityLabel::bottom(),
        )
        .unwrap()]);

        ingestor.refresh(&directory, &hosted).await.unwrap();
        let visible = inventory
            .list(&viewer(
                Level::Public,
                VerifiedKeyMaterial::Unverified,
                CompartmentSet::EMPTY,
            ))
            .await
            .unwrap();
        assert_eq!(visible.len(), 2);
        let local = visible
            .iter()
            .find(|entry| entry.kind() == InventoryKind::Local)
            .unwrap();
        assert_eq!(local.tenant(), Some("tenant-authority"));
        let federated = visible
            .iter()
            .find(|entry| entry.kind() == InventoryKind::Federated)
            .unwrap();
        assert_eq!(federated.tenant(), None);
        assert!(
            InventoryEntry::local("did:web:invalid.example", None, "", None).is_err(),
            "a local projection without an authority tenant must fail closed"
        );
    }
}
