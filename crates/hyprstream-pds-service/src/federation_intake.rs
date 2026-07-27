//! Read-only intake of foreign ATProto identities into an inventory read model.
//!
//! Resolution delegates to the hardened `did:plc` and `did:web` resolvers in
//! `hyprstream-rpc`. This service adds method dispatch, a final subject-binding
//! check, and the projection into a queryable inventory. The projection is
//! deliberately non-authoritative: a federated identity is discoverable but
//! never acquires a local hosted-account tenant.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use async_trait::async_trait;
use hyprstream_rpc::admission::DidDocResolve;
use hyprstream_rpc::auth::mac::{SecurityContext, SecurityLabel};
use hyprstream_rpc::auth::Claims;
use hyprstream_rpc::identity::UNAUTHENTICATED_DID_SENTINEL;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

/// Leak-safe failure from a direct identity-directory read.
///
/// Wire adapters must map this to HTTP 404 (or the transport's equivalent),
/// never 403. Missing objects and objects above the caller's clearance are
/// intentionally indistinguishable.
#[derive(Clone, Copy, Debug, Error, PartialEq, Eq)]
pub enum IdentityDirectoryReadError {
    /// The identity does not exist or is not visible to this caller.
    #[error("identity not found")]
    NotFound,
}

/// Resolver for the two foreign ATProto DID methods accepted by intake.
///
/// Each arm is an existing resolver implementation. In particular, the PLC arm
/// remains read-only and retains its configured egress boundary, audit
/// verification, cache, and response validation.
pub struct FederatedDidResolver {
    plc: Arc<dyn DidDocResolve>,
    web: Arc<dyn DidDocResolve>,
}

impl FederatedDidResolver {
    /// Compose the existing `did:plc` and `did:web` document resolvers.
    pub fn new(plc: Arc<dyn DidDocResolve>, web: Arc<dyn DidDocResolve>) -> Self {
        Self { plc, web }
    }
}

/// Injectable document-resolution boundary for federation intake.
#[async_trait]
pub trait FederatedDidDocumentResolver: Send + Sync {
    /// Resolve a supported foreign DID to its subject-bound document.
    async fn resolve_federated_document(&self, did: &str) -> Result<Value>;
}

#[async_trait]
impl FederatedDidDocumentResolver for FederatedDidResolver {
    async fn resolve_federated_document(&self, did: &str) -> Result<Value> {
        ensure_real_identity_did(did)?;
        let resolver = if hyprstream_rpc::did_plc::is_did_plc(did) {
            &self.plc
        } else if did.starts_with("did:web:") {
            &self.web
        } else {
            bail!("unsupported federated DID method: {did}");
        };
        let document = resolver.resolve_doc(did).await?;
        ensure!(
            document.get("id").and_then(Value::as_str) == Some(did),
            "resolved DID document id does not match requested DID {did}"
        );
        Ok(document)
    }
}

/// Authority class represented by one derived inventory entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InventoryKind {
    /// A DID with an authority-resolved local hosted-account binding.
    Local,
    /// A real foreign DID discovered read-only.
    Federated,
    /// The immutable credential-absence floor, never emitted as a real host.
    Unauthenticated,
}

/// Derived AppView projection. It is an index, never identity authority.
///
/// Live discovery data is intentionally absent. In particular, this type
/// cannot carry a QUIC URL or certificate hash; callers obtain rotating reach
/// through [`FederationIntake::resolve`] at connect time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InventoryEntry {
    did: String,
    handle: Option<String>,
    kind: InventoryKind,
    tenant: Option<String>,
    pds_endpoint: Option<String>,
}

impl InventoryEntry {
    fn federated(did: String, document: &Value) -> Result<Self> {
        Self::indexed_federated(
            did,
            extract_handle(document)?,
            extract_pds_endpoint(document)?,
        )
    }

    /// Build a local projection from an authority-owned hosted-account source.
    ///
    /// This constructor validates the invariant at the read-model boundary:
    /// only a local entry may carry a tenant, and a local entry must carry one.
    pub fn local(
        did: impl Into<String>,
        handle: Option<String>,
        tenant: impl Into<String>,
        pds_endpoint: Option<String>,
    ) -> Result<Self> {
        let entry = Self {
            did: did.into(),
            handle,
            kind: InventoryKind::Local,
            tenant: Some(tenant.into()),
            pds_endpoint,
        };
        entry.validate()?;
        Ok(entry)
    }

    /// Build a foreign projection from already-resolved federation intake.
    pub fn indexed_federated(
        did: impl Into<String>,
        handle: Option<String>,
        pds_endpoint: Option<String>,
    ) -> Result<Self> {
        let entry = Self {
            did: did.into(),
            handle,
            kind: InventoryKind::Federated,
            tenant: None,
            pds_endpoint,
        };
        entry.validate()?;
        Ok(entry)
    }

    /// Revalidate a projection before a derived store accepts it.
    pub fn validate(&self) -> Result<()> {
        ensure_real_identity_did(&self.did)?;
        ensure!(!self.did.is_empty(), "inventory DID must not be empty");
        ensure!(
            self.handle.as_deref().is_none_or(|value| !value.is_empty()),
            "inventory handle must not be empty"
        );
        ensure!(
            self.pds_endpoint
                .as_deref()
                .is_none_or(|value| !value.is_empty()),
            "inventory PDS endpoint must not be empty"
        );
        match self.kind {
            InventoryKind::Local => ensure!(
                self.tenant
                    .as_deref()
                    .is_some_and(|value| !value.is_empty()),
                "local inventory entry requires an authority-resolved tenant"
            ),
            InventoryKind::Federated => ensure!(
                self.tenant.is_none(),
                "only a local inventory entry may carry a tenant"
            ),
            InventoryKind::Unauthenticated => {
                bail!("the unauthenticated floor is not a listable inventory host")
            }
        }
        Ok(())
    }

    /// The indexed DID.
    pub fn did(&self) -> &str {
        &self.did
    }

    /// Optional ATProto handle derived from `alsoKnownAs`.
    pub fn handle(&self) -> Option<&str> {
        self.handle.as_deref()
    }

    /// Local, federated, or unauthenticated classification.
    pub fn kind(&self) -> InventoryKind {
        self.kind
    }

    /// Authority-resolved local tenant, always `None` for federation intake.
    pub fn tenant(&self) -> Option<&str> {
        self.tenant.as_deref()
    }

    /// PDS endpoint pointer. This is not live transport reach.
    pub fn pds_endpoint(&self) -> Option<&str> {
        self.pds_endpoint.as_deref()
    }
}

/// Live connect-time discovery resolved from current DID authority.
///
/// This type is separate from [`InventoryEntry`] so rotating transport
/// credentials cannot become stale AppView data.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedDiscovery {
    quic_url: Option<String>,
    cert_hash: Option<String>,
}

impl ResolvedDiscovery {
    fn from_document(document: &Value) -> Result<Self> {
        let Some(services) = document.get("service").and_then(Value::as_array) else {
            return Ok(Self {
                quic_url: None,
                cert_hash: None,
            });
        };
        let mut quic_entries = services
            .iter()
            .filter(|entry| entry.get("type").and_then(Value::as_str) == Some("QuicTransport"));
        let Some(entry) = quic_entries.next() else {
            return Ok(Self {
                quic_url: None,
                cert_hash: None,
            });
        };
        ensure!(
            quic_entries.next().is_none(),
            "DID document contains ambiguous QuicTransport discovery entries"
        );
        hyprstream_rpc::service_entry::decode_service_entry(entry)?;
        let endpoint = entry
            .get("serviceEndpoint")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("QuicTransport serviceEndpoint is not an object"))?;
        let quic_url = endpoint
            .get("uri")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let cert_hashes = endpoint
            .get("certHashes")
            .and_then(Value::as_array)
            .map(|hashes| {
                hashes
                    .iter()
                    .map(|hash| {
                        hash.as_str()
                            .map(str::to_owned)
                            .ok_or_else(|| anyhow::anyhow!("QUIC cert hash is not a string"))
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        ensure!(
            cert_hashes.len() <= 1,
            "DID document contains ambiguous QUIC certificate hashes"
        );
        Ok(Self {
            quic_url,
            cert_hash: cert_hashes.into_iter().next(),
        })
    }

    /// Current QUIC URL, if the DID document publishes one.
    pub fn quic_url(&self) -> Option<&str> {
        self.quic_url.as_deref()
    }

    /// Current certificate pin, if the DID document publishes one.
    pub fn cert_hash(&self) -> Option<&str> {
        self.cert_hash.as_deref()
    }
}

#[derive(Clone)]
struct StoredInventoryEntry {
    entry: InventoryEntry,
    label: SecurityLabel,
}

/// Non-authoritative inventory read model consumed by federation intake.
///
/// A pglite/Postgres implementation can replace the in-memory implementation
/// without changing resolution or tenant-safety logic.
#[async_trait]
pub trait IdentityInventoryReadModel: Send + Sync {
    /// Insert or refresh one derived projection and its trusted object label.
    ///
    /// The projection must already come from federation intake or an
    /// authority-owned hosted-account source. Implementations must revalidate
    /// tenant/kind invariants and reject [`UNAUTHENTICATED_DID_SENTINEL`].
    async fn upsert_derived(&self, identity: InventoryEntry, label: SecurityLabel) -> Result<()>;

    /// Query only entries dominated by the viewer's verified clearance.
    ///
    /// Filtering happens inside the read model before any response is built.
    /// The return shape intentionally has no total/hidden-count field: callers
    /// can observe only this post-filter vector's length. Implementations must
    /// never return [`UNAUTHENTICATED_DID_SENTINEL`] as a real host.
    async fn query(
        &self,
        viewer: &SecurityContext,
        filter: Option<&str>,
    ) -> Result<Vec<InventoryEntry>> {
        self.query_page(viewer, filter, None, 200).await
    }

    /// Query one deterministic, post-clearance page.
    ///
    /// `after` is an exclusive DID cursor over the stable DID ordering. The
    /// caller must enforce a finite `limit`; implementations apply it after
    /// clearance and filter predicates and never compute a pre-filter total.
    async fn query_page(
        &self,
        viewer: &SecurityContext,
        filter: Option<&str>,
        after: Option<&str>,
        limit: usize,
    ) -> Result<Vec<InventoryEntry>>;

    /// List all entries visible to this viewer.
    async fn list(&self, viewer: &SecurityContext) -> Result<Vec<InventoryEntry>> {
        self.query(viewer, None).await
    }

    /// Fetch one entry without revealing whether a filtered object exists.
    ///
    /// `None` means absent, no caller clearance, or above caller clearance.
    /// Wire adapters must map every `None` case to the same 404-style result.
    async fn get(
        &self,
        viewer: Option<&SecurityContext>,
        did: &str,
    ) -> Result<Option<InventoryEntry>>;
}

/// Thin in-memory inventory read model, useful for embedded/demo deployments.
#[derive(Default)]
pub struct InMemoryIdentityInventory {
    identities: RwLock<BTreeMap<String, StoredInventoryEntry>>,
}

#[async_trait]
impl IdentityInventoryReadModel for InMemoryIdentityInventory {
    async fn upsert_derived(&self, identity: InventoryEntry, label: SecurityLabel) -> Result<()> {
        identity.validate()?;
        let mut identities = self.identities.write();
        if let Some(existing) = identities.get(identity.did()) {
            ensure_safe_refresh(existing, &identity, &label)?;
        }
        identities.insert(
            identity.did.clone(),
            StoredInventoryEntry {
                entry: identity,
                label,
            },
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
        ensure!(limit > 0, "inventory page limit must be positive");
        let identities = self.identities.read();
        Ok(identities
            .values()
            .filter(|stored| {
                stored.entry.did() != UNAUTHENTICATED_DID_SENTINEL
                    && viewer.can_access(&stored.label)
                    && filter.is_none_or(|filter| matches_filter(&stored.entry, filter))
                    && after.is_none_or(|after| stored.entry.did() > after)
            })
            .map(|stored| stored.entry.clone())
            .take(limit)
            .collect())
    }

    async fn get(
        &self,
        viewer: Option<&SecurityContext>,
        did: &str,
    ) -> Result<Option<InventoryEntry>> {
        if did == UNAUTHENTICATED_DID_SENTINEL {
            return Ok(None);
        }
        let identities = self.identities.read();
        Ok(identities
            .get(did)
            .filter(|stored| viewer.is_some_and(|clearance| clearance.can_access(&stored.label)))
            .map(|stored| stored.entry.clone()))
    }
}

fn ensure_safe_refresh(
    existing: &StoredInventoryEntry,
    incoming: &InventoryEntry,
    incoming_label: &SecurityLabel,
) -> Result<()> {
    ensure!(
        existing.entry.kind == incoming.kind,
        "inventory refresh cannot change authority kind for {}",
        incoming.did()
    );
    ensure!(
        existing.entry.tenant == incoming.tenant,
        "inventory refresh cannot change authority-resolved tenant for {}",
        incoming.did()
    );
    ensure!(
        incoming_label.can_access(&existing.label),
        "inventory refresh cannot lower the security label for {}",
        incoming.did()
    );
    Ok(())
}

/// Resolve and project foreign identities without granting local authority.
pub struct FederationIntake {
    resolver: Arc<dyn FederatedDidDocumentResolver>,
    inventory: Arc<dyn IdentityInventoryReadModel>,
    local_issuers: Vec<String>,
}

impl FederationIntake {
    /// Construct an intake service over a resolver and derived inventory sink.
    pub fn new(
        resolver: Arc<dyn FederatedDidDocumentResolver>,
        inventory: Arc<dyn IdentityInventoryReadModel>,
        local_issuers: Vec<String>,
    ) -> Self {
        Self {
            resolver,
            inventory,
            local_issuers,
        }
    }

    /// Resolve and index a foreign DID discovered without identity claims.
    pub async fn intake(&self, did: &str) -> Result<InventoryEntry> {
        self.intake_inner(did).await
    }

    /// Resolve and index a foreign DID carrying already-verified source claims.
    ///
    /// These claims describe the foreign identity, not the local caller
    /// requesting discovery. A local issuer is rejected because it is not a
    /// federated source. For a foreign issuer, `strip_federated_tenant` runs
    /// before projection so even a signed tenant assertion cannot become a
    /// local hosted-account binding.
    pub async fn intake_with_verified_claims(
        &self,
        did: &str,
        mut claims: Claims,
    ) -> Result<InventoryEntry> {
        let local_issuers: Vec<&str> = self.local_issuers.iter().map(String::as_str).collect();
        ensure!(
            !claims.is_local_to(&local_issuers),
            "federation intake requires claims from a foreign issuer"
        );
        claims.strip_federated_tenant(&local_issuers);
        ensure!(
            claims.tenant.is_none(),
            "federated identity retained a local tenant after sanitization"
        );
        self.intake_inner(did).await
    }

    /// Resolve current connect-time discovery for a visible inventory entry.
    ///
    /// The inventory lookup happens before resolver I/O. Missing, no-clearance,
    /// and above-clearance identities all return the same 404-style error.
    pub async fn resolve(
        &self,
        did: &str,
        viewer: Option<&SecurityContext>,
    ) -> Result<ResolvedDiscovery> {
        if self.inventory.get(viewer, did).await?.is_none() {
            return Err(IdentityDirectoryReadError::NotFound.into());
        }
        let document = self.resolver.resolve_federated_document(did).await?;
        ResolvedDiscovery::from_document(&document)
    }

    async fn intake_inner(&self, did: &str) -> Result<InventoryEntry> {
        ensure_real_identity_did(did)?;
        let document = self.resolver.resolve_federated_document(did).await?;
        let identity = InventoryEntry::federated(did.to_owned(), &document)?;
        self.inventory
            .upsert_derived(identity.clone(), SecurityLabel::bottom())
            .await?;
        Ok(identity)
    }
}

fn matches_filter(entry: &InventoryEntry, filter: &str) -> bool {
    let filter = filter.trim().to_lowercase();
    filter.is_empty()
        || entry.did.to_lowercase().contains(&filter)
        || entry
            .handle
            .as_deref()
            .is_some_and(|value| value.to_lowercase().contains(&filter))
        || entry
            .tenant
            .as_deref()
            .is_some_and(|value| value.to_lowercase().contains(&filter))
        || entry
            .pds_endpoint
            .as_deref()
            .is_some_and(|value| value.to_lowercase().contains(&filter))
        || match entry.kind {
            InventoryKind::Local => "local",
            InventoryKind::Federated => "federated",
            InventoryKind::Unauthenticated => "unauthenticated",
        }
        .contains(&filter)
}

fn extract_handle(document: &Value) -> Result<Option<String>> {
    let Some(aliases) = document.get("alsoKnownAs").and_then(Value::as_array) else {
        return Ok(None);
    };
    let handles = aliases
        .iter()
        .filter_map(Value::as_str)
        .filter_map(|alias| alias.strip_prefix("at://"))
        .filter(|handle| !handle.is_empty() && !handle.contains('/'))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    ensure!(
        handles.len() <= 1,
        "DID document contains ambiguous ATProto handles"
    );
    Ok(handles.into_iter().next())
}

fn extract_pds_endpoint(document: &Value) -> Result<Option<String>> {
    let Some(services) = document.get("service").and_then(Value::as_array) else {
        return Ok(None);
    };
    let endpoints = services
        .iter()
        .filter(|entry| {
            entry.get("type").and_then(Value::as_str) == Some("AtprotoPersonalDataServer")
        })
        .map(|entry| {
            entry
                .get("serviceEndpoint")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| anyhow::anyhow!("ATProto PDS serviceEndpoint is not a string"))
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        endpoints.len() <= 1,
        "DID document contains ambiguous ATProto PDS endpoints"
    );
    Ok(endpoints.into_iter().next())
}

fn ensure_real_identity_did(did: &str) -> Result<()> {
    ensure!(
        did != UNAUTHENTICATED_DID_SENTINEL,
        "{UNAUTHENTICATED_DID_SENTINEL} is reserved for the unauthenticated floor and is not a federated identity"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use hyprstream_rpc::auth::mac::{
        CompartmentSet, Level, MacDecision, SecurityContext, VerifiedKeyMaterial,
    };
    use hyprstream_rpc::Subject;
    use hyprstream_vfs::{SyntheticMount, SyntheticNode};

    use crate::{AccountRecordReadAuthorizer, AccountRecordStore, OAUTH_ACCOUNT_RESOLVER_SUBJECT};

    const FEDERATED_DID: &str = "did:plc:ewvi7nxzyoun6zhxrhs64oiz";
    const FEDERATED_WEB_DID: &str = "did:web:alice.example";

    struct FixtureResolver {
        expected_did: &'static str,
        document: Value,
    }

    #[async_trait]
    impl DidDocResolve for FixtureResolver {
        async fn resolve_doc(&self, did: &str) -> Result<Value> {
            ensure!(did == self.expected_did, "unexpected DID {did}");
            Ok(self.document.clone())
        }
    }

    struct NeverResolver;

    #[async_trait]
    impl DidDocResolve for NeverResolver {
        async fn resolve_doc(&self, did: &str) -> Result<Value> {
            bail!("wrong resolver arm selected for {did}")
        }
    }

    struct PermitAccountReads;

    impl AccountRecordReadAuthorizer for PermitAccountReads {
        fn check_read(
            &self,
            _subject: &Subject,
            _verified_tenant: Option<&str>,
            _security_context: Option<&SecurityContext>,
            _object_id: &str,
        ) -> MacDecision {
            MacDecision::Permit
        }
    }

    fn unauthenticated_viewer() -> SecurityContext {
        SecurityContext::new(
            Level::Public,
            CompartmentSet::EMPTY,
            VerifiedKeyMaterial::Unverified,
        )
    }

    #[tokio::test]
    async fn federated_did_resolves_lists_and_has_no_hosted_tenant() {
        let resolver = Arc::new(FederatedDidResolver::new(
            Arc::new(FixtureResolver {
                expected_did: FEDERATED_DID,
                document: serde_json::json!({
                    "id": FEDERATED_DID,
                    "alsoKnownAs": ["at://alice.example"],
                    "service": [{
                        "id": format!("{FEDERATED_DID}#atproto_pds"),
                        "type": "AtprotoPersonalDataServer",
                        "serviceEndpoint": "https://pds.alice.example"
                    }]
                }),
            }),
            Arc::new(NeverResolver),
        ));
        let inventory = Arc::new(InMemoryIdentityInventory::default());
        let intake = FederationIntake::new(
            resolver,
            inventory.clone(),
            vec!["https://local.example".to_owned()],
        );
        let foreign_claims = Claims::new(FEDERATED_DID.to_owned(), 1, 2)
            .with_issuer("https://foreign.example".to_owned())
            .with_tenant("acme".to_owned());

        let resolved = intake
            .intake_with_verified_claims(FEDERATED_DID, foreign_claims)
            .await
            .unwrap();
        assert_eq!(resolved.did(), FEDERATED_DID);
        assert_eq!(resolved.handle(), Some("alice.example"));
        assert_eq!(resolved.kind(), InventoryKind::Federated);
        assert_eq!(resolved.tenant(), None);
        assert_eq!(resolved.pds_endpoint(), Some("https://pds.alice.example"));

        let listed = inventory.list(&unauthenticated_viewer()).await.unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].did(), FEDERATED_DID);
        assert_eq!(listed[0].tenant(), None);
        let projected = serde_json::to_value(&listed[0]).unwrap();
        assert!(projected.get("quicUrl").is_none());
        assert!(projected.get("certHash").is_none());
        assert!(projected.get("certHashes").is_none());
        assert!(projected.get("document").is_none());

        let account_store = AccountRecordStore::new(
            Arc::new(SyntheticMount::new(SyntheticNode::dir())),
            Arc::new(PermitAccountReads),
        );
        let tenant = account_store
            .resolve_tenant_for_hosted_did(
                &Subject::new(OAUTH_ACCOUNT_RESOLVER_SUBJECT),
                FEDERATED_DID,
            )
            .await
            .unwrap();
        assert_eq!(tenant, None);
    }

    #[tokio::test]
    async fn method_dispatch_rejects_document_subject_mismatch() {
        let resolver = FederatedDidResolver::new(
            Arc::new(FixtureResolver {
                expected_did: FEDERATED_DID,
                document: serde_json::json!({ "id": "did:plc:z23456723456723456723456" }),
            }),
            Arc::new(NeverResolver),
        );

        let error = resolver
            .resolve_federated_document(FEDERATED_DID)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("does not match"));
    }

    #[tokio::test]
    async fn did_web_arm_resolves_and_lists_without_a_tenant() {
        let resolver = Arc::new(FederatedDidResolver::new(
            Arc::new(NeverResolver),
            Arc::new(FixtureResolver {
                expected_did: FEDERATED_WEB_DID,
                document: serde_json::json!({
                    "id": FEDERATED_WEB_DID,
                    "service": [{
                        "id": format!("{FEDERATED_WEB_DID}#quic"),
                        "type": "QuicTransport",
                        "serviceEndpoint": {
                            "uri": "https://alice.example:443",
                            "webpki": true,
                            "certHashes": []
                        }
                    }]
                }),
            }),
        ));
        let inventory = Arc::new(InMemoryIdentityInventory::default());
        let intake = FederationIntake::new(resolver, inventory.clone(), Vec::new());

        let resolved = intake.intake(FEDERATED_WEB_DID).await.unwrap();
        assert_eq!(resolved.did(), FEDERATED_WEB_DID);
        assert_eq!(resolved.tenant(), None);
        assert_eq!(
            inventory.list(&unauthenticated_viewer()).await.unwrap(),
            vec![resolved]
        );
        let viewer = unauthenticated_viewer();
        let discovery = intake
            .resolve(FEDERATED_WEB_DID, Some(&viewer))
            .await
            .unwrap();
        assert_eq!(discovery.quic_url(), Some("https://alice.example:443"));
        assert_eq!(discovery.cert_hash(), None);
        let projected = serde_json::to_value(discovery).unwrap();
        assert!(projected.get("quicUrl").is_some());
        assert!(projected.get("certHash").is_some());
        assert!(projected.get("certHashes").is_none());
    }

    #[tokio::test]
    async fn inventory_listing_prefilters_above_clearance_without_a_hidden_count() {
        let inventory = InMemoryIdentityInventory::default();
        let public =
            InventoryEntry::indexed_federated("did:web:public.example", None, None).unwrap();
        inventory
            .upsert_derived(public.clone(), SecurityLabel::bottom())
            .await
            .unwrap();
        inventory
            .upsert_derived(
                InventoryEntry::indexed_federated("did:web:hidden.example", None, None).unwrap(),
                SecurityLabel::new(
                    Level::Secret,
                    hyprstream_rpc::auth::mac::Assurance::Unverified,
                    CompartmentSet::EMPTY,
                ),
            )
            .await
            .unwrap();

        let visible = inventory.list(&unauthenticated_viewer()).await.unwrap();
        assert_eq!(visible, vec![public]);
        assert_eq!(visible.len(), 1, "only the post-filter count is observable");
    }

    #[tokio::test]
    async fn above_floor_direct_access_is_not_found_not_forbidden() {
        const HIDDEN_DID: &str = "did:web:hidden.example";
        const MISSING_DID: &str = "did:web:missing.example";

        let inventory = Arc::new(InMemoryIdentityInventory::default());
        inventory.identities.write().insert(
            HIDDEN_DID.to_owned(),
            StoredInventoryEntry {
                entry: InventoryEntry {
                    did: HIDDEN_DID.to_owned(),
                    handle: None,
                    kind: InventoryKind::Federated,
                    tenant: None,
                    pds_endpoint: None,
                },
                label: SecurityLabel::new(
                    Level::Secret,
                    hyprstream_rpc::auth::mac::Assurance::Unverified,
                    CompartmentSet::EMPTY,
                ),
            },
        );
        let intake = FederationIntake::new(
            Arc::new(FederatedDidResolver::new(
                Arc::new(NeverResolver),
                Arc::new(NeverResolver),
            )),
            inventory.clone(),
            Vec::new(),
        );
        let viewer = unauthenticated_viewer();

        assert_eq!(
            inventory.get(Some(&viewer), HIDDEN_DID).await.unwrap(),
            None
        );
        assert_eq!(
            inventory.get(Some(&viewer), MISSING_DID).await.unwrap(),
            None
        );
        assert_eq!(inventory.get(None, HIDDEN_DID).await.unwrap(), None);

        for did in [HIDDEN_DID, MISSING_DID] {
            let error = intake.resolve(did, Some(&viewer)).await.unwrap_err();
            assert_eq!(
                error.downcast_ref::<IdentityDirectoryReadError>(),
                Some(&IdentityDirectoryReadError::NotFound),
                "direct access to {did} must be 404-style, never forbidden: {error:#}"
            );
        }
    }

    #[tokio::test]
    async fn federated_intake_cannot_demote_or_declassify_a_local_projection() {
        let inventory = Arc::new(InMemoryIdentityInventory::default());
        let local = InventoryEntry::local(
            FEDERATED_WEB_DID,
            Some("local.example".to_owned()),
            "tenant-authority",
            Some("https://local-pds.example".to_owned()),
        )
        .unwrap();
        inventory
            .upsert_derived(
                local.clone(),
                SecurityLabel::new(
                    Level::Secret,
                    hyprstream_rpc::auth::mac::Assurance::PqHybrid,
                    CompartmentSet::single(3),
                ),
            )
            .await
            .unwrap();
        let intake = FederationIntake::new(
            Arc::new(FederatedDidResolver::new(
                Arc::new(NeverResolver),
                Arc::new(FixtureResolver {
                    expected_did: FEDERATED_WEB_DID,
                    document: serde_json::json!({
                        "id": FEDERATED_WEB_DID,
                        "alsoKnownAs": ["at://foreign.example"],
                    }),
                }),
            )),
            inventory.clone(),
            Vec::new(),
        );

        let error = intake.intake(FEDERATED_WEB_DID).await.unwrap_err();
        assert!(error.to_string().contains("cannot change authority kind"));
        assert!(
            inventory
                .list(&unauthenticated_viewer())
                .await
                .unwrap()
                .is_empty(),
            "rejected federation refresh must not declassify the local row"
        );
        let privileged = SecurityContext::new(
            Level::Secret,
            CompartmentSet::single(3),
            VerifiedKeyMaterial::PqHybrid,
        );
        assert_eq!(
            inventory
                .get(Some(&privileged), FEDERATED_WEB_DID)
                .await
                .unwrap(),
            Some(local)
        );
    }

    #[tokio::test]
    async fn unauthenticated_sentinel_is_never_resolved_or_listed() {
        let resolver = Arc::new(FederatedDidResolver::new(
            Arc::new(NeverResolver),
            Arc::new(NeverResolver),
        ));
        let inventory = Arc::new(InMemoryIdentityInventory::default());
        let intake = FederationIntake::new(resolver, inventory.clone(), Vec::new());

        let error = intake
            .intake(UNAUTHENTICATED_DID_SENTINEL)
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("unauthenticated floor"),
            "{error:#}"
        );
        assert!(inventory
            .list(&unauthenticated_viewer())
            .await
            .unwrap()
            .is_empty());
        let error = intake
            .resolve(
                UNAUTHENTICATED_DID_SENTINEL,
                Some(&unauthenticated_viewer()),
            )
            .await
            .unwrap_err();
        assert_eq!(
            error.downcast_ref::<IdentityDirectoryReadError>(),
            Some(&IdentityDirectoryReadError::NotFound)
        );

        inventory.identities.write().insert(
            UNAUTHENTICATED_DID_SENTINEL.to_owned(),
            StoredInventoryEntry {
                entry: InventoryEntry {
                    did: UNAUTHENTICATED_DID_SENTINEL.to_owned(),
                    handle: None,
                    kind: InventoryKind::Unauthenticated,
                    tenant: None,
                    pds_endpoint: None,
                },
                label: SecurityLabel::bottom(),
            },
        );
        assert!(
            inventory
                .list(&unauthenticated_viewer())
                .await
                .unwrap()
                .is_empty(),
            "legacy/corrupt sentinel rows must be filtered from inventory listing"
        );
    }

    #[tokio::test]
    async fn local_identity_claims_cannot_enter_federation_intake() {
        let resolver = Arc::new(FederatedDidResolver::new(
            Arc::new(NeverResolver),
            Arc::new(NeverResolver),
        ));
        let inventory = Arc::new(InMemoryIdentityInventory::default());
        let intake = FederationIntake::new(
            resolver,
            inventory.clone(),
            vec!["https://local.example".to_owned()],
        );
        let local_claims = Claims::new("alice".to_owned(), 1, 2)
            .with_issuer("https://local.example".to_owned())
            .with_tenant("acme".to_owned());

        let error = intake
            .intake_with_verified_claims(FEDERATED_DID, local_claims)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("foreign issuer"));
        assert!(inventory
            .list(&unauthenticated_viewer())
            .await
            .unwrap()
            .is_empty());
    }
}
