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
use hyprstream_rpc::auth::Claims;
use parking_lot::RwLock;
use serde_json::Value;

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

/// A derived, read-only inventory projection for a foreign identity.
///
/// `tenant` is retained as an explicit optional field so API consumers can use
/// the same shape for local and federated inventory entries. Federation intake
/// constructs records only with `None`; there is no public constructor that can
/// attach a tenant.
#[derive(Clone, Debug, PartialEq)]
pub struct FederatedIdentityRecord {
    did: String,
    document: Value,
    tenant: Option<String>,
}

impl FederatedIdentityRecord {
    fn new(did: String, document: Value) -> Self {
        Self {
            did,
            document,
            tenant: None,
        }
    }

    /// The resolved foreign DID.
    pub fn did(&self) -> &str {
        &self.did
    }

    /// The validated DID document used to derive this projection.
    pub fn document(&self) -> &Value {
        &self.document
    }

    /// Always `None` for records produced by federation intake.
    pub fn tenant(&self) -> Option<&str> {
        self.tenant.as_deref()
    }
}

/// Non-authoritative inventory sink consumed by federation intake.
///
/// A pglite/Postgres implementation can replace the in-memory implementation
/// without changing resolution or tenant-safety logic.
pub trait FederatedIdentityInventory: Send + Sync {
    /// Insert or refresh one derived foreign-identity projection.
    fn upsert(&self, identity: FederatedIdentityRecord) -> Result<()>;

    /// List the currently indexed foreign identities in deterministic DID order.
    fn list(&self) -> Result<Vec<FederatedIdentityRecord>>;
}

/// Thin in-memory inventory read model, useful for embedded/demo deployments.
#[derive(Default)]
pub struct InMemoryFederatedIdentityInventory {
    identities: RwLock<BTreeMap<String, FederatedIdentityRecord>>,
}

impl FederatedIdentityInventory for InMemoryFederatedIdentityInventory {
    fn upsert(&self, identity: FederatedIdentityRecord) -> Result<()> {
        let mut identities = self.identities.write();
        identities.insert(identity.did.clone(), identity);
        Ok(())
    }

    fn list(&self) -> Result<Vec<FederatedIdentityRecord>> {
        let identities = self.identities.read();
        Ok(identities.values().cloned().collect())
    }
}

/// Resolve and project foreign identities without granting local authority.
pub struct FederationIntake {
    resolver: Arc<dyn FederatedDidDocumentResolver>,
    inventory: Arc<dyn FederatedIdentityInventory>,
    local_issuers: Vec<String>,
}

impl FederationIntake {
    /// Construct an intake service over a resolver and derived inventory sink.
    pub fn new(
        resolver: Arc<dyn FederatedDidDocumentResolver>,
        inventory: Arc<dyn FederatedIdentityInventory>,
        local_issuers: Vec<String>,
    ) -> Self {
        Self {
            resolver,
            inventory,
            local_issuers,
        }
    }

    /// Resolve and index a foreign DID discovered without identity claims.
    pub async fn intake(&self, did: &str) -> Result<FederatedIdentityRecord> {
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
    ) -> Result<FederatedIdentityRecord> {
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

    async fn intake_inner(&self, did: &str) -> Result<FederatedIdentityRecord> {
        let document = self.resolver.resolve_federated_document(did).await?;
        let identity = FederatedIdentityRecord::new(did.to_owned(), document);
        self.inventory.upsert(identity.clone())?;
        Ok(identity)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use hyprstream_rpc::auth::mac::{MacDecision, SecurityContext};
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

    #[tokio::test]
    async fn federated_did_resolves_lists_and_has_no_hosted_tenant() {
        let resolver = Arc::new(FederatedDidResolver::new(
            Arc::new(FixtureResolver {
                expected_did: FEDERATED_DID,
                document: serde_json::json!({
                    "id": FEDERATED_DID,
                    "alsoKnownAs": ["at://alice.example"]
                }),
            }),
            Arc::new(NeverResolver),
        ));
        let inventory = Arc::new(InMemoryFederatedIdentityInventory::default());
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
        assert_eq!(resolved.tenant(), None);

        let listed = inventory.list().unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].did(), FEDERATED_DID);
        assert_eq!(listed[0].tenant(), None);

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
                    "service": []
                }),
            }),
        ));
        let inventory = Arc::new(InMemoryFederatedIdentityInventory::default());
        let intake = FederationIntake::new(resolver, inventory.clone(), Vec::new());

        let resolved = intake.intake(FEDERATED_WEB_DID).await.unwrap();
        assert_eq!(resolved.did(), FEDERATED_WEB_DID);
        assert_eq!(resolved.tenant(), None);
        assert_eq!(inventory.list().unwrap(), vec![resolved]);
    }

    #[tokio::test]
    async fn local_identity_claims_cannot_enter_federation_intake() {
        let resolver = Arc::new(FederatedDidResolver::new(
            Arc::new(NeverResolver),
            Arc::new(NeverResolver),
        ));
        let inventory = Arc::new(InMemoryFederatedIdentityInventory::default());
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
        assert!(inventory.list().unwrap().is_empty());
    }
}
