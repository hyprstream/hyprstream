//! Durable PDS service seams over Hyprstream's subject-carrying 9P plane.
//!
//! This crate owns the service boundary between the pure `hyprstream-pds`
//! record library and deployment storage. It deliberately has no host-path or
//! database API: callers provide the mount rooted at `/pds`, allowing CAS (and
//! its storage-layer sealing policy) to remain below the service.
//!
//! The first demo slice is the account-record read path. A read scope can only
//! be created from [`EnvelopeContext::domain`], which derives its value from
//! `verified_tenant`. The tenant is never inferred from the subject, accepted
//! from a request payload, or supplied as a free-form method argument.

use std::sync::Arc;

use hyprstream_pds::AccountRecord;
use hyprstream_rpc::{EnvelopeContext, Subject};
use hyprstream_vfs::{Mount, MountError, OREAD};
use thiserror::Error;

/// Namespace location at which an [`AccountRecordStore`] backing mount is bound.
pub const PDS_NAMESPACE: &str = "/pds";

const ACCOUNTS_DIRECTORY: &str = "accounts";
const ACCOUNT_RECORD_FILE: &str = "account-record.cbor";
const DEFAULT_MAX_RECORD_BYTES: usize = 64 * 1024;
const READ_CHUNK_BYTES: usize = 8 * 1024;
const OAUTH_ACCOUNT_RESOLVER_SUBJECT: &str = "service:oauth";

/// Failure from the mandatory tenant boundary or the PDS record read.
#[derive(Debug, Error)]
pub enum AccountReadError {
    #[error("PDS account access denied: caller is anonymous")]
    AnonymousCaller,
    #[error("PDS account access denied: no valid verified tenant")]
    MissingVerifiedTenant,
    #[error("PDS account access denied: invalid verified tenant {0:?}")]
    InvalidVerifiedTenant(String),
    #[error("invalid hosted account label {0:?}")]
    InvalidAccountLabel(String),
    #[error("PDS hosted-account tenant resolution denied for {0:?}")]
    UnauthorizedTenantResolver(String),
    #[error("invalid hosted account DID {0:?}")]
    InvalidHostedAccountDid(String),
    #[error("hosted account DID {0:?} is bound to more than one tenant")]
    AmbiguousHostedAccountDid(String),
    #[error("PDS account record exceeds the {limit}-byte read limit")]
    RecordTooLarge { limit: usize },
    #[error("PDS account record for {requested:?} contains label {stored:?}")]
    RecordLabelMismatch { requested: String, stored: String },
    #[error("invalid PDS account record: {0}")]
    InvalidRecord(#[source] anyhow::Error),
    #[error("PDS mount operation failed: {0}")]
    Mount(#[source] MountError),
}

impl From<MountError> for AccountReadError {
    fn from(error: MountError) -> Self {
        Self::Mount(error)
    }
}

/// Read-only account record service over a mount rooted at [`PDS_NAMESPACE`].
///
/// The mount remains responsible for its normal per-operation authorization.
/// This service adds the cross-tenant invariant: every path begins with the
/// authority-verified tenant captured by [`Self::scope`].
#[derive(Clone)]
pub struct AccountRecordStore {
    pds_mount: Arc<dyn Mount>,
    max_record_bytes: usize,
}

impl AccountRecordStore {
    /// Construct a store over the mount bound at `/pds`.
    pub fn new(pds_mount: Arc<dyn Mount>) -> Self {
        Self {
            pds_mount,
            max_record_bytes: DEFAULT_MAX_RECORD_BYTES,
        }
    }

    /// Capture a tenant-scoped account reader from a verified RPC context.
    ///
    /// `EnvelopeContext::domain()` is the only tenant input. It fails closed
    /// when `verified_tenant` is missing, empty, or wildcard. This method then
    /// applies path-component validation before the tenant can reach 9P.
    pub fn scope(&self, context: &EnvelopeContext) -> Result<AccountReadScope, AccountReadError> {
        let subject = context.subject();
        if subject.is_anonymous() {
            return Err(AccountReadError::AnonymousCaller);
        }

        let tenant = context
            .domain()
            .map_err(|_| AccountReadError::MissingVerifiedTenant)?;
        validate_tenant_component(&tenant)?;

        Ok(AccountReadScope {
            pds_mount: Arc::clone(&self.pds_mount),
            max_record_bytes: self.max_record_bytes,
            tenant,
            subject,
        })
    }

    /// Resolve a hosted account DID to its authority-owned tenant binding.
    ///
    /// Unlike [`Self::scope`], this lookup starts without a tenant because the
    /// ATProto assertion proves only the DID. It therefore accepts no tenant
    /// argument: the OAuth service identity scans the account-record index and
    /// returns the tenant containing the one canonical record whose DID
    /// matches. Missing records return `Ok(None)` (federated-only identity);
    /// corrupt, denied, or ambiguous records fail closed.
    pub async fn resolve_tenant_for_hosted_did(
        &self,
        authority: &Subject,
        did: &str,
    ) -> Result<Option<String>, AccountReadError> {
        if authority.name() != Some(OAUTH_ACCOUNT_RESOLVER_SUBJECT) {
            return Err(AccountReadError::UnauthorizedTenantResolver(
                authority.to_string(),
            ));
        }
        let Some(label) = hosted_account_label(did)? else {
            return Ok(None);
        };

        let tenants = read_directory(self.pds_mount.as_ref(), &[], authority).await?;
        let mut resolved = None;
        for entry in tenants.into_iter().filter(|entry| entry.is_dir) {
            validate_tenant_component(&entry.name)?;
            let components = [
                entry.name.as_str(),
                ACCOUNTS_DIRECTORY,
                label,
                ACCOUNT_RECORD_FILE,
            ];
            let bytes = match read_file(
                self.pds_mount.as_ref(),
                &components,
                authority,
                self.max_record_bytes,
            )
            .await
            {
                Ok(bytes) => bytes,
                Err(AccountReadError::Mount(MountError::NotFound(_))) => continue,
                Err(error) => return Err(error),
            };
            let record =
                AccountRecord::from_dag_cbor(&bytes).map_err(AccountReadError::InvalidRecord)?;
            if record.name().label() != label {
                return Err(AccountReadError::RecordLabelMismatch {
                    requested: label.to_owned(),
                    stored: record.name().label().to_owned(),
                });
            }
            if record.name().did() != did {
                continue;
            }
            if resolved.is_some() {
                return Err(AccountReadError::AmbiguousHostedAccountDid(did.to_owned()));
            }
            resolved = Some(entry.name);
        }
        Ok(resolved)
    }

    #[cfg(test)]
    fn with_max_record_bytes(mut self, max_record_bytes: usize) -> Self {
        self.max_record_bytes = max_record_bytes;
        self
    }
}

/// A read capability bound to one authority-verified tenant and subject.
///
/// There is intentionally no method that changes the tenant after creation.
pub struct AccountReadScope {
    pds_mount: Arc<dyn Mount>,
    max_record_bytes: usize,
    tenant: String,
    subject: Subject,
}

impl AccountReadScope {
    /// The verified tenant captured for this scope.
    #[must_use]
    pub fn tenant(&self) -> &str {
        &self.tenant
    }

    /// Read and validate a hosted account record by allocated label.
    ///
    /// Records live at
    /// `/pds/{verified_tenant}/accounts/{label}/account-record.cbor`.
    /// A missing record returns `Ok(None)`; malformed, oversized, mislabeled,
    /// or denied records fail closed.
    pub async fn get(&self, label: &str) -> Result<Option<AccountRecord>, AccountReadError> {
        validate_account_label(label)?;
        let components = [
            self.tenant.as_str(),
            ACCOUNTS_DIRECTORY,
            label,
            ACCOUNT_RECORD_FILE,
        ];
        let bytes = match read_file(
            self.pds_mount.as_ref(),
            &components,
            &self.subject,
            self.max_record_bytes,
        )
        .await
        {
            Ok(bytes) => bytes,
            Err(AccountReadError::Mount(MountError::NotFound(_))) => return Ok(None),
            Err(error) => return Err(error),
        };

        let record =
            AccountRecord::from_dag_cbor(&bytes).map_err(AccountReadError::InvalidRecord)?;
        if record.name().label() != label {
            return Err(AccountReadError::RecordLabelMismatch {
                requested: label.to_owned(),
                stored: record.name().label().to_owned(),
            });
        }
        Ok(Some(record))
    }
}

fn validate_tenant_component(tenant: &str) -> Result<(), AccountReadError> {
    let valid = !tenant.is_empty()
        && tenant.len() <= 253
        && tenant != "."
        && tenant != ".."
        && tenant != "*"
        && tenant
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'));
    if valid {
        Ok(())
    } else {
        Err(AccountReadError::InvalidVerifiedTenant(tenant.to_owned()))
    }
}

fn validate_account_label(label: &str) -> Result<(), AccountReadError> {
    let valid = !label.is_empty()
        && label.len() <= 63
        && label
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && !label.starts_with('-')
        && !label.ends_with('-');
    if valid {
        Ok(())
    } else {
        Err(AccountReadError::InvalidAccountLabel(label.to_owned()))
    }
}

fn hosted_account_label(did: &str) -> Result<Option<&str>, AccountReadError> {
    let Some(host) = did.strip_prefix("did:web:") else {
        return Ok(None);
    };
    if host.contains([':', '/']) {
        return Err(AccountReadError::InvalidHostedAccountDid(did.to_owned()));
    }
    let label = host
        .split('.')
        .next()
        .ok_or_else(|| AccountReadError::InvalidHostedAccountDid(did.to_owned()))?;
    validate_account_label(label)
        .map_err(|_| AccountReadError::InvalidHostedAccountDid(did.to_owned()))?;
    Ok(Some(label))
}

async fn read_file(
    mount: &dyn Mount,
    components: &[&str],
    subject: &Subject,
    limit: usize,
) -> Result<Vec<u8>, AccountReadError> {
    let mut fid = mount.walk(components, subject).await?;
    if let Err(error) = mount.open(&mut fid, OREAD, subject).await {
        mount.clunk(fid, subject).await;
        return Err(error.into());
    }

    let result = read_open_fid(mount, &fid, subject, limit).await;
    mount.clunk(fid, subject).await;
    result
}

async fn read_directory(
    mount: &dyn Mount,
    components: &[&str],
    subject: &Subject,
) -> Result<Vec<hyprstream_vfs::DirEntry>, AccountReadError> {
    let mut fid = mount.walk(components, subject).await?;
    if let Err(error) = mount.open(&mut fid, OREAD, subject).await {
        mount.clunk(fid, subject).await;
        return Err(error.into());
    }
    let result = mount.readdir(&fid, subject).await.map_err(Into::into);
    mount.clunk(fid, subject).await;
    result
}

async fn read_open_fid(
    mount: &dyn Mount,
    fid: &hyprstream_vfs::Fid,
    subject: &Subject,
    limit: usize,
) -> Result<Vec<u8>, AccountReadError> {
    let stat = mount.stat(fid, subject).await?;
    if stat.size > limit as u64 {
        return Err(AccountReadError::RecordTooLarge { limit });
    }

    let mut bytes = Vec::with_capacity((stat.size as usize).min(limit));
    loop {
        let remaining_with_sentinel = limit.saturating_sub(bytes.len()).saturating_add(1);
        let count = READ_CHUNK_BYTES.min(remaining_with_sentinel) as u32;
        let chunk = mount.read(fid, bytes.len() as u64, count, subject).await?;
        if chunk.len() > count as usize {
            return Err(AccountReadError::RecordTooLarge { limit });
        }
        if chunk.is_empty() {
            break;
        }
        bytes.extend_from_slice(&chunk);
        if bytes.len() > limit {
            return Err(AccountReadError::RecordTooLarge { limit });
        }
    }
    Ok(bytes)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use ed25519_dalek::SigningKey;
    use hyprstream_crypto::pq::{ml_dsa_generate_keypair, ml_dsa_vk_bytes};
    use hyprstream_pds::did_op::{
        sign_genesis, GenesisRepoHead, GenesisRotationKeys, HostKeyEnrollment, HybridRotationKey,
        RecoveryKeyEnrollment, UserRotationKey,
    };
    use hyprstream_pds::{AllocatedAccountName, Cid, HostedAccountMint};
    use hyprstream_rpc::Subject;
    use hyprstream_vfs::{SyntheticMount, SyntheticNode};
    use rand::rngs::OsRng;

    fn account_bytes(label: &str, zone: &str) -> Vec<u8> {
        let ed = SigningKey::generate(&mut OsRng);
        let (pq, pq_vk) = ml_dsa_generate_keypair();
        let hybrid =
            HybridRotationKey::new(ed.verifying_key().to_bytes(), ml_dsa_vk_bytes(&pq_vk)).unwrap();
        let rotations = GenesisRotationKeys::new(
            UserRotationKey::new(hybrid),
            RecoveryKeyEnrollment::Declined,
            HostKeyEnrollment::Absent,
        )
        .unwrap();
        let name = AllocatedAccountName::new(label, format!("did:web:{label}.{zone}")).unwrap();
        let pending = HostedAccountMint::begin(name, rotations)
            .unwrap()
            .prepare_genesis(Cid::from_raw(zone.as_bytes()), GenesisRepoHead::EmptyRepo)
            .unwrap();
        let signature = sign_genesis(pending.unsigned_genesis(), &ed, &pq).unwrap();
        pending.seal(signature).unwrap().record_bytes().to_vec()
    }

    fn tenant_node(label: &str, record: Vec<u8>) -> SyntheticNode {
        SyntheticNode::dir().with_child(
            ACCOUNTS_DIRECTORY,
            SyntheticNode::dir().with_child(
                label,
                SyntheticNode::dir().with_child(ACCOUNT_RECORD_FILE, SyntheticNode::file(record)),
            ),
        )
    }

    fn store() -> AccountRecordStore {
        let root = SyntheticNode::dir()
            .with_child(
                "acme",
                tenant_node("alice", account_bytes("alice", "acme.example")),
            )
            .with_child(
                "beta",
                tenant_node("alice", account_bytes("alice", "beta.example")),
            );
        AccountRecordStore::new(Arc::new(SyntheticMount::new(root)))
    }

    fn context(tenant: &str) -> EnvelopeContext {
        let signer = SigningKey::generate(&mut OsRng);
        EnvelopeContext::for_test_authenticated_subject_in_tenant(
            Subject::new("alice"),
            tenant,
            signer.verifying_key(),
        )
    }

    fn oauth_authority() -> Subject {
        Subject::new(OAUTH_ACCOUNT_RESOLVER_SUBJECT)
    }

    #[tokio::test]
    async fn verified_tenant_selects_the_only_visible_account_tree() {
        let store = store();
        let acme = store.scope(&context("acme")).unwrap();
        let beta = store.scope(&context("beta")).unwrap();

        let acme_record = acme.get("alice").await.unwrap().unwrap();
        let beta_record = beta.get("alice").await.unwrap().unwrap();

        assert_eq!(acme.tenant(), "acme");
        assert_eq!(beta.tenant(), "beta");
        assert_eq!(acme_record.name().did(), "did:web:alice.acme.example");
        assert_eq!(beta_record.name().did(), "did:web:alice.beta.example");
    }

    #[tokio::test]
    async fn oauth_authority_resolves_tenant_from_matching_account_record() {
        let store = store();

        assert_eq!(
            store
                .resolve_tenant_for_hosted_did(&oauth_authority(), "did:web:alice.acme.example",)
                .await
                .unwrap()
                .as_deref(),
            Some("acme"),
        );
        assert_eq!(
            store
                .resolve_tenant_for_hosted_did(&oauth_authority(), "did:web:missing.acme.example",)
                .await
                .unwrap(),
            None,
        );
        assert_eq!(
            store
                .resolve_tenant_for_hosted_did(&oauth_authority(), "did:plc:federated-only",)
                .await
                .unwrap(),
            None,
        );
    }

    #[tokio::test]
    async fn hosted_did_resolution_requires_oauth_authority_and_is_unambiguous() {
        let denied = store()
            .resolve_tenant_for_hosted_did(&Subject::new("alice"), "did:web:alice.acme.example")
            .await
            .unwrap_err();
        assert!(matches!(
            denied,
            AccountReadError::UnauthorizedTenantResolver(_)
        ));

        let duplicated = account_bytes("alice", "acme.example");
        let root = SyntheticNode::dir()
            .with_child("acme", tenant_node("alice", duplicated.clone()))
            .with_child("beta", tenant_node("alice", duplicated));
        let ambiguous = AccountRecordStore::new(Arc::new(SyntheticMount::new(root)))
            .resolve_tenant_for_hosted_did(&oauth_authority(), "did:web:alice.acme.example")
            .await
            .unwrap_err();
        assert!(matches!(
            ambiguous,
            AccountReadError::AmbiguousHostedAccountDid(_)
        ));
    }

    #[test]
    fn missing_verified_tenant_fails_closed() {
        let signer = SigningKey::generate(&mut OsRng);
        let context = EnvelopeContext::for_test_authenticated_subject(
            Subject::new("alice"),
            signer.verifying_key(),
        );
        let error = store().scope(&context).err().expect("scope must be denied");
        assert!(matches!(error, AccountReadError::MissingVerifiedTenant));
    }

    #[tokio::test]
    async fn anonymous_and_path_injection_fail_before_mount_access() {
        let store = store();
        let signer = SigningKey::generate(&mut OsRng);
        let anonymous = EnvelopeContext::for_test_authenticated_subject_in_tenant(
            Subject::anonymous(),
            "acme",
            signer.verifying_key(),
        );
        let error = store
            .scope(&anonymous)
            .err()
            .expect("anonymous caller must be denied");
        assert!(matches!(error, AccountReadError::AnonymousCaller));

        let error = store
            .scope(&context("../acme"))
            .err()
            .expect("invalid tenant must be denied");
        assert!(matches!(error, AccountReadError::InvalidVerifiedTenant(_)));

        let scope = store.scope(&context("acme")).unwrap();
        let error = scope
            .get("../beta")
            .await
            .expect_err("invalid label must be denied");
        assert!(matches!(error, AccountReadError::InvalidAccountLabel(_)));
    }

    #[tokio::test]
    async fn oversized_and_mislabeled_records_fail_closed() {
        let oversized = SyntheticNode::dir().with_child("acme", tenant_node("alice", vec![0; 32]));
        let store = AccountRecordStore::new(Arc::new(SyntheticMount::new(oversized)))
            .with_max_record_bytes(8);
        let error = store
            .scope(&context("acme"))
            .unwrap()
            .get("alice")
            .await
            .expect_err("oversized record must fail");
        assert!(matches!(error, AccountReadError::RecordTooLarge { .. }));

        let mislabeled = SyntheticNode::dir().with_child(
            "acme",
            tenant_node("bob", account_bytes("alice", "acme.example")),
        );
        let error = AccountRecordStore::new(Arc::new(SyntheticMount::new(mislabeled)))
            .scope(&context("acme"))
            .unwrap()
            .get("bob")
            .await
            .expect_err("mislabeled record must fail");
        assert!(matches!(
            error,
            AccountReadError::RecordLabelMismatch { .. }
        ));
    }
}
