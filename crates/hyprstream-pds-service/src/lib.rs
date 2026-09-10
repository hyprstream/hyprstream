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

pub mod account_http;
pub mod federation_intake;

use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use hyprstream_pds::{
    AccountLabel, AccountRecord, ATPROTO_SIGNING_KEY_FILE, DID_DOCUMENT_FILE, GENESIS_DID_OP_FILE,
};
use hyprstream_rpc::auth::mac::{MacDecision, MacDenyReason, SecurityContext};
use hyprstream_rpc::{EnvelopeContext, Subject};
use hyprstream_vfs::{Mount, MountError, OREAD};
use thiserror::Error;

/// Namespace location at which an [`AccountRecordStore`] backing mount is bound.
pub const PDS_NAMESPACE: &str = "/pds";

/// Directory component containing hosted account records within a tenant.
pub const PDS_ACCOUNTS_DIRECTORY: &str = "accounts";
/// Public account-record publication marker.
pub const PDS_ACCOUNT_RECORD_FILE: &str = "account-record.cbor";
/// Sealed account DID-document bytes published beside the account record.
pub const PDS_ACCOUNT_DID_DOCUMENT_FILE: &str = DID_DOCUMENT_FILE;
/// Sealed account operation-log bytes published beside the account record.
pub const PDS_ACCOUNT_DID_LOG_FILE: &str = GENESIS_DID_OP_FILE;

pub mod hosted_account_mint;

/// Fixed internal principal allowed to resolve a hosted DID to its tenant.
pub const OAUTH_ACCOUNT_RESOLVER_SUBJECT: &str = "service:oauth";
const DEFAULT_MAX_RECORD_BYTES: usize = 64 * 1024;
const READ_CHUNK_BYTES: usize = 8 * 1024;
const ATPROTO_SIGNING_KEY_BYTES: usize = 32;

/// Failure from the mandatory tenant boundary or the PDS record read.
#[derive(Debug, Error)]
pub enum AccountReadError {
    #[error("PDS account access denied: caller is unauthenticated")]
    UnauthenticatedCaller,
    #[error("PDS account access denied: no valid verified tenant")]
    MissingVerifiedTenant,
    #[error("PDS account access denied: invalid verified tenant {0:?}")]
    InvalidVerifiedTenant(String),
    #[error("invalid hosted account label {0:?}")]
    InvalidAccountLabel(String),
    #[error("PDS hosted-account tenant resolution denied for {0:?}")]
    UnauthorizedTenantResolver(String),
    #[error("PDS account read denied for {subject:?} on {object:?}: {reason:?}")]
    MacDenied {
        subject: String,
        object: String,
        reason: MacDenyReason,
    },
    #[error("invalid hosted account DID {0:?}")]
    InvalidHostedAccountDid(String),
    #[error("hosted account DID {0:?} is bound to more than one tenant")]
    AmbiguousHostedAccountDid(String),
    #[error("hosted account DID index is still warming")]
    HostedDidIndexNotReady,
    #[error("PDS account record exceeds the {limit}-byte read limit")]
    RecordTooLarge { limit: usize },
    #[error("PDS account record for {requested:?} contains label {stored:?}")]
    RecordLabelMismatch { requested: String, stored: String },
    #[error("PDS account record for {requested:?} contains DID {stored:?}")]
    RecordDidMismatch { requested: String, stored: String },
    #[error("hosted account #atproto signing key for {0:?} is unavailable")]
    SigningKeyUnavailable(String),
    #[error("hosted account #atproto signing key for {0:?} is invalid")]
    InvalidSigningKey(String),
    #[error("hosted account #atproto signing key for {0:?} does not match its account record")]
    SigningKeyMismatch(String),
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

/// Mandatory MAC capability for account-record reads.
///
/// Implementations resolve the subject clearance and the trusted label of the
/// exact canonical `object_id`, apply the lattice floor, and audit every MAC
/// decision. Input-validation and mount-resolution errors are not policy
/// decisions; they fail before an object read is attempted. The account store
/// has no constructor without this capability.
pub trait AccountRecordReadAuthorizer: Send + Sync {
    /// Check a read before the store opens or reads the fid bound to
    /// `object_id`.
    ///
    /// `security_context` is derived from the verified request envelope and
    /// retains its assurance and delegated attenuation. It is `None` only when
    /// the request had no MAC clearance, or for the fixed OAuth service's
    /// tenant-index lookup before a hosted-account binding has been resolved.
    fn check_read(
        &self,
        subject: &Subject,
        verified_tenant: Option<&str>,
        security_context: Option<&SecurityContext>,
        object_id: &str,
    ) -> MacDecision;
}

/// Read-only account record service over a mount rooted at [`PDS_NAMESPACE`].
///
/// Every construction requires an [`AccountRecordReadAuthorizer`]. The store
/// binds the requested canonical name to one fid, MAC-authorizes that exact
/// name, and only then opens and reads from the same fid.
#[derive(Clone)]
pub struct AccountRecordStore {
    pds_mount: Arc<dyn Mount>,
    read_authorizer: Arc<dyn AccountRecordReadAuthorizer>,
    max_record_bytes: usize,
    hosted_did_index: Arc<tokio::sync::RwLock<Option<HostedDidIndex>>>,
    hosted_did_index_refresh: Arc<tokio::sync::Mutex<()>>,
    hosted_did_index_refreshing: Arc<AtomicBool>,
    hosted_did_index_last_attempt: Arc<parking_lot::Mutex<Option<Instant>>>,
}

struct HostedDidIndex {
    entries: BTreeMap<(String, String), Option<String>>,
    built_at: Instant,
}

const HOSTED_DID_INDEX_TTL: Duration = Duration::from_secs(5);
const HOSTED_DID_MAX_STALE: Duration = Duration::from_secs(30);
const HOSTED_DID_REFRESH_RETRY_COOLDOWN: Duration = Duration::from_secs(1);
const HOSTED_DID_NEGATIVE_TTL: Duration = HOSTED_DID_INDEX_TTL;

impl AccountRecordStore {
    /// Construct a store over the mount bound at `/pds` and a mandatory MAC
    /// read capability.
    pub fn new(
        pds_mount: Arc<dyn Mount>,
        read_authorizer: Arc<dyn AccountRecordReadAuthorizer>,
    ) -> Self {
        Self {
            pds_mount,
            read_authorizer,
            max_record_bytes: DEFAULT_MAX_RECORD_BYTES,
            hosted_did_index: Arc::new(tokio::sync::RwLock::new(None)),
            hosted_did_index_refresh: Arc::new(tokio::sync::Mutex::new(())),
            hosted_did_index_refreshing: Arc::new(AtomicBool::new(false)),
            hosted_did_index_last_attempt: Arc::new(parking_lot::Mutex::new(None)),
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
            return Err(AccountReadError::UnauthenticatedCaller);
        }

        let tenant = context
            .domain()
            .map_err(|_| AccountReadError::MissingVerifiedTenant)?;
        validate_tenant_component(&tenant)?;

        Ok(AccountReadScope {
            pds_mount: Arc::clone(&self.pds_mount),
            read_authorizer: Arc::clone(&self.read_authorizer),
            max_record_bytes: self.max_record_bytes,
            tenant,
            subject,
            security_context: context.security_context(),
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

        let index = self.hosted_did_index.read().await;
        let Some(snapshot) = index.as_ref() else {
            // A request must never become the tenant enumerator. Startup
            // refresh runs independently; callers retry after the index is
            // ready instead of amplifying a full mount scan per miss.
            self.schedule_hosted_did_index_refresh(authority.clone());
            return Err(AccountReadError::HostedDidIndexNotReady);
        };
        let age = snapshot.built_at.elapsed();
        if age >= HOSTED_DID_INDEX_TTL {
            self.schedule_hosted_did_index_refresh(authority.clone());
        }
        if age >= HOSTED_DID_INDEX_TTL + HOSTED_DID_MAX_STALE {
            return Err(AccountReadError::HostedDidIndexNotReady);
        }
        let key = (label.to_owned(), did.to_owned());
        match snapshot.entries.get(&key) {
            Some(Some(tenant)) => Ok(Some(tenant.clone())),
            Some(None) => Err(AccountReadError::AmbiguousHostedAccountDid(did.to_owned())),
            None if age < HOSTED_DID_NEGATIVE_TTL => Ok(None),
            None => {
                // A negative entry is eligible for refresh at the normal
                // snapshot interval, but remains a stable 404 while refresh
                // runs. This avoids turning ordinary missing-account traffic
                // into repeated transient OAuth/HTTP failures.
                self.schedule_hosted_did_index_refresh(authority.clone());
                Ok(None)
            }
        }
    }

    /// Start a bounded refresh without making the caller perform tenant
    /// enumeration. A stale snapshot remains available for O(1) lookups while
    /// one background task refreshes it; the refresh lock prevents duplicate
    /// scans. Before the first snapshot, lookups return
    /// [`AccountReadError::HostedDidIndexNotReady`] while this task warms it.
    pub fn schedule_hosted_did_index_refresh(&self, authority: Subject) {
        if self
            .hosted_did_index_refreshing
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        {
            let mut last_attempt = self.hosted_did_index_last_attempt.lock();
            if last_attempt
                .is_some_and(|attempt| attempt.elapsed() < HOSTED_DID_REFRESH_RETRY_COOLDOWN)
            {
                self.hosted_did_index_refreshing
                    .store(false, Ordering::Release);
                return;
            }
            *last_attempt = Some(Instant::now());
        }
        let store = self.clone();
        if std::thread::Builder::new()
            .name("hyprstream-pds-index-refresh".to_owned())
            .spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(_) => {
                    store
                        .hosted_did_index_refreshing
                        .store(false, Ordering::Release);
                    return;
                }
            };
            let refresh_store = store.clone();
            runtime.block_on(async {
                let _ = refresh_store.refresh_hosted_did_index(&authority).await;
            });
            store
                .hosted_did_index_refreshing
                .store(false, Ordering::Release);
            })
            .is_err()
        {
            self.hosted_did_index_refreshing
                .store(false, Ordering::Release);
        }
    }

    pub async fn refresh_hosted_did_index(
        &self,
        authority: &Subject,
    ) -> Result<(), AccountReadError> {
        let _refresh = self.hosted_did_index_refresh.lock().await;
        let stale = self
            .hosted_did_index
            .read()
            .await
            .as_ref()
            .map(|snapshot| snapshot.built_at.elapsed() >= HOSTED_DID_INDEX_TTL)
            .unwrap_or(true);
        if stale {
            let entries = self.build_hosted_did_index(authority).await?;
            self.hosted_did_index.write().await.replace(HostedDidIndex {
                entries,
                built_at: Instant::now(),
            });
        }
        Ok(())
    }

    async fn build_hosted_did_index(
        &self,
        authority: &Subject,
    ) -> Result<BTreeMap<(String, String), Option<String>>, AccountReadError> {
        let tenants = read_directory(
            self.pds_mount.as_ref(),
            self.read_authorizer.as_ref(),
            &[],
            authority,
            None,
            None,
        )
        .await?;
        let mut index = BTreeMap::new();
        for entry in tenants.into_iter().filter(|entry| entry.is_dir) {
            validate_tenant_component(&entry.name)?;
            let accounts_components = [entry.name.as_str(), PDS_ACCOUNTS_DIRECTORY];
            let accounts = match read_directory(
                self.pds_mount.as_ref(),
                self.read_authorizer.as_ref(),
                &accounts_components,
                authority,
                Some(entry.name.as_str()),
                None,
            )
            .await
            {
                Ok(bytes) => bytes,
                Err(AccountReadError::Mount(MountError::NotFound(_))) => continue,
                Err(error) => return Err(error),
            };
            for account in accounts.into_iter().filter(|account| account.is_dir) {
                let label = account.name;
                if hyprstream_pds::is_hosted_account_staging_directory(&label) {
                    continue;
                }
                AccountLabel::parse(&label)
                    .map_err(|_| AccountReadError::InvalidAccountLabel(label.clone()))?;
                let components = [
                    entry.name.as_str(),
                    PDS_ACCOUNTS_DIRECTORY,
                    label.as_str(),
                    PDS_ACCOUNT_RECORD_FILE,
                ];
                let bytes = match read_file(
                    self.pds_mount.as_ref(),
                    self.read_authorizer.as_ref(),
                    &components,
                    authority,
                    Some(entry.name.as_str()),
                    None,
                    self.max_record_bytes,
                )
                .await
                {
                    Ok(bytes) => bytes,
                    Err(AccountReadError::Mount(MountError::NotFound(_))) => continue,
                    Err(error) => return Err(error),
                };
                let record = AccountRecord::from_dag_cbor(&bytes)
                    .map_err(AccountReadError::InvalidRecord)?;
                if record.name().label() != label {
                    return Err(AccountReadError::RecordLabelMismatch {
                        requested: label,
                        stored: record.name().label().to_owned(),
                    });
                }
                let key = (label, record.name().did().to_owned());
                match index.entry(key) {
                    std::collections::btree_map::Entry::Vacant(slot) => {
                        slot.insert(Some(entry.name.clone()));
                    }
                    std::collections::btree_map::Entry::Occupied(mut slot) => {
                        slot.insert(None);
                    }
                }
            }
        }
        Ok(index)
    }

    /// Sign bytes with the account-specific `#atproto` key for one hosted DID.
    ///
    /// This is deliberately an authority-only operation: the caller supplies
    /// neither a tenant nor a label. The store resolves the tenant from the
    /// immutable account index, reloads the exact public account record, and
    /// verifies the secret key against that record before signing. Missing,
    /// corrupt, substituted, or ambiguous state fails closed and the secret
    /// key never leaves this service boundary.
    pub async fn sign_for_hosted_did(
        &self,
        authority: &Subject,
        did: &str,
        signing_input: &[u8],
    ) -> Result<Option<Vec<u8>>, AccountReadError> {
        use p256::ecdsa::signature::Signer as _;

        // Signing runs on authenticated request paths; never turn a cold or
        // hard-stale lookup into a synchronous tenant enumeration. Startup
        // warms the index before OAuth readiness, while this resolver remains
        // fail-closed until a snapshot is available.
        let resolved = self.resolve_tenant_for_hosted_did(authority, did).await?;
        let Some(tenant) = resolved else {
            return Ok(None);
        };
        let Some(label) = hosted_account_label(did)? else {
            return Ok(None);
        };

        let record_components = [
            tenant.as_str(),
            PDS_ACCOUNTS_DIRECTORY,
            label,
            PDS_ACCOUNT_RECORD_FILE,
        ];
        let record_bytes = read_file(
            self.pds_mount.as_ref(),
            self.read_authorizer.as_ref(),
            &record_components,
            authority,
            Some(tenant.as_str()),
            None,
            self.max_record_bytes,
        )
        .await?;
        let record =
            AccountRecord::from_dag_cbor(&record_bytes).map_err(AccountReadError::InvalidRecord)?;
        if record.name().label() != label {
            return Err(AccountReadError::RecordLabelMismatch {
                requested: label.to_owned(),
                stored: record.name().label().to_owned(),
            });
        }
        if record.name().did() != did {
            return Err(AccountReadError::RecordDidMismatch {
                requested: did.to_owned(),
                stored: record.name().did().to_owned(),
            });
        }

        let key_components = [
            tenant.as_str(),
            PDS_ACCOUNTS_DIRECTORY,
            label,
            ATPROTO_SIGNING_KEY_FILE,
        ];
        let key_bytes = match read_file(
            self.pds_mount.as_ref(),
            self.read_authorizer.as_ref(),
            &key_components,
            authority,
            Some(tenant.as_str()),
            None,
            ATPROTO_SIGNING_KEY_BYTES,
        )
        .await
        {
            Ok(bytes) => bytes,
            Err(AccountReadError::Mount(MountError::NotFound(_))) => {
                return Err(AccountReadError::SigningKeyUnavailable(did.to_owned()));
            }
            Err(AccountReadError::RecordTooLarge { .. }) => {
                return Err(AccountReadError::InvalidSigningKey(did.to_owned()));
            }
            Err(error) => return Err(error),
        };
        if key_bytes.len() != ATPROTO_SIGNING_KEY_BYTES {
            return Err(AccountReadError::InvalidSigningKey(did.to_owned()));
        }
        let signing_key = p256::ecdsa::SigningKey::from_slice(&key_bytes)
            .map_err(|_| AccountReadError::InvalidSigningKey(did.to_owned()))?;
        if signing_key.verifying_key().to_encoded_point(true)
            != record
                .atproto_verifying_key()
                .map_err(AccountReadError::InvalidRecord)?
                .to_encoded_point(true)
        {
            return Err(AccountReadError::SigningKeyMismatch(did.to_owned()));
        }

        let signature: p256::ecdsa::Signature = signing_key.sign(signing_input);
        Ok(Some(signature.to_bytes().to_vec()))
    }

    #[cfg(test)]
    fn with_max_record_bytes(mut self, max_record_bytes: usize) -> Self {
        self.max_record_bytes = max_record_bytes;
        self
    }

    /// Read one immutable public identity artifact after the hosted DID has
    /// already resolved to its authority-owned tenant.
    pub(crate) async fn read_hosted_http_artifact(
        &self,
        authority: &Subject,
        tenant: &str,
        label: &str,
        file: &str,
        limit: usize,
    ) -> Result<Vec<u8>, AccountReadError> {
        validate_tenant_component(tenant)?;
        validate_account_label(label)?;
        let components = [tenant, PDS_ACCOUNTS_DIRECTORY, label, file];
        read_file(
            self.pds_mount.as_ref(),
            self.read_authorizer.as_ref(),
            &components,
            authority,
            Some(tenant),
            None,
            limit,
        )
        .await
    }
}

/// A read capability bound to one authority-verified tenant and subject.
///
/// There is intentionally no method that changes the tenant after creation.
pub struct AccountReadScope {
    pds_mount: Arc<dyn Mount>,
    read_authorizer: Arc<dyn AccountRecordReadAuthorizer>,
    max_record_bytes: usize,
    tenant: String,
    subject: Subject,
    security_context: Option<SecurityContext>,
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
            PDS_ACCOUNTS_DIRECTORY,
            label,
            PDS_ACCOUNT_RECORD_FILE,
        ];
        let bytes = match read_file(
            self.pds_mount.as_ref(),
            self.read_authorizer.as_ref(),
            &components,
            &self.subject,
            Some(self.tenant.as_str()),
            self.security_context.as_ref(),
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
    authorizer: &dyn AccountRecordReadAuthorizer,
    components: &[&str],
    subject: &Subject,
    verified_tenant: Option<&str>,
    security_context: Option<&SecurityContext>,
    limit: usize,
) -> Result<Vec<u8>, AccountReadError> {
    let mut fid = mount.walk(components, subject).await?;
    let object_id = canonical_object_id(components);
    if let MacDecision::Deny(reason) =
        authorizer.check_read(subject, verified_tenant, security_context, &object_id)
    {
        mount.clunk(fid, subject).await;
        return Err(AccountReadError::MacDenied {
            subject: subject.to_string(),
            object: object_id,
            reason,
        });
    }

    // The walk above binds the canonical name to this fid. Authorization is
    // complete before open/stat/read, and the same fid is consumed below:
    // there is no partial resolution or post-check re-walk/handle substitution.
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
    authorizer: &dyn AccountRecordReadAuthorizer,
    components: &[&str],
    subject: &Subject,
    verified_tenant: Option<&str>,
    security_context: Option<&SecurityContext>,
) -> Result<Vec<hyprstream_vfs::DirEntry>, AccountReadError> {
    let mut fid = mount.walk(components, subject).await?;
    let object_id = canonical_object_id(components);
    if let MacDecision::Deny(reason) =
        authorizer.check_read(subject, verified_tenant, security_context, &object_id)
    {
        mount.clunk(fid, subject).await;
        return Err(AccountReadError::MacDenied {
            subject: subject.to_string(),
            object: object_id,
            reason,
        });
    }
    if let Err(error) = mount.open(&mut fid, OREAD, subject).await {
        mount.clunk(fid, subject).await;
        return Err(error.into());
    }
    let result = mount.readdir(&fid, subject).await.map_err(Into::into);
    mount.clunk(fid, subject).await;
    result
}

fn canonical_object_id(components: &[&str]) -> String {
    if components.is_empty() {
        return PDS_NAMESPACE.to_owned();
    }
    format!("{PDS_NAMESPACE}/{}", components.join("/"))
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
    use hyprstream_pds::{AllocatedAccountName, HostedAccountMint};
    use hyprstream_rpc::Subject;
    use hyprstream_vfs::{SyntheticMount, SyntheticNode};
    use rand::rngs::OsRng;

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

    struct SlowPermitAccountReads;

    impl AccountRecordReadAuthorizer for SlowPermitAccountReads {
        fn check_read(
            &self,
            _subject: &Subject,
            _verified_tenant: Option<&str>,
            _security_context: Option<&SecurityContext>,
            _object_id: &str,
        ) -> MacDecision {
            // This models the synchronous filesystem/audit work performed by
            // the production authorizer and mount. A current-thread OAuth
            // runtime must remain able to make progress while it runs.
            std::thread::sleep(Duration::from_millis(200));
            MacDecision::Permit
        }
    }

    fn permit_account_reads() -> Arc<dyn AccountRecordReadAuthorizer> {
        Arc::new(PermitAccountReads)
    }

    fn account_artifacts(label: &str, zone: &str) -> (Vec<u8>, Vec<u8>) {
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
        let mint = HostedAccountMint::begin(name, rotations).unwrap();
        let document = mint.seal_did_document("https://pds.example.com").unwrap();
        let pending = mint
            .prepare_genesis(document, GenesisRepoHead::EmptyRepo)
            .unwrap();
        let signature = sign_genesis(pending.unsigned_genesis(), &ed, &pq).unwrap();
        let account = pending.seal(signature).unwrap();
        (
            account.record_bytes().to_vec(),
            account.atproto_signing_key().to_bytes().to_vec(),
        )
    }

    fn account_bytes(label: &str, zone: &str) -> Vec<u8> {
        account_artifacts(label, zone).0
    }

    fn tenant_node(label: &str, record: Vec<u8>) -> SyntheticNode {
        SyntheticNode::dir().with_child(
            PDS_ACCOUNTS_DIRECTORY,
            SyntheticNode::dir().with_child(
                label,
                SyntheticNode::dir()
                    .with_child(PDS_ACCOUNT_RECORD_FILE, SyntheticNode::file(record)),
            ),
        )
    }

    fn tenant_node_with_key(label: &str, record: Vec<u8>, key: Vec<u8>) -> SyntheticNode {
        SyntheticNode::dir().with_child(
            PDS_ACCOUNTS_DIRECTORY,
            SyntheticNode::dir().with_child(
                label,
                SyntheticNode::dir()
                    .with_child(PDS_ACCOUNT_RECORD_FILE, SyntheticNode::file(record))
                    .with_child(ATPROTO_SIGNING_KEY_FILE, SyntheticNode::file(key)),
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
        AccountRecordStore::new(Arc::new(SyntheticMount::new(root)), permit_account_reads())
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
        store
            .refresh_hosted_did_index(&oauth_authority())
            .await
            .unwrap();

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
        assert_eq!(
            store
                .resolve_tenant_for_hosted_did(
                    &oauth_authority(),
                    hyprstream_rpc::identity::UNAUTHENTICATED_DID_SENTINEL,
                )
                .await
                .unwrap(),
            None,
        );
    }

    #[tokio::test]
    async fn hosted_did_index_ignores_mint_staging_directories() {
        let record = account_bytes("alice", "acme.example");
        let accounts = SyntheticNode::dir()
            .with_child(
                "alice",
                SyntheticNode::dir()
                    .with_child(PDS_ACCOUNT_RECORD_FILE, SyntheticNode::file(record)),
            )
            .with_child(".alice.mint-123-0", SyntheticNode::dir());
        let root = SyntheticNode::dir().with_child(
            "acme",
            SyntheticNode::dir().with_child(PDS_ACCOUNTS_DIRECTORY, accounts),
        );
        let store =
            AccountRecordStore::new(Arc::new(SyntheticMount::new(root)), permit_account_reads());

        store
            .refresh_hosted_did_index(&oauth_authority())
            .await
            .expect("staging residue must not abort index refresh");
        assert_eq!(
            store
                .resolve_tenant_for_hosted_did(&oauth_authority(), "did:web:alice.acme.example",)
                .await
                .unwrap()
                .as_deref(),
            Some("acme")
        );
    }

    #[tokio::test]
    async fn hosted_did_index_rejects_malformed_permanent_account_labels() {
        for label in [".alice.mint-*", ".garbage", "Alice", "alice_"] {
            let accounts = SyntheticNode::dir().with_child(label, SyntheticNode::dir());
            let root = SyntheticNode::dir().with_child(
                "acme",
                SyntheticNode::dir().with_child(PDS_ACCOUNTS_DIRECTORY, accounts),
            );
            let store = AccountRecordStore::new(
                Arc::new(SyntheticMount::new(root)),
                permit_account_reads(),
            );

            let error = store
                .refresh_hosted_did_index(&oauth_authority())
                .await
                .expect_err("malformed permanent labels must fail closed");
            assert!(
                matches!(error, AccountReadError::InvalidAccountLabel(ref actual) if actual == label),
                "unexpected error for {label:?}: {error:?}"
            );
        }
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
        let ambiguous_store =
            AccountRecordStore::new(Arc::new(SyntheticMount::new(root)), permit_account_reads());
        ambiguous_store
            .refresh_hosted_did_index(&oauth_authority())
            .await
            .unwrap();
        let ambiguous = ambiguous_store
            .resolve_tenant_for_hosted_did(&oauth_authority(), "did:web:alice.acme.example")
            .await
            .unwrap_err();
        assert!(matches!(
            ambiguous,
            AccountReadError::AmbiguousHostedAccountDid(_)
        ));
    }

    #[tokio::test]
    async fn cold_hosted_did_lookup_never_enumerates_on_request_path() {
        let store = store();
        let error = store
            .resolve_tenant_for_hosted_did(&oauth_authority(), "did:web:alice.acme.example")
            .await
            .unwrap_err();
        assert!(matches!(error, AccountReadError::HostedDidIndexNotReady));
    }

    #[tokio::test]
    async fn cold_hosted_did_signing_never_refreshes_on_request_path() {
        let store = store();
        let error = store
            .sign_for_hosted_did(
                &oauth_authority(),
                "did:web:alice.acme.example",
                b"header.payload",
            )
            .await
            .unwrap_err();
        assert!(matches!(error, AccountReadError::HostedDidIndexNotReady));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn scheduled_hosted_did_refresh_does_not_block_current_thread_runtime() {
        let root = SyntheticNode::dir().with_child(
            "acme",
            tenant_node("alice", account_bytes("alice", "acme.example")),
        );
        let store = AccountRecordStore::new(
            Arc::new(SyntheticMount::new(root)),
            Arc::new(SlowPermitAccountReads),
        );

        store.schedule_hosted_did_index_refresh(oauth_authority());
        tokio::time::timeout(
            Duration::from_millis(100),
            tokio::time::sleep(Duration::from_millis(10)),
        )
        .await
        .expect("index refresh must not block the HTTP runtime");
    }

    #[tokio::test]
    async fn hosted_did_negative_lookup_expires() {
        let store = store();
        store
            .hosted_did_index
            .write()
            .await
            .replace(HostedDidIndex {
                entries: BTreeMap::new(),
                built_at: Instant::now() - HOSTED_DID_NEGATIVE_TTL - Duration::from_millis(1),
            });
        let result = store
            .resolve_tenant_for_hosted_did(&oauth_authority(), "did:web:new.acme.example")
            .await;
        assert_eq!(result.unwrap(), None);
    }

    #[tokio::test]
    async fn hosted_did_positive_binding_has_a_hard_stale_deadline() {
        let store = store();
        store
            .hosted_did_index
            .write()
            .await
            .replace(HostedDidIndex {
                entries: BTreeMap::from([(
                    ("alice".to_owned(), "did:web:alice.acme.example".to_owned()),
                    Some("acme".to_owned()),
                )]),
                built_at: Instant::now() - HOSTED_DID_INDEX_TTL - HOSTED_DID_MAX_STALE,
            });
        let error = store
            .resolve_tenant_for_hosted_did(&oauth_authority(), "did:web:alice.acme.example")
            .await
            .unwrap_err();
        assert!(matches!(error, AccountReadError::HostedDidIndexNotReady));
    }

    #[tokio::test]
    async fn hosted_service_auth_signing_is_authority_bound_and_record_bound() {
        use p256::ecdsa::signature::Verifier as _;

        let did = "did:web:alice.acme.example";
        let (record, key) = account_artifacts("alice", "acme.example");
        let verifying_key = p256::ecdsa::SigningKey::from_slice(&key)
            .unwrap()
            .verifying_key()
            .to_owned();
        let root = SyntheticNode::dir().with_child(
            "acme",
            tenant_node_with_key("alice", record.clone(), key.clone()),
        );
        let store =
            AccountRecordStore::new(Arc::new(SyntheticMount::new(root)), permit_account_reads());
        store
            .refresh_hosted_did_index(&oauth_authority())
            .await
            .expect("startup warm-up must complete before signing");
        let input = b"header.payload";
        let signature = store
            .sign_for_hosted_did(&oauth_authority(), did, input)
            .await
            .unwrap()
            .expect("hosted account key");
        let signature = p256::ecdsa::Signature::from_slice(&signature).unwrap();
        verifying_key.verify(input, &signature).unwrap();

        let denied = store
            .sign_for_hosted_did(&Subject::new("alice"), did, input)
            .await
            .unwrap_err();
        assert!(matches!(
            denied,
            AccountReadError::UnauthorizedTenantResolver(_)
        ));

        let missing_root =
            SyntheticNode::dir().with_child("acme", tenant_node("alice", record.clone()));
        let missing = AccountRecordStore::new(
            Arc::new(SyntheticMount::new(missing_root)),
            permit_account_reads(),
        );
        missing
            .refresh_hosted_did_index(&oauth_authority())
            .await
            .expect("startup warm-up must complete before signing");
        let missing = missing
        .sign_for_hosted_did(&oauth_authority(), did, input)
        .await
        .unwrap_err();
        assert!(matches!(
            missing,
            AccountReadError::SigningKeyUnavailable(_)
        ));

        let (_, other_key) = account_artifacts("other", "acme.example");
        let mismatch_root = SyntheticNode::dir()
            .with_child("acme", tenant_node_with_key("alice", record, other_key));
        let mismatch = AccountRecordStore::new(
            Arc::new(SyntheticMount::new(mismatch_root)),
            permit_account_reads(),
        );
        mismatch
            .refresh_hosted_did_index(&oauth_authority())
            .await
            .expect("startup warm-up must complete before signing");
        let mismatch = mismatch
        .sign_for_hosted_did(&oauth_authority(), did, input)
        .await
        .unwrap_err();
        assert!(matches!(mismatch, AccountReadError::SigningKeyMismatch(_)));
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
    async fn unauthenticated_and_path_injection_fail_before_mount_access() {
        let store = store();
        let signer = SigningKey::generate(&mut OsRng);
        let unauthenticated = EnvelopeContext::for_test_authenticated_subject_in_tenant(
            Subject::anonymous(),
            "acme",
            signer.verifying_key(),
        );
        let error = store
            .scope(&unauthenticated)
            .err()
            .expect("unauthenticated caller must be denied");
        assert!(matches!(error, AccountReadError::UnauthenticatedCaller));

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
        let store = AccountRecordStore::new(
            Arc::new(SyntheticMount::new(oversized)),
            permit_account_reads(),
        )
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
        let error = AccountRecordStore::new(
            Arc::new(SyntheticMount::new(mislabeled)),
            permit_account_reads(),
        )
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
