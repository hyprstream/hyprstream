//! Audited MAC adapter for the tenant account-record read store (#1319).
//!
//! `AccountRecordStore` operates directly on a subject-carrying `Mount`, below
//! the network 9P translator. This adapter supplies the equivalent mandatory
//! gate at that read boundary: authority-backed subject clearance, trusted
//! path-label resolution, lattice dominance, and the canonical MAC audit sink.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(not(target_arch = "wasm32"))]
use async_trait::async_trait;
use hyprstream_pds::ATPROTO_SIGNING_KEY_FILE;
use hyprstream_pds_service::{
    AccountRecordReadAuthorizer, OAUTH_ACCOUNT_RESOLVER_SUBJECT, PDS_ACCOUNTS_DIRECTORY,
    PDS_ACCOUNT_RECORD_FILE,
};
use hyprstream_rpc::auth::mac::{
    Assurance, CompartmentSet, Level, MacDecision, MacDenyReason, ObjectLabelResolver, ObjectRef,
    SecurityContext, SecurityLabel, VerifiedKeyMaterial,
};
use hyprstream_rpc::Subject;
#[cfg(not(target_arch = "wasm32"))]
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};
#[cfg(not(target_arch = "wasm32"))]
use parking_lot::Mutex;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::{Read, Seek, SeekFrom};
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

use crate::mac::audit::{AuditRecord, AuditSink, DecisionReason};
use crate::mac::te::{Action, Decision, ObjectType, ScopeAction, SubjectType};

/// Reserved audit identities for the PDS account-read enforcement plane.
const PDS_SUBJECT_TYPE: SubjectType = SubjectType(u32::MAX - 4);
const PDS_OBJECT_TYPE: ObjectType = ObjectType(u32::MAX - 4);

fn pds_account_label() -> SecurityLabel {
    SecurityLabel::new(
        Level::Confidential,
        Assurance::Classical,
        CompartmentSet::EMPTY,
    )
}

/// Trusted structural labels for the account-store read objects.
///
/// The OAuth resolver reads only `/pds` (tenant names), while a scoped caller
/// reads only the exact publication marker below a validated tenant and
/// account label. The account-specific `#atproto` secret is labeled for the
/// fixed internal OAuth authority's service-auth signer, but no scoped
/// account-reader API exposes it. Other paths remain unlabeled and deny.
#[derive(Debug, Default, Clone, Copy)]
pub struct PdsAccountObjectLabelResolver;

impl ObjectLabelResolver for PdsAccountObjectLabelResolver {
    fn resolve(&self, object: ObjectRef<'_>) -> Option<SecurityLabel> {
        let ObjectRef::Path(components) = object else {
            return None;
        };
        match components {
            ["pds"] => Some(pds_account_label()),
            ["pds", tenant, accounts, account, file]
                if valid_tenant_component(tenant)
                    && *accounts == PDS_ACCOUNTS_DIRECTORY
                    && valid_account_component(account)
                    && matches!(*file, PDS_ACCOUNT_RECORD_FILE | ATPROTO_SIGNING_KEY_FILE) =>
            {
                Some(pds_account_label())
            }
            _ => None,
        }
    }
}

fn valid_tenant_component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 253
        && value != "."
        && value != ".."
        && value != "*"
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

fn valid_account_component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 63
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        && !value.starts_with('-')
        && !value.ends_with('-')
}

/// Descriptor-bound, read-only production mount for `/pds`.
///
/// `walk` opens the named filesystem object without reading its contents and
/// stores that descriptor in the fid. `open`, `stat`, and `read` consume that
/// same descriptor, so a rename or symlink swap after authorization cannot
/// substitute a different record. On Linux, directory enumeration also uses
/// `/proc/self/fd/{fd}` and therefore stays bound to the walked directory.
#[cfg(not(target_arch = "wasm32"))]
pub struct PdsDirectoryMount {
    root: PathBuf,
    root_directory: File,
}

#[cfg(not(target_arch = "wasm32"))]
impl PdsDirectoryMount {
    pub fn open(root: impl AsRef<Path>) -> std::io::Result<Self> {
        #[cfg(not(target_os = "linux"))]
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "descriptor-bound PDS directory mount requires Linux /proc",
        ));

        #[cfg(target_os = "linux")]
        {
            std::fs::create_dir_all(root.as_ref())?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                std::fs::set_permissions(root.as_ref(), std::fs::Permissions::from_mode(0o700))?;
            }
            let root = std::fs::canonicalize(root)?;
            let root_directory = File::open(&root)?;
            Ok(Self {
                root,
                root_directory,
            })
        }
    }

    fn relative_path(&self, components: &[&str]) -> Result<String, MountError> {
        if components
            .iter()
            .any(|component| !valid_walk_component(component))
        {
            return Err(MountError::InvalidArgument(
                "PDS path contains an invalid component".to_owned(),
            ));
        }
        Ok(if components.is_empty() {
            ".".to_owned()
        } else {
            components.join("/")
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct PdsDirectoryFid {
    handle: File,
    opened_file: Mutex<Option<File>>,
    components: Vec<String>,
    is_dir: bool,
}

#[cfg(not(target_arch = "wasm32"))]
fn valid_walk_component(component: &str) -> bool {
    !component.is_empty()
        && component != "."
        && component != ".."
        && !component.contains(['/', '\\', '\0'])
}

#[cfg(not(target_arch = "wasm32"))]
fn mount_io_error(action: &str, path: &Path, error: std::io::Error) -> MountError {
    let detail = format!("{action} {path:?}: {error}");
    if matches!(
        error.raw_os_error(),
        Some(libc::ELOOP | libc::EXDEV | libc::ENOTDIR)
    ) {
        return MountError::PermissionDenied(detail);
    }
    match error.kind() {
        std::io::ErrorKind::NotFound => MountError::NotFound(detail),
        std::io::ErrorKind::PermissionDenied => MountError::PermissionDenied(detail),
        _ => MountError::Io(detail),
    }
}

#[cfg(target_os = "linux")]
fn open_beneath_without_symlinks(root: &File, relative: &str) -> std::io::Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd as _, FromRawFd as _};

    #[repr(C)]
    struct OpenHow {
        flags: u64,
        mode: u64,
        resolve: u64,
    }

    const RESOLVE_NO_MAGICLINKS: u64 = 0x02;
    const RESOLVE_NO_SYMLINKS: u64 = 0x04;
    const RESOLVE_BENEATH: u64 = 0x08;

    let relative = CString::new(relative)
        .map_err(|_| std::io::Error::from(std::io::ErrorKind::InvalidInput))?;
    let how = OpenHow {
        flags: (libc::O_PATH | libc::O_CLOEXEC) as u64,
        mode: 0,
        resolve: RESOLVE_BENEATH | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_SYMLINKS,
    };
    // SAFETY: `relative` and `how` remain valid for the duration of the
    // syscall; `root` owns a live directory fd; and a successful return is a
    // newly owned descriptor transferred exactly once into `File`.
    let descriptor = unsafe {
        libc::syscall(
            libc::SYS_openat2,
            root.as_raw_fd(),
            relative.as_ptr(),
            &raw const how,
            std::mem::size_of::<OpenHow>(),
        )
    };
    if descriptor < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: a successful `openat2` returns a fresh owned descriptor.
    Ok(unsafe { File::from_raw_fd(descriptor as libc::c_int) })
}

#[cfg(target_os = "linux")]
fn reopen_bound_handle(handle: &File, is_dir: bool) -> std::io::Result<File> {
    use std::os::fd::AsRawFd as _;
    use std::os::unix::fs::OpenOptionsExt as _;

    let mut options = std::fs::OpenOptions::new();
    options.read(true).custom_flags(
        libc::O_CLOEXEC | libc::O_NONBLOCK | if is_dir { libc::O_DIRECTORY } else { 0 },
    );
    options.open(format!("/proc/self/fd/{}", handle.as_raw_fd()))
}

#[cfg(not(target_arch = "wasm32"))]
fn fid_state(fid: &Fid) -> Result<&PdsDirectoryFid, MountError> {
    fid.downcast_ref::<PdsDirectoryFid>()
        .ok_or_else(|| MountError::InvalidArgument("PDS mount received a foreign fid".to_owned()))
}

#[cfg(not(target_arch = "wasm32"))]
fn fid_state_mut(fid: &mut Fid) -> Result<&mut PdsDirectoryFid, MountError> {
    fid.downcast_mut::<PdsDirectoryFid>()
        .ok_or_else(|| MountError::InvalidArgument("PDS mount received a foreign fid".to_owned()))
}

#[cfg(not(target_arch = "wasm32"))]
#[async_trait]
impl Mount for PdsDirectoryMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        let relative = self.relative_path(components)?;
        let display_path = self.root.join(&relative);
        #[cfg(not(target_os = "linux"))]
        return Err(MountError::NotSupported(
            "descriptor-bound PDS walk requires Linux openat2".to_owned(),
        ));
        #[cfg(target_os = "linux")]
        {
            let handle = open_beneath_without_symlinks(&self.root_directory, &relative)
                .map_err(|error| mount_io_error("resolve during walk", &display_path, error))?;
            let metadata = handle
                .metadata()
                .map_err(|error| mount_io_error("stat walked descriptor", &display_path, error))?;
            if !metadata.is_dir() && !metadata.is_file() {
                return Err(MountError::PermissionDenied(format!(
                    "PDS walk accepts only ordinary files and directories: {display_path:?}"
                )));
            }
            Ok(Fid::new(PdsDirectoryFid {
                handle,
                opened_file: Mutex::new(None),
                components: components
                    .iter()
                    .map(|component| (*component).to_owned())
                    .collect(),
                is_dir: metadata.is_dir(),
            }))
        }
    }

    async fn open(&self, fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
        if mode & 0x03 != 0 {
            return Err(MountError::PermissionDenied(
                "PDS account mount is read-only".to_owned(),
            ));
        }
        let state = fid_state_mut(fid)?;
        #[cfg(not(target_os = "linux"))]
        return Err(MountError::NotSupported(
            "descriptor-bound PDS open requires Linux /proc".to_owned(),
        ));
        #[cfg(target_os = "linux")]
        {
            let file = reopen_bound_handle(&state.handle, state.is_dir).map_err(|error| {
                MountError::Io(format!("open walked PDS descriptor for reading: {error}"))
            })?;
            *state.opened_file.lock() = Some(file);
        }
        Ok(())
    }

    async fn read(
        &self,
        fid: &Fid,
        offset: u64,
        count: u32,
        _caller: &Subject,
    ) -> Result<Vec<u8>, MountError> {
        let state = fid_state(fid)?;
        if state.is_dir {
            return Err(MountError::IsDirectory(state.components.join("/")));
        }
        let mut opened_file = state.opened_file.lock();
        let file = opened_file
            .as_mut()
            .ok_or_else(|| MountError::InvalidArgument("PDS fid is not open".to_owned()))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| MountError::Io(format!("seek PDS record: {error}")))?;
        let mut bytes = vec![0; count as usize];
        let read = file
            .read(&mut bytes)
            .map_err(|error| MountError::Io(format!("read PDS record: {error}")))?;
        bytes.truncate(read);
        Ok(bytes)
    }

    async fn write(
        &self,
        _fid: &Fid,
        _offset: u64,
        _data: &[u8],
        _caller: &Subject,
    ) -> Result<u32, MountError> {
        Err(MountError::PermissionDenied(
            "PDS account mount is read-only".to_owned(),
        ))
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let state = fid_state(fid)?;
        if !state.is_dir {
            return Err(MountError::NotDirectory(state.components.join("/")));
        }

        #[cfg(not(target_os = "linux"))]
        return Err(MountError::NotSupported(
            "descriptor-bound PDS readdir requires Linux /proc".to_owned(),
        ));

        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd as _;
            let opened_file = state.opened_file.lock();
            let file = opened_file
                .as_ref()
                .ok_or_else(|| MountError::InvalidArgument("PDS fid is not open".to_owned()))?;
            let descriptor = PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
            let entries = std::fs::read_dir(&descriptor)
                .map_err(|error| mount_io_error("read PDS directory", &descriptor, error))?;
            let mut result = Vec::new();
            for entry in entries {
                let entry = entry.map_err(|error| {
                    mount_io_error("read PDS directory entry", &descriptor, error)
                })?;
                let name = entry.file_name().into_string().map_err(|_| {
                    MountError::InvalidArgument(
                        "PDS directory contains a non-UTF-8 name".to_owned(),
                    )
                })?;
                let file_type = entry.file_type().map_err(|error| {
                    mount_io_error("type PDS directory entry", &entry.path(), error)
                })?;
                let metadata = std::fs::symlink_metadata(entry.path()).map_err(|error| {
                    mount_io_error("stat PDS directory entry", &entry.path(), error)
                })?;
                result.push(DirEntry {
                    name,
                    is_dir: file_type.is_dir(),
                    size: metadata.len(),
                    stat: None,
                });
            }
            result.sort_by(|left, right| left.name.cmp(&right.name));
            Ok(result)
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let state = fid_state(fid)?;
        let opened_file = state.opened_file.lock();
        let metadata = opened_file
            .as_ref()
            .ok_or_else(|| MountError::InvalidArgument("PDS fid is not open".to_owned()))?
            .metadata()
            .map_err(|error| MountError::Io(format!("stat PDS descriptor: {error}")))?;
        Ok(Stat::unknown_qid(
            if metadata.is_dir() { 0x80 } else { 0 },
            metadata.len(),
            state.components.last().cloned().unwrap_or_default(),
            0,
        ))
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

/// Resolve an authority-bound PDS reader to its verified MAC context.
///
/// The tenant input is derived by `AccountRecordStore` from the verified
/// envelope, or from the authority-owned hosted-account index entry while the
/// OAuth service resolves a DID. `None` must fail closed unless the
/// implementation explicitly recognizes the fixed OAuth service authority
/// performing root enumeration.
pub trait PdsClearanceSource: Send + Sync {
    fn clearance_for(
        &self,
        subject: &Subject,
        verified_tenant: Option<&str>,
        request_context: Option<&SecurityContext>,
    ) -> Option<SecurityContext>;
}

/// Production clearance source that preserves the verified request context.
///
/// Ordinary scoped reads use the exact `SecurityContext` derived from their
/// verified envelope, including signer assurance and delegated attenuation.
/// The fixed internal OAuth resolver receives only the narrow PDS lookup
/// clearance needed to discover an authority-owned tenant binding.
#[derive(Debug, Default, Clone, Copy)]
pub struct EnrollmentPdsClearanceSource;

impl PdsClearanceSource for EnrollmentPdsClearanceSource {
    fn clearance_for(
        &self,
        subject: &Subject,
        verified_tenant: Option<&str>,
        request_context: Option<&SecurityContext>,
    ) -> Option<SecurityContext> {
        let subject_id = subject.name()?;
        if subject_id == OAUTH_ACCOUNT_RESOLVER_SUBJECT {
            return Some(SecurityContext::from_clearance(
                pds_account_label(),
                VerifiedKeyMaterial::Classical,
            ));
        }
        verified_tenant?;
        request_context.cloned()
    }
}

/// PDS account-record PEP over the trusted object-label resolver and audit WAL.
pub struct PdsAccountReadPep {
    clearance: Arc<dyn PdsClearanceSource>,
    labels: Arc<dyn ObjectLabelResolver + Send + Sync>,
    sink: Arc<dyn AuditSink>,
}

impl PdsAccountReadPep {
    pub fn new(
        clearance: Arc<dyn PdsClearanceSource>,
        labels: Arc<dyn ObjectLabelResolver + Send + Sync>,
        sink: Arc<dyn AuditSink>,
    ) -> Self {
        Self {
            clearance,
            labels,
            sink,
        }
    }

    fn check(
        &self,
        subject: &Subject,
        verified_tenant: Option<&str>,
        request_context: Option<&SecurityContext>,
        object_id: &str,
    ) -> MacDecision {
        let components: Vec<&str> = object_id
            .split('/')
            .filter(|component| !component.is_empty())
            .collect();
        let label = self.labels.resolve(ObjectRef::Path(&components));
        let Some(ctx) = self
            .clearance
            .clearance_for(subject, verified_tenant, request_context)
        else {
            return self.audit(
                subject,
                object_id,
                None,
                label,
                MacDecision::Deny(MacDenyReason::NoClearance),
            );
        };
        let Some(label) = label else {
            return self.audit(
                subject,
                object_id,
                Some(&ctx),
                None,
                MacDecision::Deny(MacDenyReason::UnlabeledObject),
            );
        };

        let decision = if ctx.can_access(&label) {
            MacDecision::Permit
        } else {
            MacDecision::Deny(MacDenyReason::FloorDeny)
        };
        self.audit(subject, object_id, Some(&ctx), Some(label), decision)
    }

    fn audit(
        &self,
        subject: &Subject,
        object_id: &str,
        ctx: Option<&SecurityContext>,
        label: Option<SecurityLabel>,
        decision: MacDecision,
    ) -> MacDecision {
        let (audit_decision, reason) = match decision {
            MacDecision::Permit => (Decision::Permit, DecisionReason::Permit),
            MacDecision::Deny(MacDenyReason::NoClearance) => {
                (Decision::Deny, DecisionReason::NoClearance)
            }
            MacDecision::Deny(MacDenyReason::UnlabeledObject) => {
                (Decision::Deny, DecisionReason::UnlabeledObject)
            }
            MacDecision::Deny(
                MacDenyReason::FloorDeny
                | MacDenyReason::NoPepInstalled
                | MacDenyReason::StaleAuthority,
            ) => (Decision::Deny, DecisionReason::FloorDeny),
        };
        let policy = crate::mac::compiled_policy();
        let record = AuditRecord {
            seq: 0,
            prev_hash: [0; 32],
            ts_unix_nanos: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |duration| duration.as_nanos()),
            decision: audit_decision,
            generation: policy.as_ref().map_or(0, |policy| policy.generation),
            policy_hash: policy.as_ref().and_then(|policy| policy.policy_hash().ok()),
            subject_type: PDS_SUBJECT_TYPE,
            subject_clearance: ctx
                .map(|context| *context.clearance())
                .unwrap_or_else(SecurityLabel::bottom),
            on_behalf_of: None,
            object_type: PDS_OBJECT_TYPE,
            object_label: label.unwrap_or_else(SecurityLabel::bottom),
            action: Action::from_scope_action(ScopeAction::Query),
            reason,
            subject_id: Some(subject.to_string()),
            object_id: Some(object_id.to_owned()),
        };

        match self.sink.record(&record) {
            Ok(()) => decision,
            Err(error) => {
                let deny_record = AuditRecord {
                    decision: Decision::Deny,
                    reason: DecisionReason::AuditFailClosed,
                    ..record
                };
                let _ = self.sink.record(&deny_record);
                tracing::error!(
                    target: "hyprstream.mac.pds_pep",
                    %error,
                    "PDS account-read decision could not be durably audited; enforcing deny"
                );
                MacDecision::Deny(MacDenyReason::FloorDeny)
            }
        }
    }
}

impl AccountRecordReadAuthorizer for PdsAccountReadPep {
    fn check_read(
        &self,
        subject: &Subject,
        verified_tenant: Option<&str>,
        security_context: Option<&SecurityContext>,
        object_id: &str,
    ) -> MacDecision {
        self.check(subject, verified_tenant, security_context, object_id)
    }
}

/// Assemble the production account-read adapter from verified request
/// contexts, trusted PDS structural labels, and the mandatory audit sink.
pub fn production_pds_account_read_authorizer(
    sink: Arc<dyn AuditSink>,
) -> Arc<dyn AccountRecordReadAuthorizer> {
    Arc::new(PdsAccountReadPep::new(
        Arc::new(EnrollmentPdsClearanceSource),
        Arc::new(PdsAccountObjectLabelResolver),
        sink,
    ))
}

/// Production composition for a published account tree: a descriptor-bound
/// mount plus the real PDS PEP and mandatory audit sink.
///
/// The caller supplies the authoritative publication root. Hyprstream does not
/// manufacture a second empty tree when no durable hosted-account publisher is
/// configured; in that case OAuth has no account store and performs no record
/// read. Once installed, there is no constructor path around this PEP.
#[cfg(not(target_arch = "wasm32"))]
pub fn production_pds_account_record_store(
    mount: Arc<PdsDirectoryMount>,
    sink: Arc<dyn AuditSink>,
) -> Arc<hyprstream_pds_service::AccountRecordStore> {
    Arc::new(hyprstream_pds_service::AccountRecordStore::new(
        mount,
        production_pds_account_read_authorizer(sink),
    ))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use ed25519_dalek::SigningKey;
    use parking_lot::Mutex;
    use rand::rngs::OsRng;

    use hyprstream_pds_service::{AccountReadError, AccountRecordStore};
    use hyprstream_rpc::auth::mac::{Assurance, CompartmentSet, Level};
    use hyprstream_rpc::EnvelopeContext;
    use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat};

    use super::*;
    use crate::mac::audit::{AuditError, AuditRecord};

    #[derive(Default)]
    struct SpySink {
        records: Mutex<Vec<AuditRecord>>,
    }

    impl AuditSink for SpySink {
        fn record(&self, record: &AuditRecord) -> Result<(), AuditError> {
            self.records.lock().push(record.clone());
            Ok(())
        }
    }

    struct PublicReaders;

    impl PdsClearanceSource for PublicReaders {
        fn clearance_for(
            &self,
            subject: &Subject,
            verified_tenant: Option<&str>,
            _request_context: Option<&SecurityContext>,
        ) -> Option<SecurityContext> {
            ((subject.name() == Some("alice") && verified_tenant == Some("acme"))
                || (subject.name() == Some(OAUTH_ACCOUNT_RESOLVER_SUBJECT)
                    && verified_tenant.is_none()))
            .then(|| {
                SecurityContext::from_clearance(
                    label(Level::Public),
                    VerifiedKeyMaterial::Classical,
                )
            })
        }
    }

    struct ConfidentialReaders;

    impl PdsClearanceSource for ConfidentialReaders {
        fn clearance_for(
            &self,
            subject: &Subject,
            verified_tenant: Option<&str>,
            _request_context: Option<&SecurityContext>,
        ) -> Option<SecurityContext> {
            (subject.name() == Some("alice") && verified_tenant == Some("acme")).then(|| {
                SecurityContext::from_clearance(
                    label(Level::Confidential),
                    VerifiedKeyMaterial::Classical,
                )
            })
        }
    }

    struct NoClearance;

    impl PdsClearanceSource for NoClearance {
        fn clearance_for(
            &self,
            _subject: &Subject,
            _verified_tenant: Option<&str>,
            _request_context: Option<&SecurityContext>,
        ) -> Option<SecurityContext> {
            None
        }
    }

    struct NoLabels;

    impl ObjectLabelResolver for NoLabels {
        fn resolve(&self, _object: ObjectRef<'_>) -> Option<SecurityLabel> {
            None
        }
    }

    #[derive(Default)]
    struct FailingSink {
        attempts: AtomicUsize,
        records: Mutex<Vec<AuditRecord>>,
    }

    impl AuditSink for FailingSink {
        fn record(&self, record: &AuditRecord) -> Result<(), AuditError> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            self.records.lock().push(record.clone());
            Err(AuditError::Io("injected PDS audit failure".to_owned()))
        }
    }

    struct AccountLabels;

    impl ObjectLabelResolver for AccountLabels {
        fn resolve(&self, object: ObjectRef<'_>) -> Option<SecurityLabel> {
            match object {
                ObjectRef::Path(["pds", "acme", "accounts", "alice", "account-record.cbor"]) => {
                    Some(label(Level::Secret))
                }
                _ => None,
            }
        }
    }

    #[derive(Default)]
    struct ReadTrapMount {
        walks: AtomicUsize,
        opens: AtomicUsize,
        stats: AtomicUsize,
        reads: AtomicUsize,
        readdirs: AtomicUsize,
    }

    #[async_trait]
    impl Mount for ReadTrapMount {
        async fn walk(&self, _components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
            self.walks.fetch_add(1, Ordering::SeqCst);
            Ok(Fid::new(()))
        }

        async fn open(
            &self,
            _fid: &mut Fid,
            _mode: u8,
            _caller: &Subject,
        ) -> Result<(), MountError> {
            self.opens.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn read(
            &self,
            _fid: &Fid,
            _offset: u64,
            _count: u32,
            _caller: &Subject,
        ) -> Result<Vec<u8>, MountError> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            Ok(b"record bytes must remain unreachable".to_vec())
        }

        async fn write(
            &self,
            _fid: &Fid,
            _offset: u64,
            _data: &[u8],
            _caller: &Subject,
        ) -> Result<u32, MountError> {
            Err(MountError::NotSupported("read-only test mount".to_owned()))
        }

        async fn readdir(
            &self,
            _fid: &Fid,
            _caller: &Subject,
        ) -> Result<Vec<DirEntry>, MountError> {
            self.readdirs.fetch_add(1, Ordering::SeqCst);
            Ok(vec![DirEntry {
                name: "tenant-bytes-must-remain-unreachable".to_owned(),
                is_dir: true,
                size: 0,
                stat: None,
            }])
        }

        async fn stat(&self, _fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
            self.stats.fetch_add(1, Ordering::SeqCst);
            Ok(Stat::unknown_qid(
                0,
                38,
                "account-record.cbor".to_owned(),
                0,
            ))
        }

        async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
    }

    fn label(level: Level) -> SecurityLabel {
        SecurityLabel::new(level, Assurance::Classical, CompartmentSet::EMPTY)
    }

    #[tokio::test]
    async fn account_store_denies_before_record_bytes_and_audits_identity() {
        let mount = Arc::new(ReadTrapMount::default());
        let sink = Arc::new(SpySink::default());
        let authorizer = Arc::new(PdsAccountReadPep::new(
            Arc::new(PublicReaders),
            Arc::new(AccountLabels),
            sink.clone(),
        ));
        let store = AccountRecordStore::new(mount.clone(), authorizer);
        let signer = SigningKey::generate(&mut OsRng);
        let context = EnvelopeContext::for_test_authenticated_subject_in_tenant(
            Subject::new("alice"),
            "acme",
            signer.verifying_key(),
        );

        let error = store
            .scope(&context)
            .unwrap()
            .get("alice")
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            AccountReadError::MacDenied {
                reason: MacDenyReason::FloorDeny,
                ..
            }
        ));

        assert_eq!(mount.walks.load(Ordering::SeqCst), 1);
        assert_eq!(mount.opens.load(Ordering::SeqCst), 0);
        assert_eq!(mount.stats.load(Ordering::SeqCst), 0);
        assert_eq!(mount.reads.load(Ordering::SeqCst), 0);
        assert_eq!(mount.readdirs.load(Ordering::SeqCst), 0);

        let records = sink.records.lock();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].decision, Decision::Deny);
        assert_eq!(records[0].reason, DecisionReason::FloorDeny);
        assert_eq!(records[0].subject_id.as_deref(), Some("alice"));
        assert_eq!(
            records[0].object_id.as_deref(),
            Some("/pds/acme/accounts/alice/account-record.cbor")
        );
        assert_eq!(records[0].subject_clearance, label(Level::Public));
        assert_eq!(records[0].object_label, label(Level::Secret));
    }

    #[tokio::test]
    async fn hosted_directory_deny_precedes_open_and_readdir_and_is_audited() {
        let mount = Arc::new(ReadTrapMount::default());
        let sink = Arc::new(SpySink::default());
        let authorizer = Arc::new(PdsAccountReadPep::new(
            Arc::new(PublicReaders),
            Arc::new(PdsAccountObjectLabelResolver),
            sink.clone(),
        ));
        let store = AccountRecordStore::new(mount.clone(), authorizer);

        let error = store
            .resolve_tenant_for_hosted_did(
                &Subject::new(OAUTH_ACCOUNT_RESOLVER_SUBJECT),
                "did:web:alice.example.test",
            )
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            AccountReadError::MacDenied {
                reason: MacDenyReason::FloorDeny,
                ..
            }
        ));
        assert_eq!(mount.walks.load(Ordering::SeqCst), 1);
        assert_eq!(mount.opens.load(Ordering::SeqCst), 0);
        assert_eq!(mount.readdirs.load(Ordering::SeqCst), 0);
        assert_eq!(mount.stats.load(Ordering::SeqCst), 0);
        assert_eq!(mount.reads.load(Ordering::SeqCst), 0);

        let records = sink.records.lock();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].decision, Decision::Deny);
        assert_eq!(records[0].reason, DecisionReason::FloorDeny);
        assert_eq!(
            records[0].subject_id.as_deref(),
            Some(OAUTH_ACCOUNT_RESOLVER_SUBJECT)
        );
        assert_eq!(records[0].object_id.as_deref(), Some("/pds"));
    }

    #[tokio::test]
    async fn audit_failure_downgrades_permit_before_record_open() {
        let mount = Arc::new(ReadTrapMount::default());
        let sink = Arc::new(FailingSink::default());
        let authorizer = Arc::new(PdsAccountReadPep::new(
            Arc::new(ConfidentialReaders),
            Arc::new(PdsAccountObjectLabelResolver),
            sink.clone(),
        ));
        let store = AccountRecordStore::new(mount.clone(), authorizer);
        let signer = SigningKey::generate(&mut OsRng);
        let context = EnvelopeContext::for_test_authenticated_subject_in_tenant(
            Subject::new("alice"),
            "acme",
            signer.verifying_key(),
        );

        let error = store
            .scope(&context)
            .unwrap()
            .get("alice")
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            AccountReadError::MacDenied {
                reason: MacDenyReason::FloorDeny,
                ..
            }
        ));
        assert_eq!(mount.walks.load(Ordering::SeqCst), 1);
        assert_eq!(mount.opens.load(Ordering::SeqCst), 0);
        assert_eq!(mount.stats.load(Ordering::SeqCst), 0);
        assert_eq!(mount.reads.load(Ordering::SeqCst), 0);
        assert_eq!(sink.attempts.load(Ordering::SeqCst), 2);
        let records = sink.records.lock();
        assert_eq!(records[0].decision, Decision::Permit);
        assert_eq!(records[1].decision, Decision::Deny);
        assert_eq!(records[1].reason, DecisionReason::AuditFailClosed);
        assert_eq!(records[1].subject_id.as_deref(), Some("alice"));
        assert_eq!(
            records[1].object_id.as_deref(),
            Some("/pds/acme/accounts/alice/account-record.cbor")
        );
    }

    #[test]
    fn missing_clearance_and_label_denials_are_audited_with_identities() {
        let sink = Arc::new(SpySink::default());
        let no_clearance = PdsAccountReadPep::new(
            Arc::new(NoClearance),
            Arc::new(PdsAccountObjectLabelResolver),
            sink.clone(),
        );
        let object = "/pds/acme/accounts/alice/account-record.cbor";
        assert_eq!(
            no_clearance.check_read(&Subject::new("alice"), Some("acme"), None, object),
            MacDecision::Deny(MacDenyReason::NoClearance)
        );

        let no_label = PdsAccountReadPep::new(
            Arc::new(ConfidentialReaders),
            Arc::new(NoLabels),
            sink.clone(),
        );
        assert_eq!(
            no_label.check_read(&Subject::new("alice"), Some("acme"), None, object),
            MacDecision::Deny(MacDenyReason::UnlabeledObject)
        );

        let records = sink.records.lock();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].reason, DecisionReason::NoClearance);
        assert_eq!(records[1].reason, DecisionReason::UnlabeledObject);
        for record in records.iter() {
            assert_eq!(record.decision, Decision::Deny);
            assert_eq!(record.subject_id.as_deref(), Some("alice"));
            assert_eq!(record.object_id.as_deref(), Some(object));
        }
    }

    #[test]
    fn production_sources_label_only_account_artifacts_and_recognize_oauth() {
        let resolver = PdsAccountObjectLabelResolver;
        assert_eq!(
            resolver.resolve(ObjectRef::Path(&["pds"])),
            Some(pds_account_label())
        );
        assert_eq!(
            resolver.resolve(ObjectRef::Path(&[
                "pds",
                "acme",
                PDS_ACCOUNTS_DIRECTORY,
                "alice",
                PDS_ACCOUNT_RECORD_FILE,
            ])),
            Some(pds_account_label())
        );
        assert_eq!(
            resolver.resolve(ObjectRef::Path(&[
                "pds",
                "acme",
                PDS_ACCOUNTS_DIRECTORY,
                "alice",
                ATPROTO_SIGNING_KEY_FILE,
            ])),
            Some(pds_account_label())
        );
        assert_eq!(
            resolver.resolve(ObjectRef::Path(&["pds", "acme", "private-key"])),
            None
        );

        let source = EnrollmentPdsClearanceSource;
        let oauth = source
            .clearance_for(&Subject::new(OAUTH_ACCOUNT_RESOLVER_SUBJECT), None, None)
            .unwrap();
        assert!(oauth.can_access(&pds_account_label()));
        assert!(source
            .clearance_for(&Subject::new("alice"), None, None)
            .is_none());

        let attenuated =
            SecurityContext::from_clearance(label(Level::Public), VerifiedKeyMaterial::Classical);
        assert_eq!(
            source
                .clearance_for(&Subject::new("alice"), Some("acme"), Some(&attenuated))
                .unwrap(),
            attenuated
        );
    }

    #[tokio::test]
    async fn production_directory_mount_keeps_the_walked_record_descriptor_bound() {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("pds");
        let record_dir = root.join("acme/accounts/alice");
        std::fs::create_dir_all(&record_dir).unwrap();
        let record = record_dir.join(PDS_ACCOUNT_RECORD_FILE);
        std::fs::write(&record, b"original").unwrap();
        let mount = PdsDirectoryMount::open(&root).unwrap();
        let subject = Subject::new("alice");
        let components = [
            "acme",
            PDS_ACCOUNTS_DIRECTORY,
            "alice",
            PDS_ACCOUNT_RECORD_FILE,
        ];
        let mut fid = mount.walk(&components, &subject).await.unwrap();

        let replaced = record_dir.join("replaced.cbor");
        std::fs::rename(&record, &replaced).unwrap();
        std::fs::write(&record, b"substitute").unwrap();

        mount
            .open(&mut fid, hyprstream_vfs::OREAD, &subject)
            .await
            .unwrap();
        assert_eq!(
            mount.read(&fid, 0, 64, &subject).await.unwrap(),
            b"original"
        );
        mount.clunk(fid, &subject).await;
    }

    #[tokio::test]
    async fn production_directory_mount_rejects_final_and_intermediate_symlinks() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("pds");
        let acme = root.join("acme/accounts/alice");
        let beta = root.join("beta/accounts/alice");
        std::fs::create_dir_all(&acme).unwrap();
        std::fs::create_dir_all(&beta).unwrap();
        let beta_record = beta.join(PDS_ACCOUNT_RECORD_FILE);
        std::fs::write(&beta_record, b"beta").unwrap();

        let final_link = acme.join(PDS_ACCOUNT_RECORD_FILE);
        symlink(&beta_record, &final_link).unwrap();
        let mount = PdsDirectoryMount::open(&root).unwrap();
        let subject = Subject::new("alice");
        let final_components = [
            "acme",
            PDS_ACCOUNTS_DIRECTORY,
            "alice",
            PDS_ACCOUNT_RECORD_FILE,
        ];
        assert!(matches!(
            mount.walk(&final_components, &subject).await,
            Err(MountError::PermissionDenied(_))
        ));

        std::fs::remove_file(&final_link).unwrap();
        std::fs::remove_dir_all(root.join("acme/accounts")).unwrap();
        symlink(root.join("beta/accounts"), root.join("acme/accounts")).unwrap();
        assert!(matches!(
            mount.walk(&final_components, &subject).await,
            Err(MountError::PermissionDenied(_))
        ));
    }

    #[tokio::test]
    async fn production_directory_mount_enumerates_only_real_root_directories() {
        use std::os::unix::fs::symlink;

        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path().join("pds");
        std::fs::create_dir_all(root.join("acme")).unwrap();
        symlink(root.join("acme"), root.join("alias")).unwrap();
        let mount = PdsDirectoryMount::open(&root).unwrap();
        let subject = Subject::new(OAUTH_ACCOUNT_RESOLVER_SUBJECT);
        let mut fid = mount.walk(&[], &subject).await.unwrap();
        mount
            .open(&mut fid, hyprstream_vfs::OREAD, &subject)
            .await
            .unwrap();
        let entries = mount.readdir(&fid, &subject).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert!(entries
            .iter()
            .any(|entry| entry.name == "acme" && entry.is_dir));
        assert!(entries
            .iter()
            .any(|entry| entry.name == "alias" && !entry.is_dir));
        mount.clunk(fid, &subject).await;
    }
}
