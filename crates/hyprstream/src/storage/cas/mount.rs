//! Subject-threaded 9P/VFS projection of the L1 CAS substrate (#813).
//!
//! The mount exposes CAS reads through the ordinary `Mount` surface, so access
//! goes through open/read/stat authorization instead of treating hashes as
//! bearer capabilities. A namespace may bind this mount wherever it chooses.

use async_trait::async_trait;
use cas_serve::StoreError;
use hyprstream_rpc::Subject;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat, OREAD};

use super::{CasError, CasSubstrate, DedupDomain};

const QTDIR: u8 = 0x80;
const QTFILE: u8 = 0x00;

/// Object class addressed through [`CasMount`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CasMountObjectKind {
    /// Full-file reconstruction by CID or legacy merkle.
    Object,
    /// Raw xorb bytes by xorb hash.
    Xorb,
}

/// Authorization request made before a CAS object is opened/read/stat'ed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CasMountAuthzRequest<'a> {
    /// Which CAS object namespace is being accessed.
    pub kind: CasMountObjectKind,
    /// CID, legacy merkle, or xorb hash, exactly as walked by the caller.
    pub address: &'a str,
    /// Dedup domain the mount is serving.
    pub domain: &'a DedupDomain,
    /// Operation name for policy/audit: currently `open`, `read`, or `stat`.
    pub operation: &'static str,
}

/// Per-op authorization hook for [`CasMount`].
///
/// The default constructor uses [`DenyAllCasAuthorizer`]. Production namespaces
/// must inject a real authorizer from #699/#767 provenance and MAC plumbing;
/// trusted local single-tenant tools can explicitly opt into
/// [`AllowAllCasAuthorizer`].
pub trait CasMountAuthorizer: Send + Sync {
    fn authorize(
        &self,
        caller: &Subject,
        request: CasMountAuthzRequest<'_>,
    ) -> Result<(), MountError>;
}

/// Fail-closed authorizer used by [`CasMount::new`].
#[derive(Debug, Default)]
pub struct DenyAllCasAuthorizer;

impl CasMountAuthorizer for DenyAllCasAuthorizer {
    fn authorize(
        &self,
        caller: &Subject,
        request: CasMountAuthzRequest<'_>,
    ) -> Result<(), MountError> {
        Err(MountError::PermissionDenied(format!(
            "CAS {} {} denied for {}: no CasMountAuthorizer installed",
            request.operation, request.address, caller
        )))
    }
}

/// Explicit allow authorizer for tests and trusted local single-tenant wiring.
#[derive(Debug, Default)]
pub struct AllowAllCasAuthorizer;

impl CasMountAuthorizer for AllowAllCasAuthorizer {
    fn authorize(
        &self,
        _caller: &Subject,
        _request: CasMountAuthzRequest<'_>,
    ) -> Result<(), MountError> {
        Ok(())
    }
}

/// CAS as a namespace mount.
///
/// Relative layout:
/// - `obj/{cid-or-legacy-merkle}` — full-file reconstruction.
/// - `xorb/{hash}` — raw xorb compatibility read.
/// - `stage/` — reserved for #814 write-then-seal ingest.
#[derive(Clone)]
pub struct CasMount {
    substrate: CasSubstrate,
    domain: DedupDomain,
    authorizer: std::sync::Arc<dyn CasMountAuthorizer>,
}

impl std::fmt::Debug for CasMount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CasMount")
            .field("domain", &self.domain)
            .finish_non_exhaustive()
    }
}

impl CasMount {
    /// Construct a fail-closed CAS mount over the given substrate/domain.
    pub fn new(substrate: CasSubstrate, domain: DedupDomain) -> Self {
        Self::with_authorizer(substrate, domain, DenyAllCasAuthorizer)
    }

    /// Construct a fail-closed CAS mount over the default local dedup domain.
    pub fn local_default(substrate: CasSubstrate) -> Self {
        Self::new(substrate, DedupDomain::local_default())
    }

    /// Construct a CAS mount with an explicit authorizer.
    pub fn with_authorizer<A>(substrate: CasSubstrate, domain: DedupDomain, authorizer: A) -> Self
    where
        A: CasMountAuthorizer + 'static,
    {
        Self {
            substrate,
            domain,
            authorizer: std::sync::Arc::new(authorizer),
        }
    }

    fn authorize(
        &self,
        caller: &Subject,
        kind: CasMountObjectKind,
        address: &str,
        operation: &'static str,
    ) -> Result<(), MountError> {
        self.authorizer.authorize(
            caller,
            CasMountAuthzRequest {
                kind,
                address,
                domain: &self.domain,
                operation,
            },
        )
    }

    async fn read_all(
        &self,
        kind: CasMountObjectKind,
        address: &str,
    ) -> Result<Vec<u8>, MountError> {
        match kind {
            CasMountObjectKind::Object => self.substrate.get(&self.domain, address).await,
            CasMountObjectKind::Xorb => self.substrate.read_xorb(&self.domain, address).await,
        }
        .map_err(map_cas_error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CasFid {
    Root,
    ObjDir,
    XorbDir,
    StageDir,
    File {
        kind: CasMountObjectKind,
        address: String,
        opened: bool,
    },
}

#[async_trait]
impl Mount for CasMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        let fid = match components {
            [] => CasFid::Root,
            ["obj"] => CasFid::ObjDir,
            ["xorb"] => CasFid::XorbDir,
            ["stage"] => CasFid::StageDir,
            ["obj", address] if !address.is_empty() => CasFid::File {
                kind: CasMountObjectKind::Object,
                address: (*address).to_owned(),
                opened: false,
            },
            ["xorb", hash] if !hash.is_empty() => CasFid::File {
                kind: CasMountObjectKind::Xorb,
                address: (*hash).to_owned(),
                opened: false,
            },
            _ => return Err(MountError::NotFound(components.join("/"))),
        };
        Ok(Fid::new(fid))
    }

    async fn open(&self, fid: &mut Fid, mode: u8, caller: &Subject) -> Result<(), MountError> {
        let inner = fid
            .downcast_mut::<CasFid>()
            .ok_or_else(|| MountError::InvalidArgument("wrong fid type for CasMount".into()))?;

        if mode & 0x03 != OREAD {
            return Err(MountError::PermissionDenied(
                "CAS mount is read-only until #814 seal ingest".into(),
            ));
        }

        match inner {
            CasFid::Root | CasFid::ObjDir | CasFid::XorbDir | CasFid::StageDir => Ok(()),
            CasFid::File {
                kind,
                address,
                opened,
            } => {
                self.authorize(caller, *kind, address, "open")?;
                *opened = true;
                Ok(())
            }
        }
    }

    async fn read(
        &self,
        fid: &Fid,
        offset: u64,
        count: u32,
        caller: &Subject,
    ) -> Result<Vec<u8>, MountError> {
        let inner = fid
            .downcast_ref::<CasFid>()
            .ok_or_else(|| MountError::InvalidArgument("wrong fid type for CasMount".into()))?;
        let CasFid::File {
            kind,
            address,
            opened,
        } = inner
        else {
            return Err(MountError::IsDirectory(
                "cannot read a CAS directory as a file".into(),
            ));
        };
        if !opened {
            return Err(MountError::InvalidArgument("fid is not open".into()));
        }

        self.authorize(caller, *kind, address, "read")?;
        let bytes = self.read_all(*kind, address).await?;
        Ok(slice_bytes(&bytes, offset, count))
    }

    async fn write(
        &self,
        _fid: &Fid,
        _offset: u64,
        _data: &[u8],
        _caller: &Subject,
    ) -> Result<u32, MountError> {
        Err(MountError::NotSupported(
            "CAS mount writes require #814 write-then-seal ingest".into(),
        ))
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let inner = fid
            .downcast_ref::<CasFid>()
            .ok_or_else(|| MountError::InvalidArgument("wrong fid type for CasMount".into()))?;
        match inner {
            CasFid::Root => Ok(vec![
                dir_entry("obj"),
                dir_entry("xorb"),
                dir_entry("stage"),
            ]),
            CasFid::ObjDir | CasFid::XorbDir | CasFid::StageDir => Ok(Vec::new()),
            CasFid::File { address, .. } => Err(MountError::NotDirectory(address.clone())),
        }
    }

    async fn stat(&self, fid: &Fid, caller: &Subject) -> Result<Stat, MountError> {
        let inner = fid
            .downcast_ref::<CasFid>()
            .ok_or_else(|| MountError::InvalidArgument("wrong fid type for CasMount".into()))?;
        match inner {
            CasFid::Root => Ok(dir_stat("", 0)),
            CasFid::ObjDir => Ok(dir_stat("obj", 1)),
            CasFid::XorbDir => Ok(dir_stat("xorb", 2)),
            CasFid::StageDir => Ok(dir_stat("stage", 3)),
            CasFid::File { kind, address, .. } => {
                self.authorize(caller, *kind, address, "stat")?;
                let len = self.read_all(*kind, address).await?.len() as u64;
                Ok(file_stat(address, len))
            }
        }
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

fn dir_entry(name: &str) -> DirEntry {
    DirEntry {
        name: name.to_owned(),
        is_dir: true,
        size: 0,
        stat: Some(dir_stat(name, hash_path(name))),
    }
}

fn dir_stat(name: &str, path: u64) -> Stat {
    Stat {
        qtype: QTDIR,
        version: 1,
        path,
        size: 0,
        name: name.to_owned(),
        mtime: 0,
    }
}

fn file_stat(name: &str, size: u64) -> Stat {
    Stat {
        qtype: QTFILE,
        version: 1,
        path: hash_path(name),
        size,
        name: name.to_owned(),
        mtime: 0,
    }
}

fn slice_bytes(bytes: &[u8], offset: u64, count: u32) -> Vec<u8> {
    let start = usize::try_from(offset)
        .unwrap_or(usize::MAX)
        .min(bytes.len());
    let end = start.saturating_add(count as usize).min(bytes.len());
    bytes[start..end].to_vec()
}

fn map_cas_error(err: CasError) -> MountError {
    match err {
        CasError::Store(StoreError::NotFound(s)) => MountError::NotFound(s),
        CasError::Store(StoreError::InvalidHash(s)) | CasError::Cid(s) | CasError::Hex(s) => {
            MountError::InvalidArgument(s)
        }
        CasError::Store(e) => MountError::Io(e.to_string()),
        CasError::UnsupportedIngestAlgorithm(algo) => {
            MountError::InvalidArgument(format!("unsupported CAS algorithm: {algo:?}"))
        }
    }
}

fn hash_path(path: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in path.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    #[derive(Default)]
    struct RecordingAuthorizer {
        deny: bool,
        calls: Mutex<Vec<(String, CasMountObjectKind, String, &'static str)>>,
    }

    impl RecordingAuthorizer {
        fn calls(&self) -> Vec<(String, CasMountObjectKind, String, &'static str)> {
            self.calls.lock().clone()
        }
    }

    impl CasMountAuthorizer for std::sync::Arc<RecordingAuthorizer> {
        fn authorize(
            &self,
            caller: &Subject,
            request: CasMountAuthzRequest<'_>,
        ) -> Result<(), MountError> {
            self.calls.lock().push((
                caller.to_string(),
                request.kind,
                request.address.to_owned(),
                request.operation,
            ));
            if self.deny {
                Err(MountError::PermissionDenied("denied by test".into()))
            } else {
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn object_read_requires_authorizer_and_slices() {
        let dir = tempfile::tempdir().unwrap();
        let substrate = CasSubstrate::new(dir.path());
        let domain = DedupDomain::local_default();
        let manifest = substrate
            .put(&domain, b"hello-cas-mount", None)
            .await
            .unwrap();
        let authz = std::sync::Arc::new(RecordingAuthorizer::default());
        let mount = CasMount::with_authorizer(substrate, domain, authz.clone());
        let caller = Subject::new("alice");

        let mut fid = mount.walk(&["obj", &manifest.cid], &caller).await.unwrap();
        mount.open(&mut fid, OREAD, &caller).await.unwrap();
        let out = mount.read(&fid, 6, 3, &caller).await.unwrap();

        assert_eq!(out, b"cas");
        assert_eq!(
            authz.calls(),
            vec![
                (
                    "alice".to_owned(),
                    CasMountObjectKind::Object,
                    manifest.cid.clone(),
                    "open",
                ),
                (
                    "alice".to_owned(),
                    CasMountObjectKind::Object,
                    manifest.cid,
                    "read",
                ),
            ]
        );
    }

    #[tokio::test]
    async fn default_authorizer_denies_even_when_hash_exists() {
        let dir = tempfile::tempdir().unwrap();
        let substrate = CasSubstrate::new(dir.path());
        let domain = DedupDomain::local_default();
        let manifest = substrate.put(&domain, b"secret bytes", None).await.unwrap();
        let mount = CasMount::new(substrate, domain);
        let caller = Subject::new("bob");

        let mut fid = mount
            .walk(&["obj", &manifest.merkle], &caller)
            .await
            .unwrap();
        let err = mount.open(&mut fid, OREAD, &caller).await.unwrap_err();

        assert!(matches!(err, MountError::PermissionDenied(_)));
    }

    #[tokio::test]
    async fn xorb_read_uses_xorb_authorization_class() {
        let dir = tempfile::tempdir().unwrap();
        let substrate = CasSubstrate::new(dir.path());
        let domain = DedupDomain::local_default();
        let manifest = substrate
            .put(&domain, b"xorb-backed-payload", None)
            .await
            .unwrap();
        let xorb = manifest.xorb_hashes.first().expect("xorb hash").clone();
        let authz = std::sync::Arc::new(RecordingAuthorizer::default());
        let mount = CasMount::with_authorizer(substrate, domain, authz.clone());
        let caller = Subject::new("alice");

        let mut fid = mount.walk(&["xorb", &xorb], &caller).await.unwrap();
        mount.open(&mut fid, OREAD, &caller).await.unwrap();
        let out = mount.read(&fid, 0, 1024, &caller).await.unwrap();

        assert!(!out.is_empty());
        assert_eq!(authz.calls()[0].1, CasMountObjectKind::Xorb);
        assert_eq!(authz.calls()[1].1, CasMountObjectKind::Xorb);
    }

    #[tokio::test]
    async fn root_lists_non_enumerating_cas_dirs() {
        let mount = CasMount::with_authorizer(
            CasSubstrate::new(tempfile::tempdir().unwrap().path()),
            DedupDomain::local_default(),
            AllowAllCasAuthorizer,
        );
        let caller = Subject::new("alice");
        let fid = mount.walk(&[], &caller).await.unwrap();
        let entries = mount.readdir(&fid, &caller).await.unwrap();
        let names: Vec<_> = entries.into_iter().map(|e| e.name).collect();

        assert_eq!(names, vec!["obj", "xorb", "stage"]);
    }
}
