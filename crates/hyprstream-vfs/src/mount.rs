//! Mount trait — in-process 9P operations without network transport.

use async_trait::async_trait;
use hyprstream_rpc::Subject;

// ─────────────────────────────────────────────────────────────────────────────
// Open mode constants (9P2000)
// ─────────────────────────────────────────────────────────────────────────────

/// Open for reading
pub const OREAD: u8 = 0;
/// Open for writing
pub const OWRITE: u8 = 1;
/// Open for read+write
pub const ORDWR: u8 = 2;
/// Truncate on open (OR'd with above)
pub const OTRUNC: u8 = 0x10;
/// Remove on clunk (OR'd with above)
pub const ORCLOSE: u8 = 0x40;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Mount-level error.
#[derive(Debug)]
pub enum MountError {
    NotFound(String),
    PermissionDenied(String),
    NotDirectory(String),
    IsDirectory(String),
    InvalidArgument(String),
    NotSupported(String),
    AlreadyExists(String),
    Io(String),
}

impl std::fmt::Display for MountError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(s) => write!(f, "not found: {s}"),
            Self::PermissionDenied(s) => write!(f, "permission denied: {s}"),
            Self::NotDirectory(s) => write!(f, "not a directory: {s}"),
            Self::IsDirectory(s) => write!(f, "is a directory: {s}"),
            Self::InvalidArgument(s) => write!(f, "invalid argument: {s}"),
            Self::NotSupported(s) => write!(f, "not supported: {s}"),
            Self::AlreadyExists(s) => write!(f, "already exists: {s}"),
            Self::Io(s) => write!(f, "I/O error: {s}"),
        }
    }
}

impl std::error::Error for MountError {}

// ─────────────────────────────────────────────────────────────────────────────
// Fid
// ─────────────────────────────────────────────────────────────────────────────

/// Opaque fid handle from a mount.
pub struct Fid(Box<dyn std::any::Any + Send + Sync>);

impl Fid {
    /// Create a new Fid wrapping an arbitrary Send+Sync value.
    pub fn new<T: std::any::Any + Send + Sync>(val: T) -> Self {
        Self(Box::new(val))
    }

    /// Downcast to a concrete type (immutable).
    pub fn downcast_ref<T: std::any::Any>(&self) -> Option<&T> {
        self.0.downcast_ref()
    }

    /// Downcast to a concrete type (mutable).
    pub fn downcast_mut<T: std::any::Any>(&mut self) -> Option<&mut T> {
        self.0.downcast_mut()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DirEntry & Stat
// ─────────────────────────────────────────────────────────────────────────────

/// Directory entry.
#[derive(Clone, Debug)]
pub struct DirEntry {
    pub name: String,
    pub is_dir: bool,
    pub size: u64,
    pub stat: Option<Stat>,
}

/// File metadata.
///
/// # Qid soundness invariant — SECURITY
///
/// In 9P, a file's `qid` is its *identity*: two files are the same file if
/// and only if their qids are equal. A qid is the triple `{qtype, version,
/// path}`. Historically this struct carried only `qtype`, flattening the qid
/// and destroying the identity signal — every file looked like the same file
/// to any qid-keyed consumer.
///
/// `version` and `path` are now threaded end-to-end so the surface is sound,
/// but the *values* are not yet a strong identity:
///   - Synthetic mounts set `version = 1` constant and `path = fnv64(path)`,
///     which is non-cryptographic and changes on rename.
///   - Registry-backed mounts set `path = inode` (rename-stable but subject to
///     inode reuse) and `version = ctime` (truncated).
///
/// **Convention:** `path == 0` means "unknown / not provided" (e.g. non-Unix
/// registry fallback, or a mount that has no meaningful file identity). It is
/// NOT a valid identity and must not be treated as one.
///
/// **No server authz, cache key, or dedup logic may key on `qid` (version or
/// path) as a sound identity until #387 lands a content-CID-derived qid with
/// an AFS-style data-version uniquifier.** Treat the qid fields here as an
/// advisory identity *hint* only. Authz must continue to key on the verified
/// `Subject` plus path resolution, never on qid.
#[derive(Clone, Debug)]
pub struct Stat {
    pub qtype: u8,
    /// Qid version (data-version). Advisory hint only — see type-level invariant.
    pub version: u32,
    /// Qid path (file identity). `0` means unknown. Advisory hint only — see type-level invariant.
    pub path: u64,
    pub size: u64,
    pub name: String,
    pub mtime: u64,
}

impl Stat {
    /// Construct a `Stat` with `version = 0, path = 0` (qid unknown).
    ///
    /// Use this for mounts that have no meaningful file identity (e.g. purely
    /// synthetic test mounts). The result carries qtype/size/name/mtime and
    /// signals via `path == 0` that the qid is not a usable identity.
    pub fn unknown_qid(qtype: u8, size: u64, name: String, mtime: u64) -> Self {
        Self { qtype, version: 0, path: 0, size, name, mtime }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Mount trait
// ─────────────────────────────────────────────────────────────────────────────

/// In-process mount backend. Implements 9P semantics without ZMQ/IPC.
///
/// Used for `/config/`, `/private/`, `/bin/`, `/env/`, and any other
/// client-local namespace entries. Every method receives the verified
/// `Subject` of the caller for per-tenant fid isolation and policy checks.
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", allow(clippy::non_send_fields_in_send_ty))]
pub trait Mount: Send + Sync {
    /// Walk path components, returning an opaque fid.
    async fn walk(&self, components: &[&str], caller: &Subject) -> Result<Fid, MountError>;

    /// Open a walked fid for I/O. `mode`: OREAD=0, OWRITE=1, ORDWR=2.
    async fn open(&self, fid: &mut Fid, mode: u8, caller: &Subject) -> Result<(), MountError>;

    /// Read bytes from an open fid at offset.
    async fn read(&self, fid: &Fid, offset: u64, count: u32, caller: &Subject) -> Result<Vec<u8>, MountError>;

    /// Write bytes to an open fid at offset. Returns bytes written.
    async fn write(&self, fid: &Fid, offset: u64, data: &[u8], caller: &Subject) -> Result<u32, MountError>;

    /// Read directory entries from an open directory fid.
    async fn readdir(&self, fid: &Fid, caller: &Subject) -> Result<Vec<DirEntry>, MountError>;

    /// Get file metadata.
    async fn stat(&self, fid: &Fid, caller: &Subject) -> Result<Stat, MountError>;

    /// Release a fid.
    async fn clunk(&self, fid: Fid, caller: &Subject);
}
