//! Mount trait — in-process 9P operations without network transport.

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
#[derive(Clone, Debug)]
pub struct Stat {
    pub qtype: u8,
    pub size: u64,
    pub name: String,
    pub mtime: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// Mount trait
// ─────────────────────────────────────────────────────────────────────────────

/// In-process mount backend. Implements 9P semantics without ZMQ/IPC.
///
/// Used for `/config/`, `/private/`, `/bin/`, `/env/`, and any other
/// client-local namespace entries. Every method receives the verified
/// `Subject` of the caller for per-tenant fid isolation and policy checks.
pub trait Mount: Send + Sync {
    /// Walk path components, returning an opaque fid.
    fn walk(&self, components: &[&str], caller: &Subject) -> Result<Fid, MountError>;

    /// Open a walked fid for I/O. `mode`: OREAD=0, OWRITE=1, ORDWR=2.
    fn open(&self, fid: &mut Fid, mode: u8, caller: &Subject) -> Result<(), MountError>;

    /// Read bytes from an open fid at offset.
    fn read(&self, fid: &Fid, offset: u64, count: u32, caller: &Subject) -> Result<Vec<u8>, MountError>;

    /// Write bytes to an open fid at offset. Returns bytes written.
    fn write(&self, fid: &Fid, offset: u64, data: &[u8], caller: &Subject) -> Result<u32, MountError>;

    /// Read directory entries from an open directory fid.
    fn readdir(&self, fid: &Fid, caller: &Subject) -> Result<Vec<DirEntry>, MountError>;

    /// Get file metadata.
    fn stat(&self, fid: &Fid, caller: &Subject) -> Result<Stat, MountError>;

    /// Release a fid.
    fn clunk(&self, fid: Fid, caller: &Subject);
}
