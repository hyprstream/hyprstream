//! Session — per-connection fid table and state for a 9P server.
//!
//! Each client connection gets its own `Session` which tracks:
//! - The fid table: maps 9P wire fid numbers to `Mount` fids
//! - The negotiated message size (`msize`)
//! - The caller identity (`Subject`) for authorization
//!
//! The session translates 9P2000.L T-messages into `Mount` trait calls and
//! produces the corresponding R-messages.

use std::collections::HashMap;
use std::sync::Arc;

use hyprstream_rpc::Subject;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat, OREAD, ORDWR};
use tracing::debug;

use crate::protocol::*;

/// Default maximum message size (8 MB).
const DEFAULT_MSIZE: u32 = 8 * 1024 * 1024;

/// 9P protocol version string.
const VERSION_9P2000_L: &str = "9P2000.L";

/// POSIX errno constants (subset used by 9P2000.L error responses).
mod errno {
    pub const EPERM: u32 = 1;
    pub const ENOENT: u32 = 2;
    pub const EIO: u32 = 5;
    pub const EBADF: u32 = 9;
    #[allow(dead_code)]
    pub const ENOMEM: u32 = 12;
    pub const EEXIST: u32 = 17;
    pub const ENOTDIR: u32 = 20;
    pub const EISDIR: u32 = 21;
    pub const EINVAL: u32 = 22;
    #[allow(dead_code)]
    pub const EROFS: u32 = 30;
    pub const ENOTSUP: u32 = 95;
    pub const EOPNOTSUPP: u32 = 95;
}

/// Per-connection session state.
pub struct Session {
    /// The mount being served.
    mount: Arc<dyn Mount>,

    /// Negotiated maximum message size.
    msize: u32,

    /// Wire fid -> VFS Fid mapping.
    fids: HashMap<u32, Fid>,

    /// Caller identity for all operations on this session.
    caller: Subject,
}

impl Session {
    /// Create a new session for a mount with the given caller identity.
    pub fn new(mount: Arc<dyn Mount>, caller: Subject) -> Self {
        Self {
            mount,
            msize: DEFAULT_MSIZE,
            fids: HashMap::new(),
            caller,
        }
    }

    /// Process a T-message and produce the corresponding R-message.
    pub async fn handle(&mut self, msg: Tmessage) -> Rmessage {
        match msg {
            Tmessage::Version(v) => self.handle_version(v),
            Tmessage::Attach(a) => self.handle_attach(a).await,
            Tmessage::Walk(w) => self.handle_walk(w).await,
            Tmessage::Lopen(o) => self.handle_lopen(o).await,
            Tmessage::Read(r) => self.handle_read(r).await,
            Tmessage::Write(w) => self.handle_write(w).await,
            Tmessage::Clunk(c) => self.handle_clunk(c).await,
            Tmessage::Readdir(r) => self.handle_readdir(r).await,
            Tmessage::GetAttr(g) => self.handle_getattr(g).await,
            Tmessage::Flush(_) => Rmessage::Flush,
            Tmessage::Statfs(_) => self.handle_statfs(),
            Tmessage::Auth(_) => error_msg(errno::EOPNOTSUPP),
        }
    }

    /// Negotiated message size.
    pub fn msize(&self) -> u32 {
        self.msize
    }

    // ── Handlers ────────────────────────────────────────────────────────────

    fn handle_version(&mut self, v: Tversion) -> Rmessage {
        self.msize = v.msize.min(DEFAULT_MSIZE);
        self.fids.clear();
        Rmessage::Version(Rversion {
            msize: self.msize,
            version: VERSION_9P2000_L.into(),
        })
    }

    async fn handle_attach(&mut self, a: Tattach) -> Rmessage {
        match self.mount.walk(&[], &self.caller).await {
            Ok(fid) => {
                self.fids.insert(a.fid, fid);
                Rmessage::Attach(Rattach {
                    qid: Qid {
                        ty: QTDIR,
                        version: 0,
                        path: 0,
                    },
                })
            }
            Err(e) => {
                debug!("attach failed: {e}");
                mount_error_to_rmsg(&e)
            }
        }
    }

    async fn handle_walk(&mut self, w: Twalk) -> Rmessage {
        if !self.fids.contains_key(&w.fid) {
            return error_msg(errno::EBADF);
        }

        let components: Vec<&str> = w.wnames.iter().map(String::as_str).collect();

        match self.mount.walk(&components, &self.caller).await {
            Ok(fid) => {
                let wqids: Vec<Qid> = (0..w.wnames.len())
                    .map(|i| Qid {
                        ty: if i == w.wnames.len() - 1 {
                            QTFILE
                        } else {
                            QTDIR
                        },
                        version: 0,
                        path: i as u64,
                    })
                    .collect();

                self.fids.insert(w.newfid, fid);
                Rmessage::Walk(Rwalk { wqids })
            }
            Err(e) => mount_error_to_rmsg(&e),
        }
    }

    async fn handle_lopen(&mut self, o: Tlopen) -> Rmessage {
        let Some(fid) = self.fids.get_mut(&o.fid) else {
            return error_msg(errno::EBADF);
        };

        let mode = flags_to_mode(o.flags);

        match self.mount.open(fid, mode, &self.caller).await {
            Ok(()) => Rmessage::Lopen(Rlopen {
                qid: Qid {
                    ty: QTFILE,
                    version: 0,
                    path: 0,
                },
                iounit: 0,
            }),
            Err(e) => mount_error_to_rmsg(&e),
        }
    }

    async fn handle_read(&self, r: Tread) -> Rmessage {
        let Some(fid) = self.fids.get(&r.fid) else {
            return error_msg(errno::EBADF);
        };

        let max_count = self.msize.saturating_sub(11);
        let count = r.count.min(max_count);

        match self.mount.read(fid, r.offset, count, &self.caller).await {
            Ok(data) => Rmessage::Read(Rread { data }),
            Err(e) => mount_error_to_rmsg(&e),
        }
    }

    async fn handle_write(&self, w: Twrite) -> Rmessage {
        let Some(fid) = self.fids.get(&w.fid) else {
            return error_msg(errno::EBADF);
        };

        match self.mount.write(fid, w.offset, &w.data, &self.caller).await {
            Ok(count) => Rmessage::Write(Rwrite { count }),
            Err(e) => mount_error_to_rmsg(&e),
        }
    }

    async fn handle_clunk(&mut self, c: Tclunk) -> Rmessage {
        if let Some(fid) = self.fids.remove(&c.fid) {
            self.mount.clunk(fid, &self.caller).await;
        }
        Rmessage::Clunk
    }

    async fn handle_readdir(&self, r: Treaddir) -> Rmessage {
        let Some(fid) = self.fids.get(&r.fid) else {
            return error_msg(errno::EBADF);
        };

        match self.mount.readdir(fid, &self.caller).await {
            Ok(entries) => {
                let data = encode_dirents(&entries, r.offset, r.count);
                Rmessage::Readdir(Rreaddir { data })
            }
            Err(e) => mount_error_to_rmsg(&e),
        }
    }

    async fn handle_getattr(&self, g: Tgetattr) -> Rmessage {
        let Some(fid) = self.fids.get(&g.fid) else {
            return error_msg(errno::EBADF);
        };

        match self.mount.stat(fid, &self.caller).await {
            Ok(stat) => Rmessage::GetAttr(stat_to_rgetattr(&stat)),
            Err(e) => mount_error_to_rmsg(&e),
        }
    }

    fn handle_statfs(&self) -> Rmessage {
        Rmessage::Statfs(Rstatfs {
            ty: 0x01021997, // V9FS_MAGIC
            bsize: 4096,
            blocks: 0,
            bfree: 0,
            bavail: 0,
            files: 0,
            ffree: 0,
            fsid: 0,
            namelen: 255,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

fn error_msg(ecode: u32) -> Rmessage {
    Rmessage::Lerror(Rlerror { ecode })
}

fn mount_error_to_rmsg(e: &MountError) -> Rmessage {
    let ecode = match e {
        MountError::NotFound(_) => errno::ENOENT,
        MountError::PermissionDenied(_) => errno::EPERM,
        MountError::NotDirectory(_) => errno::ENOTDIR,
        MountError::IsDirectory(_) => errno::EISDIR,
        MountError::InvalidArgument(_) => errno::EINVAL,
        MountError::NotSupported(_) => errno::ENOTSUP,
        MountError::AlreadyExists(_) => errno::EEXIST,
        MountError::Io(_) => errno::EIO,
    };
    error_msg(ecode)
}

fn flags_to_mode(flags: u32) -> u8 {
    match flags & 0x03 {
        0 => OREAD,
        1 => hyprstream_vfs::OWRITE,
        2 => ORDWR,
        _ => OREAD,
    }
}

/// Encode DirEntry list into 9P2000.L readdir data buffer.
fn encode_dirents(entries: &[DirEntry], offset: u64, count: u32) -> Vec<u8> {
    let start = offset as usize;
    let mut buf = Vec::new();

    for (i, entry) in entries.iter().enumerate() {
        if i < start {
            continue;
        }

        let qtype = if entry.is_dir { QTDIR } else { QTFILE };
        let dirent = Dirent {
            qid: Qid {
                ty: qtype,
                version: 0,
                path: i as u64,
            },
            offset: (i + 1) as u64,
            ty: qtype,
            name: entry.name.clone(),
        };

        if buf.len() + dirent.byte_size() > count as usize {
            break;
        }

        // Encode; Vec<u8> writer cannot fail.
        let _ = dirent.encode(&mut buf);
    }

    buf
}

fn stat_to_rgetattr(stat: &Stat) -> Rgetattr {
    let is_dir = stat.qtype & 0x80 != 0;
    let mode = if is_dir { 0o040555 } else { 0o100444 };

    Rgetattr {
        valid: 0x000007FF,
        qid: Qid {
            ty: stat.qtype,
            version: 0,
            path: 0,
        },
        mode,
        uid: 0,
        gid: 0,
        nlink: if is_dir { 2 } else { 1 },
        rdev: 0,
        size: stat.size,
        blksize: 4096,
        blocks: stat.size.div_ceil(512),
        atime_sec: stat.mtime,
        atime_nsec: 0,
        mtime_sec: stat.mtime,
        mtime_nsec: 0,
        ctime_sec: stat.mtime,
        ctime_nsec: 0,
        btime_sec: 0,
        btime_nsec: 0,
        gen: 0,
        data_version: 0,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use hyprstream_vfs::{Fid as VfsFid, Mount, MountError, Stat as VfsStat};
    use std::collections::HashMap as StdHashMap;

    struct TestMount {
        files: StdHashMap<String, Vec<u8>>,
    }

    struct TestFid {
        path: String,
    }

    #[async_trait]
    impl Mount for TestMount {
        async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<VfsFid, MountError> {
            let path = components.join("/");
            Ok(VfsFid::new(TestFid { path }))
        }

        async fn open(&self, _fid: &mut VfsFid, _mode: u8, _caller: &Subject) -> Result<(), MountError> {
            Ok(())
        }

        async fn read(&self, fid: &VfsFid, offset: u64, _count: u32, _caller: &Subject) -> Result<Vec<u8>, MountError> {
            let inner = fid.downcast_ref::<TestFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            match self.files.get(&inner.path) {
                Some(data) => {
                    let start = offset as usize;
                    if start >= data.len() {
                        Ok(vec![])
                    } else {
                        Ok(data[start..].to_vec())
                    }
                }
                None => Err(MountError::NotFound(inner.path.clone())),
            }
        }

        async fn write(&self, _fid: &VfsFid, _offset: u64, data: &[u8], _caller: &Subject) -> Result<u32, MountError> {
            Ok(data.len() as u32)
        }

        async fn readdir(&self, fid: &VfsFid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
            let inner = fid.downcast_ref::<TestFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let prefix = if inner.path.is_empty() { String::new() } else { format!("{}/", inner.path) };
            let mut entries = Vec::new();
            for key in self.files.keys() {
                if let Some(rest) = key.strip_prefix(&prefix) {
                    if !rest.contains('/') {
                        entries.push(DirEntry { name: rest.to_owned(), is_dir: false, size: 0, stat: None });
                    }
                }
            }
            Ok(entries)
        }

        async fn stat(&self, fid: &VfsFid, _caller: &Subject) -> Result<VfsStat, MountError> {
            let inner = fid.downcast_ref::<TestFid>().ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
            let size = self.files.get(&inner.path).map_or(0, |d| d.len() as u64);
            Ok(VfsStat { qtype: 0, size, name: inner.path.clone(), mtime: 0 })
        }

        async fn clunk(&self, _fid: VfsFid, _caller: &Subject) {}
    }

    fn test_mount() -> Arc<dyn Mount> {
        let mut files = StdHashMap::new();
        files.insert("status".to_owned(), b"loaded\n".to_vec());
        files.insert("config/temperature".to_owned(), b"0.7\n".to_vec());
        Arc::new(TestMount { files })
    }

    #[tokio::test]
    async fn version_negotiation() {
        let mut session = Session::new(test_mount(), Subject::new("test"));
        let resp = session.handle(Tmessage::Version(Tversion {
            msize: 65536,
            version: "9P2000.L".into(),
        })).await;
        match resp {
            Rmessage::Version(v) => {
                assert_eq!(v.version, "9P2000.L");
                assert!(v.msize <= DEFAULT_MSIZE);
            }
            other => panic!("expected Rversion, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn attach_walk_read_clunk() {
        let mut session = Session::new(test_mount(), Subject::new("test"));

        // Version
        session.handle(Tmessage::Version(Tversion {
            msize: 8192,
            version: "9P2000.L".into(),
        })).await;

        // Attach root fid=0
        let resp = session.handle(Tmessage::Attach(Tattach {
            fid: 0,
            afid: u32::MAX,
            uname: "nobody".into(),
            aname: String::new(),
            n_uname: u32::MAX,
        })).await;
        assert!(matches!(resp, Rmessage::Attach(_)));

        // Walk to "status", newfid=1
        let resp = session.handle(Tmessage::Walk(Twalk {
            fid: 0,
            newfid: 1,
            wnames: vec!["status".into()],
        })).await;
        assert!(matches!(resp, Rmessage::Walk(_)));

        // Open fid=1
        let resp = session.handle(Tmessage::Lopen(Tlopen { fid: 1, flags: 0 })).await;
        assert!(matches!(resp, Rmessage::Lopen(_)));

        // Read
        let resp = session.handle(Tmessage::Read(Tread { fid: 1, offset: 0, count: 4096 })).await;
        match resp {
            Rmessage::Read(r) => assert_eq!(r.data, b"loaded\n"),
            other => panic!("expected Rread, got {:?}", other),
        }

        // Clunk
        assert!(matches!(session.handle(Tmessage::Clunk(Tclunk { fid: 1 })).await, Rmessage::Clunk));
    }

    #[tokio::test]
    async fn read_nonexistent_returns_enoent() {
        let mut session = Session::new(test_mount(), Subject::new("test"));
        session.handle(Tmessage::Version(Tversion { msize: 8192, version: "9P2000.L".into() })).await;
        session.handle(Tmessage::Attach(Tattach {
            fid: 0, afid: u32::MAX, uname: "nobody".into(), aname: String::new(), n_uname: u32::MAX,
        })).await;
        session.handle(Tmessage::Walk(Twalk { fid: 0, newfid: 1, wnames: vec!["nonexistent".into()] })).await;
        session.handle(Tmessage::Lopen(Tlopen { fid: 1, flags: 0 })).await;

        let resp = session.handle(Tmessage::Read(Tread { fid: 1, offset: 0, count: 4096 })).await;
        match resp {
            Rmessage::Lerror(e) => assert_eq!(e.ecode, errno::ENOENT),
            other => panic!("expected Lerror(ENOENT), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn auth_returns_eopnotsupp() {
        let mut session = Session::new(test_mount(), Subject::new("test"));
        session.handle(Tmessage::Version(Tversion { msize: 8192, version: "9P2000.L".into() })).await;
        let resp = session.handle(Tmessage::Auth(Tauth {
            afid: 0, uname: "nobody".into(), aname: String::new(), n_uname: u32::MAX,
        })).await;
        assert!(matches!(resp, Rmessage::Lerror(_)));
    }
}
