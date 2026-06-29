//! Profile B (#506): preview1 `WasiDir`/`WasiFile` backed by a Subject-scoped
//! `hyprstream_vfs::Mount`.
//!
//! This is the make-or-break validation for running a WASI guest whose ENTIRE
//! filesystem is a virtual VFS [`Mount`] (not the host). The verdict in
//! `FINDINGS.md` §4 is implemented here: back a WASI preopen with an arbitrary
//! async `Mount` via the legacy **preview1** [`wasi_common::WasiDir`] /
//! [`wasi_common::WasiFile`] trait objects, NOT preview2's `wasmtime-wasi` (whose
//! preopen only takes a concrete `cap_std::fs::Dir`).
//!
//! ## The ONE policy enforcement point (MAC invariant)
//!
//! The [`MountDir`]/[`MountFile`] adapters do **NOT** authorize. They thread the
//! `Sandbox`-bound [`Subject`] into every `Mount` call and nothing more. The
//! [`Mount`] is the single PEP (a future S2/#568 label check lands THERE). There
//! is deliberately NO parallel capability/permission check in this layer — adding
//! one would create a second, divergent policy surface. The Subject is fixed at
//! construction (`MountDir { mount, subject, .. }`); the guest can neither name nor
//! forge an identity, exactly mirroring the Profile-A proxy seam.
//!
//! ## Async/sync bridge
//!
//! `wasi-common`'s `WasiDir`/`WasiFile` are **async traits**, but we link them via
//! `wasi_common::sync::add_to_linker` whose wiggle integration uses the `block_on`
//! executor: it expects every async fn to poll to `Ready` essentially immediately
//! (no real async suspension). The [`Mount`] is genuinely async, so each adapter
//! method drives the Mount future to completion on a dedicated current-thread
//! tokio [`Runtime`] via `rt.block_on(...)`. The `WasiSandbox` is driven from a
//! plain (non-async) thread (like Profile A), so this nested `block_on` is NOT on a
//! tokio worker and does not panic. From the wiggle executor's view the adapter
//! future resolves immediately — the contract holds.
//!
//! ## Path model
//!
//! `Mount` is fid-based (`walk` -> open -> read/write). WASI is path-based relative
//! to a preopen dir. [`MountDir`] holds a base path (the preopen root, e.g. the
//! Mount root) and, per `open_file`/`get_path_filestat`, splits the guest-supplied
//! relative path into components, `walk`s the Mount, then opens/reads/writes. We
//! reject `..` components (no upward traversal past the preopen) — the WASI ABI
//! also forbids it, but we enforce it here too as defense in depth (this is a path
//! shape check, NOT an authz check; authz stays in the Mount).

use std::any::Any;
use std::sync::Arc;

use hyprstream_vfs::{Mount, MountError, Subject};
use wasi_common::dir::{OpenResult, ReaddirCursor, ReaddirEntity, WasiDir};
use wasi_common::file::{FdFlags, FileType, Filestat, OFlags, WasiFile};
use wasi_common::{Error as WasiError, ErrorExt};

use tokio::runtime::Runtime;

/// Map a [`MountError`] to a preview1 `wasi_common::Error` (errno-tagged).
fn map_err(e: MountError) -> WasiError {
    match e {
        MountError::NotFound(_) => WasiError::not_found(),
        MountError::PermissionDenied(_) => WasiError::perm(),
        MountError::NotDirectory(_) => WasiError::not_dir(),
        MountError::IsDirectory(_) => WasiError::badf().context("is a directory"),
        MountError::InvalidArgument(_) => WasiError::invalid_argument(),
        MountError::NotSupported(_) => WasiError::not_supported(),
        MountError::AlreadyExists(_) => WasiError::exist(),
        MountError::Io(_) => WasiError::io(),
    }
}

/// Split a WASI-relative path (`a/b/c`, possibly with a leading/trailing slash)
/// into Mount path components, joined onto `base`. Rejects `..` and absolute paths
/// (defense-in-depth path-shape check; the Mount remains the policy point).
fn resolve(base: &[String], path: &str) -> Result<Vec<String>, WasiError> {
    let mut out = base.to_vec();
    for comp in path.split('/') {
        match comp {
            "" | "." => {}
            ".." => return Err(WasiError::not_supported().context("`..` traversal denied")),
            other => out.push(other.to_owned()),
        }
    }
    Ok(out)
}

/// A WASI preopen directory backed by a Subject-scoped [`Mount`].
///
/// The Subject is FIXED at construction and threaded into every Mount call — the
/// single MAC threading point, no parallel check.
pub struct MountDir {
    mount: Arc<dyn Mount>,
    subject: Subject,
    /// Mount path components this dir is rooted at (the preopen root = `[]`).
    base: Vec<String>,
    /// Shared current-thread runtime to drive the async Mount synchronously.
    rt: Arc<Runtime>,
}

impl MountDir {
    /// Create a preopen `WasiDir` over `mount`, rooted at the Mount root, running
    /// as `subject`. `rt` is the dedicated runtime used to `block_on` Mount futures.
    pub fn new(mount: Arc<dyn Mount>, subject: Subject, rt: Arc<Runtime>) -> Self {
        Self {
            mount,
            subject,
            base: Vec::new(),
            rt,
        }
    }

    fn child(&self, components: Vec<String>) -> Self {
        Self {
            mount: Arc::clone(&self.mount),
            subject: self.subject.clone(),
            base: components,
            rt: Arc::clone(&self.rt),
        }
    }
}

#[async_trait::async_trait]
impl WasiDir for MountDir {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn open_file(
        &self,
        _symlink_follow: bool,
        path: &str,
        oflags: OFlags,
        read: bool,
        write: bool,
        _fdflags: FdFlags,
    ) -> Result<OpenResult, WasiError> {
        let components = resolve(&self.base, path)?;
        let comp_refs: Vec<&str> = components.iter().map(String::as_str).collect();

        // Walk + stat to classify file vs directory. The Subject threads through
        // EVERY call — the Mount is the policy point. A `NotFound` is NOT fatal: it
        // means "no such entry yet", which `path_open` with O_CREAT turns into a new
        // regular file. We map only PermissionDenied (and other hard errors) to a
        // failure here — that is the Mount enforcing policy, which must surface.
        let classified = self.rt.block_on(async {
            let fid = self.mount.walk(&comp_refs, &self.subject).await?;
            let st = self.mount.stat(&fid, &self.subject).await;
            self.mount.clunk(fid, &self.subject).await;
            st
        });
        let (exists, is_dir, size) = match classified {
            Ok(st) => (true, st.qtype & 0x80 != 0, st.size),
            // Absent entry: treat as a to-be-created regular file (path_open O_CREAT).
            Err(MountError::NotFound(_)) => (false, false, 0),
            // Policy denial / other hard error: surface it (Mount is the PEP).
            Err(e) => return Err(map_err(e)),
        };

        if is_dir {
            // Opening a directory yields a dirfd (a child WasiDir scoped deeper).
            return Ok(OpenResult::Dir(Box::new(self.child(components))));
        }

        if oflags.contains(OFlags::DIRECTORY) {
            return Err(WasiError::not_dir());
        }
        if exists && oflags.contains(OFlags::EXCLUSIVE) {
            return Err(WasiError::exist());
        }
        if !exists && !oflags.contains(OFlags::CREATE) && !write {
            // Read of a non-existent file with no O_CREAT.
            return Err(WasiError::not_found());
        }

        // Open the file fid through the Mount (Subject-scoped). Truncation is
        // applied via OTRUNC on the Mount open mode when requested.
        let mode = match (read, write) {
            (true, true) => hyprstream_vfs::ORDWR,
            (false, true) => hyprstream_vfs::OWRITE,
            _ => hyprstream_vfs::OREAD,
        } | if oflags.contains(OFlags::TRUNCATE) {
            hyprstream_vfs::OTRUNC
        } else {
            0
        };

        let fid = self
            .rt
            .block_on(async {
                let mut fid = self.mount.walk(&comp_refs, &self.subject).await?;
                self.mount.open(&mut fid, mode, &self.subject).await?;
                Ok::<_, MountError>(fid)
            })
            .map_err(map_err)?;

        Ok(OpenResult::File(Box::new(MountFile {
            mount: Arc::clone(&self.mount),
            subject: self.subject.clone(),
            fid: Arc::new(parking_lot::Mutex::new(Some(fid))),
            rt: Arc::clone(&self.rt),
            initial_size: size,
            pos: parking_lot::Mutex::new(0),
        })))
    }

    async fn readdir(
        &self,
        cursor: ReaddirCursor,
    ) -> Result<Box<dyn Iterator<Item = Result<ReaddirEntity, WasiError>> + Send>, WasiError> {
        let comp_refs: Vec<&str> = self.base.iter().map(String::as_str).collect();
        let entries = self
            .rt
            .block_on(async {
                let fid = self.mount.walk(&comp_refs, &self.subject).await?;
                let e = self.mount.readdir(&fid, &self.subject).await?;
                self.mount.clunk(fid, &self.subject).await;
                Ok::<_, MountError>(e)
            })
            .map_err(map_err)?;

        // Build absolute entries (`.` and `..` are conventionally elided here; the
        // wiggle layer synthesizes them as needed). Honor the cursor for resumption.
        let start: u64 = cursor.into();
        let mut out: Vec<Result<ReaddirEntity, WasiError>> = Vec::new();
        for (i, de) in entries.into_iter().enumerate() {
            let next = (i as u64) + 1;
            if next <= start {
                continue;
            }
            out.push(Ok(ReaddirEntity {
                next: ReaddirCursor::from(next),
                inode: 0,
                name: de.name,
                filetype: if de.is_dir {
                    FileType::Directory
                } else {
                    FileType::RegularFile
                },
            }));
        }
        Ok(Box::new(out.into_iter()))
    }

    async fn get_filestat(&self) -> Result<Filestat, WasiError> {
        let comp_refs: Vec<&str> = self.base.iter().map(String::as_str).collect();
        let st = self
            .rt
            .block_on(async {
                let fid = self.mount.walk(&comp_refs, &self.subject).await?;
                let s = self.mount.stat(&fid, &self.subject).await?;
                self.mount.clunk(fid, &self.subject).await;
                Ok::<_, MountError>(s)
            })
            .map_err(map_err)?;
        Ok(Filestat {
            device_id: 0,
            inode: 0,
            filetype: FileType::Directory,
            nlink: 1,
            size: st.size,
            atim: None,
            mtim: None,
            ctim: None,
        })
    }

    async fn get_path_filestat(
        &self,
        path: &str,
        _follow_symlinks: bool,
    ) -> Result<Filestat, WasiError> {
        let components = resolve(&self.base, path)?;
        let comp_refs: Vec<&str> = components.iter().map(String::as_str).collect();
        let st = self
            .rt
            .block_on(async {
                let fid = self.mount.walk(&comp_refs, &self.subject).await?;
                let s = self.mount.stat(&fid, &self.subject).await?;
                self.mount.clunk(fid, &self.subject).await;
                Ok::<_, MountError>(s)
            })
            .map_err(map_err)?;
        let is_dir = st.qtype & 0x80 != 0;
        Ok(Filestat {
            device_id: 0,
            inode: 0,
            filetype: if is_dir {
                FileType::Directory
            } else {
                FileType::RegularFile
            },
            nlink: 1,
            size: st.size,
            atim: None,
            mtim: None,
            ctim: None,
        })
    }

    // create_dir / unlink_file / rename / symlink / hard_link / read_link /
    // set_times all default to `Err(not_supported)` from the trait. The base Mount
    // surface (walk/open/read/write/readdir/stat) is what we expose; mutation of
    // the namespace shape is deliberately out of the Profile-B fs surface (mirrors
    // the proxy seam excluding mount/bind/unmount). A future revision can route
    // create/unlink through `Mount::as_fsmount()` if a guest needs them.
}

/// A WASI file backed by an open [`Mount`] fid, Subject-scoped.
pub struct MountFile {
    mount: Arc<dyn Mount>,
    subject: Subject,
    /// The open Mount fid. `Option` so `Drop` can `clunk` it exactly once.
    fid: Arc<parking_lot::Mutex<Option<hyprstream_vfs::Fid>>>,
    rt: Arc<Runtime>,
    initial_size: u64,
    /// Sequential cursor for the non-`_at` `fd_read`/`fd_write`/`fd_seek` ABI (std
    /// `File::read`/`write` use these, not the positional `pread`/`pwrite`).
    pos: parking_lot::Mutex<u64>,
}

#[async_trait::async_trait]
impl WasiFile for MountFile {
    fn as_any(&self) -> &dyn Any {
        self
    }

    async fn get_filetype(&self) -> Result<FileType, WasiError> {
        Ok(FileType::RegularFile)
    }

    async fn get_filestat(&self) -> Result<Filestat, WasiError> {
        let guard = self.fid.lock();
        let fid = guard
            .as_ref()
            .ok_or_else(|| WasiError::badf().context("file closed"))?;
        let st = self
            .rt
            .block_on(self.mount.stat(fid, &self.subject))
            .map_err(map_err)?;
        Ok(Filestat {
            device_id: 0,
            inode: 0,
            filetype: FileType::RegularFile,
            nlink: 1,
            size: st.size,
            atim: None,
            mtim: None,
            ctim: None,
        })
    }

    async fn read_vectored_at<'a>(
        &self,
        bufs: &mut [std::io::IoSliceMut<'a>],
        offset: u64,
    ) -> Result<u64, WasiError> {
        let total: usize = bufs.iter().map(|b| b.len()).sum();
        if total == 0 {
            return Ok(0);
        }
        let data = {
            let guard = self.fid.lock();
            let fid = guard
                .as_ref()
                .ok_or_else(|| WasiError::badf().context("file closed"))?;
            self.rt
                .block_on(self.mount.read(fid, offset, total as u32, &self.subject))
                .map_err(map_err)?
        };
        // Scatter the contiguous Mount read across the guest iovecs.
        let mut copied = 0usize;
        for buf in bufs.iter_mut() {
            if copied >= data.len() {
                break;
            }
            let n = buf.len().min(data.len() - copied);
            buf[..n].copy_from_slice(&data[copied..copied + n]);
            copied += n;
        }
        Ok(copied as u64)
    }

    async fn write_vectored_at<'a>(
        &self,
        bufs: &[std::io::IoSlice<'a>],
        offset: u64,
    ) -> Result<u64, WasiError> {
        // Gather the guest iovecs into one contiguous buffer for the Mount write.
        let mut data = Vec::new();
        for b in bufs {
            data.extend_from_slice(b);
        }
        if data.is_empty() {
            return Ok(0);
        }
        let n = {
            let guard = self.fid.lock();
            let fid = guard
                .as_ref()
                .ok_or_else(|| WasiError::badf().context("file closed"))?;
            self.rt
                .block_on(self.mount.write(fid, offset, &data, &self.subject))
                .map_err(map_err)?
        };
        Ok(n as u64)
    }

    async fn read_vectored<'a>(
        &self,
        bufs: &mut [std::io::IoSliceMut<'a>],
    ) -> Result<u64, WasiError> {
        // Sequential read: read at the cursor, then advance it.
        let offset = *self.pos.lock();
        let n = self.read_vectored_at(bufs, offset).await?;
        *self.pos.lock() = offset + n;
        Ok(n)
    }

    async fn write_vectored<'a>(&self, bufs: &[std::io::IoSlice<'a>]) -> Result<u64, WasiError> {
        // Sequential write: write at the cursor, then advance it.
        let offset = *self.pos.lock();
        let n = self.write_vectored_at(bufs, offset).await?;
        *self.pos.lock() = offset + n;
        Ok(n)
    }

    async fn seek(&self, pos: std::io::SeekFrom) -> Result<u64, WasiError> {
        use std::io::SeekFrom;
        let new = match pos {
            SeekFrom::Start(o) => o,
            SeekFrom::Current(d) => {
                let cur = *self.pos.lock() as i64;
                (cur + d).max(0) as u64
            }
            SeekFrom::End(d) => {
                // Resolve the end from the current Mount size.
                let size = {
                    let guard = self.fid.lock();
                    let fid = guard
                        .as_ref()
                        .ok_or_else(|| WasiError::badf().context("file closed"))?;
                    self.rt
                        .block_on(self.mount.stat(fid, &self.subject))
                        .map_err(map_err)?
                        .size
                };
                ((size as i64) + d).max(0) as u64
            }
        };
        *self.pos.lock() = new;
        Ok(new)
    }

    async fn get_fdflags(&self) -> Result<FdFlags, WasiError> {
        Ok(FdFlags::empty())
    }

    fn num_ready_bytes(&self) -> Result<u64, WasiError> {
        Ok(self.initial_size)
    }
}

impl Drop for MountFile {
    fn drop(&mut self) {
        // Clunk the fid through the Mount as the bound Subject, exactly once.
        if let Some(fid) = self.fid.lock().take() {
            self.rt.block_on(self.mount.clunk(fid, &self.subject));
        }
    }
}
