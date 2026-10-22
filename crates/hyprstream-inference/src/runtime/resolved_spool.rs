//! Per-load disk backing for resolved channel-FP8 pointer shards.
//!
//! A resolver may still allocate one shard. Spill it before requesting the
//! next shard, so whole-checkpoint preflight retains only file descriptors,
//! rather than every resolved heap buffer plus materialized tensors. This is
//! not a global object cache: anonymous files die with their retained handles.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};

#[cfg(target_os = "linux")]
const RESERVE_BYTES: u64 = 512 << 20;
const IO_BYTES: usize = 64 << 10;

/// Only trusted process XDG/home convention selects the cache; neither model
/// paths nor RPC inputs select a spool location. Do not use /tmp (often tmpfs).
fn cache_parent() -> Result<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
                .map(|home| home.join(".cache"))
        })
        .context("resolved FP8 spool requires absolute XDG cache or HOME")?;
    // DirBuilder mode applies only to directories newly created here. Never
    // chmod a shared or pre-existing cache ancestor.
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        std::fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&base)?;
    }
    #[cfg(not(unix))]
    std::fs::create_dir_all(&base)?;
    let base = base.canonicalize()?;
    verify_owned_directory(&base, false)?;
    let app = base.join("hyprstream");
    let private = app.join("fp8-resolved-shards");
    for path in [&app, &private] {
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt as _;
            match std::fs::DirBuilder::new().mode(0o700).create(path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(error.into()),
            }
        }
        #[cfg(not(unix))]
        std::fs::create_dir_all(path)?;
        let metadata = std::fs::symlink_metadata(path)?;
        anyhow::ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "resolved FP8 cache child must be a real directory"
        );
        verify_owned_directory(path, path == &private)?;
    }
    verify_private_parent(&private)?;
    Ok(private)
}

#[cfg(unix)]
fn verify_owned_directory(parent: &Path, private: bool) -> Result<()> {
    use std::os::unix::fs::MetadataExt as _;
    let metadata = std::fs::symlink_metadata(parent)?;
    // SAFETY: geteuid has no pointer arguments or preconditions.
    let uid = unsafe { libc::geteuid() };
    anyhow::ensure!(
        metadata.is_dir()
            && !metadata.file_type().is_symlink()
            && metadata.uid() == uid
            && if private {
                metadata.mode() & 0o777 == 0o700
            } else {
                metadata.mode() & 0o022 == 0
            },
        "resolved FP8 spool parent must be owned, real, private0700"
    );
    Ok(())
}
#[cfg(not(unix))]
fn verify_owned_directory(_parent: &Path, _private: bool) -> Result<()> {
    anyhow::bail!("private resolved FP8 disk spooling is unavailable on this platform")
}

fn verify_private_parent(parent: &Path) -> Result<()> {
    verify_owned_directory(parent, true)
}

#[cfg(target_os = "linux")]
fn required_capacity(bytes: u64) -> Result<u64> {
    bytes
        .checked_add(RESERVE_BYTES)
        .context("resolved FP8 spool capacity overflow")
}

#[cfg(target_os = "linux")]
fn admit_file(file: &File, bytes: u64) -> Result<()> {
    use std::os::fd::AsRawFd as _;
    use std::os::unix::fs::MetadataExt as _;
    let metadata = file.metadata()?;
    anyhow::ensure!(
        metadata.is_file() && metadata.nlink() == 0 && metadata.mode() & 0o777 == 0o600,
        "resolved FP8 spool must be an anonymous0600 regular file"
    );
    let mut fs = std::mem::MaybeUninit::<libc::statfs>::uninit();
    // SAFETY: the owned file FD is live and fs is writable statfs storage.
    anyhow::ensure!(
        unsafe { libc::fstatfs(file.as_raw_fd(), fs.as_mut_ptr()) } == 0,
        "resolved FP8 spool filesystem check failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: successful fstatfs initialized fs.
    let fs = unsafe { fs.assume_init() };
    anyhow::ensure!(
        fs.f_type != 0x0102_1994 && fs.f_type != 0x8584_58f6,
        "resolved FP8 spool refuses memory-backed filesystem"
    );
    let mut capacity = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: live owned FD and writable statvfs storage.
    anyhow::ensure!(
        unsafe { libc::fstatvfs(file.as_raw_fd(), capacity.as_mut_ptr()) } == 0,
        "resolved FP8 spool capacity check failed: {}",
        std::io::Error::last_os_error()
    );
    // SAFETY: successful fstatvfs initialized capacity.
    let capacity = unsafe { capacity.assume_init() };
    let available = u128::from(capacity.f_bavail) * u128::from(capacity.f_frsize);
    anyhow::ensure!(
        available >= u128::from(required_capacity(bytes)?),
        "insufficient disk for resolved FP8 spool plus512MiB reserve"
    );
    Ok(())
}
#[cfg(not(target_os = "linux"))]
fn admit_file(_file: &File, _bytes: u64) -> Result<()> {
    anyhow::bail!("resolved FP8 disk filesystem admission requires Linux")
}

fn copy_verified(bytes: &[u8], file: &mut File) -> Result<()> {
    let expected: [u8; 32] = Sha256::digest(bytes).into();
    for chunk in bytes.chunks(IO_BYTES) {
        file.write_all(chunk)?;
    }
    file.flush()?;
    anyhow::ensure!(
        file.metadata()?.len() == bytes.len() as u64,
        "resolved FP8 spool length differs"
    );
    file.rewind()?;
    let mut hash = Sha256::new();
    let mut buffer = [0; IO_BYTES];
    let mut count = 0_u64;
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        count = count
            .checked_add(n as u64)
            .context("resolved FP8 readback length overflow")?;
        hash.update(&buffer[..n]);
    }
    let actual: [u8; 32] = hash.finalize().into();
    anyhow::ensure!(
        count == bytes.len() as u64 && actual == expected,
        "resolved FP8 spool readback bytes differ"
    );
    file.rewind()?;
    Ok(())
}

/// Called in spawn_blocking with the ONE current resolver allocation moved
/// into it. Cancellation cannot abandon a named payload: this function owns
/// the anonymous FD and Vec until completion, and the caller awaits it before
/// starting any next resolution. Do not clone the Vec or re-resolve the shard.
pub(super) fn spool(bytes: Vec<u8>) -> Result<File> {
    spool_in(bytes, &cache_parent()?)
}

pub(super) fn spool_in(bytes: Vec<u8>, parent: &Path) -> Result<File> {
    verify_private_parent(parent)?;
    let mut file = tempfile::tempfile_in(parent)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        // tempfile's Linux O_TMPFILE path inherits OpenOptions' default mode;
        // constrain this NEW anonymous empty inode before any payload write.
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    // FD-backed filesystem/capacity admission occurs BEFORE any payload write.
    admit_file(&file, bytes.len() as u64)?;
    copy_verified(&bytes, &mut file)?;
    // Explicitly release anonymous heap storage before the next resolver can
    // run, even while the returned same-byte File remains retained.
    drop(bytes);
    Ok(file)
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt as _;
    fn private_dir(parent: impl AsRef<Path>) -> Result<tempfile::TempDir> {
        Ok(tempfile::Builder::new()
            .permissions(std::fs::Permissions::from_mode(0o700))
            .tempdir_in(parent)?)
    }
    #[test]
    fn resolved_spool_roundtrip_is_anonymous_exact_and_released() -> Result<()> {
        use std::os::unix::fs::MetadataExt as _;
        let dir = private_dir(std::env::current_dir()?)?;
        let bytes: Vec<_> = (0..(3 * IO_BYTES + 7)).map(|i| (i % 251) as u8).collect();
        let expected = Sha256::digest(&bytes);
        let mut file = spool_in(bytes, dir.path())?;
        let fd = {
            use std::os::fd::AsRawFd as _;
            file.as_raw_fd()
        };
        let inode = file.metadata()?.ino();
        assert_eq!(file.metadata()?.nlink(), 0);
        assert_eq!(file.metadata()?.mode() & 0o777, 0o600);
        let mut actual = Vec::new();
        file.read_to_end(&mut actual)?;
        assert_eq!(Sha256::digest(&actual), expected);
        assert_eq!(std::fs::read_dir(dir.path())?.count(), 0);
        drop(file);
        // No named cache payload survives success or FD release.
        assert_eq!(std::fs::read_dir(dir.path())?.count(), 0);
        assert!(std::fs::metadata(format!("/proc/self/fd/{fd}"))
            .map(|m| m.ino() != inode)
            .unwrap_or(true));
        Ok(())
    }
    #[test]
    fn resolved_spool_private_parent_and_memory_filesystem_fail_closed() -> Result<()> {
        use std::os::unix::fs::{symlink, PermissionsExt as _};
        let dir = private_dir(std::env::current_dir()?)?;
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755))?;
        assert!(spool_in(vec![1, 2, 3], dir.path()).is_err());
        std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o700))?;
        let alias = dir.path().join("alias");
        symlink(dir.path(), &alias)?;
        assert!(spool_in(vec![1, 2, 3], &alias).is_err());
        std::fs::remove_file(alias)?;
        // /dev/shm is the actual Linux tmpfs; failure must happen beforewrite.
        let memory = private_dir("/dev/shm")?;
        let error = spool_in(vec![1, 2, 3], memory.path())
            .err()
            .context("accepted tmpfs")?;
        assert!(error.to_string().contains("memory-backed"));
        assert_eq!(std::fs::read_dir(memory.path())?.count(), 0);
        assert!(required_capacity(u64::MAX).is_err());
        Ok(())
    }
    #[test]
    fn resolved_spool_insufficient_capacity_fails_before_any_write() -> Result<()> {
        let dir = private_dir(std::env::current_dir()?)?;
        let file = tempfile::tempfile_in(dir.path())?;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
        // A tiny FD with an impossible requested capacity exercises the real
        // fstatvfs refusal without allocating or writing giant payloads.
        let error = admit_file(&file, u64::MAX - RESERVE_BYTES)
            .err()
            .context("accepted impossible disk request")?;
        assert!(error.to_string().contains("insufficient disk"));
        assert_eq!(file.metadata()?.len(), 0);
        assert_eq!(std::fs::read_dir(dir.path())?.count(), 0);
        Ok(())
    }

    #[test]
    fn resolved_spool_write_failure_leaves_no_named_payload() -> Result<()> {
        let dir = private_dir(std::env::current_dir()?)?;
        let mut file = File::options().write(true).open("/dev/full")?;
        assert!(copy_verified(&[1, 2, 3], &mut file).is_err());
        assert_eq!(std::fs::read_dir(dir.path())?.count(), 0);
        Ok(())
    }
}
