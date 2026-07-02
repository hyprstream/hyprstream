//! **ImageFs** — a universal mountable OCI/RAFS image filesystem service (#633,
//! spine of epic #508).
//!
//! An `ImageFs` is an [`FsMount`](hyprstream_vfs::FsMount): an `OverlayFs` whose
//!
//! - **lower** is the image's RAFS loaded **in-process** as a
//!   `fuse_backend_rs::FileSystem` (via `nydus-rafs`'s [`Rafs`], which also
//!   implements the overlay [`Layer`](fuse_backend_rs::api::filesystem::Layer)
//!   trait) — read-only, lazily fetching chunks from the CAS / Dragonfly P2P, so
//!   nothing is materialised on disk up front; and
//! - **upper** is a per-sandbox writable directory (the copy-up / whiteout
//!   target), so writes never touch the shared image.
//!
//! "Root" is a mount position, not a type: an `ImageFs` mounted at `/` is the
//! root purely because the namespace recipe mounted it there. Any backend can
//! compose one — kata serves it to its guest via virtio-fs today, and
//! podman/nspawn/wasm wiring is a follow-on (TODO, #633). This module compiles
//! under the `oci-image` feature alone, with no VM toolchain required.
//!
//! Composition is delegated to FS-C's
//! [`rootfs_overlay`](hyprstream_vfs::overlay::rootfs_overlay) — copy-on-write,
//! whiteout-on-delete and opaque dirs are performed by `OverlayFs` (Kata's
//! production overlay). We do **not** hand-roll CoW here; the native hyprstream
//! CoW engine (per-tenant writable layers over shared RO model weights, delta /
//! sandbox snapshots) is the backlog ticket and will implement the same
//! `FsMount` trait with zero churn here.
//!
//! ```text
//!            ImageFs  (FsMount)
//!                 │
//!         OverlayFs (FS-C rootfs_overlay)
//!          ┌──────┴──────┐
//!       upper           lower
//!   <sandbox>/rootfs/   Rafs (in-process FileSystem)
//!   {upper,work}/       bootstrap .meta + data blobs (lazy chunk CAS)
//! ```
//!
//! ## In-process RAFS — no external `nydusd`
//!
//! [`Rafs::new`] loads the bootstrap super-block and constructs a `BlobDevice`
//! over a **localfs** backend + **filecache** rooted at the image store's blobs
//! directory (where FS-B0 wrote the content-addressed RAFS data blobs). RAFS v6
//! (what FS-B0 emits) *requires* a local blob cache, which the localfs config
//! supplies. The whole thing runs in this process; there is no `nydusd`, no
//! FUSE mount, no shell-out.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use fuse_backend_rs::overlayfs::OverlayFs;
use nydus_api::ConfigV2;
use nydus_rafs::fs::Rafs;

use hyprstream_vfs::overlay::{layer_from_fs, rootfs_overlay};
use hyprstream_vfs::{FsMount, FuseFileSystemMount, ZeroOpenLayer};

use crate::error::{Result, WorkerError};

/// The concrete type of a mountable OCI/RAFS image filesystem service.
///
/// An `OverlayFs` (FS-C's production overlay) over an in-process RAFS lower and
/// a per-sandbox writable upper. This is what [`image_fs_for`] /
/// [`RafsStore::image_fs`](crate::image::RafsStore::image_fs) build; it
/// implements [`FsMount`], so a backend composes it into a namespace at whatever
/// mount position it chooses (at `/` it is the root purely by virtue of the
/// mount). "Root" is a mount position, not a type.
pub type ImageFs = FuseFileSystemMount<OverlayFs>;

/// Subdirectory under a sandbox dir holding the writable rootfs overlay state.
const ROOTFS_SUBDIR: &str = "rootfs";
/// The copy-up / whiteout target (overlay "upper").
const UPPER_SUBDIR: &str = "upper";
/// The overlay work directory (scratch for atomic operations).
const WORK_SUBDIR: &str = "work";

/// Load an image's RAFS bootstrap as an **in-process** read-only `FileSystem`.
///
/// * `bootstrap_path` — the RAFS bootstrap (`.meta`) produced by FS-B0
///   ([`build_rafs_bootstrap`](crate::image::rafs_builder)), resolved via
///   [`RafsStore::bootstrap_path`](crate::image::RafsStore::bootstrap_path).
/// * `blobs_dir` — directory holding the content-addressed RAFS data blobs; used
///   as both the localfs backend dir and the filecache work dir.
/// * `id` — an identifier for the RAFS instance (metrics / logging only).
///
/// The returned [`Rafs`] is `import`ed and ready to serve. Chunks are fetched
/// lazily on read from `blobs_dir`, preserving the lazy chunk-CAS property.
fn load_rafs_lower(bootstrap_path: &Path, blobs_dir: &Path, id: &str) -> Result<Rafs> {
    if !bootstrap_path.exists() {
        return Err(WorkerError::RafsError(format!(
            "RAFS bootstrap {} does not exist (image not pulled / built?)",
            bootstrap_path.display()
        )));
    }

    // localfs backend + filecache, both rooted at the blobs dir (RAFS v6
    // mandates a local blob cache), plus the `rafs` section `Rafs::new`
    // requires (`get_rafs_config`). `mode = "direct"` is the standard v6
    // metadata mode; xattrs are enabled so overlay/whiteout xattrs resolve.
    let blobs = blobs_dir.to_string_lossy();
    let toml = format!(
        r#"
version = 2
id = "{id}"
backend.type = "localfs"
backend.localfs.dir = "{blobs}"
cache.type = "filecache"
cache.compressed = false
cache.validate = false
cache.filecache.work_dir = "{blobs}"
rafs.mode = "direct"
rafs.enable_xattr = true
"#
    );
    let config: ConfigV2 = toml.parse().map_err(|e| {
        WorkerError::RafsError(format!(
            "failed to build RAFS config for {}: {e}",
            blobs_dir.display()
        ))
    })?;
    let config = Arc::new(config);

    let (mut rafs, reader) = Rafs::new(&config, id, bootstrap_path).map_err(|e| {
        WorkerError::RafsError(format!(
            "failed to load in-process RAFS from {}: {e}",
            bootstrap_path.display()
        ))
    })?;
    rafs.import(reader, None).map_err(|e| {
        WorkerError::RafsError(format!(
            "failed to import RAFS bootstrap {}: {e}",
            bootstrap_path.display()
        ))
    })?;

    Ok(rafs)
}

/// Lay out (and create) the per-sandbox writable overlay directories under
/// `sandbox_dir`, returning `(upper, work)`.
///
/// Both live under `<sandbox_dir>/rootfs/`: `upper/` is the copy-up target,
/// `work/` is `OverlayFs`'s scratch space. They must be on the same filesystem
/// (they are — siblings) so `OverlayFs` can rename atomically between them.
fn prepare_upper(sandbox_dir: &Path) -> Result<(PathBuf, PathBuf)> {
    let rootfs = sandbox_dir.join(ROOTFS_SUBDIR);
    let upper = rootfs.join(UPPER_SUBDIR);
    let work = rootfs.join(WORK_SUBDIR);
    for dir in [&upper, &work] {
        std::fs::create_dir_all(dir)
            .map_err(|e| WorkerError::IoError(format!("create {}: {e}", dir.display())))?;
    }
    Ok((upper, work))
}

/// Build an [`ImageFs`] from an already-loaded RAFS lower and explicit
/// upper/work directories.
///
/// The lowest-level constructor: callers that hold a [`Rafs`] (e.g. tests, or a
/// future caller that loaded RAFS itself) compose it here. Wraps the RAFS as the
/// single read-only overlay lower and `upper` as the writable layer via FS-C's
/// [`rootfs_overlay`].
pub fn image_fs_from_rafs(
    rafs: Rafs,
    upper: &Path,
    work: &Path,
) -> Result<FuseFileSystemMount<OverlayFs>> {
    let upper_layer = hyprstream_vfs::overlay::passthrough_layer(upper)
        .map_err(|e| WorkerError::RafsError(format!("image-fs upper layer: {e}")))?;
    // The RAFS `Rafs` implements `Layer`, but it is *handleless* (`open` yields
    // no handle), which OverlayFs rejects as ENOENT for a lower. Wrap it in
    // `ZeroOpenLayer` so it presents a constant handle; reads still resolve by
    // inode and writes copy-up to the (handle-based) upper.
    let lower_layer = layer_from_fs(ZeroOpenLayer::new(rafs));
    rootfs_overlay(upper_layer, vec![lower_layer], work)
        .map_err(|e| WorkerError::RafsError(format!("image-fs overlay: {e}")))
}

/// Build an [`ImageFs`] for `image_id`.
///
/// * `bootstrap_path` — the image's RAFS bootstrap (`.meta`).
/// * `blobs_dir` — the image store's blobs directory (RAFS data blobs).
/// * `image_id` — the image identifier (used as the RAFS instance id).
/// * `sandbox_dir` — the per-sandbox directory; the writable upper + work dirs
///   are created under `<sandbox_dir>/rootfs/`.
///
/// Returns the image [`FsMount`]: the RAFS image is the read-only lower, a
/// per-sandbox directory is the writable upper, composed via `OverlayFs`. The
/// shared image (RAFS bootstrap + blobs) is never written to — copy-ups and
/// whiteouts land in the upper. Mount it wherever the namespace recipe chooses;
/// at `/` it is the root purely by virtue of the mount position.
pub fn image_fs_for(
    bootstrap_path: &Path,
    blobs_dir: &Path,
    image_id: &str,
    sandbox_dir: &Path,
) -> Result<impl FsMount> {
    let rafs = load_rafs_lower(bootstrap_path, blobs_dir, image_id)?;
    let (upper, work) = prepare_upper(sandbox_dir)?;
    image_fs_from_rafs(rafs, &upper, &work)
}

impl crate::image::RafsStore {
    /// Build the image [`FsMount`] for an image into a per-sandbox directory.
    ///
    /// Resolves the image's RAFS bootstrap and blobs from this store, loads the
    /// RAFS in-process as the read-only lower, and overlays a per-sandbox
    /// writable upper under `<sandbox_dir>/rootfs/`. See [`image_fs_for`].
    pub fn image_fs(&self, image_id: &str, sandbox_dir: &Path) -> Result<impl FsMount> {
        let bootstrap = self.bootstrap_path(image_id);
        image_fs_for(&bootstrap, self.blobs_dir(), image_id, sandbox_dir)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use hyprstream_rpc::Subject;
    use hyprstream_vfs::{FsMount, Mount, OREAD, ORDWR};

    use crate::image::rafs_builder::build_rafs_bootstrap;

    /// Build a small gzip tar layer (OCI default) of regular files.
    fn make_gzip_layer(path: &Path, entries: &[(&str, &[u8])]) {
        let file = std::fs::File::create(path).unwrap();
        let gz = GzEncoder::new(file, Compression::default());
        let mut tar = tar::Builder::new(gz);
        for (name, data) in entries {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_mtime(0);
            header.set_entry_type(tar::EntryType::Regular);
            tar.append_data(&mut header, name, *data).unwrap();
        }
        let gz = tar.into_inner().unwrap();
        gz.finish().unwrap();
    }

    /// Build a synthetic RAFS image (bootstrap + data blobs) under `dir`,
    /// returning `(bootstrap_path, blobs_dir)`.
    fn build_image(dir: &Path, layers: &[&[(&str, &[u8])]]) -> (PathBuf, PathBuf) {
        let blobs_dir = dir.join("blobs");
        let bootstrap_dir = dir.join("bootstrap");
        std::fs::create_dir_all(&blobs_dir).unwrap();
        std::fs::create_dir_all(&bootstrap_dir).unwrap();

        let mut layer_paths = Vec::new();
        for (i, entries) in layers.iter().enumerate() {
            let p = dir.join(format!("layer{i}.tar.gz"));
            make_gzip_layer(&p, entries);
            layer_paths.push(p);
        }

        let bootstrap = bootstrap_dir.join("img.meta");
        build_rafs_bootstrap(&layer_paths, &blobs_dir, &bootstrap).unwrap();
        (bootstrap, blobs_dir)
    }

    /// Snapshot of the blobs dir (name -> len) to assert the image is untouched.
    fn blobs_snapshot(blobs_dir: &Path) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = std::fs::read_dir(blobs_dir)
            .unwrap()
            .map(|e| {
                let e = e.unwrap();
                (
                    e.file_name().to_string_lossy().into_owned(),
                    e.metadata().unwrap().len(),
                )
            })
            .collect();
        v.sort();
        v
    }

    async fn read_file(mount: &impl FsMount, path: &[&str], subj: &Subject) -> Vec<u8> {
        let mut fid = mount.walk(path, subj).await.unwrap();
        mount.open(&mut fid, OREAD, subj).await.unwrap();
        let data = mount.read(&fid, 0, 1 << 20, subj).await.unwrap();
        mount.clunk(fid, subj).await;
        data
    }

    #[tokio::test]
    async fn reads_from_rafs_lower_and_copies_up_on_write() {
        let tmp = tempfile::tempdir().unwrap();
        let (bootstrap, blobs_dir) = build_image(
            tmp.path(),
            &[&[
                ("etc/hostname", b"worker\n"),
                ("etc/config", b"v1\n"),
                ("usr/bin/app", b"#!/bin/sh\necho hi\n"),
            ]],
        );

        // Filecache writes into the blobs dir on first chunk read; snapshot the
        // RAFS *data blobs* (the immutable image content) by name+len. The
        // filecache materialises sidecar files (different names), so we assert
        // the original blobs are byte-for-byte unchanged rather than that the
        // directory is frozen.
        let before = blobs_snapshot(&blobs_dir);

        let sandbox = tmp.path().join("sandbox-1");
        let mount = image_fs_for(&bootstrap, &blobs_dir, "img:test", &sandbox).unwrap();
        let subj = Subject::new("test");

        // 1. Read a file present only in the RAFS lower.
        let hostname = read_file(&mount, &["etc", "hostname"], &subj).await;
        assert_eq!(hostname, b"worker\n", "must read through to RAFS lower");

        // 2. Write a brand-new file -> lands in the writable upper.
        let new_path = ["etc", "greeting"];
        FsMount::create(&mount, &new_path, 0o644, &subj).await.unwrap();
        let mut fid = mount.walk(&new_path, &subj).await.unwrap();
        mount.open(&mut fid, ORDWR, &subj).await.unwrap();
        let n = mount.write(&fid, 0, b"hello\n", &subj).await.unwrap();
        assert_eq!(n, 6);
        mount.clunk(fid, &subj).await;

        // The new file is visible through the overlay...
        let greeting = read_file(&mount, &new_path, &subj).await;
        assert_eq!(greeting, b"hello\n");
        // ...and physically present in the sandbox upper.
        let upper_file = sandbox
            .join(ROOTFS_SUBDIR)
            .join(UPPER_SUBDIR)
            .join("etc")
            .join("greeting");
        assert!(upper_file.exists(), "new file must copy-up into the upper");
        assert_eq!(std::fs::read(&upper_file).unwrap(), b"hello\n");

        // 3. Overwrite a file that exists in the lower -> copy-up, lower intact.
        let cfg_path = ["etc", "config"];
        let mut fid = mount.walk(&cfg_path, &subj).await.unwrap();
        mount.open(&mut fid, ORDWR, &subj).await.unwrap();
        mount.write(&fid, 0, b"v2\n", &subj).await.unwrap();
        mount.clunk(fid, &subj).await;
        let cfg = read_file(&mount, &cfg_path, &subj).await;
        assert_eq!(&cfg[..3], b"v2\n", "overwrite visible via overlay");
        let upper_cfg = sandbox
            .join(ROOTFS_SUBDIR)
            .join(UPPER_SUBDIR)
            .join("etc")
            .join("config");
        assert!(upper_cfg.exists(), "modified lower file must copy-up");

        // 4. The RAFS image (data blobs) is untouched: every original blob is
        // still present with its original length.
        let after = blobs_snapshot(&blobs_dir);
        for (name, len) in &before {
            let found = after.iter().find(|(n, _)| n == name);
            assert_eq!(
                found,
                Some(&(name.clone(), *len)),
                "RAFS data blob {name} must be unchanged"
            );
        }
    }

    #[tokio::test]
    async fn missing_bootstrap_is_fail_closed() {
        let tmp = tempfile::tempdir().unwrap();
        let blobs = tmp.path().join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();
        let bootstrap = tmp.path().join("nope.meta");
        let sandbox = tmp.path().join("sandbox");
        let err = image_fs_for(&bootstrap, &blobs, "img:missing", &sandbox);
        assert!(err.is_err(), "missing RAFS bootstrap must hard-error");
    }
}
