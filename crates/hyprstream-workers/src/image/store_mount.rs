//! `RafsStoreMount` — projects [`RafsStore`]'s on-disk CAS as a 9P/VFS
//! [`Mount`] (#652, follow-on to #633/#641's `ImageFs`).
//!
//! `RafsStore` (`store.rs`) is the *storage* layer: the on-disk `blobs/`,
//! `bootstrap/`, `refs/` directories driven directly via `std::fs`. That's a
//! side channel outside the namespace — nobody can walk the image store the
//! way they can walk everything else in the system. This mount exposes that
//! storage layer itself as a browsable/readable tree, so the CAS is
//! addressable as a namespace rather than a bespoke on-disk layout.
//!
//! **Out of scope** (per #652): the composed image *filesystem* a guest
//! mounts (`ImageFs`, #633/#641) and the copy-up write policy (#370) — both
//! are separate concerns from the storage-layer projection here.
//!
//! ## Layout
//!
//! ```text
//! /                              (root dir)
//! ├── refs/                      tag → image-metadata symlink targets
//! │   └── <normalized-ref>       file: the pointed-to ImageMetadata JSON
//! ├── bootstrap/                 RAFS bootstrap + per-image metadata files
//! │   └── <name>                 file: raw bytes (`.meta` bootstrap or `.json` metadata)
//! └── blobs/                     content-addressed layer/config blobs
//!     └── <digest>               file: raw blob bytes, addressable either by
//!                                 the on-disk filename (`sha256_<hex>`) or by
//!                                 the OCI digest form (`sha256:<hex>`)
//! ```
//!
//! This mirrors `RafsStore`'s own on-disk layout 1:1 (see its module docs) —
//! deliberately: the goal is namespace *parity* with the existing storage
//! layout, not a redesign of it.
//!
//! ## Read-only by design
//!
//! Ingest (`pull`/`gc`/`remove`) already has a well-formed API on `RafsStore`
//! itself and involves multi-step invariants (manifest fetch, RAFS bootstrap
//! build, ref symlink creation) that don't map cleanly onto single `Tcreate`/
//! `Twrite` calls. This mount only implements the read path
//! (`walk`/`open`/`read`/`readdir`/`stat`); `write`/`create` return
//! [`MountError::NotSupported`]. Writes to the CAS continue to go through
//! `RafsStore`'s existing ingest API — consistent with #370 (copy-up write
//! policy) being tracked separately and out of scope here.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;

use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat, Subject};

use super::store::{digest_to_filename, RafsStore};

/// Plan 9 `DMDIR` qtype bit (matches the convention used elsewhere in this
/// codebase's synthetic mounts, e.g. `hyprstream_vfs::injected::SyntheticMount`).
const QTDIR: u8 = 0x80;

/// The three top-level directories this mount projects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopDir {
    Refs,
    Bootstrap,
    Blobs,
}

impl TopDir {
    fn name(self) -> &'static str {
        match self {
            Self::Refs => "refs",
            Self::Bootstrap => "bootstrap",
            Self::Blobs => "blobs",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "refs" => Some(Self::Refs),
            "bootstrap" => Some(Self::Bootstrap),
            "blobs" => Some(Self::Blobs),
            _ => None,
        }
    }
}

/// Which node a fid resolved to.
#[derive(Clone, Debug)]
enum NodeKind {
    /// `/` — the mount root.
    Root,
    /// `/refs/`, `/bootstrap/`, or `/blobs/`.
    TopDir(TopDir),
    /// A leaf file under one of the top-level dirs, along with its resolved
    /// on-disk absolute path.
    File { dir: TopDir, on_disk: PathBuf },
}

/// Fid state for [`RafsStoreMount`].
struct StoreFid {
    kind: NodeKind,
}

/// A read-only 9P/VFS [`Mount`] projecting a [`RafsStore`]'s CAS as a
/// browsable file tree. See module docs for the layout.
pub struct RafsStoreMount {
    store: Arc<RafsStore>,
}

impl RafsStoreMount {
    /// Create a new mount over `store`.
    pub fn new(store: Arc<RafsStore>) -> Self {
        Self { store }
    }

    /// Directory backing a [`TopDir`].
    fn dir_path(&self, dir: TopDir) -> &Path {
        match dir {
            TopDir::Refs => self.store.refs_dir(),
            TopDir::Bootstrap => self.store.bootstrap_dir(),
            TopDir::Blobs => self.store.blobs_dir(),
        }
    }

    /// Resolve a leaf name under `dir` to its on-disk path, checking
    /// existence. `blobs/` additionally accepts the raw OCI digest form
    /// (`sha256:...`), translating it via the same convention `RafsStore`
    /// itself uses (`digest_to_filename`).
    fn resolve_leaf(&self, dir: TopDir, name: &str) -> Option<PathBuf> {
        let base = self.dir_path(dir);

        // Direct on-disk filename match first (works for refs/, bootstrap/,
        // and blobs/ entries already given in their sanitized form).
        let direct = base.join(name);
        if direct.exists() {
            return Some(direct);
        }

        // `blobs/sha256:<hex>` — accept the OCI digest form too.
        if dir == TopDir::Blobs && name.contains(':') {
            let sanitized = base.join(digest_to_filename(name));
            if sanitized.exists() {
                return Some(sanitized);
            }
        }

        None
    }

    /// For `refs/<name>`, the served content is the *pointed-to* metadata
    /// JSON (refs are symlinks to a metadata file in `bootstrap/`), not the
    /// symlink's own (empty) bytes.
    fn read_leaf_bytes(&self, dir: TopDir, on_disk: &Path) -> Result<Vec<u8>, MountError> {
        let path = if dir == TopDir::Refs {
            std::fs::read_link(on_disk)
                .map_err(|e| MountError::Io(format!("failed to resolve ref symlink: {e}")))?
        } else {
            on_disk.to_path_buf()
        };
        std::fs::read(&path).map_err(|e| MountError::Io(format!("failed to read {}: {e}", path.display())))
    }

    /// List the entries of a top-level directory as `DirEntry`s.
    fn list_dir(&self, dir: TopDir) -> Result<Vec<DirEntry>, MountError> {
        let base = self.dir_path(dir);
        if !base.exists() {
            return Ok(Vec::new());
        }
        let mut entries = Vec::new();
        let read_dir = std::fs::read_dir(base)
            .map_err(|e| MountError::Io(format!("failed to read {}: {e}", base.display())))?;
        for entry in read_dir {
            let entry = entry.map_err(|e| MountError::Io(e.to_string()))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let size = self.leaf_size(dir, &entry.path()).unwrap_or(0);
            entries.push(DirEntry {
                name,
                is_dir: false,
                size,
                stat: None,
            });
        }
        Ok(entries)
    }

    /// Size in bytes for a leaf entry (used by both `readdir` and `stat`).
    fn leaf_size(&self, dir: TopDir, on_disk: &Path) -> Option<u64> {
        let path = if dir == TopDir::Refs {
            std::fs::read_link(on_disk).ok()?
        } else {
            on_disk.to_path_buf()
        };
        std::fs::metadata(&path).ok().map(|m| m.len())
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
impl Mount for RafsStoreMount {
    async fn walk(&self, components: &[&str], _caller: &Subject) -> Result<Fid, MountError> {
        // NOTE on `Subject`: threaded through per the Mount contract, but this
        // mount does not itself authorize — same convention as `ExecMount`
        // (#547 MAC is the eventual enforcement point).
        let kind = match components {
            [] => NodeKind::Root,
            [top] => {
                let dir = TopDir::parse(top)
                    .ok_or_else(|| MountError::NotFound(components.join("/")))?;
                NodeKind::TopDir(dir)
            }
            [top, name] => {
                let dir = TopDir::parse(top)
                    .ok_or_else(|| MountError::NotFound(components.join("/")))?;
                let on_disk = self
                    .resolve_leaf(dir, name)
                    .ok_or_else(|| MountError::NotFound(components.join("/")))?;
                NodeKind::File { dir, on_disk }
            }
            _ => return Err(MountError::NotFound(components.join("/"))),
        };

        Ok(Fid::new(StoreFid { kind }))
    }

    async fn open(&self, _fid: &mut Fid, mode: u8, _caller: &Subject) -> Result<(), MountError> {
        // Read-only: reject any write/rdwr open (mirrors `SyntheticMount`).
        if mode & 0x03 != 0 {
            return Err(MountError::PermissionDenied(
                "RafsStoreMount is read-only".into(),
            ));
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
        let inner = fid
            .downcast_ref::<StoreFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        let data = match &inner.kind {
            NodeKind::File { dir, on_disk } => self.read_leaf_bytes(*dir, on_disk)?,
            NodeKind::Root | NodeKind::TopDir(_) => {
                return Err(MountError::IsDirectory("use readdir".into()));
            }
        };

        let start = (offset as usize).min(data.len());
        let end = start.saturating_add(count as usize).min(data.len());
        Ok(data[start..end].to_vec())
    }

    async fn write(
        &self,
        _fid: &Fid,
        _offset: u64,
        _data: &[u8],
        _caller: &Subject,
    ) -> Result<u32, MountError> {
        Err(MountError::NotSupported(
            "RafsStoreMount is read-only; ingest goes through RafsStore's pull/gc/remove API"
                .into(),
        ))
    }

    async fn readdir(&self, fid: &Fid, _caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        let inner = fid
            .downcast_ref::<StoreFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        match &inner.kind {
            NodeKind::Root => Ok([TopDir::Refs, TopDir::Bootstrap, TopDir::Blobs]
                .into_iter()
                .map(|d| DirEntry {
                    name: d.name().to_owned(),
                    is_dir: true,
                    size: 0,
                    stat: None,
                })
                .collect()),
            NodeKind::TopDir(dir) => self.list_dir(*dir),
            NodeKind::File { .. } => Err(MountError::NotDirectory(format!("{:?}", inner.kind))),
        }
    }

    async fn stat(&self, fid: &Fid, _caller: &Subject) -> Result<Stat, MountError> {
        let inner = fid
            .downcast_ref::<StoreFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;

        let (name, qtype, size) = match &inner.kind {
            NodeKind::Root => (String::new(), QTDIR, 0),
            NodeKind::TopDir(dir) => (dir.name().to_owned(), QTDIR, 0),
            NodeKind::File { dir, on_disk } => {
                let name = on_disk
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                let size = self.leaf_size(*dir, on_disk).unwrap_or(0);
                (name, 0, size)
            }
        };

        Ok(Stat::unknown_qid(qtype, size, name, 0))
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::config::ImageConfig;

    fn subject() -> Subject {
        Subject::anonymous()
    }

    /// Build a `RafsStore` rooted at a fresh tempdir and materialize a fake
    /// pulled image (ref symlink + metadata + one blob) directly on disk,
    /// without going through `pull()` (which needs network access). This
    /// exercises the mount's read path against the exact on-disk shapes
    /// `RafsStore` itself produces.
    fn make_store_with_fake_image() -> (tempfile::TempDir, Arc<RafsStore>, &'static str) {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let config = ImageConfig {
            blobs_dir: root.join("blobs"),
            bootstrap_dir: root.join("bootstrap"),
            refs_dir: root.join("refs"),
            cache_dir: root.join("cache"),
            ..Default::default()
        };
        for dir in [&config.blobs_dir, &config.bootstrap_dir, &config.refs_dir, &config.cache_dir] {
            std::fs::create_dir_all(dir).unwrap();
        }

        let image_id = "sha256:deadbeef";
        let digest_filename = digest_to_filename(image_id); // "sha256_deadbeef"

        // A layer blob, content-addressed by its (fake) digest.
        std::fs::write(config.blobs_dir.join(&digest_filename), b"blob-bytes").unwrap();

        // Metadata file in bootstrap/, as `pull()` writes it.
        let metadata = super::super::store::ImageMetadata {
            image_ref: "example.com/repo:tag".to_owned(),
            image_id: image_id.to_owned(),
            config_digest: image_id.to_owned(),
            layers: vec![image_id.to_owned()],
            host: "example.com".to_owned(),
            repository: "repo".to_owned(),
        };
        let metadata_path = config.bootstrap_dir.join(format!("{digest_filename}.json"));
        std::fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();

        // Ref symlink -> metadata file, as `pull()` creates it.
        let ref_name = "example.com_repo_tag";
        std::os::unix::fs::symlink(&metadata_path, config.refs_dir.join(ref_name)).unwrap();

        let store = Arc::new(RafsStore::new(config).unwrap());
        (tmp, store, "example.com_repo_tag")
    }

    #[tokio::test]
    async fn root_lists_three_top_dirs() {
        let (_tmp, store, _ref_name) = make_store_with_fake_image();
        let mount = RafsStoreMount::new(store);

        let fid = mount.walk(&[], &subject()).await.unwrap();
        let entries = mount.readdir(&fid, &subject()).await.unwrap();
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert!(names.contains(&"refs"));
        assert!(names.contains(&"bootstrap"));
        assert!(names.contains(&"blobs"));
        assert!(entries.iter().all(|e| e.is_dir));
    }

    #[tokio::test]
    async fn refs_dir_lists_the_ref_and_reads_pointed_metadata() {
        let (_tmp, store, ref_name) = make_store_with_fake_image();
        let mount = RafsStoreMount::new(store);

        let dir_fid = mount.walk(&["refs"], &subject()).await.unwrap();
        let entries = mount.readdir(&dir_fid, &subject()).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, ref_name);

        let mut leaf_fid = mount.walk(&["refs", ref_name], &subject()).await.unwrap();
        mount.open(&mut leaf_fid, 0, &subject()).await.unwrap();
        let data = mount.read(&leaf_fid, 0, 4096, &subject()).await.unwrap();
        let text = String::from_utf8(data).unwrap();
        assert!(text.contains("example.com/repo:tag"), "got: {text}");
    }

    #[tokio::test]
    async fn blobs_addressable_by_on_disk_name_and_oci_digest() {
        let (_tmp, store, _ref_name) = make_store_with_fake_image();
        let mount = RafsStoreMount::new(store);

        // On-disk sanitized name.
        let mut fid1 = mount.walk(&["blobs", "sha256_deadbeef"], &subject()).await.unwrap();
        mount.open(&mut fid1, 0, &subject()).await.unwrap();
        let data1 = mount.read(&fid1, 0, 4096, &subject()).await.unwrap();
        assert_eq!(data1, b"blob-bytes");

        // Raw OCI digest form.
        let mut fid2 = mount.walk(&["blobs", "sha256:deadbeef"], &subject()).await.unwrap();
        mount.open(&mut fid2, 0, &subject()).await.unwrap();
        let data2 = mount.read(&fid2, 0, 4096, &subject()).await.unwrap();
        assert_eq!(data2, b"blob-bytes");
    }

    #[tokio::test]
    async fn bootstrap_dir_lists_metadata_file() {
        let (_tmp, store, _ref_name) = make_store_with_fake_image();
        let mount = RafsStoreMount::new(store);

        let fid = mount.walk(&["bootstrap"], &subject()).await.unwrap();
        let entries = mount.readdir(&fid, &subject()).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].name.ends_with(".json"));
    }

    #[tokio::test]
    async fn read_respects_offset_and_count() {
        let (_tmp, store, _ref_name) = make_store_with_fake_image();
        let mount = RafsStoreMount::new(store);

        let mut fid = mount.walk(&["blobs", "sha256_deadbeef"], &subject()).await.unwrap();
        mount.open(&mut fid, 0, &subject()).await.unwrap();
        let data = mount.read(&fid, 5, 3, &subject()).await.unwrap();
        assert_eq!(data, b"byt"); // "blob-bytes"[5..8]
    }

    #[tokio::test]
    async fn unknown_top_dir_not_found() {
        let (_tmp, store, _ref_name) = make_store_with_fake_image();
        let mount = RafsStoreMount::new(store);
        let result = mount.walk(&["nope"], &subject()).await;
        assert!(matches!(result, Err(MountError::NotFound(_))));
    }

    #[tokio::test]
    async fn unknown_blob_not_found() {
        let (_tmp, store, _ref_name) = make_store_with_fake_image();
        let mount = RafsStoreMount::new(store);
        let result = mount.walk(&["blobs", "sha256:missing"], &subject()).await;
        assert!(matches!(result, Err(MountError::NotFound(_))));
    }

    #[tokio::test]
    async fn write_is_rejected() {
        let (_tmp, store, _ref_name) = make_store_with_fake_image();
        let mount = RafsStoreMount::new(store);
        let mut fid = mount.walk(&["blobs", "sha256_deadbeef"], &subject()).await.unwrap();
        mount.open(&mut fid, 0, &subject()).await.unwrap();
        let result = mount.write(&fid, 0, b"x", &subject()).await;
        assert!(matches!(result, Err(MountError::NotSupported(_))));
    }

    #[tokio::test]
    async fn open_for_write_rejected() {
        let (_tmp, store, _ref_name) = make_store_with_fake_image();
        let mount = RafsStoreMount::new(store);
        let mut fid = mount.walk(&["blobs", "sha256_deadbeef"], &subject()).await.unwrap();
        let result = mount.open(&mut fid, hyprstream_vfs::OWRITE, &subject()).await;
        assert!(matches!(result, Err(MountError::PermissionDenied(_))));
    }

    #[tokio::test]
    async fn stat_root_is_dir() {
        let (_tmp, store, _ref_name) = make_store_with_fake_image();
        let mount = RafsStoreMount::new(store);
        let fid = mount.walk(&[], &subject()).await.unwrap();
        let st = mount.stat(&fid, &subject()).await.unwrap();
        assert_eq!(st.qtype, QTDIR);
    }

    #[tokio::test]
    async fn stat_blob_reports_size() {
        let (_tmp, store, _ref_name) = make_store_with_fake_image();
        let mount = RafsStoreMount::new(store);
        let fid = mount.walk(&["blobs", "sha256_deadbeef"], &subject()).await.unwrap();
        let st = mount.stat(&fid, &subject()).await.unwrap();
        assert_eq!(st.size, b"blob-bytes".len() as u64);
        assert_eq!(st.qtype, 0);
    }
}
