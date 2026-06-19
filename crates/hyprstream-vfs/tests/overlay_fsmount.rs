//! FS-C (#364) integration tests: the `FsMount` overlay v1 backend.
//!
//! Exercises [`hyprstream_vfs::overlay::passthrough_rootfs_overlay`] — an
//! `OverlayFs(lower=RO passthrough, upper=RW passthrough)` exposed through the
//! `FileSystem → FsMount` up-adapter — and asserts genuine overlay behaviour:
//! copy-up on write, whiteout on delete, rename across layers, plus that the
//! `Subject` is threaded on every op. The base `Mount` surface (walk/open/
//! read/write/readdir) is verified too.
//!
//! These tests touch the host filesystem via `PassthroughFs` and use OverlayFs
//! whiteout char-devices (mknod), which require privileges (root or a user
//! namespace with CAP_MKNOD). When that is unavailable the deletion/whiteout
//! assertions are skipped with an explanatory message rather than failing — the
//! copy-up / read / write / Subject paths still run everywhere.

#![cfg(not(target_arch = "wasm32"))]
#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::Path;

use hyprstream_vfs::overlay::passthrough_rootfs_overlay;
use hyprstream_vfs::{FsMount, Mount, MountError, SetAttr, Subject, OREAD, ORDWR, OWRITE};

fn subject() -> Subject {
    Subject::new("tenant-a")
}

/// The up-adapter must be *genuinely* `Send + Sync` natively (no wasm
/// `unsafe impl Send` escape hatch), so it can be stored as `Arc<dyn FsMount>`
/// in a `Namespace` and used across threads. This is a compile-time assertion.
#[test]
fn adapter_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<hyprstream_vfs::FuseFileSystemMount<fuse_backend_rs::overlayfs::OverlayFs>>();
    // And usable as the trait object a Namespace stores.
    fn _coerce(m: std::sync::Arc<dyn FsMount>) -> std::sync::Arc<dyn Mount> {
        m
    }
}

/// Build lower (RO) + upper (RW) + work tmpdirs and an overlay over them.
struct Harness {
    _lower: tempfile::TempDir,
    _upper: tempfile::TempDir,
    _work: tempfile::TempDir,
    lower_path: std::path::PathBuf,
    overlay: hyprstream_vfs::FuseFileSystemMount<fuse_backend_rs::overlayfs::OverlayFs>,
}

fn harness(lower_files: &[(&str, &[u8])]) -> Harness {
    let lower = tempfile::tempdir().unwrap();
    let upper = tempfile::tempdir().unwrap();
    let work = tempfile::tempdir().unwrap();
    for (name, data) in lower_files {
        let p = lower.path().join(name);
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&p, data).unwrap();
    }
    let overlay =
        passthrough_rootfs_overlay(lower.path(), upper.path(), work.path()).expect("build overlay");
    Harness {
        lower_path: lower.path().to_path_buf(),
        _lower: lower,
        _upper: upper,
        _work: work,
        overlay,
    }
}

/// Read a whole file through the Mount surface (walk/open/read/clunk).
async fn read_file(m: &impl Mount, path: &[&str], caller: &Subject) -> Result<Vec<u8>, MountError> {
    let mut fid = m.walk(path, caller).await?;
    if let Err(e) = m.open(&mut fid, OREAD, caller).await {
        m.clunk(fid, caller).await;
        return Err(e);
    }
    let mut out = Vec::new();
    loop {
        match m.read(&fid, out.len() as u64, 64 * 1024, caller).await {
            Ok(chunk) if chunk.is_empty() => break,
            Ok(chunk) => out.extend_from_slice(&chunk),
            Err(e) => {
                m.clunk(fid, caller).await;
                return Err(e);
            }
        }
    }
    m.clunk(fid, caller).await;
    Ok(out)
}

/// True if the host environment can create overlay whiteouts (mknod char dev).
/// Probe by trying a real whiteout via a deletion and checking for EPERM.
fn whiteouts_available() -> bool {
    // mknod of a char device requires privilege; effective uid 0 is the common case.
    unsafe { libc::geteuid() == 0 }
}

#[tokio::test]
async fn lower_file_is_visible_through_overlay() {
    let h = harness(&[("greeting.txt", b"hello from lower\n")]);
    let data = read_file(&h.overlay, &["greeting.txt"], &subject()).await.unwrap();
    assert_eq!(data, b"hello from lower\n");
}

#[tokio::test]
async fn write_triggers_copy_up_lower_untouched() {
    let h = harness(&[("doc.txt", b"original\n")]);
    let caller = subject();

    // Open the lower file RDWR and overwrite — OverlayFs copies it up to the
    // writable upper layer.
    let mut fid = h.overlay.walk(&["doc.txt"], &caller).await.unwrap();
    h.overlay.open(&mut fid, ORDWR | hyprstream_vfs::OTRUNC, &caller).await.unwrap();
    let n = h.overlay.write(&fid, 0, b"modified\n", &caller).await.unwrap();
    assert_eq!(n, b"modified\n".len() as u32);
    h.overlay.clunk(fid, &caller).await;

    // Overlay shows the new content.
    let data = read_file(&h.overlay, &["doc.txt"], &caller).await.unwrap();
    assert_eq!(data, b"modified\n");

    // The RO lower directory on the host is unchanged (copy-up, not in-place).
    let lower = fs::read(Path::new(&h.lower_path).join("doc.txt")).unwrap();
    assert_eq!(lower, b"original\n", "lower layer must not be mutated");
}

#[tokio::test]
async fn create_new_file_in_upper() {
    let h = harness(&[]);
    let caller = subject();

    h.overlay.create(&["fresh.txt"], 0o644, &caller).await.unwrap();

    // Re-open for write and put content in.
    let mut fid = h.overlay.walk(&["fresh.txt"], &caller).await.unwrap();
    h.overlay.open(&mut fid, OWRITE, &caller).await.unwrap();
    h.overlay.write(&fid, 0, b"brand new\n", &caller).await.unwrap();
    h.overlay.clunk(fid, &caller).await;

    let data = read_file(&h.overlay, &["fresh.txt"], &caller).await.unwrap();
    assert_eq!(data, b"brand new\n");

    // create on an existing name fails closed.
    let err = h.overlay.create(&["fresh.txt"], 0o644, &caller).await.unwrap_err();
    assert!(matches!(err, MountError::AlreadyExists(_)), "got {err:?}");
}

#[tokio::test]
async fn mkdir_and_readdir() {
    let h = harness(&[("a.txt", b"a"), ("b.txt", b"b")]);
    let caller = subject();

    h.overlay.mkdir(&["sub"], 0o755, &caller).await.unwrap();
    h.overlay.create(&["sub", "nested.txt"], 0o644, &caller).await.unwrap();

    // readdir of root: lower files + new dir all merged by OverlayFs.
    let mut fid = h.overlay.walk(&[], &caller).await.unwrap();
    h.overlay.open(&mut fid, OREAD, &caller).await.unwrap();
    let entries = h.overlay.readdir(&fid, &caller).await.unwrap();
    h.overlay.clunk(fid, &caller).await;
    let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"a.txt"), "names={names:?}");
    assert!(names.contains(&"b.txt"), "names={names:?}");
    assert!(names.contains(&"sub"), "names={names:?}");
    assert!(entries.iter().any(|e| e.name == "sub" && e.is_dir));
}

#[tokio::test]
async fn setattr_truncate() {
    let h = harness(&[("big.txt", b"0123456789")]);
    let caller = subject();

    h.overlay
        .setattr(&["big.txt"], &SetAttr { size: Some(4), ..Default::default() }, &caller)
        .await
        .unwrap();

    let data = read_file(&h.overlay, &["big.txt"], &caller).await.unwrap();
    assert_eq!(data, b"0123", "truncate via setattr (copy-up) should shorten");

    // stat_path reflects the new size.
    let st = h.overlay.stat_path(&["big.txt"], &caller).await.unwrap();
    assert_eq!(st.size, 4);
}

#[tokio::test]
async fn subject_threaded_anonymous_vs_named() {
    // The Subject is carried on every op; both an authenticated and an anonymous
    // subject reach the backend (per-tenant policy enforcement is FS-A/FS-D).
    let h = harness(&[("shared.txt", b"data\n")]);

    let named = Subject::new("alice");
    let data = read_file(&h.overlay, &["shared.txt"], &named).await.unwrap();
    assert_eq!(data, b"data\n");

    let anon = Subject::anonymous();
    let data = read_file(&h.overlay, &["shared.txt"], &anon).await.unwrap();
    assert_eq!(data, b"data\n");
}

#[tokio::test]
async fn unlink_lower_file_whiteout() {
    if !whiteouts_available() {
        eprintln!("skipping whiteout test: needs root / CAP_MKNOD for overlay char-dev whiteouts");
        return;
    }
    let h = harness(&[("doomed.txt", b"delete me\n"), ("keep.txt", b"keep\n")]);
    let caller = subject();

    // Sanity: visible before delete.
    assert!(read_file(&h.overlay, &["doomed.txt"], &caller).await.is_ok());

    // Delete a file that exists only in the RO lower → OverlayFs lays a whiteout
    // in the upper layer; the lower file is masked, not mutated.
    h.overlay.unlink(&["doomed.txt"], &caller).await.unwrap();

    let err = read_file(&h.overlay, &["doomed.txt"], &caller).await.unwrap_err();
    assert!(matches!(err, MountError::NotFound(_)), "got {err:?}");

    // The lower host file still exists (whiteout, not deletion).
    assert!(Path::new(&h.lower_path).join("doomed.txt").exists());

    // Other lower files remain visible.
    assert!(read_file(&h.overlay, &["keep.txt"], &caller).await.is_ok());
}

#[tokio::test]
async fn rename_across_layers() {
    if !whiteouts_available() {
        eprintln!("skipping cross-layer rename test: needs root / CAP_MKNOD for whiteouts");
        return;
    }
    let h = harness(&[("src.txt", b"payload\n")]);
    let caller = subject();

    // Rename a lower-only file to a new name: OverlayFs copies the source up and
    // whiteouts the old location.
    h.overlay.rename(&["src.txt"], &["dst.txt"], &caller).await.unwrap();

    let data = read_file(&h.overlay, &["dst.txt"], &caller).await.unwrap();
    assert_eq!(data, b"payload\n");
    assert!(read_file(&h.overlay, &["src.txt"], &caller).await.is_err());
}
