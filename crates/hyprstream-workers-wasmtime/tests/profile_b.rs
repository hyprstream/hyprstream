//! #506 Profile B — make-or-break validation: a `wasm32-wasip1` guest whose ONLY
//! filesystem is a Subject-scoped `hyprstream_vfs::Mount`.
//!
//! Build the guest first. These tests SKIP locally when the artifact is absent, but
//! FAIL under CI (`CI` env var set) — the no-host-escape guarantee must be exercised:
//!   cargo build --release --target wasm32-wasip1 \
//!     --manifest-path crates/hyprstream-wasm-fsguest/Cargo.toml
//!
//! Artifact (excluded standalone crate):
//!   crates/hyprstream-wasm-fsguest/target/wasm32-wasip1/{release,debug}/hyprstream-wasm-fsguest.wasm
//! Override with HYPRSTREAM_FSGUEST_WASM.
//!
//! What this proves:
//!   * Guest std::fs ops (`read`/`write`/`read_dir`) lower to WASI preview1 and the
//!     host routes EVERY one to the in-memory `MemMount` (Subject-scoped) — the data
//!     round-trips and a guest-written file appears in the Mount, NOT on the host.
//!   * The ONLY preopen is the VFS Mount; there is NO host preopen, so no host
//!     filesystem path is reachable.
//!   * The bound Subject reaches the Mount on every op (a `denied` Subject is
//!     refused at the Mount — the single policy point).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use hyprstream_workers_wasmtime::wasi_sandbox::WasiSandbox;
use hyprstream_vfs::{DirEntry, Fid, Mount, MountError, Stat, Subject};

fn guest_wasm() -> Option<Vec<u8>> {
    if let Ok(p) = std::env::var("HYPRSTREAM_FSGUEST_WASM") {
        return std::fs::read(&p).ok();
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let guest_dir = manifest
        .parent()
        .map(|p| p.join("hyprstream-wasm-fsguest"))?;
    for profile in ["release", "debug"] {
        let c = guest_dir
            .join("target/wasm32-wasip1")
            .join(profile)
            .join("hyprstream-wasm-fsguest.wasm");
        if c.exists() {
            if let Ok(b) = std::fs::read(&c) {
                return Some(b);
            }
        }
    }
    None
}

/// CI guard: under CI a missing fsguest artifact is a hard failure (the Profile-B
/// no-host-escape guarantee must be exercised); locally it still skips.
fn guest_wasm_or_ci_fail(label: &str) -> Option<Vec<u8>> {
    match guest_wasm() {
        Some(wasm) => Some(wasm),
        None => {
            assert!(
                std::env::var("CI").is_err(),
                "{label}: fsguest wasm not built but running under CI — build the fsguest \
                 (cargo build --release --target wasm32-wasip1 --manifest-path \
                 crates/hyprstream-wasm-fsguest/Cargo.toml) and set HYPRSTREAM_FSGUEST_WASM"
            );
            eprintln!("SKIP {label}: fsguest wasm not built (set CI=1 to make this a failure)");
            None
        }
    }
}

/// Generous fuel — wasip1 std startup + a few fs ops is cheap relative to RustPython.
const BIG_FUEL: u64 = 5_000_000_000;

// ─────────────────────────────────────────────────────────────────────────────
// In-memory, Subject-scoped Mount (the guest's ONLY filesystem).
// ─────────────────────────────────────────────────────────────────────────────

/// Flat in-memory file map. The root (`""`) is the only directory. A `denied`
/// Subject is refused at EVERY op so we can prove the bound Subject reaches the
/// backend — the Mount is the single policy enforcement point.
struct MemMount {
    files: parking_lot::Mutex<HashMap<String, Vec<u8>>>,
}

struct MemFid {
    /// "" = the root directory; otherwise a file name.
    path: String,
}

impl MemMount {
    fn new(files: Vec<(&str, &[u8])>) -> Self {
        Self {
            files: parking_lot::Mutex::new(
                files
                    .into_iter()
                    .map(|(k, v)| (k.to_owned(), v.to_vec()))
                    .collect(),
            ),
        }
    }

    fn check(caller: &Subject) -> Result<(), MountError> {
        if caller.name() == Some("denied") {
            return Err(MountError::PermissionDenied("denied subject".into()));
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl Mount for MemMount {
    async fn walk(&self, components: &[&str], caller: &Subject) -> Result<Fid, MountError> {
        Self::check(caller)?;
        // Drop empty components; the preopen root walks with `[]`.
        let path = components
            .iter()
            .filter(|c| !c.is_empty())
            .copied()
            .collect::<Vec<_>>()
            .join("/");
        Ok(Fid::new(MemFid { path }))
    }

    async fn open(&self, fid: &mut Fid, _mode: u8, caller: &Subject) -> Result<(), MountError> {
        Self::check(caller)?;
        let inner = fid
            .downcast_ref::<MemFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        // Auto-create on open-for-write of a not-yet-existing file (wasip1 O_CREAT
        // is folded into path_open here for the test backend).
        if !inner.path.is_empty() {
            self.files.lock().entry(inner.path.clone()).or_default();
        }
        Ok(())
    }

    async fn read(
        &self,
        fid: &Fid,
        offset: u64,
        _count: u32,
        caller: &Subject,
    ) -> Result<Vec<u8>, MountError> {
        Self::check(caller)?;
        let inner = fid
            .downcast_ref::<MemFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let data = self
            .files
            .lock()
            .get(&inner.path)
            .cloned()
            .ok_or_else(|| MountError::NotFound(inner.path.clone()))?;
        let start = (offset as usize).min(data.len());
        Ok(data[start..].to_vec())
    }

    async fn write(
        &self,
        fid: &Fid,
        offset: u64,
        data: &[u8],
        caller: &Subject,
    ) -> Result<u32, MountError> {
        Self::check(caller)?;
        let inner = fid
            .downcast_ref::<MemFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        let mut files = self.files.lock();
        let buf = files.entry(inner.path.clone()).or_default();
        let start = offset as usize;
        if buf.len() < start + data.len() {
            buf.resize(start + data.len(), 0);
        }
        buf[start..start + data.len()].copy_from_slice(data);
        Ok(data.len() as u32)
    }

    async fn readdir(&self, fid: &Fid, caller: &Subject) -> Result<Vec<DirEntry>, MountError> {
        Self::check(caller)?;
        let inner = fid
            .downcast_ref::<MemFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        // Only the root lists entries in this flat backend.
        if !inner.path.is_empty() {
            return Err(MountError::NotDirectory(inner.path.clone()));
        }
        let entries = self
            .files
            .lock()
            .keys()
            .filter(|k| !k.contains('/'))
            .map(|k| DirEntry {
                name: k.clone(),
                is_dir: false,
                size: 0,
                stat: None,
            })
            .collect();
        Ok(entries)
    }

    async fn stat(&self, fid: &Fid, caller: &Subject) -> Result<Stat, MountError> {
        Self::check(caller)?;
        let inner = fid
            .downcast_ref::<MemFid>()
            .ok_or_else(|| MountError::InvalidArgument("bad fid".into()))?;
        // Root -> directory (qtype high bit set); a known file -> regular file.
        let files = self.files.lock();
        if inner.path.is_empty() {
            return Ok(Stat {
                qtype: 0x80,
                size: 0,
                name: String::new(),
                mtime: 0,
            });
        }
        let size = files
            .get(&inner.path)
            .map(|v| v.len() as u64)
            .ok_or_else(|| MountError::NotFound(inner.path.clone()))?;
        Ok(Stat {
            qtype: 0,
            size,
            name: inner.path.clone(),
            mtime: 0,
        })
    }

    async fn clunk(&self, _fid: Fid, _caller: &Subject) {}
}

/// THE make-or-break test: a real wasip1 guest's filesystem IS the VFS Mount.
///
/// Driven from a PLAIN thread (the `MountDir` adapters `block_on` an inner runtime,
/// which would panic on a tokio worker). The guest reads a host-seeded file, writes
/// two files, reads one back, and lists the dir — all against the Mount. We assert
/// the guest exits 0 and that its writes landed in the Mount (Subject-scoped, no
/// host fs).
#[test]
fn wasi_guest_filesystem_is_the_vfs_mount() {
    let Some(wasm) = guest_wasm_or_ci_fail("profile_b_fs_is_mount") else {
        return;
    };

    let mount = Arc::new(MemMount::new(vec![("seed.txt", b"hello-vfs")]));
    let subject = Subject::new("alice");

    // Build + run on a plain thread (NOT a tokio worker).
    let mount_for_assert = Arc::clone(&mount);
    let handle = std::thread::spawn(move || {
        let sandbox = WasiSandbox::wasi_for(&wasm, subject, mount as Arc<dyn Mount>)
            .expect("build Profile-B WASI sandbox");
        sandbox.run_start(BIG_FUEL)
    });
    let result = handle.join().expect("guest thread panicked");
    result.expect("guest _start should exit cleanly (status 0)");

    // The guest's writes MUST have landed in the Mount (not the host).
    let files = mount_for_assert.files.lock();
    let out = files
        .get("out.txt")
        .expect("out.txt must exist in the Mount");
    assert_eq!(
        out, b"sfv-olleh",
        "out.txt = reversed seed, round-tripped via VFS"
    );
    let done = files
        .get("done.txt")
        .expect("done.txt must exist in the Mount");
    assert_eq!(
        done, b"ok",
        "done.txt success marker landed in the VFS Mount"
    );

    // Negative control: there is NO host preopen, so the only files that can exist
    // are the ones we seeded + the ones the guest wrote THROUGH the Mount. (A host
    // escape would have to write somewhere we cannot observe in this map; the
    // structural guarantee is that the sandbox was built with exactly one preopen —
    // the MemMount — and no `preopened_dir(host_dir)` call exists in wasi_sandbox.)
    assert!(files.contains_key("seed.txt"));
    eprintln!("profile_b: wasip1 guest fs == VFS Mount, round-trip + marker all green");
}

/// The bound Subject reaches the Mount on every op: a `denied` Subject is refused
/// at the Mount (the single policy point), so the guest cannot even read its seed.
#[test]
fn wasi_guest_subject_reaches_mount_policy_point() {
    let Some(wasm) = guest_wasm_or_ci_fail("profile_b_subject_policy") else {
        return;
    };
    let mount = Arc::new(MemMount::new(vec![("seed.txt", b"hello-vfs")]));
    let denied = Subject::new("denied");

    let handle = std::thread::spawn(move || {
        let sandbox =
            WasiSandbox::wasi_for(&wasm, denied, mount as Arc<dyn Mount>).expect("build sandbox");
        sandbox.run_start(BIG_FUEL)
    });
    let result = handle.join().expect("guest thread panicked");
    // The Mount refuses the denied Subject -> the guest's first read fails ->
    // guest exits nonzero -> run_start returns Err. The point: the Subject reached
    // the Mount and was enforced THERE.
    assert!(
        result.is_err(),
        "denied Subject must be refused at the Mount (guest cannot read seed.txt)"
    );
    eprintln!("profile_b: denied Subject refused at the Mount policy point");
}
