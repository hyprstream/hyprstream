//! Kata microVM 9P-over-vsock end-to-end harness (epic #729, V4 = #733).
//!
//! Companion to `kata_spawn_e2e.rs` (#721, the **virtio-fs** rootfs path). This
//! one exercises the **native-9P-in-guest** path: the host serves the sandbox's
//! Subject-scoped tenant VFS as 9P2000.L over a second vsock channel
//! (`VFS_9P_VSOCK_PORT = 564`, V2 #731 —
//! `KataBackend::serve_tenant_vfs_9p` → `hyprstream_9p::serve_mount_vsock_raw`),
//! and an in-guest userspace 9P client (V3 #732, `workers/hypr9p-guest`) dials
//! it and operates the tree:
//!
//! ```text
//!   KataBackend::start(&mut sandbox with image_id)
//!        │  compose+serve tenant VFS (virtio-fs rootfs share, #721)
//!        │  start_vm
//!        │  serve_tenant_vfs_9p  — bind host UDS <vsock-base>_564, RAW 9P (#741)
//!        ▼
//!   exec_sync(hypr9p-guest ls /)   — guest dials AF_VSOCK (CID 2, port 564),
//!        │                           attach → readdir, prints tenant-VFS root
//!        ▼
//!   assert stdout lists models/ deltas/ stream/  (the injected tenant mounts)
//!        │
//!   [optional] exec_sync(hypr9p-guest cat <path>) — assert injected file bytes
//!        ▼
//!   stop() + destroy()
//! ```
//!
//! ## CI-safe: self-skips without assets + KVM + a staged guest helper
//!
//! Returns early (printing why) unless ALL hold, so it is inert in CI and only
//! runs opt-in on an operator-provisioned host:
//!
//! * `HYPRSTREAM_KATA_KERNEL` — readable guest kernel,
//! * `HYPRSTREAM_KATA_IMAGE`  — readable guest rootfs image,
//! * `HYPRSTREAM_KATA_RAFS_IMAGE_ID` — a tenant RAFS image id already present in
//!   a store (**required**: the 9P-over-vsock channel is only stood up when the
//!   sandbox has a tenant VFS, i.e. `image_id` is set — see `KataBackend::start`),
//! * `HYPRSTREAM_GUEST_9P_HELPER` — path, **inside the guest**, to the staged
//!   `hypr9p-guest` binary (default `/usr/local/bin/hypr9p-guest`),
//! * `/dev/kvm` present, `cloud-hypervisor` on `PATH`.
//!
//! ## Status: COMPILES; boot is gated on explicit operator authorization
//!
//! `cargo test -p hyprstream-workers --features kata-vm --no-run` builds this
//! without running it. The actual VM bring-up is performed separately, under
//! operator control — do not remove the self-skip guard to "just verify it
//! boots".
//!
//! ## What the operator boot validates (not verified here)
//!
//! 1. A guest process (running inside the kata **container**) can reach the host
//!    over `AF_VSOCK` to `(CID = VMADDR_CID_HOST = 2, port = 564)`.
//! 2. **The RAW-vsock preamble assumption (#741):** the guest's `AF_VSOCK`
//!    connect arrives at the host UDS `<vsock-base>_564` as **raw 9P bytes with
//!    NO `connect <port>\n` preamble**, so `serve_mount_vsock_raw` (which strips
//!    nothing) reads the guest's first `Tversion` correctly. This is the single
//!    most load-bearing unverified assumption in the whole path.
//! 3. The kata guest kernel's lack of a 9P `trans=fd` transport is irrelevant
//!    because the mount is userspace — confirming the V3 design choice.

#![cfg(feature = "kata")]
#![allow(clippy::print_stdout, clippy::print_stderr)]
// Operator harness: assert with test-friendly macros.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hyprstream_workers::runtime::{KataBackend, PodSandbox, PodSandboxConfig, SandboxBackend};
use hyprstream_workers::{HypervisorType, ImageConfig, PoolConfig, RafsStore};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// In-guest path to the staged `hypr9p-guest` binary, overridable so the
/// operator can stage it wherever the rootfs puts it.
const DEFAULT_GUEST_HELPER: &str = "/usr/local/bin/hypr9p-guest";

/// Resolved preflight inputs for a real 9P-over-vsock spawn.
struct Preflight {
    kernel: PathBuf,
    image: PathBuf,
    rafs_image_id: String,
    guest_helper: String,
}

/// Environment gate. `Some(Preflight)` only when every prerequisite for a real
/// boot + in-guest mount is satisfied; otherwise prints the reason and returns
/// `None` so the caller skips cleanly (keeping the test inert in CI).
fn preflight() -> Option<Preflight> {
    let kernel = readable_env_path("HYPRSTREAM_KATA_KERNEL").map_err(skip).ok()?;
    let image = readable_env_path("HYPRSTREAM_KATA_IMAGE").map_err(skip).ok()?;

    // The 9P-over-vsock channel is only stood up when the sandbox has a tenant
    // VFS (image_id set); without a RAFS image id there is nothing to serve.
    let rafs_image_id = match std::env::var("HYPRSTREAM_KATA_RAFS_IMAGE_ID") {
        Ok(id) if !id.is_empty() => id,
        _ => {
            println!(
                "[kata_9p_vsock_e2e] SKIP: HYPRSTREAM_KATA_RAFS_IMAGE_ID not set \
                 (required — the 9P-over-vsock channel needs a tenant VFS)"
            );
            return None;
        }
    };

    let guest_helper = std::env::var("HYPRSTREAM_GUEST_9P_HELPER")
        .unwrap_or_else(|_| DEFAULT_GUEST_HELPER.to_owned());

    if !Path::new("/dev/kvm").exists() {
        println!("[kata_9p_vsock_e2e] SKIP: /dev/kvm not present (no hardware virtualization)");
        return None;
    }
    if which::which("cloud-hypervisor").is_err() {
        println!("[kata_9p_vsock_e2e] SKIP: cloud-hypervisor not found on PATH");
        return None;
    }

    Some(Preflight { kernel, image, rafs_image_id, guest_helper })
}

fn skip(reason: String) -> String {
    println!("[kata_9p_vsock_e2e] SKIP: {reason}");
    reason
}

/// Read an env var as a path and confirm it exists and is readable.
fn readable_env_path(var: &str) -> Result<PathBuf, String> {
    let raw = std::env::var(var).map_err(|_| format!("{var} not set"))?;
    if raw.is_empty() {
        return Err(format!("{var} is empty"));
    }
    let path = PathBuf::from(raw);
    std::fs::File::open(&path)
        .map_err(|e| format!("{var}={} is not readable: {e}", path.display()))?;
    Ok(path)
}

/// Build a `PoolConfig` for Cloud Hypervisor from the resolved asset paths.
fn pool_config_for(pre: &Preflight, runtime_dir: PathBuf) -> PoolConfig {
    PoolConfig {
        hypervisor: HypervisorType::CloudHypervisor,
        kernel_path: pre.kernel.clone(),
        vm_image: pre.image.clone(),
        runtime_dir,
        vm_cpus: 1,
        vm_memory_mb: 512,
        create_timeout_secs: 60,
        stop_timeout_secs: 30,
        ..PoolConfig::default()
    }
}

/// Build an `ImageConfig` + `RafsStore` rooted under `base`.
fn image_store(base: &Path) -> Result<(ImageConfig, Arc<RafsStore>), Box<dyn std::error::Error>> {
    let image_config = ImageConfig {
        blobs_dir: base.join("blobs"),
        bootstrap_dir: base.join("bootstrap"),
        refs_dir: base.join("refs"),
        cache_dir: base.join("cache"),
        runtime_dir: base.join("nydus-runtime"),
        ..ImageConfig::default()
    };
    for dir in [
        &image_config.blobs_dir,
        &image_config.bootstrap_dir,
        &image_config.refs_dir,
        &image_config.cache_dir,
        &image_config.runtime_dir,
    ] {
        std::fs::create_dir_all(dir)?;
    }
    let store = Arc::new(RafsStore::new(image_config.clone())?);
    Ok((image_config, store))
}

/// The full boot → serve-9P-over-vsock → in-guest mount → teardown drive.
///
/// This is the code that WILL run later under operator authorization; here it
/// only has to compile. It self-skips (returns `Ok(())` after printing) when
/// the preflight gate is not satisfied, so it is safe to run in CI.
#[tokio::test]
async fn kata_9p_vsock_e2e() -> TestResult {
    let Some(pre) = preflight() else {
        return Ok(());
    };

    println!(
        "[kata_9p_vsock_e2e] preflight OK: kernel={} image={} rafs_image_id={} guest_helper={}",
        pre.kernel.display(),
        pre.image.display(),
        pre.rafs_image_id,
        pre.guest_helper,
    );

    let scratch = tempfile::tempdir()?;
    let runtime_dir = scratch.path().join("sandboxes");
    std::fs::create_dir_all(&runtime_dir)?;

    let (image_config, rafs_store) = image_store(scratch.path())?;

    // ─────────────────────────────────────────────────────────────────────
    // Boot the microVM. With `image_id` set, `start()` composes + serves the
    // tenant VFS AND stands up the 9P-over-vsock channel (serve_tenant_vfs_9p)
    // on VFS_9P_VSOCK_PORT (564).
    // ─────────────────────────────────────────────────────────────────────
    let pool_config = pool_config_for(&pre, runtime_dir.clone());
    let backend = KataBackend::new(image_config, Arc::clone(&rafs_store));

    assert!(
        backend.is_available(),
        "cloud-hypervisor was on PATH at preflight but backend reports unavailable"
    );
    backend.initialize(&pool_config).await?;

    let sandbox_config = PodSandboxConfig::default();
    let sandbox_id = format!("kata-9p-e2e-{}", uuid::Uuid::new_v4());
    let sandbox_path = runtime_dir.join(&sandbox_id);
    let mut sandbox = PodSandbox::new(sandbox_id.clone(), &sandbox_config, sandbox_path);
    // Tenant VFS → this is what makes `start()` stand up the 9P vsock channel.
    sandbox.set_image_id(pre.rafs_image_id.clone());
    let annotations: HashMap<String, String> = HashMap::new();

    // NOTE: boots a real Cloud Hypervisor guest — only after the preflight gate,
    // i.e. on an operator-provisioned host, never in CI.
    let handle = backend
        .start(&mut sandbox, &sandbox_config, &pool_config, &annotations)
        .await?;
    sandbox.set_backend_handle(handle);
    println!("[kata_9p_vsock_e2e] VM started for sandbox {sandbox_id}");

    // ─────────────────────────────────────────────────────────────────────
    // In-guest: run the staged hypr9p-guest client to mount+operate the 9P VFS
    // over vsock. `ls /` must list the injected tenant-VFS mounts, proving the
    // guest reached the host 9P server over vsock and the tree is the tenant
    // VFS. Wrapped in `run_guest_9p` so teardown always happens.
    // ─────────────────────────────────────────────────────────────────────
    let outcome = run_guest_9p(&backend, &sandbox, &pre).await;

    // Always tear down, even on assertion failure, so no VM is left running.
    let _ = backend.stop(&sandbox).await;
    let _ = backend.destroy(&sandbox).await;
    println!("[kata_9p_vsock_e2e] sandbox {sandbox_id} stopped + destroyed");

    outcome
}

/// Drive the in-guest 9P operations and assert on their output.
async fn run_guest_9p(
    backend: &KataBackend,
    sandbox: &PodSandbox,
    pre: &Preflight,
) -> TestResult {
    // `hypr9p-guest ls /` — the guest dials AF_VSOCK (CID 2, port 564), attaches
    // the host 9P tree, and lists the root. The injected tenant-VFS mounts
    // (`models`/`deltas`/`stream`) are exposed at the 9P root, so seeing them is
    // unambiguous evidence the vsock 9P channel is serving the tenant VFS.
    let ls_argv: Vec<String> = vec![pre.guest_helper.clone(), "ls".to_owned(), "/".to_owned()];
    let (code, stdout, stderr) = backend.exec_sync(sandbox, &ls_argv, 30).await?;
    let ls_out = String::from_utf8_lossy(&stdout);
    println!(
        "[kata_9p_vsock_e2e] guest `hypr9p-guest ls /` exit={code} stdout={ls_out:?} stderr={:?}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(code, 0, "in-guest `hypr9p-guest ls /` should exit 0");
    for injected in ["models", "deltas", "stream"] {
        assert!(
            ls_out.lines().any(|l| l.trim_end_matches('/') == injected),
            "9P root over vsock must list injected tenant-VFS mount `/{injected}` \
             (proves the guest mounted the tenant VFS over vsock 9P); got: {ls_out:?}"
        );
    }

    // Optional: `cat` a known injected file and assert its bytes. Requires the
    // operator to have injected a readable file into the tenant VFS and to point
    // HYPRSTREAM_GUEST_9P_READ_PATH at it (with HYPRSTREAM_GUEST_9P_EXPECT set to
    // its expected contents). Skipped when unset — the default tenant VFS injects
    // empty `/models` `/deltas` `/stream` dirs, so no file is guaranteed present.
    if let (Ok(read_path), Ok(expect)) = (
        std::env::var("HYPRSTREAM_GUEST_9P_READ_PATH"),
        std::env::var("HYPRSTREAM_GUEST_9P_EXPECT"),
    ) {
        let cat_argv: Vec<String> =
            vec![pre.guest_helper.clone(), "cat".to_owned(), read_path.clone()];
        let (code, stdout, stderr) = backend.exec_sync(sandbox, &cat_argv, 30).await?;
        let got = String::from_utf8_lossy(&stdout);
        println!(
            "[kata_9p_vsock_e2e] guest `hypr9p-guest cat {read_path}` exit={code} stdout={got:?} stderr={:?}",
            String::from_utf8_lossy(&stderr)
        );
        assert_eq!(code, 0, "in-guest `hypr9p-guest cat {read_path}` should exit 0");
        assert_eq!(
            got.trim_end(),
            expect.trim_end(),
            "9P read of {read_path} over vsock returned unexpected contents"
        );
    } else {
        println!(
            "[kata_9p_vsock_e2e] cat round-trip skipped (set HYPRSTREAM_GUEST_9P_READ_PATH + \
             HYPRSTREAM_GUEST_9P_EXPECT to assert an injected file's bytes)"
        );
    }

    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// V5 (#751): FUSE→9P POSIX mount
//
// Where the test above proves the *client* (`hypr9p-guest ls /`) can operate the
// tenant VFS over vsock 9P, this one proves the **POSIX-mount** deliverable of
// #751: the guest runs `hypr9p-guest --fuse-mount <mnt>`, which bridges FUSE →
// (the same reused) 9P client, and then an **arbitrary** in-guest process — a
// plain `cat <mnt>/models/hello`, NOT the 9P client — reads through the kernel
// FUSE path. Seeing the injected file's bytes via bare `cat` is the unambiguous
// evidence that the tenant VFS is a real filesystem in the guest.
//
// This is even more strongly operator-gated than the client path: FUSE mounting
// needs `/dev/fuse` + CAP_SYS_ADMIN inside the guest, which CI cannot provide.
// So on top of the shared `preflight()` gate it additionally requires the
// operator to opt in with `HYPRSTREAM_GUEST_FUSE_E2E=1` and to have injected the
// file being `cat`-ed. It compiles under `--no-run` and self-skips otherwise —
// **do not remove the guard to "just mount it"; the mount is the operator's.**

/// In-guest mountpoint for the FUSE→9P bridge; overridable for rootfs layouts
/// that put a writable mount dir elsewhere.
const DEFAULT_FUSE_MOUNT: &str = "/mnt/vfs";

/// Boot → `hypr9p-guest --fuse-mount` → bare `cat` through the FUSE mount.
///
/// Self-skips (returns `Ok(())` after printing) unless the shared preflight gate
/// AND the FUSE opt-in are satisfied, so it is inert in CI and compiles under
/// `cargo test -p hyprstream-workers --features kata-vm --no-run`.
#[tokio::test]
async fn kata_9p_fuse_mount_e2e() -> TestResult {
    let Some(pre) = preflight() else {
        return Ok(());
    };

    // FUSE mounting needs /dev/fuse + mount privilege in the guest — a CI runner
    // can't grant it, so require an explicit operator opt-in beyond preflight.
    if std::env::var("HYPRSTREAM_GUEST_FUSE_E2E").map(|v| v != "1").unwrap_or(true) {
        println!(
            "[kata_9p_fuse_mount_e2e] SKIP: set HYPRSTREAM_GUEST_FUSE_E2E=1 to run the FUSE \
             POSIX-mount path (needs /dev/fuse + CAP_SYS_ADMIN in the guest)"
        );
        return Ok(());
    }

    let mount_dir =
        std::env::var("HYPRSTREAM_GUEST_FUSE_MOUNT").unwrap_or_else(|_| DEFAULT_FUSE_MOUNT.to_owned());
    // Path to `cat` THROUGH the mount, and its expected contents. The operator
    // injects this file into the tenant VFS; the default tenant VFS injects only
    // empty dirs, so a concrete file must be provided to assert bytes.
    let (Ok(read_rel), Ok(expect)) = (
        std::env::var("HYPRSTREAM_GUEST_FUSE_READ_PATH"),
        std::env::var("HYPRSTREAM_GUEST_FUSE_EXPECT"),
    ) else {
        println!(
            "[kata_9p_fuse_mount_e2e] SKIP: set HYPRSTREAM_GUEST_FUSE_READ_PATH \
             (e.g. models/hello, relative to the mount) + HYPRSTREAM_GUEST_FUSE_EXPECT"
        );
        return Ok(());
    };

    println!(
        "[kata_9p_fuse_mount_e2e] preflight OK: kernel={} image={} rafs_image_id={} \
         guest_helper={} mount_dir={mount_dir}",
        pre.kernel.display(),
        pre.image.display(),
        pre.rafs_image_id,
        pre.guest_helper,
    );

    let scratch = tempfile::tempdir()?;
    let runtime_dir = scratch.path().join("sandboxes");
    std::fs::create_dir_all(&runtime_dir)?;
    let (image_config, rafs_store) = image_store(scratch.path())?;

    let pool_config = pool_config_for(&pre, runtime_dir.clone());
    let backend = KataBackend::new(image_config, Arc::clone(&rafs_store));
    assert!(
        backend.is_available(),
        "cloud-hypervisor was on PATH at preflight but backend reports unavailable"
    );
    backend.initialize(&pool_config).await?;

    let sandbox_config = PodSandboxConfig::default();
    let sandbox_id = format!("kata-9p-fuse-e2e-{}", uuid::Uuid::new_v4());
    let sandbox_path = runtime_dir.join(&sandbox_id);
    let mut sandbox = PodSandbox::new(sandbox_id.clone(), &sandbox_config, sandbox_path);
    sandbox.set_image_id(pre.rafs_image_id.clone());
    let annotations: HashMap<String, String> = HashMap::new();

    let handle = backend
        .start(&mut sandbox, &sandbox_config, &pool_config, &annotations)
        .await?;
    sandbox.set_backend_handle(handle);
    println!("[kata_9p_fuse_mount_e2e] VM started for sandbox {sandbox_id}");

    let outcome = run_guest_fuse(&backend, &sandbox, &pre, &mount_dir, &read_rel, &expect).await;

    let _ = backend.stop(&sandbox).await;
    let _ = backend.destroy(&sandbox).await;
    println!("[kata_9p_fuse_mount_e2e] sandbox {sandbox_id} stopped + destroyed");

    outcome
}

/// Mount the tenant VFS via FUSE→9P in the guest, then read a file through it
/// with a bare `cat` (NOT the 9P client) and assert its bytes.
async fn run_guest_fuse(
    backend: &KataBackend,
    sandbox: &PodSandbox,
    pre: &Preflight,
    mount_dir: &str,
    read_rel: &str,
    expect: &str,
) -> TestResult {
    // 1. Ensure the mountpoint exists, then start the FUSE bridge in the
    //    background (it blocks for the mount's lifetime), giving it a moment to
    //    complete the mount before we read through it.
    //
    //    `sh -c` composes the mkdir + backgrounded mount + settle in one exec so
    //    we don't depend on exec_sync spawning a persistent process.
    let mount_cmd = format!(
        "mkdir -p {mount_dir} && {helper} --fuse-mount {mount_dir} & sleep 2",
        helper = pre.guest_helper,
    );
    let mount_argv: Vec<String> =
        vec!["/bin/sh".to_owned(), "-c".to_owned(), mount_cmd.clone()];
    let (code, _out, stderr) = backend.exec_sync(sandbox, &mount_argv, 30).await?;
    println!(
        "[kata_9p_fuse_mount_e2e] mount cmd={mount_cmd:?} exit={code} stderr={:?}",
        String::from_utf8_lossy(&stderr)
    );

    // 2. The load-bearing assertion: a plain `cat` through the FUSE mount — an
    //    arbitrary process, not the hypr9p 9P client — returns the file bytes.
    let file_path = format!("{}/{}", mount_dir.trim_end_matches('/'), read_rel.trim_start_matches('/'));
    let cat_argv: Vec<String> = vec!["/bin/cat".to_owned(), file_path.clone()];
    let (code, stdout, stderr) = backend.exec_sync(sandbox, &cat_argv, 30).await?;
    let got = String::from_utf8_lossy(&stdout);
    println!(
        "[kata_9p_fuse_mount_e2e] guest `cat {file_path}` (through FUSE) exit={code} \
         stdout={got:?} stderr={:?}",
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(code, 0, "bare `cat {file_path}` through the FUSE mount should exit 0");
    assert_eq!(
        got.trim_end(),
        expect.trim_end(),
        "FUSE-mounted read of {file_path} returned unexpected contents \
         (a bare cat, proving the tenant VFS is a real POSIX filesystem in the guest)"
    );

    // 3. Best-effort unmount so no stuck mount is left behind.
    let umount_argv: Vec<String> =
        vec!["/bin/umount".to_owned(), mount_dir.to_owned()];
    let _ = backend.exec_sync(sandbox, &umount_argv, 15).await;

    Ok(())
}
