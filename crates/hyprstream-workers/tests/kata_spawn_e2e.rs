//! Kata microVM boot harness — env-gated end-to-end spawn (#721).
//!
//! This is the integration harness that drives the **real** Kata/Cloud
//! Hypervisor boot path exposed by [`KataBackend`]/[`SandboxBackend`]:
//!
//! ```text
//!   PoolConfig (env kernel + rootfs) + HypervisorType::CloudHypervisor
//!        │
//!        ▼
//!   KataBackend::new(image_config, rafs_store)
//!        │  initialize()          — enable rootless when non-root
//!        │  start(&mut sandbox)   — compose+serve tenant VFS (virtio-fs),
//!        │                          prepare_vm → attach ShareFs → start_vm
//!        ▼
//!   exec_sync(cat <known file>)  — lazily dials the kata-agent over
//!        │                          hybrid-vsock, CreateContainer →
//!        │                          StartContainer → ExecProcess → WaitProcess
//!        ▼
//!   stop() + destroy()           — stop_vm, cleanup, remove runtime dir
//! ```
//!
//! ## CI-safe: self-skips without assets + KVM
//!
//! The test returns early (printing why) unless ALL of the following hold, so
//! it is inert in CI and on any host lacking the VM toolchain, and only runs
//! opt-in on a machine an operator has explicitly provisioned:
//!
//! * `HYPRSTREAM_KATA_KERNEL` set to a readable guest kernel,
//! * `HYPRSTREAM_KATA_IMAGE`  set to a readable guest rootfs image,
//! * `/dev/kvm` present (hardware virtualization available),
//! * `cloud-hypervisor` on `PATH`.
//!
//! The asset paths are taken from the environment — never hardcoded — so the
//! same harness works against any matching kata-containers kernel/rootfs pair.
//!
//! ## Status: COMPILES; boot is gated on explicit operator authorization
//!
//! `cargo test -p hyprstream-workers --features kata-vm --no-run` builds this
//! without running it. The actual VM bring-up (which classifiers treat as a
//! sensitive action) is performed separately, under operator control, not by
//! the harness author. Do not remove the self-skip guard to "just verify it
//! boots".
//!
//! ## Tenant VFS as container rootfs (#721)
//!
//! `PodSandbox::set_image_id` (the doc-hidden seam) lets this external harness
//! instruct `start()` to compose + serve + attach the per-sandbox **tenant
//! VFS** as the guest's virtio-fs rootfs share. `start()` records the mount
//! tag; `exec_sync` then tells the kata-agent (`CreateContainer`) to mount
//! that share, by tag, as the container's rootfs at
//! `/run/kata-containers/<cid>/rootfs`. So — with `HYPRSTREAM_KATA_RAFS_IMAGE_ID`
//! set to a pre-built RAFS image in the store — the container's filesystem is
//! the hyprstream tenant VFS (RAFS rootfs + injected `/models` `/deltas`
//! `/stream`), which the assertion below verifies.
//!
//! ## Remaining gap (see `TODO(#721)`)
//!
//! * `image::rafs_builder::build_rafs_bootstrap` is `pub(crate)`, so this
//!   external harness cannot synthesize a RAFS tenant image itself; the
//!   tenant-VFS-rootfs path is exercised only when a pre-built RAFS image id
//!   is supplied via `HYPRSTREAM_KATA_RAFS_IMAGE_ID`.

#![cfg(feature = "kata-vm")]
// This is an opt-in operator harness: it intentionally prints skip reasons to
// stdout and asserts with the test-friendly macros. Allow the workspace
// restriction lints that would otherwise fire on that harness style.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use hyprstream_workers::runtime::{KataBackend, PodSandbox, PodSandboxConfig, SandboxBackend};
use hyprstream_workers::{HypervisorType, ImageConfig, PoolConfig, RafsStore};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// Resolved preflight inputs for a real spawn.
struct Preflight {
    kernel: PathBuf,
    image: PathBuf,
}

/// Environment gate. Returns `Some(Preflight)` only when every prerequisite for
/// a real boot is satisfied; otherwise prints the reason and returns `None`
/// so the caller can skip cleanly (keeping the test inert in CI).
fn preflight() -> Option<Preflight> {
    let kernel = match readable_env_path("HYPRSTREAM_KATA_KERNEL") {
        Ok(p) => p,
        Err(reason) => {
            println!("[kata_spawn_e2e] SKIP: {reason}");
            return None;
        }
    };
    let image = match readable_env_path("HYPRSTREAM_KATA_IMAGE") {
        Ok(p) => p,
        Err(reason) => {
            println!("[kata_spawn_e2e] SKIP: {reason}");
            return None;
        }
    };

    if !Path::new("/dev/kvm").exists() {
        println!("[kata_spawn_e2e] SKIP: /dev/kvm not present (no hardware virtualization)");
        return None;
    }

    if which::which("cloud-hypervisor").is_err() {
        println!("[kata_spawn_e2e] SKIP: cloud-hypervisor not found on PATH");
        return None;
    }

    Some(Preflight { kernel, image })
}

/// Read an env var as a path and confirm it exists and is readable.
fn readable_env_path(var: &str) -> Result<PathBuf, String> {
    let raw = std::env::var(var).map_err(|_| format!("{var} not set"))?;
    if raw.is_empty() {
        return Err(format!("{var} is empty"));
    }
    let path = PathBuf::from(raw);
    // A readable regular file: probe by opening it (covers permission errors
    // that a bare `exists()` would miss).
    std::fs::File::open(&path)
        .map_err(|e| format!("{var}={} is not readable: {e}", path.display()))?;
    Ok(path)
}

/// Build a `PoolConfig` for Cloud Hypervisor from the resolved asset paths and
/// a per-test runtime directory. Mirrors the fields `KataBackend` actually
/// reads (`build_hypervisor_config`): kernel, image, cpus, memory, hypervisor.
fn pool_config_for(pre: &Preflight, runtime_dir: PathBuf) -> PoolConfig {
    PoolConfig {
        hypervisor: HypervisorType::CloudHypervisor,
        kernel_path: pre.kernel.clone(),
        vm_image: pre.image.clone(),
        runtime_dir,
        // Keep the microVM small; the harness only needs a booted guest agent.
        vm_cpus: 1,
        vm_memory_mb: 512,
        // Bounded so a wedged boot fails the test rather than hanging CI.
        create_timeout_secs: 60,
        stop_timeout_secs: 30,
        ..PoolConfig::default()
    }
}

/// Build an `ImageConfig` + `RafsStore` rooted under `base`. The store backs
/// the (optional) tenant-VFS composition below; the VM's own rootfs comes from
/// `PoolConfig::vm_image`, not this store.
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

/// The full boot → handshake → exec → teardown drive.
///
/// This is the code that WILL run later under operator authorization; here it
/// only has to compile. It self-skips (returns `Ok(())` after printing) when
/// the preflight gate is not satisfied, so it is safe to run in CI.
#[tokio::test]
async fn kata_spawn_e2e() -> TestResult {
    let Some(pre) = preflight() else {
        return Ok(());
    };

    println!(
        "[kata_spawn_e2e] preflight OK: kernel={} image={}",
        pre.kernel.display(),
        pre.image.display()
    );

    // Per-test scratch: sandbox sockets, VFS socket, writable upper, etc.
    let scratch = tempfile::tempdir()?;
    let runtime_dir = scratch.path().join("sandboxes");
    std::fs::create_dir_all(&runtime_dir)?;

    let (image_config, rafs_store) = image_store(scratch.path())?;

    // Optional tenant VFS: when a pre-built RAFS image id is supplied, we tell
    // `start()` (via `set_image_id` below) to compose + serve + attach that
    // image's tenant VFS as the guest's virtio-fs rootfs share (#721). When
    // absent, the VM boots its own `PoolConfig::vm_image` rootfs and the
    // container has no rootfs (the historical failure #721 targets), so the
    // exec assertion is relaxed to the bare-guest smoke below.
    //
    // TODO(#721): `image::rafs_builder::build_rafs_bootstrap` is `pub(crate)`,
    // so this external harness cannot synthesize a tenant RAFS image itself.
    // Supply `HYPRSTREAM_KATA_RAFS_IMAGE_ID` pointing at an image already
    // present in the store to drive the tenant-VFS-as-rootfs path.
    // Resolve the tenant RAFS image id. Preference order:
    //   1. HYPRSTREAM_KATA_RAFS_IMAGE_ID — an id already present in a store.
    //   2. HYPRSTREAM_KATA_PULL_IMAGE — an OCI ref to pull+convert into THIS
    //      run's store (self-contained: the fresh temp store is otherwise empty).
    let rafs_image_id = match std::env::var("HYPRSTREAM_KATA_RAFS_IMAGE_ID").ok() {
        Some(id) => {
            println!("[kata_spawn_e2e] tenant VFS rootfs from RAFS image {id}");
            Some(id)
        }
        None => match std::env::var("HYPRSTREAM_KATA_PULL_IMAGE").ok() {
            Some(image_ref) => {
                println!("[kata_spawn_e2e] pulling+converting {image_ref} into the RAFS store …");
                // TODO(#737): `RafsStore::ensure`/`pull` drops a nested tokio
                // runtime inside its own execution → panics from ANY tokio
                // context (verified: a dedicated-thread `block_on` wrapper does
                // not help). Until #737 is fixed, prefer HYPRSTREAM_KATA_RAFS_IMAGE_ID
                // pointing at a store pre-populated by a non-async pull.
                let id = rafs_store.ensure(&image_ref).await?;
                println!("[kata_spawn_e2e] tenant VFS rootfs from pulled RAFS image {id}");
                Some(id)
            }
            None => {
                println!(
                    "[kata_spawn_e2e] tenant-VFS rootfs skipped (set \
                     HYPRSTREAM_KATA_RAFS_IMAGE_ID or HYPRSTREAM_KATA_PULL_IMAGE)"
                );
                None
            }
        },
    };

    // ─────────────────────────────────────────────────────────────────────
    // Boot the microVM via the real backend path.
    // ─────────────────────────────────────────────────────────────────────
    let pool_config = pool_config_for(&pre, runtime_dir.clone());
    let backend = KataBackend::new(image_config, Arc::clone(&rafs_store));

    assert!(
        backend.is_available(),
        "cloud-hypervisor was on PATH at preflight but backend reports unavailable"
    );

    backend.initialize(&pool_config).await?;

    let sandbox_config = PodSandboxConfig::default();
    let sandbox_id = format!("kata-e2e-{}", uuid::Uuid::new_v4());
    let sandbox_path = runtime_dir.join(&sandbox_id);
    let mut sandbox = PodSandbox::new(sandbox_id.clone(), &sandbox_config, sandbox_path);

    // #721: hand `start()` the tenant image id so it composes + serves + attaches
    // the tenant VFS as this guest's virtio-fs rootfs share (and records the
    // mount tag `exec_sync` uses to mount it as the container's rootfs). Without
    // this, `start()` attaches no share and the container has no rootfs.
    if let Some(ref id) = rafs_image_id {
        sandbox.set_image_id(id.clone());
    }
    let annotations: HashMap<String, String> = HashMap::new();

    // NOTE: this call boots a real Cloud Hypervisor guest. It only runs after
    // the preflight gate passed (assets + KVM + CH), i.e. on an operator-
    // provisioned host — never in CI.
    let handle = backend
        .start(&mut sandbox, &sandbox_config, &pool_config, &annotations)
        .await?;
    sandbox.set_backend_handle(handle);
    println!("[kata_spawn_e2e] VM started for sandbox {sandbox_id}");

    // ─────────────────────────────────────────────────────────────────────
    // Wait for the kata-agent handshake + run a command in the guest.
    //
    // `exec_sync` lazily dials the guest agent over hybrid-vsock
    // (`guest_agent_client`), then drives CreateContainer → StartContainer →
    // ExecProcess → WaitProcess. When a tenant VFS was attached, `exec_sync`
    // tells CreateContainer to mount that virtio-fs share as the container's
    // rootfs, so the exec runs with the tenant VFS as its filesystem.
    //
    // Proof that the tenant VFS *is* the container rootfs: the injected
    // hyprstream mounts (`/models`, `/deltas`, `/stream`) live at the tenant
    // VFS root and therefore at the container's `/`. They do NOT exist in a
    // bare kata guest rootfs, so listing `/` inside the container and seeing
    // them is unambiguous evidence the virtio-fs share is the container's
    // filesystem. (The bare-guest fallback, with no tenant VFS, can only smoke
    // that the agent + a container ran at all.)
    //
    // This is what the OPERATOR BOOT will validate: on a real kata 3.31.0
    // microVM the `ls /` below must exit 0 and its stdout must contain
    // `models`, `deltas`, and `stream` — i.e. the tenant VFS mounted over
    // virtio-fs is the container's root.
    //
    // TODO(#721): once `build_rafs_bootstrap` is exposed to this harness (or
    // an injected-content seam exists), also `cat` a known file the tenant VFS
    // injects (e.g. `/models/<id>/...`) and assert its bytes.
    // ─────────────────────────────────────────────────────────────────────
    let tenant_vfs = rafs_image_id.is_some();
    let exec_argv: Vec<String> = if tenant_vfs {
        vec!["ls".to_owned(), "-1".to_owned(), "/".to_owned()]
    } else {
        vec!["cat".to_owned(), "/etc/hostname".to_owned()]
    };
    let exec_result = backend.exec_sync(&sandbox, &exec_argv, 30).await;

    // Always attempt teardown, even if exec failed, so a booted VM is never
    // left running past the test.
    let (exit_code, stdout, stderr) = match exec_result {
        Ok(triple) => triple,
        Err(e) => {
            let _ = backend.stop(&sandbox).await;
            let _ = backend.destroy(&sandbox).await;
            return Err(format!("exec_sync in guest failed: {e}").into());
        }
    };

    let stdout_str = String::from_utf8_lossy(&stdout);
    println!(
        "[kata_spawn_e2e] guest exec ({:?}) exit={exit_code} stdout={stdout_str:?} stderr={:?}",
        exec_argv,
        String::from_utf8_lossy(&stderr)
    );
    assert_eq!(exit_code, 0, "guest exec should exit 0");
    if tenant_vfs {
        for injected in ["models", "deltas", "stream"] {
            assert!(
                stdout_str.lines().any(|l| l.trim() == injected),
                "container `/` must list injected tenant-VFS mount `/{injected}` \
                 (proves the virtio-fs tenant VFS is the container rootfs); got: {stdout_str:?}"
            );
        }
    } else {
        assert!(
            !stdout.is_empty(),
            "guest /etc/hostname should be non-empty (proves the container-in-VM ran)"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Teardown: stop the VM and remove all runtime state.
    // ─────────────────────────────────────────────────────────────────────
    backend.stop(&sandbox).await?;
    backend.destroy(&sandbox).await?;
    println!("[kata_spawn_e2e] sandbox {sandbox_id} stopped + destroyed");

    Ok(())
}
