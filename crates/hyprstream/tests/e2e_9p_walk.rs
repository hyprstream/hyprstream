//! E2E (#421-E3): 9P filesystem walk — registry mount + walk + create + promote.
//!
//! This is the CPU-only end-to-end validation that the registry/worktree
//! service serves real files over 9P, that `RemoteRegistryMount` walks the
//! worktree via the generated `WorktreeClient` RPC, and that the writable
//! upper layer + promote saga (`git/promote.rs`) produce a new commit OID.
//!
//! ## What this exercises
//!
//! 1. **Registry serves real files via 9P** — the existing `WorktreeRequest`
//!    RPC (`walk`/`open`/`read`/`create`/`write`/`stat`/`clunk`) backed by
//!    `ContainedFs` over a real on-disk git worktree.
//! 2. **`RemoteRegistryMount` walks the worktree** — 2-level scope resolution
//!    (`/{repo}/{worktree}/...`) via `RegistryClient::get_by_name` →
//!    `repo(id).worktree(name)`, then forwards `wnames` to the worktree 9P walk.
//! 3. **`walk /{repo}/{worktree}/` → real directory listing** — `readdir`
//!    returns the on-disk entries (e.g. `config.json`).
//! 4. **`cat config.json` → real file bytes** — `open` + `read` returns the
//!    actual committed file content.
//! 5. **Qid version/path threaded (#388)** — `stat` returns a `Stat` whose
//!    `version`/`path` come from the registry service's wire qid
//!    (`qid_from_metadata`: `version=ctime`, `path=ino`), not a flattened qid.
//! 6. **`create` + `write` → writable upper (#394)** — `create` on a walked
//!    directory fid copy-ups into the worktree's writable layer and returns an
//!    opened fid on the new file; `write` lands bytes in it.
//! 7. **`promote` → stageAll + commit → new OID** — the promote saga from
//!    `git/promote.rs` snapshots the worktree into a deterministic commit.
//!
//! ## Harness
//!
//! The test stands up a real `PolicyService` (permissive) and `RegistryService`
//! in-process over `inproc://` ZMQ (same pattern as the
//! `test_registry_service_health_check` unit test in `registry.rs`), backed by
//! a temp git repo with a `config.json`. It then constructs
//! `RemoteRegistryMount` over a `RegistryClient` and drives it through the
//! VFS `Mount` trait — the same path the `/srv/registry/{repo}/{worktree}/`
//! namespace entry uses in production.

#![allow(clippy::expect_used, clippy::unwrap_used, clippy::print_stdout)]

use std::sync::Arc;

use anyhow::{anyhow, Result};
use tempfile::TempDir;
use tokio::sync::RwLock;

// ─── harness: in-process services ───────────────────────────────────────────

use git2db::{Git2DB, RepoId};
use hyprstream_core::auth::PolicyManager;
use hyprstream_core::config::TokenConfig;
use hyprstream_core::services::{
    PolicyClient, PolicyService, RegistryClient, RegistryService,
    generated::registry_client::CreateWorktreeRequest,
};
use hyprstream_core::services::remote_registry_mount::RemoteRegistryMount;
use hyprstream_core::git::promote::{PromoteAuthor, PromoteLeaseTable, promote};

use hyprstream_rpc::crypto::{CryptoPolicy, generate_signing_keypair};
use hyprstream_rpc::envelope::{EnvelopeVerifyConfig, install_verify_config};
use hyprstream_rpc::transport::TransportConfig;
use hyprstream_rpc::Subject;
use hyprstream_vfs::{DMDIR, OREAD, OWRITE, Mount};
use hyprstream_service::{InprocManager, ServiceManager};

/// Minimal dependency-free `block_on` for driving the mount's async methods
/// from a thread that has **no tokio runtime entered**.
///
/// `RemoteRegistryMount`'s `Mount` impl does all its work via `self.rt.block_on`
/// on a private internal runtime. Awaiting those methods from a tokio executor
/// trips the "Cannot start a runtime from within a runtime" guard. So we run the
/// mount section inside `tokio::task::spawn_blocking` (no entered runtime there)
/// and drive the futures with this std-only park/unpark executor — the nested
/// `self.rt.block_on` then runs cleanly.
fn std_block_on<F: std::future::Future>(fut: F) -> F::Output {
    use std::sync::Arc;
    use std::task::{Context, Poll, Wake, Waker};

    struct ThreadWaker(std::thread::Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.unpark();
        }
    }

    let waker: Waker = Arc::new(ThreadWaker(std::thread::current())).into();
    let mut cx = Context::from_waker(&waker);
    // Safety: `fut` is owned and never moved after pinning.
    let mut fut = std::pin::pin!(fut);
    loop {
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(out) => return out,
            Poll::Pending => std::thread::park(),
        }
    }
}

/// Unique inproc endpoint names per test invocation (process-global ZMQ ctx).
fn unique_inproc(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("inproc://e3-9p-{prefix}-{n}")
}

/// Install the Classical (EdDSA-only) envelope verify policy so the test's
/// ed25519 keys are accepted. The process-global default is fail-closed
/// Hybrid, which requires PQ keys the test doesn't have.
fn install_classical_verify() {
    let _ = install_verify_config(EnvelopeVerifyConfig {
        policy: CryptoPolicy::Classical,
        pq_store: None,
    });
    let _ = hyprstream_rpc::envelope::install_response_verify_config(
        hyprstream_rpc::envelope::ResponseVerifyConfig {
            policy: CryptoPolicy::Classical,
            pq_store: None,
        },
    );
}

/// Create a temp git repo at `dir/config-repo` with a `config.json`, commit
/// it on the `main` branch, and return the worktree-bearing path plus the
/// committed config bytes.
fn make_temp_git_repo(parent: &std::path::Path) -> Result<(std::path::PathBuf, Vec<u8>)> {
    use std::fs;
    use std::process::Command;

    let repo_dir = parent.join("config-repo");
    fs::create_dir_all(&repo_dir)?;

    // `git init -b main` (modern git) — fall back to `-b` unsupported for old git.
    let init_status = Command::new("git")
        .arg("init")
        .arg("-b")
        .arg("main")
        .current_dir(&repo_dir)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .status()?;
    if !init_status.success() {
        // Older git: init then symbolic-ref HEAD → main.
        let _ = Command::new("git")
            .arg("init")
            .current_dir(&repo_dir)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_SYSTEM", "/dev/null")
            .status()?;
        let _ = Command::new("git")
            .args(["symbolic-ref", "HEAD", "refs/heads/main"])
            .current_dir(&repo_dir)
            .status()?;
    }

    // Deterministic config.json content.
    let config_json = b"{\n  \"name\": \"e3-9p-walk-fixture\",\n  \"version\": 1\n}\n";
    fs::write(repo_dir.join("config.json"), config_json)?;

    // Commit with a fixed identity so the promote OID is reproducible w.r.t.
    // the initial commit (the promote commit itself is the thing we compare).
    for (key, val) in [
        ("user.name", "e3-9p-test"),
        ("user.email", "e3-9p-test@example.invalid"),
    ] {
        let _ = Command::new("git")
            .args(["config", key, val])
            .current_dir(&repo_dir)
            .status()?;
    }
    let add_status = Command::new("git")
        .args(["add", "config.json"])
        .current_dir(&repo_dir)
        .status()?;
    if !add_status.success() {
        return Err(anyhow!("git add config.json failed"));
    }
    let commit_status = Command::new("git")
        .args(["commit", "-m", "initial: config.json fixture for e3-9p-walk"])
        .current_dir(&repo_dir)
        .status()?;
    if !commit_status.success() {
        return Err(anyhow!("git commit failed (is the repo empty / git user unset?)"));
    }

    Ok((repo_dir, config_json.to_vec()))
}

/// Stand up PolicyService + RegistryService over inproc ZMQ, and return a
/// `RegistryClient` plus the spawned-service guards (which must outlive the
/// client — dropping them stops the services).
struct Services {
    registry_client: RegistryClient,
    /// The repo id assigned at registration (used to scope repo()/worktree()).
    repo_id: String,
    // Order matters: registry handle dropped before policy handle.
    _registry_handle: hyprstream_service::SpawnedService,
    _policy_handle: hyprstream_service::SpawnedService,
    _policy_temp: TempDir,
    // The registry's base_dir — must outlive the registry service.
    _registry_base: TempDir,
    _signing_key: hyprstream_rpc::crypto::SigningKey,
}

async fn spawn_services(repo_dir: &std::path::Path, repo_name: &str) -> Result<Services> {
    let signing_key = generate_signing_keypair().0;

    // ── PolicyService (permissive) ──────────────────────────────────────────
    let policy_temp = TempDir::new()?;
    let policy_manager = Arc::new(
        PolicyManager::permissive()
            .await
            .map_err(|e| anyhow!("policy manager: {e}"))?,
    );
    let policy_endpoint = unique_inproc("policy");
    let policy_transport = TransportConfig::from_endpoint(&policy_endpoint);
    let policy_git2db = Arc::new(RwLock::new(
        Git2DB::open(policy_temp.path()).await?,
    ));
    let policy_service = PolicyService::new(
        policy_manager,
        Arc::new(signing_key.clone()),
        TokenConfig::default(),
        policy_git2db,
        policy_transport,
    );
    let manager = InprocManager::new();
    let policy_handle = manager
        .spawn(Box::new(policy_service))
        .await
        .map_err(|e| anyhow!("spawn policy: {e}"))?;

    // PolicyClient used by RegistryService for authz checks.
    let policy_client = PolicyClient::for_local_endpoint_bootstrap(
        &policy_endpoint,
        signing_key.clone(),
        signing_key.verifying_key(),
        None,
    )?;

    // ── RegistryService ─────────────────────────────────────────────────────
    //
    // RegistryService::new opens Git2DB at `registry_base` and reloads the
    // persisted `registry.json`. So we pre-register the fixture *into*
    // `registry_base` via the non-deprecated `Git2DB::register(...)` builder
    // (which honors the real worktree path and commits the metadata), then let
    // the spawned service pick it up on open.
    //
    // Why not the obvious RPC paths?
    //   * `register` RPC → deprecated `register_repository`, which ignores the
    //     supplied path and hard-requires the repo at `base_dir/<random-uuid>`
    //     (impossible: the uuid is freshly minted at call time).
    //   * `clone` RPC → routes through the global `GitManager` clone options,
    //     which in this environment default to `prefer_shallow`; a shallow fetch
    //     over the `file://` local transport is rejected by libgit2.
    // The builder is the documented, path-correct registration the service
    // itself uses elsewhere, so this is faithful — only the *bootstrap* step is
    // in-process; everything under test (walk/read/stat/create/promote) still
    // goes over the live RPC service + RemoteRegistryMount.
    let registry_base = TempDir::new()?;

    // Place a *bare* clone of the fixture under registry_base. It must be bare:
    // the storage driver creates the served worktree as a *linked* worktree off
    // this repo (`worktrees/<branch>`), and a non-bare repo's primary checkout
    // already occupies `main` — libgit2 then refuses "main is already used by
    // worktree". A bare repo has no primary checkout, so `main` is free. This is
    // the same layout CloneBuilder produces in production.
    let registry_repo_path = registry_base.path().join(repo_name);
    bare_clone(repo_dir, &registry_repo_path)?;

    let repo_id = RepoId::new();
    {
        let mut git2db = Git2DB::open(registry_base.path()).await?;
        git2db
            .register(repo_id.clone())
            .name(repo_name)
            .worktree_path(&registry_repo_path)
            .url(String::new())
            .exec()
            .await
            .map_err(|e| anyhow!("builder register: {e}"))?;
        // Drop the Git2DB so the spawned RegistryService re-opens the dir and
        // reloads the just-committed registry.json.
    }

    let registry_endpoint = unique_inproc("registry");
    let registry_transport = TransportConfig::from_endpoint(&registry_endpoint);
    let registry_service = RegistryService::new(
        registry_base.path(),
        policy_client,
        registry_transport,
        signing_key.clone(),
    )
    .await?;
    let registry_handle = manager
        .spawn(Box::new(registry_service))
        .await
        .map_err(|e| anyhow!("spawn registry: {e}"))?;

    let registry_client = RegistryClient::for_local_endpoint_bootstrap(
        &registry_endpoint,
        signing_key.clone(),
        signing_key.verifying_key(),
        None,
    )?;

    Ok(Services {
        registry_client,
        repo_id: repo_id.to_string(),
        _registry_handle: registry_handle,
        _policy_handle: policy_handle,
        _policy_temp: policy_temp,
        _registry_base: registry_base,
        _signing_key: signing_key,
    })
}

/// `git clone --bare <src> <dst>`: produce a bare repo at `dst` from the
/// fixture at `src`. The registry serves linked worktrees off this bare repo.
fn bare_clone(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    use std::process::Command;
    let status = Command::new("git")
        .arg("clone")
        .arg("--bare")
        .arg(src)
        .arg(dst)
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .env("GIT_CONFIG_SYSTEM", "/dev/null")
        .status()?;
    if !status.success() {
        return Err(anyhow!("git clone --bare {src:?} -> {dst:?} failed"));
    }
    Ok(())
}

// ─── tests ──────────────────────────────────────────────────────────────────

/// The full E2E: register → create worktree → walk → readdir → cat → stat →
/// create+write → promote → new OID.
///
/// Single test so the harness setup (services + temp repo) runs once; the
/// alternative (one test per check) would each pay the ~1s service-spawn cost.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn e2e_9p_walk_read_stat_create_promote() -> Result<()> {
    install_classical_verify();

    // ── Setup: temp git repo with config.json ──────────────────────────────
    let scratch = TempDir::new()?;
    let (repo_dir, config_bytes) = make_temp_git_repo(scratch.path())?;
    let repo_name = "config-repo".to_owned();
    let worktree_name = "main".to_owned();

    // ── Setup: spawn services (fixture pre-registered in registry_base) ─────
    let svc = spawn_services(&repo_dir, &repo_name).await?;
    let caller = Subject::anonymous();
    let repo_id = svc.repo_id.clone();
    println!("registered repo {repo_name} as {repo_id}");

    // ── Ensure a worktree exists for the `main` branch ─────────────────────
    //
    // `handle_create_worktree` is idempotent: it no-ops if the worktree already
    // exists. The fixture repo's primary worktree IS `main`, but the registry
    // serves files via `get_contained_root` which looks up *linked* worktrees
    // by `file_name()`. The primary checkout's worktree path is the repo dir
    // itself (`config-repo`), whose `file_name()` is `config-repo` — NOT `main`.
    // Creating a worktree for branch `main` yields `worktrees/main`, whose
    // `file_name()` is `main`, which is what RemoteRegistryMount scopes by.
    svc.registry_client
        .repo(&repo_id)
        .create_worktree(&CreateWorktreeRequest {
            branch: worktree_name.clone(),
        })
        .await
        .map_err(|e| anyhow!("create_worktree RPC: {e}"))?;

    // ── Build the RemoteRegistryMount over the live registry client ────────
    //
    // NB: RemoteRegistryMount clones the RegistryClient (cheap Rc/Arc clone),
    // so we can keep using svc.registry_client for the promote step below.
    let mount = RemoteRegistryMount::new(Clone::clone(&svc.registry_client));

    // CHECKS 2-6 drive the mount on a blocking thread (no tokio runtime entered)
    // so its internal `self.rt.block_on` is legal — see `std_block_on`. The
    // inproc services keep responding on the main runtime while this task parks.
    {
        let repo_name = repo_name.clone();
        let worktree_name = worktree_name.clone();
        let config_bytes = config_bytes.clone();
        tokio::task::spawn_blocking(move || {
            std_block_on(async move {
    // ───────────────────────────────────────────────────────────────────────
    // CHECK 3: walk {repo}/{worktree}/ → directory fid; readdir lists real files
    // ───────────────────────────────────────────────────────────────────────
    let dir_fid = mount
        .walk(&[repo_name.as_str(), worktree_name.as_str()], &caller)
        .await
        .map_err(|e| anyhow!("walk {repo_name}/{worktree_name}: {e}"))?;
    // open the dir fid for readdir (9P readdir requires an open dir fid).
    let mut dir_fid_opened = dir_fid;
    mount
        .open(&mut dir_fid_opened, OREAD, &caller)
        .await
        .map_err(|e| anyhow!("open dir fid: {e}"))?;
    let entries = mount
        .readdir(&dir_fid_opened, &caller)
        .await
        .map_err(|e| anyhow!("readdir: {e}"))?;
    let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
    println!("readdir {repo_name}/{worktree_name}: {names:?}");
    assert!(
        names.contains(&"config.json"),
        "readdir must list config.json; got {names:?}"
    );

    // ───────────────────────────────────────────────────────────────────────
    // CHECK 2 & 4: walk config.json → open → read → real file bytes
    // ───────────────────────────────────────────────────────────────────────
    let mut file_fid = mount
        .walk(
            &[repo_name.as_str(), worktree_name.as_str(), "config.json"],
            &caller,
        )
        .await
        .map_err(|e| anyhow!("walk config.json: {e}"))?;
    mount
        .open(&mut file_fid, OREAD, &caller)
        .await
        .map_err(|e| anyhow!("open config.json: {e}"))?;

    // Read in a loop until EOF (bounded by a generous count per read; the
    // service clamps to its iounit).
    let mut buf = Vec::new();
    let mut offset = 0u64;
    loop {
        let chunk = mount
            .read(&file_fid, offset, 65536, &caller)
            .await
            .map_err(|e| anyhow!("read config.json @ {offset}: {e}"))?;
        if chunk.is_empty() {
            break;
        }
        offset += chunk.len() as u64;
        buf.extend_from_slice(&chunk);
    }
    assert_eq!(
        buf, config_bytes,
        "cat config.json must return the real committed bytes"
    );
    println!("cat config.json: {} bytes matched", buf.len());

    // ───────────────────────────────────────────────────────────────────────
    // CHECK 5: stat → qid version/path threaded (#388 Stat widening)
    // ───────────────────────────────────────────────────────────────────────
    //
    // The registry service populates version=ctime and path=ino from real
    // filesystem metadata (qid_from_metadata). On Unix, path (inode) is
    // non-zero for a real file. We assert the bytes round-trip through the
    // mount's Stat (previously these were discarded by a flattened qid).
    let stat = mount
        .stat(&file_fid, &caller)
        .await
        .map_err(|e| anyhow!("stat config.json: {e}"))?;
    println!(
        "stat config.json: qtype={:#x} version={} path={} size={} name={:?}",
        stat.qtype, stat.version, stat.path, stat.size, stat.name
    );
    // NB: the registry's Tstat handler returns an empty name
    // (`metadata_to_np_stat("", ..)`); identity (size/qtype/qid) is what's
    // threaded through, not the name. Assert on those, not on `stat.name`.
    assert_eq!(stat.size, config_bytes.len() as u64);
    // qtype: QTFILE bit must be set (0x00 = plain file; QTDIR = 0x80).
    // The registry uses QTFILE=0x00 / QTDIR=0x80, so for a file qtype & 0x80 == 0.
    assert_eq!(
        stat.qtype & 0x80,
        0,
        "config.json stat must be a file (QTDIR unset), qtype={:#x}",
        stat.qtype
    );
    #[cfg(unix)]
    {
        // Real on-disk file → inode is non-zero. (path==0 means "unknown qid"
        // per the vfs::Stat convention; a real file must not report that.)
        assert_ne!(
            stat.path, 0,
            "qid path (inode) must be threaded through for a real file (#388)"
        );
    }

    // Clunk the file fid before we create a sibling under the same dir.
    mount.clunk(file_fid, &caller).await;

    // ───────────────────────────────────────────────────────────────────────
    // CHECK 6: create + write → writable upper (#394)
    // ───────────────────────────────────────────────────────────────────────
    //
    // Walk the directory fresh (the previous dir fid was opened for readdir;
    // create needs a walked-not-opened directory fid), then create a new file
    // and write to it. The registry's handle_create consumes the dir fid and
    // returns an opened fid on the new file.
    let mut create_dir_fid = mount
        .walk(&[repo_name.as_str(), worktree_name.as_str()], &caller)
        .await
        .map_err(|e| anyhow!("walk dir for create: {e}"))?;

    let new_file = "promoted.txt";
    let new_content = b"promoted via 9P create+write (#421-E3)\n";
    let create_stat = mount
        .create(&mut create_dir_fid, new_file, 0o644, OWRITE, &caller)
        .await
        .map_err(|e| anyhow!("create {new_file}: {e}"))?;
    println!(
        "create {new_file}: qtype={:#x} version={} path={}",
        create_stat.qtype, create_stat.version, create_stat.path
    );
    assert_eq!(create_stat.name, new_file);

    let written = mount
        .write(&create_dir_fid, 0, new_content, &caller)
        .await
        .map_err(|e| anyhow!("write {new_file}: {e}"))?;
    assert_eq!(
        written as usize,
        new_content.len(),
        "write must accept all bytes (iounit permits it for this small payload)"
    );
    mount.clunk(create_dir_fid, &caller).await;

    // Verify the write landed on disk by reading it back through the 9P path.
    let mut rb_fid = mount
        .walk(
            &[repo_name.as_str(), worktree_name.as_str(), new_file],
            &caller,
        )
        .await
        .map_err(|e| anyhow!("walk {new_file} for readback: {e}"))?;
    mount
        .open(&mut rb_fid, OREAD, &caller)
        .await
        .map_err(|e| anyhow!("open {new_file} for readback: {e}"))?;
    let mut readback = Vec::new();
    let mut off = 0u64;
    loop {
        let chunk = mount
            .read(&rb_fid, off, 65536, &caller)
            .await
            .map_err(|e| anyhow!("read {new_file} @ {off}: {e}"))?;
        if chunk.is_empty() {
            break;
        }
        off += chunk.len() as u64;
        readback.extend_from_slice(&chunk);
    }
    assert_eq!(
        readback, new_content,
        "readback of created file must match what we wrote"
    );
    mount.clunk(rb_fid, &caller).await;
    println!("create+write+readback {new_file}: {} bytes matched", readback.len());

    // Clunk the directory fid we opened for readdir earlier.
    mount.clunk(dir_fid_opened, &caller).await;
            Ok::<(), anyhow::Error>(())
            })
        })
        .await??;
    }

    // ───────────────────────────────────────────────────────────────────────
    // CHECK 7: promote → stageAll + commitWithAuthor → new OID
    // ───────────────────────────────────────────────────────────────────────
    //
    // The promote saga (git/promote.rs) takes a WorktreeClient curried to the
    // resolved repo + worktree. It runs stageAll + commitWithAuthor and returns
    // the new commit OID. The atproto pointer advance is stubbed (#355/#410),
    // so we only assert the local-commit phase here.
    let wt_client = svc
        .registry_client
        .repo(&repo_id)
        .worktree(&worktree_name);

    // Capture the pre-promote HEAD so we can prove the OID advanced.
    let pre_oid = svc
        .registry_client
        .repo(&repo_id)
        .get_head()
        .await
        .map_err(|e| anyhow!("get_head (pre): {e}"))?;
    println!("pre-promote HEAD: {pre_oid}");

    let lease_table = Arc::new(PromoteLeaseTable::new());
    let author = PromoteAuthor {
        name: "e3-9p-promote".to_owned(),
        email: "e3-9p-promote@example.invalid".to_owned(),
    };
    let outcome = promote(&repo_name, &worktree_name, &wt_client, &author, &lease_table)
        .await
        .map_err(|e| anyhow!("promote: {e}"))?;
    println!("promote commit_oid: {}", outcome.commit_oid);
    assert!(
        outcome.pointer_advanced,
        "promote should report the (stubbed) pointer advance as Ok"
    );
    assert_ne!(
        outcome.commit_oid, pre_oid,
        "promote must produce a new commit OID distinct from the pre-promote HEAD"
    );
    assert!(
        !outcome.commit_oid.is_empty(),
        "promote commit OID must be non-empty"
    );

    // The commit OID should be a 40-char hex git OID.
    assert_eq!(
        outcome.commit_oid.len(),
        40,
        "git OID is 40 hex chars; got {:?}",
        outcome.commit_oid
    );
    assert!(
        outcome
            .commit_oid
            .chars()
            .all(|c| c.is_ascii_hexdigit()),
        "commit OID must be hex: {}",
        outcome.commit_oid
    );

    Ok(())
}

/// A second walk that creates a *directory* via DMDIR, to cover the
/// `is_dir` branch of `handle_create` (the main test creates a plain file).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn e2e_9p_create_directory_and_stat_qtype() -> Result<()> {
    install_classical_verify();

    let scratch = TempDir::new()?;
    let (repo_dir, _config_bytes) = make_temp_git_repo(scratch.path())?;
    let repo_name = "config-repo".to_owned();
    let worktree_name = "main".to_owned();

    let svc = spawn_services(&repo_dir, &repo_name).await?;
    let caller = Subject::anonymous();
    let repo_id = svc.repo_id.clone();
    svc.registry_client
        .repo(&repo_id)
        .create_worktree(&CreateWorktreeRequest {
            branch: worktree_name.clone(),
        })
        .await?;

    let mount = RemoteRegistryMount::new(Clone::clone(&svc.registry_client));

    // Drive the mount on a blocking thread (no tokio runtime entered there) so
    // its internal `self.rt.block_on` is legal — see `std_block_on`. The inproc
    // services keep responding on the main runtime while this task parks.
    tokio::task::spawn_blocking(move || {
        std_block_on(async move {
            // Walk the worktree root, then create a subdirectory.
            let mut dir_fid = mount
                .walk(&[repo_name.as_str(), worktree_name.as_str()], &caller)
                .await
                .map_err(|e| anyhow!("walk root: {e}"))?;

            let subdir = "e3-subdir";
            let create_stat = mount
                .create(&mut dir_fid, subdir, DMDIR | 0o755, OREAD, &caller)
                .await
                .map_err(|e| anyhow!("create dir {subdir}: {e}"))?;

            // After create with DMDIR, the fid points at the new directory and
            // the returned Stat's qtype must carry QTDIR (0x80).
            assert_ne!(
                create_stat.qtype & 0x80,
                0,
                "create(DMDIR) must return a qtype with QTDIR set; got {:#x}",
                create_stat.qtype
            );
            assert_eq!(create_stat.name, subdir);
            println!(
                "create dir {subdir}: qtype={:#x} (QTDIR set), version={} path={}",
                create_stat.qtype, create_stat.version, create_stat.path
            );

            // Stat the fid (now the new directory) — should agree on qtype.
            // NB: the registry's Tstat handler builds its RStat with an empty
            // name (`metadata_to_np_stat("", ..)`); the name is carried by
            // walk/create qids, not stat-by-fid. So we assert on qtype only.
            let stat = mount.stat(&dir_fid, &caller).await?;
            assert_ne!(stat.qtype & 0x80, 0, "stat on created dir must report QTDIR");

            mount.clunk(dir_fid, &caller).await;
            Ok::<(), anyhow::Error>(())
        })
    })
    .await??;

    Ok(())
}
