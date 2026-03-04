# Wizard Architecture

## 1. Crate Dependency Graph

```
+-------------------------------------------------------------------------+
|                        hyprstream (native binary)                       |
|                                                                         |
|  +-------------------+  +--------------+  +---------------------------+ |
|  | wizard_handlers   |  |  gpu_detect  |  |    update_handlers        | |
|  |                   |  |              |  |                           | |
|  | handle_wizard_    |  | detect_gpu() |  | re_exec_variant()        | |
|  |   tui()          |  | detect_run_  |  | check_should_reexec()    | |
|  |                   |  |   mode()     |  | handle_update()          | |
|  +--------+----------+  +--------------+  +---------------------------+ |
|           |                                                             |
|           | creates                                                     |
|           v                                                             |
|  +----------------------------------------------+  +-----------------+ |
|  |         BootstrapManager                     |  |   release_store | |
|  |                                               |  |                 | |
|  |  rt: Handle   models_dir   signing_key        |  | ReleaseManifest | |
|  |  install_rx   install_cancel                  |  | VariantManifest | |
|  |  bootstrap_rx bootstrap_cancel                |  | FileCategories  | |
|  |  service_rx   policy_manager                  |  +-----------------+ |
|  +---------------+------------------------------+                       |
|                  | implements                                           |
+------------------+------------------------------------------------------+
                   |
    ---------------+------------- crate boundary --------------------------
                   |
+------------------+------------------------------------------------------+
|                  v           hyprstream-tui (WASI-safe)                  |
|  +----------------------------------------------+                      |
|  |       <<trait>> WizardBackend                 |                      |
|  |                                               |                      |
|  |  detect_environment()  -> EnvironmentInfo     |                      |
|  |  recommend_action()    -> InstallAction       |                      |
|  |  start_install()       -> (async launch)      |                      |
|  |  poll_install()        -> InstallPoll         |                      |
|  |  start_bootstrap()     -> (async launch)      |                      |
|  |  poll_bootstrap()      -> BootstrapPoll       |                      |
|  |  apply_template()      save_policies()        |                      |
|  |  add_user()            generate_token()       |                      |
|  |  start_services()      poll_pending()         |                      |
|  |  local_username()      templates()            |                      |
|  +----------------------+------------------------+                      |
|           ^              |                                              |
|           |              | parameterizes                                |
|           |              v                                              |
|  +--------+------+  +---------------------------+                       |
|  |MockWizardBack-|  |  WizardApp<B: Backend>    |                       |
|  |end            |  |                           |                       |
|  |               |  |  phase: WizardPhase       |                       |
|  | tick counters |  |  backend: B               |                       |
|  | canned data   |  |  install_summary          |                       |
|  +---------------+  |  users_created            |                       |
|                     |  tokens_generated         |                       |
|   (WASI / tests)    +------------+--------------+                       |
|                                  | drives                               |
|                                  v                                      |
|                     +------------------------+                          |
|                     |    WizardPhase         |                          |
|                     |                        |                          |
|                     |  Install(Screen)       |                          |
|                     |  Bootstrap(Screen)     |                          |
|                     |  PolicyTemplate(..)    |                          |
|                     |  Users(Screen)         |                          |
|                     |  Tokens(Screen)        |                          |
|                     |  Services(Screen)      |                          |
|                     |  Summary(Screen)       |                          |
|                     +------------------------+                          |
+-------------------------------------------------------------------------+
```

## 2. Async Bridge Pattern (TUI <-> Tokio)

```
  spawn_blocking thread                       Tokio runtime
  (TUI at ~30fps)                             (async I/O)
 +-----------------------+                    +------------------------+
 |                       |                    |                        |
 |  WizardApp::tick()    |                    |                        |
 |       |               |                    |                        |
 |       v               |   sync_channel(8)  |                        |
 |  backend.poll_*()  ---+---- try_recv() <---+-- tx.send(progress)    |
 |       |               |   drain-to-latest  |       |                |
 |       |               |                    |       |                |
 |  backend.start_*() --+---- rt.spawn() ----+-->  do_install()       |
 |                       |                    |     do_bootstrap()     |
 |  backend.apply_*() --+---- rt.block_on() -+-->  PolicyManager      |
 |                       |   (safe: we're on  |     write_policy()     |
 |                       |    blocking thread) |                        |
 |                       |                    |                        |
 |  Drop::drop()        |                    |                        |
 |   cancel.store(T) ---+--------------------+---> cancel.load() ->   |
 |   handle.abort()  ---+--------------------+---> task aborted        |
 +-----------------------+                    +------------------------+

  Channel semantics:
  +-------------------------------------------------------+
  |  mpsc::sync_channel(8)  -- bounded, backpressure      |
  |                                                       |
  |  poll pattern (drain-to-latest):                      |
  |    latest = None                                      |
  |    while let Ok(msg) = rx.try_recv() {                |
  |        is_terminal = matches!(Done | Failed)          |
  |        latest = Some(msg)                             |
  |        if is_terminal { break }  <-- stop early       |
  |    }                                                  |
  |    if terminal { self.rx = None } <-- prevent stale   |
  |    return latest.unwrap_or(default)                   |
  +-------------------------------------------------------+
```

## 3. Wizard Phase State Machine

```
                    +--------------+
                    |   Start      |
                    +------+-------+
                           |
                           v
              +---------------------------+
              |  Install                  |
              |                           |
              |  Detecting                |
              |    |                      |
              |    v                      |
              |  ShowFindings ------+     |
              |    |   (Skip)      |     |
              |    |(Upgrade)      |     |
              |    v               |     |
              |  SelectVariant     |     |
              |    |               |     |
              |    v               |     |
              |  Installing ----+  |     |
              |    |            |  |     |
              |    v            v  v     |
              |  Done        Failed      |
              |    |            |        |
              |    |       Skipped       |
              +----+--------+--+---------+
                   |        |  |
                   v        v  v
              +---------------------------+
              |  Bootstrap                |
              |                           |
              |  started=false            |
              |    | start_bootstrap()    |
              |    v                      |
              |  InProgress -------+      |
              |    |               |      |
              |    v               v      |
              |  Done           Failed    |
              +----+-----------------------+
                   |
                   v
              +---------------------------+
              |  PolicyTemplate           |
              |                           |
              |  has_existing_policy? -----+-- yes -> ConfirmReplace
              |    | no                   |            |
              |    v                      |            v
              |  SelectTemplate           |     (replace / skip)
              |    |                      |
              |    v                      |
              |  Applied / Skipped        |
              +----+-----------------------+
                   |
                   v
              +---------------------------+
              |  Users (loop)             |
              |                           |
              |  AskAdd ---- no ----> Done|
              |    | yes                  |
              |    v                      |
              |  EnterName                |
              |    |                      |
              |    v                      |
              |  SelectRole               |
              |    |                      |
              |    v (custom?)            |
              |  CustomResource --+       |
              |  CustomActions    |       |
              |    |<-------------+       |
              |    v                      |
              |  AskAnother ---> loop     |
              +----+-----------------------+
                   |
                   v
              +---------------------------+
              |  Tokens (per user)        |
              |                           |
              |  AskGenerate -- no -> Next|
              |    | yes                  |
              |    v                      |
              |  SelectExpiry             |
              |    |                      |
              |    v                      |
              |  ShowToken ---> NextUser  |
              |                  |        |
              |           all done -> Done|
              +----+-----------------------+
                   |
                   v
              +---------------------------+
              |  Services                 |
              |                           |
              |  AskStart -- no ---+      |
              |    | yes           |      |
              |    v               |      |
              |  Starting          |      |
              |    |               |      |
              |    v               v      |
              |  Done(started?)           |
              +----+-----------------------+
                   |
                   v
              +---------------------------+
              |  Summary                  |
              |                           |
              |  install_result           |
              |  templates_applied        |
              |  users_created            |
              |  tokens_generated         |
              |  services_started         |
              |                           |
              |  [Enter] -> quit          |
              +---------------------------+
```

## 4. GPU Detection and Re-exec Flow

```
  +-------------------------------------------------------------------+
  |                     Process Startup                                |
  |                                                                    |
  |  main()                                                            |
  |    |                                                               |
  |    +-- HYPRSTREAM_LIBTORCH_CONFIGURED=1?                           |
  |    |     yes -> skip re-exec (already running GPU binary)          |
  |    |     no  |                                                     |
  |    |         v                                                     |
  |    +-- check_should_reexec(data_dir)                               |
  |    |     |                                                         |
  |    |     +-- read active-backend -> variant_from_id()              |
  |    |     |     (allowlist: cpu|cuda128|cuda130|rocm71)             |
  |    |     |                                                         |
  |    |     +-- read active-version -> is_safe_version_string()       |
  |    |     |     (validates: no "..", only [a-zA-Z0-9.-])            |
  |    |     |                                                         |
  |    |     +-- binary exists at expected path?                       |
  |    |     |     no  -> None (continue as CPU)                       |
  |    |     |     yes -> Some(variant_id, version)                    |
  |    |     |                                                         |
  |    |     +-- re_exec_variant()                                     |
  |    |           |                                                   |
  |    |           +-- LD_LIBRARY_PATH = .../libtorch/lib:$existing    |
  |    |           +-- HYPRSTREAM_LIBTORCH_CONFIGURED = 1              |
  |    |           +-- exec() <-- replaces process (Unix execve)       |
  |    |                                                               |
  |    +-- normal CLI dispatch (running as CPU or post-reexec)         |
  +-------------------------------------------------------------------+

  Data directory layout:
  +------------------------------------------------+
  |  ~/.local/share/hyprstream/                     |
  |  +-- active-backend          "cuda130"          |
  |  +-- active-version          "0.1.0"            |
  |  +-- versions/                                  |
  |      +-- 0.1.0-cuda130/                         |
  |      |   +-- backends/                          |
  |      |       +-- cuda130/                       |
  |      |           +-- hyprstream    (binary)     |
  |      |           +-- manifest.toml              |
  |      |           +-- libtorch/                  |
  |      |               +-- lib/                   |
  |      |                   +-- libtorch.so        |
  |      |                   +-- libtorch_cuda.so   |
  |      |                   +-- libc10.so          |
  |      +-- 0.1.0-cpu/       (old, cleanable)      |
  +------------------------------------------------+
```

## 5. AppImage Backend Selection

```
  +-----------------------------------------------------+
  |  AppRun                                              |
  |                                                      |
  |  1. Wizard-installed backend (highest priority)      |
  |     +-------------------------------------------+    |
  |     | read ~/.local/share/hyprstream/            |    |
  |     |   active-backend + active-version          |    |
  |     |         |                                  |    |
  |     |  GPU_BINARY = versions/{ver}-{be}/         |    |
  |     |              backends/{be}/hyprstream       |    |
  |     |         |                                  |    |
  |     |    -x GPU_BINARY?                          |    |
  |     |    yes -> set LD_LIBRARY_PATH, exec        |    |
  |     |    no  -> fall through                     |    |
  |     +-------------------------------------------+    |
  |                    |                                 |
  |  2. Bundled backend detection                        |
  |     +-------------------------------------------+    |
  |     | $HYPRSTREAM_BACKEND set?                   |    |
  |     |   yes -> use it directly                   |    |
  |     |   no  |                                    |    |
  |     |       v                                    |    |
  |     | nvidia-smi?                                |    |
  |     |   +-- driver >= 555 -> cuda130             |    |
  |     |   +-- driver >= 535 -> cuda128             |    |
  |     |   +-- driver <  535 -> cpu                 |    |
  |     |                                            |    |
  |     | rocm-smi?                                  |    |
  |     |   +-- yes -> rocm71                        |    |
  |     |                                            |    |
  |     | none -> cpu                                |    |
  |     +-------------------+-----------------------+    |
  |                         |                            |
  |  3. Validate + launch bundled binary                 |
  |     +-------------------+-----------------------+    |
  |     | $APPDIR/usr/bin/hyprstream-$BACKEND        |    |
  |     | $APPDIR/usr/lib/$BACKEND/libtorch/lib      |    |
  |     |   exists? -> set env, exec bundled         |    |
  |     |   no?     -> fallback to cpu               |    |
  |     +-------------------------------------------+    |
  +-----------------------------------------------------+
```

## 6. git2db Pathspec-Filtered Checkout

```
  RepositoryHandle::create_filtered_worktree(
      path, "release", ["backends/cuda130/", "manifest.toml"]
  )
       |
       v
  DriverOpts {
      checkout_paths: Some(["backends/cuda130/", "manifest.toml"])
  }
       |
       v (VFS or Overlay2 driver)
  +----------------------------------------------------------+
  |  1. git2::repo.worktree()  -- full checkout              |
  |     (libgit2 has NO sparse checkout support)             |
  |                                                          |
  |  Working tree after step 1:                              |
  |  +-- backends/                                           |
  |  |   +-- cpu/hyprstream         <-- unwanted             |
  |  |   +-- cpu/manifest.toml      <-- unwanted             |
  |  |   +-- cuda130/hyprstream     [ok] wanted              |
  |  |   +-- cuda130/manifest.toml  [ok] wanted              |
  |  |   +-- rocm71/...             <-- unwanted             |
  |  +-- manifest.toml              [ok] wanted              |
  +------------------------------+---------------------------+
                                 |
                                 v
  +----------------------------------------------------------+
  |  2. apply_pathspec_filter()                              |
  |                                                          |
  |  For each index entry:                                   |
  |    path_matches_keep("backends/cpu/hyprstream",          |
  |                      ["backends/cuda130/",               |
  |                       "manifest.toml"])                  |
  |                                                          |
  |    "backends/cuda130/" ends with / -> prefix match       |
  |    "manifest.toml" no /  -> exact match only             |
  |                                                          |
  |  a) Delete non-matching files from working tree          |
  |  b) remove_empty_dirs() -- cleans backends/cpu/ etc.     |
  |  c) mark_skip_worktree():                                |
  |       set flags_extended |= 0x4000 on excluded           |
  |       entries, write index                               |
  |                                                          |
  |  Working tree after step 2:                              |
  |  +-- backends/                                           |
  |  |   +-- cuda130/hyprstream     [ok]                     |
  |  |       cuda130/manifest.toml  [ok]                     |
  |  +-- manifest.toml              [ok]                     |
  |                                                          |
  |  Index after step 2:                                     |
  |  +------------------------------------------+            |
  |  | backends/cpu/hyprstream    SKIP=0x4000    |            |
  |  | backends/cpu/manifest.toml SKIP=0x4000    |            |
  |  | backends/cuda130/hyprstream    (normal)   |            |
  |  | backends/cuda130/manifest.toml (normal)   |            |
  |  | backends/rocm71/...        SKIP=0x4000    |            |
  |  | manifest.toml                  (normal)   |            |
  |  +------------------------------------------+            |
  +----------------------------------------------------------+
```

## 7. GittorrentStorage -- P2P Distribution

```
  +-----------------------------------------------------------+
  |  git-xet-filter                                            |
  |                                                            |
  |  URL dispatch (config.rs):                                 |
  |    "gittorrent://cas.example.com" -> GittorrentStorage     |
  |    "https://cas.example.com"      -> XetStorage            |
  |    "ssh://..."                    -> XetStorage            |
  |                                                            |
  |  +---------------------------------------------------+    |
  |  |  <<trait>> StorageBackend                          |    |
  |  |                                                    |    |
  |  |  clean_file()  clean_bytes()  is_pointer()         |    |
  |  |  smudge_file() smudge_bytes()                      |    |
  |  |  smudge_from_hash() smudge_from_hash_to_file()     |    |
  |  +----------------+------------------+----------------+    |
  |                   |                  |                     |
  |         +---------+------+    +------+--------+            |
  |         | XetStorage     |    |Gittorrent-    |            |
  |         |                |    |  Storage      |            |
  |         | CAS (HTTPS)    |    |               |            |
  |         | upload/        |    | service: Arc  |            |
  |         |  download      |    |  <GitTorrent  |            |
  |         |  sessions      |    |   Service>    |            |
  |         +----------------+    |               |            |
  |                               | fallback:     |            |
  |                               |  Option<Box<  |            |
  |                               |   dyn Storage |            |
  |                               |   Backend>>   |            |
  |                               +-------+-------+            |
  |                                       |                    |
  +---------------------------------------+--------------------+
                                          |
                             smudge_from_hash(merkle_hash)
                                          |
                                          v
                         +----------------------------------+
                         |  1. Try DHT (P2P)                |
                         |     MerkleHash -> Sha256Hash     |
                         |     service.get_object(&hash)    |
                         |         |                        |
                         |    found? -> return data         |
                         |    miss?  |                      |
                         |           v                      |
                         |  2. Fallback to HTTPS CAS        |
                         |     fallback.smudge_from_hash()  |
                         |         |                        |
                         |    found? -> return data         |
                         |    miss?  -> DownloadFailed      |
                         +----------------------------------+

  Pointer formats:
  +--------------------------------------------------------+
  |  Standard XET:       {"version":"1.0","filesize":N,    |
  |                       "hash":"<merklehash>"}           |
  |                                                        |
  |  Gittorrent XET:     {"xet":"gittorrent",              |
  |                       "sha256":"<hex64>",              |
  |                       "size":N}                        |
  |                                                        |
  |  is_pointer() discriminator:                           |
  |    XetStorage:         parses as XetFileInfo JSON      |
  |    GittorrentStorage:  contains "xet":"gittorrent"     |
  +--------------------------------------------------------+
```

## 8. Release Registry Structure

```
  hyprstream-releases.git (bare repo, git2db-managed)
  |
  +-- .gitattributes          "*.so filter=xet"
  |                           "hyprstream filter=xet"
  |
  +-- manifest.toml           <-- ReleaseManifest
  |   +------------------------------------------+
  |   | [release]                                |
  |   | version = "0.1.0"                        |
  |   | libtorch_version = "2.10.0"              |
  |   | abi = "pre-cxx11"                        |
  |   |                                          |
  |   | [variants]                               |
  |   | cpu    = { host_requires = "glibc" }     |
  |   | cuda128= { host_requires = "nvidia" }    |
  |   | cuda130= { host_requires = "nvidia" }    |
  |   | rocm71 = { host_requires = "rocm" }      |
  |   +------------------------------------------+
  |
  +-- backends/
      +-- cpu/
      |   +-- hyprstream              XET pointer -> binary
      |   +-- manifest.toml           <-- VariantManifest
      |   |   +----------------------------------------+
      |   |   | [variant]                              |
      |   |   | id = "cpu"                             |
      |   |   | total_size_bytes = 800_000_000         |
      |   |   |                                        |
      |   |   | [files]                                |
      |   |   | runtime = ["hyprstream",               |
      |   |   |   "libtorch/lib/libtorch.so", ...]     |
      |   |   | dev  = ["libtorch/include/**"]         |
      |   |   | test = ["libtorch/lib/*_test*"]        |
      |   |   +----------------------------------------+
      |   +-- libtorch/lib/
      |       +-- libtorch.so
      |       +-- libc10.so ------+  same merkle hash
      |                           |  (XET CAS dedup)
      +-- cuda130/                |
      |   +-- hyprstream          |
      |   +-- manifest.toml       |
      |   |   +----------------------------------------+
      |   |   | [variant]                              |
      |   |   | id = "cuda130"                         |
      |   |   | [host]                                 |
      |   |   | min_driver_version = "555.0"           |
      |   |   | cuda_compat = "13.0"                   |
      |   |   +----------------------------------------+
      |   +-- libtorch/lib/
      |       +-- libtorch.so         (different)
      |       +-- libtorch_cuda.so    (CUDA-only)
      |       +-- libc10.so ----------+  deduped via CAS
      |
      +-- rocm71/
          +-- ...

  Selective download via runtime_pathspecs():
  +-------------------------------------------------------+
  |  VariantManifest { variant.id = "cuda130", ... }      |
  |                                                       |
  |  .runtime_pathspecs() -> [                            |
  |      "backends/cuda130/hyprstream",                   |
  |      "backends/cuda130/libtorch/lib/libtorch.so",     |
  |      "backends/cuda130/libtorch/lib/libc10.so",       |
  |      ...                                              |
  |      "manifest.toml"                                  |
  |  ]                                                    |
  |                                                       |
  |  -> passed to create_filtered_worktree()              |
  |  -> only runtime files checked out (~1.8GB)           |
  |  -> dev/test files skipped (~500MB saved)             |
  |  -> CAS dedup: shared libs downloaded once            |
  +-------------------------------------------------------+
```

## 9. Type Map -- All New Types by Crate

```
  hyprstream-tui                    hyprstream                    git2db
  ==============                    ==========                    ======
  <<trait>> WizardBackend           BootstrapManager             DriverOpts
  MockWizardBackend                   (implements trait)            .checkout_paths
  WizardApp<B>
                                    gpu_detect:                   <<trait>> Driver
  GpuKind                            detect_gpu()                 VfsDriver
    Nvidia{driver,cuda}               detect_run_mode()           Overlay2Driver
    Amd{rocm}                         detect_installed_variant()
    None                              detect_active_version()     WorktreeHandle
                                      variant_from_id()
  LibtorchVariant                     recommend_variant()         apply_pathspec_filter()
    Cpu|Cuda128|Cuda130|Rocm71        detect_environment()        mark_skip_worktree()
                                                                  path_matches_keep()
  RunMode                           update_handlers:
    AppImage|BareBinary|Dev            re_exec_variant()
                                      check_should_reexec()     git-xet-filter
  EnvironmentInfo                     handle_update()            ===============
    run_mode, gpu,                    handle_cleanup()           <<trait>> StorageBackend
    current/recommended/                                         XetStorage
    installed variant               release_store:               GittorrentStorage
                                      ReleaseManifest
  InstallAction                       VariantManifest
    Skip|Upgrade|AlreadyCurrent       FileCategories
                                      HostRequirements
  InstallPoll
    Detecting|Downloading|
    Extracting|Configuring|
    Done|Failed

  BootstrapPoll
    InProgress|Done|Failed

  OpStatus
    InProgress|Done|Failed

  TokenResult {token, expires}
  TemplateInfo {name, desc}

  WizardPhase
    Install|Bootstrap|Policy|
    Users|Tokens|Services|Summary

  InstallScreen
    Detecting|ShowFindings|
    SelectVariant|Installing|
    Done|Skipped|Failed

  BootstrapScreen
  PolicyScreen
  UserScreen + UserRecord
  TokenScreen + TokenRecord
  ServiceScreen
  SummaryScreen
```
