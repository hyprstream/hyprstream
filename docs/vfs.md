# VFS — Plan 9 Namespace for hyprstream

## Overview

hyprstream exposes its runtime as a Plan 9-inspired virtual filesystem.
Services, configuration, and commands are paths in a per-session namespace.
The namespace multiplexer lives in the `hyprstream-vfs` crate; the shell
lives in `hyprstream-workers-tcl`. The shell language is Tcl (via the
[molt](https://github.com/wduquette/molt) fork), chosen because:

- Tcl is already a "shell for everything" — variables, control flow, and string
  interpolation come free.
- Unknown commands resolve via `/bin/` (the Plan 9 PATH model), so the shell
  and filesystem are the same thing.
- molt compiles to WASM, keeping the door open for browser-side evaluation.

The core VFS is **client-side**: a mount table that routes operations by
longest-prefix match.  Services run on the server; the namespace is how the
client addresses them.  Two companion crates also expose a namespace
*server-side*:

- `hyprstream-9p` — a 9P2000.L codec + translator that bridges 9P clients
  (e.g. Wanix in the browser) to hyprstream RPC.
- `hyprstream-vfs-server` — a vhost-user-fs server that exposes a VFS
  `Namespace` to a Cloud Hypervisor microVM guest (#362), so workers see the
  same namespace as a mounted filesystem.

## Namespace layout

```
/bin/                Built-in commands (cat, ls, write, ctl, json, help, man, stream, mount)
/env/                Session variables (temperature, top_p, etc.)
/config/             Persistent configuration
/srv/model/          Model service (proxied via RemoteModelMount over RPC)
/srv/mcp/            MCP service
/private/            Per-tenant private data
/net/peer-a/srv/...  Federated peer namespaces (future)
```

`ls /` lists the top-level mount points.  `ls /srv/model/qwen3:main` lists
files inside a loaded model's synthetic tree (e.g., `status`, `ctl`, `tokenizer`).

## Using the shell

In the TUI chat, lines starting with `/` are interpreted as VFS commands.
Lines starting with `//` are sent to the model as regular messages (escape hatch).

### Bare paths vs commands

| Input | Interpretation |
|-------|----------------|
| `/srv/model/qwen3:main/status` | Bare path (no spaces, contains `/`) — treated as `cat /srv/model/qwen3:main/status` |
| `/ls /srv` | Command — leading `/` stripped, evaluated as `ls /srv` |
| `/help` | Command — evaluated as `help` |
| `//tell me about Plan 9` | Double-slash escape — sent to model as `tell me about Plan 9` |

After the leading `/` is stripped, the result is evaluated as a Tcl script.
Standard Tcl features work: `set x [cat /config/temperature]; expr {$x + 0.1}`.

## Available commands

### `cat <path> [path ...]`

Read file contents.  Multiple paths are concatenated.

```
/cat /config/temperature
0.7
```

### `ls [path]`

List directory entries.  Defaults to `/` (all mount points).
Directories are shown with a trailing `/`.

```
/ls /config
temperature
status
```

### `write <path> <data>`

Write data to a VFS file.

```
/write /env/temperature 0.9
```

### `ctl <path> <command> [args...]`

Write a command to a control file and read the response (Plan 9 ctl pattern:
open RDWR, write, read on same fid).

```
/ctl /srv/model/qwen3:main/ctl generate "Hello, world"
```

### `mount [prefix]`

List all mount points, or check if a specific prefix is mounted (read-only —
it does not modify the mount table).

```
/mount
/config
/srv/model
/bin
```

### `json parse <value>`

Convert a JSON string to a Tcl dict (useful for post-processing `cat` output
from services that emit JSON).

### `man [service [method ...]]`

Display service documentation derived from the schema annotations.

### `stream <path> <method> <args> <varname> <body>`

Start a stream from a service and iterate over received blocks, evaluating
`body` with each block bound to `varname`.

### `help`

List VFS builtins and any commands found in `/bin/`.

## `/bin/` duality

Builtins (`cat`, `ls`, `write`, `ctl`, `json`, `help`, `man`, `stream`,
`mount`) are registered directly in the Tcl interpreter.  They shadow any `/bin/` file of the same name —
the same model as bash, where builtins take priority over `$PATH` entries.

When the interpreter encounters an unknown command name `foo`, it attempts
`ctl /bin/foo <args>` before reporting an error.  This means a mount at `/bin/`
can provide additional commands without modifying the interpreter:

```
# "load" is not a builtin, but /bin/load exists as a ctl file
/load qwen3:main
ok: qwen3:main
```

Both mechanisms exist because:

1. Builtins need direct access to the interpreter (e.g., `help` introspects
   registered commands, `mount` reads the namespace).
2. `/bin/` entries are dynamically discoverable and can be added by services
   at runtime without recompiling.

## Security

The Tcl interpreter is a **guest** — it has no direct host I/O.

### Removed molt commands

These commands are removed at construction time:

| Command | Reason |
|---------|--------|
| `source` | Reads arbitrary host files via `std::fs::read_to_string` |
| `exit` | Calls `std::process::exit()` — kills the server |
| `puts` | Writes to server stdout (log injection) |
| `rename` | Can shadow or delete security-critical builtins |
| `global` | Scope manipulation unnecessary in VFS shell |
| `time` | User-controlled iteration count — CPU bomb |
| `parse`, `pdump`, `pclear` | Debug internals |
| `apply`, `eval`, `subst`, `uplevel` | Evaluate arbitrary code — enable injection via field output |

### Limits

- **Instruction limit**: 100,000 per eval (prevents infinite loops).
  Configurable via `TclShell::set_instruction_limit()`.
- **Recursion limit**: 100 (molt default override).

### Path sanitization

- Command name resolution rejects names containing `/` or `..` (prevents
  path traversal via the `/bin/` fallback).
- The namespace normalizes all paths: resolves `.` and `..`, strips duplicate
  slashes, enforces leading `/`.

### Subject identity

Every `Mount` method receives the caller's `Subject` (from `hyprstream_rpc`).
Mounts use this for per-tenant fid isolation and access control.  The subject
is set once when the `TclShell` is constructed and cannot be changed by Tcl code.

## Mount trait

The `Mount` trait (`hyprstream_vfs::Mount`, `crates/hyprstream-vfs/src/mount.rs`)
is the universal abstraction for all VFS backends.  It is an **async** trait
(`#[async_trait]`; `?Send` on `wasm32`) mirroring 9P2000 operations, and
**every method receives the verified `Subject` of the caller** for per-tenant
fid isolation and policy checks:

```rust
#[async_trait]
pub trait Mount: Send + Sync {
    async fn walk(&self, components: &[&str], caller: &Subject) -> Result<Fid, MountError>;
    async fn open(&self, fid: &mut Fid, mode: u8, caller: &Subject) -> Result<(), MountError>;
    async fn create(&self, fid: &mut Fid, name: &str, perm: u32, mode: u8, caller: &Subject)
        -> Result<Stat, MountError>;   // default: NotSupported
    async fn read(&self, fid: &Fid, offset: u64, count: u32, caller: &Subject) -> Result<Vec<u8>, MountError>;
    async fn write(&self, fid: &Fid, offset: u64, data: &[u8], caller: &Subject) -> Result<u32, MountError>;
    async fn readdir(&self, fid: &Fid, caller: &Subject) -> Result<Vec<DirEntry>, MountError>;
    async fn stat(&self, fid: &Fid, caller: &Subject) -> Result<Stat, MountError>;
    async fn clunk(&self, fid: Fid, caller: &Subject);
    fn as_fsmount(&self) -> Option<&dyn FsMount> { None }
}
```

`create` (9P `Tcreate`) defaults to `NotSupported`, so read-only synthetic
mounts stay writable-mount-free; only genuinely writable mounts (e.g. the
registry worktree mount, the union's copy-up upper layer) override it.

Full namespace mutation (`unlink`/`mkdir`/`rename`/…) lives on the `FsMount`
supertrait.  Because a `Namespace` stores every mount type-erased as
`Arc<dyn Mount>`, the `as_fsmount()` capability downcast is how a consumer
holding only a `dyn Mount` gets back to the `FsMount` vtable — the
capability-typed answer to "is this mount writable as a real filesystem?",
instead of probing for `NotSupported` at runtime.

The `Namespace` provides convenience methods that compose these primitives:
`cat` (walk+open+read_all+clunk), `read_one`, `echo` (walk+open+write+clunk),
`create`, `ctl` (walk+open(RDWR)+write+read+clunk), `ls`
(walk+open+readdir+clunk).

## Composition: bind mounts

Per ADR #651, namespace composition is expressed entirely as **bind mounts**
(`crates/hyprstream-vfs/src/namespace.rs`):

```rust
pub enum BindFlag { Replace, Before, After, Upper }

impl Namespace {
    pub fn bind_mount(&mut self, prefix: &str, target: Arc<dyn Mount>, flag: BindFlag)
        -> Result<(), NamespaceError>;
    // mount(prefix, target) == bind_mount(prefix, target, BindFlag::Replace)
}
```

- `Replace` — removes any existing mount at the prefix.
- `Before` / `After` — union semantics: the prefix's `MountEntry` holds a
  `targets: Vec<...>` list; walk tries targets in bind order, and readdir
  merges entries across all targets with **first-seen wins** deduplication.
- `Upper` — additionally records the target as the union's writable
  **copy-up upper layer** (#370).  The upper is placed first so it wins on the
  read path; a write to a path that exists only in a read-only lower layer is
  copied up to (and `create`d in) the upper before the write proceeds.

Overlay/union behavior is therefore a *policy over bind mounts* — a
write-routing and readdir-merge discipline on `MountEntry.targets` — **not a
separate overlay primitive**.  (The fuse-based POSIX `OverlayFs` used for
worker rootfs images is just another leaf that can be bound in.)

### RemoteModelMount

`RemoteModelMount` (`crates/hyprstream/src/services/remote_mount.rs`) bridges
the `Mount` trait to the generated `ModelClient` RPC: each async `Mount` call
becomes an RPC request to the model service.  Local fid numbers are tracked
in a `DashMap` (via the shared `NinePBridge`, also used by
`RemoteRegistryMount`) and mapped to remote fids returned by the model
service's `FsHandler`.  This allows `/srv/model` to proxy 9P operations to
the server-side `SyntheticTree` transparently.

## Tcl as compound protocol

The VFS shell supports three execution modes, all expressed as Tcl:

### 1. Local/local

Standard Tcl + VFS builtins operating on client-side mounts.

```tcl
set t [cat /config/temperature]
expr {$t + 0.1}
```

### 2. Local/remote

Tcl runs locally; VFS operations transparently proxy to remote services
via `RemoteModelMount` (or any future `Mount` implementation that wraps RPC).

```tcl
cat /srv/model/qwen3:main/status
ctl /srv/model/qwen3:main/ctl generate "Hello"
```

The user does not know (or care) whether the mount is local or remote.
The namespace abstracts transport.

### 3. Remote eval (future)

A federated peer's namespace is mounted at `/net/peer-a/srv/...`.
Tcl scripts that reference those paths cause transparent cross-node
RPC — the same commands work regardless of where the service lives.
