# VFS — Plan 9 Namespace for hyprstream

## Overview

hyprstream exposes its runtime as a Plan 9-inspired virtual filesystem.
Services, configuration, and commands are paths in a per-session namespace.
The shell language is Tcl (via the [molt](https://github.com/wduquette/molt) fork),
chosen because:

- Tcl is already a "shell for everything" — variables, control flow, and string
  interpolation come free.
- Unknown commands resolve via `/bin/` (the Plan 9 PATH model), so the shell
  and filesystem are the same thing.
- molt compiles to WASM, keeping the door open for browser-side evaluation.

The VFS is **client-side**: a mount table that routes operations by longest-prefix
match.  Services run on the server; the namespace is how the client addresses them.

## Namespace layout

```
/bin/                Built-in commands (cat, ls, echo, ctl, help, mount)
/env/                Session variables (temperature, top_p, etc.)
/config/             Persistent configuration
/srv/model/          Model service (proxied via RemoteModelMount over ZMQ)
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

### `echo <path> <data>`

Write data to a VFS file.

```
/echo /env/temperature 0.9
```

### `ctl <path> <command> [args...]`

Write a command to a control file and read the response (Plan 9 ctl pattern:
open RDWR, write, read on same fid).

```
/ctl /srv/model/qwen3:main/ctl generate "Hello, world"
```

### `mount [prefix]`

List all mount points, or check if a specific prefix is mounted.

```
/mount
/config
/srv/model
/bin
```

### `help`

List VFS builtins and any commands found in `/bin/`.

## Bind mounts and union directories

The namespace supports Plan 9-style `bind` in addition to `mount`.  While
`mount` attaches a `Mount` implementation at a prefix, `bind` overlays one
namespace path onto another, creating union directories.

```rust
ns.bind_mount("/tools", "/bin", BindFlag::Before)?;
```

`BindFlag` controls union order:

| Flag | Behavior |
|------|----------|
| `BindFlag::Before` | New entries appear first in readdir / shadow existing names |
| `BindFlag::After` | New entries appear after existing ones (fallback) |
| `BindFlag::Replace` | Completely replaces the target |

When `ls` is called on a union directory, entries from all bound mounts are
merged.  Duplicate names are resolved by union order (first match wins for
`cat`/`ctl`; all entries shown for `ls`).

**Tool overlay example**: user-defined `.tcl` tool scripts discovered from
`~/.config/hyprstream/tools/` are mounted at `/tools/`, then bound before
`/bin/` so they can extend (or override) built-in commands:

```
/bind -b /tools /bin
/ls /bin
load
my-custom-tool
cat
ls
...
```

## `/lang/tcl/` mount

The Tcl interpreter state is exposed as a read-only mount at `/lang/tcl/`,
allowing introspection of the running shell:

```
/lang/tcl/
  eval           Write a Tcl script, read result (ctl semantics)
  vars/          One file per variable — cat to read current value
  procs/         One file per user-defined proc — cat to read body
```

Examples:

```
/cat /lang/tcl/vars/temperature
0.7

/ls /lang/tcl/procs
my-helper
format-output

/ctl /lang/tcl/eval {expr {2 + 2}}
4
```

This is a novel extension of the Plan 9 namespace model: language runtimes
are mounts at `/lang/{name}/`.  Tcl is first; other runtimes (sh, python)
could follow the same pattern.

## Tool discovery

Tcl scripts placed in `~/.config/hyprstream/tools/` are automatically
discovered and mounted into `/bin/` at startup.  Each `.tcl` file becomes
a command whose name is the filename without the extension:

```
~/.config/hyprstream/tools/
  summarize.tcl    -> /bin/summarize
  translate.tcl    -> /bin/translate
```

The discovery scan runs once at shell construction.  Scripts are evaluated
in the sandboxed interpreter — they have no direct host I/O, only VFS
access.

## `/bin/` duality

Builtins (`cat`, `ls`, `echo`, `ctl`, `help`, `mount`) are registered directly
in the Tcl interpreter.  They shadow any `/bin/` file of the same name —
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

### Sandbox

The `hyprstream-sandbox` crate provides `SandboxedShell` — a Tcl shell with
an independently constructed namespace and resource limits. The sandbox does
NOT make authorization decisions. The policy engine (Subject + Casbin) builds
the namespace with only the mounts the subject is authorized to access.
Least privilege: empty namespace by default, policy grants access.

### Limits

- **Instruction limit**: configurable via `TclShell::set_instruction_limit()`,
  prevents infinite loops and CPU bombs.
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

The `Mount` trait (`hyprstream_vfs::Mount`) is the universal abstraction for
all VFS backends.  It mirrors 9P2000 operations:

```rust
#[async_trait]
pub trait Mount: Send + Sync {
    async fn walk(&self, components: &[&str], caller: &Subject) -> Result<Fid, MountError>;
    async fn open(&self, fid: &mut Fid, mode: u8, caller: &Subject) -> Result<(), MountError>;
    async fn read(&self, fid: &Fid, offset: u64, count: u32, caller: &Subject) -> Result<Vec<u8>, MountError>;
    async fn write(&self, fid: &Fid, offset: u64, data: &[u8], caller: &Subject) -> Result<u32, MountError>;
    async fn readdir(&self, fid: &Fid, caller: &Subject) -> Result<Vec<DirEntry>, MountError>;
    async fn stat(&self, fid: &Fid, caller: &Subject) -> Result<Stat, MountError>;
    async fn clunk(&self, fid: Fid, caller: &Subject);
}
```

All methods are async. In-process implementations (SyntheticTree) resolve
instantly. Remote implementations (RemoteMount) `.await` RPC calls directly.
Sync consumers (Tcl builtins, FUSE callbacks) use `handle.block_on()`.

The `Namespace` provides convenience methods that compose these primitives:
`cat` (walk+open+read_all+clunk), `echo` (walk+open+write+clunk),
`ctl` (walk+open(RDWR)+write+read+clunk), `ls` (walk+open+readdir+clunk).

### RemoteMount\<C: FsClient\>

`RemoteMount<C>` is a generic async `Mount` impl that bridges any service's
`FsClient` to the VFS. The `FsClient` trait (in `hyprstream-rpc`) defines
the common 9P-shaped async interface (`fs_walk`, `fs_open`, `fs_read`, etc.).

For the model service, `ModelFsAdapter` implements `FsClient` by delegating
to the generated `ModelClient` RPC. The first walk component selects the
model reference scope. `RemoteModelMount` is a type alias for
`RemoteMount<ModelFsAdapter>`.

Any service with an `fs` scope is auto-mountable:
`RemoteMount::new(service_client.fs(scope))`.

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

## FUSE/virtio-fs adapter (`hyprstream-vfsd`)

The `hyprstream-vfsd` crate exposes the VFS namespace as a real Linux
filesystem via FUSE (and optionally virtio-fs for VM guests).  This is
Linux-only.

Architecture:

- **`VfsFuse`** wraps any `Mount` implementation and translates FUSE
  operations (lookup, read, write, readdir, etc.) into `Mount` trait calls.
- An **inode table** maps FUSE inode numbers to VFS fids, handling the
  translation between the kernel's inode-based model and the 9P fid-based
  model.
- **`DaxMount`** provides DAX (direct access) mapping for model weight files,
  allowing the kernel to memory-map `.safetensors` directly without copying
  through the FUSE buffer.  This is critical for large model weights where
  read throughput matters.

Usage:

```bash
hyprstream vfsd --mountpoint /mnt/hyprstream
ls /mnt/hyprstream/srv/model/qwen3:main/
```

When running inside a VM, `hyprstream-vfsd` can serve as a virtio-fs
backend, allowing the guest kernel to mount the hyprstream namespace
natively.

## 9P2000.L wire protocol (`hyprstream-9p`)

The `hyprstream-9p` crate implements the 9P2000.L wire protocol, enabling
federation export of the VFS namespace to remote peers.

- **`NinePServer`** wraps any `Mount` implementation and speaks the 9P2000.L
  protocol over a TCP or Unix socket.  Remote clients (including other
  hyprstream instances or standard 9P clients like `9pfuse`) can attach
  and traverse the namespace.
- A custom **wire codec** handles 9P message framing (4-byte length prefix,
  message type, tag multiplexing).
- **Session management** tracks per-client attach state, fid mappings, and
  authentication via the `Subject` identity system.

This is the mechanism behind `/net/{peer}/` mounts: when a peer connects,
its exported namespace is attached via 9P and mounted at the appropriate
prefix in the local namespace.
