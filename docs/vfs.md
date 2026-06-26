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

The `Mount` trait (`hyprstream_vfs::Mount`) is the universal abstraction for
all VFS backends.  It mirrors 9P2000 operations:

```rust
pub trait Mount: Send + Sync {
    fn walk(&self, components: &[&str], caller: &Subject) -> Result<Fid, MountError>;
    fn open(&self, fid: &mut Fid, mode: u8, caller: &Subject) -> Result<(), MountError>;
    fn read(&self, fid: &Fid, offset: u64, count: u32, caller: &Subject) -> Result<Vec<u8>, MountError>;
    fn write(&self, fid: &Fid, offset: u64, data: &[u8], caller: &Subject) -> Result<u32, MountError>;
    fn readdir(&self, fid: &Fid, caller: &Subject) -> Result<Vec<DirEntry>, MountError>;
    fn stat(&self, fid: &Fid, caller: &Subject) -> Result<Stat, MountError>;
    fn clunk(&self, fid: Fid, caller: &Subject);
}
```

The `Namespace` provides convenience methods that compose these primitives:
`cat` (walk+open+read_all+clunk), `echo` (walk+open+write+clunk),
`ctl` (walk+open(RDWR)+write+read+clunk), `ls` (walk+open+readdir+clunk).

### RemoteModelMount

`RemoteModelMount` bridges the `Mount` trait to the `ModelClient` RPC.
It translates synchronous `Mount` calls into async ZMQ requests via a
dedicated single-threaded tokio runtime.  Local fid numbers are tracked
in a `DashMap` and mapped to remote fids returned by the model service's
`FsHandler`.  This allows `/srv/model` to proxy 9P operations to the
server-side `SyntheticTree` transparently.

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
