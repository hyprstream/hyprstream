# hyprstream-vfs

Plan 9-inspired VFS namespace multiplexer for hyprstream.

## What it does

A client-side mount table that routes 9P operations by path prefix. Services are auto-mounted under `/srv/{name}` from discovery. The namespace is composable — any `Mount` implementation can be bound at any path.

```
/bin/cat                        → filesystem command (CtlFile)
/env/temperature                → session variable (DynamicDir)
/srv/model/qwen3:main/status    → ModelService (via RPC or IPC)
/srv/mcp/summarize/schema       → McpService
/config/temperature             → in-process (Mount)
/net/peer-a/srv/model/...       → federated peer
```

All types are WASM-compatible. Transport is abstracted via the `Mount` trait — the same namespace API works over in-process calls, RPC, and the 9P wire protocol.

## Key types

| Type | Description |
|------|-------------|
| `Namespace` | Mount table with prefix routing; `bind()`, `walk()`, `open()`, `read()`, `write()` |
| `Mount` | Trait implemented by any VFS backend (filesystem, RPC proxy, synthetic dir) |
| `Fid` | Open file handle (maps to 9P `fid`) |
| `Stat` | File metadata |
| `BindFlag` | `Replace`, `Before`, `After` — controls union mount ordering |
| `MountTarget` | Destination for a bind (path or special targets) |

## Architecture position

```
hyprstream-vfs           ← you are here
    ↑ (depends on)
hyprstream-rpc           (Subject for caller identity)
    ↑ (used by)
hyprstream-9p            (9P wire protocol server)
hyprstream-workers       (container I/O paths)
hyprstream               (VFS service factory, /srv mounts)
```
