# hyprstream-9p

Minimal 9P2000.L server for bridging the hyprstream VFS to Wanix via DMA ring buffers.

## What it does

Implements the subset of the 9P2000.L wire protocol needed for read/write filesystem access:

| Message pair | Purpose |
|---|---|
| `Tversion` / `Rversion` | Protocol negotiation |
| `Tattach` / `Rattach` | Attach to root fid |
| `Twalk` / `Rwalk` | Path traversal (resolve fid) |
| `Tlopen` / `Rlopen` | Open file (9P2000.L variant) |
| `Tread` / `Rread` | Read data |
| `Twrite` / `Rwrite` | Write data |
| `Treaddir` / `Rreaddir` | List directory |
| `Tgetattr` / `Rgetattr` | Stat file |
| `Tclunk` / `Rclunk` | Close fid |
| `Rlerror` | Error response |

**Wire format**: all messages are 4-byte LE length-prefixed, matching the DMA ring buffer framing used by the Wanix WASM runtime.

**WASM extras** (`target_arch = "wasm32"`): `dma` module (SAB ring-buffer transport) and `wanix_mount` (mount the VFS into the Wanix namespace via DMA).

## Architecture position

```
hyprstream-vfs           (Mount trait, Namespace, Fid)
    ↑
hyprstream-9p            ← you are here
    ↑
hyprstream               (9P server factory)
www-cyberdione-ai        (Wanix WASM: mounts via DMA ring buffer)
```
