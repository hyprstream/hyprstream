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

## Server-side translator (Kata 9P)

`translator::Translator` is a server-side 9P2000.L translator: it accepts 9P
connections (TCP transport now, for standalone testing; virtio-9P later) and
translates each T-message into a [`backend::Backend`] call. The codec is
bidirectional — `msg.rs` provides T-message parsers + R-message encoders
(server side) alongside the original T-message encoders + R-message parsers
(client side).

The backend is the capnp-RPC seam. The `hyprstream` binary crate ships
`services::kata_9p_backend::ModelBackend`, which wraps the generated
`ModelClient` and turns each `Backend` call into a `nine.capnp` envelope
against the model service's `fs` scope — the inverse of `RemoteModelMount`
(VFS → RPC):

```
Kata guest (virtio-9P / TCP) ─► Translator ─► ModelBackend ─► ModelClient.fs()
   9P2000.L wire                 fid table       capnp RPC (nine.capnp)
```

`memory::MemoryBackend` is an in-process backend used by the integration tests
in `tests/tcp_translator.rs` so the translator can be exercised with no
network and no model service running.

## Architecture position

```
hyprstream-vfs           (Mount trait, Namespace, Fid)
    ↑
hyprstream-9p            ← you are here  (codec + translator + Backend trait)
    ↑
hyprstream               (ModelBackend: 9P → model-service capnp RPC)
www-cyberdione-ai        (Wanix WASM: mounts via DMA ring buffer)
```
