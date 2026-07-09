//! 9P2000.L codec and server-side translator for hyprstream.
//!
//! Two halves:
//!
//! - **Codec** ([`msg`]): the full 9P2000.L wire format. `read_from` / `write_to`
//!   style helpers for *both* client and server sides — T-message parsers +
//!   R-message encoders live alongside the original client-side encoders +
//!   R-message parsers.
//!
//! - **Translator** ([`translator`]): a server that accepts 9P connections
//!   (TCP transport now; virtio-9P later), maintains a server-side fid table,
//!   and translates each T-message into a [`backend::Backend`] call. The
//!   backend is the capnp-RPC seam: the `hyprstream` binary crate ships a
//!   `ModelBackend` that turns each call into a `nine.capnp` envelope against
//!   the model service's `fs` scope — the inverse of `RemoteModelMount`
//!   (VFS → RPC).
//!
//! ## Client side
//!
//! [`client`] is the transport-agnostic 9P2000.L client ([`client::P9Client`]),
//! driven by a [`client::P9Transport`]. Two transports back it:
//!
//! - **wasm** ([`dma`], [`wanix_mount`]) — SharedArrayBuffer DMA ring to a
//!   browser Wanix server; single-threaded, so [`wanix_mount::WanixMount`] uses
//!   `unsafe impl Send/Sync`.
//! - **native** ([`socket_transport`], [`remote_mount`]) — a real Unix/TCP
//!   socket ([`socket_transport::SocketTransport`]) exposed as a genuinely
//!   `Send + Sync` VFS [`Mount`](hyprstream_vfs::Mount)
//!   ([`remote_mount::Remote9pMount`]). This is the client half of #708: dial a
//!   remote 9P server and import its tree into the host namespace. It is the
//!   exact inverse of [`mount_backend::MountBackend`] (Mount → 9P server) and
//!   shares the same [`client::P9Transport`] / [`msg`] wire core, so a future
//!   9P-over-WebSocket server reuses the same seam.
//!
//! The wasm `WanixMount` and native `Remote9pMount` are deliberately *separate*
//! types rather than one renamed generic: they have incompatible `Send`/`Sync`
//! stories (wasm's `unsafe impl` is only sound single-threaded, native is
//! genuinely thread-safe) and are gated to disjoint targets.

pub mod msg;
pub mod client;
pub mod backend;
pub mod memory;
// The translator is the server-side TCP/UDS accept loop (tokio::net). `net`
// pulls `mio`, which has no wasm32 backend, so these modules are native-only.
// The browser/wasm build reaches the backend through the DMA/Wanix client path.
#[cfg(not(target_arch = "wasm32"))]
pub mod translator;
// Attach-time MAC seam (#568): trait interface for verifying an attach
// credential and authorizing per-op access, with inert (no-op) defaults.
// Native-only alongside `translator`, its only consumer today.
#[cfg(not(target_arch = "wasm32"))]
pub mod mac_seam;
// `MountBackend` bridges a VFS `Mount` (+ `Subject`) to the `Backend` seam so
// the translator can export it over TCP/UDS. Native-only (drives async Mount ops).
#[cfg(not(target_arch = "wasm32"))]
pub mod mount_backend;
// The socket 9P *client* Mount (import a remote 9P namespace over UDS/TCP). Uses
// `tokio::net`, so native-only — the wasm build imports via the DMA/Wanix path.
#[cfg(not(target_arch = "wasm32"))]
pub mod socket_transport;
#[cfg(not(target_arch = "wasm32"))]
pub mod remote_mount;

#[cfg(target_arch = "wasm32")]
pub mod dma;
#[cfg(target_arch = "wasm32")]
pub mod wanix_mount;

pub use backend::{Backend, OpenResult, StatResult, WalkResult};
pub use client::{P9Client, P9Transport};
#[cfg(not(target_arch = "wasm32"))]
pub use mount_backend::{AttachAuthorizer, MountBackend};
#[cfg(not(target_arch = "wasm32"))]
pub use remote_mount::Remote9pMount;
#[cfg(not(target_arch = "wasm32"))]
pub use socket_transport::SocketTransport;
#[cfg(not(target_arch = "wasm32"))]
pub use translator::{
    serve_mount_uds, serve_mount_vsock, serve_mount_vsock_raw, FidTable, Translator,
};
#[cfg(not(target_arch = "wasm32"))]
pub use mac_seam::{
    anonymous_floor, AccessDecider, Action, AllowAllDecider, AnonymousAuthenticator,
    AttachAuthenticator, AuditSink, AuditedDecider, NullAuditSink,
};
