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
//! The client side ([`client`], [`dma`], [`wanix_mount`]) bridges hyprstream's
//! VFS to Wanix via DMA and is unchanged.

pub mod msg;
pub mod client;
pub mod backend;
pub mod memory;
// The translator is the server-side TCP accept loop (tokio::net). `net` pulls
// `mio`, which has no wasm32 backend, so the whole module is native-only. The
// browser/wasm build reaches the backend through the DMA/Wanix client path.
#[cfg(not(target_arch = "wasm32"))]
pub mod translator;

#[cfg(target_arch = "wasm32")]
pub mod dma;
#[cfg(target_arch = "wasm32")]
pub mod wanix_mount;

pub use backend::{Backend, OpenResult, StatResult, WalkResult};
#[cfg(not(target_arch = "wasm32"))]
pub use translator::{FidTable, Translator};
