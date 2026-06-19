//! Plan9-inspired VFS namespace multiplexer.
//!
//! The VFS is a client-side mount table that routes 9P operations by path prefix.
//! Services are auto-mounted under `/srv/{name}` from discovery. Commands live
//! at `/bin/` (Plan9 convention). Local mounts (`/config/`, `/private/`, `/bin/`,
//! `/env/`) are reserved for Mount only.
//!
//! ```text
//! /bin/cat                        → filesystem command (CtlFile)
//! /env/temperature                → session variable (DynamicDir)
//! /srv/model/qwen3:main/status    → ModelService (via ZMQ or IPC)
//! /srv/mcp/summarize/schema       → McpService
//! /config/temperature             → in-process (Mount)
//! /net/peer-a/srv/model/...       → federated peer
//! ```
//!
//! All types are WASM-compatible. Transport is abstracted via traits.

mod fsmount;
mod mount;
mod namespace;
#[cfg(not(target_arch = "wasm32"))]
pub mod proxy;

// Native-only: the `FileSystem → FsMount` up-adapter and the OverlayFs-backed
// v1 overlay engine. Both wrap `fuse_backend_rs`, whose overlay/passthrough
// backends are Linux-only.
#[cfg(not(target_arch = "wasm32"))]
mod fuse_adapter;
#[cfg(not(target_arch = "wasm32"))]
pub mod overlay;
// Shim that makes a handleless `Layer` (e.g. RAFS) usable as an OverlayFs lower.
#[cfg(not(target_arch = "wasm32"))]
mod zero_open;

pub use fsmount::{FsMount, SetAttr};
pub use hyprstream_rpc::Subject;
pub use mount::{DirEntry, Fid, Mount, MountError, Stat, OREAD, OWRITE, ORDWR, OTRUNC, ORCLOSE};
pub use namespace::{BindFlag, MountTarget, Namespace, NamespaceError};

#[cfg(not(target_arch = "wasm32"))]
pub use fuse_adapter::FuseFileSystemMount;

// Overlay composition surface (FS-C v1 engine). Re-exported so downstream crates
// (FS-B) can build a RAFS lower as a `BoxedLayer` via [`overlay::layer_from_fs`]
// and compose it with [`overlay::rootfs_overlay`] without depending on
// `fuse-backend-rs`'s overlay internals directly.
#[cfg(not(target_arch = "wasm32"))]
pub use fuse_backend_rs::overlayfs::BoxedLayer;
#[cfg(not(target_arch = "wasm32"))]
pub use zero_open::ZeroOpenLayer;
