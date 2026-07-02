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

pub mod devfile;
mod fsmount;
mod mount;
mod namespace;
#[cfg(not(target_arch = "wasm32"))]
pub mod proxy;

// Native injected mounts (FS-D, #365): the `Send + Sync` ports of the wasm-only
// `/stream` / synthetic injected mounts, suitable for serving to a CH guest.
#[cfg(not(target_arch = "wasm32"))]
pub mod injected;

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

pub use devfile::{ControlFile, DevFileState, DevFuture, DynamicDir, FieldFile, NoSetter};
pub use fsmount::{FsMount, SetAttr};
pub use hyprstream_rpc::Subject;
pub use mount::{DirEntry, Fid, Mount, MountError, Stat, OREAD, OWRITE, ORDWR, OTRUNC, ORCLOSE, DMDIR};
pub use namespace::{BindFlag, MountTarget, Namespace, NamespaceError};

#[cfg(not(target_arch = "wasm32"))]
pub use fuse_adapter::FuseFileSystemMount;

#[cfg(not(target_arch = "wasm32"))]
pub use injected::{StreamMount, StreamRegistry, SyntheticMount, SyntheticNode};

// Overlay composition surface (FS-C v1 engine). Re-exported so downstream crates
// (FS-B) can build a RAFS lower as a `BoxedLayer` via [`overlay::layer_from_fs`]
// and compose it with [`overlay::rootfs_overlay`] without depending on
// `fuse-backend-rs`'s overlay internals directly.
#[cfg(not(target_arch = "wasm32"))]
pub use fuse_backend_rs::overlayfs::BoxedLayer;
#[cfg(not(target_arch = "wasm32"))]
pub use zero_open::ZeroOpenLayer;

// ─────────────────────────────────────────────────────────────────────────────
// Standard namespace paths
// ─────────────────────────────────────────────────────────────────────────────

/// Canonical mount paths every hyprstream namespace exposes, regardless of
/// transport (native TUI vs WASM browser).
///
/// This is the convergence contract from #389 + #391 (Option 1: shared content
/// model). Both the native namespace builder
/// (`hyprstream::cli::shell_handlers`) and the browser namespace builder
/// (`hyprstream_rpc_std::vfs_mount::build_browser_namespace`) mount these
/// content trees backed at the hyprstream spine — so `/srv/registry/foo/main`
/// resolves to the same repo/worktree content in either context.
///
/// The transport leaf (DMA/SAB ring buffers for the browser, ZMQ for native)
/// is correctly-scoped glue and is NOT part of this contract — only the
/// backing content trees are converged.
///
/// # Paths
///
/// | Path             | Backing                                            |
/// |------------------|----------------------------------------------------|
/// | `/srv/model`     | Model service: synthetic tree (status, load, ...)  |
/// | `/srv/registry`  | Registry service: worktree FS (real qids)          |
/// | `/worktree`      | Alias of `/srv/registry` (short ergonomic path)    |
///
/// Per-target paths (NOT part of the convergence contract):
///   - `/bin`, `/env`, `/lang/tcl` — native-only local synthetic mounts
///   - `/stream`, `/srv/{service}/doc` — browser-only GenericServiceMount extras
pub const STANDARD_NAMESPACE_PATHS: &[&str] = &["/srv/model", "/srv/registry", "/worktree"];
