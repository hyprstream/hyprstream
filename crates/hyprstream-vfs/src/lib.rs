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

mod mount;
mod namespace;
#[cfg(not(target_arch = "wasm32"))]
pub mod proxy;

pub use hyprstream_rpc::Subject;
pub use mount::{DirEntry, Fid, Mount, MountError, Stat, OREAD, OWRITE, ORDWR, OTRUNC, ORCLOSE, DMDIR};
pub use namespace::{BindFlag, MountTarget, Namespace, NamespaceError};

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
