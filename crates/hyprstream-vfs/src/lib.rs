//! Plan9-inspired VFS namespace multiplexer.
//!
//! The VFS is a client-side mount table that routes 9P operations by path prefix.
//! Services are auto-mounted under `/srv/{name}` from discovery. Commands live
//! at `/bin/` (Plan9 convention). Local mounts (`/config/`, `/private/`, `/bin/`,
//! `/env/`) are reserved for LocalMount only.
//!
//! ```text
//! /bin/cat                        → filesystem command (CtlFile)
//! /env/temperature                → session variable (DynamicDir)
//! /srv/model/qwen3:main/status    → ModelService (via ZMQ or IPC)
//! /srv/mcp/summarize/schema       → McpService
//! /config/temperature             → in-process (LocalMount)
//! /net/peer-a/srv/model/...       → federated peer
//! ```
//!
//! All types are WASM-compatible. Transport is abstracted via traits.

mod local_mount;
mod namespace;

pub use hyprstream_rpc::Subject;
pub use local_mount::{DirEntry, LocalFid, LocalMount, Stat};
pub use namespace::{BindFlag, MountTarget, Namespace, VfsError};
