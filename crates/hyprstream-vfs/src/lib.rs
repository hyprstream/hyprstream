//! Plan9-inspired VFS namespace multiplexer.
//!
//! The VFS is a client-side mount table that routes 9P operations by path prefix.
//! Services are auto-mounted under `/srv/{name}` from discovery. Quick commands
//! live at `/cmd/`. Local mounts (`/config/`, `/private/`) are reserved.
//!
//! ```text
//! /srv/model/qwen3:main/status    → ModelService (via ZMQ or IPC)
//! /srv/mcp/summarize/schema       → McpService
//! /cmd/load                       → quick command (LocalMount)
//! /config/temperature             → in-process (LocalMount)
//! /net/peer-a/srv/model/...       → federated peer
//! ```
//!
//! All types are WASM-compatible. Transport is abstracted via traits.

mod local_mount;
mod namespace;

pub use local_mount::{DirEntry, LocalFid, LocalMount, Stat};
pub use namespace::{BindFlag, MountTarget, Namespace, VfsError};
