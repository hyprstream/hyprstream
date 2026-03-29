//! 9P2000.L wire protocol bridge for hyprstream VFS federation.
//!
//! This crate provides the foundation for federating hyprstream VFS namespaces
//! over the 9P2000.L wire protocol. It bridges between the in-process `Mount`
//! trait (from `hyprstream-vfs`) and the 9P2000.L binary protocol, enabling:
//!
//! - **Server**: Any `Mount` impl can be exported over TCP/Unix sockets as a
//!   9P2000.L file server. Remote peers see it as a standard 9P filesystem.
//!
//! - **Client**: A remote 9P2000.L server can be mounted into the local VFS
//!   namespace as a `Mount` impl. The client translates `walk/open/read/write`
//!   calls into 9P2000.L wire messages.
//!
//! ```text
//! Local VFS                          Remote VFS
//! ─────────                          ──────────
//! /srv/model/qwen3:main/status       /srv/model/qwen3:main/status
//!         │                                    │
//!    NinePServer                          NinePClient
//!    (Mount → 9P wire)               (9P wire → Mount)
//!         │                                    │
//!         └──────── TCP / Unix socket ─────────┘
//!              9P2000.L binary protocol
//! ```
//!
//! ## Wire format
//!
//! We implement the 9P2000.L wire format directly. The `protocol` module defines
//! all T-message (client→server) and R-message (server→client) types with
//! `encode`/`decode` methods for the little-endian binary wire format.
//!
//! The Google `p9` crate (from ChromeOS) has battle-tested types but keeps its
//! protocol module private. We define our own subset covering the operations
//! needed for VFS federation: version, attach, walk, lopen, read, write, clunk,
//! readdir, getattr, and statfs.
//!
//! ## Federation model
//!
//! Federated peers appear in the namespace at `/net/{peer}/`:
//!
//! ```text
//! /net/peer-a/srv/model/qwen3:main/status
//! /net/peer-b/srv/mcp/summarize/schema
//! ```
//!
//! Each peer connection is a `NinePClient` mounted via `Namespace::mount()`.

pub mod protocol;
mod server;
mod session;

pub use protocol::{Qid, Tframe, Rframe};
pub use server::NinePServer;
pub use session::Session;
