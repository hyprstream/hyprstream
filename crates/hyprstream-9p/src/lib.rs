//! Minimal 9P2000.L server for bridging hyprstream VFS to Wanix via DMA.
//!
//! Implements only the message types needed for filesystem access:
//! - Tversion/Rversion — protocol negotiation
//! - Tattach/Rattach — attach to root
//! - Twalk/Rwalk — path traversal
//! - Tlopen/Rlopen — open file (9P2000.L variant)
//! - Tread/Rread — read data
//! - Twrite/Rwrite — write data
//! - Treaddir/Rreaddir — list directory
//! - Tgetattr/Rgetattr — stat file
//! - Tclunk/Rclunk — close fid
//! - Rlerror — error response
//!
//! Wire format: all messages are length-prefixed (4-byte LE), matching
//! the DMA ring buffer's message framing.

pub mod msg;
pub mod client;
#[cfg(target_arch = "wasm32")]
pub mod dma;
#[cfg(target_arch = "wasm32")]
pub mod wanix_mount;
