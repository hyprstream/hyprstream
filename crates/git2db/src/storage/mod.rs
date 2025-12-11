//! Storage drivers following Docker's graphdriver pattern
//!
//! This module implements a driver-based abstraction for worktree storage optimization,
//! following the proven pattern from Docker's container filesystem layer management.
//! Just as Docker uses "graphdrivers" to abstract storage backends (overlay2, btrfs, zfs),
//! we use "storage drivers" to abstract worktree optimization mechanisms.
//!
//! # Design Philosophy (from Docker)
//!
//! 1. **"Driver"** is the term - not "mechanism", "strategy", or "backend"
//! 2. Drivers are named after the **actual technology** they use
//! 3. Simple string-based selection in configuration
//! 4. `vfs` is the universal fallback (plain directories)
//!
//! # Architecture
//!
//! Git worktrees are ALWAYS used as the base mechanism. Storage drivers are optional
//! layers that sit UNDERNEATH git worktrees to provide benefits like:
//!
//! - **Copy-on-Write (CoW)**: ~80% disk space savings
//! - **Isolation**: Changes are isolated to an upper layer
//! - **Performance**: Faster creation and cleanup
//!
//! # Example
//!
//! ```toml
//! # In configuration file (Docker-style)
//! [worktree]
//! driver = "overlay2"  # or "btrfs", "reflink", "vfs", etc.
//! ```
//!
//! ```rust,no_run
//! use git2db::storage::{StorageDriver, DriverRegistry};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let registry = DriverRegistry::new();
//!
//! // Select a driver explicitly
//! let driver = registry.get_driver(StorageDriver::Overlay2)?;
//!
//! // Or use vfs as a safe fallback
//! let driver = registry.get_driver(StorageDriver::Vfs)?;
//! # Ok(())
//! # }
//! ```

mod driver;
mod overlay2;
mod reflink;
mod registry;
mod vfs;

pub use driver::{Driver, DriverError, DriverOpts, StorageDriver, WorktreeHandle, DriverFactory};
pub use registry::DriverRegistry;
