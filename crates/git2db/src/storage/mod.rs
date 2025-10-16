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
//! 5. Auto-selection with priority ordering
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
//! // Auto-select best available driver
//! let registry = DriverRegistry::new();
//! let driver = registry.get_driver(StorageDriver::Auto)?;
//!
//! // Or explicitly select a driver
//! let driver = registry.get_driver(StorageDriver::Overlay2)?;
//! # Ok(())
//! # }
//! ```

mod driver;
mod overlay2;
mod registry;
mod vfs;

pub use driver::{Driver, DriverCapabilities, DriverError, DriverOpts, StorageDriver};
pub use registry::DriverRegistry;

// Re-export individual driver configurations
pub use overlay2::Overlay2Config;
pub use vfs::VfsConfig;
