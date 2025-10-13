//! Example demonstrating the Docker-inspired worktree storage driver pattern
//!
//! This example shows how our worktree system follows Docker's graphdriver
//! pattern for managing storage backends.

use hyprstream::worktree::{StorageDriver, DriverRegistry};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("=== Hyprstream Worktree Storage Drivers (Docker Pattern) ===\n");

    // Create a driver registry (like Docker does at daemon startup)
    let mut registry = DriverRegistry::with_root(PathBuf::from("/tmp/hyprstream"));

    // List all available drivers (like `docker info`)
    println!("Available Storage Drivers:");
    println!("--------------------------");
    for driver_status in registry.list_drivers() {
        println!(
            "  {} - {} [Available: {}]",
            driver_status.info.name,
            driver_status.info.description,
            if driver_status.available { "✓" } else { "✗" }
        );
        println!(
            "    Capabilities: CoW={}, Snapshots={}, Space Savings={}%",
            driver_status.info.capabilities.cow,
            driver_status.info.capabilities.snapshots,
            driver_status.info.capabilities.space_savings
        );
    }
    println!();

    // Example 1: Auto-select best driver (Docker's default behavior)
    println!("Example 1: Auto-selection (like Docker)");
    println!("----------------------------------------");
    let driver = registry.get_driver(StorageDriver::Auto)?;
    let info = driver.info();
    println!("Auto-selected driver: {}", info.name);
    println!("Description: {}", info.description);
    println!("Performance tier: {}/5", info.capabilities.performance_tier);
    println!();

    // Example 2: Explicitly select a driver (like docker --storage-driver)
    println!("Example 2: Explicit driver selection");
    println!("------------------------------------");

    // Try overlay2 first (best performance on Linux)
    match registry.get_driver(StorageDriver::Overlay2) {
        Ok(driver) => {
            println!("✓ Using overlay2 driver (best performance)");
            let info = driver.info();
            for (key, value) in &info.status {
                println!("  {}: {}", key, value);
            }
        }
        Err(e) => {
            println!("✗ overlay2 not available: {}", e);
            println!("  (This is expected on non-Linux systems)");
        }
    }
    println!();

    // Example 3: Configuration file format (Docker-style)
    println!("Example 3: Configuration Format (Docker-style)");
    println!("----------------------------------------------");
    println!("In your config file (TOML/JSON/YAML):");
    println!();
    println!("  [worktree]");
    println!("  driver = \"overlay2\"  # Options: auto, overlay2, btrfs, reflink, hardlink, vfs");
    println!();
    println!("Or via environment variable:");
    println!("  HYPRSTREAM_STORAGE_DRIVER=btrfs");
    println!();

    // Example 4: Parse driver from string (like Docker does)
    println!("Example 4: Parsing driver names");
    println!("--------------------------------");
    let driver_names = vec!["overlay2", "btrfs", "vfs", "auto"];
    for name in driver_names {
        if let Some(driver_type) = StorageDriver::from_name(name) {
            println!("  '{}' -> StorageDriver::{:?}", name, driver_type);
        }
    }
    println!();

    // Example 5: Driver priority for auto-selection
    println!("Example 5: Auto-selection Priority");
    println!("-----------------------------------");
    println!("When using 'auto', drivers are tried in this order:");
    for (i, driver_type) in StorageDriver::auto_priority().iter().enumerate() {
        println!("  {}. {} {}",
            i + 1,
            driver_type.name(),
            if driver_type == &StorageDriver::Vfs {
                "(universal fallback)"
            } else {
                ""
            }
        );
    }
    println!();

    // Example 6: VFS as universal fallback (like Docker)
    println!("Example 6: VFS - Universal Fallback");
    println!("------------------------------------");
    let vfs_driver = registry.get_driver(StorageDriver::Vfs)?;
    let vfs_info = vfs_driver.info();
    println!("Driver: {}", vfs_info.name);
    println!("Description: {}", vfs_info.description);
    println!("Always Available: {}", vfs_driver.is_available());
    println!("Space Savings: {}%", vfs_info.capabilities.space_savings);
    println!("\nNote: VFS is always available but provides no optimization.");
    println!("      It's the fallback when no CoW mechanism works.");
    println!();

    // Example 7: Creating a worktree with a driver
    println!("Example 7: Creating a Worktree");
    println!("-------------------------------");

    // Get the best available driver
    let driver = registry.get_default_driver()?;
    println!("Using driver: {}", driver.info().name);

    // Create a test source directory
    let source_dir = PathBuf::from("/tmp/test_source");
    std::fs::create_dir_all(&source_dir)?;
    std::fs::write(source_dir.join("README.md"), b"# Test Repository")?;

    // Create a worktree
    let worktree_id = "test-worktree-001";
    match driver.create(worktree_id, &source_dir, &Default::default()) {
        Ok(worktree_path) => {
            println!("✓ Created worktree at: {}", worktree_path.display());

            // Get driver statistics
            let stats = driver.stats();
            println!("  Active worktrees: {}", stats.active_worktrees);
            if stats.space_saved > 0 {
                println!("  Space saved: {} bytes", stats.space_saved);
            }

            // Clean up
            driver.remove(worktree_id)?;
            println!("✓ Cleaned up worktree");
        }
        Err(e) => {
            println!("✗ Failed to create worktree: {}", e);
        }
    }

    // Clean up test directory
    let _ = std::fs::remove_dir_all(source_dir);
    println!();

    // Summary
    println!("=== Summary ===");
    println!("This design follows Docker's proven graphdriver pattern:");
    println!("1. Drivers are named after technologies (overlay2, btrfs, vfs)");
    println!("2. Simple string-based configuration");
    println!("3. Auto-selection with fallback to vfs");
    println!("4. Clear capability reporting");
    println!("5. Familiar to anyone who knows Docker");

    Ok(())
}