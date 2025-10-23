//! Integration tests for Docker-inspired worktree storage drivers

use hyprstream_core::worktree::{StorageDriver, DriverRegistry, DriverOpts};
use tempfile::TempDir;

#[test]
fn test_docker_naming_pattern() {
    // Test that driver names follow Docker's convention
    assert_eq!(StorageDriver::Overlay2.name(), "overlay2");
    assert_eq!(StorageDriver::Btrfs.name(), "btrfs");
    assert_eq!(StorageDriver::Vfs.name(), "vfs");
    assert_eq!(StorageDriver::Auto.name(), "auto");
}

#[test]
fn test_parse_driver_from_string() {
    // Test Docker-style string parsing
    assert_eq!(StorageDriver::from_name("overlay2"), Some(StorageDriver::Overlay2));
    assert_eq!(StorageDriver::from_name("OVERLAY2"), Some(StorageDriver::Overlay2));
    assert_eq!(StorageDriver::from_name("overlay"), Some(StorageDriver::Overlay2));
    assert_eq!(StorageDriver::from_name("btrfs"), Some(StorageDriver::Btrfs));
    assert_eq!(StorageDriver::from_name("vfs"), Some(StorageDriver::Vfs));
    assert_eq!(StorageDriver::from_name("auto"), Some(StorageDriver::Auto));
    assert_eq!(StorageDriver::from_name("invalid"), None);
}

#[test]
fn test_auto_priority_order() {
    // Test that priority order follows Docker's approach
    let priority = StorageDriver::auto_priority();

    // Performance drivers should come before fallbacks
    let overlay_pos = priority.iter().position(|d| d == &StorageDriver::Overlay2);
    let hardlink_pos = priority.iter().position(|d| d == &StorageDriver::Hardlink);
    let vfs_pos = priority.iter().position(|d| d == &StorageDriver::Vfs);

    assert!(overlay_pos < hardlink_pos);
    assert!(hardlink_pos < vfs_pos);

    // VFS should always be last (universal fallback)
    assert_eq!(priority.last(), Some(&StorageDriver::Vfs));
}

#[test]
fn test_vfs_always_available() {
    // Test that VFS driver is always available (Docker pattern)
    let temp_dir = TempDir::new().unwrap();
    let registry = DriverRegistry::with_root(temp_dir.path().to_path_buf());

    // VFS should always work
    let vfs_driver = registry.get_driver(StorageDriver::Vfs).unwrap();
    assert!(vfs_driver.is_available());

    let info = vfs_driver.info();
    assert_eq!(info.name, "vfs");
    assert!(!info.capabilities.cow);  // VFS has no CoW
    assert_eq!(info.capabilities.space_savings, 0);  // No space savings
}

#[test]
fn test_driver_registry() {
    // Test Docker-style driver registry
    let temp_dir = TempDir::new().unwrap();
    let registry = DriverRegistry::with_root(temp_dir.path().to_path_buf());

    // List all drivers
    let drivers = registry.list_drivers();
    assert!(!drivers.is_empty());

    // Check that standard drivers are registered
    let driver_names: Vec<String> = drivers.iter().map(|d| d.name.clone()).collect();
    assert!(driver_names.contains(&"overlay2".to_string()));
    assert!(driver_names.contains(&"btrfs".to_string()));
    assert!(driver_names.contains(&"reflink".to_string()));
    assert!(driver_names.contains(&"hardlink".to_string()));
    assert!(driver_names.contains(&"vfs".to_string()));
}

#[test]
fn test_auto_selection() {
    // Test auto-selection (should always succeed with VFS as fallback)
    let temp_dir = TempDir::new().unwrap();
    let registry = DriverRegistry::with_root(temp_dir.path().to_path_buf());

    // Auto should always succeed
    let driver = registry.get_driver(StorageDriver::Auto).unwrap();
    assert!(driver.is_available());

    // Should have selected something
    let info = driver.info();
    assert!(!info.name.is_empty());
}

#[test]
fn test_driver_capabilities() {
    // Test that drivers report capabilities correctly
    let temp_dir = TempDir::new().unwrap();
    let registry = DriverRegistry::with_root(temp_dir.path().to_path_buf());

    // Check VFS capabilities (lowest tier)
    let vfs = registry.get_driver(StorageDriver::Vfs).unwrap();
    let vfs_caps = vfs.info().capabilities;
    assert!(!vfs_caps.cow);
    assert!(!vfs_caps.snapshots);
    assert_eq!(vfs_caps.space_savings, 0);
    assert_eq!(vfs_caps.performance_tier, 1);

    // If overlay2 is available, check its capabilities
    if let Ok(overlay2) = registry.get_driver(StorageDriver::Overlay2) {
        if overlay2.is_available() {
            let overlay2_caps = overlay2.info().capabilities;
            assert!(overlay2_caps.cow);
            assert!(overlay2_caps.snapshots);
            assert!(overlay2_caps.space_savings > 0);
            assert!(overlay2_caps.performance_tier > vfs_caps.performance_tier);
        }
    }
}

#[test]
fn test_create_worktree_with_vfs() {
    // Test creating an actual worktree with VFS driver
    let temp_dir = TempDir::new().unwrap();
    let registry = DriverRegistry::with_root(temp_dir.path().join("storage"));

    // Create a source directory
    let source_dir = temp_dir.path().join("source");
    std::fs::create_dir_all(&source_dir).unwrap();
    std::fs::write(source_dir.join("test.txt"), b"Hello, Docker pattern!").unwrap();

    // Get VFS driver (always available)
    let driver = registry.get_driver(StorageDriver::Vfs).unwrap();

    // Create a worktree
    let worktree_id = "test-worktree-001";
    let worktree_path = driver.create(worktree_id, &source_dir, &DriverOpts::default()).unwrap();

    // Verify worktree was created
    assert!(worktree_path.exists());
    assert!(worktree_path.join("test.txt").exists());

    // Verify we can get the worktree
    let retrieved_path = driver.get(worktree_id).unwrap();
    assert_eq!(retrieved_path, worktree_path);

    // Check if worktree exists
    assert!(driver.exists(worktree_id));

    // Get stats
    let stats = driver.stats();
    assert_eq!(stats.active_worktrees, 1);
    assert_eq!(stats.space_saved, 0);  // VFS has no space savings

    // Clean up
    driver.remove(worktree_id).unwrap();
    assert!(!driver.exists(worktree_id));
}

#[test]
fn test_driver_status_report() {
    // Test Docker-style status reporting
    let temp_dir = TempDir::new().unwrap();
    let registry = DriverRegistry::with_root(temp_dir.path().to_path_buf());

    // Get status report (like `docker info`)
    let status = registry.driver_status().unwrap();

    // Should have a selected driver
    assert!(!status.storage_driver.is_empty());

    // Should have root directory
    assert_eq!(status.driver_root, temp_dir.path());

    // Should list available drivers
    assert!(!status.available_drivers.is_empty());
}

#[test]
fn test_explicit_driver_selection() {
    // Test explicit driver selection (like docker --storage-driver)
    let temp_dir = TempDir::new().unwrap();
    let mut registry = DriverRegistry::with_root(temp_dir.path().to_path_buf());

    // Set default to VFS
    registry.set_default_driver(StorageDriver::Vfs);

    // Get default driver
    let driver = registry.get_default_driver().unwrap();
    assert_eq!(driver.info().name, "vfs");
}

#[test]
fn test_fallback_behavior() {
    // Test that unavailable drivers fall back properly
    let temp_dir = TempDir::new().unwrap();
    let registry = DriverRegistry::with_root(temp_dir.path().to_path_buf());

    // Try to get a driver that might not be available
    // The registry should handle this gracefully
    match registry.get_driver(StorageDriver::Overlay2) {
        Ok(driver) => {
            // If we got it, it should be available
            if !driver.is_available() {
                // This is actually an error in our implementation
                panic!("Got unavailable driver from registry");
            }
        }
        Err(e) => {
            // This is expected on non-Linux systems
            assert!(e.to_string().contains("not available") ||
                   e.to_string().contains("not supported"));
        }
    }

    // Auto should always work (falls back to VFS)
    let auto_driver = registry.get_driver(StorageDriver::Auto).unwrap();
    assert!(auto_driver.is_available());
}