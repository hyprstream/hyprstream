//! Driver registry (Docker's graphdriver pattern)
//!
//! Manages driver registration and selection, similar to Docker's
//! daemon/graphdriver/driver_linux.go

use super::driver::{Driver, DriverError, StorageDriver};
use super::overlay2::Overlay2Driver;
use super::vfs::VfsDriver;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::info;

/// Registry of available storage drivers
///
/// Follows Docker's pattern for driver discovery and selection.
pub struct DriverRegistry {
    drivers: HashMap<String, Arc<dyn Driver>>,
}

impl DriverRegistry {
    /// Create a new registry with all available drivers
    pub fn new() -> Self {
        let mut drivers: HashMap<String, Arc<dyn Driver>> = HashMap::new();

        // Register vfs (always available)
        drivers.insert("vfs".to_string(), Arc::new(VfsDriver));

        // Register overlay2 on Linux
        #[cfg(all(target_os = "linux", feature = "overlayfs"))]
        {
            drivers.insert("overlay2".to_string(), Arc::new(Overlay2Driver::new()));
        }

        // Future: Register other drivers
        // #[cfg(feature = "btrfs")]
        // drivers.insert("btrfs".to_string(), Arc::new(BtrfsDriver));

        Self { drivers }
    }

    /// Get a driver by selection
    ///
    /// Resolves Auto to the best available driver.
    pub fn get_driver(&self, selection: StorageDriver) -> Result<Arc<dyn Driver>, DriverError> {
        match selection {
            StorageDriver::Auto => self.auto_select(),
            StorageDriver::Overlay2 => self.get("overlay2"),
            StorageDriver::Btrfs => self.get("btrfs"),
            StorageDriver::Reflink => self.get("reflink"),
            StorageDriver::Hardlink => self.get("hardlink"),
            StorageDriver::Vfs => self.get("vfs"),
        }
    }

    /// Get a driver by name
    fn get(&self, name: &str) -> Result<Arc<dyn Driver>, DriverError> {
        self.drivers
            .get(name)
            .cloned()
            .ok_or_else(|| DriverError::NotAvailable(format!("Driver '{}' not registered", name)))
    }

    /// Auto-select the best available driver (Docker pattern)
    ///
    /// Priority order:
    /// 1. overlay2 (Linux overlayfs) - best space savings
    /// 2. Future: btrfs, reflink, hardlink
    /// 3. vfs (plain directories) - always works
    fn auto_select(&self) -> Result<Arc<dyn Driver>, DriverError> {
        // Priority order (like Docker's GetDriver)
        let priorities = [
            "overlay2",  // Best: ~80% space savings on Linux
            "btrfs",     // Future: Native btrfs CoW
            "reflink",   // Future: XFS/APFS reflinks
            "hardlink",  // Future: Cross-platform hardlinks
            "vfs",       // Fallback: Always works
        ];

        for name in &priorities {
            if let Some(driver) = self.drivers.get(*name) {
                if driver.is_available() {
                    info!("Auto-selected storage driver: {}", driver.name());
                    return Ok(driver.clone());
                }
            }
        }

        // VFS should always be available as ultimate fallback
        self.get("vfs")
    }

    /// List all registered drivers
    pub fn list_drivers(&self) -> Vec<String> {
        self.drivers.keys().cloned().collect()
    }

    /// List available drivers (registered AND available on this system)
    pub fn list_available_drivers(&self) -> Vec<String> {
        self.drivers
            .values()
            .filter(|d| d.is_available())
            .map(|d| d.name().to_string())
            .collect()
    }
}

impl Default for DriverRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let registry = DriverRegistry::new();
        let drivers = registry.list_drivers();

        // VFS should always be registered
        assert!(drivers.contains(&"vfs".to_string()));

        // Overlay2 only on Linux with feature
        #[cfg(all(target_os = "linux", feature = "overlayfs"))]
        assert!(drivers.contains(&"overlay2".to_string()));
    }

    #[test]
    fn test_auto_select() {
        let registry = DriverRegistry::new();
        let driver = registry.auto_select().expect("Should always select a driver");

        // Should select best available
        println!("Auto-selected driver: {}", driver.name());
        assert!(driver.is_available());
    }

    #[test]
    fn test_vfs_always_available() {
        let registry = DriverRegistry::new();
        let driver = registry.get_driver(StorageDriver::Vfs).unwrap();
        assert_eq!(driver.name(), "vfs");
        assert!(driver.is_available());
    }

    #[test]
    fn test_list_available() {
        let registry = DriverRegistry::new();
        let available = registry.list_available_drivers();

        println!("Available drivers:");
        for name in &available {
            println!("  - {}", name);
        }

        // At least vfs should be available
        assert!(!available.is_empty());
    }
}
