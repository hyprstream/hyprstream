//! Driver registry (Docker's graphdriver pattern)
//!
//! Manages driver registration and selection, similar to Docker's
//! daemon/graphdriver/driver_linux.go

use super::driver::{Driver, DriverError, StorageDriver, DriverFactory};
use inventory;
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

        // Load drivers from inventory registry
        for factory in inventory::iter::<DriverFactory> {
            info!("Registering storage driver: {}", factory.name());
            drivers.insert(
                factory.name().to_string(),
                factory.get_driver()
            );
        }

        Self { drivers }
    }

    /// Get a driver by selection
    ///
    /// Resolves Auto to the best available driver.
    pub fn get_driver(&self, selection: StorageDriver) -> Result<Arc<dyn Driver>, DriverError> {
        match selection {
            StorageDriver::Auto => self.auto_select(),
            StorageDriver::Overlay2 => self.get("overlay2"),
            StorageDriver::Reflink => self.get("reflink"),
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
    /// 1. overlay2 (Linux overlayfs) - best space savings for incremental edits
    /// 2. reflink (cross-platform CoW) - good space savings, simpler cleanup
    /// 3. Future: btrfs, hardlink
    /// 4. vfs (plain directories) - always works
    fn auto_select(&self) -> Result<Arc<dyn Driver>, DriverError> {
        let priorities = [
            "overlay2",
            "reflink",
            "btrfs",
            "hardlink",
            "vfs",
        ];

        for name in &priorities {
            if let Some(driver) = self.drivers.get(*name) {
                if driver.is_available() {
                    info!("Auto-selected storage driver: {}", driver.name());
                    return Ok(driver.clone());
                }
            }
        }

        // No drivers available - this should never happen in practice
        Err(DriverError::NotAvailable("No storage drivers available".to_string()))
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
        let driver = registry
            .auto_select()
            .expect("Should always select a driver");

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
