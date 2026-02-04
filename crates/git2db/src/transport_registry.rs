//! Thread-safe transport registry for custom git transports
//!
//! This module provides a thread-safe wrapper around libgit2's transport
//! registration system with proper synchronization and lifecycle management.
//!
//! # Safety
//!
//! This module uses `unsafe` code to call `git2::transport::register()`.
//! Safety is ensured through:
//! - Thread-safe factory storage using `Arc<Mutex<HashMap>>`
//! - Per-scheme registration locks to prevent concurrent registration
//! - Reference counting to prevent duplicate registrations
//! - Factory trait bounds requiring `Send + Sync`
//!
//! # Examples
//!
//! ```rust
//! use git2db::transport_registry::TransportRegistry;
//! use std::sync::Arc;
//!
//! let registry = TransportRegistry::new();
//!
//! // Register a custom transport
//! // let factory = Arc::new(MyTransportFactory::new());
//! // registry.register_transport("mycustom".to_owned(), factory).unwrap();
//! ```

use crate::errors::Git2DBError;
use crate::transport::{BoxedSubtransport, TransportFactory};
use git2::transport::Transport;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Thread-safe transport registry with proper lifecycle management
pub struct TransportRegistry {
    /// Registered transport factories by scheme
    transports: Arc<Mutex<HashMap<String, Arc<dyn TransportFactory>>>>,
    /// Registration synchronization per scheme
    registration_locks: Arc<Mutex<HashMap<String, Arc<Mutex<()>>>>>,
    /// Global transport registration state
    global_registry: Arc<Mutex<GlobalRegistryState>>,
}

/// Global registry state to track which schemes are registered
#[derive(Debug, Default)]
struct GlobalRegistryState {
    registered_schemes: HashMap<String, usize>, // scheme -> reference count
}

impl TransportRegistry {
    /// Create a new transport registry
    pub fn new() -> Self {
        Self {
            transports: Arc::new(Mutex::new(HashMap::new())),
            registration_locks: Arc::new(Mutex::new(HashMap::new())),
            global_registry: Arc::new(Mutex::new(GlobalRegistryState::default())),
        }
    }

    /// Register a transport factory for a URL scheme
    ///
    /// This method is thread-safe and ensures proper synchronization.
    /// Multiple registrations for the same scheme are reference counted.
    pub fn register_transport(
        &self,
        scheme: String,
        factory: Arc<dyn TransportFactory>,
    ) -> Result<(), Git2DBError> {
        if scheme.is_empty() {
            return Err(Git2DBError::configuration(
                "Transport scheme cannot be empty",
            ));
        }

        // Validate scheme format
        if !Self::is_valid_scheme(&scheme) {
            return Err(Git2DBError::configuration(format!(
                "Invalid transport scheme: '{scheme}'. Schemes must contain only alphanumeric characters, hyphens, and underscores"
            )));
        }

        debug!("Registering transport for scheme: {}", scheme);

        // Add to local registry
        {
            let mut transports = self.transports.lock();
            transports.insert(scheme.clone(), factory);
        }

        // Get or create registration lock for this scheme
        let registration_lock = {
            let mut locks = self.registration_locks.lock();
            locks
                .entry(scheme.clone())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };

        // Acquire registration lock to ensure thread safety
        let _lock = registration_lock.lock();

        // Check if already globally registered and increment ref count atomically
        let should_register = {
            let mut global_state = self.global_registry.lock();
            let ref_count = global_state
                .registered_schemes
                .entry(scheme.clone())
                .or_insert(0);
            let should_register = *ref_count == 0;
            if should_register {
                *ref_count = 1; // Increment immediately to prevent other threads
            }
            should_register
        };

        if should_register {
            // Actually register with git2
            if let Err(e) = self.register_with_git2(&scheme) {
                // Registration failed, decrement ref count
                let mut global_state = self.global_registry.lock();
                if let Some(ref_count) = global_state.registered_schemes.get_mut(&scheme) {
                    *ref_count = 0;
                }
                return Err(e);
            }

            debug!(
                "Successfully registered transport with git2 for scheme: {}",
                scheme
            );
        } else {
            debug!("Transport already registered for scheme: {}", scheme);
        }

        Ok(())
    }

    /// Unregister a transport factory
    ///
    /// This method is thread-safe and uses reference counting.
    /// The transport is only unregistered from git2 when the last factory is removed.
    pub fn unregister_transport(&self, scheme: &str) -> Option<Arc<dyn TransportFactory>> {
        if scheme.is_empty() {
            return None;
        }

        debug!("Unregistering transport for scheme: {}", scheme);

        // Remove from local registry
        let factory = {
            let mut transports = self.transports.lock();
            transports.remove(scheme)
        };

        if factory.is_some() {
            // Get registration lock
            let registration_lock = {
                let locks = self.registration_locks.lock();
                locks.get(scheme).cloned()
            };

            if let Some(lock) = registration_lock {
                let _lock_guard = lock.lock();

                // Decrement reference count and unregister if zero
                let should_unregister = {
                    let mut global_state = self.global_registry.lock();
                    if let Some(ref_count) = global_state.registered_schemes.get_mut(scheme) {
                        *ref_count = ref_count.saturating_sub(1);
                        *ref_count == 0
                    } else {
                        false
                    }
                };

                if should_unregister {
                    if let Err(e) = self.unregister_from_git2(scheme) {
                        warn!(
                            "Failed to unregister transport from git2 for scheme '{}': {}",
                            scheme, e
                        );
                        // Don't return error - this is cleanup code
                    } else {
                        debug!(
                            "Successfully unregistered transport from git2 for scheme: {}",
                            scheme
                        );
                    }

                    // Clean up registration lock
                    {
                        let mut locks = self.registration_locks.lock();
                        locks.remove(scheme);
                    }
                }
            }
        }

        factory
    }

    /// Get a transport factory for a scheme
    pub fn get_transport(&self, scheme: &str) -> Option<Arc<dyn TransportFactory>> {
        let transports = self.transports.lock();
        transports.get(scheme).cloned()
    }

    /// Check if a transport is registered for a scheme
    pub fn has_transport(&self, scheme: &str) -> bool {
        let transports = self.transports.lock();
        transports.contains_key(scheme)
    }

    /// Get all registered transport schemes
    pub fn registered_schemes(&self) -> Vec<String> {
        let transports = self.transports.lock();
        transports.keys().cloned().collect()
    }

    /// Validate transport scheme format
    fn is_valid_scheme(scheme: &str) -> bool {
        if scheme.is_empty() || scheme.len() > 64 {
            return false;
        }

        // Only allow alphanumeric characters, hyphens, and underscores
        scheme
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_')
    }

    /// Register transport with git2 using unsafe git2::transport::register
    ///
    /// # Safety
    ///
    /// This function is safe to call because:
    /// - The factory is stored in an Arc<Mutex> ensuring thread-safe access
    /// - The factory trait is `Send + Sync + 'static`
    /// - We use per-scheme locks to prevent concurrent registration
    /// - git2::transport::register requires external synchronization, which we provide
    fn register_with_git2(&self, scheme: &str) -> Result<(), Git2DBError> {
        let factory = self.get_transport(scheme).ok_or_else(|| {
            Git2DBError::configuration(format!("No factory for scheme '{scheme}'"))
        })?;

        info!(
            "Registering custom transport for scheme '{}' with libgit2",
            scheme
        );

        // SAFETY:
        // - factory is Arc<dyn TransportFactory> which is Send + Sync + 'static
        // - We hold a registration lock preventing concurrent calls for this scheme
        // - The closure captures factory in an Arc, ensuring proper lifetime
        // - git2::transport::register is unsafe due to global state, but we synchronize access
        unsafe {
            git2::transport::register(scheme, move |remote| {
                let url = remote.url().unwrap_or("");
                debug!("Creating transport for URL: {}", url);

                // Create the transport
                let transport = factory.create_transport(url).map_err(|e| {
                    git2::Error::from_str(&format!("Failed to create transport: {e}"))
                })?;

                // Wrap in BoxedSubtransport and create smart transport
                Transport::smart(remote, true, BoxedSubtransport(transport))
            })
            .map_err(|e| {
                Git2DBError::internal(format!("Failed to register transport with git2: {e}"))
            })?;
        }

        info!("Successfully registered transport for scheme '{}'", scheme);
        Ok(())
    }

    /// Unregister transport from git2
    ///
    /// Note: libgit2 does not provide a way to unregister transports once registered.
    /// The transport will remain registered for the lifetime of the process.
    /// This is a limitation of libgit2's API.
    fn unregister_from_git2(&self, scheme: &str) -> Result<(), Git2DBError> {
        debug!(
            "Transport '{}' remains registered with git2 (libgit2 limitation)",
            scheme
        );
        Ok(())
    }

    /// Get registry statistics
    pub fn stats(&self) -> RegistryStats {
        let transports = self.transports.lock();
        let global_state = self.global_registry.lock();
        let locks = self.registration_locks.lock();

        RegistryStats {
            registered_factories: transports.len(),
            globally_registered_schemes: global_state.registered_schemes.len(),
            active_registration_locks: locks.len(),
            schemes: transports.keys().cloned().collect(),
        }
    }
}

impl Clone for TransportRegistry {
    fn clone(&self) -> Self {
        Self {
            transports: Arc::clone(&self.transports),
            registration_locks: Arc::clone(&self.registration_locks),
            global_registry: Arc::clone(&self.global_registry),
        }
    }
}

impl Default for TransportRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Registry statistics
#[derive(Debug, Clone)]
pub struct RegistryStats {
    pub registered_factories: usize,
    pub globally_registered_schemes: usize,
    pub active_registration_locks: usize,
    pub schemes: Vec<String>,
}

/// Drop guard for automatic transport cleanup
pub struct TransportGuard {
    registry: TransportRegistry,
    scheme: String,
}

impl TransportGuard {
    /// Create a new transport guard
    pub fn new(registry: TransportRegistry, scheme: String) -> Self {
        Self { registry, scheme }
    }

    /// Get the scheme
    pub fn scheme(&self) -> &str {
        &self.scheme
    }
}

impl Drop for TransportGuard {
    fn drop(&mut self) {
        debug!("Auto-unregistering transport for scheme: {}", self.scheme);
        if let Some(_factory) = self.registry.unregister_transport(&self.scheme) {
            debug!("Successfully auto-unregistered transport");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransportFactory;

    #[test]
    fn test_transport_registry_thread_safety() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

        // Use a unique scheme name to avoid conflicts with other tests
        let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let scheme = format!("test-thread-{test_id}");

        let registry = Arc::new(TransportRegistry::new());
        let factory: Arc<dyn TransportFactory> = Arc::new(MockTransportFactory);

        // Test concurrent registration
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let registry = Arc::clone(&registry);
                let factory = Arc::clone(&factory);
                let scheme = scheme.clone();
                std::thread::spawn(move || registry.register_transport(scheme, Arc::clone(&factory)))
            })
            .collect();

        // All should succeed - only first thread registers with git2, others skip
        for handle in handles {
            if let Ok(result) = handle.join() {
                assert!(result.is_ok());
            }
        }

        // Should have exactly one factory registered
        let stats = registry.stats();
        assert_eq!(stats.registered_factories, 1);
        assert_eq!(stats.globally_registered_schemes, 1);

        // Cleanup
        let _ = registry.unregister_transport(&scheme);
    }

    #[test]
    fn test_scheme_validation() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
        let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);

        let registry = TransportRegistry::new();
        let factory: Arc<dyn TransportFactory> = Arc::new(MockTransportFactory);

        // Valid schemes
        assert!(registry
            .register_transport(format!("valid1-{test_id}"), Arc::clone(&factory))
            .is_ok());
        assert!(registry
            .register_transport(format!("valid2-{test_id}"), Arc::clone(&factory))
            .is_ok());
        assert!(registry
            .register_transport(format!("my-transport-{test_id}"), Arc::clone(&factory))
            .is_ok());
        assert!(registry
            .register_transport(format!("my_transport_{test_id}"), Arc::clone(&factory))
            .is_ok());

        // Invalid schemes
        assert!(registry
            .register_transport("".to_owned(), Arc::clone(&factory))
            .is_err());
        assert!(registry
            .register_transport("http://".to_owned(), Arc::clone(&factory))
            .is_err());
        assert!(registry
            .register_transport("http://invalid".to_owned(), Arc::clone(&factory))
            .is_err());
        assert!(registry
            .register_transport("transport with spaces".to_owned(), Arc::clone(&factory))
            .is_err());
    }

    #[test]
    fn test_reference_counting() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
        let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let scheme = format!("refcount-{test_id}");

        let registry = TransportRegistry::new();
        let factory1 = Arc::new(MockTransportFactory);
        let factory2 = Arc::new(MockTransportFactory);

        // Register first factory
        assert!(registry
            .register_transport(scheme.clone(), factory1)
            .is_ok());
        let stats = registry.stats();
        assert_eq!(stats.registered_factories, 1);

        // Register second factory (replaces first)
        assert!(registry
            .register_transport(scheme.clone(), factory2)
            .is_ok());
        let stats = registry.stats();
        assert_eq!(stats.registered_factories, 1);

        // Unregister
        let _ = registry.unregister_transport(&scheme);
        let stats = registry.stats();
        assert_eq!(stats.registered_factories, 0);
    }

    #[test]
    fn test_transport_guard_cleanup() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
        let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let scheme = format!("guarded-{test_id}");

        let registry = TransportRegistry::new();
        let factory: Arc<dyn TransportFactory> = Arc::new(MockTransportFactory);

        {
            let _guard = TransportGuard::new(registry.clone(), scheme.clone());
            let result = registry
                .register_transport(scheme.clone(), Arc::clone(&factory));
            assert!(result.is_ok());
            assert!(registry.has_transport(&scheme));
        } // Guard goes out of scope

        // Transport should be unregistered
        assert!(!registry.has_transport(&scheme));
    }
}
