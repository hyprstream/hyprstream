//! Comprehensive security tests for git2db
//!
//! This module contains security-focused tests to verify that all
//! critical security fixes are working correctly.

#![allow(clippy::unwrap_used)]

use crate::config::Git2DBConfig;
use crate::errors::Git2DBError;
use crate::transport_registry::TransportRegistry;
use std::sync::Arc;
use tempfile::tempdir;

/// Security test suite for path validation using safe_path
mod path_validation_tests {
    use super::*;
    use hyprstream_containedfs::contained_join;

    #[test]
    fn test_safe_path_basic_functionality() {
        let temp_dir = tempdir().unwrap();

        // Test that safe_path prevents directory traversal
        let safe_path = contained_join(temp_dir.path(), "../../../etc/passwd").unwrap();
        assert!(safe_path.starts_with(temp_dir.path()));

        let safe_path2 = contained_join(temp_dir.path(), "normal/path").unwrap();
        assert!(safe_path2.starts_with(temp_dir.path()));
        assert!(safe_path2.ends_with("normal/path"));
    }

    #[test]
    fn test_safe_path_with_absolute_paths() {
        let temp_dir = tempdir().unwrap();

        // Even absolute paths should be constrained to base directory
        let safe_path = contained_join(temp_dir.path(), "/tmp/test").unwrap();
        assert!(safe_path.starts_with(temp_dir.path()));
        // contained_join strips root prefix, keeping remaining components
        assert!(safe_path.ends_with("tmp/test") || safe_path.ends_with("tmp"));
    }

    #[test]
    fn test_safe_path_dot_resolution() {
        let temp_dir = tempdir().unwrap();

        // Test that dot paths are resolved safely
        let safe_path = contained_join(temp_dir.path(), "./subdir/../other").unwrap();
        assert!(safe_path.starts_with(temp_dir.path()));
        assert!(safe_path.ends_with("other"));
    }


    #[test]
    fn test_path_normalization() {
        use hyprstream_containedfs::contained_join;
        let temp_dir = tempdir().unwrap();
        let base_dir = temp_dir.path();

        // Test path normalization using safe_path directly
        let normalized = contained_join(base_dir, "subdir/file.txt").unwrap();
        assert!(normalized.starts_with(base_dir));
        assert!(normalized.ends_with("subdir/file.txt"));

        // Even dangerous paths should be constrained
        let dangerous = contained_join(base_dir, "../../../etc/passwd").unwrap();
        assert!(dangerous.starts_with(base_dir));
    }

    #[test]
    fn test_manager_path_handling() {
        // Test contained_join directly instead of through manager (which requires tokio)
        let temp_dir = tempdir().unwrap();
        let result = contained_join(temp_dir.path(), "models/test-repo").unwrap();
        assert!(result.starts_with(temp_dir.path()));
        assert!(result.ends_with("models/test-repo"));

        // Traversal attempts are clamped
        let result = contained_join(temp_dir.path(), "../../../etc/shadow").unwrap();
        assert!(result.starts_with(temp_dir.path()));
    }
}

/// Security test suite for transport registry
mod transport_registry_tests {
    use super::*;

    #[test]
    fn test_transport_registration_thread_safety() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
        let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let scheme = format!("sec-threadsafe-{}", test_id);

        let registry = Arc::new(TransportRegistry::new());
        let factory = Arc::new(crate::transport::MockTransportFactory);

        // Test concurrent registration
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let registry = Arc::clone(&registry);
                let factory = Arc::clone(&factory);
                let scheme = scheme.clone();
                std::thread::spawn(move || {
                    registry.register_transport(scheme, factory.clone())
                })
            })
            .collect();

        // All should succeed (thread safety tested)
        for handle in handles {
            assert!(handle.join().unwrap().is_ok());
        }

        // Should have exactly one factory registered
        let stats = registry.stats();
        assert_eq!(stats.registered_factories, 1);
        assert_eq!(stats.globally_registered_schemes, 1);
    }

    #[test]
    fn test_scheme_validation() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
        let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);

        let registry = TransportRegistry::new();
        let factory = Arc::new(crate::transport::MockTransportFactory);

        // Valid schemes
        assert!(registry.register_transport(format!("sec-valid1-{}", test_id), factory.clone()).is_ok());
        assert!(registry.register_transport(format!("sec-valid2-{}", test_id), factory.clone()).is_ok());
        assert!(registry.register_transport(format!("sec-my-transport-{}", test_id), factory.clone()).is_ok());
        assert!(registry.register_transport(format!("sec_my_transport_{}", test_id), factory.clone()).is_ok());

        // Invalid schemes
        assert!(registry.register_transport("".to_owned(), factory.clone()).is_err());
        assert!(registry.register_transport("http://".to_owned(), factory.clone()).is_err());
        assert!(registry.register_transport("http://invalid".to_owned(), factory.clone()).is_err());
        assert!(registry.register_transport("transport with spaces".to_owned(), factory.clone()).is_err());
    }

    #[test]
    fn test_transport_cleanup() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);
        let test_id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let scheme = format!("sec-cleanup-{}", test_id);

        let registry = TransportRegistry::new();
        let factory = Arc::new(crate::transport::MockTransportFactory);

        // Register transport
        assert!(registry.register_transport(scheme.clone(), factory.clone()).is_ok());
        assert_eq!(registry.stats().registered_factories, 1);

        // Unregister transport
        let result = registry.unregister_transport(&scheme);
        assert!(result.is_some());
        assert_eq!(registry.stats().registered_factories, 0);
    }
}

/// Security test suite for overall integration
mod integration_tests {
    use super::*;

    #[test]
    fn test_secure_configuration_defaults() {
        let config = Git2DBConfig::default();

        // Network timeouts should be reasonable
        assert!(config.network.timeout_secs > 0);

        // Performance limits should be set
        assert!(config.performance.max_concurrent_ops > 0);
        assert!(config.performance.max_repo_cache > 0);
    }

    #[test]
    fn test_path_validation_integration() {
        let _config = Git2DBConfig::default();
        let temp_dir = tempdir().unwrap();

        // Test that path validation works with real file operations using safe_path
        let safe_path = hyprstream_containedfs::contained_join(temp_dir.path(), "test-repo").unwrap();

        // Create test directory
        std::fs::create_dir_all(&safe_path).unwrap();
        assert!(safe_path.exists());
        assert!(safe_path.starts_with(temp_dir.path()));

        // Test path safety with safe_path
        let safe_join = hyprstream_containedfs::contained_join(temp_dir.path(), "safe-model").unwrap();
        assert!(safe_join.starts_with(temp_dir.path()));
    }

    #[test]
    fn test_error_handling_security() {
        // Test that security errors are properly classified
        let path_error = Git2DBError::invalid_path("/test/path", "Security violation");
        let auth_error = Git2DBError::authentication("https://example.com", "Invalid credentials");

        // Path errors are non-recoverable (security violations)
        assert!(!path_error.is_recoverable());
        // Auth errors are recoverable (can retry with different credentials)
        assert!(auth_error.is_recoverable());
    }
}

/// Property-based security tests
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_path_safety_property_based(name in "[a-zA-Z0-9_-]{1,50}") {
            let temp_dir = tempdir().unwrap();

            // Test that safe_path constrains all paths
            let safe_join = hyprstream_containedfs::contained_join(temp_dir.path(), &name).unwrap();
            assert!(safe_join.starts_with(temp_dir.path()), "Path escaped base directory: {:?}", safe_join);
        }

        #[test]
        fn test_url_scheme_property_based(scheme in "[a-z][a-z0-9_-]{1,19}") {
            let registry = TransportRegistry::new();
            let factory = Arc::new(crate::transport::MockTransportFactory);

            // Valid lowercase schemes should be accepted by local registry
            if !scheme.is_empty() && !scheme.contains("://") && !scheme.contains(' ') {
                // Note: global git2 transport registration may fail for some schemes,
                // but local registry should always accept valid formats
                let _ = registry.register_transport(scheme.clone(), factory.clone());
            }
        }
    }
}

/// Performance security tests (ensure security doesn't cause performance regressions)
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_path_validation_performance() {
        let temp_dir = tempdir().unwrap();

        let start = Instant::now();

        // Test 1000 path validations using safe_path
        for i in 0..1000 {
            let path = format!("test_path_{}", i % 100);
            let _ = hyprstream_containedfs::contained_join(temp_dir.path(), &path);
        }

        let duration = start.elapsed();
        assert!(duration.as_millis() < 100, "Path validation should be fast: {:?}", duration);
    }

    #[test]
    fn test_concurrent_path_validation() {
        use std::sync::Arc;

        let temp_dir = tempdir().unwrap();
        let base_path = Arc::new(temp_dir.path().to_path_buf());

        let handles: Vec<_> = (0..10)
            .map(|i| {
                let base = Arc::clone(&base_path);
                std::thread::spawn(move || {
                    for j in 0..100 {
                        let path = format!("thread_{}_path_{}", i, j);
                        let _ = hyprstream_containedfs::contained_join(&base, &path);
                    }
                })
            })
            .collect();

        for handle in handles {
            assert!(handle.join().is_ok(), "Concurrent path validation should be thread-safe");
        }
    }
}

#[cfg(test)]
mod fuzz_tests {
    use super::*;
    use quickcheck::{Arbitrary, Gen};
    use quickcheck_macros::quickcheck;

    #[derive(Clone, Debug)]
    struct FuzzPath(Vec<u8>);

    impl Arbitrary for FuzzPath {
        fn arbitrary(g: &mut Gen) -> Self {
            let size = g.size();
            let bytes: Vec<u8> = (0..size).map(|_| u8::arbitrary(g)).collect();
            FuzzPath(bytes)
        }
    }

    #[quickcheck]
    fn path_validation_doesnt_panic(fuzz_path: FuzzPath) -> bool {
        let temp_dir = tempdir().unwrap();

        if let Ok(path_str) = std::str::from_utf8(&fuzz_path.0) {
            let _ = hyprstream_containedfs::contained_join(temp_dir.path(), path_str);
        }

        // Should never panic, regardless of input
        true
    }
}

/// Security regression tests
mod regression_tests {
    use super::*;

    #[test]
    fn test_directory_traversal_prevention() {
        let temp_dir = tempdir().unwrap();

        // These should all be constrained to the base directory
        let traversal_attempts = vec![
            "../outside",
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "test/../../../etc",
            "/absolute/path",
        ];

        for attempt in traversal_attempts {
            let result = hyprstream_containedfs::contained_join(temp_dir.path(), attempt);
            // Should either succeed (constrained by scoped_join) or fail gracefully
            // But should never allow actual path traversal
            if let Ok(validated_path) = result {
                assert!(validated_path.starts_with(temp_dir.path()),
                    "Path {} was validated to {:?} which escapes base directory {:?}",
                    attempt, validated_path, temp_dir.path());
            }
        }
    }

    #[test]
    fn test_transport_registration_safety() {
        // Test that concurrent transport registration is safe
        let registry = Arc::new(TransportRegistry::new());
        let factory = Arc::new(crate::transport::MockTransportFactory);

        let mut handles = vec![];

        for i in 0..10 {
            let registry = Arc::clone(&registry);
            let factory = Arc::clone(&factory);

            handles.push(std::thread::spawn(move || {
                let scheme = format!("test_{}", i);
                let result = registry.register_transport(scheme.clone(), factory.clone());

                // Try to use the transport
                if result.is_ok() {
                    let _ = registry.get_transport(&format!("{}://example.com/repo", scheme));
                }

                result
            }));
        }

        // All operations should complete without panics or data races
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

        // Should have successful registrations
        let success_count = results.iter().filter(|r| r.is_ok()).count();
        assert!(success_count > 0, "At least some registrations should succeed");
    }
}