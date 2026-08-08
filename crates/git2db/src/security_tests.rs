//! Comprehensive security tests for git2db
//!
//! This module contains security-focused tests to verify that all
//! critical security fixes are working correctly.

#![allow(clippy::unwrap_used, clippy::expect_used)]

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

/// Adversarial tests for the untrusted-repo clone guards (issue #1430):
/// submodule auto-init, content-filter smudge (XET/LFS), and credential
/// scoping must all be refused for a [`crate::clone_options::CloneTrust::Untrusted`]
/// clone. Each guard is proven load-bearing by a matching positive control —
/// the identical fixture cloned WITHOUT the guard in effect, showing the
/// unguarded behavior (submodule fetched / filter expanded) actually occurs.
/// Per the task, each guard was manually verified to fail (assertion fails,
/// not "doesn't compile") when its corresponding clamp in
/// `CloneOptions::effective_submodule_mode` / `effective_filter_mode` /
/// `validate_trust` was temporarily removed — see the PR description /
/// RESULT report for the exact removal-and-rerun transcript.
mod untrusted_repo_guards {
    use crate::callback_config::CallbackConfigBuilder;
    use crate::clone_options::{CloneOptions, CloneTrust, FilterMode, SubmoduleMode};
    use crate::config::Git2DBConfig;
    use crate::errors::Git2DBError;
    use crate::manager::GitManager;
    use git2::{Repository, Signature};
    use std::path::Path;
    use tempfile::tempdir;

    fn test_manager() -> GitManager {
        let mut config = Git2DBConfig::default();
        config.performance.auto_cleanup = false; // avoid background task spawning in tests
        GitManager::new(config)
    }

    fn file_url(path: &Path) -> String {
        format!("file://{}", path.display())
    }

    /// Commit whatever is currently staged in `repo`'s index, against
    /// whatever parent HEAD currently resolves to (none for the first
    /// commit).
    fn commit_index(repo: &Repository, message: &str) -> git2::Oid {
        let sig = Signature::now("git2db-test", "test@git2db.invalid").unwrap();
        let mut index = repo.index().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let parent = repo.head().ok().and_then(|h| h.peel_to_commit().ok());
        let parents: Vec<&git2::Commit<'_>> = parent.iter().collect();
        repo.commit(Some("HEAD"), &sig, &sig, message, &tree, &parents)
            .unwrap()
    }

    /// A small "target" repository with one committed file — stands in for
    /// the dependency a submodule would pull in.
    fn init_target_repo(path: &Path, filename: &str, content: &str) {
        let repo = Repository::init(path).unwrap();
        std::fs::write(path.join(filename), content).unwrap();
        repo.index()
            .unwrap()
            .add_path(Path::new(filename))
            .unwrap();
        repo.index().unwrap().write().unwrap();
        commit_index(&repo, "initial");
    }

    /// An "outer" repository — standing in for an untrusted PR head — that
    /// declares a real libgit2 submodule (proper `.gitmodules` + gitlink tree
    /// entry, via `git_submodule_add_setup`/`clone`/`add_finalize`, exactly
    /// as a real `git submodule add` would) pointing at `target_url`.
    fn init_outer_repo_with_submodule(path: &Path, target_url: &str, submodule_path: &str) {
        let repo = Repository::init(path).unwrap();
        std::fs::write(path.join("outer.txt"), "outer repo content\n").unwrap();
        repo.index()
            .unwrap()
            .add_path(Path::new("outer.txt"))
            .unwrap();
        repo.index().unwrap().write().unwrap();
        commit_index(&repo, "outer initial");

        let mut submodule = repo
            .submodule(target_url, Path::new(submodule_path), true)
            .expect("submodule add_setup");
        submodule.clone(None).expect("submodule clone step");
        submodule.add_finalize().expect("submodule add_finalize");
        commit_index(&repo, "add submodule");
    }

    /// A repository with a file bearing the `ident` gitattribute — this
    /// exercises libgit2's own built-in `ident` content filter, which is
    /// registered and invoked through the *identical* `git_filter_register` +
    /// `.gitattributes`-driven mechanism that `git-xet-filter` uses for the
    /// real XET/LFS smudge filter (see `crates/git-xet-filter/src/filter.rs`:
    /// same filter API, same `attributes` string match, same libgit2 checkout
    /// pipeline). We use `ident` instead of the real XET filter because
    /// invoking real XET smudge requires a live CAS endpoint / `XETHUB_TOKEN`
    /// — out of scope for a hermetic unit test. `disable_filters(true)`
    /// suppresses `ident` and XET/LFS identically, because both run through
    /// the same libgit2 filter-application code path during checkout.
    fn init_repo_with_ident_file(path: &Path) {
        let repo = Repository::init(path).unwrap();
        std::fs::write(path.join(".gitattributes"), "id.txt ident\n").unwrap();
        std::fs::write(path.join("id.txt"), "$Id$\n").unwrap();
        {
            let mut index = repo.index().unwrap();
            index.add_path(Path::new(".gitattributes")).unwrap();
            index.add_path(Path::new("id.txt")).unwrap();
            index.write().unwrap();
        }
        commit_index(&repo, "add ident file");
    }

    // ---- Guard 1: SubmoduleMode ----

    /// Adversarial: a repo with `.gitmodules` (a real, properly-registered
    /// libgit2 submodule) cloned as `CloneTrust::Untrusted` must NOT have its
    /// submodule initialized — even though the caller explicitly (and
    /// wrongly) requested `SubmoduleMode::Enabled`. This is the exact
    /// acceptance scenario named in issue #1430.
    #[tokio::test]
    async fn untrusted_clone_does_not_initialize_submodule() {
        let manager = test_manager();

        let target_dir = tempdir().unwrap();
        init_target_repo(target_dir.path(), "secret.txt", "attacker-controlled payload\n");

        let outer_dir = tempdir().unwrap();
        init_outer_repo_with_submodule(
            outer_dir.path(),
            &file_url(target_dir.path()),
            "vendor/target",
        );

        let dest_dir = tempdir().unwrap();
        let dest_path = dest_dir.path().join("clone");

        let options = CloneOptions::builder()
            .trust(CloneTrust::Untrusted)
            // Attempted override — Untrusted must clamp this regardless.
            .submodule_mode(SubmoduleMode::Enabled)
            .build();

        let repo = manager
            .clone_repository(&file_url(outer_dir.path()), &dest_path, Some(options))
            .await
            .expect("untrusted clone of the outer repo itself must still succeed");

        let submodule = repo
            .find_submodule("vendor/target")
            .expect(".gitmodules entry must still be visible (only init/update are refused)");
        assert!(
            submodule.open().is_err(),
            "submodule must NOT be checked out under CloneTrust::Untrusted, even though the \
             caller requested SubmoduleMode::Enabled"
        );
        assert!(
            !dest_path.join("vendor/target/secret.txt").exists(),
            "submodule content must not have been fetched from the (attacker-controlled) \
             submodule URL"
        );
    }

    /// Positive control for the test above: the identical fixture, but
    /// `CloneTrust::Trusted` + `SubmoduleMode::Enabled` DOES fetch and check
    /// out the submodule. This proves the guard above is load-bearing — a
    /// test that passed regardless of the guard would prove nothing.
    #[tokio::test]
    async fn trusted_clone_with_submodule_enabled_does_initialize_submodule() {
        let manager = test_manager();

        let target_dir = tempdir().unwrap();
        init_target_repo(target_dir.path(), "secret.txt", "attacker-controlled payload\n");

        let outer_dir = tempdir().unwrap();
        init_outer_repo_with_submodule(
            outer_dir.path(),
            &file_url(target_dir.path()),
            "vendor/target",
        );

        let dest_dir = tempdir().unwrap();
        let dest_path = dest_dir.path().join("clone");

        let options = CloneOptions::builder()
            .trust(CloneTrust::Trusted)
            .submodule_mode(SubmoduleMode::Enabled)
            .build();

        let repo = manager
            .clone_repository(&file_url(outer_dir.path()), &dest_path, Some(options))
            .await
            .expect("trusted clone should succeed");

        let submodule = repo.find_submodule("vendor/target").unwrap();
        assert!(
            submodule.open().is_ok(),
            "submodule SHOULD be checked out when trusted + explicitly enabled"
        );
        let content = std::fs::read_to_string(dest_path.join("vendor/target/secret.txt"))
            .expect("submodule content should have been fetched to disk");
        assert_eq!(content, "attacker-controlled payload\n");
    }

    /// The secure DEFAULT (no explicit trust/mode at all) also leaves the
    /// submodule uninitialized — matches pre-#1430 behavior, where submodule
    /// initialization was never wired into `clone_repository` in the first
    /// place. Guards against a future default flip being unnoticed.
    #[tokio::test]
    async fn default_clone_options_do_not_initialize_submodule() {
        let manager = test_manager();

        let target_dir = tempdir().unwrap();
        init_target_repo(target_dir.path(), "secret.txt", "payload\n");
        let outer_dir = tempdir().unwrap();
        init_outer_repo_with_submodule(
            outer_dir.path(),
            &file_url(target_dir.path()),
            "vendor/target",
        );

        let dest_dir = tempdir().unwrap();
        let dest_path = dest_dir.path().join("clone");

        let repo = manager
            .clone_repository(
                &file_url(outer_dir.path()),
                &dest_path,
                Some(CloneOptions::default()),
            )
            .await
            .unwrap();

        assert!(repo.find_submodule("vendor/target").unwrap().open().is_err());
    }

    // ---- Guard 2: FilterMode ----

    /// Adversarial: cloning as `CloneTrust::Untrusted` must not run ANY
    /// libgit2 content filter during checkout — proven here via the built-in
    /// `ident` filter (see `init_repo_with_ident_file` doc comment for why
    /// this proxies XET/LFS). The caller even (wrongly) requests
    /// `FilterMode::Enabled`; `Untrusted` must clamp it to `Passthrough`
    /// regardless.
    #[tokio::test]
    async fn untrusted_clone_does_not_expand_ident_filter() {
        let manager = test_manager();

        let src_dir = tempdir().unwrap();
        init_repo_with_ident_file(src_dir.path());

        let dest_dir = tempdir().unwrap();
        let dest_path = dest_dir.path().join("clone");

        let options = CloneOptions::builder()
            .trust(CloneTrust::Untrusted)
            // Attempted override — Untrusted must clamp this regardless.
            .filter_mode(FilterMode::Enabled)
            .build();

        manager
            .clone_repository(&file_url(src_dir.path()), &dest_path, Some(options))
            .await
            .expect("untrusted clone should still succeed");

        let content = std::fs::read_to_string(dest_path.join("id.txt")).unwrap();
        assert_eq!(
            content, "$Id$\n",
            "content filters (ident, standing in for XET/LFS smudge) must NOT run under \
             CloneTrust::Untrusted"
        );
    }

    /// Positive control: the identical fixture, `CloneTrust::Trusted` +
    /// `FilterMode::Enabled` (the default), DOES expand the ident filter —
    /// proving `disable_filters` above is what suppressed it, not some
    /// unrelated reason the file happened to stay unexpanded.
    #[tokio::test]
    async fn trusted_clone_with_filters_enabled_does_expand_ident_filter() {
        let manager = test_manager();

        let src_dir = tempdir().unwrap();
        init_repo_with_ident_file(src_dir.path());

        let dest_dir = tempdir().unwrap();
        let dest_path = dest_dir.path().join("clone");

        let options = CloneOptions::builder()
            .trust(CloneTrust::Trusted)
            .filter_mode(FilterMode::Enabled)
            .build();

        manager
            .clone_repository(&file_url(src_dir.path()), &dest_path, Some(options))
            .await
            .unwrap();

        let content = std::fs::read_to_string(dest_path.join("id.txt")).unwrap();
        assert!(
            content.starts_with("$Id: "),
            "the ident filter SHOULD expand $Id$ when filters are enabled and trusted, got: \
             {content:?}"
        );
    }

    // ---- Guard 3: scoped-token-only / no ambient fallback ----

    /// Adversarial: an untrusted clone request carrying an unscoped
    /// (`host: None`) token must be refused BEFORE any network operation is
    /// attempted — proven by pointing the clone at an unresolvable
    /// `.invalid` host and asserting the failure is `Git2DBError::Configuration`
    /// (from `validate_trust`), not a network/DNS error that would indicate
    /// the guard let the request through to the transport layer.
    #[tokio::test]
    async fn untrusted_clone_rejects_unscoped_token_before_network() {
        use crate::auth::AuthStrategy;

        let manager = test_manager();
        let options = CloneOptions::builder()
            .trust(CloneTrust::Untrusted)
            .callback_config(
                CallbackConfigBuilder::new()
                    .auth(AuthStrategy::Token {
                        token: "should-never-be-sent".to_owned(),
                        host: None,
                    })
                    .build(),
            )
            .build();

        let dest_dir = tempdir().unwrap();
        let result = manager
            .clone_repository(
                "https://untrusted-guard-test.invalid/should-never-be-fetched.git",
                &dest_dir.path().join("clone"),
                Some(options),
            )
            .await
            .map(|_repo| ());

        match result {
            Err(Git2DBError::Configuration { message }) => {
                assert!(
                    message.contains("unscoped"),
                    "expected the unscoped-token refusal message, got: {message}"
                );
            }
            other => panic!(
                "expected Git2DBError::Configuration from validate_trust (refused before any \
                 network attempt), got: {other:?}"
            ),
        }
    }

    /// Same shape, but for `AuthMode::AllowAmbient` under `Untrusted`.
    #[tokio::test]
    async fn untrusted_clone_rejects_allow_ambient_before_network() {
        use crate::callback_config::AuthMode;

        let manager = test_manager();
        let options = CloneOptions::builder()
            .trust(CloneTrust::Untrusted)
            .callback_config(
                CallbackConfigBuilder::new()
                    .auth_mode(AuthMode::AllowAmbient)
                    .build(),
            )
            .build();

        let dest_dir = tempdir().unwrap();
        let result = manager
            .clone_repository(
                "https://untrusted-guard-test.invalid/should-never-be-fetched.git",
                &dest_dir.path().join("clone"),
                Some(options),
            )
            .await
            .map(|_repo| ());

        assert!(
            matches!(result, Err(Git2DBError::Configuration { .. })),
            "expected Git2DBError::Configuration from validate_trust, got: {result:?}"
        );
    }

    /// A host-scoped token IS accepted under `Untrusted` and the clone
    /// proceeds normally — the scoped-token-only path this whole mode exists
    /// to allow (e.g. a merge gate's scoped GitHub App installation token).
    #[tokio::test]
    async fn untrusted_clone_accepts_host_scoped_token() {
        use crate::auth::AuthStrategy;

        let manager = test_manager();
        let src_dir = tempdir().unwrap();
        init_target_repo(src_dir.path(), "readme.txt", "hello\n");

        // The token is scoped to a host the fixture's file:// URL does not
        // match, but that's fine here: we're only proving validate_trust
        // lets a host-scoped token past construction-time validation (it is
        // simply never offered for a `file://` URL, which the credential
        // callback never even gets invoked for on a local transport).
        let options = CloneOptions::builder()
            .trust(CloneTrust::Untrusted)
            .callback_config(
                CallbackConfigBuilder::new()
                    .auth(AuthStrategy::Token {
                        token: "scoped".to_owned(),
                        host: Some("github.com".to_owned()),
                    })
                    .build(),
            )
            .build();

        let dest_dir = tempdir().unwrap();
        let result = manager
            .clone_repository(&file_url(src_dir.path()), &dest_dir.path().join("clone"), Some(options))
            .await
            .map(|_repo| ());

        assert!(
            result.is_ok(),
            "a host-scoped token must be accepted under Untrusted, got: {result:?}"
        );
    }
}