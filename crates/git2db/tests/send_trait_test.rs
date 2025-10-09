//! Tests demonstrating the Send-safe clone API
//!
//! This test file verifies that our API properly handles the Send trait
//! requirement for async futures while maintaining callback flexibility.

use git2db::{
    auth::AuthStrategy,
    callback_config::{CallbackConfigBuilder, ProgressConfig},
    clone_options::{CloneOptions, CloneOptionsBuilder},
    manager::GitManager,
};
use std::sync::Arc;
use tempfile::TempDir;

/// This test verifies that CloneOptions is Send
#[test]
fn test_clone_options_is_send() {
    fn assert_send<T: Send>() {}

    // Create options with callback configuration
    let _options = CloneOptionsBuilder::new()
        .callback_config(
            CallbackConfigBuilder::new()
                .auth(AuthStrategy::SshAgent {
                    username: Some("git".to_string()),
                })
                .progress(ProgressConfig::Stdout)
                .build(),
        )
        .shallow(true)
        .depth(1)
        .build();

    // This should compile - CloneOptions is Send
    assert_send::<CloneOptions>();
}

/// Custom progress reporter that is Send + Sync
struct TestProgressReporter {
    counter: Arc<parking_lot::Mutex<usize>>,
}

impl git2db::callback_config::ProgressReporter for TestProgressReporter {
    fn report(&self, stage: &str, current: usize, total: usize) {
        let mut counter = self.counter.lock();
        *counter += 1;
        println!("Progress [{}]: {}/{}", stage, current, total);
    }
}

/// This test demonstrates using the API with a custom progress reporter
#[tokio::test]
async fn test_clone_with_send_safe_progress() {
    let temp_dir = TempDir::new().unwrap();
    let target_path = temp_dir.path().join("test_repo");

    // Create a custom progress reporter (Send + Sync)
    let counter = Arc::new(parking_lot::Mutex::new(0));
    let reporter = Arc::new(TestProgressReporter {
        counter: counter.clone(),
    });

    // Create Send-safe options
    let options = CloneOptionsBuilder::new()
        .callback_config(
            CallbackConfigBuilder::new()
                .auth(AuthStrategy::Default)
                .progress(ProgressConfig::Channel(reporter))
                .build(),
        )
        .shallow(true)
        .depth(1)
        .build();

    // Initialize GitManager
    let manager = GitManager::global();

    // This compiles and works - the future is Send
    let _result = manager
        .clone_repository(
            "https://github.com/octocat/Hello-World.git",
            &target_path,
            Some(options),
        )
        .await;

    // Verify progress was reported
    let count = *counter.lock();
    assert!(count > 0, "Progress should have been reported");
}

/// This test verifies that the async future from clone_repository is Send
#[tokio::test]
async fn test_clone_future_is_send() {
    fn assert_future_send<F: std::future::Future + Send>(_f: F) {}

    let manager = GitManager::global();
    let temp_dir = TempDir::new().unwrap();
    let target_path = temp_dir.path().join("test_repo");

    let options = CloneOptionsBuilder::new()
        .callback_config(
            CallbackConfigBuilder::new()
                .auth(AuthStrategy::Default)
                .build(),
        )
        .build();

    // Get the future
    let future = manager.clone_repository(
        "https://github.com/octocat/Hello-World.git",
        &target_path,
        Some(options),
    );

    // This should compile - the future is Send
    assert_future_send(future);
}

/// Test that we can use the API from multiple async tasks
#[tokio::test]
async fn test_concurrent_clones_with_send() {
    let _manager = GitManager::global();

    // Spawn multiple tasks that all use the API
    let handles: Vec<_> = (0..3)
        .map(|i| {
            tokio::spawn(async move {
                let temp_dir = TempDir::new().unwrap();
                let target_path = temp_dir.path().join(format!("repo_{}", i));

                let options = CloneOptionsBuilder::new()
                    .callback_config(
                        CallbackConfigBuilder::new()
                            .auth(AuthStrategy::Default)
                            .progress(ProgressConfig::Stdout)
                            .build(),
                    )
                    .shallow(true)
                    .build();

                // This works because the future is Send
                let result = GitManager::global()
                    .clone_repository(
                        "https://github.com/octocat/Hello-World.git",
                        &target_path,
                        Some(options),
                    )
                    .await;

                result.is_ok()
            })
        })
        .collect();

    // All tasks should complete successfully
    for handle in handles {
        let success = handle.await.unwrap();
        assert!(success, "Clone should succeed in async task");
    }
}

/// Test that CallbackConfig properly bridges authentication
#[test]
fn test_callback_config_auth_strategies() {
    let config = CallbackConfigBuilder::new()
        .auth(AuthStrategy::SshAgent {
            username: Some("git".to_string()),
        })
        .auth(AuthStrategy::Token {
            token: "test_token".to_string(),
        })
        .auth(AuthStrategy::Default)
        .build();

    // Verify we have 3 auth strategies
    assert_eq!(config.auth.len(), 3);

    // Create callbacks (this would normally happen inside spawn_blocking)
    let callbacks = config.create_callbacks();

    // The callbacks are created successfully
    // In real usage, git2 would call these during clone operations
    drop(callbacks);
}
