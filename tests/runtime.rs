use std::sync::Arc;
use std::time::Duration;
use async_trait::async_trait;
use hyprstream::{
    runtime::{AsyncRuntime, RuntimeBuilder, ConnectionManager},
    Result, Error,
};

#[derive(Debug)]
struct MockConnection {
    is_valid: bool,
}

#[derive(Debug)]
struct MockConnectionManager {
    fail_connect: bool,
}

#[async_trait]
impl ConnectionManager for MockConnectionManager {
    type Connection = MockConnection;

    async fn connect(&self) -> Result<Self::Connection> {
        if self.fail_connect {
            Err(Error::RuntimeError("Failed to connect".into()))
        } else {
            Ok(MockConnection { is_valid: true })
        }
    }

    async fn is_valid(&self, conn: &mut Self::Connection) -> Result<()> {
        if conn.is_valid {
            Ok(())
        } else {
            Err(Error::RuntimeError("Invalid connection".into()))
        }
    }

    fn has_broken(&self, conn: &mut Self::Connection) -> bool {
        !conn.is_valid
    }
}

#[tokio::test]
async fn test_runtime_configuration() {
    let runtime = RuntimeBuilder::new()
        .worker_threads(2)
        .thread_name("test-worker")
        .build()
        .unwrap();

    assert!(Arc::strong_count(&runtime.runtime()) == 1);
}

#[tokio::test]
async fn test_connection_pool() {
    let runtime = AsyncRuntime::new().unwrap();
    let manager = MockConnectionManager { fail_connect: false };
    
    let pool = runtime.create_pool(manager).await.unwrap();
    let conn = pool.get().await.unwrap();
    
    assert!(conn.is_valid);
}

#[tokio::test]
async fn test_graceful_shutdown() {
    let mut runtime = AsyncRuntime::new().unwrap();
    let shutdown_complete = Arc::new(tokio::sync::Notify::new());
    let shutdown_complete_clone = shutdown_complete.clone();

    let handle = runtime.run_with_shutdown(|rx| async move {
        tokio::select! {
            _ = rx => {
                shutdown_complete_clone.notify_one();
                Ok(())
            }
            _ = tokio::time::sleep(Duration::from_secs(60)) => {
                Ok(())
            }
        }
    });

    // Trigger shutdown after small delay
    tokio::time::sleep(Duration::from_millis(100)).await;
    runtime.shutdown().await.unwrap();

    // Wait for shutdown to complete
    shutdown_complete.notified().await;
    handle.await.unwrap();
}

#[tokio::test]
async fn test_pool_health_checks() {
    let runtime = AsyncRuntime::new().unwrap();
    let manager = MockConnectionManager { fail_connect: false };
    
    let pool = runtime.create_pool(manager).await.unwrap();
    
    // Get connection and verify it's valid
    let mut conn = pool.get().await.unwrap();
    conn.is_valid = false;
    
    // Connection should be removed from pool on return due to health check
    drop(conn);
    
    // Next connection should be new and valid
    let conn = pool.get().await.unwrap();
    assert!(conn.is_valid);
}
