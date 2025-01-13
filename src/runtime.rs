use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::{Builder, Runtime};
use tokio::signal;
use tokio::sync::oneshot;
use bb8::Pool;
use async_trait::async_trait;

use crate::error::{Error, Result};

/// Configuration for the runtime builder
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Number of worker threads (defaults to number of CPUs)
    pub worker_threads: Option<usize>,
    /// Thread name prefix
    pub thread_name: String,
    /// Thread stack size in bytes
    pub thread_stack_size: Option<usize>,
    /// Maximum number of concurrent connections
    pub max_connections: u32,
    /// Connection timeout duration
    pub connection_timeout: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            worker_threads: None,
            thread_name: "hyprstream-worker".to_string(),
            thread_stack_size: None,
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(60),
        }
    }
}

/// Trait for connection management
#[async_trait]
pub trait ConnectionManager: Send + Sync + 'static {
    type Connection: Send + 'static;
    
    async fn connect(&self) -> Result<Self::Connection>;
    async fn is_valid(&self, conn: &mut Self::Connection) -> Result<()>;
    fn has_broken(&self, conn: &mut Self::Connection) -> bool;
}

/// Runtime builder for configuring the async runtime
pub struct RuntimeBuilder {
    config: RuntimeConfig,
}

impl RuntimeBuilder {
    pub fn new() -> Self {
        Self {
            config: RuntimeConfig::default(),
        }
    }

    pub fn worker_threads(mut self, count: usize) -> Self {
        self.config.worker_threads = Some(count);
        self
    }

    pub fn thread_name(mut self, name: impl Into<String>) -> Self {
        self.config.thread_name = name.into();
        self
    }

    pub fn thread_stack_size(mut self, size: usize) -> Self {
        self.config.thread_stack_size = Some(size);
        self
    }

    pub fn build(self) -> Result<AsyncRuntime> {
        AsyncRuntime::with_config(self.config)
    }
}

/// Main async runtime wrapper
pub struct AsyncRuntime {
    runtime: Arc<Runtime>,
    config: RuntimeConfig,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl AsyncRuntime {
    pub fn new() -> Result<Self> {
        Self::with_config(RuntimeConfig::default())
    }

    pub fn with_config(config: RuntimeConfig) -> Result<Self> {
        let mut builder = Builder::new_multi_thread();
        
        builder.enable_all()
            .thread_name(&config.thread_name);

        if let Some(threads) = config.worker_threads {
            builder.worker_threads(threads);
        }

        if let Some(stack_size) = config.thread_stack_size {
            builder.thread_stack_size(stack_size);
        }

        let runtime = builder.build()
            .map_err(|e| Error::Internal(format!("Failed to build runtime: {}", e)))?;

        Ok(Self {
            runtime: Arc::new(runtime),
            config,
            shutdown_tx: None,
        })
    }

    pub fn runtime(&self) -> Arc<Runtime> {
        self.runtime.clone()
    }

    pub async fn create_pool<M>(&self, manager: M) -> Result<Pool<M>> 
    where
        M: ConnectionManager + bb8::ManageConnection,
    {
        let pool = Pool::builder()
            .max_size(self.config.max_connections)
            .connection_timeout(self.config.connection_timeout)
            .test_on_check_out(true)
            .build(manager)
            .await
            .map_err(|e| Error::Internal(format!("Failed to create connection pool: {:?}", e)))?;

        Ok(pool)
    }

    pub async fn run_with_shutdown<F, Fut>(&mut self, f: F) -> Result<()>
    where
        F: FnOnce(oneshot::Receiver<()>) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        let (tx, rx) = oneshot::channel();
        self.shutdown_tx = Some(tx);

        // Handle shutdown signals
        let shutdown_future = async {
            match signal::ctrl_c().await {
                Ok(()) => Ok(()),
                Err(err) => Err(Error::Runtime(format!("Failed to listen for ctrl-c: {}", err))),
            }
        };

        tokio::select! {
            result = f(rx) => result,
            result = shutdown_future => result,
        }
    }

    pub async fn shutdown(&mut self) -> Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_runtime_builder() {
        let runtime = RuntimeBuilder::new()
            .worker_threads(2)
            .thread_name("test-worker")
            .thread_stack_size(3 * 1024 * 1024)
            .build()
            .unwrap();

        assert!(matches!(runtime.runtime().handle().runtime_flavor(), tokio::runtime::RuntimeFlavor::MultiThread));
    }

    #[tokio::test]
    async fn test_shutdown_signal() {
        let mut runtime = AsyncRuntime::new().unwrap();
        
        let handle = runtime.run_with_shutdown(|rx| async move {
            tokio::select! {
                _ = rx => Ok(()),
                _ = tokio::time::sleep(Duration::from_secs(1)) => Ok(()),
            }
        });

        runtime.shutdown().await.unwrap();
        handle.await.unwrap();
    }
}
