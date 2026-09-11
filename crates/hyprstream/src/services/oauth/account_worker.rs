//! Blocking account I/O is contained by a dedicated OAuth process.
//!
//! A Rust thread cannot be safely killed while inside filesystem/audit I/O.
//! Keep its actual handle and join completion; an overdue or abandoned worker
//! terminates the dedicated process instead of detaching a live listener. The
//! OAuth entrypoint checks launcher-provided containment before creating one.

use std::{thread::JoinHandle, time::Duration};

use hyprstream_rpc::error::RpcError;
use tokio::sync::oneshot;

pub(super) const WORKER_FAILURE_EXIT: i32 = 70;

fn terminate(name: &str) -> ! {
    tracing::error!(
        worker = name,
        "account worker did not stop; terminating dedicated OAuth process"
    );
    std::process::exit(WORKER_FAILURE_EXIT);
}

pub(super) struct ContainedWorker<T> {
    name: &'static str,
    thread: Option<JoinHandle<()>>,
    completion: oneshot::Receiver<T>,
}

impl<T: Send + 'static> ContainedWorker<T> {
    pub(super) fn spawn(
        name: &'static str,
        work: impl FnOnce() -> T + Send + 'static,
    ) -> Result<Self, RpcError> {
        let (tx, completion) = oneshot::channel();
        let thread = std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                let result = work();
                let _ = tx.send(result);
            })
            .map_err(|error| RpcError::SpawnFailed(format!("{name} thread spawn: {error}")))?;
        Ok(Self {
            name,
            thread: Some(thread),
            completion,
        })
    }

    pub(super) fn is_joined(&self) -> bool {
        self.thread.is_none()
    }

    pub(super) async fn result(&mut self) -> Result<T, RpcError> {
        let result = (&mut self.completion).await;
        // The sender runs after work (including runtime destruction) returns.
        // Its only remaining operation is returning from the thread closure.
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| RpcError::SpawnFailed(format!("{} thread panicked", self.name,)))?;
        }
        result.map_err(|_| {
            RpcError::SpawnFailed(format!("{} thread exited without a result", self.name,))
        })
    }

    pub(super) async fn finish(&mut self, timeout: Duration) -> Result<T, RpcError> {
        match tokio::time::timeout(timeout, self.result()).await {
            Ok(result) => result,
            Err(_) => terminate(self.name),
        }
    }
}

impl<T> Drop for ContainedWorker<T> {
    fn drop(&mut self) {
        if let Some(thread) = self.thread.take() {
            if !thread.is_finished() {
                // Covers cancellation/unwind before common teardown as well.
                terminate(self.name);
            }
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
pub(super) mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use std::{net::TcpListener, process::Command, time::Instant};

    pub(crate) fn is_child(test: &str) -> bool {
        std::env::var("HYPRSTREAM_CONTAINED_WORKER_TEST").as_deref() == Ok(test)
    }

    pub(crate) fn expect_contained_exit(test: &str, env: &[(&str, String)]) {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", test, "--nocapture"])
            .env("HYPRSTREAM_CONTAINED_WORKER_TEST", test);
        for (key, value) in env {
            command.env(key, value);
        }
        let mut child = command.spawn().unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert_eq!(status.code(), Some(WORKER_FAILURE_EXIT));
                break;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                panic!("contained worker process failed to exit within deadline");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[tokio::test]
    async fn completed_worker_is_joined() {
        let mut worker = ContainedWorker::spawn("test-completes", || 42).unwrap();
        assert_eq!(worker.finish(Duration::from_secs(1)).await.unwrap(), 42);
        assert!(worker.is_joined());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn early_shutdown_joins_real_listener_and_releases_port() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = listener.local_addr().unwrap();
        let shutdown = std::sync::Arc::new(tokio::sync::Notify::new());
        // Signal before the worker registers: the permit must survive.
        shutdown.notify_one();
        let mut worker = ContainedWorker::spawn("test-listener-stop", move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let bound = crate::server::tls::BoundHttpListener::Http(
                    tokio::net::TcpListener::from_std(listener).unwrap(),
                );
                crate::server::tls::serve_bound(
                    bound,
                    axum::Router::new(),
                    shutdown,
                    "AccountHttpTest",
                )
                .await
            })
        })
        .unwrap();
        worker
            .finish(Duration::from_secs(2))
            .await
            .unwrap()
            .unwrap();
        assert!(worker.is_joined());
        TcpListener::bind(addr).expect("joined listener must release its socket");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn listener_timeout_terminates_process_and_releases_port() {
        const TEST: &str = "services::oauth::account_worker::tests::listener_timeout_terminates_process_and_releases_port";
        if !is_child(TEST) {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let addr = listener.local_addr().unwrap();
            drop(listener);
            expect_contained_exit(
                TEST,
                &[("HYPRSTREAM_CONTAINED_WORKER_ADDR", addr.to_string())],
            );
            TcpListener::bind(addr).expect("terminated worker must release its listening socket");
            return;
        }
        let addr = std::env::var("HYPRSTREAM_CONTAINED_WORKER_ADDR").unwrap();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let mut worker = ContainedWorker::spawn("test-stuck-listener", move || {
            let _listener = TcpListener::bind(addr).unwrap();
            ready_tx.send(()).unwrap();
            loop {
                std::thread::park();
            }
        })
        .unwrap();
        ready_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let _ = worker.finish(Duration::from_millis(20)).await;
        panic!("a live timed-out listener must not be detached");
    }

    #[test]
    fn cancellation_cannot_detach_live_worker() {
        const TEST: &str =
            "services::oauth::account_worker::tests::cancellation_cannot_detach_live_worker";
        if !is_child(TEST) {
            expect_contained_exit(TEST, &[]);
            return;
        }
        let worker = ContainedWorker::spawn("test-cancelled-worker", || {
            loop {
                std::thread::park();
            }
        })
        .unwrap();
        drop(worker);
        panic!("dropping a live worker must terminate its process");
    }
}
