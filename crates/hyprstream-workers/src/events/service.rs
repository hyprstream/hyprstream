//! EventService - XPUB/XSUB proxy for event distribution
//!
//! Uses `zmq::proxy_steerable()` for efficient C-level message forwarding.
//! Publishers connect to XSUB, subscribers connect to XPUB.

use anyhow::{anyhow, Result};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use tracing::{error, info};

use super::endpoints;

/// Handle for controlling a running EventService
pub struct EventServiceHandle {
    thread: Option<JoinHandle<()>>,
    ctx: Arc<zmq::Context>,
}

impl EventServiceHandle {
    /// Stop the event service gracefully
    ///
    /// Sends TERMINATE to the control socket and waits for the proxy thread to exit.
    pub fn stop(mut self) -> Result<()> {
        let ctrl = self.ctx.socket(zmq::PAIR)?;
        ctrl.connect(endpoints::CTRL)?;
        ctrl.send("TERMINATE", 0)?;

        if let Some(handle) = self.thread.take() {
            handle.join().map_err(|_| anyhow!("Proxy thread panicked"))?;
        }
        info!("EventService stopped");
        Ok(())
    }

    /// Check if the proxy thread is still running
    pub fn is_running(&self) -> bool {
        self.thread.as_ref().map(|h| !h.is_finished()).unwrap_or(false)
    }
}

impl Drop for EventServiceHandle {
    fn drop(&mut self) {
        if self.thread.as_ref().map(|h| !h.is_finished()).unwrap_or(false) {
            // Try graceful shutdown
            if let Ok(ctrl) = self.ctx.socket(zmq::PAIR) {
                if ctrl.connect(endpoints::CTRL).is_ok() {
                    let _ = ctrl.send("TERMINATE", 0);
                }
            }
            // Wait for thread
            if let Some(handle) = self.thread.take() {
                let _ = handle.join();
            }
        }
    }
}

/// Start the EventService proxy
///
/// Spawns a dedicated thread running `zmq::proxy_steerable()` for efficient
/// C-level message forwarding between publishers (XSUB) and subscribers (XPUB).
///
/// # Arguments
///
/// * `ctx` - ZMQ context (must be shared for inproc:// to work)
///
/// # Returns
///
/// Handle for controlling the service. Call `handle.stop()` for graceful shutdown.
///
/// # Example
///
/// ```ignore
/// let ctx = global_context();
/// let handle = start_event_service(ctx.clone())?;
///
/// // ... use event bus ...
///
/// handle.stop()?;
/// ```
pub fn start_event_service(ctx: Arc<zmq::Context>) -> Result<EventServiceHandle> {
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<()>>();
    let ctx_clone = ctx.clone();

    let thread = thread::spawn(move || {
        // Create sockets in this thread (ZMQ sockets are !Send)
        let mut xsub = match ctx_clone.socket(zmq::XSUB) {
            Ok(s) => s,
            Err(e) => {
                let _ = ready_tx.send(Err(anyhow!("Failed to create XSUB socket: {}", e)));
                return;
            }
        };
        let mut xpub = match ctx_clone.socket(zmq::XPUB) {
            Ok(s) => s,
            Err(e) => {
                let _ = ready_tx.send(Err(anyhow!("Failed to create XPUB socket: {}", e)));
                return;
            }
        };
        let mut ctrl = match ctx_clone.socket(zmq::PAIR) {
            Ok(s) => s,
            Err(e) => {
                let _ = ready_tx.send(Err(anyhow!("Failed to create PAIR socket: {}", e)));
                return;
            }
        };

        // Bind sockets
        if let Err(e) = xsub.bind(endpoints::PUB) {
            let _ = ready_tx.send(Err(anyhow!("Failed to bind XSUB to {}: {}", endpoints::PUB, e)));
            return;
        }
        if let Err(e) = xpub.bind(endpoints::SUB) {
            let _ = ready_tx.send(Err(anyhow!("Failed to bind XPUB to {}: {}", endpoints::SUB, e)));
            return;
        }
        if let Err(e) = ctrl.bind(endpoints::CTRL) {
            let _ = ready_tx.send(Err(anyhow!("Failed to bind CTRL to {}: {}", endpoints::CTRL, e)));
            return;
        }

        // Signal ready
        let _ = ready_tx.send(Ok(()));

        // Run C-level proxy (blocks until TERMINATE received on ctrl socket)
        if let Err(e) = zmq::proxy_steerable(&mut xsub, &mut xpub, &mut ctrl) {
            // ETERM is expected on shutdown
            if e != zmq::Error::ETERM {
                error!("Proxy error: {}", e);
            }
        }
    });

    // Wait for bind success/failure
    ready_rx
        .recv()
        .map_err(|_| anyhow!("Proxy thread exited before signaling ready"))??;

    info!("EventService started");
    Ok(EventServiceHandle {
        thread: Some(thread),
        ctx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_start_stop() {
        let ctx = Arc::new(zmq::Context::new());
        let handle = start_event_service(ctx).expect("Failed to start EventService");
        assert!(handle.is_running());
        handle.stop().expect("Failed to stop EventService");
    }
}
