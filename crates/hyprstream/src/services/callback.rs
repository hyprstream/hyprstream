//! Callback infrastructure for spawned InferenceService instances
//!
//! Implements the callback pattern where spawned services connect BACK to
//! the spawner via DEALER/ROUTER sockets. This eliminates race conditions
//! in service startup.
//!
//! # Architecture
//!
//! ```text
//! ModelService                        InferenceService
//!     │                                     │
//!     ├─ bind ROUTER (callback)             │
//!     ├─ spawn(inference@{id}, --callback)──┤
//!     │                                     ├─ start
//!     │                                     ├─ connect DEALER to callback
//!     │                                     ├─ send Register{id, stream_endpoint}
//!     ├─ recv Register ◄────────────────────┤
//!     │   (bidirectional connection ready)  │
//!     ├─ send LoadModel ────────────────────┤
//!     │                                     ├─ load model
//!     ├─ recv LoadModelResponse ◄───────────┤
//!     ▼                                     ▼
//! ```

use crate::model_capnp;
use anyhow::{anyhow, Result};
use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use hyprstream_rpc::registry::{try_global as try_registry, SocketKind};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, trace, warn};

/// Fallback callback endpoint when registry is not initialized
const FALLBACK_CALLBACK_ENDPOINT: &str = "ipc:///run/hyprstream/model-callback.sock";

/// Timeout for waiting for callback registration
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(30);

/// Information about a registered inference instance
#[derive(Debug, Clone)]
pub struct Instance {
    /// Instance ID (e.g., "inference-a1b2c3d4")
    pub id: String,
    /// DEALER identity for routing messages
    pub identity: Vec<u8>,
    /// XPUB endpoint for token streaming
    pub stream_endpoint: String,
    /// Loaded model reference (None if not loaded)
    pub model: Option<String>,
    /// Last time this instance was used
    pub last_used: Instant,
}

/// Callback router for managing InferenceService connections
///
/// Binds a ROUTER socket that InferenceService instances connect to.
/// Each instance sends a Register message with its ID and stream endpoint.
///
/// Thread-safe: socket operations are protected by a Mutex.
pub struct CallbackRouter {
    /// ZMQ context
    #[allow(dead_code)]
    ctx: Arc<zmq::Context>,
    /// ROUTER socket for callbacks (Mutex for thread safety)
    router: Mutex<zmq::Socket>,
    /// Callback endpoint
    endpoint: String,
    /// Registered instances by ID
    instances: RwLock<HashMap<String, Instance>>,
}

impl CallbackRouter {
    /// Create and bind a new callback router using registry endpoint.
    ///
    /// Uses the endpoint registered for "model" service with ROUTER socket type.
    /// Falls back to the default IPC endpoint if registry is not initialized.
    pub fn new(ctx: Arc<zmq::Context>) -> Result<Self> {
        let endpoint = match try_registry() {
            Some(reg) => reg.endpoint("model", SocketKind::Router).to_zmq_string(),
            None => FALLBACK_CALLBACK_ENDPOINT.to_owned(),
        };
        Self::with_endpoint(ctx, &endpoint)
    }

    /// Create with a custom endpoint.
    ///
    /// Use this when you need to override the registry-provided endpoint,
    /// or for testing with custom endpoints.
    pub fn with_endpoint(ctx: Arc<zmq::Context>, endpoint: &str) -> Result<Self> {
        let router = ctx.socket(zmq::ROUTER)?;

        // Set socket options before binding
        router.set_rcvtimeo(100)?; // 100ms recv timeout for polling
        router.set_router_mandatory(true)?; // Fail if routing fails

        router.bind(endpoint)?;
        info!("CallbackRouter bound to {}", endpoint);

        Ok(Self {
            ctx,
            router: Mutex::new(router),
            endpoint: endpoint.to_owned(),
            instances: RwLock::new(HashMap::new()),
        })
    }

    /// Get the callback endpoint
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Wait for a registration from a specific instance ID
    ///
    /// Blocks until the instance registers or timeout expires.
    pub fn wait_for_register(&self, expected_id: &str) -> Result<Instance> {
        let start = Instant::now();
        let router = self.router.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;

        while start.elapsed() < CALLBACK_TIMEOUT {
            // Try to receive a message
            match router.recv_multipart(0) {
                Ok(parts) => {
                    if parts.len() < 3 {
                        warn!("Malformed callback message: {} parts", parts.len());
                        continue;
                    }

                    // ROUTER format: [identity, empty, payload]
                    let identity = parts[0].clone();
                    let payload = &parts[2];

                    // Try to parse as Register message
                    match self.parse_register(payload) {
                        Ok((id, stream_endpoint)) => {
                            if id == expected_id {
                                let instance = Instance {
                                    id: id.clone(),
                                    identity,
                                    stream_endpoint,
                                    model: None,
                                    last_used: Instant::now(),
                                };

                                info!(
                                    "Instance {} registered with stream_endpoint {}",
                                    id, instance.stream_endpoint
                                );

                                return Ok(instance);
                            } else {
                                // Store for later - another spawn in progress
                                debug!("Received registration for {} while waiting for {}", id, expected_id);
                                let mut instances = futures::executor::block_on(self.instances.write());
                                instances.insert(id.clone(), Instance {
                                    id,
                                    identity,
                                    stream_endpoint,
                                    model: None,
                                    last_used: Instant::now(),
                                });
                            }
                        }
                        Err(e) => {
                            warn!("Failed to parse register message: {}", e);
                        }
                    }
                }
                Err(zmq::Error::EAGAIN) => {
                    // Timeout on recv, continue waiting
                    continue;
                }
                Err(e) => {
                    return Err(anyhow!("Callback recv error: {}", e));
                }
            }
        }

        Err(anyhow!(
            "Timeout waiting for registration from {}",
            expected_id
        ))
    }

    /// Send a LoadModel command to an instance
    pub fn send_load_model(&self, instance: &Instance, model_ref: &str, model_path: &str) -> Result<()> {
        let payload = self.build_load_model(model_ref, model_path)?;
        let empty: Vec<u8> = Vec::new();
        let router = self.router.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;

        // ROUTER format: [identity, empty, payload]
        router.send_multipart([&instance.identity, &empty, &payload], 0)?;
        trace!("Sent LoadModel to {}", instance.id);

        Ok(())
    }

    /// Receive LoadModelResponse from an instance
    pub fn recv_load_model_response(&self, instance: &Instance) -> Result<()> {
        // Wait for response with timeout
        let start = Instant::now();
        let timeout = Duration::from_secs(60); // Model loading can take time
        let router = self.router.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;

        while start.elapsed() < timeout {
            match router.recv_multipart(0) {
                Ok(parts) => {
                    if parts.len() < 3 {
                        continue;
                    }

                    let recv_identity = &parts[0];
                    if recv_identity != &instance.identity {
                        // Message from different instance, ignore for now
                        continue;
                    }

                    let payload = &parts[2];
                    return self.parse_load_model_response(payload);
                }
                Err(zmq::Error::EAGAIN) => {
                    continue;
                }
                Err(e) => {
                    return Err(anyhow!("Recv error: {}", e));
                }
            }
        }

        Err(anyhow!("Timeout waiting for LoadModelResponse from {}", instance.id))
    }

    /// Send a shutdown command to an instance
    pub fn send_shutdown(&self, instance: &Instance) -> Result<()> {
        let payload = self.build_shutdown()?;
        let empty: Vec<u8> = Vec::new();
        let router = self.router.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;
        router.send_multipart([&instance.identity, &empty, &payload], 0)?;
        trace!("Sent Shutdown to {}", instance.id);
        Ok(())
    }

    /// Send an inference command to an instance (returns serialized InferenceRequest)
    pub fn send_infer(&self, instance: &Instance, request_bytes: &[u8]) -> Result<()> {
        let payload = self.build_infer(request_bytes)?;
        let empty: Vec<u8> = Vec::new();
        let router = self.router.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;
        router.send_multipart([&instance.identity, &empty, &payload], 0)?;
        trace!("Sent Infer to {}", instance.id);
        Ok(())
    }

    /// Receive inference response
    pub fn recv_infer_response(&self, instance: &Instance, timeout: Duration) -> Result<Vec<u8>> {
        let start = Instant::now();
        let router = self.router.lock().map_err(|e| anyhow!("Lock poisoned: {}", e))?;

        while start.elapsed() < timeout {
            match router.recv_multipart(0) {
                Ok(parts) => {
                    if parts.len() < 3 {
                        continue;
                    }

                    let recv_identity = &parts[0];
                    if recv_identity != &instance.identity {
                        continue;
                    }

                    // Return raw response bytes (InferenceResponse)
                    return Ok(parts[2].clone());
                }
                Err(zmq::Error::EAGAIN) => {
                    continue;
                }
                Err(e) => {
                    return Err(anyhow!("Recv error: {}", e));
                }
            }
        }

        Err(anyhow!("Timeout waiting for InferResponse from {}", instance.id))
    }

    // ========================================================================
    // Message parsing/building
    // ========================================================================

    fn parse_register(&self, payload: &[u8]) -> Result<(String, String)> {
        let reader = serialize::read_message(
            &mut std::io::Cursor::new(payload),
            ReaderOptions::new(),
        )?;
        let reg = reader.get_root::<model_capnp::register::Reader>()?;

        let id = reg.get_id()?.to_str()?.to_owned();
        let stream_endpoint = reg.get_stream_endpoint()?.to_str()?.to_owned();

        Ok((id, stream_endpoint))
    }

    fn build_load_model(&self, model_ref: &str, model_path: &str) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let cmd = message.init_root::<model_capnp::inference_command::Builder>();
            let mut load = cmd.init_load_model();
            load.set_model_ref(model_ref);
            load.set_model_path(model_path);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn parse_load_model_response(&self, payload: &[u8]) -> Result<()> {
        let reader = serialize::read_message(
            &mut std::io::Cursor::new(payload),
            ReaderOptions::new(),
        )?;
        let resp = reader.get_root::<model_capnp::load_model_command_response::Reader>()?;

        if resp.get_success() {
            Ok(())
        } else {
            Err(anyhow!("LoadModel failed: {}", resp.get_error()?.to_str()?))
        }
    }

    fn build_shutdown(&self) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut cmd = message.init_root::<model_capnp::inference_command::Builder>();
            cmd.set_shutdown(());
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }

    fn build_infer(&self, request_bytes: &[u8]) -> Result<Vec<u8>> {
        let mut message = Builder::new_default();
        {
            let mut cmd = message.init_root::<model_capnp::inference_command::Builder>();
            cmd.set_infer(request_bytes);
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message)?;
        Ok(bytes)
    }
}

impl Drop for CallbackRouter {
    fn drop(&mut self) {
        // Shutdown all instances
        let instances = futures::executor::block_on(self.instances.read());
        for instance in instances.values() {
            if let Err(e) = self.send_shutdown(instance) {
                warn!("Failed to send shutdown to {}: {}", instance.id, e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_callback_endpoint() {
        assert!(FALLBACK_CALLBACK_ENDPOINT.starts_with("ipc://"));
    }
}
