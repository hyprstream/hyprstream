//! Centralized endpoint registry for service discovery.
//!
//! This module provides:
//! - `EndpointMode` for determining transport type (inproc, ipc, tcp)
//! - `SocketKind` for identifying ZMQ socket types (implements Hash)
//! - `EndpointRegistry` for service endpoint management (supports multiple socket types)
//! - `ServiceRegistration` RAII guard for automatic cleanup
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_rpc::registry::{init, global, EndpointMode, SocketKind, ServiceRegistration};
//!
//! // Initialize at startup
//! init(EndpointMode::Ipc, Some(runtime_dir));
//!
//! // Services self-register (single endpoint, backward compatible)
//! let _reg = ServiceRegistration::new("policy", endpoint, None)?;
//!
//! // Services with multiple endpoints
//! let _reg = ServiceRegistration::multi("model", vec![
//!     (SocketKind::Rep, rep_endpoint),
//!     (SocketKind::Router, router_endpoint),
//! ], None)?;
//!
//! // Clients discover endpoints by socket type
//! let rep_endpoint = global().endpoint("model", SocketKind::Rep);
//! let router_endpoint = global().endpoint("model", SocketKind::Router);
//! ```

use crate::transport::TransportConfig;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;

/// ZMQ socket type identifier for endpoint registration.
///
/// This enum mirrors `zmq::SocketType` but implements `Hash` for use as HashMap keys.
/// Use `.into()` to convert to `zmq::SocketType` when needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SocketKind {
    /// REQ socket (client request)
    Req,
    /// REP socket (server reply)
    Rep,
    /// DEALER socket (async request)
    Dealer,
    /// ROUTER socket (async reply with routing)
    Router,
    /// PUB socket (publish)
    Pub,
    /// SUB socket (subscribe)
    Sub,
    /// XPUB socket (extended publish with subscriptions)
    XPub,
    /// XSUB socket (extended subscribe)
    XSub,
    /// PUSH socket (pipeline push)
    Push,
    /// PULL socket (pipeline pull)
    Pull,
    /// PAIR socket (exclusive pair)
    Pair,
    /// STREAM socket (raw TCP)
    Stream,
}

impl SocketKind {
    /// Get the endpoint suffix for this socket type.
    pub fn suffix(&self) -> &'static str {
        match self {
            SocketKind::Rep => "",
            SocketKind::Router => "-router",
            SocketKind::Pub => "-pub",
            SocketKind::XPub => "-xpub",
            SocketKind::Sub => "-sub",
            SocketKind::XSub => "-xsub",
            SocketKind::Dealer => "-dealer",
            SocketKind::Req => "-req",
            SocketKind::Push => "-push",
            SocketKind::Pull => "-pull",
            SocketKind::Pair => "-pair",
            SocketKind::Stream => "-stream",
        }
    }
}

impl From<SocketKind> for zmq::SocketType {
    fn from(kind: SocketKind) -> Self {
        match kind {
            SocketKind::Req => zmq::SocketType::REQ,
            SocketKind::Rep => zmq::SocketType::REP,
            SocketKind::Dealer => zmq::SocketType::DEALER,
            SocketKind::Router => zmq::SocketType::ROUTER,
            SocketKind::Pub => zmq::SocketType::PUB,
            SocketKind::Sub => zmq::SocketType::SUB,
            SocketKind::XPub => zmq::SocketType::XPUB,
            SocketKind::XSub => zmq::SocketType::XSUB,
            SocketKind::Push => zmq::SocketType::PUSH,
            SocketKind::Pull => zmq::SocketType::PULL,
            SocketKind::Pair => zmq::SocketType::PAIR,
            SocketKind::Stream => zmq::SocketType::STREAM,
        }
    }
}

impl From<zmq::SocketType> for SocketKind {
    fn from(st: zmq::SocketType) -> Self {
        match st {
            zmq::SocketType::REQ => SocketKind::Req,
            zmq::SocketType::REP => SocketKind::Rep,
            zmq::SocketType::DEALER => SocketKind::Dealer,
            zmq::SocketType::ROUTER => SocketKind::Router,
            zmq::SocketType::PUB => SocketKind::Pub,
            zmq::SocketType::SUB => SocketKind::Sub,
            zmq::SocketType::XPUB => SocketKind::XPub,
            zmq::SocketType::XSUB => SocketKind::XSub,
            zmq::SocketType::PUSH => SocketKind::Push,
            zmq::SocketType::PULL => SocketKind::Pull,
            zmq::SocketType::PAIR => SocketKind::Pair,
            zmq::SocketType::STREAM => SocketKind::Stream,
        }
    }
}

/// Endpoint mode determines default transport type.
///
/// - `Inproc`: In-process ZMQ endpoints (daemon mode, single process)
/// - `Ipc`: Unix domain sockets (systemd mode, auto-detects socket activation)
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum EndpointMode {
    /// In-process ZMQ (daemon mode)
    #[default]
    Inproc,
    /// Unix domain sockets (systemd mode)
    ///
    /// When running under systemd socket activation (LISTEN_FDS set),
    /// automatically uses the systemd-provided file descriptor.
    /// Otherwise, binds to a new Unix domain socket.
    Ipc,
}

/// Service registration entry with multiple endpoints by socket type.
#[derive(Debug, Clone)]
pub struct ServiceEntry {
    /// Service name
    pub name: String,
    /// Endpoints by socket type
    pub endpoints: HashMap<SocketKind, TransportConfig>,
    /// Optional description
    pub description: Option<String>,
    /// Raw `.capnp` schema bytes (compile-time embedded via `include_bytes!`)
    pub schema: Option<&'static [u8]>,
}

/// Centralized endpoint registry with self-registration.
///
/// Supports services with multiple socket types (REP, ROUTER, PUB, etc.).
///
/// # Warning
///
/// Do NOT hold the guard from `global()` across `.await` points.
/// Clone needed data before awaiting:
///
/// ```ignore
/// let endpoint = global().endpoint("foo", SocketType::REP);  // Clone the TransportConfig
/// some_async_call().await;  // Lock released
/// ```
pub struct EndpointRegistry {
    mode: EndpointMode,
    runtime_dir: Option<PathBuf>,
    services: RwLock<HashMap<String, ServiceEntry>>,
}

impl EndpointRegistry {
    /// Create a new registry with the specified mode and runtime directory.
    ///
    /// # Arguments
    ///
    /// * `mode` - Transport mode (Inproc, Ipc, or Tcp)
    /// * `runtime_dir` - Base directory for IPC sockets (required for Ipc mode)
    pub fn new(mode: EndpointMode, runtime_dir: Option<PathBuf>) -> Self {
        Self {
            mode,
            runtime_dir,
            services: RwLock::new(HashMap::new()),
        }
    }

    /// Register a service endpoint for a specific socket type.
    ///
    /// Called by services on startup to advertise their endpoints.
    pub fn register(
        &self,
        name: &str,
        socket_kind: SocketKind,
        endpoint: TransportConfig,
        description: Option<&str>,
    ) {
        let mut services = self.services.write();
        let entry = services.entry(name.to_owned()).or_insert_with(|| ServiceEntry {
            name: name.to_owned(),
            endpoints: HashMap::new(),
            description: description.map(String::from),
            schema: None,
        });
        // Update description if provided
        if let Some(desc) = description {
            entry.description = Some(desc.to_owned());
        }
        entry.endpoints.insert(socket_kind, endpoint);
    }

    /// Unregister a specific socket type for a service.
    ///
    /// If all socket types are removed, the service entry is deleted.
    pub fn unregister(&self, name: &str, socket_kind: SocketKind) {
        let mut services = self.services.write();
        if let Some(entry) = services.get_mut(name) {
            entry.endpoints.remove(&socket_kind);
            if entry.endpoints.is_empty() {
                services.remove(name);
            }
        }
    }

    /// Unregister all socket types for a service.
    pub fn unregister_all(&self, name: &str) {
        self.services.write().remove(name);
    }

    /// Get endpoint for a service and socket type.
    ///
    /// Returns the registered endpoint if available, otherwise generates
    /// a default based on the current mode and socket type.
    pub fn endpoint(&self, name: &str, socket_kind: SocketKind) -> TransportConfig {
        if let Some(entry) = self.services.read().get(name) {
            if let Some(ep) = entry.endpoints.get(&socket_kind) {
                return ep.clone();
            }
        }
        // Generate default based on mode + socket type
        self.default_endpoint(name, socket_kind)
    }

    /// Generate a default endpoint for a service and socket type.
    fn default_endpoint(&self, name: &str, socket_kind: SocketKind) -> TransportConfig {
        let suffix = socket_kind.suffix();
        match self.mode {
            EndpointMode::Inproc => {
                TransportConfig::inproc(format!("hyprstream/{name}{suffix}"))
            }
            EndpointMode::Ipc => {
                // Use normal IPC binding instead of systemd socket activation for ZMQ sockets
                // ZMQ_USE_FD has compatibility issues with systemd socket activation
                let path = self
                    .runtime_dir
                    .as_ref()
                    .map(|d| d.join(format!("{name}{suffix}.sock")))
                    .unwrap_or_else(|| PathBuf::from(format!("/tmp/hyprstream/{name}{suffix}.sock")));
                TransportConfig::ipc(path)
            }
        }
    }

    // ========================================================================
    // Backward-compatible convenience methods (default to REP socket)
    // ========================================================================

    /// Register a REP endpoint (most common case, backward compatible).
    pub fn register_rep(&self, name: &str, endpoint: TransportConfig, description: Option<&str>) {
        self.register(name, SocketKind::Rep, endpoint, description);
    }

    /// Unregister a REP endpoint (backward compatible).
    pub fn unregister_rep(&self, name: &str) {
        self.unregister(name, SocketKind::Rep);
    }

    /// Get REP endpoint for a service (backward compatible).
    ///
    /// Equivalent to `endpoint(name, SocketKind::Rep)`.
    pub fn rep_endpoint(&self, name: &str) -> TransportConfig {
        self.endpoint(name, SocketKind::Rep)
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get the current endpoint mode.
    pub fn mode(&self) -> EndpointMode {
        self.mode
    }

    /// Get the runtime directory (if set).
    pub fn runtime_dir(&self) -> Option<&PathBuf> {
        self.runtime_dir.as_ref()
    }

    /// List all registered services.
    pub fn list_services(&self) -> Vec<String> {
        self.services.read().keys().cloned().collect()
    }

    /// Get all endpoints for a service.
    pub fn service_endpoints(&self, name: &str) -> Option<HashMap<SocketKind, TransportConfig>> {
        self.services.read().get(name).map(|e| e.endpoints.clone())
    }

    /// Register a service endpoint with schema bytes.
    pub fn register_with_schema(
        &self,
        name: &str,
        socket_kind: SocketKind,
        endpoint: TransportConfig,
        description: Option<&str>,
        schema: Option<&'static [u8]>,
    ) {
        let mut services = self.services.write();
        let entry = services.entry(name.to_owned()).or_insert_with(|| ServiceEntry {
            name: name.to_owned(),
            endpoints: HashMap::new(),
            description: description.map(String::from),
            schema: None,
        });
        if let Some(desc) = description {
            entry.description = Some(desc.to_owned());
        }
        if schema.is_some() {
            entry.schema = schema;
        }
        entry.endpoints.insert(socket_kind, endpoint);
    }

    /// Get schema bytes for a service (if registered with a schema).
    pub fn service_schema(&self, name: &str) -> Option<&'static [u8]> {
        self.services.read().get(name).and_then(|e| e.schema)
    }
}

// Global registry
static REGISTRY: RwLock<Option<EndpointRegistry>> = RwLock::new(None);

/// Initialize the global endpoint registry.
///
/// Idempotent: subsequent calls are no-ops if already initialized.
///
/// # Arguments
///
/// * `mode` - Transport mode (Inproc, Ipc, or Tcp)
/// * `runtime_dir` - Base directory for IPC sockets (required for Ipc mode)
pub fn init(mode: EndpointMode, runtime_dir: Option<PathBuf>) {
    let mut guard = REGISTRY.write();
    if guard.is_none() {
        *guard = Some(EndpointRegistry::new(mode, runtime_dir));
    }
}

/// Shutdown the global registry.
///
/// Clears all registered services.
pub fn shutdown() {
    *REGISTRY.write() = None;
}

/// Get the global registry.
///
/// # Panics
///
/// Panics if the registry has not been initialized.
///
/// # Warning
///
/// Do NOT hold the returned guard across `.await` points.
/// Clone needed data before awaiting.
pub fn global() -> impl std::ops::Deref<Target = EndpointRegistry> + 'static {
    parking_lot::RwLockReadGuard::map(REGISTRY.read(), |r| {
        match r.as_ref() {
            Some(registry) => registry,
            None => panic!("EndpointRegistry not initialized - call init() first"),
        }
    })
}

/// Try to get the global registry (non-panicking).
///
/// Returns `None` if the registry has not been initialized.
pub fn try_global() -> Option<impl std::ops::Deref<Target = EndpointRegistry> + 'static> {
    let guard = REGISTRY.read();
    if guard.is_some() {
        Some(parking_lot::RwLockReadGuard::map(guard, |r| {
            // SAFETY: We checked is_some() above, so this is guaranteed to succeed
            match r.as_ref() {
                Some(registry) => registry,
                None => unreachable!("checked is_some() above"),
            }
        }))
    } else {
        None
    }
}

/// RAII guard for service registration.
///
/// Automatically unregisters the service when dropped.
///
/// # Example
///
/// ```ignore
/// // Single REP endpoint (backward compatible)
/// let _reg = ServiceRegistration::new("policy", endpoint, None)?;
///
/// // Specific socket type
/// let _reg = ServiceRegistration::with_socket_type("inference", SocketKind::XPub, endpoint, None)?;
///
/// // Multiple endpoints
/// let _reg = ServiceRegistration::multi("model", vec![
///     (SocketKind::Rep, rep_endpoint),
///     (SocketKind::Router, router_endpoint),
/// ], None)?;
/// ```
pub struct ServiceRegistration {
    name: String,
    socket_kinds: Vec<SocketKind>,
}

impl ServiceRegistration {
    /// Register a REP endpoint and return a guard (backward compatible).
    ///
    /// # Arguments
    ///
    /// * `name` - Service name for discovery
    /// * `endpoint` - The endpoint configuration
    /// * `description` - Optional human-readable description
    ///
    /// # Errors
    ///
    /// Returns an error if the registry has not been initialized.
    pub fn new(
        name: &str,
        endpoint: TransportConfig,
        description: Option<&str>,
    ) -> anyhow::Result<Self> {
        Self::with_socket_kind(name, SocketKind::Rep, endpoint, description)
    }

    /// Register an endpoint with a specific socket type.
    ///
    /// # Arguments
    ///
    /// * `name` - Service name for discovery
    /// * `socket_kind` - The ZMQ socket type
    /// * `endpoint` - The endpoint configuration
    /// * `description` - Optional human-readable description
    pub fn with_socket_kind(
        name: &str,
        socket_kind: SocketKind,
        endpoint: TransportConfig,
        description: Option<&str>,
    ) -> anyhow::Result<Self> {
        let registry = try_global().ok_or_else(|| anyhow::anyhow!("Registry not initialized"))?;
        registry.register(name, socket_kind, endpoint, description);
        Ok(Self {
            name: name.to_owned(),
            socket_kinds: vec![socket_kind],
        })
    }

    /// Register multiple endpoints at once.
    ///
    /// # Arguments
    ///
    /// * `name` - Service name for discovery
    /// * `endpoints` - Vec of (socket_kind, endpoint) pairs
    /// * `description` - Optional human-readable description
    pub fn multi(
        name: &str,
        endpoints: Vec<(SocketKind, TransportConfig)>,
        description: Option<&str>,
    ) -> anyhow::Result<Self> {
        let registry = try_global().ok_or_else(|| anyhow::anyhow!("Registry not initialized"))?;
        let mut socket_kinds = Vec::with_capacity(endpoints.len());
        for (socket_kind, endpoint) in endpoints {
            registry.register(name, socket_kind, endpoint, description);
            socket_kinds.push(socket_kind);
        }
        Ok(Self {
            name: name.to_owned(),
            socket_kinds,
        })
    }

    /// Add another socket type to an existing registration.
    pub fn add_socket_kind(
        &mut self,
        socket_kind: SocketKind,
        endpoint: TransportConfig,
        description: Option<&str>,
    ) -> anyhow::Result<()> {
        let registry = try_global().ok_or_else(|| anyhow::anyhow!("Registry not initialized"))?;
        registry.register(&self.name, socket_kind, endpoint, description);
        self.socket_kinds.push(socket_kind);
        Ok(())
    }

    /// Get the service name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the registered socket kinds.
    pub fn socket_kinds(&self) -> &[SocketKind] {
        &self.socket_kinds
    }
}

impl Drop for ServiceRegistration {
    fn drop(&mut self) {
        if let Some(registry) = try_global() {
            for socket_kind in &self.socket_kinds {
                registry.unregister(&self.name, *socket_kind);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use parking_lot::Mutex;

    // Serialize tests that use the global registry
    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    fn reset_registry() {
        *REGISTRY.write() = None;
    }

    #[test]
    fn test_inproc_defaults() {
        let _lock = TEST_MUTEX.lock();
        reset_registry();
        init(EndpointMode::Inproc, None);

        // REP socket (default, no suffix)
        let endpoint = global().endpoint("policy", SocketKind::Rep);
        assert_eq!(endpoint.to_zmq_string(), "inproc://hyprstream/policy");

        // ROUTER socket (with suffix)
        let endpoint = global().endpoint("model", SocketKind::Router);
        assert_eq!(endpoint.to_zmq_string(), "inproc://hyprstream/model-router");

        // XPUB socket (with suffix)
        let endpoint = global().endpoint("inference", SocketKind::XPub);
        assert_eq!(endpoint.to_zmq_string(), "inproc://hyprstream/inference-xpub");
    }

    #[test]
    fn test_ipc_defaults() {
        let _lock = TEST_MUTEX.lock();
        reset_registry();
        init(EndpointMode::Ipc, Some(PathBuf::from("/run/hyprstream")));

        // REP socket (default, no suffix)
        let endpoint = global().endpoint("policy", SocketKind::Rep);
        assert_eq!(endpoint.to_zmq_string(), "ipc:///run/hyprstream/policy.sock");

        // ROUTER socket (with suffix)
        let endpoint = global().endpoint("model", SocketKind::Router);
        assert_eq!(endpoint.to_zmq_string(), "ipc:///run/hyprstream/model-router.sock");
    }

    #[test]
    fn test_single_registration() -> anyhow::Result<()> {
        let _lock = TEST_MUTEX.lock();
        reset_registry();
        init(EndpointMode::Inproc, None);

        {
            let _reg = ServiceRegistration::new(
                "test-service",
                TransportConfig::inproc("custom/endpoint"),
                Some("Test service"),
            )?;

            let endpoint = global().endpoint("test-service", SocketKind::Rep);
            assert_eq!(endpoint.to_zmq_string(), "inproc://custom/endpoint");
        }

        // After drop, falls back to default
        let endpoint = global().endpoint("test-service", SocketKind::Rep);
        assert_eq!(endpoint.to_zmq_string(), "inproc://hyprstream/test-service");
        Ok(())
    }

    #[test]
    fn test_multi_registration() -> anyhow::Result<()> {
        let _lock = TEST_MUTEX.lock();
        reset_registry();
        init(EndpointMode::Inproc, None);

        {
            let _reg = ServiceRegistration::multi(
                "model",
                vec![
                    (SocketKind::Rep, TransportConfig::inproc("model/rep")),
                    (SocketKind::Router, TransportConfig::inproc("model/router")),
                ],
                Some("Model service"),
            )?;

            // Both endpoints registered
            let rep_ep = global().endpoint("model", SocketKind::Rep);
            assert_eq!(rep_ep.to_zmq_string(), "inproc://model/rep");

            let router_ep = global().endpoint("model", SocketKind::Router);
            assert_eq!(router_ep.to_zmq_string(), "inproc://model/router");
        }

        // After drop, both fall back to defaults
        let rep_ep = global().endpoint("model", SocketKind::Rep);
        assert_eq!(rep_ep.to_zmq_string(), "inproc://hyprstream/model");

        let router_ep = global().endpoint("model", SocketKind::Router);
        assert_eq!(router_ep.to_zmq_string(), "inproc://hyprstream/model-router");
        Ok(())
    }

    #[test]
    fn test_backward_compat() {
        let _lock = TEST_MUTEX.lock();
        reset_registry();
        init(EndpointMode::Inproc, None);

        // Old-style single endpoint registration
        let registry = global();
        registry.register_rep(
            "compat-service",
            TransportConfig::inproc("compat/endpoint"),
            None,
        );

        // Old-style retrieval
        let endpoint = registry.rep_endpoint("compat-service");
        assert_eq!(endpoint.to_zmq_string(), "inproc://compat/endpoint");

        // New-style retrieval works too
        let endpoint = registry.endpoint("compat-service", SocketKind::Rep);
        assert_eq!(endpoint.to_zmq_string(), "inproc://compat/endpoint");
    }

    #[test]
    fn test_try_global_uninitialized() {
        let _lock = TEST_MUTEX.lock();
        reset_registry();
        assert!(try_global().is_none());
    }

    #[test]
    fn test_list_services() {
        let _lock = TEST_MUTEX.lock();
        reset_registry();
        init(EndpointMode::Inproc, None);

        global().register("svc1", SocketKind::Rep, TransportConfig::inproc("a"), None);
        global().register("svc2", SocketKind::Router, TransportConfig::inproc("b"), None);

        let services = global().list_services();
        assert!(services.contains(&"svc1".to_owned()));
        assert!(services.contains(&"svc2".to_owned()));
    }

    #[test]
    fn test_socket_kind_conversion() {
        // No lock needed - doesn't use global registry
        // Test From<SocketKind> for zmq::SocketType
        let zmq_type: zmq::SocketType = SocketKind::Router.into();
        assert_eq!(zmq_type, zmq::SocketType::ROUTER);

        // Test From<zmq::SocketType> for SocketKind
        let socket_kind: SocketKind = zmq::SocketType::XPUB.into();
        assert_eq!(socket_kind, SocketKind::XPub);
    }
}
