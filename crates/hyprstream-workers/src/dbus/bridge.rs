//! D-Bus bridge service for container D-Bus access
//!
//! Provides a ZMQ service that proxies D-Bus requests from containers
//! to the host D-Bus buses (system and session).

use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::policy::{container_subject, DbusOperation, DbusResource};
use super::protocol::{
    BusType, DbusCallRequest, DbusCallResponse, DbusGetPropertyRequest,
    DbusRequest, DbusResponse, DbusSetPropertyRequest, DbusSubscribeRequest, DbusValue,
};
use crate::error::{Result, WorkerError};

/// Configuration for the D-Bus bridge service
#[derive(Debug, Clone)]
pub struct DbusBridgeConfig {
    /// Whether to allow system bus access
    pub allow_system_bus: bool,

    /// Whether to allow session bus access
    pub allow_session_bus: bool,

    /// Default timeout for D-Bus calls (ms)
    pub default_timeout_ms: u32,

    /// Maximum concurrent subscriptions per container
    pub max_subscriptions_per_container: usize,
}

impl Default for DbusBridgeConfig {
    fn default() -> Self {
        Self {
            allow_system_bus: true,
            allow_session_bus: true,
            default_timeout_ms: 30_000,
            max_subscriptions_per_container: 100,
        }
    }
}

/// Signal subscription tracking
#[derive(Debug)]
struct SignalSubscription {
    /// Subscription ID
    id: String,

    /// Container that created this subscription
    container_id: String,

    /// Bus type
    bus: BusType,

    /// Interface filter
    interface: String,

    /// Signal filter
    signal: String,
}

/// D-Bus bridge service
///
/// Proxies D-Bus requests from containers to the host D-Bus buses.
/// All requests go through policy checks before being forwarded.
#[cfg(feature = "dbus")]
pub struct DbusBridgeService {
    /// Configuration
    config: DbusBridgeConfig,

    /// System bus connection
    system_bus: RwLock<Option<zbus::Connection>>,

    /// Session bus connection
    session_bus: RwLock<Option<zbus::Connection>>,

    /// Active signal subscriptions
    subscriptions: DashMap<String, SignalSubscription>,

    /// Policy check function (injected for flexibility)
    policy_check: Arc<dyn PolicyChecker>,
}

/// Trait for policy checking (allows mocking in tests)
pub trait PolicyChecker: Send + Sync {
    /// Check if the subject is allowed to perform the operation on the resource
    fn check(&self, subject: &str, resource: &str, operation: &str) -> bool;
}

/// Default policy checker that denies everything (for testing)
pub struct DenyAllPolicy;

impl PolicyChecker for DenyAllPolicy {
    fn check(&self, _subject: &str, _resource: &str, _operation: &str) -> bool {
        false
    }
}

/// Policy checker that allows everything (for testing)
pub struct AllowAllPolicy;

impl PolicyChecker for AllowAllPolicy {
    fn check(&self, _subject: &str, _resource: &str, _operation: &str) -> bool {
        true
    }
}

#[cfg(feature = "dbus")]
impl DbusBridgeService {
    /// Create a new D-Bus bridge service
    pub async fn new(config: DbusBridgeConfig) -> Result<Self> {
        Self::with_policy(config, Arc::new(DenyAllPolicy)).await
    }

    /// Create a new D-Bus bridge service with a custom policy checker
    pub async fn with_policy(
        config: DbusBridgeConfig,
        policy_check: Arc<dyn PolicyChecker>,
    ) -> Result<Self> {
        Ok(Self {
            config,
            system_bus: RwLock::new(None),
            session_bus: RwLock::new(None),
            subscriptions: DashMap::new(),
            policy_check,
        })
    }

    /// Connect to D-Bus buses lazily
    async fn ensure_connected(&self, bus: BusType) -> Result<zbus::Connection> {
        match bus {
            BusType::System => {
                let mut guard = self.system_bus.write().await;
                if let Some(conn) = guard.as_ref() {
                    return Ok(conn.clone());
                }

                if !self.config.allow_system_bus {
                    return Err(WorkerError::ConfigError(
                        "System bus access not allowed".to_string(),
                    ));
                }

                let conn = zbus::Connection::system()
                    .await
                    .map_err(|e| WorkerError::Internal(format!("Failed to connect to system bus: {}", e)))?;

                *guard = Some(conn.clone());
                Ok(conn)
            }
            BusType::Session => {
                let mut guard = self.session_bus.write().await;
                if let Some(conn) = guard.as_ref() {
                    return Ok(conn.clone());
                }

                if !self.config.allow_session_bus {
                    return Err(WorkerError::ConfigError(
                        "Session bus access not allowed".to_string(),
                    ));
                }

                let conn = zbus::Connection::session()
                    .await
                    .map_err(|e| WorkerError::Internal(format!("Failed to connect to session bus: {}", e)))?;

                *guard = Some(conn.clone());
                Ok(conn)
            }
        }
    }

    /// Handle a D-Bus request from a container
    pub async fn handle_request(
        &self,
        container_id: &str,
        request: DbusRequest,
    ) -> DbusResponse {
        let subject = container_subject(container_id, None);

        match request {
            DbusRequest::Call(call) => self.handle_call(&subject, call).await,
            DbusRequest::GetProperty(get) => self.handle_get_property(&subject, get).await,
            DbusRequest::SetProperty(set) => self.handle_set_property(&subject, set).await,
            DbusRequest::Subscribe(sub) => self.handle_subscribe(&subject, container_id, sub).await,
            DbusRequest::Unsubscribe { subscription_id } => {
                self.handle_unsubscribe(&subject, container_id, &subscription_id).await
            }
        }
    }

    /// Handle a method call request
    async fn handle_call(&self, subject: &str, call: DbusCallRequest) -> DbusResponse {
        // Build resource and check policy
        let resource = DbusResource::method_call(
            call.bus,
            &call.destination,
            &call.interface,
            &call.method,
        );

        if !self.policy_check.check(subject, &resource.to_resource_string(), DbusOperation::Call.as_str()) {
            tracing::warn!(
                subject = %subject,
                resource = %resource,
                "D-Bus call denied by policy"
            );
            return DbusResponse::access_denied(&resource.to_resource_string());
        }

        // Get connection
        let conn = match self.ensure_connected(call.bus).await {
            Ok(c) => c,
            Err(e) => {
                return DbusResponse::error(
                    "org.freedesktop.DBus.Error.Failed",
                    e.to_string(),
                );
            }
        };

        // Make the D-Bus call
        let timeout = if call.timeout_ms > 0 {
            Duration::from_millis(call.timeout_ms as u64)
        } else {
            Duration::from_millis(self.config.default_timeout_ms as u64)
        };

        // Convert args to zbus values and make the call
        // This is a simplified implementation - full implementation would need
        // proper type conversion between DbusValue and zbus::Value
        match self.forward_call(&conn, &call, timeout).await {
            Ok(values) => DbusResponse::Return(DbusCallResponse::new(values)),
            Err(e) => DbusResponse::error(
                "org.freedesktop.DBus.Error.Failed",
                e.to_string(),
            ),
        }
    }

    /// Forward a D-Bus call to the actual bus
    async fn forward_call(
        &self,
        conn: &zbus::Connection,
        call: &DbusCallRequest,
        timeout: Duration,
    ) -> Result<Vec<DbusValue>> {
        use zbus::names::{BusName, InterfaceName, MemberName};
        use zbus::zvariant::ObjectPath;

        // Parse bus name
        let destination: BusName = call
            .destination
            .as_str()
            .try_into()
            .map_err(|e| WorkerError::Internal(format!("Invalid destination: {}", e)))?;

        // Parse interface name
        let interface: InterfaceName = call
            .interface
            .as_str()
            .try_into()
            .map_err(|e| WorkerError::Internal(format!("Invalid interface: {}", e)))?;

        // Parse object path
        let path: ObjectPath = call
            .path
            .as_str()
            .try_into()
            .map_err(|e| WorkerError::Internal(format!("Invalid object path: {}", e)))?;

        // Parse method name
        let method: MemberName = call
            .method
            .as_str()
            .try_into()
            .map_err(|e| WorkerError::Internal(format!("Invalid method name: {}", e)))?;

        // Make the D-Bus call with timeout
        // Note: This simplified version sends no arguments - full implementation would convert DbusValue
        let call_future = conn.call_method(Some(destination), path, Some(interface), method, &());

        let reply = tokio::time::timeout(timeout, call_future)
            .await
            .map_err(|_| WorkerError::Internal(format!(
                "D-Bus call timed out after {:?}",
                timeout
            )))?;

        match reply {
            Ok(_msg) => {
                // Full implementation would parse _msg.body() into DbusValue vec
                Ok(Vec::new())
            }
            Err(e) => Err(WorkerError::Internal(format!("D-Bus call failed: {}", e))),
        }
    }

    /// Handle a property get request
    async fn handle_get_property(&self, subject: &str, get: DbusGetPropertyRequest) -> DbusResponse {
        let resource = DbusResource::property_read(
            get.bus,
            &get.destination,
            &get.interface,
            &get.property,
        );

        if !self.policy_check.check(subject, &resource.to_resource_string(), DbusOperation::Read.as_str()) {
            return DbusResponse::access_denied(&resource.to_resource_string());
        }

        // Get connection
        let conn = match self.ensure_connected(get.bus).await {
            Ok(c) => c,
            Err(e) => {
                return DbusResponse::error(
                    "org.freedesktop.DBus.Error.Failed",
                    e.to_string(),
                );
            }
        };

        // Forward the property get request
        match self.forward_get_property(&conn, &get).await {
            Ok(value) => DbusResponse::Property { value },
            Err(e) => DbusResponse::error(
                "org.freedesktop.DBus.Error.Failed",
                e.to_string(),
            ),
        }
    }

    /// Forward a property get request to the actual bus
    async fn forward_get_property(
        &self,
        conn: &zbus::Connection,
        get: &DbusGetPropertyRequest,
    ) -> Result<DbusValue> {
        use zbus::names::{BusName, InterfaceName, MemberName};
        use zbus::zvariant::ObjectPath;

        // Parse destination
        let destination: BusName = get
            .destination
            .as_str()
            .try_into()
            .map_err(|e| WorkerError::Internal(format!("Invalid destination: {}", e)))?;

        // Parse object path
        let path: ObjectPath = get
            .path
            .as_str()
            .try_into()
            .map_err(|e| WorkerError::Internal(format!("Invalid object path: {}", e)))?;

        // Parse interface for the property
        let interface: InterfaceName = get
            .interface
            .as_str()
            .try_into()
            .map_err(|e| WorkerError::Internal(format!("Invalid interface: {}", e)))?;

        // Parse property name
        let property: MemberName = get
            .property
            .as_str()
            .try_into()
            .map_err(|e| WorkerError::Internal(format!("Invalid property name: {}", e)))?;

        // Call org.freedesktop.DBus.Properties.Get
        let properties_interface: InterfaceName = "org.freedesktop.DBus.Properties"
            .try_into()
            .expect("valid interface name");
        let get_method: MemberName = "Get"
            .try_into()
            .expect("valid method name");

        let reply = conn
            .call_method(
                Some(destination),
                path,
                Some(properties_interface),
                get_method,
                &(interface.as_str(), property.as_str()),
            )
            .await
            .map_err(|e| WorkerError::Internal(format!("D-Bus property get failed: {}", e)))?;

        // Parse the reply - Properties.Get returns a variant
        // For now, return the debug representation as a string
        // Full implementation would properly convert zvariant::Value to DbusValue
        let body = reply.body();
        Ok(DbusValue::String(format!("{:?}", body)))
    }

    /// Handle a property set request
    async fn handle_set_property(&self, subject: &str, set: DbusSetPropertyRequest) -> DbusResponse {
        let resource = DbusResource::property_write(
            set.bus,
            &set.destination,
            &set.interface,
            &set.property,
        );

        if !self.policy_check.check(subject, &resource.to_resource_string(), DbusOperation::Write.as_str()) {
            return DbusResponse::access_denied(&resource.to_resource_string());
        }

        // Get connection
        let conn = match self.ensure_connected(set.bus).await {
            Ok(c) => c,
            Err(e) => {
                return DbusResponse::error(
                    "org.freedesktop.DBus.Error.Failed",
                    e.to_string(),
                );
            }
        };

        // Forward the property set request
        match self.forward_set_property(&conn, &set).await {
            Ok(()) => DbusResponse::Return(DbusCallResponse::empty()),
            Err(e) => DbusResponse::error(
                "org.freedesktop.DBus.Error.Failed",
                e.to_string(),
            ),
        }
    }

    /// Forward a property set request to the actual bus
    async fn forward_set_property(
        &self,
        conn: &zbus::Connection,
        set: &DbusSetPropertyRequest,
    ) -> Result<()> {
        use zbus::names::{BusName, InterfaceName, MemberName};
        use zbus::zvariant::{ObjectPath, Value};

        // Parse destination
        let destination: BusName = set
            .destination
            .as_str()
            .try_into()
            .map_err(|e| WorkerError::Internal(format!("Invalid destination: {}", e)))?;

        // Parse object path
        let path: ObjectPath = set
            .path
            .as_str()
            .try_into()
            .map_err(|e| WorkerError::Internal(format!("Invalid object path: {}", e)))?;

        // Parse interface for the property
        let interface: InterfaceName = set
            .interface
            .as_str()
            .try_into()
            .map_err(|e| WorkerError::Internal(format!("Invalid interface: {}", e)))?;

        // Parse property name
        let property: MemberName = set
            .property
            .as_str()
            .try_into()
            .map_err(|e| WorkerError::Internal(format!("Invalid property name: {}", e)))?;

        // Convert DbusValue to zvariant::Value
        let zvalue = dbus_value_to_zvariant(&set.value)?;

        // Call org.freedesktop.DBus.Properties.Set
        let properties_interface: InterfaceName = "org.freedesktop.DBus.Properties"
            .try_into()
            .expect("valid interface name");
        let set_method: MemberName = "Set"
            .try_into()
            .expect("valid method name");

        conn.call_method(
            Some(destination),
            path,
            Some(properties_interface),
            set_method,
            &(interface.as_str(), property.as_str(), Value::from(zvalue)),
        )
        .await
        .map_err(|e| WorkerError::Internal(format!("D-Bus property set failed: {}", e)))?;

        Ok(())
    }

    /// Handle a signal subscription request
    async fn handle_subscribe(
        &self,
        subject: &str,
        container_id: &str,
        sub: DbusSubscribeRequest,
    ) -> DbusResponse {
        let resource = DbusResource::signal(
            sub.bus,
            sub.sender.as_deref().unwrap_or("*"),
            &sub.interface,
            &sub.signal,
        );

        if !self.policy_check.check(subject, &resource.to_resource_string(), DbusOperation::Subscribe.as_str()) {
            return DbusResponse::access_denied(&resource.to_resource_string());
        }

        // Check subscription limit
        let container_subs: usize = self
            .subscriptions
            .iter()
            .filter(|s| s.container_id == container_id)
            .count();

        if container_subs >= self.config.max_subscriptions_per_container {
            return DbusResponse::error(
                "org.freedesktop.DBus.Error.LimitsExceeded",
                format!(
                    "Maximum subscriptions ({}) exceeded",
                    self.config.max_subscriptions_per_container
                ),
            );
        }

        // Create subscription
        let subscription_id = Uuid::new_v4().to_string();

        self.subscriptions.insert(
            subscription_id.clone(),
            SignalSubscription {
                id: subscription_id.clone(),
                container_id: container_id.to_string(),
                bus: sub.bus,
                interface: sub.interface,
                signal: sub.signal,
            },
        );

        // Full implementation would set up actual D-Bus signal matching
        // and forward matching signals to the container via ZMQ PUB

        DbusResponse::Subscribed { subscription_id }
    }

    /// Handle unsubscribe request
    async fn handle_unsubscribe(
        &self,
        _subject: &str,
        container_id: &str,
        subscription_id: &str,
    ) -> DbusResponse {
        // Only allow unsubscribing own subscriptions
        if let Some((_, sub)) = self.subscriptions.remove(subscription_id) {
            if sub.container_id != container_id {
                // Put it back - not their subscription
                self.subscriptions.insert(subscription_id.to_string(), sub);
                return DbusResponse::error(
                    "org.freedesktop.DBus.Error.AccessDenied",
                    "Cannot unsubscribe from another container's subscription",
                );
            }
        }

        DbusResponse::Unsubscribed
    }

    /// Get the number of active subscriptions
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }

    /// Clean up subscriptions for a container
    pub fn cleanup_container(&self, container_id: &str) {
        self.subscriptions.retain(|_, sub| sub.container_id != container_id);
    }
}

/// Stub implementation when dbus feature is disabled
#[cfg(not(feature = "dbus"))]
pub struct DbusBridgeService {
    _private: (),
}

#[cfg(not(feature = "dbus"))]
impl DbusBridgeService {
    /// Create a new D-Bus bridge service (stub)
    pub async fn new(_config: DbusBridgeConfig) -> Result<Self> {
        Err(WorkerError::ConfigError(
            "D-Bus bridge requires the 'dbus' feature to be enabled".to_string(),
        ))
    }
}

/// Convert a DbusValue to a zvariant-compatible string representation
///
/// Note: This is a simplified conversion. Full implementation would need
/// to properly convert to zvariant::Value with correct D-Bus type signatures.
#[cfg(feature = "dbus")]
fn dbus_value_to_zvariant(value: &DbusValue) -> Result<String> {
    match value {
        DbusValue::String(s) => Ok(s.clone()),
        DbusValue::Bool(b) => Ok(b.to_string()),
        DbusValue::Int32(n) => Ok(n.to_string()),
        DbusValue::Uint32(n) => Ok(n.to_string()),
        DbusValue::Int64(n) => Ok(n.to_string()),
        DbusValue::Uint64(n) => Ok(n.to_string()),
        DbusValue::Double(n) => Ok(n.to_string()),
        DbusValue::ObjectPath(p) => Ok(p.clone()),
        DbusValue::Variant(inner) => dbus_value_to_zvariant(inner),
        DbusValue::Array(_) | DbusValue::Dict(_) | DbusValue::Bytes(_) => {
            Err(WorkerError::Internal(
                "Complex D-Bus types (Array, Dict, Bytes) not yet supported for property set".to_string(),
            ))
        }
    }
}

#[cfg(all(test, feature = "dbus"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_creation() {
        let config = DbusBridgeConfig::default();
        let bridge = DbusBridgeService::with_policy(
            config,
            Arc::new(AllowAllPolicy),
        )
        .await
        .unwrap();

        assert_eq!(bridge.subscription_count(), 0);
    }

    #[tokio::test]
    async fn test_policy_denied() {
        let config = DbusBridgeConfig::default();
        let bridge = DbusBridgeService::with_policy(
            config,
            Arc::new(DenyAllPolicy),
        )
        .await
        .unwrap();

        let request = DbusRequest::Call(DbusCallRequest::new(
            "org.freedesktop.Notifications",
            "/org/freedesktop/Notifications",
            "org.freedesktop.Notifications",
            "Notify",
        ));

        let response = bridge.handle_request("container-1", request).await;

        match response {
            DbusResponse::Error(err) => {
                assert!(err.name.contains("AccessDenied"));
            }
            _ => panic!("Expected error response"),
        }
    }

    #[tokio::test]
    async fn test_subscribe_unsubscribe() {
        let config = DbusBridgeConfig::default();
        let bridge = DbusBridgeService::with_policy(
            config,
            Arc::new(AllowAllPolicy),
        )
        .await
        .unwrap();

        // Subscribe
        let request = DbusRequest::Subscribe(DbusSubscribeRequest {
            bus: BusType::Session,
            sender: None,
            path: None,
            interface: "org.freedesktop.DBus".to_string(),
            signal: "NameOwnerChanged".to_string(),
        });

        let response = bridge.handle_request("container-1", request).await;

        let subscription_id = match response {
            DbusResponse::Subscribed { subscription_id } => subscription_id,
            _ => panic!("Expected subscribed response"),
        };

        assert_eq!(bridge.subscription_count(), 1);

        // Unsubscribe
        let request = DbusRequest::Unsubscribe { subscription_id };
        let response = bridge.handle_request("container-1", request).await;

        match response {
            DbusResponse::Unsubscribed => {}
            _ => panic!("Expected unsubscribed response"),
        }

        assert_eq!(bridge.subscription_count(), 0);
    }

    #[tokio::test]
    async fn test_cleanup_container() {
        let config = DbusBridgeConfig::default();
        let bridge = DbusBridgeService::with_policy(
            config,
            Arc::new(AllowAllPolicy),
        )
        .await
        .unwrap();

        // Create subscriptions for two containers
        for container in ["container-1", "container-2"] {
            let request = DbusRequest::Subscribe(DbusSubscribeRequest {
                bus: BusType::Session,
                sender: None,
                path: None,
                interface: "org.test".to_string(),
                signal: "Test".to_string(),
            });
            bridge.handle_request(container, request).await;
        }

        assert_eq!(bridge.subscription_count(), 2);

        // Clean up container-1
        bridge.cleanup_container("container-1");

        assert_eq!(bridge.subscription_count(), 1);
    }
}

#[cfg(all(test, not(feature = "dbus")))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bridge_disabled() {
        let config = DbusBridgeConfig::default();
        let result = DbusBridgeService::new(config).await;
        assert!(result.is_err());
    }
}
