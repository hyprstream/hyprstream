//! D-Bus bridge protocol types
//!
//! Defines the request/response types for the D-Bus proxy protocol.
//! These are serialized via JSON for simplicity (can be upgraded to Cap'n Proto later).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// D-Bus bus type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum BusType {
    /// System bus (/var/run/dbus/system_bus_socket)
    System,
    /// Session bus ($DBUS_SESSION_BUS_ADDRESS)
    #[default]
    Session,
}


impl std::fmt::Display for BusType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BusType::System => write!(f, "system"),
            BusType::Session => write!(f, "session"),
        }
    }
}

/// D-Bus value types supported by the bridge
///
/// A simplified subset of D-Bus types for common use cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DbusValue {
    /// String value
    String(String),
    /// Boolean value
    Bool(bool),
    /// Signed 32-bit integer
    Int32(i32),
    /// Unsigned 32-bit integer
    Uint32(u32),
    /// Signed 64-bit integer
    Int64(i64),
    /// Unsigned 64-bit integer
    Uint64(u64),
    /// Double-precision float
    Double(f64),
    /// Array of values
    Array(Vec<DbusValue>),
    /// Dictionary (a{sv} style)
    Dict(HashMap<String, DbusValue>),
    /// Byte array (useful for binary data)
    Bytes(Vec<u8>),
    /// Object path
    ObjectPath(String),
    /// Variant (boxed value)
    Variant(Box<DbusValue>),
}

impl DbusValue {
    /// Create a string value
    pub fn string(s: impl Into<String>) -> Self {
        DbusValue::String(s.into())
    }

    /// Create an object path value
    pub fn object_path(path: impl Into<String>) -> Self {
        DbusValue::ObjectPath(path.into())
    }

    /// Create a variant value
    pub fn variant(inner: DbusValue) -> Self {
        DbusValue::Variant(Box::new(inner))
    }

    /// Try to get as string
    pub fn as_str(&self) -> Option<&str> {
        match self {
            DbusValue::String(s) => Some(s),
            DbusValue::ObjectPath(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get as i64
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            DbusValue::Int32(n) => Some(*n as i64),
            DbusValue::Int64(n) => Some(*n),
            DbusValue::Uint32(n) => Some(*n as i64),
            _ => None,
        }
    }

    /// Try to get as bool
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            DbusValue::Bool(b) => Some(*b),
            _ => None,
        }
    }
}

/// Request to call a D-Bus method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbusCallRequest {
    /// Bus to use (system or session)
    #[serde(default)]
    pub bus: BusType,

    /// Destination bus name (e.g., "org.freedesktop.Notifications")
    pub destination: String,

    /// Object path (e.g., "/org/freedesktop/Notifications")
    pub path: String,

    /// Interface name (e.g., "org.freedesktop.Notifications")
    pub interface: String,

    /// Method name (e.g., "Notify")
    pub method: String,

    /// Method arguments
    #[serde(default)]
    pub args: Vec<DbusValue>,

    /// Timeout in milliseconds (0 = default)
    #[serde(default)]
    pub timeout_ms: u32,
}

impl DbusCallRequest {
    /// Create a new method call request
    pub fn new(
        destination: impl Into<String>,
        path: impl Into<String>,
        interface: impl Into<String>,
        method: impl Into<String>,
    ) -> Self {
        Self {
            bus: BusType::Session,
            destination: destination.into(),
            path: path.into(),
            interface: interface.into(),
            method: method.into(),
            args: Vec::new(),
            timeout_ms: 0,
        }
    }

    /// Set the bus type
    pub fn bus(mut self, bus: BusType) -> Self {
        self.bus = bus;
        self
    }

    /// Add an argument
    pub fn arg(mut self, value: DbusValue) -> Self {
        self.args.push(value);
        self
    }

    /// Set timeout
    pub fn timeout(mut self, ms: u32) -> Self {
        self.timeout_ms = ms;
        self
    }
}

/// Request to get a D-Bus property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbusGetPropertyRequest {
    /// Bus to use
    #[serde(default)]
    pub bus: BusType,

    /// Destination bus name
    pub destination: String,

    /// Object path
    pub path: String,

    /// Interface containing the property
    pub interface: String,

    /// Property name
    pub property: String,
}

/// Request to set a D-Bus property
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbusSetPropertyRequest {
    /// Bus to use
    #[serde(default)]
    pub bus: BusType,

    /// Destination bus name
    pub destination: String,

    /// Object path
    pub path: String,

    /// Interface containing the property
    pub interface: String,

    /// Property name
    pub property: String,

    /// New value
    pub value: DbusValue,
}

/// Request to subscribe to a D-Bus signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbusSubscribeRequest {
    /// Bus to use
    #[serde(default)]
    pub bus: BusType,

    /// Sender filter (optional)
    #[serde(default)]
    pub sender: Option<String>,

    /// Object path filter (optional)
    #[serde(default)]
    pub path: Option<String>,

    /// Interface filter
    pub interface: String,

    /// Signal name filter
    pub signal: String,
}

/// Response from a D-Bus method call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbusCallResponse {
    /// Return values from the method
    pub values: Vec<DbusValue>,
}

impl DbusCallResponse {
    /// Create a response with values
    pub fn new(values: Vec<DbusValue>) -> Self {
        Self { values }
    }

    /// Create an empty response
    pub fn empty() -> Self {
        Self { values: Vec::new() }
    }

    /// Get the first return value
    pub fn first(&self) -> Option<&DbusValue> {
        self.values.first()
    }
}

/// D-Bus error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbusError {
    /// Error name (e.g., "org.freedesktop.DBus.Error.ServiceUnknown")
    pub name: String,

    /// Error message
    pub message: String,
}

impl DbusError {
    /// Create a new D-Bus error
    pub fn new(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            message: message.into(),
        }
    }

    /// Create an access denied error
    pub fn access_denied(resource: &str) -> Self {
        Self::new(
            "org.freedesktop.DBus.Error.AccessDenied",
            format!("Access denied to {resource}"),
        )
    }

    /// Create a service unknown error
    pub fn service_unknown(service: &str) -> Self {
        Self::new(
            "org.freedesktop.DBus.Error.ServiceUnknown",
            format!("Service {service} is not available"),
        )
    }

    /// Create a method not found error
    pub fn method_not_found(interface: &str, method: &str) -> Self {
        Self::new(
            "org.freedesktop.DBus.Error.UnknownMethod",
            format!("Method {interface}.{method} not found"),
        )
    }
}

impl std::fmt::Display for DbusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.name, self.message)
    }
}

impl std::error::Error for DbusError {}

/// Signal received from D-Bus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DbusSignal {
    /// Sender bus name
    pub sender: String,

    /// Object path
    pub path: String,

    /// Interface name
    pub interface: String,

    /// Signal name
    pub signal: String,

    /// Signal arguments
    pub args: Vec<DbusValue>,
}

/// Unified D-Bus request type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DbusRequest {
    /// Call a method
    Call(DbusCallRequest),

    /// Get a property
    GetProperty(DbusGetPropertyRequest),

    /// Set a property
    SetProperty(DbusSetPropertyRequest),

    /// Subscribe to a signal
    Subscribe(DbusSubscribeRequest),

    /// Unsubscribe from signals
    Unsubscribe {
        /// Subscription ID to cancel
        subscription_id: String,
    },
}

/// Unified D-Bus response type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DbusResponse {
    /// Method call response
    Return(DbusCallResponse),

    /// Property value
    Property { value: DbusValue },

    /// Subscription created
    Subscribed {
        /// Subscription ID for tracking
        subscription_id: String,
    },

    /// Unsubscribed
    Unsubscribed,

    /// Error response
    Error(DbusError),
}

impl DbusResponse {
    /// Create a return response
    pub fn return_values(values: Vec<DbusValue>) -> Self {
        DbusResponse::Return(DbusCallResponse::new(values))
    }

    /// Create an error response
    pub fn error(name: impl Into<String>, message: impl Into<String>) -> Self {
        DbusResponse::Error(DbusError::new(name, message))
    }

    /// Create an access denied response
    pub fn access_denied(resource: &str) -> Self {
        DbusResponse::Error(DbusError::access_denied(resource))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dbus_value_serialization() {
        let value = DbusValue::Dict(HashMap::from([
            ("app_name".to_owned(), DbusValue::String("test".to_owned())),
            ("urgency".to_owned(), DbusValue::Uint32(1)),
        ]));

        let json = serde_json::to_string(&value).unwrap();
        let parsed: DbusValue = serde_json::from_str(&json).unwrap();

        match parsed {
            DbusValue::Dict(map) => {
                assert!(map.contains_key("app_name"));
                assert!(map.contains_key("urgency"));
            }
            _ => panic!("Expected Dict"),
        }
    }

    #[test]
    fn test_call_request_builder() {
        let request = DbusCallRequest::new(
            "org.freedesktop.Notifications",
            "/org/freedesktop/Notifications",
            "org.freedesktop.Notifications",
            "Notify",
        )
        .bus(BusType::Session)
        .arg(DbusValue::String("app".to_owned()))
        .arg(DbusValue::Uint32(0))
        .timeout(5000);

        assert_eq!(request.method, "Notify");
        assert_eq!(request.args.len(), 2);
        assert_eq!(request.timeout_ms, 5000);
    }

    #[test]
    fn test_request_serialization() {
        let request = DbusRequest::Call(DbusCallRequest::new(
            "org.example.Service",
            "/org/example/Object",
            "org.example.Interface",
            "Method",
        ));

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"type\":\"call\""));

        let parsed: DbusRequest = serde_json::from_str(&json).unwrap();
        match parsed {
            DbusRequest::Call(call) => {
                assert_eq!(call.destination, "org.example.Service");
            }
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_response_serialization() {
        let response = DbusResponse::return_values(vec![DbusValue::Uint32(42)]);

        let json = serde_json::to_string(&response).unwrap();
        assert!(json.contains("\"type\":\"return\""));
    }

    #[test]
    fn test_error_creation() {
        let err = DbusError::access_denied("dbus:session:org.example:Method");
        assert!(err.name.contains("AccessDenied"));
        assert!(err.message.contains("org.example"));
    }
}
