//! D-Bus bridge for container access to host services
//!
//! Enables containers to transparently access D-Bus services (notifications,
//! media controls, etc.) with fine-grained policy enforcement.
//!
//! # Architecture
//!
//! ```text
//! Container (GUI app)
//!     │
//!     ▼ (ZMQ over vsock/TCP)
//! DbusBridgeService
//!     │
//!     ├── PolicyClient.check(subject, resource, operation)
//!     │       └── resource: "dbus:system:org.freedesktop.Notifications:Notify"
//!     │
//!     └── zbus Connection (system or session bus)
//!             └── Forward permitted calls to actual D-Bus
//! ```
//!
//! # Protocol
//!
//! Containers send D-Bus requests via ZMQ:
//!
//! ```text
//! Request: DbusRequest { bus, destination, path, interface, method, args }
//! Response: DbusResponse { result } or DbusError { name, message }
//! ```
//!
//! # Policy Integration
//!
//! Every D-Bus operation is checked against Casbin policy:
//!
//! - Resource format: `dbus:<bus>:<destination>:<interface>:<member>`
//! - Operations: `call` (methods), `read` (properties), `write` (properties), `subscribe` (signals)
//!
//! # Example Policy
//!
//! ```csv
//! # Allow container to send desktop notifications
//! p, container:gui-app, dbus:session:org.freedesktop.Notifications:Notify, call, allow
//!
//! # Allow reading media player properties
//! p, container:gui-app, dbus:session:org.mpris.MediaPlayer2.*:@*, read, allow
//!
//! # Deny all D-Bus by default
//! p, container:*, dbus:*, *, deny
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use hyprstream_workers::dbus::{DbusBridgeService, DbusBridgeConfig};
//!
//! // Create the bridge service
//! let config = DbusBridgeConfig::default();
//! let bridge = DbusBridgeService::new(config).await?;
//!
//! // Start as ZMQ service
//! let handle = bridge.start("ipc:///run/hyprstream/dbus-bridge.sock").await?;
//! ```

mod bridge;
mod policy;
mod protocol;

pub use bridge::{AllowAllPolicy, DbusBridgeConfig, DbusBridgeService, DenyAllPolicy, PolicyChecker};
pub use policy::{container_subject, DbusOperation, DbusResource};
pub use protocol::{
    BusType, DbusCallRequest, DbusCallResponse, DbusError, DbusGetPropertyRequest, DbusRequest,
    DbusResponse, DbusSetPropertyRequest, DbusSignal, DbusSubscribeRequest, DbusValue,
};
