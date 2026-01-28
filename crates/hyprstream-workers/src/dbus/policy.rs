//! D-Bus policy resource formatters for Casbin integration
//!
//! Builds resource strings for policy checks in the format:
//! `dbus:<bus>:<destination>:<interface>:<member>`
//!
//! # Resource Format
//!
//! - Method calls: `dbus:session:org.freedesktop.Notifications:Notifications:Notify`
//! - Properties (get): `dbus:session:org.mpris.MediaPlayer2:Player:@PlaybackStatus`
//! - Properties (set): `dbus:session:org.mpris.MediaPlayer2:Player:@Volume`
//! - Signals: `dbus:session:org.freedesktop.DBus:DBus:#NameOwnerChanged`
//!
//! The `@` prefix indicates properties, `#` prefix indicates signals.
//!
//! # Policy Examples
//!
//! ```csv
//! # Allow sending desktop notifications
//! p, container:gui, dbus:session:org.freedesktop.Notifications:Notifications:Notify, call, allow
//!
//! # Allow reading any MPRIS property
//! p, container:gui, dbus:session:org.mpris.MediaPlayer2.*:*:@*, read, allow
//!
//! # Allow subscribing to NetworkManager state changes
//! p, container:gui, dbus:system:org.freedesktop.NetworkManager:NetworkManager:#StateChanged, subscribe, allow
//!
//! # Deny all D-Bus by default
//! p, container:*, dbus:*, *, deny
//! ```

use super::protocol::BusType;

/// D-Bus operation types for policy checks
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DbusOperation {
    /// Method call
    Call,
    /// Property read (Get)
    Read,
    /// Property write (Set)
    Write,
    /// Signal subscription
    Subscribe,
}

impl DbusOperation {
    /// Get the operation string for policy checks
    pub fn as_str(&self) -> &'static str {
        match self {
            DbusOperation::Call => "call",
            DbusOperation::Read => "read",
            DbusOperation::Write => "write",
            DbusOperation::Subscribe => "subscribe",
        }
    }
}

impl std::fmt::Display for DbusOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// D-Bus resource identifier for policy checks
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DbusResource {
    /// Bus type (system or session)
    pub bus: BusType,

    /// Destination bus name (e.g., "org.freedesktop.Notifications")
    pub destination: String,

    /// Interface name (e.g., "org.freedesktop.Notifications")
    pub interface: String,

    /// Member name with type prefix (method, @property, or #signal)
    pub member: String,
}

impl DbusResource {
    /// Create a resource for a method call
    ///
    /// # Arguments
    /// * `bus` - Bus type (system or session)
    /// * `destination` - Destination bus name
    /// * `interface` - Interface containing the method
    /// * `method` - Method name
    ///
    /// # Example
    /// ```ignore
    /// let resource = DbusResource::method_call(
    ///     BusType::Session,
    ///     "org.freedesktop.Notifications",
    ///     "org.freedesktop.Notifications",
    ///     "Notify",
    /// );
    /// assert_eq!(
    ///     resource.to_string(),
    ///     "dbus:session:org.freedesktop.Notifications:org.freedesktop.Notifications:Notify"
    /// );
    /// ```
    pub fn method_call(
        bus: BusType,
        destination: impl Into<String>,
        interface: impl Into<String>,
        method: impl Into<String>,
    ) -> Self {
        Self {
            bus,
            destination: destination.into(),
            interface: interface.into(),
            member: method.into(),
        }
    }

    /// Create a resource for a property read
    ///
    /// Properties are prefixed with `@` in the resource string.
    pub fn property_read(
        bus: BusType,
        destination: impl Into<String>,
        interface: impl Into<String>,
        property: impl Into<String>,
    ) -> Self {
        Self {
            bus,
            destination: destination.into(),
            interface: interface.into(),
            member: format!("@{}", property.into()),
        }
    }

    /// Create a resource for a property write
    ///
    /// Properties are prefixed with `@` in the resource string.
    pub fn property_write(
        bus: BusType,
        destination: impl Into<String>,
        interface: impl Into<String>,
        property: impl Into<String>,
    ) -> Self {
        Self {
            bus,
            destination: destination.into(),
            interface: interface.into(),
            member: format!("@{}", property.into()),
        }
    }

    /// Create a resource for a signal subscription
    ///
    /// Signals are prefixed with `#` in the resource string.
    pub fn signal(
        bus: BusType,
        destination: impl Into<String>,
        interface: impl Into<String>,
        signal: impl Into<String>,
    ) -> Self {
        Self {
            bus,
            destination: destination.into(),
            interface: interface.into(),
            member: format!("#{}", signal.into()),
        }
    }

    /// Convert to resource string for policy checks
    ///
    /// Format: `dbus:<bus>:<destination>:<interface>:<member>`
    pub fn to_resource_string(&self) -> String {
        format!(
            "dbus:{}:{}:{}:{}",
            self.bus, self.destination, self.interface, self.member
        )
    }

    /// Check if this resource matches a pattern
    ///
    /// Supports wildcards (*) in patterns.
    pub fn matches_pattern(&self, pattern: &str) -> bool {
        let resource = self.to_resource_string();
        pattern_matches(&resource, pattern)
    }
}

impl std::fmt::Display for DbusResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_resource_string())
    }
}

/// Simple wildcard pattern matching
///
/// Supports `*` as wildcard that matches any sequence of characters.
fn pattern_matches(text: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    let pattern_parts: Vec<&str> = pattern.split('*').collect();

    if pattern_parts.len() == 1 {
        // No wildcards, exact match
        return text == pattern;
    }

    let mut pos = 0;

    for (i, part) in pattern_parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }

        if let Some(found) = text[pos..].find(part) {
            if i == 0 && found != 0 {
                // First part must match at start
                return false;
            }
            pos += found + part.len();
        } else {
            return false;
        }
    }

    // If pattern doesn't end with *, text must end exactly
    if !pattern.ends_with('*') && pos != text.len() {
        return false;
    }

    true
}

/// Build a subject identifier for policy checks
///
/// Format: `container:<sandbox_id>` or `container:<sandbox_id>:<container_id>`
pub fn container_subject(sandbox_id: &str, container_id: Option<&str>) -> String {
    match container_id {
        Some(cid) => format!("container:{sandbox_id}:{cid}"),
        None => format!("container:{sandbox_id}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_method_call_resource() {
        let resource = DbusResource::method_call(
            BusType::Session,
            "org.freedesktop.Notifications",
            "org.freedesktop.Notifications",
            "Notify",
        );

        assert_eq!(
            resource.to_resource_string(),
            "dbus:session:org.freedesktop.Notifications:org.freedesktop.Notifications:Notify"
        );
    }

    #[test]
    fn test_property_read_resource() {
        let resource = DbusResource::property_read(
            BusType::Session,
            "org.mpris.MediaPlayer2.spotify",
            "org.mpris.MediaPlayer2.Player",
            "PlaybackStatus",
        );

        assert_eq!(
            resource.to_resource_string(),
            "dbus:session:org.mpris.MediaPlayer2.spotify:org.mpris.MediaPlayer2.Player:@PlaybackStatus"
        );
    }

    #[test]
    fn test_property_write_resource() {
        let resource = DbusResource::property_write(
            BusType::Session,
            "org.mpris.MediaPlayer2.spotify",
            "org.mpris.MediaPlayer2.Player",
            "Volume",
        );

        assert_eq!(
            resource.to_resource_string(),
            "dbus:session:org.mpris.MediaPlayer2.spotify:org.mpris.MediaPlayer2.Player:@Volume"
        );
    }

    #[test]
    fn test_signal_resource() {
        let resource = DbusResource::signal(
            BusType::System,
            "org.freedesktop.NetworkManager",
            "org.freedesktop.NetworkManager",
            "StateChanged",
        );

        assert_eq!(
            resource.to_resource_string(),
            "dbus:system:org.freedesktop.NetworkManager:org.freedesktop.NetworkManager:#StateChanged"
        );
    }

    #[test]
    fn test_pattern_matching() {
        // Exact match
        assert!(pattern_matches("dbus:session:org.test:iface:Method", "dbus:session:org.test:iface:Method"));

        // Wildcard at end
        assert!(pattern_matches("dbus:session:org.test:iface:Method", "dbus:session:org.test:*"));

        // Wildcard in middle
        assert!(pattern_matches("dbus:session:org.test:iface:Method", "dbus:session:*:Method"));

        // Full wildcard
        assert!(pattern_matches("dbus:session:org.test:iface:Method", "*"));

        // No match
        assert!(!pattern_matches("dbus:session:org.test:iface:Method", "dbus:system:*"));
    }

    #[test]
    fn test_resource_matches_pattern() {
        let resource = DbusResource::method_call(
            BusType::Session,
            "org.freedesktop.Notifications",
            "org.freedesktop.Notifications",
            "Notify",
        );

        assert!(resource.matches_pattern("dbus:session:org.freedesktop.Notifications:*"));
        assert!(resource.matches_pattern("dbus:session:*:Notify"));
        assert!(resource.matches_pattern("dbus:*"));
        assert!(!resource.matches_pattern("dbus:system:*"));
    }

    #[test]
    fn test_container_subject() {
        assert_eq!(
            container_subject("sandbox-123", None),
            "container:sandbox-123"
        );

        assert_eq!(
            container_subject("sandbox-123", Some("container-456")),
            "container:sandbox-123:container-456"
        );
    }

    #[test]
    fn test_dbus_operation_display() {
        assert_eq!(DbusOperation::Call.as_str(), "call");
        assert_eq!(DbusOperation::Read.as_str(), "read");
        assert_eq!(DbusOperation::Write.as_str(), "write");
        assert_eq!(DbusOperation::Subscribe.as_str(), "subscribe");
    }
}
