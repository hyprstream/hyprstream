/// Standardized path formatting for MCP tool names
///
/// Convention:
/// - Single underscore (_): Used within component names
/// - Double underscore (__): Used as separator between components
///
/// Format: `<namespace>__<component1>__<component2>__...__<name>`
///
/// Examples:
/// - `bin__tcl_execute`
/// - `user__alice__utils__string_helper`
/// - `mcp__context7__get_library_docs`
use anyhow::{anyhow, Result};

pub const SEPARATOR: &str = "__";

/// Parse an MCP-formatted path into components
pub fn parse_path(path: &str) -> Vec<&str> {
    path.split(SEPARATOR).collect()
}

/// Join components into an MCP-formatted path
pub fn join_path(components: &[&str]) -> String {
    components.join(SEPARATOR)
}

/// Extract namespace from MCP path
pub fn get_namespace(path: &str) -> Option<&str> {
    path.split(SEPARATOR).next()
}

/// Extract tool name (last component) from MCP path
pub fn get_tool_name(path: &str) -> Option<&str> {
    path.split(SEPARATOR).last()
}

/// Check if a string is a valid component (no double underscores)
pub fn is_valid_component(s: &str) -> bool {
    !s.contains(SEPARATOR)
}

/// Format a version string for use in paths (dots to underscores)
pub fn format_version(version: &str) -> String {
    format!("v{}", version.replace('.', "_"))
}

/// Parse a version string from path format (underscores to dots)
pub fn parse_version(version: &str) -> String {
    version
        .strip_prefix('v')
        .unwrap_or(version)
        .replace('_', ".")
}

/// Standardized path patterns for different namespaces
pub struct PathPattern;

impl PathPattern {
    /// Format: `bin__<name>`
    pub fn bin(name: &str) -> String {
        join_path(&["bin", name])
    }

    /// Format: `sbin__<name>`
    pub fn sbin(name: &str) -> String {
        join_path(&["sbin", name])
    }

    /// Format: `docs__<name>`
    pub fn docs(name: &str) -> String {
        join_path(&["docs", name])
    }

    /// Format: `mcp__<server>__<name>`
    pub fn mcp(server: &str, name: &str) -> String {
        join_path(&["mcp", server, name])
    }

    /// Format: `mcp__<server>__<name>__v<version>`
    pub fn mcp_versioned(server: &str, name: &str, version: &str) -> String {
        join_path(&[
            "mcp",
            server,
            name,
            &format!("v{}", version.replace('.', "_")),
        ])
    }

    /// Format: `user__<user>__<name>` or `user__<user>__<package>__<name>`
    pub fn user(user: &str, package: Option<&str>, name: &str) -> String {
        match package {
            Some(pkg) => join_path(&["user", user, pkg, name]),
            None => join_path(&["user", user, name]),
        }
    }

    /// Format: `user__<user>__<package>__<name>__v<version>`
    pub fn user_versioned(user: &str, package: &str, name: &str, version: &str) -> String {
        join_path(&[
            "user",
            user,
            package,
            name,
            &format!("v{}", version.replace('.', "_")),
        ])
    }
}

/// Parse an MCP path into structured components
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedPath {
    pub namespace: String,
    pub server: Option<String>,  // For MCP namespace
    pub user: Option<String>,    // For user namespace
    pub package: Option<String>, // For user namespace
    pub name: String,
    pub version: Option<String>,
}

impl ParsedPath {
    /// Parse an MCP-formatted path
    pub fn parse(path: &str) -> Result<Self> {
        let parts = parse_path(path);

        match parts.as_slice() {
            ["bin", name] => {
                if name.is_empty() || *name == "_" {
                    return Err(anyhow!("Tool name cannot be empty or just underscore"));
                }
                Ok(Self {
                    namespace: "bin".to_string(),
                    server: None,
                    user: None,
                    package: None,
                    name: name.to_string(),
                    version: None,
                })
            }

            ["sbin", name] => {
                if name.is_empty() || *name == "_" {
                    return Err(anyhow!("Tool name cannot be empty or just underscore"));
                }
                Ok(Self {
                    namespace: "sbin".to_string(),
                    server: None,
                    user: None,
                    package: None,
                    name: name.to_string(),
                    version: None,
                })
            }

            ["docs", name] => {
                if name.is_empty() || *name == "_" {
                    return Err(anyhow!("Tool name cannot be empty or just underscore"));
                }
                Ok(Self {
                    namespace: "docs".to_string(),
                    server: None,
                    user: None,
                    package: None,
                    name: name.to_string(),
                    version: None,
                })
            }

            ["mcp", server, name] => {
                if name.is_empty() || server.is_empty() || *name == "_" || *server == "_" {
                    return Err(anyhow!("MCP tool name and server cannot be empty or just underscore"));
                }
                Ok(Self {
                    namespace: "mcp".to_string(),
                    server: Some(server.to_string()),
                    user: None,
                    package: None,
                    name: name.to_string(),
                    version: None,
                })
            }

            ["mcp", server, name, version] if version.starts_with('v') => {
                if name.is_empty() || server.is_empty() || *name == "_" || *server == "_" {
                    return Err(anyhow!("MCP tool name and server cannot be empty or just underscore"));
                }
                Ok(Self {
                    namespace: "mcp".to_string(),
                    server: Some(server.to_string()),
                    user: None,
                    package: None,
                    name: name.to_string(),
                    version: Some(version[1..].replace('_', ".")),
                })
            }

            ["user", user, name] => {
                if name.is_empty() || user.is_empty() || *name == "_" || *user == "_" {
                    return Err(anyhow!("User tool name and user cannot be empty or just underscore"));
                }
                Ok(Self {
                    namespace: "user".to_string(),
                    server: None,
                    user: Some(user.to_string()),
                    package: None,
                    name: name.to_string(),
                    version: None,
                })
            }

            ["user", user, package, name] => {
                if name.is_empty() || user.is_empty() || package.is_empty() || *name == "_" || *user == "_" || *package == "_" {
                    return Err(anyhow!("User tool name, user, and package cannot be empty or just underscore"));
                }
                Ok(Self {
                    namespace: "user".to_string(),
                    server: None,
                    user: Some(user.to_string()),
                    package: Some(package.to_string()),
                    name: name.to_string(),
                    version: None,
                })
            }

            ["user", user, package, name, version] if version.starts_with('v') => {
                if name.is_empty() || user.is_empty() || package.is_empty() || *name == "_" || *user == "_" || *package == "_" {
                    return Err(anyhow!("User tool name, user, and package cannot be empty or just underscore"));
                }
                Ok(Self {
                    namespace: "user".to_string(),
                    server: None,
                    user: Some(user.to_string()),
                    package: Some(package.to_string()),
                    name: name.to_string(),
                    version: Some(version[1..].replace('_', ".")),
                })
            }

            _ => Err(anyhow!("Invalid MCP path format: {}", path)),
        }
    }

    /// Convert back to MCP path format
    pub fn to_path(&self) -> String {
        match self.namespace.as_str() {
            "bin" => PathPattern::bin(&self.name),
            "sbin" => PathPattern::sbin(&self.name),
            "docs" => PathPattern::docs(&self.name),
            "mcp" => {
                let server = self.server.as_ref().expect("MCP namespace requires server");
                match &self.version {
                    Some(v) => PathPattern::mcp_versioned(server, &self.name, v),
                    None => PathPattern::mcp(server, &self.name),
                }
            }
            "user" => {
                let user = self.user.as_ref().expect("User namespace requires user");
                match (&self.package, &self.version) {
                    (Some(pkg), Some(v)) => PathPattern::user_versioned(user, pkg, &self.name, v),
                    (Some(pkg), None) => PathPattern::user(user, Some(pkg), &self.name),
                    (None, _) => PathPattern::user(user, None, &self.name),
                }
            }
            _ => panic!("Unknown namespace: {}", self.namespace),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_paths() {
        assert_eq!(parse_path("bin__tcl_execute"), vec!["bin", "tcl_execute"]);
        assert_eq!(parse_path("sbin__mcp_add"), vec!["sbin", "mcp_add"]);
    }

    #[test]
    fn test_join_path() {
        assert_eq!(join_path(&["bin", "tcl_execute"]), "bin__tcl_execute");
        assert_eq!(
            join_path(&["user", "alice", "utils", "helper"]),
            "user__alice__utils__helper"
        );
    }

    #[test]
    fn test_path_patterns() {
        assert_eq!(PathPattern::bin("tcl_execute"), "bin__tcl_execute");
        assert_eq!(
            PathPattern::mcp("context7", "get_docs"),
            "mcp__context7__get_docs"
        );
        assert_eq!(
            PathPattern::user("alice", Some("utils"), "helper"),
            "user__alice__utils__helper"
        );
    }

    #[test]
    fn test_parsed_path() {
        let path = ParsedPath::parse("mcp__context7__get_library_docs").unwrap();
        assert_eq!(path.namespace, "mcp");
        assert_eq!(path.server, Some("context7".to_string()));
        assert_eq!(path.name, "get_library_docs");
        assert_eq!(path.to_path(), "mcp__context7__get_library_docs");
    }

    #[test]
    fn test_tools_with_underscores() {
        let path = ParsedPath::parse("mcp__myserver__get_user_info").unwrap();
        assert_eq!(path.namespace, "mcp");
        assert_eq!(path.server, Some("myserver".to_string()));
        assert_eq!(path.name, "get_user_info");
    }

    #[test]
    fn test_version_formatting() {
        assert_eq!(format_version("1.2.3"), "v1_2_3");
        assert_eq!(parse_version("v1_2_3"), "1.2.3");
        assert_eq!(parse_version("1_2_3"), "1.2.3");
    }

    #[test]
    fn test_component_validation() {
        assert!(is_valid_component("tcl_execute"));
        assert!(is_valid_component("get-user-info"));
        assert!(!is_valid_component("invalid__component"));
    }

    #[test]
    fn test_all_namespace_types() {
        // Test all ParsedPath patterns
        let test_cases = vec![
            (
                "bin__tcl_execute",
                "bin",
                None,
                None,
                None,
                "tcl_execute",
                None,
            ),
            ("sbin__tool_add", "sbin", None, None, None, "tool_add", None),
            (
                "docs__molt_book",
                "docs",
                None,
                None,
                None,
                "molt_book",
                None,
            ),
            (
                "mcp__context7__resolve-library-id",
                "mcp",
                Some("context7"),
                None,
                None,
                "resolve-library-id",
                None,
            ),
            (
                "mcp__myserver__tool__v1_2_3",
                "mcp",
                Some("myserver"),
                None,
                None,
                "tool",
                Some("1.2.3"),
            ),
            (
                "user__alice__my_tool",
                "user",
                None,
                Some("alice"),
                None,
                "my_tool",
                None,
            ),
            (
                "user__alice__utils__helper",
                "user",
                None,
                Some("alice"),
                Some("utils"),
                "helper",
                None,
            ),
            (
                "user__bob__math__calc__v2_0",
                "user",
                None,
                Some("bob"),
                Some("math"),
                "calc",
                Some("2.0"),
            ),
        ];

        for (path_str, ns, server, user, pkg, name, version) in test_cases {
            let parsed = ParsedPath::parse(path_str).unwrap();
            assert_eq!(parsed.namespace, ns, "Failed for path: {}", path_str);
            assert_eq!(
                parsed.server,
                server.map(String::from),
                "Failed for path: {}",
                path_str
            );
            assert_eq!(
                parsed.user,
                user.map(String::from),
                "Failed for path: {}",
                path_str
            );
            assert_eq!(
                parsed.package,
                pkg.map(String::from),
                "Failed for path: {}",
                path_str
            );
            assert_eq!(parsed.name, name, "Failed for path: {}", path_str);
            assert_eq!(
                parsed.version,
                version.map(String::from),
                "Failed for path: {}",
                path_str
            );

            // Test round-trip
            assert_eq!(
                parsed.to_path(),
                path_str,
                "Round-trip failed for: {}",
                path_str
            );
        }
    }
}
