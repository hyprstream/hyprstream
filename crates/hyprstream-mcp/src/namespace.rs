use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

/// MCP path separator
const SEPARATOR: &str = "__";

/// Parse an MCP-formatted path into components
fn parse_path(path: &str) -> Vec<&str> {
    path.split(SEPARATOR).collect()
}

/// Join components into an MCP-formatted path
fn join_path(components: &[&str]) -> String {
    components.join(SEPARATOR)
}

/// Format a version string for use in paths (dots to underscores)
fn format_version(version: &str) -> String {
    format!("v{}", version.replace('.', "_"))
}

/// Parse a version string from path format (underscores to dots)
fn parse_version(version: &str) -> String {
    version
        .strip_prefix('v')
        .unwrap_or(version)
        .replace('_', ".")
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Namespace {
    Bin,          // System tools (read-only)
    Sbin,         // System admin tools (privileged)
    Docs,         // Documentation tools (read-only)
    User(String), // User namespace
    Mcp(String),  // MCP server namespace
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolPath {
    pub namespace: Namespace,
    pub package: Option<String>,
    pub name: String,
    pub version: String,
}

impl ToolPath {
    /// Create a new system binary tool path
    pub fn bin(name: impl Into<String>) -> Self {
        Self {
            namespace: Namespace::Bin,
            package: None,
            name: name.into(),
            version: "latest".to_string(),
        }
    }

    /// Create a new system admin tool path
    pub fn sbin(name: impl Into<String>) -> Self {
        Self {
            namespace: Namespace::Sbin,
            package: None,
            name: name.into(),
            version: "latest".to_string(),
        }
    }

    /// Create a new documentation tool path
    pub fn docs(name: impl Into<String>) -> Self {
        Self {
            namespace: Namespace::Docs,
            package: None,
            name: name.into(),
            version: "latest".to_string(),
        }
    }

    /// Create a new user tool path
    pub fn user(
        user: impl Into<String>,
        package: impl Into<String>,
        name: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self {
            namespace: Namespace::User(user.into()),
            package: Some(package.into()),
            name: name.into(),
            version: version.into(),
        }
    }

    /// Create a new MCP server tool path
    pub fn mcp(
        server: impl Into<String>,
        name: impl Into<String>,
        version: impl Into<String>,
    ) -> Self {
        Self {
            namespace: Namespace::Mcp(server.into()),
            package: None,
            name: name.into(),
            version: version.into(),
        }
    }

    /// Parse a tool path from MCP name format
    /// Examples:
    /// - "bin__tcl_execute"
    /// - "sbin__tcl_tool_add"
    /// - "mcp__filesystem__read_file__v1_0"
    /// - "user__alice__utils__reverse_string__v1_0"
    /// - "user__bob__math__calculate"
    pub fn parse(path: &str) -> Result<Self> {
        Self::from_mcp_name(path)
    }

    /// Convert to MCP-compatible tool name using standardized path format
    pub fn to_mcp_name(&self) -> String {
        match &self.namespace {
            Namespace::Bin => join_path(&["bin", &self.name]),
            Namespace::Sbin => join_path(&["sbin", &self.name]),
            Namespace::Docs => join_path(&["docs", &self.name]),
            Namespace::Mcp(server) => {
                if self.version == "latest" {
                    join_path(&["mcp", server, &self.name])
                } else {
                    join_path(&["mcp", server, &self.name, &format_version(&self.version)])
                }
            }
            Namespace::User(user) => {
                if let Some(package) = &self.package {
                    if self.version == "latest" {
                        join_path(&["user", user, package, &self.name])
                    } else {
                        join_path(&["user", user, package, &self.name, &format_version(&self.version)])
                    }
                } else {
                    join_path(&["user", user, &self.name])
                }
            }
        }
    }

    /// Convert from MCP tool name back to ToolPath using standardized parsing
    pub fn from_mcp_name(mcp_name: &str) -> Result<Self> {
        let parts = parse_path(mcp_name);

        match parts.as_slice() {
            ["bin", name] => {
                if name.is_empty() || *name == "_" {
                    return Err(anyhow!("Tool name cannot be empty or just underscore"));
                }
                Ok(Self::bin(name.to_string()))
            }
            ["sbin", name] => {
                if name.is_empty() || *name == "_" {
                    return Err(anyhow!("Tool name cannot be empty or just underscore"));
                }
                Ok(Self::sbin(name.to_string()))
            }
            ["docs", name] => {
                if name.is_empty() || *name == "_" {
                    return Err(anyhow!("Tool name cannot be empty or just underscore"));
                }
                Ok(Self::docs(name.to_string()))
            }
            ["mcp", server, name] => {
                if name.is_empty() || server.is_empty() || *name == "_" || *server == "_" {
                    return Err(anyhow!("MCP tool name and server cannot be empty or just underscore"));
                }
                Ok(Self::mcp(server.to_string(), name.to_string(), "latest".to_string()))
            }
            ["mcp", server, name, version] if version.starts_with('v') => {
                if name.is_empty() || server.is_empty() || *name == "_" || *server == "_" {
                    return Err(anyhow!("MCP tool name and server cannot be empty or just underscore"));
                }
                Ok(Self::mcp(server.to_string(), name.to_string(), parse_version(version)))
            }
            ["user", user, name] => {
                if name.is_empty() || user.is_empty() || *name == "_" || *user == "_" {
                    return Err(anyhow!("User tool name and user cannot be empty or just underscore"));
                }
                Ok(Self {
                    namespace: Namespace::User(user.to_string()),
                    package: None,
                    name: name.to_string(),
                    version: "latest".to_string(),
                })
            }
            ["user", user, package, name] => {
                if name.is_empty() || user.is_empty() || package.is_empty() || *name == "_" || *user == "_" || *package == "_" {
                    return Err(anyhow!("User tool name, user, and package cannot be empty or just underscore"));
                }
                Ok(Self {
                    namespace: Namespace::User(user.to_string()),
                    package: Some(package.to_string()),
                    name: name.to_string(),
                    version: "latest".to_string(),
                })
            }
            ["user", user, package, name, version] if version.starts_with('v') => {
                if name.is_empty() || user.is_empty() || package.is_empty() || *name == "_" || *user == "_" || *package == "_" {
                    return Err(anyhow!("User tool name, user, and package cannot be empty or just underscore"));
                }
                Ok(Self {
                    namespace: Namespace::User(user.to_string()),
                    package: Some(package.to_string()),
                    name: name.to_string(),
                    version: parse_version(version),
                })
            }
            _ => Err(anyhow!("Invalid MCP path format: {}", mcp_name)),
        }
    }

    /// Check if this is a system tool (bin, sbin, or docs)
    pub fn is_system(&self) -> bool {
        matches!(
            self.namespace,
            Namespace::Bin | Namespace::Sbin | Namespace::Docs
        )
    }

    /// Check if this is an MCP server tool
    pub fn is_mcp(&self) -> bool {
        matches!(self.namespace, Namespace::Mcp(_))
    }
}

impl fmt::Display for ToolPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_mcp_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_paths() {
        assert_eq!(
            ToolPath::parse("bin__tcl_execute").unwrap(),
            ToolPath::bin("tcl_execute")
        );

        assert_eq!(
            ToolPath::parse("sbin__tcl_tool_add").unwrap(),
            ToolPath::sbin("tcl_tool_add")
        );

        assert_eq!(
            ToolPath::parse("user__alice__utils__reverse_string__v1_0").unwrap(),
            ToolPath::user("alice", "utils", "reverse_string", "1.0")
        );

        assert_eq!(
            ToolPath::parse("user__bob__math__calculate").unwrap(),
            ToolPath::user("bob", "math", "calculate", "latest")
        );
    }

    #[test]
    fn test_mcp_names() {
        assert_eq!(
            ToolPath::bin("tcl_execute").to_mcp_name(),
            "bin__tcl_execute"
        );

        assert_eq!(
            ToolPath::user("alice", "utils", "reverse_string", "1.0").to_mcp_name(),
            "user__alice__utils__reverse_string__v1_0"
        );

        assert_eq!(
            ToolPath::user("bob", "math", "calculate", "latest").to_mcp_name(),
            "user__bob__math__calculate"
        );
    }

    #[test]
    fn test_round_trip() {
        let paths = vec![
            ToolPath::bin("tcl_execute"),
            ToolPath::sbin("tcl_tool_add"),
            ToolPath::user("alice", "utils", "reverse_string", "1.0"),
            ToolPath::user("bob", "math", "calculate", "latest"),
        ];

        for path in paths {
            let mcp_name = path.to_mcp_name();
            let parsed = ToolPath::from_mcp_name(&mcp_name).unwrap();
            assert_eq!(path, parsed);
        }
    }
}
