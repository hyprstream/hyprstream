use crate::path_format::{ParsedPath, PathPattern};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fmt;

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

    /// Parse a tool path from a string representation
    /// Examples:
    /// - "/bin/tcl_execute"
    /// - "/sbin/tcl_tool_add"
    /// - "/mcp/filesystem/read_file:1.0"
    /// - "/alice/utils/reverse_string:1.0"
    /// - "/bob/math/calculate:latest"
    pub fn parse(path: &str) -> Result<Self> {
        if !path.starts_with('/') {
            return Err(anyhow!("Tool path must start with '/'"));
        }

        let parts: Vec<&str> = path[1..].split('/').collect();

        match parts.as_slice() {
            ["bin", name] => {
                let (name, _version) = Self::parse_name_version(name)?;
                if name.is_empty() {
                    return Err(anyhow!("Tool name cannot be empty"));
                }
                Ok(Self::bin(name))
            }
            ["sbin", name] => {
                let (name, _version) = Self::parse_name_version(name)?;
                if name.is_empty() {
                    return Err(anyhow!("Tool name cannot be empty"));
                }
                Ok(Self::sbin(name))
            }
            ["docs", name] => {
                let (name, _version) = Self::parse_name_version(name)?;
                if name.is_empty() {
                    return Err(anyhow!("Tool name cannot be empty"));
                }
                Ok(Self::docs(name))
            }
            ["mcp", server, name_version] => {
                let (name, version) = Self::parse_name_version(name_version)?;
                if name.is_empty() || server.is_empty() {
                    return Err(anyhow!("MCP server and tool name cannot be empty"));
                }
                Ok(Self::mcp(server.to_string(), name, version))
            }
            [user, package, name_version] => {
                if user.is_empty() {
                    return Err(anyhow!("User namespace cannot have empty user name"));
                }
                let (name, version) = Self::parse_name_version(name_version)?;
                if name.is_empty() || package.is_empty() {
                    return Err(anyhow!("User package and tool name cannot be empty"));
                }
                Ok(Self::user(
                    user.to_string(),
                    package.to_string(),
                    name,
                    version,
                ))
            }
            _ => Err(anyhow!("Invalid tool path format: {}", path)),
        }
    }

    /// Parse name:version format
    fn parse_name_version(s: &str) -> Result<(String, String)> {
        if let Some((name, version)) = s.split_once(':') {
            Ok((name.to_string(), version.to_string()))
        } else {
            Ok((s.to_string(), "latest".to_string()))
        }
    }

    /// Convert to MCP-compatible tool name using standardized path format
    pub fn to_mcp_name(&self) -> String {
        match &self.namespace {
            Namespace::Bin => PathPattern::bin(&self.name),
            Namespace::Sbin => PathPattern::sbin(&self.name),
            Namespace::Docs => PathPattern::docs(&self.name),
            Namespace::Mcp(server) => {
                if self.version == "latest" {
                    PathPattern::mcp(server, &self.name)
                } else {
                    PathPattern::mcp_versioned(server, &self.name, &self.version)
                }
            }
            Namespace::User(user) => {
                if let Some(package) = &self.package {
                    if self.version == "latest" {
                        PathPattern::user(user, Some(package), &self.name)
                    } else {
                        PathPattern::user_versioned(user, package, &self.name, &self.version)
                    }
                } else {
                    PathPattern::user(user, None, &self.name)
                }
            }
        }
    }

    /// Convert from MCP tool name back to ToolPath using standardized parsing
    pub fn from_mcp_name(mcp_name: &str) -> Result<Self> {
        let parsed = ParsedPath::parse(mcp_name)?;

        match parsed.namespace.as_str() {
            "bin" => Ok(Self::bin(parsed.name)),
            "sbin" => Ok(Self::sbin(parsed.name)),
            "docs" => Ok(Self::docs(parsed.name)),
            "mcp" => {
                let server = parsed
                    .server
                    .ok_or_else(|| anyhow!("MCP namespace requires server"))?;
                let version = parsed.version.unwrap_or_else(|| "latest".to_string());
                Ok(Self::mcp(server, parsed.name, version))
            }
            "user" => {
                let user = parsed
                    .user
                    .ok_or_else(|| anyhow!("User namespace requires user"))?;
                let version = parsed.version.unwrap_or_else(|| "latest".to_string());
                Ok(Self {
                    namespace: Namespace::User(user),
                    package: parsed.package,
                    name: parsed.name,
                    version,
                })
            }
            _ => Err(anyhow!("Unknown namespace: {}", parsed.namespace)),
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
        match &self.namespace {
            Namespace::Bin => write!(f, "/bin/{}", self.name),
            Namespace::Sbin => write!(f, "/sbin/{}", self.name),
            Namespace::Docs => write!(f, "/docs/{}", self.name),
            Namespace::Mcp(server) => {
                if self.version == "latest" {
                    write!(f, "/mcp/{}/{}", server, self.name)
                } else {
                    write!(f, "/mcp/{}/{}:{}", server, self.name, self.version)
                }
            }
            Namespace::User(user) => {
                if let Some(package) = &self.package {
                    if self.version == "latest" {
                        write!(f, "/{}/{}/{}", user, package, self.name)
                    } else {
                        write!(f, "/{}/{}/{}:{}", user, package, self.name, self.version)
                    }
                } else {
                    write!(f, "/{}/{}", user, self.name)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_paths() {
        assert_eq!(
            ToolPath::parse("/bin/tcl_execute").unwrap(),
            ToolPath::bin("tcl_execute")
        );

        assert_eq!(
            ToolPath::parse("/sbin/tcl_tool_add").unwrap(),
            ToolPath::sbin("tcl_tool_add")
        );

        assert_eq!(
            ToolPath::parse("/alice/utils/reverse_string:1.0").unwrap(),
            ToolPath::user("alice", "utils", "reverse_string", "1.0")
        );

        assert_eq!(
            ToolPath::parse("/bob/math/calculate").unwrap(),
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
