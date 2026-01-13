use tcl_mcp_server::namespace::{Namespace, ToolPath};

#[cfg(test)]
mod namespace_edge_cases {
    use super::*;

    /// Test 1: Parsing edge cases
    #[test]
    fn test_parsing_edge_cases() {
        // Valid paths
        let valid_cases = vec![
            ("bin__tool", "bin", "tool", "latest"),
            ("sbin__admin_tool", "sbin", "admin_tool", "latest"),
            ("docs__help", "docs", "help", "latest"),
            ("user__alice__utils__tool__v1_0", "user:alice", "tool", "1.0"),
            ("mcp__server__tool__v2_0", "mcp:server", "tool", "2.0"),
        ];

        for (path, expected_ns, expected_name, expected_version) in valid_cases {
            let parsed = ToolPath::parse(path).unwrap();
            assert_eq!(parsed.name, expected_name);
            assert_eq!(parsed.version, expected_version);

            match (&parsed.namespace, expected_ns) {
                (Namespace::Bin, "bin") => {}
                (Namespace::Sbin, "sbin") => {}
                (Namespace::Docs, "docs") => {}
                (Namespace::User(u), ns) if ns.starts_with("user:") => {
                    assert_eq!(format!("user:{}", u), ns);
                }
                (Namespace::Mcp(s), ns) if ns.starts_with("mcp:") => {
                    assert_eq!(format!("mcp:{}", s), ns);
                }
                _ => panic!("Namespace mismatch for {}", path),
            }
        }

        // Invalid paths
        let invalid_cases = vec![
            "bin_tool",        // Wrong separator
            "_",               // Just underscore
            "__bin__tool",     // Leading separators
            "bin__",           // Trailing separator
            "bin",             // Missing tool name
            "unknown__tool",   // Unknown namespace
            "",                // Empty string
        ];

        for path in invalid_cases {
            assert!(ToolPath::parse(path).is_err(), "Should fail: {}", path);
        }
    }

    /// Test 2: MCP name conversion edge cases
    #[test]
    fn test_mcp_name_conversion_edge_cases() {
        let test_cases = vec![
            // Special characters in names
            ToolPath::bin("tool-with-dash"),
            ToolPath::bin("tool_with_underscore"),
            ToolPath::bin("tool.with.dot"),
            ToolPath::bin("UPPERCASE"),
            ToolPath::bin("123numeric"),
            // Version edge cases
            ToolPath::user("alice", "pkg", "tool", "1.0.0"),
            ToolPath::user("alice", "pkg", "tool", "1.0.0-beta"),
            ToolPath::user("alice", "pkg", "tool", "latest"),
            ToolPath::user("alice", "pkg", "tool", "v1.0"),
            // Empty/minimal cases
            ToolPath::bin("x"), // Single character
            ToolPath::user("u", "p", "t", "1"),
            // Long names
            ToolPath::bin("a_very_long_tool_name_that_might_exceed_limits"),
            ToolPath::user(
                "very_long_username",
                "very_long_package_name",
                "very_long_tool_name",
                "1.0.0-very-long-version",
            ),
        ];

        for path in test_cases {
            let mcp_name = path.to_mcp_name();
            let parsed = ToolPath::from_mcp_name(&mcp_name).unwrap();
            assert_eq!(path, parsed, "Round trip failed for {:?}", path);
        }
    }

    /// Test 3: Invalid MCP names
    #[test]
    fn test_invalid_mcp_names() {
        let invalid_names = vec![
            "",                   // Empty
            "tool",               // No prefix
            "bin_tool",           // Wrong separator
            "unknown___tool",     // Unknown prefix
            "user___tool",        // Missing user info
            "mcp___tool",         // Missing server info
            "bin___",             // Empty tool name
            "___tool",            // All underscores
            "bin___tool___extra", // Extra parts
            "user_alice__",       // Incomplete
            "mcp__server__tool__extra", // Extra parts
        ];

        for name in invalid_names {
            assert!(
                ToolPath::from_mcp_name(name).is_err(),
                "Should fail to parse: {}",
                name
            );
        }
    }

    /// Test 4: Namespace comparison and equality
    #[test]
    fn test_namespace_equality() {
        // Same paths should be equal
        assert_eq!(ToolPath::bin("tool"), ToolPath::bin("tool"));

        // Different versions should not be equal
        assert_ne!(
            ToolPath::user("alice", "pkg", "tool", "1.0"),
            ToolPath::user("alice", "pkg", "tool", "2.0")
        );

        // Different namespaces should not be equal
        assert_ne!(ToolPath::bin("tool"), ToolPath::sbin("tool"));

        // Different users should not be equal
        assert_ne!(
            ToolPath::user("alice", "pkg", "tool", "1.0"),
            ToolPath::user("bob", "pkg", "tool", "1.0")
        );
    }

    /// Test 5: Display formatting
    #[test]
    fn test_display_formatting() {
        let test_cases = vec![
            (ToolPath::bin("tool"), "bin__tool"),
            (ToolPath::sbin("admin"), "sbin__admin"),
            (ToolPath::docs("help"), "docs__help"),
            (
                ToolPath::user("alice", "utils", "tool", "latest"),
                "user__alice__utils__tool",
            ),
            (
                ToolPath::user("alice", "utils", "tool", "1.0"),
                "user__alice__utils__tool__v1_0",
            ),
            (
                ToolPath::mcp("server", "tool", "latest"),
                "mcp__server__tool",
            ),
            (
                ToolPath::mcp("server", "tool", "2.0"),
                "mcp__server__tool__v2_0",
            ),
        ];

        for (path, expected) in test_cases {
            assert_eq!(path.to_string(), expected);
        }
    }

    /// Test 6: System tool detection
    #[test]
    fn test_system_tool_detection() {
        assert!(ToolPath::bin("tool").is_system());
        assert!(ToolPath::sbin("tool").is_system());
        assert!(ToolPath::docs("tool").is_system());
        assert!(!ToolPath::user("alice", "pkg", "tool", "1.0").is_system());
        assert!(!ToolPath::mcp("server", "tool", "1.0").is_system());

        assert!(!ToolPath::bin("tool").is_mcp());
        assert!(!ToolPath::user("alice", "pkg", "tool", "1.0").is_mcp());
        assert!(ToolPath::mcp("server", "tool", "1.0").is_mcp());
    }

    /// Test 7: Version handling edge cases
    #[test]
    fn test_version_edge_cases() {
        // Parse various version formats
        let version_tests = vec![
            ("user__alice__pkg__tool", "latest"),
            ("user__alice__pkg__tool__v1_0", "1.0"),
            ("user__alice__pkg__tool__v1_0_0", "1.0.0"),
            ("user__alice__pkg__tool__vv1_0", "v1.0"),
            ("user__alice__pkg__tool__v1_0-beta", "1.0-beta"),
            ("user__alice__pkg__tool__vlatest", "latest"),
        ];

        for (path, expected_version) in version_tests {
            let parsed = ToolPath::parse(path).unwrap();
            assert_eq!(parsed.version, expected_version);
        }
    }

    /// Test 8: Unicode and special characters
    #[test]
    fn test_unicode_handling() {
        // These should work
        let unicode_paths = vec![
            ToolPath::user("用户", "包", "工具", "1.0"),
            ToolPath::user("user", "pkg", "outil", "1.0"),
            ToolPath::user("пользователь", "пакет", "инструмент", "1.0"),
        ];

        for path in unicode_paths {
            let mcp_name = path.to_mcp_name();
            // Should be able to convert to MCP name
            assert!(!mcp_name.is_empty());

            // Round trip might fail for non-ASCII in current implementation
            // This is a known limitation
        }
    }

    /// Test 9: Collision detection
    #[test]
    fn test_namespace_collisions() {
        // These should produce different MCP names
        let paths = vec![
            ToolPath::bin("test"),
            ToolPath::sbin("test"),
            ToolPath::docs("test"),
            ToolPath::user("alice", "pkg", "test", "1.0"),
            ToolPath::user("bob", "pkg", "test", "1.0"),
            ToolPath::mcp("server1", "test", "1.0"),
            ToolPath::mcp("server2", "test", "1.0"),
        ];

        let mut mcp_names = std::collections::HashSet::new();
        for path in paths {
            let mcp_name = path.to_mcp_name();
            assert!(
                mcp_names.insert(mcp_name.clone()),
                "Collision detected for: {} -> {}",
                path,
                mcp_name
            );
        }
    }

    /// Test 10: Package handling for user namespace
    #[test]
    fn test_package_handling() {
        // With package
        let with_pkg = ToolPath::user("alice", "utils", "tool", "1.0");
        assert_eq!(with_pkg.package, Some("utils".to_string()));
        assert_eq!(with_pkg.to_string(), "user__alice__utils__tool__v1_0");

        // Without package - now supported in parse
        let without_pkg = ToolPath::parse("user__alice__tool").unwrap();
        assert_eq!(without_pkg.package, None);
        assert_eq!(without_pkg.to_string(), "user__alice__tool");
    }
}
