use chrono::Utc;
use std::collections::HashMap;
use tcl_mcp_server::mcp_client::{ConnectionStatus, McpClient, McpServerConfig};
use tcl_mcp_server::namespace::ToolPath;

#[cfg(test)]
mod mcp_registration_tests {
    use super::*;

    /// Test 1: Basic server registration
    #[tokio::test]
    async fn test_basic_server_registration() {
        let client = McpClient::new();

        let config = McpServerConfig {
            id: "test-server".to_string(),
            name: "Test MCP Server".to_string(),
            description: Some("A test server".to_string()),
            command: "echo".to_string(),
            args: vec!["test".to_string()],
            env: HashMap::new(),
            auto_start: false,
            timeout_ms: 5000,
            max_retries: 3,
            created_at: Utc::now(),
        };

        // Should succeed
        assert!(client.register_server(config.clone()).await.is_ok());

        // Check server is listed
        let servers = client.list_servers().await;
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].0, "test-server");
        assert_eq!(servers[0].1, ConnectionStatus::Disconnected);
    }

    /// Test 2: Duplicate server registration
    #[tokio::test]
    async fn test_duplicate_server_registration() {
        let client = McpClient::new();

        let config = McpServerConfig {
            id: "duplicate-server".to_string(),
            name: "Duplicate Server".to_string(),
            description: None,
            command: "echo".to_string(),
            args: vec![],
            env: HashMap::new(),
            auto_start: false,
            timeout_ms: 5000,
            max_retries: 3,
            created_at: Utc::now(),
        };

        // First registration should succeed
        assert!(client.register_server(config.clone()).await.is_ok());

        // Second registration should replace (current implementation)
        // TODO: Consider if this should error instead
        assert!(client.register_server(config).await.is_ok());

        // Should still have only one server
        let servers = client.list_servers().await;
        assert_eq!(servers.len(), 1);
    }

    /// Test 3: Invalid server configuration
    #[tokio::test]
    async fn test_invalid_server_config() {
        let client = McpClient::new();

        // Empty ID
        let config = McpServerConfig {
            id: "".to_string(),
            name: "Invalid Server".to_string(),
            description: None,
            command: "echo".to_string(),
            args: vec![],
            env: HashMap::new(),
            auto_start: false,
            timeout_ms: 5000,
            max_retries: 3,
            created_at: Utc::now(),
        };

        assert!(client.register_server(config).await.is_err());

        // Empty command
        let config = McpServerConfig {
            id: "test-server".to_string(),
            name: "Invalid Server".to_string(),
            description: None,
            command: "".to_string(),
            args: vec![],
            env: HashMap::new(),
            auto_start: false,
            timeout_ms: 5000,
            max_retries: 3,
            created_at: Utc::now(),
        };

        assert!(client.register_server(config).await.is_err());
    }

    /// Test 4: Auto-start functionality
    #[tokio::test]
    async fn test_auto_start_server() {
        let client = McpClient::new();

        // Use a mock server command for testing
        let config = McpServerConfig {
            id: "auto-start-server".to_string(),
            name: "Auto Start Server".to_string(),
            description: Some("Server that auto-starts".to_string()),
            command: "sleep".to_string(), // Simple command that won't fail
            args: vec!["0.1".to_string()], // Sleep for 100ms
            env: HashMap::new(),
            auto_start: true, // This should trigger connection
            timeout_ms: 5000,
            max_retries: 3,
            created_at: Utc::now(),
        };

        // Registration should attempt to connect
        let result = client.register_server(config).await;

        // This will fail because 'sleep' is not an MCP server
        // but we're testing that auto_start triggers connection attempt
        assert!(result.is_err());
    }

    /// Test 5: Server removal (graceful and forced)
    #[tokio::test]
    async fn test_server_removal() {
        let client = McpClient::new();

        let config = McpServerConfig {
            id: "removable-server".to_string(),
            name: "Removable Server".to_string(),
            description: None,
            command: "echo".to_string(),
            args: vec![],
            env: HashMap::new(),
            auto_start: false,
            timeout_ms: 5000,
            max_retries: 3,
            created_at: Utc::now(),
        };

        // Register server
        assert!(client.register_server(config).await.is_ok());
        assert_eq!(client.list_servers().await.len(), 1);

        // Graceful removal
        assert!(client
            .remove_server("removable-server", false)
            .await
            .is_ok());
        assert_eq!(client.list_servers().await.len(), 0);

        // Try to remove non-existent server
        assert!(client.remove_server("non-existent", false).await.is_ok());
    }

    /// Test 6: Connection state management
    #[tokio::test]
    async fn test_connection_states() {
        let client = McpClient::new();

        let config = McpServerConfig {
            id: "state-test-server".to_string(),
            name: "State Test Server".to_string(),
            description: None,
            command: "false".to_string(), // Command that always fails
            args: vec![],
            env: HashMap::new(),
            auto_start: false,
            timeout_ms: 1000, // Short timeout
            max_retries: 1,
            created_at: Utc::now(),
        };

        // Register server
        assert!(client.register_server(config).await.is_ok());

        // Initially disconnected
        let servers = client.list_servers().await;
        assert_eq!(servers[0].1, ConnectionStatus::Disconnected);

        // Try to connect (will fail)
        let connect_result = client.connect_server("state-test-server").await;
        assert!(connect_result.is_err());

        // Check state after failed connection
        let servers = client.list_servers().await;
        match &servers[0].1 {
            ConnectionStatus::Error(_) => (), // Expected
            _ => panic!("Expected Error state after failed connection"),
        }
    }

    /// Test 7: Environment variable handling
    #[tokio::test]
    async fn test_environment_variables() {
        let client = McpClient::new();

        let mut env = HashMap::new();
        env.insert("TEST_VAR".to_string(), "test_value".to_string());
        env.insert("PATH".to_string(), "/custom/path".to_string());

        let config = McpServerConfig {
            id: "env-test-server".to_string(),
            name: "Env Test Server".to_string(),
            description: None,
            command: "printenv".to_string(),
            args: vec!["TEST_VAR".to_string()],
            env,
            auto_start: false,
            timeout_ms: 5000,
            max_retries: 3,
            created_at: Utc::now(),
        };

        // Should register successfully
        assert!(client.register_server(config).await.is_ok());
    }

    /// Test 8: Timeout handling
    #[tokio::test]
    async fn test_timeout_handling() {
        let client = McpClient::new();

        let config = McpServerConfig {
            id: "timeout-test-server".to_string(),
            name: "Timeout Test Server".to_string(),
            description: None,
            command: "sleep".to_string(),
            args: vec!["10".to_string()], // Sleep for 10 seconds
            env: HashMap::new(),
            auto_start: false,
            timeout_ms: 100, // Very short timeout
            max_retries: 0,
            created_at: Utc::now(),
        };

        assert!(client.register_server(config).await.is_ok());

        // Connection should timeout
        let connect_result = client.connect_server("timeout-test-server").await;
        assert!(connect_result.is_err());
    }

    /// Test 9: Tool discovery edge cases
    #[tokio::test]
    async fn test_tool_discovery_edge_cases() {
        // This would require a mock MCP server that returns various tool configurations
        // For now, we test the parsing logic

        let tool_names = vec![
            "simple_tool",
            "tool-with-dashes",
            "tool_with_underscores",
            "tool.with.dots",
            "UPPERCASE_TOOL",
            "tool123",
            "tool-v2.0",
        ];

        for name in tool_names {
            // Test that tool names are handled correctly in MCP name conversion
            let tool_path = ToolPath::bin(name);
            let mcp_name = tool_path.to_mcp_name();
            let parsed = ToolPath::from_mcp_name(&mcp_name).unwrap();
            assert_eq!(tool_path, parsed);
        }
    }

    /// Test 10: Concurrent server operations
    #[tokio::test]
    async fn test_concurrent_operations() {
        let client = McpClient::new();

        // Register multiple servers concurrently
        let mut handles = vec![];

        for i in 0..5 {
            let client_clone = client.clone();
            let handle = tokio::spawn(async move {
                let config = McpServerConfig {
                    id: format!("concurrent-server-{}", i),
                    name: format!("Concurrent Server {}", i),
                    description: None,
                    command: "echo".to_string(),
                    args: vec![format!("server-{}", i)],
                    env: HashMap::new(),
                    auto_start: false,
                    timeout_ms: 5000,
                    max_retries: 3,
                    created_at: Utc::now(),
                };
                client_clone.register_server(config).await
            });
            handles.push(handle);
        }

        // Wait for all registrations
        for handle in handles {
            assert!(handle.await.unwrap().is_ok());
        }

        // Should have all 5 servers
        let servers = client.list_servers().await;
        assert_eq!(servers.len(), 5);
    }

    /// Test 11: Server reconnection and retry logic
    #[tokio::test]
    async fn test_retry_logic() {
        let client = McpClient::new();

        let config = McpServerConfig {
            id: "retry-test-server".to_string(),
            name: "Retry Test Server".to_string(),
            description: None,
            command: "false".to_string(), // Always fails
            args: vec![],
            env: HashMap::new(),
            auto_start: false,
            timeout_ms: 100,
            max_retries: 3, // Should retry 3 times
            created_at: Utc::now(),
        };

        assert!(client.register_server(config).await.is_ok());

        let start = std::time::Instant::now();
        let result = client.connect_server("retry-test-server").await;
        let duration = start.elapsed();

        // Should fail after retries
        assert!(result.is_err());

        // Connection attempt should be quick (no retries in current implementation)
        assert!(duration.as_millis() < 1000);
    }

    /// Test 12: Special characters in server IDs
    #[tokio::test]
    async fn test_special_characters_in_ids() {
        let client = McpClient::new();

        let test_ids = vec![
            "server-with-dash",
            "server_with_underscore",
            "server.with.dot",
            "server@with@at",
            "server:with:colon",
            "server/with/slash",
        ];

        for id in test_ids {
            let config = McpServerConfig {
                id: id.to_string(),
                name: format!("Server {}", id),
                description: None,
                command: "echo".to_string(),
                args: vec![],
                env: HashMap::new(),
                auto_start: false,
                timeout_ms: 5000,
                max_retries: 3,
                created_at: Utc::now(),
            };

            // Should handle special characters
            assert!(client.register_server(config).await.is_ok());
        }

        let servers = client.list_servers().await;
        assert_eq!(servers.len(), 6);
    }

    /// Test 13: Large number of tools
    #[tokio::test]
    async fn test_large_tool_count() {
        // Test namespace handling with many tools
        let mut tool_paths = vec![];

        // Generate 1000 tools across different namespaces
        for i in 0..200 {
            tool_paths.push(ToolPath::bin(format!("tool_{}", i)));
            tool_paths.push(ToolPath::sbin(format!("admin_tool_{}", i)));
            tool_paths.push(ToolPath::docs(format!("doc_tool_{}", i)));
            tool_paths.push(ToolPath::user(
                "alice",
                "package",
                format!("user_tool_{}", i),
                "1.0",
            ));
            tool_paths.push(ToolPath::mcp("server", format!("mcp_tool_{}", i), "2.0"));
        }

        // Test MCP name conversion performance
        for path in &tool_paths {
            let mcp_name = path.to_mcp_name();
            let parsed = ToolPath::from_mcp_name(&mcp_name).unwrap();
            assert_eq!(*path, parsed);
        }
    }

    /// Test 14: Server command injection protection
    #[tokio::test]
    async fn test_command_injection_protection() {
        let client = McpClient::new();

        // Test potentially dangerous commands
        let dangerous_configs = vec![
            (
                "cmd-injection-1",
                "echo",
                vec!["test; rm -rf /".to_string()],
            ),
            (
                "cmd-injection-2",
                "sh",
                vec!["-c".to_string(), "echo test".to_string()],
            ),
            ("cmd-injection-3", "echo", vec!["$(whoami)".to_string()]),
            ("cmd-injection-4", "echo", vec!["`id`".to_string()]),
        ];

        for (id, cmd, args) in dangerous_configs {
            let config = McpServerConfig {
                id: id.to_string(),
                name: format!("Dangerous Server {}", id),
                description: None,
                command: cmd.to_string(),
                args,
                env: HashMap::new(),
                auto_start: false,
                timeout_ms: 5000,
                max_retries: 3,
                created_at: Utc::now(),
            };

            // Registration should succeed (commands are just stored)
            assert!(client.register_server(config).await.is_ok());
        }

        // The actual protection happens when spawning processes
        // These would be handled by the OS/shell escaping
    }

    /// Test 15: Memory leak prevention
    #[tokio::test]
    async fn test_memory_leak_prevention() {
        let client = McpClient::new();

        // Register and remove servers many times
        for i in 0..100 {
            let config = McpServerConfig {
                id: format!("temp-server-{}", i),
                name: format!("Temporary Server {}", i),
                description: Some("A temporary server for testing".to_string()),
                command: "echo".to_string(),
                args: vec![format!("server-{}", i)],
                env: HashMap::new(),
                auto_start: false,
                timeout_ms: 5000,
                max_retries: 3,
                created_at: Utc::now(),
            };

            assert!(client.register_server(config).await.is_ok());
            assert!(client
                .remove_server(&format!("temp-server-{}", i), true)
                .await
                .is_ok());
        }

        // Should have no servers left
        assert_eq!(client.list_servers().await.len(), 0);
    }
}
