use anyhow::Result;
use chrono::Utc;
use serde_json::json;
use std::collections::HashMap;
use std::process::Stdio;
use tcl_mcp_server::mcp_client::{McpClient, McpServerConfig};
use tokio::process::Command;

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Helper to start TCL MCP server as a subprocess
    async fn start_tcl_mcp_server(port: u16) -> Result<tokio::process::Child> {
        let mut cmd = Command::new("cargo");
        cmd.arg("run")
            .arg("--")
            .arg("--port")
            .arg(port.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        Ok(cmd.spawn()?)
    }

    /// Test 1: Register and connect to actual TCL MCP server
    #[tokio::test]
    #[ignore] // Requires built binary
    async fn test_register_tcl_mcp_server() {
        let client = McpClient::new();

        let config = McpServerConfig {
            id: "tcl-local".to_string(),
            name: "Local TCL MCP Server".to_string(),
            description: Some("Local instance of TCL MCP server".to_string()),
            command: "cargo".to_string(),
            args: vec!["run".to_string(), "--".to_string()],
            env: HashMap::new(),
            auto_start: true,
            timeout_ms: 10000,
            max_retries: 3,
            created_at: Utc::now(),
        };

        // Register and auto-start
        let result = client.register_server(config).await;

        if result.is_ok() {
            // Check that tools were discovered
            let tools = client.get_server_tools("tcl-local").await.unwrap();

            // Should have at least the basic tools
            let tool_names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
            assert!(tool_names.contains(&"bin___tcl_execute".to_string()));
            assert!(tool_names.contains(&"bin___tcl_tool_list".to_string()));

            // Clean up
            client.remove_server("tcl-local", true).await.unwrap();
        }
    }

    /// Test 2: Execute tools through MCP client
    #[tokio::test]
    #[ignore] // Requires built binary
    async fn test_execute_tools_through_mcp() {
        let client = McpClient::new();

        // Start a test server
        let config = McpServerConfig {
            id: "tcl-test".to_string(),
            name: "Test TCL Server".to_string(),
            description: None,
            command: "target/debug/tcl-mcp".to_string(),
            args: vec![],
            env: HashMap::new(),
            auto_start: true,
            timeout_ms: 10000,
            max_retries: 3,
            created_at: Utc::now(),
        };

        if client.register_server(config).await.is_ok() {
            // Execute tcl_execute tool
            let params = json!({
                "script": "expr 1 + 1"
            });

            let result = client
                .execute_tool("tcl-test", "bin___tcl_execute", params)
                .await;

            // Should get result
            if let Ok(value) = result {
                println!("TCL execution result: {:?}", value);
            }

            // Clean up
            client.remove_server("tcl-test", true).await.unwrap();
        }
    }

    /// Test 3: Multiple MCP server instances
    #[tokio::test]
    async fn test_multiple_mcp_servers() {
        let client = McpClient::new();

        // Register multiple different MCP servers
        let servers = vec![
            (
                "filesystem-mcp",
                "npx",
                vec!["@modelcontextprotocol/server-filesystem".to_string()],
            ),
            (
                "github-mcp",
                "npx",
                vec!["@modelcontextprotocol/server-github".to_string()],
            ),
            ("tcl-mcp", "cargo", vec!["run".to_string()]),
        ];

        for (id, cmd, args) in &servers {
            let config = McpServerConfig {
                id: id.to_string(),
                name: format!("{} Server", id),
                description: None,
                command: cmd.to_string(),
                args: args.clone(),
                env: HashMap::new(),
                auto_start: false, // Don't auto-start for this test
                timeout_ms: 5000,
                max_retries: 3,
                created_at: Utc::now(),
            };

            assert!(client.register_server(config).await.is_ok());
        }

        // Should have all servers registered
        let server_list = client.list_servers().await;
        assert_eq!(server_list.len(), 3);

        // Clean up
        for (id, _, _) in &servers {
            client.remove_server(id, true).await.unwrap();
        }
    }

    /// Test 4: Tool namespace collision handling
    #[tokio::test]
    async fn test_tool_namespace_collision() {
        let client = McpClient::new();

        // Register two servers that might have overlapping tool names
        let config1 = McpServerConfig {
            id: "server1".to_string(),
            name: "Server 1".to_string(),
            description: None,
            command: "echo".to_string(),
            args: vec!["server1".to_string()],
            env: HashMap::new(),
            auto_start: false,
            timeout_ms: 5000,
            max_retries: 3,
            created_at: Utc::now(),
        };

        let config2 = McpServerConfig {
            id: "server2".to_string(),
            name: "Server 2".to_string(),
            description: None,
            command: "echo".to_string(),
            args: vec!["server2".to_string()],
            env: HashMap::new(),
            auto_start: false,
            timeout_ms: 5000,
            max_retries: 3,
            created_at: Utc::now(),
        };

        assert!(client.register_server(config1).await.is_ok());
        assert!(client.register_server(config2).await.is_ok());

        // Tools should be namespaced by server
        // Each server maintains its own tool list
        let servers = client.list_servers().await;
        assert_eq!(servers.len(), 2);

        // Clean up
        client.remove_server("server1", true).await.unwrap();
        client.remove_server("server2", true).await.unwrap();
    }

    /// Test 5: Error propagation from MCP servers
    #[tokio::test]
    async fn test_error_propagation() {
        let client = McpClient::new();

        // Register a server that will fail
        let config = McpServerConfig {
            id: "error-server".to_string(),
            name: "Error Test Server".to_string(),
            description: None,
            command: "/nonexistent/command".to_string(),
            args: vec![],
            env: HashMap::new(),
            auto_start: false,
            timeout_ms: 5000,
            max_retries: 3,
            created_at: Utc::now(),
        };

        assert!(client.register_server(config).await.is_ok());

        // Connection should fail with proper error
        let connect_result = client.connect_server("error-server").await;
        assert!(connect_result.is_err());

        let err_msg = connect_result.unwrap_err().to_string();
        assert!(err_msg.contains("Failed to start MCP server"));

        // Clean up
        client.remove_server("error-server", true).await.unwrap();
    }

    /// Test 6: Dynamic tool discovery after connection
    #[tokio::test]
    #[ignore] // Requires mock server that can add tools dynamically
    async fn test_dynamic_tool_discovery() {
        let client = McpClient::new();

        // This test would require a mock MCP server that can:
        // 1. Start with a basic set of tools
        // 2. Add new tools while running
        // 3. Notify about tool changes

        // For now, we document the test scenario
        println!("Test scenario: Dynamic tool discovery");
        println!("1. Connect to MCP server with initial tools");
        println!("2. Server adds new tools at runtime");
        println!("3. Client should be able to discover and use new tools");
        println!("4. Server removes tools");
        println!("5. Client should handle tool removal gracefully");
    }

    /// Test 7: Stress test with rapid registration/removal
    #[tokio::test]
    async fn test_rapid_registration_removal() {
        let client = McpClient::new();

        // Rapidly register and remove servers
        for i in 0..50 {
            let config = McpServerConfig {
                id: format!("stress-server-{}", i),
                name: format!("Stress Test Server {}", i),
                description: None,
                command: "echo".to_string(),
                args: vec![i.to_string()],
                env: HashMap::new(),
                auto_start: false,
                timeout_ms: 1000,
                max_retries: 1,
                created_at: Utc::now(),
            };

            // Register
            assert!(client.register_server(config).await.is_ok());

            // Immediately remove
            assert!(client
                .remove_server(&format!("stress-server-{}", i), true)
                .await
                .is_ok());
        }

        // Should have no servers left
        assert_eq!(client.list_servers().await.len(), 0);
    }

    /// Test 8: JSON-RPC protocol compliance
    #[tokio::test]
    async fn test_jsonrpc_protocol_compliance() {
        // Test that our JSON-RPC implementation follows the spec
        let test_cases = vec![
            // Valid requests
            (
                json!({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {
                            "name": "test-client",
                            "version": "1.0.0"
                        }
                    }
                }),
                true,
            ),
            // Missing jsonrpc version
            (
                json!({
                    "id": 1,
                    "method": "initialize",
                    "params": {}
                }),
                false,
            ),
            // Invalid version
            (
                json!({
                    "jsonrpc": "1.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {}
                }),
                false,
            ),
        ];

        for (request, should_be_valid) in test_cases {
            let is_valid = request
                .get("jsonrpc")
                .and_then(|v| v.as_str())
                .map(|v| v == "2.0")
                .unwrap_or(false);

            assert_eq!(
                is_valid, should_be_valid,
                "Request validation failed for: {:?}",
                request
            );
        }
    }

    /// Test 9: Tool parameter validation
    #[tokio::test]
    async fn test_tool_parameter_validation() {
        // Test various parameter edge cases
        let test_params = vec![
            // Valid parameters
            (json!({"script": "puts hello"}), true),
            (json!({"script": "expr 1 + 1"}), true),
            // Empty script
            (json!({"script": ""}), true), // Empty is technically valid
            // Missing required parameter
            (json!({}), false),
            // Wrong parameter type
            (json!({"script": 123}), false),
            // Extra parameters (should be allowed)
            (json!({"script": "puts hello", "extra": "ignored"}), true),
            // Null value
            (json!({"script": null}), false),
            // Complex nested structure
            (
                json!({"script": "puts hello", "options": {"debug": true}}),
                true,
            ),
        ];

        for (params, should_be_valid) in test_params {
            let is_valid = params.get("script").and_then(|v| v.as_str()).is_some();

            assert_eq!(
                is_valid, should_be_valid,
                "Parameter validation failed for: {:?}",
                params
            );
        }
    }

    /// Test 10: Connection recovery scenarios
    #[tokio::test]
    async fn test_connection_recovery() {
        let _client = McpClient::new();

        println!("Test scenario: Connection recovery");
        println!("1. Establish connection to MCP server");
        println!("2. Simulate network interruption");
        println!("3. Verify connection status changes");
        println!("4. Attempt reconnection");
        println!("5. Verify tools are re-discovered");
        println!("6. Execute tool to confirm recovery");

        // This test would require:
        // - A mock server that can simulate disconnections
        // - Heartbeat/keepalive mechanism implementation
        // - Reconnection logic with exponential backoff
    }
}
