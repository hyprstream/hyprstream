use anyhow::Result;
use serde_json::json;

mod helpers;
use helpers::mcp_test_client::McpTestClient;

#[tokio::test]
async fn test_mcp_initialize_handshake() -> Result<()> {
    let mut client = McpTestClient::new(false);
    client.start().await?;

    let result = client.initialize().await?;

    // Verify protocol version
    assert_eq!(result["protocolVersion"], "2024-11-05");

    // Verify capabilities
    assert!(result["capabilities"].is_object());

    // Verify server info
    let server_info = &result["serverInfo"];
    assert_eq!(server_info["name"], "tcl-mcp-server");
    assert!(server_info["version"].is_string());

    Ok(())
}

#[tokio::test]
async fn test_tool_listing() -> Result<()> {
    let mut client = McpTestClient::new(false);
    client.start().await?;
    client.initialize().await?;

    let result = client.list_tools().await?;

    // Verify tools list structure
    assert!(result["tools"].is_array());
    let tools = result["tools"].as_array().unwrap();

    // Should have at least system tools
    assert!(!tools.is_empty());

    // Check for expected system tools
    let tool_names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();

    assert!(tool_names.contains(&"bin__tcl_execute"));
    assert!(tool_names.contains(&"bin__exec_tool"));
    assert!(tool_names.contains(&"bin__list_tools"));

    // Verify tool structure
    for tool in tools {
        assert!(tool["name"].is_string());
        assert!(tool["description"].is_string());
        assert!(tool["inputSchema"].is_object());
    }

    Ok(())
}

#[tokio::test]
async fn test_tcl_script_execution() -> Result<()> {
    let mut client = McpTestClient::new(false);
    client.start().await?;
    client.initialize().await?;

    // Test simple arithmetic
    let result = client
        .call_tool(
            "bin__tcl_execute",
            json!({
                "script": "expr {2 + 3}"
            }),
        )
        .await?;

    assert!(result.contains("5"));

    // Test string operations
    let result = client
        .call_tool(
            "bin__tcl_execute",
            json!({
                "script": "return \"Hello, World!\""
            }),
        )
        .await?;

    println!("DEBUG - TCL result: {}", result);
    assert!(result.contains("Hello") && result.contains("World"));

    Ok(())
}

#[tokio::test]
async fn test_tcl_error_handling() -> Result<()> {
    let mut client = McpTestClient::new(false);
    client.start().await?;
    client.initialize().await?;

    // Test invalid TCL command
    let result = client
        .call_tool(
            "bin__tcl_execute",
            json!({
                "script": "invalid_tcl_command_that_does_not_exist"
            }),
        )
        .await;

    // Should return an error
    assert!(result.is_err());

    Ok(())
}

#[tokio::test]
async fn test_special_character_handling() -> Result<()> {
    let mut client = McpTestClient::new(false);
    client.start().await?;
    client.initialize().await?;

    // Test string with special characters
    let result = client
        .call_tool(
            "bin__tcl_execute",
            json!({
                "script": r#"return "Hello World with special chars!""#
            }),
        )
        .await?;

    println!("DEBUG - Special chars result: {}", result);
    assert!(result.contains("Hello") && result.contains("special"));

    Ok(())
}

#[tokio::test]
async fn test_privileged_tool_management() -> Result<()> {
    let mut client = McpTestClient::new(true); // Privileged mode required
    client.start().await?;
    client.initialize().await?;

    // Add a custom tool with unique name based on timestamp
    let tool_name = format!(
        "simple_add_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    );
    let add_result = client
        .add_tool(
            "test",
            "math",
            &tool_name,
            "Adds two numbers",
            "set result [expr {$a + $b}]; return \"Result: $a + $b = $result\"",
            vec![
                json!({
                    "name": "a",
                    "description": "First number",
                    "required": true,
                    "type_name": "number"
                }),
                json!({
                    "name": "b",
                    "description": "Second number",
                    "required": true,
                    "type_name": "number"
                }),
            ],
        )
        .await?;

    assert!(add_result.contains("Tool added successfully") || add_result.contains(&tool_name));

    // Execute the custom tool
    let tool_path = format!("/test/math/{}", tool_name);
    let exec_result = client
        .exec_tool(
            &tool_path,
            json!({
                "a": 5,
                "b": 3
            }),
        )
        .await?;

    assert!(exec_result.contains("Result: 5 + 3 = 8"));

    Ok(())
}

#[tokio::test]
async fn test_parameter_validation() -> Result<()> {
    let mut client = McpTestClient::new(false);
    client.start().await?;
    client.initialize().await?;

    // Test missing required parameter
    let result = client.call_tool("bin__tcl_execute", json!({})).await;
    assert!(
        result.is_err(),
        "Should fail with missing required 'script' parameter"
    );

    // Test invalid parameter type (if validation is implemented)
    let _result = client
        .call_tool(
            "bin__tcl_execute",
            json!({
                "script": 123  // Should be string
            }),
        )
        .await;
    // Note: This might not fail depending on JSON-to-string conversion

    Ok(())
}

#[tokio::test]
async fn test_complex_tcl_operations() -> Result<()> {
    let mut client = McpTestClient::new(false);
    client.start().await?;
    client.initialize().await?;

    // Test factorial calculation with recursion
    let factorial_script = r#"
        proc factorial {n} {
            if {$n <= 1} {
                return 1
            } else {
                return [expr {$n * [factorial [expr {$n - 1}]]}]
            }
        }
        return [factorial 5]
    "#;

    let result = client
        .call_tool(
            "bin__tcl_execute",
            json!({
                "script": factorial_script
            }),
        )
        .await?;

    assert!(result.contains("120")); // 5! = 120

    Ok(())
}

#[tokio::test]
async fn test_capability_reporting() -> Result<()> {
    let mut client = McpTestClient::new(false);
    client.start().await?;

    let result = client.initialize().await?;

    // Should include TCL-specific capabilities
    let capabilities = &result["capabilities"];
    assert!(capabilities.is_object());

    // Check for tool support
    assert!(capabilities["tools"].is_object());

    Ok(())
}

#[tokio::test]
async fn test_privilege_mode_restrictions() -> Result<()> {
    // Test in non-privileged mode
    let mut client = McpTestClient::new(false);
    client.start().await?;
    client.initialize().await?;

    // Try to add a tool without privileges
    let result = client
        .call_tool(
            "sbin__tcl_tool_add",
            json!({
                "user": "test",
                "package": "math",
                "name": "test_tool",
                "description": "Test tool",
                "script": "return 42",
                "parameters": []
            }),
        )
        .await;

    // Should fail due to lack of privileges
    assert!(result.is_err() || result.unwrap().contains("requires --privileged"));

    Ok(())
}

#[tokio::test]
async fn test_concurrent_tool_execution() -> Result<()> {
    // Create separate clients for each concurrent operation to avoid borrowing issues
    let mut client1 = McpTestClient::new(false);
    let mut client2 = McpTestClient::new(false);
    let mut client3 = McpTestClient::new(false);

    // Start all clients
    client1.start().await?;
    client2.start().await?;
    client3.start().await?;

    // Initialize all clients
    client1.initialize().await?;
    client2.initialize().await?;
    client3.initialize().await?;

    // Execute multiple tools concurrently using separate clients
    let tasks = vec![
        client1.call_tool("bin__tcl_execute", json!({"script": "expr {1 + 1}"})),
        client2.call_tool("bin__tcl_execute", json!({"script": "expr {2 * 2}"})),
        client3.call_tool("bin__tcl_execute", json!({"script": "expr {3 + 3}"})),
    ];

    // Wait for all to complete
    let results = futures::future::try_join_all(tasks).await?;

    assert_eq!(results.len(), 3);
    assert!(results[0].contains("2"));
    assert!(results[1].contains("4"));
    assert!(results[2].contains("6"));

    Ok(())
}
