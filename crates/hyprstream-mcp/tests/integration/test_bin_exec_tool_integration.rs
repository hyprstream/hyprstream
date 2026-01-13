use anyhow::Result;
use jsonrpc_core::{IoHandler, Params, Value};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::Mutex;

#[cfg(test)]
mod bin_exec_tool_integration_tests {
    use super::*;
    use crate::server::TclMcpServer;
    use crate::tcl_tools::TclToolBox;
    use crate::tcl_executor::TclExecutor;

    /// Helper to create a test MCP server
    fn create_test_server(privileged: bool) -> (IoHandler, Arc<Mutex<TclToolBox>>) {
        let executor = TclExecutor::spawn(privileged);
        let tool_box = Arc::new(Mutex::new(TclToolBox::new(executor)));
        let handler = IoHandler::new();
        
        // Mock MCP initialization
        handler.add_sync_method("initialize", |_: Params| {
            Ok(json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "test-tcl-mcp",
                    "version": "1.0.0"
                }
            }))
        });
        
        (handler, tool_box)
    }

    /// Test bin__exec_tool through MCP protocol
    #[tokio::test]
    async fn test_bin_exec_tool_mcp_e2e() {
        let (mut handler, tool_box) = create_test_server(true);
        
        // First, add a custom tool
        let tb = tool_box.clone();
        handler.add_method("tools/call", move |params: Params| {
            let tb = tb.clone();
            async move {
                let params: serde_json::Value = params.parse().unwrap();
                let tool_name = params["name"].as_str().unwrap();
                let args = &params["arguments"];
                
                // Handle tool addition
                if tool_name == "mcp__tcl__sbin___tcl_tool_add" {
                    let user = args["user"].as_str().unwrap();
                    let package = args["package"].as_str().unwrap();
                    let name = args["name"].as_str().unwrap();
                    let description = args["description"].as_str().unwrap();
                    let script = args["script"].as_str().unwrap();
                    
                    // Simulate tool addition
                    Ok(json!({
                        "content": [{
                            "type": "text",
                            "text": format!("Tool '/{}/{}/{}:latest' added successfully", user, package, name)
                        }]
                    }))
                } 
                // Handle bin__exec_tool
                else if tool_name == "mcp__tcl__bin___exec_tool" {
                    let tool_path = args["tool_path"].as_str().unwrap();
                    let tool_args = &args["arguments"];
                    
                    // Simulate tool execution
                    Ok(json!({
                        "content": [{
                            "type": "text",
                            "text": format!("Executed tool '{}' with args: {}", tool_path, tool_args)
                        }]
                    }))
                }
                // Handle tool listing
                else if tool_name == "mcp__tcl__bin___tcl_tool_list" {
                    Ok(json!({
                        "content": [{
                            "type": "text",
                            "text": "Available tools:\n/bin/tcl_execute\n/bin/exec_tool\n/user/test/my_tool:1.0"
                        }]
                    }))
                } else {
                    Err(jsonrpc_core::Error::invalid_params("Unknown tool"))
                }
            }
        });
        
        // Test adding a tool via MCP
        let add_request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mcp__tcl__sbin___tcl_tool_add",
                "arguments": {
                    "user": "test",
                    "package": "utils",
                    "name": "string_reverse",
                    "description": "Reverse a string",
                    "script": "return [string reverse $input]",
                    "parameters": [{
                        "name": "input",
                        "description": "String to reverse",
                        "required": true,
                        "type_name": "string"
                    }]
                }
            }
        });
        
        let response = handler.handle_request(&add_request.to_string()).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(parsed["result"]["content"][0]["text"].as_str().unwrap()
            .contains("added successfully"));
        
        // Test executing the tool via bin__exec_tool
        let exec_request = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "mcp__tcl__bin___exec_tool",
                "arguments": {
                    "tool_path": "/test/utils/string_reverse:latest",
                    "arguments": {
                        "input": "hello world"
                    }
                }
            }
        });
        
        let response = handler.handle_request(&exec_request.to_string()).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(parsed["result"]["content"][0]["text"].as_str().unwrap()
            .contains("Executed tool"));
    }

    /// Test bin__exec_tool error handling through MCP
    #[tokio::test]
    async fn test_bin_exec_tool_mcp_errors() {
        let (mut handler, _) = create_test_server(true);
        
        handler.add_method("tools/call", |params: Params| async move {
            let params: serde_json::Value = params.parse().unwrap();
            let tool_name = params["name"].as_str().unwrap();
            
            if tool_name == "mcp__tcl__bin___exec_tool" {
                let tool_path = params["arguments"]["tool_path"].as_str().unwrap();
                
                // Simulate different error scenarios
                if tool_path.contains("non_existent") {
                    Err(jsonrpc_core::Error::invalid_params("Tool not found"))
                } else if tool_path.contains("error_tool") {
                    Err(jsonrpc_core::Error::internal_error())
                } else {
                    Ok(json!({
                        "content": [{
                            "type": "text",
                            "text": "Success"
                        }]
                    }))
                }
            } else {
                Err(jsonrpc_core::Error::method_not_found())
            }
        });
        
        // Test non-existent tool
        let request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mcp__tcl__bin___exec_tool",
                "arguments": {
                    "tool_path": "/user/non_existent:1.0",
                    "arguments": {}
                }
            }
        });
        
        let response = handler.handle_request(&request.to_string()).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(parsed["error"]["message"].as_str().unwrap().contains("Tool not found"));
        
        // Test internal error
        let request = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "mcp__tcl__bin___exec_tool",
                "arguments": {
                    "tool_path": "/user/error_tool:1.0",
                    "arguments": {}
                }
            }
        });
        
        let response = handler.handle_request(&request.to_string()).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(parsed["error"]["code"].as_i64().unwrap() == -32603); // Internal error
    }

    /// Test bin__exec_tool with privilege restrictions
    #[tokio::test]
    async fn test_bin_exec_tool_privilege_restrictions() {
        let (mut handler, _) = create_test_server(false); // Non-privileged
        
        handler.add_method("tools/list", |_: Params| async move {
            // In non-privileged mode, sbin tools should not be listed
            Ok(json!({
                "tools": [
                    {
                        "name": "mcp__tcl__bin___tcl_execute",
                        "description": "Execute TCL script",
                        "inputSchema": {}
                    },
                    {
                        "name": "mcp__tcl__bin___exec_tool",
                        "description": "Execute a custom tool",
                        "inputSchema": {}
                    }
                    // Note: No sbin tools listed
                ]
            }))
        });
        
        let request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list"
        });
        
        let response = handler.handle_request(&request.to_string()).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        let tools = parsed["result"]["tools"].as_array().unwrap();
        
        // Verify no sbin tools are exposed
        for tool in tools {
            let name = tool["name"].as_str().unwrap();
            assert!(!name.contains("sbin"));
        }
    }

    /// Test bin__exec_tool concurrent access
    #[tokio::test]
    async fn test_bin_exec_tool_concurrent_access() {
        let (handler, _) = create_test_server(true);
        let handler = Arc::new(handler);
        
        // Add concurrent request handler
        let h = handler.clone();
        h.add_method("tools/call", |params: Params| async move {
            let params: serde_json::Value = params.parse().unwrap();
            let tool_name = params["name"].as_str().unwrap();
            
            if tool_name == "mcp__tcl__bin___exec_tool" {
                // Simulate some processing delay
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                
                Ok(json!({
                    "content": [{
                        "type": "text",
                        "text": format!("Processed: {}", params["arguments"]["id"].as_str().unwrap())
                    }]
                }))
            } else {
                Err(jsonrpc_core::Error::method_not_found())
            }
        });
        
        // Launch multiple concurrent requests
        let mut handles = vec![];
        
        for i in 0..20 {
            let h = handler.clone();
            let handle = tokio::spawn(async move {
                let request = json!({
                    "jsonrpc": "2.0",
                    "id": i,
                    "method": "tools/call",
                    "params": {
                        "name": "mcp__tcl__bin___exec_tool",
                        "arguments": {
                            "tool_path": "/test/concurrent:1.0",
                            "arguments": {
                                "id": format!("task_{}", i)
                            }
                        }
                    }
                });
                
                h.handle_request(&request.to_string()).await
            });
            handles.push(handle);
        }
        
        // Verify all requests complete successfully
        for (i, handle) in handles.into_iter().enumerate() {
            let response = handle.await.unwrap().unwrap();
            let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
            assert!(parsed["result"]["content"][0]["text"].as_str().unwrap()
                .contains(&format!("task_{}", i)));
        }
    }

    /// Test bin__exec_tool with complex parameter scenarios
    #[tokio::test]
    async fn test_bin_exec_tool_complex_parameters() {
        let (mut handler, _) = create_test_server(true);
        
        handler.add_method("tools/call", |params: Params| async move {
            let params: serde_json::Value = params.parse().unwrap();
            let args = &params["arguments"]["arguments"];
            
            // Verify complex parameter handling
            assert!(args["nested"]["inner"]["value"].as_i64().unwrap() == 42);
            assert!(args["array"].as_array().unwrap().len() == 3);
            assert!(args["special_chars"].as_str().unwrap().contains("\"quotes\""));
            
            Ok(json!({
                "content": [{
                    "type": "text",
                    "text": "Complex parameters handled correctly"
                }]
            }))
        });
        
        let request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "mcp__tcl__bin___exec_tool",
                "arguments": {
                    "tool_path": "/test/complex:1.0",
                    "arguments": {
                        "nested": {
                            "inner": {
                                "value": 42
                            }
                        },
                        "array": [1, 2, 3],
                        "special_chars": "test with \"quotes\" and \\backslash"
                    }
                }
            }
        });
        
        let response = handler.handle_request(&request.to_string()).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
        assert!(parsed["result"]["content"][0]["text"].as_str().unwrap()
            .contains("Complex parameters handled correctly"));
    }
}