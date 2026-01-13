use anyhow::Result;
use serde_json::json;
use tokio::sync::oneshot;

#[cfg(test)]
mod bin_exec_tool_tests {
    use super::*;
    use crate::tcl_executor::{TclCommand, TclExecutor};
    use crate::namespace::{ToolPath, Namespace};
    use crate::tcl_tools::ParameterDefinition;

    /// Test basic bin__exec_tool functionality
    #[tokio::test]
    async fn test_bin_exec_tool_basic() {
        let executor = TclExecutor::spawn(true);
        
        // Create a simple test tool
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::AddTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "test_tool", "1.0"),
            description: "Test tool for bin__exec_tool".to_string(),
            script: r#"
                return "Tool executed with args: $args"
            "#.to_string(),
            parameters: vec![
                ParameterDefinition {
                    name: "args".to_string(),
                    description: "Arguments to pass".to_string(),
                    required: false,
                    type_name: "string".to_string(),
                }
            ],
            response: tx,
        }).await.unwrap();
        
        let result = rx.await.unwrap();
        assert!(result.is_ok(), "Failed to add test tool: {:?}", result);
        
        // Execute the tool using bin__exec_tool
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::ExecuteCustomTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "test_tool", "1.0"),
            params: json!({
                "args": "hello world"
            }),
            response: tx,
        }).await.unwrap();
        
        let result = rx.await.unwrap();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "Tool executed with args: hello world");
    }

    /// Test bin__exec_tool with missing required parameters
    #[tokio::test]
    async fn test_bin_exec_tool_missing_required_params() {
        let executor = TclExecutor::spawn(true);
        
        // Create a tool with required parameters
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::AddTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "required_params_tool", "1.0"),
            description: "Test tool with required parameters".to_string(),
            script: r#"
                return "Name: $name, Age: $age"
            "#.to_string(),
            parameters: vec![
                ParameterDefinition {
                    name: "name".to_string(),
                    description: "User name".to_string(),
                    required: true,
                    type_name: "string".to_string(),
                },
                ParameterDefinition {
                    name: "age".to_string(),
                    description: "User age".to_string(),
                    required: true,
                    type_name: "number".to_string(),
                }
            ],
            response: tx,
        }).await.unwrap();
        
        rx.await.unwrap().unwrap();
        
        // Try to execute without required parameters
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::ExecuteCustomTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "required_params_tool", "1.0"),
            params: json!({
                "name": "Alice"
                // Missing required 'age' parameter
            }),
            response: tx,
        }).await.unwrap();
        
        let result = rx.await.unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Missing required parameter: age"));
    }

    /// Test bin__exec_tool with non-existent tool
    #[tokio::test]
    async fn test_bin_exec_tool_non_existent() {
        let executor = TclExecutor::spawn(true);
        
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::ExecuteCustomTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "non_existent", "1.0"),
            params: json!({}),
            response: tx,
        }).await.unwrap();
        
        let result = rx.await.unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    /// Test bin__exec_tool with different parameter types
    #[tokio::test]
    async fn test_bin_exec_tool_parameter_types() {
        let executor = TclExecutor::spawn(true);
        
        // Create a tool that uses different parameter types
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::AddTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "type_test", "1.0"),
            description: "Test different parameter types".to_string(),
            script: r#"
                set result ""
                append result "String: $str_param\n"
                append result "Number: $num_param\n"
                append result "Boolean: $bool_param\n"
                append result "Array: $array_param\n"
                return $result
            "#.to_string(),
            parameters: vec![
                ParameterDefinition {
                    name: "str_param".to_string(),
                    description: "String parameter".to_string(),
                    required: true,
                    type_name: "string".to_string(),
                },
                ParameterDefinition {
                    name: "num_param".to_string(),
                    description: "Number parameter".to_string(),
                    required: true,
                    type_name: "number".to_string(),
                },
                ParameterDefinition {
                    name: "bool_param".to_string(),
                    description: "Boolean parameter".to_string(),
                    required: true,
                    type_name: "boolean".to_string(),
                },
                ParameterDefinition {
                    name: "array_param".to_string(),
                    description: "Array parameter".to_string(),
                    required: true,
                    type_name: "array".to_string(),
                }
            ],
            response: tx,
        }).await.unwrap();
        
        rx.await.unwrap().unwrap();
        
        // Execute with different types
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::ExecuteCustomTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "type_test", "1.0"),
            params: json!({
                "str_param": "hello",
                "num_param": 42,
                "bool_param": true,
                "array_param": ["a", "b", "c"]
            }),
            response: tx,
        }).await.unwrap();
        
        let result = rx.await.unwrap().unwrap();
        assert!(result.contains("String: hello"));
        assert!(result.contains("Number: 42"));
        assert!(result.contains("Boolean: true"));
        assert!(result.contains("Array: [\"a\",\"b\",\"c\"]"));
    }

    /// Test bin__exec_tool with special characters in parameters
    #[tokio::test]
    async fn test_bin_exec_tool_special_characters() {
        let executor = TclExecutor::spawn(true);
        
        // Create a tool that echoes input
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::AddTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "echo_tool", "1.0"),
            description: "Echo tool for special character testing".to_string(),
            script: r#"
                return "Echo: $input"
            "#.to_string(),
            parameters: vec![
                ParameterDefinition {
                    name: "input".to_string(),
                    description: "Input to echo".to_string(),
                    required: true,
                    type_name: "string".to_string(),
                }
            ],
            response: tx,
        }).await.unwrap();
        
        rx.await.unwrap().unwrap();
        
        // Test with special characters
        let test_cases = vec![
            "hello \"world\"",
            "test\\backslash",
            "line1\nline2",
            "$variable",
            "{braces}",
            "[brackets]",
        ];
        
        for test_input in test_cases {
            let (tx, rx) = oneshot::channel();
            executor.send(TclCommand::ExecuteCustomTool {
                path: ToolPath::new(Namespace::User("test".to_string()), "echo_tool", "1.0"),
                params: json!({
                    "input": test_input
                }),
                response: tx,
            }).await.unwrap();
            
            let result = rx.await.unwrap();
            assert!(result.is_ok(), "Failed for input: {}", test_input);
        }
    }

    /// Test bin__exec_tool with error handling in TCL script
    #[tokio::test]
    async fn test_bin_exec_tool_script_error() {
        let executor = TclExecutor::spawn(true);
        
        // Create a tool that throws an error
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::AddTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "error_tool", "1.0"),
            description: "Tool that throws errors".to_string(),
            script: r#"
                if {$should_error == "true"} {
                    error "Intentional error for testing"
                }
                return "No error"
            "#.to_string(),
            parameters: vec![
                ParameterDefinition {
                    name: "should_error".to_string(),
                    description: "Whether to throw error".to_string(),
                    required: true,
                    type_name: "string".to_string(),
                }
            ],
            response: tx,
        }).await.unwrap();
        
        rx.await.unwrap().unwrap();
        
        // Test with error
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::ExecuteCustomTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "error_tool", "1.0"),
            params: json!({
                "should_error": "true"
            }),
            response: tx,
        }).await.unwrap();
        
        let result = rx.await.unwrap();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Intentional error"));
        
        // Test without error
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::ExecuteCustomTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "error_tool", "1.0"),
            params: json!({
                "should_error": "false"
            }),
            response: tx,
        }).await.unwrap();
        
        let result = rx.await.unwrap();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "No error");
    }

    /// Test bin__exec_tool with complex TCL operations
    #[tokio::test]
    async fn test_bin_exec_tool_complex_script() {
        let executor = TclExecutor::spawn(true);
        
        // Create a tool with complex TCL operations
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::AddTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "complex_tool", "1.0"),
            description: "Complex TCL operations".to_string(),
            script: r#"
                # Calculate factorial
                proc factorial {n} {
                    if {$n <= 1} {
                        return 1
                    }
                    return [expr {$n * [factorial [expr {$n - 1}]]}]
                }
                
                # Process list
                set numbers [split $input_list ","]
                set results {}
                
                foreach num $numbers {
                    set trimmed [string trim $num]
                    if {[string is integer $trimmed]} {
                        lappend results [factorial $trimmed]
                    } else {
                        lappend results "invalid"
                    }
                }
                
                return [join $results ","]
            "#.to_string(),
            parameters: vec![
                ParameterDefinition {
                    name: "input_list".to_string(),
                    description: "Comma-separated list of numbers".to_string(),
                    required: true,
                    type_name: "string".to_string(),
                }
            ],
            response: tx,
        }).await.unwrap();
        
        rx.await.unwrap().unwrap();
        
        // Test complex operations
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::ExecuteCustomTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "complex_tool", "1.0"),
            params: json!({
                "input_list": "5, 3, invalid, 4, 0"
            }),
            response: tx,
        }).await.unwrap();
        
        let result = rx.await.unwrap().unwrap();
        assert_eq!(result, "120,6,invalid,24,1");
    }

    /// Test bin__exec_tool concurrency
    #[tokio::test]
    async fn test_bin_exec_tool_concurrent_execution() {
        let executor = TclExecutor::spawn(true);
        
        // Create a tool that simulates work
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::AddTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "concurrent_tool", "1.0"),
            description: "Tool for concurrency testing".to_string(),
            script: r#"
                set result "Task $task_id completed"
                return $result
            "#.to_string(),
            parameters: vec![
                ParameterDefinition {
                    name: "task_id".to_string(),
                    description: "Task identifier".to_string(),
                    required: true,
                    type_name: "string".to_string(),
                }
            ],
            response: tx,
        }).await.unwrap();
        
        rx.await.unwrap().unwrap();
        
        // Execute multiple tools concurrently
        let mut handles = vec![];
        
        for i in 0..10 {
            let executor_clone = executor.clone();
            let handle = tokio::spawn(async move {
                let (tx, rx) = oneshot::channel();
                executor_clone.send(TclCommand::ExecuteCustomTool {
                    path: ToolPath::new(Namespace::User("test".to_string()), "concurrent_tool", "1.0"),
                    params: json!({
                        "task_id": format!("{}", i)
                    }),
                    response: tx,
                }).await.unwrap();
                
                rx.await.unwrap()
            });
            handles.push(handle);
        }
        
        // Wait for all to complete
        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.await.unwrap();
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), format!("Task {} completed", i));
        }
    }

    /// Test bin__exec_tool with privileged mode restrictions
    #[tokio::test]
    async fn test_bin_exec_tool_privilege_mode() {
        // Test with non-privileged executor
        let executor = TclExecutor::spawn(false);
        
        // Try to add a tool (should succeed even in non-privileged mode for user namespace)
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::AddTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "user_tool", "1.0"),
            description: "User tool in non-privileged mode".to_string(),
            script: r#"
                return "User tool executed"
            "#.to_string(),
            parameters: vec![],
            response: tx,
        }).await.unwrap();
        
        let result = rx.await.unwrap();
        assert!(result.is_ok());
        
        // Execute the user tool
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::ExecuteCustomTool {
            path: ToolPath::new(Namespace::User("test".to_string()), "user_tool", "1.0"),
            params: json!({}),
            response: tx,
        }).await.unwrap();
        
        let result = rx.await.unwrap();
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "User tool executed");
    }
}

#[cfg(test)]
mod bin_exec_tool_mcp_integration {
    use super::*;

    /// Test bin__exec_tool MCP name conversion
    #[test]
    fn test_bin_exec_tool_mcp_naming() {
        let test_cases = vec![
            (ToolPath::bin("exec_tool"), "mcp__tcl__bin___exec_tool"),
            (ToolPath::new(Namespace::User("alice".to_string()), "my_tool", "1.0"), 
             "mcp__tcl__alice___my_tool__1_0"),
            (ToolPath::new(Namespace::User("bob".to_string()), "complex-tool", "2.1.3"), 
             "mcp__tcl__bob___complex_tool__2_1_3"),
        ];
        
        for (path, expected_name) in test_cases {
            assert_eq!(path.to_mcp_name(), expected_name);
        }
    }

    /// Test bin__exec_tool discovery
    #[tokio::test]
    async fn test_bin_exec_tool_discovery() {
        let executor = TclExecutor::spawn(true);
        
        // Add multiple tools
        let tools = vec![
            ("tool1", "First tool"),
            ("tool2", "Second tool"),
            ("exec_helper", "Execution helper"),
        ];
        
        for (name, desc) in &tools {
            let (tx, rx) = oneshot::channel();
            executor.send(TclCommand::AddTool {
                path: ToolPath::new(Namespace::User("test".to_string()), name, "1.0"),
                description: desc.to_string(),
                script: format!("return \"{}\"", name),
                parameters: vec![],
                response: tx,
            }).await.unwrap();
            
            rx.await.unwrap().unwrap();
        }
        
        // List tools with filter
        let (tx, rx) = oneshot::channel();
        executor.send(TclCommand::ListTools {
            namespace: Some("test".to_string()),
            filter: Some("exec".to_string()),
            response: tx,
        }).await.unwrap();
        
        let result = rx.await.unwrap().unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].contains("exec_helper"));
    }
}