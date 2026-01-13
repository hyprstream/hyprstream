use anyhow::Result;
use serde_json::{json, Value};
use std::process::Stdio;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};

/// Mock MCP server for testing
pub struct MockMcpServer {
    port: u16,
    tools: Vec<MockTool>,
    fail_on_init: bool,
    delay_ms: u64,
}

#[derive(Clone)]
pub struct MockTool {
    pub name: String,
    pub description: String,
    pub schema: Value,
}

impl MockMcpServer {
    pub fn new(port: u16) -> Self {
        Self {
            port,
            tools: vec![],
            fail_on_init: false,
            delay_ms: 0,
        }
    }

    pub fn with_tools(mut self, tools: Vec<MockTool>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_init_failure(mut self) -> Self {
        self.fail_on_init = true;
        self
    }

    pub fn with_delay(mut self, delay_ms: u64) -> Self {
        self.delay_ms = delay_ms;
        self
    }

    /// Start the mock server (returns immediately, server runs in background)
    pub async fn start(&self) -> Result<Child> {
        // For actual implementation, this would start a simple TCP server
        // that implements the MCP protocol
        // For now, we'll use a placeholder

        let mut cmd = Command::new("echo");
        cmd.arg("mock-server")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());

        Ok(cmd.spawn()?)
    }
}

/// Helper to create standard test tools
pub fn create_test_tools() -> Vec<MockTool> {
    vec![
        MockTool {
            name: "test_echo".to_string(),
            description: "Echoes input back".to_string(),
            schema: json!({
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Message to echo"
                    }
                },
                "required": ["message"]
            }),
        },
        MockTool {
            name: "test_math".to_string(),
            description: "Performs basic math".to_string(),
            schema: json!({
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"]
                    }
                },
                "required": ["a", "b", "operation"]
            }),
        },
    ]
}

/// Helper to validate JSON-RPC messages
pub fn validate_jsonrpc_message(message: &Value) -> Result<()> {
    // Check required fields
    if !message.is_object() {
        return Err(anyhow::anyhow!("Message must be an object"));
    }

    let obj = message.as_object().unwrap();

    // Check jsonrpc version
    match obj.get("jsonrpc") {
        Some(v) if v == "2.0" => {}
        Some(v) => return Err(anyhow::anyhow!("Invalid jsonrpc version: {}", v)),
        None => return Err(anyhow::anyhow!("Missing jsonrpc field")),
    }

    // Check for either result or error (for responses)
    if obj.contains_key("id") && !obj.contains_key("method") {
        if !obj.contains_key("result") && !obj.contains_key("error") {
            return Err(anyhow::anyhow!(
                "Response must contain either result or error"
            ));
        }
    }

    // Check method (for requests)
    if obj.contains_key("method") {
        if !obj["method"].is_string() {
            return Err(anyhow::anyhow!("Method must be a string"));
        }
    }

    Ok(())
}

/// Test data generator for edge cases
pub struct TestDataGenerator;

impl TestDataGenerator {
    /// Generate tool names with special characters
    pub fn special_char_tool_names() -> Vec<String> {
        vec![
            "simple_tool".to_string(),
            "tool-with-dashes".to_string(),
            "tool.with.dots".to_string(),
            "tool@with@at".to_string(),
            "tool$with$dollar".to_string(),
            "tool#with#hash".to_string(),
            "tool with spaces".to_string(),  // Should be rejected
            "tool/with/slashes".to_string(), // Should be rejected
            "tool\\with\\backslashes".to_string(), // Should be rejected
            "ToolWithCamelCase".to_string(),
            "TOOL_WITH_UPPERCASE".to_string(),
            "tool_with_numbers_123".to_string(),
            "тул_with_unicode".to_string(),
            "🔧_emoji_tool".to_string(), // Should be rejected
        ]
    }

    /// Generate server IDs with edge cases
    pub fn edge_case_server_ids() -> Vec<String> {
        vec![
            "simple-server".to_string(),
            "server_with_underscore".to_string(),
            "server.with.dots".to_string(),
            "server-123".to_string(),
            "123-server".to_string(),
            "s".to_string(), // Single character
            "a-very-long-server-id-that-exceeds-reasonable-length-limits-and-should-probably-be-truncated".to_string(),
            "server@email.com".to_string(),
            "server:8080".to_string(),
            "server/path/to/something".to_string(),
            "".to_string(), // Empty - should fail
        ]
    }

    /// Generate malicious command attempts
    pub fn malicious_commands() -> Vec<(String, Vec<String>)> {
        vec![
            ("echo".to_string(), vec!["safe".to_string()]),
            ("echo".to_string(), vec!["test; rm -rf /".to_string()]),
            (
                "sh".to_string(),
                vec!["-c".to_string(), "echo test".to_string()],
            ),
            (
                "bash".to_string(),
                vec!["-c".to_string(), "echo test".to_string()],
            ),
            ("echo".to_string(), vec!["$(whoami)".to_string()]),
            ("echo".to_string(), vec!["`id`".to_string()]),
            ("echo".to_string(), vec!["${HOME}".to_string()]),
            (
                "cmd".to_string(),
                vec!["/c".to_string(), "echo test".to_string()],
            ),
            (
                "powershell".to_string(),
                vec!["-Command".to_string(), "Get-Process".to_string()],
            ),
        ]
    }

    /// Generate various TCL scripts for testing
    pub fn tcl_test_scripts() -> Vec<(&'static str, &'static str)> {
        vec![
            ("simple", "expr 1 + 1"),
            ("puts", "puts \"Hello, World!\""),
            ("list", "set mylist [list a b c]; llength $mylist"),
            ("error", "error \"This is a test error\""),
            ("infinite", "while {1} {puts \"loop\"}"), // Should timeout
            ("invalid", "this is not valid TCL"),
            ("empty", ""),
            ("comment_only", "# Just a comment"),
            ("multiline", "set x 1\nset y 2\nexpr $x + $y"),
            ("proc", "proc add {a b} {expr $a + $b}; add 5 3"),
            ("unicode", "puts \"Hello, 世界\""),
            ("special_chars", "puts \"Special: <>&\\\"'\""),
        ]
    }
}

/// Async test timeout helper
pub async fn with_timeout<F, T>(duration_ms: u64, future: F) -> Result<T>
where
    F: std::future::Future<Output = T>,
{
    match tokio::time::timeout(tokio::time::Duration::from_millis(duration_ms), future).await {
        Ok(result) => Ok(result),
        Err(_) => Err(anyhow::anyhow!(
            "Operation timed out after {}ms",
            duration_ms
        )),
    }
}

/// Create a test environment with temporary directories
pub struct TestEnvironment {
    pub temp_dir: tempfile::TempDir,
    pub tools_dir: std::path::PathBuf,
}

impl TestEnvironment {
    pub fn new() -> Result<Self> {
        let temp_dir = tempfile::tempdir()?;
        let tools_dir = temp_dir.path().join("tools");

        // Create standard directories
        std::fs::create_dir_all(&tools_dir.join("bin"))?;
        std::fs::create_dir_all(&tools_dir.join("sbin"))?;
        std::fs::create_dir_all(&tools_dir.join("docs"))?;
        std::fs::create_dir_all(&tools_dir.join("users"))?;

        Ok(Self {
            temp_dir,
            tools_dir,
        })
    }

    pub fn create_test_tool(&self, namespace: &str, name: &str, content: &str) -> Result<()> {
        let tool_path = self.tools_dir.join(namespace).join(format!("{}.tcl", name));
        std::fs::write(tool_path, content)?;
        Ok(())
    }
}

pub mod mcp_test_client;
