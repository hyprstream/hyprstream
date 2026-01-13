use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use tokio::time::{timeout, Duration};

/// A test client for MCP protocol communication with the TCL MCP server
pub struct McpTestClient {
    process: Option<Child>,
    privileged: bool,
}

impl McpTestClient {
    pub fn new(privileged: bool) -> Self {
        Self {
            process: None,
            privileged,
        }
    }

    /// Start the MCP server process
    pub async fn start(&mut self) -> Result<()> {
        let mut cmd = Command::new("cargo");
        cmd.args(&["run", "--bin", "tcl-mcp-server", "--"]);

        if self.privileged {
            cmd.arg("--privileged");
        }

        cmd.arg("server")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let process = cmd.spawn()?;

        // Wait a bit for the server to start
        tokio::time::sleep(Duration::from_millis(1000)).await;

        self.process = Some(process);
        Ok(())
    }

    /// Send an MCP request and get the response
    pub async fn send_request(&mut self, method: &str, params: Value) -> Result<Value> {
        let process = self
            .process
            .as_mut()
            .ok_or_else(|| anyhow!("Server not started"))?;

        let request = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        });

        let request_str = serde_json::to_string(&request)?;

        // Send request to stdin
        if let Some(stdin) = process.stdin.as_mut() {
            writeln!(stdin, "{}", request_str)?;
            stdin.flush()?;
        } else {
            return Err(anyhow!("No stdin available"));
        }

        // Read response from stdout
        if let Some(stdout) = process.stdout.as_mut() {
            let mut reader = BufReader::new(stdout);

            // Keep reading lines until we get a valid JSON response
            for _ in 0..100 {
                // Try up to 100 lines
                let mut response_line = String::new();
                match reader.read_line(&mut response_line) {
                    Ok(0) => return Err(anyhow!("EOF reached without valid JSON")),
                    Ok(_) => {
                        let trimmed = response_line.trim();
                        if trimmed.is_empty() {
                            continue;
                        }

                        // Try to parse as JSON
                        if let Ok(response) = serde_json::from_str::<Value>(trimmed) {
                            if response.is_object() && response.get("jsonrpc").is_some() {
                                if let Some(error) = response.get("error") {
                                    return Err(anyhow!("MCP error: {}", error));
                                }
                                return Ok(response.get("result").unwrap_or(&Value::Null).clone());
                            }
                        }

                        // If not valid JSON-RPC, continue reading
                    }
                    Err(e) => return Err(anyhow!("IO error: {}", e)),
                }
            }

            Err(anyhow!("Too many lines without valid JSON response"))
        } else {
            Err(anyhow!("No stdout available"))
        }
    }

    /// Initialize the MCP connection
    pub async fn initialize(&mut self) -> Result<Value> {
        self.send_request(
            "initialize",
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {}
            }),
        )
        .await
    }

    /// List available tools
    pub async fn list_tools(&mut self) -> Result<Value> {
        self.send_request("tools/list", json!({})).await
    }

    /// Call a tool with the given parameters
    pub async fn call_tool(&mut self, tool_name: &str, arguments: Value) -> Result<String> {
        let result = self
            .send_request(
                "tools/call",
                json!({
                    "name": tool_name,
                    "arguments": arguments
                }),
            )
            .await?;

        // Extract text content from MCP response
        if let Some(content) = result.get("content").and_then(|c| c.as_array()) {
            if let Some(first_content) = content.first() {
                if let Some(text) = first_content.get("text").and_then(|t| t.as_str()) {
                    return Ok(text.to_string());
                }
            }
        }

        Err(anyhow!("No text content in response: {}", result))
    }

    /// Add a new TCL tool (requires privileged mode)
    pub async fn add_tool(
        &mut self,
        user: &str,
        package: &str,
        name: &str,
        description: &str,
        script: &str,
        parameters: Vec<Value>,
    ) -> Result<String> {
        if !self.privileged {
            return Err(anyhow!("Adding tools requires privileged mode"));
        }

        let result = self
            .call_tool(
                "sbin__tcl_tool_add",
                json!({
                    "user": user,
                    "package": package,
                    "name": name,
                    "description": description,
                    "script": script,
                    "parameters": parameters
                }),
            )
            .await?;

        Ok(result)
    }

    /// Execute a tool by its path
    pub async fn exec_tool(&mut self, tool_path: &str, params: Value) -> Result<String> {
        self.call_tool(
            "bin__exec_tool",
            json!({
                "tool_path": tool_path,
                "params": params
            }),
        )
        .await
    }
}

impl Drop for McpTestClient {
    fn drop(&mut self) {
        if let Some(mut process) = self.process.take() {
            let _ = process.kill();
            let _ = process.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_client_basic_functionality() {
        let mut client = McpTestClient::new(false);

        // Start the server
        client.start().await.unwrap();

        // Initialize connection
        let init_result = client.initialize().await.unwrap();
        assert!(init_result.get("protocolVersion").is_some());

        // List tools
        let tools_result = client.list_tools().await.unwrap();
        assert!(tools_result.get("tools").is_some());
    }
}
