use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

/// A simple mock MCP server for testing
/// Can be compiled as a separate binary for integration tests
fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = BufReader::new(stdin);

    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let request: Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let response = handle_request(request);
                if let Ok(response_str) = serde_json::to_string(&response) {
                    println!("{}", response_str);
                    stdout.lock().flush().unwrap();
                }
            }
            Err(_) => break,
        }
    }
}

fn handle_request(request: Value) -> Value {
    let method = request["method"].as_str().unwrap_or("");
    let id = request["id"].as_u64();

    match method {
        "initialize" => {
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "mock-mcp-server",
                        "version": "1.0.0"
                    }
                }
            })
        }
        "tools/list" => {
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "tools": [
                        {
                            "name": "echo",
                            "description": "Echo back the input",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "message": {
                                        "type": "string",
                                        "description": "Message to echo"
                                    }
                                },
                                "required": ["message"]
                            }
                        },
                        {
                            "name": "add",
                            "description": "Add two numbers",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "a": {"type": "number"},
                                    "b": {"type": "number"}
                                },
                                "required": ["a", "b"]
                            }
                        }
                    ]
                }
            })
        }
        "tools/call" => {
            let tool_name = request["params"]["name"].as_str().unwrap_or("");
            let args = &request["params"]["arguments"];

            let result = match tool_name {
                "echo" => {
                    let message = args["message"].as_str().unwrap_or("No message");
                    json!({
                        "content": [{
                            "type": "text",
                            "text": message
                        }]
                    })
                }
                "add" => {
                    let a = args["a"].as_f64().unwrap_or(0.0);
                    let b = args["b"].as_f64().unwrap_or(0.0);
                    json!({
                        "content": [{
                            "type": "text",
                            "text": format!("{}", a + b)
                        }]
                    })
                }
                _ => {
                    return json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {
                            "code": -32601,
                            "message": format!("Tool '{}' not found", tool_name)
                        }
                    });
                }
            };

            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": result
            })
        }
        _ => {
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {
                    "code": -32601,
                    "message": "Method not found"
                }
            })
        }
    }
}
