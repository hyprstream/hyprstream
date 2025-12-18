//! MCP (Model Context Protocol) tool invocation sink
//!
//! Invokes MCP tools in response to events. The tool receives the event
//! as its input and can return results or trigger further actions.

use super::super::bus::EventSubscriber;
use super::super::EventEnvelope;
use tracing::{debug, error, info, trace, warn};

/// MCP sink loop
///
/// Receives events and invokes an MCP tool for each event.
/// The event is passed as the tool's input in JSON format.
///
/// # Parameters
/// - `subscriber`: Event subscriber
/// - `tool`: MCP tool name to invoke
/// - `endpoint`: MCP server endpoint (IPC or TCP)
pub async fn mcp_loop(mut subscriber: EventSubscriber, tool: &str, endpoint: &str) {
    info!(
        "starting MCP sink for tool '{}' at '{}' for topic '{}'",
        tool,
        endpoint,
        subscriber.topic_filter()
    );

    // TODO: Connect to MCP server when hyprstream-mcp is implemented
    // For now, this is a placeholder that logs what would be invoked.

    loop {
        match subscriber.recv().await {
            Ok(event) => {
                debug!(
                    "MCP sink received event {} (topic: {})",
                    event.id, event.topic
                );

                // Serialize event to JSON for tool input
                let input = match serde_json::to_value(&event) {
                    Ok(v) => v,
                    Err(e) => {
                        error!("failed to serialize event: {}", e);
                        continue;
                    }
                };

                // TODO: Invoke MCP tool when hyprstream-mcp is implemented
                // let result = mcp_client.call_tool(tool, input).await?;
                trace!(
                    "would invoke MCP tool '{}' at '{}' with event {}",
                    tool,
                    endpoint,
                    event.id
                );

                // Log tool invocation (placeholder)
                debug!(
                    "MCP tool '{}' would process {} event for {}",
                    tool,
                    event.topic,
                    match &event.payload {
                        crate::events::EventPayload::GenerationComplete { model_id, .. } =>
                            model_id.clone(),
                        crate::events::EventPayload::ThresholdBreach { model_id, .. } =>
                            model_id.clone(),
                        crate::events::EventPayload::TrainingStarted { model_id, .. } =>
                            model_id.clone(),
                        crate::events::EventPayload::RepositoryCloned { name, .. } => name.clone(),
                        crate::events::EventPayload::CommitCreated { repo_id, .. } =>
                            repo_id.clone(),
                        _ => "unknown".to_string(),
                    }
                );
            }
            Err(e) => {
                warn!("MCP sink recv error: {}", e);
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
    }
}

/// MCP client connection (placeholder)
///
/// When hyprstream-mcp is implemented, this will provide:
/// - Connection to MCP servers via IPC or TCP
/// - Tool discovery and invocation
/// - Result handling and error propagation
#[allow(dead_code)]
struct McpClient {
    endpoint: String,
    // connection: Option<...>,
}

#[allow(dead_code)]
impl McpClient {
    fn new(endpoint: &str) -> Self {
        Self {
            endpoint: endpoint.to_string(),
        }
    }

    async fn connect(&mut self) -> anyhow::Result<()> {
        // TODO: Establish connection to MCP server
        Ok(())
    }

    async fn call_tool(
        &self,
        tool: &str,
        input: serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        // TODO: Invoke tool and return result
        trace!("calling tool {} with input: {:?}", tool, input);
        Ok(serde_json::json!({"status": "ok"}))
    }

    async fn list_tools(&self) -> anyhow::Result<Vec<String>> {
        // TODO: List available tools from server
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_client_creation() {
        let client = McpClient::new("ipc:///tmp/mcp.sock");
        assert_eq!(client.endpoint, "ipc:///tmp/mcp.sock");
    }
}
