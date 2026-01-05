//! Flight SQL client for querying datasets
//!
//! This module provides a client for connecting to a Flight SQL server
//! and executing queries. Used by the `hyprstream flight` CLI command.

use arrow_array::{Array, RecordBatch, StringArray};
use arrow_flight::{
    decode::FlightRecordBatchStream,
    flight_service_client::FlightServiceClient,
    sql::{CommandStatementQuery, ProstMessageExt},
    FlightDescriptor,
};
use arrow_schema::SchemaRef;
use futures::TryStreamExt;
use prost::Message;
use std::sync::Arc;
use tonic::transport::{Certificate, Channel, ClientTlsConfig, Endpoint};
use tracing::{debug, info};

/// Error type for Flight client operations
#[derive(Debug, thiserror::Error)]
pub enum FlightClientError {
    #[error("Connection failed: {0}")]
    Connection(String),

    #[error("Query failed: {0}")]
    Query(String),

    #[error("Transport error: {0}")]
    Transport(#[from] tonic::transport::Error),

    #[error("Flight error: {0}")]
    Flight(#[from] arrow_flight::error::FlightError),

    #[error("Status error: {0}")]
    Status(#[from] tonic::Status),
}

/// Configuration for the Flight client
#[derive(Debug, Clone, Default)]
pub struct FlightClientConfig {
    /// TLS CA certificate for server verification (PEM)
    pub tls_ca: Option<Vec<u8>>,
    /// TLS client certificate (PEM)
    pub tls_cert: Option<Vec<u8>>,
    /// TLS client key (PEM)
    pub tls_key: Option<Vec<u8>>,
}

impl FlightClientConfig {
    pub fn with_tls_ca(mut self, ca: Vec<u8>) -> Self {
        self.tls_ca = Some(ca);
        self
    }

    pub fn with_client_cert(mut self, cert: Vec<u8>, key: Vec<u8>) -> Self {
        self.tls_cert = Some(cert);
        self.tls_key = Some(key);
        self
    }
}

/// Flight SQL client for querying datasets
pub struct FlightClient {
    client: FlightServiceClient<Channel>,
    addr: String,
}

impl FlightClient {
    /// Connect to a Flight SQL server
    ///
    /// # Arguments
    ///
    /// * `addr` - Server address (e.g., "127.0.0.1:50051" or "https://server:50051")
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let client = FlightClient::connect("127.0.0.1:50051").await?;
    /// let results = client.query("SELECT * FROM metrics").await?;
    /// ```
    pub async fn connect(addr: &str) -> Result<Self, FlightClientError> {
        Self::connect_with_config(addr, FlightClientConfig::default()).await
    }

    /// Connect to a Flight SQL server with TLS configuration
    pub async fn connect_with_config(
        addr: &str,
        config: FlightClientConfig,
    ) -> Result<Self, FlightClientError> {
        let uri = if addr.starts_with("http://") || addr.starts_with("https://") {
            addr.to_string()
        } else {
            format!("http://{}", addr)
        };

        debug!(uri = %uri, "Connecting to Flight SQL server");

        let mut endpoint = Endpoint::from_shared(uri.clone())
            .map_err(|e| FlightClientError::Connection(e.to_string()))?;

        // Configure TLS if provided
        if let Some(ca) = config.tls_ca {
            let mut tls_config = ClientTlsConfig::new().ca_certificate(Certificate::from_pem(&ca));

            if let (Some(cert), Some(key)) = (config.tls_cert, config.tls_key) {
                let identity = tonic::transport::Identity::from_pem(&cert, &key);
                tls_config = tls_config.identity(identity);
            }

            endpoint = endpoint.tls_config(tls_config)?;
        }

        let channel = endpoint.connect().await?;
        let client = FlightServiceClient::new(channel);

        info!(addr = %addr, "Connected to Flight SQL server");

        Ok(Self {
            client,
            addr: addr.to_string(),
        })
    }

    /// Get the server address
    pub fn addr(&self) -> &str {
        &self.addr
    }

    /// Execute a SQL query and return all results as record batches
    ///
    /// # Arguments
    ///
    /// * `sql` - SQL query string
    ///
    /// # Returns
    ///
    /// A vector of record batches containing the query results
    pub async fn query(&mut self, sql: &str) -> Result<Vec<RecordBatch>, FlightClientError> {
        debug!(sql = %sql, "Executing query");

        // Create the Flight SQL command
        let cmd = CommandStatementQuery {
            query: sql.to_string(),
            transaction_id: None,
        };

        // Encode using prost Message trait
        let cmd_bytes = cmd.as_any().encode_to_vec();

        // Get flight info (schema and endpoints)
        let descriptor = FlightDescriptor::new_cmd(cmd_bytes);
        let flight_info = self
            .client
            .get_flight_info(descriptor)
            .await?
            .into_inner();

        // Get the ticket from the first endpoint
        let ticket = flight_info
            .endpoint
            .first()
            .and_then(|ep| ep.ticket.clone())
            .ok_or_else(|| FlightClientError::Query("No endpoint in flight info".to_string()))?;

        // Execute do_get to retrieve data
        let stream = self.client.do_get(ticket).await?.into_inner();

        // Convert to record batches
        let flight_stream = FlightRecordBatchStream::new_from_flight_data(
            stream.map_err(|e| arrow_flight::error::FlightError::Tonic(Box::new(e))),
        );

        let batches: Vec<RecordBatch> = flight_stream.try_collect().await?;

        debug!(batch_count = batches.len(), "Query complete");

        Ok(batches)
    }

    /// Execute a SQL query and return results with schema
    pub async fn query_with_schema(
        &mut self,
        sql: &str,
    ) -> Result<(SchemaRef, Vec<RecordBatch>), FlightClientError> {
        let batches = self.query(sql).await?;
        let schema = if let Some(batch) = batches.first() {
            batch.schema()
        } else {
            Arc::new(arrow_schema::Schema::empty())
        };
        Ok((schema, batches))
    }

    /// Execute a statement that doesn't return results (INSERT, CREATE TABLE, etc.)
    pub async fn execute(&mut self, sql: &str) -> Result<(), FlightClientError> {
        let _ = self.query(sql).await?;
        Ok(())
    }

    /// Check if the server is reachable
    pub async fn health_check(&mut self) -> Result<(), FlightClientError> {
        // Simple health check - list actions
        let _actions = self
            .client
            .list_actions(arrow_flight::Empty {})
            .await?;
        Ok(())
    }
}

/// Get the string value from an array at a given index
fn get_array_value(array: &dyn Array, row_idx: usize) -> String {
    // Try to downcast to StringArray first (most common)
    if let Some(string_array) = array.as_any().downcast_ref::<StringArray>() {
        if string_array.is_null(row_idx) {
            return "NULL".to_string();
        }
        return string_array.value(row_idx).to_string();
    }

    // For other types, use debug format
    format!("{:?}", array.slice(row_idx, 1))
}

/// Format record batches as a table for display
pub fn format_batches_as_table(batches: &[RecordBatch]) -> String {
    use std::fmt::Write;

    if batches.is_empty() {
        return "No results".to_string();
    }

    let schema = batches[0].schema();
    let mut output = String::new();

    // Calculate column widths
    let mut widths: Vec<usize> = schema
        .fields()
        .iter()
        .map(|f| f.name().len())
        .collect();

    for batch in batches {
        for (col_idx, col) in batch.columns().iter().enumerate() {
            for row_idx in 0..col.len() {
                let value = get_array_value(col.as_ref(), row_idx);
                widths[col_idx] = widths[col_idx].max(value.len());
            }
        }
    }

    // Header
    let header: Vec<String> = schema
        .fields()
        .iter()
        .zip(&widths)
        .map(|(f, w)| format!("{:width$}", f.name(), width = *w))
        .collect();
    writeln!(output, "| {} |", header.join(" | ")).unwrap();

    // Separator
    let sep: Vec<String> = widths.iter().map(|w| "-".repeat(*w)).collect();
    writeln!(output, "|-{}-|", sep.join("-|-")).unwrap();

    // Rows
    for batch in batches {
        for row_idx in 0..batch.num_rows() {
            let row: Vec<String> = batch
                .columns()
                .iter()
                .zip(&widths)
                .map(|(col, w)| {
                    let value = get_array_value(col.as_ref(), row_idx);
                    format!("{:width$}", value, width = *w)
                })
                .collect();
            writeln!(output, "| {} |", row.join(" | ")).unwrap();
        }
    }

    // Row count
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    writeln!(output, "\n({} rows)", total_rows).unwrap();

    output
}

/// Format record batches as CSV
pub fn format_batches_as_csv(batches: &[RecordBatch]) -> String {
    use std::fmt::Write;

    if batches.is_empty() {
        return String::new();
    }

    let schema = batches[0].schema();
    let mut output = String::new();

    // Header
    let header: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    writeln!(output, "{}", header.join(",")).unwrap();

    // Rows
    for batch in batches {
        for row_idx in 0..batch.num_rows() {
            let row: Vec<String> = batch
                .columns()
                .iter()
                .map(|col| {
                    let val = get_array_value(col.as_ref(), row_idx);
                    if val.contains(',') || val.contains('"') {
                        format!("\"{}\"", val.replace('"', "\"\""))
                    } else {
                        val
                    }
                })
                .collect();
            writeln!(output, "{}", row.join(",")).unwrap();
        }
    }

    output
}

/// Format record batches as JSON
pub fn format_batches_as_json(batches: &[RecordBatch]) -> Result<String, serde_json::Error> {
    let mut rows = Vec::new();

    for batch in batches {
        let schema = batch.schema();
        for row_idx in 0..batch.num_rows() {
            let mut row = serde_json::Map::new();
            for (col_idx, col) in batch.columns().iter().enumerate() {
                let field_name = schema.field(col_idx).name();
                let value = get_array_value(col.as_ref(), row_idx);
                row.insert(field_name.clone(), serde_json::Value::String(value));
            }
            rows.push(serde_json::Value::Object(row));
        }
    }

    serde_json::to_string_pretty(&rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flight_client_config_builder() {
        let config = FlightClientConfig::default()
            .with_tls_ca(vec![1, 2, 3])
            .with_client_cert(vec![4, 5, 6], vec![7, 8, 9]);

        assert!(config.tls_ca.is_some());
        assert!(config.tls_cert.is_some());
        assert!(config.tls_key.is_some());
    }

    #[test]
    fn test_flight_config_builder() {
        let config = crate::FlightConfig::default()
            .with_host("0.0.0.0")
            .with_port(50052);

        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 50052);
    }
}
