//! cas-serve - CAS server for stdin/stdout transport
//!
//! This binary enables XET CAS operations over SSH by exposing a LocalClient
//! via a simple JSON-based protocol over stdin/stdout.
//!
//! # Usage
//!
//! ```bash
//! # Start the server (reads from stdin, writes to stdout)
//! cas-serve
//!
//! # With custom storage path
//! CAS_STORAGE=~/.cache/xet cas-serve
//!
//! # Via SSH
//! ssh user@host cas-serve
//! ```
//!
//! # Protocol
//!
//! Uses newline-delimited JSON (NDJSON). Each request is a single line of JSON,
//! and each response is a single line of JSON.
//!
//! See `protocol.rs` for message types.

mod protocol;

use protocol::{ErrorCode, Request, Response};
use std::io::{BufRead, Write};
use std::path::PathBuf;
use tracing::{debug, error, info, warn};

/// Storage path resolution order:
/// 1. CAS_STORAGE environment variable
/// 2. XET_CACHE_DIR environment variable
/// 3. ~/.cache/xet/ (default)
fn resolve_storage_path() -> PathBuf {
    if let Ok(path) = std::env::var("CAS_STORAGE") {
        return PathBuf::from(shellexpand::tilde(&path).into_owned());
    }
    if let Ok(path) = std::env::var("XET_CACHE_DIR") {
        return PathBuf::from(shellexpand::tilde(&path).into_owned());
    }

    // Default: ~/.cache/xet/
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("xet")
}

/// CAS server that processes requests from stdin and writes responses to stdout
struct CasServer {
    storage_path: PathBuf,
}

impl CasServer {
    fn new(storage_path: PathBuf) -> Self {
        Self { storage_path }
    }

    /// Process a single request and return a response
    async fn handle_request(&self, request: Request) -> Response {
        match request {
            Request::Ping => Response::Pong,

            Request::Shutdown => Response::ShutdownAck,

            Request::GetFile { hash } => self.handle_get_file(&hash).await,

            Request::Exists { hash } => self.handle_exists(&hash).await,

            Request::UploadXorb { data } => self.handle_upload_xorb(&data).await,

            Request::GetReconstructionInfo { hash } => {
                self.handle_get_reconstruction_info(&hash).await
            }
        }
    }

    async fn handle_get_file(&self, hash: &str) -> Response {
        // Parse the merkle hash
        let merkle_hash = match Request::parse_hash(hash) {
            Ok(h) => h,
            Err(e) => return Response::error(ErrorCode::InvalidHash, e),
        };

        // Try to read the file from local storage
        let xorb_path = self
            .storage_path
            .join("xorbs")
            .join(format!("default.{}", hash));

        match tokio::fs::read(&xorb_path).await {
            Ok(data) => Response::file(&data),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Response::error(ErrorCode::NotFound, format!("XORB not found: {}", hash))
            }
            Err(e) => Response::error(ErrorCode::IoError, format!("Failed to read file: {}", e)),
        }
    }

    async fn handle_exists(&self, hash: &str) -> Response {
        // Parse the merkle hash
        if let Err(e) = Request::parse_hash(hash) {
            return Response::error(ErrorCode::InvalidHash, e);
        }

        // Check if the XORB file exists
        let xorb_path = self
            .storage_path
            .join("xorbs")
            .join(format!("default.{}", hash));

        let exists = xorb_path.exists();
        Response::Exists { exists }
    }

    async fn handle_upload_xorb(&self, data: &str) -> Response {
        use base64::Engine;

        // Decode base64 data
        let decoded = match base64::engine::general_purpose::STANDARD.decode(data) {
            Ok(d) => d,
            Err(e) => {
                return Response::error(ErrorCode::InvalidRequest, format!("Invalid base64: {}", e))
            }
        };

        // Compute hash of the data
        let hash = merklehash::compute_data_hash(&decoded);
        let hash_hex = hash.hex();

        // Ensure xorbs directory exists
        let xorbs_dir = self.storage_path.join("xorbs");
        if let Err(e) = tokio::fs::create_dir_all(&xorbs_dir).await {
            return Response::error(
                ErrorCode::IoError,
                format!("Failed to create xorbs directory: {}", e),
            );
        }

        // Write the XORB file
        let xorb_path = xorbs_dir.join(format!("default.{}", hash_hex));
        if let Err(e) = tokio::fs::write(&xorb_path, &decoded).await {
            return Response::error(ErrorCode::UploadFailed, format!("Failed to write XORB: {}", e));
        }

        Response::UploadSuccess { hash: hash_hex }
    }

    async fn handle_get_reconstruction_info(&self, hash: &str) -> Response {
        // Parse the merkle hash
        if let Err(e) = Request::parse_hash(hash) {
            return Response::error(ErrorCode::InvalidHash, e);
        }

        // For now, return a simple placeholder
        // TODO: Implement proper reconstruction info lookup from shards
        Response::error(
            ErrorCode::NotFound,
            "Reconstruction info lookup not yet implemented",
        )
    }
}

fn main() -> anyhow::Result<()> {
    // Initialize logging to stderr (stdout is used for protocol)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    let storage_path = resolve_storage_path();
    info!("cas-serve starting with storage: {:?}", storage_path);

    // Ensure storage directory exists
    std::fs::create_dir_all(&storage_path)?;

    let server = CasServer::new(storage_path);

    // Create tokio runtime for async operations
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    // Process requests from stdin
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                error!("Failed to read line: {}", e);
                break;
            }
        };

        // Skip empty lines
        if line.trim().is_empty() {
            continue;
        }

        debug!("Received: {}", line);

        // Parse request
        let request: Request = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let response = Response::error(
                    ErrorCode::InvalidRequest,
                    format!("Failed to parse request: {}", e),
                );
                let json = serde_json::to_string(&response)?;
                writeln!(stdout_lock, "{}", json)?;
                stdout_lock.flush()?;
                continue;
            }
        };

        // Check for shutdown request
        let is_shutdown = matches!(request, Request::Shutdown);

        // Handle request
        let response = rt.block_on(server.handle_request(request));

        // Send response
        let json = serde_json::to_string(&response)?;
        debug!("Sending: {}", json);
        writeln!(stdout_lock, "{}", json)?;
        stdout_lock.flush()?;

        // Exit on shutdown
        if is_shutdown {
            info!("Shutting down");
            break;
        }
    }

    info!("cas-serve exiting");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ping_pong() {
        let server = CasServer::new(PathBuf::from("/tmp/test-cas"));
        let response = server.handle_request(Request::Ping).await;
        assert!(matches!(response, Response::Pong));
    }

    #[tokio::test]
    async fn test_exists_not_found() {
        let server = CasServer::new(PathBuf::from("/tmp/test-cas-nonexistent"));
        let response = server
            .handle_request(Request::Exists {
                hash: "0".repeat(64), // Valid hex hash format
            })
            .await;
        match response {
            Response::Exists { exists } => assert!(!exists),
            _ => panic!("Expected Exists response"),
        }
    }
}
