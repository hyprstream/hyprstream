//! Wire protocol for cas-serve
//!
//! Uses newline-delimited JSON (NDJSON) for simplicity and debuggability.
//! Each message is a single line of JSON followed by a newline.

use serde::{Deserialize, Serialize};

/// Request types that can be sent to cas-serve
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    /// Download a file by its merkle hash
    GetFile {
        /// Hex-encoded merkle hash
        hash: String,
    },

    /// Check if a XORB exists
    Exists {
        /// Hex-encoded merkle hash
        hash: String,
    },

    /// Upload a XORB
    UploadXorb {
        /// Base64-encoded XORB data
        data: String,
    },

    /// Get file reconstruction info
    GetReconstructionInfo {
        /// Hex-encoded merkle hash
        hash: String,
    },

    /// Ping to check if server is alive
    Ping,

    /// Graceful shutdown
    Shutdown,
}

/// Response types returned by cas-serve
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Response {
    /// Successful file retrieval
    File {
        /// Base64-encoded file data
        data: String,
    },

    /// XORB existence check result
    Exists {
        /// Whether the XORB exists
        exists: bool,
    },

    /// Upload success
    UploadSuccess {
        /// Hex-encoded merkle hash of uploaded XORB
        hash: String,
    },

    /// File reconstruction info
    ReconstructionInfo {
        /// JSON-serialized reconstruction info
        info: String,
    },

    /// Pong response to ping
    Pong,

    /// Error response
    Error {
        /// Error code for programmatic handling
        code: ErrorCode,
        /// Human-readable error message
        message: String,
    },

    /// Shutdown acknowledgment
    ShutdownAck,
}

/// Error codes for programmatic error handling
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// Invalid request format
    InvalidRequest,
    /// Hash not found in storage
    NotFound,
    /// IO error during operation
    IoError,
    /// Storage backend error
    StorageError,
    /// Invalid hash format
    InvalidHash,
    /// Upload failed
    UploadFailed,
    /// Internal server error
    InternalError,
}

impl Response {
    /// Create an error response
    pub fn error(code: ErrorCode, message: impl Into<String>) -> Self {
        Response::Error {
            code,
            message: message.into(),
        }
    }

    /// Create a file response from raw bytes
    pub fn file(data: &[u8]) -> Self {
        use base64::Engine;
        Response::File {
            data: base64::engine::general_purpose::STANDARD.encode(data),
        }
    }
}

impl Request {
    /// Parse hash from hex string
    pub fn parse_hash(hash: &str) -> Result<merklehash::MerkleHash, String> {
        merklehash::MerkleHash::from_hex(hash).map_err(|e| format!("Invalid hash: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_serialization() -> Result<(), serde_json::Error> {
        let req = Request::GetFile {
            hash: "abc123".to_owned(),
        };
        let json = serde_json::to_string(&req)?;
        assert!(json.contains("get_file"));
        assert!(json.contains("abc123"));

        let parsed: Request = serde_json::from_str(&json)?;
        match parsed {
            Request::GetFile { hash } => assert_eq!(hash, "abc123"),
            _ => panic!("Wrong variant"),
        }
        Ok(())
    }

    #[test]
    fn test_response_serialization() -> Result<(), serde_json::Error> {
        let resp = Response::error(ErrorCode::NotFound, "File not found");
        let json = serde_json::to_string(&resp)?;
        assert!(json.contains("error"));
        assert!(json.contains("not_found"));

        let parsed: Response = serde_json::from_str(&json)?;
        match parsed {
            Response::Error { code, message } => {
                assert_eq!(code, ErrorCode::NotFound);
                assert_eq!(message, "File not found");
            }
            _ => panic!("Wrong variant"),
        }
        Ok(())
    }
}
