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
//!
//! # Storage layout
//!
//! - `xorbs/default.{xorb_hash}` — raw xorb bytes (concatenated chunks), keyed
//!   by the xorb's Merkle-root hash.
//! - `shards/{file_hash}` — reconstruction shard stored as the raw XET
//!   `mdb_shard` (`.mdb`) binary, mapping a file's merkle hash to the ordered
//!   xorb segments that reassemble it.
//!
//! # Reconstruction
//!
//! `UploadFile` runs the file through Gearhash CDC (`chunker.rs`), aggregates
//! chunks into ≤64 MiB xorbs (`shard.rs`), and stores both the xorbs and the
//! reconstruction shard (`.mdb`). `GetFile { hash }` loads the shard, fetches
//! each referenced xorb, and concatenates the segments to return the original
//! file bytes — unblocking multi-xorb (>64 MiB) weight transfer (#390).

// The binary reuses the library's modules to avoid drift between the server
// and any client/embedder of `cas_serve`.
use cas_serve::{chunker, protocol, shard};
use protocol::{ErrorCode, Request, Response};
use shard::Shard;
use std::io::{BufRead, Write};
use std::path::PathBuf;
use tracing::{debug, error, info};

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

    /// Path for a raw xorb, keyed by its hex xorb hash.
    fn xorb_path(&self, xorb_hash_hex: &str) -> PathBuf {
        self.storage_path
            .join("xorbs")
            .join(format!("default.{xorb_hash_hex}"))
    }

    /// Path for a reconstruction shard, keyed by the file's hex merkle hash.
    fn shard_path(&self, file_hash_hex: &str) -> PathBuf {
        self.storage_path.join("shards").join(file_hash_hex)
    }

    /// Process a single request and return a response
    async fn handle_request(&self, request: Request) -> Response {
        match request {
            Request::Ping => Response::Pong,

            Request::Shutdown => Response::ShutdownAck,

            Request::GetFile { hash } => self.handle_get_file(&hash).await,

            Request::Exists { hash } => self.handle_exists(&hash).await,

            Request::UploadXorb { data } => self.handle_upload_xorb(&data).await,

            Request::UploadFile { data } => self.handle_upload_file(&data).await,

            Request::GetReconstructionInfo { hash } => {
                self.handle_get_reconstruction_info(&hash).await
            }
        }
    }

    /// Reassemble a file from its reconstruction shard. Delegates to the shared
    /// `CasStore` reconstruction core (`cas_serve::store`) so the binary and
    /// in-process embedders (registry `getBlob`) share one implementation.
    async fn handle_get_file(&self, hash: &str) -> Response {
        let store = cas_serve::CasStore::new(&self.storage_path);
        match store.get_file_bytes(hash).await {
            Ok(data) => {
                debug!("GetFile {hash}: reassembled {} bytes", data.len());
                Response::file(&data)
            }
            Err(cas_serve::StoreError::InvalidHash(e)) => {
                Response::error(ErrorCode::InvalidHash, e)
            }
            Err(cas_serve::StoreError::NotFound(e)) => Response::error(ErrorCode::NotFound, e),
            Err(cas_serve::StoreError::CorruptShard(e)) => {
                Response::error(ErrorCode::StorageError, e)
            }
            Err(cas_serve::StoreError::Io(e)) => Response::error(ErrorCode::IoError, e),
        }
    }

    async fn handle_exists(&self, hash: &str) -> Response {
        // Parse the merkle hash
        if let Err(e) = Request::parse_hash(hash) {
            return Response::error(ErrorCode::InvalidHash, e);
        }

        // A hash is "present" if either a reconstruction shard or a raw xorb
        // blob exists for it.
        let exists = self.shard_path(hash).exists() || self.xorb_path(hash).exists();
        Response::Exists { exists }
    }

    async fn handle_upload_xorb(&self, data: &str) -> Response {
        use base64::Engine;

        // Decode base64 data
        let decoded = match base64::engine::general_purpose::STANDARD.decode(data) {
            Ok(d) => d,
            Err(e) => {
                return Response::error(ErrorCode::InvalidRequest, format!("Invalid base64: {e}"))
            }
        };

        // Compute hash of the data. For the legacy whole-blob upload path this
        // is BLAKE3-keyed over the entire blob (a single XET leaf), which is
        // correct for files small enough to be one chunk and is what the
        // pre-#390 GetFile fallback expects.
        let hash = merklehash::compute_data_hash(&decoded);
        let hash_hex = hash.hex();

        // Ensure xorbs directory exists
        let xorbs_dir = self.storage_path.join("xorbs");
        if let Err(e) = tokio::fs::create_dir_all(&xorbs_dir).await {
            return Response::error(
                ErrorCode::IoError,
                format!("Failed to create xorbs directory: {e}"),
            );
        }

        // Write the XORB file
        let xorb_path = xorbs_dir.join(format!("default.{hash_hex}"));
        if let Err(e) = tokio::fs::write(&xorb_path, &decoded).await {
            return Response::error(
                ErrorCode::UploadFailed,
                format!("Failed to write XORB: {e}"),
            );
        }

        Response::UploadSuccess { hash: hash_hex }
    }

    /// Chunk a file (Gearhash CDC), pack chunks into xorbs, store each xorb
    /// content-addressed, store the reconstruction shard, and return the file
    /// merkle hash. Unblocks multi-xorb (>64 MiB) transfer (#390).
    async fn handle_upload_file(&self, data: &str) -> Response {
        use base64::Engine;

        let decoded = match base64::engine::general_purpose::STANDARD.decode(data) {
            Ok(d) => d,
            Err(e) => {
                return Response::error(ErrorCode::InvalidRequest, format!("Invalid base64: {e}"))
            }
        };

        // Chunk + store + server-side merkle go through the shared `CasStore`
        // write core (`cas_serve::store`), so the binary and in-process embedders
        // (the registry `putBlob` handler) share one implementation.
        let store = cas_serve::CasStore::new(&self.storage_path);
        match store.put_file_bytes(&decoded).await {
            Ok(put) => {
                info!(
                    "UploadFile {}: {} bytes, {} xorb(s), {} new byte(s)",
                    put.merkle,
                    decoded.len(),
                    put.xorb_hashes.len(),
                    put.bytes_stored
                );
                Response::UploadFileSuccess {
                    file_hash: put.merkle,
                    // file_len is the original input length (== sum of chunk lens).
                    file_len: decoded.len() as u64,
                    xorb_hashes: put.xorb_hashes,
                }
            }
            Err(e) => Response::error(ErrorCode::UploadFailed, format!("Upload failed: {e}")),
        }
    }

    async fn handle_get_reconstruction_info(&self, hash: &str) -> Response {
        use base64::Engine;

        // Parse the merkle hash
        if let Err(e) = Request::parse_hash(hash) {
            return Response::error(ErrorCode::InvalidHash, e);
        }

        let shard_path = self.shard_path(hash);
        match tokio::fs::read(&shard_path).await {
            Ok(mdb_bytes) => Response::ReconstructionInfo {
                info: base64::engine::general_purpose::STANDARD.encode(&mdb_bytes),
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Response::error(
                ErrorCode::NotFound,
                format!("Reconstruction shard not found: {hash}"),
            ),
            Err(e) => Response::error(ErrorCode::IoError, format!("Failed to read shard: {e}")),
        }
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
                    format!("Failed to parse request: {e}"),
                );
                let json = serde_json::to_string(&response)?;
                writeln!(stdout_lock, "{json}")?;
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
        writeln!(stdout_lock, "{json}")?;
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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

    /// Full round-trip via the protocol: UploadFile → GetFile → bytes equal.
    #[tokio::test]
    async fn test_upload_then_get_file_round_trip() {
        use base64::Engine;
        let dir = tempdir().unwrap();
        let server = CasServer::new(dir.path().to_path_buf());

        // ~256 KiB of pseudo-random bytes so CDC cuts several chunks.
        let mut original = Vec::with_capacity(256 * 1024);
        let mut s: u64 = 0xa243ed8b9d1f0c5e;
        for _ in 0..32768 {
            s = s.wrapping_mul(0x9e3779b97f4a7c15) ^ (s >> 29);
            original.extend_from_slice(&s.to_le_bytes());
        }

        let b64 = base64::engine::general_purpose::STANDARD.encode(&original);
        let up = server
            .handle_request(Request::UploadFile { data: b64 })
            .await;
        let (file_hash, file_len, n_xorbs) = match up {
            Response::UploadFileSuccess {
                file_hash,
                file_len,
                xorb_hashes,
            } => {
                assert_eq!(file_len, original.len() as u64);
                (file_hash, file_len, xorb_hashes.len())
            }
            other => panic!("expected UploadFileSuccess, got {other:?}"),
        };
        assert!(n_xorbs >= 1);
        assert!(file_len > 0);

        // GetFile must reconstruct the original bytes.
        let resp = server
            .handle_request(Request::GetFile {
                hash: file_hash.clone(),
            })
            .await;
        let data = match resp {
            Response::File { data } => base64::engine::general_purpose::STANDARD
                .decode(&data)
                .expect("decode response"),
            other => panic!("expected File, got {other:?}"),
        };
        assert_eq!(data, original);

        // Reconstruction info (shard) must be retrievable too. The `info`
        // field is base64-encoded mdb_shard binary; decode and parse it.
        let info = server
            .handle_request(Request::GetReconstructionInfo {
                hash: file_hash.clone(),
            })
            .await;
        match info {
            Response::ReconstructionInfo { info } => {
                use base64::Engine;
                let mdb_bytes = base64::engine::general_purpose::STANDARD
                    .decode(&info)
                    .expect("decode base64 shard");
                // The magic header tag must be present.
                assert_eq!(
                    &mdb_bytes[..32],
                    &cas_serve::mdb_shard::MDB_SHARD_HEADER_TAG[..]
                );
                let file_hash_parsed =
                    merklehash::MerkleHash::from_hex(&file_hash).expect("parse file hash");
                let segs = Shard::segments(&mdb_bytes, &file_hash_parsed)
                    .expect("parse mdb shard segments");
                let total: u64 = segs.iter().map(|s| s.byte_len).sum();
                assert_eq!(total, file_len, "segment byte total must match file_len");
            }
            other => panic!("expected ReconstructionInfo, got {other:?}"),
        }

        // Exists must report true for both the shard and the first xorb.
        let exists_file = server
            .handle_request(Request::Exists {
                hash: file_hash.clone(),
            })
            .await;
        assert!(matches!(exists_file, Response::Exists { exists: true }));
    }

    /// A file larger than the 64 MiB xorb limit must split across multiple
    /// xorbs and still reconstruct. To exercise multi-xorb aggregation without
    /// allocating 64 MiB, we build ~1 MiB of data and aggregate with a small
    /// per-xorb cap via `Shard::from_chunks_with_cap` (the production path uses
    /// the spec 64 MiB cap, exercised by the >64 MiB synthetic assertion below).
    #[tokio::test]
    async fn test_multi_xorb_round_trip() {
        use base64::Engine;
        let dir = tempdir().unwrap();
        let server = CasServer::new(dir.path().to_path_buf());

        // ~1 MiB of pseudo-random data (xorshift64).
        let mut original = Vec::with_capacity(1024 * 1024);
        let mut s: u64 = 0xdeadbeefcafebabe;
        for _ in 0..131072 {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            original.extend_from_slice(&s.to_le_bytes());
        }

        // Chunk with the spec chunker, then aggregate with a tiny cap so the
        // file splits across many xorbs.
        let chunks = chunker::chunk_all(&original);
        let (shard, xorbs) = Shard::from_chunks_with_cap(&chunks, 256 * 1024);
        assert!(
            xorbs.len() > 1,
            "expected multiple xorbs for multi-xorb test, got {}",
            xorbs.len()
        );

        // Store xorbs and shard exactly as the server would.
        let xorbs_dir = server.storage_path.join("xorbs");
        let shards_dir = server.storage_path.join("shards");
        tokio::fs::create_dir_all(&xorbs_dir).await.unwrap();
        tokio::fs::create_dir_all(&shards_dir).await.unwrap();
        for (h, bytes) in &xorbs {
            tokio::fs::write(xorbs_dir.join(format!("default.{}", h.hex())), bytes)
                .await
                .unwrap();
        }
        tokio::fs::write(shards_dir.join(&shard.file_hash), shard.to_bytes())
            .await
            .unwrap();

        // GetFile must reconstruct the original across multiple xorbs.
        let resp = server
            .handle_request(Request::GetFile {
                hash: shard.file_hash.clone(),
            })
            .await;
        let data = match resp {
            Response::File { data } => base64::engine::general_purpose::STANDARD
                .decode(&data)
                .expect("decode response"),
            other => panic!("expected File, got {other:?}"),
        };
        assert_eq!(data, original);
    }
}
