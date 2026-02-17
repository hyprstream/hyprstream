//! 9P convenience helpers for WorktreeClient.
//!
//! Provides high-level file operations that implement the walk/open/read-loop/clunk
//! pattern over the 9P protocol. These replace the old `read_file`/`write_file`/`mkdir`
//! convenience methods that sent entire files in a single message.

use crate::services::WorktreeClient;
use crate::services::types::{OREAD, OWRITE, DMDIR};
use anyhow::Result;
use std::sync::atomic::{AtomicU32, Ordering};

/// Atomic counter for allocating client-side fids.
static NEXT_CLIENT_FID: AtomicU32 = AtomicU32::new(100);

/// Allocate a client-side fid.
fn next_fid() -> u32 {
    NEXT_CLIENT_FID.fetch_add(1, Ordering::Relaxed)
}

/// Split a path string into 9P wname components.
fn split_path(path: &str) -> Vec<String> {
    path.split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .map(std::borrow::ToOwned::to_owned)
        .collect()
}

/// Stat result for backward compatibility with callers using the old API.
pub struct StatResult {
    pub exists: bool,
    pub is_dir: bool,
    pub size: u64,
    pub modified_at: i64,
}

impl WorktreeClient {
    /// Read an entire file via walk/open/read-loop/clunk (bounded by iounit per message).
    ///
    /// This is the replacement for the old `read_file()` method. Instead of sending
    /// the entire file in a single Cap'n Proto message (which fails for files >16MB),
    /// this reads in iounit-sized chunks via the 9P protocol.
    pub async fn read_file_chunked(&self, path: &str) -> Result<Vec<u8>> {
        let fid = next_fid();
        let wnames = split_path(path);

        // Walk to the file
        let _walk_resp = self.walk(0, fid, &wnames).await?;

        // Stat to get file size (for pre-allocation)
        let stat_resp = self.np_stat(fid).await?;
        let file_size = stat_resp.stat.length;

        // Open for reading
        let open_resp = self.open(fid, OREAD).await?;
        let iounit = open_resp.iounit;

        // Read loop — bounded by iounit per message
        let mut buf = Vec::with_capacity(file_size as usize);
        let mut offset = 0u64;
        loop {
            let resp = self.read(fid, offset, iounit).await?;
            if resp.data.is_empty() {
                break; // EOF
            }
            offset += resp.data.len() as u64;
            buf.extend_from_slice(&resp.data);
        }

        // Clunk the fid
        self.clunk(fid).await?;

        Ok(buf)
    }

    /// Write an entire file via walk-parent/create/write-loop/clunk.
    ///
    /// This is the replacement for the old `write_file()` method.
    pub async fn write_file_chunked(&self, path: &str, data: &[u8]) -> Result<()> {
        let components = split_path(path);
        let (parent_components, file_name) = if components.len() > 1 {
            (&components[..components.len() - 1], &components[components.len() - 1])
        } else if components.len() == 1 {
            // File is in the root directory
            (&components[..0], &components[0])
        } else {
            anyhow::bail!("empty path");
        };

        let fid = next_fid();

        // Walk to the parent directory
        let parent_wnames: Vec<String> = parent_components.to_vec();
        if parent_wnames.is_empty() {
            // Root directory — walk with empty wnames
            self.walk(0, fid, &[]).await?;
        } else {
            self.walk(0, fid, &parent_wnames).await?;
        }

        // Create the file (opens it for writing)
        let create_resp = self.create(fid, file_name, 0o644, OWRITE).await?;
        let iounit = create_resp.iounit;

        // Write in iounit-sized chunks
        let mut offset = 0u64;
        while offset < data.len() as u64 {
            let end = std::cmp::min(offset + iounit as u64, data.len() as u64);
            let chunk = &data[offset as usize..end as usize];
            let write_resp = self.write(fid, offset, chunk).await?;
            offset += write_resp.count as u64;
        }

        // Clunk the fid
        self.clunk(fid).await?;

        Ok(())
    }

    /// Create a directory (and parents if needed), like `mkdir -p`.
    ///
    /// This replaces the old `mkdir(path, recursive)` method.
    pub async fn mkdir_p(&self, path: &str) -> Result<()> {
        let components = split_path(path);
        if components.is_empty() {
            return Ok(());
        }

        // Walk + create each component
        let mut current_fid = next_fid();
        // Start at root
        self.walk(0, current_fid, &[]).await?;

        for component in &components {
            let child_fid = next_fid();
            // Try to walk to the component first (it may already exist)
            match self.walk(current_fid, child_fid, std::slice::from_ref(component)).await {
                Ok(_) => {
                    // Directory exists, clunk the old fid and continue
                    self.clunk(current_fid).await?;
                    current_fid = child_fid;
                }
                Err(_) => {
                    // Directory doesn't exist — create it
                    self.create(current_fid, component, DMDIR | 0o755, OREAD).await?;
                    // After create, the fid now points to the new directory
                    // current_fid is mutated by the server (fid now = new dir)
                }
            }
        }

        self.clunk(current_fid).await?;
        Ok(())
    }

    /// Stat a file by path. Returns a StatResult for backward compatibility.
    ///
    /// This replaces the old `stat(path)` method.
    pub async fn stat_path(&self, path: &str) -> Result<StatResult> {
        let fid = next_fid();
        let wnames = split_path(path);

        // Try to walk to the path
        match self.walk(0, fid, &wnames).await {
            Ok(_) => {
                let stat_resp = self.np_stat(fid).await?;
                self.clunk(fid).await?;

                let is_dir = (stat_resp.stat.qid.qtype & 0x80) != 0;
                let modified_at = stat_resp.stat.mtime as i64;
                Ok(StatResult {
                    exists: true,
                    is_dir,
                    size: stat_resp.stat.length,
                    modified_at,
                })
            }
            Err(_) => {
                // Path doesn't exist
                Ok(StatResult {
                    exists: false,
                    is_dir: false,
                    size: 0,
                    modified_at: 0,
                })
            }
        }
    }

    /// Remove a file by path via walk/remove.
    ///
    /// This replaces the old `remove(path)` method.
    pub async fn remove_path(&self, path: &str) -> Result<()> {
        let fid = next_fid();
        let wnames = split_path(path);

        self.walk(0, fid, &wnames).await?;
        self.remove(fid).await?;

        Ok(())
    }

    /// Copy a file by path via read + write.
    ///
    /// This replaces the old `copy(src, dst)` method. Since 9P has no copy op,
    /// this reads the source file and writes it to the destination.
    pub async fn copy_path(&self, src: &str, dst: &str) -> Result<()> {
        let data = self.read_file_chunked(src).await?;
        self.write_file_chunked(dst, &data).await
    }

    /// List directory entries by path.
    ///
    /// This replaces the old `list_dir(path)` method. Returns directory entries
    /// by walking to the directory, opening it, and reading the dir entries.
    /// For now, implemented via walk + stat on the directory.
    pub async fn list_dir_path(&self, path: &str) -> Result<Vec<super::FsDirEntryInfo>> {
        // Walk to the directory and open it for reading
        let fid = next_fid();
        let wnames = split_path(path);

        self.walk(0, fid, &wnames).await?;
        let open_resp = self.open(fid, OREAD).await?;
        let iounit = open_resp.iounit;

        // Read directory entries (9P encodes them as serialized stat entries)
        // For simplicity in this initial implementation, we read the raw data
        // and parse it. The server encodes FsDirEntryInfo in the read response.
        let mut entries = Vec::new();
        let mut offset = 0u64;
        loop {
            let resp = self.read(fid, offset, iounit).await?;
            if resp.data.is_empty() {
                break;
            }
            // Parse directory entries from the response data
            // Each entry is: name_len(u32) + name(utf8) + is_dir(u8) + size(u64)
            let mut cursor = 0;
            while cursor + 4 <= resp.data.len() {
                let name_len = u32::from_le_bytes(
                    resp.data[cursor..cursor + 4].try_into().unwrap_or([0; 4])
                ) as usize;
                cursor += 4;
                if cursor + name_len > resp.data.len() { break; }
                let name = String::from_utf8_lossy(&resp.data[cursor..cursor + name_len]).to_string();
                cursor += name_len;
                if cursor + 9 > resp.data.len() { break; }
                let is_dir = resp.data[cursor] != 0;
                cursor += 1;
                let size = u64::from_le_bytes(
                    resp.data[cursor..cursor + 8].try_into().unwrap_or([0; 8])
                );
                cursor += 8;
                entries.push(super::FsDirEntryInfo { name, is_dir, size });
            }
            offset += resp.data.len() as u64;
        }

        self.clunk(fid).await?;
        Ok(entries)
    }
}
