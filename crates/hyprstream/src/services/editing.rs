//! CRDT-based collaborative file editing.
//!
//! Provides a shared document model using automerge for collaborative editing
//! of files within worktrees. Multiple clients can edit the same file via
//! separate fids, sharing a single `AutoCommit` document.
//!
//! ## Session lifecycle
//!
//! 1. `editOpen` — Read file, hash it, initialize automerge doc, create session
//! 2. `editState` — Serialize current automerge doc state to text (TOML/JSON/etc.)
//! 3. `editApply` — Apply automerge changes from a client
//! 4. `flush` — Write automerge state back to disk (atomic, checks file_hash)
//! 5. `editClose` — Close this client's session (decrements client_count)

use automerge::AutoCommit;
use automerge::transaction::Transactable;
use dashmap::DashMap;
use sha2::{Sha256, Digest};
use parking_lot::Mutex;
use std::sync::Arc;
use tracing::{debug, warn};

/// Document format for parsing/serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocFormat {
    Toml,
    Json,
    Yaml,
    Csv,
    Text,
}

/// Shared CRDT document for a file, accessed by multiple clients.
///
/// Always accessed through `Arc<Mutex<SharedDoc>>` — no inner locks needed.
pub struct SharedDoc {
    pub doc: AutoCommit,
    pub path: String,
    pub format: DocFormat,
    pub dirty: bool,
    pub file_hash: [u8; 32],
    pub client_count: usize,
}

/// Canonical key for a shared document: (repo_id, worktree_name, normalized_path).
pub type DocKey = (String, String, String);

/// Table of active CRDT editing sessions.
pub struct EditingTable {
    /// Shared docs keyed by canonical file identity.
    docs: DashMap<DocKey, Arc<Mutex<SharedDoc>>>,
    /// Maps (client_identity, fid) → DocKey for session lookup and cleanup.
    sessions: DashMap<(String, u32), DocKey>,
}

impl Default for EditingTable {
    fn default() -> Self {
        Self::new()
    }
}

impl EditingTable {
    pub fn new() -> Self {
        Self {
            docs: DashMap::new(),
            sessions: DashMap::new(),
        }
    }

    /// Open (or join) a CRDT editing session for a file.
    ///
    /// If no shared doc exists for this file, creates one from `content`.
    /// If one already exists, increments client_count and returns it.
    ///
    /// Returns error if `(client_id, fid)` already has an active session.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        &self,
        client_id: &str,
        fid: u32,
        repo_id: &str,
        worktree: &str,
        rel_path: &str,
        format: DocFormat,
        content: &str,
    ) -> Result<Arc<Mutex<SharedDoc>>, String> {
        let session_key = (client_id.to_owned(), fid);
        if self.sessions.contains_key(&session_key) {
            return Err(format!(
                "fid {} already has an active editing session", fid
            ));
        }

        let doc_key: DocKey = (
            repo_id.to_owned(),
            worktree.to_owned(),
            normalize_path(rel_path),
        );

        let shared = self.docs.entry(doc_key.clone()).or_insert_with(|| {
            let mut doc = AutoCommit::new();
            // Store the file content as a text value in the automerge doc
            doc.put(automerge::ROOT, "content", content).ok();
            let hash = sha256_hash(content.as_bytes());
            debug!("Created new shared doc for {}:{}", worktree, rel_path);
            Arc::new(Mutex::new(SharedDoc {
                doc,
                path: normalize_path(rel_path),
                format,
                dirty: false,
                file_hash: hash,
                client_count: 0,
            }))
        });

        // Increment client count
        {
            let mut sd = shared.lock();
            sd.client_count += 1;
        }

        self.sessions.insert(session_key, doc_key);
        Ok(Arc::clone(&shared))
    }

    /// Get the shared doc for a client's active session.
    pub fn get_session(
        &self,
        client_id: &str,
        fid: u32,
    ) -> Option<Arc<Mutex<SharedDoc>>> {
        let session_key = (client_id.to_owned(), fid);
        let doc_key = self.sessions.get(&session_key)?;
        let shared = self.docs.get(&*doc_key)?;
        Some(Arc::clone(&shared))
    }

    /// Check if a file has an active editing session (by any client).
    pub fn has_active_session(
        &self,
        repo_id: &str,
        worktree: &str,
        rel_path: &str,
    ) -> bool {
        let doc_key: DocKey = (
            repo_id.to_owned(),
            worktree.to_owned(),
            normalize_path(rel_path),
        );
        self.docs.contains_key(&doc_key)
    }

    /// Close a client's editing session. Decrements client_count.
    /// Removes the shared doc if count reaches 0.
    ///
    /// Returns true if the session existed and was closed.
    pub fn close(&self, client_id: &str, fid: u32) -> bool {
        let session_key = (client_id.to_owned(), fid);
        let Some((_, doc_key)) = self.sessions.remove(&session_key) else {
            return false;
        };

        let should_remove = {
            if let Some(shared) = self.docs.get(&doc_key) {
                let mut sd = shared.lock();
                sd.client_count = sd.client_count.saturating_sub(1);
                sd.client_count == 0
            } else {
                false
            }
        };

        if should_remove {
            self.docs.remove(&doc_key);
            debug!("Removed shared doc (last client closed): {:?}", doc_key);
        }

        true
    }

    /// Called when a fid is reaped by the idle reaper.
    /// Logs a warning if the session was dirty.
    pub fn on_reap(&self, client_id: &str, fid: u32) {
        let session_key = (client_id.to_owned(), fid);
        if let Some((_, doc_key)) = self.sessions.remove(&session_key) {
            let should_remove = {
                if let Some(shared) = self.docs.get(&doc_key) {
                    let mut sd = shared.lock();
                    if sd.dirty {
                        warn!(
                            "Reaping fid {} with dirty editing session for {:?}",
                            fid, doc_key
                        );
                    }
                    sd.client_count = sd.client_count.saturating_sub(1);
                    sd.client_count == 0
                } else {
                    false
                }
            };

            if should_remove {
                self.docs.remove(&doc_key);
            }
        }
    }

    /// Called when a fid is clunked. Same as close but explicit name.
    pub fn on_clunk(&self, client_id: &str, fid: u32) {
        self.close(client_id, fid);
    }

    /// Check if a specific fid has an active editing session for the given file path.
    /// Used by handle_write to guard against direct writes to CRDT-edited files.
    pub fn fid_has_session(&self, client_id: &str, fid: u32) -> bool {
        let session_key = (client_id.to_owned(), fid);
        self.sessions.contains_key(&session_key)
    }

    /// Check if a file at the given path has any active editing session.
    /// Used by handle_write to guard against direct writes to CRDT-edited files.
    pub fn file_has_session(
        &self,
        repo_id: &str,
        worktree: &str,
        rel_path: &str,
    ) -> bool {
        self.has_active_session(repo_id, worktree, rel_path)
    }
}

/// Normalize a path: no leading `./`, no trailing `/`, POSIX separators.
pub fn normalize_path(path: &str) -> String {
    let p = path
        .replace('\\', "/")
        .trim_start_matches("./")
        .trim_end_matches('/')
        .to_owned();
    if p.is_empty() { ".".to_owned() } else { p }
}

/// SHA-256 hash of bytes.
pub fn sha256_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}
