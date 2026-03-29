//! Inode table mapping u64 inode numbers to VFS path components.
//!
//! The FUSE protocol addresses files by inode number. This table translates
//! between FUSE inodes and VFS paths, maintaining lookup counts for proper
//! lifetime management via FUSE `forget()`.

use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;

/// Root inode, always 1 per FUSE convention.
pub const ROOT_INODE: u64 = 1;

/// Data associated with a single inode.
pub struct InodeData {
    /// Path components from root to this node (empty for root).
    pub path: Vec<String>,
    /// FUSE lookup count — decremented by forget(), entry removed at zero.
    pub lookup_count: AtomicU64,
    /// Whether this inode represents a directory.
    pub is_dir: bool,
}

impl InodeData {
    fn new(path: Vec<String>, is_dir: bool) -> Self {
        Self {
            path,
            lookup_count: AtomicU64::new(1),
            is_dir,
        }
    }
}

/// Concurrent inode table for FUSE inode <-> VFS path translation.
///
/// Thread-safe: uses `DashMap` for lock-free concurrent access from
/// multiple FUSE worker threads.
pub struct InodeTable {
    inodes: DashMap<u64, InodeData>,
    /// Reverse map: path -> inode for dedup on repeated lookups.
    paths: DashMap<Vec<String>, u64>,
    next_inode: AtomicU64,
}

impl InodeTable {
    /// Create a new inode table with the root inode pre-allocated.
    pub fn new() -> Self {
        let table = Self {
            inodes: DashMap::new(),
            paths: DashMap::new(),
            next_inode: AtomicU64::new(ROOT_INODE + 1),
        };
        // Root inode always exists.
        table
            .inodes
            .insert(ROOT_INODE, InodeData::new(vec![], true));
        table.paths.insert(vec![], ROOT_INODE);
        table
    }

    /// Look up or allocate an inode for the given path.
    ///
    /// If the path already has an inode, increments its lookup count.
    /// Otherwise allocates a new inode number.
    ///
    /// Returns the inode number.
    pub fn lookup_or_insert(&self, path: Vec<String>, is_dir: bool) -> u64 {
        // Check if already exists.
        if let Some(existing) = self.paths.get(&path) {
            let ino = *existing;
            if let Some(entry) = self.inodes.get(&ino) {
                entry.lookup_count.fetch_add(1, Ordering::Relaxed);
            }
            return ino;
        }

        // Allocate new inode.
        let ino = self.next_inode.fetch_add(1, Ordering::Relaxed);
        self.inodes.insert(ino, InodeData::new(path.clone(), is_dir));
        self.paths.insert(path, ino);
        ino
    }

    /// Get a reference to inode data.
    pub fn get(&self, ino: u64) -> Option<dashmap::mapref::one::Ref<'_, u64, InodeData>> {
        self.inodes.get(&ino)
    }

    /// Get the path components for an inode.
    pub fn path_of(&self, ino: u64) -> Option<Vec<String>> {
        self.inodes.get(&ino).map(|e| e.path.clone())
    }

    /// Decrement lookup count. Removes the inode when count reaches zero
    /// (unless it is the root inode, which is never removed).
    pub fn forget(&self, ino: u64, nlookup: u64) {
        if ino == ROOT_INODE {
            return;
        }
        let should_remove = self
            .inodes
            .get(&ino)
            .map(|entry| {
                let prev = entry.lookup_count.fetch_sub(nlookup, Ordering::Relaxed);
                prev <= nlookup
            })
            .unwrap_or(false);

        if should_remove {
            if let Some((_, data)) = self.inodes.remove(&ino) {
                self.paths.remove(&data.path);
            }
        }
    }

    /// Number of active inodes (including root).
    pub fn len(&self) -> usize {
        self.inodes.len()
    }

    /// Whether the table contains only the root inode.
    pub fn is_empty(&self) -> bool {
        self.inodes.len() <= 1
    }
}

impl Default for InodeTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_inode_exists() {
        let table = InodeTable::new();
        assert!(table.get(ROOT_INODE).is_some());
        let root = table.get(ROOT_INODE);
        assert!(root.as_ref().is_some_and(|r| r.is_dir));
        assert!(root.as_ref().is_some_and(|r| r.path.is_empty()));
    }

    #[test]
    fn lookup_allocates_new_inode() {
        let table = InodeTable::new();
        let ino = table.lookup_or_insert(vec!["srv".to_string()], true);
        assert!(ino > ROOT_INODE);
        assert_eq!(table.len(), 2);

        let data = table.get(ino);
        assert!(data.as_ref().is_some_and(|d| d.is_dir));
        assert!(
            data.as_ref()
                .is_some_and(|d| d.path == vec!["srv".to_string()])
        );
    }

    #[test]
    fn lookup_deduplicates() {
        let table = InodeTable::new();
        let ino1 = table.lookup_or_insert(vec!["srv".to_string()], true);
        let ino2 = table.lookup_or_insert(vec!["srv".to_string()], true);
        assert_eq!(ino1, ino2);
        assert_eq!(table.len(), 2); // root + srv

        // Lookup count should be 2 (initial + re-lookup).
        let data = table.get(ino1);
        assert!(
            data.as_ref()
                .is_some_and(|d| d.lookup_count.load(Ordering::Relaxed) == 2)
        );
    }

    #[test]
    fn forget_removes_inode() {
        let table = InodeTable::new();
        let ino = table.lookup_or_insert(vec!["tmp".to_string()], false);
        assert_eq!(table.len(), 2);

        table.forget(ino, 1);
        assert_eq!(table.len(), 1); // Only root remains.
        assert!(table.get(ino).is_none());
    }

    #[test]
    fn forget_root_is_noop() {
        let table = InodeTable::new();
        table.forget(ROOT_INODE, 100);
        assert!(table.get(ROOT_INODE).is_some());
    }

    #[test]
    fn forget_partial_keeps_inode() {
        let table = InodeTable::new();
        let ino = table.lookup_or_insert(vec!["a".to_string()], true);
        // Bump lookup count to 2.
        table.lookup_or_insert(vec!["a".to_string()], true);

        table.forget(ino, 1);
        // Should still exist with count 1.
        assert!(table.get(ino).is_some());
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn hierarchical_paths() {
        let table = InodeTable::new();
        let srv = table.lookup_or_insert(vec!["srv".to_string()], true);
        let model = table.lookup_or_insert(vec!["srv".to_string(), "model".to_string()], true);
        let status = table.lookup_or_insert(
            vec![
                "srv".to_string(),
                "model".to_string(),
                "status".to_string(),
            ],
            false,
        );

        assert_ne!(srv, model);
        assert_ne!(model, status);
        assert_eq!(table.len(), 4); // root + 3 nodes

        assert_eq!(table.path_of(status), Some(vec![
            "srv".to_string(),
            "model".to_string(),
            "status".to_string(),
        ]));
    }
}
