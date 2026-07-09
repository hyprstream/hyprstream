//! Git object utilities for GitTorrent SHA256 operations

use crate::{types::{GitHash, ObjectFormat, Sha256Hash, GitRefs}, Result, Error};
use crate::crypto::hash::sha256_git;
use std::path::Path;
use std::collections::HashMap;
use git2::{Repository, ObjectType, Oid};

/// Check if a repository is exportable (has git-daemon-export-ok file)
pub fn is_exportable<P: AsRef<Path>>(repo_path: P) -> bool {
    let repo_path = repo_path.as_ref();

    // Try both bare and non-bare repository structures
    let export_ok_paths = [
        repo_path.join("git-daemon-export-ok"),
        repo_path.join(".git/git-daemon-export-ok"),
    ];

    export_ok_paths.iter().any(|path| path.exists())
}

/// Git object with its SHA256 hash and data
#[derive(Debug, Clone)]
pub struct GitObject {
    pub hash: Sha256Hash,
    pub object_type: GitObjectType,
    pub data: Vec<u8>,
    pub size: usize,
}

/// Git object types
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GitObjectType {
    Commit,
    Tree,
    Blob,
    Tag,
}

impl From<ObjectType> for GitObjectType {
    fn from(obj_type: ObjectType) -> Self {
        match obj_type {
            ObjectType::Commit => GitObjectType::Commit,
            ObjectType::Tree => GitObjectType::Tree,
            ObjectType::Tag => GitObjectType::Tag,
            // Blob and Any both map to Blob
            ObjectType::Blob | ObjectType::Any => GitObjectType::Blob,
        }
    }
}

/// Extract all Git objects from a repository
pub async fn extract_objects(repo_path: &Path) -> Result<Vec<GitObject>> {
    let repo = Repository::open(repo_path)?;
    let mut objects = Vec::new();

    // Get the object database
    let odb = repo.odb()?;

    // Collect all object IDs first
    let mut object_ids = Vec::new();
    odb.foreach(|oid| {
        object_ids.push(*oid);
        true // Continue iteration
    })?;

    // Process each object
    for oid in object_ids {
        if let Ok(obj) = odb.read(oid) {
            // Create git object format: "type size\0content"
            let type_str = match obj.kind() {
                ObjectType::Commit => "commit",
                ObjectType::Tree => "tree",
                ObjectType::Tag => "tag",
                // Blob and Any both render as "blob"
                ObjectType::Blob | ObjectType::Any => "blob",
            };

            let mut git_format = Vec::new();
            git_format.extend_from_slice(type_str.as_bytes());
            git_format.push(b' ');
            git_format.extend_from_slice(obj.len().to_string().as_bytes());
            git_format.push(0); // null terminator
            git_format.extend_from_slice(obj.data());

            // Calculate SHA256 hash of the git-formatted object
            let hash = sha256_git(&git_format)?;

            objects.push(GitObject {
                hash,
                object_type: obj.kind().into(),
                data: git_format,
                size: obj.len(),
            });
        }
    }

    tracing::info!("Extracted {} objects from repository", objects.len());
    Ok(objects)
}

/// Extract objects for a specific commit and its history
pub async fn extract_commit_objects(repo_path: &Path, commit_hash: &str) -> Result<Vec<GitObject>> {
    let repo = Repository::open(repo_path)?;
    let mut objects = Vec::new();
    let mut processed = std::collections::HashSet::new();

    // Start with the specified commit
    let commit_oid = Oid::from_str(commit_hash)?;
    let mut to_process = vec![commit_oid];

    while let Some(oid) = to_process.pop() {
        if processed.contains(&oid) {
            continue;
        }
        processed.insert(oid);

        // Get the object
        if let Ok(obj) = repo.find_object(oid, None) {
            // Add object to our list
            let git_object = create_git_object(&repo, &obj)?;
            objects.push(git_object);

            // Add referenced objects based on type
            match obj.kind() {
                Some(ObjectType::Commit) => {
                    if let Some(commit) = obj.as_commit() {
                        // Add tree
                        to_process.push(commit.tree_id());
                        // Add parents
                        for parent in commit.parent_ids() {
                            to_process.push(parent);
                        }
                    }
                }
                Some(ObjectType::Tree) => {
                    if let Some(tree) = obj.as_tree() {
                        for entry in tree.iter() {
                            to_process.push(entry.id());
                        }
                    }
                }
                _ => {} // Blobs and tags don't reference other objects
            }
        }
    }

    tracing::info!("Extracted {} objects for commit {}", objects.len(), commit_hash);
    Ok(objects)
}

/// Create a GitObject from a git2 Object
fn create_git_object(repo: &Repository, obj: &git2::Object<'_>) -> Result<GitObject> {
    let odb = repo.odb()?;
    let odb_obj = odb.read(obj.id())?;

    // Create git object format
    let type_str = match obj.kind() {
        Some(ObjectType::Commit) => "commit",
        Some(ObjectType::Tree) => "tree",
        Some(ObjectType::Tag) => "tag",
        // Blob, Any, or None all render as "blob"
        Some(ObjectType::Blob | ObjectType::Any) | None => "blob",
    };

    let mut git_format = Vec::new();
    git_format.extend_from_slice(type_str.as_bytes());
    git_format.push(b' ');
    git_format.extend_from_slice(odb_obj.len().to_string().as_bytes());
    git_format.push(0);
    git_format.extend_from_slice(odb_obj.data());

    let hash = sha256_git(&git_format)?;

    Ok(GitObject {
        hash,
        object_type: obj.kind().unwrap_or(ObjectType::Blob).into(),
        data: git_format,
        size: odb_obj.len(),
    })
}

/// Git commit object data
#[derive(Debug, Clone)]
pub struct CommitObject {
    pub tree_hash: GitHash,
    pub parent_hashes: Vec<GitHash>,
    pub author: String,
    pub committer: String,
    pub message: String,
}

/// Git tree entry
#[derive(Debug, Clone)]
pub struct TreeEntry {
    pub mode: TreeEntryMode,
    pub name: String,
    pub hash: GitHash,
}

/// Tree entry modes
#[derive(Debug, Clone, PartialEq)]
pub enum TreeEntryMode {
    File,        // 100644
    Executable,  // 100755
    Directory,   // 040000
    Symlink,     // 120000
    Gitlink,     // 160000 (submodule)
}

/// Git tree object data
#[derive(Debug, Clone)]
pub struct TreeObject {
    pub entries: Vec<TreeEntry>,
}

/// Parse a commit object from git format
pub fn parse_commit_object(data: &[u8]) -> Result<CommitObject> {
    // Parse git object header: "commit <size>\0<content>"
    let null_pos = data.iter().position(|&b| b == 0)
        .ok_or_else(|| Error::other("Invalid git object format"))?;

    let header = std::str::from_utf8(&data[..null_pos])?;
    let content = &data[null_pos + 1..];

    // Verify it's a commit
    if !header.starts_with("commit ") {
        return Err(Error::other("Not a commit object"));
    }

    let content_str = std::str::from_utf8(content)?;
    let mut tree_hash = None;
    let mut parent_hashes = Vec::new();
    let mut author = String::new();
    let mut committer = String::new();
    let mut message_start = None;

    for (i, line) in content_str.lines().enumerate() {
        if line.is_empty() {
            // Empty line indicates start of commit message
            message_start = Some(content_str.lines().skip(i + 1).collect::<Vec<_>>().join("\n"));
            break;
        }

        if let Some(hash_str) = line.strip_prefix("tree ") {
            // Parse hash (supports both SHA1 and SHA256)
            tree_hash = Some(GitHash::from_hex(hash_str.trim())?);
        } else if let Some(hash_str) = line.strip_prefix("parent ") {
            parent_hashes.push(GitHash::from_hex(hash_str.trim())?);
        } else if let Some(author_str) = line.strip_prefix("author ") {
            author = author_str.to_owned();
        } else if let Some(committer_str) = line.strip_prefix("committer ") {
            committer = committer_str.to_owned();
        }
    }

    Ok(CommitObject {
        tree_hash: tree_hash.ok_or_else(|| Error::other("Commit missing tree reference"))?,
        parent_hashes,
        author,
        committer,
        message: message_start.unwrap_or_default(),
    })
}

/// Parse a tree object from git format.
///
/// Tree entries store the referenced OID in **binary** form, which is not
/// self-describing: a SHA-1 repository uses 20-byte OIDs and a SHA-256
/// repository uses 32-byte OIDs. The caller must supply the repository's
/// [`ObjectFormat`] — typically derived from the tree object's own hash
/// (`GitHash::object_format`) — so the entries are sliced at the correct
/// width instead of a hardcoded 20.
pub fn parse_tree_object(data: &[u8], format: ObjectFormat) -> Result<TreeObject> {
    // Parse git object header: "tree <size>\0<content>"
    let null_pos = data.iter().position(|&b| b == 0)
        .ok_or_else(|| Error::other("Invalid git object format"))?;

    let header = std::str::from_utf8(&data[..null_pos])?;
    let mut content = &data[null_pos + 1..];

    // Verify it's a tree
    if !header.starts_with("tree ") {
        return Err(Error::other("Not a tree object"));
    }

    let mut entries = Vec::new();

    while !content.is_empty() {
        // Parse mode and name (terminated by null byte)
        let null_pos = content.iter().position(|&b| b == 0)
            .ok_or_else(|| Error::other("Invalid tree entry format"))?;

        let entry_header = std::str::from_utf8(&content[..null_pos])?;
        let space_pos = entry_header.find(' ')
            .ok_or_else(|| Error::other("Invalid tree entry header"))?;

        let mode_str = &entry_header[..space_pos];
        let name = &entry_header[space_pos + 1..];

        // Parse mode
        let mode = match mode_str {
            "040000" => TreeEntryMode::Directory,
            "100644" => TreeEntryMode::File,
            "100755" => TreeEntryMode::Executable,
            "120000" => TreeEntryMode::Symlink,
            "160000" => TreeEntryMode::Gitlink,
            _ => return Err(Error::other(format!("Unknown tree entry mode: {mode_str}"))),
        };

        // Hash is 20 bytes (SHA1) or 32 bytes (SHA256) after the null terminator.
        // The width is not encoded in the tree object itself, so it comes from the
        // repository's object format (carried in by the caller).
        let hash_size = format.hash_len();
        if content.len() < null_pos + 1 + hash_size {
            return Err(Error::other("Tree entry missing hash"));
        }

        let hash_bytes = &content[null_pos + 1..null_pos + 1 + hash_size];
        let hash_hex = hex::encode(hash_bytes);
        let hash = GitHash::from_hex(&hash_hex)?;

        entries.push(TreeEntry {
            mode,
            name: name.to_owned(),
            hash,
        });

        // Move to next entry
        content = &content[null_pos + 1 + hash_size..];
    }

    Ok(TreeObject { entries })
}

/// Parse a blob object from git format
pub fn parse_blob_object(data: &[u8]) -> Result<Vec<u8>> {
    // Parse git object header: "blob <size>\0<content>"
    let null_pos = data.iter().position(|&b| b == 0)
        .ok_or_else(|| Error::other("Invalid git object format"))?;

    let header = std::str::from_utf8(&data[..null_pos])?;

    // Verify it's a blob
    if !header.starts_with("blob ") {
        return Err(Error::other("Not a blob object"));
    }

    // Return the content after the header
    Ok(data[null_pos + 1..].to_vec())
}

/// Parse object references from git object data.
///
/// `format` is the repository's object format, needed to slice binary tree
/// entries at the correct OID width. Callers derive it from the hash of the
/// object being parsed (`GitHash::object_format`).
pub fn parse_object_references(object_data: &[u8], format: ObjectFormat) -> Result<Vec<GitHash>> {
    // Determine object type and parse accordingly
    let null_pos = object_data.iter().position(|&b| b == 0)
        .ok_or_else(|| Error::other("Invalid git object format"))?;

    let header = std::str::from_utf8(&object_data[..null_pos])?;

    if header.starts_with("commit ") {
        let commit = parse_commit_object(object_data)?;
        let mut refs = vec![commit.tree_hash];
        refs.extend(commit.parent_hashes);
        Ok(refs)
    } else if header.starts_with("tree ") {
        let tree = parse_tree_object(object_data, format)?;
        Ok(tree.entries.into_iter().map(|e| e.hash).collect())
    } else {
        // Blobs and tags don't reference other objects
        Ok(Vec::new())
    }
}

/// Write git objects to a repository
pub async fn write_objects(repo_path: &Path, objects: &[GitObject]) -> Result<()> {
    let repo = Repository::open(repo_path)?;
    let odb = repo.odb()?;

    for obj in objects {
        // Parse the git format to get type and content
        if let Some(null_pos) = obj.data.iter().position(|&b| b == 0) {
            let header = &obj.data[..null_pos];
            let content = &obj.data[null_pos + 1..];

            // Parse "type size" header
            if let Ok(header_str) = std::str::from_utf8(header) {
                let parts: Vec<&str> = header_str.split(' ').collect();
                if parts.len() == 2 {
                    let obj_type = match parts[0] {
                        "commit" => ObjectType::Commit,
                        "tree" => ObjectType::Tree,
                        "tag" => ObjectType::Tag,
                        // "blob" or unknown types default to Blob
                        _ => ObjectType::Blob,
                    };

                    // Write object to git database
                    let oid = odb.write(obj_type, content)?;
                    tracing::debug!("Wrote object {} as git object {}", obj.hash, oid);
                }
            }
        }
    }

    Ok(())
}

/// Convert git2::Oid to GitHash (supports both SHA1 and SHA256)
pub fn oid_to_hash(oid: Oid) -> Result<GitHash> {
    // git2 Oid.to_string() returns the hex representation
    // SHA1 = 40 hex chars, SHA256 = 64 hex chars
    GitHash::from_hex(&oid.to_string())
}

/// Convert GitHash back to git2::Oid
/// Note: git2 currently only supports SHA1, so SHA256 hashes will fail
pub fn hash_to_oid(hash: &GitHash) -> Result<Oid> {
    // git2 0.20.3 still primarily uses SHA1 Oids
    // SHA256 support in libgit2 1.9.2 is available but not yet exposed in git2-rs
    match hash {
        GitHash::Sha1(_) => {
            Oid::from_str(&hash.to_hex())
                .map_err(|e| Error::other(format!("Cannot convert hash to Oid: {e}")))
        }
        GitHash::Sha256(_) => {
            // SHA256 git repos are not yet widely supported
            // When git2 exposes SHA256 Oid support, this will work
            Err(Error::other("SHA256 Oids not yet supported by git2-rs"))
        }
    }
}

// Keep legacy function for backwards compatibility during migration
#[deprecated(note = "Use oid_to_hash instead - it supports both SHA1 and SHA256")]
pub fn oid_to_sha256(oid: Oid) -> Result<Sha256Hash> {
    let oid_str = oid.to_string();
    if oid_str.len() == 64 {
        Sha256Hash::new(oid_str)
    } else {
        Err(Error::other("SHA1 hashes are no longer supported, use SHA256"))
    }
}

// Keep legacy function for backwards compatibility during migration
#[deprecated(note = "Use hash_to_oid instead - it supports GitHash enum")]
pub fn sha256_to_oid(hash: &Sha256Hash) -> Result<Oid> {
    Oid::from_str(hash.as_str()).map_err(|e| Error::other(format!("Cannot convert SHA256 to Oid (git2 limitation): {e}")))
}


/// Extract git references from a repository
pub async fn extract_git_refs(repo_path: &Path) -> Result<GitRefs> {
    let repo = Repository::open(repo_path)?;
    let mut refs_map = HashMap::new();

    // Extract all references
    let refs = repo.references()?;
    for reference in refs {
        let reference = reference?;
        if let Some(name) = reference.name() {
            if let Some(target) = reference.target() {
                // Convert git2 Oid to GitHash (supports SHA1 and SHA256)
                let hash = oid_to_hash(target)?;
                refs_map.insert(name.to_owned(), hash);
            }
        }
    }

    // Get HEAD reference
    let head = repo.head()?;
    let head_name = head.name().unwrap_or("refs/heads/main").to_owned();

    // Current timestamp
    let created_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();

    Ok(GitRefs {
        refs: refs_map,
        head: head_name,
        created_at,
    })
}


#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use git2::Repository;
    use tempfile::TempDir;
    use std::fs;

    /// Build the raw bytes of a git tree object with a single entry whose OID
    /// is `oid` bytes wide, e.g. `"tree <size>\0100644 file\0<oid...>"`.
    fn make_tree_object(name: &str, oid: &[u8]) -> Vec<u8> {
        let mut content = Vec::new();
        content.extend_from_slice(b"100644 ");
        content.extend_from_slice(name.as_bytes());
        content.push(0);
        content.extend_from_slice(oid);

        let mut object = format!("tree {}\0", content.len()).into_bytes();
        object.extend_from_slice(&content);
        object
    }

    #[test]
    fn test_parse_tree_object_sha256_oid() {
        // A SHA-256 repository stores 32-byte OIDs in tree entries.
        let oid = [0xABu8; 32];
        let object = make_tree_object("modern.txt", &oid);

        let tree = parse_tree_object(&object, ObjectFormat::Sha256).unwrap();
        assert_eq!(tree.entries.len(), 1);
        let entry = &tree.entries[0];
        assert_eq!(entry.name, "modern.txt");
        assert_eq!(entry.hash, GitHash::Sha256(oid));
        assert_eq!(entry.hash.to_hex(), hex::encode(oid));
    }

    #[test]
    fn test_parse_tree_object_sha1_oid() {
        // A SHA-1 repository stores 20-byte OIDs; the legacy path must still work.
        let oid = [0x11u8; 20];
        let object = make_tree_object("legacy.txt", &oid);

        let tree = parse_tree_object(&object, ObjectFormat::Sha1).unwrap();
        assert_eq!(tree.entries.len(), 1);
        let entry = &tree.entries[0];
        assert_eq!(entry.name, "legacy.txt");
        assert_eq!(entry.hash, GitHash::Sha1(oid));
    }

    #[test]
    fn test_parse_tree_object_sha256_with_sha1_width_is_wrong() {
        // Regression guard for the hardcoded-20 bug: parsing a 32-byte-OID tree
        // as SHA-1 must NOT silently yield the correct 32-byte hash.
        let oid = [0xABu8; 32];
        let object = make_tree_object("modern.txt", &oid);

        // Parsing at the wrong (20-byte) width must not silently recover the
        // correct 32-byte hash: it either errors or yields a corrupt entry.
        // Threading the real object format is what avoids this.
        if let Ok(tree) = parse_tree_object(&object, ObjectFormat::Sha1) {
            assert_ne!(tree.entries[0].hash, GitHash::Sha256(oid));
        }
    }

    #[test]
    fn test_object_format_hash_len() {
        assert_eq!(ObjectFormat::Sha1.hash_len(), 20);
        assert_eq!(ObjectFormat::Sha256.hash_len(), 32);
        assert_eq!(GitHash::Sha1([0u8; 20]).object_format(), ObjectFormat::Sha1);
        assert_eq!(
            GitHash::Sha256([0u8; 32]).object_format(),
            ObjectFormat::Sha256
        );
    }

    #[tokio::test]
    async fn test_extract_objects() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;

        // Initialize repository and create a commit
        let repo = Repository::init(temp_dir.path())?;

        // Create a test file
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Hello World")?;

        // Add and commit
        let mut index = repo.index()?;
        index.add_path(Path::new("test.txt"))?;
        index.write()?;

        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let sig = git2::Signature::now("Test", "test@example.com")?;

        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            "Initial commit",
            &tree,
            &[],
        )?;

        // Extract objects
        let objects = extract_objects(temp_dir.path()).await?;

        // Should have at least: blob (file), tree, commit
        assert!(objects.len() >= 3);

        // Check that we have different object types
        let types: std::collections::HashSet<_> = objects.iter().map(|o| &o.object_type).collect();
        assert!(types.contains(&GitObjectType::Blob));
        assert!(types.contains(&GitObjectType::Tree));
        assert!(types.contains(&GitObjectType::Commit));
        Ok(())
    }

    #[test]
    fn test_is_exportable() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let temp_dir = TempDir::new()?;

        // Initialize a new repository
        let repo = Repository::init(temp_dir.path())?;
        drop(repo);

        // Should not be exportable initially
        assert!(!is_exportable(temp_dir.path()));

        // Create the export file
        let export_file = temp_dir.path().join(".git/git-daemon-export-ok");
        fs::write(export_file, "")?;

        // Should now be exportable
        assert!(is_exportable(temp_dir.path()));
        Ok(())
    }
}