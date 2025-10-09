//! Git object utilities for GitTorrent SHA256 operations

use crate::{types::{Sha256Hash, GitRefs}, Result, Error};
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
#[derive(Debug, Clone, PartialEq)]
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
            ObjectType::Blob => GitObjectType::Blob,
            ObjectType::Tag => GitObjectType::Tag,
            _ => GitObjectType::Blob, // fallback
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
                ObjectType::Blob => "blob",
                ObjectType::Tag => "tag",
                ObjectType::Any => "blob", // Default to blob for Any
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

            // Add referenced objects
            match obj.kind() {
                Some(ObjectType::Commit) => {
                    let commit = obj.as_commit().unwrap();
                    // Add tree
                    to_process.push(commit.tree_id());
                    // Add parents
                    for parent in commit.parent_ids() {
                        to_process.push(parent);
                    }
                }
                Some(ObjectType::Tree) => {
                    let tree = obj.as_tree().unwrap();
                    for entry in tree.iter() {
                        to_process.push(entry.id());
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
fn create_git_object(repo: &Repository, obj: &git2::Object) -> Result<GitObject> {
    let odb = repo.odb()?;
    let odb_obj = odb.read(obj.id())?;

    // Create git object format
    let type_str = match obj.kind() {
        Some(ObjectType::Commit) => "commit",
        Some(ObjectType::Tree) => "tree",
        Some(ObjectType::Blob) => "blob",
        Some(ObjectType::Tag) => "tag",
        _ => "blob",
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
    pub tree_hash: Sha256Hash,
    pub parent_hashes: Vec<Sha256Hash>,
    pub author: String,
    pub committer: String,
    pub message: String,
}

/// Git tree entry
#[derive(Debug, Clone)]
pub struct TreeEntry {
    pub mode: TreeEntryMode,
    pub name: String,
    pub hash: Sha256Hash,
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
            // Expect full SHA256 hash (64 hex characters)
            tree_hash = Some(Sha256Hash::new(hash_str.trim())?);
        } else if let Some(hash_str) = line.strip_prefix("parent ") {
            parent_hashes.push(Sha256Hash::new(hash_str.trim())?);
        } else if let Some(author_str) = line.strip_prefix("author ") {
            author = author_str.to_string();
        } else if let Some(committer_str) = line.strip_prefix("committer ") {
            committer = committer_str.to_string();
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

/// Parse a tree object from git format
pub fn parse_tree_object(data: &[u8]) -> Result<TreeObject> {
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
            _ => return Err(Error::other(format!("Unknown tree entry mode: {}", mode_str))),
        };

        // SHA256 hash is 32 bytes after the null terminator
        if content.len() < null_pos + 1 + 32 {
            return Err(Error::other("Tree entry missing SHA256 hash"));
        }

        let hash_bytes = &content[null_pos + 1..null_pos + 1 + 32];
        // Expect full SHA256 hash (32 bytes = 64 hex characters)
        let hash_hex = hex::encode(hash_bytes);
        let hash = Sha256Hash::new(hash_hex)?;

        entries.push(TreeEntry {
            mode,
            name: name.to_string(),
            hash,
        });

        // Move to next entry
        content = &content[null_pos + 1 + 20..];
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

/// Parse object references from git object data
pub fn parse_object_references(object_data: &[u8]) -> Result<Vec<Sha256Hash>> {
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
        let tree = parse_tree_object(object_data)?;
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
                        "blob" => ObjectType::Blob,
                        "tag" => ObjectType::Tag,
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

/// Convert git2::Oid to SHA256 hash
/// Note: git2 currently only supports SHA1, so this will need updating when git2 supports SHA256
pub fn oid_to_sha256(oid: Oid) -> Result<Sha256Hash> {
    // When git2 supports SHA256, this should directly convert
    // For now, we assume all Oids will be replaced with proper SHA256 handling
    let oid_str = oid.to_string();
    if oid_str.len() == 64 {
        // Already a SHA256
        Sha256Hash::new(oid_str)
    } else {
        // SHA1 - return error as we no longer support SHA1
        Err(Error::other("SHA1 hashes are no longer supported, use SHA256"))
    }
}

/// Convert SHA256 hash back to git2::Oid
/// Note: This will need updating when git2 supports SHA256
pub fn sha256_to_oid(hash: &Sha256Hash) -> Result<Oid> {
    // When git2 supports SHA256, this should work directly
    // For now, this will likely fail as git2 expects SHA1
    Oid::from_str(hash.as_str()).map_err(|e| Error::other(format!("Cannot convert SHA256 to Oid (git2 limitation): {}", e)))
}

/// Repository publishing - extract all objects and store in DHT
pub async fn publish_repository(repo_path: &Path, dht: &crate::dht::GitTorrentDht) -> Result<Sha256Hash> {
    use crate::dht::{GitObjectKey, GitObjectRecord};

    tracing::info!("Publishing repository: {:?}", repo_path);

    // 1. Extract all git objects
    let objects = extract_objects(repo_path).await?;
    tracing::info!("Extracted {} objects from repository", objects.len());

    // 2. Store each object in DHT
    for obj in &objects {
        let key = GitObjectKey::new(obj.hash.clone());
        let record = GitObjectRecord::new(key.clone(), obj.data.clone());

        // Store object
        dht.put_object(record).await?;

        // Announce as provider
        dht.provide(key).await?;

        tracing::debug!("Published object {} ({})", obj.hash, obj.object_type.clone() as u8);
    }

    // 3. Get HEAD commit hash
    let head_commit = get_head_commit(repo_path).await?;

    tracing::info!("Repository published. HEAD commit: {}", head_commit);
    Ok(head_commit)
}

/// Get HEAD commit hash from repository
pub async fn get_head_commit(repo_path: &Path) -> Result<Sha256Hash> {
    let repo = Repository::open(repo_path)?;
    let head = repo.head()?;
    let commit = head.peel_to_commit()?;

    // Create git format and hash it
    let git_object = create_git_object(&repo, commit.as_object())?;
    Ok(git_object.hash)
}

/// Clone a repository from a commit hash using merkle tree traversal
pub async fn clone_commit(
    commit_hash: Sha256Hash,
    target_path: &Path,
    dht: &crate::dht::GitTorrentDht
) -> Result<()> {
    use crate::dht::GitObjectKey;

    tracing::info!("Cloning commit {} to {:?}", commit_hash, target_path);

    // 1. Fetch and parse the commit object
    let commit_key = GitObjectKey::new(commit_hash.clone());
    let commit_record = dht.get_object(commit_key).await?
        .ok_or_else(|| crate::Error::not_found("Commit not found"))?;

    let commit = parse_commit_object(&commit_record.data)?;
    tracing::info!("Found commit with tree: {}", commit.tree_hash);

    // 2. Create repository directory and initialize git
    tokio::fs::create_dir_all(target_path).await?;
    let repo = Repository::init(target_path)?;

    // 3. Recursively checkout the tree
    checkout_tree(commit.tree_hash.clone(), target_path, dht).await?;

    // 4. Write commit object and set HEAD
    write_git_object(&repo, &commit_record.data).await?;
    set_head_to_commit(&repo, commit_hash).await?;

    tracing::info!("Successfully cloned repository to {:?}", target_path);
    Ok(())
}

/// Recursively checkout a tree from the DHT
pub fn checkout_tree<'a>(
    tree_hash: Sha256Hash,
    path: &'a Path,
    dht: &'a crate::dht::GitTorrentDht
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + 'a>> {
    Box::pin(async move {
    use crate::dht::GitObjectKey;

    tracing::debug!("Checking out tree {} to {:?}", tree_hash, path);

    // Fetch tree object
    let tree_key = GitObjectKey::new(tree_hash.clone());
    let tree_record = dht.get_object(tree_key).await?
        .ok_or_else(|| crate::Error::not_found("Tree not found"))?;

    let tree = parse_tree_object(&tree_record.data)?;

    for entry in tree.entries {
        let entry_path = path.join(&entry.name);

        match entry.mode {
            TreeEntryMode::Directory => {
                // Create directory and recurse
                tokio::fs::create_dir_all(&entry_path).await?;
                checkout_tree(entry.hash, &entry_path, dht).await?;
            }
            TreeEntryMode::File | TreeEntryMode::Executable => {
                // Fetch blob and write file
                let blob_key = GitObjectKey::new(entry.hash.clone());
                let blob_record = dht.get_object(blob_key).await?
                    .ok_or_else(|| crate::Error::not_found("Blob not found"))?;

                let file_content = parse_blob_object(&blob_record.data)?;
                tokio::fs::write(&entry_path, file_content).await?;

                // Set executable permissions if needed
                if entry.mode == TreeEntryMode::Executable {
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        let mut perms = tokio::fs::metadata(&entry_path).await?.permissions();
                        perms.set_mode(0o755);
                        tokio::fs::set_permissions(&entry_path, perms).await?;
                    }
                }
            }
            TreeEntryMode::Symlink => {
                // Create symlink
                let blob_key = GitObjectKey::new(entry.hash);
                let blob_record = dht.get_object(blob_key).await?
                    .ok_or_else(|| crate::Error::not_found("Symlink target not found"))?;

                let link_target = parse_blob_object(&blob_record.data)?;
                let link_target_str = String::from_utf8(link_target)?;

                #[cfg(unix)]
                tokio::fs::symlink(link_target_str, entry_path).await?;

                #[cfg(windows)]
                {
                    // Windows symlink handling would go here
                    tracing::warn!("Symlinks not supported on Windows yet");
                }
            }
            TreeEntryMode::Gitlink => {
                // Submodules - create placeholder for now
                tracing::warn!("Gitlink (submodule) not supported yet: {}", entry.name);
                tokio::fs::write(entry_path, format!("# Submodule: {}", entry.name)).await?;
            }
        }
    }

    tracing::debug!("Successfully checked out tree {}", tree_hash);
    Ok(())
    })
}

/// Write a git object to the repository's object database
pub async fn write_git_object(repo: &Repository, object_data: &[u8]) -> Result<()> {
    // Parse the git format to get type and content
    if let Some(null_pos) = object_data.iter().position(|&b| b == 0) {
        let header = &object_data[..null_pos];
        let content = &object_data[null_pos + 1..];

        // Parse "type size" header
        if let Ok(header_str) = std::str::from_utf8(header) {
            let parts: Vec<&str> = header_str.split(' ').collect();
            if parts.len() == 2 {
                let obj_type = match parts[0] {
                    "commit" => ObjectType::Commit,
                    "tree" => ObjectType::Tree,
                    "blob" => ObjectType::Blob,
                    "tag" => ObjectType::Tag,
                    _ => ObjectType::Blob,
                };

                // Write object to git database
                let odb = repo.odb()?;
                let oid = odb.write(obj_type, content)?;
                tracing::debug!("Wrote git object: {}", oid);
                return Ok(());
            }
        }
    }

    Err(crate::Error::other("Invalid git object format"))
}

/// Set repository HEAD to a specific commit
pub async fn set_head_to_commit(repo: &Repository, commit_hash: Sha256Hash) -> Result<()> {
    // Convert SHA256 to OID (temporary until git2 supports SHA256)
    let oid = sha256_to_oid(&commit_hash)?;

    // Find the commit object
    let commit = repo.find_commit(oid)?;

    // Set HEAD to this commit
    repo.set_head_detached(commit.id())?;

    tracing::debug!("Set HEAD to commit: {}", commit_hash);
    Ok(())
}

/// Fetch commit history (breadth-first traversal)
pub async fn fetch_commit_history(
    commit_hash: Sha256Hash,
    dht: &crate::dht::GitTorrentDht,
    max_depth: Option<u32>
) -> Result<Vec<Sha256Hash>> {
    use crate::dht::GitObjectKey;

    let mut commits = Vec::new();
    let mut current = Some(commit_hash);
    let mut depth = 0;

    while let Some(hash) = current {
        if let Some(max) = max_depth {
            if depth >= max { break; }
        }

        commits.push(hash.clone());

        // Fetch commit and get parent
        let commit_key = GitObjectKey::new(hash);
        let commit_record = dht.get_object(commit_key).await?
            .ok_or_else(|| crate::Error::not_found("Commit not found"))?;

        let commit = parse_commit_object(&commit_record.data)?;
        current = commit.parent_hashes.first().cloned();
        depth += 1;
    }

    Ok(commits)
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
                // Convert git2 Oid to SHA256 (temporary until git2 supports SHA256)
                let sha256 = oid_to_sha256(target)?;
                refs_map.insert(name.to_string(), sha256);
            }
        }
    }

    // Get HEAD reference
    let head = repo.head()?;
    let head_name = head.name().unwrap_or("refs/heads/main").to_string();

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

/// Store git references in the DHT
pub async fn store_git_refs(
    commit_hash: &Sha256Hash,
    refs: &GitRefs,
    dht: &crate::dht::GitTorrentDht
) -> Result<()> {
    use crate::dht::{GitObjectKey, GitObjectRecord};

    // Serialize references
    let refs_data = serde_json::to_vec(refs)?;

    // Create a special key for references: "refs:" + commit_hash
    let refs_key_str = format!("refs:{}", commit_hash);
    let refs_key_hash = crate::crypto::hash::sha256_data(refs_key_str.as_bytes())?;
    let refs_key = GitObjectKey::new(refs_key_hash);

    // Store in DHT
    let record = GitObjectRecord::new(refs_key.clone(), refs_data);
    dht.put_object(record).await?;
    dht.provide(refs_key).await?;

    tracing::info!("Stored git references for commit: {}", commit_hash);
    Ok(())
}

/// Retrieve git references from the DHT
pub async fn retrieve_git_refs(
    commit_hash: &Sha256Hash,
    dht: &crate::dht::GitTorrentDht
) -> Result<Option<GitRefs>> {
    use crate::dht::GitObjectKey;

    // Create references key
    let refs_key_str = format!("refs:{}", commit_hash);
    let refs_key_hash = crate::crypto::hash::sha256_data(refs_key_str.as_bytes())?;
    let refs_key = GitObjectKey::new(refs_key_hash);

    // Retrieve from DHT
    if let Some(record) = dht.get_object(refs_key).await? {
        let refs: GitRefs = serde_json::from_slice(&record.data)?;
        Ok(Some(refs))
    } else {
        Ok(None)
    }
}

/// Enhanced publishing with optional references
pub async fn publish_repository_with_refs(
    repo_path: &Path,
    dht: &crate::dht::GitTorrentDht,
    include_refs: bool
) -> Result<Sha256Hash> {
    // Publish all objects
    let head_commit = publish_repository(repo_path, dht).await?;

    // Optionally publish references
    if include_refs {
        let refs = extract_git_refs(repo_path).await?;
        store_git_refs(&head_commit, &refs, dht).await?;
    }

    Ok(head_commit)
}

/// Enhanced cloning with optional references
pub async fn clone_commit_with_refs(
    commit_hash: Sha256Hash,
    target_path: &Path,
    dht: &crate::dht::GitTorrentDht,
    include_refs: bool
) -> Result<()> {
    // Clone the commit
    clone_commit(commit_hash.clone(), target_path, dht).await?;

    // Optionally restore references
    if include_refs {
        if let Some(refs) = retrieve_git_refs(&commit_hash, dht).await? {
            restore_git_refs(target_path, &refs).await?;
        }
    }

    Ok(())
}

/// Restore git references to a repository
pub async fn restore_git_refs(repo_path: &Path, refs: &GitRefs) -> Result<()> {
    let repo = Repository::open(repo_path)?;

    // Restore each reference
    for (ref_name, sha256_hash) in &refs.refs {
        // Convert SHA256 back to OID (temporary)
        let oid = sha256_to_oid(sha256_hash)?;

        // Create the reference
        if let Err(e) = repo.reference(ref_name, oid, true, "Restored from GitTorrent") {
            tracing::warn!("Failed to restore reference {}: {}", ref_name, e);
        }
    }

    // Set HEAD to the correct reference
    if let Err(e) = repo.set_head(&refs.head) {
        tracing::warn!("Failed to set HEAD to {}: {}", refs.head, e);
    }

    tracing::info!("Restored {} git references", refs.refs.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use git2::Repository;
    use tempfile::TempDir;
    use std::fs;

    #[tokio::test]
    async fn test_extract_objects() {
        let temp_dir = TempDir::new().unwrap();

        // Initialize repository and create a commit
        let repo = Repository::init(temp_dir.path()).unwrap();

        // Create a test file
        let file_path = temp_dir.path().join("test.txt");
        fs::write(&file_path, "Hello World").unwrap();

        // Add and commit
        let mut index = repo.index().unwrap();
        index.add_path(Path::new("test.txt")).unwrap();
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("Test", "test@example.com").unwrap();

        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            "Initial commit",
            &tree,
            &[],
        ).unwrap();

        // Extract objects
        let objects = extract_objects(temp_dir.path()).await.unwrap();

        // Should have at least: blob (file), tree, commit
        assert!(objects.len() >= 3);

        // Check that we have different object types
        let types: std::collections::HashSet<_> = objects.iter().map(|o| &o.object_type).collect();
        assert!(types.contains(&GitObjectType::Blob));
        assert!(types.contains(&GitObjectType::Tree));
        assert!(types.contains(&GitObjectType::Commit));
    }

    #[test]
    fn test_is_exportable() {
        let temp_dir = TempDir::new().unwrap();

        // Initialize a new repository
        let repo = Repository::init(temp_dir.path()).unwrap();
        drop(repo);

        // Should not be exportable initially
        assert!(!is_exportable(temp_dir.path()));

        // Create the export file
        let export_file = temp_dir.path().join(".git/git-daemon-export-ok");
        fs::write(export_file, "").unwrap();

        // Should now be exportable
        assert!(is_exportable(temp_dir.path()));
    }
}