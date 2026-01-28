//! Integration tests for GitTorrent merkle tree protocol
//!
//! These tests verify the end-to-end workflow of the new commit-based
//! protocol described in GIT-PLAN.md

#[cfg(test)]
mod tests {
    use gittorrent::{
        types::GitTorrentUrl,
        git::objects::*,
    };
    
    use tempfile::TempDir;
    use git2::Repository;

    #[test]
    fn test_url_parsing_commit_format() {
        // Test the new commit-based URL format
        let test_hash = "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456";

        // Test simple commit URL
        let url = GitTorrentUrl::parse(&format!("gittorrent://{test_hash}")).unwrap();
        match url {
            GitTorrentUrl::Commit { ref hash } => {
                assert_eq!(hash.as_str(), test_hash);
                assert_eq!(url.to_string(), format!("gittorrent://{test_hash}"));
            }
            _ => panic!("Expected Commit variant"),
        }

        // Test commit with refs URL
        let url_with_refs = GitTorrentUrl::parse(&format!("gittorrent://{test_hash}?refs")).unwrap();
        match url_with_refs {
            GitTorrentUrl::CommitWithRefs { ref hash } => {
                assert_eq!(hash.as_str(), test_hash);
                assert!(url_with_refs.includes_refs());
                assert_eq!(url_with_refs.to_string(), format!("gittorrent://{test_hash}?refs"));
            }
            _ => panic!("Expected CommitWithRefs variant"),
        }
    }

    #[tokio::test]
    async fn test_git_object_extraction() {
        // Create a test repository with sample content
        let temp_dir = TempDir::new().unwrap();
        let repo = Repository::init(temp_dir.path()).unwrap();

        // Create a sample file and commit
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "Hello GitTorrent!").unwrap();

        let mut index = repo.index().unwrap();
        index.add_path(std::path::Path::new("test.txt")).unwrap();
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("Test", "test@example.com").unwrap();

        let _commit_id = repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            "Initial commit",
            &tree,
            &[],
        ).unwrap();

        // Test object extraction
        let objects = extract_objects(temp_dir.path()).await.unwrap();

        // Should have: blob (file), tree, commit
        assert!(objects.len() >= 3);

        let types: std::collections::HashSet<_> = objects.iter().map(|o| &o.object_type).collect();
        assert!(types.contains(&GitObjectType::Blob));
        assert!(types.contains(&GitObjectType::Tree));
        assert!(types.contains(&GitObjectType::Commit));
    }

    #[tokio::test]
    async fn test_git_object_parsing() {
        // Create a test repository
        let temp_dir = TempDir::new().unwrap();
        let repo = Repository::init(temp_dir.path()).unwrap();

        // Create and commit a file
        let file_path = temp_dir.path().join("sample.txt");
        std::fs::write(&file_path, "Sample content for parsing test").unwrap();

        let mut index = repo.index().unwrap();
        index.add_path(std::path::Path::new("sample.txt")).unwrap();
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("Test Parser", "parser@example.com").unwrap();

        let _commit_id = repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            "Test parsing commit",
            &tree,
            &[],
        ).unwrap();

        // Extract objects
        let objects = extract_objects(temp_dir.path()).await.unwrap();

        // Find and parse the commit object
        let commit_obj = objects.iter().find(|obj| obj.object_type == GitObjectType::Commit).unwrap();
        let parsed_commit = parse_commit_object(&commit_obj.data).unwrap();

        assert!(parsed_commit.message.contains("Test parsing commit"));
        assert!(parsed_commit.author.contains("Test Parser"));
        assert!(!parsed_commit.tree_hash.to_hex().is_empty());

        // Find and parse the tree object
        let tree_obj = objects.iter().find(|obj| obj.object_type == GitObjectType::Tree).unwrap();
        let parsed_tree = parse_tree_object(&tree_obj.data).unwrap();

        assert_eq!(parsed_tree.entries.len(), 1);
        assert_eq!(parsed_tree.entries[0].name, "sample.txt");
        assert_eq!(parsed_tree.entries[0].mode, TreeEntryMode::File);

        // Find and parse the blob object
        let blob_obj = objects.iter().find(|obj| obj.object_type == GitObjectType::Blob).unwrap();
        let parsed_blob = parse_blob_object(&blob_obj.data).unwrap();

        assert_eq!(parsed_blob, b"Sample content for parsing test");
    }

    #[tokio::test]
    async fn test_git_references_extraction() {
        // Create a test repository with multiple branches
        let temp_dir = TempDir::new().unwrap();
        let repo = Repository::init(temp_dir.path()).unwrap();

        // Create initial commit
        let file_path = temp_dir.path().join("README.md");
        std::fs::write(&file_path, "# Test Repository").unwrap();

        let mut index = repo.index().unwrap();
        index.add_path(std::path::Path::new("README.md")).unwrap();
        index.write().unwrap();

        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("Test", "test@example.com").unwrap();

        let commit_id = repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            "Initial commit",
            &tree,
            &[],
        ).unwrap();

        // Create a branch
        let commit = repo.find_commit(commit_id).unwrap();
        repo.branch("feature-branch", &commit, false).unwrap();

        // Extract git references
        let refs = extract_git_refs(temp_dir.path()).await.unwrap();

        assert!(!refs.refs.is_empty());
        assert!(refs.refs.contains_key("refs/heads/main") || refs.refs.contains_key("refs/heads/master"));
        assert!(refs.refs.contains_key("refs/heads/feature-branch"));
        assert!(!refs.head.is_empty());
        assert!(refs.created_at > 0);
    }

    #[test]
    fn test_protocol_design_consistency() {
        // Verify that our implementation matches the protocol design from GIT-PLAN.md

        // 1. URL format should be gittorrent://COMMIT_SHA256
        let commit_hash = "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456";
        let url = format!("gittorrent://{commit_hash}");
        let parsed = GitTorrentUrl::parse(&url).unwrap();

        match parsed {
            GitTorrentUrl::Commit { ref hash } => {
                assert_eq!(hash.as_str(), commit_hash);
            }
            _ => panic!("Should parse as Commit variant"),
        }

        // 2. Optional refs parameter should work
        let url_with_refs = format!("gittorrent://{commit_hash}?refs");
        let parsed_refs = GitTorrentUrl::parse(&url_with_refs).unwrap();

        assert!(parsed_refs.includes_refs());

        // 3. Both should have access to the commit hash
        assert_eq!(parsed.commit_hash().unwrap().as_str(), commit_hash);
        assert_eq!(parsed_refs.commit_hash().unwrap().as_str(), commit_hash);
    }

    // Note: Full end-to-end DHT tests would require setting up a test DHT network,
    // which is complex for a unit test environment. The individual components are
    // tested above, and the integration would happen in the actual DHT environment.
}