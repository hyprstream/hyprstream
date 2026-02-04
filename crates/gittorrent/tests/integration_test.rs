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
    fn test_url_parsing_commit_format() -> Result<(), Box<dyn std::error::Error>> {
        // Test the new commit-based URL format
        let test_hash = "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456";

        // Test simple commit URL
        let url = GitTorrentUrl::parse(&format!("gittorrent://{test_hash}"))?;
        match url {
            GitTorrentUrl::Commit { ref hash } => {
                assert_eq!(hash.as_str(), test_hash);
                assert_eq!(url.to_string(), format!("gittorrent://{test_hash}"));
            }
            _ => panic!("Expected Commit variant"),
        }

        // Test commit with refs URL
        let url_with_refs = GitTorrentUrl::parse(&format!("gittorrent://{test_hash}?refs"))?;
        match url_with_refs {
            GitTorrentUrl::CommitWithRefs { ref hash } => {
                assert_eq!(hash.as_str(), test_hash);
                assert!(url_with_refs.includes_refs());
                assert_eq!(url_with_refs.to_string(), format!("gittorrent://{test_hash}?refs"));
            }
            _ => panic!("Expected CommitWithRefs variant"),
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_git_object_extraction() -> Result<(), Box<dyn std::error::Error>> {
        // Create a test repository with sample content
        let temp_dir = TempDir::new()?;
        let repo = Repository::init(temp_dir.path())?;

        // Create a sample file and commit
        let file_path = temp_dir.path().join("test.txt");
        std::fs::write(&file_path, "Hello GitTorrent!")?;

        let mut index = repo.index()?;
        index.add_path(std::path::Path::new("test.txt"))?;
        index.write()?;

        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let sig = git2::Signature::now("Test", "test@example.com")?;

        let _commit_id = repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            "Initial commit",
            &tree,
            &[],
        )?;

        // Test object extraction
        let objects = extract_objects(temp_dir.path()).await?;

        // Should have: blob (file), tree, commit
        assert!(objects.len() >= 3);

        let types: std::collections::HashSet<_> = objects.iter().map(|o| &o.object_type).collect();
        assert!(types.contains(&GitObjectType::Blob));
        assert!(types.contains(&GitObjectType::Tree));
        assert!(types.contains(&GitObjectType::Commit));
        Ok(())
    }

    #[tokio::test]
    async fn test_git_object_parsing() -> Result<(), Box<dyn std::error::Error>> {
        // Create a test repository
        let temp_dir = TempDir::new()?;
        let repo = Repository::init(temp_dir.path())?;

        // Create and commit a file
        let file_path = temp_dir.path().join("sample.txt");
        std::fs::write(&file_path, "Sample content for parsing test")?;

        let mut index = repo.index()?;
        index.add_path(std::path::Path::new("sample.txt"))?;
        index.write()?;

        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let sig = git2::Signature::now("Test Parser", "parser@example.com")?;

        let _commit_id = repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            "Test parsing commit",
            &tree,
            &[],
        )?;

        // Extract objects
        let objects = extract_objects(temp_dir.path()).await?;

        // Find and parse the commit object
        let commit_obj = objects.iter()
            .find(|obj| obj.object_type == GitObjectType::Commit)
            .ok_or("No commit object found")?;
        let parsed_commit = parse_commit_object(&commit_obj.data)?;

        assert!(parsed_commit.message.contains("Test parsing commit"));
        assert!(parsed_commit.author.contains("Test Parser"));
        assert!(!parsed_commit.tree_hash.to_hex().is_empty());

        // Find and parse the tree object
        let tree_obj = objects.iter()
            .find(|obj| obj.object_type == GitObjectType::Tree)
            .ok_or("No tree object found")?;
        let parsed_tree = parse_tree_object(&tree_obj.data)?;

        assert_eq!(parsed_tree.entries.len(), 1);
        assert_eq!(parsed_tree.entries[0].name, "sample.txt");
        assert_eq!(parsed_tree.entries[0].mode, TreeEntryMode::File);

        // Find and parse the blob object
        let blob_obj = objects.iter()
            .find(|obj| obj.object_type == GitObjectType::Blob)
            .ok_or("No blob object found")?;
        let parsed_blob = parse_blob_object(&blob_obj.data)?;

        assert_eq!(parsed_blob, b"Sample content for parsing test");
        Ok(())
    }

    #[tokio::test]
    async fn test_git_references_extraction() -> Result<(), Box<dyn std::error::Error>> {
        // Create a test repository with multiple branches
        let temp_dir = TempDir::new()?;
        let repo = Repository::init(temp_dir.path())?;

        // Create initial commit
        let file_path = temp_dir.path().join("README.md");
        std::fs::write(&file_path, "# Test Repository")?;

        let mut index = repo.index()?;
        index.add_path(std::path::Path::new("README.md"))?;
        index.write()?;

        let tree_id = index.write_tree()?;
        let tree = repo.find_tree(tree_id)?;
        let sig = git2::Signature::now("Test", "test@example.com")?;

        let commit_id = repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            "Initial commit",
            &tree,
            &[],
        )?;

        // Create a branch
        let commit = repo.find_commit(commit_id)?;
        repo.branch("feature-branch", &commit, false)?;

        // Extract git references
        let refs = extract_git_refs(temp_dir.path()).await?;

        assert!(!refs.refs.is_empty());
        assert!(refs.refs.contains_key("refs/heads/main") || refs.refs.contains_key("refs/heads/master"));
        assert!(refs.refs.contains_key("refs/heads/feature-branch"));
        assert!(!refs.head.is_empty());
        assert!(refs.created_at > 0);
        Ok(())
    }

    #[test]
    fn test_protocol_design_consistency() -> Result<(), Box<dyn std::error::Error>> {
        // Verify that our implementation matches the protocol design from GIT-PLAN.md

        // 1. URL format should be gittorrent://COMMIT_SHA256
        let commit_hash = "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456";
        let url = format!("gittorrent://{commit_hash}");
        let parsed = GitTorrentUrl::parse(&url)?;

        match parsed {
            GitTorrentUrl::Commit { ref hash } => {
                assert_eq!(hash.as_str(), commit_hash);
            }
            _ => panic!("Should parse as Commit variant"),
        }

        // 2. Optional refs parameter should work
        let url_with_refs = format!("gittorrent://{commit_hash}?refs");
        let parsed_refs = GitTorrentUrl::parse(&url_with_refs)?;

        assert!(parsed_refs.includes_refs());

        // 3. Both should have access to the commit hash
        let commit_hash_1 = parsed.commit_hash().ok_or("No commit hash")?;
        assert_eq!(commit_hash_1.as_str(), commit_hash);
        let commit_hash_2 = parsed_refs.commit_hash().ok_or("No commit hash")?;
        assert_eq!(commit_hash_2.as_str(), commit_hash);
        Ok(())
    }

    // Note: Full end-to-end DHT tests would require setting up a test DHT network,
    // which is complex for a unit test environment. The individual components are
    // tested above, and the integration would happen in the actual DHT environment.
}
