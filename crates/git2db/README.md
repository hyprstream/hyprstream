# git2db - Git-native Repository Management

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

git2db provides a high-level Rust interface for managing Git repositories as a database system, with advanced features like storage drivers for space-efficient worktrees and large file (XET) integration.

---

## Features

- **🗄️ Repository Registry**: Track and manage multiple Git repositories with UUID-based identification
- **⚡ Global Repository Cache**: Thread-safe singleton for efficient repository access
- **🌿 Storage Drivers**: Space-efficient worktrees using Docker's graphdriver pattern
  - **overlay2** (Linux): ~80% disk space savings via overlayfs
  - **vfs** (fallback): Standard git worktrees, cross-platform
- **🔧 High-level Git APIs**: Branch, staging, and remote operations with clean Rust interfaces
- **📦 XET Integration**: Large file storage with lazy loading and deduplication
- **🔐 Thread-safe**: Built on Arc, RwLock, and DashMap for concurrent access

---

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
git2db = { path = "path/to/git2db" }

# For overlayfs support on Linux
git2db = { path = "path/to/git2db", features = ["overlayfs"] }

# For XET large file support
git2db = { path = "path/to/git2db", features = ["xet-storage"] }
```

### Basic Usage

```rust
use git2db::{Git2DB, GitManager, RepoId};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Open or create a registry
    let registry = Git2DB::open("/path/to/registry").await?;

    // Clone a repository
    let repo_id = registry.clone("https://github.com/user/repo.git")
        .name("my-repo")
        .shallow(true)
        .exec().await?;

    // Get a repository handle
    let handle = registry.repo(&repo_id)?;

    // Create a branch
    handle.branch().create("feature", None).await?;

    // Stage files
    handle.staging().add_all()?;

    // Fetch from remote
    handle.remote().fetch("origin").await?;

    Ok(())
}
```

---

## Core Concepts

### 1. GitManager - Global Repository Cache

```rust
use git2db::GitManager;

// Access the global singleton
let manager = GitManager::global();

// Get or cache a repository
let repo_cache = manager.get_repository("/path/to/repo")?;
let repo = repo_cache.open()?;

// Create worktrees with automatic storage driver selection
let worktree_path = manager.create_worktree(
    "/path/to/repo",
    "/path/to/worktree",
    "branch-name"
).await?;
```

**Benefits**:
- Thread-safe repository caching
- Automatic cleanup on shutdown
- Configuration-driven storage driver selection

---

### 2. Git2DB - Repository Registry

```rust
use git2db::{Git2DB, RepoId};

// Open a registry
let registry = Git2DB::open("/models").await?;

// Register a repository
let repo_id = RepoId::new();
registry.register_repository(
    &repo_id,
    Some("my-model".to_string()),
    "https://github.com/user/model.git".to_string()
)?;

// Lookup by name or URL
let repo_id = registry.resolve_name("my-model")?;
let handle = registry.repo(&repo_id)?;
```

**Features**:
- UUID-based repository identification
- Persistent metadata (`.git2db/metadata.json`)
- Name and URL-based lookups

---

### 3. RepositoryHandle - Scoped Git Operations

```rust
let handle = registry.repo(&repo_id)?;

// Branch operations
handle.branch().create("feature", None).await?;
handle.branch().checkout("feature").await?;
let branches = handle.branch().list().await?;

// Staging operations
handle.staging().add_all()?;
handle.staging().add_file("specific_file.txt")?;

// Remote operations
handle.remote().fetch("origin").await?;
handle.remote().push("origin", "main").await?;
```

**Escape Hatch** (for operations not yet in RepositoryHandle):
```rust
let repo = handle.open()?;  // Get raw git2::Repository
// Use git2 directly...
```

---

## Storage Drivers

git2db uses Docker's storage driver architecture for space-efficient worktrees:

### overlay2 (Linux, Production Default)

**How it works**:
- Uses Linux overlayfs to create copy-on-write worktrees
- Base repository is read-only lower layer
- Changes written to writable upper layer
- ~80% disk space savings

**Backends**:
- `kernel`: Direct kernel overlayfs (requires CAP_SYS_ADMIN or root)
- `userns`: User namespace overlayfs (no privileges required)
- `fuse`: FUSE-based overlayfs (fallback)
- `auto`: Automatic selection

**Configuration**:
```toml
[worktree]
driver = "overlay2"

[worktree.overlay2]
backend = "auto"  # or: kernel, userns, fuse
```

**Environment**:
```bash
export GIT2DB_WORKTREE_DRIVER=overlay2
export GIT2DB_OVERLAY_BACKEND=userns
```

---

### vfs (Cross-platform Fallback)

**How it works**:
- Standard git worktrees
- No space optimization
- Works on all platforms

**When to use**:
- Non-Linux platforms
- When overlayfs unavailable
- Testing/debugging
- CI without privileges

**Configuration**:
```toml
[worktree]
driver = "vfs"
```

---

## Configuration

git2db can be configured via:
1. Configuration file: `~/.config/git2db/config.toml`
2. Environment variables (prefix: `GIT2DB_`)
3. Programmatic API

### Example Configuration

```toml
[worktree]
# Storage driver selection
driver = "auto"  # auto-select: overlay2 on Linux, vfs elsewhere

[worktree.overlay2]
# Overlayfs backend
backend = "auto"  # try kernel → userns → fuse

[git]
# Default git signature
name = "Your Name"
email = "your.email@example.com"
```

### Environment Variables

```bash
# Worktree driver
export GIT2DB_WORKTREE_DRIVER=overlay2

# Overlayfs backend
export GIT2DB_OVERLAY_BACKEND=userns

# Git signature
export GIT2DB_GIT_NAME="CI Bot"
export GIT2DB_GIT_EMAIL="ci@example.com"
```

---

## Examples

### Clone and Branch Workflow

```rust
use git2db::Git2DB;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry = Git2DB::open("/workspace").await?;

    // Clone a repository
    let repo_id = registry.clone("https://github.com/rust-lang/rust.git")
        .name("rust")
        .shallow(true)
        .exec().await?;

    let handle = registry.repo(&repo_id)?;

    // Create feature branch
    handle.branch().create("my-feature", Some("main")).await?;
    handle.branch().checkout("my-feature").await?;

    // Make changes and stage
    handle.staging().add_all()?;

    // Commit (using escape hatch until commit() API implemented)
    let repo = handle.open()?;
    let sig = git2db::GitManager::global().create_signature(None, None)?;
    let tree = repo.find_tree(repo.index()?.write_tree()?)?;
    let parent = repo.head()?.peel_to_commit()?;
    repo.commit(
        Some("HEAD"),
        &sig,
        &sig,
        "My changes",
        &tree,
        &[&parent],
    )?;

    // Push to remote
    handle.remote().push("origin", "my-feature").await?;

    Ok(())
}
```

---

### Space-Efficient Worktrees

```rust
use git2db::GitManager;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manager = GitManager::global();

    // Create worktree (auto-selects storage driver)
    let worktree_path = manager.create_worktree(
        "/path/to/repo",
        "/tmp/worktree",
        "feature-branch"
    ).await?;

    // On Linux with overlayfs: ~80% disk space savings
    // On other platforms: standard git worktree

    println!("Worktree created at: {:?}", worktree_path);

    Ok(())
}
```

---

### Multiple Repository Management

```rust
use git2db::Git2DB;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry = Git2DB::open("/models").await?;

    // Clone multiple repositories
    for url in &[
        "https://huggingface.co/model1",
        "https://huggingface.co/model2",
        "https://huggingface.co/model3",
    ] {
        let name = url.split('/').last().unwrap();
        registry.clone(url)
            .name(name)
            .shallow(true)
            .exec().await?;
    }

    // List all repositories
    for repo in registry.list_repositories()? {
        println!("Repo: {} ({})", repo.name, repo.url);
    }

    Ok(())
}
```

---

## Building

### Basic Build

```bash
cargo build -p git2db
cargo test -p git2db
```

### With Overlayfs Support (Linux)

```bash
cargo build -p git2db --features overlayfs
cargo test -p git2db --features overlayfs
```

### With XET Large File Support

```bash
cargo build -p git2db --features xet-storage
```

### Full Feature Set

```bash
cargo build -p git2db --features overlayfs,xet-storage
```

---

## Platform Support

| Platform | overlay2 | vfs |
|----------|----------|-----|
| Linux    | ✅       | ✅  |
| macOS    | ❌       | ✅  |
| Windows  | ❌ (✅ WSL2) | ✅ |
| BSD      | ❌       | ✅  |

**Note**: overlay2 requires Linux kernel with overlayfs support. On other platforms, git2db automatically falls back to vfs driver.

---

## API Status

### Completed ✅
- `GitManager` - Global repository cache
- `Git2DB` - Repository registry
- `RepositoryHandle` - Scoped repository access
- `branch()` - Branch management
- `staging()` - Stage operations
- `remote()` - Remote operations
- Storage driver system (overlay2, vfs)
- Configuration system

### In Progress / Planned ⏳
- `commit()` API - High-level commit creation
- `merge()` API - Merge operations
- `rebase()` API - Rebase operations
- `tag()` API - Tag management
- Additional storage drivers (reflink, hardlink)

---

## Integration

git2db is designed to be integrated into larger applications. See [hyprstream](../../README.md) for a real-world example of git2db integration in an LLM inference engine.

**Key integration points**:
- Repository lifecycle management
- Space-efficient worktrees for isolated operations
- Thread-safe concurrent repository access
- Configuration-driven behavior

---

## Contributing

Contributions welcome! Please see the main project [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Additional Resources

- **[CLAUDE.md](CLAUDE.md)** - AI assistant guide with architecture details
- **[docs/COW_ARCHITECTURE.md](../../docs/COW_ARCHITECTURE.md)** - Worktree CoW mechanisms
- **[docs/GIT2DB-*.md](../../docs/)** - Design documents and migration guides

---

**Key Takeaway**: git2db treats Git repositories as first-class data structures, providing space-efficient storage and clean Rust APIs for repository management.
