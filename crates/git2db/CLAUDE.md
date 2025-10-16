# git2db - Git-native Repository Management

**Claude AI Assistant Guide** - Updated Oct 2025

## Purpose

git2db provides a high-level Rust interface for managing Git repositories as a database system, with advanced features like storage drivers, worktree management, and large file (XET) integration.

**Design Philosophy**: Git repositories are first-class data structures, not just version control tools.

---

## Architecture Overview

### Core Components

```
GitManager (Global Singleton)
    ↓
Git2DB (Registry)
    ↓
RepositoryHandle (Scoped Operations)
    ↓
Storage Drivers (overlay2, vfs, etc.)
```

#### **1. GitManager** - Global Repository Cache
**Location**: `src/manager.rs`

**Purpose**: Thread-safe singleton for repository lifecycle management

```rust
// Access the global instance
let manager = GitManager::global();

// Get or cache a repository
let repo_cache = manager.get_repository("/path/to/repo")?;
let repo = repo_cache.open()?;

// Create worktrees with storage drivers
let worktree_path = manager.create_worktree(
    "/path/to/repo",
    "/path/to/worktree",
    "feature-branch"
).await?;
```

**Key Features**:
- Thread-safe repository caching (Arc<RwLock<DashMap>>)
- Automatic cleanup on shutdown
- Configuration-driven storage driver selection
- Signature creation with git config fallback

---

#### **2. Git2DB (Registry)** - Repository Tracking
**Location**: `src/registry.rs`

**Purpose**: Track and manage multiple repositories with metadata

```rust
// Open or create a registry
let registry = Git2DB::open("/models").await?;

// Register a repository
let repo_id = registry.register_repository(
    &RepoId::new(),
    Some("my-model".to_string()),
    "https://github.com/user/model.git".to_string()
)?;

// Get a repository handle
let handle = registry.repo(&repo_id)?;
```

**Metadata Storage**: `.git2db/metadata.json` in registry directory

**Key Features**:
- UUID-based repository identification
- Name-based and URL-based lookups
- Persistent metadata across restarts
- Repository lifecycle tracking

---

#### **3. RepositoryHandle** - Scoped Operations
**Location**: `src/repository_handle.rs`

**Purpose**: Provides high-level git operations with lifetime scoping

```rust
let handle = registry.repo(&repo_id)?;

// Branch operations
handle.branch().create("feature", None).await?;
handle.branch().checkout("feature").await?;
handle.branch().list().await?;

// Staging operations
handle.staging().add_all()?;
handle.staging().add_file("specific_file.txt")?;

// Remote operations
handle.remote().fetch("origin").await?;
handle.remote().push("origin", "main").await?;

// Note: commit() API not yet implemented (use repo.open() escape hatch)
```

**Phase 3 Status** (High-level APIs):
- ✅ `branch()` - Branch management (complete)
- ✅ `staging()` - Stage operations (complete)
- ✅ `remote()` - Remote operations (complete)
- ❌ `commit()` - Commit creation (not yet implemented)
- ❌ `merge()` - Merge operations (not yet implemented)

**Escape Hatch** (when high-level API missing):
```rust
let repo = handle.open()?;  // Get raw git2::Repository
// Use git2 directly for operations not yet in RepositoryHandle
```

---

### Storage Drivers (Docker graphdriver Pattern)

**Location**: `src/storage/`

git2db uses Docker's storage driver architecture for space-efficient worktrees:

```
Storage Driver Registry
    ↓
Driver Selection (Auto/Manual)
    ↓
Backend Implementation
    ├── overlay2 (Linux overlayfs - 80% space savings) ✅
    ├── vfs (Plain git worktrees - fallback) ✅
    └── Future: reflink, hardlink, etc.
```

#### **overlay2 Driver** (Production Default on Linux)
**Location**: `src/storage/overlay2.rs`

**How it Works**:
```
Base Repository (read-only lower layer)
    ↓
Overlayfs Mount
    ├── lowerdir: /base/repo/.git/objects
    ├── upperdir: /worktree/.overlay/upper
    ├── workdir: /worktree/.overlay/work
    └── merged: /worktree (user sees this)
```

**Benefits**:
- ~80% disk space savings
- Copy-on-write semantics
- Native filesystem performance
- Transparent to applications

**Requirements**:
- Linux kernel with overlayfs support
- May require user namespaces or CAP_SYS_ADMIN

#### **vfs Driver** (Fallback)
**Location**: `src/storage/vfs.rs`

**How it Works**: Standard git worktree (no optimization)

**Use Cases**:
- Non-Linux platforms
- When overlayfs unavailable
- Testing/debugging
- CI environments without privileges

---

## Configuration

**Global Config** (`~/.config/git2db/config.toml` or env vars):

```toml
[worktree]
# Storage driver selection
driver = "auto"  # auto, overlay2, vfs

[worktree.overlay2]
# Overlayfs backend selection
backend = "auto"  # auto, fuse, kernel, userns

[git]
# Default signature
name = "Your Name"
email = "your.email@example.com"
```

**Environment Variables**:
```bash
GIT2DB_WORKTREE_DRIVER=overlay2
GIT2DB_OVERLAY_BACKEND=kernel
GIT2DB_GIT_NAME="CI Bot"
GIT2DB_GIT_EMAIL="ci@example.com"
```

---

## Integration with hyprstream

hyprstream uses git2db for ALL git operations:

### **Model Storage**
```rust
// hyprstream's ModelRegistry wraps Git2DB
let registry = Git2DB::open(models_dir).await?;

// Models are tracked repositories
let repo_id = registry.clone("https://huggingface.co/Qwen/Qwen3-0.6B")
    .name("qwen3-small")
    .exec().await?;
```

### **Repository Operations**
```rust
// All operations via git2db
let handle = registry.repo(&repo_id)?;
handle.branch().create("experiment", None).await?;
handle.staging().add_all()?;

// Commit via escape hatch (until commit() API implemented)
let repo = handle.open()?;
let sig = GitManager::global().create_signature(None, None)?;
// ... manual commit creation ...
```

### **Worktree Creation**
```rust
// Automatic storage driver selection
let worktree_path = GitManager::global()
    .create_worktree(model_path, worktree_path, "branch-name")
    .await?;

// Uses overlay2 on Linux, vfs elsewhere
```

### **Tag Creation**
```rust
// hyprstream adds simple helpers for common operations
use hyprstream::git::helpers::create_tag;

create_tag(model_path, "checkpoint-v1")?;
// Internally uses GitManager for repository caching
```

---

## Build Instructions

```bash
# Basic build (no storage drivers)
cargo build -p git2db

# With overlayfs support (Linux only)
cargo build -p git2db --features overlayfs

# With XET large file support
cargo build -p git2db --features xet-storage

# Full feature set
cargo build -p git2db --features overlayfs,xet-storage

# Run tests
cargo test -p git2db
```

---

## Key Design Patterns

### **1. Global GitManager Pattern**
```rust
// DON'T: Create manager per operation
let manager = GitManager::new(config)?;  // ❌

// DO: Use global singleton
let manager = GitManager::global();  // ✅
```

**Rationale**: Repository caching only works with shared state

---

### **2. Scoped Repository Access**
```rust
// DON'T: Hold Repository across await points
let repo = Repository::open(path)?;
some_async_operation().await;  // ❌ repo held too long

// DO: Use RepositoryHandle for scoped access
let handle = registry.repo(&repo_id)?;
handle.branch().create("new", None).await?;  // ✅ Safe
```

**Rationale**: Prevents lock contention and ensures cleanup

---

### **3. Storage Driver Abstraction**
```rust
// Application code doesn't care about driver
let wt = GitManager::global()
    .create_worktree(repo, wt_path, branch)
    .await?;

// Driver selected automatically:
// - Linux + overlayfs available → overlay2
// - Linux + no overlayfs → vfs
// - Other platforms → vfs
```

**Rationale**: Platform-specific optimization without code changes

---

### **4. UUID-based Repository Identification**
```rust
// DON'T: Use paths as IDs
let id = repo_path.to_string();  // ❌ Path changes break references

// DO: Use UUID-based RepoId
let repo_id = RepoId::new();  // ✅ Stable across moves
```

**Rationale**: Repositories can move, UUIDs can't

---

## Common Patterns

### **Clone and Register**
```rust
let registry = Git2DB::open("/models").await?;
let repo_id = registry.clone("https://github.com/user/repo.git")
    .name("my-repo")
    .shallow(true)
    .exec().await?;
```

### **Branch Workflow**
```rust
let handle = registry.repo(&repo_id)?;

// Create and checkout
handle.branch().create("feature", None).await?;
handle.branch().checkout("feature").await?;

// Make changes, stage, commit
handle.staging().add_all()?;

// Commit via escape hatch (temporary)
let repo = handle.open()?;
let sig = GitManager::global().create_signature(None, None)?;
let tree = repo.find_tree(repo.index()?.write_tree()?)?;
let parent = repo.head()?.peel_to_commit()?;
repo.commit(Some("HEAD"), &sig, &sig, "Message", &tree, &[&parent])?;
```

### **Worktree Creation**
```rust
// Simple worktree
let wt_path = GitManager::global()
    .create_worktree(repo_path, "/tmp/wt", "branch")
    .await?;

// Worktree with specific driver
// (Set GIT2DB_WORKTREE_DRIVER env var)
```

---

## Phase 3 Completion Status

**Completed** ✅:
- `RepositoryHandle` abstraction
- `branch()` API - Full branch management
- `staging()` API - Stage operations
- `remote()` API - Remote operations
- Storage driver system
- overlay2 driver (3 backends: kernel, userns, fuse)
- Configuration system

**Not Yet Implemented** ❌:
- `commit()` API - Still uses escape hatch
- `merge()` API - Complex git operation
- `rebase()` API - Complex git operation
- Tag management API - Use git2 directly

**Workaround**: Use `handle.open()` escape hatch for missing APIs

---

## Testing

```bash
# Unit tests
cargo test -p git2db

# Integration tests
cargo test -p git2db --test '*'

# Overlayfs tests (Linux only)
cargo test -p git2db --features overlayfs storage::overlay2

# With logging
RUST_LOG=git2db=debug cargo test -p git2db
```

---

## Debugging

### **Enable Detailed Tracing**
```bash
export RUST_LOG=git2db=trace
cargo run --features tracing-detail
```

### **Check Storage Driver Selection**
```rust
// Log which driver was selected
tracing::info!("Using storage driver: {}", driver_name);
```

### **Inspect Overlayfs Mounts**
```bash
# See active overlayfs mounts
mount | grep overlay
findmnt -t overlay
```

---

## Migration Notes

### **From Direct git2 Usage**
```rust
// OLD: Direct git2
let repo = Repository::open(path)?;
repo.branch("new", &commit, false)?;

// NEW: git2db
let handle = registry.repo(&repo_id)?;
handle.branch().create("new", None).await?;
```

### **From Custom Git Wrappers**
If you have custom git wrappers (like hyprstream's old BranchManager):
- ❌ Remove custom wrappers
- ✅ Use git2db APIs directly
- ✅ Add helpers in your crate for domain-specific operations

---

## Common Issues

### **"Repository not registered"**
```rust
// Ensure repository is registered first
registry.register_repository(&repo_id, name, url)?;
```

### **"Overlayfs mount failed"**
- Check kernel support: `grep overlay /proc/filesystems`
- Try different backend: `GIT2DB_OVERLAY_BACKEND=userns`
- Fallback to vfs: `GIT2DB_WORKTREE_DRIVER=vfs`

### **"Permission denied" on worktree creation**
- overlayfs may need privileges
- Use user namespaces backend: `GIT2DB_OVERLAY_BACKEND=userns`
- Or fallback to vfs driver

---

This guide should help you understand git2db's architecture and integration patterns. For specific API details, see the inline documentation in the source code.
