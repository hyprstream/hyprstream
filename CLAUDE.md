# hyprstream-torch Development Guide

**Claude AI Assistant Guide** - Updated Oct 2025 after adapter architecture simplification

---

## 🚧 Current Status (Oct 2025)

### XET Feature Status

**XET Integration**: ⚠️ **EXPERIMENTAL** (optional feature, disabled by default)
- Feature flag implemented and working correctly
- Optional in git2db, disabled by default in hyprstream
- LFS pointer detection implemented (with graceful degradation when XET disabled)
- Architecture designed for automatic smudging via git-xet-filter
- **Known Issue**: Filter implementation needs refactoring (see below)

### Recent Refactoring ✅

**Architecture Simplification**:
- ✅ Removed duplicate XetNativeStorage (863 lines deleted)
- ✅ Consolidated to lfs_xet.rs with git2db integration
- ✅ Removed XET storage fields from TorchEngine and ModelCache
- ✅ Added load_file_with_pointer_detection() for LFS fallback
- ✅ Feature flags properly implemented across all crates

---

## Build Instructions

```bash
# Set libtorch path (required)
export LIBTORCH=/mnt/hyprstream/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# Basic build (XET disabled by default - has known bugs)
cargo check
cargo build

# With XET support (EXPERIMENTAL - currently broken)
cargo build --features xet

# With OpenTelemetry support
cargo build --features otel

# ROCm (AMD GPU) build
cargo build --release --features rocm

# Run tests
cargo test --workspace
```

**Note**: Building requires libtorch (PyTorch C++ library). See README.md for download/installation instructions.

---

## Architecture Overview

hyprstream is a high-performance LLM inference and training engine built on PyTorch (libtorch) with Git-native model management.

### **Core Philosophy**
1. **Models are Git repositories** - Full version control for ML artifacts
2. **Adapters are files** - Simple, proven architecture (NOT branch-based)
3. **git2db handles all Git operations** - No custom git wrappers
4. **Storage drivers optimize disk usage** - overlay2 on Linux (~80% savings)

---

## Component Architecture

```
hyprstream-torch/
├── crates/
│   ├── hyprstream/          # Main application
│   │   ├── src/
│   │   │   ├── runtime/     # PyTorch inference engine
│   │   │   ├── storage/     # Model & adapter storage (file-based)
│   │   │   ├── git/         # git2db integration & helpers
│   │   │   ├── training/    # LoRA training system
│   │   │   ├── lora/        # LoRA implementation
│   │   │   ├── api/         # REST API (OpenAI-compatible)
│   │   │   ├── cli/         # CLI commands
│   │   │   └── server/      # HTTP server & state management
│   │   └── Cargo.toml
│   │
│   ├── git2db/              # Git repository management library ⭐
│   │   ├── src/
│   │   │   ├── manager.rs   # Global repository cache
│   │   │   ├── registry.rs  # Repository tracking (Git2DB)
│   │   │   ├── branch.rs    # Branch operations
│   │   │   ├── storage/     # Storage drivers (overlay2, vfs)
│   │   │   └── ...
│   │   ├── CLAUDE.md        # git2db AI guide
│   │   └── Cargo.toml
│   │
│   └── git-xet-filter/      # XET large file storage
│       └── ...
│
├── README.md                # User-facing documentation
├── CLAUDE.md                # This file - AI assistant guide
└── docs/                    # Architecture & planning docs
    ├── COW_ARCHITECTURE.md
    ├── GIT2DB-*.md
    └── ...
```

---

## Key Subsystems

### **1. Storage System** (`crates/hyprstream/src/storage/`)

#### **Models** - Git Repositories via git2db
**Key Files**:
- `model_storage.rs` - Model lifecycle management
- `model_registry.rs` - Wraps git2db's Git2DB registry
- `model_ref.rs` - ModelRef syntax parsing (name:branch:commit)

**Design**:
```
models_dir/
├── qwen3-small/.git       # Git repository (managed by git2db)
│   ├── config.json
│   ├── model.safetensors
│   └── adapters/          # ⭐ Adapters stored HERE
│       ├── 00_base.safetensors
│       ├── 00_base.config.json
│       ├── 01_coding.safetensors
│       └── 01_coding.config.json
└── llama3/.git
    └── ...
```

**git2db Integration**:
```rust
// hyprstream wraps git2db for model-specific operations
let registry = Git2DB::open(models_dir).await?;  // git2db registry

// Clone models via git2db
let repo_id = registry.clone("https://huggingface.co/Qwen/Qwen3-0.6B")
    .name("qwen3-small")
    .exec().await?;

// Get repository handle for operations
let handle = registry.repo(&repo_id)?;
handle.branch().create("experiment", None).await?;
```

---

#### **Adapters** - File-Based (NOT Branch-Based) ⭐
**Key Files**:
- `adapter_manager.rs` - Production adapter system (file-based)

**IMPORTANT**: As of Oct 2025, hyprstream uses **file-based adapters**, not branch-based:

**What We Removed** ❌:
- `api/adapter_storage.rs` (~600 lines) - Branch-based adapter system
- `git/branch_manager.rs` (~387 lines) - Custom UUID branch manager
- UUID adapter branches (`adapter/{uuid}`)
- Tag alias resolution for adapters
- Adapter worktree isolation

**What We Use** ✅:
- `storage/adapter_manager.rs` (~200 lines) - File-based adapter system
- Adapters stored in `model/adapters/` as `.safetensors` files
- Simple indexed naming: `00_base.safetensors`, `01_coding.safetensors`
- Config files: `00_base.config.json`
- Committed to git like any other file

**Usage**:
```rust
use crate::storage::AdapterManager;

let adapter_mgr = AdapterManager::new(model_path);

// List adapters (scans adapters/ directory)
let adapters = adapter_mgr.list_adapters()?;

// Load adapter
for adapter in adapters {
    println!("{}: {}", adapter.index, adapter.name);
    // Load from adapter.path
}
```

**Why File-Based**:
- ✅ Simple and proven
- ✅ Fast loading
- ✅ Easy composition (stack multiple adapters)
- ✅ Standard git workflows
- ✅ No complexity overhead

---

#### **XET Large File Storage** - Optional Feature (Disabled by Default)
**Key Files**:
- `git-xet-filter/` - Libgit2 filter integration for automatic LFS/XET smudging
- `storage/lfs_xet.rs` - LFS to XET bridge for non-git file operations (requires `xet` feature)
- `git2db/src/xet_filter.rs` - XET filter initialization and management

**Feature Flag Policy**:
- **git2db**: Optional (`xet-storage` feature), not enabled by default
- **hyprstream**: Optional (`xet` feature), **disabled by default** for stability

**Current Status**: ⚠️ **Experimental - Under Development**
- Architecture designed and implemented
- LFS pointer detection working (with graceful fallback when XET disabled)
- Filter callbacks need refactoring to use `git_filter_buffered_stream_new()` API
- See `docs/XET-FILTER-FIXED.md` for implementation roadmap

**Architecture** (Automatic + Fallback):
```
Git Operations (clone, checkout)
    ↓ automatic
git-xet-filter (Libgit2 Filter)
    ↓ smudges LFS/XET
    ↓ validates SHA256
Models ready to use

Non-Git Files (explicit loads)
    ↓ manual
LfsXetBridge::load_file()
    ↓ detects pointers
    ↓ smudges if needed
File ready to use
```

**How It Works** ✅:
1. **Filter initialization** - `main.rs` registers git-xet-filter globally
2. **Automatic smudging** - Git operations transparently handle LFS/XET
3. **SHA256 validation** - Defense-in-depth (XET client doesn't verify)
4. **Fallback** - `load_file()` for files outside git operations

**Automatic Usage** (Primary - via git-xet-filter):
```rust
// 1. Filter is initialized in main.rs
git2db::xet_filter::initialize(XetConfig::default()).await?;

// 2. Git operations automatically smudge
let registry = Git2DB::open(models_dir).await?;
registry.clone("https://huggingface.co/Qwen/Qwen3-0.6B").await?;
// LFS files are already smudged! No manual processing needed.

// 3. Create worktrees - files are smudged automatically
let worktree = GitManager::global()
    .create_worktree(repo, path, "main")
    .await?;
// All LFS/XET files are ready to use immediately
```

**Fallback Usage** (For non-git files):
```rust
use crate::storage::LfsXetBridge;

let bridge = LfsXetBridge::new(config).await?;

// Load file with automatic pointer detection
// (LFS spec-compliant: metadata check → 100-byte header → smudge if needed)
let weights = bridge.load_file(&model_file_path).await?;

// Or explicit smudging
let lfs_pointer = bridge.parse_lfs_pointer(&pointer_text)?;
let data = bridge.smudge_lfs_pointer(&lfs_pointer).await?;
```

**Why git-xet-filter**:
- ✅ Transparent - Users don't think about LFS
- ✅ Integrated with git operations (clone, checkout, worktree)
- ✅ Errors surface through git2 (proper error handling)
- ✅ Registered once, works everywhere
- ✅ LFS spec-compliant detection

**When to use load_file()**:
- Files outside git repositories
- Explicit pointer resolution needed
- Testing/debugging
- Operations not going through git2

---

### **2. Git Integration** (`crates/hyprstream/src/git/`)

**Key Files**:
- `mod.rs` - Re-exports git2db types, compatibility layer
- `helpers.rs` - Simple git helpers (tag creation, etc.)

**Design Pattern**: Thin wrapper over git2db

```rust
// hyprstream/src/git/mod.rs re-exports git2db
pub use git2db::{Git2DB as GitModelRegistry, TrackedRepository};

// Simple helpers for common operations
pub mod helpers;
```

**DO** ✅:
```rust
// Use git2db directly for git operations
let handle = registry.repo(&repo_id)?;
handle.branch().create("feature", None).await?;

// Use simple helpers for non-git2db operations
use crate::git::helpers::create_tag;
create_tag(model_path, "checkpoint-v1")?;
```

**DON'T** ❌:
```rust
// Don't create custom git wrappers
struct MyCustomBranchManager { ... }  // ❌ Use git2db instead

// Don't bypass git2db
let repo = Repository::open(path)?;   // ❌ Use GitManager::global()
```

---

### **3. Runtime System** (`crates/hyprstream/src/runtime/`)

**Key Files**:
- `torch_engine.rs` - Main PyTorch inference engine
- `inference.rs` - Inference pipeline
- `architectures/` - Model architectures (Qwen, Llama, etc.)
- `kv_cache.rs` - KV cache management
- `sampling.rs` - Token sampling strategies

**Design**: PyTorch (libtorch) via tch-rs bindings

**Core Types**:
```rust
pub struct TorchEngine {
    device: Device,
    models: Arc<RwLock<HashMap<String, Arc<LoadedModel>>>>,
    config: RuntimeConfig,
}

impl TorchEngine {
    pub async fn load_model(&self, model_ref: &str) -> Result<()>;
    pub async fn infer(&self, request: InferenceRequest) -> Result<InferenceResult>;
}
```

---

### **4. Training System** (`crates/hyprstream/src/training/`)

**Key Files**:
- `lora_trainer.rs` - LoRA training implementation
- `checkpoint.rs` - Checkpoint management with git integration
- `data_loader.rs` - Training data loading

**Design**: Create adapter files in `model/adapters/`, commit to git

**Workflow**:
```rust
// 1. Train LoRA adapter
let trainer = LoRATrainer::new(model, config);
trainer.train(dataset).await?;

// 2. Save adapter to model/adapters/
let adapter_path = model_path.join("adapters/01_new_adapter.safetensors");
trainer.save_adapter(&adapter_path)?;

// 3. Commit to git
let repo = GitManager::global().get_repository(model_path)?.open()?;
// ... stage and commit adapter files ...
```

**Checkpoints**:
```rust
use crate::training::CheckpointManager;

// Checkpoints auto-commit at intervals
let checkpoint_mgr = CheckpointManager::new(model_path);
checkpoint_mgr.save_checkpoint(step, weights).await?;

// Create tags for milestones
use crate::git::helpers::create_tag;
create_tag(model_path, "checkpoint-1000")?;
```

---

### **5. API/Server** (`crates/hyprstream/src/api/`, `src/server/`)

**Key Files**:
- `server/state.rs` - Server state management
- `server/routes/openai.rs` - OpenAI-compatible API
- `api/training_service.rs` - Training service (experimental)

**Design**: Axum-based REST API, OpenAI compatibility

**Removed**: `api/adapter_storage.rs` (branch-based adapters - removed Oct 2025)

---

## Common Development Patterns

### **Pattern 1: Working with Models**

```rust
use crate::storage::{ModelStorage, ModelRef};

// Initialize storage
let storage = ModelStorage::create(models_dir).await?;
let registry = storage.registry();  // git2db Git2DB

// Clone a model
let repo_id = registry.clone("https://huggingface.co/Qwen/Qwen3-0.6B")
    .name("qwen3-small")
    .shallow(true)
    .exec().await?;

// Get model path
let model_path = registry.repo(&repo_id)?.worktree();

// Parse ModelRef
let model_ref = ModelRef::parse("qwen3-small:main")?;
```

---

### **Pattern 2: Working with Adapters**

```rust
use crate::storage::AdapterManager;

// Initialize adapter manager for a model
let adapter_mgr = AdapterManager::new(model_path);

// Ensure adapters directory exists
adapter_mgr.ensure_adapters_dir()?;

// List adapters
let adapters = adapter_mgr.list_adapters()?;
for adapter in &adapters {
    println!("{:02}_{}.safetensors", adapter.index, adapter.name);
}

// Get next available index
let next_index = adapter_mgr.get_next_index()?;

// Save new adapter
let adapter_path = adapter_mgr.adapters_dir
    .join(format!("{:02}_{}.safetensors", next_index, "my_adapter"));
trainer.save_adapter(&adapter_path)?;
```

---

### **Pattern 3: Git Operations**

```rust
// For standard git operations, use git2db
let handle = registry.repo(&repo_id)?;

// Branch management
handle.branch().create("experiment", None).await?;
handle.branch().checkout("experiment").await?;
handle.branch().list().await?;

// Staging
handle.staging().add_all()?;
handle.staging().add_file("adapters/01_new.safetensors")?;

// For operations not yet in git2db, use helpers or escape hatch
use crate::git::helpers::create_tag;
create_tag(model_path, "v1.0")?;

// Escape hatch for missing APIs
let repo = handle.open()?;  // Get raw git2::Repository
// Use git2 directly...
```

---

### **Pattern 4: Tag Creation** (Common)

```rust
// Simple tag creation (e.g., for checkpoints)
use crate::git::helpers::create_tag;

create_tag(model_path, "checkpoint-step-1000")?;
create_tag(model_path, "v1.0-release")?;

// Internally uses GitManager for caching:
// 1. Get cached repository
// 2. Get HEAD commit
// 3. Create lightweight tag
// 4. Overwrites if exists
```

---

## Testing

```bash
# Run all tests
cargo test --workspace

# Test specific crate
cargo test -p hyprstream
cargo test -p git2db

# Test with logging
RUST_LOG=debug cargo test

# Test specific module
cargo test -p hyprstream storage::adapter_manager

# Integration tests
cargo test --test '*'
```

---

## Common Issues & Solutions

### **Issue: "BranchManager not found"**
**Cause**: BranchManager was removed in Oct 2025
**Solution**: Use git2db's branch operations or `git::helpers`

```rust
// OLD (removed):
let branch_mgr = BranchManager::new(model_path)?;
branch_mgr.create_tag("v1.0", "HEAD")?;

// NEW:
use crate::git::helpers::create_tag;
create_tag(model_path, "v1.0")?;
```

---

### **Issue: "AdapterStorage not found"**
**Cause**: AdapterStorage (branch-based) was removed in Oct 2025
**Solution**: Use AdapterManager (file-based)

```rust
// OLD (removed):
let adapter_storage = AdapterStorage::new(base_dir).await?;
adapter_storage.create_adapter("name", config).await?;

// NEW:
let adapter_mgr = AdapterManager::new(model_path);
// Create adapter files in adapters/ directory
// Commit to git like any other file
```

---

### **Issue: "libtorch not found"**
**Solution**: Set LIBTORCH environment variable

```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH
cargo build
```

---

### **Issue: "overlayfs mount failed"**
**Cause**: overlayfs requires Linux kernel support
**Solution**: Use different backend or fallback to vfs

```bash
# Try user namespaces backend
export GIT2DB_OVERLAY_BACKEND=userns

# Or fallback to plain worktrees
export GIT2DB_WORKTREE_DRIVER=vfs

cargo test
```

---

## git2db Integration Details

See `/crates/git2db/CLAUDE.md` for comprehensive git2db documentation.

**Quick Reference**:
- **GitManager**: `GitManager::global()` - Singleton repository cache
- **Git2DB Registry**: `Git2DB::open(dir)` - Repository tracking
- **RepositoryHandle**: `registry.repo(id)` - Scoped git operations
- **Storage Drivers**: overlay2 (Linux), vfs (fallback)

**Key APIs**:
```rust
// Repository management
let repo_cache = GitManager::global().get_repository(path)?;
let repo = repo_cache.open()?;

// High-level operations
let handle = registry.repo(&repo_id)?;
handle.branch().create("new", None).await?;
handle.staging().add_all()?;

// Worktrees (auto storage driver selection)
let wt = GitManager::global()
    .create_worktree(repo, wt_path, "branch")
    .await?;
```

---

## Architecture Decisions Log

### **Oct 2025: Adapter Architecture Simplification**

**Decision**: Remove branch-based adapter system, use file-based only

**Removed**:
- `api/adapter_storage.rs` (600 lines)
- `git/branch_manager.rs` (387 lines)
- UUID adapter branches
- Tag alias resolution
- Total: ~1,000 lines removed

**Rationale**:
1. File-based adapters already in production use
2. Branch-based adapters were experimental, unused
3. Dual systems caused confusion
4. File-based is simpler, proven, and fast
5. git2db provides all needed git operations

**Impact**:
- Net: -980 lines of code
- No breaking changes to production features
- Clearer architecture
- Easier maintenance

---

### **Oct 2025: ModelSharing Removal (Layering Violation)**

**Decision**: Remove ModelSharing module - P2P belongs at transport layer

**Removed**:
- `storage/sharing.rs` (452 lines) - ModelSharing, ShareableModelRef
- `ShareableModelRef` duplicate in `git/mod.rs` (8 lines)
- CLI Share/Import commands and handlers (89 lines)
- Total: ~550 lines removed

**Rationale**:
1. **Architectural layering violation**: P2P is a transport concern, not application concern
2. ModelSharing just wrapped `git clone` with JSON manifests
3. P2P already works via GitTorrent at transport layer
4. Signature verification was TODO (not implemented)
5. Metrics/size not validated
6. Redundant with standard git operations

**What replaced it**:
- Use standard git operations: `hyprstream clone <url>` (works with `gittorrent://` URLs)
- P2P handled transparently at transport layer (git2db + GitTorrent)
- No application-layer wrapper needed

**Impact**:
- Net: -550 lines of code
- Cleaner architecture (correct layering)
- P2P still works (at correct layer)
- Simpler to understand and maintain

---

### **Oct 2025: XET Storage Consolidation**

**Decision**: Consolidate XET operations into git2db, use LfsXetBridge for ML workflows

**Removed/Refactored**:
- `storage/xet_native.rs` (863 lines) - **Completely removed**
- Duplicated XET operations (~140 lines)
- Total: **~1,003 lines deleted**

**Added**:
- `storage/lfs_xet.rs` (520 lines) - LFS-specific ML workflows
- `git2db::xet::XetStorage` - Core XET operations
- Hash-based downloads for LFS pointer resolution
- All async I/O (was mixed blocking/async)

**Rationale**:
1. **Single source of truth**: Core XET operations now in git2db
2. **Better layering**: hyprstream = ML workflows, git2db = git + XET primitives
3. **Reusability**: Other projects can use git2db's XET without ML dependencies
4. **Efficiency**: Direct file writes, async I/O, hash-based downloads
5. **Consistency**: Matches storage drivers pattern (overlay2, vfs)

**Architecture** (3-tier):
```
Application Layer (hyprstream)
    ├── LfsXetBridge - LFS translation for Hugging Face models
    └── Uses git2db::xet
             ↓
Core Layer (git2db)
    ├── XetStorage - Core XET operations (clean, smudge, hash downloads)
    └── Uses git-xet-filter
             ↓
Filter Layer (git-xet-filter)
    └── Libgit2 filter integration (automatic git operations)
```

**Impact**:
- Net: **-1,003 lines** of redundant/duplicated code
- API clarity: `git2db::xet` for core, `LfsXetBridge` for ML
- Performance: Async I/O, direct file writes, hash-based downloads
- Maintainability: Single implementation to fix bugs
- **Complete migration**: No compatibility shims remaining
- Future: Can enable automatic smudge/clean via libgit2 filter

---

## Additional Resources

- **README.md** - User-facing documentation
- **crates/git2db/CLAUDE.md** - git2db AI guide
- **docs/COW_ARCHITECTURE.md** - Worktree CoW mechanisms
- **docs/GIT2DB-*.md** - git2db design documents
- **docs/** - Historical planning documents

---

## Quick Command Reference

```bash
# Build
export LIBTORCH=/mnt/hyprstream/libtorch
cargo build --features otel

# Test
cargo test --workspace

# Run hyprstream CLI
cargo run --bin hyprstream -- --help

# Run server
cargo run --bin hyprstream serve --port 8080

# Format code
cargo fmt --all

# Check code
cargo clippy --all-targets --all-features
```

---

This guide should help you navigate the hyprstream codebase and understand its architecture. The key takeaway: **Models are git repos, adapters are files, git2db handles git operations.**
