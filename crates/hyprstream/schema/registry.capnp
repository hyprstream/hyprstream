@0xd4e6f8a2b1c3d5e7;

using import "/annotations.capnp".mcpScope;
using import "/annotations.capnp".mcpDescription;
using import "/streaming.capnp".StreamInfo;

# Cap'n Proto schema for registry service
#
# The registry service manages git repositories (models).
# Uses REQ/REP pattern for all operations.
#
# Convention: Request variants use camelCase names. Response variants
# use the same name suffixed with "Result" to avoid Cap'n Proto naming
# collisions. The code generator strips "Result" to pair them.
# Repo-scoped ops are nested under `repo`/`repoResult`.

struct RegistryRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    # List all models available in the registry
    list @1 :Void $mcpScope(query);
    # Get repository information by ID
    get @2 :Text $mcpScope(query);
    # Get repository information by name
    getByName @3 :Text $mcpScope(query);
    # Clone a model repository from a URL
    clone @4 :CloneRequest $mcpScope(write);
    # Register an existing local repository
    register @5 :RegisterRequest $mcpScope(write);
    # Remove a repository from the registry
    remove @6 :Text $mcpScope(manage);
    # Check registry service health
    healthCheck @7 :Void;
    # Clone a model repository from a URL (streaming progress)
    cloneStream @8 :CloneRequest $mcpScope(write) $mcpDescription("Clone a model repository from a URL (streaming progress)");

    # Repository-scoped operations (requires repoId)
    repo @9 :RepositoryRequest;
  }
}

# Repository-scoped request: operations on a specific repository.
# Generator detects the non-union field (repoId) + inner union pattern
# and produces a RepositoryClient with repoId curried in.
struct RepositoryRequest {
  repoId @0 :Text;
  union {
    # Create a new worktree for the repository
    createWorktree @1 :CreateWorktreeRequest $mcpScope(write);
    # List all worktrees for the repository
    listWorktrees @2 :Void $mcpScope(query);
    # Remove a worktree from the repository
    removeWorktree @3 :RemoveWorktreeRequest $mcpScope(manage);
    # Create a new branch in the repository
    createBranch @4 :BranchRequest $mcpScope(write);
    # List all branches in the repository
    listBranches @5 :Void $mcpScope(query);
    # Checkout a branch or reference
    checkout @6 :CheckoutRequest $mcpScope(write);
    # Stage all modified files
    stageAll @7 :Void $mcpScope(write);
    # Stage specific files
    stageFiles @8 :StageFilesRequest $mcpScope(write);
    # Create a commit with staged changes
    commit @9 :CommitRequest $mcpScope(write);
    # Merge a branch into current branch
    merge @10 :MergeRequest $mcpScope(write);
    # Abort an in-progress merge
    abortMerge @11 :Void $mcpScope(write);
    # Continue a merge after resolving conflicts
    continueMerge @12 :ContinueMergeRequest $mcpScope(write);
    # Exit merge state without committing
    quitMerge @13 :Void $mcpScope(write);
    # Get the current HEAD reference
    getHead @14 :Void $mcpScope(query);
    # Get information about a specific reference
    getRef @15 :GetRefRequest $mcpScope(query);
    # Get repository status (short format)
    status @16 :Void $mcpScope(query);
    # Get detailed repository status with file changes
    detailedStatus @17 :Void $mcpScope(query);
    # List all remotes for the repository
    listRemotes @18 :Void $mcpScope(query);
    # Add a new remote to the repository
    addRemote @19 :AddRemoteRequest $mcpScope(write);
    # Remove a remote from the repository
    removeRemote @20 :RemoveRemoteRequest $mcpScope(manage);
    # Set the URL for a remote
    setRemoteUrl @21 :SetRemoteUrlRequest $mcpScope(write);
    # Rename a remote
    renameRemote @22 :RenameRemoteRequest $mcpScope(write);
    # Push commits to a remote repository
    push @23 :PushRequest $mcpScope(write);
    # Amend the last commit with new changes
    amendCommit @24 :AmendCommitRequest $mcpScope(write);
    # Create a commit with specified author information
    commitWithAuthor @25 :CommitWithAuthorRequest $mcpScope(write);
    # Stage all files including untracked files
    stageAllIncludingUntracked @26 :Void $mcpScope(write);
    # List all tags in the repository
    listTags @27 :Void $mcpScope(query);
    # Create a new tag
    createTag @28 :CreateTagRequest $mcpScope(write);
    # Delete a tag from the repository
    deleteTag @29 :DeleteTagRequest $mcpScope(manage);
    # Pull and update from remote repository
    update @30 :UpdateRequest $mcpScope(write);
    # Worktree-scoped filesystem operations
    worktree @31 :WorktreeRequest;
  }
}

struct RegistryResponse {
  # Request ID this response corresponds to
  requestId @0 :UInt64;

  # Response payload — variants suffixed with "Result" to pair with request
  union {
    error @1 :ErrorInfo;
    listResult @2 :List(TrackedRepository);
    getResult @3 :TrackedRepository;
    getByNameResult @4 :TrackedRepository;
    cloneResult @5 :TrackedRepository;
    registerResult @6 :TrackedRepository;
    removeResult @7 :Void;
    healthCheckResult @8 :HealthStatus;
    cloneStreamResult @9 :StreamInfo;
    repoResult @10 :RepositoryResponse;
  }
}

# Repository-scoped response: inner union variants match request names exactly.
struct RepositoryResponse {
  union {
    error @0 :ErrorInfo;
    createWorktree @1 :Text;
    listWorktrees @2 :List(WorktreeInfo);
    removeWorktree @3 :Void;
    createBranch @4 :Void;
    listBranches @5 :List(Text);
    checkout @6 :Void;
    stageAll @7 :Void;
    stageFiles @8 :Void;
    commit @9 :Text;
    merge @10 :Void;
    abortMerge @11 :Void;
    continueMerge @12 :Void;
    quitMerge @13 :Void;
    getHead @14 :Text;
    getRef @15 :Text;
    status @16 :RepositoryStatus;
    detailedStatus @17 :DetailedStatusInfo;
    listRemotes @18 :List(RemoteInfo);
    addRemote @19 :Void;
    removeRemote @20 :Void;
    setRemoteUrl @21 :Void;
    renameRemote @22 :Void;
    push @23 :Void;
    amendCommit @24 :Text;
    commitWithAuthor @25 :Text;
    stageAllIncludingUntracked @26 :Void;
    listTags @27 :List(Text);
    createTag @28 :Void;
    deleteTag @29 :Void;
    update @30 :Void;
    worktreeResult @31 :WorktreeResponse;
  }
}

# --- Seek direction enum ---

enum SeekWhence {
  set @0;    # SEEK_SET — from beginning
  cur @1;    # SEEK_CUR — from current position
  end @2;    # SEEK_END — from end of file
}

# --- WorktreeRequest: worktree-scoped filesystem operations ---
# Generator detects non-union field (name) + inner union
# and produces WorktreeClient with name curried in.

struct WorktreeRequest {
  name @0 :Text;
  union {
    # FD lifecycle (authorized at open, no per-op auth)
    open @1 :FsOpenRequest $mcpScope(write);
    close @2 :FsCloseRequest;
    # FD I/O (capability-based, authorized at open)
    read @3 :FsReadRequest;
    write @4 :FsWriteRequest;
    pread @5 :FsPreadRequest;
    pwrite @6 :FsPwriteRequest;
    seek @7 :FsSeekRequest;
    truncate @8 :FsTruncateRequest;
    fsync @9 :FsSyncRequest;
    # Path operations (stateless)
    stat @10 :FsPathRequest $mcpScope(query);
    mkdir @11 :FsMkdirRequest $mcpScope(write);
    remove @12 :FsPathRequest $mcpScope(write);
    rmdir @13 :FsPathRequest $mcpScope(write);
    rename @14 :FsRenameRequest $mcpScope(write);
    copy @15 :FsCopyRequest $mcpScope(write);
    listDir @16 :FsPathRequest $mcpScope(query);
    # Streaming (bulk transfer via StreamService)
    openStream @17 :FsOpenRequest $mcpScope(write);
  }
}

# Open: explicit bool fields instead of platform-specific flag bits
struct FsOpenRequest {
  path @0 :Text;
  read @1 :Bool;        # default true
  write @2 :Bool;       # default false
  create @3 :Bool;      # create if not exists
  truncate @4 :Bool;    # truncate to zero on open
  append @5 :Bool;      # writes always at end
  exclusive @6 :Bool;   # fail if exists + create
}

struct FsCloseRequest { fd @0 :UInt32; }
struct FsReadRequest { fd @0 :UInt32; length @1 :UInt64; }
struct FsWriteRequest { fd @0 :UInt32; data @1 :Data; }
struct FsPreadRequest { fd @0 :UInt32; offset @1 :UInt64; length @2 :UInt64; }
struct FsPwriteRequest { fd @0 :UInt32; offset @1 :UInt64; data @2 :Data; }
struct FsSeekRequest { fd @0 :UInt32; offset @1 :Int64; whence @2 :SeekWhence; }
struct FsTruncateRequest { fd @0 :UInt32; length @1 :UInt64; }
struct FsSyncRequest { fd @0 :UInt32; dataOnly @1 :Bool; }
struct FsPathRequest { path @0 :Text; }
struct FsMkdirRequest { path @0 :Text; recursive @1 :Bool; }
struct FsRenameRequest { src @0 :Text; dst @1 :Text; }
struct FsCopyRequest { src @0 :Text; dst @1 :Text; }

# --- WorktreeResponse ---

struct WorktreeResponse {
  union {
    error @0 :ErrorInfo;   # Default variant (fail-closed)
    open @1 :FsOpenResponse;
    close @2 :Void;
    read @3 :FsReadResponse;
    write @4 :FsWriteResponse;
    pread @5 :FsReadResponse;
    pwrite @6 :FsWriteResponse;
    seek @7 :FsSeekResponse;
    truncate @8 :Void;
    fsync @9 :Void;
    stat @10 :FsStatResponse;
    mkdir @11 :Void;
    remove @12 :Void;
    rmdir @13 :Void;
    rename @14 :Void;
    copy @15 :Void;
    listDir @16 :List(FsDirEntryInfo);
    openStream @17 :FsStreamInfoResponse;
  }
}

struct FsOpenResponse { fd @0 :UInt32; }
struct FsReadResponse { data @0 :Data; }
struct FsWriteResponse { bytesWritten @0 :UInt64; }
struct FsSeekResponse { position @0 :UInt64; }
struct FsStatResponse {
  exists @0 :Bool;
  isDir @1 :Bool;
  size @2 :UInt64;
  modifiedAt @3 :Int64;
}
struct FsDirEntryInfo { name @0 :Text; isDir @1 :Bool; size @2 :UInt64; }
struct FsStreamInfoResponse {
  fd @0 :UInt32;
  streamId @1 :Text;
  streamEndpoint @2 :Text;
  serverPubkey @3 :Data;
}

# Clone Request

struct CloneRequest {
  url @0 :Text;
  name @1 :Text;
  shallow @2 :Bool;
  depth @3 :UInt32;
  branch @4 :Text;
}

# Register Request (for local repos)

struct RegisterRequest {
  path @0 :Text;
  name @1 :Text;
  trackingRef @2 :Text;
}

# Create Worktree Request (repoId removed — curried into RepositoryClient)

struct CreateWorktreeRequest {
  path @0 :Text;
  branchName @1 :Text;
  createBranch @2 :Bool;
}

struct RemoveWorktreeRequest {
  worktreePath @0 :Text;
  force @1 :Bool;
}

struct WorktreeInfo {
  path @0 :Text;
  branchName @1 :Text;
  headOid @2 :Text;
  isLocked @3 :Bool;
  isDirty @4 :Bool;
}

# Branch Request (repoId removed — curried)

struct BranchRequest {
  branchName @0 :Text;
  startPoint @1 :Text;
}

# Checkout Request (repoId removed — curried)

struct CheckoutRequest {
  refName @0 :Text;
  createBranch @1 :Bool;
}

# Stage Files Request (repoId removed — curried)

struct StageFilesRequest {
  files @0 :List(Text);
}

# Commit Request (repoId removed — curried)

struct CommitRequest {
  message @0 :Text;
  author @1 :Text;
  email @2 :Text;
}

# Get Ref Request (repoId removed — curried)

struct GetRefRequest {
  refName @0 :Text;
}

# Tracked Repository

struct TrackedRepository {
  id @0 :Text;
  name @1 :Text;
  url @2 :Text;
  worktreePath @3 :Text;
  trackingRef @4 :Text;
  currentOid @5 :Text;
  registeredAt @6 :Int64;
}

# Health Status

struct HealthStatus {
  status @0 :Text;
  repositoryCount @1 :UInt32;
  worktreeCount @2 :UInt32;
  cacheHits @3 :UInt64;
  cacheMisses @4 :UInt64;
}

# Error Information

struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
  details @2 :Text;
}

# Remote Operations

struct RemoteInfo {
  name @0 :Text;
  url @1 :Text;
  pushUrl @2 :Text;
}

struct AddRemoteRequest {
  name @0 :Text;
  url @1 :Text;
}

struct RemoveRemoteRequest {
  name @0 :Text;
}

struct SetRemoteUrlRequest {
  name @0 :Text;
  url @1 :Text;
}

struct RenameRemoteRequest {
  oldName @0 :Text;
  newName @1 :Text;
}

# Merge Request (repoId removed — curried)

struct MergeRequest {
  source @0 :Text;
  message @1 :Text;           # Optional merge message
}

# Repository Status

struct RepositoryStatus {
  branch @0 :Text;             # Optional branch name
  headOid @1 :Text;            # Optional HEAD commit OID
  ahead @2 :UInt32;
  behind @3 :UInt32;
  isClean @4 :Bool;
  modifiedFiles @5 :List(Text); # Paths of modified files
}

# Push Request (repoId removed — curried)

struct PushRequest {
  remote @0 :Text;
  refspec @1 :Text;
  force @2 :Bool;
}

# Amend Commit Request (repoId removed — curried)

struct AmendCommitRequest {
  message @0 :Text;
}

# Commit with Author Request (repoId removed — curried)

struct CommitWithAuthorRequest {
  message @0 :Text;
  authorName @1 :Text;
  authorEmail @2 :Text;
}

# Continue Merge Request (repoId removed — curried)

struct ContinueMergeRequest {
  message @0 :Text;  # Optional merge message
}

# Create Tag Request (repoId removed — curried)

struct CreateTagRequest {
  name @0 :Text;
  target @1 :Text;  # Optional target (defaults to HEAD)
}

# Delete Tag Request (repoId removed — curried)

struct DeleteTagRequest {
  name @0 :Text;
}

# Update Request (repoId removed — curried)

struct UpdateRequest {
  refspec @0 :Text;
}

# Detailed Status Info

struct DetailedStatusInfo {
  branch @0 :Text;             # Optional branch name
  headOid @1 :Text;            # Optional HEAD commit OID
  mergeInProgress @2 :Bool;
  rebaseInProgress @3 :Bool;
  ahead @4 :UInt32;
  behind @5 :UInt32;
  files @6 :List(FileStatusInfo);
}

struct FileStatusInfo {
  path @0 :Text;
  indexStatus @1 :Text;        # Single char: A, M, D, R, ?, T, U, or empty
  worktreeStatus @2 :Text;     # Single char: A, M, D, R, ?, T, U, or empty
}
