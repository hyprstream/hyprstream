@0xd4e6f8a2b1c3d5e7;

# Cap'n Proto schema for registry service
#
# The registry service manages git repositories (models).
# Uses REQ/REP pattern for all operations.

struct RegistryRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload (union of request types)
  union {
    # Repository queries
    list @1 :Void;
    get @2 :Text;           # repo_id
    getByName @3 :Text;     # name

    # Repository operations
    clone @4 :CloneRequest;
    register @5 :RegisterRequest;
    remove @6 :Text;        # repo_id
    update @7 :UpdateRequest;

    # Worktree operations
    createWorktree @8 :WorktreeRequest;
    listWorktrees @9 :Text; # repo_id
    removeWorktree @10 :RemoveWorktreeRequest;

    # Branch operations
    createBranch @11 :BranchRequest;
    listBranches @12 :Text; # repo_id
    checkout @13 :CheckoutRequest;

    # Staging operations
    stageAll @14 :Text;     # repo_id
    stageFiles @15 :StageFilesRequest;
    commit @16 :CommitRequest;

    # Merge operations
    merge @25 :MergeRequest;

    # Status operations
    status @26 :Text;      # repo_id

    # Reference operations
    getHead @17 :Text;      # repo_id
    getRef @18 :GetRefRequest;

    # Health/Lifecycle
    healthCheck @19 :Void;

    # Remote operations
    listRemotes @20 :Text;  # repo_id
    addRemote @21 :AddRemoteRequest;
    removeRemote @22 :RemoveRemoteRequest;
    setRemoteUrl @23 :SetRemoteUrlRequest;
    renameRemote @24 :RenameRemoteRequest;
  }
}

struct RegistryResponse {
  # Request ID this response corresponds to
  requestId @0 :UInt64;

  # Response payload (union of response types)
  union {
    success @1 :Void;
    error @2 :ErrorInfo;
    repository @3 :TrackedRepository;
    repositories @4 :List(TrackedRepository);
    worktrees @5 :List(WorktreeInfo);
    branches @6 :List(Text);
    path @7 :Text;
    commitOid @8 :Text;
    refOid @9 :Text;
    health @10 :HealthStatus;
    remotes @11 :List(RemoteInfo);
    repositoryStatus @12 :RepositoryStatus;
  }
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

# Update Request

struct UpdateRequest {
  repoId @0 :Text;
  refspec @1 :Text;
}

# Worktree Request

struct WorktreeRequest {
  repoId @0 :Text;
  path @1 :Text;
  branchName @2 :Text;
  createBranch @3 :Bool;
}

struct RemoveWorktreeRequest {
  repoId @0 :Text;
  worktreePath @1 :Text;
  force @2 :Bool;
}

struct WorktreeInfo {
  path @0 :Text;
  branchName @1 :Text;
  headOid @2 :Text;
  isLocked @3 :Bool;
  isDirty @4 :Bool;
}

# Branch Request

struct BranchRequest {
  repoId @0 :Text;
  branchName @1 :Text;
  startPoint @2 :Text;
}

# Checkout Request

struct CheckoutRequest {
  repoId @0 :Text;
  refName @1 :Text;
  createBranch @2 :Bool;
}

# Stage Files Request

struct StageFilesRequest {
  repoId @0 :Text;
  files @1 :List(Text);
}

# Commit Request

struct CommitRequest {
  repoId @0 :Text;
  message @1 :Text;
  author @2 :Text;
  email @3 :Text;
}

# Get Ref Request

struct GetRefRequest {
  repoId @0 :Text;
  refName @1 :Text;
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
  repoId @0 :Text;
  name @1 :Text;
  url @2 :Text;
}

struct RemoveRemoteRequest {
  repoId @0 :Text;
  name @1 :Text;
}

struct SetRemoteUrlRequest {
  repoId @0 :Text;
  name @1 :Text;
  url @2 :Text;
}

struct RenameRemoteRequest {
  repoId @0 :Text;
  oldName @1 :Text;
  newName @2 :Text;
}

# Merge Request

struct MergeRequest {
  repoId @0 :Text;
  source @1 :Text;
  message @2 :Text;           # Optional merge message
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
