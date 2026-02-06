@0xd4e6f8a2b1c3d5e7;

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
    # Global operations (top-level)
    list @1 :Void;
    get @2 :Text;           # repo_id
    getByName @3 :Text;     # name
    clone @4 :CloneRequest;
    register @5 :RegisterRequest;
    remove @6 :Text;        # repo_id
    healthCheck @7 :Void;
    cloneStream @8 :CloneRequest;  # Clone with streaming progress

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
    createWorktree @1 :WorktreeRequest;
    listWorktrees @2 :Void;
    removeWorktree @3 :RemoveWorktreeRequest;
    createBranch @4 :BranchRequest;
    listBranches @5 :Void;
    checkout @6 :CheckoutRequest;
    stageAll @7 :Void;
    stageFiles @8 :StageFilesRequest;
    commit @9 :CommitRequest;
    merge @10 :MergeRequest;
    abortMerge @11 :Void;
    continueMerge @12 :ContinueMergeRequest;
    quitMerge @13 :Void;
    getHead @14 :Void;
    getRef @15 :GetRefRequest;
    status @16 :Void;
    detailedStatus @17 :Void;
    listRemotes @18 :Void;
    addRemote @19 :AddRemoteRequest;
    removeRemote @20 :RemoveRemoteRequest;
    setRemoteUrl @21 :SetRemoteUrlRequest;
    renameRemote @22 :RenameRemoteRequest;
    push @23 :PushRequest;
    amendCommit @24 :AmendCommitRequest;
    commitWithAuthor @25 :CommitWithAuthorRequest;
    stageAllIncludingUntracked @26 :Void;
    listTags @27 :Void;
    createTag @28 :CreateTagRequest;
    deleteTag @29 :DeleteTagRequest;
    update @30 :UpdateRequest;
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
    cloneStreamResult @9 :StreamStartedInfo;
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
  }
}

# Stream Started Info (for cloneStream)
struct StreamStartedInfo {
  streamId @0 :Text;
  streamEndpoint @1 :Text;
  serverPubkey @2 :Data;  # 32-byte DH public key for key derivation
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

# Worktree Request (repoId removed — curried into RepositoryClient)

struct WorktreeRequest {
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
