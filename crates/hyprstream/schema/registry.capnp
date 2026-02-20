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
    # Ensure a worktree exists for a branch (create if needed, return path)
    ensureWorktree @32 :EnsureWorktreeRequest $mcpScope(write) $mcpDescription("Ensure a worktree exists for a branch, creating if needed");
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
    ensureWorktree @32 :Text;
  }
}

# --- 9P2000-inspired types ---
# Qid uniquely identifies a file version (inode + ctime).
struct Qid {
  qtype @0 :UInt8;      # QTDIR=0x80 QTAPPEND=0x40 QTEXCL=0x20 QTFILE=0x00
  version @1 :UInt32;   # st_ctime (seconds)
  path @2 :UInt64;      # st_ino (unique file id)
}

# File metadata (9P stat structure).
struct NpStat {
  qid @0 :Qid;
  mode @1 :UInt32;
  atime @2 :UInt32;
  mtime @3 :UInt32;
  length @4 :UInt64;
  name @5 :Text;
  uid @6 :Text;
  gid @7 :Text;
  muid @8 :Text;
}

# --- WorktreeRequest: 9P2000-inspired worktree filesystem protocol ---
# 10 operations replacing 19 POSIX ops. Read is always offset+count,
# bounded by iounit — no unbounded readFile convenience method.
# Generator detects non-union field (name) + inner union
# and produces WorktreeClient with name curried in.

struct WorktreeRequest {
  name @0 :Text;
  union {
    # Walk: resolve path components to get a fid (like 9P Twalk)
    walk @1 :NpWalk $mcpScope(query);
    # Open: open a walked fid for I/O (like 9P Topen)
    open @2 :NpOpen $mcpScope(write);
    # Create: create a file/dir under a walked directory fid (like 9P Tcreate)
    create @3 :NpCreate $mcpScope(write);
    # Read: offset+count read, server clamps to iounit (like 9P Tread)
    read @4 :NpRead;
    # Write: offset+data write, server rejects if > iounit (like 9P Twrite)
    write @5 :NpWrite;
    # Clunk: release a fid (like 9P Tclunk)
    clunk @6 :NpClunk;
    # Remove: clunk + delete file/dir (like 9P Tremove)
    remove @7 :NpRemove $mcpScope(manage);
    # Stat: get file metadata (like 9P Tstat)
    npStat @8 :NpStatReq $mcpScope(query);
    # Wstat: modify file metadata (like 9P Twstat)
    wstat @9 :NpWstat $mcpScope(write);
    # Flush: cancel pending operation (like 9P Tflush)
    flush @10 :NpFlush;
    # Per-file control operations, scoped by fid
    ctl @11 :CtlRequest;
  }
}

# 9P Request structs

struct NpWalk {
  fid @0 :UInt32;          # source fid (0 = root, auto-attached per worktree)
  newfid @1 :UInt32;       # fid to assign to walked path
  wnames @2 :List(Text);   # path components (empty = clone fid)
}

struct NpOpen {
  fid @0 :UInt32;
  mode @1 :UInt8;          # OREAD=0 OWRITE=1 ORDWR=2; OTRUNC=0x10 ORCLOSE=0x40
}

struct NpCreate {
  fid @0 :UInt32;          # must be a walked directory fid
  name @1 :Text;           # name of file/dir to create
  perm @2 :UInt32;         # DMDIR=0x80000000 for dirs, otherwise file mode
  mode @3 :UInt8;          # open mode for the new file (same as NpOpen.mode)
}

struct NpRead {
  fid @0 :UInt32;
  offset @1 :UInt64;
  count @2 :UInt32;        # server clamps to iounit
}

struct NpWrite {
  fid @0 :UInt32;
  offset @1 :UInt64;
  data @2 :Data;           # server rejects if > iounit
}

struct NpClunk   { fid @0 :UInt32; }
struct NpRemove  { fid @0 :UInt32; }
struct NpStatReq { fid @0 :UInt32; }
struct NpWstat   { fid @0 :UInt32; stat @1 :NpStat; }
struct NpFlush   { oldtag @0 :UInt64; }

# Ensure Worktree Request (repoId removed — curried)

struct EnsureWorktreeRequest {
  branch @0 :Text;
}

# --- WorktreeResponse: 9P2000-inspired responses ---

struct WorktreeResponse {
  union {
    error @0 :ErrorInfo;
    # Walk response: qid of the walked-to file
    walk @1 :RWalk;
    # Open response: qid + iounit (max I/O per message)
    open @2 :ROpen;
    # Create response: same shape as open (qid + iounit)
    create @3 :ROpen;
    # Read response: data (len < count means EOF)
    read @4 :RRead;
    # Write response: bytes actually written
    write @5 :RWrite;
    # Clunk response: success (void)
    clunk @6 :Void;
    # Remove response: success (void)
    remove @7 :Void;
    # Stat response: full file metadata
    npStat @8 :RStat;
    # Wstat response: success (void)
    wstat @9 :Void;
    # Flush response: success (void)
    flush @10 :Void;
    # Per-file control response
    ctlResult @11 :CtlResponse;
  }
}

# 9P Response structs

struct RWalk  { qid @0 :Qid; }
struct ROpen  { qid @0 :Qid; iounit @1 :UInt32; }
struct RRead  { data @0 :Data; }
struct RWrite { count @0 :UInt32; }
struct RStat  { stat @0 :NpStat; }

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

# File Change Type Enum

enum FileChangeType {
  none @0;
  added @1;
  modified @2;
  deleted @3;
  renamed @4;
  untracked @5;
  typeChanged @6;
  conflicted @7;
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
  indexStatus @1 :FileChangeType;
  worktreeStatus @2 :FileChangeType;
}

# --- CtlRequest: per-file control operations, scoped by fid ---
# Generator detects non-union field (fid) + inner union pattern
# and produces CtlClient with fid curried in.

struct CtlRequest {
  fid @0 :UInt32;           # scope field (curried in generated CtlClient)
  union {
    # ── Git introspection ──
    status    @1 :Void               $mcpScope(query)  $mcpDescription("Git status of this file");
    log       @2 :CtlLogRequest      $mcpScope(query)  $mcpDescription("Commits touching this file");
    diff      @3 :CtlDiffRequest     $mcpScope(query)  $mcpDescription("Diff this file against a ref");
    blame     @4 :Void               $mcpScope(query)  $mcpDescription("Git blame for this file");
    checkout  @5 :CtlCheckoutRequest $mcpScope(write)  $mcpDescription("Restore file content from a ref");

    # ── File control ──
    validate  @6 :Void               $mcpScope(query)  $mcpDescription("Validate file format");
    info      @7 :Void               $mcpScope(query)  $mcpDescription("File metadata and git state");

    # ── CRDT editing ──
    editOpen  @8  :EditOpenRequest   $mcpScope(write)  $mcpDescription("Open file for CRDT editing");
    editState @9  :Void              $mcpScope(query)  $mcpDescription("Get current CRDT document state");
    editApply @10 :EditApplyRequest  $mcpScope(write)  $mcpDescription("Apply automerge CRDT change");
    editClose @11 :Void              $mcpScope(write)  $mcpDescription("Close CRDT editing session");
    # Flush: serialize CRDT state to disk (does NOT stage or commit)
    ctlFlush  @12 :Void              $mcpScope(write)  $mcpDescription("Write CRDT state to disk file");
  }
}

# ── ctl supporting structs ──

struct CtlLogRequest { maxCount @0 :UInt32; refName @1 :Text; }
struct CtlDiffRequest { refName @0 :Text; }
struct CtlCheckoutRequest { refName @0 :Text; }

enum DocFormat { toml @0; json @1; yaml @2; csv @3; text @4; }
struct EditOpenRequest { format @0 :DocFormat; }
struct EditApplyRequest { changeBytes @0 :Data; }

# ── ctl response ──

struct CtlResponse {
  union {
    error          @0 :ErrorInfo;
    status         @1 :FileStatus;
    log            @2 :List(LogEntry);
    diff           @3 :Text;
    blame          @4 :Text;
    checkout       @5 :Void;
    validate       @6 :ValidationResult;
    info           @7 :FileInfo;
    editOpen       @8 :Void;
    editState      @9 :Text;             # serialized doc (TOML/JSON/etc.)
    editApply      @10 :Void;
    editClose      @11 :Void;
    ctlFlush       @12 :Void;
  }
}

struct FileStatus { state @0 :Text; }
struct LogEntry { oid @0 :Text; message @1 :Text; author @2 :Text; timestamp @3 :UInt64; }
struct ValidationResult { valid @0 :Bool; errors @1 :List(Text); }
struct FileInfo { path @0 :Text; size @1 :UInt64; format @2 :DocFormat; editing @3 :Bool; dirty @4 :Bool; }
