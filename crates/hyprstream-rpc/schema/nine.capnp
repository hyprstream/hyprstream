@0xa1b2c3d4e5f60910;

# 9P2000-inspired filesystem protocol types.
#
# Shared by all services that expose a synthetic filesystem.
# Originally defined in registry.capnp for git worktree access,
# now factored here for reuse by model, mcp, worker, and future services.

using import "common.capnp".ErrorInfo;

# --- Identity ---

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

# --- Request types ---

struct NpWalk {
  fid @0 :UInt32;          # source fid (0 = root)
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
  perm @2 :UInt32;         # DMDIR=0x80000000 for dirs
  mode @3 :UInt8;          # open mode for the new file
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

# --- Response types ---

struct RWalk  { qid @0 :Qid; }
struct ROpen  { qid @0 :Qid; iounit @1 :UInt32; }
struct RRead  { data @0 :Data; }
struct RWrite { count @0 :UInt32; }
struct RStat  { stat @0 :NpStat; }

# --- Envelopes for scoped fs sub-requests ---
# Services import these and embed in their request/response unions.

struct NpRequest {
  union {
    walk @0 :NpWalk;
    open @1 :NpOpen;
    read @2 :NpRead;
    write @3 :NpWrite;
    clunk @4 :NpClunk;
    stat @5 :NpStatReq;
    create @6 :NpCreate;
    remove @7 :NpRemove;
    wstat @8 :NpWstat;
    flush @9 :NpFlush;
  }
}

struct NpResponse {
  union {
    error @0 :ErrorInfo;
    walk @1 :RWalk;
    open @2 :ROpen;
    read @3 :RRead;
    write @4 :RWrite;
    clunk @5 :Void;
    stat @6 :RStat;
    create @7 :ROpen;
    remove @8 :Void;
    wstat @9 :Void;
    flush @10 :Void;
  }
}
