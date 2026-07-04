# hypr9p-guest

A thin **native Go** in-guest client that operates the hyprstream tenant VFS,
served as **9P2000.L over vsock** by a kata microVM host, from **userspace** —
no kernel v9fs mount, no Rust shim.

Part of epic **#729** (V3 = **#732**). Like `workers/wanix-guest`, it is a
foreign-toolchain (Go) guest artifact and lives **outside** the Cargo workspace
on purpose: it is not a crate and is never built by `cargo`.

## The design fork, resolved: userspace client (not kernel `trans=fd`)

The host serves each sandbox's Subject-scoped tenant VFS as native 9P on a
per-sandbox vsock channel, `VFS_9P_VSOCK_PORT = 564`
(`KataBackend::serve_tenant_vfs_9p` → `hyprstream_9p::serve_mount_vsock_raw`,
#731/#741). Two guest-side mount mechanisms were on the table:

- **(A) kernel v9fs `trans=fd`** — open an `AF_VSOCK` fd, then
  `mount -t 9p -o trans=fd,rfdno=<fd>,wfdno=<fd>,version=9p2000.L`. Needs the
  9pnet **fd transport** (`CONFIG_NET_9P_FD` / `net/9p/trans_fd.c`).
- **(B) a userspace 9P client** that speaks 9P2000.L directly over the vsock
  stream and does walk/read/write/readdir itself.

**Static inspection of the shipped guest kernel decides it: (A) is impossible on
this kernel.** `vmlinux-6.18.28-194` (kata 3.31.0, ubuntu-noble) is an
uncompressed, non-stripped ELF; it contains **zero** occurrences of `9pnet_fd`,
`trans_fd`, `rfdno`, or `wfdno`, while `9pnet_virtio` **is** present. v9fs is
built with the **virtio** transport only — and cloud-hypervisor exposes no
virtio-9p device (only virtio-fs). So there is **no kernel-mountable 9P
transport in this guest at all**, and `trans=fd` is out. This binary is (B).

A POSIX-visible mountpoint (so a bare `cat <mnt>/file` works) would need a
FUSE→9P bridge. The guest kernel **does** ship FUSE (`fuse`/`virtio_fs` strings
present), so that is a viable **follow-up (V5)** — deliberately not done here
because it drags in a FUSE dependency, `/dev/fuse`, and mount privileges that
make a CI self-test impossible. This client validates exactly the epic's
acceptance surface: attach + walk + **readdir + read + write** over vsock 9P.

## What it does

```
hypr9p-guest [--vsock-cid 2] [--vsock-port 564] [--aname ""] <op> <args...>
hypr9p-guest --sock /path/to/9p.sock <op> <args...>     # UDS instead of vsock
hypr9p-guest --self-test
```

Operations:

| op | 9P | meaning |
|----|-----|---------|
| `ls <path>` | Twalk→Tlopen→Treaddir | list a directory (one basename/line, dirs suffixed `/`) |
| `cat <path>` | Twalk→Tlopen→Tread | stream a file to stdout |
| `write <path> <data>` | Twalk→Tlopen(WRONLY\|TRUNC)→Twrite, then read-back | overwrite an **existing** file and echo it back |

The guest dials `AF_VSOCK` to `(CID = VMADDR_CID_HOST = 2, port = 564)`. Under CH
hybrid-vsock a guest-initiated connect to port *N* is routed to the host UDS
`<vsock-base>_N`, and hyprstream's `serve_mount_vsock_raw` treats the first byte
as 9P with **no `connect <port>\n` preamble** (RAW mode, #741). No third-party
vsock library is used: the `p9` client needs only an `io.ReadWriteCloser`, so
the raw `AF_VSOCK` fd is wrapped in an `*os.File`.

`write` uses `Tlopen` — **not** `Tlcreate`, which the host translator does not
implement — so the target file must already exist in the tenant VFS.

## Self-test (CI smoke)

`--self-test` stands up an in-process 9P2000.L server (`fsimpl/localfs` over a
temp dir with `/models/hello`), then runs this binary's own `ls`/`cat`/`write`
against it — the full attach/walk/open/readdir/read/write path, no vsock, no VM,
no privileges. This is what the standalone CI job runs.

## Building

Not built by cargo. Static, CGO-disabled binary:

```
scripts/build-hypr9p-guest.sh [output-path]     # default: target/hypr9p-guest
```

or directly:

```
cd workers/hypr9p-guest
CGO_ENABLED=0 go build -o hypr9p-guest .
go vet ./...
go run . --self-test
```

CI builds/vets/self-tests it via `.github/workflows/hypr9p-guest.yml`
(standalone Go job, independent of the Rust workflows).

## Dependency pinning

- **`p9`** — pinned to the **progrium fork** via `replace`, in lockstep with
  `workers/wanix-guest/go.mod`. This is the same 9P implementation the
  hyprstream host translator is proven interoperable with (#506), which is why
  wire compatibility is not re-litigated here.
- **`golang.org/x/sys`** — for `AF_VSOCK` / `SockaddrVM` (no CGO).

## Staging into the guest

The operator stages the static binary into the kata guest rootfs (e.g.
`/usr/local/bin/hypr9p-guest`) or delivers it via the tenant VFS, then runs it
with `KataBackend::exec_sync`. The e2e harness
(`crates/hyprstream-workers/tests/kata_9p_vsock_e2e.rs`) takes the in-guest path
from `HYPRSTREAM_GUEST_9P_HELPER`.
