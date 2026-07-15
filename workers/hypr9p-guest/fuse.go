// FUSE→9P bridge (epic #729, V5 = #751).
//
// This file adds a **FUSE mode** to hypr9p-guest so that arbitrary in-guest
// processes — not just this binary's own ls/cat/write client — see the host
// tenant VFS as an ordinary POSIX filesystem. When `--fuse-mount <dir>` (or
// `HYPRSTREAM_GUEST_FUSE_MOUNT`) is set, we:
//
//  1. dial the same host 9P-over-vsock channel the client mode uses
//     (`AF_VSOCK` CID 2, port 564 — see main.go `dial`),
//  2. speak 9P2000.L to the host translator via the *reused* progrium/p9
//     client, and
//  3. expose that tree as a FUSE mount, proxying every FUSE op
//     (lookup/getattr/readdir/open/read/write) to a 9P walk/getattr/readdir/
//     open/read/write.
//
// # Why FUSE and not a kernel 9P mount
//
// The kata guest kernel has v9fs but only the **virtio** 9P transport, not the
// fd transport (`trans=fd`/`rfdno`/`wfdno` are absent from `vmlinux-6.18.28-194`
// — see main.go's package doc), and cloud-hypervisor exposes no virtio-9p
// device, so `mount -t 9p` is impossible in this guest. The guest kernel *does*
// ship FUSE (`fuse`=376 in the vmlinux strings), so a userspace FUSE server that
// bridges to the vsock 9P channel is the only way to get a real POSIX
// mountpoint. (The alternative — a native-kernel 9P path via a rebuilt guest
// kernel — is tracked separately in #762.)
//
// # Library choice: hanwen/go-fuse/v2
//
// We use `github.com/hanwen/go-fuse/v2` (the `fs` inode API) rather than
// `bazil.org/fuse`: go-fuse is actively maintained, is pure-Go (no cgo, so it
// keeps the CGO_ENABLED=0 static build), and its low-level `fs.InodeEmbedder`
// interfaces map one-to-one onto 9P operations. It has no dependency overlap
// with the p9 stack, so pinning it does not disturb the progrium/p9 `replace`.
//
// # Privilege / gating
//
// Mounting FUSE needs `/dev/fuse` and CAP_SYS_ADMIN (or an unprivileged-userns
// grant) inside the guest, which a CI self-test cannot provide — so, unlike the
// client modes, the FUSE path is **operator-boot-only**. The build gate for this
// code is `go build` + the non-mounting unit tests in `fuse_test.go` (errno
// mapping + the 9P-proxy layer exercised against an in-process 9P server with no
// real mount).
package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"

	"github.com/hanwen/go-fuse/v2/fs"
	"github.com/hanwen/go-fuse/v2/fuse"
	"github.com/hugelgupf/p9/linux"
	"github.com/hugelgupf/p9/p9"
)

// runFuseMount dials the host 9P server, attaches its root, and serves it as a
// FUSE filesystem at mountpoint until the mount is unmounted or the process is
// signalled (SIGINT/SIGTERM), at which point it unmounts cleanly. It blocks for
// the lifetime of the mount.
func runFuseMount(cfg config, mountpoint string) error {
	if mountpoint == "" {
		return errors.New("fuse mount: empty mountpoint")
	}

	conn, err := dial(cfg)
	if err != nil {
		return err
	}
	// The 9P client owns the connection for the mount's lifetime; closed via
	// client.Close() in the deferred teardown below.

	client, err := p9.NewClient(conn, p9.WithMessageSize(msize))
	if err != nil {
		conn.Close()
		return fmt.Errorf("9P version handshake: %w", err)
	}

	root, err := client.AttachUname(cfg.uname, cfg.aname)
	if err != nil {
		client.Close()
		return fmt.Errorf("9P attach (uname set=%t, aname=%q): %w", cfg.uname != "", cfg.aname, err)
	}

	backend := &fuseFS{root: root}
	rootNode := &p9Node{fs: backend, path: ""}

	server, err := fs.Mount(mountpoint, rootNode, fuseMountOptions())
	if err != nil {
		root.Close()
		client.Close()
		return fmt.Errorf("fuse mount %q (needs /dev/fuse + mount privilege): %w", mountpoint, err)
	}

	// Unmount on SIGINT/SIGTERM so a supervised guest process tears the mount
	// down instead of leaving a stuck mountpoint behind.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sig
		_ = server.Unmount()
	}()

	fmt.Fprintf(os.Stderr, "hypr9p-guest: FUSE mounted tenant VFS at %s\n", mountpoint)
	server.Wait() // returns after Unmount() or an external umount(8)

	signal.Stop(sig)
	root.Close()
	client.Close()
	return nil
}

// fuseMountOptions returns the FUSE mount options. FsName/Name label the mount
// in /proc/mounts so an operator can see what it is; the timeouts are modest so
// the guest reflects host-side VFS changes reasonably promptly.
func fuseMountOptions() *fs.Options {
	opts := &fs.Options{}
	opts.MountOptions.FsName = "hypr9p"
	opts.MountOptions.Name = "9p"
	return opts
}

// fuseFS is the 9P-backed backend for the FUSE tree: it owns the attached 9P
// root fid and turns each FUSE op into a 9P walk-from-root + operation. Walking
// from the root for every op (rather than caching per-node fids) keeps fid
// lifetimes trivial and leans on the p9 client's tag multiplexing for
// concurrency. These methods return go-fuse types + `syscall.Errno` directly so
// they are unit-testable against an in-process 9P server with no real mount.
type fuseFS struct {
	root p9.File
}

// walk clones a fresh fid at path (relative to the attached root). The caller
// owns the returned file and must Close it. splitPath (main.go) normalises the
// slash-separated path.
func (b *fuseFS) walk(path string) (p9.File, error) {
	_, f, err := b.root.Walk(splitPath(path))
	return f, err
}

// attr walks to path and fills a fuse.Attr from the 9P GetAttr, mapping any 9P
// Rlerror to the matching FUSE errno.
func (b *fuseFS) attr(path string, out *fuse.Attr) syscall.Errno {
	f, err := b.walk(path)
	if err != nil {
		return errnoOf(err)
	}
	defer f.Close()
	_, _, a, err := f.GetAttr(p9.AttrMaskAll)
	if err != nil {
		return errnoOf(err)
	}
	fillAttr(out, &a)
	return 0
}

// readdir walks to path, opens it read-only, and returns its entries as
// fuse.DirEntry values (basename + type + inode), paging through 9P Treaddir.
func (b *fuseFS) readdir(path string) ([]fuse.DirEntry, syscall.Errno) {
	f, err := b.walk(path)
	if err != nil {
		return nil, errnoOf(err)
	}
	defer f.Close()
	if _, _, err := f.Open(p9.ReadOnly); err != nil {
		return nil, errnoOf(err)
	}
	var entries []fuse.DirEntry
	var offset uint64
	for {
		ents, err := f.Readdir(offset, readChunk)
		if err != nil {
			return nil, errnoOf(err)
		}
		if len(ents) == 0 {
			return entries, 0
		}
		for _, e := range ents {
			// Skip the conventional "." / ".." the kernel synthesises.
			if e.Name == "." || e.Name == ".." {
				offset = e.Offset
				continue
			}
			entries = append(entries, fuse.DirEntry{
				Name: e.Name,
				Mode: qidTypeToMode(e.Type),
				Ino:  e.QID.Path,
			})
			offset = e.Offset
		}
	}
}

// open walks to path and opens it with the FUSE open flags translated to 9P
// OpenFlags, returning a handle that proxies read/write to the open fid.
func (b *fuseFS) open(path string, flags uint32) (*p9Handle, syscall.Errno) {
	f, err := b.walk(path)
	if err != nil {
		return nil, errnoOf(err)
	}
	if _, _, err := f.Open(openFlagsToP9(flags)); err != nil {
		f.Close()
		return nil, errnoOf(err)
	}
	return &p9Handle{file: f}, 0
}

// p9Node is a FUSE inode backed by a 9P path. The root has path "". Child paths
// are built by joining the parent path with the looked-up name.
type p9Node struct {
	fs.Inode
	fs   *fuseFS
	path string
}

// Interface assertions: the ops we implement.
var (
	_ fs.InodeEmbedder = (*p9Node)(nil)
	_ fs.NodeGetattrer = (*p9Node)(nil)
	_ fs.NodeLookuper  = (*p9Node)(nil)
	_ fs.NodeReaddirer = (*p9Node)(nil)
	_ fs.NodeOpener    = (*p9Node)(nil)
)

// childPath joins this node's path with a child name.
func (n *p9Node) childPath(name string) string {
	if n.path == "" {
		return name
	}
	return n.path + "/" + name
}

// Getattr proxies to the 9P GetAttr of this node's path.
func (n *p9Node) Getattr(_ context.Context, _ fs.FileHandle, out *fuse.AttrOut) syscall.Errno {
	return n.fs.attr(n.path, &out.Attr)
}

// Lookup resolves name under this node: it 9P-walks to the child, reads its
// attributes, and returns a child inode typed from the 9P mode. A missing path
// surfaces as ENOENT (from the host's Rlerror), so `ls`/`stat` of an absent
// entry behaves like a normal filesystem.
func (n *p9Node) Lookup(ctx context.Context, name string, out *fuse.EntryOut) (*fs.Inode, syscall.Errno) {
	childPath := n.childPath(name)
	if errno := n.fs.attr(childPath, &out.Attr); errno != 0 {
		return nil, errno
	}
	child := &p9Node{fs: n.fs, path: childPath}
	stable := fs.StableAttr{
		Mode: out.Attr.Mode & syscall.S_IFMT,
		Ino:  out.Attr.Ino,
	}
	return n.NewInode(ctx, child, stable), 0
}

// Readdir lists this node's directory over 9P.
func (n *p9Node) Readdir(_ context.Context) (fs.DirStream, syscall.Errno) {
	entries, errno := n.fs.readdir(n.path)
	if errno != 0 {
		return nil, errno
	}
	return fs.NewListDirStream(entries), 0
}

// Open opens this node's file over 9P and returns a proxying handle.
func (n *p9Node) Open(_ context.Context, flags uint32) (fs.FileHandle, uint32, syscall.Errno) {
	h, errno := n.fs.open(n.path, flags)
	if errno != 0 {
		return nil, 0, errno
	}
	return h, 0, 0
}

// p9Handle is an open-file handle proxying reads/writes to an open 9P fid.
type p9Handle struct {
	file p9.File
}

var (
	_ fs.FileReader   = (*p9Handle)(nil)
	_ fs.FileWriter   = (*p9Handle)(nil)
	_ fs.FileReleaser = (*p9Handle)(nil)
	_ fs.FileFlusher  = (*p9Handle)(nil)
	_ fs.FileFsyncer  = (*p9Handle)(nil)
)

// Read proxies to 9P ReadAt. A short read or io.EOF is not an error to FUSE —
// the returned ReadResult length signals end of data.
func (h *p9Handle) Read(_ context.Context, dest []byte, off int64) (fuse.ReadResult, syscall.Errno) {
	n, err := h.file.ReadAt(dest, off)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, errnoOf(err)
	}
	return fuse.ReadResultData(dest[:n]), 0
}

// Write proxies to 9P WriteAt.
func (h *p9Handle) Write(_ context.Context, data []byte, off int64) (uint32, syscall.Errno) {
	n, err := h.file.WriteAt(data, off)
	if err != nil {
		return uint32(n), errnoOf(err)
	}
	return uint32(n), 0
}

// Flush is a no-op for the byte-stream proxy (there is no client-side buffer to
// drain); returning 0 keeps close(2) from failing.
func (h *p9Handle) Flush(_ context.Context) syscall.Errno {
	return 0
}

// Fsync proxies to 9P FSync.
func (h *p9Handle) Fsync(_ context.Context, _ uint32) syscall.Errno {
	if err := h.file.FSync(); err != nil {
		return errnoOf(err)
	}
	return 0
}

// Release closes the underlying 9P fid.
func (h *p9Handle) Release(_ context.Context) syscall.Errno {
	if err := h.file.Close(); err != nil {
		return errnoOf(err)
	}
	return 0
}

// errnoOf maps a 9P client error to the FUSE syscall.Errno. The progrim/p9
// client returns server-side Rlerror as a `linux.Errno` (Linux errno numbers,
// identical to this linux/amd64 guest's `syscall.Errno`), and maps common Go
// sentinel errors (os.ErrNotExist → ENOENT, etc.) via linux.ExtractErrno. An
// unrecognised error becomes EIO. This is the piece the non-mounting unit test
// pins so `ls` of a missing path yields ENOENT rather than a generic failure.
func errnoOf(err error) syscall.Errno {
	if err == nil {
		return 0
	}
	// A syscall.Errno (e.g. from the local transport) passes straight through.
	var se syscall.Errno
	if errors.As(err, &se) {
		return se
	}
	return syscall.Errno(linux.ExtractErrno(err))
}

// fillAttr copies a 9P Attr into a fuse.Attr. The 9P2000.L mode already carries
// the full Linux mode bits (S_IFMT type + permissions), so it maps directly.
func fillAttr(out *fuse.Attr, a *p9.Attr) {
	out.Mode = uint32(a.Mode)
	out.Size = a.Size
	out.Blocks = a.Blocks
	out.Blksize = uint32(a.BlockSize)
	out.Nlink = uint32(a.NLink)
	if out.Nlink == 0 {
		out.Nlink = 1
	}
	out.Owner.Uid = uint32(a.UID)
	out.Owner.Gid = uint32(a.GID)
	out.Rdev = uint32(a.RDev)
	out.Atime = a.ATimeSeconds
	out.Atimensec = uint32(a.ATimeNanoSeconds)
	out.Mtime = a.MTimeSeconds
	out.Mtimensec = uint32(a.MTimeNanoSeconds)
	out.Ctime = a.CTimeSeconds
	out.Ctimensec = uint32(a.CTimeNanoSeconds)
}

// qidTypeToMode maps a 9P QID type to the S_IFMT bits FUSE wants in a DirEntry
// (only the high bits are consulted there).
func qidTypeToMode(t p9.QIDType) uint32 {
	switch t {
	case p9.TypeDir:
		return syscall.S_IFDIR
	case p9.TypeSymlink:
		return syscall.S_IFLNK
	default:
		return syscall.S_IFREG
	}
}

// openFlagsToP9 translates kernel open(2) flags to 9P OpenFlags. Only the access
// mode plus O_TRUNC are meaningful to the host translator (which has Tlopen but
// no Tlcreate); other flags are ignored.
func openFlagsToP9(flags uint32) p9.OpenFlags {
	var mode p9.OpenFlags
	switch flags & syscall.O_ACCMODE {
	case syscall.O_WRONLY:
		mode = p9.WriteOnly
	case syscall.O_RDWR:
		mode = p9.ReadWrite
	default:
		mode = p9.ReadOnly
	}
	if flags&syscall.O_TRUNC != 0 {
		mode |= p9.Trunc
	}
	return mode
}
