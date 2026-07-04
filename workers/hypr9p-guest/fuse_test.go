package main

import (
	"net"
	"os"
	"path/filepath"
	"syscall"
	"testing"

	"github.com/hanwen/go-fuse/v2/fuse"
	"github.com/hugelgupf/p9/fsimpl/localfs"
	"github.com/hugelgupf/p9/linux"
	"github.com/hugelgupf/p9/p9"
)

// These tests validate the FUSE→9P bridge WITHOUT mounting anything: mounting
// FUSE needs /dev/fuse + CAP_SYS_ADMIN, which CI cannot grant. Instead we pin
// (1) the pure errno-mapping function and (2) the `fuseFS` 9P-proxy layer
// exercised against an in-process localfs 9P server — the same server the
// --self-test uses — so the code that turns FUSE ops into 9P ops is covered
// without a kernel mount.

// TestErrnoOf pins the 9P-error → FUSE-errno mapping. `ls` of a missing path
// must surface ENOENT (not a generic EIO), which is the load-bearing case.
func TestErrnoOf(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want syscall.Errno
	}{
		{"nil", nil, 0},
		{"9p ENOENT", linux.ENOENT, syscall.ENOENT},
		{"9p EACCES", linux.EACCES, syscall.EACCES},
		{"9p EISDIR", linux.EISDIR, syscall.EISDIR},
		{"9p ENOTDIR", linux.ENOTDIR, syscall.ENOTDIR},
		{"9p EEXIST", linux.EEXIST, syscall.EEXIST},
		{"9p EPERM", linux.EPERM, syscall.EPERM},
		{"syscall passthrough", syscall.ENOSPC, syscall.ENOSPC},
		{"go sentinel ErrNotExist", os.ErrNotExist, syscall.ENOENT},
		{"unrecognised → EIO", errString("boom"), syscall.EIO},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := errnoOf(tc.err); got != tc.want {
				t.Fatalf("errnoOf(%v) = %d (%v), want %d (%v)", tc.err, got, got, tc.want, tc.want)
			}
		})
	}
}

type errString string

func (e errString) Error() string { return string(e) }

// newTestBackend stands up an in-process 9P2000.L server (localfs over a temp
// dir with /models/hello) on a Unix socket and returns a `fuseFS` attached to
// it — i.e. a fully-constructed FUSE backend with NO real mount. Cleanup closes
// everything.
func newTestBackend(t *testing.T) *fuseFS {
	t.Helper()
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "models"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "models", "hello"), []byte(helloBody), 0o644); err != nil {
		t.Fatal(err)
	}

	sockPath := filepath.Join(dir, "9p.sock")
	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		t.Fatal(err)
	}
	srv := p9.NewServer(localfs.Attacher(dir))
	go func() { _ = srv.Serve(ln) }()

	conn, err := net.Dial("unix", sockPath)
	if err != nil {
		t.Fatal(err)
	}
	client, err := p9.NewClient(conn, p9.WithMessageSize(msize))
	if err != nil {
		t.Fatal(err)
	}
	root, err := client.Attach("")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = root.Close()
		_ = client.Close()
		_ = ln.Close()
	})
	return &fuseFS{root: root}
}

const helloBody = "hello from the tenant VFS\n"

// TestBackendAttr checks getattr proxying: the root is a directory, and
// /models/hello is a regular file of the expected size.
func TestBackendAttr(t *testing.T) {
	b := newTestBackend(t)

	var rootAttr fuse.Attr
	if errno := b.attr("", &rootAttr); errno != 0 {
		t.Fatalf("attr(root) errno %v", errno)
	}
	if rootAttr.Mode&syscall.S_IFMT != syscall.S_IFDIR {
		t.Fatalf("root mode %o not a directory", rootAttr.Mode)
	}

	var fileAttr fuse.Attr
	if errno := b.attr("models/hello", &fileAttr); errno != 0 {
		t.Fatalf("attr(models/hello) errno %v", errno)
	}
	if fileAttr.Mode&syscall.S_IFMT != syscall.S_IFREG {
		t.Fatalf("models/hello mode %o not a regular file", fileAttr.Mode)
	}
	if fileAttr.Size != uint64(len(helloBody)) {
		t.Fatalf("models/hello size = %d, want %d", fileAttr.Size, len(helloBody))
	}
}

// TestBackendAttrMissing is the ENOENT case that makes `ls`/`stat` of a missing
// path behave like a normal filesystem.
func TestBackendAttrMissing(t *testing.T) {
	b := newTestBackend(t)
	var a fuse.Attr
	if errno := b.attr("models/does-not-exist", &a); errno != syscall.ENOENT {
		t.Fatalf("attr(missing) = %v, want ENOENT", errno)
	}
}

// TestBackendReaddir checks directory listing proxying and entry typing.
func TestBackendReaddir(t *testing.T) {
	b := newTestBackend(t)

	rootEnts, errno := b.readdir("")
	if errno != 0 {
		t.Fatalf("readdir(root) errno %v", errno)
	}
	if !hasEntry(rootEnts, "models", syscall.S_IFDIR) {
		t.Fatalf("readdir(root) missing dir 'models'; got %+v", rootEnts)
	}

	modelEnts, errno := b.readdir("models")
	if errno != 0 {
		t.Fatalf("readdir(models) errno %v", errno)
	}
	if !hasEntry(modelEnts, "hello", syscall.S_IFREG) {
		t.Fatalf("readdir(models) missing file 'hello'; got %+v", modelEnts)
	}
}

func hasEntry(ents []fuse.DirEntry, name string, ifmt uint32) bool {
	for _, e := range ents {
		if e.Name == name && e.Mode&syscall.S_IFMT == ifmt {
			return true
		}
	}
	return false
}

// TestBackendReadWrite checks the open→read and open→write proxies round-trip,
// mirroring the client-mode cat/write path but through the FUSE backend layer.
func TestBackendReadWrite(t *testing.T) {
	b := newTestBackend(t)

	// Read /models/hello via the FUSE backend.
	rh, errno := b.open("models/hello", syscall.O_RDONLY)
	if errno != 0 {
		t.Fatalf("open(read) errno %v", errno)
	}
	buf := make([]byte, 64)
	res, errno := rh.Read(nil, buf, 0)
	if errno != 0 {
		t.Fatalf("read errno %v", errno)
	}
	got, status := res.Bytes(make([]byte, 64))
	if status != fuse.OK {
		t.Fatalf("read result status %v", status)
	}
	if string(got) != helloBody {
		t.Fatalf("read = %q, want %q", got, helloBody)
	}
	if errno := rh.Release(nil); errno != 0 {
		t.Fatalf("release errno %v", errno)
	}

	// Overwrite it write-only (localfs supports Tlopen+Twrite) and read back.
	const newBody = "rewritten\n"
	wh, errno := b.open("models/hello", syscall.O_WRONLY|syscall.O_TRUNC)
	if errno != 0 {
		t.Fatalf("open(write) errno %v", errno)
	}
	n, errno := wh.Write(nil, []byte(newBody), 0)
	if errno != 0 {
		t.Fatalf("write errno %v", errno)
	}
	if int(n) != len(newBody) {
		t.Fatalf("short write: %d of %d", n, len(newBody))
	}
	if errno := wh.Release(nil); errno != 0 {
		t.Fatalf("release(write) errno %v", errno)
	}

	rh2, errno := b.open("models/hello", syscall.O_RDONLY)
	if errno != 0 {
		t.Fatalf("reopen errno %v", errno)
	}
	defer rh2.Release(nil)
	res2, errno := rh2.Read(nil, make([]byte, 64), 0)
	if errno != 0 {
		t.Fatalf("reread errno %v", errno)
	}
	got2, _ := res2.Bytes(make([]byte, 64))
	if string(got2) != newBody {
		t.Fatalf("read-back = %q, want %q", got2, newBody)
	}
}

// TestOpenFlagsToP9 pins the open(2) → 9P OpenFlags translation.
func TestOpenFlagsToP9(t *testing.T) {
	cases := []struct {
		flags uint32
		want  p9.OpenFlags
	}{
		{syscall.O_RDONLY, p9.ReadOnly},
		{syscall.O_WRONLY, p9.WriteOnly},
		{syscall.O_RDWR, p9.ReadWrite},
		{syscall.O_WRONLY | syscall.O_TRUNC, p9.WriteOnly | p9.Trunc},
	}
	for _, tc := range cases {
		if got := openFlagsToP9(tc.flags); got != tc.want {
			t.Fatalf("openFlagsToP9(%#o) = %v, want %v", tc.flags, got, tc.want)
		}
	}
}
