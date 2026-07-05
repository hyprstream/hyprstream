// Command hypr9p-guest is a thin *native* in-guest 9P client for the hyprstream
// tenant VFS served over vsock (epic #729, V3 = #732).
//
// # Why a userspace client (design fork resolved)
//
// The kata guest is expected to consume the host tenant VFS, which hyprstream
// serves as native **9P2000.L over a per-sandbox vsock channel** on
// `VFS_9P_VSOCK_PORT = 564` (host side: `KataBackend::serve_tenant_vfs_9p` →
// `hyprstream_9p::serve_mount_vsock_raw`, #731/#741). Two guest-side mount
// mechanisms were considered:
//
//   - (A) kernel v9fs `trans=fd`: open an AF_VSOCK fd, then
//     `mount -t 9p -o trans=fd,rfdno=<fd>,wfdno=<fd>,version=9p2000.L`. This
//     needs the kernel's 9pnet **fd transport** (`CONFIG_NET_9P_FD`,
//     `net/9p/trans_fd.c`).
//   - (B) a userspace 9P client that speaks 9P2000.L directly over the vsock
//     stream and performs walk / read / write / readdir itself.
//
// Static inspection of the kata 3.31.0 guest kernel
// (`vmlinux-6.18.28-194`, an uncompressed, non-stripped ELF) shows the fd
// transport is **NOT built**: zero occurrences of any of `9pnet_fd`, `trans_fd`,
// `rfdno`, `wfdno`, while `9pnet_virtio` IS present. v9fs is compiled with the
// **virtio** transport only — and cloud-hypervisor exposes no virtio-9p device
// (only virtio-fs). So there is no kernel-mountable 9P transport in this guest
// at all, and option (A) is out. This binary is option (B): it dials the host
// over AF_VSOCK and operates the 9P tree from userspace, needing nothing from
// the guest kernel beyond AF_VSOCK (already present — the kata-agent uses it).
//
// A POSIX-visible mountpoint (so a bare `cat <mnt>/file` works — for arbitrary
// in-guest processes, not just this client) needs a FUSE→9P bridge; the guest
// kernel *does* ship FUSE (`fuse`=376 in the vmlinux strings). That bridge is
// the **`--fuse-mount` mode** added in V5 (#751, see fuse.go): it reuses this
// binary's 9P client as the backend and exposes the tree via
// github.com/hanwen/go-fuse/v2. Because mounting FUSE needs `/dev/fuse` +
// CAP_SYS_ADMIN, that mode is **operator-boot-only** — a CI self-test cannot
// mount — so the client modes below keep the CI-safe `--self-test`, and the
// FUSE bridge's own gate is `go build` + the non-mounting unit tests in
// fuse_test.go. (A native-kernel 9P mount path is tracked separately in #762.)
//
// # Run modes
//
//	hypr9p-guest [--vsock-cid 2] [--vsock-port 564] [--aname ""] <op> <args...>
//	hypr9p-guest --sock /path/to/9p.sock <op> <args...>     # UDS instead of vsock
//	hypr9p-guest --fuse-mount /mnt/vfs                       # POSIX FUSE mount (operator-only)
//	hypr9p-guest --self-test
//
// Operations:
//
//	ls   <path>            readdir <path>, one basename per line (dirs suffixed /)
//	cat  <path>            read <path> in full, write bytes to stdout
//	write <path> <data>    open <path> write-only, write <data>@0, read it back
//	                       (the host translator has no Tlcreate, so <path> must
//	                       already exist in the tenant VFS)
//
// The guest dials AF_VSOCK to (CID = VMADDR_CID_HOST = 2, port = 564). Under CH
// hybrid-vsock a guest-initiated connect to port N is routed to the host UDS
// `<vsock-base>_N`; hyprstream's `serve_mount_vsock_raw` treats the first byte
// as 9P **with no `connect <port>\n` preamble** (RAW mode, #741). That raw
// framing is exactly what the first real boot confirms.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path"
	"strings"

	"github.com/hugelgupf/p9/p9"
	"golang.org/x/sys/unix"
)

// msize matches the host translator's negotiated ceiling
// (`hyprstream_9p::translator::MSG_SIZE = 8 * 1024`); the server clamps anything
// larger, but requesting the same value avoids a needless re-clamp.
const msize = 8 * 1024

// readChunk bounds a single Tread payload, leaving headroom under msize for the
// 9P Rread framing.
const readChunk = 4 * 1024

type config struct {
	vsockCID  uint
	vsockPort uint
	sock      string
	aname     string
	selfTest  bool
	fuseMount string
}

func main() {
	cfg := config{}
	flag.UintVar(&cfg.vsockCID, "vsock-cid", uint(unix.VMADDR_CID_HOST),
		"vsock context id of the host (default 2 = VMADDR_CID_HOST)")
	flag.UintVar(&cfg.vsockPort, "vsock-port", 564,
		"vsock port the host tenant-VFS 9P server listens on (VFS_9P_VSOCK_PORT)")
	flag.StringVar(&cfg.sock, "sock", os.Getenv("HYPRSTREAM_9P_SOCK"),
		"dial this Unix socket instead of vsock (env: HYPRSTREAM_9P_SOCK); for host-side debugging/self-test")
	flag.StringVar(&cfg.aname, "aname", "", "9P attach name (empty = default tree)")
	flag.BoolVar(&cfg.selfTest, "self-test", false,
		"run ls/cat/write against a throwaway in-process 9P server and exit")
	flag.StringVar(&cfg.fuseMount, "fuse-mount", os.Getenv("HYPRSTREAM_GUEST_FUSE_MOUNT"),
		"mount the tenant VFS as a POSIX FUSE filesystem at this path (env: HYPRSTREAM_GUEST_FUSE_MOUNT); "+
			"needs /dev/fuse + mount privilege, so it is operator-boot-only, not for the CI self-test")
	flag.Parse()

	if err := run(cfg, flag.Args()); err != nil {
		fmt.Fprintf(os.Stderr, "hypr9p-guest: %v\n", err)
		os.Exit(1)
	}
}

func run(cfg config, args []string) error {
	if cfg.selfTest {
		return runSelfTest()
	}
	if cfg.fuseMount != "" {
		// FUSE mode owns its own dial/attach lifecycle (a long-lived mount, not
		// a one-shot op), so it does not fall through to the client dispatch.
		return runFuseMount(cfg, cfg.fuseMount)
	}
	if len(args) < 1 {
		return errors.New("usage: hypr9p-guest [flags] <ls|cat|write> <path> [data]  (or --self-test)")
	}

	conn, err := dial(cfg)
	if err != nil {
		return err
	}
	defer conn.Close()

	client, err := p9.NewClient(conn, p9.WithMessageSize(msize))
	if err != nil {
		return fmt.Errorf("9P version handshake: %w", err)
	}
	defer client.Close()

	root, err := client.Attach(cfg.aname)
	if err != nil {
		return fmt.Errorf("9P attach (aname=%q): %w", cfg.aname, err)
	}
	defer root.Close()

	return dispatch(root, args)
}

// dispatch runs one operation against an attached 9P root.
func dispatch(root p9.File, args []string) error {
	op := args[0]
	switch op {
	case "ls":
		if len(args) != 2 {
			return errors.New("usage: ls <path>")
		}
		return doLs(root, args[1], os.Stdout)
	case "cat":
		if len(args) != 2 {
			return errors.New("usage: cat <path>")
		}
		return doCat(root, args[1], os.Stdout)
	case "write":
		if len(args) != 3 {
			return errors.New("usage: write <path> <data>")
		}
		return doWrite(root, args[1], []byte(args[2]), os.Stdout)
	default:
		return fmt.Errorf("unknown op %q (want ls|cat|write)", op)
	}
}

// dial opens the byte stream to the host 9P server: a Unix socket when --sock is
// set, otherwise AF_VSOCK to (cfg.vsockCID, cfg.vsockPort).
func dial(cfg config) (io.ReadWriteCloser, error) {
	if cfg.sock != "" {
		conn, err := net.Dial("unix", cfg.sock)
		if err != nil {
			return nil, fmt.Errorf("dial unix %q: %w", cfg.sock, err)
		}
		return conn, nil
	}
	return dialVsock(uint32(cfg.vsockCID), uint32(cfg.vsockPort))
}

// dialVsock opens a connected AF_VSOCK stream socket and wraps it in an
// *os.File (an io.ReadWriteCloser the Go runtime poller drives). No third-party
// vsock library is needed: the p9 client only requires an io.ReadWriteCloser.
func dialVsock(cid, port uint32) (io.ReadWriteCloser, error) {
	fd, err := unix.Socket(unix.AF_VSOCK, unix.SOCK_STREAM, 0)
	if err != nil {
		return nil, fmt.Errorf("socket(AF_VSOCK): %w", err)
	}
	if err := unix.Connect(fd, &unix.SockaddrVM{CID: cid, Port: port}); err != nil {
		unix.Close(fd)
		return nil, fmt.Errorf("connect vsock(cid=%d, port=%d): %w", cid, port, err)
	}
	// os.NewFile switches the fd to non-blocking and registers it with the
	// runtime network poller, so Read/Write behave like any other socket.
	return os.NewFile(uintptr(fd), fmt.Sprintf("vsock:%d:%d", cid, port)), nil
}

// walkTo walks from an attached root to the file at path (slash-separated,
// leading/trailing slashes ignored). An empty path clones the root fid.
func walkTo(root p9.File, p string) (p9.File, error) {
	names := splitPath(p)
	_, f, err := root.Walk(names)
	if err != nil {
		return nil, fmt.Errorf("walk %q: %w", p, err)
	}
	return f, nil
}

func splitPath(p string) []string {
	out := make([]string, 0, 4)
	for _, seg := range strings.Split(path.Clean("/"+p), "/") {
		if seg != "" && seg != "." {
			out = append(out, seg)
		}
	}
	return out
}

// doLs opens dirPath and prints each entry basename, one per line, dirs suffixed
// with '/'. Standard 9P2000.L requires the fid be opened before Treaddir.
func doLs(root p9.File, dirPath string, w io.Writer) error {
	f, err := walkTo(root, dirPath)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, _, err := f.Open(p9.ReadOnly); err != nil {
		return fmt.Errorf("open dir %q: %w", dirPath, err)
	}
	var offset uint64
	for {
		ents, err := f.Readdir(offset, readChunk)
		if err != nil {
			return fmt.Errorf("readdir %q: %w", dirPath, err)
		}
		if len(ents) == 0 {
			return nil
		}
		for _, e := range ents {
			name := e.Name
			if e.Type == p9.TypeDir {
				name += "/"
			}
			if _, err := fmt.Fprintln(w, name); err != nil {
				return err
			}
			offset = e.Offset
		}
	}
}

// doCat opens filePath read-only and streams its full contents to w.
func doCat(root p9.File, filePath string, w io.Writer) error {
	f, err := walkTo(root, filePath)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, _, err := f.Open(p9.ReadOnly); err != nil {
		return fmt.Errorf("open %q: %w", filePath, err)
	}
	buf := make([]byte, readChunk)
	var offset int64
	for {
		n, err := f.ReadAt(buf, offset)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return werr
			}
			offset += int64(n)
		}
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("read %q at %d: %w", filePath, offset, err)
		}
		if n == 0 {
			return nil
		}
	}
}

// doWrite opens an existing filePath write-only, writes data at offset 0, then
// reopens it read-only and echoes the read-back bytes to w — a write/read
// round-trip. The host translator supports Tlopen+Twrite but has no Tlcreate,
// so filePath must already exist in the tenant VFS.
func doWrite(root p9.File, filePath string, data []byte, w io.Writer) error {
	wf, err := walkTo(root, filePath)
	if err != nil {
		return err
	}
	// WriteOnly|Trunc so the overwrite replaces the file rather than patching
	// its first bytes (9P Twrite has no implicit truncation).
	if _, _, err := wf.Open(p9.WriteOnly | p9.Trunc); err != nil {
		wf.Close()
		return fmt.Errorf("open %q write-only (host has no Tlcreate; file must exist): %w", filePath, err)
	}
	n, err := wf.WriteAt(data, 0)
	wf.Close()
	if err != nil {
		return fmt.Errorf("write %q: %w", filePath, err)
	}
	if n != len(data) {
		return fmt.Errorf("short write to %q: wrote %d of %d bytes", filePath, n, len(data))
	}
	return doCat(root, filePath, w)
}
