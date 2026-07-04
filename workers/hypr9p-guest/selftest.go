package main

import (
	"bytes"
	"fmt"
	"net"
	"os"
	"path/filepath"

	"github.com/hugelgupf/p9/fsimpl/localfs"
	"github.com/hugelgupf/p9/p9"
)

// runSelfTest stands up an in-process 9P2000.L server (localfs over a temp dir)
// on a Unix socket, then drives this binary's own ls/cat/write operations
// against it — exercising the full attach → walk → open → readdir → read → write
// path with no vsock, no VM, and no external server. This is the CI smoke test
// (mirrors workers/wanix-guest's --self-test) and requires no privileges.
func runSelfTest() error {
	dir, err := os.MkdirTemp("", "hypr9p-guest-selftest-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)

	// A hyprstream-ish tenant tree: /models/hello + /models/llama3.
	if err := os.MkdirAll(filepath.Join(dir, "models"), 0o755); err != nil {
		return err
	}
	const helloBody = "hello from the tenant VFS\n"
	if err := os.WriteFile(filepath.Join(dir, "models", "hello"), []byte(helloBody), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "models", "llama3"), []byte("stub\n"), 0o644); err != nil {
		return err
	}

	sockPath := filepath.Join(dir, "9p.sock")
	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		return fmt.Errorf("listen unix: %w", err)
	}
	defer ln.Close()

	srv := p9.NewServer(localfs.Attacher(dir))
	go func() { _ = srv.Serve(ln) }() // returns when ln closes

	cfg := config{sock: sockPath}
	conn, err := dial(cfg)
	if err != nil {
		return err
	}
	defer conn.Close()

	client, err := p9.NewClient(conn, p9.WithMessageSize(msize))
	if err != nil {
		return fmt.Errorf("client handshake: %w", err)
	}
	defer client.Close()

	root, err := client.Attach("")
	if err != nil {
		return fmt.Errorf("attach: %w", err)
	}
	defer root.Close()

	// readdir "/" must list "models".
	var lsRoot bytes.Buffer
	if err := doLs(root, "/", &lsRoot); err != nil {
		return fmt.Errorf("ls /: %w", err)
	}
	if !hasLine(lsRoot.String(), "models/") {
		return fmt.Errorf("ls / missing 'models/'; got:\n%s", lsRoot.String())
	}

	// readdir "/models" must list "hello".
	var lsModels bytes.Buffer
	if err := doLs(root, "/models", &lsModels); err != nil {
		return fmt.Errorf("ls /models: %w", err)
	}
	if !hasLine(lsModels.String(), "hello") {
		return fmt.Errorf("ls /models missing 'hello'; got:\n%s", lsModels.String())
	}

	// cat "/models/hello" must return its body.
	var catBuf bytes.Buffer
	if err := doCat(root, "/models/hello", &catBuf); err != nil {
		return fmt.Errorf("cat /models/hello: %w", err)
	}
	if catBuf.String() != helloBody {
		return fmt.Errorf("cat /models/hello = %q, want %q", catBuf.String(), helloBody)
	}

	// write round-trip: overwrite /models/hello and read it back.
	const newBody = "rewritten\n"
	var wrBuf bytes.Buffer
	if err := doWrite(root, "/models/hello", []byte(newBody), &wrBuf); err != nil {
		return fmt.Errorf("write /models/hello: %w", err)
	}
	if wrBuf.String() != newBody {
		return fmt.Errorf("write round-trip read-back = %q, want %q", wrBuf.String(), newBody)
	}

	fmt.Println("self-test OK: attach + readdir + read + write round-trip over 9P2000.L")
	return nil
}

func hasLine(out, want string) bool {
	for _, l := range bytes.Split([]byte(out), []byte("\n")) {
		if string(l) == want {
			return true
		}
	}
	return false
}
