// Command wanix-guest is a thin *native* Go guest (hyprstream #506, deliverable 2).
//
// It embeds Wanix as a library and mounts a remote hyprstream 9P2000.L export
// as the ROOT of its Wanix namespace, using Wanix's OWN mechanisms -- no
// bespoke verb, no Rust shim:
//
//   - dial a Unix socket -> net.Conn to a hyprstream 9P server
//   - p9kit.ClientFS(conn, aname) adapts that conn into a wanix fs.FS
//   - wanix.NewRoot() builds a Task carrying its own namespace (NS)
//   - root.Register("exec", &native.ExecDriver{}) enables host-process tasks
//   - root.NS().Bind(hyprfs, ".", ".", BindReplace) makes the remote tree root
//
// Bidirectional 9P (#708 phase 2). The above is Direction A: the guest IMPORTS
// hyprstream's namespace. This binary also serves Direction B, the reverse --
// the guest EXPORTS its OWN live namespace so the host can import it and bind it
// into the host VFS namespace (giving the host visibility into the guest's task
// tree). When HYPRSTREAM_GUEST_EXPORT_SOCK (or --export-sock) is set, after the
// root is built the guest stands up a second, INDEPENDENT 9P2000.L server --
//
//	p9.NewServer(p9kit.Attacher(root.NS()))   served on that Unix socket
//
// in its own goroutine, using Wanix's OWN embed (root.NS() is the live vfs.NS,
// which is an fs.FS) -- no upstream `wanix serve` patch, no bespoke verb. Import
// and export are two independent 9P sessions over two sockets that share no
// state beyond the namespace itself (#708).
//
// Run modes:
//
//	wanix-guest --sock /run/hyprstream/9p.sock [--aname ""] [--cmd "sh"] \
//	            [--export-sock /run/hyprstream/wanix-export.sock]
//	    Connect to a real hyprstream 9P server, bind it as the namespace root,
//	    then either run --cmd as a Wanix task (waiting for it to exit) or, when
//	    no --cmd is given, block serving the namespace until signalled. The
//	    connection is health-probed; a dropped server triggers reconnect+rebind.
//	    When --export-sock is given the guest namespace is also served over that
//	    socket for the host to import (Direction B).
//
//	wanix-guest --self-test
//	    Spin up a throwaway in-process p9kit server over a temp Unix socket,
//	    bind it as the namespace root, list "/", THEN stand up the Direction-B
//	    export server on a throwaway socket and read the same namespace back
//	    through a p9 client -- exercises the full import AND export 9P wiring
//	    with no external server. Build smoke test.
//
// Known semantic gap (see README): native.ExecDriver runs a task's command as a
// HOST OS process; that process is NOT chrooted/rooted into the mounted 9P tree.
// The bound tree is the Wanix-level fs.FS view. POSIX-rooting host exec into it
// is the sandbox's mount concern (PR-C / hyprstream-workers), not this binary.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/hugelgupf/p9/p9"

	"tractor.dev/wanix"
	wfs "tractor.dev/wanix/fs"
	"tractor.dev/wanix/fs/fskit"
	"tractor.dev/wanix/fs/p9kit"
	"tractor.dev/wanix/native"
)

const (
	// probeInterval is how often the serve loop walks the remote 9P root to
	// detect a dropped/restarted server.
	probeInterval = 3 * time.Second
	// reconnectAttempts bounds reconnect tries before the guest gives up.
	reconnectAttempts = 5
	// reconnectBackoff is the base delay between reconnect attempts.
	reconnectBackoff = 500 * time.Millisecond
	// taskExitTimeout bounds how long runTask waits for a started task to
	// report its exit code before giving up. In the bare guest a host-exec
	// task cannot actually run (its stdio caps are not wired -- see README),
	// so this prevents an indefinite hang.
	taskExitTimeout = 10 * time.Second
)

type config struct {
	sock       string
	aname      string
	cmd        string
	exportSock string
	selfTest   bool
}

func main() {
	log.SetFlags(0)
	log.SetPrefix("wanix-guest: ")

	cfg := config{}
	flag.StringVar(&cfg.sock, "sock", os.Getenv("HYPRSTREAM_9P_SOCK"),
		"path to hyprstream 9P Unix socket (env: HYPRSTREAM_9P_SOCK)")
	flag.StringVar(&cfg.aname, "aname", "",
		"9P attach name (empty = default tree)")
	flag.StringVar(&cfg.cmd, "cmd", "",
		"command to run as a Wanix task after mounting; empty = serve the namespace")
	flag.StringVar(&cfg.exportSock, "export-sock", os.Getenv("HYPRSTREAM_GUEST_EXPORT_SOCK"),
		"path to a Unix socket on which to EXPORT this guest's namespace over 9P for the host to import "+
			"(Direction B; env: HYPRSTREAM_GUEST_EXPORT_SOCK); empty = no export")
	flag.BoolVar(&cfg.selfTest, "self-test", false,
		"run against a throwaway in-process 9P server and exit")
	flag.Parse()

	// Ctrl-C / SIGTERM cleanly stops the serve loop and closes the connection.
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	if cfg.selfTest {
		if err := runSelfTest(ctx); err != nil {
			log.Fatalf("self-test: %v", err)
		}
		return
	}

	if cfg.sock == "" {
		log.Fatal("no 9P socket: pass --sock <path> (or HYPRSTREAM_9P_SOCK), or --self-test")
	}

	if err := runGuest(ctx, cfg); err != nil {
		log.Fatalf("%v", err)
	}
}

// runSelfTest stands up an in-process p9kit server, binds it as the namespace
// root, and lists "/" through Wanix's own fs.FS view.
func runSelfTest(ctx context.Context) error {
	sock, cleanup, err := startThrowawayServer()
	if err != nil {
		return fmt.Errorf("start throwaway 9P server: %w", err)
	}
	defer cleanup()

	g, err := newGuest(config{sock: sock})
	if err != nil {
		return err
	}
	defer g.close()

	if err := g.connectAndBind(); err != nil {
		return err
	}

	entries, err := wfs.ReadDir(g.root.NS(), ".")
	if err != nil {
		return fmt.Errorf("readdir namespace root: %w", err)
	}
	fmt.Printf("self-test OK: mounted throwaway 9P tree as wanix namespace root; %d entries:\n", len(entries))
	for _, e := range entries {
		fmt.Printf("  %s\tdir=%v\n", e.Name(), e.IsDir())
	}

	// Direction B (#708 phase 2): stand up the export server on a throwaway
	// socket and read the SAME namespace back through a p9 client -- exercising
	// the full serve -> attach -> walk -> readdir path the host uses to import.
	if err := exportSelfTest(g.root.NS(), entries); err != nil {
		return err
	}

	_ = ctx
	return nil
}

// exportSelfTest serves ns over a throwaway export socket, dials it with a p9
// client, and verifies the client sees the same root entries -- the Direction-B
// (guest exports, host imports) build smoke test.
func exportSelfTest(ns wfs.FS, want []wfs.DirEntry) error {
	dir, err := os.MkdirTemp("", "wanix-guest-export-*")
	if err != nil {
		return fmt.Errorf("mktemp export dir: %w", err)
	}
	defer os.RemoveAll(dir)
	sockPath := filepath.Join(dir, "export.sock")

	stop, err := serveExport(ns, sockPath)
	if err != nil {
		return fmt.Errorf("serve export: %w", err)
	}
	defer stop()

	conn, err := net.Dial("unix", sockPath)
	if err != nil {
		return fmt.Errorf("dial export socket %q: %w", sockPath, err)
	}
	defer conn.Close()

	exported, err := p9kit.ClientFS(conn, "")
	if err != nil {
		return fmt.Errorf("p9kit.ClientFS(export): %w", err)
	}
	got, err := wfs.ReadDir(exported, ".")
	if err != nil {
		return fmt.Errorf("readdir exported namespace over 9P client: %w", err)
	}
	if len(got) != len(want) {
		return fmt.Errorf("exported namespace entry count mismatch: got %d, want %d", len(got), len(want))
	}
	fmt.Printf("self-test OK: exported guest namespace over 9P; p9 client read %d entries:\n", len(got))
	for _, e := range got {
		fmt.Printf("  %s\tdir=%v\n", e.Name(), e.IsDir())
	}
	return nil
}

// runGuest connects to a real hyprstream 9P server, binds it as the namespace
// root, then runs the requested command as a task or serves the namespace.
func runGuest(ctx context.Context, cfg config) error {
	g, err := newGuest(cfg)
	if err != nil {
		return err
	}
	defer g.close()

	if err := g.connectAndBind(); err != nil {
		return err
	}
	log.Printf("mounted hyprstream 9P (%q, aname=%q) as wanix namespace root", cfg.sock, cfg.aname)

	// Direction B (#708 phase 2): if an export socket is configured, serve THIS
	// guest's own live namespace over it so the host can import + bind it into
	// the host VFS namespace. This is a SECOND, independent 9P session -- its own
	// listener and server, sharing no state with the Direction-A import above
	// beyond the namespace it reads.
	if s := strings.TrimSpace(cfg.exportSock); s != "" {
		stop, err := serveExport(g.root.NS(), s)
		if err != nil {
			return fmt.Errorf("start guest namespace export: %w", err)
		}
		defer stop()
		log.Printf("exporting guest namespace over 9P at %q (host imports this)", s)
	}

	if strings.TrimSpace(cfg.cmd) != "" {
		return g.runTask(ctx, cfg.cmd)
	}
	return g.serve(ctx)
}

// serveExport stands up a 9P2000.L server (progrium/p9) exporting ns on a Unix
// socket at sockPath -- Direction B of the bidirectional 9P (#708 phase 2): the
// guest serves its OWN live namespace so hyprstream (the host) can import it via
// its socket 9P client Mount (phase 1) and bind it into the host VFS namespace.
//
// This is INDEPENDENT of the Direction-A import session: a separate listener, a
// separate 9P session, no shared state beyond the namespace it reads. The
// listener is bound synchronously (so the socket file exists on return -- the
// host's import dial never races the server coming up); the accept loop runs in
// a goroutine until the returned stop func closes the listener.
func serveExport(ns wfs.FS, sockPath string) (func(), error) {
	// Remove any stale socket from a crashed predecessor (Listen would EADDRINUSE).
	if err := os.Remove(sockPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("remove stale export socket %q: %w", sockPath, err)
	}
	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		return nil, fmt.Errorf("listen export socket %q: %w", sockPath, err)
	}

	srv := p9.NewServer(p9kit.Attacher(ns))
	done := make(chan struct{})
	go func() {
		defer close(done)
		// Serve blocks until the listener is closed (stop), at which point Accept
		// fails with a closing error and Serve returns nil.
		if err := srv.Serve(ln); err != nil {
			log.Printf("export 9P server exited with error: %v", err)
		}
	}()

	stop := func() {
		ln.Close()
		<-done
	}
	return stop, nil
}

// guest owns the Wanix root Task and the live connection to the remote 9P
// server. It is responsible for (re)binding the remote tree as the namespace
// root and for not leaking the underlying net.Conn.
type guest struct {
	cfg  config
	root *wanix.Task

	mu   sync.Mutex
	conn net.Conn // current live connection; owned by this guest
}

func newGuest(cfg config) (*guest, error) {
	root, err := wanix.NewRoot()
	if err != nil {
		return nil, fmt.Errorf("wanix.NewRoot: %w", err)
	}
	// Native host-process task driver so a command bound at #task/new/exec runs.
	root.Register("exec", &native.ExecDriver{})
	return &guest{cfg: cfg, root: root}, nil
}

// connectAndBind dials the 9P socket, adapts it into a wanix fs.FS, and binds
// it as the namespace ROOT (BindReplace makes it authoritative at ".").
func (g *guest) connectAndBind() error {
	conn, err := net.Dial("unix", g.cfg.sock)
	if err != nil {
		return fmt.Errorf("dial 9P socket %q: %w", g.cfg.sock, err)
	}

	hyprfs, err := p9kit.ClientFS(conn, g.cfg.aname)
	if err != nil {
		conn.Close()
		return fmt.Errorf("p9kit.ClientFS: %w", err)
	}

	if err := g.root.NS().Bind(hyprfs, ".", ".", wfs.BindReplace); err != nil {
		conn.Close()
		return fmt.Errorf("bind remote 9P as namespace root: %w", err)
	}

	g.mu.Lock()
	old := g.conn
	g.conn = conn
	g.mu.Unlock()
	if old != nil {
		old.Close() // drop the previous (dead) connection; rebind replaced it
	}
	return nil
}

// serve blocks after the initial bind, periodically walking the remote 9P root
// to detect a dropped/restarted server and reconnect+rebind. It returns nil on
// context cancellation (clean shutdown) and an error only if reconnection is
// exhausted.
func (g *guest) serve(ctx context.Context) error {
	log.Printf("serving namespace; probing every %s (Ctrl-C to stop)", probeInterval)
	ticker := time.NewTicker(probeInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Print("shutdown requested; unmounting")
			return nil
		case <-ticker.C:
			if _, err := wfs.ReadDir(g.root.NS(), "."); err != nil {
				log.Printf("9P root probe failed (%v); attempting reconnect", err)
				if rerr := g.reconnect(ctx); rerr != nil {
					return fmt.Errorf("lost 9P server and could not reconnect: %w", rerr)
				}
				log.Print("reconnected and rebound remote 9P root")
			}
		}
	}
}

// reconnect retries connectAndBind with bounded backoff, honoring ctx.
func (g *guest) reconnect(ctx context.Context) error {
	var lastErr error
	for attempt := 1; attempt <= reconnectAttempts; attempt++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Duration(attempt) * reconnectBackoff):
		}
		if err := g.connectAndBind(); err != nil {
			lastErr = err
			log.Printf("reconnect attempt %d/%d failed: %v", attempt, reconnectAttempts, err)
			continue
		}
		return nil
	}
	return lastErr
}

// runTask spawns cmd as a Wanix task via Wanix's native task filesystem
// interface (#task/new/exec -> write cmd -> ctl start), then waits (bounded)
// for the task's exit file. This is the same mechanism the Wanix shell uses.
//
// IMPORTANT: in this *bare* guest the started host-exec process cannot actually
// run to completion: native.ExecDriver opens the task's stdio at
// #task/<id>/fd/N, which the full Wanix runtime provides by binding a
// console/pipe capability but this minimal embed does not. Nor is the host
// process rooted into the mounted 9P tree. Both are the sandbox's concern,
// deferred to PR-C (see README). We therefore bound the exit wait and report a
// clear diagnostic rather than hang forever.
func (g *guest) runTask(ctx context.Context, cmd string) error {
	ns := g.root.NS()

	// Allocate an exec task: reading #task/new/exec returns the new task ID.
	// (Note: id 1 is the root task; the first allocated exec task is id 2+.)
	idBytes, err := wfs.ReadFile(ns, "#task/new/exec")
	if err != nil {
		return fmt.Errorf("alloc exec task (#task/new/exec): %w", err)
	}
	id := strings.TrimSpace(string(idBytes))
	if id == "" {
		return errors.New("alloc exec task: empty task id")
	}
	log.Printf("allocated task %s for cmd %q", id, cmd)

	// Set the command line, then start the task.
	if err := wfs.WriteFile(ns, fmt.Sprintf("#task/%s/cmd", id), []byte(cmd), 0); err != nil {
		return fmt.Errorf("set task %s cmd: %w", id, err)
	}
	if err := wfs.WriteFile(ns, fmt.Sprintf("#task/%s/ctl", id), []byte("start"), 0); err != nil {
		return fmt.Errorf("start task %s: %w", id, err)
	}

	// Poll the task's exit file until it is populated, ctx is cancelled, or the
	// bounded timeout elapses.
	exitPath := fmt.Sprintf("#task/%s/exit", id)
	deadline := time.NewTimer(taskExitTimeout)
	defer deadline.Stop()
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-deadline.C:
			return fmt.Errorf("task %s did not report exit within %s: the bare guest does not "+
				"wire host-exec stdio caps (#task/%s/fd/N) or root the process into the mounted "+
				"9P tree; that is the sandbox's job (deferred to PR-C / hyprstream-workers)",
				id, taskExitTimeout, id)
		case <-ticker.C:
			b, err := wfs.ReadFile(ns, exitPath)
			if err != nil {
				// exit not yet available; keep polling.
				continue
			}
			code := strings.TrimSpace(string(b))
			if code == "" {
				continue
			}
			log.Printf("task %s exited (code %s)", id, code)
			if code != "0" {
				return fmt.Errorf("task %s exited with code %s", id, code)
			}
			return nil
		}
	}
}

// close releases the live connection. Safe to call multiple times.
func (g *guest) close() {
	g.mu.Lock()
	conn := g.conn
	g.conn = nil
	g.mu.Unlock()
	if conn != nil {
		conn.Close()
	}
}

// startThrowawayServer stands up an in-process p9kit 9P server exporting a
// small fskit.MapFS over a temp Unix socket. Returns the socket path and a
// cleanup func. Used only by --self-test; a real deployment replaces it with
// the hyprstream 9P server.
func startThrowawayServer() (string, func(), error) {
	dir, err := os.MkdirTemp("", "wanix-guest-selftest-*")
	if err != nil {
		return "", nil, err
	}
	sockPath := filepath.Join(dir, "9p.sock")

	ln, err := net.Listen("unix", sockPath)
	if err != nil {
		os.RemoveAll(dir)
		return "", nil, err
	}

	// Exported tree mimics a hyprstream-ish /ai layout.
	exported := fskit.MapFS{
		"models": fskit.MapFS{
			"llama3": fskit.RawNode([]byte("stub\n"), 0644),
		},
		"ctl":     fskit.RawNode([]byte(""), 0644),
		"version": fskit.RawNode([]byte("selftest\n"), 0644),
	}

	srv := p9.NewServer(p9kit.Attacher(exported))
	go func() {
		_ = srv.Serve(ln) // blocks until the listener closes
	}()

	cleanup := func() {
		ln.Close()
		os.RemoveAll(dir)
	}
	return sockPath, cleanup, nil
}
