package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	csi "github.com/container-storage-interface/spec/lib/go/csi"
)

func TestBuildMountPlanFuseRequestsScopedTicket(t *testing.T) {
	var got mountTicketRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("method = %s", r.Method)
		}
		if r.Header.Get("Authorization") != "Bearer pod-token" {
			t.Fatalf("missing bearer token: %q", r.Header.Get("Authorization"))
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Fatal(err)
		}
		_ = json.NewEncoder(w).Encode(mountTicketResponse{Ticket: "ticket-123", Audience: "hyprstream-9p", Capability: "mount@9p://webtransport/export/tenant-a"})
	}))
	defer srv.Close()

	plan, err := buildMountPlan(config{
		defaultMounter:    "fuse",
		defaultPlane:      "webtransport",
		kubeletRootDir:    "/var/lib/kubelet",
		oauthAudience:     "hyprstream-9p",
		mountTicketURL:    srv.URL,
		transportCarrier:  "webtransport",
		transportEndpoint: "https://hyprstream-streams:7011/9p",
	}, &csi.NodePublishVolumeRequest{
		VolumeId:   "vol-a",
		TargetPath: "/var/lib/kubelet/pods/p/volumes/kubernetes.io~csi/v/mount",
		VolumeContext: map[string]string{
			attrAname:         "export:tenant-a",
			attrServiceTokens: `{"hyprstream-9p":{"token":"pod-token","expirationTimestamp":"2026-07-05T19:00:00Z"}}`,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Plane != "webtransport" || got.Aname != "export:tenant-a" {
		t.Fatalf("ticket request = %+v", got)
	}
	if plan.Mounter != "fuse" || plan.Ticket != "ticket-123" || plan.DialTarget != "https://hyprstream-streams:7011/9p" || plan.Aname != "export:tenant-a" {
		t.Fatalf("plan = %+v", plan)
	}
}

func TestBuildMountPlanKernelUsesBridge(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(mountTicketResponse{Ticket: "ticket-k"})
	}))
	defer srv.Close()

	plan, err := buildMountPlan(config{
		defaultMounter:    "fuse",
		defaultPlane:      "webtransport",
		kubeletRootDir:    "/mnt",
		oauthAudience:     "hyprstream-9p",
		mountTicketURL:    srv.URL,
		transportCarrier:  "webtransport",
		transportEndpoint: "https://node.local/9p",
		bridgeListen:      "127.0.0.1:0",
	}, &csi.NodePublishVolumeRequest{
		VolumeId:   "vol-k",
		TargetPath: "/mnt/k",
		VolumeContext: map[string]string{
			attrExportRef: "export:tenant-k",
			attrMounter:   "kernel",
		},
		Secrets: map[string]string{
			attrServiceTokens: `{"hyprstream-9p":{"token":"pod-token","expirationTimestamp":"2026-07-05T19:00:00Z"}}`,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Mounter != "kernel" || plan.KernelBridge != "127.0.0.1:0" || plan.Ticket != "ticket-k" || plan.Aname != "export:tenant-k" {
		t.Fatalf("plan = %+v", plan)
	}
}

func TestBuildMountPlanRejectsPlaintextTicketAttribute(t *testing.T) {
	_, err := buildMountPlan(config{mountTicketURL: "http://127.0.0.1"}, &csi.NodePublishVolumeRequest{
		VolumeId:   "vol",
		TargetPath: "/mnt/x",
		VolumeContext: map[string]string{
			attrAname:  "export:tenant-a",
			attrTicket: "ticket-in-etcd",
		},
	})
	if err == nil {
		t.Fatal("expected plaintext ticket rejection")
	}
}

func TestBuildMountPlanRequiresTransportEndpoint(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(mountTicketResponse{Ticket: "ticket"})
	}))
	defer srv.Close()

	_, err := buildMountPlan(config{
		defaultMounter: "fuse",
		defaultPlane:   "webtransport",
		oauthAudience:  "hyprstream-9p",
		mountTicketURL: srv.URL,
	}, &csi.NodePublishVolumeRequest{
		VolumeId:   "vol",
		TargetPath: "/mnt/x",
		VolumeContext: map[string]string{
			attrAname:         "export:tenant-a",
			attrServiceTokens: `{"hyprstream-9p":{"token":"pod-token","expirationTimestamp":"2026-07-05T19:00:00Z"}}`,
		},
	})
	if err == nil {
		t.Fatal("expected transport endpoint requirement")
	}
}

func TestBuildMountPlanRequiresProjectedServiceAccountToken(t *testing.T) {
	_, err := buildMountPlan(config{
		oauthAudience:  "hyprstream-9p",
		mountTicketURL: "http://127.0.0.1",
	}, &csi.NodePublishVolumeRequest{
		VolumeId:   "vol",
		TargetPath: "/mnt/x",
		VolumeContext: map[string]string{
			attrAname: "export:tenant-a",
		},
	})
	if err == nil {
		t.Fatal("expected service account token requirement")
	}
}

func TestBuildMountPlanRejectsLegacyNamespacePath(t *testing.T) {
	_, err := buildMountPlan(config{mountTicketURL: "http://127.0.0.1"}, &csi.NodePublishVolumeRequest{
		VolumeId:   "vol",
		TargetPath: "/mnt/x",
		VolumeContext: map[string]string{
			attrLegacyNamespacePath: "/sandboxes/sb-1",
		},
	})
	if err == nil {
		t.Fatal("expected legacy namespacePath rejection")
	}
}

func TestBuildMountPlanRejectsTargetOutsideKubeletRoot(t *testing.T) {
	_, err := buildMountPlan(config{mountTicketURL: "http://127.0.0.1", kubeletRootDir: "/var/lib/kubelet"}, &csi.NodePublishVolumeRequest{
		VolumeId:   "vol",
		TargetPath: "/etc/passwd",
		VolumeContext: map[string]string{
			attrAname: "export:tenant-a",
		},
	})
	if err == nil {
		t.Fatal("expected target path outside kubelet root rejection")
	}
}

func TestBuildMountPlanRejectsUncleanTargetPath(t *testing.T) {
	_, err := buildMountPlan(config{mountTicketURL: "http://127.0.0.1", kubeletRootDir: "/var/lib/kubelet"}, &csi.NodePublishVolumeRequest{
		VolumeId:   "vol",
		TargetPath: "/var/lib/kubelet/pods/../escape",
		VolumeContext: map[string]string{
			attrAname: "export:tenant-a",
		},
	})
	if err == nil {
		t.Fatal("expected unclean target path rejection")
	}
}

func TestBuildMountPlanRejectsSymlinkTargetPath(t *testing.T) {
	root := t.TempDir()
	target := filepath.Join(root, "pods", "pod-a", "volumes", "kubernetes.io~csi", "vol", "mount")
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("/etc", target); err != nil {
		t.Fatal(err)
	}
	_, err := buildMountPlan(config{mountTicketURL: "http://127.0.0.1", kubeletRootDir: root}, &csi.NodePublishVolumeRequest{
		VolumeId:   "vol",
		TargetPath: target,
		VolumeContext: map[string]string{
			attrAname: "export:tenant-a",
		},
	})
	if err == nil {
		t.Fatal("expected symlink target path rejection")
	}
}

func TestFuseCommandKeepsTicketOutOfArgvAndUsesUnameEnv(t *testing.T) {
	plan := mountPlan{
		TargetPath: "/mnt/hypr",
		DialTarget: "tcp://127.0.0.1:564",
		Ticket:     "ticket-secret",
	}
	cmd := fuseCommand(context.Background(), plan)
	argv := strings.Join(cmd.Args, "\x00")
	if strings.Contains(argv, "ticket-secret") {
		t.Fatalf("ticket leaked into argv: %q", cmd.Args)
	}
	if !strings.Contains(argv, "--aname") {
		t.Fatalf("export selector must be presented as aname: %q", cmd.Args)
	}
	if !hasEnv(cmd.Env, ticketEnv+"=ticket-secret") {
		t.Fatalf("missing ticket env %s in %v", ticketEnv, cmd.Env)
	}
}

func TestMountStateLivesOutsideTargetAndRoundTrips(t *testing.T) {
	stateDir := t.TempDir()
	target := t.TempDir()
	state := fuseMountState{VolumeID: "vol-a", TargetPath: target, PID: os.Getpid()}
	if err := writeMountState(stateDir, state); err != nil {
		t.Fatal(err)
	}
	// State must not land inside the target the FUSE client mounts over.
	entries, err := os.ReadDir(target)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("state must not be written under target, found %d entries", len(entries))
	}
	got, ok, err := readMountState(stateDir, "vol-a")
	if err != nil || !ok {
		t.Fatalf("read state ok=%v err=%v", ok, err)
	}
	if got.PID != state.PID || got.TargetPath != target {
		t.Fatalf("round-trip mismatch: %+v", got)
	}
}

func TestFuseMountLiveRecognizesEstablishingMounter(t *testing.T) {
	stateDir := t.TempDir()
	target := t.TempDir()
	if err := writeMountState(stateDir, fuseMountState{VolumeID: "vol-a", TargetPath: target, PID: os.Getpid()}); err != nil {
		t.Fatal(err)
	}
	// Target is not a real mount, but a mounter we started is alive: idempotent.
	live, err := fuseMountLive(stateDir, "vol-a", target)
	if err != nil {
		t.Fatal(err)
	}
	if !live {
		t.Fatal("expected alive mounter to be treated as live")
	}
}

func TestFuseMountLiveTreatsDeadMounterAsNotLive(t *testing.T) {
	stateDir := t.TempDir()
	target := t.TempDir()
	if err := writeMountState(stateDir, fuseMountState{VolumeID: "vol-a", TargetPath: target, PID: 999999999}); err != nil {
		t.Fatal(err)
	}
	live, err := fuseMountLive(stateDir, "vol-a", target)
	if err != nil {
		t.Fatal(err)
	}
	if live {
		t.Fatal("expected dead mounter over unmounted target to be treated as not live")
	}
	if _, ok, _ := readMountState(stateDir, "vol-a"); ok {
		t.Fatal("stale state should be removed")
	}
}

// Regression for the CSI idempotency bug (#867): a NodePublishVolume retry
// while the mounter is still live must NOT start a second FUSE mounter. If it
// did, executeMountPlan would invoke the (absent) hypr9p-guest binary and fail;
// returning nil proves the idempotency gate short-circuited before starting.
func TestExecuteMountPlanFuseIdempotentOnRetry(t *testing.T) {
	stateDir := t.TempDir()
	target := t.TempDir()
	if err := writeMountState(stateDir, fuseMountState{VolumeID: "vol-a", TargetPath: target, PID: os.Getpid()}); err != nil {
		t.Fatal(err)
	}
	plan := mountPlan{Mounter: "fuse", VolumeID: "vol-a", TargetPath: target, StateDir: stateDir}
	if err := executeMountPlan(context.Background(), plan); err != nil {
		t.Fatalf("retry must be idempotent, got %v", err)
	}
	if _, ok, _ := readMountState(stateDir, "vol-a"); !ok {
		t.Fatal("mount state should survive an idempotent retry")
	}
}

func TestParseMountinfoDetectsFuseMount(t *testing.T) {
	target := "/var/lib/kubelet/pods/p/volumes/kubernetes.io~csi/v/mount"
	line := "121 98 0:52 / " + target + " rw,nosuid,nodev,relatime shared:1 - fuse.hypr9p hypr9p rw,user_id=0,group_id=0\n"
	mounted, err := parseMountinfo(strings.NewReader(line), target)
	if err != nil || !mounted {
		t.Fatalf("expected fuse mount detected, mounted=%v err=%v", mounted, err)
	}
	if m, _ := parseMountinfo(strings.NewReader(line), "/some/other/path"); m {
		t.Fatal("non-matching path must not report mounted")
	}
	ext := "36 35 0:33 / /mnt/x rw,relatime shared:1 - ext4 /dev/sda1 rw\n"
	if m, _ := parseMountinfo(strings.NewReader(ext), "/mnt/x"); m {
		t.Fatal("non-fuse fstype must not count as a fuse mount")
	}
	esc := "36 35 0:33 / /mnt/with\\040space rw - fuse.hypr9p hypr9p rw\n"
	if m, _ := parseMountinfo(strings.NewReader(esc), "/mnt/with space"); !m {
		t.Fatal("expected octal-escaped mount point to match")
	}
}

func TestWaitFuseMountReadyFailsWhenMounterDies(t *testing.T) {
	// Target is not a live mount and the mounter pid is dead: NodePublish must
	// surface an error rather than reporting success before the mount is usable.
	err := waitFuseMountReady(context.Background(), t.TempDir(), 999999999)
	if err == nil {
		t.Fatal("expected error when mounter exits before establishing the mount")
	}
}

func TestUnpublishMountRemovesState(t *testing.T) {
	stateDir := t.TempDir()
	root := t.TempDir()
	target := filepath.Join(root, "mount")
	if err := os.MkdirAll(target, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := writeMountState(stateDir, fuseMountState{VolumeID: "vol-a", TargetPath: target, PID: 999999999}); err != nil {
		t.Fatal(err)
	}
	if err := unpublishMount(context.Background(), stateDir, "vol-a", target, root); err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := readMountState(stateDir, "vol-a"); ok {
		t.Fatal("state should be removed after unpublish")
	}
}

func TestUnpublishMountRejectsTargetOutsideKubeletRoot(t *testing.T) {
	if err := unpublishMount(context.Background(), t.TempDir(), "vol-a", "/etc/passwd", "/var/lib/kubelet"); err == nil {
		t.Fatal("expected target path outside kubelet root rejection")
	}
}

func hasEnv(env []string, want string) bool {
	for _, item := range env {
		if item == want {
			return true
		}
	}
	return false
}
