package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
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

func TestFuseMountProcessAliveHandlesStalePid(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, fusePIDFile), []byte("999999999\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	alive, err := fuseMountProcessAlive(dir)
	if err != nil {
		t.Fatal(err)
	}
	if alive {
		t.Fatal("expected stale pid to be treated as not mounted")
	}
	if _, err := os.Stat(filepath.Join(dir, fusePIDFile)); !os.IsNotExist(err) {
		t.Fatalf("stale pid file should be removed, stat err=%v", err)
	}
}

func TestFuseMountProcessAliveRecognizesLivePid(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, fusePIDFile), []byte(strconv.Itoa(os.Getpid())+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	alive, err := fuseMountProcessAlive(dir)
	if err != nil {
		t.Fatal(err)
	}
	if !alive {
		t.Fatal("expected current process pid to be treated as mounted")
	}
}

func TestUnpublishMountRemovesStalePidFile(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, fusePIDFile), []byte("999999999\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := unpublishMount(context.Background(), dir, filepath.Dir(dir)); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, fusePIDFile)); !os.IsNotExist(err) {
		t.Fatalf("pid file should be removed, stat err=%v", err)
	}
}

func TestUnpublishMountRejectsTargetOutsideKubeletRoot(t *testing.T) {
	if err := unpublishMount(context.Background(), "/etc/passwd", "/var/lib/kubelet"); err == nil {
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
