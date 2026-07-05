package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
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
		_ = json.NewEncoder(w).Encode(mountTicketResponse{Ticket: "ticket-123", Audience: "hyprstream-9p", Capability: "mount@9p://webtransport/tenant/a"})
	}))
	defer srv.Close()

	plan, err := buildMountPlan(config{
		defaultMounter:    "fuse",
		defaultPlane:      "webtransport",
		oauthAudience:     "hyprstream-9p",
		mountTicketURL:    srv.URL,
		transportCarrier:  "webtransport",
		transportEndpoint: "https://hyprstream-streams:7011/9p",
	}, &csi.NodePublishVolumeRequest{
		VolumeId:   "vol-a",
		TargetPath: "/var/lib/kubelet/pods/p/volumes/kubernetes.io~csi/v/mount",
		VolumeContext: map[string]string{
			attrNamespacePath: "/tenant/a",
			attrServiceTokens: `{"hyprstream-9p":{"token":"pod-token","expirationTimestamp":"2026-07-05T19:00:00Z"}}`,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Plane != "webtransport" || got.NamespacePath != "/tenant/a" {
		t.Fatalf("ticket request = %+v", got)
	}
	if plan.Mounter != "fuse" || plan.Ticket != "ticket-123" || plan.DialTarget != "https://hyprstream-streams:7011/9p" {
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
		oauthAudience:     "hyprstream-9p",
		mountTicketURL:    srv.URL,
		transportCarrier:  "webtransport",
		transportEndpoint: "https://node.local/9p",
		bridgeListen:      "127.0.0.1:0",
	}, &csi.NodePublishVolumeRequest{
		VolumeId:   "vol-k",
		TargetPath: "/mnt/k",
		VolumeContext: map[string]string{
			attrNamespacePath: "/tenant/k",
			attrMounter:       "kernel",
		},
		Secrets: map[string]string{
			attrServiceTokens: `{"hyprstream-9p":{"token":"pod-token","expirationTimestamp":"2026-07-05T19:00:00Z"}}`,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if plan.Mounter != "kernel" || plan.KernelBridge != "127.0.0.1:0" || plan.Ticket != "ticket-k" {
		t.Fatalf("plan = %+v", plan)
	}
}

func TestBuildMountPlanRejectsPlaintextTicketAttribute(t *testing.T) {
	_, err := buildMountPlan(config{mountTicketURL: "http://127.0.0.1"}, &csi.NodePublishVolumeRequest{
		VolumeId:   "vol",
		TargetPath: "/mnt/x",
		VolumeContext: map[string]string{
			attrNamespacePath: "/tenant/a",
			attrTicket:        "ticket-in-etcd",
		},
	})
	if err == nil {
		t.Fatal("expected plaintext ticket rejection")
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
			attrNamespacePath: "/tenant/a",
		},
	})
	if err == nil {
		t.Fatal("expected service account token requirement")
	}
}

func TestBuildMountPlanRejectsBadInput(t *testing.T) {
	_, err := buildMountPlan(config{mountTicketURL: "http://127.0.0.1"}, &csi.NodePublishVolumeRequest{
		VolumeId:   "vol",
		TargetPath: "/mnt/x",
		VolumeContext: map[string]string{
			attrNamespacePath: "relative",
		},
	})
	if err == nil {
		t.Fatal("expected relative namespacePath rejection")
	}
}
