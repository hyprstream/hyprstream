package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	csi "github.com/container-storage-interface/spec/lib/go/csi"
)

const (
	attrNamespacePath = "namespacePath"
	attrTenant        = "tenant"
	attrPlane         = "plane"
	attrMounter       = "mounter"
	attrCarrier       = "carrier"
	attrEndpoint      = "endpoint"
	attrTicket        = "ticket"
	attrServiceTokens = "csi.storage.k8s.io/serviceAccount.tokens"
)

type mountTicketRequest struct {
	Plane         string `json:"plane"`
	NamespacePath string `json:"namespace_path"`
	Pubkey        string `json:"pubkey,omitempty"`
}

type mountTicketResponse struct {
	Ticket     string `json:"ticket"`
	TokenType  string `json:"token_type"`
	ExpiresIn  int64  `json:"expires_in"`
	Audience   string `json:"audience"`
	Capability string `json:"capability"`
}

type serviceAccountToken struct {
	Token               string `json:"token"`
	ExpirationTimestamp string `json:"expirationTimestamp"`
}

type mountPlan struct {
	VolumeID      string
	TargetPath    string
	NamespacePath string
	Tenant        string
	Plane         string
	Mounter       string
	Carrier       string
	Endpoint      string
	Ticket        string
	DialTarget    string
	KernelBridge  string
}

func buildMountPlan(cfg config, req *csi.NodePublishVolumeRequest) (mountPlan, error) {
	if req.GetVolumeId() == "" {
		return mountPlan{}, errors.New("volume_id is required")
	}
	if req.GetTargetPath() == "" {
		return mountPlan{}, errors.New("target_path is required")
	}
	attrs := req.GetVolumeContext()
	if attrs[attrTicket] != "" {
		return mountPlan{}, errors.New("ticket volume attribute is forbidden; CSI must mint mount tickets at NodePublishVolume")
	}
	namespacePath := firstNonEmpty(attrs[attrNamespacePath], cfgValue(cfgDefaultNamespacePath(), "/"))
	plane := firstNonEmpty(attrs[attrPlane], cfg.defaultPlane)
	mounter := strings.ToLower(firstNonEmpty(attrs[attrMounter], cfg.defaultMounter))
	carrier := firstNonEmpty(attrs[attrCarrier], cfg.transportCarrier)
	endpoint := firstNonEmpty(attrs[attrEndpoint], cfg.transportEndpoint)
	if mounter != "kernel" && mounter != "fuse" {
		return mountPlan{}, fmt.Errorf("mounter must be kernel or fuse, got %q", mounter)
	}
	if namespacePath == "" || !strings.HasPrefix(namespacePath, "/") {
		return mountPlan{}, fmt.Errorf("namespacePath must be absolute, got %q", namespacePath)
	}
	bearer, err := serviceAccountBearer(req, cfg.oauthAudience)
	if err != nil {
		return mountPlan{}, err
	}
	ticket, err := requestMountTicket(context.Background(), cfg.mountTicketURL, bearer, plane, namespacePath)
	if err != nil {
		return mountPlan{}, err
	}
	plan := mountPlan{
		VolumeID:      req.GetVolumeId(),
		TargetPath:    req.GetTargetPath(),
		NamespacePath: namespacePath,
		Tenant:        attrs[attrTenant],
		Plane:         plane,
		Mounter:       mounter,
		Carrier:       carrier,
		Endpoint:      endpoint,
		Ticket:        ticket.Ticket,
		DialTarget:    endpoint,
	}
	if mounter == "kernel" {
		plan.KernelBridge = cfg.bridgeListen
	}
	return plan, nil
}

func serviceAccountBearer(req *csi.NodePublishVolumeRequest, audience string) (string, error) {
	raw := firstNonEmpty(req.GetSecrets()[attrServiceTokens], req.GetVolumeContext()[attrServiceTokens])
	if raw == "" {
		return "", fmt.Errorf("%s is required for NodePublishVolume ticket minting", attrServiceTokens)
	}
	var tokens map[string]serviceAccountToken
	if err := json.Unmarshal([]byte(raw), &tokens); err != nil {
		return "", fmt.Errorf("parse %s: %w", attrServiceTokens, err)
	}
	entry, ok := tokens[audience]
	if !ok && audience == "" {
		entry, ok = tokens[""]
	}
	if !ok {
		return "", fmt.Errorf("service account token for audience %q not found", audience)
	}
	if entry.Token == "" {
		return "", fmt.Errorf("service account token for audience %q is empty", audience)
	}
	return entry.Token, nil
}

func cfgDefaultNamespacePath() string { return "/" }

func cfgValue(v string, fallback string) string {
	if v == "" {
		return fallback
	}
	return v
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func requestMountTicket(ctx context.Context, endpoint, bearer, plane, namespacePath string) (mountTicketResponse, error) {
	if endpoint == "" {
		return mountTicketResponse{}, errors.New("mount-ticket endpoint is required")
	}
	body, err := json.Marshal(mountTicketRequest{Plane: plane, NamespacePath: namespacePath})
	if err != nil {
		return mountTicketResponse{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return mountTicketResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	client := http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return mountTicketResponse{}, fmt.Errorf("request mount ticket: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return mountTicketResponse{}, fmt.Errorf("request mount ticket: status %s", resp.Status)
	}
	var out mountTicketResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return mountTicketResponse{}, fmt.Errorf("decode mount ticket response: %w", err)
	}
	if out.Ticket == "" {
		return mountTicketResponse{}, errors.New("mount-ticket response missing ticket")
	}
	return out, nil
}

func executeMountPlan(ctx context.Context, plan mountPlan) error {
	if err := os.MkdirAll(plan.TargetPath, 0o755); err != nil {
		return fmt.Errorf("create target path: %w", err)
	}
	switch plan.Mounter {
	case "fuse":
		cmd := exec.CommandContext(ctx, "hypr9p-guest", "--dial", plan.DialTarget, "--aname", plan.Ticket, "--fuse-mount", plan.TargetPath)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			return fmt.Errorf("start FUSE 9P mounter: %w", err)
		}
		pidPath := filepath.Join(plan.TargetPath, ".hyprstream-fuse.pid")
		_ = os.WriteFile(pidPath, []byte(fmt.Sprintln(cmd.Process.Pid)), 0o600)
		return nil
	case "kernel":
		cmd := exec.CommandContext(ctx, "hyprstream-csi-bridge",
			"mount-kernel",
			"--dial", plan.DialTarget,
			"--ticket", plan.Ticket,
			"--target", plan.TargetPath,
			"--listen", plan.KernelBridge,
		)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("kernel v9fs bridge mount: %w", err)
		}
		return nil
	default:
		return fmt.Errorf("unsupported mounter %q", plan.Mounter)
	}
}
