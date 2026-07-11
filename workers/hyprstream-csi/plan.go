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
	"strconv"
	"strings"
	"syscall"
	"time"

	csi "github.com/container-storage-interface/spec/lib/go/csi"
)

const (
	attrAname               = "aname"
	attrExportRef           = "exportRef"
	attrLegacyNamespacePath = "namespacePath"
	attrTenant              = "tenant"
	attrPlane               = "plane"
	attrMounter             = "mounter"
	attrCarrier             = "carrier"
	attrEndpoint            = "endpoint"
	attrTicket              = "ticket"
	attrServiceTokens       = "csi.storage.k8s.io/serviceAccount.tokens"
	fusePIDFile             = ".hyprstream-fuse.pid"
	ticketEnv               = "HYPRSTREAM_9P_UNAME"
)

type mountTicketRequest struct {
	Plane  string `json:"plane"`
	Aname  string `json:"aname"`
	Pubkey string `json:"pubkey,omitempty"`
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
	VolumeID     string
	TargetPath   string
	Aname        string
	Tenant       string
	Plane        string
	Mounter      string
	Carrier      string
	Endpoint     string
	Ticket       string
	DialTarget   string
	KernelBridge string
}

func buildMountPlan(cfg config, req *csi.NodePublishVolumeRequest) (mountPlan, error) {
	if req.GetVolumeId() == "" {
		return mountPlan{}, errors.New("volume_id is required")
	}
	if req.GetTargetPath() == "" {
		return mountPlan{}, errors.New("target_path is required")
	}
	targetPath, err := validateTargetPath(req.GetTargetPath(), cfg.kubeletRootDir)
	if err != nil {
		return mountPlan{}, err
	}
	attrs := req.GetVolumeContext()
	if attrs[attrTicket] != "" {
		return mountPlan{}, errors.New("ticket volume attribute is forbidden; CSI must mint mount tickets at NodePublishVolume")
	}
	if attrs[attrLegacyNamespacePath] != "" {
		return mountPlan{}, errors.New("namespacePath volume attribute is obsolete; use aname/exportRef as the 9P attach selector")
	}
	aname := firstNonEmpty(attrs[attrAname], attrs[attrExportRef], cfgDefaultAname())
	plane := firstNonEmpty(attrs[attrPlane], cfg.defaultPlane)
	mounter := strings.ToLower(firstNonEmpty(attrs[attrMounter], cfg.defaultMounter))
	carrier := firstNonEmpty(attrs[attrCarrier], cfg.transportCarrier)
	endpoint := firstNonEmpty(attrs[attrEndpoint], cfg.transportEndpoint)
	if mounter != "kernel" && mounter != "fuse" {
		return mountPlan{}, fmt.Errorf("mounter must be kernel or fuse, got %q", mounter)
	}
	if endpoint == "" {
		return mountPlan{}, errors.New("transport endpoint is required; configure csi.transport.endpoint or the volume endpoint attribute")
	}
	bearer, err := serviceAccountBearer(req, cfg.oauthAudience)
	if err != nil {
		return mountPlan{}, err
	}
	ticket, err := requestMountTicket(context.Background(), cfg.mountTicketURL, bearer, plane, aname)
	if err != nil {
		return mountPlan{}, err
	}
	plan := mountPlan{
		VolumeID:   req.GetVolumeId(),
		TargetPath: targetPath,
		Aname:      aname,
		Tenant:     attrs[attrTenant],
		Plane:      plane,
		Mounter:    mounter,
		Carrier:    carrier,
		Endpoint:   endpoint,
		Ticket:     ticket.Ticket,
		DialTarget: endpoint,
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

func cfgDefaultAname() string { return "default" }

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

func validateTargetPath(targetPath, kubeletRootDir string) (string, error) {
	cleanTarget := filepath.Clean(targetPath)
	if !filepath.IsAbs(cleanTarget) {
		return "", fmt.Errorf("target_path must be absolute, got %q", targetPath)
	}
	if cleanTarget != targetPath {
		return "", fmt.Errorf("target_path must be clean, got %q", targetPath)
	}
	root := filepath.Clean(firstNonEmpty(kubeletRootDir, "/var/lib/kubelet"))
	if !filepath.IsAbs(root) {
		return "", fmt.Errorf("kubelet root dir must be absolute, got %q", kubeletRootDir)
	}
	rel, err := filepath.Rel(root, cleanTarget)
	if err != nil {
		return "", fmt.Errorf("relativize target_path: %w", err)
	}
	if rel == "." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) || rel == ".." || filepath.IsAbs(rel) {
		return "", fmt.Errorf("target_path %q must be under kubelet root %q", cleanTarget, root)
	}
	if _, err := os.Lstat(cleanTarget); err == nil {
		resolved, err := filepath.EvalSymlinks(cleanTarget)
		if err != nil {
			return "", fmt.Errorf("resolve target_path symlinks: %w", err)
		}
		if resolved != cleanTarget {
			return "", fmt.Errorf("target_path %q must not resolve through symlink %q", cleanTarget, resolved)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("stat target_path: %w", err)
	}
	parent := filepath.Dir(cleanTarget)
	for {
		if parent == root || parent == filepath.Dir(parent) {
			break
		}
		if info, err := os.Lstat(parent); err == nil {
			if info.Mode()&os.ModeSymlink != 0 {
				return "", fmt.Errorf("target_path parent %q must not be a symlink", parent)
			}
		} else if !errors.Is(err, os.ErrNotExist) {
			return "", fmt.Errorf("stat target_path parent: %w", err)
		}
		parent = filepath.Dir(parent)
	}
	return cleanTarget, nil
}

func requestMountTicket(ctx context.Context, endpoint, bearer, plane, aname string) (mountTicketResponse, error) {
	if endpoint == "" {
		return mountTicketResponse{}, errors.New("mount-ticket endpoint is required")
	}
	body, err := json.Marshal(mountTicketRequest{Plane: plane, Aname: aname})
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
		mounted, err := fuseMountProcessAlive(plan.TargetPath)
		if err != nil {
			return err
		}
		if mounted {
			return nil
		}
		cmd := fuseCommand(ctx, plan)
		if err := cmd.Start(); err != nil {
			return fmt.Errorf("start FUSE 9P mounter: %w", err)
		}
		pidPath := filepath.Join(plan.TargetPath, fusePIDFile)
		if err := os.WriteFile(pidPath, []byte(fmt.Sprintln(cmd.Process.Pid)), 0o600); err != nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
			return fmt.Errorf("write FUSE mounter pid: %w", err)
		}
		if err := cmd.Process.Release(); err != nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
			_ = os.Remove(pidPath)
			return fmt.Errorf("release FUSE mounter process: %w", err)
		}
		return nil
	case "kernel":
		cmd := exec.CommandContext(ctx, "hyprstream-csi-bridge",
			"mount-kernel",
			"--dial", plan.DialTarget,
			"--target", plan.TargetPath,
			"--listen", plan.KernelBridge,
			"--aname", plan.Aname,
		)
		cmd.Env = append(os.Environ(), ticketEnv+"="+plan.Ticket)
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

func fuseCommand(ctx context.Context, plan mountPlan) *exec.Cmd {
	cmd := exec.CommandContext(ctx, "hypr9p-guest",
		"--dial", plan.DialTarget,
		"--aname", plan.Aname,
		"--fuse-mount", plan.TargetPath,
	)
	cmd.Env = append(os.Environ(), ticketEnv+"="+plan.Ticket)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd
}

func fuseMountProcessAlive(targetPath string) (bool, error) {
	pid, ok, err := readFusePID(targetPath)
	if err != nil || !ok {
		return false, err
	}
	if err := syscall.Kill(pid, 0); err == nil {
		return true, nil
	} else if errors.Is(err, syscall.ESRCH) {
		_ = os.Remove(filepath.Join(targetPath, fusePIDFile))
		return false, nil
	} else {
		return false, fmt.Errorf("check FUSE mounter pid %d: %w", pid, err)
	}
}

func readFusePID(targetPath string) (int, bool, error) {
	raw, err := os.ReadFile(filepath.Join(targetPath, fusePIDFile))
	if errors.Is(err, os.ErrNotExist) {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, fmt.Errorf("read FUSE mounter pid: %w", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(raw)))
	if err != nil || pid <= 0 {
		return 0, false, fmt.Errorf("invalid FUSE mounter pid %q", strings.TrimSpace(string(raw)))
	}
	return pid, true, nil
}

func unpublishMount(ctx context.Context, targetPath, kubeletRootDir string) error {
	if targetPath == "" {
		return errors.New("target_path is required")
	}
	targetPath, err := validateTargetPath(targetPath, kubeletRootDir)
	if err != nil {
		return err
	}
	pid, ok, err := readFusePID(targetPath)
	if err != nil {
		return err
	}
	if ok {
		_ = syscall.Kill(pid, syscall.SIGTERM)
	}
	_ = bestEffortUnmount(ctx, targetPath)
	if ok {
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) {
			if err := syscall.Kill(pid, 0); errors.Is(err, syscall.ESRCH) {
				break
			}
			time.Sleep(50 * time.Millisecond)
		}
		if err := syscall.Kill(pid, 0); err == nil {
			_ = syscall.Kill(pid, syscall.SIGKILL)
		}
	}
	_ = os.Remove(filepath.Join(targetPath, fusePIDFile))
	return nil
}

func bestEffortUnmount(ctx context.Context, targetPath string) error {
	for _, argv := range [][]string{
		{"fusermount3", "-u", targetPath},
		{"fusermount", "-u", targetPath},
		{"umount", targetPath},
	} {
		if err := exec.CommandContext(ctx, argv[0], argv[1:]...).Run(); err == nil {
			return nil
		}
	}
	return nil
}
