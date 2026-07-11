package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
	ticketEnv               = "HYPRSTREAM_9P_UNAME"
	mountinfoPath           = "/proc/self/mountinfo"
	mountReadyTimeout       = 30 * time.Second
	mountPollInterval       = 100 * time.Millisecond
)

// fuseMountState is per-volume mount bookkeeping stored OUTSIDE the target
// directory (under the plugin's own state dir), so a NodePublishVolume retry
// can read it back through the underlying filesystem rather than through the
// live FUSE mount that shadows the target.
type fuseMountState struct {
	VolumeID   string `json:"volume_id"`
	TargetPath string `json:"target_path"`
	PID        int    `json:"pid"`
}

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
	StateDir     string
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
		StateDir:   cfg.stateDir,
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
		// Idempotency (CSI requires NodePublishVolume to be idempotent): the
		// authoritative check is whether the target is already a live FUSE
		// mount for this volume, plus whether a mounter we started is still
		// coming up. Never rely on a pid file under the target — the mount
		// shadows it, so a retry would read nothing and stack a second mount.
		live, err := fuseMountLive(plan.StateDir, plan.VolumeID, plan.TargetPath)
		if err != nil {
			return err
		}
		if live {
			return nil
		}
		if err := os.MkdirAll(plan.StateDir, 0o700); err != nil {
			return fmt.Errorf("create CSI state dir: %w", err)
		}
		cmd := fuseCommand(ctx, plan)
		if err := cmd.Start(); err != nil {
			return fmt.Errorf("start FUSE 9P mounter: %w", err)
		}
		state := fuseMountState{VolumeID: plan.VolumeID, TargetPath: plan.TargetPath, PID: cmd.Process.Pid}
		if err := writeMountState(plan.StateDir, state); err != nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
			return fmt.Errorf("write FUSE mounter state: %w", err)
		}
		if err := cmd.Process.Release(); err != nil {
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
			_ = removeMountState(plan.StateDir, plan.VolumeID)
			return fmt.Errorf("release FUSE mounter process: %w", err)
		}
		// Do not report NodePublish success until the target is an authoritative
		// live FUSE mount: the mounter can fail its 9P attach or FUSE mount after
		// Start(), and a pod must never start before its volume is usable. On
		// failure tear the mounter down so a retry starts clean.
		if err := waitFuseMountReady(ctx, plan.TargetPath, cmd.Process.Pid); err != nil {
			if processAlive(cmd.Process.Pid) {
				_ = syscall.Kill(cmd.Process.Pid, syscall.SIGTERM)
			}
			_ = removeMountState(plan.StateDir, plan.VolumeID)
			return err
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

// fuseMountLive reports whether the target already carries a live FUSE mount
// for this volume, or a mounter we started is still establishing it. It is the
// idempotency gate for NodePublishVolume retries: an authoritative mountpoint
// check first (via /proc/self/mountinfo), then the out-of-target state file.
func fuseMountLive(stateDir, volumeID, targetPath string) (bool, error) {
	mounted, err := isFuseMountPoint(targetPath)
	if err != nil {
		return false, err
	}
	if mounted {
		return true, nil
	}
	state, ok, err := readMountState(stateDir, volumeID)
	if err != nil || !ok {
		return false, err
	}
	if processAlive(state.PID) {
		return true, nil
	}
	// Mounter died before it established the mount: drop the stale record.
	_ = removeMountState(stateDir, volumeID)
	return false, nil
}

func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}

func mountStatePath(stateDir, volumeID string) string {
	sum := sha256.Sum256([]byte(volumeID))
	return filepath.Join(stateDir, hex.EncodeToString(sum[:])+".json")
}

func writeMountState(stateDir string, state fuseMountState) error {
	if stateDir == "" {
		return errors.New("CSI state dir is required")
	}
	if err := os.MkdirAll(stateDir, 0o700); err != nil {
		return fmt.Errorf("create CSI state dir: %w", err)
	}
	raw, err := json.Marshal(state)
	if err != nil {
		return err
	}
	return os.WriteFile(mountStatePath(stateDir, state.VolumeID), raw, 0o600)
}

func readMountState(stateDir, volumeID string) (fuseMountState, bool, error) {
	if stateDir == "" || volumeID == "" {
		return fuseMountState{}, false, nil
	}
	raw, err := os.ReadFile(mountStatePath(stateDir, volumeID))
	if errors.Is(err, os.ErrNotExist) {
		return fuseMountState{}, false, nil
	}
	if err != nil {
		return fuseMountState{}, false, fmt.Errorf("read FUSE mounter state: %w", err)
	}
	var state fuseMountState
	if err := json.Unmarshal(raw, &state); err != nil {
		return fuseMountState{}, false, fmt.Errorf("parse FUSE mounter state: %w", err)
	}
	return state, true, nil
}

func removeMountState(stateDir, volumeID string) error {
	if stateDir == "" || volumeID == "" {
		return nil
	}
	if err := os.Remove(mountStatePath(stateDir, volumeID)); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

// waitFuseMountReady blocks until targetPath is an authoritative live FUSE mount,
// the mounter (pid) exits first, the deadline elapses, or ctx is cancelled.
func waitFuseMountReady(ctx context.Context, targetPath string, pid int) error {
	deadline := time.Now().Add(mountReadyTimeout)
	for {
		mounted, err := isFuseMountPoint(targetPath)
		if err != nil {
			return err
		}
		if mounted {
			return nil
		}
		if !processAlive(pid) {
			return fmt.Errorf("FUSE 9P mounter exited before establishing mount at %s", targetPath)
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("FUSE 9P mount at %s not ready within %s", targetPath, mountReadyTimeout)
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(mountPollInterval):
		}
	}
}

// isFuseMountPoint reports whether targetPath is currently a FUSE mount point,
// read from the kernel's authoritative /proc/self/mountinfo.
func isFuseMountPoint(targetPath string) (bool, error) {
	f, err := os.Open(mountinfoPath)
	if err != nil {
		return false, fmt.Errorf("open %s: %w", mountinfoPath, err)
	}
	defer f.Close()
	return parseMountinfo(f, targetPath)
}

// parseMountinfo scans mountinfo lines for a FUSE mount at targetPath.
// Format: "<id> <parent> <maj:min> <root> <mountpoint> <opts> [tags...] - <fstype> <source> <superopts>".
func parseMountinfo(r io.Reader, targetPath string) (bool, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		fields := strings.Fields(scanner.Text())
		if len(fields) < 5 {
			continue
		}
		if unescapeMountField(fields[4]) != targetPath {
			continue
		}
		sep := -1
		for i := 5; i < len(fields); i++ {
			if fields[i] == "-" {
				sep = i
				break
			}
		}
		if sep < 0 || sep+1 >= len(fields) {
			continue
		}
		fstype := fields[sep+1]
		if fstype == "fuse" || strings.HasPrefix(fstype, "fuse.") {
			return true, nil
		}
	}
	if err := scanner.Err(); err != nil {
		return false, fmt.Errorf("read %s: %w", mountinfoPath, err)
	}
	return false, nil
}

// unescapeMountField decodes the octal escapes (\040 space, \011 tab, \012
// newline, \134 backslash) the kernel emits in mountinfo path fields.
func unescapeMountField(field string) string {
	if !strings.Contains(field, `\`) {
		return field
	}
	var b strings.Builder
	for i := 0; i < len(field); i++ {
		if field[i] == '\\' && i+3 < len(field) {
			// ParseUint with bitSize 8 rejects any octal escape > 0377,
			// so the byte(v) conversion below can never truncate.
			if v, err := strconv.ParseUint(field[i+1:i+4], 8, 8); err == nil {
				b.WriteByte(byte(v))
				i += 3
				continue
			}
		}
		b.WriteByte(field[i])
	}
	return b.String()
}

func unpublishMount(ctx context.Context, stateDir, volumeID, targetPath, kubeletRootDir string) error {
	if targetPath == "" {
		return errors.New("target_path is required")
	}
	targetPath, err := validateTargetPath(targetPath, kubeletRootDir)
	if err != nil {
		return err
	}
	state, ok, err := readMountState(stateDir, volumeID)
	if err != nil {
		return err
	}
	if ok && processAlive(state.PID) {
		_ = syscall.Kill(state.PID, syscall.SIGTERM)
	}
	unmountErr := unmountTarget(ctx, targetPath)
	if ok && state.PID > 0 {
		deadline := time.Now().Add(2 * time.Second)
		for time.Now().Before(deadline) {
			if !processAlive(state.PID) {
				break
			}
			time.Sleep(50 * time.Millisecond)
		}
		if processAlive(state.PID) {
			_ = syscall.Kill(state.PID, syscall.SIGKILL)
		}
	}
	// Do not report NodeUnpublish success while the target is still mounted:
	// verify the mount is gone (the authoritative check), surfacing the unmount
	// error only when the target genuinely remains mounted.
	mounted, mErr := isFuseMountPoint(targetPath)
	if mErr != nil {
		return mErr
	}
	if mounted {
		if unmountErr != nil {
			return fmt.Errorf("unmount %s: %w", targetPath, unmountErr)
		}
		return fmt.Errorf("target %s still mounted after unmount", targetPath)
	}
	return removeMountState(stateDir, volumeID)
}

// unmountTarget tries each available unmount helper and returns the last error
// if none succeed. It is a no-op (nil) when the target is already unmounted.
func unmountTarget(ctx context.Context, targetPath string) error {
	var lastErr error
	for _, argv := range [][]string{
		{"fusermount3", "-u", targetPath},
		{"fusermount", "-u", targetPath},
		{"umount", targetPath},
	} {
		if err := exec.CommandContext(ctx, argv[0], argv[1:]...).Run(); err == nil {
			return nil
		} else {
			lastErr = err
		}
	}
	return lastErr
}
