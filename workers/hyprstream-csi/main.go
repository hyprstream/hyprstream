package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	csi "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type config struct {
	endpoint          string
	driverName        string
	nodeID            string
	defaultMounter    string
	defaultPlane      string
	oauthAudience     string
	mountTicketURL    string
	transportCarrier  string
	transportEndpoint string
	bridgeListen      string
	kubeletRootDir    string
	logLevel          string
	dryRun            bool
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "csi" {
		os.Args = append([]string{os.Args[0]}, os.Args[2:]...)
	}
	if len(os.Args) > 1 && os.Args[1] == "node" {
		os.Args = append([]string{os.Args[0]}, os.Args[2:]...)
	}

	cfg := config{}
	flag.StringVar(&cfg.endpoint, "endpoint", "unix:///csi/csi.sock", "CSI listen endpoint")
	flag.StringVar(&cfg.driverName, "driver-name", "csi.hyprstream.io", "CSI driver name")
	flag.StringVar(&cfg.nodeID, "node-id", os.Getenv("NODE_NAME"), "Kubernetes node name")
	flag.StringVar(&cfg.defaultMounter, "default-mounter", "fuse", "default mounter: fuse or kernel")
	flag.StringVar(&cfg.defaultPlane, "default-plane", "webtransport", "default mount-ticket plane")
	flag.StringVar(&cfg.oauthAudience, "oauth-audience", "hyprstream-9p", "audience key for CSI service account tokenRequests")
	flag.StringVar(&cfg.mountTicketURL, "mount-ticket-url", "http://hyprstream-oauth:6791/oauth/mount-ticket", "hyprstream /oauth/mount-ticket endpoint")
	flag.StringVar(&cfg.transportCarrier, "transport-carrier", "tcp", "dial-time carrier")
	flag.StringVar(&cfg.transportEndpoint, "transport-endpoint", "", "carrier endpoint")
	flag.StringVar(&cfg.bridgeListen, "bridge-listen", "127.0.0.1:0", "node-local stream bridge listen address")
	flag.StringVar(&cfg.kubeletRootDir, "kubelet-root-dir", "/var/lib/kubelet", "kubelet root directory containing CSI target paths")
	flag.StringVar(&cfg.logLevel, "log-level", "info", "log level")
	flag.BoolVar(&cfg.dryRun, "dry-run", false, "plan NodePublishVolume without executing mounts")
	flag.Parse()

	if cfg.nodeID == "" {
		cfg.nodeID, _ = os.Hostname()
	}
	if err := run(context.Background(), cfg); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, cfg config) error {
	network, address, err := parseEndpoint(cfg.endpoint)
	if err != nil {
		return err
	}
	if network == "unix" {
		_ = os.Remove(address)
		if err := os.MkdirAll(filepath.Dir(address), 0o755); err != nil {
			return err
		}
	}
	lis, err := net.Listen(network, address)
	if err != nil {
		return fmt.Errorf("listen %s %q: %w", network, address, err)
	}
	defer lis.Close()

	srv := grpc.NewServer()
	driver := &nodeDriver{cfg: cfg}
	csi.RegisterIdentityServer(srv, driver)
	csi.RegisterNodeServer(srv, driver)

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-stop
		srv.GracefulStop()
	}()
	log.Printf("hyprstream CSI node plugin listening on %s", cfg.endpoint)
	return srv.Serve(lis)
}

func parseEndpoint(endpoint string) (string, string, error) {
	switch {
	case strings.HasPrefix(endpoint, "unix://"):
		path := strings.TrimPrefix(endpoint, "unix://")
		if path == "" {
			return "", "", fmt.Errorf("empty unix endpoint")
		}
		return "unix", path, nil
	case strings.HasPrefix(endpoint, "tcp://"):
		addr := strings.TrimPrefix(endpoint, "tcp://")
		if addr == "" {
			return "", "", fmt.Errorf("empty tcp endpoint")
		}
		return "tcp", addr, nil
	default:
		return "", "", fmt.Errorf("unsupported endpoint %q", endpoint)
	}
}

type nodeDriver struct {
	csi.UnimplementedIdentityServer
	csi.UnimplementedNodeServer
	cfg config
}

func (d *nodeDriver) GetPluginInfo(context.Context, *csi.GetPluginInfoRequest) (*csi.GetPluginInfoResponse, error) {
	return &csi.GetPluginInfoResponse{Name: d.cfg.driverName, VendorVersion: "0.1.0"}, nil
}

func (d *nodeDriver) GetPluginCapabilities(context.Context, *csi.GetPluginCapabilitiesRequest) (*csi.GetPluginCapabilitiesResponse, error) {
	return &csi.GetPluginCapabilitiesResponse{
		Capabilities: []*csi.PluginCapability{{
			Type: &csi.PluginCapability_Service_{
				Service: &csi.PluginCapability_Service{
					Type: csi.PluginCapability_Service_VOLUME_ACCESSIBILITY_CONSTRAINTS,
				},
			},
		}},
	}, nil
}

func (d *nodeDriver) Probe(context.Context, *csi.ProbeRequest) (*csi.ProbeResponse, error) {
	return &csi.ProbeResponse{}, nil
}

func (d *nodeDriver) NodeGetInfo(context.Context, *csi.NodeGetInfoRequest) (*csi.NodeGetInfoResponse, error) {
	return &csi.NodeGetInfoResponse{NodeId: d.cfg.nodeID}, nil
}

func (d *nodeDriver) NodeGetCapabilities(context.Context, *csi.NodeGetCapabilitiesRequest) (*csi.NodeGetCapabilitiesResponse, error) {
	return &csi.NodeGetCapabilitiesResponse{
		Capabilities: []*csi.NodeServiceCapability{{
			Type: &csi.NodeServiceCapability_Rpc{
				Rpc: &csi.NodeServiceCapability_RPC{
					Type: csi.NodeServiceCapability_RPC_STAGE_UNSTAGE_VOLUME,
				},
			},
		}},
	}, nil
}

func (d *nodeDriver) NodeStageVolume(context.Context, *csi.NodeStageVolumeRequest) (*csi.NodeStageVolumeResponse, error) {
	return &csi.NodeStageVolumeResponse{}, nil
}

func (d *nodeDriver) NodeUnstageVolume(context.Context, *csi.NodeUnstageVolumeRequest) (*csi.NodeUnstageVolumeResponse, error) {
	return &csi.NodeUnstageVolumeResponse{}, nil
}

func (d *nodeDriver) NodePublishVolume(ctx context.Context, req *csi.NodePublishVolumeRequest) (*csi.NodePublishVolumeResponse, error) {
	plan, err := buildMountPlan(d.cfg, req)
	if err != nil {
		return nil, status.Error(codes.InvalidArgument, err.Error())
	}
	if !d.cfg.dryRun {
		if err := executeMountPlan(ctx, plan); err != nil {
			return nil, status.Error(codes.Internal, err.Error())
		}
	}
	return &csi.NodePublishVolumeResponse{}, nil
}

func (d *nodeDriver) NodeUnpublishVolume(ctx context.Context, req *csi.NodeUnpublishVolumeRequest) (*csi.NodeUnpublishVolumeResponse, error) {
	if err := unpublishMount(ctx, req.GetTargetPath(), d.cfg.kubeletRootDir); err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}
	return &csi.NodeUnpublishVolumeResponse{}, nil
}
