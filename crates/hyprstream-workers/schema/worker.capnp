@0xb3c4d5e6f7a8b9c0;

using import "/annotations.capnp".mcpDescription;
using import "/annotations.capnp".paramDescription;
using import "/annotations.capnp".mcpScope;
using import "/annotations.capnp".cliHidden;
using import "/streaming.capnp".StreamInfo;

# Cap'n Proto schema for worker service (CRI-aligned resource scopes)
#
# WorkerService manages 4 CRI-aligned resource domains through scoped clients:
#   runtime  — node version and status
#   sandbox  — pod sandbox (Kata VM) lifecycle
#   container — container lifecycle, exec, attach
#   image    — container image management
#
# Convention: Request variants use camelCase names. Response variants
# use the same name suffixed with "Result" to avoid Cap'n Proto naming
# collisions. The code generator strips "Result" to pair them.

struct WorkerRequest {
  # Request ID for tracking
  id @0 :UInt64;

  # Request payload — 4 resource-oriented scoped clients
  union {
    runtime @1 :RuntimeRequest;
    sandbox @2 :SandboxRequest;
    container @3 :ContainerRequest;
    image @4 :ImageRequest;
  }
}

# =============================================================================
# Runtime scope (2 operations) — node version and status
# =============================================================================

struct RuntimeRequest {
  union {
    version @0 :Text
      $mcpDescription("Get runtime version information")
      $mcpScope(query);
    status @1 :StatusRequest
      $mcpDescription("Get runtime status and conditions")
      $mcpScope(query);
  }
}

# =============================================================================
# Sandbox scope (7 operations) — pod sandbox (Kata VM) lifecycle
# =============================================================================

struct SandboxRequest {
  union {
    run @0 :PodSandboxConfig
      $mcpDescription("Create and start a pod sandbox (Kata VM)")
      $mcpScope(write);
    stop @1 :Text
      $mcpDescription("Stop a running pod sandbox")
      $mcpScope(write);
    remove @2 :Text
      $mcpDescription("Remove a pod sandbox")
      $mcpScope(write);
    status @3 :PodSandboxStatusRequest
      $mcpDescription("Get pod sandbox status")
      $mcpScope(query);
    list @4 :PodSandboxFilter
      $mcpDescription("List pod sandboxes matching optional filter")
      $mcpScope(query);
    stats @5 :Text
      $mcpDescription("Get stats for a pod sandbox")
      $mcpScope(query);
    listStats @6 :PodSandboxStatsFilter
      $mcpDescription("List pod sandbox stats")
      $mcpScope(query);
  }
}

# =============================================================================
# Container scope (12 operations) — container lifecycle, exec, attach
# =============================================================================

struct ContainerRequest {
  union {
    create @0 :CreateContainerRequest
      $mcpDescription("Create a container in a pod sandbox")
      $mcpScope(write);
    start @1 :Text
      $mcpDescription("Start a created container")
      $mcpScope(write);
    stop @2 :StopContainerRequest
      $mcpDescription("Stop a running container")
      $mcpScope(write);
    remove @3 :Text
      $mcpDescription("Remove a container")
      $mcpScope(write);
    status @4 :ContainerStatusRequest
      $mcpDescription("Get container status")
      $mcpScope(query);
    list @5 :ContainerFilter
      $mcpDescription("List containers matching optional filter")
      $mcpScope(query);
    stats @6 :Text
      $mcpDescription("Get stats for a container")
      $mcpScope(query);
    listStats @7 :ContainerStatsFilter
      $mcpDescription("List container stats")
      $mcpScope(query);
    exec @8 :ExecSyncRequest
      $mcpDescription("Execute a command synchronously in a container")
      $mcpScope(write);
    attach @9 :AttachRequest
      $cliHidden;
    detach @10 :Text
      $cliHidden;
  }
}

# =============================================================================
# Image scope (5 operations) — container image management
# =============================================================================

struct ImageRequest {
  union {
    list @0 :ImageFilter
      $mcpDescription("List container images matching optional filter")
      $mcpScope(query);
    status @1 :ImageStatusRequest
      $mcpDescription("Get status of a container image")
      $mcpScope(query);
    pull @2 :PullImageRequest
      $mcpDescription("Pull a container image from a registry")
      $mcpScope(write);
    remove @3 :ImageSpec
      $mcpDescription("Remove a container image")
      $mcpScope(write);
    fsInfo @4 :Void
      $mcpDescription("Get filesystem usage information for images")
      $mcpScope(query);
  }
}

# =============================================================================
# Response
# =============================================================================

struct WorkerResponse {
  # Request ID this response corresponds to
  requestId @0 :UInt64;

  # Response payload — mirrors the 4 resource scopes
  union {
    error @1 :ErrorInfo;
    runtimeResult @2 :RuntimeResponse;
    sandboxResult @3 :SandboxResponse;
    containerResult @4 :ContainerResponse;
    imageResult @5 :ImageResponse;
  }
}

# Runtime scoped response
struct RuntimeResponse {
  union {
    error @0 :ErrorInfo;
    version @1 :VersionInfo;
    status @2 :RuntimeStatus;
  }
}

# Sandbox scoped response
struct SandboxResponse {
  union {
    error @0 :ErrorInfo;
    run @1 :Text;
    stop @2 :Void;
    remove @3 :Void;
    status @4 :PodSandboxStatusResponse;
    list @5 :List(PodSandboxInfo);
    stats @6 :PodSandboxStats;
    listStats @7 :List(PodSandboxStats);
  }
}

# Container scoped response
struct ContainerResponse {
  union {
    error @0 :ErrorInfo;
    create @1 :Text;
    start @2 :Void;
    stop @3 :Void;
    remove @4 :Void;
    status @5 :ContainerStatusResponse;
    list @6 :List(ContainerInfo);
    stats @7 :ContainerStats;
    listStats @8 :List(ContainerStats);
    exec @9 :ExecSyncResult;
    attach @10 :StreamInfo;
    detach @11 :Void;
  }
}

# Image scoped response
struct ImageResponse {
  union {
    error @0 :ErrorInfo;
    list @1 :List(ImageInfo);
    status @2 :ImageStatusResult;
    pull @3 :Text;
    remove @4 :Void;
    fsInfo @5 :List(FilesystemUsage);
  }
}

# =============================================================================
# Common Types
# =============================================================================

struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
  details @2 :Text;
}

struct KeyValue {
  key @0 :Text;
  value @1 :Text;
}

# UTC timestamp with nanosecond precision
struct Timestamp {
  seconds @0 :Int64;   # Seconds since Unix epoch (1970-01-01T00:00:00Z)
  nanos @1 :Int32;     # Nanosecond offset [0, 999999999]
}

# =============================================================================
# Pod Sandbox Types (Kata VM)
# =============================================================================

struct PodSandboxMetadata {
  name @0 :Text;
  uid @1 :Text;
  namespace @2 :Text;
  attempt @3 :UInt32;
}

struct PodSandboxConfig {
  metadata @0 :PodSandboxMetadata;
  hostname @1 :Text;
  logDirectory @2 :Text;
  dnsConfig @3 :DNSConfig;
  portMappings @4 :List(PortMapping);
  labels @5 :List(KeyValue);
  annotations @6 :List(KeyValue);
  linux @7 :LinuxPodSandboxConfig;
}

struct DNSConfig {
  servers @0 :List(Text);
  searches @1 :List(Text);
  options @2 :List(Text);
}

struct PortMapping {
  protocol @0 :Protocol;
  containerPort @1 :Int32;
  hostPort @2 :Int32;
  hostIp @3 :Text;
}

enum Protocol {
  tcp @0;
  udp @1;
  sctp @2;
}

struct LinuxPodSandboxConfig {
  cgroupParent @0 :Text;
  securityContext @1 :LinuxSandboxSecurityContext;
  sysctls @2 :List(KeyValue);
  overhead @3 :LinuxContainerResources;
  resources @4 :LinuxContainerResources;
}

struct LinuxSandboxSecurityContext {
  namespaceOptions @0 :NamespaceOption;
  selinuxOptions @1 :SELinuxOption;
  runAsUser @2 :Int64;
  runAsGroup @3 :Int64;
  readonlyRootfs @4 :Bool;
  supplementalGroups @5 :List(Int64);
  privileged @6 :Bool;
  seccompProfilePath @7 :Text;
  apparmorProfile @8 :Text;
}

struct NamespaceOption {
  network @0 :NamespaceMode;
  pid @1 :NamespaceMode;
  ipc @2 :NamespaceMode;
  targetId @3 :Text;
  userNsmode @4 :NamespaceMode;
}

enum NamespaceMode {
  pod @0;
  container @1;
  node @2;
  target @3;
}

struct SELinuxOption {
  user @0 :Text;
  role @1 :Text;
  typeLabel @2 :Text;
  level @3 :Text;
}

struct LinuxContainerResources {
  cpuPeriod @0 :Int64;
  cpuQuota @1 :Int64;
  cpuShares @2 :Int64;
  memoryLimitInBytes @3 :Int64;
  oomScoreAdj @4 :Int64;
  cpusetCpus @5 :Text;
  cpusetMems @6 :Text;
  hugepageLimits @7 :List(HugepageLimit);
  unified @8 :List(KeyValue);
  memorySwapLimitInBytes @9 :Int64;
}

struct HugepageLimit {
  pageSize @0 :Text;
  limit @1 :UInt64;
}

# Pod Sandbox Status

struct PodSandboxStatusRequest {
  podSandboxId @0 :Text $paramDescription("Pod sandbox ID");
  verbose @1 :Bool $paramDescription("Include verbose information");
}

struct PodSandboxStatusResponse {
  status @0 :PodSandboxStatus;
  info @1 :List(KeyValue);
}

struct PodSandboxStatus {
  id @0 :Text;
  metadata @1 :PodSandboxMetadata;
  state @2 :PodSandboxState;
  createdAt @3 :Timestamp;
  network @4 :PodSandboxNetworkStatus;
  linux @5 :LinuxPodSandboxStatus;
  labels @6 :List(KeyValue);
  annotations @7 :List(KeyValue);
  runtimeHandler @8 :Text;
}

enum PodSandboxState {
  sandboxReady @0;
  sandboxNotReady @1;
}

struct PodSandboxNetworkStatus {
  ip @0 :Text;
  additionalIps @1 :List(PodIP);
}

struct PodIP {
  ip @0 :Text;
}

struct LinuxPodSandboxStatus {
  namespaces @0 :Namespace;
}

struct Namespace {
  options @0 :NamespaceOption;
}

# Pod Sandbox Filter & Info

struct PodSandboxFilter {
  id @0 :Text $paramDescription("Filter by pod sandbox ID");
  state @1 :PodSandboxState;
  labelSelector @2 :List(KeyValue);
}

struct PodSandboxInfo {
  id @0 :Text;
  metadata @1 :PodSandboxMetadata;
  state @2 :PodSandboxState;
  createdAt @3 :Timestamp;
  labels @4 :List(KeyValue);
  annotations @5 :List(KeyValue);
  runtimeHandler @6 :Text;
}

# =============================================================================
# Container Types
# =============================================================================

struct ContainerMetadata {
  name @0 :Text;
  attempt @1 :UInt32;
}

struct ContainerConfig {
  metadata @0 :ContainerMetadata;
  image @1 :ImageSpec;
  command @2 :List(Text);
  args @3 :List(Text);
  workingDir @4 :Text;
  envs @5 :List(KeyValue);
  mounts @6 :List(Mount);
  devices @7 :List(Device);
  labels @8 :List(KeyValue);
  annotations @9 :List(KeyValue);
  logPath @10 :Text;
  stdin @11 :Bool;
  stdinOnce @12 :Bool;
  tty @13 :Bool;
  linux @14 :LinuxContainerConfig;
}

struct Mount {
  containerPath @0 :Text;
  hostPath @1 :Text;
  readonly @2 :Bool;
  selinuxRelabel @3 :Bool;
  propagation @4 :MountPropagation;
}

enum MountPropagation {
  propagationPrivate @0;
  propagationHostToContainer @1;
  propagationBidirectional @2;
}

struct Device {
  containerPath @0 :Text;
  hostPath @1 :Text;
  permissions @2 :Text;
}

struct LinuxContainerConfig {
  resources @0 :LinuxContainerResources;
  securityContext @1 :LinuxContainerSecurityContext;
}

struct LinuxContainerSecurityContext {
  capabilities @0 :Capability;
  privileged @1 :Bool;
  namespaceOptions @2 :NamespaceOption;
  selinuxOptions @3 :SELinuxOption;
  runAsUser @4 :Int64;
  runAsGroup @5 :Int64;
  runAsUsername @6 :Text;
  readonlyRootfs @7 :Bool;
  supplementalGroups @8 :List(Int64);
  apparmorProfile @9 :Text;
  seccompProfilePath @10 :Text;
  noNewPrivs @11 :Bool;
  maskedPaths @12 :List(Text);
  readonlyPaths @13 :List(Text);
}

struct Capability {
  addCapabilities @0 :List(Text);
  dropCapabilities @1 :List(Text);
}

# Container Requests

struct CreateContainerRequest {
  podSandboxId @0 :Text $paramDescription("Pod sandbox ID to create container in");
  config @1 :ContainerConfig;
  sandboxConfig @2 :PodSandboxConfig;
}

struct StopContainerRequest {
  containerId @0 :Text $paramDescription("Container ID to stop");
  timeout @1 :Int64 $paramDescription("Timeout in seconds before force stop");
}

struct ContainerStatusRequest {
  containerId @0 :Text $paramDescription("Container ID to query");
  verbose @1 :Bool $paramDescription("Include verbose information");
}

struct ContainerStatusResponse {
  status @0 :ContainerStatus;
  info @1 :List(KeyValue);
}

struct ContainerStatus {
  id @0 :Text;
  metadata @1 :ContainerMetadata;
  state @2 :ContainerState;
  createdAt @3 :Timestamp;
  startedAt @4 :Timestamp;
  finishedAt @5 :Timestamp;
  exitCode @6 :Int32;
  image @7 :ImageSpec;
  imageRef @8 :Text;
  reason @9 :Text;
  message @10 :Text;
  labels @11 :List(KeyValue);
  annotations @12 :List(KeyValue);
  mounts @13 :List(Mount);
  logPath @14 :Text;
}

enum ContainerState {
  containerCreated @0;
  containerRunning @1;
  containerExited @2;
  containerUnknown @3;
}

struct ContainerFilter {
  id @0 :Text $paramDescription("Filter by container ID");
  podSandboxId @1 :Text $paramDescription("Filter by pod sandbox ID");
  state @2 :ContainerState;
  labelSelector @3 :List(KeyValue);
}

struct ContainerInfo {
  id @0 :Text;
  podSandboxId @1 :Text;
  metadata @2 :ContainerMetadata;
  image @3 :ImageSpec;
  imageRef @4 :Text;
  state @5 :ContainerState;
  createdAt @6 :Timestamp;
  labels @7 :List(KeyValue);
  annotations @8 :List(KeyValue);
}

# Exec

struct ExecSyncRequest {
  containerId @0 :Text $paramDescription("Container ID to execute in");
  cmd @1 :List(Text) $paramDescription("Command to execute");
  timeout @2 :Int64 $paramDescription("Execution timeout in seconds");
}

struct ExecSyncResult {
  stdout @0 :Data;
  stderr @1 :Data;
  exitCode @2 :Int32;
}

# Terminal Attach

struct AttachRequest {
  containerId @0 :Text;
  fds @1 :List(UInt8);
}

# =============================================================================
# Stats Types
# =============================================================================

struct PodSandboxStatsFilter {
  id @0 :Text $paramDescription("Filter by pod sandbox ID");
  labelSelector @1 :List(KeyValue);
}

struct PodSandboxStats {
  attributes @0 :PodSandboxAttributes;
  linux @1 :LinuxPodSandboxStats;
}

struct PodSandboxAttributes {
  id @0 :Text;
  metadata @1 :PodSandboxMetadata;
  labels @2 :List(KeyValue);
  annotations @3 :List(KeyValue);
}

struct LinuxPodSandboxStats {
  cpu @0 :CpuUsage;
  memory @1 :MemoryUsage;
  network @2 :NetworkUsage;
  process @3 :ProcessUsage;
}

struct ContainerStatsFilter {
  id @0 :Text $paramDescription("Filter by container ID");
  podSandboxId @1 :Text $paramDescription("Filter by pod sandbox ID");
  labelSelector @2 :List(KeyValue);
}

struct ContainerStats {
  attributes @0 :ContainerAttributes;
  cpu @1 :CpuUsage;
  memory @2 :MemoryUsage;
  writableLayer @3 :FilesystemUsage;
}

struct ContainerAttributes {
  id @0 :Text;
  metadata @1 :ContainerMetadata;
  labels @2 :List(KeyValue);
  annotations @3 :List(KeyValue);
}

struct CpuUsage {
  timestamp @0 :Int64;
  usageCoreNanoSeconds @1 :UInt64;
  usageNanoCores @2 :UInt64;
}

struct MemoryUsage {
  timestamp @0 :Int64;
  workingSetBytes @1 :UInt64;
  availableBytes @2 :UInt64;
  usageBytes @3 :UInt64;
  rssBytes @4 :UInt64;
  pageFaults @5 :UInt64;
  majorPageFaults @6 :UInt64;
}

struct NetworkUsage {
  timestamp @0 :Int64;
  defaultInterface @1 :NetworkInterfaceUsage;
  interfaces @2 :List(NetworkInterfaceUsage);
}

struct NetworkInterfaceUsage {
  name @0 :Text;
  rxBytes @1 :UInt64;
  txBytes @2 :UInt64;
  rxErrors @3 :UInt64;
  txErrors @4 :UInt64;
}

struct ProcessUsage {
  timestamp @0 :Int64;
  processCount @1 :UInt64;
}

# =============================================================================
# Image Types
# =============================================================================

struct ImageSpec {
  image @0 :Text $paramDescription("Image name/reference (e.g., 'alpine:latest')");
  annotations @1 :List(KeyValue);
  runtimeHandler @2 :Text;
}

struct ImageFilter {
  image @0 :ImageSpec;
}

struct ImageStatusRequest {
  image @0 :ImageSpec;
  verbose @1 :Bool $paramDescription("Include verbose information");
}

struct ImageStatusResult {
  image @0 :ImageInfo;
  info @1 :List(KeyValue);
}

struct PullImageRequest {
  image @0 :ImageSpec;
  auth @1 :AuthConfig;
  sandboxConfig @2 :PodSandboxConfig;
}

struct AuthConfig {
  username @0 :Text;
  password @1 :Text;
  auth @2 :Text;
  serverAddress @3 :Text;
  identityToken @4 :Text;
  registryToken @5 :Text;
}

struct ImageInfo {
  id @0 :Text;
  repoTags @1 :List(Text);
  repoDigests @2 :List(Text);
  size @3 :UInt64;
  uid @4 :Int64;
  username @5 :Text;
  spec @6 :ImageSpec;
  pinned @7 :Bool;
}

struct FilesystemUsage {
  timestamp @0 :Int64;
  fsId @1 :FilesystemIdentifier;
  usedBytes @2 :UInt64;
  inodesUsed @3 :UInt64;
}

struct FilesystemIdentifier {
  mountpoint @0 :Text;
}

# =============================================================================
# Runtime Info Types
# =============================================================================

struct StatusRequest {
  verbose @0 :Bool $paramDescription("Include verbose status information");
}

struct VersionInfo {
  version @0 :Text;
  runtimeName @1 :Text;
  runtimeVersion @2 :Text;
  runtimeApiVersion @3 :Text;
}

struct RuntimeStatus {
  conditions @0 :List(RuntimeCondition);
}

struct RuntimeCondition {
  conditionType @0 :Text;
  status @1 :Bool;
  reason @2 :Text;
  message @3 :Text;
}

# =============================================================================
# Events (pub/sub)
# =============================================================================
# These events are serialized into EventEnvelope.payload (from hyprstream-rpc).
# Topics use format: worker.{entity_id}.{event_name}

struct WorkerEvent {
  union {
    sandboxStarted @0 :SandboxStarted;
    sandboxStopped @1 :SandboxStopped;
    containerStarted @2 :ContainerStarted;
    containerStopped @3 :ContainerStopped;
  }
}

struct SandboxStarted {
  sandboxId @0 :Text;
  metadata @1 :Text;    # JSON metadata
  vmPid @2 :UInt32;     # VM process ID
}

struct SandboxStopped {
  sandboxId @0 :Text;
  reason @1 :Text;
  exitCode @2 :Int32;
}

struct ContainerStarted {
  containerId @0 :Text;
  sandboxId @1 :Text;
  image @2 :Text;
}

struct ContainerStopped {
  containerId @0 :Text;
  sandboxId @1 :Text;
  exitCode @2 :Int32;
  reason @3 :Text;
}
