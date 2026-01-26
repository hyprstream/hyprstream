@0xa1b2c3d4e5f60718;

# Cap'n Proto schema for hyprstream-workers
#
# CRI-aligned RuntimeService + ImageService + WorkflowService
# PodSandbox = Kata VM, Container = OCI container within VM

# ═══════════════════════════════════════════════════════════════════════════════
# RuntimeService (CRI-aligned)
# ═══════════════════════════════════════════════════════════════════════════════

struct RuntimeRequest {
  id @0 :UInt64;

  union {
    # Runtime info
    version @1 :Text;
    status @2 :StatusRequest;

    # Pod Sandbox lifecycle (Kata VM)
    runPodSandbox @3 :PodSandboxConfig;
    stopPodSandbox @4 :Text;        # pod_sandbox_id
    removePodSandbox @5 :Text;      # pod_sandbox_id
    podSandboxStatus @6 :PodSandboxStatusRequest;
    listPodSandbox @7 :PodSandboxFilter;

    # Container lifecycle
    createContainer @8 :CreateContainerRequest;
    startContainer @9 :Text;        # container_id
    stopContainer @10 :StopContainerRequest;
    removeContainer @11 :Text;      # container_id
    containerStatus @12 :ContainerStatusRequest;
    listContainers @13 :ContainerFilter;

    # Exec
    execSync @14 :ExecSyncRequest;

    # Stats
    podSandboxStats @15 :Text;      # pod_sandbox_id
    listPodSandboxStats @16 :PodSandboxStatsFilter;
    containerStats @17 :Text;       # container_id
    listContainerStats @18 :ContainerStatsFilter;

    # Terminal attach/detach (tmux-like I/O streaming)
    attach @19 :AttachRequest;
    detach @20 :Text;               # container_id

    # FD stream authorization (DH handshake)
    startFdStream @21 :StartFdStreamRequest;
  }
}

struct RuntimeResponse {
  requestId @0 :UInt64;

  union {
    success @1 :Void;
    error @2 :ErrorInfo;
    version @3 :VersionInfo;
    status @4 :RuntimeStatus;
    sandboxId @5 :Text;
    sandboxStatus @6 :PodSandboxStatusResponse;
    sandboxes @7 :List(PodSandboxInfo);
    containerId @8 :Text;
    containerStatus @9 :ContainerStatusResponse;
    containers @10 :List(ContainerInfo);
    execResult @11 :ExecSyncResult;
    sandboxStats @12 :PodSandboxStats;
    sandboxStatsList @13 :List(PodSandboxStats);
    containerStatsResult @14 :ContainerStats;
    containerStatsList @15 :List(ContainerStats);

    # Terminal attach response
    attachResponse @16 :AttachResponse;

    # FD stream authorization response
    fdStreamAuthorized @17 :FdStreamAuthResponse;
  }
}

# ═══════════════════════════════════════════════════════════════════════════════
# ImageService (CRI-aligned)
# ═══════════════════════════════════════════════════════════════════════════════

struct ImageRequest {
  id @0 :UInt64;

  union {
    listImages @1 :ImageFilter;
    imageStatus @2 :ImageStatusRequest;
    pullImage @3 :PullImageRequest;
    removeImage @4 :ImageSpec;
    imageFsInfo @5 :Void;
  }
}

struct ImageResponse {
  requestId @0 :UInt64;

  union {
    success @1 :Void;
    error @2 :ErrorInfo;
    images @3 :List(ImageInfo);
    imageStatus @4 :ImageStatusResult;
    imageRef @5 :Text;
    fsInfo @6 :List(FilesystemUsage);
  }
}

# ═══════════════════════════════════════════════════════════════════════════════
# WorkflowService
# ═══════════════════════════════════════════════════════════════════════════════

struct WorkflowRequest {
  id @0 :UInt64;

  union {
    scanRepo @1 :Text;              # repo_id
    registerWorkflow @2 :WorkflowDef;
    listWorkflows @3 :Void;
    dispatch @4 :DispatchRequest;
    subscribe @5 :SubscribeRequest;
    unsubscribe @6 :Text;           # subscription_id
    getRun @7 :Text;                # run_id
    listRuns @8 :Text;              # workflow_id
  }
}

struct WorkflowResponse {
  requestId @0 :UInt64;

  union {
    success @1 :Void;
    error @2 :ErrorInfo;
    workflows @3 :List(WorkflowInfo);
    workflowId @4 :Text;
    runId @5 :Text;
    subscriptionId @6 :Text;
    run @7 :WorkflowRun;
    runs @8 :List(WorkflowRun);
    workflowDefs @9 :List(WorkflowDef);
  }
}

# ═══════════════════════════════════════════════════════════════════════════════
# Common Types
# ═══════════════════════════════════════════════════════════════════════════════

struct ErrorInfo {
  message @0 :Text;
  code @1 :Text;
  details @2 :Text;
}

struct KeyValue {
  key @0 :Text;
  value @1 :Text;
}

# ═══════════════════════════════════════════════════════════════════════════════
# Pod Sandbox Types (Kata VM)
# ═══════════════════════════════════════════════════════════════════════════════

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
  podSandboxId @0 :Text;
  verbose @1 :Bool;
}

struct PodSandboxStatusResponse {
  status @0 :PodSandboxStatus;
  info @1 :List(KeyValue);
}

struct PodSandboxStatus {
  id @0 :Text;
  metadata @1 :PodSandboxMetadata;
  state @2 :PodSandboxState;
  createdAt @3 :Int64;
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
  id @0 :Text;
  state @1 :PodSandboxState;
  labelSelector @2 :List(KeyValue);
}

struct PodSandboxInfo {
  id @0 :Text;
  metadata @1 :PodSandboxMetadata;
  state @2 :PodSandboxState;
  createdAt @3 :Int64;
  labels @4 :List(KeyValue);
  annotations @5 :List(KeyValue);
  runtimeHandler @6 :Text;
}

# ═══════════════════════════════════════════════════════════════════════════════
# Container Types
# ═══════════════════════════════════════════════════════════════════════════════

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

struct ImageSpec {
  image @0 :Text;
  annotations @1 :List(KeyValue);
  runtimeHandler @2 :Text;
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
  podSandboxId @0 :Text;
  config @1 :ContainerConfig;
  sandboxConfig @2 :PodSandboxConfig;
}

struct StopContainerRequest {
  containerId @0 :Text;
  timeout @1 :Int64;
}

struct ContainerStatusRequest {
  containerId @0 :Text;
  verbose @1 :Bool;
}

struct ContainerStatusResponse {
  status @0 :ContainerStatus;
  info @1 :List(KeyValue);
}

struct ContainerStatus {
  id @0 :Text;
  metadata @1 :ContainerMetadata;
  state @2 :ContainerState;
  createdAt @3 :Int64;
  startedAt @4 :Int64;
  finishedAt @5 :Int64;
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
  id @0 :Text;
  podSandboxId @1 :Text;
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
  createdAt @6 :Int64;
  labels @7 :List(KeyValue);
  annotations @8 :List(KeyValue);
}

# Exec

struct ExecSyncRequest {
  containerId @0 :Text;
  cmd @1 :List(Text);
  timeout @2 :Int64;
}

struct ExecSyncResult {
  stdout @0 :Data;
  stderr @1 :Data;
  exitCode @2 :Int32;
}

# Terminal Attach (tmux-like I/O streaming)
#
# Flow:
#   1. Client sends AttachRequest
#   2. Server returns AttachResponse with stream info and server pubkey
#   3. Client sends StartFdStreamRequest with client pubkey for DH handshake
#   4. Server returns FdStreamAuthResponse confirming authorization
#   5. Client subscribes to DH-derived topic via StreamService
#   6. Server streams FD data via StreamBuilder (HMAC-authenticated)

struct AttachRequest {
  containerId @0 :Text;
  # Optional: attach to specific FDs (default: stdout/stderr for output)
  # 0 = stdin (client->container), 1 = stdout, 2 = stderr (container->client)
  fds @1 :List(UInt8);
}

struct AttachResponse {
  containerId @0 :Text;
  # Stream ID for this attach session
  streamId @1 :Text;
  # StreamService endpoint for subscribing
  streamEndpoint @2 :Text;
  # Server's ephemeral Ristretto255 public key (32 bytes) for DH
  serverPubkey @3 :Data;
}

# FD stream authorization handshake (DH key exchange)
struct StartFdStreamRequest {
  streamId @0 :Text;
  # Client's ephemeral Ristretto255 public key (32 bytes)
  clientPubkey @1 :Data;
}

struct FdStreamAuthResponse {
  streamId @0 :Text;
  # Confirmation that stream is authorized
  authorized @1 :Bool;
}

# ═══════════════════════════════════════════════════════════════════════════════
# Stats Types
# ═══════════════════════════════════════════════════════════════════════════════

struct PodSandboxStatsFilter {
  id @0 :Text;
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
  id @0 :Text;
  podSandboxId @1 :Text;
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

# ═══════════════════════════════════════════════════════════════════════════════
# Image Types
# ═══════════════════════════════════════════════════════════════════════════════

struct ImageFilter {
  image @0 :ImageSpec;
}

struct ImageStatusRequest {
  image @0 :ImageSpec;
  verbose @1 :Bool;
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

# ═══════════════════════════════════════════════════════════════════════════════
# Runtime Info Types
# ═══════════════════════════════════════════════════════════════════════════════

struct StatusRequest {
  verbose @0 :Bool;
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

# ═══════════════════════════════════════════════════════════════════════════════
# Workflow Types
# ═══════════════════════════════════════════════════════════════════════════════

struct WorkflowDef {
  path @0 :Text;
  repoId @1 :Text;
  name @2 :Text;
  triggers @3 :List(EventTrigger);
  yaml @4 :Text;
}

struct WorkflowInfo {
  id @0 :Text;
  name @1 :Text;
  path @2 :Text;
  repoId @3 :Text;
  enabled @4 :Bool;
}

struct DispatchRequest {
  workflowId @0 :Text;
  inputs @1 :List(KeyValue);
}

struct SubscribeRequest {
  workflowId @0 :Text;
  trigger @1 :EventTrigger;
}

struct EventTrigger {
  union {
    repositoryEvent @0 :RepositoryEventTrigger;
    trainingProgress @1 :TrainingProgressTrigger;
    metricsBreach @2 :MetricsBreachTrigger;
    workflowDispatch @3 :List(InputDef);
    custom @4 :CustomTrigger;
  }
}

struct RepositoryEventTrigger {
  eventType @0 :RepoEventType;
  pattern @1 :Text;
}

enum RepoEventType {
  clone @0;
  push @1;
  commit @2;
  merge @3;
  pullRequest @4;
  tag @5;
}

struct TrainingProgressTrigger {
  modelId @0 :Text;
  minStep @1 :UInt32;
}

struct MetricsBreachTrigger {
  metricName @0 :Text;
  threshold @1 :Float64;
}

struct CustomTrigger {
  topic @0 :Text;
  pattern @1 :Text;
}

struct InputDef {
  name @0 :Text;
  description @1 :Text;
  required @2 :Bool;
  default @3 :Text;
  inputType @4 :Text;
}

struct WorkflowRun {
  id @0 :Text;
  workflowId @1 :Text;
  status @2 :RunStatus;
  startedAt @3 :Int64;
  completedAt @4 :Int64;
  jobs @5 :List(JobRun);
}

enum RunStatus {
  queued @0;
  inProgress @1;
  success @2;
  failure @3;
  cancelled @4;
}

struct JobRun {
  name @0 :Text;
  status @1 :RunStatus;
  steps @2 :List(StepRun);
}

struct StepRun {
  name @0 :Text;
  status @1 :RunStatus;
  exitCode @2 :Int32;
}

# ═══════════════════════════════════════════════════════════════════════════════
# Worker Events (for EventService pub/sub)
# ═══════════════════════════════════════════════════════════════════════════════
#
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
