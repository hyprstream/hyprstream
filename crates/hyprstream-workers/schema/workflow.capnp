@0xa1b2c3d4e5f60718;

using import "/annotations.capnp".mcpDescription;
using import "/annotations.capnp".paramDescription;
using import "/annotations.capnp".mcpScope;

# Cap'n Proto schema for workflow service (independent service)
#
# WorkflowService manages workflow definitions, dispatching, and event
# subscriptions. Operates at the orchestration level (not node-scoped).
#
# Convention: Request variants use camelCase names. Response variants
# use the same name suffixed with "Result" to avoid Cap'n Proto naming
# collisions.

struct WorkflowRequest {
  id @0 :UInt64;
  union {
    scanRepo @1 :Text
      $mcpDescription("Scan a repository for workflow definitions")
      $mcpScope(query);
    register @2 :WorkflowDef
      $mcpDescription("Register a workflow definition")
      $mcpScope(write);
    list @3 :Void
      $mcpDescription("List all registered workflows")
      $mcpScope(query);
    dispatch @4 :DispatchRequest
      $mcpDescription("Dispatch a workflow run with input parameters")
      $mcpScope(write);
    subscribe @5 :SubscribeRequest
      $mcpDescription("Subscribe to workflow events")
      $mcpScope(write);
    unsubscribe @6 :Text
      $mcpDescription("Unsubscribe from workflow events")
      $mcpScope(write);
    getRun @7 :Text
      $mcpDescription("Get status of a workflow run")
      $mcpScope(query);
    listRuns @8 :Text
      $mcpDescription("List runs for a workflow")
      $mcpScope(query);
  }
}

struct WorkflowResponse {
  requestId @0 :UInt64;
  union {
    error @1 :ErrorInfo;
    scanRepoResult @2 :List(WorkflowDef);
    registerResult @3 :Text;
    listResult @4 :List(WorkflowInfo);
    dispatchResult @5 :Text;
    subscribeResult @6 :Text;
    unsubscribeResult @7 :Void;
    getRunResult @8 :WorkflowRun;
    listRunsResult @9 :List(WorkflowRun);
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

# =============================================================================
# Workflow Types
# =============================================================================

struct WorkflowDef {
  path @0 :Text $paramDescription("Workflow definition file path");
  repoId @1 :Text $paramDescription("Repository ID");
  name @2 :Text $paramDescription("Workflow name");
  triggers @3 :List(EventTrigger);
  yaml @4 :Text $paramDescription("Workflow YAML content");
}

struct WorkflowInfo {
  id @0 :Text;
  name @1 :Text;
  path @2 :Text;
  repoId @3 :Text;
  enabled @4 :Bool;
}

struct DispatchRequest {
  workflowId @0 :Text $paramDescription("Workflow ID to dispatch");
  inputs @1 :List(KeyValue);
}

struct SubscribeRequest {
  workflowId @0 :Text $paramDescription("Workflow ID to subscribe to");
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
