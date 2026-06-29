@0xf1c2d3e4a5b6c7d8;

# Custom annotations for MCP tool generation and documentation.
#
# These annotations are extracted at build time and used to generate
# descriptive MCP tool definitions automatically.

# Unified authorization action vocabulary (S3, epic #547).
#
# This is the SINGLE action set shared by:
#   - the `$scope`/`$capability` schema annotation (compile-time baseline TE policy)
#   - the runtime `hyprstream_rpc::auth::Scope` action field
#   - the runtime `hyprstream::auth::Operation` enum (1:1 — see Operation::from_capability)
#
# There is no second, parallel vocabulary. Adding an action here is the only way
# to introduce a new authorization verb; the runtime enum mirrors this list.
#
# Each action is designed to map 1:1 onto a future UCAN `cmd` (epic #547 §11).
# Invalid action = capnp compile error (type-safe enumerant).
enum ScopeAction {
  # Read-only / side-effect-free (9p read = no side effects). TE object-class: "read".
  query      @0;  # read status/state/list           (UCAN cmd: /query)
  # Mutating / authority actions.                       TE object-class: "write".
  write      @1;  # create/update/delete persistent   (UCAN cmd: /persist)
  manage     @2;  # administrative / policy / lifecycle(UCAN cmd: /manage)
  infer      @3;  # run model inference               (UCAN cmd: /infer)
  train      @4;  # train/fine-tune                   (UCAN cmd: /train)
  serve      @5;  # serve via API surface             (UCAN cmd: /serve)
  context    @6;  # context-augmented generation      (UCAN cmd: /context)
  subscribe  @7;  # subscribe to stream/notification  (UCAN cmd: /subscribe)
  publish    @8;  # publish/broadcast to subscribers  (UCAN cmd: /publish)
  spawn      @9;  # spawn a process/task              (UCAN cmd: /spawn)
  create     @10; # create a resource                 (UCAN cmd: /create)
  # Inference-mesh authority actions (#319) — distinct from model actions so a
  # model grant can never satisfy a mesh-authority gate.
  meshInvoke @11; # umbrella mesh invoke right         (UCAN cmd: /mesh/rpc)
  meshStage  @12; # submit activation/stage to peer    (UCAN cmd: /mesh/infer/stage)
  meshDelta  @13; # submit TTT delta to peer/aggregator(UCAN cmd: /mesh/delta/submit)
}

# MCP tool description - used for method-level documentation
annotation mcpDescription(field, union, group, struct, enum, enumerant) :Text;

# Parameter description - used for field-level documentation in MCP tools
annotation paramDescription(field) :Text;

# Scope / capability annotation — the compile-time baseline TE policy + node
# object-label (S3, epic #547). MANDATORY on every interface method: a method
# with no `$scope`/`$capability` is a BUILD ERROR (no silently-public files).
#
# The annotated action is, simultaneously:
#   - the JWT/Casbin scope action the caller must hold,
#   - the node's TE object-class (read vs write — see ScopeAction comments),
#   - the (subject-type, op) the node permits = a baseline type-enforcement relation.
#
# Usage: `foo @0 :Req $scope(write);` — invalid action is a capnp compile error.
# `$capability` is an accepted alias (identical semantics); prefer `$scope`.
annotation scope(field)      :ScopeAction;
annotation capability(field) :ScopeAction;

# Explicit, audited exemption from mandatory scope (S3, epic #547).
#
# The ONLY way a method may legitimately carry no `$scope`/`$capability`. Use it
# strictly for methods that cannot require authorization without circularity, e.g.
# the PolicyService authz check itself, or a method gated by a different mechanism
# (envelope/CA attestation) documented inline. A reason string is mandatory so the
# exemption is reviewable. Absence of BOTH `$scope` and `$scopeExempt` = build error.
annotation scopeExempt(field) :Text;

# Mark as deprecated with reason
annotation deprecated(field, union, struct, enum) :Text;

# Example value for documentation
annotation example(field) :Text;

# Hide a method from the CLI (internal-only methods like session management, streaming auth)
annotation cliHidden(field) :Void;

# Fixed-size constraint for Data fields — generates [u8; N] instead of Vec<u8>.
# Usage: serverPubkey @2 :Data $fixedSize(32);
annotation fixedSize(field) :UInt32;

# VFS usage example for man pages. Complete usage scenario, not a field value hint.
# Usage: $docExample("ctl /srv/registry clone '{\"url\": \"...\"}'")
annotation docExample(field) :Text;

# Mark a field as optional in MCP tool schemas.
# Optional fields use type-appropriate defaults when absent (0 for numbers, "" for text, [] for lists).
annotation optional(field) :Void;

# Domain type path — generated client returns this Rust type via FromCapnp::read_from().
# Usage: struct Foo $domainType("runtime::VersionResponse") { ... }
annotation domainType(struct) :Text;

# Serde field rename — generates #[serde(rename = "...")] on the Rust field.
# Usage: toolType @1 :Text $serdeRename("type");
annotation serdeRename(field) :Text;
