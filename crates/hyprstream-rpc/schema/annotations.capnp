@0xae3b014583213580;

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
# Ordinals are grouped into contiguous SEMANTIC BLOCKS (S3, #569). The ordering is
# intentional and meaningful — read-class first, then resource/model authority, then
# the highest-privilege admin gate, then the cross-host mesh-authority block — so the
# numeric ordinal alone tells you the privilege class. The runtime mirror
# `hyprstream::auth::Operation` is kept 1:1 with this ordinal layout (variant order +
# as_capability/from_capability). This is a CLEAN BREAK: there is no wire-compat with
# the previous flat numbering; codegen resolves `$scope(name)` back to the enumerant
# NAME (not the ordinal), and the runtime `Scope`/`Operation` are keyed on those names,
# so the names are the stable contract and the ordinals are free to be re-grouped.
enum ScopeAction {
  # ── Block A: read-class — side-effect-free (9p read = no side effects).
  #            TE object-class: "read". Cheap to grant to a group.
  query      @0;  # read status/state/list           (UCAN cmd: /query)
  subscribe  @1;  # subscribe to stream/notification  (UCAN cmd: /subscribe)
  # ── Block B: write/authority-class — mutating or capability-bearing actions on
  #            models/resources. TE object-class: "write". Least-privilege per-node.
  write      @2;  # create/update/delete persistent   (UCAN cmd: /persist)
  create     @3;  # create a resource                 (UCAN cmd: /create)
  publish    @4;  # publish/broadcast to subscribers  (UCAN cmd: /publish)
  infer      @5;  # run model inference               (UCAN cmd: /infer)
  train      @6;  # train/fine-tune                   (UCAN cmd: /train)
  context    @7;  # context-augmented generation      (UCAN cmd: /context)
  serve      @8;  # serve via API surface             (UCAN cmd: /serve)
  spawn      @9;  # spawn a process/task              (UCAN cmd: /spawn)
  # ── Block C: admin authority — highest-privilege lifecycle/policy gate.
  manage     @10; # administrative / policy / lifecycle(UCAN cmd: /manage)
  # ── Block D: inference-mesh authority (#319) — host↔host pipeline rights, distinct
  #            from model actions so a model grant can never satisfy a mesh gate.
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
# Struct-level: the generated client returns this Rust type for the whole struct.
#   Usage: struct Foo $domainType("runtime::VersionResponse") { ... }
# Field-level: the field maps to this Rust newtype (wire type unchanged, e.g. Text).
#   Usage: serviceDid @0 :Text $domainType("hyprstream_rpc::identity::Did");
annotation domainType(field, struct) :Text;

# Serde field rename — generates #[serde(rename = "...")] on the Rust field.
# Usage: toolType @1 :Text $serdeRename("type");
annotation serdeRename(field) :Text;

# ── VFS / 9p projection (epic #539) ───────────────────────────────────────
# The generated 9p/VFS file surface projected from each service's method
# structure. Captured at build time (T1, #540); consumed by the mount
# generator (T2, #541). These are metadata only — declaring them does not
# emit a mount by itself.

# Node kind a method projects to in the generated mount.
enum VfsNodeKind {
  file   @0;   # readable (maybe writable) leaf
  dir    @1;   # directory (readdir)
  ctl    @2;   # write-a-verb control file
  stream @3;   # /stream-style pipe (data/info/ctl)
  query  @4;   # synthetic clone/query file
}

# Mount-relative path a method binds to (fid-0 rooted, NEVER absolute).
# `{braces}` mark param segments that bind method args.
# Usage: getByName @3 :Text $vfsPath("{name}");
annotation vfsPath(field, struct) :Text;

# Override the inferred VFS node kind for a method.
# Usage: clone @4 :CloneRequest $vfsKind(ctl);
annotation vfsKind(field) :VfsNodeKind;

# Read path returns raw mmap bytes, bypassing capnp (DMA hot path).
annotation vfsBulk(field) :Void;

# Exclude a method from the generated mount.
annotation vfsHidden(field) :Void;

# MAC security label this generated node carries — carrier (a) of #699: the label
# is a property of the node's TYPE, derived from the schema annotation at
# generation (T2 emits it onto VfsNode). Format: "<level>" or
# "<level>:<assurance>" or "<level>:<assurance>:<compBit,compBit>". Absent ⇒ the
# generated node is unlabeled ⇒ deny ⇒ a genesis-coverage finding (there is no
# permissive default). Usage: getStatus @0 :Status $vfsMac("internal");
annotation vfsMac(field) :Text;
