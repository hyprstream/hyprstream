# HyprStream system ontology

**Status:** Normative. Adopted through issue [#1098](https://github.com/hyprstream/hyprstream/issues/1098) from the reviewed ontology epic [#1091](https://github.com/hyprstream/hyprstream/issues/1091). Changes require the governance process below.

HyprStream simplifies by one repeated move: replace parallel surfaces with one capability-scoped vocabulary and a layer law. This document is the repository source of truth for that vocabulary. It distinguishes current implementation from partially converged and target architecture; a target entry is a reviewed direction, not a claim that its type exists today.

Status terms have these meanings:

- **CURRENT:** the canonical form exists in the repository and is the form new work uses.
- **PARTIAL:** the canonical direction exists, but competing representations or incomplete enforcement remain.
- **TARGET:** the design is ratified, but its canonical form is not yet embodied in the repository.

## Governance and precedence

The ontology-change owner is **@ewindisch**, as repository architecture owner. A change to a canonical noun, action, law, layer assignment, reserved name, or the machine-readable manifest requires a pull request approved by that owner. If the owner authors the pull request, another repository maintainer must approve it. The pull request must identify affected consumers and ratified decisions, and must update the human-readable section and manifest together.

For ontology facts, this document takes precedence over descriptive summaries elsewhere in the repository, including `CLAUDE.md`. It does not silently supersede narrower ratified decisions. These remain authoritative within their scopes and are incorporated here by reference:

- `CLAUDE.md` **Naming (don't relitigate)** owns the settled MAC/RBAC spellings.
- ADR [#651](https://github.com/hyprstream/hyprstream/issues/651) owns `Namespace::bind_mount` over `Arc<dyn Mount>` as the single composition primitive, with overlay as a policy and legitimate leaf/down-adapters preserved.
- Networking epic [#1090](https://github.com/hyprstream/hyprstream/issues/1090), post-review revision 7, owns **Promote** as the durable STEP operation and **admit** as ephemeral run admission.

Changing one of those decisions requires explicitly amending its source and this document in the same change or dependency chain. Silence is not supersession.

The JSON manifest in [Machine-readable registry](#machine-readable-registry) is the extraction source for generated artifacts. The tables explain the registry to humans. A disagreement between them is an invalid change and must not merge.

## Canonical nouns

| Noun | Status | Canonical form | Absorbs or forbids |
|---|---|---|---|
| **File / Mount** | CURRENT | `hyprstream_vfs::Mount` in a `Namespace` | Bespoke resource APIs that could be files; sibling protocol planes that should be mounted. |
| **Namespace** | CURRENT | `hyprstream_vfs::Namespace` plus `bind_mount`, with `Subject` on calls; see ADR #651. | Global mount tables, server-dictated paths, and a second composition primitive. Leaf and down-adapters remain legitimate. |
| **Identity** | CURRENT | Durable `hyprstream_rpc::identity::Did`; the standard string form at serialization boundaries. Runtime principal `hyprstream_rpc::envelope::Subject`, derived at each trust boundary from verified evidence rather than by one mechanical Did-to-Subject conversion. | Untyped DID parameters inside trusted Rust APIs, `user: String` principals, and names or node locations treated as identity. `Did::new` remains deliberately lenient; strict validation occurs when data enters a trust boundary. |
| **Locator** | CURRENT | `hyprstream_rpc::transport::TransportConfig` and the wire `hyprstream_rpc::stream_info::Destination`, carried as untrusted reach descriptors associated with resolved identity. | Locators treated as identity, authority, or proof of reachability. |
| **Capability** | CURRENT | `hyprstream_rpc::auth::ucan::Capability` chain, lowered to `hyprstream::mac::compiled::SignedPolicy`, then enforced through the PDP/AVC pipeline. | Ambient authority, configuration as membership, caller-supplied clearance, and a scope string treated as terminal authority. |
| **Label** | CURRENT; enforcement PARTIAL | `hyprstream_rpc::auth::mac::SecurityLabel`. | Plaintext labels treated as authoritative contract fields, or transport/hardware tier conflated with assurance. |
| **Artifact** | PARTIAL | Inter-crate immutable address `hyprstream_rpc::cid::Cid`; Hybrid-PQC COSE for security-critical artifacts; Git semantics for versioned artifacts. | Bare, non-self-describing digests crossing crate boundaries. Store-native hashes remain valid internally and retain algorithm identity. |
| **Envelope** | CURRENT | `hyprstream_rpc::envelope::SignedEnvelope` with composite COSE signing. | Transport trust as the application trust root and unsigned cross-trust-boundary messages. |
| **Stream** | CURRENT | MoQ track plus `hyprstream_rpc::stream_info::StreamOpt`. | Parallel event planes and independently re-derived QoS vocabularies. |
| **Task** | TARGET (#829) | A zero-ambient-authority unit of running work. An isolation boundary is where a Task runs; Job/Step are bounded workflow roles; Pane/App are views. | Unqualified service/worker/sandbox/job/session terms used as interchangeable architectural work units. Qualified domain terms remain valid. |
| **Attachment** | PARTIAL | The relationship produced by `attach`: credentials verified once, then an opaque handle plus generation/epoch identifies cached trusted context. | Per-operation credential walks and joins that cannot be revoked. |
| **Segment** | TARGET (#1090) | `SegmentId` equals cid512 of the genesis grant. A Segment is Namespace times Capability applied to networking; v0 intentionally has no segment machinery. | Subnets or VNIs treated as membership, and asserted topology treated as verified reachability. |

Derived concepts are not peer nouns. A **Model** is a versioned Artifact repository: `ModelRef` is its mutable/versioned address, an artifact CID identifies immutable content, and an instance identifier names a runtime instance; the near-vestigial `ModelId(Uuid)` projection is to be removed. Adapter, Policy, Token, Event, and Plan/Partition are likewise defined from the canonical nouns rather than establishing parallel identity or authority systems.

## Canonical actions

| Action family | Meaning | Discipline |
|---|---|---|
| **resolve / reach** | Resolve identity to verified facts and determine verifiable reachability. | Locators are untrusted; accepted facts and freshness evidence are verified. |
| **grant / attenuate** | Author and narrow authority through UCAN. | Delegation only narrows; zero standing privilege is the posture. |
| **attach / revoke** | Join a scoped resource; invalidate that relationship. | Verify credentials at attach, mediate through cached context, and revoke by generation/epoch. Grant revocation also relies on short TTLs. |
| **move** | Transport bytes over an attached path. | Trust rides in the signed envelope or artifact, not in the carrier. |
| **place** | Select eligible and preferred candidates. | Use the #523 filter, rank, select, and explain substrate; do not fork selection vocabularies. |
| **partition** | Shape a workload across selected resources. | Use pure value types; a partitioner does not assert trust, reach, or placement facts. |
| **admit** | Issue evidence-gated, ephemeral run clearance. | Run admission is `admit(plan)`; microbatch admission is `PipelineJobRecovery::admit`. Both expire with evidence and are never called promote. |
| **run / recover** | Execute and recover with fenced failure semantics. | Use epochs/generations and fail closed. |
| **promote** | Advance a durable artifact in the STEP loop. | Reserved for Stage, Train, Evaluate, Promote; human- or agent-API-gated and represented by durable filesystem/Git state. |
| **audit** | Append a signed decision record. | If a required record cannot be durably audited, deny. |
| **compile** | Lower authored authority into an enforceable policy. | This is an S5 transformation, not a lifecycle action or synonym for admit/promote. |

## The eight laws

1. **Everything is a file.** A resource that can be a Mount must be one. ADR #651's bind-mount union is the composition mechanism; adapters at the boundary do not create a second composition system.
2. **Identity over location.** DIDs identify. Transports and locators are replaceable, untrusted carriers.
3. **Two gates, not one.** A capability controls attachment or reachability: without its grant, a namespaced or Segment resource is unaddressable. Authorization through Casbin/MAC controls operations on anything addressable: a public RPC endpoint may be addressable and then denied. Neither gate substitutes for the other.
4. **The layer law.** Facts flow upward. A component consumes verified facts from lower layers and must not assert those facts itself.
5. **Attach-time credentials, complete mediation.** Verify credentials once at attach, mediate every operation through cached trusted context and the AVC, and revoke by generation. “Never per-op” forbids repeated credential verification, not the #547 complete-mediation invariant.
6. **Fail closed; sign what is security-critical.** Unverifiable input floors or denies. Hybrid-PQC COSE is mandatory for security-critical artifacts and cross-trust-boundary messages such as grants, policies, audit records, envelopes, and descriptors. Ordinary immutable content derives integrity from CID or Git structure and does not require a redundant signature.
7. **One vocabulary per concept; at most two representations.** A concept may have its canonical wire/string representation and one typed internal representation. A new surface reuses them or amends this document.
8. **Names are scoped.** Reserved names are governed by this document and the ratified naming sources it references; a similar operation in another scope does not silently reuse a reserved word.

## Module-aware layer map

Layers describe ownership of facts and behavior, not crate boundaries. A crate can span layers when its modules have explicit responsibilities. Generated projections and adapters do not acquire authority merely because they expose a lower-layer type.

| Layer | Current crates and modules | Boundary rule |
|---|---|---|
| **TRUST** | `hyprstream-rpc::{crypto, auth, identity, envelope}`, `hyprstream-crypto`, `hyprstream/src/{mac,auth}`, and `hyprstream-service::service::trust_store`. `hyprstream-pds::{at9p_gate,at9p_chain}` currently hosts TRUST verification inside an otherwise STORE crate. | Own verified identity, claims, capability, labels, signing, policy compilation, and decisions. The PDS placement is a layer-law finding: relocate it toward RPC/auth or explicitly preserve and annotate the boundary. |
| **REACH / CONNECT** | `hyprstream-discovery`, locator/resolution modules in `hyprstream-p2p` and `hyprstream-rpc`, and TARGET Segment machinery from #1090. | REACH produces verified endpoint/freshness facts. CONNECT owns capability-scoped membership. Neither treats a locator or topology hint as proof. |
| **MOVE** | `hyprstream-rpc::{transport,dial,moq_stream,moq_event}`, the 9P wire codec and translator in `hyprstream-9p`, and guest/RDMA data-plane implementations. | Move bytes and preserve signed context; do not decide identity, placement, or membership. `hyprstream-rpc` therefore spans TRUST and MOVE by module. |
| **SURFACE** | `hyprstream-vfs` canonical composition, `hyprstream-vfs-server` down-adapter, mount adapters in `hyprstream-9p`, and `hyprstream-rpc-std::vfs_mount`. | Present resources through Namespace/Mount. `hyprstream-9p` spans MOVE for wire/translation and SURFACE for adapters. Preserve ADR #651's leaf-adapter exception. |
| **STORE** | `git2db`, `git-xet-filter`, `cas-serve`, storage modules in `hyprstream-pds`, `hyprstream/src/storage`, `hyprstream-containedfs`, and durable ledger modules in `hyprstream-ledger`. | Persist native formats behind canonical inter-crate identifiers. A store does not establish identity or authorization by possession alone. |
| **WORK** | Spawner/factory/manager modules in `hyprstream-service`, `hyprstream-workers` and engine crates, `hyprstream/src/services`, and workload orchestration in `hyprstream-k8s`. | Create and host authorized work without inventing identity, authority, or placement semantics. Trust-store modules remain TRUST even when housed in a WORK crate. |
| **SHAPE / ADMIT / RUN** | `hyprstream/src/{runtime,training,inference}`, workflow/runtime modules in worker crates, and the interhost pipeline after #1090's rescope. | SHAPE/PARTITION consumes an ordered candidate set; ADMIT combines current verified evidence; RUN executes and recovers. Lower-layer facts never originate here. |
| **EDGE** | `hyprstream/src/{api,server}`, `hyprstream-tui`, `hyprstream-compositor`, `waxterm`, `chat-core`, `hyprstream-flight`, and external query/export surfaces in `hyprstream-metrics`. | Parse, render, and adapt at trust boundaries. Derive trusted context at entry rather than flattening canonical types. `chat-core` currently has no workspace dependents, so any claim that it owns a canonical shared type remains TARGET until consumers use it or another survivor is chosen. |
| **BUILD / PROJECTION** | `hyprstream-rpc-build`, `hyprstream-rpc-derive`, generated schema projections in `hyprstream-rpc-std`, and repository tooling. | Generate or validate another layer's vocabulary. Generated projections do not become a second normative vocabulary. |
| **SHARED LEAF** | `hyprstream-util` at present. | Shared code must be concept-named and must not become a domain grab-bag or assert facts for dependent layers; the concept-named-leaf target is tracked by #1091 R9. |

The networking specialization orders its operational facts as TRUST, REACH, CONNECT, MOVE, PLACE, PARTITION, then ADMIT/RUN. REACH facts feed PLACE; PLACE output feeds PARTITION; all meet at ADMIT. No edge runs downward by assertion—for example, a partition must never manufacture a `lan_id` reachability fact.

## Naming registry and reconciliation

The canonical registry adds only facts not owned by the referenced ratified sources:

- **Session**, when unqualified in architecture, means a stateful Attachment. Qualified standards/domain terms such as OAuth session or editing session remain valid.
- **Cid** is the only inter-crate immutable digest noun. Native hash types remain internal and algorithm-explicit.
- **Subject** is a verified runtime principal and is never constructed as authority from an unverified string.
- **Resolve** names the resolver family; individual endpoint, identity, record, membership, and label roles remain distinct when their input, output, or trust semantics differ.

The MAC/RBAC spellings remain single-homed in `CLAUDE.md` under **Naming (don't relitigate)** and are not copied here. Worker-engine crate naming remains single-homed in the `CLAUDE.md` crate map. Promote/admit semantics remain owned by #1090 revision 7. Namespace composition remains owned by ADR #651.

## Machine-readable registry

Consumers extract the strict JSON object between the markers. The marker names and `schema_version` are compatibility surfaces; changing them is an ontology change. JSON object ordering is not significant, but entry `id` and action `token` values are stable identifiers.

<!-- BEGIN SYSTEM-ONTOLOGY-MANIFEST -->
```json
{
  "schema_version": 1,
  "status_values": ["CURRENT", "PARTIAL", "TARGET"],
  "types": [
    {
      "id": "file_mount",
      "name": "File / Mount",
      "status": "CURRENT",
      "rust_types": ["hyprstream_vfs::Mount"]
    },
    {
      "id": "namespace",
      "name": "Namespace",
      "status": "CURRENT",
      "rust_types": ["hyprstream_vfs::Namespace"],
      "decision": "https://github.com/hyprstream/hyprstream/issues/651"
    },
    {
      "id": "identity",
      "name": "Identity",
      "status": "CURRENT",
      "rust_types": [
        "hyprstream_rpc::identity::Did",
        "hyprstream_rpc::envelope::Subject"
      ],
      "wire_form": "DID string"
    },
    {
      "id": "locator",
      "name": "Locator",
      "status": "CURRENT",
      "rust_types": [
        "hyprstream_rpc::transport::TransportConfig",
        "hyprstream_rpc::stream_info::Destination"
      ]
    },
    {
      "id": "capability",
      "name": "Capability",
      "status": "CURRENT",
      "rust_types": [
        "hyprstream_rpc::auth::ucan::Capability",
        "hyprstream::mac::compiled::SignedPolicy"
      ]
    },
    {
      "id": "label",
      "name": "Label",
      "status": "CURRENT",
      "rust_types": ["hyprstream_rpc::auth::mac::SecurityLabel"],
      "implementation_note": "production enforcement remains PARTIAL"
    },
    {
      "id": "artifact",
      "name": "Artifact",
      "status": "PARTIAL",
      "rust_types": ["hyprstream_rpc::cid::Cid"],
      "wire_form": "CID multihash"
    },
    {
      "id": "envelope",
      "name": "Envelope",
      "status": "CURRENT",
      "rust_types": ["hyprstream_rpc::envelope::SignedEnvelope"]
    },
    {
      "id": "stream",
      "name": "Stream",
      "status": "CURRENT",
      "rust_types": ["hyprstream_rpc::stream_info::StreamOpt"],
      "wire_form": "MoQ track"
    },
    {
      "id": "task",
      "name": "Task",
      "status": "TARGET",
      "rust_types": [],
      "decision": "https://github.com/hyprstream/hyprstream/issues/829"
    },
    {
      "id": "attachment",
      "name": "Attachment",
      "status": "PARTIAL",
      "rust_types": [],
      "wire_form": "opaque handle plus generation or epoch"
    },
    {
      "id": "segment",
      "name": "Segment",
      "status": "TARGET",
      "rust_types": [],
      "wire_form": "SegmentId = cid512(genesis grant)",
      "decision": "https://github.com/hyprstream/hyprstream/issues/1090"
    }
  ],
  "actions": [
    {"token": "resolve", "family": "resolve_reach", "kind": "lifecycle"},
    {"token": "reach", "family": "resolve_reach", "kind": "lifecycle"},
    {"token": "grant", "family": "grant_attenuate", "kind": "lifecycle"},
    {"token": "attenuate", "family": "grant_attenuate", "kind": "lifecycle"},
    {"token": "attach", "family": "attach_revoke", "kind": "lifecycle"},
    {"token": "revoke", "family": "attach_revoke", "kind": "lifecycle"},
    {"token": "move", "family": "move", "kind": "lifecycle"},
    {"token": "place", "family": "place", "kind": "lifecycle"},
    {"token": "partition", "family": "partition", "kind": "lifecycle"},
    {
      "token": "admit",
      "family": "admit",
      "kind": "lifecycle",
      "scopes": ["run", "microbatch"]
    },
    {"token": "run", "family": "run_recover", "kind": "lifecycle"},
    {"token": "recover", "family": "run_recover", "kind": "lifecycle"},
    {
      "token": "promote",
      "family": "promote",
      "kind": "lifecycle",
      "reserved_scope": "STEP"
    },
    {"token": "audit", "family": "audit", "kind": "lifecycle"},
    {"token": "compile", "family": "compile", "kind": "transformation"}
  ]
}
```
<!-- END SYSTEM-ONTOLOGY-MANIFEST -->

## Dialect atlas

Atlas measurements are valid only when produced by the committed `tools/ontology-sweep.sh` at a pinned Git SHA. That script is not present in this change; issue [#1100](https://github.com/hyprstream/hyprstream/issues/1100) owns it, the observability row, and shrinkage metrics. The provisional observations below come from the coordination sweep associated with pinned snapshot `66b622cbd7a8e3c4d295279254f66c2bd57cabd7` (2026-07-18). **Every count is pending #1100** and must be replaced or confirmed by an in-repository run before an atlas-only number is treated as normative.

| Family | Provisional observation at `66b622cbd7a8e3c4d295279254f66c2bd57cabd7` | Collapse target |
|---|---|---|
| **WHO** | 18 bare-string DID parameter occurrences in `placement_index.rs`; approximately 6 Subject-family types; model identity in 3 forms — **pending #1100**. | Typed `Did` internally and string at boundaries; `Subject` derived per boundary from verified evidence; `ModelRef` survives as the mutable/versioned model address. |
| **MAY-DO** | `Operation` has 15 variants while `all()` returns 7 and omits 8; approximately 10 PEPs/facades; `#[authorize]` appears in 1 service-handler file — **pending #1100**. JWT scopes are retired compatibility surface, not a rival authority system. | One UCAN to Casbin to compiled policy to MAC/AVC authority pipeline. Protocol PEPs retain orthogonal visibility, quota, DPoP, and discretionary checks while composing the mandatory floor. Generate action projections from the schema source. |
| **RESOLVE** | 9 `*Resolver` traits plus 4 equivalent-role contracts with other suffixes, 13 total candidates for semantic classification — **pending #1100**. | One resolver family with distinct endpoint, identity/key/reach, and selected-service roles. Collapse only contracts with equivalent input, output, and trust semantics. |
| **DIAL** | 7 conceptual entry points across RPC/MoQ, native/WASM, injected VFS, and crypto-store variants — **pending #1100**. | A shared target/options/evidence vocabulary, not necessarily one literal function. |
| **DIGEST** | At least 8 digest/address framings across CID, Git, XET, OCI, P2P, and CAS families — **pending #1100**. | CID/multihash at crate boundaries; native hashes stay internal and algorithm-explicit. The CasSubstrate crate-layering decision remains open in #1091. |
| **SURFACE** | `ToolCallFormat` in 3 definitions; `ModelConfig` in 2 different concepts; training configuration in at least 4 projections; Session types in at least 4 forms; `Spawnable` has 1 trait definition — **pending #1100**. The apparent OpenAI triple is one surface split across DTO, handler, and hosting layers. | Merge genuine duplicates, rename same-named different concepts, and preserve legitimate adapters/projections. `chat-core` becomes a survivor only when actual consumers use it. |
| **RUNNING WORK** | 4 sandbox abstractions and approximately 4–5 session meanings — **pending #1100**. | TARGET Task glossary from #829. Preserve qualified OAuth/editing/workflow meanings; reserve unqualified architectural Session for Attachment. |
| **OBSERVABILITY** | No reviewed count exists — **pending #1100**. | Measure metrics, Flight, and OpenTelemetry surfaces before choosing any collapse target. |

Security-relevant findings are tracked independently rather than hidden inside a vocabulary count: the MCP stdio failed-token-to-anonymous downgrade is #1095; the live XET `AllowAllCasAuthorizer` is #1094; the `Operation::all()` omission is #1096; and DID contract reconciliation is #1097.

The quarterly sweep is owned by **@ewindisch**. Each atlas update must record the full SHA, exact `ontology-sweep.sh` invocation, tool version/schema version, result, and any excluded or unavailable measurements in its pull request.
