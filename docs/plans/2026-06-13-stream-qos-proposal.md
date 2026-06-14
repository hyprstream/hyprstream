# Proposal: Schema-Declared Stream QoS (#213)

**Status:** draft for review · Part of epic #131 (M2), Phase A1 · 2026-06-13
Supersedes the "Stream QoS" section of `2026-06-12-m2-streaming-plane.md`.

## Summary

A stream's delivery + integrity contract — anywhere from **durable & ordered** to
**out-of-order media** — is declared **in the schema** on the streaming method, and
**codegen realizes a matched producer + consumer** from that one declaration, so the two
ends can never silently disagree on QoS or integrity. Integrity enforcement is
**policy-selected and fail-closed**. The common schema defines only the **vocabulary**
(the policy type + annotation); applications declare their **policy values** locally.

## Motivation

- We need both **durable-ordered** streams (reliable token streams, audit/training logs:
  in-order, no loss, resumable) and **out-of-order media** streams (event/telemetry
  fan-out: skip-to-live, lossy) over moq — chosen **per application use case**.
- moq deliberately delivers Groups **out-of-order, with gaps, evicting by age** (it's
  built for media). A consumer written for in-order/lossless assumptions is unsafe on it
  (#163: stale-Group replay, silent truncation). The correct integrity check therefore
  **depends on the declared contract** — there is no single right verifier.
- Producer and consumer must **agree** on QoS + integrity or you get silent corruption or
  a security downgrade. Making the contract a single schema declaration, realized by
  codegen on both ends, makes disagreement unrepresentable.

## Design

### 1. Vocabulary (common schema) — the policy type system

Each axis is a `union` of `(void | group)` so a strategy carries its own parameters
(strongly typed; capnp ordinal-evolution gives wire forward/back compat).

```capnp
# Ordering + (media) replay window. ordered = strict (gap fatal, chained MAC).
struct Ordering {
  union {
    ordered @0 :Void;
    unordered :group { replayWindow @1 :UInt32; }   # media: reject seq <= last-seen - window
  }
}
# Delivery guarantee. atLeastOnce carries dedup + resume params.
struct Delivery {
  union {
    atMostOnce @0 :Void;
    atLeastOnce :group { dedupWindow @1 :UInt32; resumable @2 :Bool; }
  }
}
# Truncation policy — its OWN axis (terminal frames are opt-in; inference uses none).
struct Completion { union { terminal @0 :Void; none @1 :Void; } }
# Relay-side retention window (late-join / resume). liveOnly = smallest surface (#174).
struct Retention { union { liveOnly @0 :Void; groups @1 :UInt32; seconds @2 :UInt32; } }
# Backpressure on publish saturation. block = lossless (EAGAIN contract).
struct Backpressure { union { block @0 :Void; dropOldest :group { highWater @1 :UInt32; } } }

# The full per-stream contract, composed of the axes above.
struct StreamContract {
  ordering     @0 :Ordering;
  delivery     @1 :Delivery;
  completion   @2 :Completion;
  retention    @3 :Retention;
  backpressure @4 :Backpressure;
}

# Applied to a streaming method's StreamInfo response variant.
annotation streamPolicy(field) :StreamContract;
```

### 2. Naming (consistency-grounded)

Assessed against the repo's existing annotations and capnp semantics:
- **Form:** every existing annotation is a lowerCamelCase noun/adjective naming the
  metadata kind (`fixedSize`, `domainType`, `mcpScope`, `serdeRename`, `cliHidden`). →
  the annotation is the noun **`streamPolicy`** (verb/"with-" forms rejected as inconsistent).
- **Annotation ≠ type identifier** is already the norm (`mcpScope :ScopeAction`,
  `domainType :Text`) — and capnpc-rust *enforces* it (a struct and annotation that
  snake_case to the same module collide). → the type is **`StreamContract`**: distinct
  identifier, self-descriptive (the stream's delivery contract), no `Qos` abbreviation, no
  collision, and no overlap with the `Delivery` axis word.
- **No `$Rust.name` escape hatch** is used in the repo; introducing one just to reuse a
  spelling would itself be inconsistent. Resolve by naming.

### 3. Placement: vocabulary common, values application-local

- **Vocabulary** (the axes + `StreamContract` + `streamPolicy` annotation) lives in one
  shared, importable schema so there is a single annotation node (one ID, matched by name).
  *(Lives in `streaming.capnp` — with the streaming types it decorates; `annotations.capnp` is for cross-cutting directives only.)*
- **Policy values are NOT standardized centrally.** Each application declares its policy
  where it's used — inline at the call site, or an app-local `const :StreamContract` if
  reused within that app:

  ```capnp
  # inline (most application-tied):
  streamInfo @0 :StreamInfo $streamPolicy((
    ordering = (ordered = void),
    delivery = (atLeastOnce = (dedupWindow = 256, resumable = true)),
    completion = (none = void), retention = (groups = 256), backpressure = (block = void)));

  # or reused within one app's schema (e.g. inference.capnp):
  const tokenStream :StreamContract = ( ... );
  streamInfo @0 :StreamInfo $streamPolicy(.tokenStream);
  ```
  We bless a named policy as "generic" only once real call sites prove the grouping.

- **Corrected codegen rationale (this was previously stated wrong):** codegen does **not**
  require the vocabulary to be common. The annotation extractor is **generic over capnp
  value kinds with no hardcoded type knowledge**, and `capnpc` resolves the annotation
  value through `using import` from *anywhere* and embeds it into each schema's CGR. So
  placement is an **organizational** choice (single annotation ID, uniform interpretation),
  not a hard constraint.

### 4. Schema-modeling: fixed struct + capnp evolution

`StreamContract` is a **fixed struct, one field per axis** (not an open
`List(PolicyFacet)`): strongly typed, total/simple resolution (the type guarantees
exactly-one-of-each axis). Disciplines this commits us to:
- **Ordinal hygiene** — add fields/variants only; never renumber/reuse/retype (codegen-gated).
- **Explicit-or-safe defaults** — codegen *requires every axis to be set* for a compile-time
  policy (absence = build error). Where capnp *can* zero-fill (wire structs / a policy
  advertised to an external peer), the **`@0` variant is the strict/fail-closed default**
  (ordering=ordered, completion=terminal, backpressure=block, retention=liveOnly).
  Security-bearing strictness is carried as union **variants** (unknown variant →
  `NotInSchema`, detectable/rejectable), not bare fields (silently zero-defaulted).
- If federated cross-version policy *negotiation* is ever needed (#168), wrap the struct in
  `List(PolicyFacet)` + criticality bits then — a non-breaking evolution.

### 5. Codegen pipeline (the real mechanism)

1. `capnpc` compiles each schema → CGR, with the `$streamPolicy` value fully resolved and
   embedded (imports resolved from wherever the vocabulary lives).
2. **Build-time extractor (`hyprstream-rpc-build`) must be extended** to handle
   **struct-valued annotations**. Today `extract_value_json` (lib.rs:219) handles only
   `Text/Bool/UInt32/Enum/Void`; a struct value falls through to presence-only
   (`value: true`), dropping the policy. The extractor must **recurse into struct values**
   (union `which()` discriminant + nested group fields) into the metadata JSON. *This is
   the core of #216 — it corrects the earlier "just lift the annotation" framing, and it's
   generic (works for any struct-valued annotation, not just `streamPolicy`).*
3. **Resolver validation (fail-closed):** reject contradictory/unimplemented combos at
   build time (e.g. `dropOldest` + `atLeastOnce`; `completion=terminal` on an `atMostOnce`
   live stream) with a clear error.
4. **Codegen (Rust derive + `ts_codegen`)** reads the resolved policy and emits a *matched*:
   - **producer/service stub** → moq Track config: retention depth, terminal emission, MAC mode;
   - **client/consumer** → `StreamHandle`/`StreamVerifier` in the matching mode.
   Any policy the runtime can't realize → `compile_error!`. One source of truth ⇒
   producer/consumer cannot silently mismatch.

### 6. Policy-selected, fail-closed integrity (generalizes #163)

The `ordering` axis picks the MAC scheme; `completion` picks the truncation check; **any
checkable violation terminates the stream** (fail-closed, matching #160's RPC posture).

| | `ordering = ordered` | `ordering = unordered` (media) |
|---|---|---|
| Group sequence | `seq == expected_next`; gap = **fatal** | gaps **allowed** (skip-to-live) |
| MAC scheme | chained `prevMac`, `epoch+seq` bound in | **per-Group** MAC over `(track ‖ epoch ‖ seq ‖ payload)` — self-authenticating |
| MAC verify fail | **fatal** | **fatal** |
| Replay (`seq ≤ last-seen`) | caught by chain | **fatal** (reject within `replayWindow`) |
| Eviction-by-age | bounded by `retention` floor | expected |

Orthogonal, set by **`completion`**: `terminal` → require a `Complete`/`Error` payload
before EOF (EOF-without-terminal = **fatal**, truncation defense); `none` → EOF accepted
(truncation not detectable — the explicit choice for inference/live). Media uses **per-Group
self-authentication** so a deliberate gap doesn't break verification.

### 7. Wire / schema change — `seq` (not `groupSeq`)

`StreamBlock` (`streaming.capnp`) gains a **producer-assigned, transport-neutral** sequence
(`moq_stream.rs:286` already assigns it: `next_group += 1` → handed to moq as the Group id, so
it is *ours*, not moq's — the name `groupSeq` was misleading). It is **per key-epoch**:
```capnp
seq   @N :UInt64;   # producer-assigned, monotonic per epoch; = the moq Group id on the moq transport
epoch @M :UInt64;   # key-epoch (see §8); bumps on re-key / producer restart
```
- ordered: `MAC = HMAC(key_epoch, prevMac ‖ epoch ‖ seq ‖ payload)`
- media:   `MAC = HMAC(key_epoch, track ‖ epoch ‖ seq ‖ payload)`

One `StreamBlock` shape across policies; only the MAC input + consumer enforcement differ.
Works on non-moq transports too (we author `seq`; moq merely mirrors it as its Group id).
(Binding `seq` changes the bytes vs the legacy ZMQ frame, so Phase-A differential testing
compares **semantics**, not raw bytes.)

### 8. Security properties of the sequence

The monotonic `seq` is the **anti-replay / anti-reorder / anti-truncation** primitive — and a
liability if misused. Invariants:
1. **`(key, epoch, seq)` must be unique forever.** A producer restart that reset `seq` under the
   same key would re-open accepted offsets *and* (if `seq` feeds a nonce) reuse nonces. → re-key
   (bump `epoch`) on restart / periodically; never wrap `u64` under one key (non-issue at u64).
2. **AEAD nonce derived deterministically** as `PRF(stream_key, epoch ‖ seq)` — eliminates the
   on-wire nonce field and guarantees uniqueness from invariant 1 (closes counter-as-nonce reuse).
3. **MAC always binds track/topic + epoch + seq** so a frame can't be replayed into another stream
   or another epoch at the same offset.
4. **Count/rate is a relay'd-path metadata side channel** — `seq` *is* the moq `group_id`, which the
   relay must read (see §9). Accepted and documented; QoS-invariant (applies to ordered **and**
   media — media's skip-to-live actively needs `latest()`). Not hideable without forgoing a moq relay.
5. **Strict ordering is a cheap DoS lever** — `ordering=ordered` (gap = fatal) lets an attacker who
   can drop one Group kill the stream. A documented availability-vs-integrity trade the policy selects.
6. `seq` carries **no secrecy** (fully predictable); never use it as a capability/authz token, and
   gate historical seek/replay by access control (#174).

### 9. What the relay sees / where trust rests (threat model)

- **Content + integrity-ordering:** end-to-end via the shared **keys** (MAC + AEAD); the relay is
  blind — it never verifies MACs or reads payloads.
- **Topic:** a **DH-*negotiated* identifier, not a secret.** It is the moq Track name the relay
  routes by → visible to the relay. Its DH derivation only prevents pre-negotiation squatting; it is
  **not** access control. Security rests on the keys, not topic secrecy.
- **Numeric sequence:** **visible** — moq-lite needs a monotonic, comparable `group_id` for
  `latest()`, `start_group` resume, priority, and drop-ranges (`publisher.rs:434`, `priority.rs:37`,
  `subscribe.rs:221`). So an encrypted/opaque sequence would break moq-lite; the relay is blind to
  *meaning*, not to the *counter it sorts by*.
- **Escape hatches** (if count-privacy is a hard requirement, this is a *transport/topology* choice,
  not a QoS-axis one): direct transport (QUIC encrypts the group headers from the network), or an
  oblivious-relay mode that forgoes moq-lite's absolute-offset resume/latest.

## Phase A1 build breakdown

| # | Task | Notes |
|---|---|---|
| **#215** | Vocabulary schema (axes + `StreamContract` + `streamPolicy` annotation) | **built** (annotation `streamPolicy`, type `StreamContract` per §2) |
| **#216** | Extractor: **struct-valued annotation extraction** + resolver metadata + fail-closed validation | core = recurse struct/union/group in `extract_value_json` (§5.2) |
| **#217** | Codegen (Rust derive): policy → matched producer/consumer + build gate | §5.4 |
| **#218** | Codegen (`ts_codegen`): policy → matched browser producer/consumer | §5.4 (M4) |
| **#219** | `StreamBlock.seq` + `epoch` + bind into MAC (producer + verifier) | §7–8 |
| **#220** | StreamService → moq Origin: flip primary, delete rejoin buffer | #134 M2a substrate |

**A2** (separate breakdown): the two integrity *enforcement* modes (§6) + adversarial tests
(ordered: reorder/gap/drop-terminal must reject; media: gaps accepted, forgery/replay rejected).

## Decisions locked
- Schema-declared QoS realized by codegen on both ends; fail-closed.
- Raw axes (parameterized union-of-groups), **fixed struct** (`StreamContract`), capnp ordinal-evolution.
- `completion` is its own axis (terminal opt-in); `@0` = fail-closed default.
- Naming: annotation `streamPolicy`, type `StreamContract`.
- Vocabulary common, **policy values application-local** (no central named policies).
- Integrity is **policy-selected** (ordered chained-MAC vs media per-Group), `epoch+seq` bound into the MAC.

## Resolved
- **Placement (was open):** the vocabulary lives in **`streaming.capnp`** (with `StreamInfo`
  it decorates + `StreamBlock` #219 touches; app schemas already import it). `annotations.capnp`
  stays for cross-cutting codegen/doc directives only. Not per-app (types must be shared).

## Open questions
1. **Replay/dedup window units** — `groups` count (current) vs time? (Leaning groups, ties to retention.)
2. Whether `ts_codegen` (#218) lands in A1 or defers to the M4 browser phase.

## Deferred / related
- **Durable retention is best-effort under moq-lite alone** (recency cache). A persistent
  Origin backend (Kafka-grade replay from an arbitrary in-retention offset) is tracked in
  **#222**, along with the **over-declared-retention behavior** decision (fail-closed at
  registration vs clamp + advertise effective retention) — which may add a `retention`
  sub-flag here (`guaranteed` vs `bestEffort`). Consumer-group load-balancing → #135;
  log compaction → #177; partitioning → a Track-naming convention.
