# Task attachment spike boundary record

**Status:** proposed, bounded implementation spike. This is neither a
production task scheduler nor an admission claim for the agent-OS design.

## Decision

This stack adds a narrow, fake-backed path from a locally verified attachment
operation grant to the existing Worker `/exec` projection. A `TaskId` remains a
reference, not an authority bearer. The authority for each Task operation is a
current, scoped attachment binding supplied by the caller at that operation;
the task record retains only correlation data and never stores a grant that a
later caller could borrow.

The changes deliberately keep identity, dispatch-MAC authorization, delegated
capability, namespace admission, and stream reachability as separate concerns.
In particular, a verified subject is not implicitly allowed to spawn a task.

## What the spike proves

| Boundary | Current behavior | Limit of the claim |
| --- | --- | --- |
| Attachment identity | On native builds, `VerifiedAttachment::from_envelope` accepts only the private provenance marker installed by verified RPC dispatch. Public callback and public signed-envelope context constructors are rejected. A 9P authorizer may carry an already-issued opaque grant only when its exact subject matches the verified attach identity. | The production 9P credential path issues no delegated Task grant. The local-root constructor and scoped grant issuer are test-only; identity, MAC state, and generic 9P write permission are not translated into one. |
| Operation authority | `AttachmentOperationGrant` is opaque and generation-bound. `TaskSpawn`, `TaskAttach`, `TaskSignal`, `TaskRead`, and `TaskPublish` are checked at their effect boundaries. | A production issuer must join a dispatch PEP/MAC decision and a distinct delegated-capability decision. Neither a dispatch MAC permit nor a JWT `cap` is silently treated as the other. |
| Revocation | The current caller grant is rechecked before and after awaited pool/stream state, and before output publication. A task record binds attachment id, generation, and subject but contains no reusable scope grant. | Revocation that wins before a fence stops the next effect. It cannot retrospectively cancel a backend call that already crossed its fence. Durable cancellation/recovery is deferred. |
| Task contract | `TaskService` supplies spawn, attach, signal, wait/result, and snapshot shapes over the existing `/exec` Worker projection. Argv spawn requires `TaskSpawn` plus `TaskPublish`; terminal signals require `TaskSignal` plus `TaskPublish`. | Child creation fails closed: no parent lineage or delegated attenuation is represented yet. |
| `/exec` projection | `Mount::walk_with_context` provides a default-compatible seam; `MountBackend` retains an authorizer-returned exact grant at attach and calls the dynamic override. `ExecMount` stores that context on its fid, so a same-subject legacy caller cannot list or walk an attachment-bound Task. | The production 9P credential path does not yet issue the dual-evidence grant, and there is no `/exec/clone` VFS node; `TaskService::spawn_task` is the explicit allocation seam. |
| Namespace field | `NamespaceManifestDigest` records a hash of caller-provided description bytes alongside the Task. | This is asserted contract metadata only. The Worker neither derives a `Namespace` nor an effective mount description from it, and does not supply it to `PodSandboxConfig`; it is not namespace admission or a sandbox-binding proof. |
| Result content | `ContentDigest` is content identity and `TaskContentRecord` is trusted-service observation/association metadata. Input association does not assert upstream production. | No output bytes are materialized into CAS, and public record fields are not signed or self-authenticating. A digest is not a retrievable artifact or independent provenance evidence. |
| fd carrier | The local fd adapter has bounded 9P reads, explicit local truncation indication, and matching terminal MoQ metadata; the regression consumes the local carrier to test its encoding. | The standalone origin exports no Iroh endpoint, subscriber, admission, MAC-key handoff, or client-reachable locator. `TaskReaches` advertises no MoQ topic until that exists. |

`TaskPayload::Content` remains in the forward contract but this Worker rejects
it before sandbox allocation. Accepting a digest without fetching and
materializing it would create a false-success running Task. Likewise, this
spike has no Fersh fixture or application integration.

## Effect and visibility rules

1. A grant is accepted only at a trusted boundary; normal callers cannot
   construct an attachment id, authority generation, lease, or operation grant.
2. `TaskAttachmentBinding` and `VfsOpContext` retain the caller's exact grant.
   A current `TaskRead` grant may read but cannot borrow the original spawn
   grant from the task record to signal through `ctl`.
3. An attachment-bound Task is hidden from an unbound or wrong-attachment
   lookup and `readdir`, rather than revealed as a permission-denied object.
4. A task-service error reports `StaleAttachment` only when rechecking the
   current binding shows revocation. A still-current, correctly scoped binding
   denied by another boundary, such as an armed lifecycle policy, reports the
   opaque `PermissionDenied` result instead.
5. Ctl responses are per fid and preserve ordinary offset/count reads. Stream
   ownership mutations are owner-checked with an entry generation, preventing
   replacement/ABA updates from one subject from changing another's topic.

The current lifecycle policy seam is intentionally not a claim that production
MAC enforcement is active. An armed policy is fail-closed; wiring the PEP and
its verified context to all production mount construction remains separate.

## Regression coverage in this spike

- callback and public signed-context construction cannot mint an attachment;
- cross-fid ctl response isolation and multi-chunk reads retain unread data;
- stream owner isolation and replacement-race owner mismatch behavior;
- context-bound Task invisibility to a same-subject legacy mount caller;
- verified 9P attach propagation to a context-aware dynamic `Mount`, exact-grant
  reattach binding (including scopes), and revocation before the next ctl effect;
- read-only and signal-only current grants cannot borrow wider spawn authority;
- revocation after an awaited fd read/publish boundary is denied;
- an armed lifecycle-policy denial remains `PermissionDenied`, not stale;
- failed or revoke-during-exec spawn rolls back the task record, fd state, and
  sandbox allocation;
- content payload rejection leaves no active sandbox; and
- bounded fd output visibly marks local truncation and sends matching terminal
  carrier metadata.

These are deterministic local/fake-backend regressions. They do not establish
Kata, HephVM, a live Iroh peer, a MoQ subscriber, browser/WASM task delivery,
or Fersh execution.

## Proposed review and merge stack

Every edge below is a hard dependency or shared-file dependency, so the PRs
are intentionally a linear stack. Review may proceed on all draft PRs, but
merge strictly from top to bottom:

```text
main
  -> codex/spike-rust-2024-gen-compat
  -> codex/spike-fid-ctl-isolation
  -> codex/spike-stream-owner-isolation
  -> codex/spike-exec-devfile-state
  -> codex/spike-verified-attachment-context
  -> codex/spike-standard-task-contract
  -> codex/spike-worker-attachment-vertical
  -> codex/spike-attachment-revocation-fences
  -> codex/spike-task-attachment-record
  -> codex/spike-ninep-vfs-context
```

| Layer | Branch | Review purpose |
| --- | --- | --- |
| 1 | `codex/spike-rust-2024-gen-compat` | Rust 2024 `r#gen` compatibility prerequisite. |
| 2 | `codex/spike-fid-ctl-isolation` | Per-fid service ctl reply state and chunked read behavior. |
| 3 | `codex/spike-stream-owner-isolation` | Subject-scoped stream visibility and atomic owner/generation mutation checks. |
| 4 | `codex/spike-exec-devfile-state` | `/exec` ctl state and non-fd offset/count behavior. |
| 5 | `codex/spike-verified-attachment-context` | Verified attachment provenance, scoped grants, and VFS operation context. |
| 6 | `codex/spike-standard-task-contract` | Neutral Task contract and explicit composite operation requirements. |
| 7 | `codex/spike-worker-attachment-vertical` | Fake Worker vertical slice, rollback, local carrier semantics, and honest reaches. |
| 8 | `codex/spike-attachment-revocation-fences` | Current-caller grant fencing, policy-denial taxonomy, and effect rechecks. |
| 9 | `codex/spike-task-attachment-record` | This boundary record and stack manifest. |
| 10 | `codex/spike-ninep-vfs-context` | Verified 9P attach to VFS operation-context propagation, exact-grant reattach fencing, and dynamic `ExecMount` retention. |

The stack must be rebased and revalidated at every head before merge. A future
production follow-up must add the production dual-evidence 9P grant issuer and
listener hand-off, namespace derivation/admission, durable
artifact materialization, and reachable authenticated stream handoff rather
than widening this local prototype by implication.
