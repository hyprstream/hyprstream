# Tenancy spike findings — issue #1128

Date: 2026-07-21. Branch: `spike/1128-tenancy`. Method: evidence over argument — every
verdict below is backed by a runnable test committed on this branch, plus file:line
citations for the guard or gap. Gap-demonstration tests assert the *current broken*
behavior and are named `*_gap1128`; each carries a comment stating its assertion must be
inverted when the gap is fixed.

**Headline: the MoQ/announce plane is RED.** Cross-tenant announce enumeration and
unauthenticated relay publish are demonstrated end-to-end on loopback. Per the issue,
this blocks second-customer onboarding on shared hosting until follow-ups land.

## Verdict summary

| Plane | Verdict | Evidence |
|---|---|---|
| Casbin per-subject isolation over RPC | GREEN (baseline) | `two_subjects_per_subject_isolation_over_rpc` |
| Casbin *domain* isolation over RPC | RED — domains unwired | `domain_grant_is_inert_over_rpc_gap1128` |
| MoQ/announce plane (shared endpoint) | **RED — fail-open + unguarded relay** | `shared_endpoint_announces_cross_tenant_gap1128`, `relay_mode_client_publishes_without_authz_gap1128` |
| Per-Subject VFS (#715 Model B) | GREEN, one caveat | `two_sandboxes_are_isolated` (pre-existing), `subject_is_attribution_only_not_guard_gap1128` |
| Admission per-Subject counters | GREEN | `concurrent_two_subjects_no_cross_credit` |
| Event-prefix wrapped-key scoping (policy.capnp:317-357) | RED — domains inert + prefix takeover | `event_prefix_domain_grant_inert_gap1128`, `event_prefix_takeover_gap1128` |

## 1. Casbin domain isolation baseline — GREEN for subjects, RED for domains

What holds: dispatch gates every non-`$scopeExempt` method with `authorize(ctx, …)`
using the cryptographically verified subject, never caller-asserted identity
(`crates/hyprstream-rpc-derive/src/codegen/handler.rs:628-637`,
`crates/hyprstream-rpc/src/service/svc.rs:259-271`). Deny-by-default with deny-on-error
(`crates/hyprstream/src/auth/policy_manager.rs:450-453`).

Evidence: `two_subjects_per_subject_isolation_over_rpc`
(`crates/hyprstream/tests/policy_over_iroh.rs`) — two clients with distinct signing keys
resolve to distinct subjects over iroh; A (granted) succeeds, B (no grants) is denied
`Unauthorized`. PASS.

The gap: **no code derives a Casbin domain from DID/credentials.** Every production
authorize call hardcodes `domain: "*"` (`crates/hyprstream/src/services/registry.rs:1701`,
`crates/hyprstream/src/services/policy.rs:271`,
`crates/hyprstream/src/services/worker.rs:53`,
`crates/hyprstream/src/cli/policy_handlers.rs:321`). The matcher
`(p.dom == "*" || r.dom == p.dom)` (`policy_manager.rs:152-155`) is asymmetric: with
`r.dom="*"` pinned, a rule scoped to domain `tenant-a` can never match any RPC request.
Enforcer-level cross-domain denial is unit-proven (`test_mesh_tenant_isolation`,
`policy_manager.rs:1239`) but unreachable from the network.

Evidence: `domain_grant_is_inert_over_rpc_gap1128` — a grant
`(tenant-a-user, tenant-a, data:orders, read)` returns DENY over the RPC `check` path but
ALLOW when evaluated with domain `tenant-a` directly. PASS (asserts the broken behavior).

Secondary: `registry.rs:1733` — `handle_list` filtering does `unwrap_or(true)` (default
allow) when the policy service is unavailable; the primary authorize fails closed, the
list filter does not.

## 2. MoQ/announce plane — RED (spike-blocking)

The scoping primitive is sound but dead on the only live path. `tenant_scoped_consumer`
(`crates/hyprstream-rpc/src/moq_authz.rs:113-116`) correctly narrows an `OriginConsumer`
to `{tenant}/` (proven in-process, `moq_authz.rs:429-488`). But the quinn WebTransport
`/moq` serve path stamps every peer `PeerIdentity::anonymous()`
(`crates/hyprstream-rpc/src/transport/quinn_transport.rs:435`), so
`MoqAuthzConfig::tenant_for` returns `None`, and the `None => consumer` arm at
`quinn_transport.rs:457` **fails open to the process-global unscoped consumer**.
Production wiring installs `MoqAuthzConfig::default()` — no resolver at all
(`crates/hyprstream-service/src/service/spawner/service.rs:139-163`, TODO(#276)).
Relay mode is worse: `quinn_transport.rs:412-413` hands the session to
`moq_net::Server::with_origin` with no authz whatsoever. The iroh `moql` path refuses
every connection today (`crates/hyprstream-rpc/src/transport/iroh_moq.rs:202-212`,
anonymous always rejected), so its correct scoping logic (:218-245) is dead code; the
"defense lives on the moql path" claim at `quinn_transport.rs:226-227` is vacuous until
#1027 supplies wire identity.

Evidence (both PASS, demonstrating the leak on loopback,
`crates/hyprstream-rpc/tests/moq_tenancy.rs`):

- `shared_endpoint_announces_cross_tenant_gap1128` — server wired exactly like
  production; two independent clients dial the same `/moq` endpoint; **both** enumerate
  both `alice/streams/run-1/i0` and `bob/streams/run-9/i0`.
- `relay_mode_client_publishes_without_authz_gap1128` — an anonymous client announces
  `mallory/injected/i0` into a relay origin; the relay ingests and re-serves it to a
  second anonymous subscriber.

Scope of the leak: announce/broadcast **names** cross tenants (metadata enumeration),
plus moq-layer subscribe access to any announced broadcast and unauthenticated relay
publish. Frame *content* remains protected by the unguessable DH-derived topic + AEAD +
chained HMAC (relay is blind by construction). This is a real cross-tenant data leak on a
shared endpoint, not a policy bug. Because peers are anonymous at the transport, no
`tenant_resolver` could ever distinguish two subjects here — the gap is architectural;
the fix requires authenticating the `/moq` CONNECT (TODO already at
`quinn_transport.rs:441-446`).

## 3. Per-Subject VFS (#715 Model B) — GREEN, one caveat

The guard is structural: each sandbox gets a private forked `Namespace`, a private CoW
upper, and a private endpoint (`crates/hyprstream-workers/src/runtime/sandbox_fs.rs:105-155`);
"sandbox A's namespace contains only A's rootfs… A cannot name — let alone read — B's
tree" (module doc, `sandbox_fs.rs:28-36`). Path traversal clamps at `/`
(`crates/hyprstream-vfs/src/namespace.rs:728-748`). Model B FUSE mounts set
`allow_other(false)` (`crates/hyprstream-vfs-server/src/server.rs:250`), and the host
boundary is per-sandbox container rooting (`runtime/nspawn.rs:193`).

Evidence: pre-existing `two_sandboxes_are_isolated` (`sandbox_fs.rs:559`) passes
unmodified — two subjects, distinct images, no cross-visibility of files, streams, or
writes.

Caveat (demonstrated): `Subject` is attribution-only, not an enforcement token —
production `Mount` impls ignore `_caller: &Subject`
(`crates/hyprstream-vfs/src/injected.rs:113,326`). New test
`subject_is_attribution_only_not_guard_gap1128` shows tenant-B's `Subject` passed as
caller on tenant-A's namespace reads tenant-A's files. Isolation therefore rests entirely
on never sharing a namespace/endpoint; there is no per-op subject authorization as a
second line of defense. Also flagged: no subject→image binding at compose time
(`sandbox_fs.rs:105-120`) and no permission hardening on sandbox dirs/`vfs.sock` against
same-uid host processes.

## 4. Admission per-Subject counters — GREEN

Per-subject counters are individually keyed (`active_by_subject: HashMap<String, usize>`,
`crates/hyprstream-workers/src/runtime/admission.rs:321`); check-and-commit is atomic
under a single mutex in `try_admit_locked` (:386-453) — no TOCTOU window; release
decrements exactly the reserved record (:465-488); anonymous subjects are rejected
fail-closed (:653-659); group selectors can't consume another tenant's partition
(:511-546).

Evidence: `concurrent_two_subjects_no_cross_credit` — 64 concurrent tasks across
`tenant-a`/`tenant-b` against `max_per_subject: 4` on a multi-thread runtime: each
subject's observed in-flight peaked at exactly 4, all excess reserves returned
`AdmissionDenied`, counters drained to zero, and A's exhausted quota never affected B.
PASS.

Soft spots outside the counter: the key is the bare subject *name*, not the DID (name
normalization is the identity plane's problem), and quotas are node-local/in-memory by
design (#922/#925 pending for the cross-node ledger).

## 5. Event-prefix wrapped-key distribution (policy.capnp:317-357) — RED

Two distinct gaps:

1. **Per-tenant domains are structurally unenforceable on this plane.** The handlers
   enforce with hardcoded `r.dom="*"` (`crates/hyprstream/src/services/policy.rs:271`),
   and the asymmetric matcher means domain-scoped publish/subscribe grants never fire.
   Isolation degenerates to cluster-wide per-subject-string rules. Evidence:
   `event_prefix_domain_grant_inert_gap1128` — publish grant in domain `tenant-a` denies
   `RegisterEventPrefix` even for the granted subject. PASS (asserts broken behavior).
2. **Prefix takeover.** State is one flat process-wide map
   (`policy.rs:74`), and `handle_register_event_prefix` does an unconditional
   `prefixes.insert(...)` (`policy.rs:1123-1129`) with no ownership check (contrast
   `GroupKeyRegistry::register_group` rejecting duplicates,
   `crates/hyprstream-rpc/src/crypto/group_key.rs:390-392`). Evidence:
   `event_prefix_takeover_gap1128` — subject B, holding an ordinary dom="*" publish
   grant, re-registers A's prefix `orders`; a subsequent `SubscribeEventPrefix` returns
   B's publisher key and schema. PASS (asserts broken behavior).

The wrap crypto itself binds only `(subject DID, prefix)` via AAD
(`group_key.rs:436-442`, `crates/hyprstream-rpc/src/crypto/event_crypto.rs:194-217`) —
no tenant enters the derivation, so nothing cryptographic compensates for the policy
gaps. There were no pre-existing tests for any of the event-prefix handlers.

## Proposed follow-up issues (one per gap)

### Issue A — Authenticate the `/moq` WebTransport CONNECT and bind tenant scoping (BLOCKS second-customer onboarding)

> On a shared multi-tenant endpoint the `/moq` serve path stamps every peer
> `PeerIdentity::anonymous()` (`crates/hyprstream-rpc/src/transport/quinn_transport.rs:435`),
> `tenant_for` yields `None`, and `quinn_transport.rs:457` fails open to the unscoped
> `OriginConsumer`; relay mode (`:412-413`) has no authz at all. Demonstrated end-to-end:
> `crates/hyprstream-rpc/tests/moq_tenancy.rs`
> (`shared_endpoint_announces_cross_tenant_gap1128`,
> `relay_mode_client_publishes_without_authz_gap1128`) — any connected client enumerates
> every tenant's broadcast names and can publish into a relay origin.
> Fix: authenticate the CONNECT (client cert or app-level token; TODO at
> `quinn_transport.rs:441-446`), wire a real `tenant_resolver` in
> `crates/hyprstream-service/src/service/spawner/service.rs:139-163`, make the anonymous/
> unresolved case fail closed (drop the session) instead of `None => consumer`, and gate
> relay publish. When fixed, invert the two `*_gap1128` test assertions to per-tenant
> visibility. Refs: #1128, #276, #1027.

### Issue B — Derive the Casbin request domain from verified identity on the RPC path

> Every production authorize call hardcodes `domain: "*"`
> (`crates/hyprstream/src/services/registry.rs:1701`, `services/policy.rs:271`,
> `services/worker.rs:53`, `cli/policy_handlers.rs:321`), so domain-scoped policy rules
> can never match an RPC request (asymmetric matcher,
> `crates/hyprstream/src/auth/policy_manager.rs:152-155`). Demonstrated:
> `domain_grant_is_inert_over_rpc_gap1128` and `event_prefix_domain_grant_inert_gap1128`
> in `crates/hyprstream/tests/policy_over_iroh.rs`. Enforcer-level domain isolation is
> already proven (`test_mesh_tenant_isolation`) — the wiring is missing. Fix: thread a
> tenant/domain from the verified DID/JWT (pairwise presentation, #928) into
> `check_with_domain`, and consider domain-scoping role assignments (`g = _, _` is
> currently domain-less). Also make `registry.rs:1733` list-filtering fail closed.

### Issue C — Event-prefix registry: ownership check and tenant scoping

> `handle_register_event_prefix` unconditionally inserts into a flat process-wide map
> (`crates/hyprstream/src/services/policy.rs:74`, `:1123-1129`), letting any subject with
> a dom="*" publish grant silently hijack another tenant's prefix (publisher key,
> subscribers, and wrapped keys replaced). Demonstrated:
> `event_prefix_takeover_gap1128`. Fix: reject re-registration by a non-owner (match
> `GroupKeyRegistry::register_group`'s duplicate rejection,
> `crates/hyprstream-rpc/src/crypto/group_key.rs:390-392`), namespace prefixes per tenant
> (depends on Issue B), and bind the tenant into the wrap AAD
> (`crates/hyprstream-rpc/src/crypto/group_key.rs:436-442`).

### Issue D — VFS: per-op subject authorization inside a namespace (defense in depth)

> `Subject` is attribution-only: production `Mount` impls ignore `_caller`
> (`crates/hyprstream-vfs/src/injected.rs:113,326`). Isolation currently rests solely on
> per-sandbox namespace privacy — sound today
> (`crates/hyprstream-workers/src/runtime/sandbox_fs.rs:559`), but any future path that
> serves two subjects through one namespace (e.g. the `from_namespace` sharing path,
> `sandbox_fs.rs:169-181`) has no second line of defense. Demonstrated:
> `subject_is_attribution_only_not_guard_gap1128`. Fix: enforce per-op subject checks at
> the namespace layer (the test-only `SubjectGatedMount` shows the trait supports it),
> add subject→image binding at compose time (`sandbox_fs.rs:105-120`), and harden sandbox
> dir/`vfs.sock` permissions.

## Reproduce

```sh
export LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 \
  LD_LIBRARY_PATH=$HOME/.local/lib/python3.14/site-packages/torch/lib \
  CARGO_TARGET_DIR=$PWD/target
cargo test -p hyprstream --test policy_over_iroh
cargo test -p hyprstream-rpc --test moq_tenancy
cargo test -p hyprstream-workers --features oci-image --lib -- runtime::admission runtime::sandbox_fs
```

All suites green at commit time (gap tests green by asserting current broken behavior).
