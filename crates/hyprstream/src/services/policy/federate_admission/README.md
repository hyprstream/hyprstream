# Disabled Federate authority-at-use core

The earlier disabled H2 admission core remains unchanged: it authenticates an
enrolled local `service:oauth` caller, resolves the upstream identity through
the admitted UserStore, requires the same immutable account UUID, an explicitly
active account, and a signed hosted-DID record, and intersects concrete requested
scopes with fixed client/resource ceilings and current `PolicyManager` checks.
It does not auto-provision an account or add an account/grant store. Prepare
computes an observed-decision digest; redemption recomputes the whole decision
and compares equality. The digest is not a global mutation epoch, and a later
cross-store mutation may still leave a stale short-lived session. No atomic
PostgreSQL/PDS/Casbin snapshot is claimed.

Admission still commits session and source replay atomically before a receipt;
there is no token signer. Its OAuth-only lookup/revoke boundary and 2-second
aggregate deadline remain unchanged. Uncached tenant resolution reads exact
signed account-record paths across at most 256 tenants and rejects missing,
ambiguous or corrupt records; it does not replace the existing cached caller.

This source slice adds a fresh current-account, signed-tenant, scope-ceiling,
and Policy check for one exact Registry/Model operation, plus a PostgreSQL
primary-record lookup that binds the active session to its unique durable source
identity. The per-use decision returns no bearer, delegated service credential,
or reusable permit. Once dispatch wiring exists, each subsequent request or
distinct tool call must invoke it again. This helper adds no per-token,
per-chunk, or timer polling; the request/tool handler wiring and stream behavior
remain unimplemented in this source slice.

The code remains private, uninstalled and default-deny: no RPC route, service
factory, proof/session constructor, client/host configuration, or runtime switch
is added. An ordinary deserialized `Session`, subject string, or browser field
cannot construct the store's `PrimaryRecord`; only the exact active lookup does.
The caller must still bind the full record to the verified host JWT and original
holder proof, then consume v16 request replay before protected dispatch.

## Remaining staged integration

1. H3a/H3b must add and independently review the verified request-local session
   context: strict host-token binding, original-holder proof verification,
   full-record lookup/capability, and durable v16 request replay. Current main has
   no production constructor for the evidence accepted by this H2 core.
2. Add the authenticated Policy RPC operation and install its single authoritative
   decision owner only after writer-endpoint account/session access, exact signed
   tenant reads and committed-policy reload semantics are qualified.
3. Wire the returned one-request decision through authenticated RPC Registry and
   Model dispatch, then require a fresh check immediately before each distinct
   tool call. Deny this profile on unclassified 9P, in-process, HTTP, CLI and
   background paths. Model→Inference stays inside the same logical Model request
   and carries the original caller/session context; it does not become a new
   authority or reusable delegated credential.
4. Add dispatch-isolation, tool-pre-effect, stream-continuation, cross-replica,
   and real post-ack revocation tests before any enablement. A prior authorized
   stream may finish; a new request/tool call after the mutation acknowledgement
   must deny. No token/chunk/timer polling is permitted.

The `postgres` feature is only a compile/test boundary here. No route or DB
connection is installed by this patch. No migration, schema, DB role, AWS/GitLab
setting, live DB, or runtime state is changed.
