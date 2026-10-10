# Federate authority-at-use core (not installed by production factories)

The admission core authenticates an enrolled local `service:oauth` caller and
uses the host-verified Dex `federated_claims` ATProto DID as the Policy
principal. The signed Dex `(issuer, opaque sub)` is retained only as source
provenance and durably bound to that DID; admission does not create or read a
local username/profile row. A deterministic host-scoped UUID is an internal
session-store key, not a second user identity. The separate hosted-account
`did:web` is not used as the person's identity or to discover a tenant.
Concrete requested scopes are intersected with fixed client/resource ceilings
and current `PolicyManager` checks under `(DID, tenant, resource, action)`. The
staging tenant is accepted only when exactly one existing tenant has a grant
for the requested staging scope; zero or multiple tenant matches deny. No
separate tenant-membership store is added. Prepare computes an
observed-decision digest; redemption recomputes the decision and compares
equality. The digest is not a global mutation epoch, and a later cross-store
mutation may still leave a stale short-lived session. No atomic
PostgreSQL/Casbin snapshot is claimed.

Admission still commits session and source replay atomically before a receipt;
there is no token signer. Its OAuth-only lookup/revoke boundary and 2-second
aggregate deadline remain unchanged. Tenant resolution examines current
tenant-scoped Policy grants and rejects missing or ambiguous matches; it does
not read or scan signed hosted-account records.

This source slice adds a fresh scope-ceiling and DID-scoped Policy check for one
exact Registry/Model operation, plus a PostgreSQL primary-record lookup that
binds the active session to its durable verified source DID. Each call first
performs a fresh, server-timed PostgreSQL lookup of session/source state, then
rechecks current DID grants (including explicit deny rules used for
suspension). The lookup fails closed unless schema v5 stores the source DID and
the durable one-to-one issuer/subject↔DID binding, and the replay
`(host,sid,client_id)` index is present. The per-use decision returns no bearer,
delegated service credential, or reusable permit. Once dispatch wiring exists,
each subsequent request or distinct tool call must invoke it again. This helper
adds no per-token, per-chunk, or timer polling; the request/tool handler wiring
and stream behavior remain unimplemented in this source slice.

The code remains private, uninstalled and default-deny: no RPC route, service
factory, proof/session constructor, client/host configuration, or runtime switch
is added. An ordinary deserialized `Session`, subject string, or browser field
cannot construct the store's `PrimaryRecord`; only the exact active lookup does.
The caller must still bind the full record to the verified host JWT and original
holder proof, then consume v16 request replay before protected dispatch.

## Remaining staged integration

1. Add and independently review the verified request-local session context:
   strict host-token binding, original-holder proof verification, full-record
   lookup/capability, and durable v16 request replay. Current production has no
   constructor for the evidence accepted by this core.
2. Add the authenticated Policy RPC operation and install its single
   authoritative decision owner only after the session store and current
   DID-scoped grant reads are qualified.
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

The `postgres` feature is only a compile/test boundary here. Source includes
version-3 bounded replay lookup, version-4 verified-source-DID, and version-5
one-to-one identity-binding migrations;
they are not run by the crate or applied to any database. No route or DB
connection is installed, and no DB role, AWS/GitLab setting, live DB, or runtime
state is changed.
