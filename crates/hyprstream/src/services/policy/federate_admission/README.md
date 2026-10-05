# Disabled H2 Policy admission core

Compiled under `hyprstream/postgres`, with no route, runtime factory, production
evidence constructor or enablement switch. Default construction denies. This is
the internal account/Policy/storage boundary, not end-to-end login or H2 dispatch.

The core requires a verified local `service:oauth` envelope whose holder matches
the authority-owned enrollment entry. It uses admitted UserStore external binding
and immutable account UUID, requires explicitly active account and a signed hosted
DID record, and intersects concrete requested scopes with fixed client/resource
ceilings and existing PolicyManager decisions. No auto-provisioning or new account
or grant store. Unsupported wildcard scopes deny rather than broadening authority.

Prepare returns an opaque observed-decision revision. Redemption repeats the read
and requires the complete decision to match. This revision hashes the observed
decision and its inputs; it is NOT a global mutation epoch or proof no mutation
occurred and reverted. No cross-store atomic snapshot is claimed. A mutation after
the read but before H1 commit can leave a stale short-lived session. That accepted
race requires fresh authorization at every protected request and distinct tool
call; an already-authorized stream may finish, without token/chunk/timer polling.

H1 commits session/replay atomically before returning a receipt; no token signing
occurs here. Lookup/revoke are OAuth-only in this slice, bounded and host-pinned.
H3's separately reviewed consumer-capability lookup is not replaced or broadened.
Each operation has a capacity permit without queuing and a single aggregate two
second deadline. Uncertain DB commit denies; no local replay fallback.

Fresh tenant lookup bypasses the shared index, reads only exact account-record
paths across at most 256 tenants, and rejects missing/ambiguous/corrupt records.
Its authority remains the admitted PDS mount; a deployment with an independently
cached/stale mount is not qualified by this implementation. Timeout/capacity are
enforced by the Policy caller. There is no change to existing cached callers.

## Required integration before enablement

- H4 strict source/dual-possession verifier and approved typed evidence constructor,
  binding exact challenge, keys, source, scopes, client and resource. Offline
  fixtures are not cryptographic verification evidence.
- Wire the internal operation into authenticated Policy RPC, preserving its enrolled
  OAuth restriction, with bounded decode and audited operation permission. No
  browser registration DTO or verification boolean is introduced here.
- Admit writer-endpoint UserStore/session connections, authoritative signed-record
  mount and one serving Policy instance. No deployment or schema apply in this PR.
- H3 full-record capability adapter, request-local proof integration, cached
  activation, durable request replay; H2 fresh request/tool authority gate. These
  remain absent. A receipt is neither a reusable authorization permit nor a token.

All Cargo and hooks use BuildQ. Focused filters: `h2_admission` on the application
with `postgres`, and `h2_current_tenant` on `hyprstream-pds-service`. The ignored
application PG fixture requires an explicitly supplied local `H2_TEST_SOCKET` and
creates/drops its own random database; it never uses a deployment DSN. Its accepted
race test proves stale admission and denial on the next authority read, NOT live
Registry/Model handler isolation. H1's separate durability suite covers store
concurrency/roles/cleanup; do not attribute those fixtures to live H2 RPC.
