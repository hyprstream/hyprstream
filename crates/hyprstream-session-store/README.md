# Policy session store — H1/H3a disabled source foundation

This crate is deliberately **not installed by any runtime factory**. It is a native
PostgreSQL adapter with a typed internal API, separated from the inference-linked
application so its real database tests need no GPU/ML build. It adds no account
store, token signer, HTTP route, pool, TLS bypass, or static enrollment changes.
The existing FileBackedSessionRegistry remains untouched.

## Trust boundary and next integration

Only Policy may construct `Admission` after resolving the admitted account,
signed tenant and authoritative grants. OAuth is the trusted source-token and
hybrid possession verifier under the reviewed A0 contract. This crate checks
storage invariants, expiry and generation, **not** JWTs, key validity, nonce
commitments, signatures or account authority. Input types intentionally have no
Serde deserialization. Public Rust construction is not authenticated RPC admission.

A future Policy adapter must use the admitted TLS/role-scoped PostgreSQL connector,
enforce bounded pool/operation deadlines, authenticate its dedicated OAuth caller,
and compare the entire challenge binding and current account/grant revision. It
must check active status again before signing and never sign on an ambiguous
commit. Request handlers must compare lookup results to credential claims and
verify v16 proof; a successful lookup is not an RPC authorization decision.

H1 exposes admit, full active lookup, binding comparison, revoke and cleanup only.
The `hyprstream/postgres` feature includes a private Policy reader, with an optional
provider defaulting to None and no production installer. The internal
`resolveSessionPrimary` RPC is MAC/scope guarded, additionally verifies a registered
service key, and requires an explicit service/key/tenant/resource capability.
No policy-template permission was added. H2 Policy/account plumbing,
H3 async v16 lookup/revocation consumers and H4 OAuth validation/issuance remain
separate changes; naming follows the original A0 host slices.

## Explicit migration assets (not applied to staging)

Existing repository convention: SQL assets plus opt-in disposable PostgreSQL tests
(see `hyprstream-ledger/sql` and `tests/postgres_live.rs`). `001_admission.sql` is a
one-time, transactional version-1 migration, followed by `002_primary_lookup.sql`
(version 2); no runtime boot migration and no
silent repeat/downgrade. It creates only `federate_session` tables. `roles.sql`
contains explicit grants; provisioning creates these dedicated NOLOGIN roles:

```sql
CREATE ROLE hs_policy_runtime NOLOGIN;
CREATE ROLE hs_profile_control NOLOGIN;
CREATE ROLE hs_session_cleanup NOLOGIN;
CREATE ROLE hs_session_migration NOLOGIN;
```

The migration identity owns the schema/tables and needs CREATE on the selected
existing database. Version 2 requires disabled profiles and an empty sessions table;
it cannot backfill legacy collision provenance. Drain/expire legacy sessions first,
then apply both migrations and the current role grants explicitly. Runtime/control/cleanup identities must not
inherit migration ownership, superuser, PUBLIC rights or any broader role. Runtime
has SELECT plus column-scoped INSERT (excluding proof_epoch) on sessions, INSERT on
replay, USAGE on the epoch sequence, UPDATE(status) only on sessions, and
SELECT+UPDATE(lock_version) only on profile state. Control has SELECT and
UPDATE(enabled,authority_generation,collision_inventory_id). Inventory changes
require generation rotation in the same row update; a trigger rejects otherwise.
Cleanup alone has scoped DELETE rights.
No runtime role can seed a profile, enable it, rotate it or change grants/keys.
The operator migration role can seed the exact host/profile with a fresh random
32-byte generation and a complete inventory ID; `enabled` defaults false. Missing profile denies. This crate does not
seed or enable any deployment profile.

Admission locks the exact profile row with FOR UPDATE in READ COMMITTED, checks
its generation and enabled state, reads wall-clock time **after** lock acquisition,
and inserts session/replay atomically. All three source replay indexes remain
unique across generations. Control UPDATEs contend on the same row. Any conflict
rolls back the new session. Commit precedes any returned receipt; errors suppress
PostgreSQL detail strings to avoid logging parameters. Cancellation/unknown commit
must not trigger a token retry; caller requires fresh login.

Replay retention is max(source_iat+330, source_exp+30, session_expiry); cleanup deletes
expired replay before sessions past expiry+30 with no remaining replay reference,
in one transaction. Both component uniqueness constraints remain effective until
that cleanup, including revoked sessions. Revoke does
not delete replay protection. No perpetual tombstones. Ordinary restart preserves
committed rows; database/connection errors never fall back to memory. Silent full
snapshot rollback is not detectable: follow the accepted A0 stop/clear challenges/
rotate generation/invalidate all sessions/wait >330 seconds/reopen procedure. No
external fence is introduced.

## H3a primary provenance and consumption boundary

Every admitted session gets a positive immutable database identity `proof_epoch`.
It is not caller-selected and ordinary reconnect preserves it. Full lookup returns
`ActiveSessionPrimary` only when schema/profile/suite/status/expiry, generation and
session/profile inventory IDs match. `PrimaryLookup` compares the complete expected
credential binding (including hybrid content thumbprint), configured inventory and
explicit caller capability. Its SQL transaction and operation deadline are two
seconds; concurrency is bounded with no positive cache. Historical grant revision
does not authorize any operation.

`CollisionInventory::from_sources` consumes **trusted public configuration input**,
checks the complete declared source set and bounds/canonicalizes it. The caller
must enumerate every configured static and foreign-protocol source and propagate
read/parse failure; absence is not an empty success. No live filesystem/key loader
or secret reader is installed in H3a. Wiring that complete loader, role-scoped TLS
pool and explicit capabilities remains a reviewed integration gate; arbitrary
Rust construction is not cryptographic provenance. This storage layer still relies
on admission's upstream key/possession validation; H3b must validate cryptographic
records before proof consumption.

No H3b resolver, cached-activation path, signing/issuance, stream gate or H2 grant
decision is implemented here. A database epoch reload test is not a test of v16
request replay: H3b must exercise two verifier processes with the same durable
replay authority and show duplicate rejection after restart, plus denial when
history is missing. New epochs must never be used to bypass lost history.

## Focused tests

Run every Cargo command through the fleet BuildQ wrapper. Test only a disposable
local PostgreSQL 16 cluster; no remote DSN is accepted by the test harness. It
creates its own database/NOLOGIN roles and removes them even if a scenario panics.
Use an otherwise empty dedicated cluster because role names are fixed.

```sh
HS_SESSION_TEST_SOCKET=/path/to/disposable/pg/socket \
  /home/birdetta/.codex/skills/hyprstream-build/scripts/hyprstream-build.sh run -- \
  cargo test -p hyprstream-session-store --test postgres_live -- --ignored --nocapture
```

The live fixture is explicitly ignored without opt-in, not silently counted as a
passing DB test. It exercises actual roles, SELECT-only lock failure, restricted
control privileges, two-source races, each replay index, atomic rollback, full
binding/expiry/generation, revoke, reconnect, connection loss, retention/cleanup,
and both admission-vs-disable/rotation orderings. Test-only advisory locks and a
trigger provide barriers; `pg_blocking_pids` proves blocking, rather than sleeps.
Removing production FOR UPDATE must fail the causal ordering assertion.
