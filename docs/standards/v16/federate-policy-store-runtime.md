# Federate Policy store runtime prerequisites

This packet prepares two separate PostgreSQL connections for the Federate
Policy service. It does not install Federate admission or enable the profile.
The current Policy factory still leaves its Federate readers disabled.

The migration owner applies `hyprstream-session-store/sql/001_admission.sql`,
`002_epoch_inventory.sql`, and `003_primary_lookup_index.sql` in order, then
the matching `roles.sql` and `002_roles.sql` grants. These files never run at
application startup. The migration owner provisions a dedicated login that
inherits only `hs_policy_runtime`; the profile-control and cleanup roles are
separate. A missing schema, version other than 3, excessive runtime rights,
or unavailable database must fail startup before a reader is installed.

The Policy account reader has a different login from the OAuth account writer.
It needs SELECT on `users`, `oidc_bindings`, `user_did_bindings`, and
`user_encryption_keys`, plus the existing deployment DEK unwrap capability.
It must have no account-table mutation, schema CREATE, or database CREATE
rights. Its constructor does not run the account-store migration.

Both connections consume password-bearing PostgreSQL URLs from projected
files, with `sslmode=verify-full` and a pinned CA file. Paths are:

| Reader | URL file path variable | CA file path variable |
|---|---|---|
| Policy account | `HYPRSTREAM_POLICY_ACCOUNTS_URL_FILE` | `HYPRSTREAM_POLICY_ACCOUNTS_SSLROOTCERT_FILE` |
| Policy session/replay | `HYPRSTREAM_FEDERATE_SESSION_URL_FILE` | `HYPRSTREAM_FEDERATE_SESSION_SSLROOTCERT_FILE` |

Do not reuse the OAuth account writer URL for Policy. Do not put a URL or
password in TOML, a CI variable expanded into a command line, or a log.

The remaining activation packet must construct the trusted full collision
inventory, exact serving generation and fixed host/client/scope profile; open
the signed hosted-account mount with a mandatory MAC audit sink; and install
the Policy readers and authenticated OAuth/Registry/Model paths together.
Until all of those inputs exist, Federate Policy RPC handlers deny by default.

## Policy↔OAuth RPC packet (source-only)

OAuth and Policy run in separate processes. Policy's private `Decision` is
never accepted back from a browser-supplied tuple. The source-only RPC schema
adds `prepareFederateSession` and `commitFederateSession`, both requiring the
direct enrolled OAuth service caller, while the production provider slot stays
`None`:

1. OAuth generates the random challenge ID and supplies it with its verified
   source metadata, canonical requested scopes, creation time, and the
   source-nonce-committed Ed/PQ public keys. Policy reads current account,
   signed tenant and grants under the fixed profile. It returns an opaque
   random handle and the authoritative public challenge binding. Policy keeps
   the source, challenge, keys, decision, generation, inventory and expiry
   behind the handle in a bounded in-memory map. The handle never goes to the
   browser. A Policy restart drops pending handles and therefore fails closed;
   there is no reconstruction from client fields.
   The current staging IaC declares one EC2 instance and one Policy container,
   not a multi-replica Policy service. If that topology changes, a commit sent
   to a different replica will deny; add affinity or a scoped durable pending
   store before scaling. Verify the live instance and Policy socket readiness
   before enabling this path.
2. OAuth binds that handle to the one-use challenge and verifies **both**
   holder signatures over the exact challenge. The browser cannot supply or
   edit the handle's account, tenant or grant contents.
3. `commit(verified_possession, handle)` authenticates OAuth, consumes the
   handle before database access, validates exact source/challenge/key match
   and freshness, re-reads current account/tenant/grants,
   compares the decision, and atomically commits source replay plus the
   session through `Store::admit`. A duplicate commit denies; ambiguous
   database outcomes never return a usable receipt.
   Only the returned committed `Session` may supply signed PoP token claims.

The handle must not become a reusable delegated credential for Registry or
Model. Their admission uses separate holder proof and current Policy checks
at every protected request or distinct tool call.

## OAuth-side adapter and activation boundary

`oauth::federate_policy::PolicyFederateAdmission` converts the issuer's
verified source and possession types to the two typed Policy RPCs. It uses
checked time conversions, requires the exact requested-scope echo and a
canonical granted subset, keeps the 32-byte handle server-side, and rejects
a committed receipt with different Ed/PQ keys or an invalid proof epoch.
`OAuthService` does not construct or install it: its Federate issuer slot
remains absent and the browser routes return 503. The adapter is not an
alternative signer or a fallback Policy authority.

Before installation, Policy needs a complete collision inventory with an
authoritative required source-ID list, the exact serving generation and
fixed scope ceilings, a signed PDS account mount with mandatory MAC audit
sink, and separately provisioned scoped account-reader and session-writer
PostgreSQL logins. The current best-effort `foreign_protocol_keys` collector
cannot prove inventory completeness because it skips unreadable sources;
feeding its result to `CollisionInventory::from_sources` would be unsafe.
Staging IaC does not yet project those Policy role URLs or provision the
session generation/profile. The runtime factory must remain default-off until
these are made source-controlled and validated together.
