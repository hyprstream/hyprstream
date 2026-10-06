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

## Next Policy↔OAuth handoff

OAuth and Policy may run in separate processes. Policy's current private
`Decision` cannot be passed to OAuth as a browser-supplied tuple or assumed to
survive in OAuth memory. The next packet must expose authenticated, enrolled
OAuth-only prepare/redeem RPCs:

1. `prepare(verified_source, canonical_requested_scopes)` reads current account,
   signed tenant and grants under the fixed profile. It returns a random
   challenge identifier and an opaque Policy-authenticated prepared-decision
   handle. The handle binds the source token digest, account, tenant, scopes,
   generation, inventory and expiry; its contents are not browser authority.
   A Policy-owned pending row or sealed/MACed handle is acceptable only when
   restart and replay behavior is explicitly tested.
2. OAuth binds that handle to the one-use challenge and verifies **both**
   holder signatures over the exact challenge. The browser cannot supply or
   edit the handle's account, tenant or grant contents.
3. `redeem(verified_possession, handle)` authenticates OAuth, validates the
   handle and challenge freshness, re-reads current account/tenant/grants,
   compares the decision, and atomically commits source replay plus the
   session through `Store::admit`. A duplicate redeem across processes denies.
   Only the returned committed `Session` may supply signed PoP token claims.

The handle must not become a reusable delegated credential for Registry or
Model. Their admission uses separate holder proof and current Policy checks
at every protected request or distinct tool call.
