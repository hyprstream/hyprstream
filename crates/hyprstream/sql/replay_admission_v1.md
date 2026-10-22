# Replay admission migration v1

Apply `replay_admission_v1.sql` through the approved migration channel. It does
not run at service startup. The migration creates only the replay schema,
table, expiry index, and a `NOLOGIN` privilege group. The database operator
grants that group to the dedicated replay login role. Do not grant the replay
login access to credential or PDS tables. The runtime needs only `USAGE` on
`replay_admission` and `SELECT, INSERT, UPDATE, DELETE` on `entries_v1`.
The migration is apply-once and transactional. Before touching the schema or
table, it fails closed if any role with that name already exists: role
attributes alone cannot prove the absence of direct privileges or owned
objects. It never adopts, alters, or grants privileges to a preexisting role.
If the precondition fails, use a separately reviewed database procedure or a
fresh migration target; do not retry by weakening this check.

Build the runtime with `postgres-replay`, then set
`HYPRSTREAM_REPLAY_ADMISSION_DOMAIN=shared-postgres`,
`HYPRSTREAM_REPLAY_SERVICE_DOMAIN` to one stable name shared by all verifiers
for that service, `HYPRSTREAM_REPLAY_POSTGRES_URL_FILE` to a role-scoped URL
file containing `sslmode=verify-full`, and
`HYPRSTREAM_REPLAY_POSTGRES_SSLROOTCERT_FILE` to the pinned CA PEM. Different
service domains need different names. The URL is never accepted directly in
an environment variable. Startup verifies connectivity, schema and grants;
failure leaves replay admission uninstalled. At runtime, pool errors, queue
overload and timeouts deny admission. There is no in-memory fallback.

The primary key is `(service_domain, partition, key_digest)`. Partition 1 is
authenticated proof, 2 is unattributed proof, and 3 is mediated query. The
digest uses a distinct versioned prefix for proof and mediated identifiers.
The `INSERT ... ON CONFLICT DO UPDATE ... WHERE expired` statement serializes
same-key contenders through the unique index and row lock. A live row cannot
be replaced. A row with `expires_at` at or before the database statement time
may be replaced, and a proposed already-expired row is never inserted.
Cleanup deletes at most 512 expired rows per pass, skips locked rows, and
rechecks expiry before deletion. The worker runs cleanup every 64 admissions
and while idle. Those passes retain no unexpired accepted row. Expired rows
may be deleted while a same-key renewal waits for the row lock. In controlled
PostgreSQL 16 scratch interleavings, the waiting `ON CONFLICT` statement
inserted after that delete committed; when renewal locked first, cleanup
skipped its row. Both orders admitted one renewal and denied its duplicate.
No unexpired accepted row is removed early. An expiry that
passes between the client precheck and the database statement is also reported
as `Replayed` rather than `Failed`; both results deny dispatch. Expired rows
may remain longer during a database outage; admissions deny while the database
is unavailable. Unattributed proofs still require a domain-wide challenge
source, which this migration does not provide, so they deny in shared mode.
