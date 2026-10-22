# RDS-backed PDS record store (operator contract)

Issue #1257. The PDS record store (`PdsRecordStore`) has two backends:

- **RocksDB (local)** — the workstation/dev default. One writer (registry),
  read-only resolvers, single-node durability under
  `<deployment-data>/pds-store`.
- **Postgres (RDS Multi-AZ)** — the deployed posture. Two stateless PDS
  services in two AZs share one RDS instance; a write in AZ-a is immediately
  readable in AZ-b. Selected whenever a records-role binding resolves.

The binary must be built with `--features pds-postgres` for the RDS backend.
A node with a resolved binding on a binary built without the feature **fails
startup** — it never silently falls back to the local store.

## Configuring the binding

Resolution order (first hit wins for each value), implemented by
`RdsConfig::resolved_from_env`:

1. Explicit TOML (`config.toml`):

   ```toml
   [rds]
   url_file = "/run/hyprstream/credentials/records-url"
   root_cert_file = "/run/hyprstream/credentials/rds-ca.pem"
   cell_id = "demo-leaf"          # optional; default "demo-leaf"
   ```

2. Records-role scoped environment variables — each carries a **path**, never
   the secret itself (metal RDS runtime contract v1.1):

   - `HYPRSTREAM_RECORDS_URL_FILE` → path to the URL file
   - `HYPRSTREAM_RECORDS_SSLROOTCERT_FILE` → path to the pinned CA PEM

3. Shared credentials directory: if `HYPRSTREAM_POSTGRES_CREDENTIALS_PATH` is
   set **and** a `records-url` file exists in it, the binding resolves to
   `$HYPRSTREAM_POSTGRES_CREDENTIALS_PATH/records-url` with the CA at
   `rds-ca.pem` alongside it. A credentials directory that holds other roles
   but no `records-url` does **not** activate the records backend.

The URL file holds one newline-terminated libpq URL. The password-bearing URL
never transits the process environment or TOML — only file paths do.

### Rendered-but-broken vs absent

The directory fallback distinguishes **absent** from **broken**:

- `records-url` genuinely absent → local backend (nothing was rendered).
- `records-url` present but a dangling symlink → binding resolves and the
  subsequent read **fails closed**.
- `records-url` present but not stat-able (e.g. unreadable parent) →
  resolution itself **fails closed**.

A broken binding must never silently become "not rendered": a node whose AZ
peer runs RDS must not quietly come up on a local store.

## URL and TLS contract

The URL must:

- use scheme `postgresql` or `postgres`;
- name one non-loopback host — a DNS name or an IP literal (IPv4 or bracketed
  IPv6, e.g. `postgresql://records@…@[2001:db8::10]/records?sslmode=verify-full`);
- contain exactly one query pair `sslmode=verify-full`.

`verify-full` is enforced with no escape hatch: the URL is translated to the
driver's `require` mode and the rustls connector supplies real verification —
the pinned CA from `root_cert_file` is the **only** trust root
(`WebPkiServerVerifier`, standard hostname verification, SNI = the validated
host). Query-level `host`/`hostaddr` overrides are neutralized. There is no
`sslmode` value other than `verify-full` that validates.

## Fail-closed semantics

When a binding resolves, **every** production path uses the shared store and
fails closed:

- Backend open pings RDS at startup; unavailable → the service does not start
  (registry publisher construction is fatal, matching the already-fatal
  resolver side — PDS publish is never silently disabled in RDS posture).
- The QUIC startup gate (`classify_pds_store_for_quic`) and the native
  announcement refresh read accepted states from the shared store, never from
  a local RocksDB.
- The bootstrap CLI (`hyprstream pds init-deployment-store`,
  `hyprstream pds provision-services`, `hyprstream service inspect`) reads and
  writes the shared store. `init-deployment-store` refuses to initialize over
  a store that already holds accepted states and otherwise writes the
  first-boot marker into Postgres; the registry deletes that marker in the
  same transaction as its first accepted-state commit.
- Every backend/connection error propagates as `Err` — the store never
  returns empty-as-absent on a Postgres failure.

With no binding resolved (local/workstation), behavior is unchanged and
publisher-construction failure remains best-effort (warn + PDS publish
disabled).

The at9p duplicity alarm WAL stays **host-local** at
`<deployment-data>/pds-store/at9p-duplicity.wal` in both modes (it is a
per-host tamper journal, not shared state). In RDS posture the local
`pds-store` directory may hold only that WAL — never a record store.

## Consistency guarantees

- Repo head and at9p accepted-state advance are SQL compare-and-swaps
  (`UPDATE … WHERE key AND value` / `INSERT … ON CONFLICT DO NOTHING`,
  rows-affected checked) in one transaction with the rest of the write set —
  two publishers in two AZs cannot lose an update; the loser gets an explicit
  conflict.
- Multi-statement reads (repo record scan + commit, at9p state+checkpoint
  pair) run in one read-only **REPEATABLE READ** transaction on one pooled
  connection — a single MVCC snapshot, so a competing AZ commit cannot tear a
  read between statements.
- Schema migration is idempotent under a transaction-scoped advisory lock.
- Failover bounds: bounded pool (wait/create/recycle timeouts), driver
  `connect_timeout`, TCP keepalive + `TCP_USER_TIMEOUT`, server-side
  `statement_timeout`.

## Testing

Unit/contract tests run unconditionally. Live tests run only when the operator
provides a scratch database URL in a **file** pointed to by
`HYPRSTREAM_POSTGRES_TEST_URL_FILE` (a path, never a direct env var — the same
file-backed credential model as production). Without it they skip green.

Be aware of what that means for CI: the live cross-AZ semantics (two-handle
visibility, stale-head CAS rejection, concurrent single-winner advance,
REPEATABLE READ snapshot consistency, first-boot marker) are executed locally
against a scratch Postgres but are **not executed in CI today** — no CI job
provides a Postgres service, and the verify-full TLS path is exercised by the
adversarial-connector unit test (unpinned issuer rejected, SNI pinned), not
end-to-end against a real RDS endpoint.
