# Authentication Storage Configuration

Hyprstream uses two persistent stores for authentication state:

- **User store** — user profiles, Ed25519 public keys, active/suspended status,
  external IdP identifiers (SCIM `externalId`), and reverse indexes used for
  pubkey challenge-response auth.
- **Token store** — OAuth 2.0 refresh tokens, bound to a client, subject, scopes,
  and optionally an Ed25519 key (`cnf.jwk` continuity on refresh).

Both stores share a backend selection. Two backends are available:

| Backend | Default | Requires | Best for |
|---------|---------|----------|----------|
| `rocksdb` | yes | nothing | Single-node deployments |
| `valkey` | no | `--features valkey` + running Valkey/Redis | Multi-instance, horizontal scaling |

---

## Configuration

```toml
[credentials]
backend = "rocksdb"    # "rocksdb" (default) or "valkey"

[credentials.valkey]
url = "redis://127.0.0.1:6379"   # default
```

The `[credentials.valkey]` table is ignored when `backend = "rocksdb"`.

---

## RocksDB backend (default)

No additional infrastructure required. Hyprstream opens two embedded RocksDB
databases at startup under the credentials directory:

```
~/.config/hyprstream/credentials/
  users.db/          ← user profiles and pubkeys
  oauth-tokens/      ← refresh tokens
```

The exact path is `config_dir()/credentials/`, where `config_dir()` follows XDG
on Linux (`~/.config/hyprstream/`) with a fallback to `/etc/hyprstream/credentials/`.

### User store key scheme

```
user:{username}       → Cap'n Proto UserInfo (profile + pubkeys)
pubkey:{fingerprint}  → username  (reverse lookup for challenge-response auth)
```

### Token store behaviour

Refresh tokens are stored as JSON. Expiry is enforced lazily: on `GET`, if the
token is past its `expires_at_unix` timestamp it is deleted and `None` is
returned. No background sweep is needed.

### Constraints

- Single process only — RocksDB does not support concurrent writers across
  processes. Running two hyprstream instances against the same RocksDB directory
  will corrupt the database.
- Not replicated — loss of the credentials directory means loss of all user
  accounts and active refresh tokens.

### Security

RocksDB files are created with `0600` permissions. There is no encryption at
rest; rely on filesystem-level access control or full-disk encryption.

---

## Valkey backend

Valkey (and Redis-compatible servers) provide shared, replicated storage suitable
for multi-instance deployments.

### Compile-time requirement

The Valkey backend is behind a feature flag and must be compiled in:

```bash
cargo build --features valkey
```

If `backend = "valkey"` is set in config but the binary was compiled without
`--features valkey`, hyprstream logs an error at startup and falls back to
treating the store as unconfigured (SCIM and token endpoints report 503).

### Connection

```toml
[credentials.valkey]
url = "redis://127.0.0.1:6379"
```

Standard Redis URL format: `redis://[user:password@]host[:port][/db]` or
`rediss://...` for TLS. A connection pool of 8 connections is created at startup
and shared across the user store and token store.

### User store key schema

All keys are prefixed `hs:` to avoid collision with other applications sharing
the same Valkey instance.

| Key | Type | Value |
|-----|------|-------|
| `hs:users` | SET | All registered usernames |
| `hs:user:{username}` | STRING | JSON `UserProfile` |
| `hs:user:{username}:keys` | SET | Fingerprints of registered pubkeys |
| `hs:key:{fingerprint}` | STRING | JSON pubkey data (base64, label, timestamps) |
| `hs:keyowner:{fingerprint}` | STRING | Username (fingerprint → user reverse index) |
| `hs:idx:sub:{sub}` | STRING | Username (UUID → username reverse index) |
| `hs:idx:extid:{extid}` | STRING | Username (externalId → username reverse index) |

### Token store key schema

| Key | Type | Value |
|-----|------|-------|
| `hs:token:{token}` | STRING | JSON `RefreshTokenEntry` |

Refresh tokens are stored with a native Valkey TTL (`SETEX`). Expired tokens
are evicted automatically by Valkey — no lazy-expiry logic is needed on read.

### SCIM filter support

The SCIM list endpoint (`GET /scim/v2/Users?filter=...`) behaviour differs
between backends:

| Filter expression | RocksDB | Valkey |
|-------------------|---------|--------|
| `userName eq "alice"` | in-memory scan | O(1) direct key lookup |
| `id eq "<uuid>"` | in-memory scan | O(1) via `hs:idx:sub:` index |
| `externalId eq "<val>"` | in-memory scan | O(1) via `hs:idx:extid:` index |
| `active eq true/false` | in-memory scan | SMEMBERS + in-memory filter |
| `attr pr` (presence) | in-memory scan | SMEMBERS + in-memory filter |
| Compound expressions (`and`, `or`) | in-memory scan | HTTP 400 `invalidFilter` |

Hyprstream's `ServiceProviderConfig` declares `filter.supported = true` and
`filter.maxResults = 100`. Real IdP SCIM provisioning clients (Okta, Entra,
Google Workspace) only send `userName eq` and `externalId eq` expressions;
compound filters are not sent in practice.

### Multi-instance considerations

- All hyprstream instances must point to the same Valkey server (or cluster).
- Valkey replication provides read scaling; writes go to the primary.
- Refresh token TTLs are set atomically on write — no clock synchronisation
  required between instances beyond normal NTP accuracy.
- The `hs:users` SET is the authoritative list of usernames. It is updated
  atomically with user creation and deletion via `SADD`/`SREM`.

---

## Token store: what happens when unconfigured

If the credentials directory cannot be opened (RocksDB) or the Valkey connection
fails at startup, hyprstream continues running but logs a warning:

```
WARN Could not open refresh token store — tokens will not survive restart
```

In this state:

- Refresh tokens issued during the session are stored only in memory and lost
  on restart.
- Access tokens remain valid until their TTL expires.
- SCIM and user endpoints are unaffected (separate from the token store).

---

## Operations

### Inspecting the Valkey key space

```bash
# Count users
valkey-cli SCARD hs:users

# List all usernames
valkey-cli SMEMBERS hs:users

# Inspect a user profile
valkey-cli GET hs:user:alice

# Count active refresh tokens
valkey-cli KEYS 'hs:token:*' | wc -l

# Flush all refresh tokens (e.g. force re-login after security incident)
valkey-cli --scan --pattern 'hs:token:*' | xargs valkey-cli DEL
```

### Inspecting RocksDB

The `hyprstream user` CLI reads directly from the RocksDB user store:

```bash
hyprstream user list
hyprstream user keys list alice
```

See [`users.md`](users.md) for the full CLI surface and rotation procedure.

### Migrating from RocksDB to Valkey

No automated migration tool exists yet. Manual steps:

1. Export users from RocksDB via `hyprstream user list` and re-register them
   via SCIM `POST /scim/v2/Users` against the Valkey-backed instance.
2. Existing refresh tokens cannot be migrated — users will need to re-authenticate.
3. Update config to `backend = "valkey"`, rebuild with `--features valkey`,
   restart.

Active sessions using access tokens (short-lived JWTs) are unaffected by the
migration since they do not touch the token store until refresh.

---

## Related

- [OAuth provider configuration](deployment/oauth-providers.md) — configuring
  external OIDC/OAuth2 IdPs for login delegation
- [SCIM 2.0 endpoints](deployment/keycloak.md) — trusted issuer federation
  (separate from the `[credentials]` backend)
