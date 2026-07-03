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

## OAuth clients

A separate concept from users and tokens: an OAuth **client** is the app
asking to authenticate a user. Hyprstream supports two identification
mechanisms for clients, chosen at request time:

### CIMD (Client ID Metadata Document) — preferred for production

The client's `client_id` is itself an **HTTPS URL** pointing to a JSON
metadata document the authorization server fetches at request time.
Implemented in `services/oauth/authorize.rs:160` via
`fetch_client_metadata` (`services/oauth/registration.rs:391`).

```
client_id = "https://app.example.com/.well-known/oauth-client-metadata"
```

At `GET /oauth/authorize` time the AS:

1. Recognises the HTTPS-prefixed `client_id`.
2. Fetches the URL over HTTPS with public CA trust.
3. Validates the document's own `client_id` field matches the URL it
   was fetched from.
4. Checks the request's `redirect_uri` against the document's
   `redirect_uris` array.
5. Caches the resulting `RegisteredClient` in a bounded TTL cache
   (`services/oauth/cimd_cache.rs`); the entry TTL follows the document's
   HTTP `Cache-Control: max-age` / `Expires`, clamped to 5 minutes–24 hours.
   Cache misses (and expiries) re-fetch.

**Trust model:** the AS trusts the document because it was served over a
publicly-trusted TLS channel from the URL the document claims. There is
no AS-side registration step, no shared secret, no DCR endpoint
interaction. Whoever controls the domain controls the client.

**No AS-side storage** of the client record is required (the cache is a
performance hint, not a source of truth). This is the path that scales
to many independent clients without operator coordination, and is the
client-identification model used by atproto OAuth.

Suitable for any client served from a public HTTPS origin with
CA-trusted certificates. Unsuitable when the AS cannot reach the URL
(air-gapped deployments, loopback dev workflows).

### DCR (Dynamic Client Registration, RFC 7591) — dev / fallback

The client `POST`s to `/oauth/register` describing itself; the AS
returns a generated UUID `client_id` the client uses thereafter. The
record lives in `OAuthState.clients`
(`services/oauth/state.rs:304`) — a `RwLock<HashMap>` populated by
`register_client` (`services/oauth/registration.rs:87`).

**State is in-memory only.** Process restart wipes all DCR registrations.
This is correct behaviour for dev workflows where the client (a webapp
running on `localhost:3000`) cannot expose a CIMD URL reachable by the AS,
and is acceptable because the dev session lifetime maps cleanly to the
process lifetime.

### Which path does a given request take?

The branch is decided at `authorize.rs:160` purely by whether the
incoming `client_id` starts with `https://`. There is no per-deployment
toggle; CIMD and DCR coexist and clients pick by how they identify
themselves.

| If `client_id` is… | Path | Storage |
|---|---|---|
| `https://...` | CIMD: fetch, validate, cache | None persisted; bounded in-memory TTL cache |
| anything else (UUID, etc.) | DCR: lookup in `state.clients` | In-memory HashMap |

### Storage backends

Neither path persists OAuth client records to RocksDB or Valkey. CIMD
uses a bounded TTL cache (`services/oauth/cimd_cache.rs`) — entries
expire per the document's HTTP cache headers (clamped 5 min–24 h) and
are re-fetched on miss, so no persistence is needed (the document at the
`client_id` URL is the source of truth). DCR registrations are in-memory
only and survive just the process lifetime.

### Recommendation for new deployments

- **Production webapps:** publish a CIMD metadata document at a stable
  HTTPS URL. Use that URL as `client_id`. No registration call ever.
- **Local dev:** use DCR. The browser at `http://localhost:3000` cannot
  serve CIMD reachable by hyprstream over HTTPS, so DCR is the right
  fallback. The in-memory state is bounded by dev session length.
- **Air-gapped / private CA deployments:** publish the CIMD document on
  the internal HTTPS endpoint, and ensure hyprstream's HTTP client
  trusts the internal CA. (Not currently a configurable knob.)

---

## Related

- [OAuth provider configuration](deployment/oauth-providers.md) — configuring
  external OIDC/OAuth2 IdPs for login delegation
- [SCIM 2.0 endpoints](deployment/keycloak.md) — trusted issuer federation
  (separate from the `[credentials]` backend)
