# OAuth / OIDC Provider Configuration

Hyprstream can delegate user authentication to external identity providers. Two
provider kinds are supported, each with different discovery and token-validation paths:

| Kind | Discovery | Token validation | Use when |
|------|-----------|-----------------|----------|
| `oidc` | Automatic (OIDC discovery) | `id_token` JWT + JWKS | Google, Dex, Keycloak, Okta, any compliant OIDC IdP |
| `oauth2` | Manual (fixed endpoints) | Userinfo HTTP call | GitHub, Discord, any OAuth 2.0 provider without OIDC |

Providers are configured under `oauth.oidc_providers` in `hyprstream.toml`:

```toml
[oauth.oidc_providers.my-provider]
kind = "oidc"            # or "oauth2"
client_id = "..."
client_secret = "..."
```

---

## Common fields

These fields apply to both provider kinds:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `client_id` | string | **required** | OAuth 2.0 client ID |
| `client_secret` | string | optional | Client secret (omit for public clients) |
| `scopes` | list | *kind-specific* | Requested OAuth scopes |
| `user_mapping` | string | `"namespaced"` | How external `sub` maps to local username |
| `provisioning` | string | `"deny"` | Whether to auto-create users on first login |
| `default_scopes` | list | `[]` | Hyprstream scopes granted to new users |
| `allowed_domains` | list | `[]` | Domain allow-list used by `provisioning = "allowlist"` |
| `clock_skew_seconds` | int | `60` | Allowed clock skew for token expiry checks |
| `allow_http` | bool | `false` | Allow HTTP (non-TLS) discovery endpoints (dev only) |

### `user_mapping`

| Value | Local subject |
|-------|---------------|
| `"namespaced"` | `provider-slug:external-sub` (e.g. `github:12345678`). Default. |
| `"email"` | Email address from external claims (requires `email_verified = true`) |
| `{ claim = { name = "..." } }` | The value of a specific claim (e.g. `preferred_username`) |
| `"didweb"` | **Disabled.** Configuration and login both fail closed because the former path-form account DID was outside the atproto profile (#1159). A node with this setting refuses to start until it is changed; existing path-form profiles remain unusable under a replacement strategy until E4 (#1176) migrates them. Host-form account minting is tracked separately in #1163. |

`"namespaced"` is the default and recommended for most deployments — it is stable,
collision-free, and works for providers (like GitHub) whose `sub` is a numeric ID.

Example of the claim strategy:

```toml
user_mapping = { claim = { name = "preferred_username" } }
```

### `provisioning`

| Value | Behaviour |
|-------|----------|
| `"deny"` | Reject logins from users not already registered locally. **Default.** |
| `"auto"` | Create a local user on first login; grant `default_scopes` |
| `"allowlist"` | Auto-provision only when the mapped subject matches `allowed_domains` — the email domain for email-shaped subjects, or the provider slug for namespaced subjects; anything else is rejected |

### `default_scopes`

Hyprstream-level permission scopes granted to newly provisioned users, in the format
`action:resource_type:resource_id`:

```toml
default_scopes = ["infer:model:*", "read:stream:*"]
```

Standard scopes:

| Scope | Permission |
|-------|-----------|
| `infer:model:*` | Run inference on any model |
| `read:stream:*` | Read any stream |
| `write:stream:*` | Write to any stream |

---

## Kind: `oidc` — OpenID Connect (Google, Dex, Keycloak, Okta)

Additional fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `issuer_url` | string | **yes** | OIDC issuer URL (used for discovery) |

The discovery document is fetched from `{issuer_url}/.well-known/openid-configuration`.
PKCE (S256) and a nonce are sent with every authorization request.

### Google

```toml
[oauth.oidc_providers.google]
kind = "oidc"
issuer_url = "https://accounts.google.com"
client_id = "YOUR_CLIENT_ID.apps.googleusercontent.com"
client_secret = "YOUR_CLIENT_SECRET"
scopes = ["openid", "email", "profile"]
user_mapping = "namespaced"
provisioning = "auto"
default_scopes = ["infer:model:*"]
```

### Dex (self-hosted)

```toml
[oauth.oidc_providers.dex]
kind = "oidc"
issuer_url = "https://dex.example.com"
client_id = "hyprstream"
client_secret = "dex-client-secret"
user_mapping = "namespaced"
provisioning = "auto"
default_scopes = ["infer:model:*"]
```

### Keycloak

```toml
[oauth.oidc_providers.keycloak]
kind = "oidc"
issuer_url = "https://keycloak.example.com/realms/my-realm"
client_id = "hyprstream"
client_secret = "keycloak-client-secret"
user_mapping = "namespaced"
provisioning = "deny"   # users must be pre-registered
```

---

## Kind: `oauth2` — Generic OAuth 2.0 (GitHub, Discord, etc.)

For providers that offer OAuth 2.0 but not OIDC. You must supply all three endpoints
manually, plus a `claim_mapping` if the userinfo response uses non-standard field names.

Additional fields:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `authorization_endpoint` | string | **yes** | Authorization URL |
| `token_endpoint_url` | string | **yes** | Token exchange URL |
| `userinfo_endpoint` | string | **yes** | Userinfo URL |
| `pkce_supported` | bool | `true` | Set `false` if the provider rejects PKCE |
| `claim_mapping` | table | `{}` | Map provider field names to standard claim names |

### `claim_mapping`

Hyprstream normalises the userinfo response to a standard set of claims. If the provider
uses different field names, override them:

| Mapping key | Default field | Standard claim |
|-------------|--------------|----------------|
| `sub` | `"sub"` | Stable user identifier |
| `name` | `"name"` | Display name (nullable) |
| `email` | `"email"` | Email address (nullable) |
| `email_verified` | `"email_verified"` | bool (absent → `false`) |

### GitHub

GitHub does not support OIDC discovery or PKCE. The `id` field in the userinfo response
is a JSON number (e.g. `12345678`); Hyprstream coerces it to a string. With
`user_mapping = "namespaced"` the local username becomes `github:12345678` — stable
across GitHub username changes.

```toml
[oauth.oidc_providers.github]
kind = "oauth2"
client_id = "YOUR_GITHUB_CLIENT_ID"
client_secret = "YOUR_GITHUB_CLIENT_SECRET"
authorization_endpoint = "https://github.com/login/oauth/authorize"
token_endpoint_url = "https://github.com/login/oauth/access_token"
userinfo_endpoint = "https://api.github.com/user"
scopes = ["read:user", "user:email"]
pkce_supported = false
user_mapping = "namespaced"
provisioning = "auto"
default_scopes = ["infer:model:*"]

[oauth.oidc_providers.github.claim_mapping]
sub = "id"
name = "login"
email = "email"
```

**GitHub OAuth app setup:**
1. Go to GitHub → Settings → Developer Settings → OAuth Apps → New OAuth App
2. Set *Authorization callback URL* to `https://YOUR_HYPRSTREAM_HOST/oauth/callback/github`
3. Copy the client ID and generate a client secret

### Discord

```toml
[oauth.oidc_providers.discord]
kind = "oauth2"
client_id = "YOUR_DISCORD_APPLICATION_ID"
client_secret = "YOUR_DISCORD_CLIENT_SECRET"
authorization_endpoint = "https://discord.com/api/oauth2/authorize"
token_endpoint_url = "https://discord.com/api/oauth2/token"
userinfo_endpoint = "https://discord.com/api/users/@me"
scopes = ["identify", "email"]
user_mapping = "namespaced"
provisioning = "auto"
default_scopes = ["infer:model:*"]

[oauth.oidc_providers.discord.claim_mapping]
sub = "id"
name = "username"
email = "email"
email_verified = "verified"
```

Discord `id` is a string snowflake, so no coercion is needed.

---

## Registering the redirect URI

The callback URL Hyprstream registers with the provider is:

```
https://YOUR_HYPRSTREAM_HOST/oauth/callback/<provider-slug>
```

where `<provider-slug>` is the key used in `[oauth.oidc_providers.<slug>]`.

---

## Trusted issuers (federation without login delegation)

For server-to-server flows (e.g. validating JWTs issued by Keycloak at the token
endpoint), use `oauth.trusted_issuers` instead of `oidc_providers`:

```toml
[oauth.trusted_issuers."https://keycloak.example.com/realms/my-realm"]
jwks_uri = "https://keycloak.example.com/realms/my-realm/protocol/openid-connect/certs"
jwks_cache_ttl_secs = 300
```

`trusted_issuers` is a TOML table keyed by the exact issuer URL (matching the token's
`iss` claim). Per-issuer fields: `jwks_uri` (optional — auto-discovered when omitted),
`jwks_cache_ttl_secs` (default 300), and `allow_http` (dev only). This allows Bearer
authentication with externally issued JWTs, plus RFC 8693 token exchange and RFC 7523
JWT bearer grants, without requiring a browser login flow. See the
[Keycloak guide](keycloak.md) for a worked example.
