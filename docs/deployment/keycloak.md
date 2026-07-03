# Keycloak Deployment Guide: OIDC Federation with hyprstream

Keycloak is a popular self-hosted OIDC provider. This guide shows how to configure Keycloak as a `trusted_issuers` entry so that Keycloak-managed users can authenticate to hyprstream without needing a local account.

## Prerequisites

- Keycloak 21+ running at `https://keycloak.example.com`
- The realm must sign tokens with **EdDSA (Ed25519)** — hyprstream's federation key resolver only accepts `kty: OKP` / `crv: Ed25519` keys from a JWKS (`crates/hyprstream/src/auth/federation.rs`). RS256-signed tokens (the Keycloak default) will fail verification; configure an EdDSA realm key and set the client's access-token signature algorithm accordingly.

## 1. Keycloak Realm Setup

1. Log in to the Keycloak Admin Console.
2. Create a new realm (e.g., `myrealm`), or use an existing one.
3. Under **Clients**, click **Create client**:
   - **Client type:** OpenID Connect
   - **Client ID:** `hyprstream`
   - **Valid redirect URIs:** `http://localhost:6791/oauth/callback` (dev), your production callback URL
   - **Access type:** Public (PKCE only — no client secret)
4. Under **Client scopes**, ensure `openid` and `profile` are in the default list.

## 2. Find the Keycloak JWKS URI

The Keycloak JWKS URI follows this pattern:

```
https://keycloak.example.com/realms/{realm}/protocol/openid-connect/certs
```

You can verify it by fetching the AS metadata:

```bash
curl -s https://keycloak.example.com/realms/myrealm/.well-known/openid-configuration | jq .jwks_uri
```

## 3. Configure hyprstream

Add the following to `hyprstream.toml`:

```toml
[oauth.trusted_issuers."https://keycloak.example.com/realms/myrealm"]
jwks_uri = "https://keycloak.example.com/realms/myrealm/protocol/openid-connect/certs"
jwks_cache_ttl_secs = 300
```

> **Important:** The key in `trusted_issuers` must match the `iss` claim in Keycloak JWTs exactly, including any trailing slashes.

## 4. Grant Access to Keycloak Users

Keycloak users are identified in hyprstream policy as `{issuer_url}:{username}`. Grant a role to a specific Keycloak user:

```bash
hyprstream quick policy role add \
  "https://keycloak.example.com/realms/myrealm:alice" ttt.user
```

To grant all Keycloak-authenticated users a role, add a wildcard policy rule directly to `policy.csv`:

```csv
p, https://keycloak.example.com/realms/myrealm:*, *, model:*, infer.generate, allow
```

## 5. User Login Flow

There is no `hyprstream login` command, and hyprstream does not proxy the Keycloak login. Users obtain a token **directly from Keycloak** using any grant Keycloak supports (authorization code + PKCE, Keycloak's own device flow, etc.) and present it to hyprstream as a Bearer token:

```bash
# Example: Keycloak device flow from a terminal
curl -s -X POST \
  "https://keycloak.example.com/realms/myrealm/protocol/openid-connect/auth/device" \
  -d "client_id=hyprstream"
# ...visit verification_uri, then poll the token endpoint for an access token

# Use the Keycloak-issued token against hyprstream
curl -H "Authorization: Bearer $KEYCLOAK_ACCESS_TOKEN" \
  https://hyprstream.example.com/v1/models
```

On each request, hyprstream's auth middleware (`crates/hyprstream/src/server/middleware.rs`) extracts the token's `iss` claim, checks it against `trusted_issuers`, fetches (and caches) the Keycloak JWKS to verify the signature, and constructs a federated Casbin subject of the form `{issuer_url}:{sub}` — e.g. `https://keycloak.example.com/realms/myrealm:alice`. Keycloak-issued tokens can also be exchanged at hyprstream's token endpoint via RFC 8693 token exchange or RFC 7523 JWT bearer grants.

> **Note:** hyprstream *does* implement its own RFC 8628 device flow (`POST /oauth/device`), but that flow authenticates against hyprstream's local user store via Ed25519 challenge-response — it is not how you log in with a federated Keycloak identity.

## 6. Troubleshooting

### `iss` mismatch

The `iss` claim in a Keycloak token must match the `trusted_issuers` key **exactly**. Check for trailing slashes:

```bash
# Inspect token iss claim
jwt_token="..." # paste token
echo "$jwt_token" | cut -d. -f2 | base64 -d 2>/dev/null | jq .iss
```

If Keycloak includes a trailing slash (`https://keycloak.example.com/realms/myrealm/`), update the config key to match.

### Token audience (`aud`) mismatch

hyprstream validates federated tokens against its own resource URL (`config.oai.resource_url()` — the `oai.external_url` setting, or derived from the OAI host/port). The validation is lenient for federated tokens: a token with **no** `aud` claim is accepted; a token with a **wrong** `aud` is rejected. If Keycloak is configured with audience mappers, either:
- Remove the audience mapper in Keycloak (absent `aud` passes), or
- Set the mapped audience to hyprstream's resource URL (set `oai.external_url` in `hyprstream.toml` if the default derivation doesn't match)

### Clock skew

JWT `exp` and `iat` validation is sensitive to clock differences. Ensure NTP is synchronized on both the Keycloak server and hyprstream nodes:

```bash
timedatectl status | grep synchronized
```

### JWKS fetch failure

If hyprstream cannot reach the Keycloak JWKS URI, check:
1. Network connectivity from hyprstream to Keycloak
2. TLS certificate validity (hyprstream uses the system trust store)
3. Set `jwks_uri` explicitly in config to bypass auto-discovery

## 7. Security Notes

- Tokens are verified against Keycloak's public key on every request (cached for `jwks_cache_ttl_secs`).
- Compromised tokens expire after the JWT `exp` time; no revocation check is performed.
- Use short token lifetimes (≤15 minutes) in Keycloak for production deployments.
- Keycloak users do not get automatic access — a Casbin policy grant is always required.
