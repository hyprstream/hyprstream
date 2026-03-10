# Keycloak Deployment Guide: OIDC Federation with hyprstream

Keycloak is a popular self-hosted OIDC provider. This guide shows how to configure Keycloak as a `trusted_issuers` entry so that Keycloak-managed users can authenticate to hyprstream without needing a local account.

## Prerequisites

- Keycloak 21+ running at `https://keycloak.example.com`
- hyprstream with `trusted_issuers` support (Task 11 auth/federation)

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

Users authenticate via the standard OAuth 2.0 device authorization flow, specifying the Keycloak issuer:

```bash
hyprstream login --issuer https://keycloak.example.com/realms/myrealm
```

This opens the Keycloak login page in a browser. On success, Keycloak issues a JWT. hyprstream receives the token, checks that `iss` matches a trusted issuer, fetches the Keycloak JWKS to verify the signature, and constructs a federated subject `https://keycloak.example.com/realms/myrealm:alice` for Casbin policy enforcement.

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

If Keycloak is configured with audience mappers, the `aud` claim may not match what hyprstream expects. Either:
- Remove the audience mapper in Keycloak, or
- Set `oauth.audience` in `hyprstream.toml` to match the Keycloak audience value

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
